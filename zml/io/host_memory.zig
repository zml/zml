//! Loader-owned host arenas and per-load block leases.

const std = @import("std");
const Alignment = std.mem.Alignment;
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const Device = @import("../platform.zig").Device;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const Target = @import("../platform.zig").Target;

const log = std.log.scoped(.@"zml/mem");

const NumaPlacement = @import("../mem.zig").NumaPlacement;

/// Host arenas owned by one direct loader. Calibration fills the initial
/// arena set during initialization; loading reuses and grows it. Deinitialize
/// after all transfers finish and before the platform.
pub const Workspace = struct {
    pub const Options = struct {
        /// Safety guard on the arenas' total host memory (pinned on the DMA
        /// targets), not an allocation target.
        max_mapped_bytes: usize = 16 * 1024 * 1024 * 1024,
        numa: NumaPlacement = .memory_nodes,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    backend: Backend,
    /// Owned by `arena_mutex` while `growToBlocks` maps arenas concurrently.
    allocations: std.ArrayListUnmanaged(ArenaAllocation) = .empty,
    arena_mutex: std.Io.Mutex = .init,
    max_mapped_bytes: usize,
    mapped_bytes: std.atomic.Value(usize) = .init(0),

    /// Each allocation strategy keeps only the state it uses. Mutable NUMA
    /// state is protected by `arena_mutex` during concurrent growth.
    const Backend = union(enum) {
        pjrt_host: PjrtHost,
        dma_map: Pages,
        pageable: Pages,
        testing,

        const PjrtHost = struct {
            platform: *const Platform,
            host_nodes: std.ArrayListUnmanaged(HostNode),
        };

        const Pages = struct {
            platform: *const Platform,
            /// Zero leaves placement to the kernel. Automatic placement can
            /// fall back to zero; explicit placement fails if refused.
            numa_mask: u64 = 0,
            numa_explicit: bool = false,
        };

        const Strategy = enum {
            dma_map,
            pjrt_host,
            pageable,
            unsupported,
        };

        fn strategy(target: Target) Strategy {
            return switch (target) {
                .cuda, .oneapi => .dma_map,
                .rocm => .pjrt_host,
                .cpu => .pageable,
                .tpu, .neuron, .metal => .unsupported,
            };
        }

        fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            numa: NumaPlacement,
        ) !Backend {
            const selected_strategy = strategy(platform.target);
            return switch (selected_strategy) {
                .dma_map, .pageable => {
                    const mask: u64, const explicit = switch (numa) {
                        .memory_nodes => .{ memoryNodeMask(allocator, io), false },
                        .nodes => |value| .{ value, true },
                        .none => .{ 0, false },
                    };
                    const pages: Pages = .{
                        .platform = platform,
                        // Interleaving over one node is that node; leave it to the kernel.
                        .numa_mask = if (explicit or @popCount(mask) > 1) mask else 0,
                        .numa_explicit = explicit,
                    };
                    return switch (selected_strategy) {
                        .dma_map => .{ .dma_map = pages },
                        .pageable => .{ .pageable = pages },
                        .pjrt_host, .unsupported => unreachable,
                    };
                },
                .pjrt_host => {
                    var host_nodes: std.ArrayListUnmanaged(HostNode) = .empty;
                    errdefer host_nodes.deinit(allocator);
                    var known = true;
                    devices: for (platform.devices, 0..) |device, device_index| {
                        if (device.memory(.host_pinned) == null) return error.PinnedHostMemoryUnavailable;
                        const node = device.numaNode();
                        known = known and node != null;
                        if (node != null) for (host_nodes.items) |existing| {
                            if (existing.node == node) continue :devices;
                        };
                        try host_nodes.append(allocator, .{ .device_index = device_index, .node = node });
                    }
                    if (!known) {
                        host_nodes.clearRetainingCapacity();
                        for (platform.devices, 0..) |_, device_index| {
                            try host_nodes.append(allocator, .{ .device_index = device_index, .node = null });
                        }
                    }
                    return .{ .pjrt_host = .{
                        .platform = platform,
                        .host_nodes = host_nodes,
                    } };
                },
                .unsupported => error.DmaBenchmarkUnsupported,
            };
        }

        /// Allocator adapter private to page-backed workspace arenas. It
        /// applies huge-page alignment/advice and optionally registers the
        /// pages with the selected backend's PJRT client.
        const MapAllocator = struct {
            const transparent_huge_page_size = 2 * 1024 * 1024;

            parent: std.mem.Allocator,
            /// Null leaves the pages pageable for CPU transfers.
            platform: ?*const Platform,

            fn init(parent: std.mem.Allocator, platform: *const Platform) MapAllocator {
                return .{ .parent = parent, .platform = platform };
            }

            fn initPageable(parent: std.mem.Allocator) MapAllocator {
                return .{ .parent = parent, .platform = null };
            }

            fn allocator(self: *const MapAllocator) std.mem.Allocator {
                return .{
                    .ptr = @constCast(self),
                    .vtable = &.{
                        .alloc = alloc,
                        .resize = resize,
                        .remap = remap,
                        .free = free,
                    },
                };
            }

            fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
                const self: *const MapAllocator = @ptrCast(@alignCast(ctx));
                const effective_alignment = effectiveAlignment(alignment, len);
                const allocation = self.parent.rawAlloc(len, effective_alignment, ret_addr);
                if (allocation) |loc| {
                    const data = loc[0..len];
                    adviseHugePages(data);
                    if (self.platform) |platform| {
                        platform.pjrt_client.dmaMap(platform.pjrt_api, @ptrCast(data)) catch {
                            self.parent.rawFree(data, effective_alignment, ret_addr);
                            return null;
                        };
                    }
                }
                return allocation;
            }

            fn resize(_: *anyopaque, _: []u8, _: Alignment, _: usize, _: usize) bool {
                return false;
            }

            fn remap(_: *anyopaque, _: []u8, _: Alignment, _: usize, _: usize) ?[*]u8 {
                return null;
            }

            fn free(ctx: *anyopaque, buf: []u8, alignment: Alignment, ret_addr: usize) void {
                const self: *const MapAllocator = @ptrCast(@alignCast(ctx));
                if (self.platform) |platform| {
                    platform.pjrt_client.dmaUnmap(platform.pjrt_api, @ptrCast(buf)) catch unreachable;
                }
                self.parent.rawFree(buf, effectiveAlignment(alignment, buf.len), ret_addr);
            }

            fn effectiveAlignment(alignment: Alignment, len: usize) Alignment {
                if (comptime builtin.os.tag != .linux) return alignment;
                if (len < transparent_huge_page_size) return alignment;
                return alignment.max(.fromByteUnits(transparent_huge_page_size));
            }

            fn adviseHugePages(data: []u8) void {
                if (comptime builtin.os.tag != .linux) return;
                if (data.len < transparent_huge_page_size) return;

                const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(data.ptr);
                std.posix.madvise(ptr, data.len, std.posix.MADV.HUGEPAGE) catch |err| {
                    log.warn("MADV_HUGEPAGE failed for DMA buffer at 0x{x} ({Bi:.2}): {s}", .{
                        @intFromPtr(data.ptr),
                        data.len,
                        @errorName(err),
                    });
                };
            }
        };
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Workspace.Options,
    ) !Workspace {
        if (platform.devices.len == 0 or platform.devices.len > 64)
            return error.DmaDeviceMismatch;
        const device_kind = platform.devices[0].kind();
        for (platform.devices[1..]) |device| {
            if (!std.mem.eql(u8, device_kind, device.kind()))
                return error.HeterogeneousDmaUnsupported;
        }

        if (opts.max_mapped_bytes < minimum_mapped_bytes)
            return error.InvalidDmaLoadConfig;
        switch (opts.numa) {
            .nodes => if (comptime builtin.os.tag != .linux)
                return error.DmaBenchmarkNumaUnsupported,
            .memory_nodes, .none => {},
        }

        return .{
            .allocator = allocator,
            .io = io,
            .backend = try .init(allocator, io, platform, opts.numa),
            .max_mapped_bytes = opts.max_mapped_bytes,
        };
    }

    pub fn deinit(self: *Workspace) void {
        const io = self.io;
        const mapped_bytes = self.mapped_bytes.load(.acquire);
        const started: std.Io.Timestamp = .now(io, .awake);
        for (self.allocations.items) |allocation| self.unmapArena(allocation);
        self.allocations.deinit(self.allocator);
        switch (self.backend) {
            .pjrt_host => |*host| host.host_nodes.deinit(self.allocator),
            .dma_map, .pageable, .testing => {},
        }
        const elapsed_ns = elapsedNanoseconds(started, .now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }

    /// Borrows the newest retained arena that fits, when no arena growth is running.
    pub fn findArena(self: *const Workspace, minimum_len: usize) ?[]u8 {
        var index = self.allocations.items.len;
        while (index != 0) {
            index -= 1;
            const arena = self.allocations.items[index].data();
            if (arena.len >= minimum_len) return arena;
        }
        return null;
    }

    /// Retains a new arena within the mapped-byte ceiling. The workspace
    /// has one owner; callers serialize growth except for the parts
    /// budgeted together by `growToBlocks`.
    pub fn allocate(self: *Workspace, bytes: usize) ![]u8 {
        const mapped_bytes = self.mapped_bytes.load(.acquire);
        if (bytes > self.max_mapped_bytes - mapped_bytes)
            return error.DmaMappedBudgetExceeded;
        const started: std.Io.Timestamp = .now(self.io, .awake);
        const allocation = try self.mapArena(bytes);
        errdefer self.unmapArena(allocation);
        const mapped_at: std.Io.Timestamp = .now(self.io, .awake);
        const replacement = allocation.data();
        const placement_mask = mask: {
            self.arena_mutex.lockUncancelable(self.io);
            defer self.arena_mutex.unlock(self.io);
            try self.allocations.append(self.allocator, allocation);
            _ = self.mapped_bytes.fetchAdd(replacement.len, .release);
            break :mask switch (self.backend) {
                .dma_map, .pageable => |pages| pages.numa_mask,
                .pjrt_host, .testing => 0,
            };
        };
        const finished_at: std.Io.Timestamp = .now(self.io, .awake);
        const elapsed_ms = @as(f64, @floatFromInt(elapsedNanoseconds(started, finished_at))) / std.time.ns_per_ms;
        switch (allocation) {
            .pjrt_host => |pinned| log.info("DMA arena kind=pjrt_host device={d} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                pinned.device_index,
                @intFromPtr(replacement.ptr),
                replacement.len,
                elapsed_ms,
            }),
            .dma_map, .pageable => log.info("DMA arena kind={s} placement={s} nodes=0x{x} address=0x{x} size={Bi:.2} allocation_ms={d:.3} map_ms={d:.3}", .{
                @tagName(allocation),
                placementName(placement_mask),
                placement_mask,
                @intFromPtr(replacement.ptr),
                replacement.len,
                elapsed_ms,
                @as(f64, @floatFromInt(elapsedNanoseconds(started, mapped_at))) / std.time.ns_per_ms,
            }),
        }
        return replacement;
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    pub fn usableBlocks(self: *const Workspace, block_size: usize) usize {
        var usable: usize = 0;
        for (self.allocations.items) |arena| usable += arena.data().len / block_size;
        return usable;
    }

    /// Maps the blocks missing below `target_blocks` as up to
    /// `growth_parallelism` arenas registered concurrently. The aggregate
    /// check keeps a partial growth from crossing the ceiling. Requires a
    /// workspace with no other growth in progress and a nonzero block_size.
    pub fn growToBlocks(self: *Workspace, block_size: usize, target_blocks: usize) !void {
        const usable_blocks = self.usableBlocks(block_size);
        const missing_blocks = target_blocks -| usable_blocks;
        if (missing_blocks == 0) return;
        const mapped_bytes = self.mapped_bytes.load(.acquire);
        if (missing_blocks > (self.max_mapped_bytes - mapped_bytes) / block_size)
            return error.DmaMappedBudgetExceeded;

        const Worker = struct {
            workspace: *Workspace,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                _ = worker.workspace.allocate(worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                };
            }
        };
        const parts = @min(growth_parallelism, missing_blocks);
        const blocks_per_part = missing_blocks / parts + @intFromBool(missing_blocks % parts != 0);
        var first_error: std.atomic.Value(u16) = .init(0);
        var group: std.Io.Group = .init;
        var group_error: ?anyerror = null;
        var remaining_blocks = missing_blocks;
        while (remaining_blocks != 0) {
            const part_blocks = @min(blocks_per_part, remaining_blocks);
            remaining_blocks -= part_blocks;
            group.concurrent(self.io, Worker.run, .{Worker{
                .workspace = self,
                .bytes = part_blocks * block_size,
                .first_error = &first_error,
            }}) catch |err| {
                group_error = err;
                break;
            };
        }
        group.await(self.io) catch |err| if (group_error == null) {
            group_error = err;
        };
        if (group_error) |err| return err;
        const error_code = first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
    }

    /// Creates ordinary allocator-backed arenas for tests without a PJRT platform.
    pub fn initForTesting(
        allocator: std.mem.Allocator,
        io: std.Io,
        max_mapped_bytes: usize,
    ) !Workspace {
        if (!builtin.is_test) @compileError("initForTesting is only available in tests");
        return .{
            .allocator = allocator,
            .io = io,
            .backend = .testing,
            .max_mapped_bytes = max_mapped_bytes,
        };
    }

    // The largest supported calibration block must fit.
    const minimum_mapped_bytes = 32 * 1024 * 1024;

    /// Concurrent arena registrations used by `growToBlocks`.
    const growth_parallelism = 4;

    /// One NUMA node the ROCm devices report, with a device to allocate
    /// through and the bytes allocated so far; unknown nodes degrade to one
    /// entry per device.
    const HostNode = struct {
        device_index: usize,
        node: ?usize,
        bytes: usize = 0,
    };

    fn mapArena(self: *Workspace, bytes: usize) !ArenaAllocation {
        switch (self.backend) {
            .pjrt_host => |*host| {
                const device_index = index: {
                    self.arena_mutex.lockUncancelable(self.io);
                    defer self.arena_mutex.unlock(self.io);
                    var emptiest = &host.host_nodes.items[0];
                    for (host.host_nodes.items[1..]) |*host_node| {
                        if (host_node.bytes < emptiest.bytes) emptiest = host_node;
                    }
                    emptiest.bytes += bytes;
                    break :index emptiest.device_index;
                };
                const memory = host.platform.devices[device_index].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable;
                return .{ .pjrt_host = try .init(memory, device_index, bytes) };
            },
            .dma_map, .pageable, .testing => {
                const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
                // The placement allocator sits between the page allocation
                // and the registration; both are built per call because the
                // workspace moves. A pageable arena is placed and huge-page
                // advised like a mapped one, never registered.
                var numa: NumaAllocator = .{
                    .parent = self.allocator,
                    .mask = self.placementMask(),
                    .explicit = switch (self.backend) {
                        .dma_map, .pageable => |pages| pages.numa_explicit,
                        .testing => false,
                        .pjrt_host => unreachable,
                    },
                };
                const pages: Backend.MapAllocator = switch (self.backend) {
                    .dma_map => |mapped| .init(numa.allocator(), mapped.platform),
                    .pageable, .testing => .initPageable(numa.allocator()),
                    .pjrt_host => unreachable,
                };
                const arena = try pages.allocator().alignedAlloc(u8, alignment, bytes);
                if (numa.mask == 0) self.leaveUnplaced();
                return if (self.backend == .dma_map) .{ .dma_map = arena } else .{ .pageable = arena };
            },
        }
    }

    /// Frees through the allocator that made the arena; the placement
    /// allocator adds nothing to a free.
    fn unmapArena(self: *Workspace, allocation: ArenaAllocation) void {
        switch (allocation) {
            .pjrt_host => |pinned| pinned.deinit(),
            .dma_map => |arena| {
                const dma_map: Backend.MapAllocator = .init(self.allocator, self.backend.dma_map.platform);
                dma_map.allocator().free(arena);
            },
            .pageable => |arena| {
                const pageable: Backend.MapAllocator = .initPageable(self.allocator);
                pageable.allocator().free(arena);
            },
        }
    }

    fn placementName(mask: u64) []const u8 {
        return switch (@popCount(mask)) {
            0 => "unplaced",
            1 => "bind",
            else => "interleave",
        };
    }

    fn placementMask(self: *Workspace) u64 {
        self.arena_mutex.lockUncancelable(self.io);
        defer self.arena_mutex.unlock(self.io);
        return switch (self.backend) {
            .dma_map, .pageable => |pages| pages.numa_mask,
            .pjrt_host, .testing => 0,
        };
    }

    /// The kernel refused the automatic placement: later arenas stay
    /// unplaced. The only transition the mask ever makes, so concurrent
    /// growth workers cannot undo it.
    fn leaveUnplaced(self: *Workspace) void {
        self.arena_mutex.lockUncancelable(self.io);
        defer self.arena_mutex.unlock(self.io);
        switch (self.backend) {
            .dma_map, .pageable => |*pages| pages.numa_mask = 0,
            .pjrt_host, .testing => {},
        }
    }
};

/// A per-load view of fixed-size blocks carved from the workspace's arenas.
/// The workspace retains arena ownership; this view owns only free-list
/// metadata.
pub const BlockPool = struct {
    pub const Error = anyerror;

    pub const Block = []u8;

    pub const Lease = struct {
        pool: *BlockPool,
        io: std.Io,
        data: []u8,
        remaining: std.atomic.Value(usize),

        pub fn init(pool: *BlockPool, io: std.Io, block: Block, references: usize) Lease {
            std.debug.assert(references > 0);
            return .{
                .pool = pool,
                .io = io,
                .data = block,
                .remaining = .init(references),
            };
        }

        /// Completes one reference and returns whether this was the final one.
        pub fn complete(self: *Lease) bool {
            const previous = self.remaining.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous == 1) self.pool.release(self.io, self.data);
            return previous == 1;
        }
    };

    allocator: std.mem.Allocator,
    workspace: *Workspace,
    free_blocks: std.ArrayListUnmanaged(Block) = .empty,
    block_size: usize,
    max_mapped_bytes: usize,
    mapped_bytes: usize,
    /// Blocks kept mapped as the growth floor: the DMA stage of every device.
    reserve: usize,
    capacity: usize = 0,
    newly_mapped_bytes: usize = 0,
    unused_tail_bytes: usize = 0,
    slab_blocks: usize,
    in_use: usize = 0,
    high_water: usize = 0,
    closed: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    /// Builds a fresh free-list view from every retained arena. Arena tails
    /// smaller than one selected block remain mapped and are reported unused.
    /// The workspace must outlive this view.
    pub fn init(
        allocator: std.mem.Allocator,
        workspace: *Workspace,
        block_size: usize,
        max_mapped_bytes: usize,
        reserve: usize,
    ) !BlockPool {
        const mapped_bytes = workspace.mapped_bytes.load(.acquire);
        if (block_size == 0 or mapped_bytes > max_mapped_bytes)
            return error.RequestExceedsCapacity;
        var self: BlockPool = .{
            .allocator = allocator,
            .workspace = workspace,
            .block_size = block_size,
            .max_mapped_bytes = max_mapped_bytes,
            .mapped_bytes = mapped_bytes,
            .reserve = reserve,
            .slab_blocks = @max(@as(usize, 1), default_slab_size / block_size),
        };
        errdefer self.deinit();
        var enumerated_bytes: usize = 0;
        for (workspace.allocations.items) |*allocation| {
            const arena = allocation.data();
            enumerated_bytes += arena.len;
            try self.attachArena(arena);
        }
        if (enumerated_bytes != mapped_bytes) return error.InvalidDmaWorkspace;
        if (self.reservedGrowthBlocks() > self.remainingBlockBudget())
            return error.RequestExceedsCapacity;
        return self;
    }

    pub fn deinit(self: *BlockPool) void {
        std.debug.assert(self.in_use == 0 and self.free_blocks.items.len == self.capacity);
        self.free_blocks.deinit(self.allocator);
        self.* = undefined;
    }

    /// Leases `output.len` blocks atomically, mapping a slab when the free
    /// list is short and the budget allows, otherwise waiting for releases.
    /// Allocates nothing once its arenas are attached.
    pub fn acquireMany(self: *BlockPool, io: std.Io, output: []Block) Error!void {
        if (output.len == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.closed) return error.Closed;
        if (!self.canEverAcquire(output.len)) return error.RequestExceedsCapacity;
        while (self.free_blocks.items.len < output.len) {
            if (self.closed) return error.Closed;
            if (try self.grow()) continue;
            if (self.in_use == 0) return error.RequestExceedsCapacity;
            self.condition.waitUncancelable(io, &self.mutex);
        }
        for (output) |*block| block.* = self.free_blocks.pop().?;
        self.in_use += output.len;
        self.high_water = @max(self.high_water, self.in_use);
    }

    pub fn releaseMany(self: *BlockPool, io: std.Io, blocks: []const Block) void {
        if (blocks.len == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(blocks.len <= self.in_use);
        for (blocks) |block| {
            std.debug.assert(block.len == self.block_size);
            self.free_blocks.appendAssumeCapacity(block);
        }
        self.in_use -= blocks.len;
        self.condition.broadcast(io);
    }

    pub fn release(self: *BlockPool, io: std.Io, block: Block) void {
        self.releaseMany(io, &.{block});
    }

    pub fn close(self: *BlockPool, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.condition.broadcast(io);
    }

    /// Requests of `blocks_per_request` blocks the pool holds without
    /// mapping anything.
    pub fn retainedRequestWidth(self: *const BlockPool, blocks_per_request: usize) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return self.capacity / blocks_per_request;
    }

    /// Requests of `blocks_per_request` blocks that can be leased without
    /// mapping a slab and without eating into the DMA stage reserve.
    pub fn growthFreeRequestWidth(self: *const BlockPool, blocks_per_request: usize) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return (self.capacity -| self.reserve) / blocks_per_request;
    }

    /// The largest request width the pool could support if each request
    /// consumes `blocks_per_request` blocks. Arena tails count against the
    /// mapped-byte cap but do not contribute usable blocks.
    pub fn potentialRequestWidth(self: *const BlockPool, blocks_per_request: usize) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return (self.capacity + self.remainingBlockBudget()) / blocks_per_request;
    }

    const default_slab_size = 64 * 1024 * 1024;

    fn canEverAcquire(self: *const BlockPool, blocks: usize) bool {
        const remaining_blocks = self.remainingBlockBudget();
        const reserved_growth = self.reservedGrowthBlocks();
        if (reserved_growth > remaining_blocks) return false;
        return blocks <= @max(self.capacity, self.reserve) + (remaining_blocks - reserved_growth);
    }

    fn remainingBlockBudget(self: *const BlockPool) usize {
        return (self.max_mapped_bytes -| self.mapped_bytes) / self.block_size;
    }

    fn reservedGrowthBlocks(self: *const BlockPool) usize {
        return self.reserve -| self.capacity;
    }

    fn grow(self: *BlockPool) !bool {
        const block_count = @min(self.slab_blocks, self.remainingBlockBudget());
        if (block_count == 0) return false;
        try self.allocateSlab(block_count);
        return true;
    }

    fn allocateSlab(self: *BlockPool, block_count: usize) !void {
        const slab_len = block_count * self.block_size;
        const mapped_before = self.workspace.mapped_bytes.load(.acquire);
        if (mapped_before != self.mapped_bytes) return error.InvalidDmaWorkspace;
        const slab = try self.workspace.allocate(slab_len);
        const mapped_after = self.workspace.mapped_bytes.load(.acquire);
        if (slab.len != slab_len or mapped_after < mapped_before or
            mapped_after - mapped_before != slab.len or mapped_after > self.max_mapped_bytes)
            return error.InvalidDmaWorkspace;
        self.mapped_bytes = mapped_after;
        try self.attachArena(slab);
        self.newly_mapped_bytes += slab.len;
    }

    fn attachArena(self: *BlockPool, arena: []u8) !void {
        if (arena.len == 0) return error.InvalidDmaWorkspace;
        const block_count = arena.len / self.block_size;
        // Leased blocks are absent from `free_blocks`, so reserving relative to
        // its current length can leave too little space to return them after a
        // slab is attached under load. Keep storage sized for total capacity.
        try self.free_blocks.ensureTotalCapacity(self.allocator, self.capacity + block_count);
        for (0..block_count) |index| {
            self.free_blocks.appendAssumeCapacity(arena[index * self.block_size ..][0..self.block_size]);
        }
        self.capacity += block_count;
        self.unused_tail_bytes += arena.len % self.block_size;
    }
};

const ArenaAllocation = union(enum) {
    /// Our pages, registered with the plugin through `dmaMap`.
    dma_map: []align(std.heap.page_size_min) u8,
    /// Our pages, never registered: the CPU plugin's arenas, and every
    /// arena in tests without a platform.
    pageable: []align(std.heap.page_size_min) u8,
    /// The plugin's pinned host memory, borrowed through a PJRT buffer.
    pjrt_host: PinnedHostAllocation,

    fn data(self: *const ArenaAllocation) []u8 {
        return switch (self.*) {
            .dma_map, .pageable => |bytes| bytes,
            .pjrt_host => |allocation| allocation.data,
        };
    }
};

const PinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,
    device_index: usize,

    fn init(memory: *const Memory, device_index: usize, size: usize) !PinnedHostAllocation {
        const api = memory.platform.pjrt_api;
        const buffer = try memory.platform.pjrt_client.createUninitializedBuffer(api, .{
            .dims = &.{@intCast(size)},
            .element_type = .u8,
            .layout = .{
                .tiled = .{
                    .minor_to_major = &.{0},
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = .{ .memory = memory.pjrt_memory },
        });
        errdefer buffer.deinit(api);
        if (!buffer.isOnCpu(api)) return error.PinnedHostMemoryNotHostVisible;

        // The writable pointer is borrowed from PJRT. Keep both the external
        // reference and its owning buffer alive for the arena's whole lifetime.
        try buffer.increaseExternalReferenceCount(api);
        errdefer buffer.decreaseExternalReferenceCount(api) catch {};
        const ptr: [*]u8 = @ptrCast(try buffer.opaqueDeviceMemoryDataPointer(api));
        return .{
            .buffer = buffer,
            .api = api,
            .data = ptr[0..size],
            .device_index = device_index,
        };
    }

    fn deinit(self: PinnedHostAllocation) void {
        self.buffer.decreaseExternalReferenceCount(self.api) catch unreachable;
        self.buffer.deinit(self.api);
    }
};

/// Applies a NUMA policy to each allocation of its parent, before the
/// caller maps it for DMA.
const NumaAllocator = struct {
    const mpol_bind = 2;
    const mpol_interleave = 3;

    parent: std.mem.Allocator,
    /// One bit binds, several interleave, zero applies nothing. Cleared
    /// when the kernel refuses an automatic placement.
    mask: u64,
    explicit: bool,

    fn allocator(self: *NumaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        const allocation = self.parent.rawAlloc(len, alignment, ret_addr) orelse return null;
        if (self.mask == 0) return allocation;
        if (comptime builtin.os.tag != .linux) {
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }

        const node_mask: [1]u64 = .{self.mask};
        const highest_node: usize = 63 - @clz(self.mask);
        const rc = std.os.linux.syscall6(
            .mbind,
            @intFromPtr(allocation),
            len,
            if (@popCount(self.mask) == 1) mpol_bind else mpol_interleave,
            @intFromPtr(&node_mask),
            // Linux get_nodes() decrements maxnode before copying the mask;
            // raw callers include the same extra sentinel bit as libnuma.
            highest_node + 2,
            0,
        );
        if (std.os.linux.errno(rc) == .SUCCESS) return allocation;
        if (self.explicit) {
            log.err("unable to place DMA arena ({Bi:.2}) on NUMA nodes 0x{x}: {s}", .{
                len,
                self.mask,
                @tagName(std.os.linux.errno(rc)),
            });
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }
        log.warn("NUMA placement of DMA arenas over nodes 0x{x} refused ({s}); leaving them unplaced", .{
            self.mask,
            @tagName(std.os.linux.errno(rc)),
        });
        self.mask = 0;
        return allocation;
    }

    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *NumaAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, alignment, ret_addr);
    }
};

/// Bits of `/sys/devices/system/node/has_memory`, or zero when unreadable.
/// Nodes 64 and above cannot be represented and are dropped.
fn memoryNodeMask(allocator: std.mem.Allocator, io: std.Io) u64 {
    if (comptime builtin.os.tag != .linux) return 0;
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        "/sys/devices/system/node/has_memory",
        allocator,
        .limited(4096),
    ) catch return 0;
    defer allocator.free(contents);
    return parseNodeList(contents);
}

/// Parses a kernel node list ("0-1", "0,2-3") into a mask; zero on any error.
fn parseNodeList(text: []const u8) u64 {
    var mask: u64 = 0;
    var ranges = std.mem.tokenizeAny(u8, text, ", \n");
    while (ranges.next()) |range| {
        var ends = std.mem.splitScalar(u8, range, '-');
        const first = std.fmt.parseInt(usize, ends.first(), 10) catch return 0;
        const last = if (ends.next()) |end|
            std.fmt.parseInt(usize, end, 10) catch return 0
        else
            first;
        if (last < first or ends.next() != null) return 0;
        var node = first;
        while (node <= last) : (node += 1) {
            if (node < 64) mask |= @as(u64, 1) << @intCast(node);
        }
    }
    return mask;
}

fn elapsedNanoseconds(started: std.Io.Timestamp, finished: std.Io.Timestamp) u64 {
    return @intCast(@max(started.durationTo(finished).nanoseconds, 0));
}

test "Workspace backend strategy is exhaustive" {
    try std.testing.expectEqual(Workspace.Backend.Strategy.dma_map, Workspace.Backend.strategy(.cuda));
    try std.testing.expectEqual(Workspace.Backend.Strategy.dma_map, Workspace.Backend.strategy(.oneapi));
    try std.testing.expectEqual(Workspace.Backend.Strategy.pjrt_host, Workspace.Backend.strategy(.rocm));
    try std.testing.expectEqual(Workspace.Backend.Strategy.pageable, Workspace.Backend.strategy(.cpu));
    try std.testing.expectEqual(Workspace.Backend.Strategy.unsupported, Workspace.Backend.strategy(.tpu));
    try std.testing.expectEqual(Workspace.Backend.Strategy.unsupported, Workspace.Backend.strategy(.neuron));
    try std.testing.expectEqual(Workspace.Backend.Strategy.unsupported, Workspace.Backend.strategy(.metal));
}

test "parseNodeList accepts kernel node lists" {
    try std.testing.expectEqual(@as(u64, 0b11), parseNodeList("0-1\n"));
    try std.testing.expectEqual(@as(u64, 0b1101), parseNodeList("0,2-3"));
    try std.testing.expectEqual(@as(u64, 0b1), parseNodeList("0"));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList(""));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList("1-0"));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList("x"));
    try std.testing.expectEqual(@as(u64, 0b1), parseNodeList("0,64"));
}

test "Workspace finds retained arenas behind newer smaller allocations" {
    var workspace = try Workspace.initForTesting(std.testing.allocator, std.testing.io, 256);
    defer workspace.deinit();

    try std.testing.expect(workspace.findArena(1) == null);
    const large = try workspace.allocate(128);
    const small = try workspace.allocate(64);
    try std.testing.expectEqual(@intFromPtr(large.ptr), @intFromPtr(workspace.findArena(100).?.ptr));
    try std.testing.expectEqual(@intFromPtr(small.ptr), @intFromPtr(workspace.findArena(32).?.ptr));
    try std.testing.expect(workspace.findArena(129) == null);
    try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes.load(.acquire));
}

test "Workspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var workspace = try Workspace.initForTesting(allocator, std.testing.io, 256);
            defer workspace.deinit();

            _ = try workspace.allocate(64);
            _ = try workspace.allocate(128);
            try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes.load(.acquire));
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.allocate(128));
            try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes.load(.acquire));
            try std.testing.expectEqual(@as(usize, 128), workspace.findArena(100).?.len);
            try std.testing.expectEqual(@as(usize, 3), workspace.usableBlocks(64));
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes.load(.acquire));
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes.load(.acquire));
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.growToBlocks(64, 5));
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes.load(.acquire));
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "Workspace growth maps missing blocks as concurrent arenas" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, 64 * 64);
    defer workspace.deinit();

    _ = try workspace.allocate(3 * 64);
    try workspace.growToBlocks(64, 13);
    try std.testing.expectEqual(@as(usize, 13), workspace.usableBlocks(64));
    // Ten missing blocks over four parts: 3, 3, 3, 1.
    try std.testing.expectEqual(@as(usize, 5), workspace.allocations.items.len);
    var total: usize = 0;
    for (workspace.allocations.items[1..]) |arena| total += arena.data().len;
    try std.testing.expectEqual(@as(usize, 10 * 64), total);
    try workspace.growToBlocks(64, 14);
    try std.testing.expectEqual(@as(usize, 6), workspace.allocations.items.len);
}

test "BlockPool acquires request blocks atomically" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    var pool = try BlockPool.init(allocator, &workspace, 64, 4 * 64, 0);
    defer pool.deinit();

    var first: [3]BlockPool.Block = undefined;
    try pool.acquireMany(io, &first);
    try std.testing.expectEqual(@as(usize, 3 * 64), pool.high_water * pool.block_size);
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.mapped_bytes);
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.newly_mapped_bytes);
    var oversized: [5]BlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &oversized));

    var started: std.Io.Event = .unset;
    var acquired: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *BlockPool, io_: std.Io, started_: *std.Io.Event, acquired_: *std.Io.Event) void {
            var blocks: [2]BlockPool.Block = undefined;
            started_.set(io_);
            pool_.acquireMany(io_, &blocks) catch unreachable;
            acquired_.set(io_);
            pool_.releaseMany(io_, &blocks);
        }
    }.run, .{ &pool, io, &started, &acquired });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!acquired.isSet());

    pool.releaseMany(io, &first);
    try group.await(io);
    try std.testing.expect(acquired.isSet());
}

test "BlockPool retains free-list capacity when growing with blocks leased" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    var pool = try BlockPool.init(allocator, &workspace, 64, 9 * 64, 0);
    defer pool.deinit();
    pool.slab_blocks = 1;

    var held: [9]BlockPool.Block = undefined;
    try pool.acquireMany(io, &held);
    try std.testing.expect(pool.free_blocks.capacity >= pool.capacity);
    pool.releaseMany(io, &held);
}

test "BlockPool acquisition allocates nothing once its arenas are attached" {
    const io = std.testing.io;
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{});
    var workspace = try Workspace.initForTesting(std.testing.allocator, io, std.math.maxInt(usize));
    defer workspace.deinit();

    var pool = try BlockPool.init(failing.allocator(), &workspace, 64, 4 * 64, 0);
    defer pool.deinit();

    var blocks: [3]BlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    pool.releaseMany(io, &blocks);

    failing.fail_index = failing.alloc_index;
    try pool.acquireMany(io, &blocks);
    try std.testing.expect(!failing.has_induced_failure);
    pool.releaseMany(io, &blocks);
}

test "BlockPool close wakes blocked bulk acquisitions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    var pool = try BlockPool.init(allocator, &workspace, 64, 2 * 64, 0);
    defer pool.deinit();

    var held: [2]BlockPool.Block = undefined;
    try pool.acquireMany(io, &held);
    var started: std.Io.Event = .unset;
    var result: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *BlockPool, io_: std.Io, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            var block: [1]BlockPool.Block = undefined;
            started_.set(io_);
            pool_.acquireMany(io_, &block) catch |err| {
                result_.store(@intFromError(err), .release);
                return;
            };
            pool_.releaseMany(io_, &block);
        }
    }.run, .{ &pool, io, &started, &result });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    pool.close(io);
    try group.await(io);
    pool.releaseMany(io, &held);
    try std.testing.expectEqual(@intFromError(error.Closed), result.load(.acquire));
}

test "BlockPool lease returns a block after out-of-order callbacks" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    var pool = try BlockPool.init(allocator, &workspace, 64, 64, 0);
    defer pool.deinit();

    var blocks: [1]BlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    var lease: BlockPool.Lease = .init(&pool, io, blocks[0], 4);
    var group: std.Io.Group = .init;
    for ([_]i64{ 4, 1, 3, 2 }) |delay_ms| {
        try group.concurrent(io, struct {
            fn run(lease_: *BlockPool.Lease, io_: std.Io, delay_ms_: i64) void {
                io_.sleep(.fromMilliseconds(delay_ms_), .awake) catch unreachable;
                _ = lease_.complete();
            }
        }.run, .{ &lease, io, delay_ms });
    }
    try group.await(io);
    try std.testing.expect(lease.remaining.load(.acquire) == 0);

    var reacquired: [1]BlockPool.Block = undefined;
    try pool.acquireMany(io, &reacquired);
    try std.testing.expectEqual(@intFromPtr(blocks[0].ptr), @intFromPtr(reacquired[0].ptr));
    pool.releaseMany(io, &reacquired);
}

test "BlockPool reblocks retained arenas and grows on demand" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    _ = try workspace.allocate(150);
    _ = try workspace.allocate(70);
    _ = try workspace.allocate(65);

    var pool = try BlockPool.init(allocator, &workspace, 64, 349, 5);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 285), pool.mapped_bytes);
    try std.testing.expectEqual(@as(usize, 29), pool.unused_tail_bytes);
    try std.testing.expectEqual(@as(usize, 4), try pool.retainedRequestWidth(1));

    var blocks: [5]BlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    try std.testing.expectEqual(@as(usize, 4), workspace.allocations.items.len);
    try std.testing.expectEqual(@as(usize, 349), pool.mapped_bytes);
    try std.testing.expectEqual(@as(usize, 64), pool.newly_mapped_bytes);
    try std.testing.expectEqual(@as(usize, 29), pool.unused_tail_bytes);
    try std.testing.expectEqual(@as(usize, 5 * 64), pool.high_water * pool.block_size);
    pool.releaseMany(io, &blocks);
}

test "BlockPool potential request width accounts for retained arena tails" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    _ = try workspace.allocate(127);
    _ = try workspace.allocate(127);

    var pool = try BlockPool.init(allocator, &workspace, 64, 574, 4);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 126), pool.unused_tail_bytes);
    try std.testing.expectEqual(@as(usize, 3), try pool.potentialRequestWidth(2));
    try std.testing.expectEqual(@as(usize, 0), try pool.potentialRequestWidth(8));
    try std.testing.expectError(error.InvalidRequestBlockCount, pool.potentialRequestWidth(0));
    // A reserve the budget cannot cover is refused up front.
    if (BlockPool.init(allocator, &workspace, 64, 574, 8)) |result| {
        var unexpected = result;
        unexpected.deinit();
        return error.ExpectedCapacityError;
    } else |err| {
        try std.testing.expectEqual(error.RequestExceedsCapacity, err);
    }
}

test "BlockPool growth-free width subtracts the DMA stage" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    _ = try workspace.allocate(128 * 64);

    // Eight devices at eight in-flight blocks each reserve 64 of 128 blocks.
    var pool = try BlockPool.init(allocator, &workspace, 64, 2 * 128 * 64, 64);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 128), try pool.retainedRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 64), try pool.growthFreeRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 32), try pool.growthFreeRequestWidth(2));
    try std.testing.expectError(error.InvalidRequestBlockCount, pool.growthFreeRequestWidth(0));
}

test "BlockPool growth-free width saturates when the reserve covers the pool" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    _ = try workspace.allocate(2 * 64);

    // The pool has not grown to its reserve yet, which is allowed only while
    // the mapped-byte budget can still cover the deficit.
    var pool = try BlockPool.init(allocator, &workspace, 64, 16 * 64, 5);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 0), try pool.growthFreeRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 2), try pool.retainedRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 16), try pool.potentialRequestWidth(1));
}

test "BlockPool rejects requests that can never fit without leasing" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();

    _ = try workspace.allocate(2 * 64);

    var pool = try BlockPool.init(allocator, &workspace, 64, 2 * 64, 0);
    defer pool.deinit();

    var impossible: [3]BlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &impossible));
    try std.testing.expectEqual(@as(usize, 0), pool.high_water * pool.block_size);
    var fits: [2]BlockPool.Block = undefined;
    try pool.acquireMany(io, &fits);
    pool.releaseMany(io, &fits);
}
