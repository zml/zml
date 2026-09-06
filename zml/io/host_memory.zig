//! Loader-owned host arenas and per-load block leases.

const std = @import("std");
const Alignment = std.mem.Alignment;
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const Device = @import("../platform.zig").Device;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;

const log = std.log.scoped(.@"zml/mem");

// The largest supported calibration block must fit.
const minimum_mapped_bytes = 32 * 1024 * 1024;

/// One ROCm host-memory allocation path and its bytes allocated so far.
const HostNode = struct {
    any_device_index: usize,
    total_allocated_bytes: usize = 0,
};

/// Each allocation strategy keeps only the state it uses.
const Backend = union(enum) {
    pjrt_host: PjrtHost,
    dma_map: Pages,
    pageable: Pages,

    const PjrtHost = struct {
        platform: *const Platform,
        host_nodes: []HostNode,
        allocations: std.ArrayListUnmanaged(PinnedHostAllocation) = .empty,
    };

    const Pages = struct {
        allocator: HugePageAllocator,
        allocations: std.ArrayListUnmanaged([]align(std.heap.page_size_min) u8) = .empty,
    };

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
    ) !Backend {
        // Interleaving over one node is that node; leave it to the kernel.
        const numa_mask = interleaveMask(memoryNodeMask(allocator, io));
        return switch (platform.target) {
            .cuda, .oneapi => .{ .dma_map = .{
                .allocator = .init(allocator, platform, numa_mask),
            } },
            .cpu => .{ .pageable = .{
                .allocator = .initPageable(allocator, numa_mask),
            } },
            .rocm => {
                var discovered_nodes: std.ArrayListUnmanaged(struct {
                    device_index: usize,
                    node: usize,
                }) = .empty;
                defer discovered_nodes.deinit(allocator);
                devices: for (platform.devices, 0..) |device, device_index| {
                    if (device.memory(.host_pinned) == null) return error.PinnedHostMemoryUnavailable;
                    const node = device.numaNode() orelse continue;
                    for (discovered_nodes.items) |existing| {
                        if (existing.node == node) continue :devices;
                    }
                    try discovered_nodes.append(allocator, .{ .device_index = device_index, .node = node });
                }
                const host_nodes = try allocator.alloc(HostNode, @max(discovered_nodes.items.len, 1));
                errdefer allocator.free(host_nodes);
                if (discovered_nodes.items.len == 0) {
                    host_nodes[0] = .{ .any_device_index = 0 };
                } else {
                    for (host_nodes, discovered_nodes.items) |*host_node, discovered| {
                        host_node.* = .{ .any_device_index = discovered.device_index };
                    }
                }
                return .{ .pjrt_host = .{
                    .platform = platform,
                    .host_nodes = host_nodes,
                } };
            },
            .tpu, .neuron, .metal => error.DmaBenchmarkUnsupported,
        };
    }

    fn deinit(self: *Backend, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .pjrt_host => |*host| {
                for (host.allocations.items) |allocation| allocation.deinit();
                host.allocations.deinit(allocator);
                allocator.free(host.host_nodes);
            },
            .dma_map, .pageable => |*pages| {
                for (pages.allocations.items) |allocation| pages.allocator.free(allocation);
                pages.allocations.deinit(allocator);
            },
        }
    }

    fn arenaCount(self: *const Backend) usize {
        return switch (self.*) {
            .pjrt_host => |host| host.allocations.items.len,
            .dma_map, .pageable => |pages| pages.allocations.items.len,
        };
    }

    fn arenaAt(self: *const Backend, index: usize) []u8 {
        return switch (self.*) {
            .pjrt_host => |host| host.allocations.items[index].data,
            .dma_map, .pageable => |pages| pages.allocations.items[index],
        };
    }

    fn allocate(
        self: *Backend,
        allocator: std.mem.Allocator,
        io: std.Io,
        bytes: usize,
    ) ![]u8 {
        switch (self.*) {
            .pjrt_host => |*host| {
                const started: std.Io.Timestamp = .now(io, .awake);
                var host_node_index: usize = 0;
                for (host.host_nodes[1..], 1..) |host_node, index| {
                    if (host_node.total_allocated_bytes < host.host_nodes[host_node_index].total_allocated_bytes)
                        host_node_index = index;
                }
                const any_device_index = host.host_nodes[host_node_index].any_device_index;
                const memory = host.platform.devices[any_device_index].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable;
                const allocation: PinnedHostAllocation = try .init(memory, any_device_index, bytes);
                errdefer allocation.deinit();
                try host.allocations.append(allocator, allocation);
                host.host_nodes[host_node_index].total_allocated_bytes += allocation.data.len;
                log.info("DMA arena kind=pjrt_host device={d} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                    any_device_index,
                    @intFromPtr(allocation.data.ptr),
                    allocation.data.len,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, .now(io, .awake)))) / std.time.ns_per_ms,
                });
                return allocation.data;
            },
            .dma_map, .pageable => |*pages| {
                const started: std.Io.Timestamp = .now(io, .awake);
                const allocation = try pages.allocator.alloc(bytes);
                const mapped_at: std.Io.Timestamp = .now(io, .awake);
                errdefer pages.allocator.free(allocation);
                try pages.allocations.append(allocator, allocation);
                const finished: std.Io.Timestamp = .now(io, .awake);
                const numa_mask = pages.allocator.numa_mask;
                const placement = if (numa_mask == 0) "unplaced" else "interleave";
                log.info("DMA arena kind={s} placement={s} nodes=0x{x} address=0x{x} size={Bi:.2} allocation_ms={d:.3} map_ms={d:.3}", .{
                    @tagName(self.*),
                    placement,
                    numa_mask,
                    @intFromPtr(allocation.ptr),
                    allocation.len,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, finished))) / std.time.ns_per_ms,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, mapped_at))) / std.time.ns_per_ms,
                });
                return allocation;
            },
        }
    }

    /// Private owner of page-backed arena allocation policy. It applies
    /// huge-page alignment/advice and optionally registers the pages with
    /// the selected backend's PJRT client.
    const HugePageAllocator = struct {
        const transparent_huge_page_size = 2 * 1024 * 1024;
        const mpol_interleave = 3;

        parent: std.mem.Allocator,
        /// Null leaves the pages pageable for CPU transfers.
        platform: ?*const Platform,
        /// Zero leaves placement to the kernel. Automatic placement can
        /// fall back to zero if the kernel refuses it.
        numa_mask: u64,

        fn init(parent: std.mem.Allocator, platform: *const Platform, numa_mask: u64) HugePageAllocator {
            return .{
                .parent = parent,
                .platform = platform,
                .numa_mask = numa_mask,
            };
        }

        fn initPageable(parent: std.mem.Allocator, numa_mask: u64) HugePageAllocator {
            return .{
                .parent = parent,
                .platform = null,
                .numa_mask = numa_mask,
            };
        }

        fn alloc(self: *HugePageAllocator, len: usize) ![]align(std.heap.page_size_min) u8 {
            const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
            const effective_alignment = effectiveAlignment(alignment, len);
            const ptr = self.parent.rawAlloc(len, effective_alignment, @returnAddress()) orelse
                return error.OutOfMemory;
            const data: []align(std.heap.page_size_min) u8 = @alignCast(ptr[0..len]);
            self.place(data);
            adviseHugePages(data);
            if (self.platform) |platform| {
                platform.pjrt_client.dmaMap(platform.pjrt_api, @ptrCast(data)) catch {
                    self.parent.rawFree(data, effective_alignment, @returnAddress());
                    return error.OutOfMemory;
                };
            }
            return data;
        }

        fn place(self: *HugePageAllocator, data: []u8) void {
            const mask = self.numa_mask;
            if (mask == 0) return;
            if (comptime builtin.os.tag != .linux) return;

            const node_mask: [1]u64 = .{mask};
            const highest_node: usize = 63 - @clz(mask);
            const rc = std.os.linux.syscall6(
                .mbind,
                @intFromPtr(data.ptr),
                data.len,
                mpol_interleave,
                @intFromPtr(&node_mask),
                // Linux get_nodes() decrements maxnode before copying the mask;
                // raw callers include the same extra sentinel bit as libnuma.
                highest_node + 2,
                0,
            );
            if (std.os.linux.errno(rc) == .SUCCESS) return;
            log.warn("NUMA placement of DMA arenas over nodes 0x{x} refused ({s}); leaving them unplaced", .{
                mask,
                @tagName(std.os.linux.errno(rc)),
            });
            self.leaveUnplaced(mask);
        }

        fn leaveUnplaced(self: *HugePageAllocator, attempted_mask: u64) void {
            if (self.numa_mask == attempted_mask) self.numa_mask = 0;
        }

        fn free(self: *const HugePageAllocator, buf: []align(std.heap.page_size_min) u8) void {
            if (self.platform) |platform| {
                platform.pjrt_client.dmaUnmap(platform.pjrt_api, @ptrCast(buf)) catch unreachable;
            }
            const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
            self.parent.rawFree(buf, effectiveAlignment(alignment, buf.len), @returnAddress());
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

/// Host arenas owned by one direct loader. Calibration fills the initial
/// arena set during initialization; loading reuses and grows it. Deinitialize
/// after all transfers finish and before the platform.
pub const Workspace = struct {
    pub const Options = struct {
        /// Safety guard on the arenas' total host memory (pinned on the DMA
        /// targets), not an allocation target.
        max_mapped_bytes: usize = 16 * 1024 * 1024 * 1024,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    backend: Backend,
    max_mapped_bytes: usize,
    mapped_bytes: usize = 0,

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
        return .{
            .allocator = allocator,
            .io = io,
            .backend = try .init(allocator, io, platform),
            .max_mapped_bytes = opts.max_mapped_bytes,
        };
    }

    pub fn deinit(self: *Workspace) void {
        const io = self.io;
        const mapped_bytes = self.mapped_bytes;
        const started: std.Io.Timestamp = .now(io, .awake);
        self.backend.deinit(self.allocator);
        const elapsed_ns = elapsedNanoseconds(started, .now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }

    /// Borrows the newest retained arena that fits, when no arena growth is running.
    pub fn findArena(self: *const Workspace, minimum_len: usize) ?[]u8 {
        var index = self.backend.arenaCount();
        while (index != 0) {
            index -= 1;
            const arena = self.backend.arenaAt(index);
            if (arena.len >= minimum_len) return arena;
        }
        return null;
    }

    /// Retains one new arena within the mapped-byte ceiling.
    pub fn allocate(self: *Workspace, bytes: usize) ![]u8 {
        if (bytes > self.max_mapped_bytes - self.mapped_bytes)
            return error.DmaMappedBudgetExceeded;
        const allocation = try self.backend.allocate(self.allocator, self.io, bytes);
        self.mapped_bytes += allocation.len;
        return allocation;
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    pub fn usableBlocks(self: *const Workspace, block_size: usize) usize {
        var usable: usize = 0;
        for (0..self.backend.arenaCount()) |index| usable += self.backend.arenaAt(index).len / block_size;
        return usable;
    }

    /// Maps the blocks missing below `target_blocks` as one arena. Requires a
    /// nonzero block_size; the workspace has one coordinating owner.
    pub fn growToBlocks(self: *Workspace, block_size: usize, target_blocks: usize) !void {
        const usable_blocks = self.usableBlocks(block_size);
        const missing_blocks = target_blocks -| usable_blocks;
        if (missing_blocks == 0) return;
        if (missing_blocks > (self.max_mapped_bytes - self.mapped_bytes) / block_size)
            return error.DmaMappedBudgetExceeded;

        _ = try self.allocate(missing_blocks * block_size);
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
            .backend = .{ .pageable = .{
                .allocator = .initPageable(allocator, 0),
            } },
            .max_mapped_bytes = max_mapped_bytes,
        };
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
        const mapped_bytes = workspace.mapped_bytes;
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
        for (0..workspace.backend.arenaCount()) |index| {
            const arena = workspace.backend.arenaAt(index);
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
        const mapped_before = self.workspace.mapped_bytes;
        if (mapped_before != self.mapped_bytes) return error.InvalidDmaWorkspace;
        const slab = try self.workspace.allocate(slab_len);
        const mapped_after = self.workspace.mapped_bytes;
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

/// Interleaving is useful only when the host exposes several memory nodes.
fn interleaveMask(memory_node_mask: u64) u64 {
    return if (@popCount(memory_node_mask) > 1) memory_node_mask else 0;
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

test "HugePageAllocator retains automatic NUMA fallback state" {
    var allocator: Workspace.Backend.HugePageAllocator = .initPageable(std.testing.allocator, 0b11);
    allocator.leaveUnplaced(0b11);
    try std.testing.expectEqual(@as(u64, 0), allocator.numa_mask);
}

test "automatic NUMA placement interleaves only multiple memory nodes" {
    try std.testing.expectEqual(@as(u64, 0), interleaveMask(0));
    try std.testing.expectEqual(@as(u64, 0), interleaveMask(0b1));
    try std.testing.expectEqual(@as(u64, 0), interleaveMask(0b1000));
    try std.testing.expectEqual(@as(u64, 0b1001), interleaveMask(0b1001));
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
    try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes);
}

test "Workspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var workspace = try Workspace.initForTesting(allocator, std.testing.io, 256);
            defer workspace.deinit();

            _ = try workspace.allocate(64);
            _ = try workspace.allocate(128);
            try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes);
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.allocate(128));
            try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes);
            try std.testing.expectEqual(@as(usize, 128), workspace.findArena(100).?.len);
            try std.testing.expectEqual(@as(usize, 3), workspace.usableBlocks(64));
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes);
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes);
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.growToBlocks(64, 5));
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "Workspace growth maps missing blocks as one arena" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, 64 * 64);
    defer workspace.deinit();

    _ = try workspace.allocate(3 * 64);
    try workspace.growToBlocks(64, 13);
    try std.testing.expectEqual(@as(usize, 13), workspace.usableBlocks(64));
    try std.testing.expectEqual(@as(usize, 2), workspace.backend.arenaCount());
    try std.testing.expectEqual(@as(usize, 10 * 64), workspace.backend.arenaAt(1).len);
    try workspace.growToBlocks(64, 14);
    try std.testing.expectEqual(@as(usize, 3), workspace.backend.arenaCount());
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
    try std.testing.expectEqual(@as(usize, 4), workspace.backend.arenaCount());
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
