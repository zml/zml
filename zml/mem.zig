const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;
const assert = std.debug.assert;
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const Buffer = @import("buffer.zig").Buffer;
const Device = @import("platform.zig").Device;
const Memory = @import("platform.zig").Memory;
const meta = @import("meta.zig");
const Platform = @import("platform.zig").Platform;
const Target = @import("platform.zig").Target;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/mem");

pub const DmaAllocator = union(enum) {
    passthrough: std.mem.Allocator,
    uib: UninitializedBufferAllocator,
    dmam: DmaMapAllocator,

    pub fn init(parent: std.mem.Allocator, device: *const Device) DmaAllocator {
        return switch (device.platform.target) {
            .cuda, .oneapi, .rocm => .{ .dmam = .init(parent, device.platform) },
            .tpu => .{ .uib = .init(device.memory(.host_pinned).?) },
            .cpu, .neuron, .metal => .{ .passthrough = parent },
        };
    }

    pub fn allocator(self: *const DmaAllocator) std.mem.Allocator {
        return switch (self.*) {
            .passthrough => |a| a,
            inline else => |*a| a.allocator(),
        };
    }
};

pub const UninitializedBufferAllocator = struct {
    memory: *const Memory,

    const Header = struct {
        buffer: *pjrt.Buffer,
    };

    pub fn init(memory: *const Memory) UninitializedBufferAllocator {
        return .{
            .memory = memory,
        };
    }

    pub fn allocator(self: *const UninitializedBufferAllocator) std.mem.Allocator {
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

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const self: *UninitializedBufferAllocator = @ptrCast(@alignCast(ctx));
        const pjrt_api = self.memory.platform.pjrt_api;
        const pjrt_client = self.memory.platform.pjrt_client;

        const total_len = std.mem.alignForward(usize, @sizeOf(Header) + len, alignment.toByteUnits());

        const pjrt_buffer = pjrt_client.createUninitializedBuffer(pjrt_api, .{
            .dims = &.{@intCast(total_len)},
            .element_type = .u8,
            .layout = .{
                .tiled = .{
                    .minor_to_major = &.{0},
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = .{ .memory = self.memory.pjrt_memory },
        }) catch return null;

        const opaque_ptr: [*]u8 = @ptrCast(pjrt_buffer.opaqueDeviceMemoryDataPointer(pjrt_api) catch unreachable);
        const data_with_header: []u8 = opaque_ptr[0..total_len];

        const header = std.mem.bytesAsValue(Header, opaque_ptr);
        header.* = .{
            .buffer = pjrt_buffer,
        };
        const offset = std.mem.alignForward(usize, @sizeOf(Header), alignment.toByteUnits());
        return @ptrCast(data_with_header[offset..]);
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: Alignment, ret_addr: usize) void {
        _ = ret_addr;
        const self: *UninitializedBufferAllocator = @ptrCast(@alignCast(ctx));
        const pjrt_api = self.memory.platform.pjrt_api;
        const header: *Header = @ptrFromInt(std.mem.alignBackward(usize, @intFromPtr(buf.ptr) - @sizeOf(Header), alignment.toByteUnits()));
        header.buffer.deinit(pjrt_api);
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return null;
    }
};

/// Host allocator for the loader's page-backed arenas. An allocation of at
/// least one huge page is aligned to it and, on Linux, advised into
/// transparent huge pages (a valid ordinary-page mapping when unavailable).
/// With a platform the pages are then registered with its PJRT client
/// through `dmaMap` (CUDA, oneAPI); without one they stay plain pages,
/// which is all the CPU plugin's transfers read from.
pub const DmaMapAllocator = struct {
    const transparent_huge_page_size = 2 * 1024 * 1024;

    parent: std.mem.Allocator,
    /// Null registers nothing.
    platform: ?*const Platform,

    pub fn init(parent: std.mem.Allocator, platform: *const Platform) DmaMapAllocator {
        return .{
            .parent = parent,
            .platform = platform,
        };
    }

    pub fn initPageable(parent: std.mem.Allocator) DmaMapAllocator {
        return .{
            .parent = parent,
            .platform = null,
        };
    }

    pub fn allocator(self: *const DmaMapAllocator) Allocator {
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

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *const DmaMapAllocator = @ptrCast(@alignCast(ctx));
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

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *const DmaMapAllocator = @ptrCast(@alignCast(ctx));
        if (self.platform) |platform| {
            platform.pjrt_client.dmaUnmap(platform.pjrt_api, @ptrCast(buf[0..buf.len])) catch unreachable;
        }
        self.parent.rawFree(buf, effectiveAlignment(alignment, buf.len), ret_addr);
    }

    /// Nothing below one huge page can be backed by one.
    fn effectiveAlignment(alignment: Alignment, len: usize) Alignment {
        if (comptime builtin.os.tag != .linux) return alignment;
        if (len < transparent_huge_page_size) return alignment;
        return alignment.max(.fromByteUnits(transparent_huge_page_size));
    }

    fn adviseHugePages(data: []u8) void {
        if (comptime builtin.os.tag != .linux) {
            return;
        }
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

/// Where the pages of the pinned arenas go. This is a host-memory choice,
/// not a device one: the loader's readers copy page cache into the arenas
/// and the engines read them back, and on a two-node host the node holding
/// the file's page cache is the one that saturates. Interleaving over the
/// memory nodes was never the worst placement on any device count measured,
/// while every single node was the worst on at least one; a device's own
/// node is not used (CTX.md, seventh pass).
pub const NumaPlacement = union(enum) {
    /// Interleave page by page over every host node that has memory
    /// (`/sys/devices/system/node/has_memory`). A single node, or no
    /// readable list, applies no policy.
    memory_nodes,
    /// One bit per node: a single bit binds, several interleave.
    nodes: u64,
    /// No policy: wherever the kernel and the driver put the pages.
    none,
};
// The policy applies to the page-backed arenas, `dmaMap`ed on the GPU
// targets and plain on CPU. ROCm's PJRT-pinned arenas are spread evenly
// over the nodes the devices report instead.

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

/// Owned, reusable host workspace for the direct loader: one pool of arenas
/// (pinned on the DMA targets, plain pages on CPU) retained across
/// benchmarks and loaders, borrowed by only one of them at a time.
/// Deinitialize it before its platform.
pub const DmaWorkspace = struct {
    // Preserve the minimum workspace capacity required by existing callers.
    const minimum_mapped_bytes = 32 * 1024 * 1024;

    const Status = enum(u8) {
        idle,
        in_use,
        destroying,
    };

    pub const Options = struct {
        /// Safety guard on the arenas' total host memory (pinned on the DMA
        /// targets), not an allocation target.
        max_mapped_bytes: usize = 16 * 1024 * 1024 * 1024,
        numa: NumaPlacement = .memory_nodes,
    };

    /// One NUMA node the ROCm devices report, with a device to allocate
    /// through and the bytes allocated so far; unknown nodes degrade to one
    /// entry per device.
    const HostNode = struct {
        device_index: usize,
        node: ?usize,
        bytes: usize = 0,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    /// Null in tests, where arenas are ordinary allocations.
    platform: ?*const Platform,
    /// ROCm maps pinned host memory through PJRT instead of `dmaMap`
    /// (registering through `dmaMap` there costs ~4.5 s/GiB). Each arena is
    /// allocated through a device on the node holding the fewest arena bytes
    /// so far, which keeps the pinned set balanced over the nodes: on eight
    /// MI300X all arenas on one node cost 30% of the load and a 61/39 split
    /// 20%. Empty when arenas are `dmaMap`ed; owned by `arena_mutex`.
    host_nodes: std.ArrayListUnmanaged(HostNode) = .empty,
    /// What `mapArena` produces, decided once from the target.
    arena_kind: ArenaKind,
    /// Node bits the next arena is placed on; zero leaves it to the kernel.
    /// Owned by `arena_mutex`: `growToBlocks` maps arenas concurrently, and
    /// a refused automatic placement clears it.
    numa_mask: u64,
    /// An explicit placement fails the arena when the kernel refuses it;
    /// the automatic one falls back to no policy.
    numa_explicit: bool,
    /// Owned by `arena_mutex` while `growToBlocks` maps arenas concurrently.
    allocations: std.ArrayListUnmanaged(DmaArenaAllocation) = .empty,
    /// The most recently mapped arena, empty before growth.
    source: []u8 = &.{},
    arena_mutex: std.Io.Mutex = .init,
    max_mapped_bytes: usize,
    allocated_bytes: std.atomic.Value(usize) = .init(0),
    status: std.atomic.Value(Status) = .init(.idle),

    pub const ArenaKind = std.meta.Tag(DmaArenaAllocation);

    /// How `target`'s arenas are made, or null for a platform that keeps
    /// the buffered backend. The direct loader needs a PJRT client that takes
    /// the arenas straight into its async transfer manager: the three DMA
    /// targets, and the CPU plugin, which copies from ordinary pages. TPU,
    /// neuron and metal stay buffered until their transfer path has been
    /// measured.
    fn arenaKind(target: Target) ?ArenaKind {
        return switch (target) {
            .cuda, .oneapi => .dma_map,
            .rocm => .pjrt_host,
            .cpu => .pageable,
            .tpu, .neuron, .metal => null,
        };
    }

    pub fn isSupported(platform: *const Platform) bool {
        return arenaKind(platform.target) != null;
    }

    pub fn validatePlatform(platform: *const Platform) !void {
        if (platform.devices.len == 0 or platform.devices.len > 64)
            return error.DmaDeviceMismatch;
        const device_kind = platform.devices[0].kind();
        for (platform.devices[1..]) |device| {
            if (!std.mem.eql(u8, device_kind, device.kind()))
                return error.HeterogeneousDmaUnsupported;
        }
    }

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: DmaWorkspace.Options,
    ) !DmaWorkspace {
        const arena_kind = arenaKind(platform.target) orelse return error.DmaBenchmarkUnsupported;
        try validatePlatform(platform);
        if (opts.max_mapped_bytes < minimum_mapped_bytes)
            return error.InvalidDmaLoadConfig;
        const numa_mask: u64, const numa_explicit = switch (opts.numa) {
            .memory_nodes => .{ memoryNodeMask(allocator, io), false },
            .nodes => |mask| .{ mask, true },
            .none => .{ 0, false },
        };
        if (numa_explicit and comptime builtin.os.tag != .linux)
            return error.DmaBenchmarkNumaUnsupported;
        var host_nodes: std.ArrayListUnmanaged(HostNode) = .empty;
        errdefer host_nodes.deinit(allocator);
        if (arena_kind == .pjrt_host) {
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
        }
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .host_nodes = host_nodes,
            .arena_kind = arena_kind,
            // Interleaving over one node is that node; leave it to the kernel.
            .numa_mask = if (numa_explicit or @popCount(numa_mask) > 1) numa_mask else 0,
            .numa_explicit = numa_explicit,
            .max_mapped_bytes = opts.max_mapped_bytes,
        };
    }

    /// Creates ordinary allocator-backed arenas for tests without a PJRT platform.
    pub fn initForTesting(
        allocator: std.mem.Allocator,
        io: std.Io,
        max_mapped_bytes: usize,
    ) !DmaWorkspace {
        if (!builtin.is_test) @compileError("initForTesting is only available in tests");
        return .{
            .allocator = allocator,
            .io = io,
            .platform = null,
            .arena_kind = .pageable,
            .numa_mask = 0,
            .numa_explicit = false,
            .max_mapped_bytes = max_mapped_bytes,
        };
    }

    pub fn acquire(self: *DmaWorkspace) !void {
        if (self.status.cmpxchgStrong(
            .idle,
            .in_use,
            .acq_rel,
            .acquire,
        ) != null) return error.DmaWorkspaceBusy;
    }

    pub fn release(self: *DmaWorkspace) void {
        const previous = self.status.swap(.idle, .release);
        std.debug.assert(previous == .in_use);
    }

    pub fn retainedMappedBytes(self: *const DmaWorkspace) usize {
        return self.allocatedBytes();
    }

    pub fn maxMappedBytes(self: *const DmaWorkspace) usize {
        return self.max_mapped_bytes;
    }

    pub fn deinit(self: *DmaWorkspace) void {
        if (self.status.cmpxchgStrong(
            .idle,
            .destroying,
            .acq_rel,
            .acquire,
        ) != null) @panic("DmaWorkspace.deinit called while borrowed");
        const io = self.io;
        const mapped_bytes = self.allocatedBytes();
        const started: std.Io.Timestamp = .now(io, .awake);
        for (self.allocations.items) |allocation| self.unmapArena(allocation);
        self.allocations.deinit(self.allocator);
        self.host_nodes.deinit(self.allocator);
        const elapsed_ns = elapsedNanoseconds(started, .now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }

    /// Returns the most recently allocated arena, or an empty slice before growth.
    pub fn latestArena(self: *const DmaWorkspace) []u8 {
        return self.source;
    }

    /// Borrows the most recent retained arena of at least `minimum_len`
    /// bytes; the caller must have allocated one.
    pub fn arenaAtLeast(self: *const DmaWorkspace, minimum_len: usize) []const u8 {
        var index = self.allocations.items.len;
        while (index != 0) {
            index -= 1;
            const arena = self.allocations.items[index].data();
            if (arena.len >= minimum_len) return arena;
        }
        unreachable;
    }

    /// The one arena growth path: the calibration ring, the post-selection
    /// reserve, the pre-grown source working set and load-time demand growth
    /// all map here. Refuses to cross the mapped ceiling, times the mapping,
    /// and returns the new arena. The workspace must be borrowed; callers
    /// serialize growth except for the pre-budgeted parts of `growToBlocks`.
    pub fn allocate(self: *DmaWorkspace, bytes: usize) ![]u8 {
        if (try std.math.add(usize, self.allocatedBytes(), bytes) > self.max_mapped_bytes)
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
            _ = self.allocated_bytes.fetchAdd(replacement.len, .release);
            self.source = replacement;
            break :mask self.numa_mask;
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

    fn placementName(mask: u64) []const u8 {
        return switch (@popCount(mask)) {
            0 => "unplaced",
            1 => "bind",
            else => "interleave",
        };
    }

    fn placementMask(self: *DmaWorkspace) u64 {
        self.arena_mutex.lockUncancelable(self.io);
        defer self.arena_mutex.unlock(self.io);
        return self.numa_mask;
    }

    /// The kernel refused the automatic placement: later arenas stay
    /// unplaced. The only transition the mask ever makes, so concurrent
    /// growth workers cannot undo it.
    fn leaveUnplaced(self: *DmaWorkspace) void {
        self.arena_mutex.lockUncancelable(self.io);
        defer self.arena_mutex.unlock(self.io);
        self.numa_mask = 0;
    }

    fn mapArena(self: *DmaWorkspace, bytes: usize) !DmaArenaAllocation {
        switch (self.arena_kind) {
            .pjrt_host => {
                const device_index = index: {
                    self.arena_mutex.lockUncancelable(self.io);
                    defer self.arena_mutex.unlock(self.io);
                    var emptiest = &self.host_nodes.items[0];
                    for (self.host_nodes.items[1..]) |*host_node| {
                        if (host_node.bytes < emptiest.bytes) emptiest = host_node;
                    }
                    emptiest.bytes += bytes;
                    break :index emptiest.device_index;
                };
                const memory = self.platform.?.devices[device_index].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable;
                return .{ .pjrt_host = try .init(memory, device_index, bytes) };
            },
            .dma_map, .pageable => {
                const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
                // The placement allocator sits between the page allocation
                // and the registration; both are built per call because the
                // workspace moves. A pageable arena is placed and huge-page
                // advised like a mapped one, never registered.
                var numa: NumaAllocator = .{
                    .parent = self.allocator,
                    .mask = self.placementMask(),
                    .explicit = self.numa_explicit,
                };
                const pages: DmaMapAllocator = if (self.arena_kind == .dma_map)
                    .init(numa.allocator(), self.platform.?)
                else
                    .initPageable(numa.allocator());
                const arena = try pages.allocator().alignedAlloc(u8, alignment, bytes);
                if (numa.mask == 0) self.leaveUnplaced();
                return if (self.arena_kind == .dma_map) .{ .dma_map = arena } else .{ .pageable = arena };
            },
        }
    }

    /// Frees through the allocator that made the arena; the placement
    /// allocator adds nothing to a free.
    fn unmapArena(self: *DmaWorkspace, allocation: DmaArenaAllocation) void {
        switch (allocation) {
            .pjrt_host => |pinned| pinned.deinit(),
            .dma_map => |arena| {
                const dma_map: DmaMapAllocator = .init(self.allocator, self.platform.?);
                dma_map.allocator().free(arena);
            },
            .pageable => |arena| {
                const pageable: DmaMapAllocator = .initPageable(self.allocator);
                pageable.allocator().free(arena);
            },
        }
    }

    fn allocatedBytes(self: *const DmaWorkspace) usize {
        return self.allocated_bytes.load(.acquire);
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    pub fn usableBlocks(self: *const DmaWorkspace, block_size: usize) !usize {
        var usable: usize = 0;
        for (self.allocations.items) |arena| {
            usable = std.math.add(usize, usable, arena.data().len / block_size) catch
                return error.DmaMappedBudgetExceeded;
        }
        return usable;
    }

    /// Arenas mapped at once by `growToBlocks`. Registering pinned memory
    /// scales with threads: on gb300-2 two 528 MiB arenas registered
    /// concurrently in 190 ms against 186 ms for one.
    const growth_parallelism = 4;

    /// Maps the blocks missing below `target_blocks` as up to
    /// `growth_parallelism` arenas registered concurrently. The aggregate
    /// check keeps a partial growth from crossing the ceiling. Requires a
    /// borrowed workspace and a nonzero block_size.
    pub fn growToBlocks(self: *DmaWorkspace, block_size: usize, target_blocks: usize) !void {
        const usable_blocks = try self.usableBlocks(block_size);
        const missing_blocks = target_blocks -| usable_blocks;
        if (missing_blocks == 0) return;
        const missing_bytes = std.math.mul(usize, missing_blocks, block_size) catch
            return error.DmaMappedBudgetExceeded;
        const mapped_after_growth = std.math.add(usize, self.allocatedBytes(), missing_bytes) catch
            return error.DmaMappedBudgetExceeded;
        if (mapped_after_growth > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;

        const Worker = struct {
            workspace: *DmaWorkspace,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                _ = worker.workspace.allocate(worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                };
            }
        };
        const parts = @min(growth_parallelism, missing_blocks);
        const blocks_per_part = std.math.divCeil(usize, missing_blocks, parts) catch unreachable;
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
};

const DmaArenaAllocation = union(enum) {
    /// Our pages, registered with the plugin through `dmaMap`.
    dma_map: []align(std.heap.page_size_min) u8,
    /// Our pages, never registered: the CPU plugin's arenas, and every
    /// arena in tests without a platform.
    pageable: []align(std.heap.page_size_min) u8,
    /// The plugin's pinned host memory, borrowed through a PJRT buffer.
    pjrt_host: PjrtPinnedHostAllocation,

    fn data(self: *const DmaArenaAllocation) []u8 {
        return switch (self.*) {
            .dma_map, .pageable => |bytes| bytes,
            .pjrt_host => |allocation| allocation.data,
        };
    }
};

const PjrtPinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,
    device_index: usize,

    fn init(memory: *const Memory, device_index: usize, size: usize) !PjrtPinnedHostAllocation {
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

    fn deinit(self: PjrtPinnedHostAllocation) void {
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

fn elapsedNanoseconds(started: std.Io.Timestamp, finished: std.Io.Timestamp) u64 {
    return @intCast(@max(started.durationTo(finished).nanoseconds, 0));
}

/// A per-load view of fixed-size blocks carved from the workspace's arenas.
/// The workspace retains arena ownership; this view owns only free-list
/// metadata.
pub const DmaBlockPool = struct {
    pub const Error = anyerror;
    const default_slab_size = 64 * 1024 * 1024;

    pub const Block = []u8;

    pub const Lease = struct {
        pool: *DmaBlockPool,
        io: std.Io,
        data: []u8,
        remaining: std.atomic.Value(usize),

        pub fn init(pool: *DmaBlockPool, io: std.Io, block: Block, references: usize) Lease {
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

        pub fn isComplete(self: *const Lease) bool {
            return self.remaining.load(.acquire) == 0;
        }
    };

    const ArenaOrigin = enum {
        retained,
        newly_mapped,
    };

    allocator: std.mem.Allocator,
    workspace: *DmaWorkspace,
    free_blocks: std.ArrayListUnmanaged(Block) = .empty,
    arenas: std.ArrayListUnmanaged([]u8) = .empty,
    block_size: usize,
    max_mapped_bytes: usize,
    mapped_bytes: usize,
    /// Blocks kept mapped as the growth floor: the DMA stage of every device.
    reserve: usize,
    capacity: usize = 0,
    retained_mapped_bytes: usize = 0,
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
    /// The workspace must remain borrowed and outlive this view.
    pub fn init(
        allocator: std.mem.Allocator,
        workspace: *DmaWorkspace,
        block_size: usize,
        max_mapped_bytes: usize,
        reserve: usize,
    ) !DmaBlockPool {
        const mapped_bytes = workspace.retainedMappedBytes();
        if (block_size == 0 or mapped_bytes > max_mapped_bytes)
            return error.RequestExceedsCapacity;
        var self: DmaBlockPool = .{
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
            enumerated_bytes = std.math.add(usize, enumerated_bytes, arena.len) catch
                return error.InvalidDmaWorkspace;
            try self.attachArena(arena, .retained);
        }
        if (enumerated_bytes != mapped_bytes) return error.InvalidDmaWorkspace;
        if (self.reservedGrowthBlocks() > self.remainingBlockBudget())
            return error.RequestExceedsCapacity;
        return self;
    }

    pub fn deinit(self: *DmaBlockPool) void {
        std.debug.assert(self.in_use == 0 and self.free_blocks.items.len == self.capacity);
        self.free_blocks.deinit(self.allocator);
        self.arenas.deinit(self.allocator);
        self.* = undefined;
    }

    /// Leases `output.len` blocks atomically, mapping a slab when the free
    /// list is short and the budget allows, otherwise waiting for releases.
    /// Allocates nothing once its arenas are attached.
    pub fn acquireMany(self: *DmaBlockPool, io: std.Io, output: []Block) Error!void {
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

    fn canEverAcquire(self: *const DmaBlockPool, blocks: usize) bool {
        const remaining_blocks = self.remainingBlockBudget();
        const reserved_growth = self.reservedGrowthBlocks();
        if (reserved_growth > remaining_blocks) return false;
        return blocks <= @max(self.capacity, self.reserve) + (remaining_blocks - reserved_growth);
    }

    fn remainingBlockBudget(self: *const DmaBlockPool) usize {
        return (self.max_mapped_bytes -| self.mapped_bytes) / self.block_size;
    }

    fn reservedGrowthBlocks(self: *const DmaBlockPool) usize {
        return self.reserve -| self.capacity;
    }

    fn grow(self: *DmaBlockPool) !bool {
        const block_count = @min(self.slab_blocks, self.remainingBlockBudget());
        if (block_count == 0) return false;
        try self.allocateSlab(block_count);
        return true;
    }

    /// Requests of `blocks_per_request` blocks the pool holds without
    /// mapping anything.
    pub fn retainedRequestWidth(self: *const DmaBlockPool, blocks_per_request: usize) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return self.capacity / blocks_per_request;
    }

    /// Requests of `blocks_per_request` blocks that can be leased without
    /// mapping a slab and without eating into the DMA stage reserve.
    pub fn growthFreeRequestWidth(self: *const DmaBlockPool, blocks_per_request: usize) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return (self.capacity -| self.reserve) / blocks_per_request;
    }

    /// The largest request width the pool could support if each request
    /// consumes `blocks_per_request` blocks. Arena tails count against the
    /// mapped-byte cap but do not contribute usable blocks.
    pub fn potentialRequestWidth(self: *const DmaBlockPool, blocks_per_request: usize) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return (self.capacity + self.remainingBlockBudget()) / blocks_per_request;
    }

    pub fn releaseMany(self: *DmaBlockPool, io: std.Io, blocks: []const Block) void {
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

    pub fn release(self: *DmaBlockPool, io: std.Io, block: Block) void {
        self.releaseMany(io, &.{block});
    }

    pub fn close(self: *DmaBlockPool, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.condition.broadcast(io);
    }

    pub fn highWaterBytes(self: *const DmaBlockPool) usize {
        return self.high_water * self.block_size;
    }

    pub fn mappedBytes(self: *const DmaBlockPool) usize {
        return self.mapped_bytes;
    }

    pub fn unusedTailBytes(self: *const DmaBlockPool) usize {
        return self.unused_tail_bytes;
    }

    pub fn newlyMappedBytes(self: *const DmaBlockPool) usize {
        return self.newly_mapped_bytes;
    }

    fn allocateSlab(self: *DmaBlockPool, block_count: usize) !void {
        const slab_len = try std.math.mul(usize, block_count, self.block_size);
        const mapped_before = self.workspace.retainedMappedBytes();
        if (mapped_before != self.mapped_bytes) return error.InvalidDmaWorkspace;
        const slab = try self.workspace.allocate(slab_len);
        const mapped_after = self.workspace.retainedMappedBytes();
        if (slab.len != slab_len or mapped_after < mapped_before or
            mapped_after - mapped_before != slab.len or mapped_after > self.max_mapped_bytes)
            return error.InvalidDmaWorkspace;
        self.mapped_bytes = mapped_after;
        try self.attachArena(slab, .newly_mapped);
    }

    fn attachArena(self: *DmaBlockPool, arena: []u8, origin: ArenaOrigin) !void {
        if (arena.len == 0) return error.InvalidDmaWorkspace;
        const arena_start = @intFromPtr(arena.ptr);
        const arena_end = std.math.add(usize, arena_start, arena.len) catch
            return error.InvalidDmaWorkspace;
        for (self.arenas.items) |existing| {
            const existing_start = @intFromPtr(existing.ptr);
            const existing_end = std.math.add(usize, existing_start, existing.len) catch
                return error.InvalidDmaWorkspace;
            if (arena_start < existing_end and existing_start < arena_end)
                return error.InvalidDmaWorkspace;
        }
        const block_count = arena.len / self.block_size;
        // Leased blocks are absent from `free_blocks`, so reserving relative to
        // its current length can leave too little space to return them after a
        // slab is attached under load. Keep storage sized for total capacity.
        try self.free_blocks.ensureTotalCapacity(self.allocator, self.capacity + block_count);
        try self.arenas.ensureUnusedCapacity(self.allocator, 1);
        self.arenas.appendAssumeCapacity(arena);
        for (0..block_count) |index| {
            self.free_blocks.appendAssumeCapacity(arena[index * self.block_size ..][0..self.block_size]);
        }
        self.capacity += block_count;
        const unused_tail_bytes = arena.len % self.block_size;
        switch (origin) {
            .retained => self.retained_mapped_bytes += arena.len,
            .newly_mapped => self.newly_mapped_bytes += arena.len,
        }
        self.unused_tail_bytes += unused_tail_bytes;
    }
};

pub const FixedBufferPool = struct {
    buffer: []u8,
    block_size: usize,
    q_buf: []const u16,
    q: std.Io.Queue(u16),

    pub fn init(allocator: std.mem.Allocator, buffer_: []u8, blocks_: u16) !FixedBufferPool {
        const block_size = @divExact(buffer_.len, blocks_);
        const q_buf = try allocator.alloc(u16, blocks_);
        for (q_buf, 0..) |*idx, i| {
            idx.* = @intCast(i);
        }
        var q: std.Io.Queue(u16) = .init(q_buf);
        q.type_erased.len = q.type_erased.buffer.len; // make the queue full
        return .{
            .buffer = buffer_,
            .block_size = block_size,
            .q = q,
            .q_buf = q_buf,
        };
    }

    pub fn deinit(self: *FixedBufferPool, allocator: std.mem.Allocator) void {
        allocator.free(self.q_buf);
    }

    pub fn get(self: *FixedBufferPool, io: std.Io) ![]u8 {
        const idx = try self.q.getOneUncancelable(io);
        return self.buffer[idx * self.block_size ..][0..self.block_size];
    }

    fn inRange(sub_buffer: []const u8, buffer: []const u8) bool {
        return @intFromPtr(sub_buffer.ptr) >= @intFromPtr(buffer.ptr) and
            @intFromPtr(sub_buffer[sub_buffer.len - 1 ..].ptr) <= @intFromPtr(buffer[buffer.len - 1 ..].ptr);
    }

    pub fn put(self: *FixedBufferPool, io: std.Io, buf: []u8) void {
        // is the pointer in range ?
        std.debug.assert(inRange(buf, self.buffer));
        const idx = @divExact(@intFromPtr(buf.ptr) - @intFromPtr(self.buffer.ptr), self.block_size);
        self.q.putOneUncancelable(io, @intCast(idx)) catch unreachable;
    }
};

test "parseNodeList accepts kernel node lists" {
    try std.testing.expectEqual(@as(u64, 0b11), parseNodeList("0-1\n"));
    try std.testing.expectEqual(@as(u64, 0b1101), parseNodeList("0,2-3"));
    try std.testing.expectEqual(@as(u64, 0b1), parseNodeList("0"));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList(""));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList("1-0"));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList("x"));
    try std.testing.expectEqual(@as(u64, 0b1), parseNodeList("0,64"));
}

test "DmaWorkspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 256);
            defer workspace.deinit();
            try workspace.acquire();
            defer workspace.release();
            _ = try workspace.allocate(64);
            _ = try workspace.allocate(128);
            try std.testing.expectEqual(@as(usize, 192), workspace.retainedMappedBytes());
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.allocate(128));
            try std.testing.expectEqual(@as(usize, 192), workspace.retainedMappedBytes());
            try std.testing.expectEqual(@as(usize, 128), workspace.arenaAtLeast(100).len);
            try std.testing.expectEqual(@as(usize, 3), try workspace.usableBlocks(64));
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.retainedMappedBytes());
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.retainedMappedBytes());
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.growToBlocks(64, 5));
            try std.testing.expectEqual(@as(usize, 256), workspace.retainedMappedBytes());
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "DmaWorkspace growth maps missing blocks as concurrent arenas" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 64 * 64);
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(3 * 64);
    try workspace.growToBlocks(64, 13);
    try std.testing.expectEqual(@as(usize, 13), try workspace.usableBlocks(64));
    // Ten missing blocks over four parts: 3, 3, 3, 1.
    try std.testing.expectEqual(@as(usize, 5), workspace.allocations.items.len);
    var total: usize = 0;
    for (workspace.allocations.items[1..]) |arena| total += arena.data().len;
    try std.testing.expectEqual(@as(usize, 10 * 64), total);
    try workspace.growToBlocks(64, 14);
    try std.testing.expectEqual(@as(usize, 6), workspace.allocations.items.len);
}

test "DmaBlockPool acquires request blocks atomically" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 4 * 64, 0);
    defer pool.deinit();

    var first: [3]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &first);
    try std.testing.expectEqual(@as(usize, 3 * 64), pool.highWaterBytes());
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.newlyMappedBytes());
    var oversized: [5]DmaBlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &oversized));

    var started: std.Io.Event = .unset;
    var acquired: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *DmaBlockPool, io_: std.Io, started_: *std.Io.Event, acquired_: *std.Io.Event) void {
            var blocks: [2]DmaBlockPool.Block = undefined;
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

test "DmaBlockPool retains free-list capacity when growing with blocks leased" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 9 * 64, 0);
    defer pool.deinit();
    pool.slab_blocks = 1;

    var held: [9]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &held);
    try std.testing.expect(pool.free_blocks.capacity >= pool.capacity);
    pool.releaseMany(io, &held);
}

test "DmaBlockPool acquisition allocates nothing once its arenas are attached" {
    const io = std.testing.io;
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{});
    var workspace = try DmaWorkspace.initForTesting(std.testing.allocator, io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(failing.allocator(), &workspace, 64, 4 * 64, 0);
    defer pool.deinit();

    var blocks: [3]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    pool.releaseMany(io, &blocks);

    failing.fail_index = failing.alloc_index;
    try pool.acquireMany(io, &blocks);
    try std.testing.expect(!failing.has_induced_failure);
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool close wakes blocked bulk acquisitions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 2 * 64, 0);
    defer pool.deinit();

    var held: [2]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &held);
    var started: std.Io.Event = .unset;
    var result: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *DmaBlockPool, io_: std.Io, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            var block: [1]DmaBlockPool.Block = undefined;
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

test "DmaBlockPool lease returns a block after out-of-order callbacks" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 64, 0);
    defer pool.deinit();

    var blocks: [1]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    var lease: DmaBlockPool.Lease = .init(&pool, io, blocks[0], 4);
    var group: std.Io.Group = .init;
    for ([_]i64{ 4, 1, 3, 2 }) |delay_ms| {
        try group.concurrent(io, struct {
            fn run(lease_: *DmaBlockPool.Lease, io_: std.Io, delay_ms_: i64) void {
                io_.sleep(.fromMilliseconds(delay_ms_), .awake) catch unreachable;
                _ = lease_.complete();
            }
        }.run, .{ &lease, io, delay_ms });
    }
    try group.await(io);
    try std.testing.expect(lease.isComplete());

    var reacquired: [1]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &reacquired);
    try std.testing.expectEqual(@intFromPtr(blocks[0].ptr), @intFromPtr(reacquired[0].ptr));
    pool.releaseMany(io, &reacquired);
}

test "DmaBlockPool reblocks retained arenas and grows on demand" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(150);
    _ = try workspace.allocate(70);
    _ = try workspace.allocate(65);

    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 349, 5);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 285), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 29), pool.unusedTailBytes());
    try std.testing.expectEqual(@as(usize, 4), try pool.retainedRequestWidth(1));

    var blocks: [5]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    try std.testing.expectEqual(@as(usize, 4), workspace.allocations.items.len);
    try std.testing.expectEqual(@as(usize, 349), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 64), pool.newlyMappedBytes());
    try std.testing.expectEqual(@as(usize, 29), pool.unusedTailBytes());
    try std.testing.expectEqual(@as(usize, 5 * 64), pool.highWaterBytes());
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool potential request width accounts for retained arena tails" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(127);
    _ = try workspace.allocate(127);

    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 574, 4);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 126), pool.unusedTailBytes());
    try std.testing.expectEqual(@as(usize, 3), try pool.potentialRequestWidth(2));
    try std.testing.expectEqual(@as(usize, 0), try pool.potentialRequestWidth(8));
    try std.testing.expectError(error.InvalidRequestBlockCount, pool.potentialRequestWidth(0));
    // A reserve the budget cannot cover is refused up front.
    try std.testing.expectError(
        error.RequestExceedsCapacity,
        DmaBlockPool.init(allocator, &workspace, 64, 574, 8),
    );
}

test "DmaBlockPool growth-free width subtracts the DMA stage" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(128 * 64);

    // Eight devices at eight in-flight blocks each reserve 64 of 128 blocks.
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 2 * 128 * 64, 64);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 128), try pool.retainedRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 64), try pool.growthFreeRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 32), try pool.growthFreeRequestWidth(2));
    try std.testing.expectError(error.InvalidRequestBlockCount, pool.growthFreeRequestWidth(0));
}

test "DmaBlockPool growth-free width saturates when the reserve covers the pool" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(2 * 64);

    // The pool has not grown to its reserve yet, which is allowed only while
    // the mapped-byte budget can still cover the deficit.
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 16 * 64, 5);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 0), try pool.growthFreeRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 2), try pool.retainedRequestWidth(1));
    try std.testing.expectEqual(@as(usize, 16), try pool.potentialRequestWidth(1));
}

test "DmaBlockPool rejects requests that can never fit without leasing" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(2 * 64);

    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 2 * 64, 0);
    defer pool.deinit();

    var impossible: [3]DmaBlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &impossible));
    try std.testing.expectEqual(@as(usize, 0), pool.highWaterBytes());
    var fits: [2]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &fits);
    pool.releaseMany(io, &fits);
}

/// Return a clone of a type with Tensors replaced by Buffer.
/// Non-Tensor metadata is stripped out of the resulting struct.
/// Recursively descends into the type.
pub fn Bufferized(comptime T: type) type {
    @setEvalBranchQuota(10_000);
    return meta.MapRestrict(Tensor, Buffer).map(T);
}

fn bufferizeInner(allocator: std.mem.Allocator, model: anytype, bufferized_: *Bufferized(@TypeOf(model))) !void {
    @setEvalBranchQuota(10_000);
    const Model = @TypeOf(model);
    const ModelBufferized = Bufferized(Model);

    if (ModelBufferized == Buffer) {
        bufferized_._shards = .empty;
        return;
    }

    const type_info = @typeInfo(ModelBufferized);
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            var initialized_fields: usize = 0;
            errdefer inline for (struct_type_info.fields, 0..) |field, index| {
                if (index < initialized_fields)
                    deinitBufferizedInner(allocator, &@field(bufferized_, field.name));
            };
            inline for (struct_type_info.fields) |field| {
                try bufferizeInner(allocator, @field(model, field.name), &@field(bufferized_, field.name));
                initialized_fields += 1;
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(ModelBufferized, @tagName(tag), undefined);
                    try bufferizeInner(allocator, v, &@field(bufferized_, @tagName(tag)));
                },
            }
        },
        .optional => |optional_type_info| {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = @as(optional_type_info.child, undefined);
                try bufferizeInner(allocator, model.?, &bufferized_.*.?);
            }
        },
        .pointer => |p| {
            switch (p.size) {
                .slice => {
                    const allocated = try allocator.alignedAlloc(p.child, .fromByteUnits(p.alignment orelse @alignOf(p.child)), model.len);
                    var initialized: usize = 0;
                    errdefer {
                        for (allocated[0..initialized]) |*element| deinitBufferizedInner(allocator, element);
                        allocator.free(allocated);
                    }
                    for (model, allocated) |src, *dst| {
                        try bufferizeInner(allocator, src, dst);
                        initialized += 1;
                    }
                    bufferized_.* = allocated;
                },
                else => unreachable,
            }
        },
        .array => |info| {
            var initialized: usize = 0;
            errdefer for (bufferized_.*[0..initialized]) |*element| deinitBufferizedInner(allocator, element);
            inline for (0..info.len) |index| {
                try bufferizeInner(allocator, model[index], &bufferized_.*[index]);
                initialized = index + 1;
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

fn deinitBufferizedInner(allocator: std.mem.Allocator, value: anytype) void {
    const Ptr = @TypeOf(value);
    const T = @typeInfo(Ptr).pointer.child;
    if (T == Buffer) {
        const buffer: *Buffer = @constCast(value);
        buffer.deinit();
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.fields) |field| {
            deinitBufferizedInner(allocator, &@field(value, field.name));
        },
        .@"union" => switch (value.*) {
            inline else => |*payload| deinitBufferizedInner(allocator, payload),
        },
        .optional => if (value.*) |*payload| {
            deinitBufferizedInner(allocator, payload);
        },
        .pointer => |info| switch (info.size) {
            .slice => {
                for (value.*) |*element| deinitBufferizedInner(allocator, element);
                allocator.free(value.*);
            },
            else => unreachable,
        },
        .array => for (&value.*) |*element| deinitBufferizedInner(allocator, element),
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

/// Deinitializes every accelerator buffer and frees the recursive slice
/// storage allocated by `bufferize`.
pub fn deinitBufferized(allocator: std.mem.Allocator, comptime ModelType: type, bufferized: *Bufferized(ModelType)) void {
    deinitBufferizedInner(allocator, bufferized);
}

/// Convert a model to its bufferized form by replacing Tensor fields with Buffer
/// and allocating any required slices using the provided allocator.
pub inline fn bufferize(allocator: std.mem.Allocator, comptime ModelType: type, model: *const ModelType) !Bufferized(ModelType) {
    var bufferized: Bufferized(ModelType) = undefined;
    try bufferizeInner(allocator, model.*, &bufferized);
    return bufferized;
}

test "bufferize rolls back earlier slice fields on allocation failure" {
    const Model = struct {
        first: []const Tensor,
        second: []const Tensor,
    };
    var tensors: [1]Tensor = undefined;
    const model: Model = .{ .first = &tensors, .second = &tensors };
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 1 });

    if (bufferize(failing.allocator(), Model, &model)) |result| {
        var unexpected = result;
        deinitBufferized(failing.allocator(), Model, &unexpected);
        return error.ExpectedOutOfMemory;
    } else |err| {
        try std.testing.expect(err == error.OutOfMemory);
    }
    try std.testing.expect(failing.has_induced_failure);
    try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
}

test "deinitBufferized frees recursive const slices" {
    const Layer = struct { weights: []const Tensor };
    const Model = struct {
        layers: []const Layer,
        fixed: [2]Tensor,
    };
    var weights: [2]Tensor = undefined;
    const layers = [_]Layer{
        .{ .weights = weights[0..1] },
        .{ .weights = weights[1..2] },
    };
    const model: Model = .{ .layers = &layers, .fixed = undefined };
    var bufferized = try bufferize(std.testing.allocator, Model, &model);
    deinitBufferized(std.testing.allocator, Model, &bufferized);
}
