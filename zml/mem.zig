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

/// Host allocator for CUDA and oneAPI DMA mappings. Linux allocations request
/// transparent huge-page backing, but remain valid ordinary-page mappings when
/// unavailable.
pub const DmaMapAllocator = struct {
    const transparent_huge_page_size = 2 * 1024 * 1024;

    parent: std.mem.Allocator,
    platform: *const Platform,

    pub fn init(parent: std.mem.Allocator, platform: *const Platform) DmaMapAllocator {
        return .{
            .parent = parent,
            .platform = platform,
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
        const effective_alignment = self.effectiveAlignment(alignment);
        const allocation = self.parent.rawAlloc(len, effective_alignment, ret_addr);
        if (allocation) |loc| {
            const data = loc[0..len];
            self.adviseHugePages(data);
            self.platform.pjrt_client.dmaMap(self.platform.pjrt_api, @ptrCast(data)) catch {
                self.parent.rawFree(data, effective_alignment, ret_addr);
                return null;
            };
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
        self.platform.pjrt_client.dmaUnmap(self.platform.pjrt_api, @ptrCast(buf[0..buf.len])) catch unreachable;
        self.parent.rawFree(buf, self.effectiveAlignment(alignment), ret_addr);
    }

    fn effectiveAlignment(self: *const DmaMapAllocator, alignment: Alignment) Alignment {
        _ = self;
        if (comptime builtin.os.tag != .linux) return alignment;
        return alignment.max(.fromByteUnits(transparent_huge_page_size));
    }

    fn adviseHugePages(self: *const DmaMapAllocator, data: []u8) void {
        _ = self;
        if (comptime builtin.os.tag != .linux) {
            return;
        }

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

fn resolveNumaNodes(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    override: []const usize,
    disabled: bool,
) ![]?usize {
    const result = try allocator.alloc(?usize, platform.devices.len);
    @memset(result, null);
    if (disabled) return result;
    if (override.len != 0) {
        for (override, result) |node, *stored| stored.* = node;
        return result;
    }

    for (platform.devices, 0..) |_, device_index| {
        const node = platform.devices[device_index].numaNode() orelse {
            @memset(result, null);
            return result;
        };
        if (node >= NumaAllocator.max_nodes) {
            @memset(result, null);
            return result;
        }
        result[device_index] = node;
    }
    if (comptime builtin.os.tag != .linux) @memset(result, null);
    return result;
}

const KnownPoolTopology = struct {
    pool_count: usize = 0,
    pool_nodes: [64]usize = undefined,
    first_device_indices: [64]usize = undefined,
    device_pool_indices: [64]usize = undefined,

    fn init(device_numa_nodes: []const ?usize) !KnownPoolTopology {
        var result: KnownPoolTopology = .{};
        devices: for (device_numa_nodes, 0..) |maybe_node, device_index| {
            const node = maybe_node orelse return error.InvalidDmaLoadConfig;
            for (result.pool_nodes[0..result.pool_count], 0..) |known, pool_index| {
                if (known == node) {
                    result.device_pool_indices[device_index] = pool_index;
                    continue :devices;
                }
            }
            const pool_index = result.pool_count;
            result.pool_nodes[pool_index] = node;
            result.first_device_indices[pool_index] = device_index;
            result.device_pool_indices[device_index] = pool_index;
            result.pool_count += 1;
        }
        return result;
    }
};

/// Owned, reusable host-DMA workspace. It retains mapped arenas across
/// benchmarks and loaders, and may be borrowed by only one of them at a time.
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
        /// Safety guard on total pinned host memory, not an allocation target.
        max_mapped_bytes: usize = 16 * 1024 * 1024 * 1024,
        /// Optional device-index to NUMA-node override. When absent, complete
        /// PJRT topology selects local pools; otherwise one shared pool is used.
        device_numa_nodes: []const usize = &.{},
        /// Forces one shared unbound pool even when topology is known.
        disable_numa_pools: bool = false,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    pools: []DmaArenaPool,
    device_pool_indices: []usize,
    max_mapped_bytes: usize,
    allocated_bytes: std.atomic.Value(usize) = .init(0),
    status: std.atomic.Value(Status) = .init(.idle),

    pub fn isSupported(platform: *const Platform) bool {
        return platform.target == .cuda or platform.target == .rocm or
            platform.target == .oneapi;
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
        if (!isSupported(platform)) return error.DmaBenchmarkUnsupported;
        try validatePlatform(platform);
        if (opts.max_mapped_bytes < minimum_mapped_bytes)
            return error.InvalidDmaLoadConfig;
        if (opts.device_numa_nodes.len != 0) {
            if (opts.disable_numa_pools or opts.device_numa_nodes.len != platform.devices.len)
                return error.InvalidDmaLoadConfig;
            if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
            for (opts.device_numa_nodes) |node| {
                if (node >= NumaAllocator.max_nodes) return error.InvalidDmaLoadConfig;
            }
        }
        const resolved_numa_nodes = try resolveNumaNodes(
            allocator,
            platform,
            opts.device_numa_nodes,
            opts.disable_numa_pools,
        );
        defer allocator.free(resolved_numa_nodes);
        return initResolved(
            allocator,
            io,
            platform,
            resolved_numa_nodes,
            opts.max_mapped_bytes,
        );
    }

    /// Creates ordinary allocator-backed arenas for tests without a PJRT platform.
    pub fn initForTesting(
        allocator: std.mem.Allocator,
        io: std.Io,
        node_count: usize,
        max_mapped_bytes: usize,
    ) !DmaWorkspace {
        if (!builtin.is_test) @compileError("initForTesting is only available in tests");
        std.debug.assert(node_count > 0 and node_count <= 64);
        const pools = try allocator.alloc(DmaArenaPool, node_count);
        errdefer allocator.free(pools);
        const device_pool_indices = try allocator.alloc(usize, node_count);
        for (pools, device_pool_indices, 0..) |*pool, *pool_index, index| {
            pool.* = .{
                .numa_allocator = .{ .parent = allocator, .node = null },
                .dma_allocator = .{ .passthrough = allocator },
                .pjrt_host_memory = null,
            };
            pool_index.* = index;
        }
        return .{
            .allocator = allocator,
            .io = io,
            .pools = pools,
            .device_pool_indices = device_pool_indices,
            .max_mapped_bytes = max_mapped_bytes,
        };
    }

    fn initResolved(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        device_numa_nodes: []const ?usize,
        max_mapped_bytes: usize,
    ) !DmaWorkspace {
        if (device_numa_nodes.len != platform.devices.len)
            return error.DmaDeviceMismatch;
        const topology: ?KnownPoolTopology = if (device_numa_nodes[0] == null)
            null
        else
            try .init(device_numa_nodes);
        const pool_count = if (topology) |known| known.pool_count else 1;
        const pools = try allocator.alloc(DmaArenaPool, pool_count);
        errdefer allocator.free(pools);
        const device_pool_indices = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(device_pool_indices);
        if (topology == null) {
            const pool = &pools[0];
            pool.numa_allocator = .{ .parent = allocator, .node = null };
            pool.dma_allocator = .{ .dmam = .init(pool.numa_allocator.allocator(), platform) };
            pool.pjrt_host_memory = if (platform.target == .rocm)
                platform.devices[0].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable
            else
                null;
            pool.allocations = .empty;
            pool.source = &.{};
            @memset(device_pool_indices, 0);
            return .{
                .allocator = allocator,
                .io = io,
                .pools = pools,
                .device_pool_indices = device_pool_indices,
                .max_mapped_bytes = max_mapped_bytes,
            };
        }

        const known = topology.?;
        @memcpy(device_pool_indices, known.device_pool_indices[0..device_numa_nodes.len]);
        for (
            known.pool_nodes[0..known.pool_count],
            known.first_device_indices[0..known.pool_count],
            pools,
        ) |node, device_index, *pool| {
            pool.numa_allocator = .{ .parent = allocator, .node = node };
            pool.dma_allocator = .{ .dmam = .init(pool.numa_allocator.allocator(), platform) };
            pool.pjrt_host_memory = if (platform.target == .rocm)
                platform.devices[device_index].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable
            else
                null;
            pool.allocations = .empty;
            pool.source = &.{};
        }
        return .{
            .allocator = allocator,
            .io = io,
            .pools = pools,
            .device_pool_indices = device_pool_indices,
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

    pub fn numaPoolCount(self: *const DmaWorkspace) usize {
        return self.pools.len;
    }

    pub fn hasStrictAffinity(self: *const DmaWorkspace) bool {
        return self.pools[0].numa_allocator.node != null;
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
        for (self.pools) |*pool| {
            for (pool.allocations.items) |allocation| {
                allocation.deinit(pool.dma_allocator.allocator());
            }
            pool.allocations.deinit(self.allocator);
        }
        self.allocator.free(self.device_pool_indices);
        self.allocator.free(self.pools);
        const elapsed_ns = elapsedNanoseconds(started, .now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }

    /// Returns the most recently allocated arena, or an empty slice before growth.
    pub fn latestArena(self: *const DmaWorkspace, pool_index: usize) []u8 {
        return self.pools[pool_index].source;
    }

    /// Borrows a retained arena. The caller must have allocated an arena of at
    /// least `minimum_len` in the device's pool.
    pub fn arenaForDevice(
        self: *const DmaWorkspace,
        device_index: usize,
        minimum_len: usize,
    ) []const u8 {
        const pool = &self.pools[self.device_pool_indices[device_index]];
        var index = pool.allocations.items.len;
        while (index != 0) {
            index -= 1;
            const arena = pool.allocations.items[index].data();
            if (arena.len >= minimum_len) return arena;
        }
        unreachable;
    }

    /// The one arena growth path: the calibration ring, the post-selection
    /// reserves, the pre-grown source working set and load-time demand growth
    /// all map here. Refuses to cross the mapped ceiling, times the mapping,
    /// and returns the new arena. The workspace must be borrowed; callers must
    /// serialize growth except for the pre-budgeted workers in growToBlockTargets.
    pub fn allocate(self: *DmaWorkspace, pool_index: usize, bytes: usize) ![]u8 {
        if (try std.math.add(usize, self.allocatedBytes(), bytes) > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;
        const pool = &self.pools[pool_index];
        const started: std.Io.Timestamp = .now(self.io, .awake);
        var allocation: DmaArenaAllocation = if (pool.pjrt_host_memory) |host_memory|
            .{ .pjrt_host = try .init(host_memory, bytes) }
        else blk: {
            const dma_map_allocator = pool.dma_allocator.allocator();
            break :blk .{ .dma_map = try dma_map_allocator.alignedAlloc(
                u8,
                .fromByteUnits(std.heap.page_size_min),
                bytes,
            ) };
        };
        errdefer allocation.deinit(pool.dma_allocator.allocator());
        const mapped_at: std.Io.Timestamp = .now(self.io, .awake);
        try pool.allocations.append(self.allocator, allocation);
        const replacement = allocation.data();
        _ = self.allocated_bytes.fetchAdd(replacement.len, .release);
        pool.source = replacement;
        const finished_at: std.Io.Timestamp = .now(self.io, .awake);
        const map_ns = if (pool.pjrt_host_memory != null)
            0
        else
            elapsedNanoseconds(started, mapped_at);
        const elapsed_ns = elapsedNanoseconds(started, finished_at);
        const allocation_kind = if (pool.pjrt_host_memory != null) "pjrt_host" else "dma_map";
        if (pool.numa_allocator.node) |node| {
            log.info("DMA mapped arena numa_node={d} allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                node,
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        } else {
            log.info("DMA mapped arena numa_node=single allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        }
        return replacement;
    }

    fn allocatedBytes(self: *const DmaWorkspace) usize {
        return self.allocated_bytes.load(.acquire);
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    pub fn usableBlocks(self: *const DmaWorkspace, pool_index: usize, block_size: usize) !usize {
        const pool = &self.pools[pool_index];
        var usable: usize = 0;
        for (pool.allocations.items) |arena| {
            usable = std.math.add(usize, usable, arena.data().len / block_size) catch
                return error.DmaMappedBudgetExceeded;
        }
        return usable;
    }

    /// Grows every pool that holds fewer than `block_targets[pool]` blocks,
    /// mapping the missing slabs of independent nodes concurrently. The
    /// aggregate check keeps a partial growth from crossing the ceiling.
    /// Requires a borrowed workspace, nonzero block_size and one target per pool.
    pub fn growToBlockTargets(
        self: *DmaWorkspace,
        block_size: usize,
        block_targets: []const usize,
    ) !void {
        const missing_bytes = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(missing_bytes);
        var missing_total: usize = 0;
        for (block_targets, missing_bytes, 0..) |target, *missing, pool_index| {
            const usable_blocks = try self.usableBlocks(pool_index, block_size);
            missing.* = std.math.mul(usize, target -| usable_blocks, block_size) catch
                return error.DmaMappedBudgetExceeded;
            missing_total = std.math.add(usize, missing_total, missing.*) catch
                return error.DmaMappedBudgetExceeded;
        }
        if (missing_total == 0) return;
        const mapped_after_growth = std.math.add(
            usize,
            self.allocatedBytes(),
            missing_total,
        ) catch return error.DmaMappedBudgetExceeded;
        if (mapped_after_growth > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;

        const Worker = struct {
            pools: *DmaWorkspace,
            pool_index: usize,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                _ = worker.pools.allocate(worker.pool_index, worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(
                        0,
                        @intFromError(err),
                        .release,
                        .monotonic,
                    );
                };
            }
        };
        var first_error: std.atomic.Value(u16) = .init(0);
        var group: std.Io.Group = .init;
        var group_error: ?anyerror = null;
        for (missing_bytes, 0..) |bytes, pool_index| {
            if (bytes == 0) continue;
            group.concurrent(self.io, Worker.run, .{Worker{
                .pools = self,
                .pool_index = pool_index,
                .bytes = bytes,
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

const DmaArenaPool = struct {
    numa_allocator: NumaAllocator,
    dma_allocator: DmaAllocator,
    pjrt_host_memory: ?*const Memory,
    allocations: std.ArrayListUnmanaged(DmaArenaAllocation) = .empty,
    source: []u8 = &.{},
};

const DmaArenaAllocation = union(enum) {
    dma_map: []align(std.heap.page_size_min) u8,
    pjrt_host: PjrtPinnedHostAllocation,

    fn data(self: *const DmaArenaAllocation) []u8 {
        return switch (self.*) {
            .dma_map => |bytes| bytes,
            .pjrt_host => |allocation| allocation.data,
        };
    }

    fn deinit(
        self: DmaArenaAllocation,
        dma_map_allocator: std.mem.Allocator,
    ) void {
        switch (self) {
            .dma_map => |bytes| dma_map_allocator.free(bytes),
            .pjrt_host => |allocation| allocation.deinit(),
        }
    }
};

const PjrtPinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,

    fn init(memory: *const Memory, size: usize) !PjrtPinnedHostAllocation {
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
        };
    }

    fn deinit(self: PjrtPinnedHostAllocation) void {
        self.buffer.decreaseExternalReferenceCount(self.api) catch unreachable;
        self.buffer.deinit(self.api);
    }
};

const NumaAllocator = struct {
    const max_nodes = 1024;
    const mpol_bind = 2;

    parent: std.mem.Allocator,
    node: ?usize,

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
        const node = self.node orelse return allocation;
        if (comptime builtin.os.tag != .linux) {
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }

        const word_bits = @bitSizeOf(usize);
        var node_mask: [max_nodes / word_bits]usize = @splat(0);
        node_mask[node / word_bits] = @as(usize, 1) << @intCast(node % word_bits);
        const rc = std.os.linux.syscall6(
            .mbind,
            @intFromPtr(allocation),
            len,
            mpol_bind,
            @intFromPtr(&node_mask),
            // Linux get_nodes() decrements maxnode before copying the mask;
            // raw callers include the same extra sentinel bit as libnuma.
            node + 2,
            0,
        );
        if (std.os.linux.errno(rc) != .SUCCESS) {
            log.err("unable to bind DMA benchmark allocation ({Bi:.2}) to NUMA node {d}: {s}", .{
                len,
                node,
                @tagName(std.os.linux.errno(rc)),
            });
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }
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

/// A per-load view of fixed-size blocks carved from owned DMA arenas. The
/// workspace retains arena ownership; this view owns only free-list metadata.
pub const DmaBlockPool = struct {
    pub const Error = anyerror;
    const default_slab_size = 64 * 1024 * 1024;
    const unassigned_node = std.math.maxInt(usize);

    /// A zero mask means the shared-pool fallback. Otherwise each bit names an
    /// eligible NUMA pool. One bit is a strict local requirement; multiple
    /// bits describe a replicated block that may use any participating node.
    pub const Affinity = struct {
        eligible_nodes: u64 = 0,

        pub fn node(node_index: usize) Affinity {
            std.debug.assert(node_index < 64);
            return .{ .eligible_nodes = @as(u64, 1) << @intCast(node_index) };
        }

        pub fn replicated(eligible_nodes: u64) Affinity {
            std.debug.assert(eligible_nodes != 0);
            return .{ .eligible_nodes = eligible_nodes };
        }
    };

    pub const NodeStats = struct {
        retained_mapped_bytes: usize,
        newly_mapped_bytes: usize,
        leased_high_water_bytes: usize,
        unused_tail_bytes: usize,
    };

    pub const Block = struct {
        data: []u8,
        node_index: usize,
    };

    pub const Lease = struct {
        pool: *DmaBlockPool,
        io: std.Io,
        data: []u8,
        node_index: usize,
        remaining: std.atomic.Value(usize),

        pub fn init(pool: *DmaBlockPool, io: std.Io, block: Block, references: usize) Lease {
            std.debug.assert(references > 0);
            return .{
                .pool = pool,
                .io = io,
                .data = block.data,
                .node_index = block.node_index,
                .remaining = .init(references),
            };
        }

        /// Completes one reference and returns whether this was the final one.
        pub fn complete(self: *Lease) bool {
            const previous = self.remaining.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous == 1) self.pool.release(self.io, .{
                .data = self.data,
                .node_index = self.node_index,
            });
            return previous == 1;
        }

        pub fn isComplete(self: *const Lease) bool {
            return self.remaining.load(.acquire) == 0;
        }
    };

    pub const AcquireScratch = struct {
        allocator: std.mem.Allocator,
        assignments: []usize,
        masks: []u64,
        available: []usize,
        planned: []usize,
        potential: []usize,

        fn init(allocator: std.mem.Allocator, maximum_jobs: usize, node_count: usize) !AcquireScratch {
            const assignments = try allocator.alloc(usize, maximum_jobs);
            errdefer allocator.free(assignments);
            const masks = try allocator.alloc(u64, maximum_jobs);
            errdefer allocator.free(masks);
            const available = try allocator.alloc(usize, node_count);
            errdefer allocator.free(available);
            const planned = try allocator.alloc(usize, node_count);
            errdefer allocator.free(planned);
            const potential = try allocator.alloc(usize, node_count);
            return .{
                .allocator = allocator,
                .assignments = assignments,
                .masks = masks,
                .available = available,
                .planned = planned,
                .potential = potential,
            };
        }

        pub fn deinit(self: *AcquireScratch) void {
            self.allocator.free(self.potential);
            self.allocator.free(self.planned);
            self.allocator.free(self.available);
            self.allocator.free(self.masks);
            self.allocator.free(self.assignments);
            self.* = undefined;
        }
    };

    const Node = struct {
        free_blocks: std.ArrayListUnmanaged(Block) = .empty,
        arenas: std.ArrayListUnmanaged([]u8) = .empty,
        capacity: usize = 0,
        in_use: usize = 0,
        high_water: usize = 0,
        reserve: usize = 0,
        retained_mapped_bytes: usize = 0,
        newly_mapped_bytes: usize = 0,
        unused_tail_bytes: usize = 0,
    };

    const ArenaOrigin = enum {
        retained,
        newly_mapped,
    };

    allocator: std.mem.Allocator,
    workspace: *DmaWorkspace,
    nodes: []Node,
    block_size: usize,
    max_mapped_bytes: usize,
    mapped_bytes: usize,
    unused_tail_bytes: usize = 0,
    newly_mapped_bytes: usize = 0,
    slab_blocks: usize,
    in_use: usize = 0,
    high_water: usize = 0,
    next_node: usize = 0,
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
        reserves: []const usize,
    ) !DmaBlockPool {
        const mapped_bytes = workspace.retainedMappedBytes();
        if (block_size == 0 or workspace.numaPoolCount() == 0 or workspace.numaPoolCount() > 64 or
            reserves.len != workspace.numaPoolCount() or mapped_bytes > max_mapped_bytes)
            return error.RequestExceedsCapacity;
        const nodes = try allocator.alloc(Node, workspace.numaPoolCount());
        for (nodes, reserves) |*node, reserve| node.* = .{ .reserve = reserve };
        var self: DmaBlockPool = .{
            .allocator = allocator,
            .workspace = workspace,
            .nodes = nodes,
            .block_size = block_size,
            .max_mapped_bytes = max_mapped_bytes,
            .mapped_bytes = mapped_bytes,
            .slab_blocks = @max(@as(usize, 1), default_slab_size / block_size),
        };
        errdefer self.deinit();
        var enumerated_bytes: usize = 0;
        for (0..workspace.numaPoolCount()) |node_index| {
            for (workspace.pools[node_index].allocations.items) |*allocation| {
                const arena = allocation.data();
                enumerated_bytes = std.math.add(usize, enumerated_bytes, arena.len) catch
                    return error.InvalidDmaWorkspace;
                try self.attachArena(node_index, arena, .retained);
            }
        }
        if (enumerated_bytes != mapped_bytes) return error.InvalidDmaWorkspace;
        if (self.reservedGrowthBlocks() > self.remainingBlockBudget())
            return error.RequestExceedsCapacity;
        return self;
    }

    pub fn deinit(self: *DmaBlockPool) void {
        std.debug.assert(self.in_use == 0);
        for (self.nodes) |*node| {
            std.debug.assert(node.in_use == 0 and node.free_blocks.items.len == node.capacity);
            node.free_blocks.deinit(self.allocator);
            node.arenas.deinit(self.allocator);
        }
        self.allocator.free(self.nodes);
        self.* = undefined;
    }

    /// Allocates matching state for one concurrent caller. Reuse it across
    /// acquisitions, but do not share it between calls that may overlap.
    pub fn acquireScratch(
        self: *const DmaBlockPool,
        allocator: std.mem.Allocator,
        maximum_jobs: usize,
    ) !AcquireScratch {
        if (maximum_jobs > self.max_mapped_bytes / self.block_size)
            return error.RequestExceedsCapacity;
        return .init(allocator, maximum_jobs, self.nodes.len);
    }

    pub fn acquireMany(
        self: *DmaBlockPool,
        io: std.Io,
        output: []Block,
        affinities: []const Affinity,
        scratch: *AcquireScratch,
    ) Error!void {
        if (output.len != affinities.len) return error.InvalidAffinity;
        if (output.len == 0) return;
        if (output.len > scratch.assignments.len or output.len > scratch.masks.len or
            scratch.available.len != self.nodes.len or scratch.planned.len != self.nodes.len or
            scratch.potential.len != self.nodes.len)
            return error.RequestExceedsCapacity;

        const assignments = scratch.assignments[0..output.len];
        const masks = scratch.masks[0..output.len];
        const available = scratch.available;
        const planned = scratch.planned;
        const potential = scratch.potential;
        for (affinities, masks) |affinity, *mask| mask.* = try self.affinityMask(affinity);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.closed) return error.Closed;
        if (!self.canEverAcquire(masks, assignments, planned, potential))
            return error.RequestExceedsCapacity;
        while (true) {
            if (self.closed) return error.Closed;
            for (self.nodes, available) |node, *free| {
                free.* = node.free_blocks.items.len;
            }
            const blocked_job = self.planAssignments(masks, available, assignments, planned);
            if (blocked_job == null) break;
            if (try self.growEligible(masks[blocked_job.?], planned)) continue;
            if (self.in_use == 0) return error.RequestExceedsCapacity;
            self.condition.waitUncancelable(io, &self.mutex);
        }

        for (output, assignments) |*block, node_index| {
            const node = &self.nodes[node_index];
            block.* = node.free_blocks.pop().?;
            std.debug.assert(block.node_index == node_index);
            node.in_use += 1;
            node.high_water = @max(node.high_water, node.in_use);
        }
        self.next_node = (assignments[assignments.len - 1] + 1) % self.nodes.len;
        self.in_use += output.len;
        self.high_water = @max(self.high_water, self.in_use);
    }

    fn affinityMask(self: *const DmaBlockPool, affinity: Affinity) !u64 {
        const all_nodes = if (self.nodes.len == 64)
            std.math.maxInt(u64)
        else
            (@as(u64, 1) << @intCast(self.nodes.len)) - 1;
        if (affinity.eligible_nodes == 0) return all_nodes;
        if (affinity.eligible_nodes & ~all_nodes != 0) return error.InvalidAffinity;
        return affinity.eligible_nodes;
    }

    /// Finds an atomic assignment with an augmenting-path search. This is
    /// necessary even for a tiny request: greedily placing a replicated block
    /// can consume the only block eligible for a later strict-local block.
    fn planAssignments(
        self: *DmaBlockPool,
        masks: []const u64,
        limits: []const usize,
        assignments: []usize,
        counts: []usize,
    ) ?usize {
        @memset(assignments, unassigned_node);
        @memset(counts, 0);
        for (masks, 0..) |_, job_index| {
            var visited_nodes: u64 = 0;
            if (!self.assignJob(job_index, masks, limits, assignments, counts, &visited_nodes))
                return job_index;
        }
        return null;
    }

    fn assignJob(
        self: *DmaBlockPool,
        job_index: usize,
        masks: []const u64,
        limits: []const usize,
        assignments: []usize,
        counts: []usize,
        visited_nodes: *u64,
    ) bool {
        const mask = masks[job_index];
        var selected: ?usize = null;
        for (0..self.nodes.len) |offset| {
            const node_index = (self.next_node + offset) % self.nodes.len;
            const node_bit = @as(u64, 1) << @intCast(node_index);
            if (mask & node_bit == 0 or visited_nodes.* & node_bit != 0 or
                counts[node_index] == limits[node_index])
                continue;
            if (selected) |current| {
                const lhs_leased = self.nodes[node_index].in_use + counts[node_index];
                const rhs_leased = self.nodes[current].in_use + counts[current];
                const lhs_capacity = @max(@as(usize, 1), self.nodes[node_index].capacity);
                const rhs_capacity = @max(@as(usize, 1), self.nodes[current].capacity);
                if (@as(u128, lhs_leased) * rhs_capacity >=
                    @as(u128, rhs_leased) * lhs_capacity) continue;
            }
            selected = node_index;
        }
        if (selected) |node_index| {
            visited_nodes.* |= @as(u64, 1) << @intCast(node_index);
            assignments[job_index] = node_index;
            counts[node_index] += 1;
            return true;
        }

        // Every eligible node is full. Try to move one of its current jobs to
        // another eligible node before declaring this request blocked.
        for (0..self.nodes.len) |offset| {
            const node_index = (self.next_node + offset) % self.nodes.len;
            const node_bit = @as(u64, 1) << @intCast(node_index);
            if (mask & node_bit == 0 or visited_nodes.* & node_bit != 0) continue;
            visited_nodes.* |= node_bit;
            for (assignments, 0..) |assignment, other_job| {
                if (assignment != node_index) continue;
                assignments[other_job] = unassigned_node;
                counts[node_index] -= 1;
                if (self.assignJob(
                    other_job,
                    masks,
                    limits,
                    assignments,
                    counts,
                    visited_nodes,
                )) {
                    assignments[job_index] = node_index;
                    counts[node_index] += 1;
                    return true;
                }
                assignments[other_job] = node_index;
                counts[node_index] += 1;
            }
        }
        return false;
    }

    fn canEverAcquire(
        self: *DmaBlockPool,
        masks: []const u64,
        assignments: []usize,
        counts: []usize,
        potential: []usize,
    ) bool {
        const remaining_blocks = self.remainingBlockBudget();
        const reserved_growth = self.reservedGrowthBlocks();
        if (reserved_growth > remaining_blocks) return false;
        for (self.nodes, potential) |node, *capacity| {
            capacity.* = @max(node.capacity, node.reserve);
        }

        @memset(assignments, unassigned_node);
        @memset(counts, 0);
        var matched: usize = 0;
        for (masks, 0..) |_, job_index| {
            var visited_nodes: u64 = 0;
            if (self.assignJob(
                job_index,
                masks,
                potential,
                assignments,
                counts,
                &visited_nodes,
            )) matched += 1;
        }
        return masks.len - matched <= remaining_blocks - reserved_growth;
    }

    fn remainingBlockBudget(self: *const DmaBlockPool) usize {
        return (self.max_mapped_bytes -| self.mapped_bytes) / self.block_size;
    }

    fn reservedGrowthBlocks(self: *const DmaBlockPool) usize {
        var total: usize = 0;
        for (self.nodes) |node| total +|= node.reserve -| node.capacity;
        return total;
    }

    fn capacityBlocks(self: *const DmaBlockPool) usize {
        var blocks: usize = 0;
        for (self.nodes) |node| blocks += node.capacity;
        return blocks;
    }

    /// Requests of `blocks_per_request` blocks the pool holds without
    /// mapping anything: the retained (pre-grown) capacity of every node,
    /// or of the smallest node when every request must come from one node
    /// (`strict_affinity`), since a submission may put all of them there.
    pub fn retainedRequestWidth(
        self: *const DmaBlockPool,
        blocks_per_request: usize,
        strict_affinity: bool,
    ) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        if (!strict_affinity) return self.capacityBlocks() / blocks_per_request;
        var smallest: usize = std.math.maxInt(usize);
        for (self.nodes) |node| smallest = @min(smallest, node.capacity);
        return smallest / blocks_per_request;
    }

    /// Requests of `blocks_per_request` blocks that can be leased without
    /// mapping a slab and without eating into the DMA stage, which every node
    /// reserves for the devices attached to it. The subtraction has to stay
    /// node-wise: `Node.reserve` is that node's stage, so charging a
    /// machine-wide stage count against one node's capacity makes every node
    /// look emptier than it is. On eight MI300X that charged 64 stage
    /// requests against a node holding 64, left nothing, and pinned every
    /// adaptive load to width 1 (5.34 s against 1.42 s with one shared pool).
    pub fn growthFreeRequestWidth(
        self: *const DmaBlockPool,
        blocks_per_request: usize,
        strict_affinity: bool,
    ) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        if (!strict_affinity) {
            var total: usize = 0;
            for (self.nodes) |node| total +|= node.capacity -| node.reserve;
            return total / blocks_per_request;
        }
        var smallest: usize = std.math.maxInt(usize);
        for (self.nodes) |node| smallest = @min(smallest, node.capacity -| node.reserve);
        return smallest / blocks_per_request;
    }

    /// Returns the largest request width that the pool could support if each
    /// request consumes `blocks_per_request` blocks and admissions may draw
    /// from aggregate capacity. Arena tails count against the mapped-byte cap
    /// but do not contribute usable blocks.
    pub fn aggregatePotentialRequestWidth(
        self: *const DmaBlockPool,
        blocks_per_request: usize,
    ) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        return (self.capacityBlocks() + self.remainingBlockBudget()) / blocks_per_request;
    }

    /// Returns the smallest request width that any one strict-affinity node
    /// could eventually support. Growth needed to satisfy every node's reserve
    /// is protected first; the remaining growth budget may then be used by the
    /// node being evaluated.
    pub fn minimumStrictAffinityRequestWidth(
        self: *const DmaBlockPool,
        blocks_per_request: usize,
    ) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        const remaining_blocks = self.remainingBlockBudget();
        const reserved_growth = self.reservedGrowthBlocks();
        if (reserved_growth > remaining_blocks) return error.RequestExceedsCapacity;
        const shareable_growth = remaining_blocks - reserved_growth;
        var minimum_width: usize = std.math.maxInt(usize);
        for (self.nodes) |node| {
            const potential_blocks = @max(node.capacity, node.reserve) + shareable_growth;
            minimum_width = @min(minimum_width, potential_blocks / blocks_per_request);
        }
        return minimum_width;
    }

    fn growEligible(self: *DmaBlockPool, eligible: u64, planned: []const usize) !bool {
        const remaining_blocks = self.remainingBlockBudget();
        if (remaining_blocks == 0) return false;
        var selected: ?usize = null;
        for (0..self.nodes.len) |offset| {
            const node_index = (self.next_node + offset) % self.nodes.len;
            if (eligible & (@as(u64, 1) << @intCast(node_index)) == 0) continue;
            var other_deficit: usize = 0;
            for (self.nodes, 0..) |node, index| {
                if (index != node_index) other_deficit +|= node.reserve -| node.capacity;
            }
            if (remaining_blocks <= other_deficit) continue;
            if (selected) |current| {
                const lhs_leased = self.nodes[node_index].in_use + planned[node_index];
                const rhs_leased = self.nodes[current].in_use + planned[current];
                const lhs_capacity = @max(@as(usize, 1), self.nodes[node_index].capacity);
                const rhs_capacity = @max(@as(usize, 1), self.nodes[current].capacity);
                if (@as(u128, lhs_leased) * rhs_capacity >=
                    @as(u128, rhs_leased) * lhs_capacity) continue;
            }
            selected = node_index;
        }
        const node_index = selected orelse return false;
        var other_deficit: usize = 0;
        for (self.nodes, 0..) |node, index| {
            if (index == node_index) continue;
            other_deficit +|= node.reserve -| node.capacity;
        }
        if (remaining_blocks <= other_deficit) return false;
        const block_count = @min(self.slab_blocks, remaining_blocks - other_deficit);
        if (block_count == 0) return false;
        try self.allocateSlab(node_index, block_count);
        self.next_node = (node_index + 1) % self.nodes.len;
        return true;
    }

    pub fn releaseMany(self: *DmaBlockPool, io: std.Io, blocks: []const Block) void {
        if (blocks.len == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(blocks.len <= self.in_use);
        for (blocks) |block| {
            std.debug.assert(block.data.len == self.block_size and block.node_index < self.nodes.len);
            const node = &self.nodes[block.node_index];
            node.free_blocks.appendAssumeCapacity(block);
            std.debug.assert(node.in_use > 0);
            node.in_use -= 1;
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

    pub fn nodeCount(self: *const DmaBlockPool) usize {
        return self.nodes.len;
    }

    /// Returns a node-local snapshot. Callers report these statistics only
    /// after all leases have drained, just like the aggregate pool counters.
    pub fn nodeStats(self: *const DmaBlockPool, node_index: usize) NodeStats {
        const node = &self.nodes[node_index];
        return .{
            .retained_mapped_bytes = node.retained_mapped_bytes,
            .newly_mapped_bytes = node.newly_mapped_bytes,
            .leased_high_water_bytes = node.high_water * self.block_size,
            .unused_tail_bytes = node.unused_tail_bytes,
        };
    }

    fn allocateSlab(self: *DmaBlockPool, node_index: usize, block_count: usize) !void {
        const slab_len = try std.math.mul(usize, block_count, self.block_size);
        const mapped_before = self.workspace.retainedMappedBytes();
        if (mapped_before != self.mapped_bytes) return error.InvalidDmaWorkspace;
        const slab = try self.workspace.allocate(node_index, slab_len);
        const mapped_after = self.workspace.retainedMappedBytes();
        if (slab.len != slab_len or mapped_after < mapped_before or
            mapped_after - mapped_before != slab.len or mapped_after > self.max_mapped_bytes)
            return error.InvalidDmaWorkspace;
        self.mapped_bytes = mapped_after;
        self.newly_mapped_bytes +|= mapped_after - mapped_before;
        try self.attachArena(node_index, slab, .newly_mapped);
    }

    fn attachArena(
        self: *DmaBlockPool,
        node_index: usize,
        arena: []u8,
        origin: ArenaOrigin,
    ) !void {
        if (node_index >= self.nodes.len or arena.len == 0) return error.InvalidDmaWorkspace;
        const arena_start = @intFromPtr(arena.ptr);
        const arena_end = std.math.add(usize, arena_start, arena.len) catch
            return error.InvalidDmaWorkspace;
        for (self.nodes) |existing_node| {
            for (existing_node.arenas.items) |existing| {
                const existing_start = @intFromPtr(existing.ptr);
                const existing_end = std.math.add(usize, existing_start, existing.len) catch
                    return error.InvalidDmaWorkspace;
                if (arena_start < existing_end and existing_start < arena_end)
                    return error.InvalidDmaWorkspace;
            }
        }
        const node = &self.nodes[node_index];
        const block_count = arena.len / self.block_size;
        // Leased blocks are absent from `free_blocks`, so reserving relative to
        // its current length can leave too little space to return them after a
        // slab is attached under load. Keep storage sized for total capacity.
        try node.free_blocks.ensureTotalCapacity(self.allocator, node.capacity + block_count);
        try node.arenas.ensureUnusedCapacity(self.allocator, 1);
        node.arenas.appendAssumeCapacity(arena);
        for (0..block_count) |index| {
            node.free_blocks.appendAssumeCapacity(.{
                .data = arena[index * self.block_size ..][0..self.block_size],
                .node_index = node_index,
            });
        }
        node.capacity += block_count;
        const unused_tail_bytes = arena.len % self.block_size;
        switch (origin) {
            .retained => node.retained_mapped_bytes += arena.len,
            .newly_mapped => node.newly_mapped_bytes += arena.len,
        }
        node.unused_tail_bytes += unused_tail_bytes;
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

test "DmaWorkspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, 256);
            defer workspace.deinit();
            try workspace.acquire();
            defer workspace.release();
            _ = try workspace.allocate(0, 64);
            _ = try workspace.allocate(1, 128);
            try std.testing.expectEqual(@as(usize, 192), workspace.retainedMappedBytes());
            try std.testing.expectError(error.DmaMappedBudgetExceeded, workspace.allocate(0, 128));
            try std.testing.expectEqual(@as(usize, 192), workspace.retainedMappedBytes());
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

fn acquireDmaBlocksForTest(
    pool: *DmaBlockPool,
    io: std.Io,
    output: []DmaBlockPool.Block,
    affinities: []const DmaBlockPool.Affinity,
) !void {
    var scratch = try pool.acquireScratch(std.testing.allocator, output.len);
    defer scratch.deinit();
    try pool.acquireMany(io, output, affinities, &scratch);
}

test "DmaBlockPool acquires request blocks atomically" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 1, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 4 * 64, &.{0});
    defer pool.deinit();

    var first: [3]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(&pool, io, &first, &.{ .{}, .{}, .{} });
    try std.testing.expectEqual(@as(usize, 3 * 64), pool.highWaterBytes());
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.mappedBytes());
    try std.testing.expectEqualDeep(
        DmaBlockPool.NodeStats{
            .retained_mapped_bytes = 0,
            .newly_mapped_bytes = 4 * 64,
            .leased_high_water_bytes = 3 * 64,
            .unused_tail_bytes = 0,
        },
        pool.nodeStats(0),
    );
    var oversized: [5]DmaBlockPool.Block = undefined;
    try std.testing.expectError(
        error.RequestExceedsCapacity,
        acquireDmaBlocksForTest(&pool, io, &oversized, &.{ .{}, .{}, .{}, .{}, .{} }),
    );

    var started: std.Io.Event = .unset;
    var acquired: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *DmaBlockPool, io_: std.Io, started_: *std.Io.Event, acquired_: *std.Io.Event) void {
            var blocks: [2]DmaBlockPool.Block = undefined;
            started_.set(io_);
            acquireDmaBlocksForTest(pool_, io_, &blocks, &.{ .{}, .{} }) catch unreachable;
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
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 1, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 9 * 64, &.{0});
    defer pool.deinit();
    pool.slab_blocks = 1;

    var held: [9]DmaBlockPool.Block = undefined;
    const affinities: [held.len]DmaBlockPool.Affinity = @splat(.{});
    try acquireDmaBlocksForTest(&pool, io, &held, &affinities);

    try std.testing.expect(pool.nodes[0].free_blocks.capacity >= pool.nodes[0].capacity);
    pool.releaseMany(io, &held);
}

test "DmaBlockPool acquisition reuses matching scratch" {
    const io = std.testing.io;
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{});
    var workspace = try DmaWorkspace.initForTesting(std.testing.allocator, io, 1, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(
        failing.allocator(),
        &workspace,
        64,
        4 * 64,
        &.{0},
    );
    defer pool.deinit();
    var scratch = try pool.acquireScratch(std.testing.allocator, 3);
    defer scratch.deinit();

    var blocks: [3]DmaBlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks, &.{ .{}, .{}, .{} }, &scratch);
    pool.releaseMany(io, &blocks);

    failing.fail_index = failing.alloc_index;
    try pool.acquireMany(io, &blocks, &.{ .{}, .{}, .{} }, &scratch);
    try std.testing.expect(!failing.has_induced_failure);
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool matching scratch cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var scratch = try DmaBlockPool.AcquireScratch.init(allocator, 8, 2);
            defer scratch.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "DmaBlockPool close wakes blocked bulk acquisitions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 1, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 2 * 64, &.{0});
    defer pool.deinit();

    var held: [2]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(&pool, io, &held, &.{ .{}, .{} });
    var started: std.Io.Event = .unset;
    var result: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *DmaBlockPool, io_: std.Io, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            var block: [1]DmaBlockPool.Block = undefined;
            started_.set(io_);
            acquireDmaBlocksForTest(pool_, io_, &block, &.{.{}}) catch |err| {
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

test "DmaBlockPool lease returns a replicated block after out-of-order callbacks" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 1, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    var pool = try DmaBlockPool.init(allocator, &workspace, 64, 64, &.{0});
    defer pool.deinit();

    var blocks: [1]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(&pool, io, &blocks, &.{.{}});
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
    try acquireDmaBlocksForTest(&pool, io, &reacquired, &.{.{}});
    try std.testing.expectEqual(@intFromPtr(blocks[0].data.ptr), @intFromPtr(reacquired[0].data.ptr));
    pool.releaseMany(io, &reacquired);
}

test "DmaBlockPool preserves strict affinity when a replica is planned first" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 64);
    _ = try workspace.allocate(1, 64);

    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        2 * 64,
        &.{ 0, 0 },
    );
    defer pool.deinit();

    var blocks: [2]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(
        &pool,
        io,
        &blocks,
        &.{ .replicated(0b11), .node(0) },
    );
    try std.testing.expectEqual(@as(usize, 1), blocks[0].node_index);
    try std.testing.expectEqual(@as(usize, 0), blocks[1].node_index);
    try std.testing.expectEqual(@as(usize, 2 * 64), workspace.retainedMappedBytes());
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool balances cross-node replicas by leased capacity" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 4 * 64);
    _ = try workspace.allocate(1, 4 * 64);

    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        8 * 64,
        &.{ 0, 0 },
    );
    defer pool.deinit();

    var blocks: [4]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(
        &pool,
        io,
        &blocks,
        &.{ .replicated(0b11), .replicated(0b11), .replicated(0b11), .replicated(0b11) },
    );
    var per_node = [_]usize{ 0, 0 };
    for (blocks) |block| per_node[block.node_index] += 1;
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, &per_node);
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool reblocks retained arenas and grows only the required node" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 150);
    _ = try workspace.allocate(0, 70);
    _ = try workspace.allocate(1, 65);

    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        349,
        &.{ 3, 2 },
    );
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 285), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 29), pool.unusedTailBytes());

    var blocks: [5]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(
        &pool,
        io,
        &blocks,
        &.{ .node(0), .node(0), .node(0), .node(1), .node(1) },
    );
    for (blocks[0..3]) |block| {
        try std.testing.expectEqual(@as(usize, 0), block.node_index);
    }
    for (blocks[3..]) |block| {
        try std.testing.expectEqual(@as(usize, 1), block.node_index);
    }
    try std.testing.expectEqual(@as(usize, 4), workspace.pools[0].allocations.items.len + workspace.pools[1].allocations.items.len);
    try std.testing.expectEqual(@as(usize, 349), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 64), pool.newlyMappedBytes());
    try std.testing.expectEqual(@as(usize, 29), pool.unusedTailBytes());
    try std.testing.expectEqual(@as(usize, 2), pool.nodeCount());
    try std.testing.expectEqualDeep(
        DmaBlockPool.NodeStats{
            .retained_mapped_bytes = 220,
            .newly_mapped_bytes = 0,
            .leased_high_water_bytes = 3 * 64,
            .unused_tail_bytes = 28,
        },
        pool.nodeStats(0),
    );
    try std.testing.expectEqualDeep(
        DmaBlockPool.NodeStats{
            .retained_mapped_bytes = 65,
            .newly_mapped_bytes = 64,
            .leased_high_water_bytes = 2 * 64,
            .unused_tail_bytes = 1,
        },
        pool.nodeStats(1),
    );
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool potential request widths account for retained arena tails" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 127);
    _ = try workspace.allocate(1, 127);

    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        574,
        &.{ 2, 2 },
    );
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 126), pool.unusedTailBytes());
    try std.testing.expectEqual(@as(usize, 3), try pool.aggregatePotentialRequestWidth(2));
    try std.testing.expectEqual(@as(usize, 2), try pool.minimumStrictAffinityRequestWidth(2));
    try std.testing.expectEqual(@as(usize, 0), try pool.aggregatePotentialRequestWidth(8));
    try std.testing.expectEqual(@as(usize, 0), try pool.minimumStrictAffinityRequestWidth(6));
    try std.testing.expectError(
        error.InvalidRequestBlockCount,
        pool.aggregatePotentialRequestWidth(0),
    );
    try std.testing.expectError(
        error.InvalidRequestBlockCount,
        pool.minimumStrictAffinityRequestWidth(0),
    );
}

test "DmaBlockPool growth-free width subtracts each node's own DMA stage" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 64 * 64);
    _ = try workspace.allocate(1, 64 * 64);

    // Eight MI300X: four devices per node at eight in-flight blocks each, so
    // every node retains 64 blocks and reserves 32 of them for its own stage.
    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        2 * 64 * 64,
        &.{ 32, 32 },
    );
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 64), try pool.retainedRequestWidth(1, true));
    // The machine-wide stage is 64 requests. Charging all of it against one
    // node left nothing and pinned the read width to 1; only that node's own
    // 32 may be subtracted.
    try std.testing.expectEqual(@as(usize, 32), try pool.growthFreeRequestWidth(1, true));
    try std.testing.expectEqual(@as(usize, 64), try pool.growthFreeRequestWidth(1, false));
    try std.testing.expectEqual(@as(usize, 16), try pool.growthFreeRequestWidth(2, true));
    try std.testing.expectError(
        error.InvalidRequestBlockCount,
        pool.growthFreeRequestWidth(0, true),
    );
}

test "DmaBlockPool growth-free width saturates when a reserve covers the node" {
    const allocator = std.testing.allocator;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 8 * 64);
    _ = try workspace.allocate(1, 2 * 64);

    // Node 1 has not grown to its reserve yet, which the pool allows only
    // while the mapped-byte budget can still cover the deficit.
    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        16 * 64,
        &.{ 4, 5 },
    );
    defer pool.deinit();

    // Node 1 holds two blocks against a reserve of five: no headroom, and no
    // underflow. The strict answer is the emptiest node's.
    try std.testing.expectEqual(@as(usize, 0), try pool.growthFreeRequestWidth(1, true));
    try std.testing.expectEqual(@as(usize, 4), try pool.growthFreeRequestWidth(1, false));
}

test "DmaBlockPool rejects impossible and invalid affinities without leasing" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var workspace = try DmaWorkspace.initForTesting(allocator, std.testing.io, 2, std.math.maxInt(usize));
    defer workspace.deinit();
    try workspace.acquire();
    defer workspace.release();
    _ = try workspace.allocate(0, 64);
    _ = try workspace.allocate(1, 64);

    var pool = try DmaBlockPool.init(
        allocator,
        &workspace,
        64,
        2 * 64,
        &.{ 0, 0 },
    );
    defer pool.deinit();

    var impossible: [2]DmaBlockPool.Block = undefined;
    try std.testing.expectError(
        error.RequestExceedsCapacity,
        acquireDmaBlocksForTest(&pool, io, &impossible, &.{ .node(0), .node(0) }),
    );
    var invalid: [1]DmaBlockPool.Block = undefined;
    try std.testing.expectError(
        error.InvalidAffinity,
        acquireDmaBlocksForTest(&pool, io, &invalid, &.{.{ .eligible_nodes = 0b100 }}),
    );
    var local: [1]DmaBlockPool.Block = undefined;
    try acquireDmaBlocksForTest(&pool, io, &local, &.{.node(0)});
    try std.testing.expectEqual(@as(usize, 0), local[0].node_index);
    pool.releaseMany(io, &local);
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
