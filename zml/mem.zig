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

/// A per-load view of fixed-size blocks carved from owned DMA arenas. The
/// provider retains arena ownership; this view owns only free-list metadata.
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

    /// Type-erased bridge to the resource-owned arena workspace.
    pub const ArenaProvider = struct {
        context: *anyopaque,
        node_count: usize,
        arenaCountFn: *const fn (*anyopaque, usize) usize,
        arenaFn: *const fn (*anyopaque, usize, usize) []u8,
        allocateFn: *const fn (*anyopaque, usize, usize) anyerror![]u8,
        mappedBytesFn: *const fn (*anyopaque) usize,

        fn arenaCount(self: ArenaProvider, node_index: usize) usize {
            return self.arenaCountFn(self.context, node_index);
        }

        fn arena(self: ArenaProvider, node_index: usize, arena_index: usize) []u8 {
            return self.arenaFn(self.context, node_index, arena_index);
        }

        fn allocate(self: ArenaProvider, node_index: usize, len: usize) ![]u8 {
            return self.allocateFn(self.context, node_index, len);
        }

        fn mappedBytes(self: ArenaProvider) usize {
            return self.mappedBytesFn(self.context);
        }
    };

    pub const Lease = struct {
        pool: *DmaBlockPool,
        io: std.Io,
        data: []u8,
        remaining: std.atomic.Value(usize),

        pub fn init(pool: *DmaBlockPool, io: std.Io, data: []u8, references: usize) Lease {
            std.debug.assert(references > 0);
            return .{ .pool = pool, .io = io, .data = data, .remaining = .init(references) };
        }

        pub fn complete(self: *Lease) void {
            const previous = self.remaining.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous == 1) self.pool.release(self.io, self.data);
        }

        pub fn isComplete(self: *const Lease) bool {
            return self.remaining.load(.acquire) == 0;
        }
    };

    const SlabSource = union(enum) {
        dma: DmaAllocator,
        testing: std.mem.Allocator,

        fn allocator(self: *const SlabSource) std.mem.Allocator {
            return switch (self.*) {
                .dma => |*dma| dma.allocator(),
                .testing => |allocator_| allocator_,
            };
        }
    };

    const Node = struct {
        free_blocks: std.ArrayListUnmanaged([]u8) = .empty,
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
    slab_source: ?SlabSource,
    provider: ?ArenaProvider,
    nodes: []Node,
    block_size: usize,
    max_mapped_bytes: usize,
    mapped_bytes: usize,
    unused_tail_bytes: usize = 0,
    newly_mapped_bytes: usize = 0,
    slab_blocks: usize,
    owned_slabs: std.ArrayListUnmanaged([]u8) = .empty,
    in_use: usize = 0,
    high_water: usize = 0,
    next_node: usize = 0,
    closed: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        block_size: usize,
        max_bytes: usize,
    ) !DmaBlockPool {
        if (platform.devices.len == 0) return error.RequestExceedsCapacity;
        return initWithSlabSource(
            allocator,
            .{ .dma = .init(allocator, &platform.devices[0]) },
            block_size,
            max_bytes,
        );
    }

    fn initForTest(
        allocator: std.mem.Allocator,
        slab_allocator: std.mem.Allocator,
        block_size: usize,
        max_bytes: usize,
    ) !DmaBlockPool {
        return initWithSlabSource(allocator, .{ .testing = slab_allocator }, block_size, max_bytes);
    }

    fn initWithSlabSource(
        allocator: std.mem.Allocator,
        slab_source: SlabSource,
        block_size: usize,
        max_bytes: usize,
    ) !DmaBlockPool {
        if (block_size == 0 or max_bytes < block_size) return error.RequestExceedsCapacity;
        const nodes = try allocator.alloc(Node, 1);
        nodes[0] = .{};
        const max_blocks = max_bytes / block_size;
        var self: DmaBlockPool = .{
            .allocator = allocator,
            .slab_source = slab_source,
            .provider = null,
            .nodes = nodes,
            .block_size = block_size,
            .max_mapped_bytes = max_bytes,
            .mapped_bytes = 0,
            .slab_blocks = @max(@as(usize, 1), default_slab_size / block_size),
        };
        errdefer self.deinit();
        try self.nodes[0].free_blocks.ensureTotalCapacityPrecise(allocator, max_blocks);
        try self.nodes[0].arenas.ensureTotalCapacityPrecise(
            allocator,
            std.math.divCeil(usize, max_blocks, @max(@as(usize, 1), default_slab_size / block_size)) catch unreachable,
        );
        return self;
    }

    /// Builds a fresh free-list view from every retained arena. Arena tails
    /// smaller than one selected block remain mapped and are reported unused.
    pub fn initFromProvider(
        allocator: std.mem.Allocator,
        provider: ArenaProvider,
        block_size: usize,
        max_mapped_bytes: usize,
        reserves: []const usize,
    ) !DmaBlockPool {
        const mapped_bytes = provider.mappedBytes();
        if (block_size == 0 or provider.node_count == 0 or provider.node_count > 64 or
            reserves.len != provider.node_count or mapped_bytes > max_mapped_bytes)
            return error.RequestExceedsCapacity;
        const nodes = try allocator.alloc(Node, provider.node_count);
        for (nodes, reserves) |*node, reserve| node.* = .{ .reserve = reserve };
        var self: DmaBlockPool = .{
            .allocator = allocator,
            .slab_source = null,
            .provider = provider,
            .nodes = nodes,
            .block_size = block_size,
            .max_mapped_bytes = max_mapped_bytes,
            .mapped_bytes = mapped_bytes,
            .slab_blocks = @max(@as(usize, 1), default_slab_size / block_size),
        };
        errdefer self.deinit();
        var enumerated_bytes: usize = 0;
        for (0..provider.node_count) |node_index| {
            for (0..provider.arenaCount(node_index)) |arena_index| {
                const arena = provider.arena(node_index, arena_index);
                enumerated_bytes = std.math.add(usize, enumerated_bytes, arena.len) catch
                    return error.InvalidArenaProvider;
                try self.attachArena(node_index, arena, .retained);
            }
        }
        if (enumerated_bytes != mapped_bytes) return error.InvalidArenaProvider;
        if (self.reservedGrowthBlocks() > self.remainingBlockBudget())
            return error.RequestExceedsCapacity;
        return self;
    }

    pub fn deinit(self: *DmaBlockPool) void {
        std.debug.assert(self.in_use == 0);
        if (self.slab_source) |*source| {
            const slab_allocator = source.allocator();
            for (self.owned_slabs.items) |slab| slab_allocator.rawFree(slab, .of(u8), @returnAddress());
        }
        self.owned_slabs.deinit(self.allocator);
        for (self.nodes) |*node| {
            std.debug.assert(node.in_use == 0 and node.free_blocks.items.len == node.capacity);
            node.free_blocks.deinit(self.allocator);
            node.arenas.deinit(self.allocator);
        }
        self.allocator.free(self.nodes);
        self.* = undefined;
    }

    pub fn acquireMany(
        self: *DmaBlockPool,
        io: std.Io,
        output: [][]u8,
        affinities: []const Affinity,
    ) Error!u64 {
        if (output.len != affinities.len) return error.InvalidAffinity;
        if (output.len == 0) return 0;
        const assignments = try self.allocator.alloc(usize, output.len);
        defer self.allocator.free(assignments);
        const masks = try self.allocator.alloc(u64, output.len);
        defer self.allocator.free(masks);
        for (affinities, masks) |affinity, *mask| mask.* = try self.affinityMask(affinity);
        const available = try self.allocator.alloc(usize, self.nodes.len);
        defer self.allocator.free(available);
        const planned = try self.allocator.alloc(usize, self.nodes.len);
        defer self.allocator.free(planned);
        const potential = try self.allocator.alloc(usize, self.nodes.len);
        defer self.allocator.free(potential);

        const started: std.Io.Timestamp = .now(io, .awake);
        var waited = false;
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
            waited = true;
            self.condition.waitUncancelable(io, &self.mutex);
        }

        for (output, assignments) |*block, node_index| {
            const node = &self.nodes[node_index];
            block.* = node.free_blocks.pop().?;
            node.in_use += 1;
            node.high_water = @max(node.high_water, node.in_use);
        }
        self.next_node = (assignments[assignments.len - 1] + 1) % self.nodes.len;
        self.in_use += output.len;
        self.high_water = @max(self.high_water, self.in_use);
        if (!waited) return 0;
        return @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));
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

    /// Returns the largest request width that the pool could support if each
    /// request consumes `blocks_per_request` blocks and admissions may draw
    /// from aggregate capacity. Call this after refreshing provider arenas and
    /// before the first lease. Arena tails count against the mapped-byte cap
    /// but do not contribute usable blocks.
    pub fn aggregatePotentialRequestWidth(
        self: *const DmaBlockPool,
        blocks_per_request: usize,
    ) !usize {
        if (blocks_per_request == 0) return error.InvalidRequestBlockCount;
        var usable_blocks: usize = 0;
        for (self.nodes) |node| usable_blocks += node.capacity;
        return (usable_blocks + self.remainingBlockBudget()) / blocks_per_request;
    }

    /// Returns the smallest request width that any one strict-affinity node
    /// could eventually support. Growth needed to satisfy every node's reserve
    /// is protected first; the remaining growth budget may then be used by the
    /// node being evaluated. Call this after refreshing provider arenas and
    /// before the first lease.
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

    pub fn releaseMany(self: *DmaBlockPool, io: std.Io, blocks: []const []u8) void {
        if (blocks.len == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(blocks.len <= self.in_use);
        for (blocks) |block| {
            const node_index = self.nodeForBlock(block) orelse unreachable;
            const node = &self.nodes[node_index];
            node.free_blocks.appendAssumeCapacity(block);
            std.debug.assert(node.in_use > 0);
            node.in_use -= 1;
        }
        self.in_use -= blocks.len;
        self.condition.broadcast(io);
    }

    pub fn release(self: *DmaBlockPool, io: std.Io, block: []u8) void {
        self.releaseMany(io, &.{block});
    }

    fn nodeForBlock(self: *const DmaBlockPool, block: []const u8) ?usize {
        if (block.len != self.block_size) return null;
        const address = @intFromPtr(block.ptr);
        for (self.nodes, 0..) |node, node_index| {
            for (node.arenas.items) |arena| {
                const start = @intFromPtr(arena.ptr);
                if (address < start) continue;
                const offset = address - start;
                const usable_len = arena.len - arena.len % self.block_size;
                if (usable_len < self.block_size) continue;
                if (offset % self.block_size == 0 and offset <= usable_len -| self.block_size)
                    return node_index;
            }
        }
        return null;
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

    /// Discovers arenas appended to a provider after pool initialization and
    /// before the first lease. The provider must remain quiescent for the
    /// duration of this call. Existing arenas must remain an unchanged prefix
    /// of each provider node's append-only arena list.
    pub fn refreshProviderArenas(self: *DmaBlockPool, io: std.Io) !void {
        const provider = self.provider orelse return error.NotProviderBacked;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.closed) return error.Closed;
        if (self.in_use != 0 or self.high_water != 0) return error.DmaBlockPoolAlreadyUsed;

        const SnapshotArena = struct {
            node_index: usize,
            arena_index: usize,
            data: []u8,
            appended: bool,
        };

        const attached_counts = try self.allocator.alloc(usize, self.nodes.len);
        defer self.allocator.free(attached_counts);
        const future_capacities = try self.allocator.alloc(usize, self.nodes.len);
        defer self.allocator.free(future_capacities);
        var arena_count: usize = 0;
        for (self.nodes, attached_counts, future_capacities, 0..) |node, *attached, *capacity, node_index| {
            attached.* = node.arenas.items.len;
            capacity.* = node.capacity;
            const provider_count = provider.arenaCount(node_index);
            if (provider_count < attached.*) return error.InvalidArenaProvider;
            arena_count = std.math.add(usize, arena_count, provider_count) catch
                return error.InvalidArenaProvider;
        }

        const snapshot = try self.allocator.alloc(SnapshotArena, arena_count);
        defer self.allocator.free(snapshot);
        var snapshot_index: usize = 0;
        var enumerated_bytes: usize = 0;
        var attached_bytes: usize = 0;
        for (self.nodes, attached_counts, future_capacities, 0..) |node, attached_count, *capacity, node_index| {
            for (0..provider.arenaCount(node_index)) |arena_index| {
                const data = provider.arena(node_index, arena_index);
                if (data.len == 0) return error.InvalidArenaProvider;
                _ = std.math.add(usize, @intFromPtr(data.ptr), data.len) catch
                    return error.InvalidArenaProvider;
                enumerated_bytes = std.math.add(usize, enumerated_bytes, data.len) catch
                    return error.InvalidArenaProvider;
                const appended = arena_index >= attached_count;
                if (!appended) {
                    const existing = node.arenas.items[arena_index];
                    if (existing.ptr != data.ptr or existing.len != data.len)
                        return error.InvalidArenaProvider;
                    attached_bytes = std.math.add(usize, attached_bytes, data.len) catch
                        return error.InvalidArenaProvider;
                } else {
                    capacity.* = std.math.add(usize, capacity.*, data.len / self.block_size) catch
                        return error.InvalidArenaProvider;
                }
                snapshot[snapshot_index] = .{
                    .node_index = node_index,
                    .arena_index = arena_index,
                    .data = data,
                    .appended = appended,
                };
                snapshot_index += 1;
            }
        }

        // Validate the whole snapshot before changing any pool metadata, so a
        // reordered or overlapping append cannot be partially adopted.
        for (snapshot, 0..) |candidate, index| {
            const candidate_start = @intFromPtr(candidate.data.ptr);
            const candidate_end = candidate_start + candidate.data.len;
            for (snapshot[0..index]) |existing| {
                const existing_start = @intFromPtr(existing.data.ptr);
                const existing_end = existing_start + existing.data.len;
                if (candidate_start < existing_end and existing_start < candidate_end)
                    return error.InvalidArenaProvider;
            }
        }

        const provider_mapped_bytes = provider.mappedBytes();
        if (attached_bytes != self.mapped_bytes or
            enumerated_bytes != provider_mapped_bytes or
            provider_mapped_bytes < self.mapped_bytes)
            return error.InvalidArenaProvider;
        if (provider_mapped_bytes > self.max_mapped_bytes)
            return error.RequestExceedsCapacity;
        const future_remaining_blocks =
            (self.max_mapped_bytes - provider_mapped_bytes) / self.block_size;
        var future_reserved_growth: usize = 0;
        for (self.nodes, future_capacities) |node, capacity| {
            future_reserved_growth +|= node.reserve -| capacity;
        }
        if (future_reserved_growth > future_remaining_blocks)
            return error.RequestExceedsCapacity;

        for (snapshot) |candidate| {
            if (!candidate.appended) continue;
            const node = &self.nodes[candidate.node_index];
            if (node.arenas.items.len != candidate.arena_index)
                return error.InvalidArenaProvider;
            try self.attachArena(candidate.node_index, candidate.data, .newly_mapped);
            self.mapped_bytes = try std.math.add(usize, self.mapped_bytes, candidate.data.len);
            self.newly_mapped_bytes = try std.math.add(
                usize,
                self.newly_mapped_bytes,
                candidate.data.len,
            );
        }
        std.debug.assert(self.mapped_bytes == provider_mapped_bytes);
    }

    fn allocateSlab(self: *DmaBlockPool, node_index: usize, block_count: usize) !void {
        const slab_len = try std.math.mul(usize, block_count, self.block_size);
        const slab = if (self.provider) |provider| blk: {
            const mapped_before = provider.mappedBytes();
            if (mapped_before != self.mapped_bytes) return error.InvalidArenaProvider;
            const allocated = try provider.allocate(node_index, slab_len);
            const mapped_after = provider.mappedBytes();
            if (allocated.len != slab_len or mapped_after < mapped_before or
                mapped_after - mapped_before != allocated.len or mapped_after > self.max_mapped_bytes)
                return error.InvalidArenaProvider;
            self.mapped_bytes = mapped_after;
            self.newly_mapped_bytes +|= mapped_after - mapped_before;
            break :blk allocated;
        } else blk: {
            const slab_allocator = self.slab_source.?.allocator();
            const slab_ptr = slab_allocator.rawAlloc(slab_len, .of(u8), @returnAddress()) orelse
                return error.OutOfMemory;
            const owned = slab_ptr[0..slab_len];
            errdefer slab_allocator.rawFree(owned, .of(u8), @returnAddress());
            try self.owned_slabs.append(self.allocator, owned);
            break :blk owned;
        };
        if (self.provider == null) {
            self.mapped_bytes = try std.math.add(usize, self.mapped_bytes, slab.len);
            self.newly_mapped_bytes = try std.math.add(usize, self.newly_mapped_bytes, slab.len);
        }
        try self.attachArena(node_index, slab, .newly_mapped);
    }

    fn attachArena(
        self: *DmaBlockPool,
        node_index: usize,
        arena: []u8,
        origin: ArenaOrigin,
    ) !void {
        if (node_index >= self.nodes.len or arena.len == 0) return error.InvalidArenaProvider;
        const arena_start = @intFromPtr(arena.ptr);
        const arena_end = std.math.add(usize, arena_start, arena.len) catch
            return error.InvalidArenaProvider;
        for (self.nodes) |existing_node| {
            for (existing_node.arenas.items) |existing| {
                const existing_start = @intFromPtr(existing.ptr);
                const existing_end = std.math.add(usize, existing_start, existing.len) catch
                    return error.InvalidArenaProvider;
                if (arena_start < existing_end and existing_start < arena_end)
                    return error.InvalidArenaProvider;
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
            node.free_blocks.appendAssumeCapacity(
                arena[index * self.block_size ..][0..self.block_size],
            );
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

const TestDmaArenaProvider = struct {
    allocator: std.mem.Allocator,
    node_count: usize,
    nodes: [2]std.ArrayListUnmanaged([]u8) = .{ .empty, .empty },
    growth_allocations: usize = 0,

    fn init(allocator: std.mem.Allocator, node_count: usize) TestDmaArenaProvider {
        std.debug.assert(node_count > 0 and node_count <= 2);
        return .{ .allocator = allocator, .node_count = node_count };
    }

    fn deinit(self: *TestDmaArenaProvider) void {
        for (self.nodes[0..self.node_count]) |*node| {
            for (node.items) |allocation| self.allocator.free(allocation);
            node.deinit(self.allocator);
        }
        self.* = undefined;
    }

    fn addArena(self: *TestDmaArenaProvider, node_index: usize, len: usize) ![]u8 {
        const allocation = try self.allocator.alloc(u8, len);
        errdefer self.allocator.free(allocation);
        try self.nodes[node_index].append(self.allocator, allocation);
        return allocation;
    }

    fn provider(self: *TestDmaArenaProvider) DmaBlockPool.ArenaProvider {
        return .{
            .context = self,
            .node_count = self.node_count,
            .arenaCountFn = arenaCount,
            .arenaFn = arenaAt,
            .allocateFn = allocate,
            .mappedBytesFn = mappedBytes,
        };
    }

    fn arenaCount(context: *anyopaque, node_index: usize) usize {
        const self: *TestDmaArenaProvider = @ptrCast(@alignCast(context));
        return self.nodes[node_index].items.len;
    }

    fn arenaAt(context: *anyopaque, node_index: usize, arena_index: usize) []u8 {
        const self: *TestDmaArenaProvider = @ptrCast(@alignCast(context));
        return self.nodes[node_index].items[arena_index];
    }

    fn allocate(context: *anyopaque, node_index: usize, len: usize) ![]u8 {
        const self: *TestDmaArenaProvider = @ptrCast(@alignCast(context));
        const allocation = try self.addArena(node_index, len);
        self.growth_allocations += 1;
        return allocation;
    }

    fn mappedBytes(context: *anyopaque) usize {
        const self: *TestDmaArenaProvider = @ptrCast(@alignCast(context));
        var total: usize = 0;
        for (self.nodes[0..self.node_count]) |node| {
            for (node.items) |allocation| total += allocation.len;
        }
        return total;
    }
};

test "DmaBlockPool acquires request blocks atomically" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = try DmaBlockPool.initForTest(allocator, allocator, 64, 4 * 64);
    defer pool.deinit();

    var first: [3][]u8 = undefined;
    _ = try pool.acquireMany(io, &first, &.{ .{}, .{}, .{} });
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
    var oversized: [5][]u8 = undefined;
    try std.testing.expectError(
        error.RequestExceedsCapacity,
        pool.acquireMany(io, &oversized, &.{ .{}, .{}, .{}, .{}, .{} }),
    );

    var started: std.Io.Event = .unset;
    var acquired: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *DmaBlockPool, io_: std.Io, started_: *std.Io.Event, acquired_: *std.Io.Event) void {
            var blocks: [2][]u8 = undefined;
            started_.set(io_);
            _ = pool_.acquireMany(io_, &blocks, &.{ .{}, .{} }) catch unreachable;
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
    var pool = try DmaBlockPool.initForTest(allocator, allocator, 64, 9 * 64);
    defer pool.deinit();
    pool.slab_blocks = 1;

    var held: [9][]u8 = undefined;
    const affinities: [held.len]DmaBlockPool.Affinity = @splat(.{});
    _ = try pool.acquireMany(io, &held, &affinities);

    try std.testing.expect(pool.nodes[0].free_blocks.capacity >= pool.nodes[0].capacity);
    pool.releaseMany(io, &held);
}

test "DmaBlockPool close wakes blocked bulk acquisitions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = try DmaBlockPool.initForTest(allocator, allocator, 64, 2 * 64);
    defer pool.deinit();

    var held: [2][]u8 = undefined;
    _ = try pool.acquireMany(io, &held, &.{ .{}, .{} });
    var started: std.Io.Event = .unset;
    var result: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *DmaBlockPool, io_: std.Io, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            var block: [1][]u8 = undefined;
            started_.set(io_);
            _ = pool_.acquireMany(io_, &block, &.{.{}}) catch |err| {
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
    var pool = try DmaBlockPool.initForTest(allocator, allocator, 64, 64);
    defer pool.deinit();

    var blocks: [1][]u8 = undefined;
    _ = try pool.acquireMany(io, &blocks, &.{.{}});
    var lease: DmaBlockPool.Lease = .init(&pool, io, blocks[0], 4);
    var group: std.Io.Group = .init;
    for ([_]i64{ 4, 1, 3, 2 }) |delay_ms| {
        try group.concurrent(io, struct {
            fn run(lease_: *DmaBlockPool.Lease, io_: std.Io, delay_ms_: i64) void {
                io_.sleep(.fromMilliseconds(delay_ms_), .awake) catch unreachable;
                lease_.complete();
            }
        }.run, .{ &lease, io, delay_ms });
    }
    try group.await(io);
    try std.testing.expect(lease.isComplete());

    var reacquired: [1][]u8 = undefined;
    _ = try pool.acquireMany(io, &reacquired, &.{.{}});
    try std.testing.expectEqual(@intFromPtr(blocks[0].ptr), @intFromPtr(reacquired[0].ptr));
    pool.releaseMany(io, &reacquired);
}

test "DmaBlockPool preserves strict affinity when a replica is planned first" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var provider: TestDmaArenaProvider = .init(allocator, 2);
    defer provider.deinit();
    _ = try provider.addArena(0, 64);
    _ = try provider.addArena(1, 64);

    var pool = try DmaBlockPool.initFromProvider(
        allocator,
        provider.provider(),
        64,
        2 * 64,
        &.{ 0, 0 },
    );
    defer pool.deinit();

    var blocks: [2][]u8 = undefined;
    _ = try pool.acquireMany(
        io,
        &blocks,
        &.{ .replicated(0b11), .node(0) },
    );
    try std.testing.expectEqual(@as(?usize, 1), pool.nodeForBlock(blocks[0]));
    try std.testing.expectEqual(@as(?usize, 0), pool.nodeForBlock(blocks[1]));
    try std.testing.expectEqual(@as(usize, 0), provider.growth_allocations);
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool balances cross-node replicas by leased capacity" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var provider: TestDmaArenaProvider = .init(allocator, 2);
    defer provider.deinit();
    _ = try provider.addArena(0, 4 * 64);
    _ = try provider.addArena(1, 4 * 64);

    var pool = try DmaBlockPool.initFromProvider(
        allocator,
        provider.provider(),
        64,
        8 * 64,
        &.{ 0, 0 },
    );
    defer pool.deinit();

    var blocks: [4][]u8 = undefined;
    _ = try pool.acquireMany(
        io,
        &blocks,
        &.{ .replicated(0b11), .replicated(0b11), .replicated(0b11), .replicated(0b11) },
    );
    var per_node = [_]usize{ 0, 0 };
    for (blocks) |block| per_node[pool.nodeForBlock(block).?] += 1;
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, &per_node);
    pool.releaseMany(io, &blocks);
}

test "DmaBlockPool reblocks retained arenas and grows only the required node" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var provider: TestDmaArenaProvider = .init(allocator, 2);
    defer provider.deinit();
    _ = try provider.addArena(0, 150);
    _ = try provider.addArena(0, 70);
    _ = try provider.addArena(1, 65);

    var pool = try DmaBlockPool.initFromProvider(
        allocator,
        provider.provider(),
        64,
        349,
        &.{ 3, 2 },
    );
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 285), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 29), pool.unusedTailBytes());

    var blocks: [5][]u8 = undefined;
    _ = try pool.acquireMany(
        io,
        &blocks,
        &.{ .node(0), .node(0), .node(0), .node(1), .node(1) },
    );
    for (blocks[0..3]) |block| {
        try std.testing.expectEqual(@as(?usize, 0), pool.nodeForBlock(block));
    }
    for (blocks[3..]) |block| {
        try std.testing.expectEqual(@as(?usize, 1), pool.nodeForBlock(block));
    }
    try std.testing.expectEqual(@as(usize, 1), provider.growth_allocations);
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

test "DmaBlockPool refreshes append-only provider arenas before use" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var provider: TestDmaArenaProvider = .init(allocator, 2);
    defer provider.deinit();
    _ = try provider.addArena(0, 64);
    _ = try provider.addArena(0, 64);
    _ = try provider.addArena(1, 64);

    var pool = try DmaBlockPool.initFromProvider(
        allocator,
        provider.provider(),
        64,
        390,
        &.{ 3, 3 },
    );
    defer pool.deinit();

    std.mem.swap([]u8, &provider.nodes[0].items[0], &provider.nodes[0].items[1]);
    try std.testing.expectError(error.InvalidArenaProvider, pool.refreshProviderArenas(io));
    std.mem.swap([]u8, &provider.nodes[0].items[0], &provider.nodes[0].items[1]);

    _ = try provider.addArena(0, 70);
    _ = try provider.addArena(1, 128);
    try pool.refreshProviderArenas(io);
    try std.testing.expectEqual(@as(usize, 390), pool.mappedBytes());
    try std.testing.expectEqual(@as(usize, 198), pool.newlyMappedBytes());
    try std.testing.expectEqual(@as(usize, 6), pool.unusedTailBytes());
    try std.testing.expectEqualDeep(
        DmaBlockPool.NodeStats{
            .retained_mapped_bytes = 128,
            .newly_mapped_bytes = 70,
            .leased_high_water_bytes = 0,
            .unused_tail_bytes = 6,
        },
        pool.nodeStats(0),
    );
    try std.testing.expectEqualDeep(
        DmaBlockPool.NodeStats{
            .retained_mapped_bytes = 64,
            .newly_mapped_bytes = 128,
            .leased_high_water_bytes = 0,
            .unused_tail_bytes = 0,
        },
        pool.nodeStats(1),
    );

    var blocks: [6][]u8 = undefined;
    _ = try pool.acquireMany(
        io,
        &blocks,
        &.{ .node(0), .node(0), .node(0), .node(1), .node(1), .node(1) },
    );
    try std.testing.expectEqual(@as(usize, 0), provider.growth_allocations);
    pool.releaseMany(io, &blocks);
    try std.testing.expectError(error.DmaBlockPoolAlreadyUsed, pool.refreshProviderArenas(io));
}

test "DmaBlockPool potential request widths account for refreshed arena tails" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var provider: TestDmaArenaProvider = .init(allocator, 2);
    defer provider.deinit();
    _ = try provider.addArena(0, 64);
    _ = try provider.addArena(1, 64);

    var pool = try DmaBlockPool.initFromProvider(
        allocator,
        provider.provider(),
        64,
        574,
        &.{ 2, 2 },
    );
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 4), try pool.aggregatePotentialRequestWidth(2));
    try std.testing.expectEqual(@as(usize, 3), try pool.minimumStrictAffinityRequestWidth(2));

    // These arenas consume almost two more blocks of mapped budget without
    // producing a usable block on either node.
    _ = try provider.addArena(0, 63);
    _ = try provider.addArena(1, 63);
    try pool.refreshProviderArenas(io);
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

test "DmaBlockPool rejects impossible and invalid affinities without leasing" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var provider: TestDmaArenaProvider = .init(allocator, 2);
    defer provider.deinit();
    _ = try provider.addArena(0, 64);
    _ = try provider.addArena(1, 64);

    var pool = try DmaBlockPool.initFromProvider(
        allocator,
        provider.provider(),
        64,
        2 * 64,
        &.{ 0, 0 },
    );
    defer pool.deinit();

    var impossible: [2][]u8 = undefined;
    try std.testing.expectError(
        error.RequestExceedsCapacity,
        pool.acquireMany(io, &impossible, &.{ .node(0), .node(0) }),
    );
    var invalid: [1][]u8 = undefined;
    try std.testing.expectError(
        error.InvalidAffinity,
        pool.acquireMany(io, &invalid, &.{.{ .eligible_nodes = 0b100 }}),
    );
    var local: [1][]u8 = undefined;
    _ = try pool.acquireMany(io, &local, &.{.node(0)});
    try std.testing.expectEqual(@as(?usize, 0), pool.nodeForBlock(local[0]));
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
