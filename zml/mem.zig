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
            .cuda, .oneapi => .{ .dmam = .init(parent, device.platform) },
            .tpu => .{ .uib = .init(device.memory(.host_pinned).?) },
            .rocm, .cpu, .neuron, .metal => .{ .passthrough = parent },
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

/// Host allocator for CUDA DMA mappings. Linux allocations request transparent
/// huge-page backing, but remain valid ordinary-page mappings when unavailable.
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

pub const DynamicBufferPool = struct {
    const Node = struct { next: ?*Node };
    const alignment: std.mem.Alignment = .of(Node);

    pub const Acquisition = struct {
        buffer: []u8,
        wait_ns: u64,
    };

    block_size: usize,
    max_blocks: usize,
    limit: std.atomic.Value(usize),

    free_stack: std.atomic.Value(?*Node) = .init(null),
    in_flight: std.atomic.Value(usize) = .init(0),
    allocated: std.atomic.Value(usize) = .init(0),
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    pub fn init(max_blocks: usize, block_size: usize) DynamicBufferPool {
        std.debug.assert(max_blocks > 0);
        std.debug.assert(block_size >= @sizeOf(Node));
        return .{
            .block_size = block_size,
            .max_blocks = max_blocks,
            .limit = .init(max_blocks),
        };
    }

    pub fn deinit(self: *DynamicBufferPool, allocator: std.mem.Allocator) void {
        std.debug.assert(self.in_flight.load(.acquire) == 0);
        while (self.pop()) |node| {
            allocator.rawFree(self.nodeToSlice(node), alignment, @returnAddress());
            _ = self.allocated.fetchSub(1, .release);
        }
        std.debug.assert(self.allocated.load(.acquire) == 0);
    }

    pub fn currentLimit(self: *const DynamicBufferPool) usize {
        return self.limit.load(.acquire);
    }

    pub fn inFlight(self: *const DynamicBufferPool) usize {
        return self.in_flight.load(.acquire);
    }

    pub fn allocatedBlocks(self: *const DynamicBufferPool) usize {
        return self.allocated.load(.acquire);
    }

    pub fn setLimit(self: *DynamicBufferPool, io: std.Io, new_limit: usize) void {
        std.debug.assert(new_limit > 0 and new_limit <= self.max_blocks);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        const old_limit = self.limit.swap(new_limit, .acq_rel);
        if (new_limit > old_limit) {
            for (old_limit..new_limit) |_| {
                self.condition.signal(io);
            }
        }
    }

    pub fn trim(self: *DynamicBufferPool, allocator: std.mem.Allocator, io: std.Io, target: usize) void {
        std.debug.assert(target <= self.max_blocks);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        while (self.allocated.load(.acquire) > target) {
            const node = self.pop() orelse break;
            allocator.rawFree(self.nodeToSlice(node), alignment, @returnAddress());
            _ = self.allocated.fetchSub(1, .release);
        }
    }

    pub fn get(self: *DynamicBufferPool, allocator: std.mem.Allocator, io: std.Io) ![]u8 {
        return (try self.getWithWait(allocator, io)).buffer;
    }

    /// Returns the time spent blocked by the pool limit. Allocation and mutex
    /// contention are deliberately excluded from the admission-wait signal.
    pub fn getWithWait(self: *DynamicBufferPool, allocator: std.mem.Allocator, io: std.Io) !Acquisition {
        var wait_ns: u64 = 0;
        while (true) {
            var in_flight = self.in_flight.load(.acquire);
            while (in_flight < self.limit.load(.acquire)) {
                if (self.in_flight.cmpxchgWeak(in_flight, in_flight + 1, .release, .acquire)) |actual| {
                    in_flight = actual;
                    continue;
                }

                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);

                if (self.pop()) |node| {
                    return .{ .buffer = self.nodeToSlice(node), .wait_ns = wait_ns };
                }

                errdefer {
                    _ = self.in_flight.fetchSub(1, .release);
                    self.condition.signal(io);
                }
                // Pool blocks are uninitialized storage. `Allocator.alignedAlloc`
                // materializes Zig's undefined-memory poison across the entire
                // allocation, which is especially expensive for large pinned
                // DMA blocks that the reader immediately overwrites.
                const ptr = allocator.rawAlloc(self.block_size, alignment, @returnAddress()) orelse return error.OutOfMemory;
                const buffer = ptr[0..self.block_size];
                _ = self.allocated.fetchAdd(1, .release);
                return .{ .buffer = @alignCast(buffer), .wait_ns = wait_ns };
            }

            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            while (self.in_flight.load(.acquire) >= self.limit.load(.acquire)) {
                const wait_started: std.Io.Timestamp = .now(io, .awake);
                try self.condition.wait(io, &self.mutex);
                wait_ns +|= @intCast(@max(wait_started.untilNow(io, .awake).nanoseconds, 0));
            }
        }
    }

    pub fn put(self: *DynamicBufferPool, io: std.Io, buf: []u8) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        const node: *Node = @ptrCast(@alignCast(buf.ptr));
        var head = self.free_stack.load(.acquire);
        while (true) {
            node.next = head;
            if (self.free_stack.cmpxchgWeak(head, node, .release, .acquire)) |actual_head| {
                head = actual_head;
            } else {
                break;
            }
        }

        _ = self.in_flight.fetchSub(1, .release);
        self.condition.signal(io);
    }

    fn pop(self: *DynamicBufferPool) ?*Node {
        var head = self.free_stack.load(.acquire);
        while (true) {
            if (self.free_stack.cmpxchgWeak(head, if (head) |h| h.next else null, .acquire, .acquire)) |actual| {
                head = actual;
                continue;
            }

            return head;
        }
    }

    fn nodeToSlice(self: DynamicBufferPool, node: *Node) []u8 {
        const ptr: [*]u8 = @ptrCast(node);
        return ptr[0..self.block_size];
    }
};

test "DynamicBufferPool trims unused blocks" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool: DynamicBufferPool = .init(4, 64);
    defer pool.deinit(allocator);

    const first = try pool.get(allocator, io);
    const second = try pool.get(allocator, io);
    try std.testing.expectEqual(2, pool.allocatedBlocks());

    pool.setLimit(io, 1);
    pool.trim(allocator, io, 1);
    try std.testing.expectEqual(2, pool.allocatedBlocks());

    pool.put(io, first);
    pool.trim(allocator, io, 1);
    pool.put(io, second);

    try std.testing.expectEqual(1, pool.currentLimit());
    try std.testing.expectEqual(1, pool.allocatedBlocks());

    pool.trim(allocator, io, 0);
    try std.testing.expectEqual(0, pool.allocatedBlocks());
}

test "DynamicBufferPool applies runtime limits without oversubscription" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool: DynamicBufferPool = .init(2, 64);
    defer pool.deinit(allocator);

    var group: std.Io.Group = .init;
    var group_awaited = false;
    var acquired: [3]std.Io.Event = @splat(.unset);
    var release: [3]std.Io.Event = @splat(.unset);
    defer {
        for (&release) |*event| event.set(io);
        if (!group_awaited) group.await(io) catch {};
    }

    const Worker = struct {
        fn run(
            pool_: *DynamicBufferPool,
            allocator_: std.mem.Allocator,
            acquired_: *std.Io.Event,
            release_: *std.Io.Event,
            io_: std.Io,
        ) std.Io.Cancelable!void {
            const buffer = pool_.get(allocator_, io_) catch unreachable;
            defer pool_.put(io_, buffer);
            acquired_.set(io_);
            try release_.wait(io_);
        }
    };

    pool.setLimit(io, 1);
    try group.concurrent(io, Worker.run, .{ &pool, allocator, &acquired[0], &release[0], io });
    try acquired[0].wait(io);

    try group.concurrent(io, Worker.run, .{ &pool, allocator, &acquired[1], &release[1], io });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!acquired[1].isSet());

    pool.setLimit(io, 2);
    try acquired[1].wait(io);
    try std.testing.expectEqual(2, pool.inFlight());

    pool.setLimit(io, 1);
    try group.concurrent(io, Worker.run, .{ &pool, allocator, &acquired[2], &release[2], io });
    release[0].set(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!acquired[2].isSet());

    release[1].set(io);
    try acquired[2].wait(io);
    release[2].set(io);
    try group.await(io);
    group_awaited = true;
    try std.testing.expectEqual(0, pool.inFlight());
}

test "DynamicBufferPool reports capacity wait separately from allocation" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool: DynamicBufferPool = .init(1, 64);
    defer pool.deinit(allocator);

    const first = try pool.getWithWait(allocator, io);
    try std.testing.expectEqual(0, first.wait_ns);

    var group: std.Io.Group = .init;
    var group_awaited = false;
    var first_returned = false;
    var started: std.Io.Event = .unset;
    var measured_wait_ns: std.atomic.Value(u64) = .init(0);
    defer {
        if (!first_returned) pool.put(io, first.buffer);
        if (!group_awaited) group.await(io) catch {};
    }

    const Worker = struct {
        fn run(
            pool_: *DynamicBufferPool,
            allocator_: std.mem.Allocator,
            started_: *std.Io.Event,
            measured_wait_ns_: *std.atomic.Value(u64),
            io_: std.Io,
        ) void {
            started_.set(io_);
            const acquisition = pool_.getWithWait(allocator_, io_) catch unreachable;
            measured_wait_ns_.store(acquisition.wait_ns, .release);
            pool_.put(io_, acquisition.buffer);
        }
    };

    try group.concurrent(io, Worker.run, .{ &pool, allocator, &started, &measured_wait_ns, io });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    pool.put(io, first.buffer);
    first_returned = true;
    try group.await(io);
    group_awaited = true;

    try std.testing.expect(measured_wait_ns.load(.acquire) > 0);
}

test "DynamicBufferPool rolls admission back after allocation failure" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool: DynamicBufferPool = .init(1, 64);
    defer pool.deinit(allocator);

    try std.testing.expectError(error.OutOfMemory, pool.get(std.testing.failing_allocator, io));
    try std.testing.expectEqual(0, pool.inFlight());
    try std.testing.expectEqual(0, pool.allocatedBlocks());

    const acquisition = try pool.getWithWait(allocator, io);
    const buffer = acquisition.buffer;
    try std.testing.expectEqual(0, acquisition.wait_ns);
    try std.testing.expectEqual(1, pool.inFlight());
    try std.testing.expectEqual(1, pool.allocatedBlocks());
    pool.put(io, buffer);
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
            inline for (struct_type_info.fields) |field| {
                try bufferizeInner(allocator, @field(model, field.name), &@field(bufferized_, field.name));
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
                    bufferized_.* = try allocator.alignedAlloc(p.child, .fromByteUnits(p.alignment orelse @alignOf(p.child)), model.len);
                    for (model, bufferized_.*) |src, *dst| {
                        try bufferizeInner(allocator, src, dst);
                    }
                },
                else => unreachable,
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

/// Convert a model to its bufferized form by replacing Tensor fields with Buffer
/// and allocating any required slices using the provided allocator.
pub inline fn bufferize(allocator: std.mem.Allocator, comptime ModelType: type, model: *const ModelType) !Bufferized(ModelType) {
    var bufferized: Bufferized(ModelType) = undefined;
    try bufferizeInner(allocator, model.*, &bufferized);
    return bufferized;
}
