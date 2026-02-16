const std = @import("std");
const Allocator = std.mem.Allocator;
const Alignment = std.mem.Alignment;
const assert = std.debug.assert;

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
            .cuda => .{ .dmam = .init(parent, device.platform) },
            .tpu, .neuron => .{ .uib = .init(device.memory(.host_pinned)) },
            .rocm, .cpu => .{ .passthrough = parent },
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

pub const DmaMapAllocator = struct {
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
        const allocation = self.parent.rawAlloc(len, alignment, ret_addr);
        if (allocation) |loc| {
            const data = loc[0..len];
            self.platform.pjrt_client.dmaMap(self.platform.pjrt_api, @ptrCast(data)) catch {
                self.parent.rawFree(data, alignment, ret_addr);
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
        self.parent.rawFree(buf, alignment, ret_addr);
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
    const alignment: std.mem.Alignment = .max(.of(u8), .of(Node));
    const ItemPtr = [*]align(alignment.toByteUnits()) u8;
    const Slice = []align(alignment.toByteUnits()) u8;

    block_size: usize,
    max_blocks: usize,

    arena_state: std.heap.ArenaAllocator.State = .{},
    free_stack: std.atomic.Value(?*Node) = .init(null),
    in_flight: std.atomic.Value(usize) = .init(0),
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    pub fn init(max_blocks: usize, block_size: usize) DynamicBufferPool {
        return .{
            .block_size = block_size,
            .max_blocks = max_blocks,
        };
    }

    pub fn deinit(self: *DynamicBufferPool, allocator: std.mem.Allocator) void {
        self.arena_state.promote(allocator).deinit();
    }

    pub fn get(self: *DynamicBufferPool, allocator: std.mem.Allocator, io: std.Io) ![]u8 {
        while (true) {
            var in_flight = self.in_flight.load(.acquire);
            while (in_flight < self.max_blocks) {
                if (self.in_flight.cmpxchgWeak(in_flight, in_flight + 1, .release, .acquire)) |actual| {
                    in_flight = actual;
                    continue;
                }

                if (self.pop()) |node| {
                    return self.nodeToSlice(node);
                }

                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);

                var arena = self.arena_state.promote(allocator);
                defer self.arena_state = arena.state;

                errdefer _ = self.in_flight.fetchSub(1, .release);
                const buffer = try arena.allocator().alignedAlloc(u8, alignment, self.block_size);
                return @alignCast(buffer);
            }

            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            while (self.in_flight.load(.acquire) >= self.max_blocks) {
                try self.condition.wait(io, &self.mutex);
            }
        }
    }

    pub fn put(self: *DynamicBufferPool, io: std.Io, buf: []u8) void {
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

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
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

    fn nodeToSlice(self: *DynamicBufferPool, node: *Node) Slice {
        const ptr: ItemPtr = @ptrCast(@alignCast(node));
        return ptr[0..self.block_size];
    }
};

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
                    try bufferizeInner(allocator, v, @field(bufferized_, @tagName(tag)));
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
                    bufferized_.* = try allocator.alignedAlloc(p.child, .fromByteUnits(p.alignment), model.len);
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
