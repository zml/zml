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
            .rocm, .oneapi, .tpu, .furiosa => .{ .uib = .init(device.memory(.host_pinned).?) },
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

pub const DynamicBufferPool = struct {
    const Node = struct { next: ?*Node };
    const alignment: std.mem.Alignment = .of(Node);

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

    fn nodeToSlice(self: DynamicBufferPool, node: *Node) []u8 {
        const ptr: [*]u8 = @ptrCast(node);
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

/// The field of `Bufferized(Model)` that holds the struct `Model`'s field
/// `name`, or null when `Bufferized` strips it. `Bufferized` keeps the
/// fields holding tensors, in order, and renumbers a tuple's.
fn bufferizedField(comptime Model: type, comptime name: []const u8) ?[]const u8 {
    @setEvalBranchQuota(10_000);
    const info = @typeInfo(Model).@"struct";
    var index: usize = 0;
    for (info.field_names, info.field_types, info.field_attrs) |field_name, Field, attrs| {
        if (attrs.@"comptime" or !meta.Contains(Field, Tensor)) continue;
        if (std.mem.eql(u8, field_name, name)) return @typeInfo(Bufferized(Model)).@"struct".field_names[index];
        index += 1;
    }
    return null;
}

fn bufferizeInner(allocator: std.mem.Allocator, model: anytype, bufferized_: *Bufferized(@TypeOf(model))) !void {
    @setEvalBranchQuota(10_000);
    const Model = @TypeOf(model);
    const ModelBufferized = Bufferized(Model);

    if (ModelBufferized == Buffer) {
        bufferized_.* = .{
            // I'm not sure I like that. I'd rather set all fields but _shards than leaving most of the m undefined.
            ._shape = model._shape,
            ._shards = .empty,
            ._platform = undefined,
        };
        return;
    }

    const type_info = @typeInfo(ModelBufferized);
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            var initialized_fields: usize = 0;
            errdefer inline for (struct_type_info.field_names, 0..) |field_name, index| {
                if (index < initialized_fields)
                    deinitBufferized(allocator, &@field(bufferized_, field_name));
            };
            inline for (@typeInfo(Model).@"struct".field_names) |field_name| {
                if (comptime bufferizedField(Model, field_name)) |buffer_field| {
                    try bufferizeInner(allocator, @field(model, field_name), &@field(bufferized_, buffer_field));
                    initialized_fields += 1;
                }
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
                    const allocated = try allocator.alignedAlloc(p.child, .fromByteUnits(p.attrs.@"align" orelse @alignOf(p.child)), model.len);
                    var initialized: usize = 0;
                    errdefer {
                        for (allocated[0..initialized]) |*element| deinitBufferized(allocator, element);
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
            errdefer for (bufferized_.*[0..initialized]) |*element| deinitBufferized(allocator, element);
            inline for (0..info.len) |index| {
                try bufferizeInner(allocator, model[index], &bufferized_.*[index]);
                initialized = index + 1;
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

/// Deinitializes every accelerator buffer of the tree `value` points to and
/// frees the recursive slice storage allocated by `bufferize`.
pub fn deinitBufferized(allocator: std.mem.Allocator, value: anytype) void {
    const Ptr = @TypeOf(value);
    const T = @typeInfo(Ptr).pointer.child;
    if (T == Tensor) @compileError("mem.deinitBufferized takes the buffers bufferize returns, not the model");
    if (T == Buffer) {
        const buffer: *Buffer = @constCast(value);
        buffer.deinit();
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| inline for (info.field_names) |field_name| {
            deinitBufferized(allocator, &@field(value, field_name));
        },
        .@"union" => switch (value.*) {
            inline else => |*payload| deinitBufferized(allocator, payload),
        },
        .optional => if (value.*) |*payload| {
            deinitBufferized(allocator, payload);
        },
        .pointer => |info| switch (info.size) {
            .slice => {
                for (value.*) |*element| deinitBufferized(allocator, element);
                allocator.free(value.*);
            },
            else => @compileError("mem.deinitBufferized takes the buffers bufferize returns, not " ++ @typeName(T)),
        },
        .array => for (&value.*) |*element| deinitBufferized(allocator, element),
        // What `Bufferized` makes of a union arm or a whole tree without a tensor.
        .void => {},
        else => @compileError("mem.deinitBufferized takes the buffers bufferize returns, not " ++ @typeName(T)),
    }
}

test "bufferize maps a tuple's fields as Bufferized renumbers them" {
    const Model = struct { u32, Tensor, []Tensor, [2]Tensor };
    const a: Tensor = .init(.{2}, .f32);
    const b: Tensor = .init(.{3}, .f16);
    var tensors = [_]Tensor{ a, b };
    const model: Model = .{ 3, a, &tensors, .{ b, a } };
    var bufferized = try bufferize(std.testing.allocator, Model, &model);
    defer deinitBufferized(std.testing.allocator, &bufferized);
    try std.testing.expect(bufferized[0].shape().eql(a.shape()));
    try std.testing.expectEqual(2, bufferized[1].len);
    try std.testing.expect(bufferized[1][1].shape().eql(b.shape()));
    try std.testing.expect(bufferized[2][0].shape().eql(b.shape()));
    try std.testing.expect(bufferized[2][1].shape().eql(a.shape()));
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
        deinitBufferized(failing.allocator(), &unexpected);
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
    deinitBufferized(std.testing.allocator, &bufferized);
}

test "deinitBufferized frees slices under optionals and unions" {
    const Layer = struct { weights: []const Tensor };
    const Ffn = union(enum) { dense: Layer, experts: []const Layer, none: void };
    const Model = struct {
        present: ?Layer,
        absent: ?Layer,
        ffns: [3]Ffn,
    };
    var weights: [3]Tensor = undefined;
    const experts = [_]Layer{ .{ .weights = weights[0..1] }, .{ .weights = weights[1..3] } };
    const model: Model = .{
        .present = .{ .weights = &weights },
        .absent = null,
        .ffns = .{ .{ .dense = .{ .weights = weights[2..3] } }, .{ .experts = &experts }, .none },
    };
    var bufferized = try bufferize(std.testing.allocator, Model, &model);
    deinitBufferized(std.testing.allocator, &bufferized);
}
