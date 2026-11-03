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

/// Convert the model `model` points to into its bufferized form by replacing
/// Tensor fields with Buffer and allocating any required slices using the
/// provided allocator.
pub inline fn bufferize(allocator: std.mem.Allocator, model: anytype) !Bufferized(@TypeOf(model.*)) {
    var bufferized: Bufferized(@TypeOf(model.*)) = undefined;
    try bufferizeInner(allocator, model.*, &bufferized);
    return bufferized;
}

/// A tensor of a model, the buffer at its place in the model's buffers, and
/// its field path from the root, as `.layers[1].ffn.weight`.
pub const Pair = struct { tensor: *const Tensor, buffer: *Buffer, path: []const u8 };

/// Pairs each tensor of the model `model` points to with the buffer at its
/// place in `bufferized`, in the order `meta.visit` finds the tensors. The
/// pairs and their paths live in `arena`. The two trees must have the
/// structure `bufferize` gives: the same union arms, the same optionals set,
/// slices of the same lengths. `report`, an empty fixed writer, builds the
/// paths and cuts them short when full. At the first place the trees differ
/// the walk stops and returns `error.Mismatch`, and `report` ends with the
/// path there and what differs.
pub fn pairBufferized(
    arena: std.mem.Allocator,
    model: anytype,
    bufferized: *Bufferized(@TypeOf(model.*)),
    report: *std.Io.Writer,
) error{ Mismatch, OutOfMemory }![]Pair {
    var pairs: std.ArrayList(Pair) = try .initCapacity(arena, meta.count(Tensor, model));
    try pairBufferizedInner(arena, model, bufferized, report, &pairs);
    return pairs.items;
}

/// `report.end` is restored after each part that pairs, so at a tensor or a
/// mismatch the report holds the path down to it.
fn pairBufferizedInner(
    arena: std.mem.Allocator,
    model: anytype,
    bufferized: anytype,
    report: *std.Io.Writer,
    pairs: *std.ArrayList(Pair),
) error{ Mismatch, OutOfMemory }!void {
    const Model = @typeInfo(@TypeOf(model)).pointer.child;
    if (comptime !meta.Contains(Model, Tensor)) return;
    if (Model == Tensor) return pairs.appendAssumeCapacity(.{
        .tensor = model,
        // `bufferize` allocates every slice mutable, a `[]const` one included.
        .buffer = @constCast(bufferized),
        .path = try arena.dupe(u8, report.buffered()),
    });

    const depth = report.end;
    switch (@typeInfo(Model)) {
        .@"struct" => |info| inline for (info.field_names) |name| {
            if (comptime bufferizedField(Model, name)) |buffer_field| {
                report.print(".{s}", .{name}) catch {};
                try pairBufferizedInner(arena, &@field(model, name), &@field(bufferized, buffer_field), report, pairs);
                report.end = depth;
            }
        },
        .@"union" => {
            const tag = std.meta.activeTag(model.*);
            if (tag != std.meta.activeTag(bufferized.*)) {
                report.print(": union arm .{t} in the model, .{t} in the buffers", .{ tag, std.meta.activeTag(bufferized.*) }) catch {};
                return error.Mismatch;
            }
            switch (model.*) {
                inline else => |*payload, arm| {
                    report.print(".{s}", .{@tagName(arm)}) catch {};
                    try pairBufferizedInner(arena, payload, &@field(bufferized, @tagName(arm)), report, pairs);
                    report.end = depth;
                },
            }
        },
        .optional => {
            if ((model.* == null) != (bufferized.* == null)) {
                report.print(": {s} in the model, {s} in the buffers", .{ if (model.* == null) "null" else "set", if (bufferized.* == null) "null" else "set" }) catch {};
                return error.Mismatch;
            }
            if (model.*) |*payload| try pairBufferizedInner(arena, payload, &bufferized.*.?, report, pairs);
        },
        .pointer => |info| {
            if (info.size != .slice) @compileError("mem.pairBufferized: " ++ @typeName(Model) ++ " is not a slice");
            if (model.len != bufferized.len) {
                report.print(": a slice of {d} in the model, of {d} in the buffers", .{ model.len, bufferized.len }) catch {};
                return error.Mismatch;
            }
            for (model.*, bufferized.*, 0..) |*element, *buffer, i| {
                report.print("[{d}]", .{i}) catch {};
                try pairBufferizedInner(arena, element, buffer, report, pairs);
                report.end = depth;
            }
        },
        .array => for (model, bufferized, 0..) |*element, *buffer, i| {
            report.print("[{d}]", .{i}) catch {};
            try pairBufferizedInner(arena, element, buffer, report, pairs);
            report.end = depth;
        },
        else => @compileError("mem.pairBufferized: cannot walk " ++ @typeName(Model)),
    }
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
    var bufferized = try bufferize(std.testing.allocator, &model);
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

    if (bufferize(failing.allocator(), &model)) |result| {
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
    var bufferized = try bufferize(std.testing.allocator, &model);
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
    var bufferized = try bufferize(std.testing.allocator, &model);
    deinitBufferized(std.testing.allocator, &bufferized);
}

test "pairBufferized pairs tensors and buffers in visit order" {
    const Layer = struct { weights: []Tensor, scale: ?Tensor };
    const Ffn = union(enum) { dense: Layer, experts: []Layer, none: void };
    const Model = struct { ffns: [3]Ffn, head: Tensor, layers: u32 };
    var weights: [4]Tensor = undefined;
    var experts = [_]Layer{ .{ .weights = weights[1..2], .scale = null }, .{ .weights = weights[2..4], .scale = weights[3] } };
    var model: Model = .{
        .ffns = .{ .{ .dense = .{ .weights = weights[0..1], .scale = weights[0] } }, .{ .experts = &experts }, .none },
        .head = undefined,
        .layers = 2,
    };
    var bufferized = try bufferize(std.testing.allocator, &model);
    defer deinitBufferized(std.testing.allocator, &bufferized);

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var report_buffer: [64]u8 = undefined;
    var report: std.Io.Writer = .fixed(&report_buffer);
    const pairs = try pairBufferized(arena.allocator(), &model, &bufferized, &report);
    try std.testing.expectEqual(0, report.end);

    const Visited = struct {
        tensors: [8]*const Tensor = undefined,
        buffers: [8]*Buffer = undefined,
        tensor_count: usize = 0,
        buffer_count: usize = 0,
    };
    var visited: Visited = .{};
    meta.forEachVisit(&model, *const Tensor, struct {
        fn call(_: usize, tensor: *const Tensor, out: *Visited) void {
            out.tensors[out.tensor_count] = tensor;
            out.tensor_count += 1;
        }
    }.call, .{&visited});
    meta.forEachVisit(&bufferized, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer, out: *Visited) void {
            out.buffers[out.buffer_count] = buffer;
            out.buffer_count += 1;
        }
    }.call, .{&visited});
    const paths = [_][]const u8{
        ".ffns[0].dense.weights[0]",
        ".ffns[0].dense.scale",
        ".ffns[1].experts[0].weights[0]",
        ".ffns[1].experts[1].weights[0]",
        ".ffns[1].experts[1].weights[1]",
        ".ffns[1].experts[1].scale",
        ".head",
    };
    try std.testing.expectEqual(paths.len, pairs.len);
    try std.testing.expectEqual(pairs.len, visited.tensor_count);
    try std.testing.expectEqual(pairs.len, visited.buffer_count);
    // Not `expectEqualSlices`: printing a `*Buffer` reaches code a CPU test build lacks.
    for (pairs, paths, 0..) |pair, path, i| {
        try std.testing.expect(pair.tensor == visited.tensors[i]);
        try std.testing.expect(pair.buffer == visited.buffers[i]);
        try std.testing.expectEqualStrings(path, pair.path);
    }
}

test "pairBufferized stops at the first place the trees differ and names it" {
    const Layer = struct { weights: []Tensor, scale: ?Tensor };
    const Ffn = union(enum) { dense: Layer, none: void };
    const Model = struct { ffns: []Ffn, head: Tensor };
    var weights: [3]Tensor = undefined;
    var built = [_]Ffn{ .{ .dense = .{ .weights = weights[0..2], .scale = null } }, .none };
    const built_model: Model = .{ .ffns = &built, .head = undefined };
    var bufferized = try bufferize(std.testing.allocator, &built_model);
    defer deinitBufferized(std.testing.allocator, &bufferized);

    const Case = struct { ffns: [2]Ffn, report: []const u8 };
    const cases = [_]Case{
        .{ .ffns = .{ .none, .none }, .report = ".ffns[0]: union arm .none in the model, .dense in the buffers" },
        .{
            .ffns = .{ .{ .dense = .{ .weights = weights[0..2], .scale = weights[2] } }, .none },
            .report = ".ffns[0].dense.scale: set in the model, null in the buffers",
        },
        .{
            .ffns = .{ .{ .dense = .{ .weights = weights[0..1], .scale = null } }, .none },
            .report = ".ffns[0].dense.weights: a slice of 1 in the model, of 2 in the buffers",
        },
    };
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |case| {
        var ffns = case.ffns;
        const model: Model = .{ .ffns = &ffns, .head = undefined };
        var report_buffer: [128]u8 = undefined;
        var report: std.Io.Writer = .fixed(&report_buffer);
        // Not `expectError`: printing a `*Buffer` reaches code a CPU test build lacks.
        if (pairBufferized(arena.allocator(), &model, &bufferized, &report)) |_| {
            return error.TestExpectedMismatch;
        } else |err| try std.testing.expectEqual(error.Mismatch, err);
        try std.testing.expectEqualStrings(case.report, report.buffered());
    }
}
