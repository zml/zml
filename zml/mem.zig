//! Generic buffer storage and conversion from tensor models to buffer models.

const std = @import("std");

const Buffer = @import("buffer.zig").Buffer;
const meta = @import("meta.zig");
const Tensor = @import("tensor.zig").Tensor;

/// Return a clone of a type with Tensors replaced by Buffer.
/// Non-Tensor metadata is stripped out of the resulting struct.
/// Recursively descends into the type.
pub fn Bufferized(comptime T: type) type {
    @setEvalBranchQuota(10_000);
    return meta.MapRestrict(Tensor, Buffer).map(T);
}

/// Convert a model to its bufferized form by replacing Tensor fields with Buffer
/// and allocating any required slices using the provided allocator.
pub inline fn bufferize(allocator: std.mem.Allocator, comptime ModelType: type, model: *const ModelType) !Bufferized(ModelType) {
    var bufferized: Bufferized(ModelType) = undefined;
    try bufferizeInner(allocator, model.*, &bufferized);
    return bufferized;
}

/// Deinitializes every accelerator buffer and frees the recursive slice
/// storage allocated by `bufferize`.
pub fn deinitBufferized(allocator: std.mem.Allocator, comptime ModelType: type, bufferized: *Bufferized(ModelType)) void {
    deinitBufferizedInner(allocator, bufferized);
}

/// Calls `callback(context, tensor, buffer)` for each tensor of `model` and
/// the buffer at its place in `bufferized`, in the order `meta.visit` finds
/// the tensors. The two trees must have the structure `bufferize` gives:
/// the same union arms, the same optionals set, slices of the same lengths.
/// At the first place they differ the walk stops and returns
/// `error.Mismatch`, the pairs before it already passed to `callback`, and
/// `report` (a fixed writer, cut short when full) ends with the field path
/// from the root, as `.layers[1].ffn`, and what differs there.
pub fn pairBufferized(
    comptime ModelType: type,
    model: *const ModelType,
    bufferized: *Bufferized(ModelType),
    report: *std.Io.Writer,
    context: anytype,
    comptime callback: fn (@TypeOf(context), *const Tensor, *Buffer) void,
) error{Mismatch}!void {
    return pairBufferizedInner(model, bufferized, report, context, callback);
}

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

    pub fn put(self: *FixedBufferPool, io: std.Io, buf: []u8) void {
        std.debug.assert(inRange(buf, self.buffer));
        const idx = @divExact(@intFromPtr(buf.ptr) - @intFromPtr(self.buffer.ptr), self.block_size);
        self.q.putOneUncancelable(io, @intCast(idx)) catch unreachable;
    }

    fn inRange(sub_buffer: []const u8, buffer: []const u8) bool {
        return @intFromPtr(sub_buffer.ptr) >= @intFromPtr(buffer.ptr) and
            @intFromPtr(sub_buffer[sub_buffer.len - 1 ..].ptr) <= @intFromPtr(buffer[buffer.len - 1 ..].ptr);
    }
};

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
                    deinitBufferizedInner(allocator, &@field(bufferized_, field_name));
            };
            inline for (struct_type_info.field_names) |field_name| {
                try bufferizeInner(allocator, @field(model, field_name), &@field(bufferized_, field_name));
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
                    const allocated = try allocator.alignedAlloc(p.child, .fromByteUnits(p.attrs.@"align" orelse @alignOf(p.child)), model.len);
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
        .@"struct" => |info| inline for (info.field_names) |field_name| {
            deinitBufferizedInner(allocator, &@field(value, field_name));
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

/// `report.end` is restored after each part that pairs, so on a mismatch the
/// report holds the path down to it.
fn pairBufferizedInner(model: anytype, bufferized: anytype, report: *std.Io.Writer, context: anytype, comptime callback: anytype) error{Mismatch}!void {
    const Model = @typeInfo(@TypeOf(model)).pointer.child;
    if (comptime !meta.Contains(Model, Tensor)) return;
    // `bufferize` allocates every slice mutable, a `[]const` one included.
    if (Model == Tensor) return callback(context, model, @constCast(bufferized));

    const depth = report.end;
    switch (@typeInfo(Model)) {
        .@"struct" => |info| {
            // Bufferized fields are the model's fields holding tensors, in
            // order; a tuple's are renumbered.
            const buffer_fields = @typeInfo(Bufferized(Model)).@"struct".field_names;
            comptime var index: usize = 0;
            inline for (info.field_names, info.field_types, info.field_attrs) |name, Field, attrs| {
                if (attrs.@"comptime" or comptime !meta.Contains(Field, Tensor)) continue;
                report.print(".{s}", .{name}) catch {};
                try pairBufferizedInner(&@field(model, name), &@field(bufferized, buffer_fields[index]), report, context, callback);
                report.end = depth;
                index += 1;
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
                    try pairBufferizedInner(payload, &@field(bufferized, @tagName(arm)), report, context, callback);
                    report.end = depth;
                },
            }
        },
        .optional => {
            if ((model.* == null) != (bufferized.* == null)) {
                report.print(": {s} in the model, {s} in the buffers", .{ if (model.* == null) "null" else "set", if (bufferized.* == null) "null" else "set" }) catch {};
                return error.Mismatch;
            }
            if (model.*) |*payload| try pairBufferizedInner(payload, &bufferized.*.?, report, context, callback);
        },
        .pointer => |info| {
            if (info.size != .slice) @compileError("mem.pairBufferized: " ++ @typeName(Model) ++ " is not a slice");
            if (model.len != bufferized.len) {
                report.print(": a slice of {d} in the model, of {d} in the buffers", .{ model.len, bufferized.len }) catch {};
                return error.Mismatch;
            }
            for (model.*, bufferized.*, 0..) |*element, *buffer, i| {
                report.print("[{d}]", .{i}) catch {};
                try pairBufferizedInner(element, buffer, report, context, callback);
                report.end = depth;
            }
        },
        .array => for (model, bufferized, 0..) |*element, *buffer, i| {
            report.print("[{d}]", .{i}) catch {};
            try pairBufferizedInner(element, buffer, report, context, callback);
            report.end = depth;
        },
        else => @compileError("mem.pairBufferized: cannot walk " ++ @typeName(Model)),
    }
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
    deinitBufferized(std.testing.allocator, Model, &bufferized);
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
    var bufferized = try bufferize(std.testing.allocator, Model, &model);
    defer deinitBufferized(std.testing.allocator, Model, &bufferized);

    const Pairs = struct {
        tensors: [8]*const Tensor = undefined,
        buffers: [8]*Buffer = undefined,
        len: usize = 0,

        fn add(self: *@This(), tensor: *const Tensor, buffer: *Buffer) void {
            self.tensors[self.len] = tensor;
            self.buffers[self.len] = buffer;
            self.len += 1;
        }
    };
    var pairs: Pairs = .{};
    var report_buffer: [64]u8 = undefined;
    var report: std.Io.Writer = .fixed(&report_buffer);
    try pairBufferized(Model, &model, &bufferized, &report, &pairs, Pairs.add);
    try std.testing.expectEqual(0, report.end);

    var tensors: Pairs = .{};
    meta.forEachVisit(&model, *const Tensor, struct {
        fn call(_: usize, tensor: *const Tensor, out: *Pairs) void {
            out.tensors[out.len] = tensor;
            out.len += 1;
        }
    }.call, .{&tensors});
    var buffers: Pairs = .{};
    meta.forEachVisit(&bufferized, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer, out: *Pairs) void {
            out.buffers[out.len] = buffer;
            out.len += 1;
        }
    }.call, .{&buffers});
    try std.testing.expectEqual(7, pairs.len);
    try std.testing.expectEqual(pairs.len, tensors.len);
    try std.testing.expectEqual(pairs.len, buffers.len);
    // Not `expectEqualSlices`: printing a `*Buffer` reaches code a CPU test build lacks.
    for (0..pairs.len) |i| {
        try std.testing.expect(pairs.tensors[i] == tensors.tensors[i]);
        try std.testing.expect(pairs.buffers[i] == buffers.buffers[i]);
    }
}

test "pairBufferized stops at the first place the trees differ and names it" {
    const Layer = struct { weights: []Tensor, scale: ?Tensor };
    const Ffn = union(enum) { dense: Layer, none: void };
    const Model = struct { ffns: []Ffn, head: Tensor };
    var weights: [3]Tensor = undefined;
    var built = [_]Ffn{ .{ .dense = .{ .weights = weights[0..2], .scale = null } }, .none };
    const built_model: Model = .{ .ffns = &built, .head = undefined };
    var bufferized = try bufferize(std.testing.allocator, Model, &built_model);
    defer deinitBufferized(std.testing.allocator, Model, &bufferized);

    const Case = struct { ffns: [2]Ffn, pairs: usize, report: []const u8 };
    const cases = [_]Case{
        .{ .ffns = .{ .none, .none }, .pairs = 0, .report = ".ffns[0]: union arm .none in the model, .dense in the buffers" },
        .{
            .ffns = .{ .{ .dense = .{ .weights = weights[0..2], .scale = weights[2] } }, .none },
            .pairs = 2,
            .report = ".ffns[0].dense.scale: set in the model, null in the buffers",
        },
        .{
            .ffns = .{ .{ .dense = .{ .weights = weights[0..1], .scale = null } }, .none },
            .pairs = 0,
            .report = ".ffns[0].dense.weights: a slice of 1 in the model, of 2 in the buffers",
        },
    };
    for (cases) |case| {
        var ffns = case.ffns;
        const model: Model = .{ .ffns = &ffns, .head = undefined };
        var pairs: usize = 0;
        var report_buffer: [128]u8 = undefined;
        var report: std.Io.Writer = .fixed(&report_buffer);
        try std.testing.expectError(error.Mismatch, pairBufferized(Model, &model, &bufferized, &report, &pairs, struct {
            fn call(count: *usize, _: *const Tensor, _: *Buffer) void {
                count.* += 1;
            }
        }.call));
        try std.testing.expectEqual(case.pairs, pairs);
        try std.testing.expectEqualStrings(case.report, report.buffered());
    }
}
