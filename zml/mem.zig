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

/// Convert the model `model` points to into its bufferized form by replacing
/// Tensor fields with Buffer and allocating any required slices using the
/// provided allocator.
pub inline fn bufferize(allocator: std.mem.Allocator, model: anytype) !Bufferized(@TypeOf(model.*)) {
    var bufferized: Bufferized(@TypeOf(model.*)) = undefined;
    try bufferizeInner(allocator, model.*, &bufferized);
    return bufferized;
}

/// A tensor of a model and the buffer at its place in the model's buffers.
pub const Pair = struct { tensor: *const Tensor, buffer: *Buffer };

/// Pairs each tensor of the model `model` points to with the buffer at its
/// place in `bufferized`, in the order `meta.visit` finds the tensors. The
/// caller frees the pairs. The two trees must have the structure `bufferize`
/// gives: the same union arms, the same optionals set, slices of the same
/// lengths. At the first place they differ the walk stops and returns
/// `error.Mismatch`, and `report` (a fixed writer, cut short when full) ends
/// with the field path from the root, as `.layers[1].ffn`, and what differs
/// there.
pub fn pairBufferized(
    allocator: std.mem.Allocator,
    model: anytype,
    bufferized: *Bufferized(@TypeOf(model.*)),
    report: *std.Io.Writer,
) error{ Mismatch, OutOfMemory }![]Pair {
    var pairs: std.ArrayList(Pair) = try .initCapacity(allocator, meta.count(Tensor, model));
    errdefer pairs.deinit(allocator);
    try pairBufferizedInner(model, bufferized, report, &pairs);
    return pairs.toOwnedSlice(allocator);
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
            else => unreachable,
        },
        .array => for (&value.*) |*element| deinitBufferized(allocator, element),
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

/// `report.end` is restored after each part that pairs, so on a mismatch the
/// report holds the path down to it.
fn pairBufferizedInner(model: anytype, bufferized: anytype, report: *std.Io.Writer, pairs: *std.ArrayList(Pair)) error{Mismatch}!void {
    const Model = @typeInfo(@TypeOf(model)).pointer.child;
    if (comptime !meta.Contains(Model, Tensor)) return;
    // `bufferize` allocates every slice mutable, a `[]const` one included.
    if (Model == Tensor) return pairs.appendAssumeCapacity(.{ .tensor = model, .buffer = @constCast(bufferized) });

    const depth = report.end;
    switch (@typeInfo(Model)) {
        .@"struct" => |info| inline for (info.field_names) |name| {
            if (comptime bufferizedField(Model, name)) |buffer_field| {
                report.print(".{s}", .{name}) catch {};
                try pairBufferizedInner(&@field(model, name), &@field(bufferized, buffer_field), report, pairs);
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
                    try pairBufferizedInner(payload, &@field(bufferized, @tagName(arm)), report, pairs);
                    report.end = depth;
                },
            }
        },
        .optional => {
            if ((model.* == null) != (bufferized.* == null)) {
                report.print(": {s} in the model, {s} in the buffers", .{ if (model.* == null) "null" else "set", if (bufferized.* == null) "null" else "set" }) catch {};
                return error.Mismatch;
            }
            if (model.*) |*payload| try pairBufferizedInner(payload, &bufferized.*.?, report, pairs);
        },
        .pointer => |info| {
            if (info.size != .slice) @compileError("mem.pairBufferized: " ++ @typeName(Model) ++ " is not a slice");
            if (model.len != bufferized.len) {
                report.print(": a slice of {d} in the model, of {d} in the buffers", .{ model.len, bufferized.len }) catch {};
                return error.Mismatch;
            }
            for (model.*, bufferized.*, 0..) |*element, *buffer, i| {
                report.print("[{d}]", .{i}) catch {};
                try pairBufferizedInner(element, buffer, report, pairs);
                report.end = depth;
            }
        },
        .array => for (model, bufferized, 0..) |*element, *buffer, i| {
            report.print("[{d}]", .{i}) catch {};
            try pairBufferizedInner(element, buffer, report, pairs);
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

test "bufferize maps a tuple's fields as Bufferized renumbers them" {
    const Model = struct { u32, Tensor, []const Tensor };
    var tensors: [2]Tensor = undefined;
    const model: Model = .{ 3, undefined, &tensors };
    var bufferized = try bufferize(std.testing.allocator, &model);
    defer deinitBufferized(std.testing.allocator, &bufferized);
    try std.testing.expectEqual(2, bufferized[1].len);
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

    var report_buffer: [64]u8 = undefined;
    var report: std.Io.Writer = .fixed(&report_buffer);
    const pairs = try pairBufferized(std.testing.allocator, &model, &bufferized, &report);
    defer std.testing.allocator.free(pairs);
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
    try std.testing.expectEqual(7, pairs.len);
    try std.testing.expectEqual(pairs.len, visited.tensor_count);
    try std.testing.expectEqual(pairs.len, visited.buffer_count);
    // Not `expectEqualSlices`: printing a `*Buffer` reaches code a CPU test build lacks.
    for (pairs, 0..) |pair, i| {
        try std.testing.expect(pair.tensor == visited.tensors[i]);
        try std.testing.expect(pair.buffer == visited.buffers[i]);
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
    for (cases) |case| {
        var ffns = case.ffns;
        const model: Model = .{ .ffns = &ffns, .head = undefined };
        var report_buffer: [128]u8 = undefined;
        var report: std.Io.Writer = .fixed(&report_buffer);
        // Not `expectError`: printing a `*Buffer` reaches code a CPU test build lacks.
        if (pairBufferized(std.testing.allocator, &model, &bufferized, &report)) |pairs| {
            std.testing.allocator.free(pairs);
            return error.TestExpectedMismatch;
        } else |err| try std.testing.expectEqual(error.Mismatch, err);
        try std.testing.expectEqualStrings(case.report, report.buffered());
    }
}
