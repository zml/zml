const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const ConstSlice = @import("slice.zig").ConstSlice;
const Platform = @import("platform.zig").Platform;
const zml = @import("zml.zig");

const log = std.log.scoped(.@"zml/testing");

var _platform: ?Platform = null;

pub fn env() Platform {
    if (!builtin.is_test) @compileError("Cannot use zml.testing.env outside of a test block");
    if (_platform == null) {
        zml.init();

        _platform = Platform.init(.cpu, std.testing.io, .{}) catch unreachable;
    }

    return _platform.?;
}

/// In neural network we generally care about the relative precision,
/// but on a given dimension, if the output is close to 0, then the precision
/// don't matter as much.
pub fn approxEq(comptime Float: type, l: Float, r: Float, tolerance: Float) bool {
    const closeRel = std.math.approxEqRel(Float, l, r, @floatCast(tolerance));
    const closeAbs = std.math.approxEqAbs(Float, l, r, @floatCast(tolerance / 2));
    return closeRel or closeAbs;
}

/// Testing utility. Accepts both zml.Buffer and zml.HostBuffer but zml.Buffer will be copied to the
/// host for comparison !
pub fn expectClose(io: std.Io, left_: anytype, right_: anytype, tolerance: f32) !void {
    const allocator = if (builtin.is_test) std.testing.allocator else std.heap.smp_allocator;
    var left: ConstSlice, const should_free_left = if (@TypeOf(left_) == zml.Buffer) b: {
        const slice = try left_.toSliceAlloc(allocator, io);
        break :b .{ slice.constSlice(), true };
    } else .{ left_, false };

    var right: ConstSlice, const should_free_right = if (@TypeOf(right_) == zml.Buffer) b: {
        const slice = try right_.toSliceAlloc(allocator, io);
        break :b .{ slice.constSlice(), true };
    } else .{ right_, false };

    defer {
        if (should_free_left) left.free(allocator);
        if (should_free_right) right.free(allocator);
    }
    errdefer log.err("\n--> Left: {0f}{0d:24.3}\n--> Right: {1f}{1d:24.3}", .{ left, right });
    if (!std.mem.eql(i64, left.shape.dims(), right.shape.dims())) {
        log.err("left.shape() {f} != right.shape() {f}", .{ left.shape, right.shape });
        return error.TestUnexpectedResult;
    }
    if (left.dtype() != right.dtype() and !(left.dtype() == .f16 and right.dtype() == .bf16)) {
        log.err("left.dtype ({}) != right.dtype ({})", .{ left.dtype(), right.dtype() });
        return error.TestUnexpectedResult;
    }

    switch (left.dtype()) {
        inline .bf16,
        .f16,
        .f32,
        .f64,
        .f4e2m1,
        .f8e3m4,
        .f8e4m3,
        .f8e4m3b11fnuz,
        .f8e4m3fn,
        .f8e4m3fnuz,
        .f8e5m2,
        .f8e5m2fnuz,
        .f8e8m0,
        => |t| {
            const L = t.toZigType();
            const left_data = left.items(L);
            switch (right.dtype()) {
                inline .bf16,
                .f16,
                .f32,
                .f64,
                .f8e4m3b11fnuz,
                .f8e4m3fn,
                .f8e4m3fnuz,
                .f8e5m2,
                .f8e5m2fnuz,
                => |rt| {
                    const R = rt.toZigType();
                    const right_data = right.items(R);
                    for (left_data, right_data, 0..) |l, r, i| {
                        if (!approxEq(f32, zml.floats.floatCast(f32, l), zml.floats.floatCast(f32, r), tolerance)) {
                            log.err("left.data != right_data.\n < {d:40.3} \n > {d:40.3}\n error at idx {d}: {d:.3} != {d:.3}", .{ stdx.fmt.slice(center(left_data, i)), stdx.fmt.slice(center(right_data, i)), i, left_data[i], right_data[i] });
                            return error.TestUnexpectedResult;
                        }
                    }
                },
                else => unreachable,
            }
        },
        inline .bool, .u2, .u4, .u8, .u16, .u32, .u64, .i2, .i4, .i8, .i16, .i32, .i64 => |t| {
            const T = t.toZigType();
            return std.testing.expectEqualSlices(T, left.items(T), right.items(T));
        },
        .c64, .c128 => @panic("TODO: support comparison of complex"),
    }
}

pub fn expectEqualShapes(expected: zml.Shape, actual: zml.Shape) error{TestExpectedEqual}!void {
    if (expected.eqlWithTags(actual)) return;

    std.debug.print("Expected {f}, got {f}", .{ expected, actual });
    return error.TestExpectedEqual;
}

fn BufferizedWithArgs(comptime T: type) type {
    return zml.meta.MapType(zml.Tensor, zml.Buffer).map(T);
}

/// Automatically calls the executable with the given arguments, taking care of arguments and results allocation.
/// This helper can be used in tests to make the code a bit less verbose.
/// It doesn't handle pointers in inputs/outputs structs.
pub fn autoCall(allocator: std.mem.Allocator, io: std.Io, exe: *const zml.exe.Exe, func: anytype, inputs: zml.Bufferized(stdx.meta.FnArgs(func))) !zml.Bufferized(stdx.meta.FnResult(func)) {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(inputs);
    exe.call(args, &results, io);

    var output: zml.Bufferized(stdx.meta.FnResult(func)) = undefined;
    results.fill(.{&output});

    return output;
}

fn center(slice: anytype, i: usize) @TypeOf(slice) {
    const c = 20;
    const start = if (i < c) 0 else i - c;
    const end = @min(start + 2 * c, slice.len);
    return slice[start..end];
}

pub inline fn expectEqual(expected: anytype, actual: @TypeOf(expected)) !void {
    return std.testing.expectEqual(expected, actual);
}
