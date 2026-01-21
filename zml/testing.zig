const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const Slice = @import("slice.zig").Slice;
const Platform = @import("platform.zig").Platform;
const zml = @import("zml.zig");

const log = std.log.scoped(.@"zml/testing");

var _platform: ?Platform = null;

pub fn env() Platform {
    if (!builtin.is_test) @compileError("Cannot use zml.testing.env outside of a test block");
    if (_platform == null) {
        _platform = Platform.auto(std.testing.allocator, std.testing.io, .{}) catch unreachable;
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
    var left: Slice, const should_free_left = if (@TypeOf(left_) == zml.Buffer) b: {
        const slice = try left_.toSliceAlloc(allocator, io);
        break :b .{ slice, true };
    } else .{ left_, false };

    var right: Slice, const should_free_right = if (@TypeOf(right_) == zml.Buffer) b: {
        const slice = try right_.toSliceAlloc(allocator, io);
        break :b .{ slice, true };
    } else .{ right_, false };

    defer {
        if (should_free_left) left.free(allocator);
        if (should_free_right) right.free(allocator);
    }
    errdefer log.err("\n--> Left: {0d:24.3}\n--> Right: {1d:24.3}", .{ left, right });
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
            const left_data = left.constItems(L);
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
                    const right_data = right.constItems(R);
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
            return std.testing.expectEqualSlices(T, left.constItems(T), right.constItems(T));
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
    exe.callOpts(io, args, &results, .{ .wait = true });

    var output: zml.Bufferized(stdx.meta.FnResult(func)) = undefined;
    results.fill(.{&output});

    return output;
}

fn countWithPrefix(store: zml.io.TensorStore.View, prefix: []const u8) usize {
    var count: usize = 0;
    var it = store.store.registry.iterator();
    while (it.next()) |entry| {
        if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
            count += 1;
        }
    }

    return count;
}

/// Test a layer implementation against reference activations stored in a TensorStore.
/// The layer is expected to have a method with the name given by `func`.
/// The reference activations are expected to be stored under the keys:
/// - `{name}.in.0`, `{name}.in.1`, ... for inputs
/// - `{name}.out.0`, `{name}.out.1`, ... for outputs
///
/// The layer weights are expected to be provided in `layer_weights`.
/// The comparison tolerance can be adjusted with `tolerance`.
///
/// The function arguments need to be valid when initialized to undefined, meaning no slices,
/// no unions, no pointers.
pub fn testlayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: zml.Platform,
    layer: anytype,
    comptime func: std.meta.DeclEnum(@TypeOf(layer)),
    activation_store: zml.io.TensorStore.View,
    comptime name: []const u8,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    tolerance: f32,
) !void {
    const forward = @field(@TypeOf(layer), @tagName(func));
    const ArgsT = stdx.meta.Tail(stdx.meta.FnArgs(forward));

    var args: ArgsT = undefined;

    const LocalContext = struct {
        activation_store: zml.io.TensorStore.View,
        index: usize = 0,
    };

    const input_count = zml.meta.count(zml.Tensor, &args);
    const expected_input_count = countWithPrefix(activation_store, name ++ ".in");
    if (input_count != expected_input_count) {
        log.warn("Reference models uses {d} inputs, but implementation uses {d}", .{ expected_input_count, input_count });
    }

    const store_input = activation_store.withPrefix(name ++ ".in");
    var ctx = LocalContext{ .activation_store = store_input };
    try zml.meta.visit(struct {
        fn cb(ctx_: *const LocalContext, tensor: *zml.Tensor) !void {
            var buffer: [256]u8 = undefined;
            const subkey = std.fmt.bufPrint(&buffer, "{d}", .{ctx_.index}) catch unreachable;

            tensor.* = ctx_.activation_store.createTensor(subkey);
        }
    }.cb, &ctx, &args);

    const exe = try platform.compile(allocator, io, layer, func, args);
    defer exe.deinit();

    const output_count = exe.output_shapes.len;
    const store_output = activation_store.withPrefix(name ++ ".out");
    const expected_output_count = countWithPrefix(store_output, name ++ ".out");
    if (output_count != expected_output_count) {
        log.warn("Reference models produces {d} outputs, but implementation produces {d}", .{ expected_output_count, output_count });
    }

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    var args_buffers = try zml.io.loadBuffersFromId(allocator, io, platform, args, activation_store, .{ .size = 4096, .concurrency = 1 }, .{ .size = 4096, .concurrency = 1 });
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &args_buffers);

    exe_args.set(.{ layer_weights, args_buffers });

    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, output_count);
    defer allocator.free(results);

    exe_results.fill(.{results});

    var failed: bool = false;
    var reader_buffer: [4096]u8 = undefined;
    for (0..output_count) |i| {
        const expected_slice = b: {
            var buffer: [256]u8 = undefined;
            const subkey = try std.fmt.bufPrint(&buffer, "{d}", .{i});
            var reader = try store_output.getReader(subkey, io, &reader_buffer);
            defer reader.deinit();
            const shape = store_output.getShape(subkey).?;
            const expected_bytes = try reader.interface.readAlloc(allocator, shape.byteSize());
            break :b zml.Slice.init(shape, expected_bytes);
        };
        defer expected_slice.free(allocator);

        const output_slice = try results[i].toSliceAlloc(allocator, io);
        defer output_slice.free(allocator);

        zml.testing.expectClose(io, expected_slice, output_slice, tolerance) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                log.err("{s}.{d} doesn't match !", .{ name ++ ".out", i });
                failed = true;
                continue;
            },
            else => return err,
        };
    }

    if (failed) return error.TestUnexpectedResult;
    log.info("all good for {s} !", .{name});
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
