const std = @import("std");
const builtin = @import("builtin");

const zml = @import("zml.zig");
const meta = @import("meta.zig");
const shapesOf = @import("tensor.zig").shapesOf;

const log = std.log.scoped(.zml_testing);

var _ctx: ?zml.Context = null;

pub fn env() zml.Platform {
    if (!builtin.is_test) @compileError("Cannot use zml.testing.env outside of a test block");
    if (_ctx == null) {
        _test_compile_opts = if (initCacheDir())
            .{
                .cache_location = "/tmp/zml/tests/cache",
                .xla_dump_to = "/tmp/zml/tests/",
            }
        else
            .{};

        _ctx = zml.Context.init() catch unreachable;
    }
    return _ctx.?.platforms.get(.cpu).?.withCompilationOptions(_test_compile_opts);
}

var _test_compile_opts: zml.CompilationOptions = .{};

fn initCacheDir() bool {
    const tmp = std.fs.openDirAbsolute("/tmp", .{}) catch return false;
    tmp.makePath("zml/tests/cache") catch return false;
    return true;
}

/// In neural network we generally care about the relative precision,
/// but on a given dimension, if the output is close to 0, then the precision
/// don't matter as much.
pub fn approxEq(comptime Float: type, l: Float, r: Float, tolerance: Float) bool {
    const closeRel = std.math.approxEqRel(Float, l, r, @floatCast(tolerance));
    const closeAbs = std.math.approxEqAbs(Float, l, r, @floatCast(tolerance / 2));
    return closeRel or closeAbs;
}

/// Testing utility. Accepts both Tensor and HostBuffer but Tensor will be copied to the
/// host for comparison !
pub fn expectClose(left_: anytype, right_: anytype, tolerance: f32) !void {
    const allocator = if (builtin.is_test) std.testing.allocator else std.heap.page_allocator;
    var left: zml.HostBuffer, const should_free_left = if (@TypeOf(left_) == zml.Buffer)
        .{ try left_.toHostAlloc(allocator), true }
    else
        .{ left_, false };

    var right: zml.HostBuffer, const should_free_right = if (@TypeOf(right_) == zml.Buffer)
        .{ try right_.toHostAlloc(allocator), true }
    else
        .{ right_, false };

    defer {
        if (should_free_left) left.deinit(allocator);
        if (should_free_right) right.deinit(allocator);
    }

    if (!std.mem.eql(i64, left.shape().dims(), right.shape().dims())) {
        log.err("left.shape() {} != right.shape() {}", .{ left.shape(), right.shape() });
        return error.TestUnexpectedResult;
    }
    if (left.dtype() != right.dtype() and !(left.dtype() == .f16 and right.dtype() == .bf16)) {
        log.err("left.dtype ({}) != right.dtype ({})", .{ left.dtype(), right.dtype() });
        return error.TestUnexpectedResult;
    }
    if (!left.isContiguous()) {
        log.err("Left input isn't contiguous: {} ({d})", .{ left, left.strides().? });
    }
    if (!right.isContiguous()) {
        log.err("Right input isn't contiguous: {} ({d})", .{ right, right.strides().? });
    }
    switch (left.dtype()) {
        inline .bf16,
        .f16,
        .f32,
        .f64,
        .f8e4m3b11fnuz,
        .f8e4m3fn,
        .f8e4m3fnuz,
        .f8e5m2,
        .f8e5m2fnuz,
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
                            log.err("left.data != right_data.\n < {d:.3} \n > {d:.3}\n  error at idx {d}: {d:.3} != {d:.3}", .{ center(left_data, i), center(right_data, i), i, left_data[i], right_data[i] });
                            return error.TestUnexpectedResult;
                        }
                    }
                },
                else => unreachable,
            }
        },
        inline .bool, .u4, .u8, .u16, .u32, .u64, .i4, .i8, .i16, .i32, .i64 => |t| {
            const T = t.toZigType();
            const left_data = left.items(T);
            const right_data = right.items(T);
            if (!std.mem.eql(T, left_data, right_data)) {
                log.err("left.data ({any}) != right.data ({any})", .{ left_data[0..10], right_data[0..10] });
                return error.TestUnexpectedResult;
            }
        },
        .c64, .c128 => @panic("TODO: support comparison of complex"),
    }
}

pub fn expectEqualShapes(expected: zml.Shape, actual: zml.Shape) error{TestExpectedEqual}!void {
    if (expected.eqlWithTags(actual)) return;

    std.debug.print("Expected {}, got {}", .{ expected, actual });
    return error.TestExpectedEqual;
}

/// Compile a function and immediatly call it with the given buffers.
/// The compiled module is discarded after the call.
/// Useful during testing when a module is typically called only once.
pub fn compileAndCall(platform: zml.Platform, func: anytype, buffer_args: zml.Bufferized(meta.FnParams(func))) !zml.Bufferized(zml.meta.FnResult(func)) {
    // This simplify test API and also ensure this fn isn't used outside of tests.
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const Local = struct {
        pub fn bufferToShape(_: void, x: zml.Buffer) zml.Shape {
            return x.shape();
        }
    };
    var shape_args: zml.ShapeOf(meta.FnParams(func)) = undefined;
    try meta.mapAlloc(Local.bufferToShape, allocator, {}, buffer_args, &shape_args);

    const mod = try zml.compileFn(allocator, func, shape_args, platform);
    defer mod.deinit();

    return mod.call(buffer_args);
}

/// Compile a function and immediatly call it with the given buffers.
/// The compiled module is discarded after the call.
/// Useful during testing when a module is typically called only once.
pub fn compileAndCallWithTensors(platform: zml.Platform, func: anytype, shape_args: zml.ShapeOf(meta.FnParams(func)), buffer_args: zml.Bufferized(meta.FnParams(func))) !zml.Bufferized(zml.meta.FnResult(func)) {
    // This simplify test API and also ensure this fn isn't used outside of tests.
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const mod = try zml.compileFn(allocator, func, shape_args, platform);
    defer mod.deinit();

    return mod.call(buffer_args);
}

pub fn testLayer(
    platform: zml.Platform,
    activations: zml.aio.BufferStore,
    comptime name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    tolerance: f32,
) !void {
    try testLayerOut(platform, activations, name, name ++ ".out", layer, layer_weights, tolerance);
}

pub fn testLayerOut(
    platform: zml.Platform,
    activations: zml.aio.BufferStore,
    comptime name: []const u8,
    comptime out_name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    tolerance: f32,
) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    log.info("Testing {s}", .{name});

    const fwd = @TypeOf(layer).forward;
    const FwdSign = zml.module.ModuleSignature(fwd);

    const input_tensors = try zml.aio.populateModelWithPrefix(FwdSign.ArgsT, alloc, activations, name ++ ".in");
    const input_shapes = try shapesOf(input_tensors, alloc);

    const n_in = zml.module.countTensors(&input_tensors);
    const n_in_exp = activations.countLayers(name ++ ".in");
    if (n_in != n_in_exp) {
        log.warn("Reference models uses {d} inputs, but implementation uses {d}", .{ n_in_exp, n_in });
    }

    const exe = try zml.compileModel(alloc, layer, .forward, input_shapes, platform);

    const n_out_exp = activations.countLayers(out_name);
    if (exe.inner.result_buffer_count != n_out_exp) {
        log.warn("Reference models produces {d} outputs, but implementation produces {d}", .{ n_out_exp, exe.inner.result_buffer_count });
    }
    const mod = try exe.prepare(alloc, layer_weights);

    var input_buffers = try zml.aio.loadModelBuffersWithPrefix(FwdSign.ArgsT, input_tensors, activations, alloc, platform, name ++ ".in");
    defer zml.aio.unloadBuffers(&input_buffers);
    _ = mod.call(input_buffers);

    var buf: [1024]u8 = undefined;
    for (0..mod.inner.result_buffer_count) |i| {
        const full_name = std.fmt.bufPrint(&buf, "{s}.{d}", .{ out_name, i }) catch unreachable;
        const expected_out = activations.get(full_name) orelse {
            log.warn("Output buffer not found: {s}", .{full_name});
            continue;
        };
        zml.testing.expectClose(expected_out, mod.getOutputBuffer(i), tolerance) catch |err| {
            log.err("{s}.{d} doesn't match !", .{ out_name, i });
            return err;
        };
    }

    log.info("all good for {s} !", .{name});
}

pub inline fn expectEqual(expected: anytype, actual: @TypeOf(expected)) !void {
    return std.testing.expectEqual(expected, actual);
}

fn center(slice: anytype, i: usize) @TypeOf(slice) {
    const c = 20;
    const start = if (i < c) 0 else i - c;
    const end = @min(start + 2 * c, slice.len);
    return slice[start..end];
}
