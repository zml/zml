const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const meta = @import("meta.zig");
const shapesOf = @import("tensor.zig").shapesOf;
const zml = @import("zml.zig");

const log = std.log.scoped(.@"zml/testing");

var _platform: ?zml.Platform = null;

pub fn env() zml.Platform {
    if (!builtin.is_test) @compileError("Cannot use zml.testing.env outside of a test block");
    if (_platform == null) {
        var ctx = zml.Context.init() catch unreachable;
        _platform = ctx.autoPlatform(.{}).withCompilationOptions(.{
            .xla_dump_to = "/tmp/zml/tests/",
            .sharding_enabled = true,
        });
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

/// Testing utility. Accepts both Tensor and HostBuffer but Tensor will be copied to the
/// host for comparison !
pub fn expectClose(left_: anytype, right_: anytype, tolerance: f32) !void {
    const allocator = if (builtin.is_test) std.testing.allocator else std.heap.smp_allocator;
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
    errdefer log.err("\n--> Left: {}\n--> Right: {}", .{ left.pretty(), right.pretty() });

    if (!std.mem.eql(i64, left.shape().dims(), right.shape().dims())) {
        log.err("left.shape() {} != right.shape() {}", .{ left.shape(), right.shape() });
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
            return std.testing.expectEqualSlices(T, left.items(T), right.items(T));
        },
        .c64, .c128 => @panic("TODO: support comparison of complex"),
    }
}

pub fn expectEqualShapes(expected: zml.Shape, actual: zml.Shape) error{TestExpectedEqual}!void {
    if (expected.eqlWithTags(actual)) return;

    std.debug.print("Expected {}, got {}", .{ expected, actual });
    return error.TestExpectedEqual;
}

/// Returns a mirrored version of T where each Tensor has been replaced by a Buffer.
/// This is similar to zml.Bufferized,
/// but also keep every other fields that could be used during compilation.
///
/// see `compileAndCall`.
pub fn BufferizedWithArgs(comptime T: type) type {
    // TODO: we should strip out the non-buffer fields.
    // Currently it's confusing cause the Bufferized struct contains field that are never read.
    // Also it will simplify the layout of the Bufferized struct.
    // accelerating the calls to execute.
    return meta.MapType(zml.Tensor, zml.Buffer).map(T);
}

/// Compile a function and immediatly call it with the given buffers.
/// The compiled module is discarded after the call.
/// Useful during testing when a module is typically called only once.
///
/// Note: `func` needs explicit types on all parameters.
/// To test a function with `anytype` (typically for tagged API), you need to create a specialized version of it with specific types.
pub fn compileAndCall(
    platform: zml.Platform,
    func: anytype,
    buffer_and_args: BufferizedWithArgs(stdx.meta.FnArgs(func)),
) !zml.Bufferized(stdx.meta.FnResult(func)) {
    // This simplify test API and also ensure this fn isn't used outside of tests.
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const Local = struct {
        pub fn bufferToShape(_: void, x: zml.Buffer) zml.Shape {
            return x.shape();
        }
    };
    var shape_args: zml.ShapeOf(stdx.meta.FnArgs(func)) = undefined;
    try meta.mapAlloc(Local.bufferToShape, arena.allocator(), {}, buffer_and_args, &shape_args);

    var mod = try zml.compileFn(allocator, func, shape_args, platform);
    defer mod.deinit();

    // Note: we don't use the type safe API of mod,
    // cause mod.call expects a `zml.Bufferized` while we have `BufferizedWithArgs`.
    mod.inner.prepare(buffer_and_args);
    mod.inner._unsafeCall();

    var result: zml.Bufferized(stdx.meta.FnResult(func)) = undefined;
    mod.inner._unsafeAssignResults(@TypeOf(result), &result);
    return result;
}

/// Compile a function and immediatly call it with the given buffers.
/// The compiled module is discarded after the call.
/// Useful during testing when a module is typically called only once.
pub fn compileAndCallWithTensors(
    platform: zml.Platform,
    func: anytype,
    shape_args: zml.ShapeOf(stdx.meta.FnArgs(func)),
    buffer_args: zml.Bufferized(stdx.meta.FnArgs(func)),
) !zml.Bufferized(stdx.meta.FnResult(func)) {
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
    const FwdSign = zml.ModuleSignature(fwd);

    const input_tensors = try zml.aio.populateModelWithPrefix(FwdSign.ArgsT, alloc, activations, name ++ ".in");
    const input_shapes = try shapesOf(input_tensors, alloc);

    const n_in = zml.module.countTensors(&input_tensors);
    const n_in_exp = activations.countLayers(name ++ ".in");
    if (n_in != n_in_exp) {
        log.warn("Reference models uses {d} inputs, but implementation uses {d}", .{ n_in_exp, n_in });
    }

    const exe = try zml.compileModel(alloc, fwd, layer, input_shapes, platform);

    const n_out_exp = activations.countLayers(out_name);
    if (exe.inner.result_shapes.len != n_out_exp) {
        log.warn("Reference models produces {d} outputs, but implementation produces {d}", .{ n_out_exp, exe.inner.result_shapes.len });
    }
    const mod = exe.prepare(layer_weights);

    const FetchCtx = struct {
        store: zml.aio.BufferStore,
        index: u32,
        prefix: std.ArrayListUnmanaged(u8),
        platform: zml.Platform,

        fn fetch(ctx: *@This(), x: zml.Tensor) zml.Buffer {
            _ = x;
            defer ctx.index += 1;
            var full_prefix = ctx.*.prefix;
            _ = full_prefix.writer(undefined).print("{d}", .{ctx.index}) catch unreachable;
            log.info("prefix: {s}", .{full_prefix.items});
            const host = ctx.store.get(full_prefix.items) orelse {
                log.err("Didn't find test input: {s}", .{full_prefix.items});
                @panic("Missing test input");
            };
            return host.toDevice(ctx.platform) catch unreachable;
        }
    };

    // Note: zml.populateModelWithPrefix isn't enough,
    // because it assumes we have the same structure in the activation file
    // than in the function signature.
    // But for sake of decoupling the reference implementation
    // and ZML code that's not always the case.
    {
        var input_buffers: zml.Bufferized(FwdSign.ArgsT) = undefined;
        var fetch_ctx: FetchCtx = .{ .store = activations, .index = 0, .prefix = .{}, .platform = platform };
        try fetch_ctx.prefix.ensureTotalCapacity(alloc, name.len + 32);
        fetch_ctx.prefix.appendSliceAssumeCapacity(name ++ ".in.");
        try zml.meta.mapAlloc(FetchCtx.fetch, alloc, &fetch_ctx, input_tensors, &input_buffers);
        defer zml.aio.unloadBuffers(&input_buffers);
        _ = mod.call(input_buffers);
    }

    var buf: [1024]u8 = undefined;
    var failed: bool = false;
    for (0..mod.inner.result_shapes.len) |i| {
        const full_name = std.fmt.bufPrint(&buf, "{s}.{d}", .{ out_name, i }) catch unreachable;
        const expected_out = activations.get(full_name) orelse {
            log.warn("Output buffer not found: {s}", .{full_name});
            continue;
        };
        zml.testing.expectClose(expected_out, mod.inner.getOutputBuffer(i), tolerance) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                log.err("{s}.{d} doesn't match !", .{ out_name, i });
                failed = true;
                continue;
            },
            else => return err,
        };
    }

    if (failed) return error.TestUnexpectedResult;
    log.info("all good for {s} !", .{name});
}

test testLayer {
    const platform = env();

    // create a model
    const layer: zml.nn.Linear = .{
        .weight = zml.Tensor{ ._shape = zml.Shape.init(.{ 5, 2 }, .f32), ._id = .{ .buffer_id = 42 } },
    };
    const layer_weights: zml.Bufferized(zml.nn.Linear) = .{
        .weight = try zml.Buffer.fromArray(
            platform,
            [5][2]f32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 2 }, .{ -1, -1 }, .{ -1, 0 } },
        ),
    };

    // create a buffer store containing the activations:
    var activations = try zml.aio.BufferStore.init(std.testing.allocator, &.{});
    defer activations.deinit();
    {
        const input = zml.HostBuffer.fromArray(&[2]f32{ 1, -1 });
        try activations.buffers.put(activations.arena.allocator(), "model.layer.in.0", input);
        const output = zml.HostBuffer.fromArray(&[5]f32{ 0, -1, -1, 0, -1 });
        try activations.buffers.put(activations.arena.allocator(), "model.layer.out.0", output);
    }

    // test the ZML layer reproduces the "captured" activations:
    try zml.testing.testLayer(platform, activations, "model.layer", layer, layer_weights, 1e-5);
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
