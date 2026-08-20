const std = @import("std");

const zml = @import("zml");
const moe = @import("kimi_k3/moe.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(500_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_grouped_mxfp4_tests --fixture=<primitive-reference.safetensors>
        \\
        \\Run grouped MXFP4 boundary/routing differentials on NVIDIA CUDA only.
        \\
    ;
};

const Inputs = struct {
    input: zml.Tensor,
    packed_values: zml.Tensor,
    scale: zml.Tensor,
};

const Result = struct {
    n1_native: zml.Tensor,
    n1_slow: zml.Tensor,
    n63_native: zml.Tensor,
    n63_slow: zml.Tensor,
    n64_native: zml.Tensor,
    n64_slow: zml.Tensor,
    n65_native: zml.Tensor,
    n65_slow: zml.Tensor,
    route_native: zml.Tensor,
    route_slow: zml.Tensor,
    invalid_native: zml.Tensor,
    invalid_expected: zml.Tensor,
    weighted_native: zml.Tensor,
    weighted_slow: zml.Tensor,
};

fn makeBank(packed_values: zml.Tensor, scale: zml.Tensor, n: i64) moe.Mxfp4Bank {
    const values = packed_values.slice1d(.out, .{ .end = 1 })
        .insertAxes(0, .{.expert})
        .broad(zml.Shape.init(.{ .expert = 5, .out = n, .kw = 16 }, packed_values.dtype()));
    const scales = scale.slice1d(.out, .{ .end = 1 })
        .insertAxes(0, .{.expert})
        .broad(zml.Shape.init(.{ .expert = 5, .out = n, .block = 1 }, scale.dtype()));
    return .{ .values = values, .scale = scales };
}

fn native(input: zml.Tensor, ids: zml.Tensor, bank: moe.Mxfp4Bank) zml.Tensor {
    return moe.nativeBankLinear(input, ids, bank, zml.Shape.toTag(.n)).convert(.f32);
}

fn slow(input: zml.Tensor, ids: zml.Tensor, bank: moe.Mxfp4Bank) zml.Tensor {
    return moe.bankLinear(input, ids, bank).convert(.f32);
}

fn forward(inputs: Inputs) Result {
    const input = inputs.input.convert(.bf16);
    const flat_route = zml.Tensor.arange(.{ .end = 12 }, .i32).withTags(.{.flat_route});
    // Each token deliberately selects duplicate experts; expert 4 stays empty.
    const ids = flat_route.divByConst(2)
        .remainder(zml.Tensor.scalar(4, .i32))
        .reshape(.{ .token = 3, .route = 4 });
    const route_input = input.reshape(.{ .token = 3, .route = 1, .d = 32 })
        .broad(zml.Shape.init(.{ .token = 3, .route = 4, .d = 32 }, .bf16));

    const bank1 = makeBank(inputs.packed_values, inputs.scale, 1);
    const bank63 = makeBank(inputs.packed_values, inputs.scale, 63);
    const bank64 = makeBank(inputs.packed_values, inputs.scale, 64);
    const bank65 = makeBank(inputs.packed_values, inputs.scale, 65);
    const n1_native = native(input, ids, bank1);
    const n1_slow = slow(input, ids, bank1);
    const n63_native = native(input, ids, bank63);
    const n63_slow = slow(input, ids, bank63);
    const n64_native = native(input, ids, bank64);
    const n64_slow = slow(input, ids, bank64);
    const n65_native = native(input, ids, bank65);
    const n65_slow = slow(input, ids, bank65);
    const route_native = native(route_input, ids, bank65);
    const route_slow = slow(route_input, ids, bank65);

    const valid = flat_route.cmp(.LT, zml.Tensor.scalar(11, .i32));
    const invalid_ids = valid.select(
        ids.flatten().withTags(.{.flat_route}),
        zml.Tensor.scalar(5, .i32),
    ).reshape(.{ .token = 3, .route = 4 });
    const invalid_native = native(input, invalid_ids, bank65);
    const invalid_mask = valid.reshape(.{ .token = 3, .route = 4, .one = 1 })
        .broad(invalid_native.shape().withDtype(.bool));
    const invalid_expected = invalid_mask.select(n65_native, zml.Tensor.zeroes(n65_native.shape()));

    const route_weights = zml.Tensor.arange(.{ .end = 12 }, .f32)
        .addConstant(1)
        .scale(0.05)
        .reshape(.{ .token = 3, .route = 4, .one = 1 });
    const weighted_native = n65_native.mul(route_weights.broad(n65_native.shape())).sum(.route).squeeze(.route);
    const weighted_slow = n65_slow.mul(route_weights.broad(n65_slow.shape())).sum(.route).squeeze(.route);

    return .{
        .n1_native = n1_native,
        .n1_slow = n1_slow,
        .n63_native = n63_native,
        .n63_slow = n63_slow,
        .n64_native = n64_native,
        .n64_slow = n64_slow,
        .n65_native = n65_native,
        .n65_slow = n65_slow,
        .route_native = route_native,
        .route_slow = route_slow,
        .invalid_native = invalid_native,
        .invalid_expected = invalid_expected,
        .weighted_native = weighted_native,
        .weighted_slow = weighted_slow,
    };
}

const tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 0.25,
    .relative_tolerance = 0.025,
    .minimum_close_fraction = 1.0,
};

fn compare(
    allocator: std.mem.Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    name: []const u8,
    actual: zml.Buffer,
    expected: zml.Buffer,
    opts: zml.testing.CompareOpts,
) !void {
    var actual_host = try actual.toSliceAlloc(allocator, io);
    defer actual_host.free(allocator);
    var expected_host = try expected.toSliceAlloc(allocator, io);
    defer expected_host.free(allocator);
    const report = try zml.testing.compareSlices(
        allocator,
        f32,
        f32,
        actual_host.items(f32),
        expected_host.items(f32),
        opts,
    );
    // KIMI_K3_TEMP_REMOVE_M20: per-case activation error reports and execution
    // timing are bring-up diagnostics removed after the permanent suite lands.
    try stdout.print("KIMI_K3_GROUPED_MXFP4_CASE name={s} shape={f}\n{f}\n", .{ name, actual.shape(), report });
    try stdout.flush();
    try zml.testing.expectClose(io, actual_host, expected_host, opts);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.35 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    const sharding = platform.replicated_sharding;

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer registry.deinit();
    var tensor_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer tensor_store.deinit();
    const store = tensor_store.view();
    var buffers: zml.Bufferized(Inputs) = .{
        .input = try support.loadBuffer(allocator, io, platform, store, "mxfp4.linear_input", .{ .token, .d }, sharding),
        .packed_values = try support.loadBuffer(allocator, io, platform, store, "mxfp4.packed", .{ .out, .kw }, sharding),
        .scale = try support.loadBuffer(allocator, io, platform, store, "mxfp4.scale_e8m0", .{ .out, .block }, sharding),
    };
    defer zml.Buffer.deinitAll(Inputs, &buffers);
    const tensors: Inputs = .{
        .input = .fromShape(buffers.input.shape()),
        .packed_values = .fromShape(buffers.packed_values.shape()),
        .scale = .fromShape(buffers.scale.shape()),
    };
    const compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    const exe = try platform.compileFn(allocator, io, forward, .{tensors}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    const compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - compile_started, 1000);
    const execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var result = try zml.testing.autoCall(allocator, io, &exe, forward, .{buffers});
    defer zml.Buffer.deinitAll(Result, &result);
    const execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - execute_started, 1000);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    const stdout = &stdout_file.interface;
    try compare(allocator, io, stdout, "n1_partial_k32", result.n1_native, result.n1_slow, tolerance);
    try compare(allocator, io, stdout, "n63_partial_k32", result.n63_native, result.n63_slow, tolerance);
    try compare(allocator, io, stdout, "n64_partial_k32", result.n64_native, result.n64_slow, tolerance);
    try compare(allocator, io, stdout, "n65_partial_k32", result.n65_native, result.n65_slow, tolerance);
    try compare(allocator, io, stdout, "route_input", result.route_native, result.route_slow, tolerance);
    try compare(allocator, io, stdout, "invalid_sentinel_zero", result.invalid_native, result.invalid_expected, .exact_match);
    try compare(allocator, io, stdout, "weighted_reduction", result.weighted_native, result.weighted_slow, tolerance);
    try stdout.print("KIMI_K3_GROUPED_MXFP4_ALL_PASS cases=7 experts=5 empty_experts=1 duplicate_routes=true backend=cuda compile_us={} execute_us={}\n", .{ compile_us, execute_us });
    try stdout.flush();
}
