const std = @import("std");

const zml = @import("zml");

fn normalizeOcpEncoding(weight: zml.Tensor) zml.Tensor {
    return zml.fp8.normalizeOcpEncodingForFnuz(weight);
}

test "OCP FP8 exceptional encodings are safe to bitcast to FNUZ" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;

    const weight: zml.Tensor = .init(.{ .n = 6 }, .f8e4m3fn);
    var exe = try platform.compileFn(allocator, io, normalizeOcpEncoding, .{weight}, .{});
    defer exe.deinit();

    const input_bytes = [_]u8{ 0x00, 0x80, 0x7f, 0xff, 0x38, 0xb8 };
    const expected_bytes = [_]u8{ 0x00, 0x00, 0x80, 0x80, 0x38, 0xb8 };
    var input: zml.Buffer = try .fromBytes(io, platform, weight.shape(), .replicated, &input_bytes);
    defer input.deinit();
    var output = try zml.testing.autoCall(allocator, io, &exe, normalizeOcpEncoding, .{input});
    defer output.deinit();
    var output_host = try output.toSliceAlloc(allocator, io);
    defer output_host.free(allocator);
    try std.testing.expectEqualSlices(u8, &expected_bytes, output_host.constData());
}

fn blockDot(x: zml.Tensor, weight: zml.Tensor, weight_scale: zml.Tensor) zml.Tensor {
    const output_shape = zml.Shape.init(.{ .m = x.dim(0), .n = weight.dim(0) }, .bf16);
    return zml.fp8.nativeBlockScaledDot(x, weight, weight_scale, output_shape);
}

fn blockDotError(x: zml.Tensor, weight: zml.Tensor, weight_scale: zml.Tensor) zml.Tensor {
    const actual = zml.nn.scaledDot(x, weight, null, weight_scale, .k);
    const expanded_scale = weight_scale
        .stutter(&.{ 128, 128 })
        .slice(0, .{ .end = weight.dim(0) })
        .slice(1, .{ .end = weight.dim(1) })
        .withTags(.{ .n, .k });
    const reference_weight = weight.convert(.bf16).mul(expanded_scale.convert(.bf16));
    const reference = x.dot(reference_weight, .k).convert(.bf16);
    return actual.sub(reference);
}

const PreparedQuantizerOutputs = struct {
    routed_q: zml.Tensor,
    routed_scale: zml.Tensor,
    shared_q: zml.Tensor,
    shared_scale: zml.Tensor,
};

fn comparePreparedAndOldSharedQuantizers(x: zml.Tensor) PreparedQuantizerOutputs {
    const prepared = zml.moe.triton.prepareBlock128Fp8Activation(x, false);
    const grouped = x.convert(.f32).splitAxis(.k, .{
        .fp8_ks = -1,
        .fp8_block = 128,
    });
    const reference_scale = grouped.abs().max(.fp8_block)
        .maximum(.scalar(1e-10, .f32))
        .scale(1.0 / 448.0);
    const reference_q = grouped.div(reference_scale.broad(grouped.shape()))
        .clamp(.scalar(-448.0, .f32), .scalar(448.0, .f32))
        .convert(.f8e4m3fn)
        .reshape(prepared.q.shape());
    return .{
        .routed_q = prepared.q,
        .routed_scale = prepared.scale,
        .shared_q = reference_q,
        .shared_scale = reference_scale.squeeze(.fp8_block).reshape(prepared.scale.shape()),
    };
}

test "routed block FP8 preparation bitwise matches the old CUDA shared quantizer" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const m = 7;
    const k = 256;
    const x: zml.Tensor = .init(.{ .m = m, .k = k }, .bf16);
    var exe = try platform.compileFn(allocator, io, comparePreparedAndOldSharedQuantizers, .{x}, .{});
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    const x_host = try allocator.alloc(BFloat16, x.shape().count());
    defer allocator.free(x_host);
    for (x_host, 0..) |*value, i| {
        const centered: i32 = @intCast(i % 251);
        value.* = BFloat16.fromF32(@as(f32, @floatFromInt(centered - 125)) / 37.0);
    }
    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(x_host));
    defer x_buffer.deinit();
    var output = try zml.testing.autoCall(allocator, io, &exe, comparePreparedAndOldSharedQuantizers, .{x_buffer});
    defer zml.Buffer.deinitAll(PreparedQuantizerOutputs, &output);
    var routed_q = try output.routed_q.toSliceAlloc(allocator, io);
    defer routed_q.free(allocator);
    var routed_scale = try output.routed_scale.toSliceAlloc(allocator, io);
    defer routed_scale.free(allocator);
    var shared_q = try output.shared_q.toSliceAlloc(allocator, io);
    defer shared_q.free(allocator);
    var shared_scale = try output.shared_scale.toSliceAlloc(allocator, io);
    defer shared_scale.free(allocator);
    try std.testing.expectEqualSlices(u8, routed_q.constData(), shared_q.constData());
    try std.testing.expectEqualSlices(u8, routed_scale.constData(), shared_scale.constData());
}

fn preparedBlockDotMatchesOldSharedQuantizer(
    x: zml.Tensor,
    weight: zml.Tensor,
    weight_scale: zml.Tensor,
) zml.Tensor {
    const comparison_shape = zml.Shape.init(.{ .m = x.dim(0), .n = weight.dim(0) }, .bool);
    return zml.ops.manualComputation(
        .{ x, weight, weight_scale },
        comparison_shape,
        {},
        (struct {
            fn body(_: void, _: std.mem.Allocator, inputs: []const zml.Tensor, local_output: zml.Shape) zml.Tensor {
                const output_shape = local_output.withDtype(.bf16);
                const old = zml.fp8.nativeBlockScaledDotLocal(inputs[0], inputs[1], inputs[2], output_shape);
                const prepared = zml.moe.triton.prepareBlock128Fp8Activation(
                    inputs[0],
                    zml.Compiler.current().platform.target == .rocm,
                );
                const reused = zml.fp8.nativeBlockScaledDotPreparedLocal(prepared, inputs[1], inputs[2], output_shape);
                return old.bitCast(.u16).cmp(.EQ, reused.bitCast(.u16));
            }
        }).body,
    );
}

test "prepared routed activation matches the old shared FP8 quantizer for finite BF16 inputs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda and platform.target != .rocm) return error.SkipZigTest;

    const m = 7;
    const n = 256;
    const k = 256;
    const x: zml.Tensor = .init(.{ .m = m, .k = k }, .bf16);
    const weight: zml.Tensor = .init(.{ .n = n, .k = k }, .f8e4m3fn);
    const weight_scale: zml.Tensor = .init(.{ .nb = n / 128, .kb = k / 128 }, .f32);
    var exe = try platform.compileFn(
        allocator,
        io,
        preparedBlockDotMatchesOldSharedQuantizer,
        .{ x, weight, weight_scale },
        .{},
    );
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    const Float8 = zml.floats.Float8E4M3FN;
    const x_host = try allocator.alloc(BFloat16, x.shape().count());
    defer allocator.free(x_host);
    for (x_host, 0..) |*value, i| {
        const centered: i32 = @intCast(i % 251);
        value.* = BFloat16.fromF32(@as(f32, @floatFromInt(centered - 125)) / 37.0);
    }
    const weight_host = try allocator.alloc(Float8, weight.shape().count());
    defer allocator.free(weight_host);
    for (weight_host, 0..) |*value, i| {
        const centered: i32 = @intCast(i % 13);
        value.* = Float8.fromF32(@as(f32, @floatFromInt(centered - 6)) / 4.0);
    }
    const scale_host = [_][2]f32{
        .{ 0.25, 1.5 },
        .{ 2.0, 0.75 },
    };

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(x_host));
    defer x_buffer.deinit();
    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), .replicated, std.mem.sliceAsBytes(weight_host));
    defer weight_buffer.deinit();
    var scale_buffer: zml.Buffer = try .fromBytes(io, platform, weight_scale.shape(), .replicated, std.mem.asBytes(&scale_host));
    defer scale_buffer.deinit();
    var output = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        preparedBlockDotMatchesOldSharedQuantizer,
        .{ x_buffer, weight_buffer, scale_buffer },
    );
    defer output.deinit();
    var output_host = try output.toSliceAlloc(allocator, io);
    defer output_host.free(allocator);
    for (output_host.constData()) |equal| try std.testing.expectEqual(@as(u8, 1), equal);
}

fn testCudaBlockDotCase(m: i64, n: i64, k: i64) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const x: zml.Tensor = .init(.{ .m = m, .k = k }, .bf16);
    const weight: zml.Tensor = .init(.{ .n = n, .k = k }, .f8e4m3fn);
    const weight_scale: zml.Tensor = .init(.{
        .nb = std.math.divCeil(i64, n, 128) catch unreachable,
        .kb = std.math.divCeil(i64, k, 128) catch unreachable,
    }, .f32);

    var exe = try platform.compileFn(allocator, io, blockDotError, .{ x, weight, weight_scale }, .{});
    defer exe.deinit();

    const x_host = try allocator.alloc(zml.floats.BFloat16, x.shape().count());
    defer allocator.free(x_host);
    for (x_host, 0..) |*value, i| {
        const v: f32 = if ((i % @as(usize, @intCast(k))) % 128 < 64) 0.25 else 1.0;
        value.* = zml.floats.BFloat16.fromF32(v);
    }

    const weight_host = try allocator.alloc(zml.floats.Float8E4M3FN, weight.shape().count());
    defer allocator.free(weight_host);
    for (weight_host, 0..) |*value, i| {
        const v: f32 = if ((i / @as(usize, @intCast(k))) % 2 == 0) 1.0 else -0.5;
        value.* = zml.floats.Float8E4M3FN.fromF32(v);
    }

    const scale_host = try allocator.alloc(f32, weight_scale.shape().count());
    defer allocator.free(scale_host);
    const scale_values = [_]f32{ 0.25, 0.5, 1.0, 2.0 };
    for (scale_host, 0..) |*value, i| value.* = scale_values[i % scale_values.len];

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.sliceAsBytes(x_host));
    defer x_buffer.deinit();
    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), .replicated, std.mem.sliceAsBytes(weight_host));
    defer weight_buffer.deinit();
    var scale_buffer: zml.Buffer = try .fromBytes(io, platform, weight_scale.shape(), .replicated, std.mem.sliceAsBytes(scale_host));
    defer scale_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, blockDotError, .{ x_buffer, weight_buffer, scale_buffer });
    defer output.deinit();

    const expected_host = try allocator.alloc(zml.floats.BFloat16, output.shape().count());
    defer allocator.free(expected_host);
    @memset(expected_host, zml.floats.BFloat16.fromF32(0.0));
    const expected: zml.Slice = .init(output.shape(), std.mem.sliceAsBytes(expected_host));
    try zml.testing.expectClose(io, expected, output, .{ .absolute_tolerance = 4.0, .relative_tolerance = 0.01 });
}

test "CUDA XLA block-scaled E4M3FN GEMM decode and prefill" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    // XLA's block-128 W8A8 arm requires complete N and K tiles. Cover its
    // decode, batched-decode, prefill and long-contraction geometries.
    try testCudaBlockDotCase(1, 128, 128);
    try testCudaBlockDotCase(1, 256, 256);
    try testCudaBlockDotCase(16, 256, 256);
    try testCudaBlockDotCase(65, 256, 256);
    try testCudaBlockDotCase(1, 256, 2048);
    try testCudaBlockDotCase(64, 256, 2048);
}

const ShardedBlockDotOutputs = struct {
    column: zml.Tensor,
    row: zml.Tensor,
};

fn blockDotsSharded(
    x: zml.Tensor,
    column_weight: zml.Tensor,
    column_scale: zml.Tensor,
    row_weight: zml.Tensor,
    row_scale: zml.Tensor,
) ShardedBlockDotOutputs {
    return .{
        .column = zml.nn.scaledDot(
            x.withPartitioning(.{ .m = .replicated, .k = .replicated }),
            column_weight.withPartitioning(.{ .n = .model, .k = .replicated }),
            null,
            column_scale.withPartitioning(.{ .nb = .model, .kb = .replicated }),
            .k,
        ),
        .row = zml.nn.scaledDot(
            x.withPartitioning(.{ .m = .replicated, .k = .model }),
            row_weight.withPartitioning(.{ .n = .replicated, .k = .model }),
            null,
            row_scale.withPartitioning(.{ .nb = .replicated, .kb = .model }),
            .k,
        ),
    };
}

test "CUDA XLA block-scaled FP8 compiles with model sharding" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const width: i64 = @intCast(128 * platform.devices.len);
    const blocks = @divExact(width, 128);
    const x: zml.Tensor = .init(.{ .m = 16, .k = width }, .bf16);
    const column_weight: zml.Tensor = .init(.{ .n = width, .k = width }, .f8e4m3fn);
    const column_scale: zml.Tensor = .init(.{ .nb = blocks, .kb = blocks }, .f32);
    const row_weight: zml.Tensor = .init(.{ .n = width, .k = width }, .f8e4m3fn);
    const row_scale: zml.Tensor = .init(.{ .nb = blocks, .kb = blocks }, .f32);

    const model_sharding = try @constCast(platform).registerSharding("fp8_test_model", .mesh(.{ .model = .high_bandwidth }));
    var exe = try platform.compileFn(
        allocator,
        io,
        blockDotsSharded,
        .{ x, column_weight, column_scale, row_weight, row_scale },
        .{ .shardings = &.{model_sharding} },
    );
    defer exe.deinit();
}

test "ROCm Triton block-scaled FP8 GEMM" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;

    const m = 1;
    // This shape exercises the small-M split-K path as well as the final
    // FP32 partial reduction.
    const n = 128;
    const k = 2048;
    const x: zml.Tensor = .init(.{ .m = m, .k = k }, .bf16);
    const weight: zml.Tensor = .init(.{ .n = n, .k = k }, .f8e4m3fn);
    const weight_scale: zml.Tensor = .init(.{ .nb = n / 128, .kb = k / 128 }, .f32);

    var exe = try platform.compileFn(allocator, io, blockDot, .{ x, weight, weight_scale }, .{});
    defer exe.deinit();

    const one_bf16 = zml.floats.BFloat16.fromF32(1.0);
    const one_fp8 = zml.floats.Float8E4M3FN.fromF32(1.0);
    const x_host: [m][k]zml.floats.BFloat16 = @splat(@splat(one_bf16));
    const weight_host: [n][k]zml.floats.Float8E4M3FN = @splat(@splat(one_fp8));
    // llmd performs this FN -> FNUZ scale conversion once while loading.
    const scale_host: [n / 128][k / 128]f32 = @splat(@splat(2.0));

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x.shape(), .replicated, std.mem.asBytes(&x_host));
    defer x_buffer.deinit();
    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), .replicated, std.mem.asBytes(&weight_host));
    defer weight_buffer.deinit();
    var scale_buffer: zml.Buffer = try .fromBytes(io, platform, weight_scale.shape(), .replicated, std.mem.asBytes(&scale_host));
    defer scale_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, blockDot, .{ x_buffer, weight_buffer, scale_buffer });
    defer output.deinit();

    const expected_host: [m][n]zml.floats.BFloat16 = @splat(@splat(zml.floats.BFloat16.fromF32(k)));
    const expected: zml.Slice = .init(zml.Shape.init(.{ .m = m, .n = n }, .bf16), std.mem.asBytes(&expected_host));
    try zml.testing.expectClose(io, expected, output, .{ .absolute_tolerance = 1.0, .relative_tolerance = 0.01 });
}

fn absorbedDots(q: zml.Tensor, latent: zml.Tensor, weight: zml.Tensor, weight_scale: zml.Tensor) zml.Tensor {
    const key_dim: usize = @intCast(q.dim(2));
    const value_dim: usize = 128;
    const key_shape = q.shape().setDim(2, latent.dim(2)).setTag(2, .latent).withDtype(.bf16);
    const value_shape = latent.shape().setDim(2, @intCast(value_dim)).setTag(2, .value).withDtype(.bf16);
    const key = zml.fp8.rocmAbsorbedKeyDot(q, weight, weight_scale, key_dim, value_dim, key_shape);
    const value = zml.fp8.rocmAbsorbedValueDot(latent, weight, weight_scale, key_dim, value_dim, value_shape);
    return key.slice(.latent, .{ .end = @intCast(value_dim) }).rename(.{ .latent = .value }).add(value);
}

fn absorbedDotsSharded(q: zml.Tensor, latent: zml.Tensor, weight: zml.Tensor, weight_scale: zml.Tensor) zml.Tensor {
    return absorbedDots(
        q.withPartitioning(.{ .m = .replicated, .h = .model, .key = .replicated }),
        latent.withPartitioning(.{ .m = .replicated, .h = .model, .latent = .replicated }),
        weight.withPartitioning(.{ .n = .model, .k = .replicated }),
        weight_scale.withPartitioning(.{ .nb = .model, .kb = .replicated }),
    );
}

test "ROCm GLM absorbed block-scaled FP8 projections compile with model sharding" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;

    const heads = 64;
    const key_dim = 192;
    const value_dim = 128;
    const latent_dim = 512;
    const q: zml.Tensor = .init(.{ .m = 1, .h = heads, .key = key_dim }, .bf16);
    const latent: zml.Tensor = .init(.{ .m = 1, .h = heads, .latent = latent_dim }, .bf16);
    const weight: zml.Tensor = .init(.{ .n = heads * (key_dim + value_dim), .k = latent_dim }, .f8e4m3fn);
    const weight_scale: zml.Tensor = .init(.{ .nb = (heads * (key_dim + value_dim)) / 128, .kb = latent_dim / 128 }, .f32);

    const model_sharding = try @constCast(platform).registerSharding("fp8_test_model", .mesh(.{ .model = .high_bandwidth }));
    var exe = try platform.compileFn(
        allocator,
        io,
        absorbedDotsSharded,
        .{ q, latent, weight, weight_scale },
        .{ .shardings = &.{model_sharding} },
    );
    defer exe.deinit();
}

test "ROCm GLM absorbed block-scaled FP8 projections" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;

    const m = 1;
    const heads = 2;
    const key_dim = 192;
    const value_dim = 128;
    const latent_dim = 512;
    const q: zml.Tensor = .init(.{ .m = m, .h = heads, .key = key_dim }, .bf16);
    const latent: zml.Tensor = .init(.{ .m = m, .h = heads, .latent = latent_dim }, .bf16);
    const weight: zml.Tensor = .init(.{ .n = heads * (key_dim + value_dim), .k = latent_dim }, .f8e4m3fn);
    const weight_scale: zml.Tensor = .init(.{ .nb = (heads * (key_dim + value_dim)) / 128, .kb = latent_dim / 128 }, .f32);

    var exe = try platform.compileFn(allocator, io, absorbedDots, .{ q, latent, weight, weight_scale }, .{});
    defer exe.deinit();

    const one_bf16 = zml.floats.BFloat16.fromF32(1.0);
    const one_fp8 = zml.floats.Float8E4M3FN.fromF32(1.0);
    const q_host = try allocator.alloc(zml.floats.BFloat16, q.shape().count());
    defer allocator.free(q_host);
    @memset(q_host, one_bf16);
    const latent_host = try allocator.alloc(zml.floats.BFloat16, latent.shape().count());
    defer allocator.free(latent_host);
    @memset(latent_host, one_bf16);
    const weight_host = try allocator.alloc(zml.floats.Float8E4M3FN, weight.shape().count());
    defer allocator.free(weight_host);
    @memset(weight_host, one_fp8);
    const scale_host = try allocator.alloc(f32, weight_scale.shape().count());
    defer allocator.free(scale_host);
    @memset(scale_host, 2.0);

    var q_buffer: zml.Buffer = try .fromBytes(io, platform, q.shape(), .replicated, std.mem.sliceAsBytes(q_host));
    defer q_buffer.deinit();
    var latent_buffer: zml.Buffer = try .fromBytes(io, platform, latent.shape(), .replicated, std.mem.sliceAsBytes(latent_host));
    defer latent_buffer.deinit();
    var weight_buffer: zml.Buffer = try .fromBytes(io, platform, weight.shape(), .replicated, std.mem.sliceAsBytes(weight_host));
    defer weight_buffer.deinit();
    var scale_buffer: zml.Buffer = try .fromBytes(io, platform, weight_scale.shape(), .replicated, std.mem.sliceAsBytes(scale_host));
    defer scale_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, absorbedDots, .{ q_buffer, latent_buffer, weight_buffer, scale_buffer });
    defer output.deinit();

    const expected_value = @as(f32, key_dim + latent_dim);
    const expected_host = try allocator.alloc(zml.floats.BFloat16, output.shape().count());
    defer allocator.free(expected_host);
    @memset(expected_host, zml.floats.BFloat16.fromF32(expected_value));
    try zml.testing.expectClose(
        io,
        zml.Slice.init(output.shape(), std.mem.sliceAsBytes(expected_host)),
        output,
        .{ .absolute_tolerance = 8.0, .relative_tolerance = 0.02 },
    );
}

fn fusedMoe(
    hidden: zml.Tensor,
    w1: zml.Tensor,
    w1_scale: zml.Tensor,
    w2: zml.Tensor,
    w2_scale: zml.Tensor,
    topk_weight: zml.Tensor,
    topk_id: zml.Tensor,
) zml.Tensor {
    const global_num_experts = 288;
    const global_expert_ids = zml.Tensor.arange(.{ .end = global_num_experts }, .i32).withTags(.{.expert});
    const expert_map = global_expert_ids.cmp(.LT, zml.Tensor.scalar(w1.dim(.expert), .i32)).select(
        global_expert_ids,
        zml.Tensor.scalar(-1, .i32),
    );
    return zml.moe.triton.fusedExpertsImpl(
        hidden,
        w1,
        w2,
        topk_weight,
        topk_id,
        .{},
        .{
            .activation = .silu,
            .activation_threshold = 7,
            .global_num_experts = global_num_experts,
            .expert_map = expert_map,
            .w1_scale = w1_scale,
            .w2_scale = w2_scale,
        },
    ) catch unreachable;
}

fn reduceExpertRoutes(routes: zml.Tensor, weights: zml.Tensor) zml.Tensor {
    return zml.moe.triton.reduceExpertRoutes(routes, weights);
}

fn reduceExpertRoutesTop8NoMap(routes: zml.Tensor, weights: zml.Tensor, ids: zml.Tensor) zml.Tensor {
    return zml.moe.triton.reduceExpertRoutesTop8(routes, weights, ids, null, 8, 8);
}

fn reduceExpertRoutesTop8WithMap(routes: zml.Tensor, weights: zml.Tensor, ids: zml.Tensor, expert_map: zml.Tensor) zml.Tensor {
    return zml.moe.triton.reduceExpertRoutesTop8(routes, weights, ids, expert_map, 4, 2);
}

test "Triton MoE applies router weights and reduces routes in FP32" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda and platform.target != .rocm) return error.SkipZigTest;

    const routes: zml.Tensor = .init(.{ .token = 1, .topk = 2, .out = 1 }, .bf16);
    const weights: zml.Tensor = .init(.{ .token = 1, .topk = 2 }, .f32);
    var exe = try platform.compileFn(allocator, io, reduceExpertRoutes, .{ routes, weights }, .{});
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    const one = BFloat16.fromF32(1.0);
    const routes_host = [2]BFloat16{ one, one };
    const weights_host = [2]f32{ 0.001, 0.3 };
    var routes_buffer: zml.Buffer = try .fromBytes(io, platform, routes.shape(), .replicated, std.mem.asBytes(&routes_host));
    defer routes_buffer.deinit();
    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), .replicated, std.mem.asBytes(&weights_host));
    defer weights_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, reduceExpertRoutes, .{ routes_buffer, weights_buffer });
    defer output.deinit();
    var output_host = try output.toSliceAlloc(allocator, io);
    defer output_host.free(allocator);

    const expected = BFloat16.fromF32(weights_host[0] + weights_host[1]);
    const prematurely_rounded = BFloat16.fromF32(
        BFloat16.fromF32(weights_host[0]).toF32() +
            BFloat16.fromF32(weights_host[1]).toF32(),
    );
    try std.testing.expect(@as(u16, @bitCast(expected)) != @as(u16, @bitCast(prematurely_rounded)));
    try std.testing.expectEqual(
        @as(u16, @bitCast(expected)),
        @as(u16, @bitCast(output_host.items(BFloat16)[0])),
    );
}

test "CUDA Triton MoE top-8 route reduction preserves sequential FP32 association" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const hidden_size = 4096;
    const routes: zml.Tensor = .init(.{ .token = 1, .topk = 8, .out = hidden_size }, .bf16);
    const weights: zml.Tensor = .init(.{ .token = 1, .topk = 8 }, .f32);
    const ids: zml.Tensor = .init(.{ .token = 1, .topk = 8 }, .i32);
    var exe = try platform.compileFn(allocator, io, reduceExpertRoutesTop8NoMap, .{ routes, weights, ids }, .{});
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    // With weights of 1/8 these become [2^30, 1, -2^30, 1, ...].
    // A sequential FP32 fold produces 5, while a balanced reduction can
    // produce 4 by associating the two large cancellation branches last.
    const route_values = [8]f32{ 8589934592.0, 8.0, -8589934592.0, 8.0, 8.0, 8.0, 8.0, 8.0 };
    const weights_host: [8]f32 = @splat(0.125);
    const ids_host = [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const routes_host = try allocator.alloc(BFloat16, routes.shape().count());
    defer allocator.free(routes_host);
    for (route_values, 0..) |value, route| {
        @memset(routes_host[route * hidden_size .. (route + 1) * hidden_size], BFloat16.fromF32(value));
    }

    var routes_buffer: zml.Buffer = try .fromBytes(io, platform, routes.shape(), .replicated, std.mem.sliceAsBytes(routes_host));
    defer routes_buffer.deinit();
    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), .replicated, std.mem.asBytes(&weights_host));
    defer weights_buffer.deinit();
    var ids_buffer: zml.Buffer = try .fromBytes(io, platform, ids.shape(), .replicated, std.mem.asBytes(&ids_host));
    defer ids_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, reduceExpertRoutesTop8NoMap, .{ routes_buffer, weights_buffer, ids_buffer });
    defer output.deinit();
    var output_host = try output.toSliceAlloc(allocator, io);
    defer output_host.free(allocator);

    const expected_bits: u16 = @bitCast(BFloat16.fromF32(5.0));
    for (output_host.items(BFloat16)) |value| {
        try std.testing.expectEqual(expected_bits, @as(u16, @bitCast(value)));
    }
}

test "CUDA Triton MoE top-8 route reduction preserves FP32 router weights" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const hidden_size = 1024;
    const routes: zml.Tensor = .init(.{ .token = 1, .topk = 8, .out = hidden_size }, .bf16);
    const weights: zml.Tensor = .init(.{ .token = 1, .topk = 8 }, .f32);
    const ids: zml.Tensor = .init(.{ .token = 1, .topk = 8 }, .i32);
    var exe = try platform.compileFn(allocator, io, reduceExpertRoutesTop8NoMap, .{ routes, weights, ids }, .{});
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    const routes_host = try allocator.alloc(BFloat16, routes.shape().count());
    defer allocator.free(routes_host);
    @memset(routes_host, BFloat16.fromF32(1.0));
    const weights_host = [8]f32{ 0.001, 0.3, 0, 0, 0, 0, 0, 0 };
    const ids_host = [8]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };

    var routes_buffer: zml.Buffer = try .fromBytes(io, platform, routes.shape(), .replicated, std.mem.sliceAsBytes(routes_host));
    defer routes_buffer.deinit();
    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), .replicated, std.mem.asBytes(&weights_host));
    defer weights_buffer.deinit();
    var ids_buffer: zml.Buffer = try .fromBytes(io, platform, ids.shape(), .replicated, std.mem.asBytes(&ids_host));
    defer ids_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, reduceExpertRoutesTop8NoMap, .{ routes_buffer, weights_buffer, ids_buffer });
    defer output.deinit();
    var output_host = try output.toSliceAlloc(allocator, io);
    defer output_host.free(allocator);

    const expected = BFloat16.fromF32(weights_host[0] + weights_host[1]);
    const prematurely_rounded = BFloat16.fromF32(
        BFloat16.fromF32(weights_host[0]).toF32() +
            BFloat16.fromF32(weights_host[1]).toF32(),
    );
    const expected_bits: u16 = @bitCast(expected);
    try std.testing.expect(expected_bits != @as(u16, @bitCast(prematurely_rounded)));
    for (output_host.items(BFloat16)) |value| {
        try std.testing.expectEqual(expected_bits, @as(u16, @bitCast(value)));
    }
}

test "CUDA Triton MoE top-8 route reduction skips invalid non-finite routes" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const hidden_size = 1024;
    const routes: zml.Tensor = .init(.{ .token = 1, .topk = 8, .out = hidden_size }, .bf16);
    const weights: zml.Tensor = .init(.{ .token = 1, .topk = 8 }, .f32);
    const ids: zml.Tensor = .init(.{ .token = 1, .topk = 8 }, .i32);
    const expert_map: zml.Tensor = .init(.{ .expert = 4 }, .i32);
    var exe = try platform.compileFn(
        allocator,
        io,
        reduceExpertRoutesTop8WithMap,
        .{ routes, weights, ids, expert_map },
        .{},
    );
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    const routes_host = try allocator.alloc(BFloat16, routes.shape().count());
    defer allocator.free(routes_host);
    @memset(routes_host, BFloat16.fromF32(std.math.nan(f32)));
    @memset(routes_host[0 * hidden_size .. 1 * hidden_size], BFloat16.fromF32(2.0));
    @memset(routes_host[4 * hidden_size .. 5 * hidden_size], BFloat16.fromF32(4.0));
    const weights_host = [8]f32{
        0.25,
        std.math.nan(f32),
        std.math.inf(f32),
        -std.math.inf(f32),
        0.125,
        std.math.nan(f32),
        std.math.inf(f32),
        -std.math.inf(f32),
    };
    // Expert 1 maps to -1 and expert 3 maps past the local range, while -1,
    // -2, 4, and 99 exercise globally invalid ids without an OOB map lookup.
    const ids_host = [8]i32{ 0, 1, -1, 4, 2, 3, -2, 99 };
    const expert_map_host = [4]i32{ 0, -1, 1, 2 };

    var routes_buffer: zml.Buffer = try .fromBytes(io, platform, routes.shape(), .replicated, std.mem.sliceAsBytes(routes_host));
    defer routes_buffer.deinit();
    var weights_buffer: zml.Buffer = try .fromBytes(io, platform, weights.shape(), .replicated, std.mem.asBytes(&weights_host));
    defer weights_buffer.deinit();
    var ids_buffer: zml.Buffer = try .fromBytes(io, platform, ids.shape(), .replicated, std.mem.asBytes(&ids_host));
    defer ids_buffer.deinit();
    var expert_map_buffer: zml.Buffer = try .fromBytes(io, platform, expert_map.shape(), .replicated, std.mem.asBytes(&expert_map_host));
    defer expert_map_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, reduceExpertRoutesTop8WithMap, .{
        routes_buffer,
        weights_buffer,
        ids_buffer,
        expert_map_buffer,
    });
    defer output.deinit();
    var output_host = try output.toSliceAlloc(allocator, io);
    defer output_host.free(allocator);

    const expected_bits: u16 = @bitCast(BFloat16.fromF32(1.0));
    for (output_host.items(BFloat16)) |value| {
        try std.testing.expectEqual(expected_bits, @as(u16, @bitCast(value)));
    }
}

fn testFusedMoeBlockFp8(comptime tokens: usize) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda and platform.target != .rocm) return error.SkipZigTest;

    // Only the first 32 expert weights are needed by these routes;
    // global_num_experts retains GLM-5.3's 288-expert assignment while keeping
    // the test smaller. The intermediate width is the TP2-local geometry.
    const experts = 32;
    const topk = 8;
    const hidden_size = 4096;
    const intermediate_size = 1024;
    const hidden: zml.Tensor = .init(.{ .b = tokens, .s = 1, .d = hidden_size }, .bf16);
    const w1: zml.Tensor = .init(.{ .expert = experts, .out = 2 * intermediate_size, .in = hidden_size }, .f8e4m3fn);
    const w1_scale: zml.Tensor = .init(.{ .expert = experts, .nb = 2 * intermediate_size / 128, .kb = hidden_size / 128 }, .f32);
    const w2: zml.Tensor = .init(.{ .expert = experts, .out = hidden_size, .mid = intermediate_size }, .f8e4m3fn);
    const w2_scale: zml.Tensor = .init(.{ .expert = experts, .nb = hidden_size / 128, .kb = intermediate_size / 128 }, .f32);
    const topk_weight: zml.Tensor = .init(.{ .b = tokens, .s = 1, .top_expert = topk }, .f32);
    const topk_id: zml.Tensor = .init(.{ .b = tokens, .s = 1, .top_expert = topk }, .i32);

    var exe = try platform.compileFn(
        allocator,
        io,
        fusedMoe,
        .{ hidden, w1, w1_scale, w2, w2_scale, topk_weight, topk_id },
        .{},
    );
    defer exe.deinit();

    const one_bf16 = zml.floats.BFloat16.fromF32(1.0);
    const one_fp8 = zml.floats.Float8E4M3FN.fromF32(1.0);
    const hidden_host = try allocator.alloc(zml.floats.BFloat16, hidden.shape().count());
    defer allocator.free(hidden_host);
    @memset(hidden_host, one_bf16);
    const w1_host = try allocator.alloc(zml.floats.Float8E4M3FN, w1.shape().count());
    defer allocator.free(w1_host);
    @memset(w1_host, one_fp8);
    const w1_scale_host = try allocator.alloc(f32, w1_scale.shape().count());
    defer allocator.free(w1_scale_host);
    @memset(w1_scale_host, if (platform.target == .cuda) 1.0 else 2.0);
    const w2_host = try allocator.alloc(zml.floats.Float8E4M3FN, w2.shape().count());
    defer allocator.free(w2_host);
    @memset(w2_host, one_fp8);
    const w2_scale_host = try allocator.alloc(f32, w2_scale.shape().count());
    defer allocator.free(w2_scale_host);
    const native_scale: f32 = if (platform.target == .cuda) 1.0 else 2.0;
    const w2_n_blocks: usize = @intCast(hidden_size / 128);
    const w2_k_blocks: usize = @intCast(intermediate_size / 128);
    for (0..experts) |expert| {
        for (0..w2_n_blocks) |n_block| {
            for (0..w2_k_blocks) |k_block| {
                const scale_index = (expert * w2_n_blocks + n_block) * w2_k_blocks + k_block;
                const factor = 1.0 +
                    @as(f32, @floatFromInt(expert)) / 64.0 +
                    @as(f32, @floatFromInt(n_block)) / 512.0 +
                    @as(f32, @floatFromInt(k_block)) / 128.0;
                w2_scale_host[scale_index] = native_scale * factor;
            }
        }
    }
    const selected_experts = [_]i32{ 0, 1, 7, 17, 31 };
    var topk_weight_host: [tokens][topk]f32 = undefined;
    var topk_id_host: [tokens][topk]i32 = undefined;
    for (&topk_id_host, &topk_weight_host) |*token_ids, *token_weights| {
        for (token_ids, token_weights, 0..) |*id, *weight, i| {
            id.* = if (i + 1 == topk)
                -2
            else if (i + 2 == topk)
                288
            else if (i + 3 == topk)
                287
            else
                selected_experts[i];
            // Non-local and globally invalid routes deliberately carry
            // non-finite weights. They must be skipped like vLLM does, rather
            // than allowing a masked zero route times NaN/Inf to propagate.
            weight.* = if (i + 1 == topk)
                std.math.nan(f32)
            else if (i + 2 == topk)
                std.math.inf(f32)
            else if (i + 3 == topk)
                -std.math.inf(f32)
            else
                1.0 / @as(f32, @floatFromInt(topk - 3));
        }
    }

    var hidden_buffer: zml.Buffer = try .fromBytes(io, platform, hidden.shape(), .replicated, std.mem.sliceAsBytes(hidden_host));
    defer hidden_buffer.deinit();
    var w1_buffer: zml.Buffer = try .fromBytes(io, platform, w1.shape(), .replicated, std.mem.sliceAsBytes(w1_host));
    defer w1_buffer.deinit();
    var w1_scale_buffer: zml.Buffer = try .fromBytes(io, platform, w1_scale.shape(), .replicated, std.mem.sliceAsBytes(w1_scale_host));
    defer w1_scale_buffer.deinit();
    var w2_buffer: zml.Buffer = try .fromBytes(io, platform, w2.shape(), .replicated, std.mem.sliceAsBytes(w2_host));
    defer w2_buffer.deinit();
    var w2_scale_buffer: zml.Buffer = try .fromBytes(io, platform, w2_scale.shape(), .replicated, std.mem.sliceAsBytes(w2_scale_host));
    defer w2_scale_buffer.deinit();
    var topk_weight_buffer: zml.Buffer = try .fromBytes(io, platform, topk_weight.shape(), .replicated, std.mem.asBytes(&topk_weight_host));
    defer topk_weight_buffer.deinit();
    var topk_id_buffer: zml.Buffer = try .fromBytes(io, platform, topk_id.shape(), .replicated, std.mem.asBytes(&topk_id_host));
    defer topk_id_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, fusedMoe, .{
        hidden_buffer,
        w1_buffer,
        w1_scale_buffer,
        w2_buffer,
        w2_scale_buffer,
        topk_weight_buffer,
        topk_id_buffer,
    });
    defer output.deinit();

    const expected_host = try allocator.alloc(zml.floats.BFloat16, output.shape().count());
    defer allocator.free(expected_host);
    const swiglu_limit: f32 = 7;
    const clipped_activation = swiglu_limit / (1.0 + @exp(-swiglu_limit)) * swiglu_limit;
    for (expected_host, 0..) |*value, index| {
        const n_block = index % hidden_size / 128;
        var factor_sum: f32 = 0;
        for (selected_experts) |expert_id| {
            for (0..w2_k_blocks) |k_block| {
                factor_sum += 1.0 +
                    @as(f32, @floatFromInt(expert_id)) / 64.0 +
                    @as(f32, @floatFromInt(n_block)) / 512.0 +
                    @as(f32, @floatFromInt(k_block)) / 128.0;
            }
        }
        const expected_value = 128.0 * clipped_activation * factor_sum / @as(f32, selected_experts.len);
        value.* = zml.floats.BFloat16.fromF32(expected_value);
    }
    const expected: zml.Slice = .init(output.shape(), std.mem.sliceAsBytes(expected_host));
    try zml.testing.expectClose(io, expected, output, .{ .absolute_tolerance = 256.0, .relative_tolerance = 0.01 });
}

test "CUDA and ROCm Triton fused MoE block-scaled FP8 GEMMs" {
    // Sixteen tokens exercise the 288-expert histogram/alignment path; eight
    // exercise the naive decode assignment path and its pointer sanitization.
    try testFusedMoeBlockFp8(16);
    try testFusedMoeBlockFp8(8);
}

fn addLocalSharedExpert(
    _: void,
    local_input: zml.Tensor,
    prepared_a1: zml.fp8.PreparedBlock128Activation,
    local_routed: zml.Tensor,
    local_operands: []const zml.Tensor,
) zml.Tensor {
    std.debug.assert(local_operands.len == 4);
    const gate_up_weight = local_operands[0];
    const gate_up_scale = local_operands[1];
    const down_weight = local_operands[2];
    const down_scale = local_operands[3];

    const gate_up_shape = local_input.shape()
        .setDim(.d, gate_up_weight.dim(.out))
        .setTag(.d, .out)
        .withDtype(.bf16);
    const gate_up = zml.fp8.nativeBlockScaledDotPreparedLocal(
        prepared_a1,
        gate_up_weight,
        gate_up_scale,
        gate_up_shape,
    );
    const intermediate_size = @divExact(gate_up.dim(.out), 2);
    const gate = gate_up.slice(.out, .{ .end = intermediate_size }).convert(.f32);
    const up = gate_up.slice(.out, .{ .start = intermediate_size }).convert(.f32);
    const activation = gate.silu().mul(up).convert(.bf16).rename(.{ .out = .mid });

    const local_shared = zml.fp8.nativeBlockScaledDotLocal(
        activation,
        down_weight,
        down_scale,
        local_routed.shape(),
    );
    return local_routed.add(local_shared.convert(local_routed.dtype()));
}

fn fusedMoeWithLocalEpilogue(
    hidden: zml.Tensor,
    w1: zml.Tensor,
    w1_scale: zml.Tensor,
    w2: zml.Tensor,
    w2_scale: zml.Tensor,
    topk_weight: zml.Tensor,
    topk_id: zml.Tensor,
    shared_gate_up_weight: zml.Tensor,
    shared_gate_up_scale: zml.Tensor,
    shared_down_weight: zml.Tensor,
    shared_down_scale: zml.Tensor,
) zml.Tensor {
    const metadata: zml.moe.Metadata = .{ .triton = .init(.{}) };
    const parameters: zml.moe.Parameters = .{ .triton = .init(.{
        .num_experts_per_tok = @intCast(topk_id.dim(.top_expert)),
        .activation = .silu,
    }) };
    return zml.moe.forwardMoeWithReduceEpilogue(
        hidden.withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated }),
        topk_id.withPartitioning(.{ .b = .replicated, .s = .replicated, .top_expert = .replicated }),
        topk_weight.withPartitioning(.{ .b = .replicated, .s = .replicated, .top_expert = .replicated }),
        w1.withPartitioning(.{ .expert = .experts, .out = .replicated, .in = .replicated }),
        w1_scale.withPartitioning(.{ .expert = .experts, .nb = .replicated, .kb = .replicated }),
        null,
        w2.withPartitioning(.{ .expert = .experts, .out = .replicated, .mid = .replicated }),
        w2_scale.withPartitioning(.{ .expert = .experts, .nb = .replicated, .kb = .replicated }),
        null,
        null,
        null,
        .{ .quant_scheme = .fp8_block128 },
        metadata,
        parameters,
        .{
            shared_gate_up_weight.withPartitioning(.{ .out = .model, .in = .replicated }),
            shared_gate_up_scale.withPartitioning(.{ .nb = .model, .kb = .replicated }),
            shared_down_weight.withPartitioning(.{ .out = .replicated, .mid = .model }),
            shared_down_scale.withPartitioning(.{ .nb = .replicated, .kb = .model }),
        },
        {},
        addLocalSharedExpert,
    ) catch unreachable;
}

test "expert-parallel FP8 MoE compiles top-8 reduction with a model-sharded shared expert epilogue" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const experts: i64 = @intCast(platform.devices.len);
    const hidden_size = 128;
    const routed_intermediate_size = 128;
    const shared_intermediate_size = 128 * experts;
    const hidden: zml.Tensor = .init(.{ .b = 1, .s = 1, .d = hidden_size }, .bf16);
    const w1: zml.Tensor = .init(.{ .expert = experts, .out = 2 * routed_intermediate_size, .in = hidden_size }, .f8e4m3fn);
    const w1_scale: zml.Tensor = .init(.{ .expert = experts, .nb = 2, .kb = 1 }, .f32);
    const w2: zml.Tensor = .init(.{ .expert = experts, .out = hidden_size, .mid = routed_intermediate_size }, .f8e4m3fn);
    const w2_scale: zml.Tensor = .init(.{ .expert = experts, .nb = 1, .kb = 1 }, .f32);
    const topk_weight: zml.Tensor = .init(.{ .b = 1, .s = 1, .top_expert = 8 }, .f32);
    const topk_id: zml.Tensor = .init(.{ .b = 1, .s = 1, .top_expert = 8 }, .i32);
    const shared_gate_up_weight: zml.Tensor = .init(.{ .out = 2 * shared_intermediate_size, .in = hidden_size }, .f8e4m3fn);
    const shared_gate_up_scale: zml.Tensor = .init(.{ .nb = 2 * experts, .kb = 1 }, .f32);
    const shared_down_weight: zml.Tensor = .init(.{ .out = hidden_size, .mid = shared_intermediate_size }, .f8e4m3fn);
    const shared_down_scale: zml.Tensor = .init(.{ .nb = 1, .kb = experts }, .f32);

    const expert_sharding = try @constCast(platform).registerSharding(
        "fp8_test_mixed_experts",
        .mesh(.{ .experts = .high_bandwidth }),
    );
    const model_sharding = try @constCast(platform).registerSharding(
        "fp8_test_mixed_model",
        .mesh(.{ .model = .high_bandwidth }),
    );
    var exe = try platform.compileFn(
        allocator,
        io,
        fusedMoeWithLocalEpilogue,
        .{
            hidden,
            w1,
            w1_scale,
            w2,
            w2_scale,
            topk_weight,
            topk_id,
            shared_gate_up_weight,
            shared_gate_up_scale,
            shared_down_weight,
            shared_down_scale,
        },
        .{ .shardings = &.{ expert_sharding, model_sharding } },
    );
    defer exe.deinit();
}

fn expectedConstantBlockFp8MlpPartial(gate_up_scale: f32, down_scale: f32, routed_weight: f32) f32 {
    const BFloat16 = zml.floats.BFloat16;
    // The test input sums to one over each 128-wide contraction. Both the
    // gate and up blocks use the same scale, so account for each BF16 output
    // boundary in the two block-FP8 dots.
    const gate_up = BFloat16.fromF32(gate_up_scale).toF32();
    const activation = BFloat16.fromF32(gate_up / (1.0 + @exp(-gate_up)) * gate_up).toF32();
    return BFloat16.fromF32(128.0 * activation * down_scale * routed_weight).toF32();
}

test "CUDA expert-parallel FP8 MoE numerically combines routed and model-sharded shared experts" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda or platform.devices.len != 2) return error.SkipZigTest;

    const experts = 2;
    const hidden_size = 128;
    const routed_intermediate_size = 128;
    const shared_intermediate_size = 256;
    const hidden: zml.Tensor = .init(.{ .b = 1, .s = 1, .d = hidden_size }, .bf16);
    const w1: zml.Tensor = zml.Tensor.init(
        .{ .expert = experts, .out = 2 * routed_intermediate_size, .in = hidden_size },
        .f8e4m3fn,
    ).withPartitioning(.{ .expert = .experts, .out = .replicated, .in = .replicated });
    const w1_scale: zml.Tensor = zml.Tensor.init(
        .{ .expert = experts, .nb = 2, .kb = 1 },
        .f32,
    ).withPartitioning(.{ .expert = .experts, .nb = .replicated, .kb = .replicated });
    const w2: zml.Tensor = zml.Tensor.init(
        .{ .expert = experts, .out = hidden_size, .mid = routed_intermediate_size },
        .f8e4m3fn,
    ).withPartitioning(.{ .expert = .experts, .out = .replicated, .mid = .replicated });
    const w2_scale: zml.Tensor = zml.Tensor.init(
        .{ .expert = experts, .nb = 1, .kb = 1 },
        .f32,
    ).withPartitioning(.{ .expert = .experts, .nb = .replicated, .kb = .replicated });
    const topk_weight: zml.Tensor = .init(.{ .b = 1, .s = 1, .top_expert = 2 }, .f32);
    const topk_id: zml.Tensor = .init(.{ .b = 1, .s = 1, .top_expert = 2 }, .i32);
    const shared_gate_up_weight: zml.Tensor = zml.Tensor.init(
        .{ .out = 2 * shared_intermediate_size, .in = hidden_size },
        .f8e4m3fn,
    ).withPartitioning(.{ .out = .model, .in = .replicated });
    const shared_gate_up_scale: zml.Tensor = zml.Tensor.init(
        .{ .nb = 4, .kb = 1 },
        .f32,
    ).withPartitioning(.{ .nb = .model, .kb = .replicated });
    const shared_down_weight: zml.Tensor = zml.Tensor.init(
        .{ .out = hidden_size, .mid = shared_intermediate_size },
        .f8e4m3fn,
    ).withPartitioning(.{ .out = .replicated, .mid = .model });
    const shared_down_scale: zml.Tensor = zml.Tensor.init(
        .{ .nb = 1, .kb = 2 },
        .f32,
    ).withPartitioning(.{ .nb = .replicated, .kb = .model });

    const expert_sharding = try @constCast(platform).registerSharding(
        "fp8_test_numerical_experts",
        .mesh(.{ .experts = .high_bandwidth }),
    );
    const model_sharding = try @constCast(platform).registerSharding(
        "fp8_test_numerical_model",
        .mesh(.{ .model = .high_bandwidth }),
    );
    var exe = try platform.compileFn(
        allocator,
        io,
        fusedMoeWithLocalEpilogue,
        .{
            hidden,
            w1,
            w1_scale,
            w2,
            w2_scale,
            topk_weight,
            topk_id,
            shared_gate_up_weight,
            shared_gate_up_scale,
            shared_down_weight,
            shared_down_scale,
        },
        .{ .shardings = &.{ expert_sharding, model_sharding } },
    );
    defer exe.deinit();

    const BFloat16 = zml.floats.BFloat16;
    const Float8 = zml.floats.Float8E4M3FN;
    const one_fp8 = Float8.fromF32(1.0);
    const zero_fp8 = Float8.fromF32(0.0);

    const hidden_host = try allocator.alloc(BFloat16, hidden.shape().count());
    defer allocator.free(hidden_host);
    @memset(hidden_host, BFloat16.fromF32(1.0 / 128.0));

    const w1_host = try allocator.alloc(Float8, w1.shape().count());
    defer allocator.free(w1_host);
    @memset(w1_host, one_fp8);
    const w1_scale_host = [_][2][1]f32{
        .{ .{1.0}, .{1.0} },
        .{ .{2.0}, .{2.0} },
    };

    const w2_host = try allocator.alloc(Float8, w2.shape().count());
    defer allocator.free(w2_host);
    @memset(w2_host, zero_fp8);
    for (0..experts) |expert| {
        for (0..hidden_size / 2) |out| {
            const start = (expert * hidden_size + out) * routed_intermediate_size;
            @memset(w2_host[start .. start + routed_intermediate_size], one_fp8);
        }
    }
    const w2_scale_host = [_][1][1]f32{
        .{.{1.0}},
        .{.{1.0}},
    };

    const topk_weight_host = [1][1][2]f32{.{.{ 1.0, 1.0 }}};
    const topk_id_host = [1][1][2]i32{.{.{ 0, 1 }}};

    const shared_gate_up_weight_host = try allocator.alloc(Float8, shared_gate_up_weight.shape().count());
    defer allocator.free(shared_gate_up_weight_host);
    @memset(shared_gate_up_weight_host, one_fp8);
    // Each rank receives two consecutive output blocks (one local gate and
    // one local up block). The 1x versus 4x scales make rank-0 scale reuse on
    // rank 1 produce a large, deterministic numerical error.
    const shared_gate_up_scale_host = [4][1]f32{ .{1.0}, .{1.0}, .{4.0}, .{4.0} };

    const shared_down_weight_host = try allocator.alloc(Float8, shared_down_weight.shape().count());
    defer allocator.free(shared_down_weight_host);
    @memset(shared_down_weight_host, zero_fp8);
    for (hidden_size / 2..hidden_size) |out| {
        const start = out * shared_intermediate_size;
        @memset(shared_down_weight_host[start .. start + shared_intermediate_size], one_fp8);
    }
    // The contracted scale axis is model-sharded: rank 0 receives 1x and
    // rank 1 receives 2x. This independently catches scale broadcasting in
    // the shared down projection.
    const shared_down_scale_host = [1][2]f32{.{ 1.0, 2.0 }};

    var hidden_buffer: zml.Buffer = try .fromBytes(io, platform, hidden.shape(), .replicated, std.mem.sliceAsBytes(hidden_host));
    defer hidden_buffer.deinit();
    var w1_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        w1.shape(),
        expert_sharding,
        std.mem.sliceAsBytes(w1_host),
    );
    defer w1_buffer.deinit();
    var w1_scale_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        w1_scale.shape(),
        expert_sharding,
        std.mem.asBytes(&w1_scale_host),
    );
    defer w1_scale_buffer.deinit();
    var w2_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        w2.shape(),
        expert_sharding,
        std.mem.sliceAsBytes(w2_host),
    );
    defer w2_buffer.deinit();
    var w2_scale_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        w2_scale.shape(),
        expert_sharding,
        std.mem.asBytes(&w2_scale_host),
    );
    defer w2_scale_buffer.deinit();
    var topk_weight_buffer: zml.Buffer = try .fromBytes(io, platform, topk_weight.shape(), .replicated, std.mem.asBytes(&topk_weight_host));
    defer topk_weight_buffer.deinit();
    var topk_id_buffer: zml.Buffer = try .fromBytes(io, platform, topk_id.shape(), .replicated, std.mem.asBytes(&topk_id_host));
    defer topk_id_buffer.deinit();
    var shared_gate_up_weight_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        shared_gate_up_weight.shape(),
        model_sharding,
        std.mem.sliceAsBytes(shared_gate_up_weight_host),
    );
    defer shared_gate_up_weight_buffer.deinit();
    var shared_gate_up_scale_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        shared_gate_up_scale.shape(),
        model_sharding,
        std.mem.asBytes(&shared_gate_up_scale_host),
    );
    defer shared_gate_up_scale_buffer.deinit();
    var shared_down_weight_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        shared_down_weight.shape(),
        model_sharding,
        std.mem.sliceAsBytes(shared_down_weight_host),
    );
    defer shared_down_weight_buffer.deinit();
    var shared_down_scale_buffer: zml.Buffer = try .fromBytes(
        io,
        platform,
        shared_down_scale.shape(),
        model_sharding,
        std.mem.asBytes(&shared_down_scale_host),
    );
    defer shared_down_scale_buffer.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, fusedMoeWithLocalEpilogue, .{
        hidden_buffer,
        w1_buffer,
        w1_scale_buffer,
        w2_buffer,
        w2_scale_buffer,
        topk_weight_buffer,
        topk_id_buffer,
        shared_gate_up_weight_buffer,
        shared_gate_up_scale_buffer,
        shared_down_weight_buffer,
        shared_down_scale_buffer,
    });
    defer output.deinit();

    const routed_expected_f32 = BFloat16.fromF32(
        expectedConstantBlockFp8MlpPartial(1.0, 1.0, 1.0) +
            expectedConstantBlockFp8MlpPartial(2.0, 1.0, 1.0),
    ).toF32();
    const shared_expected_f32 = BFloat16.fromF32(
        expectedConstantBlockFp8MlpPartial(1.0, 1.0, 1.0) +
            expectedConstantBlockFp8MlpPartial(4.0, 2.0, 1.0),
    ).toF32();
    var expected_host: [hidden_size]BFloat16 = undefined;
    for (&expected_host, 0..) |*value, out| {
        value.* = BFloat16.fromF32(if (out < hidden_size / 2) routed_expected_f32 else shared_expected_f32);
    }
    const expected: zml.Slice = .init(output.shape(), std.mem.asBytes(&expected_host));
    try zml.testing.expectClose(
        io,
        expected,
        output,
        .{ .absolute_tolerance = 8.0, .relative_tolerance = 0.02, .minimum_close_fraction = 1.0 },
    );
}

const MixedScaleMoeOutputs = struct {
    w1_only: zml.Tensor,
    w2_only: zml.Tensor,
};

fn fusedMoeWithMixedScalePresence(
    hidden: zml.Tensor,
    fp8_w1: zml.Tensor,
    w1_scale: zml.Tensor,
    bf16_w1: zml.Tensor,
    fp8_w2: zml.Tensor,
    w2_scale: zml.Tensor,
    bf16_w2: zml.Tensor,
    topk_weight: zml.Tensor,
    topk_id: zml.Tensor,
) MixedScaleMoeOutputs {
    const metadata: zml.moe.Metadata = .{ .triton = .init(.{}) };
    const parameters: zml.moe.Parameters = .{ .triton = .init(.{
        .num_experts_per_tok = 2,
        .activation = .silu,
    }) };
    const local_hidden = hidden.withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated });
    const local_topk_id = topk_id.withPartitioning(.{ .b = .replicated, .s = .replicated, .top_expert = .replicated });
    const local_topk_weight = topk_weight.withPartitioning(.{ .b = .replicated, .s = .replicated, .top_expert = .replicated });
    const options: zml.moe.Options = .{};

    return .{
        .w1_only = zml.moe.forwardMoe(
            local_hidden,
            local_topk_id,
            local_topk_weight,
            fp8_w1.withPartitioning(.{ .expert = .experts, .out = .replicated, .in = .replicated }),
            w1_scale.withPartitioning(.{}),
            null,
            bf16_w2.withPartitioning(.{ .expert = .experts, .out = .replicated, .mid = .replicated }),
            null,
            null,
            null,
            null,
            options,
            metadata,
            parameters,
        ) catch unreachable,
        .w2_only = zml.moe.forwardMoe(
            local_hidden,
            local_topk_id,
            local_topk_weight,
            bf16_w1.withPartitioning(.{ .expert = .experts, .out = .replicated, .in = .replicated }),
            null,
            null,
            fp8_w2.withPartitioning(.{ .expert = .experts, .out = .replicated, .mid = .replicated }),
            w2_scale.withPartitioning(.{}),
            null,
            null,
            null,
            options,
            metadata,
            parameters,
        ) catch unreachable,
    };
}

test "expert-parallel Triton MoE compiles with either FP8 stage scaled" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const experts: i64 = @intCast(platform.devices.len);
    const hidden_size = 128;
    const intermediate_size = 128;
    const hidden: zml.Tensor = .init(.{ .b = 8, .s = 1, .d = hidden_size }, .bf16);
    const fp8_w1: zml.Tensor = .init(.{ .expert = experts, .out = 2 * intermediate_size, .in = hidden_size }, .f8e4m3fn);
    const w1_scale: zml.Tensor = .init(.{}, .bf16);
    const bf16_w1: zml.Tensor = .init(.{ .expert = experts, .out = 2 * intermediate_size, .in = hidden_size }, .bf16);
    const fp8_w2: zml.Tensor = .init(.{ .expert = experts, .out = hidden_size, .mid = intermediate_size }, .f8e4m3fn);
    const w2_scale: zml.Tensor = .init(.{}, .bf16);
    const bf16_w2: zml.Tensor = .init(.{ .expert = experts, .out = hidden_size, .mid = intermediate_size }, .bf16);
    const topk_weight: zml.Tensor = .init(.{ .b = 8, .s = 1, .top_expert = 2 }, .f32);
    const topk_id: zml.Tensor = .init(.{ .b = 8, .s = 1, .top_expert = 2 }, .i32);

    const expert_sharding = try @constCast(platform).registerSharding(
        "fp8_test_mixed_scale_experts",
        .mesh(.{ .experts = .high_bandwidth }),
    );
    var exe = try platform.compileFn(
        allocator,
        io,
        fusedMoeWithMixedScalePresence,
        .{ hidden, fp8_w1, w1_scale, bf16_w1, fp8_w2, w2_scale, bf16_w2, topk_weight, topk_id },
        .{ .shardings = &.{expert_sharding} },
    );
    defer exe.deinit();
}
