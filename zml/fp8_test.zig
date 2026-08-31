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
    return zml.fp8.tritonBlockScaledDot(x, weight, weight_scale, output_shape);
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

test "CUDA Triton block-scaled E4M3FN GEMM direct decode and prefill" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    // A one-tile scale grid remains native even though scheme classification
    // intentionally treats the ambiguous shape as per-tensor FP8.
    try testCudaBlockDotCase(1, 64, 128);

    // N=193 exercises a partially populated final 128-row scale block.
    try testCudaBlockDotCase(1, 193, 256);
    try testCudaBlockDotCase(65, 193, 256);
}

test "CUDA Triton block-scaled E4M3FN GEMM split-K decode and prefill" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    // K=2048 and small N select split-K; N=129 keeps the final scale block partial.
    try testCudaBlockDotCase(1, 129, 2048);
    try testCudaBlockDotCase(65, 129, 2048);
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
            weight.* = if (i + 3 >= topk) 0 else 1.0 / @as(f32, @floatFromInt(topk - 3));
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
