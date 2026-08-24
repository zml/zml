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
    var exe = try zml.module.compile(allocator, io, normalizeOcpEncoding, .{weight}, platform, .{});
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
    return switch (zml.fp8.Backend.selected()) {
        .triton => zml.fp8.tritonBlockScaledDot(x, weight, weight_scale, output_shape),
        .ck => zml.fp8.ckBlockScaledDot(x, weight, weight_scale, output_shape),
    };
}

test "ROCm selected-backend block-scaled FP8 GEMM" {
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

    var exe = try zml.module.compile(allocator, io, blockDot, .{ x, weight, weight_scale }, platform, .{});
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
    return key.slice1d(.latent, .{ .end = @intCast(value_dim) }).rename(.{ .latent = .value }).add(value);
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
    var exe = try zml.module.compile(
        allocator,
        io,
        absorbedDotsSharded,
        .{ q, latent, weight, weight_scale },
        platform,
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

    var exe = try zml.module.compile(allocator, io, absorbedDots, .{ q, latent, weight, weight_scale }, platform, .{});
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
    return zml.moe.triton.fusedExpertsImpl(
        hidden,
        w1,
        w2,
        topk_weight,
        topk_id,
        .{},
        .{
            .activation = .silu,
            .global_num_experts = 256,
            .w1_scale = w1_scale,
            .w2_scale = w2_scale,
        },
    ) catch unreachable;
}

test "ROCm Triton fused MoE block-scaled FP8 GEMMs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm or zml.fp8.Backend.selected() != .triton) return error.SkipZigTest;

    // The real TP4-local GLM geometry is E=256. Only the first 32 expert
    // weights are needed by these routes; global_num_experts still keeps the
    // assignment decision identical while making the test much smaller.
    const experts = 32;
    const tokens = 16;
    const topk = 8;
    const hidden_size = 6144;
    const intermediate_size = 512;
    const hidden: zml.Tensor = .init(.{ .b = tokens, .s = 1, .d = hidden_size }, .bf16);
    const w1: zml.Tensor = .init(.{ .expert = experts, .out = 2 * intermediate_size, .in = hidden_size }, .f8e4m3fn);
    const w1_scale: zml.Tensor = .init(.{ .expert = experts, .nb = 2 * intermediate_size / 128, .kb = hidden_size / 128 }, .f32);
    const w2: zml.Tensor = .init(.{ .expert = experts, .out = hidden_size, .mid = intermediate_size }, .f8e4m3fn);
    const w2_scale: zml.Tensor = .init(.{ .expert = experts, .nb = hidden_size / 128, .kb = intermediate_size / 128 }, .f32);
    const topk_weight: zml.Tensor = .init(.{ .b = tokens, .s = 1, .top_expert = topk }, .f32);
    const topk_id: zml.Tensor = .init(.{ .b = tokens, .s = 1, .top_expert = topk }, .i32);

    var exe = try zml.module.compile(
        allocator,
        io,
        fusedMoe,
        .{ hidden, w1, w1_scale, w2, w2_scale, topk_weight, topk_id },
        platform,
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
    @memset(w1_scale_host, 2.0);
    const w2_host = try allocator.alloc(zml.floats.Float8E4M3FN, w2.shape().count());
    defer allocator.free(w2_host);
    @memset(w2_host, one_fp8);
    const w2_scale_host = try allocator.alloc(f32, w2_scale.shape().count());
    defer allocator.free(w2_scale_host);
    @memset(w2_scale_host, 2.0);
    const topk_weight_host: [tokens][topk]f32 = @splat(@splat(1.0 / @as(f32, @floatFromInt(topk))));
    var topk_id_host: [tokens][topk]i32 = undefined;
    for (&topk_id_host) |*token_ids| {
        for (token_ids, 0..) |*id, i| id.* = @intCast(i);
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
    @memset(expected_host, zml.floats.BFloat16.fromF32(hidden_size * hidden_size * intermediate_size));
    const expected: zml.Slice = .init(output.shape(), std.mem.sliceAsBytes(expected_host));
    try zml.testing.expectClose(io, expected, output, .{ .absolute_tolerance = 65536.0, .relative_tolerance = 0.01 });
}

test "ROCm native AITER fused MoE at GLM geometry" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .rocm or zml.fp8.Backend.selected() != .ck) return error.SkipZigTest;

    const hidden_size = 6144;
    const intermediate_size = 256;
    const hidden: zml.Tensor = .init(.{ .b = 1, .s = 1, .d = hidden_size }, .bf16);
    const w1: zml.Tensor = .init(.{ .expert = 1, .out = 2 * intermediate_size, .in = hidden_size }, .f8e4m3fn);
    const w1_scale: zml.Tensor = .init(.{ .expert = 1, .nb = 2 * intermediate_size / 128, .kb = hidden_size / 128 }, .f32);
    const w2: zml.Tensor = .init(.{ .expert = 1, .out = hidden_size, .mid = intermediate_size }, .f8e4m3fn);
    const w2_scale: zml.Tensor = .init(.{ .expert = 1, .nb = hidden_size / 128, .kb = intermediate_size / 128 }, .f32);
    const topk_weight: zml.Tensor = .init(.{ .b = 1, .s = 1, .top_expert = 1 }, .f32);
    const topk_id: zml.Tensor = .init(.{ .b = 1, .s = 1, .top_expert = 1 }, .i32);

    var exe = try zml.module.compile(
        allocator,
        io,
        fusedMoe,
        .{ hidden, w1, w1_scale, w2, w2_scale, topk_weight, topk_id },
        platform,
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
    const w2_host = try allocator.alloc(zml.floats.Float8E4M3FN, w2.shape().count());
    defer allocator.free(w2_host);
    @memset(w2_host, one_fp8);
    const w1_scale_host = try allocator.alloc(f32, w1_scale.shape().count());
    defer allocator.free(w1_scale_host);
    @memset(w1_scale_host, 2.0);
    const w2_scale_host = try allocator.alloc(f32, w2_scale.shape().count());
    defer allocator.free(w2_scale_host);
    @memset(w2_scale_host, 2.0);
    const topk_weight_host: [1]f32 = .{1.0};
    const topk_id_host: [1]i32 = .{0};

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

    const expected_value = @as(f32, hidden_size * hidden_size * intermediate_size);
    const expected_element = zml.floats.BFloat16.fromF32(expected_value);
    const expected_host = try allocator.alloc(zml.floats.BFloat16, hidden_size);
    defer allocator.free(expected_host);
    @memset(expected_host, expected_element);
    try zml.testing.expectClose(
        io,
        zml.Slice.init(output.shape(), std.mem.sliceAsBytes(expected_host)),
        output,
        .{ .absolute_tolerance = expected_value * 0.02, .relative_tolerance = 0.02 },
    );
}
