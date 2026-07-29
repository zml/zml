const std = @import("std");

const platforms = @import("platforms");
const zml = @import("zml");

const cutlass = zml.moe.cutlass_flashinfer;
const BFloat16 = zml.floats.BFloat16;

const TestMoe = struct {
    fn cutlassForward(
        hidden: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weights: zml.Tensor,
        gate_up: zml.Tensor,
        down: zml.Tensor,
    ) zml.Tensor {
        const backend: zml.moe.Backend = .cuda_flashinfer_cutlass;
        const metadata = zml.moe.Metadata.init(.fromBackend(backend));
        const parameters = zml.moe.Parameters.init(
            .fromBackend(backend, 1, .silu),
        );
        return zml.moe.forwardMoe(
            hidden,
            topk_ids,
            topk_weights,
            gate_up,
            null,
            null,
            down,
            null,
            null,
            metadata,
            parameters,
        ) catch |err| std.debug.panic(
            "FlashInfer CUTLASS backend graph construction failed: {}",
            .{err},
        );
    }

    fn tritonForward(
        hidden: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weights: zml.Tensor,
        gate_up: zml.Tensor,
        down: zml.Tensor,
    ) zml.Tensor {
        const backend: zml.moe.Backend = .triton;
        const metadata = zml.moe.Metadata.init(.fromBackend(backend));
        const parameters = zml.moe.Parameters.init(
            .fromBackend(backend, 1, .silu),
        );
        return zml.moe.forwardMoe(
            hidden,
            topk_ids,
            topk_weights,
            gate_up,
            null,
            null,
            down,
            null,
            null,
            metadata,
            parameters,
        ) catch |err| std.debug.panic(
            "Triton backend graph construction failed: {}",
            .{err},
        );
    }
};

test "BF16 backend matches Triton after routing" {
    @setEvalBranchQuota(10_000);
    if (comptime !platforms.isEnabled(.cuda)) return;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return;
    const devices = platform.pjrt_client.devices(platform.pjrt_api);
    const cc = zml.platform.cuda.tryGetComputeCapabilities(platform, devices[0]) orelse
        return;
    const supported_architecture = std.mem.eql(u8, cc, "9.0") or
        std.mem.eql(u8, cc, "10.0") or
        std.mem.eql(u8, cc, "12.0");
    if (!supported_architecture) return;
    try std.testing.expect(cutlass.isAvailable(platform));

    try std.testing.expectEqual(
        zml.moe.Backend.cuda_flashinfer_cutlass,
        try zml.moe.Backend.auto(platform, .bf16),
    );
    const tactics = try cutlass.tacticCounts(0, .bf16);
    try std.testing.expect(tactics.gemm1 > 0);
    try std.testing.expect(tactics.gemm2 > 0);

    const num_experts = 2;
    const num_tokens = 4;
    const hidden_size = 128;
    const intermediate_size = 128;

    const hidden = zml.Tensor.init(
        .{ .b = 1, .s = num_tokens, .d = hidden_size },
        .bf16,
    );
    const topk_ids = zml.Tensor.init(
        .{ .b = 1, .s = num_tokens, .top_expert = 1 },
        .i32,
    );
    const topk_weights = zml.Tensor.init(
        .{ .b = 1, .s = num_tokens, .top_expert = 1 },
        .f32,
    );
    const gate_up = zml.Tensor.init(
        .{
            .expert = num_experts,
            .dout = 2 * intermediate_size,
            .d = hidden_size,
        },
        .bf16,
    );
    const down = zml.Tensor.init(
        .{
            .expert = num_experts,
            .d = hidden_size,
            .dout = intermediate_size,
        },
        .bf16,
    );

    var cutlass_exe = try platform.compileFn(
        allocator,
        io,
        TestMoe.cutlassForward,
        .{ hidden, topk_ids, topk_weights, gate_up, down },
        .{ .program_name = "flashinfer_cutlass_moe_bf16_test" },
    );
    defer cutlass_exe.deinit();
    var triton_exe = try platform.compileFn(
        allocator,
        io,
        TestMoe.tritonForward,
        .{ hidden, topk_ids, topk_weights, gate_up, down },
        .{ .program_name = "triton_moe_bf16_reference" },
    );
    defer triton_exe.deinit();

    const hidden_host = try allocator.alloc(BFloat16, num_tokens * hidden_size);
    defer allocator.free(hidden_host);
    for (hidden_host, 0..) |*value, i| {
        const signed: i32 = @intCast(i % 17);
        value.* = BFloat16.fromF32(@as(f32, @floatFromInt(signed - 8)) / 16.0);
    }

    const gate_up_host = try allocator.alloc(
        BFloat16,
        num_experts * 2 * intermediate_size * hidden_size,
    );
    defer allocator.free(gate_up_host);
    @memset(gate_up_host, BFloat16.fromF32(0));
    for (0..num_experts) |expert| {
        const expert_offset = expert * 2 * intermediate_size * hidden_size;
        const expert_scale = 1.0 + @as(f32, @floatFromInt(expert)) * 0.25;
        for (0..intermediate_size) |i| {
            gate_up_host[expert_offset + i * hidden_size + i] =
                BFloat16.fromF32(0.5 * expert_scale);
            gate_up_host[
                expert_offset + (intermediate_size + i) * hidden_size + i
            ] = BFloat16.fromF32(0.25 * expert_scale);
        }
    }

    const down_host = try allocator.alloc(
        BFloat16,
        num_experts * hidden_size * intermediate_size,
    );
    defer allocator.free(down_host);
    @memset(down_host, BFloat16.fromF32(0));
    for (0..num_experts) |expert| {
        const expert_offset = expert * hidden_size * intermediate_size;
        for (0..hidden_size) |i| {
            down_host[expert_offset + i * intermediate_size + i] =
                BFloat16.fromF32(1);
        }
    }

    const ids_host = [num_tokens]i32{ 0, 1, 0, 1 };
    const weights_host = [num_tokens]f32{ 1, 1, 1, 1 };

    var hidden_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        hidden.shape(),
        .replicated,
        std.mem.sliceAsBytes(hidden_host),
    );
    defer hidden_buffer.deinit();
    var ids_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        topk_ids.shape(),
        .replicated,
        std.mem.asBytes(&ids_host),
    );
    defer ids_buffer.deinit();
    var weights_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        topk_weights.shape(),
        .replicated,
        std.mem.asBytes(&weights_host),
    );
    defer weights_buffer.deinit();
    var gate_up_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        gate_up.shape(),
        .replicated,
        std.mem.sliceAsBytes(gate_up_host),
    );
    defer gate_up_buffer.deinit();
    var down_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        down.shape(),
        .replicated,
        std.mem.sliceAsBytes(down_host),
    );
    defer down_buffer.deinit();

    var cutlass_output = try zml.testing.autoCall(
        allocator,
        io,
        &cutlass_exe,
        TestMoe.cutlassForward,
        .{
            hidden_buffer,
            ids_buffer,
            weights_buffer,
            gate_up_buffer,
            down_buffer,
        },
    );
    defer cutlass_output.deinit();
    var triton_output = try zml.testing.autoCall(
        allocator,
        io,
        &triton_exe,
        TestMoe.tritonForward,
        .{
            hidden_buffer,
            ids_buffer,
            weights_buffer,
            gate_up_buffer,
            down_buffer,
        },
    );
    defer triton_output.deinit();

    try zml.testing.expectClose(io, triton_output, cutlass_output, .{
        .absolute_tolerance = 2e-2,
        .relative_tolerance = 2e-2,
        .epsilon_relative = 1e-3,
        .minimum_close_fraction = 0.99,
    });
}
