const std = @import("std");

const zml = @import("zml");
const flashinfer_moe = zml.flashinfer_moe;

const log = std.log.scoped(.flashinfer_moe_kernel_test);

const NUM_EXPERTS: usize = 4;
const HIDDEN_SIZE: usize = 128;
const OUT_FEATURES: usize = 128;
const TOKENS_PER_EXPERT: [NUM_EXPERTS]i64 = .{ 2, 1, 3, 2 };
const TOTAL_TOKENS: usize = 8;

const Demo = struct {
    pub fn forward(
        activations_bf16: zml.Tensor,
        weights_fp4_e2m1_packed: zml.Tensor,
        scales_ue8m0: zml.Tensor,
        expert_first_token_offset_device: zml.Tensor,
    ) zml.Tensor {
        return flashinfer_moe.flashinferMoeForward(
            activations_bf16,
            weights_fp4_e2m1_packed,
            scales_ue8m0,
            expert_first_token_offset_device,
            .{
                .num_experts = @intCast(NUM_EXPERTS),
                .hidden_size = @intCast(HIDDEN_SIZE),
                .out_features = @intCast(OUT_FEATURES),
                .output_shape = zml.Shape.init(.{ TOTAL_TOKENS, OUT_FEATURES }, .bf16),
            },
        );
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    if (platform.target != .cuda) return error.RequiresCuda;

    try flashinfer_moe.load(allocator, io);
    try flashinfer_moe.register(platform);

    const groups_per_n: usize = HIDDEN_SIZE / 32;
    const scale_outer: usize = HIDDEN_SIZE / 128;

    const activations_t: zml.Tensor = .init(.{ TOTAL_TOKENS, HIDDEN_SIZE }, .bf16);
    const weights_t: zml.Tensor = .init(.{ NUM_EXPERTS, OUT_FEATURES, HIDDEN_SIZE / 2 }, .u8);
    const scales_t: zml.Tensor = .init(.{ NUM_EXPERTS, scale_outer, OUT_FEATURES * 4 }, .f8e8m0);
    const expert_offsets_t: zml.Tensor = .init(.{NUM_EXPERTS + 1}, .i64);

    var exe = try platform.compileFn(allocator, io, Demo.forward, .{ activations_t, weights_t, scales_t, expert_offsets_t });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    var activations_host = try zml.Slice.alloc(allocator, activations_t.shape());
    defer activations_host.free(allocator);
    var weights_host = try zml.Slice.alloc(allocator, weights_t.shape());
    defer weights_host.free(allocator);
    var scales_host = try zml.Slice.alloc(allocator, scales_t.shape());
    defer scales_host.free(allocator);
    var expert_offsets_host = try zml.Slice.alloc(allocator, expert_offsets_t.shape());
    defer expert_offsets_host.free(allocator);

    {
        const x = activations_host.items(zml.floats.BFloat16);
        for (0..x.len) |i| x[i] = zml.floats.BFloat16.fromF32(@floatFromInt((i % 13) + 1));
    }
    @memset(weights_host.items(u8), 0x11);

    {
        const logical_scales = try allocator.alloc(zml.floats.Float8E8M0, NUM_EXPERTS * OUT_FEATURES * groups_per_n);
        defer allocator.free(logical_scales);
        for (0..logical_scales.len) |i| logical_scales[i] = zml.floats.Float8E8M0.fromF32(1.0);

        const dst = scales_host.items(zml.floats.Float8E8M0);
        for (0..NUM_EXPERTS) |e| {
            for (0..OUT_FEATURES) |n| {
                for (0..groups_per_n) |g| {
                    const src_idx = (e * OUT_FEATURES + n) * groups_per_n + g;
                    const g_outer = g / 4;
                    const g_inner = g % 4;
                    const n4 = n * 4 + g_inner;
                    const dst_idx = (e * scale_outer + g_outer) * (OUT_FEATURES * 4) + n4;
                    dst[dst_idx] = logical_scales[src_idx];
                }
            }
        }
    }

    {
        const offsets = expert_offsets_host.items(i64);
        offsets[0] = 0;
        for (0..NUM_EXPERTS) |e| offsets[e + 1] = offsets[e] + TOKENS_PER_EXPERT[e];
    }

    var activations_buf = try zml.Buffer.fromSlice(io, platform, activations_host);
    defer activations_buf.deinit();
    var weights_buf = try zml.Buffer.fromSlice(io, platform, weights_host);
    defer weights_buf.deinit();
    var scales_buf = try zml.Buffer.fromSlice(io, platform, scales_host);
    defer scales_buf.deinit();
    var offsets_buf = try zml.Buffer.fromSlice(io, platform, expert_offsets_host);
    defer offsets_buf.deinit();

    args.set(.{ activations_buf, weights_buf, scales_buf, offsets_buf });
    exe.callOpts(io, args, &results, .{ .wait = true });

    var out_buf = results.get(zml.Buffer);
    defer out_buf.deinit();
    _ = try out_buf.await(io);

    var out_host = try zml.Slice.alloc(allocator, zml.Shape.init(.{ TOTAL_TOKENS, OUT_FEATURES }, .bf16));
    defer out_host.free(allocator);
    try out_buf.toSlice(io, out_host);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    const w = &stdout.interface;
    try w.print(
        "flashinfer_moe_kernel launch ok. output shape=[{d},{d}] first value={d}\n",
        .{ TOTAL_TOKENS, OUT_FEATURES, out_host.items(zml.floats.BFloat16)[0].toF32() },
    );
}
