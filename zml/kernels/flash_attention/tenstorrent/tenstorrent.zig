const zml = @import("../../../zml.zig");

pub fn isAvailable(platform: *const zml.Platform) bool {
    return platform.target == .tenstorrent and platform.devices.len == 1;
}

const Decode = zml.kernel.tenstorrent.Kernel(void, .{
    .name = "attention_decode",
    .inputs = &.{ "q", "k", "v", "position" },
    .outputs = &.{"output"},
    .run = run,
});

fn run(b: *zml.kernel.tenstorrent.Builder, _: void) zml.kernel.tenstorrent.FinishError!void {
    b.returns(&.{b.attentionDecode(b.arg(0), b.arg(1), b.arg(2), b.arg(3))});
}

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    // TTNN's specialized decode operation consumes one query. Keep the
    // general implementation for prefill and unsupported physical shapes.
    if (q.dim(.q) != 1 or q.dtype() != .bf16 or k.dtype() != .bf16 or v.dtype() != .bf16 or
        @mod(q.dim(.hd), 32) != 0 or @mod(k.dim(.k), 32) != 0)
    {
        return zml.attention.attention(q, k, v, token_index, .{ .stablehlo = {} }, .{ .stablehlo = {} });
    }
    const batched = q.shape().hasTag(.b) != null;
    const batch = if (batched) q.dim(.b) else 1;
    const qc = if (batched) q.transpose(.{ .q, .b, .h, .hd }) else q.transpose(.{ .q, .h, .hd }).reshape(.{ 1, batch, q.dim(.h), q.dim(.hd) });
    const kc = if (batched) k.transpose(.{ .b, .h, .k, .hd }) else k.transpose(.{ .h, .k, .hd }).reshape(.{ batch, k.dim(.h), k.dim(.k), k.dim(.hd) });
    const vc = if (batched) v.transpose(.{ .b, .h, .k, .hd }) else v.transpose(.{ .h, .k, .hd }).reshape(.{ batch, v.dim(.h), v.dim(.k), v.dim(.hd) });
    // Match the clamped gather used by the general causal mask. Clamp before
    // casting so large unsigned positions cannot wrap into negative indices.
    const position = token_index.maximum(.scalar(0, token_index.dtype())).minimum(.scalar(k.dim(.k) - 1, token_index.dtype())).convert(.i32).broadcastLeft(.init(.{batch}, .i32));
    const result = Decode.call(.{ .q = qc, .k = kc, .v = vc, .position = position }, .{ .output = qc.shape() }, {}).output;
    return if (batched) result.transpose(q.shape()) else result.reshape(.{ .q = 1, .h = q.dim(.h), .hd = q.dim(.hd) }).transpose(q.shape());
}
