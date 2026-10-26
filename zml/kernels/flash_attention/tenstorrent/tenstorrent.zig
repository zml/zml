const zml = @import("../../../zml.zig");

pub fn isAvailable(platform: *const zml.Platform) bool {
    return platform.target == .tenstorrent and platform.devices.len <= 2;
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
    if (zml.Compiler.current().platform.devices.len > 1) {
        const mesh = k.shape()._sharding.mesh orelse return fallback(q, k, v, token_index);
        var head_partition = k.shape().partition(.h);
        // Computed cache slices retain their mesh but leave partitioning open
        // for Shardy. Constrain the model axis before entering the local kernel.
        if (head_partition == .open) {
            const axis = mesh.resolveLogicalAxis(zml.Shape.toTag(.model)) orelse return fallback(q, k, v, token_index);
            head_partition = .sharded(@intCast(axis));
        }
        if (!head_partition.isSharded()) return fallback(q, k, v, token_index);
        const q_shape = q.shape().withPartitioning(mesh, zml.Sharding.Partitioning.replicated(q.rank()).set(q.shape().axis(.h), head_partition));
        const k_shape = k.shape().withPartitioning(mesh, zml.Sharding.Partitioning.replicated(k.rank()).set(k.shape().axis(.h), head_partition));
        const v_shape = v.shape().withPartitioning(mesh, zml.Sharding.Partitioning.replicated(v.rank()).set(v.shape().axis(.h), head_partition));
        _ = zml.Sharding.shardedShape(q_shape) catch return fallback(q, k, v, token_index);
        _ = zml.Sharding.shardedShape(k_shape) catch return fallback(q, k, v, token_index);
        _ = zml.Sharding.shardedShape(v_shape) catch return fallback(q, k, v, token_index);
        return zml.ops.manualComputation(LocalAttention.run, .{
            .q = q.withPartitioning(mesh, q_shape._sharding.partition),
            .k = k.withPartitioning(mesh, k_shape._sharding.partition),
            .v = v.withPartitioning(mesh, v_shape._sharding.partition),
            .position = token_index.withPartitioning(mesh, zml.Sharding.Partitioning.replicated(token_index.rank())),
        }, q_shape);
    }
    return attentionLocal(q, k, v, token_index);
}

fn fallback(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, position: zml.Tensor) zml.Tensor {
    return zml.attention.attention(q, k, v, position, .{ .stablehlo = {} }, .{ .stablehlo = {} });
}

const LocalAttention = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    position: zml.Tensor,

    fn run(self: @This(), _: zml.Shape) zml.Tensor {
        return attentionLocal(self.q, self.k, self.v, self.position);
    }
};

fn attentionLocal(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
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
