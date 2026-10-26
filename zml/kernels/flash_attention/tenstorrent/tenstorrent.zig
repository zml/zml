const std = @import("std");
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

const PrefillOptions = struct { scale: f32, is_causal: bool };
const Prefill = zml.kernel.tenstorrent.Kernel(PrefillOptions, .{
    .name = "flash_attention_prefill",
    .inputs = &.{ "q", "k", "v" },
    .outputs = &.{"output"},
    .run = runPrefill,
});
const MaskedPrefill = zml.kernel.tenstorrent.Kernel(PrefillOptions, .{
    .name = "flash_attention_offset_prefill",
    .inputs = &.{ "q", "k", "v", "mask" },
    .outputs = &.{"output"},
    .run = runPrefill,
});
fn runPrefill(b: *zml.kernel.tenstorrent.Builder, options: PrefillOptions) zml.kernel.tenstorrent.FinishError!void {
    const mask = if (options.is_causal) null else b.arg(3);
    b.returns(&.{b.attention(b.arg(0), b.arg(1), b.arg(2), mask, .{ .is_causal = options.is_causal, .scale = options.scale })});
}

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    // TTNN FlashAttention operates on BF16 tiles. Keep other dtypes exact
    // through the reference path instead of silently reducing their precision.
    if (q.dtype() != .bf16 or k.dtype() != .bf16 or v.dtype() != .bf16) {
        return fallback(q, k, v, token_index);
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
    if (q.dim(.q) != 1 or @mod(q.dim(.hd), 32) != 0 or @mod(k.dim(.k), 32) != 0) {
        return prefillLocal(q, k, v, token_index);
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

fn prefillLocal(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const batched = q.shape().hasTag(.b) != null;
    const batch = if (batched) q.dim(.b) else 1;
    const queries = @divTrunc(q.dim(.q) + 31, 32) * 32;
    const keys = @divTrunc(k.dim(.k) + 31, 32) * 32;
    const width = @divTrunc(q.dim(.hd) + 31, 32) * 32;
    const qp = q.pad(0, .{ .q = zml.Tensor.Pad{ .high = queries - q.dim(.q) }, .hd = zml.Tensor.Pad{ .high = width - q.dim(.hd) } });
    const kp = k.pad(0, .{ .k = zml.Tensor.Pad{ .high = keys - k.dim(.k) }, .hd = zml.Tensor.Pad{ .high = width - k.dim(.hd) } });
    const vp = v.pad(0, .{ .k = zml.Tensor.Pad{ .high = keys - v.dim(.k) }, .hd = zml.Tensor.Pad{ .high = width - v.dim(.hd) } });
    const qc = if (batched) qp.transpose(.{ .b, .h, .q, .hd }) else qp.transpose(.{ .h, .q, .hd }).reshape(.{ batch, q.dim(.h), queries, width });
    const kc = if (batched) kp.transpose(.{ .b, .h, .k, .hd }) else kp.transpose(.{ .h, .k, .hd }).reshape(.{ batch, k.dim(.h), keys, width });
    const vc = if (batched) vp.transpose(.{ .b, .h, .k, .hd }) else vp.transpose(.{ .h, .k, .hd }).reshape(.{ batch, v.dim(.h), keys, width });
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    // Valid full prefill has t=0. Native causality avoids even a dense mask.
    const result = if (q.dim(.q) == k.dim(.k))
        Prefill.call(.{ .q = qc, .k = kc, .v = vc }, .{ .output = qc.shape() }, .{ .scale = scale, .is_causal = true }).output
    else blk: {
        // ZML aligns queries at t+i, not at the bottom-right of the KV buffer.
        // Reuse the last real query's mask for padded query rows, so no row is
        // fully masked; padded keys remain invisible and padded outputs are cut.
        const mask_shape: zml.Shape = .init(.{ .b = batch, .q = queries, .k = keys }, .i32);
        const row = zml.Tensor.iota(mask_shape, .q).minimum(.scalar(q.dim(.q) - 1, .i32));
        const column = zml.Tensor.iota(mask_shape, .k);
        const position = token_index.convert(.i32).broadcast(mask_shape, if (token_index.rank() == 0) &.{} else &.{0});
        const allowed = column.cmp(.LE, position.add(row)).logical(.AND, column.cmp(.LT, .scalar(k.dim(.k), .i32)));
        const mask = allowed.select(.scalar(0, .bf16), .scalar(-std.math.inf(f32), .bf16)).reshape(.{ batch, 1, queries, keys });
        break :blk MaskedPrefill.call(.{ .q = qc, .k = kc, .v = vc, .mask = mask }, .{ .output = qc.shape() }, .{ .scale = scale, .is_causal = false }).output;
    };
    const tagged = result.reshape(.{ .b = batch, .h = q.dim(.h), .q = queries, .hd = width }).slice(.q, .{ .end = q.dim(.q) }).slice(.hd, .{ .end = q.dim(.hd) });
    return if (batched) tagged.transpose(q.shape()) else tagged.reshape(.{ .h = q.dim(.h), .q = q.dim(.q), .hd = q.dim(.hd) }).transpose(q.shape());
}
