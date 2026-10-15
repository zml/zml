//! qaic: the graph plugin's paged attention over its own KV pool (`qaic$paged_attn`). The attention writes the step's new K
//! and V rows into the pool itself, in place, and reads its pages there: the KV cache tensors are never read or written
//! (`KvCache.update` leaves them; they size the pool by their page count: a cache a pool). Pages of 16 rows, head_dim
//! a multiple of 64, causal masks, optionally a sliding window (each token sees its last `sliding_window` positions)
//! and attention sinks (each head's extra softmax logit, `opts.sink` [h]).
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const triton = @import("triton_attention.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const Tensor = zml.Tensor;

/// q [b, hkv, hg, hd], k and v [b, hkv, hd] (this step's rows), the cache [page, k_chunk, hkv, hd]; each device the pool
/// of its heads (q's partitioning).
pub fn pagedAttention(parameters: triton.paged.Parameters, q: Tensor, k: Tensor, v: Tensor, k_cache: Tensor, opts: AttentionOptions) Tensor {
    const window: u32 = switch (opts.mask) {
        .standard => |mask| b: {
            stdx.debug.assert(mask.causal, "qaic paged attention is causal", .{});
            break :b mask.sliding_window orelse 0;
        },
        .indices => stdx.debug.panic("qaic paged attention has no sparse mask", .{}),
    };
    if (opts.sink) |sink| stdx.debug.assert(sink.rank() == 1, "qaic paged attention: sinks [h], got {f}", .{sink});
    const tokens = q.dim(.b);
    const page_size = k_cache.dim(.k_chunk);
    stdx.debug.assert(page_size == 16, "qaic paged attention takes pages of 16 rows, got {}", .{page_size});
    // Each token's (`.t`) sequence, position and cache slot (seq_lens counts the step's tokens); padding tokens: no slot.
    const sequence_count = parameters.seq_lens.dim(.b);
    const sequence = triton.paged.tokenToSequence(parameters.query_start_len, tokens).rename(.{ .q = .t });
    const token = Tensor.iota(sequence.shape(), .t);
    const query_end = parameters.query_start_len.slice(.b, .{ .start = 1 }).gather(.{ .b = sequence }, .{});
    const position = parameters.seq_lens.gather(.{ .b = sequence }, .{}).sub(query_end).add(token);
    const total = parameters.query_start_len.slice(.b, .single(sequence_count));
    const active = token.cmp(.LT, total.broad(sequence.shape()));
    const pages = parameters.block_table.gather(.{ .b = sequence }, .{}); // [t, p]
    const page = position.divByConst(page_size);
    const own_page = Tensor.iota(pages.shape(), .p).cmp(.EQ, page.broad(pages.shape())).select(pages, .zeroes(pages.shape())).sum(.p).squeeze(.p);
    const slot = active.select(own_page.scale(page_size).add(position.remainderConst(page_size)), .scalar(std.math.maxInt(i32), .i32));
    const lens = active.select(position.addConstant(1), .scalar(0, .i32)); // (causal: the keys up to the token's own)
    const scale = opts.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    const context: Context = .{
        .q = q,
        .k = headsLike(k, q),
        .v = headsLike(v, q),
        .slot = replicatedOn(slot, q),
        .pages = replicatedOn(pages, q),
        .lens = replicatedOn(lens, q),
        .scale = scale,
        .num_pages = k_cache.dim(.page),
        .window = window,
        .sink = if (opts.sink) |sink| sinksLike(sink, q) else null,
    };
    return zml.ops.manualComputation(Context.body, context, q.shape());
}

/// x's `.hkv` axis partitioned as q's, its other axes replicated: every device the KV heads of its query heads (the
/// kernel pairs them by position).
fn headsLike(x: Tensor, q: Tensor) Tensor {
    const mesh = q.shape()._sharding.mesh orelse return x;
    return x.withPartitioning(mesh, zml.Sharding.Partitioning.replicated(x.rank()).set(x.axis(.hkv), q.shape().partition(.hkv)));
}

/// sinks [h] (q's heads in order) partitioned as q's heads: every device its heads' sinks.
fn sinksLike(sink: Tensor, q: Tensor) Tensor {
    const mesh = q.shape()._sharding.mesh orelse return sink;
    return sink.withPartitioning(mesh, zml.Sharding.Partitioning.replicated(1).set(0, q.shape().partition(.hkv)));
}

fn replicatedOn(x: Tensor, like: Tensor) Tensor {
    return x.withPartitioning(like.shape()._sharding.mesh orelse return x, zml.Sharding.Partitioning.replicated(x.rank()));
}

const Context = struct {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    slot: Tensor,
    pages: Tensor,
    lens: Tensor,
    scale: f32,
    num_pages: i64,
    window: u32,
    sink: ?Tensor,

    fn body(self: Context, _: zml.Shape) Tensor {
        const kv_heads = self.q.dim(.hkv);
        const qh = self.q.merge(.{ .h = .{ .hkv, .hg } }).transpose(.{ .b, .h, .hd }).convert(.f16);
        const kh = self.k.transpose(.{ .b, .hkv, .hd }).convert(.f16);
        const vh = self.v.transpose(.{ .b, .hkv, .hd }).convert(.f16);
        const config = .{ .scale = self.scale, .pages = self.num_pages, .window = @as(i64, self.window) };
        const slot = self.slot.convert(.i32);
        const pages = self.pages.convert(.i32);
        const lens = self.lens.convert(.i32);
        const out = if (self.sink) |sink|
            zml.ops.customCall("qaic$paged_attn", .{ qh, kh, vh, slot, pages, lens, sink.convert(.f32) }, qh.shape(), config, .{ .has_side_effect = false })
        else
            zml.ops.customCall("qaic$paged_attn", .{ qh, kh, vh, slot, pages, lens }, qh.shape(), config, .{ .has_side_effect = false });
        return out.convert(self.q.dtype()).splitAxis(.h, .{ .hkv = kv_heads, .hg = .auto }).transpose(self.q.shape());
    }
};
