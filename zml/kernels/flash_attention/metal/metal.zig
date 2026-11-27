const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const tiled = @import("../../paged_attention/metal/paged_attn.zig").PagedAttentionTiled;

pub const Metadata = struct {
    num_tokens: zml.Tensor,

    pub fn init() Metadata {
        return .{ .num_tokens = .fromShape(.scalar(.u32)) };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        return .{ .num_tokens = try zml.Buffer.scalar(io, platform, 0, .u32) };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        self.num_tokens.deinit();
    }
};

pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, causal: bool) zml.Tensor {
    if (!causal) return nonCausalAttention(q, k, v);

    const qc = q.transpose(.{ .h, .q, .hd });
    const kc = k.transpose(.{ .h, .k, .hd });
    const vc = v.transpose(.{ .h, .k, .hd });
    const tok_i32 = token_index.convert(.i32);

    const attn = if (q.dim(.q) > 1)
        zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32, metadata.num_tokens }, qc.shape(), .{}, .{ .has_side_effect = false })
    else
        zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32 }, qc.shape(), .{}, .{ .has_side_effect = false });

    return attn.transpose(q.shape());
}

/// Reuse the native tiled kernel with contiguous pages. Only the last page of
/// each sequence is padded; context_lens keeps its padding out of the softmax.
fn nonCausalAttention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
    stdx.debug.assert(q.shape().hasTags(.{ .q, .h, .hd }) and k.shape().hasTags(.{ .k, .h, .hd }) and v.shape().hasTags(.{ .k, .h, .hd }), "Metal attention requires q/h/hd and k/h/hd tags", .{});
    const batched = q.shape().hasTag(.b) != null;
    const rank: usize = if (batched) 4 else 3;
    stdx.debug.assert(q.rank() == rank and k.rank() == rank and v.rank() == rank, "Metal attention expects rank-3 tensors or rank-4 tensors with a batch axis", .{});
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype(), "Metal attention requires matching Q/K/V dtypes", .{});
    stdx.debug.assert(q.dtype() == .bf16 or q.dtype() == .f16, "Non-causal Metal attention supports bf16 and f16", .{});
    stdx.debug.assert(q.dim(.hd) == k.dim(.hd) and k.dim(.hd) == v.dim(.hd), "Metal attention requires matching head dimensions", .{});
    stdx.debug.assert(k.dim(.h) > 0 and q.dim(.h) > 0 and @rem(q.dim(.h), k.dim(.h)) == 0 and k.dim(.h) == v.dim(.h), "Metal attention requires Hq divisible by Hkv and matching K/V heads", .{});
    stdx.debug.assert(q.dim(.q) > 0 and k.dim(.k) > 0 and k.dim(.k) == v.dim(.k), "Metal attention requires positive query/key lengths and matching K/V lengths", .{});
    if (batched) {
        stdx.debug.assert(k.shape().hasTag(.b) != null and v.shape().hasTag(.b) != null and q.dim(.b) > 0 and q.dim(.b) == k.dim(.b) and q.dim(.b) == v.dim(.b), "Metal attention requires matching batch dimensions", .{});
    }

    const batch: i32 = if (batched) @intCast(q.dim(.b)) else 1;
    const query_len: i32 = @intCast(q.dim(.q));
    const key_len: i32 = @intCast(k.dim(.k));
    const heads: i32 = @intCast(q.dim(.h));
    const kv_heads: i32 = @intCast(k.dim(.h));
    const head_dim: i32 = @intCast(q.dim(.hd));
    const tile = tiled.tileConfig(head_dim) orelse std.debug.panic("Non-causal Metal attention: unsupported head dimension {d}", .{head_dim});
    const page_size = 32;
    const pages = std.math.divCeil(i32, key_len, page_size) catch unreachable;
    const padding = pages * page_size - key_len;

    const queries = if (batched) q.transpose(.{ .b, .q, .h, .hd }) else q.transpose(.{ .q, .h, .hd });
    const keys = if (batched) k.transpose(.{ .b, .k, .h, .hd }) else k.transpose(.{ .k, .h, .hd });
    const values = if (batched) v.transpose(.{ .b, .k, .h, .hd }) else v.transpose(.{ .k, .h, .hd });
    const table_shape: zml.Shape = .init(.{ .b = batch, .p = pages }, .i32);
    const block_table = zml.Tensor.iota(table_shape, .b).mul(zml.Tensor.scalar(pages, .i32)).add(zml.Tensor.iota(table_shape, .p));
    const context_lens = zml.Tensor.scalar(key_len, .i32).broad(.init(.{ .b = batch }, .i32));
    const query_starts = zml.Tensor.iota(.init(.{batch + 1}, .i32), 0).mul(zml.Tensor.scalar(query_len, .i32));
    const cfg: tiled.Cfg = .{
        .dtype = zml.kernel.metal.from(q.dtype()),
        .head_size = head_dim,
        .block_size = page_size,
        .bq = tile[0],
        .tile_kv = tile[1],
        .num_threads = tile[2],
        .causal = false,
    };
    // The kernel maps blocks to packed sequences using floor(start / BQ) + seq.
    const query_blocks = @divTrunc(batch * query_len, tile[0]) + batch;
    const out = tiled.Kernel.call(.{
        .q = queries,
        .k_cache = if (padding == 0) keys else keys.pad(0, .{ .k = zml.Tensor.Pad{ .high = padding } }),
        .v_cache = if (padding == 0) values else values.pad(0, .{ .k = zml.Tensor.Pad{ .high = padding } }),
        .block_tables = block_table,
        .context_lens = context_lens,
        .cu_seqlens_q = query_starts,
    }, .{ .out = queries.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(heads), @intCast(query_blocks), 1 },
        .threadgroup = .{ @intCast(tile[2]), 1, 1 },
        .threadgroup_memory = tiled.threadgroupMemory(cfg),
        .constants = .{
            .num_kv_heads = .of(kv_heads),
            .scale = .of(1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)))),
            .softcapping = .of(@as(f32, 0)),
            .max_num_blocks_per_seq = .of(pages),
            .q_stride = .of(heads * head_dim),
            .kv_block_stride = .of(page_size * kv_heads * head_dim),
            .kv_head_stride = .of(head_dim),
            .num_seqs = .of(batch),
            .sliding_window = .of(@as(i32, -1)),
        },
    }).out;
    return out.transpose(q.shape());
}
