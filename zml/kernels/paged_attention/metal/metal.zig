const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const AttentionOptions = @import("../options.zig").AttentionOptions;
const paged_attn = @import("paged_attn.zig");

const log = std.log.scoped(.@"zml/attention/metal");

test {
    _ = paged_attn;
}

pub const Options = struct {
    batch_size: usize,
    max_num_pages: usize,
    max_seqlen_q: usize,
    is_prefill: bool,

    pub fn isPrefill(self: Options) bool {
        return self.is_prefill;
    }

    pub fn maxNumPages(self: Options) usize {
        return self.max_num_pages;
    }
};

pub const Parameters = struct {
    block_table: zml.Tensor,
    seq_lens: zml.Tensor,
    query_start_len: zml.Tensor,
    options_: Options,

    pub fn init(options_: Options) Parameters {
        return .{
            .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
            .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
            .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
            .options_ = options_,
        };
    }

    pub fn allocationSize(self: Parameters) usize {
        var allocation_size: usize = 0;

        allocation_size += self.block_table.byteSize();
        allocation_size += self.seq_lens.byteSize();
        allocation_size += self.query_start_len.byteSize();

        return allocation_size;
    }

    pub fn options(self: Parameters) Options {
        return self.options_;
    }
};

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    if (opts.mask == .indices) @panic("Sparse Attention is not implemented on Metal");

    const num_kv_heads = q.dim(.hkv);

    const qh = q.merge(.{ .h = .{ .hkv, .hg } }).transpose(.{ .b, .h, .hd });
    const kc = k_cache.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const vc = v_cache.transpose(.{ .page, .k_chunk, .hkv, .hd });

    const sliding_window: i32 = if (opts.mask.slidingWindow()) |window| @intCast(window) else -1;

    const out = dispatch(qh, kc, vc, parameters.block_table, parameters.seq_lens, parameters.query_start_len, .{
        .is_causal = opts.mask.isCausal(),
        .sliding_window = sliding_window,
        .scale = opts.scale orelse 1.0 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd)))),
    });

    // [total_q_tokens, num_heads, head_dim] -> {.b, .hkv, .hg, .hd} (q's order)
    return out.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = .auto }).transpose(q.shape());
}

const DispatchOptions = struct {
    is_causal: bool,
    /// Below 0 for global attention.
    sliding_window: i32,
    scale: f32,
};

/// Launches fa_vec_paged for causal bf16 decode and paged_attention_tiled
/// otherwise.
/// `q` and the result are `[tokens, heads, head_dim]`, the caches
/// `[pages, block_size, kv_heads, head_dim]`, `block_table`
/// `[seqs, max_blocks]`, `seq_lens` `[seqs]` and `query_start_len`
/// `[seqs + 1]`, all i32.
fn dispatch(q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, block_table: zml.Tensor, seq_lens: zml.Tensor, query_start_len: zml.Tensor, opts: DispatchOptions) zml.Tensor {
    const dtype = q.dtype();
    if (dtype != .bf16 and dtype != .f16) std.debug.panic("paged attention: Metal takes bf16 or f16 queries, not {t}", .{dtype});
    if (k_cache.dtype() != dtype or v_cache.dtype() != dtype) std.debug.panic("paged attention: Metal takes caches of the queries' {t}, not {t} and {t}", .{ dtype, k_cache.dtype(), v_cache.dtype() });

    const total_q_tokens: i32 = @intCast(q.dim(0));
    const num_heads: i32 = @intCast(q.dim(1));
    const head_dim: i32 = @intCast(q.dim(2));
    const block_size: i32 = @intCast(k_cache.dim(1));
    const num_kv_heads: i32 = @intCast(k_cache.dim(2));
    const num_seqs: i32 = @intCast(block_table.dim(0));
    const max_blocks: i32 = @intCast(block_table.dim(1));

    const vec_decode = opts.is_causal and total_q_tokens == num_seqs and
        (head_dim == 32 or head_dim == 64 or head_dim == 128 or head_dim == 256 or head_dim == 512) and
        dtype == .bf16;

    if (vec_decode) {
        const args: paged_attn.FaVecPaged.HostArgs = .{
            .num_heads = num_heads,
            .n_kv = num_kv_heads,
            .max_blocks = max_blocks,
            .num_seqs = num_seqs,
            .scale = opts.scale,
            .sliding_window = opts.sliding_window,
        };

        const inputs: paged_attn.FaVecPaged.Kernel.Inputs = .{
            .q = q,
            .k_cache = k_cache,
            .v_cache = v_cache,
            .block_tables = block_table,
            .seq_lens = seq_lens,
            .cu_seqlens_q = query_start_len,
        };

        const kv_capacity = max_blocks * block_size;
        const span = if (opts.sliding_window < 0) kv_capacity else @min(kv_capacity, opts.sliding_window + 32);
        const nrows = total_q_tokens * num_heads;
        const launch = vecLaunch(nrows, @divExact(num_heads, num_kv_heads), span, coreCount());

        const cfg: paged_attn.FaVecPaged.Cfg = .{
            .head_dim = head_dim,
            .block_size = block_size,
            .position_stride = num_kv_heads * head_dim,
            .heads = launch.heads,
            .splits = launch.splits,
        };
        const dst = if (launch.splits == 1) q.shape() else zml.Shape.init(.{nrows * (head_dim * launch.splits + 2 * launch.splits)}, .f32);
        const out = paged_attn.FaVecPaged.Kernel.call(inputs, .{ .dst = dst }, .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(launch.splits, launch.splits_per_threadgroup)), @intCast(@divExact(num_heads, launch.heads)), @intCast(total_q_tokens) },
            .threadgroup = .{ 32, @intCast(launch.heads * launch.splits_per_threadgroup), 1 },
            .constants = .{ .args = .of(args) },
        }).dst;
        if (launch.splits == 1) return out;

        return paged_attn.FaVecPagedReduce.Kernel.call(.{ .htmp = out, .seq_lens = seq_lens, .cu_seqlens_q = query_start_len }, .{ .dst = q.shape() }, .{
            .cfg = .{ .splits = launch.splits, .head_dim = head_dim },
            .grid = .{ @intCast(nrows), @intCast(@divExact(head_dim, 4 * paged_attn.FaVecPagedReduce.simdgroups)), 1 },
            .threadgroup = .{ 32, paged_attn.FaVecPagedReduce.simdgroups, 1 },
            .constants = .{ .args = .of(args) },
        }).dst;
    }

    const tile = paged_attn.PagedAttentionTiled.tileConfig(head_dim) orelse std.debug.panic("paged attention: head_dim {d} unsupported", .{head_dim});

    const cfg: paged_attn.PagedAttentionTiled.Cfg = .{
        .dtype = zml.kernel.metal.from(dtype),
        .head_size = head_dim,
        .block_size = block_size,
        .bq = tile[0],
        .tile_kv = tile[1],
        .num_threads = tile[2],
        .causal = opts.is_causal,
    };

    const total_q_blocks = @divTrunc(total_q_tokens, tile[0]) + num_seqs;

    const out = paged_attn.PagedAttentionTiled.Kernel.call(.{
        .q = q,
        .k_cache = k_cache,
        .v_cache = v_cache,
        .block_tables = block_table,
        .context_lens = seq_lens,
        .cu_seqlens_q = query_start_len,
    }, .{ .out = q.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(num_heads), @intCast(total_q_blocks), 1 },
        .threadgroup = .{ @intCast(tile[2]), 1, 1 },
        .threadgroup_memory = paged_attn.PagedAttentionTiled.threadgroupMemory(cfg),
        .constants = .{
            .num_kv_heads = .of(num_kv_heads),
            .scale = .of(opts.scale),
            .softcapping = .of(@as(f32, 0)),
            .max_num_blocks_per_seq = .of(max_blocks),
            .q_stride = .of(num_heads * head_dim),
            .kv_block_stride = .of(block_size * num_kv_heads * head_dim),
            .kv_head_stride = .of(head_dim),
            .num_seqs = .of(num_seqs),
            .sliding_window = .of(opts.sliding_window),
        },
    }).out;

    // The tiled kernel leaves the padding rows untouched, so they are zeroed in
    // its output. Zeroing a fresh buffer first, from cu_seqlens_q alone, would
    // make XLA share that buffer across every call and copy it for each.
    return paged_attn.ZeroPaddingRows.Kernel.call(.{ .cu_seqlens_q = query_start_len, .out = out }, .{ .out = q.shape() }, .{
        .cfg = .{ .dtype = cfg.dtype },
        .grid = .{ @intCast(total_q_tokens), 1, 1 },
        .threadgroup = .{ 256, 1, 1 },
        .constants = .{ .num_seqs = .of(num_seqs), .row_stride = .of(num_heads * head_dim) },
    }).out;
}

const VecLaunch = struct {
    heads: i32,
    splits: i32,
    splits_per_threadgroup: i32,
};

/// At most 256 threads, which every Apple GPU pipeline accepts. Splits past a
/// row's length exit early, so `splits` follows the core and row counts.
fn vecLaunch(rows: i32, group: i32, span: i32, cores: i32) VecLaunch {
    var heads: i32 = @min(group, 4);
    while (@rem(group, heads) != 0) heads -= 1;
    const want: i32 = if (rows <= 2 * cores) 128 else 64;
    const chunks = @divFloor(span + 31, 32);
    var splits: i32 = 1;
    while (splits * 2 <= @min(want, chunks)) splits *= 2;
    return .{ .heads = heads, .splits = splits, .splits_per_threadgroup = @min(splits, @divFloor(8, heads)) };
}

fn coreCount() i32 {
    const platform = zml.Compiler.current().platform;
    if (platform.devices.len == 0) return 1;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 1;
    return @intCast(std.math.clamp(attribute.int64, 1, 1024));
}

test vecLaunch {
    try std.testing.expectEqual(VecLaunch{ .heads = 4, .splits = 128, .splits_per_threadgroup = 2 }, vecLaunch(16, 4, 128000, 40));
    try std.testing.expectEqual(VecLaunch{ .heads = 2, .splits = 64, .splits_per_threadgroup = 4 }, vecLaunch(256, 2, 128000, 40));
    try std.testing.expectEqual(VecLaunch{ .heads = 4, .splits = 64, .splits_per_threadgroup = 2 }, vecLaunch(512, 4, 128000, 40));
    try std.testing.expectEqual(VecLaunch{ .heads = 3, .splits = 8, .splits_per_threadgroup = 2 }, vecLaunch(48, 6, 288, 40));
    try std.testing.expectEqual(VecLaunch{ .heads = 1, .splits = 1, .splits_per_threadgroup = 1 }, vecLaunch(8, 1, 20, 40));
}
