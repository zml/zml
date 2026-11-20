const stdx = @import("stdx");

const zml = @import("zml/core");
pub const AttentionOptions = @import("zml/kernels/paged_attention/options").AttentionOptions;
const kernel = @import("attention.zig");
const sparse = @import("sparse_attention.zig");
const sm10x = @import("sm10x/sm10x.zig");
const sm120 = @import("sm120/sm120.zig");

test {
    _ = kernel;
    _ = sparse;
    _ = sm10x;
    _ = sm120;
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
        return self.block_table.byteSize() + self.seq_lens.byteSize() + self.query_start_len.byteSize();
    }

    pub fn options(self: Parameters) Options {
        return self.options_;
    }
};

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major > 8 or (cc.major == 8 and cc.minor >= 9);
}

pub fn multiprocessorCount(platform: *const zml.Platform) usize {
    return platform.coreCountPerDevice() orelse 1;
}

pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    stdx.debug.assert(isAvailable(zml.Compiler.current().platform), "CuTe paged attention requires SM89 or newer", .{});
    stdx.debug.assert(q.dtype() == k.dtype() and q.dtype() == v.dtype() and (q.dtype() == .bf16 or q.dtype() == .f16), "CuTe paged attention requires matching BF16 or FP16 Q/K/V", .{});
    stdx.debug.assert(q.dim(.hd) == 32 or q.dim(.hd) == 64 or q.dim(.hd) == 128 or q.dim(.hd) == 256 or q.dim(.hd) == 512, "CuTe paged attention supports head dimensions 32, 64, 128, 256 and 512", .{});

    return switch (opts.mask) {
        .indices => pagedSparseAttention(parameters, q, k, v, opts),
        else => {
            if (sm10x.isAvailable(zml.Compiler.current().platform))
                return sm10x.pagedAttention(parameters, q, k, v, opts);
            if (sm120.isAvailable(zml.Compiler.current().platform))
                return sm120.pagedAttention(parameters, q, k, v, opts);
            return pagedAttentionGeneric(parameters, q, k, v, opts);
        },
    };
}

pub fn pagedAttentionGeneric(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    stdx.debug.assert(opts.mask != .indices, "pagedAttention does not support indices mask", .{});

    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });

    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});

    const indices: zml.Tensor = .zeroes(.init(.{ q_.dim(.b), 1 }, .i32));

    const cfg: kernel.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages = k_.dim(.page),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .causal = opts.mask.isCausal(),
        .window = opts.mask.slidingWindow() orelse 0,
        .scale = opts.scale orelse (1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd))))),
        .sink = opts.sink != null,
        .indexed = false,
        .topk = indices.dim(-1),
    };

    const sink: zml.Tensor = if (opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else .zeroes(.init(.{cfg.kv_heads * cfg.group}, .f32));

    return kernel.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
        .indices = indices,
    }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.group), 1 },
        .block = .{ 32, 1, 1 },
    }).out.transpose(q.shape());
}

fn pagedSparseAttention(parameters: Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    stdx.debug.assert(opts.mask == .indices, "Sparse attention requires an indices mask", .{});

    const q_ = q.transpose(.{ .b, .hkv, .hg, .hd });
    const k_ = k.transpose(.{ .page, .k_chunk, .hkv, .hd });
    const v_ = v.transpose(.{ .page, .k_chunk, .hkv, .hd });

    stdx.debug.assert(q_.dim(.hkv) == k_.dim(.hkv) and k_.shape().eql(v_.shape()) and k_.dim(.hd) == q_.dim(.hd) and q_.dim(.hg) > 0, "CuTe paged attention requires aligned local Q/KV heads and matching caches", .{});

    stdx.debug.assert(opts.mask.indices.rank() == 2 and opts.mask.indices.dtype() == .i32, "Sparse indices must be an i32 [query, topk] tensor, got {f}", .{opts.mask.indices.shape()});
    stdx.debug.assert(opts.mask.indices.shape().hasTags(.{ .b, .topk }), "Sparse indices require .b and .topk tags", .{});
    stdx.debug.assert(opts.mask.indices.dim(.b) == q.dim(.b) and opts.mask.indices.dim(.topk) > 0, "Sparse indices {f} must match queries {f} with a nonempty selection capacity", .{ opts.mask.indices.shape(), q.shape() });
    const indices = opts.mask.indices.transpose(.{ .b, .topk });

    const all_decode = !parameters.options_.is_prefill;
    const short_selection = all_decode and q_.dim(.hd) <= 128 and indices.dim(.topk) <= 128;
    const fine_tile = all_decode and q_.dim(.b) == 1 and ((q_.dim(.hd) == 512 and indices.dim(.topk) <= 512) or (q_.dim(.hd) >= 128 and q_.dim(.hd) < 512 and q_.dim(.hg) <= 16));
    const split_features = all_decode and (q_.dim(.hd) == 512 or (fine_tile and indices.dim(.topk) > 128));

    var cfg: sparse.Config = .{
        .head_dim = q_.dim(.hd),
        .tokens = q_.dim(.b),
        .batch = parameters.block_table.dim(.b),
        .kv_heads = k_.dim(.hkv),
        .group = q_.dim(.hg),
        .pages_per_seq = parameters.block_table.dim(.p),
        .page_size = k_.dim(.k_chunk),
        .topk = indices.dim(.topk),
        .dtype = zml.kernel.cute.from(q_.dtype()),
        .kv_tile = if (short_selection) (if (indices.dim(.topk) <= 32) @as(i64, 32) else 128) else if (fine_tile) 16 else if (all_decode or q_.dim(.hd) == 512) 32 else 64,
        .warps = if (split_features) 8 else 4,
        .scale = opts.scale orelse 1 / @sqrt(@as(f32, @floatFromInt(q_.dim(.hd)))),
        .sink = opts.sink != null,
        .decode = all_decode and parameters.options_.max_seqlen_q == 1,
    };

    const platform = zml.Compiler.current().platform;
    const cc = zml.platform.cuda.computeCapability(platform).?;

    const narrowDecode = cfg.decode and cfg.tokens == 1 and cfg.batch == 1 and
        ((cfg.head_dim == 64 and cfg.group <= 8 and cfg.topk > 128) or
            (cfg.head_dim == 512 and cfg.group <= 16 and cfg.topk > 512));

    // Underfilled GB300 grids benefit from eight real query rows per CTA.
    // Vector merges amortize each partition's softmax state across features.
    const narrowWork = cfg.kv_heads * @divTrunc(cfg.group + 7, 8) * @divTrunc(cfg.topk + 31, 32);
    if (cc.major == 10 and cc.minor == 3 and narrowDecode and narrowWork <= @as(i64, @intCast(multiprocessorCount(platform)))) {
        cfg.head_rows = 8;
        cfg.warps = 8;
        cfg.kv_tile = 32;
        cfg.merge_vector = if (cfg.head_dim == 512) 8 else 4;
        cfg.merge_lanes = if (cfg.head_dim == 512) 32 else 8;
    }

    cfg.merge_parts = if (cfg.head_dim >= 128 and cfg.head_dim <= 256 and cfg.tokens * cfg.kv_heads * cfg.group <= 16) 16 else 8;
    stdx.debug.assert(cfg.merge_lanes == 8 or cfg.merge_lanes == 16 or cfg.merge_lanes == 32, "Invalid sparse merge lanes", .{});
    stdx.debug.assert(@mod(cfg.head_dim, cfg.mergeFeatures()) == 0, "Sparse merge tile must divide the head dimension", .{});

    const work = @max(1, cfg.tokens * cfg.kv_heads * cfg.headTiles());

    // Short selections avoid a merge; longer decode selections expose one
    // selected tile per partition until the device has enough parallel work.
    const target = @as(i64, @intCast(multiprocessorCount(zml.Compiler.current().platform))) * 16;
    const tiles = @divTrunc(cfg.topk + cfg.kv_tile - 1, cfg.kv_tile);
    var desired: i64 = 1;
    while (desired * work < target) desired *= 2;
    cfg.splits = if (short_selection) 1 else if (all_decode) @max(1, @min(128, @min(tiles, desired))) else if (cfg.head_dim <= 128) 1 else @max(1, @min(32, @min(@divTrunc(@divTrunc(target, 8) + work - 1, work), @divTrunc(cfg.topk, 128))));
    const sink = if (opts.sink) |s| s.convert(.f32).reshape(.{cfg.kv_heads * cfg.group}) else zml.Tensor.scalar(0, .f32);

    if (cfg.group == 1 and cfg.topk <= 8) return sparse.Tiny.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
        .indices = indices,
    }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads), 1 },
        .block = .{ if (cfg.topk > 4) @as(i32, 256) else 128, 1, 1 },
    }).out.transpose(q.shape());

    const out = sparse.Kernel.call(.{
        .q = q_,
        .k = k_,
        .v = v_,
        .table = parameters.block_table,
        .lengths = parameters.seq_lens,
        .starts = parameters.query_start_len,
        .sink = sink,
        .indices = indices,
    }, .{ .out = if (cfg.splits > 1) zml.Shape.init(.{ cfg.tokens, cfg.kv_heads * cfg.group, cfg.splits, cfg.partialWidth() }, .f32) else q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.headTiles()), @intCast(cfg.splits) },
        .block = .{ @intCast(cfg.warps * 32), 1, 1 },
    }).out;

    const result = if (cfg.splits > 1) sparse.Merge.call(.{ .partials = out, .starts = parameters.query_start_len, .sink = sink }, .{ .out = q_.shape() }, .{
        .cfg = cfg,
        .grid = .{ @intCast(cfg.tokens), @intCast(cfg.kv_heads * cfg.group), @intCast(@divExact(cfg.head_dim, cfg.mergeFeatures())) },
        .block = .{ 128, 1, 1 },
    }).out else out;

    return result.transpose(q.shape());
}
