const std = @import("std");
const stdx = zml.stdx;

const zml = @import("zml");
const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.deepseek_v4);

pub const QuantizationConfig = struct {
    weight_block_size: []usize,
};

pub const Config = struct {
    max_position_embeddings: u32,
    expert_dtype: []const u8, //< find better type
    num_hidden_layers: u16,
    rms_norm_eps: f32,
    routed_scaling_factor: f32,
    num_hash_layers: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    swiglu_limit: f32,
    compress_ratios: []i64,
    rope_scaling: RopeScaling,
    rope_theta: f32,
    compress_rope_theta: f32,
    quantization_config: QuantizationConfig,
    sliding_window: u32,
    num_attention_heads: u32,
    head_dim: u32,
    qk_rope_head_dim: u32,
    o_groups: u32,
    o_lora_rank: u32,
    index_head_dim: u32,
    index_n_heads: u32,
    index_topk: u32,
    hc_eps: f32,
    hc_mult: u32,
    hc_sinkhorn_iters: u32,
};

const RopeScaling = struct {
    beta_fast: f32,
    beta_slow: f32,
    factor: f32,
    original_max_position_embeddings: u32,
};

const RopeOpts = struct {
    beta_fast: f32,
    beta_slow: f32,
    factor: f32,
    original_max_position_embeddings: u32,
    rope_theta: f32,
};

pub const Buffers = zml.Bufferized(Model);

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,
    tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, eps: f32, tag: anytype) RmsNorm {
        return .{
            .weight = weight,
            .eps = eps,
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const norm = zml.nn.rmsNorm(x, self.tag, self.eps);
        return norm.mul(self.weight.broad(x.shape()));
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }
};

pub const LinearF32 = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
    tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, bias: ?zml.Tensor, tag: anytype) LinearF32 {
        return .{
            .weight = weight,
            .bias = bias,
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn forward(self: LinearF32, x: zml.Tensor) zml.Tensor {
        var y = x.dot(self.weight.convert(.f32), self.tag);
        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LinearF32)) void {
        self.weight.deinit();
    }
};

// fn dequantizeWeights(weight: zml.Tensor, scale: zml.Tensor, block_size: usize) zml.Tensor {
//     const w = weight.splitAxis(0, .{ .auto, block_size }).splitAxis(-1, .{ .auto, block_size });
//     const s= scale.insertAxes(1, .{.b}).insertAxes(.last, .{.d}).withTags(.{.a, .b, .c, .d});
//     return w.convert(.bf16).mul(s.convert(.bf16).broad(w.shape())).reshape(weight.convert(.bf16).shape());
// }

// TODO: rewrite
fn dequantizeWeights(weight: zml.Tensor, scale: zml.Tensor, block_size: usize) zml.Tensor {
    // Reshape scale from [8, 32] -> [8, 1, 32, 1] to inject dummy block dimensions
    const scale_4d_shape = scale.shape().insert(1, .{1}).insert(.last, .{1});
    const scale_4d = scale.convert(.bf16).reshape(scale_4d_shape);

    // Broadcast along the block dimensions (block size = 128)
    // [8, 1, 32, 1] -> [8, 128, 32, 128]
    const broad_shape = scale.shape().insert(1, .{block_size}).insert(.last, .{block_size});
    const scale_broad = scale_4d.broad(broad_shape);

    // Flatten the 4D expanded scale back to 2D matrix matching the weights: [1024, 4096]
    const target_shape = weight.shape();
    const scale_expanded = scale_broad.reshape(target_shape);

    // Element-wise multiplication completes the dequantization pipeline
    return weight.convert(.bf16).mul(scale_expanded);
}

fn to_fp8(x: zml.Tensor, block_size: u32) struct { zml.Tensor, zml.Tensor } {
    // x = [seq, d=4096]
    // block = 128
    stdx.debug.assert(@mod(x.dim(-1), block_size) == 0, "last dimension {} must be divisible by block_size={}", .{x.dim(-1), block_size});

    // x_blocked = [seq, 32, 128]
    const x_blocked = x.splitAxis(-1, .{ .n = .auto, .m = block_size }).convert(.f32);

    const min_value = zml.Tensor.scalar(1e-4, x_blocked.dtype());

    var max_block = x_blocked.abs().max(-1);
    max_block = zml.Tensor.select(max_block.cmp(.LT, min_value), min_value, max_block);

    const fp8_min = zml.Tensor.constant(zml.DataType.f8e4m3fn.minValue()).convert(x_blocked.dtype());
    const fp8_max = zml.Tensor.constant(zml.DataType.f8e4m3fn.maxValue()).convert(x_blocked.dtype());
    const fp8_max_inv = ones(fp8_max.shape()).div(fp8_max);

    const scale = max_block.mul(fp8_max_inv);
    // scale.print("scale");
    const x_fp8 = x_blocked.div(scale).clamp(fp8_min, fp8_max).reshape(x.shape());
    // x_fp8.convert(.f8e4m3fn).print("x_fp8");
    // const s = scale.convert(.f8e8m0).reshape(.{ scale.dim(0), scale.dim(1)});
    // s.print("scale");

    return .{ x_fp8.convert(.f8e4m3fn), scale.convert(.f8e8m0).reshape(.{ scale.dim(0), scale.dim(1) }) };
}

const FP8Linear = struct {
    scale: zml.Tensor,
    weight: zml.Tensor,
    block_size: usize,
    tag: zml.Shape.Tag,

    pub fn init(store: zml.io.TensorStore.View, tagz: anytype, block_size: usize, proj_tag: anytype) FP8Linear {
        return .{
            .scale = store.createTensor("scale", null, .replicated),
            .weight = store.createTensor("weight", tagz, .replicated),
            .block_size = block_size,
            .tag = zml.Shape.toTag(proj_tag),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FP8Linear)) void {
        self.scale.deinit();
        self.weight.deinit();
    }

    pub fn forward(self: FP8Linear, x: zml.Tensor) zml.Tensor {
        return x.dot(dequantizeWeights(self.weight, self.scale, self.block_size), self.tag);
    }
};

// TODO: Use union(enum) depending if it's either a SCA or HCA layer.
const Compressor = struct {
    norm: RmsNorm,
    wgate: LinearF32,
    wkv: LinearF32,
    ape: zml.Tensor,
    ratio: i64,
    rotate: bool,
    overlap: bool,
    rope_opts: RopeOpts,
    head_dim: u32,
    rope_head_dim: i64,
    nope_head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize, rotate: bool, rope_opts: RopeOpts) ?Compressor {
        if (layer_idx >= config.compress_ratios.len or config.compress_ratios[layer_idx] == 0) return null;

        const ratio = config.compress_ratios[layer_idx];
        const compressor_store = store.withPrefix("compressor");

        return .{
            .norm = .init(compressor_store.createTensor("norm.weight", .{.hd}, .replicated), config.rms_norm_eps, .hd),
            .wgate = .init(compressor_store.createTensor("wgate.weight", .{.hc, .d}, .replicated), null, .d),
            .wkv = .init(compressor_store.createTensor("wkv.weight", .{.hc, .d}, .replicated), null, .d),
            .ape = compressor_store.createTensor("ape", .{.r, .hc}, .replicated),
            .ratio = ratio,
            .rotate = rotate,
            .overlap = (ratio == 4),
            .rope_opts = rope_opts,
            .head_dim = config.head_dim,
            .rope_head_dim = config.qk_rope_head_dim,
            .nope_head_dim = config.head_dim - config.qk_rope_head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Compressor)) void {
        RmsNorm.unloadBuffers(self.norm);
        LinearF32.unloadBuffers(self.wgate);
        LinearF32.unloadBuffers(self.wkv);
        self.ape.deinit();
    }

    pub fn overlap_transform(self: Compressor, x: zml.Tensor, v: f32) zml.Tensor {
        // shape(x) = [b,s,r,2d]
        const right_half = x.slice(&.{
            .{},
            .{},
            .{},
            .{ .start = self.head_dim },
        });

        const left_half = x.slice(&.{
            .{},
            .{ .end = x.dim(.seq) - 1},
            .{},
            .{ .end = self.head_dim },
        });

        // TODO: rewrite
        const pad_shape: zml.Shape = .init(.{.batch = x.dim(.batch), .seq = 1, .r = self.ratio, .hd = self.head_dim}, x.dtype());
        const pad = zml.Tensor.zeroes(pad_shape).addConstant(v);
        const left_padded = zml.Tensor.concatenate(&.{pad, left_half}, 1);

        return zml.Tensor.concatenate(&.{left_padded, right_half}, 2);
    }

    fn forwardPrefill(self: Compressor, x: zml.Tensor) struct { ?zml.Tensor, ?zml.Tensor } {
        const seqlen = x.dim(.seq);

        // Nothing to compress my friend
        if (seqlen < self.ratio) return .{ null, null };

        const x32 = x.convert(.f32);
        var kv = self.wkv.forward(x32);
        var score = self.wgate.forward(x32);

        const remainder = @mod(seqlen, self.ratio);
        const cutoff = seqlen - remainder;

        if (remainder > 0) {
            kv = kv.split(.seq, &.{cutoff, remainder})[0];
            score = score.split(.seq, &.{cutoff, remainder})[0];
        }

        kv = kv.splitAxis(.seq, .{ .seq = .auto, .r = self.ratio });
        score = score.splitAxis(.seq, .{ .seq = .auto, .r = self.ratio });
        score = score.add(self.ape.broad(score.shape()));

        if (self.overlap) {
            kv = self.overlap_transform(kv, 0);
            score = self.overlap_transform(score, -std.math.inf(f32));
        }

        const pos_idx = zml.Tensor.arange(.{ .end = cutoff, .step = self.ratio}, .i64).withTags(.{.seq});
        return .{ kv.mul(score.broad(kv.shape()).softmax(.r)).sum(.r).squeeze(.r), pos_idx };
    }

    fn forwardDecode(self: Compressor, x: zml.Tensor) struct { ?zml.Tensor, ?zml.Tensor } {
        const x32 = x.convert(.f32);
        const kv = self.wkv.forward(x32);
        _ = kv; // autofix
        var score = self.wgate.forward(x32);
        score = score.add(self.ape.broad(score.shape()));

        return .{ null, null };
    }

    pub fn forward(self: Compressor, x: zml.Tensor, start_pos: i64) ?zml.Tensor {
        // shape(x) = [batch,seq,r]
        const maybe_kv, const maybe_pos_idx = if (start_pos == 0) self.forwardPrefill(x) else self.forwardDecode(x);
        if (maybe_kv == null) return null;

        var kv = maybe_kv.?.convert(x.dtype());

        kv = self.norm.forward(kv);
        kv = apply_rope(kv, maybe_pos_idx.?, self.nope_head_dim, self.rope_head_dim, self.rope_opts);

        if (self.rotate) {
            kv = hadamard_rotation(kv);
            // TODO: quantize to FP4
        }

        // TODO: update cache

        return kv;
    }
};

fn find_correction_dim(dim: u32, max_seq_len: u32, num_rotations: f32, base: f32) f32 {
    const dim_f32: f32 = @floatFromInt(dim);
    const max_seq_len_f32: f32 = @floatFromInt(max_seq_len);

    return dim_f32 * @log10(max_seq_len_f32 / (num_rotations * 2 * std.math.pi)) / (2.0 * @log10(base));
}

fn find_correction_range(dim: u32, opts: RopeOpts) struct { f32, f32 } {
     const low = std.math.floor(find_correction_dim(dim, opts.original_max_position_embeddings, opts.beta_fast, opts.rope_theta));
     const high = std.math.ceil(find_correction_dim(dim, opts.original_max_position_embeddings, opts.beta_slow, opts.rope_theta));
     return .{ @max(low, 0), @min(high, @as(f32, @floatFromInt(dim-1))) };
}

fn linear_ramp_factor(min: f32, max: f32, dim: u32) zml.Tensor {
    const new_max = if (min == max) max + 0.001 else max;

    const linear_func = zml.Tensor.arange(.{ .end = dim }, .f32).addConstant(-min).divByConst(new_max - min);
    const shape = linear_func.shape();

    return linear_func.clamp(zml.Tensor.zeroes(shape), ones(shape));
}

fn precompute_yarn(dim: u32, opts: RopeOpts) zml.Tensor {
    var freqs = zml.Tensor.scalar(opts.rope_theta, .f32);
    freqs = freqs.pow(zml.Tensor.arange(.{ .end = dim, .step = 2 }, .f32).divByConst(dim));
    freqs = zml.Tensor.scalar(1.0, .f32).div(freqs);

    if (opts.original_max_position_embeddings > 0) {
        const low, const high = find_correction_range(dim, opts);
        const ramp_factor = linear_ramp_factor(low, high, dim/2);
        const smooth = zml.Tensor.scalar(1.0, .f32).sub(ramp_factor);

        freqs = freqs.divByConst(opts.factor).mul(ramp_factor).add(freqs.mul(smooth));
    }

    return freqs;
}

fn apply_yarn(x: zml.Tensor, pos_idx: zml.Tensor, opts: RopeOpts, reverse: bool) zml.Tensor {
    // TODO: add assert on .s and .hd
    const freqs = precompute_yarn(@intCast(x.dim(.hd)), opts).withTags(.{.hd});

    const inv_freq = zml.Tensor.outer(pos_idx.convert(.f32), freqs);

    const x_real, const x_imag = zml.nn.splitRealImg(x, .interleaved);
    const cos = inv_freq.cos().convert(x.dtype()).broad(x_real.shape());
    const sin = inv_freq.sin().convert(x.dtype()).broad(x_real.shape());

    const y_real = x_real.mul(cos).sub(x_imag.mul(sin));
    const y_imag = x_real.mul(sin).add(x_imag.mul(cos));

    // TODO: make it branchless
    if (reverse) {
        const y_imag_conj = y_imag.scale(-1);
        return zml.nn.mergeRealImg(y_real, y_imag_conj, .interleaved);
    }

    return zml.nn.mergeRealImg(y_real, y_imag, .interleaved);
}

fn apply_reverse_rope(x: zml.Tensor, pos_idx: zml.Tensor, nope_dim: i64, rope_dim: i64, opts: RopeOpts) zml.Tensor {
    const split = x.split(-1, &.{nope_dim, rope_dim});
    const x_rope = apply_yarn(split[1], pos_idx, opts, true);
    return zml.Tensor.concatenate(&.{split[0], x_rope}, -1);
}

fn apply_rope(x: zml.Tensor, pos_idx: zml.Tensor, nope_dim: i64, rope_dim: i64, opts: RopeOpts) zml.Tensor {
    const split = x.split(-1, &.{nope_dim, rope_dim});
    const x_rope = apply_yarn(split[1], pos_idx, opts, false);
    return zml.Tensor.concatenate(&.{split[0], x_rope}, -1);
}

// TODO: impl FWHT <https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform>
fn hadamard_rotation(x: zml.Tensor) zml.Tensor {
    const n = x.dim(-1);
    stdx.debug.assert(@mod(n, 2) == 0, "expect last dimension to be a power of 2, got: {}", .{ n });

    var H = zml.Tensor.scalar(1, x.dtype()).reshape(.{ 1, 1 });

    while (H.dim(0) < n) {
        const H_plus = zml.Tensor.concatenate(&.{H, H}, 1);
        const H_moins = zml.Tensor.concatenate(&.{H, H.scale(-1)}, 1);
        H = zml.Tensor.concatenate(&.{H_plus, H_moins}, 0);
    }

    const scale = zml.Tensor.scalar(n, x.dtype()).rsqrt();
    return x.dot(H.withTags(.{ .a, .hd }), .hd).mul(scale);
}

pub const KVCache = struct {
    kv: zml.Tensor, 

    pub fn init(mdl: Model, config: Config, batch_size: u32, seqlen: u32) KVCache {
        const kv_shape: zml.Shape = .init(.{
            .layer = config.num_hidden_layers,
            .batch = batch_size,
            .q = seqlen,
            .kv = config.num_attention_heads,
        }, mdl.embeds.weight.dtype());

        return .{ .kv = .fromShape(kv_shape) };
    }

    pub fn initBuffers(self: KVCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KVCache) {
        return .{ .kv = try zml.Buffer.uninitialized(io, platform, self.kv.shape(), sharding, .{}) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KVCache)) void {
        self.kv.deinit();
    }

    pub fn get(self: KVCache, cache_index: zml.Tensor) zml.Tensor {
        return self.kv.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KVCache, new_kv: zml.Tensor, token_position: zml.Tensor, cache_index: zml.Tensor) KVCache {
        const kv_shape = self.kv.shape().drop(.layer);
        const layer = cache_index.broad(token_position.shape());
        return .{
            .kv = self.kv.scatterSlices(.{ .layer = layer, .kv = token_position }, new_kv.transpose(kv_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.kv),
        };
    }
};

const Indexer = struct {
    proj: zml.nn.Linear,
    wq_b: FP8Linear,
    compressor: Compressor,
    index_head_dim: u32,
    index_n_heads: u32,
    index_topk: u32,
    rope_opts: RopeOpts,
    rope_head_dim: u32,
    nope_head_dim: u32,
    ratio: u32,
    softmax_scale: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize, rope_opts: RopeOpts) ?Indexer {
        if (layer_idx >= config.compress_ratios.len or config.compress_ratios[layer_idx] != 4) return null;

        const compressor: ?Compressor = .init(store, config, layer_idx, true, rope_opts);
        stdx.debug.assert(compressor != null, "expected non null compressor from indexer", .{});

        const block_size = config.quantization_config.weight_block_size[0];

        const indexer_store = store.withPrefix("indexer");

        return .{
            .proj = .init(indexer_store.createTensor("weights_proj.weight", .{.sq, .d}, .replicated), null, .d),
            .wq_b = .init(indexer_store.withPrefix("wq_b"), .{ .hd, .d }, block_size, .d),
            .compressor = compressor.?,
            .index_head_dim = config.index_head_dim,
            .index_n_heads = config.index_n_heads,
            .index_topk = config.index_topk,
            .rope_opts = rope_opts,
            .rope_head_dim = config.qk_rope_head_dim,
            .nope_head_dim = config.index_head_dim - config.qk_rope_head_dim, //< Not sure about `index_head_dim`
            .ratio = 4,
            .softmax_scale = std.math.pow(f32,@floatFromInt(config.index_head_dim), -0.5),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Indexer)) void {
        self.proj.weight.deinit();
        FP8Linear.unloadBuffers(&self.wq_b);
        Compressor.unloadBuffers(&self.compressor);
    }

    pub fn forward(self: Indexer, x: zml.Tensor, qr: zml.Tensor, start_pos: i64) zml.Tensor {
        const pos_idx = zml.Tensor.arange(.{ .start = start_pos, .end = start_pos + x.dim(.seq) }, .f32).withTags(.{ .seq });

        var q = self.wq_b.forward(qr);
        q = q.splitAxis(-1, .{ .h = self.index_n_heads, .hd = .auto });
        q = apply_rope(q, pos_idx, self.nope_head_dim, self.rope_head_dim, self.rope_opts);
        q = hadamard_rotation(q);

        // TODO: FP4 quant q
        // NOTE: executed to update the kv cache
        // _ = self.compressor.forward(x, start_pos);

        var weights = self.proj.forward(x);
        weights = weights.mul(zml.Tensor.scalar(@as(f32, @floatFromInt(self.index_n_heads)), weights.dtype()).broad(weights.shape()).rsqrt().scale(self.softmax_scale));
        //
        // const kv = qr; //< TODO: get it from KV cache
        //
        // var index_score = q.dot(kv, .d);
        // index_score = index_score.relu().mul(weights).sum(2).squeeze(2);
        //
        // const offset = 0; //< TBD
        // const end_pos: u32 = @intCast(start_pos + x.dim(.seq));
        // const k =  @min(self.index_topk, @mod(end_pos, self.ratio));
        //
        const seq = x.dim(.seq);

        const topk_idxs = blk: {
            if (start_pos == 0) {
                const a = zml.Tensor.iota(.init(.{ seq, @divFloor(seq, self.ratio) }, .i64), 1).convert(.i64);
                const b = zml.Tensor.arange(.{ .start = 1, .end = seq + 1 }, .i64).reshape(.{ seq, 1}).divByConst(self.ratio).broad(a.shape()).convert(.i64);
                const mask = a.cmp(.GT,  b);
                _ = mask; // autofix
                // _ = mask; // autofix

                // index_score.add(zml.Tensor.select(mask, zml.Tensor.scalar(-std.math.inf(f32), .f32).broad(mask), zeroes(index_score.shape())));

                // var topk_idxs = index_score.topK(-1, k, .{}).indices;
                //
                // const mask_ = topk_idxs.cmp(.GT, b);
                // topk_idxs = zml.Tensor.select(mask_, zml.Tensor.scalar(-1, topk_idxs.dtype()).broad(topk_idxs.shape()), topk_idxs.addConstant(offset));
                // break :blk topk_idxs;
            } else {
                // var topk_idxs = index_score.topK(-1, k, .{}).indices;
                // topk_idxs = topk_idxs.addConstant(offset);
                // break :blk topk_idxs;
            }

            break :blk x;
        };

        return topk_idxs;
    }
};

const Attention = struct {
    attn_sink: zml.Tensor,
    kv_norm: RmsNorm,
    q_norm: RmsNorm,
    wq_a: FP8Linear,
    wq_b: FP8Linear,
    wkv: FP8Linear,
    wo_a: FP8Linear,
    wo_b: FP8Linear,
    compressor: ?Compressor,
    indexer: ?Indexer,
    eps: f32,
    rope_opts: RopeOpts,
    window_size: u32,
    local_heads: u32,
    head_dim: u32,
    softmax_scale: f32,
    rope_head_dim: i64,
    nope_head_dim: i64,
    o_groups: u32,
    o_lora_rank: u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize) Attention {
        const block_size: usize = config.quantization_config.weight_block_size[0];

        stdx.debug.assert(layer_idx < config.compress_ratios.len, "expected layer indices ({}) to be lower than compress ratios ({})", .{layer_idx, config.compress_ratios.len});
        const compress_ratio= config.compress_ratios[layer_idx];

        const rope_opts: RopeOpts = .{
            .original_max_position_embeddings = if (compress_ratio == 0) 0 else config.rope_scaling.original_max_position_embeddings,
            .rope_theta = if (compress_ratio == 0)  config.rope_theta else config.compress_rope_theta,
            .beta_fast = config.rope_scaling.beta_fast,
            .beta_slow = config.rope_scaling.beta_slow,
            .factor = config.rope_scaling.factor,
        };

        return .{
            .attn_sink = store.createTensor("attn_sink", .{ .h }, .replicated),
            .kv_norm = .init(store.createTensor("kv_norm.weight", .{ .hd }, .replicated), config.rms_norm_eps, .hd),
            .q_norm = .init(store.createTensor("q_norm.weight", .{ .q }, .replicated), config.rms_norm_eps, .q),
            .wq_a = .init(store.withPrefix("wq_a"), .{ .q, .d }, block_size, .d),
            .wq_b = .init(store.withPrefix("wq_b"), .{ .hd, .q }, block_size, .q),
            .wkv = .init(store.withPrefix("wkv"), .{ .hd, .d }, block_size, .d),
            .wo_a = .init(store.withPrefix("wo_a"), .{ .hd, .d }, block_size, .d),
            .wo_b = .init(store.withPrefix("wo_b"), .{ .hd, .d }, block_size, .d),
            .compressor = .init(store, config, layer_idx, false, rope_opts),
            .indexer = .init(store, config, layer_idx, rope_opts),
            .eps = config.rms_norm_eps,
            .rope_opts = rope_opts,
            .window_size = config.sliding_window,
            .local_heads = config.num_attention_heads,
            .head_dim = config.head_dim,
            .softmax_scale = @sqrt(@floatFromInt(config.head_dim)),
            .rope_head_dim = config.qk_rope_head_dim,
            .nope_head_dim = config.head_dim - config.qk_rope_head_dim,
            .o_groups = config.o_groups,
            .o_lora_rank = config.o_lora_rank,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        self.attn_sink.deinit();
        RmsNorm.unloadBuffers(&self.kv_norm);
        RmsNorm.unloadBuffers(&self.q_norm);
        FP8Linear.unloadBuffers(&self.wq_a);
        FP8Linear.unloadBuffers(&self.wq_b);
        FP8Linear.unloadBuffers(&self.wkv);
        FP8Linear.unloadBuffers(&self.wo_a);
        FP8Linear.unloadBuffers(&self.wo_b);
    }

    // TODO: Add KV cache
    pub fn forward(
        self: Attention,
        x: zml.Tensor, 
        token_idx_offset: zml.Tensor,
        layer_idx: zml.Tensor,
        cache: KVCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KVCache } {
        _ = token_idx_offset; // autofix
        _ = layer_idx; // autofix
        _ = attention_metadata; // autofix
        _ = attention_parameters; // autofix
        // shape(x) = [batch, seq, d]
        // const start_pos = token_idx_offset.choose1d(0, 0);
        const start_pos: i64 = 0;
        const pos_idx = zml.Tensor.arange(.{ .start = start_pos, .end = start_pos + x.dim(.seq) }, .i64).withTags(.{ .seq });

        // shape(q) = [batch, q, h=64, hd=512]
        var q = self.q_norm.forward(self.wq_a.forward(x));
        q = self.wq_b.forward(q).splitAxis(.hd, .{ .h = self.local_heads, .hd = self.head_dim });
        q = q.mul(q.powByConst(2).mean(.hd).addConstant(self.eps)).rsqrt();
        q = apply_rope(q, pos_idx, self.nope_head_dim, self.rope_head_dim, self.rope_opts);
        q = q.rename(.{ .seq = .q });

        // TODO: FP8 quantize non-rope kv dims and let the rope-dim in bf16?
        // shape(kv) = [batch, k, hd=512]
        var kv = self.kv_norm.forward(self.wkv.forward(x));
        kv = apply_rope(kv, pos_idx, self.nope_head_dim, self.rope_head_dim, self.rope_opts);
        kv = kv.rename(.{ .seq = .k });

        // const new_kv_cache = kv_cache.update(kv, tokens_position_offset, cache_index);
        // kv = new_kv_cache.keys(cache_index);
        const cache_ = cache;

        const topk = topk_window(self.window_size, x.dim(.batch), x.dim(.seq), start_pos).withTags(.{.batch, .seq, .topk});

        var attn = sparse_attn(q, kv, self.attn_sink, topk, self.softmax_scale);
        attn = apply_reverse_rope(attn.rename(.{ .q = .seq }), pos_idx, self.nope_head_dim, self.rope_head_dim, self.rope_opts);
        attn = attn.reshape(.{ .batch = attn.dim(0), .seq = attn.dim(1), .g = self.o_groups, .d = self.head_dim * self.o_groups });

        var v = self.wo_a;
        v.weight = v.weight.splitAxis(.hd, .{ .g = self.o_groups, .r = self.o_lora_rank });

        var o = v.forward(attn);
        // TODO: figure out why this reshape is needed
        o = o.reshape(.{ .batch = o.dim(.batch), .seq = o.dim(.seq), .g = o.dim(.g), .r = o.dim(.r)});
        o = o.merge(.{ .d = .{ .g, .r}});

        return .{ self.wo_b.forward(o), cache_ };
    }
};

fn ones(s: zml.Shape) zml.Tensor {
    return zml.Tensor.constant(s.dtype().one()).broad(s);
}

fn topk_window(window_size: i64, batch_size: i64, seqlen: i64, start_pos: i64) zml.Tensor {
    const matrix = blk: {
        if (start_pos == 0) {
            const base = zml.Tensor.arange(.{ .end = seqlen }, .i64).reshape(.{seqlen, 1});

            // NOTE: equivalent to (base - window_size +1).clamp(0)
            const base_clamped = zml.Tensor.select(
            base.cmp(.LT, zml.Tensor.scalar(window_size + 1, .i64)),
            zml.Tensor.zeroes(base.shape()),
            base,
            );

            const cols = @min(seqlen, window_size);
            const row_indices = zml.Tensor.arange(.{ .end = cols }, .i64).reshape(.{1, cols});

            const matrix_shape: zml.Shape = .init(.{seqlen, cols}, .i64);
            var matrix = base_clamped.broad(matrix_shape).add(row_indices.broad(matrix_shape));
            matrix = zml.Tensor.select(
                matrix.cmp(.GT, base.broad(matrix_shape)),
                zml.Tensor.scalar(-1, .i64).broad(matrix_shape),
                matrix
            );

            break :blk matrix;
        } else if (start_pos >= window_size - 1) {
            const cutoff = @mod(start_pos, window_size);
            break :blk zml.Tensor.concatenate(&.{
                zml.Tensor.arange(.{ .start = cutoff + 1, .end = window_size }, .i64),
                zml.Tensor.arange(.{ .end = cutoff + 1}, .i64),
            }, 0);
        } else {
            const base = zml.Tensor.arange(.{ .end = start_pos + 1 }, .i64);
            const pad = zml.Tensor.scalar(-1, .i64).repeat(&.{@intCast(window_size - start_pos - 1)});
            break :blk zml.Tensor.concatenate(&.{ base, pad }, 0);
        }
    };

    return matrix.insertAxes(0, .{ .batch }).reshape(.{ batch_size, matrix.dim(0), matrix.dim(1) });
}

// From: <https://github.com/tile-ai/tilelang/blob/4e873220313c7e4dbfa0538582bc32ac5c81b1eb/examples/deepseek_v4/sparse_attn_fwd_sm90.py#L162>
pub fn sparse_attn(q: zml.Tensor, kv: zml.Tensor, attn_sink: zml.Tensor, topk: zml.Tensor, scale: ?f32) zml.Tensor {
    // q = [batch, q, h, hd], kv = [batch, k, hd], topk = [batch, seq, topk]
    const selected_kv = kv.gather(.{ .k = topk }, .{}).rename(.{.seq = .q, .topk = .k}).convert(.f32);

    const dims = zml.nn.collectDims(.{ .h, .q, .k, .hd }, &.{ q, kv }, .strict) catch {
        stdx.debug.panic("Inputs have incompatible shapes (q: {f}, kv: {f}, attn_mask: ).", .{q, kv});
    };

    const sqrt_head_dim = if (scale) | m | m else 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
    const head_scaling = zml.Tensor.scalar(sqrt_head_dim, selected_kv.dtype());

    const q_32 = q.convert(.f32);
    const scores = q_32.dot(selected_kv, .hd).mul(head_scaling);

    const sink_shape = q.shape().set(.hd, 1);
    const sink = attn_sink.insertAxes(0, .{ .batch, .q }).insertAxes(.last, .{ .hd }).broad(sink_shape);
    const scores_sink = zml.Tensor.concatenate(&.{scores, sink.convert(scores.dtype())}, -1);

    const attn_weights = scores_sink.convert(.f32).softmax(.k).convert(q_32.dtype());
    const attn_weights_non_sink = attn_weights.slice(&.{
        .{},
        .{},
        .{},
        .{ .end = topk.dim(.topk) },
    });
    const attn = attn_weights_non_sink.dot(selected_kv, .k);
    return attn.convert(.bf16);
}

const MoE = struct {
    gate: Gate,
    experts: []Expert,
    shared_experts: Expert,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, i: usize) !MoE {
        const experts = try allocator.alloc(Expert, 1);
        // const experts = try allocator.alloc(Expert, config.n_routed_experts);

        for(experts, 0..) |*expert, expert_idx| {
            const experts_store = store.withPrefix("experts").withLayer(expert_idx);
            expert.* = .init(experts_store, config);
        }

        return .{
            .gate = .init(store.withPrefix("gate"), config, i),
            .experts = experts,
            .shared_experts = .init(store.withPrefix("shared_experts"), config),
        };
    }

    pub fn deinit(self: MoE, allocator: std.mem.Allocator) void {
        allocator.free(self.experts);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MoE), allocator: std.mem.Allocator) void {
        Gate.unloadBuffers(&self.gate);
        for(self.experts) |*expert| {
            Expert.unloadBuffers(expert);
        }
        allocator.free(self.experts);
        Expert.unloadBuffers(&self.shared_experts);
    }

    pub fn forward(self: MoE, x: zml.Tensor, input_ids: zml.Tensor) zml.Tensor {
        const weight, const indices = self.gate.forward(x, input_ids);
        _ = weight; // autofix

        // TODO: use zml.moe.forwardMoe

        const counts = bincount(indices.flatten(), @intCast(self.experts.len));
        _ = counts; // autofix

        var y = zml.Tensor.constant(x.dtype().zero()).broad(x.shape());
        for (self.experts, 0..) |expert, i| {
            _ = expert; // autofix
            _ = i; // autofix
        }

        y = y.add(self.shared_experts.forward(x, null));

        return y;
    }
};

// TODO: implement either bincount with scatter_add/scatter_slices (but IR can be shitty) OR one_hot
fn bincount(x: zml.Tensor, num_bins: ?i64) zml.Tensor {
    _ = num_bins; // autofix
    stdx.debug.assert(x.rank() == 1, "expect input tensor to be rank == 1, while we got rank == {}", .{x.rank()});
    return x;
    // const bins_size = num_bins orelse x.dim(-1) + 1;
    // const bins_shape: zml.Shape = .init(.{ bins_size }, .u64);
    //
    // const bins_idx = zml.Tensor.iota(bins_shape, 0);
    // const mask = x.unsqueeze(-1).cmp(bins_idx.unsqueeze(0));
    //
    // // Convert booleans to u64 (1s and 0s)
    // const counts_matrix = mask.cast(.u64);
    //
    // // Sum along the token dimension (axis 0) to get total counts per expert
    // return counts_matrix.sum(.{ 0 });
    //
    // const ones = zml.Tensor.constant(x.dtype().one()).broad(bins_shape);
    // const zeros = zml.Tensor.constant(x.dtype().zero()).broad(bins_shape).withTags(.{.i});
    // return zeros.scatterSlices(.{.i = x}, ones, .{});
}

const Gate = struct {
    const Kind = union(enum) {
        bias: zml.Tensor,
        tid2eid: zml.Tensor,
    };

    k: u32,
    kind: Kind,
    proj: LinearF32,
    scaling_factor: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, i: usize) Gate {
        // TODO: use `maybeTensor` instead of the id?
        const op = blk: {
            if (i < config.num_hash_layers) {
                break :blk Kind{ .tid2eid = store.createTensor("tid2eid", .{.tid, .eid}, .replicated) };
            } else {
                break :blk Kind { .bias = store.createTensor("bias", .{.expert}, .replicated) };
            }
        };

        return .{
            .k = config.num_experts_per_tok,
            .kind = op,
            .proj = .init(store.createTensor("weight", .{.expert, .d}, .replicated), null, .d),
            .scaling_factor = config.routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gate)) void {
        self.proj.weight.deinit();
        switch(self.kind) {
            .bias, .tid2eid => |*t| t.deinit(),
        }
    }

    // TODO: add to `nn.zig`
    fn softplus(x: zml.Tensor, threshold: f32) zml.Tensor {
        const s = x.exp().addConstant(1).log();
        const mask = x.cmp(.GT, zml.Tensor.scalar(threshold, .f32).broad(x.shape()));
        return zml.Tensor.select(mask, x, s);
    }

    pub fn forward(self: Gate, x: zml.Tensor, input_ids: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // TODO: support `softmax`, `sigmoid`?
        const scores = softplus(self.proj.forward(x.convert(.f32)), 20.0).sqrt();

        const indices = switch(self.kind) {
            // TODO: remove `renameTag`?
            .bias => |bias| scores.add(bias.broad(scores.shape())).topK(-1, self.k, .{}).indices.renameTag(.expert, .eid).convert(.i64),
            .tid2eid => |tid2eid| tid2eid.gather(.{.tid = input_ids}, .{}),
        };

        var weights = scores.gather(.{ .expert = indices }, .{});
        weights = weights.div(weights.sum(.eid));
        weights = weights.scale(self.scaling_factor);

        return .{weights, indices.convert(.i32)};
    }
};

const Expert = struct {
    w1: FP8Linear,
    w2: FP8Linear,
    w3: FP8Linear,
    activation_threshold: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Expert {
        const block_size= config.quantization_config.weight_block_size[0];

        return .{
            .w1 = .init(store.withPrefix("w1"), .{ .dint, .d }, block_size, .d),
            .w2 = .init(store.withPrefix("w2"), .{ .d, .dint }, block_size, .dint),
            .w3 = .init(store.withPrefix("w3"), .{ .dint, .d }, block_size, .d),
            .activation_threshold = config.swiglu_limit,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Expert)) void {
        FP8Linear.unloadBuffers(&self.w1);
        FP8Linear.unloadBuffers(&self.w2);
        FP8Linear.unloadBuffers(&self.w3);
    }

    pub fn forward(self: Expert, x: zml.Tensor, weights: ?zml.Tensor) zml.Tensor {
        // FIX: close_fraction ~= 40%
        const threshold = zml.Tensor.scalar(self.activation_threshold, .f32);
        const m_threshold = zml.Tensor.scalar(-self.activation_threshold, .f32);

        const up = self.w3.forward(x).convert(.f32).clamp(m_threshold, threshold);
        var gate = self.w1.forward(x).convert(.f32);
        gate = zml.Tensor.select(gate.cmp(.GT, threshold), threshold, gate);

        var x_ = gate.silu().mul(up);
        if (weights) | w | {
            x_ = x_.mul(w);
        }
        return self.w2.forward(x_.convert(x.dtype()));
    }
};

const Layer = struct {
    const SinkhornOpts = struct {
        scale: zml.Tensor,
        func: zml.nn.Linear,
        base: zml.Tensor,
    };

    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    hc_attn: SinkhornOpts,
    hc_ffn: SinkhornOpts,
    attn: Attention,
    ffn: MoE,
    norm_eps: f32,
    hc_eps: f32,
    hc_mult: u32,
    hc_sinkhorn_iters: u32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, i: usize) !Layer {
        return .{
            .attn_norm = .init(store.createTensor("attn_norm.weight", .{ .d }, .replicated), config.rms_norm_eps, .d),
            .ffn_norm = .init(store.createTensor("ffn_norm.weight", .{ .d }, .replicated), config.rms_norm_eps, .d),
            .attn = .init(store.withPrefix("attn"), config, i),
            .ffn = try .init(allocator, store.withPrefix("ffn"), config, i),
            .norm_eps = config.rms_norm_eps, //< TODO: CHECK
            .hc_eps = config.hc_eps,
            .hc_mult = config.hc_mult,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .hc_attn = .{
                .base = store.createTensor("hc_attn_base", .{.b}, .replicated),
                .func = .init(store.createTensor("hc_attn_fn", .{.b, .d}, .replicated), null, .d),
                .scale = store.createTensor("hc_attn_scale", .{.scale}, .replicated),
            },
            .hc_ffn = .{
                .base = store.createTensor("hc_ffn_base", .{.b}, .replicated),
                .func = .init(store.createTensor("hc_ffn_fn", .{.b, .d}, .replicated), null, .d),
                .scale = store.createTensor("hc_ffn_scale", .{.b}, .replicated),
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Layer), allocator: std.mem.Allocator) void {
        self.attn_norm.weight.deinit();
        self.ffn_norm.weight.deinit();
        self.hc_attn.base.deinit();
        self.hc_attn.func.weight.deinit();
        self.hc_attn.scale.deinit();
        self.hc_ffn.base.deinit();
        self.hc_ffn.func.weight.deinit();
        self.hc_ffn.scale.deinit();
        Attention.unloadBuffers(&self.attn);
        MoE.unloadBuffers(&self.ffn, allocator);
    }

    // TODO: maybe implement mHC-lite: <https://arxiv.org/html/2601.05732v1>
    fn hc_split_sinkhorn(mixes: zml.Tensor, n: usize, mult: u32, eps: f32, opts: SinkhornOpts) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        // mixes = [batch, seq, hc, d]
        const pre_mix = mixes.slice1d(.b, .{ .end = mult });
        const post_mix = mixes.slice1d(.b, .{ .start = mult, .end = 2 * mult });
        const comb_mix = mixes.slice1d(.b, .{ .start = 2 * mult });

        const base_pre = opts.base.slice1d(0, .{ .end = mult });
        const base_post = opts.base.slice1d(0, .{ .start = mult, .end = 2 * mult });
        const base_comb = opts.base.slice1d(0, .{ .start = 2 * mult }).splitAxis(-1, .{ mult, .auto});

        const pre = pre_mix.mul(opts.scale.choose1d(0, 0)).add(base_pre.broad(pre_mix.shape())).sigmoid().addConstant(eps);
        const post = post_mix.mul(opts.scale.choose1d(0, 1)).add(base_post.broad(post_mix.shape())).sigmoid().scale(2);

        var comb = comb_mix.splitAxis(-1, .{ mult, .auto });
        comb = comb.mul(opts.scale.choose1d(0, 2)).add(base_comb.insertAxes(0, .{ .batch, .seq }).broad(comb.shape()));
        comb = comb.softmax(-1).addConstant(eps);
        comb = comb.div(comb.sum(-2).addConstant(eps));

        for(1..n) |_| {
            comb = comb.div(comb.sum(-1).addConstant(eps));
            comb = comb.div(comb.sum(-2).addConstant(eps));
        }

        return .{ pre, post, comb };
    }

    fn hc_pre(self: Layer, x: zml.Tensor, opts: SinkhornOpts) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        // shape(x) = [batch, seq, hc, d]
        const x_ = x.merge(.{ .d = .{ .hc, .d }}).convert(.f32);

        const mixes = blk: {
            const rsqrt = x_.powByConst(2).mean(.d).addConstant(self.norm_eps).rsqrt();
            const m = opts.func.forward(x_);
            break :blk m.mul(rsqrt.broad(m.shape()));
        };

        const pre, const post, const comb = hc_split_sinkhorn(mixes, self.hc_sinkhorn_iters, self.hc_mult, self.hc_eps, opts);

        const y = x.convert(.f32).mul(pre.appendAxes(.{.d}).broad(x.shape())).sum(.hc).squeeze(.hc);
        return .{ y.convert(x.dtype()), post, comb };
    }

    fn hc_post(x: zml.Tensor, residual: zml.Tensor, post: zml.Tensor, comb: zml.Tensor) zml.Tensor {
        // x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        const p = post.appendAxes(.{ .hd }).rename(.{ .b = .hc });
        const x_ = x.insertAxes(.hd, .{ .hc }).convert(.f32);

        const s: zml.Shape = x_.shape().set(.hc, p.dim(.hc));
        const m = x_.broad(s).mul(p.broad(s));

        const r_32 = residual.insertAxes(.hc, .{ .hc2 }).convert(.f32);
        const c = comb.appendAxes(.{ .d });//withTags(.{ .batch, .seq, .hc, .d });
        const s2 = c.shape().set(.d, r_32.dim(.d));
        const m2 = r_32.broad(s2).mul(c.broad(s2)).sum(2).squeeze(2).withTags(.{ .batch, .seq, .hc, .dd });
        
        const f = m.add(m2).rename(.{ .hd = .d });
        return f.convert(x.dtype());
    }

    pub fn forward(
        self: Layer,
        x: zml.Tensor,
        tokens: zml.Tensor,
        tokens_idx_offset: zml.Tensor,
        layer_idx: zml.Tensor,
        cache: KVCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KVCache } {
        // x: [batch, seq, hc, d]
        var x_, const post, const comb = self.hc_pre(x, self.hc_attn);
        x_ = self.attn_norm.forward(x_);
        x_, const cache_ = self.attn.forward(x_, tokens_idx_offset, layer_idx, cache, attention_metadata, attention_parameters);
        x_ = hc_post(x_, x, post, comb);

        var x_2, const post_2, const comb_2 = self.hc_pre(x_, self.hc_ffn);

        x_2 = self.ffn_norm.forward(x_2);
        x_2 = self.ffn.forward(x_2, tokens);
        x_2 = hc_post(x_2.rename(.{ .d = .hd}), x_, post_2, comb_2);

        return .{ x_2, cache_ };
    }
};

const LmHead = struct {
    norm: RmsNorm,
    voc_proj: LinearF32,
    hc_base: zml.Tensor,
    hc_fn: zml.nn.Linear,
    hc_scale: zml.Tensor,
    hc_eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) LmHead {
        return .{
            .norm = .init(store.createTensor("norm.weight", .{.d}, .replicated), config.rms_norm_eps, .d,),
            .voc_proj = .init(store.createTensor("head.weight", .{.voc, .d}, .replicated), null, .d),
            .hc_base = store.createTensor("hc_head_base", .{.hc}, .replicated),
            .hc_fn = .init(store.createTensor("hc_head_fn", .{.hc, .d}, .replicated), null, .d),
            .hc_scale = store.createTensor("hc_head_scale", .{.batch}, .replicated),
            .hc_eps = config.hc_eps,
        };
    }

    fn hc_head(self: LmHead, x: zml.Tensor) zml.Tensor {
        // shape(x) = [b,s,hc,d]
        const x_ = x.merge(.{ .d = .{ .hc, .d }}).convert(.f32);

        const rsqrt = blk: {
            const variance = x_.powByConst(2).mean(.d);
            break :blk zml.Tensor.rsqrt(variance.addConstant(self.norm.eps));
        };

        var mixes = self.hc_fn.forward(x_);
        mixes = mixes.mul(rsqrt.broad(mixes.shape()));

        const pre = blk: {
            // FIX: `broad` tags.
            const s = mixes.shape();
            break :blk mixes.mul(self.hc_scale.broad(s)).add(self.hc_base.broad(s)).sigmoid().addConstant(self.hc_eps);
        };

        return x.convert(.f32).mul(pre.broad(x.shape())).sum(.hc).squeeze(.hc).convert(x.dtype());
    }

    pub fn forward(self: LmHead, x: zml.Tensor) zml.Tensor {
        const x_ = self.voc_proj.forward(self.norm.forward(self.hc_head(x)).convert(.f32));
        return x_.slice1d(.seq, .{ .start = x_.dim(.seq) - 1 }).squeeze(.seq);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RmsNorm.unloadBuffers(&self.norm);
        self.voc_proj.weight.deinit();
        self.hc_base.deinit();
        self.hc_fn.weight.deinit();
        self.hc_scale.deinit();
    }
};

pub const Model = struct {
    embeds: zml.nn.TokenEmbedding,
    layers: []Layer,
    lm_head: LmHead, 
    hc_mult: u32,
    sampling_strategy: zml.nn.SamplingStrategy,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, generation: common.GenerationOptions) !Model {
        const layers = try allocator.alloc(Layer, config.num_hidden_layers);

        for (layers, 0..) |*layer, i| {
            const layer_store = store.withPrefix("layers").withLayer(i);
            layer.* = try .init(allocator, layer_store, config, i);
        }

        return .{
            .embeds = .{
                .weight = store.createTensor("embed.weight", .{ .voc, .d }, .replicated),
            },
            .layers = layers,
            .lm_head = .init(store, config),
            .hc_mult = config.hc_mult,
            .sampling_strategy = generation.sampling_strategy,
        };
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        for(self.layers) |*layer| {
            layer.*.ffn.deinit(allocator);
        }
        allocator.free(self.layers);
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: []const zml.Sharding,
    ) !zml.Bufferized(Model) {
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const took_ns: usize = @max(1, @as(usize, @intCast(took.toNanoseconds())));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{
                total_bytes,
                took,
                total_bytes * std.time.ns_per_s / took_ns,
            });
        }

        //progress.increaseEstimatedTotalItems(store.view().count());

        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .dma_chunks = 8,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .parallelism = 16,
            .total_bytes = &total_bytes,
            .shardings = shardings,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        self.embeds.weight.deinit();

        for (self.layers) |*layer| {
            Layer.unloadBuffers(layer, allocator);
        }
        allocator.free(self.layers);

        LmHead.unloadBuffers(&self.lm_head);
    }

    pub fn forward(
        self: Model,
        tokens: zml.Tensor,
        tokens_idx_offset: zml.Tensor,
        rng: zml.Tensor.Rng,
        cache: KVCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, zml.Tensor.Rng, KVCache } {
        stdx.debug.assert(tokens.shape().hasTags(.{ .batch, .seq, }), "expect input tokens to has tags (.batch, .seq) but got {f}", .{ tokens.shape() });

        var hidden = self.embeds.forward(tokens); //< [batch, seq, d]
        hidden = hidden.insertAxes(.d, .{ .hc }).repeat1d(.hc, @intCast(self.hc_mult)); //< [batch, seq, hc, d]

        var cache_ = cache;
        var layer_idx = zml.Tensor.scalar(@as(u32, 0), .u32);

        for(self.layers) |layer| {
            hidden, cache_ = layer.forward(hidden, tokens, tokens_idx_offset, layer_idx, cache_, attention_metadata, attention_parameters); //< [batch, seq, d]
            layer_idx = layer_idx.addConstant(1);
        }

        const logits = self.lm_head.forward(hidden); //< [batch, voc]

        const new_tokens, const new_rng = zml.nn.sampleTokens(logits, self.sampling_strategy, rng);
        return .{ new_tokens.convert(tokens.dtype()), new_rng, cache_ };
    }
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .inner = try .init(allocator, store, parsed_config.value, generation),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        return self.inner.loadBuffers(allocator, io, platform, store, progress, &shardings.all());
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self; // autofix
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        const opts = inference.CompilationParameters.init(@intCast(seqlen), self.parsed_config.value, self.inner, shardings, backend); 
        return try inference.CompiledModel.init(allocator, io, @constCast(platform), self.inner, @intCast(seqlen), progress, opts);
    }
};
