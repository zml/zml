const std = @import("std");
const stdx = zml.stdx;

const zml = @import("zml");
const common = @import("../common.zig");
const inference = @import("inference.zig");
const attention = @import("sparse_mla.zig");

const log = std.log.scoped(.deepseek_v4);

pub const QuantizationConfig = struct {
    weight_block_size: []usize,
};

pub const Config = struct {
    hidden_size: u32,
    max_position_embeddings: u32,
    expert_dtype: []const u8, //< find better type
    num_hidden_layers: u16,
    rms_norm_eps: f32,
    routed_scaling_factor: f32,
    num_hash_layers: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    swiglu_limit: f32,
    compress_ratios: []u32,
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

fn dequantizeScaledBlocks(weight: zml.Tensor, scale: zml.Tensor, block_size: usize, dtype: zml.DataType) zml.Tensor {
    // Reshape scale from [8, 32] -> [8, 1, 32, 1] to inject dummy block dimensions
    const scale_4d_shape = scale.shape().insert(1, .{1}).insert(.last, .{1});
    const scale_4d = scale.convert(dtype).reshape(scale_4d_shape);

    // Broadcast along the block dimensions (block size = 128)
    // [8, 1, 32, 1] -> [8, 128, 32, 128]
    const broad_shape = scale.shape().insert(1, .{block_size}).insert(.last, .{block_size});
    const scale_broad = scale_4d.broad(broad_shape);

    // Flatten the 4D expanded scale back to 2D matrix matching the weights: [1024, 4096]
    const target_shape = weight.shape();
    const scale_expanded = scale_broad.reshape(target_shape);

    // Element-wise multiplication completes the dequantization pipeline
    return weight.convert(dtype).mul(scale_expanded);
}

fn fastRoundScale(scale: zml.Tensor) zml.Tensor {
    // Matches kernel.py fast_round_scale(): 2 ** ceil(log2(scale)).
    // DeepSeek V4 uses MXFP/UE8M0 activation scales for FP8 GEMMs.
    const exponent = scale.log().divByConst(@log(2.0)).ceil();
    return ones(scale.shape()).scale(2.0).pow(exponent);
}

fn to_fp8(x: zml.Tensor, block_size: u32, round_scale: bool) struct { zml.Tensor, zml.Tensor } {
    // x = [seq, d=4096]
    // block = 128
    stdx.debug.assert(@mod(x.dim(-1), block_size) == 0, "last dimension {} must be divisible by block_size={}", .{ x.dim(-1), block_size });

    // x_blocked = [seq, 32, 128]
    const x_blocked = x.splitAxis(-1, .{ .n = .auto, .m = block_size }).convert(.f32);

    const min_value = zml.Tensor.scalar(1e-4, x_blocked.dtype());

    var max_block = x_blocked.abs().max(-1);
    max_block = zml.Tensor.select(max_block.cmp(.LT, min_value), min_value, max_block);

    const fp8_min = zml.Tensor.constant(zml.DataType.f8e4m3fn.minValue()).convert(x_blocked.dtype());
    const fp8_max = zml.Tensor.constant(zml.DataType.f8e4m3fn.maxValue()).convert(x_blocked.dtype());
    const fp8_max_inv = ones(fp8_max.shape()).div(fp8_max);

    var scale = max_block.mul(fp8_max_inv);
    if (round_scale) {
        scale = fastRoundScale(scale);
    }

    const x_fp8 = x_blocked.div(scale).clamp(fp8_min, fp8_max).reshape(x.shape());

    return .{ x_fp8.convert(.f8e4m3fn), scale.convert(.f8e8m0) };
}

fn fp4ActQuant(x: zml.Tensor, block_size: u32) zml.Tensor {
    stdx.debug.assert(@mod(x.dim(-1), block_size) == 0, "last dimension {} must be divisible by block_size={}", .{ x.dim(-1), block_size });

    const x_blocked = x.splitAxis(-1, .{ .n = .auto, .m = block_size }).convert(.f32);
    const min_value = zml.Tensor.scalar(6.0 * std.math.pow(f32, 2.0, -126), x_blocked.dtype());

    var max_block = x_blocked.abs().max(-1);
    max_block = zml.Tensor.select(max_block.cmp(.LT, min_value), min_value, max_block);

    const fp4_min = zml.Tensor.constant(zml.DataType.f4e2m1.minValue()).convert(x_blocked.dtype());
    const fp4_max = zml.Tensor.constant(zml.DataType.f4e2m1.maxValue()).convert(x_blocked.dtype());
    const fp4_max_inv = ones(fp4_max.shape()).div(fp4_max);

    const scale = fastRoundScale(max_block.mul(fp4_max_inv));
    const x_fp4 = x_blocked
        .div(scale.broad(x_blocked.shape()))
        .clamp(fp4_min, fp4_max)
        .convert(.f4e2m1)
        .convert(.f32);

    return x_fp4
        .mul(scale.broad(x_fp4.shape()))
        .reshape(x.shape().withDtype(.f32))
        .convert(x.dtype());
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

    fn dequantizeActivationF32(x_fp8: zml.Tensor, scale: zml.Tensor, block_size: usize, contract_tag: zml.Shape.Tag) zml.Tensor {
        var x = x_fp8.splitAxis(contract_tag, .{ .kb = .auto, .bk = block_size }).convert(.f32);
        x = x.mul(scale.convert(.f32).broad(x.shape()));
        return x.reshape(x_fp8.shape().withDtype(.f32));
    }

    pub fn forward(self: FP8Linear, x: zml.Tensor) zml.Tensor {
        const x_fp8, const x_scale = to_fp8(x, @intCast(self.block_size), true);
        const x_dequant = dequantizeActivationF32(x_fp8, x_scale, self.block_size, self.tag);
        const weight_dequant = dequantizeScaledBlocks(self.weight, self.scale, self.block_size, .f32);
        return x_dequant.dot(weight_dequant, self.tag).convert(.bf16);
    }
};

pub const Compressor = struct {
    norm: RmsNorm,
    wgate: LinearF32,
    wkv: LinearF32,
    ape: zml.Tensor,
    ratio: u32,
    overlap: bool,
    rope_opts: RopeOpts,
    head_dim: u32,
    rope_head_dim: u32,
    nope_head_dim: u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, head_dim: u32, compression_ratio: u32, rope_opts: RopeOpts) Compressor {
        return .{
            .norm = .init(store.createTensor("norm.weight", .{.d}, .replicated), config.rms_norm_eps, .d),
            .wgate = .init(store.createTensor("wgate.weight", .{ .hd, .d }, .replicated), null, .d),
            .wkv = .init(store.createTensor("wkv.weight", .{ .hd, .d }, .replicated), null, .d),
            .ape = store.createTensor("ape", .{ .r, .hd }, .replicated),
            .ratio = compression_ratio,
            .overlap = (compression_ratio == 4),
            .rope_opts = rope_opts,
            .head_dim = head_dim,
            .rope_head_dim = config.qk_rope_head_dim,
            .nope_head_dim = head_dim - config.qk_rope_head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Compressor)) void {
        RmsNorm.unloadBuffers(&self.norm);
        LinearF32.unloadBuffers(&self.wgate);
        LinearF32.unloadBuffers(&self.wkv);
        self.ape.deinit();
    }

    pub fn overlap_transform(self: Compressor, x: zml.Tensor, v: zml.Tensor) zml.Tensor {
        // shape(x) = [b,s,r,2d]
        const right_half = x.slice(&.{
            .{},
            .{},
            .{},
            .{ .start = self.head_dim },
        });

        const left_half = x.slice(&.{
            .{},
            .{ .end = x.dim(.seq) - 1 },
            .{},
            .{ .end = self.head_dim },
        });

        // TODO: rewrite
        const pad_shape: zml.Shape = .init(.{ .batch = x.dim(.batch), .seq = 1, .r = self.ratio, .hd = self.head_dim }, x.dtype());
        const pad = zml.Tensor.zeroes(pad_shape).add(v.broad(pad_shape));
        const left_padded = zml.Tensor.concatenate(&.{ pad, left_half }, 1);

        return zml.Tensor.concatenate(&.{ left_padded, right_half }, 2);
    }

    pub fn forwardPrefill(
        self: Compressor,
        x: zml.Tensor,
        seqlen: zml.Tensor, //< the actual seqlen since `x.dim(.seq)` is the size of the allocated buffer
        state: CompressorState,
        state_idx: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor, CompressorState } {
        // stdx.debug.assert(@mod(x.dim(.seq), self.ratio) == 0, ".seq should be divisible by {}", .{self.ratio});

        var kv = self.wkv.forward(x);
        var score = self.wgate.forward(x);

        const cutoff = seqlen.divByConst(self.ratio).scale(self.ratio);
        const remainder = seqlen.remainderConst(self.ratio);

        const ape = self.ape.insertAxes(0, .{.batch});

        var new_state = state;

        const score_inf = zml.Tensor.constant(score.dtype().minValue());
        const ratio_dim: i64 = @intCast(self.ratio);
        const state_seq = @max(ratio_dim, @divTrunc(x.dim(.seq) + ratio_dim - 1, ratio_dim) * ratio_dim);
        const state_pad = state_seq - x.dim(.seq);
        const kv_state = if (state_pad == 0) kv else kv.pad(0, .{ .seq = zml.Tensor.Pad{ .high = state_pad } });
        const score_state = if (state_pad == 0) score else score.pad(0, .{ .seq = zml.Tensor.Pad{ .high = state_pad } });

        if (self.overlap) {
            var cutoff_last_block = cutoff.subConstant(self.ratio);
            cutoff_last_block = zml.Tensor.select(
                cutoff_last_block.cmp(.LT, .zeroes(cutoff_last_block.shape())),
                .zeroes(cutoff_last_block.shape()),
                cutoff_last_block,
            );

            var kv_last_compressed = kv_state.dynamicSlice1d(kv_state.axis(.seq), .{ .start = cutoff_last_block, .len = self.ratio });
            var score_last_compressed = score_state.dynamicSlice1d(score_state.axis(.seq), .{ .start = cutoff_last_block, .len = self.ratio });
            score_last_compressed = score_last_compressed.add(ape.broad(score_last_compressed.shape()));

            var mask = cutoff.cmp(.GE, .scalar(self.ratio, cutoff.dtype()));
            mask = mask.insertAxes(0, .{.batch});
            mask = mask.insertAxes(.last, .{.hd});

            kv_last_compressed = zml.Tensor.select(mask.broad(kv_last_compressed.shape()), kv_last_compressed, .zeroes(kv_last_compressed.shape()));
            score_last_compressed = zml.Tensor.select(mask.broad(score_last_compressed.shape()), score_last_compressed, score_inf.broad(score_last_compressed.shape()));

            new_state = new_state.update(kv_last_compressed, score_last_compressed, zml.Tensor.arange(.{ .end = self.ratio }, .u32).withTags(.{.seq}), state_idx);
        }

        var valid_uncompressible = zml.Tensor.arange(.{ .end = self.ratio }, .u32).cmp(.LT, remainder);
        valid_uncompressible = valid_uncompressible.insertAxes(0, .{.batch}).insertAxes(.last, .{.hd});

        var kv_uncompressible = kv_state.dynamicSlice1d(kv_state.axis(.seq), .{ .start = cutoff, .len = self.ratio });
        var score_uncompressible = score_state.dynamicSlice1d(score_state.axis(.seq), .{ .start = cutoff, .len = self.ratio });
        score_uncompressible = score_uncompressible.add(ape.broad(score_uncompressible.shape()));

        kv_uncompressible = zml.Tensor.select(valid_uncompressible.broad(kv_uncompressible.shape()), kv_uncompressible, .zeroes(kv_uncompressible.shape()));
        score_uncompressible = zml.Tensor.select(valid_uncompressible.broad(score_uncompressible.shape()), score_uncompressible, score_inf.broad(score_uncompressible.shape()));

        const uncompressible_cache_ids = blk: {
            const state_offset = if (self.overlap) self.ratio else 0;
            break :blk zml.Tensor.arange(.{ .end = self.ratio }, .u32).addConstant(state_offset).withTags(.{.seq});
        };

        new_state = new_state.update(kv_uncompressible, score_uncompressible, uncompressible_cache_ids, state_idx);

        const end = @divTrunc(x.dim(.seq), self.ratio) * self.ratio;
        if (end == 0) {
            return .{
                zml.Tensor.zeroes(kv.shape().set(.seq, 0)),
                zml.Tensor.arange(.{ .end = 0, .step = self.ratio }, .i64).withTags(.{.seq}),
                new_state.reuseBuffer(state),
            };
        }

        kv = kv.slice1d(.seq, .{ .end = end });
        score = score.slice1d(.seq, .{ .end = end });

        kv = kv.splitAxis(.seq, .{ .seq = .auto, .r = self.ratio });
        score = score.splitAxis(.seq, .{ .seq = .auto, .r = self.ratio });
        score = score.add(self.ape.broad(score.shape()));

        if (self.overlap) {
            kv = self.overlap_transform(kv, zml.Tensor.scalar(0, kv.dtype()));
            score = self.overlap_transform(score, score_inf);
        }

        kv = kv.mul(score.broad(kv.shape()).softmax(.r)).sum(.r).squeeze(.r);

        const base = zml.Tensor.arange(.{ .end = kv.dim(.seq) }, .u32);
        const valid_mask = base.cmp(.LT, seqlen.divByConst(self.ratio)).insertAxes(0, .{.batch}).insertAxes(.last, .{.hd}).broad(kv.shape());

        const kv_compressed = zml.Tensor.select(valid_mask, kv, .zeroes(kv.shape()));
        const pos_idx = zml.Tensor.arange(.{ .end = end, .step = self.ratio }, .i64).withTags(.{.seq});

        return .{ kv_compressed, pos_idx, new_state.reuseBuffer(state) };
    }

    pub fn forwardDecode(
        self: Compressor,
        x: zml.Tensor,
        offset: zml.Tensor, //< equivalent to `start_pos`
        state: CompressorState,
        state_idx: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor, CompressorState } {
        const offset_mod = offset.remainderConst(self.ratio);

        const x32 = x.convert(.f32);
        var kv = self.wkv.forward(x32);
        var score = self.wgate.forward(x32);
        score = score.add(self.ape.gather(.{ .r = offset_mod.squeeze(.batch) }, .{}).insertAxes(0, .{ .batch, .seq }));

        var new_state = blk: {
            const pos_idx = if (self.overlap) offset_mod.addConstant(self.ratio) else offset_mod;
            break :blk state.update(kv, score, pos_idx, state_idx);
        };

        kv = new_state.kv(state_idx);
        score = new_state.score(state_idx);

        if (self.overlap) {
            const shifted_state = new_state.update(
                kv.slice1d(.seq, .{ .start = self.ratio }),
                score.slice1d(.seq, .{ .start = self.ratio }),
                zml.Tensor.arange(.{ .end = self.ratio }, .u32).withTags(.{.seq}),
                state_idx,
            );

            // Don't care on computing garbage since it will be masked by the caller, however, we must
            // shift the cache only when we have a new compressed block.
            const should_compress = offset.addConstant(1).remainderConst(self.ratio).cmp(.EQ, zml.Tensor.zeroes(offset.shape()));
            new_state = CompressorState.select(should_compress, shifted_state, new_state);

            kv = zml.Tensor.concatenate(&.{
                kv.slice(&.{ .{}, .{ .end = self.ratio }, .{ .end = self.head_dim } }),
                kv.slice(&.{ .{}, .{ .start = self.ratio }, .{ .start = self.head_dim } }),
            }, .seq);

            score = zml.Tensor.concatenate(&.{
                score.slice(&.{ .{}, .{ .end = self.ratio }, .{ .end = self.head_dim } }),
                score.slice(&.{ .{}, .{ .start = self.ratio }, .{ .start = self.head_dim } }),
            }, .seq);
        }

        const pos_idx = offset.convert(.i64).addConstant(1).subConstant(@as(i64, @intCast(self.ratio))).withTags(.{.seq});
        return .{ kv.mul(score.broad(kv.shape()).softmax(.seq)).sum(.seq), pos_idx.convert(.i64), new_state.reuseBuffer(state) };
    }

    pub fn forward(
        self: Compressor,
        x: zml.Tensor,
        seqlen: zml.Tensor, //< the actual seqlen since `x.dim(.seq)` is the size of the allocated buffer
        offset: zml.Tensor, //< equivalent to `start_pos`
        state: CompressorState,
        state_idx: zml.Tensor,
    ) struct { zml.Tensor, CompressorState } {
        const is_prefill = x.dim(.seq) > 1;
        var kv, const pos_idx, const new_state = if (is_prefill) self.forwardPrefill(x.convert(.f32), seqlen, state, state_idx) else self.forwardDecode(x.convert(.f32), offset, state, state_idx);
        if (kv.dim(.seq) == 0) {
            return .{ zml.Tensor.zeroes(kv.shape().withDtype(x.dtype())).rename(.{ .seq = .kv }), new_state.reuseBuffer(state) };
        }

        const precomputed_freqs_cis = precompute_yarn(self.rope_head_dim, self.rope_opts);
        const freqs_cis = zml.Tensor.outer(pos_idx.convert(.f32), precomputed_freqs_cis);

        kv = kv.rename(.{ .hd = .d });
        kv = self.norm.forward(kv.convert(x.dtype()));
        kv = apply_rope(kv.rename(.{ .d = .hd }), freqs_cis, self.nope_head_dim, self.rope_head_dim);

        return .{ kv.rename(.{ .seq = .kv }), new_state.reuseBuffer(state) };
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
    return .{ @max(low, 0), @min(high, @as(f32, @floatFromInt(dim - 1))) };
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
        const ramp_factor = linear_ramp_factor(low, high, dim / 2);
        const smooth = zml.Tensor.scalar(1.0, .f32).sub(ramp_factor);

        freqs = freqs.divByConst(opts.factor).mul(ramp_factor).add(freqs.mul(smooth));
    }

    return freqs.withTags(.{.hd});
}

fn apply_yarn(x: zml.Tensor, inv_freq: zml.Tensor, reverse: bool) zml.Tensor {
    const x_real, const x_imag = zml.nn.splitRealImg(x, .interleaved);
    const cos = inv_freq.cos().convert(x.dtype()).broad(x_real.shape());
    var sin = inv_freq.sin().convert(x.dtype()).broad(x_real.shape());
    if (reverse) {
        sin = sin.scale(-1);
    }

    const y_real = x_real.mul(cos).sub(x_imag.mul(sin));
    const y_imag = x_real.mul(sin).add(x_imag.mul(cos));

    return zml.nn.mergeRealImg(y_real, y_imag, .interleaved);
}

fn apply_reverse_rope(x: zml.Tensor, freqs_cis: zml.Tensor, nope_dim: i64, rope_dim: i64) zml.Tensor {
    const split = x.split(-1, &.{ nope_dim, rope_dim });
    const x_rope = apply_yarn(split[1], freqs_cis, true);
    return zml.Tensor.concatenate(&.{ split[0], x_rope }, -1);
}

fn apply_rope(x: zml.Tensor, freqs_cis: zml.Tensor, nope_dim: i64, rope_dim: i64) zml.Tensor {
    const split = x.split(-1, &.{ nope_dim, rope_dim });
    const x_rope = apply_yarn(split[1], freqs_cis, false);
    return zml.Tensor.concatenate(&.{ split[0], x_rope }, -1);
}

// TODO: impl FWHT <https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform>
fn hadamard_rotation(x: zml.Tensor) zml.Tensor {
    const n = x.dim(-1);
    stdx.debug.assert(@mod(n, 2) == 0, "expect last dimension to be a power of 2, got: {}", .{n});

    var H = zml.Tensor.scalar(1, x.dtype()).reshape(.{ 1, 1 });

    while (H.dim(0) < n) {
        const H_plus = zml.Tensor.concatenate(&.{ H, H }, 1);
        const H_moins = zml.Tensor.concatenate(&.{ H, H.scale(-1) }, 1);
        H = zml.Tensor.concatenate(&.{ H_plus, H_moins }, 0);
    }

    const scale = zml.Tensor.scalar(n, x.dtype()).rsqrt();
    return x.dot(H.withTags(.{ .a, .hd }), .hd).mul(scale).rename(.{ .a = .hd });
}

pub const KvCache = struct {
    kv: zml.Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{ .kv = .fromShape(kv_shape) };
    }

    pub fn initBuffers(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KvCache) {
        return .{ .kv = try zml.Buffer.uninitialized(io, platform, self.kv.shape(), sharding, .{}) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KvCache)) void {
        self.kv.deinit();
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{ .kv = self.kv.reuseBuffer(other.kv) };
    }

    pub fn get(self: KvCache, cache_index: zml.Tensor) zml.Tensor {
        return self.kv.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_kv: zml.Tensor, token_position: zml.Tensor, cache_index: zml.Tensor) KvCache {
        const kv_shape = self.kv.shape().drop(.layer);
        const layer = cache_index.broad(token_position.shape());
        return .{
            .kv = self.kv.scatterSlices(.{ .layer = layer, .kv = token_position }, new_kv.transpose(kv_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.kv),
        };
    }
};

pub const CompressorState = struct {
    kv_state: zml.Tensor,
    score_state: zml.Tensor,

    pub fn init(config: Config, batch_size: u32, compression_ratio: u32, head_dim: u32) CompressorState {
        // NOTE: allocate twice the `compression_ratio` due to the overlaping mechanism.
        const coff: i64 = if (compression_ratio == 4) 2 else 1;

        const shape: zml.Shape = .init(.{
            .layer = config.num_hidden_layers,
            .batch = batch_size,
            .seq = coff * compression_ratio,
            .hd = coff * head_dim,
        }, .f32);

        return .{
            .kv_state = .fromShape(shape),
            .score_state = .fromShape(shape), //< TODO: Set to -inf
        };
    }

    pub fn initBuffers(self: CompressorState, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(CompressorState) {
        var score_slice: zml.Slice = try .alloc(allocator, self.score_state.shape());
        defer score_slice.free(allocator);
        @memset(score_slice.items(f32), -std.math.inf(f32));

        const score_state: zml.Buffer = try .fromSlice(io, platform, score_slice, sharding);

        return .{
            .kv_state = try zml.Buffer.uninitialized(io, platform, self.kv_state.shape(), sharding, .{}),
            // .score_state = try zml.Buffer.uninitialized(io, platform, self.score_state.shape(), sharding, .{}),
            .score_state = score_state,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(CompressorState)) void {
        self.kv_state.deinit();
        self.score_state.deinit();
    }

    pub fn kv(self: CompressorState, cache_index: zml.Tensor) zml.Tensor {
        return self.kv_state.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn score(self: CompressorState, cache_index: zml.Tensor) zml.Tensor {
        return self.score_state.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn reuseBuffer(self: CompressorState, other: CompressorState) CompressorState {
        return .{
            .kv_state = self.kv_state.reuseBuffer(other.kv_state),
            .score_state = self.score_state.reuseBuffer(other.score_state),
        };
    }

    pub fn update(self: CompressorState, new_kv: zml.Tensor, new_score: zml.Tensor, token_position: zml.Tensor, cache_index: zml.Tensor) CompressorState {
        const kv_shape = self.kv_state.shape().drop(.layer);
        const score_shape = self.score_state.shape().drop(.layer);
        const layer = cache_index.broad(token_position.shape());
        return .{
            .kv_state = self.kv_state.scatterSlices(.{ .layer = layer, .seq = token_position }, new_kv.transpose(kv_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.kv_state),
            .score_state = self.score_state.scatterSlices(.{ .layer = layer, .seq = token_position }, new_score.transpose(score_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.score_state),
        };
    }

    pub fn select(pred: zml.Tensor, on_true: CompressorState, on_false: CompressorState) CompressorState {
        return .{
            .kv_state = zml.Tensor.select(pred.broad(on_true.kv_state.shape()), on_true.kv_state, on_false.kv_state).reuseBuffer(on_false.kv_state),
            .score_state = zml.Tensor.select(pred.broad(on_true.score_state.shape()), on_true.score_state, on_false.score_state).reuseBuffer(on_false.score_state),
        };
    }
};

pub const IndexerCache = struct {
    kv: KvCache,
    state: CompressorState,

    pub fn init(mdl: Model, config: Config, batch_size: u32, seqlen: u32, compression_ratio: u32) IndexerCache {
        const indexer_shape: zml.Shape = .init(.{
            .layer = config.num_hidden_layers,
            .batch = batch_size,
            .kv = @divFloor(seqlen, compression_ratio),
            .hd = config.index_head_dim,
        }, mdl.embeds.embeds.weight.dtype());

        return .{
            .kv = .init(indexer_shape),
            .state = .init(config, batch_size, compression_ratio, config.index_head_dim),
        };
    }

    pub fn initBuffers(self: IndexerCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(IndexerCache) {
        return .{
            .kv = try self.kv.initBuffers(io, platform, sharding),
            .state = try self.state.initBuffers(allocator, io, platform, sharding),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(IndexerCache)) void {
        KvCache.unloadBuffers(&self.kv);
        CompressorState.unloadBuffers(&self.state);
    }

    pub fn reuseBuffer(self: IndexerCache, other: IndexerCache) IndexerCache {
        return .{
            .kv = self.kv.reuseBuffer(other.kv),
            .state = self.state.reuseBuffer(other.state),
        };
    }
};

pub const CSACache = struct {
    state: CompressorState,
    indexer: IndexerCache,
    compressed_kv: KvCache,

    pub fn init(mdl: Model, config: Config, batch_size: u32, seqlen: u32) CSACache {
        const compression_ratio = 4;

        const compressed_shape: zml.Shape = .init(.{
            .layer = config.num_hidden_layers,
            .batch = batch_size,
            .kv = @divTrunc(seqlen, compression_ratio),
            .hd = config.head_dim,
        }, mdl.embeds.embeds.weight.dtype());

        return .{
            .state = .init(config, batch_size, compression_ratio, config.head_dim),
            .indexer = .init(mdl, config, batch_size, seqlen, compression_ratio),
            .compressed_kv = .init(compressed_shape),
        };
    }

    pub fn initBuffers(self: CSACache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(CSACache) {
        return .{
            .state = try self.state.initBuffers(allocator, io, platform, sharding),
            .indexer = try self.indexer.initBuffers(allocator, io, platform, sharding),
            .compressed_kv = try self.compressed_kv.initBuffers(io, platform, sharding),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(CSACache)) void {
        CompressorState.unloadBuffers(&self.state);
        IndexerCache.unloadBuffers(&self.indexer);
        KvCache.unloadBuffers(&self.compressed_kv);
    }

    pub fn reuseBuffer(self: CSACache, other: CSACache) CSACache {
        return .{
            .state = self.state.reuseBuffer(other.state),
            .indexer = self.indexer.reuseBuffer(other.indexer),
            .compressed_kv = self.compressed_kv.reuseBuffer(other.compressed_kv),
        };
    }
};

pub const HCACache = struct {
    state: CompressorState,
    compressed_kv: KvCache,

    pub fn init(
        mdl: Model,
        config: Config,
        batch_size: u32,
        seqlen: u32,
    ) HCACache {
        const compression_ratio = 128;

        const compressed_shape: zml.Shape = .init(.{
            .layer = config.num_hidden_layers,
            .batch = batch_size,
            .kv = @divTrunc(seqlen, compression_ratio),
            .hd = config.head_dim,
        }, mdl.embeds.embeds.weight.dtype());

        return .{
            .state = .init(config, batch_size, compression_ratio, config.head_dim),
            .compressed_kv = .init(compressed_shape),
        };
    }

    pub fn initBuffers(self: HCACache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(HCACache) {
        return .{
            .state = try self.state.initBuffers(allocator, io, platform, sharding),
            .compressed_kv = try self.compressed_kv.initBuffers(io, platform, sharding),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(HCACache)) void {
        CompressorState.unloadBuffers(&self.state);
        KvCache.unloadBuffers(&self.compressed_kv);
    }

    pub fn reuseBuffer(self: HCACache, other: HCACache) HCACache {
        return .{
            .state = self.state.reuseBuffer(other.state),
            .compressed_kv = self.compressed_kv.reuseBuffer(other.compressed_kv),
        };
    }
};

pub const Cache = struct {
    sliding_window: KvCache,
    hca: HCACache,
    csa: CSACache,

    pub fn init(mdl: Model, config: Config, batch_size: u32, seqlen: u32) Cache {
        const kv_shape: zml.Shape = .init(.{
            .layer = config.num_hidden_layers,
            .batch = batch_size,
            .kv = config.sliding_window,
            .hd = config.head_dim,
        }, mdl.embeds.embeds.weight.dtype());

        return .{
            .sliding_window = .init(kv_shape),
            .hca = .init(mdl, config, batch_size, seqlen),
            .csa = .init(mdl, config, batch_size, seqlen),
        };
    }

    pub fn initBuffers(self: Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(Cache) {
        return .{
            .sliding_window = try self.sliding_window.initBuffers(io, platform, sharding),
            .hca = try self.hca.initBuffers(allocator, io, platform, sharding),
            .csa = try self.csa.initBuffers(allocator, io, platform, sharding),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Cache)) void {
        KvCache.unloadBuffers(&self.sliding_window);
        HCACache.unloadBuffers(&self.hca);
        CSACache.unloadBuffers(&self.csa);
    }

    pub fn reuseBuffer(self: Cache, other: Cache) Cache {
        return .{
            .sliding_window = self.kv.reuseBuffer(other.kv),
            .hca = self.hca.reuseBuffer(other.state),
            .csa = self.csa.reuseBuffer(other.state),
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

    pub fn init(store: zml.io.TensorStore.View, config: Config, compression_ratio: u32, rope_opts: RopeOpts) Indexer {
        const block_size = config.quantization_config.weight_block_size[0];

        return .{
            .proj = .init(store.createTensor("weights_proj.weight", .{ .sq, .d }, .{ .sq = .model, .d = .replicated }), null, .d),
            // .wq_b = .init(store.withPrefix("wq_b"), .{ .hd, .d }, block_size, .d),
            .wq_b = .{
                .scale = store.createTensor("wq_b.scale", null, .replicated),
                .weight = store.createTensor("wq_b.weight", .{ .hd, .d }, .{ .hd = .model, .d = .replicated }),
                .block_size = block_size,
                .tag = zml.Shape.toTag(.d),
            },
            .compressor = .init(store.withPrefix("compressor"), config, config.index_head_dim, compression_ratio, rope_opts),
            .index_head_dim = config.index_head_dim,
            .index_n_heads = config.index_n_heads,
            .index_topk = config.index_topk,
            .rope_opts = rope_opts,
            .rope_head_dim = config.qk_rope_head_dim,
            .nope_head_dim = config.index_head_dim - config.qk_rope_head_dim, //< Not sure about `index_head_dim`
            .ratio = compression_ratio,
            .softmax_scale = std.math.pow(f32, @floatFromInt(config.index_head_dim), -0.5),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Indexer)) void {
        self.proj.weight.deinit();
        FP8Linear.unloadBuffers(&self.wq_b);
        Compressor.unloadBuffers(&self.compressor);
    }

    pub fn forward(
        self: Indexer,
        x: zml.Tensor,
        qr: zml.Tensor,
        actual_seqlen: zml.Tensor,
        position: zml.Tensor,
        offset: zml.Tensor,
        cache: IndexerCache,
        cache_idx: zml.Tensor,
    ) struct { zml.Tensor, IndexerCache } {
        const precomputed_freqs_cis = precompute_yarn(self.rope_head_dim, self.rope_opts);
        const freqs_cis = blk: {
            const sh = position.shape().insert(.last, .{ .seq = x.dim(.seq) });
            const pos_idx = zml.Tensor.iota(sh, .seq).convert(.u32).add(position.convert(.u32).broad(sh)).broad(sh);
            break :blk zml.Tensor.outer(pos_idx.convert(.f32), precomputed_freqs_cis);
        };

        var q = self.wq_b.forward(qr);
        q = q.splitAxis(-1, .{ .h = self.index_n_heads, .hd = .auto });
        q = apply_rope(q, freqs_cis, self.nope_head_dim, self.rope_head_dim);
        q = hadamard_rotation(q);
        q = fp4ActQuant(q, 32);
        q = q.rename(.{ .hd = .d });

        var new_cache = cache;

        var kv, const new_state = self.compressor.forward(x, actual_seqlen, position, cache.state, cache_idx);
        new_cache.state = new_state;
        kv = blk: {
            kv = hadamard_rotation(kv);
            kv = fp4ActQuant(kv, 32);
            new_cache.kv = new_cache.kv.update(kv, compressedCacheIds(position, self.ratio, kv.dim(.kv)), cache_idx);

            break :blk new_cache.kv.get(cache_idx).rename(.{ .kv = .seq });
        };

        const compressed_mask = blk: {
            const kv_idx = zml.Tensor.arange(.{ .end = kv.dim(.seq) }, .u32);
            const offset_idx = if (x.dim(.seq) > 1) zml.Tensor.scalar(x.dim(.seq), .u32) else position.addConstant(1);
            break :blk kv_idx.cmp(.LT, offset_idx.divByConst(self.ratio).broad(kv_idx.shape())).withTags(.{.t}).insertAxes(0, .{ .batch, .seq });
        };

        var weights = self.proj.forward(x);
        weights = weights.mul(zml.Tensor.scalar(@as(f32, @floatFromInt(self.index_n_heads)), weights.dtype()).broad(weights.shape()).rsqrt().scale(self.softmax_scale));

        var index_score = q.dot(kv.rename(.{ .hd = .d, .seq = .t }), .d);
        index_score = index_score.relu().mul(weights.appendAxes(.{.t}).broad(index_score.shape())).sum(2).squeeze(2);
        index_score = zml.Tensor.select(compressed_mask.broad(index_score.shape()), index_score, zml.Tensor.constant(index_score.dtype().minValue()).broad(index_score.shape()));

        const is_prefill = x.dim(.seq) > 1;

        const topk_idxs = blk: {
            const seqlen = x.dim(.seq);
            const k: u32 = @intCast(@min(self.index_topk, kv.dim(.seq)));
            const offset_ = offset.insertAxes(.last, .{ .seq, .t }).convert(.i32);
            if (is_prefill) {
                const a = zml.Tensor.iota(.init(.{ seqlen, kv.dim(.seq) }, .i32), 1).convert(.i32);
                const b = zml.Tensor.arange(.{ .start = 1, .end = seqlen + 1 }, .i32).reshape(.{ seqlen, 1 }).divByConst(self.ratio).broad(a.shape()).convert(.i32);
                const mask = a.cmp(.GE, b).insertAxes(0, .{.batch});

                const selected = zml.Tensor.select(mask, zml.Tensor.constant(index_score.dtype().minValue()).broad(mask.shape()), zml.Tensor.zeroes(index_score.shape()));
                index_score = index_score.add(selected.broad(index_score.shape()));

                var topk_idxs = index_score.topK(-1, k, .{}).indices;

                const mask_ = topk_idxs.cmp(.GE, b.insertAxes(0, .{.batch}));
                break :blk zml.Tensor.select(mask_, zml.Tensor.scalar(-1, topk_idxs.dtype()).broad(topk_idxs.shape()), topk_idxs.add(offset_.broad(topk_idxs.shape())));
            } else {
                var topk_idxs = index_score.topK(-1, k, .{}).indices;
                const valid_count = position.addConstant(1).divByConst(self.ratio).insertAxes(.last, .{ .seq, .t }).convert(topk_idxs.dtype());
                const mask_ = topk_idxs.cmp(.GE, valid_count.broad(topk_idxs.shape()));
                break :blk zml.Tensor.select(mask_, zml.Tensor.scalar(-1, topk_idxs.dtype()).broad(topk_idxs.shape()), topk_idxs.add(offset_.broad(topk_idxs.shape())));
            }
        };

        return .{ topk_idxs.slice1d(.t, .{ .end = 2 }).convert(.i64), new_cache };
    }
};

const CompressionKind = union(enum) {
    none,
    csa: CSACompressor,
    hca: HCACompressor,
};

const CSACompressor = struct {
    indexer: Indexer,
    compressor: Compressor,

    pub fn unloadBuffers(self: *zml.Bufferized(CSACompressor)) void {
        Compressor.unloadBuffers(&self.compressor);
        Indexer.unloadBuffers(&self.indexer);
    }

    pub fn forward(
        self: CSACompressor,
        x: zml.Tensor,
        qr: zml.Tensor,
        seqlen: zml.Tensor,
        offset: zml.Tensor,
        window_size: u32,
        cache: CSACache,
        cache_idx: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor, CSACache } {
        var kv_compressed, const new_compressor_state = self.compressor.forward(x, seqlen, offset, cache.state, cache_idx);

        const is_prefill = x.dim(.seq) > 1;
        const compressed_offset: u32 = if (is_prefill) @intCast(x.dim(.seq)) else window_size;
        const topk_indexed, const new_indexer_cache = self.indexer.forward(
            x,
            qr.rename(.{ .q = .d }),
            seqlen,
            offset,
            zml.Tensor.scalar(compressed_offset, .u32),
            cache.indexer,
            cache_idx,
        );

        var new_cache = cache;
        new_cache.state = new_compressor_state;
        new_cache.indexer = new_indexer_cache;
        new_cache.compressed_kv = new_cache.compressed_kv.update(
            kv_compressed,
            compressedCacheIds(offset, self.compressor.ratio, kv_compressed.dim(.kv)),
            cache_idx,
        );

        kv_compressed = new_cache.compressed_kv.get(cache_idx);

        return .{ kv_compressed, topk_indexed, new_cache.reuseBuffer(cache) };
    }
};

const HCACompressor = struct {
    compressor: Compressor,

    pub fn unloadBuffers(self: *zml.Bufferized(HCACompressor)) void {
        Compressor.unloadBuffers(&self.compressor);
    }

    pub fn forward(
        self: HCACompressor,
        x: zml.Tensor,
        seqlen: zml.Tensor,
        offset: zml.Tensor,
        window_size: u32,
        cache: HCACache,
        cache_idx: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor, HCACache } {
        var kv_compressed, const new_compressor_state = self.compressor.forward(x, seqlen, offset, cache.state, cache_idx);

        var new_cache = cache;
        new_cache.state = new_compressor_state;
        new_cache.compressed_kv = new_cache.compressed_kv.update(kv_compressed, compressedCacheIds(offset, self.compressor.ratio, kv_compressed.dim(.kv)), cache_idx);

        kv_compressed = new_cache.compressed_kv.get(cache_idx);

        const is_prefill = x.dim(.seq) > 1;
        const compressed_offset: u32 = if (is_prefill) @intCast(x.dim(.seq)) else window_size;
        const topk_compressed = compressed_topk2(self.compressor.ratio, seqlen, @intCast(x.dim(.seq)), offset, compressed_offset, @intCast(kv_compressed.dim(.kv)), is_prefill);

        return .{ kv_compressed, topk_compressed, new_cache.reuseBuffer(cache) };
    }
};

pub const Attention = struct {
    compression: CompressionKind,
    attn_sink: zml.Tensor,
    kv_norm: RmsNorm,
    q_norm: RmsNorm,
    wq_a: FP8Linear,
    wq_b: FP8Linear,
    wkv: FP8Linear,
    wo_a_weight: zml.Tensor,
    wo_a_scale: zml.Tensor,
    wo_b: FP8Linear,
    eps: f32,
    rope_opts: RopeOpts,
    window_size: u32,
    local_heads: u32,
    head_dim: u32,
    softmax_scale: f32,
    rope_head_dim: u32,
    nope_head_dim: u32,
    o_groups: u32,
    o_lora_rank: u32,
    block_scale: u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, compression_kind: CompressionKind, rope_opts: RopeOpts) Attention {
        const block_size: usize = config.quantization_config.weight_block_size[0];

        return .{
            .compression = compression_kind,
            .attn_sink = store.createTensor("attn_sink", .{.h}, .replicated),
            .kv_norm = .init(store.createTensor("kv_norm.weight", .{.hd}, .replicated), config.rms_norm_eps, .hd),
            .q_norm = .init(store.createTensor("q_norm.weight", .{.q}, .replicated), config.rms_norm_eps, .q),
            .wq_a = .init(store.withPrefix("wq_a"), .{ .q, .d }, block_size, .d),
            // .wq_b = .init(store.withPrefix("wq_b"), .{ .hd, .q }, block_size, .q),
            .wq_b = .{
                .scale = store.createTensor("wq_b.scale", null, .replicated),
                .weight = store.createTensor("wq_b.weight", .{ .hd, .q }, .{ .hd = .model, .q = .replicated }),
                .block_size = block_size,
                .tag = zml.Shape.toTag(.q),
            },
            .wkv = .init(store.withPrefix("wkv"), .{ .hd, .d }, block_size, .d),
            .wo_a_weight = store.createTensor("wo_a.weight", .{ .hd, .d }, .{ .hd = .model, .d = .replicated }),
            .wo_a_scale = store.createTensor("wo_a.scale", null, .replicated),
            // .wo_b = .init(store.withPrefix("wo_b"), .{ .hd, .d }, block_size, .d),
            .wo_b = .{
                .scale = store.createTensor("wo_b.scale", null, .replicated),
                .weight = store.createTensor("wo_b.weight", .{ .hd, .d }, .{ .hd = .model, .d = .replicated }),
                .block_size = block_size,
                .tag = zml.Shape.toTag(.d),
            },
            .eps = config.rms_norm_eps,
            .rope_opts = rope_opts,
            .window_size = config.sliding_window,
            .local_heads = config.num_attention_heads,
            .head_dim = config.head_dim,
            .softmax_scale = std.math.pow(f32, @floatFromInt(config.head_dim), -0.5),
            .rope_head_dim = config.qk_rope_head_dim,
            .nope_head_dim = config.head_dim - config.qk_rope_head_dim,
            .o_groups = config.o_groups,
            .o_lora_rank = config.o_lora_rank,
            .block_scale = @intCast(block_size),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        self.attn_sink.deinit();
        RmsNorm.unloadBuffers(&self.kv_norm);
        RmsNorm.unloadBuffers(&self.q_norm);
        FP8Linear.unloadBuffers(&self.wq_a);
        FP8Linear.unloadBuffers(&self.wq_b);
        FP8Linear.unloadBuffers(&self.wkv);
        self.wo_a_weight.deinit();
        self.wo_a_scale.deinit();
        FP8Linear.unloadBuffers(&self.wo_b);

        switch (self.compression) {
            .none => {},
            .csa => |*compressor| CSACompressor.unloadBuffers(&compressor.*),
            .hca => |*compressor| HCACompressor.unloadBuffers(&compressor.*),
        }
    }

    pub fn forward(
        self: Attention,
        x: zml.Tensor,
        offset_idx: zml.Tensor,
        actual_seqlen: zml.Tensor,
        layer_idx: zml.Tensor,
        cache: Cache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, Cache } {
        _ = attention_parameters;
        // shape(x) = [batch, seq, d]
        const precomputed_freqs_cis = precompute_yarn(self.rope_head_dim, self.rope_opts);
        const freqs_cis = blk: {
            const sh = offset_idx.shape().insert(.last, .{ .seq = x.dim(.seq) });
            const pos_idx = zml.Tensor.iota(sh, .seq).convert(.u32).add(offset_idx.broad(sh));
            break :blk zml.Tensor.outer(pos_idx.convert(.f32), precomputed_freqs_cis);
        };

        const qr = self.q_norm.forward(self.wq_a.forward(x));

        var q = self.wq_b.forward(qr).splitAxis(.hd, .{ .h = self.local_heads, .hd = self.head_dim });
        q = q.mul(q.powByConst(2).mean(.hd).addConstant(self.eps).rsqrt());
        q = apply_rope(q, freqs_cis, self.nope_head_dim, self.rope_head_dim);

        var kv = self.kv_norm.forward(self.wkv.forward(x));
        kv = apply_rope(kv, freqs_cis, self.nope_head_dim, self.rope_head_dim);

        q = q.rename(.{ .seq = .q });
        kv = kv.rename(.{ .seq = .kv });

        const is_prefill = x.dim(.seq) > 1;

        var new_cache = cache;
        new_cache.sliding_window = blk: {
            if (is_prefill) {
                const n_tokens = @min(self.window_size, x.dim(.seq));
                // TODO: Add support when `actual_seqlen > self.window_size`
                const ids_dtype = actual_seqlen.dtype();

                const start = zml.Tensor.scalar(0, ids_dtype);
                const kv_gathered = kv.dynamicSlice(.{ .kv = zml.Tensor.DynSlice{ .start = start, .len = n_tokens } });

                const pos_ids = zml.Tensor.arange(.{ .end = n_tokens }, ids_dtype).withTags(.{.kv});
                break :blk new_cache.sliding_window.update(kv_gathered, pos_ids, layer_idx);
            } else {
                var pos_ids = zml.Tensor.iota(offset_idx.shape().insert(.last, .{ .kv = x.dim(.seq) }), .kv).convert(.u32);
                pos_ids = pos_ids.add(offset_idx.broad(pos_ids.shape())).remainderConst(self.window_size);
                break :blk new_cache.sliding_window.update(kv, pos_ids, layer_idx);
            }
        };

        // NOTE: during prefill attend on the full kv while during decode attend on the sliding window.
        kv = if (is_prefill) kv else new_cache.sliding_window.get(layer_idx);

        var topk = topk_window(offset_idx, actual_seqlen, x.dim(.seq), self.window_size, is_prefill);

        switch (self.compression) {
            .none => {},
            .csa => |compressor| {
                const kv_compressed, const topk_compressed, const new_csa_cache = compressor.forward(x, qr, actual_seqlen, offset_idx, self.window_size, cache.csa, layer_idx);
                new_cache.csa = new_csa_cache;

                kv = zml.Tensor.concatenate(&.{ kv, kv_compressed }, .kv);
                topk = zml.Tensor.concatenate(&.{ topk, topk_compressed }, .topk);
            },
            .hca => |compressor| {
                const kv_compressed, const topk_compressed, const new_hca_cache = compressor.forward(x, actual_seqlen, offset_idx, self.window_size, cache.hca, layer_idx);
                new_cache.hca = new_hca_cache;

                kv = zml.Tensor.concatenate(&.{ kv, kv_compressed }, .kv);
                topk = zml.Tensor.concatenate(&.{ topk, topk_compressed }, .topk);
            },
        }

        var attn = attention.sparseAttentionMLA(q, kv, self.attn_sink, topk, self.softmax_scale, attention_metadata, .cuda_fa2);
        attn = apply_reverse_rope(attn.rename(.{ .q = .seq }), freqs_cis, self.nope_head_dim, self.rope_head_dim);
        attn = attn.reshape(.{ .batch = attn.dim(0), .seq = attn.dim(1), .g = self.o_groups, .d = .auto });

        // TODO: Rework to do it during load.
        const wo_a = dequantizeScaledBlocks(self.wo_a_weight, self.wo_a_scale, self.block_scale, .bf16)
            .splitAxis(.hd, .{ .g = self.o_groups, .r = .auto });

        var o = attn.dot(wo_a, .d);
        o = o.mergeTranspose(.{ .g, .r }, .d);

        return .{ self.wo_b.forward(o).rename(.{ .hd = .d }), new_cache };
    }
};

fn ones(s: zml.Shape) zml.Tensor {
    return zml.Tensor.constant(s.dtype().one()).broad(s);
}

fn compressedCacheIds(offset: zml.Tensor, ratio: u32, kv_len: i64) zml.Tensor {
    const ids_shape = offset.shape().insert(.last, .{ .kv = kv_len });
    var ids = zml.Tensor.iota(ids_shape, .kv).convert(.u32);
    ids = ids.add(offset.divByConst(ratio).broad(ids_shape));
    return ids;
}

fn topk_window(offset_idx: zml.Tensor, seqlen: zml.Tensor, dim: i64, window_size: i64, is_prefill: bool) zml.Tensor {
    // offset_idx: [batch], seqlen: [{u32}]
    if (is_prefill) {
        const shape = offset_idx.shape().append(.{ .seq = dim }).append(.{ .topk = window_size });

        const base = zml.Tensor.iota(shape, .seq).convert(.i64);

        const v = base.subConstant(window_size).addConstant(1);
        const base_clamped = zml.Tensor.select(
            v.cmp(.LT, zml.Tensor.zeroes(v.shape())),
            zml.Tensor.zeroes(v.shape()),
            v,
        );
        const cols = zml.Tensor.iota(shape, .topk).convert(.i64);

        var matrix = base_clamped.broad(shape).add(cols);

        const before = matrix.cmp(.LE, base);
        const before_seqlen = base.cmp(.LT, seqlen.convert(.i64).broad(shape));
        const valid = before.logical(.AND, before_seqlen);

        matrix = zml.Tensor.select(valid, matrix, zml.Tensor.scalar(-1, .i64).broad(shape));
        return matrix;
    }

    const offset_idx_64 = offset_idx.convert(.i64);
    const pos_mod = offset_idx_64.remainderConst(window_size);

    const ids = zml.Tensor.iota(offset_idx_64.shape().insert(.last, .{ .topk = window_size }), .topk).convert(.i64);
    const matrix_wrap = ids.add(pos_mod.broad(ids.shape())).addConstant(1).remainderConst(window_size);

    const matrix_pad = zml.Tensor.select(
        ids.cmp(.LE, offset_idx_64.broad(ids.shape())),
        ids,
        zml.Tensor.scalar(-1, .i64).broad(ids.shape()),
    );

    const selected = offset_idx_64.insertAxes(.last, .{.topk}).broad(ids.shape())
        .cmp(.GE, zml.Tensor.scalar(window_size - 1, .i64).broad(ids.shape()));

    const matrix = zml.Tensor.select(selected, matrix_wrap, matrix_pad);
    return matrix.insertAxes(.topk, .{.seq});
}

fn compressed_topk2(ratio: u32, actual_seqlen: zml.Tensor, seqlen: u32, offset_idx: zml.Tensor, offset: u32, max_compressed: u32, is_prefill: bool) zml.Tensor {
    if (is_prefill) {
        const shape = offset_idx.shape().append(.{ .seq = seqlen }).append(.{ .topk = max_compressed });

        const row = zml.Tensor.iota(shape, .seq).addConstant(1).convert(.i64);
        const col = zml.Tensor.iota(shape, .topk).convert(.i64);

        const valid_count = row.divByConst(ratio); // [0, 0, 0, 0, 1, 1, 1, 1, ..., r, r, r, r]

        const before_row = col.cmp(.LT, valid_count);
        const before_actual = row.convert(.u32).cmp(.LE, actual_seqlen.broad(row.shape()));
        const valid = before_row.logical(.AND, before_actual);

        const matrix = col.addConstant(offset);
        return zml.Tensor.select(valid, matrix, zml.Tensor.scalar(-1, .i64).broad(shape));
    }

    const shape = offset_idx.shape().append(.{ .seq = seqlen }).append(.{ .topk = max_compressed });
    const base = zml.Tensor.iota(shape, .topk).convert(.u32);

    const valid_count = offset_idx.addConstant(1).divByConst(ratio);

    const valid = base.cmp(.LT, valid_count.broad(shape));
    const matrix = base.addConstant(offset).convert(.i64);

    const masked = zml.Tensor.select(
        valid,
        matrix,
        zml.Tensor.scalar(-1, .i64).broad(matrix.shape()),
    );

    return masked;
}

const MoE = struct {
    router: Gate,
    gate_up: zml.Tensor,
    gate_up_scale: zml.Tensor,
    down: zml.Tensor,
    down_scale: zml.Tensor,
    shared_experts: SharedExpert,
    block_size: u32,
    activation_threshold: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, i: usize) !MoE {
        const experts_store = store.withPrefix("experts");

        return .{
            .router = .init(store.withPrefix("gate"), config, i),
            .gate_up = experts_store.createTensor("w13.weight", .{ .expert, .dout, .d }, .{ .expert = .experts, .dout = .replicated, .d = .replicated }),
            .gate_up_scale = experts_store.createTensor("w13.scale", .{ .expert, .dout, .d }, .{ .expert = .experts, .dout = .replicated, .d = .replicated }),
            .down = experts_store.createTensor("w2.weight", .{ .expert, .d, .dout }, .{ .expert = .experts, .dout = .replicated, .d = .replicated }),
            .down_scale = experts_store.createTensor("w2.scale", .{ .expert, .d, .dout }, .{ .expert = .experts, .dout = .replicated, .d = .replicated }),
            .shared_experts = .init(store.withPrefix("shared_experts"), config),
            .block_size = @intCast(config.quantization_config.weight_block_size[0]),
            .activation_threshold = config.swiglu_limit,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MoE)) void {
        Gate.unloadBuffers(&self.router);
        self.gate_up.deinit();
        self.gate_up_scale.deinit();
        self.down.deinit();
        self.down_scale.deinit();
        SharedExpert.unloadBuffers(&self.shared_experts);
    }

    pub fn forward(
        self: MoE,
        x: zml.Tensor,
        input_ids: zml.Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) zml.Tensor {
        stdx.debug.assert(x.shape().hasTags(.{ .batch, .seq, .d }), "expect input tokens to has tags (.batch, .seq, .d) but got {f}", .{x.shape()});
        stdx.debug.assert(input_ids.shape().hasTags(.{
            .batch,
            .seq,
        }), "expect input tokens to has tags (.batch, .seq) but got {f}", .{input_ids.shape()});
        const topk_weight, const topk_ids = self.router.forward(x, input_ids);

        const moe_input = x.merge(.{ .b = .{ .batch, .seq } });

        var routed = zml.moe.forwardMoe_fp4(
            moe_input,
            topk_ids,
            topk_weight,
            self.gate_up,
            self.gate_up_scale,
            null,
            self.down,
            self.down_scale,
            null,
            self.activation_threshold,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("A16W4 MoE failed: {}", .{err});

        routed = routed.reshape(x.shape());
        routed = routed.convert(.f32).add(self.shared_experts.forward(x).convert(.f32));
        return routed.convert(x.dtype());
    }
};

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
                break :blk Kind{ .tid2eid = store.createTensor("tid2eid", .{ .tid, .eid }, .replicated) };
            } else {
                break :blk Kind{ .bias = store.createTensor("bias", .{.expert}, .replicated) };
            }
        };

        return .{
            .k = config.num_experts_per_tok,
            .kind = op,
            .proj = .init(store.createTensor("weight", .{ .expert, .d }, .replicated), null, .d),
            .scaling_factor = config.routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gate)) void {
        self.proj.weight.deinit();
        switch (self.kind) {
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

        const indices = switch (self.kind) {
            .bias => |bias| scores.add(bias.broad(scores.shape())).topK(-1, self.k, .{}).indices.renameTag(.expert, .eid).convert(.i64),
            .tid2eid => |tid2eid| tid2eid.gather(.{ .tid = input_ids }, .{}),
        };

        var weights = scores.gather(.{ .expert = indices }, .{});
        weights = weights.div(weights.sum(.eid));
        weights = weights.scale(self.scaling_factor);

        return .{ weights, indices.convert(.i32) };
    }
};

const SharedExpert = struct {
    w1: FP8Linear,
    w2: FP8Linear,
    w3: FP8Linear,
    activation_threshold: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) SharedExpert {
        const block_size = config.quantization_config.weight_block_size[0];

        return .{
            .w1 = .init(store.withPrefix("w1"), .{ .dint, .d }, block_size, .d),
            .w2 = .init(store.withPrefix("w2"), .{ .d, .dint }, block_size, .dint),
            .w3 = .init(store.withPrefix("w3"), .{ .dint, .d }, block_size, .d),
            .activation_threshold = config.swiglu_limit,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SharedExpert)) void {
        FP8Linear.unloadBuffers(&self.w1);
        FP8Linear.unloadBuffers(&self.w2);
        FP8Linear.unloadBuffers(&self.w3);
    }

    pub fn forward(self: SharedExpert, x: zml.Tensor) zml.Tensor {
        const threshold = zml.Tensor.scalar(self.activation_threshold, .f32);

        const up = self.w3.forward(x).convert(.f32).clamp(threshold.negate(), threshold);
        var gate = self.w1.forward(x).convert(.f32);
        gate = zml.Tensor.select(gate.cmp(.GT, threshold), threshold, gate);

        const x_ = gate.silu().mul(up);
        return self.w2.forward(x_.convert(x.dtype()));
    }
};

pub const Layer = struct {
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

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_idx: usize) !Layer {
        stdx.debug.assert(layer_idx < config.compress_ratios.len, "expected layer indices ({}) to be lower than compress ratios ({})", .{ layer_idx, config.compress_ratios.len });

        const compression_ratio = config.compress_ratios[layer_idx];

        const rope_opts: RopeOpts = .{
            .original_max_position_embeddings = if (compression_ratio == 0) 0 else config.rope_scaling.original_max_position_embeddings,
            .rope_theta = if (compression_ratio == 0) config.rope_theta else config.compress_rope_theta,
            .beta_fast = config.rope_scaling.beta_fast,
            .beta_slow = config.rope_scaling.beta_slow,
            .factor = config.rope_scaling.factor,
        };

        const kind: CompressionKind = switch (compression_ratio) {
            0 => .none,
            4 => .{
                .csa = .{
                    .compressor = .init(store.withPrefix("attn.compressor"), config, config.head_dim, compression_ratio, rope_opts),
                    .indexer = .init(store.withPrefix("attn.indexer"), config, compression_ratio, rope_opts),
                },
            },
            128 => .{ .hca = .{
                .compressor = .init(store.withPrefix("attn.compressor"), config, config.head_dim, compression_ratio, rope_opts),
            } },
            else => stdx.debug.assert(false, "{d} is an unexpected compression ratio. values should be either [full=0;csa=4;hca=128]", .{compression_ratio}),
        };

        return .{
            .attn_norm = .init(store.createTensor("attn_norm.weight", .{.d}, .replicated), config.rms_norm_eps, .d),
            .ffn_norm = .init(store.createTensor("ffn_norm.weight", .{.d}, .replicated), config.rms_norm_eps, .d),
            .attn = .init(store.withPrefix("attn"), config, kind, rope_opts),
            .ffn = try .init(store.withPrefix("ffn"), config, layer_idx),
            .norm_eps = config.rms_norm_eps,
            .hc_eps = config.hc_eps,
            .hc_mult = config.hc_mult,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .hc_attn = .{
                .base = store.createTensor("hc_attn_base", .{.hc}, .replicated),
                .func = .init(store.createTensor("hc_attn_fn", .{ .hc, .d }, .replicated), null, .d),
                .scale = store.createTensor("hc_attn_scale", .{.scale}, .replicated),
            },
            .hc_ffn = .{
                .base = store.createTensor("hc_ffn_base", .{.hc}, .replicated),
                .func = .init(store.createTensor("hc_ffn_fn", .{ .hc, .d }, .replicated), null, .d),
                .scale = store.createTensor("hc_ffn_scale", .{.b}, .replicated),
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Layer)) void {
        self.attn_norm.weight.deinit();
        self.ffn_norm.weight.deinit();
        self.hc_attn.base.deinit();
        self.hc_attn.func.weight.deinit();
        self.hc_attn.scale.deinit();
        self.hc_ffn.base.deinit();
        self.hc_ffn.func.weight.deinit();
        self.hc_ffn.scale.deinit();
        Attention.unloadBuffers(&self.attn);
        MoE.unloadBuffers(&self.ffn);
    }

    // TODO: maybe implement mHC-lite: <https://arxiv.org/html/2601.05732v1>?
    fn hc_split_sinkhorn(mixes: zml.Tensor, n: usize, mult: u32, eps: f32, opts: SinkhornOpts) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        // mixes = [batch, seq, hc, d]
        const pre_mix = mixes.slice1d(.hc, .{ .end = mult });
        const post_mix = mixes.slice1d(.hc, .{ .start = mult, .end = 2 * mult });
        const comb_mix = mixes.slice1d(.hc, .{ .start = 2 * mult });

        const base_pre = opts.base.slice1d(0, .{ .end = mult });
        const base_post = opts.base.slice1d(0, .{ .start = mult, .end = 2 * mult });
        const base_comb = opts.base.slice1d(0, .{ .start = 2 * mult }).splitAxis(-1, .{ .hc_in = mult, .hc = .auto });

        const pre = pre_mix.mul(opts.scale.choose1d(0, 0)).add(base_pre.broad(pre_mix.shape())).sigmoid().addConstant(eps);
        const post = post_mix.mul(opts.scale.choose1d(0, 1)).add(base_post.broad(post_mix.shape())).sigmoid().scale(2);

        var comb = comb_mix.splitAxis(-1, .{ .hc_in = mult, .hc = .auto });
        comb = comb.mul(opts.scale.choose1d(0, 2)).add(base_comb.insertAxes(0, .{ .batch, .seq }).broad(comb.shape()));
        comb = comb.softmax(.hc).addConstant(eps);
        comb = comb.div(comb.sum(.hc_in).addConstant(eps));

        for (1..n) |_| {
            comb = comb.div(comb.sum(.hc).addConstant(eps));
            comb = comb.div(comb.sum(.hc_in).addConstant(eps));
        }

        return .{ pre, post, comb };
    }

    fn hc_pre(self: Layer, x: zml.Tensor, opts: SinkhornOpts) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        // shape(x) = [batch, seq, hc, d]
        const x_ = x.merge(.{ .d = .{ .hc, .d } }).convert(.f32);

        const mixes = blk: {
            const m = opts.func.forward(x_);
            const rsqrt = x_.powByConst(2).mean(.d).addConstant(self.norm_eps).rsqrt();
            break :blk m.mul(rsqrt.broad(m.shape()));
        };

        const pre, const post, const comb = hc_split_sinkhorn(mixes, self.hc_sinkhorn_iters, self.hc_mult, self.hc_eps, opts);

        const y = x.convert(.f32).mul(pre.appendAxes(.{.d}).broad(x.shape())).sum(.hc).squeeze(.hc);
        return .{ y.convert(x.dtype()), post, comb };
    }

    fn hc_post(x: zml.Tensor, residual: zml.Tensor, post: zml.Tensor, comb: zml.Tensor) zml.Tensor {
        // x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        const p = post.appendAxes(.{.d});
        const x_ = x.insertAxes(.d, .{.hc}).convert(.f32);

        const s: zml.Shape = x_.shape().set(.hc, p.dim(.hc));
        const m = x_.broad(s).mul(p.broad(s));

        const r_32 = residual.rename(.{ .hc = .hc_in }).insertAxes(.d, .{.hc}).convert(.f32);
        const c = comb.appendAxes(.{.d}); //withTags(.{ .batch, .seq, .hc, .d });
        const s2 = c.shape().set(.d, r_32.dim(.d));
        const m2 = r_32.broad(s2).mul(c.broad(s2)).sum(.hc_in).squeeze(.hc_in);

        const f = m.add(m2);
        return f.convert(x.dtype());
    }

    pub fn forward(
        self: Layer,
        x: zml.Tensor,
        tokens: zml.Tensor,
        tokens_idx_offset: zml.Tensor,
        seqlen: zml.Tensor,
        layer_idx: zml.Tensor,
        cache: Cache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, Cache, zml.Tensor } {
        // x: [batch, seq, hc, d]
        var x_, const post, const comb = self.hc_pre(x, self.hc_attn);
        x_ = self.attn_norm.forward(x_);
        x_, const cache_ = self.attn.forward(x_, tokens_idx_offset, seqlen, layer_idx, cache, attention_metadata, attention_parameters);
        x_ = hc_post(x_, x, post, comb);

        var x_2, const post_2, const comb_2 = self.hc_pre(x_, self.hc_ffn);
        x_2 = self.ffn_norm.forward(x_2);
        x_2 = self.ffn.forward(x_2, tokens, moe_metadata, moe_parameters);
        x_2 = hc_post(x_2, x_, post_2, comb_2);

        return .{ x_2, cache_, layer_idx.addConstant(1) };
    }
};

const LmHead = struct {
    norm: RmsNorm,
    voc_proj: LinearF32,
    hc_base: zml.Tensor,
    hc_fn: zml.nn.Linear,
    hc_scale: zml.Tensor,
    hc_eps: f32,
    sampling_strategy: zml.nn.SamplingStrategy,

    pub fn init(store: zml.io.TensorStore.View, config: Config, generation: common.GenerationOptions) LmHead {
        return .{
            .norm = .init(
                store.createTensor("norm.weight", .{.d}, .replicated),
                config.rms_norm_eps,
                .d,
            ),
            .voc_proj = .init(store.createTensor("head.weight", .{ .voc, .d }, .replicated), null, .d),
            .hc_base = store.createTensor("hc_head_base", .{.hc}, .replicated),
            .hc_fn = .init(store.createTensor("hc_head_fn", .{ .hc, .d }, .replicated), null, .d),
            .hc_scale = store.createTensor("hc_head_scale", .{.batch}, .replicated),
            .hc_eps = config.hc_eps,
            .sampling_strategy = generation.sampling_strategy,
        };
    }

    fn hc_head(self: LmHead, x: zml.Tensor) zml.Tensor {
        // shape(x) = [b,s,hc,d]
        const x_ = x.merge(.{ .d = .{ .hc, .d } }).convert(.f32);

        const rsqrt = x_.powByConst(2).mean(.d).addConstant(self.norm.eps).rsqrt();

        var mixes = self.hc_fn.forward(x_);
        mixes = mixes.mul(rsqrt.broad(mixes.shape()));

        const mixes_shape = mixes.shape();
        const pre = mixes.mul(self.hc_scale.broad(mixes_shape)).add(self.hc_base.broad(mixes_shape)).sigmoid().addConstant(self.hc_eps);

        return x.convert(.f32).mul(pre.broad(x.shape())).sum(.hc).squeeze(.hc).convert(x.dtype());
    }

    pub fn forward(self: LmHead, x: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const logits = self.voc_proj.forward(self.norm.forward(self.hc_head(x)).convert(.f32)); //.choose1d(.seq, x.dim(.seq) - 1);
        const new_tokens, const new_rng = zml.nn.sampleTokens(logits, self.sampling_strategy, rng);
        return .{ new_tokens.convert(.u32), new_rng };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RmsNorm.unloadBuffers(&self.norm);
        self.voc_proj.weight.deinit();
        self.hc_base.deinit();
        self.hc_fn.weight.deinit();
        self.hc_scale.deinit();
    }
};

const TokenEmbedding = struct {
    embeds: zml.nn.TokenEmbedding,
    hc_mult: u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) TokenEmbedding {
        return .{
            .embeds = .{ .weight = store.createTensor("weight", .{ .voc, .d }, .replicated) },
            .hc_mult = config.hc_mult,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TokenEmbedding)) void {
        self.embeds.weight.deinit();
    }

    pub fn forwardTest(self: TokenEmbedding, tokens: zml.Tensor) zml.Tensor {
        return self.embeds.forward(tokens); //< [batch, seq, d]
    }

    pub fn forward(self: TokenEmbedding, tokens: zml.Tensor) zml.Tensor {
        stdx.debug.assert(tokens.shape().hasTags(.{
            .batch,
            .seq,
        }), "expect input tokens to has tags (.batch, .seq) but got {f}", .{tokens.shape()});
        const hidden = self.embeds.forward(tokens); //< [batch, seq, d]
        return hidden.insertAxes(.d, .{.hc}).repeat1d(.hc, @intCast(self.hc_mult)); //< [batch, seq, hc, d]
    }
};

pub const Model = struct {
    embeds: TokenEmbedding,
    layers: []Layer,
    lm_head: LmHead,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, generation: common.GenerationOptions) !Model {
        //const layers = try allocator.alloc(Layer, 2);
        const layers = try allocator.alloc(Layer, config.num_hidden_layers);

        for (layers, 0..) |*layer, i| {
            const layer_store = store.withPrefix("layers").withLayer(i);
            layer.* = try .init(layer_store, config, i);
        }

        return .{
            .embeds = .init(store.withPrefix("embed"), config),
            .layers = layers,
            .lm_head = .init(store, config, generation),
        };
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
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
            _ = took_ns; // autofix
            // log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{
            //     total_bytes,
            //     took,
            //     total_bytes * std.time.ns_per_s / took_ns,
            // });
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
        TokenEmbedding.unloadBuffers(&self.embeds);

        for (self.layers) |*layer| {
            Layer.unloadBuffers(layer);
        }
        allocator.free(self.layers);

        LmHead.unloadBuffers(&self.lm_head);
    }

    pub fn forward(
        self: Model,
        tokens: zml.Tensor,
        tokens_idx_offset: zml.Tensor,
        seqlen: zml.Tensor,
        rng: zml.Tensor.Rng,
        cache: Cache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, zml.Tensor.Rng, Cache } {
        var hidden = self.embeds.forward(tokens); //< [batch, seq, hc, d]

        var new_cache = cache;
        var layer_idx = zml.Tensor.scalar(@as(u32, 0), .u32);

        for (self.layers) |layer| {
            hidden, new_cache, layer_idx = layer.forward(
                hidden,
                tokens,
                tokens_idx_offset,
                seqlen,
                layer_idx,
                new_cache,
                attention_metadata,
                attention_parameters,
                moe_metadata,
                moe_parameters,
            ); //< [batch, seq, d]
        }

        const new_tokens, const new_rng = self.lm_head.forward(hidden, rng); //< [batch, voc]
        return .{ new_tokens.convert(tokens.dtype()), new_rng, new_cache };
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
        const config = self.parsed_config.value;

        if (!std.mem.eql(u8, config.expert_dtype, "fp4")) {
            return error.UnsupportedExpertType;
        }

        //const moe_backend: zml.moe.Backend = try .auto(platform, self.inner.layers[0].ffn.shared_experts.w1.weight.dtype());
        const moe_backend: zml.moe.Backend = .vanilla;
        const opts = inference.CompilationParameters.init(@intCast(seqlen), config, self.inner, shardings, backend, moe_backend);
        return try inference.CompiledModel.init(allocator, io, @constCast(platform), self.inner, @intCast(seqlen), progress, opts);
    }
};
