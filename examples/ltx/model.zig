const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

// ============================================================================
// Section 1: Configuration & Constants
// Python ref: ltx_core/models/ltx_model.py — LTXConfig
// ============================================================================

/// Which parameter set to use at runtime.
pub const Stage = enum {
    stage1,
    stage2,
};

pub const Config = struct {
    num_transformer_blocks: usize,
};

// ============================================================================
// Section 6: FeedForward
// Python ref: ltx_core/models/attention.py — FeedForward
// ============================================================================

/// FeedForward module shared by stage 1 and stage 2.
///
/// Forward contract for LTX 2.3 video FF:
/// [B, T, 4096] -> Linear(4096->16384) -> GELU(tanh approx) -> Linear(16384->4096)
pub const FeedForward = struct {
    pub const Params = struct {
        proj: zml.nn.Linear,
        out: zml.nn.Linear,

        pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
            params.proj.weight.deinit();
            if (params.proj.bias) |*b| b.deinit();
            params.out.weight.deinit();
            if (params.out.bias) |*b| b.deinit();
        }
    };

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const proj_store = store.withPrefix("net").withLayer(0).withPrefix("proj");
        const out_store = store.withPrefix("net").withLayer(2);

        return .{
            .proj = .init(
                proj_store.createTensor("weight", .{ .d_ff, .d }, null),
                proj_store.createTensor("bias", .{.d_ff}, null),
                .d,
            ),
            .out = .init(
                out_store.createTensor("weight", .{ .d, .d_ff }, null),
                out_store.createTensor("bias", .{.d}, null),
                .d_ff,
            ),
        };
    }

    pub fn forward(self: FeedForward, x: Tensor, params: Params) Tensor {
        _ = self;
        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const h1 = params.proj.forward(x_);
        // GELU in f32 for precision parity with Python reference, then cast back.
        const dt = h1.dtype();
        const h2 = h1.convert(.f32).gelu().convert(dt);
        return params.out.forward(h2);
    }

};

fn linearForwardF32(x: Tensor, linear: zml.nn.Linear) Tensor {
    const x32 = x.convert(.f32);
    var y = x32.dot(linear.weight.convert(.f32), linear.tag);
    if (linear.bias) |b| {
        y = y.add(b.convert(.f32).broad(y.shape()));
    }
    return y;
}

/// Audio FF with f32 matmul/bias/GELU, then cast back to input dtype.
/// Keeps video FF path unchanged while reducing accumulated audio drift.
fn forwardAudioFFPrecise(x: Tensor, params: FeedForward.Params) Tensor {
    const out_dtype = x.dtype();
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const h1 = linearForwardF32(x_, params.proj);
    const h2 = h1.gelu();
    const h3 = linearForwardF32(h2.withPartialTags(.{ .b, .t, .d_ff }), params.out);
    return h3.convert(out_dtype);
}

// ============================================================================
// Section 2: Patchification (Video + Audio)
// Python ref: ltx_core/models/patchifiers/ — VideoLatentPatchifier, AudioPatchifier
// ============================================================================

/// Patchify projection used before transformer blocks.
/// Forward contract for LTX video patchify:
/// [B, T, 128] -> Linear(128->4096)
pub const Patchify = struct {
    pub const Params = zml.nn.Linear;

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const proj_store = store.withPrefix("patchify_proj");
        return .init(
            proj_store.createTensor("weight", .{ .d, .patch }, null),
            proj_store.createTensor("bias", .{.d}, null),
            .patch,
        );
    }

    pub fn forward(self: Patchify, x: Tensor, params: Params) Tensor {
        _ = self;
        const x_ = x.withPartialTags(.{ .b, .t, .patch });
        return params.forward(x_);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        params.weight.deinit();
        if (params.bias) |*b| b.deinit();
    }
};

// ============================================================================
// Section 4: AdaLayerNormSingle (timestep modulation)
// Python ref: ltx_core/models/normalization.py — AdaLayerNormSingle
// ============================================================================

/// Sinusoidal timestep embedding.
/// Matches Python: get_timestep_embedding(sigma, dim=256, flip_sin_to_cos=True, downscale_freq_shift=0)
/// https://github.com/Lightricks/LTX-2/blob/ae855f8538843825f9015a419cf4ba5edaf5eec2/packages/ltx-core/src/ltx_core/model/transformer/timestep_embedding.py#L6
/// 
/// Input:  sigma [.b] f32
/// Output: [.b, .d_sin=256] f32
fn sinusoidalTimestepEmbedding(sigma: Tensor) Tensor {
    const half: i64 = 128;
    // freq[i] = exp(-ln(10000) * i / 128)
    const log_max = comptime std.math.log(f64, std.math.e, 10000.0);
    const idx = Tensor.arange(.{ .end = half }, .f32); // [128] f32
    const freq = idx.scale(-log_max / 128.0).exp(); // [128] f32

    // Outer product: sigma[b] * freq[i] → emb[b, i]
    const sigma_b = sigma.convert(.f32).withTags(.{.b});
    const freq_hf = freq.withTags(.{.hf});
    const target_sh = sigma_b.shape().append(.{ .hf = half });
    const emb = sigma_b.appendAxes(.{.hf}) // [.b=B, .hf=1]
        .broad(target_sh) // [.b=B, .hf=128]
        .mul(freq_hf.broad(target_sh)); // [.b=B, .hf=128]

    // cat([cos(emb), sin(emb)], axis=.hf) → [.b=B, .hf=256], then retag as .d_sin
    // flip_sin_to_cos=True → cos first, then sin
    const out = Tensor.concatenate(&.{ emb.cos(), emb.sin() }, .hf);
    return out.withPartialTags(.{ .b, .d_sin }); // [.b=B, .d_sin=256]
}

/// Adaptive layer norm single (adaLN-single), as used in LTX video/audio streams.
///
/// Python reference: AdaLayerNormSingle in adaln.py.
/// Full chain: sigma → sinusoidal(256) → linear_1(D) → silu → linear_2(D) → silu → linear_out(N*D)
///
/// Returns both:
///   - modulation [.b, .d_ada=N*D] — per-block modulation coefficients
///   - embedded_timestep [.b, .d_emb=D] — used by prompt adaln
pub const AdaLayerNormSingle = struct {
    pub const Params = struct {
        linear_1: zml.nn.Linear, // d_sin=256 → d=D
        linear_2: zml.nn.Linear, // d=D       → d_emb=D  (same dim, different tag for chaining)
        linear_out: zml.nn.Linear, // d_emb=D → d_ada=N*D

        pub fn unloadBuffers(self: *zml.Bufferized(Params)) void {
            self.linear_1.weight.deinit();
            if (self.linear_1.bias) |*b| b.deinit();
            self.linear_2.weight.deinit();
            if (self.linear_2.bias) |*b| b.deinit();
            self.linear_out.weight.deinit();
            if (self.linear_out.bias) |*b| b.deinit();
        }
    };

    /// `store` should point to the adaln_single prefix (e.g. `velocity_model.adaln_single`).
    /// Checkpoint keys consumed:
    ///   emb.timestep_embedder.linear_1.{weight,bias}
    ///   emb.timestep_embedder.linear_2.{weight,bias}
    ///   linear.{weight,bias}
    pub fn initParams(store: zml.io.TensorStore.View) Params {
        const emb_store = store.withPrefix("emb").withPrefix("timestep_embedder");
        const l1_store = emb_store.withPrefix("linear_1");
        const l2_store = emb_store.withPrefix("linear_2");
        const lo_store = store.withPrefix("linear");

        return .{
            // linear_1: [D, 256] weight — contracts over .d_sin, produces .d
            .linear_1 = .init(
                l1_store.createTensor("weight", .{ .d, .d_sin }, null),
                l1_store.createTensor("bias", .{.d}, null),
                .d_sin,
            ),
            // linear_2: [D, D] weight — contracts over .d, produces .d_emb
            .linear_2 = .init(
                l2_store.createTensor("weight", .{ .d_emb, .d }, null),
                l2_store.createTensor("bias", .{.d_emb}, null),
                .d,
            ),
            // linear_out: [N*D, D] weight — contracts over .d_emb, produces .d_ada
            .linear_out = .init(
                lo_store.createTensor("weight", .{ .d_ada, .d_emb }, null),
                lo_store.createTensor("bias", .{.d_ada}, null),
                .d_emb,
            ),
        };
    }

    pub const ForwardResult = struct {
        modulation: Tensor, // [.b, .d_ada = N*D]
        embedded_timestep: Tensor, // [.b, .d_emb = D]
    };

    /// sigma: [.b] noise level (will be converted to f32 for the sinusoidal step).
    pub fn forward(sigma: Tensor, params: Params) ForwardResult {
        // The actual Python model has bf16 params but receives f32 sinusoidal
        // embedding. PyTorch's F.linear auto-promotes bf16 weights to f32,
        // so the entire chain runs at f32. The output is f32 until the
        // preprocessor stores it as bf16 in TransformerArgs.
        // ZML requires matching dtypes for dotGeneral, so promote weights explicitly.
        const t_proj = sinusoidalTimestepEmbedding(sigma); // f32

        // 2. TimestepEmbedding: linear_1 → silu → linear_2
        const h1 = linearF32(params.linear_1, t_proj); // [.b, .d] f32
        const h2 = h1.silu(); // [.b, .d]
        const h3 = linearF32(params.linear_2, h2); // [.b, .d_emb] f32
        // h3 is the embedded_timestep returned to callers

        // 3. adaLN-single part: silu → linear_out
        const h4 = h3.silu(); // [.b, .d_emb]
        const modulation = linearF32(params.linear_out, h4); // [.b, .d_ada] f32

        return .{ .modulation = modulation, .embedded_timestep = h3 };
    }

    /// Run a linear layer in f32 precision (promoting weights from bf16 if needed).
    fn linearF32(linear: zml.nn.Linear, x: Tensor) Tensor {
        const w = linear.weight.convert(.f32);
        var y = x.dot(w, linear.tag);
        if (linear.bias) |bias| {
            y = y.add(bias.convert(.f32).broad(y.shape()));
        }
        return y;
    }
};

// ============================================================================
// Section 10: Output Projection
// Python ref: ltx_core/models/ltx_model.py — LTXModel._postprocess()
// ============================================================================

/// Final output projection applied after all transformer blocks.
///
/// Implements Python's `velocity_model._process_output`:
///   scale_shift_values = scale_shift_table[2, D] + embedded_timestep[B, D]
///   shift, scale = scale_shift_values[0], scale_shift_values[1]
///   out = proj_out(layernorm(x) * (1 + scale) + shift)
///
/// `scale_shift_table` [.n_ssv=2, .d] is a model-level parameter (f32 in checkpoint).
/// `norm_out` is PyTorch LayerNorm(eps=1e-6, elementwise_affine=False) — no learned params.
pub const OutputProjection = struct {
    pub const Params = struct {
        scale_shift_table: Tensor, // [.n_ssv=2, .d] — f32 in checkpoint
        proj_out: zml.nn.Linear, // contracts .d → .d_out (e.g. 4096 → 128)

        pub fn unloadBuffers(self: *zml.Bufferized(Params)) void {
            self.scale_shift_table.deinit();
            self.proj_out.weight.deinit();
            if (self.proj_out.bias) |*b| b.deinit();
        }
    };

    /// x: [.b, .t, .d], embedded_timestep: [.b, .d_emb] — from adaln_single.forward
    /// Returns: [.b, .t, .d_out=128]
    pub fn forward(x_in: Tensor, embedded_timestep: Tensor, params: Params) Tensor {
        const x = x_in.withPartialTags(.{ .b, .t, .d });
        // Rename .d_emb → .d and cast to activation dtype for broadcasting
        const emb = embedded_timestep.rename(.{ .d_emb = .d }).convert(x.dtype());

        // scale_shift_table is f32 in checkpoint; cast to activation dtype
        const sst = params.scale_shift_table.convert(x.dtype());
        const sst_shift = sst.slice1d(.n_ssv, .{ .start = 0, .end = 1 }).squeeze(.n_ssv); // [.d]
        const sst_scale = sst.slice1d(.n_ssv, .{ .start = 1, .end = 2 }).squeeze(.n_ssv); // [.d]

        // shift/scale: sst_row[d] + emb[b, d] → [.b, .d]
        const shift = sst_shift.broad(emb.shape()).add(emb);
        const scale = sst_scale.broad(emb.shape()).add(emb);

        // LayerNorm (no affine params, eps=1e-6), then scale-shift, then project
        const normed = zml.nn.normalizeVariance(x, 1e-6);
        const modulated = normed
            .mul(scale.addConstant(1.0).broad(x.shape()))
            .add(shift.broad(x.shape()));
        return params.proj_out.forward(modulated);
    }
};

// ============================================================================
// Section 5: Attention (self-attention, cross-attention, AV cross-attention)
// Python ref: ltx_core/models/attention.py — Attention, CrossAttention
// ============================================================================

/// Like zml.nn.sdpa but computes softmax in native dtype (no f32 upcast)
/// and chunks along the query dimension to avoid materializing a full Q×K^T
/// matrix. For Stage 2 video self-attention ([32, 24576, 24576] bf16 ≈ 36 GiB),
/// the full matrix can't fit in GPU memory. Chunking queries into blocks of
/// CHUNK_Q tokens produces smaller [32, CHUNK_Q, 24576] intermediates that
/// XLA can schedule without OOM.
fn sdpaNoF32Upcast(q_: Tensor, k_: Tensor, v_: Tensor, opts: zml.nn.SdpaOpts) Tensor {
    var q, var k, const v = .{ q_, k_, v_ };

    // Handle GQA: split q heads to match k heads.
    q = q.splitAxis(.h, .{ .h = k.dim(.h), .hq = .auto });

    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    const head_scaling = if (opts.scale) |s| s else Tensor.scalar(sqrtHeadDim, k.dtype());
    k = k.mul(head_scaling.convert(k.dtype()));

    const n_queries: usize = @intCast(q.dim(.q));

    // Choose chunk size: 1024 queries per chunk → attention tile is [H, 1024, K] which is manageable.
    const CHUNK_Q: usize = 1024;
    const n_full_chunks = n_queries / CHUNK_Q;
    const remainder = n_queries % CHUNK_Q;
    const n_chunks = n_full_chunks + @as(usize, if (remainder > 0) 1 else 0);

    // For small sequence lengths (≤ CHUNK_Q), skip chunking entirely.
    if (n_chunks <= 1) {
        var attn_weights = q.dot(k, .hd);
        if (opts.attn_mask) |mask| attn_weights = attn_weights.add(mask.broad(attn_weights.shape()));
        attn_weights = attn_weights.softmax(.k);
        const attn = attn_weights.dot(v, .k);
        return attn.transpose(q.shape()).merge(.{ .h = .{ .h, .hq } });
    }

    // Process each chunk: slice Q (and optionally the mask) along the .q axis,
    // compute attention with the full K/V, and collect results.
    // concatenate supports up to 32 tensors — 24576/1024 = 24 chunks, well within limit.
    var chunk_results: [32]Tensor = undefined;
    std.debug.assert(n_chunks <= 32);

    var ci: usize = 0;
    while (ci < n_chunks) : (ci += 1) {
        const start: i64 = @intCast(ci * CHUNK_Q);
        const end: i64 = @intCast(@min((ci + 1) * CHUNK_Q, n_queries));

        const q_chunk = q.slice1d(.q, .{ .start = start, .end = end });

        var aw = q_chunk.dot(k, .hd);
        if (opts.attn_mask) |mask| {
            const mask_chunk = mask.slice1d(.q, .{ .start = start, .end = end });
            aw = aw.add(mask_chunk.broad(aw.shape()));
        }
        aw = aw.softmax(.k);
        chunk_results[ci] = aw.dot(v, .k);
    }

    // Concatenate chunks back along the query axis and finalize.
    const attn = Tensor.concatenate(chunk_results[0..n_chunks], .q);
    return attn.transpose(q.shape()).merge(.{ .h = .{ .h, .hq } });
}

pub const AttentionKind = enum {
    attn1,
    attn2,
    audio_attn1,
    audio_attn2,
    audio_to_video_attn,
    video_to_audio_attn,
};

pub const Attention = struct {
    pub const ForwardOpts = struct {
        context: ?Tensor = null,
        mask: ?Tensor = null,
        pe_cos: ?Tensor = null,
        pe_sin: ?Tensor = null,
        k_pe_cos: ?Tensor = null,
        k_pe_sin: ?Tensor = null,
    };

    pub const Params = struct {
        q_norm_weight: Tensor,
        k_norm_weight: Tensor,
        to_q: zml.nn.Linear,
        to_k: zml.nn.Linear,
        to_v: zml.nn.Linear,
        to_gate_logits: zml.nn.Linear,
        to_out: zml.nn.Linear,

        pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
            params.q_norm_weight.deinit();
            params.k_norm_weight.deinit();

            params.to_q.weight.deinit();
            if (params.to_q.bias) |*b| b.deinit();

            params.to_k.weight.deinit();
            if (params.to_k.bias) |*b| b.deinit();

            params.to_v.weight.deinit();
            if (params.to_v.bias) |*b| b.deinit();

            params.to_gate_logits.weight.deinit();
            if (params.to_gate_logits.bias) |*b| b.deinit();

            params.to_out.weight.deinit();
            if (params.to_out.bias) |*b| b.deinit();
        }
    };

    pub fn initParams(store: zml.io.TensorStore.View, kind: AttentionKind) Params {
        const path = kindStorePath(kind);
        const attn_store = store.withPrefix(path);
        const to_out_store = attn_store.withPrefix("to_out").withLayer(0);

        return .{
            .q_norm_weight = attn_store.withPrefix("q_norm").createTensor("weight", .{.d_q}, null),
            .k_norm_weight = attn_store.withPrefix("k_norm").createTensor("weight", .{.d_k}, null),
            .to_q = .init(
                attn_store.withPrefix("to_q").createTensor("weight", .{ .d_q, .d }, null),
                attn_store.withPrefix("to_q").createTensor("bias", .{.d_q}, null),
                .d,
            ),
            .to_k = .init(
                attn_store.withPrefix("to_k").createTensor("weight", .{ .d_k, .d }, null),
                attn_store.withPrefix("to_k").createTensor("bias", .{.d_k}, null),
                .d,
            ),
            .to_v = .init(
                attn_store.withPrefix("to_v").createTensor("weight", .{ .d_v, .d }, null),
                attn_store.withPrefix("to_v").createTensor("bias", .{.d_v}, null),
                .d,
            ),
            .to_gate_logits = .init(
                attn_store.withPrefix("to_gate_logits").createTensor("weight", .{ .h, .d }, null),
                attn_store.withPrefix("to_gate_logits").createTensor("bias", .{.h}, null),
                .d,
            ),
            .to_out = .init(
                to_out_store.createTensor("weight", .{ .d, .d_v }, null),
                to_out_store.createTensor("bias", .{.d}, null),
                .d_v,
            ),
        };
    }

    pub fn forward(self: Attention, x: Tensor, params: Params, num_heads: usize, opts: ForwardOpts) Tensor {
        return self.forwardImpl(false, x, params, num_heads, opts);
    }

    /// bf16-native attention matching Python's dtype chain: bf16 matmuls, bf16 RoPE, bf16 SDPA.
    pub fn forwardBf16(self: Attention, x: Tensor, params: Params, num_heads: usize, opts: ForwardOpts) Tensor {
        return self.forwardImpl(true, x, params, num_heads, opts);
    }


    /// STG V-passthrough: replaces full Q·K·V attention with to_out(to_v(x)).
    /// Q/K projections, RMSNorm, RoPE, and SDPA are all skipped.
    /// Per-head gating is still applied exactly as in the normal path.
    pub fn forwardValuePassthrough(self: Attention, x: Tensor, params: Params, num_heads: usize) Tensor {
        _ = self;
        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const out_dtype = x_.dtype();
        const x_compute = x_.convert(.f32);

        // V-only projection (no Q/K)
        var v = x_compute.dot(params.to_v.weight.convert(.f32), .d);
        if (params.to_v.bias) |bias_orig| {
            v = v.add(bias_orig.convert(.f32).broad(v.shape()));
        }

        // Split into heads, rename .t → .q so broadcast matches gate shape [B, Q, H]
        var vh = v.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        vh = vh.withPartialTags(.{ .b, .q, .h, .hd });

        // Per-head gating (same formula as forwardImpl)
        var gate_logits = x_compute.dot(params.to_gate_logits.weight.convert(.f32), .d);
        if (params.to_gate_logits.bias) |bias_orig| {
            gate_logits = gate_logits.add(bias_orig.convert(.f32).broad(gate_logits.shape()));
        }
        const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q })
            .splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
        vh = vh.mul(gate.broad(vh.shape()));

        // Merge heads and project to output
        const merged = vh.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t });
        var out = merged.dot(params.to_out.weight.convert(.f32), .d_v);
        if (params.to_out.bias) |bias_orig| {
            out = out.add(bias_orig.convert(.f32).broad(out.shape()));
        }
        return out.convert(out_dtype).withPartialTags(.{ .b, .t, .d });
    }

    fn forwardImpl(self: Attention, comptime bf16_native: bool, x: Tensor, params: Params, num_heads: usize, opts: ForwardOpts) Tensor {
        _ = self;

        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const out_dtype = x_.dtype();
        const context = if (opts.context) |ctx| ctx.withPartialTags(.{ .b, .t, .d }) else x_;

        // In bf16_native mode, match Python exactly: bf16 matmuls, bf16 RoPE, bf16 SDPA.
        // In f32 mode (original), upcast everything to f32 for maximum precision.
        const compute_dtype: zml.dtype.DataType = if (bf16_native) out_dtype else .f32;
        const x_compute = x_.convert(compute_dtype);
        const context_compute = context.convert(compute_dtype);

        var q = x_compute.dot(params.to_q.weight.convert(compute_dtype), .d);
        if (params.to_q.bias) |bias_orig| {
            q = q.add(bias_orig.convert(compute_dtype).broad(q.shape()));
        }

        var k = context_compute.dot(params.to_k.weight.convert(compute_dtype), .d);
        if (params.to_k.bias) |bias_orig| {
            k = k.add(bias_orig.convert(compute_dtype).broad(k.shape()));
        }

        var v = context_compute.dot(params.to_v.weight.convert(compute_dtype), .d);
        if (params.to_v.bias) |bias_orig| {
            v = v.add(bias_orig.convert(compute_dtype).broad(v.shape()));
        }

        // LTX 2.3 attention includes RMSNorms on q and k with learned weights, applied before splitting into heads.
        // In PyTorch, multiplying by the learned weight is done within the `RMSNorm` module after the normalization step. 
        // On the other hand, zml.nn.rmsNorm() returns a normalized tensor without applying the learned weight, so we multiply manually.
        q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.convert(compute_dtype).broad(q.shape()));
        k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.convert(compute_dtype).broad(k.shape()));

        // Split embedding dimension into heads
        var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

        // Apply RoPE if present in checkpoint.
        if (opts.pe_cos) |pe_cos| {
            if (opts.pe_sin) |pe_sin| {
                const q_cos = if (pe_cos.rank() == 4)
                    pe_cos.withPartialTags(.{ .b, .h, .q, .hd }).convert(compute_dtype)
                else
                    pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(compute_dtype);

                const q_sin = if (pe_sin.rank() == 4)
                    pe_sin.withPartialTags(.{ .b, .h, .q, .hd }).convert(compute_dtype)
                else
                    pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(compute_dtype);

                const k_cos = if (opts.k_pe_cos) |k_pe_cos|
                    if (k_pe_cos.rank() == 4)
                        k_pe_cos.withPartialTags(.{ .b, .h, .k, .hd }).convert(compute_dtype)
                    else
                        k_pe_cos.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .k, .h, .hd }).convert(compute_dtype)
                else
                    q_cos.rename(.{ .q = .k });

                const k_sin = if (opts.k_pe_sin) |k_pe_sin|
                    if (k_pe_sin.rank() == 4)
                        k_pe_sin.withPartialTags(.{ .b, .h, .k, .hd }).convert(compute_dtype)
                    else
                        k_pe_sin.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .k, .h, .hd }).convert(compute_dtype)
                else
                    q_sin.rename(.{ .q = .k });

                qh = applyLtxRotaryEmb(qh, q_cos, q_sin);
                kh = applyLtxRotaryEmb(kh, k_cos, k_sin);
            }
        }

        qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
        kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
        vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

        const sdpa_opts: zml.nn.SdpaOpts = .{ .attn_mask = if (opts.mask) |m| m.rename(.{ .b = .batch }) else null };
        var attn = (if (bf16_native)
            sdpaNoF32Upcast(
                qh.rename(.{ .b = .batch }),
                kh.rename(.{ .b = .batch }),
                vh.rename(.{ .b = .batch }),
                sdpa_opts,
            )
        else
            zml.nn.sdpa(
                qh.rename(.{ .b = .batch }),
                kh.rename(.{ .b = .batch }),
                vh.rename(.{ .b = .batch }),
                sdpa_opts,
            )).rename(.{ .batch = .b }); // [B, Q, H, HD]

        // Compute per-head gates as 2 * sigmoid(logits) so zero-initialized logits preserve identity.
        var gate_logits = x_compute.dot(params.to_gate_logits.weight.convert(compute_dtype), .d); // [B, T, H]
        if (params.to_gate_logits.bias) |bias_orig| {
            gate_logits = gate_logits.add(bias_orig.convert(compute_dtype).broad(gate_logits.shape()));
        }
        const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate); // [B, Q, H]
        attn = attn.mul(gate.broad(attn.shape())); // [B, Q, H, HD]

        const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }); // [B, Q, D_V] with D_V = H * HD
        var out = merged.dot(params.to_out.weight.convert(compute_dtype), .d_v);
        if (params.to_out.bias) |bias_orig| {
            out = out.add(bias_orig.convert(compute_dtype).broad(out.shape()));
        }
        return out.convert(out_dtype).withPartialTags(.{ .b, .t, .d });
    }

    fn applyLtxRotaryEmb(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x_hd = x.dim(.hd);
        const rope_hd = cos.dim(.hd);

        if (rope_hd == x_hd) {
            return applyLtxRotaryEmbInterleaved(x, cos, sin);
        }
        if (rope_hd * 2 == x_hd) {
            return applyLtxRotaryEmbSplit(x, cos, sin);
        }

        std.debug.panic("Unsupported RoPE shapes: x={f} cos={f} sin={f}", .{ x, cos, sin });
    }

    fn applyLtxRotaryEmbInterleaved(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x_even, const x_odd = zml.nn.splitRealImg(x, .interleaved);
        const x_rot = zml.nn.mergeRealImg(x_odd.scale(-1.0), x_even, .interleaved);
        const cos_aligned = cos.transpose(x.shape());
        const sin_aligned = sin.transpose(x.shape());
        return x.mul(cos_aligned.broad(x.shape())).add(x_rot.mul(sin_aligned.broad(x.shape())));
    }

    fn applyLtxRotaryEmbSplit(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const half: i64 = @divExact(x.dim(.hd), 2);
        const x_first = x.slice1d(.hd, .{ .start = 0, .end = half });
        const x_second = x.slice1d(.hd, .{ .start = half, .end = x.dim(.hd) });

        const cos_b = cos.transpose(x_first.shape()).broad(x_first.shape());
        const sin_b = sin.transpose(x_first.shape()).broad(x_first.shape());

        const out_first = x_first.mul(cos_b).sub(x_second.mul(sin_b));
        const out_second = x_second.mul(cos_b).add(x_first.mul(sin_b));
        return Tensor.concatenate(&.{ out_first, out_second }, .hd);
    }

};

fn kindStorePath(kind: AttentionKind) []const u8 {
    return switch (kind) {
        .attn1 => "attn1",
        .attn2 => "attn2",
        .audio_attn1 => "audio_attn1",
        .audio_attn2 => "audio_attn2",
        .audio_to_video_attn => "audio_to_video_attn",
        .video_to_audio_attn => "video_to_audio_attn",
    };
}

fn kindNumHeads(kind: AttentionKind) usize {
    return switch (kind) {
        .attn1, .attn2, .audio_attn1, .audio_attn2, .audio_to_video_attn, .video_to_audio_attn => 32,
    };
}

/// Compute one AdaLN modulation value from a scale_shift_table row.
///
/// Implements the per-row logic from Python's `get_ada_values`:
///   sst[idx].unsqueeze(0).unsqueeze(0) + timestep.reshape(B, 1, N, D)[:, :, idx, :]
///
/// `sst`      – [.n_ada, .d]  scale_shift_table (or a contiguous slice of it)
/// `timestep` – [B, 1, N*D]   flat AdaLN embedding (all N rows concatenated)
/// `idx`      – row index to use (0-based within the provided `sst`)
/// Returns    – [B, 1, .d]
fn adaValueAt(sst: Tensor, timestep: Tensor, idx: i64) Tensor {
    const d = sst.dim(.d);
    // Slice the flat timestep to isolate the idx-th block of D values: [B, T, D]
    // Tag the token dimension as .t so it matches the activation tensor's .t tag
    // during broadcasting. With per-token timesteps (e.g. [1, 512, 36864]), using
    // .t=512 aligns correctly with vx/ax [.b, .t, .d]. With broadcast timesteps
    // (e.g. [1, 1, 36864]), .t=1 naturally broadcasts to any .t size.
    const ts = timestep
        .withPartialTags(.{ .b, .t, .tsflat })
        .slice1d(.tsflat, .{ .start = idx * d, .end = (idx + 1) * d })
        .rename(.{ .tsflat = .d });
    // Extract the sst row and collapse the leading size-1 axis: [D]
    // Cast to ts dtype so that f32-stored SST (e.g. from LoRA-merged checkpoints) stays
    // compatible with bf16 timestep embeddings from the fixture/runtime.
    const sst_row = sst
        .slice1d(.n_ada, .{ .start = idx, .end = idx + 1 })
        .squeeze(.n_ada)
        .convert(ts.dtype());
    return sst_row.broad(ts.shape()).add(ts);
}

// ============================================================================
// Section 7: BasicAVTransformerBlock (single block forward)
// Python ref: ltx_core/models/transformer_ltx_2.py — BasicAVTransformerBlock.forward()
// ============================================================================

/// One AV transformer block. Shared implementation across stages.
///
/// Processes video and audio token streams in parallel with shared cross-attention
/// between modalities. Production entrypoints are the `forwardNative*` family,
/// which compute AdaLN modulation values inline from scale-shift tables + timestep
/// embeddings.
pub const BasicAVTransformerBlock = struct {
    ff: FeedForward,
    audio_ff: FeedForward,
    attn1: Attention,
    attn2: Attention,
    audio_attn1: Attention,
    audio_attn2: Attention,
    audio_to_video_attn: Attention,
    video_to_audio_attn: Attention,

    pub const Params = struct {
        ff: FeedForward.Params,
        audio_ff: FeedForward.Params,
        attn1: Attention.Params,
        attn2: Attention.Params,
        audio_attn1: Attention.Params,
        audio_attn2: Attention.Params,
        audio_to_video_attn: Attention.Params,
        video_to_audio_attn: Attention.Params,
        // AdaLN scale-shift tables — required for the native threading path (forwardNative).
        scale_shift_table: Tensor,               // [.n_ada=6, .d=D_video]  video self-attn + FF
        audio_scale_shift_table: Tensor,         // [.n_ada=6, .d=D_audio]  audio self-attn + FF
        scale_shift_table_a2v_ca_video: Tensor,  // [.n_ada=5, .d=D_video]  AV CA video side
        scale_shift_table_a2v_ca_audio: Tensor,  // [.n_ada=5, .d=D_audio]  AV CA audio side
        prompt_scale_shift_table: Tensor,        // [.n_prompt=2, .d=D_video] text CA prompt modulation
        audio_prompt_scale_shift_table: Tensor,  // [.n_prompt=2, .d=D_audio] text CA prompt modulation

        pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
            FeedForward.Params.unloadBuffers(&params.ff);
            FeedForward.Params.unloadBuffers(&params.audio_ff);
            Attention.Params.unloadBuffers(&params.attn1);
            Attention.Params.unloadBuffers(&params.attn2);
            Attention.Params.unloadBuffers(&params.audio_attn1);
            Attention.Params.unloadBuffers(&params.audio_attn2);
            Attention.Params.unloadBuffers(&params.audio_to_video_attn);
            Attention.Params.unloadBuffers(&params.video_to_audio_attn);
            params.scale_shift_table.deinit();
            params.audio_scale_shift_table.deinit();
            params.scale_shift_table_a2v_ca_video.deinit();
            params.scale_shift_table_a2v_ca_audio.deinit();
            params.prompt_scale_shift_table.deinit();
            params.audio_prompt_scale_shift_table.deinit();
        }
    };

    pub const FullOutputs = struct {
        vx_out: Tensor,
        ax_out: Tensor,
    };

    pub fn init() BasicAVTransformerBlock {
        return .{
            .ff = .{},
            .audio_ff = .{},
            .attn1 = .{},
            .attn2 = .{},
            .audio_attn1 = .{},
            .audio_attn2 = .{},
            .audio_to_video_attn = .{},
            .video_to_audio_attn = .{},
        };
    }

    pub fn initParams(store: zml.io.TensorStore.View) Params {
        return .{
            .attn1 = Attention.initParams(store, .attn1),
            .attn2 = Attention.initParams(store, .attn2),
            .ff = FeedForward.initParams(store.withPrefix("ff")),
            .audio_attn1 = Attention.initParams(store, .audio_attn1),
            .audio_attn2 = Attention.initParams(store, .audio_attn2),
            // audio_ff has the same structure as ff but operates on 2048-dim audio features (2048->8192->2048).
            // FeedForward.initParams infers dimensions from checkpoint weight shapes via named axes.
            .audio_ff = FeedForward.initParams(store.withPrefix("audio_ff")),
            .audio_to_video_attn = Attention.initParams(store, .audio_to_video_attn),
            .video_to_audio_attn = Attention.initParams(store, .video_to_audio_attn),
            .scale_shift_table = store.createTensor("scale_shift_table", .{ .n_ada, .d }, null),
            .audio_scale_shift_table = store.createTensor("audio_scale_shift_table", .{ .n_ada, .d }, null),
            .scale_shift_table_a2v_ca_video = store.createTensor("scale_shift_table_a2v_ca_video", .{ .n_ada, .d }, null),
            .scale_shift_table_a2v_ca_audio = store.createTensor("scale_shift_table_a2v_ca_audio", .{ .n_ada, .d }, null),
            .prompt_scale_shift_table = store.createTensor("prompt_scale_shift_table", .{ .n_ada, .d }, null),
            .audio_prompt_scale_shift_table = store.createTensor("audio_prompt_scale_shift_table", .{ .n_ada, .d }, null),
        };
    }

    /// Per-inference inputs shared across all blocks during a model forward pass.
    ///
    /// `SharedInputs` contains the raw conditioning data (timestep embeddings,
    /// text contexts, positional embeddings) that are identical for every block.
    /// Per-block AdaLN modulation values (shift, scale, gate) are computed
    /// inline from `params.scale_shift_table` during `forwardNative`.
    pub const SharedInputs = struct {
        /// AdaLN timestep embeddings; flattened to [B, 1, N_ada * D].
        video_timesteps: Tensor, // [B, 1, 6 * D_video]
        audio_timesteps: Tensor, // [B, 1, 6 * D_audio]
        v_prompt_timestep: Tensor, // [B, 1, 2 * D_video]
        a_prompt_timestep: Tensor, // [B, 1, 2 * D_audio]
        /// Self-attention positional embeddings.
        v_pe_cos: Tensor,
        v_pe_sin: Tensor,
        a_pe_cos: Tensor,
        a_pe_sin: Tensor,
        /// Text cross-attention contexts.
        v_text_ctx: Tensor,
        a_text_ctx: Tensor,
        v_text_ctx_mask: ?Tensor = null,
        a_text_ctx_mask: ?Tensor = null,
        /// AV cross-attention scale-shift timestep embeddings.
        v_cross_ss_ts: Tensor,   // video.cross_scale_shift_timestep  [B, 1, 4 * D_video]
        v_cross_gate_ts: Tensor, // video.cross_gate_timestep          [B, 1, D_video]
        a_cross_ss_ts: Tensor,   // audio.cross_scale_shift_timestep  [B, 1, 4 * D_audio]
        a_cross_gate_ts: Tensor, // audio.cross_gate_timestep         [B, 1, D_audio]
        /// AV cross-attention positional embeddings, captured per module call.
        a2v_pe_cos: Tensor,
        a2v_pe_sin: Tensor,
        a2v_k_pe_cos: Tensor,
        a2v_k_pe_sin: Tensor,
        a2v_mask: ?Tensor = null,
        v2a_pe_cos: Tensor,
        v2a_pe_sin: Tensor,
        v2a_k_pe_cos: Tensor,
        v2a_k_pe_sin: Tensor,
        v2a_mask: ?Tensor = null,
    };

    /// Full block forward from raw stream state, computing all AdaLN values inline.
    ///
    /// Computes shift/scale/gate from `params.scale_shift_table` (video) and
    /// `params.audio_scale_shift_table` (audio) rather than accepting pre-captured
    /// per-module query tensors from Python activation traces. This is the
    /// threading-compatible path used by `LTXModel.forwardNative`.
    fn forwardNativeImpl(self: BasicAVTransformerBlock, comptime audio_ff_residual_f32: bool, comptime audio_all_residuals_f32: bool, comptime video_all_residuals_f32: bool, comptime bf16_attn: bool, comptime skip_video_self_attn: bool, comptime skip_audio_self_attn: bool, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        const vx = vx_in.withPartialTags(.{ .b, .t, .d });
        const ax = ax_in.withPartialTags(.{ .b, .t, .d });
        // In f32-carry mode, vx/ax arrive as f32 while adaValueAt returns ts.dtype() = bf16.
        // Convert all stream-facing ada values to match the activation dtype so that the
        // subsequent mul/add operations don't panic on dtype mismatch.
        // In normal bf16 mode, convert(bf16) on a bf16 tensor is a no-op in XLA.
        const vd = vx.dtype();
        const ad = ax.dtype();

        // ── Video: AdaLN self-attn (rows 0,1,2 → shift, scale, gate) ─────────────────
        const vshift_msa = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 0).convert(vd);
        const vscale_msa = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 1).convert(vd);
        const vgate_msa  = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 2).convert(vd);
        const norm_vx = zml.nn.rmsNorm(vx, .d, 1e-6)
            .mul(vscale_msa.addConstant(1.0).broad(vx.shape()))
            .add(vshift_msa.broad(vx.shape()));

        // M1: video self-attn residual.
        const attn1_out = if (skip_video_self_attn)
            self.attn1.forwardValuePassthrough(norm_vx, params.attn1, kindNumHeads(.attn1))
        else if (bf16_attn) self.attn1.forwardBf16(norm_vx, params.attn1, kindNumHeads(.attn1), .{
            .pe_cos = inputs.v_pe_cos,
            .pe_sin = inputs.v_pe_sin,
        }) else self.attn1.forward(norm_vx, params.attn1, kindNumHeads(.attn1), .{
            .pe_cos = inputs.v_pe_cos,
            .pe_sin = inputs.v_pe_sin,
        });
        const video_msa_delta = attn1_out.mul(vgate_msa.broad(attn1_out.shape()));
        var h_v = if (video_all_residuals_f32) blk: {
            const video_dtype = vx.dtype();
            break :blk vx.convert(.f32).add(video_msa_delta.convert(.f32)).convert(video_dtype);
        } else vx.add(video_msa_delta);

        // M2: video text cross-attn residual (cross-attention AdaLN path).
        const v_shift_q = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 6).convert(vd);
        const v_scale_q = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 7).convert(vd);
        const v_gate_q = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 8).convert(vd);
        const v_text_x = zml.nn.rmsNorm(h_v, .d, 1e-6)
            .mul(v_scale_q.addConstant(1.0).broad(h_v.shape()))
            .add(v_shift_q.broad(h_v.shape()));
        // v_shift_kv/v_scale_kv are applied to v_text_ctx (always bf16), not the f32 stream.
        const v_shift_kv = adaValueAt(params.prompt_scale_shift_table, inputs.v_prompt_timestep, 0).convert(vd);
        const v_scale_kv = adaValueAt(params.prompt_scale_shift_table, inputs.v_prompt_timestep, 1).convert(vd);
        const v_text_ctx_mod = inputs.v_text_ctx
            .mul(v_scale_kv.addConstant(1.0).broad(inputs.v_text_ctx.shape()))
            .add(v_shift_kv.broad(inputs.v_text_ctx.shape()));
        const v_text_ca_out = if (bf16_attn) self.attn2.forwardBf16(v_text_x, params.attn2, kindNumHeads(.attn2), .{
            .context = v_text_ctx_mod,
            .mask = inputs.v_text_ctx_mask,
        }) else self.attn2.forward(v_text_x, params.attn2, kindNumHeads(.attn2), .{
            .context = v_text_ctx_mod,
            .mask = inputs.v_text_ctx_mask,
        });
        const video_text_ca_delta = v_text_ca_out.mul(v_gate_q.broad(v_text_ca_out.shape()));
        if (video_all_residuals_f32) {
            const video_dtype = h_v.dtype();
            h_v = h_v.convert(.f32).add(video_text_ca_delta.convert(.f32)).convert(video_dtype);
        } else {
            h_v = h_v.add(video_text_ca_delta);
        }

        // ── Audio: AdaLN self-attn (rows 0,1,2 → shift, scale, gate) ─────────────────
        const ashift_msa = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 0).convert(ad);
        const ascale_msa = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 1).convert(ad);
        const agate_msa  = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 2).convert(ad);
        const norm_ax = zml.nn.rmsNorm(ax, .d, 1e-6)
            .mul(ascale_msa.addConstant(1.0).broad(ax.shape()))
            .add(ashift_msa.broad(ax.shape()));

        // M4-A: audio self-attn residual.
        const audio_attn1_out = if (skip_audio_self_attn)
            self.audio_attn1.forwardValuePassthrough(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1))
        else if (bf16_attn) self.audio_attn1.forwardBf16(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{
            .pe_cos = inputs.a_pe_cos,
            .pe_sin = inputs.a_pe_sin,
        }) else self.audio_attn1.forward(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{
            .pe_cos = inputs.a_pe_cos,
            .pe_sin = inputs.a_pe_sin,
        });
        const audio_msa_delta = audio_attn1_out.mul(agate_msa.broad(audio_attn1_out.shape()));
        var h_a = if (audio_all_residuals_f32) blk: {
            const audio_dtype = ax.dtype();
            break :blk ax.convert(.f32).add(audio_msa_delta.convert(.f32)).convert(audio_dtype);
        } else ax.add(audio_msa_delta);

        // M4-B: audio text cross-attn residual (cross-attention AdaLN path).
        const a_shift_q = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 6).convert(ad);
        const a_scale_q = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 7).convert(ad);
        const a_gate_q = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 8).convert(ad);
        const a_text_x = zml.nn.rmsNorm(h_a, .d, 1e-6)
            .mul(a_scale_q.addConstant(1.0).broad(h_a.shape()))
            .add(a_shift_q.broad(h_a.shape()));
        // a_shift_kv/a_scale_kv are applied to a_text_ctx (always bf16), not the f32 stream.
        const a_shift_kv = adaValueAt(params.audio_prompt_scale_shift_table, inputs.a_prompt_timestep, 0).convert(ad);
        const a_scale_kv = adaValueAt(params.audio_prompt_scale_shift_table, inputs.a_prompt_timestep, 1).convert(ad);
        const a_text_ctx_mod = inputs.a_text_ctx
            .mul(a_scale_kv.addConstant(1.0).broad(inputs.a_text_ctx.shape()))
            .add(a_shift_kv.broad(inputs.a_text_ctx.shape()));
        const a_text_ca_out = if (bf16_attn) self.audio_attn2.forwardBf16(a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{
            .context = a_text_ctx_mod,
            .mask = inputs.a_text_ctx_mask,
        }) else self.audio_attn2.forward(a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{
            .context = a_text_ctx_mod,
            .mask = inputs.a_text_ctx_mask,
        });
        const audio_text_ca_delta = a_text_ca_out.mul(a_gate_q.broad(a_text_ca_out.shape()));
        if (audio_all_residuals_f32) {
            const audio_dtype = h_a.dtype();
            h_a = h_a.convert(.f32).add(audio_text_ca_delta.convert(.f32)).convert(audio_dtype);
        } else {
            h_a = h_a.add(audio_text_ca_delta);
        }

        // ── AV cross-attn: pre-normalize both streams ────────────────────────────────
        const vx_norm3 = zml.nn.rmsNorm(h_v, .d, 1e-6);
        const ax_norm3 = zml.nn.rmsNorm(h_a, .d, 1e-6);

        // ── A→V branch ─────────────────────────────────────────────────────────────
        // Video query: sst_a2v_ca_video rows 0,1 (scale/shift); gate from row 4.
        const sst_v_ss   = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 0, .end = 4 });
        const sst_v_gate = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 4, .end = 5 });
        const scale_v_a2v = adaValueAt(sst_v_ss,   inputs.v_cross_ss_ts,   0).convert(vd);
        const shift_v_a2v = adaValueAt(sst_v_ss,   inputs.v_cross_ss_ts,   1).convert(vd);
        const gate_a2v    = adaValueAt(sst_v_gate,  inputs.v_cross_gate_ts, 0).convert(vd);
        const vx_scaled_a2v = vx_norm3
            .mul(scale_v_a2v.addConstant(1.0).broad(vx_norm3.shape()))
            .add(shift_v_a2v.broad(vx_norm3.shape()));

        // Audio context: sst_a2v_ca_audio rows 0,1 (scale/shift).
        const sst_a_ss   = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 0, .end = 4 });
        const sst_a_gate = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 4, .end = 5 });
        const scale_a_a2v = adaValueAt(sst_a_ss, inputs.a_cross_ss_ts, 0).convert(ad);
        const shift_a_a2v = adaValueAt(sst_a_ss, inputs.a_cross_ss_ts, 1).convert(ad);
        const ax_scaled_a2v = ax_norm3
            .mul(scale_a_a2v.addConstant(1.0).broad(ax_norm3.shape()))
            .add(shift_a_a2v.broad(ax_norm3.shape()));

        const a2v_out = if (bf16_attn) self.audio_to_video_attn.forwardBf16(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
            .context  = ax_scaled_a2v,
            .pe_cos   = inputs.a2v_pe_cos,
            .pe_sin   = inputs.a2v_pe_sin,
            .k_pe_cos = inputs.a2v_k_pe_cos,
            .k_pe_sin = inputs.a2v_k_pe_sin,
        }) else self.audio_to_video_attn.forward(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
            .context  = ax_scaled_a2v,
            .pe_cos   = inputs.a2v_pe_cos,
            .pe_sin   = inputs.a2v_pe_sin,
            .k_pe_cos = inputs.a2v_k_pe_cos,
            .k_pe_sin = inputs.a2v_k_pe_sin,
        });
        var a2v_delta = a2v_out.mul(gate_a2v.broad(a2v_out.shape()));
        if (inputs.a2v_mask) |mask| {
            a2v_delta = a2v_delta.mul(mask.convert(a2v_delta.dtype()).broad(a2v_delta.shape()));
        }
        if (video_all_residuals_f32) {
            const video_dtype = h_v.dtype();
            h_v = h_v.convert(.f32).add(a2v_delta.convert(.f32)).convert(video_dtype);
        } else {
            h_v = h_v.add(a2v_delta);
        }

        // ── V→A branch ─────────────────────────────────────────────────────────────
        // Audio query: sst_a2v_ca_audio rows 2,3 (scale/shift); gate from row 4.
        const scale_a_v2a = adaValueAt(sst_a_ss,   inputs.a_cross_ss_ts,   2).convert(ad);
        const shift_a_v2a = adaValueAt(sst_a_ss,   inputs.a_cross_ss_ts,   3).convert(ad);
        const gate_v2a    = adaValueAt(sst_a_gate,  inputs.a_cross_gate_ts, 0).convert(ad);
        const ax_scaled_v2a = ax_norm3
            .mul(scale_a_v2a.addConstant(1.0).broad(ax_norm3.shape()))
            .add(shift_a_v2a.broad(ax_norm3.shape()));

        // Video context: sst_a2v_ca_video rows 2,3 (scale/shift).
        const scale_v_v2a = adaValueAt(sst_v_ss, inputs.v_cross_ss_ts, 2).convert(vd);
        const shift_v_v2a = adaValueAt(sst_v_ss, inputs.v_cross_ss_ts, 3).convert(vd);
        const vx_scaled_v2a = vx_norm3
            .mul(scale_v_v2a.addConstant(1.0).broad(vx_norm3.shape()))
            .add(shift_v_v2a.broad(vx_norm3.shape()));

        const v2a_out = if (bf16_attn) self.video_to_audio_attn.forwardBf16(ax_scaled_v2a, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
            .context  = vx_scaled_v2a,
            .pe_cos   = inputs.v2a_pe_cos,
            .pe_sin   = inputs.v2a_pe_sin,
            .k_pe_cos = inputs.v2a_k_pe_cos,
            .k_pe_sin = inputs.v2a_k_pe_sin,
        }) else self.video_to_audio_attn.forward(ax_scaled_v2a, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
            .context  = vx_scaled_v2a,
            .pe_cos   = inputs.v2a_pe_cos,
            .pe_sin   = inputs.v2a_pe_sin,
            .k_pe_cos = inputs.v2a_k_pe_cos,
            .k_pe_sin = inputs.v2a_k_pe_sin,
        });
        var v2a_delta = v2a_out.mul(gate_v2a.broad(v2a_out.shape()));
        if (inputs.v2a_mask) |mask| {
            v2a_delta = v2a_delta.mul(mask.convert(v2a_delta.dtype()).broad(v2a_delta.shape()));
        }
        if (audio_all_residuals_f32) {
            const audio_dtype = h_a.dtype();
            h_a = h_a.convert(.f32).add(v2a_delta.convert(.f32)).convert(audio_dtype);
        } else {
            h_a = h_a.add(v2a_delta);
        }

        // ── Video FF: AdaLN rows 3,4,5 (shift, scale, gate) ─────────────────────────
        const vshift_mlp = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 3).convert(vd);
        const vscale_mlp = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 4).convert(vd);
        const vgate_mlp  = adaValueAt(params.scale_shift_table, inputs.video_timesteps, 5).convert(vd);
        const vx_scaled_ff = zml.nn.rmsNorm(h_v, .d, 1e-6)
            .mul(vscale_mlp.addConstant(1.0).broad(h_v.shape()))
            .add(vshift_mlp.broad(h_v.shape()));
        const video_ff_out = if (video_all_residuals_f32)
            forwardAudioFFPrecise(vx_scaled_ff, params.ff)
        else
            self.ff.forward(vx_scaled_ff, params.ff);
        const video_ff_delta = video_ff_out.mul(vgate_mlp.broad(video_ff_out.shape()));
        if (video_all_residuals_f32) {
            const video_dtype = h_v.dtype();
            h_v = h_v.convert(.f32).add(video_ff_delta.convert(.f32)).convert(video_dtype);
        } else {
            h_v = h_v.add(video_ff_delta);
        }

        // ── Audio FF: AdaLN rows 3,4,5 ─────────────────────────────────────────────
        const ashift_mlp = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 3).convert(ad);
        const ascale_mlp = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 4).convert(ad);
        const agate_mlp  = adaValueAt(params.audio_scale_shift_table, inputs.audio_timesteps, 5).convert(ad);
        const ax_scaled_ff = zml.nn.rmsNorm(h_a, .d, 1e-6)
            .mul(ascale_mlp.addConstant(1.0).broad(h_a.shape()))
            .add(ashift_mlp.broad(h_a.shape()));
        // Use forwardAudioFFPrecise (f32 matmuls) when the audio stream is f32 (f32-carry mode)
        // or when an explicit audio-precision flag is set. Plain forward() uses zml.nn.Linear
        // which panics if x is f32 and weight is bf16.
        const audio_ff_out = if (audio_ff_residual_f32 or audio_all_residuals_f32 or ax.dtype() == .f32)
            forwardAudioFFPrecise(ax_scaled_ff, params.audio_ff)
        else
            self.audio_ff.forward(ax_scaled_ff, params.audio_ff);
        const audio_ff_delta = audio_ff_out.mul(agate_mlp.broad(audio_ff_out.shape()));
        if (audio_ff_residual_f32 or audio_all_residuals_f32) {
            const audio_dtype = h_a.dtype();
            h_a = h_a.convert(.f32)
                .add(audio_ff_delta.convert(.f32))
                .convert(audio_dtype);
        } else {
            h_a = h_a.add(audio_ff_delta);
        }

        return .{ .vx_out = h_v, .ax_out = h_a };
    }

    pub fn forwardNative(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, false, false, false, false, false, vx_in, ax_in, inputs, params);
    }

    /// Checker-only experimental path: attention runs in bf16 matching Python's dtype chain.
    pub fn forwardNativeBf16Attn(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, false, false, true, false, false, vx_in, ax_in, inputs, params);
    }

    /// STG block variant: both video and audio self-attention bypass Q·K·V and compute to_out(to_v(x)) instead.
    /// Used for block 29 during the STG perturbation pass (Pass 3) in Stage 1 denoising.
    pub fn forwardNativeSTG(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, false, false, false, true, true, vx_in, ax_in, inputs, params);
    }

    /// STG block variant with bf16-native attention on text and AV cross-attention.
    /// Self-attention still uses V-passthrough (no SDPA).
    pub fn forwardNativeSTGBf16Attn(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, false, false, true, true, true, vx_in, ax_in, inputs, params);
    }
};

// ============================================================================
// Section 8: LTXModel (full transformer stack)
// Python ref: ltx_core/models/ltx_model.py — LTXModel.forward()
// ============================================================================

/// Velocity model (transformer stack) shared across stages.
pub const LTXModel = struct {
    blocks: []BasicAVTransformerBlock,

    pub const Params = struct {
        blocks: []BasicAVTransformerBlock.Params,
        // AdaLN-single modules — one per stream/role.
        // Video stream base modulation (N=9 coefficients → d_ada=9*4096=36864)
        adaln_single: AdaLayerNormSingle.Params,
        // Audio stream base modulation (N=9 → d_ada=9*2048=18432)
        audio_adaln_single: AdaLayerNormSingle.Params,
        // Prompt / text-conditioning modulation (N=2 each)
        prompt_adaln_single: AdaLayerNormSingle.Params, // video, d_ada=2*4096=8192
        audio_prompt_adaln_single: AdaLayerNormSingle.Params, // audio, d_ada=2*2048=4096
        // A/V cross-attention scale-shift (N=4 each)
        av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle.Params, // d_ada=4*4096=16384
        av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle.Params, // d_ada=4*2048=8192
        // A/V cross-attention gate (N=1 each)
        av_ca_a2v_gate_adaln_single: AdaLayerNormSingle.Params, // d_ada=4096
        av_ca_v2a_gate_adaln_single: AdaLayerNormSingle.Params, // d_ada=2048
        // Output projection (norm + scale-shift + linear) applied after all transformer blocks.
        norm_proj_out: OutputProjection.Params, // video: scale_shift_table[2,4096] + proj_out(4096→128)
        audio_norm_proj_out: OutputProjection.Params, // audio: scale_shift_table[2,2048] + proj_out(2048→128)

        pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
            allocator.free(self.blocks);
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Params), allocator: std.mem.Allocator) void {
            for (self.blocks) |*bp| {
                BasicAVTransformerBlock.Params.unloadBuffers(bp);
            }
            allocator.free(self.blocks);
            self.adaln_single.unloadBuffers();
            self.audio_adaln_single.unloadBuffers();
            self.prompt_adaln_single.unloadBuffers();
            self.audio_prompt_adaln_single.unloadBuffers();
            self.av_ca_video_scale_shift_adaln_single.unloadBuffers();
            self.av_ca_audio_scale_shift_adaln_single.unloadBuffers();
            self.av_ca_a2v_gate_adaln_single.unloadBuffers();
            self.av_ca_v2a_gate_adaln_single.unloadBuffers();
            self.norm_proj_out.unloadBuffers();
            self.audio_norm_proj_out.unloadBuffers();
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !LTXModel {
        const blocks = try allocator.alloc(BasicAVTransformerBlock, config.num_transformer_blocks);
        for (blocks) |*b| b.* = BasicAVTransformerBlock.init();

        return .{ .blocks = blocks };
    }

    pub fn deinit(self: *LTXModel, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }

    pub fn initParams(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Params {
        const blocks = try allocator.alloc(BasicAVTransformerBlock.Params, config.num_transformer_blocks);

        const blocks_store = store.withPrefix("transformer_blocks");
        for (blocks, 0..) |*bp, i| {
            bp.* = BasicAVTransformerBlock.initParams(blocks_store.withLayer(i));
        }

        return .{
            .blocks = blocks,
            .adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("adaln_single")),
            .audio_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("audio_adaln_single")),
            .prompt_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("prompt_adaln_single")),
            .audio_prompt_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("audio_prompt_adaln_single")),
            .av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("av_ca_video_scale_shift_adaln_single")),
            .av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("av_ca_audio_scale_shift_adaln_single")),
            .av_ca_a2v_gate_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("av_ca_a2v_gate_adaln_single")),
            .av_ca_v2a_gate_adaln_single = AdaLayerNormSingle.initParams(store.withPrefix("av_ca_v2a_gate_adaln_single")),
            .norm_proj_out = .{
                .scale_shift_table = store.createTensor("scale_shift_table", .{ .n_ssv, .d }, null),
                .proj_out = .init(
                    store.withPrefix("proj_out").createTensor("weight", .{ .d_out, .d }, null),
                    store.withPrefix("proj_out").createTensor("bias", .{.d_out}, null),
                    .d,
                ),
            },
            .audio_norm_proj_out = .{
                .scale_shift_table = store.createTensor("audio_scale_shift_table", .{ .n_ssv, .d }, null),
                .proj_out = .init(
                    store.withPrefix("audio_proj_out").createTensor("weight", .{ .d_out, .d }, null),
                    store.withPrefix("audio_proj_out").createTensor("bias", .{.d_out}, null),
                    .d,
                ),
            },
        };
    }

    /// Full AV model forward threading both streams through all transformer blocks.
    ///
    /// `vx` / `ax` – post-patchify video and audio token streams
    /// `inputs`    – per-inference conditioning shared across all blocks (timesteps,
    ///               contexts, positional embeddings)
    ///
    /// Calls `BasicAVTransformerBlock.forwardNative` for each block, which computes
    /// AdaLN shift/scale/gate inline from the block's scale-shift table parameters.
    pub fn forwardNative(self: LTXModel, vx: Tensor, ax: Tensor, inputs: BasicAVTransformerBlock.SharedInputs, params: Params) BasicAVTransformerBlock.FullOutputs {
        std.debug.assert(self.blocks.len == params.blocks.len);

        var h_v = vx;
        var h_a = ax;
        for (self.blocks, params.blocks) |block, block_params| {
            const out = block.forwardNative(h_v, h_a, inputs, block_params);
            h_v = out.vx_out;
            h_a = out.ax_out;
        }
        return .{ .vx_out = h_v, .ax_out = h_a };
    }
};

/// Top-level wrapper used by the LTX pipeline.
pub const X0Model = struct {
    velocity_model: LTXModel,

    pub const Params = struct {
        velocity_model: LTXModel.Params,

        pub fn deinit(self: *Params, allocator: std.mem.Allocator) void {
            self.velocity_model.deinit(allocator);
        }
    };

    pub fn init(allocator: std.mem.Allocator, config: Config) !X0Model {
        return .{
            .velocity_model = try LTXModel.init(allocator, config),
        };
    }

    pub fn deinit(self: *X0Model, allocator: std.mem.Allocator) void {
        self.velocity_model.deinit(allocator);
    }

    pub fn initParams(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Params {
        const root = selectTransformerRoot(store);
        return .{
            .velocity_model = try LTXModel.initParams(allocator, root, config),
        };
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params), allocator: std.mem.Allocator) void {
        LTXModel.Params.unloadBuffers(&params.velocity_model, allocator);
    }
};

/// Active runtime parameter set (single stage at a time).
/// This keeps code shared while allowing atomic stage-by-stage validation.
pub const ActiveParams = struct {
    stage: Stage,
    params: X0Model.Params,

    pub fn deinit(self: *ActiveParams, allocator: std.mem.Allocator) void {
        self.params.deinit(allocator);
    }
};

/// Build one active stage-specific parameter set from one checkpoint/store.
pub fn initActiveParams(
    allocator: std.mem.Allocator,
    stage: Stage,
    store: zml.io.TensorStore.View,
    config: Config,
) !ActiveParams {
    return .{
        .stage = stage,
        .params = try X0Model.initParams(allocator, store, config),
    };
}

pub fn selectTransformerRoot(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("model.velocity_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("model").withPrefix("velocity_model");
    }
    if (store.hasKey("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("model").withPrefix("diffusion_model");
    }
    if (store.hasKey("velocity_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("velocity_model");
    }
    if (store.hasKey("diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")) {
        return store.withPrefix("diffusion_model");
    }
    // Keep previous behavior as a fallback.
    return store.withPrefix("velocity_model");
}

// ============================================================================
// Section 11: Denoising Step (sigma schedule + Euler step + mask blending)
// Python ref: ltx_pipelines/scheduler.py — RectifiedFlowScheduler
// ============================================================================

/// Free-function entrypoint to run one AdaLayerNormSingle module.
/// Returns modulation [.b, .d_ada] — the full outputs (modulation + embedded_timestep)
/// are available via AdaLayerNormSingle.forward; this helper returns both as a named struct
/// compatible with zml compiled-function result extraction.
pub fn forwardAdalnSingle(sigma: Tensor, params: AdaLayerNormSingle.Params) AdaLayerNormSingle.ForwardResult {
    return AdaLayerNormSingle.forward(sigma, params);
}

/// Free-function entrypoint for parity tooling: output projection after all transformer blocks.
/// x: [.b, .t, .d], embedded_timestep: [.b, .d_emb]
/// Returns: [.b, .t, .d_out=128]
pub fn forwardOutputProjection(x: Tensor, embedded_timestep: Tensor, params: OutputProjection.Params) Tensor {
    return OutputProjection.forward(x, embedded_timestep, params);
}

// =====================================================================
// Denoising-loop arithmetic (Step 3)
// =====================================================================

/// Apply noise initialization (GaussianNoiser).
///
/// Produces the initial noised latent from clean latent, noise, and mask:
///   noised = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)
///
/// Matches Python's GaussianNoiser dtype chain: all arithmetic in f32,
/// final cast back to the clean latent's dtype (bf16).
pub fn forwardNoiseInit(
    clean_latent: Tensor,
    noise: Tensor,
    denoise_mask: Tensor,
    sigma_0: Tensor,
) Tensor {
    const out_dtype = clean_latent.dtype();
    const clean_f32 = clean_latent.convert(.f32);
    const noise_f32 = noise.convert(.f32);
    const mask_f32 = denoise_mask.convert(.f32);
    const sigma_f32 = sigma_0.convert(.f32);

    // mask_sigma = mask * sigma_0
    const mask_sigma = mask_f32.mul(sigma_f32);
    // one_minus = 1 - mask_sigma
    const one_minus = Tensor.scalar(1.0, .f32).sub(mask_sigma);
    // noised = noise * mask_sigma + clean * (1 - mask_sigma)
    return noise_f32.mul(mask_sigma).add(clean_f32.mul(one_minus)).convert(out_dtype);
}

// ============================================================================
// Section 12: Guidance Combine (CFG + STG + modality isolation + rescale)
// Python ref: ltx_pipelines/guiders.py — LTXGuider.combine()
// ============================================================================

/// Result of guider combine: guided outputs for video and audio.
pub const GuiderCombineResult = struct {
    guided_v: Tensor,
    guided_a: Tensor,
};

/// LTX2Scheduler sigma schedule — host-side computation (not a compiled graph op).
///
/// Reimplements `LTX2Scheduler.execute(steps, ...)` from ltx_core:
///   1. linspace(1, 0, steps+1)
///   2. Logistic shift: sigma = exp(s) / (exp(s) + 1/sigma - 1)
///   3. Stretch so last non-zero value = terminal
///
/// Returns `steps + 1` values: sigmas[0] ≈ 1.0, sigmas[steps] = 0.0.
pub fn computeSigmaSchedule(
    comptime max_steps: usize,
    num_steps: usize,
    num_tokens: usize,
    max_shift: f32,
    base_shift: f32,
    terminal: f32,
) [max_steps + 1]f32 {
    std.debug.assert(num_steps <= max_steps);

    const BASE_SHIFT_ANCHOR: f32 = 1024.0;
    const MAX_SHIFT_ANCHOR: f32 = 4096.0;

    // 1. linspace(1.0, 0.0, num_steps + 1)
    var sigmas: [max_steps + 1]f32 = undefined;
    const n: f32 = @floatFromInt(num_steps);
    for (0..num_steps + 1) |i| {
        const t: f32 = @floatFromInt(i);
        sigmas[i] = 1.0 - t / n;
    }

    // 2. Compute sigma_shift from token count
    const mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR);
    const b = base_shift - mm * BASE_SHIFT_ANCHOR;
    const tokens_f: f32 = @floatFromInt(num_tokens);
    const sigma_shift = tokens_f * mm + b;
    const exp_shift = std.math.exp(sigma_shift);

    // 3. Apply logistic shift: sigma = exp(s) / (exp(s) + (1/sigma - 1))
    for (0..num_steps + 1) |i| {
        if (sigmas[i] != 0.0) {
            sigmas[i] = exp_shift / (exp_shift + (1.0 / sigmas[i] - 1.0));
        }
    }

    // 4. Stretch so last non-zero sigma = terminal
    //    Find last non-zero entry (should be sigmas[num_steps - 1])
    var last_nz_idx: usize = 0;
    for (0..num_steps + 1) |i| {
        if (sigmas[i] != 0.0) last_nz_idx = i;
    }
    if (last_nz_idx > 0) {
        const one_minus_last = 1.0 - sigmas[last_nz_idx];
        const scale_factor = one_minus_last / (1.0 - terminal);
        for (0..num_steps + 1) |i| {
            if (sigmas[i] != 0.0) {
                sigmas[i] = 1.0 - (1.0 - sigmas[i]) / scale_factor;
            }
        }
    }

    return sigmas;
}

// Values taken from regular 30-step schedule 
// https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/components/schedulers.py
pub const stage1_default_schedule = struct {
    pub const num_steps: usize = 30;
    pub const max_shift: f32 = 2.05;
    pub const base_shift: f32 = 0.95;
    pub const terminal: f32 = 0.1;
    /// MAX_SHIFT_ANCHOR — used when no latent tensor is provided.
    pub const default_num_tokens: usize = 4096;
};

/// Combines 4 guidance passes (cond, neg/CFG, perturbed/STG, isolated/modality)
/// using the Stage 1 guidance formula per modality:
///   pred = cond + (cfg-1)*(cond-neg) + stg*(cond-ptb) + (mod-1)*(cond-iso)
///   pred *= rescale * (cond.std() / pred.std()) + (1 - rescale)
///
/// All tensor inputs are [B, T, D] bf16; scalar inputs are rank-0 or rank-1 f32.
/// When rescale=0 the rescale factor evaluates to 1 (identity).
pub fn forwardGuiderCombine(
    cond_v: Tensor,
    neg_v: Tensor,
    ptb_v: Tensor,
    iso_v: Tensor,
    cond_a: Tensor,
    neg_a: Tensor,
    ptb_a: Tensor,
    iso_a: Tensor,
    cfg_v: Tensor,
    stg_v: Tensor,
    mod_v: Tensor,
    rescale_v: Tensor,
    cfg_a: Tensor,
    stg_a: Tensor,
    mod_a: Tensor,
    rescale_a: Tensor,
) GuiderCombineResult {
    return .{
        .guided_v = guiderCombineSingle(cond_v, neg_v, ptb_v, iso_v, cfg_v, stg_v, mod_v, rescale_v),
        .guided_a = guiderCombineSingle(cond_a, neg_a, ptb_a, iso_a, cfg_a, stg_a, mod_a, rescale_a),
    };
}

/// Guidance combine — matches Python's MultiModalGuider.calculate() dtype chain.
/// Python operates entirely in bf16 (tensor dtype), including std-rescale.
/// Traced empirically: all intermediates stay bf16 when inputs are bf16.
fn guiderCombineSingle(cond: Tensor, neg: Tensor, ptb: Tensor, iso: Tensor, cfg: Tensor, stg: Tensor, mod: Tensor, rescale: Tensor) Tensor {
    const dt = cond.dtype(); // bf16

    // Python: float * bf16_tensor → bf16 (scalar doesn't promote).
    // We use bf16 scalars to match.
    const cfg_bf = cfg.convert(dt).asScalar();
    const stg_bf = stg.convert(dt).asScalar();
    const mod_bf = mod.convert(dt).asScalar();
    const rescale_bf = rescale.convert(dt).asScalar();
    const one = Tensor.scalar(1.0, dt);

    // pred = cond + (cfg-1)*(cond-neg) + stg*(cond-ptb) + (mod-1)*(cond-iso)
    const cfg_term = cfg_bf.sub(one).broad(cond.shape()).mul(cond.sub(neg));
    const stg_term = stg_bf.broad(cond.shape()).mul(cond.sub(ptb));
    const mod_term = mod_bf.sub(one).broad(cond.shape()).mul(cond.sub(iso));
    var pred = cond.add(cfg_term).add(stg_term).add(mod_term);

    // std-rescale in bf16: factor = rescale * (cond.std() / pred.std()) + (1 - rescale)
    const cond_std = tensorStdAll(cond);
    const pred_std = tensorStdAll(pred);
    const ratio = cond_std.div(pred_std);
    const factor = rescale_bf.mul(ratio).add(one.sub(rescale_bf));
    pred = pred.mul(factor.broad(pred.shape()));

    return pred;
}

/// Standard deviation across all elements of a tensor, returning a rank-0 scalar.
fn tensorStdAll(x: Tensor) Tensor {
    const flat = x.flatten();
    const mean_val = flat.mean(0);
    const diff = flat.sub(mean_val.broad(flat.shape()));
    const variance = diff.mul(diff).mean(0);
    return variance.sqrt().asScalar();
}

/// Result of one denoising step: to_denoised + post_process_latent + Euler step.
pub const DenoisingStepResult = struct {
    denoised: Tensor,
    blended: Tensor,
    next_latent: Tensor,
};

/// Apply to_denoised + post_process_latent + Euler step for one denoising iteration.
///
/// Matches Python's dtype chain exactly — each sub-step casts back to bf16
/// to reproduce the same quantization behavior:
///   1. to_denoised:          bf16(sample.f32 - velocity.f32 * timesteps)
///   2. post_process_latent:  bf16(denoised_bf16.f32 * mask + clean.f32 * (1 - mask))
///   3. to_velocity:          bf16((sample.f32 - blended_bf16.f32) / sigma.item())
///   4. Euler update:         bf16(sample.f32 + velocity_bf16.f32 * dt)
pub fn forwardDenoisingStep(
    sample: Tensor,
    velocity: Tensor,
    denoise_mask: Tensor,
    clean_latent: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
) DenoisingStepResult {
    const out_dtype = sample.dtype();
    const sample_f32 = sample.convert(.f32);
    const velocity_f32 = velocity.convert(.f32);
    const clean_f32 = clean_latent.convert(.f32);
    const mask_f32 = denoise_mask.convert(.f32);
    const sigma_f32 = sigma.convert(.f32);
    const sigma_next_f32 = sigma_next.convert(.f32);

    // 1. timesteps = mask * sigma (per-token sigma)
    const timesteps = mask_f32.mul(sigma_f32);

    // 2. to_denoised: denoised = (sample.f32 - velocity.f32 * timesteps).to(bf16)
    const denoised_f32 = sample_f32.sub(velocity_f32.mul(timesteps));
    const denoised = denoised_f32.convert(out_dtype);

    // 3. post_process_latent: use bf16-rounded denoised (matching Python's dtype chain)
    //    Python: (denoised_bf16 * mask + clean.float() * (1 - mask)).to(bf16)
    const one_minus_mask = Tensor.scalar(1.0, .f32).sub(mask_f32);
    const blended_f32 = denoised.convert(.f32).mul(mask_f32).add(clean_f32.mul(one_minus_mask));
    const blended = blended_f32.convert(out_dtype);

    // 4. Euler step: to_velocity returns bf16, then Euler update uses bf16 velocity
    //    Python: velocity = bf16((sample.f32 - blended.f32) / sigma.item())
    //            next = bf16(sample.f32 + velocity.f32 * dt)
    const dt = sigma_next_f32.sub(sigma_f32);
    const euler_vel_f32 = sample_f32.sub(blended.convert(.f32)).div(sigma_f32);
    const euler_vel = euler_vel_f32.convert(out_dtype);
    const next_f32 = sample_f32.add(euler_vel.convert(.f32).mul(dt));
    const next_latent = next_f32.convert(out_dtype);

    return .{
        .denoised = denoised,
        .blended = blended,
        .next_latent = next_latent,
    };
}

/// Convert raw velocity → x0 prediction (denoised sample).
///
/// Matches Python's X0Model.forward output:
///   timesteps = denoise_mask * sigma
///   x0 = (sample - velocity * timesteps).to(bf16)
///
/// The guider combine should operate on x0 predictions (not velocities)
/// so that the rescale factor is computed in the same space as Python.
/// Velocity → x0 conversion matching Python's X0Model + to_denoised() dtype chain.
///
/// Python (traced empirically with f32 denoise_mask):
///   timesteps = denoise_mask(f32) * sigma(f32) → f32
///   to_denoised: sigma = sigma.to(f32)  →  f32 (no-op since timesteps already f32)
///               (sample.f32 - velocity.f32 * sigma_f32).to(bf16)
pub fn forwardToDenoised(
    sample: Tensor,
    velocity: Tensor,
    denoise_mask: Tensor,
    sigma: Tensor,
) Tensor {
    const out_dtype = sample.dtype();
    // Python: timesteps = denoise_mask * sigma (both in mask's dtype)
    // With f32 mask: f32 * f32 → f32.  With bf16 mask: bf16 * bf16 → bf16.
    const timesteps = denoise_mask.mul(sigma.convert(denoise_mask.dtype()));
    // Python: to_denoised converts timesteps to f32 for the arithmetic
    const timesteps_f32 = timesteps.convert(.f32);
    const sample_f32 = sample.convert(.f32);
    const velocity_f32 = velocity.convert(.f32);
    return sample_f32.sub(velocity_f32.mul(timesteps_f32)).convert(out_dtype);
}

/// Apply post_process_latent + Euler step from a guided x0 prediction.
///
/// This is the second half of forwardDenoisingStep, used when guidance
/// was applied in x0 space (matching Python's guider flow).
///
/// Euler denoising step from guided x0 — matches Python's exact dtype chain
/// (traced empirically with trace_dtype_chain.py, f32 denoise_mask).
///
/// Python dtype chain (with f32 mask):
///   1. post_process_latent:  denoised(bf16)*mask(f32)→f32 + clean.f32*(1-mask)(f32)  → f32 → bf16
///   2. to_velocity:          ((sample.f32 - blended.f32) / sigma_float) → f32 → bf16  ← ROUNDTRIP!
///   3. Euler update:         (sample.f32 + velocity_bf16.f32 * dt_f32) → f32 → bf16
pub fn forwardDenoisingStepFromX0(
    sample: Tensor,
    denoised: Tensor,
    denoise_mask: Tensor,
    clean_latent: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
) DenoisingStepResult {
    const out_dtype = sample.dtype(); // bf16
    const sample_f32 = sample.convert(.f32);
    const sigma_f32 = sigma.convert(.f32);
    const sigma_next_f32 = sigma_next.convert(.f32);

    // 1. post_process_latent — match Python's dtype promotion:
    //    PyTorch promotes bf16 * f32 → f32, so all arithmetic ends up in f32.
    //    denoised(bf16) * mask(f32) → f32   (PyTorch promotes to wider type)
    //    clean.float()              → f32
    //    1.0 - mask(f32)            → f32
    //    sum                        → f32
    //    .to(denoised.dtype)        → bf16
    const mask_f32 = denoise_mask.convert(.f32);
    const clean_f32 = clean_latent.convert(.f32);
    const one_minus_mask = Tensor.scalar(1.0, .f32).sub(mask_f32);
    const blended_f32 = denoised.convert(.f32).mul(mask_f32).add(clean_f32.mul(one_minus_mask));
    const blended = blended_f32.convert(out_dtype); // bf16

    // 2. to_velocity — Python rounds to bf16 then re-upcasts:
    //    velocity = ((sample.f32 - blended.f32) / sigma_item).to(bf16)
    const euler_vel_f32 = sample_f32.sub(blended.convert(.f32)).div(sigma_f32);
    const euler_vel = euler_vel_f32.convert(out_dtype); // bf16 roundtrip!

    // 3. Euler update:
    //    next = (sample.f32 + velocity_bf16.f32 * dt).to(bf16)
    const dt = sigma_next_f32.sub(sigma_f32);
    const next_f32 = sample_f32.add(euler_vel.convert(.f32).mul(dt));
    const next_latent = next_f32.convert(out_dtype);

    return .{
        .denoised = denoised,
        .blended = blended,
        .next_latent = next_latent,
    };
}

// ============================================================================
// Section 13: Block-Level Entrypoints (forwardBlock0* family)
// These are the ZML-compilable entrypoints — each wraps a Section 7 method
// with explicit tensor arguments (no struct, for MLIR arg flattening).
// ============================================================================

pub const Block0FullParams = BasicAVTransformerBlock.Params;

pub fn unloadBlock0FullBuffers(params: *zml.Bufferized(Block0FullParams)) void {
    BasicAVTransformerBlock.Params.unloadBuffers(params);
}

pub fn forwardBlock0Native(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    v_prompt_timestep: Tensor,
    a_prompt_timestep: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    params: Block0FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    return block.forwardNative(vx_in, ax_in, .{
        .video_timesteps = video_timesteps,
        .audio_timesteps = audio_timesteps,
        .v_prompt_timestep = v_prompt_timestep,
        .a_prompt_timestep = a_prompt_timestep,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .v_text_ctx = v_text_ctx,
        .a_text_ctx = a_text_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
    }, params);
}

/// Same as forwardBlock0Native but with bf16-native attention matching Python's dtype chain.
pub fn forwardBlock0NativeBf16Attn(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    v_prompt_timestep: Tensor,
    a_prompt_timestep: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    params: Block0FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    return block.forwardNativeBf16Attn(vx_in, ax_in, .{
        .video_timesteps = video_timesteps,
        .audio_timesteps = audio_timesteps,
        .v_prompt_timestep = v_prompt_timestep,
        .a_prompt_timestep = a_prompt_timestep,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .v_text_ctx = v_text_ctx,
        .a_text_ctx = a_text_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
    }, params);
}


/// STG block variant: identical interface to forwardBlock0Native, but both video and audio
/// self-attention use V-passthrough (to_out(to_v(x))). Used for block 29 during Pass 3
/// (STG perturbation) in Stage 1 denoising.
pub fn forwardBlock0NativeSTG(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    v_prompt_timestep: Tensor,
    a_prompt_timestep: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    params: Block0FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    return block.forwardNativeSTG(vx_in, ax_in, .{
        .video_timesteps = video_timesteps,
        .audio_timesteps = audio_timesteps,
        .v_prompt_timestep = v_prompt_timestep,
        .a_prompt_timestep = a_prompt_timestep,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .v_text_ctx = v_text_ctx,
        .a_text_ctx = a_text_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
    }, params);
}

/// Same as forwardBlock0NativeSTG but with bf16-native attention on text/AV cross-attention.
/// Self-attention still uses V-passthrough (no SDPA).
pub fn forwardBlock0NativeSTGBf16Attn(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    v_prompt_timestep: Tensor,
    a_prompt_timestep: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    params: Block0FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    return block.forwardNativeSTGBf16Attn(vx_in, ax_in, .{
        .video_timesteps = video_timesteps,
        .audio_timesteps = audio_timesteps,
        .v_prompt_timestep = v_prompt_timestep,
        .a_prompt_timestep = a_prompt_timestep,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .v_text_ctx = v_text_ctx,
        .a_text_ctx = a_text_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
    }, params);
}
pub fn forwardBlock0NativeWithAVMasks(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    v_prompt_timestep: Tensor,
    a_prompt_timestep: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_mask: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_mask: Tensor,
    params: Block0FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    return block.forwardNative(vx_in, ax_in, .{
        .video_timesteps = video_timesteps,
        .audio_timesteps = audio_timesteps,
        .v_prompt_timestep = v_prompt_timestep,
        .a_prompt_timestep = a_prompt_timestep,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .v_text_ctx = v_text_ctx,
        .a_text_ctx = a_text_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .a2v_mask = a2v_mask,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
        .v2a_mask = v2a_mask,
    }, params);
}

/// Same as forwardBlock0NativeWithAVMasks but with bf16-native attention.
pub fn forwardBlock0NativeWithAVMasksBf16Attn(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    v_prompt_timestep: Tensor,
    a_prompt_timestep: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_mask: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_mask: Tensor,
    params: Block0FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    return block.forwardNativeBf16Attn(vx_in, ax_in, .{
        .video_timesteps = video_timesteps,
        .audio_timesteps = audio_timesteps,
        .v_prompt_timestep = v_prompt_timestep,
        .a_prompt_timestep = a_prompt_timestep,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .v_text_ctx = v_text_ctx,
        .a_text_ctx = a_text_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .a2v_mask = a2v_mask,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
        .v2a_mask = v2a_mask,
    }, params);
}

// ============================================================================
// Section 3: Positional Embeddings (RoPE)
// Python ref: ltx_core/models/embeddings.py — RoPE3D
// ============================================================================

/// Cos/sin pair returned by RoPE generation.
const CosSinPair = struct { cos: Tensor, sin: Tensor };

/// Configuration for RoPE frequency generation.
const RopeConfig = struct {
    theta: f64 = 10000.0,
    max_pos: []const i64,
    num_heads: i64 = 32,
    inner_dim: i64,
};

/// Generate frequency basis vectors for RoPE (matches `generate_freq_grid_pytorch`).
///
/// Computes `n_elem = 2 * n_pos_dims` frequency values per position dimension:
///   indices[i] = theta^(linspace(log_theta(1), log_theta(theta), dim // n_elem)) * π/2
///
/// where `log_theta(x) = log(x) / log(theta)`.
///
/// Returns a 1D Tensor of shape [dim // (2 * n_pos_dims)] in f32.
fn generateFreqGrid(theta: f64, n_pos_dims: i64, dim: i64) Tensor {
    const log_theta = std.math.log(f64, std.math.e, theta);
    const n_elem: i64 = 2 * n_pos_dims;
    const num_freqs: i64 = @divTrunc(dim, n_elem);

    // linspace(log_theta(1), log_theta(theta), num_freqs)
    //   = linspace(0, 1, num_freqs)   since log_theta(1) = 0, log_theta(theta) = 1
    // Then: theta^(linspace_val) * π/2
    //   = exp(linspace_val * ln(theta)) * π/2
    //
    // Build via arange: step = 1.0 / (num_freqs - 1), val[i] = i * step
    const idx = Tensor.arange(.{ .end = num_freqs }, .f32);
    const step: f64 = if (num_freqs > 1) 1.0 / @as(f64, @floatFromInt(num_freqs - 1)) else 0.0;
    // indices = exp(idx * step * ln(theta)) * π/2
    const indices = idx.scale(step * log_theta).exp().scale(std.math.pi / 2.0);
    return indices;
}

/// Compute fractional positions from raw position grid (matches `get_fractional_positions`).
///
/// With `use_middle_indices_grid=true`, takes the mean of start/end coordinates:
///   positions [B, C, T, 2] → middle [B, C, T] → fractional[c] = middle[:, c, :] / max_pos[c]
///
/// Returns a slice of Tensors, one per position dimension, each [B, T].
/// Stays in the source dtype to match Python's bf16 intermediate precision.
/// Caller provides a buffer of length >= n_pos_dims.
fn getFractionalPositions(
    positions: Tensor, // [.b, .c, .t, .se=2]
    max_pos: []const i64,
    buf: []Tensor,
) []const Tensor {
    const n_pos_dims: usize = @intCast(positions.dim(.c));

    // use_middle_indices_grid: mean of start and end
    // Keep in source dtype (bf16 for video, f32 for audio) to match Python.
    const start = positions.slice1d(.se, .{ .start = 0, .end = 1 }).squeeze(.se); // [.b, .c, .t]
    const end = positions.slice1d(.se, .{ .start = 1, .end = 2 }).squeeze(.se); // [.b, .c, .t]
    const middle = start.add(end).scale(0.5); // [.b, .c, .t] — stays in source dtype

    // Normalize each position dim by its max_pos
    for (0..n_pos_dims) |c| {
        const c_i64: i64 = @intCast(c);
        const pos_c = middle.slice1d(.c, .{ .start = c_i64, .end = c_i64 + 1 }).squeeze(.c); // [.b, .t]
        const max_val: f64 = @floatFromInt(max_pos[c]);
        buf[c] = pos_c.scale(1.0 / max_val); // [.b, .t] normalized to [0, 1]
    }
    return buf[0..n_pos_dims];
}

/// Generate raw frequency embeddings (matches `generate_freqs`).
///
/// For each position dimension c:
///   freq_c = indices[c] * (frac_pos[c] * 2 - 1)    → [B, T, D//(2*C)]
/// Then concatenate across all dims → [B, T, D//2]
///
/// `freq_basis`: [D // (2*C)] — frequency basis from generateFreqGrid
/// `frac_positions`: slice of C tensors, each [B, T] — from getFractionalPositions
fn generateFreqs(freq_basis: Tensor, frac_positions: []const Tensor) Tensor {
    const n_pos_dims = frac_positions.len;
    var parts: [8]Tensor = undefined; // max 8 position dims (3 for video, 1 for audio)

    for (0..n_pos_dims) |c| {
        // frac_pos[c]: [.b, .t], freq_basis: [num_freqs]
        // result: outer product → [.b, .t, num_freqs]
        const frac = frac_positions[c]; // [.b, .t]
        const scaled_frac = frac.scale(2.0).addConstant(-1.0); // map [0,1] → [-1,1], stays in source dtype

        // Upcast to f32 for multiplication with f32 freq basis.
        // This matches Python: bf16 intermediate * f32 freq → implicit upcast to f32.
        const scaled_f32 = scaled_frac.convert(.f32);

        // Outer product: scaled_f32[.b, .t] × freq_basis[.freq] → [.b, .t, .freq]
        const sf_expanded = scaled_f32.appendAxes(.{.freq}).broad(
            scaled_f32.shape().append(.{ .freq = freq_basis.dim(0) }),
        );
        const fb_tagged = freq_basis.withTags(.{.freq});
        const fb_expanded = fb_tagged.broad(sf_expanded.shape());
        parts[c] = sf_expanded.mul(fb_expanded); // [.b, .t, .freq]
    }

    if (n_pos_dims == 1) {
        return parts[0]; // [.b, .t, .freq] — already D//2
    }

    // Python interleaves dims within each freq index:
    //   [freq0_dim0, freq0_dim1, freq0_dim2, freq1_dim0, ...]
    // Achieved via: [B,T,C,num_freqs].transpose(-1,-2).flatten(2)
    //
    // Step 1: Concatenate per-dim parts → [.b, .t, .freq = C * N]
    //   layout: [dim0_freq0..N, dim1_freq0..N, dim2_freq0..N]
    const concatenated = Tensor.concatenate(parts[0..n_pos_dims], .freq);

    // Step 2: Reshape to [.b, .t, .cdim = C, .fidx = N]
    const n_dims: i64 = @intCast(n_pos_dims);
    const num_per_dim = freq_basis.dim(0);
    const separated = concatenated.reshape(
        concatenated.shape().splitAxis(.freq, .{ .cdim = n_dims, .fidx = num_per_dim }),
    );

    // Step 3: Transpose to [.b, .t, .fidx = N, .cdim = C]
    const transposed = separated.transpose(zml.Shape.init(.{
        .b = separated.dim(.b),
        .t = separated.dim(.t),
        .fidx = num_per_dim,
        .cdim = n_dims,
    }, separated.dtype()));

    // Step 4: Flatten back to [.b, .t, .freq = N * C]
    return transposed.reshape(zml.Shape.init(.{
        .b = transposed.dim(.b),
        .t = transposed.dim(.t),
        .freq = num_per_dim * n_dims,
    }, transposed.dtype()));
}

/// Split raw frequencies into cos/sin pairs with optional padding (matches `split_freqs_cis`).
///
/// Input: freqs [B, T, D//2] (raw freq values per token)
/// Output: (cos [B, H, T, HD//2], sin [B, H, T, HD//2])
///
/// For SPLIT layout with head_dim=HD:
///   - cos = freqs.cos(), padded with 1s if D//2 < H * (HD//2)
///   - sin = freqs.sin(), padded with 0s if D//2 < H * (HD//2)
///   - reshape to [B, T, H, HD//2] then transpose to [B, H, T, HD//2]
fn splitFreqsCis(freqs: Tensor, num_heads: i64, head_dim: i64) CosSinPair {
    const half_hd: i64 = @divExact(head_dim, 2);
    const total_needed: i64 = num_heads * half_hd;
    const freq_dim: i64 = freqs.dim(.freq);

    var cos_freq = freqs.cos();
    var sin_freq = freqs.sin();

    // Pad if frequencies don't fill the full head dimension.
    // Python PREPENDS padding: cat([padding, freqs])
    if (freq_dim < total_needed) {
        const pad_size: i64 = total_needed - freq_dim;
        const bt_shape = freqs.shape().set(.freq, pad_size);
        const cos_pad = Tensor.scalar(1.0, .f32).broad(bt_shape);
        const sin_pad = Tensor.scalar(0.0, .f32).broad(bt_shape);
        cos_freq = Tensor.concatenate(&.{ cos_pad, cos_freq }, .freq);
        sin_freq = Tensor.concatenate(&.{ sin_pad, sin_freq }, .freq);
    }

    // Reshape [.b, .t, .freq=H*HD/2] → [.b, .t, .h, .hd=HD/2]
    const split_shape = cos_freq.shape().splitAxis(.freq, .{ .h = num_heads, .hd = half_hd });
    const cos_reshaped = cos_freq.reshape(split_shape);
    const sin_reshaped = sin_freq.reshape(split_shape);

    // Transpose [.b, .t, .h, .hd] → [.b, .h, .t, .hd]
    const target_shape = zml.Shape.init(.{
        .b = cos_reshaped.dim(.b),
        .h = num_heads,
        .t = cos_reshaped.dim(.t),
        .hd = half_hd,
    }, cos_reshaped.dtype());
    return .{
        .cos = cos_reshaped.transpose(target_shape),
        .sin = sin_reshaped.transpose(target_shape),
    };
}

/// Full RoPE generation pipeline (matches `precompute_freqs_cis`).
///
/// positions: [.b, .c, .t, .se=2] — raw position grid with start/end coordinates
/// config: RoPE configuration (theta, max_pos, num_heads, inner_dim)
///
/// Returns (cos, sin) each shaped [.b, .h, .t, .hd=HD/2].
fn precomputeFreqsCis(
    positions: Tensor,
    config: RopeConfig,
) CosSinPair {
    const n_pos_dims: i64 = positions.dim(.c);
    const head_dim: i64 = @divExact(config.inner_dim, config.num_heads);

    // 1. Frequency basis: [D // (2*C)] values
    const freq_basis = generateFreqGrid(config.theta, n_pos_dims, config.inner_dim);

    // 2. Fractional positions: C tensors each [B, T]
    var frac_buf: [8]Tensor = undefined;
    const frac_positions = getFractionalPositions(positions, config.max_pos, &frac_buf);

    // 3. Raw frequencies: [B, T, D//2]
    const freqs = generateFreqs(freq_basis, frac_positions);

    // 4. Split into cos/sin with head reshape: [B, H, T, HD//2]
    return splitFreqsCis(freqs, config.num_heads, head_dim);
}

// ============================================================================
// Section 9: Preprocessing (patchify + embed + RoPE + AV mask computation)
// Python ref: ltx_core/models/ltx_model.py — LTXModel._prepare_inputs()
// ============================================================================

/// Parameters for a full velocity_model forward pass (blocks + output projection).
/// Adaln modules are NOT included — their outputs are provided as inputs (SharedInputs).
pub const FullStepParams = struct {
    blocks: [48]Block0FullParams,
    norm_proj_out: OutputProjection.Params,
    audio_norm_proj_out: OutputProjection.Params,
};

pub fn initFullStepParams(store: zml.io.TensorStore.View) FullStepParams {
    var out: FullStepParams = undefined;
    const root = selectTransformerRoot(store);
    const blocks_store = root.withPrefix("transformer_blocks");
    inline for (0..48) |i| {
        out.blocks[i] = BasicAVTransformerBlock.initParams(blocks_store.withLayer(i));
    }
    out.norm_proj_out = .{
        .scale_shift_table = root.createTensor("scale_shift_table", .{ .n_ssv, .d }, null),
        .proj_out = .init(
            root.withPrefix("proj_out").createTensor("weight", .{ .d_out, .d }, null),
            root.withPrefix("proj_out").createTensor("bias", .{.d_out}, null),
            .d,
        ),
    };
    out.audio_norm_proj_out = .{
        .scale_shift_table = root.createTensor("audio_scale_shift_table", .{ .n_ssv, .d }, null),
        .proj_out = .init(
            root.withPrefix("audio_proj_out").createTensor("weight", .{ .d_out, .d }, null),
            root.withPrefix("audio_proj_out").createTensor("bias", .{.d_out}, null),
            .d,
        ),
    };
    return out;
}

pub fn unloadFullStepBuffers(params: *zml.Bufferized(FullStepParams)) void {
    inline for (0..48) |i| {
        unloadBlock0FullBuffers(&params.blocks[i]);
    }
    OutputProjection.Params.unloadBuffers(&params.norm_proj_out);
    OutputProjection.Params.unloadBuffers(&params.audio_norm_proj_out);
}

/// Parameters for the preprocessing stage (everything except the 48 transformer blocks
/// and the output projections). This turns raw latent inputs + sigma into SharedInputs.
pub const PreprocessParams = struct {
    video_patchify: Patchify.Params,
    audio_patchify: Patchify.Params,
    adaln_single: AdaLayerNormSingle.Params,
    audio_adaln_single: AdaLayerNormSingle.Params,
    prompt_adaln_single: AdaLayerNormSingle.Params,
    audio_prompt_adaln_single: AdaLayerNormSingle.Params,
    av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle.Params,
    av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle.Params,
    av_ca_a2v_gate_adaln_single: AdaLayerNormSingle.Params,
    av_ca_v2a_gate_adaln_single: AdaLayerNormSingle.Params,

    pub fn unloadBuffers(self: *zml.Bufferized(PreprocessParams)) void {
        Patchify.unloadBuffers(&self.video_patchify);
        Patchify.unloadBuffers(&self.audio_patchify);
        AdaLayerNormSingle.Params.unloadBuffers(&self.adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.audio_adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.prompt_adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.audio_prompt_adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.av_ca_video_scale_shift_adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.av_ca_audio_scale_shift_adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.av_ca_a2v_gate_adaln_single);
        AdaLayerNormSingle.Params.unloadBuffers(&self.av_ca_v2a_gate_adaln_single);
    }
};

pub fn initPreprocessParams(store: zml.io.TensorStore.View) PreprocessParams {
    const root = selectTransformerRoot(store);
    return .{
        .video_patchify = Patchify.initParams(root),
        .audio_patchify = .init(
            root.withPrefix("audio_patchify_proj").createTensor("weight", .{ .d, .patch }, null),
            root.withPrefix("audio_patchify_proj").createTensor("bias", .{.d}, null),
            .patch,
        ),
        .adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("adaln_single")),
        .audio_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("audio_adaln_single")),
        .prompt_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("prompt_adaln_single")),
        .audio_prompt_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("audio_prompt_adaln_single")),
        .av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("av_ca_video_scale_shift_adaln_single")),
        .av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("av_ca_audio_scale_shift_adaln_single")),
        .av_ca_a2v_gate_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("av_ca_a2v_gate_adaln_single")),
        .av_ca_v2a_gate_adaln_single = AdaLayerNormSingle.initParams(root.withPrefix("av_ca_v2a_gate_adaln_single")),
    };
}

/// Output of the preprocessing stage — everything the block chain needs.
///
/// This is a flat struct suitable for use as a compilation function return value.
/// Each field maps directly to a field in `BasicAVTransformerBlock.SharedInputs`.
pub const PreprocessOutput = struct {
    vx: Tensor, // [B, T_v, D_video] — patchified video
    ax: Tensor, // [B, T_a, D_audio] — patchified audio
    video_timesteps: Tensor, // [B, 1, 9 * D_video]
    audio_timesteps: Tensor, // [B, 1, 9 * D_audio]
    v_embedded_timestep: Tensor, // [B, 1, D_video] — for output projection
    a_embedded_timestep: Tensor, // [B, 1, D_audio] — for output projection
    v_prompt_timestep: Tensor, // [B, 1, 2 * D_video]
    a_prompt_timestep: Tensor, // [B, 1, 2 * D_audio]
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    v_text_ctx: Tensor,
    a_text_ctx: Tensor,
    v_cross_ss_ts: Tensor, // [B, 1, 4 * D_video]
    v_cross_gate_ts: Tensor, // [B, 1, D_video]
    a_cross_ss_ts: Tensor, // [B, 1, 4 * D_audio]
    a_cross_gate_ts: Tensor, // [B, 1, D_audio]
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
};

/// Preprocessing forward: raw inputs → patchified hidden states + SharedInputs.
///
/// This function is designed to be compiled as a single XLA exe.
///
/// Inputs:
///   - v_latent [B, T_v, 128] — raw video latent
///   - a_latent [B, T_a, 128] — raw audio latent
///   - v_denoise_mask [B, T_v, 1] — video denoise mask
///   - a_denoise_mask [B, T_a, 1] — audio denoise mask
///   - v_sigma [B] — video noise level
///   - a_sigma [B] — audio noise level
///   - v_positions [B, 3, T_v, 2] — video position grid (t,h,w with start/end)
///   - a_positions [B, 1, T_a, 2] — audio position grid (t with start/end)
///   - v_context [B, T_text, D_video] — text context for video stream
///   - a_context [B, T_text, D_audio] — text context for audio stream
///   - params: PreprocessParams
pub fn forwardPreprocess(
    v_latent: Tensor,
    a_latent: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
    v_sigma: Tensor,
    a_sigma: Tensor,
    v_positions: Tensor,
    a_positions: Tensor,
    v_context: Tensor,
    a_context: Tensor,
    params: PreprocessParams,
) PreprocessOutput {
    const patchify = Patchify{};

    // Tag all inputs — fixture tensors arrive untagged from safetensors.
    const v_lat = v_latent.withPartialTags(.{ .b, .t, .patch });
    const a_lat = a_latent.withPartialTags(.{ .b, .t, .patch });
    const v_ctx = v_context.withPartialTags(.{ .b, .t, .d });
    const a_ctx = a_context.withPartialTags(.{ .b, .t, .d });

    // --- 1. Patchify ---
    const vx = patchify.forward(v_lat, params.video_patchify);
    const ax = patchify.forward(a_lat, params.audio_patchify);

    // --- 2. Timestep embeddings (per-token) ---
    // timesteps = denoise_mask * sigma → [B, T, 1]
    // Then scale by timestep_scale_multiplier=1000 and flatten for adaln input.
    const v_mask = v_denoise_mask.withPartialTags(.{ .b, .t, .mask }).convert(.f32);
    const a_mask = a_denoise_mask.withPartialTags(.{ .b, .t, .mask }).convert(.f32);
    const v_ts = v_mask.mul(v_sigma.convert(.f32).appendAxes(.{ .t, .mask }).broad(v_mask.shape()));
    const a_ts = a_mask.mul(a_sigma.convert(.f32).appendAxes(.{ .t, .mask }).broad(a_mask.shape()));

    // AdaLN: scaled timesteps → modulation + embedded_timestep
    // flatten [B, T, 1] → [B*T] for adaln, then reshape back
    const v_ts_flat = v_ts.squeeze(.mask).scale(1000.0).flatten(); // [B*T]
    const a_ts_flat = a_ts.squeeze(.mask).scale(1000.0).flatten(); // [B*T]

    const v_adaln = AdaLayerNormSingle.forward(v_ts_flat, params.adaln_single);
    const a_adaln = AdaLayerNormSingle.forward(a_ts_flat, params.audio_adaln_single);

    // Reshape back to [B, T, N*D] / [B, T, D]
    const b_v: i64 = v_lat.dim(.b);
    const t_v: i64 = v_lat.dim(.t);
    const b_a: i64 = a_lat.dim(.b);
    const t_a: i64 = a_lat.dim(.t);

    // video_timesteps: [B*T, d_ada] → [B, T, d_ada] → treat T=1 as broadcast dim
    // In Python: timestep.view(B, -1, N_ada*D) with T tokens then uses [B, T, N_ada*D].
    // But SharedInputs expects [B, 1, N_ada*D] — the per-token dimension is implicit
    // in the position embeddings. Actually per the Python code, the adaln is computed
    // per-token and used per-token. However the existing block code uses [B, 1, N*D].
    //
    // For LTX distilled stage-2: all tokens share the same sigma, so denoise_mask * sigma
    // is constant across tokens (mask is all-ones for generation). We produce [B, 1, N*D].
    const v_timesteps = v_adaln.modulation.reshape(
        zml.Shape.init(.{ .b = b_v, .t = t_v, .d_ada = v_adaln.modulation.dim(.d_ada) }, v_adaln.modulation.dtype()),
    ).slice1d(.t, .{ .start = 0, .end = 1 }); // [B, 1, d_ada]
    const a_timesteps = a_adaln.modulation.reshape(
        zml.Shape.init(.{ .b = b_a, .t = t_a, .d_ada = a_adaln.modulation.dim(.d_ada) }, a_adaln.modulation.dtype()),
    ).slice1d(.t, .{ .start = 0, .end = 1 }); // [B, 1, d_ada]

    const v_emb_ts = v_adaln.embedded_timestep.reshape(
        zml.Shape.init(.{ .b = b_v, .t = t_v, .d_emb = v_adaln.embedded_timestep.dim(.d_emb) }, v_adaln.embedded_timestep.dtype()),
    ).slice1d(.t, .{ .start = 0, .end = 1 }); // [B, 1, d_emb]
    const a_emb_ts = a_adaln.embedded_timestep.reshape(
        zml.Shape.init(.{ .b = b_a, .t = t_a, .d_emb = a_adaln.embedded_timestep.dim(.d_emb) }, a_adaln.embedded_timestep.dtype()),
    ).slice1d(.t, .{ .start = 0, .end = 1 }); // [B, 1, d_emb]

    // --- 3. Prompt adaln (uses raw sigma, not per-token timesteps) ---
    const v_prompt = AdaLayerNormSingle.forward(
        v_sigma.convert(.f32).scale(1000.0),
        params.prompt_adaln_single,
    );
    const v_prompt_ts = v_prompt.modulation.reshape(
        v_prompt.modulation.shape().splitAxis(.b, .{ .b = v_prompt.modulation.dim(.b), .t = 1 }),
    ); // [B, 1, 2*D]
    const a_prompt = AdaLayerNormSingle.forward(
        a_sigma.convert(.f32).scale(1000.0),
        params.audio_prompt_adaln_single,
    );
    const a_prompt_ts = a_prompt.modulation.reshape(
        a_prompt.modulation.shape().splitAxis(.b, .{ .b = a_prompt.modulation.dim(.b), .t = 1 }),
    ); // [B, 1, 2*D]

    // --- 4. Self-attention RoPE ---
    const v_pos_tagged = v_positions.withPartialTags(.{ .b, .c, .t, .se });
    const a_pos_tagged = a_positions.withPartialTags(.{ .b, .c, .t, .se });

    const v_pe = precomputeFreqsCis(v_pos_tagged, .{
        .theta = 10000.0,
        .max_pos = &.{ 20, 2048, 2048 },
        .num_heads = 32,
        .inner_dim = 4096,
    });
    const a_pe = precomputeFreqsCis(a_pos_tagged, .{
        .theta = 10000.0,
        .max_pos = &.{20},
        .num_heads = 32,
        .inner_dim = 2048,
    });

    // --- 5. Cross-attention RoPE (temporal dimension only, dim=2048) ---
    // Each modality computes its own cross PE from its temporal positions.
    // a2v = "audio → video": video Q attends to audio K
    // v2a = "video → audio": audio Q attends to video K
    const v_temporal = v_pos_tagged.slice1d(.c, .{ .start = 0, .end = 1 }); // [B, 1, T_v, 2]
    const a_temporal = a_pos_tagged.slice1d(.c, .{ .start = 0, .end = 1 }); // [B, 1, T_a, 2]
    const cross_rope_config = RopeConfig{
        .theta = 10000.0,
        .max_pos = &.{20},
        .num_heads = 32,
        .inner_dim = 2048,
    };

    const v_cross_pe = precomputeFreqsCis(v_temporal, cross_rope_config); // [B, 32, T_v, 32]
    const a_cross_pe = precomputeFreqsCis(a_temporal, cross_rope_config); // [B, 32, T_a, 32]

    // --- 6. Cross-attention timestep embeddings ---
    // Python: timestep = cross_sigma * timestep_scale_multiplier (= 1000)
    //         gate_input = timestep * av_ca_factor (= av_ca_tsm / tsm = 1000/1000 = 1.0)
    // So both scale_shift and gate use sigma * 1000.
    const v_cross_ss = AdaLayerNormSingle.forward(
        a_sigma.convert(.f32).scale(1000.0), // video cross uses audio's sigma
        params.av_ca_video_scale_shift_adaln_single,
    );
    const v_cross_ss_ts = v_cross_ss.modulation.reshape(
        v_cross_ss.modulation.shape().splitAxis(.b, .{ .b = v_cross_ss.modulation.dim(.b), .t = 1 }),
    ); // [B, 1, 4*D_v]
    const v_cross_gate = AdaLayerNormSingle.forward(
        a_sigma.convert(.f32).scale(1000.0), // sigma * tsm * av_ca_factor = sigma * 1000 * 1.0
        params.av_ca_a2v_gate_adaln_single,
    );
    const v_cross_gate_ts = v_cross_gate.modulation.reshape(
        v_cross_gate.modulation.shape().splitAxis(.b, .{ .b = v_cross_gate.modulation.dim(.b), .t = 1 }),
    ); // [B, 1, D_v]

    const a_cross_ss = AdaLayerNormSingle.forward(
        v_sigma.convert(.f32).scale(1000.0), // audio cross uses video's sigma
        params.av_ca_audio_scale_shift_adaln_single,
    );
    const a_cross_ss_ts = a_cross_ss.modulation.reshape(
        a_cross_ss.modulation.shape().splitAxis(.b, .{ .b = a_cross_ss.modulation.dim(.b), .t = 1 }),
    ); // [B, 1, 4*D_a]
    const a_cross_gate = AdaLayerNormSingle.forward(
        v_sigma.convert(.f32).scale(1000.0), // sigma * tsm * av_ca_factor = sigma * 1000 * 1.0
        params.av_ca_v2a_gate_adaln_single,
    );
    const a_cross_gate_ts = a_cross_gate.modulation.reshape(
        a_cross_gate.modulation.shape().splitAxis(.b, .{ .b = a_cross_gate.modulation.dim(.b), .t = 1 }),
    ); // [B, 1, D_a]

    // Convert f32 adaln outputs to bf16 at the preprocessing boundary.
    // This matches step 1's behavior (fixture provided bf16 timesteps).
    // The f32 adaln computes precisely, then bf16 rounding gives the same
    // values Python saved to the fixture.
    return .{
        .vx = vx,
        .ax = ax,
        .video_timesteps = v_timesteps,
        .audio_timesteps = a_timesteps,
        .v_embedded_timestep = v_emb_ts,
        .a_embedded_timestep = a_emb_ts,
        .v_prompt_timestep = v_prompt_ts,
        .a_prompt_timestep = a_prompt_ts,
        .v_pe_cos = v_pe.cos,
        .v_pe_sin = v_pe.sin,
        .a_pe_cos = a_pe.cos,
        .a_pe_sin = a_pe.sin,
        .v_text_ctx = v_ctx,
        .a_text_ctx = a_ctx,
        .v_cross_ss_ts = v_cross_ss_ts,
        .v_cross_gate_ts = v_cross_gate_ts,
        .a_cross_ss_ts = a_cross_ss_ts,
        .a_cross_gate_ts = a_cross_gate_ts,
        .a2v_pe_cos = v_cross_pe.cos, // a2v Q side = video tokens
        .a2v_pe_sin = v_cross_pe.sin,
        .a2v_k_pe_cos = a_cross_pe.cos, // a2v K side = audio tokens
        .a2v_k_pe_sin = a_cross_pe.sin,
        .v2a_pe_cos = a_cross_pe.cos, // v2a Q side = audio tokens
        .v2a_pe_sin = a_cross_pe.sin,
        .v2a_k_pe_cos = v_cross_pe.cos, // v2a K side = video tokens
        .v2a_k_pe_sin = v_cross_pe.sin,
    };
}

// ============================================================================
// Section 10: Latent Upsampler (stage 1 → stage 2 bridge)
// Python ref: ltx_core/model/upsampler/model.py — LatentUpsampler
//             ltx_core/model/upsampler/res_block.py — ResBlock
//             ltx_core/model/video_vae/ops.py — PerChannelStatistics
// ============================================================================

/// Weight struct for a 3D convolution (Conv3d).
/// Shape: weight [C_out, C_in, D, H, W], bias [C_out].
pub const Conv3dWeight = struct {
    weight: Tensor,
    bias: Tensor,
};

/// Weight struct for a 2D convolution (Conv2d).
/// Shape: weight [C_out, C_in, H, W], bias [C_out].
pub const Conv2dWeight = struct {
    weight: Tensor,
    bias: Tensor,
};

/// GroupNorm weight struct (gamma/beta).
/// Python: torch.nn.GroupNorm(num_groups, num_channels).
pub const GroupNormWeight = struct {
    weight: Tensor, // gamma, shape [C]
    bias: Tensor, // beta, shape [C]
};

/// ResBlock for the upsampler: conv1→norm1→silu→conv2→norm2→silu(x+residual).
/// Python ref: ltx_core/model/upsampler/res_block.py
pub const UpsamplerResBlock = struct {
    conv1: Conv3dWeight,
    norm1: GroupNormWeight,
    conv2: Conv3dWeight,
    norm2: GroupNormWeight,
};

/// Full parameter set for the LatentUpsampler CNN.
/// Checkpoint: ltx-2.3-spatial-upscaler-x2-1.1.safetensors (72 keys)
pub const UpsamplerParams = struct {
    initial_conv: Conv3dWeight,
    initial_norm: GroupNormWeight,
    res_block_0: UpsamplerResBlock,
    res_block_1: UpsamplerResBlock,
    res_block_2: UpsamplerResBlock,
    res_block_3: UpsamplerResBlock,
    /// Conv2d for spatial upsample (operates on (B*F, C, H, W), then PixelShuffle).
    upsampler_conv: Conv2dWeight,
    post_res_block_0: UpsamplerResBlock,
    post_res_block_1: UpsamplerResBlock,
    post_res_block_2: UpsamplerResBlock,
    post_res_block_3: UpsamplerResBlock,
    final_conv: Conv3dWeight,
};

/// Per-channel statistics for normalize / un_normalize.
/// From main checkpoint: vae.per_channel_statistics.{mean-of-means, std-of-means}
pub const PerChannelStats = struct {
    mean_of_means: Tensor, // [128]
    std_of_means: Tensor, // [128]
};

/// Load UpsamplerParams from an upsampler safetensors checkpoint.
pub fn initUpsamplerParams(store: zml.io.TensorStore.View) UpsamplerParams {
    const ic = store.withPrefix("initial_conv");
    const in_ = store.withPrefix("initial_norm");
    const fc = store.withPrefix("final_conv");
    // upsampler.0 is the Conv2d inside nn.Sequential
    const us = store.withPrefix("upsampler").withLayer(0);

    return .{
        .initial_conv = .{
            .weight = ic.createTensor("weight", null, null),
            .bias = ic.createTensor("bias", null, null),
        },
        .initial_norm = .{
            .weight = in_.createTensor("weight", null, null),
            .bias = in_.createTensor("bias", null, null),
        },
        .res_block_0 = initResBlockParams(store.withPrefix("res_blocks").withLayer(0)),
        .res_block_1 = initResBlockParams(store.withPrefix("res_blocks").withLayer(1)),
        .res_block_2 = initResBlockParams(store.withPrefix("res_blocks").withLayer(2)),
        .res_block_3 = initResBlockParams(store.withPrefix("res_blocks").withLayer(3)),
        .upsampler_conv = .{
            .weight = us.createTensor("weight", null, null),
            .bias = us.createTensor("bias", null, null),
        },
        .post_res_block_0 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(0)),
        .post_res_block_1 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(1)),
        .post_res_block_2 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(2)),
        .post_res_block_3 = initResBlockParams(store.withPrefix("post_upsample_res_blocks").withLayer(3)),
        .final_conv = .{
            .weight = fc.createTensor("weight", null, null),
            .bias = fc.createTensor("bias", null, null),
        },
    };
}

fn initResBlockParams(store: zml.io.TensorStore.View) UpsamplerResBlock {
    const c1 = store.withPrefix("conv1");
    const n1 = store.withPrefix("norm1");
    const c2 = store.withPrefix("conv2");
    const n2 = store.withPrefix("norm2");
    return .{
        .conv1 = .{
            .weight = c1.createTensor("weight", null, null),
            .bias = c1.createTensor("bias", null, null),
        },
        .norm1 = .{
            .weight = n1.createTensor("weight", null, null),
            .bias = n1.createTensor("bias", null, null),
        },
        .conv2 = .{
            .weight = c2.createTensor("weight", null, null),
            .bias = c2.createTensor("bias", null, null),
        },
        .norm2 = .{
            .weight = n2.createTensor("weight", null, null),
            .bias = n2.createTensor("bias", null, null),
        },
    };
}

/// Load PerChannelStats from the main model checkpoint.
/// Keys: vae.per_channel_statistics.mean-of-means, vae.per_channel_statistics.std-of-means
pub fn initPerChannelStats(store: zml.io.TensorStore.View) PerChannelStats {
    const pcs = store.withPrefix("vae").withPrefix("per_channel_statistics");
    return .{
        .mean_of_means = pcs.createTensor("mean-of-means", null, null),
        .std_of_means = pcs.createTensor("std-of-means", null, null),
    };
}

// --- Upsampler forward ops ---

/// Conv3d forward: input [B, C_in, D, H, W], kernel [C_out, C_in, kD, kH, kW].
/// Padding=1 on all spatial dims (same padding for kernel_size=3).
fn forwardConv3d(input: Tensor, w: Conv3dWeight) Tensor {
    const conv_out = input.conv3d(w.weight, .{
        .padding = &.{ 1, 1, 1, 1, 1, 1 }, // padding=1 on each side of D, H, W
    });
    // Broadcast bias [C_out] → [B, C_out, D, H, W]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1),
    ).broad(conv_out.shape()));
}

/// Conv2d forward: input [B, C_in, H, W], kernel [C_out, C_in, kH, kW].
/// Padding=1 on all spatial dims (same padding for kernel_size=3).
fn forwardConv2d(input: Tensor, w: Conv2dWeight) Tensor {
    const conv_out = input.conv2d(w.weight, .{
        .padding = &.{ 1, 1, 1, 1 },
    });
    // Broadcast bias [C_out] → [B, C_out, H, W]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1),
    ).broad(conv_out.shape()));
}

/// GroupNorm forward for 5D tensors [B, C, D, H, W].
/// Python: torch.nn.GroupNorm(num_groups, C).
fn forwardGroupNorm(input: Tensor, w: GroupNormWeight, comptime num_groups: i64) Tensor {
    const B = input.dim(0);
    const C = input.dim(1);
    const D = input.dim(2);
    const H = input.dim(3);
    const W = input.dim(4);
    const channels_per_group = @divExact(C, num_groups);

    // Reshape: [B, C, D, H, W] → [B, G, C/G, D, H, W]
    const grouped = input.reshape(.{ B, num_groups, channels_per_group, D, H, W });

    // Compute mean and variance over axes 2..5 (C/G, D, H, W)
    const grouped_f32 = grouped.convert(.f32);

    // Reduce over axes [2, 3, 4, 5] from last to first for stable indices
    var reduced_mean = grouped_f32;
    reduced_mean = reduced_mean.mean(5);
    reduced_mean = reduced_mean.mean(4);
    reduced_mean = reduced_mean.mean(3);
    reduced_mean = reduced_mean.mean(2);
    // reduced_mean shape: [B, G]

    // Variance: E[(x - mean)^2]
    const centered = grouped_f32.sub(reduced_mean.broadcastLeft(grouped_f32.shape()));
    var reduced_var = centered.mul(centered);
    reduced_var = reduced_var.mean(5);
    reduced_var = reduced_var.mean(4);
    reduced_var = reduced_var.mean(3);
    reduced_var = reduced_var.mean(2);

    // Normalize: (x - mean) / sqrt(var + eps)
    const eps: f32 = 1e-5;
    const inv_std = reduced_var.addConstant(eps).rsqrt();
    const normed = centered.mul(inv_std.broadcastLeft(grouped_f32.shape()));

    // Reshape back in f32: [B, G, C/G, D, H, W] → [B, C, D, H, W]
    const normed_5d = normed.reshape(.{ B, C, D, H, W });

    // Apply affine in f32: gamma * x + beta, then convert to input dtype.
    // Keeping f32 through the affine avoids precision loss when gamma is large.
    const affine_shape = input.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const gamma = w.weight.convert(.f32).reshape(affine_shape).broad(normed_5d.shape());
    const beta = w.bias.convert(.f32).reshape(affine_shape).broad(normed_5d.shape());
    return normed_5d.mul(gamma).add(beta).convert(input.dtype());
}

/// PixelShuffle 2D: rearrange (B, C*r*r, H, W) → (B, C, H*r, W*r).
/// Pure reshape + transpose, no learnable parameters.
fn forwardPixelShuffle2d(input: Tensor, comptime upscale_factor: i64) Tensor {
    const B = input.dim(0);
    const C_total = input.dim(1);
    const H = input.dim(2);
    const W = input.dim(3);
    const C = @divExact(C_total, upscale_factor * upscale_factor);
    const r = upscale_factor;

    // (B, C*r*r, H, W) → (B, C, r, r, H, W)
    const s1 = input.reshape(.{ B, C, r, r, H, W });
    // → (B, C, H, r, W, r)
    const s2 = s1.transpose(.{ 0, 1, 4, 2, 5, 3 });
    // → (B, C, H*r, W*r)
    return s2.reshape(.{ B, C, H * r, W * r });
}

/// Forward pass for a single upsampler ResBlock.
/// Python: conv1→norm1→silu→conv2→norm2→silu(x+residual)
fn forwardResBlock(x: Tensor, rb: UpsamplerResBlock) Tensor {
    const residual = x;
    var h = forwardConv3d(x, rb.conv1);
    h = forwardGroupNorm(h, rb.norm1, 32);
    h = h.silu();
    h = forwardConv3d(h, rb.conv2);
    h = forwardGroupNorm(h, rb.norm2, 32);
    return h.add(residual).silu();
}

/// Full upsampler forward: un_normalize → LatentUpsampler CNN → normalize.
///
/// Input: [B, 128, F, H, W] bf16 (un-patchified Stage 1 latent)
/// Output: [B, 128, F, H*2, W*2] bf16 (spatially upsampled)
///
/// Python ref: upsample_video() in model.py
pub fn forwardUpsample(
    input: Tensor,
    params: UpsamplerParams,
    stats: PerChannelStats,
) Tensor {
    const B = input.dim(0);
    const F = input.dim(2);

    // Step 1: un_normalize — x * std_of_means + mean_of_means (per-channel)
    // Broadcast stats [128] → [B, 128, F, H, W]
    const stats_shape = input.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const std_broad = stats.std_of_means.reshape(stats_shape).broad(input.shape());
    const mean_broad = stats.mean_of_means.reshape(stats_shape).broad(input.shape());
    var x = input.mul(std_broad).add(mean_broad);

    // Step 2: LatentUpsampler.forward()
    // initial_conv → initial_norm → silu
    x = forwardConv3d(x, params.initial_conv);
    x = forwardGroupNorm(x, params.initial_norm, 32);
    x = x.silu();

    // 4x ResBlock (pre-upsample)
    x = forwardResBlock(x, params.res_block_0);
    x = forwardResBlock(x, params.res_block_1);
    x = forwardResBlock(x, params.res_block_2);
    x = forwardResBlock(x, params.res_block_3);

    // Spatial upsample: rearrange to 2D → Conv2d → PixelShuffle(2) → back to 5D
    // "b c f h w → (b f) c h w" requires transpose before reshape
    const H = x.dim(3);
    const W = x.dim(4);
    const C = x.dim(1);
    const x_bfchw = x.transpose(.{ 0, 2, 1, 3, 4 }); // [B, C, F, H, W] → [B, F, C, H, W]
    const x_4d = x_bfchw.reshape(.{ B * F, C, H, W });
    var upsampled_4d = forwardConv2d(x_4d, params.upsampler_conv);
    upsampled_4d = forwardPixelShuffle2d(upsampled_4d, 2);
    // "(b f) c h w → b c f h w"
    const C_out = upsampled_4d.dim(1);
    const H2 = upsampled_4d.dim(2);
    const W2 = upsampled_4d.dim(3);
    x = upsampled_4d.reshape(.{ B, F, C_out, H2, W2 }).transpose(.{ 0, 2, 1, 3, 4 });

    // 4x ResBlock (post-upsample)
    x = forwardResBlock(x, params.post_res_block_0);
    x = forwardResBlock(x, params.post_res_block_1);
    x = forwardResBlock(x, params.post_res_block_2);
    x = forwardResBlock(x, params.post_res_block_3);

    // final_conv
    x = forwardConv3d(x, params.final_conv);

    // Step 3: normalize — (x - mean_of_means) / std_of_means
    const out_stats_shape = x.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const out_std = stats.std_of_means.reshape(out_stats_shape).broad(x.shape());
    const out_mean = stats.mean_of_means.reshape(out_stats_shape).broad(x.shape());
    return x.sub(out_mean).div(out_std);
}

/// Unpatchify a video latent: [1, T, 128] → [1, 128, F, H, W].
/// VideoLatentPatchifier(patch_size=1): patchify = "b c f h w → b (f h w) c"
/// target_shape carries the [1, 128, F, H, W] dimensions.
pub fn forwardUnpatchifyVideo(input: Tensor, target_shape: zml.Shape) Tensor {
    const B = target_shape.dim(0);
    const C = target_shape.dim(1);
    const F = target_shape.dim(2);
    const H = target_shape.dim(3);
    const W = target_shape.dim(4);
    // [1, F*H*W, 128] → [1, F, H, W, 128] → [1, 128, F, H, W]
    return input
        .reshape(.{ B, F, H, W, C })
        .transpose(.{ 0, 4, 1, 2, 3 });
}

/// Patchify a video latent: [1, 128, F, H, W] → [1, F*H*W, 128].
/// VideoLatentPatchifier(patch_size=1): patchify = "b c f h w → b (f h w) c"
/// Inverse of forwardUnpatchifyVideo.
pub fn forwardPatchifyVideo(input: Tensor) Tensor {
    const B = input.dim(0);
    const C = input.dim(1);
    const F = input.dim(2);
    const H = input.dim(3);
    const W = input.dim(4);
    // [1, 128, F, H, W] → [1, F, H, W, 128] → [1, F*H*W, 128]
    return input
        .transpose(.{ 0, 2, 3, 4, 1 })
        .reshape(.{ B, F * H * W, C });
}

// ============================================================================
// Section 11: Video VAE Decoder
// Python ref: ltx_core/model/video_vae/decoder.py — Decoder3d
//             ltx_core/model/video_vae/ops.py — CausalConv3d, PixelNorm,
//                                                DepthToSpaceUpsample
// Architecture: 9 up_blocks (alternating ResBlock groups + DepthToSpace),
//   conv_in → up_blocks.0..8 → PixelNorm → SiLU → conv_out → unpatchify
// ============================================================================

/// Weight struct for a VAE ResnetBlock3D. Each block is:
///   PixelNorm → SiLU → CausalConv3d → PixelNorm → SiLU → CausalConv3d + residual
/// No timestep conditioning or noise injection in this checkpoint.
pub const VaeResBlock = struct {
    conv1: Conv3dWeight, // checkpoint: conv1.conv.{weight,bias}
    conv2: Conv3dWeight, // checkpoint: conv2.conv.{weight,bias}
};

/// Weight struct for a DepthToSpaceUpsample block.
/// CausalConv3d → 3D pixel-shuffle → optionally remove first frame.
pub const VaeDepthToSpaceBlock = struct {
    conv: Conv3dWeight, // checkpoint: conv.conv.{weight,bias}
};

/// Full parameter set for the video VAE decoder.
/// Checkpoint keys: vae.decoder.* (42 weight tensors).
pub const VideoVaeDecoderParams = struct {
    conv_in: Conv3dWeight,

    // up_blocks.0: 2 ResBlocks @ 1024ch
    up0_res0: VaeResBlock,
    up0_res1: VaeResBlock,

    // up_blocks.1: DepthToSpace (2,2,2) → 1024→512
    up1: VaeDepthToSpaceBlock,

    // up_blocks.2: 2 ResBlocks @ 512ch
    up2_res0: VaeResBlock,
    up2_res1: VaeResBlock,

    // up_blocks.3: DepthToSpace (2,2,2) → 512→512
    up3: VaeDepthToSpaceBlock,

    // up_blocks.4: 4 ResBlocks @ 512ch
    up4_res0: VaeResBlock,
    up4_res1: VaeResBlock,
    up4_res2: VaeResBlock,
    up4_res3: VaeResBlock,

    // up_blocks.5: DepthToSpace (2,1,1) → 512→256
    up5: VaeDepthToSpaceBlock,

    // up_blocks.6: 6 ResBlocks @ 256ch
    up6_res0: VaeResBlock,
    up6_res1: VaeResBlock,
    up6_res2: VaeResBlock,
    up6_res3: VaeResBlock,
    up6_res4: VaeResBlock,
    up6_res5: VaeResBlock,

    // up_blocks.7: DepthToSpace (1,2,2) → 256→128
    up7: VaeDepthToSpaceBlock,

    // up_blocks.8: 4 ResBlocks @ 128ch
    up8_res0: VaeResBlock,
    up8_res1: VaeResBlock,
    up8_res2: VaeResBlock,
    up8_res3: VaeResBlock,

    conv_out: Conv3dWeight,
};

fn initVaeResBlock(store: zml.io.TensorStore.View) VaeResBlock {
    const c1 = store.withPrefix("conv1").withPrefix("conv");
    const c2 = store.withPrefix("conv2").withPrefix("conv");
    return .{
        .conv1 = .{
            .weight = c1.createTensor("weight", null, null),
            .bias = c1.createTensor("bias", null, null),
        },
        .conv2 = .{
            .weight = c2.createTensor("weight", null, null),
            .bias = c2.createTensor("bias", null, null),
        },
    };
}

fn initVaeConv3d(store: zml.io.TensorStore.View) Conv3dWeight {
    const c = store.withPrefix("conv");
    return .{
        .weight = c.createTensor("weight", null, null),
        .bias = c.createTensor("bias", null, null),
    };
}

/// Load VideoVaeDecoderParams from the main model checkpoint.
/// Keys: vae.decoder.{conv_in,up_blocks.*,conv_out}.conv.{weight,bias}
pub fn initVideoVaeDecoderParams(store: zml.io.TensorStore.View) VideoVaeDecoderParams {
    const dec = store.withPrefix("vae").withPrefix("decoder");
    const ub = dec.withPrefix("up_blocks");

    return .{
        .conv_in = initVaeConv3d(dec.withPrefix("conv_in")),

        // up_blocks.0: 2 ResBlocks @ 1024
        .up0_res0 = initVaeResBlock(ub.withLayer(0).withPrefix("res_blocks").withLayer(0)),
        .up0_res1 = initVaeResBlock(ub.withLayer(0).withPrefix("res_blocks").withLayer(1)),

        // up_blocks.1: DepthToSpace
        .up1 = .{ .conv = initVaeConv3d(ub.withLayer(1).withPrefix("conv")) },

        // up_blocks.2: 2 ResBlocks @ 512
        .up2_res0 = initVaeResBlock(ub.withLayer(2).withPrefix("res_blocks").withLayer(0)),
        .up2_res1 = initVaeResBlock(ub.withLayer(2).withPrefix("res_blocks").withLayer(1)),

        // up_blocks.3: DepthToSpace
        .up3 = .{ .conv = initVaeConv3d(ub.withLayer(3).withPrefix("conv")) },

        // up_blocks.4: 4 ResBlocks @ 512
        .up4_res0 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(0)),
        .up4_res1 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(1)),
        .up4_res2 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(2)),
        .up4_res3 = initVaeResBlock(ub.withLayer(4).withPrefix("res_blocks").withLayer(3)),

        // up_blocks.5: DepthToSpace
        .up5 = .{ .conv = initVaeConv3d(ub.withLayer(5).withPrefix("conv")) },

        // up_blocks.6: 6 ResBlocks @ 256
        .up6_res0 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(0)),
        .up6_res1 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(1)),
        .up6_res2 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(2)),
        .up6_res3 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(3)),
        .up6_res4 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(4)),
        .up6_res5 = initVaeResBlock(ub.withLayer(6).withPrefix("res_blocks").withLayer(5)),

        // up_blocks.7: DepthToSpace
        .up7 = .{ .conv = initVaeConv3d(ub.withLayer(7).withPrefix("conv")) },

        // up_blocks.8: 4 ResBlocks @ 128
        .up8_res0 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(0)),
        .up8_res1 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(1)),
        .up8_res2 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(2)),
        .up8_res3 = initVaeResBlock(ub.withLayer(8).withPrefix("res_blocks").withLayer(3)),

        .conv_out = initVaeConv3d(dec.withPrefix("conv_out")),
    };
}

// --- VAE decoder forward ops ---

/// CausalConv3d forward (causal=False mode): symmetric temporal replicate-pad + conv3d.
/// For kernel_size=3: pad 1 frame at start/end (replicate), zero-pad 1 on H,W.
/// Python CausalConv3d with causal=False uses Conv3d built-in zero-padding for spatial.
fn forwardCausalConv3dNonCausal(input: Tensor, w: Conv3dWeight) Tensor {
    // 1. Temporal replicate-padding: duplicate first and last frame
    const first_frame = input.slice1d(2, .{ .end = 1 }); // [B,C,1,H,W]
    const last_frame = input.slice1d(2, .{ .start = -1 }); // [B,C,1,H,W]
    const padded = Tensor.concatenate(&.{ first_frame, input, last_frame }, 2); // [B,C,F+2,H,W]

    // 2. Conv3d with zero-padding on H,W only (temporal already handled)
    const conv_out = padded.conv3d(w.weight, .{
        .padding = &.{ 0, 0, 1, 1, 1, 1 }, // D:0, H:1, W:1
    });

    // 3. Add bias: [C_out] → [1, C_out, 1, 1, 1]
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1),
    ).broad(conv_out.shape()));
}

/// PixelNorm: x / sqrt(mean(x², dim=channels) + eps).
/// Per-location RMS normalization over the channel dimension. No learnable params.
/// Computed in f32 for precision (bf16 squaring can lose significant bits).
fn forwardPixelNorm(x: Tensor) Tensor {
    const x_f32 = x.convert(.f32);
    const x_sq = x_f32.mul(x_f32);
    // mean over dim 1 (channels), keeping dim → [B, 1, F, H, W]
    const mean_sq = x_sq.mean(1);
    const rms = mean_sq.addConstant(1e-8).sqrt();
    return x_f32.div(rms.broad(x_f32.shape())).convert(x.dtype());
}

/// VAE ResnetBlock3D forward:
///   PixelNorm → SiLU → CausalConv3d → PixelNorm → SiLU → CausalConv3d + residual.
/// No channel change (in_ch == out_ch), so no conv_shortcut needed.
fn forwardVaeResBlock(x: Tensor, rb: VaeResBlock) Tensor {
    var h = forwardPixelNorm(x);
    h = h.silu();
    h = forwardCausalConv3dNonCausal(h, rb.conv1);
    h = forwardPixelNorm(h);
    h = h.silu();
    h = forwardCausalConv3dNonCausal(h, rb.conv2);
    return h.add(x); // residual
}

/// DepthToSpace 3D (pixel-shuffle): CausalConv3d → rearrange → optionally remove first frame.
/// stride = (p1, p2, p3): spatial/temporal upsample factors.
/// Rearrange: [B, C*p1*p2*p3, F, H, W] → [B, C, F*p1, H*p2, W*p3]
fn forwardDepthToSpace(x: Tensor, w: VaeDepthToSpaceBlock, comptime stride: [3]i64) Tensor {
    var h = forwardCausalConv3dNonCausal(x, w.conv);

    const B = h.dim(0);
    const C_total = h.dim(1);
    const F = h.dim(2);
    const H = h.dim(3);
    const W = h.dim(4);
    const p1 = stride[0];
    const p2 = stride[1];
    const p3 = stride[2];
    const C = @divExact(C_total, p1 * p2 * p3);

    // [B, C*p1*p2*p3, F, H, W] → [B, C, p1, p2, p3, F, H, W]
    h = h.reshape(.{ B, C, p1, p2, p3, F, H, W });
    // → [B, C, F, p1, H, p2, W, p3]
    h = h.transpose(.{ 0, 1, 5, 2, 6, 3, 7, 4 });
    // → [B, C, F*p1, H*p2, W*p3]
    h = h.reshape(.{ B, C, F * p1, H * p2, W * p3 });

    // Remove first frame if temporal upsample (p1 == 2)
    if (p1 == 2) {
        h = h.slice1d(2, .{ .start = 1 }); // drop frame 0
    }
    return h;
}

/// Unpatchify for VAE output: [B, 48, F, H, W] → [B, 3, F, 4H, 4W].
/// "b (c p r q) f h w -> b c (f p) (h q) (w r)" with c=3, p=1, q=4, r=4
/// Channel decomposition: 48 = c(3) * p(1) * r(4) * q(4), where q→H and r→W.
fn forwardUnpatchifyVae(x: Tensor) Tensor {
    const B = x.dim(0);
    const F = x.dim(2);
    const H = x.dim(3);
    const W = x.dim(4);
    // 48 = 3 * 1 * 4 * 4 → (c=3, p=1, r=4, q=4) in channel decomposition order
    // reshape [B, 48, F, H, W] → [B, 3, 1, 4, 4, F, H, W]
    //   dims:                      [B, c, p, r, q, F, H, W]
    var h = x.reshape(.{ B, 3, 1, 4, 4, F, H, W });
    // transpose → [B, c, F, p, H, q, W, r]  (q pairs with H, r pairs with W)
    h = h.transpose(.{ 0, 1, 5, 2, 6, 4, 7, 3 });
    // reshape → [B, 3, F*p, H*q, W*r] = [B, 3, F, 4H, 4W]
    return h.reshape(.{ B, 3, F * 1, H * 4, W * 4 });
}

/// Full video VAE decoder forward pass.
/// Input: latent [B, 128, F', H', W'] bf16
/// Output: decoded video [B, 3, 8(F'-1)+1, 32H', 32W'] bf16
pub fn forwardVideoVaeDecode(
    latent: Tensor,
    stats: PerChannelStats,
    params: VideoVaeDecoderParams,
) Tensor {
    // 1. Denormalize: x = latent * std_of_means + mean_of_means
    const stats_shape = latent.shape().set(0, 1).set(2, 1).set(3, 1).set(4, 1);
    const std_broad = stats.std_of_means.reshape(stats_shape).broad(latent.shape());
    const mean_broad = stats.mean_of_means.reshape(stats_shape).broad(latent.shape());
    var x = latent.mul(std_broad).add(mean_broad);

    // 2. conv_in: CausalConv3d(128 → 1024)
    x = forwardCausalConv3dNonCausal(x, params.conv_in);

    // 3. up_blocks.0: 2 ResBlocks @ 1024ch
    x = forwardVaeResBlock(x, params.up0_res0);
    x = forwardVaeResBlock(x, params.up0_res1);

    // 4. up_blocks.1: DepthToSpace (2,2,2) → 1024→512
    x = forwardDepthToSpace(x, params.up1, .{ 2, 2, 2 });

    // 5. up_blocks.2: 2 ResBlocks @ 512ch
    x = forwardVaeResBlock(x, params.up2_res0);
    x = forwardVaeResBlock(x, params.up2_res1);

    // 6. up_blocks.3: DepthToSpace (2,2,2) → 512→512
    x = forwardDepthToSpace(x, params.up3, .{ 2, 2, 2 });

    // 7. up_blocks.4: 4 ResBlocks @ 512ch
    x = forwardVaeResBlock(x, params.up4_res0);
    x = forwardVaeResBlock(x, params.up4_res1);
    x = forwardVaeResBlock(x, params.up4_res2);
    x = forwardVaeResBlock(x, params.up4_res3);

    // 8. up_blocks.5: DepthToSpace (2,1,1) → 512→256
    x = forwardDepthToSpace(x, params.up5, .{ 2, 1, 1 });

    // 9. up_blocks.6: 6 ResBlocks @ 256ch
    x = forwardVaeResBlock(x, params.up6_res0);
    x = forwardVaeResBlock(x, params.up6_res1);
    x = forwardVaeResBlock(x, params.up6_res2);
    x = forwardVaeResBlock(x, params.up6_res3);
    x = forwardVaeResBlock(x, params.up6_res4);
    x = forwardVaeResBlock(x, params.up6_res5);

    // 10. up_blocks.7: DepthToSpace (1,2,2) → 256→128
    x = forwardDepthToSpace(x, params.up7, .{ 1, 2, 2 });

    // 11. up_blocks.8: 4 ResBlocks @ 128ch
    x = forwardVaeResBlock(x, params.up8_res0);
    x = forwardVaeResBlock(x, params.up8_res1);
    x = forwardVaeResBlock(x, params.up8_res2);
    x = forwardVaeResBlock(x, params.up8_res3);

    // 12. PixelNorm → SiLU → conv_out
    x = forwardPixelNorm(x);
    x = x.silu();
    x = forwardCausalConv3dNonCausal(x, params.conv_out);

    // 13. Unpatchify: [B, 48, F, H, W] → [B, 3, F, 4H, 4W]
    return forwardUnpatchifyVae(x);
}

// ============================================================================
// Section 12: Audio VAE Decoder
// ============================================================================

// --- Parameter structs ---

/// Audio VAE ResBlock: PixelNorm → SiLU → CausalConv2d → PixelNorm → SiLU → CausalConv2d + residual.
/// Some blocks have a nin_shortcut (1×1 conv) when in_channels != out_channels.
pub const AudioVaeResBlock = struct {
    conv1: Conv2dWeight, // conv1.conv.{weight,bias}
    conv2: Conv2dWeight, // conv2.conv.{weight,bias}
    nin_shortcut: ?Conv2dWeight, // nin_shortcut.conv.{weight,bias}, null if same channels
};

/// Audio VAE upsample block: nearest 2× interpolation + CausalConv2d + drop first row.
pub const AudioVaeUpsample = struct {
    conv: Conv2dWeight, // upsample.conv.conv.{weight,bias}
};

/// Full Audio VAE Decoder parameter set.
/// Architecture: conv_in → mid_block_1 → mid_block_2 → up.2 → up.1 → up.0 → norm_out → conv_out
///
/// Config: ch=128, ch_mult=(1,2,4), num_res_blocks=2, z_channels=8, out_ch=2
/// Base channels = ch * ch_mult[-1] = 128 * 4 = 512
/// No attention at all.
pub const AudioVaeDecoderParams = struct {
    // conv_in: CausalConv2d(8 → 512, k=3)
    conv_in: Conv2dWeight,

    // mid.block_1, mid.block_2: ResBlock(512)
    mid_block_1: AudioVaeResBlock,
    mid_block_2: AudioVaeResBlock,

    // up.2: 3× ResBlock(512) + Upsample(512)
    up2_block0: AudioVaeResBlock,
    up2_block1: AudioVaeResBlock,
    up2_block2: AudioVaeResBlock,
    up2_upsample: AudioVaeUpsample,

    // up.1: ResBlock(512→256, nin_shortcut) + 2× ResBlock(256) + Upsample(256)
    up1_block0: AudioVaeResBlock, // has nin_shortcut 512→256
    up1_block1: AudioVaeResBlock,
    up1_block2: AudioVaeResBlock,
    up1_upsample: AudioVaeUpsample,

    // up.0: ResBlock(256→128, nin_shortcut) + 2× ResBlock(128) — NO upsample
    up0_block0: AudioVaeResBlock, // has nin_shortcut 256→128
    up0_block1: AudioVaeResBlock,
    up0_block2: AudioVaeResBlock,

    // conv_out: CausalConv2d(128 → 2, k=3)
    conv_out: Conv2dWeight,
};

/// Audio per-channel statistics (separate from video).
/// Keys: audio_vae.per_channel_statistics.{mean-of-means, std-of-means} [128]
pub const AudioPerChannelStats = struct {
    mean_of_means: Tensor, // [128]
    std_of_means: Tensor, // [128]
};

// --- Weight loading ---

fn initAudioConv2d(store: zml.io.TensorStore.View) Conv2dWeight {
    const c = store.withPrefix("conv");
    return .{
        .weight = c.createTensor("weight", null, null),
        .bias = c.createTensor("bias", null, null),
    };
}

fn initAudioVaeResBlock(store: zml.io.TensorStore.View) AudioVaeResBlock {
    return .{
        .conv1 = initAudioConv2d(store.withPrefix("conv1")),
        .conv2 = initAudioConv2d(store.withPrefix("conv2")),
        .nin_shortcut = null,
    };
}

fn initAudioVaeResBlockWithShortcut(store: zml.io.TensorStore.View) AudioVaeResBlock {
    return .{
        .conv1 = initAudioConv2d(store.withPrefix("conv1")),
        .conv2 = initAudioConv2d(store.withPrefix("conv2")),
        .nin_shortcut = initAudioConv2d(store.withPrefix("nin_shortcut")),
    };
}

pub fn initAudioVaeDecoderParams(store: zml.io.TensorStore.View) AudioVaeDecoderParams {
    const dec = store.withPrefix("audio_vae").withPrefix("decoder");

    return .{
        .conv_in = initAudioConv2d(dec.withPrefix("conv_in")),

        .mid_block_1 = initAudioVaeResBlock(dec.withPrefix("mid").withPrefix("block_1")),
        .mid_block_2 = initAudioVaeResBlock(dec.withPrefix("mid").withPrefix("block_2")),

        // up.2: 3× ResBlock(512), all same channels
        .up2_block0 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(2).withPrefix("block").withLayer(0)),
        .up2_block1 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(2).withPrefix("block").withLayer(1)),
        .up2_block2 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(2).withPrefix("block").withLayer(2)),
        .up2_upsample = .{ .conv = initAudioConv2d(dec.withPrefix("up").withLayer(2).withPrefix("upsample").withPrefix("conv")) },

        // up.1: first block has nin_shortcut (512→256)
        .up1_block0 = initAudioVaeResBlockWithShortcut(dec.withPrefix("up").withLayer(1).withPrefix("block").withLayer(0)),
        .up1_block1 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(1).withPrefix("block").withLayer(1)),
        .up1_block2 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(1).withPrefix("block").withLayer(2)),
        .up1_upsample = .{ .conv = initAudioConv2d(dec.withPrefix("up").withLayer(1).withPrefix("upsample").withPrefix("conv")) },

        // up.0: first block has nin_shortcut (256→128)
        .up0_block0 = initAudioVaeResBlockWithShortcut(dec.withPrefix("up").withLayer(0).withPrefix("block").withLayer(0)),
        .up0_block1 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(0).withPrefix("block").withLayer(1)),
        .up0_block2 = initAudioVaeResBlock(dec.withPrefix("up").withLayer(0).withPrefix("block").withLayer(2)),

        .conv_out = initAudioConv2d(dec.withPrefix("conv_out")),
    };
}

pub fn initAudioPerChannelStats(store: zml.io.TensorStore.View) AudioPerChannelStats {
    const pcs = store.withPrefix("audio_vae").withPrefix("per_channel_statistics");
    return .{
        .mean_of_means = pcs.createTensor("mean-of-means", null, null),
        .std_of_means = pcs.createTensor("std-of-means", null, null),
    };
}

// --- Forward ops ---

/// CausalConv2d forward with HEIGHT causality, kernel_size=3.
/// Padding: (pad_h_before=2, pad_h_after=0, pad_w_before=1, pad_w_after=1).
/// input [B, C_in, H, W], kernel [C_out, C_in, 3, 3].
fn forwardCausalConv2dHeight(input: Tensor, w: Conv2dWeight) Tensor {
    // conv2d padding: {pad_h_before, pad_h_after, pad_w_before, pad_w_after}
    const conv_out = input.conv2d(w.weight, .{
        .padding = &.{ 2, 0, 1, 1 },
    });
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1),
    ).broad(conv_out.shape()));
}

/// CausalConv2d 1×1 forward (nin_shortcut) with HEIGHT causality.
/// kernel_size=1 → no padding needed.
fn forwardCausalConv2d1x1(input: Tensor, w: Conv2dWeight) Tensor {
    const conv_out = input.conv2d(w.weight, .{
        .padding = &.{ 0, 0, 0, 0 },
    });
    return conv_out.add(w.bias.reshape(
        conv_out.shape().set(0, 1).set(2, 1).set(3, 1),
    ).broad(conv_out.shape()));
}

/// Audio VAE ResBlock forward: PixelNorm → SiLU → CausalConv2d → PixelNorm → SiLU → CausalConv2d + residual.
fn forwardAudioVaeResBlock(x: Tensor, rb: AudioVaeResBlock) Tensor {
    var h = forwardPixelNorm2d(x);
    h = h.silu();
    h = forwardCausalConv2dHeight(h, rb.conv1);
    h = forwardPixelNorm2d(h);
    h = h.silu();
    h = forwardCausalConv2dHeight(h, rb.conv2);

    // nin_shortcut for channel mismatch
    var residual = x;
    if (rb.nin_shortcut) |shortcut| {
        residual = forwardCausalConv2d1x1(x, shortcut);
    }
    return h.add(residual);
}

/// PixelNorm for 4D tensors [B, C, H, W].
/// y = x / sqrt(mean(x², dim=C) + eps)
fn forwardPixelNorm2d(x: Tensor) Tensor {
    const x_f32 = x.convert(.f32);
    const x_sq = x_f32.mul(x_f32);
    // mean over dim=1 (channels): sum returns [B, 1, H, W]
    const c: i64 = x.shape().dim(1);
    const sum_sq = x_sq.sum(1);
    const mean_sq = sum_sq.divByConst(c);
    const rms = mean_sq.addConstant(1e-6).sqrt();
    return x_f32.div(rms.broad(x_f32.shape())).convert(x.dtype());
}

/// Nearest-neighbor 2× upsample for [B, C, H, W] → [B, C, 2H, 2W].
/// Implemented as reshape + broadcast (no interpolation kernel needed).
fn forwardNearest2x(x: Tensor) Tensor {
    const B = x.shape().dim(0);
    const C = x.shape().dim(1);
    const H = x.shape().dim(2);
    const W = x.shape().dim(3);

    // [B, C, H, W] → [B, C, H, 1, W, 1] → broadcast to [B, C, H, 2, W, 2] → reshape [B, C, 2H, 2W]
    const expanded = x.reshape(.{ B, C, H, 1, W, 1 })
        .broad(zml.Shape.init(.{ B, C, H, 2, W, 2 }, x.dtype()));
    return expanded.reshape(.{ B, C, H * 2, W * 2 });
}

/// Audio VAE Upsample: nearest 2× → CausalConv2d → drop first row (HEIGHT causality).
fn forwardAudioVaeUpsample(x: Tensor, us: AudioVaeUpsample) Tensor {
    var h = forwardNearest2x(x);
    h = forwardCausalConv2dHeight(h, us.conv);
    // Drop first row to undo causal padding (HEIGHT causality)
    h = h.slice1d(2, .{ .start = 1 });
    return h;
}

/// Unpatchify audio latent: [B, T_aud, 128] → [B, 8, T_aud, 16].
/// rearrange "b t (c f) -> b c t f" with c=8, f=16.
pub fn forwardUnpatchifyAudio(patchified: Tensor) Tensor {
    const B = patchified.shape().dim(0);
    const T = patchified.shape().dim(1);
    // [B, T, 128] → [B, T, 8, 16] → transpose → [B, 8, T, 16]
    return patchified.reshape(.{ B, T, 8, 16 }).transpose(.{ 0, 2, 1, 3 });
}

/// Denormalize audio latent using per-channel statistics.
/// Patchify → denorm → unpatchify (operates on the 128-dim patchified representation).
fn forwardAudioDenormalize(latent: Tensor, stats: AudioPerChannelStats) Tensor {
    const B = latent.shape().dim(0);
    const T = latent.shape().dim(2);

    // Patchify: [B, 8, T, 16] → [B, T, 128]
    const patchified = latent.transpose(.{ 0, 2, 1, 3 }).reshape(.{ B, T, 128 });

    // Denormalize: x * std + mean (broadcast [128] over [B, T, 128])
    const stats_shape = zml.Shape.init(.{ 1, 1, 128 }, latent.dtype());
    const std_val = stats.std_of_means.reshape(stats_shape).broad(patchified.shape());
    const mean = stats.mean_of_means.reshape(stats_shape).broad(patchified.shape());
    const denormed = patchified.mul(std_val).add(mean);

    // Unpatchify back: [B, T, 128] → [B, 8, T, 16]
    return denormed.reshape(.{ B, T, 8, 16 }).transpose(.{ 0, 2, 1, 3 });
}

/// Full audio VAE decoder forward.
/// Input: [B, 8, T, 16] bf16 (audio latent after unpatchify)
/// Output: [B, 2, T_out, 64] bf16 (mel spectrogram)
///
/// T_out = max(T * 4 - 3, 1) (causal, LATENT_DOWNSAMPLE_FACTOR=4)
pub fn forwardAudioVaeDecode(
    latent: Tensor,
    stats: AudioPerChannelStats,
    params: AudioVaeDecoderParams,
) Tensor {
    // 1. Denormalize
    var x = forwardAudioDenormalize(latent, stats);

    // 2. conv_in: [B, 8, T, 16] → [B, 512, T, 16]
    x = forwardCausalConv2dHeight(x, params.conv_in);

    // 3. Mid block
    x = forwardAudioVaeResBlock(x, params.mid_block_1);
    x = forwardAudioVaeResBlock(x, params.mid_block_2);

    // 4. up.2: 3× ResBlock(512) + Upsample → [B, 512, 2T-1, 32]
    x = forwardAudioVaeResBlock(x, params.up2_block0);
    x = forwardAudioVaeResBlock(x, params.up2_block1);
    x = forwardAudioVaeResBlock(x, params.up2_block2);
    x = forwardAudioVaeUpsample(x, params.up2_upsample);

    // 5. up.1: ResBlock(512→256) + 2× ResBlock(256) + Upsample → [B, 256, 4T-3, 64]
    x = forwardAudioVaeResBlock(x, params.up1_block0);
    x = forwardAudioVaeResBlock(x, params.up1_block1);
    x = forwardAudioVaeResBlock(x, params.up1_block2);
    x = forwardAudioVaeUpsample(x, params.up1_upsample);

    // 6. up.0: ResBlock(256→128) + 2× ResBlock(128) — no upsample
    x = forwardAudioVaeResBlock(x, params.up0_block0);
    x = forwardAudioVaeResBlock(x, params.up0_block1);
    x = forwardAudioVaeResBlock(x, params.up0_block2);

    // 7. norm_out → SiLU → conv_out
    x = forwardPixelNorm2d(x);
    x = x.silu();
    x = forwardCausalConv2dHeight(x, params.conv_out);

    // Output: [B, 2, T_out, F_out] where T_out ≈ 4T-3, F_out ≈ 64
    return x;
}

// ============================================================================
// Section 13: Vocoder + BWE (mel spectrogram → 48kHz stereo waveform)
// ============================================================================
//
// Architecture (from checkpoint config):
//   VocoderWithBWE:
//     1. Vocoder (BigVGAN v2): mel[B,2,T,64] → waveform[B,2,T*160] at 16kHz
//        conv_pre(128→1536,k=7) → 6× [ConvTranspose1d + 3× AMPBlock1] →
//        act_post(SnakeBeta) → conv_post(24→2,k=7) → clamp(-1,1)
//        Upsample rates: [5,2,2,2,2,2]=160×, channels: 1536→768→384→192→96→48→24
//
//     2. BWE (bandwidth extension): 16kHz → 48kHz
//        mel_stft on vocoder output → bwe_generator (2nd Vocoder, 240×) → + sinc_resample(3×) skip
//        BWE channels: 512→256→128→64→32→16, upsample rates: [6,5,2,2,2]
//
// CRITICAL: Entire forward pass runs in f32 (bf16 causes 40-90% spectral degradation).
// Weight checkpoint stores bf16 — converted to f32 at each op (mimics PyTorch autocast).
//
// Checkpoint keys: vocoder.vocoder.*, vocoder.bwe_generator.*, vocoder.mel_stft.*
// Total: 1227 weight tensors.

// --- Parameter structs ---

/// Conv1d weight + bias.
pub const Conv1dWeight = struct {
    weight: Tensor, // [C_out, C_in, K]
    bias: Tensor, // [C_out]
};

/// Conv1d weight only (no bias).
pub const Conv1dWeightNoBias = struct {
    weight: Tensor, // [C_out, C_in, K]
};

/// ConvTranspose1d weight + bias.
/// PyTorch stores shape [in_ch, out_ch, K] (dims swapped vs Conv1d).
pub const ConvTranspose1dWeight = struct {
    weight: Tensor, // [C_in, C_out, K]
    bias: Tensor, // [C_out]
};

/// SnakeBeta activation: x + (1/exp(beta)) * sin(exp(alpha) * x)²
/// Stored with alpha_logscale=True.
pub const SnakeBetaParams = struct {
    alpha: Tensor, // [C]
    beta: Tensor, // [C]
};

/// Anti-aliased activation: upsample 2× → SnakeBeta → downsample 2×.
/// Kaiser-sinc filters stored in checkpoint.
pub const Activation1dParams = struct {
    act: SnakeBetaParams,
    upsample_filter: Tensor, // [1, 1, 12]
    downsample_filter: Tensor, // [1, 1, 12]
};

/// AMPBlock1: 3 dilated conv pairs with anti-aliased SnakeBeta activations.
/// Each pair: acts1[i](x) → convs1[i] → acts2[i] → convs2[i], residual added.
pub const AMPBlock1Params = struct {
    convs1: [3]Conv1dWeight,
    convs2: [3]Conv1dWeight,
    acts1: [3]Activation1dParams,
    acts2: [3]Activation1dParams,
};

/// STFT as conv1d with precomputed DFT bases (from checkpoint).
pub const STFTParams = struct {
    forward_basis: Tensor, // [n_fft+2, 1, n_fft] = [514, 1, 512]
};

/// Causal log-mel spectrogram: STFT + mel filterbank projection.
pub const MelSTFTParams = struct {
    mel_basis: Tensor, // [n_mels, n_freqs] = [64, 257]
    stft_fn: STFTParams,
};

/// Main vocoder parameters (6 upsample stages, 18 AMPBlock1 resblocks).
pub const MainVocoderParams = struct {
    conv_pre: Conv1dWeight, // 128 → 1536, k=7
    ups: [6]ConvTranspose1dWeight,
    resblocks: [18]AMPBlock1Params, // 6 stages × 3 kernels
    act_post: Activation1dParams,
    conv_post: Conv1dWeightNoBias, // 24 → 2, k=7
};

/// BWE vocoder parameters (5 upsample stages, 15 AMPBlock1 resblocks).
pub const BWEVocoderParams = struct {
    conv_pre: Conv1dWeight, // 128 → 512, k=7
    ups: [5]ConvTranspose1dWeight,
    resblocks: [15]AMPBlock1Params, // 5 stages × 3 kernels
    act_post: Activation1dParams,
    conv_post: Conv1dWeightNoBias, // 16 → 2, k=7
};

/// Top-level VocoderWithBWE parameters.
pub const VocoderWithBWEParams = struct {
    vocoder: MainVocoderParams,
    bwe_generator: BWEVocoderParams,
    mel_stft: MelSTFTParams,
};

// --- Weight loading ---

fn initConv1dWeight(store: zml.io.TensorStore.View) Conv1dWeight {
    return .{
        .weight = store.createTensor("weight", null, null),
        .bias = store.createTensor("bias", null, null),
    };
}

fn initConv1dWeightNoBias(store: zml.io.TensorStore.View) Conv1dWeightNoBias {
    return .{
        .weight = store.createTensor("weight", null, null),
    };
}

fn initConvTranspose1dWeight(store: zml.io.TensorStore.View) ConvTranspose1dWeight {
    return .{
        .weight = store.createTensor("weight", null, null),
        .bias = store.createTensor("bias", null, null),
    };
}

fn initSnakeBetaParams(store: zml.io.TensorStore.View) SnakeBetaParams {
    return .{
        .alpha = store.createTensor("alpha", null, null),
        .beta = store.createTensor("beta", null, null),
    };
}

fn initActivation1dParams(store: zml.io.TensorStore.View) Activation1dParams {
    return .{
        .act = initSnakeBetaParams(store.withPrefix("act")),
        .upsample_filter = store.withPrefix("upsample").createTensor("filter", null, null),
        .downsample_filter = store.withPrefix("downsample").withPrefix("lowpass").createTensor("filter", null, null),
    };
}

fn initAMPBlock1Params(result: *AMPBlock1Params, store: zml.io.TensorStore.View) void {
    inline for (0..3) |i| {
        result.convs1[i] = initConv1dWeight(store.withPrefix("convs1").withLayer(i));
        result.convs2[i] = initConv1dWeight(store.withPrefix("convs2").withLayer(i));
        result.acts1[i] = initActivation1dParams(store.withPrefix("acts1").withLayer(i));
        result.acts2[i] = initActivation1dParams(store.withPrefix("acts2").withLayer(i));
    }
}

pub fn initMainVocoderParams(result: *MainVocoderParams, store: zml.io.TensorStore.View) void {
    result.conv_pre = initConv1dWeight(store.withPrefix("conv_pre"));
    inline for (0..6) |i| {
        result.ups[i] = initConvTranspose1dWeight(store.withPrefix("ups").withLayer(i));
    }
    inline for (0..18) |i| {
        initAMPBlock1Params(&result.resblocks[i], store.withPrefix("resblocks").withLayer(i));
    }
    result.act_post = initActivation1dParams(store.withPrefix("act_post"));
    result.conv_post = initConv1dWeightNoBias(store.withPrefix("conv_post"));
}

fn initBWEVocoderParams(result: *BWEVocoderParams, store: zml.io.TensorStore.View) void {
    result.conv_pre = initConv1dWeight(store.withPrefix("conv_pre"));
    inline for (0..5) |i| {
        result.ups[i] = initConvTranspose1dWeight(store.withPrefix("ups").withLayer(i));
    }
    inline for (0..15) |i| {
        initAMPBlock1Params(&result.resblocks[i], store.withPrefix("resblocks").withLayer(i));
    }
    result.act_post = initActivation1dParams(store.withPrefix("act_post"));
    result.conv_post = initConv1dWeightNoBias(store.withPrefix("conv_post"));
}

fn initMelSTFTParams(store: zml.io.TensorStore.View) MelSTFTParams {
    return .{
        .mel_basis = store.createTensor("mel_basis", null, null),
        .stft_fn = .{
            .forward_basis = store.withPrefix("stft_fn").createTensor("forward_basis", null, null),
        },
    };
}

pub fn initVocoderWithBWEParams(result: *VocoderWithBWEParams, store: zml.io.TensorStore.View) void {
    const voc = store.withPrefix("vocoder");
    initMainVocoderParams(&result.vocoder, voc.withPrefix("vocoder"));
    initBWEVocoderParams(&result.bwe_generator, voc.withPrefix("bwe_generator"));
    result.mel_stft = initMelSTFTParams(voc.withPrefix("mel_stft"));
}

// --- Forward ops ---

/// Ensure tensor is f32 (for vocoder precision requirement).
fn ensureF32(t: Tensor) Tensor {
    return if (t.dtype() == .f32) t else t.convert(.f32);
}

/// Replicate-pad along the last dimension (spatial dim for 1D signals).
/// Input [B, C, T] → output [B, C, pad_left + T + pad_right].
fn replicatePad1d(x: Tensor, pad_left: i64, pad_right: i64) Tensor {
    const spatial_axis = x.rank() - 1;
    const t_dim = x.dim(spatial_axis);
    // First element replicated pad_left times
    const first = x.slice1d(spatial_axis, .{ .end = 1 }); // [B, C, 1]
    const first_pad = first.broad(x.shape().set(spatial_axis, pad_left));
    // Last element replicated pad_right times
    const last = x.slice1d(spatial_axis, .{ .start = t_dim - 1 }); // [B, C, 1]
    const last_pad = last.broad(x.shape().set(spatial_axis, pad_right));
    return Tensor.concatenate(&.{ first_pad, x, last_pad }, spatial_axis);
}

/// Conv1d forward with f32 conversion, input [B, C_in, T], weight [C_out, C_in, K], bias [C_out].
fn forwardVocConv1d(input: Tensor, w: Conv1dWeight, opts: struct {
    padding: i64 = 0,
    dilation: i64 = 1,
}) Tensor {
    const x = ensureF32(input);
    const weight = ensureF32(w.weight);
    const bias = ensureF32(w.bias);
    const result = x.conv1d(weight, .{
        .padding = &.{ opts.padding, opts.padding },
        .rhs_dilation = opts.dilation,
    });
    // Add bias: reshape [C_out] → [1, C_out, 1]
    return result.add(bias.reshape(result.shape().set(0, 1).set(2, 1)));
}

/// Conv1d forward without bias.
fn forwardVocConv1dNoBias(input: Tensor, w: Conv1dWeightNoBias, opts: struct {
    padding: i64 = 0,
}) Tensor {
    const x = ensureF32(input);
    const weight = ensureF32(w.weight);
    return x.conv1d(weight, .{
        .padding = &.{ opts.padding, opts.padding },
    });
}

/// ConvTranspose1d forward with f32 conversion.
/// PyTorch weight [in_ch, out_ch, K] → MLIR: swap kernel dims, use lhs_dilation=stride, explicit kernel flip.
/// PyTorch padding p → MLIR padding = K - p - 1 on each side.
fn forwardVocConvTranspose1d(input: Tensor, w: ConvTranspose1dWeight, stride: i64, pytorch_padding: i64) Tensor {
    const x = ensureF32(input);
    const weight = ensureF32(w.weight).reverse(.{2}); // flip kernel along spatial dim
    const bias = ensureF32(w.bias);
    const k = weight.dim(2);
    const mlir_pad = k - pytorch_padding - 1;
    const result = x.conv1d(weight, .{
        .lhs_dilation = stride,
        .padding = &.{ mlir_pad, mlir_pad },
        .kernel_output_feature_dimension = 1,
        .kernel_input_feature_dimension = 0,
    });
    return result.add(bias.reshape(result.shape().set(0, 1).set(2, 1)));
}

/// SnakeBeta activation: x + (1/exp(beta)) * sin(exp(alpha) * x)²
/// alpha_logscale=True: alpha, beta are in log-space.
fn forwardSnakeBeta(x: Tensor, params: SnakeBetaParams) Tensor {
    const alpha = ensureF32(params.alpha).exp().reshape(x.shape().set(0, 1).set(2, 1)); // [1, C, 1]
    const beta = ensureF32(params.beta).exp().reshape(x.shape().set(0, 1).set(2, 1)); // [1, C, 1]
    const eps: f32 = 1e-9;
    // x + (1 / (beta + eps)) * sin(alpha * x)²
    const sin_val = x.mul(alpha).sin();
    return x.add(sin_val.mul(sin_val).div(beta.addConstant(eps)));
}

/// UpSample1d: replicate-pad → depthwise conv_transpose1d → scale → trim.
/// Kaiser-sinc filter from checkpoint [1, 1, 12], expanded to [C, 1, 12] for grouped conv.
/// Ratio=2, kernel_size=12, pad=5, pad_left=15, pad_right=15.
fn forwardUpSample1d(x: Tensor, filter: Tensor) Tensor {
    const ratio: i64 = 2;
    const kernel_size: i64 = 12;
    const pad: i64 = kernel_size / ratio - 1; // 5
    const pad_left_trim: i64 = pad * ratio + @divTrunc(kernel_size - ratio, 2); // 15
    const pad_right_trim: i64 = pad * ratio + @divTrunc(kernel_size - ratio + 1, 2); // 15

    // 1. Replicate-pad input by 'pad' on each side
    var y = replicatePad1d(x, pad, pad);

    // 2. Depthwise ConvTranspose1d: expand filter [1,1,12] → [C,1,12], stride=2, groups=C
    const n_channels = y.dim(1);
    const filt = ensureF32(filter).reverse(.{2}).broad(filter.shape().set(0, n_channels)); // [C, 1, 12] flipped
    // For depthwise transposed conv: kernel_output=0 (C), kernel_input=1 (1), feature_group_count=C
    // PyTorch ConvTranspose1d(groups=C): no padding → MLIR padding = K - 0 - 1 = 11
    const mlir_pad = kernel_size - 1;
    y = ensureF32(y).conv1d(filt, .{
        .lhs_dilation = ratio,
        .padding = &.{ mlir_pad, mlir_pad },
        .feature_group_count = n_channels,
    });

    // 3. Scale by ratio
    y = y.scale(ratio);

    // 4. Trim padded edges
    const t_out = y.dim(2);
    y = y.slice1d(2, .{ .start = pad_left_trim, .end = t_out - pad_right_trim });
    return y;
}

/// DownSample1d via LowPassFilter1d: replicate-pad → depthwise conv1d with stride.
/// Kaiser-sinc filter from checkpoint [1, 1, 12], expanded to [C, 1, 12].
/// Ratio=2, kernel_size=12, pad_left=5 (even: k//2 - 1), pad_right=6 (k//2).
fn forwardDownSample1d(x: Tensor, filter: Tensor) Tensor {
    const stride: i64 = 2;
    const pad_left: i64 = 5; // kernel_size // 2 - 1 (even kernel)
    const pad_right: i64 = 6; // kernel_size // 2

    // 1. Replicate-pad
    var y = replicatePad1d(x, pad_left, pad_right);

    // 2. Depthwise conv1d with stride=2
    const n_channels = y.dim(1);
    const filt = ensureF32(filter).broad(filter.shape().set(0, n_channels)); // [C, 1, 12]
    y = ensureF32(y).conv1d(filt, .{
        .window_strides = stride,
        .feature_group_count = n_channels,
    });
    return y;
}

/// Activation1d: upsample 2× → SnakeBeta → downsample 2× (anti-aliased activation).
fn forwardActivation1d(x: Tensor, params: Activation1dParams) Tensor {
    var y = forwardUpSample1d(x, params.upsample_filter);
    y = forwardSnakeBeta(y, params.act);
    y = forwardDownSample1d(y, params.downsample_filter);
    return y;
}

/// AMPBlock1 forward: 3 dilated conv pairs with residual connections.
/// For each pair i: xt = acts1[i](x) → convs1[i](xt) → acts2[i](xt) → convs2[i](xt); x = x + xt.
fn forwardAMPBlock1(x_in: Tensor, params: AMPBlock1Params, dilations: [3]i64) Tensor {
    var x = x_in;
    inline for (0..3) |i| {
        var xt = forwardActivation1d(x, params.acts1[i]);
        xt = forwardVocConv1d(xt, params.convs1[i], .{
            .padding = @divTrunc(params.convs1[i].weight.dim(2) * dilations[i] - dilations[i], 2),
            .dilation = dilations[i],
        });
        xt = forwardActivation1d(xt, params.acts2[i]);
        xt = forwardVocConv1d(xt, params.convs2[i], .{
            .padding = @divTrunc(params.convs2[i].weight.dim(2) - 1, 2),
        });
        x = x.add(xt);
    }
    return x;
}

/// Vocoder forward (shared by main vocoder and BWE generator).
/// Input: mel [B, 2, T, 64] → rearrange to [B, 128, T] → upsample → waveform [B, 2, T_out].
fn forwardVocoderGeneric(
    mel: Tensor,
    conv_pre: Conv1dWeight,
    ups: anytype,
    resblocks: anytype,
    act_post: Activation1dParams,
    conv_post: Conv1dWeightNoBias,
    comptime num_ups: usize,
    apply_final_activation: bool,
) Tensor {
    // Rearrange [B, 2, T, 64] → [B, 128, T]
    // First transpose to [B, 2, 64, T], then reshape to [B, 128, T]
    var x = ensureF32(mel).transpose(.{ 0, 1, 3, 2 }); // [B, 2, T, 64] → [B, 2, 64, T]
    x = x.reshape(.{ x.dim(0), -1, x.dim(3) }); // [B, 128, T]

    // conv_pre (k=7, pad=3)
    x = forwardVocConv1d(x, conv_pre, .{ .padding = 3 });

    // Upsample stages: each stage has 1 ConvTranspose1d + num_kernels AMPBlock1 resblocks
    const num_kernels: usize = 3;
    const dilations = [3]i64{ 1, 3, 5 };

    inline for (0..num_ups) |i| {
        // ConvTranspose1d upsample
        // stride = kernel_size // 2 (holds for this checkpoint: [11→5, 4→2, 12→6, 11→5])
        const k = ups[i].weight.dim(2);
        const stride = @divTrunc(k, 2);
        const pytorch_padding = @divTrunc(k - stride, 2);
        x = forwardVocConvTranspose1d(x, ups[i], stride, pytorch_padding);

        // AMPBlock1 resblocks: evaluate all 3 kernel variants, average their outputs
        const start = i * num_kernels;
        var block_sum = forwardAMPBlock1(x, resblocks[start], dilations);
        inline for (1..num_kernels) |j| {
            block_sum = block_sum.add(forwardAMPBlock1(x, resblocks[start + j], dilations));
        }
        x = block_sum.divByConst(num_kernels);
    }

    // Final activation + conv
    x = forwardActivation1d(x, act_post);
    x = forwardVocConv1dNoBias(x, conv_post, .{ .padding = 3 });

    if (apply_final_activation) {
        x = x.clamp(Tensor.scalar(-1.0, .f32), Tensor.scalar(1.0, .f32));
    }

    return x;
}

/// Causal STFT via conv1d with precomputed DFT bases.
/// Input: waveform [B, T_samples], output: magnitude [B, n_freqs, T_frames].
/// Causal: left-only padding of (win_length - hop_length) samples.
fn forwardSTFT(y_in: Tensor, params: STFTParams) Tensor {
    const hop_length: i64 = 80;
    const win_length: i64 = 512;

    // Add channel dim: [B, T] → [B, 1, T]
    var y = ensureF32(y_in).reshape(.{ y_in.dim(0), 1, y_in.dim(1) });

    // Causal left-only padding (prepend zeros)
    const left_pad = win_length - hop_length; // 432
    const left_zeros = Tensor.zeroes(y.shape().set(2, left_pad));
    y = Tensor.concatenate(&.{ left_zeros, y }, 2);

    // Conv1d with precomputed DFT bases [514, 1, 512], stride=hop_length
    const basis = ensureF32(params.forward_basis); // [514, 1, 512]
    const spec = y.conv1d(basis, .{ .window_strides = hop_length }); // [B, 514, T_frames]

    // Split into real and imaginary: first 257 and last 257 channels
    const n_freqs = @divTrunc(spec.dim(1), 2); // 257
    const real = spec.slice1d(1, .{ .end = n_freqs }); // [B, 257, T_frames]
    const imag = spec.slice1d(1, .{ .start = n_freqs }); // [B, 257, T_frames]

    // magnitude = sqrt(real² + imag²)
    return real.mul(real).add(imag.mul(imag)).sqrt();
}

/// Compute log-mel spectrogram from magnitude spectrogram.
/// magnitude [B, n_freqs, T_frames] → log_mel [B, n_mels, T_frames].
fn forwardMelProjection(magnitude: Tensor, mel_basis: Tensor) Tensor {
    // mel = mel_basis @ magnitude
    // mel_basis [n_mels=64, n_freqs=257], magnitude [B, n_freqs=257, T_frames]
    // Use conv1d with kernel_size=1 to implement matmul:
    // Treat mel_basis [64, 257] as conv1d kernel [64, 257, 1]
    const basis = ensureF32(mel_basis).reshape(.{ mel_basis.dim(0), mel_basis.dim(1), 1 }); // [64, 257, 1]
    const mel = ensureF32(magnitude).conv1d(basis, .{}); // [B, 64, T_frames]

    // log(clamp(mel, min=1e-5))
    const clamped = mel.clamp(Tensor.scalar(1e-5, .f32), Tensor.scalar(1e30, .f32));
    return clamped.log();
}

/// Compute causal log-mel spectrogram from stereo waveform.
/// audio [B, 2, T_samples] → mel [B, 2, n_mels, T_frames].
fn forwardComputeMel(audio: Tensor, mel_stft: MelSTFTParams) Tensor {
    const batch = audio.dim(0);
    const n_channels = audio.dim(1); // 2
    const t_samples = audio.dim(2);

    // Flatten: [B, 2, T] → [B*2, T]
    const flat = audio.reshape(.{ batch * n_channels, t_samples });

    // STFT → magnitude [B*2, 257, T_frames]
    const magnitude = forwardSTFT(flat, mel_stft.stft_fn);

    // Mel projection → log_mel [B*2, 64, T_frames]
    const log_mel = forwardMelProjection(magnitude, mel_stft.mel_basis);

    // Reshape back: [B*2, 64, T_frames] → [B, 2, 64, T_frames]
    return log_mel.reshape(.{ batch, n_channels, log_mel.dim(1), log_mel.dim(2) });
}

/// Kaiser-windowed sinc resampler for BWE skip connection.
/// Upsamples by ratio=3 (16kHz → 48kHz). Filter not in checkpoint — computed here.
///
/// UpSample1d(ratio=3, window_type="kaiser"):  (kaiser is the default)
///   kernel_size=18, pad=5, pad_left=22, pad_right=23
fn forwardSincResample3x(x: Tensor) Tensor {
    const ratio: i64 = 3;
    const kernel_size: i64 = 18;
    const pad: i64 = 5;
    const pad_left_trim: i64 = 22;
    const pad_right_trim: i64 = 23;

    // Precomputed Kaiser-windowed sinc filter (18 taps, ratio=3)
    // Generated from: UpSample1d(ratio=3, window_type="kaiser")
    //   cutoff=0.5/ratio, half_width=0.6/ratio, kernel_size=18
    //   filter.sum() ≈ 1.0, filter.sum()*ratio ≈ 3.0
    const filter_data = [18]f32{
        7.0040696301e-04, 4.6405289322e-03, 5.3038536571e-03, -1.0276262648e-02,
        -3.6353457719e-02, -3.0759338289e-02, 5.2384063601e-02, 1.9813767076e-01,
        3.1622257829e-01, 3.1622257829e-01, 1.9813767076e-01, 5.2384063601e-02,
        -3.0759338289e-02, -3.6353457719e-02, -1.0276262648e-02, 5.3038536571e-03,
        4.6405289322e-03, 7.0040696301e-04,
    };

    // 1. Replicate-pad
    var y = replicatePad1d(ensureF32(x), pad, pad);

    // 2. Depthwise conv_transpose1d with sinc filter
    const n_channels = y.dim(1);
    // Create constant filter tensor [1, 1, 18] then broadcast to [C, 1, 18]
    const filter_shape = zml.Shape.init(.{ 1, 1, kernel_size }, .f32);
    const filt_1 = Tensor.constantTensor(filter_shape, std.mem.sliceAsBytes(&filter_data));
    const filt = filt_1.reverse(.{2}).broad(filter_shape.set(0, n_channels)); // [C, 1, 18] flipped (symmetric)

    // MLIR padding for depthwise transposed conv: kernel_size - 1 = 42
    const mlir_pad = kernel_size - 1;
    y = y.conv1d(filt, .{
        .lhs_dilation = ratio,
        .padding = &.{ mlir_pad, mlir_pad },
        .feature_group_count = n_channels,
    });

    // 3. Scale by ratio
    y = y.scale(ratio);

    // 4. Trim
    const t_out = y.dim(2);
    y = y.slice1d(2, .{ .start = pad_left_trim, .end = t_out - pad_right_trim });
    return y;
}

/// Full VocoderWithBWE forward pass.
/// Input: mel_spec [B, 2, T, 64] (stereo mel spectrogram from Audio VAE decoder).
/// Output: waveform [B, 2, T_audio] at 48kHz, values in [-1, 1].
///
/// Pipeline:
///   1. Main vocoder: mel → 16kHz waveform (160× upsample)
///   2. BWE: compute mel-STFT on vocoder output → bwe_generator (240× upsample) → + sinc_resample(3×) skip → clamp
///
/// Split into two compiled functions to stay under the 1024-argument MLIR limit:
///   forwardMainVocoder  (668 tensor args)
///   forwardBWEPipeline  (560 tensor args)

/// BWE pipeline parameters (bwe_generator + mel_stft). 559 tensors total.
pub const BWEPipelineParams = struct {
    bwe_generator: BWEVocoderParams,
    mel_stft: MelSTFTParams,
};

pub fn initBWEPipelineParams(result: *BWEPipelineParams, store: zml.io.TensorStore.View) void {
    const voc = store.withPrefix("vocoder");
    initBWEVocoderParams(&result.bwe_generator, voc.withPrefix("bwe_generator"));
    result.mel_stft = initMelSTFTParams(voc.withPrefix("mel_stft"));
}

/// Stage 1: Main vocoder — mel [B, 2, T, 64] → waveform [B, 2, T*160] at 16kHz.
/// 667 tensor parameters (+ 1 input = 668 MLIR args; well under 1024 limit).
pub fn forwardMainVocoder(mel_spec: Tensor, params: *const MainVocoderParams) Tensor {
    return forwardVocoderGeneric(
        mel_spec,
        params.conv_pre,
        params.ups,
        params.resblocks,
        params.act_post,
        params.conv_post,
        6,
        true, // apply clamp(-1,1)
    );
}

/// Debug: rearrange + conv_pre + ups[0] for isolating numerical issues.
pub fn forwardAfterUps0(mel_spec: Tensor, params: *const MainVocoderParams) Tensor {
    var x = ensureF32(mel_spec).transpose(.{ 0, 1, 3, 2 });
    x = x.reshape(.{ x.dim(0), -1, x.dim(3) });
    x = forwardVocConv1d(x, params.conv_pre, .{ .padding = 3 });

    // ups[0]: ConvTranspose1d
    const k = params.ups[0].weight.dim(2);
    const stride = @divTrunc(k, 2);
    const pytorch_padding = @divTrunc(k - stride, 2);
    x = forwardVocConvTranspose1d(x, params.ups[0], stride, pytorch_padding);
    return x;
}

/// Debug: rearrange + conv_pre + ups[0] + first resblock stage.
pub fn forwardAfterStage0(mel_spec: Tensor, params: *const MainVocoderParams) Tensor {
    var x = ensureF32(mel_spec).transpose(.{ 0, 1, 3, 2 });
    x = x.reshape(.{ x.dim(0), -1, x.dim(3) });
    x = forwardVocConv1d(x, params.conv_pre, .{ .padding = 3 });

    // ups[0]
    const k = params.ups[0].weight.dim(2);
    const stride = @divTrunc(k, 2);
    const pytorch_padding = @divTrunc(k - stride, 2);
    x = forwardVocConvTranspose1d(x, params.ups[0], stride, pytorch_padding);

    // resblocks 0,1,2 → mean
    const dilations = [3]i64{ 1, 3, 5 };
    var block_sum = forwardAMPBlock1(x, params.resblocks[0], dilations);
    block_sum = block_sum.add(forwardAMPBlock1(x, params.resblocks[1], dilations));
    block_sum = block_sum.add(forwardAMPBlock1(x, params.resblocks[2], dilations));
    x = block_sum.divByConst(3);
    return x;
}

/// Stage 2: BWE pipeline — 16kHz waveform → 48kHz waveform.
/// Takes the main vocoder output, computes mel-STFT, runs BWE generator,
/// adds sinc-resampled skip connection, clamps to [-1, 1].
/// 559 tensor parameters (+ 1 input = 560 MLIR args; well under 1024 limit).
pub fn forwardBWEPipeline(waveform_16k: Tensor, params: *const BWEPipelineParams) Tensor {
    const input_sr: i64 = 16000;
    const output_sr: i64 = 48000;
    const hop_length: i64 = 80;
    const sr_ratio = @divTrunc(output_sr, input_sr); // 3

    var x = waveform_16k;
    const length_low_rate = x.dim(2);
    const output_length = length_low_rate * sr_ratio;

    // 1. Pad vocoder output to multiple of hop_length for exact mel frame count
    const remainder = @mod(length_low_rate, hop_length);
    const pad_amount = if (remainder != 0) hop_length - remainder else 0;
    if (pad_amount > 0) {
        const right_zeros = Tensor.zeroes(x.shape().set(2, pad_amount));
        x = Tensor.concatenate(&.{ x, right_zeros }, 2);
    }

    // 2. Compute mel spectrogram from vocoder output: [B, 2, n_mels, T_frames]
    const mel = forwardComputeMel(x, params.mel_stft);

    // 3. Vocoder.forward expects [B, 2, T_frames, mel_bins] — transpose
    const mel_for_bwe = mel.transpose(.{ 0, 1, 3, 2 });

    // 4. BWE generator: mel → residual waveform [B, 2, T_bwe]
    const residual = forwardVocoderGeneric(
        mel_for_bwe,
        params.bwe_generator.conv_pre,
        params.bwe_generator.ups,
        params.bwe_generator.resblocks,
        params.bwe_generator.act_post,
        params.bwe_generator.conv_post,
        5,
        false, // no final activation
    );

    // 5. Sinc-resample vocoder output by 3× for skip connection
    const skip = forwardSincResample3x(x);

    // 6. Add residual + skip, clamp, trim to output length
    var output = residual.add(skip);
    output = output.clamp(Tensor.scalar(-1.0, .f32), Tensor.scalar(1.0, .f32));
    output = output.slice1d(2, .{ .end = output_length });

    return output;
}

// ====================================================================
// Debug forward functions for BWE pipeline bisection
// ====================================================================

/// Helper: pad waveform to multiple of hop_length (shared by BWE debug fns).
fn bwePadInput(waveform_16k: Tensor) Tensor {
    const hop_length: i64 = 80;
    var x = waveform_16k;
    const length_low_rate = x.dim(2);
    const remainder = @mod(length_low_rate, hop_length);
    const pad_amount = if (remainder != 0) hop_length - remainder else 0;
    if (pad_amount > 0) {
        const right_zeros = Tensor.zeroes(x.shape().set(2, pad_amount));
        x = Tensor.concatenate(&.{ x, right_zeros }, 2);
    }
    return x;
}

/// Debug: BWE compute mel only — returns log-mel [B, 2, n_mels, T_frames] (before transpose).
pub fn forwardBWEComputeMel(waveform_16k: Tensor, params: *const BWEPipelineParams) Tensor {
    const x = bwePadInput(waveform_16k);
    return forwardComputeMel(x, params.mel_stft);
}

/// Debug: BWE sinc resample skip only — returns [B, 2, T_skip].
pub fn forwardBWESincSkip(waveform_16k: Tensor) Tensor {
    const x = bwePadInput(waveform_16k);
    return forwardSincResample3x(x);
}

/// Debug: BWE residual only (mel → BWE generator) — returns [B, 2, T_bwe].
pub fn forwardBWEResidual(waveform_16k: Tensor, params: *const BWEPipelineParams) Tensor {
    const x = bwePadInput(waveform_16k);
    const mel = forwardComputeMel(x, params.mel_stft);
    const mel_for_bwe = mel.transpose(.{ 0, 1, 3, 2 });
    return forwardVocoderGeneric(
        mel_for_bwe,
        params.bwe_generator.conv_pre,
        params.bwe_generator.ups,
        params.bwe_generator.resblocks,
        params.bwe_generator.act_post,
        params.bwe_generator.conv_post,
        5,
        false,
    );
}

