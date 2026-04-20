const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
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

    /// x: [.b, .t, .d], embedded_timestep: [.b, .t, .d_emb] (broadcasts across T)
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

/// Like zml.nn.sdpa but computes softmax in native dtype (no f32 upcast).
///
/// Hierarchical concatenation that works around Tensor.concatenate's 32-tensor limit.
/// Groups chunks into batches of up to 32, concatenates each group, then concatenates
/// the groups. Supports up to 32×32 = 1024 chunks.
fn hierarchicalConcatenate(chunks: []const Tensor, comptime axis: anytype) Tensor {
    const MAX_CONCAT = 32;
    if (chunks.len <= MAX_CONCAT) {
        return Tensor.concatenate(chunks, axis);
    }
    const n_groups = (chunks.len + MAX_CONCAT - 1) / MAX_CONCAT;
    stdx.debug.assert(n_groups <= MAX_CONCAT, "hierarchicalConcatenate: n_groups={} exceeds {}", .{ n_groups, MAX_CONCAT });
    var groups: [MAX_CONCAT]Tensor = undefined;
    for (0..n_groups) |g| {
        const start = g * MAX_CONCAT;
        const end = @min((g + 1) * MAX_CONCAT, chunks.len);
        groups[g] = Tensor.concatenate(chunks[start..end], axis);
    }
    return Tensor.concatenate(groups[0..n_groups], axis);
}

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
    // Tensor.concatenate supports up to 32 tensors, so we use hierarchical
    // concatenation for large sequence lengths (e.g. Stage 2 at 1024×1536 → 93696 tokens → 92 chunks).
    const MAX_CHUNKS = 1024;
    var chunk_results: [MAX_CHUNKS]Tensor = undefined;
    stdx.debug.assert(n_chunks <= MAX_CHUNKS, "sdpaNoF32Upcast: n_chunks={} exceeds MAX_CHUNKS={}", .{ n_chunks, MAX_CHUNKS });

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

    // Hierarchical concatenation: Tensor.concatenate supports max 32 tensors,
    // so group into batches of 32, concatenate each group, then concatenate groups.
    const attn = hierarchicalConcatenate(chunk_results[0..n_chunks], .q);
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

/// Per-token masked variant of adaValueAt.
///
/// Blends two broadcast timestep sources [B, 1, d_ada] using a per-token
/// mask [B, T, 1] to produce per-token modulation [B, T, D]:
///   result = mask * (sst[idx] + ts_sigma[idx]) + (1-mask) * (sst[idx] + ts_zero[idx])
///
/// This avoids materializing the full [B, T, d_ada] tensor (~905MB for T=6144)
/// by blending AFTER slicing to D (~96MB per value). XLA can reuse these
/// small temporaries sequentially across the 9 ada values per modality.
fn adaValueAtMasked(sst: Tensor, ts_sigma: Tensor, ts_zero: Tensor, mask: Tensor, idx: i64) Tensor {
    const d = sst.dim(.d);
    const slice_ts = struct {
        fn f(ts: Tensor, i: i64, dim: i64) Tensor {
            return ts
                .withPartialTags(.{ .b, .t, .tsflat })
                .slice1d(.tsflat, .{ .start = i * dim, .end = (i + 1) * dim })
                .rename(.{ .tsflat = .d });
        }
    }.f;
    const ts_s = slice_ts(ts_sigma, idx, d); // [B, 1, D]
    const ts_z = slice_ts(ts_zero, idx, d); // [B, 1, D]
    const sst_row = sst
        .slice1d(.n_ada, .{ .start = idx, .end = idx + 1 })
        .squeeze(.n_ada)
        .convert(ts_s.dtype()); // [D]
    const val_s = sst_row.broad(ts_s.shape()).add(ts_s); // [B, 1, D]
    const val_z = sst_row.broad(ts_z.shape()).add(ts_z); // [B, 1, D]
    // mask: [B, T, mask=1] → rename .mask → .d for broadcast with [B, 1, D]
    const m = mask.withPartialTags(.{ .b, .t, .mask }).rename(.{ .mask = .d }).convert(.f32); // [B, T, d=1]
    const one_minus = Tensor.scalar(1.0, .f32).sub(m); // [B, T, d=1]
    // Broadcast both sides to [B, T, D] — XLA fuses this into the elementwise op
    const target = zml.Shape.init(.{ .b = m.dim(.b), .t = m.dim(.t), .d = val_s.dim(.d) }, .f32);
    return m.broad(target).mul(val_s.convert(.f32).broad(target))
        .add(one_minus.broad(target).mul(val_z.convert(.f32).broad(target)));
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
        scale_shift_table: Tensor, // [.n_ada=6, .d=D_video]  video self-attn + FF
        audio_scale_shift_table: Tensor, // [.n_ada=6, .d=D_audio]  audio self-attn + FF
        scale_shift_table_a2v_ca_video: Tensor, // [.n_ada=5, .d=D_video]  AV CA video side
        scale_shift_table_a2v_ca_audio: Tensor, // [.n_ada=5, .d=D_audio]  AV CA audio side
        prompt_scale_shift_table: Tensor, // [.n_prompt=2, .d=D_video] text CA prompt modulation
        audio_prompt_scale_shift_table: Tensor, // [.n_prompt=2, .d=D_audio] text CA prompt modulation

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
        /// AdaLN timestep embeddings at sigma; broadcast [B, 1, N_ada * D].
        video_timesteps: Tensor, // [B, 1, 6 * D_video]
        audio_timesteps: Tensor, // [B, 1, 6 * D_audio]
        /// AdaLN timestep embeddings at zero; broadcast [B, 1, N_ada * D].
        /// Used with denoise_mask to produce per-token modulation inside blocks.
        video_timesteps_zero: Tensor, // [B, 1, 6 * D_video]
        audio_timesteps_zero: Tensor, // [B, 1, 6 * D_audio]
        /// Denoise mask for per-token timestep blending inside blocks.
        v_denoise_mask: Tensor, // [B, T_v, 1]
        a_denoise_mask: Tensor, // [B, T_a, 1]
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
        v_cross_ss_ts: Tensor, // video.cross_scale_shift_timestep  [B, 1, 4 * D_video]
        v_cross_gate_ts: Tensor, // video.cross_gate_timestep          [B, 1, D_video]
        a_cross_ss_ts: Tensor, // audio.cross_scale_shift_timestep  [B, 1, 4 * D_audio]
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
        const vshift_msa = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 0).convert(vd);
        const vscale_msa = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 1).convert(vd);
        const vgate_msa = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 2).convert(vd);
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
        const v_shift_q = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 6).convert(vd);
        const v_scale_q = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 7).convert(vd);
        const v_gate_q = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 8).convert(vd);
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
        const ashift_msa = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 0).convert(ad);
        const ascale_msa = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 1).convert(ad);
        const agate_msa = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 2).convert(ad);
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
        const a_shift_q = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 6).convert(ad);
        const a_scale_q = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 7).convert(ad);
        const a_gate_q = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 8).convert(ad);
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
        const sst_v_ss = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 0, .end = 4 });
        const sst_v_gate = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 4, .end = 5 });
        const scale_v_a2v = adaValueAt(sst_v_ss, inputs.v_cross_ss_ts, 0).convert(vd);
        const shift_v_a2v = adaValueAt(sst_v_ss, inputs.v_cross_ss_ts, 1).convert(vd);
        const gate_a2v = adaValueAt(sst_v_gate, inputs.v_cross_gate_ts, 0).convert(vd);
        const vx_scaled_a2v = vx_norm3
            .mul(scale_v_a2v.addConstant(1.0).broad(vx_norm3.shape()))
            .add(shift_v_a2v.broad(vx_norm3.shape()));

        // Audio context: sst_a2v_ca_audio rows 0,1 (scale/shift).
        const sst_a_ss = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 0, .end = 4 });
        const sst_a_gate = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 4, .end = 5 });
        const scale_a_a2v = adaValueAt(sst_a_ss, inputs.a_cross_ss_ts, 0).convert(ad);
        const shift_a_a2v = adaValueAt(sst_a_ss, inputs.a_cross_ss_ts, 1).convert(ad);
        const ax_scaled_a2v = ax_norm3
            .mul(scale_a_a2v.addConstant(1.0).broad(ax_norm3.shape()))
            .add(shift_a_a2v.broad(ax_norm3.shape()));

        const a2v_out = if (bf16_attn) self.audio_to_video_attn.forwardBf16(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
            .context = ax_scaled_a2v,
            .pe_cos = inputs.a2v_pe_cos,
            .pe_sin = inputs.a2v_pe_sin,
            .k_pe_cos = inputs.a2v_k_pe_cos,
            .k_pe_sin = inputs.a2v_k_pe_sin,
        }) else self.audio_to_video_attn.forward(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
            .context = ax_scaled_a2v,
            .pe_cos = inputs.a2v_pe_cos,
            .pe_sin = inputs.a2v_pe_sin,
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
        const scale_a_v2a = adaValueAt(sst_a_ss, inputs.a_cross_ss_ts, 2).convert(ad);
        const shift_a_v2a = adaValueAt(sst_a_ss, inputs.a_cross_ss_ts, 3).convert(ad);
        const gate_v2a = adaValueAt(sst_a_gate, inputs.a_cross_gate_ts, 0).convert(ad);
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
            .context = vx_scaled_v2a,
            .pe_cos = inputs.v2a_pe_cos,
            .pe_sin = inputs.v2a_pe_sin,
            .k_pe_cos = inputs.v2a_k_pe_cos,
            .k_pe_sin = inputs.v2a_k_pe_sin,
        }) else self.video_to_audio_attn.forward(ax_scaled_v2a, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
            .context = vx_scaled_v2a,
            .pe_cos = inputs.v2a_pe_cos,
            .pe_sin = inputs.v2a_pe_sin,
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
        const vshift_mlp = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 3).convert(vd);
        const vscale_mlp = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 4).convert(vd);
        const vgate_mlp = adaValueAtMasked(params.scale_shift_table, inputs.video_timesteps, inputs.video_timesteps_zero, inputs.v_denoise_mask, 5).convert(vd);
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
        const ashift_mlp = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 3).convert(ad);
        const ascale_mlp = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 4).convert(ad);
        const agate_mlp = adaValueAtMasked(params.audio_scale_shift_table, inputs.audio_timesteps, inputs.audio_timesteps_zero, inputs.a_denoise_mask, 5).convert(ad);
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

    /// bf16-native attention path: attention runs in bf16 matching Python's dtype chain.
    /// Activated via --bf16-attn-stage1 / --bf16-attn-stage2 CLI flags.
    pub fn forwardNativeBf16Attn(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, false, false, true, false, false, vx_in, ax_in, inputs, params);
    }

    /// STG block variant: both video and audio self-attention bypass Q·K·V and compute to_out(to_v(x)) instead.
    /// Used for block 28 (0-indexed) during the STG perturbation pass (Pass 3) in Stage 1 denoising.
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

/// Output projection after all transformer blocks: hidden → velocity.
/// x: [.b, .t, .d], embedded_timestep: [.b, .t, .d_emb]
/// Returns: [.b, .t, .d_out=128]
pub fn forwardOutputProjection(x: Tensor, embedded_timestep: Tensor, params: OutputProjection.Params) Tensor {
    return OutputProjection.forward(x, embedded_timestep, params);
}

// =====================================================================
// Denoising-loop arithmetic (Step 3)
// =====================================================================

/// Generate Gaussian noise via Box-Muller transform.
///
/// Takes an Rng state and produces N(0,1) noise of the given shape.
/// Uses two uniform samples: Z = sqrt(-2*ln(U1)) * cos(2π*U2).
/// Returns the updated Rng state and noise tensor in bf16.
pub fn forwardGenerateNoise(rng: Tensor.Rng, target_shape: Tensor) struct { Tensor.Rng, Tensor } {
    const shape = target_shape._shape;
    const eps = std.math.floatEps(f32);

    const rng1, const uni1 = rng.uniform(shape.withDtype(.f32), .{ .min = eps, .max = 1.0 });
    const rng2, const uni2 = rng1.uniform(shape.withDtype(.f32), .{ .min = 0.0, .max = 1.0 });

    // Box-Muller: Z = sqrt(-2 * ln(U1)) * cos(2π * U2)
    const r = uni1.log().scale(-2.0).sqrt();
    const theta = uni2.scale(2.0 * std.math.pi);
    const noise = r.mul(theta.cos());

    return .{ rng2, noise.convert(shape.dtype()) };
}

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

/// Default token count for sigma-shift when no latent tensor is available.
/// Matches Python's `default_number_of_tokens = MAX_SHIFT_ANCHOR` in schedulers.py.
pub const MAX_SHIFT_ANCHOR: usize = 4096;

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

    // 1. linspace(1.0, 0.0, num_steps + 1)
    var sigmas: [max_steps + 1]f32 = undefined;
    const n: f32 = @floatFromInt(num_steps);
    for (0..num_steps + 1) |i| {
        const t: f32 = @floatFromInt(i);
        sigmas[i] = 1.0 - t / n;
    }

    // 2. Compute sigma_shift from token count
    const x1: f32 = 1024.0; // BASE_SHIFT_ANCHOR
    const x2: f32 = 4096.0; // MAX_SHIFT_ANCHOR
    const mm = (max_shift - base_shift) / (x2 - x1);
    const b = base_shift - mm * x1;
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

/// Stage 2 distilled sigma schedule (3 denoising steps).
/// From ltx_pipelines.utils.constants.STAGE_2_DISTILLED_SIGMA_VALUES.
pub const stage2_distilled_sigmas: [4]f32 = .{ 0.909375, 0.725, 0.421875, 0.0 };

// Values taken from regular 30-step schedule
// https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/components/schedulers.py
pub const stage1_default_schedule = struct {
    pub const num_steps: usize = 30;
    pub const max_shift: f32 = 2.05;
    pub const base_shift: f32 = 0.95;
    pub const terminal: f32 = 0.1;
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
/// Velocity → x0 conversion matching Python's X0Model + to_denoised() dtype chain.
/// The guider combine operates on x0 predictions (not velocities) so that the
/// rescale factor is computed in the same space as Python.
///
/// Python dtype chain (traced empirically with f32 denoise_mask):
///   timesteps = denoise_mask(f32) * sigma(f32) → f32
///   x0 = (sample.f32 - velocity.f32 * timesteps_f32).to(bf16)
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
    video_timesteps_zero: Tensor,
    audio_timesteps_zero: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
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
        .video_timesteps_zero = video_timesteps_zero,
        .audio_timesteps_zero = audio_timesteps_zero,
        .v_denoise_mask = v_denoise_mask,
        .a_denoise_mask = a_denoise_mask,
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
    video_timesteps_zero: Tensor,
    audio_timesteps_zero: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
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
        .video_timesteps_zero = video_timesteps_zero,
        .audio_timesteps_zero = audio_timesteps_zero,
        .v_denoise_mask = v_denoise_mask,
        .a_denoise_mask = a_denoise_mask,
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
/// self-attention use V-passthrough (to_out(to_v(x))). Used for block 28 (0-indexed) during
/// Pass 3 (STG perturbation) in Stage 1 denoising.
pub fn forwardBlock0NativeSTG(
    vx_in: Tensor,
    ax_in: Tensor,
    video_timesteps: Tensor,
    audio_timesteps: Tensor,
    video_timesteps_zero: Tensor,
    audio_timesteps_zero: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
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
        .video_timesteps_zero = video_timesteps_zero,
        .audio_timesteps_zero = audio_timesteps_zero,
        .v_denoise_mask = v_denoise_mask,
        .a_denoise_mask = a_denoise_mask,
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
    video_timesteps_zero: Tensor,
    audio_timesteps_zero: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
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
        .video_timesteps_zero = video_timesteps_zero,
        .audio_timesteps_zero = audio_timesteps_zero,
        .v_denoise_mask = v_denoise_mask,
        .a_denoise_mask = a_denoise_mask,
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
    video_timesteps_zero: Tensor,
    audio_timesteps_zero: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
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
        .video_timesteps_zero = video_timesteps_zero,
        .audio_timesteps_zero = audio_timesteps_zero,
        .v_denoise_mask = v_denoise_mask,
        .a_denoise_mask = a_denoise_mask,
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
    video_timesteps_zero: Tensor,
    audio_timesteps_zero: Tensor,
    v_denoise_mask: Tensor,
    a_denoise_mask: Tensor,
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
        .video_timesteps_zero = video_timesteps_zero,
        .audio_timesteps_zero = audio_timesteps_zero,
        .v_denoise_mask = v_denoise_mask,
        .a_denoise_mask = a_denoise_mask,
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
    video_timesteps: Tensor, // [B, 1, 9 * D_video] broadcast (sigma-based)
    audio_timesteps: Tensor, // [B, 1, 9 * D_audio] broadcast (sigma-based)
    video_timesteps_zero: Tensor, // [B, 1, 9 * D_video] broadcast (zero-based)
    audio_timesteps_zero: Tensor, // [B, 1, 9 * D_audio] broadcast (zero-based)
    v_denoise_mask: Tensor, // [B, T_v, 1]
    a_denoise_mask: Tensor, // [B, T_a, 1]
    v_embedded_timestep: Tensor, // [B, T_v, D_video] per-token (blended)
    a_embedded_timestep: Tensor, // [B, T_a, D_audio] per-token (blended)
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

    // Tag all inputs — tensors arrive untagged from safetensors.
    const v_lat = v_latent.withPartialTags(.{ .b, .t, .patch });
    const a_lat = a_latent.withPartialTags(.{ .b, .t, .patch });
    const v_ctx = v_context.withPartialTags(.{ .b, .t, .d });
    const a_ctx = a_context.withPartialTags(.{ .b, .t, .d });

    const v_mask = v_denoise_mask.withPartialTags(.{ .b, .t, .mask });
    const a_mask = a_denoise_mask.withPartialTags(.{ .b, .t, .mask });

    // --- 1. Patchify ---
    const vx = patchify.forward(v_lat, params.video_patchify);
    const ax = patchify.forward(a_lat, params.audio_patchify);

    // --- 2. Timestep embeddings ---
    //
    // Compute AdaLN modulation at BOTH sigma and zero. Pass both broadcast
    // [B, 1, d_ada] sets plus the mask to blocks, which blend per-token inside
    // adaValueAtMasked at the [B, T, D] level (~96MB) instead of materializing
    // the full [B, T, d_ada] (~905MB) here.
    //
    // For embedded_timestep (used by output projection, only [B, T, D_emb] = 48MB),
    // we CAN afford to blend here since it's much smaller.
    const v_sigma_scaled = v_sigma.convert(.f32).scale(1000.0); // [B]
    const a_sigma_scaled = a_sigma.convert(.f32).scale(1000.0); // [B]
    const v_adaln = AdaLayerNormSingle.forward(v_sigma_scaled, params.adaln_single);
    const a_adaln = AdaLayerNormSingle.forward(a_sigma_scaled, params.audio_adaln_single);

    // Zero timestep modulation (for conditioned tokens)
    const v_zero = Tensor.zeroes(v_sigma_scaled.shape().withDtype(.f32));
    const a_zero = Tensor.zeroes(a_sigma_scaled.shape().withDtype(.f32));
    const v_adaln_zero = AdaLayerNormSingle.forward(v_zero, params.adaln_single);
    const a_adaln_zero = AdaLayerNormSingle.forward(a_zero, params.audio_adaln_single);

    // Reshape [B, d_ada] → [B, 1, d_ada] for broadcast across T in adaValueAtMasked
    const v_timesteps = v_adaln.modulation.reshape(
        v_adaln.modulation.shape().splitAxis(.b, .{ .b = v_adaln.modulation.dim(.b), .t = 1 }),
    );
    const a_timesteps = a_adaln.modulation.reshape(
        a_adaln.modulation.shape().splitAxis(.b, .{ .b = a_adaln.modulation.dim(.b), .t = 1 }),
    );
    const v_timesteps_zero = v_adaln_zero.modulation.reshape(
        v_adaln_zero.modulation.shape().splitAxis(.b, .{ .b = v_adaln_zero.modulation.dim(.b), .t = 1 }),
    );
    const a_timesteps_zero = a_adaln_zero.modulation.reshape(
        a_adaln_zero.modulation.shape().splitAxis(.b, .{ .b = a_adaln_zero.modulation.dim(.b), .t = 1 }),
    );

    // Blend embedded_timestep per-token: [B, T, D_emb] — small enough to materialize
    // sigma: [B, 1, D_emb], zero: [B, 1, D_emb], mask: [B, T, 1]
    const v_emb_s = v_adaln.embedded_timestep.reshape(
        v_adaln.embedded_timestep.shape().splitAxis(.b, .{ .b = v_adaln.embedded_timestep.dim(.b), .t = 1 }),
    );
    const v_emb_z = v_adaln_zero.embedded_timestep.reshape(
        v_adaln_zero.embedded_timestep.shape().splitAxis(.b, .{ .b = v_adaln_zero.embedded_timestep.dim(.b), .t = 1 }),
    );
    const a_emb_s = a_adaln.embedded_timestep.reshape(
        a_adaln.embedded_timestep.shape().splitAxis(.b, .{ .b = a_adaln.embedded_timestep.dim(.b), .t = 1 }),
    );
    const a_emb_z = a_adaln_zero.embedded_timestep.reshape(
        a_adaln_zero.embedded_timestep.shape().splitAxis(.b, .{ .b = a_adaln_zero.embedded_timestep.dim(.b), .t = 1 }),
    );
    // mask [B, T, mask=1] → rename → [B, T, d_emb=1] for broadcast
    const v_mask_emb = v_mask.convert(.f32).rename(.{ .mask = .d_emb });
    const v_one_minus_emb = Tensor.scalar(1.0, .f32).sub(v_mask_emb);
    const v_emb_target = zml.Shape.init(.{ .b = v_mask_emb.dim(.b), .t = v_mask_emb.dim(.t), .d_emb = v_emb_s.dim(.d_emb) }, .f32);
    const v_emb_ts = v_mask_emb.broad(v_emb_target).mul(v_emb_s.convert(.f32).broad(v_emb_target))
        .add(v_one_minus_emb.broad(v_emb_target).mul(v_emb_z.convert(.f32).broad(v_emb_target)));
    const a_mask_emb = a_mask.convert(.f32).rename(.{ .mask = .d_emb });
    const a_one_minus_emb = Tensor.scalar(1.0, .f32).sub(a_mask_emb);
    const a_emb_target = zml.Shape.init(.{ .b = a_mask_emb.dim(.b), .t = a_mask_emb.dim(.t), .d_emb = a_emb_s.dim(.d_emb) }, .f32);
    const a_emb_ts = a_mask_emb.broad(a_emb_target).mul(a_emb_s.convert(.f32).broad(a_emb_target))
        .add(a_one_minus_emb.broad(a_emb_target).mul(a_emb_z.convert(.f32).broad(a_emb_target)));

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
    // The f32 adaln computes precisely, then bf16 rounding matches
    // Python's dtype chain.
    return .{
        .vx = vx,
        .ax = ax,
        .video_timesteps = v_timesteps,
        .audio_timesteps = a_timesteps,
        .video_timesteps_zero = v_timesteps_zero,
        .audio_timesteps_zero = a_timesteps_zero,
        .v_denoise_mask = v_mask,
        .a_denoise_mask = a_mask,
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
