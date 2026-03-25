const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

/// Which parameter set to use at runtime.
pub const Stage = enum {
    stage1,
    stage2,
};

pub const Config = struct {
    num_transformer_blocks: usize,
};

/// FeedForward module shared by stage 1 and stage 2.
///
/// Forward contract for LTX 2.3 video FF:
/// [B, T, 4096] -> Linear(4096->16384) -> GELU(tanh approx) -> Linear(16384->4096)
pub const FeedForward = struct {
    pub const Params = struct {
        proj: zml.nn.Linear,
        out: zml.nn.Linear,
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
        const h1 = params.proj.forward(x_); // In LTX pipeline, this projection is fused with the GELU approximation, but we separate it here for parity bring-up and diagnostics.
        const h2 = forwardFFGeluF32(h1); // Keep GELU math in f32 for parity with reference behavior before casting back to the input dtype.
        return params.out.forward(h2);
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        params.proj.weight.deinit();
        if (params.proj.bias) |*b| b.deinit();
        params.out.weight.deinit();
        if (params.out.bias) |*b| b.deinit();
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
        const wdtype = params.linear_1.weight.dtype();

        // 1. Sinusoidal embedding: [.b] → [.b, .d_sin=256] (f32) → cast to weight dtype
        const t_proj = sinusoidalTimestepEmbedding(sigma).convert(wdtype);

        // 2. TimestepEmbedding: linear_1 → silu → linear_2
        const h1 = params.linear_1.forward(t_proj); // [.b, .d]
        const h2 = h1.silu(); // [.b, .d]
        const h3 = params.linear_2.forward(h2); // [.b, .d_emb]
        // h3 is the embedded_timestep returned to callers

        // 3. adaLN-single part: silu → linear_out
        const h4 = h3.silu(); // [.b, .d_emb]
        const modulation = params.linear_out.forward(h4); // [.b, .d_ada]

        return .{ .modulation = modulation, .embedded_timestep = h3 };
    }
};

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
        _ = self;

        const x_ = x.withPartialTags(.{ .b, .t, .d });
        const out_dtype = x_.dtype();
        const x_f32 = x_.convert(.f32);
        const context = if (opts.context) |ctx| ctx.withPartialTags(.{ .b, .t, .d }) else x_;
        const context_f32 = context.convert(.f32);

        // Keep attention math in f32 to match reference kernel accumulation behavior,
        // then cast back at the output dtype for downstream parity/compatibility.
        var q = x_f32.dot(params.to_q.weight.convert(.f32), .d);
        if (params.to_q.bias) |bias_orig| {
            q = q.add(bias_orig.convert(.f32).broad(q.shape()));
        }

        var k = context_f32.dot(params.to_k.weight.convert(.f32), .d);
        if (params.to_k.bias) |bias_orig| {
            k = k.add(bias_orig.convert(.f32).broad(k.shape()));
        }

        var v = context_f32.dot(params.to_v.weight.convert(.f32), .d);
        if (params.to_v.bias) |bias_orig| {
            v = v.add(bias_orig.convert(.f32).broad(v.shape()));
        }

        // LTX 2.3 attention includes RMSNorms on q and k with learned weights, applied before splitting into heads.
        // In PyTorch, multiplying by the learned weight is done within the `RMSNorm` module after the normalization step. 
        // On the other hand, zml.nn.rmsNorm() returns a normalized tensor without applying the learned weight, so we multiply manually.
        q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.convert(.f32).broad(q.shape()));
        k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.convert(.f32).broad(k.shape()));

        // Split embedding dimension into heads
        // `q`: `[B, Tq, heads * head_dim]` --> `qh`: `[B, Q, H, HD]`
        // `k`: `[B, Tk, heads * head_dim]` --> `kh`: `[B, K, H, HD]`
        // `v`: `[B, Tk, heads * head_dim]` --> `vh`: `[B, K, H, HD]`
        var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
        var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

        // Apply RoPE if present in checkpoint.
        // Checkpoint tensors come in two layouts: rank-4 [B, H, T, HD] (tagged head-first as [B, H, Q/K, HD])
        // or rank-2 [T, H*HD] (split and tagged token-first as [B, Q/K, H, HD]). Both are equivalent under
        // ZML's named-axis system; applyLtxRotaryEmb handles alignment via transpose().
        // When no k-specific RoPE tensors are present, the q tensors are reused for k.
        if (opts.pe_cos) |pe_cos| {
            if (opts.pe_sin) |pe_sin| {
                const q_cos = if (pe_cos.rank() == 4)
                    pe_cos.withPartialTags(.{ .b, .h, .q, .hd }).convert(.f32)
                else
                    pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);

                const q_sin = if (pe_sin.rank() == 4)
                    pe_sin.withPartialTags(.{ .b, .h, .q, .hd }).convert(.f32)
                else
                    pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);

                const k_cos = if (opts.k_pe_cos) |k_pe_cos|
                    if (k_pe_cos.rank() == 4)
                        k_pe_cos.withPartialTags(.{ .b, .h, .k, .hd }).convert(.f32)
                    else
                        k_pe_cos.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32)
                else
                    q_cos.rename(.{ .q = .k });

                const k_sin = if (opts.k_pe_sin) |k_pe_sin|
                    if (k_pe_sin.rank() == 4)
                        k_pe_sin.withPartialTags(.{ .b, .h, .k, .hd }).convert(.f32)
                    else
                        k_pe_sin.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32)
                else
                    q_sin.rename(.{ .q = .k });

                // LTX is a video generation model needing coordinates for temporal + spatial (height/width) dimensions, not flat sequential 
                // token positions, so the RoPE implementation is customized but mathematically equivalent to standard RoPE.
                qh = applyLtxRotaryEmb(qh, q_cos, q_sin);
                kh = applyLtxRotaryEmb(kh, k_cos, k_sin);
            }
        }

        qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
        kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
        vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

        // ZML uses named axes. 
        // The sdpa primitive expects a specific batch axis name, `.batch`, so it is temporarily renamed here. 
        var attn = zml.nn.sdpa(
            qh.rename(.{ .b = .batch }),
            kh.rename(.{ .b = .batch }),
            vh.rename(.{ .b = .batch }),
            .{ .attn_mask = if (opts.mask) |m| m.rename(.{ .b = .batch }) else null },
        ).rename(.{ .batch = .b }); // [B, Q, H, HD]

        // Compute per-head gates as 2 * sigmoid(logits) so zero-initialized logits preserve identity.
        var gate_logits = x_f32.dot(params.to_gate_logits.weight.convert(.f32), .d); // [B, T, H]
        if (params.to_gate_logits.bias) |bias_orig| {
            gate_logits = gate_logits.add(bias_orig.convert(.f32).broad(gate_logits.shape()));
        }
        const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate); // [B, Q, H]
        attn = attn.mul(gate.broad(attn.shape())); // [B, Q, H, HD]

        const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }); // [B, Q, D_V] with D_V = H * HD
        var out = merged.dot(params.to_out.weight.convert(.f32), .d_v);
        if (params.to_out.bias) |bias_orig| {
            out = out.add(bias_orig.convert(.f32).broad(out.shape()));
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

/// One AV transformer block. Shared implementation across stages.
///
/// `forward` implements full Python BasicAVTransformerBlock semantics (M6 parity):
///   video stream: self-attn → text cross-attn → A->V cross-attn → FF (all with AdaLN gates)
///   audio stream: self-attn → text cross-attn → V->A cross-attn → FF (all with AdaLN gates)
///
/// Inputs are pre-computed per-module query tensors and AdaLN values captured from Python,
/// which are required to preserve hidden preprocessing in text/AV cross-attn paths.
///
/// `forwardFFBoundary` is the legacy simplified single-stream path (FF only, no residuals),
/// kept for backward compatibility with older checker targets and `LTXModel.forward`.
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
    };

    pub const VideoStreamInputs = struct {
        vx_in: Tensor,
        norm_vx: Tensor,
        v_text_x: Tensor,
        v_pe_cos: Tensor,
        v_pe_sin: Tensor,
        vgate_msa: Tensor,
        vgate_text_ca: Tensor, // M2 output gate (row 8 of scale_shift_table)
        v_text_ctx: Tensor,
        a2v_x: Tensor,
        a2v_ctx: Tensor,
        a2v_pe_cos: Tensor,
        a2v_pe_sin: Tensor,
        a2v_k_pe_cos: Tensor,
        a2v_k_pe_sin: Tensor,
        a2v_gate: Tensor,
        a2v_mask: Tensor,
        vx_scaled: Tensor,
        vgate_mlp: Tensor,
    };

    pub const AudioStreamInputs = struct {
        ax_in: Tensor,
        norm_ax: Tensor,
        a_text_x: Tensor,
        a_pe_cos: Tensor,
        a_pe_sin: Tensor,
        agate_msa: Tensor,
        agate_text_ca: Tensor, // M4-B output gate (row 8 of audio_scale_shift_table)
        a_text_ctx: Tensor,
        v2a_x: Tensor,
        v2a_ctx: Tensor,
        v2a_pe_cos: Tensor,
        v2a_pe_sin: Tensor,
        v2a_k_pe_cos: Tensor,
        v2a_k_pe_sin: Tensor,
        v2a_gate: Tensor,
        v2a_mask: Tensor,
        ax_scaled: Tensor,
        agate_mlp: Tensor,
    };

    pub const FullInputs = struct {
        video: VideoStreamInputs,
        audio: AudioStreamInputs,
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

    pub fn forward(self: BasicAVTransformerBlock, inputs: FullInputs, params: Params) FullOutputs {
        return .{
            .vx_out = self.forwardVideoStream(inputs.video, params),
            .ax_out = self.forwardAudioStream(inputs.audio, params),
        };
    }

    pub fn forwardFFBoundary(self: BasicAVTransformerBlock, x: Tensor, params: Params) Tensor {
        _ = self.attn1.forward(x, params.attn1, kindNumHeads(.attn1), .{});
        // NOTE: residual/gating/adaln wiring belongs at block level and will be added here.
        // audio_ff is defined on the audio stream (2048-dim) and cannot consume video x (4096-dim).
        // Keep this simplified single-stream path focused on the video ff boundary.
        return self.ff.forward(x, params.ff);
    }

    pub fn forwardVideoStream(self: BasicAVTransformerBlock, inputs: VideoStreamInputs, params: Params) Tensor {
        // M1: video self-attn residual
        const attn1_out = self.attn1.forward(inputs.norm_vx, params.attn1, kindNumHeads(.attn1), .{
            .pe_cos = inputs.v_pe_cos,
            .pe_sin = inputs.v_pe_sin,
        });
        var h = inputs.vx_in.withPartialTags(.{ .b, .t, .d })
            .add(attn1_out.mul(inputs.vgate_msa.broad(attn1_out.shape())));

        // M2: video text cross-attn residual (query from captured module input)
        const text_ca_out = self.attn2.forward(inputs.v_text_x, params.attn2, kindNumHeads(.attn2), .{
            .context = inputs.v_text_ctx,
        });
        h = h.add(text_ca_out.mul(inputs.vgate_text_ca.broad(text_ca_out.shape())));

        // M5-A: A->V cross-attn residual (query from captured module input)
        const a2v_out = self.audio_to_video_attn.forward(inputs.a2v_x, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
            .context = inputs.a2v_ctx,
            .pe_cos = inputs.a2v_pe_cos,
            .pe_sin = inputs.a2v_pe_sin,
            .k_pe_cos = inputs.a2v_k_pe_cos,
            .k_pe_sin = inputs.a2v_k_pe_sin,
        });
        h = h.add(a2v_out.mul(inputs.a2v_gate.broad(a2v_out.shape())).mul(inputs.a2v_mask.broad(a2v_out.shape())));

        // M3: video FF residual
        const ff_out = self.ff.forward(inputs.vx_scaled, params.ff);
        h = h.add(ff_out.mul(inputs.vgate_mlp.broad(ff_out.shape())));

        return h;
    }

    pub fn forwardAudioStream(self: BasicAVTransformerBlock, inputs: AudioStreamInputs, params: Params) Tensor {
        // M4-A: audio self-attn residual
        const attn1_out = self.audio_attn1.forward(inputs.norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{
            .pe_cos = inputs.a_pe_cos,
            .pe_sin = inputs.a_pe_sin,
        });
        var h = inputs.ax_in.withPartialTags(.{ .b, .t, .d })
            .add(attn1_out.mul(inputs.agate_msa.broad(attn1_out.shape())));

        // M4-B: audio text cross-attn residual (query from captured module input)
        const text_ca_out = self.audio_attn2.forward(inputs.a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{
            .context = inputs.a_text_ctx,
        });
        h = h.add(text_ca_out.mul(inputs.agate_text_ca.broad(text_ca_out.shape())));

        // M5-B: V->A cross-attn residual (query from captured module input)
        const v2a_out = self.video_to_audio_attn.forward(inputs.v2a_x, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
            .context = inputs.v2a_ctx,
            .pe_cos = inputs.v2a_pe_cos,
            .pe_sin = inputs.v2a_pe_sin,
            .k_pe_cos = inputs.v2a_k_pe_cos,
            .k_pe_sin = inputs.v2a_k_pe_sin,
        });
        h = h.add(v2a_out.mul(inputs.v2a_gate.broad(v2a_out.shape())).mul(inputs.v2a_mask.broad(v2a_out.shape())));

        // M4-C: audio FF residual
        const ff_out = self.audio_ff.forward(inputs.ax_scaled, params.audio_ff);
        h = h.add(ff_out.mul(inputs.agate_mlp.broad(ff_out.shape())));

        return h;
    }

    pub fn unloadBuffers(params: *zml.Bufferized(Params)) void {
        FeedForward.unloadBuffers(&params.ff);
        FeedForward.unloadBuffers(&params.audio_ff);
        Attention.unloadBuffers(&params.attn1);
        Attention.unloadBuffers(&params.attn2);
        Attention.unloadBuffers(&params.audio_attn1);
        Attention.unloadBuffers(&params.audio_attn2);
        Attention.unloadBuffers(&params.audio_to_video_attn);
        Attention.unloadBuffers(&params.video_to_audio_attn);
        params.scale_shift_table.deinit();
        params.audio_scale_shift_table.deinit();
        params.scale_shift_table_a2v_ca_video.deinit();
        params.scale_shift_table_a2v_ca_audio.deinit();
        params.prompt_scale_shift_table.deinit();
        params.audio_prompt_scale_shift_table.deinit();
    }

    /// Per-inference inputs shared across all blocks during a model forward pass.
    ///
    /// Unlike `FullInputs`, which carries pre-computed per-module AdaLN values captured
    /// from Python activation traces, `SharedInputs` contains only the raw conditioning
    /// data (timestep embeddings, text contexts, positional embeddings) that are identical
    /// for every block. Per-block AdaLN modulation values (shift, scale, gate) are computed
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
    ///
    /// For parity checkers that operate on pre-captured per-module inputs, use
    /// `forward(FullInputs, Params)` instead.
    fn forwardNativeImpl(self: BasicAVTransformerBlock, comptime audio_ff_residual_f32: bool, comptime audio_all_residuals_f32: bool, comptime video_all_residuals_f32: bool, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
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
        const attn1_out = self.attn1.forward(norm_vx, params.attn1, kindNumHeads(.attn1), .{
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
        const v_shift_kv = adaValueAt(params.prompt_scale_shift_table, inputs.v_prompt_timestep, 0);
        const v_scale_kv = adaValueAt(params.prompt_scale_shift_table, inputs.v_prompt_timestep, 1);
        const v_text_ctx_mod = inputs.v_text_ctx
            .mul(v_scale_kv.addConstant(1.0).broad(inputs.v_text_ctx.shape()))
            .add(v_shift_kv.broad(inputs.v_text_ctx.shape()));
        const v_text_ca_out = self.attn2.forward(v_text_x, params.attn2, kindNumHeads(.attn2), .{
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
        const audio_attn1_out = self.audio_attn1.forward(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{
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
        const a_shift_kv = adaValueAt(params.audio_prompt_scale_shift_table, inputs.a_prompt_timestep, 0);
        const a_scale_kv = adaValueAt(params.audio_prompt_scale_shift_table, inputs.a_prompt_timestep, 1);
        const a_text_ctx_mod = inputs.a_text_ctx
            .mul(a_scale_kv.addConstant(1.0).broad(inputs.a_text_ctx.shape()))
            .add(a_shift_kv.broad(inputs.a_text_ctx.shape()));
        const a_text_ca_out = self.audio_attn2.forward(a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{
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

        const a2v_out = self.audio_to_video_attn.forward(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
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

        const v2a_out = self.video_to_audio_attn.forward(ax_scaled_v2a, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
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
        return self.forwardNativeImpl(false, false, false, vx_in, ax_in, inputs, params);
    }

    /// Checker-only experimental path: only the final audio FF residual add is computed in f32.
    /// Outputs are converted back to the original stream dtype.
    pub fn forwardNativeAudioFFResidualF32(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(true, false, false, vx_in, ax_in, inputs, params);
    }

    /// Checker-only experimental path: all audio residual adds are computed in f32.
    /// Outputs are converted back to the original stream dtype.
    pub fn forwardNativeAudioAllResidualsF32(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, true, false, vx_in, ax_in, inputs, params);
    }

    /// Checker-only experimental path: all video residual adds are computed in f32, and video FF uses f32 matmul.
    /// Outputs are converted back to the original stream dtype.
    pub fn forwardNativeVideoAllResidualsF32(self: BasicAVTransformerBlock, vx_in: Tensor, ax_in: Tensor, inputs: SharedInputs, params: Params) FullOutputs {
        return self.forwardNativeImpl(false, false, true, vx_in, ax_in, inputs, params);
    }
};

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
                BasicAVTransformerBlock.unloadBuffers(bp);
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

    pub fn forward(self: LTXModel, x: Tensor, params: Params) Tensor {
        std.debug.assert(self.blocks.len == params.blocks.len);

        var h = x;
        for (self.blocks, params.blocks) |block, block_params| {
            // LTXModel.forward still exposes the legacy single-stream interface.
            // Full M6 parity path is available via BasicAVTransformerBlock.forward.
            h = block.forwardFFBoundary(h, block_params);
        }
        return h;
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

    pub fn forward(self: X0Model, x: Tensor, params: Params) Tensor {
        return self.velocity_model.forward(x, params.velocity_model);
    }

    /// Focused entrypoint used for FF parity bring-up.
    pub fn forwardBlock0FF(_: X0Model, x: Tensor, params: Params) Tensor {
        const ff = FeedForward{};
        return ff.forward(x, params.velocity_model.blocks[0].ff);
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

/// Free-function entrypoint for parity tooling: block0 FF only.
pub fn forwardBlock0FF(x: Tensor, params: X0Model.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(x, params.velocity_model.blocks[0].ff);
}

/// Free-function entrypoint for simplified FF-boundary parity bring-up.
///
/// This does not represent full Python block semantics yet; it is used by
/// examples/ltx:block0_ff_boundary_check as a temporary surrogate during bring-up.
pub fn forwardBlock0FFBoundary(x: Tensor, params: BasicAVTransformerBlock.Params) Tensor {
    const block = BasicAVTransformerBlock.init();
    return block.forwardFFBoundary(x, params);
}

/// Backward-compatible alias for older scripts still referring to `forwardBlock0`.
pub fn forwardBlock0(x: Tensor, params: BasicAVTransformerBlock.Params) Tensor {
    return forwardBlock0FFBoundary(x, params);
}

/// Focused FF-only entrypoint for parity bring-up.
pub fn forwardFF(x: Tensor, params: FeedForward.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(x, params);
}

/// Cast a single tensor to f32. Used by block_slice_48_check for f32-carry chain init.
pub fn castToF32(x: Tensor) Tensor {
    return x.convert(.f32);
}

/// Focused FF net.0 (Linear 4096->16384) entrypoint.
pub fn forwardFFLinear1(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    return params.proj.forward(x_);
}

/// Focused FF net.2 (Linear 16384->4096) entrypoint.
pub fn forwardFFLinear2(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d_ff });
    return params.out.forward(x_);
}

/// Focused GELU entrypoint matching current model behavior.
pub fn forwardFFGeluBf16(x: Tensor) Tensor {
    return x.gelu();
}

/// Focused GELU entrypoint that upcasts to f32 for GELU math then casts back.
pub fn forwardFFGeluF32(x: Tensor) Tensor {
    const dt = x.dtype();
    return x.convert(.f32).gelu().convert(dt);
}

/// Focused composed path: ff.net.0 projection followed by GELU.
pub fn forwardFFLinear1Gelu(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const h1 = params.proj.forward(x_);
    return forwardFFGeluF32(h1);
}

/// Focused composed path: ff.net.0 projection followed by GELU in f32 math.
pub fn forwardFFLinear1GeluF32(x: Tensor, params: FeedForward.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const h1 = params.proj.forward(x_);
    return forwardFFGeluF32(h1);
}

/// Build only block0.ff params from the selected transformer root.
pub fn initBlock0FFParams(store: zml.io.TensorStore.View) FeedForward.Params {
    const root = selectTransformerRoot(store);
    return FeedForward.initParams(root.withPrefix("transformer_blocks").withLayer(0).withPrefix("ff"));
}

pub fn initBlock0Params(store: zml.io.TensorStore.View) BasicAVTransformerBlock.Params {
    const root = selectTransformerRoot(store);
    return BasicAVTransformerBlock.initParams(root.withPrefix("transformer_blocks").withLayer(0));
}

pub fn unloadBlock0Buffers(params: *zml.Bufferized(BasicAVTransformerBlock.Params)) void {
    BasicAVTransformerBlock.unloadBuffers(params);
}

pub fn unloadBlock0FFBuffers(params: *zml.Bufferized(FeedForward.Params)) void {
    FeedForward.unloadBuffers(params);
}

pub fn initBlock0AudioFFParams(store: zml.io.TensorStore.View) FeedForward.Params {
    const root = selectTransformerRoot(store);
    return FeedForward.initParams(root.withPrefix("transformer_blocks").withLayer(0).withPrefix("audio_ff"));
}

pub fn unloadBlock0AudioFFBuffers(params: *zml.Bufferized(FeedForward.Params)) void {
    FeedForward.unloadBuffers(params);
}

/// Build patchify params from the selected transformer root.
pub fn initPatchifyParams(store: zml.io.TensorStore.View) Patchify.Params {
    const root = selectTransformerRoot(store);
    return Patchify.initParams(root);
}

/// Focused patchify-only entrypoint for parity bring-up.
pub fn forwardPatchify(x: Tensor, params: Patchify.Params) Tensor {
    const patchify = Patchify{};
    return patchify.forward(x, params);
}

pub fn unloadPatchifyBuffers(params: *zml.Bufferized(Patchify.Params)) void {
    Patchify.unloadBuffers(params);
}

/// Build audio patchify params from the selected transformer root.
pub fn initAudioPatchifyParams(store: zml.io.TensorStore.View) Patchify.Params {
    const root = selectTransformerRoot(store);
    return .init(
        root.withPrefix("audio_patchify_proj").createTensor("weight", .{ .d, .patch }, null),
        root.withPrefix("audio_patchify_proj").createTensor("bias", .{.d}, null),
        .patch,
    );
}

pub fn initBlock0AttentionParams(store: zml.io.TensorStore.View, kind: AttentionKind) Attention.Params {
    const root = selectTransformerRoot(store);
    return Attention.initParams(root.withPrefix("transformer_blocks").withLayer(0), kind);
}

pub fn unloadBlock0AttentionBuffers(params: *zml.Bufferized(Attention.Params)) void {
    Attention.unloadBuffers(params);
}

pub fn forwardBlock0Attn1(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn1), .{});
}

/// M1 – video self-attn given pre-computed AdaLN-normalised input.
///
/// Tests: attn1(norm_vx, pe_cos, pe_sin) == reference attn1 output.
///
/// `norm_vx` is the AdaLN-modulated query tensor fed into attn1 in the Python
/// BasicAVTransformerBlock:
///     norm_vx = rms_norm(vx) * (1 + vscale_msa) + vshift_msa
/// By accepting norm_vx directly the checker does not need the full AdaLN stack
/// and focuses purely on attention correctness with realistic conditioned inputs.
pub fn forwardBlock0VideoSelfAttn(norm_vx: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(norm_vx, params, kindNumHeads(.attn1), .{
        .pe_cos = pe_cos,
        .pe_sin = pe_sin,
    });
}

/// M1 full residual – video self-attn residual equation given pre-computed AdaLN values.
///
/// Tests: vx + attn1(norm_vx, pe) * vgate_msa == reference vx_out.
///
/// Python source (transformer.py):
///     vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
///         self.scale_shift_table, B, video.timesteps, slice(0, 3))
///     norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
///     vx = vx + self.attn1(norm_vx, pe=..., mask=...) * vgate_msa
///
/// `vx`       – [B, T, D]  video tokens before self-attn (residual base)
/// `norm_vx`  – [B, T, D]  AdaLN-normalised vx fed into attn1
/// `pe_cos`   – positional embeddings (cosine component)
/// `pe_sin`   – positional embeddings (sine component)
/// `vgate_msa` – [B, 1, D] AdaLN gate (broadcast over T)
pub fn forwardBlock0VideoSelfAttnResidual(vx: Tensor, norm_vx: Tensor, pe_cos: Tensor, pe_sin: Tensor, vgate_msa: Tensor, params: Attention.Params) Tensor {
    const attn_out = forwardBlock0VideoSelfAttn(norm_vx, pe_cos, pe_sin, params);
    return vx.withPartialTags(.{ .b, .t, .d }).add(attn_out.mul(vgate_msa.broad(attn_out.shape())));
}

/// M1 residual algebra-only check using precomputed attention output from fixture.
///
/// Tests: vx + attn1_out * vgate_msa == vx_out
/// without re-running attention, so this stage isolates gate+broadcast+residual math.
pub fn forwardBlock0VideoSelfAttnResidualFromAttnOut(vx: Tensor, attn_out: Tensor, vgate_msa: Tensor) Tensor {
    return vx.withPartialTags(.{ .b, .t, .d }).add(attn_out.withPartialTags(.{ .b, .t, .d }).mul(vgate_msa.broad(attn_out.shape())));
}

pub fn forwardBlock0Attn1WithPeCosSin(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn1), .{ .pe_cos = pe_cos, .pe_sin = pe_sin });
}

pub fn forwardBlock0Attn1WithPeCosSinMask(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, mask: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn1), .{ .pe_cos = pe_cos, .pe_sin = pe_sin, .mask = mask });
}

/// Diagnostic: returns q after to_q projection + RMS norm.
/// Output shape: [B, T, D_Q] — identical layout to Python's q_norm_diag0.
pub fn forwardBlock0Attn1DiagQNorm(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const q = params.to_q.forward(x_);
    return zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
}

/// Diagnostic: returns q after projection (pre-RMSNorm).
/// Output shape: [B, T, D_Q].
pub fn forwardBlock0Attn1DiagQProj(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const q = params.to_q.forward(x_);
    return q.withPartialTags(.{ .b, .t, .d_q });
}

/// Diagnostic: q projection with f32 accumulation, cast back to input dtype.
/// Output shape: [B, T, D_Q].
pub fn forwardBlock0Attn1DiagQProjF32(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const x_f32 = x_.convert(.f32);
    const weight_f32 = params.to_q.weight.convert(.f32);
    var y_f32 = x_f32.dot(weight_f32, .d);
    if (params.to_q.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    return y_f32.convert(x.dtype()).withPartialTags(.{ .b, .t, .d_q });
}

/// Diagnostic: returns qh after projection + norm + head-split + RoPE.
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagQRot(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const q = params.to_q.forward(x_);
    var qh = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    qh = qh.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    return qh.withPartialTags(.{ .b, .q, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: returns kh after projection + norm + head-split + RoPE.
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagKRot(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const k = params.to_k.forward(x_);
    var kh = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));
    kh = kh.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const k_cos = (if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd })).rename(.{ .q = .k });
    const k_sin = (if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd })).rename(.{ .q = .k });

    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);
    return kh.withPartialTags(.{ .b, .k, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: returns k after projection (pre-RMSNorm).
/// Output shape: [B, T, D_K].
pub fn forwardBlock0Attn1DiagKProj(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const k = params.to_k.forward(x_);
    return k.withPartialTags(.{ .b, .t, .d_k });
}

/// Diagnostic: k projection with f32 accumulation, cast back to input dtype.
/// Output shape: [B, T, D_K].
pub fn forwardBlock0Attn1DiagKProjF32(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const x_f32 = x_.convert(.f32);
    const weight_f32 = params.to_k.weight.convert(.f32);
    var y_f32 = x_f32.dot(weight_f32, .d);
    if (params.to_k.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    return y_f32.convert(x.dtype()).withPartialTags(.{ .b, .t, .d_k });
}

/// Diagnostic: returns vh after projection + head-split (no RoPE for V).
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagVProj(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const v = params.to_v.forward(x_);
    return v.withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic helper: split a precomputed V projection [B, T, Dv] into heads.
/// Output shape: [B, T, H, HD].
pub fn forwardBlock0Attn1DiagVHeadFromProj(v_proj: Tensor) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const v = v_proj.withPartialTags(.{ .b, .t, .d_v });
    const vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    return vh.withPartialTags(.{ .b, .k, .h, .hd });
}

/// Diagnostic: returns vh after projection + head-split (no RoPE for V).
/// Output shape: [B, T, H, HD] — ZML-native layout matching Python reference saved with layout=BTHD.
pub fn forwardBlock0Attn1DiagVHead(x: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const v = params.to_v.forward(x_);
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    return vh.withPartialTags(.{ .b, .k, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: V projection using f32 accumulation (precision-corrected variant).
/// This upcasts context to f32 before matmul for accumulated precision, then downcasts back.
/// Returns: [B, T, Dv]
pub fn forwardBlock0Attn1DiagVProjF32(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    // Upcast to f32 for matmul accumulation
    const x_f32 = x_.convert(.f32);
    // Weights must be upcast too; assume weight stored in same dtype as bias (typically bf16)
    const weight_f32 = params.to_v.weight.convert(.f32);
    // Perform matmul in f32
    var y_f32 = x_f32.dot(weight_f32, .d);
    // Add bias if present
    if (params.to_v.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    // Downcast back to original dtype
    const v = y_f32.convert(x.dtype());
    return v.withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic: V projection head-split using f32 accumulation (precision-corrected variant).
/// Returns: [B, T, H, HD]
pub fn forwardBlock0Attn1DiagVHeadF32(x: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const v_f32 = forwardBlock0Attn1DiagVProjF32(x, params);
    var vh = v_f32.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    return vh.withPartialTags(.{ .b, .k, .h, .hd }); // [B, T, H, HD]
}

/// Diagnostic: returns raw SDPA output before gate and before merge/to_out.
/// Output shape: [B, T, H, HD]
pub fn forwardBlock0Attn1DiagSdpaOut(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd });
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    const attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    return attn.withPartialTags(.{ .b, .q, .h, .hd });
}

/// Diagnostic: returns raw SDPA output computed manually in f32 (before gate/merge).
/// Output shape: [B, T, H, HD]
pub fn forwardBlock0Attn1DiagSdpaOutManualF32(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const out_dtype = x.dtype();
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    const hd: f32 = @floatFromInt(qh.dim(.hd));
    const scale = Tensor.scalar(1.0 / std.math.sqrt(hd), .f32);
    const k_scaled = kh.mul(scale.broad(kh.shape()));

    var scores = qh.dot(k_scaled, .hd).softmax(.k);
    var attn = scores.dot(vh, .k);
    attn = attn.transpose(qh.shape()).convert(out_dtype);
    return attn.withPartialTags(.{ .b, .q, .h, .hd });
}

/// Diagnostic: returns gate logits before sigmoid/scale.
/// Output shape: [B, T, H]
pub fn forwardBlock0Attn1DiagGateLogits(x: Tensor, params: Attention.Params) Tensor {
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    return params.to_gate_logits.forward(x_).withPartialTags(.{ .b, .t, .h });
}

/// Diagnostic: applies only to_out projection on a pre_to_out tensor.
/// Input shape: [B, T, D_V], output shape: [B, T, D].
pub fn forwardBlock0Attn1DiagToOutOnly(pre_to_out: Tensor, params: Attention.Params) Tensor {
    const x_ = pre_to_out.withPartialTags(.{ .b, .t, .d_v });
    return params.to_out.forward(x_).withPartialTags(.{ .b, .t, .d });
}


/// Diagnostic: to_out projection with f32 accumulation, cast back to input dtype.
/// Used to isolate whether PyTorch uses f32 intermediate accumulation in to_out.
/// Input shape: [B, T, D_V], output shape: [B, T, D].
pub fn forwardBlock0Attn1DiagToOutOnlyF32(pre_to_out: Tensor, params: Attention.Params) Tensor {
    const x_ = pre_to_out.withPartialTags(.{ .b, .t, .d_v });
    const x_f32 = x_.convert(.f32);
    const weight_f32 = params.to_out.weight.convert(.f32);
    var y_f32 = x_f32.dot(weight_f32, .d_v);
    if (params.to_out.bias) |bias_orig| {
        const bias_f32 = bias_orig.convert(.f32);
        y_f32 = y_f32.add(bias_f32.broad(y_f32.shape()));
    }
    return y_f32.convert(pre_to_out.dtype()).withPartialTags(.{ .b, .t, .d });
}

/// Diagnostic: full attn1 path in f32 for all linear/attention intermediates, cast back at output.
/// This isolates the best-case gain obtainable from precision-only changes.
pub fn forwardBlock0Attn1DiagAllF32(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const out_dtype = x.dtype();
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const x_f32 = x_.convert(.f32);

    var q = x_f32.dot(params.to_q.weight.convert(.f32), .d);
    if (params.to_q.bias) |bias_orig| {
        q = q.add(bias_orig.convert(.f32).broad(q.shape()));
    }

    var k = x_f32.dot(params.to_k.weight.convert(.f32), .d);
    if (params.to_k.bias) |bias_orig| {
        k = k.add(bias_orig.convert(.f32).broad(k.shape()));
    }

    var v = x_f32.dot(params.to_v.weight.convert(.f32), .d);
    if (params.to_v.bias) |bias_orig| {
        v = v.add(bias_orig.convert(.f32).broad(v.shape()));
    }

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.convert(.f32).broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.convert(.f32).broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd }).convert(.f32)
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd }).convert(.f32)
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    var gate_logits = x_f32.dot(params.to_gate_logits.weight.convert(.f32), .d);
    if (params.to_gate_logits.bias) |bias_orig| {
        gate_logits = gate_logits.add(bias_orig.convert(.f32).broad(gate_logits.shape()));
    }
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    const merged = attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t });
    var out_f32 = merged.dot(params.to_out.weight.convert(.f32), .d_v);
    if (params.to_out.bias) |bias_orig| {
        out_f32 = out_f32.add(bias_orig.convert(.f32).broad(out_f32.shape()));
    }

    return out_f32.convert(out_dtype).withPartialTags(.{ .b, .t, .d });
}

/// Diagnostic: returns merged attention after SDPA and gate, before to_out.
/// Output shape: [B, T, D_V]
pub fn forwardBlock0Attn1DiagPreToOut(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: same as forwardBlock0Attn1DiagPreToOut, but runs SDPA core in f32.
/// This helps isolate whether bf16 attention precision is the source of parity drift.
pub fn forwardBlock0Attn1DiagPreToOutF32Sdpa(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    var gate_logits = params.to_gate_logits.forward(x_).convert(.f32);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: pre-to_out with no gate applied.
pub fn forwardBlock0Attn1DiagPreToOutNoGate(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    const attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: pre-to_out with sigmoid(gate_logits) gate (without *2 scaling).
pub fn forwardBlock0Attn1DiagPreToOutSigmoidGate(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: explicit SDPA math (f32) instead of zml.nn.sdpa.
/// This isolates potential semantic differences inside the shared sdpa primitive.
pub fn forwardBlock0Attn1DiagPreToOutManualSdpaF32(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    const hd: f32 = @floatFromInt(qh.dim(.hd));
    const scale = Tensor.scalar(1.0 / std.math.sqrt(hd), .f32);
    const k_scaled = kh.mul(scale.broad(kh.shape()));

    var attn_weights = qh.dot(k_scaled, .hd).softmax(.k);
    var attn = attn_weights.dot(vh, .k);
    attn = attn.transpose(qh.shape());

    const gate_logits = params.to_gate_logits.forward(x_).convert(.f32);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: run sdpa with head-first layout [b,h,q,hd] like PyTorch,
/// then swap back to [b,q,h,hd] before merge.
pub fn forwardBlock0Attn1DiagPreToOutHeadFirstSdpa(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).swapAxes(.q, .h);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).swapAxes(.k, .h);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).swapAxes(.k, .h);

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    // Bring back to [b,q,h,hd] so gating + merge match the main pipeline.
    attn = attn.swapAxes(.h, .q);

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: same as pre_to_out with sigmoid gate, but flatten heads as [hd, h].
/// Uses mergeTranspose to safely merge non-contiguous axes.
pub fn forwardBlock0Attn1DiagPreToOutAltMergeTranspose(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin);

    qh = qh.withPartialTags(.{ .b, .q, .h, .hd });
    kh = kh.withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.mergeTranspose(.{ .hd, .h }, .d_v).rename(.{ .q = .t }).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: head-first layout [b,h,q,hd] with native zml.nn.sdpa only
/// (no manual SDPA, no f32 accumulation). Tests if layout alone explains the gap.
pub fn forwardBlock0Attn1DiagPreToOutHeadFirstOnly(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd });
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd });
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd });

    // Reorder to head-first [b,h,q,hd] before SDPA
    qh = qh.swapAxes(.q, .h);
    kh = kh.swapAxes(.k, .h);
    vh = vh.swapAxes(.k, .h);

    var attn = zml.nn.sdpa(
        qh.rename(.{ .b = .batch }),
        kh.rename(.{ .b = .batch }),
        vh.rename(.{ .b = .batch }),
        .{},
    ).rename(.{ .batch = .b });

    // Reorder back to [b,q,h,hd] for gating + merge
    attn = attn.swapAxes(.h, .q);

    const gate_logits = params.to_gate_logits.forward(x_);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

/// Diagnostic ablation: manual SDPA f32 logic with head-first layout [b,h,q,hd].
/// Combines both fixes: explicit scaling/softmax in f32 + PyTorch head-first layout.
pub fn forwardBlock0Attn1DiagPreToOutManualSdpaHeadFirst(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const num_heads = kindNumHeads(.attn1);
    const x_ = x.withPartialTags(.{ .b, .t, .d });
    const out_dtype = x.dtype();

    var q = params.to_q.forward(x_);
    var k = params.to_k.forward(x_);
    var v = params.to_v.forward(x_);

    q = zml.nn.rmsNorm(q, .d_q, 1e-6).mul(params.q_norm_weight.broad(q.shape()));
    k = zml.nn.rmsNorm(k, .d_k, 1e-6).mul(params.k_norm_weight.broad(k.shape()));

    var qh = q.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var kh = k.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });
    var vh = v.rename(.{ .t = .k }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto });

    const q_cos = if (pe_cos.rank() == 4)
        pe_cos.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_cos.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const q_sin = if (pe_sin.rank() == 4)
        pe_sin.withPartialTags(.{ .b, .h, .q, .hd })
    else
        pe_sin.rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), .hd = .auto }).withPartialTags(.{ .b, .q, .h, .hd });
    const k_cos = q_cos.rename(.{ .q = .k });
    const k_sin = q_sin.rename(.{ .q = .k });

    qh = Attention.applyLtxRotaryEmb(qh, q_cos, q_sin).withPartialTags(.{ .b, .q, .h, .hd }).convert(.f32);
    kh = Attention.applyLtxRotaryEmb(kh, k_cos, k_sin).withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);
    vh = vh.withPartialTags(.{ .b, .k, .h, .hd }).convert(.f32);

    // Reorder to head-first [b,h,q,hd] before manual SDPA
    qh = qh.swapAxes(.q, .h);
    kh = kh.swapAxes(.k, .h);
    vh = vh.swapAxes(.k, .h);

    const hd: f32 = @floatFromInt(qh.dim(.hd));
    const scale = Tensor.scalar(1.0 / std.math.sqrt(hd), .f32);
    const k_scaled = kh.mul(scale.broad(kh.shape()));

    var attn_weights = qh.dot(k_scaled, .hd).softmax(.k);
    var attn = attn_weights.dot(vh, .k);

    // Reorder back to [b,q,h,hd] for gating + merge
    attn = attn.swapAxes(.h, .q);

    const gate_logits = params.to_gate_logits.forward(x_).convert(.f32);
    const gate = gate_logits.sigmoid().scale(2.0).rename(.{ .t = .q }).splitAxis(-1, .{ .h = @as(i64, @intCast(num_heads)), ._gate = .auto }).squeeze(._gate);
    attn = attn.mul(gate.broad(attn.shape()));

    return attn.merge(.{ .d_v = .{ .h, .hd } }).rename(.{ .q = .t }).convert(out_dtype).withPartialTags(.{ .b, .t, .d_v });
}

pub fn forwardBlock0Attn2(x: Tensor, context: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn2), .{ .context = context });
}

pub fn forwardBlock0Attn2WithContextMask(x: Tensor, context: Tensor, context_mask: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.attn2), .{ .context = context, .mask = context_mask });
}

/// M2 residual algebra-only check using precomputed text cross-attn delta from fixture.
///
/// Tests: vx_after_self_attn + text_ca_out == vx_after_text_ca
/// without re-running text cross-attention internals.
pub fn forwardBlock0VideoTextCaResidualFromDelta(vx: Tensor, text_ca_out: Tensor) Tensor {
    return vx.withPartialTags(.{ .b, .t, .d }).add(text_ca_out.withPartialTags(.{ .b, .t, .d }));
}

/// M3 residual algebra-only check using precomputed FF output from fixture.
///
/// Tests: vx_before_ff + ff_out * vgate_mlp == vx_after_ff
pub fn forwardBlock0VideoFFResidualFromFFOut(vx: Tensor, ff_out: Tensor, vgate_mlp: Tensor) Tensor {
    return vx.withPartialTags(.{ .b, .t, .d }).add(ff_out.withPartialTags(.{ .b, .t, .d }).mul(vgate_mlp.broad(ff_out.shape())));
}

pub fn forwardBlock0AudioAttn1(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn1), .{});
}

pub fn forwardBlock0AudioAttn1WithPeCosSin(x: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn1), .{ .pe_cos = pe_cos, .pe_sin = pe_sin });
}

pub fn forwardBlock0AudioAttn2(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn2), .{});
}

pub fn forwardBlock0AudioAttn2WithContext(x: Tensor, context: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_attn2), .{ .context = context });
}

pub fn forwardBlock0AudioToVideoAttn(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_to_video_attn), .{});
}

pub fn forwardBlock0AudioToVideoAttnWithContextPeKPe(x: Tensor, context: Tensor, pe_cos: Tensor, pe_sin: Tensor, k_pe_cos: Tensor, k_pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.audio_to_video_attn), .{
        .context = context,
        .pe_cos = pe_cos,
        .pe_sin = pe_sin,
        .k_pe_cos = k_pe_cos,
        .k_pe_sin = k_pe_sin,
    });
}

pub fn forwardBlock0VideoToAudioAttn(x: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.video_to_audio_attn), .{});
}

pub fn forwardBlock0VideoToAudioAttnWithContextPeKPe(x: Tensor, context: Tensor, pe_cos: Tensor, pe_sin: Tensor, k_pe_cos: Tensor, k_pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(x, params, kindNumHeads(.video_to_audio_attn), .{
        .context = context,
        .pe_cos = pe_cos,
        .pe_sin = pe_sin,
        .k_pe_cos = k_pe_cos,
        .k_pe_sin = k_pe_sin,
    });
}

// ── M4: Audio branch parity entrypoints ────────────────────────────────────

/// M4-A – audio self-attn given pre-computed AdaLN-normalised input.
///
/// Tests: audio_attn1(norm_ax, pe_cos, pe_sin) == reference audio_attn1 output.
///
/// Python source (transformer.py):
///     ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
///         self.audio_scale_shift_table, B, audio.timesteps, slice(0, 3))
///     norm_ax = rms_norm(ax) * (1 + ascale_msa) + ashift_msa
///     ax = ax + self.audio_attn1(norm_ax, pe=audio.positional_embeddings, ...) * agate_msa
pub fn forwardBlock0AudioSelfAttn(norm_ax: Tensor, pe_cos: Tensor, pe_sin: Tensor, params: Attention.Params) Tensor {
    const attn = Attention{};
    return attn.forward(norm_ax, params, kindNumHeads(.audio_attn1), .{
        .pe_cos = pe_cos,
        .pe_sin = pe_sin,
    });
}

/// M4-A residual algebra-only check using precomputed audio self-attn output from fixture.
///
/// Tests: ax + audio_attn1_out * agate_msa == ax_out
pub fn forwardBlock0AudioSelfAttnResidualFromAttnOut(ax: Tensor, attn_out: Tensor, agate_msa: Tensor) Tensor {
    return ax.withPartialTags(.{ .b, .t, .d }).add(attn_out.withPartialTags(.{ .b, .t, .d }).mul(agate_msa.broad(attn_out.shape())));
}

/// M4-B residual algebra-only check using precomputed audio text-ca delta from fixture.
///
/// Tests: ax_after_self_attn + audio_text_ca_out == ax_after_text_ca
///
/// Python source (transformer.py, _apply_text_cross_attention for audio):
///     ax = ax + self._apply_text_cross_attention(ax, audio.context, self.audio_attn2, ...)
pub fn forwardBlock0AudioTextCaResidualFromDelta(ax: Tensor, audio_text_ca_out: Tensor) Tensor {
    return ax.withPartialTags(.{ .b, .t, .d }).add(audio_text_ca_out.withPartialTags(.{ .b, .t, .d }));
}

/// M4-C audio FF given pre-computed AdaLN-normalised input.
///
/// Tests: audio_ff(ax_scaled) == reference audio_ff_out.
///
/// Python source (transformer.py):
///     ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
///         self.audio_scale_shift_table, B, audio.timesteps, slice(3, 6))
///     ax_scaled = rms_norm(ax) * (1 + ascale_mlp) + ashift_mlp
///     ax = ax + self.audio_ff(ax_scaled) * agate_mlp
pub fn forwardBlock0AudioFF(ax_scaled: Tensor, params: FeedForward.Params) Tensor {
    const ff = FeedForward{};
    return ff.forward(ax_scaled, params);
}

/// M4-C residual algebra-only check using precomputed audio FF output from fixture.
///
/// Tests: ax_before_ff + audio_ff_out * agate_mlp == ax_after_ff
pub fn forwardBlock0AudioFFResidualFromFFOut(ax: Tensor, ff_out: Tensor, agate_mlp: Tensor) Tensor {
    return ax.withPartialTags(.{ .b, .t, .d }).add(ff_out.withPartialTags(.{ .b, .t, .d }).mul(agate_mlp.broad(ff_out.shape())));
}

// ── M5: AV cross-attn parity entrypoints ──────────────────────────────────

/// M5-A gated residual delta for A->V branch.
///
/// Tests: a2v_delta == audio_to_video_attn_out * gate_out_a2v * a2v_mask
pub fn forwardBlock0A2VDeltaFromAttnOut(attn_out: Tensor, gate: Tensor, mask: Tensor) Tensor {
    return attn_out.withPartialTags(.{ .b, .t, .d })
        .mul(gate.broad(attn_out.shape()))
        .mul(mask.broad(attn_out.shape()));
}

/// M5-B gated residual delta for V->A branch.
///
/// Tests: v2a_delta == video_to_audio_attn_out * gate_out_v2a * v2a_mask
pub fn forwardBlock0V2ADeltaFromAttnOut(attn_out: Tensor, gate: Tensor, mask: Tensor) Tensor {
    return attn_out.withPartialTags(.{ .b, .t, .d })
        .mul(gate.broad(attn_out.shape()))
        .mul(mask.broad(attn_out.shape()));
}

// ── M6: full block-0 stream parity entrypoints ────────────────────────────

/// Combined params for full block-0 parity check (M6).
/// This now aliases the canonical block params used by BasicAVTransformerBlock.
pub const Block0FullParams = BasicAVTransformerBlock.Params;

pub fn initBlock0FullParams(store: zml.io.TensorStore.View) Block0FullParams {
    return initBlock0Params(store);
}

pub fn unloadBlock0FullBuffers(params: *zml.Bufferized(Block0FullParams)) void {
    unloadBlock0Buffers(params);
}

/// M6 – full video stream forward.
///
/// Composes M1 → M2 → M5-A → M3 for the video token stream:
///   vx = vx_in + attn1(norm_vx, pe) * vgate_msa          // M1
///   vx = vx   + attn2(v_text_x, v_text_ctx)              // M2
///   vx = vx   + a2v_attn(a2v_x, a2v_ctx, pe, k_pe)
///                   * a2v_gate * a2v_mask                 // M5-A
///   vx = vx   + ff(vx_scaled) * vgate_mlp                // M3
///
/// `norm_vx`, `v_text_x`, `a2v_x`, and `vx_scaled` are exact per-module inputs
/// captured from Python to preserve hidden preprocessing in text/AV cross-attn.
pub fn forwardBlock0VideoStream(
    vx_in: Tensor,
    norm_vx: Tensor,
    v_text_x: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    vgate_msa: Tensor,
    vgate_text_ca: Tensor,
    v_text_ctx: Tensor,
    a2v_x: Tensor,
    a2v_ctx: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_gate: Tensor,
    a2v_mask: Tensor,
    vx_scaled: Tensor,
    vgate_mlp: Tensor,
    params: Block0FullParams,
) Tensor {
    const block = BasicAVTransformerBlock.init();
    return block.forwardVideoStream(.{
        .vx_in = vx_in,
        .norm_vx = norm_vx,
        .v_text_x = v_text_x,
        .v_pe_cos = v_pe_cos,
        .v_pe_sin = v_pe_sin,
        .vgate_msa = vgate_msa,
        .vgate_text_ca = vgate_text_ca,
        .v_text_ctx = v_text_ctx,
        .a2v_x = a2v_x,
        .a2v_ctx = a2v_ctx,
        .a2v_pe_cos = a2v_pe_cos,
        .a2v_pe_sin = a2v_pe_sin,
        .a2v_k_pe_cos = a2v_k_pe_cos,
        .a2v_k_pe_sin = a2v_k_pe_sin,
        .a2v_gate = a2v_gate,
        .a2v_mask = a2v_mask,
        .vx_scaled = vx_scaled,
        .vgate_mlp = vgate_mlp,
    }, params);
}

/// M6 – full audio stream forward.
///
/// Composes M4-A → M4-B → M5-B → M4-C for the audio token stream:
///   ax = ax_in + audio_attn1(norm_ax, pe) * agate_msa    // M4-A
///   ax = ax   + audio_attn2(a_text_x, a_text_ctx)        // M4-B
///   ax = ax   + v2a_attn(v2a_x, v2a_ctx, pe, k_pe)
///                   * v2a_gate * v2a_mask                 // M5-B
///   ax = ax   + audio_ff(ax_scaled) * agate_mlp          // M4-C
///
/// `norm_ax`, `a_text_x`, `v2a_x`, and `ax_scaled` are exact per-module inputs
/// captured from Python to preserve hidden preprocessing in text/AV cross-attn.
pub fn forwardBlock0AudioStream(
    ax_in: Tensor,
    norm_ax: Tensor,
    a_text_x: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    agate_msa: Tensor,
    agate_text_ca: Tensor,
    a_text_ctx: Tensor,
    v2a_x: Tensor,
    v2a_ctx: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_gate: Tensor,
    v2a_mask: Tensor,
    ax_scaled: Tensor,
    agate_mlp: Tensor,
    params: Block0FullParams,
) Tensor {
    const block = BasicAVTransformerBlock.init();
    return block.forwardAudioStream(.{
        .ax_in = ax_in,
        .norm_ax = norm_ax,
        .a_text_x = a_text_x,
        .a_pe_cos = a_pe_cos,
        .a_pe_sin = a_pe_sin,
        .agate_msa = agate_msa,
        .agate_text_ca = agate_text_ca,
        .a_text_ctx = a_text_ctx,
        .v2a_x = v2a_x,
        .v2a_ctx = v2a_ctx,
        .v2a_pe_cos = v2a_pe_cos,
        .v2a_pe_sin = v2a_pe_sin,
        .v2a_k_pe_cos = v2a_k_pe_cos,
        .v2a_k_pe_sin = v2a_k_pe_sin,
        .v2a_gate = v2a_gate,
        .v2a_mask = v2a_mask,
        .ax_scaled = ax_scaled,
        .agate_mlp = agate_mlp,
    }, params);
}

pub const Block0VideoStageOutputs = struct {
    vx_after_msa: Tensor,
    vx_after_text_ca: Tensor,
    vx_after_a2v: Tensor,
    ff_proj_out: Tensor,
    ff_gelu_out: Tensor,
    ff_out: Tensor,
    vx_out: Tensor,
};

/// Debug path for block-0 video stream that returns intermediate residual states.
pub fn forwardBlock0VideoStreamStages(
    vx_in: Tensor,
    norm_vx: Tensor,
    v_text_x: Tensor,
    v_pe_cos: Tensor,
    v_pe_sin: Tensor,
    vgate_msa: Tensor,
    vgate_text_ca: Tensor,
    v_text_ctx: Tensor,
    a2v_x: Tensor,
    a2v_ctx: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_gate: Tensor,
    a2v_mask: Tensor,
    vx_scaled: Tensor,
    vgate_mlp: Tensor,
    params: Block0FullParams,
) Block0VideoStageOutputs {
    const block = BasicAVTransformerBlock.init();

    const attn1_out = block.attn1.forward(norm_vx, params.attn1, kindNumHeads(.attn1), .{
        .pe_cos = v_pe_cos,
        .pe_sin = v_pe_sin,
    });
    const vx_after_msa = vx_in.withPartialTags(.{ .b, .t, .d })
        .add(attn1_out.mul(vgate_msa.broad(attn1_out.shape())));

    const text_ca_out = block.attn2.forward(v_text_x, params.attn2, kindNumHeads(.attn2), .{
        .context = v_text_ctx,
    });
    const vx_after_text_ca = vx_after_msa.add(text_ca_out.mul(vgate_text_ca.broad(text_ca_out.shape())));

    const a2v_out = block.audio_to_video_attn.forward(a2v_x, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
        .context = a2v_ctx,
        .pe_cos = a2v_pe_cos,
        .pe_sin = a2v_pe_sin,
        .k_pe_cos = a2v_k_pe_cos,
        .k_pe_sin = a2v_k_pe_sin,
    });
    const vx_after_a2v = vx_after_text_ca.add(
        a2v_out.mul(a2v_gate.broad(a2v_out.shape())).mul(a2v_mask.broad(a2v_out.shape())),
    );

    const ff_proj_out = linearForwardF32(vx_scaled.withPartialTags(.{ .b, .t, .d }), params.ff.proj);
    const ff_gelu_out = ff_proj_out.gelu();
    const ff_out = linearForwardF32(ff_gelu_out.withPartialTags(.{ .b, .t, .d_ff }), params.ff.out).convert(vx_in.dtype());
    const vx_out = vx_after_a2v.add(ff_out.mul(vgate_mlp.broad(ff_out.shape())));

    return .{
        .vx_after_msa = vx_after_msa,
        .vx_after_text_ca = vx_after_text_ca,
        .vx_after_a2v = vx_after_a2v,
        .ff_proj_out = ff_proj_out.convert(vx_in.dtype()),
        .ff_gelu_out = ff_gelu_out.convert(vx_in.dtype()),
        .ff_out = ff_out,
        .vx_out = vx_out,
    };
}

pub const Block0AudioStageOutputs = struct {
    ax_after_msa: Tensor,
    ax_after_text_ca: Tensor,
    ax_after_v2a: Tensor,
    ff_proj_out: Tensor,
    ff_gelu_out: Tensor,
    ff_out: Tensor,
    ax_out: Tensor,
};

/// Debug path for block-0 audio stream that returns intermediate residual states.
pub fn forwardBlock0AudioStreamStages(
    ax_in: Tensor,
    norm_ax: Tensor,
    a_text_x: Tensor,
    a_pe_cos: Tensor,
    a_pe_sin: Tensor,
    agate_msa: Tensor,
    agate_text_ca: Tensor,
    a_text_ctx: Tensor,
    v2a_x: Tensor,
    v2a_ctx: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_gate: Tensor,
    v2a_mask: Tensor,
    ax_scaled: Tensor,
    agate_mlp: Tensor,
    params: Block0FullParams,
) Block0AudioStageOutputs {
    const block = BasicAVTransformerBlock.init();

    const attn1_out = block.audio_attn1.forward(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{
        .pe_cos = a_pe_cos,
        .pe_sin = a_pe_sin,
    });
    const ax_after_msa = ax_in.withPartialTags(.{ .b, .t, .d })
        .add(attn1_out.mul(agate_msa.broad(attn1_out.shape())));

    const text_ca_out = block.audio_attn2.forward(a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{
        .context = a_text_ctx,
    });
    const ax_after_text_ca = ax_after_msa.add(text_ca_out.mul(agate_text_ca.broad(text_ca_out.shape())));

    const v2a_out = block.video_to_audio_attn.forward(v2a_x, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
        .context = v2a_ctx,
        .pe_cos = v2a_pe_cos,
        .pe_sin = v2a_pe_sin,
        .k_pe_cos = v2a_k_pe_cos,
        .k_pe_sin = v2a_k_pe_sin,
    });
    const ax_after_v2a = ax_after_text_ca.add(
        v2a_out.mul(v2a_gate.broad(v2a_out.shape())).mul(v2a_mask.broad(v2a_out.shape())),
    );

    const ff_proj_out = linearForwardF32(ax_scaled.withPartialTags(.{ .b, .t, .d }), params.audio_ff.proj);
    const ff_gelu_out = ff_proj_out.gelu();
    const ff_out = linearForwardF32(ff_gelu_out.withPartialTags(.{ .b, .t, .d_ff }), params.audio_ff.out).convert(ax_in.dtype());
    const ax_out = ax_after_v2a.add(ff_out.mul(agate_mlp.broad(ff_out.shape())));

    return .{
        .ax_after_msa = ax_after_msa,
        .ax_after_text_ca = ax_after_text_ca,
        .ax_after_v2a = ax_after_v2a,
        .ff_proj_out = ff_proj_out.convert(ax_in.dtype()),
        .ff_gelu_out = ff_gelu_out.convert(ax_in.dtype()),
        .ff_out = ff_out,
        .ax_out = ax_out,
    };
}

/// Native block-0 video stream forward using inline AdaLN computation.
///
/// This path mirrors Python block logic by deriving modulation values directly
/// from scale-shift tables + timestep embeddings (no precomputed norm/gate inputs).
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

/// Checker-only experimental entrypoint: same as forwardBlock0NativeWithAVMasks,
/// but computes all video residual adds in f32 and video FF uses f32 matmul.
pub fn forwardBlock0NativeWithAVMasksVideoAllResidualsF32(
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
    return block.forwardNativeVideoAllResidualsF32(vx_in, ax_in, .{
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

pub const Block0NativeAudioIntermediates = struct {
    norm_ax: Tensor,
    a_text_x: Tensor,
    v2a_x: Tensor,
    ax_scaled_ff: Tensor,
};

pub const Block0NativeVideoIntermediates = struct {
    norm_vx: Tensor,
    v_text_x: Tensor,
    vx_norm3: Tensor,
    a2v_x: Tensor,
    a2v_ctx: Tensor,
    a2v_gate: Tensor,
    vx_scaled_ff: Tensor,
};

/// Debug helper: returns native-computed video intermediates used by the block path.
pub fn forwardBlock0NativeVideoIntermediatesWithAVMasks(
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
) Block0NativeVideoIntermediates {
    const block = BasicAVTransformerBlock.init();

    const vx = vx_in.withPartialTags(.{ .b, .t, .d });
    const ax = ax_in.withPartialTags(.{ .b, .t, .d });

    const vshift_msa = adaValueAt(params.scale_shift_table, video_timesteps, 0);
    const vscale_msa = adaValueAt(params.scale_shift_table, video_timesteps, 1);
    const vgate_msa = adaValueAt(params.scale_shift_table, video_timesteps, 2);
    const norm_vx = zml.nn.rmsNorm(vx, .d, 1e-6)
        .mul(vscale_msa.addConstant(1.0).broad(vx.shape()))
        .add(vshift_msa.broad(vx.shape()));
    const attn1_out = block.attn1.forward(norm_vx, params.attn1, kindNumHeads(.attn1), .{ .pe_cos = v_pe_cos, .pe_sin = v_pe_sin });
    var h_v = vx.add(attn1_out.mul(vgate_msa.broad(attn1_out.shape())));

    const v_shift_q = adaValueAt(params.scale_shift_table, video_timesteps, 6);
    const v_scale_q = adaValueAt(params.scale_shift_table, video_timesteps, 7);
    const v_gate_q = adaValueAt(params.scale_shift_table, video_timesteps, 8);
    const v_text_x = zml.nn.rmsNorm(h_v, .d, 1e-6)
        .mul(v_scale_q.addConstant(1.0).broad(h_v.shape()))
        .add(v_shift_q.broad(h_v.shape()));
    const v_shift_kv = adaValueAt(params.prompt_scale_shift_table, v_prompt_timestep, 0);
    const v_scale_kv = adaValueAt(params.prompt_scale_shift_table, v_prompt_timestep, 1);
    const v_text_ctx_mod = v_text_ctx
        .mul(v_scale_kv.addConstant(1.0).broad(v_text_ctx.shape()))
        .add(v_shift_kv.broad(v_text_ctx.shape()));
    const v_text_ca_out = block.attn2.forward(v_text_x, params.attn2, kindNumHeads(.attn2), .{ .context = v_text_ctx_mod });
    h_v = h_v.add(v_text_ca_out.mul(v_gate_q.broad(v_text_ca_out.shape())));

    const ashift_msa = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 0);
    const ascale_msa = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 1);
    const agate_msa = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 2);
    const norm_ax = zml.nn.rmsNorm(ax, .d, 1e-6)
        .mul(ascale_msa.addConstant(1.0).broad(ax.shape()))
        .add(ashift_msa.broad(ax.shape()));
    const audio_attn1_out = block.audio_attn1.forward(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{ .pe_cos = a_pe_cos, .pe_sin = a_pe_sin });
    var h_a = ax.add(audio_attn1_out.mul(agate_msa.broad(audio_attn1_out.shape())));

    const a_shift_q = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 6);
    const a_scale_q = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 7);
    const a_gate_q = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 8);
    const a_text_x = zml.nn.rmsNorm(h_a, .d, 1e-6)
        .mul(a_scale_q.addConstant(1.0).broad(h_a.shape()))
        .add(a_shift_q.broad(h_a.shape()));
    const a_shift_kv = adaValueAt(params.audio_prompt_scale_shift_table, a_prompt_timestep, 0);
    const a_scale_kv = adaValueAt(params.audio_prompt_scale_shift_table, a_prompt_timestep, 1);
    const a_text_ctx_mod = a_text_ctx
        .mul(a_scale_kv.addConstant(1.0).broad(a_text_ctx.shape()))
        .add(a_shift_kv.broad(a_text_ctx.shape()));
    const a_text_ca_out = block.audio_attn2.forward(a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{ .context = a_text_ctx_mod });
    h_a = h_a.add(a_text_ca_out.mul(a_gate_q.broad(a_text_ca_out.shape())));

    const vx_norm3 = zml.nn.rmsNorm(h_v, .d, 1e-6);
    const ax_norm3 = zml.nn.rmsNorm(h_a, .d, 1e-6);

    const sst_v_ss = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 0, .end = 4 });
    const sst_v_gate = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 4, .end = 5 });
    const scale_v_a2v = adaValueAt(sst_v_ss, v_cross_ss_ts, 0);
    const shift_v_a2v = adaValueAt(sst_v_ss, v_cross_ss_ts, 1);
    const gate_a2v = adaValueAt(sst_v_gate, v_cross_gate_ts, 0);
    const vx_scaled_a2v = vx_norm3
        .mul(scale_v_a2v.addConstant(1.0).broad(vx_norm3.shape()))
        .add(shift_v_a2v.broad(vx_norm3.shape()));

    const sst_a_ss = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 0, .end = 4 });
    const sst_a_gate = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 4, .end = 5 });
    const scale_a_a2v = adaValueAt(sst_a_ss, a_cross_ss_ts, 0);
    const shift_a_a2v = adaValueAt(sst_a_ss, a_cross_ss_ts, 1);
    const ax_scaled_a2v = ax_norm3
        .mul(scale_a_a2v.addConstant(1.0).broad(ax_norm3.shape()))
        .add(shift_a_a2v.broad(ax_norm3.shape()));

    const a2v_out = block.audio_to_video_attn.forward(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
        .context = ax_scaled_a2v,
        .pe_cos = a2v_pe_cos,
        .pe_sin = a2v_pe_sin,
        .k_pe_cos = a2v_k_pe_cos,
        .k_pe_sin = a2v_k_pe_sin,
    });
    h_v = h_v.add(a2v_out.mul(gate_a2v.broad(a2v_out.shape())).mul(a2v_mask.broad(a2v_out.shape())));

    const scale_a_v2a = adaValueAt(sst_a_ss, a_cross_ss_ts, 2);
    const shift_a_v2a = adaValueAt(sst_a_ss, a_cross_ss_ts, 3);
    const gate_v2a = adaValueAt(sst_a_gate, a_cross_gate_ts, 0);
    const ax_scaled_v2a = ax_norm3
        .mul(scale_a_v2a.addConstant(1.0).broad(ax_norm3.shape()))
        .add(shift_a_v2a.broad(ax_norm3.shape()));

    const scale_v_v2a = adaValueAt(sst_v_ss, v_cross_ss_ts, 2);
    const shift_v_v2a = adaValueAt(sst_v_ss, v_cross_ss_ts, 3);
    const vx_scaled_v2a = vx_norm3
        .mul(scale_v_v2a.addConstant(1.0).broad(vx_norm3.shape()))
        .add(shift_v_v2a.broad(vx_norm3.shape()));

    const v2a_out = block.video_to_audio_attn.forward(ax_scaled_v2a, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
        .context = vx_scaled_v2a,
        .pe_cos = v2a_pe_cos,
        .pe_sin = v2a_pe_sin,
        .k_pe_cos = v2a_k_pe_cos,
        .k_pe_sin = v2a_k_pe_sin,
    });
    h_a = h_a.add(v2a_out.mul(gate_v2a.broad(v2a_out.shape())).mul(v2a_mask.broad(v2a_out.shape())));

    const vshift_mlp = adaValueAt(params.scale_shift_table, video_timesteps, 3);
    const vscale_mlp = adaValueAt(params.scale_shift_table, video_timesteps, 4);
    const vx_scaled_ff = zml.nn.rmsNorm(h_v, .d, 1e-6)
        .mul(vscale_mlp.addConstant(1.0).broad(h_v.shape()))
        .add(vshift_mlp.broad(h_v.shape()));

    return .{
        .norm_vx = norm_vx,
        .v_text_x = v_text_x,
        .vx_norm3 = vx_norm3,
        .a2v_x = vx_scaled_a2v,
        .a2v_ctx = ax_scaled_a2v,
        .a2v_gate = gate_a2v,
        .vx_scaled_ff = vx_scaled_ff,
    };
}

/// Debug helper: returns native-computed audio intermediates used by the block path.
pub fn forwardBlock0NativeAudioIntermediatesWithAVMasks(
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
) Block0NativeAudioIntermediates {
    const block = BasicAVTransformerBlock.init();

    const vx = vx_in.withPartialTags(.{ .b, .t, .d });
    const ax = ax_in.withPartialTags(.{ .b, .t, .d });

    const vshift_msa = adaValueAt(params.scale_shift_table, video_timesteps, 0);
    const vscale_msa = adaValueAt(params.scale_shift_table, video_timesteps, 1);
    const vgate_msa = adaValueAt(params.scale_shift_table, video_timesteps, 2);
    const norm_vx = zml.nn.rmsNorm(vx, .d, 1e-6)
        .mul(vscale_msa.addConstant(1.0).broad(vx.shape()))
        .add(vshift_msa.broad(vx.shape()));
    const attn1_out = block.attn1.forward(norm_vx, params.attn1, kindNumHeads(.attn1), .{ .pe_cos = v_pe_cos, .pe_sin = v_pe_sin });
    var h_v = vx.add(attn1_out.mul(vgate_msa.broad(attn1_out.shape())));

    const v_shift_q = adaValueAt(params.scale_shift_table, video_timesteps, 6);
    const v_scale_q = adaValueAt(params.scale_shift_table, video_timesteps, 7);
    const v_gate_q = adaValueAt(params.scale_shift_table, video_timesteps, 8);
    const v_text_x = zml.nn.rmsNorm(h_v, .d, 1e-6)
        .mul(v_scale_q.addConstant(1.0).broad(h_v.shape()))
        .add(v_shift_q.broad(h_v.shape()));
    const v_shift_kv = adaValueAt(params.prompt_scale_shift_table, v_prompt_timestep, 0);
    const v_scale_kv = adaValueAt(params.prompt_scale_shift_table, v_prompt_timestep, 1);
    const v_text_ctx_mod = v_text_ctx
        .mul(v_scale_kv.addConstant(1.0).broad(v_text_ctx.shape()))
        .add(v_shift_kv.broad(v_text_ctx.shape()));
    const v_text_ca_out = block.attn2.forward(v_text_x, params.attn2, kindNumHeads(.attn2), .{ .context = v_text_ctx_mod });
    h_v = h_v.add(v_text_ca_out.mul(v_gate_q.broad(v_text_ca_out.shape())));

    const ashift_msa = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 0);
    const ascale_msa = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 1);
    const agate_msa = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 2);
    const norm_ax = zml.nn.rmsNorm(ax, .d, 1e-6)
        .mul(ascale_msa.addConstant(1.0).broad(ax.shape()))
        .add(ashift_msa.broad(ax.shape()));
    const audio_attn1_out = block.audio_attn1.forward(norm_ax, params.audio_attn1, kindNumHeads(.audio_attn1), .{ .pe_cos = a_pe_cos, .pe_sin = a_pe_sin });
    var h_a = ax.add(audio_attn1_out.mul(agate_msa.broad(audio_attn1_out.shape())));

    const a_shift_q = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 6);
    const a_scale_q = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 7);
    const a_gate_q = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 8);
    const a_text_x = zml.nn.rmsNorm(h_a, .d, 1e-6)
        .mul(a_scale_q.addConstant(1.0).broad(h_a.shape()))
        .add(a_shift_q.broad(h_a.shape()));
    const a_shift_kv = adaValueAt(params.audio_prompt_scale_shift_table, a_prompt_timestep, 0);
    const a_scale_kv = adaValueAt(params.audio_prompt_scale_shift_table, a_prompt_timestep, 1);
    const a_text_ctx_mod = a_text_ctx
        .mul(a_scale_kv.addConstant(1.0).broad(a_text_ctx.shape()))
        .add(a_shift_kv.broad(a_text_ctx.shape()));
    const a_text_ca_out = block.audio_attn2.forward(a_text_x, params.audio_attn2, kindNumHeads(.audio_attn2), .{ .context = a_text_ctx_mod });
    h_a = h_a.add(a_text_ca_out.mul(a_gate_q.broad(a_text_ca_out.shape())));

    const vx_norm3 = zml.nn.rmsNorm(h_v, .d, 1e-6);
    const ax_norm3 = zml.nn.rmsNorm(h_a, .d, 1e-6);

    const sst_v_ss = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 0, .end = 4 });
    const sst_v_gate = params.scale_shift_table_a2v_ca_video.slice1d(.n_ada, .{ .start = 4, .end = 5 });
    const scale_v_a2v = adaValueAt(sst_v_ss, v_cross_ss_ts, 0);
    const shift_v_a2v = adaValueAt(sst_v_ss, v_cross_ss_ts, 1);
    const gate_a2v = adaValueAt(sst_v_gate, v_cross_gate_ts, 0);
    const vx_scaled_a2v = vx_norm3
        .mul(scale_v_a2v.addConstant(1.0).broad(vx_norm3.shape()))
        .add(shift_v_a2v.broad(vx_norm3.shape()));

    const sst_a_ss = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 0, .end = 4 });
    const sst_a_gate = params.scale_shift_table_a2v_ca_audio.slice1d(.n_ada, .{ .start = 4, .end = 5 });
    const scale_a_a2v = adaValueAt(sst_a_ss, a_cross_ss_ts, 0);
    const shift_a_a2v = adaValueAt(sst_a_ss, a_cross_ss_ts, 1);
    const ax_scaled_a2v = ax_norm3
        .mul(scale_a_a2v.addConstant(1.0).broad(ax_norm3.shape()))
        .add(shift_a_a2v.broad(ax_norm3.shape()));

    const a2v_out = block.audio_to_video_attn.forward(vx_scaled_a2v, params.audio_to_video_attn, kindNumHeads(.audio_to_video_attn), .{
        .context = ax_scaled_a2v,
        .pe_cos = a2v_pe_cos,
        .pe_sin = a2v_pe_sin,
        .k_pe_cos = a2v_k_pe_cos,
        .k_pe_sin = a2v_k_pe_sin,
    });
    h_v = h_v.add(a2v_out.mul(gate_a2v.broad(a2v_out.shape())).mul(a2v_mask.broad(a2v_out.shape())));

    const scale_a_v2a = adaValueAt(sst_a_ss, a_cross_ss_ts, 2);
    const shift_a_v2a = adaValueAt(sst_a_ss, a_cross_ss_ts, 3);
    const gate_v2a = adaValueAt(sst_a_gate, a_cross_gate_ts, 0);
    const ax_scaled_v2a = ax_norm3
        .mul(scale_a_v2a.addConstant(1.0).broad(ax_norm3.shape()))
        .add(shift_a_v2a.broad(ax_norm3.shape()));

    const scale_v_v2a = adaValueAt(sst_v_ss, v_cross_ss_ts, 2);
    const shift_v_v2a = adaValueAt(sst_v_ss, v_cross_ss_ts, 3);
    const vx_scaled_v2a = vx_norm3
        .mul(scale_v_v2a.addConstant(1.0).broad(vx_norm3.shape()))
        .add(shift_v_v2a.broad(vx_norm3.shape()));

    const v2a_out = block.video_to_audio_attn.forward(ax_scaled_v2a, params.video_to_audio_attn, kindNumHeads(.video_to_audio_attn), .{
        .context = vx_scaled_v2a,
        .pe_cos = v2a_pe_cos,
        .pe_sin = v2a_pe_sin,
        .k_pe_cos = v2a_k_pe_cos,
        .k_pe_sin = v2a_k_pe_sin,
    });
    h_a = h_a.add(v2a_out.mul(gate_v2a.broad(v2a_out.shape())).mul(v2a_mask.broad(v2a_out.shape())));

    const ashift_mlp = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 3);
    const ascale_mlp = adaValueAt(params.audio_scale_shift_table, audio_timesteps, 4);
    const ax_scaled_ff = zml.nn.rmsNorm(h_a, .d, 1e-6)
        .mul(ascale_mlp.addConstant(1.0).broad(h_a.shape()))
        .add(ashift_mlp.broad(h_a.shape()));

    return .{
        .norm_ax = norm_ax,
        .a_text_x = a_text_x,
        .v2a_x = ax_scaled_v2a,
        .ax_scaled_ff = ax_scaled_ff,
    };
}

pub fn forwardBlock0NativeVideo(
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
) Tensor {
    const out = forwardBlock0Native(
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        params,
    );
    return out.vx_out;
}

/// Native block-0 audio stream forward using inline AdaLN computation.
pub fn forwardBlock0NativeAudio(
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
) Tensor {
    const out = forwardBlock0Native(
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        params,
    );
    return out.ax_out;
}

/// Fixed-size 8-block params used by T3 contiguous-slice parity checks.
pub const BlockSlice8FullParams = struct {
    blocks: [8]Block0FullParams,
};

/// Build local 8-block params from a checkpoint that is already re-indexed to
/// transformer_blocks.0..7.
pub fn initBlockSlice8FullParams(store: zml.io.TensorStore.View) BlockSlice8FullParams {
    var out: BlockSlice8FullParams = undefined;
    const root = selectTransformerRoot(store);
    const blocks_store = root.withPrefix("transformer_blocks");
    inline for (0..8) |i| {
        out.blocks[i] = BasicAVTransformerBlock.initParams(blocks_store.withLayer(i));
    }
    return out;
}

pub fn unloadBlockSlice8FullBuffers(params: *zml.Bufferized(BlockSlice8FullParams)) void {
    inline for (0..8) |i| {
        unloadBlock0FullBuffers(&params.blocks[i]);
    }
}

pub const BlockSlice48FullParams = struct {
    blocks: [48]Block0FullParams,
};

pub fn initBlockSlice48FullParams(store: zml.io.TensorStore.View) BlockSlice48FullParams {
    var out: BlockSlice48FullParams = undefined;
    const root = selectTransformerRoot(store);
    const blocks_store = root.withPrefix("transformer_blocks");
    inline for (0..48) |i| {
        out.blocks[i] = BasicAVTransformerBlock.initParams(blocks_store.withLayer(i));
    }
    return out;
}

pub fn unloadBlockSlice48FullBuffers(params: *zml.Bufferized(BlockSlice48FullParams)) void {
    inline for (0..48) |i| {
        unloadBlock0FullBuffers(&params.blocks[i]);
    }
}

/// Native 8-block full stream forward using inline AdaLN computation.
fn forwardBlockSlice8NativeImpl(
    comptime audio_ff_residual_f32: bool,
    comptime audio_all_residuals_f32: bool,
    comptime video_all_residuals_f32: bool,
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
    v_text_ctx_mask: ?Tensor,
    a_text_ctx_mask: ?Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_masks: ?Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_masks: ?Tensor,
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    const block = BasicAVTransformerBlock.init();
    var h_v = vx_in;
    var h_a = ax_in;
    // Keep this loop runtime (not inline-unrolled) to avoid excessive stack/code growth
    // when tracing the 8-block graph in the native parity checker.
    for (params.blocks, 0..) |block_params, i| {
        const block_idx: i64 = @intCast(i);
        const a2v_mask = if (a2v_masks) |masks|
            masks.withPartialTags(.{ .block, .b, .t, .d }).slice1d(.block, .{ .start = block_idx, .end = block_idx + 1 }).squeeze(.block)
        else
            null;
        const v2a_mask = if (v2a_masks) |masks|
            masks.withPartialTags(.{ .block, .b, .t, .d }).slice1d(.block, .{ .start = block_idx, .end = block_idx + 1 }).squeeze(.block)
        else
            null;
        const shared: BasicAVTransformerBlock.SharedInputs = .{
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
            .v_text_ctx_mask = v_text_ctx_mask,
            .a_text_ctx_mask = a_text_ctx_mask,
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
        };
        const out = if (audio_all_residuals_f32)
            block.forwardNativeAudioAllResidualsF32(h_v, h_a, shared, block_params)
        else if (audio_ff_residual_f32)
            block.forwardNativeAudioFFResidualF32(h_v, h_a, shared, block_params)
        else if (video_all_residuals_f32)
            block.forwardNativeVideoAllResidualsF32(h_v, h_a, shared, block_params)
        else
            block.forwardNative(h_v, h_a, shared, block_params);
        h_v = out.vx_out;
        h_a = out.ax_out;
    }
    return .{ .vx_out = h_v, .ax_out = h_a };
}

pub fn forwardBlockSlice8Native(
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
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    return forwardBlockSlice8NativeImpl(
        false,
        false,
        false,
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        null,
        null,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        null,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        null,
        params,
    );
}

pub fn forwardBlockSlice8NativeWithTextMasks(
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
    v_text_ctx_mask: Tensor,
    a_text_ctx_mask: Tensor,
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
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    return forwardBlockSlice8NativeImpl(
        false,
        false,
        false,
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_text_ctx_mask,
        a_text_ctx_mask,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        null,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        null,
        params,
    );
}

pub fn forwardBlockSlice8NativeWithAVMasks(
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
    a2v_masks: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_masks: Tensor,
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    return forwardBlockSlice8NativeImpl(
        false,
        false,
        false,
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        null,
        null,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        a2v_masks,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        v2a_masks,
        params,
    );
}

pub fn forwardBlockSlice8NativeWithAllMasks(
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
    v_text_ctx_mask: Tensor,
    a_text_ctx_mask: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_masks: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_masks: Tensor,
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    return forwardBlockSlice8NativeImpl(
        false,
        false,
        false,
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_text_ctx_mask,
        a_text_ctx_mask,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        a2v_masks,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        v2a_masks,
        params,
    );
}

/// Checker-only experimental entrypoint: same as forwardBlockSlice8NativeWithAllMasks,
/// but computes only the final audio FF residual add in f32.
pub fn forwardBlockSlice8NativeWithAllMasksAudioFFResidualF32(
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
    v_text_ctx_mask: Tensor,
    a_text_ctx_mask: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_masks: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_masks: Tensor,
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    return forwardBlockSlice8NativeImpl(
        true,
        false,
        false,
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_text_ctx_mask,
        a_text_ctx_mask,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        a2v_masks,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        v2a_masks,
        params,
    );
}

/// Checker-only experimental entrypoint: same as forwardBlockSlice8NativeWithAllMasks,
/// but computes all audio residual adds in f32.
pub fn forwardBlockSlice8NativeWithAllMasksAudioAllResidualsF32(
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
    v_text_ctx_mask: Tensor,
    a_text_ctx_mask: Tensor,
    v_cross_ss_ts: Tensor,
    v_cross_gate_ts: Tensor,
    a_cross_ss_ts: Tensor,
    a_cross_gate_ts: Tensor,
    a2v_pe_cos: Tensor,
    a2v_pe_sin: Tensor,
    a2v_k_pe_cos: Tensor,
    a2v_k_pe_sin: Tensor,
    a2v_masks: Tensor,
    v2a_pe_cos: Tensor,
    v2a_pe_sin: Tensor,
    v2a_k_pe_cos: Tensor,
    v2a_k_pe_sin: Tensor,
    v2a_masks: Tensor,
    params: BlockSlice8FullParams,
) BasicAVTransformerBlock.FullOutputs {
    return forwardBlockSlice8NativeImpl(
        false,
        true,
        false,
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_text_ctx_mask,
        a_text_ctx_mask,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        a2v_masks,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        v2a_masks,
        params,
    );
}

/// Native 8-block video stream forward using inline AdaLN computation.
pub fn forwardBlockSlice8NativeVideo(
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
    v_text_ctx_mask: Tensor,
    a_text_ctx_mask: Tensor,
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
    params: BlockSlice8FullParams,
) Tensor {
    return forwardBlockSlice8NativeWithTextMasks(
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_text_ctx_mask,
        a_text_ctx_mask,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        params,
    ).vx_out;
}

/// Native 8-block audio stream forward using inline AdaLN computation.
pub fn forwardBlockSlice8NativeAudio(
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
    v_text_ctx_mask: Tensor,
    a_text_ctx_mask: Tensor,
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
    params: BlockSlice8FullParams,
) Tensor {
    return forwardBlockSlice8NativeWithTextMasks(
        vx_in,
        ax_in,
        video_timesteps,
        audio_timesteps,
        v_prompt_timestep,
        a_prompt_timestep,
        v_pe_cos,
        v_pe_sin,
        a_pe_cos,
        a_pe_sin,
        v_text_ctx,
        a_text_ctx,
        v_text_ctx_mask,
        a_text_ctx_mask,
        v_cross_ss_ts,
        v_cross_gate_ts,
        a_cross_ss_ts,
        a_cross_gate_ts,
        a2v_pe_cos,
        a2v_pe_sin,
        a2v_k_pe_cos,
        a2v_k_pe_sin,
        v2a_pe_cos,
        v2a_pe_sin,
        v2a_k_pe_cos,
        v2a_k_pe_sin,
        params,
    ).ax_out;
}

// ---------------------------------------------------------------------------
// Full transformer step: 48 blocks + OutputProjection (Step 1 parity)
// ---------------------------------------------------------------------------

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


