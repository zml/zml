//! MiniMax-H3 DiT.
//!
//!   1. refine text tokens
//!   2. MM-RoPE from packed (t,h,w)
//!   3. time embed every σ → AdaLN tables (4 unique time slots)
//!   4. pack text + noisy audio + noisy video into one sequence
//!   5. 50 AdaLN blocks (Pi-H3: distinct early LoRA and late replacement kernels)
//!   6. AdaLN and modality heads, then Euler on both FP32 states
//!   7. Pi-H3: hand off DWD-normalized video rows from the last block/evaluation
//!
//! Block:
//!   msa, mlp = AdaLN(temb)
//!   h ← h + gate_msa * Attn( RMS(h) * (1+scale_msa) + shift_msa )
//!   h ← h + gate_mlp * FF  ( RMS(h) * (1+scale_mlp) + shift_mlp )

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");
const pack = @import("pack.zig");
const pi = @import("dit_pi.zig");

const log = std.log.scoped(.minimax_h3);

const DitConfig = config.DitConfig;
const linear = ops.linear;
const rms = ops.rms;
const load = ops.load;
const ropeCat3 = ops.ropeCat3;
const Run = ops.Run;
const Packed = pack.Packed;

/// Final denoiser outputs. `video` holds
///     - packed latents (H3)
///     - or the final evaluation DWD features (Pi-H3).
pub const DenoiseOutput = struct {
    video: zml.Buffer,
    audio: zml.Buffer,

    pub fn deinit(self: *DenoiseOutput) void {
        self.video.deinit();
        self.audio.deinit();
    }
};

const Rotary = struct { cos: zml.Tensor, sin: zml.Tensor };

fn shiftScale(x: zml.Tensor, shift: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    const dt = x.dtype();
    return x.mul(zml.Tensor.scalar(1.0, dt).add(scale.squeeze(.k).convert(dt).broad(x.shape())))
        .add(shift.squeeze(.k).convert(dt).broad(x.shape()));
}

fn residualGate(x: zml.Tensor, gate: zml.Tensor, y: zml.Tensor) zml.Tensor {
    return x.add(gate.squeeze(.k).convert(y.dtype()).broad(y.shape()).mul(y));
}

/// Dense `{b,s,d}` pack: text | audio | video. Update-slice matches scatter's FA2 layout.
fn packedHidden(text: zml.Tensor, audio: zml.Tensor, video: zml.Tensor) zml.Tensor {
    const text_len = text.dim(.s);
    const audio_len = audio.dim(.s);
    var hidden = zml.Tensor.zeroes(zml.Shape.init(.{
        .b = text.dim(.b),
        .s = text_len + audio_len + video.dim(.s),
        .d = text.dim(.d),
    }, text.dtype()));
    hidden = hidden.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(0, .i32) }, text);
    hidden = hidden.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(@as(i32, @intCast(text_len)), .i32) }, audio);
    hidden = hidden.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(@as(i32, @intCast(text_len + audio_len)), .i32) }, video);
    return hidden.replicate();
}

// =============================================================================
// Feed-forward (SwiGLU)
// =============================================================================

const SwiGlu = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    fc1_lora: ?pi.LoRA,
    fc2_lora: ?pi.LoRA,
    pi_h3: bool, // Pi-H3 numerics (FP32 SwiGLU product)

    pub fn init(store: zml.io.TensorStore.View, pi_h3: bool, lora: bool) SwiGlu {
        return .{
            .fc1 = linear(store, "net.0.proj.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
            .fc2 = linear(store, "net.2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
            .fc1_lora = if (lora) pi.LoRA.init(store.withPrefix("net.0.proj"), true) else null,
            .fc2_lora = if (lora) pi.LoRA.init(store.withPrefix("net.2"), false) else null,
            .pi_h3 = pi_h3,
        };
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        var projected = self.fc1.forward(x, x.dtype());
        if (self.fc1_lora) |lora| projected = lora.apply(projected, x);
        const value, const gate = projected.chunkExact(-1, 2);
        // LakonLab's CUDA SwiGLU computes silu(gate) * up in FP32 ([up, gate] layout)
        // and stores it as BF16:
        // https://github.com/Lakonik/LakonLab/blob/2ee2b334374ec91b7cee371a85a524bf4ba644d3/lakonlab/ops/activation.py#L14-L26
        const product = if (self.pi_h3)
            gate.convert(.f32).silu().mul(value.convert(.f32)).convert(x.dtype())
        else
            gate.silu().mul(value);
        const hidden = product.rename(.{ .dout = .d });
        const y = self.fc2.forward(hidden, hidden.dtype());
        return if (self.fc2_lora) |lora| lora.apply(y, hidden) else y;
    }
};

// =============================================================================
// Attention (bidirectional)
// =============================================================================

const Attention = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    out: zml.nn.Linear,
    out_lora: ?pi.LoRA,
    q_norm: ops.RmsNorm,
    k_norm: ops.RmsNorm,
    num_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitConfig, lora: bool) Attention {
        const qkv = .{ .dout = .model, .d = .replicated };
        return .{
            .q = linear(store, "to_q.weight", null, qkv, .replicated),
            .k = linear(store, "to_k.weight", null, qkv, .replicated),
            .v = linear(store, "to_v.weight", null, qkv, .replicated),
            .out = linear(store, "to_out.0.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
            .out_lora = if (lora) pi.LoRA.init(store.withPrefix("to_out.0"), false) else null,
            .q_norm = rms(store.withPrefix("norm_q"), .{.hd}, cfg.qk_norm_eps),
            .k_norm = rms(store.withPrefix("norm_k"), .{.hd}, cfg.qk_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .head_dim = cfg.attention_head_dim,
        };
    }

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?Rotary, backend: zml.attention.Backend) zml.Tensor {
        const heads = .{ .h = self.num_heads, .hd = self.head_dim };
        const x_qkv = x.withPartitioning(.model, .{ .d = .replicated });
        var q: zml.Tensor = self.q_norm.forward(self.q.forward(x_qkv, x_qkv.dtype()).splitAxis(.dout, heads).withPartitioning(.model, .{ .h = .model }));
        var k: zml.Tensor = self.k_norm.forward(self.k.forward(x_qkv, x_qkv.dtype()).splitAxis(.dout, heads).withPartitioning(.model, .{ .h = .model }));
        const v = self.v.forward(x_qkv, x_qkv.dtype()).splitAxis(.dout, heads).withPartitioning(.model, .{ .h = .model });
        if (rotary) |pe| {
            q = ops.applyRotary(q, pe.cos, pe.sin);
            k = ops.applyRotary(k, pe.cos, pe.sin);
        }

        const metadata: zml.attention.Metadata = .init(.fromBackend(backend));
        const parameters: zml.attention.Parameters = .init(.fromBackend(backend));
        // Diffusion Transformer and text refinement use bidirectional full sequence attention. heads sharded on .model dimension.
        const attn = zml.attention.attentionOpts(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            zml.Tensor.scalar(0, .u32),
            metadata,
            parameters,
            .{ .causal = false },
            zml.Compiler.current().mesh(.model),
            .head(.model),
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        const base = self.out.forward(attn, attn.dtype());
        const y: zml.Tensor = if (self.out_lora) |lora| lora.apply(base, attn) else base;
        return y.rename(.{ .dout = .d }).withPartitioning(.model, .{ .d = .replicated });
    }
};

// =============================================================================
// Time embedder  (t → temb)
// =============================================================================

/// Timestep conditioning:
/// H3:     t → sinusoidal embedding/MLP → 2688-wide temb → AdaLN projection
/// Pi-H3:  t → interpolate [1025, 8] table → 8 coordinates → AdaLN projection
const TimeEmbedder = union(enum) {
    sinusoidal: struct { proj_in: zml.nn.Linear, proj_out: zml.nn.Linear, freq_dim: i64, rope_theta: f32 },
    table: zml.Tensor, // for pi-h3, time is embedded in a lookup table

    pub const Input = struct { model: TimeEmbedder, timestep: zml.Tensor };
    pub const Output = struct { temb: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: DitConfig, pi_h3: bool) TimeEmbedder {
        if (pi_h3) return .{ .table = store.createTensor("table", .{ .row, .d }, .replicated, .{}) };
        return .{ .sinusoidal = .{
            .proj_in = linear(store, "linear_1.weight", "linear_1.bias", .replicated, .{}),
            .proj_out = linear(store, "linear_2.weight", "linear_2.bias", .replicated, .{}),
            .freq_dim = cfg.freq_dim,
            .rope_theta = cfg.rope_theta,
        } };
    }

    pub fn forward(input: Input) Output {
        var temb: zml.Tensor = undefined;
        switch (input.model) {
            .sinusoidal => |model| {
                const inv = zml.nn.invFreq(model.freq_dim, .{
                    .layout = .real_im_pass,
                    .scaling = .{ .default = .{ .rope_theta = model.rope_theta } },
                }).withTags(.{.f});
                const angles = input.timestep.convert(.f32).withPartialTags(.{.n}).outer(inv);
                const features = zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d });
                const hidden = model.proj_in.forward(features, features.dtype()).silu().rename(.{ .dout = .d });
                temb = model.proj_out.forward(hidden, hidden.dtype()).rename(.{ .dout = .d });
            },
            .table => |table| {
                const p = input.timestep.convert(.f32).clamp(.scalar(0, .f32), .scalar(1, .f32)).scale(1024.0);
                const lower = p.floor().minimum(.scalar(1023, .f32));
                const lo = lower.convert(.i32);
                // Interpolate the stored FP32 rows without first rounding them to BF16.
                const a = table.gather(.{ .row = lo }, .{});
                const b = table.gather(.{ .row = lo.addConstant(1) }, .{});
                const w = p.sub(lower).broad(a.shape());
                const delta = b.sub(a);
                temb = w.cmp(.LT, .scalar(0.5, .f32)).select(
                    a.add(w.mul(delta)),
                    b.sub(zml.Tensor.scalar(1, .f32).sub(w).mul(delta)),
                );
            },
        }
        return .{ .temb = temb };
    }
};

// =============================================================================
// AdaLN
// =============================================================================

fn AdaLn(comptime expand: i64, comptime modalities: i64) type {
    return struct {
        const Self = @This();
        linear: zml.nn.Linear,
        hidden_size: i64,
        pi_h3: bool,
        pub const PrepareInput = struct { adaln: Self, temb: zml.Tensor, steps: i64, slots: i64 };
        pub const PrepareOutput = struct { table: zml.Tensor };

        pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, pi_h3: bool) Self {
            return .{
                .linear = linear(store, "linear.weight", "linear.bias", .replicated, .{}),
                .hidden_size = hidden_size,
                .pi_h3 = pi_h3,
            };
        }

        pub fn prepare(input: PrepareInput) PrepareOutput {
            const adaln = input.adaln;
            const dt = adaln.linear.weight.dtype();
            const projected = if (adaln.pi_h3)
                ops.bf16Linear(adaln.linear, input.temb)
            else
                adaln.linear.forward(input.temb.silu().convert(dt), dt);
            const raw = projected.splitAxis(.dout, .{ .mod = modalities, .k = expand, .d = adaln.hidden_size });
            const table = if (comptime modalities == 1) raw.squeeze(.mod) else raw;
            return .{ .table = table.splitAxis(.n, .{ .t = input.steps, .n = input.slots }) };
        }
    };
}

const BlockAdaLn = AdaLn(6, config.modality_count);
const FinalAdaLn = AdaLn(2, 1);

// =============================================================================
// Transformer block
// =============================================================================

const Block = struct {
    norm1: ops.RmsNorm,
    attn: Attention,
    norm2: ops.RmsNorm,
    mlp: SwiGlu,
    pub const Input = struct {
        layer: Block,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
        attn_backend: zml.attention.Backend,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: DitConfig, pi_h3: bool, lora: bool) Block {
        return .{
            .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(store.withPrefix("attn"), cfg, lora),
            .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(store.withPrefix("ff"), pi_h3, lora),
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const c = input.table.gather(.{ .t = input.step }, .{}).merge(.{ .n = .{ .n, .mod } })
            .gather(.{ .n = input.adaln_indices }, .{}).chunkExact(.k, 6);

        const residual = input.hidden.withPartitioning(.model, .{ .d = .replicated });
        const attn_out = self.attn.forward(
            shiftScale(self.norm1.forward(residual), c[0], c[1]),
            .{ .cos = input.cos, .sin = input.sin },
            input.attn_backend,
        );
        const x1 = residualGate(residual, c[2], attn_out).withPartitioning(.model, .{ .d = .replicated });
        const mlp_out = self.mlp.forward(shiftScale(self.norm2.forward(x1), c[3], c[4])).rename(.{ .dout = .d });
        return .{
            .hidden = residualGate(x1, c[5], mlp_out).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(input.hidden),
        };
    }

    fn refine(self: Block, x: zml.Tensor, backend: zml.attention.Backend) zml.Tensor {
        const residual = x.withPartitioning(.model, .{ .d = .replicated });
        const x1 = residual.add(self.attn.forward(self.norm1.forward(residual), null, backend));
        return x1.add(self.mlp.forward(self.norm2.forward(x1)).rename(.{ .dout = .d }))
            .withPartitioning(.model, .{ .d = .replicated })
            .reuseBuffer(x);
    }
};

// =============================================================================
// Compiled kernels (text refine, pack, finish, RoPE, Euler)
// =============================================================================

/// Linear `context_embedder` then two refiner blocks.
const TextPrep = struct {
    condition_proj: zml.nn.Linear,
    blocks: []Block,
    final_norm: ops.RmsNorm,
    pi_h3: bool,
    pub const Input = struct { model: TextPrep, text: zml.Tensor, attn_backend: zml.attention.Backend };
    pub const Output = struct { text: zml.Tensor };

    /// Nested `blocks` slice is not freed by `Buffer.deinitAll`.
    pub fn unload(self: *zml.Bufferized(TextPrep), allocator: std.mem.Allocator) void {
        zml.nn.Linear.unloadBuffers(&self.condition_proj);
        for (self.blocks) |*block| zml.Buffer.deinitAll(Block, block);
        allocator.free(self.blocks);
        self.final_norm.weight.deinit();
    }

    pub fn forward(input: Input) Output {
        const text_in = input.text.convert(input.model.condition_proj.weight.dtype());
        const projected = if (input.model.pi_h3)
            ops.bf16Linear(input.model.condition_proj, input.text)
        else
            input.model.condition_proj.forward(text_in, text_in.dtype());
        var text = projected.rename(.{ .dout = .d });
        text = text.convert(input.model.final_norm.weight.dtype());
        for (input.model.blocks) |block| text = block.refine(text, input.attn_backend);
        return .{ .text = input.model.final_norm.forward(text) };
    }
};

/// Pack refined text, projected audio, and video patches into one sequence.
/// Writes into a dense `{b,s,d}` buffer (same layout scatter used) so FA2 sees a contiguous packed seq.
const PatchEmbed = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    pi_h3: bool,
    pub const Input = struct {
        model: PatchEmbed,
        video: zml.Tensor,
        audio: zml.Tensor,
        text: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        if (input.model.pi_h3) return .{ .hidden = packedHidden(
            input.text,
            ops.bf16Linear(input.model.audio_proj, input.audio).rename(.{ .dout = .d }),
            ops.bf16Linear(input.model.video_proj, input.video).rename(.{ .dout = .d }),
        ) };
        const dt = input.text.dtype();
        const video_in = input.video.convert(input.model.video_proj.weight.dtype());
        const audio_in = input.audio.convert(input.model.audio_proj.weight.dtype());
        const video = input.model.video_proj.forward(video_in, video_in.dtype()).rename(.{ .dout = .d }).convert(dt);
        const audio = input.model.audio_proj.forward(audio_in, audio_in.dtype()).rename(.{ .dout = .d }).convert(dt);
        return .{ .hidden = packedHidden(input.text, audio, video) };
    }
};

/// Final RMS + time-slot AdaLN, then the video/audio heads on those rows only.
const Finish = struct {
    norm: ops.RmsNorm,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,
    text_len: i64,
    audio_len: i64,
    pi_h3: bool,
    pub const Input = struct {
        model: Finish,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        timestep_indices: zml.Tensor,
    };
    pub const Output = struct { video: zml.Tensor, audio: zml.Tensor };

    pub fn forward(input: Input) Output {
        const mods = input.table.gather(.{ .t = input.step }, .{});
        const n = input.model.norm.forward(input.hidden.withPartitioning(.model, .{ .d = .replicated }));
        const shift, const scale = mods.gather(.{ .n = input.timestep_indices }, .{}).chunkExact(.k, 2);
        // Diffusers keeps the heads in FP32 (`_keep_in_fp32_modules`): https://github.com/huggingface/diffusers/blob/16eb5cc4fd8262fd33b3c9119a12977c25cc4985/src/diffusers/models/transformers/transformer_minimax_h3.py#L449
        // LakonLab clears that list and casts the whole Pi-H3 model to BF16: https://github.com/Lakonik/LakonLab/blob/2ee2b334374ec91b7cee371a85a524bf4ba644d3/lakonlab/models/architectures/diffusers/minimax_h3.py#L364
        const head_dt = if (input.model.pi_h3) .bf16 else input.model.video_out.weight.dtype();
        const audio_end = input.model.text_len + input.model.audio_len;
        const audio_s: zml.Tensor.Slice = .{ .start = input.model.text_len, .end = audio_end };
        const video_s: zml.Tensor.Slice = .{ .start = audio_end };
        const video_in = shiftScale(n.slice(.s, video_s), shift.slice(.s, video_s), scale.slice(.s, video_s)).convert(head_dt);
        const audio_in = shiftScale(n.slice(.s, audio_s), shift.slice(.s, audio_s), scale.slice(.s, audio_s)).convert(head_dt);
        if (input.model.pi_h3) return .{
            .video = ops.bf16Linear(input.model.video_out, video_in).rename(.{ .dout = .d }),
            .audio = ops.bf16Linear(input.model.audio_out, audio_in).rename(.{ .dout = .d }),
        };
        return .{
            .video = input.model.video_out.forward(video_in, video_in.dtype()).rename(.{ .dout = .d }),
            .audio = input.model.audio_out.forward(audio_in, audio_in.dtype()).rename(.{ .dout = .d }),
        };
    }
};

/// Cos/sin from packed (t,h,w) via `ops.ropeCat3`.
const Rope = struct {
    pub const Input = struct { position_ids: zml.Tensor, rope_freq_dim: i64, rope_theta: f32, out_dtype: zml.DataType };
    pub const Output = struct { cos: zml.Tensor, sin: zml.Tensor };

    pub fn forward(input: Input) Output {
        const emb = ropeCat3(input.position_ids, zml.nn.invFreq(2 * input.rope_freq_dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = input.rope_theta } },
        }).withTags(.{.f}));
        return .{ .cos = emb.cos().convert(input.out_dtype), .sin = emb.sin().convert(input.out_dtype) };
    }
};

/// Euler step (η=0) for the data-ward velocity v: x + (σ − σ')·v in FP32, in the operation order of each path's reference scheduler.
const Euler = struct {
    pub const Input = struct { sample: zml.Tensor, velocity: zml.Tensor, sigma: zml.Tensor, sigma_next: zml.Tensor, pi_h3: bool };
    pub const Output = struct { sample: zml.Tensor };

    pub fn apply(input: Input) Output {
        const x = input.sample;
        const v = input.velocity.convert(.f32);
        // LakonLab FlowEulerODEScheduler: x + (−v)(σ' − σ).
        if (input.pi_h3) return .{ .sample = x.add(v.negate().mul(input.sigma_next.sub(input.sigma))).reuseBuffer(x) };

        // Diffusers MiniMaxH3Scheduler: r·x + (1 − r)·x0, with x0 = x + σv and r = σ'/σ.
        const x0 = x.add(v.mul(input.sigma.broad(x.shape())));
        const r = input.sigma_next.div(input.sigma).broad(x.shape());
        return .{ .sample = r.mul(x).add(zml.Tensor.scalar(1.0, .f32).sub(r).mul(x0)).reuseBuffer(x) };

        // TODO: Formulas for FlowEulerODEScheduler and MiniMaxH3Scheduler are the same.
        // TODO: But the operation order and numerical rounding may differ, so we keep them separate for now.
        // TODO: Test a single step of both schedulers to ensure numerical consistency.
    }
};

// =============================================================================
// DiT
// =============================================================================

pub const Dit = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    condition_proj: zml.nn.Linear,
    time_embedder: TimeEmbedder,
    refiner_blocks: []Block,
    refiner_norm: ops.RmsNorm,
    blocks: []Block,
    adalns: []BlockAdaLn,
    final_adaln: FinalAdaLn,
    final_norm: ops.RmsNorm,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,
    dwd_projection: ?zml.nn.Linear,
    cfg: DitConfig,
    compiled: ?Compiled = null,
    pi_h3: bool, // Pi-H3 path: pruned base modules and numerics, late replacement blocks, DWD feature handoff.

    const Compiled = struct {
        prepare_text: zml.FnExe(TextPrep.forward),
        prepare_rope: zml.FnExe(Rope.forward),
        embed_patches: zml.FnExe(PatchEmbed.forward),
        prepare_temb: zml.FnExe(TimeEmbedder.forward),
        prepare_adaln: zml.FnExe(BlockAdaLn.prepare),
        prepare_final_adaln: zml.FnExe(FinalAdaLn.prepare),
        block: zml.FnExe(Block.forward),
        late_block: ?zml.FnExe(Block.forward),
        late_adaln: ?zml.FnExe(BlockAdaLn.prepare),
        features: ?zml.FnExe(pi.Features.forward),
        finish: zml.FnExe(Finish.forward),
        apply_video: zml.FnExe(Euler.apply),
        apply_audio: zml.FnExe(Euler.apply),

        fn deinit(self: *Compiled) void {
            self.prepare_text.deinit();
            self.prepare_rope.deinit();
            self.embed_patches.deinit();
            self.prepare_temb.deinit();
            self.prepare_adaln.deinit();
            self.prepare_final_adaln.deinit();
            self.block.deinit();
            if (self.late_block) |*exe| exe.deinit();
            if (self.late_adaln) |*exe| exe.deinit();
            if (self.features) |*exe| exe.deinit();
            self.finish.deinit();
            self.apply_video.deinit();
            self.apply_audio.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: DitConfig) !Dit {
        // time_embedder.table exists only in the pruned Pi-H3 base. adapter.open() has already validated its shape.
        const pi_h3 = store.getShape("time_embedder.table") != null;

        const n_layers: usize = @intCast(cfg.num_layers);
        const blocks = try allocator.alloc(Block, n_layers);
        errdefer allocator.free(blocks);
        const adalns = try allocator.alloc(BlockAdaLn, n_layers);
        errdefer allocator.free(adalns);
        const refiner = store.withPrefix("token_refiner");
        const refiner_blocks = try allocator.alloc(Block, @intCast(cfg.num_refiner_layers));
        errdefer allocator.free(refiner_blocks);
        for (refiner_blocks, 0..) |*block, i| block.* = .init(refiner.withPrefix("refiner_blocks").withLayer(i), cfg, pi_h3, false);
        for (blocks, adalns, 0..) |*block, *adaln, i| {
            const layer = store.withPrefix("transformer_blocks").withLayer(i);
            const lora = pi_h3 and i < pi.first_replacement_block;
            block.* = .init(layer, cfg, pi_h3, lora);
            adaln.* = .init(layer.withPrefix("adaln_proj"), cfg.hidden_size, pi_h3);
        }
        return .{
            .video_proj = linear(store, "proj_in.weight", "proj_in.bias", .replicated, .{}),
            .audio_proj = linear(store, "audio_proj_in.weight", "audio_proj_in.bias", .replicated, .{}),
            .condition_proj = linear(store, "context_embedder.weight", "context_embedder.bias", .replicated, .{}),
            .time_embedder = .init(store.withPrefix("time_embedder"), cfg, pi_h3),
            .refiner_blocks = refiner_blocks,
            .refiner_norm = rms(refiner.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
            .blocks = blocks,
            .adalns = adalns,
            .final_adaln = .init(store.withPrefix("norm_out"), cfg.hidden_size, pi_h3),
            .final_norm = rms(store.withPrefix("norm_out.norm"), .{.d}, cfg.final_norm_eps),
            .video_out = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .{}),
            .audio_out = linear(store, "audio_proj_out.weight", "audio_proj_out.bias", .replicated, .{}),
            .dwd_projection = if (pi_h3) linear(store.withPrefix("dwd_norm_out"), "linear.weight", "linear.bias", .replicated, .{}) else null,
            .cfg = cfg,
            .pi_h3 = pi_h3,
        };
    }

    pub fn deinit(self: *Dit, allocator: std.mem.Allocator) void {
        if (self.compiled) |*c| c.deinit();
        allocator.free(self.refiner_blocks);
        allocator.free(self.blocks);
        allocator.free(self.adalns);
    }

    /// The Pi variants have distinct executable signatures: early blocks [0, 45]
    /// carry LoRA tensors and FP32 folded biases; late blocks [46, 49] take FP16
    /// replacement linears and biases. Never bind one to the other's runner.
    pub fn compile(self: *Dit, run: *const Run, geo: config.Geometry, text_len: u32, packed_run: Packed, text_dt: zml.DataType) !void {
        const attn: zml.attention.Backend = .auto(run.platform);
        if (self.pi_h3 and (run.platform.target != .cuda or attn != .flash_attention))
            return error.PiRequiresFusedCudaAttention;
        const attnName: []const u8 = switch (attn) {
            .stablehlo => "stablehlo",
            .flash_attention => |backend| @tagName(backend),
        };

        const layer = self.blocks[0];
        const seq_len: u32 = packed_run.layout.seqLen();
        const steps: u32 = @intCast(packed_run.video.sigmas.len - 1);
        const flat_n: i64 = @intCast(steps * config.timestep_slot_count);
        const temb_dim = switch (self.time_embedder) {
            .sinusoidal => |m| m.proj_out.weight.dim(.dout),
            .table => |table| table.dim(.d),
        };
        log.info("dit attn={s} seq={d} audio_tokens={d} devices={d}", .{
            attnName,
            seq_len,
            geo.audio_tokens,
            run.platform.devices.len,
        });
        var node = run.progress.start("Compiling MiniMax-H3 DiT", 0);
        defer node.end();
        const dt = layer.norm1.weight.dtype();
        const patch_part: PatchEmbed = .{
            .video_proj = self.video_proj,
            .audio_proj = self.audio_proj,
            .pi_h3 = self.pi_h3,
        };

        const prepare_text = try zml.FnExe(TextPrep.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_prepare_text",
        }, .{.{
            .model = .{
                .condition_proj = self.condition_proj,
                .blocks = self.refiner_blocks,
                .final_norm = self.refiner_norm,
                .pi_h3 = self.pi_h3,
            },
            .text = .init(.{ .b = 1, .s = text_len, .d = self.cfg.text_dim }, text_dt),
            .attn_backend = attn,
        }});
        errdefer prepare_text.deinit();
        const prepare_rope = try zml.FnExe(Rope.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_prepare_rope",
        }, .{.{
            .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
            .rope_freq_dim = self.cfg.rope_freq_dim,
            .rope_theta = self.cfg.rope_theta,
            .out_dtype = dt,
        }});
        errdefer prepare_rope.deinit();
        const embed_patches = try zml.FnExe(PatchEmbed.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_embed_patches",
        }, .{.{
            .model = patch_part,
            .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
            .audio = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
            .text = .init(.{ .b = 1, .s = text_len, .d = self.cfg.hidden_size }, dt),
        }});
        errdefer embed_patches.deinit();
        const prepare_temb = try zml.FnExe(TimeEmbedder.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_prepare_temb",
        }, .{.{
            .model = self.time_embedder,
            .timestep = .init(.{ .n = flat_n }, .f32),
        }});
        errdefer prepare_temb.deinit();
        var adaln_input: BlockAdaLn.PrepareInput = .{
            .adaln = self.adalns[0],
            .temb = .init(.{ .n = flat_n, .d = temb_dim }, .f32),
            .steps = steps,
            .slots = config.timestep_slot_count,
        };
        const prepare_adaln = try zml.FnExe(BlockAdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_prepare_adaln",
        }, .{adaln_input});
        errdefer prepare_adaln.deinit();
        const late_adaln = if (self.pi_h3) late: {
            adaln_input.adaln = self.adalns[pi.first_replacement_block];
            break :late try zml.FnExe(BlockAdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
                .meshes = &run.mesh,
                .program_name = "pi_h3_prepare_late_adaln",
            }, .{adaln_input});
        } else null;
        errdefer if (late_adaln) |exe| exe.deinit();
        const prepare_final_adaln = try zml.FnExe(FinalAdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_prepare_final_adaln",
        }, .{.{
            .adaln = self.final_adaln,
            .temb = .init(.{ .n = flat_n, .d = temb_dim }, .f32),
            .steps = steps,
            .slots = config.timestep_slot_count,
        }});
        errdefer prepare_final_adaln.deinit();

        var block_input: Block.Input = .{
            .layer = layer,
            .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = self.cfg.hidden_size }, dt),
            .table = zml.Tensor.init(.{ .t = steps, .n = config.timestep_slot_count, .mod = config.modality_count, .k = 6, .d = self.cfg.hidden_size }, dt),
            .step = zml.Tensor.init(.{}, .u32),
            .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
            .cos = zml.Tensor.init(.{ .s = seq_len, .f = self.cfg.rotaryDim() }, dt),
            .sin = zml.Tensor.init(.{ .s = seq_len, .f = self.cfg.rotaryDim() }, dt),
            .attn_backend = attn,
        };
        const block_exe = try zml.FnExe(Block.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_block",
        }, .{block_input});
        errdefer block_exe.deinit();
        const late_block = if (self.pi_h3) late: {
            block_input.layer = self.blocks[pi.first_replacement_block];
            break :late try zml.FnExe(Block.forward).compile(run.allocator, run.io, run.platform, .{
                .meshes = &run.mesh,
                .program_name = "pi_h3_late_block",
            }, .{block_input});
        } else null;
        errdefer if (late_block) |exe| exe.deinit();
        const features = if (self.dwd_projection) |projection| try zml.FnExe(pi.Features.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "pi_h3_final_features",
        }, .{.{
            .model = .{ .projection = projection, .first_video = packed_run.layout.videoStart(), .eps = self.cfg.final_norm_eps },
            .hidden = block_input.hidden,
            .temb = adaln_input.temb,
            .step = .init(.{}, .u32),
            .timestep_indices = .init(.{ .s = seq_len }, .u32),
        }}) else null;
        errdefer if (features) |exe| exe.deinit();
        const finish_exe = try zml.FnExe(Finish.forward).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_finish",
        }, .{.{
            .model = .{
                .norm = self.final_norm,
                .video_out = self.video_out,
                .audio_out = self.audio_out,
                .text_len = text_len,
                .audio_len = geo.audio_tokens,
                .pi_h3 = self.pi_h3,
            },
            .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = self.cfg.hidden_size }, dt),
            .table = zml.Tensor.init(.{ .t = steps, .n = config.timestep_slot_count, .k = 2, .d = self.cfg.hidden_size }, dt),
            .step = zml.Tensor.init(.{}, .u32),
            .timestep_indices = .init(.{ .s = seq_len }, .u32),
        }});
        errdefer finish_exe.deinit();
        // Two Euler kernels: same math, different `{s,d}` (video 96-d vs audio 32-d).
        const apply_video = try zml.FnExe(Euler.apply).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_apply_video",
        }, .{.{
            .sample = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
            .velocity = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, if (self.pi_h3) .bf16 else .f32),
            .sigma = .init(.{}, .f32),
            .sigma_next = .init(.{}, .f32),
            .pi_h3 = self.pi_h3,
        }});
        errdefer apply_video.deinit();
        const apply_audio = try zml.FnExe(Euler.apply).compile(run.allocator, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "minimax_h3_apply_audio",
        }, .{.{
            .sample = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
            .velocity = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, if (self.pi_h3) .bf16 else .f32),
            .sigma = .init(.{}, .f32),
            .sigma_next = .init(.{}, .f32),
            .pi_h3 = self.pi_h3,
        }});
        self.compiled = .{
            .prepare_text = prepare_text,
            .prepare_rope = prepare_rope,
            .embed_patches = embed_patches,
            .prepare_temb = prepare_temb,
            .prepare_adaln = prepare_adaln,
            .prepare_final_adaln = prepare_final_adaln,
            .block = block_exe,
            .late_block = late_block,
            .late_adaln = late_adaln,
            .features = features,
            .finish = finish_exe,
            .apply_video = apply_video,
            .apply_audio = apply_audio,
        };
    }

    /// Encoder hidden + packed layout → denoised video and audio tokens.
    pub fn denoise(
        self: *const Dit,
        run: *const Run,
        store: *zml.io.TensorStore,
        geo: config.Geometry,
        text: zml.Buffer,
        text_len: u32,
        packed_run: Packed,
        seed: u64,
    ) !DenoiseOutput {
        const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
        const allocator = run.allocator;
        const io = run.io;
        const drawn = try pack.noise(allocator, seed, geo);
        defer drawn.deinit(allocator);
        const layout = packed_run.layout;
        const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
        const audio_shape = zml.Shape.init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32);
        const seq_len: usize = layout.seqLen();
        const steps = packed_run.video.sigmas.len - 1;
        const n_blocks = self.blocks.len;
        const slot_n: usize = config.timestep_slot_count;

        const flat_n = steps * slot_n;
        const flat_t = try allocator.alloc(f32, flat_n);
        defer allocator.free(flat_t);
        const all_tidx = try allocator.alloc(u32, steps * seq_len);
        defer allocator.free(all_tidx);
        const all_adaln = try allocator.alloc(u32, steps * seq_len);
        defer allocator.free(all_adaln);
        for (0..steps) |i| {
            pack.writeRowPlan(
                layout,
                packed_run.video.time(i),
                packed_run.audio.time(i),
                all_tidx[i * seq_len ..][0..seq_len],
                all_adaln[i * seq_len ..][0..seq_len],
                flat_t[i * slot_n ..][0..slot_n],
            );
        }

        var pos_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len, .ax = 3 }, .f32), std.mem.sliceAsBytes(layout.positions));
        defer pos_buf.deinit();
        var refined_text: zml.Buffer = undefined;
        {
            const text_part: TextPrep = .{
                .condition_proj = self.condition_proj,
                .blocks = self.refiner_blocks,
                .final_norm = self.refiner_norm,
                .pi_h3 = self.pi_h3,
            };

            var text_bufs = try load(run, store, TextPrep, &text_part, null);
            defer TextPrep.unload(&text_bufs, allocator);
            var text_runner = try zml.FnExe(TextPrep.forward).Runner(.{.model}).init(&compiled.prepare_text, allocator, .{ .model = text_bufs });
            defer text_runner.deinit(allocator);
            text_runner.run(io, .{
                .inputs = .{ .text = text },
                .outputs = .{ .text = &refined_text },
            }, .{ .wait = true });
        }
        defer refined_text.deinit();

        var rope_runner = try zml.FnExe(Rope.forward).Runner(.{}).init(&compiled.prepare_rope, allocator, .{});
        defer rope_runner.deinit(allocator);
        var cos: zml.Buffer = undefined;
        var sin: zml.Buffer = undefined;
        rope_runner.run(io, .{
            .inputs = .{ .position_ids = pos_buf },
            .outputs = .{ .cos = &cos, .sin = &sin },
        }, .{ .wait = true });
        defer cos.deinit();
        defer sin.deinit();

        var flat_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .n = flat_n }, .f32), std.mem.sliceAsBytes(flat_t));
        defer flat_buf.deinit();
        var all_temb: zml.Buffer = undefined;
        {
            var time_bufs = try load(run, store, TimeEmbedder, &self.time_embedder, null);
            defer zml.Buffer.deinitAll(TimeEmbedder, &time_bufs);
            var temb_runner = try zml.FnExe(TimeEmbedder.forward).Runner(.{.model}).init(&compiled.prepare_temb, allocator, .{ .model = time_bufs });
            defer temb_runner.deinit(allocator);
            temb_runner.run(io, .{
                .inputs = .{ .timestep = flat_buf },
                .outputs = .{ .temb = &all_temb },
            }, .{ .wait = true });
        }
        defer all_temb.deinit();

        var tables = try allocator.alloc(zml.Buffer, n_blocks);
        var tables_filled: usize = 0;
        defer {
            for (tables[0..tables_filled]) |*tb| tb.deinit();
            allocator.free(tables);
        }
        var cores = try allocator.alloc(zml.Bufferized(Block), n_blocks);
        var cores_filled: usize = 0;
        defer {
            for (cores[0..cores_filled]) |*core| zml.Buffer.deinitAll(Block, core);
            allocator.free(cores);
        }
        var loader: zml.io.Loader = try .init(allocator, run.platform, ops.loader_opts);
        defer loader.deinit();
        const AdaLnRunner = zml.FnExe(BlockAdaLn.prepare).Runner(.{.adaln});
        for (self.adalns, self.blocks, tables, cores, 0..) |adaln, block, *table, *core, i| {
            var adaln_bufs = try load(run, store, BlockAdaLn, &adaln, &loader);
            defer zml.Buffer.deinitAll(BlockAdaLn, &adaln_bufs);
            const late = self.pi_h3 and i >= pi.first_replacement_block;
            const exe = if (late) &compiled.late_adaln.? else &compiled.prepare_adaln;
            var adaln_runner = try AdaLnRunner.init(exe, allocator, .{ .adaln = adaln_bufs });
            defer adaln_runner.deinit(allocator);
            adaln_runner.run(io, .{
                .inputs = .{ .temb = all_temb },
                .outputs = .{ .table = table },
            }, .{ .wait = true });
            tables_filled += 1;
            core.* = try load(run, store, Block, &block, &loader);
            cores_filled += 1;
        }

        var final_table: zml.Buffer = undefined;
        {
            var final_adaln = try load(run, store, FinalAdaLn, &self.final_adaln, null);
            defer zml.Buffer.deinitAll(FinalAdaLn, &final_adaln);
            var final_runner = try zml.FnExe(FinalAdaLn.prepare).Runner(.{.adaln}).init(&compiled.prepare_final_adaln, allocator, .{ .adaln = final_adaln });
            defer final_runner.deinit(allocator);
            final_runner.run(io, .{
                .inputs = .{ .temb = all_temb },
                .outputs = .{ .table = &final_table },
            }, .{ .wait = true });
        }
        defer final_table.deinit();

        const patch_part: PatchEmbed = .{
            .video_proj = self.video_proj,
            .audio_proj = self.audio_proj,
            .pi_h3 = self.pi_h3,
        };
        var patch_bufs = try load(run, store, PatchEmbed, &patch_part, null);
        defer zml.Buffer.deinitAll(PatchEmbed, &patch_bufs);
        var patch_runner = try zml.FnExe(PatchEmbed.forward).Runner(.{.model}).init(&compiled.embed_patches, allocator, .{ .model = patch_bufs });
        defer patch_runner.deinit(allocator);
        const finish_part: Finish = .{
            .norm = self.final_norm,
            .video_out = self.video_out,
            .audio_out = self.audio_out,
            .text_len = text_len,
            .audio_len = geo.audio_tokens,
            .pi_h3 = self.pi_h3,
        };
        var finish_bufs = try load(run, store, Finish, &finish_part, null);
        defer zml.Buffer.deinitAll(Finish, &finish_bufs);
        var finish_runner = try zml.FnExe(Finish.forward).Runner(.{.model}).init(&compiled.finish, allocator, .{ .model = finish_bufs });
        defer finish_runner.deinit(allocator);

        const BlockRunner = zml.FnExe(Block.forward).Runner(.{.layer});
        const block_runners = try allocator.alloc(BlockRunner, n_blocks);
        var runners_ready: usize = 0;
        defer {
            for (block_runners[0..runners_ready]) |*r| r.deinit(allocator);
            allocator.free(block_runners);
        }
        for (block_runners, cores, 0..) |*r, core, i| {
            const late = self.pi_h3 and i >= pi.first_replacement_block;
            const exe = if (late) &compiled.late_block.? else &compiled.block;
            r.* = try BlockRunner.init(exe, allocator, .{ .layer = core });
            runners_ready += 1;
        }
        var apply_v = try zml.FnExe(Euler.apply).Runner(.{}).init(&compiled.apply_video, allocator, .{});
        defer apply_v.deinit(allocator);
        var apply_a = try zml.FnExe(Euler.apply).Runner(.{}).init(&compiled.apply_audio, allocator, .{});
        defer apply_a.deinit(allocator);
        var video_buf = try zml.Buffer.fromBytes(io, run.platform, video_shape, std.mem.sliceAsBytes(drawn.video));
        errdefer video_buf.deinit();
        var audio_buf = try zml.Buffer.fromBytes(io, run.platform, audio_shape, std.mem.sliceAsBytes(drawn.audio));
        errdefer audio_buf.deinit();
        var final_features: ?zml.Buffer = null;
        errdefer if (final_features) |*features| features.deinit();

        log.info("denoise: blocks={d} seq={d} audio_tokens={d} devices={d}", .{
            n_blocks,
            seq_len,
            geo.audio_tokens,
            run.platform.devices.len,
        });
        const denoise_start: std.Io.Timestamp = .now(io, .awake);
        for (0..steps) |step_i| {
            const step_start: std.Io.Timestamp = .now(io, .awake);
            var adaln_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len }, .u32), std.mem.sliceAsBytes(all_adaln[step_i * seq_len ..][0..seq_len]));
            defer adaln_buf.deinit();
            var time_idx = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len }, .u32), std.mem.sliceAsBytes(all_tidx[step_i * seq_len ..][0..seq_len]));
            defer time_idx.deinit();
            var step_u32: u32 = @intCast(step_i);
            var step_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .u32), std.mem.asBytes(&step_u32));
            defer step_buf.deinit();
            var sigma_v_item = packed_run.video.sigmas[step_i];
            var sigma_v = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), std.mem.asBytes(&sigma_v_item));
            defer sigma_v.deinit();
            var sigma_v_next_item = packed_run.video.sigmas[step_i + 1];
            var sigma_v_next = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), std.mem.asBytes(&sigma_v_next_item));
            defer sigma_v_next.deinit();
            var sigma_a_item = packed_run.audio.sigmas[step_i];
            var sigma_a = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), std.mem.asBytes(&sigma_a_item));
            defer sigma_a.deinit();
            var sigma_a_next_item = packed_run.audio.sigmas[step_i + 1];
            var sigma_a_next = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), std.mem.asBytes(&sigma_a_next_item));
            defer sigma_a_next.deinit();

            var hidden: zml.Buffer = undefined;
            patch_runner.run(io, .{
                .inputs = .{
                    .video = video_buf,
                    .audio = audio_buf,
                    .text = refined_text,
                },
                .outputs = .{ .hidden = &hidden },
            }, .{ .wait = true });
            defer hidden.deinit();

            // Host sync per block.
            for (block_runners, tables) |*block_runner, table| {
                block_runner.run(io, .{
                    .inputs = .{
                        .hidden = hidden,
                        .table = table,
                        .step = step_buf,
                        .adaln_indices = adaln_buf,
                        .cos = cos,
                        .sin = sin,
                    },
                    .outputs = .{ .hidden = &hidden },
                }, .{ .wait = true });
            }

            // Pi-H3 decodes video (and only video, not audio) from the final-step DiT features instead of the video latents. This block computes the DWD decoder input out of the final DiT hidden states.
            if (self.pi_h3 and step_i + 1 == steps) {
                const feature_model: pi.Features = .{
                    .projection = self.dwd_projection.?,
                    .first_video = layout.videoStart(),
                    .eps = self.cfg.final_norm_eps,
                };
                var weights = try load(run, store, pi.Features, &feature_model, &loader);
                defer zml.Buffer.deinitAll(pi.Features, &weights);
                var runner = try zml.FnExe(pi.Features.forward).Runner(.{.model}).init(&compiled.features.?, allocator, .{ .model = weights });
                defer runner.deinit(allocator);
                var features: zml.Buffer = undefined;
                runner.run(io, .{
                    .inputs = .{ .hidden = hidden, .temb = all_temb, .step = step_buf, .timestep_indices = time_idx },
                    .outputs = .{ .features = &features },
                }, .{ .wait = true });
                final_features = features;
            }
            var video_out: zml.Buffer = undefined;
            var audio_out: zml.Buffer = undefined;
            finish_runner.run(io, .{
                .inputs = .{
                    .hidden = hidden,
                    .table = final_table,
                    .step = step_buf,
                    .timestep_indices = time_idx,
                },
                .outputs = .{ .video = &video_out, .audio = &audio_out },
            }, .{ .wait = true });
            defer video_out.deinit();
            defer audio_out.deinit();

            apply_v.run(io, .{
                .inputs = .{ .sample = video_buf, .velocity = video_out, .sigma = sigma_v, .sigma_next = sigma_v_next },
                .outputs = .{ .sample = &video_buf },
            }, .{ .wait = true });

            apply_a.run(io, .{
                .inputs = .{ .sample = audio_buf, .velocity = audio_out, .sigma = sigma_a, .sigma_next = sigma_a_next },
                .outputs = .{ .sample = &audio_buf },
            }, .{ .wait = true });

            log.info("denoise {d}/{d} t_video={d:.4} t_audio={d:.4} [{f}]", .{
                step_i + 1,
                steps,
                packed_run.video.time(step_i),
                packed_run.audio.time(step_i),
                step_start.untilNow(io, .awake),
            });
        }

        log.info("denoise: ok steps={d} [{f}]", .{ steps, denoise_start.untilNow(io, .awake) });
        if (self.pi_h3) {
            video_buf.deinit();
            return .{ .video = final_features.?, .audio = audio_buf };
        }
        return .{ .video = video_buf, .audio = audio_buf };
    }
};
