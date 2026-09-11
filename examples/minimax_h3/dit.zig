//! MiniMax-H3 DiT.
//!
//!   1. refine text tokens
//!   2. MM-RoPE from packed (t,h,w)
//!   3. time embed every σ → AdaLN tables (4 unique time slots)
//!   4. pack text + noisy audio + noisy video (concat, or scatter when conditioned)
//!   5. 50 AdaLN blocks (one compiled layer, one runner per block)
//!   6. AdaLN the packed sequence, fp32 video/audio heads on those rows, Euler both
//!   7. Euler (η=0): x0 = x + σ v;  x' = (σ'/σ) x + (1 − σ'/σ) x0
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

const log = std.log.scoped(.minimax_h3);

const DitConfig = config.DitConfig;
const linear = ops.linear;
const rms = ops.rms;
const load = ops.load;
const ropeCat3 = ops.ropeCat3;
const Run = ops.Run;
const Packed = pack.Packed;

pub const DeviceLatents = struct {
    video: zml.Buffer,
    audio: zml.Buffer,

    pub fn deinit(self: *DeviceLatents) void {
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

/// Dense `{b,s,d}` pack: text | audio | video.
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
    return hidden.withPartitioning(.{ .d = .replicated });
}

// =============================================================================
// Feed-forward (SwiGLU)
// =============================================================================

const SwiGlu = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGlu {
        return .{
            .fc1 = linear(store, "net.0.proj.weight", null, .{ .dout = .model, .d = .replicated }, .replicated),
            .fc2 = linear(store, "net.2.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
        };
    }

    pub fn forward(self: SwiGlu, x: zml.Tensor) zml.Tensor {
        const value, const gate = self.fc1.forward(x).chunkExact(-1, 2);
        return self.fc2.forward(gate.silu().mul(value).rename(.{ .dout = .d }));
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
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    num_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitConfig) Attention {
        const qkv = .{ .dout = .model, .d = .replicated };
        return .{
            .q = linear(store, "to_q.weight", null, qkv, .replicated),
            .k = linear(store, "to_k.weight", null, qkv, .replicated),
            .v = linear(store, "to_v.weight", null, qkv, .replicated),
            .out = linear(store, "to_out.0.weight", null, .{ .dout = .replicated, .d = .model }, .replicated),
            .q_norm = rms(store.withPrefix("norm_q"), .{.hd}, cfg.qk_norm_eps),
            .k_norm = rms(store.withPrefix("norm_k"), .{.hd}, cfg.qk_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .head_dim = cfg.attention_head_dim,
        };
    }

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?Rotary, backend: zml.attention.Backend) zml.Tensor {
        const heads = .{ .h = self.num_heads, .hd = self.head_dim };
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q_norm.forward(self.q.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model }));
        var k = self.k_norm.forward(self.k.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model }));
        const v = self.v.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        if (rotary) |pe| {
            q = zml.nn.applyRotary(q, pe.cos, pe.sin);
            k = zml.nn.applyRotary(k, pe.cos, pe.sin);
        }
        return self.out.forward(zml.attention.dense(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            backend,
            .{ .is_causal = false },
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } })).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

// =============================================================================
// Time embedder  (sinusoidal t → temb)
// =============================================================================

const TimeEmbedder = struct {
    proj_in: zml.nn.Linear,
    proj_out: zml.nn.Linear,
    freq_dim: i64,
    rope_theta: f32,
    pub const Input = struct { model: TimeEmbedder, timestep: zml.Tensor };
    pub const Output = struct { temb: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: DitConfig) TimeEmbedder {
        return .{
            .proj_in = linear(store, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .proj_out = linear(store, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
            .freq_dim = cfg.freq_dim,
            .rope_theta = cfg.rope_theta,
        };
    }

    pub fn forward(input: Input) Output {
        const inv = zml.nn.invFreq(input.model.freq_dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = input.model.rope_theta } },
        }).withTags(.{.f});
        const angles = input.timestep.convert(.f32).withPartialTags(.{.n}).outer(inv);
        const features = zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d });
        return .{
            .temb = input.model.proj_out.forward(
                input.model.proj_in.forward(features).silu().rename(.{ .dout = .d }),
            ).rename(.{ .dout = .d }),
        };
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
        pub const PrepareInput = struct { adaln: Self, temb: zml.Tensor, steps: i64, slots: i64 };
        pub const PrepareOutput = struct { table: zml.Tensor };

        pub fn init(store: zml.io.TensorStore.View, hidden_size: i64) Self {
            return .{
                .linear = linear(store, "linear.weight", "linear.bias", .replicated, .replicated),
                .hidden_size = hidden_size,
            };
        }

        pub fn prepare(input: PrepareInput) PrepareOutput {
            const raw = input.adaln.linear.forward(input.temb.silu().convert(input.adaln.linear.weight.dtype()))
                .splitAxis(.dout, .{ .mod = modalities, .k = expand, .d = input.adaln.hidden_size });
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
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
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

    pub fn init(store: zml.io.TensorStore.View, cfg: DitConfig) Block {
        return .{
            .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(store.withPrefix("attn"), cfg),
            .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(store.withPrefix("ff")),
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const c = input.table.gather(.{ .t = input.step }, .{}).merge(.{ .n = .{ .n, .mod } })
            .gather(.{ .n = input.adaln_indices }, .{}).chunkExact(.k, 6);

        const residual = input.hidden.withPartitioning(.{ .d = .replicated });
        const attn_out = self.attn.forward(
            shiftScale(self.norm1.forward(residual), c[0], c[1]),
            .{ .cos = input.cos, .sin = input.sin },
            input.attn_backend,
        );
        const x1 = residualGate(residual, c[2], attn_out).withPartitioning(.{ .d = .replicated });
        const mlp_out = self.mlp.forward(shiftScale(self.norm2.forward(x1), c[3], c[4])).rename(.{ .dout = .d });
        return .{
            .hidden = residualGate(x1, c[5], mlp_out).withPartitioning(.{ .d = .replicated }).reuseBuffer(input.hidden),
        };
    }

    fn refine(self: Block, x: zml.Tensor, backend: zml.attention.Backend) zml.Tensor {
        const residual = x.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.attn.forward(self.norm1.forward(residual), null, backend));
        return x1.add(self.mlp.forward(self.norm2.forward(x1)).rename(.{ .dout = .d }))
            .withPartitioning(.{ .d = .replicated })
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
    final_norm: zml.nn.RmsNorm,
    pub const Input = struct { model: TextPrep, text: zml.Tensor, attn_backend: zml.attention.Backend };
    pub const Output = struct { text: zml.Tensor };

    /// Nested `blocks` slice is not freed by `Buffer.deinitAll`.
    fn unload(self: *zml.Bufferized(TextPrep), allocator: std.mem.Allocator) void {
        zml.nn.Linear.unloadBuffers(&self.condition_proj);
        for (self.blocks) |*block| zml.Buffer.deinitAll(Block, block);
        allocator.free(self.blocks);
        self.final_norm.weight.deinit();
    }

    pub fn forward(input: Input) Output {
        var text = input.model.condition_proj.forward(input.text.convert(input.model.condition_proj.weight.dtype())).rename(.{ .dout = .d });
        text = text.convert(input.model.final_norm.weight.dtype());
        for (input.model.blocks) |block| text = block.refine(text, input.attn_backend);
        return .{ .text = input.model.final_norm.forward(text) };
    }
};

/// Pack refined text, projected audio, and video patches into one sequence.
const PatchEmbed = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    pub const Input = struct {
        model: PatchEmbed,
        video: zml.Tensor,
        audio: zml.Tensor,
        text: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const dt = input.text.dtype();
        const video = input.model.video_proj.forward(input.video.convert(input.model.video_proj.weight.dtype())).rename(.{ .dout = .d }).convert(dt);
        const audio = input.model.audio_proj.forward(input.audio.convert(input.model.audio_proj.weight.dtype())).rename(.{ .dout = .d }).convert(dt);
        return .{ .hidden = packedHidden(input.text, audio, video) };
    }
};

/// Official scatter: text / video / audio rows land at layout indices, not a 3-way concat.
const ScatterPatch = struct {
    video_proj: zml.nn.Linear,
    audio_proj: zml.nn.Linear,
    seq: i64,
    pub const Input = struct {
        model: ScatterPatch,
        video: zml.Tensor,
        audio: zml.Tensor,
        text: zml.Tensor,
        text_indices: zml.Tensor,
        video_indices: zml.Tensor,
        audio_indices: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const dt = input.text.dtype();
        const video = input.model.video_proj.forward(input.video.convert(input.model.video_proj.weight.dtype())).rename(.{ .dout = .d }).convert(dt);
        const audio = input.model.audio_proj.forward(input.audio.convert(input.model.audio_proj.weight.dtype())).rename(.{ .dout = .d }).convert(dt);
        var hidden = zml.Tensor.zeroes(zml.Shape.init(.{
            .b = input.text.dim(.b),
            .s = input.model.seq,
            .d = input.text.dim(.d),
        }, dt));
        hidden = hidden.scatterSlices(.{ .s = input.text_indices.withTags(.{.s}) }, input.text, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        hidden = hidden.scatterSlices(.{ .s = input.video_indices.withTags(.{.s}) }, video, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        hidden = hidden.scatterSlices(.{ .s = input.audio_indices.withTags(.{.s}) }, audio, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        return .{ .hidden = hidden.withPartitioning(.{ .d = .replicated }) };
    }
};

/// Final RMS + time-slot AdaLN, then fp32 video/audio heads on those rows only.
const Finish = struct {
    norm: zml.nn.RmsNorm,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,
    text_len: i64,
    audio_len: i64,
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
        const n = input.model.norm.forward(input.hidden.withPartitioning(.{ .d = .replicated }));
        const shift, const scale = mods.gather(.{ .n = input.timestep_indices }, .{}).chunkExact(.k, 2);
        const head_dt = input.model.video_out.weight.dtype();
        const audio_end = input.model.text_len + input.model.audio_len;
        const audio_s: zml.Tensor.Slice = .{ .start = input.model.text_len, .end = audio_end };
        const video_s: zml.Tensor.Slice = .{ .start = audio_end };
        return .{
            .video = input.model.video_out.forward(
                shiftScale(n.slice(.s, video_s), shift.slice(.s, video_s), scale.slice(.s, video_s)).convert(head_dt),
            ).rename(.{ .dout = .d }),
            .audio = input.model.audio_out.forward(
                shiftScale(n.slice(.s, audio_s), shift.slice(.s, audio_s), scale.slice(.s, audio_s)).convert(head_dt),
            ).rename(.{ .dout = .d }),
        };
    }
};

const GatherFinish = struct {
    norm: zml.nn.RmsNorm,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,
    pub const Input = struct {
        model: GatherFinish,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        timestep_indices: zml.Tensor,
        video_indices: zml.Tensor,
        audio_indices: zml.Tensor,
    };
    pub const Output = struct { video: zml.Tensor, audio: zml.Tensor };

    pub fn forward(input: Input) Output {
        const mods = input.table.gather(.{ .t = input.step }, .{});
        const video_h = input.hidden.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const audio_h = input.hidden.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const video_t = input.timestep_indices.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const audio_t = input.timestep_indices.gather(.{ .s = input.audio_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s });
        const n_v = input.model.norm.forward(video_h.withPartitioning(.{ .d = .replicated }));
        const n_a = input.model.norm.forward(audio_h.withPartitioning(.{ .d = .replicated }));
        const v_shift, const v_scale = mods.gather(.{ .n = video_t }, .{}).chunkExact(.k, 2);
        const a_shift, const a_scale = mods.gather(.{ .n = audio_t }, .{}).chunkExact(.k, 2);
        const head_dt = input.model.video_out.weight.dtype();
        return .{
            .video = input.model.video_out.forward(shiftScale(n_v, v_shift, v_scale).convert(head_dt)).rename(.{ .dout = .d }),
            .audio = input.model.audio_out.forward(shiftScale(n_a, a_shift, a_scale).convert(head_dt)).rename(.{ .dout = .d }),
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

/// Rectified-flow Euler (η=0):  x0 = x + σ v;  x' = (σ'/σ) x + (1 − σ'/σ) x0.
const Euler = struct {
    pub const Input = struct { sample: zml.Tensor, velocity: zml.Tensor, sigma: zml.Tensor, sigma_next: zml.Tensor };
    pub const Output = struct { sample: zml.Tensor };

    pub fn apply(input: Input) Output {
        const x = input.sample;
        const v = input.velocity.convert(x.dtype());
        const sigma = input.sigma.convert(x.dtype()).broad(x.shape());
        const x0 = x.add(v.mul(sigma));
        const ratio = input.sigma_next.convert(.f32).div(input.sigma.convert(.f32)).broad(x.convert(.f32).shape());
        return .{
            .sample = ratio.mul(x.convert(.f32))
                .add(zml.Tensor.scalar(1.0, .f32).sub(ratio).mul(x0.convert(.f32)))
                .convert(x.dtype())
                .reuseBuffer(x),
        };
    }
};

/// Euler, then copy the condition prefix back so keyframes/refs stay put.
const EulerHold = struct {
    pub const Input = struct {
        sample: zml.Tensor,
        velocity: zml.Tensor,
        sigma: zml.Tensor,
        sigma_next: zml.Tensor,
        hold: i64,
    };
    pub const Output = struct { sample: zml.Tensor };

    pub fn apply(input: Input) Output {
        const stepped = Euler.apply(.{
            .sample = input.sample,
            .velocity = input.velocity,
            .sigma = input.sigma,
            .sigma_next = input.sigma_next,
        }).sample;
        const kept = input.sample.slice(.s, .{ .start = 0, .end = input.hold });
        return .{ .sample = stepped.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(0, .i32) }, kept) };
    }
};

const TakeTail = struct {
    pub const Input = struct { sample: zml.Tensor, hold: i64 };
    pub const Output = struct { sample: zml.Tensor };

    pub fn forward(input: Input) Output {
        return .{ .sample = input.sample.slice(.s, .{ .start = input.hold }) };
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
    refiner_norm: zml.nn.RmsNorm,
    blocks: []Block,
    adalns: []BlockAdaLn,
    final_adaln: FinalAdaLn,
    final_norm: zml.nn.RmsNorm,
    video_out: zml.nn.Linear,
    audio_out: zml.nn.Linear,
    cfg: DitConfig,
    compiled: ?Compiled = null,

    const Compiled = struct {
        prepare_text: zml.FnExe(TextPrep.forward),
        prepare_rope: zml.FnExe(Rope.forward),
        embed_patches: ?zml.FnExe(PatchEmbed.forward) = null,
        scatter_embed: ?zml.FnExe(ScatterPatch.forward) = null,
        prepare_temb: zml.FnExe(TimeEmbedder.forward),
        prepare_adaln: zml.FnExe(BlockAdaLn.prepare),
        prepare_final_adaln: zml.FnExe(FinalAdaLn.prepare),
        block: zml.FnExe(Block.forward),
        finish: ?zml.FnExe(Finish.forward) = null,
        gather_finish: ?zml.FnExe(GatherFinish.forward) = null,
        apply_video: ?zml.FnExe(Euler.apply) = null,
        apply_audio: ?zml.FnExe(Euler.apply) = null,
        apply_hold: ?zml.FnExe(EulerHold.apply) = null,
        apply_audio_hold: ?zml.FnExe(EulerHold.apply) = null,
        take_tail: ?zml.FnExe(TakeTail.forward) = null,
        take_audio_tail: ?zml.FnExe(TakeTail.forward) = null,

        fn deinit(self: *Compiled) void {
            self.prepare_text.deinit();
            self.prepare_rope.deinit();
            if (self.embed_patches) |*e| e.deinit();
            if (self.scatter_embed) |*e| e.deinit();
            self.prepare_temb.deinit();
            self.prepare_adaln.deinit();
            self.prepare_final_adaln.deinit();
            self.block.deinit();
            if (self.finish) |*e| e.deinit();
            if (self.gather_finish) |*e| e.deinit();
            if (self.apply_video) |*e| e.deinit();
            if (self.apply_audio) |*e| e.deinit();
            if (self.apply_hold) |*e| e.deinit();
            if (self.apply_audio_hold) |*e| e.deinit();
            if (self.take_tail) |*e| e.deinit();
            if (self.take_audio_tail) |*e| e.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: DitConfig) !Dit {
        const n_layers: usize = @intCast(cfg.num_layers);
        const blocks = try allocator.alloc(Block, n_layers);
        errdefer allocator.free(blocks);
        const adalns = try allocator.alloc(BlockAdaLn, n_layers);
        errdefer allocator.free(adalns);
        const refiner = store.withPrefix("token_refiner");
        const refiner_blocks = try allocator.alloc(Block, @intCast(cfg.num_refiner_layers));
        errdefer allocator.free(refiner_blocks);
        for (refiner_blocks, 0..) |*block, i| block.* = .init(refiner.withPrefix("refiner_blocks").withLayer(i), cfg);
        for (blocks, adalns, 0..) |*block, *adaln, i| {
            const layer = store.withPrefix("transformer_blocks").withLayer(i);
            block.* = .init(layer, cfg);
            adaln.* = .init(layer.withPrefix("adaln_proj"), cfg.hidden_size);
        }
        return .{
            .video_proj = linear(store, "proj_in.weight", "proj_in.bias", .replicated, .replicated),
            .audio_proj = linear(store, "audio_proj_in.weight", "audio_proj_in.bias", .replicated, .replicated),
            .condition_proj = linear(store, "context_embedder.weight", "context_embedder.bias", .replicated, .replicated),
            .time_embedder = .init(store.withPrefix("time_embedder"), cfg),
            .refiner_blocks = refiner_blocks,
            .refiner_norm = rms(refiner.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
            .blocks = blocks,
            .adalns = adalns,
            .final_adaln = .init(store.withPrefix("norm_out"), cfg.hidden_size),
            .final_norm = rms(store.withPrefix("norm_out.norm"), .{.d}, cfg.final_norm_eps),
            .video_out = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
            .audio_out = linear(store, "audio_proj_out.weight", "audio_proj_out.bias", .replicated, .replicated),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *Dit, allocator: std.mem.Allocator) void {
        if (self.compiled) |*c| c.deinit();
        allocator.free(self.refiner_blocks);
        allocator.free(self.blocks);
        allocator.free(self.adalns);
    }

    pub fn dropCompiled(self: *Dit) void {
        if (self.compiled) |*c| {
            c.deinit();
            self.compiled = null;
        }
    }

    /// Text, RoPE, pack, temb, AdaLN, final AdaLN, block, finish, plus Euler or hold/tail.
    /// Shapes come from `packed_run`.
    pub fn compile(self: *Dit, run: *const Run, geo: config.Geometry, text_len: u32, packed_run: Packed, text_dt: zml.DataType) !void {
        const attn = zml.attention.Backend.auto(run.platform);
        const layer = self.blocks[0];
        const seq_len: u32 = packed_run.layout.seqLen();
        const steps: u32 = @intCast(packed_run.video.sigmas.len - 1);
        const flat_n: i64 = @intCast(steps * config.timestep_slot_count);
        const temb_dim = self.time_embedder.proj_out.weight.dim(.dout);
        log.info("dit attn={s} pack={s} seq={d} audio_tokens={d} devices={d}", .{
            @tagName(attn),
            if (packed_run.layout.scatter()) "scatter" else "concat",
            seq_len,
            geo.audio_tokens,
            run.platform.devices.len,
        });
        const hold_video = packed_run.layout.cond_video_len != 0;
        const hold_audio = packed_run.layout.cond_audio_len != 0;
        const n_kernels: usize = 8 + @as(usize, if (hold_video) 2 else 1) + @as(usize, if (hold_audio) 2 else 1);
        var node = run.progress.start("Compiling MiniMax-H3 DiT", n_kernels);
        defer node.end();
        const dt = layer.norm1.weight.dtype();

        const prepare_text = try zml.FnExe(TextPrep.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_prepare_text",
        }, .{.{
            .model = .{
                .condition_proj = self.condition_proj,
                .blocks = self.refiner_blocks,
                .final_norm = self.refiner_norm,
            },
            .text = .init(.{ .b = 1, .s = text_len, .d = self.cfg.text_dim }, text_dt),
            .attn_backend = attn,
        }});
        errdefer prepare_text.deinit();
        const prepare_rope = try zml.FnExe(Rope.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_prepare_rope",
        }, .{.{
            .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
            .rope_freq_dim = self.cfg.rope_freq_dim,
            .rope_theta = self.cfg.rope_theta,
            .out_dtype = dt,
        }});
        errdefer prepare_rope.deinit();
        var embed_patches: ?zml.FnExe(PatchEmbed.forward) = null;
        var scatter_embed: ?zml.FnExe(ScatterPatch.forward) = null;
        if (packed_run.layout.scatter()) {
            scatter_embed = try zml.FnExe(ScatterPatch.forward).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_scatter",
            }, .{.{
                .model = .{
                    .video_proj = self.video_proj,
                    .audio_proj = self.audio_proj,
                    .seq = seq_len,
                },
                .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .audio = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .text = .init(.{ .b = 1, .s = text_len, .d = self.cfg.hidden_size }, dt),
                .text_indices = .init(.{ .s = text_len }, .u32),
                .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
                .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
            }});
        } else {
            embed_patches = try zml.FnExe(PatchEmbed.forward).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_embed_patches",
            }, .{.{
                .model = .{
                    .video_proj = self.video_proj,
                    .audio_proj = self.audio_proj,
                },
                .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .audio = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .text = .init(.{ .b = 1, .s = text_len, .d = self.cfg.hidden_size }, dt),
            }});
        }
        errdefer if (embed_patches) |*e| e.deinit();
        errdefer if (scatter_embed) |*e| e.deinit();
        const prepare_temb = try zml.FnExe(TimeEmbedder.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_prepare_temb",
        }, .{.{
            .model = self.time_embedder,
            .timestep = .init(.{ .n = flat_n }, .f32),
        }});
        errdefer prepare_temb.deinit();
        const prepare_adaln = try zml.FnExe(BlockAdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_prepare_adaln",
        }, .{.{
            .adaln = self.adalns[0],
            .temb = .init(.{ .n = flat_n, .d = temb_dim }, .f32),
            .steps = steps,
            .slots = config.timestep_slot_count,
        }});
        errdefer prepare_adaln.deinit();
        const prepare_final_adaln = try zml.FnExe(FinalAdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_prepare_final_adaln",
        }, .{.{
            .adaln = self.final_adaln,
            .temb = .init(.{ .n = flat_n, .d = temb_dim }, .f32),
            .steps = steps,
            .slots = config.timestep_slot_count,
        }});
        errdefer prepare_final_adaln.deinit();

        const block_exe = try zml.FnExe(Block.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_block",
        }, .{.{
            .layer = layer,
            .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = self.cfg.hidden_size }, dt),
            .table = zml.Tensor.init(.{ .t = steps, .n = config.timestep_slot_count, .mod = config.modality_count, .k = 6, .d = self.cfg.hidden_size }, dt),
            .step = zml.Tensor.init(.{}, .u32),
            .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
            .cos = zml.Tensor.init(.{ .s = seq_len, .f = self.cfg.rotaryDim() }, dt),
            .sin = zml.Tensor.init(.{ .s = seq_len, .f = self.cfg.rotaryDim() }, dt),
            .attn_backend = attn,
        }});
        errdefer block_exe.deinit();
        var finish_exe: ?zml.FnExe(Finish.forward) = null;
        var gather_finish: ?zml.FnExe(GatherFinish.forward) = null;
        if (packed_run.layout.scatter()) {
            gather_finish = try zml.FnExe(GatherFinish.forward).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_finish",
            }, .{.{
                .model = .{
                    .norm = self.final_norm,
                    .video_out = self.video_out,
                    .audio_out = self.audio_out,
                },
                .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = self.cfg.hidden_size }, dt),
                .table = zml.Tensor.init(.{ .t = steps, .n = config.timestep_slot_count, .k = 2, .d = self.cfg.hidden_size }, dt),
                .step = zml.Tensor.init(.{}, .u32),
                .timestep_indices = .init(.{ .s = seq_len }, .u32),
                .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
                .audio_indices = .init(.{ .s = geo.audio_tokens }, .u32),
            }});
        } else {
            finish_exe = try zml.FnExe(Finish.forward).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_finish",
            }, .{.{
                .model = .{
                    .norm = self.final_norm,
                    .video_out = self.video_out,
                    .audio_out = self.audio_out,
                    .text_len = text_len,
                    .audio_len = geo.audio_tokens,
                },
                .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = self.cfg.hidden_size }, dt),
                .table = zml.Tensor.init(.{ .t = steps, .n = config.timestep_slot_count, .k = 2, .d = self.cfg.hidden_size }, dt),
                .step = zml.Tensor.init(.{}, .u32),
                .timestep_indices = .init(.{ .s = seq_len }, .u32),
            }});
        }
        errdefer if (finish_exe) |*e| e.deinit();
        errdefer if (gather_finish) |*e| e.deinit();
        var apply_video: ?zml.FnExe(Euler.apply) = null;
        var apply_audio: ?zml.FnExe(Euler.apply) = null;
        var apply_hold: ?zml.FnExe(EulerHold.apply) = null;
        var apply_audio_hold: ?zml.FnExe(EulerHold.apply) = null;
        var take_tail: ?zml.FnExe(TakeTail.forward) = null;
        var take_audio_tail: ?zml.FnExe(TakeTail.forward) = null;
        errdefer if (apply_video) |*e| e.deinit();
        errdefer if (apply_audio) |*e| e.deinit();
        errdefer if (apply_hold) |*e| e.deinit();
        errdefer if (apply_audio_hold) |*e| e.deinit();
        errdefer if (take_tail) |*e| e.deinit();
        errdefer if (take_audio_tail) |*e| e.deinit();
        if (!hold_video) {
            apply_video = try zml.FnExe(Euler.apply).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_apply_video",
            }, .{.{
                .sample = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .velocity = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .sigma = .init(.{}, .f32),
                .sigma_next = .init(.{}, .f32),
            }});
        }
        if (!hold_audio) {
            apply_audio = try zml.FnExe(Euler.apply).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_apply_audio",
            }, .{.{
                .sample = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .velocity = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .sigma = .init(.{}, .f32),
                .sigma_next = .init(.{}, .f32),
            }});
        }
        if (hold_video) {
            const hold: i64 = packed_run.layout.cond_video_len;
            apply_hold = try zml.FnExe(EulerHold.apply).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_apply_video_hold",
            }, .{.{
                .sample = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .velocity = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .sigma = .init(.{}, .f32),
                .sigma_next = .init(.{}, .f32),
                .hold = hold,
            }});
            take_tail = try zml.FnExe(TakeTail.forward).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_take_tail",
            }, .{.{
                .sample = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
                .hold = hold,
            }});
        }
        if (hold_audio) {
            const hold: i64 = packed_run.layout.cond_audio_len;
            apply_audio_hold = try zml.FnExe(EulerHold.apply).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_apply_audio_hold",
            }, .{.{
                .sample = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .velocity = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .sigma = .init(.{}, .f32),
                .sigma_next = .init(.{}, .f32),
                .hold = hold,
            }});
            take_audio_tail = try zml.FnExe(TakeTail.forward).compile(run.allocator, run.io, run.platform, .{
                .shardings = &run.mesh,
                .program_name = "minimax_h3_take_audio_tail",
            }, .{.{
                .sample = .init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32),
                .hold = hold,
            }});
        }
        self.compiled = .{
            .prepare_text = prepare_text,
            .prepare_rope = prepare_rope,
            .embed_patches = embed_patches,
            .scatter_embed = scatter_embed,
            .prepare_temb = prepare_temb,
            .prepare_adaln = prepare_adaln,
            .prepare_final_adaln = prepare_final_adaln,
            .block = block_exe,
            .finish = finish_exe,
            .gather_finish = gather_finish,
            .apply_video = apply_video,
            .apply_audio = apply_audio,
            .apply_hold = apply_hold,
            .apply_audio_hold = apply_audio_hold,
            .take_tail = take_tail,
            .take_audio_tail = take_audio_tail,
        };
        apply_video = null;
        apply_audio = null;
        apply_hold = null;
        apply_audio_hold = null;
        take_tail = null;
        take_audio_tail = null;
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
    ) !DeviceLatents {
        const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
        const allocator = run.allocator;
        const io = run.io;
        var drawn = try pack.noise(allocator, seed, geo, packed_run);
        defer drawn.deinit(allocator);
        const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
        const audio_shape = zml.Shape.init(.{ .b = 1, .s = geo.audio_tokens, .d = geo.audio_dim }, .f32);
        const layout = packed_run.layout;
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

        var pos_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len, .ax = 3 }, .f32), .replicated, std.mem.sliceAsBytes(layout.positions));
        defer pos_buf.deinit();
        var adaln_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len }, .u32), .replicated, std.mem.sliceAsBytes(all_adaln[0..seq_len]));
        defer adaln_buf.deinit();
        var time_idx = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len }, .u32), .replicated, std.mem.sliceAsBytes(all_tidx[0..seq_len]));
        defer time_idx.deinit();

        const text_part: TextPrep = .{
            .condition_proj = self.condition_proj,
            .blocks = self.refiner_blocks,
            .final_norm = self.refiner_norm,
        };
        var text_bufs = try load(run, store, TextPrep, &text_part, null);
        defer TextPrep.unload(&text_bufs, allocator);
        var text_runner = try zml.FnExe(TextPrep.forward).Runner(.{.model}).init(&compiled.prepare_text, allocator, .{ .model = text_bufs });
        defer text_runner.deinit(allocator);
        var refined_text: zml.Buffer = undefined;
        text_runner.run(io, .{ .inputs = .{ .text = text }, .outputs = .{ .text = &refined_text }, .opts = .{ .wait = true } });
        defer refined_text.deinit();

        var rope_runner = try zml.FnExe(Rope.forward).Runner(.{}).init(&compiled.prepare_rope, allocator, .{});
        defer rope_runner.deinit(allocator);
        var cos: zml.Buffer = undefined;
        var sin: zml.Buffer = undefined;
        rope_runner.run(io, .{ .inputs = .{ .position_ids = pos_buf }, .outputs = .{ .cos = &cos, .sin = &sin }, .opts = .{ .wait = true } });
        defer cos.deinit();
        defer sin.deinit();

        var flat_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .n = flat_n }, .f32), .replicated, std.mem.sliceAsBytes(flat_t));
        defer flat_buf.deinit();
        var time_bufs = try load(run, store, TimeEmbedder, &self.time_embedder, null);
        var all_temb: zml.Buffer = undefined;
        {
            var temb_runner = try zml.FnExe(TimeEmbedder.forward).Runner(.{.model}).init(&compiled.prepare_temb, allocator, .{ .model = time_bufs });
            defer temb_runner.deinit(allocator);
            temb_runner.run(io, .{ .inputs = .{ .timestep = flat_buf }, .outputs = .{ .temb = &all_temb }, .opts = .{ .wait = true } });
        }
        defer all_temb.deinit();
        zml.Buffer.deinitAll(TimeEmbedder, &time_bufs);

        var tables = try allocator.alloc(zml.Buffer, n_blocks);
        var tables_filled: usize = 0;
        errdefer {
            for (tables[0..tables_filled]) |*tb| tb.deinit();
            allocator.free(tables);
        }
        var cores = try allocator.alloc(zml.Bufferized(Block), n_blocks);
        var cores_filled: usize = 0;
        errdefer {
            for (cores[0..cores_filled]) |*core| zml.Buffer.deinitAll(Block, core);
            allocator.free(cores);
        }
        var loader: zml.io.Loader = try .init(allocator, run.platform, ops.loader_opts);
        defer loader.deinit();
        const AdaLnRunner = zml.FnExe(BlockAdaLn.prepare).Runner(.{.adaln});
        for (self.adalns, self.blocks, tables, cores) |adaln, block, *table, *core| {
            var adaln_bufs = try load(run, store, BlockAdaLn, &adaln, &loader);
            defer zml.Buffer.deinitAll(BlockAdaLn, &adaln_bufs);
            var adaln_runner = try AdaLnRunner.init(&compiled.prepare_adaln, allocator, .{ .adaln = adaln_bufs });
            defer adaln_runner.deinit(allocator);
            adaln_runner.run(io, .{ .inputs = .{ .temb = all_temb }, .outputs = .{ .table = table }, .opts = .{ .wait = true } });
            tables_filled += 1;
            core.* = try load(run, store, Block, &block, &loader);
            cores_filled += 1;
        }

        var final_table: zml.Buffer = undefined;
        {
            var final_adaln = try load(run, store, FinalAdaLn, &self.final_adaln, null);
            var final_runner = try zml.FnExe(FinalAdaLn.prepare).Runner(.{.adaln}).init(&compiled.prepare_final_adaln, allocator, .{ .adaln = final_adaln });
            defer final_runner.deinit(allocator);
            final_runner.run(io, .{ .inputs = .{ .temb = all_temb }, .outputs = .{ .table = &final_table }, .opts = .{ .wait = true } });
            zml.Buffer.deinitAll(FinalAdaLn, &final_adaln);
        }
        defer final_table.deinit();

        const scatter = packed_run.layout.scatter();
        var text_idx_buf: ?zml.Buffer = null;
        var video_idx_buf: ?zml.Buffer = null;
        var audio_idx_buf: ?zml.Buffer = null;
        defer if (text_idx_buf) |*b| b.deinit();
        defer if (video_idx_buf) |*b| b.deinit();
        defer if (audio_idx_buf) |*b| b.deinit();
        if (scatter) {
            text_idx_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = text_len }, .u32), .replicated, std.mem.sliceAsBytes(layout.text_indices));
            video_idx_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = geo.video_tokens }, .u32), .replicated, std.mem.sliceAsBytes(layout.video_indices));
            audio_idx_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = geo.audio_tokens }, .u32), .replicated, std.mem.sliceAsBytes(layout.audio_indices));
        }

        const patch_part: PatchEmbed = .{
            .video_proj = self.video_proj,
            .audio_proj = self.audio_proj,
        };
        var patch_bufs: ?zml.Bufferized(PatchEmbed) = null;
        defer if (patch_bufs) |*b| zml.Buffer.deinitAll(PatchEmbed, b);
        const PatchRunner = zml.FnExe(PatchEmbed.forward).Runner(.{.model});
        var patch_runner: ?PatchRunner = null;
        defer if (patch_runner) |*r| r.deinit(allocator);

        const scatter_part: ScatterPatch = .{
            .video_proj = self.video_proj,
            .audio_proj = self.audio_proj,
            .seq = @intCast(seq_len),
        };
        var scatter_bufs: ?zml.Bufferized(ScatterPatch) = null;
        defer if (scatter_bufs) |*b| zml.Buffer.deinitAll(ScatterPatch, b);
        const ScatterRunner = zml.FnExe(ScatterPatch.forward).Runner(.{.model});
        var scatter_runner: ?ScatterRunner = null;
        defer if (scatter_runner) |*r| r.deinit(allocator);

        if (compiled.scatter_embed) |*exe| {
            scatter_bufs = try load(run, store, ScatterPatch, &scatter_part, null);
            scatter_runner = try ScatterRunner.init(exe, allocator, .{ .model = scatter_bufs.? });
        } else if (compiled.embed_patches) |*exe| {
            patch_bufs = try load(run, store, PatchEmbed, &patch_part, null);
            patch_runner = try PatchRunner.init(exe, allocator, .{ .model = patch_bufs.? });
        } else return error.NotCompiled;

        const finish_part: Finish = .{
            .norm = self.final_norm,
            .video_out = self.video_out,
            .audio_out = self.audio_out,
            .text_len = text_len,
            .audio_len = geo.audio_tokens,
        };
        var finish_bufs: ?zml.Bufferized(Finish) = null;
        defer if (finish_bufs) |*b| zml.Buffer.deinitAll(Finish, b);
        const FinishRunner = zml.FnExe(Finish.forward).Runner(.{.model});
        var finish_runner: ?FinishRunner = null;
        defer if (finish_runner) |*r| r.deinit(allocator);

        const gather_part: GatherFinish = .{
            .norm = self.final_norm,
            .video_out = self.video_out,
            .audio_out = self.audio_out,
        };
        var gather_bufs: ?zml.Bufferized(GatherFinish) = null;
        defer if (gather_bufs) |*b| zml.Buffer.deinitAll(GatherFinish, b);
        const GatherRunner = zml.FnExe(GatherFinish.forward).Runner(.{.model});
        var gather_runner: ?GatherRunner = null;
        defer if (gather_runner) |*r| r.deinit(allocator);

        if (compiled.gather_finish) |*exe| {
            gather_bufs = try load(run, store, GatherFinish, &gather_part, null);
            gather_runner = try GatherRunner.init(exe, allocator, .{ .model = gather_bufs.? });
        } else if (compiled.finish) |*exe| {
            finish_bufs = try load(run, store, Finish, &finish_part, null);
            finish_runner = try FinishRunner.init(exe, allocator, .{ .model = finish_bufs.? });
        } else return error.NotCompiled;

        const BlockRunner = zml.FnExe(Block.forward).Runner(.{.layer});
        const block_runners = try allocator.alloc(BlockRunner, n_blocks);
        var runners_ready: usize = 0;
        defer {
            for (block_runners[0..runners_ready]) |*r| r.deinit(allocator);
            allocator.free(block_runners);
        }
        for (block_runners, cores) |*r, core| {
            r.* = try BlockRunner.init(&compiled.block, allocator, .{ .layer = core });
            runners_ready += 1;
        }
        const EulerRunner = zml.FnExe(Euler.apply).Runner(.{});
        var apply_v: ?EulerRunner = null;
        defer if (apply_v) |*r| r.deinit(allocator);
        if (compiled.apply_video) |*exe| {
            apply_v = try EulerRunner.init(exe, allocator, .{});
        }
        var apply_a: ?EulerRunner = null;
        defer if (apply_a) |*r| r.deinit(allocator);
        if (compiled.apply_audio) |*exe| {
            apply_a = try EulerRunner.init(exe, allocator, .{});
        }
        const HoldRunner = zml.FnExe(EulerHold.apply).Runner(.{});
        var apply_h: ?HoldRunner = null;
        defer if (apply_h) |*r| r.deinit(allocator);
        if (compiled.apply_hold) |*exe| {
            apply_h = try HoldRunner.init(exe, allocator, .{});
        }
        var apply_ah: ?HoldRunner = null;
        defer if (apply_ah) |*r| r.deinit(allocator);
        if (compiled.apply_audio_hold) |*exe| {
            apply_ah = try HoldRunner.init(exe, allocator, .{});
        }
        var video_buf = try zml.Buffer.fromBytes(io, run.platform, video_shape, .replicated, std.mem.sliceAsBytes(drawn.video));
        errdefer video_buf.deinit();
        var audio_buf = try zml.Buffer.fromBytes(io, run.platform, audio_shape, .replicated, std.mem.sliceAsBytes(drawn.audio));
        errdefer audio_buf.deinit();

        log.info("denoise: blocks={d} seq={d} audio_tokens={d} devices={d}", .{
            n_blocks,
            seq_len,
            geo.audio_tokens,
            run.platform.devices.len,
        });
        const denoise_start: std.Io.Timestamp = .now(io, .awake);
        for (0..steps) |step_i| {
            const step_start: std.Io.Timestamp = .now(io, .awake);
            if (step_i != 0) {
                adaln_buf.deinit();
                adaln_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len }, .u32), .replicated, std.mem.sliceAsBytes(all_adaln[step_i * seq_len ..][0..seq_len]));
                time_idx.deinit();
                time_idx = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq_len }, .u32), .replicated, std.mem.sliceAsBytes(all_tidx[step_i * seq_len ..][0..seq_len]));
            }
            var step_u32: u32 = @intCast(step_i);
            var step_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .u32), .replicated, std.mem.asBytes(&step_u32));
            defer step_buf.deinit();
            var sigma_v_item = packed_run.video.sigmas[step_i];
            var sigma_v = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&sigma_v_item));
            defer sigma_v.deinit();
            var sigma_v_next_item = packed_run.video.sigmas[step_i + 1];
            var sigma_v_next = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&sigma_v_next_item));
            defer sigma_v_next.deinit();
            var sigma_a_item = packed_run.audio.sigmas[step_i];
            var sigma_a = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&sigma_a_item));
            defer sigma_a.deinit();
            var sigma_a_next_item = packed_run.audio.sigmas[step_i + 1];
            var sigma_a_next = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&sigma_a_next_item));
            defer sigma_a_next.deinit();

            var hidden: zml.Buffer = undefined;
            if (scatter_runner) |*r| {
                r.run(io, .{
                    .inputs = .{
                        .video = video_buf,
                        .audio = audio_buf,
                        .text = refined_text,
                        .text_indices = text_idx_buf.?,
                        .video_indices = video_idx_buf.?,
                        .audio_indices = audio_idx_buf.?,
                    },
                    .outputs = .{ .hidden = &hidden },
                    .opts = .{ .wait = true },
                });
            } else {
                patch_runner.?.run(io, .{
                    .inputs = .{
                        .video = video_buf,
                        .audio = audio_buf,
                        .text = refined_text,
                    },
                    .outputs = .{ .hidden = &hidden },
                    .opts = .{ .wait = true },
                });
            }
            defer hidden.deinit();

            for (block_runners, tables) |*block_runner, table| {
                var next: zml.Buffer = undefined;
                block_runner.run(io, .{
                    .inputs = .{
                        .hidden = hidden,
                        .table = table,
                        .step = step_buf,
                        .adaln_indices = adaln_buf,
                        .cos = cos,
                        .sin = sin,
                    },
                    .outputs = .{ .hidden = &next },
                    .opts = .{ .wait = true },
                });
                hidden.deinit();
                hidden = next;
            }

            var video_out: zml.Buffer = undefined;
            var audio_out: zml.Buffer = undefined;
            if (gather_runner) |*r| {
                r.run(io, .{
                    .inputs = .{
                        .hidden = hidden,
                        .table = final_table,
                        .step = step_buf,
                        .timestep_indices = time_idx,
                        .video_indices = video_idx_buf.?,
                        .audio_indices = audio_idx_buf.?,
                    },
                    .outputs = .{ .video = &video_out, .audio = &audio_out },
                    .opts = .{ .wait = true },
                });
            } else {
                finish_runner.?.run(io, .{
                    .inputs = .{
                        .hidden = hidden,
                        .table = final_table,
                        .step = step_buf,
                        .timestep_indices = time_idx,
                    },
                    .outputs = .{ .video = &video_out, .audio = &audio_out },
                    .opts = .{ .wait = true },
                });
            }
            defer video_out.deinit();
            defer audio_out.deinit();

            var next_video: zml.Buffer = undefined;
            if (apply_h) |*r| {
                r.run(io, .{
                    .inputs = .{ .sample = video_buf, .velocity = video_out, .sigma = sigma_v, .sigma_next = sigma_v_next },
                    .outputs = .{ .sample = &next_video },
                    .opts = .{ .wait = true },
                });
            } else if (apply_v) |*r| {
                r.run(io, .{
                    .inputs = .{ .sample = video_buf, .velocity = video_out, .sigma = sigma_v, .sigma_next = sigma_v_next },
                    .outputs = .{ .sample = &next_video },
                    .opts = .{ .wait = true },
                });
            } else return error.NotCompiled;
            video_buf.deinit();
            video_buf = next_video;

            var next_audio: zml.Buffer = undefined;
            if (apply_ah) |*r| {
                r.run(io, .{
                    .inputs = .{ .sample = audio_buf, .velocity = audio_out, .sigma = sigma_a, .sigma_next = sigma_a_next },
                    .outputs = .{ .sample = &next_audio },
                    .opts = .{ .wait = true },
                });
            } else if (apply_a) |*r| {
                r.run(io, .{
                    .inputs = .{ .sample = audio_buf, .velocity = audio_out, .sigma = sigma_a, .sigma_next = sigma_a_next },
                    .outputs = .{ .sample = &next_audio },
                    .opts = .{ .wait = true },
                });
            } else return error.NotCompiled;
            audio_buf.deinit();
            audio_buf = next_audio;

            log.info("denoise {d}/{d} t_video={d:.4} t_audio={d:.4} [{f}]", .{
                step_i + 1,
                steps,
                packed_run.video.time(step_i),
                packed_run.audio.time(step_i),
                step_start.untilNow(io, .awake),
            });
        }

        log.info("denoise: ok steps={d} [{f}]", .{ steps, denoise_start.untilNow(io, .awake) });
        if (compiled.take_tail) |*exe| {
            var tail = try zml.FnExe(TakeTail.forward).Runner(.{}).init(exe, allocator, .{});
            defer tail.deinit(allocator);
            var trimmed: zml.Buffer = undefined;
            tail.run(io, .{
                .inputs = .{ .sample = video_buf },
                .outputs = .{ .sample = &trimmed },
                .opts = .{ .wait = true },
            });
            video_buf.deinit();
            video_buf = trimmed;
        }
        if (compiled.take_audio_tail) |*exe| {
            var tail = try zml.FnExe(TakeTail.forward).Runner(.{}).init(exe, allocator, .{});
            defer tail.deinit(allocator);
            var trimmed: zml.Buffer = undefined;
            tail.run(io, .{
                .inputs = .{ .sample = audio_buf },
                .outputs = .{ .sample = &trimmed },
                .opts = .{ .wait = true },
            });
            audio_buf.deinit();
            audio_buf = trimmed;
        }
        for (tables) |*tb| tb.deinit();
        allocator.free(tables);
        for (cores) |*core| zml.Buffer.deinitAll(Block, core);
        allocator.free(cores);
        return .{ .video = video_buf, .audio = audio_buf };
    }
};
