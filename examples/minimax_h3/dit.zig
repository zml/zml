//! MiniMax-H3 DiT. Python: `MiniMaxH3Transformer3DModel`.
//!
//!   1. refine text tokens (`MiniMaxH3TokenRefiner`)
//!   2. MM-RoPE from packed (t,h,w)
//!   3. time embed every σ → AdaLN tables
//!   4. scatter text + noisy video into one sequence
//!   5. 50 AdaLN blocks (compiled as 5 groups of 10)
//!   6. project video rows → velocity v
//!   7. Euler (η=0): x0 = x + σ v;  x' = (σ'/σ) x + (1 − σ'/σ) x0
//!
//! Block (`MiniMaxH3TransformerBlock`):
//!   shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = AdaLN(temb)
//!   h ← h + gate_msa * Attn( RMS(h) * (1+scale_msa) + shift_msa )
//!   h ← h + gate_mlp * FF  ( RMS(h) * (1+scale_mlp) + shift_mlp )

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");
const pack = @import("pack.zig");

const log = std.log.scoped(.minimax_h3);

const DitCfg = config.Config;
const linear = ops.linear;
const rms = ops.rms;
const load = ops.load;
const ropeCat3 = ops.ropeCat3;
const Run = ops.Run;
const Packed = pack.Packed;

/// 10 transformer blocks per XLA kernel — a 50-block kernel OOMs at 768P.
const group_size: u32 = 10;

fn shiftScale(x: zml.Tensor, shift: zml.Tensor, scale: zml.Tensor) zml.Tensor {
    const dt = x.dtype();
    return x.mul(zml.Tensor.scalar(1.0, dt).add(scale.squeeze(.k).convert(dt).broad(x.shape())))
        .add(shift.squeeze(.k).convert(dt).broad(x.shape()));
}

fn residualGate(x: zml.Tensor, gate: zml.Tensor, y: zml.Tensor) zml.Tensor {
    return x.add(gate.squeeze(.k).convert(y.dtype()).broad(y.shape()).mul(y));
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
// Attention  (`MiniMaxH3Attention`, bidirectional)
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
    attn_backend: zml.attention.Backend = .vanilla,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) Attention {
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

    pub fn forward(self: Attention, x: zml.Tensor, rotary: ?struct { zml.Tensor, zml.Tensor }) zml.Tensor {
        const heads = .{ .h = self.num_heads, .hd = self.head_dim };
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        var k = self.k.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        const v = self.v.forward(x_qkv).splitAxis(.dout, heads).withPartitioning(.{ .h = .model });
        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);
        if (rotary) |pe| {
            q = zml.nn.applyRotary(q, pe[0], pe[1]);
            k = zml.nn.applyRotary(k, pe[0], pe[1]);
        }
        return self.out.forward(zml.attention.dense(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            self.attn_backend,
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
    pub const Input = struct { model: TimeEmbedder, timestep: zml.Tensor, freq_dim: i64 };
    pub const Output = struct { temb: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) TimeEmbedder {
        const p = store.withPrefix("time_embedder");
        return .{
            .proj_in = linear(p, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .proj_out = linear(p, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
        };
    }

    pub fn outDim(self: TimeEmbedder) i64 {
        return self.proj_out.weight.dim(.dout);
    }

    pub fn forward(input: Input) Output {
        const inv = zml.nn.invFreq(input.freq_dim, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = 10000.0 } },
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
// AdaLN  (`MiniMaxH3AdaLayerNormModulation`)
// =============================================================================

/// Maps temb → (shift, scale, gate) × {msa, mlp}, one set per (timestep, modality).
const AdaLn = struct {
    linear: zml.nn.Linear,
    hidden_size: i64,
    expand: i64,
    modalities: i64,
    pub const PrepareInput = struct { adaln: AdaLn, temb: zml.Tensor, steps: i64, slots: i64 };
    pub const PrepareOutput = struct { table: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, hidden_size: i64, expand: i64, modalities: i64) AdaLn {
        return .{
            .linear = linear(store, "linear.weight", "linear.bias", .replicated, .replicated),
            .hidden_size = hidden_size,
            .expand = expand,
            .modalities = modalities,
        };
    }

    pub fn forward(self: AdaLn, temb: zml.Tensor) zml.Tensor {
        const raw = self.linear.forward(temb.silu().convert(self.linear.weight.dtype()));
        return if (self.modalities == 1)
            raw.splitAxis(.dout, .{ .k = self.expand, .d = self.hidden_size })
        else
            raw.splitAxis(.dout, .{ .mod = self.modalities, .k = self.expand, .d = self.hidden_size });
    }

    pub fn prepare(input: PrepareInput) PrepareOutput {
        return .{ .table = input.adaln.forward(input.temb).splitAxis(.n, .{ .t = input.steps, .n = input.slots }) };
    }
};

// =============================================================================
// Transformer block  (`MiniMaxH3TransformerBlock`)
// =============================================================================

const BlockCore = struct {
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
    mlp: SwiGlu,
    pub const Input = struct {
        layer: BlockCore,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const gathered = input.table.gather(.{ .t = input.step }, .{});
        const mods = if (gathered.shape().hasTag(.mod)) |_| gathered.merge(.{ .n = .{ .n, .mod } }) else gathered;
        const shift_msa, const scale_msa, const gate_msa, const shift_mlp, const scale_mlp, const gate_mlp =
            mods.gather(.{ .n = input.adaln_indices }, .{}).chunkExact(.k, 6);

        const residual = input.hidden.withPartitioning(.{ .d = .replicated });
        const attn_out = self.attn.forward(
            shiftScale(self.norm1.forward(residual), shift_msa, scale_msa),
            .{ input.cos, input.sin },
        );
        const x1 = residualGate(residual, gate_msa, attn_out).withPartitioning(.{ .d = .replicated });
        const mlp_out = self.mlp.forward(
            shiftScale(self.norm2.forward(x1), shift_mlp, scale_mlp),
        ).rename(.{ .dout = .d });
        return .{
            .hidden = residualGate(x1, gate_mlp, mlp_out).withPartitioning(.{ .d = .replicated }).reuseBuffer(input.hidden),
        };
    }
};

/// `group_size` BlockCores in one compiled kernel.
const BlockGroup = struct {
    layers: []BlockCore,
    pub const Input = struct {
        group: BlockGroup,
        hidden: zml.Tensor,
        tables: []zml.Tensor,
        step: zml.Tensor,
        adaln_indices: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        var hidden = input.hidden;
        for (input.group.layers, input.tables) |layer, table| {
            hidden = BlockCore.forward(.{
                .layer = layer,
                .hidden = hidden,
                .table = table,
                .step = input.step,
                .adaln_indices = input.adaln_indices,
                .cos = input.cos,
                .sin = input.sin,
            }).hidden;
        }
        return .{ .hidden = hidden };
    }
};

/// Weights for one layer: AdaLN projector + `BlockCore`.
const DitBlock = struct {
    core: BlockCore,
    adaln: AdaLn,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) DitBlock {
        return .{
            .core = .{
                .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
                .attn = .init(store.withPrefix("attn"), cfg),
                .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
                .mlp = .init(store.withPrefix("ff")),
            },
            .adaln = .init(store.withPrefix("adaln_proj"), cfg.hidden_size, 6, config.modality_count),
        };
    }
};

// =============================================================================
// Text refiner  (`MiniMaxH3TokenRefiner`)
// =============================================================================

const TokenRefinerBlock = struct {
    norm1: zml.nn.RmsNorm,
    attn: Attention,
    norm2: zml.nn.RmsNorm,
    mlp: SwiGlu,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) TokenRefinerBlock {
        return .{
            .norm1 = rms(store.withPrefix("norm1"), .{.d}, cfg.norm_eps),
            .attn = .init(store.withPrefix("attn"), cfg),
            .norm2 = rms(store.withPrefix("norm2"), .{.d}, cfg.norm_eps),
            .mlp = .init(store.withPrefix("ff")),
        };
    }

    pub fn forward(self: TokenRefinerBlock, x: zml.Tensor) zml.Tensor {
        const residual = x.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.attn.forward(self.norm1.forward(residual), null));
        return x1.add(self.mlp.forward(self.norm2.forward(x1)).rename(.{ .dout = .d }))
            .withPartitioning(.{ .d = .replicated })
            .reuseBuffer(x);
    }
};

// =============================================================================
// Final layer  (`MiniMaxH3AdaLayerNormOut` + `proj_out`)
// =============================================================================

const FinalLayer = struct {
    norm: zml.nn.RmsNorm,
    adaln: AdaLn,
    video_out: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, cfg: DitCfg) FinalLayer {
        return .{
            .norm = rms(store.withPrefix("norm_out.norm"), .{.d}, cfg.final_norm_eps),
            .adaln = .init(store.withPrefix("norm_out"), cfg.hidden_size, 2, 1),
            .video_out = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
        };
    }
};

// =============================================================================
// DiT
// =============================================================================

pub const Dit = struct {
    video_proj: zml.nn.Linear,
    condition_proj: zml.nn.Linear,
    time_embedder: TimeEmbedder,
    refiner_blocks: []TokenRefinerBlock,
    refiner_norm: zml.nn.RmsNorm,
    blocks: []DitBlock,
    final_layer: FinalLayer,
    cfg: DitCfg,
    compiled: ?Compiled = null,

    const Compiled = struct {
        prepare_text: zml.FnExe(TextPrep.forward),
        prepare_rope: zml.FnExe(Rope.forward),
        embed_patches: zml.FnExe(PatchEmbed.forward),
        prepare_temb: zml.FnExe(TimeEmbedder.forward),
        prepare_adaln: zml.FnExe(AdaLn.prepare),
        prepare_final_adaln: zml.FnExe(AdaLn.prepare),
        block_group: zml.FnExe(BlockGroup.forward),
        finish: zml.FnExe(FinishCore.forward),
        apply_video: zml.FnExe(Euler.apply),

        fn deinit(self: *Compiled) void {
            self.prepare_text.deinit();
            self.prepare_rope.deinit();
            self.embed_patches.deinit();
            self.prepare_temb.deinit();
            self.prepare_adaln.deinit();
            self.prepare_final_adaln.deinit();
            self.block_group.deinit();
            self.finish.deinit();
            self.apply_video.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Dit {
        const cfg: DitCfg = .{};
        const blocks = try allocator.alloc(DitBlock, @intCast(cfg.num_layers));
        errdefer allocator.free(blocks);
        const refiner = store.withPrefix("token_refiner");
        const refiner_blocks = try allocator.alloc(TokenRefinerBlock, @intCast(cfg.num_refiner_layers));
        errdefer allocator.free(refiner_blocks);
        for (refiner_blocks, 0..) |*block, i| block.* = .init(refiner.withPrefix("refiner_blocks").withLayer(i), cfg);
        for (blocks, 0..) |*block, i| block.* = .init(store.withPrefix("transformer_blocks").withLayer(i), cfg);
        return .{
            .video_proj = linear(store, "proj_in.weight", "proj_in.bias", .replicated, .replicated),
            .condition_proj = linear(store, "context_embedder.weight", "context_embedder.bias", .replicated, .replicated),
            .time_embedder = .init(store),
            .refiner_blocks = refiner_blocks,
            .refiner_norm = rms(refiner.withPrefix("final_norm"), .{.d}, cfg.final_norm_eps),
            .blocks = blocks,
            .final_layer = FinalLayer.init(store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *Dit, allocator: std.mem.Allocator) void {
        if (self.compiled) |*c| c.deinit();
        allocator.free(self.refiner_blocks);
        allocator.free(self.blocks);
    }

    /// Compile the nine DiT kernels. Shapes come from `packed_run`.
    pub fn compile(self: *Dit, run: *const Run, geo: config.Geometry, text_len: u32, packed_run: Packed, text_dt: zml.DataType) !void {
        return compileDit(
            self,
            run,
            geo,
            text_len,
            packed_run.layout.seqLen(),
            @intCast(packed_run.video.stepCount()),
            text_dt,
        );
    }

    /// Encoder hidden + packed layout → denoised video tokens `{s, 96}`.
    pub fn denoise(
        self: *const Dit,
        run: *const Run,
        store: *zml.io.TensorStore,
        geo: config.Geometry,
        text: zml.Buffer,
        text_len: u32,
        packed_run: Packed,
        seed: u64,
    ) ![]f32 {
        return denoiseDit(self, run, store, geo, text, text_len, packed_run, seed);
    }

    fn textPrep(self: Dit) TextPrep {
        return .{ .condition_proj = self.condition_proj, .blocks = self.refiner_blocks, .final_norm = self.refiner_norm };
    }

    fn patchEmbed(self: Dit) PatchEmbed {
        return .{ .video_proj = self.video_proj, .hidden_size = self.cfg.hidden_size, .seq = 0 };
    }

    fn finishCore(self: Dit) FinishCore {
        return .{ .norm = self.final_layer.norm, .video_out = self.final_layer.video_out };
    }
};

/// Linear `context_embedder` then two refiner blocks.
const TextPrep = struct {
    condition_proj: zml.nn.Linear,
    blocks: []TokenRefinerBlock,
    final_norm: zml.nn.RmsNorm,
    pub const Input = struct { model: TextPrep, text: zml.Tensor };
    pub const Output = struct { text: zml.Tensor };

    fn unload(self: *zml.Bufferized(TextPrep), allocator: std.mem.Allocator) void {
        zml.nn.Linear.unloadBuffers(&self.condition_proj);
        for (self.blocks) |*block| zml.Buffer.deinitAll(TokenRefinerBlock, block);
        allocator.free(self.blocks);
        self.final_norm.weight.deinit();
    }

    pub fn forward(input: Input) Output {
        var text = input.model.condition_proj.forward(input.text.convert(input.model.condition_proj.weight.dtype())).rename(.{ .dout = .d });
        text = text.convert(input.model.final_norm.weight.dtype());
        for (input.model.blocks) |block| text = block.forward(text);
        return .{ .text = input.model.final_norm.forward(text) };
    }
};

/// Scatter refined text and projected video patches into the packed sequence.
const PatchEmbed = struct {
    video_proj: zml.nn.Linear,
    hidden_size: i64,
    seq: i64 = 0,
    pub const Input = struct {
        model: PatchEmbed,
        video: zml.Tensor,
        text: zml.Tensor,
        video_indices: zml.Tensor,
        text_indices: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const video = input.model.video_proj.forward(input.video.convert(input.model.video_proj.weight.dtype())).rename(.{ .dout = .d });
        var hidden = zml.Tensor.zeroes(zml.Shape.init(
            .{ .b = input.text.dim(.b), .s = input.model.seq, .d = input.model.hidden_size },
            input.text.dtype(),
        ));
        hidden = hidden.scatterSlices(.{ .s = input.text_indices.withTags(.{.s}) }, input.text, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        hidden = hidden.scatterSlices(.{ .s = input.video_indices.withTags(.{.s}) }, video.convert(input.text.dtype()), .{ .update_fn = zml.Tensor.ScatterOpts.override });
        return .{ .hidden = hidden.withPartitioning(.{ .d = .replicated }) };
    }
};

/// Gather video rows, AdaLN, project to 96-wide velocity tokens.
const FinishCore = struct {
    norm: zml.nn.RmsNorm,
    video_out: zml.nn.Linear,
    pub const Input = struct {
        model: FinishCore,
        hidden: zml.Tensor,
        table: zml.Tensor,
        step: zml.Tensor,
        timestep_indices: zml.Tensor,
        video_indices: zml.Tensor,
    };
    pub const Output = struct { video: zml.Tensor };

    pub fn forward(input: Input) Output {
        const n = input.model.norm.forward(
            input.hidden.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s }).withPartitioning(.{ .d = .replicated }),
        );
        const selected = input.table.gather(.{ .t = input.step }, .{}).gather(.{
            .n = input.timestep_indices.gather(.{ .s = input.video_indices.withTags(.{.idx}) }, .{}).rename(.{ .idx = .s }),
        }, .{});
        const shift, const scale = selected.chunkExact(.k, 2);
        return .{ .video = input.model.video_out.forward(shiftScale(n, shift, scale).convert(input.model.video_out.weight.dtype())) };
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

// =============================================================================
// Compile
// =============================================================================

/// Nine kernels: text, RoPE, patch scatter, temb, AdaLN, final AdaLN, block group, finish, Euler.
fn compileDit(
    self: *Dit,
    run: *const Run,
    geo: config.Geometry,
    text_len: u32,
    seq_len: u32,
    steps: u32,
    text_dt: zml.DataType,
) !void {
    var model = self.*;
    const attn = zml.attention.Backend.auto(run.platform);
    for (model.blocks) |*block| block.core.attn.attn_backend = attn;
    log.info("dit attn={s} group={d} seq={d} devices={d}", .{
        @tagName(attn),
        group_size,
        seq_len,
        run.platform.devices.len,
    });
    var node = run.progress.start("Compiling MiniMax-H3 DiT", 9);
    defer node.end();
    const dt = model.blocks[0].core.norm1.weight.dtype();
    var patch_part = model.patchEmbed();
    patch_part.seq = seq_len;

    const prepare_text = try zml.FnExe(TextPrep.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_prepare_text",
    }, .{.{
        .model = model.textPrep(),
        .text = .init(.{ .b = 1, .s = text_len, .d = model.cfg.text_dim }, text_dt),
    }});
    errdefer prepare_text.deinit();
    const prepare_rope = try zml.FnExe(Rope.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_prepare_rope",
    }, .{.{
        .position_ids = .init(.{ .s = seq_len, .ax = 3 }, .f32),
        .rope_freq_dim = model.cfg.rope_freq_dim,
        .rope_theta = model.cfg.rope_theta,
        .out_dtype = dt,
    }});
    errdefer prepare_rope.deinit();
    const embed_patches = try zml.FnExe(PatchEmbed.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_embed_patches",
    }, .{.{
        .model = patch_part,
        .video = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .text = .init(.{ .b = 1, .s = text_len, .d = model.cfg.hidden_size }, dt),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
        .text_indices = .init(.{ .s = text_len }, .u32),
    }});
    errdefer embed_patches.deinit();
    const prepare_temb = try zml.FnExe(TimeEmbedder.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_prepare_temb",
    }, .{.{
        .model = model.time_embedder,
        .timestep = .init(.{ .n = steps }, .f32),
        .freq_dim = model.cfg.freq_dim,
    }});
    errdefer prepare_temb.deinit();
    const prepare_adaln = try zml.FnExe(AdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_prepare_adaln",
    }, .{.{
        .adaln = model.blocks[0].adaln,
        .temb = .init(.{ .n = steps, .d = model.time_embedder.outDim() }, .f32),
        .steps = steps,
        .slots = 1,
    }});
    errdefer prepare_adaln.deinit();
    const prepare_final_adaln = try zml.FnExe(AdaLn.prepare).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_prepare_final_adaln",
    }, .{.{
        .adaln = model.final_layer.adaln,
        .temb = .init(.{ .n = steps, .d = model.time_embedder.outDim() }, .f32),
        .steps = steps,
        .slots = 1,
    }});
    errdefer prepare_final_adaln.deinit();

    const layers = try run.allocator.alloc(BlockCore, group_size);
    defer run.allocator.free(layers);
    const tables = try run.allocator.alloc(zml.Tensor, group_size);
    defer run.allocator.free(tables);
    for (layers, tables, 0..) |*layer, *tab, i| {
        layer.* = model.blocks[i].core;
        tab.* = zml.Tensor.init(.{ .t = steps, .n = 1, .mod = config.modality_count, .k = 6, .d = model.cfg.hidden_size }, dt);
    }
    const block_group = try zml.FnExe(BlockGroup.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_block_group",
    }, .{.{
        .group = .{ .layers = layers },
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = model.cfg.hidden_size }, dt),
        .tables = tables,
        .step = zml.Tensor.init(.{}, .u32),
        .adaln_indices = zml.Tensor.init(.{ .s = seq_len }, .u32),
        .cos = zml.Tensor.init(.{ .s = seq_len, .f = model.cfg.rotaryDim() }, dt),
        .sin = zml.Tensor.init(.{ .s = seq_len, .f = model.cfg.rotaryDim() }, dt),
    }});
    errdefer block_group.deinit();
    const finish_exe = try zml.FnExe(FinishCore.forward).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_finish",
    }, .{.{
        .model = model.finishCore(),
        .hidden = zml.Tensor.init(.{ .b = 1, .s = seq_len, .d = model.cfg.hidden_size }, dt),
        .table = zml.Tensor.init(.{ .t = steps, .n = 1, .k = 2, .d = model.cfg.hidden_size }, dt),
        .step = zml.Tensor.init(.{}, .u32),
        .timestep_indices = .init(.{ .s = seq_len }, .u32),
        .video_indices = .init(.{ .s = geo.video_tokens }, .u32),
    }});
    errdefer finish_exe.deinit();
    const apply_video = try zml.FnExe(Euler.apply).compile(run.allocator, run.io, run.platform, .{
        .shardings = run.mesh(),
        .program_name = "minimax_h3_apply_video",
    }, .{.{
        .sample = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .velocity = .init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32),
        .sigma = .init(.{}, .f32),
        .sigma_next = .init(.{}, .f32),
    }});
    self.compiled = .{
        .prepare_text = prepare_text,
        .prepare_rope = prepare_rope,
        .embed_patches = embed_patches,
        .prepare_temb = prepare_temb,
        .prepare_adaln = prepare_adaln,
        .prepare_final_adaln = prepare_final_adaln,
        .block_group = block_group,
        .finish = finish_exe,
        .apply_video = apply_video,
    };
}

// =============================================================================
// Denoise
// =============================================================================

fn denoiseDit(
    self: *const Dit,
    run: *const Run,
    store: *zml.io.TensorStore,
    geo: config.Geometry,
    text: zml.Buffer,
    text_len: u32,
    packed_run: Packed,
    seed: u64,
) ![]f32 {
    const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
    const model = self;
    const allocator = run.allocator;
    const io = run.io;
    const video = try pack.noise(allocator, seed, geo.latent_t, geo.latent_h, geo.latent_w, model.cfg.patch_size);
    errdefer allocator.free(video);
    const video_shape = zml.Shape.init(.{ .b = 1, .s = geo.video_tokens, .d = geo.video_patch_dim }, .f32);
    const layout = packed_run.layout;
    const seq = layout.seqLen();
    const steps = packed_run.video.stepCount();
    const n_blocks = model.blocks.len;
    std.debug.assert(n_blocks % group_size == 0);

    const flat_t = try allocator.alloc(f32, steps);
    defer allocator.free(flat_t);
    for (flat_t, packed_run.video.sigmas[0..steps]) |*tv, s| tv.* = 1.0 - s;
    const tidx = try allocator.alloc(u32, seq);
    defer allocator.free(tidx);
    const adaln = try allocator.alloc(u32, seq);
    defer allocator.free(adaln);
    @memset(tidx, 0);
    layout.writeAdalnIndices(adaln, tidx);

    var pos_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq, .ax = 3 }, .f32), .replicated, std.mem.sliceAsBytes(layout.positions));
    defer pos_buf.deinit();
    var video_idx = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = geo.video_tokens }, .u32), .replicated, std.mem.sliceAsBytes(layout.video_indices));
    defer video_idx.deinit();
    var text_idx = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = text_len }, .u32), .replicated, std.mem.sliceAsBytes(layout.text_indices));
    defer text_idx.deinit();
    var adaln_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq }, .u32), .replicated, std.mem.sliceAsBytes(adaln));
    defer adaln_buf.deinit();
    var time_idx = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .s = seq }, .u32), .replicated, std.mem.sliceAsBytes(tidx));
    defer time_idx.deinit();

    const text_part = model.textPrep();
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

    var flat_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{ .n = steps }, .f32), .replicated, std.mem.sliceAsBytes(flat_t));
    defer flat_buf.deinit();
    var time_bufs = try load(run, store, TimeEmbedder, &model.time_embedder, null);
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
    var cores = try allocator.alloc(zml.Bufferized(BlockCore), n_blocks);
    var cores_filled: usize = 0;
    errdefer {
        for (cores[0..cores_filled]) |*core| zml.Buffer.deinitAll(BlockCore, core);
        allocator.free(cores);
    }
    var loader: zml.io.Loader = try .init(allocator, run.platform, ops.loader_opts);
    defer loader.deinit();
    const AdaLnRunner = zml.FnExe(AdaLn.prepare).Runner(.{.adaln});
    var adaln_runner: ?AdaLnRunner = null;
    defer if (adaln_runner) |*r| r.deinit(allocator);
    var prev_adaln: ?zml.Bufferized(AdaLn) = null;
    defer if (prev_adaln) |*a| zml.Buffer.deinitAll(AdaLn, a);
    for (0..n_blocks) |block_i| {
        const adaln_bufs = try load(run, store, AdaLn, &model.blocks[block_i].adaln, &loader);
        if (adaln_runner) |*r| {
            r.rebake(.{ .adaln = adaln_bufs });
            if (prev_adaln) |*a| zml.Buffer.deinitAll(AdaLn, a);
        } else {
            adaln_runner = try AdaLnRunner.init(&compiled.prepare_adaln, allocator, .{ .adaln = adaln_bufs });
        }
        prev_adaln = adaln_bufs;
        var table: zml.Buffer = undefined;
        adaln_runner.?.run(io, .{ .inputs = .{ .temb = all_temb }, .outputs = .{ .table = &table }, .opts = .{ .wait = true } });
        tables[block_i] = table;
        tables_filled += 1;
        cores[block_i] = try load(run, store, BlockCore, &model.blocks[block_i].core, &loader);
        cores_filled += 1;
    }

    var final_table: zml.Buffer = undefined;
    {
        var final_adaln = try load(run, store, AdaLn, &model.final_layer.adaln, null);
        var final_runner = try AdaLnRunner.init(&compiled.prepare_final_adaln, allocator, .{ .adaln = final_adaln });
        defer final_runner.deinit(allocator);
        final_runner.run(io, .{ .inputs = .{ .temb = all_temb }, .outputs = .{ .table = &final_table }, .opts = .{ .wait = true } });
        zml.Buffer.deinitAll(AdaLn, &final_adaln);
    }
    defer final_table.deinit();

    const patch_part = model.patchEmbed();
    var patch_bufs = try load(run, store, PatchEmbed, &patch_part, null);
    defer zml.Buffer.deinitAll(PatchEmbed, &patch_bufs);
    var patch_runner = try zml.FnExe(PatchEmbed.forward).Runner(.{.model}).init(&compiled.embed_patches, allocator, .{ .model = patch_bufs });
    defer patch_runner.deinit(allocator);
    const finish_part = model.finishCore();
    var finish_bufs = try load(run, store, FinishCore, &finish_part, null);
    defer zml.Buffer.deinitAll(FinishCore, &finish_bufs);
    var finish_runner = try zml.FnExe(FinishCore.forward).Runner(.{.model}).init(&compiled.finish, allocator, .{ .model = finish_bufs });
    defer finish_runner.deinit(allocator);

    const GroupRunner = zml.FnExe(BlockGroup.forward).Runner(.{.group});
    const n_groups = n_blocks / group_size;
    const group_runners = try allocator.alloc(GroupRunner, n_groups);
    var runners_ready: usize = 0;
    defer {
        for (group_runners[0..runners_ready]) |*r| r.deinit(allocator);
        allocator.free(group_runners);
    }
    var g: usize = 0;
    while (g < n_groups) : (g += 1) {
        group_runners[g] = try GroupRunner.init(&compiled.block_group, allocator, .{
            .group = .{ .layers = cores[g * group_size ..][0..group_size] },
        });
        runners_ready += 1;
    }
    var apply_v = try zml.FnExe(Euler.apply).Runner(.{}).init(&compiled.apply_video, allocator, .{});
    defer apply_v.deinit(allocator);
    var video_buf = try zml.Buffer.fromBytes(io, run.platform, video_shape, .replicated, std.mem.sliceAsBytes(video));
    defer video_buf.deinit();

    log.info("denoise: blocks={d} groups={d} seq={d} devices={d}", .{
        n_blocks,
        n_groups,
        seq,
        run.platform.devices.len,
    });
    const denoise_start: std.Io.Timestamp = .now(io, .awake);
    var step_i: usize = 0;
    while (step_i < steps) : (step_i += 1) {
        // pack → 5×10 AdaLN blocks → velocity → Euler
        const step_start: std.Io.Timestamp = .now(io, .awake);
        var step_u32: u32 = @intCast(step_i);
        var step_buf = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .u32), .replicated, std.mem.asBytes(&step_u32));
        defer step_buf.deinit();
        var sigma_v_item = packed_run.video.sigmas[step_i];
        var sigma_v = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&sigma_v_item));
        defer sigma_v.deinit();
        var sigma_v_next_item = packed_run.video.sigmas[step_i + 1];
        var sigma_v_next = try zml.Buffer.fromBytes(io, run.platform, .init(.{}, .f32), .replicated, std.mem.asBytes(&sigma_v_next_item));
        defer sigma_v_next.deinit();

        var hidden: zml.Buffer = undefined;
        patch_runner.run(io, .{
            .inputs = .{ .video = video_buf, .text = refined_text, .video_indices = video_idx, .text_indices = text_idx },
            .outputs = .{ .hidden = &hidden },
            .opts = .{ .wait = true },
        });
        var held: std.ArrayList(zml.Buffer) = .empty;
        defer {
            for (held.items) |*buf| buf.deinit();
            held.deinit(allocator);
        }
        try held.append(allocator, hidden);

        g = 0;
        while (g < n_groups) : (g += 1) {
            var next: zml.Buffer = undefined;
            group_runners[g].run(io, .{
                .inputs = .{
                    .hidden = hidden,
                    .tables = tables[g * group_size ..][0..group_size],
                    .step = step_buf,
                    .adaln_indices = adaln_buf,
                    .cos = cos,
                    .sin = sin,
                },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden = next;
            try held.append(allocator, next);
        }

        var video_out: zml.Buffer = undefined;
        finish_runner.run(io, .{
            .inputs = .{ .hidden = hidden, .table = final_table, .step = step_buf, .timestep_indices = time_idx, .video_indices = video_idx },
            .outputs = .{ .video = &video_out },
            .opts = .{ .wait = true },
        });
        defer video_out.deinit();

        var next_video: zml.Buffer = undefined;
        apply_v.run(io, .{
            .inputs = .{ .sample = video_buf, .velocity = video_out, .sigma = sigma_v, .sigma_next = sigma_v_next },
            .outputs = .{ .sample = &next_video },
            .opts = .{ .wait = true },
        });
        video_buf.deinit();
        video_buf = next_video;
        log.info("denoise {d}/{d} t={d:.4} [{f}]", .{
            step_i + 1,
            steps,
            packed_run.video.time(step_i),
            step_start.untilNow(io, .awake),
        });
    }

    try video_buf.toSlice(io, .init(video_shape, std.mem.sliceAsBytes(video)));
    log.info("denoise: ok steps={d} [{f}]", .{ steps, denoise_start.untilNow(io, .awake) });
    for (tables) |*tb| tb.deinit();
    allocator.free(tables);
    for (cores) |*core| zml.Buffer.deinitAll(BlockCore, core);
    allocator.free(cores);
    return video;
}
