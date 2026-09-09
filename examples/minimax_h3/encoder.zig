//! Qwen text tower. MiniMax-H3 `text_encoder` (first 50 of 64 layers).
//!
//!   tokens → embed → 50 × (RMSNorm → GQA causal attn → RMSNorm → SwiGLU) → hidden [s, 5120]

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");

const EncoderConfig = config.EncoderConfig;
const linear = ops.linear;
const rms = ops.rms;
const load = ops.load;
const Run = ops.Run;

/// Qwen3 eager: `(q @ k.T) * scale` then fp32 softmax. Scale-on-K (`zml.nn.sdpa`) drifts in bf16.
/// DiT uses `zml.attention.dense` (FA2); this path stays local because of that scale.
fn qwenSdpa(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor) zml.Tensor {
    var q = q_.splitAxis(.h, .{ .h = k_.dim(.h), .hq = .auto });
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(q.dim(.hd))));
    const mask = zml.nn.causalAttnMask(.{ .q = q.dim(.q), .k = k_.dim(.k) }, .f32, null);
    var scores = q.dot(k_, .hd).convert(.f32).scale(scale);
    scores = scores.add(mask.broad(scores.shape()));
    const attn = scores.softmax(.k).convert(q.dtype()).dot(v_, .k);
    return attn.transpose(q.shape()).merge(.{ .h = .{ .h, .hq } });
}

const EmbedTokens = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    pub const Input = struct { embedding: EmbedTokens, tokens: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        return .{
            .hidden = input.embedding.embed_tokens.forward(input.tokens.withPartialTags(.{.s}))
                .withPartialTags(.{.d})
                .withPartitioning(.{ .d = .replicated }),
        };
    }
};

// =============================================================================
// SwiGLU MLP
// =============================================================================

const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = linear(store, "up_proj.weight", null, .{ .dout = .model }, .replicated),
            .gate_proj = linear(store, "gate_proj.weight", null, .{ .dout = .model }, .replicated),
            .down_proj = linear(store, "down_proj.weight", null, .{ .d = .model }, .replicated),
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.down_proj.forward(
            self.gate_proj.forward(x).silu().mul(self.up_proj.forward(x)).rename(.{ .dout = .d }),
        );
    }
};

// =============================================================================
// Causal GQA attention  (64 query / 8 KV heads)
// =============================================================================

const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, cfg: EncoderConfig) SelfAttn {
        return .{
            .q_proj = linear(store, "q_proj.weight", null, .{ .dout = .model }, .replicated),
            .k_proj = linear(store, "k_proj.weight", null, .{ .dout = .model }, .replicated),
            .v_proj = linear(store, "v_proj.weight", null, .{ .dout = .model }, .replicated),
            .o_proj = linear(store, "o_proj.weight", null, .{ .d = .model }, .replicated),
            .q_norm = rms(store.withPrefix("q_norm"), .{.hd}, cfg.rms_norm_eps),
            .k_norm = rms(store.withPrefix("k_norm"), .{.hd}, cfg.rms_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
        };
    }

    pub fn forward(self: SelfAttn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        const q_heads = .{ .h = self.num_heads, .hd = self.head_dim };
        const kv_heads = .{ .h = self.num_kv_heads, .hd = self.head_dim };
        const q = zml.nn.applyRotary(
            self.q_norm.forward(self.q_proj.forward(x_qkv).splitAxis(.dout, q_heads).withPartitioning(.{ .h = .model })),
            cos,
            sin,
        );
        const k = zml.nn.applyRotary(
            self.k_norm.forward(self.k_proj.forward(x_qkv).splitAxis(.dout, kv_heads).withPartitioning(.{ .h = .model })),
            cos,
            sin,
        );
        const v = self.v_proj.forward(x_qkv).splitAxis(.dout, kv_heads).withPartitioning(.{ .h = .model });
        const attn = qwenSdpa(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), v.rename(.{ .s = .k }))
            .rename(.{ .q = .s })
            .merge(.{ .d = .{ .h, .hd } });
        return self.o_proj.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

// =============================================================================
// Transformer layer
// =============================================================================

/// One Qwen block:  h ← h + Attn(RMS(h));  h ← h + MLP(RMS(h)).
const TransformerLayer = struct {
    input_layernorm: zml.nn.RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: zml.nn.RmsNorm,
    mlp: Mlp,
    pub const Input = struct { layer: TransformerLayer, hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: EncoderConfig) TransformerLayer {
        return .{
            .input_layernorm = rms(store.withPrefix("input_layernorm"), .{.d}, cfg.rms_norm_eps),
            .self_attn = .init(store.withPrefix("self_attn"), cfg),
            .post_attention_layernorm = rms(store.withPrefix("post_attention_layernorm"), .{.d}, cfg.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual = input.hidden.withPartitioning(.{ .d = .replicated });
        const x1 = residual.add(self.self_attn.forward(self.input_layernorm.forward(residual), input.cos, input.sin))
            .withPartitioning(.{ .d = .replicated });
        return .{
            .hidden = x1.add(self.mlp.forward(self.post_attention_layernorm.forward(x1)).rename(.{ .dout = .d }))
                .withPartitioning(.{ .d = .replicated })
                .reuseBuffer(input.hidden),
        };
    }
};

fn uploadF32(run: *const Run, shape: zml.Shape, values: []const f32) !zml.Buffer {
    switch (shape.dtype()) {
        .f32 => return zml.Buffer.fromBytes(run.io, run.platform, shape, .replicated, std.mem.sliceAsBytes(values)),
        .bf16 => {
            const converted = try run.allocator.alloc(zml.floats.BFloat16, values.len);
            defer run.allocator.free(converted);
            for (converted, values) |*dst, src| dst.* = .fromF32(src);
            return zml.Buffer.fromBytes(run.io, run.platform, shape, .replicated, std.mem.sliceAsBytes(converted));
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

/// Qwen interleaved RoPE: each frequency is written into both halves of the head.
fn fillInterleavedRope(theta: f32, seq_len: u32, head_dim: u32, cos: []f32, sin: []f32) void {
    const hd: usize = head_dim;
    const half = hd / 2;
    for (0..seq_len) |pos| {
        for (0..half) |f| {
            const ang = @as(f32, @floatFromInt(pos)) / std.math.pow(
                f32,
                theta,
                @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)),
            );
            const c = @cos(ang);
            const s = @sin(ang);
            cos[pos * hd + f] = c;
            cos[pos * hd + half + f] = c;
            sin[pos * hd + f] = s;
            sin[pos * hd + half + f] = s;
        }
    }
}

// =============================================================================
// Encoder
// =============================================================================

pub const Encoder = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    cfg: EncoderConfig,
    compiled: ?Compiled = null,

    const Compiled = struct {
        embed: zml.FnExe(EmbedTokens.forward),
        layer: zml.FnExe(TransformerLayer.forward),

        fn deinit(self: *Compiled) void {
            self.embed.deinit();
            self.layer.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Encoder {
        const cfg: EncoderConfig = .{};
        const lm = store.withPrefix("model.language_model");
        const layers = try allocator.alloc(TransformerLayer, @intCast(cfg.used_hidden_layers));
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| layer.* = .init(lm.withPrefix("layers").withLayer(i), cfg);
        return .{
            .embed_tokens = .{ .weight = lm.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *Encoder, allocator: std.mem.Allocator) void {
        if (self.compiled) |*c| c.deinit();
        allocator.free(self.layers);
    }

    /// Compile embed + one layer kernel (all 50 layers share the layer executable).
    pub fn compile(self: *Encoder, run: *const Run, text_len: u32) !void {
        var node = run.progress.start("Compiling MiniMax-H3 encoder", 2);
        defer node.end();
        const dt = self.embed_tokens.weight.dtype();
        const embed = try zml.FnExe(EmbedTokens.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_encoder_embed",
        }, .{.{
            .embedding = .{ .embed_tokens = self.embed_tokens },
            .tokens = .init(.{ .b = 1, .s = text_len }, .u32),
        }});
        errdefer embed.deinit();
        const layer = try zml.FnExe(TransformerLayer.forward).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_encoder_layer",
        }, .{.{
            .layer = self.layers[0],
            .hidden = .init(.{ .b = 1, .s = text_len, .d = self.cfg.hidden_size }, dt),
            .cos = .init(.{ .s = text_len, .hd = self.cfg.head_dim }, dt),
            .sin = .init(.{ .s = text_len, .hd = self.cfg.head_dim }, dt),
        }});
        self.compiled = .{ .embed = embed, .layer = layer };
    }

    /// Token ids → hidden states `[1, S, 5120]` for the DiT text refiner.
    pub fn encodeText(
        self: *const Encoder,
        run: *const Run,
        store: *zml.io.TensorStore,
        tokens: []const u32,
    ) !zml.Buffer {
        const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
        const seq_len: u32 = @intCast(tokens.len);
        const head_dim: u32 = @intCast(self.cfg.head_dim);
        var token_buf = try zml.Buffer.fromBytes(run.io, run.platform, .init(.{ .b = 1, .s = tokens.len }, .u32), .replicated, std.mem.sliceAsBytes(tokens));
        defer token_buf.deinit();

        const embed_part = EmbedTokens{ .embed_tokens = self.embed_tokens };
        var embed_bufs = try load(run, store, EmbedTokens, &embed_part, null);
        defer zml.Buffer.deinitAll(EmbedTokens, &embed_bufs);
        var embed_runner = try zml.FnExe(EmbedTokens.forward).Runner(.{.embedding}).init(&compiled.embed, run.allocator, .{ .embedding = embed_bufs });
        defer embed_runner.deinit(run.allocator);
        var hidden: zml.Buffer = undefined;
        embed_runner.run(run.io, .{ .inputs = .{ .tokens = token_buf }, .outputs = .{ .hidden = &hidden }, .opts = .{ .wait = true } });
        errdefer hidden.deinit();

        const cos = try run.allocator.alloc(f32, seq_len * head_dim);
        defer run.allocator.free(cos);
        const sin = try run.allocator.alloc(f32, seq_len * head_dim);
        defer run.allocator.free(sin);
        fillInterleavedRope(self.cfg.rope_theta, seq_len, head_dim, cos, sin);
        var cos_buf = try uploadF32(run, .init(.{ .s = seq_len, .hd = head_dim }, self.embed_tokens.weight.dtype()), cos);
        defer cos_buf.deinit();
        var sin_buf = try uploadF32(run, .init(.{ .s = seq_len, .hd = head_dim }, self.embed_tokens.weight.dtype()), sin);
        defer sin_buf.deinit();

        var loader: zml.io.Loader = try .init(run.allocator, run.platform, ops.loader_opts);
        defer loader.deinit();
        const LayerRunner = zml.FnExe(TransformerLayer.forward).Runner(.{.layer});
        for (0..self.layers.len) |layer_i| {
            var layer_bufs = try load(run, store, TransformerLayer, &self.layers[layer_i], &loader);
            defer zml.Buffer.deinitAll(TransformerLayer, &layer_bufs);
            var layer_runner = try LayerRunner.init(&compiled.layer, run.allocator, .{ .layer = layer_bufs });
            defer layer_runner.deinit(run.allocator);
            var next: zml.Buffer = undefined;
            // Host sync per layer.
            layer_runner.run(run.io, .{
                .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            hidden.deinit();
            hidden = next;
        }
        return hidden;
    }
};
