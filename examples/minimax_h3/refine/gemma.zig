const std = @import("std");

const zml = @import("zml");

const sku = @import("../recipe/sku.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3_stage2);

// =============================================================================
// refine/gemma.zig — Gemma 4 12B text encoder + DualLinear proj
//
// Compiled once and reused by every SKU. Global k_proj is replicated.
// =============================================================================

pub const default_path = "output/ltx-bf16/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors";
pub const hidden: i64 = 3840;
pub const layers_n: u32 = 48;
pub const pad_len: u32 = sku.prompt_tokens;
pub const heads: i64 = 16;
pub const slide_kv: i64 = 8;
pub const slide_hd: i64 = 256;
pub const global_kv: i64 = 1;
pub const global_hd: i64 = 512;
pub const vocab: i64 = 262144;
pub const rms_eps: f32 = 1e-6;
pub const embed_scale: f32 = 61.967733; // sqrt(3840)
pub const bos_id: u32 = 2;
pub const pad_id: u32 = 0;
/// Official `layer="all"` keeps the last 10 tokens of each hidden (and final-norm).
pub const keep_tokens: u32 = 10;
pub const stack_layers: u32 = layers_n + 1;
pub const video_dim: i64 = 4096;
pub const audio_dim: i64 = 2048;
pub const proj_dim: i64 = video_dim + audio_dim;
pub const default_tokenizer = "output/gemma4_tokenizer.json";

pub const weight_paths = [_][]const u8{
    default_path,
    "output/ltx-bf16/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "/var/models/super-accel/ltx-bf16/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "/var/models/super-accel/ltx/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    sku.hf_gemma,
};

const load = @import("load.zig");

const linear = weights.linear;
const rmsNorm = weights.rmsNorm;

const qkv_part = .{ .dout = .model, .d = .replicated };
const out_part = .{ .dout = .replicated, .d = .model };
const mlp_up_part = .{ .dout = .model, .d = .replicated };
const mlp_down_part = .{ .dout = .replicated, .d = .model };

fn isGlobal(index: u32) bool {
    return index % 6 == 5;
}

/// Official LTX Gemma4: `[BOS] + encode(prompt)`, then left-pad with 0.
pub fn padPromptTokens(dst: []u32, ids: []const u32) void {
    @memset(dst, pad_id);
    const concat_len = ids.len + 1;
    if (concat_len >= dst.len) {
        const skip = concat_len - dst.len;
        if (skip == 0) {
            dst[0] = bos_id;
            @memcpy(dst[1..], ids[0 .. dst.len - 1]);
        } else {
            @memcpy(dst, ids[skip - 1 ..][0..dst.len]);
        }
        return;
    }
    const start = dst.len - concat_len;
    dst[start] = bos_id;
    @memcpy(dst[start + 1 ..][0..ids.len], ids);
}

pub fn tokenizePrompt(allocator: std.mem.Allocator, io: std.Io, prompt: []const u8) ![]u32 {
    const path = load.firstExisting(io, &.{
        "output/ltx-bf16/text_encoders/tokenizer.json",
        default_tokenizer,
        "/var/models/super-accel/ltx/text_encoders/tokenizer.json",
        "/var/models/super-accel/ltx/text_encoders/gemma4_tokenizer.json",
        sku.hf_gemma_tokenizer,
    }) orelse return error.GemmaMissing;
    var tok = try zml.tokenizer.Tokenizer.fromFile(allocator, io, path);
    defer tok.deinit();
    var enc = try tok.encoder();
    defer enc.deinit();
    const ids = try enc.encodeAlloc(allocator, prompt);
    defer allocator.free(ids);
    const out = try allocator.alloc(u32, pad_len);
    padPromptTokens(out, ids);
    log.info("stage2 gemma tokens from {s} prompt_ids={d}", .{ path, ids.len });
    return out;
}

fn sdpaScale1(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    return zml.nn.sdpa(q, k, v, .{
        .attn_mask = mask,
        .scale = zml.Tensor.scalar(1.0, q.dtype()),
    });
}

const Mlp = struct {
    gate: zml.nn.Linear,
    up: zml.nn.Linear,
    down: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .gate = linear(store, "gate_proj.weight", null, mlp_up_part, .replicated),
            .up = linear(store, "up_proj.weight", null, mlp_up_part, .replicated),
            .down = linear(store, "down_proj.weight", null, mlp_down_part, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.nn.Linear.unloadBuffers(&self.gate);
        zml.nn.Linear.unloadBuffers(&self.up);
        zml.nn.Linear.unloadBuffers(&self.down);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const x_q = x.withPartitioning(.{ .d = .replicated });
        const g = self.gate.forward(x_q).gelu().rename(.{ .dout = .d });
        const u = self.up.forward(x_q).rename(.{ .dout = .d });
        return self.down.forward(g.mul(u)).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const Sliding = struct {
    input_ln: zml.nn.RmsNorm,
    post_attn: zml.nn.RmsNorm,
    pre_ff: zml.nn.RmsNorm,
    post_ff: zml.nn.RmsNorm,
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    o: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    mlp: Mlp,
    layer_scalar: zml.Tensor,

    pub const Input = struct {
        layer: Sliding,
        hidden: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
        mask: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) Sliding {
        const attn = store.withPrefix("self_attn");
        return .{
            .input_ln = rmsNorm(store.withPrefix("input_layernorm"), .{.d}, rms_eps),
            .post_attn = rmsNorm(store.withPrefix("post_attention_layernorm"), .{.d}, rms_eps),
            .pre_ff = rmsNorm(store.withPrefix("pre_feedforward_layernorm"), .{.d}, rms_eps),
            .post_ff = rmsNorm(store.withPrefix("post_feedforward_layernorm"), .{.d}, rms_eps),
            .q = linear(attn, "q_proj.weight", null, qkv_part, .replicated),
            .k = linear(attn, "k_proj.weight", null, qkv_part, .replicated),
            .v = linear(attn, "v_proj.weight", null, qkv_part, .replicated),
            .o = linear(attn, "o_proj.weight", null, out_part, .replicated),
            .q_norm = rmsNorm(attn.withPrefix("q_norm"), .{.hd}, rms_eps),
            .k_norm = rmsNorm(attn.withPrefix("k_norm"), .{.hd}, rms_eps),
            .mlp = .init(store.withPrefix("mlp")),
            .layer_scalar = store.createTensor("layer_scalar", .{.x}, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Sliding)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.input_ln);
        zml.nn.RmsNorm.unloadBuffers(&self.post_attn);
        zml.nn.RmsNorm.unloadBuffers(&self.pre_ff);
        zml.nn.RmsNorm.unloadBuffers(&self.post_ff);
        zml.nn.Linear.unloadBuffers(&self.q);
        zml.nn.Linear.unloadBuffers(&self.k);
        zml.nn.Linear.unloadBuffers(&self.v);
        zml.nn.Linear.unloadBuffers(&self.o);
        zml.nn.RmsNorm.unloadBuffers(&self.q_norm);
        zml.nn.RmsNorm.unloadBuffers(&self.k_norm);
        Mlp.unloadBuffers(&self.mlp);
        self.layer_scalar.deinit();
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual0 = input.hidden.withPartitioning(.{ .d = .replicated });
        const x = self.input_ln.forward(residual0);
        var q = self.q.forward(x).splitAxis(.dout, .{ .h = heads, .hd = slide_hd }).withPartitioning(.{ .h = .model });
        var k = self.k.forward(x).splitAxis(.dout, .{ .h = slide_kv, .hd = slide_hd }).withPartitioning(.{ .h = .model });
        const v_raw = self.v.forward(x).splitAxis(.dout, .{ .h = slide_kv, .hd = slide_hd }).withPartitioning(.{ .h = .model });
        const v = zml.nn.rmsNorm(v_raw, .hd, rms_eps);
        q = zml.nn.applyRotary(self.q_norm.forward(q), input.cos, input.sin);
        k = zml.nn.applyRotary(self.k_norm.forward(k), input.cos, input.sin);
        const attn = sdpaScale1(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), v.rename(.{ .s = .k }), input.mask)
            .rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        const a = self.post_attn.forward(self.o.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated }));
        const x1 = residual0.add(a);
        const m = self.post_ff.forward(self.mlp.forward(self.pre_ff.forward(x1)));
        const x2 = x1.add(m);
        const scale = self.layer_scalar.convert(x2.dtype()).withTags(.{.x}).squeeze(.x);
        return .{ .hidden = x2.mul(scale.broad(x2.shape())).reuseBuffer(input.hidden) };
    }
};

pub const Global = struct {
    input_ln: zml.nn.RmsNorm,
    post_attn: zml.nn.RmsNorm,
    pre_ff: zml.nn.RmsNorm,
    post_ff: zml.nn.RmsNorm,
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    o: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    mlp: Mlp,
    layer_scalar: zml.Tensor,

    pub const Input = struct {
        layer: Global,
        hidden: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
        mask: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) Global {
        const attn = store.withPrefix("self_attn");
        return .{
            .input_ln = rmsNorm(store.withPrefix("input_layernorm"), .{.d}, rms_eps),
            .post_attn = rmsNorm(store.withPrefix("post_attention_layernorm"), .{.d}, rms_eps),
            .pre_ff = rmsNorm(store.withPrefix("pre_feedforward_layernorm"), .{.d}, rms_eps),
            .post_ff = rmsNorm(store.withPrefix("post_feedforward_layernorm"), .{.d}, rms_eps),
            .q = linear(attn, "q_proj.weight", null, qkv_part, .replicated),
            .k = linear(attn, "k_proj.weight", null, .replicated, .replicated), // 1 KV head: cannot shard .h
            .o = linear(attn, "o_proj.weight", null, out_part, .replicated),
            .q_norm = rmsNorm(attn.withPrefix("q_norm"), .{.hd}, rms_eps),
            .k_norm = rmsNorm(attn.withPrefix("k_norm"), .{.hd}, rms_eps),
            .mlp = .init(store.withPrefix("mlp")),
            .layer_scalar = store.createTensor("layer_scalar", .{.x}, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Global)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.input_ln);
        zml.nn.RmsNorm.unloadBuffers(&self.post_attn);
        zml.nn.RmsNorm.unloadBuffers(&self.pre_ff);
        zml.nn.RmsNorm.unloadBuffers(&self.post_ff);
        zml.nn.Linear.unloadBuffers(&self.q);
        zml.nn.Linear.unloadBuffers(&self.k);
        zml.nn.Linear.unloadBuffers(&self.o);
        zml.nn.RmsNorm.unloadBuffers(&self.q_norm);
        zml.nn.RmsNorm.unloadBuffers(&self.k_norm);
        Mlp.unloadBuffers(&self.mlp);
        self.layer_scalar.deinit();
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual0 = input.hidden.withPartitioning(.{ .d = .replicated });
        const x = self.input_ln.forward(residual0);
        var q = self.q.forward(x).splitAxis(.dout, .{ .h = heads, .hd = global_hd }).withPartitioning(.{ .h = .model });
        const k_raw = self.k.forward(x).splitAxis(.dout, .{ .h = global_kv, .hd = global_hd });
        const v = zml.nn.rmsNorm(k_raw, .hd, rms_eps);
        var k = self.k_norm.forward(k_raw);
        q = zml.nn.applyRotary(self.q_norm.forward(q), input.cos, input.sin);
        k = zml.nn.applyRotary(k, input.cos, input.sin);
        const attn = sdpaScale1(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), v.rename(.{ .s = .k }), input.mask)
            .rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        const a = self.post_attn.forward(self.o.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated }));
        const x1 = residual0.add(a);
        const m = self.post_ff.forward(self.mlp.forward(self.pre_ff.forward(x1)));
        const x2 = x1.add(m);
        const scale = self.layer_scalar.convert(x2.dtype()).withTags(.{.x}).squeeze(.x);
        return .{ .hidden = x2.mul(scale.broad(x2.shape())).reuseBuffer(input.hidden) };
    }
};

pub const Embed = struct {
    weight: zml.Tensor,
    pub const Input = struct { model: Embed, tokens: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) Embed {
        return .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .replicated) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Embed)) void {
        self.weight.deinit();
    }

    pub fn forward(input: Input) Output {
        const emb = zml.nn.TokenEmbedding{ .weight = input.model.weight };
        const h = emb.forward(input.tokens.withPartialTags(.{.s})).withPartialTags(.{.d}).withPartitioning(.{ .d = .replicated });
        return .{ .hidden = h.scale(embed_scale) };
    }
};

pub const FinalNorm = struct {
    norm: zml.nn.RmsNorm,
    pub const Input = struct { model: FinalNorm, hidden: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) FinalNorm {
        return .{ .norm = rmsNorm(store.withPrefix("norm"), .{.d}, rms_eps) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FinalNorm)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(input: Input) Output {
        return .{ .hidden = input.model.norm.forward(input.hidden) };
    }
};

pub const DualLinear = struct {
    video: zml.nn.Linear,
    audio: zml.nn.Linear,

    pub const Input = struct { model: DualLinear, stack: zml.Tensor };
    pub const Output = struct { proj: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) DualLinear {
        const p = store.withPrefix("text_embedding_projection");
        return .{
            .video = linear(p, "video_aggregate_embed.weight", "video_aggregate_embed.bias", .replicated, .replicated),
            .audio = linear(p, "audio_aggregate_embed.weight", "audio_aggregate_embed.bias", .replicated, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DualLinear)) void {
        zml.nn.Linear.unloadBuffers(&self.video);
        zml.nn.Linear.unloadBuffers(&self.audio);
    }

    pub fn forward(input: Input) Output {
        // stack: [n, l=49, s, d=3840]  → movedim(1,-1) → [n,s,d,l]
        var x = input.stack.withPartialTags(.{ .n, .l, .s, .d }).transpose(.{ .n, .s, .d, .l });
        const xf = x.convert(.f32);
        const mean_sq = xf.mul(xf).mean(.d);
        x = xf.div(mean_sq.addConstant(1e-6).sqrt().broad(xf.shape()));
        x = x.merge(.{ .d = .{ .d, .l } });
        const src: f32 = @floatFromInt(hidden);
        const v = input.model.video.forward(x.scale(@sqrt(@as(f32, @floatFromInt(video_dim)) / src)).convert(input.model.video.weight.dtype())).rename(.{ .dout = .d });
        const a = input.model.audio.forward(x.scale(@sqrt(@as(f32, @floatFromInt(audio_dim)) / src)).convert(input.model.audio.weight.dtype())).rename(.{ .dout = .d });
        return .{ .proj = zml.Tensor.concatenate(&.{ v, a }, .d) };
    }
};

pub fn fillSlideRope(cos: []f32, sin: []f32, seq: u32) void {
    const hd: u32 = 256;
    const nfreq = hd / 2;
    std.debug.assert(cos.len == seq * hd);
    const th: f32 = 10_000.0;
    var s: u32 = 0;
    while (s < seq) : (s += 1) {
        var i: u32 = 0;
        while (i < nfreq) : (i += 1) {
            const freq = 1.0 / std.math.pow(f32, th, @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(hd)));
            const ang = @as(f32, @floatFromInt(s)) * freq;
            const c = @cos(ang);
            const si = @sin(ang);
            cos[s * hd + i] = c;
            cos[s * hd + nfreq + i] = c;
            sin[s * hd + i] = si;
            sin[s * hd + nfreq + i] = si;
        }
    }
}

pub fn fillGlobalRope(cos: []f32, sin: []f32, seq: u32) void {
    const hd: u32 = 512;
    const nfreq = hd / 2;
    const rope_pairs: u32 = 64;
    std.debug.assert(cos.len == seq * hd);
    const th: f32 = 1_000_000.0;
    var s: u32 = 0;
    while (s < seq) : (s += 1) {
        var i: u32 = 0;
        while (i < nfreq) : (i += 1) {
            const freq = if (i < rope_pairs)
                1.0 / std.math.pow(f32, th, @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(hd)))
            else
                0.0;
            const ang = @as(f32, @floatFromInt(s)) * freq;
            const c = @cos(ang);
            const si = @sin(ang);
            cos[s * hd + i] = c;
            cos[s * hd + nfreq + i] = c;
            sin[s * hd + i] = si;
            sin[s * hd + nfreq + i] = si;
        }
    }
}

pub fn fillMask(out: []f32, tokens: []const u32, seq: u32) void {
    std.debug.assert(out.len == seq * seq);
    const neg = -3.38953139e38; // bf16 -inf
    var q: u32 = 0;
    while (q < seq) : (q += 1) {
        var k: u32 = 0;
        while (k < seq) : (k += 1) {
            const blocked = (k > q) or (tokens[k] == 0);
            out[q * seq + k] = if (blocked) neg else 0;
        }
    }
}

pub fn padTokens(allocator: std.mem.Allocator, ids: []const u32) ![]u32 {
    const out = try allocator.alloc(u32, pad_len);
    @memset(out, 0);
    const n = @min(ids.len, pad_len);
    @memcpy(out[pad_len - n ..], ids[ids.len - n ..]);
    if (n != 0 and out[pad_len - n] != 2) {
        if (n < pad_len) {
            const start = pad_len - n - 1;
            @memset(out[start..], 0);
            out[start] = 2;
            @memcpy(out[start + 1 ..], ids[ids.len - n ..]);
        }
    }
    return out;
}

pub fn unpaddedCount(tokens: []const u32) u32 {
    var n: u32 = 0;
    var i: usize = tokens.len;
    while (i > 0) {
        i -= 1;
        if (tokens[i] == 0) break;
        n += 1;
    }
    return n;
}

pub fn modelView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    // Official HF/LTX Gemma4 uses `model.*`. Comfy-exported checkpoints use
    // `gemma3_12b.transformer.model.*`.
    if (store.hasKey("model.embed_tokens.weight")) return store.withPrefix("model");
    return store.withPrefix("gemma3_12b.transformer.model");
}

const KeepIn = struct { hidden: zml.Tensor };
const KeepOut = struct { snap: zml.Tensor };

/// Official `layer="all"` keeps the last `keep_tokens` of each hidden (and final-norm).
fn keepTail(input: KeepIn) KeepOut {
    const x = input.hidden.withPartialTags(.{ .n, .s, .d });
    return .{ .snap = x.slice(.s, .{ .start = pad_len - keep_tokens, .end = pad_len }).convert(.f32) };
}

pub const Compiled = struct {
    embed_exe: zml.FnExe(Embed.forward),
    slide_exe: zml.FnExe(Sliding.forward),
    global_exe: zml.FnExe(Global.forward),
    norm_exe: zml.FnExe(FinalNorm.forward),
    proj_exe: zml.FnExe(DualLinear.forward),
    keep_exe: zml.FnExe(keepTail),
    embed_bufs: zml.Bufferized(Embed),
    slide_bufs: [40]zml.Bufferized(Sliding) = undefined,
    global_bufs: [8]zml.Bufferized(Global) = undefined,
    norm_bufs: zml.Bufferized(FinalNorm),
    proj_bufs: zml.Bufferized(DualLinear),
    slide_n: u32 = 0,
    global_n: u32 = 0,

    pub fn deinit(self: *Compiled) void {
        Embed.unloadBuffers(&self.embed_bufs);
        var i: u32 = 0;
        while (i < self.slide_n) : (i += 1) Sliding.unloadBuffers(&self.slide_bufs[i]);
        i = 0;
        while (i < self.global_n) : (i += 1) Global.unloadBuffers(&self.global_bufs[i]);
        FinalNorm.unloadBuffers(&self.norm_bufs);
        DualLinear.unloadBuffers(&self.proj_bufs);
        self.embed_exe.deinit();
        self.slide_exe.deinit();
        self.global_exe.deinit();
        self.norm_exe.deinit();
        self.proj_exe.deinit();
        self.keep_exe.deinit();
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
) !*Compiled {
    progress.increaseEstimatedTotalItems(5);
    const out = try allocator.create(Compiled);
    errdefer allocator.destroy(out);
    out.slide_n = 0;
    out.global_n = 0;

    const root = modelView(store.view());
    const embed_m = Embed.init(root);
    const slide_m = Sliding.init(root.withPrefix("layers").withLayer(0));
    const glob_m = Global.init(root.withPrefix("layers").withLayer(5));
    const norm_m = FinalNorm.init(root);
    const proj_m = DualLinear.init(store.view());

    const hidden_sh: zml.Tensor = .init(.{ .n = 1, .s = pad_len, .d = hidden }, .bf16);
    const mask_sh: zml.Tensor = .init(.{ .q = pad_len, .k = pad_len }, .bf16);
    const slide_cos: zml.Tensor = .init(.{ .s = pad_len, .hd = slide_hd }, .bf16);
    const slide_sin: zml.Tensor = .init(.{ .s = pad_len, .hd = slide_hd }, .bf16);
    const glob_cos: zml.Tensor = .init(.{ .s = pad_len, .hd = global_hd }, .bf16);
    const glob_sin: zml.Tensor = .init(.{ .s = pad_len, .hd = global_hd }, .bf16);

    out.embed_exe = try zml.FnExe(Embed.forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_gemma_embed",
    }, .{.{ .model = embed_m, .tokens = .init(.{ .n = 1, .s = pad_len }, .u32) }});
    out.slide_exe = try zml.FnExe(Sliding.forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_gemma_slide",
    }, .{.{ .layer = slide_m, .hidden = hidden_sh, .cos = slide_cos, .sin = slide_sin, .mask = mask_sh }});
    out.global_exe = try zml.FnExe(Global.forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_gemma_global",
    }, .{.{ .layer = glob_m, .hidden = .init(.{ .n = 1, .s = pad_len, .d = hidden }, .bf16), .cos = glob_cos, .sin = glob_sin, .mask = .init(.{ .q = pad_len, .k = pad_len }, .bf16) }});
    out.norm_exe = try zml.FnExe(FinalNorm.forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_gemma_norm",
    }, .{.{ .model = norm_m, .hidden = hidden_sh }});
    out.proj_exe = try zml.FnExe(DualLinear.forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_gemma_proj",
    }, .{.{ .model = proj_m, .stack = .init(.{ .n = 1, .l = stack_layers, .s = keep_tokens, .d = hidden }, .f32) }});
    out.keep_exe = try zml.FnExe(keepTail).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_gemma_keep",
    }, .{.{ .hidden = hidden_sh }});
    out.embed_bufs = try weights.load(allocator, io, platform, store, shardings, Embed, &embed_m, progress, null);
    out.norm_bufs = try weights.load(allocator, io, platform, store, shardings, FinalNorm, &norm_m, progress, null);
    out.proj_bufs = try weights.load(allocator, io, platform, store, shardings, DualLinear, &proj_m, progress, null);
    errdefer out.deinit();

    var li: u32 = 0;
    while (li < layers_n) : (li += 1) {
        const layer_store = root.withPrefix("layers").withLayer(li);
        if (isGlobal(li)) {
            const m = Global.init(layer_store);
            out.global_bufs[out.global_n] = try weights.load(allocator, io, platform, store, shardings, Global, &m, progress, null);
            out.global_n += 1;
        } else {
            const m = Sliding.init(layer_store);
            out.slide_bufs[out.slide_n] = try weights.load(allocator, io, platform, store, shardings, Sliding, &m, progress, null);
            out.slide_n += 1;
        }
    }
    log.info("compile Gemma4-12B layers={d} slide={d} global={d}", .{ layers_n, out.slide_n, out.global_n });
    return out;
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *Compiled,
    tokens: []const u32,
) ![]f32 {
    var tok_buf = try weights.fromItems(io, platform, .init(.{ .n = 1, .s = pad_len }, .u32), tokens);
    defer tok_buf.deinit();

    const cos_s = try allocator.alloc(f32, pad_len * @as(usize, @intCast(slide_hd)));
    defer allocator.free(cos_s);
    const sin_s = try allocator.alloc(f32, cos_s.len);
    defer allocator.free(sin_s);
    fillSlideRope(cos_s, sin_s, pad_len);
    const cos_g = try allocator.alloc(f32, pad_len * @as(usize, @intCast(global_hd)));
    defer allocator.free(cos_g);
    const sin_g = try allocator.alloc(f32, cos_g.len);
    defer allocator.free(sin_g);
    fillGlobalRope(cos_g, sin_g, pad_len);
    const mask = try allocator.alloc(f32, pad_len * pad_len);
    defer allocator.free(mask);
    fillMask(mask, tokens, pad_len);

    var cos_s_b = try weights.fromF32(allocator, io, platform, .init(.{ .s = pad_len, .hd = slide_hd }, .bf16), cos_s);
    defer cos_s_b.deinit();
    var sin_s_b = try weights.fromF32(allocator, io, platform, .init(.{ .s = pad_len, .hd = slide_hd }, .bf16), sin_s);
    defer sin_s_b.deinit();
    var cos_g_b = try weights.fromF32(allocator, io, platform, .init(.{ .s = pad_len, .hd = global_hd }, .bf16), cos_g);
    defer cos_g_b.deinit();
    var sin_g_b = try weights.fromF32(allocator, io, platform, .init(.{ .s = pad_len, .hd = global_hd }, .bf16), sin_g);
    defer sin_g_b.deinit();
    var mask_b = try weights.fromF32(allocator, io, platform, .init(.{ .q = pad_len, .k = pad_len }, .bf16), mask);
    defer mask_b.deinit();

    var embed_run = try zml.FnExe(Embed.forward).Runner(.{.model}).init(&compiled.embed_exe, allocator, .{ .model = compiled.embed_bufs });
    defer embed_run.deinit(allocator);
    var keep_run = try zml.FnExe(keepTail).Runner(.{}).init(&compiled.keep_exe, allocator, .{});
    defer keep_run.deinit(allocator);
    const SlideRunner = zml.FnExe(Sliding.forward).Runner(.{.layer});
    const GlobalRunner = zml.FnExe(Global.forward).Runner(.{.layer});
    var slide_run: ?SlideRunner = null;
    defer if (slide_run) |*r| r.deinit(allocator);
    var glob_run: ?GlobalRunner = null;
    defer if (glob_run) |*r| r.deinit(allocator);

    var snaps: [stack_layers]zml.Buffer = undefined;
    var n_snap: u32 = 0;
    defer {
        var si: u32 = 0;
        while (si < n_snap) : (si += 1) snaps[si].deinit();
    }

    var state: zml.Buffer = undefined;
    embed_run.run(io, .{ .inputs = .{ .tokens = tok_buf }, .outputs = .{ .hidden = &state }, .opts = .{ .wait = false } });

    var slide_i: u32 = 0;
    var glob_i: u32 = 0;
    var li: u32 = 0;
    while (li < layers_n) : (li += 1) {
        keep_run.run(io, .{
            .inputs = .{ .hidden = state },
            .outputs = .{ .snap = &snaps[li] },
            .opts = .{ .wait = false },
        });
        n_snap += 1;
        var next: zml.Buffer = undefined;
        if (isGlobal(li)) {
            if (glob_run) |*r| {
                r.rebake(.{ .layer = compiled.global_bufs[glob_i] });
            } else {
                glob_run = try GlobalRunner.init(&compiled.global_exe, allocator, .{ .layer = compiled.global_bufs[glob_i] });
            }
            glob_run.?.run(io, .{
                .inputs = .{ .hidden = state, .cos = cos_g_b, .sin = sin_g_b, .mask = mask_b },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            glob_i += 1;
        } else {
            if (slide_run) |*r| {
                r.rebake(.{ .layer = compiled.slide_bufs[slide_i] });
            } else {
                slide_run = try SlideRunner.init(&compiled.slide_exe, allocator, .{ .layer = compiled.slide_bufs[slide_i] });
            }
            slide_run.?.run(io, .{
                .inputs = .{ .hidden = state, .cos = cos_s_b, .sin = sin_s_b, .mask = mask_b },
                .outputs = .{ .hidden = &next },
                .opts = .{ .wait = true },
            });
            slide_i += 1;
        }
        state.deinit();
        state = next;
    }
    var norm_run = try zml.FnExe(FinalNorm.forward).Runner(.{.model}).init(&compiled.norm_exe, allocator, .{ .model = compiled.norm_bufs });
    defer norm_run.deinit(allocator);
    var normed: zml.Buffer = undefined;
    norm_run.run(io, .{ .inputs = .{ .hidden = state }, .outputs = .{ .hidden = &normed }, .opts = .{ .wait = false } });
    state.deinit();
    keep_run.run(io, .{
        .inputs = .{ .hidden = normed },
        .outputs = .{ .snap = &snaps[layers_n] },
        .opts = .{ .wait = true },
    });
    n_snap += 1;
    normed.deinit();

    const row = keep_tokens * @as(usize, @intCast(hidden));
    const stack = try allocator.alloc(f32, stack_layers * row);
    errdefer allocator.free(stack);
    var si: u32 = 0;
    while (si < stack_layers) : (si += 1) {
        try snaps[si].toSlice(io, .init(snaps[si].shape(), std.mem.sliceAsBytes(stack[si * row ..][0..row])));
    }

    var stack_b = try weights.fromF32(allocator, io, platform, .init(.{ .n = 1, .l = stack_layers, .s = keep_tokens, .d = hidden }, .f32), stack);
    defer stack_b.deinit();
    allocator.free(stack);
    var proj_run = try zml.FnExe(DualLinear.forward).Runner(.{.model}).init(&compiled.proj_exe, allocator, .{ .model = compiled.proj_bufs });
    defer proj_run.deinit(allocator);
    var proj: zml.Buffer = undefined;
    proj_run.run(io, .{ .inputs = .{ .stack = stack_b }, .outputs = .{ .proj = &proj }, .opts = .{ .wait = true } });
    defer proj.deinit();

    const slice = try proj.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const n = proj.shape().count();
    const out = try allocator.alloc(f32, n);
    const raw = slice.data();
    switch (proj.shape().dtype()) {
        .f32 => @memcpy(out, std.mem.bytesAsSlice(f32, raw[0 .. n * 4])),
        .bf16 => {
            const src = std.mem.bytesAsSlice(u16, raw[0 .. n * 2]);
            for (src, out) |b, *d| d.* = @as(f32, @bitCast(@as(u32, b) << 16));
        },
        else => return error.UnsupportedProjDtype,
    }
    return out;
}
