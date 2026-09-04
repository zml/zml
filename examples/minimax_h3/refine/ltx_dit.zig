const std = @import("std");

const zml = @import("zml");

const lora = @import("../recipe/lora.zig");
const load = @import("load.zig");
const sku = @import("../recipe/sku.zig");
const sol_attn = @import("sol_attn.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3_stage2);

// =============================================================================
// refine/ltx_dit.zig — LTX-2.5 22B DiT
//
// Block weights are shared across SKUs. Embed/finish graphs are per geometry.
// =============================================================================

pub const default_path = "output/ltx-bf16/ltx-2.5-22b-dev-transformer-bf16.safetensors";
pub const fused_path = "output/ltx-bf16/ltx-2.5-22b-dev-transformer-bf16-lora08-convrot.safetensors";
pub const comfy_path = "/var/models/super-accel/ltx/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors";
pub const weight_paths = [_][]const u8{
    fused_path,
    default_path,
    comfy_path,
    sku.hf_ltx_dit,
};
pub const default_lora = sku.hf_ltx_lora;
pub const lora_paths = [_][]const u8{
    "/var/models/super-accel/ltx/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "output/ltx-bf16/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    default_lora,
};
pub const hidden: i64 = 4096;
pub const heads: i64 = 32;
pub const head_dim: i64 = 128;
pub const layers_n: u32 = 48;
pub const in_ch: i64 = 128;

const linear = weights.linear;

const qkv_part = .{ .dout = .model, .d = .replicated };
const out_part = .{ .dout = .replicated, .d = .model };

fn attnOut(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, backend: zml.attention.Backend) zml.Tensor {
    var qq = q.rename(.{ .s = .q });
    var kk = k.rename(.{ .s = .k });
    var vv = v.rename(.{ .s = .k });
    if (qq.shape().hasTag(.n) != null) {
        qq = qq.rename(.{ .n = .b });
        kk = kk.rename(.{ .n = .b });
        vv = vv.rename(.{ .n = .b });
    }
    const out = zml.attention.dense(qq, kk, vv, backend, .{ .is_causal = false });
    return if (out.shape().hasTag(.b) != null)
        out.rename(.{ .b = .n, .q = .s })
    else
        out.rename(.{ .q = .s });
}

fn taggedQkv(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor } {
    var qq = q.rename(.{ .s = .q });
    var kk = k.rename(.{ .s = .k });
    var vv = v.rename(.{ .s = .k });
    if (qq.shape().hasTag(.n) != null) {
        qq = qq.rename(.{ .n = .b });
        kk = kk.rename(.{ .n = .b });
        vv = vv.rename(.{ .n = .b });
    }
    return .{ .q = qq, .k = kk, .v = vv };
}

fn solOut(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, tau: zml.Tensor) zml.Tensor {
    const t = taggedQkv(q, k, v);
    const out = sol_attn.forward(t.q, t.k, t.v, tau);
    return if (out.shape().hasTag(.b) != null)
        out.rename(.{ .b = .n, .q = .s })
    else
        out.rename(.{ .q = .s });
}

fn rms(x: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .d, 1e-6);
}

fn rmsAdaln(x: zml.Tensor, scale: zml.Tensor, shift: zml.Tensor) zml.Tensor {
    const n = rms(x);
    const one = zml.Tensor.scalar(1.0, n.dtype());
    return n.mul(one.add(scale.convert(n.dtype()).broad(n.shape()))).add(shift.convert(n.dtype()).broad(n.shape()));
}

fn layerNormNoAffine(x: zml.Tensor) zml.Tensor {
    return zml.nn.normalizeVariance(x, 1e-6);
}

const Attn = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    o: zml.nn.Linear,
    gate: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    attn_backend: zml.attention.Backend = .vanilla,
    sol_self: bool = false,

    pub fn init(store: zml.io.TensorStore.View) Attn {
        return .{
            .q = linear(store, "to_q.weight", "to_q.bias", qkv_part, .replicated),
            .k = linear(store, "to_k.weight", "to_k.bias", qkv_part, .replicated),
            .v = linear(store, "to_v.weight", "to_v.bias", qkv_part, .replicated),
            .o = linear(store, "to_out.0.weight", "to_out.0.bias", out_part, .replicated),
            .gate = linear(store, "to_gate_logits.weight", "to_gate_logits.bias", qkv_part, .replicated),
            .q_norm = weights.rmsNorm(store.withPrefix("q_norm"), .{.d}, 1e-5),
            .k_norm = weights.rmsNorm(store.withPrefix("k_norm"), .{.d}, 1e-5),
            .attn_backend = .vanilla,
            .sol_self = false,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attn)) void {
        zml.nn.Linear.unloadBuffers(&self.q);
        zml.nn.Linear.unloadBuffers(&self.k);
        zml.nn.Linear.unloadBuffers(&self.v);
        zml.nn.Linear.unloadBuffers(&self.o);
        zml.nn.Linear.unloadBuffers(&self.gate);
        zml.nn.RmsNorm.unloadBuffers(&self.q_norm);
        zml.nn.RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn mergeLora(self: *zml.Bufferized(Attn), bundle: *const lora.Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, prefix: []const u8) !void {
        var buf: [128]u8 = undefined;
        try bundle.mergeLtx(allocator, io, platform, &self.q.weight, try std.fmt.bufPrint(&buf, "{s}.to_q", .{prefix}));
        try bundle.mergeLtx(allocator, io, platform, &self.k.weight, try std.fmt.bufPrint(&buf, "{s}.to_k", .{prefix}));
        try bundle.mergeLtx(allocator, io, platform, &self.v.weight, try std.fmt.bufPrint(&buf, "{s}.to_v", .{prefix}));
        try bundle.mergeLtx(allocator, io, platform, &self.o.weight, try std.fmt.bufPrint(&buf, "{s}.to_out.0", .{prefix}));
        try bundle.mergeLtx(allocator, io, platform, &self.gate.weight, try std.fmt.bufPrint(&buf, "{s}.to_gate_logits", .{prefix}));
    }

    pub fn forward(self: Attn, x: zml.Tensor, ctx: zml.Tensor, cos: ?zml.Tensor, sin: ?zml.Tensor, tau: zml.Tensor) zml.Tensor {
        const x_q = x.withPartitioning(.{ .d = .replicated });
        const ctx_q = ctx.withPartitioning(.{ .d = .replicated });
        var q = self.q_norm.forward(self.q.forward(x_q).rename(.{ .dout = .d }));
        var k = self.k_norm.forward(self.k.forward(ctx_q).rename(.{ .dout = .d }));
        const v = self.v.forward(ctx_q).rename(.{ .dout = .d });
        q = q.splitAxis(.d, .{ .h = heads, .hd = head_dim }).withPartitioning(.{ .h = .model });
        k = k.splitAxis(.d, .{ .h = heads, .hd = head_dim }).withPartitioning(.{ .h = .model });
        const vv = v.splitAxis(.d, .{ .h = heads, .hd = head_dim }).withPartitioning(.{ .h = .model });
        var out: zml.Tensor = undefined;
        if (cos) |c| {
            q = zml.nn.applyRotary(q, c, sin.?);
            k = zml.nn.applyRotary(k, c, sin.?);
        }
        // Official Super: self-attn layer 0 dense; layers 1–47 Sol-Attn. Cross-attn stays dense.
        if (self.sol_self) {
            out = solOut(q, k, vv, tau);
        } else {
            out = attnOut(q, k, vv, self.attn_backend);
        }
        const gates = self.gate.forward(x_q).sigmoid().scale(2).rename(.{ .dout = .h }).broad(out.shape());
        out = out.mul(gates).merge(.{ .d = .{ .h, .hd } });
        return self.o.forward(out).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

const Ff = struct {
    inn: zml.nn.Linear,
    out: zml.nn.Linear,
    pub fn init(store: zml.io.TensorStore.View) Ff {
        return .{
            .inn = linear(store, "net.0.proj.weight", null, qkv_part, .replicated),
            .out = linear(store, "net.2.weight", null, out_part, .replicated),
        };
    }
    pub fn unloadBuffers(self: *zml.Bufferized(Ff)) void {
        zml.nn.Linear.unloadBuffers(&self.inn);
        zml.nn.Linear.unloadBuffers(&self.out);
    }
    pub fn mergeLora(self: *zml.Bufferized(Ff), bundle: *const lora.Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, prefix: []const u8) !void {
        var buf: [128]u8 = undefined;
        try bundle.mergeLtx(allocator, io, platform, &self.inn.weight, try std.fmt.bufPrint(&buf, "{s}.net.0.proj", .{prefix}));
        try bundle.mergeLtx(allocator, io, platform, &self.out.weight, try std.fmt.bufPrint(&buf, "{s}.net.2", .{prefix}));
    }
    pub fn forward(self: Ff, x: zml.Tensor) zml.Tensor {
        const x_q = x.withPartitioning(.{ .d = .replicated });
        return self.out.forward(self.inn.forward(x_q).gelu().rename(.{ .dout = .d }))
            .rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const Block = struct {
    attn1: Attn,
    attn2: Attn,
    ff: Ff,
    table: zml.Tensor,
    prompt_table: zml.Tensor,

    pub const Input = struct {
        layer: Block,
        hidden: zml.Tensor,
        context: zml.Tensor,
        ada: zml.Tensor,
        prompt_ada: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
        tau: zml.Tensor,
    };
    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View) Block {
        return .{
            .attn1 = .init(store.withPrefix("attn1")),
            .attn2 = .init(store.withPrefix("attn2")),
            .ff = .init(store.withPrefix("ff")),
            .table = store.createTensor("scale_shift_table", .{ .k, .d }, .replicated),
            .prompt_table = store.createTensor("prompt_scale_shift_table", .{ .k, .d }, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Block)) void {
        Attn.unloadBuffers(&self.attn1);
        Attn.unloadBuffers(&self.attn2);
        Ff.unloadBuffers(&self.ff);
        self.table.deinit();
        self.prompt_table.deinit();
    }

    pub fn mergeLora(self: *zml.Bufferized(Block), bundle: *const lora.Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, index: u32) !void {
        var p: [64]u8 = undefined;
        const a1 = try std.fmt.bufPrint(&p, "transformer_blocks.{d}.attn1", .{index});
        try Attn.mergeLora(&self.attn1, bundle, allocator, io, platform, a1);
        const a2 = try std.fmt.bufPrint(&p, "transformer_blocks.{d}.attn2", .{index});
        try Attn.mergeLora(&self.attn2, bundle, allocator, io, platform, a2);
        const ff = try std.fmt.bufPrint(&p, "transformer_blocks.{d}.ff", .{index});
        try Ff.mergeLora(&self.ff, bundle, allocator, io, platform, ff);
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const hidden_in = input.hidden.withPartialTags(.{ .n, .s, .d }).withPartitioning(.{ .d = .replicated });
        const ada = input.ada.convert(hidden_in.dtype()).withPartialTags(.{ .n, .t, .d });
        const table = self.table.convert(hidden_in.dtype()).withPartialTags(.{ .k, .d });
        const ada_k = ada.splitAxis(.d, .{ .k = 9, .d = hidden });
        const mods = table.reshape(.{ .n = 1, .t = 1, .k = 9, .d = hidden }).add(ada_k);
        const parts = mods.chunkExact(.k, 9);
        const shift_msa = parts[0].squeeze(.k).squeeze(.t);
        const scale_msa = parts[1].squeeze(.k).squeeze(.t);
        const gate_msa = parts[2].squeeze(.k).squeeze(.t);
        const shift_mlp = parts[3].squeeze(.k).squeeze(.t);
        const scale_mlp = parts[4].squeeze(.k).squeeze(.t);
        const gate_mlp = parts[5].squeeze(.k).squeeze(.t);
        const shift_q = parts[6].squeeze(.k).squeeze(.t);
        const scale_q = parts[7].squeeze(.k).squeeze(.t);
        const gate_q = parts[8].squeeze(.k).squeeze(.t);

        const attn1_raw = self.attn1.forward(rmsAdaln(hidden_in, scale_msa, shift_msa), rmsAdaln(hidden_in, scale_msa, shift_msa), input.cos, input.sin, input.tau);
        const attn1 = attn1_raw.mul(gate_msa.broad(attn1_raw.shape()));
        const x1 = hidden_in.add(attn1);

        const ptab = self.prompt_table.convert(hidden_in.dtype()).withPartialTags(.{ .k, .d });
        const pada = input.prompt_ada.convert(hidden_in.dtype()).withPartialTags(.{ .n, .t, .d });
        const pmods = ptab.reshape(.{ .n = 1, .t = 1, .k = 2, .d = hidden }).add(pada.splitAxis(.d, .{ .k = 2, .d = hidden }));
        const pparts = pmods.chunkExact(.k, 2);
        const shift_kv = pparts[0].squeeze(.k).squeeze(.t);
        const scale_kv = pparts[1].squeeze(.k).squeeze(.t);
        const ctx = input.context.convert(hidden_in.dtype()).withPartialTags(.{ .n, .s, .d }).withPartitioning(.{ .d = .replicated });
        const one = zml.Tensor.scalar(1.0, ctx.dtype());
        const ctx_m = ctx.mul(one.add(scale_kv.broad(ctx.shape()))).add(shift_kv.broad(ctx.shape()));
        // Cross-attn: Q from x (tag .s tokens), K/V from context (also .s). Rename context seq.
        const ctx_k = ctx_m.rename(.{ .s = .c });
        const q_in = rmsAdaln(x1, scale_q, shift_q);
        const attn2 = self.attn2.forward(q_in, ctx_k.rename(.{ .c = .s }), null, null, input.tau).mul(gate_q.broad(q_in.shape()));
        const x2 = x1.add(attn2);

        const ff_raw = self.ff.forward(rmsAdaln(x2, scale_mlp, shift_mlp));
        const ff = ff_raw.mul(gate_mlp.broad(ff_raw.shape()));
        const x3 = x2.add(ff);
        return .{ .hidden = x3.reuseBuffer(input.hidden) };
    }
};

pub const Embed = struct {
    patch: zml.nn.Linear,
    time1: zml.nn.Linear,
    time2: zml.nn.Linear,
    ada: zml.nn.Linear,
    prompt1: zml.nn.Linear,
    prompt2: zml.nn.Linear,
    prompt_ada: zml.nn.Linear,
    keyframes: zml.Tensor,
    tokens: i64,
    time: i64,
    height: i64,
    width: i64,

    pub const Input = struct {
        model: Embed,
        latent: zml.Tensor,
        timestep: zml.Tensor,
    };
    pub const Output = struct {
        hidden: zml.Tensor,
        ada: zml.Tensor,
        prompt_ada: zml.Tensor,
        embedded: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View, tokens: i64, time: i64, height: i64, width: i64) Embed {
        const te = store.withPrefix("adaln_single.emb.timestep_embedder");
        const pe = store.withPrefix("prompt_adaln_single.emb.timestep_embedder");
        return .{
            .patch = linear(store, "patchify_proj.weight", "patchify_proj.bias", .replicated, .replicated),
            .time1 = linear(te, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .time2 = linear(te, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
            .ada = linear(store.withPrefix("adaln_single"), "linear.weight", "linear.bias", .replicated, .replicated),
            .prompt1 = linear(pe, "linear_1.weight", "linear_1.bias", .replicated, .replicated),
            .prompt2 = linear(pe, "linear_2.weight", "linear_2.bias", .replicated, .replicated),
            .prompt_ada = linear(store.withPrefix("prompt_adaln_single"), "linear.weight", "linear.bias", .replicated, .replicated),
            .keyframes = store.createTensor("keyframes_abs_pos_embedding", .{ .n, .d }, .replicated),
            .tokens = tokens,
            .time = time,
            .height = height,
            .width = width,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Embed)) void {
        zml.nn.Linear.unloadBuffers(&self.patch);
        zml.nn.Linear.unloadBuffers(&self.time1);
        zml.nn.Linear.unloadBuffers(&self.time2);
        zml.nn.Linear.unloadBuffers(&self.ada);
        zml.nn.Linear.unloadBuffers(&self.prompt1);
        zml.nn.Linear.unloadBuffers(&self.prompt2);
        zml.nn.Linear.unloadBuffers(&self.prompt_ada);
        self.keyframes.deinit();
    }

    pub fn mergeLora(self: *zml.Bufferized(Embed), bundle: *const lora.Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        try bundle.mergeLtx(allocator, io, platform, &self.patch.weight, "patchify_proj");
        try bundle.mergeLtx(allocator, io, platform, &self.time1.weight, "adaln_single.emb.timestep_embedder.linear_1");
        try bundle.mergeLtx(allocator, io, platform, &self.time2.weight, "adaln_single.emb.timestep_embedder.linear_2");
        try bundle.mergeLtx(allocator, io, platform, &self.ada.weight, "adaln_single.linear");
        try bundle.mergeLtx(allocator, io, platform, &self.prompt1.weight, "prompt_adaln_single.emb.timestep_embedder.linear_1");
        try bundle.mergeLtx(allocator, io, platform, &self.prompt2.weight, "prompt_adaln_single.emb.timestep_embedder.linear_2");
        try bundle.mergeLtx(allocator, io, platform, &self.prompt_ada.weight, "prompt_adaln_single.linear");
    }

    pub fn forward(input: Input) Output {
        const self = input.model;
        var x = input.latent.withTags(.{ .n, .c, .t, .h, .w }).transpose(.{ .n, .t, .h, .w, .c });
        x = x.reshape(.{ .n = 1, .s = self.tokens, .d = in_ch }).convert(self.patch.weight.dtype());
        var hidden_t = self.patch.forward(x).rename(.{ .dout = .d });
        const per_frame = self.height * self.width;
        const first = hidden_t.slice(.s, .{ .start = 0, .end = per_frame });
        const kf = self.keyframes.convert(hidden_t.dtype()).withPartialTags(.{ .n, .d }).squeeze(.n).broad(first.shape());
        const marked = first.add(kf);
        const rest = hidden_t.slice(.s, .{ .start = per_frame, .end = self.tokens });
        hidden_t = zml.Tensor.concatenate(&.{ marked, rest }, .s);

        const ts = input.timestep.convert(.f32).withPartialTags(.{.n}).scale(1000);
        const inv = zml.nn.invFreq(256, .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = 10_000.0 } },
        }).withTags(.{.f});
        const angles = ts.outer(inv);
        const feat = zml.Tensor.concatenate(&.{ angles.cos(), angles.sin() }, .f).rename(.{ .f = .d }).convert(self.time1.weight.dtype());
        const embedded = self.time2.forward(self.time1.forward(feat).silu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        const ada = self.ada.forward(embedded.silu()).rename(.{ .dout = .d });
        const pemb = self.prompt2.forward(self.prompt1.forward(feat).silu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        const pada = self.prompt_ada.forward(pemb.silu()).rename(.{ .dout = .d });
        return .{
            .hidden = hidden_t,
            .ada = ada.reshape(.{ .n = 1, .t = 1, .d = 9 * hidden }),
            .prompt_ada = pada.reshape(.{ .n = 1, .t = 1, .d = 2 * hidden }),
            .embedded = embedded.reshape(.{ .n = 1, .t = 1, .d = hidden }),
        };
    }
};

pub const Finish = struct {
    table: zml.Tensor,
    proj: zml.nn.Linear,
    tokens: i64,
    time: i64,
    height: i64,
    width: i64,

    pub const Input = struct {
        model: Finish,
        hidden: zml.Tensor,
        embedded: zml.Tensor,
    };
    pub const Output = struct { latent: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, tokens: i64, time: i64, height: i64, width: i64) Finish {
        return .{
            .table = store.createTensor("scale_shift_table", .{ .k, .d }, .replicated),
            .proj = linear(store, "proj_out.weight", "proj_out.bias", .replicated, .replicated),
            .tokens = tokens,
            .time = time,
            .height = height,
            .width = width,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Finish)) void {
        self.table.deinit();
        zml.nn.Linear.unloadBuffers(&self.proj);
    }

    pub fn mergeLora(self: *zml.Bufferized(Finish), bundle: *const lora.Bundle, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
        try bundle.mergeLtx(allocator, io, platform, &self.proj.weight, "proj_out");
    }

    pub fn forward(input: Input) Output {
        const self = input.model;
        var x = input.hidden.withPartialTags(.{ .n, .s, .d });
        const emb = input.embedded.convert(x.dtype()).withPartialTags(.{ .n, .t, .d });
        const table = self.table.convert(x.dtype()).withPartialTags(.{ .k, .d });
        const table_b = table.reshape(.{ .n = 1, .t = 1, .k = 2, .d = hidden });
        const emb_b = emb.reshape(.{ .n = 1, .t = 1, .k = 1, .d = hidden }).broad(table_b.shape());
        const mods = table_b.add(emb_b);
        const parts = mods.chunkExact(.k, 2);
        const shift = parts[0].squeeze(.k).squeeze(.t);
        const scale = parts[1].squeeze(.k).squeeze(.t);
        x = layerNormNoAffine(x);
        const one = zml.Tensor.scalar(1.0, x.dtype());
        x = x.mul(one.add(scale.broad(x.shape()))).add(shift.broad(x.shape()));
        x = self.proj.forward(x).rename(.{ .dout = .c });
        x = x.reshape(.{ .n = 1, .t = self.time, .h = self.height, .w = self.width, .c = in_ch });
        return .{ .latent = x.transpose(.{ .n, .c, .t, .h, .w }).convert(.f32) };
    }
};

pub fn tokenCount(t: u32, h: u32, w: u32) u32 {
    return t * h * w;
}

pub const Compiled = struct {
    embed: zml.FnExe(Embed.forward),
    block: zml.FnExe(Block.forward),
    block_sol: ?zml.FnExe(Block.forward) = null,
    finish: zml.FnExe(Finish.forward),
    embed_bufs: zml.Bufferized(Embed),
    finish_bufs: zml.Bufferized(Finish),
    blocks: []zml.Bufferized(Block),
    n: u32 = 0,
    tokens: u32,
    time: u32,
    height: u32,
    width: u32,
    allocator: std.mem.Allocator,
    owns_weights: bool = true,

    pub fn deinit(self: *Compiled) void {
        self.embed.deinit();
        self.block.deinit();
        if (self.block_sol) |*exe| exe.deinit();
        self.finish.deinit();
        if (!self.owns_weights) return;
        Embed.unloadBuffers(&self.embed_bufs);
        Finish.unloadBuffers(&self.finish_bufs);
        var i: u32 = 0;
        while (i < self.n) : (i += 1) Block.unloadBuffers(&self.blocks[i]);
        self.allocator.free(self.blocks);
    }

    pub fn blockExe(self: *Compiled, layer_i: u32) *zml.FnExe(Block.forward) {
        if (layer_i != 0) {
            if (self.block_sol) |*sol| return sol;
        }
        return &self.block;
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    time: u32,
    height: u32,
    width: u32,
    bundle: ?*const lora.Bundle,
    reuse: ?*Compiled,
) !*Compiled {
    const tokens = tokenCount(time, height, width);
    progress.increaseEstimatedTotalItems(3 + layers_n);
    const view = load.viewFor(store.view(), "patchify_proj.weight", &.{
        "model.diffusion_model",
    });
    const embed_m = Embed.init(view, tokens, time, height, width);
    var block_m = Block.init(view.withPrefix("transformer_blocks").withLayer(0));
    const attn_backend = zml.attention.Backend.auto(platform);
    block_m.attn1.attn_backend = attn_backend;
    block_m.attn2.attn_backend = attn_backend;
    const finish_m = Finish.init(view, tokens, time, height, width);
    const hidden_sh: zml.Tensor = .init(.{ .n = 1, .s = tokens, .d = hidden }, .bf16);
    const ctx_sh: zml.Tensor = .init(.{ .n = 1, .s = 1024, .d = hidden }, .bf16);
    const ada_sh: zml.Tensor = .init(.{ .n = 1, .t = 1, .d = 9 * hidden }, .bf16);
    const pada_sh: zml.Tensor = .init(.{ .n = 1, .t = 1, .d = 2 * hidden }, .bf16);
    const cos_sh = zml.Tensor.init(.{ .s = tokens, .h = heads, .hd = head_dim }, .bf16).withPartitioning(.{ .h = .model });
    const sin_sh = zml.Tensor.init(.{ .s = tokens, .h = heads, .hd = head_dim }, .bf16).withPartitioning(.{ .h = .model });

    const out = try allocator.create(Compiled);
    errdefer allocator.destroy(out);

    log.info("compile LTX embed/block/finish tokens={d} reuse={s}", .{ tokens, if (reuse != null) "weights" else "load" });
    out.* = .{
        .embed = try zml.FnExe(Embed.forward).compile(allocator, io, platform, .{
            .shardings = shardings,
            .program_name = "minimax_h3_ltx_embed",
        }, .{.{
            .model = embed_m,
            .latent = .init(.{ .n = 1, .c = 128, .t = time, .h = height, .w = width }, .f32),
            .timestep = .init(.{ .n = 1 }, .f32),
        }}),
        .block = try zml.FnExe(Block.forward).compile(allocator, io, platform, .{
            .shardings = shardings,
            .program_name = "minimax_h3_ltx_block",
        }, .{.{
            .layer = block_m,
            .hidden = hidden_sh,
            .context = ctx_sh,
            .ada = ada_sh,
            .prompt_ada = pada_sh,
            .cos = cos_sh,
            .sin = sin_sh,
            .tau = .init(.{}, .f32),
        }}),
        .finish = try zml.FnExe(Finish.forward).compile(allocator, io, platform, .{
            .shardings = shardings,
            .program_name = "minimax_h3_ltx_finish",
        }, .{.{
            .model = finish_m,
            .hidden = .init(.{ .n = 1, .s = tokens, .d = hidden }, .bf16),
            .embedded = .init(.{ .n = 1, .t = 1, .d = hidden }, .bf16),
        }}),
        .embed_bufs = undefined,
        .finish_bufs = undefined,
        .blocks = &.{},
        .n = 0,
        .tokens = tokens,
        .time = time,
        .height = height,
        .width = width,
        .allocator = allocator,
        .owns_weights = false,
    };
    errdefer out.embed.deinit();
    errdefer out.block.deinit();
    errdefer out.finish.deinit();
    if (sol_attn.tokensOk(tokens)) {
        out.block_sol = compileSolBlock(
            allocator,
            io,
            platform,
            shardings,
            block_m,
            hidden_sh,
            ctx_sh,
            ada_sh,
            pada_sh,
            cos_sh,
            sin_sh,
        ) catch |err| blk: {
            log.warn("LTX Sol-Attn compile failed ({}); layers 1-47 stay dense", .{err});
            break :blk null;
        };
    } else {
        log.warn("LTX Sol-Attn skipped tokens={d} (need multiple of {d})", .{ tokens, sol_attn.block_size });
        out.block_sol = null;
    }
    errdefer if (out.block_sol) |*exe| exe.deinit();
    if (out.block_sol != null) {
        log.info("LTX Sol-Attn layers 1-47", .{});
    } else {
        log.warn("LTX Sol-Attn unavailable; layers 1-47 stay dense", .{});
    }
    if (reuse) |src| {
        out.embed_bufs = src.embed_bufs;
        out.finish_bufs = src.finish_bufs;
        out.blocks = src.blocks;
        out.n = src.n;
        out.owns_weights = false;
        log.info("compile LTX DiT tokens={d} {d}x{d}x{d} shared weights", .{ tokens, width, height, time });
        return out;
    }

    const block_bufs = try allocator.alloc(zml.Bufferized(Block), layers_n);
    errdefer allocator.free(block_bufs);
    out.embed_bufs = try weights.load(allocator, io, platform, store, shardings, Embed, &embed_m, progress, null);
    errdefer Embed.unloadBuffers(&out.embed_bufs);
    out.finish_bufs = try weights.load(allocator, io, platform, store, shardings, Finish, &finish_m, progress, null);
    errdefer Finish.unloadBuffers(&out.finish_bufs);
    out.blocks = block_bufs;
    out.owns_weights = true;
    if (bundle) |b| {
        try Embed.mergeLora(&out.embed_bufs, b, allocator, io, platform);
        try Finish.mergeLora(&out.finish_bufs, b, allocator, io, platform);
    }
    var i: u32 = 0;
    while (i < layers_n) : (i += 1) {
        const m = Block.init(view.withPrefix("transformer_blocks").withLayer(i));
        out.blocks[i] = try weights.load(allocator, io, platform, store, shardings, Block, &m, progress, null);
        if (bundle) |b| try Block.mergeLora(&out.blocks[i], b, allocator, io, platform, i);
        out.n += 1;
        if (i % 8 == 0) log.info("loaded LTX block {d}/{d}", .{ i, layers_n });
    }
    log.info("compile LTX DiT tokens={d} {d}x{d}x{d}", .{ tokens, width, height, time });
    return out;
}

fn compileSolBlock(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    block_m: Block,
    hidden_sh: zml.Tensor,
    ctx_sh: zml.Tensor,
    ada_sh: zml.Tensor,
    pada_sh: zml.Tensor,
    cos_sh: zml.Tensor,
    sin_sh: zml.Tensor,
) !zml.FnExe(Block.forward) {
    var sol_m = block_m;
    sol_m.attn1.sol_self = true;
    return zml.FnExe(Block.forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_ltx_block_sol",
    }, .{.{
        .layer = sol_m,
        .hidden = hidden_sh,
        .context = ctx_sh,
        .ada = ada_sh,
        .prompt_ada = pada_sh,
        .cos = cos_sh,
        .sin = sin_sh,
        .tau = .init(.{}, .f32),
    }});
}
