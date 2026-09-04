const std = @import("std");

const zml = @import("zml");

const weights = @import("../recipe/weights.zig");
const load = @import("load.zig");

const log = std.log.scoped(.minimax_h3_stage2);

const linear = weights.linear;

// =============================================================================
// refine/connector.zig — Gemma hidden states → LTX context
//
// 8 layers, shared across SKUs.
// =============================================================================

const qkv_part = .{ .dout = .model, .d = .replicated };
const out_part = .{ .dout = .replicated, .d = .model };

pub const dim: i64 = 4096;
pub const heads: i64 = 32;
pub const head_dim: i64 = 128;
pub const layers_n: u32 = 8;
pub const registers: u32 = 128;
pub const min_tokens: u32 = 1024;

fn rms(x: zml.Tensor) zml.Tensor {
    return zml.nn.rmsNorm(x, .d, 1e-6);
}

const Attn = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    o: zml.nn.Linear,
    gate: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,

    pub fn init(store: zml.io.TensorStore.View) Attn {
        return .{
            .q = linear(store, "to_q.weight", "to_q.bias", qkv_part, .replicated),
            .k = linear(store, "to_k.weight", "to_k.bias", qkv_part, .replicated),
            .v = linear(store, "to_v.weight", "to_v.bias", qkv_part, .replicated),
            .o = linear(store, "to_out.0.weight", "to_out.0.bias", out_part, .replicated),
            .gate = linear(store, "to_gate_logits.weight", "to_gate_logits.bias", qkv_part, .replicated),
            .q_norm = weights.rmsNorm(store.withPrefix("q_norm"), .{.d}, 1e-5),
            .k_norm = weights.rmsNorm(store.withPrefix("k_norm"), .{.d}, 1e-5),
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

    pub fn forward(self: Attn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_q = x.withPartitioning(.{ .d = .replicated });
        var q = self.q_norm.forward(self.q.forward(x_q).rename(.{ .dout = .d }));
        var k = self.k_norm.forward(self.k.forward(x_q).rename(.{ .dout = .d }));
        const v = self.v.forward(x_q).rename(.{ .dout = .d });
        q = q.splitAxis(.d, .{ .h = heads, .hd = head_dim }).withPartitioning(.{ .h = .model });
        k = k.splitAxis(.d, .{ .h = heads, .hd = head_dim }).withPartitioning(.{ .h = .model });
        const vv = v.splitAxis(.d, .{ .h = heads, .hd = head_dim }).withPartitioning(.{ .h = .model });
        q = zml.nn.applyRotary(q, cos, sin);
        k = zml.nn.applyRotary(k, cos, sin);
        var out = zml.nn.sdpa(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), vv.rename(.{ .s = .k }), .{})
            .rename(.{ .q = .s });
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
            .inn = linear(store, "net.0.proj.weight", "net.0.proj.bias", qkv_part, .replicated),
            .out = linear(store, "net.2.weight", "net.2.bias", out_part, .replicated),
        };
    }
    pub fn unloadBuffers(self: *zml.Bufferized(Ff)) void {
        zml.nn.Linear.unloadBuffers(&self.inn);
        zml.nn.Linear.unloadBuffers(&self.out);
    }
    pub fn forward(self: Ff, x: zml.Tensor) zml.Tensor {
        const x_q = x.withPartitioning(.{ .d = .replicated });
        return self.out.forward(self.inn.forward(x_q).gelu().rename(.{ .dout = .d }))
            .rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const Block = struct {
    attn: Attn,
    ff: Ff,
    pub const Input = struct { layer: Block, hidden: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) Block {
        return .{
            .attn = .init(store.withPrefix("attn1")),
            .ff = .init(store.withPrefix("ff")),
        };
    }
    pub fn unloadBuffers(self: *zml.Bufferized(Block)) void {
        Attn.unloadBuffers(&self.attn);
        Ff.unloadBuffers(&self.ff);
    }
    pub fn forward(input: Input) Output {
        const self = input.layer;
        const hidden = input.hidden.withPartitioning(.{ .d = .replicated });
        const a = self.attn.forward(rms(hidden), input.cos, input.sin);
        const x1 = hidden.add(a);
        const f = self.ff.forward(rms(x1));
        return .{ .hidden = x1.add(f).reuseBuffer(input.hidden) };
    }
};

pub const Pad = struct {
    registers: zml.Tensor,
    pub const Input = struct { model: Pad, text: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View) Pad {
        return .{ .registers = store.createTensor("learnable_registers", .{ .r, .d }, .replicated) };
    }
    pub fn unloadBuffers(self: *zml.Bufferized(Pad)) void {
        self.registers.deinit();
    }

    /// text: [n, s=keep_tokens, d] → [n, 1024, d]
    pub fn forward(input: Input) Output {
        const text = input.text.withPartialTags(.{ .n, .s, .d });
        const n = text.dim(.n);
        const used = text.dim(.s);
        const regs = input.model.registers.convert(text.dtype()).withPartialTags(.{ .r, .d });
        const copies: i64 = 8;
        var tiled = regs;
        var i: i64 = 1;
        while (i < copies) : (i += 1) {
            tiled = zml.Tensor.concatenate(&.{ tiled, regs }, .r);
        }
        const tail = tiled.slice(.r, .{ .start = used, .end = min_tokens }).rename(.{ .r = .s });
        const tail_b = tail.reshape(.{ .n = n, .s = tail.dim(.s), .d = tail.dim(.d) });
        return .{ .hidden = zml.Tensor.concatenate(&.{ text, tail_b }, .s).convert(.bf16) };
    }
};

pub const Finish = struct {
    pub const Input = struct { hidden: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };
    pub fn forward(input: Input) Output {
        return .{ .hidden = rms(input.hidden) };
    }
};

pub const Compiled = struct {
    pad: zml.FnExe(Pad.forward),
    block: zml.FnExe(Block.forward),
    finish: zml.FnExe(Finish.forward),
    pad_bufs: zml.Bufferized(Pad),
    blocks: [layers_n]zml.Bufferized(Block) = undefined,
    n: u32 = 0,

    pub fn deinit(self: *Compiled) void {
        Pad.unloadBuffers(&self.pad_bufs);
        var i: u32 = 0;
        while (i < self.n) : (i += 1) Block.unloadBuffers(&self.blocks[i]);
        self.pad.deinit();
        self.block.deinit();
        self.finish.deinit();
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    text_len: u32,
) !Compiled {
    progress.increaseEstimatedTotalItems(3);
    const root = load.viewFor(store.view(), "video_embeddings_connector.learnable_registers", &.{
        "model.diffusion_model",
    }).withPrefix("video_embeddings_connector");
    const pad_m = Pad.init(root);
    const block_m = Block.init(root.withPrefix("transformer_1d_blocks").withLayer(0));
    const hidden_sh: zml.Tensor = .init(.{ .n = 1, .s = min_tokens, .d = dim }, .bf16);
    const cos_sh = zml.Tensor.init(.{ .s = min_tokens, .h = heads, .hd = head_dim }, .bf16).withPartitioning(.{ .h = .model });
    const sin_sh = zml.Tensor.init(.{ .s = min_tokens, .h = heads, .hd = head_dim }, .bf16).withPartitioning(.{ .h = .model });
    var out: Compiled = .{
        .pad = try zml.FnExe(Pad.forward).compile(allocator, io, platform, .{
            .shardings = shardings,
            .program_name = "minimax_h3_ltx_conn_pad",
        }, .{.{ .model = pad_m, .text = .init(.{ .n = 1, .s = text_len, .d = dim }, .f32) }}),
        .block = try zml.FnExe(Block.forward).compile(allocator, io, platform, .{
            .shardings = shardings,
            .program_name = "minimax_h3_ltx_conn_block",
        }, .{.{ .layer = block_m, .hidden = hidden_sh, .cos = cos_sh, .sin = sin_sh }}),
        .finish = try zml.FnExe(Finish.forward).compile(allocator, io, platform, .{
            .shardings = shardings,
            .program_name = "minimax_h3_ltx_conn_finish",
        }, .{.{ .hidden = .init(.{ .n = 1, .s = min_tokens, .d = dim }, .bf16) }}),
        .pad_bufs = try weights.load(allocator, io, platform, store, shardings, Pad, &pad_m, progress, null),
    };
    var i: u32 = 0;
    while (i < layers_n) : (i += 1) {
        const m = Block.init(root.withPrefix("transformer_1d_blocks").withLayer(i));
        out.blocks[i] = try weights.load(allocator, io, platform, store, shardings, Block, &m, progress, null);
        out.n += 1;
    }
    log.info("compile LTX video connector layers={d} tokens={d}", .{ layers_n, min_tokens });
    return out;
}
