// SPDX-License-Identifier: Apache-2.0
// Equations adapted from BreezeBlue/breeze-tts and Hugging Face Transformers.
const std = @import("std");
const zml = @import("zml");
pub const Tensor = zml.Tensor;
pub const View = zml.io.TensorStore.View;
pub fn linear(s: View) zml.nn.Linear {
    return .init(s.createTensor("weight", .{ .o, .d }, .replicated), if (s.hasKey("bias")) s.createTensor("bias", .{.o}, .replicated) else null, .d);
}
pub fn project(l: zml.nn.Linear, x: Tensor) Tensor {
    return l.forward(x).rename(.{ .o = .d });
}
pub fn norm(x: Tensor, w: Tensor, eps: f32, offset: bool) Tensor {
    const f = x.convert(.f32);
    const n = zml.nn.rmsNorm(f, .d, eps);
    return n.mul((if (offset) w.convert(.f32).addConstant(1) else w.convert(.f32)).broad(n.shape())).convert(x.dtype());
}
pub fn layerNorm(x: Tensor, w: Tensor, b: Tensor, eps: f32) Tensor {
    const f = x.convert(.f32);
    const c = f.sub(f.mean(.d).broad(f.shape()));
    return c.mul(c.mul(c).mean(.d).addConstant(eps).rsqrt().broad(c.shape())).mul(w.convert(.f32).broad(f.shape())).add(b.convert(.f32).broad(f.shape())).convert(x.dtype());
}
pub fn unload(buffers: anytype) void {
    zml.meta.forEachVisit(buffers, *zml.Buffer, struct {
        fn call(_: usize, b: *zml.Buffer) void {
            b.deinit();
        }
    }.call, .{});
}
pub fn load(comptime T: type, model: *const T, a: std.mem.Allocator, io: std.Io, p: *const zml.Platform, store: *zml.io.TensorStore) !zml.Bufferized(T) {
    var loader: zml.io.Loader = try .init(a, p, .{ .dma_chunks = 4, .dma_chunk_size = 16 * zml.MiB, .parallelism = 4 });
    defer loader.deinit();
    var b = try zml.mem.bufferize(a, T, model);
    try loader.load(io, T, model, &b, store, &.{}, .{});
    try loader.await(io);
    return b;
}
pub const Cache = struct {
    k: Tensor,
    v: Tensor,
    pub fn init(len: usize, heads: usize, hd: usize) Cache {
        return .{ .k = .init(.{ .k = len, .h = heads, .hd = hd }, .bf16), .v = .init(.{ .k = len, .h = heads, .hd = hd }, .bf16) };
    }
};
pub fn mask(qn: i64, kn: i64, index: Tensor, window: ?u32, bidirectional: bool, dtype: zml.DataType) Tensor {
    const shape = zml.Shape.init(.{ .q = qn, .k = kn }, .i32);
    const q = Tensor.arange(.{ .end = qn }, .i32).withTags(.{.q}).add(index.convert(.i32)).broad(shape);
    const k = Tensor.arange(.{ .end = kn }, .i32).withTags(.{.k}).broad(shape);
    var valid = if (bidirectional) Tensor.scalar(true, .bool).broad(shape.withDtype(.bool)) else k.cmp(.LE, q);
    if (window) |w| {
        const delta = k.sub(q);
        const local = if (bidirectional) delta.cmp(.GE, Tensor.scalar(-@as(i64, (w + 1) / 2) + 1, .i32)).logical(.AND, delta.cmp(.LE, Tensor.scalar(w / 2, .i32))) else delta.cmp(.GT, Tensor.scalar(-@as(i64, w), .i32));
        valid = valid.logical(.AND, local);
    }
    return valid.select(Tensor.scalar(0, dtype).broad(shape.withDtype(dtype)), Tensor.scalar(-std.math.inf(f32), dtype).broad(shape.withDtype(dtype)));
}
pub const Kind = enum { backbone, depth, text, codec };
pub const Layer = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    o: zml.nn.Linear,
    gate: zml.nn.Linear,
    up: zml.nn.Linear,
    down: zml.nn.Linear,
    pre: Tensor,
    mid: Tensor,
    post_attn: ?Tensor,
    post_ff: ?Tensor,
    qnorm: ?Tensor,
    knorm: ?Tensor,
    attn_scale: ?Tensor,
    ff_scale: ?Tensor,
    kind: Kind,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    eps: f32,
    rope: zml.nn.RopeOpts,
    window: ?u32,
    pub fn init(s: View, kind: Kind, i: usize) Layer {
        const text = kind == .text;
        const codec = kind == .codec;
        const attn = s.withPrefix("self_attn");
        return .{
            .q = linear(attn.withPrefix("q_proj")),
            .k = linear(attn.withPrefix("k_proj")),
            .v = linear(attn.withPrefix("v_proj")),
            .o = linear(attn.withPrefix("o_proj")),
            .gate = linear(s.withPrefix("mlp.gate_proj")),
            .up = linear(s.withPrefix("mlp.up_proj")),
            .down = linear(s.withPrefix("mlp.down_proj")),
            .pre = s.createTensor(if (text) "pre_self_attn_layernorm.weight" else "input_layernorm.weight", .{.d}, .replicated),
            .mid = s.createTensor(if (text) "pre_feedforward_layernorm.weight" else "post_attention_layernorm.weight", .{.d}, .replicated),
            .post_attn = if (text) s.createTensor("post_self_attn_layernorm.weight", .{.d}, .replicated) else null,
            .post_ff = if (text) s.createTensor("post_feedforward_layernorm.weight", .{.d}, .replicated) else null,
            .qnorm = if (kind == .backbone or text) attn.createTensor("q_norm.weight", .{.d}, .replicated) else null,
            .knorm = if (kind == .backbone or text) attn.createTensor("k_norm.weight", .{.d}, .replicated) else null,
            .attn_scale = if (codec) s.createTensor("self_attn_layer_scale.scale", .{.d}, .replicated) else null,
            .ff_scale = if (codec) s.createTensor("mlp_layer_scale.scale", .{.d}, .replicated) else null,
            .kind = kind,
            .heads = switch (kind) {
                .text => 4,
                .depth => 8,
                else => 16,
            },
            .kv_heads = switch (kind) {
                .text => 1,
                .depth => 2,
                .backbone => 8,
                .codec => 16,
            },
            .hd = if (text) 256 else if (codec) 64 else 128,
            .eps = if (text or kind == .backbone) 1e-6 else 1e-5,
            .window = if (text and (i + 1) % 6 != 0) 512 else if (codec) 72 else null,
            .rope = .{ .scaling = switch (kind) {
                .backbone => .{ .default = .{ .rope_theta = 1000000 } },
                .depth => .{ .llama3 = .{ .rope_theta = 500000, .factor = 32, .high_freq_factor = 0.0078125, .low_freq_factor = 0.001953125, .original_max_position_embeddings = 16 } },
                .text => if ((i + 1) % 6 == 0) .{ .linear = .{ .rope_theta = 1000000, .factor = 8 } } else .{ .default = .{ .rope_theta = 10000 } },
                .codec => .{ .default = .{ .rope_theta = 10000 } },
            } },
        };
    }
    fn qkv(self: Layer, x: Tensor, index: Tensor) [3]Tensor {
        const n = norm(x, self.pre, self.eps, self.kind == .text);
        var q = project(self.q, n).splitAxis(.d, .{ .h = self.heads, .hd = self.hd });
        var k = project(self.k, n).splitAxis(.d, .{ .h = self.kv_heads, .hd = self.hd });
        if (self.qnorm) |w| q = norm(q.rename(.{ .hd = .d }), w, self.eps, self.kind == .text).rename(.{ .d = .hd });
        if (self.knorm) |w| k = norm(k.rename(.{ .hd = .d }), w, self.eps, self.kind == .text).rename(.{ .d = .hd });
        const pos = Tensor.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{.s}).add(index);
        return .{ zml.nn.rope(q, pos, self.rope).rename(.{ .s = .q }), zml.nn.rope(k, pos, self.rope).rename(.{ .s = .k }), project(self.v, n).splitAxis(.d, .{ .h = self.kv_heads, .hd = self.hd }).rename(.{ .s = .k }) };
    }
    fn finish(self: Layer, x: Tensor, attn: Tensor) Tensor {
        var r = project(self.o, attn.rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } }));
        if (self.post_attn) |w| r = norm(r, w, self.eps, true);
        if (self.attn_scale) |w| r = r.mul(w.broad(r.shape()));
        const h = x.add(r);
        const n = norm(h, self.mid, self.eps, self.kind == .text);
        const g = project(self.gate, n);
        r = project(self.down, (if (self.kind == .text) g.gelu() else g.silu()).mul(project(self.up, n)));
        if (self.post_ff) |w| r = norm(r, w, self.eps, true);
        if (self.ff_scale) |w| r = r.mul(w.broad(r.shape()));
        return h.add(r);
    }
    pub fn forward(self: Layer, x: Tensor) Tensor {
        const q, const k, const v = self.qkv(x, Tensor.scalar(0, .u32));
        return self.finish(x, zml.nn.sdpa(q, k, v, .{ .attn_mask = mask(x.dim(.s), x.dim(.s), Tensor.scalar(0, .u32), self.window, self.kind == .text, x.dtype()) }));
    }
    pub fn cached(self: Layer, x: Tensor, index: Tensor, c: Cache) struct { Tensor, Cache } {
        const q, const k, const v = self.qkv(x, index);
        const next: Cache = .{ .k = c.k.dynamicUpdateSlice(.{ .k = index }, k).reuseBuffer(c.k), .v = c.v.dynamicUpdateSlice(.{ .k = index }, v).reuseBuffer(c.v) };
        const a = if (x.dim(.s) > 1) zml.nn.sdpa(q, k, v, .{ .attn_mask = mask(x.dim(.s), x.dim(.s), Tensor.scalar(0, .u32), null, false, x.dtype()) }) else zml.nn.sdpa(q, next.k, next.v, .{ .attn_mask = mask(1, c.k.dim(.k), index, null, false, x.dtype()) });
        return .{ self.finish(x, a), next };
    }
};
