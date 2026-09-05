// SPDX-License-Identifier: Apache-2.0
// Voxtral layer equations follow vLLM-Omni; copyright contributors to the vLLM project.
// Native Zig/ZML implementation. See README.md for the pinned reference.
const std = @import("std");
const zml = @import("zml");
pub const Tensor = zml.Tensor;
pub const View = zml.io.TensorStore.View;

pub fn linear(store: View) zml.nn.Linear {
    return .init(store.createTensor("weight", .{ .dout, .d }, .replicated), null, .d);
}

pub fn project(layer: zml.nn.Linear, x: Tensor) Tensor {
    return layer.forward(x).rename(.{ .dout = .d });
}

pub fn norm(x: Tensor, weight: Tensor, eps: f32) Tensor {
    return zml.nn.rmsNorm(x, .d, eps).mul(weight.broad(x.shape()));
}

pub fn unload(buffers: anytype) void {
    zml.meta.forEachVisit(buffers, *zml.Buffer, struct {
        fn call(_: usize, buffer: *zml.Buffer) void {
            buffer.deinit();
        }
    }.call, .{});
}

pub fn load(comptime T: type, model: *const T, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore) !zml.Bufferized(T) {
    var loader: zml.io.Loader = try .init(allocator, platform, .{ .dma_chunks = 4, .dma_chunk_size = 16 * zml.MiB, .parallelism = 4 });
    defer loader.deinit();
    var buffers = try zml.mem.bufferize(allocator, T, model);
    try loader.load(io, T, model, &buffers, store, &.{}, .{});
    try loader.await(io);
    return buffers;
}

pub const Cache = struct {
    k: Tensor,
    v: Tensor,

    pub fn init(len: u32) Cache {
        return .{ .k = .init(.{ .k = len, .h = 8, .hd = 128 }, .bf16), .v = .init(.{ .k = len, .h = 8, .hd = 128 }, .bf16) };
    }
};

pub const Layer = struct {
    wq: zml.nn.Linear,
    wk: zml.nn.Linear,
    wv: zml.nn.Linear,
    wo: zml.nn.Linear,
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,
    attention_norm: Tensor,
    ffn_norm: Tensor,
    q_norm: ?Tensor,
    k_norm: ?Tensor,
    attention_scale: ?Tensor,
    ffn_scale: ?Tensor,
    heads: u32,
    kv_heads: u32,
    eps: f32,

    pub fn init(store: View, codec: bool) Layer {
        const attention = store.withPrefix("attention");
        const ffn = store.withPrefix("feed_forward");
        return .{
            .wq = linear(attention.withPrefix("wq")),
            .wk = linear(attention.withPrefix("wk")),
            .wv = linear(attention.withPrefix("wv")),
            .wo = linear(attention.withPrefix("wo")),
            .w1 = linear(ffn.withPrefix("w1")),
            .w2 = linear(ffn.withPrefix("w2")),
            .w3 = linear(ffn.withPrefix("w3")),
            .attention_norm = store.createTensor("attention_norm.weight", .{.d}, .replicated),
            .ffn_norm = store.createTensor("ffn_norm.weight", .{.d}, .replicated),
            .q_norm = if (codec) attention.createTensor("q_norm.weight", .{.d}, .replicated) else null,
            .k_norm = if (codec) attention.createTensor("k_norm.weight", .{.d}, .replicated) else null,
            .attention_scale = if (codec) store.createTensor("attention_scale", .{.d}, .replicated) else null,
            .ffn_scale = if (codec) store.createTensor("ffn_scale", .{.d}, .replicated) else null,
            .heads = if (codec) 8 else 32,
            .kv_heads = 8,
            .eps = if (codec) 1e-2 else 1e-5,
        };
    }

    fn qkv(self: Layer, x: Tensor) [3]Tensor {
        const h = norm(x, self.attention_norm, self.eps);
        var q = project(self.wq, h);
        var k = project(self.wk, h);
        const v = project(self.wv, h);
        if (self.q_norm) |weight| q = norm(q, weight, 1e-6);
        if (self.k_norm) |weight| k = norm(k, weight, 1e-6);
        return .{
            q.splitAxis(.d, .{ .h = self.heads, .hd = 128 }),
            k.splitAxis(.d, .{ .h = self.kv_heads, .hd = 128 }),
            v.splitAxis(.d, .{ .h = self.kv_heads, .hd = 128 }),
        };
    }

    fn finish(self: Layer, x: Tensor, attn: Tensor) Tensor {
        var r = project(self.wo, attn.rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } }));
        if (self.attention_scale) |scale| r = r.mul(scale.broad(r.shape()));
        const h = x.add(r);
        const n = norm(h, self.ffn_norm, self.eps);
        r = project(self.w2, project(self.w1, n).silu().mul(project(self.w3, n)));
        if (self.ffn_scale) |scale| r = r.mul(scale.broad(r.shape()));
        return h.add(r);
    }

    pub fn forward(self: Layer, x: Tensor, mask: ?Tensor) Tensor {
        const q, const k, const v = self.qkv(x);
        return self.finish(x, zml.nn.sdpa(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), v.rename(.{ .s = .k }), .{ .attn_mask = mask }));
    }

    pub fn cached(self: Layer, x: Tensor, index: Tensor, cache: Cache) struct { Tensor, Cache } {
        var q, var k, var v = self.qkv(x);
        const positions = Tensor.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{.s}).add(index);
        const rope: zml.nn.RopeOpts = .{ .layout = .interleaved, .scaling = .{ .default = .{ .rope_theta = 1000000 } } };
        q = zml.nn.rope(q, positions, rope).rename(.{ .s = .q });
        k = zml.nn.rope(k, positions, rope).rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        const updated: Cache = .{
            .k = cache.k.dynamicUpdateSlice(.{ .k = index }, k).reuseBuffer(cache.k),
            .v = cache.v.dynamicUpdateSlice(.{ .k = index }, v).reuseBuffer(cache.v),
        };
        // Prefill only attends to this prompt; decode uses the populated prefix.
        var result: Tensor = undefined;
        if (x.dim(.s) > 1) {
            const mask = causalMask(x.dim(.s), x.dim(.s), Tensor.scalar(0, .u32), null, x.dtype());
            result = zml.nn.sdpa(q, k, v, .{ .attn_mask = mask });
        } else if (zml.Compiler.current().platform.target == .metal) {
            result = zml.attention.attention(q, updated.k, updated.v, index, .{ .metal_fa = .{ .num_tokens = Tensor.scalar(1, .u32) } }, .{ .metal_fa = {} });
        } else {
            const mask = causalMask(1, cache.k.dim(.k), index, null, x.dtype());
            result = zml.nn.sdpa(q, updated.k, updated.v, .{ .attn_mask = mask });
        }
        return .{ self.finish(x, result), updated };
    }
};

pub fn causalMask(queries: i64, keys: i64, index: Tensor, window: ?u32, dtype: zml.DataType) Tensor {
    const shape = zml.Shape.init(.{ .q = queries, .k = keys }, .i32);
    const q = Tensor.arange(.{ .end = queries }, .i32).withTags(.{.q}).add(index.convert(.i32)).broad(shape);
    const k = Tensor.arange(.{ .end = keys }, .i32).withTags(.{.k}).broad(shape);
    var valid = k.cmp(.LE, q);
    if (window) |w| valid = valid.select(k.cmp(.GE, q.addConstant(-@as(i64, w))), Tensor.scalar(false, .bool).broad(shape.withDtype(.bool)));
    return valid.select(Tensor.scalar(0, dtype).broad(shape.withDtype(dtype)), Tensor.scalar(-std.math.inf(f32), dtype).broad(shape.withDtype(dtype)));
}
