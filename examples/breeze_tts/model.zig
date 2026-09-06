// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const zml = @import("zml");
const l = @import("layers.zig");
const T = zml.Tensor;
pub const BackboneCache = [28]l.Cache;
pub const DepthCache = [12]l.Cache;
pub const TextEncoder = struct {
    embedding: T,
    eoi: T,
    layers: []l.Layer,
    norm: T,
    projection: zml.nn.Linear,
    pub fn init(a: std.mem.Allocator, s: l.View) !TextEncoder {
        const v = s.withPrefix("text_encoder");
        const self: TextEncoder = .{ .embedding = v.createTensor("embed_tokens.weight", .{ .vocab, .d }, .replicated), .eoi = v.createTensor("embed_tokens.eoi_embedding", .{.d}, .replicated), .layers = try a.alloc(l.Layer, 26), .norm = v.createTensor("norm.weight", .{.d}, .replicated), .projection = l.linear(s.withPrefix("text_encoder_proj")) };
        for (self.layers, 0..) |*b, i| b.* = .init(v.withPrefix("layers").withLayer(i), .text, i);
        return self;
    }
    pub fn forward(self: TextEncoder, ids: T) T {
        var x = self.embedding.gather(.{ .vocab = ids }, .{}).scale(@sqrt(@as(f32, 1152)));
        x = ids.cmp(.EQ, T.scalar(256000, .u32)).broad(x.shape().withDtype(.bool)).select(self.eoi.broad(x.shape()), x);
        for (self.layers) |b| x = b.forward(x);
        return l.project(self.projection, l.norm(x, self.norm, 1e-6, true));
    }
};
pub const Backbone = struct {
    embedding: T,
    layers: []l.Layer,
    norm: T,
    head: zml.nn.Linear,
    pub fn init(a: std.mem.Allocator, s: l.View) !Backbone {
        const self: Backbone = .{ .embedding = s.createTensor("depth_decoder.model.embed_tokens.weight", .{ .vocab, .d }, .replicated), .layers = try a.alloc(l.Layer, 28), .norm = s.createTensor("backbone_model.norm.weight", .{.d}, .replicated), .head = l.linear(s.withPrefix("lm_head")) };
        for (self.layers, 0..) |*b, i| b.* = .init(s.withPrefix("backbone_model.layers").withLayer(i), .backbone, i);
        return self;
    }
    pub fn embed(self: Backbone, codes: T) T {
        const offset = T.arange(.{ .end = 16 }, .u32).withTags(.{.cb}).scale(2051);
        return self.embedding.gather(.{ .vocab = codes.add(offset.broad(codes.shape())) }, .{}).sum(.cb).squeeze(.cb);
    }
    pub fn prefill(self: Backbone, prompt: T, cache: BackboneCache) struct { T, T, BackboneCache } {
        return self.forward(prompt, T.scalar(0, .u32), cache);
    }
    pub fn decode(self: Backbone, codes: T, index: T, cache: BackboneCache) struct { T, T, BackboneCache } {
        return self.forward(self.embed(codes), index, cache);
    }
    fn forward(self: Backbone, input: T, index: T, cache: BackboneCache) struct { T, T, BackboneCache } {
        var x = input;
        var updated: BackboneCache = undefined;
        for (self.layers, cache, &updated) |b, c, *next| x, next.* = b.cached(x, index, c);
        const hidden = l.norm(x.slice(.s, .{ .start = x.dim(.s) - 1, .len = 1 }), self.norm, 1e-6, false);
        return .{ hidden, l.project(self.head, hidden), updated };
    }
};
pub const Depth = struct {
    embedding: T,
    projection: zml.nn.Linear,
    layers: []l.Layer,
    norm: T,
    head: T,
    pub fn init(a: std.mem.Allocator, s: l.View) !Depth {
        const v = s.withPrefix("depth_decoder");
        const self: Depth = .{ .embedding = v.createTensor("model.embed_tokens.weight", .{ .vocab, .d }, .replicated), .projection = l.linear(v.withPrefix("model.inputs_embeds_projector")), .layers = try a.alloc(l.Layer, 12), .norm = v.createTensor("model.norm.weight", .{.d}, .replicated), .head = v.createTensor("codebooks_head.weight", .{ .cb, .d, .vocab }, .replicated) };
        for (self.layers, 0..) |*b, i| b.* = .init(v.withPrefix("model.layers").withLayer(i), .depth, i);
        return self;
    }
    pub fn prefill(self: Depth, hidden: T, first: T, cache: DepthCache) struct { T, DepthCache } {
        const token = self.embedding.gather(.{ .vocab = first }, .{}).reshape(.{ .s = 1, .d = 2048 });
        return self.forward(T.concatenate(&.{ hidden, token }, .s), T.scalar(0, .u32), T.scalar(0, .u32), cache);
    }
    pub fn decode(self: Depth, token: T, index: T, cache: DepthCache) struct { T, DepthCache } {
        const id = token.add(index.sub(T.scalar(1, .u32)).scale(2051));
        return self.forward(self.embedding.gather(.{ .vocab = id }, .{}).reshape(.{ .s = 1, .d = 2048 }), index, index.sub(T.scalar(1, .u32)), cache);
    }
    fn forward(self: Depth, input: T, index: T, cb: T, cache: DepthCache) struct { T, DepthCache } {
        var x = l.project(self.projection, input);
        var updated: DepthCache = undefined;
        for (self.layers, cache, &updated) |b, c, *next| x, next.* = b.cached(x, index, c);
        x = l.norm(x.slice(.s, .{ .start = x.dim(.s) - 1, .len = 1 }), self.norm, 1e-5, false);
        return .{ x.dot(self.head.gather(.{ .cb = cb }, .{}), .{.d}).rename(.{ .vocab = .d }), updated };
    }
};
