// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM-Omni's Voxtral TTS model; copyright contributors to the vLLM project.
// Native Zig/ZML implementation. See README.md for the pinned reference.
const std = @import("std");
const zml = @import("zml");
const layers = @import("layers.zig");
const Tensor = zml.Tensor;
const Layer = layers.Layer;
const project = layers.project;
const norm = layers.norm;
pub const Cache = [26]layers.Cache;

pub const LanguageModel = struct {
    text_embedding: Tensor,
    audio_embedding: Tensor,
    blocks: []Layer,
    norm_weight: Tensor,

    pub fn init(allocator: std.mem.Allocator, store: layers.View) !LanguageModel {
        var self: LanguageModel = undefined;
        self.blocks = try allocator.alloc(Layer, 26);
        self.text_embedding = store.createTensor("mm_audio_embeddings.tok_embeddings.weight", .{ .vocab, .d }, .replicated);
        self.audio_embedding = store.createTensor("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", .{ .vocab, .d }, .replicated);
        self.norm_weight = store.createTensor("norm.weight", .{.d}, .replicated);
        for (self.blocks, 0..) |*block, i| block.* = .init(store.withPrefix("layers").withLayer(i), false);
        return self;
    }

    pub fn prefill(self: LanguageModel, tokens: Tensor, voice: Tensor, cache: Cache) struct { Tensor, Cache } {
        var h = self.text_embedding.gather(.{ .vocab = tokens }, .{});
        // [BOS, BEGIN_AUDIO, <voice embeddings>, TEXT_TO_AUDIO, text, AUDIO_TO_TEXT, BEGIN_AUDIO]
        h = h.dynamicUpdateSlice(.{ .s = Tensor.scalar(2, .u32) }, voice);
        return self.forward(h, Tensor.scalar(0, .u32), cache);
    }

    pub fn decode(self: LanguageModel, codes: Tensor, index: Tensor, cache: Cache) struct { Tensor, Cache } {
        var offsets: [37]u32 = undefined;
        offsets[0] = 0;
        for (1..37) |i| offsets[i] = @intCast(8194 + (i - 1) * 23);
        const ids = codes.add(Tensor.constantTensor(.init(.{ .cb = 37 }, .u32), std.mem.asBytes(&offsets)));
        const h = self.audio_embedding.gather(.{ .vocab = ids }, .{}).convert(.f32).sum(.cb).squeeze(.cb).convert(.bf16).appendAxes(.{.s}).transpose(.{ .s, .d });
        return self.forward(h, index, cache);
    }

    fn forward(self: LanguageModel, x: Tensor, index: Tensor, cache: Cache) struct { Tensor, Cache } {
        var h = x;
        var updated = cache;
        for (self.blocks, 0..) |block, i| {
            h, updated[i] = block.cached(h, index, cache[i]);
        }
        return .{ norm(h.slice(.s, .{ .start = h.dim(.s) - 1, .len = 1 }), self.norm_weight, 1e-5), updated };
    }
};

pub const AcousticModel = struct {
    input: zml.nn.Linear,
    time: zml.nn.Linear,
    llm: zml.nn.Linear,
    semantic: zml.nn.Linear,
    acoustic: zml.nn.Linear,
    norm_weight: Tensor,
    blocks: []Layer,

    pub fn init(allocator: std.mem.Allocator, store_: layers.View) !AcousticModel {
        const store = store_.withPrefix("acoustic_transformer");
        const self: AcousticModel = .{
            .input = layers.linear(store.withPrefix("input_projection")),
            .time = layers.linear(store.withPrefix("time_projection")),
            .llm = layers.linear(store.withPrefix("llm_projection")),
            .semantic = layers.linear(store.withPrefix("semantic_codebook_output")),
            .acoustic = layers.linear(store.withPrefix("acoustic_codebook_output")),
            .norm_weight = store.createTensor("norm.weight", .{.d}, .replicated),
            .blocks = try allocator.alloc(Layer, 3),
        };
        for (self.blocks, 0..) |*block, i| block.* = .init(store.withPrefix("layers").withLayer(i), false);
        return self;
    }

    pub fn generate(self: AcousticModel, hidden: Tensor, noise: Tensor, steps: u32, guidance: f32) Tensor {
        const logits = project(self.semantic, hidden).slice(.d, .{ .start = 1, .end = 8194 }).convert(.f32);
        const semantic = logits.argMax(.d).indices.addConstant(1).convert(.u32).reshape(.{ .cb = 1 });
        const conditional = project(self.llm, hidden).appendAxes(.{.b}).transpose(.{ .s, .b, .d });
        const conditioning = Tensor.concatenate(&.{ conditional, Tensor.zeroes(conditional.shape()) }, .b);
        var sampled = noise.convert(.bf16);
        var times: [3072]f32 = undefined;
        for (0..steps) |i| {
            // The reference casts the Euler schedule to the model dtype first.
            const t: f32 = zml.floats.BFloat16.fromF32(@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps))).toF32();
            const t_next: f32 = zml.floats.BFloat16.fromF32(@as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(steps))).toF32();
            for (0..1536) |j| {
                const angle = t * @exp(-@log(@as(f32, 10000)) * @as(f32, @floatFromInt(j)) / 1536);
                times[j] = @cos(angle);
                times[1536 + j] = @sin(angle);
            }
            const time_input = Tensor.constantTensor(.init(.{ .s = 1, .d = 3072 }, .f32), std.mem.asBytes(&times)).convert(.bf16);
            const t_proj = project(self.time, time_input).appendAxes(.{.b}).transpose(.{ .s, .b, .d }).broad(conditioning.shape());
            const x_proj = project(self.input, sampled.withTags(.{.d}).appendAxes(.{.s}).transpose(.{ .s, .d })).appendAxes(.{.b}).transpose(.{ .s, .b, .d }).broad(conditioning.shape());
            var h = Tensor.concatenate(&.{ x_proj, t_proj, conditioning }, .s);
            for (self.blocks) |block| h = block.forward(h, null);
            const velocity = project(self.acoustic, norm(h.slice(.s, .single(0)), self.norm_weight, 1e-5));
            const alpha = zml.floats.BFloat16.fromF32(guidance).toF32();
            const v = velocity.slice(.b, .single(0)).scale(alpha).add(velocity.slice(.b, .single(1)).scale(1 - alpha));
            sampled = sampled.add(v.withTags(.{.cb}).scale(t_next - t));
        }
        // Metal's BF16-to-integer saturation lowering emits an invalid bound;
        // widen before rounding/conversion (the quantization math stays BF16).
        const codes = quantize(sampled);
        return Tensor.concatenate(&.{ semantic, codes }, .cb);
    }
};

pub fn quantize(sampled: Tensor) Tensor {
    const value = sampled.maximum(Tensor.scalar(-1, .bf16)).minimum(Tensor.scalar(1, .bf16)).addConstant(1).scale(10).convert(.f32);
    // round_nearest_even is not legalized by Apple's Metal compiler. Values
    // are nonnegative: truncate, then increment above half or at an odd tie.
    const lower = value.convert(.u32);
    const fraction = value.sub(lower.convert(.f32));
    const half = Tensor.scalar(0.5, .f32).broad(value.shape());
    const odd = lower.logical(.AND, Tensor.scalar(1, .u32)).cmp(.EQ, Tensor.scalar(1, .u32));
    const increment = fraction.cmp(.GT, half).select(Tensor.scalar(true, .bool).broad(odd.shape()), fraction.cmp(.EQ, half).select(odd, Tensor.scalar(false, .bool).broad(odd.shape())));
    return lower.add(increment.convert(.u32)).addConstant(2);
}
