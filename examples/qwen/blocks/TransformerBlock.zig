const std = @import("std");
const zml = @import("zml");
const CompileArgs = @import("../compiler.zig").CompileArgs;

const Mlp = @import("./Mlp.zig");
const Attn = @import("./Attn.zig");

pub const TransformerBlock = @This();

img_norm1: zml.nn.LayerNorm,
img_norm2: zml.nn.LayerNorm,
img_mlp: Mlp,
attn: Attn,

pub fn load(
    self: *const TransformerBlock,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
) !zml.Bufferized(TransformerBlock) {
    var buffers = try zml.mem.bufferize(allocator, TransformerBlock, self);
    errdefer unloadBuffers(&buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    errdefer loader.deinit();

    try loader.load(io, TransformerBlock, self, &buffers, store, &.{}, .{});
    try loader.await(io);

    return buffers;
}

pub fn unloadBuffers(self: *zml.Bufferized(TransformerBlock)) void {
    Mlp.unloadBuffers(&self.img_mlp);
    Attn.unloadBuffers(&self.attn);
}

pub fn forward(
    self: TransformerBlock,
    x_: zml.Tensor,
    modulation: zml.Tensor,
    rotary_emb: zml.Tensor,
    target_token_mask: zml.Tensor,
    args: CompileArgs,
) zml.Tensor {
    var x = x_;
    const mod1, const mod2 = modulation.chunkExact(-1, 2);
    const img_modulated1, const img_gate1 = modulate(self.img_norm1.forward(x), mod1, target_token_mask);

    const attn = self.attn.forward(img_modulated1, rotary_emb, args);
    x = x.add(img_gate1.tanh().mul(attn));

    const img_modulated2, const img_gate2 = modulate(self.img_norm2.forward(x), mod2, target_token_mask);
    x = x.add(img_gate2.tanh().mul(self.img_mlp.forward(img_modulated2)));

    return x;
}

fn modulate(
    x: zml.Tensor,
    modulation: zml.Tensor,
    target_token_mask: zml.Tensor,
) [2]zml.Tensor {
    const scale, const gate = modulation.chunkExact(-1, 2);
    const modulated_scale = select_modulation_rows(scale, target_token_mask);
    const modulated_gate = select_modulation_rows(gate, target_token_mask);

    return [2]zml.Tensor{
        x.mul(modulated_scale.add(zml.Tensor.scalar(1, scale.dtype()))),
        modulated_gate,
    };
}

fn select_modulation_rows(x: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    const broad = x.withTags(.{ .mod, .d }).broad(.init(.{ .x = 1, .mod = 2, .d = x.dim(1) }, x.dtype()));
    const real, const zero = broad.chunkExact(.mod, 2);

    const tagged = mask.withTags(.{.d});
    const mask_ = tagged.broad(.init(.{ .x = 1, .d = tagged.dim(.d), .mod = 1 }, .bool))
        .broad(.init(.{ .x = 1, .d = tagged.dim(.d), .mod = real.dim(.d) }, .bool));

    // Condition (mask) : (1, 16415,    1)  -> Expands to (1, 16415, 4096)
    // Input 1 (real)   : (1,     1, 4096)  -> Expands to (1, 16415, 4096)
    // Input 2 (zero)   : (1,     1, 4096)  -> Expands to (1, 16415, 4096)
    return mask_.select(
        real.broad(.init(.{ .x = 1, .d = mask_.dim(.d), .mod = real.dim(.d) }, real.dtype())),
        zero.broad(.init(.{ .x = 1, .d = mask_.dim(.d), .mod = real.dim(.d) }, real.dtype())),
    );
}
