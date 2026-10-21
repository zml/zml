const std = @import("std");
const zml = @import("zml");

const QwenImage21RMS_norm = @import("./QwenImage21RMS_norm.zig");
const Conv2d = @import("./Conv2d.zig");

// QwenImage21AttentionBlock(
//   (norm): QwenImage21RMS_norm()
//   (to_qkv): Conv2d(1152, 3456, kernel_size=(1, 1), stride=(1, 1))
//   (proj): Conv2d(1152, 1152, kernel_size=(1, 1), stride=(1, 1))
// )

pub const QwenImage21AttentionBlock = @This();

norm: QwenImage21RMS_norm,
to_qkv: Conv2d,
proj: Conv2d,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21AttentionBlock)) void {
    QwenImage21RMS_norm.unloadBuffers(&self.norm);
    Conv2d.unloadBuffers(&self.to_qkv);
    Conv2d.unloadBuffers(&self.proj);
}

pub fn forward(self: QwenImage21AttentionBlock, x: zml.Tensor) zml.Tensor {
    // std.debug.print("x.merge(.{ .n = .{ .kx, .ky } }) {f}\n", .{x.merge(.{ .n = .{ .kx, .ky } }).shape()}); // x.merge(.{ .n = .{ .kx, .ky } }) {bs=1,c=1152,t=1,n=64,bf16}
    const normed = self.norm.forward(x.merge(.{ .n = .{ .kx, .ky } }));
    // std.debug.print("normed {f}\n", .{normed.shape()});

    const qkv = self.to_qkv.forward(normed);
    // std.debug.print("qkv {f}\n", .{qkv.shape()});
    const q, const k, const v = qkv.chunkExact(.c, 3);
    // std.debug.print("q {f}\n", .{q.shape()});

    const scale = std.math.pow(f32, @floatFromInt(x.dim(.c)), -0.5);
    const scores = q.dot(k.rename(.{ .n = .m }), .c).scale(scale);
    // std.debug.print("scores: {f}\n", .{scores.shape()});
    const attn = scores.softmax(-1);

    // std.debug.print("attn: {f}\n", .{attn.shape()});
    // std.debug.print("v: {f}\n", .{v.shape()});

    const pre_out = v.rename(.{ .n = .m }).dot(attn, .m).transpose(.{ .bs, .c, .t, .n });
    // std.debug.print("pre_out: {f}\n", .{pre_out.shape()});

    return self.proj.forward(pre_out).unflatten(.n, 8).withTags(.{ .bs, .c, .t, .kx, .ky }).add(x);
}
