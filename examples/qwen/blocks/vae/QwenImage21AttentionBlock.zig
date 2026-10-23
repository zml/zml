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
    const normed = self.norm.forward(x.merge(.{ .n = .{ .kx, .ky } }));

    const qkv = self.to_qkv.forward(normed);
    const q, const k, const v = qkv.chunkExact(.c, 3);

    const scale = std.math.pow(f32, @floatFromInt(x.dim(.c)), -0.5);
    const scores = q.dot(k.rename(.{ .n = .m }), .c).scale(scale);
    const attn = scores.softmax(-1);

    const pre_out = v.rename(.{ .n = .m }).dot(attn, .m).transpose(.{ .bs, .c, .t, .n });
    return self.proj.forward(pre_out).unflatten(.n, x.dim(.kx)).withTags(.{ .bs, .c, .t, .kx, .ky }).add(x);
}
