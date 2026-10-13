const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextRMSNorm = @This();

weight: zml.Tensor,
/// Defaults to 1e-6
eps: f32,
tag: zml.Shape.Tag,

pub fn init(weight: zml.Tensor, tag: anytype) Qwen3VLTextRMSNorm {
    return Qwen3VLTextRMSNorm{ .weight = weight, .eps = 1e-6, .tag = zml.Shape.toTag(tag) };
}

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextRMSNorm)) void {
    self.weight.deinit();
}
