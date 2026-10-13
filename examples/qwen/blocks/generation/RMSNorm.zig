const zml = @import("zml");

// From diffusers/models/normalization.py, with is_torch_npu_available() == False
pub const RMSNorm = @This();

// Removed the optional
weight: zml.Tensor,
/// Defaults to 1e-6
eps: f32,
tag: zml.Shape.Tag,

pub fn init(weight: zml.Tensor, eps: ?f32, tag: anytype) RMSNorm {
    return .{
        .weight = weight,
        .eps = eps orelse 1e-6,
        .tag = zml.Shape.toTag(tag),
    };
}

pub fn unloadBuffers(self: *zml.Bufferized(RMSNorm)) void {
    self.weight.deinit();
}

// The basic nn.LayerNorm
pub fn forward(self: RMSNorm, x: zml.Tensor) zml.Tensor {
    var normalized = zml.nn.rmsNorm(x, self.tag, self.eps);
    return normalized.mul(self.weight.withTags(.{.hd}).broad(normalized.shape()));
}
