const zml = @import("zml");

pub const QwenImage21Upsample = @This();

scale_factor: f32 = 2,

pub fn forward(self: QwenImage21Upsample, x: zml.Tensor) zml.Tensor {
    return zml.nn.upsample(x.convert(.f32), .{ .mode = .nearest, .scale_factor = &.{ self.scale_factor, self.scale_factor } }).convert(x.dtype());
}
