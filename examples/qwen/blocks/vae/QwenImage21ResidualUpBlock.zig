const std = @import("std");
const zml = @import("zml");

const QwenImage21DupUp3D = @import("./QwenImage21DupUp3D.zig");
const QwenImage21ResidualBlock = @import("./QwenImage21ResidualBlock.zig");
const QwenImage21Resample = @import("./QwenImage21Resample.zig");

pub const QwenImage21ResidualUpBlock = @import("./QwenImage21ResidualUpBlock.zig");

avg_shortcut: QwenImage21DupUp3D,
resnets: []QwenImage21ResidualBlock,
upsampler: ?QwenImage21Resample,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21ResidualUpBlock)) void {
    for (self.resnets) |*resnet| {
        QwenImage21ResidualBlock.unloadBuffers(resnet);
    }
    if (self.upsampler) |*upsampler| {
        QwenImage21Resample.unloadBuffers(upsampler);
    }
}
