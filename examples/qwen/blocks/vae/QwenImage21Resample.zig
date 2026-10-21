const std = @import("std");
const zml = @import("zml");

const QwenImage21Upsample = @import("./QwenImage21Upsample.zig");
const Conv2d = @import("./Conv2d.zig");
const QwenImage21CausalConv3d = @import("./QwenImage21CausalConv3d.zig");

pub const QwenImage21Resample = @This();

upsample: QwenImage21Upsample,
conv: Conv2d,
time_conv: ?QwenImage21CausalConv3d,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21Resample)) void {
    Conv2d.unloadBuffers(&self.conv);
    if (self.time_conv) |*conv| {
        QwenImage21CausalConv3d.unloadBuffers(conv);
    }
}
