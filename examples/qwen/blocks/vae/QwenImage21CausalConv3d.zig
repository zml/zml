const std = @import("std");
const zml = @import("zml");

const Conv2d = @import("./Conv2d.zig");

pub const QwenImage21CausalConv3d = @This();

conv: Conv2d,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21CausalConv3d)) void {
    Conv2d.unloadBuffers(&self.conv);
}

pub fn forward(self: QwenImage21CausalConv3d, x: zml.Tensor) zml.Tensor {
    const no_temporal = x.squeeze(.t);
    const conved = self.conv.forward(no_temporal.convert(.f32)).convert(x.dtype());
    return conved.insertAxes(.kx, .{.t});
}
