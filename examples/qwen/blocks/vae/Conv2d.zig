const zml = @import("zml");
const std = @import("std");
const stdx = @import("stdx");

pub const Conv2d = @This();

weight: zml.Tensor,
bias: zml.Tensor,

kernel_size: usize,
padding: i64 = 0,
stride: i64 = 1,

pub fn unloadBuffers(self: *zml.Bufferized(Conv2d)) void {
    self.weight.deinit();
    self.bias.deinit();
}

pub fn forward(self: Conv2d, x: zml.Tensor) zml.Tensor {
    stdx.debug.assert(self.weight.dim(2) == self.kernel_size, "Kernel size is wrong", .{});
    stdx.debug.assert(self.weight.dim(3) == self.kernel_size, "Kernel size is wrong", .{});

    return zml.Tensor.conv2d(x, self.weight, .{
        .padding = &.{ self.padding, self.padding, self.padding, self.padding },
        .window_strides = &.{ self.stride, self.stride },
    });
}
