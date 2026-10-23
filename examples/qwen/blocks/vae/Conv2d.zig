const zml = @import("zml");
const std = @import("std");
const stdx = @import("stdx");

pub const Conv2d = @This();

weight: zml.Tensor,
bias: zml.Tensor,

stride: i64 = 1,

pub fn unloadBuffers(self: *zml.Bufferized(Conv2d)) void {
    self.weight.deinit();
    self.bias.deinit();
}

pub fn forward(self: Conv2d, x: zml.Tensor) zml.Tensor {
    // All the VAE convs are "same" convs: the padding is inferred from the square, odd kernel.
    const kernel_size = self.weight.dim(.kx);
    stdx.debug.assert(kernel_size == self.weight.dim(.ky) and @mod(kernel_size, 2) == 1, "Expected a square odd kernel, got {f}", .{self.weight});
    const padding = @divExact(kernel_size - 1, 2);
    const conved = zml.Tensor.conv2d(x.convert(self.weight.dtype()), self.weight, .{
        .padding = &.{ padding, padding, padding, padding },
        .window_strides = &.{ self.stride, self.stride },
    });
    return conved.add(self.bias.broad(conved.shape())).convert(x.dtype());
}
