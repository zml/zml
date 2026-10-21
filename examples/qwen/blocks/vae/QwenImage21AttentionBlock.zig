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
