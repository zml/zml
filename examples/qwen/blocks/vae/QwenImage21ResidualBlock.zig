const zml = @import("zml");

const QwenImage21RMS_norm = @import("./QwenImage21RMS_norm.zig");
const QwenImage21CausalConv3d = @import("./QwenImage21CausalConv3d.zig");

// QwenImage21ResidualBlock(
//   (nonlinearity): SiLU()
//   (norm1): QwenImage21RMS_norm()
//   (conv1): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//   (norm2): QwenImage21RMS_norm()
//   (dropout): Dropout(p=0.0, inplace=False)
//   (conv2): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//   (conv_shortcut): Identity()
// )

pub const QwenImage21ResidualBlock = @This();

norm1: QwenImage21RMS_norm,
conv1: QwenImage21CausalConv3d,
norm2: QwenImage21RMS_norm,
conv2: QwenImage21CausalConv3d,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21ResidualBlock)) void {
    QwenImage21CausalConv3d.unloadBuffers(&self.conv1);
    QwenImage21CausalConv3d.unloadBuffers(&self.conv2);
}
