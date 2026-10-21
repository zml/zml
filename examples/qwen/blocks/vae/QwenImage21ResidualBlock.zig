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
conv_shortcut: ?QwenImage21CausalConv3d,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21ResidualBlock)) void {
    QwenImage21RMS_norm.unloadBuffers(&self.norm1);
    QwenImage21CausalConv3d.unloadBuffers(&self.conv1);
    QwenImage21RMS_norm.unloadBuffers(&self.norm2);
    QwenImage21CausalConv3d.unloadBuffers(&self.conv2);
    if (self.conv_shortcut) |*conv_shortcut| {
        QwenImage21CausalConv3d.unloadBuffers(conv_shortcut);
    }
}

pub fn forward(self: QwenImage21ResidualBlock, x: zml.Tensor) zml.Tensor {
    const h = if (self.conv_shortcut) |conv_shortcut| conv_shortcut.forward(x) else x;

    const normed1 = self.norm1.forward(x);
    const silu1 = normed1.convert(.f32).silu().convert(x.dtype());
    const conved1 = self.conv1.forward(silu1);

    const normed2 = self.norm2.forward(conved1);
    const silu2 = normed2.convert(.f32).silu().convert(x.dtype());
    const conved2 = self.conv2.forward(silu2);

    return conved2.add(h);
}
