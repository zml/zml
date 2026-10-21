const std = @import("std");
const zml = @import("zml");

const QwenImage21AttentionBlock = @import("./QwenImage21AttentionBlock.zig");
const QwenImage21ResidualBlock = @import("./QwenImage21ResidualBlock.zig");

// QwenImage21MidBlock(
//   (attentions): ModuleList(
//     (0): QwenImage21AttentionBlock(
//       (norm): QwenImage21RMS_norm()
//       (to_qkv): Conv2d(1152, 3456, kernel_size=(1, 1), stride=(1, 1))
//       (proj): Conv2d(1152, 1152, kernel_size=(1, 1), stride=(1, 1))
//     )
//   )
//   (resnets): ModuleList(
//     (0-1): 2 x QwenImage21ResidualBlock(
//       (nonlinearity): SiLU()
//       (norm1): QwenImage21RMS_norm()
//       (conv1): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//       (norm2): QwenImage21RMS_norm()
//       (dropout): Dropout(p=0.0, inplace=False)
//       (conv2): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//       (conv_shortcut): Identity()
//     )
//   )
// )

pub const QwenImage21MidBlock = @This();

attentions: QwenImage21AttentionBlock, // Only one
resnets: []QwenImage21ResidualBlock,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21MidBlock)) void {
    QwenImage21AttentionBlock.unloadBuffers(&self.attentions);
    for (self.resnets) |*resnet| {
        QwenImage21ResidualBlock.unloadBuffers(resnet);
    }
}
