const std = @import("std");
const zml = @import("zml");

const QwenImage21CausalConv3d = @import("./QwenImage21CausalConv3d.zig");
const QwenImage21MidBlock = @import("./QwenImage21MidBlock.zig");
const QwenImage21ResidualUpBlock = @import("./QwenImage21ResidualUpBlock.zig");
const QwenImage21RMS_norm = @import("./QwenImage21RMS_norm.zig");

const CompileArgs = @import("../../compiler.zig").CompileArgs;

// QwenImage21Decoder3d(
//   (nonlinearity): SiLU()
//   (conv_in): QwenImage21CausalConv3d(64, 1152, kernel_size=(3, 3), stride=(1, 1))
//   (mid_block): QwenImage21MidBlock(
//     (attentions): ModuleList(
//       (0): QwenImage21AttentionBlock(
//         (norm): QwenImage21RMS_norm()
//         (to_qkv): Conv2d(1152, 3456, kernel_size=(1, 1), stride=(1, 1))
//         (proj): Conv2d(1152, 1152, kernel_size=(1, 1), stride=(1, 1))
//       )
//     )
//     (resnets): ModuleList(
//       (0-1): 2 x QwenImage21ResidualBlock(
//         (nonlinearity): SiLU()
//         (norm1): QwenImage21RMS_norm()
//         (conv1): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//         (norm2): QwenImage21RMS_norm()
//         (dropout): Dropout(p=0.0, inplace=False)
//         (conv2): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//         (conv_shortcut): Identity()
//       )
//     )
//   )
//   (up_blocks): ModuleList(
//     (0-1): 2 x QwenImage21ResidualUpBlock(
//       (avg_shortcut): QwenImage21DupUp3D()
//       (resnets): ModuleList(
//         (0-2): 3 x QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(1152, 1152, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): Identity()
//         )
//       )
//       (upsampler): QwenImage21Resample(
//         (resample): Sequential(
//           (0): QwenImage21Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact')
//           (1): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//         )
//         (time_conv): QwenImage21CausalConv3d(1152, 2304, kernel_size=(1, 1), stride=(1, 1))
//       )
//     )
//     (2): QwenImage21ResidualUpBlock(
//       (avg_shortcut): QwenImage21DupUp3D()
//       (resnets): ModuleList(
//         (0): QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(1152, 576, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(576, 576, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): QwenImage21CausalConv3d(1152, 576, kernel_size=(1, 1), stride=(1, 1))
//         )
//         (1-2): 2 x QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(576, 576, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(576, 576, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): Identity()
//         )
//       )
//       (upsampler): QwenImage21Resample(
//         (resample): Sequential(
//           (0): QwenImage21Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact')
//           (1): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//         )
//         (time_conv): QwenImage21CausalConv3d(576, 1152, kernel_size=(1, 1), stride=(1, 1))
//       )
//     )
//     (3): QwenImage21ResidualUpBlock(
//       (avg_shortcut): QwenImage21DupUp3D()
//       (resnets): ModuleList(
//         (0): QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(576, 288, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(288, 288, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): QwenImage21CausalConv3d(576, 288, kernel_size=(1, 1), stride=(1, 1))
//         )
//         (1-2): 2 x QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(288, 288, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(288, 288, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): Identity()
//         )
//       )
//       (upsampler): QwenImage21Resample(
//         (resample): Sequential(
//           (0): QwenImage21Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact')
//           (1): Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//         )
//       )
//     )
//     (4): QwenImage21ResidualUpBlock(
//       (resnets): ModuleList(
//         (0): QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(288, 144, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(144, 144, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): QwenImage21CausalConv3d(288, 144, kernel_size=(1, 1), stride=(1, 1))
//         )
//         (1-2): 2 x QwenImage21ResidualBlock(
//           (nonlinearity): SiLU()
//           (norm1): QwenImage21RMS_norm()
//           (conv1): QwenImage21CausalConv3d(144, 144, kernel_size=(3, 3), stride=(1, 1))
//           (norm2): QwenImage21RMS_norm()
//           (dropout): Dropout(p=0.0, inplace=False)
//           (conv2): QwenImage21CausalConv3d(144, 144, kernel_size=(3, 3), stride=(1, 1))
//           (conv_shortcut): Identity()
//         )
//       )
//     )
//   )
//   (norm_out): QwenImage21RMS_norm()
//   (conv_out): QwenImage21CausalConv3d(144, 4, kernel_size=(3, 3), stride=(1, 1))
// )

const QwenImage21Decoder3d = @This();

conv_in: QwenImage21CausalConv3d,
mid_block: QwenImage21MidBlock,
up_blocks: []QwenImage21ResidualUpBlock,
norm_out: QwenImage21RMS_norm,
conv_out: QwenImage21CausalConv3d,

pub fn load(
    self: *const QwenImage21Decoder3d,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
) !zml.Bufferized(QwenImage21Decoder3d) {
    var buffers = try zml.mem.bufferize(allocator, QwenImage21Decoder3d, self);
    errdefer unloadBuffers(&buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 256 * zml.MiB,
        .parallelism = 16,
    });
    defer loader.deinit();

    try loader.load(io, QwenImage21Decoder3d, self, &buffers, store, platform.shardings.values(), .{});
    try loader.await(io);

    return buffers;
}

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21Decoder3d)) void {
    QwenImage21CausalConv3d.unloadBuffers(&self.conv_in);
    QwenImage21MidBlock.unloadBuffers(&self.mid_block);
    for (self.up_blocks) |*block| {
        QwenImage21ResidualUpBlock.unloadBuffers(block);
    }
    QwenImage21RMS_norm.unloadBuffers(&self.norm_out);
    QwenImage21CausalConv3d.unloadBuffers(&self.conv_out);
}

pub fn forward(self: QwenImage21Decoder3d, input_ids: zml.Tensor) zml.Tensor {
    const conved_in = self.conv_in.forward(input_ids);
    const mid = self.mid_block.forward(conved_in);

    var blocked = mid;
    for (self.up_blocks) |block| {
        blocked = block.forward(blocked);
    }

    const normed_out = self.norm_out.forward(blocked).silu();
    return self.conv_out.forward(normed_out);
}
