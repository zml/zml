const std = @import("std");
const log = std.log;

const zml = @import("zml");
const block = @import("../blocks.zig");

const VaeStore = @This();

const MID_BLOCK_RES_COUNT = 2;
const UP_RESIDUAL_BLOCK_COUNT: usize = 5;
const UP_RESNET_BLOCK_COUNT: usize = 3;

store: zml.io.TensorStore = undefined,
registry: zml.safetensors.TensorRegistry = undefined,

model: block.AutoencoderKLQwenImage21,
model_buffer: zml.Bufferized(block.AutoencoderKLQwenImage21),

// From vae/diffusion_pytorch_model.safetensors
//  decoder.conv_in.bias                                            (1152,)               F32
//  decoder.conv_in.weight                                          (1152, 64, 3, 3)      F32

//  decoder.mid_block.attentions.0.norm.gamma                       (1152, 1, 1)          F32
//  decoder.mid_block.attentions.0.proj.bias                        (1152,)               F32
//  decoder.mid_block.attentions.0.proj.weight                      (1152, 1152, 1, 1)    F32
//  decoder.mid_block.attentions.0.to_qkv.bias                      (3456,)               F32
//  decoder.mid_block.attentions.0.to_qkv.weight                    (3456, 1152, 1, 1)    F32
//  decoder.mid_block.resnets.0.conv1.bias                          (1152,)               F32
//  decoder.mid_block.resnets.0.conv1.weight                        (1152, 1152, 3, 3)    F32
//  decoder.mid_block.resnets.0.conv2.bias                          (1152,)               F32
//  decoder.mid_block.resnets.0.conv2.weight                        (1152, 1152, 3, 3)    F32
//  decoder.mid_block.resnets.0.norm1.gamma                         (1152, 1, 1, 1)       F32
//  decoder.mid_block.resnets.0.norm2.gamma                         (1152, 1, 1, 1)       F32

//  decoder.up_blocks.0.resnets.0.conv1.bias                        (1152,)               F32
//  decoder.up_blocks.0.resnets.0.conv1.weight                      (1152, 1152, 3, 3)    F32
//  decoder.up_blocks.0.resnets.0.conv2.bias                        (1152,)               F32
//  decoder.up_blocks.0.resnets.0.conv2.weight                      (1152, 1152, 3, 3)    F32
//  decoder.up_blocks.0.resnets.0.norm1.gamma                       (1152, 1, 1, 1)       F32
//  decoder.up_blocks.0.resnets.0.norm2.gamma                       (1152, 1, 1, 1)       F32
//  decoder.up_blocks.0.resnets.1.conv1.bias                        (1152,)               F32
//  decoder.up_blocks.0.upsampler.resample.1.bias                   (1152,)               F32
//  decoder.up_blocks.0.upsampler.resample.1.weight                 (1152, 1152, 3, 3)    F32
//  decoder.up_blocks.0.upsampler.time_conv.bias                    (2304,)               F32
//  decoder.up_blocks.0.upsampler.time_conv.weight                  (2304, 1152, 1, 1)    F32

//  decoder.norm_out.gamma                                          (144, 1, 1, 1)        F32

//  decoder.conv_out.bias                                           (4,)                  F32
//  decoder.conv_out.weight                                         (4, 144, 3, 3)        F32

pub fn load_tensors(
    self: *VaeStore,
    allocator: std.mem.Allocator,
    io: std.Io,
    folder: []const u8,
) !void {
    var transformer_buffer: [256]u8 = undefined;
    const transformer_model_registry_path = try std.fmt.bufPrint(&transformer_buffer, "{s}/{s}", .{ folder, "vae/diffusion_pytorch_model.safetensors" });
    self.registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, transformer_model_registry_path);
    log.info("Transformer: Found {} tensors in {s}", .{ self.registry.tensors.count(), transformer_model_registry_path });

    self.store = zml.io.TensorStore.fromRegistry(allocator, &self.registry);
    const store_view = self.store.view().withPrefix("decoder");
    const conv_in = block.QwenImage21CausalConv3d{
        .conv = block.Conv2d{
            .weight = store_view.createTensor("conv_in.weight", .{ .bs, .c, .kx, .ky }, .replicated),
            .bias = store_view.createTensor("conv_in.bias", .{.c}, .replicated),
            .kernel_size = 3,
            .padding = 1,
        },
    };

    const mid_block = mid_block: {
        const mid_block_view = store_view.withPrefix("mid_block");

        const attentions_view = mid_block_view.withPrefix("attentions.0");
        const attentions = block.QwenImage21AttentionBlock{
            .norm = block.QwenImage21RMS_norm{
                .gamma = attentions_view.createTensor("norm.gamma", .{ .c, .kx, .ky }, .replicated),
                .tag = zml.Shape.toTag(.c),
            },
            .to_qkv = block.Conv2d{
                .weight = attentions_view.createTensor("to_qkv.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                .bias = attentions_view.createTensor("to_qkv.bias", .{.c}, .replicated),
                .kernel_size = 1,
            },
            .proj = block.Conv2d{
                .weight = attentions_view.createTensor("proj.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                .bias = attentions_view.createTensor("proj.bias", .{.c}, .replicated),
                .kernel_size = 1,
            },
        };

        const resnets: []block.QwenImage21ResidualBlock =
            try allocator.alloc(block.QwenImage21ResidualBlock, MID_BLOCK_RES_COUNT);
        for (0..MID_BLOCK_RES_COUNT) |i| {
            const resnet_view = mid_block_view.withPrefix("resnets").withLayer(i);
            resnets[i] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = null,
            };
        }

        break :mid_block block.QwenImage21MidBlock{
            .attentions = attentions,
            .resnets = resnets,
        };
    };

    const up_blocks = try load_up_blocks(allocator, store_view);

    const norm_out = block.QwenImage21RMS_norm{
        .gamma = store_view.createTensor("norm_out.gamma", .{ .c, .t, .kx, .ky }, .replicated),
        .tag = zml.Shape.toTag(.c),
    };
    const conv_out = block.QwenImage21CausalConv3d{
        .conv = block.Conv2d{
            .weight = store_view.createTensor("conv_out.weight", .{ .bs, .c, .kx, .ky }, .replicated),
            .bias = store_view.createTensor("conv_out.bias", .{.c}, .replicated),
            .kernel_size = 3,
            .padding = 1,
        },
    };

    const decoder = block.QwenImage21Decoder3d{
        .conv_in = conv_in,
        .mid_block = mid_block,
        .up_blocks = up_blocks,
        .norm_out = norm_out,
        .conv_out = conv_out,
    };

    const post_quant_conv = block.QwenImage21CausalConv3d{
        .conv = block.Conv2d{
            .weight = self.store.view().createTensor("post_quant_conv.weight", .{ .bs, .c, .kx, .ky }, .replicated),
            .bias = self.store.view().createTensor("post_quant_conv.bias", .{.c}, .replicated),
            .kernel_size = 1,
        },
    };
    self.model = block.AutoencoderKLQwenImage21{ .decoder = decoder, .post_quant_conv = post_quant_conv };
}

fn load_up_blocks(
    allocator: std.mem.Allocator,
    store_view: zml.io.TensorStore.View,
) ![]block.QwenImage21ResidualUpBlock {
    const up_blocks: []block.QwenImage21ResidualUpBlock =
        try allocator.alloc(block.QwenImage21ResidualUpBlock, UP_RESIDUAL_BLOCK_COUNT);

    up_blocks[0] = b0: {
        const resup_store = store_view.withPrefix("up_blocks").withLayer(0);

        const resnets: []block.QwenImage21ResidualBlock =
            try allocator.alloc(block.QwenImage21ResidualBlock, UP_RESNET_BLOCK_COUNT);
        for (0..UP_RESNET_BLOCK_COUNT) |i| {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(i);
            resnets[i] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = null,
            };
        }

        const upsampler_view = resup_store.withPrefix("upsampler");
        const upsampler = block.QwenImage21Resample{
            .upsample = block.QwenImage21Upsample{},
            .conv = block.Conv2d{
                .weight = upsampler_view.createTensor("resample.1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                .bias = upsampler_view.createTensor("resample.1.bias", .{.c}, .replicated),
                .kernel_size = 3,
                .padding = 1,
            },
            // .time_conv = block.QwenImage21CausalConv3d{
            //     .conv = block.Conv2d{
            //         .weight = upsampler_view.createTensor("time_conv.weight", .{ .bs, .c, .kx, .ky }, .replicated),
            //         .bias = upsampler_view.createTensor("time_conv.bias", .{.c}, .replicated),
            //         .kernel_size = 1,
            //     },
            // },
        };

        break :b0 block.QwenImage21ResidualUpBlock{
            .avg_shortcut = block.QwenImage21DupUp3D{ .out_channels = 1152, .factor_t = 2, .factor_s = 2 },
            .resnets = resnets,
            .upsampler = upsampler,
        };
    };

    up_blocks[1] = b1: {
        const resup_store = store_view.withPrefix("up_blocks").withLayer(1);

        const resnets: []block.QwenImage21ResidualBlock =
            try allocator.alloc(block.QwenImage21ResidualBlock, UP_RESNET_BLOCK_COUNT);
        for (0..UP_RESNET_BLOCK_COUNT) |i| {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(i);
            resnets[i] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = null,
            };
        }

        const upsampler_view = resup_store.withPrefix("upsampler");
        const upsampler = block.QwenImage21Resample{
            .upsample = block.QwenImage21Upsample{},
            .conv = block.Conv2d{
                .weight = upsampler_view.createTensor("resample.1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                .bias = upsampler_view.createTensor("resample.1.bias", .{.c}, .replicated),
                .kernel_size = 3,
                .padding = 1,
            },
            // .time_conv = block.QwenImage21CausalConv3d{
            //     .conv = block.Conv2d{
            //         .weight = upsampler_view.createTensor("time_conv.weight", .{ .bs, .c, .kx, .ky }, .replicated),
            //         .bias = upsampler_view.createTensor("time_conv.bias", .{.c}, .replicated),
            //         .kernel_size = 1,
            //     },
            // },
        };

        break :b1 block.QwenImage21ResidualUpBlock{
            .avg_shortcut = block.QwenImage21DupUp3D{ .out_channels = 1152, .factor_t = 2, .factor_s = 2 },
            .resnets = resnets,
            .upsampler = upsampler,
        };
    };

    up_blocks[2] = b2: {
        const resup_store = store_view.withPrefix("up_blocks").withLayer(2);

        const resnets: []block.QwenImage21ResidualBlock =
            try allocator.alloc(block.QwenImage21ResidualBlock, UP_RESNET_BLOCK_COUNT);

        {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(0);
            resnets[0] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv_shortcut.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv_shortcut.bias", .{.c}, .replicated),
                        .kernel_size = 1,
                    },
                },
            };
        }

        for (1..UP_RESNET_BLOCK_COUNT) |i| {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(i);
            resnets[i] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = null,
            };
        }

        const upsampler_view = resup_store.withPrefix("upsampler");
        const upsampler = block.QwenImage21Resample{
            .upsample = block.QwenImage21Upsample{},
            .conv = block.Conv2d{
                .weight = upsampler_view.createTensor("resample.1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                .bias = upsampler_view.createTensor("resample.1.bias", .{.c}, .replicated),
                .kernel_size = 3,
                .padding = 1,
            },
            // .time_conv = block.QwenImage21CausalConv3d{
            //     .conv = block.Conv2d{
            //         .weight = upsampler_view.createTensor("time_conv.weight", .{ .bs, .c, .kx, .ky }, .replicated),
            //         .bias = upsampler_view.createTensor("time_conv.bias", .{.c}, .replicated),
            //         .kernel_size = 1,
            //     },
            // },
        };

        break :b2 block.QwenImage21ResidualUpBlock{
            .avg_shortcut = block.QwenImage21DupUp3D{ .out_channels = 576, .factor_t = 2, .factor_s = 2 },
            .resnets = resnets,
            .upsampler = upsampler,
        };
    };

    up_blocks[3] = b3: {
        const resup_store = store_view.withPrefix("up_blocks").withLayer(3);

        const resnets: []block.QwenImage21ResidualBlock =
            try allocator.alloc(block.QwenImage21ResidualBlock, UP_RESNET_BLOCK_COUNT);

        {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(0);
            resnets[0] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv_shortcut.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv_shortcut.bias", .{.c}, .replicated),
                        .kernel_size = 1,
                    },
                },
            };
        }

        for (1..UP_RESNET_BLOCK_COUNT) |i| {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(i);
            resnets[i] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = null,
            };
        }

        const upsampler_view = resup_store.withPrefix("upsampler");
        const upsampler = block.QwenImage21Resample{
            .upsample = block.QwenImage21Upsample{},
            .conv = block.Conv2d{
                .weight = upsampler_view.createTensor("resample.1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                .bias = upsampler_view.createTensor("resample.1.bias", .{.c}, .replicated),
                .kernel_size = 3,
                .padding = 1,
            },
        };

        break :b3 block.QwenImage21ResidualUpBlock{
            .avg_shortcut = block.QwenImage21DupUp3D{ .out_channels = 288, .factor_t = 1, .factor_s = 2 },
            .resnets = resnets,
            .upsampler = upsampler,
        };
    };

    up_blocks[4] = b4: {
        const resup_store = store_view.withPrefix("up_blocks").withLayer(4);

        const resnets: []block.QwenImage21ResidualBlock =
            try allocator.alloc(block.QwenImage21ResidualBlock, UP_RESNET_BLOCK_COUNT);

        {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(0);
            resnets[0] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv_shortcut.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv_shortcut.bias", .{.c}, .replicated),
                        .kernel_size = 1,
                    },
                },
            };
        }

        for (1..UP_RESNET_BLOCK_COUNT) |i| {
            const resnet_view = resup_store.withPrefix("resnets").withLayer(i);
            resnets[i] = block.QwenImage21ResidualBlock{
                .norm1 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm1.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv1 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv1.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv1.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .norm2 = block.QwenImage21RMS_norm{
                    .gamma = resnet_view.createTensor("norm2.gamma", .{ .c, .t, .kx, .ky }, .replicated),
                    .tag = zml.Shape.toTag(.c),
                },
                .conv2 = block.QwenImage21CausalConv3d{
                    .conv = block.Conv2d{
                        .weight = resnet_view.createTensor("conv2.weight", .{ .bs, .c, .kx, .ky }, .replicated),
                        .bias = resnet_view.createTensor("conv2.bias", .{.c}, .replicated),
                        .kernel_size = 3,
                        .padding = 1,
                    },
                },
                .conv_shortcut = null,
            };
        }

        break :b4 block.QwenImage21ResidualUpBlock{
            .avg_shortcut = null,
            .resnets = resnets,
            .upsampler = null,
        };
    };

    return up_blocks;
}

pub fn load_buffers(self: *VaeStore, allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !void {
    self.model_buffer = try self.model.load(allocator, io, platform, &self.store);
}

pub fn deinit(self: *VaeStore, allocator: std.mem.Allocator) void {
    block.AutoencoderKLQwenImage21.unloadBuffers(&self.model_buffer);

    for (self.model.decoder.up_blocks) |b| {
        allocator.free(b.resnets);
    }
    allocator.free(self.model.decoder.up_blocks);
    allocator.free(self.model.decoder.mid_block.resnets);

    for (self.model_buffer.decoder.up_blocks) |b| {
        allocator.free(b.resnets);
    }
    allocator.free(self.model_buffer.decoder.up_blocks);
    allocator.free(self.model_buffer.decoder.mid_block.resnets);

    self.store.deinit();
    self.registry.deinit();
}
