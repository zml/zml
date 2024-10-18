const std = @import("std");

const zml = @import("zml");
const meta = zml.meta;
const asynk = @import("async");
const flags = @import("tigerbeetle/flags");

const transformer = @import("transformer.zig");
const ClipTextTransformer = transformer.ClipTextTransformer;

const log = std.log.scoped(.sdxl);

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain, .{});
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var model_arena = std.heap.ArenaAllocator.init(allocator);
    defer model_arena.deinit();
    const arena = model_arena.allocator();

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/sdxl/cache");
    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform().withCompilationOptions(.{
        .cache_location = "/tmp/zml/sdxl/cache",
        .xla_dump_to = "/tmp/zml/sdxl",
    });

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    const vae_model_path = args[1];
    const unet_model_path = args[2];
    _ = unet_model_path; // autofix
    const prompt_encoder_model_path = args[3];
    const vocab_path = args[4];
    const prompt = args[5];
    const activation_path = args[6];
    var activations = try zml.aio.detectFormatAndOpen(allocator, activation_path);
    defer activations.deinit();

    log.info("Prompt: {s}", .{prompt});

    const prompt_encoder_config: transformer.ClipTextConfig = .{
        .max_position_embeddings = 77,
        .num_heads = 16,
        .layer_norm_eps = 1e-5,
        .hidden_act = .gelu,
    };

    const tokenizer = try loadSdTurboTokenizer(allocator, vocab_path);
    defer tokenizer.deinit();
    const prompt_tokens = try tokenizer.encode(allocator, prompt, .{
        .add_bos = true,
        .add_eos = true,
        .pad_to = prompt_encoder_config.max_position_embeddings,
        .debug = true,
    });
    log.info("prompt tokens: {d}", .{prompt_tokens});
    defer allocator.free(prompt_tokens);

    const prompt_dev = try zml.Buffer.fromSlice(
        platform,
        .{ .s = prompt_encoder_config.max_position_embeddings },
        prompt_tokens,
    );
    _ = prompt_dev; // autofix

    const prompt_encoder_exe = blk: {
        var local_arena = std.heap.ArenaAllocator.init(allocator);
        defer local_arena.deinit();
        const arena_alloc = local_arena.allocator();

        var store = try zml.aio.detectFormatAndOpen(allocator, prompt_encoder_model_path);
        defer store.deinit();

        var prompt_encoder = try zml.aio.populateModelWithPrefix(ClipTextTransformer, arena_alloc, store, "text_model");
        prompt_encoder.init(prompt_encoder_config);

        const prompt_encoder_weights = try zml.aio.loadModelBuffersWithPrefix(ClipTextTransformer, prompt_encoder, store, arena_alloc, platform, "text_model");
        log.info("Loaded prompt encoder from {s}, found {} buffers: {}", .{ prompt_encoder_model_path, store.buffers.count(), prompt_encoder });

        // try testPromptEncoder(platform, activations, prompt_encoder, prompt_encoder_weights);

        const exe = try zml.compileModel(
            allocator,
            prompt_encoder,
            .forward,
            .{zml.Shape.init(.{ .s = prompt_encoder.config.max_position_embeddings }, .i32)},
            platform,
        );

        break :blk try exe.prepare(allocator, prompt_encoder_weights);
    };
    defer prompt_encoder_exe.deinit();

    // const prompt_embed = prompt_encoder_exe.call(.{prompt_dev});
    // _ = prompt_embed; // autofix

    const vae_store = try zml.aio.detectFormatAndOpen(allocator, vae_model_path);
    defer vae_store.deinit();
    var vae = try zml.aio.populateModel(AutoEncoderKl, arena, vae_store);
    const vae_weights = try zml.aio.loadModelBuffers(AutoEncoderKl, vae, vae_store, arena, platform);
    // defer zml.aio.unloadBuffers(vae_weights);

    log.info("Loaded VAE from {s}, found {} buffers.", .{ vae_model_path, vae_store.buffers.count() });
    const group_norm_conf: struct { u32, f32 } = .{ 32, 1e-5 };

    log.info("vae.group_norm -> {}", .{vae.decoder.mid_block.attentions[0].group_norm});
    zml.meta.visit(struct {
        fn cb(conf: struct { u32, f32 }, group_norm: *zml.nn.GroupNorm) void {
            const group_size, const eps = conf;
            group_norm.group_size = group_size;
            group_norm.eps = eps;
        }
    }.cb, group_norm_conf, &vae);
    log.info("vae.group_norm -> {}", .{vae.decoder.mid_block.attentions[0].group_norm});
    try testVae(platform, activations, vae.decoder, vae_weights.decoder);

    // var unet_store = try zml.aio.detectFormatAndOpen(allocator, unet_model_path);
    // defer unet_store.deinit();

    // log.info("Loaded unet from {s}, found {} buffers.", .{ unet_model_path, unet_store.buffers.count() });

    // var unet = try zml.aio.populateModel(Unet2DConditionModel, arena, unet_store);
    // const group_norm_conf: struct { u32, f32 } = .{ 32, 1e-5 };
    // zml.meta.visit(struct {
    //     fn cb(conf: struct { u32, f32 }, group_norm: *GroupNorm) void {
    //         const group_size, const eps = conf;
    //         group_norm.group_size = group_size;
    //         group_norm.eps = eps;
    //     }
    // }.cb, group_norm_conf, &unet);
    // zml.meta.visit(struct {
    //     fn cb(eps: f32, layer_norm: *zml.nn.LayerNorm) void {
    //         layer_norm.eps = eps;
    //     }
    // }.cb, 1e-5, &unet);
    // log.info("Unet: {}", .{unet});

    // const unet_weights = try zml.aio.loadModelBuffers(Unet2DConditionModel, unet, unet_store, arena, platform);
    // _ = unet_weights; // autofix
    // try testUnet(platform, activations, unet, unet_weights);
}

fn testPromptEncoder(platform: zml.Platform, activations: zml.aio.BufferStore, encoder: ClipTextTransformer, encoder_weights: zml.Bufferized(ClipTextTransformer)) !void {
    try zml.testing.testLayer(platform, activations, "text_encoder.text_model.encoder.layers.1.mlp", encoder.encoder.layers[1].mlp, encoder_weights.encoder.layers[1].mlp, 0.05);
    try zml.testing.testLayer(platform, activations, "text_encoder.text_model.encoder.layers.1", encoder.encoder.layers[1], encoder_weights.encoder.layers[1], 0.05);
    try zml.testing.testLayer(platform, activations, "text_encoder.text_model.embeddings", encoder.embeddings, encoder_weights.embeddings, 0.001);
    try zml.testing.testLayer(platform, activations, "text_encoder.text_model", encoder, encoder_weights, 0.1);
}

fn testUnet(platform: zml.Platform, activations: zml.aio.BufferStore, unet: Unet2DConditionModel, unet_weights: zml.Bufferized(Unet2DConditionModel)) !void {
    try zml.testing.testLayer(platform, activations, "unet.conv_in", unet.conv_in, unet_weights.conv_in, 0.005);
    try zml.testing.testLayer(platform, activations, "unet.conv_out", unet.conv_out, unet_weights.conv_out, 0.005);
    try zml.testing.testLayer(platform, activations, "unet.conv_norm_out", unet.conv_norm_out, unet_weights.conv_norm_out, 0.005);

    // Down block
    if (false) {
        try zml.testing.testLayer(platform, activations, "unet.down_blocks.3.resnets.0", unet.down_blocks.@"3".resnets[0], unet_weights.down_blocks.@"3".resnets[0], 0.05);

        try zml.testing.testLayer(platform, activations, "unet.down_blocks.3", unet.down_blocks.@"3", unet_weights.down_blocks.@"3", 0.05);

        try zml.testing.testLayer(platform, activations, "unet.down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0", unet.down_blocks.@"0".attentions[1].transformer_blocks[0].ff.net.@"0", unet_weights.down_blocks.@"0".attentions[1].transformer_blocks[0].ff.net.@"0", 0.05);
        try zml.testing.testLayer(platform, activations, "unet.down_blocks.0.attentions.1.transformer_blocks.0", unet.down_blocks.@"0".attentions[1].transformer_blocks[0], unet_weights.down_blocks.@"0".attentions[1].transformer_blocks[0], 0.05);
        try zml.testing.testLayer(platform, activations, "unet.down_blocks.0.attentions.1", unet.down_blocks.@"0".attentions[1], unet_weights.down_blocks.@"0".attentions[1], 0.05);
        try zml.testing.testLayer(platform, activations, "unet.down_blocks.0", unet.down_blocks.@"0", unet_weights.down_blocks.@"0", 0.05);
    }

    // Middle block.
    if (false) {
        try zml.testing.testLayer(platform, activations, "unet.mid_block", unet.mid_block, unet_weights.mid_block, 0.05);
    }

    // Up blocks
    {
        try zml.testing.testLayer(platform, activations, "unet.up_blocks.0.upsamplers.0", unet.up_blocks.@"0".upsamplers.@"0", unet_weights.up_blocks.@"0".upsamplers.@"0", 0.05);

        // TODO: dig into the precision issues.
        // try zml.testing.testLayer(platform, activations, "unet.up_blocks.0", unet.up_blocks.@"0", unet_weights.up_blocks.@"0", 0.05);
    }
}

fn testVae(platform: zml.Platform, activations: zml.aio.BufferStore, vae: AutoEncoderKl.Decoder, vae_weights: zml.Bufferized(AutoEncoderKl.Decoder)) !void {
    // TODO this seems a bit high
    // try zml.testing.testLayer(platform, activations, "vae.decoder.conv_in", vae.conv_in, vae_weights.conv_in, 0.02);
    // try zml.testing.testLayer(platform, activations, "vae.decoder.conv_out", vae.conv_out, vae_weights.conv_out, 0.02);
    try zml.testing.testLayer(platform, activations, "vae.decoder.mid_block", vae.mid_block, vae_weights.mid_block, 0.02);
}

pub const AutoEncoderKl = struct {
    decoder: Decoder,
    post_quant_conv: Conv2dSame,

    const Decoder = struct {
        conv_in: Conv2dSame,
        // up_blocks:
        mid_block: UNetMidBlock2D,
        conv_norm_out: zml.nn.GroupNorm,
        conv_act: zml.nn.Activation = .silu,
        conv_out: Conv2dSame,
    };

    pub const UNetMidBlock2D = struct {
        attentions: []Attention,
        resnets: []ResnetBlock2D,

        pub fn forward(self: UNetMidBlock2D, images: zml.Tensor) zml.Tensor {
            const x = images.withTags(.{ .b, .channels, .height, .width });
            var hidden = self.resnets[0].forward(x, null);

            for (self.attentions, self.resnets[1..]) |attn, resnet| {
                hidden = attn.forward(hidden);
                hidden = resnet.forward(hidden, null);
            }
            return hidden;
        }
    };

    pub const Attention = struct {
        group_norm: zml.nn.GroupNorm,
        to_q: zml.nn.Linear,
        to_k: zml.nn.Linear,
        to_v: zml.nn.Linear,
        to_out: struct { zml.nn.Linear },

        pub fn forward(self: Attention, input: zml.Tensor) zml.Tensor {
            const x0 = if (input.shape().isFullyTagged()) input else input.withTags(.{ .b, .channels, .height, .width });
            var x = x0.merge(.{ .hw = .{ .height, .width } });
            x = zml.nn.groupNorm(x, .channels, .hw, self.group_norm);
            x = x.contiguous(.{.channels});
            // log.info("x: {}", .{x});

            const ctx = x.rename(.{ .hw = .k });
            const nh = 1;
            const q = zml.call(self.to_q, .forward, .{x.rename(.{ .hw = .q })}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            // log.info("q: {}", .{q});
            const k = zml.call(self.to_k, .forward, .{ctx}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            // log.info("k: {}", .{k});
            const v = zml.call(self.to_v, .forward, .{ctx}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            // log.info("v: {}", .{v});

            const attn_output = zml.nn.sdpa(q, k, v, .{});
            const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .hw, .d = .channels });
            var out = zml.call(self.to_out.@"0", .forward, .{attn});

            out = out.splitAxis(.hw, .{ .height = x0.dim(.height), .width = .auto });
            out = out.transpose(x0.shape());
            return out;
        }
    };
};

pub const Unet2DConditionModel = struct {
    conv_in: Conv2dSame,

    // time_proj
    // time_embedding
    down_blocks: DownBlocks,
    mid_block: UNetMidBlock2DCrossAttn,
    up_blocks: UpBlocks,

    conv_norm_out: zml.nn.GroupNorm,
    conv_act: zml.nn.Activation = .silu,
    conv_out: Conv2dSame,
};

pub const Conv2dSame = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn forward(self: Conv2dSame, input: zml.Tensor) zml.Tensor {
        const x = input.withPartialTags(.{ .channels, .height, .width });
        log.warn("Conv2dSame({}, {})", .{ self.weight, input });
        const y = zml.Tensor.conv2d(x, self.weight, .{
            .padding = &.{ 1, 1, 1, 1 },
            .input_feature_dimension = x.axis(.channels),
            .input_spatial_dimensions = &.{ x.axis(.width), x.axis(.height) },
        });
        return y.add(self.bias.withTags(.{.channels}).broad(y.shape()));
    }
};

pub fn groupNorm2D(weights: zml.nn.GroupNorm, input: zml.Tensor) zml.Tensor {
    // log.warn("GroupNorm({}, {})", .{ self.weight, input });
    const x = input.withPartialTags(.{ .channels, .height, .width }).merge(.{ .hw = .{ .height, .width } });

    const x_norm = zml.nn.groupNorm(x, .channels, .hw, weights);
    return x_norm.reshape(input.shape());
}

pub const DownBlocks = struct {
    @"0": CrossAttnDownBlock2D,
    @"1": CrossAttnDownBlock2D,
    @"2": CrossAttnDownBlock2D,
    @"3": DownBlock2D,

    pub fn forward(self: DownBlocks, images: zml.Tensor, time_embedding: zml.Tensor, encoder_hidden_states: zml.Tensor) zml.Tensor {
        var hidden = images;

        hidden = self.@"0".forward(hidden, time_embedding, encoder_hidden_states);
        hidden = self.@"1".forward(hidden, time_embedding, encoder_hidden_states);
        hidden = self.@"2".forward(hidden, time_embedding, encoder_hidden_states);
        hidden = self.@"3".forward(hidden, time_embedding);

        return hidden;
    }

    pub const CrossAttnDownBlock2D = struct {
        attentions: []Transformer2DModel,
        resnets: []ResnetBlock2D,
        downsamplers: struct { struct { conv: Conv2dHalf } },

        pub fn forward(
            self: CrossAttnDownBlock2D,
            images: zml.Tensor,
            time_embedding: zml.Tensor,
            encoder_hidden_states: zml.Tensor,
        ) zml.Tensor {
            var hidden = images;
            for (self.resnets, self.attentions) |resnet, attn| {
                hidden = zml.call(resnet, .forward, .{ hidden, time_embedding });
                hidden = zml.call(attn, .forward, .{ hidden, encoder_hidden_states });
            }
            hidden = zml.call(self.downsamplers[0].conv, .forward, .{hidden});
            return hidden;
        }
    };

    pub const DownBlock2D = struct {
        resnets: []ResnetBlock2D,

        pub fn forward(self: DownBlock2D, images: zml.Tensor, time_embedding: zml.Tensor) zml.Tensor {
            var y = images;
            for (self.resnets) |resnet| {
                y = zml.call(resnet, .forward, .{ y, time_embedding });
            }
            return y;
        }
    };

    pub const Conv2dHalf = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn forward(self: Conv2dHalf, input: zml.Tensor) zml.Tensor {
            const x = input.withPartialTags(.{ .channels, .height, .width });
            const y = zml.Tensor.conv2d(
                x,
                self.weight,
                .{
                    .padding = &.{ 1, 1, 1, 1 },
                    .window_strides = &.{ 2, 2 },
                },
            );
            return y.add(self.bias.withTags(.{.channels}).broad(y.shape()));
        }
    };
};

pub const UNetMidBlock2DCrossAttn = struct {
    attentions: []Transformer2DModel,
    resnets: []ResnetBlock2D,

    pub fn forward(self: UNetMidBlock2DCrossAttn, images: zml.Tensor, time_embedding: zml.Tensor, encoder_hidden_states: zml.Tensor) zml.Tensor {
        var hidden = self.resnets[0].forward(images, time_embedding);

        for (self.attentions, self.resnets[1..]) |attn, resnet| {
            hidden = attn.forward(hidden, encoder_hidden_states);
            hidden = resnet.forward(hidden, time_embedding);
        }
        return hidden;
    }
};

pub const UpBlocks = struct {
    @"0": UpBlock2D,
    @"1": CrossAttnUpBlock2D,
    @"2": CrossAttnUpBlock2D,
    @"3": CrossAttnUpBlock2D,

    pub fn forward(self: UpBlocks, images: zml.Tensor, time_embedding: zml.Tensor, encoder_hidden_states: zml.Tensor) zml.Tensor {
        var hidden = images;

        hidden = self.@"0".forward(hidden, time_embedding);
        hidden = self.@"1".forward(hidden, time_embedding, encoder_hidden_states);
        hidden = self.@"2".forward(hidden, time_embedding, encoder_hidden_states);
        hidden = self.@"3".forward(hidden, time_embedding, encoder_hidden_states);

        return hidden;
    }

    const UpBlock2D = struct {
        resnets: []ResnetBlock2D,
        upsamplers: struct { Upsampler },

        // TODO: zml handle const slice as inputs
        pub fn forward(
            self: UpBlock2D,
            images: zml.Tensor,
            time_embedding: zml.Tensor,
            downscaled_images: []zml.Tensor,
        ) zml.Tensor {
            var hidden = images;
            const n = self.resnets.len;
            zml.meta.assert(n == downscaled_images.len, "this UpBlock2D expects {} downscaled images, got: {}", .{ n, downscaled_images.len });
            for (self.resnets, 0..) |resnet, i| {
                hidden = zml.Tensor.concatenate(&.{ hidden, downscaled_images[n - 1 - i] }, 1);
                hidden = zml.call(resnet, .forward, .{ hidden, time_embedding });
            }

            hidden = zml.call(self.upsamplers[0], .forward, .{hidden});
            return hidden;
        }
    };

    const CrossAttnUpBlock2D = struct {};

    const Upsampler = struct {
        conv: Conv2dSame,
        pub fn forward(self: Upsampler, images: zml.Tensor) zml.Tensor {
            var hidden = images;
            hidden = zml.nn.nearest(hidden, &.{ 2, 2 });
            hidden = self.conv.forward(hidden);
            return hidden;
        }
    };
};

pub const ResnetBlock2D = struct {
    norm1: zml.nn.GroupNorm,
    conv1: Conv2dSame,
    time_emb_proj: ?zml.nn.Linear,
    norm2: zml.nn.GroupNorm,
    conv2: Conv2dSame,
    nonlinearity: zml.nn.Activation = .silu,
    conv_shortcut: ?struct { weight: zml.Tensor },
    // TODO: output_scale_factor ?

    pub fn forward(self: ResnetBlock2D, images: zml.Tensor, time_embedding: ?zml.Tensor) zml.Tensor {
        const x = images.withPartialTags(.{ .batch, .channels, .height, .width });
        var hidden = groupNorm2D(self.norm1, x);
        hidden = self.nonlinearity.forward(hidden);
        hidden = self.conv1.forward(hidden);

        if (self.time_emb_proj) |time_emb_proj| {
            var t_emb = time_embedding.?.withPartialTags(.{.channels});
            t_emb = self.nonlinearity.forward(t_emb);
            t_emb = time_emb_proj.forward(t_emb);
            hidden = hidden.add(t_emb.appendAxes(.{ .height, .width }).broad(hidden.shape()));
        } else {
            zml.meta.assert(time_embedding == null, "This model doesn' support time embeddings", .{});
        }

        hidden = groupNorm2D(self.norm2, hidden);
        hidden = self.nonlinearity.forward(hidden);
        hidden = self.conv2.forward(hidden);

        return if (self.conv_shortcut) |conv_shortcut| {
            const kernel = conv_shortcut.weight.withTags(.{ .channels_out, .channels, .w, .h });
            const shortcut = kernel.squeeze(.w).squeeze(.h);
            const x2 = x.dot(shortcut, .{.channels}).rename(.{ .channels_out = .channels }).contiguous(.{ .channels, .height, .width });
            return x2.add(hidden);
        } else x.add(hidden);
    }
};

pub const Transformer2DModel = struct {
    norm: zml.nn.GroupNorm,
    proj_in: zml.nn.Linear,
    transformer_blocks: []BasicTransformerBlock,
    proj_out: zml.nn.Linear,

    pub fn forward(self: Transformer2DModel, images: zml.Tensor, encoder_hidden_states: zml.Tensor) zml.Tensor {
        const x = images.withPartialTags(.{ .b, .channels, .height, .width });
        var hidden = x;
        hidden = groupNorm2D(self.norm, hidden);
        hidden = hidden.merge(.{ .hw = .{ .height, .width } }).contiguous(.{.channels});
        hidden = self.proj_in.forward(hidden);

        for (self.transformer_blocks) |block| {
            hidden = zml.call(block, .forward, .{ hidden, encoder_hidden_states });
        }

        hidden = self.proj_out.forward(hidden);
        hidden = hidden.contiguous(.{.hw}).splitAxis(.hw, .{ .height = x.dim(.height), .width = .auto });
        return hidden.add(x);
    }
};

pub const BasicTransformerBlock = struct {
    norm1: zml.nn.LayerNorm,
    attn1: Attention,
    norm2: zml.nn.LayerNorm,
    attn2: Attention,
    norm3: zml.nn.LayerNorm,
    ff: struct { net: struct { GEGLU, void, zml.nn.Linear } },

    const Attention = struct {
        to_q: zml.nn.Linear,
        to_k: zml.nn.Linear,
        to_v: zml.nn.Linear,
        to_out: struct { zml.nn.Linear },

        pub fn forward(self: Attention, input: zml.Tensor, context: zml.Tensor) zml.Tensor {
            const x = if (input.shape().isFullyTagged()) input else input.withTags(.{ .b, .hw, .channels });
            const ctx = if (input.shape().isFullyTagged()) context else context.withTags(.{ .b, .k, .channels });
            const nh = 5;
            const q = zml.call(self.to_q, .forward, .{x.rename(.{ .hw = .q })}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            const k = zml.call(self.to_k, .forward, .{ctx}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            const v = zml.call(self.to_v, .forward, .{ctx}).splitAxis(.channels, .{ .h = nh, .hd = .auto });

            const attn_output = zml.nn.sdpa(q, k, v, .{});
            const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .hw, .d = .channels });
            const out = zml.call(self.to_out.@"0", .forward, .{attn});
            return out;
        }
    };

    const GEGLU = struct {
        proj: zml.nn.Linear,

        pub fn forward(self: GEGLU, x: zml.Tensor) zml.Tensor {
            const hidden, const gate = self.proj.forward(x).chunkExact(-1, 2);
            return hidden.mul(gate.gelu());
        }
    };

    pub fn forward(self: BasicTransformerBlock, images: zml.Tensor, encoder_hidden_state: zml.Tensor) zml.Tensor {
        var hidden = images;
        {
            // First self-attention
            var y = groupNorm2D(self.norm1, hidden);
            y = self.attn1.forward(y, y.withPartialTags(.{ .k, .channels }));
            hidden = hidden.add(y);
        }

        {
            // Then cross-attention
            var y = groupNorm2D(self.norm2, hidden);
            y = self.attn2.forward(y, encoder_hidden_state.withTags(.{ .b, .k, .channels }));
            hidden = hidden.add(y);
        }

        {
            var y = groupNorm2D(self.norm3, hidden);
            y = self.ff.net.@"0".forward(y); // GEGLU
            y = self.ff.net.@"2".forward(y); // Linear
            hidden = hidden.add(y);
        }
        return hidden;
    }
};

pub fn loadSdTurboTokenizer(allocator: std.mem.Allocator, vocab_json_path: []const u8) !zml.tokenizer.Tokenizer {
    const file = try std.fs.cwd().openFile(vocab_json_path, .{});
    defer file.close();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const file_content = try file.readToEndAlloc(arena, 32 * 1024 * 1024);

    const info = try std.json.parseFromSliceLeaky(std.json.Value, arena, file_content, .{
        .duplicate_field_behavior = .use_last,
    });

    const json_vocab = switch (info) {
        .object => |obj| obj,
        else => return error.InvalidFormat,
    };

    const n: u32 = @intCast(json_vocab.count() + 1);
    const normalizer: zml.tokenizer.Normalizer = .{ .flags = .{
        .remove_extra_whitespaces = true,
        .add_dummy_prefix = false,
        .add_dummy_suffix = true,
        .lower_case_ascii = true,
        .split_on_punct_ascii = true,
    } };
    var tokenizer = try zml.tokenizer.Tokenizer.init(allocator, n, 256, normalizer, undefined, true);
    const tok_alloc = tokenizer.arena_state.allocator();
    for (json_vocab.keys(), json_vocab.values()) |key, value| {
        const idx: u32 = switch (value) {
            .integer => |i| @intCast(i),
            else => return error.InvalidFormat,
        };
        // Replaced "</w>" suffix by " "
        var word = try tok_alloc.dupe(u8, key);
        if (std.mem.endsWith(u8, word, "</w>")) {
            word = word[0 .. word.len - "</w>".len + 1];
            word[word.len - 1] = ' ';
        }
        tokenizer.addOwnedTokenByIndex(idx, @floatFromInt(n - idx), word);
    }
    tokenizer.addOwnedTokenByIndex(n - 1, 0, try tok_alloc.dupe(u8, " "));

    tokenizer.special_tokens = .{
        .bos = tokenizer.token_lookup.get("<|startoftext|>").?,
        .eos = tokenizer.token_lookup.get("<|endoftext|>").?,
        .unk = tokenizer.token_lookup.get("<|endoftext|>").?,
        .pad = tokenizer.token_lookup.get("!").?,
    };
    return tokenizer;
}
