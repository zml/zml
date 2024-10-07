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
    _ = vae_model_path; // autofix
    const unet_model_path = args[2];
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

    const prompt_embed = prompt_encoder_exe.call(.{prompt_dev});
    _ = prompt_embed; // autofix

    // var vae_weights = try zml.aio.detectFormatAndOpen(allocator, vae_model_path);
    // defer vae_weights.deinit();

    // log.info("Loaded VAE from {s}, found {} buffers.", .{vae_model_path, vae_weights.buffers.count()});

    var unet_store = try zml.aio.detectFormatAndOpen(allocator, unet_model_path);
    defer unet_store.deinit();

    log.info("Loaded unet from {s}, found {} buffers.", .{ unet_model_path, unet_store.buffers.count() });

    var unet = try zml.aio.populateModel(Unet2DConditionModel, arena, unet_store);
    unet.conv_norm_out.group_size = 32;
    unet.conv_norm_out.eps = 1e-5;
    const group_norm_conf: struct { u32, f32 } = .{ 32, 1e-5 };
    zml.meta.visit(struct {
        fn cb(conf: struct { u32, f32 }, group_norm: *GroupNorm) void {
            const group_size, const eps = conf;
            group_norm.group_size = group_size;
            group_norm.eps = eps;
        }
    }.cb, group_norm_conf, &unet);
    log.info("Unet: {}", .{unet});

    const unet_weights = try zml.aio.loadModelBuffers(Unet2DConditionModel, unet, unet_store, arena, platform);
    try testUnet(platform, activations, unet, unet_weights);
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

    try zml.testing.testLayer(platform, activations, "unet.down_blocks.3.resnets.0", unet.down_blocks.@"3".resnets[0], unet_weights.down_blocks.@"3".resnets[0], 0.05);

    try zml.testing.testLayer(platform, activations, "unet.down_blocks.3", unet.down_blocks.@"3", unet_weights.down_blocks.@"3", 0.05);

    try zml.testing.testLayer(platform, activations, "unet.down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0", unet.down_blocks.@"0".attentions[1].transformer_blocks[0].ff.net.@"0", unet_weights.down_blocks.@"0".attentions[1].transformer_blocks[0].ff.net.@"0", 0.05);
    try zml.testing.testLayer(platform, activations, "unet.down_blocks.0.attentions.1.transformer_blocks.0", unet.down_blocks.@"0".attentions[1].transformer_blocks[0], unet_weights.down_blocks.@"0".attentions[1].transformer_blocks[0], 0.05);
    try zml.testing.testLayer(platform, activations, "unet.down_blocks.0.attentions.1", unet.down_blocks.@"0".attentions[1], unet_weights.down_blocks.@"0".attentions[1], 0.05);
    try zml.testing.testLayer(platform, activations, "unet.down_blocks.0", unet.down_blocks.@"0", unet_weights.down_blocks.@"0", 0.05);
}

pub const Unet2DConditionModel = struct {
    conv_in: Conv2dSame,

    // time_proj
    // time_embedding
    down_blocks: DownBlocks,
    // up_blocks
    // mid_block

    conv_norm_out: GroupNorm,
    conv_act: zml.nn.Activation = .silu,
    conv_out: Conv2dSame,
};

pub const Conv2dSame = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn forward(self: Conv2dSame, input: zml.Tensor) zml.Tensor {
        const x = input.withPartialTags(.{ .channels, .width, .height });
        const y = zml.Tensor.conv2d(
            x,
            self.weight,
            .{ .padding = &.{ 1, 1, 1, 1 } },
        );
        return y.add(self.bias.withTags(.{.channels}).broad(y.shape()));
    }
};

pub const GroupNorm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    group_size: u32,
    eps: f32,

    pub fn forward(self: GroupNorm, input: zml.Tensor) zml.Tensor {
        const x = input.withPartialTags(.{ .channels, .width, .height });
        var x_grouped = x.splitAxis(.channels, .{ .group = self.group_size, .c = .auto });
        x_grouped = x_grouped.merge(.{ .cwh = .{ .c, .width, .height } });

        const normed = zml.nn.normalizeVariance(x_grouped, .cwh, self.eps).reshape(x.shape());

        var out = normed.mul(self.weight.withTags(.{.channels}).broad(x.shape()));
        out = out.add(self.bias.withTags(.{.channels}).broad(x.shape()));

        return out;
    }
};

pub const DownBlocks = struct {
    @"0": CrossAttnDownBlock2D,
    @"1": CrossAttnDownBlock2D,
    @"2": CrossAttnDownBlock2D,
    @"3": DownBlock2D,

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
            const x = input.withPartialTags(.{ .channels, .width, .height });
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

pub const ResnetBlock2D = struct {
    norm1: GroupNorm,
    conv1: Conv2dSame,
    time_emb_proj: zml.nn.Linear,
    norm2: GroupNorm,
    conv2: Conv2dSame,
    nonlinearity: zml.nn.Activation = .silu,
    // TODO: output_scale_factor ?

    pub fn forward(self: ResnetBlock2D, images: zml.Tensor, time_embedding: zml.Tensor) zml.Tensor {
        const x = images.withPartialTags(.{ .channels, .width, .height });
        var hidden = self.norm1.forward(x);
        hidden = self.nonlinearity.forward(hidden);
        hidden = self.conv1.forward(hidden);

        var t_emb = time_embedding.withPartialTags(.{.channels});
        t_emb = self.nonlinearity.forward(t_emb);
        t_emb = self.time_emb_proj.forward(t_emb);

        hidden = hidden.add(t_emb.appendAxes(.{ .width, .height }).broad(x.shape()));
        hidden = self.norm2.forward(hidden);

        hidden = self.nonlinearity.forward(hidden);
        hidden = self.conv2.forward(hidden);

        return x.add(hidden);
    }
};

pub const Transformer2DModel = struct {
    norm: GroupNorm,
    proj_in: zml.nn.Linear,
    transformer_blocks: []BasicTransformerBlock,
    proj_out: zml.nn.Linear,

    pub fn forward(self: Transformer2DModel, images: zml.Tensor, encoder_hidden_states: zml.Tensor) zml.Tensor {
        const x = images.withPartialTags(.{ .b, .channels, .width, .height });
        var hidden = x;
        hidden = self.norm.forward(hidden);
        hidden = hidden.merge(.{ .wh = .{ .width, .height } }).contiguous(.{.channels});
        hidden = self.proj_in.forward(hidden);

        for (self.transformer_blocks) |block| {
            hidden = zml.call(block, .forward, .{ hidden, encoder_hidden_states });
        }

        hidden = self.proj_out.forward(hidden);
        hidden = hidden.contiguous(.{.wh}).splitAxis(.wh, .{ .width = x.dim(.width), .height = x.dim(.height) });
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
            const x = if (input.shape().isFullyTagged()) input else input.withTags(.{ .b, .wh, .channels });
            const ctx = if (input.shape().isFullyTagged()) context else context.withTags(.{ .b, .k, .channels });
            const nh = 5;
            const q = zml.call(self.to_q, .forward, .{x.rename(.{ .wh = .q })}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            const k = zml.call(self.to_k, .forward, .{ctx}).splitAxis(.channels, .{ .h = nh, .hd = .auto });
            const v = zml.call(self.to_v, .forward, .{ctx}).splitAxis(.channels, .{ .h = nh, .hd = .auto });

            const attn_output = zml.nn.sdpa(q, k, v, .{});
            const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .wh, .d = .channels });
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
            var y = self.norm1.forward(hidden);
            y = self.attn1.forward(y, y.withPartialTags(.{ .k, .channels }));
            hidden = hidden.add(y);
        }

        {
            // Then cross-attention
            var y = self.norm2.forward(hidden);
            y = self.attn2.forward(y, encoder_hidden_state.withTags(.{ .b, .k, .channels }));
            hidden = hidden.add(y);
        }

        {
            var y = self.norm3.forward(hidden);
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
        .escape_whitespaces = false,
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
