const std = @import("std");

const zml = @import("zml");
const meta = zml.meta;
const asynk = @import("async");
const flags = @import("tigerbeetle/flags");

const transformer = @import("transformer.zig");

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
    _ = vocab_path; // autofix
    const prompt = args[5];
    const activation_path = args[6];

    log.info("Prompt: {s}", .{prompt});

    var prompt_encoder_store = try zml.aio.detectFormatAndOpen(allocator, prompt_encoder_model_path);
    defer prompt_encoder_store.deinit();

    const prompt_encoder = try zml.aio.populateModel(transformer.Llama, arena, prompt_encoder_store);

    log.info("Loaded prompt encoder from {s}, found {} buffers: {}", .{ prompt_encoder_model_path, prompt_encoder_store.buffers.count(), prompt_encoder });

    // var vae_weights = try zml.aio.detectFormatAndOpen(allocator, vae_model_path);
    // defer vae_weights.deinit();

    // log.info("Loaded VAE from {s}, found {} buffers.", .{vae_model_path, vae_weights.buffers.count()});

    var unet_store = try zml.aio.detectFormatAndOpen(allocator, unet_model_path);
    defer unet_store.deinit();

    log.info("Loaded unet from {s}, found {} buffers.", .{ unet_model_path, unet_store.buffers.count() });

    var unet = try zml.aio.populateModel(Unet2DConditionModel, arena, unet_store);
    unet.conv_norm_out.group_size = 32;
    unet.conv_norm_out.eps = 1e-5;
    for (unet.down_blocks.@"3".resnets) |*resnet| {
        resnet.norm1.group_size = 32;
        resnet.norm1.eps = 1e-5;
        resnet.norm2.group_size = 32;
        resnet.norm2.eps = 1e-5;
    }
    log.info("Unet: {}", .{unet});

    var activations = try zml.aio.detectFormatAndOpen(allocator, activation_path);
    defer activations.deinit();

    const unet_weights = try zml.aio.loadModelBuffers(Unet2DConditionModel, unet, unet_store, arena, platform);

    try zml.testing.testLayer(platform, activations, "unet.conv_in", unet.conv_in, unet_weights.conv_in, 5e-3);
    try zml.testing.testLayer(platform, activations, "unet.conv_out", unet.conv_out, unet_weights.conv_out, 5e-3);
    try zml.testing.testLayer(platform, activations, "unet.conv_norm_out", unet.conv_norm_out, unet_weights.conv_norm_out, 5e-3);

    try zml.testing.testLayer(platform, activations, "unet.down_blocks.3.resnets.0", unet.down_blocks.@"3".resnets[0], unet_weights.down_blocks.@"3".resnets[0], 5e-2);
}

pub const Unet2DConditionModel = struct {
    conv_in: Conv2d,

    // time_proj
    // time_embedding
    down_blocks: DownBlocks,
    // up_blocks
    // mid_block

    conv_norm_out: GroupNorm,
    conv_act: zml.nn.Activation = .silu,
    conv_out: Conv2d,
};

pub const Conv2d = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn forward(self: Conv2d, input: zml.Tensor) zml.Tensor {
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

    pub const CrossAttnDownBlock2D = struct {};

    pub const DownBlock2D = struct {
        resnets: []ResnetBlock2D,
    };
};

pub const ResnetBlock2D = struct {
    norm1: GroupNorm,
    conv1: Conv2d,
    time_emb_proj: zml.nn.Linear,
    norm2: GroupNorm,
    conv2: Conv2d,
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

        log.warn("hidden: {}, temb: {}", .{ hidden, t_emb });
        hidden = hidden.add(t_emb.appendAxes(.{ .width, .height }).broad(x.shape()));
        hidden = self.norm2.forward(hidden);

        hidden = self.nonlinearity.forward(hidden);
        hidden = self.conv2.forward(hidden);

        return x.add(hidden);
    }
};
