const std = @import("std");
const stdx = @import("stdx");
const log = std.log;

const zml = @import("zml");
const block = @import("./blocks.zig");
const store = @import("./store.zig");
const testing = @import("./testing.zig");
const CompileArgs = @import("./compiler.zig").CompileArgs;

const CliArgs = struct {
    pub const help =
        \\ torch2zml --folder=<path> --activations=<path>
    ;
    folder: [:0]const u8,
    activations: ?[:0]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const model_sharding = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth }));
    const backend = zml.attention.Backend.auto(platform);
    // const backend: zml.attention.Backend = .vanilla;
    // 32 from num_attention_heads in text_encoder/config.json and 16415 is a dynamic value
    const attention_metadata: zml.attention.Metadata = .init(.fromBackend(backend, 16415, 32));
    const attention_parameters: zml.attention.Parameters = .init(.fromBackend(backend));

    const compile_args: CompileArgs = .{
        .width = 2048,
        .height = 2048,

        .in_channels = 64,
        .num_inference_steps = 40,
        .vae_scale_factor = 16,

        .seq_len = 16415,
        .num_attention_heads = 32,
        .head_dim = 128,
        .rope_theta = 5000000,
        .attention_metadata = attention_metadata,
        .attention_parameters = attention_parameters,

        .max_image_seq_len = 8192,
        .base_shift = 0.5,
        .max_shift = 0.9,
    };
    const compile_args_buffers: zml.Bufferized(CompileArgs) = .{
        .attention_metadata = try attention_metadata.initBuffer(io, platform, model_sharding),
    };

    if (cli_args.activations == null) {
        // Run the pipeline then
        std.debug.print("\n\nStarting testing > Pipeline\n\n", .{});

        var pipeline = try block.Pipeline.init(allocator, io, platform, cli_args.folder);
        const prompt = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement";
        try pipeline.run_once(prompt, .{ .seq_len = 16415, .compile_args = compile_args, .compile_args_buffers = compile_args_buffers });
        return;
    }

    const activations_path = cli_args.activations.?;

    var activations_store: store.ActivationStore = undefined;
    try activations_store.load(allocator, io, activations_path);
    defer activations_store.deinit();

    if (std.mem.containsAtLeast(u8, activations_path, 1, "block.")) {
        std.debug.print("\n\nStarting testing > TransformerBlock\n\n", .{});

        var transformer_store: store.TransformerStore = undefined;
        try transformer_store.load_tensors(allocator, io, cli_args.folder);
        try transformer_store.load_buffers(allocator, io, platform);
        defer transformer_store.deinit(allocator);

        try testing.testLayer(
            allocator,
            io,
            platform,
            "transformer",
            &activations_store.store,
            BlockWrapper{ .tblock = transformer_store.model, .args = compile_args },
            .{ .tblock = transformer_store.model_buffer, .args = compile_args_buffers },
        );
    } else if (std.mem.containsAtLeast(u8, activations_path, 1, "text.")) {
        std.debug.print("\n\nStarting testing > LanguageModel\n\n", .{});

        var language_model_store: store.LanguageModelStore = undefined;
        try language_model_store.load_tensors(allocator, io, cli_args.folder);
        try language_model_store.load_buffers(allocator, io, platform);
        defer language_model_store.deinit(allocator);

        try testing.testLayer(
            allocator,
            io,
            platform,
            "text_encoder.model.language_model.rotary_emb",
            &activations_store.store,
            TextWrapper{ .model = language_model_store.model.language_model.rotary_emb, .args = compile_args },
            .{ .args = compile_args_buffers },
        );
    } else if (std.mem.containsAtLeast(u8, activations_path, 1, "vae.")) {
        std.debug.print("\n\nStarting testing > VaeModel\n\n", .{});

        var vae_model_store: store.VaeModelStore = undefined;
        try vae_model_store.load_tensors(allocator, io, cli_args.folder);
        try vae_model_store.load_buffers(allocator, io, platform);
        defer vae_model_store.deinit(allocator);

        try testing.testLayer(
            allocator,
            io,
            platform,
            "vae.decoder",
            &activations_store.store,
            VaeWrapper{ .model = vae_model_store.model, .args = compile_args },
            .{ .model = vae_model_store.model_buffer, .args = compile_args_buffers },
        );
    } else {
        std.debug.print("NOTHING IS BEING TESTED\n", .{});
        unreachable;
    }
}

const BlockWrapper = struct {
    tblock: block.QwenImage21Transformer2DModel,
    args: CompileArgs,

    pub fn forward(
        self: BlockWrapper,
        x: zml.Tensor,
        timestep_: zml.Tensor,
        encoder_hidden_states: zml.Tensor,
        img_mask: zml.Tensor,
    ) zml.Tensor {
        return self.tblock.forward(
            x.withTags(.{ .bs, .seq_len, .d }),
            encoder_hidden_states.withTags(.{ .bs, .seq_len, .d }),
            timestep_.withTags(.{.bs}),
            img_mask,
            self.args,
        );
    }
};

const TextWrapper = struct {
    model: block.Qwen3VLTextRotaryEmbedding,
    args: CompileArgs,

    pub fn forward(
        self: TextWrapper,
        x: zml.Tensor,
        pos_idx: zml.Tensor,
        // cos: zml.Tensor,
        // sin: zml.Tensor,
    ) [2]zml.Tensor {
        const tagged = x.withTags(.{ .bs, .seq_len, .d });
        return self.model.forward(tagged, pos_idx, self.args);
    }
};

const VaeWrapper = struct {
    model: block.AutoencoderKLQwenImage21,
    args: CompileArgs,

    pub fn forward(
        self: VaeWrapper,
        x: zml.Tensor,
    ) zml.Tensor {
        // For this to work, comment post_quant_conv.forward and the dec.clamp
        const tagged = x.withTags(.{ .bs, .c, .t, .kx, .ky });
        return self.model.forward(tagged);
    }
};

test {
    _ = @import("./blocks/tokenizer.zig");
    _ = @import("./blocks/vae/AutoencoderKLQwenImage21.zig");
    _ = @import("./blocks/TorchGenerator.zig");
}
