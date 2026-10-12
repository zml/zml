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
    activations: [:0]const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var transformer_blocks_store = try store.TransformerStore.load(allocator, io, platform, cli_args.folder);
    defer transformer_blocks_store.deinit(allocator);

    var activations_store = try store.ActivationStore.load(allocator, io, cli_args.activations);
    defer activations_store.deinit();

    std.debug.print("\n\nStarting testing\n\n", .{});

    const model_sharding = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth }));
    const backend = zml.attention.Backend.auto(platform);
    // 32 from num_attention_heads in text_encoder/config.json and 16415 is a dynamic value
    const attention_metadata: zml.attention.Metadata = .init(.fromBackend(backend, 16415, 32));
    const attention_parameters: zml.attention.Parameters = .init(.fromBackend(backend));

    const compile_args: CompileArgs = .{
        .seq_len = 16415,
        .num_attention_heads = 32,
        .attention_metadata = attention_metadata,
        .attention_parameters = attention_parameters,
    };
    const compile_args_buffers: zml.Bufferized(CompileArgs) = .{
        .attention_metadata = try attention_metadata.initBuffer(io, platform, model_sharding),
    };

    try testing.testLayer(
        allocator,
        io,
        platform,
        "transformer.transformer_blocks.0",
        &activations_store.store,
        Wrapper{ .tblock = transformer_blocks_store.transformer_blocks[0], .args = compile_args },
        .{ .tblock = transformer_blocks_store.transformer_block_buffers[0], .args = compile_args_buffers },
    );

    // const pipeline = block.Pipeline.init(allocator, io, platform);
    // try pipeline.runOnce("Coucou la team, créé un personnage burger", .{ .seq_len = 16415 });
}

const Wrapper = struct {
    tblock: block.TransformerBlock,
    args: CompileArgs,

    pub fn forward(
        self: Wrapper,
        x: zml.Tensor,
        modulation: zml.Tensor,
        rotary_emb: zml.Tensor,
        target_token_mask: zml.Tensor,
    ) zml.Tensor {
        const tagged = x.withTags(.{ .bs, .seq_len, .d });
        return self.tblock.forward(tagged, modulation, rotary_emb.withTags(.{ .seq_len, .d }), target_token_mask, self.args);
    }
};

test {
    _ = @import("./blocks/tokenizer.zig");
}
