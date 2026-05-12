const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dflash = @import("dflash_model.zig");
const llama = @import("llama/llama_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.dflash);

const prompt =
    "Paris is the city of lights, fun, and love where everyone can enjoy the amazing culture, food, and art.";

const Args = struct {
    model: []const u8,
    target_model: []const u8,

    pub const help =
        \\Use dflash --model=<path> --target-model=<path>
        \\
        \\Compile the DFlash draft model, apply the target lm_head, and print greedy tokens.
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const target_repo = try zml.safetensors.resolveModelRepo(io, args.target_model);

    const parsed_config = try parseConfig(dflash.Config, allocator, io, repo);
    defer parsed_config.deinit();
    const parsed_target_config = try parseConfig(llama.Config, allocator, io, target_repo);
    defer parsed_target_config.deinit();

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var target_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, target_repo);
    defer target_registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var target_store: zml.io.TensorStore = .fromRegistry(allocator, &target_registry);
    defer target_store.deinit();

    var tokenizer = try loadTokenizer(allocator, io, target_repo);
    defer tokenizer.deinit();

    const draft_model = try dflash.Model.init(allocator, store.view(), parsed_config.value);
    defer draft_model.deinit(allocator);

    const target_model = try llama.Model.init(allocator, target_store.view(), parsed_target_config.value, .{
        .sampling_strategy = .{},
        .max_seq_len = parsed_config.value.block_size,
    });
    defer target_model.deinit(allocator);

    const shardings: Shardings = try .init(platform);
    const all_shardings = shardings.all();

    const block_size = parsed_config.value.block_size;
    const input_tokens_tensor: zml.Tensor = .init(.{ .s = block_size }, .u32);
    const token_index_tensor: zml.Tensor = .init(.{}, .u32);
    const target_kv_cache = llama.KvCache.init(.init(.{
        .layer = parsed_target_config.value.num_hidden_layers,
        .k = block_size,
        .h = parsed_target_config.value.num_key_value_heads,
        .hd = parsed_target_config.value.head_dim orelse parsed_target_config.value.hidden_size / parsed_target_config.value.num_attention_heads,
    }, target_model.model.embed_tokens.weight.dtype()));
    const attention_backend: zml.attention.attention.Backend = .vanilla;
    const attention_metadata = zml.attention.attention.Metadata.init(.fromBackend(
        attention_backend,
        block_size,
        parsed_target_config.value.num_attention_heads,
    ));
    const attention_parameters = zml.attention.attention.Parameters.init(.fromBackend(attention_backend));

    log.info("Compiling model...", .{});
    var exe = try platform.compileFn(
        allocator,
        io,
        runDFlash,
        .{
            draft_model,
            target_model,
            input_tokens_tensor,
            token_index_tensor,
            target_kv_cache,
            attention_metadata,
            attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    defer exe.deinit();

    log.info("Loading weights...", .{});
    var model_buffers = try zml.io.load(dflash.Model, &draft_model, allocator, io, platform, &store, .{
        .parallelism = 16,
        .shardings = &all_shardings,
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer dflash.Model.unloadBuffers(&model_buffers, allocator);

    var target_buffers = try zml.io.load(llama.Model, &target_model, allocator, io, platform, &target_store, .{
        .parallelism = 16,
        .shardings = &all_shardings,
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer llama.Model.unloadBuffers(&target_buffers, allocator);

    const token_ids = try promptTokens(allocator, &tokenizer, parsed_target_config.value, block_size);
    defer allocator.free(token_ids);

    var input_tokens_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(input_tokens_tensor.shape(), std.mem.sliceAsBytes(token_ids)),
        .replicated,
    );
    defer input_tokens_buffer.deinit();

    var token_index: u32 = 0;
    var token_index_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(token_index_tensor.shape(), std.mem.asBytes(&token_index)),
        .replicated,
    );
    defer token_index_buffer.deinit();

    var target_kv_cache_buffers = try llama.KvCache.initBuffer(target_kv_cache, io, platform, shardings.model);
    defer llama.KvCache.deinitBuffer(&target_kv_cache_buffers);

    var attention_metadata_buffers = try attention_metadata.initBuffer(io, platform, shardings.model);
    defer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe_args.set(.{
        model_buffers,
        target_buffers,
        input_tokens_buffer,
        token_index_buffer,
        target_kv_cache_buffers,
        attention_metadata_buffers,
    });
    exe.call(exe_args, &results);

    var target_token_buffer, var noise_token_buffer, var speculative_token_buffer = results.get(struct {
        zml.Buffer,
        zml.Buffer,
        zml.Buffer,
    });
    defer target_token_buffer.deinit();
    defer noise_token_buffer.deinit();
    defer speculative_token_buffer.deinit();

    var target_token_slice = try target_token_buffer.toSliceAlloc(allocator, io);
    defer target_token_slice.free(allocator);

    var noise_token_slice = try noise_token_buffer.toSliceAlloc(allocator, io);
    defer noise_token_slice.free(allocator);

    var speculative_token_slice = try speculative_token_buffer.toSliceAlloc(allocator, io);
    defer speculative_token_slice.free(allocator);

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try printTokens(allocator, &tokenizer, &stdout.interface, "prompt", token_ids);
    try printTokens(allocator, &tokenizer, &stdout.interface, "target_next", target_token_slice.constItems(u32));
    try printTokens(allocator, &tokenizer, &stdout.interface, "noise", noise_token_slice.constItems(u32));
    try stdout.interface.print("speculative_tokens shape: {f}\n", .{speculative_token_slice.shape});

    const speculative_tokens = speculative_token_slice.constItems(u32);
    try printTokens(allocator, &tokenizer, &stdout.interface, "speculative", speculative_tokens);
    try stdout.interface.flush();
}

fn runDFlash(
    draft_model: dflash.Model,
    target_model: llama.Model,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: llama.KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
    const target_hidden, const noise_embedding, const target_token, const noise_tokens = target_model.dflashInputs(
        tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        draft_model.target_layer_ids,
        draft_model.mask_token_id.?,
    );
    const position_ids = zml.Tensor.arange(.{ .end = draft_model.block_size * 2 }, .u32).withTags(.{.s});
    const hidden = draft_model.forward(target_hidden, noise_embedding, position_ids);
    const draft_tokens = target_model.greedyTokensFromHidden(hidden);
    const speculative_tokens = zml.Tensor.concatenate(&.{
        target_token,
        draft_tokens.slice1d(.s, .{ .start = 1, .end = draft_tokens.dim(.s) }),
    }, .s);
    return .{ target_token, noise_tokens, speculative_tokens };
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);

    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);

    return try .fromBytes(allocator, bytes);
}

fn promptTokens(allocator: std.mem.Allocator, tokenizer: *zml.tokenizer.Tokenizer, config: llama.Config, block_size: u32) ![]u32 {
    var token_ids = try allocator.alloc(u32, block_size);
    errdefer allocator.free(token_ids);
    @memset(token_ids, firstEosToken(config));

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const encoded = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(encoded);

    const len = @min(token_ids.len, encoded.len);
    @memcpy(token_ids[0..len], encoded[0..len]);
    return token_ids;
}

fn firstEosToken(config: llama.Config) u32 {
    return switch (config.eos_token_id.value) {
        .int => |eos| eos,
        .ints => |eos_list| eos_list[0],
    };
}

fn printTokens(
    allocator: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    stdout: *std.Io.Writer,
    comptime name: []const u8,
    token_ids: []const u32,
) !void {
    try stdout.print("{s}_ids: {any}\n", .{ name, token_ids });

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();
    var text = try decoder.decodeAlloc(allocator, token_ids);
    defer text.deinit(allocator);
    try stdout.print("{s}_text: {s}\n", .{ name, text.items });

    try stdout.print("{s}_tokens: [", .{name});
    for (token_ids, 0..) |token_id, i| {
        if (i != 0) try stdout.writeAll(", ");
        const one = [1]u32{token_id};
        var piece_decoder = try tokenizer.decoder();
        defer piece_decoder.deinit();
        var piece = try piece_decoder.decodeAlloc(allocator, &one);
        defer piece.deinit(allocator);
        try stdout.print("{}=\"{s}\"", .{ token_id, piece.items });
    }
    try stdout.writeAll("]\n");
}

const Shardings = struct {
    model: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }
};

fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
