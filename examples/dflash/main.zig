const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dflash = @import("dflash_model.zig");
const llama = @import("llama/llama_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.dflash);

const default_prompt =
    "Paris is the city of lights, fun, and love where everyone can enjoy the amazing culture, food, and art.";

const Args = struct {
    model: []const u8,
    target_model: []const u8,
    prompt: []const u8 = default_prompt,

    pub const help =
        \\Use dflash --model=<path> --target-model=<path> [--prompt=<text>]
        \\
        \\Compile target prefill, DFlash draft, and target verification separately.
        \\
        \\Options:
        \\  --model=<path>          Path to the DFlash model repository
        \\  --target-model=<path>   Path to the target LLaMA model repository
        \\  --prompt=<text>         Prompt to tokenize; defaults to the built-in smoke-test prompt
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
    const max_seq_len = block_size * 2;
    const input_tokens_tensor: zml.Tensor = .init(.{ .s = block_size }, .u32);
    const block_tokens_tensor: zml.Tensor = .init(.{ .s = block_size }, .u32);
    const target_hidden_tensor: zml.Tensor = .init(.{
        .s = block_size,
        .d = draft_model.target_layer_ids.len * parsed_target_config.value.hidden_size,
    }, target_model.model.embed_tokens.weight.dtype());
    const noise_embedding_tensor: zml.Tensor = .init(.{ .s = block_size, .d = parsed_target_config.value.hidden_size }, target_model.model.embed_tokens.weight.dtype());
    const draft_position_ids_tensor: zml.Tensor = .init(.{ .s = max_seq_len }, .u32);
    const token_index_tensor: zml.Tensor = .init(.{}, .u32);
    const target_kv_cache = llama.KvCache.init(.init(.{
        .layer = parsed_target_config.value.num_hidden_layers,
        .k = max_seq_len,
        .h = parsed_target_config.value.num_key_value_heads,
        .hd = parsed_target_config.value.head_dim orelse parsed_target_config.value.hidden_size / parsed_target_config.value.num_attention_heads,
    }, target_model.model.embed_tokens.weight.dtype()));
    const draft_kv_cache = dflash.KvCache.init(.init(.{
        .layer = parsed_config.value.num_hidden_layers,
        .k = max_seq_len,
        .h = parsed_config.value.num_key_value_heads,
        .hd = parsed_config.value.head_dim orelse parsed_config.value.hidden_size / parsed_config.value.num_attention_heads,
    }, .f32));
    const attention_backend: zml.attention.attention.Backend = .vanilla;
    const attention_metadata = zml.attention.attention.Metadata.init(.fromBackend(
        attention_backend,
        max_seq_len,
        parsed_target_config.value.num_attention_heads,
    ));
    const attention_parameters = zml.attention.attention.Parameters.init(.fromBackend(attention_backend));
    const target_layers = TargetLayers.init(draft_model.target_layer_ids);

    log.info("Compiling target prefill...", .{});
    var target_prefill_exe = try platform.compileFn(
        allocator,
        io,
        targetPrefill,
        .{
            target_model,
            target_layers,
            input_tokens_tensor,
            token_index_tensor,
            target_kv_cache,
            attention_metadata,
            attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    defer target_prefill_exe.deinit();

    log.info("Compiling target embedding...", .{});
    var target_embed_exe = try platform.compileFn(
        allocator,
        io,
        targetEmbed,
        .{ target_model, block_tokens_tensor },
        .{ .shardings = &all_shardings },
    );
    defer target_embed_exe.deinit();

    log.info("Compiling DFlash draft...", .{});
    var draft_exe = try platform.compileFn(
        allocator,
        io,
        draftForward,
        .{
            draft_model,
            target_hidden_tensor,
            noise_embedding_tensor,
            draft_position_ids_tensor,
            draft_kv_cache,
            token_index_tensor,
        },
        .{ .shardings = &all_shardings },
    );
    defer draft_exe.deinit();

    log.info("Compiling target lm_head...", .{});
    var target_lm_head_exe = try platform.compileFn(
        allocator,
        io,
        targetGreedyFromHidden,
        .{ target_model, noise_embedding_tensor },
        .{ .shardings = &all_shardings },
    );
    defer target_lm_head_exe.deinit();

    log.info("Compiling target verify...", .{});
    var target_verify_exe = try platform.compileFn(
        allocator,
        io,
        targetVerify,
        .{
            target_model,
            target_layers,
            block_tokens_tensor,
            token_index_tensor,
            target_kv_cache,
            attention_metadata,
            attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    defer target_verify_exe.deinit();

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

    const token_ids = try promptTokens(allocator, &tokenizer, parsed_target_config.value, block_size, args.prompt);
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

    const draft_position_ids = try allocator.alloc(u32, max_seq_len);
    defer allocator.free(draft_position_ids);
    for (draft_position_ids, 0..) |*pos, i| pos.* = @intCast(i);

    var draft_position_ids_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(draft_position_ids_tensor.shape(), std.mem.sliceAsBytes(draft_position_ids)),
        .replicated,
    );
    defer draft_position_ids_buffer.deinit();

    var target_kv_cache_buffers = try llama.KvCache.initBuffer(target_kv_cache, io, platform, shardings.model);
    defer llama.KvCache.deinitBuffer(&target_kv_cache_buffers);

    var draft_kv_cache_buffers = try dflash.KvCache.initBuffer(draft_kv_cache, io, platform, shardings.model);
    defer dflash.KvCache.deinitBuffer(&draft_kv_cache_buffers);

    var attention_metadata_buffers = try attention_metadata.initBuffer(io, platform, shardings.model);
    defer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

    var prefill_args = try target_prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);

    var prefill_results = try target_prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    prefill_args.set(.{
        target_buffers,
        input_tokens_buffer,
        token_index_buffer,
        target_kv_cache_buffers,
        attention_metadata_buffers,
    });
    target_prefill_exe.call(prefill_args, &prefill_results);

    var target_hidden_buffer, var target_token_buffer, var prefilled_target_kv_cache_buffers = prefill_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        llama.KvCache.Buffer,
    });
    defer target_hidden_buffer.deinit();
    defer target_token_buffer.deinit();
    replaceTargetKvCacheBuffers(&target_kv_cache_buffers, &prefilled_target_kv_cache_buffers);

    var target_token_slice = try target_token_buffer.toSliceAlloc(allocator, io);
    defer target_token_slice.free(allocator);

    const target_token = target_token_slice.constItems(u32)[0];
    var block_tokens = try allocator.alloc(u32, block_size);
    defer allocator.free(block_tokens);
    @memset(block_tokens, parsed_config.value.dflash_config.mask_token_id.?);
    block_tokens[0] = target_token;

    var block_tokens_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(block_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
        .replicated,
    );
    defer block_tokens_buffer.deinit();

    var embed_args = try target_embed_exe.args(allocator);
    defer embed_args.deinit(allocator);
    var embed_results = try target_embed_exe.results(allocator);
    defer embed_results.deinit(allocator);
    embed_args.set(.{ target_buffers, block_tokens_buffer });
    target_embed_exe.call(embed_args, &embed_results);

    var noise_embedding_buffer = embed_results.get(zml.Buffer);
    defer noise_embedding_buffer.deinit();

    var draft_cache_index: u32 = 0;
    var draft_cache_index_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(token_index_tensor.shape(), std.mem.asBytes(&draft_cache_index)),
        .replicated,
    );
    defer draft_cache_index_buffer.deinit();

    var draft_args = try draft_exe.args(allocator);
    defer draft_args.deinit(allocator);
    var draft_results = try draft_exe.results(allocator);
    defer draft_results.deinit(allocator);
    draft_args.set(.{
        model_buffers,
        target_hidden_buffer,
        noise_embedding_buffer,
        draft_position_ids_buffer,
        draft_kv_cache_buffers,
        draft_cache_index_buffer,
    });
    draft_exe.call(draft_args, &draft_results);

    var draft_hidden_buffer, var updated_draft_kv_cache_buffers = draft_results.get(struct {
        zml.Buffer,
        dflash.KvCache.Buffer,
    });
    defer draft_hidden_buffer.deinit();
    replaceDraftKvCacheBuffers(&draft_kv_cache_buffers, &updated_draft_kv_cache_buffers);

    var lm_head_args = try target_lm_head_exe.args(allocator);
    defer lm_head_args.deinit(allocator);
    var lm_head_results = try target_lm_head_exe.results(allocator);
    defer lm_head_results.deinit(allocator);
    lm_head_args.set(.{ target_buffers, draft_hidden_buffer });
    target_lm_head_exe.call(lm_head_args, &lm_head_results);

    var draft_token_buffer = lm_head_results.get(zml.Buffer);
    defer draft_token_buffer.deinit();

    var draft_token_slice = try draft_token_buffer.toSliceAlloc(allocator, io);
    defer draft_token_slice.free(allocator);
    const draft_tokens = draft_token_slice.constItems(u32);
    for (block_tokens[1..], draft_tokens[1..block_size]) |*dst, src| dst.* = src;
    var drafted_block_tokens_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(block_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
        .replicated,
    );
    replaceBuffer(&block_tokens_buffer, &drafted_block_tokens_buffer);

    token_index = block_size;
    var verify_token_index_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(token_index_tensor.shape(), std.mem.asBytes(&token_index)),
        .replicated,
    );
    defer verify_token_index_buffer.deinit();

    var verify_args = try target_verify_exe.args(allocator);
    defer verify_args.deinit(allocator);
    var verify_results = try target_verify_exe.results(allocator);
    defer verify_results.deinit(allocator);
    verify_args.set(.{
        target_buffers,
        block_tokens_buffer,
        verify_token_index_buffer,
        target_kv_cache_buffers,
        attention_metadata_buffers,
    });
    target_verify_exe.call(verify_args, &verify_results);

    var verified_hidden_buffer, var posterior_token_buffer, var verified_target_kv_cache_buffers = verify_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        llama.KvCache.Buffer,
    });
    defer verified_hidden_buffer.deinit();
    defer posterior_token_buffer.deinit();
    replaceTargetKvCacheBuffers(&target_kv_cache_buffers, &verified_target_kv_cache_buffers);

    var posterior_token_slice = try posterior_token_buffer.toSliceAlloc(allocator, io);
    defer posterior_token_slice.free(allocator);
    const posterior_tokens = posterior_token_slice.constItems(u32);

    var valid_draft_tokens: usize = 0;
    while (valid_draft_tokens + 1 < block_size and block_tokens[valid_draft_tokens + 1] == posterior_tokens[valid_draft_tokens]) {
        valid_draft_tokens += 1;
    }
    const correction_token = posterior_tokens[valid_draft_tokens];
    const committed_tokens = valid_draft_tokens + 1;

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try printTokens(allocator, &tokenizer, &stdout.interface, "prompt", token_ids);
    try printTokens(allocator, &tokenizer, &stdout.interface, "target_next", target_token_slice.constItems(u32));
    try printTokens(allocator, &tokenizer, &stdout.interface, "draft_block", block_tokens);
    try printTokens(allocator, &tokenizer, &stdout.interface, "posterior", posterior_tokens);
    try stdout.interface.print("first_mismatch_index: {}\n", .{valid_draft_tokens});
    try stdout.interface.print("valid_draft_tokens: {}\n", .{valid_draft_tokens});
    try stdout.interface.print("committed_tokens: {}\n", .{committed_tokens});
    try printTokens(allocator, &tokenizer, &stdout.interface, "correction", &.{correction_token});
    try stdout.interface.print("target_cache_logical_len: {}\n", .{block_size + committed_tokens});
    try stdout.interface.print("draft_cache_logical_len_after_crop: {}\n", .{block_size});
    try stdout.interface.flush();
}

const TargetLayers = struct {
    ids: [32]u32 = undefined,
    len: usize = 0,

    pub fn init(ids: []const u32) TargetLayers {
        stdx.debug.assert(ids.len <= 32, "DFlash example supports at most 32 target layers, got {}", .{ids.len});
        var res: TargetLayers = .{};
        res.len = ids.len;
        @memcpy(res.ids[0..ids.len], ids);
        return res;
    }

    pub fn slice(self: *const TargetLayers) []const u32 {
        return self.ids[0..self.len];
    }
};

fn targetPrefill(
    target_model: llama.Model,
    target_layers: TargetLayers,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: llama.KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) struct { zml.Tensor, zml.Tensor, llama.KvCache } {
    return target_model.dflashPrefill(
        tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
    );
}

fn targetVerify(
    target_model: llama.Model,
    target_layers: TargetLayers,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: llama.KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) struct { zml.Tensor, zml.Tensor, llama.KvCache } {
    return target_model.dflashVerify(
        tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
    );
}

fn targetEmbed(target_model: llama.Model, tokens: zml.Tensor) zml.Tensor {
    return target_model.embedTokens(tokens);
}

fn draftForward(
    draft_model: dflash.Model,
    target_hidden: zml.Tensor,
    noise_embedding: zml.Tensor,
    position_ids: zml.Tensor,
    draft_kv_cache: dflash.KvCache,
    cache_index: zml.Tensor,
) struct { zml.Tensor, dflash.KvCache } {
    return draft_model.forwardCached(target_hidden, noise_embedding, position_ids, draft_kv_cache, cache_index);
}

fn targetGreedyFromHidden(target_model: llama.Model, hidden: zml.Tensor) zml.Tensor {
    return target_model.greedyTokensFromHidden(hidden);
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);

    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);

    return try .fromBytes(allocator, bytes);
}

fn promptTokens(
    allocator: std.mem.Allocator,
    tokenizer: *zml.tokenizer.Tokenizer,
    config: llama.Config,
    block_size: u32,
    prompt: []const u8,
) ![]u32 {
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

fn replaceTargetKvCacheBuffers(dst: *llama.KvCache.Buffer, src: *llama.KvCache.Buffer) void {
    replaceBuffer(&dst.k, &src.k);
    replaceBuffer(&dst.v, &src.v);
}

fn replaceDraftKvCacheBuffers(dst: *dflash.KvCache.Buffer, src: *dflash.KvCache.Buffer) void {
    replaceBuffer(&dst.k, &src.k);
    replaceBuffer(&dst.v, &src.v);
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
