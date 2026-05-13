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
    max_seq_len: u32 = 256,

    pub const help =
        \\Use dflash --model=<path> --target-model=<path> [--prompt=<text>]
        \\
        \\Compile target prefill, DFlash draft, and target verification separately.
        \\
        \\Options:
        \\  --model=<path>          Path to the DFlash model repository
        \\  --target-model=<path>   Path to the target LLaMA model repository
        \\  --prompt=<text>         Prompt to tokenize; defaults to the built-in smoke-test prompt
        \\  --max-seq-len=<n>       Decode until this many accepted positions; defaults to 256
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

    const block_size = parsed_config.value.block_size;
    if (args.max_seq_len < block_size + 1) {
        log.err("--max-seq-len must be greater than DFlash block_size ({}), got {}", .{ block_size, args.max_seq_len });
        return error.InvalidMaxSeqLen;
    }
    const max_seq_len = args.max_seq_len;
    const cache_seq_len = max_seq_len + block_size;

    const target_model = try llama.Model.init(allocator, target_store.view(), parsed_target_config.value, .{
        .sampling_strategy = .{},
        .max_seq_len = cache_seq_len,
    });
    defer target_model.deinit(allocator);

    const shardings: Shardings = try .init(platform);
    const all_shardings = shardings.all();

    const target_hidden_dim = draft_model.target_layer_ids.len * parsed_target_config.value.hidden_size;
    const input_tokens_tensor: zml.Tensor = .init(.{ .s = block_size }, .u32);
    const block_tokens_tensor: zml.Tensor = .init(.{ .s = block_size }, .u32);
    const noise_embedding_tensor: zml.Tensor = .init(.{ .s = block_size, .d = parsed_target_config.value.hidden_size }, target_model.model.embed_tokens.weight.dtype());
    const token_index_tensor: zml.Tensor = .init(.{}, .u32);
    const target_kv_cache = llama.KvCache.init(.init(.{
        .layer = parsed_target_config.value.num_hidden_layers,
        .k = cache_seq_len,
        .h = parsed_target_config.value.num_key_value_heads,
        .hd = parsed_target_config.value.head_dim orelse parsed_target_config.value.hidden_size / parsed_target_config.value.num_attention_heads,
    }, target_model.model.embed_tokens.weight.dtype()));
    const draft_kv_cache = dflash.KvCache.init(.init(.{
        .layer = parsed_config.value.num_hidden_layers,
        .k = cache_seq_len,
        .h = parsed_config.value.num_key_value_heads,
        .hd = parsed_config.value.head_dim orelse parsed_config.value.hidden_size / parsed_config.value.num_attention_heads,
    }, .f32));
    const attention_backend: zml.attention.attention.Backend = .auto(platform);
    const attention_metadata = zml.attention.attention.Metadata.init(.fromBackend(
        attention_backend,
        cache_seq_len,
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

    log.info("Compiling DFlash draft variants...", .{});
    const draft_exes = try allocator.alloc(DraftExe, block_size);
    defer {
        for (draft_exes) |*draft_exe| draft_exe.deinit();
        allocator.free(draft_exes);
    }
    for (draft_exes, 0..) |*draft_exe, i| {
        const context_len: u32 = @intCast(i + 1);
        const target_hidden_tensor: zml.Tensor = .init(.{
            .s = context_len,
            .d = target_hidden_dim,
        }, target_model.model.embed_tokens.weight.dtype());
        const draft_position_ids_tensor: zml.Tensor = .init(.{ .s = context_len + block_size }, .u32);
        draft_exe.* = .{
            .context_len = context_len,
            .position_len = context_len + block_size,
            .exe = try platform.compileFn(
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
            ),
        };
    }

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

    log.info("Compiling hidden prefix variants...", .{});
    const hidden_prefix_exes = try allocator.alloc(PrefixExe, block_size);
    defer {
        for (hidden_prefix_exes) |*prefix_exe| prefix_exe.deinit();
        allocator.free(hidden_prefix_exes);
    }
    const verified_hidden_tensor: zml.Tensor = .init(.{
        .s = block_size,
        .d = target_hidden_dim,
    }, target_model.model.embed_tokens.weight.dtype());
    for (hidden_prefix_exes, 0..) |*prefix_exe, i| {
        const prefix_len: u32 = @intCast(i + 1);
        prefix_exe.* = .{
            .len = prefix_len,
            .exe = try platform.compileFn(
                allocator,
                io,
                hiddenPrefix,
                .{ PrefixSpec{ .len = prefix_len }, verified_hidden_tensor },
                .{ .shardings = &all_shardings },
            ),
        };
    }

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

    var target_kv_cache_buffers = try initZeroTargetKvCacheBuffer(allocator, io, platform, target_kv_cache, shardings.model);
    defer llama.KvCache.deinitBuffer(&target_kv_cache_buffers);

    var draft_kv_cache_buffers = try initZeroDraftKvCacheBuffer(allocator, io, platform, draft_kv_cache, shardings.model);
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

    var generated = try std.ArrayList(u32).initCapacity(allocator, max_seq_len + 1);
    defer generated.deinit(allocator);
    try generated.appendSlice(allocator, token_ids);
    try generated.append(allocator, target_token_slice.constItems(u32)[0]);

    var block_tokens = try allocator.alloc(u32, block_size);
    defer allocator.free(block_tokens);

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try printTokens(allocator, &tokenizer, &stdout.interface, "prompt", token_ids);
    try printTokens(allocator, &tokenizer, &stdout.interface, "target_next", target_token_slice.constItems(u32));

    var start: u32 = block_size;
    var draft_cache_len: u32 = 0;
    var target_hidden_increment_buffer = target_hidden_buffer;
    var owns_target_hidden_increment_buffer = false;
    var step: usize = 0;
    const decode_started_at: std.Io.Timestamp = .now(io, .awake);
    while (start < max_seq_len) : (step += 1) {
        const step_started_at: std.Io.Timestamp = .now(io, .awake);
        const context_len = start - draft_cache_len;
        stdx.debug.assert(context_len >= 1 and context_len <= block_size, "invalid DFlash context length {}", .{context_len});
        const selected_draft_exe = &draft_exes[@as(usize, @intCast(context_len - 1))];

        @memset(block_tokens, parsed_config.value.dflash_config.mask_token_id.?);
        block_tokens[0] = generated.items[@intCast(start)];

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

        const draft_position_ids = try arangeAlloc(allocator, draft_cache_len, selected_draft_exe.position_len);
        defer allocator.free(draft_position_ids);
        var draft_position_ids_buffer: zml.Buffer = try .fromSlice(
            io,
            platform,
            zml.Slice.init(.init(.{ .s = selected_draft_exe.position_len }, .u32), std.mem.sliceAsBytes(draft_position_ids)),
            .replicated,
        );
        defer draft_position_ids_buffer.deinit();

        var draft_cache_index = draft_cache_len;
        var draft_cache_index_buffer: zml.Buffer = try .fromSlice(
            io,
            platform,
            zml.Slice.init(token_index_tensor.shape(), std.mem.asBytes(&draft_cache_index)),
            .replicated,
        );
        defer draft_cache_index_buffer.deinit();

        var draft_args = try selected_draft_exe.exe.args(allocator);
        defer draft_args.deinit(allocator);
        var draft_results = try selected_draft_exe.exe.results(allocator);
        defer draft_results.deinit(allocator);
        draft_args.set(.{
            model_buffers,
            target_hidden_increment_buffer,
            noise_embedding_buffer,
            draft_position_ids_buffer,
            draft_kv_cache_buffers,
            draft_cache_index_buffer,
        });
        selected_draft_exe.exe.call(draft_args, &draft_results);

        var draft_hidden_buffer, var updated_draft_kv_cache_buffers = draft_results.get(struct {
            zml.Buffer,
            dflash.KvCache.Buffer,
        });
        defer draft_hidden_buffer.deinit();
        replaceDraftKvCacheBuffers(&draft_kv_cache_buffers, &updated_draft_kv_cache_buffers);
        draft_cache_len = start;

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

        token_index = start;
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
        const committed_tokens: u32 = @intCast(valid_draft_tokens + 1);

        const generated_step_start = start;
        for (block_tokens[0..@as(usize, @intCast(committed_tokens))]) |token| {
            try setGeneratedToken(&generated, allocator, start, token);
            start += 1;
            if (start >= max_seq_len) break;
        }
        if (start < max_seq_len) {
            try setGeneratedToken(&generated, allocator, start, correction_token);
        }

        const selected_prefix_exe = &hidden_prefix_exes[@as(usize, @intCast(committed_tokens - 1))];
        var prefix_args = try selected_prefix_exe.exe.args(allocator);
        defer prefix_args.deinit(allocator);
        var prefix_results = try selected_prefix_exe.exe.results(allocator);
        defer prefix_results.deinit(allocator);
        prefix_args.set(.{verified_hidden_buffer});
        selected_prefix_exe.exe.call(prefix_args, &prefix_results);

        const next_target_hidden_increment_buffer = prefix_results.get(zml.Buffer);
        if (owns_target_hidden_increment_buffer) target_hidden_increment_buffer.deinit();
        target_hidden_increment_buffer = next_target_hidden_increment_buffer;
        owns_target_hidden_increment_buffer = true;
        verified_hidden_buffer.deinit();

        const generated_step_end = @min(start, max_seq_len);
        var generated_step_text = try decodeTokens(allocator, &tokenizer, generated.items[generated_step_start..generated_step_end]);
        defer generated_step_text.deinit(allocator);
        var correction_text = try decodeTokens(allocator, &tokenizer, &.{correction_token});
        defer correction_text.deinit(allocator);

        try stdout.interface.print(
            "step={} start={} context_len={} valid_draft_tokens={} committed_tokens={} correction={} elapsed={f} correction_text=\"{s}\" text=\"{s}\"\n",
            .{ step, start, context_len, valid_draft_tokens, committed_tokens, correction_token, step_started_at.untilNow(io, .awake), correction_text.items, generated_step_text.items },
        );
    }
    if (owns_target_hidden_increment_buffer) target_hidden_increment_buffer.deinit();

    const decode_elapsed = decode_started_at.untilNow(io, .awake);
    const decoded_tokens: u32 = start - block_size;
    const decode_seconds = @as(f64, @floatFromInt(decode_elapsed.nanoseconds)) / std.time.ns_per_s;
    const tokens_per_second = @as(f64, @floatFromInt(decoded_tokens)) / decode_seconds;
    try printTokens(allocator, &tokenizer, &stdout.interface, "generated", generated.items[0..@min(generated.items.len, max_seq_len)]);
    try stdout.interface.print("target_cache_logical_len: {}\n", .{start});
    try stdout.interface.print("draft_cache_logical_len_after_crop: {}\n", .{draft_cache_len});
    try stdout.interface.print("decode_elapsed={f} decoded_tokens={} tokens_per_second={d:.3}\n", .{ decode_elapsed, decoded_tokens, tokens_per_second });
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

const DraftExe = struct {
    context_len: u32,
    position_len: u32,
    exe: zml.Exe,

    pub fn deinit(self: *DraftExe) void {
        self.exe.deinit();
    }
};

const PrefixSpec = struct {
    len: u32,
};

const PrefixExe = struct {
    len: u32,
    exe: zml.Exe,

    pub fn deinit(self: *PrefixExe) void {
        self.exe.deinit();
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

fn hiddenPrefix(spec: PrefixSpec, hidden: zml.Tensor) zml.Tensor {
    return hidden.withPartialTags(.{ .s, .d }).slice1d(.s, .{ .start = 0, .end = spec.len });
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

fn decodeTokens(
    allocator: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    token_ids: []const u32,
) !std.ArrayList(u8) {
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();
    return decoder.decodeAlloc(allocator, token_ids);
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

fn arangeAlloc(allocator: std.mem.Allocator, start: u32, len: u32) ![]u32 {
    const res = try allocator.alloc(u32, len);
    for (res, 0..) |*item, i| item.* = start + @as(u32, @intCast(i));
    return res;
}

fn setGeneratedToken(tokens: *std.ArrayList(u32), allocator: std.mem.Allocator, index: u32, token: u32) !void {
    const idx: usize = @intCast(index);
    if (idx < tokens.items.len) {
        tokens.items[idx] = token;
    } else {
        stdx.debug.assert(idx == tokens.items.len, "generated token stream has a gap at index {}", .{index});
        try tokens.append(allocator, token);
    }
}

fn initZeroTargetKvCacheBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    kv_cache: llama.KvCache,
    sharding: zml.Sharding,
) !llama.KvCache.Buffer {
    return .{
        .k = try zeroBuffer(allocator, io, platform, kv_cache.k.shape(), sharding),
        .v = try zeroBuffer(allocator, io, platform, kv_cache.v.shape(), sharding),
    };
}

fn initZeroDraftKvCacheBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    kv_cache: dflash.KvCache,
    sharding: zml.Sharding,
) !dflash.KvCache.Buffer {
    return .{
        .k = try zeroBuffer(allocator, io, platform, kv_cache.k.shape(), sharding),
        .v = try zeroBuffer(allocator, io, platform, kv_cache.v.shape(), sharding),
    };
}

fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromSlice(io, platform, zml.Slice.init(shape, bytes), sharding);
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
