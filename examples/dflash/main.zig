const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dflash = @import("dflash_model.zig");
const llama = @import("llama/llama_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.dflash);

const prefill_seq_len = 64;

const default_prompt =
    "Paris is the city of lights, fun, and love where everyone can enjoy the amazing culture, food, and art.";

const Args = struct {
    model: []const u8,
    target_model: []const u8,
    prompt: []const u8 = default_prompt,
    max_seq_len: u32 = 256,
    verbose: bool = false,

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
        \\  --verbose               Print per-step speculative decoding diagnostics
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

    const token_ids = try promptTokens(allocator, &tokenizer, parsed_target_config.value, args.prompt);
    defer allocator.free(token_ids);
    const prompt_len: u32 = @intCast(token_ids.len);
    if (token_ids.len == 0) {
        log.err("prompt produced no tokens", .{});
        return error.EmptyPrompt;
    }
    if (token_ids.len > prefill_seq_len) {
        std.debug.panic("prompt token count {} exceeds DFlash prefill length {}", .{ token_ids.len, prefill_seq_len });
    }
    if (args.max_seq_len <= token_ids.len) {
        log.err("--max-seq-len must be greater than prompt token count ({}), got {}", .{ token_ids.len, args.max_seq_len });
        return error.InvalidMaxSeqLen;
    }

    const draft_model = try dflash.Model.init(allocator, store.view(), parsed_config.value);
    defer draft_model.deinit(allocator);

    const block_size = parsed_config.value.block_size;
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
    const input_tokens_tensor: zml.Tensor = .init(.{ .s = prefill_seq_len }, .u32);
    const block_tokens_tensor: zml.Tensor = .init(.{ .s = block_size }, .u32);
    const token_index_tensor: zml.Tensor = .init(.{}, .u32);
    const active_context_len_tensor: zml.Tensor = .init(.{}, .u32);
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
    const attention_backend = zml.attention.attention.Backend.auto(platform);
    log.info("Selected target attention backend: {}", .{attention_backend});
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
            TargetPrefillContext{ .len = @intCast(token_ids.len) },
            input_tokens_tensor,
            token_index_tensor,
            target_kv_cache,
            attention_metadata,
            attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    defer target_prefill_exe.deinit();

    log.info("Compiling DFlash draft token variants...", .{});
    const draft_context_lens = [_]u32{ 1, 2, 3, 5, 10, prefill_seq_len };
    const draft_token_exes = try allocator.alloc(DraftTokenExe, draft_context_lens.len);
    defer {
        for (draft_token_exes) |*draft_exe| draft_exe.deinit();
        allocator.free(draft_token_exes);
    }
    const target_hidden_tensor: zml.Tensor = .init(.{
        .s = prefill_seq_len,
        .d = target_hidden_dim,
    }, target_model.model.embed_tokens.weight.dtype());
    for (draft_token_exes, draft_context_lens) |*draft_exe, context_len| {
        draft_exe.* = .{
            .context_len = context_len,
            .exe = try platform.compileFn(
                allocator,
                io,
                draftTokens,
                .{
                    DraftContext{ .len = context_len },
                    draft_model,
                    target_model,
                    target_hidden_tensor,
                    block_tokens_tensor,
                    draft_kv_cache,
                    token_index_tensor,
                    active_context_len_tensor,
                },
                .{ .shardings = &all_shardings },
            ),
        };
    }

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

    const padded_token_ids = try paddedPromptTokens(allocator, token_ids, prefill_seq_len);
    defer allocator.free(padded_token_ids);

    var input_tokens_buffer: zml.Buffer = try .fromSlice(
        io,
        platform,
        zml.Slice.init(input_tokens_tensor.shape(), std.mem.sliceAsBytes(padded_token_ids)),
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
    if (args.verbose) {
        try printTokens(allocator, &tokenizer, &stdout.interface, "prompt", token_ids);
        try printTokens(allocator, &tokenizer, &stdout.interface, "target_next", target_token_slice.constItems(u32));
    } else {
        var prompt_text = try decodeTokens(allocator, &tokenizer, token_ids);
        defer prompt_text.deinit(allocator);
        try stdout.interface.writeAll(prompt_text.items);
        try stdout.interface.flush();
    }

    var stream_decoder: ?zml.tokenizer.Tokenizer.Decoder = if (args.verbose) null else try tokenizer.decoder();
    defer if (stream_decoder) |*decoder| decoder.deinit();
    const decoder_out_buffer: []u8 = if (args.verbose) &.{} else try allocator.alloc(u8, 256);
    defer if (!args.verbose) allocator.free(decoder_out_buffer);

    var start: u32 = @intCast(token_ids.len);
    var draft_cache_base: u32 = 0;
    var target_hidden_block_buffer = target_hidden_buffer;
    var owns_target_hidden_block_buffer = false;
    var step: usize = 0;
    var total_valid_draft_tokens: usize = 0;
    var total_draft_tokens_checked: usize = 0;
    var full_accept_steps: usize = 0;
    var zero_accept_steps: usize = 0;
    var min_valid_draft_tokens: usize = block_size - 1;
    var max_valid_draft_tokens: usize = 0;
    const valid_draft_token_histogram = try allocator.alloc(usize, block_size);
    defer allocator.free(valid_draft_token_histogram);
    @memset(valid_draft_token_histogram, 0);
    const decode_started_at: std.Io.Timestamp = .now(io, .awake);
    var stopped_on_eos = false;
    while (start < max_seq_len and !stopped_on_eos) : (step += 1) {
        const step_started_at: std.Io.Timestamp = .now(io, .awake);
        const context_len = start - draft_cache_base;
        stdx.debug.assert(context_len >= 1 and context_len <= prefill_seq_len, "invalid DFlash context length {}", .{context_len});
        const selected_draft_exe = findDraftTokenExe(draft_token_exes, context_len);

        @memset(block_tokens, parsed_config.value.dflash_config.mask_token_id.?);
        block_tokens[0] = generated.items[@intCast(start)];

        var block_tokens_buffer: zml.Buffer = try .fromSlice(
            io,
            platform,
            zml.Slice.init(block_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
            .replicated,
        );
        defer block_tokens_buffer.deinit();

        var draft_cache_index = draft_cache_base;
        var draft_cache_index_buffer: zml.Buffer = try .fromSlice(
            io,
            platform,
            zml.Slice.init(token_index_tensor.shape(), std.mem.asBytes(&draft_cache_index)),
            .replicated,
        );
        defer draft_cache_index_buffer.deinit();

        var draft_active_context_len: u32 = context_len;
        var draft_active_context_len_buffer: zml.Buffer = try .fromSlice(
            io,
            platform,
            zml.Slice.init(active_context_len_tensor.shape(), std.mem.asBytes(&draft_active_context_len)),
            .replicated,
        );
        defer draft_active_context_len_buffer.deinit();

        var draft_args = try selected_draft_exe.exe.args(allocator);
        defer draft_args.deinit(allocator);
        var draft_results = try selected_draft_exe.exe.results(allocator);
        defer draft_results.deinit(allocator);
        draft_args.set(.{
            model_buffers,
            target_buffers,
            target_hidden_block_buffer,
            block_tokens_buffer,
            draft_kv_cache_buffers,
            draft_cache_index_buffer,
            draft_active_context_len_buffer,
        });
        selected_draft_exe.exe.call(draft_args, &draft_results);

        var draft_token_buffer, var updated_draft_kv_cache_buffers = draft_results.get(struct {
            zml.Buffer,
            dflash.KvCache.Buffer,
        });
        defer draft_token_buffer.deinit();
        replaceDraftKvCacheBuffers(&draft_kv_cache_buffers, &updated_draft_kv_cache_buffers);
        draft_cache_base = start;

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

        const verified_hidden_buffer, var posterior_token_buffer, var verified_target_kv_cache_buffers = verify_results.get(struct {
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
        total_valid_draft_tokens += valid_draft_tokens;
        total_draft_tokens_checked += block_size - 1;
        min_valid_draft_tokens = @min(min_valid_draft_tokens, valid_draft_tokens);
        max_valid_draft_tokens = @max(max_valid_draft_tokens, valid_draft_tokens);
        valid_draft_token_histogram[valid_draft_tokens] += 1;
        if (valid_draft_tokens == block_size - 1) full_accept_steps += 1;
        if (valid_draft_tokens == 0) zero_accept_steps += 1;
        const correction_token = posterior_tokens[valid_draft_tokens];
        const committed_tokens: u32 = @intCast(valid_draft_tokens + 1);

        const generated_step_start = start;
        for (block_tokens[0..@as(usize, @intCast(committed_tokens))]) |token| {
            if (isEosToken(&parsed_target_config.value, token)) {
                stopped_on_eos = true;
                break;
            }
            try setGeneratedToken(&generated, allocator, start, token);
            start += 1;
            if (start >= max_seq_len) break;
        }
        if (!stopped_on_eos and start < max_seq_len) {
            if (isEosToken(&parsed_target_config.value, correction_token)) {
                stopped_on_eos = true;
            } else {
                try setGeneratedToken(&generated, allocator, start, correction_token);
            }
        }

        // Keep the full verified target block on device. The next selected draft
        // executable slices the valid prefix using its static context length.
        if (owns_target_hidden_block_buffer) target_hidden_block_buffer.deinit();
        target_hidden_block_buffer = verified_hidden_buffer;
        owns_target_hidden_block_buffer = true;

        const generated_step_end = @min(start, max_seq_len);
        var generated_step_text = try decodeTokens(allocator, &tokenizer, generated.items[generated_step_start..generated_step_end]);
        defer generated_step_text.deinit(allocator);
        if (args.verbose) {
            var correction_text = try decodeTokens(allocator, &tokenizer, &.{correction_token});
            defer correction_text.deinit(allocator);
            try stdout.interface.print(
                "step={} start={} context_len={} valid_draft_tokens={} committed_tokens={} correction={} elapsed={f} correction_text=\"{s}\" text=\"{s}\"\n",
                .{ step, start, context_len, valid_draft_tokens, committed_tokens, correction_token, step_started_at.untilNow(io, .awake), correction_text.items, generated_step_text.items },
            );
        } else {
            for (generated.items[generated_step_start..generated_step_end]) |token| {
                try stdout.interface.writeAll(try stream_decoder.?.feedOne(token, decoder_out_buffer));
            }
            try stdout.interface.flush();
        }
    }
    if (owns_target_hidden_block_buffer) target_hidden_block_buffer.deinit();

    const decode_elapsed = decode_started_at.untilNow(io, .awake);
    const decoded_tokens: u32 = start - prompt_len;
    const decode_seconds = @as(f64, @floatFromInt(decode_elapsed.nanoseconds)) / std.time.ns_per_s;
    const tokens_per_second = @as(f64, @floatFromInt(decoded_tokens)) / decode_seconds;
    const draft_acceptance_rate = if (total_draft_tokens_checked == 0)
        0
    else
        @as(f64, @floatFromInt(total_valid_draft_tokens)) / @as(f64, @floatFromInt(total_draft_tokens_checked));
    const avg_valid_draft_tokens = if (step == 0)
        0
    else
        @as(f64, @floatFromInt(total_valid_draft_tokens)) / @as(f64, @floatFromInt(step));
    if (step == 0) min_valid_draft_tokens = 0;
    if (args.verbose) {
        var generated_text = try decodeTokens(allocator, &tokenizer, generated.items[0..@min(generated.items.len, max_seq_len)]);
        defer generated_text.deinit(allocator);
        try stdout.interface.print("\n--- generated_text ---\n{s}\n", .{generated_text.items});
    } else {
        try stdout.interface.writeAll(try stream_decoder.?.finalize(decoder_out_buffer));
        try stdout.interface.writeByte('\n');
    }
    try stdout.interface.print(
        \\
        \\--- decode_summary ---
        \\  elapsed: {f}
        \\  decoded_tokens: {}
        \\  tokens_per_second: {d:.3}
        \\  steps: {}
        \\  target_cache_logical_len: {}
        \\  draft_cache_base_index: {}
        \\  stopped_on_eos: {}
        \\  valid_draft_tokens:
        \\    avg: {d:.3}
        \\    min: {}
        \\    max: {}
        \\    total: {}
        \\  draft_acceptance_rate: {d:.3}
        \\  full_accept_steps: {}
        \\  zero_accept_steps: {}
        \\
    ,
        .{
            decode_elapsed,
            decoded_tokens,
            tokens_per_second,
            step,
            start,
            draft_cache_base,
            stopped_on_eos,
            avg_valid_draft_tokens,
            min_valid_draft_tokens,
            max_valid_draft_tokens,
            total_valid_draft_tokens,
            draft_acceptance_rate,
            full_accept_steps,
            zero_accept_steps,
        },
    );
    try stdout.interface.print("\n--- valid_draft_tokens_histogram ---\n", .{});
    const histogram_bar_width: usize = 80;
    for (valid_draft_token_histogram, 0..) |count, valid_tokens| {
        if (count == 0) {
            try stdout.interface.print("  {d:>2}: {d:>4}\n", .{ valid_tokens, count });
            continue;
        }
        const pct = if (step == 0)
            0
        else
            100.0 * @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(step));
        const bar_len = @max(@as(usize, 1), count * histogram_bar_width / step);
        try stdout.interface.print("  {d:>2}: {d:>4} ({d:>5.1}%)  ", .{ valid_tokens, count, pct });
        try stdout.interface.splatByteAll('x', bar_len);
        try stdout.interface.writeByte('\n');
    }
    try stdout.interface.flush();
}

const DraftContext = struct {
    len: u32,
};

const TargetPrefillContext = struct {
    len: u32,
};

const DraftTokenExe = struct {
    context_len: u32,
    exe: zml.Exe,

    pub fn deinit(self: *DraftTokenExe) void {
        self.exe.deinit();
    }
};

fn findDraftTokenExe(draft_token_exes: []DraftTokenExe, context_len: u32) *DraftTokenExe {
    for (draft_token_exes) |*draft_exe| {
        if (context_len <= draft_exe.context_len) return draft_exe;
    }
    std.debug.panic("no DFlash draft executable compiled for context length {}", .{context_len});
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
    context: TargetPrefillContext,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: llama.KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) struct { zml.Tensor, zml.Tensor, llama.KvCache } {
    const prefill_tokens = tokens.withPartialTags(.{.s}).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    const target_hidden, const target_token, const updated_kv_cache = target_model.dflashPrefill(
        prefill_tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
    );
    return .{ padTargetHidden(target_hidden, tokens.dim(.s)), target_token, updated_kv_cache };
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
    const target_hidden, const target_token, const updated_kv_cache = target_model.dflashVerify(
        tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
    );
    return .{ padTargetHidden(target_hidden, prefill_seq_len), target_token, updated_kv_cache };
}

fn draftTokens(
    context: DraftContext,
    draft_model: dflash.Model,
    target_model: llama.Model,
    target_hidden_block: zml.Tensor,
    block_tokens: zml.Tensor,
    draft_kv_cache: dflash.KvCache,
    cache_index: zml.Tensor,
    active_context_len: zml.Tensor,
) struct { zml.Tensor, dflash.KvCache } {
    const target_hidden_slice = target_hidden_block.withPartialTags(.{ .s, .d }).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    const active_mask = zml.Tensor.iota(.init(.{ .s = context.len }, .u32), .s).convert(.u32)
        .cmp(.LT, active_context_len.convert(.u32).broad(.init(.{ .s = context.len }, .u32)));
    const target_hidden = active_mask.broad(target_hidden_slice.shape())
        .select(target_hidden_slice, zml.Tensor.constant(target_hidden_slice.dtype().zero()).broad(target_hidden_slice.shape()));
    const noise_embedding = target_model.embedTokens(block_tokens);
    const context_position_ids = zml.Tensor.arange(.{ .end = context.len }, cache_index.dtype())
        .withTags(.{.s})
        .add(cache_index.broad(.init(.{ .s = context.len }, cache_index.dtype())));
    const proposal_position_ids = zml.Tensor.arange(.{ .end = block_tokens.dim(.s) }, cache_index.dtype())
        .withTags(.{.s})
        .add(active_context_len.convert(cache_index.dtype()).broad(.init(.{ .s = block_tokens.dim(.s) }, cache_index.dtype())))
        .add(cache_index.broad(.init(.{ .s = block_tokens.dim(.s) }, cache_index.dtype())));
    const position_ids = zml.Tensor.concatenate(&.{ context_position_ids, proposal_position_ids }, .s);
    const hidden, const updated_kv_cache = draft_model.forwardCached(target_hidden, noise_embedding, position_ids, draft_kv_cache, cache_index, active_context_len);
    return .{ target_model.greedyTokensFromHidden(hidden), updated_kv_cache };
}

fn padTargetHidden(target_hidden_: zml.Tensor, hidden_len: i64) zml.Tensor {
    const target_hidden = target_hidden_.withPartialTags(.{ .s, .d });
    if (target_hidden.dim(.s) == hidden_len) return target_hidden;

    stdx.debug.assert(target_hidden.dim(.s) < hidden_len, "target hidden length {} exceeds DFlash block size {}", .{ target_hidden.dim(.s), hidden_len });
    const padding = zml.Tensor.constant(target_hidden.dtype().zero()).broad(.init(.{
        .s = hidden_len - target_hidden.dim(.s),
        .d = target_hidden.dim(.d),
    }, target_hidden.dtype()));
    return zml.Tensor.concatenate(&.{ target_hidden, padding }, .s);
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
    prompt: []const u8,
) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const start_header = tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
    const end_header = tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
    const eot = tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
    const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
    try tokens.ensureUnusedCapacity(prompt.len);

    const w: *std.Io.Writer = &tokens.writer;
    try encoder.appendTokens(w, &.{ config.bos_token_id, start_header });
    try encoder.encode(w, "user");
    try encoder.appendTokens(w, &.{ end_header, newline });
    try encoder.encode(w, prompt);
    try encoder.appendTokens(w, &.{ eot, newline, start_header });
    try encoder.encode(w, "assistant");
    try encoder.appendTokens(w, &.{ end_header, newline });

    return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
}

fn paddedPromptTokens(allocator: std.mem.Allocator, token_ids: []const u32, block_size: u32) ![]u32 {
    const padded = try allocator.alloc(u32, block_size);
    @memset(padded, 0);
    @memcpy(padded[0..token_ids.len], token_ids);
    return padded;
}

fn firstEosToken(config: llama.Config) u32 {
    return switch (config.eos_token_id.value) {
        .int => |eos| eos,
        .ints => |eos_list| eos_list[0],
    };
}

fn isEosToken(config: *const llama.Config, token_id: u32) bool {
    return switch (config.eos_token_id.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
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
