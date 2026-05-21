const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const datasets = @import("datasets.zig");
const dflash = @import("dflash_model.zig");
const llama_inference = @import("llama/llama_inference.zig");
const llama = @import("llama/llama_model.zig");
const stats = @import("stats.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};
const log = std.log.scoped(.dflash_benchmark);

const default_prefill_seq_len = 64;

const Args = struct {
    model: []const u8,
    target_model: []const u8,
    dataset_path: []const u8,
    dataset: datasets.Dataset,
    split: []const u8 = "",
    samples: usize,
    seed: u64 = 1,
    max_new_tokens: u32 = 1024,
    max_prompt_tokens: u32 = 512,
    temperature: f32 = 0.0,
    output_json: []const u8 = "",
    verbose: bool = false,

    pub const help =
        \\Options:
        \\  --model=<path>             Path to the DFlash model repository
        \\  --target-model=<path>      Path to the target LLaMA model repository
        \\  --dataset-path=<path>      Path to JSON/JSONL dataset file
        \\  --dataset=<name>           math500|sharegpt|alpaca|swe_bench_lite|mt_bench|generic_jsonl|generic_json
        \\  --split=<name>             Dataset split label; defaults by dataset
        \\  --samples=<n>              Number of randomly selected valid samples
        \\  --seed=<n>                 Deterministic sample/RNG seed; defaults to 1
        \\  --max-new-tokens=<n>       Generated output tokens per method; defaults to 1024
        \\  --max-prompt-tokens=<n>    Minimum prompt cache capacity; defaults to 512; actual selected prompts may be longer and run in chunks
        \\  --temperature=<t>          Sampling temperature; 0 uses greedy decoding
        \\  --output-json=<path>       Optional result JSON output path
        \\  --verbose                  Print each selected prompt and both generated answers
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, Args);

    var project = try initProject(init, args);
    defer project.deinit();

    const split: ?[]const u8 = if (args.split.len == 0) null else args.split;
    var loaded_samples = datasets.loadSamples(allocator, init.io, .{
        .dataset = args.dataset,
        .split = split,
        .path = args.dataset_path,
        .samples = args.samples,
        .seed = args.seed,
    }) catch |err| switch (err) {
        datasets.LoadError.UnsupportedParquetDataset => {
            log.err("{s}", .{datasets.parquetConversionMessage()});
            return err;
        },
        datasets.LoadError.NotEnoughValidSamples => {
            log.err("dataset did not contain {} valid samples after filtering", .{args.samples});
            return err;
        },
        else => return err,
    };
    defer loaded_samples.deinit(allocator);

    log.info("Loaded dataset rows: total={} valid={} selected={} malformed={} empty={} overlong={}", .{
        loaded_samples.stats.total_rows,
        loaded_samples.stats.valid_rows,
        loaded_samples.stats.selected_rows,
        loaded_samples.stats.skipped_malformed,
        loaded_samples.stats.skipped_empty,
        loaded_samples.stats.skipped_overlong,
    });

    const tokenized_prompts = try tokenizeSamples(allocator, project, loaded_samples.samples, args.max_new_tokens);
    defer deinitTokenizedPrompts(allocator, tokenized_prompts);

    var models = try initModelsAndCaches(project, args, maxPromptLen(tokenized_prompts));
    defer models.deinit(allocator);

    var compiled = try compileShared(allocator, project, models);
    defer compiled.deinit(allocator);

    var loaded_buffers = try loadBuffers(allocator, project, models);
    defer loaded_buffers.deinit(allocator);

    var sample_results = try std.ArrayList(stats.SampleResult).initCapacity(allocator, loaded_samples.samples.len);
    defer {
        for (sample_results.items) |*sample_result| deinitSampleResult(allocator, sample_result);
        sample_results.deinit(allocator);
    }

    var stdout = std.Io.File.stdout().writerStreaming(project.io, &.{});
    const writer = &stdout.interface;

    for (loaded_samples.samples, 0..) |sample, i| {
        const prompt = tokenized_prompts[i] orelse {
            log.warn("skipping sample id={s}: EmptyPrompt", .{sample.id});
            continue;
        };

        var baseline_result = try runBaseline(
            allocator,
            project,
            &models,
            &compiled,
            loaded_buffers,
            prompt,
            args.seed +% @as(u64, @intCast(i)),
        );
        defer baseline_result.deinit(allocator);

        var dflash_result = try runDFlash(
            allocator,
            project,
            &models,
            &compiled,
            loaded_buffers,
            prompt,
            args.seed +% @as(u64, @intCast(i)),
        );
        defer dflash_result.deinit(allocator);

        const quality = stats.compareTokenIds(baseline_result.generated.items, dflash_result.generated.items);
        const dataset_name = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ @tagName(sample.source_dataset), sample.source_split });
        var owns_dataset_name = true;
        errdefer if (owns_dataset_name) allocator.free(dataset_name);
        const sample_id = try allocator.dupe(u8, sample.id);
        var owns_sample_id = true;
        errdefer if (owns_sample_id) allocator.free(sample_id);
        var baseline_text = try decodeTokens(allocator, &project.tokenizer, baseline_result.generated.items);
        const baseline_text_slice = baseline_text.toOwnedSlice(allocator) catch |err| {
            baseline_text.deinit(allocator);
            return err;
        };
        var owns_baseline_text_slice = true;
        errdefer if (owns_baseline_text_slice) allocator.free(baseline_text_slice);
        var dflash_text = try decodeTokens(allocator, &project.tokenizer, dflash_result.generated.items);
        const dflash_text_slice = dflash_text.toOwnedSlice(allocator) catch |err| {
            dflash_text.deinit(allocator);
            return err;
        };
        var owns_dflash_text_slice = true;
        errdefer if (owns_dflash_text_slice) allocator.free(dflash_text_slice);

        const baseline_tokens = try baseline_result.generated.toOwnedSlice(allocator);
        var owns_baseline_tokens = true;
        errdefer if (owns_baseline_tokens) allocator.free(baseline_tokens);
        const dflash_tokens = try dflash_result.generated.toOwnedSlice(allocator);
        var owns_dflash_tokens = true;
        errdefer if (owns_dflash_tokens) allocator.free(dflash_tokens);
        const acceptance_lengths = try dflash_result.acceptance_lengths.toOwnedSlice(allocator);
        var owns_acceptance_lengths = true;
        errdefer if (owns_acceptance_lengths) allocator.free(acceptance_lengths);
        const committed_lengths = try dflash_result.committed_lengths.toOwnedSlice(allocator);
        var owns_committed_lengths = true;
        errdefer if (owns_committed_lengths) allocator.free(committed_lengths);
        const histogram = try allocator.dupe(u64, dflash_result.valid_draft_token_histogram);
        var owns_histogram = true;
        errdefer if (owns_histogram) allocator.free(histogram);

        const sample_result: stats.SampleResult = .{
            .id = sample_id,
            .dataset = dataset_name,
            .prompt_tokens = prompt.token_ids.len,
            .baseline = .{
                .token_ids = baseline_tokens,
                .decoded_tokens = baseline_tokens.len,
                .generated_text = baseline_text_slice,
                .elapsed = .{ .nanoseconds = baseline_result.decode_elapsed.nanoseconds },
                .stopped_on_eos = baseline_result.stopped_on_eos,
            },
            .dflash = .{
                .token_ids = dflash_tokens,
                .decoded_tokens = dflash_tokens.len,
                .generated_text = dflash_text_slice,
                .elapsed = .{ .nanoseconds = dflash_result.decode_elapsed.nanoseconds },
                .stopped_on_eos = dflash_result.stopped_on_eos,
                .acceptance_lengths = acceptance_lengths,
                .committed_lengths = committed_lengths,
                .acceptance_length_histogram = histogram,
            },
            .quality = quality,
        };

        try stats.printSample(writer, i + 1, loaded_samples.samples.len, sample_result);
        if (args.verbose) {
            try printVerboseSample(writer, sample.prompt, baseline_text_slice, dflash_text_slice);
        }
        try writer.flush();
        try sample_results.append(allocator, sample_result);
        owns_dataset_name = false;
        owns_sample_id = false;
        owns_baseline_text_slice = false;
        owns_dflash_text_slice = false;
        owns_baseline_tokens = false;
        owns_dflash_tokens = false;
        owns_acceptance_lengths = false;
        owns_committed_lengths = false;
        owns_histogram = false;
    }

    const summary = try stats.computeSummary(allocator, sample_results.items);
    defer summary.deinit(allocator);
    try stats.printReport(writer, summary);
    try writer.flush();

    if (args.output_json.len != 0) {
        const output = try std.Io.Dir.createFile(.cwd(), project.io, args.output_json, .{});
        defer output.close(project.io);
        var json_writer = output.writerStreaming(project.io, &.{});
        try writeJsonReport(&json_writer.interface, args, split, sample_results.items, summary);
        try json_writer.interface.flush();
    }
}

fn printVerboseSample(
    writer: *std.Io.Writer,
    prompt: []const u8,
    baseline_text: []const u8,
    dflash_text: []const u8,
) !void {
    try writer.writeAll("  Prompt:\n");
    try writeIndentedBlock(writer, prompt);
    try writer.writeAll("  Baseline answer:\n");
    try writeIndentedBlock(writer, baseline_text);
    try writer.writeAll("  DFlash answer:\n");
    try writeIndentedBlock(writer, dflash_text);
}

fn writeIndentedBlock(writer: *std.Io.Writer, text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        try writer.writeAll("    ");
        try writer.writeAll(line);
        try writer.writeByte('\n');
    }
}

fn writeJsonReport(
    writer: *std.Io.Writer,
    args: Args,
    split: ?[]const u8,
    sample_results: []const stats.SampleResult,
    summary: stats.Summary,
) !void {
    try writer.writeAll("{\"config\":{");
    try writer.writeAll("\"model\":");
    try writeJsonString(writer, args.target_model);
    try writer.writeAll(",\"draft_model\":");
    try writeJsonString(writer, args.model);
    try writer.writeAll(",\"dataset\":");
    try writeJsonString(writer, @tagName(args.dataset));
    try writer.writeAll(",\"split\":");
    try writeJsonString(writer, split orelse "");
    try writer.print(",\"sample_count\":{d},\"seed\":{d},\"max_tokens\":{d},\"temperature\":{d}", .{
        sample_results.len,
        args.seed,
        args.max_new_tokens,
        args.temperature,
    });
    try writer.writeAll("},\"samples\":[");
    for (sample_results, 0..) |sample, i| {
        if (i != 0) try writer.writeByte(',');
        try writeSampleJson(writer, sample);
    }
    try writer.writeAll("],\"summary\":");
    try writeSummaryJson(writer, summary);
    try writer.writeAll("}\n");
}

fn writeSampleJson(writer: *std.Io.Writer, sample: stats.SampleResult) !void {
    try writer.writeAll("{\"id\":");
    try writeJsonString(writer, sample.id);
    try writer.writeAll(",\"dataset\":");
    try writeJsonString(writer, sample.dataset);
    try writer.print(",\"prompt_tokens\":{d},\"baseline\":", .{sample.prompt_tokens});
    try writeMethodJson(writer, sample.baseline, false);
    try writer.writeAll(",\"dflash\":");
    try writeMethodJson(writer, sample.dflash, true);
    try writer.print(",\"quality\":{{\"exact_match\":{},\"compared_tokens\":{d},\"first_mismatch_index\":", .{
        sample.quality.exact_match,
        sample.quality.compared_tokens,
    });
    if (sample.quality.first_mismatch_index) |idx| {
        try writer.print("{d}", .{idx});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll("}}}");
}

fn writeMethodJson(writer: *std.Io.Writer, method: stats.MethodResult, include_acceptance: bool) !void {
    try writer.print("{{\"decoded_tokens\":{d},\"elapsed_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tps\":{d:.3},\"stopped_on_eos\":{},\"token_ids\":", .{
        method.tokenCount(),
        durationMs(method.elapsed),
        method.tpotMs(),
        method.tokensPerSecond(),
        method.stopped_on_eos,
    });
    try writeU32Array(writer, method.token_ids);
    try writer.writeAll(",\"generated_text\":");
    try writeJsonString(writer, method.generated_text);
    if (include_acceptance) {
        try writer.writeAll(",\"acceptance_lengths\":");
        try writeU32Array(writer, method.acceptance_lengths);
        try writer.writeAll(",\"committed_lengths\":");
        try writeU32Array(writer, method.committed_lengths);
        try writer.writeAll(",\"acceptance_length_histogram\":");
        try writeU64Array(writer, method.acceptance_length_histogram);
    }
    try writer.writeByte('}');
}

fn writeSummaryJson(writer: *std.Io.Writer, summary: stats.Summary) !void {
    try writer.writeAll("{\"dataset\":");
    try writeJsonString(writer, summary.dataset);
    try writer.print(",\"sample_count\":{d},\"baseline_tpot_ms\":{d:.6},\"baseline_tps\":{d:.6},\"dflash_tpot_ms\":{d:.6},\"dflash_tps\":{d:.6},\"speedup\":{d:.6},\"tau\":{d:.6},\"exact_match_count\":{d}", .{
        summary.sample_count,
        summary.baseline_tpot_ms,
        summary.baseline_tps,
        summary.dflash_tpot_ms,
        summary.dflash_tps,
        summary.speedup,
        summary.tau,
        summary.exact_match_count,
    });
    try writer.writeAll(",\"per_position_acceptance_rates\":");
    try writeF64Array(writer, summary.per_position_acceptance_rates);
    try writer.writeAll(",\"acceptance_length_histogram\":");
    try writeU64Array(writer, summary.acceptance_length_histogram);
    try writer.writeAll(",\"acceptance_length_histogram_rates\":");
    try writeF64Array(writer, summary.acceptance_length_histogram_rates);
    try writer.writeByte('}');
}

fn durationMs(duration: stats.Duration) f64 {
    return @as(f64, @floatFromInt(duration.nanoseconds)) / std.time.ns_per_ms;
}

fn writeJsonString(writer: *std.Io.Writer, value: []const u8) !void {
    try writer.writeByte('"');
    for (value) |c| switch (c) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...8, 11, 12, 14...0x1f => try writer.print("\\u{x:0>4}", .{c}),
        else => try writer.writeByte(c),
    };
    try writer.writeByte('"');
}

fn writeU32Array(writer: *std.Io.Writer, items: []const u32) !void {
    try writer.writeByte('[');
    for (items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writer.print("{d}", .{item});
    }
    try writer.writeByte(']');
}

fn writeU64Array(writer: *std.Io.Writer, items: []const u64) !void {
    try writer.writeByte('[');
    for (items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writer.print("{d}", .{item});
    }
    try writer.writeByte(']');
}

fn writeF64Array(writer: *std.Io.Writer, items: []const f64) !void {
    try writer.writeByte('[');
    for (items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writer.print("{d:.6}", .{item});
    }
    try writer.writeByte(']');
}

const Project = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    vfs_file: zml.io.VFS.File,
    http_client: std.http.Client,
    hf_vfs: zml.io.VFS.HF,
    vfs: zml.io.VFS,
    platform: *zml.Platform,
    repo: std.Io.Dir,
    target_repo: std.Io.Dir,
    parsed_config: std.json.Parsed(dflash.Config),
    parsed_target_config: std.json.Parsed(llama.Config),
    registry: zml.safetensors.TensorRegistry,
    target_registry: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,
    target_store: zml.io.TensorStore,
    tokenizer: zml.tokenizer.Tokenizer,
    shardings: Shardings,

    pub fn deinit(self: *Project) void {
        const allocator = self.allocator;
        self.tokenizer.deinit();
        self.target_store.deinit();
        self.store.deinit();
        self.target_registry.deinit();
        self.registry.deinit();
        self.parsed_target_config.deinit();
        self.parsed_config.deinit();
        self.platform.deinit(allocator, self.io);
        self.vfs.deinit();
        self.hf_vfs.deinit();
        self.http_client.deinit();
        self.vfs_file.deinit();
        allocator.destroy(self);
    }
};

fn initProject(init: std.process.Init, args: Args) !*Project {
    const allocator = init.gpa;
    const project = try allocator.create(Project);
    errdefer allocator.destroy(project);

    project.allocator = allocator;
    project.vfs_file = .init(allocator, init.io, .{});
    errdefer project.vfs_file.deinit();

    project.http_client = .{ .allocator = allocator, .io = init.io };
    errdefer project.http_client.deinit();

    project.hf_vfs = try .auto(allocator, init.io, &project.http_client, init.environ_map);
    errdefer project.hf_vfs.deinit();

    project.vfs = try .init(allocator, init.io);
    errdefer project.vfs.deinit();

    try project.vfs.register("file", project.vfs_file.io());
    try project.vfs.register("hf", project.hf_vfs.io());
    project.io = project.vfs.io();

    project.platform = try .auto(allocator, project.io, .{});
    errdefer project.platform.deinit(allocator, project.io);
    log.info("\n{f}", .{project.platform.fmtVerbose()});

    project.repo = try zml.safetensors.resolveModelRepo(project.io, args.model);
    project.target_repo = try zml.safetensors.resolveModelRepo(project.io, args.target_model);

    project.parsed_config = try parseConfig(dflash.Config, allocator, project.io, project.repo);
    errdefer project.parsed_config.deinit();
    project.parsed_target_config = try parseConfig(llama.Config, allocator, project.io, project.target_repo);
    errdefer project.parsed_target_config.deinit();

    project.registry = try .fromRepo(allocator, project.io, project.repo);
    errdefer project.registry.deinit();
    project.target_registry = try .fromRepo(allocator, project.io, project.target_repo);
    errdefer project.target_registry.deinit();

    project.store = .fromRegistry(allocator, &project.registry);
    errdefer project.store.deinit();
    project.target_store = .fromRegistry(allocator, &project.target_registry);
    errdefer project.target_store.deinit();

    project.tokenizer = try llama.loadTokenizer(allocator, project.io, project.target_repo);
    errdefer project.tokenizer.deinit();

    project.shardings = try .init(project.platform);
    return project;
}

const TokenizedPrompt = struct {
    token_ids: []u32,
    prompt_len: u32,
    max_seq_len: u32,

    pub fn deinit(self: TokenizedPrompt, allocator: std.mem.Allocator) void {
        allocator.free(self.token_ids);
    }
};

fn tokenizeSamples(allocator: std.mem.Allocator, project: *Project, samples: []const datasets.Sample, max_new_tokens: u32) ![]?TokenizedPrompt {
    const prompts = try allocator.alloc(?TokenizedPrompt, samples.len);
    errdefer allocator.free(prompts);
    @memset(prompts, null);
    errdefer deinitTokenizedPrompts(allocator, prompts);

    for (samples, 0..) |sample, i| {
        prompts[i] = tokenizePrompt(project, sample.prompt, max_new_tokens) catch |err| switch (err) {
            error.EmptyPrompt => null,
            else => return err,
        };
    }
    return prompts;
}

fn deinitTokenizedPrompts(allocator: std.mem.Allocator, prompts: []?TokenizedPrompt) void {
    for (prompts) |maybe_prompt| {
        if (maybe_prompt) |prompt| prompt.deinit(allocator);
    }
    allocator.free(prompts);
}

fn maxPromptLen(prompts: []const ?TokenizedPrompt) u32 {
    var max_len: u32 = 0;
    for (prompts) |maybe_prompt| {
        if (maybe_prompt) |prompt| max_len = @max(max_len, prompt.prompt_len);
    }
    return max_len;
}

fn tokenizePrompt(project: *Project, prompt_text: []const u8, max_new_tokens: u32) !TokenizedPrompt {
    const token_ids = try llama.tokenizePrompt(project.allocator, &project.tokenizer, project.parsed_target_config.value, prompt_text);
    errdefer project.allocator.free(token_ids);

    if (token_ids.len == 0) return error.EmptyPrompt;

    return .{
        .token_ids = token_ids,
        .prompt_len = @intCast(token_ids.len),
        .max_seq_len = @intCast(token_ids.len + max_new_tokens),
    };
}

const ModelsAndCaches = struct {
    draft_model: dflash.Model,
    target_model: llama.Model,
    input_tokens_tensor: zml.Tensor,
    block_tokens_tensor: zml.Tensor,
    decode_token_tensor: zml.Tensor,
    token_index_tensor: zml.Tensor,
    active_context_len_tensor: zml.Tensor,
    target_kv_cache: llama.KvCache,
    draft_kv_cache: dflash.KvCache,
    rng: zml.Tensor.Rng,
    sampling: llama.SamplingConfig,
    target_attention: llama_inference.TargetAttention,
    target_layers: llama_inference.TargetLayers,
    block_size: u32,

    pub fn deinit(self: ModelsAndCaches, allocator: std.mem.Allocator) void {
        self.target_model.deinit(allocator);
        self.draft_model.deinit(allocator);
    }
};

fn initModelsAndCaches(project: *Project, args: Args, selected_max_prompt_tokens: u32) !ModelsAndCaches {
    const draft_model = try dflash.Model.init(project.allocator, project.store.view(), project.parsed_config.value);
    errdefer draft_model.deinit(project.allocator);

    const target_model = try llama.Model.init(project.allocator, project.target_store.view(), project.parsed_target_config.value);
    errdefer target_model.deinit(project.allocator);

    const block_size = project.parsed_config.value.block_size;
    const cache_prompt_tokens = @max(args.max_prompt_tokens, selected_max_prompt_tokens);
    const cache_seq_len = @max(cache_prompt_tokens + args.max_new_tokens, default_prefill_seq_len) + block_size;
    const target_attention = llama_inference.TargetAttention.init(project.platform, cache_seq_len, project.parsed_target_config.value.num_attention_heads);
    log.info("Selected target attention backend: {}", .{target_attention.backend});

    return .{
        .draft_model = draft_model,
        .target_model = target_model,
        .input_tokens_tensor = .init(.{ .s = default_prefill_seq_len }, .u32),
        .block_tokens_tensor = .init(.{ .s = block_size }, .u32),
        .decode_token_tensor = .init(.{ .s = 1 }, .u32),
        .token_index_tensor = .init(.{}, .u32),
        .active_context_len_tensor = .init(.{}, .u32),
        .target_kv_cache = target_model.initKvCache(project.parsed_target_config.value, cache_seq_len),
        .draft_kv_cache = draft_model.initKvCache(project.parsed_config.value, cache_seq_len),
        .rng = .init(),
        .sampling = .{ .temperature = args.temperature },
        .target_attention = target_attention,
        .target_layers = llama_inference.TargetLayers.init(draft_model.target_layer_ids),
        .block_size = block_size,
    };
}

const CompiledExes = struct {
    target_prefill_exe: zml.Exe,
    prefill_draft_token_exe: DraftTokenExe,
    draft_token_exe: DraftTokenExe,
    target_verify_exe: zml.Exe,
    target_decode_exe: zml.Exe,

    pub fn deinit(self: *CompiledExes, allocator: std.mem.Allocator) void {
        self.target_decode_exe.deinit();
        self.target_verify_exe.deinit();
        self.draft_token_exe.deinit(allocator);
        self.prefill_draft_token_exe.deinit(allocator);
        self.target_prefill_exe.deinit();
    }
};

fn compileShared(allocator: std.mem.Allocator, project: *Project, models: ModelsAndCaches) !CompiledExes {
    const all_shardings = project.shardings.all();

    log.info("Compiling target prefill chunk...", .{});
    const target_prefill_exe = try llama_inference.compileTargetPrefill(
        allocator,
        project.io,
        project.platform,
        models.target_model,
        models.target_layers,
        default_prefill_seq_len,
        models.input_tokens_tensor,
        models.active_context_len_tensor,
        models.token_index_tensor,
        models.target_kv_cache,
        models.rng,
        models.sampling,
        models.target_attention,
        &all_shardings,
    );
    errdefer target_prefill_exe.deinit();

    log.info("Compiling DFlash prefill drafter...", .{});
    const prefill_target_hidden_tensor = llama_inference.targetHiddenTensor(models.target_model, project.parsed_target_config.value, models.target_layers, default_prefill_seq_len);
    var prefill_draft_token_exe = try compileDraftTokenExe(allocator, project, models, &all_shardings, default_prefill_seq_len, prefill_target_hidden_tensor);
    errdefer prefill_draft_token_exe.deinit(allocator);

    log.info("Compiling DFlash steady-state drafter...", .{});
    const draft_target_hidden_tensor = llama_inference.targetHiddenTensor(models.target_model, project.parsed_target_config.value, models.target_layers, models.block_size);
    var draft_token_exe = try compileDraftTokenExe(allocator, project, models, &all_shardings, models.block_size, draft_target_hidden_tensor);
    errdefer draft_token_exe.deinit(allocator);

    log.info("Compiling target verify...", .{});
    const target_verify_exe = try llama_inference.compileTargetVerify(
        allocator,
        project.io,
        project.platform,
        models.target_model,
        models.target_layers,
        models.block_size,
        models.block_tokens_tensor,
        models.token_index_tensor,
        models.target_kv_cache,
        models.rng,
        models.sampling,
        models.target_attention,
        &all_shardings,
    );
    errdefer target_verify_exe.deinit();

    log.info("Compiling target decode...", .{});
    const target_decode_exe = try project.platform.compileFn(
        allocator,
        project.io,
        targetDecode,
        .{
            models.target_model,
            models.target_layers,
            models.decode_token_tensor,
            models.token_index_tensor,
            models.target_kv_cache,
            models.rng,
            models.sampling,
            models.target_attention.metadata,
            models.target_attention.parameters,
        },
        .{ .shardings = &all_shardings },
    );

    return .{
        .target_prefill_exe = target_prefill_exe,
        .prefill_draft_token_exe = prefill_draft_token_exe,
        .draft_token_exe = draft_token_exe,
        .target_verify_exe = target_verify_exe,
        .target_decode_exe = target_decode_exe,
    };
}

fn compileDraftTokenExe(
    allocator: std.mem.Allocator,
    project: *Project,
    models: ModelsAndCaches,
    shardings: []const zml.Sharding,
    context_len: u32,
    target_hidden_tensor: zml.Tensor,
) !DraftTokenExe {
    var draft_token_exe: DraftTokenExe = .{
        .exe = try project.platform.compileFn(
            allocator,
            project.io,
            draftTokens,
            .{
                DraftContext{ .len = context_len },
                models.draft_model,
                models.target_model,
                target_hidden_tensor,
                models.block_tokens_tensor,
                models.draft_kv_cache,
                models.token_index_tensor,
                models.active_context_len_tensor,
                models.rng,
                models.sampling,
            },
            .{ .shardings = shardings },
        ),
        .args = undefined,
        .results = undefined,
    };
    errdefer draft_token_exe.exe.deinit();

    draft_token_exe.args = try draft_token_exe.exe.args(allocator);
    errdefer draft_token_exe.args.deinit(allocator);

    draft_token_exe.results = try draft_token_exe.exe.results(allocator);
    return draft_token_exe;
}

const LoadedBuffers = struct {
    model_buffers: dflash.Buffers,
    target_buffers: llama.Buffers,

    pub fn deinit(self: *LoadedBuffers, allocator: std.mem.Allocator) void {
        llama.Model.unloadBuffers(&self.target_buffers, allocator);
        dflash.Model.unloadBuffers(&self.model_buffers, allocator);
    }
};

fn loadBuffers(allocator: std.mem.Allocator, project: *Project, models: ModelsAndCaches) !LoadedBuffers {
    const all_shardings = project.shardings.all();
    log.info("Loading weights...", .{});
    var model_buffers = try models.draft_model.loadBuffers(allocator, project.io, project.platform, &project.store, &all_shardings);
    errdefer dflash.Model.unloadBuffers(&model_buffers, allocator);
    const target_buffers = try models.target_model.loadBuffers(allocator, project.io, project.platform, &project.target_store, &all_shardings);
    return .{
        .model_buffers = model_buffers,
        .target_buffers = target_buffers,
    };
}

const PrefillOutput = struct {
    target_hidden_buffer: zml.Buffer,
    target_token: u32,
};

fn deinitPrefillOutput(output: PrefillOutput) void {
    var copy = output;
    copy.target_hidden_buffer.deinit();
}

fn runPrefill(
    allocator: std.mem.Allocator,
    project: *Project,
    target_prefill_exe: zml.Exe,
    loaded_buffers: LoadedBuffers,
    input_tokens_buffer: zml.Buffer,
    active_len_buffer: zml.Buffer,
    token_index_buffer: zml.Buffer,
    target_kv_cache_buffers: *llama.KvCache.Buffer,
    rng_buffer: *zml.Tensor.Rng.Buffer,
    attention_metadata_buffers: anytype,
) !PrefillOutput {
    var prefill_args = try target_prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);

    var prefill_results = try target_prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    prefill_args.set(.{
        loaded_buffers.target_buffers,
        input_tokens_buffer,
        active_len_buffer,
        token_index_buffer,
        target_kv_cache_buffers.*,
        rng_buffer.*,
        attention_metadata_buffers,
    });
    target_prefill_exe.call(prefill_args, &prefill_results);

    const target_hidden_buffer, var target_token_buffer, var prefilled_target_kv_cache_buffers, var updated_rng_buffer = prefill_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        llama.KvCache.Buffer,
        zml.Tensor.Rng.Buffer,
    });
    defer target_token_buffer.deinit();
    llama.KvCache.replaceBuffers(target_kv_cache_buffers, &prefilled_target_kv_cache_buffers);
    replaceRngBuffer(rng_buffer, &updated_rng_buffer);

    var target_token_slice = try target_token_buffer.toSliceAlloc(allocator, project.io);
    defer target_token_slice.free(allocator);

    return .{
        .target_hidden_buffer = target_hidden_buffer,
        .target_token = target_token_slice.constItems(u32)[0],
    };
}

const MethodRunResult = struct {
    generated: std.ArrayList(u32),
    acceptance_lengths: std.ArrayList(u32),
    committed_lengths: std.ArrayList(u32),
    valid_draft_token_histogram: []u64,
    decode_elapsed: stats.Duration,
    stopped_on_eos: bool,

    pub fn deinit(self: *MethodRunResult, allocator: std.mem.Allocator) void {
        allocator.free(self.valid_draft_token_histogram);
        self.committed_lengths.deinit(allocator);
        self.acceptance_lengths.deinit(allocator);
        self.generated.deinit(allocator);
    }
};

const RuntimeState = struct {
    target_kv_cache_buffers: llama.KvCache.Buffer,
    rng_buffer: zml.Tensor.Rng.Buffer,
    attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata),

    pub fn deinit(self: *RuntimeState) void {
        llama.KvCache.deinitBuffer(&self.target_kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffer);
        zml.attention.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
    }
};

fn initInputAndState(
    project: *Project,
    models: *ModelsAndCaches,
    seed: u64,
) !RuntimeState {
    var target_kv_cache_buffers = try models.target_kv_cache.initZeroBuffer(project.allocator, project.io, project.platform, project.shardings.model);
    errdefer llama.KvCache.deinitBuffer(&target_kv_cache_buffers);

    var rng_buffer = try zml.Tensor.Rng.initBuffer(project.io, project.platform, .replicated, seed);
    errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    var attention_metadata_buffers = try models.target_attention.metadata.initBuffer(project.io, project.platform, project.shardings.model);
    errdefer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

    return .{
        .target_kv_cache_buffers = target_kv_cache_buffers,
        .rng_buffer = rng_buffer,
        .attention_metadata_buffers = attention_metadata_buffers,
    };
}

fn runPrefillChunks(
    allocator: std.mem.Allocator,
    project: *Project,
    models: *ModelsAndCaches,
    compiled: *CompiledExes,
    loaded_buffers: LoadedBuffers,
    prompt: TokenizedPrompt,
    target_kv_cache_buffers: *llama.KvCache.Buffer,
    rng_buffer: *zml.Tensor.Rng.Buffer,
    attention_metadata_buffers: anytype,
) !PrefillOutput {
    var offset: usize = 0;
    var final_output: ?PrefillOutput = null;
    errdefer if (final_output) |output| deinitPrefillOutput(output);
    const tail_len = @min(prompt.token_ids.len, default_prefill_seq_len);
    const tail_start = prompt.token_ids.len - tail_len;

    while (offset < prompt.token_ids.len) {
        const remaining = prompt.token_ids.len - offset;
        const active_len: u32 = if (remaining <= default_prefill_seq_len and offset != tail_start)
            @intCast(tail_len)
        else
            @intCast(@min(remaining, default_prefill_seq_len));
        const chunk_offset = if (remaining <= default_prefill_seq_len and offset != tail_start) tail_start else offset;
        const active_len_usize: usize = @intCast(active_len);
        const chunk_tokens = try paddedPromptTokens(allocator, prompt.token_ids[chunk_offset..][0..active_len_usize], default_prefill_seq_len);
        defer allocator.free(chunk_tokens);

        var input_tokens_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.input_tokens_tensor.shape(), std.mem.sliceAsBytes(chunk_tokens)),
            .replicated,
        );
        defer input_tokens_buffer.deinit();

        var active_len_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.active_context_len_tensor.shape(), std.mem.asBytes(&active_len)),
            .replicated,
        );
        defer active_len_buffer.deinit();

        var token_index: u32 = @intCast(chunk_offset);
        var token_index_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.token_index_tensor.shape(), std.mem.asBytes(&token_index)),
            .replicated,
        );
        defer token_index_buffer.deinit();

        const output = try runPrefill(
            allocator,
            project,
            compiled.target_prefill_exe,
            loaded_buffers,
            input_tokens_buffer,
            active_len_buffer,
            token_index_buffer,
            target_kv_cache_buffers,
            rng_buffer,
            attention_metadata_buffers,
        );

        if (final_output) |previous| deinitPrefillOutput(previous);
        final_output = output;
        offset += if (chunk_offset == tail_start and offset != tail_start) remaining else active_len;
    }

    const output = final_output.?;
    final_output = null;
    return output;
}

fn runBaseline(
    allocator: std.mem.Allocator,
    project: *Project,
    models: *ModelsAndCaches,
    compiled: *CompiledExes,
    loaded_buffers: LoadedBuffers,
    prompt: TokenizedPrompt,
    seed: u64,
) !MethodRunResult {
    var state = try initInputAndState(project, models, seed);
    defer state.deinit();

    const prefill_output = try runPrefillChunks(
        allocator,
        project,
        models,
        compiled,
        loaded_buffers,
        prompt,
        &state.target_kv_cache_buffers,
        &state.rng_buffer,
        state.attention_metadata_buffers,
    );
    defer deinitPrefillOutput(prefill_output);

    var generated = try std.ArrayList(u32).initCapacity(allocator, prompt.max_seq_len - prompt.prompt_len);
    errdefer generated.deinit(allocator);
    const started_at: std.Io.Timestamp = .now(project.io, .awake);
    try generated.append(allocator, prefill_output.target_token);

    var decode_args = try compiled.target_decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try compiled.target_decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    var current_token = prefill_output.target_token;
    var stopped_on_eos = llama.isEosToken(&project.parsed_target_config.value, current_token);
    var absolute_pos: u32 = prompt.prompt_len + 1;

    while (absolute_pos < prompt.max_seq_len and !stopped_on_eos) : (absolute_pos += 1) {
        var current_token_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.decode_token_tensor.shape(), std.mem.asBytes(&current_token)),
            .replicated,
        );
        defer current_token_buffer.deinit();

        var token_index = absolute_pos - 1;
        var token_index_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.token_index_tensor.shape(), std.mem.asBytes(&token_index)),
            .replicated,
        );
        defer token_index_buffer.deinit();

        decode_args.set(.{
            loaded_buffers.target_buffers,
            current_token_buffer,
            token_index_buffer,
            state.target_kv_cache_buffers,
            state.rng_buffer,
            state.attention_metadata_buffers,
        });
        compiled.target_decode_exe.call(decode_args, &decode_results);

        var token_buffer, var updated_target_kv_cache_buffers, var updated_rng_buffer = decode_results.get(struct {
            zml.Buffer,
            llama.KvCache.Buffer,
            zml.Tensor.Rng.Buffer,
        });
        defer token_buffer.deinit();
        llama.KvCache.replaceBuffers(&state.target_kv_cache_buffers, &updated_target_kv_cache_buffers);
        replaceRngBuffer(&state.rng_buffer, &updated_rng_buffer);

        var token_slice = try token_buffer.toSliceAlloc(allocator, project.io);
        defer token_slice.free(allocator);
        current_token = token_slice.constItems(u32)[0];
        if (llama.isEosToken(&project.parsed_target_config.value, current_token)) {
            stopped_on_eos = true;
        } else {
            try generated.append(allocator, current_token);
        }
    }

    const histogram = try allocator.alloc(u64, 0);
    errdefer allocator.free(histogram);
    return .{
        .generated = generated,
        .acceptance_lengths = .empty,
        .committed_lengths = .empty,
        .valid_draft_token_histogram = histogram,
        .decode_elapsed = .{ .nanoseconds = started_at.untilNow(project.io, .awake).nanoseconds },
        .stopped_on_eos = stopped_on_eos,
    };
}

fn runDFlash(
    allocator: std.mem.Allocator,
    project: *Project,
    models: *ModelsAndCaches,
    compiled: *CompiledExes,
    loaded_buffers: LoadedBuffers,
    prompt: TokenizedPrompt,
    seed: u64,
) !MethodRunResult {
    var state = try initInputAndState(project, models, seed);
    defer state.deinit();

    var draft_kv_cache_buffers = try models.draft_kv_cache.initZeroBuffer(allocator, project.io, project.platform, project.shardings.model);
    defer dflash.KvCache.deinitBuffer(&draft_kv_cache_buffers);

    const prefill_output = try runPrefillChunks(
        allocator,
        project,
        models,
        compiled,
        loaded_buffers,
        prompt,
        &state.target_kv_cache_buffers,
        &state.rng_buffer,
        state.attention_metadata_buffers,
    );

    var generated = try std.ArrayList(u32).initCapacity(allocator, prompt.max_seq_len + 1);
    errdefer generated.deinit(allocator);
    try generated.appendSlice(allocator, prompt.token_ids);
    try generated.append(allocator, prefill_output.target_token);

    var output_tokens = try std.ArrayList(u32).initCapacity(allocator, prompt.max_seq_len - prompt.prompt_len);
    errdefer output_tokens.deinit(allocator);

    var acceptance_lengths: std.ArrayList(u32) = .empty;
    errdefer acceptance_lengths.deinit(allocator);
    var committed_lengths: std.ArrayList(u32) = .empty;
    errdefer committed_lengths.deinit(allocator);
    var valid_draft_token_histogram = try allocator.alloc(u64, models.block_size + 1);
    errdefer allocator.free(valid_draft_token_histogram);
    @memset(valid_draft_token_histogram, 0);

    var block_tokens = try allocator.alloc(u32, models.block_size);
    defer allocator.free(block_tokens);

    var verify_args = try compiled.target_verify_exe.args(allocator);
    defer verify_args.deinit(allocator);
    var verify_results = try compiled.target_verify_exe.results(allocator);
    defer verify_results.deinit(allocator);

    var start: u32 = prompt.prompt_len;
    const last_prefill_chunk_len: u32 = @min(prompt.prompt_len, default_prefill_seq_len);
    var draft_cache_base: u32 = prompt.prompt_len - last_prefill_chunk_len;
    var first_draft_step = true;
    var target_hidden_block_buffer = prefill_output.target_hidden_buffer;
    var owns_target_hidden_block_buffer = true;
    const started_at: std.Io.Timestamp = .now(project.io, .awake);
    var stopped_on_eos = false;

    while (start < prompt.max_seq_len and !stopped_on_eos) {
        const context_len = start - draft_cache_base;
        const draft_exe = if (first_draft_step) &compiled.prefill_draft_token_exe else &compiled.draft_token_exe;

        @memset(block_tokens, project.parsed_config.value.dflash_config.mask_token_id.?);
        block_tokens[0] = generated.items[@intCast(start)];

        var block_tokens_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.block_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
            .replicated,
        );
        defer block_tokens_buffer.deinit();

        var draft_cache_index = draft_cache_base;
        var draft_cache_index_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.token_index_tensor.shape(), std.mem.asBytes(&draft_cache_index)),
            .replicated,
        );
        defer draft_cache_index_buffer.deinit();

        var draft_active_context_len: u32 = context_len;
        var draft_active_context_len_buffer: zml.Buffer = try .fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.active_context_len_tensor.shape(), std.mem.asBytes(&draft_active_context_len)),
            .replicated,
        );
        defer draft_active_context_len_buffer.deinit();

        draft_exe.args.set(.{
            loaded_buffers.model_buffers,
            loaded_buffers.target_buffers,
            target_hidden_block_buffer,
            block_tokens_buffer,
            draft_kv_cache_buffers,
            draft_cache_index_buffer,
            draft_active_context_len_buffer,
            state.rng_buffer,
        });
        draft_exe.exe.call(draft_exe.args, &draft_exe.results);

        var draft_token_buffer, var draft_logits_buffer, var updated_draft_kv_cache_buffers, var updated_rng_buffer = draft_exe.results.get(struct {
            zml.Buffer,
            zml.Buffer,
            dflash.KvCache.Buffer,
            zml.Tensor.Rng.Buffer,
        });
        defer draft_token_buffer.deinit();
        defer draft_logits_buffer.deinit();
        dflash.KvCache.replaceBuffers(&draft_kv_cache_buffers, &updated_draft_kv_cache_buffers);
        replaceRngBuffer(&state.rng_buffer, &updated_rng_buffer);
        draft_cache_base = start;
        first_draft_step = false;

        var draft_token_slice = try draft_token_buffer.toSliceAlloc(allocator, project.io);
        defer draft_token_slice.free(allocator);
        const draft_tokens = draft_token_slice.constItems(u32);
        for (block_tokens[1..], draft_tokens[1..models.block_size]) |*dst, src| dst.* = src;
        var drafted_block_tokens_buffer = try zml.Buffer.fromSlice(
            project.io,
            project.platform,
            zml.Slice.init(models.block_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
            .replicated,
        );
        replaceBuffer(&block_tokens_buffer, &drafted_block_tokens_buffer);

        var verification = try runVerification(
            allocator,
            project,
            models,
            compiled,
            loaded_buffers,
            &verify_args,
            &verify_results,
            block_tokens_buffer,
            draft_logits_buffer,
            start,
            &state.target_kv_cache_buffers,
            &state.rng_buffer,
            state.attention_metadata_buffers,
        );
        defer verification.deinitTransient(allocator);

        const valid_draft_tokens: usize = @intCast(verification.valid_draft_tokens_slice.constItems(u32)[0]);
        const correction_token = verification.correction_token_slice.constItems(u32)[0];
        const remaining_output_slots: usize = @intCast(prompt.max_seq_len - start);
        const max_usable_draft_tokens = if (remaining_output_slots == 0) 0 else remaining_output_slots - 1;
        const usable_draft_tokens = @min(valid_draft_tokens, max_usable_draft_tokens);
        valid_draft_token_histogram[valid_draft_tokens] += 1;
        try acceptance_lengths.append(allocator, @intCast(valid_draft_tokens));
        try committed_lengths.append(allocator, @intCast(usable_draft_tokens + 1));

        const committed_tokens: u32 = @intCast(usable_draft_tokens + 1);
        for (block_tokens[0..@as(usize, @intCast(committed_tokens))]) |token| {
            if (llama.isEosToken(&project.parsed_target_config.value, token)) {
                stopped_on_eos = true;
                break;
            }
            try setGeneratedToken(&generated, allocator, start, token);
            try output_tokens.append(allocator, token);
            start += 1;
            if (start >= prompt.max_seq_len) break;
        }
        if (!stopped_on_eos and start < prompt.max_seq_len) {
            if (llama.isEosToken(&project.parsed_target_config.value, correction_token)) {
                stopped_on_eos = true;
            } else {
                try setGeneratedToken(&generated, allocator, start, correction_token);
            }
        }

        if (owns_target_hidden_block_buffer) target_hidden_block_buffer.deinit();
        target_hidden_block_buffer = verification.verified_hidden_buffer;
        owns_target_hidden_block_buffer = true;
    }
    if (owns_target_hidden_block_buffer) target_hidden_block_buffer.deinit();

    generated.deinit(allocator);
    return .{
        .generated = output_tokens,
        .acceptance_lengths = acceptance_lengths,
        .committed_lengths = committed_lengths,
        .valid_draft_token_histogram = valid_draft_token_histogram,
        .decode_elapsed = .{ .nanoseconds = started_at.untilNow(project.io, .awake).nanoseconds },
        .stopped_on_eos = stopped_on_eos,
    };
}

const VerificationOutput = struct {
    verified_hidden_buffer: zml.Buffer,
    valid_draft_tokens_buffer: zml.Buffer,
    correction_token_buffer: zml.Buffer,
    valid_draft_tokens_slice: zml.Slice,
    correction_token_slice: zml.Slice,

    pub fn deinitTransient(self: *VerificationOutput, allocator: std.mem.Allocator) void {
        self.correction_token_slice.free(allocator);
        self.valid_draft_tokens_slice.free(allocator);
        self.correction_token_buffer.deinit();
        self.valid_draft_tokens_buffer.deinit();
    }
};

fn runVerification(
    allocator: std.mem.Allocator,
    project: *Project,
    models: *ModelsAndCaches,
    compiled: *CompiledExes,
    loaded_buffers: LoadedBuffers,
    verify_args: *zml.Exe.Arguments,
    verify_results: *zml.Exe.Results,
    block_tokens_buffer: zml.Buffer,
    draft_logits_buffer: zml.Buffer,
    token_index: u32,
    target_kv_cache_buffers: *llama.KvCache.Buffer,
    rng_buffer: *zml.Tensor.Rng.Buffer,
    attention_metadata_buffers: anytype,
) !VerificationOutput {
    var verify_token_index = token_index;
    var verify_token_index_buffer: zml.Buffer = try .fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(models.token_index_tensor.shape(), std.mem.asBytes(&verify_token_index)),
        .replicated,
    );
    defer verify_token_index_buffer.deinit();

    verify_args.set(.{
        loaded_buffers.target_buffers,
        block_tokens_buffer,
        draft_logits_buffer,
        verify_token_index_buffer,
        target_kv_cache_buffers.*,
        rng_buffer.*,
        attention_metadata_buffers,
    });
    compiled.target_verify_exe.call(verify_args.*, verify_results);

    const verified_hidden_buffer, const valid_draft_tokens_buffer, const correction_token_buffer, var verified_target_kv_cache_buffers, var updated_rng_buffer = verify_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        zml.Buffer,
        llama.KvCache.Buffer,
        zml.Tensor.Rng.Buffer,
    });
    llama.KvCache.replaceBuffers(target_kv_cache_buffers, &verified_target_kv_cache_buffers);
    replaceRngBuffer(rng_buffer, &updated_rng_buffer);

    return .{
        .verified_hidden_buffer = verified_hidden_buffer,
        .valid_draft_tokens_buffer = valid_draft_tokens_buffer,
        .correction_token_buffer = correction_token_buffer,
        .valid_draft_tokens_slice = try valid_draft_tokens_buffer.toSliceAlloc(allocator, project.io),
        .correction_token_slice = try correction_token_buffer.toSliceAlloc(allocator, project.io),
    };
}

const DraftContext = struct {
    len: u32,
};

const DraftTokenExe = struct {
    exe: zml.Exe,
    args: zml.Exe.Arguments,
    results: zml.Exe.Results,

    pub fn deinit(self: *DraftTokenExe, allocator: std.mem.Allocator) void {
        self.results.deinit(allocator);
        self.args.deinit(allocator);
        self.exe.deinit();
    }
};

fn draftTokens(
    context: DraftContext,
    draft_model: dflash.Model,
    target_model: llama.Model,
    target_hidden_block: zml.Tensor,
    block_tokens: zml.Tensor,
    draft_kv_cache: dflash.KvCache,
    cache_index: zml.Tensor,
    active_context_len: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: llama.SamplingConfig,
) struct { zml.Tensor, zml.Tensor, dflash.KvCache, zml.Tensor.Rng } {
    const target_hidden_slice = target_hidden_block.withPartialTags(.{ .s, .d }).slice1d(.s, .{
        .start = 0,
        .end = context.len,
    });
    const active_mask = zml.Tensor.iota(.init(.{ .s = context.len }, .u32), .s).convert(.u32)
        .cmp(.LT, active_context_len.convert(.u32).broad(.init(.{ .s = context.len }, .u32)));
    const target_hidden = active_mask.broad(target_hidden_slice.shape())
        .select(target_hidden_slice, zml.Tensor.constant(target_hidden_slice.dtype().zero()).broad(target_hidden_slice.shape()));
    const noise_embedding = target_model.embedForward(block_tokens);
    const context_position_ids = zml.Tensor.arange(.{ .end = context.len }, cache_index.dtype())
        .withTags(.{.s})
        .add(cache_index.broad(.init(.{ .s = context.len }, cache_index.dtype())));
    const proposal_position_ids = zml.Tensor.arange(.{ .end = block_tokens.dim(.s) }, cache_index.dtype())
        .withTags(.{.s})
        .add(active_context_len.convert(cache_index.dtype()).broad(.init(.{ .s = block_tokens.dim(.s) }, cache_index.dtype())))
        .add(cache_index.broad(.init(.{ .s = block_tokens.dim(.s) }, cache_index.dtype())));
    const position_ids = zml.Tensor.concatenate(&.{ context_position_ids, proposal_position_ids }, .s);
    const hidden, const updated_kv_cache = draft_model.forward(target_hidden, noise_embedding, position_ids, draft_kv_cache, cache_index, active_context_len);
    const draft_logits = target_model.logitsForward(hidden);
    const topk: u32 = if (sampling.temperature < 0.00001) 1 else @intCast(draft_logits.dim(.voc));
    const sampled_tokens, const updated_rng = zml.nn.sampleTokens(draft_logits, .{ .topk = topk, .temperature = sampling.temperature }, rng);
    return .{ sampled_tokens, draft_logits, updated_kv_cache, updated_rng };
}

fn targetDecode(
    target_model: llama.Model,
    target_layers: llama_inference.TargetLayers,
    tokens: zml.Tensor,
    token_index: zml.Tensor,
    target_kv_cache: llama.KvCache,
    rng: zml.Tensor.Rng,
    sampling: llama.SamplingConfig,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) struct { zml.Tensor, llama.KvCache, zml.Tensor.Rng } {
    const unused_target_hidden, const target_token, const updated_kv_cache, const updated_rng = target_model.prefillForward(
        tokens,
        token_index,
        target_kv_cache,
        attention_metadata,
        attention_parameters,
        target_layers.slice(),
        sampling,
        rng,
    );
    _ = unused_target_hidden;
    return .{ target_token, updated_kv_cache, updated_rng };
}

fn paddedPromptTokens(allocator: std.mem.Allocator, token_ids: []const u32, block_size: u32) ![]u32 {
    const padded = try allocator.alloc(u32, block_size);
    @memset(padded, 0);
    @memcpy(padded[0..token_ids.len], token_ids);
    return padded;
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

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn replaceRngBuffer(dst: *zml.Tensor.Rng.Buffer, src: *zml.Tensor.Rng.Buffer) void {
    replaceBuffer(&dst._state, &src._state);
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}

fn deinitSampleResult(allocator: std.mem.Allocator, sample: *stats.SampleResult) void {
    allocator.free(sample.id);
    allocator.free(sample.dataset);
    allocator.free(sample.baseline.token_ids);
    allocator.free(sample.baseline.generated_text);
    allocator.free(sample.dflash.token_ids);
    allocator.free(sample.dflash.generated_text);
    allocator.free(sample.dflash.acceptance_lengths);
    allocator.free(sample.dflash.committed_lengths);
    allocator.free(sample.dflash.acceptance_length_histogram);
}
