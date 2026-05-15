const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dflash = @import("dflash_model.zig");
const llama_inference = @import("llama/llama_inference.zig");
const llama = @import("llama/llama_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};
const log = std.log.scoped(.dflash);

const prefill_seq_len = 64;
const default_prompt =
    "Tell me a story about Paris, the city of lights, fun, and romance.";

const Args = struct {
    model: []const u8,
    target_model: []const u8,
    prompt: []const u8 = default_prompt,
    max_seq_len: u32 = 256,

    pub const help =
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
    const args = stdx.flags.parse(init.minimal.args, Args);

    var project = try initProject(init, args);
    defer project.deinit();

    var prompt = try tokenizePrompt(project, args);
    defer prompt.deinit(allocator);

    var models = try initModelsAndCaches(project, prompt);
    defer models.deinit(allocator);

    var compiled = try compile(allocator, project, models, prompt);
    defer compiled.deinit(allocator);

    var loaded_buffers = try loadBuffers(allocator, project, models);
    defer loaded_buffers.deinit(allocator);

    var input_tokens_buffer: zml.Buffer = try .fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(models.input_tokens_tensor.shape(), std.mem.sliceAsBytes(prompt.padded_token_ids)),
        .replicated,
    );
    defer input_tokens_buffer.deinit();

    var token_index: u32 = 0;
    var token_index_buffer: zml.Buffer = try .fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(models.token_index_tensor.shape(), std.mem.asBytes(&token_index)),
        .replicated,
    );
    defer token_index_buffer.deinit();

    var target_kv_cache_buffers = try models.target_kv_cache.initZeroBuffer(allocator, project.io, project.platform, project.shardings.model);
    defer llama.KvCache.deinitBuffer(&target_kv_cache_buffers);

    var draft_kv_cache_buffers = try models.draft_kv_cache.initZeroBuffer(allocator, project.io, project.platform, project.shardings.model);
    defer dflash.KvCache.deinitBuffer(&draft_kv_cache_buffers);

    var attention_metadata_buffers = try models.target_attention.metadata.initBuffer(project.io, project.platform, project.shardings.model);
    defer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

    var prefill_output = try runPrefill(
        allocator,
        project,
        &compiled,
        loaded_buffers,
        input_tokens_buffer,
        token_index_buffer,
        &target_kv_cache_buffers,
        attention_metadata_buffers,
    );
    defer prefill_output.target_hidden_buffer.deinit();

    var stdout = std.Io.File.stdout().writerStreaming(project.io, &.{});
    var decode_result = try runSpeculativeDecoding(
        allocator,
        project,
        prompt,
        &models,
        &compiled,
        loaded_buffers,
        prefill_output,
        &target_kv_cache_buffers,
        &draft_kv_cache_buffers,
        attention_metadata_buffers,
        &stdout.interface,
    );
    defer decode_result.deinit(allocator);

    try printDecodeAnalysis(&stdout.interface, decode_result);
    try stdout.interface.flush();
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
    padded_token_ids: []u32,
    prompt_len: u32,
    max_seq_len: u32,

    pub fn deinit(self: TokenizedPrompt, allocator: std.mem.Allocator) void {
        allocator.free(self.padded_token_ids);
        allocator.free(self.token_ids);
    }
};

fn tokenizePrompt(project: *Project, args: Args) !TokenizedPrompt {
    const token_ids = try llama.tokenizePrompt(project.allocator, &project.tokenizer, project.parsed_target_config.value, args.prompt);
    errdefer project.allocator.free(token_ids);

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

    const padded_token_ids = try paddedPromptTokens(project.allocator, token_ids, prefill_seq_len);
    return .{
        .token_ids = token_ids,
        .padded_token_ids = padded_token_ids,
        .prompt_len = @intCast(token_ids.len),
        .max_seq_len = args.max_seq_len,
    };
}

const ModelsAndCaches = struct {
    draft_model: dflash.Model,
    target_model: llama.Model,
    input_tokens_tensor: zml.Tensor,
    block_tokens_tensor: zml.Tensor,
    token_index_tensor: zml.Tensor,
    active_context_len_tensor: zml.Tensor,
    target_kv_cache: llama.KvCache,
    draft_kv_cache: dflash.KvCache,
    target_attention: llama_inference.TargetAttention,
    target_layers: llama_inference.TargetLayers,
    block_size: u32,

    pub fn deinit(self: ModelsAndCaches, allocator: std.mem.Allocator) void {
        self.target_model.deinit(allocator);
        self.draft_model.deinit(allocator);
    }
};

fn initModelsAndCaches(project: *Project, prompt: TokenizedPrompt) !ModelsAndCaches {
    const draft_model = try dflash.Model.init(project.allocator, project.store.view(), project.parsed_config.value);
    errdefer draft_model.deinit(project.allocator);

    const target_model = try llama.Model.init(project.allocator, project.target_store.view(), project.parsed_target_config.value);
    errdefer target_model.deinit(project.allocator);

    const block_size = project.parsed_config.value.block_size;
    const cache_seq_len = @max(prompt.max_seq_len, prefill_seq_len) + block_size;
    const target_attention = llama_inference.TargetAttention.init(project.platform, cache_seq_len, project.parsed_target_config.value.num_attention_heads);
    log.info("Selected target attention backend: {}", .{target_attention.backend});

    return .{
        .draft_model = draft_model,
        .target_model = target_model,
        .input_tokens_tensor = .init(.{ .s = prefill_seq_len }, .u32),
        .block_tokens_tensor = .init(.{ .s = block_size }, .u32),
        .token_index_tensor = .init(.{}, .u32),
        .active_context_len_tensor = .init(.{}, .u32),
        .target_kv_cache = target_model.initKvCache(project.parsed_target_config.value, cache_seq_len),
        .draft_kv_cache = draft_model.initKvCache(project.parsed_config.value, cache_seq_len),
        .target_attention = target_attention,
        .target_layers = llama_inference.TargetLayers.init(draft_model.target_layer_ids),
        .block_size = block_size,
    };
}

const CompiledExes = struct {
    target_prefill_exe: zml.Exe,
    draft_token_exes: []DraftTokenExe,
    target_verify_exe: zml.Exe,

    pub fn deinit(self: *CompiledExes, allocator: std.mem.Allocator) void {
        self.target_verify_exe.deinit();
        for (self.draft_token_exes) |*draft_exe| draft_exe.deinit(allocator);
        allocator.free(self.draft_token_exes);
        self.target_prefill_exe.deinit();
    }
};

fn compile(allocator: std.mem.Allocator, project: *Project, models: ModelsAndCaches, prompt: TokenizedPrompt) !CompiledExes {
    const all_shardings = project.shardings.all();

    log.info("Compiling target prefill...", .{});
    const target_prefill_exe = try llama_inference.compileTargetPrefill(
        allocator,
        project.io,
        project.platform,
        models.target_model,
        models.target_layers,
        prompt.prompt_len,
        models.input_tokens_tensor,
        models.token_index_tensor,
        models.target_kv_cache,
        models.target_attention,
        &all_shardings,
    );

    log.info("Compiling DFlash draft token variants...", .{});
    const draft_context_lens = [_]u32{ 1, 2, 3, 5, 10, prefill_seq_len };
    const draft_token_exes = try allocator.alloc(DraftTokenExe, draft_context_lens.len);
    const target_hidden_tensor = llama_inference.targetHiddenTensor(models.target_model, project.parsed_target_config.value, models.target_layers, prefill_seq_len);
    for (draft_token_exes, draft_context_lens) |*draft_exe, context_len| {
        draft_exe.context_len = context_len;
        draft_exe.exe = try project.platform.compileFn(
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
            },
            .{ .shardings = &all_shardings },
        );
        draft_exe.args = try draft_exe.exe.args(allocator);
        draft_exe.results = try draft_exe.exe.results(allocator);
    }

    log.info("Compiling target verify...", .{});
    const target_verify_exe = try llama_inference.compileTargetVerify(
        allocator,
        project.io,
        project.platform,
        models.target_model,
        models.target_layers,
        prefill_seq_len,
        models.block_tokens_tensor,
        models.token_index_tensor,
        models.target_kv_cache,
        models.target_attention,
        &all_shardings,
    );

    return .{
        .target_prefill_exe = target_prefill_exe,
        .draft_token_exes = draft_token_exes,
        .target_verify_exe = target_verify_exe,
    };
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
    return .{
        .model_buffers = try models.draft_model.loadBuffers(allocator, project.io, project.platform, &project.store, &all_shardings),
        .target_buffers = try models.target_model.loadBuffers(allocator, project.io, project.platform, &project.target_store, &all_shardings),
    };
}

const PrefillOutput = struct {
    target_hidden_buffer: zml.Buffer,
    target_token: u32,
};

fn runPrefill(
    allocator: std.mem.Allocator,
    project: *Project,
    compiled: *CompiledExes,
    loaded_buffers: LoadedBuffers,
    input_tokens_buffer: zml.Buffer,
    token_index_buffer: zml.Buffer,
    target_kv_cache_buffers: *llama.KvCache.Buffer,
    attention_metadata_buffers: anytype,
) !PrefillOutput {
    var prefill_args = try compiled.target_prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);

    var prefill_results = try compiled.target_prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    prefill_args.set(.{
        loaded_buffers.target_buffers,
        input_tokens_buffer,
        token_index_buffer,
        target_kv_cache_buffers.*,
        attention_metadata_buffers,
    });
    compiled.target_prefill_exe.call(prefill_args, &prefill_results);

    const target_hidden_buffer, var target_token_buffer, var prefilled_target_kv_cache_buffers = prefill_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        llama.KvCache.Buffer,
    });
    defer target_token_buffer.deinit();
    llama.KvCache.replaceBuffers(target_kv_cache_buffers, &prefilled_target_kv_cache_buffers);

    var target_token_slice = try target_token_buffer.toSliceAlloc(allocator, project.io);
    defer target_token_slice.free(allocator);

    return .{
        .target_hidden_buffer = target_hidden_buffer,
        .target_token = target_token_slice.constItems(u32)[0],
    };
}

const VerificationOutput = struct {
    verified_hidden_buffer: zml.Buffer,
    posterior_token_buffer: zml.Buffer,
    posterior_token_slice: zml.Slice,

    pub fn deinitTransient(self: *VerificationOutput, allocator: std.mem.Allocator) void {
        self.posterior_token_slice.free(allocator);
        self.posterior_token_buffer.deinit();
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
    token_index: u32,
    target_kv_cache_buffers: *llama.KvCache.Buffer,
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
        verify_token_index_buffer,
        target_kv_cache_buffers.*,
        attention_metadata_buffers,
    });
    compiled.target_verify_exe.call(verify_args.*, verify_results);

    const verified_hidden_buffer, const posterior_token_buffer, var verified_target_kv_cache_buffers = verify_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        llama.KvCache.Buffer,
    });
    llama.KvCache.replaceBuffers(target_kv_cache_buffers, &verified_target_kv_cache_buffers);

    return .{
        .verified_hidden_buffer = verified_hidden_buffer,
        .posterior_token_buffer = posterior_token_buffer,
        .posterior_token_slice = try posterior_token_buffer.toSliceAlloc(allocator, project.io),
    };
}

const DecodeResult = struct {
    generated: std.ArrayList(u32),
    decode_elapsed: std.Io.Duration,
    decoded_tokens: u32,
    step: usize,
    target_cache_logical_len: u32,
    draft_cache_base: u32,
    stopped_on_eos: bool,
    total_valid_draft_tokens: usize,
    total_draft_tokens_checked: usize,
    min_valid_draft_tokens: usize,
    max_valid_draft_tokens: usize,
    full_accept_steps: usize,
    zero_accept_steps: usize,
    valid_draft_token_histogram: []usize,

    pub fn deinit(self: *DecodeResult, allocator: std.mem.Allocator) void {
        allocator.free(self.valid_draft_token_histogram);
        self.generated.deinit(allocator);
    }
};

fn runSpeculativeDecoding(
    allocator: std.mem.Allocator,
    project: *Project,
    prompt: TokenizedPrompt,
    models: *ModelsAndCaches,
    compiled: *CompiledExes,
    loaded_buffers: LoadedBuffers,
    prefill_output: PrefillOutput,
    target_kv_cache_buffers: *llama.KvCache.Buffer,
    draft_kv_cache_buffers: *dflash.KvCache.Buffer,
    attention_metadata_buffers: anytype,
    stdout: *std.Io.Writer,
) !DecodeResult {
    var generated = try std.ArrayList(u32).initCapacity(allocator, prompt.max_seq_len + 1);
    errdefer generated.deinit(allocator);
    try generated.appendSlice(allocator, prompt.token_ids);
    try generated.append(allocator, prefill_output.target_token);

    var prompt_text = try decodeTokens(allocator, &project.tokenizer, prompt.token_ids);
    defer prompt_text.deinit(allocator);
    try stdout.writeAll(prompt_text.items);
    try stdout.flush();

    var stream_decoder = try project.tokenizer.decoder();
    defer stream_decoder.deinit();
    const decoder_out_buffer = try allocator.alloc(u8, 256);
    defer allocator.free(decoder_out_buffer);

    var block_tokens = try allocator.alloc(u32, models.block_size);
    defer allocator.free(block_tokens);

    var verify_args = try compiled.target_verify_exe.args(allocator);
    defer verify_args.deinit(allocator);
    var verify_results = try compiled.target_verify_exe.results(allocator);
    defer verify_results.deinit(allocator);

    var start: u32 = prompt.prompt_len;
    var draft_cache_base: u32 = 0;
    var target_hidden_block_buffer = prefill_output.target_hidden_buffer;
    var owns_target_hidden_block_buffer = false;
    var step: usize = 0;
    var total_valid_draft_tokens: usize = 0;
    var total_draft_tokens_checked: usize = 0;
    var full_accept_steps: usize = 0;
    var zero_accept_steps: usize = 0;
    var min_valid_draft_tokens: usize = models.block_size - 1;
    var max_valid_draft_tokens: usize = 0;
    const valid_draft_token_histogram = try allocator.alloc(usize, models.block_size);
    errdefer allocator.free(valid_draft_token_histogram);
    @memset(valid_draft_token_histogram, 0);
    const decode_started_at: std.Io.Timestamp = .now(project.io, .awake);
    var stopped_on_eos = false;

    while (start < prompt.max_seq_len and !stopped_on_eos) : (step += 1) {
        const context_len = start - draft_cache_base;
        stdx.debug.assert(context_len >= 1 and context_len <= prefill_seq_len, "invalid DFlash context length {}", .{context_len});
        const selected_draft_exe = findDraftTokenExe(compiled.draft_token_exes, context_len);

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

        selected_draft_exe.args.set(.{
            loaded_buffers.model_buffers,
            loaded_buffers.target_buffers,
            target_hidden_block_buffer,
            block_tokens_buffer,
            draft_kv_cache_buffers.*,
            draft_cache_index_buffer,
            draft_active_context_len_buffer,
        });
        selected_draft_exe.exe.call(selected_draft_exe.args, &selected_draft_exe.results);

        var draft_token_buffer, var updated_draft_kv_cache_buffers = selected_draft_exe.results.get(struct {
            zml.Buffer,
            dflash.KvCache.Buffer,
        });
        defer draft_token_buffer.deinit();
        dflash.KvCache.replaceBuffers(draft_kv_cache_buffers, &updated_draft_kv_cache_buffers);
        draft_cache_base = start;

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
            start,
            target_kv_cache_buffers,
            attention_metadata_buffers,
        );
        defer verification.deinitTransient(allocator);
        const posterior_tokens = verification.posterior_token_slice.constItems(u32);

        var valid_draft_tokens: usize = 0;
        while (valid_draft_tokens + 1 < models.block_size and block_tokens[valid_draft_tokens + 1] == posterior_tokens[valid_draft_tokens]) {
            valid_draft_tokens += 1;
        }
        total_valid_draft_tokens += valid_draft_tokens;
        total_draft_tokens_checked += models.block_size - 1;
        min_valid_draft_tokens = @min(min_valid_draft_tokens, valid_draft_tokens);
        max_valid_draft_tokens = @max(max_valid_draft_tokens, valid_draft_tokens);
        valid_draft_token_histogram[valid_draft_tokens] += 1;
        if (valid_draft_tokens == models.block_size - 1) full_accept_steps += 1;
        if (valid_draft_tokens == 0) zero_accept_steps += 1;

        const correction_token = posterior_tokens[valid_draft_tokens];
        const committed_tokens: u32 = @intCast(valid_draft_tokens + 1);
        const generated_step_start = start;
        for (block_tokens[0..@as(usize, @intCast(committed_tokens))]) |token| {
            if (llama.isEosToken(&project.parsed_target_config.value, token)) {
                stopped_on_eos = true;
                break;
            }
            try setGeneratedToken(&generated, allocator, start, token);
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

        const generated_step_end = @min(start, prompt.max_seq_len);
        for (generated.items[generated_step_start..generated_step_end]) |token| {
            try stdout.writeAll(try stream_decoder.feedOne(token, decoder_out_buffer));
        }
        try stdout.flush();
    }
    if (owns_target_hidden_block_buffer) target_hidden_block_buffer.deinit();

    try stdout.writeAll(try stream_decoder.finalize(decoder_out_buffer));
    try stdout.writeByte('\n');

    if (step == 0) min_valid_draft_tokens = 0;
    return .{
        .generated = generated,
        .decode_elapsed = decode_started_at.untilNow(project.io, .awake),
        .decoded_tokens = start - prompt.prompt_len,
        .step = step,
        .target_cache_logical_len = start,
        .draft_cache_base = draft_cache_base,
        .stopped_on_eos = stopped_on_eos,
        .total_valid_draft_tokens = total_valid_draft_tokens,
        .total_draft_tokens_checked = total_draft_tokens_checked,
        .min_valid_draft_tokens = min_valid_draft_tokens,
        .max_valid_draft_tokens = max_valid_draft_tokens,
        .full_accept_steps = full_accept_steps,
        .zero_accept_steps = zero_accept_steps,
        .valid_draft_token_histogram = valid_draft_token_histogram,
    };
}

fn printDecodeAnalysis(stdout: *std.Io.Writer, result: DecodeResult) !void {
    const decode_seconds = @as(f64, @floatFromInt(result.decode_elapsed.nanoseconds)) / std.time.ns_per_s;
    const tokens_per_second = @as(f64, @floatFromInt(result.decoded_tokens)) / decode_seconds;
    const draft_acceptance_rate = if (result.total_draft_tokens_checked == 0)
        0
    else
        @as(f64, @floatFromInt(result.total_valid_draft_tokens)) / @as(f64, @floatFromInt(result.total_draft_tokens_checked));
    const avg_valid_draft_tokens = if (result.step == 0)
        0
    else
        @as(f64, @floatFromInt(result.total_valid_draft_tokens)) / @as(f64, @floatFromInt(result.step));

    try stdout.print(
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
            result.decode_elapsed,
            result.decoded_tokens,
            tokens_per_second,
            result.step,
            result.target_cache_logical_len,
            result.draft_cache_base,
            result.stopped_on_eos,
            avg_valid_draft_tokens,
            result.min_valid_draft_tokens,
            result.max_valid_draft_tokens,
            result.total_valid_draft_tokens,
            draft_acceptance_rate,
            result.full_accept_steps,
            result.zero_accept_steps,
        },
    );
    try stdout.print("\n--- valid_draft_tokens_histogram ---\n", .{});
    const histogram_bar_width: usize = 80;
    for (result.valid_draft_token_histogram, 0..) |count, valid_tokens| {
        if (count == 0) {
            try stdout.print("  {d:>2}: {d:>4}\n", .{ valid_tokens, count });
            continue;
        }
        const pct = if (result.step == 0)
            0
        else
            100.0 * @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(result.step));
        const bar_len = @max(@as(usize, 1), count * histogram_bar_width / result.step);
        try stdout.print("  {d:>2}: {d:>4} ({d:>5.1}%)  ", .{ valid_tokens, count, pct });
        try stdout.splatByteAll('x', bar_len);
        try stdout.writeByte('\n');
    }
}

const DraftContext = struct {
    len: u32,
};

const DraftTokenExe = struct {
    context_len: u32,
    exe: zml.Exe,
    args: zml.Exe.Arguments,
    results: zml.Exe.Results,

    pub fn deinit(self: *DraftTokenExe, allocator: std.mem.Allocator) void {
        self.results.deinit(allocator);
        self.args.deinit(allocator);
        self.exe.deinit();
    }
};

fn findDraftTokenExe(draft_token_exes: []DraftTokenExe, context_len: u32) *DraftTokenExe {
    for (draft_token_exes) |*draft_exe| {
        if (context_len <= draft_exe.context_len) return draft_exe;
    }
    std.debug.panic("no DFlash draft executable compiled for context length {}", .{context_len});
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
    return .{ target_model.sampleForward(hidden), updated_kv_cache };
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

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
