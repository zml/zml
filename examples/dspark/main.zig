const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("common.zig");
const dspark = @import("dspark_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};
const log = std.log.scoped(.dspark);

const default_prompt = "The capital of France is";
const qwen3_tokenizer_repo = "hf://Qwen/Qwen3-4B";

const Args = struct {
    model: []const u8,
    prompt: []const u8 = default_prompt,
    seqlen: u32 = 0,
    temperature: f32 = 0.0,

    pub const help =
        \\Options:
        \\  --model=<path>          Path to the Qwen3 DSpark model repository
        \\  --prompt=<text>         Prompt to tokenize; defaults to a short smoke-test prompt
        \\  --seqlen=<n>            Number of draft positions; defaults to config.block_size
        \\  --temperature=<t>       Sampling temperature; 0 uses greedy decoding
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, Args);

    var project = try initProject(init, args.model);
    defer project.deinit();

    const model = try dspark.Model.init(allocator, project.store.view(), project.parsed_config.value);
    defer model.deinit(allocator);

    const token_ids = try tokenizePrompt(allocator, &project.tokenizer, args.prompt);
    defer allocator.free(token_ids);
    if (token_ids.len == 0) return error.EmptyPrompt;

    const seqlen = if (args.seqlen == 0) project.parsed_config.value.block_size else args.seqlen;
    const block_tokens = try paddedBlockTokens(allocator, token_ids, seqlen, model.mask_token_id);
    defer allocator.free(block_tokens);

    try runStandalone(allocator, project, model, block_tokens, seqlen, args.temperature);
}

fn runStandalone(
    allocator: std.mem.Allocator,
    project: *Project,
    model: dspark.Model,
    block_tokens: []const u32,
    seqlen: u32,
    temperature: f32,
) !void {
    const target_hidden_len = seqlen;
    const hidden_size: usize = @intCast(project.parsed_config.value.hidden_size);
    const target_hidden_width: i64 = @intCast(hidden_size * model.target_layer_ids.len);
    const target_hidden_tensor: zml.Tensor = .init(.{ .s = target_hidden_len, .d = target_hidden_width }, model.fc.weight.dtype());
    const block_tokens_tensor: zml.Tensor = .init(.{ .s = seqlen }, .u32);
    const cache_index_tensor: zml.Tensor = .init(.{}, .u32);
    const active_context_len_tensor: zml.Tensor = .init(.{}, .u32);
    const kv_cache = model.initKvCache(project.parsed_config.value, target_hidden_len + seqlen);
    const rng: zml.Tensor.Rng = .init();
    const sampling: dspark.SamplingConfig = .{ .temperature = temperature };

    const all_shardings = project.shardings.all();
    log.info("Compiling Qwen3 DSpark standalone block...", .{});
    var exe = try project.platform.compileFn(
        allocator,
        project.io,
        runDraftBlock,
        .{
            model,
            target_hidden_tensor,
            block_tokens_tensor,
            kv_cache,
            cache_index_tensor,
            active_context_len_tensor,
            rng,
            sampling,
        },
        .{ .shardings = &all_shardings },
    );
    defer exe.deinit();

    log.info("Loading Qwen3 DSpark weights...", .{});
    var model_buffers = try model.loadBuffers(allocator, project.io, project.platform, &project.store, &all_shardings);
    defer dspark.Model.unloadBuffers(&model_buffers, allocator);

    var target_hidden_buffer = try common.zeroBuffer(
        allocator,
        project.io,
        project.platform,
        target_hidden_tensor.shape(),
        project.shardings.model,
    );
    defer target_hidden_buffer.deinit();

    var block_tokens_buffer = try zml.Buffer.fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(block_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
        .replicated,
    );
    defer block_tokens_buffer.deinit();

    var kv_cache_buffers = try kv_cache.initZeroBuffer(allocator, project.io, project.platform, project.shardings.model);
    defer dspark.KvCache.deinitBuffer(&kv_cache_buffers);

    var cache_index: u32 = 0;
    var cache_index_buffer = try zml.Buffer.fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(cache_index_tensor.shape(), std.mem.asBytes(&cache_index)),
        .replicated,
    );
    defer cache_index_buffer.deinit();

    var active_context_len: u32 = target_hidden_len;
    var active_context_len_buffer = try zml.Buffer.fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(active_context_len_tensor.shape(), std.mem.asBytes(&active_context_len)),
        .replicated,
    );
    defer active_context_len_buffer.deinit();

    var rng_buffer = try zml.Tensor.Rng.initBuffer(project.io, project.platform, .replicated, 0);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{
        model_buffers,
        target_hidden_buffer,
        block_tokens_buffer,
        kv_cache_buffers,
        cache_index_buffer,
        active_context_len_buffer,
        rng_buffer,
    });
    exe.call(exe_args, &exe_results);

    var sampled_tokens_buffer, var updated_rng = exe_results.get(struct {
        zml.Buffer,
        zml.Tensor.Rng.Buffer,
    });
    defer sampled_tokens_buffer.deinit();
    defer zml.Tensor.Rng.deinitBuffer(&updated_rng);

    try printSampledTokens(allocator, project, block_tokens, sampled_tokens_buffer);
}

fn runDraftBlock(
    model: dspark.Model,
    target_hidden: zml.Tensor,
    block_tokens: zml.Tensor,
    kv_cache: dspark.KvCache,
    cache_index: zml.Tensor,
    active_context_len: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: dspark.SamplingConfig,
) struct { zml.Tensor, zml.Tensor.Rng } {
    const target = target_hidden.withPartialTags(.{ .s, .d });
    const tokens = block_tokens.withPartialTags(.{.s});
    const positions = zml.Tensor.arange(.{ .end = target.dim(.s) + tokens.dim(.s) }, .i64).withTags(.{.s});
    const hidden, const updated_kv_cache = model.forward(target, tokens, positions, kv_cache, cache_index, active_context_len);
    _ = updated_kv_cache;
    const sampled, const draft_logits, const updated_rng = model.draftBlock(hidden, tokens, sampling, rng);
    _ = draft_logits;
    return .{ sampled.convert(.u32), updated_rng };
}

fn printSampledTokens(
    allocator: std.mem.Allocator,
    project: *Project,
    block_tokens: []const u32,
    sampled_tokens_buffer: zml.Buffer,
) !void {
    var sampled_tokens_slice = try sampled_tokens_buffer.toSliceAlloc(allocator, project.io);
    defer sampled_tokens_slice.free(allocator);
    const sampled_tokens = sampled_tokens_slice.constItems(u32);

    var decoded = try decodeTokens(allocator, &project.tokenizer, sampled_tokens);
    defer decoded.deinit(allocator);

    var stdout = std.Io.File.stdout().writerStreaming(project.io, &.{});
    try stdout.interface.print("input token ids: {any}\n", .{block_tokens});
    try stdout.interface.print("draft token ids: {any}\n", .{sampled_tokens});
    try stdout.interface.print("draft text: {s}\n", .{decoded.items});
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
    parsed_config: std.json.Parsed(dspark.Config),
    registry: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,
    tokenizer: zml.tokenizer.Tokenizer,
    shardings: common.Shardings,

    pub fn deinit(self: *Project) void {
        const allocator = self.allocator;
        self.tokenizer.deinit();
        self.store.deinit();
        self.registry.deinit();
        self.parsed_config.deinit();
        self.platform.deinit(allocator, self.io);
        self.vfs.deinit();
        self.hf_vfs.deinit();
        self.http_client.deinit();
        self.vfs_file.deinit();
        allocator.destroy(self);
    }
};

fn initProject(init: std.process.Init, model_path: []const u8) !*Project {
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

    project.repo = try zml.safetensors.resolveModelRepo(project.io, model_path);
    project.parsed_config = try common.parseConfig(dspark.Config, allocator, project.io, project.repo);
    errdefer project.parsed_config.deinit();

    project.registry = try .fromRepo(allocator, project.io, project.repo);
    errdefer project.registry.deinit();

    project.store = .fromRegistry(allocator, &project.registry);
    errdefer project.store.deinit();

    project.tokenizer = try loadTokenizer(allocator, project.io, project.repo, project.parsed_config.value);
    errdefer project.tokenizer.deinit();

    project.shardings = try .init(project.platform);
    return project;
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, config: dspark.Config) !zml.tokenizer.Tokenizer {
    return loadTokenizerFromDir(allocator, io, dir) catch |err| switch (err) {
        error.FileNotFound => {
            if (!config.isQwen3()) return err;
            log.info("tokenizer.json not found in draft repo; falling back to {s}", .{qwen3_tokenizer_repo});
            const tokenizer_repo = try zml.safetensors.resolveModelRepo(io, qwen3_tokenizer_repo);
            return loadTokenizerFromDir(allocator, io, tokenizer_repo);
        },
        else => return err,
    };
}

fn loadTokenizerFromDir(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    return try .fromBytes(allocator, bytes);
}

fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: *const zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    return encoder.encodeAlloc(allocator, prompt);
}

fn paddedBlockTokens(allocator: std.mem.Allocator, token_ids: []const u32, seqlen: u32, pad_token_id: u32) ![]u32 {
    const padded = try allocator.alloc(u32, seqlen);
    @memset(padded, pad_token_id);
    const copy_len = @min(token_ids.len, seqlen);
    @memcpy(padded[0..copy_len], token_ids[0..copy_len]);
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
