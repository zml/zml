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

const Args = struct {
    model: []const u8,
    prompt: []const u8 = default_prompt,
    seqlen: u32 = 8,
    topk: u32 = 1,
    temperature: f32 = 0.0,

    pub const help =
        \\Options:
        \\  --model=<path>          Path to the DeepSeek V4 model repository
        \\  --prompt=<text>         Prompt to tokenize; defaults to a short smoke-test prompt
        \\  --seqlen=<n>            Number of draft positions to run in one block; defaults to 8
        \\  --topk=<n>              Sampling top-k; defaults to 1
        \\  --temperature=<t>       Sampling temperature; 0 uses greedy decoding
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, Args);

    var project = try initProject(init, args.model);
    defer project.deinit();

    const token_ids = try tokenizePrompt(allocator, &project.tokenizer, args.prompt);
    defer allocator.free(token_ids);
    if (token_ids.len == 0) return error.EmptyPrompt;

    const model = try dspark.Model.init(allocator, project.store.view(), project.parsed_config.value);
    defer model.deinit(allocator);

    const block_tokens = try paddedBlockTokens(allocator, token_ids, args.seqlen, model.blockPadToken());
    defer allocator.free(block_tokens);

    switch (model) {
        .deepseek => |deepseek_model| try runDeepSeekStandalone(allocator, project, deepseek_model, block_tokens, args),
        .qwen3 => |qwen3_model| try runQwenStandalone(allocator, project, qwen3_model, block_tokens, args),
    }
}

fn runDeepSeekStandalone(
    allocator: std.mem.Allocator,
    project: *Project,
    model: dspark.DeepSeekModel,
    block_tokens: []const u32,
    args: Args,
) !void {
    const moe_dtype = model.layers[0].ffn.experts[0].w1.weight.dtype();
    const moe_backend = try zml.moe.Backend.auto(project.platform, moe_dtype);
    const moe_parameters = zml.moe.Parameters.init(.fromBackend(moe_backend, project.parsed_config.value.num_experts_per_tok, .silu));
    const moe_metadata = initMoeMetadata(model, args.seqlen, moe_backend);

    const input_tokens_tensor: zml.Tensor = .init(.{ .s = args.seqlen }, .u32);
    const aux_hidden_tensor: zml.Tensor = .init(.{ .s = args.seqlen, .d = model.target_aux_width }, moe_dtype);
    const rng: zml.Tensor.Rng = .init();
    const sampling: dspark.SamplingConfig = .{
        .topk = args.topk,
        .temperature = args.temperature,
    };

    const all_shardings = project.shardings.all();
    log.info("Compiling DSpark standalone block...", .{});
    var exe = try project.platform.compileFn(
        allocator,
        project.io,
        runDraftBlock,
        .{
            model,
            input_tokens_tensor,
            aux_hidden_tensor,
            rng,
            sampling,
            moe_metadata,
            moe_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    defer exe.deinit();

    log.info("Loading DSpark weights...", .{});
    var model_buffers = try model.loadBuffers(allocator, project.io, project.platform, &project.store, &all_shardings);
    defer dspark.DeepSeekModel.unloadBuffers(&model_buffers, allocator);

    var input_tokens_buffer = try zml.Buffer.fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(input_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
        .replicated,
    );
    defer input_tokens_buffer.deinit();

    var aux_hidden_buffer = try common.zeroBuffer(
        allocator,
        project.io,
        project.platform,
        aux_hidden_tensor.shape(),
        project.shardings.model,
    );
    defer aux_hidden_buffer.deinit();

    var rng_buffer = try zml.Tensor.Rng.initBuffer(project.io, project.platform, .replicated, 0);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    var moe_metadata_buffers = try moe_metadata.initBuffer(project.io, project.platform);
    defer zml.moe.Metadata.deinitBuffer(&moe_metadata_buffers);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{
        model_buffers,
        input_tokens_buffer,
        aux_hidden_buffer,
        rng_buffer,
        moe_metadata_buffers,
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

fn runQwenStandalone(
    allocator: std.mem.Allocator,
    project: *Project,
    model: dspark.Qwen3Model,
    block_tokens: []const u32,
    args: Args,
) !void {
    const aux_dtype = model.fc.weight.dtype();
    const input_tokens_tensor: zml.Tensor = .init(.{ .s = args.seqlen }, .u32);
    const aux_hidden_tensor: zml.Tensor = .init(.{ .s = args.seqlen, .d = model.target_aux_width }, aux_dtype);
    const rng: zml.Tensor.Rng = .init();
    const sampling: dspark.SamplingConfig = .{
        .topk = args.topk,
        .temperature = args.temperature,
    };

    const all_shardings = project.shardings.all();
    log.info("Compiling Qwen3 DSpark standalone block...", .{});
    var exe = try project.platform.compileFn(
        allocator,
        project.io,
        runQwenDraftBlock,
        .{
            model,
            input_tokens_tensor,
            aux_hidden_tensor,
            rng,
            sampling,
        },
        .{ .shardings = &all_shardings },
    );
    defer exe.deinit();

    log.info("Loading Qwen3 DSpark weights...", .{});
    var model_buffers = try model.loadBuffers(allocator, project.io, project.platform, &project.store, &all_shardings);
    defer dspark.Qwen3Model.unloadBuffers(&model_buffers, allocator);

    var input_tokens_buffer = try zml.Buffer.fromSlice(
        project.io,
        project.platform,
        zml.Slice.init(input_tokens_tensor.shape(), std.mem.sliceAsBytes(block_tokens)),
        .replicated,
    );
    defer input_tokens_buffer.deinit();

    var aux_hidden_buffer = try common.zeroBuffer(
        allocator,
        project.io,
        project.platform,
        aux_hidden_tensor.shape(),
        project.shardings.model,
    );
    defer aux_hidden_buffer.deinit();

    var rng_buffer = try zml.Tensor.Rng.initBuffer(project.io, project.platform, .replicated, 0);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{
        model_buffers,
        input_tokens_buffer,
        aux_hidden_buffer,
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

    project.tokenizer = try loadTokenizer(allocator, project.io, project.repo);
    errdefer project.tokenizer.deinit();

    project.shardings = try .init(project.platform);
    return project;
}

fn runDraftBlock(
    model: dspark.DeepSeekModel,
    input_tokens: zml.Tensor,
    aux_hidden_states: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: dspark.SamplingConfig,
    moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
) struct { zml.Tensor, zml.Tensor.Rng } {
    const tokens = input_tokens.withPartialTags(.{.s});
    const positions = zml.Tensor.arange(.{ .end = tokens.dim(.s) }, .i64).withTags(.{.s});
    const hidden = model.forward(tokens, positions, aux_hidden_states, moe_metadata, moe_parameters);
    const logits = model.draftLogits(hidden, tokens);
    const sampled, const updated_rng = model.sample(logits, sampling, rng);
    return .{ sampled.convert(.u32), updated_rng };
}

fn runQwenDraftBlock(
    model: dspark.Qwen3Model,
    input_tokens: zml.Tensor,
    aux_hidden_states: zml.Tensor,
    rng: zml.Tensor.Rng,
    sampling: dspark.SamplingConfig,
) struct { zml.Tensor, zml.Tensor.Rng } {
    const tokens = input_tokens.withPartialTags(.{.s});
    const positions = zml.Tensor.arange(.{ .end = tokens.dim(.s) }, .i64).withTags(.{.s});
    const hidden = model.forward(tokens, positions, aux_hidden_states);
    const logits = model.draftLogits(hidden, tokens);
    const sampled, const updated_rng = dspark.Model.sample(.{ .qwen3 = model }, logits, sampling, rng);
    return .{ sampled.convert(.u32), updated_rng };
}

fn initMoeMetadata(model: dspark.DeepSeekModel, token_len: u32, backend: zml.moe.Backend) zml.moe.Metadata {
    _ = token_len;
    const num_experts: i64 = @intCast(model.layers[0].ffn.experts.len);

    const gate_up_shape = zml.Shape.init(.{
        .expert = num_experts,
        .out = model.layers[0].ffn.experts[0].w1.weight.dim(.dout) * 2,
    }, model.layers[0].ffn.experts[0].w1.weight.dtype());
    const down_shape = zml.Shape.init(.{
        .expert = num_experts,
        .out = model.layers[0].ffn.experts[0].w2.weight.dim(.dout),
    }, model.layers[0].ffn.experts[0].w2.weight.dtype());

    return switch (backend) {
        .triton => .init(.{
            .triton = .{
                .w1_zero_bias_shape = gate_up_shape,
                .w2_zero_bias_shape = down_shape,
            },
        }),
        .mosaic_tpu, .metal => .init(.fromBackend(backend)),
    };
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
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
