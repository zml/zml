const std = @import("std");

const models = @import("llm_models");
const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.distributed_llm);

const AttentionBackend = enum {
    vanilla,
    cuda_fa2,
};

const CliArgs = struct {
    pub const help =
        \\Usage:
        \\  distributed_llm --model=<absolute-path> --prompt=<text>
        \\    [--seqlen=128] [--topk=1]
        \\    --backend=<vanilla|cuda_fa2>
        \\    COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    model: []const u8,
    prompt: []const u8,
    seqlen: u32 = 128,
    topk: u32 = 1,
    backend: AttentionBackend,
    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

const BufferStats = struct {
    local_bytes: usize = 0,
    expected_shards: u32,

    fn add(self: *BufferStats, buffer: *const zml.Buffer) !void {
        if (buffer.numShards() != self.expected_shards) {
            return error.UnexpectedBufferShards;
        }
        self.local_bytes += buffer.localByteSize();
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs);
    const distributed_args = args.positional;
    if (args.model.len == 0 or
        !std.fs.path.isAbsolute(args.model) or
        args.prompt.len == 0 or
        args.seqlen == 0 or
        args.topk != 1 or
        distributed_args.processCount != 2 or
        distributed_args.rank >= distributed_args.processCount or
        distributed_args.namespace.len == 0)
    {
        return error.InvalidArguments;
    }

    var file_vfs: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer file_vfs.deinit();
    const io = file_vfs.io();

    var platform = try zml.Platform.init(allocator, io, .cuda, .{
        .distributed = .{
            .coordinator_address = try .parseLiteral(
                distributed_args.coordinator,
            ),
            .process_index = distributed_args.rank,
            .process_count = distributed_args.processCount,
            .namespace = distributed_args.namespace,
            .local_device_ids = &.{ 0, 1 },
        },
        .xla_gpu = .{
            .allocator = .{ .bfc = .{ .preallocate = false } },
        },
    });
    defer platform.deinit(allocator, io);

    if (platform.globalDevices().len != 4 or
        platform.addressableDevices().len != 2)
    {
        return error.UnexpectedTopology;
    }
    if (distributed_args.rank == 0) log.info("\n{f}", .{platform.fmtVerbose()});

    const backend: zml.attention.Backend = switch (args.backend) {
        .vanilla => .vanilla,
        .cuda_fa2 => .cuda_fa2,
    };
    for (platform.globalDevices()) |device| {
        if (device.processIndex() != distributed_args.rank) continue;
        const compute_capability = if (device.attribute(
            "compute_capability",
        )) |attribute| switch (attribute) {
            .string => |value| value,
            else => null,
        } else null;
        log.info(
            "rank={d} device={d} gpu=\"{s}\" compute_capability={s}",
            .{
                distributed_args.rank,
                device.id(),
                device.kind(),
                compute_capability orelse "unknown",
            },
        );
        if (backend == .cuda_fa2 and
            (compute_capability == null or
                (!std.mem.eql(u8, compute_capability.?, "8.9") and
                    !std.mem.eql(u8, compute_capability.?, "12.0"))))
        {
            return error.UnsupportedGpuArchitecture;
        }
    }
    if (!backend.isAvailable(platform)) {
        return error.UnsupportedAttentionBackend;
    }
    log.info(
        "rank={d} backend={s} global_devices={d} local_devices={d}",
        .{
            distributed_args.rank,
            @tagName(backend),
            platform.globalDevices().len,
            platform.addressableDevices().len,
        },
    );

    const shardings: models.Shardings = .{
        .model = try platform.registerShardingWithStrategy(
            "model",
            .mesh(.{ .model = .high_bandwidth }),
            .parseBindings(.{
                .model = .{ .network, .link },
            }),
        ),
        .experts = try platform.registerShardingWithStrategy(
            "experts",
            .mesh(.{ .experts = .high_bandwidth }),
            .parseBindings(.{
                .experts = .{ .network, .link },
            }),
        ),
    };
    if (shardings.model.numPartitionsForLogicalAxis(.model) != 4 or
        shardings.experts.numPartitionsForLogicalAxis(.experts) != 4)
    {
        return error.UnexpectedTopology;
    }

    log.info("rank={d} resolving model repository", .{distributed_args.rank});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    if (try models.detectModelType(allocator, io, repo) != .llama) {
        return error.UnsupportedModelType;
    }
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(
        allocator,
        io,
        repo,
    );
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var model = try models.LoadedModel.load(
        allocator,
        io,
        repo,
        store.view(),
        .{ .sampling_strategy = .{ .topk = args.topk } },
    );
    defer model.deinit(allocator);
    const config = model.llama.parsed_config.value;
    const sharded_dimensions = [_]struct { []const u8, u32 }{
        .{ "model.embed_tokens.weight hidden", config.hidden_size },
        .{ "model.layers.*.mlp intermediate", config.intermediate_size },
        .{ "model.layers.*.self_attn.q_proj heads", config.num_attention_heads },
        .{ "model.layers.*.self_attn.kv_proj heads", config.num_key_value_heads },
        .{
            "lm_head.weight vocabulary",
            if (model.llama.inner.lm_head == null) 4 else config.vocab_size,
        },
    };
    for (sharded_dimensions) |dimension| {
        if (dimension[1] == 0 or @mod(dimension[1], 4) != 0) {
            log.err(
                "four-way model sharding requires {s}={d} to be divisible by 4",
                .{ dimension[0], dimension[1] },
            );
            return error.IncompatibleModelDimensions;
        }
    }

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();
    var tokenizer = try loadTokenizer(allocator, io, repo, &progress);
    defer tokenizer.deinit();
    try platform.barrier("llm-repository-ready");

    var compiled_model = try allocator.create(models.CompiledModel);
    defer allocator.destroy(compiled_model);
    compiled_model.* = try models.LoadedModel.compile(
        &model,
        allocator,
        io,
        platform,
        backend,
        shardings,
        args.seqlen,
        &progress,
    );
    defer compiled_model.deinit();
    try platform.barrier("llm-compiled");

    var model_buffers = try models.LoadedModel.loadBuffers(
        &model,
        allocator,
        io,
        platform,
        &store,
        &progress,
        shardings,
    );
    defer model.unloadBuffers(&model_buffers, allocator);
    var buffer_stats: BufferStats = .{
        .expected_shards = @intCast(platform.addressableDevices().len),
    };
    try zml.meta.visit(BufferStats.add, &buffer_stats, &model_buffers);
    log.info(
        "rank={d} checkpoint_bytes={Bi:.2} local_weight_bytes={Bi:.2}",
        .{
            distributed_args.rank,
            registry.totalBytes(),
            buffer_stats.local_bytes,
        },
    );
    try platform.barrier("llm-weights-loaded");

    var session = try compiled_model.newSession(
        allocator,
        io,
        platform,
        &model_buffers,
        tokenizer,
    );
    defer session.deinit();
    const prompt_tokens = try session.tokenizePrompt(allocator, args.prompt);
    defer allocator.free(prompt_tokens);
    if (prompt_tokens.len == 0 or prompt_tokens.len >= args.seqlen) {
        return error.PromptTooLong;
    }
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, args.seqlen);
    defer tokens.deinit(allocator);
    try tokens.appendSlice(allocator, prompt_tokens);

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    var discard_buffer: [256]u8 = undefined;
    var discarding: std.Io.Writer.Discarding = .init(&discard_buffer);
    const output = if (distributed_args.rank == 0)
        &stdout.interface
    else
        &discarding.writer;
    try output.writeAll("\nGenerated response:\n");
    try session.runPrefill(tokens.items);
    const decode_started: std.Io.Timestamp = .now(io, .awake);
    try session.runDecode(&tokens, output);
    const decode_time = decode_started.untilNow(io, .awake);
    try output.writeAll("\n");
    try output.flush();

    const generated_tokens = tokens.items[prompt_tokens.len..];
    const token_hash = std.hash.Wyhash.hash(
        0,
        std.mem.sliceAsBytes(generated_tokens),
    );
    log.info(
        "rank={d} token_count={d} token_hash={x:0>16} " ++
            "decode={d:.2} token/s ({f})",
        .{
            distributed_args.rank,
            generated_tokens.len,
            token_hash,
            @as(f64, @floatFromInt(generated_tokens.len)) * std.time.ns_per_s /
                @as(f64, @floatFromInt(decode_time.nanoseconds)),
            decode_time,
        },
    );
    try platform.barrier("llm-generation-finished");
}

fn loadTokenizer(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    progress: *std.Progress.Node,
) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();
    const file = try repo.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(
        allocator,
        try file.length(io),
    );
    defer allocator.free(bytes);
    return .fromBytes(allocator, bytes);
}
