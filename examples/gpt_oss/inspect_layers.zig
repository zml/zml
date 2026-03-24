const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const gpt_oss = @import("gpt_oss.zig");
const GptOss = gpt_oss.GptOss;
const main_mod = @import("main.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: ?[]const u8 = null,
    seqlen: u32 = 1024,
    moe_backend: ?[]const u8 = null,
    prompt: ?[]const u8 = null,
};

const EmbedHarness = struct {
    embed_tokens: zml.nn.TokenEmbedding,

    pub fn forward(self: @This(), tokens: Tensor) Tensor {
        return gpt_oss.Model.embed(self.embed_tokens, tokens.withPartialTags(.{.s}));
    }
};

const LayerHarness = struct {
    layer: gpt_oss.TransformerLayer,

    pub fn forward(
        self: @This(),
        x: Tensor,
        token_index: Tensor,
        kv_cache: gpt_oss.KvCache,
        token_mask: Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) Tensor {
        return self.layer.forward(
            x.withPartialTags(.{ .s, .d }),
            token_index,
            kv_cache,
            token_mask.withPartialTags(.{.s}),
            moe_metadata,
            moe_parameters,
        )[0];
    }
};

const NormHarness = struct {
    norm: gpt_oss.RmsNorm,

    pub fn forward(self: @This(), x: Tensor) Tensor {
        return self.norm.forward(x.withPartialTags(.{ .s, .d }));
    }
};

const SelfAttnHarness = struct {
    self_attn: gpt_oss.SelfAttn,

    pub fn forward(self: @This(), x: Tensor, token_index: Tensor, kv_cache: gpt_oss.KvCache) Tensor {
        return self.self_attn.forward(
            x.withPartialTags(.{ .s, .d }),
            token_index,
            kv_cache,
        )[0];
    }
};

const LayerInternalsHarness = struct {
    layer: gpt_oss.TransformerLayer,

    pub fn forward(
        self: @This(),
        x: Tensor,
        token_index: Tensor,
        kv_cache: gpt_oss.KvCache,
        token_mask: Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { Tensor, Tensor, Tensor, Tensor, Tensor } {
        const x0 = x.withPartialTags(.{ .s, .d });
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.layer.input_layernorm.forward(x0_replicated);
        const delta0 = self.layer.self_attn.forward(x0_normalized, token_index, kv_cache)[0];
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.layer.post_attention_layernorm.forward(x1);
        const mlp_out = self.layer.mlp.forward(
            x1_normalized,
            token_mask.withPartialTags(.{.s}),
            moe_metadata,
            moe_parameters,
        );
        return .{ x0_normalized, delta0, x1, x1_normalized, mlp_out };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const io = init.io;

    const args = parseArgs(init);
    const prompt = args.prompt orelse return error.NoPrompt;

    const repo = try zml.safetensors.resolveModelRepo(io, args.model.?);
    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const options: GptOss.Options = .{
        .max_seq_len = args.seqlen,
        .max_prompt_len = args.seqlen,
        .tokens_per_expert_ratio = 1.0,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };

    const model: GptOss = try .init(allocator, store.view(), config, options);
    defer model.deinit(allocator);

    var tokenizer = try loadTokenizer(allocator, io, repo);
    defer tokenizer.deinit();

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    if (platform.target != .cuda) return error.RequiresCuda;

    const dtype = model.model.embed_tokens.weight.dtype();
    const moe_backend: zml.moe.Backend = if (args.moe_backend) |name| b: {
        if (std.mem.eql(u8, name, "flashinfer")) break :b .flashinfer;
        if (std.mem.eql(u8, name, "triton")) break :b .triton;
        return error.UnknownBackend;
    } else try zml.moe.Backend.auto(platform, dtype);

    try moe_backend.load(allocator);
    try moe_backend.register(platform);

    const tp_mesh: zml.sharding.LogicalMesh = try .init("tp_mesh", .{ .model = .high_bandwidth });
    const tp_strategy: zml.sharding.Strategy = try .suggest(tp_mesh, platform.physical_mesh);
    const sharding_tp: zml.sharding.Sharding = try .initFromStrategy(platform, tp_mesh, tp_strategy);
    const shardings = &.{sharding_tp};

    var progress = std.Progress.start(io, .{ .root_name = args.model.? });
    defer progress.end();

    var model_buffers = try GptOss.load(&model, allocator, io, platform, &store, shardings, &progress);
    defer GptOss.unloadBuffers(&model_buffers, allocator);

    switch (moe_backend) {
        .flashinfer => try gpt_oss.preprocessFlashinferSm90Mxfp4(allocator, io, platform, &model_buffers),
        .triton => try gpt_oss.preprocessTritonSm90Mxfp4(allocator, io, platform, &model_buffers),
    }

    const prompt_tok_buf = try allocator.alloc(u32, options.max_seq_len);
    defer allocator.free(prompt_tok_buf);
    const prompt_tok: []const u32 = try main_mod.tokenizePrompt(tokenizer, prompt, false, prompt_tok_buf);

    const tokens_shape = zml.Shape.init(.{ .s = options.max_seq_len }, .u32);
    var tokens_slice: zml.Slice = try .alloc(allocator, tokens_shape);
    defer tokens_slice.free(allocator);
    @memset(tokens_slice.items(u32), 0);
    @memcpy(tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);

    var token_mask_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = options.max_seq_len }, .bool));
    defer token_mask_slice.free(allocator);
    @memset(token_mask_slice.items(bool)[0..prompt_tok.len], true);
    @memset(token_mask_slice.items(bool)[prompt_tok.len..], false);

    var tokens_buffer = try zml.Buffer.fromSlice(io, platform, tokens_slice, sharding_tp);
    defer tokens_buffer.deinit();
    var token_mask_buffer = try zml.Buffer.fromSlice(io, platform, token_mask_slice, sharding_tp);
    defer token_mask_buffer.deinit();
    var token_index_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding_tp);
    defer token_index_buffer.deinit();

    const kv_cache = gpt_oss.KvCache.init(.init(.{
        .layer = model.model.layers.len,
        .k = args.seqlen,
        .h = config.num_key_value_heads,
        .hd = config.head_dim,
    }, dtype));
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform, sharding_tp);
    defer gpt_oss.KvCache.deinitBuffer(&kv_cache_buffers);
    try zeroKvCacheBuffers(allocator, io, platform, sharding_tp, &kv_cache_buffers);

    const moe_metadata: zml.moe.Metadata = .init(.fromBackend(moe_backend));
    const moe_parameters: zml.moe.Parameters = .init(.fromBackend(moe_backend));
    var moe_metadata_buffers = try moe_metadata.initBuffer(io, platform);
    defer zml.moe.Metadata.deinitBuffer(&moe_metadata_buffers);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    try stdout.interface.print("backend={s} seqlen={d} prompt_tokens={d}\n", .{ @tagName(moe_backend), args.seqlen, prompt_tok.len });

    const embed_harness: EmbedHarness = .{ .embed_tokens = model.model.embed_tokens };
    const embed_harness_buffers: zml.Bufferized(EmbedHarness) = .{ .embed_tokens = model_buffers.model.embed_tokens };
    const embed_tokens = Tensor.init(.{ .s = options.max_seq_len }, .u32);
    var hidden_buffer = try runSingleOutput(
        allocator,
        io,
        platform,
        embed_harness,
        embed_harness_buffers,
        .{embed_tokens},
        .{tokens_buffer},
        shardings,
    );
    defer hidden_buffer.deinit();
    try printDigest(allocator, io, &stdout.interface, "embed", hidden_buffer);

    if (model.model.layers.len > 0) {
        const layer0 = model.model.layers[0];
        const layer0_buffers = model_buffers.model.layers[0];

        var layer0_index_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding_tp);
        defer layer0_index_buffer.deinit();

        const internals_harness: LayerInternalsHarness = .{ .layer = layer0 };
        const internals_buffers: zml.Bufferized(LayerInternalsHarness) = .{ .layer = layer0_buffers };
        var input_ln_buffer, var self_attn_buffer, var residual_buffer, var post_ln_buffer, var moe_buffer = try runFiveOutputs(
            allocator,
            io,
            platform,
            internals_harness,
            internals_buffers,
            .{
                Tensor.init(.{ .s = options.max_seq_len, .d = config.hidden_size }, dtype),
                Tensor.init(.{}, .u32),
                gpt_oss.KvCache{
                    .k = kv_cache.k,
                    .v = kv_cache.v,
                    .layer_index = Tensor.init(.{}, .u32),
                },
                Tensor.init(.{ .s = options.max_seq_len }, .bool),
                moe_metadata,
                moe_parameters,
            },
            .{
                hidden_buffer,
                token_index_buffer,
                .{
                    .k = kv_cache_buffers.k,
                    .v = kv_cache_buffers.v,
                    .layer_index = layer0_index_buffer,
                },
                token_mask_buffer,
                moe_metadata_buffers,
            },
            shardings,
        );
        defer input_ln_buffer.deinit();
        defer self_attn_buffer.deinit();
        defer residual_buffer.deinit();
        defer post_ln_buffer.deinit();
        defer moe_buffer.deinit();
        try printDigest(allocator, io, &stdout.interface, "layer.0.input_layernorm", input_ln_buffer);
        try printDigest(allocator, io, &stdout.interface, "layer.0.self_attn", self_attn_buffer);
        try printDigest(allocator, io, &stdout.interface, "layer.0.residual_after_attn", residual_buffer);
        try printDigest(allocator, io, &stdout.interface, "layer.0.post_attention_layernorm", post_ln_buffer);
        try printDigest(allocator, io, &stdout.interface, "layer.0.mlp", moe_buffer);
    }

    for (model.model.layers, model_buffers.model.layers, 0..) |layer, layer_buffers, layer_index| {
        const harness: LayerHarness = .{ .layer = layer };
        const harness_buffers: zml.Bufferized(LayerHarness) = .{ .layer = layer_buffers };
        const x = Tensor.init(.{ .s = options.max_seq_len, .d = config.hidden_size }, dtype);
        const token_index = Tensor.init(.{}, .u32);
        const token_mask = Tensor.init(.{ .s = options.max_seq_len }, .bool);

        var layer_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(layer_index)), .u32, sharding_tp);
        defer layer_index_buffer.deinit();

        const output = try runSingleOutput(
            allocator,
            io,
            platform,
            harness,
            harness_buffers,
            .{
                x,
                token_index,
                gpt_oss.KvCache{
                    .k = kv_cache.k,
                    .v = kv_cache.v,
                    .layer_index = Tensor.init(.{}, .u32),
                },
                token_mask,
                moe_metadata,
                moe_parameters,
            },
            .{
                hidden_buffer,
                token_index_buffer,
                .{
                    .k = kv_cache_buffers.k,
                    .v = kv_cache_buffers.v,
                    .layer_index = layer_index_buffer,
                },
                token_mask_buffer,
                moe_metadata_buffers,
            },
            shardings,
        );
        hidden_buffer.deinit();
        hidden_buffer = output;
        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "layer.{d}", .{layer_index});
        try printDigest(allocator, io, &stdout.interface, label, hidden_buffer);
    }

    const norm_harness: NormHarness = .{ .norm = model.model.norm };
    const norm_harness_buffers: zml.Bufferized(NormHarness) = .{ .norm = model_buffers.model.norm };
    const norm_x = Tensor.init(.{ .s = options.max_seq_len, .d = config.hidden_size }, dtype);
    var norm_buffer = try runSingleOutput(
        allocator,
        io,
        platform,
        norm_harness,
        norm_harness_buffers,
        .{norm_x},
        .{hidden_buffer},
        shardings,
    );
    defer norm_buffer.deinit();
    try printDigest(allocator, io, &stdout.interface, "norm", norm_buffer);
}

fn parseArgs(init: std.process.Init) Args {
    var ret: Args = .{};
    var it = init.minimal.args.iterate();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--model=")) {
            ret.model = arg["--model=".len..];
        } else if (std.mem.startsWith(u8, arg, "--seqlen=")) {
            ret.seqlen = std.fmt.parseUnsigned(u32, arg["--seqlen=".len..], 10) catch ret.seqlen;
        } else if (std.mem.startsWith(u8, arg, "--moe_backend=")) {
            ret.moe_backend = arg["--moe_backend=".len..];
        } else if (std.mem.startsWith(u8, arg, "--prompt=")) {
            ret.prompt = arg["--prompt=".len..];
        }
    }
    if (ret.model == null) @panic("Missing --model");
    return ret;
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(GptOss.Config) {
    var parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(gpt_oss.GptOss.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    parsed_config.value.rope_scaling.setRopeTheta(parsed_config.value.rope_theta);
    return parsed_config;
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    errdefer allocator.free(bytes);
    return try .fromBytes(allocator, io, bytes);
}

fn runSingleOutput(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    harness: anytype,
    harness_buffers: zml.Bufferized(@TypeOf(harness)),
    args: anytype,
    input_buffers: anytype,
    shardings: []const zml.sharding.Sharding,
) !zml.Buffer {
    const exe = try platform.compile(allocator, io, harness, .forward, args, .{ .shardings = shardings });
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ harness_buffers, input_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var output: zml.Buffer = undefined;
    exe_results.fill(.{&output});
    return output;
}

fn runFiveOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    harness: anytype,
    harness_buffers: zml.Bufferized(@TypeOf(harness)),
    args: anytype,
    input_buffers: anytype,
    shardings: []const zml.sharding.Sharding,
) !struct { zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer, zml.Buffer } {
    const exe = try platform.compile(allocator, io, harness, .forward, args, .{ .shardings = shardings });
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ harness_buffers, input_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var out0: zml.Buffer = undefined;
    var out1: zml.Buffer = undefined;
    var out2: zml.Buffer = undefined;
    var out3: zml.Buffer = undefined;
    var out4: zml.Buffer = undefined;
    exe_results.fill(.{ &out0, &out1, &out2, &out3, &out4 });
    return .{ out0, out1, out2, out3, out4 };
}

fn printDigest(
    allocator: std.mem.Allocator,
    io: std.Io,
    writer: *std.Io.Writer,
    label: []const u8,
    buffer: zml.Buffer,
) !void {
    var host = try buffer.toSliceAlloc(allocator, io);
    defer host.free(allocator);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(host.data());
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    var digest_hex: [64]u8 = undefined;
    for (digest, 0..) |byte, i| {
        const hi = byte >> 4;
        const lo = byte & 0x0f;
        digest_hex[2 * i] = if (hi < 10) '0' + hi else 'a' + (hi - 10);
        digest_hex[2 * i + 1] = if (lo < 10) '0' + lo else 'a' + (lo - 10);
    }

    try writer.print("{s} shape={f} sha256={s}\n", .{
        label,
        host.shape,
        digest_hex[0..],
    });
}


fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    buffer: *zml.Buffer,
) !void {
    const shape = buffer.shape();
    const zero_slice: zml.Slice = try .alloc(allocator, shape);
    defer zero_slice.free(allocator);
    @memset(zero_slice.data(), 0);

    var zeroed = try zml.Buffer.fromSlice(io, platform, zero_slice, sharding);
    errdefer zeroed.deinit();
    buffer.deinit();
    buffer.* = zeroed;
}

fn zeroKvCacheBuffers(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    cache: *zml.Bufferized(gpt_oss.KvCache),
) !void {
    try zeroBuffer(allocator, io, platform, sharding, &cache.k);
    try zeroBuffer(allocator, io, platform, sharding, &cache.v);
}
