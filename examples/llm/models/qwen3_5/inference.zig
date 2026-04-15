const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5);

const CompileModelResult = struct {
    prefill: ComposedKernelExe,
    decode: ComposedKernelExe,
};

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, shardings: common.Shardings) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
        return .{
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, model_partitions),
            .rng = .init(),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: ComposedKernelExe,
    decode: ComposedKernelExe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        qwen_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compile_result = try compileModel(allocator, io, platform, qwen_model, parameters, progress);
        return .{ .loaded_model = loaded_model, .prefill = compile_result.prefill, .decode = compile_result.decode, .params = parameters };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

pub const Args = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
};

const SelfAttnLayerProgram = struct {
    layer: model.TransformerLayer,

    pub fn forward(
        self: SelfAttnLayerProgram,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
        layer_index: zml.Tensor,
    ) struct { zml.Tensor, model.KvCache } {
        return self.layer.forward(
            hidden,
            token_index,
            .{
                .parent = kv_cache,
                .cache = .{
                    .self_attn = .{
                        .k = kv_cache.self_attn.k,
                        .v = kv_cache.self_attn.v,
                        .layer_index = layer_index,
                    },
                },
            },
        );
    }
};

const LinearAttnLayerProgram = struct {
    layer: model.TransformerLayer,

    pub fn forward(
        self: LinearAttnLayerProgram,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
        layer_index: zml.Tensor,
    ) struct { zml.Tensor, model.KvCache } {
        _ = token_index;
        return self.layer.forward(
            hidden,
            zml.Tensor.scalar(0, .u32),
            .{
                .parent = kv_cache,
                .cache = .{
                    .linear_attn = .{
                        .conv_state = kv_cache.gated_delta_net.conv_state,
                        .recurrent_state = kv_cache.gated_delta_net.recurrent_state,
                        .layer_index = layer_index,
                    },
                },
            },
        );
    }
};

const HeadProgram = struct {
    norm: model.RmsNorm,
    lm_head: zml.nn.Linear,

    pub fn forward(
        self: HeadProgram,
        hidden: zml.Tensor,
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
        sampling_strategy: zml.nn.SamplingStrategy,
    ) struct { zml.Tensor, zml.Tensor.Rng } {
        const normalized = self.norm.forward(hidden);
        const logits = self.lm_head.forward(normalized.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const new_tokens, const new_rng = zml.nn.sampleTokens(logits, sampling_strategy, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_rng };
    }
};

pub const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    self_attn_layer: ?zml.Exe = null,
    linear_attn_layer: ?zml.Exe = null,
    head: zml.Exe,

    pub fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        if (self.self_attn_layer) |exe| exe.deinit();
        if (self.linear_attn_layer) |exe| exe.deinit();
        self.head.deinit();
    }

    pub fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = undefined;
        {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers.text_model.embed_tokens, args.tokens_buf });
            self.embed_tokens.call(exe_args, &results);
            results.fill(.{&hidden_buf});
        }
        defer hidden_buf.deinit();

        const replicated_sharding = try zml.sharding.replicatedSharding(args.platform);
        var self_attn_layer_index: u32 = 0;
        var linear_attn_layer_index: u32 = 0;

        for (args.model_buffers.text_model.layers) |layer_buffers| {
            var exe: zml.Exe = undefined;
            var layer_index: u32 = undefined;

            switch (layer_buffers.attn) {
                .self_attn => {
                    exe = self.self_attn_layer orelse unreachable;
                    layer_index = self_attn_layer_index;
                    self_attn_layer_index += 1;

                    const layer_program_buffers: zml.Bufferized(SelfAttnLayerProgram) = .{
                        .layer = layer_buffers,
                    };
                    var layer_index_buf = try zml.Buffer.scalar(args.io, args.platform, layer_index, .u32, replicated_sharding);
                    defer layer_index_buf.deinit();

                    var exe_args = try exe.args(args.allocator);
                    defer exe_args.deinit(args.allocator);
                    var results = try exe.results(args.allocator);
                    defer results.deinit(args.allocator);

                    exe_args.set(.{
                        layer_program_buffers,
                        &hidden_buf,
                        args.token_index_buf,
                        args.kv_cache_buffers,
                        &layer_index_buf,
                    });
                    exe.call(exe_args, &results);
                    results.fill(.{ &hidden_buf, args.kv_cache_buffers });
                },
                .linear_attn => {
                    exe = self.linear_attn_layer orelse unreachable;
                    layer_index = linear_attn_layer_index;
                    linear_attn_layer_index += 1;

                    const layer_program_buffers: zml.Bufferized(LinearAttnLayerProgram) = .{
                        .layer = layer_buffers,
                    };
                    var layer_index_buf = try zml.Buffer.scalar(args.io, args.platform, layer_index, .u32, replicated_sharding);
                    defer layer_index_buf.deinit();

                    var exe_args = try exe.args(args.allocator);
                    defer exe_args.deinit(args.allocator);
                    var results = try exe.results(args.allocator);
                    defer results.deinit(args.allocator);

                    exe_args.set(.{
                        layer_program_buffers,
                        &hidden_buf,
                        args.token_index_buf,
                        args.kv_cache_buffers,
                        &layer_index_buf,
                    });
                    exe.call(exe_args, &results);
                    results.fill(.{ &hidden_buf, args.kv_cache_buffers });
                },
            }
        }

        const head_program_buffers: zml.Bufferized(HeadProgram) = .{
            .norm = args.model_buffers.text_model.norm,
            .lm_head = args.model_buffers.lm_head,
        };

        var exe_args = try self.head.args(args.allocator);
        defer exe_args.deinit(args.allocator);
        var results = try self.head.results(args.allocator);
        defer results.deinit(args.allocator);

        exe_args.set(.{
            head_program_buffers,
            hidden_buf,
            args.tokens_buf,
            args.rng_buf,
        });
        self.head.call(exe_args, &results);
        results.fill(.{ args.tokens_buf, args.rng_buf });
    }
};

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationParameters,
    progress: *std.Progress.Node,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen_model_: model.Model,
            parameters_: CompilationParameters,
            progress_: *std.Progress.Node,
        ) !ComposedKernelExe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill...", 1);
            defer node_.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill [{f}]", .{now_.untilNow(io_, .awake)});

            return compileComposedKernelExe(
                allocator_,
                io_,
                platform_,
                qwen_model_,
                parameters_,
                parameters_.prefill_tokens.shape().dim(.s),
                progress_,
            );
        }
    }.call, .{ allocator, io, platform, qwen_model, parameters, progress });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen_model_: model.Model,
            parameters_: CompilationParameters,
            progress_: *std.Progress.Node,
        ) !ComposedKernelExe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode...", 1);
            defer node_.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode [{f}]", .{now_.untilNow(io_, .awake)});

            return compileComposedKernelExe(
                allocator_,
                io_,
                platform_,
                qwen_model_,
                parameters_,
                parameters_.decode_tokens.shape().dim(.s),
                progress_,
            );
        }
    }.call, .{ allocator, io, platform, qwen_model, parameters, progress });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill = try prefill_future.await(io);
    const decode = try decode_future.await(io);

    return .{
        .prefill = prefill,
        .decode = decode,
    };
}

fn compileComposedKernelExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationParameters,
    seqlen: i64,
    progress: *std.Progress.Node,
) !ComposedKernelExe {
    return .{
        .embed_tokens = try compileEmbedTokens(allocator, io, platform, qwen_model.text_model.embed_tokens, parameters, seqlen, progress),
        .self_attn_layer = try compileSelfAttnLayer(allocator, io, platform, qwen_model, parameters, seqlen, progress),
        .linear_attn_layer = try compileLinearAttnLayer(allocator, io, platform, qwen_model, parameters, seqlen, progress),
        .head = try compileHead(allocator, io, platform, qwen_model, parameters, seqlen, progress),
    };
}

fn compileEmbedTokens(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    embed_tokens: zml.nn.TokenEmbedding,
    parameters: CompilationParameters,
    seqlen: i64,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling embed_tokens...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled embed_tokens [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .b = 1, .s = seqlen }, .u32);
    const all_shardings = parameters.shardings.all();
    return platform.compile(allocator, io, embed_tokens, .forward, .{tokens}, .{ .shardings = &all_shardings });
}

fn compileSelfAttnLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationParameters,
    seqlen: i64,
    progress: *std.Progress.Node,
) !?zml.Exe {
    const layer = firstLayer(qwen_model.text_model.layers, .full_attention) orelse return null;

    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling self-attention layer...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled self-attention layer [{f}]", .{now.untilNow(io, .awake)});

    const hidden: zml.Tensor = .init(.{ .b = 1, .s = seqlen, .d = qwen_model.config.text_config.hidden_size }, qwen_model.text_model.embed_tokens.weight.dtype());
    const layer_index: zml.Tensor = .init(.{}, .u32);
    const all_shardings = parameters.shardings.all();
    return try platform.compile(
        allocator,
        io,
        SelfAttnLayerProgram{ .layer = layer },
        .forward,
        .{
            hidden,
            parameters.token_index,
            parameters.kv_cache,
            layer_index,
        },
        .{ .shardings = &all_shardings },
    );
}

fn compileLinearAttnLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationParameters,
    seqlen: i64,
    progress: *std.Progress.Node,
) !?zml.Exe {
    const layer = firstLayer(qwen_model.text_model.layers, .linear_attention) orelse return null;

    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling linear-attention layer...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled linear-attention layer [{f}]", .{now.untilNow(io, .awake)});

    const hidden: zml.Tensor = .init(.{ .b = 1, .s = seqlen, .d = qwen_model.config.text_config.hidden_size }, qwen_model.text_model.embed_tokens.weight.dtype());
    const layer_index: zml.Tensor = .init(.{}, .u32);
    const all_shardings = parameters.shardings.all();
    return try platform.compile(
        allocator,
        io,
        LinearAttnLayerProgram{ .layer = layer },
        .forward,
        .{
            hidden,
            parameters.token_index,
            parameters.kv_cache,
            layer_index,
        },
        .{ .shardings = &all_shardings },
    );
}

fn compileHead(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationParameters,
    seqlen: i64,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling head...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled head [{f}]", .{now.untilNow(io, .awake)});

    const hidden: zml.Tensor = .init(.{ .b = 1, .s = seqlen, .d = qwen_model.config.text_config.hidden_size }, qwen_model.text_model.embed_tokens.weight.dtype());
    const all_shardings = parameters.shardings.all();
    return platform.compile(
        allocator,
        io,
        HeadProgram{
            .norm = qwen_model.text_model.norm,
            .lm_head = qwen_model.lm_head,
        },
        .forward,
        .{
            hidden,
            .init(.{ .b = 1, .s = seqlen }, .u32),
            parameters.rng,
            qwen_model.gen_options.sampling_strategy,
        },
        .{ .shardings = &all_shardings },
    );
}

fn firstLayer(layers: []const model.TransformerLayer, layer_type: model.LayerType) ?model.TransformerLayer {
    for (layers) |layer| {
        switch (layer.attn) {
            .self_attn => if (layer_type == .full_attention) return layer,
            .linear_attn => if (layer_type == .linear_attention) return layer,
        }
    }
    return null;
}
