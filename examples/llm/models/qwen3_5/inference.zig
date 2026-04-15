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
        var state = try RunState.init(args.allocator, args.io, args.platform, self, args.model_buffers);
        defer state.deinit(args.allocator);
        try self.runWithState(args, &state);
    }

    pub fn runWithState(self: *const ComposedKernelExe, args: Args, state: *RunState) !void {
        var hidden_buf: zml.Buffer = undefined;
        {
            state.embed_args.setPartial(.{args.tokens_buf}, 0);
            self.embed_tokens.call(state.embed_args, &state.embed_results);
            state.embed_results.fill(.{&hidden_buf});
        }
        defer hidden_buf.deinit();

        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;

        for (args.model_buffers.text_model.layers) |layer_buffers| {
            switch (layer_buffers.attn) {
                .self_attn => {
                    const exe = self.self_attn_layer orelse unreachable;
                    const exe_args = &state.self_attn_layer_args[self_attn_layer_index];
                    const layer_index_buf = &state.self_attn_layer_index_buffers[self_attn_layer_index];
                    self_attn_layer_index += 1;

                    exe_args.setPartial(.{
                        &hidden_buf,
                        args.token_index_buf,
                        args.kv_cache_buffers,
                        layer_index_buf,
                    }, 0);
                    exe.call(exe_args.*, &state.self_attn_layer_results.?);
                    state.self_attn_layer_results.?.fill(.{ &hidden_buf, args.kv_cache_buffers });
                },
                .linear_attn => {
                    const exe = self.linear_attn_layer orelse unreachable;
                    const exe_args = &state.linear_attn_layer_args[linear_attn_layer_index];
                    const layer_index_buf = &state.linear_attn_layer_index_buffers[linear_attn_layer_index];
                    linear_attn_layer_index += 1;

                    exe_args.setPartial(.{
                        &hidden_buf,
                        args.token_index_buf,
                        args.kv_cache_buffers,
                        layer_index_buf,
                    }, 0);
                    exe.call(exe_args.*, &state.linear_attn_layer_results.?);
                    state.linear_attn_layer_results.?.fill(.{ &hidden_buf, args.kv_cache_buffers });
                },
            }
        }

        state.head_args.setPartial(.{
            hidden_buf,
            args.tokens_buf,
            args.rng_buf,
        }, 0);
        self.head.call(state.head_args, &state.head_results);
        state.head_results.fill(.{ args.tokens_buf, args.rng_buf });
    }
};

pub const RunState = struct {
    embed_args: zml.Exe.Arguments,
    embed_results: zml.Exe.Results,
    self_attn_layer_args: []zml.Exe.Arguments,
    self_attn_layer_results: ?zml.Exe.Results,
    linear_attn_layer_args: []zml.Exe.Arguments,
    linear_attn_layer_results: ?zml.Exe.Results,
    head_args: zml.Exe.Arguments,
    head_results: zml.Exe.Results,
    self_attn_layer_index_buffers: []zml.Buffer,
    linear_attn_layer_index_buffers: []zml.Buffer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        exe: *const ComposedKernelExe,
        model_buffers: *model.Buffers,
    ) !RunState {
        var embed_args = try exe.embed_tokens.args(allocator);
        errdefer embed_args.deinit(allocator);
        embed_args.bake(.{model_buffers.text_model.embed_tokens});

        var embed_results = try exe.embed_tokens.results(allocator);
        errdefer embed_results.deinit(allocator);

        const replicated_sharding = try zml.sharding.replicatedSharding(platform);

        const self_attn_layer_count = countSelfAttnLayers(model_buffers.text_model.layers);
        const self_attn_layer_args = try allocator.alloc(zml.Exe.Arguments, self_attn_layer_count);
        errdefer allocator.free(self_attn_layer_args);
        var initialized_self_attn_layer_args: usize = 0;
        errdefer {
            for (self_attn_layer_args[0..initialized_self_attn_layer_args]) |*args_| args_.deinit(allocator);
        }

        const self_attn_layer_index_buffers = try allocator.alloc(zml.Buffer, self_attn_layer_count);
        errdefer allocator.free(self_attn_layer_index_buffers);
        var initialized_self_attn_layer_index_buffers: usize = 0;
        errdefer {
            for (self_attn_layer_index_buffers[0..initialized_self_attn_layer_index_buffers]) |*buffer| buffer.deinit();
        }

        if (exe.self_attn_layer) |self_attn_exe| {
            var self_attn_dense_index: usize = 0;
            for (model_buffers.text_model.layers) |layer_buffers| switch (layer_buffers.attn) {
                .self_attn => {
                    self_attn_layer_args[self_attn_dense_index] = try self_attn_exe.args(allocator);
                    initialized_self_attn_layer_args += 1;
                    const layer_program_buffers: zml.Bufferized(SelfAttnLayerProgram) = .{
                        .layer = layer_buffers,
                    };
                    self_attn_layer_args[self_attn_dense_index].bake(.{layer_program_buffers});

                    self_attn_layer_index_buffers[self_attn_dense_index] = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(self_attn_dense_index)), .u32, replicated_sharding);
                    initialized_self_attn_layer_index_buffers += 1;

                    self_attn_dense_index += 1;
                },
                else => {},
            };
        }

        var self_attn_layer_results: ?zml.Exe.Results = if (exe.self_attn_layer) |self_attn_exe|
            try self_attn_exe.results(allocator)
        else
            null;
        errdefer if (self_attn_layer_results) |*results_| results_.deinit(allocator);

        const linear_attn_layer_count = countLinearAttnLayers(model_buffers.text_model.layers);
        const linear_attn_layer_args = try allocator.alloc(zml.Exe.Arguments, linear_attn_layer_count);
        errdefer allocator.free(linear_attn_layer_args);
        var initialized_linear_attn_layer_args: usize = 0;
        errdefer {
            for (linear_attn_layer_args[0..initialized_linear_attn_layer_args]) |*args_| args_.deinit(allocator);
        }

        const linear_attn_layer_index_buffers = try allocator.alloc(zml.Buffer, linear_attn_layer_count);
        errdefer allocator.free(linear_attn_layer_index_buffers);
        var initialized_linear_attn_layer_index_buffers: usize = 0;
        errdefer {
            for (linear_attn_layer_index_buffers[0..initialized_linear_attn_layer_index_buffers]) |*buffer| buffer.deinit();
        }

        if (exe.linear_attn_layer) |linear_attn_exe| {
            var linear_attn_dense_index: usize = 0;
            for (model_buffers.text_model.layers) |layer_buffers| switch (layer_buffers.attn) {
                .linear_attn => {
                    linear_attn_layer_args[linear_attn_dense_index] = try linear_attn_exe.args(allocator);
                    initialized_linear_attn_layer_args += 1;
                    const layer_program_buffers: zml.Bufferized(LinearAttnLayerProgram) = .{
                        .layer = layer_buffers,
                    };
                    linear_attn_layer_args[linear_attn_dense_index].bake(.{layer_program_buffers});

                    linear_attn_layer_index_buffers[linear_attn_dense_index] = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(linear_attn_dense_index)), .u32, replicated_sharding);
                    initialized_linear_attn_layer_index_buffers += 1;

                    linear_attn_dense_index += 1;
                },
                else => {},
            };
        }

        var linear_attn_layer_results: ?zml.Exe.Results = if (exe.linear_attn_layer) |linear_attn_exe|
            try linear_attn_exe.results(allocator)
        else
            null;
        errdefer if (linear_attn_layer_results) |*results_| results_.deinit(allocator);

        var head_args = try exe.head.args(allocator);
        errdefer head_args.deinit(allocator);
        const head_program_buffers: zml.Bufferized(HeadProgram) = .{
            .norm = model_buffers.text_model.norm,
            .lm_head = model_buffers.lm_head,
        };
        head_args.bake(.{head_program_buffers});

        var head_results = try exe.head.results(allocator);
        errdefer head_results.deinit(allocator);

        return .{
            .embed_args = embed_args,
            .embed_results = embed_results,
            .self_attn_layer_args = self_attn_layer_args,
            .self_attn_layer_results = self_attn_layer_results,
            .linear_attn_layer_args = linear_attn_layer_args,
            .linear_attn_layer_results = linear_attn_layer_results,
            .head_args = head_args,
            .head_results = head_results,
            .self_attn_layer_index_buffers = self_attn_layer_index_buffers,
            .linear_attn_layer_index_buffers = linear_attn_layer_index_buffers,
        };
    }

    pub fn deinit(self: *RunState, allocator: std.mem.Allocator) void {
        self.embed_args.deinit(allocator);
        self.embed_results.deinit(allocator);
        for (self.self_attn_layer_args) |*args_| args_.deinit(allocator);
        allocator.free(self.self_attn_layer_args);
        if (self.self_attn_layer_results) |*results_| results_.deinit(allocator);
        for (self.linear_attn_layer_args) |*args_| args_.deinit(allocator);
        allocator.free(self.linear_attn_layer_args);
        if (self.linear_attn_layer_results) |*results_| results_.deinit(allocator);
        self.head_args.deinit(allocator);
        self.head_results.deinit(allocator);
        for (self.self_attn_layer_index_buffers) |*buffer| buffer.deinit();
        allocator.free(self.self_attn_layer_index_buffers);
        for (self.linear_attn_layer_index_buffers) |*buffer| buffer.deinit();
        allocator.free(self.linear_attn_layer_index_buffers);
    }
};

fn countSelfAttnLayers(layers: []const zml.Bufferized(model.TransformerLayer)) usize {
    var count: usize = 0;
    for (layers) |layer_buffers| switch (layer_buffers.attn) {
        .self_attn => count += 1,
        else => {},
    };
    return count;
}

fn countLinearAttnLayers(layers: []const zml.Bufferized(model.TransformerLayer)) usize {
    var count: usize = 0;
    for (layers) |layer_buffers| switch (layer_buffers.attn) {
        .linear_attn => count += 1,
        else => {},
    };
    return count;
}

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
