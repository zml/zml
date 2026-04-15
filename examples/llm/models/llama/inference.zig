const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.llama);

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
    seqlen: usize,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend, shardings: common.Shardings) CompilationParameters {
        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(.init(.{
                .layer = mdl.model.layers.len,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
            }, mdl.model.embed_tokens.weight.dtype())),
            .rng = .init(),
            .attention_metadata = .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            .attention_parameters = .init(.fromBackend(backend)),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

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
        llama_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compiled = try compileModel(allocator, io, platform, llama_model, parameters, progress);
        return .{ .loaded_model = loaded_model, .prefill = compiled.prefill, .decode = compiled.decode, .params = parameters };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

const CompileModelResult = struct {
    prefill: ComposedKernelExe,
    decode: ComposedKernelExe,
};

pub const Args = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
};

const LayerProgram = struct {
    layer: model.TransformerLayer,

    pub fn forward(
        self: LayerProgram,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
        layer_index: zml.Tensor,
        attention_metadata: attention.Metadata,
        attention_parameters: attention.Parameters,
    ) struct { zml.Tensor, model.KvCache } {
        return self.layer.forward(
            hidden,
            token_index,
            .{
                .k = kv_cache.k,
                .v = kv_cache.v,
                .layer_index = layer_index,
            },
            attention_metadata,
            attention_parameters,
        );
    }
};

const HeadProgram = struct {
    norm: model.RmsNorm,
    lm_head: ?zml.nn.Linear,
    embed_tokens: zml.nn.TokenEmbedding,

    pub fn forward(
        self: HeadProgram,
        hidden: zml.Tensor,
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
        gen_opts: zml.nn.SamplingStrategy,
    ) struct { zml.Tensor, zml.Tensor.Rng } {
        const normalized = self.norm.forward(hidden);

        var logits = blk: {
            if (self.lm_head) |lm_head| {
                break :blk lm_head.forward(normalized).rename(.{ .dout = .d });
            }
            break :blk self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(normalized, .d);
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const new_tokens, const new_rng = zml.nn.sampleTokens(logits, gen_opts, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_rng };
    }
};

pub const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    layer: zml.Exe,
    head: zml.Exe,

    pub fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        self.layer.deinit();
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

        for (args.model_buffers.model.layers, 0..) |layer_buffers, layer_index| {
            _ = layer_buffers;
            state.layer_args[layer_index].setPartial(.{
                &hidden_buf,
                args.token_index_buf,
                args.kv_cache_buffers,
                &state.layer_index_buffers[layer_index],
            }, 0);
            state.layer_args[layer_index].setPartial(.{
                args.attention_metadata_buffers,
            }, 6);
            self.layer.call(state.layer_args[layer_index], &state.layer_results);
            state.layer_results.fill(.{ &hidden_buf, args.kv_cache_buffers });
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
    layer_args: []zml.Exe.Arguments,
    layer_results: zml.Exe.Results,
    head_args: zml.Exe.Arguments,
    head_results: zml.Exe.Results,
    layer_index_buffers: []zml.Buffer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        exe: *const ComposedKernelExe,
        model_buffers: *model.Buffers,
    ) !RunState {
        var embed_args = try exe.embed_tokens.args(allocator);
        errdefer embed_args.deinit(allocator);
        embed_args.bake(.{model_buffers.model.embed_tokens});

        var embed_results = try exe.embed_tokens.results(allocator);
        errdefer embed_results.deinit(allocator);

        const layer_args = try allocator.alloc(zml.Exe.Arguments, model_buffers.model.layers.len);
        errdefer allocator.free(layer_args);
        var initialized_layer_args: usize = 0;
        errdefer {
            for (layer_args[0..initialized_layer_args]) |*args_| args_.deinit(allocator);
        }

        for (model_buffers.model.layers, 0..) |layer_buffers, i| {
            layer_args[i] = try exe.layer.args(allocator);
            initialized_layer_args += 1;
            const layer_program_buffers: zml.Bufferized(LayerProgram) = .{
                .layer = layer_buffers,
            };
            layer_args[i].bake(.{layer_program_buffers});
        }

        var layer_results = try exe.layer.results(allocator);
        errdefer layer_results.deinit(allocator);

        var head_args = try exe.head.args(allocator);
        errdefer head_args.deinit(allocator);
        const head_program_buffers: zml.Bufferized(HeadProgram) = .{
            .norm = model_buffers.model.norm,
            .lm_head = model_buffers.lm_head,
            .embed_tokens = model_buffers.model.embed_tokens,
        };
        head_args.bake(.{head_program_buffers});

        var head_results = try exe.head.results(allocator);
        errdefer head_results.deinit(allocator);

        const replicated_sharding = try zml.sharding.replicatedSharding(platform);
        const layer_index_buffers = try allocator.alloc(zml.Buffer, model_buffers.model.layers.len);
        errdefer allocator.free(layer_index_buffers);
        var initialized_layer_index_buffers: usize = 0;
        errdefer {
            for (layer_index_buffers[0..initialized_layer_index_buffers]) |*buffer| buffer.deinit();
        }

        for (layer_index_buffers, 0..) |*buffer, i| {
            buffer.* = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(i)), .u32, replicated_sharding);
            initialized_layer_index_buffers += 1;
        }

        return .{
            .embed_args = embed_args,
            .embed_results = embed_results,
            .layer_args = layer_args,
            .layer_results = layer_results,
            .head_args = head_args,
            .head_results = head_results,
            .layer_index_buffers = layer_index_buffers,
        };
    }

    pub fn deinit(self: *RunState, allocator: std.mem.Allocator) void {
        self.embed_args.deinit(allocator);
        self.embed_results.deinit(allocator);
        for (self.layer_args) |*args_| args_.deinit(allocator);
        allocator.free(self.layer_args);
        self.layer_results.deinit(allocator);
        self.head_args.deinit(allocator);
        self.head_results.deinit(allocator);
        for (self.layer_index_buffers) |*buffer| buffer.deinit();
        allocator.free(self.layer_index_buffers);
    }
};

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
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
            llama_model_: model.Model,
            parameters_: CompilationParameters,
            progress_: *std.Progress.Node,
        ) !ComposedKernelExe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill...", 1);
            defer node_.end();

            return compileComposedKernelExe(
                allocator_,
                io_,
                platform_,
                llama_model_,
                parameters_,
                @intCast(parameters_.prefill_tokens.shape().dim(.s)),
                progress_,
            );
        }
    }.call, .{ allocator, io, platform, llama_model, parameters, progress });
    var prefill_future_awaited = false;
    errdefer if (!prefill_future_awaited) if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            llama_model_: model.Model,
            parameters_: CompilationParameters,
            progress_: *std.Progress.Node,
        ) !ComposedKernelExe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode...", 1);
            defer node_.end();

            return compileComposedKernelExe(
                allocator_,
                io_,
                platform_,
                llama_model_,
                parameters_,
                @intCast(parameters_.decode_tokens.shape().dim(.s)),
                progress_,
            );
        }
    }.call, .{ allocator, io, platform, llama_model, parameters, progress });
    var decode_future_awaited = false;
    errdefer if (!decode_future_awaited) if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill = try prefill_future.await(io);
    prefill_future_awaited = true;
    errdefer prefill.deinit();

    const decode = try decode_future.await(io);
    decode_future_awaited = true;

    return .{
        .prefill = prefill,
        .decode = decode,
    };
}

fn compileComposedKernelExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    seqlen: usize,
    progress: *std.Progress.Node,
) !ComposedKernelExe {
    return .{
        .embed_tokens = try compileEmbedTokens(allocator, io, platform, llama_model.model.embed_tokens, parameters, seqlen, progress),
        .layer = try compileLayer(allocator, io, platform, llama_model.model.layers[0], parameters, seqlen, progress),
        .head = try compileHead(allocator, io, platform, llama_model, parameters, seqlen, progress),
    };
}

fn compileEmbedTokens(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    embed_tokens: zml.nn.TokenEmbedding,
    parameters: CompilationParameters,
    seqlen: usize,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling embed_tokens...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled embed_tokens [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);
    const all_shardings = parameters.shardings.all();
    return platform.compile(allocator, io, embed_tokens, .forward, .{tokens}, .{ .shardings = &all_shardings });
}

fn compileLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    layer: model.TransformerLayer,
    parameters: CompilationParameters,
    seqlen: usize,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling transformer layer...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled transformer layer [{f}]", .{now.untilNow(io, .awake)});

    const hidden: zml.Tensor = .init(.{ .s = seqlen, .d = layer.input_layernorm.weight.shape().dim(.d) }, layer.input_layernorm.weight.dtype());
    const layer_index: zml.Tensor = .init(.{}, .u32);
    const all_shardings = parameters.shardings.all();
    return platform.compile(
        allocator,
        io,
        LayerProgram{ .layer = layer },
        .forward,
        .{
            hidden,
            parameters.token_index,
            parameters.kv_cache,
            layer_index,
            parameters.attention_metadata,
            parameters.attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
}

fn compileHead(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    seqlen: usize,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling head...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled head [{f}]", .{now.untilNow(io, .awake)});

    const hidden: zml.Tensor = .init(
        .{
            .s = seqlen,
            .d = llama_model.model.norm.weight.shape().dim(.d),
        },
        llama_model.model.embed_tokens.weight.dtype(),
    );
    const all_shardings = parameters.shardings.all();
    return platform.compile(
        allocator,
        io,
        HeadProgram{
            .norm = llama_model.model.norm,
            .lm_head = llama_model.lm_head,
            .embed_tokens = llama_model.model.embed_tokens,
        },
        .forward,
        .{
            hidden,
            .init(.{ .s = seqlen }, .u32),
            parameters.rng,
            llama_model.gen_opts,
        },
        .{ .shardings = &all_shardings },
    );
}
