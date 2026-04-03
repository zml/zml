const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5_moe);

const CompileModelResult = struct {
    prefill_embedding_exe: zml.Exe,
    decode_embedding_exe: zml.Exe,
    prefill_full_layer_exe: ?zml.Exe,
    decode_full_layer_exe: ?zml.Exe,
    prefill_linear_layer_exe: ?zml.Exe,
    decode_linear_layer_exe: ?zml.Exe,
    prefill_sampling_exe: zml.Exe,
    decode_sampling_exe: zml.Exe,
};

pub const CompilationParameters = struct {
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    prefill_moe_metadata: zml.moe.Metadata,
    decode_moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, moe_backend: zml.moe.Backend, shardings: common.Shardings) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        return .{
            .kv_cache = .init(config, 1, seqlen, dtype, .f32),
            .rng = .init(),
            .prefill_moe_metadata = initMoeMetadata(mdl, @intCast(seqlen), 1, moe_backend),
            .decode_moe_metadata = initMoeMetadata(mdl, 1, 1, moe_backend),
            .moe_parameters = .init(.fromBackend(moe_backend, config.text_config.num_experts_per_tok)),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill_embedding_exe: zml.Exe,
    decode_embedding_exe: zml.Exe,
    prefill_full_layer_exe: ?zml.Exe,
    decode_full_layer_exe: ?zml.Exe,
    prefill_linear_layer_exe: ?zml.Exe,
    decode_linear_layer_exe: ?zml.Exe,
    prefill_sampling_exe: zml.Exe,
    decode_sampling_exe: zml.Exe,
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
        return .{
            .loaded_model = loaded_model,
            .prefill_embedding_exe = compile_result.prefill_embedding_exe,
            .decode_embedding_exe = compile_result.decode_embedding_exe,
            .prefill_full_layer_exe = compile_result.prefill_full_layer_exe,
            .decode_full_layer_exe = compile_result.decode_full_layer_exe,
            .prefill_linear_layer_exe = compile_result.prefill_linear_layer_exe,
            .decode_linear_layer_exe = compile_result.decode_linear_layer_exe,
            .prefill_sampling_exe = compile_result.prefill_sampling_exe,
            .decode_sampling_exe = compile_result.decode_sampling_exe,
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill_embedding_exe.deinit();
        self.decode_embedding_exe.deinit();
        if (self.prefill_full_layer_exe) |exe| exe.deinit();
        if (self.decode_full_layer_exe) |exe| exe.deinit();
        if (self.prefill_linear_layer_exe) |exe| exe.deinit();
        if (self.decode_linear_layer_exe) |exe| exe.deinit();
        self.prefill_sampling_exe.deinit();
        self.decode_sampling_exe.deinit();
    }
};

pub const Inference = CompiledModel;

fn compileSelfAttnLayerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: model.TransformerLayer,
    hidden: zml.Tensor,
    token_index: zml.Tensor,
    cache: model.KvCache.SelfAttnCache,
    config: model.Config,
    moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
    shardings: [2]zml.sharding.Sharding,
    progress: *std.Progress.Node,
    label: []const u8,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    const compiling_label = try std.fmt.allocPrint(allocator, "Compiling {s}...", .{label});
    defer allocator.free(compiling_label);
    var node = progress.start(compiling_label, 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} [{f}]", .{ label, now.untilNow(io, .awake) });

    return platform.compile(allocator, io, mdl, .forwardSelfAttn, .{ hidden, token_index, cache, config, moe_metadata, moe_parameters }, .{ .shardings = &shardings });
}

fn compileLinearAttnLayerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: model.TransformerLayer,
    hidden: zml.Tensor,
    token_index: zml.Tensor,
    cache: model.KvCache.GatedDeltaNetCache,
    config: model.Config,
    moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
    shardings: [2]zml.sharding.Sharding,
    progress: *std.Progress.Node,
    label: []const u8,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    const compiling_label = try std.fmt.allocPrint(allocator, "Compiling {s}...", .{label});
    defer allocator.free(compiling_label);
    var node = progress.start(compiling_label, 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} [{f}]", .{ label, now.untilNow(io, .awake) });

    return platform.compile(allocator, io, mdl, .forwardLinearAttn, .{ hidden, token_index, cache, config, moe_metadata, moe_parameters }, .{ .shardings = &shardings });
}

fn findFirstLayerIndex(layer_types: []const model.LayerType, target: model.LayerType) ?usize {
    for (layer_types, 0..) |layer_type, index| {
        if (layer_type == target) return index;
    }
    return null;
}

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen35_model: model.Model,
    parameters: CompilationParameters,
    progress: *std.Progress.Node,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    const all_shardings = parameters.shardings.all();
    const prefill_len: usize = @intCast(parameters.seqlen);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});
    log.info("Compiling model for platform {any} with prefill length {d}...", .{ platform.target, prefill_len });

    const prefill_tokens = zml.Tensor.init(.{ .b = 1, .s = prefill_len }, .u32);
    const decode_tokens = zml.Tensor.init(.{ .b = 1, .s = 1 }, .u32);
    const prefill_hidden = zml.Tensor.init(.{ .b = 1, .s = prefill_len, .d = qwen35_model.config.text_config.hidden_size }, qwen35_model.text_model.embed_tokens.weight.dtype());
    const decode_hidden = zml.Tensor.init(.{ .b = 1, .s = 1, .d = qwen35_model.config.text_config.hidden_size }, qwen35_model.text_model.embed_tokens.weight.dtype());
    const token_index = zml.Tensor.init(.{}, .u32);
    const self_attn_cache: model.KvCache.SelfAttnCache = .{
        .k = parameters.kv_cache.self_attn.k,
        .v = parameters.kv_cache.self_attn.v,
        .layer_index = zml.Tensor.init(.{}, .u32),
    };
    const linear_attn_cache: model.KvCache.GatedDeltaNetCache = .{
        .conv_state = parameters.kv_cache.gated_delta_net.conv_state,
        .recurrent_state = parameters.kv_cache.gated_delta_net.recurrent_state,
        .layer_index = zml.Tensor.init(.{}, .u32),
    };

    var prefill_embedding_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: zml.nn.TokenEmbedding,
            prefill_tokens_: zml.Tensor,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill embedding...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill embedding [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .forward, .{prefill_tokens_}, .{ .shardings = &shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.embed_tokens, prefill_tokens, all_shardings, progress });
    errdefer if (prefill_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    var decode_embedding_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: zml.nn.TokenEmbedding,
            decode_tokens_: zml.Tensor,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode embedding...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode embedding [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .forward, .{decode_tokens_}, .{ .shardings = &shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.embed_tokens, decode_tokens, all_shardings, progress });
    errdefer if (decode_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const prefill_embedding_exe = try prefill_embedding_future.await(io);
    const decode_embedding_exe = try decode_embedding_future.await(io);

    const full_layer_index = findFirstLayerIndex(qwen35_model.config.text_config.layer_types, .full_attention) orelse return error.MissingFullAttentionLayer;
    const linear_layer_index = findFirstLayerIndex(qwen35_model.config.text_config.layer_types, .linear_attention) orelse return error.MissingLinearAttentionLayer;
    const full_layer_model = qwen35_model.text_model.layers[full_layer_index];
    const linear_layer_model = qwen35_model.text_model.layers[linear_layer_index];

    var prefill_full_layer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            layer_model_: model.TransformerLayer,
            hidden_: zml.Tensor,
            token_index_: zml.Tensor,
            cache_: model.KvCache.SelfAttnCache,
            config_: model.Config,
            moe_metadata_: zml.moe.Metadata,
            moe_parameters_: zml.moe.Parameters,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            return compileSelfAttnLayerExe(allocator_, io_, platform_, layer_model_, hidden_, token_index_, cache_, config_, moe_metadata_, moe_parameters_, shardings_, progress_, "prefill full-attention layer...");
        }
    }.call, .{ allocator, io, platform, full_layer_model, prefill_hidden, token_index, self_attn_cache, qwen35_model.config, parameters.prefill_moe_metadata, parameters.moe_parameters, all_shardings, progress });
    errdefer if (prefill_full_layer_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_full_layer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            layer_model_: model.TransformerLayer,
            hidden_: zml.Tensor,
            token_index_: zml.Tensor,
            cache_: model.KvCache.SelfAttnCache,
            config_: model.Config,
            moe_metadata_: zml.moe.Metadata,
            moe_parameters_: zml.moe.Parameters,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            return compileSelfAttnLayerExe(allocator_, io_, platform_, layer_model_, hidden_, token_index_, cache_, config_, moe_metadata_, moe_parameters_, shardings_, progress_, "decode full-attention layer...");
        }
    }.call, .{ allocator, io, platform, full_layer_model, decode_hidden, token_index, self_attn_cache, qwen35_model.config, parameters.decode_moe_metadata, parameters.moe_parameters, all_shardings, progress });
    errdefer if (decode_full_layer_future.cancel(io)) |v| v.deinit() else |_| {};

    var prefill_full_layer_exe: ?zml.Exe = null;
    errdefer if (prefill_full_layer_exe) |exe| exe.deinit();
    var decode_full_layer_exe: ?zml.Exe = null;
    errdefer if (decode_full_layer_exe) |exe| exe.deinit();
    prefill_full_layer_exe = try prefill_full_layer_future.await(io);
    decode_full_layer_exe = try decode_full_layer_future.await(io);

    var prefill_linear_layer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            layer_model_: model.TransformerLayer,
            hidden_: zml.Tensor,
            token_index_: zml.Tensor,
            cache_: model.KvCache.GatedDeltaNetCache,
            config_: model.Config,
            moe_metadata_: zml.moe.Metadata,
            moe_parameters_: zml.moe.Parameters,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            return compileLinearAttnLayerExe(allocator_, io_, platform_, layer_model_, hidden_, token_index_, cache_, config_, moe_metadata_, moe_parameters_, shardings_, progress_, "prefill linear-attention layer...");
        }
    }.call, .{ allocator, io, platform, linear_layer_model, prefill_hidden, token_index, linear_attn_cache, qwen35_model.config, parameters.prefill_moe_metadata, parameters.moe_parameters, all_shardings, progress });
    errdefer if (prefill_linear_layer_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_linear_layer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            layer_model_: model.TransformerLayer,
            hidden_: zml.Tensor,
            token_index_: zml.Tensor,
            cache_: model.KvCache.GatedDeltaNetCache,
            config_: model.Config,
            moe_metadata_: zml.moe.Metadata,
            moe_parameters_: zml.moe.Parameters,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            return compileLinearAttnLayerExe(allocator_, io_, platform_, layer_model_, hidden_, token_index_, cache_, config_, moe_metadata_, moe_parameters_, shardings_, progress_, "decode linear-attention layer...");
        }
    }.call, .{ allocator, io, platform, linear_layer_model, decode_hidden, token_index, linear_attn_cache, qwen35_model.config, parameters.decode_moe_metadata, parameters.moe_parameters, all_shardings, progress });
    errdefer if (decode_linear_layer_future.cancel(io)) |v| v.deinit() else |_| {};

    var prefill_linear_layer_exe: ?zml.Exe = null;
    errdefer if (prefill_linear_layer_exe) |exe| exe.deinit();
    var decode_linear_layer_exe: ?zml.Exe = null;
    errdefer if (decode_linear_layer_exe) |exe| exe.deinit();

    prefill_linear_layer_exe = try prefill_linear_layer_future.await(io);
    decode_linear_layer_exe = try decode_linear_layer_future.await(io);

    var prefill_sampling_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: model.TextModel,
            prefill_hidden_: zml.Tensor,
            rng_: zml.Tensor.Rng,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill sampling...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill sampling [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .sampleTokens, .{ prefill_hidden_, rng_, null }, .{ .shardings = &shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model, prefill_hidden, parameters.rng, all_shardings, progress });
    errdefer if (prefill_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};
    const prefill_sampling_exe = try prefill_sampling_future.await(io);

    var decode_sampling_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: model.TextModel,
            decode_hidden_: zml.Tensor,
            rng_: zml.Tensor.Rng,
            token_index_: zml.Tensor,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode sampling...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode sampling [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, model_, .sampleTokens, .{ decode_hidden_, rng_, token_index_ }, .{ .shardings = &shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model, decode_hidden, parameters.rng, token_index, all_shardings, progress });
    errdefer if (decode_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const decode_sampling_exe = try decode_sampling_future.await(io);

    return .{
        .prefill_embedding_exe = prefill_embedding_exe,
        .decode_embedding_exe = decode_embedding_exe,
        .prefill_full_layer_exe = prefill_full_layer_exe,
        .decode_full_layer_exe = decode_full_layer_exe,
        .prefill_linear_layer_exe = prefill_linear_layer_exe,
        .decode_linear_layer_exe = decode_linear_layer_exe,
        .prefill_sampling_exe = prefill_sampling_exe,
        .decode_sampling_exe = decode_sampling_exe,
    };
}

fn initMoeMetadata(qwen_model: model.Model, token_len: usize, batch_size: u32, backend: zml.moe.Backend) zml.moe.Metadata {
    if (qwen_model.config.text_config.num_experts_per_tok == null) {
        return .init(.fromBackend(backend));
    }

    var w1_zero_bias_shape: ?zml.Shape = null;
    var w2_zero_bias_shape: ?zml.Shape = null;
    var first_out_shape: ?zml.Shape = null;
    var second_out_shape: ?zml.Shape = null;

    const num_experts_per_tok = qwen_model.config.text_config.num_experts_per_tok.?;
    const num_experts = qwen_model.config.text_config.num_experts.?;

    for (qwen_model.text_model.layers) |layer| {
        const gate_up_shape = zml.Shape.init(.{
            .expert = num_experts,
            .out = layer.moe.shared_expert.gate_proj.weight.dim(.dout),
        }, .bf16);
        const down_shape = zml.Shape.init(.{
            .expert = num_experts,
            .out = layer.moe.shared_expert.gate_proj.weight.dim(.d),
        }, .bf16);
        const first_out = zml.Shape.init(.{
            .total_tokens = batch_size * token_len * num_experts_per_tok,
            .out = layer.moe.shared_expert.gate_proj.weight.dim(.dout) * 2,
        }, .bf16);
        const second_out = zml.Shape.init(.{
            .token = batch_size * token_len,
            .topk = num_experts_per_tok,
            .out = layer.moe.shared_expert.down_proj.weight.dim(.d),
        }, .bf16);

        if (w1_zero_bias_shape == null) {
            w1_zero_bias_shape = gate_up_shape;
            w2_zero_bias_shape = down_shape;
            first_out_shape = first_out;
            second_out_shape = second_out;
            continue;
        }

        if (!w1_zero_bias_shape.?.eql(gate_up_shape) or !w2_zero_bias_shape.?.eql(down_shape) or !first_out_shape.?.eql(first_out) or !second_out_shape.?.eql(second_out)) {
            log.warn("MoE bias shapes differ across layers; using shapes from the first layer", .{});
            break;
        }
    }

    return switch (backend) {
        .triton => .init(.{
            .triton = .{
                .w1_zero_bias_shape = w1_zero_bias_shape,
                .w2_zero_bias_shape = w2_zero_bias_shape,
            },
        }),
    };
}
