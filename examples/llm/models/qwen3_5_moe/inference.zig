const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5_moe);

const ShardedEmbedding = struct {
    model: zml.nn.TokenEmbedding,

    pub fn forward(self: ShardedEmbedding, tokens: zml.Tensor) zml.Tensor {
        return self.model.forward(tokens).withPartitioning(.{
            .b = .replicated,
            .s = .replicated,
            .d = .replicated,
        });
    }
};

const CompileModelResult = struct {
    prefill_self_attn_exe: ?zml.Exe,
    prefill_linear_attn_exe: ?zml.Exe,
    prefill_embedding_exe: zml.Exe,
    prefill_sampling_exe: zml.Exe,
    decode_self_attn_exe: ?zml.Exe,
    decode_linear_attn_exe: ?zml.Exe,
    decode_embedding_exe: zml.Exe,
    decode_sampling_exe: zml.Exe,
};

fn makeDumpSubdir(allocator: std.mem.Allocator, io: std.Io, base: ?[]const u8, subdir: []const u8) !?[]const u8 {
    const base_ = base orelse return null;
    const full = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_, subdir });
    errdefer allocator.free(full);
    try std.Io.Dir.cwd().createDirPath(io, full);
    return full;
}

pub const CompilationParameters = struct {
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    prefill_moe_metadata: zml.moe.Metadata,
    decode_moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,
    seqlen: u32,
    shardings: common.Shardings,
    xla_dump_to: ?[]const u8,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, moe_backend: zml.moe.Backend, shardings: common.Shardings) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);

        return .{
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, model_partitions),
            .rng = .init(),
            .prefill_moe_metadata = initMoeMetadata(mdl, @intCast(seqlen), 1, moe_backend),
            .decode_moe_metadata = initMoeMetadata(mdl, 1, 1, moe_backend),
            .moe_parameters = .init(.fromBackend(moe_backend, config.text_config.num_experts_per_tok, zml.moe.triton.Parameters.ActivationMode.silu)),
            .seqlen = seqlen,
            .shardings = shardings,
            .xla_dump_to = "/tmp/xla_hlo",
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill_self_attn_exe: ?zml.Exe,
    prefill_linear_attn_exe: ?zml.Exe,
    prefill_embedding_exe: zml.Exe,
    prefill_sampling_exe: zml.Exe,
    decode_self_attn_exe: ?zml.Exe,
    decode_linear_attn_exe: ?zml.Exe,
    decode_embedding_exe: zml.Exe,
    decode_sampling_exe: zml.Exe,
    params: CompilationParameters,
    allocator: std.mem.Allocator,

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
            .prefill_self_attn_exe = compile_result.prefill_self_attn_exe,
            .prefill_linear_attn_exe = compile_result.prefill_linear_attn_exe,
            .prefill_embedding_exe = compile_result.prefill_embedding_exe,
            .prefill_sampling_exe = compile_result.prefill_sampling_exe,
            .decode_self_attn_exe = compile_result.decode_self_attn_exe,
            .decode_linear_attn_exe = compile_result.decode_linear_attn_exe,
            .decode_embedding_exe = compile_result.decode_embedding_exe,
            .decode_sampling_exe = compile_result.decode_sampling_exe,
            .params = parameters,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill_embedding_exe.deinit();
        if (self.prefill_self_attn_exe) |*exe| exe.deinit();
        if (self.prefill_linear_attn_exe) |*exe| exe.deinit();
        self.prefill_sampling_exe.deinit();
        self.decode_embedding_exe.deinit();
        if (self.decode_self_attn_exe) |*exe| exe.deinit();
        if (self.decode_linear_attn_exe) |*exe| exe.deinit();
        self.decode_sampling_exe.deinit();
    }

    pub fn duplicateExe(
        self: *const CompiledModel,
        allocator: std.mem.Allocator,
        platform: *const zml.Platform,
    ) !CompiledModel {
        const dup_prefill_self_attn_exe = if (self.prefill_self_attn_exe) |exe| try cloneExe(allocator, platform, exe) else null;
        errdefer if (dup_prefill_self_attn_exe) |*exe| exe.deinit();
        const dup_prefill_linear_attn_exe = if (self.prefill_linear_attn_exe) |exe| try cloneExe(allocator, platform, exe) else null;
        errdefer if (dup_prefill_linear_attn_exe) |*exe| exe.deinit();
        const dup_decode_self_attn_exe = if (self.decode_self_attn_exe) |exe| try cloneExe(allocator, platform, exe) else null;
        errdefer if (dup_decode_self_attn_exe) |*exe| exe.deinit();
        const dup_decode_linear_attn_exe = if (self.decode_linear_attn_exe) |exe| try cloneExe(allocator, platform, exe) else null;
        errdefer if (dup_decode_linear_attn_exe) |*exe| exe.deinit();

        return .{
            .loaded_model = self.loaded_model,
            .prefill_self_attn_exe = dup_prefill_self_attn_exe,
            .prefill_linear_attn_exe = dup_prefill_linear_attn_exe,
            .prefill_embedding_exe = try cloneExe(allocator, platform, self.prefill_embedding_exe),
            .prefill_sampling_exe = try cloneExe(allocator, platform, self.prefill_sampling_exe),
            .decode_self_attn_exe = dup_decode_self_attn_exe,
            .decode_linear_attn_exe = dup_decode_linear_attn_exe,
            .decode_embedding_exe = try cloneExe(allocator, platform, self.decode_embedding_exe),
            .decode_sampling_exe = try cloneExe(allocator, platform, self.decode_sampling_exe),
            .params = self.params,
            .allocator = allocator,
        };
    }
};

pub const Inference = CompiledModel;

fn cloneExe(allocator: std.mem.Allocator, platform: *const zml.Platform, exe: zml.Exe) !zml.Exe {
    const pjrt_exe = try exe.exe.executable(exe.platform.pjrt_api);
    var result = try pjrt_exe.serialize(exe.platform.pjrt_api);
    defer result.deinit();

    const exec_clone = try platform.pjrt_client.deserializeAndLoad(platform.pjrt_api, result.bytes);

    return zml.Exe.init(
        allocator,
        platform,
        exec_clone,
        exe.num_devices,
        exe.num_partitions,
        exe.input_shapes,
        exe.output_shapes,
        exe.input_shardings,
        exe.output_shardings,
    );
}

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
    shardings: []const zml.Sharding,
    xla_dump_to: ?[]const u8,
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

    return platform.compile(allocator, io, mdl, .forwardSelfAttn, .{ hidden, token_index, cache, config, moe_metadata, moe_parameters }, .{
        .shardings = shardings,
        .xla_dump_to = xla_dump_to,
    });
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
    shardings: []const zml.Sharding,
    xla_dump_to: ?[]const u8,
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

    return platform.compile(allocator, io, mdl, .forwardLinearAttn, .{ hidden, token_index, cache, config, moe_metadata, moe_parameters }, .{
        .shardings = shardings,
        .xla_dump_to = xla_dump_to,
    });
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
    const hidden_dtype = qwen35_model.text_model.embed_tokens.weight.dtype();
    const prefill_hidden: zml.Tensor = .fromShape(zml.Shape.init(
        .{ .b = 1, .s = prefill_len, .d = qwen35_model.config.text_config.hidden_size },
        hidden_dtype,
    ).withPartitioning(.{
        .b = .replicated,
        .s = .replicated,
        .d = .replicated,
    }));
    const token_index = zml.Tensor.init(.{}, .u32);
    const decode_hidden: zml.Tensor = .fromShape(zml.Shape.init(
        .{ .b = 1, .s = 1, .d = qwen35_model.config.text_config.hidden_size },
        hidden_dtype,
    ).withPartitioning(.{
        .b = .replicated,
        .s = .replicated,
        .d = .replicated,
    }));
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
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill embedding...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill embedding [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, ShardedEmbedding{ .model = model_ }, .forward, .{prefill_tokens_}, .{ .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.embed_tokens, prefill_tokens, &all_shardings, progress });
    errdefer if (prefill_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const prefill_embedding_exe = try prefill_embedding_future.await(io);

    var prefill_sampling_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            sampler_: model.Sampler,
            prefill_hidden_: zml.Tensor,
            rng_: zml.Tensor.Rng,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill sampling...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill sampling [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, sampler_, .sampleTokens, .{ prefill_hidden_, rng_, null }, .{ .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.sampler(), prefill_hidden, parameters.rng, &all_shardings, progress });
    errdefer if (prefill_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};
    const prefill_sampling_exe = try prefill_sampling_future.await(io);

    var decode_embedding_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            model_: zml.nn.TokenEmbedding,
            decode_tokens_: zml.Tensor,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode embedding...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode embedding [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, ShardedEmbedding{ .model = model_ }, .forward, .{decode_tokens_}, .{ .shardings = shardings_ });
        }
    }.call, .{
        allocator,
        io,
        platform,
        qwen35_model.text_model.embed_tokens,
        decode_tokens,
        &all_shardings,
        progress,
    });
    errdefer if (decode_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};
    const decode_embedding_exe = try decode_embedding_future.await(io);

    var decode_sampling_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            sampler_: model.Sampler,
            decode_hidden_: zml.Tensor,
            rng_: zml.Tensor.Rng,
            token_index_: zml.Tensor,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode sampling...", 1);
            defer node_.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode sampling [{f}]", .{now_.untilNow(io_, .awake)});
            return platform_.compile(allocator_, io_, sampler_, .sampleTokens, .{ decode_hidden_, rng_, token_index_ }, .{ .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, qwen35_model.text_model.sampler(), decode_hidden, parameters.rng, token_index, &all_shardings, progress });
    errdefer if (decode_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};
    const decode_sampling_exe = try decode_sampling_future.await(io);

    // Compile one executable per layer type and reuse it across matching layers.
    var prefill_self_attn_exe: ?zml.Exe = null;
    var prefill_linear_attn_exe: ?zml.Exe = null;
    var decode_self_attn_exe: ?zml.Exe = null;
    var decode_linear_attn_exe: ?zml.Exe = null;
    errdefer if (prefill_self_attn_exe) |*exe| exe.deinit();
    errdefer if (prefill_linear_attn_exe) |*exe| exe.deinit();
    errdefer if (decode_self_attn_exe) |*exe| exe.deinit();
    errdefer if (decode_linear_attn_exe) |*exe| exe.deinit();

    for (0..@as(usize, @intCast(qwen35_model.config.text_config.num_hidden_layers))) |layer_index| {
        const layer_model = qwen35_model.text_model.layers[layer_index];
        const layer_type = qwen35_model.config.text_config.layer_types[layer_index];

        switch (layer_type) {
            .full_attention => {
                if (prefill_self_attn_exe == null) {
                    prefill_self_attn_exe = try compileSelfAttnLayerExe(
                        allocator,
                        io,
                        platform,
                        layer_model,
                        prefill_hidden,
                        token_index,
                        self_attn_cache,
                        qwen35_model.config,
                        parameters.prefill_moe_metadata,
                        parameters.moe_parameters,
                        &all_shardings,
                        parameters.xla_dump_to,
                        progress,
                        "prefill full-attention (template)",
                    );
                }
                if (decode_self_attn_exe == null) {
                    decode_self_attn_exe = try compileSelfAttnLayerExe(
                        allocator,
                        io,
                        platform,
                        layer_model,
                        decode_hidden,
                        token_index,
                        self_attn_cache,
                        qwen35_model.config,
                        parameters.decode_moe_metadata,
                        parameters.moe_parameters,
                        &all_shardings,
                        parameters.xla_dump_to,
                        progress,
                        "decode full-attention (template)",
                    );
                }
            },
            .linear_attention => {
                if (prefill_linear_attn_exe == null) {
                    prefill_linear_attn_exe = try compileLinearAttnLayerExe(
                        allocator,
                        io,
                        platform,
                        layer_model,
                        prefill_hidden,
                        token_index,
                        linear_attn_cache,
                        qwen35_model.config,
                        parameters.prefill_moe_metadata,
                        parameters.moe_parameters,
                        &all_shardings,
                        parameters.xla_dump_to,
                        progress,
                        "prefill linear-attention (template)",
                    );
                }
                if (decode_linear_attn_exe == null) {
                    decode_linear_attn_exe = try compileLinearAttnLayerExe(
                        allocator,
                        io,
                        platform,
                        layer_model,
                        decode_hidden,
                        token_index,
                        linear_attn_cache,
                        qwen35_model.config,
                        parameters.decode_moe_metadata,
                        parameters.moe_parameters,
                        &all_shardings,
                        parameters.xla_dump_to,
                        progress,
                        "decode linear-attention (template)",
                    );
                }
            },
        }
    }

    return .{
        .prefill_self_attn_exe = prefill_self_attn_exe,
        .prefill_linear_attn_exe = prefill_linear_attn_exe,
        .prefill_embedding_exe = prefill_embedding_exe,
        .prefill_sampling_exe = prefill_sampling_exe,
        .decode_self_attn_exe = decode_self_attn_exe,
        .decode_linear_attn_exe = decode_linear_attn_exe,
        .decode_embedding_exe = decode_embedding_exe,
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
