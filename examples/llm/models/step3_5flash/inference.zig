const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");
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
    xla_dump_to: ?[]const u8,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, moe_backend: zml.moe.Backend, shardings: common.Shardings) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
        return .{
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, model_partitions),
            .rng = .init(),
            .moe_parameters = .init(.fromBackend(moe_backend, config.text_config.num_experts_per_tok, zml.moe.triton.Parameters.ActivationMode.silu)),
            .seqlen = seqlen,
            .shardings = shardings,
            .xla_dump_to = "/home/ubuntu/xla_dump",
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    decode_embedding_exe: zml.Exe,
    decode_full_layer_exe: ?zml.Exe,
    decode_sampling_exe: zml.Exe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        step3p5_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compile_result = try compileModel(allocator, io, platform, step3p5_model, parameters, progress);
        return .{
            .loaded_model = loaded_model,
            .decode_embedding_exe = compile_result.decode_embedding_exe,
            .decode_full_layer_exe = compile_result.decode_full_layer_exe,
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

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    step3p5_model: model.Model,
    parameters: CompilationParameters,
    progress: *std.Progress.Node,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    const all_shardings = parameters.shardings.all();
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    const decode_tokens = zml.Tensor.init(.{ .b = 1, .s = 1 }, .u32);
    const hidden_dtype = step3p5_model.text_model.embed_tokens.weight.dtype();

    const decode_hidden: zml.Tensor = .fromShape(zml.Shape.init(
        .{ .b = 1, .s = 1, .d = step3p5_model.config.text_config.hidden_size },
        hidden_dtype,
    ).withPartitioning(.{
        .b = .replicated,
        .s = .replicated,
        .d = .replicated,
    }));
    const token_index = zml.Tensor.init(.{}, .u32);
    const self_attn_cache: model.KvCache.SelfAttnCache = .{
        .k = parameters.kv_cache.self_attn.k,
        .v = parameters.kv_cache.self_attn.v,
        .layer_index = zml.Tensor.init(.{}, .u32),
    };

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
            return platform_.compile(allocator_, io_, model_, .forward, .{decode_tokens_}, .{ .shardings = shardings_ });
        }
    }.call, .{ allocator, io, platform, step3p5_model.text_model.embed_tokens, decode_tokens, &all_shardings, progress });
    errdefer if (decode_embedding_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const decode_embedding_exe = try decode_embedding_future.await(io);

    const full_layer_index = 0;
    const full_layer_model = step3p5_model.text_model.layers[full_layer_index];

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
            shardings_: []const zml.Sharding,
            xla_dump_to_: ?[]const u8,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            return compileSelfAttnLayerExe(allocator_, io_, platform_, layer_model_, hidden_, token_index_, cache_, config_, moe_metadata_, moe_parameters_, shardings_, xla_dump_to_, progress_, "decode full-attention layer...");
        }
    }.call, .{ allocator, io, platform, full_layer_model, decode_hidden, token_index, self_attn_cache, step3p5_model.config, parameters.decode_moe_metadata, parameters.moe_parameters, &all_shardings, parameters.xla_dump_to, progress });
    errdefer if (decode_full_layer_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_full_layer_exe: ?zml.Exe = null;
    errdefer if (decode_full_layer_exe) |exe| exe.deinit();
    decode_full_layer_exe = try decode_full_layer_future.await(io);

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
    }.call, .{ allocator, io, platform, step3p5_model.text_model.sampler(), decode_hidden, parameters.rng, token_index, &all_shardings, progress });
    errdefer if (decode_sampling_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const decode_sampling_exe = try decode_sampling_future.await(io);

    return .{
        .decode_embedding_exe = decode_embedding_exe,
        .decode_full_layer_exe = decode_full_layer_exe,
        .decode_sampling_exe = decode_sampling_exe,
    };
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
