const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5);
const Phase = common.Phase;

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
        return .{
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, shardings.model),
            .rng = .init(),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    io: std.Io,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
};

pub const CompiledModel = struct {
    allocator: std.mem.Allocator,
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe.Runner,
    decode: KernelExe.Runner,
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
        const prefill_exe = try compileKernel(allocator, io, platform, qwen_model, parameters, @intCast(parameters.prefill_tokens.dim(.s)), .prefill, progress);
        var prefill = try KernelExe.Runner.init(prefill_exe, allocator);
        errdefer prefill.deinit(allocator);
        const decode_exe = try compileKernel(allocator, io, platform, qwen_model, parameters, @intCast(parameters.decode_tokens.dim(.s)), .decode, progress);
        return .{
            .allocator = allocator,
            .loaded_model = loaded_model,
            .prefill = prefill,
            .decode = try KernelExe.Runner.init(decode_exe, allocator),
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit(self.allocator);
        self.decode.deinit(self.allocator);
    }
};

pub const Inference = CompiledModel;

pub const KernelExe = zml.TypedMultiExe(.{
    .embed = model.EmbedTokens.forward,
    .full_attention = model.TransformerLayer.forwardSelfAttn,
    .linear_attention = model.TransformerLayer.forwardLinearAttn,
    .sample = model.Sampler.sampleTokens,
});

pub fn run(runner: *KernelExe.Runner, args: Args, layer_index_buffers: []const zml.Buffer) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.run(.embed, args.io, .{
        .inputs = .{
            .embedding = .{ .embed_tokens = args.model_buffers.text_model.embed_tokens },
            .tokens = args.tokens_buf.*,
        },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (args.model_buffers.text_model.layers, 0..) |layer_buffers, i| {
        switch (layer_buffers.attn) {
            .full_attention => {
                const layer_index_buffer = layer_index_buffers[i];
                var layer_cache: zml.Bufferized(model.KvCache.SelfAttnCache) = .{
                    .k = args.kv_cache_buffers.self_attn.k,
                    .v = args.kv_cache_buffers.self_attn.v,
                    .layer_index = layer_index_buffer,
                };
                runner.run(.full_attention, args.io, .{
                    .inputs = .{
                        .layer = layer_buffers,
                        .hidden = hidden_buffer,
                        .token_index = args.token_index_buf.*,
                        .cache = layer_cache,
                    },
                    .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
                });
                args.kv_cache_buffers.self_attn.k = layer_cache.k;
                args.kv_cache_buffers.self_attn.v = layer_cache.v;
            },
            .linear_attention => {
                const layer_index_buffer = layer_index_buffers[i];
                var layer_cache: zml.Bufferized(model.KvCache.GatedDeltaNetCache) = .{
                    .conv_state = args.kv_cache_buffers.gated_delta_net.conv_state,
                    .recurrent_state = args.kv_cache_buffers.gated_delta_net.recurrent_state,
                    .layer_index = layer_index_buffer,
                };
                runner.run(.linear_attention, args.io, .{
                    .inputs = .{
                        .layer = layer_buffers,
                        .hidden = hidden_buffer,
                        .token_index = args.token_index_buf.*,
                        .cache = layer_cache,
                    },
                    .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
                });
                args.kv_cache_buffers.gated_delta_net.conv_state = layer_cache.conv_state;
                args.kv_cache_buffers.gated_delta_net.recurrent_state = layer_cache.recurrent_state;
            },
        }
    }

    runner.run(.sample, args.io, .{
        .inputs = .{
            .sampler = .{
                .norm = args.model_buffers.text_model.norm,
                .lm_head = args.model_buffers.lm_head,
            },
            .hidden = hidden_buffer,
            .rng = args.rng_buffers.*,
            .token_index = args.token_index_buf.*,
        },
        .outputs = .{
            .tokens = args.tokens_buf,
            .rng = args.rng_buffers,
            .token_index = args.token_index_buf,
        },
    });
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationOptions,
    seqlen: usize,
    phase: Phase,
    progress: *std.Progress.Node,
) !KernelExe {
    const full_index = findFirstLayerIndex(qwen_model.config.text_config.layer_types, .full_attention) orelse return error.MissingFullAttentionLayer;
    const linear_index = findFirstLayerIndex(qwen_model.config.text_config.layer_types, .linear_attention) orelse return error.MissingLinearAttentionLayer;

    const embed = try compileEmbed(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
    errdefer embed.deinit();
    const full_attention = try compileFullAttention(allocator, io, platform, qwen_model, parameters, seqlen, full_index, phase, progress);
    errdefer full_attention.deinit();
    const linear_attention = try compileLinearAttention(allocator, io, platform, qwen_model, parameters, seqlen, linear_index, phase, progress);
    errdefer linear_attention.deinit();
    const sample = try compileSample(allocator, io, platform, qwen_model, parameters, seqlen, phase, progress);
    errdefer sample.deinit();
    return .init(.{ .embed = embed, .full_attention = full_attention, .linear_attention = linear_attention, .sample = sample });
}

fn compileEmbed(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, phase: Phase, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("embed tokens"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "embed tokens", io, from);
    return platform.compileFn(allocator, io, model.EmbedTokens.forward, .{.{
        .embedding = .{ .embed_tokens = mdl.text_model.embed_tokens },
        .tokens = zml.Tensor.init(.{ .b = 1, .s = seqlen }, .u32),
    }}, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3_5", "embed_tokens") });
}

fn compileFullAttention(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, layer_index: usize, phase: Phase, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("full attention layer"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "full attention layer", io, from);
    return platform.compileFn(allocator, io, model.TransformerLayer.forwardSelfAttn, .{.{
        .layer = mdl.text_model.layers[layer_index],
        .hidden = hiddenTensor(mdl, seqlen),
        .token_index = parameters.token_index,
        .cache = .{
            .k = parameters.kv_cache.self_attn.k,
            .v = parameters.kv_cache.self_attn.v,
            .layer_index = zml.Tensor.init(.{}, .u32),
        },
    }}, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3_5", "full_attention_layer") });
}

fn compileLinearAttention(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, layer_index: usize, phase: Phase, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("linear attention layer"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "linear attention layer", io, from);
    return platform.compileFn(allocator, io, model.TransformerLayer.forwardLinearAttn, .{.{
        .layer = mdl.text_model.layers[layer_index],
        .hidden = hiddenTensor(mdl, seqlen),
        .token_index = parameters.token_index,
        .cache = .{
            .conv_state = parameters.kv_cache.gated_delta_net.conv_state,
            .recurrent_state = parameters.kv_cache.gated_delta_net.recurrent_state,
            .layer_index = zml.Tensor.init(.{}, .u32),
        },
    }}, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3_5", "linear_attention_layer") });
}

fn compileSample(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationOptions, seqlen: usize, phase: Phase, progress: *std.Progress.Node) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("sampler"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "sampler", io, from);
    return platform.compileFn(allocator, io, model.Sampler.sampleTokens, .{.{
        .sampler = mdl.sampler(),
        .hidden = hiddenTensor(mdl, seqlen),
        .rng = parameters.rng,
        .token_index = parameters.token_index,
    }}, .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName("qwen3_5", "sampler") });
}

fn hiddenTensor(mdl: model.Model, seqlen: usize) zml.Tensor {
    return .fromShape(zml.Shape.init(
        .{ .b = 1, .s = seqlen, .d = mdl.config.text_config.hidden_size },
        mdl.text_model.embed_tokens.weight.dtype(),
    ).withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated }));
}

fn findFirstLayerIndex(layer_types: []const model.LayerType, target: model.LayerType) ?usize {
    for (layer_types, 0..) |layer_type, index| if (layer_type == target) return index;
    return null;
}
