const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5_moe);

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
        return .{
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, shardings.model),
            .rng = .init(),
            .prefill_moe_metadata = initMoeMetadata(mdl, @intCast(seqlen), 1, moe_backend),
            .decode_moe_metadata = initMoeMetadata(mdl, 1, 1, moe_backend),
            .moe_parameters = .init(.fromBackend(moe_backend, config.text_config.num_experts_per_tok, zml.moe.ActivationMode.silu)),
            .seqlen = seqlen,
            .shardings = shardings,
            .xla_dump_to = "/home/ubuntu/xla_dump",
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const LayerIndexBuffer = union(enum) {
    self_attn: zml.Buffer,
    linear_attn: zml.Buffer,
};

pub const RunArgs = struct {
    io: std.Io,
    tokens_buffer: *zml.Buffer,
    full_attention_token_index_buffer: *zml.Buffer,
    linear_attention_token_index_buffer: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    layer_index_buffers: []const LayerIndexBuffer,
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe,
    decode: KernelExe,
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
        const prefill = try compileKernel(allocator, io, platform, qwen_model, parameters, parameters.seqlen, parameters.prefill_moe_metadata, "prefill", progress);
        errdefer prefill.deinit();
        const decode = try compileKernel(allocator, io, platform, qwen_model, parameters, 1, parameters.decode_moe_metadata, "decode", progress);
        return .{
            .loaded_model = loaded_model,
            .prefill = prefill,
            .decode = decode,
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

pub const KernelExe = struct {
    embed: zml.FnExe(model.EmbedTokens.forward),
    full_attention: zml.FnExe(model.TransformerLayer.forwardSelfAttn),
    linear_attention: zml.FnExe(model.TransformerLayer.forwardLinearAttn),
    sample: zml.FnExe(model.Sampler.sampleTokens),

    pub fn deinit(self: *const KernelExe) void {
        self.embed.deinit();
        self.full_attention.deinit();
        self.linear_attention.deinit();
        self.sample.deinit();
    }
};

pub const LayerRunner = union(enum) {
    full_attention: zml.FnExe(model.TransformerLayer.forwardSelfAttn).Runner(.{.layer}),
    linear_attention: zml.FnExe(model.TransformerLayer.forwardLinearAttn).Runner(.{.layer}),

    pub fn deinit(self: *LayerRunner, allocator: std.mem.Allocator) void {
        switch (self.*) {
            inline else => |*runner| runner.deinit(allocator),
        }
    }
};

pub const KernelRunner = struct {
    embed: zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}),
    layers: []LayerRunner,
    sample: zml.FnExe(model.Sampler.sampleTokens).Runner(.{.sampler}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        var embed = try zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}).init(&exe.embed, allocator, .{
            .embedding = buffers.text_model.embed_tokens,
        });
        errdefer embed.deinit(allocator);

        const layers = try allocator.alloc(LayerRunner, buffers.text_model.layers.len);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        for (layers, buffers.text_model.layers) |*layer, layer_buffers| {
            layer.* = switch (layer_buffers.attn) {
                .self_attn => .{ .full_attention = try zml.FnExe(model.TransformerLayer.forwardSelfAttn).Runner(.{.layer}).init(
                    &exe.full_attention,
                    allocator,
                    .{ .layer = layer_buffers },
                ) },
                .linear_attn => .{ .linear_attention = try zml.FnExe(model.TransformerLayer.forwardLinearAttn).Runner(.{.layer}).init(
                    &exe.linear_attention,
                    allocator,
                    .{ .layer = layer_buffers },
                ) },
            };
            initialized_layers += 1;
        }

        var sample = try zml.FnExe(model.Sampler.sampleTokens).Runner(.{.sampler}).init(&exe.sample, allocator, .{
            .sampler = .{
                .norm = buffers.text_model.norm,
                .lm_head = buffers.text_model.lm_head,
            },
        });
        errdefer sample.deinit(allocator);

        return .{ .embed = embed, .layers = layers, .sample = sample };
    }

    pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
        self.embed.deinit(allocator);
        for (self.layers) |*layer| layer.deinit(allocator);
        allocator.free(self.layers);
        self.sample.deinit(allocator);
    }
};

pub fn run(runner: *KernelRunner, args: RunArgs) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.embed.run(args.io, .{
        .inputs = .{
            .tokens = args.tokens_buffer.*,
        },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (runner.layers, args.layer_index_buffers) |*layer_runner, layer_index_buffer| {
        switch (layer_runner.*) {
            .full_attention => |*layer| {
                const index_buffer = switch (layer_index_buffer) {
                    .self_attn => |buffer| buffer,
                    .linear_attn => unreachable,
                };
                var layer_cache: zml.Bufferized(model.KvCache.SelfAttnCache) = .{
                    .k = args.kv_cache_buffers.self_attn.k,
                    .v = args.kv_cache_buffers.self_attn.v,
                    .layer_index = index_buffer,
                };
                layer.run(args.io, .{
                    .inputs = .{
                        .hidden = hidden_buffer,
                        .token_index = args.full_attention_token_index_buffer.*,
                        .cache = layer_cache,
                        .moe_metadata = args.moe_metadata_buffers,
                    },
                    .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
                });
                args.kv_cache_buffers.self_attn.k = layer_cache.k;
                args.kv_cache_buffers.self_attn.v = layer_cache.v;
            },
            .linear_attention => |*layer| {
                const index_buffer = switch (layer_index_buffer) {
                    .linear_attn => |buffer| buffer,
                    .self_attn => unreachable,
                };
                var layer_cache: zml.Bufferized(model.KvCache.GatedDeltaNetCache) = .{
                    .conv_state = args.kv_cache_buffers.gated_delta_net.conv_state,
                    .recurrent_state = args.kv_cache_buffers.gated_delta_net.recurrent_state,
                    .layer_index = index_buffer,
                };
                layer.run(args.io, .{
                    .inputs = .{
                        .hidden = hidden_buffer,
                        .token_index = args.linear_attention_token_index_buffer.*,
                        .cache = layer_cache,
                        .moe_metadata = args.moe_metadata_buffers,
                    },
                    .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
                });
                args.kv_cache_buffers.gated_delta_net.conv_state = layer_cache.conv_state;
                args.kv_cache_buffers.gated_delta_net.recurrent_state = layer_cache.recurrent_state;
            },
        }
    }

    runner.sample.run(args.io, .{
        .inputs = .{
            .hidden = hidden_buffer,
            .rng = args.rng_buffers.*,
            .token_index = args.full_attention_token_index_buffer.*,
        },
        .outputs = .{
            .tokens = args.tokens_buffer,
            .rng = args.rng_buffers,
            .token_index = args.full_attention_token_index_buffer,
        },
    });
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: model.Model,
    parameters: CompilationParameters,
    seqlen: usize,
    moe_metadata: zml.moe.Metadata,
    phase: []const u8,
    progress: *std.Progress.Node,
) !KernelExe {
    const full_index = findFirstLayerIndex(mdl.config.text_config.layer_types, .full_attention) orelse return error.MissingFullAttentionLayer;
    const linear_index = findFirstLayerIndex(mdl.config.text_config.layer_types, .linear_attention) orelse return error.MissingLinearAttentionLayer;

    var embed_future = try io.concurrent(compileEmbed, .{ allocator, io, platform, mdl, parameters, seqlen, phase, progress });
    errdefer if (embed_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var full_attention_future = try io.concurrent(compileFullAttention, .{ allocator, io, platform, mdl, parameters, seqlen, full_index, moe_metadata, phase, progress });
    errdefer if (full_attention_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var linear_attention_future = try io.concurrent(compileLinearAttention, .{ allocator, io, platform, mdl, parameters, seqlen, linear_index, moe_metadata, phase, progress });
    errdefer if (linear_attention_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var sample_future = try io.concurrent(compileSample, .{ allocator, io, platform, mdl, parameters, seqlen, phase, progress });
    errdefer if (sample_future.cancel(io)) |exe| exe.deinit() else |_| {};

    const embed = try embed_future.await(io);
    errdefer embed.deinit();
    const full_attention = try full_attention_future.await(io);
    errdefer full_attention.deinit();
    const linear_attention = try linear_attention_future.await(io);
    errdefer linear_attention.deinit();
    const sample = try sample_future.await(io);
    errdefer sample.deinit();

    return .{
        .embed = embed,
        .full_attention = full_attention,
        .linear_attention = linear_attention,
        .sample = sample,
    };
}

fn compileEmbed(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.EmbedTokens.forward) {
    return compileExe(allocator, io, platform, model.EmbedTokens.forward, .{.{
        .embedding = mdl.text_model.embed_tokens,
        .tokens = zml.Tensor.init(.{ .b = 1, .s = seqlen }, .u32),
    }}, parameters, progress, phase, "embedding");
}

fn compileFullAttention(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, layer_index: usize, moe_metadata: zml.moe.Metadata, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.TransformerLayer.forwardSelfAttn) {
    return compileExe(allocator, io, platform, model.TransformerLayer.forwardSelfAttn, .{.{
        .layer = mdl.text_model.layers[layer_index],
        .hidden = hiddenTensor(mdl, seqlen),
        .token_index = zml.Tensor.init(.{}, .u32),
        .cache = .{
            .k = parameters.kv_cache.self_attn.k,
            .v = parameters.kv_cache.self_attn.v,
            .layer_index = zml.Tensor.init(.{}, .u32),
        },
        .config = mdl.config,
        .moe_metadata = moe_metadata,
        .moe_parameters = parameters.moe_parameters,
    }}, parameters, progress, phase, "full-attention layer");
}

fn compileLinearAttention(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, layer_index: usize, moe_metadata: zml.moe.Metadata, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.TransformerLayer.forwardLinearAttn) {
    return compileExe(allocator, io, platform, model.TransformerLayer.forwardLinearAttn, .{.{
        .layer = mdl.text_model.layers[layer_index],
        .hidden = hiddenTensor(mdl, seqlen),
        .token_index = zml.Tensor.init(.{}, .u32),
        .cache = .{
            .conv_state = parameters.kv_cache.gated_delta_net.conv_state,
            .recurrent_state = parameters.kv_cache.gated_delta_net.recurrent_state,
            .layer_index = zml.Tensor.init(.{}, .u32),
        },
        .config = mdl.config,
        .moe_metadata = moe_metadata,
        .moe_parameters = parameters.moe_parameters,
    }}, parameters, progress, phase, "linear-attention layer");
}

fn compileSample(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.Sampler.sampleTokens) {
    return compileExe(allocator, io, platform, model.Sampler.sampleTokens, .{.{
        .sampler = mdl.text_model.sampler(),
        .hidden = hiddenTensor(mdl, seqlen),
        .rng = parameters.rng,
        .token_index = zml.Tensor.init(.{}, .u32),
    }}, parameters, progress, phase, "sampling");
}

fn compileExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    comptime function: anytype,
    args: std.meta.ArgsTuple(@TypeOf(function)),
    parameters: CompilationParameters,
    progress: *std.Progress.Node,
    phase: []const u8,
    component: []const u8,
) !zml.FnExe(function) {
    progress.increaseEstimatedTotalItems(1);
    const label = try std.fmt.allocPrint(allocator, "Compiling {s} {s}...", .{ phase, component });
    defer allocator.free(label);
    var node = progress.start(label, 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} {s} [{f}]", .{ phase, component, now.untilNow(io, .awake) });
    return zml.FnExe(function).compile(allocator, io, platform, .{
        .shardings = &parameters.shardings.all(),
        .xla_dump_to = parameters.xla_dump_to,
    }, args);
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
        .mosaic_tpu, .metal => .init(.fromBackend(backend)),
    };
}
