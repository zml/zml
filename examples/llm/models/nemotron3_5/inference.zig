const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.nemotron3_5);

pub const CompilationParameters = struct {
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
    moe_backend: zml.moe.Backend,
    seqlen: u32,
    meshes: common.Meshes,
    xla_dump_to: ?[]const u8,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: zml.attention.Backend, moe_backend: zml.moe.Backend, meshes: common.Meshes) CompilationParameters {
        const dtype = mdl.backbone.embed_tokens.weight.dtype();
        return .{
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, meshes.model),
            .rng = .init(),
            .attention_metadata = .init(.fromBackend(backend)),
            .attention_parameters = .init(.fromBackend(backend)),
            .moe_backend = moe_backend,
            .seqlen = seqlen,
            .meshes = meshes,
            .xla_dump_to = "/home/ubuntu/xla_dump",
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const LayerIndexBuffer = union(enum) {
    self_attn: zml.Buffer,
    mamba: usize,
    moe: void,
};

pub const RunArgs = struct {
    io: std.Io,
    tokens_buffer: *zml.Buffer,
    next_token_buffer: *zml.Buffer,
    token_index_buffer: *zml.Buffer,
    active_length_buffer: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(zml.attention.Metadata),
    layer_index_buffers: []const LayerIndexBuffer,
    mamba_cache_buffers: []model.Mamba2.Cache.Buffers,
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
        mdl: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill = try compileKernel(allocator, io, platform, mdl, parameters, parameters.seqlen, "prefill", progress);
        errdefer prefill.deinit();
        const decode = try compileKernel(allocator, io, platform, mdl, parameters, 1, "decode", progress);
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
    full_attention: zml.FnExe(model.BackboneLayer.forwardSelfAttn),
    mamba: zml.FnExe(model.BackboneLayer.forwardMamba),
    moe: zml.FnExe(model.BackboneLayer.forwardMoe),
    sample: zml.FnExe(model.Sampler.sampleTokens),

    pub fn deinit(self: *const KernelExe) void {
        self.embed.deinit();
        self.full_attention.deinit();
        self.mamba.deinit();
        self.moe.deinit();
        self.sample.deinit();
    }
};

pub const LayerRunner = union(enum) {
    full_attention: zml.FnExe(model.BackboneLayer.forwardSelfAttn).Runner(.{.layer}),
    mamba: zml.FnExe(model.BackboneLayer.forwardMamba).Runner(.{.layer}),
    moe: zml.FnExe(model.BackboneLayer.forwardMoe).Runner(.{.layer}),

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
            .embedding = buffers.backbone.embed_tokens,
        });
        errdefer embed.deinit(allocator);

        const layers = try allocator.alloc(LayerRunner, buffers.backbone.layers.len);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        for (layers, buffers.backbone.layers) |*layer, layer_buffers| {
            layer.* = switch (layer_buffers.mixer) {
                .attn => .{
                    .full_attention = try zml.FnExe(model.BackboneLayer.forwardSelfAttn).Runner(.{.layer}).init(
                        &exe.full_attention,
                        allocator,
                        .{ .layer = layer_buffers },
                    ),
                },
                .mamba => .{
                    .mamba = try zml.FnExe(model.BackboneLayer.forwardMamba).Runner(.{.layer}).init(
                        &exe.mamba,
                        allocator,
                        .{ .layer = layer_buffers },
                    ),
                },
                .moe => .{
                    .moe = try zml.FnExe(model.BackboneLayer.forwardMoe).Runner(.{.layer}).init(
                        &exe.moe,
                        allocator,
                        .{ .layer = layer_buffers },
                    ),
                },
            };
            initialized_layers += 1;
        }

        var sample = try zml.FnExe(model.Sampler.sampleTokens).Runner(.{.sampler}).init(&exe.sample, allocator, .{
            .sampler = .{
                .norm = buffers.backbone.norm,
                .lm_head = buffers.backbone.lm_head,
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
    }, .{});
    defer hidden_buffer.deinit();

    for (runner.layers, args.layer_index_buffers) |*layer_runner, layer_index_buffer| {
        switch (layer_runner.*) {
            .full_attention => |*layer| {
                const index_buffer = switch (layer_index_buffer) {
                    .self_attn => |buffer| buffer,
                    .mamba, .moe => unreachable,
                };
                var layer_cache: zml.Bufferized(model.KvCache.SelfAttnCache) = .{
                    .k = args.kv_cache_buffers.self_attn.k,
                    .v = args.kv_cache_buffers.self_attn.v,
                    .layer_index = index_buffer,
                };
                layer.run(args.io, .{
                    .inputs = .{
                        .hidden = hidden_buffer,
                        .token_index = args.token_index_buffer.*,
                        .cache = layer_cache,
                        .attention_metadata = args.attention_metadata_buffers.*,
                    },
                    .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
                }, .{});
                args.kv_cache_buffers.self_attn.k = layer_cache.k;
                args.kv_cache_buffers.self_attn.v = layer_cache.v;
                layer_cache.layer_index.deinit();
            },
            .mamba => |*layer| {
                const index = switch (layer_index_buffer) {
                    .mamba => |index| index,
                    else => unreachable,
                };
                const cache = &args.mamba_cache_buffers[index];

                layer.run(args.io, .{
                    .inputs = .{
                        .hidden = hidden_buffer,
                        .cache = cache.*,
                        .active_length = args.active_length_buffer.*,
                    },
                    .outputs = .{
                        .hidden = &hidden_buffer,
                        .cache = cache,
                    },
                }, .{});
            },
            .moe => |*layer| {
                layer.run(args.io, .{
                    .inputs = .{ .hidden = hidden_buffer },
                    .outputs = .{ .hidden = &hidden_buffer },
                }, .{});
            },
        }
    }

    runner.sample.run(args.io, .{
        .inputs = .{
            .hidden = hidden_buffer,
            .tokens = args.tokens_buffer.*,
            .active_length = args.active_length_buffer.*,
            .rng = args.rng_buffers.*,
            .token_index = args.token_index_buffer.*,
        },
        .outputs = .{
            .tokens = args.next_token_buffer,
            .rng = args.rng_buffers,
            .token_index = args.token_index_buffer,
        },
    }, .{});
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    mdl: model.Model,
    parameters: CompilationParameters,
    seqlen: usize,
    phase: []const u8,
    progress: *std.Progress.Node,
) !KernelExe {
    const full_index = findFirstLayerIndex(mdl.backbone.layers, .attn) orelse return error.MissingFullAttentionLayer;
    const mamba_index = findFirstLayerIndex(mdl.backbone.layers, .mamba) orelse return error.MissingMambaLayer;
    const moe_index = findFirstLayerIndex(mdl.backbone.layers, .moe) orelse return error.MissingMoeLayer;

    var embed_future = try io.concurrent(compileEmbed, .{ allocator, io, platform, mdl, parameters, seqlen, phase, progress });
    errdefer if (embed_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var full_attention_future = try io.concurrent(compileFullAttention, .{ allocator, io, platform, mdl, parameters, seqlen, full_index, phase, progress });
    errdefer if (full_attention_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var mamba_future = try io.concurrent(compileMamba, .{ allocator, io, platform, mdl, parameters, seqlen, mamba_index, phase, progress });
    errdefer if (mamba_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var moe_future = try io.concurrent(compileMoe, .{ allocator, io, platform, mdl, parameters, seqlen, moe_index, phase, progress });
    errdefer if (moe_future.cancel(io)) |exe| exe.deinit() else |_| {};
    var sample_future = try io.concurrent(compileSample, .{ allocator, io, platform, mdl, parameters, seqlen, phase, progress });
    errdefer if (sample_future.cancel(io)) |exe| exe.deinit() else |_| {};

    const embed = try embed_future.await(io);
    errdefer embed.deinit();
    const full_attention = try full_attention_future.await(io);
    errdefer full_attention.deinit();
    const mamba = try mamba_future.await(io);
    errdefer mamba.deinit();
    const moe = try moe_future.await(io);
    errdefer moe.deinit();
    const sample = try sample_future.await(io);
    errdefer sample.deinit();

    return .{
        .embed = embed,
        .full_attention = full_attention,
        .mamba = mamba,
        .moe = moe,
        .sample = sample,
    };
}

fn compileEmbed(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.EmbedTokens.forward) {
    return compileExe(allocator, io, platform, model.EmbedTokens.forward, .{.{
        .embedding = mdl.backbone.embed_tokens,
        .tokens = zml.Tensor.init(.{ .b = 1, .s = seqlen }, .u32),
    }}, parameters, progress, phase, "embedding");
}

fn compileFullAttention(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, layer_index: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.BackboneLayer.forwardSelfAttn) {
    return compileExe(allocator, io, platform, model.BackboneLayer.forwardSelfAttn, .{.{
        .layer = mdl.backbone.layers[layer_index],
        .hidden = hiddenTensor(mdl, seqlen, parameters.meshes.model),
        .token_index = zml.Tensor.init(.{}, .u32),
        .cache = .{
            .k = parameters.kv_cache.self_attn.k,
            .v = parameters.kv_cache.self_attn.v,
            .layer_index = zml.Tensor.init(.{}, .u32),
        },
        .attention_metadata = parameters.attention_metadata,
        .attention_parameters = parameters.attention_parameters,
    }}, parameters, progress, phase, "full-attention layer");
}

fn compileMamba(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, layer_index: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.BackboneLayer.forwardMamba) {
    const hidden = hiddenTensor(mdl, seqlen, parameters.meshes.model);
    return compileExe(allocator, io, platform, model.BackboneLayer.forwardMamba, .{.{
        .layer = mdl.backbone.layers[layer_index],
        .hidden = hidden,
        .cache = model.Mamba2.Cache.init(mdl.config, hidden.dim(.b), hidden.dtype()),
        .active_length = zml.Tensor.init(.{}, .u32),
    }}, parameters, progress, phase, "mamba layer");
}

fn compileMoe(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, layer_index: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.BackboneLayer.forwardMoe) {
    return compileExe(allocator, io, platform, model.BackboneLayer.forwardMoe, .{.{
        .layer = mdl.backbone.layers[layer_index],
        .hidden = hiddenTensor(mdl, seqlen, parameters.meshes.model),
        .moe_backend = parameters.moe_backend,
    }}, parameters, progress, phase, "moe layer");
}

fn compileSample(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mdl: model.Model, parameters: CompilationParameters, seqlen: usize, phase: []const u8, progress: *std.Progress.Node) !zml.FnExe(model.Sampler.sampleTokens) {
    return compileExe(allocator, io, platform, model.Sampler.sampleTokens, .{.{
        .sampler = mdl.backbone.sampler(),
        .hidden = hiddenTensor(mdl, seqlen, parameters.meshes.model),
        .tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
        .active_length = .init(.{}, .u32),
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
        .meshes = &parameters.meshes.all(),
        .xla_dump_to = parameters.xla_dump_to,
    }, args);
}

fn hiddenTensor(mdl: model.Model, seqlen: usize, mesh: *const zml.Mesh) zml.Tensor {
    return .fromShape(zml.Shape.init(
        .{ .b = 1, .s = seqlen, .d = mdl.config.llm_config.hidden_size },
        mdl.backbone.embed_tokens.weight.dtype(),
    ).withPartitioning(mesh, .{ .b = .replicated, .s = .replicated, .d = .replicated }));
}

fn findFirstLayerIndex(layers: []const model.BackboneLayer, target: std.meta.Tag(@FieldType(model.BackboneLayer, "mixer"))) ?usize {
    for (layers, 0..) |layer, index| if (std.meta.activeTag(layer.mixer) == target) return index;
    return null;
}
