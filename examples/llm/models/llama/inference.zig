const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.llama);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: zml.attention.Metadata,
    prefill_attention_parameters: zml.attention.Parameters,
    decode_attention_parameters: zml.attention.Parameters,
    seqlen: usize,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: zml.attention.Backend, shardings: common.Shardings) CompilationParameters {
        const head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads);

        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(.init(.{
                .layer = mdl.model.layers.len,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = head_dim,
            }, mdl.model.embed_tokens.weight.dtype())),
            .rng = .init(),
            .attention_metadata = switch (backend) {
                .attnd => .{ .attnd = .init() },
                else => .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            },
            .prefill_attention_parameters = switch (backend) {
                .attnd => .{ .attnd = .init(.{
                    .model_id = .@"llama-3.1-8B",
                    .head_dim = head_dim,
                    .num_attention_heads = config.num_attention_heads,
                    .num_kv_heads = @intCast(config.num_key_value_heads),
                    .is_prefill = true,
                }) },
                else => .init(.fromBackend(backend)),
            },
            .decode_attention_parameters = switch (backend) {
                .attnd => .{ .attnd = .init(.{
                    .model_id = .@"llama-3.1-8B",
                    .head_dim = head_dim,
                    .num_attention_heads = config.num_attention_heads,
                    .num_kv_heads = @intCast(config.num_key_value_heads),
                    .is_prefill = false,
                }) },
                else => .init(.fromBackend(backend)),
            },
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    io: std.Io,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(zml.attention.Metadata),
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
        llama_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill = try compileKernel(allocator, io, platform, llama_model, parameters, @intCast(parameters.prefill_tokens.dim(.s)), parameters.prefill_attention_parameters, .prefill, progress);
        errdefer prefill.deinit();
        const decode = try compileKernel(allocator, io, platform, llama_model, parameters, @intCast(parameters.decode_tokens.dim(.s)), parameters.decode_attention_parameters, .decode, progress);

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

pub const EmbedExe = zml.TypedExe(model.EmbedTokens.forward);
pub const LayerExe = zml.TypedExe(model.TransformerLayer.forward);
pub const SampleExe = zml.TypedExe(model.LmHead.forward);

pub const KernelExe = struct {
    embed: EmbedExe,
    layer: LayerExe,
    sample: SampleExe,

    pub fn deinit(self: *const KernelExe) void {
        self.embed.deinit();
        self.layer.deinit();
        self.sample.deinit();
    }
};

pub const KernelRunner = struct {
    embed: EmbedExe.Runner(.{.embedding}),
    layers: []LayerExe.Runner(.{.layer}),
    sample: SampleExe.Runner(.{.lm_head}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        var embed = try EmbedExe.Runner(.{.embedding}).init(&exe.embed, allocator, .{
            .embedding = .{ .embed_tokens = buffers.model.embed_tokens },
        });
        errdefer embed.deinit(allocator);

        const layers = try allocator.alloc(LayerExe.Runner(.{.layer}), buffers.model.layers.len);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        for (layers, buffers.model.layers) |*layer, layer_buffers| {
            layer.* = try LayerExe.Runner(.{.layer}).init(&exe.layer, allocator, .{ .layer = layer_buffers });
            initialized_layers += 1;
        }

        var sample = try SampleExe.Runner(.{.lm_head}).init(&exe.sample, allocator, .{
            .lm_head = .{
                .lm_head = buffers.lm_head,
                .embed_tokens = buffers.model.embed_tokens,
                .norm = buffers.model.norm,
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

pub fn run(runner: *KernelRunner, args: Args, kv_cache_index_buffers: []const zml.Buffer) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.embed.run(args.io, .{
        .inputs = .{
            .tokens = args.tokens_buf.*,
        },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (runner.layers, kv_cache_index_buffers) |*layer, kv_cache_index_buffer| {
        layer.run(args.io, .{
            .inputs = .{
                .hidden = hidden_buffer,
                .token_index = args.token_index_buf.*,
                .kv_cache = args.kv_cache_buffers.*,
                .kv_cache_index = kv_cache_index_buffer,
                .attention_metadata = args.attention_metadata_buffers.*,
            },
            .outputs = .{
                .hidden = &hidden_buffer,
                .kv_cache = args.kv_cache_buffers,
            },
        });
    }

    runner.sample.run(args.io, .{
        .inputs = .{
            .hidden = hidden_buffer,
            .tokens = args.tokens_buf.*,
            .rng = args.rng_buffers.*,
        },
        .outputs = .{
            .tokens = args.tokens_buf,
            .rng = args.rng_buffers,
        },
    });
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationOptions,
    seqlen: usize,
    attention_parameters: zml.attention.Parameters,
    phase: Phase,
    progress: *std.Progress.Node,
) !KernelExe {
    const embed = try compileEmbed(allocator, io, platform, llama_model.model.embed_tokens, parameters, seqlen, phase, progress);
    errdefer embed.deinit();
    const layer = try compileLayer(allocator, io, platform, llama_model, parameters, seqlen, attention_parameters, phase, progress);
    errdefer layer.deinit();
    const sample = try compileSample(allocator, io, platform, llama_model, parameters, seqlen, phase, progress);
    errdefer sample.deinit();
    return .{ .embed = embed, .layer = layer, .sample = sample };
}

fn compileEmbed(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    embed_tokens: zml.nn.TokenEmbedding,
    parameters: CompilationOptions,
    seqlen: usize,
    phase: Phase,
    progress: *std.Progress.Node,
) !EmbedExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("embed_tokens"), 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "embed_tokens", io, from);

    const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

    return EmbedExe.compile(allocator, io, platform, .{
        .shardings = &parameters.shardings.all(),
        .program_name = phase.programName("llama", "embed_tokens"),
    }, .{.{
        .embedding = .{ .embed_tokens = embed_tokens },
        .tokens = tokens,
    }});
}

fn compileLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationOptions,
    seqlen: usize,
    attention_parameters: zml.attention.Parameters,
    phase: Phase,
    progress: *std.Progress.Node,
) !LayerExe {
    progress.increaseEstimatedTotalItems(1);

    var node = progress.start(phase.startMessage("transformer layer"), 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "transformer layer", io, from);

    const hidden: zml.Tensor = .fromShape(zml.Shape.init(
        .{ .s = seqlen, .d = llama_model.config.hidden_size },
        llama_model.model.embed_tokens.weight.dtype(),
    ).withPartitioning(.{ .d = .replicated }));

    const kv_cache_index: zml.Tensor = .init(.{}, .u32);

    return LayerExe.compile(
        allocator,
        io,
        platform,
        .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("llama", "layer"),
        },
        .{.{
            .layer = llama_model.model.layers[0],
            .hidden = hidden,
            .token_index = parameters.token_index,
            .kv_cache = parameters.kv_cache,
            .kv_cache_index = kv_cache_index,
            .attention_metadata = parameters.attention_metadata,
            .attention_parameters = attention_parameters,
        }},
    );
}

fn compileSample(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationOptions,
    seqlen: usize,
    phase: Phase,
    progress: *std.Progress.Node,
) !SampleExe {
    progress.increaseEstimatedTotalItems(1);

    var node = progress.start(phase.startMessage("lm_head"), 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "lm_head", io, from);

    const hidden: zml.Tensor = .fromShape(zml.Shape.init(
        .{ .s = seqlen, .d = llama_model.config.hidden_size },
        llama_model.model.embed_tokens.weight.dtype(),
    ).withPartitioning(.{ .d = .replicated }));

    const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

    return SampleExe.compile(allocator, io, platform, .{
        .shardings = &parameters.shardings.all(),
        .program_name = phase.programName("llama", "lm_head"),
    }, .{.{
        .lm_head = model.LmHead.init(llama_model),
        .hidden = hidden,
        .tokens = tokens,
        .rng = parameters.rng,
    }});
}
