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
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(zml.attention.Metadata),
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
        llama_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill_exe = try compileKernel(allocator, io, platform, llama_model, parameters, @intCast(parameters.prefill_tokens.dim(.s)), parameters.prefill_attention_parameters, .prefill, progress);
        var prefill = try KernelExe.Runner.init(prefill_exe, allocator);
        errdefer prefill.deinit(allocator);
        const decode_exe = try compileKernel(allocator, io, platform, llama_model, parameters, @intCast(parameters.decode_tokens.dim(.s)), parameters.decode_attention_parameters, .decode, progress);

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
    .layer = model.TransformerLayer.forward,
    .sample = model.LmHead.forward,
});

pub fn run(runner: *KernelExe.Runner, args: Args, kv_cache_index_buffers: []const zml.Buffer) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.run(.embed, args.io, .{
        .inputs = .{
            .embedding = .{ .embed_tokens = args.model_buffers.model.embed_tokens },
            .tokens = args.tokens_buf.*,
        },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (args.model_buffers.model.layers, kv_cache_index_buffers) |layer_buffers, kv_cache_index_buffer| {
        runner.run(.layer, args.io, .{
            .inputs = .{
                .layer = layer_buffers,
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

    runner.run(.sample, args.io, .{
        .inputs = .{
            .lm_head = .{
                .lm_head = args.model_buffers.lm_head,
                .embed_tokens = args.model_buffers.model.embed_tokens,
                .norm = args.model_buffers.model.norm,
            },
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
    return .init(.{ .embed = embed, .layer = layer, .sample = sample });
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
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("embed_tokens"), 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "embed_tokens", io, from);

    const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

    return platform.compileFn(allocator, io, model.EmbedTokens.forward, .{.{
        .embedding = .{ .embed_tokens = embed_tokens },
        .tokens = tokens,
    }}, .{
        .shardings = &parameters.shardings.all(),
        .program_name = phase.programName("llama", "embed_tokens"),
    });
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
) !zml.Exe {
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

    return platform.compileFn(
        allocator,
        io,
        model.TransformerLayer.forward,
        .{.{
            .layer = llama_model.model.layers[0],
            .hidden = hidden,
            .token_index = parameters.token_index,
            .kv_cache = parameters.kv_cache,
            .kv_cache_index = kv_cache_index,
            .attention_metadata = parameters.attention_metadata,
            .attention_parameters = attention_parameters,
        }},
        .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("llama", "layer"),
        },
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
) !zml.Exe {
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

    return platform.compileFn(allocator, io, model.LmHead.forward, .{.{
        .lm_head = model.LmHead.init(llama_model),
        .hidden = hidden,
        .tokens = tokens,
        .rng = parameters.rng,
    }}, .{
        .shardings = &parameters.shardings.all(),
        .program_name = phase.programName("llama", "lm_head"),
    });
}
