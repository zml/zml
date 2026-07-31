const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const Phase = common.Phase;
const model = @import("model.zig");

const log = std.log.scoped(.lfm);

pub const CompilationParameters = struct {
    hidden_dim: usize,
    batch_dim: usize,
    rng: zml.Tensor.Rng,
    cache: model.Cache,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: zml.attention.Backend, shardings: common.Shardings) CompilationParameters {
        stdx.debug.assert(seqlen >= config.conv_L_cache, "seqlen ({}) must be at least conv_L_cache ({})", .{ seqlen, config.conv_L_cache });
        const cache: model.Cache = .{
            .kv = .init(.init(.{
                .layer = mdl.num_attention_layers,
                .batch = 1,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = config.hidden_size / config.num_attention_heads,
            }, mdl.embed_tokens.weight.dtype())),
            .conv = .init(.init(.{
                .layer = mdl.num_conv_layers,
                .batch = 1,
                .seq = config.conv_L_cache,
                .d = config.hidden_size,
            }, mdl.embed_tokens.weight.dtype())),
        };

        return .{
            .hidden_dim = config.hidden_size,
            .batch_dim = 1,
            .rng = .init(),
            .cache = cache,
            .attention_metadata = .init(.fromBackend(backend, seqlen, config.num_attention_heads)),
            .attention_parameters = .init(.fromBackend(backend)),
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
    tokens_pos_buf: *zml.Buffer,
    actual_seq_len_buf: *zml.Buffer,
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    cache_buffers: *zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata),
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
        platform: *zml.Platform,
        loaded_model: *const model.LoadedModel,
        mdl: model.Model,
        opts: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill_exe = try compileKernel(allocator, io, platform, mdl, opts, opts.seqlen, .prefill, progress);
        var prefill = try KernelExe.Runner.init(prefill_exe, allocator);
        errdefer prefill.deinit(allocator);
        const decode_exe = try compileKernel(allocator, io, platform, mdl, opts, 1, .decode, progress);
        return .{
            .allocator = allocator,
            .loaded_model = loaded_model,
            .prefill = prefill,
            .decode = try KernelExe.Runner.init(decode_exe, allocator),
            .params = opts,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit(self.allocator);
        self.decode.deinit(self.allocator);
    }
};

pub const Inference = CompiledModel;

pub const KernelExe = zml.TypedMultiExe(.{
    .embed = model.TokenEmbedding.forward,
    .conv = model.DecoderLayer.forward,
    .self_attn = model.DecoderLayer.forward,
    .sample = model.LmHead.forward,
});

pub fn run(
    runner: *KernelExe.Runner,
    args: Args,
    conv_cache_index_buffer: *zml.Buffer,
    kv_cache_index_buffer: *zml.Buffer,
) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.run(.embed, args.io, .{
        .inputs = .{
            .embedding = args.model_buffers.embed_tokens,
            .tokens = args.tokens_buf.*,
        },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (args.model_buffers.layers) |layer_buffers| {
        switch (layer_buffers.operator) {
            inline else => |_, operator| runner.run(@field(KernelExe.Field, @tagName(operator)), args.io, .{
                .inputs = .{
                    .layer = layer_buffers,
                    .hidden = hidden_buffer,
                    .tokens_position_offset = args.tokens_pos_buf.*,
                    .actual_seq_len = args.actual_seq_len_buf.*,
                    .cache = args.cache_buffers.*,
                    .conv_cache_index = conv_cache_index_buffer.*,
                    .kv_cache_index = kv_cache_index_buffer.*,
                    .attention_metadata = args.attention_metadata_buffers,
                },
                .outputs = .{
                    .hidden = &hidden_buffer,
                    .cache = args.cache_buffers,
                    .conv_cache_index = conv_cache_index_buffer,
                    .kv_cache_index = kv_cache_index_buffer,
                },
            }),
        }
    }

    runner.run(.sample, args.io, .{
        .inputs = .{
            .lm_head = args.model_buffers.lm_head,
            .embed_tokens = args.model_buffers.embed_tokens,
            .hidden = hidden_buffer,
            .tokens = args.tokens_buf.*,
            .rng = args.rng_buf.*,
        },
        .outputs = .{
            .tokens = args.tokens_buf,
            .rng = args.rng_buf,
        },
    });
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    phase: Phase,
    progress: *std.Progress.Node,
) !KernelExe {
    const embed = try compileEmbed(allocator, io, platform, mdl.embed_tokens, opts, seqlen, phase, progress);
    errdefer embed.deinit();
    const conv_layer = try compileLayer(allocator, io, platform, mdl, opts, seqlen, .conv, phase, progress);
    errdefer conv_layer.deinit();
    const attn_layer = try compileLayer(allocator, io, platform, mdl, opts, seqlen, .full_attention, phase, progress);
    errdefer attn_layer.deinit();
    const sample = try compileSample(allocator, io, platform, mdl, opts, seqlen, phase, progress);
    errdefer sample.deinit();
    return .init(.{ .embed = embed, .conv = conv_layer, .self_attn = attn_layer, .sample = sample });
}

fn compileEmbed(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    embed_tokens: model.TokenEmbedding,
    opts: CompilationOptions,
    seqlen: u32,
    phase: Phase,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("embed_tokens"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "embed_tokens", io, from);

    return platform.compileFn(allocator, io, model.TokenEmbedding.forward, .{.{
        .embedding = embed_tokens,
        .tokens = zml.Tensor.init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32),
    }}, .{
        .shardings = &opts.shardings.all(),
        .program_name = phase.programName("lfm2", "embed_tokens"),
    });
}

fn compileLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    comptime kind: model.OperatorKind,
    phase: Phase,
    progress: *std.Progress.Node,
) !zml.Exe {
    const label = switch (kind) {
        .conv => "conv layer",
        .full_attention => "attn layer",
    };
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage(label), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, label, io, from);

    const layer = for (mdl.layers) |candidate| {
        const candidate_kind: model.OperatorKind = switch (candidate.operator) {
            .conv => .conv,
            .self_attn => .full_attention,
        };
        if (candidate_kind == kind) break candidate;
    } else unreachable;

    return platform.compileFn(allocator, io, model.DecoderLayer.forward, .{.{
        .layer = layer,
        .hidden = zml.Tensor.init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype()),
        .tokens_position_offset = zml.Tensor.init(.{ .batch = opts.batch_dim }, .u32),
        .actual_seq_len = zml.Tensor.init(.{}, .u32),
        .cache = opts.cache,
        .conv_cache_index = zml.Tensor.init(.{}, .u32),
        .kv_cache_index = zml.Tensor.init(.{}, .u32),
        .attention_metadata = opts.attention_metadata,
        .attention_parameters = opts.attention_parameters,
        .conv_parameters = .{ .is_prefill = phase.isPrefill() },
    }}, .{
        .shardings = &opts.shardings.all(),
        .program_name = phase.programName("lfm2", if (kind == .conv) "conv_layer" else "attn_layer"),
    });
}

fn compileSample(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    phase: Phase,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("lm_head"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "lm_head", io, from);

    return platform.compileFn(allocator, io, model.LmHead.forward, .{.{
        .lm_head = mdl.lm_head,
        .embed_tokens = mdl.embed_tokens,
        .hidden = zml.Tensor.init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype()),
        .tokens = zml.Tensor.init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32),
        .rng = opts.rng,
    }}, .{
        .shardings = &opts.shardings.all(),
        .program_name = phase.programName("lfm2", "lm_head"),
    });
}
