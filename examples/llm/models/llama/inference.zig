const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");
const weight_packing = @import("packed_weights.zig");

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
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const prefill = try compileKernel(allocator, io, platform, loaded_model.packing.weights, parameters, @intCast(parameters.prefill_tokens.dim(.s)), parameters.prefill_attention_parameters, .prefill, progress);
        errdefer prefill.deinit();
        const decode = try compileKernel(allocator, io, platform, loaded_model.packing.weights, parameters, @intCast(parameters.decode_tokens.dim(.s)), parameters.decode_attention_parameters, .decode, progress);

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

pub const Forward = struct {
    pub const Input = struct {
        weights: weight_packing.Weights,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
    };

    pub fn forward(input: Input) Output {
        const tokens, const kv_cache, const rng = input.weights.unpack().forward(
            input.tokens,
            input.token_index,
            input.kv_cache,
            input.rng,
            input.attention_metadata,
            input.attention_parameters,
        );
        return .{ .tokens = tokens, .kv_cache = kv_cache, .rng = rng };
    }
};

pub const KernelExe = zml.FnExe(Forward.forward);

pub const KernelRunner = struct {
    forward: KernelExe.Runner(.{.weights}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const weight_packing.Buffers) !KernelRunner {
        return .{ .forward = try .init(exe, allocator, .{ .weights = buffers.* }) };
    }

    pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
        self.forward.deinit(allocator);
    }
};

pub fn run(runner: *KernelRunner, args: Args) void {
    runner.forward.run(args.io, .{
        .inputs = .{
            .tokens = args.tokens_buf.*,
            .token_index = args.token_index_buf.*,
            .kv_cache = args.kv_cache_buffers.*,
            .rng = args.rng_buffers.*,
            .attention_metadata = args.attention_metadata_buffers.*,
        },
        .outputs = .{
            .tokens = args.tokens_buf,
            .kv_cache = args.kv_cache_buffers,
            .rng = args.rng_buffers,
        },
    });
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    weights: weight_packing.Weights,
    parameters: CompilationOptions,
    seqlen: usize,
    attention_parameters: zml.attention.Parameters,
    phase: Phase,
    progress: *std.Progress.Node,
) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("forward"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "forward", io, from);

    return KernelExe.compile(allocator, io, platform, .{
        .shardings = &parameters.shardings.all(),
        .program_name = phase.programName("llama", "forward"),
    }, .{.{
        .weights = weights,
        .tokens = zml.Tensor.init(.{ .s = seqlen }, .u32),
        .token_index = parameters.token_index,
        .kv_cache = parameters.kv_cache,
        .rng = parameters.rng,
        .attention_metadata = parameters.attention_metadata,
        .attention_parameters = attention_parameters,
    }});
}
