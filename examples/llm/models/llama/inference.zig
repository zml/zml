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
    meshes: common.Meshes,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: zml.attention.Backend, meshes: common.Meshes) CompilationParameters {
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
            }, mdl.model.embed_tokens.weight.dtype()), meshes.model),
            .rng = .init(),
            .attention_metadata = .init(.fromBackend(backend)),
            .prefill_attention_parameters = .init(.fromBackend(backend)),
            .decode_attention_parameters = .init(.fromBackend(backend)),
            .seqlen = seqlen,
            .meshes = meshes,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    io: std.Io,
    tokens_buf: *zml.Buffer,
    next_token_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    active_length_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(zml.attention.Metadata),
    // Furiosa decode must replace the donated position handle; prefill must not.
    advance_position: bool = false,
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

const Forward = struct {
    pub const Input = struct {
        weights: model.Model,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
        decode: bool,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
        token_index: ?zml.Tensor = null,
    };

    pub fn forward(input: Input) Output {
        var hidden = model.EmbedTokens.forward(.{
            .embedding = .{ .embed_tokens = input.weights.model.embed_tokens },
            .tokens = input.tokens,
        }).hidden;
        var kv_cache = input.kv_cache;
        for (input.weights.model.layers, 0..) |layer, index| {
            const result = model.TransformerLayer.forward(.{
                .layer = layer,
                .hidden = hidden,
                .token_index = input.token_index,
                .kv_cache = kv_cache,
                .kv_cache_index = .scalar(@as(u32, @intCast(index)), .u32),
                .attention_metadata = input.attention_metadata,
                .attention_parameters = input.attention_parameters,
            });
            hidden = result.hidden;
            kv_cache = result.kv_cache;
        }
        // LmHead owns final normalization; Llama.forward would apply it twice.
        const result = model.LmHead.forward(.{
            .lm_head = .init(input.weights),
            .hidden = hidden,
            .active_length = input.active_length,
            .rng = input.rng,
        });
        return .{
            .tokens = if (input.tokens.dim(.s) == 1) result.tokens.reuseBuffer(input.tokens) else result.tokens,
            .kv_cache = kv_cache.reuseBuffer(input.kv_cache),
            .rng = .{ ._state = result.rng._state.reuseBuffer(input.rng._state) },
            .token_index = if (zml.Compiler.current().platform.target == .furiosa and input.decode)
                @import("furiosa.zig").nextPosition(input.token_index)
            else
                null,
        };
    }
};

pub const KernelExe = zml.FnExe(Forward.forward);

pub const KernelRunner = struct {
    forward: KernelExe.Runner(.{.weights}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        return .{ .forward = try .init(exe, allocator, .{ .weights = buffers.* }) };
    }

    pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
        self.forward.deinit(allocator);
    }
};

pub fn run(runner: *KernelRunner, args: Args) void {
    var token_index: ?zml.Buffer = if (args.advance_position) args.token_index_buf.* else null;
    runner.forward.run(args.io, .{
        .inputs = .{
            .tokens = args.tokens_buf.*,
            .token_index = args.token_index_buf.*,
            .active_length = args.active_length_buf.*,
            .kv_cache = args.kv_cache_buffers.*,
            .rng = args.rng_buffers.*,
            .attention_metadata = args.attention_metadata_buffers.*,
        },
        .outputs = .{
            .tokens = args.next_token_buf,
            .kv_cache = args.kv_cache_buffers,
            .rng = args.rng_buffers,
            .token_index = &token_index,
        },
    }, .{});
    if (token_index) |buffer| args.token_index_buf.* = buffer;
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
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(phase.startMessage("forward"), 1);
    defer node.end();

    const from: std.Io.Timestamp = .now(io, .awake);
    defer phase.logCompileDone(log, "forward", io, from);

    return KernelExe.compile(allocator, io, platform, .{
        .meshes = &parameters.meshes.all(),
        .program_name = phase.programName("llama", "forward"),
    }, .{.{
        .weights = llama_model,
        .tokens = zml.Tensor.init(.{ .s = seqlen }, .u32),
        .token_index = parameters.token_index,
        .active_length = .init(.{}, .u32),
        .kv_cache = parameters.kv_cache,
        .rng = parameters.rng,
        .attention_metadata = parameters.attention_metadata,
        .attention_parameters = attention_parameters,
        .decode = phase == .decode,
    }});
}
