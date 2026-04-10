const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.llama);

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
    seqlen: usize,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend, shardings: common.Shardings) CompilationParameters {
        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(.init(.{
                .layer = mdl.model.layers.len,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
            }, mdl.model.embed_tokens.weight.dtype())),
            .rng = .init(),
            .attention_metadata = .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            .attention_parameters = .init(.fromBackend(backend)),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
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
        const compiled = try compileModel(allocator, io, platform, llama_model, parameters, progress);
        return .{ .loaded_model = loaded_model, .prefill_exe = compiled.prefill_exe, .decode_exe = compiled.decode_exe, .params = parameters };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill_exe.deinit();
        self.decode_exe.deinit();
    }
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    progress: *std.Progress.Node,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    const all_shardings = parameters.shardings.all();

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            llama_model_: model.Model,
            parameters_: CompilationParameters,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill...", 1);
            defer node_.end();

            return platform_.compile(
                allocator_,
                io_,
                llama_model_,
                .forward,
                .{
                    parameters_.prefill_tokens,
                    parameters_.token_index,
                    parameters_.kv_cache,
                    parameters_.rng,
                    parameters_.attention_metadata,
                    parameters_.attention_parameters,
                },
                .{ .shardings = &shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, llama_model, parameters, all_shardings, progress });
    var prefill_future_awaited = false;
    errdefer if (!prefill_future_awaited) if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            llama_model_: model.Model,
            parameters_: CompilationParameters,
            shardings_: [2]zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode...", 1);
            defer node_.end();

            return platform_.compile(
                allocator_,
                io_,
                llama_model_,
                .forward,
                .{
                    parameters_.decode_tokens,
                    parameters_.token_index,
                    parameters_.kv_cache,
                    parameters_.rng,
                    parameters_.attention_metadata,
                    parameters_.attention_parameters,
                },
                .{ .shardings = &shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, llama_model, parameters, all_shardings, progress });
    var decode_future_awaited = false;
    errdefer if (!decode_future_awaited) if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    prefill_future_awaited = true;
    errdefer prefill_exe.deinit();

    const decode_exe = try decode_future.await(io);
    decode_future_awaited = true;

    return .{
        .prefill_exe = prefill_exe,
        .decode_exe = decode_exe,
    };
}
