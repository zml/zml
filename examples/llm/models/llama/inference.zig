const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const Shardings = @import("../common.zig").Shardings;
const model = @import("model.zig");

const log = std.log.scoped(.llama);

pub const CompilationOptions = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend) CompilationOptions {
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
        };
    }
};

pub const Inference = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        llama_model: model.Model,
        parameters: CompilationOptions,
        shardings: Shardings,
        progress: *std.Progress.Node,
    ) !Inference {
        const compiled = try compileModel(allocator, io, platform, llama_model, parameters, shardings, progress);
        return .{ .prefill_exe = compiled.prefill_exe, .decode_exe = compiled.decode_exe };
    }

    pub fn deinit(self: *Inference) void {
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
    parameters: CompilationOptions,
    shardings: Shardings,
    progress: *std.Progress.Node,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    const all_shardings = shardings.all();

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            llama_model_: model.Model,
            parameters_: CompilationOptions,
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
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            llama_model_: model.Model,
            parameters_: CompilationOptions,
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
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    return .{
        .prefill_exe = try prefill_future.await(io),
        .decode_exe = try decode_future.await(io),
    };
}
