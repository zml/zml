const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5);

const CompileModelResult = struct {
    prefill_exe: ?zml.Exe = null,
    decode_exe: ?zml.Exe = null,
};

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
        const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
        return .{
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, 1, seqlen, dtype, .f32, model_partitions),
            .rng = .init(),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

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
        qwen_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compile_result = try compileModel(allocator, io, platform, qwen_model, parameters, progress);
        return .{ .loaded_model = loaded_model, .prefill_exe = compile_result.prefill_exe.?, .decode_exe = compile_result.decode_exe.?, .params = parameters };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill_exe.deinit();
        self.decode_exe.deinit();
    }
};

pub const Inference = CompiledModel;

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
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
            qwen_model_: model.Model,
            parameters_: CompilationParameters,
            shardings_: [1] *const zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling prefill...", 1);
            defer node_.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill [{f}]", .{now_.untilNow(io_, .awake)});

            return platform_.compile(
                allocator_,
                io_,
                qwen_model_,
                .forward,
                .{
                    parameters_.prefill_tokens,
                    parameters_.token_index,
                    parameters_.kv_cache,
                    parameters_.rng,
                },
                .{ .shardings = &shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, qwen_model, parameters, all_shardings, progress });
    errdefer if (prefill_future.cancel(io)) |v| v.deinit() else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen_model_: model.Model,
            parameters_: CompilationParameters,
            shardings_: [1]*const zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node_ = progress_.start("Compiling decode...", 1);
            defer node_.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode [{f}]", .{now_.untilNow(io_, .awake)});

            return platform_.compile(
                allocator_,
                io_,
                qwen_model_,
                .forward,
                .{
                    parameters_.decode_tokens,
                    parameters_.token_index,
                    parameters_.kv_cache,
                    parameters_.rng,
                },
                .{ .shardings = &shardings_ },
            );
        }
    }.call, .{ allocator, io, platform, qwen_model, parameters, all_shardings, progress });
    errdefer if (decode_future.cancel(io)) |v| v.deinit() else |_| {};

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{
        .prefill_exe = prefill_exe,
        .decode_exe = decode_exe,
    };
}
