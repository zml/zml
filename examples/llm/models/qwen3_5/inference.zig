const std = @import("std");

const zml = @import("zml");

const Shardings = @import("../common.zig").Shardings;
const model = @import("model.zig");

const log = std.log.scoped(.qwen3_5);

const CompileModelResult = struct {
    prefill_exe: ?zml.Exe = null,
    decode_exe: ?zml.Exe = null,
};

pub const CompilationOptions = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32) CompilationOptions {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        return .{
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(config, 1, seqlen, dtype, .f32),
            .rng = .init(),
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
        qwen_model: model.Model,
        parameters: CompilationOptions,
        shardings: Shardings,
        progress: *std.Progress.Node,
    ) !Inference {
        const compile_result = try compileModel(allocator, io, platform, qwen_model, parameters, shardings, progress);
        return .{ .prefill_exe = compile_result.prefill_exe.?, .decode_exe = compile_result.decode_exe.? };
    }

    pub fn deinit(self: *Inference) void {
        self.prefill_exe.deinit();
        self.decode_exe.deinit();
    }
};

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: model.Model,
    parameters: CompilationOptions,
    shardings: Shardings,
    progress: *std.Progress.Node,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});

    const all_shardings = [_]zml.sharding.Sharding{shardings.replicated};

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen_model_: model.Model,
            parameters_: CompilationOptions,
            shardings_: [1]zml.sharding.Sharding,
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
            parameters_: CompilationOptions,
            shardings_: [1]zml.sharding.Sharding,
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

    return .{
        .prefill_exe = try prefill_future.await(io),
        .decode_exe = try decode_future.await(io),
    };
}
