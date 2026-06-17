const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.step3_5flash);

pub const CompilationParameters = struct {
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(
        mdl: model.Model,
        config: model.Config,
        seqlen: u32,
        shardings: common.Shardings,
    ) CompilationParameters {
        const dtype = mdl.text_model.embed_tokens.weight.dtype();
        const num_layers: i64 = @intCast(mdl.text_model.layers.len);
        const num_kv_heads: i64 = @intCast(config.num_attention_groups);
        const head_dim: i64 = @intCast(config.head_dim);
        const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);

        const raw_kv_shape = zml.Shape.init(.{
            .layer = num_layers,
            .b = @as(i64, 1),
            .k = @as(i64, @intCast(seqlen)),
            .h = num_kv_heads,
            .hd = head_dim,
        }, dtype);
        const kv_shape = model.partitionKvCacheShape(raw_kv_shape, num_kv_heads, model_partitions);

        return .{
            // Attn.forward uses `token_index.slice1d(0, ...)`, so it must be 1-D.
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{ .s = 1 }, .u32),
            .kv_cache = .init(kv_shape),
            .rng = .init(),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    exe: zml.Exe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        step3p5_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start("Compiling step3.5 Model.forward...", 1);
        defer node.end();

        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("Compiled step3.5 Model.forward [{f}]", .{start.untilNow(io, .awake)});

        const all_shardings = parameters.shardings.all();

        const exe = try platform.compile(
            allocator,
            io,
            step3p5_model,
            .forward,
            .{
                parameters.decode_tokens,
                parameters.token_index,
                parameters.kv_cache,
                parameters.rng,
            },
            .{ .shardings = &all_shardings },
        );

        return .{
            .loaded_model = loaded_model,
            .exe = exe,
            .params = parameters,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.exe.deinit();
    }
};
