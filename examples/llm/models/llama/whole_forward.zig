const std = @import("std");
const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

pub const Forward = struct {
    pub const Input = struct {
        weights: model.Model,
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
            .tokens = input.tokens,
            .rng = input.rng,
        });
        return .{ .tokens = result.tokens, .kv_cache = kv_cache.reuseBuffer(input.kv_cache), .rng = result.rng };
    }
};

pub const Exe = zml.FnExe(Forward.forward);

pub fn compile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, input: Forward.Input, shardings: common.Shardings, progress: *std.Progress.Node) !Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start(common.Phase.decode.startMessage("forward"), 1);
    defer node.end();
    const from: std.Io.Timestamp = .now(io, .awake);
    defer common.Phase.decode.logCompileDone(std.log.scoped(.llama), "forward", io, from);
    return Exe.compile(allocator, io, platform, .{
        .shardings = &shardings.all(),
        .program_name = common.Phase.decode.programName("llama", "forward"),
    }, .{input});
}

pub const Runner = struct {
    forward: Exe.Runner(.{.weights}),

    pub fn init(allocator: std.mem.Allocator, exe: *const Exe, buffers: *const model.Buffers) !Runner {
        return .{ .forward = try .init(exe, allocator, .{ .weights = buffers.* }) };
    }

    pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
        self.forward.deinit(allocator);
    }

    pub const Args = struct {
        io: std.Io,
        tokens: *zml.Buffer,
        token_index: zml.Buffer,
        kv_cache: *zml.Bufferized(model.KvCache),
        rng: *zml.Bufferized(zml.Tensor.Rng),
        attention_metadata: zml.Bufferized(zml.attention.Metadata),
    };

    pub fn run(self: *Runner, args: Args) void {
        // Results replace handles even when PJRT declines optional donation.
        // Pending execution retains its allocations after these inputs are released.
        var tokens = args.tokens.*;
        var kv_cache = args.kv_cache.*;
        var rng = args.rng.*;
        defer tokens.deinit();
        defer model.KvCache.deinitBuffer(&kv_cache);
        defer zml.Tensor.Rng.deinitBuffer(&rng);
        self.forward.run(args.io, .{
            .inputs = .{
                .tokens = tokens,
                .token_index = args.token_index,
                .kv_cache = kv_cache,
                .rng = rng,
                .attention_metadata = args.attention_metadata,
            },
            .outputs = .{ .tokens = args.tokens, .kv_cache = args.kv_cache, .rng = args.rng },
        });
    }
};

/// Adapts layered inference with optional whole-model Furiosa decode.
pub fn Dispatch(comptime base: anytype, comptime Parameters: type, comptime Args: type, comptime enabled: bool) type {
    const WholeExe = Exe;
    const WholeRunner = Runner;
    const compileWhole = compile;
    return struct {
        pub const KernelExe = union(enum) {
            layered: base.Exe,
            whole: WholeExe,

            pub fn deinit(self: *const KernelExe) void {
                switch (self.*) {
                    inline else => |*exe| exe.deinit(),
                }
            }
        };

        pub const KernelRunner = union(enum) {
            layered: base.Runner,
            whole: WholeRunner,

            pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
                return switch (exe.*) {
                    .layered => |*layered| .{ .layered = try .init(allocator, layered, buffers) },
                    .whole => |*whole| .{ .whole = try .init(allocator, whole, buffers) },
                };
            }

            pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
                switch (self.*) {
                    inline else => |*runner| runner.deinit(allocator),
                }
            }
        };

        pub fn run(runner: *KernelRunner, args: Args, kv_cache_index_buffers: []const zml.Buffer) void {
            switch (runner.*) {
                .layered => |*layered| base.run(layered, args, kv_cache_index_buffers),
                .whole => |*whole| whole.run(.{
                    .io = args.io,
                    .tokens = args.tokens_buf,
                    .token_index = args.token_index_buf.*,
                    .kv_cache = args.kv_cache_buffers,
                    .rng = args.rng_buffers,
                    .attention_metadata = args.attention_metadata_buffers.*,
                }),
            }
        }

        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            llama_model: model.Model,
            parameters: Parameters,
            seqlen: usize,
            attention_parameters: zml.attention.Parameters,
            phase: common.Phase,
            progress: *std.Progress.Node,
        ) !KernelExe {
            if (enabled and platform.target == .furiosa and phase == .decode) {
                return .{ .whole = try compileWhole(allocator, io, platform, .{
                    .weights = llama_model,
                    .tokens = parameters.decode_tokens,
                    .token_index = parameters.token_index,
                    .kv_cache = parameters.kv_cache,
                    .rng = parameters.rng,
                    .attention_metadata = parameters.attention_metadata,
                    .attention_parameters = attention_parameters,
                }, parameters.shardings, progress) };
            }
            return .{ .layered = try base.compile(allocator, io, platform, llama_model, parameters, seqlen, attention_parameters, phase, progress) };
        }
    };
}
