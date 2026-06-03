const model = @import("model.zig");

const std = @import("std");
const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("../common.zig");

const log = std.log.scoped(.deepseek);

pub const CompilationParameters = struct {
    batch_dim: usize,
    seqlen: u32,
    rng: zml.Tensor.Rng,
    cache: model.KVCache,
    shardings: common.Shardings,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,

    pub fn init(seqlen: u32, config: model.Config, mdl: model.Model, shardings: common.Shardings, backend: attention.Backend) CompilationParameters {
        const batch_size = 1;
        return .{
            .batch_dim = batch_size,
            .seqlen = seqlen,
            .rng = .init(),
            .cache = .init(mdl, config, batch_size, seqlen),
            .shardings = shardings,
            .attention_metadata = .init(.fromBackend(backend, seqlen, config.num_attention_heads)),
            .attention_parameters = .init(.fromBackend(backend)),
        };
    }
};

pub const CompiledModel = struct {
    prefill: KernelExe,
    decode: KernelExe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        opts: CompilationParameters,
    ) !CompiledModel {
        return .{
            .prefill = try compileKernel(allocator, io, platform, mdl, seqlen, progress, opts),
            .decode = try compileKernel(allocator, io, platform, mdl, 1, progress, opts),
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    seqlen: u32,
    progress: *std.Progress.Node,
    opts: CompilationParameters,
) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling kernel...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
    const tokens_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);

    const exe = try platform.compile(
        allocator,
        io,
        mdl,
        .forward,
        .{
                tokens,
                tokens_idx_offset ,
                opts.rng,
                opts.cache,
                opts.attention_metadata,
                opts.attention_parameters,
               },
        .{ .shardings = &opts.shardings.all(), }
    );
    return .{ .exe = exe };
}


const KernelExe = struct {
    exe: zml.Exe,

    pub fn deinit(self: KernelExe) void {
        self.exe.deinit();
    }
};

