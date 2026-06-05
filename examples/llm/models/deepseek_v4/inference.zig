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
    cache: model.Cache,
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
    params: CompilationParameters,

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
            .params = opts,
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

pub const KernelArgs = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    tokens_pos_buf: *zml.Buffer,
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    cache_buffers: *zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
};

const KernelExe = struct {
    exe: zml.Exe,

    pub fn deinit(self: KernelExe) void {
        self.exe.deinit();
    }

    pub fn run(self: KernelExe, args: KernelArgs) !void {
        var exe_args = try self.exe.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        exe_args.set(.{
            args.model_buffers,
            args.tokens_buf,
            args.tokens_pos_buf,
            args.rng_buf,
            args.cache_buffers,
            args.attention_metadata_buffers,
        });

        var results = try self.exe.results(args.allocator);
        defer results.deinit(args.allocator);

        self.exe.call(exe_args, &results);

        var tokens_buffer, const cache_buffer,  var rng_buffer = results.get(struct{
            zml.Buffer,
            zml.Bufferized(model.Cache),
            zml.Bufferized(zml.Tensor.Rng)
        });
        _ = cache_buffer; // autofix

        updateBuffer(args.tokens_buf, &tokens_buffer);
        // updateBuffer(&args.cache_buffers.kv, &cache_buffer.kv);
        updateBuffer(&args.rng_buf._state, &rng_buffer._state);
    }
};

fn updateBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn sameBufferHandle(lhs: zml.Buffer, rhs: zml.Buffer) bool {
    if (lhs._shards.len != rhs._shards.len) return false;

    for (lhs._shards.constSlice(), rhs._shards.constSlice()) |lhs_shards, rhs_shards| {
        if(lhs_shards != rhs_shards) return false;
    }

    return true;
}
