const model = @import("model.zig");

const std = @import("std");
const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("../common.zig");

const log = std.log.scoped(.deepseek);

pub const CompilationParameters = struct {
    batch_dim: usize,
    hidden_dim: u32,
    hc_mult: u32,
    seqlen: u32,
    rng: zml.Tensor.Rng,
    cache: model.Cache,
    shardings: common.Shardings,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
    moe_metadata: zml.moe.Metadata,
    moe_parameters: zml.moe.Parameters,

    pub fn init(seqlen: u32, config: model.Config, mdl: model.Model, shardings: common.Shardings, backend: attention.Backend, moe_backend: zml.moe.Backend) CompilationParameters {
        const batch_size = 1;

        return .{
            .batch_dim = batch_size,
            .hidden_dim = config.hidden_size,
            .hc_mult = config.hc_mult,
            .seqlen = seqlen,
            .rng = .init(),
            .cache = .init(mdl, config, batch_size, seqlen),
            .shardings = shardings,
            .attention_metadata = .init(.fromBackend(backend, seqlen, config.num_attention_heads)),
            .attention_parameters = .init(.fromBackend(backend)),
            .moe_metadata = .init(.fromBackend(moe_backend)),
            .moe_parameters = .init(.fromBackend(moe_backend, config.num_experts_per_tok, .silu)),
        };
    }
};

pub const CompiledModel = struct {
    // prefill: KernelExe,
    // decode: KernelExe,
    prefill: ComposedKernelExe,
    decode: ComposedKernelExe,
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
            // .prefill = try compileKernel(allocator, io, platform, mdl, seqlen, progress, opts),
            // .decode = try compileDecodeKernel(allocator, io, platform, mdl, 1, progress, opts),
            .prefill = try .init(allocator, io, platform, mdl, seqlen, progress, .prefill, opts),
            .decode = try .init(allocator, io, platform, mdl, 1, progress, .decode, opts),
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
    var node = progress.start("Compiling prefill kernel...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled prefill kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
    const tokens_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);

    const actual_seqlen: zml.Tensor = .init(.{}, .u32);

    const exe = try platform.compile(allocator, io, mdl, .forward, .{
        tokens,
        tokens_idx_offset,
        actual_seqlen,
        opts.rng,
        opts.cache,
        opts.attention_metadata,
        opts.attention_parameters,
        opts.moe_metadata,
        opts.moe_parameters,
    }, .{
        .shardings = &opts.shardings.all(),
    });
    return .{ .exe = exe };
}

fn compileDecodeKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    seqlen: u32,
    progress: *std.Progress.Node,
    opts: CompilationParameters,
) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling decode kernel...", 1);
    defer node.end();

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled decode kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
    const tokens_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);

    const actual_seqlen: zml.Tensor = .init(.{}, .u32);

    const exe = try platform.compile(allocator, io, mdl, .forward, .{
        tokens,
        tokens_idx_offset,
        actual_seqlen,
        opts.rng,
        opts.cache,
        opts.attention_metadata,
        opts.attention_parameters,
        opts.moe_metadata,
        opts.moe_parameters,
    }, .{
        .shardings = &opts.shardings.all(),
    });
    return .{ .exe = exe };
}

pub const KernelArgs = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    tokens_pos_buf: *zml.Buffer,
    seqlen_buf: *zml.Buffer,
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    cache_buffers: *zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
    moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
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
            args.seqlen_buf,
            args.rng_buf,
            args.cache_buffers,
            args.attention_metadata_buffers,
            args.moe_metadata_buffers,
        });

        var results = try self.exe.results(args.allocator);
        defer results.deinit(args.allocator);

        self.exe.call(exe_args, &results);

        var tokens_buffer, var rng_buffer, var cache_buffer = results.get(struct {
            zml.Buffer,
            zml.Bufferized(zml.Tensor.Rng),
            zml.Bufferized(model.Cache),
        });

        updateBuffer(args.tokens_buf, &tokens_buffer);
        updateBuffer(&args.rng_buf._state, &rng_buffer._state);
        updateCacheBuffers(args.cache_buffers, &cache_buffer);
    }
};

fn updateBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn updateCacheBuffers(dst: *zml.Bufferized(model.Cache), src: *zml.Bufferized(model.Cache)) void {
    updateBuffer(&dst.sliding_window.kv, &src.sliding_window.kv);

    updateBuffer(&dst.hca.state.kv_state, &src.hca.state.kv_state);
    updateBuffer(&dst.hca.state.score_state, &src.hca.state.score_state);
    updateBuffer(&dst.hca.compressed_kv.kv, &src.hca.compressed_kv.kv);

    updateBuffer(&dst.csa.state.kv_state, &src.csa.state.kv_state);
    updateBuffer(&dst.csa.state.score_state, &src.csa.state.score_state);
    updateBuffer(&dst.csa.indexer.state.kv_state, &src.csa.indexer.state.kv_state);
    updateBuffer(&dst.csa.indexer.state.score_state, &src.csa.indexer.state.score_state);
    updateBuffer(&dst.csa.indexer.kv.kv, &src.csa.indexer.kv.kv);
    updateBuffer(&dst.csa.compressed_kv.kv, &src.csa.compressed_kv.kv);
}

fn sameBufferHandle(lhs: zml.Buffer, rhs: zml.Buffer) bool {
    if (lhs._shards.len != rhs._shards.len) return false;

    for (lhs._shards.constSlice(), rhs._shards.constSlice()) |lhs_shards, rhs_shards| {
        if (lhs_shards != rhs_shards) return false;
    }

    return true;
}

pub const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    attn_layer: zml.Exe,
    csa_attn_layer: zml.Exe,
    csa2_attn_layer: zml.Exe,
    hca_attn_layer: zml.Exe,
    lm_head: zml.Exe,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !ComposedKernelExe {
        const embed_tokens, const attn_layer, const csa_attn_layer, const csa2_attn_layer, const hca_attn_layer, const lm_head = b: {
            var embed_compile_fut = io.async(struct {
                fn call(
                    allocator_: std.mem.Allocator,
                    io_: std.Io,
                    platform_: *zml.Platform,
                    mdl_: model.Model,
                    seqlen_: u32,
                    progress_: *std.Progress.Node,
                    phase_: common.Phase,
                    opts_: CompilationParameters,
                ) !zml.Exe {
                    return ComposedKernelExe.compileEmbedTokens(
                        allocator_,
                        io_,
                        platform_,
                        mdl_,
                        seqlen_,
                        progress_,
                        phase_,
                        opts_,
                    );
                }
            }.call, .{ allocator, io, platform, mdl, seqlen, progress, phase, opts });
            errdefer if (embed_compile_fut.cancel(io)) |exe| {
                exe.deinit();
            } else |_| {};

            var attn_compile_fut = io.async(struct {
                fn call(
                    allocator_: std.mem.Allocator,
                    io_: std.Io,
                    platform_: *zml.Platform,
                    mdl_: model.Model,
                    seqlen_: u32,
                    progress_: *std.Progress.Node,
                    phase_: common.Phase,
                    opts_: CompilationParameters,
                ) !zml.Exe {
                    return ComposedKernelExe.compileAttnLayer(
                        allocator_,
                        io_,
                        platform_,
                        mdl_,
                        seqlen_,
                        progress_,
                        phase_,
                        opts_,
                    );
                }
            }.call, .{ allocator, io, platform, mdl, seqlen, progress, phase, opts });
            errdefer if (attn_compile_fut.cancel(io)) |exe| {
                exe.deinit();
            } else |_| {};

            var csa_attn_compile_fut = io.async(struct {
                fn call(
                    allocator_: std.mem.Allocator,
                    io_: std.Io,
                    platform_: *zml.Platform,
                    mdl_: model.Model,
                    seqlen_: u32,
                    progress_: *std.Progress.Node,
                    phase_: common.Phase,
                    opts_: CompilationParameters,
                ) !zml.Exe {
                    return ComposedKernelExe.compileCSAAttnLayer(
                        allocator_,
                        io_,
                        platform_,
                        mdl_,
                        seqlen_,
                        progress_,
                        phase_,
                        opts_,
                    );
                }
            }.call, .{ allocator, io, platform, mdl, seqlen, progress, phase, opts });
            errdefer if (csa_attn_compile_fut.cancel(io)) |exe| {
                exe.deinit();
            } else |_| {};

            var csa2_attn_compile_fut = io.async(struct {
                fn call(
                    allocator_: std.mem.Allocator,
                    io_: std.Io,
                    platform_: *zml.Platform,
                    mdl_: model.Model,
                    seqlen_: u32,
                    progress_: *std.Progress.Node,
                    phase_: common.Phase,
                    opts_: CompilationParameters,
                ) !zml.Exe {
                    return ComposedKernelExe.compileCSA2AttnLayer(
                        allocator_,
                        io_,
                        platform_,
                        mdl_,
                        seqlen_,
                        progress_,
                        phase_,
                        opts_,
                    );
                }
            }.call, .{ allocator, io, platform, mdl, seqlen, progress, phase, opts });
            errdefer if (csa2_attn_compile_fut.cancel(io)) |exe| {
                exe.deinit();
            } else |_| {};

            var hca_attn_compile_fut = io.async(struct {
                fn call(
                    allocator_: std.mem.Allocator,
                    io_: std.Io,
                    platform_: *zml.Platform,
                    mdl_: model.Model,
                    seqlen_: u32,
                    progress_: *std.Progress.Node,
                    phase_: common.Phase,
                    opts_: CompilationParameters,
                ) !zml.Exe {
                    return ComposedKernelExe.compileHCAAttnLayer(
                        allocator_,
                        io_,
                        platform_,
                        mdl_,
                        seqlen_,
                        progress_,
                        phase_,
                        opts_,
                    );
                }
            }.call, .{ allocator, io, platform, mdl, seqlen, progress, phase, opts });
            errdefer if (hca_attn_compile_fut.cancel(io)) |exe| {
                exe.deinit();
            } else |_| {};

            var lm_head_compile_fut = io.async(struct {
                fn call(
                    allocator_: std.mem.Allocator,
                    io_: std.Io,
                    platform_: *zml.Platform,
                    mdl_: model.Model,
                    seqlen_: u32,
                    progress_: *std.Progress.Node,
                    phase_: common.Phase,
                    opts_: CompilationParameters,
                ) !zml.Exe {
                    return ComposedKernelExe.compileHCAAttnLayer(
                        allocator_,
                        io_,
                        platform_,
                        mdl_,
                        seqlen_,
                        progress_,
                        phase_,
                        opts_,
                    );
                }
            }.call, .{ allocator, io, platform, mdl, seqlen, progress, phase, opts });
            errdefer if (lm_head_compile_fut.cancel(io)) |exe| {
                exe.deinit();
            } else |_| {};

            break :b .{
                embed_compile_fut.await(io) catch return error.Unexpected,
                attn_compile_fut.await(io) catch return error.Unexpected,
                csa_attn_compile_fut.await(io) catch return error.Unexpected,
                csa2_attn_compile_fut.await(io) catch return error.Unexpected,
                hca_attn_compile_fut.await(io) catch return error.Unexpected,
                lm_head_compile_fut.await(io) catch return error.Unexpected,
            };
        };

        return .{
            .embed_tokens = embed_tokens,
            .attn_layer = attn_layer,
            .csa_attn_layer = csa_attn_layer,
            .csa2_attn_layer = csa2_attn_layer,
            .hca_attn_layer = hca_attn_layer,
            .lm_head = lm_head,
        };
    }

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        self.attn_layer.deinit();
        self.csa_attn_layer.deinit();
        self.csa2_attn_layer.deinit();
        self.hca_attn_layer.deinit();
        self.lm_head.deinit();
    }

    fn compileEmbedTokens(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("token embeddings"), 1);
        defer node.end();

        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "token embeddings", io, from);

        return platform.compile(allocator, io, mdl.embeds, .forward, .{tokens}, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("deepseek_v4", "embeddings"),
        });
    }

    fn compileAttnLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("attention"), 1);
        defer node.end();

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .hc = opts.hc_mult, .d = opts.hidden_dim }, mdl.embeds.embeds.weight.dtype());
        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
        const token_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
        const actual_seqlen: zml.Tensor = .init(.{}, .u32);
        const layer_idx: zml.Tensor = .init(.{}, .u32);

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "attention", io, from);

        return platform.compile(allocator, io, mdl.layers[0], .forward, .{
            hidden,
            tokens,
            token_idx_offset,
            actual_seqlen,
            layer_idx,
            opts.cache,
            opts.attention_metadata,
            opts.attention_parameters,
            opts.moe_metadata,
            opts.moe_parameters,
        }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("deepseek_v4", "full_attn"),
        });
    }

    fn compileCSAAttnLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("CSA attention"), 1);
        defer node.end();

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .hc = opts.hc_mult, .d = opts.hidden_dim }, mdl.embeds.embeds.weight.dtype());
        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
        const token_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
        const actual_seqlen: zml.Tensor = .init(.{}, .u32);
        const layer_idx: zml.Tensor = .init(.{}, .u32);

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "CSA attention", io, from);

        return platform.compile(allocator, io, mdl.layers[2], .forward, .{
            hidden,
            tokens,
            token_idx_offset,
            actual_seqlen,
            layer_idx,
            opts.cache,
            opts.attention_metadata,
            opts.attention_parameters,
            opts.moe_metadata,
            opts.moe_parameters,
        }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("deepseek_v4", "csa_attn"),
        });
    }

    fn compileCSA2AttnLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("CSA attention"), 1);
        defer node.end();

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .hc = opts.hc_mult, .d = opts.hidden_dim }, mdl.embeds.embeds.weight.dtype());
        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
        const token_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
        const actual_seqlen: zml.Tensor = .init(.{}, .u32);
        const layer_idx: zml.Tensor = .init(.{}, .u32);

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "CSA attention", io, from);

        return platform.compile(allocator, io, mdl.layers[4], .forward, .{
            hidden,
            tokens,
            token_idx_offset,
            actual_seqlen,
            layer_idx,
            opts.cache,
            opts.attention_metadata,
            opts.attention_parameters,
            opts.moe_metadata,
            opts.moe_parameters,
        }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("deepseek_v4", "csa2_attn"),
        });
    }

    fn compileHCAAttnLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("HCA attention"), 1);
        defer node.end();

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .hc = opts.hc_mult, .d = opts.hidden_dim }, mdl.embeds.embeds.weight.dtype());
        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
        const token_idx_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
        const actual_seqlen: zml.Tensor = .init(.{}, .u32);
        const layer_idx: zml.Tensor = .init(.{}, .u32);

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "HCA attention", io, from);

        return platform.compile(allocator, io, mdl.layers[3], .forward, .{
            hidden,
            tokens,
            token_idx_offset,
            actual_seqlen,
            layer_idx,
            opts.cache,
            opts.attention_metadata,
            opts.attention_parameters,
            opts.moe_metadata,
            opts.moe_parameters,
        }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("deepseek_v4", "hca_attn"),
        });
    }

    fn compileLmHead(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        seqlen: u32,
        progress: *std.Progress.Node,
        phase: common.Phase,
        opts: CompilationParameters,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("lm head"), 1);
        defer node.end();

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .hc = opts.hc_mult, .d = opts.hidden_dim }, mdl.embeds.embeds.weight.dtype());

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "lm head", io, from);

        return platform.compile(allocator, io, mdl.lm_head, .forward, .{ hidden, opts.rng }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("deepseek_v4", "lm_head"),
        });
    }

    pub fn run(self: *const ComposedKernelExe, args: KernelArgs) !void {
        var cache_index_buffer: zml.Buffer = try .scalar(args.io, args.platform, 0, .u32);
        defer cache_index_buffer.deinit();

        var hidden_buffer = try self.runEmbeds(args);
        defer hidden_buffer.deinit();

        for (args.model_buffers.layers, 0..) |layer_buffer, i| {
            const exe = switch (layer_buffer.attn.compression) {
                .none => &self.attn_layer,
                .csa => if (i == 2) &self.csa_attn_layer else &self.csa2_attn_layer,
                .hca => &self.hca_attn_layer,
            };
            var new_hidden_buffer, var new_cache_buffer, var new_cache_index_buffer = try self.runLayer(exe, args, layer_buffer, &hidden_buffer, cache_index_buffer);
            updateBuffer(&hidden_buffer, &new_hidden_buffer);
            updateBuffer(&cache_index_buffer, &new_cache_index_buffer);
            updateCacheBuffers(args.cache_buffers, &new_cache_buffer);
        }

        var new_tokens_buffer, var new_rng_buffer = try self.runLmHead(args, &hidden_buffer, args.rng_buf);
        updateBuffer(args.tokens_buf, &new_tokens_buffer);
        updateBuffer(&args.rng_buf._state, &new_rng_buffer._state);
    }

    fn runEmbeds(self: *const ComposedKernelExe, args: KernelArgs) !zml.Buffer {
        var exe_args = try self.embed_tokens.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        exe_args.set(.{ args.model_buffers.embeds, args.tokens_buf });

        var results = try self.embed_tokens.results(args.allocator);
        defer results.deinit(args.allocator);

        self.embed_tokens.call(exe_args, &results);

        return results.get(zml.Buffer);
    }

    fn runLayer(self: *const ComposedKernelExe, exe: *const zml.Exe, args: KernelArgs, layer_buffer: zml.Bufferized(model.Layer), hidden_buffer: *zml.Buffer, cache_idx_buffer: zml.Buffer) !struct { zml.Buffer, zml.Bufferized(model.Cache), zml.Buffer } {
        _ = self; // autofix
        var exe_args = try exe.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        exe_args.set(.{
            layer_buffer,
            hidden_buffer,
            args.tokens_buf,
            args.tokens_pos_buf,
            args.seqlen_buf,
            &cache_idx_buffer,
            args.cache_buffers,
            args.attention_metadata_buffers,
            args.moe_metadata_buffers,
        });

        var results = try exe.results(args.allocator);
        defer results.deinit(args.allocator);

        exe.call(exe_args, &results);

        return results.get(struct {
            zml.Buffer,
            zml.Bufferized(model.Cache),
            zml.Buffer,
        });
    }

    fn runLmHead(self: *const ComposedKernelExe, args: KernelArgs, hidden_buffer: *zml.Buffer, rng_buffer: *zml.Bufferized(zml.Tensor.Rng)) !struct { zml.Buffer, zml.Bufferized(zml.Tensor.Rng) } {
        var exe_args = try self.lm_head.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        exe_args.set(.{ args.model_buffers.lm_head, hidden_buffer, rng_buffer });

        var results = try self.lm_head.results(args.allocator);
        defer results.deinit(args.allocator);

        self.lm_head.call(exe_args, &results);

        return results.get(struct {
            zml.Buffer,
            zml.Bufferized(zml.Tensor.Rng),
        });
    }
};
