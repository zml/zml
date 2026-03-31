const std = @import("std");

const tracy = @import("tracy");
const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.lfm);

pub const CompilationParameters = struct {
    hidden_dim: usize,
    batch_dim: usize,
    single: bool,
    rng: zml.Tensor.Rng,
    cache: model.Cache,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend, single: bool, shardings: common.Shardings) CompilationParameters {
        stdx.debug.assert(seqlen >= config.conv_L_cache, "seqlen ({}) must be at least conv_L_cache ({})", .{ seqlen, config.conv_L_cache });
        const cache: model.Cache = .{
            .kv = .init(.init(.{
                .layer = mdl.num_attention_layers,
                .batch = 1,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = config.hidden_size / config.num_attention_heads,
            }, mdl.embed_tokens.weight.dtype())),
            .conv = .init(.init(.{
                .layer = mdl.num_conv_layers,
                .batch = 1,
                .seq = config.conv_L_cache,
                .d = config.hidden_size,
            }, mdl.embed_tokens.weight.dtype())),
        };

        return .{
            .single = single,
            .hidden_dim = config.hidden_size,
            .batch_dim = 1,
            .rng = .init(),
            .cache = cache,
            .attention_metadata = .init(.fromBackend(backend, seqlen, config.num_attention_heads)),
            .attention_parameters = .init(.fromBackend(backend)),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

pub const Args = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    tokens_pos_buf: *zml.Buffer,
    actual_seq_len_buf: *zml.Buffer,
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    cache_buffers: *zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe,
    decode: KernelExe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        loaded_model: *const model.LoadedModel,
        mdl: model.Model,
        opts: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        if (opts.single) {
            const prefill_exe = try compileSingleKernelExe(allocator, io, platform, mdl, opts, opts.seqlen, true, progress, opts.shardings);
            const decode_exe = try compileSingleKernelExe(allocator, io, platform, mdl, opts, 1, false, progress, opts.shardings);

            return .{
                .loaded_model = loaded_model,
                .prefill = .{ .single = prefill_exe },
                .decode = .{ .single = decode_exe },
                .params = opts,
            };
        } else {
            const prefill_exe = try compileComposedKernelExe(allocator, io, platform, mdl, opts, opts.seqlen, true, progress, opts.shardings);
            const decode_exe = try compileComposedKernelExe(allocator, io, platform, mdl, opts, 1, false, progress, opts.shardings);

            return .{
                .loaded_model = loaded_model,
                .prefill = .{ .composed = prefill_exe },
                .decode = .{ .composed = decode_exe },
                .params = opts,
            };
        }
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

pub const KernelExe = union(enum) {
    single: SingleKernelExe,
    composed: ComposedKernelExe,

    pub fn deinit(self: KernelExe) void {
        switch (self) {
            .single => |exe| exe.deinit(),
            .composed => |exe| exe.deinit(),
        }
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        switch (self.*) {
            .single => |*exe| try exe.run(args),
            .composed => |*exe| try exe.run(args),
        }
    }
};

pub const SingleKernelExe = struct {
    exe: zml.Exe,

    fn deinit(self: SingleKernelExe) void {
        self.exe.deinit();
    }

    pub fn run(self: *const SingleKernelExe, args: Args) !void {
        var exe_args = try self.exe.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        var results = try self.exe.results(args.allocator);
        defer results.deinit(args.allocator);

        exe_args.set(.{
            args.model_buffers,
            args.tokens_buf,
            args.tokens_pos_buf,
            args.actual_seq_len_buf,
            args.rng_buf,
            args.cache_buffers,
            args.attention_metadata_buffers,
        });
        self.exe.call(exe_args, &results);

        var new_tokens, var new_cache, var new_rng = results.get(struct {
            zml.Buffer,
            zml.Bufferized(model.Cache),
            zml.Bufferized(zml.Tensor.Rng),
        });
        replaceBuffer(args.tokens_buf, &new_tokens);
        replaceCacheBuffers(args.cache_buffers, &new_cache);
        replaceBuffer(&args.rng_buf._state, &new_rng._state);
    }
};

fn compileSingleKernelExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    is_prefill: bool,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !SingleKernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling single kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled single kernel [{f}]", .{now.untilNow(io, .awake)});

    const actual_seq_len: zml.Tensor = .init(.{}, .u32);
    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);

    const all_shardings = shardings.all();
    const exe = try platform.compile(
        allocator,
        io,
        mdl,
        .forward,
        .{
            tokens,
            token_position_offset,
            actual_seq_len,
            opts.rng,
            opts.cache,
            opts.attention_metadata,
            opts.attention_parameters,
            model.ConvParameters{ .is_prefill = is_prefill },
        },
        .{ .shardings = &all_shardings },
    );

    return .{ .exe = exe };
}

pub const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    conv_layer: zml.Exe,
    attn_layer: zml.Exe,
    lm_head: zml.Exe,

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        self.conv_layer.deinit();
        self.attn_layer.deinit();
        self.lm_head.deinit();
    }

    pub fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = b: {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers.embed_tokens, args.tokens_buf });
            self.embed_tokens.call(exe_args, &results);
            break :b results.get(zml.Buffer);
        };
        defer hidden_buf.deinit();

        const replicated_sharding = try zml.sharding.replicatedSharding(args.platform);
        var conv_cache_index_buf: zml.Buffer = try .scalar(args.io, args.platform, @as(u32, 0), .u32, replicated_sharding);
        defer conv_cache_index_buf.deinit();
        var kv_cache_index_buf: zml.Buffer = try .scalar(args.io, args.platform, @as(u32, 0), .u32, replicated_sharding);
        defer kv_cache_index_buf.deinit();

        for (args.model_buffers.layers) |layer_bufs| {
            const exe = switch (layer_bufs.operator) {
                .conv => &self.conv_layer,
                .self_attn => &self.attn_layer,
            };

            var exe_args = try exe.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try exe.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{
                layer_bufs,
                &hidden_buf,
                args.tokens_pos_buf,
                args.actual_seq_len_buf,
                args.cache_buffers,
                &conv_cache_index_buf,
                &kv_cache_index_buf,
                args.attention_metadata_buffers,
            });
            exe.call(exe_args, &results);

            var new_hidden, var new_cache, var new_conv_cache_index, var new_kv_cache_index = results.get(struct {
                zml.Buffer,
                zml.Bufferized(model.Cache),
                zml.Buffer,
                zml.Buffer,
            });
            replaceBuffer(&hidden_buf, &new_hidden);
            replaceCacheBuffers(args.cache_buffers, &new_cache);
            replaceBuffer(&conv_cache_index_buf, &new_conv_cache_index);
            replaceBuffer(&kv_cache_index_buf, &new_kv_cache_index);
        }

        {
            var exe_args = try self.lm_head.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.lm_head.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers.lm_head, hidden_buf, args.model_buffers.embed_tokens, args.tokens_buf, args.rng_buf });
            self.lm_head.call(exe_args, &results);
            var new_tokens, var new_rng = results.get(struct {
                zml.Buffer,
                zml.Bufferized(zml.Tensor.Rng),
            });
            replaceBuffer(args.tokens_buf, &new_tokens);
            replaceBuffer(&args.rng_buf._state, &new_rng._state);
        }
    }
};

fn replaceCacheBuffers(dst: *zml.Bufferized(model.Cache), src: *zml.Bufferized(model.Cache)) void {
    replaceBuffer(&dst.conv.state, &src.conv.state);
    replaceBuffer(&dst.kv.k, &src.kv.k);
    replaceBuffer(&dst.kv.v, &src.kv.v);
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}

fn compileComposedKernelExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    is_prefill: bool,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !ComposedKernelExe {
    const embed_tokens_exe = try compileEmbedTokens(allocator, io, platform, mdl.embed_tokens, opts, seqlen, progress, shardings);
    const conv_layer_exe = try compileConvLayer(allocator, io, platform, mdl, opts, seqlen, is_prefill, progress, shardings);
    const attn_layer_exe = try compileAttnLayer(allocator, io, platform, mdl, opts, seqlen, is_prefill, progress, shardings);
    const lm_head_exe = try compileLmHead(allocator, io, platform, mdl, opts, seqlen, progress, shardings);
    return .{
        .embed_tokens = embed_tokens_exe,
        .conv_layer = conv_layer_exe,
        .attn_layer = attn_layer_exe,
        .lm_head = lm_head_exe,
    };
}

fn compileEmbedTokens(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    embed_tokens: model.TokenEmbedding,
    opts: CompilationOptions,
    seqlen: u32,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling embed_tokens...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled embed_tokens [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
    const all_shardings = shardings.all();
    return platform.compile(allocator, io, embed_tokens, .forward, .{tokens}, .{ .shardings = &all_shardings });
}

fn compileConvLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    is_prefill: bool,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling conv layer...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled conv layer [{f}]", .{now.untilNow(io, .awake)});

    const conv_layer = for (mdl.layers) |layer| {
        if (layer.operator == .conv) break layer;
    } else unreachable;

    const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype());
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
    const actual_seq_len: zml.Tensor = .init(.{}, .u32);
    const conv_cache_index: zml.Tensor = .init(.{}, .u32);
    const kv_cache_index: zml.Tensor = .init(.{}, .u32);

    const all_shardings = shardings.all();
    return platform.compile(allocator, io, conv_layer, .forward, .{
        hidden,
        token_position_offset,
        actual_seq_len,
        opts.cache,
        conv_cache_index,
        kv_cache_index,
        opts.attention_metadata,
        opts.attention_parameters,
        model.ConvParameters{ .is_prefill = is_prefill },
    }, .{ .shardings = &all_shardings });
}

fn compileAttnLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    is_prefill: bool,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling attn layer...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled attn layer [{f}]", .{now.untilNow(io, .awake)});

    const attn_layer = for (mdl.layers) |layer| {
        if (layer.operator == .self_attn) break layer;
    } else unreachable;

    const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype());
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
    const actual_seq_len: zml.Tensor = .init(.{}, .u32);
    const conv_cache_index: zml.Tensor = .init(.{}, .u32);
    const kv_cache_index: zml.Tensor = .init(.{}, .u32);

    const all_shardings = shardings.all();
    return platform.compile(allocator, io, attn_layer, .forward, .{
        hidden,
        token_position_offset,
        actual_seq_len,
        opts.cache,
        conv_cache_index,
        kv_cache_index,
        opts.attention_metadata,
        opts.attention_parameters,
        model.ConvParameters{ .is_prefill = is_prefill },
    }, .{ .shardings = &all_shardings });
}

fn compileLmHead(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    mdl: model.Model,
    opts: CompilationOptions,
    seqlen: u32,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling lm_head...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled lm_head [{f}]", .{now.untilNow(io, .awake)});

    const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype());
    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);
    const all_shardings = shardings.all();
    return platform.compile(allocator, io, mdl.lm_head, .forward, .{ hidden, mdl.embed_tokens, tokens, opts.rng }, .{ .shardings = &all_shardings });
}
