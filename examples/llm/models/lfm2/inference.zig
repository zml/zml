const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const Phase = common.Phase;
const model = @import("model.zig");

const log = std.log.scoped(.lfm);
pub const CompilationParameters = struct {
    hidden_dim: usize,
    batch_dim: usize,
    rng: zml.Tensor.Rng,
    cache: model.Cache,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: zml.attention.Backend, shardings: common.Shardings) CompilationParameters {
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
    attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata),
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
        return .{
            .loaded_model = loaded_model,
            .prefill = try KernelExe.init(allocator, io, platform, mdl, opts, opts.seqlen, .prefill, progress),
            .decode = try KernelExe.init(allocator, io, platform, mdl, opts, 1, .decode, progress),
            .params = opts,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

pub const Inference = CompiledModel;

pub const KernelExe = struct {
    composed: ComposedKernelExe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        opts: CompilationOptions,
        seqlen: u32,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !KernelExe {
        return .{
            .composed = try ComposedKernelExe.init(allocator, io, platform, mdl, opts, seqlen, phase, progress),
        };
    }

    pub fn deinit(self: KernelExe) void {
        self.composed.deinit();
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        try self.composed.run(args);
    }
};

pub const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    conv_layer: zml.Exe,
    attn_layer: zml.Exe,
    lm_head: zml.Exe,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        opts: CompilationOptions,
        seqlen: u32,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !ComposedKernelExe {
        const embed_tokens = try ComposedKernelExe.compileEmbedTokens(allocator, io, platform, mdl.embed_tokens, opts, seqlen, phase, progress);
        errdefer embed_tokens.deinit();

        const conv_layer = try ComposedKernelExe.compileConvLayer(allocator, io, platform, mdl, opts, seqlen, phase, progress);
        errdefer conv_layer.deinit();

        const attn_layer = try ComposedKernelExe.compileAttnLayer(allocator, io, platform, mdl, opts, seqlen, phase, progress);
        errdefer attn_layer.deinit();

        const lm_head = try ComposedKernelExe.compileLmHead(allocator, io, platform, mdl, opts, seqlen, phase, progress);
        errdefer lm_head.deinit();

        return .{
            .embed_tokens = embed_tokens,
            .conv_layer = conv_layer,
            .attn_layer = attn_layer,
            .lm_head = lm_head,
        };
    }

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

        var conv_cache_index_buf: zml.Buffer = try .scalar(args.io, args.platform, 0, .u32);
        defer conv_cache_index_buf.deinit();
        var kv_cache_index_buf: zml.Buffer = try .scalar(args.io, args.platform, 0, .u32);
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
            ComposedKernelExe.runLayer(exe, &exe_args, &results, args.cache_buffers, &hidden_buf, &conv_cache_index_buf, &kv_cache_index_buf);
        }

        var exe_args = try self.lm_head.args(args.allocator);
        defer exe_args.deinit(args.allocator);
        var results = try self.lm_head.results(args.allocator);
        defer results.deinit(args.allocator);

        exe_args.set(.{ args.model_buffers.lm_head, hidden_buf, args.model_buffers.embed_tokens, args.tokens_buf, args.rng_buf });
        self.runLmHead(&exe_args, &results, args.tokens_buf, args.rng_buf);
    }

    fn runLayer(
        exe: *const zml.Exe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        cache_buffers: *zml.Bufferized(model.Cache),
        hidden_buf: *zml.Buffer,
        conv_cache_index_buf: *zml.Buffer,
        kv_cache_index_buf: *zml.Buffer,
    ) void {
        exe.call(exe_args.*, results);

        var new_hidden, var new_cache, var new_conv_cache_index, var new_kv_cache_index = results.get(struct {
            zml.Buffer,
            zml.Bufferized(model.Cache),
            zml.Buffer,
            zml.Buffer,
        });
        ComposedKernelExe.replaceBuffer(hidden_buf, &new_hidden);
        ComposedKernelExe.replaceCacheBuffers(cache_buffers, &new_cache);
        ComposedKernelExe.replaceBuffer(conv_cache_index_buf, &new_conv_cache_index);
        ComposedKernelExe.replaceBuffer(kv_cache_index_buf, &new_kv_cache_index);
    }

    fn runLmHead(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        tokens_buf: *zml.Buffer,
        rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    ) void {
        self.lm_head.call(exe_args.*, results);

        var new_tokens, var new_rng = results.get(struct {
            zml.Buffer,
            zml.Bufferized(zml.Tensor.Rng),
        });
        ComposedKernelExe.replaceBuffer(tokens_buf, &new_tokens);
        ComposedKernelExe.replaceBuffer(&rng_buf._state, &new_rng._state);
    }

    fn replaceCacheBuffers(dst: *zml.Bufferized(model.Cache), src: *zml.Bufferized(model.Cache)) void {
        ComposedKernelExe.replaceBuffer(&dst.conv.state, &src.conv.state);
        ComposedKernelExe.replaceBuffer(&dst.kv.k, &src.kv.k);
        ComposedKernelExe.replaceBuffer(&dst.kv.v, &src.kv.v);
    }

    fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
        if (!ComposedKernelExe.sameBufferHandle(dst.*, src.*)) {
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

    fn compileEmbedTokens(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        embed_tokens: model.TokenEmbedding,
        opts: CompilationOptions,
        seqlen: u32,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("embed_tokens"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "embed_tokens", io, from);

        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);

        return platform.compile(allocator, io, embed_tokens, .forward, .{tokens}, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("lfm2", "embed_tokens"),
        });
    }

    fn compileConvLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        opts: CompilationOptions,
        seqlen: u32,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("conv layer"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "conv layer", io, from);

        const conv_layer = for (mdl.layers) |layer| {
            if (layer.operator == .conv) break layer;
        } else unreachable;

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype());
        const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
        const actual_seq_len: zml.Tensor = .init(.{}, .u32);
        const conv_cache_index: zml.Tensor = .init(.{}, .u32);
        const kv_cache_index: zml.Tensor = .init(.{}, .u32);

        return platform.compile(allocator, io, conv_layer, .forward, .{
            hidden,
            token_position_offset,
            actual_seq_len,
            opts.cache,
            conv_cache_index,
            kv_cache_index,
            opts.attention_metadata,
            opts.attention_parameters,
            model.ConvParameters{ .is_prefill = phase.isPrefill() },
        }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("lfm2", "conv_layer"),
        });
    }

    fn compileAttnLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        opts: CompilationOptions,
        seqlen: u32,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("attn layer"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "attn layer", io, from);

        const attn_layer = for (mdl.layers) |layer| {
            if (layer.operator == .self_attn) break layer;
        } else unreachable;

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype());
        const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);
        const actual_seq_len: zml.Tensor = .init(.{}, .u32);
        const conv_cache_index: zml.Tensor = .init(.{}, .u32);
        const kv_cache_index: zml.Tensor = .init(.{}, .u32);

        return platform.compile(allocator, io, attn_layer, .forward, .{
            hidden,
            token_position_offset,
            actual_seq_len,
            opts.cache,
            conv_cache_index,
            kv_cache_index,
            opts.attention_metadata,
            opts.attention_parameters,
            model.ConvParameters{ .is_prefill = phase.isPrefill() },
        }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("lfm2", "attn_layer"),
        });
    }

    fn compileLmHead(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        mdl: model.Model,
        opts: CompilationOptions,
        seqlen: u32,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("lm_head"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "lm_head", io, from);

        const hidden: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen, .d = opts.hidden_dim }, mdl.embed_tokens.weight.dtype());
        const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = seqlen }, .u32);

        return platform.compile(allocator, io, mdl.lm_head, .forward, .{ hidden, mdl.embed_tokens, tokens, opts.rng }, .{
            .shardings = &opts.shardings.all(),
            .program_name = phase.programName("lfm2", "lm_head"),
        });
    }
};
