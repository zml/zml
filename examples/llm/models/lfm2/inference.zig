const std = @import("std");

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
    /// Owns the per-layer cache tensor slices.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend, single: bool, shardings: common.Shardings) !CompilationParameters {
        stdx.debug.assert(seqlen >= config.conv_L_cache, "seqlen ({}) must be at least conv_L_cache ({})", .{ seqlen, config.conv_L_cache });
        const cache: model.Cache = .{
            // TT-FIX: per-layer rank-4 KV with `{.batch, .h, .k, .hd}` dim
            // order (head before seq) — matches the canonical layout that
            // tt-mlir's `CacheFillUpdatePattern` matches. Llama uses the
            // same order.
            .kv = try .init(allocator, .init(.{
                .batch = 1,
                .h = config.num_key_value_heads,
                .k = seqlen,
                .hd = config.hidden_size / config.num_attention_heads,
            }, mdl.embed_tokens.weight.dtype()), @intCast(mdl.num_attention_layers)),
            .conv = try .init(allocator, .init(.{
                .batch = 1,
                .seq = config.conv_L_cache,
                .d = config.hidden_size,
            }, mdl.embed_tokens.weight.dtype()), @intCast(mdl.num_conv_layers)),
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
            .allocator = allocator,
        };
    }

    pub fn deinit(self: CompilationParameters) void {
        self.cache.kv.deinit(self.allocator);
        self.cache.conv.deinit(self.allocator);
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
        self.params.deinit();
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

        // TT-FIX: with per-layer `[]Buffer` cache fields, `Results.get`
        // returns an `undefined` struct and tries to walk its slices —
        // segfaults. `fill` writes into pre-allocated buffers
        // (matches llama's path). Model.forward also no longer returns
        // the rng — see the TT-FIX comment in Model.forward.
        results.fill(.{ args.tokens_buf, args.cache_buffers });
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
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .i32);

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
        .{
            .shardings = &all_shardings,
            .tt_enable_trace = null,
        },
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
        const embed_w_shape = args.model_buffers.embed_tokens.weight.shape();
        const hidden_shape = zml.Shape.init(.{
            .batch = args.tokens_buf.shape().dim(.batch),
            .seq = args.tokens_buf.shape().dim(.seq),
            .d = embed_w_shape.dim(.d),
        }, embed_w_shape.dtype());
        var hidden_buf: zml.Buffer = try .uninitialized(args.io, args.platform, hidden_shape, args.platform.replicated_sharding, .{});
        defer hidden_buf.deinit();

        {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers.embed_tokens, args.tokens_buf });
            self.embed_tokens.call(exe_args, &results);
            results.fill(.{&hidden_buf});
        }

        var conv_idx: usize = 0;
        var attn_idx: usize = 0;
        for (args.model_buffers.layers) |layer_bufs| {
            const is_conv = layer_bufs.operator == .conv;
            const exe = if (is_conv) &self.conv_layer else &self.attn_layer;

            var exe_args = try exe.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try exe.results(args.allocator);
            defer results.deinit(args.allocator);

            if (is_conv) {
                exe_args.set(.{
                    layer_bufs,
                    &hidden_buf,
                    args.tokens_pos_buf,
                    args.actual_seq_len_buf,
                    &args.cache_buffers.conv.state[conv_idx],
                    args.attention_metadata_buffers,
                });
                exe.call(exe_args, &results);
                results.fill(.{ &hidden_buf, &args.cache_buffers.conv.state[conv_idx] });
                conv_idx += 1;
            } else {
                exe_args.set(.{
                    layer_bufs,
                    &hidden_buf,
                    args.tokens_pos_buf,
                    args.actual_seq_len_buf,
                    &args.cache_buffers.kv.k[attn_idx],
                    &args.cache_buffers.kv.v[attn_idx],
                    args.attention_metadata_buffers,
                });
                exe.call(exe_args, &results);
                results.fill(.{ &hidden_buf, &args.cache_buffers.kv.k[attn_idx], &args.cache_buffers.kv.v[attn_idx] });
                attn_idx += 1;
            }
        }

        {
            var exe_args = try self.lm_head.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.lm_head.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers.lm_head, hidden_buf, args.model_buffers.embed_tokens, args.tokens_buf, args.rng_buf });
            self.lm_head.call(exe_args, &results);
            results.fill(.{ args.tokens_buf, args.rng_buf });
        }
    }
};

fn replaceCacheBuffers(dst: *zml.Bufferized(model.Cache), src: *zml.Bufferized(model.Cache)) void {
    for (dst.conv.state, src.conv.state) |*d, *s| replaceBuffer(d, s);
    for (dst.kv.k, src.kv.k) |*d, *s| replaceBuffer(d, s);
    for (dst.kv.v, src.kv.v) |*d, *s| replaceBuffer(d, s);
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
    return platform.compile(allocator, io, embed_tokens, .forward, .{tokens}, .{ .shardings = &all_shardings, .tt_enable_trace = null });
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
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .i32);
    const actual_seq_len: zml.Tensor = .init(.{}, .u32);
    // TT-FIX: pass a single-layer conv state — each call sees only its own
    // slot so the kernel stays layer-agnostic (host loop picks the right
    // `state[conv_idx]`).
    const layer_conv_state: zml.Tensor = opts.cache.conv.state[0];

    const all_shardings = shardings.all();
    // TT-FIX: the conv2d weight is a parameter (is_parameter=true from the store)
    // so the `to_layout(device→SystemMemory)` prep gets hoisted into a `LoadCachedOp`
    // via `ConstEvalHoist`. The `LoadCachedOp` cache key includes `inputVersions`,
    // so calling the same compiled kernel across 10 conv layers with 10 different
    // weight tensors gets 10 distinct cache entries — re-prep happens once per layer
    // at first execute, then is cached.
    return platform.compile(allocator, io, conv_layer, .forwardConvComposed, .{
        hidden,
        token_position_offset,
        actual_seq_len,
        layer_conv_state,
        opts.attention_metadata,
        opts.attention_parameters,
        model.ConvParameters{ .is_prefill = is_prefill },
    }, .{
        .shardings = &all_shardings,
        // Tag the hidden (args[1]) and conv_state (args[4]) as kv_cache so
        // TTNNLayout doesn't force them to row major. The previous kernel
        // hands them off in tile layout — without the marker each composed
        // kernel emits a `ttnn.to_layout(arg, tile)` at entry, paid 16× per
        // token across the layer loop.
        .tt_kv_cache_args = &.{ 1, 4 },
        .tt_enable_trace = null,
    });
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
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .i32);
    const actual_seq_len: zml.Tensor = .init(.{}, .u32);
    const layer_k: zml.Tensor = opts.cache.kv.k[0];
    const layer_v: zml.Tensor = opts.cache.kv.v[0];

    const all_shardings = shardings.all();
    return platform.compile(allocator, io, attn_layer, .forwardAttnComposed, .{
        hidden,
        token_position_offset,
        actual_seq_len,
        layer_k,
        layer_v,
        opts.attention_metadata,
        opts.attention_parameters,
        model.ConvParameters{ .is_prefill = is_prefill },
    }, .{
        .shardings = &all_shardings,
        // K (args[4]) and V (args[5]) already auto-tag via the
        // `paged_update_cache` user; the hidden (args[1]) does not, so mark
        // it explicitly to keep its tile layout across the layer boundary.
        .tt_kv_cache_args = &.{1},
        .tt_enable_trace = null,
    });
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
    // TT-FIX: mdl.lm_head weights (args[0]) and mdl.embed_tokens (args[2]) both
    // carry is_parameter=true from TensorStore.View.maybeCreateTensor, so they
    // auto-tag as `<parameter>` — the 256 MB embed weight no longer re-tiles
    // every decode call.
    return platform.compile(allocator, io, mdl.lm_head, .forward, .{ hidden, mdl.embed_tokens, tokens, opts.rng }, .{
        .shardings = &all_shardings,
        .program_name = "lfm2_composed_lm_head",
        // The hidden (args[1]) arrives in tile layout from the last layer
        // kernel — mark it so lm_head doesn't add a `ttnn.to_layout` at entry.
        .tt_kv_cache_args = &.{1},
        .tt_enable_trace = null,
    });
}
