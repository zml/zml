const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.llama);

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
    seqlen: u32,
    single: bool,
    hidden_size: u32,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend, single: bool, shardings: common.Shardings) CompilationParameters {
        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(.init(.{
                .layer = mdl.model.layers.len,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
            }, mdl.model.embed_tokens.weight.dtype())),
            .rng = .init(),
            .attention_metadata = .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            .attention_parameters = .init(.fromBackend(backend)),
            .seqlen = seqlen,
            .single = single,
            .hidden_size = config.hidden_size,
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
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buf: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *zml.Bufferized(attention.Metadata),
};

pub const CompiledModel = struct {
    loaded_model: *const model.LoadedModel,
    prefill: KernelExe,
    decode: KernelExe,
    params: CompilationParameters,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const model.LoadedModel,
        llama_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        if (parameters.single) {
            const prefill_exe, const decode_exe = b: {
                var prefill_future = io.async(compileSingleKernelExe, .{ allocator, io, platform, llama_model, parameters, parameters.prefill_tokens, "prefill", progress });
                errdefer if (prefill_future.cancel(io)) |e| e.deinit() else |_| {};

                var decode_future = io.async(compileSingleKernelExe, .{ allocator, io, platform, llama_model, parameters, parameters.decode_tokens, "decode", progress });
                errdefer if (decode_future.cancel(io)) |e| e.deinit() else |_| {};

                break :b .{
                    prefill_future.await(io) catch unreachable,
                    decode_future.await(io) catch unreachable,
                };
            };

            return .{
                .loaded_model = loaded_model,
                .prefill = .{ .single = prefill_exe },
                .decode = .{ .single = decode_exe },
                .params = parameters,
            };
        } else {
            const prefill_exe, const decode_exe = b: {
                var prefill_future = io.async(compileComposedKernelExe, .{ allocator, io, platform, llama_model, parameters, parameters.seqlen, "prefill", progress });
                errdefer if (prefill_future.cancel(io)) |e| e.deinit() else |_| {};

                var decode_future = io.async(compileComposedKernelExe, .{ allocator, io, platform, llama_model, parameters, @as(u32, 1), "decode", progress });
                errdefer if (decode_future.cancel(io)) |e| e.deinit() else |_| {};

                break :b .{
                    prefill_future.await(io) catch unreachable,
                    decode_future.await(io) catch unreachable,
                };
            };

            return .{
                .loaded_model = loaded_model,
                .prefill = .{ .composed = prefill_exe },
                .decode = .{ .composed = decode_exe },
                .params = parameters,
            };
        }
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

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
            args.token_index_buf,
            args.kv_cache_buffers,
            args.rng_buf,
            args.attention_metadata_buffers,
        });
        self.exe.call(exe_args, &results);

        results.fill(.{ args.tokens_buf, args.kv_cache_buffers, args.rng_buf });
    }
};

fn compileSingleKernelExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    tokens: zml.Tensor,
    label: []const u8,
    progress: *std.Progress.Node,
) !SingleKernelExe {
    progress.increaseEstimatedTotalItems(1);
    var name_buf: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "Compiling {s}...", .{label}) catch "Compiling...";
    var node = progress.start(name, 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} [{f}]", .{ label, now.untilNow(io, .awake) });

    const all_shardings = parameters.shardings.all();
    const exe = try platform.compile(
        allocator,
        io,
        llama_model,
        .forward,
        .{
            tokens,
            parameters.token_index,
            parameters.kv_cache,
            parameters.rng,
            parameters.attention_metadata,
            parameters.attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

pub const ComposedKernelExe = struct {
    embed_tokens: zml.Exe,
    transformer_layer: zml.Exe,
    lm_head: zml.Exe,

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        self.transformer_layer.deinit();
        self.lm_head.deinit();
    }

    pub fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = b: {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers.model.embed_tokens, args.tokens_buf });
            self.embed_tokens.call(exe_args, &results);
            break :b results.get(zml.Buffer);
        };
        defer hidden_buf.deinit();

        const replicated_sharding = try zml.sharding.replicatedSharding(args.platform);
        var layer_index_buf: zml.Buffer = try .scalar(args.io, args.platform, @as(u32, 0), .u32, replicated_sharding);
        defer layer_index_buf.deinit();

        for (args.model_buffers.model.layers) |layer_bufs| {
            var exe_args = try self.transformer_layer.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.transformer_layer.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{
                layer_bufs,
                &hidden_buf,
                args.token_index_buf,
                &layer_index_buf,
                args.kv_cache_buffers,
                args.attention_metadata_buffers,
            });
            self.transformer_layer.call(exe_args, &results);

            var new_hidden, var new_kv_cache, var new_layer_index = results.get(struct {
                zml.Buffer,
                zml.Bufferized(model.KvCache),
                zml.Buffer,
            });
            replaceBuffer(&hidden_buf, &new_hidden);
            replaceKvCacheBuffers(args.kv_cache_buffers, &new_kv_cache);
            replaceBuffer(&layer_index_buf, &new_layer_index);
        }

        {
            var exe_args = try self.lm_head.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            var results = try self.lm_head.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{ args.model_buffers, &hidden_buf, args.tokens_buf, args.rng_buf });
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

fn replaceKvCacheBuffers(dst: *zml.Bufferized(model.KvCache), src: *zml.Bufferized(model.KvCache)) void {
    replaceBuffer(&dst.k, &src.k);
    replaceBuffer(&dst.v, &src.v);
    replaceBuffer(&dst.layer_index, &src.layer_index);
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
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    seqlen: u32,
    label: []const u8,
    progress: *std.Progress.Node,
) !ComposedKernelExe {
    const embed_exe, const layer_exe, const lm_head_exe = b: {
        var embed_future = io.async(compileEmbedTokens, .{ allocator, io, platform, llama_model, parameters, seqlen, label, progress });
        errdefer if (embed_future.cancel(io)) |e| e.deinit() else |_| {};

        var layer_future = io.async(compileTransformerLayer, .{ allocator, io, platform, llama_model, parameters, seqlen, label, progress });
        errdefer if (layer_future.cancel(io)) |e| e.deinit() else |_| {};

        var lm_head_future = io.async(compileLmHead, .{ allocator, io, platform, llama_model, parameters, seqlen, label, progress });
        errdefer if (lm_head_future.cancel(io)) |e| e.deinit() else |_| {};

        break :b .{
            embed_future.await(io) catch unreachable,
            layer_future.await(io) catch unreachable,
            lm_head_future.await(io) catch unreachable,
        };
    };

    return .{
        .embed_tokens = embed_exe,
        .transformer_layer = layer_exe,
        .lm_head = lm_head_exe,
    };
}

fn compileEmbedTokens(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    seqlen: u32,
    label: []const u8,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var name_buf: [96]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "Compiling {s} embed_tokens...", .{label}) catch "Compiling embed_tokens...";
    var node = progress.start(name, 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} embed_tokens [{f}]", .{ label, now.untilNow(io, .awake) });

    const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);
    const all_shardings = parameters.shardings.all();
    return platform.compile(allocator, io, llama_model.model.embed_tokens, .forward, .{tokens}, .{ .shardings = &all_shardings });
}

fn compileTransformerLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    seqlen: u32,
    label: []const u8,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var name_buf: [96]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "Compiling {s} transformer layer...", .{label}) catch "Compiling transformer layer...";
    var node = progress.start(name, 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} transformer layer [{f}]", .{ label, now.untilNow(io, .awake) });

    const hidden: zml.Tensor = .init(.{ .s = seqlen, .d = parameters.hidden_size }, llama_model.model.embed_tokens.weight.dtype());
    const token_index: zml.Tensor = .init(.{}, .u32);
    const layer_index: zml.Tensor = .init(.{}, .u32);

    const all_shardings = parameters.shardings.all();
    return platform.compile(allocator, io, llama_model.model.layers[0], .forwardComposed, .{
        hidden,
        token_index,
        layer_index,
        parameters.kv_cache,
        parameters.attention_metadata,
        parameters.attention_parameters,
    }, .{ .shardings = &all_shardings });
}

fn compileLmHead(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    llama_model: model.Model,
    parameters: CompilationParameters,
    seqlen: u32,
    label: []const u8,
    progress: *std.Progress.Node,
) !zml.Exe {
    progress.increaseEstimatedTotalItems(1);
    var name_buf: [96]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "Compiling {s} lm_head...", .{label}) catch "Compiling lm_head...";
    var node = progress.start(name, 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled {s} lm_head [{f}]", .{ label, now.untilNow(io, .awake) });

    const hidden: zml.Tensor = .init(.{ .s = seqlen, .d = parameters.hidden_size }, llama_model.model.embed_tokens.weight.dtype());
    const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);
    const all_shardings = parameters.shardings.all();
    return platform.compile(allocator, io, llama_model, .finalize, .{ hidden, tokens, parameters.rng }, .{ .shardings = &all_shardings });
}
