const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.step3_5flash);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: zml.attention.attention.Metadata,
    prefill_attention_parameters: zml.attention.attention.Parameters,
    decode_attention_parameters: zml.attention.attention.Parameters,
    seqlen: u32,
    shardings: common.Shardings,

    pub fn init(
        mdl: model.Model,
        config: model.Config,
        seqlen: u32,
        backend: zml.attention.attention.Backend,
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
            .prefill_tokens = .init(.{ .b = 1, .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .b = 1, .s = 1 }, .u32),
            .token_index = .init(.{ .s = 1 }, .u32),
            .kv_cache = .init(kv_shape),
            .rng = .init(),
            .attention_metadata = .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            .prefill_attention_parameters = .init(.fromBackend(backend)),
            .decode_attention_parameters = .init(.fromBackend(backend)),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const CompilationOptions = CompilationParameters;

const LayerExeKind = .{
    .dense_full,
    .dense_sliding,
    .moe_full,
    .moe_sliding,
    .moe_sliding_with_limit,
    .moe_sliding_with_limit_shared,
};

pub const Args = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
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
        step3p5_model: model.Model,
        parameters: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        return .{
            .loaded_model = loaded_model,
            .prefill = try .init(
                allocator,
                io,
                platform,
                step3p5_model,
                parameters,
                @intCast(parameters.prefill_tokens.dim(.s)),
                .prefill,
                progress,
            ),
            .decode = try .init(
                allocator,
                io,
                platform,
                step3p5_model,
                parameters,
                @intCast(parameters.decode_tokens.dim(.s)),
                .decode,
                progress,
            ),
            .params = parameters,
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

    pub const Runner = struct {
        exe: *const ComposedKernelExe,
        embed_args: zml.exe.Exe.Arguments,
        embed_results: zml.exe.Exe.Results,
        layers: Layers,
        sampler_args: zml.exe.Exe.Arguments,
        sampler_results: zml.exe.Exe.Results,

        const Layers = struct {
            args: []zml.exe.Exe.Arguments,
            results: []zml.exe.Exe.Results,
            layer_indices: []zml.Buffer,
            layer_kinds: []const LayerExeKind,

            fn init(
                allocator: std.mem.Allocator,
                io: std.Io,
                platform: *const zml.Platform,
                exe: *const ComposedKernelExe,
                model_buffers: *model.Buffers,
            ) !Layers {
                const args = try allocator.alloc(zml.exe.Exe.Arguments, model_buffers.text_model.layers.len);
                errdefer allocator.free(args);

                const results = try allocator.alloc(zml.exe.Exe.Results, model_buffers.text_model.layers.len);
                errdefer allocator.free(results);

                const layer_indices = try allocator.alloc(zml.Buffer, model_buffers.text_model.layers.len);
                errdefer allocator.free(layer_indices);

                var initialized_args: usize = 0;
                errdefer {
                    for (args[0..initialized_args]) |*exe_args| exe_args.deinit(allocator);
                }

                var initialized_results: usize = 0;
                errdefer {
                    for (results[0..initialized_results]) |*exe_results| exe_results.deinit(allocator);
                }

                var initialized_layer_indices: usize = 0;
                errdefer {
                    for (layer_indices[0..initialized_layer_indices]) |*layer_index| layer_index.deinit();
                }

                const layer_types = exe.layer_kinds[0..model_buffers.text_model.layers.len];
                for (model_buffers.text_model.layers, layer_types, 0..) |layer_bufs, layer_kind, i| {
                    args[i] = try exe.layerExe(layer_kind).args(allocator);
                    initialized_args = i + 1;
                    args[i].bake(layer_bufs);

                    results[i] = try exe.layerExe(layer_kind).results(allocator);
                    initialized_results = i + 1;

                    layer_indices[i] = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(i)), .u32);
                    initialized_layer_indices = i + 1;
                }

                return .{
                    .args = args,
                    .results = results,
                    .layer_indices = layer_indices,
                    .layer_kinds = layer_types,
                };
            }

            fn deinit(self: *Layers, allocator: std.mem.Allocator) void {
                for (self.args) |*exe_args| exe_args.deinit(allocator);
                allocator.free(self.args);

                for (self.results) |*exe_results| exe_results.deinit(allocator);
                allocator.free(self.results);

                for (self.layer_indices) |*layer_index| layer_index.deinit();
                allocator.free(self.layer_indices);
            }
        };

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            exe: *const ComposedKernelExe,
            model_buffers: *model.Buffers,
        ) !Runner {
            var embed_args = try exe.embed_tokens.args(allocator);
            errdefer embed_args.deinit(allocator);
            embed_args.bake(ComposedKernelExe.embedTokensBuffers(model_buffers));

            var embed_results = try exe.embed_tokens.results(allocator);
            errdefer embed_results.deinit(allocator);

            var layers = try Layers.init(allocator, io, platform, exe, model_buffers);
            errdefer layers.deinit(allocator);

            var sampler_args = try exe.sampler.args(allocator);
            errdefer sampler_args.deinit(allocator);
            sampler_args.bake(ComposedKernelExe.samplerBuffers(model_buffers));

            var sampler_results = try exe.sampler.results(allocator);
            errdefer sampler_results.deinit(allocator);

            return .{
                .exe = exe,
                .embed_args = embed_args,
                .embed_results = embed_results,
                .layers = layers,
                .sampler_args = sampler_args,
                .sampler_results = sampler_results,
            };
        }

        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.embed_args.deinit(allocator);
            self.embed_results.deinit(allocator);
            self.layers.deinit(allocator);
            self.sampler_args.deinit(allocator);
            self.sampler_results.deinit(allocator);
        }

        pub fn run(self: *Runner, args: Args) !void {
            var hidden_buf: zml.Buffer = b: {
                self.embed_args.set(.{args.tokens_buf});
                self.exe.embed_tokens.call(self.embed_args, &self.embed_results);
                break :b self.embed_results.get(zml.Buffer);
            };
            defer hidden_buf.deinit();

            for (
                self.layers.args,
                self.layers.results,
                self.layers.layer_indices,
                self.layers.layer_kinds,
            ) |*exe_args, *results, *layer_index_buf, layer_kind| {
                self.exe.runLayer(exe_args, results, layer_kind, args, &hidden_buf, layer_index_buf);
            }

            self.exe.runSampler(&self.sampler_args, &self.sampler_results, args, &hidden_buf);
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        step3p5_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !KernelExe {
        return .{
            .composed = try .init(allocator, io, platform, step3p5_model, parameters, seqlen, phase, progress),
        };
    }

    pub fn deinit(self: KernelExe) void {
        self.composed.deinit();
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        try self.composed.run(args);
    }

    pub fn initRunner(
        self: *const KernelExe,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model_buffers: *model.Buffers,
    ) !Runner {
        return .init(allocator, io, platform, &self.composed, model_buffers);
    }
};

const ComposedKernelExe = struct {
    allocator: std.mem.Allocator,
    embed_tokens: zml.Exe,
    dense_full_layer: ?zml.Exe,
    dense_sliding_layer: ?zml.Exe,
    moe_full_layer: ?zml.Exe,
    moe_sliding_layer: ?zml.Exe,
    moe_sliding_with_limit_layer: ?zml.Exe,
    moe_sliding_with_limit_shared_layer: ?zml.Exe,
    sampler: zml.Exe,
    phase: Phase,

    const EmbedTokens = struct {
        embed_tokens: zml.nn.TokenEmbedding,

        pub fn forward(self: EmbedTokens, tokens_: zml.Tensor) zml.Tensor {
            const tokens = tokens_.withPartialTags(.{.s});
            return self.embed_tokens.forward(tokens)
                .withPartialTags(.{.d})
                .withPartitioning(.{ .d = .replicated });
        }
    };

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        step3p5_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !ComposedKernelExe {
        const embed_tokens = try compileEmbedTokens(allocator, io, platform, step3p5_model.text_model.embed_tokens, parameters, seqlen, phase, progress);
        errdefer embed_tokens.deinit();

        // We compile constants and parameters into layer executables. We require an example for each layer kind.
        // For now, I have collected the indices of said examples.
        const dense_full_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, 0, phase, .{ .attn = .full_attention, .ffn = .mlp }, progress);
        errdefer if (dense_full_layer) |exe| exe.deinit();

        const dense_sliding_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, 1, phase, .{ .attn = .sliding_attention, .ffn = .mlp }, progress);
        errdefer if (dense_sliding_layer) |exe| exe.deinit();

        const moe_full_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, 3, phase, .{ .attn = .full_attention, .ffn = .moe }, progress);
        errdefer if (moe_full_layer) |exe| exe.deinit();

        const moe_sliding_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, 4, phase, .{ .attn = .sliding_attention, .ffn = .moe }, progress);
        errdefer if (moe_sliding_layer) |exe| exe.deinit();

        const moe_sliding_with_limit_layer = compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, 42, phase, .{ .attn = .full_attention, .ffn = .moe }, progress);
        errdefer if (moe_sliding_with_limit_layer) |exe| exe.deinit();

        const moe_sliding_with_limit_shared_layer = compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, 43, phase, .{ .attn = .full_attention, .ffn = .moe }, progress);
        errdefer if (moe_sliding_with_limit_shared_layer) |exe| exe.deinit();

        const sampler = try compileSampler(allocator, io, platform, step3p5_model, parameters, seqlen, phase, progress);
        errdefer sampler.deinit();

        return .{
            .allocator = allocator,
            .embed_tokens = embed_tokens,
            .dense_full_layer = dense_full_layer,
            .dense_sliding_layer = dense_sliding_layer,
            .moe_full_layer = moe_full_layer,
            .moe_sliding_layer = moe_sliding_layer,
            .moe_sliding_with_limit_layer = moe_sliding_with_limit_layer,
            .moe_sliding_with_limit_shared_layer = moe_sliding_with_limit_shared_layer,
            .sampler = sampler,
            .phase = phase,
        };
    }

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        if (self.full_mlp_layer) |exe| exe.deinit();
        if (self.sliding_mlp_layer) |exe| exe.deinit();
        if (self.full_moe_layer) |exe| exe.deinit();
        if (self.sliding_moe_layer) |exe| exe.deinit();
        self.sampler.deinit();
        self.allocator.free(self.layer_kinds);
    }

    fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = b: {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            exe_args.bake(embedTokensBuffers(args.model_buffers));

            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.set(.{args.tokens_buf});
            self.embed_tokens.call(exe_args, &results);
            break :b results.get(zml.Buffer);
        };
        defer hidden_buf.deinit();

        for (args.model_buffers.text_model.layers, self.layer_kinds, 0..) |layer_bufs, layer_kind, layer_index| {
            var exe_args = try self.layerExe(layer_kind).args(args.allocator);
            defer exe_args.deinit(args.allocator);
            exe_args.bake(layer_bufs);

            var results = try self.layerExe(layer_kind).results(args.allocator);
            defer results.deinit(args.allocator);

            var layer_index_buf = try zml.Buffer.scalar(args.io, args.platform, @as(u32, @intCast(layer_index)), .u32);
            defer layer_index_buf.deinit();

            self.runLayer(&exe_args, &results, layer_kind, args, &hidden_buf, &layer_index_buf);
        }

        {
            var exe_args = try self.sampler.args(args.allocator);
            defer exe_args.deinit(args.allocator);
            exe_args.bake(samplerBuffers(args.model_buffers));

            var results = try self.sampler.results(args.allocator);
            defer results.deinit(args.allocator);

            self.runSampler(&exe_args, &results, args, &hidden_buf);
        }
    }

    fn runLayer(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        layer_kind: LayerExeKind,
        args: Args,
        hidden_buf: *zml.Buffer,
        layer_index_buf: *zml.Buffer,
    ) void {
        const layer_cache: zml.Bufferized(model.KvCache) = .{
            .k = args.kv_cache_buffers.k,
            .v = args.kv_cache_buffers.v,
            .layer_index = layer_index_buf.*,
        };
        exe_args.set(.{ hidden_buf, args.token_index_buf, layer_cache });

        self.layerExe(layer_kind).call(exe_args.*, results);

        var new_hidden, var new_cache = results.get(struct { zml.Buffer, zml.Bufferized(model.KvCache) });
        replaceBuffer(hidden_buf, &new_hidden);
        replaceBuffer(&args.kv_cache_buffers.k, &new_cache.k);
        replaceBuffer(&args.kv_cache_buffers.v, &new_cache.v);
        releaseBuffer(layer_index_buf.*, &new_cache.layer_index);
    }

    fn runSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        switch (self.phase) {
            .prefill => self.runPrefillSampler(exe_args, results, args, hidden_buf),
            .decode => self.runDecodeSampler(exe_args, results, args, hidden_buf),
        }
    }

    fn runPrefillSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{ hidden_buf, args.rng_buffers, null });
        self.sampler.call(exe_args.*, results);

        var new_tokens, var new_rng = results.get(struct { zml.Buffer, zml.Bufferized(zml.Tensor.Rng) });
        replaceBuffer(args.tokens_buf, &new_tokens);
        replaceBuffer(&args.rng_buffers._state, &new_rng._state);
    }

    fn runDecodeSampler(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{ hidden_buf, args.rng_buffers, args.token_index_buf });
        self.sampler.call(exe_args.*, results);

        var new_tokens, var new_rng, var new_token_index = results.get(struct { zml.Buffer, zml.Bufferized(zml.Tensor.Rng), zml.Buffer });
        replaceBuffer(args.tokens_buf, &new_tokens);
        replaceBuffer(&args.rng_buffers._state, &new_rng._state);
        replaceBuffer(args.token_index_buf, &new_token_index);
    }

    fn embedTokensBuffers(model_buffers: *const model.Buffers) zml.Bufferized(EmbedTokens) {
        return .{ .embed_tokens = model_buffers.text_model.embed_tokens };
    }

    fn samplerBuffers(model_buffers: *const model.Buffers) zml.Bufferized(model.Sampler) {
        return .{
            .norm = model_buffers.text_model.norm,
            .lm_head = model_buffers.lm_head,
        };
    }

    fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
        if (!sameBufferHandle(dst.*, src.*)) dst.deinit();
        dst.* = src.*;
    }

    fn releaseBuffer(expected: zml.Buffer, actual: *zml.Buffer) void {
        if (!sameBufferHandle(expected, actual.*)) actual.deinit();
    }

    fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
        if (a._shards.len != b._shards.len) return false;
        for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
            if (a_shard != b_shard) return false;
        }
        return true;
    }

    fn layerExe(self: *const ComposedKernelExe, kind: LayerExeKind) zml.Exe {
        return switch (kind.attn) {
            .full_attention => switch (kind.ffn) {
                .mlp => self.full_mlp_layer orelse unreachable,
                .moe => self.full_moe_layer orelse unreachable,
            },
            .sliding_attention => switch (kind.ffn) {
                .mlp => self.sliding_mlp_layer orelse unreachable,
                .moe => self.sliding_moe_layer orelse unreachable,
            },
        };
    }

    fn compileEmbedTokens(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        embed_tokens: zml.nn.TokenEmbedding,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("embed tokens"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "embed tokens", io, from);

        const tokens: zml.Tensor = .init(.{ .b = 1, .s = seqlen }, .u32);
        return platform.compile(allocator, io, EmbedTokens{ .embed_tokens = embed_tokens }, .forward, .{tokens}, .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("step3_5flash", "embed_tokens"),
        });
    }

    fn inferLayerKind(step3p5_model: model.Model, layer_index: usize) LayerExeKind {
        const layer = step3p5_model.layers[layer_index];

        const attention_kind = if (layer.attn.enable_sliding_window)
            .sliding_attention
        else
            .full_attention;

        var swiglu_limit: ?f32 = null;
        var shared_limit: ?f32 = null;
        const ffn_kind = switch (layer.ffn) {
            .mlp => |mlp| blk: {
                swiglu_limit = mlp.limit;
                break :blk .dense;
            },
            .moe => |moe| blk: {
                swiglu_limit = moe.limit;
                shared_limit = moe.shared_limit; // is this used
                break :blk .moe;
            },
        };

        // config lives under parameters.config
        const key = .{
            ffn_kind,
            attention_kind,
            swiglu_limit != null,
            shared_limit != null,
        };

        return switch (key) {
            .{ .dense, .full_attention, false, false } => .dense_full,
            .{ .dense, .sliding_attention, false, false } => .dense_sliding,
            .{ .moe, .full_attention, false, false } => .moe_full,
            .{ .moe, .sliding_attention, false, false } => .moe_sliding,
            .{ .moe, .sliding_attention, true, false } => .moe_sliding_with_limit,
            .{ .moe, .sliding_attention, true, true } => .moe_sliding_with_limit_shared,
        };
    }

    fn compileLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        step3p5_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        layer_index: usize,
        phase: Phase,
        comptime component: []const u8,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage(component), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, component, io, from);

        const hidden_tensor = hidden(step3p5_model, seqlen);
        const layer_cache: model.KvCache = .{
            .k = parameters.kv_cache.k,
            .v = parameters.kv_cache.v,
            .layer_index = zml.Tensor.init(.{}, .u32),
        };

        const attention_parameters = switch (phase) {
            .prefill => parameters.prefill_attention_parameters,
            .decode => parameters.decode_attention_parameters,
        }; // autofix

        return platform.compile(allocator, io, step3p5_model.text_model.layers[layer_index], .forward, .{
            hidden_tensor,
            parameters.token_index,
            layer_cache,
            parameters.attention_metadata,
            attention_parameters,
        }, .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("step3_5flash", component),
        });
    }

    fn compileSampler(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        step3p5_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("sampler"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "sampler", io, from);

        const token_index: ?zml.Tensor = switch (phase) {
            .prefill => null,
            .decode => parameters.token_index,
        };

        return platform.compile(
            allocator,
            io,
            step3p5_model.sampler(),
            .sampleTokens,
            .{ hidden(step3p5_model, seqlen), parameters.rng, token_index },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("step3_5flash", "sampler"),
            },
        );
    }

    fn hidden(step3p5_model: model.Model, seqlen: usize) zml.Tensor {
        return .fromShape(zml.Shape.init(
            .{ .b = 1, .s = seqlen, .d = step3p5_model.config.hidden_size },
            step3p5_model.text_model.embed_tokens.weight.dtype(),
        ).withPartitioning(.{
            .b = .replicated,
            .s = .replicated,
            .d = .replicated,
        }));
    }
};
