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

// const LayerKind = struct {
//     attn: model.AttnType,
//     ffn: model.FfnType,
// };

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
            layer_kinds: []const LayerKind,

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
    embed_tokens: zml.Exe,
    full_mlp_layer: ?zml.Exe,
    sliding_mlp_layer: ?zml.Exe,
    full_moe_layer: ?zml.Exe,
    sliding_moe_layer: ?zml.Exe,
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

        const layer_kinds = try layerKinds(allocator, step3p5_model.config, step3p5_model.text_model.layers.len);
        errdefer allocator.free(layer_kinds);

        const full_mlp_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, phase, .{ .attn = .full_attention, .ffn = .mlp }, "full_mlp_layer", progress);
        errdefer if (full_mlp_layer) |exe| exe.deinit();

        const sliding_mlp_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, phase, .{ .attn = .sliding_attention, .ffn = .mlp }, "sliding_mlp_layer", progress);
        errdefer if (sliding_mlp_layer) |exe| exe.deinit();

        const full_moe_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, phase, .{ .attn = .full_attention, .ffn = .moe }, "full_moe_layer", progress);
        errdefer if (full_moe_layer) |exe| exe.deinit();

        const sliding_moe_layer = try compileLayer(allocator, io, platform, step3p5_model, parameters, seqlen, phase, .{ .attn = .sliding_attention, .ffn = .moe }, "sliding_moe_layer", progress);
        errdefer if (sliding_moe_layer) |exe| exe.deinit();

        const sampler = try compileSampler(allocator, io, platform, step3p5_model, parameters, seqlen, phase, progress);
        errdefer sampler.deinit();

        return .{
            .embed_tokens = embed_tokens,
            .full_mlp_layer = full_mlp_layer,
            .sliding_mlp_layer = sliding_mlp_layer,
            .full_moe_layer = full_moe_layer,
            .sliding_moe_layer = sliding_moe_layer,
            .sampler = sampler,
            .layer_kinds = layer_kinds,
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
        layer_kind: LayerKind,
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
        _ = new_cache; // autofix
        _ = new_hidden; // autofix
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
        };

        return platform.compile(
            allocator,
            io,
            step3p5_model.text_model.layers[layer_index],
            .forward,
            .{ hidden_tensor, parameters.token_index, layer_cache, parameters.attention_metadata, attention_parameters },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("step3_5flash", component),
            },
        );
    }
};
