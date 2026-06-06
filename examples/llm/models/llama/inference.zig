const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("../common.zig");
const model = @import("model.zig");

const log = std.log.scoped(.llama);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: model.KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: attention.Metadata,
    prefill_attention_parameters: attention.Parameters,
    decode_attention_parameters: attention.Parameters,
    seqlen: usize,
    shardings: common.Shardings,

    pub fn init(mdl: model.Model, config: model.Config, seqlen: u32, backend: attention.Backend, shardings: common.Shardings) CompilationParameters {
        const head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads);

        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(.init(.{
                .layer = mdl.model.layers.len,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = head_dim,
            }, mdl.model.embed_tokens.weight.dtype())),
            .rng = .init(),
            .attention_metadata = switch (backend) {
                .attnd => .{ .attnd = .init() },
                else => .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            },
            .prefill_attention_parameters = switch (backend) {
                .attnd => .{ .attnd = .init(.{
                    .model_id = .@"llama-3.1-8B",
                    .head_dim = head_dim,
                    .num_attention_heads = config.num_attention_heads,
                    .num_kv_heads = @intCast(config.num_key_value_heads),
                    .is_prefill = true,
                }) },
                else => .init(.fromBackend(backend)),
            },
            .decode_attention_parameters = switch (backend) {
                .attnd => .{ .attnd = .init(.{
                    .model_id = .@"llama-3.1-8B",
                    .head_dim = head_dim,
                    .num_attention_heads = config.num_attention_heads,
                    .num_kv_heads = @intCast(config.num_key_value_heads),
                    .is_prefill = false,
                }) },
                else => .init(.fromBackend(backend)),
            },
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
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(model.KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(attention.Metadata),
    // Index of the last real prompt token (num_tokens - 1). Only consumed by the
    // PREFILL lm_head, which applies the output projection to that single position
    // instead of all `seqlen` positions (the rest are never read). Ignored by decode.
    last_token_index_buf: *zml.Buffer,
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
        return .{
            .loaded_model = loaded_model,
            .prefill = try KernelExe.init(allocator, io, platform, llama_model, parameters, @intCast(parameters.prefill_tokens.dim(.s)), parameters.prefill_attention_parameters, .prefill, progress),
            .decode = try KernelExe.init(allocator, io, platform, llama_model, parameters, @intCast(parameters.decode_tokens.dim(.s)), parameters.decode_attention_parameters, .decode, progress),
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
        lm_head_args: zml.exe.Exe.Arguments,
        lm_head_results: zml.exe.Exe.Results,

        const Layers = struct {
            args: []zml.exe.Exe.Arguments,
            results: []zml.exe.Exe.Results,
            kv_cache_indices: []zml.Buffer,

            fn init(
                allocator: std.mem.Allocator,
                io: std.Io,
                platform: *const zml.Platform,
                exe: *const ComposedKernelExe,
                model_buffers: *model.Buffers,
            ) !Layers {
                const args = try allocator.alloc(zml.exe.Exe.Arguments, model_buffers.model.layers.len);
                errdefer allocator.free(args);

                const results = try allocator.alloc(zml.exe.Exe.Results, model_buffers.model.layers.len);
                errdefer allocator.free(results);

                const kv_cache_indices = try allocator.alloc(zml.Buffer, model_buffers.model.layers.len);
                errdefer allocator.free(kv_cache_indices);

                var initialized_args: usize = 0;
                errdefer {
                    for (args[0..initialized_args]) |*exe_args| {
                        exe_args.deinit(allocator);
                    }
                }

                var initialized_results: usize = 0;
                errdefer {
                    for (results[0..initialized_results]) |*exe_results| {
                        exe_results.deinit(allocator);
                    }
                }

                var initialized_kv_cache_indices: usize = 0;
                errdefer {
                    for (kv_cache_indices[0..initialized_kv_cache_indices]) |*kv_cache_index| {
                        kv_cache_index.deinit();
                    }
                }

                for (model_buffers.model.layers, 0..) |layer_bufs, i| {
                    args[i] = try exe.layer.args(allocator);
                    initialized_args = i + 1;
                    args[i].bake(layer_bufs);

                    results[i] = try exe.layer.results(allocator);
                    initialized_results = i + 1;

                    kv_cache_indices[i] = try zml.Buffer.scalar(io, platform, i, .u32);
                    initialized_kv_cache_indices = i + 1;
                }

                return .{ .args = args, .results = results, .kv_cache_indices = kv_cache_indices };
            }

            fn deinit(self: *Layers, allocator: std.mem.Allocator) void {
                for (self.args) |*exe_args| {
                    exe_args.deinit(allocator);
                }
                allocator.free(self.args);

                for (self.results) |*exe_results| {
                    exe_results.deinit(allocator);
                }
                allocator.free(self.results);

                for (self.kv_cache_indices) |*kv_cache_index| {
                    kv_cache_index.deinit();
                }
                allocator.free(self.kv_cache_indices);
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

            var lm_head_args = try exe.lm_head.args(allocator);
            errdefer lm_head_args.deinit(allocator);

            lm_head_args.bake(ComposedKernelExe.lmHeadBuffers(model_buffers));

            var lm_head_results = try exe.lm_head.results(allocator);
            errdefer lm_head_results.deinit(allocator);

            return .{
                .exe = exe,
                .embed_args = embed_args,
                .embed_results = embed_results,
                .layers = layers,
                .lm_head_args = lm_head_args,
                .lm_head_results = lm_head_results,
            };
        }

        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.embed_args.deinit(allocator);
            self.embed_results.deinit(allocator);
            self.layers.deinit(allocator);
            self.lm_head_args.deinit(allocator);
            self.lm_head_results.deinit(allocator);
        }

        pub fn run(self: *Runner, args: Args) !void {
            var hidden_buf: zml.Buffer = b: {
                self.embed_args.set(.{args.tokens_buf});
                self.exe.embed_tokens.call(self.embed_args, &self.embed_results);

                break :b self.embed_results.get(zml.Buffer);
            };
            defer hidden_buf.deinit();

            for (self.layers.args, self.layers.results, self.layers.kv_cache_indices) |*exe_args, *results, *kv_cache_index_buf| {
                self.exe.runLayer(exe_args, results, args, &hidden_buf, kv_cache_index_buf);
            }

            self.exe.runLmHead(&self.lm_head_args, &self.lm_head_results, args, &hidden_buf);
        }
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        llama_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        attention_parameters: attention.Parameters,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !KernelExe {
        return .{
            .composed = try ComposedKernelExe.init(allocator, io, platform, llama_model, parameters, seqlen, attention_parameters, phase, progress),
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
    layer: zml.Exe,
    lm_head: zml.Exe,
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
        llama_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        attention_parameters: attention.Parameters,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !ComposedKernelExe {
        const embed_tokens = try ComposedKernelExe.compileEmbedTokens(allocator, io, platform, llama_model.model.embed_tokens, parameters, seqlen, phase, progress);
        errdefer embed_tokens.deinit();

        const layer = try ComposedKernelExe.compileLayer(allocator, io, platform, llama_model, parameters, seqlen, attention_parameters, phase, progress);
        errdefer layer.deinit();

        const lm_head = try ComposedKernelExe.compileLmHead(allocator, io, platform, llama_model, parameters, seqlen, phase, progress);
        errdefer lm_head.deinit();

        return .{
            .embed_tokens = embed_tokens,
            .layer = layer,
            .lm_head = lm_head,
            .phase = phase,
        };
    }

    fn deinit(self: ComposedKernelExe) void {
        self.embed_tokens.deinit();
        self.layer.deinit();
        self.lm_head.deinit();
    }

    fn run(self: *const ComposedKernelExe, args: Args) !void {
        var hidden_buf: zml.Buffer = b: {
            var exe_args = try self.embed_tokens.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.embed_tokens.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.bake(ComposedKernelExe.embedTokensBuffers(args.model_buffers));
            exe_args.set(.{args.tokens_buf});

            self.embed_tokens.call(exe_args, &results);

            break :b results.get(zml.Buffer);
        };
        defer hidden_buf.deinit();

        for (args.model_buffers.model.layers, 0..) |layer_bufs, i| {
            var exe_args = try self.layer.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.layer.results(args.allocator);
            defer results.deinit(args.allocator);

            var kv_cache_index_buf: zml.Buffer = try .scalar(args.io, args.platform, i, .u32);
            defer kv_cache_index_buf.deinit();

            exe_args.bake(layer_bufs);

            self.runLayer(&exe_args, &results, args, &hidden_buf, &kv_cache_index_buf);
        }

        {
            var exe_args = try self.lm_head.args(args.allocator);
            defer exe_args.deinit(args.allocator);

            var results = try self.lm_head.results(args.allocator);
            defer results.deinit(args.allocator);

            exe_args.bake(ComposedKernelExe.lmHeadBuffers(args.model_buffers));

            self.runLmHead(&exe_args, &results, args, &hidden_buf);
        }
    }

    fn runLayer(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
        kv_cache_index_buf: *zml.Buffer,
    ) void {
        exe_args.set(.{
            hidden_buf,
            args.token_index_buf,
            args.kv_cache_buffers,
            kv_cache_index_buf,
            args.attention_metadata_buffers,
        });

        self.layer.call(exe_args.*, results);

        var new_hidden, var new_kv_cache = results.get(struct {
            zml.Buffer,
            zml.Bufferized(model.KvCache),
        });

        ComposedKernelExe.replaceBuffer(hidden_buf, &new_hidden);
        ComposedKernelExe.replaceKvCacheBuffers(args.kv_cache_buffers, &new_kv_cache);
    }

    fn runLmHead(
        self: *const ComposedKernelExe,
        exe_args: *zml.exe.Exe.Arguments,
        results: *zml.exe.Exe.Results,
        args: Args,
        hidden_buf: *zml.Buffer,
    ) void {
        // Prefill applies the lm_head to only the last real token (index given by
        // last_token_index_buf); decode runs over its single position as before.
        switch (self.phase) {
            .prefill => exe_args.set(.{ hidden_buf, args.tokens_buf, args.rng_buffers, args.last_token_index_buf }),
            .decode => exe_args.set(.{ hidden_buf, args.tokens_buf, args.rng_buffers }),
        }
        self.lm_head.call(exe_args.*, results);

        var new_tokens, var new_rng = results.get(struct {
            zml.Buffer,
            zml.Bufferized(zml.Tensor.Rng),
        });

        ComposedKernelExe.replaceBuffer(args.tokens_buf, &new_tokens);
        ComposedKernelExe.replaceBuffer(&args.rng_buffers._state, &new_rng._state);
    }

    fn embedTokensBuffers(model_buffers: *const model.Buffers) zml.Bufferized(EmbedTokens) {
        return .{
            .embed_tokens = model_buffers.model.embed_tokens,
        };
    }

    fn lmHeadBuffers(model_buffers: *const model.Buffers) zml.Bufferized(model.LmHead) {
        return .{
            .lm_head = model_buffers.lm_head,
            .embed_tokens = model_buffers.model.embed_tokens,
            .norm = model_buffers.model.norm,
        };
    }

    fn replaceKvCacheBuffers(dst: *zml.Bufferized(model.KvCache), src: *zml.Bufferized(model.KvCache)) void {
        ComposedKernelExe.replaceBuffer(&dst.k, &src.k);
        ComposedKernelExe.replaceBuffer(&dst.v, &src.v);
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
        platform: *const zml.Platform,
        embed_tokens: zml.nn.TokenEmbedding,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);
        var node = progress.start(phase.startMessage("embed_tokens"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "embed_tokens", io, from);

        const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

        return platform.compile(allocator, io, EmbedTokens{ .embed_tokens = embed_tokens }, .forward, .{tokens}, .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("llama", "embed_tokens"),
        });
    }

    fn compileLayer(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        llama_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        attention_parameters: attention.Parameters,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        const Layer = struct {
            layer: model.TransformerLayer,

            pub fn forward(
                self: @This(),
                hidden: zml.Tensor,
                token_index: zml.Tensor,
                kv_cache: model.KvCache,
                kv_cache_index: zml.Tensor,
                attention_metadata: attention.Metadata,
                attention_parameters_: attention.Parameters,
            ) struct { zml.Tensor, model.KvCache } {
                const new_hidden, const new_kv_cache, _ = self.layer.forward(
                    hidden,
                    token_index,
                    kv_cache,
                    kv_cache_index,
                    attention_metadata,
                    attention_parameters_,
                );

                return .{ new_hidden, new_kv_cache };
            }
        };

        progress.increaseEstimatedTotalItems(1);

        var node = progress.start(phase.startMessage("transformer layer"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "transformer layer", io, from);

        const hidden: zml.Tensor = .fromShape(zml.Shape.init(
            .{ .s = seqlen, .d = llama_model.config.hidden_size },
            llama_model.model.embed_tokens.weight.dtype(),
        ).withPartitioning(.{ .d = .replicated }));

        const kv_cache_index: zml.Tensor = .init(.{}, .u32);

        return platform.compile(
            allocator,
            io,
            Layer{ .layer = llama_model.model.layers[0] },
            .forward,
            .{
                hidden,
                parameters.token_index,
                parameters.kv_cache,
                kv_cache_index,
                parameters.attention_metadata,
                attention_parameters,
            },
            .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("llama", "layer"),
            },
        );
    }

    fn compileLmHead(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        llama_model: model.Model,
        parameters: CompilationOptions,
        seqlen: usize,
        phase: Phase,
        progress: *std.Progress.Node,
    ) !zml.Exe {
        progress.increaseEstimatedTotalItems(1);

        var node = progress.start(phase.startMessage("lm_head"), 1);
        defer node.end();

        const from: std.Io.Timestamp = .now(io, .awake);
        defer phase.logCompileDone(log, "lm_head", io, from);

        const hidden: zml.Tensor = .fromShape(zml.Shape.init(
            .{ .s = seqlen, .d = llama_model.config.hidden_size },
            llama_model.model.embed_tokens.weight.dtype(),
        ).withPartitioning(.{ .d = .replicated }));

        const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

        // PREFILL only needs the next token, i.e. the logits/argmax of the LAST real
        // prompt position (session reads tokens[num_tokens-1]); the other `seqlen-1`
        // positions are discarded. Running the lm_head over all of them is ~8% of
        // prefill at seqlen 16k (a [16384,128256] matmul+argmax). Apply it to that one
        // position instead — backend-agnostic, byte-identical greedy output. Decode is
        // already single-position, so it keeps the plain LmHead.forward.
        if (phase == .prefill) {
            const PrefillLmHead = struct {
                lm_head: model.LmHead,

                pub fn forward(
                    self: @This(),
                    hidden_: zml.Tensor,
                    tokens_: zml.Tensor,
                    rng: zml.Tensor.Rng,
                    last_index: zml.Tensor,
                ) struct { zml.Tensor, zml.Tensor.Rng } {
                    const toks = tokens_.withPartialTags(.{.s});
                    const last_hidden = hidden_.withPartialTags(.{ .s, .d })
                        .dynamicSlice(.{ .s = zml.Tensor.DynSlice{ .start = last_index, .len = 1 } });
                    const last_tok = toks.dynamicSlice(.{ .s = zml.Tensor.DynSlice{ .start = last_index, .len = 1 } });

                    const next_token, const new_rng = self.lm_head.forward(last_hidden, last_tok, rng);

                    // Write the single predicted token back at last_index so the output
                    // keeps the [seqlen] shape the runner/session expect.
                    const out = toks.dynamicUpdateSlice(.{ .s = last_index }, next_token).reuseBuffer(tokens_);
                    return .{ out, new_rng };
                }
            };

            const last_index: zml.Tensor = .init(.{}, .u32);
            return platform.compile(allocator, io, PrefillLmHead{ .lm_head = model.LmHead.init(llama_model) }, .forward, .{ hidden, tokens, parameters.rng, last_index }, .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName("llama", "lm_head"),
            });
        }

        return platform.compile(allocator, io, model.LmHead.init(llama_model), .forward, .{ hidden, tokens, parameters.rng }, .{
            .shardings = &parameters.shardings.all(),
            .program_name = phase.programName("llama", "lm_head"),
        });
    }
};
