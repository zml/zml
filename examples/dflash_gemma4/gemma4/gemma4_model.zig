const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

pub const SamplingConfig = struct {
    temperature: f32 = 0.0,
};

pub const LayerType = enum {
    full_attention,
    sliding_attention,
};

pub const Config = struct {
    head_dim: u32 = 256,
    global_head_dim: ?u32 = null,
    hidden_size: u32 = 5376,
    num_hidden_layers: u32 = 60,
    num_attention_heads: u32 = 32,
    num_key_value_heads: u32 = 16,
    num_global_key_value_heads: ?u32 = null,
    rms_norm_eps: f32 = 1e-6,
    rope_parameters: ?RopeParameters = null,
    sliding_window: u32 = 1024,
    attention_k_eq_v: bool = true,
    final_logit_softcapping: ?f32 = null,
    num_kv_shared_layers: u32 = 0,
    hidden_size_per_layer_input: u32 = 0,
    vocab_size_per_layer_input: u32 = 0,
    use_double_wide_mlp: bool = false,
    bos_token_id: u32 = 2,
    eos_token_id: stdx.json.Union(union(enum) {
        int: u32,
        ints: []u32,
    }) = .{ .value = .{ .int = 1 } },
    layer_types: []LayerType = &.{},
    layer_types_allocated_by_fixup: bool = false,
    weight_prefix: []const u8 = "",

    const default_theta = .{
        .global = @as(f32, 1_000_000),
        .local = @as(f32, 10_000),
    };

    pub const RopeParameters = struct {
        full_attention: ?zml.nn.RopeOpts.Scaling = null,
        sliding_attention: ?zml.nn.RopeOpts.Scaling = null,
    };

    pub fn fixup(self: *Config, allocator: std.mem.Allocator) !void {
        if (self.num_global_key_value_heads == null) self.num_global_key_value_heads = self.num_key_value_heads;
        if (self.global_head_dim == null) self.global_head_dim = self.head_dim;

        const defaults: RopeParameters = .{
            .full_attention = .{ .proportional = .{
                .partial_rotary_factor = 0.25,
                .rope_theta = default_theta.global,
            } },
            .sliding_attention = .{ .default = .{ .rope_theta = default_theta.local } },
        };
        self.rope_parameters = if (self.rope_parameters) |rp| rp else defaults;
        if (self.rope_parameters.?.full_attention == null) self.rope_parameters.?.full_attention = defaults.full_attention;
        if (self.rope_parameters.?.sliding_attention == null) self.rope_parameters.?.sliding_attention = defaults.sliding_attention;

        if (self.layer_types.len == 0) {
            const layer_types = try allocator.alloc(LayerType, self.num_hidden_layers);
            for (layer_types, 0..) |*layer_type, i| {
                layer_type.* = if ((i + 1) % 6 == 0) .full_attention else .sliding_attention;
            }
            self.layer_types = layer_types;
            self.layer_types_allocated_by_fixup = true;
        }
    }
};

pub const ParsedConfig = struct {
    arena: std.heap.ArenaAllocator,
    value: Config,

    pub fn deinit(self: *ParsedConfig) void {
        self.arena.deinit();
    }
};

pub fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !ParsedConfig {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(arena_allocator, &file_reader.interface);
    defer reader.deinit();

    const parsed = try std.json.parseFromTokenSourceLeaky(std.json.Value, arena_allocator, &reader, .{ .allocate = .alloc_always });
    const has_text_config = parsed == .object and parsed.object.get("text_config") != null;
    const config_value = if (has_text_config) parsed.object.get("text_config").? else parsed;
    var config = try std.json.parseFromValueLeaky(Config, arena_allocator, config_value, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    config.weight_prefix = if (has_text_config) "model.language_model" else "";
    try config.fixup(arena_allocator);

    return .{ .arena = arena, .value = config };
}

pub const Gemma4CacheKind = enum {
    sliding,
    full,
};

pub const Gemma4CacheLayout = struct {
    layer_cache_kind: []Gemma4CacheKind,
    physical_cache_layer: []u32,
    updates_cache: []bool,
    num_sliding_cache_layers: u32,
    num_full_cache_layers: u32,

    pub fn init(allocator: std.mem.Allocator, layer_cache_kind_: []const Gemma4CacheKind, num_kv_shared_layers: u32) !Gemma4CacheLayout {
        if (num_kv_shared_layers > layer_cache_kind_.len) return error.InvalidGemma4CacheLayout;

        const layer_count = layer_cache_kind_.len;
        const layer_cache_kind = try allocator.alloc(Gemma4CacheKind, layer_count);
        errdefer allocator.free(layer_cache_kind);
        @memcpy(layer_cache_kind, layer_cache_kind_);

        const physical_cache_layer = try allocator.alloc(u32, layer_count);
        errdefer allocator.free(physical_cache_layer);
        const updates_cache = try allocator.alloc(bool, layer_count);
        errdefer allocator.free(updates_cache);

        const shared_start_layer = layer_count - num_kv_shared_layers;
        var num_sliding_cache_layers: u32 = 0;
        var num_full_cache_layers: u32 = 0;
        var sliding_anchor: ?u32 = null;
        var full_anchor: ?u32 = null;

        for (layer_cache_kind, 0..) |cache_kind, i| {
            const shared = i >= shared_start_layer;
            if (!shared) {
                updates_cache[i] = true;
                switch (cache_kind) {
                    .sliding => {
                        physical_cache_layer[i] = num_sliding_cache_layers;
                        sliding_anchor = num_sliding_cache_layers;
                        num_sliding_cache_layers += 1;
                    },
                    .full => {
                        physical_cache_layer[i] = num_full_cache_layers;
                        full_anchor = num_full_cache_layers;
                        num_full_cache_layers += 1;
                    },
                }
            } else {
                updates_cache[i] = false;
                physical_cache_layer[i] = switch (cache_kind) {
                    .sliding => sliding_anchor orelse return error.InvalidGemma4CacheLayout,
                    .full => full_anchor orelse return error.InvalidGemma4CacheLayout,
                };
            }
        }

        return .{
            .layer_cache_kind = layer_cache_kind,
            .physical_cache_layer = physical_cache_layer,
            .updates_cache = updates_cache,
            .num_sliding_cache_layers = num_sliding_cache_layers,
            .num_full_cache_layers = num_full_cache_layers,
        };
    }

    pub fn deinit(self: *Gemma4CacheLayout, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_cache_kind);
        allocator.free(self.physical_cache_layer);
        allocator.free(self.updates_cache);
    }
};

const CachePool = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub const Buffer = zml.Bufferized(CachePool);

    fn init(layer_count: u32, cache_seq_len: u32, num_kv_heads: u32, head_dim: u32, dtype: zml.DataType) CachePool {
        const shape = zml.Shape.init(.{
            .layer = layer_count,
            .k = cache_seq_len,
            .h = num_kv_heads,
            .hd = head_dim,
        }, dtype).withPartitioning(.{ .h = .model });
        return .{ .k = .fromShape(shape), .v = .fromShape(shape) };
    }

    fn deinitBuffer(buffer: *Buffer) void {
        buffer.k.deinit();
        buffer.v.deinit();
    }

    fn initZeroBuffer(pool: CachePool, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .k = try zeroBuffer(allocator, io, platform, pool.k.shape(), sharding),
            .v = try zeroBuffer(allocator, io, platform, pool.v.shape(), sharding),
        };
    }

    fn replaceBuffers(dst: *Buffer, src: *Buffer) void {
        replaceBuffer(&dst.k, &src.k);
        replaceBuffer(&dst.v, &src.v);
    }

    fn keysAt(pool: CachePool, physical_layer: u32) zml.Tensor {
        return pool.k.slice1d(.layer, .single(physical_layer));
    }

    fn valuesAt(pool: CachePool, physical_layer: u32) zml.Tensor {
        return pool.v.slice1d(.layer, .single(physical_layer));
    }

    fn updateAt(pool: CachePool, physical_layer: u32, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) CachePool {
        const update_shape = pool.k.shape().drop(.layer);
        const layer: zml.Tensor = .scalar(physical_layer, .u32);
        return .{
            .k = pool.k.scatterSlices(.{ .layer = layer, .k = token_index }, new_k.convert(pool.k.dtype()).transpose(update_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(pool.k),
            .v = pool.v.scatterSlices(.{ .layer = layer, .k = token_index }, new_v.convert(pool.v.dtype()).transpose(update_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(pool.v),
        };
    }

    fn reuseBuffer(pool: CachePool, other: CachePool) CachePool {
        return .{ .k = pool.k.reuseBuffer(other.k), .v = pool.v.reuseBuffer(other.v) };
    }
};

pub const KvCache = struct {
    sliding: CachePool,
    full: CachePool,
    layout: Gemma4CacheLayout,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(allocator: std.mem.Allocator, config: Config, cache_seq_len: u32, dtype: zml.DataType) !KvCache {
        const layer_kinds = try allocator.alloc(Gemma4CacheKind, config.num_hidden_layers);
        defer allocator.free(layer_kinds);
        for (config.layer_types, 0..) |layer_type, i| {
            layer_kinds[i] = layerTypeToCacheKind(layer_type);
        }
        const layout = try Gemma4CacheLayout.init(allocator, layer_kinds, config.num_kv_shared_layers);
        errdefer {
            var layout_ = layout;
            layout_.deinit(allocator);
        }

        return .{
            .sliding = .init(layout.num_sliding_cache_layers, cache_seq_len, config.num_key_value_heads, config.head_dim, dtype),
            .full = .init(layout.num_full_cache_layers, cache_seq_len, config.num_global_key_value_heads.?, config.global_head_dim.?, dtype),
            .layout = layout,
        };
    }

    pub fn deinit(self: KvCache, allocator: std.mem.Allocator) void {
        var layout = self.layout;
        layout.deinit(allocator);
    }

    pub fn initZeroBuffer(self: KvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .sliding = try self.sliding.initZeroBuffer(allocator, io, platform, sharding),
            .full = try self.full.initZeroBuffer(allocator, io, platform, sharding),
        };
    }

    pub fn deinitBuffer(buffer: *Buffer) void {
        CachePool.deinitBuffer(&buffer.sliding);
        CachePool.deinitBuffer(&buffer.full);
    }

    pub fn replaceBuffers(dst: *Buffer, src: *Buffer) void {
        CachePool.replaceBuffers(&dst.sliding, &src.sliding);
        CachePool.replaceBuffers(&dst.full, &src.full);
    }

    fn keysAt(self: KvCache, logical_layer: usize) zml.Tensor {
        const physical = self.layout.physical_cache_layer[logical_layer];
        return switch (self.layout.layer_cache_kind[logical_layer]) {
            .sliding => self.sliding.keysAt(physical),
            .full => self.full.keysAt(physical),
        };
    }

    fn valuesAt(self: KvCache, logical_layer: usize) zml.Tensor {
        const physical = self.layout.physical_cache_layer[logical_layer];
        return switch (self.layout.layer_cache_kind[logical_layer]) {
            .sliding => self.sliding.valuesAt(physical),
            .full => self.full.valuesAt(physical),
        };
    }

    fn updateAt(self: KvCache, logical_layer: usize, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        if (!self.layout.updates_cache[logical_layer]) return self;
        const physical = self.layout.physical_cache_layer[logical_layer];
        return switch (self.layout.layer_cache_kind[logical_layer]) {
            .sliding => .{
                .sliding = self.sliding.updateAt(physical, new_k, new_v, token_index),
                .full = self.full,
                .layout = self.layout,
            },
            .full => .{
                .sliding = self.sliding,
                .full = self.full.updateAt(physical, new_k, new_v, token_index),
                .layout = self.layout,
            },
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .sliding = self.sliding.reuseBuffer(other.sliding),
            .full = self.full.reuseBuffer(other.full),
            .layout = self.layout,
        };
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    model: Gemma4Text,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Model {
        const model_store = if (config.weight_prefix.len == 0) store else store.withPrefix(config.weight_prefix);
        return .{
            .model = try .init(allocator, model_store, config),
            .config = config,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn initKvCache(self: Model, allocator: std.mem.Allocator, config: Config, cache_seq_len: u32) !KvCache {
        return .init(allocator, config, cache_seq_len, self.model.embed_tokens.embed_tokens.weight.dtype());
    }

    pub fn loadBuffers(self: *const Model, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding) !Buffers {
        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .parallelism = 16,
            .shardings = shardings,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
        });
    }

    pub fn unloadBuffers(self: *Buffers, allocator: std.mem.Allocator) void {
        Gemma4Text.unloadBuffers(&self.model, allocator);
    }

    pub fn embedForward(self: Model, tokens: zml.Tensor) zml.Tensor {
        return self.model.embed_tokens.forward(tokens.withPartialTags(.{.s})).withPartialTags(.{.d});
    }

    pub fn logitsForward(self: Model, hidden_: zml.Tensor) zml.Tensor {
        const hidden = self.model.norm.forward(hidden_.withPartialTags(.{ .s, .d }));
        const lm_head: zml.nn.Linear = .init(
            self.model.embed_tokens.embed_tokens.weight.withTags(.{ .voc, .d }).withPartitioning(.{ .voc = .replicated, .d = .model }),
            null,
            .d,
        );
        var logits = lm_head.forward(hidden).withPartialTags(.{.voc});
        if (self.config.final_logit_softcapping) |cap| {
            logits = logits.divByConst(cap).tanh().mul(zml.Tensor.scalar(cap, logits.dtype()).broad(logits.shape()));
        }
        return logits;
    }

    fn sampleTargetTokens(self: Model, hidden: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        return sampleLogits(self.logitsForward(hidden), sampling, rng);
    }

    fn sampleLogits(logits_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const logits = logits_.withPartialTags(.{.voc});
        const topk: u32 = if (sampling.temperature < 0.00001) 1 else @intCast(logits.dim(.voc));
        const tokens, const updated_rng = zml.nn.sampleTokens(logits, .{ .topk = topk, .temperature = sampling.temperature }, rng);
        return .{ tokens.convert(.u32), updated_rng };
    }

    fn sampleLastTargetToken(self: Model, hidden_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const hidden = hidden_.withPartialTags(.{ .s, .d });
        const last = hidden.slice1d(.s, .single(hidden.dim(.s) - 1)).reshape(.{ .s = 1, .d = hidden.dim(.d) });
        return self.sampleTargetTokens(last, sampling, rng);
    }

    pub fn prefillForward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        target_layer_ids: []const u32,
        sampling: SamplingConfig,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor, KvCache, zml.Tensor.Rng } {
        const target_hidden, const out, const updated_kv_cache = self.model.forward(tokens_.withPartialTags(.{.s}), token_index, kv_cache, target_layer_ids);
        const sampled_last, const updated_rng = self.sampleLastTargetToken(out, sampling, rng);
        return .{ target_hidden, sampled_last, updated_kv_cache, updated_rng };
    }

    pub fn verifyForward(
        self: Model,
        tokens_: zml.Tensor,
        draft_logits_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        target_layer_ids: []const u32,
        sampling: SamplingConfig,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor, zml.Tensor, KvCache, zml.Tensor.Rng } {
        const target_hidden, const out, const updated_kv_cache = self.model.forward(tokens_.withPartialTags(.{.s}), token_index, kv_cache, target_layer_ids);
        const target_logits = self.logitsForward(out);
        const valid_draft_tokens, const correction_token, const updated_rng = verifyDraftTokens(tokens_, draft_logits_, target_logits, sampling, rng);
        return .{ target_hidden, valid_draft_tokens, correction_token, updated_kv_cache, updated_rng };
    }
};

pub fn isEosToken(config: *const Config, token_id: u32) bool {
    if (token_id == 50 or token_id == 106) return true;
    return switch (config.eos_token_id.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}

fn verifyDraftTokens(draft_tokens_: zml.Tensor, draft_logits_: zml.Tensor, target_logits_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor, zml.Tensor.Rng } {
    const draft_tokens = draft_tokens_.withPartialTags(.{.s});
    const draft_logits = draft_logits_.withPartialTags(.{ .s, .voc });
    const target_logits = target_logits_.withPartialTags(.{ .s, .voc });
    const draft_len = draft_tokens.dim(.s);
    const verify_len = draft_len - 1;
    stdx.debug.assert(draft_len >= 2, "DFlash verify requires anchor + at least one proposal", .{});
    stdx.debug.assert(draft_logits.dim(.s) == draft_len, "draft logits/token length mismatch", .{});
    stdx.debug.assert(target_logits.dim(.s) == draft_len, "target logits/token length mismatch", .{});
    stdx.debug.assert(draft_logits.dim(.voc) == target_logits.dim(.voc), "draft/target vocab mismatch", .{});

    if (sampling.temperature < 0.00001) {
        const sampled_tokens = target_logits.argMax(.voc).indices.squeeze(.voc).convert(.u32);
        const proposals = draft_tokens.slice1d(.s, .{ .start = 1, .end = draft_len });
        const posterior = sampled_tokens.slice1d(.s, .{ .start = 0, .end = verify_len });
        const matches = proposals.cmp(.EQ, posterior);
        const candidate_idx = zml.Tensor.iota(.init(.{ .s = verify_len }, .u32), .s).convert(.u32);
        const sentinel = zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32).broad(candidate_idx.shape());
        const valid_draft_tokens = matches.select(sentinel, candidate_idx).min(.s).squeeze(.s);
        const correction_token = sampled_tokens.gather(.{ .s = valid_draft_tokens }, .{}).convert(.u32);
        return .{ valid_draft_tokens, correction_token, rng };
    }

    const proposals = draft_tokens.slice1d(.s, .{ .start = 1, .end = draft_len });
    const draft_proposal_logits = draft_logits.slice1d(.s, .{ .start = 1, .end = draft_len });
    const target_proposal_logits = target_logits.slice1d(.s, .{ .start = 0, .end = verify_len });
    const draft_probs = tokenProbs(draft_proposal_logits, sampling);
    const target_probs = tokenProbs(target_proposal_logits, sampling);

    const p = target_probs.gather(.{ .voc = proposals }, .{});
    const q = draft_probs.gather(.{ .voc = proposals }, .{});
    const epsilon = zml.Tensor.scalar(std.math.floatEps(f32), .f32).broad(q.shape());
    const acceptance = p.div(q.maximum(epsilon));

    const accept_rng, const u = rng.uniform(acceptance.shape().withDtype(.f32), .{ .min = std.math.floatEps(f32), .max = 1 });
    const accepted = acceptance.cmp(.GT, u);
    const candidate_idx = zml.Tensor.iota(.init(.{ .s = verify_len }, .u32), .s).convert(.u32);
    const sentinel = zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32).broad(candidate_idx.shape());
    const valid_draft_tokens = accepted.select(sentinel, candidate_idx).min(.s).squeeze(.s);

    const residual_probs = target_probs.sub(draft_probs).relu();
    const residual_idx = valid_draft_tokens.minimum(zml.Tensor.scalar(@as(u32, @intCast(verify_len - 1)), .u32));
    const residual = residual_probs.gather(.{ .s = residual_idx }, .{});
    const residual_logits = residual.log();

    const no_rejection = valid_draft_tokens.cmp(.EQ, zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32));
    const target_correction_logits = target_logits.gather(.{ .s = valid_draft_tokens }, .{}).convert(.f32).scale(1 / sampling.temperature);
    const correction_logits = no_rejection.broad(target_correction_logits.shape()).select(target_correction_logits, residual_logits);
    const correction_token, const updated_rng = Model.sampleLogits(correction_logits, .{ .temperature = 1.0 }, accept_rng);
    return .{ valid_draft_tokens, correction_token.convert(.u32), updated_rng };
}

fn tokenProbs(logits_: zml.Tensor, sampling: SamplingConfig) zml.Tensor {
    const logits = logits_.withPartialTags(.{.voc}).convert(.f32);
    return logits.scale(1 / sampling.temperature).softmax(.voc);
}

pub const Gemma4Text = struct {
    embed_tokens: Gemma4TextScaledWordEmbedding,
    norm: Gemma4RmsNorm,
    layers: []Gemma4DecoderLayer,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Gemma4Text {
        const layers = try allocator.alloc(Gemma4DecoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("layers").withLayer(i), config, config.layer_types[i], i);
        }
        return .{
            .embed_tokens = .init(store.withPrefix("embed_tokens"), config),
            .norm = .init(store.withPrefix("norm"), config.rms_norm_eps, true),
            .layers = layers,
        };
    }

    pub fn deinit(self: Gemma4Text, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4Text), allocator: std.mem.Allocator) void {
        Gemma4TextScaledWordEmbedding.unloadBuffers(&self.embed_tokens);
        Gemma4RmsNorm.unloadBuffers(&self.norm);
        for (self.layers) |*layer| Gemma4DecoderLayer.unloadBuffers(layer);
        allocator.free(self.layers);
    }

    pub fn forward(self: Gemma4Text, tokens: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache, target_layer_ids: []const u32) struct { zml.Tensor, zml.Tensor, KvCache } {
        var hidden = self.embed_tokens.forward(tokens);
        var updated_kv_cache = kv_cache;
        var selected: [32]zml.Tensor = undefined;
        var selected_len: usize = 0;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(hidden, token_index, updated_kv_cache);
            for (target_layer_ids) |target_layer_id| {
                if (target_layer_id == i) {
                    selected[selected_len] = hidden;
                    selected_len += 1;
                    break;
                }
            }
        }

        const target_hidden = zml.Tensor.concatenate(selected[0..selected_len], .d);
        return .{ target_hidden, hidden, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const Gemma4TextScaledWordEmbedding = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    scale: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Gemma4TextScaledWordEmbedding {
        return .{
            .embed_tokens = .{ .weight = store.createTensor("weight", .{ .voc, .d }, .{ .voc = .model, .d = .replicated }) },
            .scale = std.math.sqrt(@as(f32, @floatFromInt(config.hidden_size))),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4TextScaledWordEmbedding)) void {
        self.embed_tokens.weight.deinit();
    }

    pub fn forward(self: Gemma4TextScaledWordEmbedding, tokens: zml.Tensor) zml.Tensor {
        const embeds = self.embed_tokens.forward(tokens).withPartialTags(.{.d});
        return embeds.mul(zml.Tensor.scalar(self.scale, embeds.dtype()).broad(embeds.shape()));
    }
};

pub const Gemma4DecoderLayer = struct {
    input_layernorm: Gemma4RmsNorm,
    self_attn: Gemma4Attention,
    post_attention_layernorm: Gemma4RmsNorm,
    pre_feedforward_layernorm: Gemma4RmsNorm,
    post_feedforward_layernorm: Gemma4RmsNorm,
    mlp: Gemma4Mlp,
    layer_scalar: Gemma4LayerScalar,
    layer_index: usize,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType, layer_index: usize) Gemma4DecoderLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps, true),
            .self_attn = .init(store.withPrefix("self_attn"), config, layer_type, layer_index),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps, true),
            .pre_feedforward_layernorm = .init(store.withPrefix("pre_feedforward_layernorm"), config.rms_norm_eps, true),
            .post_feedforward_layernorm = .init(store.withPrefix("post_feedforward_layernorm"), config.rms_norm_eps, true),
            .mlp = .init(store.withPrefix("mlp")),
            .layer_scalar = .init(store),
            .layer_index = layer_index,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4DecoderLayer)) void {
        Gemma4RmsNorm.unloadBuffers(&self.input_layernorm);
        Gemma4Attention.unloadBuffers(&self.self_attn);
        Gemma4RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Gemma4RmsNorm.unloadBuffers(&self.pre_feedforward_layernorm);
        Gemma4Mlp.unloadBuffers(&self.mlp);
        Gemma4RmsNorm.unloadBuffers(&self.post_feedforward_layernorm);
        Gemma4LayerScalar.unloadBuffers(&self.layer_scalar);
    }

    pub fn forward(self: Gemma4DecoderLayer, hidden_states_: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        stdx.debug.assert(hidden_states_.rank() >= 2 and hidden_states_.shape().hasTags(.{ .s, .d }), "Gemma4DecoderLayer expected {{.s, .d}}, received: {f}", .{hidden_states_});
        var hidden_states = hidden_states_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        var residual = hidden_states;

        hidden_states = self.input_layernorm.forward(hidden_states);
        hidden_states, const updated_kv_cache = self.self_attn.forward(hidden_states, token_index, kv_cache);
        hidden_states = self.post_attention_layernorm.forward(hidden_states.rename(.{ .dout = .d }));
        hidden_states = hidden_states.add(residual).withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });

        residual = hidden_states;
        hidden_states = self.pre_feedforward_layernorm.forward(hidden_states);
        hidden_states = self.mlp.forward(hidden_states);
        hidden_states = self.post_feedforward_layernorm.forward(hidden_states.rename(.{ .dout = .d }));
        hidden_states = hidden_states.add(residual).withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });

        hidden_states = self.layer_scalar.forward(hidden_states);
        return .{ hidden_states.reuseBuffer(hidden_states_), updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const Gemma4Attention = struct {
    q_norm: Gemma4RmsNorm,
    q_proj: zml.nn.Linear,
    k_norm: Gemma4RmsNorm,
    k_proj: zml.nn.Linear,
    v_norm: Gemma4RmsNorm,
    v_proj: ?zml.nn.Linear,
    o_proj: zml.nn.Linear,
    config: Config,
    layer_type: LayerType,
    layer_index: usize,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType, layer_index: usize) Gemma4Attention {
        const v_proj_weight = store.withPrefix("v_proj").maybeCreateTensor("weight", .{ .dout, .d }, .{ .dout = .model });
        const v_proj_bias = if (v_proj_weight != null) store.withPrefix("v_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }) else null;
        return .{
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps, true),
            .q_proj = .init(store.withPrefix("q_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model }), store.withPrefix("q_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps, true),
            .k_proj = .init(store.withPrefix("k_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model }), store.withPrefix("k_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .v_norm = .init(store.withPrefix("v_norm"), config.rms_norm_eps, false),
            .v_proj = if (v_proj_weight) |weight| .init(weight, v_proj_bias, .d) else null,
            .o_proj = .init(store.withPrefix("o_proj").createTensor("weight", .{ .dout, .d }, .{ .d = .model }), store.withPrefix("o_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .replicated }), .d),
            .config = config,
            .layer_type = layer_type,
            .layer_index = layer_index,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4Attention)) void {
        Gemma4RmsNorm.unloadBuffers(&self.q_norm);
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*bias| bias.deinit();
        Gemma4RmsNorm.unloadBuffers(&self.k_norm);
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        Gemma4RmsNorm.unloadBuffers(&self.v_norm);
        if (self.v_proj) |*v_proj| {
            v_proj.weight.deinit();
            if (v_proj.bias) |*bias| bias.deinit();
        }
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();
    }

    fn ropeScaling(self: Gemma4Attention) zml.nn.RopeOpts.Scaling {
        return switch (self.layer_type) {
            .full_attention => self.config.rope_parameters.?.full_attention.?,
            .sliding_attention => self.config.rope_parameters.?.sliding_attention.?,
        };
    }

    fn numKvHeads(self: Gemma4Attention) u32 {
        return switch (self.layer_type) {
            .full_attention => self.config.num_global_key_value_heads.?,
            .sliding_attention => self.config.num_key_value_heads,
        };
    }

    fn headDim(self: Gemma4Attention) u32 {
        return switch (self.layer_type) {
            .full_attention => self.config.global_head_dim.?,
            .sliding_attention => self.config.head_dim,
        };
    }

    pub fn forward(self: Gemma4Attention, x_: zml.Tensor, token_index: zml.Tensor, kv_cache_: KvCache) struct { zml.Tensor, KvCache } {
        const x = x_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const num_heads = self.config.num_attention_heads;
        const num_kv_heads = self.numKvHeads();
        const head_dim = self.headDim();

        stdx.debug.assert(num_heads % num_kv_heads == 0, "Gemma4Attention expected query heads to be divisible by kv heads, got {} query heads and {} kv heads", .{ num_heads, num_kv_heads });

        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = num_heads, .hd = head_dim });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = head_dim });
        var v = if (self.v_proj) |v_proj| b: {
            break :b v_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = head_dim });
        } else b: {
            stdx.debug.assert(self.config.attention_k_eq_v and self.layer_type == .full_attention, "missing v_proj only supported for full attention with attention_k_eq_v enabled", .{});
            break :b k;
        };

        q = q.withPartitioning(.{ .h = .model });
        k = k.withPartitioning(.{ .h = .model });
        v = v.withPartitioning(.{ .h = .model });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        v = self.v_norm.forward(v.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const pos_index = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).add(token_index.broad(.init(.{ .s = x.dim(.s) }, token_index.dtype())));
        const rope_opts: zml.nn.RopeOpts = .{ .layout = .sequential, .scaling = self.ropeScaling() };
        q = zml.nn.rope(q, pos_index, rope_opts).rename(.{ .s = .q });
        k = zml.nn.rope(k, pos_index, rope_opts).rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const updated_kv_cache = kv_cache_.updateAt(self.layer_index, k, v, token_index);
        var cached_k = updated_kv_cache.keysAt(self.layer_index).convert(q.dtype()).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        var cached_v = updated_kv_cache.valuesAt(self.layer_index).convert(q.dtype()).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const mask = self.attentionMask(q.dim(.q), cached_k.dim(.k), token_index, q.dtype());
        const attn = zml.nn.sdpa(q.convert(.f32), cached_k.convert(.f32), cached_v.convert(.f32), .{
            .attn_mask = mask.convert(.f32),
            .scale = zml.Tensor.scalar(@as(f32, 1.0), .f32),
        })
            .withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated })
            .merge(.{ .d = .{ .h, .hd } })
            .rename(.{ .q = .s })
            .convert(self.o_proj.weight.dtype());
        return .{ self.o_proj.forward(attn), updated_kv_cache };
    }

    fn attentionMask(self: Gemma4Attention, q_len: i64, k_len: i64, token_index: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        const shape = zml.Shape.init(.{ .q = q_len, .k = k_len }, .bool);
        const q_abs = zml.Tensor.iota(shape, .q).convert(token_index.dtype()).add(token_index.broad(shape.withDtype(token_index.dtype())));
        const k_idx = zml.Tensor.iota(shape, .k).convert(token_index.dtype());
        var valid = k_idx.cmp(.LE, q_abs);
        if (self.layer_type == .sliding_attention) {
            valid = valid.logical(.AND, q_abs.cmp(.LT, k_idx.addConstant(self.config.sliding_window)));
        }
        const zeros = zml.Tensor.constant(dtype.zero()).broad(shape.withDtype(dtype));
        const minus_inf = zml.Tensor.constant(dtype.minValue()).broad(shape.withDtype(dtype));
        return valid.select(zeros, minus_inf);
    }
};

pub const Gemma4RmsNorm = struct {
    weight: ?zml.Tensor,
    eps: f32,
    with_scale: bool,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, with_scale: bool) Gemma4RmsNorm {
        return .{ .weight = if (with_scale) store.createTensor("weight", .{.d}, .{ .d = .replicated }) else null, .eps = eps, .with_scale = with_scale };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4RmsNorm)) void {
        if (self.weight) |*weight| weight.deinit();
    }

    pub fn forward(self: Gemma4RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        var output = x.convert(.f32).mul(zml.Tensor.rsqrt(x.convert(.f32).powByConst(2).mean(.d).addConstant(self.eps)));
        if (self.with_scale) {
            const weight = self.weight orelse std.debug.panic("expected RMSNorm scale", .{});
            output = output.mul(weight.convert(.f32).withTags(.{.d}).broad(output.shape()));
        }
        return output.convert(input.dtype());
    }
};

pub const Gemma4LayerScalar = struct {
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Gemma4LayerScalar {
        return .{ .weight = store.createTensor("layer_scalar", .{.d}, .{ .d = .replicated }) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4LayerScalar)) void {
        self.weight.deinit();
    }

    pub fn forward(self: Gemma4LayerScalar, hidden: zml.Tensor) zml.Tensor {
        return hidden.mul(self.weight.convert(hidden.dtype()).withTags(.{.d}).broad(hidden.shape()));
    }
};

pub const Gemma4Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Gemma4Mlp {
        return .{
            .up_proj = .init(store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model }), store.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .gate_proj = .init(store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model }), store.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
            .down_proj = .init(store.withPrefix("down_proj").createTensor("weight", .{ .dout, .d }, .{ .d = .model }), store.withPrefix("down_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .replicated }), .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gemma4Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Gemma4Mlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.gelu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output);
    }
};

fn layerTypeToCacheKind(layer_type: LayerType) Gemma4CacheKind {
    return switch (layer_type) {
        .full_attention => .full,
        .sliding_attention => .sliding,
    };
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, sharding: zml.Sharding) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromSlice(io, platform, zml.Slice.init(shape, bytes), sharding);
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) dst.deinit();
    dst.* = src.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
