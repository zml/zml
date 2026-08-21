const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");
const layer_ops = @import("layer.zig");
const runtime_weights = @import("runtime_weights.zig");

const log = std.log.scoped(.kimi_k3);

pub const Config = struct {
    model_type: []const u8,
    text_config: TextConfig,

    pub fn validate(self: Config) !void {
        if (!std.mem.eql(u8, self.model_type, "kimi_k3")) return error.InvalidKimiK3ModelType;
        const text = self.text_config;
        if (text.num_hidden_layers != 93 or text.hidden_size != 7168) return error.InvalidKimiK3Dimensions;
        if (text.linear_attn_config.kda_layers.len != 69 or text.linear_attn_config.full_attn_layers.len != 24) return error.InvalidKimiK3Schedule;
        if (text.first_k_dense_replace != 1 or text.num_experts != 896 or text.num_experts_per_token != 16) return error.InvalidKimiK3MoeContract;
        if (text.linear_attn_config.num_heads != 96 or text.linear_attn_config.head_dim != 128 or text.linear_attn_config.short_conv_kernel_size != 4) return error.InvalidKimiK3KdaContract;

        var seen: [93]bool = @splat(false);
        for (text.linear_attn_config.kda_layers) |one_based| {
            if (one_based < 1 or one_based > 93 or seen[@intCast(one_based - 1)]) return error.InvalidKimiK3Schedule;
            seen[@intCast(one_based - 1)] = true;
        }
        for (text.linear_attn_config.full_attn_layers) |one_based| {
            if (one_based < 1 or one_based > 93 or seen[@intCast(one_based - 1)]) return error.InvalidKimiK3Schedule;
            seen[@intCast(one_based - 1)] = true;
        }
        for (seen) |present| if (!present) return error.InvalidKimiK3Schedule;
    }
};

pub const TextConfig = struct {
    hidden_size: i64,
    intermediate_size: i64,
    max_position_embeddings: i64,
    num_hidden_layers: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rms_norm_eps: f32,
    first_k_dense_replace: i64,
    moe_layer_freq: i64,
    num_experts: i64,
    num_experts_per_token: i64,
    num_shared_experts: i64,
    moe_intermediate_size: i64,
    routed_expert_hidden_size: i64,
    kv_lora_rank: i64,
    q_lora_rank: i64,
    qk_nope_head_dim: i64,
    qk_rope_head_dim: i64,
    v_head_dim: i64,
    attn_res_block_size: i64,
    linear_attn_config: LinearAttentionConfig,

    pub fn isKdaLayer(self: TextConfig, zero_based_layer: usize) bool {
        const one_based: i64 = @intCast(zero_based_layer + 1);
        for (self.linear_attn_config.kda_layers) |layer| {
            if (layer == one_based) return true;
        }
        return false;
    }

    pub fn layerKind(self: TextConfig, zero_based_layer: usize) LayerKind {
        if (self.isKdaLayer(zero_based_layer)) {
            return if (zero_based_layer < self.first_k_dense_replace) .kda_dense else .kda_moe;
        }
        return .mla_moe;
    }
};

pub const LinearAttentionConfig = struct {
    kda_layers: []const i64,
    full_attn_layers: []const i64,
    gate_lower_bound: f32,
    head_dim: i64,
    num_heads: i64,
    short_conv_kernel_size: i64,
    use_full_rank_gate: bool,
};

pub const LayerKind = enum {
    kda_dense,
    kda_moe,
    mla_moe,
};

pub const LayerSelection = struct {
    first_layer: usize = 0,
    layer_limit: ?usize = null,

    pub fn end(self: LayerSelection, config: Config) !usize {
        const total: usize = @intCast(config.text_config.num_hidden_layers);
        if (self.first_layer > total) return error.InvalidLayerSelection;
        const count = self.layer_limit orelse total - self.first_layer;
        if (count > total - self.first_layer) return error.InvalidLayerSelection;
        return self.first_layer + count;
    }
};

pub const TensorRequirement = struct {
    name: []const u8,
    tensor: zml.Tensor,
};

pub const LayerWeights = struct {
    logical_index: usize,
    tensors: []TensorRequirement,
};

pub const TransformerLayer = union(LayerKind) {
    kda_dense: LayerWeights,
    kda_moe: LayerWeights,
    mla_moe: LayerWeights,

    pub fn kind(self: TransformerLayer) LayerKind {
        return std.meta.activeTag(self);
    }

    pub fn weights(self: *const TransformerLayer) *const LayerWeights {
        return switch (self.*) {
            inline else => |*value| value,
        };
    }
};

pub const KdaCacheLayout = struct {
    conv_state_shape: [3]i64 = .{ 1, 12288, 4 },
    recurrent_state_shape: [4]i64 = .{ 1, 96, 128, 128 },
    recurrent_dtype: zml.DataType = .f32,
};

pub const MlaCacheLayout = struct {
    latent_width: i64 = 512,
    extra_key_width: i64 = 64,
};

pub const LayerCache = union(enum) {
    kda: KdaCacheLayout,
    mla: MlaCacheLayout,
};

pub const ModelPlan = struct {
    layers: []LayerKind,

    pub fn init(allocator: std.mem.Allocator, config: Config, selection: LayerSelection) !ModelPlan {
        const last = try selection.end(config);
        const layers = try allocator.alloc(LayerKind, last - selection.first_layer);
        errdefer allocator.free(layers);
        for (layers, selection.first_layer..) |*kind, logical_index| {
            kind.* = config.text_config.layerKind(logical_index);
        }
        return .{ .layers = layers };
    }

    pub fn deinit(self: ModelPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

pub const CachePlan = struct {
    layers: []LayerCache,

    pub fn init(allocator: std.mem.Allocator, config: Config, selection: LayerSelection) !CachePlan {
        const last = try selection.end(config);
        const layers = try allocator.alloc(LayerCache, last - selection.first_layer);
        errdefer allocator.free(layers);
        for (layers, selection.first_layer..) |*cache, logical_index| {
            cache.* = if (config.text_config.isKdaLayer(logical_index)) .{ .kda = .{} } else .{ .mla = .{} };
        }
        return .{ .layers = layers };
    }

    pub fn deinit(self: CachePlan, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

/// Stable mapping from logical layers to packed per-family cache arrays.
/// Attention Residual workspace is deliberately absent: it is block-local
/// scratch state, not token-persistent generation state.
pub const CacheOrdinal = union(enum) {
    kda: usize,
    mla: usize,
};

pub const CacheMemory = struct {
    kda_bytes: u64,
    mla_bytes: u64,
    total_bytes: u64,
};

pub const PackedCachePlan = struct {
    layer_ordinals: []CacheOrdinal,
    kda_count: usize,
    mla_count: usize,
    attn_res_persisted: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: Config) !PackedCachePlan {
        const layer_count: usize = @intCast(config.text_config.num_hidden_layers);
        const ordinals = try allocator.alloc(CacheOrdinal, layer_count);
        errdefer allocator.free(ordinals);
        var kda_count: usize = 0;
        var mla_count: usize = 0;
        for (ordinals, 0..) |*cache_ordinal, logical_index| {
            if (config.text_config.isKdaLayer(logical_index)) {
                cache_ordinal.* = .{ .kda = kda_count };
                kda_count += 1;
            } else {
                cache_ordinal.* = .{ .mla = mla_count };
                mla_count += 1;
            }
        }
        return .{
            .layer_ordinals = ordinals,
            .kda_count = kda_count,
            .mla_count = mla_count,
        };
    }

    pub fn deinit(self: PackedCachePlan, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_ordinals);
    }

    pub fn ordinal(self: PackedCachePlan, logical_layer: usize) !CacheOrdinal {
        if (logical_layer >= self.layer_ordinals.len) return error.CacheLayerOutOfRange;
        return self.layer_ordinals[logical_layer];
    }

    /// Validate an append before any dynamic cache write. The checked add
    /// distinguishes integer overflow from a configured context overflow.
    pub fn validateAppend(position: u64, new_tokens: u64, max_tokens: u64) !u64 {
        const end = std.math.add(u64, position, new_tokens) catch return error.CachePositionOverflow;
        if (end > max_tokens) return error.CacheCapacityExceeded;
        return end;
    }

    /// Persistent cache bytes for BF16 convolution/MLA state and FP32 KDA
    /// recurrence. AttnRes contributes zero persistent bytes by contract.
    pub fn memoryBytes(self: PackedCachePlan, batch: u64, mla_tokens: u64) !CacheMemory {
        const conv_values_per_kda: u64 = 3 * 12288 * 4;
        const recurrent_values_per_kda: u64 = 96 * 128 * 128;
        const conv_bytes_per_kda = try std.math.mul(u64, conv_values_per_kda, 2);
        const recurrent_bytes_per_kda = try std.math.mul(u64, recurrent_values_per_kda, 4);
        const kda_bytes_per_layer = try std.math.add(u64, conv_bytes_per_kda, recurrent_bytes_per_kda);
        var kda_bytes = try std.math.mul(u64, kda_bytes_per_layer, @intCast(self.kda_count));
        kda_bytes = try std.math.mul(u64, kda_bytes, batch);

        const mla_bytes_per_layer_token: u64 = (512 + 64) * 2;
        var mla_bytes = try std.math.mul(u64, mla_bytes_per_layer_token, @intCast(self.mla_count));
        mla_bytes = try std.math.mul(u64, mla_bytes, batch);
        mla_bytes = try std.math.mul(u64, mla_bytes, mla_tokens);
        return .{
            .kda_bytes = kda_bytes,
            .mla_bytes = mla_bytes,
            .total_bytes = try std.math.add(u64, kda_bytes, mla_bytes),
        };
    }
};

pub const Model = struct {
    arena: std.heap.ArenaAllocator,
    layers: []TransformerLayer,
    config: Config,
    selection: LayerSelection,
    runtime_head: runtime_weights.HeadTensors,
    runtime_layer0: layer_ops.Layer0Weights,

    pub const GenOptions = struct {
        sampling_strategy: zml.nn.SamplingStrategy = .{},
        max_seq_len: i64,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        gen_options: GenOptions,
    ) !Model {
        return initSelected(allocator, store, config, gen_options, .{});
    }

    pub fn initSelected(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        gen_options: GenOptions,
        selection: LayerSelection,
    ) !Model {
        _ = gen_options;
        try config.validate();
        const last = try selection.end(config);
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();
        const layers = try arena_allocator.alloc(TransformerLayer, last - selection.first_layer);
        for (layers, selection.first_layer..) |*layer, logical_index| {
            layer.* = try initLayer(arena_allocator, store.root(), config, logical_index);
        }
        return .{
            .arena = arena,
            .layers = layers,
            .config = config,
            .selection = selection,
            .runtime_head = runtime_weights.HeadTensors.init(store.root()),
            .runtime_layer0 = layer_ops.Layer0Weights.init(store.root()),
        };
    }

    /// Build the complete operator/cache schedule without binding tensors for
    /// layers that are not present in the current minimum-weight checkpoint.
    /// This is a compile-readiness gate only; runtime construction continues
    /// to require every selected tensor through initSelected.
    pub fn initCompileOnly(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        selection: LayerSelection,
    ) !Model {
        try config.validate();
        const last = try selection.end(config);
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();
        const layers = try arena_allocator.alloc(TransformerLayer, last - selection.first_layer);
        const no_tensors = try arena_allocator.alloc(TensorRequirement, 0);
        for (layers, selection.first_layer..) |*planned, logical_index| {
            const weights: LayerWeights = .{ .logical_index = logical_index, .tensors = no_tensors };
            planned.* = switch (config.text_config.layerKind(logical_index)) {
                .kda_dense => .{ .kda_dense = weights },
                .kda_moe => .{ .kda_moe = weights },
                .mla_moe => .{ .mla_moe = weights },
            };
        }
        return .{
            .arena = arena,
            .layers = layers,
            .config = config,
            .selection = selection,
            .runtime_head = runtime_weights.HeadTensors.init(store.root()),
            .runtime_layer0 = layer_ops.Layer0Weights.init(store.root()),
        };
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn requestedTensorCount(self: Model) usize {
        var count: usize = 0;
        for (self.layers) |layer| count += layer.weights().tensors.len;
        return count;
    }

    pub fn isIgnoredTensorName(name: []const u8) bool {
        return std.mem.startsWith(u8, name, "vision_tower.") or
            std.mem.startsWith(u8, name, "mm_projector.");
    }
};

pub const Buffers = struct {
    head: runtime_weights.HeadWeights,
    layer0: zml.Bufferized(layer_ops.Layer0Weights),
    loader: runtime_weights.Loader,
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed.deinit();
        const options: Model.GenOptions = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed.value.text_config.max_position_embeddings,
        };
        const selection: LayerSelection = .{ .layer_limit = generation.kimi_k3_layer_limit };
        if (generation.kimi_k3_layer_limit) |limit| {
            if (limit == 0) return error.InvalidKimiK3LayerLimit;
            log.warn("Explicit partial Kimi K3 layer selection enabled: {}/93 layers", .{limit});
        }
        return .{
            .inner = try .initSelected(allocator, store, parsed.value, options, selection),
            .parsed_config = parsed,
        };
    }

    pub fn initCompileOnly(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        layer_limit: usize,
    ) !LoadedModel {
        const parsed = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed.deinit();
        const selection: LayerSelection = .{ .layer_limit = layer_limit };
        if (layer_limit == 0) return error.InvalidKimiK3LayerLimit;
        log.warn(
            "Compile-only Kimi K3 schedule enabled: {}/93 layers; nonresident tensors are not validated",
            .{layer_limit},
        );
        return .{
            .inner = try Model.initCompileOnly(
                allocator,
                store,
                parsed.value,
                selection,
            ),
            .parsed_config = parsed,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        if (platform.target != .cuda) return error.NvidiaCudaRequired;
        if (self.inner.selection.first_layer != 0 or self.inner.layers.len == 0) {
            return error.UnsupportedKimiK3RuntimeSelection;
        }
        var node = progress.start("Loading Kimi K3 resident weights...", 2);
        defer node.end();
        const loader: runtime_weights.Loader = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .store = store,
            .model_sharding = shardings.model,
            .expert_sharding = shardings.experts,
        };
        var head = try loader.loadHead();
        errdefer zml.Buffer.deinitAll(runtime_weights.HeadTensors, &head);
        node.completeOne();
        const layer0_buffers = try loader.loadLayer0();
        node.completeOne();
        return .{ .head = head, .layer0 = layer0_buffers, .loader = loader };
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        zml.Buffer.deinitAll(runtime_weights.HeadTensors, &buffers.head);
        zml.Buffer.deinitAll(layer_ops.Layer0Weights, &buffers.layer0);
        _ = allocator;
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = try inference.CompilationParameters.init(self.inner, seqlen, shardings);
        return inference.CompiledModel.init(allocator, io, platform, self, params, progress);
    }
};

const common_tensor_suffixes = [_][]const u8{
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
};

const kda_tensor_suffixes = [_][]const u8{
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.q_conv1d.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.o_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.b_proj.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
};

const mla_tensor_suffixes = [_][]const u8{
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
};

const dense_tensor_suffixes = [_][]const u8{
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
};

const moe_tensor_suffixes = [_][]const u8{
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
};

fn appendRequired(
    list: *std.ArrayList(TensorRequirement),
    allocator: std.mem.Allocator,
    store: zml.io.TensorStore.View,
    logical_index: usize,
    suffix: []const u8,
) !void {
    const name = try std.fmt.allocPrint(
        allocator,
        "language_model.model.layers.{d}.{s}",
        .{ logical_index, suffix },
    );
    const tensor = store.maybeCreateTensor(name, null, .replicated) orelse {
        log.err("Missing required Kimi K3 tensor: {s}", .{name});
        return error.MissingRequiredTensor;
    };
    try list.append(allocator, .{ .name = name, .tensor = tensor });
}

fn appendSuffixes(
    list: *std.ArrayList(TensorRequirement),
    allocator: std.mem.Allocator,
    store: zml.io.TensorStore.View,
    logical_index: usize,
    suffixes: []const []const u8,
) !void {
    for (suffixes) |suffix| try appendRequired(list, allocator, store, logical_index, suffix);
}

fn appendMoe(
    list: *std.ArrayList(TensorRequirement),
    allocator: std.mem.Allocator,
    store: zml.io.TensorStore.View,
    config: Config,
    logical_index: usize,
) !void {
    try appendSuffixes(list, allocator, store, logical_index, &moe_tensor_suffixes);
    const experts: usize = @intCast(config.text_config.num_experts);
    for (0..experts) |expert| {
        for ([_][]const u8{ "w1", "w2", "w3" }) |projection| {
            for ([_][]const u8{ "weight_packed", "weight_scale" }) |component| {
                const suffix = try std.fmt.allocPrint(
                    allocator,
                    "block_sparse_moe.experts.{d}.{s}.{s}",
                    .{ expert, projection, component },
                );
                try appendRequired(list, allocator, store, logical_index, suffix);
            }
        }
    }
}

fn initLayer(
    allocator: std.mem.Allocator,
    store: zml.io.TensorStore.View,
    config: Config,
    logical_index: usize,
) !TransformerLayer {
    const kind = config.text_config.layerKind(logical_index);
    const capacity: usize = switch (kind) {
        .kda_dense => 23,
        .kda_moe => 5404,
        .mla_moe => 5398,
    };
    var requirements = try std.ArrayList(TensorRequirement).initCapacity(allocator, capacity);
    try appendSuffixes(&requirements, allocator, store, logical_index, &common_tensor_suffixes);
    switch (kind) {
        .kda_dense => {
            try appendSuffixes(&requirements, allocator, store, logical_index, &kda_tensor_suffixes);
            try appendSuffixes(&requirements, allocator, store, logical_index, &dense_tensor_suffixes);
        },
        .kda_moe => {
            try appendSuffixes(&requirements, allocator, store, logical_index, &kda_tensor_suffixes);
            try appendMoe(&requirements, allocator, store, config, logical_index);
        },
        .mla_moe => {
            try appendSuffixes(&requirements, allocator, store, logical_index, &mla_tensor_suffixes);
            try appendMoe(&requirements, allocator, store, config, logical_index);
        },
    }
    std.debug.assert(requirements.items.len == capacity);
    const weights: LayerWeights = .{
        .logical_index = logical_index,
        .tensors = try requirements.toOwnedSlice(allocator),
    };
    return switch (kind) {
        .kda_dense => .{ .kda_dense = weights },
        .kda_moe => .{ .kda_moe = weights },
        .mla_moe => .{ .mla_moe = weights },
    };
}
