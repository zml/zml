const std = @import("std");

const zml = @import("zml");

const kda = @import("kda.zig");
const layer = @import("layer.zig");
const mla = @import("mla.zig");

const log = std.log.scoped(.kimi_k3_weights);

pub const expert_count: usize = 896;

/// Mutable instrumentation shared by copied Loader values. Payload counters
/// cover explicit runtime layer reads; head/layer-0 bulk loading is separate.
pub const LoadStats = struct {
    layer_loads: usize = 0,
    payload_reads: u64 = 0,
    payload_bytes: u64 = 0,

    fn recordPayload(self: *LoadStats, bytes: usize) void {
        self.payload_reads += 1;
        self.payload_bytes += bytes;
    }
};

pub const expert_component_bytes = struct {
    pub const w1_values: u64 = expert_count * 3072 * 1792;
    pub const w1_scale: u64 = expert_count * 3072 * 112;
    pub const w2_values: u64 = expert_count * 3584 * 1536;
    pub const w2_scale: u64 = expert_count * 3584 * 96;
    pub const w3_values: u64 = expert_count * 3072 * 1792;
    pub const w3_scale: u64 = expert_count * 3072 * 112;
    pub const peak_host_staging: u64 = @max(w1_values, w2_values, w3_values);
    pub const device_bank: u64 = w1_values + w1_scale + w2_values + w2_scale + w3_values + w3_scale;
};

/// Contiguous global expert ownership for one expert-parallel rank. Uneven
/// partitions are intentional so any device count can be preflighted without
/// inventing or dropping an expert.
pub const ExpertPartition = struct {
    rank: usize,
    ranks: usize,
    first: usize,
    end: usize,

    pub fn init(rank: usize, ranks: usize) !ExpertPartition {
        if (ranks == 0 or rank >= ranks) return error.InvalidExpertParallelRank;
        const base = expert_count / ranks;
        const remainder = expert_count % ranks;
        const first = rank * base + @min(rank, remainder);
        const partition_count = base + @intFromBool(rank < remainder);
        return .{ .rank = rank, .ranks = ranks, .first = first, .end = first + partition_count };
    }

    pub fn count(self: ExpertPartition) usize {
        return self.end - self.first;
    }

    pub fn contains(self: ExpertPartition, global_expert: usize) bool {
        return global_expert >= self.first and global_expert < self.end;
    }
};

/// Logical tensor/expert-parallel ownership. ZML model and expert shardings
/// carry these independent channels to compiled executables and buffer
/// transfers; this plan validates the degrees before any checkpoint is opened.
pub const DistributedPlan = struct {
    device_count: usize,
    tensor_parallel: usize,
    expert_parallel: usize,

    pub fn init(device_count: usize, tensor_parallel: usize) !DistributedPlan {
        if (device_count == 0 or tensor_parallel == 0 or device_count % tensor_parallel != 0) {
            return error.InvalidKimiK3ParallelPlan;
        }
        const tensor_parallel_dims = [_]usize{ 7168, 12288, 1536, 512, 128, 96 };
        for (tensor_parallel_dims) |dim| {
            if (dim % tensor_parallel != 0) return error.InvalidKimiK3TensorParallelDegree;
        }
        return .{
            .device_count = device_count,
            .tensor_parallel = tensor_parallel,
            .expert_parallel = device_count / tensor_parallel,
        };
    }

    pub fn expertPartition(self: DistributedPlan, expert_rank: usize) !ExpertPartition {
        return ExpertPartition.init(expert_rank, self.expert_parallel);
    }
};

/// Production loading keeps the head and dense layer 0 resident, then stages
/// exactly one MoE layer at a time. Expert components are read expert by
/// expert into one bounded host buffer before transfer through the expert
/// sharding; no dequantized full-model copy is materialized on the host.
pub const StreamingLoadPlan = struct {
    resident_layers: usize = 1,
    staged_layers: usize = 92,
    peak_host_staging_bytes: u64 = expert_component_bytes.peak_host_staging,
    expert_bank_device_bytes: u64 = expert_component_bytes.device_bank,

    pub fn init(layer_count: usize) !StreamingLoadPlan {
        if (layer_count != 93) return error.InvalidKimiK3StreamingLayerCount;
        return .{};
    }
};

pub const HeadTensors = struct {
    embedding: zml.Tensor,
    output_res_norm: zml.Tensor,
    output_res_projection: zml.Tensor,
    final_norm: zml.Tensor,
    lm_head: zml.Tensor,

    pub fn init(root: zml.io.TensorStore.View) HeadTensors {
        return .{
            .embedding = root.createTensor("language_model.model.embed_tokens.weight", .{ .voc, .d }, .replicated),
            .output_res_norm = root.createTensor("language_model.model.output_attn_res_norm.weight", .{.d}, .replicated),
            .output_res_projection = root.createTensor("language_model.model.output_attn_res_proj.weight", .{ .one, .d }, .replicated),
            .final_norm = root.createTensor("language_model.model.norm.weight", .{.d}, .replicated),
            .lm_head = root.createTensor("language_model.lm_head.weight", .{ .voc, .d }, .replicated),
        };
    }

    pub fn initSharded(root: zml.io.TensorStore.View) HeadTensors {
        return .{
            .embedding = root.createTensor("language_model.model.embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }),
            .output_res_norm = root.createTensor("language_model.model.output_attn_res_norm.weight", .{.d}, .replicated),
            .output_res_projection = root.createTensor("language_model.model.output_attn_res_proj.weight", .{ .one, .d }, .replicated),
            .final_norm = root.createTensor("language_model.model.norm.weight", .{.d}, .replicated),
            .lm_head = root.createTensor("language_model.lm_head.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }),
        };
    }
};

pub const HeadWeights = zml.Bufferized(HeadTensors);

pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    model_sharding: zml.Sharding,
    expert_sharding: zml.Sharding,
    stats: ?*LoadStats = null,

    fn rootView(self: Loader) zml.io.TensorStore.View {
        return self.store.view();
    }

    fn layerKey(self: Loader, layer_index: usize, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(
            self.allocator,
            "language_model.model.layers.{d}.{s}",
            .{ layer_index, suffix },
        );
    }

    fn loadRoot(self: Loader, key: []const u8, tags: anytype) !zml.Buffer {
        const shape = self.rootView().getShape(key) orelse {
            log.err("Missing required runtime tensor: {s}", .{key});
            return error.MissingKimiK3RuntimeWeight;
        };
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.rootView().getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        if (self.stats) |stats| stats.recordPayload(bytes.len);
        return zml.Buffer.fromBytes(
            self.io,
            self.platform,
            shape.withTags(tags),
            self.model_sharding,
            bytes,
        );
    }

    fn loadLayer(self: Loader, layer_index: usize, suffix: []const u8, tags: anytype) !zml.Buffer {
        const key = try self.layerKey(layer_index, suffix);
        defer self.allocator.free(key);
        return self.loadRoot(key, tags);
    }

    fn loadLayerAs(self: Loader, layer_index: usize, suffix: []const u8, target: zml.Shape) !zml.Buffer {
        const key = try self.layerKey(layer_index, suffix);
        defer self.allocator.free(key);
        const source = self.rootView().getShape(key) orelse return error.MissingKimiK3RuntimeWeight;
        if (source.byteSize() != target.byteSize()) return error.KimiK3RuntimeWeightReshapeMismatch;
        const bytes = try self.allocator.alloc(u8, source.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.rootView().getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        if (self.stats) |stats| stats.recordPayload(bytes.len);
        return zml.Buffer.fromBytes(self.io, self.platform, target, self.model_sharding, bytes);
    }

    fn loadExpertComponent(
        self: Loader,
        layer_index: usize,
        projection: []const u8,
        component: []const u8,
        target: zml.Shape,
    ) !zml.Buffer {
        const per_expert = @divExact(target.byteSize(), expert_count);
        const bytes = try self.allocator.alloc(u8, target.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        for (0..expert_count) |expert| {
            const suffix = try std.fmt.allocPrint(
                self.allocator,
                "block_sparse_moe.experts.{d}.{s}.{s}",
                .{ expert, projection, component },
            );
            defer self.allocator.free(suffix);
            const key = try self.layerKey(layer_index, suffix);
            defer self.allocator.free(key);
            const source = self.rootView().getShape(key) orelse {
                log.err("Missing expert tensor: {s}", .{key});
                return error.MissingKimiK3ExpertWeight;
            };
            if (source.byteSize() != per_expert) return error.KimiK3ExpertShapeMismatch;
            var reader = try self.rootView().getReader(key, self.io, &io_buffer);
            defer reader.deinit();
            _ = try reader.interface.readSliceAll(bytes[expert * per_expert ..][0..per_expert]);
            if (self.stats) |stats| stats.recordPayload(per_expert);
        }
        return zml.Buffer.fromBytes(self.io, self.platform, target, self.expert_sharding, bytes);
    }

    pub fn loadHead(self: Loader) !HeadWeights {
        const symbolic = HeadTensors.initSharded(self.rootView());
        var buffers = try zml.mem.bufferize(self.allocator, HeadTensors, &symbolic);
        errdefer zml.Buffer.deinitAll(HeadTensors, &buffers);
        var tensor_loader: zml.io.Loader = try .init(self.allocator, self.platform, .{
            .parallelism = 1,
            .dma_chunks = 2,
            .dma_chunk_size = 256 * zml.MiB,
        });
        defer tensor_loader.deinit();
        tensor_loader.load(
            self.io,
            HeadTensors,
            &symbolic,
            &buffers,
            self.store,
            &.{self.model_sharding},
            .{},
        );
        try tensor_loader.await(self.io);
        return buffers;
    }

    pub fn loadLayer0(self: Loader) !zml.Bufferized(layer.Layer0Weights) {
        const symbolic = layer.Layer0Weights.initSharded(self.rootView());
        var buffers = try zml.mem.bufferize(self.allocator, layer.Layer0Weights, &symbolic);
        errdefer zml.Buffer.deinitAll(layer.Layer0Weights, &buffers);
        var tensor_loader: zml.io.Loader = try .init(self.allocator, self.platform, .{
            .parallelism = 1,
            .dma_chunks = 2,
            .dma_chunk_size = 256 * zml.MiB,
        });
        defer tensor_loader.deinit();
        tensor_loader.load(
            self.io,
            layer.Layer0Weights,
            &symbolic,
            &buffers,
            self.store,
            &.{self.model_sharding},
            .{},
        );
        try tensor_loader.await(self.io);
        return buffers;
    }

    fn loadCommon(self: Loader, layer_index: usize) !zml.Bufferized(layer.MoeLayerWeights) {
        var result: zml.Bufferized(layer.MoeLayerWeights) = undefined;
        result.attention_res_norm = try self.loadLayer(layer_index, "self_attention_res_norm.weight", .{.d});
        errdefer result.attention_res_norm.deinit();
        result.attention_res_projection = try self.loadLayer(layer_index, "self_attention_res_proj.weight", .{ .one, .d });
        errdefer result.attention_res_projection.deinit();
        result.input_norm = try self.loadLayer(layer_index, "input_layernorm.weight", .{.d});
        errdefer result.input_norm.deinit();
        result.mlp_res_norm = try self.loadLayer(layer_index, "mlp_res_norm.weight", .{.d});
        errdefer result.mlp_res_norm.deinit();
        result.mlp_res_projection = try self.loadLayer(layer_index, "mlp_res_proj.weight", .{ .one, .d });
        errdefer result.mlp_res_projection.deinit();
        result.post_attention_norm = try self.loadLayer(layer_index, "post_attention_layernorm.weight", .{.d});
        errdefer result.post_attention_norm.deinit();

        result.moe.gate.weight = try self.loadLayer(layer_index, "block_sparse_moe.gate.weight", .{ .expert, .d });
        errdefer result.moe.gate.weight.deinit();
        result.moe.gate.correction_bias = try self.loadLayer(layer_index, "block_sparse_moe.gate.e_score_correction_bias", .{.expert});
        errdefer result.moe.gate.correction_bias.deinit();

        result.moe.experts.w1.values = try self.loadExpertComponent(layer_index, "w1", "weight_packed", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .kw = 1792 }, .u8));
        errdefer result.moe.experts.w1.values.deinit();
        result.moe.experts.w1.scale = try self.loadExpertComponent(layer_index, "w1", "weight_scale", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .block = 112 }, .u8));
        errdefer result.moe.experts.w1.scale.deinit();
        result.moe.experts.w2.values = try self.loadExpertComponent(layer_index, "w2", "weight_packed", zml.Shape.init(.{ .expert = expert_count, .latent = 3584, .kw = 1536 }, .u8));
        errdefer result.moe.experts.w2.values.deinit();
        result.moe.experts.w2.scale = try self.loadExpertComponent(layer_index, "w2", "weight_scale", zml.Shape.init(.{ .expert = expert_count, .latent = 3584, .block = 96 }, .u8));
        errdefer result.moe.experts.w2.scale.deinit();
        result.moe.experts.w3.values = try self.loadExpertComponent(layer_index, "w3", "weight_packed", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .kw = 1792 }, .u8));
        errdefer result.moe.experts.w3.values.deinit();
        result.moe.experts.w3.scale = try self.loadExpertComponent(layer_index, "w3", "weight_scale", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .block = 112 }, .u8));
        errdefer result.moe.experts.w3.scale.deinit();

        result.moe.dense.routed_down = try self.loadLayer(layer_index, "block_sparse_moe.routed_expert_down_proj.weight", .{ .latent, .d });
        errdefer result.moe.dense.routed_down.deinit();
        result.moe.dense.routed_norm = try self.loadLayer(layer_index, "block_sparse_moe.routed_expert_norm.weight", .{.latent});
        errdefer result.moe.dense.routed_norm.deinit();
        result.moe.dense.routed_up = try self.loadLayer(layer_index, "block_sparse_moe.routed_expert_up_proj.weight", .{ .d, .latent });
        errdefer result.moe.dense.routed_up.deinit();
        result.moe.dense.shared_gate = try self.loadLayer(layer_index, "block_sparse_moe.shared_experts.gate_proj.weight", .{ .intermediate, .d });
        errdefer result.moe.dense.shared_gate.deinit();
        result.moe.dense.shared_up = try self.loadLayer(layer_index, "block_sparse_moe.shared_experts.up_proj.weight", .{ .intermediate, .d });
        errdefer result.moe.dense.shared_up.deinit();
        result.moe.dense.shared_down = try self.loadLayer(layer_index, "block_sparse_moe.shared_experts.down_proj.weight", .{ .d, .intermediate });
        return result;
    }

    pub fn loadKdaMoe(self: Loader, layer_index: usize) !zml.Bufferized(layer.KdaMoeWeights) {
        if (self.stats) |stats| stats.layer_loads += 1;
        var result: zml.Bufferized(layer.KdaMoeWeights) = undefined;
        result.common = try self.loadCommon(layer_index);
        errdefer zml.Buffer.deinitAll(layer.MoeLayerWeights, &result.common);
        result.attention.q_weight = try self.loadLayerAs(layer_index, "self_attn.q_proj.weight", zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated }));
        errdefer result.attention.q_weight.deinit();
        result.attention.k_weight = try self.loadLayerAs(layer_index, "self_attn.k_proj.weight", zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated }));
        errdefer result.attention.k_weight.deinit();
        result.attention.v_weight = try self.loadLayerAs(layer_index, "self_attn.v_proj.weight", zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated }));
        errdefer result.attention.v_weight.deinit();
        result.attention.q_conv_weight = try self.loadLayerAs(layer_index, "self_attn.q_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32).withPartitioning(.{ .channel = .model, .kernel = .replicated }));
        errdefer result.attention.q_conv_weight.deinit();
        result.attention.k_conv_weight = try self.loadLayerAs(layer_index, "self_attn.k_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32).withPartitioning(.{ .channel = .model, .kernel = .replicated }));
        errdefer result.attention.k_conv_weight.deinit();
        result.attention.v_conv_weight = try self.loadLayerAs(layer_index, "self_attn.v_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32).withPartitioning(.{ .channel = .model, .kernel = .replicated }));
        errdefer result.attention.v_conv_weight.deinit();
        result.attention.decay_a_weight = try self.loadLayer(layer_index, "self_attn.f_a_proj.weight", .{ .out, .d });
        errdefer result.attention.decay_a_weight.deinit();
        result.attention.decay_b_weight = try self.loadLayerAs(layer_index, "self_attn.f_b_proj.weight", zml.Shape.init(.{ .channel = 12288, .rank = 128 }, .bf16).withPartitioning(.{ .channel = .model, .rank = .replicated }));
        errdefer result.attention.decay_b_weight.deinit();
        result.attention.a_log = try self.loadLayer(layer_index, "self_attn.A_log", .{.h});
        errdefer result.attention.a_log.deinit();
        result.attention.dt_bias = try self.loadLayerAs(layer_index, "self_attn.dt_bias", zml.Shape.init(.{ .h = 96, .k = 128 }, .f32).withPartitioning(.{ .h = .model, .k = .replicated }));
        errdefer result.attention.dt_bias.deinit();
        result.attention.beta_weight = try self.loadLayerAs(layer_index, "self_attn.b_proj.weight", zml.Shape.init(.{ .out = 96, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated }));
        errdefer result.attention.beta_weight.deinit();
        result.attention.gate_weight = try self.loadLayerAs(layer_index, "self_attn.g_proj.weight", zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated }));
        errdefer result.attention.gate_weight.deinit();
        result.attention.norm_weight = try self.loadLayer(layer_index, "self_attn.o_norm.weight", .{.v});
        errdefer result.attention.norm_weight.deinit();
        result.attention.output_weight = try self.loadLayerAs(layer_index, "self_attn.o_proj.weight", zml.Shape.init(.{ .d = 7168, .out = 12288 }, .bf16).withPartitioning(.{ .d = .replicated, .out = .model }));
        return result;
    }

    pub fn loadMlaMoe(self: Loader, layer_index: usize) !zml.Bufferized(layer.MlaMoeWeights) {
        if (self.stats) |stats| stats.layer_loads += 1;
        var result: zml.Bufferized(layer.MlaMoeWeights) = undefined;
        result.common = try self.loadCommon(layer_index);
        errdefer zml.Buffer.deinitAll(layer.MoeLayerWeights, &result.common);
        result.attention.q_a_proj = try self.loadLayer(layer_index, "self_attn.q_a_proj.weight", .{ .rank, .d });
        errdefer result.attention.q_a_proj.deinit();
        result.attention.q_a_norm = try self.loadLayer(layer_index, "self_attn.q_a_layernorm.weight", .{.rank});
        errdefer result.attention.q_a_norm.deinit();
        result.attention.q_b_proj = try self.loadLayerAs(layer_index, "self_attn.q_b_proj.weight", zml.Shape.init(.{ .mix = 18432, .rank = 1536 }, .bf16).withPartitioning(.{ .mix = .model, .rank = .replicated }));
        errdefer result.attention.q_b_proj.deinit();
        result.attention.kv_a_proj = try self.loadLayer(layer_index, "self_attn.kv_a_proj_with_mqa.weight", .{ .kv_mix, .d });
        errdefer result.attention.kv_a_proj.deinit();
        result.attention.kv_a_norm = try self.loadLayer(layer_index, "self_attn.kv_a_layernorm.weight", .{.kv_rank});
        errdefer result.attention.kv_a_norm.deinit();
        result.attention.kv_b_proj = try self.loadLayerAs(layer_index, "self_attn.kv_b_proj.weight", zml.Shape.init(.{ .kv_mix = 24576, .kv_rank = 512 }, .bf16).withPartitioning(.{ .kv_mix = .model, .kv_rank = .replicated }));
        errdefer result.attention.kv_b_proj.deinit();
        result.attention.gate_proj = try self.loadLayerAs(layer_index, "self_attn.g_proj.weight", zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated }));
        errdefer result.attention.gate_proj.deinit();
        result.attention.output_proj = try self.loadLayerAs(layer_index, "self_attn.o_proj.weight", zml.Shape.init(.{ .d = 7168, .out = 12288 }, .bf16).withPartitioning(.{ .d = .replicated, .out = .model }));
        return result;
    }

    pub fn zeroKdaCache(self: Loader) !zml.Bufferized(kda.Cache) {
        return .{
            .q_conv = try zeroBuffer(self, zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16).withPartitioning(.{ .b = .replicated, .channel = .model, .kernel = .replicated })),
            .k_conv = try zeroBuffer(self, zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16).withPartitioning(.{ .b = .replicated, .channel = .model, .kernel = .replicated })),
            .v_conv = try zeroBuffer(self, zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16).withPartitioning(.{ .b = .replicated, .channel = .model, .kernel = .replicated })),
            .recurrent_state = try zeroBuffer(self, zml.Shape.init(.{ .b = 1, .h = 96, .v = 128, .k = 128 }, .f32).withPartitioning(.{ .b = .replicated, .h = .model, .v = .replicated, .k = .replicated })),
        };
    }

    pub fn zeroMlaCache(self: Loader, capacity: usize) !zml.Bufferized(mla.SessionCache) {
        return .{
            .compressed = try zeroBuffer(self, zml.Shape.init(.{ .b = 1, .k = capacity, .kv_rank = 512 }, .bf16)),
            .extra_key = try zeroBuffer(self, zml.Shape.init(.{ .b = 1, .k = capacity, .hd = 64 }, .bf16)),
        };
    }

    pub fn zeroBlocks(self: Loader, source_slots: usize) !zml.Buffer {
        return zeroBuffer(self, zml.Shape.init(.{ .token = 1, .source = source_slots, .d = 7168 }, .bf16));
    }

    fn zeroBuffer(self: Loader, shape: zml.Shape) !zml.Buffer {
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        @memset(bytes, 0);
        return zml.Buffer.fromBytes(self.io, self.platform, shape, self.model_sharding, bytes);
    }
};

pub fn tensor(buffer: zml.Buffer) zml.Tensor {
    return .fromShape(buffer.shape());
}

fn symbolicCommon() layer.MoeLayerWeights {
    return .{
        .attention_res_norm = .init(.{ .d = 7168 }, .bf16),
        .attention_res_projection = .init(.{ .one = 1, .d = 7168 }, .bf16),
        .input_norm = .init(.{ .d = 7168 }, .bf16),
        .mlp_res_norm = .init(.{ .d = 7168 }, .bf16),
        .mlp_res_projection = .init(.{ .one = 1, .d = 7168 }, .bf16),
        .post_attention_norm = .init(.{ .d = 7168 }, .bf16),
        .moe = .{
            .gate = .{
                .weight = .init(.{ .expert = expert_count, .d = 7168 }, .bf16),
                .correction_bias = .init(.{ .expert = expert_count }, .f32),
            },
            .experts = .{
                .w1 = .{
                    .values = .init(.{ .expert = expert_count, .intermediate = 3072, .kw = 1792 }, .u8),
                    .scale = .init(.{ .expert = expert_count, .intermediate = 3072, .block = 112 }, .u8),
                },
                .w2 = .{
                    .values = .init(.{ .expert = expert_count, .latent = 3584, .kw = 1536 }, .u8),
                    .scale = .init(.{ .expert = expert_count, .latent = 3584, .block = 96 }, .u8),
                },
                .w3 = .{
                    .values = .init(.{ .expert = expert_count, .intermediate = 3072, .kw = 1792 }, .u8),
                    .scale = .init(.{ .expert = expert_count, .intermediate = 3072, .block = 112 }, .u8),
                },
            },
            .dense = .{
                .routed_down = .init(.{ .latent = 3584, .d = 7168 }, .bf16),
                .routed_norm = .init(.{ .latent = 3584 }, .bf16),
                .routed_up = .init(.{ .d = 7168, .latent = 3584 }, .bf16),
                .shared_gate = .init(.{ .intermediate = 6144, .d = 7168 }, .bf16),
                .shared_up = .init(.{ .intermediate = 6144, .d = 7168 }, .bf16),
                .shared_down = .init(.{ .d = 7168, .intermediate = 6144 }, .bf16),
            },
        },
    };
}

pub fn symbolicKdaMoe() layer.KdaMoeWeights {
    return .{
        .common = symbolicCommon(),
        .attention = .{
            .q_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated })),
            .k_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated })),
            .v_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated })),
            .q_conv_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32).withPartitioning(.{ .channel = .model, .kernel = .replicated })),
            .k_conv_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32).withPartitioning(.{ .channel = .model, .kernel = .replicated })),
            .v_conv_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32).withPartitioning(.{ .channel = .model, .kernel = .replicated })),
            .decay_a_weight = .init(.{ .out = 128, .d = 7168 }, .bf16),
            .decay_b_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .channel = 12288, .rank = 128 }, .bf16).withPartitioning(.{ .channel = .model, .rank = .replicated })),
            .a_log = .init(.{ .h = 128 }, .f32),
            .dt_bias = zml.Tensor.fromShape(zml.Shape.init(.{ .h = 96, .k = 128 }, .f32).withPartitioning(.{ .h = .model, .k = .replicated })),
            .beta_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .out = 96, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated })),
            .gate_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated })),
            .norm_weight = .init(.{ .v = 128 }, .f32),
            .output_weight = zml.Tensor.fromShape(zml.Shape.init(.{ .d = 7168, .out = 12288 }, .bf16).withPartitioning(.{ .d = .replicated, .out = .model })),
        },
    };
}

pub fn symbolicMlaMoe() layer.MlaMoeWeights {
    return .{
        .common = symbolicCommon(),
        .attention = .{
            .q_a_proj = .init(.{ .rank = 1536, .d = 7168 }, .bf16),
            .q_a_norm = .init(.{ .rank = 1536 }, .bf16),
            .q_b_proj = zml.Tensor.fromShape(zml.Shape.init(.{ .mix = 18432, .rank = 1536 }, .bf16).withPartitioning(.{ .mix = .model, .rank = .replicated })),
            .kv_a_proj = .init(.{ .kv_mix = 576, .d = 7168 }, .bf16),
            .kv_a_norm = .init(.{ .kv_rank = 512 }, .bf16),
            .kv_b_proj = zml.Tensor.fromShape(zml.Shape.init(.{ .kv_mix = 24576, .kv_rank = 512 }, .bf16).withPartitioning(.{ .kv_mix = .model, .kv_rank = .replicated })),
            .gate_proj = zml.Tensor.fromShape(zml.Shape.init(.{ .out = 12288, .d = 7168 }, .bf16).withPartitioning(.{ .out = .model, .d = .replicated })),
            .output_proj = zml.Tensor.fromShape(zml.Shape.init(.{ .d = 7168, .out = 12288 }, .bf16).withPartitioning(.{ .d = .replicated, .out = .model })),
        },
    };
}

pub fn symbolicKdaCache() kda.Cache {
    return .{
        .q_conv = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16).withPartitioning(.{ .b = .replicated, .channel = .model, .kernel = .replicated })),
        .k_conv = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16).withPartitioning(.{ .b = .replicated, .channel = .model, .kernel = .replicated })),
        .v_conv = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16).withPartitioning(.{ .b = .replicated, .channel = .model, .kernel = .replicated })),
        .recurrent_state = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .h = 96, .v = 128, .k = 128 }, .f32).withPartitioning(.{ .b = .replicated, .h = .model, .v = .replicated, .k = .replicated })),
    };
}

pub fn symbolicMlaCache(capacity: usize) mla.SessionCache {
    return .{
        .compressed = .init(.{ .b = 1, .k = capacity, .kv_rank = 512 }, .bf16),
        .extra_key = .init(.{ .b = 1, .k = capacity, .hd = 64 }, .bf16),
    };
}
