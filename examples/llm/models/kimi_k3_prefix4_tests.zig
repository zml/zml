const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");
const layer = @import("kimi_k3/layer.zig");
const mla = @import("kimi_k3/mla.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(1_200_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const expert_count: usize = 896;
const route_weight_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 2e-3,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 1.0,
};

const Args = struct {
    weights: []const u8,
    layer0_fixture: []const u8,
    layer_family_fixture: []const u8,
    prefix4_fixture: []const u8,
    head_only: bool = false,
    profile: bool = false,
    profile_repository: []const u8 = "/tmp/kimi-k3-profile",
    profile_session: []const u8 = "milestone-15-prefix4",

    pub const help =
        \\Use kimi_k3_prefix4_tests --weights=<S4-directory> [fixture arguments]
        \\
        \\Run a four-layer Kimi K3 prefill and cached continuation with all 896
        \\experts per MoE layer. NVIDIA CUDA is mandatory; CPU inference is absent.
        \\
    ;
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    weights: zml.io.TensorStore.View,
    layer0_fixture: zml.io.TensorStore.View,
    family_fixture: zml.io.TensorStore.View,
    prefix4_fixture: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,

    fn weightKey(self: *Context, layer_index: usize, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(
            self.allocator,
            "language_model.model.layers.{d}.{s}",
            .{ layer_index, suffix },
        );
    }

    fn loadWeight(self: *Context, layer_index: usize, suffix: []const u8, tags: anytype) !zml.Buffer {
        const key = try self.weightKey(layer_index, suffix);
        defer self.allocator.free(key);
        return support.loadBuffer(
            self.allocator,
            self.io,
            self.platform,
            self.weights,
            key,
            tags,
            self.sharding,
        );
    }

    fn loadWeightAs(self: *Context, layer_index: usize, suffix: []const u8, target: zml.Shape) !zml.Buffer {
        const key = try self.weightKey(layer_index, suffix);
        defer self.allocator.free(key);
        const source = self.weights.getShape(key) orelse return error.MissingPrefix4Weight;
        if (source.byteSize() != target.byteSize()) return error.Prefix4WeightReshapeMismatch;
        const bytes = try self.allocator.alloc(u8, source.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.weights.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, target, self.sharding, bytes);
    }

    // KIMI_K3_TEMP_REMOVE_M20: the progress markers explain long, bounded
    // full-bank staging and are removed when production model loading owns it.
    fn loadExpertComponent(
        self: *Context,
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
            const key = try self.weightKey(layer_index, suffix);
            defer self.allocator.free(key);
            const source = self.weights.getShape(key) orelse return error.MissingPrefix4Expert;
            if (source.byteSize() != per_expert) return error.Prefix4ExpertShapeMismatch;
            var reader = try self.weights.getReader(key, self.io, &io_buffer);
            defer reader.deinit();
            _ = try reader.interface.readSliceAll(bytes[expert * per_expert ..][0..per_expert]);
            if ((expert + 1) % 128 == 0) {
                try self.stdout.print(
                    "KIMI_K3_PREFIX4_LOAD layer={} projection={s} component={s} experts={}/{} host_bytes={}\n",
                    .{ layer_index, projection, component, expert + 1, expert_count, target.byteSize() },
                );
                try self.stdout.flush();
            }
        }
        return zml.Buffer.fromBytes(self.io, self.platform, target, self.sharding, bytes);
    }

    fn compare(self: *Context, store: zml.io.TensorStore.View, key: []const u8, actual: zml.Buffer, opts: zml.testing.CompareOpts) !void {
        // KIMI_K3_TEMP_REMOVE_M20: boundary markers identify the first divergent
        // activation and are removed with the differential harness in cleanup.
        try self.stdout.print("KIMI_K3_PREFIX4_CHECK key={s}\n", .{key});
        try self.stdout.flush();
        try support.compare(
            self.allocator,
            self.io,
            self.platform,
            store,
            key,
            actual,
            opts,
            self.sharding,
        );
    }

    fn loadCommon(self: *Context, layer_index: usize) !zml.Bufferized(layer.MoeLayerWeights) {
        return .{
            .attention_res_norm = try self.loadWeight(layer_index, "self_attention_res_norm.weight", .{.d}),
            .attention_res_projection = try self.loadWeight(layer_index, "self_attention_res_proj.weight", .{ .one, .d }),
            .input_norm = try self.loadWeight(layer_index, "input_layernorm.weight", .{.d}),
            .mlp_res_norm = try self.loadWeight(layer_index, "mlp_res_norm.weight", .{.d}),
            .mlp_res_projection = try self.loadWeight(layer_index, "mlp_res_proj.weight", .{ .one, .d }),
            .post_attention_norm = try self.loadWeight(layer_index, "post_attention_layernorm.weight", .{.d}),
            .moe = .{
                .gate = .{
                    .weight = try self.loadWeight(layer_index, "block_sparse_moe.gate.weight", .{ .expert, .d }),
                    .correction_bias = try self.loadWeight(layer_index, "block_sparse_moe.gate.e_score_correction_bias", .{.expert}),
                },
                .experts = .{
                    .w1 = .{
                        .values = try self.loadExpertComponent(layer_index, "w1", "weight_packed", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .kw = 1792 }, .u8)),
                        .scale = try self.loadExpertComponent(layer_index, "w1", "weight_scale", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .block = 112 }, .u8)),
                    },
                    .w2 = .{
                        .values = try self.loadExpertComponent(layer_index, "w2", "weight_packed", zml.Shape.init(.{ .expert = expert_count, .latent = 3584, .kw = 1536 }, .u8)),
                        .scale = try self.loadExpertComponent(layer_index, "w2", "weight_scale", zml.Shape.init(.{ .expert = expert_count, .latent = 3584, .block = 96 }, .u8)),
                    },
                    .w3 = .{
                        .values = try self.loadExpertComponent(layer_index, "w3", "weight_packed", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .kw = 1792 }, .u8)),
                        .scale = try self.loadExpertComponent(layer_index, "w3", "weight_scale", zml.Shape.init(.{ .expert = expert_count, .intermediate = 3072, .block = 112 }, .u8)),
                    },
                },
                .dense = .{
                    .routed_down = try self.loadWeight(layer_index, "block_sparse_moe.routed_expert_down_proj.weight", .{ .latent, .d }),
                    .routed_norm = try self.loadWeight(layer_index, "block_sparse_moe.routed_expert_norm.weight", .{.latent}),
                    .routed_up = try self.loadWeight(layer_index, "block_sparse_moe.routed_expert_up_proj.weight", .{ .d, .latent }),
                    .shared_gate = try self.loadWeight(layer_index, "block_sparse_moe.shared_experts.gate_proj.weight", .{ .intermediate, .d }),
                    .shared_up = try self.loadWeight(layer_index, "block_sparse_moe.shared_experts.up_proj.weight", .{ .intermediate, .d }),
                    .shared_down = try self.loadWeight(layer_index, "block_sparse_moe.shared_experts.down_proj.weight", .{ .d, .intermediate }),
                },
            },
        };
    }

    fn loadKdaWeights(self: *Context, layer_index: usize) !zml.Bufferized(layer.KdaMoeWeights) {
        return .{
            .common = try self.loadCommon(layer_index),
            .attention = .{
                .q_weight = try self.loadWeight(layer_index, "self_attn.q_proj.weight", .{ .out, .d }),
                .k_weight = try self.loadWeight(layer_index, "self_attn.k_proj.weight", .{ .out, .d }),
                .v_weight = try self.loadWeight(layer_index, "self_attn.v_proj.weight", .{ .out, .d }),
                .q_conv_weight = try self.loadWeightAs(layer_index, "self_attn.q_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32)),
                .k_conv_weight = try self.loadWeightAs(layer_index, "self_attn.k_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32)),
                .v_conv_weight = try self.loadWeightAs(layer_index, "self_attn.v_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32)),
                .decay_a_weight = try self.loadWeight(layer_index, "self_attn.f_a_proj.weight", .{ .out, .d }),
                .decay_b_weight = try self.loadWeight(layer_index, "self_attn.f_b_proj.weight", .{ .channel, .rank }),
                .a_log = try self.loadWeight(layer_index, "self_attn.A_log", .{.h}),
                .dt_bias = try self.loadWeightAs(layer_index, "self_attn.dt_bias", zml.Shape.init(.{ .h = 96, .k = 128 }, .f32)),
                .beta_weight = try self.loadWeight(layer_index, "self_attn.b_proj.weight", .{ .out, .d }),
                .gate_weight = try self.loadWeight(layer_index, "self_attn.g_proj.weight", .{ .out, .d }),
                .norm_weight = try self.loadWeight(layer_index, "self_attn.o_norm.weight", .{.v}),
                .output_weight = try self.loadWeight(layer_index, "self_attn.o_proj.weight", .{ .d, .out }),
            },
        };
    }

    fn loadMlaWeights(self: *Context, layer_index: usize) !zml.Bufferized(layer.MlaMoeWeights) {
        return .{
            .common = try self.loadCommon(layer_index),
            .attention = .{
                .q_a_proj = try self.loadWeight(layer_index, "self_attn.q_a_proj.weight", .{ .rank, .d }),
                .q_a_norm = try self.loadWeight(layer_index, "self_attn.q_a_layernorm.weight", .{.rank}),
                .q_b_proj = try self.loadWeight(layer_index, "self_attn.q_b_proj.weight", .{ .mix, .rank }),
                .kv_a_proj = try self.loadWeight(layer_index, "self_attn.kv_a_proj_with_mqa.weight", .{ .kv_mix, .d }),
                .kv_a_norm = try self.loadWeight(layer_index, "self_attn.kv_a_layernorm.weight", .{.kv_rank}),
                .kv_b_proj = try self.loadWeight(layer_index, "self_attn.kv_b_proj.weight", .{ .kv_mix, .kv_rank }),
                .gate_proj = try self.loadWeight(layer_index, "self_attn.g_proj.weight", .{ .out, .d }),
                .output_proj = try self.loadWeight(layer_index, "self_attn.o_proj.weight", .{ .d, .out }),
            },
        };
    }

    fn zeroKdaCache(self: *Context) !zml.Bufferized(kda.Cache) {
        return .{
            .q_conv = try support.zeroBuffer(self.allocator, self.io, self.platform, zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16), self.sharding),
            .k_conv = try support.zeroBuffer(self.allocator, self.io, self.platform, zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16), self.sharding),
            .v_conv = try support.zeroBuffer(self.allocator, self.io, self.platform, zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16), self.sharding),
            .recurrent_state = try support.zeroBuffer(self.allocator, self.io, self.platform, zml.Shape.init(.{ .b = 1, .h = 96, .v = 128, .k = 128 }, .f32), self.sharding),
        };
    }
};

fn tensor(buffer: zml.Buffer) zml.Tensor {
    return .fromShape(buffer.shape());
}

fn commonTensors(weights: zml.Bufferized(layer.MoeLayerWeights)) layer.MoeLayerWeights {
    return .{
        .attention_res_norm = tensor(weights.attention_res_norm),
        .attention_res_projection = tensor(weights.attention_res_projection),
        .input_norm = tensor(weights.input_norm),
        .mlp_res_norm = tensor(weights.mlp_res_norm),
        .mlp_res_projection = tensor(weights.mlp_res_projection),
        .post_attention_norm = tensor(weights.post_attention_norm),
        .moe = .{
            .gate = .{ .weight = tensor(weights.moe.gate.weight), .correction_bias = tensor(weights.moe.gate.correction_bias) },
            .experts = .{
                .w1 = .{ .values = tensor(weights.moe.experts.w1.values), .scale = tensor(weights.moe.experts.w1.scale) },
                .w2 = .{ .values = tensor(weights.moe.experts.w2.values), .scale = tensor(weights.moe.experts.w2.scale) },
                .w3 = .{ .values = tensor(weights.moe.experts.w3.values), .scale = tensor(weights.moe.experts.w3.scale) },
            },
            .dense = .{
                .routed_down = tensor(weights.moe.dense.routed_down),
                .routed_norm = tensor(weights.moe.dense.routed_norm),
                .routed_up = tensor(weights.moe.dense.routed_up),
                .shared_gate = tensor(weights.moe.dense.shared_gate),
                .shared_up = tensor(weights.moe.dense.shared_up),
                .shared_down = tensor(weights.moe.dense.shared_down),
            },
        },
    };
}

fn kdaTensors(weights: zml.Bufferized(layer.KdaMoeWeights)) layer.KdaMoeWeights {
    return .{
        .common = commonTensors(weights.common),
        .attention = .{
            .q_weight = tensor(weights.attention.q_weight), .k_weight = tensor(weights.attention.k_weight),
            .v_weight = tensor(weights.attention.v_weight), .q_conv_weight = tensor(weights.attention.q_conv_weight),
            .k_conv_weight = tensor(weights.attention.k_conv_weight), .v_conv_weight = tensor(weights.attention.v_conv_weight),
            .decay_a_weight = tensor(weights.attention.decay_a_weight), .decay_b_weight = tensor(weights.attention.decay_b_weight),
            .a_log = tensor(weights.attention.a_log), .dt_bias = tensor(weights.attention.dt_bias),
            .beta_weight = tensor(weights.attention.beta_weight), .gate_weight = tensor(weights.attention.gate_weight),
            .norm_weight = tensor(weights.attention.norm_weight), .output_weight = tensor(weights.attention.output_weight),
        },
    };
}

fn mlaTensors(weights: zml.Bufferized(layer.MlaMoeWeights)) layer.MlaMoeWeights {
    return .{
        .common = commonTensors(weights.common),
        .attention = .{
            .q_a_proj = tensor(weights.attention.q_a_proj), .q_a_norm = tensor(weights.attention.q_a_norm),
            .q_b_proj = tensor(weights.attention.q_b_proj), .kv_a_proj = tensor(weights.attention.kv_a_proj),
            .kv_a_norm = tensor(weights.attention.kv_a_norm), .kv_b_proj = tensor(weights.attention.kv_b_proj),
            .gate_proj = tensor(weights.attention.gate_proj), .output_proj = tensor(weights.attention.output_proj),
        },
    };
}

fn kdaCacheTensors(cache: zml.Bufferized(kda.Cache)) kda.Cache {
    return .{
        .q_conv = tensor(cache.q_conv), .k_conv = tensor(cache.k_conv),
        .v_conv = tensor(cache.v_conv), .recurrent_state = tensor(cache.recurrent_state),
    };
}

fn mlaCacheTensors(cache: zml.Bufferized(mla.LatentCache)) mla.LatentCache {
    return .{ .compressed = tensor(cache.compressed), .extra_key = tensor(cache.extra_key) };
}

const ExpandedCache = struct { key: zml.Tensor, value: zml.Tensor };

fn expandLatent(cache: mla.LatentCache, weights: mla.Weights) ExpandedCache {
    const kv = cache.compressed.dot(weights.kv_b_proj, .kv_rank);
    const heads: i64 = 96;
    const split = kv.splitAxis(.kv_mix, .{ .h = heads, .kv_width = 256 })
        .transpose(.{ .b, .h, .k, .kv_width });
    const pass = split.slice1d(.kv_width, .{ .start = 0, .end = 128 }).rename(.{ .kv_width = .hd });
    const value = split.slice1d(.kv_width, .{ .start = 128, .end = 256 }).rename(.{ .kv_width = .v });
    const extra = cache.extra_key.reshape(.{
        .b = cache.extra_key.dim(.b), .h = 1, .k = cache.extra_key.dim(.k), .hd = 64,
    }).broad(zml.Shape.init(.{
        .b = cache.extra_key.dim(.b), .h = heads, .k = cache.extra_key.dim(.k), .hd = 64,
    }, cache.extra_key.dtype()));
    return .{ .key = zml.Tensor.concatenate(&.{ pass, extra }, .hd), .value = value };
}

const RouteAlignment = struct {
    matched_ids: zml.Tensor,
    aligned_weights: zml.Tensor,
    aligned_outputs: zml.Tensor,
};

// KIMI_K3_TEMP_REMOVE_M20: top-K ordering is implementation-defined. This
// diagnostic proves the exact global expert set and aligns weights/outputs by
// expert ID; production routing and aggregation remain unchanged.
fn alignRoute(actual_ids_raw: zml.Tensor, actual_weights_raw: zml.Tensor, outputs_raw: zml.Tensor, expected_ids: zml.Tensor) RouteAlignment {
    const actual_ids = actual_ids_raw.rename(.{ .route = .actual_route });
    const actual_weights = actual_weights_raw.rename(.{ .route = .actual_route });
    const match_shape = zml.Shape.init(.{
        .b = expected_ids.dim(.b), .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route),
    }, .i64);
    const expected_grid = expected_ids.reshape(.{
        .b = expected_ids.dim(.b), .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route), .actual_route = 1,
    }).broad(match_shape);
    const actual_grid = actual_ids.reshape(.{
        .b = actual_ids.dim(.b), .s = actual_ids.dim(.s),
        .expected_route = 1, .actual_route = actual_ids.dim(.actual_route),
    }).broad(match_shape);
    const matches = actual_grid.cmp(.EQ, expected_grid);
    const found = matches.convert(.i32).sum(.actual_route).squeeze(.actual_route)
        .cmp(.GT, zml.Tensor.scalar(0, .i32));
    const weight_grid = actual_weights.reshape(.{
        .b = actual_ids.dim(.b), .s = actual_ids.dim(.s),
        .expected_route = 1, .actual_route = actual_ids.dim(.actual_route),
    }).broad(match_shape.withDtype(.f32));
    const outputs = outputs_raw.rename(.{ .route = .actual_route });
    const output_shape = zml.Shape.init(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route),
        .latent = outputs.dim(.latent),
    }, outputs.dtype());
    const output_grid = outputs.reshape(.{
        .b = actual_ids.dim(.b), .s = actual_ids.dim(.s),
        .expected_route = 1, .actual_route = actual_ids.dim(.actual_route),
        .latent = outputs.dim(.latent),
    }).broad(output_shape);
    const output_matches = matches.reshape(.{
        .b = expected_ids.dim(.b), .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route), .latent = 1,
    }).broad(output_shape.withDtype(.bool));
    return .{
        .matched_ids = found.select(expected_ids, zml.Tensor.scalar(-1, .i64).broad(expected_ids.shape())),
        .aligned_weights = matches.convert(.f32).mul(weight_grid).sum(.actual_route).squeeze(.actual_route),
        .aligned_outputs = output_matches.select(output_grid, zml.Tensor.scalar(0, outputs.dtype()).broad(output_shape))
            .sum(.actual_route).squeeze(.actual_route).merge(.{ .token = .{ .b, .s } })
            .rename(.{ .expected_route = .route }),
    };
}

const EmbeddingSplit = struct { warm: zml.Tensor, decode: zml.Tensor };

fn embed(tokens: zml.Tensor, embedding: zml.Tensor) zml.Tensor {
    return embedding.gather(.{ .voc = tokens.convert(.u32) }, .{});
}

fn splitEmbedding(embedding: zml.Tensor) EmbeddingSplit {
    return .{
        .warm = embedding.slice1d(.s, .{ .start = 0, .end = 3 }),
        .decode = embedding.slice1d(.s, .{ .start = 3, .end = 4 }),
    };
}

fn makeActive(context: *Context) !zml.Buffer {
    const yes = [_]u8{1};
    return zml.Buffer.fromBytes(
        context.io,
        context.platform,
        zml.Shape.init(.{ .source = 1 }, .bool),
        context.sharding,
        &yes,
    );
}

fn kdaPrefill(input: zml.Tensor, blocks: zml.Tensor, active: zml.Tensor, weights: layer.KdaMoeWeights, cache: kda.Cache) layer.KdaMoeResult {
    return layer.forwardKdaMoePrefill(input, blocks, active, weights, cache, .{ .top_k = 16 });
}

fn kdaDecode(input: zml.Tensor, blocks: zml.Tensor, active: zml.Tensor, weights: layer.KdaMoeWeights, cache: kda.Cache) layer.KdaMoeResult {
    return layer.forwardKdaMoeDecode(input, blocks, active, weights, cache, .{ .top_k = 16 });
}

fn mlaPrefill(input: zml.Tensor, blocks: zml.Tensor, active: zml.Tensor, weights: layer.MlaMoeWeights) layer.MlaMoeResult {
    return layer.forwardMlaMoePrefill(input, blocks, active, weights, .{ .top_k = 16 });
}

fn mlaContinue(input: zml.Tensor, blocks: zml.Tensor, active: zml.Tensor, weights: layer.MlaMoeWeights, cache: mla.LatentCache) layer.MlaMoeResult {
    return layer.forwardMlaMoeContinue(input, blocks, active, weights, cache, .{ .top_k = 16 });
}

const KdaStreams = struct {
    full: zml.Bufferized(layer.KdaMoeResult),
    warm: zml.Bufferized(layer.KdaMoeResult),
    decode: zml.Bufferized(layer.KdaMoeResult),
};

const MlaStreams = struct {
    full: zml.Bufferized(layer.MlaMoeResult),
    warm: zml.Bufferized(layer.MlaMoeResult),
    decode: zml.Bufferized(layer.MlaMoeResult),
};

fn deinitKdaStreams(streams: *KdaStreams) void {
    zml.Buffer.deinitAll(layer.KdaMoeResult, &streams.full);
    zml.Buffer.deinitAll(layer.KdaMoeResult, &streams.warm);
    zml.Buffer.deinitAll(layer.KdaMoeResult, &streams.decode);
}

fn deinitMlaStreams(streams: *MlaStreams) void {
    zml.Buffer.deinitAll(layer.MlaMoeResult, &streams.full);
    zml.Buffer.deinitAll(layer.MlaMoeResult, &streams.warm);
    zml.Buffer.deinitAll(layer.MlaMoeResult, &streams.decode);
}

const RouteOverlap = struct { matched: usize, total: usize };

fn measureRouteOverlap(context: *Context, actual: zml.Buffer, expected: zml.Buffer) !RouteOverlap {
    var actual_host = try actual.toSliceAlloc(context.allocator, context.io);
    defer actual_host.free(context.allocator);
    var expected_host = try expected.toSliceAlloc(context.allocator, context.io);
    defer expected_host.free(context.allocator);
    const actual_ids = actual_host.items(i64);
    const expected_ids = expected_host.items(i64);
    if (actual_ids.len != expected_ids.len or actual_ids.len % 16 != 0) {
        return error.InvalidPrefix4RouteShape;
    }
    var matched: usize = 0;
    for (expected_ids, 0..) |expected_id, index| {
        const base = @divFloor(index, 16) * 16;
        for (actual_ids[base..][0..16]) |actual_id| {
            if (actual_id == expected_id) {
                matched += 1;
                break;
            }
        }
    }
    return .{ .matched = matched, .total = expected_ids.len };
}

fn compareMoeResult(context: *Context, store: zml.io.TensorStore.View, prefix: []const u8, result: anytype) !void {
    const names = [_][]const u8{
        "selected_input", "input_norm", "attention_output", "prefix_after_attention",
        "selected_mlp", "moe_input", "moe.routed_down",
        "moe.combined_latent", "moe.routed_norm", "moe.routed_up", "moe.shared_output",
        "moe.output", "output",
    };
    const values = .{
        result.selected_input, result.input_norm, result.attention_output, result.prefix_after_attention,
        result.selected_mlp, result.moe_input, result.moe_result.routed_down,
        result.moe_result.combined_latent, result.moe_result.routed_norm, result.moe_result.routed_up,
        result.moe_result.shared_output, result.moe_result.output, result.output,
    };
    inline for (names, values) |name, value| {
        const key = try std.fmt.allocPrint(context.allocator, "{s}.{s}", .{ prefix, name });
        defer context.allocator.free(key);
        try context.compare(store, key, value, support.bf16_tolerance);
    }
    const ids_key = try std.fmt.allocPrint(context.allocator, "{s}.route.global_ids", .{prefix});
    defer context.allocator.free(ids_key);
    var expected_ids = try support.loadBuffer(
        context.allocator,
        context.io,
        context.platform,
        store,
        ids_key,
        .{ .b, .s, .expected_route },
        context.sharding,
    );
    defer expected_ids.deinit();
    const overlap = try measureRouteOverlap(context, result.moe_result.route.topk_ids, expected_ids);
    try context.stdout.print(
        "KIMI_K3_PREFIX4_ROUTE_OVERLAP prefix={s} matched={} total={} fraction={d:.4}\n",
        .{ prefix, overlap.matched, overlap.total, @as(f64, @floatFromInt(overlap.matched)) / @as(f64, @floatFromInt(overlap.total)) },
    );
    try context.stdout.flush();
    if (overlap.matched * 100 < overlap.total * 85) return error.InsufficientPrefix4RouteOverlap;
    if (overlap.matched == overlap.total) {
        const align_exe = try context.platform.compileFn(
            context.allocator,
            context.io,
            alignRoute,
            .{
                tensor(result.moe_result.route.topk_ids),
                tensor(result.moe_result.route.topk_weights),
                tensor(result.moe_result.route_outputs),
                tensor(expected_ids),
            },
            .{ .shardings = &.{context.sharding} },
        );
        defer align_exe.deinit();
        var aligned = try zml.testing.autoCall(
            context.allocator,
            context.io,
            &align_exe,
            alignRoute,
            .{ result.moe_result.route.topk_ids, result.moe_result.route.topk_weights, result.moe_result.route_outputs, expected_ids },
        );
        defer zml.Buffer.deinitAll(RouteAlignment, &aligned);
        try context.compare(store, ids_key, aligned.matched_ids, .{});
        const weights_key = try std.fmt.allocPrint(context.allocator, "{s}.route.weights", .{prefix});
        defer context.allocator.free(weights_key);
        try context.compare(store, weights_key, aligned.aligned_weights, route_weight_tolerance);
        const outputs_key = try std.fmt.allocPrint(context.allocator, "{s}.moe.route_outputs", .{prefix});
        defer context.allocator.free(outputs_key);
        try context.compare(store, outputs_key, aligned.aligned_outputs, support.bf16_tolerance);
    }
}

fn compareKdaCache(context: *Context, store: zml.io.TensorStore.View, prefix: []const u8, cache: zml.Bufferized(kda.Cache)) !void {
    const values = .{ cache.q_conv, cache.k_conv, cache.v_conv, cache.recurrent_state };
    const suffixes = [_][]const u8{ "conv0", "conv1", "conv2", "recurrent" };
    inline for (suffixes, values, 0..) |suffix, value, index| {
        const key = try std.fmt.allocPrint(context.allocator, "{s}.{s}", .{ prefix, suffix });
        defer context.allocator.free(key);
        try context.compare(store, key, value, if (index == 3) support.state_tolerance else support.bf16_tolerance);
    }
}

fn runKda(
    context: *Context,
    layer_index: usize,
    full_input: zml.Buffer,
    warm_input: zml.Buffer,
    decode_input: zml.Buffer,
    full_blocks: zml.Buffer,
    warm_blocks: zml.Buffer,
    decode_blocks: zml.Buffer,
) !KdaStreams {
    const load_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var weights = try context.loadKdaWeights(layer_index);
    defer zml.Buffer.deinitAll(layer.KdaMoeWeights, &weights);
    const load_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - load_started, 1000);
    var active = try makeActive(context);
    defer active.deinit();
    var zero_cache = try context.zeroKdaCache();
    defer zml.Buffer.deinitAll(kda.Cache, &zero_cache);
    const symbolic = kdaTensors(weights);
    const cache_symbolic = kdaCacheTensors(zero_cache);

    const compile_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    const full_exe = try context.platform.compileFn(context.allocator, context.io, kdaPrefill, .{ tensor(full_input), tensor(full_blocks), tensor(active), symbolic, cache_symbolic }, .{ .shardings = &.{context.sharding} });
    defer full_exe.deinit();
    const warm_exe = try context.platform.compileFn(context.allocator, context.io, kdaPrefill, .{ tensor(warm_input), tensor(warm_blocks), tensor(active), symbolic, cache_symbolic }, .{ .shardings = &.{context.sharding} });
    defer warm_exe.deinit();
    const decode_exe = try context.platform.compileFn(context.allocator, context.io, kdaDecode, .{ tensor(decode_input), tensor(decode_blocks), tensor(active), symbolic, cache_symbolic }, .{ .shardings = &.{context.sharding} });
    defer decode_exe.deinit();
    const compile_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - compile_started, 1000);

    const execute_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var full = try zml.testing.autoCall(context.allocator, context.io, &full_exe, kdaPrefill, .{ full_input, full_blocks, active, weights, zero_cache });
    errdefer zml.Buffer.deinitAll(layer.KdaMoeResult, &full);
    var warm = try zml.testing.autoCall(context.allocator, context.io, &warm_exe, kdaPrefill, .{ warm_input, warm_blocks, active, weights, zero_cache });
    errdefer zml.Buffer.deinitAll(layer.KdaMoeResult, &warm);
    var decode = try zml.testing.autoCall(context.allocator, context.io, &decode_exe, kdaDecode, .{ decode_input, decode_blocks, active, weights, warm.cache });
    errdefer zml.Buffer.deinitAll(layer.KdaMoeResult, &decode);
    const execute_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - execute_started, 1000);

    const full_prefix = try std.fmt.allocPrint(context.allocator, "layer{}", .{layer_index});
    defer context.allocator.free(full_prefix);
    try compareMoeResult(context, context.family_fixture, full_prefix, full);
    const warm_key = try std.fmt.allocPrint(context.allocator, "layer{}.warm.output", .{layer_index});
    defer context.allocator.free(warm_key);
    try context.compare(context.prefix4_fixture, warm_key, warm.output, support.bf16_tolerance);
    const decode_prefix = try std.fmt.allocPrint(context.allocator, "layer{}.decode", .{layer_index});
    defer context.allocator.free(decode_prefix);
    try compareMoeResult(context, context.prefix4_fixture, decode_prefix, decode);
    const cache_in = try std.fmt.allocPrint(context.allocator, "layer{}.decode.cache_in", .{layer_index});
    defer context.allocator.free(cache_in);
    try compareKdaCache(context, context.prefix4_fixture, cache_in, warm.cache);
    const cache_out = try std.fmt.allocPrint(context.allocator, "layer{}.decode.cache_out", .{layer_index});
    defer context.allocator.free(cache_out);
    try compareKdaCache(context, context.prefix4_fixture, cache_out, decode.cache);

    try context.stdout.print(
        "KIMI_K3_PREFIX4_LAYER_PASS layer={} attention=kda experts={} load_us={} compile_us={} execute_us={} routing=global\n",
        .{ layer_index, expert_count, load_us, compile_us, execute_us },
    );
    try context.stdout.flush();
    return .{ .full = full, .warm = warm, .decode = decode };
}

fn runMla(
    context: *Context,
    layer_index: usize,
    full_input: zml.Buffer,
    warm_input: zml.Buffer,
    decode_input: zml.Buffer,
    full_blocks: zml.Buffer,
    warm_blocks: zml.Buffer,
    decode_blocks: zml.Buffer,
) !MlaStreams {
    const load_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var weights = try context.loadMlaWeights(layer_index);
    defer zml.Buffer.deinitAll(layer.MlaMoeWeights, &weights);
    const load_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - load_started, 1000);
    var active = try makeActive(context);
    defer active.deinit();
    const symbolic = mlaTensors(weights);

    const compile_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    const full_exe = try context.platform.compileFn(context.allocator, context.io, mlaPrefill, .{ tensor(full_input), tensor(full_blocks), tensor(active), symbolic }, .{ .shardings = &.{context.sharding} });
    defer full_exe.deinit();
    const warm_exe = try context.platform.compileFn(context.allocator, context.io, mlaPrefill, .{ tensor(warm_input), tensor(warm_blocks), tensor(active), symbolic }, .{ .shardings = &.{context.sharding} });
    defer warm_exe.deinit();
    const warm_cache_shape: mla.LatentCache = .{
        .compressed = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .k = 3, .kv_rank = 512 }, .bf16)),
        .extra_key = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .k = 3, .hd = 64 }, .bf16)),
    };
    const decode_exe = try context.platform.compileFn(context.allocator, context.io, mlaContinue, .{ tensor(decode_input), tensor(decode_blocks), tensor(active), symbolic, warm_cache_shape }, .{ .shardings = &.{context.sharding} });
    defer decode_exe.deinit();
    const decode_cache_shape: mla.LatentCache = .{
        .compressed = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .k = 4, .kv_rank = 512 }, .bf16)),
        .extra_key = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .k = 4, .hd = 64 }, .bf16)),
    };
    const warm_expand_exe = try context.platform.compileFn(context.allocator, context.io, expandLatent, .{ warm_cache_shape, symbolic.attention }, .{ .shardings = &.{context.sharding} });
    defer warm_expand_exe.deinit();
    const decode_expand_exe = try context.platform.compileFn(context.allocator, context.io, expandLatent, .{ decode_cache_shape, symbolic.attention }, .{ .shardings = &.{context.sharding} });
    defer decode_expand_exe.deinit();
    const compile_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - compile_started, 1000);

    const execute_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var full = try zml.testing.autoCall(context.allocator, context.io, &full_exe, mlaPrefill, .{ full_input, full_blocks, active, weights });
    errdefer zml.Buffer.deinitAll(layer.MlaMoeResult, &full);
    var warm = try zml.testing.autoCall(context.allocator, context.io, &warm_exe, mlaPrefill, .{ warm_input, warm_blocks, active, weights });
    errdefer zml.Buffer.deinitAll(layer.MlaMoeResult, &warm);
    var decode = try zml.testing.autoCall(context.allocator, context.io, &decode_exe, mlaContinue, .{ decode_input, decode_blocks, active, weights, warm.cache });
    errdefer zml.Buffer.deinitAll(layer.MlaMoeResult, &decode);
    var warm_expanded = try zml.testing.autoCall(context.allocator, context.io, &warm_expand_exe, expandLatent, .{ warm.cache, weights.attention });
    defer zml.Buffer.deinitAll(ExpandedCache, &warm_expanded);
    var decode_expanded = try zml.testing.autoCall(context.allocator, context.io, &decode_expand_exe, expandLatent, .{ decode.cache, weights.attention });
    defer zml.Buffer.deinitAll(ExpandedCache, &decode_expanded);
    const execute_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - execute_started, 1000);

    const full_prefix = try std.fmt.allocPrint(context.allocator, "layer{}", .{layer_index});
    defer context.allocator.free(full_prefix);
    try compareMoeResult(context, context.family_fixture, full_prefix, full);
    const warm_key = try std.fmt.allocPrint(context.allocator, "layer{}.warm.output", .{layer_index});
    defer context.allocator.free(warm_key);
    try context.compare(context.prefix4_fixture, warm_key, warm.output, support.bf16_tolerance);
    const decode_prefix = try std.fmt.allocPrint(context.allocator, "layer{}.decode", .{layer_index});
    defer context.allocator.free(decode_prefix);
    try compareMoeResult(context, context.prefix4_fixture, decode_prefix, decode);
    const cache_in_key = try std.fmt.allocPrint(context.allocator, "layer{}.decode.cache_in.key", .{layer_index});
    defer context.allocator.free(cache_in_key);
    const cache_in_value = try std.fmt.allocPrint(context.allocator, "layer{}.decode.cache_in.value", .{layer_index});
    defer context.allocator.free(cache_in_value);
    try context.compare(context.prefix4_fixture, cache_in_key, warm_expanded.key, support.bf16_tolerance);
    try context.compare(context.prefix4_fixture, cache_in_value, warm_expanded.value, support.bf16_tolerance);
    const cache_out_key = try std.fmt.allocPrint(context.allocator, "layer{}.decode.cache_out.key", .{layer_index});
    defer context.allocator.free(cache_out_key);
    const cache_out_value = try std.fmt.allocPrint(context.allocator, "layer{}.decode.cache_out.value", .{layer_index});
    defer context.allocator.free(cache_out_value);
    try context.compare(context.prefix4_fixture, cache_out_key, decode_expanded.key, support.bf16_tolerance);
    try context.compare(context.prefix4_fixture, cache_out_value, decode_expanded.value, support.bf16_tolerance);

    try context.stdout.print(
        "KIMI_K3_PREFIX4_LAYER_PASS layer={} attention=mla experts={} load_us={} compile_us={} execute_us={} routing=global\n",
        .{ layer_index, expert_count, load_us, compile_us, execute_us },
    );
    try context.stdout.flush();
    return .{ .full = full, .warm = warm, .decode = decode };
}

const HeadWeights = struct {
    embedding: zml.Buffer,
    output_res_norm: zml.Buffer,
    output_res_projection: zml.Buffer,
    final_norm: zml.Buffer,
    lm_head: zml.Buffer,
};

fn loadHeadWeights(context: *Context) !HeadWeights {
    return .{
        .embedding = try support.loadBuffer(context.allocator, context.io, context.platform, context.weights, "language_model.model.embed_tokens.weight", .{ .voc, .d }, context.sharding),
        .output_res_norm = try support.loadBuffer(context.allocator, context.io, context.platform, context.weights, "language_model.model.output_attn_res_norm.weight", .{.d}, context.sharding),
        .output_res_projection = try support.loadBuffer(context.allocator, context.io, context.platform, context.weights, "language_model.model.output_attn_res_proj.weight", .{ .one, .d }, context.sharding),
        .final_norm = try support.loadBuffer(context.allocator, context.io, context.platform, context.weights, "language_model.model.norm.weight", .{.d}, context.sharding),
        .lm_head = try support.loadBuffer(context.allocator, context.io, context.platform, context.weights, "language_model.lm_head.weight", .{ .voc, .d }, context.sharding),
    };
}

fn deinitHeadWeights(weights: *HeadWeights) void {
    weights.embedding.deinit();
    weights.output_res_norm.deinit();
    weights.output_res_projection.deinit();
    weights.final_norm.deinit();
    weights.lm_head.deinit();
}

fn headTensors(weights: HeadWeights) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
    return .{ tensor(weights.output_res_norm), tensor(weights.output_res_projection), tensor(weights.final_norm), tensor(weights.lm_head) };
}

fn compareHead(context: *Context, prefix: []const u8, actual: zml.Bufferized(layer.DiagnosticHeadResult)) !void {
    const names = [_][]const u8{ "output_attn_res.candidates", "output_attn_res.weights", "output_attn_res.out", "final_norm.out", "logits", "greedy_token" };
    const values = .{ actual.output_candidates, actual.output_selector_weights, actual.output_selected, actual.final_norm, actual.logits, actual.greedy_token };
    inline for (names, values, 0..) |name, value, index| {
        const key = try std.fmt.allocPrint(context.allocator, "{s}.{s}", .{ prefix, name });
        defer context.allocator.free(key);
        const opts: zml.testing.CompareOpts = if (index == 1) route_weight_tolerance else if (index == 5) .{} else support.bf16_tolerance;
        try context.compare(context.prefix4_fixture, key, value, opts);
    }
}

fn checkChainedGreedyTie(context: *Context, prefix: []const u8, actual: zml.Bufferized(layer.DiagnosticHeadResult)) !void {
    const logits_key = try std.fmt.allocPrint(context.allocator, "{s}.logits", .{prefix});
    defer context.allocator.free(logits_key);
    var expected_logits = try support.loadBuffer(context.allocator, context.io, context.platform, context.prefix4_fixture, logits_key, actual.logits.shape().tags(), context.sharding);
    defer expected_logits.deinit();
    const token_key = try std.fmt.allocPrint(context.allocator, "{s}.greedy_token", .{prefix});
    defer context.allocator.free(token_key);
    var expected_token_buffer = try support.loadBuffer(context.allocator, context.io, context.platform, context.prefix4_fixture, token_key, actual.greedy_token.shape().tags(), context.sharding);
    defer expected_token_buffer.deinit();
    var actual_host = try actual.logits.toSliceAlloc(context.allocator, context.io);
    defer actual_host.free(context.allocator);
    var expected_host = try expected_logits.toSliceAlloc(context.allocator, context.io);
    defer expected_host.free(context.allocator);
    const actual_values = actual_host.items(zml.floats.BFloat16);
    const expected_values = expected_host.items(zml.floats.BFloat16);
    const vocab: usize = @intCast(actual.logits.shape().dim(.voc));
    const offset = actual_values.len - vocab;
    var expected_max = -std.math.inf(f32);
    var actual_max = -std.math.inf(f32);
    for (expected_values[offset..]) |value| expected_max = @max(expected_max, value.toF32());
    for (actual_values[offset..]) |value| actual_max = @max(actual_max, value.toF32());
    const actual_token = try actual.greedy_token.getValue(i64, context.io);
    const expected_token = try expected_token_buffer.getValue(i64, context.io);
    if (actual_token < 0 or actual_token >= vocab) return error.InvalidPrefix4GreedyToken;
    const actual_index: usize = @intCast(actual_token);
    const expected_index: usize = @intCast(expected_token);
    const actual_value = actual_values[offset + actual_index].toF32();
    const expected_value_at_actual = expected_values[offset + actual_index].toF32();
    const actual_value_at_expected = actual_values[offset + expected_index].toF32();
    try context.stdout.print(
        "KIMI_K3_PREFIX4_GREEDY_TIE prefix={s} actual_token={} official_token={} actual_logit={d:.6} official_logit_at_actual={d:.6} actual_logit_at_official={d:.6} actual_max={d:.6} official_max={d:.6}\n",
        .{ prefix, actual_token, expected_token, actual_value, expected_value_at_actual, actual_value_at_expected, actual_max, expected_max },
    );
    try context.stdout.flush();
    if (expected_value_at_actual != expected_max and actual_value_at_expected != actual_max) return error.Prefix4GreedyTieSetsDisjoint;
}

fn compareChainedHead(context: *Context, prefix: []const u8, actual: zml.Bufferized(layer.DiagnosticHeadResult)) !void {
    const names = [_][]const u8{ "output_attn_res.candidates", "output_attn_res.weights", "output_attn_res.out", "final_norm.out", "logits", "greedy_token" };
    const values = .{ actual.output_candidates, actual.output_selector_weights, actual.output_selected, actual.final_norm, actual.logits, actual.greedy_token };
    inline for (names, values, 0..) |name, value, index| {
        if (index == 5) continue;
        const key = try std.fmt.allocPrint(context.allocator, "{s}.{s}", .{ prefix, name });
        defer context.allocator.free(key);
        const opts: zml.testing.CompareOpts = switch (index) {
            1 => route_weight_tolerance,
            3 => .{ .absolute_tolerance = 2.5e-1, .relative_tolerance = 1e-1, .minimum_close_fraction = 0.99 },
            4 => .{ .absolute_tolerance = 5e-1, .relative_tolerance = 1e-1, .minimum_close_fraction = 0.95 },
            5 => .{},
            else => support.bf16_tolerance,
        };
        try context.compare(context.prefix4_fixture, key, value, opts);
    }
    try checkChainedGreedyTie(context, prefix, actual);
}

fn runReferenceHead(context: *Context, weights: HeadWeights) !void {
    var full_hidden = try support.loadBuffer(context.allocator, context.io, context.platform, context.family_fixture, "layer3.output", .{ .b, .s, .d }, context.sharding);
    defer full_hidden.deinit();
    var full_blocks = try support.loadBuffer(context.allocator, context.io, context.platform, context.layer0_fixture, "prefix.layer0.block_residual.out", .{ .token, .source, .d }, context.sharding);
    defer full_blocks.deinit();
    var decode_hidden = try support.loadBuffer(context.allocator, context.io, context.platform, context.prefix4_fixture, "layer3.decode.output", .{ .b, .s, .d }, context.sharding);
    defer decode_hidden.deinit();
    var decode_blocks = try support.loadBuffer(context.allocator, context.io, context.platform, context.prefix4_fixture, "prefix.layer0.decode.block_residual", .{ .token, .source, .d }, context.sharding);
    defer decode_blocks.deinit();
    const symbolic = headTensors(weights);
    const full_exe = try context.platform.compileFn(context.allocator, context.io, layer.diagnosticHead, .{ tensor(full_hidden), tensor(full_blocks), symbolic[0], symbolic[1], symbolic[2], symbolic[3] }, .{ .shardings = &.{context.sharding} });
    defer full_exe.deinit();
    const decode_exe = try context.platform.compileFn(context.allocator, context.io, layer.diagnosticHead, .{ tensor(decode_hidden), tensor(decode_blocks), symbolic[0], symbolic[1], symbolic[2], symbolic[3] }, .{ .shardings = &.{context.sharding} });
    defer decode_exe.deinit();
    var full = try zml.testing.autoCall(context.allocator, context.io, &full_exe, layer.diagnosticHead, .{ full_hidden, full_blocks, weights.output_res_norm, weights.output_res_projection, weights.final_norm, weights.lm_head });
    defer zml.Buffer.deinitAll(layer.DiagnosticHeadResult, &full);
    var decode = try zml.testing.autoCall(context.allocator, context.io, &decode_exe, layer.diagnosticHead, .{ decode_hidden, decode_blocks, weights.output_res_norm, weights.output_res_projection, weights.final_norm, weights.lm_head });
    defer zml.Buffer.deinitAll(layer.DiagnosticHeadResult, &decode);
    try compareHead(context, "prefix", full);
    try compareHead(context, "decode", decode);
    try context.stdout.writeAll("KIMI_K3_PREFIX4_REFERENCE_HEAD_PASS inputs=official tolerance=strict\n");
    try context.stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.95 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;

    var profiler: ?zml.Platform.Profiler = null;
    defer if (profiler) |*active_profiler| {
        _ = active_profiler.stop() catch {};
        active_profiler.deinit();
    };
    if (args.profile) {
        profiler = try platform.profiler(allocator, io, .{
            .repository_path = args.profile_repository,
            .session_id = args.profile_session,
        });
        try profiler.?.start();
    }

    var weight_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.weights);
    defer weight_registry.deinit();
    var weight_store: zml.io.TensorStore = .fromRegistry(allocator, &weight_registry);
    defer weight_store.deinit();
    var layer0_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.layer0_fixture);
    defer layer0_registry.deinit();
    var layer0_store: zml.io.TensorStore = .fromRegistry(allocator, &layer0_registry);
    defer layer0_store.deinit();
    var family_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.layer_family_fixture);
    defer family_registry.deinit();
    var family_store: zml.io.TensorStore = .fromRegistry(allocator, &family_registry);
    defer family_store.deinit();
    var prefix4_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.prefix4_fixture);
    defer prefix4_registry.deinit();
    var prefix4_store: zml.io.TensorStore = .fromRegistry(allocator, &prefix4_registry);
    defer prefix4_store.deinit();
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    var context: Context = .{
        .allocator = allocator, .io = io, .platform = platform,
        .weights = weight_store.view(), .layer0_fixture = layer0_store.view(),
        .family_fixture = family_store.view(), .prefix4_fixture = prefix4_store.view(),
        .sharding = platform.replicated_sharding, .stdout = &stdout_file.interface,
    };

    const head_load_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var head_weights = try loadHeadWeights(&context);
    defer deinitHeadWeights(&head_weights);
    const head_load_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - head_load_started, 1000);
    try runReferenceHead(&context, head_weights);
    if (args.head_only) return;
    var tokens = try support.loadBuffer(allocator, io, platform, context.prefix4_fixture, "prefix.token_ids", .{ .b, .s }, context.sharding);
    defer tokens.deinit();
    const embed_exe = try platform.compileFn(allocator, io, embed, .{ tensor(tokens), tensor(head_weights.embedding) }, .{ .shardings = &.{context.sharding} });
    defer embed_exe.deinit();
    var embedding = try zml.testing.autoCall(allocator, io, &embed_exe, embed, .{ tokens, head_weights.embedding });
    defer embedding.deinit();
    try context.compare(context.prefix4_fixture, "prefix.embedding.out", embedding, support.bf16_tolerance);
    const split_exe = try platform.compileFn(allocator, io, splitEmbedding, .{tensor(embedding)}, .{ .shardings = &.{context.sharding} });
    defer split_exe.deinit();
    var embedding_split = try zml.testing.autoCall(allocator, io, &split_exe, splitEmbedding, .{embedding});
    defer zml.Buffer.deinitAll(EmbeddingSplit, &embedding_split);

    const layer0_weights = layer.Layer0Weights.init(context.weights);
    var layer0_buffers = try zml.mem.bufferize(allocator, layer.Layer0Weights, &layer0_weights);
    defer zml.Buffer.deinitAll(layer.Layer0Weights, &layer0_buffers);
    var loader: zml.io.Loader = try .init(allocator, platform, .{ .parallelism = 1, .dma_chunks = 2, .dma_chunk_size = 256 * zml.MiB });
    defer loader.deinit();
    const layer0_load_started = std.Io.Clock.now(.real, io).toNanoseconds();
    loader.load(io, layer.Layer0Weights, &layer0_weights, &layer0_buffers, &weight_store, &.{context.sharding}, .{});
    try loader.await(io);
    const layer0_load_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - layer0_load_started, 1000);
    var zero_cache = try context.zeroKdaCache();
    defer zml.Buffer.deinitAll(kda.Cache, &zero_cache);
    const cache_symbolic = kdaCacheTensors(zero_cache);
    const layer0_compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    const full0_exe = try platform.compileFn(allocator, io, layer.forwardLayer0, .{ tensor(embedding), layer0_weights, cache_symbolic }, .{ .shardings = &.{context.sharding} });
    defer full0_exe.deinit();
    const warm0_exe = try platform.compileFn(allocator, io, layer.forwardLayer0, .{ tensor(embedding_split.warm), layer0_weights, cache_symbolic }, .{ .shardings = &.{context.sharding} });
    defer warm0_exe.deinit();
    const decode0_exe = try platform.compileFn(allocator, io, layer.forwardLayer0, .{ tensor(embedding_split.decode), layer0_weights, cache_symbolic }, .{ .shardings = &.{context.sharding} });
    defer decode0_exe.deinit();
    const layer0_compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - layer0_compile_started, 1000);
    const layer0_execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var full0 = try zml.testing.autoCall(allocator, io, &full0_exe, layer.forwardLayer0, .{ embedding, layer0_buffers, zero_cache });
    defer zml.Buffer.deinitAll(layer.Layer0Result, &full0);
    var warm0 = try zml.testing.autoCall(allocator, io, &warm0_exe, layer.forwardLayer0, .{ embedding_split.warm, layer0_buffers, zero_cache });
    defer zml.Buffer.deinitAll(layer.Layer0Result, &warm0);
    var decode0 = try zml.testing.autoCall(allocator, io, &decode0_exe, layer.forwardLayer0, .{ embedding_split.decode, layer0_buffers, warm0.cache });
    defer zml.Buffer.deinitAll(layer.Layer0Result, &decode0);
    const layer0_execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - layer0_execute_started, 1000);
    try context.compare(context.layer0_fixture, "prefix.layer0.out", full0.output, support.bf16_tolerance);
    try context.compare(context.layer0_fixture, "prefix.layer0.block_residual.out", full0.block_residual, support.bf16_tolerance);
    try context.compare(context.prefix4_fixture, "prefix.layer0.warm.output", warm0.output, support.bf16_tolerance);
    try context.compare(context.prefix4_fixture, "prefix.layer0.decode.output", decode0.output, support.bf16_tolerance);
    try context.compare(context.prefix4_fixture, "prefix.layer0.decode.block_residual", decode0.block_residual, support.bf16_tolerance);
    try compareKdaCache(&context, context.prefix4_fixture, "decode.layer0.cache_in.cache", warm0.cache);
    try compareKdaCache(&context, context.prefix4_fixture, "decode.layer0.cache_out.cache", decode0.cache);
    try stdout_file.interface.print(
        "KIMI_K3_PREFIX4_LAYER_PASS layer=0 attention=kda_dense load_us={} compile_us={} execute_us={}\n",
        .{ layer0_load_us, layer0_compile_us, layer0_execute_us },
    );
    try stdout_file.interface.flush();

    var layer1_streams = try runKda(&context, 1, full0.output, warm0.output, decode0.output, full0.block_residual, warm0.block_residual, decode0.block_residual);
    defer deinitKdaStreams(&layer1_streams);
    var layer2_streams = try runKda(&context, 2, layer1_streams.full.output, layer1_streams.warm.output, layer1_streams.decode.output, full0.block_residual, warm0.block_residual, decode0.block_residual);
    defer deinitKdaStreams(&layer2_streams);
    var layer3_streams = try runMla(&context, 3, layer2_streams.full.output, layer2_streams.warm.output, layer2_streams.decode.output, full0.block_residual, warm0.block_residual, decode0.block_residual);
    defer deinitMlaStreams(&layer3_streams);

    const head_symbolic = headTensors(head_weights);
    const head_compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    const full_head_exe = try platform.compileFn(allocator, io, layer.diagnosticHead, .{ tensor(layer3_streams.full.output), tensor(full0.block_residual), head_symbolic[0], head_symbolic[1], head_symbolic[2], head_symbolic[3] }, .{ .shardings = &.{context.sharding} });
    defer full_head_exe.deinit();
    const decode_head_exe = try platform.compileFn(allocator, io, layer.diagnosticHead, .{ tensor(layer3_streams.decode.output), tensor(decode0.block_residual), head_symbolic[0], head_symbolic[1], head_symbolic[2], head_symbolic[3] }, .{ .shardings = &.{context.sharding} });
    defer decode_head_exe.deinit();
    const head_compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - head_compile_started, 1000);
    const head_execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var full_head = try zml.testing.autoCall(allocator, io, &full_head_exe, layer.diagnosticHead, .{ layer3_streams.full.output, full0.block_residual, head_weights.output_res_norm, head_weights.output_res_projection, head_weights.final_norm, head_weights.lm_head });
    defer zml.Buffer.deinitAll(layer.DiagnosticHeadResult, &full_head);
    var decode_head = try zml.testing.autoCall(allocator, io, &decode_head_exe, layer.diagnosticHead, .{ layer3_streams.decode.output, decode0.block_residual, head_weights.output_res_norm, head_weights.output_res_projection, head_weights.final_norm, head_weights.lm_head });
    defer zml.Buffer.deinitAll(layer.DiagnosticHeadResult, &decode_head);
    const head_execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - head_execute_started, 1000);
    try compareChainedHead(&context, "prefix", full_head);
    try compareChainedHead(&context, "decode", decode_head);
    try stdout_file.interface.print(
        "KIMI_K3_PREFIX4_HEAD_PASS load_us={} compile_us={} execute_us={}\n",
        .{ head_load_us, head_compile_us, head_execute_us },
    );
    try stdout_file.interface.writeAll("KIMI_K3_PREFIX4_ALL_PASS layers=0,1,2,3 experts_per_moe=896 prefill=4 warm=3 decode=1 backend=cuda routing=global\n");
    try stdout_file.interface.flush();
}
