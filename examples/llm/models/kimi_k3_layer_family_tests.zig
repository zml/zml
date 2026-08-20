const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");
const layer = @import("kimi_k3/layer.zig");
const mla = @import("kimi_k3/mla.zig");
const router = @import("kimi_k3/router.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(800_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_layer_family_tests --fixture=<layer-family-reference.safetensors>
        \\
        \\Run real-weight KDA+MoE and MLA+MoE prefill/decode parity on NVIDIA CUDA.
        \\
    ;
};

// CUDA GEMM reduction order changes with the token batch shape. The selected
// expert set remains exact; this bound covers the measured FP32 weight drift
// while still requiring every aligned route weight to match.
const route_weight_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 2e-3,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 1.0,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,

    fn key(self: *Context, layer_index: usize, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "layer{}.{s}", .{ layer_index, suffix });
    }

    fn load(self: *Context, layer_index: usize, suffix: []const u8, tags: anytype) !zml.Buffer {
        const tensor_key = try self.key(layer_index, suffix);
        defer self.allocator.free(tensor_key);
        return support.loadBuffer(
            self.allocator,
            self.io,
            self.platform,
            self.store,
            tensor_key,
            tags,
            self.sharding,
        );
    }

    fn loadAs(self: *Context, layer_index: usize, suffix: []const u8, target: zml.Shape) !zml.Buffer {
        const tensor_key = try self.key(layer_index, suffix);
        defer self.allocator.free(tensor_key);
        const source = self.store.getShape(tensor_key) orelse return error.MissingLayerFamilyFixture;
        if (source.byteSize() != target.byteSize()) return error.LayerFamilyReshapeSizeMismatch;
        const bytes = try self.allocator.alloc(u8, source.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(tensor_key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, target, self.sharding, bytes);
    }

    fn compare(
        self: *Context,
        layer_index: usize,
        mode: []const u8,
        suffix: []const u8,
        actual: zml.Buffer,
        opts: zml.testing.CompareOpts,
    ) !void {
        const fixture_suffix = if (mode.len == 0)
            try self.allocator.dupe(u8, suffix)
        else
            try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ mode, suffix });
        defer self.allocator.free(fixture_suffix);
        const tensor_key = try self.key(layer_index, fixture_suffix);
        defer self.allocator.free(tensor_key);
        try support.compare(
            self.allocator,
            self.io,
            self.platform,
            self.store,
            tensor_key,
            actual,
            opts,
            self.sharding,
        );
    }

    fn readI64(self: *Context, layer_index: usize, suffix: []const u8) ![]i64 {
        const tensor_key = try self.key(layer_index, suffix);
        defer self.allocator.free(tensor_key);
        const shape = self.store.getShape(tensor_key) orelse return error.MissingLayerFamilyRouteMap;
        if (shape.dtype() != .i64) return error.InvalidLayerFamilyRouteMap;
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(tensor_key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        const values = try self.allocator.alloc(i64, @divExact(bytes.len, @sizeOf(i64)));
        for (values, 0..) |*value, i| {
            value.* = std.mem.readInt(i64, bytes[i * 8 ..][0..8], .little);
        }
        return values;
    }

    // KIMI_K3_TEMP_REMOVE_M20: reconstructing the compact global-to-local
    // expert map from fixture routes is isolated bring-up code removed in M20.
    fn selectedIds(self: *Context, layer_index: usize, count: usize) !zml.Buffer {
        const global = try self.readI64(layer_index, "route.global_ids");
        defer self.allocator.free(global);
        const local = try self.readI64(layer_index, "route.local_ids");
        defer self.allocator.free(local);
        if (global.len != local.len) return error.InvalidLayerFamilyRouteMap;
        const selected = try self.allocator.alloc(i64, count);
        defer self.allocator.free(selected);
        @memset(selected, -1);
        for (global, local) |global_id, local_id| {
            if (local_id < 0 or local_id >= @as(i64, @intCast(count))) return error.InvalidLayerFamilyRouteMap;
            const slot: usize = @intCast(local_id);
            if (selected[slot] != -1 and selected[slot] != global_id) {
                return error.InconsistentLayerFamilyRouteMap;
            }
            selected[slot] = global_id;
        }
        for (selected) |global_id| {
            if (global_id < 0) return error.IncompleteLayerFamilyRouteMap;
        }
        return zml.Buffer.fromBytes(
            self.io,
            self.platform,
            zml.Shape.init(.{ .selected = count }, .i64),
            self.sharding,
            std.mem.sliceAsBytes(selected),
        );
    }

    fn active(self: *Context) !zml.Buffer {
        const yes = [_]u8{1};
        return zml.Buffer.fromBytes(
            self.io,
            self.platform,
            zml.Shape.init(.{ .source = 1 }, .bool),
            self.sharding,
            &yes,
        );
    }

    fn loadCommon(self: *Context, layer_index: usize) !zml.Bufferized(layer.MoeLayerWeights) {
        return .{
            .attention_res_norm = try self.load(layer_index, "weights.layer.self_attention_res_norm.weight", .{.d}),
            .attention_res_projection = try self.load(layer_index, "weights.layer.self_attention_res_proj.weight", .{ .one, .d }),
            .input_norm = try self.load(layer_index, "weights.layer.input_layernorm.weight", .{.d}),
            .mlp_res_norm = try self.load(layer_index, "weights.layer.mlp_res_norm.weight", .{.d}),
            .mlp_res_projection = try self.load(layer_index, "weights.layer.mlp_res_proj.weight", .{ .one, .d }),
            .post_attention_norm = try self.load(layer_index, "weights.layer.post_attention_layernorm.weight", .{.d}),
            .moe = .{
                .gate = .{
                    .weight = try self.load(layer_index, "weights.layer.block_sparse_moe.gate.weight", .{ .expert, .d }),
                    .correction_bias = try self.load(layer_index, "weights.layer.block_sparse_moe.gate.e_score_correction_bias", .{.expert}),
                },
                .experts = .{
                    .w1 = .{
                        .values = try self.load(layer_index, "weights.selected.w1.packed", .{ .expert, .intermediate, .kw }),
                        .scale = try self.load(layer_index, "weights.selected.w1.scale", .{ .expert, .intermediate, .block }),
                    },
                    .w2 = .{
                        .values = try self.load(layer_index, "weights.selected.w2.packed", .{ .expert, .latent, .kw }),
                        .scale = try self.load(layer_index, "weights.selected.w2.scale", .{ .expert, .latent, .block }),
                    },
                    .w3 = .{
                        .values = try self.load(layer_index, "weights.selected.w3.packed", .{ .expert, .intermediate, .kw }),
                        .scale = try self.load(layer_index, "weights.selected.w3.scale", .{ .expert, .intermediate, .block }),
                    },
                },
                .dense = .{
                    .routed_down = try self.load(layer_index, "weights.dense.routed_down", .{ .latent, .d }),
                    .routed_norm = try self.load(layer_index, "weights.dense.routed_norm", .{.latent}),
                    .routed_up = try self.load(layer_index, "weights.dense.routed_up", .{ .d, .latent }),
                    .shared_gate = try self.load(layer_index, "weights.dense.shared_gate", .{ .intermediate, .d }),
                    .shared_up = try self.load(layer_index, "weights.dense.shared_up", .{ .intermediate, .d }),
                    .shared_down = try self.load(layer_index, "weights.dense.shared_down", .{ .d, .intermediate }),
                },
            },
        };
    }

    fn loadKdaWeights(self: *Context, layer_index: usize) !zml.Bufferized(layer.KdaMoeWeights) {
        return .{
            .common = try self.loadCommon(layer_index),
            .attention = .{
                .q_weight = try self.load(layer_index, "weights.layer.self_attn.q_proj.weight", .{ .out, .d }),
                .k_weight = try self.load(layer_index, "weights.layer.self_attn.k_proj.weight", .{ .out, .d }),
                .v_weight = try self.load(layer_index, "weights.layer.self_attn.v_proj.weight", .{ .out, .d }),
                .q_conv_weight = try self.loadAs(layer_index, "weights.layer.self_attn.q_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32)),
                .k_conv_weight = try self.loadAs(layer_index, "weights.layer.self_attn.k_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32)),
                .v_conv_weight = try self.loadAs(layer_index, "weights.layer.self_attn.v_conv1d.weight", zml.Shape.init(.{ .channel = 12288, .kernel = 4 }, .f32)),
                .decay_a_weight = try self.load(layer_index, "weights.layer.self_attn.f_a_proj.weight", .{ .out, .d }),
                .decay_b_weight = try self.load(layer_index, "weights.layer.self_attn.f_b_proj.weight", .{ .channel, .rank }),
                .a_log = try self.load(layer_index, "weights.layer.self_attn.A_log", .{.h}),
                .dt_bias = try self.loadAs(layer_index, "weights.layer.self_attn.dt_bias", zml.Shape.init(.{ .h = 96, .k = 128 }, .f32)),
                .beta_weight = try self.load(layer_index, "weights.layer.self_attn.b_proj.weight", .{ .out, .d }),
                .gate_weight = try self.load(layer_index, "weights.layer.self_attn.g_proj.weight", .{ .out, .d }),
                .norm_weight = try self.load(layer_index, "weights.layer.self_attn.o_norm.weight", .{.v}),
                .output_weight = try self.load(layer_index, "weights.layer.self_attn.o_proj.weight", .{ .d, .out }),
            },
        };
    }

    fn loadMlaWeights(self: *Context, layer_index: usize) !zml.Bufferized(layer.MlaMoeWeights) {
        return .{
            .common = try self.loadCommon(layer_index),
            .attention = .{
                .q_a_proj = try self.load(layer_index, "weights.layer.self_attn.q_a_proj.weight", .{ .rank, .d }),
                .q_a_norm = try self.load(layer_index, "weights.layer.self_attn.q_a_layernorm.weight", .{.rank}),
                .q_b_proj = try self.load(layer_index, "weights.layer.self_attn.q_b_proj.weight", .{ .mix, .rank }),
                .kv_a_proj = try self.load(layer_index, "weights.layer.self_attn.kv_a_proj_with_mqa.weight", .{ .kv_mix, .d }),
                .kv_a_norm = try self.load(layer_index, "weights.layer.self_attn.kv_a_layernorm.weight", .{.kv_rank}),
                .kv_b_proj = try self.load(layer_index, "weights.layer.self_attn.kv_b_proj.weight", .{ .kv_mix, .kv_rank }),
                .gate_proj = try self.load(layer_index, "weights.layer.self_attn.g_proj.weight", .{ .out, .d }),
                .output_proj = try self.load(layer_index, "weights.layer.self_attn.o_proj.weight", .{ .d, .out }),
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

    fn loadKdaCache(self: *Context, layer_index: usize, mode: []const u8) !zml.Bufferized(kda.Cache) {
        const q_suffix = try std.fmt.allocPrint(self.allocator, "{s}.conv0", .{mode});
        defer self.allocator.free(q_suffix);
        const k_suffix = try std.fmt.allocPrint(self.allocator, "{s}.conv1", .{mode});
        defer self.allocator.free(k_suffix);
        const v_suffix = try std.fmt.allocPrint(self.allocator, "{s}.conv2", .{mode});
        defer self.allocator.free(v_suffix);
        const recurrent_suffix = try std.fmt.allocPrint(
            self.allocator,
            "{s}.recurrent",
            .{mode},
        );
        defer self.allocator.free(recurrent_suffix);
        return .{
            .q_conv = try self.load(layer_index, q_suffix, .{ .b, .channel, .kernel }),
            .k_conv = try self.load(layer_index, k_suffix, .{ .b, .channel, .kernel }),
            .v_conv = try self.load(layer_index, v_suffix, .{ .b, .channel, .kernel }),
            .recurrent_state = try self.load(layer_index, recurrent_suffix, .{ .b, .h, .v, .k }),
        };
    }

    fn compareCommon(self: *Context, layer_index: usize, mode: []const u8, result: anytype, global: zml.Bufferized(GlobalAlignment), local: zml.Bufferized(GlobalAlignment), aligned_route_outputs: zml.Buffer) !void {
        const boundaries = [_][]const u8{
            "selected_input", "input_norm", "attention_output", "prefix_after_attention",
            "selected_mlp", "moe_input", "moe.routed_down", "moe.route_outputs",
            "moe.combined_latent", "moe.routed_norm", "moe.routed_up",
            "moe.shared_output", "moe.output", "output",
        };
        const values = .{
            result.selected_input, result.input_norm, result.attention_output, result.prefix_after_attention,
            result.selected_mlp, result.moe_input, result.moe_result.routed_down, aligned_route_outputs,
            result.moe_result.combined_latent, result.moe_result.routed_norm, result.moe_result.routed_up,
            result.moe_result.shared_output, result.moe_result.output, result.output,
        };
        inline for (boundaries, values) |boundary, value| {
            // KIMI_K3_TEMP_REMOVE_M20: boundary progress markers identify the
            // first divergent composed activation and are removed in cleanup.
            try self.stdout.print("KIMI_K3_LAYER_FAMILY_CHECK layer={} mode={s} boundary={s}\n", .{
                layer_index, if (mode.len == 0) "prefill" else mode, boundary,
            });
            try self.stdout.flush();
            try self.compare(layer_index, mode, boundary, value, support.bf16_tolerance);
        }
        // KIMI_K3_TEMP_REMOVE_M20: route progress markers disambiguate the
        // independent full router from the compact composed-layer router.
        try self.stdout.print("KIMI_K3_LAYER_FAMILY_CHECK layer={} mode={s} boundary=route.global_ids\n", .{ layer_index, if (mode.len == 0) "prefill" else mode });
        try self.stdout.flush();
        try self.compare(layer_index, mode, "route.global_ids", global.matched_ids, .{});
        try self.stdout.print("KIMI_K3_LAYER_FAMILY_CHECK layer={} mode={s} boundary=route.local_ids\n", .{ layer_index, if (mode.len == 0) "prefill" else mode });
        try self.stdout.flush();
        try self.compare(layer_index, mode, "route.local_ids", local.matched_ids, .{});
        try self.stdout.print("KIMI_K3_LAYER_FAMILY_CHECK layer={} mode={s} boundary=route.global_weights\n", .{ layer_index, if (mode.len == 0) "prefill" else mode });
        try self.stdout.flush();
        try self.compare(layer_index, mode, "route.weights", global.aligned_weights, route_weight_tolerance);
        try self.stdout.print("KIMI_K3_LAYER_FAMILY_CHECK layer={} mode={s} boundary=route.local_weights\n", .{ layer_index, if (mode.len == 0) "prefill" else mode });
        try self.stdout.flush();
        try self.compare(layer_index, mode, "route.weights", local.aligned_weights, route_weight_tolerance);
    }

    fn compareKdaCache(self: *Context, layer_index: usize, mode: []const u8, cache: zml.Bufferized(kda.Cache)) !void {
        const conv0 = if (mode.len == 0) "cache.conv0" else "conv0";
        const conv1 = if (mode.len == 0) "cache.conv1" else "conv1";
        const conv2 = if (mode.len == 0) "cache.conv2" else "conv2";
        const recurrent = if (mode.len == 0) "cache.recurrent" else "recurrent";
        try self.compare(layer_index, mode, conv0, cache.q_conv, support.bf16_tolerance);
        try self.compare(layer_index, mode, conv1, cache.k_conv, support.bf16_tolerance);
        try self.compare(layer_index, mode, conv2, cache.v_conv, support.bf16_tolerance);
        try self.compare(layer_index, mode, recurrent, cache.recurrent_state, support.state_tolerance);
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
            .gate = .{
                .weight = tensor(weights.moe.gate.weight),
                .correction_bias = tensor(weights.moe.gate.correction_bias),
            },
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
            .q_weight = tensor(weights.attention.q_weight),
            .k_weight = tensor(weights.attention.k_weight),
            .v_weight = tensor(weights.attention.v_weight),
            .q_conv_weight = tensor(weights.attention.q_conv_weight),
            .k_conv_weight = tensor(weights.attention.k_conv_weight),
            .v_conv_weight = tensor(weights.attention.v_conv_weight),
            .decay_a_weight = tensor(weights.attention.decay_a_weight),
            .decay_b_weight = tensor(weights.attention.decay_b_weight),
            .a_log = tensor(weights.attention.a_log),
            .dt_bias = tensor(weights.attention.dt_bias),
            .beta_weight = tensor(weights.attention.beta_weight),
            .gate_weight = tensor(weights.attention.gate_weight),
            .norm_weight = tensor(weights.attention.norm_weight),
            .output_weight = tensor(weights.attention.output_weight),
        },
    };
}

fn mlaTensors(weights: zml.Bufferized(layer.MlaMoeWeights)) layer.MlaMoeWeights {
    return .{
        .common = commonTensors(weights.common),
        .attention = .{
            .q_a_proj = tensor(weights.attention.q_a_proj),
            .q_a_norm = tensor(weights.attention.q_a_norm),
            .q_b_proj = tensor(weights.attention.q_b_proj),
            .kv_a_proj = tensor(weights.attention.kv_a_proj),
            .kv_a_norm = tensor(weights.attention.kv_a_norm),
            .kv_b_proj = tensor(weights.attention.kv_b_proj),
            .gate_proj = tensor(weights.attention.gate_proj),
            .output_proj = tensor(weights.attention.output_proj),
        },
    };
}

fn kdaCacheTensors(cache: zml.Bufferized(kda.Cache)) kda.Cache {
    return .{
        .q_conv = tensor(cache.q_conv),
        .k_conv = tensor(cache.k_conv),
        .v_conv = tensor(cache.v_conv),
        .recurrent_state = tensor(cache.recurrent_state),
    };
}

fn mlaCacheTensors(cache: zml.Bufferized(mla.LatentCache)) mla.LatentCache {
    return .{
        .compressed = tensor(cache.compressed),
        .extra_key = tensor(cache.extra_key),
    };
}

// KIMI_K3_TEMP_REMOVE_M20: the local router is created only for the compact
// selected-expert fixture. The independent global router remains authoritative.
fn localize(weights: layer.MoeLayerWeights, selected: zml.Tensor) layer.MoeLayerWeights {
    var result = weights;
    const indices = selected.convert(.i32);
    result.moe.gate = .{
        .weight = weights.moe.gate.weight.gather(.{ .expert = indices }, .{}).rename(.{ .selected = .expert }),
        .correction_bias = weights.moe.gate.correction_bias.gather(.{ .expert = indices }, .{}).rename(.{ .selected = .expert }),
    };
    return result;
}

const GlobalAlignment = struct {
    matched_ids: zml.Tensor,
    aligned_weights: zml.Tensor,
};

// KIMI_K3_TEMP_REMOVE_M20: Top-K ordering is implementation-defined. This
// diagnostic requires the exact global expert set and aligns weights by ID.
fn alignGlobal(route: router.Result, expected_ids: zml.Tensor) GlobalAlignment {
    const actual_ids = route.topk_ids.rename(.{ .route = .actual_route });
    const actual_weights = route.topk_weights.rename(.{ .route = .actual_route });
    const match_shape = zml.Shape.init(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route),
    }, .i64);
    const expected_grid = expected_ids.reshape(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = 1,
    }).broad(match_shape);
    const actual_grid = actual_ids.reshape(.{
        .b = actual_ids.dim(.b),
        .s = actual_ids.dim(.s),
        .expected_route = 1,
        .actual_route = actual_ids.dim(.actual_route),
    }).broad(match_shape);
    const matches = actual_grid.cmp(.EQ, expected_grid);
    const found = matches.convert(.i32).sum(.actual_route).squeeze(.actual_route)
        .cmp(.GT, zml.Tensor.scalar(0, .i32));
    const matched_ids = found.select(
        expected_ids,
        zml.Tensor.scalar(-1, .i64).broad(expected_ids.shape()),
    );
    const weight_grid = actual_weights.reshape(.{
        .b = actual_ids.dim(.b),
        .s = actual_ids.dim(.s),
        .expected_route = 1,
        .actual_route = actual_ids.dim(.actual_route),
    }).broad(match_shape.withDtype(.f32));
    return .{
        .matched_ids = matched_ids,
        .aligned_weights = matches.convert(.f32).mul(weight_grid)
            .sum(.actual_route).squeeze(.actual_route),
    };
}

// KIMI_K3_TEMP_REMOVE_M20: align route-indexed expert activations by local ID
// solely for comparison; production aggregation remains order independent.
fn alignRouteOutputs(values: zml.Tensor, actual_ids_raw: zml.Tensor, expected_ids: zml.Tensor) zml.Tensor {
    const actual_ids = actual_ids_raw.rename(.{ .route = .actual_route });
    const match_shape = zml.Shape.init(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route),
    }, .i64);
    const expected_grid = expected_ids.reshape(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = 1,
    }).broad(match_shape);
    const actual_grid = actual_ids.reshape(.{
        .b = actual_ids.dim(.b),
        .s = actual_ids.dim(.s),
        .expected_route = 1,
        .actual_route = actual_ids.dim(.actual_route),
    }).broad(match_shape);
    const matches = actual_grid.cmp(.EQ, expected_grid);
    const output_shape = zml.Shape.init(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route),
        .latent = values.dim(.latent),
    }, values.dtype());
    const value_grid = values.reshape(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = 1,
        .actual_route = actual_ids.dim(.actual_route),
        .latent = values.dim(.latent),
    }).broad(output_shape);
    const match_grid = matches.reshape(.{
        .b = expected_ids.dim(.b),
        .s = expected_ids.dim(.s),
        .expected_route = expected_ids.dim(.expected_route),
        .actual_route = actual_ids.dim(.actual_route),
        .latent = 1,
    }).broad(output_shape.withDtype(.bool));
    return value_grid.mul(match_grid.convert(values.dtype()))
        .sum(.actual_route).squeeze(.actual_route)
        .merge(.{ .token = .{ .b, .s } })
        .rename(.{ .expected_route = .route });
}

const KdaHarnessResult = struct {
    layer_result: layer.KdaMoeResult,
    global: GlobalAlignment,
    local: GlobalAlignment,
    aligned_route_outputs: zml.Tensor,
};

fn kdaPrefill(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.KdaMoeWeights,
    cache: kda.Cache,
    selected: zml.Tensor,
    official_moe_input: zml.Tensor,
    expected_global_ids: zml.Tensor,
    expected_local_ids: zml.Tensor,
) KdaHarnessResult {
    var local_weights = weights;
    local_weights.common = localize(weights.common, selected);
    const result = layer.forwardKdaMoePrefill(input, blocks, active, local_weights, cache, .{ .top_k = 16 });
    const global_route = router.forward(official_moe_input, weights.common.moe.gate, .{ .top_k = 16 });
    return .{
        .layer_result = result,
        .global = alignGlobal(global_route, expected_global_ids),
        .local = alignGlobal(result.moe_result.route, expected_local_ids),
        .aligned_route_outputs = alignRouteOutputs(result.moe_result.route_outputs, result.moe_result.route.topk_ids, expected_local_ids),
    };
}

fn kdaDecode(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.KdaMoeWeights,
    cache: kda.Cache,
    selected: zml.Tensor,
    official_moe_input: zml.Tensor,
    expected_global_ids: zml.Tensor,
    expected_local_ids: zml.Tensor,
) KdaHarnessResult {
    var local_weights = weights;
    local_weights.common = localize(weights.common, selected);
    const result = layer.forwardKdaMoeDecode(input, blocks, active, local_weights, cache, .{ .top_k = 16 });
    const global_route = router.forward(official_moe_input, weights.common.moe.gate, .{ .top_k = 16 });
    return .{
        .layer_result = result,
        .global = alignGlobal(global_route, expected_global_ids),
        .local = alignGlobal(result.moe_result.route, expected_local_ids),
        .aligned_route_outputs = alignRouteOutputs(result.moe_result.route_outputs, result.moe_result.route.topk_ids, expected_local_ids),
    };
}

const Expanded = struct {
    key: zml.Tensor,
    value: zml.Tensor,
};

fn expandLatent(cache: mla.LatentCache, weights: mla.Weights) Expanded {
    const kv = cache.compressed.dot(weights.kv_b_proj, .kv_rank);
    const heads: i64 = 96;
    const split = kv.splitAxis(.kv_mix, .{ .h = heads, .kv_width = 256 })
        .transpose(.{ .b, .h, .k, .kv_width });
    const pass = split.slice1d(.kv_width, .{ .start = 0, .end = 128 }).rename(.{ .kv_width = .hd });
    const value = split.slice1d(.kv_width, .{ .start = 128, .end = 256 }).rename(.{ .kv_width = .v });
    const extra = cache.extra_key.reshape(.{
        .b = cache.extra_key.dim(.b),
        .h = 1,
        .k = cache.extra_key.dim(.k),
        .hd = 64,
    }).broad(zml.Shape.init(.{
        .b = cache.extra_key.dim(.b),
        .h = heads,
        .k = cache.extra_key.dim(.k),
        .hd = 64,
    }, cache.extra_key.dtype()));
    return .{ .key = zml.Tensor.concatenate(&.{ pass, extra }, .hd), .value = value };
}

const MlaHarnessResult = struct {
    layer_result: layer.MlaMoeResult,
    global: GlobalAlignment,
    local: GlobalAlignment,
    aligned_route_outputs: zml.Tensor,
    expanded: Expanded,
};

fn mlaPrefill(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.MlaMoeWeights,
    selected: zml.Tensor,
    official_moe_input: zml.Tensor,
    expected_global_ids: zml.Tensor,
    expected_local_ids: zml.Tensor,
) MlaHarnessResult {
    var local_weights = weights;
    local_weights.common = localize(weights.common, selected);
    const result = layer.forwardMlaMoePrefill(input, blocks, active, local_weights, .{ .top_k = 16 });
    const global_route = router.forward(official_moe_input, weights.common.moe.gate, .{ .top_k = 16 });
    return .{
        .layer_result = result,
        .global = alignGlobal(global_route, expected_global_ids),
        .local = alignGlobal(result.moe_result.route, expected_local_ids),
        .aligned_route_outputs = alignRouteOutputs(result.moe_result.route_outputs, result.moe_result.route.topk_ids, expected_local_ids),
        .expanded = expandLatent(result.cache, weights.attention),
    };
}

fn mlaWarm(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.MlaMoeWeights,
) mla.LatentCache {
    const warm_input = input.slice1d(.s, .{ .start = 0, .end = 3 });
    const warm_blocks = blocks.slice1d(.token, .{ .start = 0, .end = 3 });
    return layer.forwardMlaMoePrefill(warm_input, warm_blocks, active, weights, .{ .top_k = 16 }).cache;
}

fn mlaDecode(
    input: zml.Tensor,
    blocks: zml.Tensor,
    active: zml.Tensor,
    weights: layer.MlaMoeWeights,
    cache: mla.LatentCache,
    selected: zml.Tensor,
    official_moe_input: zml.Tensor,
    expected_global_ids: zml.Tensor,
    expected_local_ids: zml.Tensor,
) MlaHarnessResult {
    var local_weights = weights;
    local_weights.common = localize(weights.common, selected);
    const result = layer.forwardMlaMoeContinue(input, blocks, active, local_weights, cache, .{ .top_k = 16 });
    const global_route = router.forward(official_moe_input, weights.common.moe.gate, .{ .top_k = 16 });
    return .{
        .layer_result = result,
        .global = alignGlobal(global_route, expected_global_ids),
        .local = alignGlobal(result.moe_result.route, expected_local_ids),
        .aligned_route_outputs = alignRouteOutputs(result.moe_result.route_outputs, result.moe_result.route.topk_ids, expected_local_ids),
        .expanded = expandLatent(result.cache, weights.attention),
    };
}

fn runKda(context: *Context, layer_index: usize) !void {
    const load_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var weights = try context.loadKdaWeights(layer_index);
    defer zml.Buffer.deinitAll(layer.KdaMoeWeights, &weights);
    const expert_count: usize = @intCast(weights.common.moe.experts.w1.values.shape().dim(.expert));
    var selected = try context.selectedIds(layer_index, expert_count);
    defer selected.deinit();
    var active = try context.active();
    defer active.deinit();
    const load_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - load_started, 1000);

    const cases = .{
        .{ "", kdaPrefill },
        .{ "decode", kdaDecode },
    };
    inline for (cases) |case| {
        const mode = case[0];
        const input_suffix = if (mode.len == 0) "input" else "decode.input";
        const block_suffix = if (mode.len == 0) "block_residual" else "decode.block_residual";
        var input = try context.load(layer_index, input_suffix, .{ .b, .s, .d });
        defer input.deinit();
        var blocks = try context.load(layer_index, block_suffix, .{ .token, .source, .d });
        defer blocks.deinit();
        const route_suffix = if (mode.len == 0) "route.global_ids" else "decode.route.global_ids";
        var expected_ids = try context.load(layer_index, route_suffix, .{ .b, .s, .expected_route });
        defer expected_ids.deinit();
        const local_suffix = if (mode.len == 0) "route.local_ids" else "decode.route.local_ids";
        var expected_local_ids = try context.load(layer_index, local_suffix, .{ .b, .s, .expected_route });
        defer expected_local_ids.deinit();
        const moe_suffix = if (mode.len == 0) "moe_input" else "decode.moe_input";
        var official_moe_input = try context.load(layer_index, moe_suffix, .{ .b, .s, .d });
        defer official_moe_input.deinit();
        var cache = if (mode.len == 0)
            try context.zeroKdaCache()
        else
            try context.loadKdaCache(layer_index, "decode.cache_in");
        defer zml.Buffer.deinitAll(kda.Cache, &cache);
        const symbolic_weights = kdaTensors(weights);
        const compile_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
        const exe = try context.platform.compileFn(
            context.allocator,
            context.io,
            case[1],
            .{
                tensor(input), tensor(blocks), tensor(active), symbolic_weights,
                kdaCacheTensors(cache), tensor(selected), tensor(official_moe_input), tensor(expected_ids), tensor(expected_local_ids),
            },
            .{ .shardings = &.{context.sharding} },
        );
        defer exe.deinit();
        const compile_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - compile_started, 1000);
        const execute_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            context.allocator,
            context.io,
            &exe,
            case[1],
            .{ input, blocks, active, weights, cache, selected, official_moe_input, expected_ids, expected_local_ids },
        );
        defer zml.Buffer.deinitAll(KdaHarnessResult, &actual);
        const execute_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - execute_started, 1000);
        try context.compareCommon(layer_index, mode, actual.layer_result, actual.global, actual.local, actual.aligned_route_outputs);
        const cache_mode = if (mode.len == 0) "" else "decode.cache_out";
        try context.compareKdaCache(layer_index, cache_mode, actual.layer_result.cache);
        try context.stdout.print(
            "KIMI_K3_LAYER_FAMILY_PASS layer={} attention=kda mode={s} experts={} boundaries=24 load_us={} compile_us={} execute_us={}\n",
            .{ layer_index, if (mode.len == 0) "prefill" else mode, expert_count, load_us, compile_us, execute_us },
        );
        try context.stdout.flush();
    }
}

fn runMla(context: *Context, layer_index: usize) !void {
    const load_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var weights = try context.loadMlaWeights(layer_index);
    defer zml.Buffer.deinitAll(layer.MlaMoeWeights, &weights);
    const expert_count: usize = @intCast(weights.common.moe.experts.w1.values.shape().dim(.expert));
    var selected = try context.selectedIds(layer_index, expert_count);
    defer selected.deinit();
    var active = try context.active();
    defer active.deinit();
    var full_input = try context.load(layer_index, "input", .{ .b, .s, .d });
    defer full_input.deinit();
    var full_blocks = try context.load(layer_index, "block_residual", .{ .token, .source, .d });
    defer full_blocks.deinit();
    const load_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - load_started, 1000);
    const symbolic_weights = mlaTensors(weights);

    const prefill_compile_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var prefill_expected_ids = try context.load(layer_index, "route.global_ids", .{ .b, .s, .expected_route });
    defer prefill_expected_ids.deinit();
    var prefill_expected_local_ids = try context.load(layer_index, "route.local_ids", .{ .b, .s, .expected_route });
    defer prefill_expected_local_ids.deinit();
    var prefill_official_moe_input = try context.load(layer_index, "moe_input", .{ .b, .s, .d });
    defer prefill_official_moe_input.deinit();
    const prefill_exe = try context.platform.compileFn(
        context.allocator,
        context.io,
        mlaPrefill,
        .{ tensor(full_input), tensor(full_blocks), tensor(active), symbolic_weights, tensor(selected), tensor(prefill_official_moe_input), tensor(prefill_expected_ids), tensor(prefill_expected_local_ids) },
        .{ .shardings = &.{context.sharding} },
    );
    defer prefill_exe.deinit();
    const prefill_compile_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - prefill_compile_started, 1000);
    const prefill_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var prefill = try zml.testing.autoCall(
        context.allocator,
        context.io,
        &prefill_exe,
        mlaPrefill,
        .{ full_input, full_blocks, active, weights, selected, prefill_official_moe_input, prefill_expected_ids, prefill_expected_local_ids },
    );
    defer zml.Buffer.deinitAll(MlaHarnessResult, &prefill);
    const prefill_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - prefill_started, 1000);
    try context.compareCommon(layer_index, "", prefill.layer_result, prefill.global, prefill.local, prefill.aligned_route_outputs);
    try context.compare(layer_index, "", "cache.key", prefill.expanded.key, support.bf16_tolerance);
    try context.compare(layer_index, "", "cache.value", prefill.expanded.value, support.bf16_tolerance);
    try context.stdout.print(
        "KIMI_K3_LAYER_FAMILY_PASS layer={} attention=mla mode=prefill experts={} boundaries=22 load_us={} compile_us={} execute_us={}\n",
        .{ layer_index, expert_count, load_us, prefill_compile_us, prefill_us },
    );
    try context.stdout.flush();

    const warm_compile_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    const warm_exe = try context.platform.compileFn(
        context.allocator,
        context.io,
        mlaWarm,
        .{ tensor(full_input), tensor(full_blocks), tensor(active), symbolic_weights },
        .{ .shardings = &.{context.sharding} },
    );
    defer warm_exe.deinit();
    const warm_compile_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - warm_compile_started, 1000);
    const warm_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var warm_cache = try zml.testing.autoCall(
        context.allocator,
        context.io,
        &warm_exe,
        mlaWarm,
        .{ full_input, full_blocks, active, weights },
    );
    defer zml.Buffer.deinitAll(mla.LatentCache, &warm_cache);
    const warm_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - warm_started, 1000);

    var decode_input = try context.load(layer_index, "decode.input", .{ .b, .s, .d });
    defer decode_input.deinit();
    var decode_blocks = try context.load(layer_index, "decode.block_residual", .{ .token, .source, .d });
    defer decode_blocks.deinit();
    var decode_expected_ids = try context.load(layer_index, "decode.route.global_ids", .{ .b, .s, .expected_route });
    defer decode_expected_ids.deinit();
    var decode_expected_local_ids = try context.load(layer_index, "decode.route.local_ids", .{ .b, .s, .expected_route });
    defer decode_expected_local_ids.deinit();
    var decode_official_moe_input = try context.load(layer_index, "decode.moe_input", .{ .b, .s, .d });
    defer decode_official_moe_input.deinit();
    const decode_compile_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    const decode_exe = try context.platform.compileFn(
        context.allocator,
        context.io,
        mlaDecode,
        .{
            tensor(decode_input), tensor(decode_blocks), tensor(active), symbolic_weights,
            mlaCacheTensors(warm_cache), tensor(selected), tensor(decode_official_moe_input), tensor(decode_expected_ids), tensor(decode_expected_local_ids),
        },
        .{ .shardings = &.{context.sharding} },
    );
    defer decode_exe.deinit();
    const decode_compile_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - decode_compile_started, 1000);
    const decode_started = std.Io.Clock.now(.real, context.io).toNanoseconds();
    var decode = try zml.testing.autoCall(
        context.allocator,
        context.io,
        &decode_exe,
        mlaDecode,
        .{ decode_input, decode_blocks, active, weights, warm_cache, selected, decode_official_moe_input, decode_expected_ids, decode_expected_local_ids },
    );
    defer zml.Buffer.deinitAll(MlaHarnessResult, &decode);
    const decode_us = @divTrunc(std.Io.Clock.now(.real, context.io).toNanoseconds() - decode_started, 1000);
    try context.compareCommon(layer_index, "decode", decode.layer_result, decode.global, decode.local, decode.aligned_route_outputs);
    try context.compare(layer_index, "decode.cache_out", "key", decode.expanded.key, support.bf16_tolerance);
    try context.compare(layer_index, "decode.cache_out", "value", decode.expanded.value, support.bf16_tolerance);
    try context.stdout.print(
        "KIMI_K3_LAYER_FAMILY_PASS layer={} attention=mla mode=decode experts={} boundaries=22 warm_compile_us={} warm_execute_us={} compile_us={} execute_us={}\n",
        .{ layer_index, expert_count, warm_compile_us, warm_us, decode_compile_us, decode_us },
    );
    try context.stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    var context: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
    };

    // Run and release one family at a time to bound H100 residency.
    try runKda(&context, 1);
    try runKda(&context, 2);
    try runMla(&context, 3);
    try stdout_file.interface.writeAll(
        "KIMI_K3_LAYER_FAMILY_ALL_PASS layers=1,2,3 prefill=3 decode=3 backend=cuda global_routes=exact\n",
    );
    try stdout_file.interface.flush();
}
