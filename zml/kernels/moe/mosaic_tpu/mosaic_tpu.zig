const std = @import("std");

const mtt = @import("dsl/mosaic_tpu/builder");
const stdx = @import("stdx");

const nn = @import("zml/nn");

const zml = @import("zml/core");
const Tensor = zml.Tensor;
const Activation = @import("zml/kernels/moe/api").Activation;
const GlobalOptions = @import("zml/kernels/moe/api").Options;
const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
pub const gmm_ep = @import("gmm_ep.zig");

const log = std.log.scoped(.@"zml/moe/mosaic_tpu");

var gate_up_transpose: std.atomic.Value(bool) = .init(false);
var down_transpose: std.atomic.Value(bool) = .init(false);

fn validateOptions(opts: GlobalOptions) void {
    stdx.debug.assert(!opts.quantize_input, "Optional FP8 input quantization requires the Triton MoE backend", .{});
    stdx.debug.assert(opts.routing_weight_placement == .after_down, "Non-Triton MoE backends require routing weights after the down projection", .{});
}

fn validateInputs(hidden: Tensor, gate_up: Tensor, down: Tensor, weights: Tensor, ids: Tensor) !void {
    switch (hidden.dtype()) {
        .bf16, .f16, .f32 => {},
        else => return error.UnsupportedType,
    }
    switch (gate_up.dtype()) {
        .bf16, .f16, .f32 => {},
        else => return error.UnsupportedType,
    }
    switch (down.dtype()) {
        .bf16, .f16, .f32 => {},
        else => return error.UnsupportedType,
    }
    switch (weights.dtype()) {
        .bf16, .f16, .f32 => {},
        else => return error.UnsupportedType,
    }
    if (ids.dtype() != .i32) return error.UnsupportedType;
    if (hidden.dim(.in) != gate_up.dim(.in)) return error.InvalidShape;
    if (@rem(gate_up.dim(.out), 2) != 0) return error.InvalidShape;
    if (down.dim(.mid) != @divFloor(gate_up.dim(.out), 2)) return error.InvalidShape;
    if (ids.dim(.token) != hidden.dim(.token) or weights.dim(.token) != hidden.dim(.token)) return error.InvalidShape;
    if (ids.dim(.topk) != weights.dim(.topk)) return error.InvalidShape;
    if (gate_up.dim(.expert) != down.dim(.expert)) return error.InvalidShape;
}

fn warningTranspose(flag: *std.atomic.Value(bool), projection_name: []const u8, expected_shape: []const u8) void {
    if (!flag.swap(true, .monotonic)) {
        log.warn("{s} MoE weights are transposed at runtime; weights must be loaded with the correct shape {s} to avoid the transpose at runtime.", .{ projection_name, expected_shape });
    }
}

fn canonicalizeGateUp(w1: Tensor, hidden_size: i64) !Tensor {
    if (w1.rank() != 3) return error.InvalidShape;

    if (w1.dim(1) == hidden_size) return w1.withTags(.{ .expert, .in, .out });
    if (w1.dim(2) == hidden_size) {
        warningTranspose(&gate_up_transpose, "gate_up_proj", "[expert, in, out]");
        return w1.withTags(.{ .expert, .out, .in }).transpose(.{ .expert, .in, .out });
    }
    return error.InvalidShape;
}

fn canonicalizeDown(w2: Tensor, hidden_size: i64, gate_up_out: i64) !Tensor {
    if (w2.rank() != 3) return error.InvalidShape;

    const mid_size = @divFloor(gate_up_out, 2);
    if (w2.dim(1) == mid_size and w2.dim(2) == hidden_size) return w2.withTags(.{ .expert, .mid, .out });
    if (w2.dim(1) == hidden_size and w2.dim(2) == mid_size) {
        warningTranspose(&down_transpose, "down_proj", "[expert, mid, out]");
        return w2.withTags(.{ .expert, .out, .mid }).transpose(.{ .expert, .mid, .out });
    }
    return error.InvalidShape;
}

fn histogramCounts(ids: Tensor, num_bins: i64) Tensor {
    const zeros = Tensor.zeroes(.init(.{ .expert = num_bins }, .i32));
    const ones = Tensor.constant(.{ .i32 = 1 }).broad(ids.shape().withDtype(.i32));
    return zeros.scatterSlices(
        .{ .expert = ids },
        ones,
        .{
            .update_fn = Tensor.ScatterOpts.increment,
            .indices_are_unique = false,
        },
    );
}

fn alignSortedRowsByGroup(rows: Tensor, expert_ids_sorted: Tensor, group_sizes: Tensor, tile_m: i64) struct {
    rows: Tensor,
    group_sizes: Tensor,
    positions: Tensor,
} {
    const num_groups = group_sizes.dim(.expert);
    const max_padded_rows = rows.dim(.token) + num_groups * (tile_m - 1);

    const group_ends = group_sizes.cumulativeSum(.expert);
    const group_starts = Tensor.concatenate(&.{
        Tensor.zeroes(.init(.{ .expert = 1 }, .i32)),
        group_ends.slice(.expert, .{ .end = num_groups - 1 }),
    }, .expert);

    const padded_group_sizes = group_sizes.addConstant(tile_m - 1).divByConst(tile_m).mul(Tensor.scalar(tile_m, .i32));
    const padded_group_ends = padded_group_sizes.cumulativeSum(.expert);
    const padded_group_starts = Tensor.concatenate(&.{
        Tensor.zeroes(.init(.{ .expert = 1 }, .i32)),
        padded_group_ends.slice(.expert, .{ .end = num_groups - 1 }),
    }, .expert);

    const logical_positions = Tensor.arange(.{ .end = rows.dim(.token) }, .i32).withTags(.{.token});
    const group_start_per_token = group_starts.gather(.{ .expert = expert_ids_sorted }, .{});
    const padded_group_start_per_token = padded_group_starts.gather(.{ .expert = expert_ids_sorted }, .{});
    const padded_positions = logical_positions.sub(group_start_per_token).add(padded_group_start_per_token).withTags(.{.token});

    const padded = Tensor.zeroes(rows.shape().setDim(.token, max_padded_rows))
        .scatterSlices(
        .{ .token = padded_positions },
        rows,
        .{ .indices_are_unique = true, .update_fn = Tensor.ScatterOpts.override },
    );

    return .{
        .rows = padded,
        .group_sizes = padded_group_sizes,
        .positions = padded_positions,
    };
}

fn permuteSmallBatch(hidden: Tensor, sorted_route: Tensor, topk: i64) Tensor {
    const num_tokens = hidden.dim(.token);
    const num_routes = num_tokens * topk;

    const token_indices = Tensor.arange(.{ .end = num_tokens }, .i32)
        .withTags(.{.token})
        .repeat1d(.token, @intCast(topk))
        .rename(.{ .token = .route });
    const token_indices_sorted = token_indices.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .route });

    const onehot_shape = zml.Shape.init(.{ .route = num_routes, .token_src = num_tokens }, .i32);
    const token_indices_2d = token_indices_sorted.insertAxes(.last, .{.token_src}).broad(onehot_shape);
    const source_tokens_2d = Tensor.arange(.{ .end = num_tokens }, .i32)
        .withTags(.{.token_src})
        .insertAxes(0, .{.route})
        .broad(onehot_shape);
    const onehot = token_indices_2d.cmp(.EQ, source_tokens_2d).convert(hidden.dtype());

    return onehot.dot(hidden.rename(.{ .token = .token_src }), .token_src).rename(.{ .route = .token });
}

fn gmmDType(dtype: zml.DataType) mtt.DType {
    return switch (zml.kernel.mosaic_tpu.from(dtype)) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        else => stdx.debug.panic("mosaic_tpu MoE expects bf16/f16/f32 tensors, got {}", .{dtype}),
    };
}

pub fn callGmmEp(
    lhs: Tensor,
    rhs_k_n: Tensor,
    group_sizes: Tensor,
    out_dtype: zml.DataType,
    fuse_act: gmm_ep.Cfg.FuseAct,
) Tensor {
    const k = lhs.dim(1);
    const n = rhs_k_n.dim(.out);
    const out_n = switch (fuse_act) {
        .none => n,
        .silu, .gelu => @divExact(n, 2),
    };
    const aligned_k = @divFloor(k + 127, 128) * 128;
    const aligned_n = if (fuse_act == .none) @divFloor(out_n + 127, 128) * 128 else out_n;

    const cfg: gmm_ep.Cfg = .{
        .size_m = lhs.dim(.token),
        .size_k = k,
        .size_n = n,
        .size_group = rhs_k_n.dim(.expert),
        .size_lhs_group = group_sizes.dim(.expert),
        .tile_m = 128,
        .tile_k = aligned_k,
        .tile_n = if (fuse_act == .none) aligned_n else out_n,
        .lhs_dtype = gmmDType(lhs.dtype()),
        .rhs_dtype = gmmDType(rhs_k_n.dtype()),
        .out_dtype = gmmDType(out_dtype),
        .acc_dtype = .f32,
        .size_lhs_sublane = 8,
        .fuse_act = fuse_act,
    };

    const out = gmm_ep.TpuKernel.call(
        .{
            .group_sizes = group_sizes.withTags(.{.expert}),
            .group_offset = Tensor.zeroes(.init(.{1}, .i32)),
            .lhs = lhs,
            .rhs = rhs_k_n,
        },
        .{
            .out = zml.Shape.init(.{ .token = lhs.dim(.token), .out = aligned_n }, out_dtype),
        },
        .{
            .cfg = cfg,
            .extras = .{
                // todo leverage tpu api like this https://github.com/openxla/tokamax/blob/f65135e85acd7571531be91d4d3241e1929421f0/tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu_v2_tgmm_kernel.py#L692
                .vmem_limit_bytes = 32 * 1024 * 1024,
            },
        },
    ).out;

    return out.slice(.out, .{ .end = out_n });
}

pub fn fusedExperts(
    hidden_states: zml.Tensor,
    routing: LocalRouting,
    gate_up_: nn.Linear,
    down_: nn.Linear,
    opts: GlobalOptions,
) zml.Tensor {
    validateOptions(opts);
    const expert_map_ = routing.expertMap();

    const tokens = hidden_states.dim(.t);
    const hidden = hidden_states.reshape(.{ .token = tokens, .in = hidden_states.dim(.d) });
    // TODO(Corentin): Better error message
    const gate_up = canonicalizeGateUp(gate_up_.weight, hidden.dim(.in)) catch |e| stdx.debug.panic("Invalid gate_up weights for fusedExperts: {}", .{e});
    const down = canonicalizeDown(down_.weight, hidden.dim(.in), gate_up.dim(.out)) catch |e| stdx.debug.panic("Invalid down weights for fusedExperts: {}", .{e});
    const weights = routing.weights.reshape(.{ .token = tokens, .topk = routing.weights.dim(.topk) });
    const ids = routing.ids.reshape(.{ .token = tokens, .topk = routing.ids.dim(.topk) });

    const num_experts = routing.global_num_experts;
    if (expert_map_) |expert_map| {
        if (expert_map.dtype() != .i32) stdx.debug.panic("expert_map must be of dtype i32, got {}", .{expert_map.dtype()});
        if (expert_map.rank() != 1 or expert_map.dim(.expert) != num_experts) stdx.debug.panic("expert_map must be a 1D tensor of length num_experts, got shape {}", .{expert_map.shape()});
    }

    const flat_ids_global = ids.transpose(.{ .topk, .token }).flatten().withTags(.{.route});
    const flat_weights = weights.transpose(.{ .topk, .token }).flatten().withTags(.{.route});

    const flat_ids_local, const local_route_mask = if (expert_map_) |expert_map| blk: {
        const mapped = expert_map.gather(.{ .expert = flat_ids_global }, .{}).withTags(.{.route});
        const route_mask = mapped.cmp(.GE, Tensor.scalar(0, .i32).broad(mapped.shape()));
        const safe_ids = route_mask.select(mapped, Tensor.zeroes(mapped.shape()));
        break :blk .{ safe_ids, route_mask };
    } else .{
        flat_ids_global,
        Tensor.scalar(true, .bool).broad(flat_ids_global.shape()),
    };

    const sorted_route = flat_ids_local.argsort(.route, .{ .descending = false }).rename(.{ .route = .perm });
    const hidden_sorted = if (expert_map_ != null)
        permuteSmallBatch(hidden, sorted_route, ids.dim(.topk))
    else blk: {
        // If not sharded, we can save the sort + gather by just repeating the hidden states for each expert and gathering with the sorted route
        const hidden_repeated = hidden.repeat1d(.token, @intCast(ids.dim(.topk))).rename(.{ .token = .route });
        break :blk hidden_repeated.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    };
    const expert_ids_sorted = flat_ids_local.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const local_route_mask_sorted = local_route_mask.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const sorted_weights = flat_weights.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const router_weights_sorted = local_route_mask_sorted.select(
        sorted_weights,
        Tensor.zeroes(sorted_weights.shape()),
    );

    const group_sizes = histogramCounts(flat_ids_local, gate_up.dim(.expert));
    const tile_m: i64 = 128;
    const aligned_hidden = alignSortedRowsByGroup(hidden_sorted, expert_ids_sorted, group_sizes, tile_m);

    var gate_up_out = callGmmEp(aligned_hidden.rows, gate_up, aligned_hidden.group_sizes, hidden.dtype(), .none)
        .gather(.{ .token = aligned_hidden.positions.rename(.{ .token = .route }) }, .{})
        .rename(.{ .route = .token });

    if (gate_up_.bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        gate_up_out = gate_up_out.add(bias_per_token);
    }

    const activated = opts.activation.applyActivation(gate_up_out, .concatenated);

    const aligned_activated = alignSortedRowsByGroup(activated, expert_ids_sorted, group_sizes, tile_m);
    var down_out = callGmmEp(aligned_activated.rows, down, aligned_activated.group_sizes, hidden.dtype(), .none)
        .gather(.{ .token = aligned_activated.positions.rename(.{ .token = .route }) }, .{})
        .rename(.{ .route = .token });

    if (down_.bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        down_out = down_out.add(bias_per_token);
    }

    const weighted = down_out.mul(router_weights_sorted.convert(down_out.dtype()).broad(down_out.shape()));

    const restore_indices = sorted_route.argsort(.perm, .{ .descending = false }).rename(.{ .perm = .restore });
    const unsorted = weighted.gather(.{ .token = restore_indices }, .{}).rename(.{ .restore = .route });
    const restored = unsorted.reshape(.{ .topk = ids.dim(.topk), .token = hidden.dim(.token), .out = down.dim(.out) });
    const combined = restored.sum(.topk).squeeze(.topk);
    return combined.reshape(.{ .t = tokens, .d = down.dim(.out) });
}
