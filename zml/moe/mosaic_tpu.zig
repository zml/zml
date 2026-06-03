const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;

const gmm_ep = @import("mosaic_tpu_kernels/gmm_ep.zig");
const mtt = @import("kernels/mosaic_tpu/builder");

const log = std.log.scoped(.moe_mosaic_tpu);

var warned_gate_up_runtime_transpose: std.atomic.Value(bool) = .init(false);
var warned_down_runtime_transpose: std.atomic.Value(bool) = .init(false);

pub const ActivationMode = enum {
    silu,
    relu,
    quick_gelu_plus_one,
};

pub const Options = struct {
    activation: ActivationMode = .silu,
    global_num_experts: i64 = -1,
    expert_map: ?Tensor = null,
    w1_scale: ?Tensor = null,
    w2_scale: ?Tensor = null,
    w1_bias: ?Tensor = null,
    w2_bias: ?Tensor = null,
};

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: ActivationMode,

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: ActivationMode = .silu,
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
        };
    }
};

pub const Metadata = struct {
    pub const InitOptions = struct {};

    pub fn init(opts: InitOptions) Metadata {
        _ = opts;
        return .{};
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        _ = io;
        _ = platform;
        return {};
    }
};

pub fn deinitBuffer(bufferized: *zml.Bufferized(Metadata)) void {
    _ = bufferized;
}

fn validateOptions(opts: Options) !void {
    if (opts.activation != .silu and opts.activation != .relu and opts.activation != .quick_gelu_plus_one) return error.UnsupportedActivation;
    if (opts.expert_map != null and opts.global_num_experts == -1) return error.InvalidShape;
    if (opts.w1_scale != null or opts.w2_scale != null) return error.UnsupportedQuantization;
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

fn warnRuntimeWeightTranspose(flag: *std.atomic.Value(bool), projection_name: []const u8, expected_shape: []const u8) void {
    if (!flag.swap(true, .monotonic)) {
        log.warn("{s} MoE weights are transposed at runtime; weights must be loaded with the correct shape {s} to avoid the transpose at runtime.", .{ projection_name, expected_shape });
    }
}

fn canonicalizeGateUpForGmm(w1: Tensor, hidden_size: i64) !Tensor {
    if (w1.rank() != 3) return error.InvalidShape;

    if (w1.dim(1) == hidden_size) return w1.withTags(.{ .expert, .in, .out });
    if (w1.dim(2) == hidden_size) {
        warnRuntimeWeightTranspose(&warned_gate_up_runtime_transpose, "gate_up_proj", "[expert, in, out]");
        return w1.withTags(.{ .expert, .out, .in }).transpose(.{ .expert, .in, .out });
    }
    return error.InvalidShape;
}

fn canonicalizeDownForGmm(w2: Tensor, hidden_size: i64, gate_up_out: i64) !Tensor {
    if (w2.rank() != 3) return error.InvalidShape;

    const mid_size = @divFloor(gate_up_out, 2);
    if (w2.dim(1) == mid_size and w2.dim(2) == hidden_size) return w2.withTags(.{ .expert, .mid, .out });
    if (w2.dim(1) == hidden_size and w2.dim(2) == mid_size) {
        warnRuntimeWeightTranspose(&warned_down_runtime_transpose, "down_proj", "[expert, mid, out]");
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
        group_ends.slice1d(.expert, .{ .end = num_groups - 1 }),
    }, .expert);

    const padded_group_sizes = group_sizes.addConstant(tile_m - 1).divByConst(tile_m).mul(Tensor.scalar(tile_m, .i32));
    const padded_group_ends = padded_group_sizes.cumulativeSum(.expert);
    const padded_group_starts = Tensor.concatenate(&.{
        Tensor.zeroes(.init(.{ .expert = 1 }, .i32)),
        padded_group_ends.slice1d(.expert, .{ .end = num_groups - 1 }),
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

fn permuteHiddenOneHotSmallBatchEp(hidden: Tensor, sorted_route: Tensor, topk: i64) Tensor {
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

fn callGmmEp(
    lhs: Tensor,
    rhs_k_n: Tensor,
    group_sizes: Tensor,
    out_dtype: zml.DataType,
) Tensor {
    const n = rhs_k_n.dim(.out);
    const aligned_n = @divFloor(n + 127, 128) * 128;
    const cfg: gmm_ep.Cfg = .{
        .size_m = lhs.dim(.token),
        .size_k = lhs.dim(1),
        .size_n = n,
        .size_group = rhs_k_n.dim(.expert),
        .size_lhs_group = group_sizes.dim(.expert),
        .tile_m = 8,
        .tile_k = lhs.dim(1),
        .tile_n = n,
        .lhs_dtype = gmmDType(lhs.dtype()),
        .rhs_dtype = gmmDType(rhs_k_n.dtype()),
        .out_dtype = gmmDType(out_dtype),
        .acc_dtype = .f32,
        .size_lhs_sublane = 8,
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
        .{ .cfg = cfg },
    ).out;

    return out.slice1d(.out, .{ .end = n });
}

fn applyActivation(x: Tensor, mode: ActivationMode) Tensor {
    const gate, const up = zml.nn.splitRealImg(x, .sequential);
    return switch (mode) {
        .silu => gate.silu().mul(up),
        .relu => x.relu().powByConst(2),
        .quick_gelu_plus_one => blk: {
            const gate_clamped = gate.minimum(Tensor.scalar(7, gate.dtype()).broad(gate.shape()));
            const up_clamped = up.clamp(
                Tensor.scalar(-7, up.dtype()).broad(up.shape()),
                Tensor.scalar(7, up.dtype()).broad(up.shape()),
            );
            break :blk gate_clamped.quickGelu().mul(up_clamped.addConstant(1));
        },
    };
}

pub fn fusedExpertsImpl(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    metadata: Metadata,
    opts: Options,
) !Tensor {
    _ = metadata;
    try validateOptions(opts);

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    const hidden = hidden_states.reshape(.{ .token = b * s, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = try canonicalizeGateUpForGmm(w1, hidden.dim(.in));
    const down = try canonicalizeDownForGmm(w2, hidden.dim(.in), gate_up.dim(.out));
    const weights = topk_weights.reshape(.{ .token = b * s, .topk = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = b * s, .topk = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    const num_experts = if (opts.global_num_experts != -1) opts.global_num_experts else gate_up.dim(.expert);
    try validateInputs(hidden, gate_up, down, weights, ids);
    if (opts.expert_map) |expert_map| {
        if (expert_map.dtype() != .i32) return error.UnsupportedType;
        if (expert_map.rank() != 1 or expert_map.dim(.expert) != num_experts) return error.InvalidShape;
    }

    const flat_ids_global = ids.transpose(.{ .topk, .token }).flatten().withTags(.{.route});
    const flat_weights = weights.transpose(.{ .topk, .token }).flatten().withTags(.{.route});

    const flat_ids_local, const local_route_mask = if (opts.expert_map) |expert_map| blk: {
        const mapped = expert_map.gather(.{ .expert = flat_ids_global }, .{}).withTags(.{.route});
        const route_mask = mapped.cmp(.GE, Tensor.scalar(0, .i32).broad(mapped.shape()));
        const safe_ids = route_mask.select(mapped, Tensor.zeroes(mapped.shape()));
        break :blk .{ safe_ids, route_mask };
    } else .{
        flat_ids_global,
        Tensor.scalar(true, .bool).broad(flat_ids_global.shape()),
    };

    const sorted_route = flat_ids_local.argsort(.route, .{ .descending = false }).rename(.{ .route = .perm });
    const hidden_sorted = if (opts.expert_map != null)
        permuteHiddenOneHotSmallBatchEp(hidden, sorted_route, ids.dim(.topk))
    else blk: {
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

    const tile_m: i64 = 8;
    const aligned_hidden = alignSortedRowsByGroup(hidden_sorted, expert_ids_sorted, group_sizes, tile_m);
    var gate_up_out = callGmmEp(aligned_hidden.rows, gate_up, aligned_hidden.group_sizes, hidden.dtype())
        .gather(.{ .token = aligned_hidden.positions.rename(.{ .token = .route }) }, .{})
        .rename(.{ .route = .token });
    if (opts.w1_bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        gate_up_out = gate_up_out.add(bias_per_token);
    }

    const activated = applyActivation(gate_up_out, opts.activation);

    const aligned_activated = alignSortedRowsByGroup(activated, expert_ids_sorted, group_sizes, tile_m);
    var down_out = callGmmEp(aligned_activated.rows, down, aligned_activated.group_sizes, hidden.dtype())
        .gather(.{ .token = aligned_activated.positions.rename(.{ .token = .route }) }, .{})
        .rename(.{ .route = .token });
    if (opts.w2_bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        down_out = down_out.add(bias_per_token);
    }

    const weighted = down_out.mul(router_weights_sorted.convert(down_out.dtype()).broad(down_out.shape()));

    const restore_indices = sorted_route.argsort(.perm, .{ .descending = false }).rename(.{ .perm = .restore });
    const unsorted = weighted.gather(.{ .token = restore_indices }, .{}).rename(.{ .restore = .route });
    const restored = unsorted.reshape(.{ .topk = ids.dim(.topk), .token = hidden.dim(.token), .out = down.dim(.out) });
    const combined = restored.sum(.topk).squeeze(.topk);
    return combined.reshape(.{ .b = b, .s = s, .d = down.dim(.out) });
}
