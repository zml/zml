const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;

const common = @import("common.zig");
const megablox_gmm = @import("mosaic_tpu_kernels/megablox_gmm.zig");
const mtt = @import("kernels/mosaic_tpu/builder");

pub const Options = struct {
    activation: common.ActivationMode = .silu,
    global_num_experts: i64 = -1,
    expert_map: ?Tensor = null,
    w1_scale: ?Tensor = null,
    w2_scale: ?Tensor = null,
    w1_bias: ?Tensor = null,
    w2_bias: ?Tensor = null,
};

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: common.ActivationMode,

    pub const ActivationMode = common.ActivationMode;

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: common.ActivationMode = .silu,
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
    if (opts.activation != .silu and opts.activation != .relu and opts.activation != .quick_gelu_plus_one) {
        return error.UnsupportedActivation;
    }
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
    if (@rem(common.gateUpOutDim(gate_up), 2) != 0) return error.InvalidShape;
    if (down.dim(.mid) != @divFloor(common.gateUpOutDim(gate_up), 2)) return error.InvalidShape;
    if (ids.dim(.token) != hidden.dim(.token) or weights.dim(.token) != hidden.dim(.token)) return error.InvalidShape;
    if (ids.dim(.topk) != weights.dim(.topk)) return error.InvalidShape;
    if (gate_up.dim(.expert) != down.dim(.expert)) return error.InvalidShape;
}

fn histogramCounts(ids: Tensor, num_bins: i64) Tensor {
    // Convert a flat list of expert ids into per-expert counts. This gives the
    // segment sizes that the grouped matmul metadata builder consumes.
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

fn repeatIdsFromCounts(counts: Tensor, total_len: i64, axis_name: anytype, max_id: i32) Tensor {
    // Invert a histogram back into sorted ids. For example counts [2, 0, 3]
    // becomes [0, 0, 2, 2, 2]. This is used to rebuild group/tile ownership
    // arrays from segment sizes.
    const count_ends = counts.cumulativeSum(0).rename(.{ .expert = .bucket });
    const positions = Tensor.arange(.{ .end = total_len }, .i32).withTags(.{axis_name});

    const shape_2d = zml.Shape.init(.{ .grid = total_len, .bucket = counts.dim(.expert) }, .i32);
    const positions_2d = positions.insertAxes(.last, .{.bucket}).broad(shape_2d);
    const count_ends_2d = count_ends.insertAxes(0, .{axis_name}).broad(shape_2d);
    const completed = positions_2d.cmp(.GE, count_ends_2d).convert(.i32);

    const ids = completed.sum(.bucket).squeeze(.bucket);
    return ids.minimum(Tensor.scalar(max_id, .i32).broad(ids.shape()));
}

fn buildGroupMetadata(group_sizes: Tensor, padded_rows: i64, tile_m: i64) struct {
    group_offsets: Tensor,
    group_ids: Tensor,
    m_tile_ids: Tensor,
    num_active_tiles: Tensor,
    group_offset: Tensor,
} {
    // Build the StableHLO-side metadata expected by the Megablox grouped
    // matmul kernel:
    // - group_offsets: prefix sums of expert segment boundaries
    // - group_ids: for each logical grid slot, which expert it belongs to
    // - m_tile_ids: for each logical grid slot, which M tile it should visit
    //
    // The custom call itself only performs grouped GEMM; all routing and
    // schedule bookkeeping stays outside in the backend.
    const num_groups = group_sizes.dim(.expert);
    const group_ends = group_sizes.cumulativeSum(.expert);
    const group_offsets = Tensor.concatenate(&.{
        Tensor.zeroes(.init(.{ .expert = 1 }, .i32)),
        group_ends,
    }, .expert).rename(.{ .expert = .group });

    const group_starts = Tensor.concatenate(&.{
        Tensor.zeroes(.init(.{ .group = 1 }, .i32)),
        group_ends.slice1d(.expert, .{ .end = num_groups - 1 }).rename(.{ .expert = .group }),
    }, .group);

    const rounded_group_ends = group_ends.addConstant(tile_m - 1).divByConst(tile_m).mul(Tensor.scalar(tile_m, .i32));
    const rounded_group_starts = group_starts.divByConst(tile_m).mul(Tensor.scalar(tile_m, .i32)).rename(.{ .group = .expert });
    const group_is_empty = group_sizes.cmp(.EQ, Tensor.zeroes(group_sizes.shape()));
    const rounded_group_sizes = Tensor.select(
        group_is_empty,
        Tensor.zeroes(group_sizes.shape()),
        rounded_group_ends.sub(rounded_group_starts),
    );
    const group_tiles = rounded_group_sizes.divByConst(tile_m);

    const tiles_m = std.math.divExact(i64, padded_rows, tile_m) catch unreachable;
    const metadata_len = tiles_m + num_groups - 1;
    const group_ids = repeatIdsFromCounts(group_tiles, metadata_len, .grid, @intCast(num_groups - 1)).withTags(.{.grid});

    const group_offsets_prefix = group_offsets.slice1d(.group, .{ .end = num_groups }).rename(.{ .group = .expert });
    const aligned_start = group_offsets_prefix.remainder(Tensor.scalar(tile_m, .i32)).cmp(.EQ, Tensor.zeroes(group_sizes.shape()));
    const partial_tile_mask = aligned_start.logical(.OR, group_is_empty);
    const valid_partial_visit = partial_tile_mask.not();
    const partial_tile_ids = group_offsets_prefix.divByConst(tile_m);
    const safe_partial_tile_ids = Tensor.select(
        valid_partial_visit,
        partial_tile_ids,
        Tensor.zeroes(partial_tile_ids.shape()),
    );
    const partial_tile_visits = histogramCounts(safe_partial_tile_ids.rename(.{ .expert = .partial }), tiles_m)
        .add(Tensor.constant(.{ .i32 = 1 }).broad(.init(.{ .expert = tiles_m }, .i32)));
    const m_tile_ids = repeatIdsFromCounts(partial_tile_visits, metadata_len, .grid, @intCast(tiles_m - 1)).withTags(.{.grid});
    const num_active_tiles = group_tiles.sum(.expert).squeeze(.expert).reshape(.{1});

    return .{
        .group_offsets = group_offsets,
        .group_ids = group_ids,
        .m_tile_ids = m_tile_ids,
        .num_active_tiles = num_active_tiles,
        .group_offset = Tensor.zeroes(.init(.{1}, .i32)),
    };
}

fn applyActivation(x: Tensor, mode: common.ActivationMode) Tensor {
    // Apply the logical MoE activation between the two expert matmuls. The
    // gate/up projection is packed as [gate, up], so we split it here before
    // the second grouped matmul.
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

fn padRowsToTile(x: Tensor, tile_m: i64) Tensor {
    // The grouped matmul kernel operates on a fixed M tile size. Routed tokens
    // are therefore padded to a tile multiple before the custom call.
    const rem = @mod(x.dim(0), tile_m);
    if (rem == 0) return x;
    return x.pad(0, .{ .token = Tensor.Pad{ .high = tile_m - rem } });
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

fn gmmDType(dtype: zml.DataType) mtt.DType {
    return switch (zml.kernel.mosaic_tpu.from(dtype)) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        else => stdx.debug.panic("mosaic_tpu MoE expects bf16/f16/f32 tensors, got {}", .{dtype}),
    };
}

fn callMegabloxGmm(
    lhs: Tensor,
    rhs: Tensor,
    group_sizes: Tensor,
    out_dtype: zml.DataType,
) Tensor {
    // Lower one grouped matmul custom call. By the time we get here, rows have
    // already been sorted by local expert and `group_sizes` describes the
    // resulting contiguous expert segments.
    const tile_m: i64 = 8;
    const metadata = buildGroupMetadata(group_sizes.withTags(.{.expert}), lhs.dim(.token), tile_m);
    const cfg: megablox_gmm.Cfg = .{
        .m = @intCast(lhs.dim(.token)),
        .k = @intCast(lhs.dim(1)),
        .n = @intCast(rhs.dim(.out)),
        .num_groups = @intCast(rhs.dim(.expert)),
        .num_active_tiles = @intCast(metadata.num_active_tiles.dim(0)),
        .tm = @intCast(tile_m),
        .tk = 128,
        .tn = 128,
        .dtype = gmmDType(lhs.dtype()),
        .preferred_element_type = gmmDType(out_dtype),
        .transpose_rhs = true,
    };

    return megablox_gmm.Kernel.call(
        .{
            .group_offsets = metadata.group_offsets,
            .group_ids = metadata.group_ids,
            .m_tile_ids = metadata.m_tile_ids,
            .num_active_tiles = metadata.num_active_tiles,
            .group_offset = metadata.group_offset,
            .lhs = lhs,
            .rhs = rhs,
        },
        .{
            .out = zml.Shape.init(.{ .token = lhs.dim(.token), .out = rhs.dim(.out) }, out_dtype),
        },
        .{ .cfg = cfg },
    ).out;
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

    // Canonicalize inputs into the shared logical MoE layout. TPU flattens the
    // gate/up weight one step further because the grouped matmul kernel
    // consumes a plain [expert, out, in] matrix.
    const inputs = common.canonicalizeInputs(hidden_states, w1, w2, topk_weights, topk_ids);
    const b = inputs.b;
    const s = inputs.s;
    const hidden = inputs.hidden;
    const gate_up = common.flattenGateUpForMatmul(inputs.gate_up);
    const down = inputs.down;
    const weights = inputs.weights;
    const ids = inputs.ids;

    const num_experts = if (opts.global_num_experts != -1) opts.global_num_experts else gate_up.dim(.expert);
    try validateInputs(hidden, gate_up, down, weights, ids);
    if (opts.expert_map) |expert_map| {
        if (expert_map.dtype() != .i32) return error.UnsupportedType;
        if (expert_map.rank() != 1 or expert_map.dim(.expert) != num_experts) return error.InvalidShape;
    }

    // Expand [token, topk] routing into a flat routed-token view. Each
    // original token now appears once per selected expert.
    const flat_ids_global = ids.transpose(.{ .topk, .token }).flatten().withTags(.{.route});
    const flat_weights = weights.transpose(.{ .topk, .token }).flatten().withTags(.{.route});

    // In expert-parallel mode, routing ids are global. Remap them to local
    // expert ids for this shard. Non-local routes are temporarily rewritten to
    // a safe local id and then masked back to zero contribution via router
    // weights after sorting.
    const flat_ids_local, const local_route_mask = if (opts.expert_map) |expert_map| blk: {
        const mapped = expert_map.gather(.{ .expert = flat_ids_global }, .{}).withTags(.{.route});
        const route_mask = mapped.cmp(.GE, Tensor.scalar(0, .i32).broad(mapped.shape()));
        const safe_ids = route_mask.select(mapped, Tensor.zeroes(mapped.shape()));
        break :blk .{ safe_ids, route_mask };
    } else .{
        flat_ids_global,
        Tensor.scalar(true, .bool).broad(flat_ids_global.shape()),
    };

    // Duplicate hidden rows once per routed expert and sort all routed
    // structures by local expert id so each expert becomes one contiguous
    // segment. Temporary permutation tensors use dedicated axis names, but the
    // tensors passed to grouped matmul are renamed back to the Triton-style
    // `token` axis.
    const hidden_repeated = hidden.repeat1d(.token, @intCast(ids.dim(.topk))).rename(.{ .token = .route });
    const sorted_route = flat_ids_local.argsort(.route, .{ .descending = false }).rename(.{ .route = .perm });
    const hidden_sorted = hidden_repeated.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const expert_ids_sorted = flat_ids_local.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const local_route_mask_sorted = local_route_mask.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const sorted_weights = flat_weights.gather(.{ .route = sorted_route }, .{}).rename(.{ .perm = .token });
    const router_weights_sorted = local_route_mask_sorted.select(
        sorted_weights,
        Tensor.zeroes(sorted_weights.shape()),
    );

    // Recover contiguous group sizes after sorting. This segment descriptor is
    // what drives the grouped matmul metadata builder.
    const group_sizes = histogramCounts(flat_ids_local, gate_up.dim(.expert));

    // First expert matmul: hidden x gate_up. Bias remains outside the custom
    // call and is applied per routed token in StableHLO.
    const use_reference_moe = true;
    var gate_up_out = if (use_reference_moe) blk: {
        const gate_up_per_token = gate_up.gather(.{ .expert = expert_ids_sorted }, .{});
        break :blk hidden_sorted.dot(gate_up_per_token, .in);
    } else blk: {
        const tile_m: i64 = 8;
        const aligned_hidden = alignSortedRowsByGroup(hidden_sorted, expert_ids_sorted, group_sizes, tile_m);
        break :blk callMegabloxGmm(aligned_hidden.rows, gate_up, aligned_hidden.group_sizes, hidden.dtype())
            .gather(.{ .token = aligned_hidden.positions.rename(.{ .token = .route }) }, .{})
            .rename(.{ .route = .token });
    };
    if (opts.w1_bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        gate_up_out = gate_up_out.add(bias_per_token);
    }

    // Apply the logical MoE activation between the two grouped matmuls.
    const activated = applyActivation(gate_up_out, opts.activation);

    // Second expert matmul: activated x down. As above, bias is handled in
    // StableHLO so the kernel only implements grouped GEMM.
    var down_out = if (use_reference_moe) blk: {
        const down_per_token = down.gather(.{ .expert = expert_ids_sorted }, .{});
        break :blk activated.rename(.{ .out = .mid }).dot(down_per_token, .mid);
    } else blk: {
        const tile_m: i64 = 8;
        const aligned_activated = alignSortedRowsByGroup(activated, expert_ids_sorted, group_sizes, tile_m);
        break :blk callMegabloxGmm(aligned_activated.rows, down, aligned_activated.group_sizes, hidden.dtype())
            .gather(.{ .token = aligned_activated.positions.rename(.{ .token = .route }) }, .{})
            .rename(.{ .route = .token });
    };
    if (opts.w2_bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        down_out = down_out.add(bias_per_token);
    }

    // Apply router weights after the second matmul. Non-local expert-parallel
    // routes have already been zero-masked here.
    const weighted = down_out.mul(router_weights_sorted.convert(down_out.dtype()).broad(down_out.shape()));

    // Undo the expert sort, reshape the routed copies back to [topk, token,
    // out], and sum across top-k to recover the final MoE output.
    const restore_indices = sorted_route.argsort(.perm, .{ .descending = false }).rename(.{ .perm = .restore });
    const unsorted = weighted.gather(.{ .token = restore_indices }, .{}).rename(.{ .restore = .route });
    const restored = unsorted.reshape(.{ .topk = ids.dim(.topk), .token = hidden.dim(.token), .out = down.dim(.out) });
    const combined = restored.sum(.topk).squeeze(.topk);
    return combined.reshape(.{ .b = b, .s = s, .d = down.dim(.out) });
}
