const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const common = @import("common.zig");
const megablox_gmm = @import("mosaic_tpu_kernels/megablox_gmm.zig");

pub const Options = struct {
    activation: common.ActivationMode = .silu,
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

fn repeatIdsFromCounts(counts: Tensor, total_len: i64, axis_name: anytype, max_id: i32) Tensor {
    const count_ends = counts.cumulativeSum(0).rename(.{ .expert = .bucket });
    const positions = Tensor.arange(.{ .end = total_len }, .i32).withTags(.{axis_name});
    const positions_2d = positions.insertAxes(.last, .{.bucket});
    const count_ends_2d = count_ends.insertAxes(.first, .{axis_name});
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
        .add(Tensor.constant(.{ .i32 = 1 }).broad(.init(.{ .expert = tiles_m }, .i32)))
        .rename(.{ .expert = .tile });
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
    const gate, const up = zml.nn.splitRealImg(x, .sequential);
    return switch (mode) {
        .silu => gate.silu().mul(up),
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
    const rem = @mod(x.dim(0), tile_m);
    if (rem == 0) return x;
    return x.pad(0, .{ .token = Tensor.Pad{ .high = tile_m - rem } });
}

fn gmmDType(dtype: zml.DataType) zml.kernel.mosaic_tpu.DType {
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
    const tile_m: i64 = 128;
    const metadata = buildGroupMetadata(group_sizes.withTags(.{.expert}), lhs.dim(.token), tile_m);
    const cfg: megablox_gmm.Cfg = .{
        .m = @intCast(lhs.dim(.token)),
        .k = @intCast(lhs.dim(1)),
        .n = @intCast(rhs.dim(.out)),
        .num_groups = @intCast(rhs.dim(.expert)),
        .num_active_tiles = @intCast(metadata.num_active_tiles.dim(0)),
        .tm = 128,
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
            .out = Shape.init(.{ .token = lhs.dim(.token), .out = rhs.dim(.out) }, out_dtype),
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

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);

    const hidden = hidden_states.reshape(.{ .token = b * s, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = w1.withTags(.{ .expert, .out, .in });
    const down = w2.withTags(.{ .expert, .out, .mid });
    const weights = topk_weights.reshape(.{ .token = b * s, .topk = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = b * s, .topk = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    try validateInputs(hidden, gate_up, down, weights, ids);

    const routed_tokens = hidden.dim(.token) * ids.dim(.topk);
    const flat_ids = ids.transpose(.{ .topk, .token }).flatten().withTags(.{.route});
    const flat_weights = weights.transpose(.{ .topk, .token }).flatten().withTags(.{.route});

    // StableHLO permutation step: duplicate each token once per top-k expert and
    // then sort those routed copies by expert id so the grouped matmul sees one
    // contiguous segment per expert.
    const hidden_repeated = hidden.repeat1d(.token, @intCast(ids.dim(.topk))).rename(.{ .token = .route });
    const sorted_route = flat_ids.argsort(.route, .{ .descending = false });
    const hidden_sorted = hidden_repeated.gather(.{ .route = sorted_route }, .{});
    const expert_ids_sorted = flat_ids.gather(.{ .route = sorted_route }, .{});
    const router_weights_sorted = flat_weights.gather(.{ .route = sorted_route }, .{});

    // StableHLO histogram step: scatter-add one per routed token to recover the
    // per-expert group sizes that the Mosaic grouped matmul expects.
    const group_sizes = histogramCounts(flat_ids, gate_up.dim(.expert));
    const hidden_sorted_padded = padRowsToTile(hidden_sorted, 128);

    // The custom call only handles the grouped matmul itself. All routing,
    // metadata construction, biasing, activation, and unpermutation stay in
    // StableHLO so the MoE backend remains easy to inspect and evolve.
    var gate_up_out = callMegabloxGmm(hidden_sorted_padded, gate_up, group_sizes, hidden.dtype())
        .slice1d(.token, .{ .end = routed_tokens });
    if (opts.w1_bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        gate_up_out = gate_up_out.add(bias_per_token);
    }

    const activated = applyActivation(gate_up_out, opts.activation);
    const activated_padded = padRowsToTile(activated, 128);
    var down_out = callMegabloxGmm(activated_padded, down, group_sizes, hidden.dtype())
        .slice1d(.token, .{ .end = routed_tokens });
    if (opts.w2_bias) |bias| {
        const bias_per_token = bias.withTags(.{ .expert, .out }).gather(.{ .expert = expert_ids_sorted }, .{});
        down_out = down_out.add(bias_per_token);
    }

    const weighted = down_out.mul(router_weights_sorted.convert(down_out.dtype()).broad(down_out.shape()));

    // StableHLO unpermutation step: invert the expert sort, reshape the routed
    // copies back to [topk, token, out], and reduce across top-k to combine the
    // expert contributions for each original token.
    const restore_indices = sorted_route.argsort(.route, .{ .descending = false });
    const unsorted = weighted.gather(.{ .route = restore_indices }, .{});
    const restored = unsorted.reshape(.{ .topk = ids.dim(.topk), .token = hidden.dim(.token), .out = down.dim(.out) });
    const combined = restored.sum(.topk).squeeze(.topk);
    return combined.reshape(.{ .b = b, .s = s, .d = down.dim(.out) });
}
