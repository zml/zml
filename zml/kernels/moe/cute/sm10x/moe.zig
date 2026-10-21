//! SM100 CuTe MoE, MXFP8 x MXFP4 (the Zig port of the CuTe DSL `NativeMoE`)
//! and NVFP4 x NVFP4: routing, routed input quantization, persistent up GEMM
//! with a fused SwiGLU + quantization epilogue, persistent down GEMM storing
//! rows in route order, and the top-k reduction. MXFP4 applies routing weights
//! before the down projection, with the reference model's BF16 rounding.
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../../../../zml.zig");
pub const mxfp4 = @import("mxfp4.zig");
pub const mxfp4_persistent = @import("persistent_mxfp4.zig");
pub const nvfp4 = @import("nvfp4.zig");
pub const nvfp4_persistent = @import("persistent_nvfp4.zig");
const reduce = @import("reduce.zig");
const triton_kernels = @import("../../triton/moe.zig");

pub const Mxfp4Inputs = struct {
    /// `[tokens, hidden]` BF16.
    x: zml.Tensor,
    /// `[tokens, topk]` FP32.
    routing_weights: zml.Tensor,
    w1: zml.Tensor,
    /// `w1` scales in the 128x4 tensor-core layout (rows tiled by 4x32, columns by 4).
    s1: zml.Tensor,
    w2: zml.Tensor,
    /// `w2` scales in the 128x4 tensor-core layout (rows tiled by 4x32, columns by 4).
    s2: zml.Tensor,
    /// `[tokens, topk]` local expert ids.
    ids: zml.Tensor,
};

/// The boundary quantizer handles four 128-column tiles per CTA. The GEMMs
/// support 128-column tiles and a fixed set of reduction tile counts.
pub fn validateMxfp4Shapes(hidden: i64, intermediate: i64) void {
    stdx.debug.assert(hidden > 0 and @rem(hidden, 4 * 128) == 0, "cute backend requires hidden dimension to be a positive multiple of 512, got {}", .{hidden});
    stdx.debug.assert(intermediate > 0 and @rem(intermediate, 128) == 0, "cute backend requires intermediate dimension to be a positive multiple of 128, got {}", .{intermediate});
    stdx.debug.assert(supportedReductionTiles(@divTrunc(hidden, 128)), "cute backend does not support hidden reduction tile count {}, got hidden dimension {}", .{ @divTrunc(hidden, 128), hidden });
    stdx.debug.assert(supportedReductionTiles(@divTrunc(intermediate, 128)), "cute backend does not support intermediate reduction tile count {}, got intermediate dimension {}", .{ @divTrunc(intermediate, 128), intermediate });
}

fn supportedReductionTiles(tiles: i64) bool {
    return switch (tiles) {
        16, 18, 32, 40 => true,
        else => false,
    };
}

/// Route each token/top-k pair to its own one-row group instead of grouping
/// routes by expert. A token's top-k experts are distinct, so for a single
/// token grouping cannot share any weight tile and only adds the routing
/// kernels. Python `grouped=False`.
fn isDirect(tokens: i64) bool {
    return tokens == 1;
}

/// GEMM tile N, which is also the routed expert-group size. Wider tiles
/// reuse each weight tile across more routes but pad more rows per expert,
/// so they only pay off once experts receive tens of routes.
fn tileN(tokens: i64, experts: i64, topk: i64) i64 {
    // Direct routing keeps one row per group: the narrowest MMA wastes the
    // least of each B tile and leaves room for a 12-stage A/B ring.
    if (isDirect(tokens)) return 8;
    if (tokens < 512) return 16;
    // Prefill: each group re-reads the whole expert weight matrix, so widen
    // the tile once experts hold enough routes to fill it. Half the routes
    // are remote under expert parallelism, hence the conservative estimate.
    // The thresholds are measured on the DeepSeek V4.1 expert geometry.
    const routes_per_expert = @divTrunc(tokens * topk, 2 * experts);
    if (routes_per_expert >= 128) return 128;
    if (routes_per_expert >= 32) return 64;
    return 32;
}

/// One persistent CTA per SM. Each CTA needs the full 227 KiB of shared
/// memory, so launching more CTAs than SMs (e.g. 160 on a 152-SM GB300)
/// runs the surplus as a second wave and nearly doubles the GEMM time.
fn persistentCtas() i64 {
    const platform = zml.Compiler.current().platform;
    if (platform.devices.len == 0) return 148;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 148;
    return if (attribute.int64 > 0) attribute.int64 else 148;
}

// TODO(Corentin): settle on groups/capacity.

/// Upper bound on the number of `group`-row expert groups for all routes.
fn computeCapacity(tokens: i64, topk: i64, experts: i64, group: i64) i64 {
    return @divTrunc(tokens * topk + group - 1, group) + experts - 1;
}

/// Where each route lives in the grouped GEMM operands.
const Routing = struct {
    /// Expert groups, i.e. the GEMMs' L extent.
    groups: i64,
    /// Route -> physical routed row. Null for direct routing, where route
    /// `r` is row `r`.
    route_map: ?zml.Tensor,
    /// Physical routed row -> route. Direct routing does not read it.
    route_inverse: zml.Tensor,
    /// `[group experts | group sizes | active group count]` for grouped
    /// routing, `[route experts]` for direct routing.
    schedule: zml.Tensor,
};

const atomic_route_threshold = 6 * 2048;

fn route(tokens: i64, experts: i64, topk: i64, n: i64, direct: bool, ids: zml.Tensor) Routing {
    const routes = tokens * topk;
    const flat_ids = ids.convert(.i32).reshape(.{routes});
    if (direct) return .{ .groups = routes, .route_map = null, .route_inverse = flat_ids, .schedule = flat_ids };
    const capacity = computeCapacity(tokens, topk, experts, n);

    std.debug.assert(capacity > 0);
    // `route` rescans every route once per expert: cheapest for decode, but
    // quadratic. Past a few thousand tokens, count with atomics instead.
    const r: struct { counts: zml.Tensor, pos: zml.Tensor } = if (routes >= atomic_route_threshold) blk: {
        const counted = triton_kernels.RouteAtomic.Kernel.call(
            .{ .ids = ids, .zeros = zml.Tensor.zeroes(.init(.{experts}, .i32)) },
            .{ .counts = .init(.{experts}, .i32), .pos = .init(.{routes}, .i32) },
            .{
                .cfg = .{
                    .tokens = tokens,
                    .topk = topk,
                    .expert_offset = 0,
                    .experts = experts,
                },
                .grid = .{ @intCast(@divTrunc(routes + triton_kernels.RouteAtomic.route_block - 1, triton_kernels.RouteAtomic.route_block)), 1, 1 },
                .num_warps = 4,
                .num_stages = 1,
                .output_operand_aliases = .{ .counts = .zeros },
            },
        );
        break :blk .{ .counts = counted.counts, .pos = counted.pos };
    } else blk: {
        const scanned = triton_kernels.Route.Kernel.call(
            .{ .ids = ids },
            .{ .counts = .init(.{experts}, .i32), .pos = .init(.{routes}, .i32) },
            .{
                .cfg = .{ .tokens = tokens, .topk = topk, .expert_offset = 0 },
                .grid = .{ @intCast(experts), 1, 1 },
                .num_warps = 4,
                .num_stages = 1,
            },
        );
        break :blk .{ .counts = scanned.counts, .pos = scanned.pos };
    };

    const s = triton_kernels.Schedule.Kernel.call(
        .{ .ids = ids, .counts = r.counts, .pos = r.pos },
        .{
            .map = .init(.{capacity * n}, .i32),
            .perm = .init(.{tokens * topk}, .i32),
            .sched = .init(.{2 * capacity + 1}, .i32),
        },
        .{
            .cfg = .{
                .tokens = tokens,
                .topk = topk,
                .expert_offset = 0,
                .experts = experts,
                .capacity = capacity,
                .group = n,
            },
            .grid = .{ @intCast(@divTrunc(tokens * topk + 127, 128)), 1, 1 },
            .num_warps = 4,
            .num_stages = 1,
        },
    );

    return .{ .groups = capacity, .route_map = s.perm, .route_inverse = s.map, .schedule = s.sched };
}

/// `[tokens, hidden]` BF16 output of the MXFP4 MoE layer. `tile_n` is the
/// GEMM tile N, for tuning; null selects it from the batch size.
pub fn forwardMxfp4(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, swiglu_limit: ?f32, a: Mxfp4Inputs, tile_n: ?i64) zml.Tensor {
    const n = tile_n orelse tileN(tokens, experts, topk);
    const direct = isDirect(tokens);
    const routing = route(tokens, experts, topk, n, direct, a.ids);
    const routes = tokens * topk;

    const up_cfg: mxfp4_persistent.Config = .{
        .experts = experts,
        .m = 2 * intermediate,
        .n = n,
        .k = hidden,
        .groups = routing.groups,
        .persistent_ctas = persistentCtas(),
        .epilogue = .swiglu_mxfp8,
        .routes = routes,
        .direct = direct,
        .swiglu_limit = swiglu_limit,
    };
    var down_cfg = up_cfg;
    down_cfg.m = hidden;
    down_cfg.k = intermediate;
    down_cfg.epilogue = .route_rows;

    const q1 = mxfp4.groupedQuantize(a.x, routing.route_map orelse routing.schedule, .{
        .tokens = tokens,
        .hidden = hidden,
        .topk = topk,
        .group_size = up_cfg.groupRows(),
        .capacity = routing.groups,
        .direct = direct,
    });
    const q2 = mxfp4_persistent.upQuantized(up_cfg, .{
        .weight = a.w1,
        .schedule = routing.schedule,
        .input_quant = q1.q,
        .weight_scale = a.s1.bitCast(.u8),
        .input_scale = q1.s,
    }, routing.route_inverse, a.routing_weights.reshape(.{routes}));
    const down = mxfp4_persistent.downRows(down_cfg, .{
        .weight = a.w2,
        .schedule = routing.schedule,
        .input_quant = q2.q,
        .weight_scale = a.s2.bitCast(.u8),
        .input_scale = q2.s,
    }, routing.route_inverse);
    return reduce.topk(down, routing.route_map orelse routing.schedule, .{
        .tokens = tokens,
        .hidden = hidden,
        .topk = topk,
        .columns_per_cta = reduce.columnsPerCta(tokens, hidden),
    });
}

pub const Nvfp4Inputs = struct {
    /// `[tokens, hidden]` BF16.
    x: zml.Tensor,
    /// `[tokens, topk]` FP32.
    routing_weights: zml.Tensor,
    /// `[tokens, topk]` local expert ids, -1 for routes of other ranks.
    ids: zml.Tensor,
    /// `[experts, 2 * intermediate, hidden / 2]` packed FP4, gate/up rows interleaved.
    w1: zml.Tensor,
    /// `w1` E4M3 scales `[experts, 2 * intermediate, hidden / 16]` in the 128x4 layout.
    s1: zml.Tensor,
    /// `[experts, hidden, intermediate / 2]` packed FP4.
    w2: zml.Tensor,
    /// `w2` E4M3 scales `[experts, hidden, intermediate / 16]` in the 128x4 layout.
    s2: zml.Tensor,
    /// `[experts]` FP32 dequantization scale of each GEMM output, which folds
    /// in the inverse of the GEMM input's global scale.
    alpha1: zml.Tensor,
    alpha2: zml.Tensor,
    /// `[experts]` FP32 global scale each GEMM's input is multiplied by
    /// before NVFP4 quantization.
    act1: zml.Tensor,
    act2: zml.Tensor,
};

pub const Nvfp4Options = struct {
    swiglu_limit: ?f32 = null,
    routing_weight_placement: zml.moe.RoutingWeightPlacement = .after_down,
    /// GEMM tile N; null selects it from the batch size.
    tile_n: ?i64 = null,
};

pub fn validateNvfp4Shapes(hidden: i64, intermediate: i64) void {
    const k_tile = nvfp4_persistent.k_tile;
    stdx.debug.assert(@rem(hidden, k_tile) == 0, "cute NVFP4 backend expects hidden to be a multiple of {}, got {}", .{ k_tile, hidden });
    stdx.debug.assert(@rem(intermediate, k_tile) == 0, "cute NVFP4 backend expects intermediate to be a multiple of {}, got {}", .{ k_tile, intermediate });
}

/// NVFP4 GEMM tile N: the expected routes of one expert, so that each
/// expert's weight tiles are read once per group instead of once per few rows.
fn nvfp4TileN(tokens: i64, experts: i64, topk: i64) i64 {
    if (isDirect(tokens)) return 8;
    const per_expert = std.math.divCeil(i64, tokens * topk, experts) catch unreachable;
    return @intCast(std.math.clamp(std.math.ceilPowerOfTwoAssert(u64, @intCast(per_expert)), 16, 128));
}

/// `[tokens, hidden]` BF16 output of the NVFP4 MoE layer.
pub fn forwardNvfp4(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, a: Nvfp4Inputs, opts: Nvfp4Options) zml.Tensor {
    validateNvfp4Shapes(hidden, intermediate);
    const n = opts.tile_n orelse nvfp4TileN(tokens, experts, topk);
    const direct = isDirect(tokens);
    const routes = tokens * topk;
    const flat_ids = a.ids.convert(.i32).reshape(.{routes});
    const routing = route(tokens, experts, topk, n, direct, a.ids);
    const route_map = routing.route_map orelse flat_ids;
    const weights = a.routing_weights.convert(.f32).reshape(.{routes});

    const up_cfg: nvfp4_persistent.Config = .{
        .experts = experts,
        .m = 2 * intermediate,
        .n = n,
        .k = hidden,
        .groups = routing.groups,
        .persistent_ctas = persistentCtas(),
        .epilogue = .swiglu_nvfp4,
        .routes = routes,
        .direct = direct,
        .swiglu_limit = opts.swiglu_limit,
        .routing_weights = opts.routing_weight_placement == .before_down,
    };
    var down_cfg = up_cfg;
    down_cfg.m = hidden;
    down_cfg.k = intermediate;
    down_cfg.epilogue = .route_rows;
    down_cfg.routing_weights = opts.routing_weight_placement == .after_down;

    const q1 = nvfp4.groupedQuantize(a.x, route_map, flat_ids, a.act1, .{
        .tokens = tokens,
        .hidden = hidden,
        .topk = topk,
        .experts = experts,
        .group_size = up_cfg.groupRows(),
        .capacity = routing.groups,
        .direct = direct,
    });
    const q2 = nvfp4_persistent.upQuantized(up_cfg, .{
        .weight = a.w1,
        .schedule = routing.schedule,
        .input_quant = q1.q,
        .weight_scale = a.s1.bitCast(.u8),
        .input_scale = q1.s,
        .alpha = a.alpha1,
    }, routing.route_inverse, weights, a.act2);
    const down = nvfp4_persistent.downRows(down_cfg, .{
        .weight = a.w2,
        .schedule = routing.schedule,
        .input_quant = q2.q,
        .weight_scale = a.s2.bitCast(.u8),
        .input_scale = q2.s,
        .alpha = a.alpha2,
    }, routing.route_inverse, weights);
    return reduce.topk(down, route_map, .{
        .tokens = tokens,
        .hidden = hidden,
        .topk = topk,
        .columns_per_cta = reduce.columnsPerCta(tokens, hidden),
    });
}

test {
    _ = mxfp4;
    _ = mxfp4_persistent;
    _ = nvfp4;
    _ = nvfp4_persistent;
    _ = reduce;
}
