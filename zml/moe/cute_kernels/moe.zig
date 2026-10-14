//! SM100 CuTe MXFP8 x MXFP4 MoE, the Zig port of the CuTe DSL `NativeMoE`:
//! routing, routed MXFP8 input quantization, persistent up GEMM with a fused
//! SwiGLU + MXFP8 epilogue, persistent down GEMM storing rows in route order,
//! and the top-k reduction. Routing weights are applied before the down
//! projection, with the reference model's BF16 rounding.
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../../zml.zig");
pub const boundary = @import("mxfp4.zig");
pub const persistent = @import("persistent_mxfp4.zig");
const triton_kernels = @import("../triton/triton.zig");

pub const Inputs = struct {
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

/// The kernels are tuned for the DeepSeek V4.1 expert geometry, shared by the
/// main MoE and the D-Spark drafter. Expert count and top-k are parameters.
pub fn validateShapes(hidden: i64, intermediate: i64) void {
    stdx.debug.assert(hidden == 5120, "cute backend expects hidden dimension to be 5120, got {}", .{hidden});
    stdx.debug.assert(intermediate == 2304, "cute backend expects intermediate dimension to be 2304, got {}", .{intermediate});
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

/// `[tokens, hidden]` BF16 output of the MoE layer.
pub fn forward(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, swiglu_limit: f32, a: Inputs) zml.Tensor {
    return forwardWithTile(tokens, hidden, intermediate, experts, topk, swiglu_limit, a, null);
}

/// `forward` with an explicit GEMM tile N, for tuning. `null` selects it from
/// the batch size.
pub fn forwardWithTile(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, swiglu_limit: f32, a: Inputs, tile_n: ?i64) zml.Tensor {
    const n = tile_n orelse tileN(tokens, experts, topk);
    const direct = isDirect(tokens);
    const routing = route(tokens, experts, topk, n, direct, a.ids);
    const routes = tokens * topk;

    const up_cfg: persistent.Config = .{
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

    const q1 = boundary.groupedQuantize(a.x, routing.route_map orelse routing.schedule, .{
        .tokens = tokens,
        .hidden = hidden,
        .topk = topk,
        .group_size = up_cfg.groupRows(),
        .capacity = routing.groups,
        .direct = direct,
    });
    const q2 = persistent.upQuantized(up_cfg, .{
        .weight = a.w1,
        .schedule = routing.schedule,
        .input_quant = q1.q,
        .weight_scale = a.s1.bitCast(.u8),
        .input_scale = q1.s,
    }, routing.route_inverse, a.routing_weights.reshape(.{routes}));
    const down = persistent.downRows(down_cfg, .{
        .weight = a.w2,
        .schedule = routing.schedule,
        .input_quant = q2.q,
        .weight_scale = a.s2.bitCast(.u8),
        .input_scale = q2.s,
    }, routing.route_inverse);
    return boundary.combine(down, routing.route_map orelse routing.schedule, .{
        .tokens = tokens,
        .hidden = hidden,
        .topk = topk,
        .columns_per_cta = boundary.combineColumns(tokens, hidden),
    });
}

test {
    _ = boundary;
    _ = persistent;
}
