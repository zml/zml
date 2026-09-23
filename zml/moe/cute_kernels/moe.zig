//! SM100 CuTe MXFP8 x MXFP4 MoE, the Zig port of the CuTe DSL `NativeMoE`:
//! routing, routed MXFP8 input quantization, persistent up GEMM with a fused
//! SwiGLU + MXFP8 epilogue, persistent down GEMM storing rows in route order,
//! and the top-k reduction. Routing weights are applied before the down
//! projection, with the reference model's BF16 rounding.
const std = @import("std");

const zml = @import("../../zml.zig");
const triton_mxfp4 = @import("../triton_mxfp4.zig");
pub const boundary = @import("mxfp4.zig");
pub const persistent = @import("persistent_mxfp4.zig");

pub const Inputs = struct {
    /// `[tokens, hidden]` BF16.
    x: zml.Tensor,
    /// `[tokens, topk]` FP32.
    routing_weights: zml.Tensor,
    w1: zml.Tensor,
    /// `w1` scales in the 128x4 layout of `packWeightScales`.
    s1: zml.Tensor,
    w2: zml.Tensor,
    /// `w2` scales in the 128x4 layout of `packWeightScales`.
    s2: zml.Tensor,
    /// `[tokens, topk]` local expert ids.
    ids: zml.Tensor,
};

/// The kernels are tuned for the DeepSeek V4.1 expert geometry, shared by the
/// main MoE and the D-Spark drafter. Expert count and top-k are parameters.
pub fn isSupported(tokens: i64, hidden: i64, intermediate: i64) bool {
    return hidden == 5120 and intermediate == 2304 and tokens >= 1 and tokens <= 16384;
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

/// Upper bound on the number of `group`-row expert groups for all routes.
fn capacity(tokens: i64, topk: i64, experts: i64, group: i64) i64 {
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

fn route(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, n: i64, direct: bool, ids: zml.Tensor) Routing {
    const routes = tokens * topk;
    const flat_ids = ids.convert(.i32).reshape(.{routes});
    if (direct) return .{ .groups = routes, .route_map = null, .route_inverse = flat_ids, .schedule = flat_ids };
    const groups = capacity(tokens, topk, experts, n);
    const cfg: triton_mxfp4.kernels.Config = .{
        .tokens = tokens,
        .hidden = hidden,
        .intermediate = intermediate,
        .experts = experts,
        .global_experts = experts,
        .topk = topk,
        .swiglu_limit = 10,
    };
    const schedule = triton_mxfp4.kernels.scheduleForNative(cfg, flat_ids.reshape(.{ tokens, topk }), groups, n);
    return .{ .groups = groups, .route_map = schedule.route_map, .route_inverse = schedule.route_inverse, .schedule = schedule.schedule };
}

/// Pack E8M0 weight scales in the exact 128x4 swizzle consumed by the TMA
/// scale-factor descriptor. Loaders apply this once at load time.
pub fn packWeightScales(scales: zml.Tensor) zml.Tensor {
    const experts = scales.dim(0);
    const rows = scales.dim(1);
    const groups = scales.dim(2);
    const packed_scales = scales.reshape(.{ experts, @divExact(rows, 128), 4, 32, @divExact(groups, 4), 4 })
        .transpose(.{ 0, 1, 4, 3, 2, 5 });
    return withShapeOf(packed_scales, scales);
}

/// Inverse of `packWeightScales`: recover the linear `[E, rows, k/32]` layout.
pub fn unpackWeightScales(scales: zml.Tensor) zml.Tensor {
    const experts = scales.dim(0);
    const rows = scales.dim(1);
    const groups = scales.dim(2);
    const linear_scales = scales.reshape(.{ experts, @divExact(rows, 128), @divExact(groups, 4), 32, 4, 4 })
        .transpose(.{ 0, 1, 4, 3, 2, 5 });
    return withShapeOf(linear_scales, scales);
}

/// The swizzle only permutes scales within an expert, so the result keeps the
/// input tags and partitioning. `reshape` alone resets the partitioning, and
/// expert-parallel callers would then see unsharded scales.
fn withShapeOf(swizzled: zml.Tensor, scales: zml.Tensor) zml.Tensor {
    var result = swizzled.reshape(scales.shape());
    result._shape = scales.shape();
    return result;
}

/// `[tokens, hidden]` BF16 output of the MoE layer.
pub fn forward(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, a: Inputs) zml.Tensor {
    return forwardWithTile(tokens, hidden, intermediate, experts, topk, a, null);
}

/// `forward` with an explicit GEMM tile N, for tuning. `null` selects it from
/// the batch size.
pub fn forwardWithTile(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, a: Inputs, tile_n: ?i64) zml.Tensor {
    if (!isSupported(tokens, hidden, intermediate)) @panic("unsupported SM100 CuTe MoE shape");
    const n = tile_n orelse tileN(tokens, experts, topk);
    const direct = isDirect(tokens);
    const routing = route(tokens, hidden, intermediate, experts, topk, n, direct, a.ids);
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
