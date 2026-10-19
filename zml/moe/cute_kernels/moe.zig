//! SM100 CuTe MXFP8 x MXFP4 MoE, the Zig port of the CuTe DSL `NativeMoE`:
//! routing, routed MXFP8 input quantization, persistent up GEMM with a fused
//! SwiGLU + MXFP8 epilogue, persistent down GEMM storing rows in route order,
//! and the top-k reduction. Routing weights are applied before the down
//! projection, with the reference model's BF16 rounding.
const std = @import("std");
const stdx = @import("stdx");

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
    stdx.debug.assert(hidden == 5120, "cute_mxfp4 backend expects hidden dimension to be 5120, got {}", .{hidden});
    stdx.debug.assert(intermediate == 2304, "cute_mxfp4 backend expects intermediate dimension to be 2304, got {}", .{intermediate});
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
    if (routes_per_expert >= 64) return 128;
    if (routes_per_expert >= 32) return 64;
    return 32;
}

/// One persistent CTA per SM. Each CTA needs the full 227 KiB of shared
/// memory, so launching more CTAs than SMs (e.g. 160 on a 152-SM GB300)
/// runs the surplus as a second wave and nearly doubles the GEMM time.
fn persistentCtas(paired: bool) i64 {
    const platform = zml.Compiler.current().platform;
    stdx.debug.assert(platform.devices.len > 0, "CuTe MoE requires a CUDA device", .{});
    var count: i64 = std.math.maxInt(i64);
    for (platform.devices) |device| {
        const attribute = device.pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse @panic("missing CUDA SM count");
        stdx.debug.assert(attribute == .int64 and attribute.int64 > 0, "invalid CUDA SM count", .{});
        count = @min(count, attribute.int64);
    }
    return if (paired) count - @mod(count, 2) else count;
}

fn twoCta(n: i64, hidden: i64, intermediate: i64) bool {
    return n >= 64 and @rem(hidden, 256) == 0 and @rem(2 * intermediate, 256) == 0;
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
        // Unused: only the routing and scheduling kernels of that config run.
        .swiglu_limit = 0,
    };
    const schedule = triton_mxfp4.kernels.scheduleForCute(cfg, flat_ids.reshape(.{ tokens, topk }), groups, n);
    return .{ .groups = groups, .route_map = schedule.route_map, .route_inverse = schedule.route_inverse, .schedule = schedule.schedule };
}

/// `[tokens, hidden]` BF16 output of the MoE layer.
pub fn forward(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, swiglu_limit: f32, a: Inputs) zml.Tensor {
    return forwardWithTile(tokens, hidden, intermediate, experts, topk, swiglu_limit, a, null, null);
}

/// `forward` with an explicit GEMM tile N, for tuning. `null` selects it from
/// the batch size.
fn forwardWithTile(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, swiglu_limit: f32, a: Inputs, tile_n: ?i64, two_cta: ?bool) zml.Tensor {
    const direct = isDirect(tokens);
    const n = if (direct) 8 else tile_n orelse tileN(tokens, experts, topk);
    const paired = two_cta orelse (!direct and twoCta(n, hidden, intermediate));
    stdx.debug.assert(!paired or (!direct and twoCta(n, hidden, intermediate)), "invalid CuTe CTA pair geometry", .{});
    const routing = route(tokens, hidden, intermediate, experts, topk, n, direct, a.ids);
    const routes = tokens * topk;

    const up_cfg: persistent.Config = .{
        .experts = experts,
        .m = 2 * intermediate,
        .n = n,
        .k = hidden,
        .groups = routing.groups,
        .persistent_ctas = persistentCtas(paired),
        .epilogue = .swiglu_mxfp8,
        .routes = routes,
        .direct = direct,
        .swiglu_limit = swiglu_limit,
        .two_cta = paired,
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
        .lanes_per_block = boundary.quantizeLanes(tokens),
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
    });
}

test {
    _ = boundary;
    _ = persistent;
}

test "CuTe CTA-pair GEMMs match single-CTA GEMMs bit for bit" {
    const platform = zml.testing.env();
    if (!@import("../cute_mxfp4.zig").isAvailable(platform)) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    // A few experts with uneven route counts, some routes owned by another
    // expert-parallel rank (-1), and enough groups per expert to run several
    // persistent tiles per cluster on both accumulator stages.
    const tokens = 384;
    const topk = 4;
    const experts = 4;
    const hidden = 5120;
    const intermediate = 2304;
    const Fns = struct {
        fn both(x: zml.Tensor, ids: zml.Tensor, weights: zml.Tensor, w1: zml.Tensor, s1: zml.Tensor, w2: zml.Tensor, s2: zml.Tensor, tile: i64) [2]zml.Tensor {
            const inputs: Inputs = .{ .x = x, .routing_weights = weights, .w1 = w1, .s1 = s1, .w2 = w2, .s2 = s2, .ids = ids };
            return .{
                forwardWithTile(tokens, hidden, intermediate, experts, topk, 10.0, inputs, tile, true),
                forwardWithTile(tokens, hidden, intermediate, experts, topk, 10.0, inputs, tile, false),
            };
        }
    };
    const shapes = [_]zml.Shape{
        .init(.{ tokens, hidden }, .bf16),
        .init(.{ tokens, topk }, .i32),
        .init(.{ tokens, topk }, .f32),
        .init(.{ experts, 2 * intermediate, hidden / 2 }, .u8),
        .init(.{ experts, 2 * intermediate, hidden / 32 }, .u8),
        .init(.{ experts, hidden, intermediate / 2 }, .u8),
        .init(.{ experts, hidden, intermediate / 32 }, .u8),
    };
    var rng = std.Random.DefaultPrng.init(17);
    const random = rng.random();
    var buffers: [shapes.len]zml.Buffer = undefined;
    for (&buffers, shapes, 0..) |*buffer, shape, i| {
        const slice = try zml.Slice.alloc(allocator, shape);
        defer slice.free(allocator);
        switch (i) {
            0 => for (slice.items(zml.floats.BFloat16)) |*v| {
                v.* = .fromF32(random.float(f32) * 2 - 1);
            },
            1 => for (slice.items(i32), 0..) |*id, r| {
                // Distinct experts per token; one in five routes is remote.
                const expert: i32 = @intCast((r / topk + r % topk) % (experts + 1));
                id.* = if (expert == experts) -1 else expert;
            },
            2 => for (slice.items(f32)) |*w| {
                w.* = 0.1 + 0.3 * random.float(f32);
            },
            // FP4 weights, and E8M0 scales of 2^-9 .. 2^-6.
            3, 5 => random.bytes(slice.data()),
            else => for (slice.data()) |*b| {
                b.* = 118 + random.uintLessThan(u8, 4);
            },
        }
        buffer.* = try .fromSlice(io, platform, slice);
    }
    defer for (&buffers) |*b| b.deinit();

    inline for (.{ 64, 128 }) |tile| {
        var exe = try platform.compileFn(allocator, io, Fns.both, .{
            zml.Tensor.fromShape(shapes[0]), zml.Tensor.fromShape(shapes[1]), zml.Tensor.fromShape(shapes[2]),
            zml.Tensor.fromShape(shapes[3]), zml.Tensor.fromShape(shapes[4]), zml.Tensor.fromShape(shapes[5]),
            zml.Tensor.fromShape(shapes[6]), @as(i64, tile),
        }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var results: [2]zml.Buffer = undefined;
        runner.run(io, .{ buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6] }, &results, .{ .wait = true });
        defer for (&results) |*r| r.deinit();
        const pair = try results[0].toSliceAlloc(allocator, io);
        defer pair.free(allocator);
        const single = try results[1].toSliceAlloc(allocator, io);
        defer single.free(allocator);
        try std.testing.expectEqualSlices(u8, single.constData(), pair.constData());
    }
}
