//! SM100 CuTe MXFP8 x MXFP4 MoE, the Zig port of the CuTe DSL `NativeMoE`:
//! routing, routed MXFP8 input quantization, persistent up GEMM with a fused
//! SwiGLU + MXFP8 epilogue, persistent down GEMM storing rows in route order,
//! and the top-k reduction. Routing weights are applied before the down
//! projection, with the reference model's BF16 rounding.
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
pub const boundary = @import("mxfp4.zig");
pub const persistent = @import("persistent_mxfp4.zig");
const triton_kernels = @import("../triton/moe.zig");

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

/// The boundary quantizer handles four 128-column tiles per CTA. The GEMMs
/// support 128-column tiles and a fixed set of reduction tile counts.
pub fn validateShapes(hidden: i64, intermediate: i64) void {
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
fn tileN(tokens: i64, global_experts: i64, topk: i64) i64 {
    // Direct routing keeps one row per group: the narrowest MMA wastes the
    // least of each B tile and leaves room for a 12-stage A/B ring.
    if (isDirect(tokens)) return 8;
    // Prefill: each group re-reads the whole expert weight matrix, so widen
    // the tile once experts hold enough routes to fill it. Use the global
    // count: the same estimate applies to unsharded weights and any EP mesh.
    // The thresholds are measured on the DeepSeek V4.1 expert geometry, with
    // the 64- and 128-row tiles on CTA pairs (`twoCta`): from 64 routes per
    // expert, reading each expert's weights once in half-empty 128-row
    // groups beats reading them twice in 64-row ones.
    const routes_per_expert = @divTrunc(tokens * topk, global_experts);
    if (routes_per_expert >= 64) return 128;
    if (routes_per_expert >= 32) return 64;
    if (routes_per_expert >= 16) return 32;
    return 16;
}

/// Whether the tile-`n` GEMMs run each 256xN MMA on a cluster of two CTAs
/// (`persistent.Config.two_cta`): the prefill tiles do, as long as both
/// GEMMs' output rows split into 256-row tiles. Each CTA then loads half of
/// a group's activations per weight tile, which leaves room for a deeper
/// load ring.
fn twoCta(n: i64, hidden: i64, intermediate: i64) bool {
    return n >= 64 and @rem(hidden, 256) == 0 and @rem(2 * intermediate, 256) == 0;
}

pub const LaunchPlan = struct {
    tile_n: i64,
    two_cta: bool,
    persistent_ctas: i64,
    quantize_lanes: i64,

    pub fn validate(self: LaunchPlan, platform: *const zml.Platform, tokens: i64, hidden: i64, intermediate: i64, global_experts: i64, topk: i64) !void {
        const tuning: Tuning = .{
            .tile_n = self.tile_n,
            .two_cta = self.two_cta,
            .persistent_ctas = self.persistent_ctas,
            .quantize_lanes = self.quantize_lanes,
        };
        const resolved = try tuning.resolveFor(platform, tokens, hidden, intermediate, global_experts, topk);
        if (!std.meta.eql(self, resolved)) return error.InvalidCuteLaunchPlan;
    }
};

/// Compile-time overrides for experiments. Workload selection belongs to the
/// benchmark; launch parameters belong to the backend that implements them.
pub const Tuning = struct {
    tile_n: ?i64 = null,
    two_cta: ?bool = null,
    persistent_ctas: ?i64 = null,
    quantize_lanes: ?i64 = null,

    pub fn resolveFor(self: Tuning, platform: *const zml.Platform, tokens: i64, hidden: i64, intermediate: i64, global_experts: i64, topk: i64) !LaunchPlan {
        return self.resolve(tokens, hidden, intermediate, global_experts, topk, try persistentCtas(platform));
    }

    pub fn resolve(self: Tuning, tokens: i64, hidden: i64, intermediate: i64, global_experts: i64, topk: i64, sm_count: i64) !LaunchPlan {
        const requested_n = self.tile_n orelse tileN(tokens, global_experts, topk);
        switch (requested_n) {
            8, 16, 32, 64, 128 => {},
            else => return error.InvalidCuteTileN,
        }
        // Direct decode has one live row per group and a dedicated narrow
        // epilogue. Tile overrides tune grouped routing only.
        const n = if (isDirect(tokens)) 8 else requested_n;
        const paired = self.two_cta orelse (!isDirect(tokens) and twoCta(n, hidden, intermediate));
        if (paired and (isDirect(tokens) or !twoCta(n, hidden, intermediate))) return error.InvalidCuteCtaPair;
        const ctas = self.persistent_ctas orelse (if (paired) sm_count - @mod(sm_count, 2) else sm_count);
        // Paired launches need complete clusters. More CTAs than SMs create
        // another wave of these shared-memory-heavy persistent kernels.
        if (ctas < (if (paired) @as(i64, 2) else 1) or ctas > sm_count or (paired and @mod(ctas, 2) != 0)) return error.InvalidCutePersistentCtas;
        const lanes = self.quantize_lanes orelse boundary.quantizeLanes(tokens);
        if (lanes != 1 and lanes != 4) return error.InvalidCuteQuantizeLanes;
        const threads = @divExact(hidden, @divExact(32, lanes));
        if ((lanes == 1 and threads > 1024) or (lanes == 4 and @mod(threads, 128) != 0)) return error.InvalidCuteQuantizeGeometry;
        return .{ .tile_n = n, .two_cta = paired, .persistent_ctas = ctas, .quantize_lanes = lanes };
    }
};

/// The same executable may run on every device in the mesh. Size its grid for
/// the smallest participating GPU instead of assuming a particular GB model.
fn persistentCtas(platform: *const zml.Platform) !i64 {
    if (platform.devices.len == 0) return error.MissingCudaCoreCount;
    var count: i64 = std.math.maxInt(i64);
    for (platform.devices) |device| {
        const attribute = device.pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return error.MissingCudaCoreCount;
        if (attribute != .int64 or attribute.int64 <= 0) return error.MissingCudaCoreCount;
        count = @min(count, attribute.int64);
    }
    return count;
}

pub fn launchPlan(platform: *const zml.Platform, tokens: i64, hidden: i64, intermediate: i64, global_experts: i64, topk: i64) !LaunchPlan {
    return (Tuning{}).resolveFor(platform, tokens, hidden, intermediate, global_experts, topk);
}

test "CuTe launch policy uses global experts and validates tuning overrides" {
    const defaults: Tuning = .{};
    const sparse = try defaults.resolve(512, 5120, 2304, 384, 6, 152);
    try std.testing.expectEqual(@as(i64, 16), sparse.tile_n);
    try std.testing.expectEqual(@as(i64, 4), sparse.quantize_lanes);
    const dense = try defaults.resolve(256, 5120, 2304, 8, 6, 152);
    try std.testing.expectEqual(@as(i64, 128), dense.tile_n);
    const ep2 = try defaults.resolve(2048, 5120, 2304, 384, 6, 152);
    try std.testing.expectEqual(@as(i64, 64), ep2.tile_n);
    try std.testing.expect(ep2.two_cta);
    try std.testing.expectEqual(@as(i64, 4), ep2.quantize_lanes);
    const prefill = try defaults.resolve(4096, 5120, 2304, 384, 6, 152);
    try std.testing.expectEqual(@as(i64, 1), prefill.quantize_lanes);
    const unsharded = try defaults.resolve(2048, 5120, 2304, 192, 6, 148);
    try std.testing.expectEqual(@as(i64, 128), unsharded.tile_n);
    const odd = try defaults.resolve(2048, 5120, 2304, 384, 6, 147);
    try std.testing.expectEqual(@as(i64, 146), odd.persistent_ctas);
    const direct = try defaults.resolve(1, 5120, 2304, 384, 6, 132);
    try std.testing.expectEqual(@as(i64, 8), direct.tile_n);
    try std.testing.expect(!direct.two_cta);
    inline for (.{
        .{ Tuning{ .tile_n = 7 }, error.InvalidCuteTileN },
        .{ Tuning{ .tile_n = 32, .two_cta = true }, error.InvalidCuteCtaPair },
        .{ Tuning{ .persistent_ctas = 153 }, error.InvalidCutePersistentCtas },
        .{ Tuning{ .persistent_ctas = 151 }, error.InvalidCutePersistentCtas },
        .{ Tuning{ .quantize_lanes = 2 }, error.InvalidCuteQuantizeLanes },
    }) |case| try std.testing.expectError(case[1], case[0].resolve(2048, 5120, 2304, 384, 6, 152));
    try std.testing.expectError(error.InvalidCuteCtaPair, (Tuning{ .two_cta = true }).resolve(1, 5120, 2304, 384, 6, 152));
}

test "CuTe explicit plans validate device limits and direct routing" {
    const platform = zml.testing.env();
    if (!@import("cute.zig").isAvailable(platform)) return error.SkipZigTest;
    const plan = try (Tuning{ .tile_n = 64, .two_cta = false, .quantize_lanes = 1 }).resolveFor(platform, 17, 5120, 2304, 8, 4);
    try plan.validate(platform, 17, 5120, 2304, 8, 4);
    try std.testing.expectEqual(@as(i64, 64), plan.tile_n);
    try std.testing.expectEqual(@as(i64, 1), plan.quantize_lanes);
    try std.testing.expect(!plan.two_cta);
    var invalid = plan;
    invalid.persistent_ctas += 1;
    try std.testing.expectError(error.InvalidCutePersistentCtas, invalid.validate(platform, 17, 5120, 2304, 8, 4));
    try std.testing.expectError(error.InvalidCuteLaunchPlan, plan.validate(platform, 1, 5120, 2304, 8, 4));
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

/// Low-level entry point for a resolved plan; the public expert wrapper validates it.
pub fn forward(tokens: i64, hidden: i64, intermediate: i64, experts: i64, topk: i64, swiglu_limit: ?f32, a: Inputs, plan: LaunchPlan) zml.Tensor {
    const n = plan.tile_n;
    const direct = isDirect(tokens);
    const routing = route(tokens, experts, topk, n, direct, a.ids);
    const routes = tokens * topk;

    const up_cfg: persistent.Config = .{
        .experts = experts,
        .m = 2 * intermediate,
        .n = n,
        .k = hidden,
        .groups = routing.groups,
        .persistent_ctas = plan.persistent_ctas,
        .epilogue = .swiglu_mxfp8,
        .routes = routes,
        .direct = direct,
        .swiglu_limit = swiglu_limit,
        .two_cta = plan.two_cta,
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
        .lanes_per_block = plan.quantize_lanes,
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
    if (!@import("cute.zig").isAvailable(platform)) return error.SkipZigTest;
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
            var outputs: [2]zml.Tensor = undefined;
            for ([_]bool{ true, false }, &outputs) |paired, *output| {
                const tuning: Tuning = .{ .tile_n = tile, .two_cta = paired };
                const plan = tuning.resolveFor(zml.Compiler.current().platform, tokens, hidden, intermediate, experts, topk) catch unreachable;
                output.* = forward(tokens, hidden, intermediate, experts, topk, 10.0, inputs, plan);
            }
            return outputs;
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
