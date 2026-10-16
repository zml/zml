//! Reproducible expert weights, activations, and routing for benchmark workloads.
const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

pub const Experts = struct {
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
};

pub const Generated = struct {
    prepared: Experts,
    canonical: ?Experts,
};

/// Describe the setup output, including optional operands and static layout metadata.
/// No compiler context or device allocation is needed.
pub fn templates(w: cases.Workload, check: bool) Generated {
    var c = w.case;
    c.experts = w.global_experts;
    return generateMode(undefined, c, .template, check, false);
}

/// Keep the structured metadata while binding the actual compiled output shapes,
/// including the expert partitioning established by the setup graph.
pub fn withOutputShapes(template: Generated, shapes: []const Shape) Generated {
    const Context = struct {
        shapes: []const Shape,
        index: usize = 0,

        fn assign(self: *@This(), tensor: *Tensor) void {
            tensor.* = .fromShape(self.shapes[self.index]);
            self.index += 1;
        }
    };
    var result = template;
    var context: Context = .{ .shapes = shapes };
    zml.meta.visit(Context.assign, &context, &result);
    std.debug.assert(context.index == shapes.len);
    return result;
}

/// Run once per graph configuration, outside the measured forward pass.
/// A runtime seed prevents XLA from embedding full expert arrays as constants.
pub fn generate(seed: Tensor, c: cases.Case, data: cases.Data, check: bool, expert_parallel: bool) Generated {
    return generateMode(seed, c, if (data == .random) .random else .fixture, check, expert_parallel);
}

const Mode = enum { template, fixture, random };

fn generateMode(seed: Tensor, c: cases.Case, mode: Mode, check: bool, expert_parallel: bool) Generated {
    const canonical = expertWeights(c, mode, seed);
    var prepared = prepareForBackend(canonical, c.backend, mode == .template);
    if (expert_parallel) zml.meta.visit(shardExperts, c.experts, &prepared);
    return .{ .prepared = prepared, .canonical = if (check) canonical else null };
}

fn shardExperts(experts: i64, tensor: *Tensor) void {
    // Projection weights, biases, and scales all begin with the expert axis;
    // scalar activation scales are replicated across ranks.
    if (tensor.rank() == 0) return;
    std.debug.assert(tensor.dim(0) == experts);
    tensor.* = tensor.renameTag(0, .expert).withPartitioning(.experts, .{ .expert = .experts });
}

// The runtime offset keeps deterministic data out of XLA constant folding.
fn generateDeterministicTensor(shape: Shape, multiplier: i64, divisor: f64, seed: Tensor) Tensor {
    return Tensor.arange(.{ .end = @intCast(shape.count()) }, .i64).add(seed.convert(.i64))
        .mul(Tensor.scalar(multiplier, .i64)).addConstant(3)
        .remainder(Tensor.scalar(17, .i64)).addConstant(-8)
        .convert(.f32).scale(1 / divisor).reshape(shape.withDtype(.f32));
}

/// Select exactly representable FP4 values with distinct even/odd sequences.
fn generateFp4Tensor(shape: Shape, multiplier: i64, seed: Tensor) Tensor {
    const values = Tensor.constantTensor(.init(.{ .value = 16 }, .f32), std.mem.sliceAsBytes(&[_]f32{
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    }));
    const index = Tensor.arange(.{ .end = @intCast(shape.count()) }, .i64).add(seed.convert(.i64));
    const pair = index.div(Tensor.scalar(2, .i64));
    // Prime periods distinguish experts, output rows, and gate/up halves.
    const even = pair.mul(Tensor.scalar(multiplier, .i64)).addConstant(3)
        .remainder(Tensor.scalar(17, .i64)).remainder(Tensor.scalar(16, .i64));
    const odd = pair.mul(Tensor.scalar(3, .i64)).addConstant(7)
        .remainder(Tensor.scalar(19, .i64)).remainder(Tensor.scalar(16, .i64));
    const indices = index.remainder(Tensor.scalar(2, .i64)).cmp(.EQ, Tensor.scalar(0, .i64)).select(even, odd);
    return values.gather(.{ .value = indices }, .{}).convert(.f4e2m1).reshape(shape.withDtype(.f4e2m1));
}

/// Create a Linear projection with a reproducible pattern, optionally quantized and biased.
fn linear(shape: Shape, scheme: ?zml.Quantization.Scheme, bias: bool, multiplier: i64, mode: Mode, seed: Tensor) zml.nn.Linear {
    const projection_bias = if (bias) weightTensor(shape.remove(2), 3, 32, mode, seed) else null;

    const s = scheme orelse {
        const weight = weightTensor(shape, multiplier, 16, mode, seed);
        // Keep activation magnitudes bounded as the swept hidden widths grow.
        const fan_in: f64 = @floatFromInt(shape.dim(2));
        return .{
            .weight = if (mode == .template) weight else weight.scale(@sqrt(3.0 / fan_in)),
            .tag = shape.tag(2),
            .bias = projection_bias,
        };
    };

    const weight = switch (s) {
        .mxfp4, .nvfp4 => switch (mode) {
            .template => Tensor.fromShape(shape.withDtype(.f4e2m1)),
            .fixture => generateFp4Tensor(shape, multiplier, seed),
            .random => randomFp4(shape, seed.addConstant(multiplier)),
        },
        else => weightTensor(shape.withDtype(.f8e4m3fn), multiplier, 16, mode, seed),
    };

    const scale_shape: Shape = switch (s) {
        .mxfp4, .mxfp8 => .init(.{ shape.dim(0), shape.dim(1), @divExact(shape.dim(2), 32) }, .f8e8m0),
        .nvfp4 => .init(.{ shape.dim(0), shape.dim(1), @divExact(shape.dim(2), 16) }, .f8e4m3fn),
        .fp8_per_tensor => .init(.{ shape.dim(0), 1, 1 }, .f32),
        .fp8_per_channel => .init(.{ shape.dim(0), shape.dim(1), 1 }, .f32),
        .fp8_block128 => .init(.{ shape.dim(0), @divExact(shape.dim(1), 128), @divExact(shape.dim(2), 128) }, .f32),
        .fp8_block32 => .init(.{ shape.dim(0), @divExact(shape.dim(1), 32), @divExact(shape.dim(2), 32) }, .f32),
    };

    const scales = quantizationScales(scale_shape, shape.dim(2), s, mode, seed, multiplier);

    const input_scale: ?zml.Quantization.GlobalScale = if (s == .nvfp4) .{ .value = if (mode == .template) Tensor.init(.{}, .f32) else Tensor.scalar(1, .f32), .operation = .multiply } else null;
    const global_scale: ?zml.Quantization.GlobalScale = if (s == .nvfp4) .{ .value = if (mode == .template) Tensor.init(.{ .expert = shape.dim(0) }, .f32) else Tensor.scalar(1, .f32).broad(.init(.{ .expert = shape.dim(0) }, .f32)), .operation = .multiply } else null;

    return .{
        .weight = weight,
        .tag = shape.tag(2),
        .bias = projection_bias,
        .quantization = .{
            .scheme = s,
            .scales = scales,
            .input_scale = input_scale,
            .global_scale = global_scale,
        },
    };
}

/// `template` describes canonical operands without requiring a compiler context.
fn expertWeights(c: cases.Case, mode: Mode, seed: Tensor) Experts {
    const projection_width: i64 = switch (c.activation) {
        .silu, .gelu, .relu => c.intermediate,
        else => 2 * c.intermediate,
    };
    return .{
        .gate_up = linear(.init(.{ .expert = c.experts, .dout = projection_width, .d = c.width }, c.activation_dtype), c.scheme, c.bias, 7, mode, seed),
        .down = linear(.init(.{ .expert = c.experts, .d = c.width, .dout = c.intermediate }, c.activation_dtype), c.scheme, c.bias, 11, mode, seed),
    };
}

/// Interleave the gate/up halves of the output axis: [E, 2, N/2, K] -> [E, N/2, 2, K].
fn interleave(tensor: Tensor) Tensor {
    const tags = tensor.shape().tags();
    return tensor.withTags(.{ .expert, .out, .in })
        .splitAxis(.out, .{ .projection = 2, .mid = @divExact(tensor.dim(1), 2) })
        .transpose(.{ .expert, .mid, .projection, .in })
        .merge(.{ .out = .{ .mid, .projection } }).withTags(tags);
}

/// Blackwell block scales: [E, N/128, 4, 32, K/4, 4] ->
/// [E, N/128, K/4, 32, 4, 4], retaining the declared [E, N, K] shape.
fn swizzleScales(scales: Tensor) Tensor {
    return scales.reshape(.{ scales.dim(0), @divExact(scales.dim(1), 128), 4, 32, @divExact(scales.dim(2), 4), 4 })
        .transpose(.{ 0, 1, 4, 3, 2, 5 }).reshape(scales.shape());
}

fn packFp4Bytes(weight: Tensor) Tensor {
    std.debug.assert(weight.dtype() == .f4e2m1);
    return weight.reshape(weight.shape().setDim(2, @divExact(weight.dim(2), 2)).append(.{ .nibble = 2 })).bitCast(.u8);
}

fn prepareGateUp(tensor: Tensor, layout: zml.moe.ExpertsLayout) Tensor {
    const ordered = switch (layout.gate_up_order) {
        .gate_up => tensor,
        .up_gate => blk: {
            const mid = @divExact(tensor.dim(1), 2);
            break :blk Tensor.concatenate(&.{ tensor.slice(1, .{ .start = mid }), tensor.slice(1, .{ .end = mid }) }, 1);
        },
    };
    return switch (layout.gate_up) {
        .concatenated => ordered,
        .interleaved => interleave(ordered),
    };
}

/// Adapt a copy of both projections to the backend's storage and layout contract.
fn prepareForBackend(canonical: Experts, backend: zml.moe.Backend, template: bool) Experts {
    var prepared = canonical;
    const scheme = canonical.gate_up.quantizationScheme();
    const layout = backend.expertsLayout(scheme) catch |err| zml.stdx.debug.panic("benchmark expert layout: {}", .{err});
    // Plain activations have one projection, so gate/up ordering does not apply.
    const gated = canonical.gate_up.weight.dim(1) == 2 * canonical.down.weight.dim(2);
    if (gated and !template) {
        prepared.gate_up.weight = prepareGateUp(canonical.gate_up.weight, layout);
        if (canonical.gate_up.bias) |bias| prepared.gate_up.bias = prepareGateUp(bias.appendAxes(.{.dummy}), layout).squeeze(.dummy);
        if (prepared.gate_up.quantization) |*q| q.scales = prepareGateUp(q.scales, layout);
    }

    for ([_]*zml.nn.Linear{ &prepared.gate_up, &prepared.down }) |projection| {
        if (layout.packing == .swizzled_scales or layout.packing == .flashinfer_nvfp4) {
            const q = &projection.quantization.?;
            if (!template) q.scales = swizzleScales(q.scales);
            q.swizzled_scales = true;
        }
        switch (backend) {
            .cute => {
                projection.weight = if (template) .fromShape(projection.weight.shape().setDim(2, @divExact(projection.weight.dim(2), 2)).withDtype(.u8)) else packFp4Bytes(projection.weight);
                const q = &projection.quantization.?;
                q.scales = if (template) .fromShape(q.scales.shape().withDtype(.u8)) else q.scales.bitCast(.u8);
            },
            .triton => if (scheme == .mxfp4) {
                projection.weight = if (template) .fromShape(projection.weight.shape().setDim(2, @divExact(projection.weight.dim(2), 2)).withDtype(.u8)) else packFp4Bytes(projection.weight);
            },
            .mosaic_tpu => {
                projection.weight = if (template) .fromShape(projection.weight.shape().transpose(.{ 0, 2, 1 })) else projection.weight.transpose(.{ 0, 2, 1 });
            },
            .metal => if (projection.quantization) |*q| {
                if (q.scheme != .nvfp4) q.scales = if (template) .fromShape(q.scales.shape().withDtype(.bf16)) else q.scales.convert(.bf16);
            },
            else => {},
        }
    }

    if (backend == .metal and scheme == .nvfp4) {
        const global = &prepared.gate_up.quantization.?.global_scale.?;
        global.value = if (template) .fromShape(global.value.shape().append(.{ .proj = 2 })) else global.value.appendAxes(.{.proj}).repeat1d(.proj, 2);
    }
    return prepared;
}

fn randomBits(shape: Shape, seed: Tensor) Tensor {
    var x = Tensor.arange(.{ .end = @intCast(shape.count()) }, .u32).add(seed);
    x = x.logical(.XOR, x.shiftRightLogical(Tensor.scalar(16, .u32))).mul(Tensor.scalar(0x7feb352d, .u32));
    x = x.logical(.XOR, x.shiftRightLogical(Tensor.scalar(15, .u32))).mul(Tensor.scalar(0x846ca68b, .u32));
    return x.logical(.XOR, x.shiftRightLogical(Tensor.scalar(16, .u32))).reshape(shape.withDtype(.u32));
}

fn weightTensor(shape: Shape, multiplier: i64, divisor: f64, mode: Mode, seed: Tensor) Tensor {
    return switch (mode) {
        .template => Tensor.fromShape(shape),
        .fixture => generateDeterministicTensor(shape, multiplier, divisor, seed).convert(shape.dtype()),
        .random => randomBits(shape, seed.addConstant(multiplier)).shiftRightLogical(Tensor.scalar(8, .u32))
            .convert(.f32).scale(2.0 / 16777216.0).addConstant(-1).convert(shape.dtype()),
    };
}

fn randomFp4(shape: Shape, seed: Tensor) Tensor {
    const values = Tensor.constantTensor(.init(.{ .value = 16 }, .f32), std.mem.sliceAsBytes(&[_]f32{
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    }));
    const index = randomBits(shape, seed).remainder(Tensor.scalar(16, .u32)).convert(.i32).flatten();
    return values.gather(.{ .value = index }, .{}).convert(.f4e2m1).reshape(shape.withDtype(.f4e2m1));
}

fn quantizationScales(shape: Shape, fan_in: i64, scheme: zml.Quantization.Scheme, mode: Mode, seed: Tensor, multiplier: i64) Tensor {
    if (mode == .template) return .fromShape(shape);
    const indices = switch (mode) {
        .template => unreachable,
        .fixture => Tensor.arange(.{ .end = @intCast(shape.count()) }, .u32).add(seed),
        .random => randomBits(shape, seed.addConstant(multiplier + 17)).flatten(),
    };
    // Power-of-two scales are exactly representable by each quantization format.
    // Compensate for fan-in while retaining varied scales across blocks and experts.
    const fp4 = scheme == .mxfp4 or scheme == .nvfp4;
    var base: f32 = @exp2(@floor(-0.5 * @log2(@as(f32, @floatFromInt(fan_in))))) / @as(f32, if (fp4) 8 else 2);
    if (scheme == .nvfp4) base = @max(base, 1.0 / 512.0);
    const values = Tensor.constantTensor(.init(.{ .value = 4 }, .f32), std.mem.sliceAsBytes(&[_]f32{ 1, 2, 4, 8 })).scale(base);
    return values.gather(.{ .value = indices.remainder(Tensor.scalar(4, .u32)).convert(.i32) }, .{})
        .convert(shape.dtype()).reshape(shape);
}

pub fn makeInput(allocator: std.mem.Allocator, w: cases.Workload) !zml.Slice {
    const c = w.case;
    const host = try zml.Slice.alloc(allocator, .init(.{ .t = c.tokens, .d = c.width }, c.activation_dtype));
    var rng: std.Random.DefaultPrng = .init(w.seed +% 0x9e3779b97f4a7c15);
    switch (c.activation_dtype) {
        inline .bf16, .f16, .f32 => |dtype| {
            for (host.items(dtype.toZigType()), 0..) |*value, i| {
                const f = if (w.data == .random) rng.random().float(f32) * 2 - 1 else @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + i / 128) % 23)) - 11)) / 32;
                const scaled = f * w.input_scale;
                value.* = if (dtype == .bf16) .fromF32(scaled) else @floatCast(scaled);
            }
        },
        else => unreachable,
    }
    return host;
}

pub const Routes = struct {
    ids: []i32,
    weights: []f32,
    counts: []i64,
    local_routes: i64 = 0,
    touched: i64 = 0,
    maximum: i64 = 0,
    padded_rows: [5]i64 = @splat(0),
    ranks: []RankStatistics,

    pub fn deinit(r: Routes, allocator: std.mem.Allocator) void {
        allocator.free(r.ids);
        allocator.free(r.weights);
        allocator.free(r.counts);
        allocator.free(r.ranks);
    }
};

pub const RankStatistics = struct {
    routes: i64 = 0,
    touched_experts: i64 = 0,
    maximum_routes: i64 = 0,
    padded_rows: [5]i64 = @splat(0),
};

pub fn generateRoutes(allocator: std.mem.Allocator, w: cases.Workload) !Routes {
    const c = w.case;
    const count: usize = @intCast(c.tokens * c.topk);
    var r: Routes = .{
        .ids = try allocator.alloc(i32, count),
        .weights = undefined,
        .counts = undefined,
        .ranks = undefined,
    };
    errdefer allocator.free(r.ids);
    r.weights = try allocator.alloc(f32, count);
    errdefer allocator.free(r.weights);
    r.counts = try allocator.alloc(i64, @intCast(w.global_experts));
    errdefer allocator.free(r.counts);
    @memset(r.counts, 0);
    r.ranks = try allocator.alloc(RankStatistics, @intCast(@divExact(w.global_experts, c.experts)));
    errdefer allocator.free(r.ranks);
    @memset(r.ranks, .{});
    var eligible: std.ArrayList(i32) = .empty;
    defer eligible.deinit(allocator);
    for (0..@intCast(w.global_experts)) |i| {
        if (w.active_experts == 0 or @mod(i, @as(usize, @intCast(c.experts))) < w.active_experts) try eligible.append(allocator, @intCast(i));
    }
    if (c.topk > eligible.items.len) return error.InsufficientExperts;
    var rng: std.Random.DefaultPrng = .init(w.seed);
    var popularity: std.Random.DefaultPrng = .init(w.seed ^ 0xd1b54a32d192ed03);
    const logits = try allocator.alloc(f32, eligible.items.len);
    defer allocator.free(logits);
    const keys = try allocator.alloc(f32, logits.len);
    defer allocator.free(keys);
    for (logits) |*l| l.* = if (w.routing == .skewed) w.sigma * popularity.random().floatNorm(f32) else 0;
    const k: usize = @intCast(c.topk);
    for (0..@intCast(c.tokens)) |token| {
        if (w.routing != .balanced) for (keys, logits) |*key, l| {
            key.* = l - @log(-@log(@max(rng.random().float(f32), 1e-20)));
        };
        for (0..k) |slot| {
            const index = token * k + slot;
            const chosen = if (w.routing == .balanced) index % eligible.items.len else blk: {
                var best: usize = 0;
                for (keys, 0..) |key, i| if (key > keys[best]) {
                    best = i;
                };
                keys[best] = -std.math.inf(f32);
                break :blk best;
            };
            const id = eligible.items[chosen];
            r.ids[index] = id;
            r.weights[index] = 0.1 + 0.3 * rng.random().float(f32);
            r.counts[@intCast(id)] += 1;
        }
        const token_weights = r.weights[token * k ..][0..k];
        var total_weight: f32 = 0;
        for (token_weights) |weight| total_weight += weight;
        for (token_weights) |*weight| weight.* = (weight.* / total_weight) * w.routing_scale;
    }
    for (r.counts, 0..) |n, expert| {
        const rank = &r.ranks[expert / @as(usize, @intCast(c.experts))];
        rank.routes += n;
        rank.touched_experts += @intFromBool(n > 0);
        rank.maximum_routes = @max(rank.maximum_routes, n);
        for ([_]i64{ 8, 16, 32, 64, 128 }, &rank.padded_rows) |tile, *rows| rows.* += @divTrunc(n + tile - 1, tile) * tile;
    }
    r.local_routes = r.ranks[0].routes;
    r.touched = r.ranks[0].touched_experts;
    r.maximum = r.ranks[0].maximum_routes;
    r.padded_rows = r.ranks[0].padded_rows;
    return r;
}

test "routing is reproducible, distinct, and respects active experts" {
    const allocator = std.testing.allocator;
    inline for (std.meta.tags(cases.Routing)) |routing| {
        const w: cases.Workload = .{
            .case = .{ .backend = .stablehlo, .tokens = 13, .topk = 3, .experts = 4 },
            .routing = routing,
            .global_experts = 8,
            .active_experts = 2,
            .sigma = 0.6,
            .seed = 42,
            .data = .fixture,
        };
        const a = try generateRoutes(allocator, w);
        defer a.deinit(allocator);
        const b = try generateRoutes(allocator, w);
        defer b.deinit(allocator);
        try std.testing.expectEqualSlices(i32, a.ids, b.ids);
        try std.testing.expectEqualSlices(f32, a.weights, b.weights);
        for (a.ids, 0..) |id, i| {
            try std.testing.expect(@mod(id, 4) < 2);
            for (a.ids[i - i % 3 .. i]) |previous| try std.testing.expect(id != previous);
        }
        for (0..@intCast(w.case.tokens)) |token| {
            var sum: f32 = 0;
            for (a.weights[token * 3 ..][0..3]) |weight| sum += weight;
            try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
        }
        if (routing == .balanced) try std.testing.expect(@abs(a.counts[0] - a.counts[1]) <= 1);
        try std.testing.expectEqual(@as(i64, 0), a.counts[2]);
        try std.testing.expectEqual(@as(i64, 0), a.counts[3]);
        try std.testing.expectEqual(w.case.tokens * w.case.topk, a.ranks[0].routes + a.ranks[1].routes);
        try std.testing.expectEqual(a.local_routes, a.ranks[0].routes);
        try std.testing.expectEqualSlices(i64, &a.padded_rows, &a.ranks[0].padded_rows);
    }
}

test "zero inputs and scaled routing retain the selected expert workload" {
    const allocator = std.testing.allocator;
    var w: cases.Workload = .{
        .case = .{ .tokens = 3, .topk = 2, .experts = 4 },
        .routing = .uniform,
        .global_experts = 8,
        .active_experts = 0,
        .sigma = 0.6,
        .seed = 1,
        .data = .random,
        .input_scale = 0,
        .routing_scale = 2.5,
    };
    const host = try makeInput(allocator, w);
    defer host.free(allocator);
    for (host.items(zml.DataType.bf16.toZigType())) |value| try std.testing.expectEqual(@as(f32, 0), value.toF32());
    const a = try generateRoutes(allocator, w);
    defer a.deinit(allocator);
    w.routing_scale = 1;
    const b = try generateRoutes(allocator, w);
    defer b.deinit(allocator);
    try std.testing.expectEqualSlices(i32, a.ids, b.ids);
    for (a.weights, b.weights) |actual, original| try std.testing.expectEqual(original * 2.5, actual);
}

test "expert templates retain packed layouts and optional scale metadata" {
    var w: cases.Workload = .{
        .case = .{ .backend = .cute, .scheme = .mxfp4, .width = 5120, .intermediate = 2304, .experts = 4 },
        .routing = .balanced,
        .global_experts = 8,
        .active_experts = 0,
        .sigma = 0.6,
        .seed = 1,
        .data = .random,
    };
    const cute = templates(w, true);
    try std.testing.expectEqual(zml.DataType.u8, cute.prepared.gate_up.weight.dtype());
    try std.testing.expectEqual(@as(i64, 2560), cute.prepared.gate_up.weight.dim(2));
    try std.testing.expectEqual(@as(i64, 8), cute.prepared.gate_up.weight.dim(0));
    try std.testing.expect(cute.prepared.gate_up.quantization.?.swizzled_scales);
    try std.testing.expect(cute.prepared.down.quantization.?.swizzled_scales);
    try std.testing.expectEqual(zml.DataType.f4e2m1, cute.canonical.?.gate_up.weight.dtype());
    try std.testing.expect(!cute.canonical.?.gate_up.quantization.?.swizzled_scales);

    w.case = .{ .backend = .metal, .scheme = .nvfp4, .width = 128, .intermediate = 128, .experts = 4 };
    const metal = templates(w, true);
    const prepared_q = metal.prepared.gate_up.quantization.?;
    const canonical_q = metal.canonical.?.gate_up.quantization.?;
    try std.testing.expectEqual(@as(usize, 0), prepared_q.input_scale.?.value.rank());
    try std.testing.expectEqual(.multiply, prepared_q.input_scale.?.operation);
    try std.testing.expectEqual(.multiply, prepared_q.global_scale.?.operation);
    try std.testing.expectEqual(@as(i64, 2), prepared_q.global_scale.?.value.dim(.proj));
    try std.testing.expectEqual(@as(usize, 1), canonical_q.global_scale.?.value.rank());

    w.case = .{ .backend = .stablehlo, .bias = true, .activation = .relu, .experts = 4 };
    const plain = templates(w, false);
    try std.testing.expect(plain.canonical == null);
    try std.testing.expect(plain.prepared.gate_up.quantization == null);
    try std.testing.expect(plain.prepared.gate_up.bias != null);
    try std.testing.expectEqual(w.case.intermediate, plain.prepared.gate_up.weight.dim(1));
}
