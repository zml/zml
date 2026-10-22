//! The CuTe MXFP4 MoE on MegaMoE's weights (`persistent.WeightLayout.megamoe`)
//! against the same checkpoint weights in its own layout. The GEMMs see the
//! same weight rows in another order and the same scale atoms at other
//! addresses, so the outputs must match bit for bit. Needs an SM100 GPU.
const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const cute = zml.moe.cute;
const kernels = cute.kernels;
const transforms = zml.moe.megamoe.weights;

const hidden = 5120;
const intermediate = 2304;
const experts = 4;
const topk = 4;
const swiglu_limit = 10.0;

/// The checkpoint's expert tensors: packed e2m1 weights and 1x32 UE8M0 scales,
/// gate (`w1`) and up (`w3`) `[E, I, ..]`, down (`w2`) `[E, H, ..]`.
const checkpoint_shapes = [_]zml.Shape{
    .init(.{ experts, intermediate, hidden / 2 }, .u8),
    .init(.{ experts, intermediate, hidden / 32 }, .u8),
    .init(.{ experts, intermediate, hidden / 2 }, .u8),
    .init(.{ experts, intermediate, hidden / 32 }, .u8),
    .init(.{ experts, hidden, intermediate / 2 }, .u8),
    .init(.{ experts, hidden, intermediate / 32 }, .u8),
};

/// Gate and up rows interleaved one by one: `[E, I, K]` twice -> `[E, 2I, K]`.
fn interleaveGateUp(gate: Tensor, up: Tensor) Tensor {
    return Tensor.stack(&.{ gate, up }, 2, .half).reshape(.{ gate.dim(0), 2 * gate.dim(1), gate.dim(2) });
}

/// 128x4 scale atoms in K-major order: `[E, N/128, 4, 32, K/4, 4]` ->
/// `[E, N/128, K/4, 32, 4, 4]`, keeping the `[E, N, K]` shape.
fn swizzleScales(scales: Tensor) Tensor {
    return scales.reshape(.{ scales.dim(0), @divExact(scales.dim(1), 128), 4, 32, @divExact(scales.dim(2), 4), 4 })
        .transpose(.{ 0, 1, 4, 3, 2, 5 }).reshape(scales.shape());
}

const Weights = struct {
    own: struct { w1: Tensor, s1: Tensor, w2: Tensor, s2: Tensor },
    megamoe: zml.moe.megamoe.Weights,

    fn init(w1: Tensor, s1: Tensor, w3: Tensor, s3: Tensor, w2: Tensor, s2: Tensor) Weights {
        const l1 = transforms.routedL1(w1, s1, w3, s3);
        const l2 = transforms.routedL2(w2, s2);
        const tags = .{ .expert, .out, .in };
        return .{
            .own = .{
                .w1 = interleaveGateUp(w1, w3).withTags(tags),
                .s1 = swizzleScales(interleaveGateUp(s1, s3)).withTags(tags),
                .w2 = w2.withTags(tags),
                .s2 = swizzleScales(s2).withTags(tags),
            },
            .megamoe = .{
                .l1_w = l1.w.withTags(tags),
                .l1_sf = l1.sf.withTags(tags),
                .l2_w = l2.w.withTags(tags),
                .l2_sf = l2.sf.withTags(tags),
            },
        };
    }
};

const Fns = struct {
    /// Both layouts through `kernels.forward` with one launch plan: `tile_n`
    /// 0 and `two_cta` -1 leave the choice to the default plan.
    fn launch(x: Tensor, ids: Tensor, weights: Tensor, w1: Tensor, s1: Tensor, w3: Tensor, s3: Tensor, w2: Tensor, s2: Tensor, tile_n: i64, two_cta: i64) [2]Tensor {
        const tokens = x.dim(0);
        const tuning: kernels.Tuning = .{
            .tile_n = if (tile_n == 0) null else tile_n,
            .two_cta = if (two_cta < 0) null else two_cta == 1,
        };
        const plan = tuning.resolveFor(zml.Compiler.current().platform, tokens, hidden, intermediate, experts, topk) catch unreachable;
        const layouts: Weights = .init(w1, s1, w3, s3, w2, s2);
        const own: kernels.Inputs = .{
            .x = x,
            .routing_weights = weights,
            .w1 = layouts.own.w1,
            .s1 = layouts.own.s1,
            .w2 = layouts.own.w2,
            .s2 = layouts.own.s2,
            .ids = ids,
        };
        var megamoe = own;
        megamoe.w1 = layouts.megamoe.l1_w;
        megamoe.s1 = layouts.megamoe.l1_sf;
        megamoe.w2 = layouts.megamoe.l2_w;
        megamoe.s2 = layouts.megamoe.l2_sf;
        megamoe.weight_layout = .megamoe;
        return .{
            kernels.forward(tokens, hidden, intermediate, experts, topk, swiglu_limit, own, plan),
            kernels.forward(tokens, hidden, intermediate, experts, topk, swiglu_limit, megamoe, plan),
        };
    }

    /// Both layouts through the public entry points.
    fn public(x: Tensor, ids: Tensor, weights: Tensor, w1: Tensor, s1: Tensor, w3: Tensor, s3: Tensor, w2: Tensor, s2: Tensor) [2]Tensor {
        const layouts: Weights = .init(w1, s1, w3, s3, w2, s2);
        const options: zml.moe.Options = .{
            .activation = .{ .swiglu = .{ .limit = swiglu_limit } },
            .quantize_input = false,
            .routing_weight_placement = .before_down,
        };
        const input = x.withTags(.{ .t, .d });
        const routes = ids.withTags(.{ .t, .topk });
        const routing_weights = weights.withTags(.{ .t, .topk });
        const gate_up: zml.nn.Linear = .{
            .weight = layouts.own.w1,
            .tag = zml.Shape.toTag(.in),
            .quantization = .{ .scheme = .mxfp4, .scales = layouts.own.s1, .swizzled_scales = true },
        };
        const down: zml.nn.Linear = .{
            .weight = layouts.own.w2,
            .tag = zml.Shape.toTag(.in),
            .quantization = .{ .scheme = .mxfp4, .scales = layouts.own.s2, .swizzled_scales = true },
        };
        return .{
            cute.fusedExperts(input, routes, routing_weights, gate_up, down, options),
            cute.fusedMegaMoEExperts(input, routes, routing_weights, layouts.megamoe, options),
        };
    }
};

/// Random packed e2m1 bytes and UE8M0 scales of 2^-9 .. 2^-6.
fn checkpointBuffers(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, buffers: *[checkpoint_shapes.len]zml.Buffer) !void {
    var rng = std.Random.DefaultPrng.init(41);
    const random = rng.random();
    for (buffers, checkpoint_shapes, 0..) |*buffer, shape, i| {
        const slice = try zml.Slice.alloc(allocator, shape);
        defer slice.free(allocator);
        if (i % 2 == 0) {
            random.bytes(slice.data());
        } else for (slice.data()) |*byte| {
            byte.* = 118 + random.uintLessThan(u8, 4);
        }
        buffer.* = try .fromSlice(io, platform, slice);
    }
}

/// Activations, routes (one in five owned by another rank: -1) and routing weights.
fn routeBuffers(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, tokens: i64, seed: u64) ![3]zml.Buffer {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    const shapes = [_]zml.Shape{ .init(.{ tokens, hidden }, .bf16), .init(.{ tokens, topk }, .i32), .init(.{ tokens, topk }, .f32) };
    var buffers: [3]zml.Buffer = undefined;
    for (&buffers, shapes, 0..) |*buffer, shape, i| {
        const slice = try zml.Slice.alloc(allocator, shape);
        defer slice.free(allocator);
        switch (i) {
            0 => for (slice.items(zml.floats.BFloat16)) |*v| {
                v.* = .fromF32(random.float(f32) * 2 - 1);
            },
            1 => for (slice.items(i32), 0..) |*id, r| {
                const expert: i32 = @intCast((r / topk + r % topk) % (experts + 1));
                id.* = if (expert == experts) -1 else expert;
            },
            else => for (slice.items(f32)) |*w| {
                w.* = 0.1 + 0.3 * random.float(f32);
            },
        }
        buffer.* = try .fromSlice(io, platform, slice);
    }
    return buffers;
}

fn expectSameBits(allocator: std.mem.Allocator, io: std.Io, results: *[2]zml.Buffer) !void {
    const own = try results[0].toSliceAlloc(allocator, io);
    defer own.free(allocator);
    const megamoe = try results[1].toSliceAlloc(allocator, io);
    defer megamoe.free(allocator);
    // A comparison of zeros would prove nothing.
    var nonzero: usize = 0;
    for (own.items(zml.floats.BFloat16)) |v| nonzero += @intFromBool(v.toF32() != 0);
    try std.testing.expect(nonzero > own.items(zml.floats.BFloat16).len / 2);
    try std.testing.expectEqualSlices(u8, own.constData(), megamoe.constData());
}

test "CuTe MoE on MegaMoE's weight layout matches its own layout bit for bit" {
    const platform = zml.testing.env();
    if (!cute.isAvailable(platform)) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var weights: [checkpoint_shapes.len]zml.Buffer = undefined;
    try checkpointBuffers(allocator, io, platform, &weights);
    defer for (&weights) |*b| b.deinit();
    const w: [checkpoint_shapes.len]Tensor = .{
        .fromShape(checkpoint_shapes[0]), .fromShape(checkpoint_shapes[1]), .fromShape(checkpoint_shapes[2]),
        .fromShape(checkpoint_shapes[3]), .fromShape(checkpoint_shapes[4]), .fromShape(checkpoint_shapes[5]),
    };

    // Tokens, tile N (0: default) and CTA pairs (-1: default). One token routes
    // directly (tile N 8); six is a DSpark decode step.
    const launches = [_][3]i64{
        .{ 1, 0, -1 },
        .{ 6, 0, -1 },
        .{ 384, 16, 0 },
        .{ 384, 32, 0 },
        .{ 384, 64, 0 },
        .{ 384, 128, 0 },
        .{ 384, 64, 1 },
        .{ 384, 128, 1 },
    };
    for (launches, 0..) |l, i| {
        const tokens = l[0];
        var routes = try routeBuffers(allocator, io, platform, tokens, i);
        defer for (&routes) |*b| b.deinit();
        var exe = try platform.compileFn(allocator, io, Fns.launch, .{
            Tensor.fromShape(routes[0].shape()), Tensor.fromShape(routes[1].shape()), Tensor.fromShape(routes[2].shape()),
            w[0],                                w[1],                                w[2],
            w[3],                                w[4],                                w[5],
            l[1],                                l[2],
        }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var results: [2]zml.Buffer = undefined;
        runner.run(io, .{ routes[0], routes[1], routes[2], weights[0], weights[1], weights[2], weights[3], weights[4], weights[5] }, &results, .{ .wait = true });
        defer for (&results) |*r| r.deinit();
        expectSameBits(allocator, io, &results) catch |err| {
            std.debug.print("launch tokens={d} tile_n={d} two_cta={d}\n", .{ l[0], l[1], l[2] });
            return err;
        };
    }

    {
        var routes = try routeBuffers(allocator, io, platform, 6, 99);
        defer for (&routes) |*b| b.deinit();
        var exe = try platform.compileFn(allocator, io, Fns.public, .{
            Tensor.fromShape(routes[0].shape()), Tensor.fromShape(routes[1].shape()), Tensor.fromShape(routes[2].shape()),
            w[0],                                w[1],                                w[2],
            w[3],                                w[4],                                w[5],
        }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var results: [2]zml.Buffer = undefined;
        runner.run(io, .{ routes[0], routes[1], routes[2], weights[0], weights[1], weights[2], weights[3], weights[4], weights[5] }, &results, .{ .wait = true });
        defer for (&results) |*r| r.deinit();
        try expectSameBits(allocator, io, &results);
    }
}
