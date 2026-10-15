//! Single-GPU harness for the CuTe MXFP4 MoE (`zml.moe.cute_mxfp4.kernels`)
//! at the DeepSeek V4.1 expert geometry under expert-parallel conditions:
//! every token carries `topk` global expert ids out of `global_experts`, and
//! only the `experts` ids starting at `expert_offset` are computed here.
//!
//! Per-kernel times come from running it under `nsys`; the harness itself
//! prints the routing statistics, the wall time per call, and can dump the
//! output or compare it against an earlier dump.
const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const kernels = zml.moe.cute_mxfp4.kernels;

const log = std.log.scoped(.moe_bench);

const hidden = 5120;
const intermediate = 2304;
const swiglu_limit: f32 = 10.0;

/// `balanced` gives every global expert exactly `tokens * topk / global_experts`
/// routes (token `t` slot `k` goes to expert `(t * topk + k) % global_experts`).
const Routing = enum { uniform, skewed, balanced };

const Params = struct {
    tokens: i64,
    topk: i64,
    experts: i64,
    expert_offset: i64,
    tile: ?i64,
};

fn forward(x: zml.Tensor, ids: zml.Tensor, weights: zml.Tensor, w1: zml.Tensor, s1: zml.Tensor, w2: zml.Tensor, s2: zml.Tensor, p: Params) zml.Tensor {
    const start: zml.Tensor = .scalar(p.expert_offset, .i32);
    const end: zml.Tensor = .scalar(p.expert_offset + p.experts, .i32);
    // Same masking as `cute_mxfp4.Context.body` under expert parallelism.
    const local = ids.cmp(.GE, start).logical(.AND, ids.cmp(.LT, end));
    const local_ids = local.select(ids.sub(start), .scalar(-1, .i32));
    return kernels.forwardWithTile(p.tokens, hidden, intermediate, p.experts, p.topk, swiglu_limit, .{
        .x = x,
        .routing_weights = weights,
        .w1 = w1,
        .s1 = s1,
        .w2 = w2,
        .s2 = s2,
        .ids = local_ids,
    }, p.tile);
}

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ moe_bench --tokens=4096 [--routing=uniform|skewed] [--sigma=0.6] [--expert_offset=0]
            \\           [--tile=64] [--iters=20] [--dump=out.bin] [--compare=ref.bin]
        ;
        tokens: i64 = 4096,
        topk: i64 = 6,
        global_experts: i64 = 384,
        experts: i64 = 192,
        expert_offset: i64 = 0,
        routing: Routing = .uniform,
        /// Standard deviation of the per-expert popularity logits of `skewed`.
        sigma: f32 = 0.6,
        /// Restrict routing to the first `active_experts` experts of this
        /// rank and of the other rank (0: all), so that the number of touched
        /// local experts varies at a fixed route count.
        active_experts: i64 = 0,
        /// Tile N override; 0 keeps the kernel's own choice.
        tile: i64 = 0,
        iters: u32 = 20,
        seed: u64 = 1,
        dump: ?[]const u8 = null,
        compare: ?[]const u8 = null,
    };

    const allocator = init.gpa;
    const io = init.io;
    const args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    if (platform.devices.len > 0) {
        if (platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count")) |sms| log.info("core_count {d}", .{sms.int64});
    }

    const tokens = args.tokens;
    const topk = args.topk;
    const routes: usize = @intCast(tokens * topk);

    // Routing: `topk` distinct global experts per token.
    var ids_host = try allocator.alloc(i32, routes);
    defer allocator.free(ids_host);
    var weights_host = try allocator.alloc(f32, routes);
    defer allocator.free(weights_host);
    {
        var rng = std.Random.DefaultPrng.init(args.seed);
        const random = rng.random();
        // Expert popularity is fixed across seeds of the token draws.
        var popularity_rng = std.Random.DefaultPrng.init(12345);
        const logits = try allocator.alloc(f32, @intCast(args.global_experts));
        defer allocator.free(logits);
        for (logits, 0..) |*l, expert| {
            l.* = switch (args.routing) {
                .uniform, .balanced => 0,
                .skewed => args.sigma * popularity_rng.random().floatNorm(f32),
            };
            const local_index = @rem(@as(i64, @intCast(expert)), args.experts);
            if (args.active_experts > 0 and local_index >= args.active_experts) l.* = -std.math.inf(f32);
        }
        const keys = try allocator.alloc(f32, logits.len);
        defer allocator.free(keys);
        for (0..@intCast(tokens)) |t| {
            if (args.routing == .balanced) {
                for (0..@intCast(topk)) |k| {
                    const route = t * @as(usize, @intCast(topk)) + k;
                    ids_host[route] = @intCast(route % @as(usize, @intCast(args.global_experts)));
                    weights_host[route] = 0.1 + 0.3 * random.float(f32);
                }
                continue;
            }
            // Gumbel top-k samples experts without replacement in proportion
            // to exp(logit).
            for (keys, logits) |*key, l| {
                const u = @max(random.float(f32), 1e-20);
                key.* = l - @log(-@log(u));
            }
            for (0..@intCast(topk)) |k| {
                var best: usize = 0;
                for (keys, 0..) |key, e| {
                    if (key > keys[best]) best = e;
                }
                ids_host[t * @as(usize, @intCast(topk)) + k] = @intCast(best);
                keys[best] = -std.math.inf(f32);
                weights_host[t * @as(usize, @intCast(topk)) + k] = 0.1 + 0.3 * random.float(f32);
            }
        }
    }

    // Routing statistics of this rank.
    const counts = try allocator.alloc(i64, @intCast(args.experts));
    defer allocator.free(counts);
    @memset(counts, 0);
    var local_routes: i64 = 0;
    for (ids_host) |id| {
        if (id >= args.expert_offset and id < args.expert_offset + args.experts) {
            counts[@intCast(id - args.expert_offset)] += 1;
            local_routes += 1;
        }
    }
    var touched: i64 = 0;
    var max_count: i64 = 0;
    for (counts) |c| {
        if (c > 0) touched += 1;
        max_count = @max(max_count, c);
    }
    const mean = @as(f64, @floatFromInt(local_routes)) / @as(f64, @floatFromInt(args.experts));
    log.info("tokens {d} routing {s} local routes {d} touched experts {d}/{d} mean {d:.1} max {d} ({d:.2}x)", .{
        tokens,                                                @tagName(args.routing), local_routes, touched, args.experts, mean, max_count,
        @as(f64, @floatFromInt(max_count)) / @max(mean, 1e-9),
    });
    for ([_]i64{ 8, 16, 32, 64, 128 }) |n| {
        var groups: i64 = 0;
        for (counts) |c| groups += @divTrunc(c + n - 1, n);
        log.info("  tile {d}: {d} groups, padded rows {d} ({d:.1}% padding)", .{ n, groups, groups * n, 100.0 * (1.0 - @as(f64, @floatFromInt(local_routes)) / @as(f64, @floatFromInt(@max(groups * n, 1)))) });
    }

    // Shapes of one rank's expert weights.
    const e = args.experts;
    const x_shape: zml.Shape = .init(.{ tokens, hidden }, .bf16);
    const ids_shape: zml.Shape = .init(.{ tokens, topk }, .i32);
    const weights_shape: zml.Shape = .init(.{ tokens, topk }, .f32);
    const w1_shape: zml.Shape = .init(.{ .expert = e, .out = 2 * intermediate, .d = hidden / 2 }, .u8);
    const s1_shape: zml.Shape = .init(.{ .expert = e, .out = 2 * intermediate, .d = hidden / 32 }, .u8);
    const w2_shape: zml.Shape = .init(.{ .expert = e, .out = hidden, .d = intermediate / 2 }, .u8);
    const s2_shape: zml.Shape = .init(.{ .expert = e, .out = hidden, .d = intermediate / 32 }, .u8);

    const params: Params = .{
        .tokens = tokens,
        .topk = topk,
        .experts = e,
        .expert_offset = args.expert_offset,
        .tile = if (args.tile == 0) null else args.tile,
    };

    var exe = blk: {
        const now: std.Io.Timestamp = .now(io, .awake);
        defer log.info("compiled in {f}", .{now.untilNow(io, .awake)});
        break :blk try platform.compileFn(allocator, io, forward, .{
            zml.Tensor.fromShape(x_shape),
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(weights_shape),
            zml.Tensor.fromShape(w1_shape),
            zml.Tensor.fromShape(s1_shape),
            zml.Tensor.fromShape(w2_shape),
            zml.Tensor.fromShape(s2_shape),
            params,
        }, .{});
    };
    defer exe.deinit();

    // Deterministic operands: random FP4 nibbles, E8M0 scales near 2^-7 and
    // BF16 activations in [-1, 1].
    var data_rng = std.Random.DefaultPrng.init(args.seed +% 0x9e3779b97f4a7c15);
    const data_random = data_rng.random();
    var x_buffer = blk: {
        const slice = try zml.Slice.alloc(allocator, x_shape);
        defer slice.free(allocator);
        for (slice.items(zml.floats.BFloat16)) |*v| v.* = .fromF32(data_random.float(f32) * 2 - 1);
        break :blk try zml.Buffer.fromSlice(io, platform, slice);
    };
    defer x_buffer.deinit();
    var ids_buffer = try zml.Buffer.fromBytes(io, platform, ids_shape, std.mem.sliceAsBytes(ids_host));
    defer ids_buffer.deinit();
    var weights_buffer = try zml.Buffer.fromBytes(io, platform, weights_shape, std.mem.sliceAsBytes(weights_host));
    defer weights_buffer.deinit();
    var w1_buffer = try randomBytes(allocator, io, platform, w1_shape, data_random, .weight);
    defer w1_buffer.deinit();
    var s1_buffer = try randomBytes(allocator, io, platform, s1_shape, data_random, .scale);
    defer s1_buffer.deinit();
    var w2_buffer = try randomBytes(allocator, io, platform, w2_shape, data_random, .weight);
    defer w2_buffer.deinit();
    var s2_buffer = try randomBytes(allocator, io, platform, s2_shape, data_random, .scale);
    defer s2_buffer.deinit();

    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    const inputs = .{ x_buffer, ids_buffer, weights_buffer, w1_buffer, s1_buffer, w2_buffer, s2_buffer };

    for (0..3) |_| {
        var out: zml.Buffer = undefined;
        runner.run(io, inputs, .{&out}, .{ .wait = true });
        out.deinit();
    }

    const start: std.Io.Timestamp = .now(io, .awake);
    for (0..args.iters) |_| {
        var out: zml.Buffer = undefined;
        runner.run(io, inputs, .{&out}, .{});
        out.deinit();
    }
    var out: zml.Buffer = undefined;
    runner.run(io, inputs, .{&out}, .{ .wait = true });
    defer out.deinit();
    const elapsed_ns: f64 = @floatFromInt(start.untilNow(io, .awake).toNanoseconds());
    const per_call_us = elapsed_ns / @as(f64, @floatFromInt(args.iters + 1)) / 1000;
    const flops = 2.0 * @as(f64, @floatFromInt(local_routes)) * (2.0 * intermediate * hidden + hidden * intermediate);
    const weight_bytes = @as(f64, @floatFromInt(touched)) * (2.0 * intermediate * hidden + hidden * intermediate) * (0.5 + 1.0 / 32.0);
    log.info("wall {d:.1} us/call ({d} iters): {d:.0} TFLOP/s on local routes, {d:.0} GB/s of touched expert weights", .{
        per_call_us,
        args.iters + 1,
        flops / per_call_us / 1e6,
        weight_bytes / per_call_us / 1e3,
    });

    const result = try out.toSliceAlloc(allocator, io);
    defer result.free(allocator);
    const values = result.constItems(zml.floats.BFloat16);
    var nonfinite: usize = 0;
    var sum_abs: f64 = 0;
    for (values) |v| {
        const f = v.toF32();
        if (!std.math.isFinite(f)) nonfinite += 1 else sum_abs += @abs(f);
    }
    log.info("output mean |y| {d:.5}, non-finite {d}", .{ sum_abs / @as(f64, @floatFromInt(values.len)), nonfinite });

    if (args.dump) |path| {
        const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
        defer file.close(io);
        try file.writePositionalAll(io, result.constData(), 0);
        log.info("dumped output to {s}", .{path});
    }
    if (args.compare) |path| {
        const reference_bytes = try stdx.Io.Dir.readFileAlloc(.cwd(), io, path, allocator, .unlimited);
        defer allocator.free(reference_bytes);
        if (reference_bytes.len != result.constData().len) stdx.flags.fatal("{s} has {d} bytes, expected {d}", .{ path, reference_bytes.len, result.constData().len });
        const reference: []align(1) const zml.floats.BFloat16 = std.mem.bytesAsSlice(zml.floats.BFloat16, reference_bytes);
        var max_abs: f32 = 0;
        var max_ref: f32 = 0;
        var mismatches: usize = 0;
        var exact: usize = 0;
        var sum_sq_diff: f64 = 0;
        var sum_sq_ref: f64 = 0;
        for (values, reference) |actual_bf16, expected_bf16| {
            const actual = actual_bf16.toF32();
            const expected = expected_bf16.toF32();
            const diff = @abs(actual - expected);
            if (diff == 0) exact += 1;
            max_abs = @max(max_abs, diff);
            max_ref = @max(max_ref, @abs(expected));
            sum_sq_diff += diff * diff;
            sum_sq_ref += expected * expected;
            // The MoE tests' tolerances.
            if (!(diff <= 0.015625 + 0.02 * @abs(expected))) mismatches += 1;
        }
        log.info("compare to {s}: exact {d}/{d}, max |diff| {d:.6} (max |ref| {d:.4}), relative RMS {e:.3}, outside atol 1/64 + rtol 0.02: {d}", .{
            path, exact, values.len, max_abs, max_ref, @sqrt(sum_sq_diff / @max(sum_sq_ref, 1e-30)), mismatches,
        });
        if (mismatches != 0) std.process.exit(2);
    }
}

fn randomBytes(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, random: std.Random, comptime kind: enum { weight, scale }) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    const bytes = slice.data();
    switch (kind) {
        .weight => random.bytes(bytes),
        // E8M0 exponents 2^-9 .. 2^-6; never the 0xff NaN encoding.
        .scale => for (bytes) |*b| {
            b.* = 118 + random.uintLessThan(u8, 4);
        },
    }
    return .fromSlice(io, platform, slice);
}
