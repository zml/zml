const std = @import("std");

const zml = @import("zml");
const Shape = zml.Shape;
const Tensor = zml.Tensor;
const stdx = zml.stdx;
//const Tracer = zml.tracer.Tracer;

const log = std.log.scoped(.moe_generated);

const Fwd = struct {
    pub fn forward(hidden_states: Tensor, w1: Tensor, w2: Tensor, topk_weights: Tensor, topk_ids: Tensor) Tensor {
        return zml.general_triton_moe.fusedExpertsImpl(hidden_states, w1, w2, topk_weights, topk_ids, .{}) catch |err| {
            std.debug.panic("generated MoE backend failed: {}", .{err});
        };
    }
};

const RandomGen = struct {
    pub fn generateTensorWithRng(
        shape: Shape,
        seed: u128,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
    ) !zml.Buffer {
        const GenTensor = struct {
            pub fn gen(rand: Tensor.Rng, shape_: Shape) struct { Tensor.Rng, Tensor } {
                return rand.uniform(shape_, .{ .min = -1, .max = 1 });
            }
        };
        const sharding_replicated = try zml.sharding.replicatedSharding(platform);

        var exe = try zml.module.compile(
            allocator,
            io,
            GenTensor.gen,
            .{ Tensor.Rng.init(), shape },
            platform,
            .{ .shardings = &.{sharding_replicated} },
        );
        defer exe.deinit();

        var rng_buffer = try Tensor.Rng.initBuffer(platform, seed, io, sharding_replicated);
        defer rng_buffer._state.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        args.set(.{rng_buffer});

        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        exe.callOpts(io, args, &results, .{ .wait = true });

        var output: zml.Bufferized(stdx.meta.FnResult(GenTensor.gen)) = undefined;
        results.fill(.{&output});
        defer Tensor.Rng.deinitBuffer(&output.@"0");

        return output.@"1";
    }
};

fn createRandomBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: Shape,
    seed: u128,
) !zml.Buffer {
    return try RandomGen.generateTensorWithRng(shape, seed, allocator, io, platform);
}

fn createTopkIdsBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    token_count: i64,
    num_experts: i64,
    topk: i64,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    if (topk > num_experts) return error.InvalidShape;

    const ids_shape = Shape.init(.{ .token = token_count, .topk = topk }, .i32);
    const host = try allocator.alloc(i32, @intCast(token_count * topk));
    defer allocator.free(host);

    // Generate random topk_ids on host (they need to be in range [0, num_experts))
    var prng = std.Random.Pcg.init(9999);
    const random = prng.random();
    for (host) |*val| {
        val.* = @intCast(random.intRangeAtMost(i32, 0, @intCast(num_experts - 1)));
    }

    return zml.Buffer.fromBytes(io, platform, ids_shape, sharding, std.mem.sliceAsBytes(host));
}

fn runCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    m: i64,
    h: i64,
    e: i64,
    n: i64,
    k: i64,
    topk: i64,
) !void {
    const hidden_shape = Shape.init(.{ .token = m, .in = h }, .bf16);
    const w1_shape = Shape.init(.{ .expert = e, .out = n, .in = h }, .bf16);
    const w2_shape = Shape.init(.{ .expert = e, .out = h, .mid = k }, .bf16);
    const topk_weights_shape = Shape.init(.{ .token = m, .topk = topk }, .f32);
    const topk_ids_shape = Shape.init(.{ .token = m, .topk = topk }, .i32);

    var hidden_states = try createRandomBuffer(allocator, io, platform, hidden_shape, 1111);
    defer hidden_states.deinit();
    var w1 = try createRandomBuffer(allocator, io, platform, w1_shape, 2222);
    defer w1.deinit();
    var w2 = try createRandomBuffer(allocator, io, platform, w2_shape, 3333);
    defer w2.deinit();
    var topk_weights = try createRandomBuffer(allocator, io, platform, topk_weights_shape, 4444);
    defer topk_weights.deinit();
    var topk_ids = try createTopkIdsBuffer(allocator, io, platform, m, e, topk, sharding);
    defer topk_ids.deinit();

    std.log.info("general_triton_test: using M={d}", .{m});
    std.log.info("hidden states shapes: {f}", .{hidden_shape});
    std.log.info("weight 1 shapes: {f}", .{w1_shape});
    std.log.info("weight 2 shapes: {f}", .{w2_shape});
    std.log.info("topk shapes: {f}", .{topk_weights_shape});
    std.log.info("expected output shape: M={d}, H={d}", .{ m, h });
    std.log.info("topk ids shapes: {f}", .{topk_ids_shape});

    var exe = try zml.module.compile(
        allocator,
        io,
        Fwd.forward,
        .{
            Tensor.fromShape(hidden_shape),
            Tensor.fromShape(w1_shape),
            Tensor.fromShape(w2_shape),
            Tensor.fromShape(topk_weights_shape),
            Tensor.fromShape(topk_ids_shape),
        },
        platform,
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    // Warmup run 1
    {
        const message = std.fmt.allocPrintSentinel(allocator, "Warmup M={d}", .{m}, 0) catch unreachable;
        defer allocator.free(message);
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        args.set(.{ hidden_states, w1, w2, topk_weights, topk_ids });

        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        // const tracer_id = Tracer.frameStart(message);

        exe.callOpts(io, args, &results, .{ .wait = true });
        // Tracer.frameEnd(tracer_id, message);

        var output1 = results.get(zml.Buffer);
        defer output1.deinit();
    }

    // Warmup run 2
    {
        const message = std.fmt.allocPrintSentinel(allocator, "Run M={d}", .{m}, 0) catch unreachable;
        defer allocator.free(message);

        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        args.set(.{ hidden_states, w1, w2, topk_weights, topk_ids });

        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        // const tracer_id = Tracer.frameStart(message);

        exe.callOpts(io, args, &results, .{ .wait = true });
        // Tracer.frameEnd(tracer_id, message);

        var output2 = results.get(zml.Buffer);
        defer output2.deinit();
    }

    // Timed measurement run
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ hidden_states, w1, w2, topk_weights, topk_ids });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Execution duration [{f}]", .{
        now.untilNow(io, .awake),
    });
    exe.callOpts(io, args, &results, .{ .wait = true });

    var output = results.get(zml.Buffer);
    defer output.deinit();

    std.debug.print("info(general_triton_test): actual output shape: {{{},{},bf16}}\n", .{ output.shape().dim(0), output.shape().dim(1) });
    std.debug.print("info(general_triton_test): run succeeded\n", .{});
    std.debug.print("-\n", .{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    if (platform.target != .cuda) {
        log.warn("platform is not CUDA, skipping execution", .{});
        return;
    }

    const sharding_replicated = try zml.sharding.replicatedSharding(platform);

    const h: i64 = 2048;
    const e: i64 = 256;
    const n: i64 = 1024;
    const k: i64 = 512;
    const topk: i64 = 8;

    const m_values = [_]i64{ 1, 8, 16, 32, 64, 128, 256, 512, 1024 };

    for (m_values) |m| {
        runCase(allocator, io, platform, sharding_replicated, m, h, e, n, k, topk) catch |err| {
            log.err("failed to run case with M={}: {}", .{ m, err });
        };
    }
}
