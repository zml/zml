const std = @import("std");

const zml = @import("zml");
const router = @import("kimi_k3/router.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(300_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    weights: []const u8,
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_router_tests --weights=<S4-directory> --fixture=<router-reference.safetensors>
        \\
        \\Run real and adversarial Kimi K3 router parity on NVIDIA CUDA only.
        \\
    ;
};

const strict_fp32: zml.testing.CompareOpts = .{
    .absolute_tolerance = 1e-5,
    .relative_tolerance = 1e-5,
    .minimum_close_fraction = 1.0,
};

fn compareCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    name: []const u8,
    actual: zml.Bufferized(router.Result),
    sharding: zml.Sharding,
) !void {
    const boundaries = [_][]const u8{
        "logits",
        "raw_scores",
        "selection_scores",
        "topk_raw_weights",
        "topk_weights",
    };
    const values = .{
        actual.logits,
        actual.raw_scores,
        actual.selection_scores,
        actual.topk_raw_weights,
        actual.topk_weights,
    };
    inline for (boundaries, values) |boundary, value| {
        const key = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ name, boundary });
        defer allocator.free(key);
        try support.compare(allocator, io, platform, store, key, value, strict_fp32, sharding);
    }
    const ids_key = try std.fmt.allocPrint(allocator, "{s}.topk_ids", .{name});
    defer allocator.free(ids_key);
    try support.compare(allocator, io, platform, store, ids_key, actual.topk_ids, .{}, sharding);
}

fn runCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    name: []const u8,
    hidden: zml.Buffer,
    weights: zml.Bufferized(router.Weights),
    config: router.Config,
    sharding: zml.Sharding,
) !void {
    const weight_tensors: router.Weights = .{
        .weight = .fromShape(weights.weight.shape()),
        .correction_bias = .fromShape(weights.correction_bias.shape()),
    };
    const exe = try platform.compileFn(
        allocator,
        io,
        router.forward,
        .{ zml.Tensor.fromShape(hidden.shape()), weight_tensors, config },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();
    const started = std.Io.Clock.now(.real, io).toNanoseconds();
    var actual = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        router.forward,
        .{ hidden, weights },
    );
    defer zml.Buffer.deinitAll(router.Result, &actual);
    const elapsed_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - started, 1000);
    try compareCase(allocator, io, platform, store, name, actual, sharding);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    // KIMI_K3_TEMP_REMOVE_M20: aligned route timing and boundary inventory are
    // bring-up diagnostics removed from the production route planner in M20.
    try stdout_file.interface.print(
        "KIMI_K3_ROUTER_PASS case={s} boundaries=6 tokens={} experts={} top_k={} elapsed_us={}\n",
        .{ name, hidden.shape().dim(.b) * hidden.shape().dim(.s), weights.weight.shape().dim(.expert), config.top_k, elapsed_us },
    );
    try stdout_file.interface.flush();
}

fn loadSyntheticWeights(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    name: []const u8,
    sharding: zml.Sharding,
) !zml.Bufferized(router.Weights) {
    const weight_key = try std.fmt.allocPrint(allocator, "{s}.weight", .{name});
    defer allocator.free(weight_key);
    const bias_key = try std.fmt.allocPrint(allocator, "{s}.correction_bias", .{name});
    defer allocator.free(bias_key);
    return .{
        .weight = try support.loadBuffer(allocator, io, platform, store, weight_key, .{ .expert, .d }, sharding),
        .correction_bias = try support.loadBuffer(allocator, io, platform, store, bias_key, .{.expert}, sharding),
    };
}

fn loadHidden(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    name: []const u8,
    sharding: zml.Sharding,
) !zml.Buffer {
    const key = try std.fmt.allocPrint(allocator, "{s}.hidden", .{name});
    defer allocator.free(key);
    return support.loadBuffer(allocator, io, platform, store, key, .{ .b, .s, .d }, sharding);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.75 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    const sharding = platform.replicated_sharding;

    var fixture_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer fixture_registry.deinit();
    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();
    const fixture = fixture_store.view();

    const synthetic = .{
        .{ "tie", router.Config{ .top_k = 16 } },
        .{ "bias", router.Config{ .top_k = 16 } },
        .{ "grouped", router.Config{ .top_k = 4, .num_expert_group = 4, .topk_group = 2, .routed_scaling_factor = 1.25 } },
    };
    inline for (synthetic) |case| {
        var hidden = try loadHidden(allocator, io, platform, fixture, case[0], sharding);
        defer hidden.deinit();
        var weights = try loadSyntheticWeights(allocator, io, platform, fixture, case[0], sharding);
        defer zml.Buffer.deinitAll(router.Weights, &weights);
        try runCase(allocator, io, platform, fixture, case[0], hidden, weights, case[1], sharding);
    }

    var weight_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.weights);
    defer weight_registry.deinit();
    var weight_store: zml.io.TensorStore = .fromRegistry(allocator, &weight_registry);
    defer weight_store.deinit();
    const real_weights = router.Weights.init(
        weight_store.view().withPrefix("language_model.model.layers.1.block_sparse_moe.gate"),
    );
    var real_weight_buffers = try zml.mem.bufferize(allocator, router.Weights, &real_weights);
    defer zml.Buffer.deinitAll(router.Weights, &real_weight_buffers);
    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 64 * zml.MiB,
    });
    defer loader.deinit();
    loader.load(io, router.Weights, &real_weights, &real_weight_buffers, &weight_store, &.{sharding}, .{});
    try loader.await(io);
    var real_hidden = try loadHidden(allocator, io, platform, fixture, "real", sharding);
    defer real_hidden.deinit();
    try runCase(
        allocator,
        io,
        platform,
        fixture,
        "real",
        real_hidden,
        real_weight_buffers,
        .{ .top_k = 16 },
        sharding,
    );
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    try stdout_file.interface.print(
        "KIMI_K3_ROUTER_ALL_PASS cases=4 real_tokens=4 exact_sets=true backend=cuda\n",
        .{},
    );
    try stdout_file.interface.flush();
}
