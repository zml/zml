const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");
const layer = @import("kimi_k3/layer.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(300_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    weights: []const u8,
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_prefix_tests --weights=<S2-directory> --fixture=<s2-prefix.safetensors>
        \\
        \\Run embedding through one-layer diagnostic logits on NVIDIA CUDA only.
        \\
    ;
};

const selector_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 2e-4,
    .relative_tolerance = 2e-3,
    .minimum_close_fraction = 1.0,
};

fn comparePrefix(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    actual: zml.Bufferized(layer.PrefixResult),
    sharding: zml.Sharding,
) !void {
    try support.compare(allocator, io, platform, store, "prefix.embedding.out", actual.embedding, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.layer0.out", actual.layer_output, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.layer0.block_residual.out", actual.block_residual, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.layer0.cache.conv_state.0.out", actual.cache.q_conv, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.layer0.cache.conv_state.1.out", actual.cache.k_conv, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.layer0.cache.conv_state.2.out", actual.cache.v_conv, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.layer0.cache.recurrent_state.out", actual.cache.recurrent_state, support.state_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.output_attn_res.candidates", actual.output_candidates, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.output_attn_res.weights", actual.output_selector_weights, selector_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.output_attn_res.out", actual.output_selected, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.final_norm.out", actual.final_norm, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.logits", actual.logits, support.bf16_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "prefix.greedy_token", actual.greedy_token, .{}, sharding);
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

    var weight_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.weights);
    defer weight_registry.deinit();
    var weight_store: zml.io.TensorStore = .fromRegistry(allocator, &weight_registry);
    defer weight_store.deinit();
    const weights = layer.PrefixWeights.init(weight_store.view());
    var weight_buffers = try zml.mem.bufferize(allocator, layer.PrefixWeights, &weights);
    defer zml.Buffer.deinitAll(layer.PrefixWeights, &weight_buffers);
    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 512 * zml.MiB,
    });
    defer loader.deinit();
    const load_started = std.Io.Clock.now(.real, io).toNanoseconds();
    loader.load(io, layer.PrefixWeights, &weights, &weight_buffers, &weight_store, &.{sharding}, .{});
    try loader.await(io);
    const load_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - load_started, 1000);

    var fixture_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer fixture_registry.deinit();
    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();
    var tokens = try support.loadBuffer(
        allocator,
        io,
        platform,
        fixture_store.view(),
        "prefix.token_ids",
        .{ .b, .s },
        sharding,
    );
    defer tokens.deinit();
    const conv_shape = zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16);
    const state_shape = zml.Shape.init(.{ .b = 1, .h = 96, .v = 128, .k = 128 }, .f32);
    var cache: zml.Bufferized(kda.Cache) = .{
        .q_conv = try support.zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .k_conv = try support.zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .v_conv = try support.zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .recurrent_state = try support.zeroBuffer(allocator, io, platform, state_shape, sharding),
    };
    defer zml.Buffer.deinitAll(kda.Cache, &cache);
    const cache_tensors: kda.Cache = .{
        .q_conv = .fromShape(cache.q_conv.shape()),
        .k_conv = .fromShape(cache.k_conv.shape()),
        .v_conv = .fromShape(cache.v_conv.shape()),
        .recurrent_state = .fromShape(cache.recurrent_state.shape()),
    };

    const compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    const exe = try platform.compileFn(
        allocator,
        io,
        layer.forwardPrefix,
        .{ zml.Tensor.fromShape(tokens.shape()), weights, cache_tensors },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();
    const compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - compile_started, 1000);
    const execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var actual = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        layer.forwardPrefix,
        .{ tokens, weight_buffers, cache },
    );
    defer zml.Buffer.deinitAll(layer.PrefixResult, &actual);
    const execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - execute_started, 1000);
    try comparePrefix(allocator, io, platform, fixture_store.view(), actual, sharding);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    // KIMI_K3_TEMP_REMOVE_M20: full-prefix boundary inventory and synchronized
    // timings are Gate A diagnostics removed from the production hot path.
    try stdout_file.interface.print(
        "KIMI_K3_PREFIX_PASS boundaries=13 load_us={} compile_us={} execute_us={} logits={f}\n",
        .{ load_us, compile_us, execute_us, actual.logits.shape() },
    );
    try stdout_file.interface.flush();
}
