const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");
const layer = @import("kimi_k3/layer.zig");

comptime {
    @setEvalBranchQuota(300_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    weights: []const u8,
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_layer0_tests --weights=<shard-1.safetensors> --fixture=<s1-layer0-lenN.safetensors>
        \\
        \\Run real-weight Kimi K3 layer-0 parity on NVIDIA CUDA only.
        \\
    ;
};

pub const bf16_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 5e-2,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 0.995,
};

pub const state_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 5e-3,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 0.995,
};

pub fn loadBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    key: []const u8,
    tags: anytype,
    sharding: zml.Sharding,
) !zml.Buffer {
    const shape = store.getShape(key) orelse return error.MissingLayer0Fixture;
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(bytes);
    return zml.Buffer.fromBytes(io, platform, shape.withTags(tags), sharding, bytes);
}

pub fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, bytes);
}

pub fn compare(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    key: []const u8,
    actual: zml.Buffer,
    opts: zml.testing.CompareOpts,
    sharding: zml.Sharding,
) !void {
    const shape = store.getShape(key) orelse return error.MissingLayer0Expected;
    var expected = try loadBuffer(allocator, io, platform, store, key, shape.tags(), sharding);
    defer expected.deinit();
    try zml.testing.expectClose(io, actual, expected, opts);
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
    const weights = layer.Layer0Weights.init(weight_store.view());
    var weight_buffers = try zml.mem.bufferize(allocator, layer.Layer0Weights, &weights);
    defer zml.Buffer.deinitAll(layer.Layer0Weights, &weight_buffers);
    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 256 * zml.MiB,
    });
    defer loader.deinit();
    const load_started = std.Io.Clock.now(.real, io).toNanoseconds();
    loader.load(io, layer.Layer0Weights, &weights, &weight_buffers, &weight_store, &.{sharding}, .{});
    try loader.await(io);
    const load_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - load_started, 1000);

    var fixture_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer fixture_registry.deinit();
    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();
    var input = try loadBuffer(allocator, io, platform, fixture_store.view(), "layers.0.input", .{ .b, .s, .d }, sharding);
    defer input.deinit();
    const input_tensor = zml.Tensor.fromShape(input.shape());
    const conv_shape = zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16);
    const state_shape = zml.Shape.init(.{ .b = 1, .h = 96, .v = 128, .k = 128 }, .f32);
    var cache: zml.Bufferized(kda.Cache) = .{
        .q_conv = try zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .k_conv = try zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .v_conv = try zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .recurrent_state = try zeroBuffer(allocator, io, platform, state_shape, sharding),
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
        layer.forwardLayer0,
        .{ input_tensor, weights, cache_tensors },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();
    const compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - compile_started, 1000);
    const execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var actual = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        layer.forwardLayer0,
        .{ input, weight_buffers, cache },
    );
    defer zml.Buffer.deinitAll(layer.Layer0Result, &actual);
    const execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - execute_started, 1000);

    try compare(allocator, io, platform, fixture_store.view(), "layers.0.input_layernorm.out", actual.input_norm, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.kda.out", actual.kda_output, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.attnres.block_residual.out", actual.block_residual, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.post_attention_layernorm.out", actual.post_attention_norm, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.mlp.gate_proj.out", actual.mlp_gate, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.mlp.up_proj.out", actual.mlp_up, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.mlp.situ.out", actual.mlp_situ, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.mlp.out", actual.mlp_output, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.out", actual.output, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.kda.conv_state.0.out", actual.cache.q_conv, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.kda.conv_state.1.out", actual.cache.k_conv, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.kda.conv_state.2.out", actual.cache.v_conv, bf16_tolerance, sharding);
    try compare(allocator, io, platform, fixture_store.view(), "layers.0.kda.recurrent_state.out", actual.cache.recurrent_state, state_tolerance, sharding);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    // KIMI_K3_TEMP_REMOVE_M20: real-layer boundary inventory and synchronized
    // load/compile/execute timing are Gate A diagnostics removed in cleanup.
    try stdout_file.interface.print(
        "KIMI_K3_LAYER0_PASS boundaries=13 load_us={} compile_us={} execute_us={} input={f} output={f}\n",
        .{ load_us, compile_us, execute_us, input.shape(), actual.output.shape() },
    );
    try stdout_file.interface.flush();
}
