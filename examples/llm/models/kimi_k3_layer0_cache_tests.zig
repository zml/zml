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
        \\Use kimi_k3_layer0_cache_tests --weights=<S1-directory> --fixture=<prefill4-decode1.safetensors>
        \\
        \\Run real-weight Kimi K3 layer-0 prefill/cache/decode parity on NVIDIA CUDA only.
        \\
    ;
};

fn compareLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    prefix: []const u8,
    actual: zml.Bufferized(layer.Layer0Result),
    sharding: zml.Sharding,
) !void {
    const names = [_][]const u8{
        "layers.0.input_layernorm.out",
        "layers.0.kda.out",
        "layers.0.attnres.block_residual.out",
        "layers.0.post_attention_layernorm.out",
        "layers.0.mlp.gate_proj.out",
        "layers.0.mlp.up_proj.out",
        "layers.0.mlp.situ.out",
        "layers.0.mlp.out",
        "layers.0.out",
    };
    const values = .{
        actual.input_norm,
        actual.kda_output,
        actual.block_residual,
        actual.post_attention_norm,
        actual.mlp_gate,
        actual.mlp_up,
        actual.mlp_situ,
        actual.mlp_output,
        actual.output,
    };
    inline for (names, values) |name, value| {
        const key = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ prefix, name });
        defer allocator.free(key);
        try support.compare(
            allocator,
            io,
            platform,
            store,
            key,
            value,
            support.bf16_tolerance,
            sharding,
        );
    }
}

fn compareCache(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    prefix: []const u8,
    cache: zml.Bufferized(kda.Cache),
    suffix: []const u8,
    sharding: zml.Sharding,
) !void {
    const conv = .{ cache.q_conv, cache.k_conv, cache.v_conv };
    inline for (conv, 0..) |value, index| {
        const key = try std.fmt.allocPrint(
            allocator,
            "{s}.cache.conv_state.{}.{s}",
            .{ prefix, index, suffix },
        );
        defer allocator.free(key);
        try support.compare(
            allocator,
            io,
            platform,
            store,
            key,
            value,
            support.bf16_tolerance,
            sharding,
        );
    }
    const state_key = try std.fmt.allocPrint(
        allocator,
        "{s}.cache.recurrent_state.{s}",
        .{ prefix, suffix },
    );
    defer allocator.free(state_key);
    try support.compare(
        allocator,
        io,
        platform,
        store,
        state_key,
        cache.recurrent_state,
        support.state_tolerance,
        sharding,
    );
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
    loader.load(io, layer.Layer0Weights, &weights, &weight_buffers, &weight_store, &.{sharding}, .{});
    try loader.await(io);

    var fixture_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer fixture_registry.deinit();
    var fixture_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer fixture_store.deinit();

    var prefill_input = try support.loadBuffer(
        allocator,
        io,
        platform,
        fixture_store.view(),
        "prefill.layers.0.input",
        .{ .b, .s, .d },
        sharding,
    );
    defer prefill_input.deinit();
    const conv_shape = zml.Shape.init(.{ .b = 1, .channel = 12288, .kernel = 4 }, .bf16);
    const state_shape = zml.Shape.init(.{ .b = 1, .h = 96, .v = 128, .k = 128 }, .f32);
    var zero_cache: zml.Bufferized(kda.Cache) = .{
        .q_conv = try support.zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .k_conv = try support.zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .v_conv = try support.zeroBuffer(allocator, io, platform, conv_shape, sharding),
        .recurrent_state = try support.zeroBuffer(allocator, io, platform, state_shape, sharding),
    };
    defer zml.Buffer.deinitAll(kda.Cache, &zero_cache);
    const cache_tensors: kda.Cache = .{
        .q_conv = .fromShape(zero_cache.q_conv.shape()),
        .k_conv = .fromShape(zero_cache.k_conv.shape()),
        .v_conv = .fromShape(zero_cache.v_conv.shape()),
        .recurrent_state = .fromShape(zero_cache.recurrent_state.shape()),
    };

    const prefill_exe = try platform.compileFn(
        allocator,
        io,
        layer.forwardLayer0,
        .{ zml.Tensor.fromShape(prefill_input.shape()), weights, cache_tensors },
        .{ .shardings = &.{sharding} },
    );
    defer prefill_exe.deinit();
    const prefill_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var prefill = try zml.testing.autoCall(
        allocator,
        io,
        &prefill_exe,
        layer.forwardLayer0,
        .{ prefill_input, weight_buffers, zero_cache },
    );
    defer zml.Buffer.deinitAll(layer.Layer0Result, &prefill);
    const prefill_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - prefill_started, 1000);
    try compareLayer(allocator, io, platform, fixture_store.view(), "prefill", prefill, sharding);
    try compareCache(allocator, io, platform, fixture_store.view(), "prefill", prefill.cache, "out", sharding);
    // The same ZML buffers are checked against Moonshot's explicit decode inputs,
    // proving that no host-side cache transformation occurs at the handoff.
    try compareCache(allocator, io, platform, fixture_store.view(), "decode", prefill.cache, "in", sharding);

    var decode_input = try support.loadBuffer(
        allocator,
        io,
        platform,
        fixture_store.view(),
        "decode.layers.0.input",
        .{ .b, .s, .d },
        sharding,
    );
    defer decode_input.deinit();
    const decode_exe = try platform.compileFn(
        allocator,
        io,
        layer.forwardLayer0,
        .{ zml.Tensor.fromShape(decode_input.shape()), weights, cache_tensors },
        .{ .shardings = &.{sharding} },
    );
    defer decode_exe.deinit();
    const decode_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var decode = try zml.testing.autoCall(
        allocator,
        io,
        &decode_exe,
        layer.forwardLayer0,
        .{ decode_input, weight_buffers, prefill.cache },
    );
    defer zml.Buffer.deinitAll(layer.Layer0Result, &decode);
    const decode_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - decode_started, 1000);
    try compareLayer(allocator, io, platform, fixture_store.view(), "decode", decode, sharding);
    try compareCache(allocator, io, platform, fixture_store.view(), "decode", decode.cache, "out", sharding);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    // KIMI_K3_TEMP_REMOVE_M20: synchronized phase timings are Gate A
    // diagnostics and must be removed from the production hot path in M20.
    try stdout_file.interface.print(
        "KIMI_K3_LAYER0_CACHE_PASS boundaries=30 prefill_us={} decode_us={}\n",
        .{ prefill_us, decode_us },
    );
    try stdout_file.interface.flush();
}
