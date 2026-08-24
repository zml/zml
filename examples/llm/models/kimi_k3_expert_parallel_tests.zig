const std = @import("std");

const zml = @import("zml");
const common = @import("common.zig");
const moe = @import("kimi_k3/moe.zig");

pub const std_options: std.Options = .{ .log_level = .info };

fn forwardRouted(
    input: zml.Tensor,
    expert_ids: zml.Tensor,
    route_weights: zml.Tensor,
    experts: moe.ExpertBank,
) moe.RoutedExpertResult {
    return moe.routedExpertsForTest(input, expert_ids, route_weights, experts);
}

fn expertShape(shape: zml.Shape, partitioned: bool) zml.Shape {
    return if (partitioned) shape.withPartitioning(.{ .expert = .experts }) else shape;
}

fn filledBytes(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    value: u8,
) !zml.Buffer {
    var host = try zml.Slice.alloc(allocator, shape);
    defer host.free(allocator);
    @memset(host.items(u8), value);
    return zml.Buffer.fromSlice(io, platform, host, sharding);
}

fn filledF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    value: f32,
) !zml.Buffer {
    var host = try zml.Slice.alloc(allocator, shape);
    defer host.free(allocator);
    @memset(host.items(f32), value);
    return zml.Buffer.fromSlice(io, platform, host, sharding);
}
fn filledBf16(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    value: f32,
) !zml.Buffer {
    var host = try zml.Slice.alloc(allocator, shape);
    defer host.free(allocator);
    @memset(host.items(zml.floats.BFloat16), zml.floats.BFloat16.fromF32(value));
    return zml.Buffer.fromSlice(io, platform, host, sharding);
}

fn expertIds(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    values: *const [16]i32,
) !zml.Buffer {
    const shape = zml.Shape.init(.{ .token = 4, .route = 4 }, .i32);
    var host = try zml.Slice.alloc(allocator, shape);
    defer host.free(allocator);
    @memcpy(host.items(i32), values);
    return zml.Buffer.fromSlice(io, platform, host, platform.replicated_sharding);
}

fn createExperts(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    expert_sharding: zml.Sharding,
    partitioned: bool,
) !zml.Bufferized(moe.ExpertBank) {
    const sharding = if (partitioned) expert_sharding else platform.replicated_sharding;
    const values_shape = expertShape(zml.Shape.init(.{ .expert = 896, .intermediate = 64, .kw = 32 }, .u8), partitioned);
    const scales_shape = expertShape(zml.Shape.init(.{ .expert = 896, .intermediate = 64, .block = 2 }, .u8), partitioned);
    const down_values_shape = expertShape(zml.Shape.init(.{ .expert = 896, .latent = 64, .kw = 32 }, .u8), partitioned);
    const down_scales_shape = expertShape(zml.Shape.init(.{ .expert = 896, .latent = 64, .block = 2 }, .u8), partitioned);

    var result: zml.Bufferized(moe.ExpertBank) = undefined;
    result.w1.values = try filledBytes(allocator, io, platform, values_shape, sharding, 0x11);
    errdefer result.w1.values.deinit();
    result.w1.scale = try filledBytes(allocator, io, platform, scales_shape, sharding, 127);
    errdefer result.w1.scale.deinit();
    result.w2.values = try filledBytes(allocator, io, platform, down_values_shape, sharding, 0x11);
    errdefer result.w2.values.deinit();
    result.w2.scale = try filledBytes(allocator, io, platform, down_scales_shape, sharding, 127);
    errdefer result.w2.scale.deinit();
    result.w3.values = try filledBytes(allocator, io, platform, values_shape, sharding, 0x11);
    errdefer result.w3.values.deinit();
    result.w3.scale = try filledBytes(allocator, io, platform, scales_shape, sharding, 127);
    return result;
}

fn expertTensors(buffers: *const zml.Bufferized(moe.ExpertBank)) moe.ExpertBank {
    return .{
        .w1 = .{ .values = .fromShape(buffers.w1.values.shape()), .scale = .fromShape(buffers.w1.scale.shape()) },
        .w2 = .{ .values = .fromShape(buffers.w2.values.shape()), .scale = .fromShape(buffers.w2.scale.shape()) },
        .w3 = .{ .values = .fromShape(buffers.w3.values.shape()), .scale = .fromShape(buffers.w3.scale.shape()) },
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.20 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda or platform.devices.len != 4) return error.KimiK3ExpertParallelTestRequiresFourCudaDevices;
    const shardings: common.Shardings = try .init(platform);
    const compilation_shardings = shardings.all();

    const input_shape = zml.Shape.init(.{ .token = 4, .d = 64 }, .bf16);
    const ids_shape = zml.Shape.init(.{ .token = 4, .route = 4 }, .i32);
    const route_weights_shape = zml.Shape.init(.{ .token = 4, .route = 4 }, .f32);
    var input = try filledBf16(allocator, io, platform, input_shape, platform.replicated_sharding, 1.0);
    defer input.deinit();
    var route_weights = try filledF32(allocator, io, platform, route_weights_shape, platform.replicated_sharding, 0.25);
    defer route_weights.deinit();

    const boundary_values = [16]i32{ 223, 224, 447, 448, 671, 672, 895, 223, 0, 1, 2, 3, 224, 225, 226, 227 };
    const rank0_only_values = [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    var boundary_ids = try expertIds(allocator, io, platform, &boundary_values);
    defer boundary_ids.deinit();
    var rank0_only_ids = try expertIds(allocator, io, platform, &rank0_only_values);
    defer rank0_only_ids.deinit();

    var replicated_experts = try createExperts(allocator, io, platform, shardings.experts, false);
    defer zml.Buffer.deinitAll(moe.ExpertBank, &replicated_experts);
    var partitioned_experts = try createExperts(allocator, io, platform, shardings.experts, true);
    defer zml.Buffer.deinitAll(moe.ExpertBank, &partitioned_experts);
    if (partitioned_experts.w1.values.numShards() != 4 or partitioned_experts.w2.values.numShards() != 4 or partitioned_experts.w3.values.numShards() != 4) {
        return error.KimiK3ExpertParallelRuntimeShardCountMismatch;
    }

    const replicated_exe = try platform.compileFn(
        allocator,
        io,
        forwardRouted,
        .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(route_weights_shape),
            expertTensors(&replicated_experts),
        },
        .{ .shardings = &compilation_shardings },
    );
    defer replicated_exe.deinit();
    const partitioned_exe = try platform.compileFn(
        allocator,
        io,
        forwardRouted,
        .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(ids_shape),
            zml.Tensor.fromShape(route_weights_shape),
            expertTensors(&partitioned_experts),
        },
        .{ .shardings = &compilation_shardings },
    );
    defer partitioned_exe.deinit();

    inline for (.{ boundary_ids, rank0_only_ids }) |ids| {
        var replicated = try zml.testing.autoCall(
            allocator,
            io,
            &replicated_exe,
            forwardRouted,
            .{ input, ids, route_weights, replicated_experts },
        );
        defer zml.Buffer.deinitAll(moe.RoutedExpertResult, &replicated);
        var partitioned = try zml.testing.autoCall(
            allocator,
            io,
            &partitioned_exe,
            forwardRouted,
            .{ input, ids, route_weights, partitioned_experts },
        );
        defer zml.Buffer.deinitAll(moe.RoutedExpertResult, &partitioned);
        try zml.testing.expectClose(io, replicated.route_outputs, partitioned.route_outputs, .exact_match);
        try zml.testing.expectClose(io, replicated.combined_latent_f32, partitioned.combined_latent_f32, .exact_match);
    }

    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try stdout.interface.writeAll(
        "KIMI_K3_EXPERT_PARALLEL_PASS ranks=4 local_experts=224 " ++
            "boundaries=223/224,447/448,671/672,895 same_rank_routes=true " ++
            "empty_ranks=true all_ranks=true replicated_oracle=exact runtime_shards=4\n",
    );
    try stdout.interface.flush();
}
