const std = @import("std");

const zml = @import("zml");
const runtime_weights = @import("kimi_k3/runtime_weights.zig");

comptime {
    @setEvalBranchQuota(500_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    expected_devices: usize = 4,

    pub const help =
        \\Use kimi_k3_distributed_tests [--expected-devices=4]
        \\
        \\Run the Kimi K3 physical CUDA topology and deterministic collective
        \\preflight. Exactly four visible devices are required for Milestone 22.
        \\
    ;
};

const CollectiveResult = struct {
    all_reduce: zml.Tensor,
    all_gather: zml.Tensor,
    reduce_scatter: zml.Tensor,
    broadcast: zml.Tensor,
};

fn collectiveForward(input: zml.Tensor, weight: zml.Tensor) CollectiveResult {
    // Reducing a model-sharded dimension to a replicated scalar makes the
    // SPMD partitioner emit the supported cross-device reduction.
    const all_reduce = input.sum(.item).withPartitioning(.{ .item = .replicated });

    const all_gather = input.withPartitioning(.{ .item = .replicated });

    // A dot product sharded on its contracting axis produces partial sums;
    // sharding the output on a different axis requires a real reduce-scatter.
    const reduce_scatter = input.rename(.{ .item = .k }).dot(weight, .k)
        .withPartitioning(.{ .out = .model });

    const broadcast = zml.Tensor.scalar(7.0, .f32)
        .broad(input.shape())
        .withPartitioning(.{ .item = .replicated });

    return .{
        .all_reduce = all_reduce,
        .all_gather = all_gather,
        .reduce_scatter = reduce_scatter,
        .broadcast = broadcast,
    };
}

fn expectPlan(device_count: usize, tensor_parallel: usize, expert_parallel: usize, experts_per_rank: usize) !void {
    const plan = try runtime_weights.DistributedPlan.init(device_count, tensor_parallel);
    if (plan.device_count != device_count or
        plan.tensor_parallel != tensor_parallel or
        plan.expert_parallel != expert_parallel)
    {
        return error.KimiK3DistributedPlanMismatch;
    }

    var next_expert: usize = 0;
    for (0..plan.expert_parallel) |rank| {
        const partition = try plan.expertPartition(rank);
        if (partition.rank != rank or partition.ranks != expert_parallel or
            partition.first != next_expert or partition.count() != experts_per_rank)
        {
            return error.KimiK3ExpertOwnershipMismatch;
        }
        next_expert = partition.end;
    }
    if (next_expert != runtime_weights.expert_count) return error.KimiK3ExpertCoverageMismatch;
}

fn expectInvalidPlans() !void {
    if (runtime_weights.DistributedPlan.init(4, 3)) |_| {
        return error.KimiK3InvalidPlanAccepted;
    } else |err| if (err != error.InvalidKimiK3ParallelPlan) {
        return err;
    }
    if (runtime_weights.DistributedPlan.init(4, 0)) |_| {
        return error.KimiK3InvalidPlanAccepted;
    } else |err| if (err != error.InvalidKimiK3ParallelPlan) {
        return err;
    }
    if (runtime_weights.DistributedPlan.init(0, 1)) |_| {
        return error.KimiK3InvalidPlanAccepted;
    } else |err| if (err != error.InvalidKimiK3ParallelPlan) {
        return err;
    }
}

fn awaitCollectiveResult(io: std.Io, result: anytype) !void {
    try result.all_reduce.await(io);
    try result.all_gather.await(io);
    try result.reduce_scatter.await(io);
    try result.broadcast.await(io);
}

fn hostItems(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) !zml.Slice {
    const slice = try buffer.toSliceAlloc(allocator, io);
    if (slice.shape.dtype() != .f32) {
        slice.free(allocator);
        return error.KimiK3CollectiveDtypeMismatch;
    }
    return slice;
}

fn expectConstant(items: []const f32, expected: f32) !void {
    for (items) |actual| {
        if (actual != expected) return error.KimiK3CollectiveValueMismatch;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.expected_devices != 4) return error.KimiK3Milestone22RequiresFourDevices;

    try expectPlan(4, 4, 1, 896);
    try expectPlan(4, 2, 2, 448);
    try expectPlan(4, 1, 4, 224);
    try expectInvalidPlans();

    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.05 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    if (platform.devices.len != args.expected_devices or
        platform.physical_mesh.devices_in_canonical_order.len != args.expected_devices)
    {
        return error.KimiK3CudaDeviceCountMismatch;
    }

    for (platform.devices, 0..) |device, index| {
        if (device.localHardwareId() != index) return error.KimiK3CudaDeviceOrderMismatch;
    }

    const sharding = try platform.registerSharding(
        "kimi_k3_m22_model",
        .mesh(.{ .model = .high_bandwidth }),
    );
    if (@as(usize, @intCast(sharding.data.numPartitions())) != args.expected_devices) return error.KimiK3ShardingPartitionCountMismatch;

    const input_shape = zml.Shape.init(.{ .item = 16 }, .f32)
        .withPartitioning(.{ .item = .model });
    const weight_shape = zml.Shape.init(.{ .k = 16, .out = 16 }, .f32)
        .withPartitioning(.{ .k = .model, .out = .replicated });
    const input_tensor = zml.Tensor.fromShape(input_shape);
    const weight_tensor = zml.Tensor.fromShape(weight_shape);
    const exe = try platform.compileFn(
        allocator,
        io,
        collectiveForward,
        .{ input_tensor, weight_tensor },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();

    var host_input = try zml.Slice.alloc(allocator, input_shape);
    defer host_input.free(allocator);
    for (host_input.items(f32), 0..) |*value, index| value.* = @floatFromInt(index + 1);
    var input_buffer = try zml.Buffer.fromSlice(io, platform, host_input, sharding);
    defer input_buffer.deinit();

    var host_weight = try zml.Slice.alloc(allocator, weight_shape);
    defer host_weight.free(allocator);
    @memset(host_weight.items(f32), 1.0);
    var weight_buffer = try zml.Buffer.fromSlice(io, platform, host_weight, sharding);
    defer weight_buffer.deinit();
    if (input_buffer.numShards() != args.expected_devices or
        weight_buffer.numShards() != args.expected_devices)
    {
        return error.KimiK3InputShardCountMismatch;
    }

    var warm = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        collectiveForward,
        .{ input_buffer, weight_buffer },
    );
    defer zml.Buffer.deinitAll(CollectiveResult, &warm);
    try awaitCollectiveResult(io, warm);

    const collective_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var result = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        collectiveForward,
        .{ input_buffer, weight_buffer },
    );
    defer zml.Buffer.deinitAll(CollectiveResult, &result);
    try awaitCollectiveResult(io, result);
    const collective_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - collective_started, 1000);
    if (result.all_reduce.numShards() != args.expected_devices or
        result.all_gather.numShards() != args.expected_devices or
        result.reduce_scatter.numShards() != args.expected_devices or
        result.broadcast.numShards() != args.expected_devices)
    {
        return error.KimiK3CollectiveShardCountMismatch;
    }

    var all_reduce = try hostItems(allocator, io, result.all_reduce);
    defer all_reduce.free(allocator);
    try expectConstant(all_reduce.items(f32), 136.0);

    var all_gather = try hostItems(allocator, io, result.all_gather);
    defer all_gather.free(allocator);
    for (all_gather.items(f32), 0..) |actual, index| {
        const expected: f32 = @floatFromInt(index + 1);
        if (actual != expected) return error.KimiK3AllGatherMismatch;
    }

    var reduce_scatter = try hostItems(allocator, io, result.reduce_scatter);
    defer reduce_scatter.free(allocator);
    try expectConstant(reduce_scatter.items(f32), 136.0);

    var broadcast = try hostItems(allocator, io, result.broadcast);
    defer broadcast.free(allocator);
    try expectConstant(broadcast.items(f32), 7.0);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    try stdout_file.interface.print(
        "KIMI_K3_DISTRIBUTED_PASS backend=cuda devices={} partitions={} " ++
            "logical_layouts=tp4_ep1,tp2_ep2,tp1_ep4 physical_layout=tp4_ep1 collectives=all_reduce,all_gather,reduce_scatter,broadcast " ++
            "expert_ranges=896,448,224 collective_us={} logical_collective_payload_bytes=132 " ++
            "estimated_ring_wire_bytes_all_ranks=408 timed_host_transfers=0\n",
        .{ platform.devices.len, @as(usize, @intCast(sharding.data.numPartitions())), collective_us },
    );
    try stdout_file.interface.flush();
}
