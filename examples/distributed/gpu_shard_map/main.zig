//! Runnable ZML equivalent of ../../../../gpu_shard_map.py.

const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_host = 4;
const features_per_gpu = 8;

const Statistics = struct {
    partial_sums: zml.Tensor,
    global_sum: zml.Tensor,
};

const ShardStatistics = struct {
    pub fn forward(input: zml.Tensor) Statistics {
        const partial_shape = zml.Shape.init(.{ .partition = 4 }, .f32)
            .withPartitioning(.{ .partition = .data });
        const mapped = zml.ops.manualComputation(
            input,
            [2]zml.Shape{ partial_shape, .scalar(.f32) },
            {},
            (struct {
                fn body(
                    _: void,
                    allocator: std.mem.Allocator,
                    local_input: zml.Tensor,
                    local_outputs: []const zml.Shape,
                ) []const zml.Tensor {
                    const local_sum = local_input
                        .sum(.rows)
                        .sum(.features)
                        .reshape(local_outputs[0]);
                    const global_sum = zml.ops.allReduceAxes(
                        local_sum,
                        .{.data},
                        zml.Tensor.add,
                    ).reshape(local_outputs[1]);
                    const outputs = allocator.alloc(zml.Tensor, 2) catch
                        unreachable;
                    outputs[0] = local_sum;
                    outputs[1] = global_sum;
                    return outputs;
                }
            }).body,
        );
        return .{
            .partial_sums = mapped[0],
            .global_sum = mapped[1],
        };
    }
};

fn verifyInput(
    allocator: std.mem.Allocator,
    io: std.Io,
    features: usize,
    input: *const zml.Buffer,
) !void {
    var shards = input.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2 or slices.get(0).size != 2 or
            slices.get(1).start != 0 or
            slices.get(1).size != @as(i64, @intCast(features)))
        {
            return error.UnexpectedInputPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        if (local.constItems(f32).len != 32) {
            return error.UnexpectedInputPlacement;
        }
        for (local.constItems(f32)) |value| {
            if (value != 1) return error.UnexpectedInputValue;
        }
    }
}

fn verifyPartialSums(
    allocator: std.mem.Allocator,
    io: std.Io,
    partial_sums: *const zml.Buffer,
) !void {
    var starts: [2]i64 = undefined;
    var shard_count: usize = 0;
    var shards = partial_sums.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 1 or slices.get(0).size != 1 or
            slices.get(0).start < 0 or slices.get(0).start >= 4)
        {
            return error.UnexpectedPartialPlacement;
        }
        for (starts[0..shard_count]) |start| {
            if (start == slices.get(0).start) {
                return error.DuplicatePartialShard;
            }
        }
        starts[shard_count] = slices.get(0).start;
        shard_count += 1;

        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        if (local.constItems(f32)[0] != 32) {
            return error.UnexpectedPartialSum;
        }
    }
    if (shard_count != 2) return error.UnexpectedShardCount;
}

fn verifyGlobalSum(
    allocator: std.mem.Allocator,
    io: std.Io,
    global_sum: *const zml.Buffer,
) !void {
    var shards = global_sum.shards();
    while (shards.next()) |shard| {
        if (shard.globalSlices().len != 0) {
            return error.UnexpectedGlobalPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        if (local.constItems(f32)[0] != 128) {
            return error.UnexpectedGlobalSum;
        }
    }
    if (try global_sum.getValue(f32, io) != 128) {
        return error.UnexpectedGlobalSum;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    try distributed_example.expectTopology(platform, 4, 2);
    const data_sharding = try distributed_example.dataSharding(platform);

    const features = platform.addressableDevices().len * features_per_gpu;
    const input_shape = zml.Shape.init(.{
        .rows = platform.processCount() * rows_per_host,
        .features = features,
    }, .f32).withPartitioning(.{
        .rows = .data,
        .features = .replicated,
    });
    const host_data = try distributed_example.allocateValues(
        allocator,
        input_shape,
        .ones,
    );
    defer host_data.free(allocator);
    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        host_data,
        data_sharding,
    );
    defer input.deinit();
    try distributed_example.expectShardCounts(&input, 4, 2);
    try distributed_example.expectAddressable(platform, &input);
    try verifyInput(allocator, io, features, &input);

    var executable = try platform.compileFn(
        allocator,
        io,
        ShardStatistics.forward,
        .{zml.Tensor.fromShape(input_shape)},
        .{
            .shardings = &.{data_sharding},
            .program_name = "gpu-shard-map",
        },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{input});
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var output = results.get(zml.Bufferized(Statistics));
    defer output.partial_sums.deinit();
    defer output.global_sum.deinit();
    try distributed_example.expectShardCounts(&output.partial_sums, 4, 2);
    try distributed_example.expectShardCounts(&output.global_sum, 4, 2);
    try distributed_example.expectAddressable(platform, &output.partial_sums);
    try distributed_example.expectAddressable(platform, &output.global_sum);
    try verifyPartialSums(allocator, io, &output.partial_sums);
    try verifyGlobalSum(allocator, io, &output.global_sum);
    try distributed_example.printLocalShards(
        allocator,
        io,
        &output.partial_sums,
        "partial_sum_expected_32",
    );
    try distributed_example.printLocalShards(
        allocator,
        io,
        &output.global_sum,
        "global_sum_expected_128",
    );

    try platform.barrier("gpu-shard-map-before-shutdown");
}
