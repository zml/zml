const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_partition = 4;
const feature_count = 8;

const CollectiveOutput = struct {
    partition_sum: zml.Tensor,
    manual_sum: zml.Tensor,
};

const Collectives = struct {
    pub fn forward(input: zml.Tensor) CollectiveOutput {
        const partition_sum = zml.ops.manualComputation(
            input,
            zml.Shape.scalar(.u32),
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    _: zml.Tensor,
                    local_output_shape: zml.Shape,
                ) zml.Tensor {
                    const local_value = zml.ops.partitionId()
                        .add(zml.Tensor.scalar(1, .u32));
                    return zml.ops.allReduce(
                        local_value,
                        zml.Tensor.add,
                    ).reshape(local_output_shape);
                }
            }).body,
        );
        const manual_sum = zml.ops.manualComputation(
            input,
            zml.Shape.scalar(.f32),
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    local_input: zml.Tensor,
                    local_output_shape: zml.Shape,
                ) zml.Tensor {
                    std.debug.assert(
                        local_input.dim(.rows) == rows_per_partition,
                    );
                    std.debug.assert(
                        local_input.dim(.features) == feature_count,
                    );
                    const local_sum = local_input.sum(.rows)
                        .sum(.features)
                        .reshape(.{});
                    return zml.ops.allReduce(
                        local_sum,
                        zml.Tensor.add,
                    ).reshape(local_output_shape);
                }
            }).body,
        );
        return .{
            .partition_sum = partition_sum,
            .manual_sum = manual_sum,
        };
    }
};

fn verifyReplicatedScalar(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    label: []const u8,
    buffer: *const zml.Buffer,
    expected: anytype,
    global_device_count: usize,
    local_device_count: usize,
) !void {
    if (buffer.numGlobalShards() !=
        @as(u32, @intCast(global_device_count)) or
        buffer.numShards() != @as(u32, @intCast(local_device_count)))
    {
        return error.UnexpectedShardCount;
    }

    const T = @TypeOf(expected);
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        if (shard.globalSlices().len != 0) {
            return error.UnexpectedScalarPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const actual = local.constItems(T)[0];
        if (actual != expected) {
            std.log.err(
                "{s} mismatch: rank={d} device={d} " ++
                    "expected={any} actual={any}",
                .{
                    label,
                    rank,
                    shard.globalDeviceId(),
                    expected,
                    actual,
                },
            );
            return error.UnexpectedCollectiveValue;
        }
        std.debug.print(
            "rank={d} device={d} {s}={any}\n",
            .{ rank, shard.globalDeviceId(), label, actual },
        );
    }
    if (try buffer.getValue(T, io) != expected) {
        return error.UnexpectedCollectiveValue;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);

    const global_device_count = platform.globalDevices().len;
    const local_device_count = platform.addressableDevices().len;
    if (global_device_count != 4 or local_device_count != 2) {
        return error.UnexpectedTopology;
    }

    const data_sharding = try distributed_example.dataSharding(platform);
    const input_shape = zml.Shape.init(.{
        .rows = global_device_count * rows_per_partition,
        .features = feature_count,
    }, .f32).withPartitioning(.{
        .rows = .data,
        .features = .replicated,
    });
    const host_input = try zml.Slice.alloc(allocator, input_shape);
    defer host_input.free(allocator);
    @memset(host_input.items(f32), 1);

    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        host_input,
        data_sharding,
    );
    defer input.deinit();
    if (input.numGlobalShards() !=
        @as(u32, @intCast(global_device_count)) or
        input.numShards() != @as(u32, @intCast(local_device_count)))
    {
        return error.UnexpectedShardCount;
    }

    var executable = try platform.compileFn(
        allocator,
        io,
        Collectives.forward,
        .{zml.Tensor.fromShape(input_shape)},
        .{
            .shardings = &.{data_sharding},
            .program_name = "distributed-all-reduce",
        },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    arguments.set(.{input});
    executable.call(arguments, &results);

    var output = results.get(zml.Bufferized(CollectiveOutput));
    defer output.partition_sum.deinit();
    defer output.manual_sum.deinit();
    try output.partition_sum.await(io);
    try output.manual_sum.await(io);

    const partitions: u32 = @intCast(global_device_count);
    const expected_partition_sum = partitions * (partitions + 1) / 2;
    const expected_manual_sum: f32 = @floatFromInt(input_shape.count());
    try verifyReplicatedScalar(
        allocator,
        io,
        job.process_index,
        "partition_sum",
        &output.partition_sum,
        expected_partition_sum,
        global_device_count,
        local_device_count,
    );
    try verifyReplicatedScalar(
        allocator,
        io,
        job.process_index,
        "manual_sum",
        &output.manual_sum,
        expected_manual_sum,
        global_device_count,
        local_device_count,
    );
    std.debug.print(
        "rank={d} local_shape={d}x{d} partition_sum={d} " ++
            "manual_sum={d}\n",
        .{
            job.process_index,
            rows_per_partition,
            feature_count,
            expected_partition_sum,
            expected_manual_sum,
        },
    );
    try platform.barrier("distributed-all-reduce-before-shutdown");
}
