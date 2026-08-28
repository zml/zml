const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_host = 4;
const input_features = 8;
const hidden_per_model_partition = 4;
const output_features = 6;
const tolerance: f32 = 0.00001;

const Output = struct {
    column: zml.Tensor,
    gathered: zml.Tensor,
    automatic: zml.Tensor,
    explicit: zml.Tensor,
    model_group_sum: zml.Tensor,
};

const TensorParallel = struct {
    pub fn forward(
        input: zml.Tensor,
        column_weight: zml.Tensor,
        row_weight: zml.Tensor,
    ) Output {
        const column = input.dot(column_weight, .input)
            .withPartitioning(.{
            .batch = .data,
            .hidden = .model,
        });
        const gathered_shape = column.shape().withPartitioning(.{
            .batch = .data,
            .hidden = .replicated,
        });
        const gathered = zml.ops.manualComputation(
            column,
            gathered_shape,
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    local_column: zml.Tensor,
                    local_output_shape: zml.Shape,
                ) zml.Tensor {
                    std.debug.assert(
                        local_column.dim(.batch) == rows_per_host,
                    );
                    std.debug.assert(
                        local_column.dim(.hidden) ==
                            hidden_per_model_partition,
                    );
                    const result = zml.ops.allGatherAxes(
                        local_column,
                        .{.model},
                        .hidden,
                    );
                    std.debug.assert(result.shape().eql(local_output_shape));
                    return result;
                }
            }).body,
        );

        const automatic = column.dot(row_weight, .hidden)
            .withPartitioning(.{
            .batch = .data,
            .output = .replicated,
        });
        const explicit = zml.ops.manualComputation(
            .{ column, row_weight },
            automatic.shape(),
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    local_inputs: []const zml.Tensor,
                    local_output_shape: zml.Shape,
                ) zml.Tensor {
                    const local_column = local_inputs[0];
                    const local_weight = local_inputs[1];
                    std.debug.assert(
                        local_column.dim(.batch) == rows_per_host,
                    );
                    std.debug.assert(
                        local_column.dim(.hidden) ==
                            hidden_per_model_partition,
                    );
                    std.debug.assert(
                        local_weight.dim(.hidden) ==
                            hidden_per_model_partition,
                    );
                    const partial = local_column.dot(
                        local_weight,
                        .hidden,
                    );
                    std.debug.assert(partial.shape().eql(local_output_shape));
                    return zml.ops.allReduceAxes(
                        partial,
                        .{.model},
                        zml.Tensor.add,
                    );
                }
            }).body,
        );

        const data_groups = @divExact(input.dim(.batch), rows_per_host);
        const group_shape = zml.Shape.init(
            .{ .data_group = data_groups },
            .u32,
        ).withPartitioning(.{ .data_group = .data });
        const model_group_sum = zml.ops.manualComputation(
            column,
            group_shape,
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    _: zml.Tensor,
                    local_output_shape: zml.Shape,
                ) zml.Tensor {
                    const value = zml.ops.partitionId()
                        .add(zml.Tensor.scalar(1, .u32));
                    return zml.ops.allReduceAxes(
                        value,
                        .{.model},
                        zml.Tensor.add,
                    ).reshape(local_output_shape);
                }
            }).body,
        );
        return .{
            .column = column,
            .gathered = gathered,
            .automatic = automatic,
            .explicit = explicit,
            .model_group_sum = model_group_sum,
        };
    }
};

fn expectShardCounts(
    buffer: *const zml.Buffer,
    global: usize,
    local: usize,
) !void {
    if (buffer.numGlobalShards() != @as(u32, @intCast(global)) or
        buffer.numShards() != @as(u32, @intCast(local)))
    {
        return error.UnexpectedShardCount;
    }
}

fn expectAddressable(
    platform: *const zml.Platform,
    buffer: *const zml.Buffer,
) !void {
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        var found = false;
        for (platform.addressableDevices()) |device| {
            found = found or device.id() == shard.globalDeviceId();
        }
        if (!found) return error.RemoteBufferShard;
    }
}

fn expectClose(
    label: []const u8,
    rank: usize,
    partition: u32,
    index: usize,
    expected: f32,
    actual: f32,
) !void {
    if (@abs(expected - actual) <= tolerance) return;
    std.log.err(
        "{s}: rank={d} partition={d} index={d} " ++
            "expected={d} actual={d}",
        .{ label, rank, partition, index, expected, actual },
    );
    return error.UnexpectedTensorParallelValue;
}

fn verifyMatrix(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    label: []const u8,
    buffer: *const zml.Buffer,
    reference: []const f32,
    columns: usize,
    shard_columns: bool,
) !void {
    var column_starts: [2]i64 = undefined;
    var shard_count: usize = 0;
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2) return error.UnexpectedOutputPlacement;
        const row_slice = slices.get(0);
        const column_slice = slices.get(1);
        if (row_slice.start != @as(i64, @intCast(rank * rows_per_host)) or
            row_slice.size != rows_per_host)
        {
            return error.UnexpectedOutputPlacement;
        }
        if (shard_columns) {
            if (column_slice.size != hidden_per_model_partition or
                @mod(column_slice.start, hidden_per_model_partition) != 0)
            {
                return error.UnexpectedOutputPlacement;
            }
            for (column_starts[0..shard_count]) |start| {
                if (start == column_slice.start) {
                    return error.DuplicateModelShard;
                }
            }
            column_starts[shard_count] = column_slice.start;
        } else if (column_slice.start != 0 or
            column_slice.size != @as(i64, @intCast(columns)))
        {
            return error.UnexpectedOutputPlacement;
        }

        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const local_columns: usize = @intCast(column_slice.size);
        for (local.constItems(f32), 0..) |actual, index| {
            const global_row: usize = @intCast(
                row_slice.start + @as(i64, @intCast(index / local_columns)),
            );
            const global_column: usize = @intCast(
                column_slice.start + @as(i64, @intCast(index % local_columns)),
            );
            const global_index = global_row * columns + global_column;
            try expectClose(
                label,
                rank,
                shard.globalDeviceId(),
                global_index,
                reference[global_index],
                actual,
            );
        }
        const values = local.constItems(f32);
        std.debug.print(
            "rank={d} partition={d} {s} slices={any} values={any}\n",
            .{
                rank,
                shard.globalDeviceId(),
                label,
                slices.constSlice(),
                values[0..@min(values.len, 4)],
            },
        );
        shard_count += 1;
    }
    if (shard_count != 2) return error.UnexpectedShardCount;
}

fn verifyWeightPlacement(
    buffer: *const zml.Buffer,
    hidden_axis: usize,
) !void {
    var starts: [2]i64 = undefined;
    var count: usize = 0;
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        const hidden = shard.globalSlices().get(hidden_axis);
        if (hidden.size != hidden_per_model_partition) {
            return error.UnexpectedWeightPlacement;
        }
        for (starts[0..count]) |start| {
            if (start == hidden.start) return error.DuplicateModelShard;
        }
        starts[count] = hidden.start;
        count += 1;
    }
    if (count != 2) return error.UnexpectedShardCount;
}

fn verifyInputPlacement(rank: usize, buffer: *const zml.Buffer) !void {
    var count: usize = 0;
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2 or
            slices.get(0).start != @as(i64, @intCast(rank * rows_per_host)) or
            slices.get(0).size != rows_per_host or
            slices.get(1).start != 0 or
            slices.get(1).size != input_features)
        {
            return error.UnexpectedInputPlacement;
        }
        count += 1;
    }
    if (count != 2) return error.UnexpectedShardCount;
}

fn verifyGroupSum(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    buffer: *const zml.Buffer,
) !void {
    const expected: u32 = if (rank == 0) 3 else 7;
    var count: usize = 0;
    var shards = buffer.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 1 or slices.get(0).start != @as(i64, @intCast(rank)) or
            slices.get(0).size != 1)
        {
            return error.UnexpectedGroupPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const actual = local.constItems(u32)[0];
        if (actual != expected) {
            std.log.err(
                "model group mismatch: rank={d} partition={d} " ++
                    "expected={d} actual={d}",
                .{ rank, shard.globalDeviceId(), expected, actual },
            );
            return error.UnexpectedModelGroupSum;
        }
        count += 1;
    }
    if (count != 2) return error.UnexpectedShardCount;
    std.debug.print(
        "rank={d} model_group_sum={d}\n",
        .{ rank, expected },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const global_devices = platform.globalDevices().len;
    const local_devices = platform.addressableDevices().len;
    if (platform.processCount() != 2 or global_devices != 4 or
        local_devices != 2)
    {
        return error.UnexpectedTopology;
    }

    const sharding = try distributed_example.hybridSharding(platform);
    const data_partitions: usize = @intCast(
        sharding.numPartitionsForLogicalAxis(.data),
    );
    const model_partitions: usize = @intCast(
        sharding.numPartitionsForLogicalAxis(.model),
    );
    if (data_partitions != 2 or model_partitions != 2) {
        return error.UnexpectedTopology;
    }
    const batch = data_partitions * rows_per_host;
    const hidden = model_partitions * hidden_per_model_partition;
    std.debug.print(
        "rank={d} global_devices={d} local_devices={d} " ++
            "data={d} model={d}\n",
        .{
            job.process_index,
            global_devices,
            local_devices,
            data_partitions,
            model_partitions,
        },
    );

    const input_shape = zml.Shape.init(.{
        .batch = batch,
        .input = input_features,
    }, .f32).withPartitioning(.{
        .batch = .data,
        .input = .replicated,
    });
    const column_weight_shape = zml.Shape.init(.{
        .hidden = hidden,
        .input = input_features,
    }, .f32).withPartitioning(.{
        .hidden = .model,
        .input = .replicated,
    });
    const row_weight_shape = zml.Shape.init(.{
        .output = output_features,
        .hidden = hidden,
    }, .f32).withPartitioning(.{
        .output = .replicated,
        .hidden = .model,
    });

    const host_input = try zml.Slice.alloc(allocator, input_shape);
    defer host_input.free(allocator);
    for (host_input.items(f32), 0..) |*value, index| {
        const row = index / input_features;
        const feature = index % input_features;
        value.* = @floatFromInt(row + feature % 3 + 1);
    }
    const host_column_weight = try zml.Slice.alloc(
        allocator,
        column_weight_shape,
    );
    defer host_column_weight.free(allocator);
    for (host_column_weight.items(f32), 0..) |*value, index| {
        const hidden_index = index / input_features;
        const feature = index % input_features;
        value.* = @floatFromInt(
            (hidden_index + 1) * (feature % 3 + 1),
        );
    }
    const host_row_weight = try zml.Slice.alloc(
        allocator,
        row_weight_shape,
    );
    defer host_row_weight.free(allocator);
    for (host_row_weight.items(f32), 0..) |*value, index| {
        const output = index / hidden;
        const hidden_index = index % hidden;
        value.* = @floatFromInt(output % 3 + hidden_index % 2 + 1);
    }

    const column_reference = try allocator.alloc(f32, batch * hidden);
    defer allocator.free(column_reference);
    for (0..batch) |row| {
        for (0..hidden) |hidden_index| {
            var value: f32 = 0;
            for (0..input_features) |feature| {
                value += host_input.constItems(f32)[
                    row * input_features + feature
                ] * host_column_weight.constItems(f32)[
                    hidden_index * input_features + feature
                ];
            }
            column_reference[row * hidden + hidden_index] = value;
        }
    }
    const output_reference = try allocator.alloc(
        f32,
        batch * output_features,
    );
    defer allocator.free(output_reference);
    for (0..batch) |row| {
        for (0..output_features) |output| {
            var value: f32 = 0;
            for (0..hidden) |hidden_index| {
                value += column_reference[row * hidden + hidden_index] *
                    host_row_weight.constItems(f32)[
                        output * hidden + hidden_index
                    ];
            }
            output_reference[row * output_features + output] = value;
        }
    }
    if (output_reference[0] ==
        output_reference[rows_per_host * output_features])
    {
        return error.ReferenceDoesNotIsolateHostBatches;
    }

    var input = try zml.Buffer.fromSlice(io, platform, host_input, sharding);
    defer input.deinit();
    var column_weight = try zml.Buffer.fromSlice(
        io,
        platform,
        host_column_weight,
        sharding,
    );
    defer column_weight.deinit();
    var row_weight = try zml.Buffer.fromSlice(
        io,
        platform,
        host_row_weight,
        sharding,
    );
    defer row_weight.deinit();
    try expectShardCounts(&input, global_devices, local_devices);
    try expectShardCounts(&column_weight, global_devices, local_devices);
    try expectShardCounts(&row_weight, global_devices, local_devices);
    try expectAddressable(platform, &input);
    try expectAddressable(platform, &column_weight);
    try expectAddressable(platform, &row_weight);
    try verifyInputPlacement(job.process_index, &input);
    try verifyWeightPlacement(&column_weight, 0);
    try verifyWeightPlacement(&row_weight, 1);

    var executable = try platform.compileFn(
        allocator,
        io,
        TensorParallel.forward,
        .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(column_weight_shape),
            zml.Tensor.fromShape(row_weight_shape),
        },
        .{
            .shardings = &.{sharding},
            .program_name = "distributed-tensor-parallel",
        },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ input, column_weight, row_weight });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var output = results.get(zml.Bufferized(Output));
    defer output.column.deinit();
    defer output.gathered.deinit();
    defer output.automatic.deinit();
    defer output.explicit.deinit();
    defer output.model_group_sum.deinit();
    try expectShardCounts(&output.column, global_devices, local_devices);
    try expectShardCounts(&output.gathered, global_devices, local_devices);
    try expectShardCounts(&output.automatic, global_devices, local_devices);
    try expectShardCounts(&output.explicit, global_devices, local_devices);
    try expectShardCounts(
        &output.model_group_sum,
        global_devices,
        local_devices,
    );
    try verifyMatrix(
        allocator,
        io,
        job.process_index,
        "column",
        &output.column,
        column_reference,
        hidden,
        true,
    );
    try verifyMatrix(
        allocator,
        io,
        job.process_index,
        "gathered",
        &output.gathered,
        column_reference,
        hidden,
        false,
    );
    try verifyMatrix(
        allocator,
        io,
        job.process_index,
        "automatic",
        &output.automatic,
        output_reference,
        output_features,
        false,
    );
    try verifyMatrix(
        allocator,
        io,
        job.process_index,
        "explicit",
        &output.explicit,
        output_reference,
        output_features,
        false,
    );
    try verifyGroupSum(
        allocator,
        io,
        job.process_index,
        &output.model_group_sum,
    );

    try platform.barrier("distributed-tensor-parallel-before-shutdown");
}
