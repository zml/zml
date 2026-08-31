const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: distributed_tensor_parallel COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

const rows_per_host = 4;
const input_features = 8;
const hidden_per_model_partition = 4;
const output_features = 6;

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
            .withPartitioning(.{ .batch = .data, .hidden = .model });
        const gathered = zml.ops.manualComputation(
            column,
            column.shape().withPartitioning(.{
                .batch = .data,
                .hidden = .replicated,
            }),
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    local: zml.Tensor,
                    _: zml.Shape,
                ) zml.Tensor {
                    return zml.ops.allGatherAxes(
                        local,
                        .{.model},
                        .hidden,
                    );
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
                    local: []const zml.Tensor,
                    _: zml.Shape,
                ) zml.Tensor {
                    return zml.ops.allReduceAxes(
                        local[0].dot(local[1], .hidden),
                        .{.model},
                        zml.Tensor.add,
                    );
                }
            }).body,
        );
        const model_group_sum = zml.ops.manualComputation(
            column,
            zml.Shape.init(.{ .data_group = 2 }, .u32)
                .withPartitioning(.{ .data_group = .data }),
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    _: zml.Tensor,
                    local_shape: zml.Shape,
                ) zml.Tensor {
                    return zml.ops.allReduceAxes(
                        zml.ops.partitionId().add(
                            zml.Tensor.scalar(1, .u32),
                        ),
                        .{.model},
                        zml.Tensor.add,
                    ).reshape(local_shape);
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs).positional;
    if (args.processCount == 0 or
        args.rank >= args.processCount or
        args.namespace.len == 0)
    {
        return error.InvalidDistributedJob;
    }
    var platform = try zml.Platform.init(allocator, io, .cuda, .{
        .distributed = .{
            .coordinator_address = try .parseLiteral(args.coordinator),
            .process_index = args.rank,
            .process_count = args.processCount,
            .namespace = args.namespace,
            .local_device_ids = &.{ 0, 1 },
        },
        .xla_gpu = .{
            .allocator = .{ .bfc = .{ .preallocate = false } },
        },
    });
    defer platform.deinit(allocator, io);

    if (platform.globalDevices().len != 4 or
        platform.addressableDevices().len != 2)
    {
        return error.UnexpectedTopology;
    }
    const sharding = try platform.registerShardingWithStrategy(
        "data-model",
        .mesh(.{
            .data = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .parseBindings(.{
            .data = .network,
            .model = .link,
        }),
    );
    if (sharding.numPartitionsForLogicalAxis(.data) != 2 or
        sharding.numPartitionsForLogicalAxis(.model) != 2)
    {
        return error.UnexpectedTopology;
    }
    const input_shape = zml.Shape.init(.{
        .batch = 2 * rows_per_host,
        .input = input_features,
    }, .f32).withPartitioning(.{
        .batch = .data,
        .input = .replicated,
    });
    const column_weight_shape = zml.Shape.init(.{
        .hidden = 2 * hidden_per_model_partition,
        .input = input_features,
    }, .f32).withPartitioning(.{
        .hidden = .model,
        .input = .replicated,
    });
    const row_weight_shape = zml.Shape.init(.{
        .output = output_features,
        .hidden = 2 * hidden_per_model_partition,
    }, .f32).withPartitioning(.{
        .output = .replicated,
        .hidden = .model,
    });

    const host_input = try zml.Slice.alloc(allocator, input_shape);
    defer host_input.free(allocator);
    @memset(host_input.items(f32), 1);
    const host_column_weight = try zml.Slice.alloc(
        allocator,
        column_weight_shape,
    );
    defer host_column_weight.free(allocator);
    @memset(host_column_weight.items(f32), 1);
    const host_row_weight = try zml.Slice.alloc(allocator, row_weight_shape);
    defer host_row_weight.free(allocator);
    @memset(host_row_weight.items(f32), 1);

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
    inline for (
        .{ "column", "gathered", "automatic", "explicit" },
        .{
            &output.column,
            &output.gathered,
            &output.automatic,
            &output.explicit,
        },
        .{
            input_features,
            input_features,
            input_features * 2 * hidden_per_model_partition,
            input_features * 2 * hidden_per_model_partition,
        },
    ) |label, buffer, expected| {
        var shards = buffer.shards();
        while (shards.next()) |shard| {
            const local = try shard.toSliceAlloc(allocator, io);
            defer local.free(allocator);
            const values = local.constItems(f32);
            for (values) |value| {
                if (value != expected) return error.UnexpectedValue;
            }
            std.debug.print(
                "{s}: device={d} slices={any} values={any}\n",
                .{
                    label,
                    shard.globalDeviceId(),
                    shard.globalSlices().constSlice(),
                    values[0..@min(values.len, 8)],
                },
            );
        }
    }
    const expected_group_sum: u32 = if (args.rank == 0) 3 else 7;
    var group_shards = output.model_group_sum.shards();
    while (group_shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        for (local.constItems(u32)) |value| {
            if (value != expected_group_sum) {
                return error.UnexpectedGroupSum;
            }
        }
    }
    std.debug.print(
        "rank={d} model_group_sum={d}\n",
        .{ args.rank, expected_group_sum },
    );
    try platform.barrier("distributed-tensor-parallel-before-shutdown");
}
