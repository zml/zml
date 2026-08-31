const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: distributed_all_reduce COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

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
    const data_sharding = try platform.registerShardingWithStrategy(
        "data",
        .mesh(.{ .data = .low_bandwidth }),
        .parseBindings(.{ .data = .{ .network, .link } }),
    );
    const input_shape = zml.Shape.init(.{
        .rows = platform.globalDevices().len * rows_per_partition,
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

    const partitions: u32 = @intCast(platform.globalDevices().len);
    const expected_partition_sum = partitions * (partitions + 1) / 2;
    const expected_manual_sum: f32 = @floatFromInt(input_shape.count());
    if (try output.partition_sum.getValue(u32, io) !=
        expected_partition_sum or
        try output.manual_sum.getValue(f32, io) != expected_manual_sum)
    {
        return error.UnexpectedCollectiveValue;
    }
    std.debug.print(
        "rank={d} local_shape={d}x{d} partition_sum={d} " ++
            "manual_sum={d}\n",
        .{
            platform.processIndex(),
            rows_per_partition,
            feature_count,
            expected_partition_sum,
            expected_manual_sum,
        },
    );
    try platform.barrier("distributed-all-reduce-before-shutdown");
}
