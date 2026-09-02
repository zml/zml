const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: gpu_shard_map COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

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
                    const outputs = allocator.alloc(zml.Tensor, 2) catch
                        unreachable;
                    outputs[0] = local_sum;
                    outputs[1] = zml.ops.allReduceAxes(
                        local_sum,
                        .{.data},
                        zml.Tensor.add,
                    ).reshape(local_outputs[1]);
                    return outputs;
                }
            }).body,
        );
        return .{ .partial_sums = mapped[0], .global_sum = mapped[1] };
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
        "data",
        .mesh(.{ .data = .low_bandwidth }),
        .parseBindings(.{ .data = .{ .network, .link } }),
    );
    const shape = zml.Shape.init(.{
        .rows = platform.processCount() * 4,
        .features = platform.addressableDevices().len * 8,
    }, .f32).withPartitioning(.{
        .rows = .data,
        .features = .replicated,
    });
    const host_data = try zml.Slice.alloc(allocator, shape);
    defer host_data.free(allocator);
    @memset(host_data.items(f32), 1);
    var input = try zml.Buffer.fromSlice(io, platform, host_data, sharding);
    defer input.deinit();

    var executable = try platform.compileFn(
        allocator,
        io,
        ShardStatistics.forward,
        .{zml.Tensor.fromShape(shape)},
        .{ .shardings = &.{sharding}, .program_name = "gpu-shard-map" },
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
    var shards = output.partial_sums.shards();
    while (shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        for (values) |value| {
            if (value != 32) return error.UnexpectedPartialSum;
        }
        std.debug.print(
            "partial_sum: device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                values,
            },
        );
    }
    const global_sum = try output.global_sum.getValue(f32, io);
    if (global_sum != 128) {
        return error.UnexpectedGlobalSum;
    }
    std.debug.print("global_sum={d}\n", .{global_sum});
    try platform.barrier("gpu-shard-map-before-shutdown");
}
