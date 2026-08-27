//! Proposed ZML equivalent of ../../../../gpu_shard_map.py.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const Statistics = struct {
    partial_sums: zml.Tensor,
    global_sum: zml.Tensor,
};

const ShardStatistics = struct {
    pub fn forward(input: zml.Tensor) Statistics {
        const partial_shape = zml.Shape.init(.{ .partition = 4 }, .f32)
            .withPartitioning(.{ .partition = .data });
        const global_shape = zml.Shape.scalar(.f32);

        const mapped = zml.ops.manualComputation(
            input,
            [2]zml.Shape{ partial_shape, global_shape },
            {},
            (struct {
                fn body(
                    _: void,
                    _: std.mem.Allocator,
                    local_input: zml.Tensor,
                    local_outputs: []const zml.Shape,
                ) [2]zml.Tensor {
                    const local_sum = local_input
                        .sum(.rows)
                        .sum(.features)
                        .reshape(local_outputs[0]);
                    const global_sum = zml.ops.allReduceAxes(
                        local_sum,
                        .{.data},
                        zml.Tensor.add,
                    ).reshape(local_outputs[1]);
                    return .{ local_sum, global_sum };
                }
            }).body,
        );
        return .{
            .partial_sums = mapped[0],
            .global_sum = mapped[1],
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const data_sharding = try common.dataSharding(platform);

    const rows_per_host = 4;
    const features_per_gpu = 8;
    const input_shape = zml.Shape.init(.{
        .rows = platform.processCount() * rows_per_host,
        .features = platform.addressableDevices().len * features_per_gpu,
    }, .f32).withPartitioning(.{
        .rows = .data,
        .features = .replicated,
    });

    var executable = try platform.compileFn(
        allocator,
        io,
        ShardStatistics.forward,
        .{zml.Tensor.fromShape(input_shape)},
        .{ .shardings = &.{data_sharding} },
    );
    defer executable.deinit();

    const host_data = try common.allocateValues(
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

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{input});
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var partial_sums = results.get(.partial_sums);
    defer partial_sums.deinit();
    var global_sum = results.get(.global_sum);
    defer global_sum.deinit();
    try common.printLocalShards(
        allocator,
        io,
        &partial_sums,
        "partial_sum_expected_32",
    );
    try common.printLocalShards(
        allocator,
        io,
        &global_sum,
        "global_sum_expected_128",
    );

    try platform.barrier(io, "gpu-shard-map-before-shutdown");
}
