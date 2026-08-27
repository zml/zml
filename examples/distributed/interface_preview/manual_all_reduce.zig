//! Named-axis collective interface preview.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const PartitionSum = struct {
    pub fn forward(dummy: zml.Tensor) zml.Tensor {
        _ = dummy;

        const local_value = zml.ops.partitionId()
            .add(zml.Tensor.scalar(1, .u32));
        return zml.ops.allReduceAxes(
            local_value,
            .{.data},
            zml.Tensor.add,
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const data_sharding = try common.dataSharding(platform);

    const scalar_shape = zml.Shape.scalar(.u32);
    var executable = try platform.compileFn(
        allocator,
        io,
        PartitionSum.forward,
        .{zml.Tensor.fromShape(scalar_shape)},
        .{ .shardings = &.{data_sharding} },
    );
    defer executable.deinit();

    const zero: u32 = 0;
    var dummy = try zml.Buffer.fromBytes(
        io,
        platform,
        scalar_shape,
        zml.Sharding.replicated,
        std.mem.asBytes(&zero),
    );
    defer dummy.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{dummy});
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var global_sum = results.get(zml.Buffer);
    defer global_sum.deinit();
    var shards = global_sum.shards();
    while (shards.next()) |shard| {
        const value = try shard.toSliceAlloc(allocator, io);
        defer value.free(allocator);
        std.debug.print(
            "device={d} partition_sum={d} (expected 10)\n",
            .{ shard.globalDeviceId(), value.items(u32)[0] },
        );
    }
}
