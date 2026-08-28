//! Four-way batch-sharded matrix multiplication interface preview.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const Matmul = struct {
    pub fn forward(
        input: zml.Tensor,
        weights: zml.Tensor,
    ) zml.Tensor {
        return input.dot(weights, .feature).withPartitioning(.{
            .batch = .data,
            .output = .replicated,
        });
    }
};

fn sequence(
    allocator: std.mem.Allocator,
    shape: zml.Shape,
    start: f32,
) !zml.Slice {
    const result = try zml.Slice.alloc(allocator, shape);
    for (result.items(f32), 0..) |*value, index| {
        value.* = start + @as(f32, @floatFromInt(index));
    }
    return result;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const data_sharding = try common.dataSharding(platform);

    const input_shape = zml.Shape.init(
        .{ .batch = 64, .feature = 16 },
        .f32,
    ).withPartitioning(.{
        .batch = .data,
        .feature = .replicated,
    });
    const weight_shape = zml.Shape.init(
        .{ .feature = 16, .output = 8 },
        .f32,
    ).withPartitioning(.{
        .feature = .replicated,
        .output = .replicated,
    });

    var executable = try platform.compileFn(
        allocator,
        io,
        Matmul.forward,
        .{
            zml.Tensor.fromShape(input_shape),
            zml.Tensor.fromShape(weight_shape),
        },
        .{ .shardings = &.{data_sharding} },
    );
    defer executable.deinit();

    const input_host = try sequence(allocator, input_shape, 0);
    defer input_host.free(allocator);
    const weights_host = try sequence(allocator, weight_shape, 1);
    defer weights_host.free(allocator);

    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        input_host,
        data_sharding,
    );
    defer input.deinit();
    var weights = try zml.Buffer.fromSlice(
        io,
        platform,
        weights_host,
        zml.Sharding.replicated,
    );
    defer weights.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ input, weights });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var output = results.get(zml.Buffer);
    defer output.deinit();
    std.debug.assert(output.numGlobalShards() == 4);
    std.debug.assert(output.numShards() == 2);

    var shards = output.shards();
    while (shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        std.debug.print(
            "device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices(),
                local.items(f32),
            },
        );
    }
}
