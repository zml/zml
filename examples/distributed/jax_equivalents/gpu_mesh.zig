//! Proposed ZML equivalent of ../../../../gpu_mesh.py.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const GlobalSum = struct {
    pub fn forward(input: zml.Tensor) zml.Tensor {
        return input.sum(.rows).sum(.columns).reshape(.{});
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const sharding = try common.hostGpuSharding(platform);

    const host_count = platform.processCount();
    const gpus_per_host = platform.addressableDevices().len;
    const global_shape = zml.Shape.init(.{
        .rows = host_count * 4,
        .columns = gpus_per_host * 8,
    }, .f32).withPartitioning(.{
        .rows = .host,
        .columns = .gpu,
    });

    const host_data = try common.allocateValues(
        allocator,
        global_shape,
        .sequence,
    );
    defer host_data.free(allocator);
    var global_array = try zml.Buffer.fromSlice(
        io,
        platform,
        host_data,
        sharding,
    );
    defer global_array.deinit();

    std.debug.print(
        "global shape={f}, global shards={d}, local shards={d}\n",
        .{
            global_shape,
            global_array.numGlobalShards(),
            global_array.numShards(),
        },
    );
    try common.printLocalShards(
        allocator,
        io,
        &global_array,
        "global_array",
    );

    var executable = try platform.compileFn(
        allocator,
        io,
        GlobalSum.forward,
        .{zml.Tensor.fromShape(global_shape)},
        .{ .shardings = &.{sharding} },
    );
    defer executable.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{global_array});
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var global_sum = results.get(zml.Buffer);
    defer global_sum.deinit();
    try common.printLocalShards(
        allocator,
        io,
        &global_sum,
        "replicated_global_sum",
    );

    try platform.barrier("gpu-mesh-before-shutdown");
}
