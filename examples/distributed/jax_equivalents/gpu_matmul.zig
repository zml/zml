//! Proposed ZML equivalent of ../../../../gpu_matmul.py.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const MatmulRelu = struct {
    pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        return a.dot(b, .contracting).relu().withPartitioning(.{
            .left = .host,
            .right = .gpu,
        });
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    const sharding = try common.hostGpuSharding(platform);

    const rows = platform.processCount() * 4;
    const columns = platform.addressableDevices().len * 8;
    const a_shape = zml.Shape.init(.{
        .left = rows,
        .contracting = columns,
    }, .f32).withPartitioning(.{
        .left = .host,
        .contracting = .gpu,
    });
    const b_shape = zml.Shape.init(.{
        .right = rows,
        .contracting = columns,
    }, .f32).withPartitioning(.{
        .right = .host,
        .contracting = .gpu,
    });

    var executable = try platform.compileFn(
        allocator,
        io,
        MatmulRelu.forward,
        .{
            zml.Tensor.fromShape(a_shape),
            zml.Tensor.fromShape(b_shape),
        },
        .{ .shardings = &.{sharding} },
    );
    defer executable.deinit();

    const host_a = try common.allocateValues(allocator, a_shape, .ones);
    defer host_a.free(allocator);
    const host_b = try common.allocateValues(allocator, b_shape, .ones);
    defer host_b.free(allocator);
    var a = try zml.Buffer.fromSlice(io, platform, host_a, sharding);
    defer a.deinit();
    var b = try zml.Buffer.fromSlice(io, platform, host_b, sharding);
    defer b.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ a, b });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var c = results.get(zml.Buffer);
    defer c.deinit();
    try common.printLocalShards(allocator, io, &c, "relu(A @ B.T)");

    try platform.barrier(io, "gpu-matmul-before-shutdown");
}
