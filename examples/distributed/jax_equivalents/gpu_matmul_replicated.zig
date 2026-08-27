//! Proposed ZML equivalent of ../../../../gpu_matmul_replicat.py.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");
const zml = @import("zml");

const Matmul = struct {
    pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        return a.dot(b, .contracting).withPartitioning(.{
            .rows = .host,
            .output = .replicated,
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
        .rows = rows,
        .contracting = columns,
    }, .f32).withPartitioning(.{
        .rows = .host,
        .contracting = .gpu,
    });
    const b_shape = zml.Shape.init(.{
        .contracting = columns,
        .output = columns,
    }, .f32).withPartitioning(.{
        .contracting = .replicated,
        .output = .replicated,
    });

    var executable = try platform.compileFn(
        allocator,
        io,
        Matmul.forward,
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
    var b = try zml.Buffer.fromSlice(
        io,
        platform,
        host_b,
        zml.Sharding.replicated,
    );
    defer b.deinit();

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ a, b });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    // Matches JAX's inferred P("host", None): rows are split between hosts,
    // while each host's two GPUs hold equivalent output replicas.
    var c = results.get(zml.Buffer);
    defer c.deinit();
    try common.printLocalShards(allocator, io, &c, "A @ replicated_B");

    try platform.barrier(io, "gpu-matmul-replicated-before-shutdown");
}
