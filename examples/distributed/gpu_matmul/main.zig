//! Runnable ZML equivalent of ../../../../gpu_matmul.py.

const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_host = 4;
const columns_per_gpu = 8;

const MatmulRelu = struct {
    pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        return a.dot(b, .contracting).relu().withPartitioning(.{
            .left = .host,
            .right = .gpu,
        });
    }
};

fn verifyOutput(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    output: *const zml.Buffer,
) !void {
    var right_starts: [2]i64 = undefined;
    var shard_count: usize = 0;
    var shards = output.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2) return error.UnexpectedOutputPlacement;
        const left = slices.get(0);
        const right = slices.get(1);
        if (left.start != @as(i64, @intCast(rank * rows_per_host)) or
            left.size != rows_per_host or
            right.size != rows_per_host or
            (right.start != 0 and right.start != rows_per_host))
        {
            return error.UnexpectedOutputPlacement;
        }
        for (right_starts[0..shard_count]) |start| {
            if (start == right.start) return error.DuplicateOutputShard;
        }
        right_starts[shard_count] = right.start;
        shard_count += 1;

        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        for (local.constItems(f32)) |value| {
            if (value != 16) return error.UnexpectedMatmulValue;
        }
    }
    if (shard_count != 2) return error.UnexpectedShardCount;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    try distributed_example.expectTopology(platform, 4, 2);
    const sharding = try distributed_example.hostGpuSharding(platform);

    const rows = platform.processCount() * rows_per_host;
    const columns = platform.addressableDevices().len * columns_per_gpu;
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
        .{
            .shardings = &.{sharding},
            .program_name = "gpu-matmul",
        },
    );
    defer executable.deinit();

    const host_a = try distributed_example.allocateValues(
        allocator,
        a_shape,
        .ones,
    );
    defer host_a.free(allocator);
    const host_b = try distributed_example.allocateValues(
        allocator,
        b_shape,
        .ones,
    );
    defer host_b.free(allocator);
    var a = try zml.Buffer.fromSlice(io, platform, host_a, sharding);
    defer a.deinit();
    var b = try zml.Buffer.fromSlice(io, platform, host_b, sharding);
    defer b.deinit();
    try distributed_example.expectShardCounts(&a, 4, 2);
    try distributed_example.expectShardCounts(&b, 4, 2);
    try distributed_example.expectAddressable(platform, &a);
    try distributed_example.expectAddressable(platform, &b);

    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ a, b });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var c = results.get(zml.Buffer);
    defer c.deinit();
    try distributed_example.expectShardCounts(&c, 4, 2);
    try distributed_example.expectAddressable(platform, &c);
    try verifyOutput(allocator, io, job.process_index, &c);
    try distributed_example.printLocalShards(
        allocator,
        io,
        &c,
        "relu(A @ B.T)",
    );

    try platform.barrier("gpu-matmul-before-shutdown");
}
