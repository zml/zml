//! Runnable ZML equivalent of ../../../../gpu_matmul_replicat.py.

const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_host = 4;
const columns_per_gpu = 8;

const Matmul = struct {
    pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        return a.dot(b, .contracting).withPartitioning(.{
            .rows = .host,
            .output = .replicated,
        });
    }
};

fn verifyReplicatedWeight(
    allocator: std.mem.Allocator,
    io: std.Io,
    columns: usize,
    weight: *const zml.Buffer,
) !void {
    var shards = weight.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2 or
            slices.get(0).start != 0 or
            slices.get(0).size != @as(i64, @intCast(columns)) or
            slices.get(1).start != 0 or
            slices.get(1).size != @as(i64, @intCast(columns)))
        {
            return error.UnexpectedWeightPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        for (local.constItems(f32)) |value| {
            if (value != 1) return error.UnexpectedWeightValue;
        }
    }
}

fn verifyOutput(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    columns: usize,
    output: *const zml.Buffer,
) !void {
    var shard_count: usize = 0;
    var shards = output.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2 or
            slices.get(0).start !=
                @as(i64, @intCast(rank * rows_per_host)) or
            slices.get(0).size != rows_per_host or
            slices.get(1).start != 0 or
            slices.get(1).size != @as(i64, @intCast(columns)))
        {
            return error.UnexpectedOutputPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        for (local.constItems(f32)) |value| {
            if (value != 16) return error.UnexpectedMatmulValue;
        }
        shard_count += 1;
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
        .{
            .shardings = &.{sharding},
            .program_name = "gpu-matmul-replicated",
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
    var b = try zml.Buffer.fromSlice(
        io,
        platform,
        host_b,
        zml.Sharding.replicated,
    );
    defer b.deinit();
    try distributed_example.expectShardCounts(&a, 4, 2);
    try distributed_example.expectShardCounts(&b, 4, 2);
    try distributed_example.expectAddressable(platform, &a);
    try distributed_example.expectAddressable(platform, &b);
    try verifyReplicatedWeight(allocator, io, columns, &b);

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
    try verifyOutput(allocator, io, job.process_index, columns, &c);
    try distributed_example.printLocalShards(
        allocator,
        io,
        &c,
        "A @ replicated_B",
    );

    try platform.barrier("gpu-matmul-replicated-before-shutdown");
}
