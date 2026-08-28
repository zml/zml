//! Runnable ZML equivalent of ../../../../gpu_mesh.py.

const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const rows_per_host = 4;
const columns_per_gpu = 8;

const GlobalSum = struct {
    pub fn forward(input: zml.Tensor) zml.Tensor {
        return input.sum(.rows).sum(.columns).reshape(.{});
    }
};

fn verifyInput(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    columns: usize,
    host_data: zml.Slice,
    input: *const zml.Buffer,
) !void {
    var column_starts: [2]i64 = undefined;
    var shard_count: usize = 0;
    var shards = input.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        if (slices.len != 2) return error.UnexpectedInputPlacement;
        const rows_slice = slices.get(0);
        const columns_slice = slices.get(1);
        if (rows_slice.start != @as(i64, @intCast(rank * rows_per_host)) or
            rows_slice.size != rows_per_host or
            columns_slice.size != columns_per_gpu or
            (columns_slice.start != 0 and
                columns_slice.start != columns_per_gpu))
        {
            return error.UnexpectedInputPlacement;
        }
        for (column_starts[0..shard_count]) |start| {
            if (start == columns_slice.start) {
                return error.DuplicateInputShard;
            }
        }
        column_starts[shard_count] = columns_slice.start;
        shard_count += 1;

        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        for (local.constItems(f32), 0..) |actual, index| {
            const local_row = index / columns_per_gpu;
            const local_column = index % columns_per_gpu;
            const global_row: usize = @intCast(
                rows_slice.start + @as(i64, @intCast(local_row)),
            );
            const global_column: usize = @intCast(
                columns_slice.start + @as(i64, @intCast(local_column)),
            );
            const expected = host_data.constItems(f32)[
                global_row * columns + global_column
            ];
            if (actual != expected) return error.UnexpectedInputValue;
        }
    }
    if (shard_count != 2) return error.UnexpectedShardCount;
}

fn verifyGlobalSum(
    allocator: std.mem.Allocator,
    io: std.Io,
    output: *const zml.Buffer,
) !void {
    var shards = output.shards();
    while (shards.next()) |shard| {
        if (shard.globalSlices().len != 0) {
            return error.UnexpectedOutputPlacement;
        }
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        if (local.constItems(f32)[0] != 8128) {
            return error.UnexpectedGlobalSum;
        }
    }
    if (try output.getValue(f32, io) != 8128) {
        return error.UnexpectedGlobalSum;
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    try distributed_example.expectTopology(platform, 4, 2);
    const sharding = try distributed_example.hostGpuSharding(platform);

    const columns = platform.addressableDevices().len * columns_per_gpu;
    const global_shape = zml.Shape.init(.{
        .rows = platform.processCount() * rows_per_host,
        .columns = columns,
    }, .f32).withPartitioning(.{
        .rows = .host,
        .columns = .gpu,
    });
    const host_data = try distributed_example.allocateValues(
        allocator,
        global_shape,
        .sequence,
    );
    defer host_data.free(allocator);
    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        host_data,
        sharding,
    );
    defer input.deinit();
    try distributed_example.expectShardCounts(&input, 4, 2);
    try distributed_example.expectAddressable(platform, &input);
    try verifyInput(
        allocator,
        io,
        job.process_index,
        columns,
        host_data,
        &input,
    );
    try distributed_example.printLocalShards(
        allocator,
        io,
        &input,
        "global_array",
    );

    var executable = try platform.compileFn(
        allocator,
        io,
        GlobalSum.forward,
        .{zml.Tensor.fromShape(global_shape)},
        .{
            .shardings = &.{sharding},
            .program_name = "gpu-mesh",
        },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    arguments.set(.{input});
    executable.callOpts(io, arguments, &results, .{ .wait = true });
    var global_sum = results.get(zml.Buffer);
    defer global_sum.deinit();
    try distributed_example.expectShardCounts(&global_sum, 4, 2);
    try distributed_example.expectAddressable(platform, &global_sum);
    try verifyGlobalSum(allocator, io, &global_sum);
    try distributed_example.printLocalShards(
        allocator,
        io,
        &global_sum,
        "replicated_global_sum",
    );

    try platform.barrier("gpu-mesh-before-shutdown");
}
