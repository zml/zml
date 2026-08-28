const std = @import("std");

const distributed_example = @import("distributed_example");
const zml = @import("zml");

const Increment = struct {
    pub fn forward(input: zml.Tensor) zml.Tensor {
        return input.addConstant(1).withPartitioning(.{ .data = .data });
    }
};

fn verifyLocalOutput(
    allocator: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    input: zml.Slice,
    output: *const zml.Buffer,
) !void {
    var shards = output.shards();
    while (shards.next()) |shard| {
        const slices = shard.globalSlices();
        const data_slice = slices.get(0);
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);

        for (local.constItems(f32), 0..) |actual, i| {
            const global_index: usize = @intCast(
                data_slice.start + @as(i64, @intCast(i)),
            );
            const expected = input.constItems(f32)[global_index] + 1;
            if (actual != expected) return error.UnexpectedOutput;
        }
        std.debug.print(
            "rank={d} device={d} slices={any} values={any}\n",
            .{
                rank,
                shard.globalDeviceId(),
                slices.constSlice(),
                local.constItems(f32),
            },
        );
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);

    const data_sharding = try distributed_example.dataSharding(platform);
    const global_shape = zml.Shape.init(.{ .data = 16 }, .f32)
        .withPartitioning(.{ .data = .data });
    const host_input = try zml.Slice.alloc(allocator, global_shape);
    defer host_input.free(allocator);
    for (host_input.items(f32), 0..) |*value, i| {
        value.* = @floatFromInt(i);
    }

    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        host_input,
        data_sharding,
    );
    defer input.deinit();
    if (input.numGlobalShards() != 4 or input.numShards() != 2) {
        return error.UnexpectedShardCount;
    }

    var executable = try platform.compileFn(
        allocator,
        io,
        Increment.forward,
        .{zml.Tensor.fromShape(global_shape)},
        .{
            .shardings = &.{data_sharding},
            .program_name = "distributed-identity",
        },
    );
    defer executable.deinit();
    var executable_arguments = try executable.args(allocator);
    defer executable_arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    executable_arguments.set(.{input});
    executable.callOpts(
        io,
        executable_arguments,
        &results,
        .{ .wait = true },
    );
    var output = results.get(zml.Buffer);
    defer output.deinit();
    try verifyLocalOutput(
        allocator,
        io,
        job.process_index,
        host_input,
        &output,
    );

    const global_destination = try zml.Slice.alloc(allocator, global_shape);
    defer global_destination.free(allocator);
    @memset(global_destination.data(), 0xa5);
    const original = try allocator.dupe(u8, global_destination.constData());
    defer allocator.free(original);
    if (output.toSlice(io, global_destination)) |_| {
        return error.ExpectedGlobalReadRequiresGather;
    } else |err| switch (err) {
        error.GlobalReadRequiresGather => {},
        else => return err,
    }
    if (!std.mem.eql(u8, original, global_destination.constData())) {
        return error.PartialGlobalRead;
    }

    executable_arguments.set(.{input});
    executable.call(executable_arguments, &results);
    var async_output = results.get(zml.Buffer);
    defer async_output.deinit();
    try async_output.await(io);
    try verifyLocalOutput(
        allocator,
        io,
        job.process_index,
        host_input,
        &async_output,
    );

    std.debug.print(
        "rank={d} global_devices={d} local_devices={d} " ++
            "global_shards={d} local_shards={d}\n",
        .{
            job.process_index,
            platform.globalDevices().len,
            platform.addressableDevices().len,
            output.numGlobalShards(),
            output.numShards(),
        },
    );
    try platform.barrier("distributed-identity-before-shutdown");
}
