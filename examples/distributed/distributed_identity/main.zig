const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: distributed_identity COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

const Increment = struct {
    pub fn forward(input: zml.Tensor) zml.Tensor {
        return input.addConstant(1).withPartitioning(.{ .data = .data });
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs).positional;
    if (args.processCount == 0 or
        args.rank >= args.processCount or
        args.namespace.len == 0)
    {
        return error.InvalidDistributedJob;
    }
    var platform = try zml.Platform.init(allocator, io, .cuda, .{
        .distributed = .{
            .coordinator_address = try .parseLiteral(args.coordinator),
            .process_index = args.rank,
            .process_count = args.processCount,
            .namespace = args.namespace,
            .local_device_ids = &.{ 0, 1 },
        },
        .xla_gpu = .{
            .allocator = .{ .bfc = .{ .preallocate = false } },
        },
    });
    defer platform.deinit(allocator, io);

    if (platform.globalDevices().len != 4 or
        platform.addressableDevices().len != 2)
    {
        return error.UnexpectedTopology;
    }
    const sharding = try platform.registerShardingWithStrategy(
        "data",
        .mesh(.{ .data = .low_bandwidth }),
        .parseBindings(.{ .data = .{ .network, .link } }),
    );
    const shape = zml.Shape.init(.{ .data = 16 }, .f32)
        .withPartitioning(.{ .data = .data });
    const host_input = try zml.Slice.alloc(allocator, shape);
    defer host_input.free(allocator);
    @memset(host_input.items(f32), 1);
    var input = try zml.Buffer.fromSlice(io, platform, host_input, sharding);
    defer input.deinit();

    var executable = try platform.compileFn(
        allocator,
        io,
        Increment.forward,
        .{zml.Tensor.fromShape(shape)},
        .{
            .shardings = &.{sharding},
            .program_name = "distributed-identity",
        },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);

    arguments.set(.{input});
    executable.callOpts(io, arguments, &results, .{ .wait = true });
    var output = results.get(zml.Buffer);
    defer output.deinit();
    var shards = output.shards();
    while (shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        for (values) |value| {
            if (value != 2) return error.UnexpectedValue;
        }
        std.debug.print(
            "input_plus_one: device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                values[0..@min(values.len, 8)],
            },
        );
    }

    const destination = try zml.Slice.alloc(allocator, shape);
    defer destination.free(allocator);
    if (output.toSlice(io, destination)) |_| {
        return error.ExpectedGlobalReadRequiresGather;
    } else |err| switch (err) {
        error.GlobalReadRequiresGather => {},
        else => return err,
    }

    arguments.set(.{input});
    executable.call(arguments, &results);
    var async_output = results.get(zml.Buffer);
    defer async_output.deinit();
    try async_output.await(io);
    var async_shards = async_output.shards();
    while (async_shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        for (local.constItems(f32)) |value| {
            if (value != 2) return error.UnexpectedValue;
        }
    }
    try platform.barrier("distributed-identity-before-shutdown");
}
