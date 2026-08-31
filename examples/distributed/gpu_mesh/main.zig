//! Runnable ZML equivalent of ../../../../gpu_mesh.py.

const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: gpu_mesh COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

const GlobalSum = struct {
    pub fn forward(input: zml.Tensor) zml.Tensor {
        return input.sum(.rows).sum(.columns).reshape(.{});
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
        "host-gpu",
        .mesh(.{
            .host = .low_bandwidth,
            .gpu = .high_bandwidth,
        }),
        .parseBindings(.{
            .host = .network,
            .gpu = .link,
        }),
    );
    const shape = zml.Shape.init(.{
        .rows = platform.processCount() * 4,
        .columns = platform.addressableDevices().len * 8,
    }, .f32).withPartitioning(.{
        .rows = .host,
        .columns = .gpu,
    });
    const host_data = try zml.Slice.alloc(allocator, shape);
    defer host_data.free(allocator);
    for (host_data.items(f32), 0..) |*value, index| {
        value.* = @floatFromInt(index);
    }
    var input = try zml.Buffer.fromSlice(io, platform, host_data, sharding);
    defer input.deinit();
    if (input.numGlobalShards() != 4 or input.numShards() != 2) {
        return error.UnexpectedShardCount;
    }
    var input_shards = input.shards();
    while (input_shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        std.debug.print(
            "global_array: device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                values[0..@min(values.len, 8)],
            },
        );
    }

    var executable = try platform.compileFn(
        allocator,
        io,
        GlobalSum.forward,
        .{zml.Tensor.fromShape(shape)},
        .{ .shardings = &.{sharding}, .program_name = "gpu-mesh" },
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
    const sum = try global_sum.getValue(f32, io);
    if (sum != 8128) {
        return error.UnexpectedGlobalSum;
    }
    std.debug.print("replicated_global_sum={d}\n", .{sum});
    try platform.barrier("gpu-mesh-before-shutdown");
}
