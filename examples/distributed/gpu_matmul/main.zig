//! Runnable ZML equivalent of ../../../../gpu_matmul.py.

const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: gpu_matmul COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

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

    const host_a = try zml.Slice.alloc(allocator, a_shape);
    defer host_a.free(allocator);
    @memset(host_a.items(f32), 1);
    const host_b = try zml.Slice.alloc(allocator, b_shape);
    defer host_b.free(allocator);
    @memset(host_b.items(f32), 1);
    var a = try zml.Buffer.fromSlice(io, platform, host_a, sharding);
    defer a.deinit();
    var b = try zml.Buffer.fromSlice(io, platform, host_b, sharding);
    defer b.deinit();

    var executable = try platform.compileFn(
        allocator,
        io,
        MatmulRelu.forward,
        .{ zml.Tensor.fromShape(a_shape), zml.Tensor.fromShape(b_shape) },
        .{ .shardings = &.{sharding}, .program_name = "gpu-matmul" },
    );
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ a, b });
    executable.callOpts(io, arguments, &results, .{ .wait = true });

    var output = results.get(zml.Buffer);
    defer output.deinit();
    var shards = output.shards();
    while (shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        for (values) |value| {
            if (value != 16) return error.UnexpectedValue;
        }
        std.debug.print(
            "relu(A @ B.T): device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                values[0..@min(values.len, 8)],
            },
        );
    }
    try platform.barrier("gpu-matmul-before-shutdown");
}
