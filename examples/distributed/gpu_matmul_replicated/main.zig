//! Runnable ZML equivalent of ../../../../gpu_matmul_replicat.py.

const std = @import("std");

const zml = @import("zml");

const CliArgs = struct {
    pub const help =
        \\Usage: gpu_matmul_replicated COORDINATOR RANK PROCESS_COUNT NAMESPACE
    ;

    positional: struct {
        coordinator: []const u8,
        rank: usize,
        processCount: usize,
        namespace: []const u8,
    },
};

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

    const host_a = try zml.Slice.alloc(allocator, a_shape);
    defer host_a.free(allocator);
    @memset(host_a.items(f32), 1);
    const host_b = try zml.Slice.alloc(allocator, b_shape);
    defer host_b.free(allocator);
    @memset(host_b.items(f32), 1);
    var a = try zml.Buffer.fromSlice(io, platform, host_a, sharding);
    defer a.deinit();
    var b = try zml.Buffer.fromSlice(
        io,
        platform,
        host_b,
        platform.replicated_sharding,
    );
    defer b.deinit();
    var weight_shards = b.shards();
    while (weight_shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        std.debug.print(
            "replicated_B: device={d} slices={any} values={any}\n",
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
        Matmul.forward,
        .{ zml.Tensor.fromShape(a_shape), zml.Tensor.fromShape(b_shape) },
        .{
            .shardings = &.{sharding},
            .program_name = "gpu-matmul-replicated",
        },
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
    var output_shards = output.shards();
    while (output_shards.next()) |shard| {
        const local = try shard.toSliceAlloc(allocator, io);
        defer local.free(allocator);
        const values = local.constItems(f32);
        for (values) |value| {
            if (value != 16) return error.UnexpectedValue;
        }
        std.debug.print(
            "A @ replicated_B: device={d} slices={any} values={any}\n",
            .{
                shard.globalDeviceId(),
                shard.globalSlices().constSlice(),
                values[0..@min(values.len, 8)],
            },
        );
    }
    try platform.barrier("gpu-matmul-replicated-before-shutdown");
}
