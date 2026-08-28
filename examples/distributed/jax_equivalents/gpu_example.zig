//! Runnable ZML equivalent of ../../../../gpu_example.py.

const std = @import("std");

const zml = @import("zml");

const Job = struct {
    coordinator_address: std.Io.net.IpAddress,
    process_index: usize,
    process_count: usize,
    namespace: []const u8,

    fn parse(init: std.process.Init) !Job {
        var iterator = init.minimal.args.iterate();
        _ = iterator.next();

        const coordinator = iterator.next() orelse return usage();
        const process_index = iterator.next() orelse return usage();
        const process_count = iterator.next() orelse return usage();
        const namespace = iterator.next() orelse return usage();
        if (iterator.next() != null) return usage();

        const rank = try std.fmt.parseInt(usize, process_index, 10);
        const world = try std.fmt.parseInt(usize, process_count, 10);
        if (world == 0 or rank >= world or namespace.len == 0) {
            return error.InvalidDistributedJob;
        }

        return .{
            .coordinator_address = try .parseLiteral(coordinator),
            .process_index = rank,
            .process_count = world,
            .namespace = namespace,
        };
    }
};

fn usage() error{InvalidArguments} {
    std.debug.print(
        \\Usage: gpu_example COORDINATOR RANK PROCESS_COUNT NAMESPACE
        \\
        \\Example:
        \\  gpu_example 100.80.27.10:8910 0 2 zml-run-001
        \\
    , .{});
    return error.InvalidArguments;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try Job.parse(init);

    var platform = try zml.Platform.init(allocator, io, .cuda, .{
        .distributed = .{
            .coordinator_address = job.coordinator_address,
            .process_index = job.process_index,
            .process_count = job.process_count,
            .namespace = job.namespace,
            .local_device_ids = &.{ 0, 1 },
        },
        .xla_gpu = .{
            .allocator = .{
                .bfc = .{ .preallocate = false },
            },
        },
    });
    defer platform.deinit(allocator, io);

    std.debug.print(
        "process id = {d}\nglobal devices = {d}\nlocal devices = {d}\n",
        .{
            platform.processIndex(),
            platform.globalDevices().len,
            platform.addressableDevices().len,
        },
    );
    for (platform.globalDevices()) |device| {
        std.debug.print(
            "  id={d} process={d} addressable={} kind={s}\n",
            .{
                device.id(),
                device.processIndex(),
                device.isAddressable(),
                device.kind(),
            },
        );
    }
    std.debug.print("physical mesh:{f}\n", .{platform.physical_mesh});

    try platform.barrier("gpu-example-before-shutdown");
}
