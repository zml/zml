const std = @import("std");

const zml = @import("zml");

const Arguments = struct {
    coordinator_address: std.Io.net.IpAddress,
    rank: usize,
    process_count: usize,
    namespace: []const u8,
};

fn usage() void {
    std.debug.print(
        \\Usage: distributed_info COORDINATOR RANK PROCESS_COUNT NAMESPACE
        \\
        \\Example (one process and two GPUs per host):
        \\  rank 0: distributed_info 100.80.27.10:8910 0 2 run-001
        \\  rank 1: distributed_info 100.80.27.10:8910 1 2 run-001
        \\
    , .{});
}

fn parseArguments(init: std.process.Init) !Arguments {
    var iterator = init.minimal.args.iterate();
    _ = iterator.next();

    const address_text = iterator.next() orelse {
        usage();
        return error.MissingCoordinatorAddress;
    };
    const rank_text = iterator.next() orelse {
        usage();
        return error.MissingRank;
    };
    const process_count_text = iterator.next() orelse {
        usage();
        return error.MissingProcessCount;
    };
    const namespace = iterator.next() orelse {
        usage();
        return error.MissingNamespace;
    };
    if (iterator.next() != null) {
        usage();
        return error.TooManyArguments;
    }

    const rank = try std.fmt.parseInt(usize, rank_text, 10);
    const process_count = try std.fmt.parseInt(
        usize,
        process_count_text,
        10,
    );
    if (process_count == 0 or rank >= process_count) {
        return error.InvalidRank;
    }

    return .{
        .coordinator_address = try .parseLiteral(address_text),
        .rank = rank,
        .process_count = process_count,
        .namespace = namespace,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const arguments = try parseArguments(init);

    var platform = try zml.Platform.init(allocator, io, .cuda, .{
        .distributed = .{
            .coordinator_address = arguments.coordinator_address,
            .process_index = arguments.rank,
            .process_count = arguments.process_count,
            .namespace = arguments.namespace,
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
        "rank={d}: platform={s} global_devices={d}" ++
            " addressable_devices={d}\n",
        .{
            platform.processIndex(),
            @tagName(platform.target),
            platform.globalDevices().len,
            platform.addressableDevices().len,
        },
    );
    std.debug.print(
        "rank={d}: global device list:\n",
        .{platform.processIndex()},
    );
    for (platform.globalDevices()) |device| {
        std.debug.print(
            "  id={d} process={d} addressable={} kind={s}",
            .{
                device.id(),
                device.processIndex(),
                device.isAddressable(),
                device.kind(),
            },
        );
        if (platform.addressableDeviceById(device.id())) |local| {
            std.debug.print(
                " local_hardware_id={d}",
                .{local.localHardwareId()},
            );
        }
        std.debug.print("\n", .{});
    }
    std.debug.print(
        "rank={d}: physical mesh:{f}\n",
        .{ platform.processIndex(), platform.physical_mesh },
    );

    try platform.barrier("device-info-printed");
    std.debug.print("rank={d}: complete\n", .{platform.processIndex()});
}
