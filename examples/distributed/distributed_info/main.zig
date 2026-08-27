const std = @import("std");

const coordinator = @import("coordinator.zig");
const pjrt = @import("pjrt");
const platforms = @import("platforms");

const Arguments = struct {
    coordinator_address: std.Io.net.IpAddress,
    rank: usize,
    process_count: usize,
};

fn usage() void {
    std.debug.print(
        \\Usage: distributed_info COORDINATOR_IP:PORT RANK PROCESS_COUNT
        \\
        \\Example (one process and two GPUs per host):
        \\  rank 0: distributed_info 100.80.27.10:8910 0 2
        \\  rank 1: distributed_info 100.80.27.10:8910 1 2
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
    };
}

fn printDevice(
    api: *const pjrt.Api,
    device: *const pjrt.Device,
) void {
    const description = device.getDescription(api);
    const addressable = device.isAddressable(api);
    if (addressable) {
        std.debug.print(
            "  id={d} process={d} addressable=true" ++
                " local_hardware_id={d} kind={s}\n",
            .{
                description.id(api),
                description.processIndex(api),
                device.localHardwareId(api),
                description.kind(api),
            },
        );
    } else {
        std.debug.print(
            "  id={d} process={d} addressable=false kind={s}\n",
            .{
                description.id(api),
                description.processIndex(api),
                description.kind(api),
            },
        );
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const arguments = try parseArguments(init);

    var server: ?coordinator.Server = if (arguments.rank == 0)
        try .init(allocator, io, arguments.coordinator_address)
    else
        null;
    defer if (server) |*value| value.deinit();

    const server_thread: ?std.Thread = if (server) |*value|
        try .spawn(.{}, coordinator.Server.runThread, .{value})
    else
        null;

    var coordinator_client: coordinator.Client = .{
        .io = io,
        .address = arguments.coordinator_address,
    };
    defer if (server_thread) |thread| {
        coordinator_client.shutdown() catch {};
        thread.join();
    };

    const api = try platforms.load(allocator, io, .cuda);
    const visible_devices = [_]i64{ 0, 1 };
    const create_options = [_]pjrt.NamedValue{
        .init(.int64, "node_id", @intCast(arguments.rank)),
        .init(
            .int64,
            "num_nodes",
            @intCast(arguments.process_count),
        ),
        .init(.int64list, "visible_devices", &visible_devices),
        .init(.string, "allocator", "bfc"),
        .init(.bool, "preallocate", false),
        .init(.bool, "use_tfrt_gpu_client", true),
    };
    const key_value_store = coordinator_client.keyValueStore();

    std.debug.print(
        "rank={d}: creating distributed PJRT client through {f}\n",
        .{ arguments.rank, arguments.coordinator_address },
    );
    const client = try pjrt.Client.initWithKeyValueStore(
        api,
        &create_options,
        &key_value_store,
    );
    defer client.deinit(api);

    const global_devices = client.devices(api);
    const addressable_devices = client.addressableDevices(api);
    std.debug.print(
        "rank={d}: platform={s} global_devices={d}" ++
            " addressable_devices={d}\n",
        .{
            arguments.rank,
            client.platformName(api),
            global_devices.len,
            addressable_devices.len,
        },
    );

    std.debug.print("rank={d}: global device list:\n", .{arguments.rank});
    for (global_devices) |device| printDevice(api, device);
    std.debug.print(
        "rank={d}: addressable device list:\n",
        .{arguments.rank},
    );
    for (addressable_devices) |device| printDevice(api, device);

    try coordinator_client.barrier(
        allocator,
        arguments.rank,
        arguments.process_count,
    );
    std.debug.print("rank={d}: complete\n", .{arguments.rank});
}
