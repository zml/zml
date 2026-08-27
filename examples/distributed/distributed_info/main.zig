const std = @import("std");

const platforms = @import("platforms");
const zml = @import("zml");

const pjrt = zml.pjrt;

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

    const visible_devices = [_]i64{ 0, 1 };
    var runtime = try zml.distributed.Runtime.init(allocator, io, .{
        .coordinator_address = arguments.coordinator_address,
        .process_index = arguments.rank,
        .process_count = arguments.process_count,
        .namespace = arguments.namespace,
        .local_device_ids = &visible_devices,
    });
    defer runtime.deinit();

    const api = try platforms.load(allocator, io, .cuda);
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
    std.debug.print(
        "rank={d}: creating distributed PJRT client through {f}\n",
        .{ arguments.rank, arguments.coordinator_address },
    );
    var client: ?*pjrt.Client = try pjrt.Client.initWithKeyValueStore(
        api,
        &create_options,
        try runtime.keyValueStore(),
    );
    defer if (client) |value| value.deinit(api);

    const global_devices = client.?.devices(api);
    const addressable_devices = client.?.addressableDevices(api);
    std.debug.print(
        "rank={d}: platform={s} global_devices={d}" ++
            " addressable_devices={d}\n",
        .{
            arguments.rank,
            client.?.platformName(api),
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

    try runtime.barrier("device-info-printed");
    std.debug.print("rank={d}: complete\n", .{arguments.rank});
    try runtime.destroyPjrtClient(api, &client);
}
