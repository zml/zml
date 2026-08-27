//! Runnable ZML equivalent of ../../../../gpu_example.py.

const std = @import("std");

const platforms = @import("platforms");
const zml = @import("zml");

const pjrt = zml.pjrt;

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

fn printDevice(api: *const pjrt.Api, device: *const pjrt.Device) void {
    const description = device.getDescription(api);
    std.debug.print(
        "  id={d} process={d} addressable={} kind={s}\n",
        .{
            description.id(api),
            description.processIndex(api),
            device.isAddressable(api),
            description.kind(api),
        },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try Job.parse(init);
    const visible_devices = [_]i64{ 0, 1 };

    var runtime = try zml.distributed.Runtime.init(allocator, io, .{
        .coordinator_address = job.coordinator_address,
        .process_index = job.process_index,
        .process_count = job.process_count,
        .namespace = job.namespace,
        .local_device_ids = &visible_devices,
    });
    defer runtime.deinit();

    const api = try platforms.load(allocator, io, .cuda);
    const create_options = [_]pjrt.NamedValue{
        .init(.int64, "node_id", @intCast(job.process_index)),
        .init(.int64, "num_nodes", @intCast(job.process_count)),
        .init(.int64list, "visible_devices", &visible_devices),
        .init(.string, "allocator", "bfc"),
        .init(.bool, "preallocate", false),
        .init(.bool, "use_tfrt_gpu_client", true),
    };
    var client: ?*pjrt.Client = try .initWithKeyValueStore(
        api,
        &create_options,
        try runtime.keyValueStore(),
    );
    defer if (client) |value| value.deinit(api);

    const global_devices = client.?.devices(api);
    const local_devices = client.?.addressableDevices(api);
    std.debug.print(
        "process id = {d}\nglobal devices = {d}\nlocal devices = {d}\n",
        .{ job.process_index, global_devices.len, local_devices.len },
    );

    for (global_devices) |device| printDevice(api, device);

    try runtime.barrier("gpu-example-before-shutdown");
    try runtime.destroyPjrtClient(api, &client);
}
