//! Runnable ZML equivalent of ../../../../gpu_example.py.

const std = @import("std");

const distributed_example = @import("distributed_example");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try distributed_example.Job.parse(init);

    var platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);
    try distributed_example.expectTopology(platform, 4, 2);
    if (platform.processIndex() != job.process_index or
        platform.processCount() != 2)
    {
        return error.UnexpectedProcessTopology;
    }

    var devices_per_process: [2]usize = @splat(0);
    for (platform.globalDevices()) |device| {
        const process_index = device.processIndex();
        if (process_index >= devices_per_process.len or
            device.isAddressable() != (process_index == job.process_index))
        {
            return error.UnexpectedDeviceOwnership;
        }
        devices_per_process[process_index] += 1;
    }
    if (devices_per_process[0] != 2 or devices_per_process[1] != 2) {
        return error.UnexpectedDeviceOwnership;
    }

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
