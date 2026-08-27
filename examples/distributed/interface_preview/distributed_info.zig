//! Final public bootstrap and device-discovery interface preview.
//! This file intentionally has no Bazel target.

const std = @import("std");

const common = @import("common.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const job = try common.Job.parse(init);

    const platform = try job.openPlatform(allocator, io);
    defer platform.deinit(allocator, io);

    std.debug.print(
        "distributed={} process={d}/{d}\n",
        .{
            platform.isDistributed(),
            platform.processIndex(),
            platform.processCount(),
        },
    );

    std.debug.print("global devices:\n", .{});
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

    std.debug.print("addressable devices:\n", .{});
    for (platform.addressableDevices()) |device| {
        std.debug.print(
            "  global_id={d} local_hardware_id={d}\n",
            .{ device.id(), device.localHardwareId() },
        );
    }

    try platform.barrier(io, "distributed-info-complete");
}
