//! Proposed ZML equivalent of ../../../../gpu_example.py.
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
        "process id = {d}\nglobal devices = {d}\nlocal devices = {d}\n",
        .{
            platform.processIndex(),
            platform.globalDevices().len,
            platform.addressableDevices().len,
        },
    );

    for (platform.globalDevices()) |device| {
        std.debug.print(
            "global id={d} process={d} addressable={} kind={s}\n",
            .{
                device.id(),
                device.processIndex(),
                device.isAddressable(),
                device.kind(),
            },
        );
    }
    for (platform.addressableDevices()) |device| {
        std.debug.print(
            "local global_id={d} hardware_id={d}\n",
            .{ device.id(), device.localHardwareId() },
        );
    }

    try platform.barrier(io, "gpu-example-before-shutdown");
}
