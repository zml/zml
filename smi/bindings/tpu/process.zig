const std = @import("std");
const pi = @import("../../info/process_info.zig");
const ProcessShadowList = @import("../../shadow_list.zig").ShadowList(pi.ProcessInfo);
const DeviceInfo = @import("../../info/device_info.zig").DeviceInfo;
const Worker = @import("../../worker.zig").Worker;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, devices_per_chip: u32, device_infos: []*DeviceInfo, list: *ProcessShadowList) !void {
    try w.spawn(io, scanLoop, .{ io, w, allocator, list, devices_per_chip, device_infos });
}

fn scanLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *ProcessShadowList, devices_per_chip: u32, device_infos: []*DeviceInfo) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);

    var sl = list.shadow();
    defer sl.deinit(allocator);

    while (w.isRunning()) {
        const start: std.Io.Timestamp = .now(io, .awake);

        sl.clearRetainingCapacity();

        scan(io, allocator, &sl, devices_per_chip, device_infos);

        sl.swap(io);

        const elapsed = start.untilNow(io, .awake);
        if (elapsed.nanoseconds < interval.nanoseconds) {
            io.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
        }
    }
}

fn scan(io: std.Io, allocator: std.mem.Allocator, sl: *ProcessShadowList.Shadow, devices_per_chip: u32, infos: []*DeviceInfo) void {
    var proc_dir = std.Io.Dir.openDirAbsolute(io, "/proc", .{ .iterate = true }) catch return;
    defer proc_dir.close(io);

    var proc_it = proc_dir.iterate();
    while (proc_it.next(io) catch null) |proc_entry| {
        const pid = std.fmt.parseInt(u32, proc_entry.name, 10) catch continue;

        var fd_path_buf: [64]u8 = undefined;
        const fd_sub = std.fmt.bufPrint(&fd_path_buf, "{d}/fd", .{pid}) catch continue;
        var fd_dir = proc_dir.openDir(io, fd_sub, .{ .iterate = true }) catch continue;
        defer fd_dir.close(io);

        var fd_it = fd_dir.iterate();
        while (fd_it.next(io) catch null) |fd_entry| {
            var link_buf: [256]u8 = undefined;
            const len = fd_dir.readLink(io, fd_entry.name, &link_buf) catch continue;
            const target = link_buf[0..len];

            if (parseChipIndex(target)) |chip_idx| {
                const dev_idx: u8 = @intCast(chip_idx * devices_per_chip);
                var info: pi.ProcessInfo = .{ .pid = pid, .device_idx = dev_idx };

                if (dev_idx < infos.len) {
                    const tpu = infos[dev_idx].tpu;

                    if (tpu.mem_used_bytes) |mem| {
                        info.dev_mem_kib = @intCast(mem / 1024);
                    }

                    if (tpu.util_percent) |util| {
                        info.dev_util_percent = @intCast(util);
                    }
                }

                sl.append(allocator, info) catch return;
                break;
            }
        }
    }
}

fn parseChipIndex(target: []const u8) ?u32 {
    if (std.mem.startsWith(u8, target, "/dev/accel")) {
        return std.fmt.parseInt(u32, target["/dev/accel".len..], 10) catch null;
    }

    if (std.mem.startsWith(u8, target, "/dev/vfio/")) {
        return std.fmt.parseInt(u32, target["/dev/vfio/".len..], 10) catch null;
    }

    return null;
}
