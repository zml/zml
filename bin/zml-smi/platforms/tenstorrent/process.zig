const std = @import("std");
const smi_info = @import("zml-smi/info");
const pi = smi_info.process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;
const Sysfs = @import("tenstorrent.zig").Sysfs;

const dev_prefix = "/dev/tenstorrent/";

pub fn init(collector: *Collector, list: *ProcessDoubleBuffer, tt: *const Sysfs, dev_offset: u16) !void {
    try collector.spawnPoll(pollOnce, .{ collector.gpa, collector.io, list, tt, dev_offset });
}

fn pollOnce(allocator: std.mem.Allocator, io: std.Io, list: *ProcessDoubleBuffer, tt: *const Sysfs, dev_offset: u16) void {
    scan(allocator, io, list.back(), tt, dev_offset);
    list.swap();
}

fn scan(allocator: std.mem.Allocator, io: std.Io, back: *std.ArrayList(pi.ProcessInfo), tt: *const Sysfs, dev_offset: u16) void {
    back.clearRetainingCapacity();

    const Key = struct { device_idx: u16, pid: u32 };
    var seen: std.AutoHashMapUnmanaged(Key, void) = .{};
    defer seen.deinit(allocator);

    var proc_dir = std.Io.Dir.openDirAbsolute(io, "/proc", .{ .iterate = true }) catch return;
    defer proc_dir.close(io);

    var proc_it = proc_dir.iterate();
    while (proc_it.next(io) catch null) |proc_entry| {
        const pid = std.fmt.parseInt(u32, proc_entry.name, 10) catch continue;

        var fd_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const fd_sub = std.fmt.bufPrint(&fd_path_buf, "{s}/fd", .{proc_entry.name}) catch continue;

        var fd_dir = proc_dir.openDir(io, fd_sub, .{ .iterate = true }) catch continue;
        defer fd_dir.close(io);

        var fd_it = fd_dir.iterate();
        while (fd_it.next(io) catch null) |fd_entry| {
            var link_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const len = fd_dir.readLink(io, fd_entry.name, &link_buf) catch continue;
            const dev_index = parseDeviceIndex(link_buf[0..len]) orelse continue;

            if (dev_index >= tt.devices.len) {
                continue;
            }

            const device_idx: u16 = @intCast(dev_index + dev_offset);
            const key: Key = .{ .device_idx = device_idx, .pid = pid };

            if ((seen.getOrPut(allocator, key) catch continue).found_existing) {
                continue;
            }

            back.append(allocator, .{ .pid = pid, .device_idx = device_idx }) catch return;
        }
    }
}

fn parseDeviceIndex(target: []const u8) ?usize {
    if (!std.mem.startsWith(u8, target, dev_prefix)) {
        return null;
    }

    return std.fmt.parseInt(usize, target[dev_prefix.len..], 10) catch null;
}
