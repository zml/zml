const std = @import("std");
const pi = @import("../../info/process_info.zig");
const ProcessInfo = pi.ProcessInfo;
const DeviceInfo = @import("../../info/device_info.zig").DeviceInfo;
const worker = @import("../../worker.zig");

const max_owners: usize = 64;

const TpuOwner = struct {
    pid: u32 = 0,
    device_idx: u8 = 0,
};

var cache: [2][max_owners]TpuOwner = .{.{TpuOwner{}} ** max_owners} ** 2;
var cache_counts: [2]u32 = .{ 0, 0 };
var current: std.atomic.Value(u8) = .init(0);
var g_devices_per_chip: u32 = 1;
var g_device_infos: [64]*DeviceInfo = undefined;
var g_device_count: u32 = 0;

pub fn init(io: std.Io, devices_per_chip: u32, device_infos: []*DeviceInfo) !void {
    g_devices_per_chip = devices_per_chip;
    const n = @min(device_infos.len, 64);
    @memcpy(g_device_infos[0..n], device_infos[0..n]);
    g_device_count = @intCast(n);
    try worker.spawnCustomWorker(io, scanLoop, .{io});
}

fn scanLoop(io: std.Io) void {
    const interval: std.Io.Duration = .fromMilliseconds(2000);
    while (worker.isRunning()) {
        scan(io);
        io.sleep(interval, .awake) catch {};
    }
}

fn scan(io: std.Io) void {
    const back: u8 = 1 - current.load(.acquire);
    var n: u32 = 0;

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
                if (n >= max_owners) break;
                cache[back][n] = .{
                    .pid = pid,
                    .device_idx = @intCast(chip_idx * g_devices_per_chip),
                };
                n += 1;
                break;
            }
        }
    }

    cache_counts[back] = n;
    current.store(back, .release);
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

pub fn enrichProcesses(result: []ProcessInfo) void {
    const idx = current.load(.acquire);
    const entries = cache[idx][0..cache_counts[idx]];
    const infos = g_device_infos[0..g_device_count];

    for (entries) |entry| {
        if (entry.pid == 0) continue;
        for (result) |*info| {
            if (info.pid == entry.pid) {
                info.device_idx = entry.device_idx;

                // Read memory/util from device info (already updated by metrics workers)
                if (entry.device_idx < infos.len) {
                    const tpu = infos[entry.device_idx].tpu;
                    if (tpu.mem_used_bytes) |mem| {
                        info.gpu_mem_kib = @intCast(mem / 1024);
                    }
                    if (tpu.util_percent) |util| {
                        info.gpu_util_percent = @intCast(util);
                    }
                }
            }
        }
    }
}
