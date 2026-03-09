const std = @import("std");
const amdsmi = @import("amdsmi.zig");
const pi = @import("../../info/process_info.zig");
const Worker = @import("../../worker.zig").Worker;

const bdf_len = "0000:00:00.0".len;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo)) !void {
    try w.spawnCustomWorker(io, pollLoop, .{ io, w, allocator, list });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo)) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    var pci_slots: std.ArrayList([bdf_len]u8) = .{};
    defer pci_slots.deinit(allocator);

    const device_count = amdsmi.getDeviceCount() catch 0;
    pci_slots.ensureTotalCapacity(allocator, device_count) catch {};
    for (0..device_count) |i| {
        const handle = amdsmi.getHandleByIndex(@intCast(i)) catch continue;
        const bdf_id = amdsmi.getBdfId(handle) catch continue;
        pci_slots.appendAssumeCapacity(formatBdf(bdf_id));
    }

    var shadow: std.ArrayList(pi.ProcessInfo) = .{};
    defer shadow.deinit(allocator);

    while (w.isRunning()) {
        shadow.clearRetainingCapacity();

        for (0..device_count) |dev_idx| {
            const handle = amdsmi.getHandleByIndex(@intCast(dev_idx)) catch continue;

            var buf: [64]amdsmi.ProcInfo = undefined;
            const procs = amdsmi.getProcessList(handle, &buf) catch continue;

            const pci_slot = if (dev_idx < pci_slots.items.len) &pci_slots.items[dev_idx] else continue;

            for (procs) |proc| {
                shadow.append(allocator, .{
                    .pid = proc.pid,
                    .device_idx = @intCast(dev_idx),
                    .gpu_util_percent = @intCast(proc.engine_usage.gfx + proc.engine_usage.enc),
                    .gpu_mem_kib = readProcessGpuVram(io, proc.pid, pci_slot),
                }) catch break;
            }
        }

        const tmp = list.*;
        list.* = shadow;
        shadow = tmp;

        io.sleep(interval, .awake) catch {};
    }
}

/// Read per-GPU VRAM from /proc/<pid>/fdinfo/, filtered by PCI slot.
/// amdsmi reports global VRAM across all GPUs; fdinfo gives per-GPU values.
fn readProcessGpuVram(io: std.Io, pid: u32, pci_slot: *const [bdf_len]u8) u64 {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const dir_path = std.fmt.bufPrint(&path_buf, "/proc/{d}/fdinfo", .{pid}) catch return 0;

    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return 0;
    defer dir.close(io);

    var total_vram_kib: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        var file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const file_path = std.fmt.bufPrint(&file_buf, "{s}/{s}", .{ dir_path, entry.name }) catch continue;

        var buf: [4096]u8 = undefined;
        const data = std.Io.Dir.readFile(.cwd(), io, file_path, &buf) catch continue;

        if (std.mem.indexOf(u8, data, "drm-driver:\tamdgpu") == null) continue;
        if (std.mem.indexOf(u8, data, pci_slot) == null) continue;

        total_vram_kib += parseVramKib(data);
    }

    return total_vram_kib;
}

fn parseVramKib(data: []const u8) u64 {
    // Support both new and old fdinfo keys (see amdgpu_fdinfo.c)
    for ([_][]const u8{ "drm-memory-vram:\t", "vram mem:\t" }) |key| {
        const pos = std.mem.indexOf(u8, data, key) orelse continue;
        const rest = data[pos + key.len ..];
        const end = std.mem.indexOfAny(u8, rest, " \t\n") orelse rest.len;
        return std.fmt.parseInt(u64, rest[0..end], 10) catch 0;
    }
    return 0;
}

fn formatBdf(bdf_id: u64) [bdf_len]u8 {
    var buf: [bdf_len]u8 = undefined;
    _ = std.fmt.bufPrint(&buf, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{
        (bdf_id >> 16) & 0xFFFF,
        (bdf_id >> 8) & 0xFF,
        (bdf_id >> 3) & 0x1F,
        bdf_id & 0x7,
    }) catch unreachable;
    return buf;
}
