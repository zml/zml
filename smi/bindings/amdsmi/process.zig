const std = @import("std");
const AmdSmi = @import("amdsmi.zig");
const pi = @import("../../info/process_info.zig");
const ProcessShadowList = @import("../../shadow_list.zig").ShadowList(pi.ProcessInfo);
const Worker = @import("../../worker.zig").Worker;

const bdf_len = "0000:00:00.0".len;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *ProcessShadowList, amdsmi: *const AmdSmi) !void {
    try w.spawn(io, pollLoop, .{ io, w, allocator, list, amdsmi });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *ProcessShadowList, amdsmi: *const AmdSmi) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    var pci_slots: std.ArrayList([bdf_len]u8) = .{};
    defer pci_slots.deinit(allocator);

    const device_count = amdsmi.getDeviceCount();
    pci_slots.ensureTotalCapacity(allocator, device_count) catch {};
    for (0..device_count) |i| {
        const handle = amdsmi.getHandleByIndex(@intCast(i)) catch continue;
        const bdf_id = amdsmi.getBdfId(handle) catch continue;
        pci_slots.appendAssumeCapacity(formatBdf(bdf_id));
    }

    var sl = list.shadow();
    defer sl.deinit(allocator);

    while (w.isRunning()) {
        const start: std.Io.Timestamp = .now(io, .awake);

        sl.clearRetainingCapacity();

        for (0..device_count) |dev_idx| {
            const handle = amdsmi.getHandleByIndex(@intCast(dev_idx)) catch continue;

            const procs = amdsmi.getProcessList(allocator, handle) catch continue;
            defer if (procs.len > 0) allocator.free(procs);

            const pci_slot = if (dev_idx < pci_slots.items.len) &pci_slots.items[dev_idx] else continue;

            for (procs) |proc| {
                sl.append(allocator, .{
                    .pid = proc.pid,
                    .device_idx = @intCast(dev_idx),
                    .dev_util_percent = @intCast(proc.engine_usage.gfx + proc.engine_usage.enc),
                    .dev_mem_kib = readProcessGpuVram(io, proc.pid, pci_slot),
                }) catch break;
            }
        }

        sl.swap(io);

        const elapsed = start.untilNow(io, .awake);
        if (elapsed.nanoseconds < interval.nanoseconds) {
            io.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
        }
    }
}

/// Reads per-GPU VRAM from /proc/<pid>/fdinfo/, filtered by PCI slot.
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
