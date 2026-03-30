const std = @import("std");
const c = std.c;
const AmdSmi = @import("amdsmi.zig");
const pi = @import("../../info/process_info.zig");
const ProcessDoubleBuffer = @import("../../utils/double_buffer.zig").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Worker = @import("../../worker.zig").Worker;

const bdf_len = "0000:00:00.0".len;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *ProcessDoubleBuffer, amdsmi: *const AmdSmi, dev_offset: u8) !void {
    try w.spawn(io, pollLoop, .{ io, w, allocator, list, amdsmi, dev_offset });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *ProcessDoubleBuffer, amdsmi: *const AmdSmi, dev_offset: u8) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    const device_count = amdsmi.deviceCount();
    const pci_slots = allocator.alloc(?[bdf_len]u8, device_count) catch return;
    defer allocator.free(pci_slots);
    for (pci_slots, 0..) |*slot, i| {
        const handle = amdsmi.handleByIndex(@intCast(i)) catch {
            slot.* = null;
            continue;
        };
        slot.* = formatBdf(amdsmi.bdfId(handle) catch {
            slot.* = null;
            continue;
        });
    }

    while (w.isRunning()) {
        const start: std.Io.Timestamp = .now(io, .awake);
        const back = list.back();

        back.clearRetainingCapacity();

        for (0..device_count) |dev_idx| {
            const handle = amdsmi.handleByIndex(@intCast(dev_idx)) catch continue;

            const procs = blk: {
                const saved = c.dup(c.STDERR_FILENO);
                const devnull = c.open("/dev/null", .{ .ACCMODE = .WRONLY });
                if (devnull >= 0) {
                    _ = c.dup2(devnull, c.STDERR_FILENO);
                    _ = c.close(devnull);
                }
                defer if (saved >= 0) {
                    _ = c.dup2(saved, c.STDERR_FILENO);
                    _ = c.close(saved);
                };
                break :blk amdsmi.processList(allocator, handle) catch continue;
            };
            defer {
                if (procs.len > 0) {
                    allocator.free(procs);
                }
            }

            const pci_slot = if (dev_idx < pci_slots.len) &(pci_slots[dev_idx] orelse continue) else continue;

            for (procs) |proc| {
                back.append(allocator, .{
                    .pid = proc.pid,
                    .device_idx = @intCast(dev_idx + dev_offset),
                    .dev_util_percent = @intCast(proc.engine_usage.gfx + proc.engine_usage.enc),
                    .dev_mem_kib = readProcessGpuVram(io, proc.pid, pci_slot),
                }) catch break;
            }
        }

        list.swap();

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

        total_vram_kib += parseFdinfoVram(io, file_path, pci_slot);
    }

    return total_vram_kib;
}

fn parseFdinfoVram(io: std.Io, path: []const u8, pci_slot: *const [bdf_len]u8) u64 {
    var file = std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only }) catch return 0;
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);

    var found_pci = false;
    var vram_kib: ?u64 = null;

    while (true) {
        const line = reader.interface.takeDelimiterExclusive('\n') catch break;
        reader.interface.toss(1);

        if (std.mem.indexOf(u8, line, pci_slot) != null) found_pci = true;

        // Support both new and old fdinfo keys (see amdgpu_fdinfo.c)
        inline for ([_][]const u8{ "drm-memory-vram:\t", "vram mem:\t" }) |key| {
            if (std.mem.startsWith(u8, line, key)) {
                const rest = line[key.len..];
                const end = std.mem.indexOfAny(u8, rest, " \t\n") orelse rest.len;
                vram_kib = std.fmt.parseInt(u64, rest[0..end], 10) catch 0;
            }
        }

        if (found_pci and vram_kib != null) return vram_kib.?;
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
