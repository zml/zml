const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/qaic");

const qualcomm_vendor_id = 0x17cb;
/// Cloud AI 100 (Standard, Pro and Ultra SoCs all report 0xa100).
const aic100_device_id = 0xa100;
/// AIC080 and AIC200: same driver, not targeted by this plugin.
const unsupported_device_ids = [_]u16{ 0xa080, 0xa110 };

pub fn isEnabled() bool {
    return platforms_options.qaic_enabled;
}

fn readHexId(io: std.Io, path: []const u8) ?u16 {
    var buf: [16]u8 = undefined;
    const text = std.Io.Dir.readFile(.cwd(), io, path, &buf) catch return null;
    const trimmed = std.mem.trim(u8, text, " \n");
    const digits = if (std.mem.startsWith(u8, trimmed, "0x")) trimmed[2..] else trimmed;
    return std.fmt.parseInt(u16, digits, 16) catch null;
}

/// Whether sysfs lists an AIC100: never opens the device, never issues an ioctl, sets no environment variable. The
/// plugin does the real enumeration at client create.
fn probeSysfs(io: std.Io) bool {
    var dir = std.Io.Dir.cwd().openDir(io, "/sys/class/accel", .{ .iterate = true }) catch return false;
    defer dir.close(io);

    var found = false;
    var usable: usize = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.startsWith(u8, entry.name, "accel")) continue;
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const vendor_path = std.fmt.bufPrint(&path_buf, "/sys/class/accel/{s}/device/vendor", .{entry.name}) catch continue;
        if (readHexId(io, vendor_path) != qualcomm_vendor_id) continue;
        const device_path = std.fmt.bufPrint(&path_buf, "/sys/class/accel/{s}/device/device", .{entry.name}) catch continue;
        const device_id = readHexId(io, device_path) orelse continue;

        if (device_id != aic100_device_id) {
            if (std.mem.indexOfScalar(u16, &unsupported_device_ids, device_id) != null) {
                log.warn("{s}: Qualcomm device 0x{x} is not supported by the qaic plugin (AIC100 only)", .{ entry.name, device_id });
            }
            continue;
        }

        found = true;
        const node_path = std.fmt.bufPrint(&path_buf, "/dev/accel/{s}", .{entry.name}) catch continue;
        std.Io.Dir.accessAbsolute(io, node_path, .{ .read = true, .write = true }) catch |err| {
            log.warn("{s}: {}: check the qaic driver and the permissions of /dev/accel (container: --device, --group-add)", .{ node_path, err });
            continue;
        };
        usable += 1;
    }
    if (found and usable == 0) {
        log.warn("an AIC100 is present but no /dev/accel node is readable and writable: client create will fail", .{});
    }
    return found;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) {
        return error.Unavailable;
    }

    if (!probeSysfs(io)) return error.Unavailable;

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libzml_qaic/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for the qaic runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libzml_qaic.so" });
        break :blk .loadFrom(path);
    };
}
