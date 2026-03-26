const std = @import("std");
const str = @import("../../str.zig");

pub fn bytesToMb(val: ?u64) u64 {
    return if (val) |v| v / (1024 * 1024) else 0;
}

pub fn parseLoadAvg(raw: ?[256]u8) [3]f32 {
    const s = if (raw) |*b| str.slice(b) else return .{ 0, 0, 0 };
    var result: [3]f32 = .{ 0, 0, 0 };
    var it = std.mem.splitScalar(u8, s, ' ');
    for (&result) |*r| {
        const token = it.next() orelse break;
        r.* = std.fmt.parseFloat(f32, token) catch 0;
    }
    return result;
}

pub fn formatUptime(arena: std.mem.Allocator, seconds: u64) std.mem.Allocator.Error![]const u8 {
    const days = seconds / 86400;
    const hours = (seconds % 86400) / 3600;
    const mins = (seconds % 3600) / 60;
    if (days > 0) {
        return std.fmt.allocPrint(arena, "{d}d {d}h {d}m", .{ days, hours, mins });
    } else if (hours > 0) {
        return std.fmt.allocPrint(arena, "{d}h {d}m", .{ hours, mins });
    } else {
        return std.fmt.allocPrint(arena, "{d}m", .{mins});
    }
}

pub fn fmtMem(arena: std.mem.Allocator, kib: u64) std.mem.Allocator.Error![]const u8 {
    if (kib >= 1024 * 1024) {
        const whole = kib / (1024 * 1024);
        const frac = (kib % (1024 * 1024)) * 10 / (1024 * 1024);
        return std.fmt.allocPrint(arena, "{d}.{d}G", .{ whole, frac });
    } else if (kib >= 1024) {
        const whole = kib / 1024;
        const frac = (kib % 1024) * 10 / 1024;
        return std.fmt.allocPrint(arena, "{d}.{d}M", .{ whole, frac });
    } else {
        return std.fmt.allocPrint(arena, "{d}K", .{kib});
    }
}

pub fn formatBandwidth(arena: std.mem.Allocator, kbps: u64) std.mem.Allocator.Error![]const u8 {
    if (kbps >= 1_000_000) {
        const gbps = @as(f64, @floatFromInt(kbps)) / 1_000_000.0;
        return std.fmt.allocPrint(arena, " {d:.1} GB/s", .{gbps});
    } else if (kbps >= 1_000) {
        const mbps = @as(f64, @floatFromInt(kbps)) / 1_000.0;
        return std.fmt.allocPrint(arena, " {d:.1} MB/s", .{mbps});
    } else {
        return std.fmt.allocPrint(arena, " {d} KB/s", .{kbps});
    }
}

pub fn repeatStr(arena: std.mem.Allocator, char: []const u8, count: u16) std.mem.Allocator.Error![]const u8 {
    const buf = try arena.alloc(u8, char.len * count);
    for (0..count) |i| {
        @memcpy(buf[i * char.len ..][0..char.len], char);
    }
    return buf;
}

/// Maps values from [min_val, max_val] to [0, 100], clamped.
pub fn normalizeRange(arena: std.mem.Allocator, raw: []const u64, min_val: u64, max_val: u64) std.mem.Allocator.Error![]u8 {
    const result = try arena.alloc(u8, raw.len);
    const range = max_val -| min_val;
    for (raw, 0..) |v, i| {
        result[i] = if (v <= min_val)
            0
        else if (v >= max_val)
            100
        else if (range > 0)
            @intCast((v - min_val) * 100 / range)
        else
            0;
    }
    return result;
}

pub fn trunc(s: []const u8, max: usize) []const u8 {
    return s[0..@min(s.len, max)];
}

pub const CommonDeviceFields = struct {
    name: ?[256]u8,
    util_percent: u8,
    mem_used: u64,
    mem_total: u64,

    pub fn nameSlice(self: *const CommonDeviceFields) []const u8 {
        return str.optSlice(&self.name);
    }
};

pub fn commonDeviceFields(info: anytype) CommonDeviceFields {
    return .{
        .name = info.name,
        .util_percent = @intCast(@min(info.util_percent orelse 0, 100)),
        .mem_used = info.mem_used_bytes orelse 0,
        .mem_total = info.mem_total_bytes orelse 0,
    };
}
