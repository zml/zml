const std = @import("std");

pub fn readInt(io: std.Io, path: []const u8) !u64 {
    var buf: [64]u8 = undefined;
    const data = try std.Io.Dir.readFile(.cwd(), io, path, &buf);
    return std.fmt.parseInt(u64, std.mem.trimEnd(u8, data, &std.ascii.whitespace), 10);
}

pub fn readString(io: std.Io, path: []const u8) ![256]u8 {
    var result: [256]u8 = .{0} ** 256;
    const data = try std.Io.Dir.readFile(.cwd(), io, path, &result);
    result[std.mem.trimEnd(u8, data, &std.ascii.whitespace).len] = 0;
    return result;
}

pub fn readFieldInt(io: std.Io, path: []const u8, comptime key: []const u8) !u64 {
    var buf: [4096]u8 = undefined;
    const data = try std.Io.Dir.readFile(.cwd(), io, path, &buf);
    var iter = std.mem.splitScalar(u8, data, '\n');
    while (iter.next()) |line| {
        if (std.mem.startsWith(u8, line, key)) {
            const rest = std.mem.trimStart(u8, line[key.len..], &(.{' '} ++ .{':'} ++ .{'\t'}));
            const end = std.mem.indexOfAny(u8, rest, &(.{' '} ++ .{'\t'} ++ .{'\n'})) orelse rest.len;
            return std.fmt.parseInt(u64, rest[0..end], 10);
        }
    }
    return error.NotFound;
}

pub fn readFieldString(io: std.Io, path: []const u8, comptime key: []const u8) ![256]u8 {
    var buf: [4096]u8 = undefined;
    const data = try std.Io.Dir.readFile(.cwd(), io, path, &buf);
    var iter = std.mem.splitScalar(u8, data, '\n');
    while (iter.next()) |line| {
        if (std.mem.startsWith(u8, line, key)) {
            var result: [256]u8 = .{0} ** 256;
            const rest = std.mem.trimStart(u8, line[key.len..], &(.{' '} ++ .{':'} ++ .{'\t'}));
            const trimmed = std.mem.trimEnd(u8, rest, &std.ascii.whitespace);
            const len = @min(trimmed.len, 255);
            @memcpy(result[0..len], trimmed[0..len]);
            return result;
        }
    }
    return error.NotFound;
}
