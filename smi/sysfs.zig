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
    return nthTokenInt(try findField(io, path, key, &buf), 0);
}

pub fn nthTokenInt(data: []const u8, n: usize) u64 {
    var i: usize = 0;

    var iter = std.mem.tokenizeAny(u8, data, " \t");
    while (iter.next()) |tok| : (i += 1) {
        if (i == n) return std.fmt.parseInt(u64, tok, 10) catch 0;
    }

    return 0;
}

pub fn readFieldString(io: std.Io, path: []const u8, comptime key: []const u8) ![256]u8 {
    var buf: [4096]u8 = undefined;
    const value = try findField(io, path, key, &buf);
    var result: [256]u8 = .{0} ** 256;
    const trimmed = std.mem.trimEnd(u8, value, &std.ascii.whitespace);
    const len = @min(trimmed.len, 255);
    @memcpy(result[0..len], trimmed[0..len]);

    return result;
}

fn findField(io: std.Io, path: []const u8, comptime key: []const u8, buf: *[4096]u8) ![]const u8 {
    const data = try std.Io.Dir.readFile(.cwd(), io, path, buf);

    var iter = std.mem.splitScalar(u8, data, '\n');
    while (iter.next()) |line| {
        if (std.mem.startsWith(u8, line, key)) {
            return std.mem.trimStart(u8, line[key.len..], " :\t");
        }
    }

    return error.NotFound;
}
