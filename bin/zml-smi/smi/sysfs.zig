const std = @import("std");
const units = @import("units.zig");

pub const Dir = struct {
    handle: std.Io.Dir,
    io: std.Io,

    pub fn open(io: std.Io, path: []const u8) !Dir {
        return .{ .handle = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }), .io = io };
    }

    pub fn openDir(self: Dir, sub_path: []const u8) !Dir {
        return .{ .handle = try self.handle.openDir(self.io, sub_path, .{ .iterate = true }), .io = self.io };
    }

    pub fn close(self: Dir) void {
        self.handle.close(self.io);
    }

    pub fn read(self: Dir, sub_path: []const u8, buffer: []u8) ![]const u8 {
        const file = try self.handle.openFile(self.io, sub_path, .{});
        defer file.close(self.io);

        const len = try readAll(file, self.io, buffer);
        var extra: [1]u8 = undefined;
        if (len == buffer.len and try readAll(file, self.io, &extra) != 0) return error.StreamTooLong;
        return std.mem.trim(u8, buffer[0..len], &std.ascii.whitespace);
    }

    pub fn head(self: Dir, sub_path: []const u8, buffer: []u8) ![]const u8 {
        const file = try self.handle.openFile(self.io, sub_path, .{});
        defer file.close(self.io);

        return buffer[0..try readAll(file, self.io, buffer)];
    }

    pub fn int(self: Dir, comptime T: type, sub_path: []const u8) !T {
        var buffer: [64]u8 = undefined;
        return parseInt(T, try self.read(sub_path, &buffer));
    }

    pub fn scaled(self: Dir, sub_path: []const u8, from: units.Prefix, to: units.Prefix) !u64 {
        return units.scale(try self.int(u64, sub_path), from, to);
    }

    pub fn field(self: Dir, sub_path: []const u8, key: []const u8, buffer: []u8) ![]const u8 {
        const file = try self.handle.openFile(self.io, sub_path, .{});
        defer file.close(self.io);

        var reader = file.readerStreaming(self.io, buffer);
        while (true) {
            const line = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
                // A line longer than the buffer is not the one we look for.
                error.StreamTooLong => {
                    _ = reader.interface.discardDelimiterInclusive('\n') catch return error.NotFound;
                    continue;
                },
                error.ReadFailed => return reader.err.?,
            } orelse return error.NotFound;
            if (fieldValue(line, key)) |value| return value;
        }
    }

    pub fn iterate(self: Dir) std.Io.Dir.Iterator {
        return self.handle.iterate();
    }

    pub fn readLink(self: Dir, sub_path: []const u8, buffer: []u8) ![]const u8 {
        return buffer[0..try self.handle.readLink(self.io, sub_path, buffer)];
    }
};

pub fn exists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

pub fn parseInt(comptime T: type, text: []const u8) !T {
    if (std.mem.startsWith(u8, text, "0x")) return std.fmt.parseInt(T, text[2..], 16);
    return std.fmt.parseInt(T, text, 10);
}

pub fn firstNumber(text: []const u8) ?u64 {
    var words = std.mem.tokenizeAny(u8, text, " \t");
    return std.fmt.parseInt(u64, words.next() orelse return null, 10) catch null;
}

pub fn fieldValue(line: []const u8, key: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, line, key)) return null;
    const rest = line[key.len..];
    if (rest.len > 0 and std.mem.indexOfScalar(u8, ":= \t", rest[0]) == null) return null;
    return std.mem.trim(u8, rest, ":= \t");
}

fn readAll(file: std.Io.File, io: std.Io, buffer: []u8) !usize {
    var len: usize = 0;
    while (len < buffer.len) {
        len += file.readStreaming(io, &.{buffer[len..]}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };
    }
    return len;
}

test fieldValue {
    try std.testing.expectEqualStrings("0000:03:00.0", fieldValue("PCI_SLOT_NAME=0000:03:00.0", "PCI_SLOT_NAME").?);
    try std.testing.expectEqualStrings("AMD Ryzen", fieldValue("model name\t: AMD Ryzen", "model name").?);
    try std.testing.expectEqual(null, fieldValue("model names: x", "model name"));
}
