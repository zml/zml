const std = @import("std");
const units = @import("units.zig");

/// A directory of /sys, /proc or /dev, kept open for the life of a device.
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

    /// Trimmed content of a file. The slice points into `buffer`.
    /// A file that does not fit is error.StreamTooLong, never a silent cut.
    pub fn read(self: Dir, sub_path: []const u8, buffer: []u8) ![]const u8 {
        return std.mem.trim(u8, try self.readPrefix(sub_path, buffer, true), &std.ascii.whitespace);
    }

    /// The first bytes of a file, cut at `buffer.len`: for /proc/<pid>/cmdline.
    pub fn head(self: Dir, sub_path: []const u8, buffer: []u8) ![]const u8 {
        return self.readPrefix(sub_path, buffer, false);
    }

    fn readPrefix(self: Dir, sub_path: []const u8, buffer: []u8, whole_file: bool) ![]const u8 {
        const file = try self.handle.openFile(self.io, sub_path, .{});
        defer file.close(self.io);
        var reader = file.readerStreaming(self.io, &.{});
        const len = reader.interface.readSliceShort(buffer) catch return reader.err.?;
        if (whole_file and len == buffer.len) {
            var extra: [1]u8 = undefined;
            if ((reader.interface.readSliceShort(&extra) catch return reader.err.?) != 0) {
                return error.StreamTooLong;
            }
        }
        return buffer[0..len];
    }

    /// A number, in decimal or in hex with a 0x prefix.
    pub fn int(self: Dir, comptime T: type, sub_path: []const u8) !T {
        var buffer: [64]u8 = undefined;
        return parseInt(T, try self.read(sub_path, &buffer));
    }

    /// `int`, converted between unit prefixes.
    pub fn scaled(self: Dir, sub_path: []const u8, from: units.Prefix, to: units.Prefix) !u64 {
        return units.scale(try self.int(u64, sub_path), from, to);
    }

    /// Value of `key` in a "key: value" file such as /proc/meminfo.
    /// Reads line by line, with `buffer` as the line.
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

/// Whether a path exists: the usual body of `detect`.
pub fn exists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// A number, in decimal or in hex with a 0x prefix.
pub fn parseInt(comptime T: type, text: []const u8) !T {
    if (std.mem.cutPrefix(u8, text, "0x")) |digits| return std.fmt.parseInt(T, digits, 16);
    return std.fmt.parseInt(T, text, 10);
}

/// The first number of a value: "100 ns" is 100.
pub fn firstNumber(text: []const u8) ?u64 {
    var words = std.mem.tokenizeAny(u8, text, " \t");
    return std.fmt.parseInt(u64, words.next() orelse return null, 10) catch null;
}

/// The value of `key` on a "key: value" or "key=value" line.
pub fn fieldValue(line: []const u8, key: []const u8) ?[]const u8 {
    const rest = std.mem.cutPrefix(u8, line, key) orelse return null;
    if (rest.len > 0 and std.mem.indexOfScalar(u8, ":= \t", rest[0]) == null) return null;
    return std.mem.trim(u8, rest, ":= \t");
}
