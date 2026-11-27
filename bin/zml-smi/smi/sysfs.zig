const std = @import("std");

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
        const text = try self.handle.readFile(self.io, sub_path, buffer);
        if (text.len == buffer.len) return error.StreamTooLong;
        return std.mem.trim(u8, text, &std.ascii.whitespace);
    }

    pub fn int(self: Dir, comptime T: type, sub_path: []const u8) !T {
        var buffer: [64]u8 = undefined;
        return std.fmt.parseInt(T, try self.read(sub_path, &buffer), 0);
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

            if (fieldValue(line, key)) |value| {
                return value;
            }
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

pub fn firstNumber(text: []const u8) ?u64 {
    var words = std.mem.tokenizeAny(u8, text, " \t");
    return std.fmt.parseInt(u64, words.next() orelse return null, 10) catch null;
}

pub fn fieldValue(line: []const u8, key: []const u8) ?[]const u8 {
    const rest = std.mem.cutPrefix(u8, line, key) orelse return null;

    if (rest.len > 0 and std.mem.findScalar(u8, ":= \t", rest[0]) == null) return null;
    return std.mem.trim(u8, rest, ":= \t");
}

test fieldValue {
    try std.testing.expectEqualStrings("0000:03:00.0", fieldValue("PCI_SLOT_NAME=0000:03:00.0", "PCI_SLOT_NAME").?);
    try std.testing.expectEqualStrings("AMD Ryzen", fieldValue("model name\t: AMD Ryzen", "model name").?);
    try std.testing.expectEqual(null, fieldValue("model names: x", "model name"));
}
