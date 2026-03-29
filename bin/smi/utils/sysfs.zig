const std = @import("std");

pub fn readInt(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !u64 {
    const data = try readFirstLine(allocator, io, path);
    return std.fmt.parseInt(u64, data, 10);
}

pub fn readString(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return readFirstLine(allocator, io, path);
}

pub fn readFieldInt(allocator: std.mem.Allocator, io: std.Io, path: []const u8, comptime key: []const u8) !u64 {
    return nthTokenInt(try findField(allocator, io, path, key), 0);
}

pub fn nthTokenInt(data: []const u8, n: usize) u64 {
    var i: usize = 0;

    var iter = std.mem.tokenizeAny(u8, data, " \t");
    while (iter.next()) |tok| : (i += 1) {
        if (i == n) {
            return std.fmt.parseInt(u64, tok, 10) catch 0;
        }
    }

    return 0;
}

pub fn readFieldString(allocator: std.mem.Allocator, io: std.Io, path: []const u8, comptime key: []const u8) ![]const u8 {
    const value = try findField(allocator, io, path, key);
    return std.mem.trimEnd(u8, value, &std.ascii.whitespace);
}

fn findField(allocator: std.mem.Allocator, io: std.Io, path: []const u8, comptime key: []const u8) ![]const u8 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);

    while (true) {
        var line_writer: std.Io.Writer.Allocating = .init(allocator);
        _ = reader.interface.streamDelimiter(&line_writer.writer, '\n') catch return error.NotFound;
        reader.interface.toss(1);

        const line = line_writer.toOwnedSlice() catch return error.NotFound;
        if (std.mem.startsWith(u8, line, key)) {
            return std.mem.trimStart(u8, line[key.len..], " :\t");
        }
    }
}

pub fn readFirstLine(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var writer: std.Io.Writer.Allocating = .init(allocator);
    _ = reader.interface.streamDelimiter(&writer.writer, '\n') catch |err| switch (err) {
        error.EndOfStream => 0,
        else => |e| return e,
    };
    return writer.toOwnedSlice();
}
