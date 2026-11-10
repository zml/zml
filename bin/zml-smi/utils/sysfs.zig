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

    var line_writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer line_writer.deinit();

    while (true) {
        line_writer.clearRetainingCapacity();
        _ = reader.interface.streamDelimiter(&line_writer.writer, '\n') catch return error.NotFound;
        reader.interface.toss(1);

        const line = line_writer.written();
        if (std.mem.startsWith(u8, line, key)) {
            const owned = line_writer.toOwnedSlice() catch return error.NotFound;
            return std.mem.trimStart(u8, owned[key.len..], " :\t");
        }
    }
}

/// Reads a procfs file into `buf`, truncated to `buf.len`.
///
/// procfs looks the process (and, for fdinfo, the fd) up again on every read,
/// so a file that opened fine fails to read with ENOENT or ESRCH once its
/// process exits or closes that fd. zml-smi hits this constantly on its own
/// fdinfo, whose fds its poller threads open and close. std.Io does not expect
/// these errnos and prints a stack trace in Debug builds, which garbles the
/// TUI, so this reads with plain syscalls and returns error.FileNotFound.
pub fn readProcFile(io: std.Io, path: []const u8, buf: []u8) ![]u8 {
    const file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var len: usize = 0;
    while (len < buf.len) {
        const rc = std.posix.system.read(file.handle, buf[len..].ptr, buf.len - len);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) break;
                len += @intCast(rc);
            },
            .INTR => {},
            .NOENT, .SRCH => return error.FileNotFound,
            else => return error.Unexpected,
        }
    }
    return buf[0..len];
}

pub fn readFirstLine(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer writer.deinit();
    _ = reader.interface.streamDelimiter(&writer.writer, '\n') catch |err| switch (err) {
        error.EndOfStream => 0,
        else => |e| return e,
    };
    return writer.toOwnedSlice();
}
