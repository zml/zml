const builtin = @import("builtin");
const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/file");

// Switch the given file descriptor to use buffered I/O mode.
fn switchToBufferedIO(file: std.fs.File) !bool {
    const fd = file.handle;

    const flags = try std.posix.fcntl(fd, std.posix.F.GETFL, 0);

    if (builtin.target.os.tag == .linux) {
        if (!@hasField(std.posix.O, "DIRECT")) {
            return true;
        }

        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        if ((flags & direct_flag) == 0) {
            return true;
        }

        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags & ~@as(c_uint, @bitCast(@as(u32, @intCast(direct_flag)))));
        return result == 0;
    } else if (builtin.target.os.tag == .macos) {
        if (!@hasField(std.posix.F, "NOCACHE")) {
            return true;
        }

        const nocache_flag: c_int = @bitCast(std.posix.F{ .NOCACHE = true });
        if ((flags & nocache_flag) == 0) {
            return true;
        }

        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags & ~@as(c_uint, @bitCast(@as(u32, @intCast(nocache_flag)))));
        return result == 0;
    } else {
        return true;
    }
}

// Switch the given file descriptor to use direct I/O mode.
fn switchToDirectIO(file: std.Io.File) !bool {
    const fd = file.handle;

    const flags = try std.posix.fcntl(fd, std.posix.F.GETFL, 0);

    if (builtin.target.os.tag == .linux) {
        if (!@hasField(std.posix.O, "DIRECT")) {
            return false;
        }

        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags | direct_flag);
        return result == 0;
    } else if (builtin.target.os.tag == .macos) {
        const has_no_cache = @hasField(std.posix.F, "NOCACHE");
        if (!has_no_cache) {
            return false;
        }

        const nocache_flag: c_int = @bitCast(std.posix.F{ .NOCACHE = true });
        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags | nocache_flag);
        return result == 0;
    } else {
        return false;
    }
}

test "switchToDirectIO and switchToBufferedIO" {
    const allocator = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const filename = "file.bin";
    _ = try createBinFile(tmp_dir, filename, 8 * 1024, null);

    const file_path = try tmp_dir.dir.realpathAlloc(allocator, filename);
    defer allocator.free(file_path);

    var file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_write });
    defer file.close();

    // Test buffered I/O
    var unaligned_buf: [10]u8 = undefined;
    try file.seekTo(1);
    const bytes_read_buffered = try file.readAll(&unaligned_buf);
    try std.testing.expectEqual(10, bytes_read_buffered);

    // Switch to Direct I/O and test aligned read
    _ = switchToDirectIO(file);

    const blk_size = 4 * 1024;
    const aligned_buf = try allocator.alignedAlloc(u8, .fromByteUnits(blk_size), blk_size * 2);
    defer allocator.free(aligned_buf);

    try file.seekTo(0);
    const bytes_read_aligned = try file.readAll(aligned_buf);
    try std.testing.expectEqual(aligned_buf.len, bytes_read_aligned);

    // Switch back to Buffered I/O and test unaligned read again
    _ = try switchToBufferedIO(file);

    try file.seekTo(1);
    const bytes_read_buffered_again = try file.readAll(&unaligned_buf);
    try std.testing.expectEqual(unaligned_buf.len, bytes_read_buffered_again);
}

// Utility to create a binary file with a simple byte pattern for testing.
fn createBinFile(tmp_dir: std.testing.TmpDir, filename: []const u8, size: usize, alignment: ?usize) !usize {
    var file = try tmp_dir.dir.createFile(filename, .{});
    defer file.close();

    var writer_buffer: [64 * 1024]u8 = undefined;
    var file_writer = file.writer(&writer_buffer);

    var pattern_chunk: [64 * 1024]u8 = undefined;
    for (&pattern_chunk, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    var remaining_bytes_to_write = if (alignment) |a| std.mem.alignForward(usize, size, a) else size;
    while (remaining_bytes_to_write > 0) {
        const chunk_len = @min(remaining_bytes_to_write, pattern_chunk.len);
        try file_writer.interface.writeAll(pattern_chunk[0..chunk_len]);
        remaining_bytes_to_write -= chunk_len;
    }
    try file_writer.interface.flush();

    return try file_writer.file.getEndPos();
}

pub const File = struct {
    pub const Config = struct {
        direct_io: bool = false,
    };

    inner: std.Io,
    vtable: std.Io.VTable,

    config: Config,

    pub fn init(inner: std.Io, config: Config) File {
        var vtable = inner.vtable.*;
        vtable.dirMake = dirMake;
        vtable.dirMakePath = dirMakePath;
        vtable.dirMakeOpenPath = dirMakeOpenPath;
        vtable.dirStatPath = dirStatPath;
        vtable.dirAccess = dirAccess;
        vtable.dirCreateFile = dirCreateFile;
        vtable.dirOpenFile = dirOpenFile;
        vtable.dirOpenDir = dirOpenDir;

        return .{
            .inner = inner,
            .vtable = vtable,
            .config = config,
        };
    }

    pub fn io(self: *File) std.Io {
        return .{
            .userdata = self,
            .vtable = &self.vtable,
        };
    }

    fn stripScheme(path: []const u8) []const u8 {
        if (std.mem.startsWith(u8, path, "file://")) {
            return path[7..];
        }
        return path;
    }

    fn dirMake(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirMake(self.inner.userdata, dir, stripScheme(sub_path), mode);
    }

    fn dirMakePath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirMakePath(self.inner.userdata, dir, stripScheme(sub_path), mode);
    }

    fn dirMakeOpenPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.MakeOpenPathError!std.Io.Dir {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirMakeOpenPath(self.inner.userdata, dir, stripScheme(sub_path), options);
    }

    fn dirStatPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.StatPathOptions,
    ) std.Io.Dir.StatPathError!std.Io.File.Stat {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirStatPath(self.inner.userdata, dir, stripScheme(sub_path), options);
    }

    fn dirAccess(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.AccessOptions,
    ) std.Io.Dir.AccessError!void {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirAccess(self.inner.userdata, dir, stripScheme(sub_path), options);
    }

    fn dirCreateFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.CreateFlags,
    ) std.Io.File.OpenError!std.Io.File {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirCreateFile(self.inner.userdata, dir, stripScheme(sub_path), flags);
    }

    fn dirOpenFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenError!std.Io.File {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        const file = try self.inner.vtable.dirOpenFile(self.inner.userdata, dir, stripScheme(sub_path), flags);

        if (self.config.direct_io) {
            const result = switchToDirectIO(file) catch |err| {
                log.err("Failed to switch to Direct I/O mode: {any}", .{err});
                file.close(self.inner);
                return std.Io.File.OpenError.Unexpected;
            };

            log.warn(
                "Opened file {s} with Direct I/O mode: {}",
                .{ stripScheme(sub_path), result },
            );
        }

        return file;
    }

    fn dirOpenDir(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        return self.inner.vtable.dirOpenDir(self.inner.userdata, dir, stripScheme(sub_path), options);
    }
};
