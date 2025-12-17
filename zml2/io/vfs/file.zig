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

const MAX_DIRECT_IO_BUFFER_SIZE = 264 * 1024 * 1024; // 128 MB example

pub const File = struct {
    pub const Config = struct {
        direct_io: bool = false,
    };

    inner: std.Io,
    vtable: std.Io.VTable,

    allocator: std.mem.Allocator,

    direct_io_buffer: ?[]u8,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, config: Config) !File {
        var vtable = inner.vtable.*;
        vtable.dirMake = dirMake;
        vtable.dirMakePath = dirMakePath;
        vtable.dirMakeOpenPath = dirMakeOpenPath;
        vtable.dirStatPath = dirStatPath;
        vtable.dirAccess = dirAccess;
        vtable.dirCreateFile = dirCreateFile;
        vtable.dirOpenFile = dirOpenFile;
        vtable.dirOpenDir = dirOpenDir;
        vtable.fileReadPositional = fileReadPositional;

        return .{
            .allocator = allocator,
            .inner = inner,
            .vtable = vtable,
            .config = config,
            .direct_io_buffer = if (config.direct_io) try allocator.alignedAlloc(u8, .fromByteUnits(8 * 1024), MAX_DIRECT_IO_BUFFER_SIZE) else null,
        };
    }

    pub fn deinit(self: *File) void {
        if (self.direct_io_buffer) |buffer| {
            self.allocator.free(buffer);
            self.direct_io_buffer = null;
        }
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
            // Only enable Direct I/O for regular files opened for reading and that
            // are at least one block in size. This avoids breaking reads of small
            // text files (JSON index) which are typically read with unaligned buffers.
            if (self.inner.vtable.fileStat(self.inner.userdata, file)) |st| {
                const is_reg = st.kind == .file;
                const size_ok = st.size >= 4096;
                if (is_reg and flags.mode == .read_only and size_ok) {
                    const result = switchToDirectIO(file) catch |err| {
                        log.err("Failed to switch to Direct I/O mode: {any}", .{err});
                        file.close(self.inner);
                        return std.Io.File.OpenError.Unexpected;
                    };
                    log.warn("Opened file {s} with Direct I/O mode: {}", .{ stripScheme(sub_path), result });
                } else {
                    log.debug("Skipping Direct I/O for {s} (kind={}, size={d})", .{ stripScheme(sub_path), st.kind, st.size });
                }
            } else |err| {
                // Couldn't stat the file — be conservative and skip Direct I/O.
                log.debug("Could not stat file {s}: {any} — skipping Direct I/O", .{ stripScheme(sub_path), err });
            }
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

    const preadv_sym = if (std.posix.lfs64_abi) std.posix.system.preadv64 else std.posix.system.preadv;

    fn fileReadPositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]u8,
        position: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));

        if (self.config.direct_io) {
            const alignment_bytes: u64 = 4 * 1024; // 4 KB

            const aligned_pos = std.mem.alignBackward(u64, position, alignment_bytes);
            const post_offset_in_alignment = position - aligned_pos;

            var total_requested_len: usize = 0;
            for (buffer) |buf| {
                total_requested_len += buf.len;
            }

            // Calculate the total length needed for the aligned read operation.
            // This includes the offset from `aligned_pos` to `position` plus the user's requested length,
            // all forward-aligned to `alignment_bytes`.
            const needed_aligned_len_raw = post_offset_in_alignment + total_requested_len;
            const aligned_read_len = std.mem.alignForward(usize, needed_aligned_len_raw, alignment_bytes);

            // Check if the aligned read length exceeds our stack buffer limit.
            if (aligned_read_len > MAX_DIRECT_IO_BUFFER_SIZE) {
                // Depending on requirements, you might want to:
                // 1. Return an error (as done here).
                // 2. Read in chunks (which would involve looping and potentially more complex state).
                // 3. Revert to non-direct I/O.
                std.log.err("Direct I/O read size {d} exceeds MAX_DIRECT_IO_BUFFER_SIZE {d}", .{ aligned_read_len, MAX_DIRECT_IO_BUFFER_SIZE });
                return error.SystemResources; // Or a more specific error like error.BufferTooSmall
            }

            // Stack-allocate a buffer for the aligned read.
            // This buffer needs to be aligned. 'var' in Zig often places on stack.
            // `@align` ensures the variable is aligned as specified.
            // Slice the buffer to the actual size we need for this specific read.
            const temp_read_buf_slice = self.direct_io_buffer.?[0..aligned_read_len];

            // Create the iovec for the aligned buffer.
            var iovecs_buffer: [1]std.posix.iovec = undefined; // Only one iovec for our single aligned buffer
            iovecs_buffer[0] = .{ .base = temp_read_buf_slice.ptr, .len = temp_read_buf_slice.len };
            const dest = iovecs_buffer[0..1];

            while (true) {
                const rc = preadv_sym(file.handle, dest.ptr, @intCast(dest.len), @bitCast(aligned_pos));
                switch (std.posix.errno(rc)) {
                    .SUCCESS => {
                        const bytes_read_total_into_temp: u64 = @intCast(rc);

                        // Calculate the number of bytes effectively read *after* the initial offset.
                        const effective_bytes_after_offset = if (bytes_read_total_into_temp > post_offset_in_alignment)
                            bytes_read_total_into_temp - post_offset_in_alignment
                        else
                            0;

                        // Determine how many bytes we can actually copy to the user's buffer.
                        const bytes_to_copy_to_user = @min(effective_bytes_after_offset, total_requested_len);

                        // Copy data from the temporary aligned buffer to the user's potentially unaligned buffers
                        var current_offset_in_temp_read = post_offset_in_alignment;
                        var copied_bytes: usize = 0;

                        for (buffer) |buf| {
                            const bytes_remaining_to_copy = bytes_to_copy_to_user - copied_bytes;
                            if (bytes_remaining_to_copy == 0) break;

                            const bytes_to_copy_for_this_buf = @min(buf.len, bytes_remaining_to_copy);
                            if (bytes_to_copy_for_this_buf == 0) continue; // Skip empty user buffers

                            @memcpy(buf[0..bytes_to_copy_for_this_buf], temp_read_buf_slice[current_offset_in_temp_read .. current_offset_in_temp_read + bytes_to_copy_for_this_buf]);

                            current_offset_in_temp_read += bytes_to_copy_for_this_buf;
                            copied_bytes += bytes_to_copy_for_this_buf;
                        }
                        return copied_bytes;
                    },
                    .INTR => continue,
                    .CANCELED => return error.Canceled,

                    .INVAL => |_| return error.Unexpected,
                    .FAULT => |_| return error.Unexpected,
                    .SRCH => return error.ProcessNotFound,
                    .AGAIN => return error.WouldBlock,
                    .BADF => |_| return error.Unexpected,
                    .IO => return error.InputOutput,
                    .ISDIR => return error.IsDir,
                    .NOBUFS => return error.SystemResources,
                    .NOMEM => return error.SystemResources,
                    .NOTCONN => return error.SocketUnconnected,
                    .CONNRESET => return error.ConnectionResetByPeer,
                    .TIMEDOUT => return error.Timeout,
                    .NXIO => return error.Unseekable,
                    .SPIPE => return error.Unseekable,
                    .OVERFLOW => return error.Unseekable,
                    else => |err| return std.posix.unexpectedErrno(err),
                }
            }
        }

        return self.inner.vtable.fileReadPositional(self.inner.userdata, file, buffer, position);
    }
};
