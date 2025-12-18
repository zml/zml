const builtin = @import("builtin");
const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/file");

pub const DirectIoError = error{
    UnsupportedPlatform,
    MisalignedBuffer,
    BufferTooSmall,
    UnexpectedFcntlResult,
} || std.posix.FcntlError;

fn canUseDirectIO() bool {
    if (builtin.target.os.tag == .linux) {
        return @hasField(std.posix.O, "DIRECT");
    }
    return false;
}

// Check if the given file descriptor is currently in Direct I/O mode.
fn useDirectIO(file: std.Io.File) DirectIoError!bool {
    if (canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        return (flags & direct_flag) != 0;
    } else {
        return DirectIoError.UnsupportedPlatform;
    }
}

// Switch the given file descriptor to use buffered I/O mode.
fn switchToBufferedIO(file: std.fs.File) DirectIoError!void {
    if (canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        if ((flags & direct_flag) == 0) return;

        const result = try std.posix.fcntl(file.handle, std.posix.F.SETFL, flags & ~@as(c_uint, @bitCast(@as(u32, @intCast(direct_flag)))));
        if (result != 0) return DirectIoError.UnexpectedFcntlResult;
    } else {
        return DirectIoError.UnsupportedPlatform;
    }
}

// Switch the given file descriptor to use Direct I/O mode.
fn switchToDirectIO(file: std.Io.File) DirectIoError!void {
    if (canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });

        const result = try std.posix.fcntl(file.handle, std.posix.F.SETFL, flags | direct_flag);
        if (result != 0) return DirectIoError.UnexpectedFcntlResult;
    } else {
        return DirectIoError.UnsupportedPlatform;
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
        direct_io_alignment: std.mem.Alignment = .fromByteUnits(4 * 1024),
    };

    allocator: std.mem.Allocator,
    direct_io_map: std.AutoHashMapUnmanaged(std.Io.File.Handle, std.Io.File.Handle),
    mutex: std.Io.Mutex,
    config: Config,

    inner: std.Io,
    vtable: std.Io.VTable,

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, config: Config) File {
        var vtable = inner.vtable.*;
        vtable.dirMake = dirMake;
        vtable.dirMakePath = dirMakePath;
        vtable.dirMakeOpenPath = dirMakeOpenPath;
        vtable.dirStatPath = dirStatPath;
        vtable.dirAccess = dirAccess;
        vtable.dirCreateFile = dirCreateFile;
        vtable.dirOpenFile = dirOpenFile;
        vtable.fileClose = fileClose;
        vtable.dirOpenDir = dirOpenDir;
        vtable.fileReadPositional = fileReadPositional;

        return .{
            .allocator = allocator,
            .direct_io_map = .{},
            .mutex = .init,
            .config = config,
            .inner = inner,
            .vtable = vtable,
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

    fn canOpenWithDirectIO(self: *File, file: std.Io.File, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!bool {
        const file_stat = self.inner.vtable.fileStat(self.inner.userdata, file) catch |err| switch (err) {
            else => return std.Io.File.OpenError.Unexpected,
        };

        return self.config.direct_io and flags.mode == .read_only and file_stat.size >= self.config.direct_io_alignment.toByteUnits();
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

        const use_direct_io = try self.canOpenWithDirectIO(file, flags);
        errdefer file.close(self.inner);

        if (use_direct_io) {
            const direct_io_file = try self.inner.vtable.dirOpenFile(self.inner.userdata, dir, stripScheme(sub_path), flags);
            errdefer direct_io_file.close(self.inner);

            switchToDirectIO(direct_io_file) catch |err| {
                log.err("Failed to switch to Direct I/O mode: {any}", .{err});
                return std.Io.File.OpenError.Unexpected;
            };

            self.mutex.lockUncancelable(self.inner);
            defer self.mutex.unlock(self.inner);

            self.direct_io_map.put(self.allocator, file.handle, direct_io_file.handle) catch |err| {
                log.err("Failed to insert Direct I/O file into map: {any}", .{err});
                return std.Io.File.OpenError.Unexpected;
            };
        }

        return file;
    }

    fn fileClose(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));

        self.mutex.lockUncancelable(self.inner);
        const direct_io_handle = self.direct_io_map.fetchRemove(file.handle);
        self.mutex.unlock(self.inner);

        if (direct_io_handle) |handle| {
            const direct_io_file: std.Io.File = .{
                .handle = handle.value,
            };

            self.inner.vtable.fileClose(self.inner.userdata, direct_io_file);
        }

        return self.inner.vtable.fileClose(self.inner.userdata, file);
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

        self.mutex.lockUncancelable(self.inner);
        const direct_io_handle = self.direct_io_map.get(file.handle);
        self.mutex.unlock(self.inner);

        var buffers_aligned = true;
        const alignment_bytes: usize = self.config.direct_io_alignment.toByteUnits();
        for (buffer) |buf| {
            const ptr_addr = @intFromPtr(buf.ptr);
            if (!std.mem.isAligned(ptr_addr, alignment_bytes) or
                !std.mem.isAligned(buf.len, alignment_bytes) or
                (buf.len < alignment_bytes))
            {
                buffers_aligned = false;
                break;
            }
        }

        const pos_is_aligned = std.mem.isAligned(@as(usize, position), alignment_bytes);

        const file_to_use = if (direct_io_handle) |handle| blk: {
            if (buffers_aligned and pos_is_aligned) {
                log.warn("Using Direct I/O for file read at position {d}", .{position});
                break :blk std.Io.File{ .handle = handle };
            } else {
                break :blk file;
            }
        } else blk: {
            break :blk file;
        };

        // // Check if the file descriptor is currently in direct I/O mode.
        // var has_direct_io = false;
        // if (builtin.target.os.tag == .linux) {
        //     if (@hasField(std.posix.O, "DIRECT")) {
        //         const flags = std.posix.fcntl(file.handle, std.posix.F.GETFL, 0) catch 0;
        //         const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        //         has_direct_io = (flags & direct_flag) != 0;
        //     }
        // } else if (builtin.target.os.tag == .macos) {
        //     if (@hasField(std.posix.F, "NOCACHE")) {
        //         const flags = std.posix.fcntl(file.handle, std.posix.F.GETFL, 0) catch 0;
        //         const nocache_flag: c_int = @bitCast(std.posix.F{ .NOCACHE = true });
        //         has_direct_io = (flags & nocache_flag) != 0;
        //     }
        // }
        // if (self.config.direct_io and has_direct_io) {
        //     const max_iovecs = 8;
        //     const alignment_bytes: usize = 4096; // Typical O_DIRECT alignment

        //     // 1. Align the position backwards.
        //     // Note: Since we are reading directly into the user buffer, if 'position'
        //     // is not equal to 'aligned_pos', the data at 'buffer[0]' will correspond
        //     // to 'aligned_pos', effectively shifting the data. The caller must handle this.
        //     const aligned_pos = std.mem.alignBackward(u64, position, alignment_bytes);

        //     var iovecs: [max_iovecs]std.posix.iovec = undefined;
        //     var iovec_count: usize = 0;

        //     for (buffer) |buf| {
        //         if (iovec_count >= max_iovecs) break;

        //         // 2. Filter invalid buffers
        //         if (buf.len < alignment_bytes) {
        //             // Buffer is too small to hold even one block.
        //             // Since we can't do partial block reads with O_DIRECT, we skip or stop.
        //             // Stopping is safer for contiguous memory logic.
        //             break;
        //         }

        //         // 3. Enforce Pointer Alignment
        //         // We cannot fix an unaligned pointer without a copy.
        //         if (@intFromPtr(buf.ptr) % alignment_bytes != 0) {
        //             return error.Unexpected; // Or error.SystemResources / error.Unexpected
        //         }

        //         // 4. Align Length Downwards (Read Less)
        //         // We truncate the read length to be a multiple of the block size.
        //         const aligned_len = std.mem.alignBackward(usize, buf.len, alignment_bytes);

        //         if (aligned_len > 0) {
        //             iovecs[iovec_count] = .{ .base = buf.ptr, .len = aligned_len };
        //             iovec_count += 1;
        //         }
        //     }

        //     if (iovec_count == 0) {
        //         // Either no buffers provided, or all buffers were too small/unaligned.
        //         return 0;
        //     }

        //     const dest = iovecs[0..iovec_count];

        //     while (true) {
        //         const rc = preadv_sym(file.handle, dest.ptr, @intCast(dest.len), @bitCast(aligned_pos));
        //         switch (std.posix.errno(rc)) {
        //             .SUCCESS => return @intCast(rc),
        //             .INTR => continue,
        //             .CANCELED => return error.Canceled,
        //             .INVAL => {
        //                 // This typically happens if pointers/lengths/offsets were not aligned.
        //                 // Since we aligned everything above, this implies the underlying
        //                 // storage might have stricter requirements or other system issues.
        //                 return error.Unexpected;
        //             },
        //             .FAULT => |_| return error.Unexpected,
        //             .SRCH => return error.ProcessNotFound,
        //             .AGAIN => return error.WouldBlock,
        //             .BADF => |_| return error.Unexpected,
        //             .IO => return error.InputOutput,
        //             .ISDIR => return error.IsDir,
        //             .NOBUFS => return error.SystemResources,
        //             .NOMEM => return error.SystemResources,
        //             .NOTCONN => return error.SocketUnconnected,
        //             .CONNRESET => return error.ConnectionResetByPeer,
        //             .TIMEDOUT => return error.Timeout,
        //             .NXIO => return error.Unseekable,
        //             .SPIPE => return error.Unseekable,
        //             .OVERFLOW => return error.Unseekable,
        //             else => |err| return std.posix.unexpectedErrno(err),
        //         }
        //     }
        // }

        return self.inner.vtable.fileReadPositional(self.inner.userdata, file_to_use, buffer, position);
    }
};
