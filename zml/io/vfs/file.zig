const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

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

fn useDirectIO(file: std.Io.File) DirectIoError!bool {
    if (comptime canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        return (flags & direct_flag) != 0;
    } else {
        return DirectIoError.UnsupportedPlatform;
    }
}

fn switchToBufferedIO(file: std.fs.File) DirectIoError!void {
    if (comptime canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        if ((flags & direct_flag) == 0) return;

        const result = try std.posix.fcntl(file.handle, std.posix.F.SETFL, flags & ~@as(c_uint, @bitCast(@as(u32, @intCast(direct_flag)))));
        if (result != 0) return DirectIoError.UnexpectedFcntlResult;
    } else {
        return DirectIoError.UnsupportedPlatform;
    }
}

fn switchToDirectIO(file: std.Io.File) DirectIoError!void {
    if (builtin.os.tag == .linux and canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });

        const result = try std.posix.fcntl(file.handle, std.posix.F.SETFL, flags | direct_flag);
        if (result != 0) return DirectIoError.UnexpectedFcntlResult;
    } else {
        return DirectIoError.UnsupportedPlatform;
    }
}

pub const File = struct {
    pub const Config = struct {
        direct_io: bool = true,
        direct_io_alignment: std.mem.Alignment = .fromByteUnits(4 * 1024),
    };

    const Handle = struct { inner_handle: std.Io.File.Handle, direct_io_handle: ?std.Io.File.Handle };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .{},
    config: Config,
    base: VFSBase,

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, config: Config) File {
        return .{
            .allocator = allocator,
            .config = config,
            .base = .init(inner),
        };
    }

    pub fn deinit(self: *File) void {
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);
    }

    pub fn io(self: *File) std.Io {
        return .{
            .userdata = &self.base,
            .vtable = &comptime VFSBase.vtable(.{
                .dirOpenDir = dirOpenDir,
                .dirStat = dirStat,
                .dirStatFile = dirStatFile,
                .dirAccess = dirAccess,
                .dirOpenFile = dirOpenFile,
                .dirClose = dirClose,
                .dirRead = dirRead,
                .dirRealPath = dirRealPath,
                .dirRealPathFile = dirRealPathFile,
                .fileStat = fileStat,
                .fileLength = fileLength,
                .fileClose = fileClose,
                .fileReadStreaming = fileReadStreaming,
                .fileReadPositional = fileReadPositional,
                .fileSeekBy = fileSeekBy,
                .fileSeekTo = fileSeekTo,
                .fileRealPath = fileRealPath,
            }),
        };
    }

    fn openHandle(self: *File) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| {
            return .{ idx, self.handles.at(idx) };
        }
        return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *File, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *File, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        return self.handles.at(@intCast(file.handle));
    }

    fn canOpenWithDirectIO(self: *File, file: std.Io.File, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!bool {
        const file_stat = self.base.inner.vtable.fileStat(self.base.inner.userdata, file) catch |err| switch (err) {
            else => return std.Io.File.OpenError.Unexpected,
        };

        return self.config.direct_io and flags.mode == .read_only and file_stat.size >= self.config.direct_io_alignment.toByteUnits();
    }

    fn innerFile(self: *File, file: std.Io.File, data: []const []u8, position: u64) std.Io.File {
        const handle = self.getFileHandle(file);
        const direct_io_handle = handle.direct_io_handle orelse return .{ .handle = handle.inner_handle };

        const alignment_bytes: usize = self.config.direct_io_alignment.toByteUnits();

        if (!std.mem.isAligned(@as(usize, position), alignment_bytes)) {
            log.warn("<<< use inner position postion={d} alignment_bytes={d}", .{ @as(usize, position), alignment_bytes });
            return .{ .handle = handle.inner_handle };
        }

        var total_size: usize = 0;
        for (data) |buf| {
            if (!std.mem.isAligned(@intFromPtr(buf.ptr), alignment_bytes)) {
                log.warn("<<< use inner buf buf={*} alignment_bytes={d}", .{ buf, alignment_bytes });
                return .{ .handle = handle.inner_handle };
            }
            total_size += buf.len;
        }

        if (!std.mem.isAligned(total_size, alignment_bytes)) {
            log.warn("<<< use inner total total_size={d} alignment_bytes={d}", .{ total_size, alignment_bytes });
            return .{ .handle = handle.inner_handle };
        }

        // log.warn(">>> use direct io", .{});
        return .{ .handle = direct_io_handle };
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirOpenDir(self.base.inner.userdata, dir, sub_path, options);
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirStat(self.base.inner.userdata, dir);
    }

    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirStatFile(self.base.inner.userdata, dir, sub_path, options);
    }

    fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirAccess(self.base.inner.userdata, dir, sub_path, options);
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const inner_file = try self.base.inner.vtable.dirOpenFile(self.base.inner.userdata, dir, sub_path, flags);
        errdefer inner_file.close(self.base.inner);

        const use_direct_io = try self.canOpenWithDirectIO(inner_file, flags);

        var direct_io_handle: ?std.Io.File.Handle = null;
        if (use_direct_io) {
            const direct_io_file = try self.base.inner.vtable.dirOpenFile(self.base.inner.userdata, dir, sub_path, flags);
            defer if (direct_io_handle == null) direct_io_file.close(self.base.inner);

            switchToDirectIO(direct_io_file) catch |err| switch (err) {
                DirectIoError.UnsupportedPlatform => {
                    log.debug("Direct I/O is not supported on this platform", .{});
                },
                else => {
                    log.err("Failed to switch to Direct I/O mode: {any}", .{err});
                    return std.Io.File.OpenError.Unexpected;
                },
            };

            if (useDirectIO(direct_io_file) catch false) {
                direct_io_handle = direct_io_file.handle;
            }
        }

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = .{
            .inner_handle = inner_file.handle,
            .direct_io_handle = direct_io_handle,
        };

        return .{ .handle = @intCast(idx) };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirClose(self.base.inner.userdata, dirs);
    }

    fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirRead(self.base.inner.userdata, reader, entries);
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirRealPath(self.base.inner.userdata, dir, out_buffer);
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.base.inner.vtable.dirRealPathFile(self.base.inner.userdata, dir, path_name, out_buffer);
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const inner_file: std.Io.File = .{ .handle = handle.inner_handle };
        return self.base.inner.vtable.fileStat(self.base.inner.userdata, inner_file);
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const inner_file: std.Io.File = .{ .handle = handle.inner_handle };
        return self.base.inner.vtable.fileLength(self.base.inner.userdata, inner_file);
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            const handle = self.getFileHandle(file);

            if (handle.direct_io_handle) |dio_handle| {
                const direct_io_file: std.Io.File = .{ .handle = dio_handle };
                self.base.inner.vtable.fileClose(self.base.inner.userdata, &.{direct_io_file});
            }

            const inner_file: std.Io.File = .{ .handle = handle.inner_handle };
            self.base.inner.vtable.fileClose(self.base.inner.userdata, &.{inner_file});

            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const position: u64 = @intCast(std.posix.system.lseek(handle.inner_handle, 0, std.posix.SEEK.CUR));
        const inner_file = self.innerFile(file, data, position);
        return self.base.inner.vtable.fileReadStreaming(self.base.inner.userdata, inner_file, data);
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const inner_file = self.innerFile(file, data, offset);
        return self.base.inner.vtable.fileReadPositional(self.base.inner.userdata, inner_file, data, offset);
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const inner_file: std.Io.File = .{ .handle = handle.inner_handle };
        return self.base.inner.vtable.fileSeekBy(self.base.inner.userdata, inner_file, relative_offset);
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const inner_file: std.Io.File = .{ .handle = handle.inner_handle };
        return self.base.inner.vtable.fileSeekTo(self.base.inner.userdata, inner_file, absolute_offset);
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *File = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const inner_file: std.Io.File = .{ .handle = handle.inner_handle };
        return self.base.inner.vtable.fileRealPath(self.base.inner.userdata, inner_file, out_buffer);
    }
};
