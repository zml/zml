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
    if (comptime builtin.target.os.tag == .linux) {
        return @hasField(std.posix.O, "DIRECT");
    }
    return false;
}

fn useDirectIO(file: std.Io.File) DirectIoError!bool {
    if (canUseDirectIO()) {
        const flags = try std.posix.fcntl(file.handle, std.posix.F.GETFL, 0);
        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        return (flags & direct_flag) != 0;
    } else {
        return DirectIoError.UnsupportedPlatform;
    }
}

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
        vtable.fileReadStreaming = fileReadStreaming;

        return .{
            .allocator = allocator,
            .direct_io_map = .{},
            .mutex = .init,
            .config = config,
            .inner = inner,
            .vtable = vtable,
        };
    }

    pub fn deinit(self: *File) void {
        self.direct_io_map.deinit(self.allocator);
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

    fn innerFile(
        self: *File,
        file: std.Io.File,
        buffers: [][]u8,
        position: u64,
    ) std.Io.File {
        const alignment_bytes: usize = self.config.direct_io_alignment.toByteUnits();
        var buffers_aligned = true;

        for (buffers) |buf| {
            const ptr_addr = @intFromPtr(buf.ptr);
            if (!std.mem.isAligned(ptr_addr, alignment_bytes) or
                !std.mem.isAligned(buf.len, alignment_bytes) or
                (buf.len < alignment_bytes))
            {
                buffers_aligned = false;
                break;
            }
        }

        const pos_aligned = std.mem.isAligned(@as(usize, position), alignment_bytes);

        self.mutex.lockUncancelable(self.inner);
        const direct_io_handle = self.direct_io_map.get(file.handle);
        self.mutex.unlock(self.inner);

        return if (direct_io_handle) |handle| blk: {
            if (buffers_aligned and pos_aligned) {
                break :blk std.Io.File{ .handle = handle };
            } else {
                break :blk file;
            }
        } else blk: {
            break :blk file;
        };
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

    fn fileReadPositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]u8,
        position: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        const inner_file = self.innerFile(file, buffer, position);

        return self.inner.vtable.fileReadPositional(self.inner.userdata, inner_file, buffer, position);
    }

    fn fileReadStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]u8,
    ) std.Io.File.Reader.Error!usize {
        const self: *File = @ptrCast(@alignCast(userdata orelse unreachable));
        const position = std.posix.lseek_CUR_get(file.handle) catch return std.Io.File.Reader.Error.Unexpected;
        const inner_file = self.innerFile(file, buffer, position);

        return self.inner.vtable.fileReadStreaming(self.inner.userdata, inner_file, buffer);
    }
};
