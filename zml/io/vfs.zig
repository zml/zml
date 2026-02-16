const std = @import("std");

const vfs = @import("vfs");

const log = std.log.scoped(.@"zml/io/vfs");

pub const VFS = struct {
    pub const File = vfs.File;
    pub const HTTP = vfs.HTTP;
    pub const HF = vfs.HF;
    pub const S3 = vfs.S3;
    const VFSBase = vfs.VFSBase;

    const Handles = std.ArrayList(Handle);
    const ClosedHandles = std.ArrayList(u32);

    const CWD_HANDLE: u32 = 0;

    const Handle = struct { handle: u32, backend_idx: ?usize, flags: std.Io.File.Flags = .{ .nonblocking = false } };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,

    backends: std.StringArrayHashMapUnmanaged(std.Io) = .empty,
    handles: Handles = .{},
    closed_handles: ClosedHandles = .{},

    base: VFSBase,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io) !VFS {
        const base = VFSBase.init(base_io);

        var handles: Handles = .{};
        try handles.append(allocator, .{ .handle = CWD_HANDLE, .backend_idx = null });

        return .{
            .allocator = allocator,
            .handles = handles,
            .base = base,
        };
    }

    pub fn deinit(self: *VFS) void {
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);
        self.backends.deinit(self.allocator);
    }

    pub fn register(self: *VFS, scheme: []const u8, backend: std.Io) std.mem.Allocator.Error!void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        try self.backends.put(self.allocator, scheme, backend);
    }

    pub fn unregister(self: *VFS, scheme: []const u8) bool {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        return self.backends.remove(scheme);
    }

    pub fn io(self: *VFS) std.Io {
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
                .fileReadPositional = fileReadPositional,
                .fileSeekBy = fileSeekBy,
                .fileSeekTo = fileSeekTo,
                .fileRealPath = fileRealPath,
            }),
        };
    }

    fn openHandle(self: *VFS) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| {
            return .{ idx, &self.handles.items[idx] };
        }
        return .{ @intCast(self.handles.items.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *VFS, idx: u32) !void {
        if (idx == CWD_HANDLE) return;

        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *VFS, file: std.Io.File) struct { *Handle, std.Io } {
        self.mutex.lockUncancelable(self.base.inner);
        const handle = &self.handles.items[@intCast(file.handle)];
        self.mutex.unlock(self.base.inner);

        return .{ handle, self.getBackend(handle.backend_idx) };
    }

    fn getDirHandle(self: *VFS, dir: std.Io.Dir) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (std.meta.eql(dir, std.Io.Dir.cwd())) return &self.handles.items[CWD_HANDLE];
        return &self.handles.items[@intCast(dir.handle)];
    }

    fn getScheme(self: *VFS, backend_idx: ?usize) ?[]const u8 {
        if (backend_idx) |idx| return self.backends.entries.items(.key)[idx] else return null;
    }

    fn getBackend(self: *VFS, backend_idx: ?usize) std.Io {
        if (backend_idx) |idx| return self.backends.entries.items(.value)[idx] else return self.base.inner;
    }

    fn lookupDir(self: *VFS, dir: std.Io.Dir, sub_path: ?[]const u8) !struct { ?usize, std.Io.Dir, std.Io } {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            if (sub_path == null) return .{ null, dir, self.base.inner };
            if (std.fs.path.isAbsolutePosix(sub_path.?)) return .{ null, dir, self.base.inner };

            const uri = std.Uri.parse(sub_path.?) catch null;
            if (uri) |u| {
                const backend_idx: ?usize = for (self.backends.entries.items(.key), 0..) |s, idx| {
                    if (std.mem.eql(u8, u.scheme, s)) break idx;
                } else null;
                if (backend_idx == null) return error.VFSNotRegistered;
                return .{ backend_idx, std.Io.Dir.cwd(), self.getBackend(backend_idx) };
            } else {
                return .{ null, std.Io.Dir.cwd(), self.base.inner };
            }
        } else {
            const handle = self.getDirHandle(dir);
            if (handle.backend_idx) |backend_idx| {
                return .{ backend_idx, .{ .handle = @intCast(handle.handle) }, self.getBackend(backend_idx) };
            } else {
                return .{ null, .{ .handle = @intCast(handle.handle) }, self.base.inner };
            }
        }
    }

    fn stripScheme(path: []const u8) []const u8 {
        const uri = std.Uri.parse(path) catch return path;
        return path[uri.scheme.len + 3 ..];
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const backend_idx, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for opening dir '{s}' : {any}", .{ sub_path, err });
            return std.Io.Dir.OpenError.Unexpected;
        };
        const fs_dir = try backend.vtable.dirOpenDir(backend.userdata, dir_, stripScheme(sub_path), options);
        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = .{
            .handle = @intCast(fs_dir.handle),
            .backend_idx = backend_idx,
        };
        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupDir(dir, null) catch |err| {
            log.err("Failed to lookup backend for dir stat : {any}", .{err});
            return std.Io.Dir.StatError.Unexpected;
        };
        return backend.vtable.dirStat(backend.userdata, dir_);
    }

    pub fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for dir stat file '{s}' : {any}", .{ sub_path, err });
            return std.Io.Dir.StatFileError.Unexpected;
        };
        return backend.vtable.dirStatFile(backend.userdata, dir_, stripScheme(sub_path), options);
    }

    pub fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for dir access '{s}' : {any}", .{ sub_path, err });
            return std.Io.Dir.AccessError.Unexpected;
        };
        return backend.vtable.dirAccess(backend.userdata, dir_, stripScheme(sub_path), options);
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const backend_idx, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for opening file '{s}' : {any}", .{ sub_path, err });
            return std.Io.File.OpenError.Unexpected;
        };
        const file = try backend.vtable.dirOpenFile(backend.userdata, dir_, stripScheme(sub_path), flags);
        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = .{
            .handle = @intCast(file.handle),
            .backend_idx = backend_idx,
            .flags = file.flags,
        };
        return .{ .handle = @intCast(idx), .flags = file.flags };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| {
            _, const dir_, const backend = self.lookupDir(dir, null) catch |err| {
                log.err("Failed to lookup backend for closing dir : {any}", .{err});
                continue;
            };
            backend.vtable.dirClose(backend.userdata, &.{dir_});
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupDir(reader.dir, null) catch |err| {
            log.err("Failed to lookup backend for dir real path : {any}", .{err});
            return std.Io.Dir.RealPathError.Unexpected;
        };

        const original_dir = reader.dir;
        reader.dir = dir_;
        defer reader.dir = original_dir;

        return backend.vtable.dirRead(backend.userdata, reader, entries);
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const backend_idx, const dir_, const backend = self.lookupDir(dir, null) catch |err| {
            log.err("Failed to lookup backend for dir real path : {any}", .{err});
            return std.Io.Dir.RealPathError.Unexpected;
        };

        if (self.getScheme(backend_idx)) |s| {
            const prefix = try std.fmt.bufPrint(out_buffer, "{s}://", .{s});
            const path_len = try backend.vtable.dirRealPath(backend.userdata, dir_, out_buffer[prefix.len..]);
            return prefix.len + path_len;
        } else {
            return try backend.vtable.dirRealPath(backend.userdata, dir_, out_buffer);
        }
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const backend_idx, const dir_, const backend = self.lookupDir(dir, path_name) catch |err| {
            log.err("Failed to lookup backend for dir real path file '{s}' : {any}", .{ path_name, err });
            return std.Io.Dir.RealPathFileError.Unexpected;
        };

        if (self.getScheme(backend_idx)) |s| {
            const prefix = try std.fmt.bufPrint(out_buffer, "{s}://", .{s});
            const path_len = try backend.vtable.dirRealPathFile(backend.userdata, dir_, path_name, out_buffer[prefix.len..]);
            return prefix.len + path_len;
        } else {
            return try backend.vtable.dirRealPathFile(backend.userdata, dir_, path_name, out_buffer);
        }
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle, const backend = self.getFileHandle(file);
        return backend.vtable.fileStat(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags });
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle, const backend = self.getFileHandle(file);
        return backend.vtable.fileLength(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags });
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            const handle, const backend = self.getFileHandle(file);
            backend.vtable.fileClose(backend.userdata, &.{.{ .handle = @intCast(handle.handle), .flags = handle.flags }});
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle, const backend = self.getFileHandle(file);
        return backend.vtable.fileReadPositional(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, data, offset);
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle, const backend = self.getFileHandle(file);
        return backend.vtable.fileSeekBy(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, relative_offset);
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle, const backend = self.getFileHandle(file);
        return backend.vtable.fileSeekTo(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, absolute_offset);
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle, const backend = self.getFileHandle(file);

        if (self.getScheme(handle.backend_idx)) |s| {
            const prefix = try std.fmt.bufPrint(out_buffer, "{s}://", .{s});
            const path_len = try backend.vtable.fileRealPath(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, out_buffer[prefix.len..]);
            return prefix.len + path_len;
        } else {
            return try backend.vtable.fileRealPath(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, out_buffer);
        }
    }
};
