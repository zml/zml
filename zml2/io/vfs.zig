const std = @import("std");
const vfs = @import("vfs");

const log = std.log.scoped(.@"zml/io/vfs");

pub const VFS = struct {
    pub const File = vfs.File;
    pub const HTTP = vfs.HTTP;
    pub const HF = vfs.HF;
    const VFSBase = vfs.VFSBase;

    const Handle = struct {
        pub const Type = enum {
            file,
            directory,
        };

        type: Type,
        handle: u32,
        scheme: []const u8,
        backend: std.Io,
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,

    backends: std.StringHashMapUnmanaged(std.Io) = .{},
    handles: std.ArrayList(Handle) = .{},
    closed_handles: std.ArrayList(u32) = .{},

    base: VFSBase,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io) !VFS {
        const base = VFSBase.init(base_io);

        var handles: std.ArrayList(Handle) = .{};
        // cwd handle
        try handles.append(allocator, .{
            .type = .directory,
            .handle = 0,
            .scheme = "",
            .backend = base.inner,
        });

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

    fn openHandle(self: *VFS) !struct { u32, *Handle } {
        if (self.closed_handles.pop()) |idx| {
            return .{ idx, &self.handles.items[idx] };
        }
        return .{ @intCast(self.handles.items.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *VFS, idx: u32) !void {
        if (idx == 0) return; // cwd
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *VFS, file: std.Io.File) *Handle {
        return &self.handles.items[@intCast(file.handle)];
    }

    fn getDirHandle(self: *VFS, dir: std.Io.Dir) *Handle {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return &self.handles.items[0];
        }

        return &self.handles.items[@intCast(dir.handle)];
    }

    fn lookupFromDir(self: *VFS, dir: std.Io.Dir, sub_path: ?[]const u8) !struct { []const u8, std.Io.Dir, std.Io } {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            if (sub_path == null) {
                return .{ "", dir, self.base.inner };
            }

            if (std.fs.path.isAbsolutePosix(sub_path.?)) {
                return .{ "", dir, self.base.inner };
            }

            const uri = try std.Uri.parse(sub_path.?);
            return .{ uri.scheme, std.Io.Dir.cwd(), self.backends.get(uri.scheme).? };
        } else {
            const handle = self.getDirHandle(dir);
            return .{ handle.scheme, .{ .handle = @intCast(handle.handle) }, handle.backend };
        }
    }

    fn stripScheme(path: []const u8) []const u8 {
        const uri = std.Uri.parse(path) catch return path;
        return path[uri.scheme.len + 3 ..];
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));

        const scheme, const dir_, const backend = self.lookupFromDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for opening dir '{s}' : {any}", .{ sub_path, err });
            return std.Io.Dir.OpenError.Unexpected;
        };
        const fs_dir = try backend.vtable.dirOpenDir(backend.userdata, dir_, stripScheme(sub_path), options);

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = .{
            .type = .directory,
            .handle = @intCast(fs_dir.handle),
            .scheme = scheme,
            .backend = backend,
        };

        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupFromDir(dir, null) catch |err| {
            log.err("Failed to lookup backend for dir stat : {any}", .{err});
            return std.Io.Dir.StatError.Unexpected;
        };
        return backend.vtable.dirStat(backend.userdata, dir_);
    }

    pub fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupFromDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for dir stat file '{s}' : {any}", .{ sub_path, err });
            return std.Io.Dir.StatFileError.Unexpected;
        };
        return backend.vtable.dirStatFile(backend.userdata, dir_, sub_path, options);
    }

    pub fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        _, const dir_, const backend = self.lookupFromDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for dir access '{s}' : {any}", .{ sub_path, err });
            return std.Io.Dir.AccessError.Unexpected;
        };
        return backend.vtable.dirAccess(backend.userdata, dir_, sub_path, options);
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const scheme, const dir_, const backend = self.lookupFromDir(dir, sub_path) catch |err| {
            log.err("Failed to lookup backend for opening file '{s}' : {any}", .{ sub_path, err });
            return std.Io.File.OpenError.Unexpected;
        };
        const file = try backend.vtable.dirOpenFile(backend.userdata, dir_, stripScheme(sub_path), flags);
        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = .{
            .type = .file,
            .handle = @intCast(file.handle),
            .scheme = scheme,
            .backend = backend,
        };

        return .{ .handle = @intCast(idx) };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| {
            _, const dir_, const backend = self.lookupFromDir(dir, null) catch |err| {
                log.err("Failed to lookup backend for closing dir : {any}", .{err});
                continue;
            };
            backend.vtable.dirClose(backend.userdata, &.{dir_});
            self.closeHandle(@intCast(dir_.handle)) catch unreachable;
        }
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const scheme, const dir_, const backend = self.lookupFromDir(dir, null) catch |err| {
            log.err("Failed to lookup backend for dir real path : {any}", .{err});
            return std.Io.Dir.RealPathError.Unexpected;
        };
        const prefix_len = scheme.len + 3;
        @memcpy(out_buffer[0..scheme.len], scheme);
        @memcpy(out_buffer[scheme.len..prefix_len], "://");
        const path_len = try backend.vtable.dirRealPath(backend.userdata, dir_, out_buffer[prefix_len..]);
        return prefix_len + path_len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const scheme, const dir_, const backend = self.lookupFromDir(dir, null) catch |err| {
            log.err("Failed to lookup backend for dir real path file '{s}' : {any}", .{ path_name, err });
            return std.Io.Dir.RealPathFileError.Unexpected;
        };
        const prefix_len = scheme.len + 3;
        @memcpy(out_buffer[0..scheme.len], scheme);
        @memcpy(out_buffer[scheme.len..prefix_len], "://");
        const path_len = try backend.vtable.dirRealPathFile(backend.userdata, dir_, path_name, out_buffer[prefix_len..]);
        return prefix_len + path_len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return handle.backend.vtable.fileStat(handle.backend.userdata, .{ .handle = @intCast(handle.handle) });
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return handle.backend.vtable.fileLength(handle.backend.userdata, .{ .handle = @intCast(handle.handle) });
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            const handle = self.getFileHandle(file);
            handle.backend.vtable.fileClose(handle.backend.userdata, &.{.{ .handle = @intCast(handle.handle) }});
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return handle.backend.vtable.fileReadStreaming(handle.backend.userdata, .{ .handle = @intCast(handle.handle) }, data);
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return handle.backend.vtable.fileReadPositional(handle.backend.userdata, .{ .handle = @intCast(handle.handle) }, data, offset);
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return handle.backend.vtable.fileSeekBy(handle.backend.userdata, .{ .handle = @intCast(handle.handle) }, relative_offset);
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return handle.backend.vtable.fileSeekTo(handle.backend.userdata, .{ .handle = @intCast(handle.handle) }, absolute_offset);
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const prefix_len = handle.scheme.len + 3;
        @memcpy(out_buffer[0..handle.scheme.len], handle.scheme);
        @memcpy(out_buffer[handle.scheme.len..prefix_len], "://");
        const path_len = try handle.backend.vtable.fileRealPath(handle.backend.userdata, .{ .handle = @intCast(handle.handle) }, out_buffer[prefix_len..]);
        return prefix_len + path_len;
    }
};
