const std = @import("std");

const stdx = @import("stdx");

pub const File = @import("file.zig").File;
pub const GCS = @import("gcs.zig").GCS;
pub const HF = @import("hf.zig").HF;
pub const HTTP = @import("http.zig").HTTP;
pub const S3 = @import("s3.zig").S3;
pub const VFSBase = @import("base.zig").VFSBase;

test {
    _ = @import("gcs.zig");
}

const log = std.log.scoped(.@"zml/vfs");

/// Indices 0-2 are the standard streams, so `std.Io.File.stdout()` and
/// friends resolve to their own descriptor; the cwd placeholder follows.
const CWD_HANDLE: u32 = 3;

const VFS = @This();
const Handle = struct {
    handle: u32,
    backend_idx: ?usize,
    flags: std.Io.File.Flags = .{ .nonblocking = false },

    fn innerFile(self: *const Handle) std.Io.File {
        return .{ .handle = @intCast(self.handle), .flags = self.flags };
    }
};

allocator: std.mem.Allocator,
mutex: std.Io.Mutex = .init,

backends: std.StringArrayHashMapUnmanaged(std.Io) = .empty,
handles: stdx.SegmentedList(Handle, 128) = .{},
closed_handles: std.ArrayList(u32) = .empty,

base: VFSBase,

pub fn init(allocator: std.mem.Allocator, base_io: std.Io) !VFS {
    const base = VFSBase.init(base_io);

    var handles: @FieldType(@This(), "handles") = .{};
    try handles.append(allocator, .{ .handle = std.posix.STDIN_FILENO, .backend_idx = null });
    try handles.append(allocator, .{ .handle = std.posix.STDOUT_FILENO, .backend_idx = null });
    try handles.append(allocator, .{ .handle = std.posix.STDERR_FILENO, .backend_idx = null });
    try handles.append(allocator, .{ .handle = CWD_HANDLE, .backend_idx = null });
    std.debug.assert(handles.len == CWD_HANDLE + 1);

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
            .operate = operate,
            .dirOpenDir = dirOpenDir,
            .dirStat = dirStat,
            .dirStatFile = dirStatFile,
            .dirAccess = dirAccess,
            .dirCreateFile = dirCreateFile,
            .dirOpenFile = dirOpenFile,
            .dirClose = dirClose,
            .dirRead = dirRead,
            .dirRealPath = dirRealPath,
            .dirRealPathFile = dirRealPathFile,
            .fileStat = fileStat,
            .fileLength = fileLength,
            .fileClose = fileClose,
            .fileWritePositional = fileWritePositional,
            .fileWriteFileStreaming = fileWriteFileStreaming,
            .fileWriteFilePositional = fileWriteFilePositional,
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
        return .{ idx, self.handles.at(idx) };
    }
    return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
}

fn closeHandle(self: *VFS, idx: u32) !void {
    if (idx == CWD_HANDLE) return;

    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);

    try self.closed_handles.append(self.allocator, idx);
}

fn getFileHandle(self: *VFS, file: std.Io.File) struct { *Handle, std.Io } {
    self.mutex.lockUncancelable(self.base.inner);
    const handle = self.handles.at(@intCast(file.handle));
    self.mutex.unlock(self.base.inner);

    return .{ handle, self.getBackend(handle.backend_idx) };
}

fn getDirHandle(self: *VFS, dir: std.Io.Dir) *Handle {
    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);

    if (std.meta.eql(dir, std.Io.Dir.cwd())) return self.handles.at(CWD_HANDLE);
    return self.handles.at(@intCast(dir.handle));
}

fn getScheme(self: *VFS, backend_idx: ?usize) ?[]const u8 {
    if (backend_idx) |idx| return self.backends.entries.items(.key)[idx] else return null;
}

fn getBackend(self: *VFS, backend_idx: ?usize) std.Io {
    if (backend_idx) |idx| return self.backends.entries.items(.value)[idx] else return self.base.inner;
}

fn lookupDir(self: *VFS, dir: std.Io.Dir, sub_path: ?[]const u8) !struct { ?usize, std.Io.Dir, std.Io } {
    // A scheme-qualified path (e.g. "hf://owner/model/file") is absolute: it
    // is resolved from the scheme's root regardless of `dir`. Without this,
    // opening such a path relative to an already-open dir of the same
    // backend double-prefixes the dir's path.
    if (sub_path) |sp| {
        if (std.mem.indexOf(u8, sp, "://") != null) {
            const uri = std.Uri.parse(sp) catch return error.VFSNotRegistered;
            const backend_idx: usize = for (self.backends.entries.items(.key), 0..) |s, idx| {
                if (std.mem.eql(u8, uri.scheme, s)) break idx;
            } else return error.VFSNotRegistered;
            return .{ backend_idx, std.Io.Dir.cwd(), self.getBackend(backend_idx) };
        }
    }

    if (std.meta.eql(dir, std.Io.Dir.cwd())) {
        // Without "://", a path from cwd is local, a colon in it included.
        return .{ null, dir, self.base.inner };
    } else {
        const handle = self.getDirHandle(dir);
        if (handle.backend_idx) |backend_idx| {
            return .{ backend_idx, .{ .handle = @intCast(handle.handle) }, self.getBackend(backend_idx) };
        } else {
            return .{ null, .{ .handle = @intCast(handle.handle) }, self.base.inner };
        }
    }
}

/// `path` without its "scheme://" prefix; a name with a colon and no "://"
/// is a plain local name.
fn stripScheme(path: []const u8) []const u8 {
    const i = std.mem.indexOf(u8, path, "://") orelse return path;
    return path[i + 3 ..];
}

fn operate(userdata: ?*anyopaque, operation: std.Io.Operation) std.Io.Cancelable!std.Io.Operation.Result {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    switch (operation) {
        .file_read_streaming => |o| {
            const handle, const backend = self.getFileHandle(o.file);
            return backend.vtable.operate(backend.userdata, .{ .file_read_streaming = .{
                .file = .{ .handle = @intCast(handle.handle), .flags = handle.flags },
                .data = o.data,
            } });
        },
        .file_write_streaming => |o| {
            const handle, const backend = self.getFileHandle(o.file);
            return backend.vtable.operate(backend.userdata, .{ .file_write_streaming = .{
                .file = .{ .handle = @intCast(handle.handle), .flags = handle.flags },
                .header = o.header,
                .data = o.data,
                .splat = o.splat,
            } });
        },
        .device_io_control, .net_receive, .net_read, .net_send, .net_write => {
            return self.base.inner.vtable.operate(self.base.inner.userdata, operation);
        },
    }
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

pub fn dirCreateFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.Dir.CreateFileOptions) std.Io.File.OpenError!std.Io.File {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const backend_idx, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
        log.err("Failed to lookup backend for dir create file '{s}' : {any}", .{ sub_path, err });
        return std.Io.File.OpenError.Unexpected;
    };

    const file = try backend.vtable.dirCreateFile(backend.userdata, dir_, stripScheme(sub_path), flags);
    errdefer file.close(backend);
    const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
    handle.* = .{
        .handle = @intCast(file.handle),
        .backend_idx = backend_idx,
        .flags = file.flags,
    };
    return .{ .handle = @intCast(idx), .flags = file.flags };
}

fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.Dir.OpenFileOptions) std.Io.File.OpenError!std.Io.File {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const backend_idx, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
        log.err("Failed to lookup backend for opening file '{s}' : {any}", .{ sub_path, err });
        return std.Io.File.OpenError.Unexpected;
    };
    const file = try backend.vtable.dirOpenFile(backend.userdata, dir_, stripScheme(sub_path), flags);
    errdefer file.close(backend);
    const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
    handle.* = .{
        .handle = @intCast(file.handle),
        .backend_idx = backend_idx,
        .flags = file.flags,
    };
    return .{ .handle = @intCast(idx), .flags = handle.flags };
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
        return std.Io.Dir.Reader.Error.Unexpected;
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
        const path_len = try backend.vtable.dirRealPathFile(backend.userdata, dir_, stripScheme(path_name), out_buffer[prefix.len..]);
        return prefix.len + path_len;
    } else {
        return try backend.vtable.dirRealPathFile(backend.userdata, dir_, stripScheme(path_name), out_buffer);
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

fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) std.Io.File.WritePositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    return backend.vtable.fileWritePositional(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, header, data, splat, offset);
}

fn fileWriteFileStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.File.Writer.WriteFileError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const own_io = self.io();
    if (reader.io.userdata != own_io.userdata or reader.io.vtable != own_io.vtable) return error.Unimplemented;

    const dst_handle, const dst_backend = self.getFileHandle(file);
    const src_handle, const src_backend = self.getFileHandle(reader.file);
    if (src_backend.userdata != dst_backend.userdata or src_backend.vtable != dst_backend.vtable) return error.Unimplemented;

    const original_src_io = reader.io;
    const original_src_handle = reader.file;
    reader.io = src_backend;
    reader.file = .{ .handle = @intCast(src_handle.handle), .flags = src_handle.flags };
    defer {
        reader.io = original_src_io;
        reader.file = original_src_handle;
    }

    return dst_backend.vtable.fileWriteFileStreaming(dst_backend.userdata, .{ .handle = @intCast(dst_handle.handle), .flags = dst_handle.flags }, header, reader, limit);
}

fn fileWriteFilePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit, offset: u64) std.Io.File.WriteFilePositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const own_io = self.io();
    if (reader.io.userdata != own_io.userdata or reader.io.vtable != own_io.vtable) return error.Unimplemented;

    const dst_handle, const dst_backend = self.getFileHandle(file);
    const src_handle, const src_backend = self.getFileHandle(reader.file);
    if (src_backend.userdata != dst_backend.userdata or src_backend.vtable != dst_backend.vtable) return error.Unimplemented;

    const original_src_io = reader.io;
    const original_src_handle = reader.file;
    reader.io = src_backend;
    reader.file = .{ .handle = @intCast(src_handle.handle), .flags = src_handle.flags };
    defer {
        reader.io = original_src_io;
        reader.file = original_src_handle;
    }

    return dst_backend.vtable.fileWriteFilePositional(dst_backend.userdata, .{ .handle = @intCast(dst_handle.handle), .flags = dst_handle.flags }, header, reader, limit, offset);
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

test "standard streams resolve to their own descriptors" {
    var vfs: VFS = try .init(std.testing.allocator, std.testing.io);
    defer vfs.deinit();

    const files = [_]std.Io.File{ .stdin(), .stdout(), .stderr() };
    const fds = [_]std.posix.fd_t{ std.posix.STDIN_FILENO, std.posix.STDOUT_FILENO, std.posix.STDERR_FILENO };
    for (files, fds) |file, fd| {
        const handle, const backend = vfs.getFileHandle(file);
        try std.testing.expectEqual(fd, handle.innerFile().handle);
        try std.testing.expect(backend.vtable == vfs.base.inner.vtable);
    }
}

test "copy a local file through the VFS" {
    const allocator = std.testing.allocator;
    const process_io = std.testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source_data: [8192]u8 = @splat('x');
    try tmp.dir.writeFile(process_io, .{
        .sub_path = "source",
        .data = &source_data,
    });

    var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(process_io, &path_buffer);
    const source_uri = try std.fmt.allocPrint(allocator, "file://{s}/source", .{path_buffer[0..path_len]});
    defer allocator.free(source_uri);
    const destination_uri = try std.fmt.allocPrint(allocator, "file://{s}/destination", .{path_buffer[0..path_len]});
    defer allocator.free(destination_uri);

    var file_backend: File = .init(allocator, process_io, .{});
    defer file_backend.deinit();
    var vfs: VFS = try .init(allocator, process_io);
    defer vfs.deinit();
    try vfs.register("file", file_backend.io());

    const vfs_io = vfs.io();
    {
        var source = try std.Io.Dir.openFile(.cwd(), vfs_io, source_uri, .{});
        defer source.close(vfs_io);
        var destination = try std.Io.Dir.createFile(.cwd(), vfs_io, destination_uri, .{});
        defer destination.close(vfs_io);

        var reader: std.Io.File.Reader = .initStreaming(source, vfs_io, &.{});
        var copy_buffer: [4096]u8 align(4096) = undefined;
        var writer = destination.writer(vfs_io, &copy_buffer);
        try std.testing.expectEqual(source_data.len, try reader.interface.streamRemaining(&writer.interface));
        try writer.interface.flush();
    }

    const copied = try tmp.dir.readFileAlloc(process_io, "destination", allocator, .limited(source_data.len + 1));
    defer allocator.free(copied);
    try std.testing.expectEqualSlices(u8, &source_data, copied);
}

test "a name with a colon and no \"://\" is local" {
    const testing_io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing_io, .{ .sub_path = "notes:old.txt", .data = "bytes" });
    var path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const root = path_buffer[0..try tmp.dir.realPath(testing_io, &path_buffer)];

    var vfs: VFS = try .init(std.testing.allocator, testing_io);
    defer vfs.deinit();
    const vfs_io = vfs.io();

    const backend_idx, _, _ = try vfs.lookupDir(.cwd(), "notes:old.txt");
    try std.testing.expectEqual(null, backend_idx);

    const dir = try std.Io.Dir.openDir(.cwd(), vfs_io, root, .{});
    defer dir.close(vfs_io);
    const file = try dir.openFile(vfs_io, "notes:old.txt", .{});
    defer file.close(vfs_io);
    var contents: [5]u8 = undefined;
    try std.testing.expectEqual(contents.len, try file.readPositionalAll(vfs_io, &contents, 0));
    try std.testing.expectEqualStrings("bytes", &contents);

    var real_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const real_path = real_path_buffer[0..try dir.realPathFile(vfs_io, "notes:old.txt", &real_path_buffer)];
    try std.testing.expect(std.mem.endsWith(u8, real_path, "/notes:old.txt"));
}
