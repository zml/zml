const std = @import("std");

const stdx = @import("stdx");

pub const File = @import("file.zig").File;
pub const GCS = @import("gcs.zig").GCS;
pub const HF = @import("hf.zig").HF;
pub const HTTP = @import("http.zig").HTTP;
pub const S3 = @import("s3.zig").S3;
const base_module = @import("base.zig");
pub const Backend = base_module.Backend;
pub const ReadHints = base_module.ReadHints;
pub const ReadStats = base_module.ReadStats;
pub const ReadStatsProvider = base_module.ReadStatsProvider;
pub const VFSBase = base_module.VFSBase;

test {
    _ = @import("http_acceptance_test.zig");
}

const log = std.log.scoped(.@"zml/vfs");

const CWD_HANDLE: u32 = 0;

const VFS = @This();
const Handle = struct { handle: u32, backend_idx: ?usize, flags: std.Io.File.Flags = .{ .nonblocking = false } };

pub const LoadProfile = struct {
    /// Generic fallback used by callers that do not prepare a profile from a
    /// VFS path. This value is borrowed and does not require deinitialization.
    pub const default: LoadProfile = .{
        .name = "default",
        .read_chunk_size = 16 * 1024 * 1024,
        .high_latency = false,
        .stats = null,
    };

    /// Local-file profile returned for paths without a registered URI scheme.
    pub const local: LoadProfile = .{
        .name = "local",
        .read_chunk_size = 8 * 1024 * 1024,
        .high_latency = false,
        .stats = null,
    };

    name: []const u8,
    /// Minimum source request size. The loader may increase it to match the
    /// independently calibrated DMA block size.
    read_chunk_size: usize,
    high_latency: bool,
    stats: ?ReadStatsProvider,
};

allocator: std.mem.Allocator,
mutex: std.Io.Mutex = .init,

backends: std.StringArrayHashMapUnmanaged(Backend) = .empty,
handles: stdx.SegmentedList(Handle, 128) = .{},
closed_handles: std.ArrayList(u32) = .empty,

base: VFSBase,

pub fn init(allocator: std.mem.Allocator, base_io: std.Io) !VFS {
    const base = VFSBase.init(base_io);

    var handles: @FieldType(@This(), "handles") = .{};
    try handles.append(allocator, .{ .handle = CWD_HANDLE, .backend_idx = null });
    try handles.append(allocator, .{ .handle = std.posix.STDIN_FILENO, .backend_idx = null });
    try handles.append(allocator, .{ .handle = std.posix.STDOUT_FILENO, .backend_idx = null });
    try handles.append(allocator, .{ .handle = std.posix.STDERR_FILENO, .backend_idx = null });

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
    return self.registerBackend(scheme, .{ .io = backend });
}

pub fn registerBackend(self: *VFS, scheme: []const u8, backend: Backend) std.mem.Allocator.Error!void {
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
        .vtable = ioVTable(),
    };
}

fn ioVTable() *const std.Io.VTable {
    return &comptime VFSBase.vtable(.{
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
    });
}

/// Prepares the source tuning and feedback provider for one model load.
/// Returned strings and providers borrow backend state, so this VFS and its
/// registered backend must outlive the load.
pub fn loadProfile(self: *VFS, path: []const u8) !LoadProfile {
    if (std.mem.indexOf(u8, path, "://") == null) return .local;
    const uri = std.Uri.parse(path) catch return error.VFSNotRegistered;
    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);
    for (self.backends.entries.items(.key), 0..) |scheme, index| {
        if (!std.mem.eql(u8, uri.scheme, scheme)) continue;
        const backend = self.backends.entries.items(.value)[index];
        return .{
            .name = scheme,
            .read_chunk_size = backend.read_hints.read_chunk_size,
            .high_latency = backend.read_hints.high_latency,
            .stats = backend.read_stats,
        };
    }
    return error.VFSNotRegistered;
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
    if (backend_idx) |idx| return self.backends.entries.items(.value)[idx].io else return self.base.inner;
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
        .device_io_control, .file_write_streaming, .net_receive => {
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

pub fn dirCreateFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) std.Io.File.OpenError!std.Io.File {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const backend_idx, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
        log.err("Failed to lookup backend for dir create file '{s}' : {any}", .{ sub_path, err });
        return std.Io.File.OpenError.Unexpected;
    };

    const file = try backend.vtable.dirCreateFile(backend.userdata, dir_, stripScheme(sub_path), flags);
    const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
    handle.* = .{
        .handle = @intCast(file.handle),
        .backend_idx = backend_idx,
    };
    return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
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

fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) std.Io.File.WritePositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    return backend.vtable.fileWritePositional(backend.userdata, .{ .handle = @intCast(handle.handle), .flags = handle.flags }, header, data, splat, offset);
}

fn fileWriteFileStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.File.Writer.WriteFileError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));

    const dst_handle, const dst_backend = self.getFileHandle(file);
    const src_handle, _ = self.getFileHandle(reader.file);

    const original_src_handle = reader.file;
    reader.file = .{ .handle = @intCast(src_handle.handle), .flags = src_handle.flags };
    defer reader.file = original_src_handle;

    return dst_backend.vtable.fileWriteFileStreaming(dst_backend.userdata, .{ .handle = @intCast(dst_handle.handle), .flags = dst_handle.flags }, header, reader, limit);
}

fn fileWriteFilePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit, offset: u64) std.Io.File.WriteFilePositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));

    const dst_handle, const dst_backend = self.getFileHandle(file);
    const src_handle, _ = self.getFileHandle(reader.file);

    const original_src_handle = reader.file;
    reader.file = .{ .handle = @intCast(src_handle.handle), .flags = src_handle.flags };
    defer reader.file = original_src_handle;

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

test "VFS prepares load profiles for local and registered paths" {
    var filesystem = try VFS.init(std.testing.allocator, std.testing.io);
    defer filesystem.deinit();
    try filesystem.registerBackend("test", .{
        .io = std.testing.io,
        .read_hints = .{
            .read_chunk_size = 32 * 1024 * 1024,
            .high_latency = true,
        },
    });

    const profile = try filesystem.loadProfile("test://bucket/object");
    try std.testing.expectEqualStrings("test", profile.name);
    try std.testing.expectEqual(@as(usize, 32 * 1024 * 1024), profile.read_chunk_size);
    try std.testing.expect(profile.high_latency);

    const absolute = try filesystem.loadProfile("/tmp/model.safetensors");
    try std.testing.expectEqualDeep(LoadProfile.local, absolute);
    const relative = try filesystem.loadProfile("models/model.safetensors");
    try std.testing.expectEqualDeep(LoadProfile.local, relative);
    try std.testing.expectError(
        error.VFSNotRegistered,
        filesystem.loadProfile("missing://bucket/object"),
    );
}

test "VFS reports the configured load profile for every bundled backend" {
    var client: std.http.Client = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    defer client.deinit();

    var file: File = .init(std.testing.allocator, std.testing.io, .{});
    defer file.deinit();
    var http = try HTTP.init(std.testing.allocator, std.testing.io, &client, .https);
    defer http.deinit();
    var s3 = try S3.init(std.testing.allocator, std.testing.io, &client, .{
        .endpoint_url = "https://s3.amazonaws.com",
        .region = "us-east-1",
    }, .{});
    defer s3.deinit();
    var gcs = try GCS.init(std.testing.allocator, std.testing.io, &client, .{});
    defer gcs.deinit();
    var hf = try HF.init(std.testing.allocator, std.testing.io, &client, null, .{});
    defer hf.deinit();

    var filesystem = try VFS.init(std.testing.allocator, std.testing.io);
    defer filesystem.deinit();
    try filesystem.registerBackend("file", file.backend());
    try filesystem.registerBackend("https", http.backend());
    try filesystem.registerBackend("s3", s3.backend());
    try filesystem.registerBackend("gs", gcs.backend());
    try filesystem.registerBackend("hf", hf.backend());

    const Case = struct {
        path: []const u8,
        name: []const u8,
        read_chunk_size: usize,
        high_latency: bool,
    };
    const cases = [_]Case{
        .{ .path = "file:///tmp/model", .name = "file", .read_chunk_size = 8 * 1024 * 1024, .high_latency = false },
        .{ .path = "https://example.com/model", .name = "https", .read_chunk_size = 16 * 1024 * 1024, .high_latency = true },
        .{ .path = "s3://bucket/model", .name = "s3", .read_chunk_size = 16 * 1024 * 1024, .high_latency = true },
        .{ .path = "gs://bucket/model", .name = "gs", .read_chunk_size = 16 * 1024 * 1024, .high_latency = true },
        .{ .path = "hf://owner/model", .name = "hf", .read_chunk_size = 32 * 1024 * 1024, .high_latency = true },
    };
    for (cases) |case| {
        const profile = try filesystem.loadProfile(case.path);
        try std.testing.expectEqualStrings(case.name, profile.name);
        try std.testing.expectEqual(case.read_chunk_size, profile.read_chunk_size);
        try std.testing.expectEqual(case.high_latency, profile.high_latency);
    }
}
