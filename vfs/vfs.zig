//! A `std.Io` over local files and remote objects. A bare path or a
//! `file://` URI is served by the inner io; a URI such as
//! `hf://org/model/model.safetensors` or `s3://bucket/key` by the backend
//! registered for its scheme (`registerBackend`, or the four of `Remotes`).
//! Readers take `io()` and need not know which they read; one that must (to
//! size its reads) asks `fromIo`.
//!
//! The VFS, and every backend registered on it, must outlive each file
//! opened through it.
//!
//! A remote path is read-only: creating or deleting a file or a directory
//! there, or opening a file for writing, is logged at err and fails with
//! `ReadOnlyFileSystem`. A local file fails as the inner io does (a denied
//! one is `AccessDenied` there).
const std = @import("std");

const stdx = @import("stdx");

pub const GCS = @import("gcs.zig").GCS;
pub const HF = @import("hf.zig").HF;
pub const HTTP = @import("http.zig").HTTP;
pub const S3 = @import("s3.zig").S3;
const base_module = @import("base.zig");
pub const Backend = base_module.Backend;
pub const ReadHints = base_module.ReadHints;
const VFSBase = base_module.VFSBase;

test {
    _ = @import("gcs.zig");
}

const log = std.log.scoped(.@"zml/vfs");
/// For the refusals of writes to a remote path, which tests trigger on
/// purpose; every other err here still fails a test (`scopedLog`).
const refusal_log = base_module.scopedLog(.@"zml/vfs");

/// Indices 0-2 are the standard streams, so `std.Io.File.stdout()` and
/// friends resolve to their own descriptor; the cwd placeholder follows.
const CWD_HANDLE: u32 = 3;
/// Local files are the VFS's own: a bare path or a `file://` URI resolves
/// to the inner `Io`, never to a registered backend.
const local_scheme = "file";

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

backends: std.StringArrayHashMapUnmanaged(Backend) = .empty,
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

/// Serves `scheme` with `backend`, replacing any backend registered under
/// it. `scheme` is kept, not copied: it must outlive the VFS.
pub fn registerBackend(self: *VFS, scheme: []const u8, backend: Backend) std.mem.Allocator.Error!void {
    std.debug.assert(!std.mem.eql(u8, scheme, local_scheme)); // file is the VFS's own (local_scheme)
    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);

    try self.backends.put(self.allocator, scheme, backend);
}

/// The remote schemes on one HTTP client: `https`, `hf`, `s3` and `gs`, each
/// configured from the environment through its backend's `auto` (plain
/// `HTTP.init` for `https`), the client honoring the proxy variables.
/// Initialized in place, since the backends point at the client; it must not
/// move afterwards. Close every file opened through these schemes before
/// `deinit`; the VFS must not reach them after it.
pub const Remotes = struct {
    /// The proxies' allocations, which must outlive the client.
    arena: std.heap.ArenaAllocator,
    client: std.http.Client,
    https: HTTP,
    hf: HF,
    s3: S3,
    gs: GCS,

    /// Builds the client on the VFS's base io and registers the four
    /// schemes, replacing any backend registered under them. `environ` is
    /// read here only: tokens, credentials and proxies are copied. On error
    /// nothing is registered.
    pub fn init(self: *Remotes, allocator: std.mem.Allocator, vfs: *VFS, environ: *std.process.Environ.Map) !void {
        const base_io = vfs.base.inner;
        self.arena = .init(allocator);
        errdefer self.arena.deinit();
        self.client = .{ .allocator = allocator, .io = base_io };
        errdefer self.client.deinit();
        try self.client.initDefaultProxies(self.arena.allocator(), environ);
        self.https = try .init(allocator, base_io, &self.client, .https);
        errdefer self.https.deinit();
        self.hf = try .auto(allocator, base_io, &self.client, environ);
        errdefer self.hf.deinit();
        self.s3 = try .auto(allocator, base_io, &self.client, environ);
        errdefer self.s3.deinit();
        self.gs = try .auto(allocator, base_io, &self.client, environ);
        errdefer self.gs.deinit();

        vfs.mutex.lockUncancelable(base_io);
        defer vfs.mutex.unlock(base_io);
        // Room first: once a scheme is registered, nothing can fail.
        try vfs.backends.ensureUnusedCapacity(vfs.allocator, 4);
        vfs.backends.putAssumeCapacity("https", self.https.backend());
        vfs.backends.putAssumeCapacity("hf", self.hf.backend());
        vfs.backends.putAssumeCapacity("s3", self.s3.backend());
        vfs.backends.putAssumeCapacity("gs", self.gs.backend());
    }

    pub fn deinit(self: *Remotes) void {
        self.gs.deinit();
        self.s3.deinit();
        self.hf.deinit();
        self.https.deinit();
        self.client.deinit();
        self.arena.deinit();
    }
};

/// The `std.Io` of this VFS. It points into the VFS, which must not move
/// once this was called.
pub fn io(self: *VFS) std.Io {
    return .{
        .userdata = &self.base,
        .vtable = &io_vtable,
    };
}

/// The VFS behind an `Io`, or null for any other implementation: a reader
/// that opened a file through `any_io` asks this VFS about that handle.
pub fn fromIo(any_io: std.Io) ?*VFS {
    if (any_io.vtable != &io_vtable) return null;
    return @fieldParentPtr("base", VFSBase.as(any_io.userdata));
}

const io_vtable: std.Io.VTable = VFSBase.vtable(.{
    .operate = operate,
    .dirCreateDir = dirCreateDir,
    .dirCreateDirPath = dirCreateDirPath,
    .dirCreateDirPathOpen = dirCreateDirPathOpen,
    .dirOpenDir = dirOpenDir,
    .dirStat = dirStat,
    .dirStatFile = dirStatFile,
    .dirAccess = dirAccess,
    .dirCreateFile = dirCreateFile,
    .dirCreateFileAtomic = dirCreateFileAtomic,
    .dirOpenFile = dirOpenFile,
    .dirClose = dirClose,
    .dirRead = dirRead,
    .dirRealPath = dirRealPath,
    .dirRealPathFile = dirRealPathFile,
    .dirDeleteFile = dirDeleteFile,
    .dirDeleteDir = dirDeleteDir,
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

/// The read hints of the backend serving `path`, for a reader sizing its
/// requests. A path resolves as it does for `openFile`: a bare path or a
/// `file://` URI is local (`ReadHints.local`), and a scheme no backend
/// serves is `VFSNotRegistered`.
pub fn readHints(self: *VFS, path: []const u8) !ReadHints {
    const backend_idx, _, _ = try self.lookupDir(.cwd(), path);
    const index = backend_idx orelse return .local;
    return self.backends.entries.items(.value)[index].read_hints;
}

/// Whether `path` resolves to the inner `Io`'s files; an unregistered scheme
/// does not. Opens nothing.
pub fn isLocal(self: *VFS, path: []const u8) bool {
    const backend_idx, _, _ = self.lookupDir(.cwd(), path) catch return false;
    return backend_idx == null;
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
            if (std.mem.eql(u8, uri.scheme, local_scheme)) return .{ null, std.Io.Dir.cwd(), self.base.inner };
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

/// The inner `Io`'s directory for creating or deleting `sub_path` from
/// `dir`. A remote path is refused: no remote backend writes, and the base
/// layer under each one would write a local file in its place.
fn writableDir(self: *VFS, dir: std.Io.Dir, sub_path: []const u8, operation: []const u8) error{ ReadOnlyFileSystem, Unexpected }!std.Io.Dir {
    const backend_idx, const dir_, _ = self.lookupDir(dir, sub_path) catch |err| {
        log.err("Failed to lookup backend to {s} '{s}' : {any}", .{ operation, sub_path, err });
        return error.Unexpected;
    };
    if (self.getScheme(backend_idx)) |scheme| {
        refusal_log.err("Cannot {s} '{s}': {s} paths are read-only", .{ operation, sub_path, scheme });
        return error.ReadOnlyFileSystem;
    }
    return dir_;
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

fn dirCreateDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirError!void {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const dir_ = try self.writableDir(dir, sub_path, "create directory");
    return self.base.inner.vtable.dirCreateDir(self.base.inner.userdata, dir_, stripScheme(sub_path), permissions);
}

fn dirCreateDirPath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirPathError!std.Io.Dir.CreatePathStatus {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const dir_ = try self.writableDir(dir, sub_path, "create directory");
    return self.base.inner.vtable.dirCreateDirPath(self.base.inner.userdata, dir_, stripScheme(sub_path), permissions);
}

fn dirCreateDirPathOpen(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions, options: std.Io.Dir.OpenOptions) std.Io.Dir.CreateDirPathOpenError!std.Io.Dir {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const dir_ = try self.writableDir(dir, sub_path, "create directory");
    const inner = self.base.inner;
    const fs_dir = try inner.vtable.dirCreateDirPathOpen(inner.userdata, dir_, stripScheme(sub_path), permissions, options);
    errdefer fs_dir.close(inner);
    const idx, const handle = self.openHandle() catch return std.Io.Dir.CreateDirPathOpenError.Unexpected;
    handle.* = .{
        .handle = @intCast(fs_dir.handle),
        .backend_idx = null,
    };
    return .{ .handle = @intCast(idx) };
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
    const dir_ = try self.writableDir(dir, sub_path, "create file");
    const inner = self.base.inner;
    const file = try inner.vtable.dirCreateFile(inner.userdata, dir_, stripScheme(sub_path), flags);
    errdefer file.close(inner);
    const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
    handle.* = .{
        .handle = @intCast(file.handle),
        .backend_idx = null,
        .flags = file.flags,
    };
    return .{ .handle = @intCast(idx), .flags = file.flags };
}

/// The atomic file holds the inner `Io`'s handles, not this VFS's.
fn dirCreateFileAtomic(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.CreateFileAtomicOptions) std.Io.Dir.CreateFileAtomicError!std.Io.File.Atomic {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const dir_ = try self.writableDir(dir, sub_path, "create file");
    return self.base.inner.vtable.dirCreateFileAtomic(self.base.inner.userdata, dir_, stripScheme(sub_path), options);
}

fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.Dir.OpenFileOptions) std.Io.File.OpenError!std.Io.File {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const backend_idx, const dir_, const backend = self.lookupDir(dir, sub_path) catch |err| {
        log.err("Failed to lookup backend for opening file '{s}' : {any}", .{ sub_path, err });
        return std.Io.File.OpenError.Unexpected;
    };
    if (flags.mode != .read_only and backend_idx != null) {
        refusal_log.err("Cannot open '{s}' for writing: {s} paths are read-only", .{ sub_path, self.getScheme(backend_idx).? });
        return error.ReadOnlyFileSystem;
    }
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

fn dirDeleteFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteFileError!void {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const dir_ = try self.writableDir(dir, sub_path, "delete file");
    return self.base.inner.vtable.dirDeleteFile(self.base.inner.userdata, dir_, stripScheme(sub_path));
}

fn dirDeleteDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteDirError!void {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const dir_ = try self.writableDir(dir, sub_path, "delete directory");
    return self.base.inner.vtable.dirDeleteDir(self.base.inner.userdata, dir_, stripScheme(sub_path));
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

test "VFS gives the read hints of local and registered paths" {
    var filesystem = try VFS.init(std.testing.allocator, std.testing.io);
    defer filesystem.deinit();
    const hints: ReadHints = .{ .read_chunk_size = 32 * 1024 * 1024, .high_latency = true };
    try filesystem.registerBackend("test", .{ .io = std.testing.io, .read_hints = hints });

    try std.testing.expectEqual(hints, try filesystem.readHints("test://bucket/object"));
    try std.testing.expect(!filesystem.isLocal("test://bucket/object"));

    for ([_][]const u8{ "/tmp/model.safetensors", "models/model.safetensors", "file:///tmp/model.safetensors" }) |path| {
        try std.testing.expectEqual(ReadHints{ .read_chunk_size = 8 * 1024 * 1024, .high_latency = false }, try filesystem.readHints(path));
        try std.testing.expect(filesystem.isLocal(path));
    }
    try std.testing.expectError(error.VFSNotRegistered, filesystem.readHints("missing://bucket/object"));
    try std.testing.expect(!filesystem.isLocal("missing://bucket/object"));
    try std.testing.expectEqual(@as(?*VFS, &filesystem), VFS.fromIo(filesystem.io()));
    try std.testing.expectEqual(@as(?*VFS, null), VFS.fromIo(std.testing.io));
}

test "Remotes registers every remote scheme with its read hints" {
    var filesystem = try VFS.init(std.testing.allocator, std.testing.io);
    defer filesystem.deinit();
    // No credentials and no proxies: nothing is read from the user's home.
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();
    var remotes: Remotes = undefined;
    try remotes.init(std.testing.allocator, &filesystem, &environ);
    defer remotes.deinit();

    const Case = struct { path: []const u8, local: bool = false, hints: ReadHints };
    const remote: ReadHints = .{ .read_chunk_size = 16 * 1024 * 1024, .high_latency = true };
    const cases = [_]Case{
        .{ .path = "file:///tmp/model", .local = true, .hints = .local },
        .{ .path = "/var/models/model", .local = true, .hints = .local },
        .{ .path = "https://example.com/model", .hints = remote },
        .{ .path = "s3://bucket/model", .hints = remote },
        .{ .path = "gs://bucket/model", .hints = remote },
        .{ .path = "hf://owner/model", .hints = .{ .read_chunk_size = 32 * 1024 * 1024, .high_latency = true } },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.hints, try filesystem.readHints(case.path));
        try std.testing.expectEqual(case.local, filesystem.isLocal(case.path));
        if (case.local) continue;
        // A remote real path carries its scheme once, resolved without a request.
        var real_path: [64]u8 = undefined;
        try std.testing.expectEqualStrings(case.path, real_path[0..try std.Io.Dir.cwd().realPathFile(filesystem.io(), case.path, &real_path)]);
    }
}

test "VFS serves bare paths and file:// URIs from the inner Io" {
    const testing_io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const created = try tmp.dir.createFile(testing_io, "bare.bin", .{ .read = true });
    try created.writePositionalAll(testing_io, "bare path bytes", 0);
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try created.realPath(testing_io, &path_buffer)];
    created.close(testing_io);

    var filesystem = try VFS.init(std.testing.allocator, testing_io);
    defer filesystem.deinit();
    const vfs_io = filesystem.io();

    var uri_buffer: [std.fs.max_path_bytes + 8]u8 = undefined;
    const uri = try std.fmt.bufPrint(&uri_buffer, "file://{s}", .{path});
    for ([_][]const u8{ path, uri }) |open_path| {
        var real_buffer: [std.fs.max_path_bytes]u8 = undefined;
        try std.testing.expectEqualStrings(path, real_buffer[0..try std.Io.Dir.cwd().realPathFile(vfs_io, open_path, &real_buffer)]);
        const opened = try std.Io.Dir.openFile(.cwd(), vfs_io, open_path, .{ .mode = .read_only });
        defer opened.close(vfs_io);
        const handle, _ = filesystem.getFileHandle(opened);
        try std.testing.expectEqual(@as(?usize, null), handle.backend_idx);
        var contents: [15]u8 = undefined;
        try std.testing.expectEqual(contents.len, try opened.readPositionalAll(vfs_io, &contents, 0));
        try std.testing.expectEqualStrings("bare path bytes", &contents);
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

    // `file://` is the VFS's own scheme: no backend to register.
    var vfs: VFS = try .init(allocator, process_io);
    defer vfs.deinit();

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

test "creates and deletes on a local path go to the inner Io, from a path or an open directory" {
    const testing_io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buffer[0..try tmp.dir.realPath(testing_io, &path_buffer)];
    var vfs: VFS = try .init(std.testing.allocator, testing_io);
    defer vfs.deinit();
    const vfs_io = vfs.io();

    var uri_buffer: [std.fs.max_path_bytes + 16]u8 = undefined;
    {
        const opened = try std.Io.Dir.cwd().createDirPathOpen(vfs_io, try std.fmt.bufPrint(&uri_buffer, "file://{s}/a/b", .{root}), .{});
        defer opened.close(vfs_io);
        // Relative to the descriptor the VFS's handle stands for.
        try opened.createDir(vfs_io, "c", .default_dir);
        try opened.writeFile(vfs_io, .{ .sub_path = "c/file", .data = "bytes" });
        try tmp.dir.access(testing_io, "a/b/c/file", .{});
        try opened.deleteFile(vfs_io, "c/file");
        try opened.deleteDir(vfs_io, "c");
        // A colon without "://" is part of a local name.
        try opened.writeFile(vfs_io, .{ .sub_path = "notes:old.txt", .data = "bytes" });
        try tmp.dir.access(testing_io, "a/b/notes:old.txt", .{});
        try opened.deleteFile(vfs_io, "notes:old.txt");
        try std.testing.expectError(error.FileNotFound, tmp.dir.access(testing_io, "a/b/notes:old.txt", .{}));
    }
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(testing_io, "a/b/c", .{}));
    try std.Io.Dir.cwd().createDirPath(vfs_io, try std.fmt.bufPrint(&uri_buffer, "{s}/a/d", .{root}));
    try tmp.dir.access(testing_io, "a/d", .{});
    try std.Io.Dir.cwd().deleteDir(vfs_io, try std.fmt.bufPrint(&uri_buffer, "{s}/a/d", .{root}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(testing_io, "a/d", .{}));
}

test "creates and deletes on a remote path fail and write nothing anywhere" {
    const testing_io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing_io, .{ .sub_path = "existing", .data = "bytes" });
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buffer[0..try tmp.dir.realPath(testing_io, &path_buffer)];

    // The inner `Io` itself serves `s3`, rooted at the test's directory: a
    // write that got past the VFS would land there.
    var vfs: VFS = try .init(std.testing.allocator, testing_io);
    defer vfs.deinit();
    try vfs.registerBackend("s3", .{ .io = testing_io });
    const vfs_io = vfs.io();
    const cwd = std.Io.Dir.cwd();

    var uri_buffer: [std.fs.max_path_bytes + 16]u8 = undefined;
    const uri = struct {
        fn at(buffer: []u8, root_: []const u8, name: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buffer, "s3://{s}/{s}", .{ root_, name });
        }
    }.at;
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.createFile(vfs_io, try uri(&uri_buffer, root, "key"), .{}));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.openFile(vfs_io, try uri(&uri_buffer, root, "existing"), .{ .mode = .read_write }));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.createFileAtomic(vfs_io, try uri(&uri_buffer, root, "key"), .{}));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.createDir(vfs_io, try uri(&uri_buffer, root, "dir"), .default_dir));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.createDirPath(vfs_io, try uri(&uri_buffer, root, "a/b")));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.createDirPathOpen(vfs_io, try uri(&uri_buffer, root, "a/b"), .{}));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.deleteFile(vfs_io, try uri(&uri_buffer, root, "existing")));
    try std.testing.expectError(error.ReadOnlyFileSystem, cwd.deleteDir(vfs_io, try uri(&uri_buffer, root, "dir")));
    // A directory opened on the remote is no way around it.
    const bucket = try cwd.openDir(vfs_io, try std.fmt.bufPrint(&uri_buffer, "s3://{s}", .{root}), .{});
    defer bucket.close(vfs_io);
    try std.testing.expectError(error.ReadOnlyFileSystem, bucket.createFile(vfs_io, "key", .{}));
    try std.testing.expectError(error.ReadOnlyFileSystem, bucket.deleteFile(vfs_io, "existing"));

    for ([_][]const u8{ "key", "dir", "a" }) |name| {
        try std.testing.expectError(error.FileNotFound, tmp.dir.access(testing_io, name, .{}));
    }
    try tmp.dir.access(testing_io, "existing", .{});
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
