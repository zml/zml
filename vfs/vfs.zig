const std = @import("std");

const stdx = @import("stdx");

pub const GCS = @import("gcs.zig").GCS;
pub const HF = @import("hf.zig").HF;
pub const HTTP = @import("http.zig").HTTP;
pub const S3 = @import("s3.zig").S3;
const base_module = @import("base.zig");
const direct_io = @import("direct_io.zig");
pub const Backend = base_module.Backend;
pub const ReadHints = base_module.ReadHints;
pub const ReadStats = base_module.ReadStats;
pub const ReadStatsProvider = base_module.ReadStatsProvider;
pub const VFSBase = base_module.VFSBase;
/// See `useDirectIo`.
pub const DirectIo = direct_io.Policy;

test {
    _ = @import("http_acceptance_test.zig");
}

const log = std.log.scoped(.@"zml/vfs");

const CWD_HANDLE: u32 = 0;
/// Local files are the VFS's own: a bare path or a `file://` URI resolves
/// to the inner `Io`, never to a registered backend.
const local_scheme = "file";

const VFS = @This();
const Handle = struct {
    handle: u32,
    backend_idx: ?usize,
    flags: std.Io.File.Flags = .{ .nonblocking = false },
    /// Opened for reading only. `useDirectIo` needs it: the flag constrains
    /// writes too.
    read_only: bool = false,
    /// A local file's read mode (`useDirectIo`). Transitions that touch the
    /// descriptor's flag happen under the VFS mutex; reads only load it.
    direct: std.atomic.Value(direct_io.State) = .init(.undecided),

    fn innerFile(self: *const Handle) std.Io.File {
        return .{ .handle = @intCast(self.handle), .flags = self.flags };
    }

    fn fd(self: *const Handle) std.posix.fd_t {
        return @intCast(self.handle);
    }

    /// Whether reads of this handle go past the page cache right now.
    fn isDirect(self: *const Handle) bool {
        if (comptime !direct_io.supported) return false;
        return self.direct.load(.acquire) == .direct;
    }
};

pub const LoadProfile = struct {
    /// Generic fallback used by callers that do not prepare a profile from a
    /// VFS path. This value is borrowed and does not require deinitialization.
    pub const default: LoadProfile = .{
        .name = "default",
        .read_chunk_size = 16 * 1024 * 1024,
        .high_latency = false,
        .direct_io_alignment = null,
        .stats = null,
    };

    /// Local files read without a VFS: buffered, exact reads.
    pub const local: LoadProfile = .{
        .name = "local",
        .read_chunk_size = 8 * 1024 * 1024,
        .high_latency = false,
        .direct_io_alignment = null,
        .stats = null,
    };

    name: []const u8,
    /// Minimum source request size. The loader may increase it to match the
    /// independently calibrated DMA block size.
    read_chunk_size: usize,
    high_latency: bool,
    /// What a direct read of one of this profile's files must meet in
    /// offset, buffers and total once the VFS that opened the file made it
    /// direct (`useDirectIo`); null when no read of them is ever direct.
    direct_io_alignment: ?usize,
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

pub const RegisterError = error{
    /// `file` is the VFS's own scheme (`local_scheme`).
    ReservedScheme,
} || std.mem.Allocator.Error;

pub fn registerBackend(self: *VFS, scheme: []const u8, backend: Backend) RegisterError!void {
    if (std.mem.eql(u8, scheme, local_scheme)) return error.ReservedScheme;
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

/// Prepares the source tuning and feedback provider for one model load. A
/// path resolves as it does for `openFile`: a bare path or a `file://` URI
/// is local, served by the inner `Io`, and the loader may read it directly
/// (`useDirectIo`).
/// Returned strings and providers borrow backend state, so this VFS and its
/// registered backend must outlive the load.
pub fn loadProfile(self: *VFS, path: []const u8) !LoadProfile {
    const backend_idx, _, _ = try self.lookupDir(.cwd(), path);
    const index = backend_idx orelse {
        var profile = LoadProfile.local;
        profile.direct_io_alignment = if (direct_io.supported) direct_io.alignment else null;
        return profile;
    };
    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);
    const backend = self.backends.entries.items(.value)[index];
    return .{
        .name = self.backends.entries.items(.key)[index],
        .read_chunk_size = backend.read_hints.read_chunk_size,
        .high_latency = backend.read_hints.high_latency,
        .direct_io_alignment = null,
        .stats = backend.read_stats,
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

/// Direct reads for a local file under `policy`: sets the flag on the
/// descriptor of a read-only regular file at least one alignment unit long
/// (and, under `auto`, mostly out of the page cache) and returns true, after
/// which every read of the file must meet `LoadProfile.direct_io_alignment`
/// in offset, buffers and total. False leaves the file buffered: another
/// backend's file, `off`, a cached file under `auto`, a filesystem that
/// refuses the flag, or a file whose direct read the kernel already
/// rejected. The first call under `on` or `auto` decides for the handle's
/// life, since `O_DIRECT` belongs to the open file description and a caller
/// answered buffered plans exact reads that a direct descriptor would
/// reject; the caller must ask before it reads. A read the kernel rejects
/// afterwards is answered buffered, logged once, and the file stays
/// buffered.
pub fn useDirectIo(self: *VFS, file: std.Io.File, policy: DirectIo) bool {
    if (comptime !direct_io.supported) return false;
    if (policy == .off) return false;
    const handle, _ = self.getFileHandle(file);
    if (handle.backend_idx != null) return false;
    switch (handle.direct.load(.acquire)) {
        .direct => return true,
        .buffered => return false,
        .undecided => {},
    }
    if (!handle.read_only) return decideBuffered(handle);
    const stat = handle.innerFile().stat(self.base.inner) catch return decideBuffered(handle);
    if (stat.kind != .file or stat.size < direct_io.alignment) return decideBuffered(handle);
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const name = self.localName(handle, &path_buffer);
    if (policy == .auto) {
        const cached = direct_io.cachedFraction(handle.fd(), stat.size) orelse {
            log.debug("{s}: page cache residency unknown; reading buffered", .{name});
            return decideBuffered(handle);
        };
        if (cached > direct_io.cold_fraction) {
            log.debug("{s}: {d:.0}% cached; reading buffered", .{ name, cached * 100 });
            return decideBuffered(handle);
        }
        log.debug("{s}: {d:.0}% cached", .{ name, cached * 100 });
    }
    switch (self.enterDirect(handle)) {
        .direct => {
            log.debug("{s}: reading direct", .{name});
            return true;
        },
        .buffered => return false,
        .refused => |err| {
            log.debug("{s}: O_DIRECT refused ({t}); reading buffered", .{ name, err });
            return false;
        },
    }
}

/// Decides buffered for an undecided handle; a decision already taken
/// stands.
fn decideBuffered(handle: *Handle) bool {
    _ = handle.direct.cmpxchgStrong(.undecided, .buffered, .acq_rel, .acquire);
    return handle.direct.load(.acquire) == .direct;
}

const DirectEntry = union(enum) { direct, buffered, refused: std.os.linux.E };

/// Sets the flag on an undecided handle's descriptor and records the
/// decision, under the mutex: the flag and the state change together, so a
/// demotion racing with this call cannot leave the flag on a buffered
/// handle. A filesystem that refuses the flag leaves the handle buffered.
fn enterDirect(self: *VFS, handle: *Handle) DirectEntry {
    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);
    switch (handle.direct.load(.acquire)) {
        .direct => return .direct,
        .buffered => return .buffered,
        .undecided => {},
    }
    const err = direct_io.enable(handle.fd());
    if (err != .SUCCESS) {
        handle.direct.store(.buffered, .release);
        return .{ .refused = err };
    }
    handle.direct.store(.direct, .release);
    return .direct;
}

/// Takes a direct handle back to buffered: the flag comes off the
/// descriptor before the state says buffered, under the mutex, so the inner
/// `Io` never reads a descriptor with the flag on. True for the call that
/// did it, which then says why.
fn leaveDirect(self: *VFS, handle: *Handle) bool {
    if (comptime !direct_io.supported) return false;
    self.mutex.lockUncancelable(self.base.inner);
    defer self.mutex.unlock(self.base.inner);
    if (handle.direct.load(.acquire) != .direct) return false;
    direct_io.disable(handle.fd());
    handle.direct.store(.buffered, .release);
    return true;
}

/// The path of a local handle, for a log line.
fn localName(self: *VFS, handle: *Handle, buffer: []u8) []const u8 {
    const len = handle.innerFile().realPath(self.base.inner, buffer) catch return "<local file>";
    return buffer[0..len];
}

/// The kernel rejected a direct read. An aligned one is the filesystem
/// refusing direct I/O it accepted the flag for, a warning; a misaligned
/// one is the reader's, either its plan or a continuation after a short
/// read (which a network filesystem may answer), an error. Either way the
/// file is read buffered from now on.
fn refuseDirect(self: *VFS, handle: *Handle, err: std.os.linux.E, data: []const []u8, offset: u64) void {
    if (!self.leaveDirect(handle)) return;
    var total: usize = 0;
    var aligned = offset % direct_io.alignment == 0;
    for (data) |buffer| {
        if (buffer.len == 0) continue;
        aligned = aligned and @intFromPtr(buffer.ptr) % direct_io.alignment == 0;
        total += buffer.len;
    }
    aligned = aligned and total % direct_io.alignment == 0;
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const name = self.localName(handle, &path_buffer);
    if (aligned) {
        log.warn("{s}: direct read refused ({t}) at offset={d} total={d}; reading the file buffered from now on", .{ name, err, offset, total });
    } else {
        log.err("{s}: misaligned read of a direct file ({t}) at offset={d} first_buffer=0x{x} total={d} alignment={d}; reading the file buffered from now on", .{
            name,
            err,
            offset,
            if (data.len != 0) @intFromPtr(data[0].ptr) else 0,
            total,
            direct_io.alignment,
        });
    }
}

/// A local file as the inner `Io` may read it. The inner `Io` treats the
/// errno of a rejected direct read as a programmer error, so a read the VFS
/// does not issue itself (streaming, or as the source of a file copy) takes
/// the flag off a direct file first.
fn innerReadable(self: *VFS, handle: *Handle) std.Io.File {
    if (handle.isDirect() and self.leaveDirect(handle)) {
        var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
        log.warn("{s}: buffered read of a direct file; reading it buffered from now on", .{self.localName(handle, &path_buffer)});
    }
    return handle.innerFile();
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
            return self.schemeRoot(uri.scheme);
        }
    }

    if (std.meta.eql(dir, std.Io.Dir.cwd())) {
        if (sub_path == null) return .{ null, dir, self.base.inner };
        if (std.fs.path.isAbsolutePosix(sub_path.?)) return self.localRoot();
        const uri = std.Uri.parse(sub_path.?) catch return self.localRoot();
        return self.schemeRoot(uri.scheme);
    } else {
        const handle = self.getDirHandle(dir);
        if (handle.backend_idx) |backend_idx| {
            return .{ backend_idx, .{ .handle = @intCast(handle.handle) }, self.getBackend(backend_idx) };
        } else {
            return .{ null, .{ .handle = @intCast(handle.handle) }, self.base.inner };
        }
    }
}

/// The root of a scheme: the inner `Io` for local files, otherwise the
/// backend registered for it.
fn schemeRoot(self: *VFS, scheme: []const u8) !struct { ?usize, std.Io.Dir, std.Io } {
    if (std.mem.eql(u8, scheme, local_scheme)) return self.localRoot();
    const backend_idx = self.backends.getIndex(scheme) orelse return error.VFSNotRegistered;
    return .{ backend_idx, std.Io.Dir.cwd(), self.getBackend(backend_idx) };
}

fn localRoot(self: *VFS) struct { ?usize, std.Io.Dir, std.Io } {
    return .{ null, std.Io.Dir.cwd(), self.base.inner };
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
                .file = self.innerReadable(handle),
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
        .read_only = flags.mode == .read_only,
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
    return backend.vtable.fileStat(backend.userdata, handle.innerFile());
}

fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    return backend.vtable.fileLength(backend.userdata, handle.innerFile());
}

fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    for (files) |file| {
        const handle, const backend = self.getFileHandle(file);
        backend.vtable.fileClose(backend.userdata, &.{handle.innerFile()});
        self.closeHandle(@intCast(file.handle)) catch unreachable;
    }
}

fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) std.Io.File.WritePositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    return backend.vtable.fileWritePositional(backend.userdata, handle.innerFile(), header, data, splat, offset);
}

fn fileWriteFileStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.File.Writer.WriteFileError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));

    const dst_handle, const dst_backend = self.getFileHandle(file);
    const src_handle, _ = self.getFileHandle(reader.file);

    const original_src_handle = reader.file;
    reader.file = self.innerReadable(src_handle);
    defer reader.file = original_src_handle;

    return dst_backend.vtable.fileWriteFileStreaming(dst_backend.userdata, dst_handle.innerFile(), header, reader, limit);
}

fn fileWriteFilePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit, offset: u64) std.Io.File.WriteFilePositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));

    const dst_handle, const dst_backend = self.getFileHandle(file);
    const src_handle, _ = self.getFileHandle(reader.file);

    const original_src_handle = reader.file;
    reader.file = self.innerReadable(src_handle);
    defer reader.file = original_src_handle;

    return dst_backend.vtable.fileWriteFilePositional(dst_backend.userdata, dst_handle.innerFile(), header, reader, limit, offset);
}

fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    if (comptime direct_io.supported) {
        if (handle.isDirect()) {
            switch (try direct_io.readPositional(handle.fd(), data, offset)) {
                .bytes => |bytes| return bytes,
                .refused => |err| self.refuseDirect(handle, err, data, offset),
            }
        }
    }
    return backend.vtable.fileReadPositional(backend.userdata, handle.innerFile(), data, offset);
}

fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    return backend.vtable.fileSeekBy(backend.userdata, handle.innerFile(), relative_offset);
}

fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);
    return backend.vtable.fileSeekTo(backend.userdata, handle.innerFile(), absolute_offset);
}

fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
    const self: *VFS = @fieldParentPtr("base", VFSBase.as(userdata));
    const handle, const backend = self.getFileHandle(file);

    if (self.getScheme(handle.backend_idx)) |s| {
        const prefix = try std.fmt.bufPrint(out_buffer, "{s}://", .{s});
        const path_len = try backend.vtable.fileRealPath(backend.userdata, handle.innerFile(), out_buffer[prefix.len..]);
        return prefix.len + path_len;
    } else {
        return try backend.vtable.fileRealPath(backend.userdata, handle.innerFile(), out_buffer);
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

    const local_alignment: ?usize = if (direct_io.supported) direct_io.alignment else null;
    for ([_][]const u8{ "/tmp/model.safetensors", "models/model.safetensors", "file:///tmp/model.safetensors" }) |path| {
        const local = try filesystem.loadProfile(path);
        try std.testing.expectEqualStrings("local", local.name);
        try std.testing.expectEqual(@as(usize, 8 * 1024 * 1024), local.read_chunk_size);
        try std.testing.expect(!local.high_latency);
        try std.testing.expectEqual(local_alignment, local.direct_io_alignment);
        try std.testing.expect(local.stats == null);
    }
    try std.testing.expectError(
        error.VFSNotRegistered,
        filesystem.loadProfile("missing://bucket/object"),
    );
    try std.testing.expectError(error.ReservedScheme, filesystem.registerBackend("file", .{ .io = std.testing.io }));
    try std.testing.expectEqual(@as(?*VFS, &filesystem), VFS.fromIo(filesystem.io()));
    try std.testing.expectEqual(@as(?*VFS, null), VFS.fromIo(std.testing.io));
}

test "VFS reports the configured load profile for every bundled backend" {
    var client: std.http.Client = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };
    defer client.deinit();

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
    try filesystem.registerBackend("https", http.backend());
    try filesystem.registerBackend("s3", s3.backend());
    try filesystem.registerBackend("gs", gcs.backend());
    try filesystem.registerBackend("hf", hf.backend());

    const Case = struct {
        path: []const u8,
        name: []const u8,
        read_chunk_size: usize,
        high_latency: bool,
        direct_io_alignment: ?usize = null,
    };
    const local_alignment: ?usize = if (direct_io.supported) direct_io.alignment else null;
    const cases = [_]Case{
        .{ .path = "file:///tmp/model", .name = "local", .read_chunk_size = 8 * 1024 * 1024, .high_latency = false, .direct_io_alignment = local_alignment },
        .{ .path = "/var/models/model", .name = "local", .read_chunk_size = 8 * 1024 * 1024, .high_latency = false, .direct_io_alignment = local_alignment },
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
        try std.testing.expectEqual(case.direct_io_alignment, profile.direct_io_alignment);
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
        const opened = try std.Io.Dir.openFile(.cwd(), vfs_io, open_path, .{ .mode = .read_only });
        defer opened.close(vfs_io);
        const handle, _ = filesystem.getFileHandle(opened);
        try std.testing.expectEqual(@as(?usize, null), handle.backend_idx);
        // A file this small is never direct.
        try std.testing.expect(!filesystem.useDirectIo(opened, .on));
        var contents: [15]u8 = undefined;
        try std.testing.expectEqual(contents.len, try opened.readPositionalAll(vfs_io, &contents, 0));
        try std.testing.expectEqualStrings("bare path bytes", &contents);
    }
}

test "VFS reads the same bytes under every direct I/O policy" {
    const testing_io = std.testing.io;
    const alignment = 4096;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var contents: [3 * alignment + 100]u8 = undefined;
    for (&contents, 0..) |*byte, i| byte.* = @truncate(i * 7 + 3);
    const created = try tmp.dir.createFile(testing_io, "data.bin", .{ .read = true });
    try created.writePositionalAll(testing_io, &contents, 0);
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buffer[0..try created.realPath(testing_io, &path_buffer)];
    created.close(testing_io);

    var filesystem = try VFS.init(std.testing.allocator, testing_io);
    defer filesystem.deinit();
    const vfs_io = filesystem.io();
    const buffer = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(alignment), 2 * alignment);
    defer std.testing.allocator.free(buffer);
    const plain: []u8 = buffer;
    for ([_]DirectIo{ .on, .auto, .off }) |policy| {
        const file = try std.Io.Dir.openFile(.cwd(), vfs_io, path, .{ .mode = .read_only });
        defer file.close(vfs_io);
        // Direct when the policy and the filesystem allow it, buffered
        // otherwise: `off`, or under `auto` a file just written and thus
        // cached. The bytes are the same either way.
        const direct = filesystem.useDirectIo(file, policy);
        if (policy != .on) try std.testing.expect(!direct);
        // The decision stands; `off` neither asks nor changes it.
        try std.testing.expect(!filesystem.useDirectIo(file, .off));
        try std.testing.expectEqual(direct, filesystem.useDirectIo(file, policy));

        // Aligned offset, buffer and length.
        @memset(buffer, 0);
        try std.testing.expectEqual(buffer.len, try file.readPositionalAll(vfs_io, buffer, alignment));
        try std.testing.expectEqualSlices(u8, contents[alignment..][0..buffer.len], buffer);

        // One aligned call that runs past the end of the file is cut there.
        @memset(buffer, 0);
        const past_end = try file.readPositional(vfs_io, &.{plain}, 2 * alignment);
        try std.testing.expectEqual(contents.len - 2 * alignment, past_end);
        try std.testing.expectEqualSlices(u8, contents[2 * alignment ..], buffer[0..past_end]);
        try std.testing.expectEqual(direct, filesystem.useDirectIo(file, policy));

        // A streaming read takes the file off direct reads for good. (A
        // misaligned positional read of a direct file does too, with an
        // error logged, which the test runner would count; a filesystem
        // such as tmpfs may serve it instead, so it is not exercised here.)
        var head: [8]u8 = undefined;
        const head_slice: []u8 = &head;
        try std.testing.expectEqual(head.len, try file.readStreaming(vfs_io, &.{head_slice}));
        try std.testing.expectEqualSlices(u8, contents[0..head.len], &head);
        try std.testing.expect(!filesystem.useDirectIo(file, policy));

        // Buffered from now on: unaligned reads and the end of the file.
        var tail: [50]u8 = undefined;
        try std.testing.expectEqual(tail.len, try file.readPositionalAll(vfs_io, &tail, alignment + 3));
        try std.testing.expectEqualSlices(u8, contents[alignment + 3 ..][0..tail.len], &tail);
        try std.testing.expectEqual(@as(usize, 0), try file.readPositional(vfs_io, &.{plain}, contents.len));
    }
}
