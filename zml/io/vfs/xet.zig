//! VFS provider that reads HuggingFace files through the XET CAS protocol.
//!
//! Repo-scoped, like `hf.zig`: constructed for a single `repo/revision`, it
//! exchanges the HF token for a CAS read token, lists the repo's XET-backed
//! files (path -> {hash, size}), and serves positional reads by reconstructing
//! only the requested byte range from the CAS (see `../xet.zig`).
//!
//! Unlike the `resolve/`-based path in `hf.zig`, this fetches deduplicated
//! chunks directly from the CAS, so overlapping content across files/revisions
//! is only downloaded once (given a shared chunk cache).
//!
//! Handle bookkeeping mirrors `http.zig`.

const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;
const xet_core = @import("xet_core.zig");

const log = std.log.scoped(.@"zml/io/vfs/xet");

pub const Xet = struct {
    const Entry = struct {
        hash: [32]u8,
        size: u64,
    };

    const Handle = struct {
        pub const Type = enum { file, directory };

        type: Type,
        path: []const u8,
        hash: [32]u8,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, type_: Type, path: []const u8, hash: [32]u8, size: u64) !Handle {
            const owned = try allocator.dupe(u8, path);
            return .{
                .type = type_,
                .path = owned,
                .hash = hash,
                .pos = 0,
                .size = size,
            };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.path);
        }
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    session: *xet_core.Session,
    /// path -> {hash, size} for every XET-backed file in the repo/revision.
    entries: std.StringHashMapUnmanaged(Entry),
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .empty,
    base: VFSBase,

    /// Build a provider for `repo_id`@`revision`. Performs the CAS token
    /// exchange and lists the repo's XET files up front. `session` must outlive
    /// the provider (it owns the CAS connection); the caller owns it.
    pub fn init(
        allocator: std.mem.Allocator,
        inner: std.Io,
        environ: std.process.Environ,
        session: *xet_core.Session,
        repo_type: []const u8,
        repo_id: []const u8,
        revision: []const u8,
        hf_token: []const u8,
    ) !Xet {
        const files = try xet_core.listXetFiles(allocator, inner, environ, repo_type, repo_id, revision, hf_token);
        defer allocator.free(files);

        var entries: std.StringHashMapUnmanaged(Entry) = .empty;
        errdefer {
            var it = entries.iterator();
            while (it.next()) |kv| allocator.free(kv.key_ptr.*);
            entries.deinit(allocator);
            for (files) |f| allocator.free(f.hash_hex);
        }

        for (files) |f| {
            const hash = try xet_core.parseHash(f.hash_hex);
            allocator.free(f.hash_hex);
            try entries.put(allocator, f.path, .{ .hash = hash, .size = f.size });
        }

        return .{
            .allocator = allocator,
            .base = .init(inner),
            .session = session,
            .entries = entries,
        };
    }

    pub fn deinit(self: *Xet) void {
        var idx: usize = 0;
        while (idx < self.handles.len) : (idx += 1) {
            const is_closed = for (self.closed_handles.items) |closed_idx| {
                if (closed_idx == idx) break true;
            } else false;
            if (!is_closed) self.handles.at(idx).deinit(self.allocator);
        }
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);

        var it = self.entries.iterator();
        while (it.next()) |kv| self.allocator.free(kv.key_ptr.*);
        self.entries.deinit(self.allocator);
    }

    pub fn io(self: *Xet) std.Io {
        return .{
            .userdata = &self.base,
            .vtable = &comptime VFSBase.vtable(.{
                .dirStat = dirStat,
                .dirStatFile = dirStatFile,
                .dirAccess = dirAccess,
                .dirOpenFile = dirOpenFile,
                .dirClose = dirClose,
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

    fn openHandle(self: *Xet) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| {
            return .{ idx, self.handles.at(idx) };
        }
        return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *Xet, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        self.handles.at(idx).deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *Xet, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(file.handle));
    }

    /// Look up a repo-relative path in the XET file table.
    fn lookup(self: *Xet, sub_path: []const u8) ?Entry {
        if (self.entries.get(sub_path)) |e| return e;
        // Tolerate a leading "./" or "/".
        const trimmed = std.mem.trimStart(u8, sub_path, "./");
        return self.entries.get(trimmed);
    }

    fn dirStat(_: ?*anyopaque, _: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        return .{
            .inode = 0,
            .nlink = 0,
            .size = 0,
            .permissions = .fromMode(0o444),
            .kind = .directory,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 0,
        };
    }

    fn dirStatFile(userdata: ?*anyopaque, _: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        const entry = self.lookup(sub_path) orelse return std.Io.Dir.StatFileError.FileNotFound;
        return fileStatFromSize(entry.size);
    }

    fn dirAccess(userdata: ?*anyopaque, _: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        if (self.lookup(sub_path) == null) return std.Io.Dir.AccessError.FileNotFound;
    }

    fn dirOpenFile(userdata: ?*anyopaque, _: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        const entry = self.lookup(sub_path) orelse return std.Io.File.OpenError.FileNotFound;

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, sub_path, entry.hash, entry.size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| self.closeHandle(@intCast(dir.handle)) catch unreachable;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, _: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        _ = userdata;
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{path_name}) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        return fileStatFromSize(self.getFileHandle(file).size);
    }

    fn fileStatFromSize(size: u64) std.Io.File.Stat {
        return .{
            .inode = 0,
            .nlink = 0,
            .size = size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 1,
        };
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| self.closeHandle(@intCast(file.handle)) catch unreachable;
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| {
            log.err("xet read failed for {s} at {d}: {any}", .{ handle.path, offset, err });
            return std.Io.File.ReadPositionalError.Unexpected;
        };
    }

    fn performRead(self: *Xet, handle: *Handle, data: []const []u8, offset: u64) !usize {
        if (offset >= handle.size or data.len == 0 or data[0].len == 0) return 0;

        const remaining = handle.size - offset;
        const take = @min(remaining, @as(u64, data[0].len));
        const end = offset + take;

        const bytes = try self.session.readRange(handle.hash, offset, end);
        defer self.allocator.free(bytes);

        @memcpy(data[0][0..bytes.len], bytes);
        return bytes.len;
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        handle.pos = if (relative_offset >= 0)
            handle.pos + @as(u64, @intCast(relative_offset))
        else
            handle.pos - @as(u64, @intCast(-relative_offset));
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        self.getFileHandle(file).pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *Xet = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.path}) catch return std.Io.File.RealPathError.SystemResources;
        return path.len;
    }
};
