const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/http");

pub const HTTP = struct {
    const Handle = struct {
        pub const Type = enum {
            file,
            directory,
        };

        type: Type,
        uri: []const u8,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, type_: Type, path: []const u8, size: u64) !Handle {
            const uri = try allocator.dupe(u8, path);
            errdefer allocator.free(uri);

            return .{
                .type = type_,
                .uri = uri,
                .pos = 0,
                .size = size,
            };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.uri);
        }
    };

    const Protocol = enum { http, https };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    client: *std.http.Client,
    protocol: Protocol,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .{},
    base: VFSBase,

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, protocol: Protocol) !HTTP {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .protocol = protocol,
        };
    }

    pub fn deinit(self: *HTTP) void {
        var idx: usize = 0;
        while (idx < self.handles.len) : (idx += 1) {
            const is_closed = for (self.closed_handles.items) |closed_idx| {
                if (closed_idx == idx) break true;
            } else false;

            if (!is_closed) {
                self.handles.at(idx).deinit(self.allocator);
            }
        }
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);
    }

    pub fn io(self: *HTTP) std.Io {
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

    fn openHandle(self: *HTTP) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| {
            return .{ idx, self.handles.at(idx) };
        }
        return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *HTTP, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        self.handles.at(idx).deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *HTTP, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(file.handle));
    }

    fn getDirHandle(self: *HTTP, dir: std.Io.Dir) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(dir.handle));
    }

    fn resolvePath(self: *HTTP, dir: std.Io.Dir, sub_path: []const u8, out_buffer: []u8) ![]u8 {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return try std.fmt.bufPrint(out_buffer, "{s}", .{sub_path});
        }

        const handle = self.getDirHandle(dir);

        const trimmed_uri = std.mem.trimEnd(u8, handle.uri, "/");
        const trimmed_sub_path = std.mem.trimStart(u8, sub_path, "/");

        if (trimmed_uri.len == 0) return try std.fmt.bufPrint(out_buffer, "{s}", .{trimmed_sub_path});
        if (trimmed_sub_path.len == 0) return try std.fmt.bufPrint(out_buffer, "{s}", .{trimmed_uri});
        return try std.fmt.bufPrint(out_buffer, "{s}/{s}", .{ trimmed_uri, trimmed_sub_path });
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, path, 0) catch return std.Io.Dir.OpenError.Unexpected;

        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);

        return .{
            .inode = @intCast(0),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .directory,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
        };
    }

    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch return std.Io.Dir.StatFileError.Unexpected;

        return .{
            .inode = @intCast(0),
            .nlink = 0,
            .size = size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
        };
    }

    fn dirAccess(_: ?*anyopaque, _: std.Io.Dir, _: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {}

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));

        const size = self.fetchSize(dir, sub_path) catch return std.Io.File.OpenError.Unexpected;

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, path, size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx) };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRead(_: ?*anyopaque, _: *std.Io.Dir.Reader, _: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        log.err("dirRead unsupported", .{});
        return std.Io.Dir.Reader.Error.Unexpected;
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));

        const handle = self.getFileHandle(file);

        return .{
            .inode = @intCast(file.handle),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
        };
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const total = self.performRead(handle, data, handle.pos) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, handle.pos, err });
            return std.Io.File.Reader.Error.Unexpected;
        };
        handle.pos += @intCast(total);
        return total;
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.Reader.Error.Unexpected;
        };
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);

        handle.pos = if (relative_offset >= 0)
            handle.pos + @as(u64, @intCast(relative_offset))
        else
            handle.pos - @as(u64, @intCast(-relative_offset));
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.File.RealPathError.SystemResources;
        return path.len;
    }

    fn fetchSize(self: *HTTP, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        var redirect_buffer: [8 * 1024]u8 = undefined;
        var aux_buffer: []u8 = &redirect_buffer;

        const scheme = @tagName(self.protocol);
        const url = try self.resolvePath(dir, sub_path, &path_buffer);
        const full_url = try std.fmt.bufPrint(aux_buffer, "{s}://{s}", .{ scheme, url });

        var uri = std.Uri.parse(full_url) catch return std.Io.File.OpenError.BadPathName;
        while (true) {
            var req = try self.client.request(.HEAD, uri, .{
                .redirect_behavior = .not_allowed,
                .headers = .{ .accept_encoding = .{ .override = "identity" } },
            });
            defer req.deinit();

            try req.sendBodiless();

            var res = try req.receiveHead(&redirect_buffer);

            switch (res.head.status.class()) {
                .server_error, .client_error => {
                    log.err("Failed to fetch tree size for {s}", .{url});
                    log.err("{s}", .{res.head.bytes});
                    return error.ServerError;
                },
                .informational => return error.UnexpectedStatus,
                .success => return res.head.content_length.?,
                .redirect => {
                    const location = res.head.location.?;
                    @memcpy(aux_buffer[0..location.len], location);
                    uri = uri.resolveInPlace(location.len, &aux_buffer) catch unreachable;
                    continue;
                },
            }
        }
    }

    fn performRead(self: *HTTP, handle: *Handle, data: []const []u8, offset: u64) !usize {
        if (offset >= handle.size) return 0;

        var range_buf: [64]u8 = undefined;
        const range_header = blk: {
            var total_bytes: u64 = 0;
            for (data) |buf| {
                total_bytes += @as(u64, buf.len);
            }
            const remaining = handle.size - offset;
            const take = @min(remaining, total_bytes);
            const end = offset + take - 1;
            break :blk std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ offset, end }) catch unreachable;
        };

        var url_buffer: [8 * 1024]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "{s}://{s}", .{ @tagName(self.protocol), handle.uri });
        const uri: std.Uri = try .parse(url);

        var req = try self.client.request(.GET, uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .partial_content and res.head.status != .ok) {
            log.err("Failed to perform read for {s}", .{handle.uri});
            log.err("{s}", .{res.head.bytes});
            return error.RequestFailed;
        }

        const content_range = blk: {
            var it = res.head.iterateHeaders();
            while (it.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) {
                    break :blk parseContentRange(header.value);
                }
            }
            break :blk null;
        };

        const reader = res.reader(&.{});

        if (content_range) |cr| {
            if (cr.start < offset) {
                try reader.discardAll(offset - cr.start);
            }
        }

        return try reader.readSliceShort(data[0]);
    }

    const ContentRange = struct {
        start: u64,
        end: u64,
        total: u64,
    };

    fn parseContentRange(value: []const u8) ?ContentRange {
        const space = std.mem.indexOfScalar(u8, value, ' ') orelse return null;
        const dash = std.mem.indexOfScalar(u8, value, '-') orelse return null;
        const slash = std.mem.indexOfScalar(u8, value, '/') orelse return null;

        return .{
            .start = std.fmt.parseInt(u64, value[space + 1 .. dash], 10) catch return null,
            .end = std.fmt.parseInt(u64, value[dash + 1 .. slash], 10) catch return null,
            .total = std.fmt.parseInt(u64, value[slash + 1 ..], 10) catch return null,
        };
    }
};
