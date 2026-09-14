const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;
const Backend = @import("base.zig").Backend;
const AtomicReadStats = @import("base.zig").AtomicReadStats;
const range_read = @import("range_read.zig");
const request = @import("request.zig");

const log = std.log.scoped(.@"zml/vfs/http");

pub const HTTP = struct {
    pub const InitOpts = struct {
        /// Retries per request for failures that are not rate limiting.
        max_retries: usize = 5,
        retry_initial_delay: std.Io.Duration = .fromMilliseconds(500),
        retry_max_delay: std.Io.Duration = .fromSeconds(30),
        /// Longest hold one throttle may arm over every request of this
        /// backend, a server-named delay included.
        max_hold: std.Io.Duration = .fromSeconds(120),
        /// Continuous rate limiting for longer than this fails the requests
        /// with `error.RateLimited`.
        throttle_budget: std.Io.Duration = .fromSeconds(300),
    };

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
    governor: request.Governor,
    read_stats: AtomicReadStats = .{},
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .empty,
    base: VFSBase,

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, protocol: Protocol) !HTTP {
        return initWithOptions(allocator, inner, http_client, protocol, .{});
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        protocol: Protocol,
        opts: InitOpts,
    ) !HTTP {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .protocol = protocol,
            .governor = .init(.fromOptions(opts)),
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
                .operate = operate,
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

    pub fn backend(self: *HTTP) Backend {
        return .{
            .io = self.io(),
            .read_hints = .{ .high_latency = true },
            .read_stats = self.read_stats.provider(),
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

    fn operate(userdata: ?*anyopaque, operation: std.Io.Operation) std.Io.Cancelable!std.Io.Operation.Result {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        switch (operation) {
            .file_read_streaming => |o| {
                const handle = self.getFileHandle(o.file);
                const total = self.performRead(handle, o.data, handle.pos) catch |err| {
                    log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, handle.pos, err });
                    return .{ .file_read_streaming = error.EndOfStream };
                };

                if (total == 0) {
                    return .{ .file_read_streaming = error.EndOfStream };
                }

                handle.pos += @intCast(total);
                return .{ .file_read_streaming = total };
            },
            .file_write_streaming, .device_io_control, .net_receive => {
                return self.base.inner.vtable.operate(self.base.inner.userdata, operation);
            },
        }
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, path, 0) catch return std.Io.Dir.OpenError.Unexpected;

        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
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
            .block_size = 0,
        };
    }

    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const size = self.fetchSize(dir, sub_path) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => return std.Io.Dir.StatFileError.Unexpected,
        };

        return .{
            .inode = @intCast(0),
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

    fn dirAccess(_: ?*anyopaque, _: std.Io.Dir, _: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {}

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));

        const size = self.fetchSize(dir, sub_path) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => return std.Io.File.OpenError.Unexpected,
        };

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, path, size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRead(_: ?*anyopaque, _: *std.Io.Dir.Reader, _: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        log.err("dirRead unsupported", .{});
        return std.Io.Dir.Reader.Error.Unexpected;
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));

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
            .block_size = 1,
        };
    }

    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| switch (err) {
            // A cancelled task must not surface as an I/O failure: every
            // wait in the governed loop is a cancellation point.
            error.Canceled => return error.Canceled,
            else => {
                log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
                return std.Io.File.ReadPositionalError.Unexpected;
            },
        };
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);

        handle.pos = if (relative_offset >= 0)
            handle.pos + @as(u64, @intCast(relative_offset))
        else
            handle.pos - @as(u64, @intCast(-relative_offset));
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
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
            // Each hop is one governed request: a server that rate limits
            // the HEAD holds this backend as a throttled GET would.
            var hop: SizeHop = .{ .http = self, .uri = uri, .url = url };
            const outcome = try request.perform(SizeHop.Result, self.requestContext(), .{
                .backend = "http",
                .target = url,
                .unavailable = .server_failure,
                .key = request.authorityOf(uri),
            }, &hop, SizeHop.attempt);
            switch (outcome) {
                .size => |size| return size,
                .redirect => |location| {
                    if (location.len > aux_buffer.len) return error.HttpRedirectLocationOversize;
                    @memcpy(aux_buffer[0..location.len], location);
                    uri = uri.resolveInPlace(location.len, &aux_buffer) catch unreachable;
                },
            }
        }
    }

    /// One HEAD of the redirect chain: the size, or the `Location` to
    /// follow, copied into the hop's own storage because the head buffer
    /// dies with the response.
    const SizeHop = struct {
        const Result = union(enum) { size: u64, redirect: []const u8 };

        http: *HTTP,
        uri: std.Uri,
        url: []const u8,
        location: [8 * 1024]u8 = undefined,

        fn attempt(self: *SizeHop, _: request.Attempt) anyerror!request.Outcome(Result) {
            var head_buffer: [8 * 1024]u8 = undefined;
            return request.exchange(Result, self.http.requestContext(), self.uri, .{
                .method = .HEAD,
                .headers = .{ .accept_encoding = .{ .override = "identity" } },
                .redirects = .surface,
                .head_buffer = &head_buffer,
            }, .{
                .backend = "http",
                .target = self.url,
                .unavailable = .server_failure,
                .key = request.authorityOf(self.uri),
            }, self, SizeHop.consume);
        }

        fn consume(self: *SizeHop, res: *std.http.Client.Response) anyerror!Result {
            switch (res.head.status.class()) {
                .success => return .{ .size = res.head.content_length orelse return error.MissingContentLength },
                .redirect => {
                    const location = res.head.location orelse return error.HttpRedirectLocationMissing;
                    if (location.len > self.location.len) return error.HttpRedirectLocationOversize;
                    @memcpy(self.location[0..location.len], location);
                    return .{ .redirect = self.location[0..location.len] };
                },
                else => return error.UnexpectedStatus,
            }
        }
    };

    /// Everything a governed request needs from this backend.
    fn requestContext(self: *HTTP) request.Context {
        return .{
            .io = self.base.inner,
            .client = self.client,
            .governor = &self.governor,
            .stats = &self.read_stats,
        };
    }

    fn performRead(self: *HTTP, handle: *Handle, data: []const []u8, offset: u64) !usize {
        var url_buffer: [8 * 1024]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "{s}://{s}", .{ @tagName(self.protocol), handle.uri });
        const uri: std.Uri = try .parse(url);
        var prepared: range_read.PreparedRequest = .{ .uri = uri };
        return range_read.performRangeRead(self.requestContext(), .{
            .request = .{
                .backend = "http",
                .target = url,
                .unavailable = .server_failure,
                .key = request.authorityOf(uri),
            },
            .context = &prepared,
            .prepare = range_read.prepareStatic,
        }, data, offset, range_read.readSize(handle.size, offset, data));
    }
};
