const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;
const Backend = @import("base.zig").Backend;
const ReadStats = @import("base.zig").ReadStats;
const AtomicReadStats = @import("base.zig").AtomicReadStats;
const range_read = @import("range_read.zig");

const log = std.log.scoped(.@"zml/io/vfs/http");

pub const HTTP = struct {
    pub const InitOpts = struct {
        minimum_request_size: usize = 16 << 20,
        max_retries: usize = 5,
        retry_initial_delay: std.Io.Duration = .fromMilliseconds(500),
        retry_max_delay: std.Io.Duration = .fromSeconds(30),
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
    minimum_request_size: usize,
    max_retries: usize,
    retry_initial_delay: std.Io.Duration,
    retry_max_delay: std.Io.Duration,
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
        range_read.assertValidOptions(opts.minimum_request_size, opts.retry_initial_delay, opts.retry_max_delay);

        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .protocol = protocol,
            .minimum_request_size = opts.minimum_request_size,
            .max_retries = opts.max_retries,
            .retry_initial_delay = opts.retry_initial_delay,
            .retry_max_delay = opts.retry_max_delay,
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
            .read_hints = .{
                .minimum_request_size = self.minimum_request_size,
                .high_latency = true,
            },
            .read_stats = .{ .userdata = self, .snapshotFn = readStatsSnapshot },
        };
    }

    fn readStatsSnapshot(userdata: *anyopaque) ReadStats {
        const self: *HTTP = @ptrCast(@alignCast(userdata));
        return self.read_stats.snapshot();
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
            .block_size = 1,
        };
    }

    fn dirAccess(_: ?*anyopaque, _: std.Io.Dir, _: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {}

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *HTTP = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));

        const size = self.fetchSize(dir, sub_path) catch return std.Io.File.OpenError.Unexpected;

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
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.ReadPositionalError.Unexpected;
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
        const read_size = range_read.readSize(handle.size, offset, data);
        if (read_size == 0) return 0;

        var url_buffer: [8 * 1024]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "{s}://{s}", .{ @tagName(self.protocol), handle.uri });
        const uri: std.Uri = try .parse(url);

        var attempt: usize = 0;
        while (true) {
            switch (try self.performReadAttempt(uri, url, data, offset, read_size, attempt == 0)) {
                .success => return read_size,
                .retry => |retry| {
                    self.read_stats.recordFailure(read_size, retry.failure);
                    if (attempt >= self.max_retries) return error.RetriesExhausted;

                    self.read_stats.recordRetry();
                    const delay = retry.delay orelse range_read.fullJitterDelay(
                        self.base.inner,
                        self.retry_initial_delay,
                        self.retry_max_delay,
                        attempt,
                    );
                    self.read_stats.recordRetryDelay(read_size, delay);
                    self.base.inner.sleep(delay, .awake) catch return error.RetriesExhausted;
                    attempt += 1;
                },
            }
        }
    }

    fn performReadAttempt(
        self: *HTTP,
        uri: std.Uri,
        url: []const u8,
        data: []const []u8,
        offset: u64,
        read_size: usize,
        include_timing_sample: bool,
    ) !range_read.AttemptResult {
        var range_buf: [64]u8 = undefined;
        const range_header = std.fmt.bufPrint(
            &range_buf,
            "bytes={d}-{d}",
            .{ offset, offset + @as(u64, @intCast(read_size - 1)) },
        ) catch unreachable;

        self.read_stats.recordAttempt(read_size);
        const attempt_started: std.Io.Timestamp = .now(self.base.inner, .awake);
        var req = self.client.request(.GET, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        }) catch |err| switch (err) {
            error.Timeout => {
                log.warn("Failed to connect: {}", .{err});
                return .{ .retry = .{ .failure = .timeout } };
            },
            error.ConnectionRefused,
            error.ConnectionResetByPeer,
            error.HostUnreachable,
            error.NetworkUnreachable,
            error.NetworkDown,
            error.NameServerFailure,
            => {
                log.warn("Failed to connect: {}", .{err});
                return .{ .retry = .{ .failure = .transient } };
            },
            else => {
                log.err("Failed to connect: {}", .{err});
                return err;
            },
        };
        defer req.deinit();

        req.sendBodiless() catch |err| switch (err) {
            error.WriteFailed => {
                log.warn("Failed to send headers: {}", .{err});
                return .{ .retry = .{ .failure = .transient } };
            },
        };

        var redirect_buffer: [8 * 1024]u8 = undefined;
        var res = req.receiveHead(&redirect_buffer) catch |err| switch (err) {
            error.Timeout => {
                log.warn("Failed to receive headers: {}", .{err});
                return .{ .retry = .{ .failure = .timeout } };
            },
            error.HttpConnectionClosing,
            error.HttpRequestTruncated,
            error.ReadFailed,
            error.WriteFailed,
            error.ConnectionRefused,
            error.ConnectionResetByPeer,
            error.HostUnreachable,
            error.NetworkUnreachable,
            error.NetworkDown,
            error.NameServerFailure,
            => {
                log.warn("Failed to receive headers: {}", .{err});
                return .{ .retry = .{ .failure = .transient } };
            },
            else => {
                log.err("Failed to receive headers: {}", .{err});
                return err;
            },
        };

        if (res.head.status != .partial_content and res.head.status != .ok) {
            const retry = retryForStatus(res.head.status) orelse {
                log.err("Failed to read {s}: {s}", .{ url, res.head.bytes });
                return error.RequestFailed;
            };
            log.warn("Failed to read {s}: {s}", .{ url, res.head.bytes });
            return .{ .retry = retry };
        }

        const content_range = blk: {
            var it = res.head.iterateHeaders();
            while (it.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) {
                    break :blk range_read.parseContentRange(header.value);
                }
            }
            break :blk null;
        };

        const timing = range_read.readResponse(
            self.base.inner,
            res.reader(&.{}),
            res.head.status,
            content_range,
            offset,
            data,
            read_size,
        ) catch |err| switch (err) {
            error.EndOfStream, error.ReadFailed => {
                log.warn("Failed to read from response: {}", .{err});
                return .{ .retry = .{ .failure = .transient } };
            },
            else => {
                log.err("Failed to read from response: {}", .{err});
                return err;
            },
        };
        self.read_stats.recordSuccess(
            read_size,
            timing.ttfbNanoseconds(attempt_started),
            timing.bodyNanoseconds(),
            include_timing_sample,
        );
        return .{ .success = timing };
    }

    fn retryForStatus(status: std.http.Status) ?range_read.Retry {
        return switch (status) {
            .request_timeout => .{ .failure = .timeout },
            .too_many_requests => .{ .failure = .throttle },
            else => if (status.class() == .server_error)
                .{ .failure = .server_failure }
            else
                null,
        };
    }
};

test "HTTP retry status classification is typed" {
    try std.testing.expectEqual(
        @import("base.zig").ReadFailure.timeout,
        HTTP.retryForStatus(.request_timeout).?.failure,
    );
    try std.testing.expectEqual(
        @import("base.zig").ReadFailure.throttle,
        HTTP.retryForStatus(.too_many_requests).?.failure,
    );
    try std.testing.expectEqual(
        @import("base.zig").ReadFailure.server_failure,
        HTTP.retryForStatus(.bad_gateway).?.failure,
    );
    try std.testing.expect(HTTP.retryForStatus(.not_found) == null);
}
