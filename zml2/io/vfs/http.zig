const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/http");

pub const HTTP = struct {
    const HandleId = i32;

    pub const FileHandle = struct {
        allocator: std.mem.Allocator,

        url: []const u8,
        pos: u64,
        size: u64,
        redirect_buffer: [std.fs.max_path_bytes]u8,
        request: ?*std.http.Client.Request,
        response: ?std.http.Client.Response,
        body_reader: ?*std.Io.Reader,

        total_requests: u64 = 0,
        total_latency_ns: u64 = 0,
        min_latency_ns: u64 = std.math.maxInt(u64),
        max_latency_ns: u64 = 0,

        pub fn init(allocator: std.mem.Allocator, parent_path: []const u8, sub_path: []const u8) std.mem.Allocator.Error!*FileHandle {
            const handle = try allocator.create(FileHandle);
            errdefer allocator.destroy(handle);

            handle.* = .{
                .allocator = allocator,
                .url = try resolveUri(allocator, parent_path, sub_path),
                .pos = 0,
                .size = undefined,
                .redirect_buffer = undefined,
                .request = null,
                .response = null,
                .body_reader = null,
            };

            log.info("Created HTTP FileHandle for URL: {s}", .{handle.url});

            return handle;
        }

        pub fn deinit(self: *FileHandle) void {
            self.releaseRequest();
            self.allocator.free(self.url);
            self.allocator.destroy(self);
        }

        pub fn releaseRequest(self: *FileHandle) void {
            if (self.request) |req| {
                req.deinit();
                self.allocator.destroy(req);
                self.request = null;
                self.response = null;
                self.body_reader = null;
            }
        }

        pub fn recordLatency(self: *FileHandle, latency_ns: u64) void {
            self.total_requests += 1;
            self.total_latency_ns += latency_ns;
            self.min_latency_ns = @min(self.min_latency_ns, latency_ns);
            self.max_latency_ns = @max(self.max_latency_ns, latency_ns);
        }

        pub fn avgLatencyMs(self: *FileHandle) u64 {
            if (self.total_requests == 0) return 0;
            return (self.total_latency_ns / self.total_requests) / std.time.ns_per_ms;
        }

        fn resolveUri(allocator: std.mem.Allocator, parent_path: []const u8, sub_path: []const u8) std.mem.Allocator.Error![]const u8 {
            if (std.mem.startsWith(u8, sub_path, "http://") or
                std.mem.startsWith(u8, sub_path, "https://"))
            {
                return try allocator.dupe(u8, sub_path);
            }

            const base_uri = std.Uri.parse(parent_path) catch unreachable;
            var aux_buf: [std.fs.max_path_bytes]u8 = undefined;
            // if (sub_path.len > aux_buf.len) return error.InvalidPath;
            @memcpy(aux_buf[0..sub_path.len], sub_path);
            var aux_slice: []u8 = aux_buf[0..];
            const resolved = std.Uri.resolveInPlace(base_uri, sub_path.len, &aux_slice) catch unreachable;
            return std.fmt.allocPrint(allocator, "{f}", .{resolved.fmt(.{
                .scheme = true,
                .authority = true,
                .path = true,
                .query = true,
                .fragment = true,
            })});
        }
    };

    pub const DirHandle = struct {
        allocator: std.mem.Allocator,
        url: []const u8,

        pub fn init(allocator: std.mem.Allocator, parent_path: []const u8, sub_path: []const u8) std.mem.Allocator.Error!*DirHandle {
            const self = try allocator.create(DirHandle);
            errdefer allocator.destroy(self);

            var url = try FileHandle.resolveUri(allocator, parent_path, sub_path);
            errdefer allocator.free(url);

            // Ensure directory URLs have a trailing slash.
            // According to RFC 3986, if the parent path lacks a trailing slash, it is treated as a file and the last segment is replaced.

            if (url.len == 0 or url[url.len - 1] != '/') {
                const with_slash = try std.fmt.allocPrint(allocator, "{s}/", .{url});
                allocator.free(url);
                url = with_slash;
            }

            log.info("Created HTTP DirHandle for URL: {s}", .{url});

            self.* = .{
                .allocator = allocator,
                .url = url,
            };

            return self;
        }

        pub fn deinit(self: *DirHandle) void {
            self.allocator.free(self.url);
            self.allocator.destroy(self);
        }
    };

    pub const Config = struct {
        max_fetch_attempts: u32 = 10,
        request_range_min: u64 = 8 * 1024,
        request_range_max: u64 = 128 * 1024 * 1024,
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex,

    file_handles: std.AutoHashMapUnmanaged(HandleId, *FileHandle),
    dir_handles: std.AutoHashMapUnmanaged(HandleId, *DirHandle),
    next_file_handle_id: HandleId,
    next_dir_handle_id: HandleId,

    client: *std.http.Client,
    config: Config,

    vtable: std.Io.VTable,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io, http_client: *std.http.Client, config: Config) std.mem.Allocator.Error!HTTP {
        return .{
            .allocator = allocator,
            .mutex = .init,
            .file_handles = .{},
            .dir_handles = .{},
            .next_file_handle_id = 0,
            .next_dir_handle_id = 0,
            .client = http_client,
            .config = config,
            .vtable = makeVTable(base_io),
        };
    }

    pub fn deinit(self: *HTTP) void {
        var file_it = self.file_handles.valueIterator();
        while (file_it.next()) |handle_ptr| {
            handle_ptr.*.deinit();
        }
        self.file_handles.deinit(self.allocator);

        var dir_it = self.dir_handles.valueIterator();
        while (dir_it.next()) |handle_ptr| {
            handle_ptr.*.deinit();
        }
        self.dir_handles.deinit(self.allocator);
    }

    pub fn io(self: *HTTP) std.Io {
        return .{
            .vtable = &self.vtable,
            .userdata = self,
        };
    }

    fn registerFileHandle(self: *HTTP, handle: *FileHandle) std.mem.Allocator.Error!HandleId {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_id = self.next_file_handle_id;
        self.next_file_handle_id += 1;

        try self.file_handles.put(self.allocator, handle_id, handle);

        return handle_id;
    }

    fn fileHandle(self: *HTTP, file: std.Io.File) ?*FileHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        return self.file_handles.get(file.handle);
    }

    fn registerDirHandle(self: *HTTP, handle: *DirHandle) std.mem.Allocator.Error!HandleId {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_id = self.next_dir_handle_id;
        self.next_dir_handle_id += 1;

        try self.dir_handles.put(self.allocator, handle_id, handle);

        return handle_id;
    }

    fn dirHandle(self: *HTTP, dir: std.Io.Dir) ?DirHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return .{
                .allocator = self.allocator,
                .url = "",
            };
        }

        return self.dir_handles.get(dir.handle).?.*;
    }

    fn openFile(
        self: *HTTP,
        dir: std.Io.Dir,
        sub_path: []const u8,
    ) std.Io.File.OpenError!std.Io.File {
        const dir_handle = self.dirHandle(dir) orelse {
            log.err("Directory handle not found for dirOpenFile with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.File.OpenError.FileNotFound;
        };

        const handle = FileHandle.init(self.allocator, dir_handle.url, sub_path) catch {
            log.err("Failed to create FileHandle in openFile with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.File.OpenError.SystemResources;
        };
        errdefer handle.deinit();

        const handle_id = self.registerFileHandle(handle) catch {
            log.err("Failed to register FileHandle in openFile with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.File.OpenError.SystemResources;
        };

        const file: std.Io.File = .{ .handle = @intCast(handle_id) };

        const stat = self.fetchFileStat(file) catch |err| switch (err) {
            error.AccessDenied, error.SystemResources => {
                log.err("File not found during openFile for handle={d} url={s}", .{ file.handle, handle.url });
                return std.Io.File.OpenError.FileNotFound;
            },
            else => {
                log.err("Failed to fetch file stat during openFile for handle={d} url={s}", .{ file.handle, handle.url });
                return std.Io.File.OpenError.Unexpected;
            },
        };

        handle.size = stat.size;

        return file;
    }

    fn openDir(
        self: *HTTP,
        dir: std.Io.Dir,
        sub_path: []const u8,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        const dir_handle = self.dirHandle(dir) orelse {
            log.err("Directory handle not found for dirOpenFile", .{});
            return std.Io.File.OpenError.FileNotFound;
        };

        const handle = DirHandle.init(self.allocator, dir_handle.url, sub_path) catch {
            log.err("Failed to create DirHandle in dirOpenFile", .{});
            return std.Io.Dir.OpenError.SystemResources;
        };
        errdefer handle.deinit();

        const handle_id = self.registerDirHandle(handle) catch return std.Io.Dir.OpenError.SystemResources;

        return .{ .handle = @intCast(handle_id) };
    }

    fn closeDir(self: *HTTP, dir: std.Io.Dir) void {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_kv = self.dir_handles.fetchRemove(dir.handle) orelse {
            log.warn("Attempted to close non-existent dir handle: {d}", .{dir.handle});
            return;
        };

        handle_kv.value.deinit();
    }

    fn closeFile(self: *HTTP, file: std.Io.File) void {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_kv = self.file_handles.fetchRemove(file.handle) orelse {
            log.warn("Attempted to close non-existent file handle: {d}", .{file.handle});
            return;
        };

        handle_kv.value.deinit();
    }

    fn setAbsolutePos(self: *HTTP, file: std.Io.File, pos: u64) std.Io.File.SeekError!void {
        const handle = self.fileHandle(file) orelse return std.Io.File.SeekError.Unexpected;
        handle.releaseRequest();
        handle.pos = pos;
    }

    fn setRelativeOffset(self: *HTTP, file: std.Io.File, offset: i64) std.Io.File.SeekError!void {
        const handle = self.fileHandle(file) orelse return std.Io.File.SeekError.Unexpected;
        handle.releaseRequest();

        const new_pos = @as(i64, @intCast(handle.pos)) + offset;
        handle.pos = @intCast(new_pos);
    }

    fn performPositionalRead(self: *HTTP, file: std.Io.File, data: [][]u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const handle = self.fileHandle(file) orelse return std.Io.File.ReadPositionalError.Unexpected;

        const saved_pos = handle.pos;

        if (handle.pos != offset) {
            handle.releaseRequest();
            handle.pos = offset;
        }

        const bytes_read = try self.performRead(file, data);

        handle.pos = saved_pos;
        handle.releaseRequest();

        return bytes_read;
    }

    fn performRead(self: *HTTP, file: std.Io.File, data: [][]u8) std.Io.File.Reader.Error!usize {
        const handle = self.fileHandle(file) orelse {
            log.err("Invalid file handle in HTTP VFS read", .{});
            return std.Io.File.Reader.Error.Unexpected;
        };

        if (data.len == 0) return 0;

        var total_read: usize = 0;

        for (data) |buf| {
            var offset: usize = 0;

            while (offset < buf.len) {
                // Check EOF against known file size
                if (handle.pos >= handle.size) {
                    return total_read;
                }

                const start = handle.pos;
                var desired_read_len: u64 = @max(@as(u64, @intCast(buf.len)), self.config.request_range_min);
                if (desired_read_len > self.config.request_range_max) desired_read_len = self.config.request_range_max;

                const end = if (handle.size > 0 and start < handle.size) @min(start + desired_read_len - 1, handle.size - 1) else start + desired_read_len - 1;

                try self.requestRead(file, start, end);
                const reader = handle.body_reader.?;

                const dest = buf[offset..];
                var iov = [_][]u8{dest};

                const n = reader.readVec(&iov) catch |err| switch (err) {
                    error.EndOfStream => {
                        log.debug("Reached EndOfStream for handle={d} pos={d}, will re-request remaining", .{ file.handle, handle.pos });
                        handle.releaseRequest();
                        continue;
                    },
                    else => {
                        return std.Io.File.Reader.Error.Unexpected;
                    },
                };

                offset += n;
                handle.pos += n;
                total_read += n;
            }
        }
        return total_read;
    }

    fn requestRead(self: *HTTP, file: std.Io.File, start: u64, end: u64) std.Io.File.Reader.Error!void {
        const handle = self.fileHandle(file) orelse return std.Io.File.Reader.Error.Unexpected;

        if (handle.request) |_| return;

        std.debug.assert(start <= end);

        var range_buf: [64]u8 = undefined;
        const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ start, end }) catch unreachable;

        var timer = std.time.Timer.start() catch {
            log.err("Failed to start timer for HTTP request", .{});
            return std.Io.File.Reader.Error.SystemResources;
        };

        const request = self.allocator.create(std.http.Client.Request) catch {
            log.err("Failed to allocate HTTP request", .{});
            return std.Io.File.Reader.Error.SystemResources;
        };
        errdefer self.allocator.destroy(request);

        const uri = std.Uri.parse(handle.url) catch |err| {
            log.err("Failed to parse URL during read request: {}", .{err});
            return std.Io.File.Reader.Error.Unexpected;
        };

        request.* = self.client.request(.GET, uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        }) catch |err| {
            log.err("Failed to create HTTP GET request: {}", .{err});
            return std.Io.File.Reader.Error.SystemResources;
        };
        errdefer request.deinit();

        request.sendBodiless() catch |err| {
            log.err("Failed to send HTTP GET request: {}", .{err});
            return std.Io.File.Reader.Error.SystemResources;
        };

        const response = request.receiveHead(&handle.redirect_buffer) catch |err| {
            log.err("Failed to receive HTTP response head for handle={d} url={s}: {}", .{ file.handle, handle.url, err });
            return std.Io.File.Reader.Error.Unexpected;
        };

        const latency = timer.read();
        handle.recordLatency(latency);

        log.debug("HTTP call: handle={d} status={s} content_length={?} range={d}-{d} ({d}) latency={d}ms avg={d}ms min={d}ms max={d}ms for {s}", .{
            file.handle,
            @tagName(response.head.status),
            response.head.content_length,
            start,
            end,
            end - start + 1,
            latency / std.time.ns_per_ms,
            handle.avgLatencyMs(),
            handle.min_latency_ns / std.time.ns_per_ms,
            handle.max_latency_ns / std.time.ns_per_ms,
            handle.url,
        });

        if (response.head.status != .partial_content and response.head.status != .ok) {
            request.deinit();
            self.allocator.destroy(request);
            return std.Io.File.Reader.Error.Unexpected;
        }

        handle.request = request;
        handle.response = response;
        handle.body_reader = handle.response.?.reader(&.{});
    }

    pub const FileStat = struct {
        inode: u64,
        size: u64,
    };

    fn fetchFileStat(self: *HTTP, file: std.Io.File) std.Io.File.StatError!FileStat {
        const handle = self.fileHandle(file) orelse return std.Io.File.StatError.Unexpected;

        var attempt: u32 = 0;
        while (attempt < self.config.max_fetch_attempts) : (attempt += 1) {
            const req = self.allocator.create(std.http.Client.Request) catch {
                log.err("Failed to allocate HTTP HEAD request", .{});
                return std.Io.File.StatError.SystemResources;
            };

            req.* = self.client.request(.HEAD, std.Uri.parse(handle.url) catch return std.Io.File.StatError.Unexpected, .{
                .headers = .{ .accept_encoding = .{ .override = "identity" } },
            }) catch {
                self.allocator.destroy(req);
                continue;
            };

            req.sendBodiless() catch {
                req.deinit();
                self.allocator.destroy(req);
                continue;
            };

            var response = req.receiveHead(&handle.redirect_buffer) catch {
                req.deinit();
                self.allocator.destroy(req);
                continue;
            };

            defer {
                req.deinit();
                self.allocator.destroy(req);
            }

            switch (response.head.status) {
                .moved_permanently, .found, .see_other, .temporary_redirect, .permanent_redirect => {
                    if (response.head.location) |location| {
                        var buf: [std.fs.max_path_bytes]u8 = undefined;

                        if (location.len > buf.len) return std.Io.File.StatError.Unexpected;
                        @memcpy(buf[0..location.len], location);

                        var aux_buf: []u8 = buf[0..];

                        const uri = std.Uri.parse(handle.url) catch |err| {
                            log.err("Failed to parse URL during redirect: {}", .{err});
                            return std.Io.File.StatError.Unexpected;
                        };

                        const resolved = std.Uri.resolveInPlace(uri, location.len, &aux_buf) catch |err| {
                            log.err("Failed to resolve redirect URL: {}", .{err});
                            return std.Io.File.StatError.Unexpected;
                        };
                        const new_url = std.fmt.allocPrint(self.allocator, "{f}", .{resolved.fmt(.{
                            .scheme = true,
                            .authority = true,
                            .path = true,
                            .query = true,
                            .fragment = true,
                        })}) catch |err| {
                            log.err("Failed to allocate new URL during redirect: {}", .{err});
                            return std.Io.File.StatError.SystemResources;
                        };

                        self.allocator.free(handle.url);
                        handle.url = new_url;

                        continue;
                    }
                },
                .ok, .partial_content => {
                    const size = response.head.content_length orelse {
                        log.err("HTTP stat response missing Content-Length", .{});
                        return std.Io.File.StatError.SystemResources;
                    };

                    return .{
                        .inode = @intCast(file.handle),
                        .size = size,
                    };
                },
                .internal_server_error, .bad_gateway, .service_unavailable, .gateway_timeout => {
                    std.Io.sleep(self.io(), .{ .nanoseconds = 50 * std.time.ns_per_ms }, .real) catch |err| {
                        log.err("Sleep failed during HTTP stat retry: {}", .{err});
                    };
                    continue;
                },
                else => {
                    log.err("HTTP stat failed with status: {s}", .{@tagName(response.head.status)});
                    return std.Io.File.StatError.Unexpected;
                },
            }
        }

        log.err("HTTP stat failed after {d} attempts", .{self.config.max_fetch_attempts});
        return std.Io.File.StatError.Unexpected;
    }

    fn makeVTable(base_io: std.Io) std.Io.VTable {
        var vtable = base_io.vtable.*;
        vtable.dirMake = dirMake;
        vtable.dirMakePath = dirMakePath;
        vtable.dirMakeOpenPath = dirMakeOpenPath;
        vtable.dirStat = dirStat;
        vtable.dirStatPath = dirStatPath;
        vtable.dirAccess = dirAccess;
        vtable.dirCreateFile = dirCreateFile;
        vtable.dirOpenFile = dirOpenFile;
        vtable.dirOpenDir = dirOpenDir;
        vtable.dirClose = dirClose;
        vtable.fileStat = fileStat;
        vtable.fileClose = fileClose;
        vtable.fileWriteStreaming = fileWriteStreaming;
        vtable.fileWritePositional = fileWritePositional;
        vtable.fileReadStreaming = fileReadStreaming;
        vtable.fileReadPositional = fileReadPositional;
        vtable.fileSeekBy = fileSeekBy;
        vtable.fileSeekTo = fileSeekTo;
        vtable.openSelfExe = openSelfExe;

        return vtable;
    }

    fn dirMake(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        _ = userdata;
        _ = dir;
        _ = sub_path;
        _ = mode;
        log.err("HTTP VFS is read-only, dirMake not supported", .{});
        return std.Io.Dir.MakeError.ReadOnlyFileSystem;
    }

    fn dirMakePath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        mode: std.Io.Dir.Mode,
    ) std.Io.Dir.MakeError!void {
        _ = userdata;
        _ = dir;
        _ = sub_path;
        _ = mode;
        log.err("HTTP VFS is read-only, dirMakePath not supported", .{});
        return std.Io.Dir.MakeError.ReadOnlyFileSystem;
    }

    fn dirMakeOpenPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.MakeOpenPathError!std.Io.Dir {
        _ = userdata;
        _ = dir;
        _ = sub_path;
        _ = options;
        log.err("HTTP VFS is read-only, dirMakeOpenPath not supported", .{});
        return std.Io.Dir.MakeOpenPathError.Unexpected;
    }

    fn dirStat(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
    ) std.Io.Dir.StatError!std.Io.Dir.Stat {
        _ = userdata;
        _ = dir;
        log.err("HTTP VFS does not support dirStat", .{});
        return std.Io.Dir.StatError.AccessDenied;
    }

    fn dirStatPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.StatPathOptions,
    ) std.Io.Dir.StatPathError!std.Io.File.Stat {
        _ = userdata;
        _ = dir;
        _ = sub_path;
        _ = options;
        @panic("TODO: implement HTTP VFS dirStatPath");
    }

    fn dirAccess(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.AccessOptions,
    ) std.Io.Dir.AccessError!void {
        _ = userdata;
        _ = dir;
        _ = sub_path;

        if (options.write) {
            return std.Io.Dir.AccessError.ReadOnlyFileSystem;
        }

        log.err("HTTP VFS does not support dirAccess", .{});
        return std.Io.Dir.AccessError.AccessDenied;
    }

    fn dirCreateFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.CreateFlags,
    ) std.Io.File.OpenError!std.Io.File {
        _ = userdata;
        _ = dir;
        _ = sub_path;
        _ = flags;
        log.err("HTTP VFS is read-only, dirCreateFile not supported", .{});
        return std.Io.File.OpenError.Unexpected;
    }

    fn dirOpenFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenError!std.Io.File {
        _ = flags;
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.openFile(dir, sub_path);
    }

    fn dirOpenDir(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        _ = options;
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.openDir(dir, sub_path);
    }

    fn dirClose(userdata: ?*anyopaque, dir: std.Io.Dir) void {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        self.closeDir(dir);
    }

    fn fileStat(
        userdata: ?*anyopaque,
        file: std.Io.File,
    ) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        const stat = try self.fetchFileStat(file);

        return .{
            .inode = stat.inode,
            .kind = .file,
            .size = stat.size,
            .mode = 0o444,
            .atime = .{ .nanoseconds = 0 },
            .mtime = .{ .nanoseconds = 0 },
            .ctime = .{ .nanoseconds = 0 },
        };
    }

    fn fileClose(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        self.closeFile(file);
    }

    fn fileWriteStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]const u8,
    ) std.Io.File.WriteStreamingError!usize {
        _ = userdata;
        _ = file;
        _ = buffer;
        log.err("HTTP VFS is read-only, fileWriteStreaming not supported", .{});
        return std.Io.File.WriteStreamingError.Unexpected;
    }

    fn fileWritePositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]const u8,
        offset: u64,
    ) std.Io.File.WritePositionalError!usize {
        _ = userdata;
        _ = file;
        _ = buffer;
        _ = offset;
        log.err("HTTP VFS is read-only, fileWritePositional not supported", .{});
        return std.Io.File.WritePositionalError.Unexpected;
    }

    fn fileReadStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
    ) std.Io.File.Reader.Error!usize {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.performRead(file, data);
    }

    fn fileReadPositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
        offset: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.performPositionalRead(file, data, offset);
    }

    fn fileSeekBy(
        userdata: ?*anyopaque,
        file: std.Io.File,
        relative_offset: i64,
    ) std.Io.File.SeekError!void {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        try self.setRelativeOffset(file, relative_offset);
    }

    fn fileSeekTo(
        userdata: ?*anyopaque,
        file: std.Io.File,
        absolute_offset: u64,
    ) std.Io.File.SeekError!void {
        const self: *HTTP = @ptrCast(@alignCast(userdata orelse unreachable));
        try self.setAbsolutePos(file, absolute_offset);
    }

    fn openSelfExe(
        userdata: ?*anyopaque,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenSelfExeError!std.Io.File {
        _ = userdata;
        _ = flags;
        log.err("HTTP VFS does not support openSelfExe", .{});
        return std.Io.File.OpenSelfExeError.NotSupported;
    }
};
