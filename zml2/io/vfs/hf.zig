const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/hf");

// todo : refactor this code to align with http vfs

pub const HF = struct {
    pub const Auth = union(enum) {
        none,
        hf_token: []const u8,

        pub const AuthError = error{
            MissingHFToken,
            InvalidHFToken,
            MissingHomePath,
            MissingConfigFile,
        } || std.process.GetEnvVarOwnedError || std.fs.File.OpenError || std.fs.File.Reader.Error || std.Io.Reader.ReadAllocError || std.mem.Allocator.Error;

        pub fn deinit(self: *Auth, allocator: std.mem.Allocator) void {
            switch (self.*) {
                .none => {},
                .hf_token => |token| {
                    allocator.free(token);
                },
            }
        }

        pub fn auto(allocator: std.mem.Allocator, io_: std.Io) AuthError!Auth {
            return fromHomeConfig(allocator, io_) catch |err| switch (err) {
                AuthError.MissingHomePath => return fromEnv(allocator),
                AuthError.MissingConfigFile => return fromEnv(allocator),
                else => |e| return e,
            };
        }

        pub fn fromEnv(allocator: std.mem.Allocator) AuthError!Auth {
            const token = std.process.getEnvVarOwned(allocator, "HF_TOKEN") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return AuthError.MissingHFToken,
                else => |e| return e,
            };
            defer allocator.free(token);

            return .{ .hf_token = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token}) };
        }

        pub fn fromHomeConfig(allocator: std.mem.Allocator, io_: std.Io) AuthError!Auth {
            const home_path = std.process.getEnvVarOwned(allocator, "HOME") catch |err| switch (err) {
                error.EnvironmentVariableNotFound => return AuthError.MissingHomePath,
                else => |e| return e,
            };
            defer allocator.free(home_path);

            const token_path = try std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/token", .{home_path});
            defer allocator.free(token_path);

            const file = std.Io.File.openAbsolute(io_, token_path, .{ .mode = .read_only }) catch |err| switch (err) {
                error.FileNotFound => return AuthError.MissingConfigFile,
                else => |e| return e,
            };

            var reader = file.reader(io_, &.{});
            const size = reader.getSize() catch unreachable;

            const token = try reader.interface.readAlloc(allocator, size);
            defer allocator.free(token);

            return .{ .hf_token = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token}) };
        }
    };

    const Config = struct {
        auth: Auth = .none,
        range_size: u64 = 128 * 1024 * 1024, // todo: rename field
    };

    allocator: std.mem.Allocator,
    client: *std.http.Client, // todo: rename http_client
    http_headers: []std.http.Header,
    config: Config,

    file_handles: std.AutoHashMapUnmanaged(HandleId, *FileHandle),
    dir_handles: std.AutoHashMapUnmanaged(HandleId, *DirHandle),
    next_handle_id: HandleId,
    mutex: std.Io.Mutex,
    vtable: std.Io.VTable,

    pub const HandleId = i32;
    const max_retries = 10;

    pub const FileHandle = struct {
        id: HandleId,
        allocator: std.mem.Allocator,
        hf_path: []const u8,
        pos: u64,
        built_url: []const u8,
        size: ?u64,
        redirect_buffer: [1024 * 1024]u8,
        request: ?*std.http.Client.Request,
        response: ?std.http.Client.Response,
        body_reader: ?*std.Io.Reader,
        response_bytes_remaining: u64,

        pub fn init(allocator: std.mem.Allocator, id: HandleId, hf_path: []const u8) !*FileHandle {
            const handle = try allocator.create(FileHandle);
            handle.* = .{
                .id = id,
                .allocator = allocator,
                .hf_path = try allocator.dupe(u8, hf_path),
                .pos = 0,
                .built_url = "",
                .size = null,
                .redirect_buffer = undefined,
                .request = null,
                .response = null,
                .body_reader = null,
                .response_bytes_remaining = 0,
            };
            return handle;
        }

        pub fn deinit(self: *FileHandle) void {
            self.releaseRequest();
            self.allocator.free(self.hf_path);
            self.allocator.free(self.built_url);
            self.allocator.destroy(self);
        }

        pub fn releaseRequest(self: *FileHandle) void {
            if (self.request) |req| {
                req.deinit();
                self.allocator.destroy(req);
                self.request = null;
                self.response = null;
                self.body_reader = null;
                self.response_bytes_remaining = 0;
            }
        }
    };

    pub const DirHandle = struct {
        id: HandleId,
        allocator: std.mem.Allocator,
        hf_path: []const u8,
        built_url: []const u8,
        redirect_buffer: []u8,
        request: ?*std.http.Client.Request,
        response: ?std.http.Client.Response,
        body_reader: ?*std.Io.Reader,

        pub fn init(allocator: std.mem.Allocator, id: HandleId, url: []const u8) std.mem.Allocator.Error!*DirHandle {
            const self = try allocator.create(DirHandle);
            errdefer allocator.destroy(self);

            const redirect_buffer = try allocator.alloc(u8, 1024 * 1024);
            errdefer allocator.free(redirect_buffer);

            const owned_path = try allocator.dupe(u8, url);
            errdefer allocator.free(owned_path);

            self.* = .{
                .id = id,
                .allocator = allocator,
                .hf_path = owned_path,
                .built_url = "",
                .redirect_buffer = redirect_buffer,
                .request = null,
                .response = null,
                .body_reader = null,
            };

            return self;
        }

        pub fn deinit(self: *DirHandle) void {
            self.allocator.free(self.hf_path);
            self.allocator.free(self.built_url);
            self.allocator.free(self.redirect_buffer);
            self.allocator.destroy(self);
        }
    };

    pub const InitArgs = struct {
        allocator: std.mem.Allocator,
        io: std.Io,
        http_client: *std.http.Client,
        config: Config,
    };

    pub fn init(args: InitArgs) !HF {
        var self: HF = .{
            .allocator = args.allocator,
            .client = args.http_client,
            .http_headers = try args.allocator.alloc(std.http.Header, 2),
            .config = args.config,
            .file_handles = .{},
            .dir_handles = .{},
            .next_handle_id = 1,
            .mutex = .init,
            .vtable = makeVTable(args.io),
        };

        switch (self.config.auth) {
            .hf_token => |token| {
                self.http_headers[0] = .{ .name = "Authorization", .value = token };
            },
            else => {},
        }

        return self;
    }

    pub fn deinit(self: *HF) void {
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
        self.allocator.free(self.http_headers);
    }

    pub fn io(self: *HF) std.Io {
        return .{
            .vtable = &self.vtable,
            .userdata = self,
        };
    }

    fn allocateHandleId(self: *HF) HandleId {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const id = self.next_handle_id;
        self.next_handle_id += 1;
        return id;
    }

    fn registerFileHandle(self: *HF, handle: *FileHandle) std.mem.Allocator.Error!void {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        try self.file_handles.put(self.allocator, handle.id, handle);
    }

    fn unregisterFileHandle(self: *HF, id: HandleId) ?*FileHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const kv = self.file_handles.fetchRemove(id) orelse return null;
        return kv.value;
    }

    fn getFileHandle(self: *HF, id: HandleId) ?*FileHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        return self.file_handles.get(id);
    }

    fn registerDirHandle(self: *HF, handle: *DirHandle) std.mem.Allocator.Error!void {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        try self.dir_handles.put(self.allocator, handle.id, handle);
    }

    fn unregisterDirHandle(self: *HF, id: HandleId) ?*DirHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const kv = self.dir_handles.fetchRemove(id) orelse return null;
        return kv.value;
    }

    fn getDirHandle(self: *HF, id: HandleId) ?*DirHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());
        return self.dir_handles.get(id);
    }

    fn resolvePath(self: *HF, dir_handle: std.Io.Dir.Handle, sub_path: []const u8) ![]const u8 {
        if (std.mem.startsWith(u8, sub_path, "hf://")) {
            return try self.allocator.dupe(u8, sub_path);
        }

        const dir = self.getDirHandle(dir_handle) orelse {
            return error.DirectoryNotfound;
        };

        return joinPath(self.allocator, dir.hf_path, sub_path);
    }

    pub fn joinPath(allocator: std.mem.Allocator, base: []const u8, path: []const u8) ![]const u8 {
        if (base.len == 0) {
            return try allocator.dupe(u8, path);
        }
        if (path.len == 0) {
            return try allocator.dupe(u8, base);
        }

        const base_has_slash = base[base.len - 1] == '/';
        const path_has_slash = path[0] == '/';

        if (base_has_slash and path_has_slash) {
            return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base, path[1..] });
        } else if (base_has_slash or path_has_slash) {
            return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base, path });
        } else {
            return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base, path });
        }
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
        return error.Unexpected;
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
        return error.Unexpected;
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
        return error.Unexpected;
    }

    pub fn updateBuiltUrl(self: *HF, url_ptr: *[]const u8, new_url: []const u8) !void {
        const owned_url = try self.allocator.dupe(u8, new_url);
        errdefer self.allocator.free(owned_url);

        self.allocator.free(url_ptr.*);
        url_ptr.* = owned_url;
    }

    fn dirStat(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
    ) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.getDirHandle(@intCast(dir.handle)) orelse return error.Unexpected;
        const stat_result = self.statDir(handle) catch return error.Unexpected;

        return .{
            .inode = @intCast(handle.id),
            .kind = .directory,
            .size = stat_result.size,
            .mode = 0o555,
            .atime = .{ .nanoseconds = 0 },
            .mtime = .{ .nanoseconds = 0 },
            .ctime = .{ .nanoseconds = 0 },
        };
    }

    fn statDir(self: *HF, handle: *DirHandle) !StatResult {
        const path = handle.hf_path[5..];
        var path_it = std.mem.splitSequence(u8, path, "/");
        const namespace = path_it.next() orelse return error.Unexpected;
        const repo = path_it.next() orelse return error.Unexpected;
        const rev = "main";
        // Build API URL
        var url_buf: [512]u8 = undefined;
        const url = std.fmt.bufPrint(
            &url_buf,
            "https://huggingface.co/api/models/{s}/{s}/treesize/{s}/",
            .{ namespace, repo, rev },
        ) catch return error.SystemResources;

        self.updateBuiltUrl(&handle.built_url, url) catch return error.SystemResources;

        return try self.fetchStatDir(handle);
    }

    pub const StatResult = struct {
        size: u64,
    };

    fn fetchStatDir(self: *HF, handle: *DirHandle) !StatResult {
        var attempt: u32 = 0;
        while (attempt < max_retries) : (attempt += 1) {
            const req = self.allocator.create(std.http.Client.Request) catch continue;

            req.* = self.client.request(.GET, std.Uri.parse(handle.built_url) catch return error.Unexpected, .{
                .headers = .{ .accept_encoding = .{ .override = "identity" } },
                .extra_headers = self.http_headers[0..1],
            }) catch {
                self.allocator.destroy(req);
                continue;
            };

            req.sendBodiless() catch {
                req.deinit();
                self.allocator.destroy(req);
                continue;
            };

            var response = req.receiveHead(handle.redirect_buffer) catch {
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
                        var buf: [4096]u8 = undefined;
                        if (location.len > buf.len) return error.InvalidRedirect;
                        @memcpy(buf[0..location.len], location);

                        var aux_buf: []u8 = buf[0..];
                        const resolved = try std.Uri.resolveInPlace(try std.Uri.parse(handle.built_url), location.len, &aux_buf);
                        const new_url = try std.fmt.allocPrint(self.allocator, "{f}", .{resolved.fmt(.{
                            .scheme = true,
                            .authority = true,
                            .path = true,
                            .query = true,
                            .fragment = true,
                        })});
                        self.allocator.free(handle.built_url);
                        handle.built_url = new_url;

                        continue;
                    }
                },
                .ok => {
                    var body_buf: [256]u8 = undefined;
                    var body_reader = response.reader(&.{});
                    var iov = [_][]u8{&body_buf};
                    const body_len = try body_reader.readVec(&iov);
                    const body = body_buf[0..body_len];

                    var parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, body, .{});
                    defer parsed.deinit();

                    const size_value = parsed.value.object.get("size") orelse return error.InvalidFormat;
                    const size = size_value.integer;

                    return .{ .size = @intCast(size) };
                },
                .internal_server_error, .bad_gateway, .service_unavailable, .gateway_timeout => {
                    continue;
                },
                else => {
                    log.err("HTTP stat dir failed with status: {s} for {s}", .{ @tagName(response.head.status), handle.built_url });
                    return error.HttpRequestFailed;
                },
            }
        }

        return error.Unexpected;
    }

    fn dirStatPath(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.StatPathOptions,
    ) std.Io.Dir.StatPathError!std.Io.File.Stat {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        _ = options;

        const file = dirOpenFile(userdata, dir, sub_path, .{}) catch return std.Io.Dir.StatPathError.BadPathName;

        const handle = self.getFileHandle(@intCast(file.handle)) orelse return error.Unexpected;
        defer fileClose(userdata, file);

        const stat_result = self.fetchStat(handle) catch return error.Unexpected;

        return .{
            .inode = @intCast(file.handle),
            .kind = .file,
            .size = stat_result.size,
            .mode = 0o444,
            .atime = .{ .nanoseconds = 0 },
            .mtime = .{ .nanoseconds = 0 },
            .ctime = .{ .nanoseconds = 0 },
        };
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

        // todo: check
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
        return std.Io.File.OpenError.Unexpected;
    }

    fn dirOpenFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenError!std.Io.File {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        _ = flags;

        const path = self.resolvePath(@intCast(dir.handle), sub_path) catch return error.SystemResources;
        defer self.allocator.free(path);

        return try self.openFile(path);
    }

    fn openFile(self: *HF, path: []const u8) std.Io.File.OpenError!std.Io.File {
        const url = hfPathToHttps(self.allocator, path) catch return error.SystemResources;
        defer self.allocator.free(url);
        const id = self.allocateHandleId();

        // Check if it's a valid file
        var handle = FileHandle.init(self.allocator, id, path) catch return error.SystemResources;
        errdefer handle.deinit();
        self.updateBuiltUrl(&handle.built_url, url) catch return error.SystemResources;

        _ = self.fetchStat(handle) catch return error.FileNotFound;

        // URL ownership transferred to handle

        self.registerFileHandle(handle) catch return error.SystemResources;

        return .{ .handle = @intCast(id) };
    }

    fn dirOpenDir(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        _ = options;
        const path = self.resolvePath(@intCast(dir.handle), sub_path) catch return error.Unexpected;
        defer self.allocator.free(path);

        return try self.openDir(path);
    }

    fn openDir(self: *HF, path: []const u8) std.Io.Dir.OpenError!std.Io.Dir {
        const url = hfPathToHttps(self.allocator, path) catch return error.SystemResources;
        defer self.allocator.free(url);
        // Ensure URL ends with /
        const dir_url = if (url.len > 0 and url[url.len - 1] != '/') blk: {
            const new_url = std.fmt.allocPrint(self.allocator, "{s}/", .{url}) catch return error.Unexpected;
            self.allocator.free(url);
            break :blk new_url;
        } else url;
        errdefer if (dir_url.ptr != url.ptr) self.allocator.free(dir_url);

        const id = self.allocateHandleId();
        const handle = DirHandle.init(self.allocator, id, path) catch return error.Unexpected;

        // Ensure it's a correct directory
        const stat = self.statDir(handle) catch return error.NotDir;
        _ = stat;

        errdefer handle.deinit();

        self.registerDirHandle(handle) catch return error.Unexpected;

        return .{ .handle = @intCast(id) };
    }

    fn dirClose(userdata: ?*anyopaque, dir: std.Io.Dir) void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.unregisterDirHandle(@intCast(dir.handle)) orelse return;
        handle.deinit();
    }

    fn fileStat(
        userdata: ?*anyopaque,
        file: std.Io.File,
    ) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.getFileHandle(@intCast(file.handle)) orelse return error.Unexpected;

        // call self.httpengine.fetchStat(handle)
        // log.debug("Fetching stat for file handle {d}", handle.id);
        const stat_result = self.fetchStat(handle) catch return error.Unexpected;

        return .{
            .inode = @intCast(file.handle),
            .kind = .file,
            .size = stat_result.size,
            .mode = 0o444,
            .atime = .{ .nanoseconds = 0 },
            .mtime = .{ .nanoseconds = 0 },
            .ctime = .{ .nanoseconds = 0 },
        };
    }

    pub fn fetchStat(self: *HF, handle: *FileHandle) !StatResult {
        var attempt: u32 = 0;
        while (attempt < max_retries) : (attempt += 1) {
            const req = self.allocator.create(std.http.Client.Request) catch continue;

            req.* = self.client.request(.HEAD, std.Uri.parse(handle.built_url) catch return error.Unexpected, .{
                .headers = .{ .accept_encoding = .{ .override = "identity" } },
                .extra_headers = self.http_headers[0..1],
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
                        var buf: [4096]u8 = undefined;
                        if (location.len > buf.len) return error.InvalidRedirect;
                        @memcpy(buf[0..location.len], location);

                        var aux_buf: []u8 = buf[0..];
                        const resolved = try std.Uri.resolveInPlace(try std.Uri.parse(handle.built_url), location.len, &aux_buf);
                        const new_url = try std.fmt.allocPrint(self.allocator, "{f}", .{resolved.fmt(.{
                            .scheme = true,
                            .authority = true,
                            .path = true,
                            .query = true,
                            .fragment = true,
                        })});
                        self.allocator.free(handle.built_url);
                        handle.built_url = new_url;

                        continue;
                    }
                },
                .ok, .partial_content => {
                    const size = response.head.content_length orelse return error.MissingContentLength;
                    handle.size = size;

                    return .{ .size = size };
                },
                .internal_server_error, .bad_gateway, .service_unavailable, .gateway_timeout => {
                    continue;
                },
                else => {
                    log.err("HTTP stat failed with status: {s} - {s}", .{ @tagName(response.head.status), handle.built_url });
                    return error.HttpRequestFailed;
                },
            }
        }

        return error.Unexpected;
    }

    fn fileClose(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.unregisterFileHandle(@intCast(file.handle)) orelse return;
        handle.deinit();
    }

    fn fileWriteStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        buffer: [][]const u8,
    ) std.Io.File.WriteStreamingError!usize {
        _ = userdata;
        _ = file;
        _ = buffer;
        return error.Unexpected;
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
        return error.Unexpected;
    }

    fn fileReadStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
    ) std.Io.File.Reader.Error!usize {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.getFileHandle(@intCast(file.handle)) orelse return error.Unexpected;
        return self.performRead(handle, data, null) catch return error.Unexpected;
    }

    fn fileReadPositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
        offset: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.getFileHandle(@intCast(file.handle)) orelse return error.Unexpected;
        return self.performRead(handle, data, offset) catch return error.Unexpected;
    }

    fn fileSeekBy(
        userdata: ?*anyopaque,
        file: std.Io.File,
        relative_offset: i64,
    ) std.Io.File.SeekError!void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.getFileHandle(@intCast(file.handle)) orelse return error.Unexpected;

        seekBy(handle, relative_offset) catch return error.Unexpected;
    }

    fn fileSeekTo(
        userdata: ?*anyopaque,
        file: std.Io.File,
        absolute_offset: u64,
    ) std.Io.File.SeekError!void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.getFileHandle(@intCast(file.handle)) orelse return error.Unexpected;

        seekTo(handle, absolute_offset) catch return error.Unexpected;
    }

    pub fn seekBy(handle: *FileHandle, relative_offset: i64) !void {
        handle.releaseRequest();
        if (relative_offset >= 0) {
            handle.pos +|= @intCast(relative_offset);
        } else {
            const abs_offset: u64 = @intCast(-relative_offset);
            if (abs_offset > handle.pos) {
                handle.pos = 0;
            } else {
                handle.pos -= abs_offset;
            }
        }
    }

    // Go back to VFS
    pub fn seekTo(handle: *FileHandle, absolute_offset: u64) !void {
        handle.releaseRequest();
        handle.pos = absolute_offset;
    }

    fn openSelfExe(
        userdata: ?*anyopaque,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenSelfExeError!std.Io.File {
        _ = userdata;
        _ = flags;
        return error.NotSupported;
    }

    fn hfPathToHttps(allocator: std.mem.Allocator, hf_path: []const u8) ![]const u8 {
        // Remove the "hf://" prefix
        const path = hf_path[5..hf_path.len];
        var path_it = std.mem.splitSequence(u8, path, "/");
        const namespace = path_it.next() orelse return error.InvalidPath;
        const repo = path_it.next() orelse return error.InvalidPath;
        const rev = "main";
        var file: []const u8 = "";
        if (path_it.next()) |file_path| {
            file = file_path;
        }

        var final_url_buf: [std.fs.max_path_bytes]u8 = undefined;

        const final_url = try std.fmt.bufPrint(
            &final_url_buf,
            "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
            .{ namespace, repo, rev, file },
        );

        return try allocator.dupe(u8, final_url);
    }

    fn ensureConnection(self: *HF, handle: *FileHandle) !void {
        if (handle.body_reader != null) return;

        if (handle.size == null) {
            const stat = try self.fetchStat(handle);
            handle.size = stat.size;
        }

        const size = handle.size.?;

        if (handle.pos >= size) return;

        const start = handle.pos;
        const end = @min(start + self.config.range_size, size) - 1;

        var range_buf: [64]u8 = undefined;
        const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ start, end }) catch unreachable;

        const request = try self.allocator.create(std.http.Client.Request);
        errdefer self.allocator.destroy(request);

        const uri = try std.Uri.parse(handle.built_url);

        self.http_headers[1] = .{ .name = "Range", .value = range_header };

        request.* = try self.client.request(.GET, uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = self.http_headers[0..2],
        });
        // errdefer request.deinit();

        try request.sendBodiless();

        const response = try request.receiveHead(&handle.redirect_buffer);

        if (response.head.status != .partial_content and response.head.status != .ok) {
            request.deinit();
            self.allocator.destroy(request);
            return error.Unexpected;
        }

        handle.request = request;
        handle.response = response;
        handle.body_reader = handle.response.?.reader(&.{});
    }

    fn doRead(self: *HF, handle: *FileHandle, data: [][]u8) !usize {
        if (data.len == 0) return 0;

        var total_read: usize = 0;

        for (data) |buf| {
            var offset: usize = 0;

            while (offset < buf.len) {
                // Check EOF against known file size
                if (handle.size) |size| {
                    if (handle.pos >= size) {
                        return total_read;
                    }
                }

                try self.ensureConnection(handle);
                const reader = handle.body_reader orelse {
                    return total_read;
                };

                // Read into remaining buffer
                const dest = buf[offset..];
                var iov = [_][]u8{dest};

                const n = reader.readVec(&iov) catch |err| switch (err) {
                    error.EndOfStream => {
                        // Current HTTP range exhausted, need new request
                        handle.releaseRequest();
                        continue;
                    },
                    else => {
                        return err;
                    },
                };

                // Data not ready yet, just retry same read
                if (n == 0) {
                    continue;
                }

                offset += n;
                handle.pos += n;
                total_read += n;
            }
        }

        return total_read;
    }

    fn performRead(self: *HF, handle: *FileHandle, data: [][]u8, offset: ?u64) !usize {
        const saved_pos = handle.pos;
        const is_positional = offset != null;

        if (offset) |o| {
            if (handle.pos != o) {
                handle.releaseRequest();
                handle.pos = o;
            }
        }

        const bytes_read = try self.doRead(handle, data);

        if (is_positional) {
            handle.pos = saved_pos;
            handle.releaseRequest();
        }

        return bytes_read;
    }
};
