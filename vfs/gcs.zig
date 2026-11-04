const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const range_read = @import("range_read.zig");
const request = @import("request.zig");
const VFSBase = @import("base.zig").VFSBase;
const Backend = @import("base.zig").Backend;

const log = std.log.scoped(.@"zml/vfs/gcs");

const MetadataUrl = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
const TokenUrl = "https://oauth2.googleapis.com/token";

/// Check if running on Google Compute Engine / Cloud Run host infrastructure,
/// because metadata server probing should only happen on GCP.
/// Mirrors the TPU platform detection approach.
fn isOnGCP(io: std.Io) !bool {
    if (builtin.os.tag != .linux) return false;

    const GoogleComputeEngine = "Google Compute Engine";
    var buffer: [GoogleComputeEngine.len]u8 = undefined;
    return std.mem.eql(
        u8,
        GoogleComputeEngine,
        try std.Io.Dir.readFile(
            .cwd(),
            io,
            "/sys/devices/virtual/dmi/id/product_name",
            &buffer,
        ),
    );
}

const OAuthToken = struct {
    // https://developers.google.com/identity/protocols/oauth2#size
    const Size = 2048;

    access_token: []const u8,
    expires_in: u64,
};

const Credentials = union(enum) {
    const AuthorizedUser = struct {
        type: []const u8,
        client_id: []const u8,
        client_secret: []const u8,
        refresh_token: []const u8,
        quota_project_id: ?[]const u8 = null,
        account: []const u8,
        universe_domain: []const u8,
    };

    const ServiceAccount = struct {
        type: []const u8,
        client_email: []const u8,
        private_key: []const u8,
        token_uri: []const u8,
        quota_project_id: ?[]const u8 = null,
    };

    authorized_user: AuthorizedUser,
    service_account: ServiceAccount,
    metadata_server: void,

    pub fn jsonParse(allocator: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) std.json.ParseError(@TypeOf(source.*))!Credentials {
        return jsonParseFromValue(
            allocator,
            try std.json.innerParse(std.json.Value, allocator, source, options),
            options,
        );
    }

    pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) std.json.ParseFromValueError!Credentials {
        return switch (source) {
            .object => |obj| blk: {
                const obj_type = obj.get("type") orelse return std.json.ParseFromValueError.UnexpectedToken;
                if (std.ascii.eqlIgnoreCase(obj_type.string, "authorized_user")) {
                    break :blk .{
                        .authorized_user = try std.json.parseFromValueLeaky(
                            AuthorizedUser,
                            allocator,
                            source,
                            options,
                        ),
                    };
                } else if (std.ascii.eqlIgnoreCase(obj_type.string, "service_account")) {
                    break :blk .{
                        .service_account = try std.json.parseFromValueLeaky(
                            ServiceAccount,
                            allocator,
                            source,
                            options,
                        ),
                    };
                }
                log.warn("Unknown credentials type: {s}", .{obj_type.string});
                return std.json.ParseFromValueError.InvalidEnumTag;
            },
            else => std.json.ParseFromValueError.UnexpectedToken,
        };
    }
};

const ReadState = struct { index: usize, objects: [][]const u8 };

pub const GCS = struct {
    const authorization_header_size = "Bearer ".len + OAuthToken.Size;

    const Token = struct {
        /// Guards the token. A refresh is a governed request that may wait
        /// out a hold or retry for minutes, so it runs under this lock, not
        /// under `GCS.mutex`, which every handle lookup takes.
        mutex: std.Io.Mutex = .init,
        header_storage: []u8,
        header_len: usize = 0,
        expires_at: std.Io.Timestamp,
        /// Refreshes that failed other than by a cancel. A reader that
        /// queued behind one fails with it instead of running its own.
        failures: std.atomic.Value(u64) = .init(0),

        fn expired(self: *const Token, io_: std.Io) bool {
            const now: std.Io.Timestamp = .now(io_, .real);
            return now.toSeconds() >= self.expires_at.toSeconds();
        }

        fn header(self: *const Token) []const u8 {
            return self.header_storage[0..self.header_len];
        }
    };

    pub const Config = struct {
        credentials: ?Credentials = null,
        endpoint_uri: std.Uri,
        /// Where tokens come from; a test points them at a local server.
        metadata_uri: std.Uri = std.Uri.parse(MetadataUrl) catch unreachable,
        token_uri: std.Uri = std.Uri.parse(TokenUrl) catch unreachable,
    };

    const Handle = struct {
        pub const Type = enum { file, directory };

        type: Type,
        uri: []const u8,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, type_: Type, path: []const u8, size: u64) !Handle {
            return .{
                .type = type_,
                .uri = try allocator.dupe(u8, path),
                .pos = 0,
                .size = size,
            };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.uri);
        }
    };

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    mutex: std.Io.Mutex = .init,
    config: Config,
    token: Token,
    governor: request.Governor,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .empty,
    dir_read_states: std.AutoHashMapUnmanaged(*std.Io.Dir.Reader, ReadState) = .{},
    base: VFSBase,

    pub const Options = struct {
        credentials: ?union(enum) {
            json: *std.Io.Reader,
            metadata_server: void,
        } = null,
        endpoint_url: []const u8 = "https://storage.googleapis.com",
        retry: request.Retry = .{},
    };

    pub const InitError = error{
        InvalidCredentialJson,
        Unexpected,
    } || std.mem.Allocator.Error;

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, options: Options) InitError!GCS {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const config: Config = .{
            .credentials = if (options.credentials) |creds| switch (creds) {
                .json => |reader| blk: {
                    var json_reader: std.json.Reader = .init(allocator, reader);
                    defer json_reader.deinit();

                    break :blk std.json.parseFromTokenSourceLeaky(
                        Credentials,
                        arena.allocator(),
                        &json_reader,
                        .{ .allocate = .alloc_always },
                    ) catch return InitError.InvalidCredentialJson;
                },
                .metadata_server => .{ .metadata_server = {} },
            } else null,
            .endpoint_uri = std.Uri.parse(try arena.allocator().dupe(u8, options.endpoint_url)) catch return InitError.Unexpected,
        };

        const token: Token = .{
            .header_storage = try arena.allocator().alloc(u8, authorization_header_size),
            .expires_at = .zero,
        };

        return .{
            .allocator = allocator,
            .arena = arena,
            .base = .init(inner),
            .config = config,
            .token = token,
            // GCS answers rate limiting with `503` as well as `429`.
            .governor = .init(inner, http_client, "gcs", .throttle, options.retry),
        };
    }

    pub fn auto(allocator: std.mem.Allocator, inner_io: std.Io, http_client: *std.http.Client, environ_map: *std.process.Environ.Map) !GCS {
        var jsonBuffer: [1024]u8 = undefined;

        const credentials_file: ?std.Io.File = if (environ_map.get("GOOGLE_APPLICATION_CREDENTIALS")) |json_path| blk: {
            break :blk std.Io.Dir.openFile(.cwd(), inner_io, json_path, .{}) catch |err| {
                log.err("Unable to open GCS credentials file {s}, skipping: {t}", .{ json_path, err });
                break :blk null;
            };
        } else if (applicationDefaultCredentials(inner_io, environ_map)) |f|
            f
        else
            null;
        defer if (credentials_file) |f| f.close(inner_io);

        if (credentials_file) |f| blk: {
            var reader = f.reader(inner_io, &jsonBuffer);
            return GCS.init(allocator, inner_io, http_client, .{ .credentials = .{ .json = &reader.interface } }) catch |err| switch (err) {
                InitError.InvalidCredentialJson => {
                    log.err("Invalid GCS credentials JSON", .{});
                    break :blk;
                },
                else => return err,
            };
        }

        if (isOnGCP(inner_io) catch false) {
            return .init(allocator, inner_io, http_client, .{ .credentials = .{ .metadata_server = {} } });
        }

        return .init(allocator, inner_io, http_client, .{});
    }

    fn applicationDefaultCredentials(io_: std.Io, environ_map: *std.process.Environ.Map) ?std.Io.File {
        var buffer: [1024]u8 = undefined;
        const path = stdx.Io.Dir.path.bufJoin(&buffer, switch (builtin.os.tag) {
            .windows => &.{ environ_map.get("APPDATA") orelse return null, "gcloud", "application_default_credentials.json" },
            else => &.{ environ_map.get("HOME") orelse return null, ".config", "gcloud", "application_default_credentials.json" },
        }) catch unreachable;
        return std.Io.Dir.openFile(.cwd(), io_, path, .{}) catch null;
    }

    /// The metadata server's token, through the governed loop: a throttled
    /// token endpoint holds the backend like any other request.
    fn refreshMetadataServerToken(self: *GCS, buffer: []u8) ![]const u8 {
        var token: TokenRequest = .{
            .uri = self.config.metadata_uri,
            .target = "the metadata server",
            .buffer = buffer,
            .extra_headers = &.{.{ .name = "Metadata-Flavor", .value = "Google" }},
        };
        return request.perform([]const u8, &self.governor, token.spec(), &token, TokenRequest.attempt);
    }

    fn refreshAuthorizedUserToken(self: *GCS, authorized_user: Credentials.AuthorizedUser, buffer: []u8) ![]const u8 {
        // Its own buffer: a retried attempt sends it again after the
        // response was written into `buffer`.
        var payload_buffer: [2 * 1024]u8 = undefined;
        const payload = try std.fmt.bufPrint(&payload_buffer, "grant_type=refresh_token&client_id={f}&client_secret={f}&refresh_token={f}", .{
            std.fmt.alt(std.Uri.Component{ .raw = authorized_user.client_id }, .formatQuery),
            std.fmt.alt(std.Uri.Component{ .raw = authorized_user.client_secret }, .formatQuery),
            std.fmt.alt(std.Uri.Component{ .raw = authorized_user.refresh_token }, .formatQuery),
        });
        var token: TokenRequest = .{
            .uri = self.config.token_uri,
            .target = "the ADC token endpoint",
            .buffer = buffer,
            .method = .POST,
            .payload = payload,
            .content_type = .{ .override = "application/x-www-form-urlencoded" },
        };
        return request.perform([]const u8, &self.governor, token.spec(), &token, TokenRequest.attempt);
    }

    /// One OAuth token exchange, whose body it copies into `buffer`.
    const TokenRequest = struct {
        uri: std.Uri,
        target: []const u8,
        buffer: []u8,
        method: std.http.Method = .GET,
        payload: ?[]u8 = null,
        content_type: std.http.Client.Request.Headers.Value = .default,
        extra_headers: []const std.http.Header = &.{},

        fn spec(self: *const TokenRequest) request.RequestSpec {
            return .{ .target = self.target, .key = request.authorityOf(self.uri) };
        }

        fn attempt(self: *TokenRequest, current: request.Attempt) anyerror!request.Outcome([]const u8) {
            return request.exchange([]const u8, current, self.uri, .{
                .method = self.method,
                .content_type = self.content_type,
                .extra_headers = self.extra_headers,
                .payload = self.payload,
                .redirects = .follow,
                // A refused exchange's body says why (`invalid_grant`, ...).
                .log_refusal_body = true,
            }, self, TokenRequest.consume);
        }

        fn consume(self: *TokenRequest, res: *std.http.Client.Response) anyerror![]const u8 {
            var writer: std.Io.Writer = .fixed(self.buffer);
            _ = try res.reader(&.{}).streamRemaining(&writer);
            return writer.buffered();
        }
    };

    fn refreshServiceAccountToken(io_: std.Io, client: *std.http.Client, service_account: Credentials.ServiceAccount, buffer: []u8) ![]const u8 {
        _ = io_;
        _ = client;
        _ = service_account;
        _ = buffer;
        return error.Unimplemented;
    }

    fn refreshToken(self: *GCS) !void {
        var buffer: [4 * 1024]u8 = undefined;
        const payload = switch (self.config.credentials.?) {
            .authorized_user => |authorized_user| try self.refreshAuthorizedUserToken(authorized_user, &buffer),
            .service_account => |service_account| try refreshServiceAccountToken(self.base.inner, self.governor.client, service_account, &buffer),
            .metadata_server => try self.refreshMetadataServerToken(&buffer),
        };
        const oauth_token = try std.json.parseFromSlice(OAuthToken, self.allocator, payload, .{
            .allocate = .alloc_if_needed,
            .ignore_unknown_fields = true,
        });
        defer oauth_token.deinit();
        const header = try std.fmt.bufPrint(self.token.header_storage, "Bearer {s}", .{oauth_token.value.access_token});
        self.token.header_len = header.len;
        self.token.expires_at = std.Io.Clock.now(.real, self.base.inner).addDuration(.fromSeconds(@intCast(oauth_token.value.expires_in)));
    }

    fn getOrRefreshToken(self: *GCS, authorization_buffer: []u8) !std.http.Client.Request.Headers.Value {
        if (self.config.credentials == null) {
            return .omit;
        }

        const failures = self.token.failures.load(.monotonic);
        try self.token.mutex.lock(self.base.inner);
        defer self.token.mutex.unlock(self.base.inner);
        if (self.token.expired(self.base.inner)) {
            // The refresh this reader waited behind failed, and its reader
            // logged why: repeating it, with a retry budget of its own, would
            // only keep every other queued reader waiting.
            if (self.token.failures.load(.monotonic) != failures) return error.RequestFailed;
            self.refreshToken() catch |err| {
                if (err != error.Canceled) _ = self.token.failures.fetchAdd(1, .monotonic);
                return err;
            };
        }
        // A copy: a refresh by another reader rewrites the token's storage.
        return .{ .override = try std.fmt.bufPrint(authorization_buffer, "{s}", .{self.token.header()}) };
    }

    pub fn deinit(self: *GCS) void {
        self.governor.logTotals();
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

        var it = self.dir_read_states.iterator();
        while (it.next()) |entry| {
            for (entry.value_ptr.objects) |obj| {
                self.allocator.free(obj);
            }
            self.allocator.free(entry.value_ptr.objects);
        }
        self.dir_read_states.deinit(self.allocator);
        self.arena.deinit();
    }

    pub fn io(self: *GCS) std.Io {
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

    pub fn backend(self: *GCS) Backend {
        return .{
            .io = self.io(),
            .read_hints = .{ .high_latency = true },
        };
    }

    fn openHandle(self: *GCS) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| return .{ idx, self.handles.at(idx) };

        return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *GCS, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        self.handles.at(idx).deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *GCS, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(file.handle));
    }

    fn getDirHandle(self: *GCS, dir: std.Io.Dir) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(dir.handle));
    }

    fn resolvePath(self: *GCS, dir: std.Io.Dir, sub_path: []const u8, out_buffer: []u8) ![]u8 {
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
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        switch (operation) {
            .file_read_streaming => |o| {
                const handle = self.getFileHandle(o.file);
                const total = self.performRead(handle, o.data, handle.pos) catch |err| switch (self.governor.readFailure(handle.uri, handle.pos, err)) {
                    error.Canceled => return error.Canceled,
                    error.Unexpected => return .{ .file_read_streaming = error.InputOutput },
                };

                if (total == 0) {
                    return .{ .file_read_streaming = error.EndOfStream };
                }

                handle.pos += @intCast(total);
                return .{ .file_read_streaming = total };
            },
            .file_write_streaming, .device_io_control, .net_receive, .net_read, .net_send, .net_write => {
                return self.base.inner.vtable.operate(self.base.inner.userdata, operation);
            },
        }
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, path, 0) catch return std.Io.Dir.OpenError.Unexpected;

        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getDirHandle(dir);

        return .{
            .inode = @intCast(@intFromPtr(handle)),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .directory,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 1,
        };
    }

    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.StatFileError.SystemResources;
        const size = self.fetchSize(path) catch |err| return self.governor.openFailure(path, err);

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

    fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.AccessError.SystemResources;
        _ = self.fetchSize(path) catch |err| return self.governor.openFailure(path, err);
    }

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenFileOptions) std.Io.File.OpenError!std.Io.File {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;
        const size = self.fetchSize(path) catch |err| return self.governor.openFailure(path, err);

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, path, size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));

        if (reader.state == .finished) return 0;

        if (reader.state == .reset) {
            if (self.dir_read_states.fetchRemove(reader)) |kv| {
                for (kv.value.objects) |obj| self.allocator.free(obj);
                self.allocator.free(kv.value.objects);
            }

            const handle = self.getDirHandle(reader.dir);
            const objects = self.listObjects(handle.uri) catch |err| switch (err) {
                error.Canceled => return error.Canceled,
                error.OutOfMemory => return error.SystemResources,
                // Logged by `exchange`.
                error.RequestFailed => return error.Unexpected,
                else => {
                    log.err("listing {s} failed: {t}", .{ handle.uri, err });
                    return error.Unexpected;
                },
            };

            self.dir_read_states.put(self.allocator, reader, .{
                .index = 0,
                .objects = objects,
            }) catch return std.Io.Dir.Reader.Error.Unexpected;

            reader.state = if (objects.len > 0) .reading else .finished;

            if (objects.len == 0) return 0;
        }

        const state = self.dir_read_states.getPtr(reader) orelse return std.Io.Dir.Reader.Error.Unexpected;

        var count: usize = 0;
        while (count < entries.len and state.index < state.objects.len) {
            const obj_key = state.objects[state.index];
            const kind: std.Io.File.Kind = if (std.mem.endsWith(u8, obj_key, "/")) .directory else .file;

            const name = if (std.mem.lastIndexOfScalar(u8, std.mem.trimEnd(u8, obj_key, "/"), '/')) |idx|
                obj_key[idx + 1 ..]
            else
                obj_key;

            entries[count] = .{
                .name = std.mem.trimEnd(u8, name, "/"),
                .kind = kind,
                .inode = state.index,
            };
            count += 1;
            state.index += 1;
        }

        if (state.index >= state.objects.len) {
            reader.state = .finished;
        }

        return count;
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
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
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| self.governor.readFailure(handle.uri, offset, err);
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);

        if (relative_offset >= 0) {
            handle.pos = handle.pos +| @as(u64, @intCast(relative_offset));
        } else {
            const abs_offset: u64 = @abs(relative_offset);
            if (abs_offset > handle.pos) {
                handle.pos = 0;
            } else {
                handle.pos -= abs_offset;
            }
        }
    }

    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.File.RealPathError.SystemResources;
        return path.len;
    }

    fn pathComponents(_: *GCS, path_: []const u8) struct { []const u8, []const u8 } {
        const path = std.mem.trim(u8, path_, "/");

        if (std.mem.indexOfScalar(u8, path, '/')) |idx| {
            return .{ path[0..idx], if (idx + 1 < path.len) path[idx + 1 ..] else "" };
        } else {
            return .{ path, "" };
        }
    }

    /// The object's URI. A VFS path is `bucket/key`; the request target
    /// needs `/bucket/key`, formatted into `buffer`, which the URI borrows.
    fn gcsUri(self: *GCS, path: []const u8, buffer: []u8) !std.Uri {
        var uri = self.config.endpoint_uri;
        const absolute = if (std.mem.startsWith(u8, path, "/")) path else std.fmt.bufPrint(buffer, "/{s}", .{path}) catch return error.NameTooLong;
        uri.path = .{ .raw = absolute };
        uri.query = .{ .percent_encoded = "alt=media" };
        return uri;
    }

    fn listObjectsBody(self: *GCS, bucket: []const u8, key_prefix: []const u8) ![]u8 {
        var query_buf: [4096]u8 = undefined;
        var query_writer = std.Io.Writer.fixed(&query_buf);

        try query_writer.writeAll("delimiter=");
        try std.Uri.Component.percentEncode(&query_writer, "/", gcsEncodeIsValid);
        try query_writer.writeAll("&max-keys=1000");

        if (key_prefix.len > 0) {
            try query_writer.writeAll("&prefix=");
            try std.Uri.Component.percentEncode(&query_writer, key_prefix, gcsEncodeIsValid);
            if (!std.mem.endsWith(u8, key_prefix, "/")) {
                try std.Uri.Component.percentEncode(&query_writer, "/", gcsEncodeIsValid);
            }
        }

        var path_buf: [256]u8 = undefined;
        var uri = self.config.endpoint_uri;
        uri.path = .{ .percent_encoded = try std.fmt.bufPrint(&path_buf, "/{s}", .{bucket}) };
        uri.query = .{ .percent_encoded = query_writer.buffered() };

        var listing: Listing = .{ .gcs = self, .uri = uri };
        return request.perform([]u8, &self.governor, .{ .target = bucket, .key = request.authorityOf(uri) }, &listing, Listing.attempt);
    }

    /// The bucket listing. The bearer token is copied per attempt, so one
    /// that expired during a hold is refreshed before the next try.
    const Listing = struct {
        gcs: *GCS,
        uri: std.Uri,
        authorization: [authorization_header_size]u8 = undefined,

        fn attempt(self: *Listing, current: request.Attempt) anyerror!request.Outcome([]u8) {
            return request.exchange([]u8, current, self.uri, .{
                .authorization = try self.gcs.getOrRefreshToken(&self.authorization),
                .extra_headers = if (self.gcs.quotaProjectId()) |project|
                    &.{.{ .name = "x-goog-user-project", .value = project }}
                else
                    &.{},
            }, self, Listing.consume);
        }

        fn consume(self: *Listing, res: *std.http.Client.Response) anyerror![]u8 {
            const allocator = self.gcs.allocator;
            return if (res.head.content_length) |content_len|
                try res.reader(&.{}).readAlloc(allocator, content_len)
            else
                try res.reader(&.{}).allocRemaining(allocator, .limited(1024 * 1024));
        }
    };

    fn listObjects(self: *GCS, prefix: []const u8) ![][]const u8 {
        const bucket, const key_prefix = self.pathComponents(prefix);
        const body = try self.listObjectsBody(bucket, key_prefix);
        defer self.allocator.free(body);

        return try self.parseListObjectsResponse(body, key_prefix);
    }

    fn gcsEncodeIsValid(c: u8) bool {
        return switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '.', '_', '~' => true,
            else => false,
        };
    }

    fn quotaProjectId(self: *const GCS) ?[]const u8 {
        const credentials = self.config.credentials orelse return null;
        return switch (credentials) {
            .authorized_user => |authorized_user| authorized_user.quota_project_id,
            .service_account => |service_account| service_account.quota_project_id,
            .metadata_server => null,
        };
    }

    fn parseListObjectsResponse(self: *GCS, xml: []const u8, prefix: []const u8) ![][]const u8 {
        var results: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (results.items) |item| self.allocator.free(item);
            results.deinit(self.allocator);
        }

        inline for (.{ "Key", "Prefix" }) |tag| {
            var pos: usize = 0;
            while (std.mem.indexOfPos(u8, xml, pos, "<" ++ tag ++ ">")) |start| {
                const v_start = start + tag.len + 2;
                const end = std.mem.indexOfPos(u8, xml, v_start, "</" ++ tag ++ ">") orelse break;
                const val = xml[v_start..end];

                const is_self = std.mem.eql(u8, std.mem.trimEnd(u8, val, "/"), std.mem.trimEnd(u8, prefix, "/"));
                if (val.len > 0 and !is_self) {
                    try results.append(self.allocator, try self.allocator.dupe(u8, val));
                }
                pos = end + tag.len + 3;
            }
        }

        return try results.toOwnedSlice(self.allocator);
    }

    fn fetchSize(self: *GCS, path: []const u8) !u64 {
        var uri_path_buffer: [8 * 1024 + 1]u8 = undefined;
        const uri = try self.gcsUri(path, &uri_path_buffer);
        var head: SizeRequest = .{ .gcs = self, .uri = uri };
        return request.perform(u64, &self.governor, .{ .target = path, .key = request.authorityOf(uri) }, &head, SizeRequest.attempt);
    }

    /// The object's size: one HEAD. An absent or denied object fails the
    /// request (`ExchangeOptions.open`).
    const SizeRequest = struct {
        gcs: *GCS,
        uri: std.Uri,
        authorization: [authorization_header_size]u8 = undefined,

        fn attempt(self: *SizeRequest, current: request.Attempt) anyerror!request.Outcome(u64) {
            return request.exchange(u64, current, self.uri, .{
                .method = .HEAD,
                .authorization = try self.gcs.getOrRefreshToken(&self.authorization),
                .open = true,
            }, self, SizeRequest.consume);
        }

        fn consume(_: *SizeRequest, res: *std.http.Client.Response) anyerror!u64 {
            return res.head.content_length orelse error.MissingContentLength;
        }
    };

    fn performRead(self: *GCS, handle: *Handle, data: []const []u8, offset: u64) !usize {
        var uri_path_buffer: [8 * 1024 + 1]u8 = undefined;
        var bearer: BearerRequest = .{ .gcs = self, .uri = try self.gcsUri(handle.uri, &uri_path_buffer) };
        return range_read.performRangeRead(&self.governor, .{
            .request = .{ .target = handle.uri, .key = request.authorityOf(bearer.uri) },
        }, &bearer, BearerRequest.prepare, data, offset, handle.size);
    }

    /// The bearer token is copied per attempt, so one that expired during a
    /// retry delay is refreshed before the next GET.
    const BearerRequest = struct {
        gcs: *GCS,
        uri: std.Uri,
        authorization: [authorization_header_size]u8 = undefined,

        fn prepare(self: *BearerRequest, _: request.Attempt, _: std.http.Header) anyerror!range_read.PreparedRequest {
            return .{ .uri = self.uri, .authorization = try self.gcs.getOrRefreshToken(&self.authorization) };
        }
    };
};

test "GCS object URIs are rooted at the endpoint" {
    var backend: GCS = undefined;
    backend.config = .{ .endpoint_uri = try std.Uri.parse("https://storage.googleapis.com") };

    var path_buffer: [64]u8 = undefined;
    const uri = try backend.gcsUri("bucket/dir/object.bin", &path_buffer);
    var uri_buffer: [128]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&uri_buffer, "{f}", .{uri});
    try std.testing.expectEqualStrings("https://storage.googleapis.com/bucket/dir/object.bin?alt=media", formatted);
}

test "GCS parses XML bucket listing objects and common prefixes" {
    const xml =
        "<?xml version='1.0' encoding='UTF-8'?>" ++
        "<ListBucketResult xmlns='http://doc.s3.amazonaws.com/2006-03-01'>" ++
        "<Name>bucket</Name>" ++
        "<Prefix>LC08/01/001/002/</Prefix>" ++
        "<Contents><Key>LC08/01/001/002/scene_$folder$</Key><Size>6</Size></Contents>" ++
        "<CommonPrefixes><Prefix>LC08/01/001/002/scene/</Prefix></CommonPrefixes>" ++
        "<CommonPrefixes><Prefix>LC08/01/001/002/</Prefix></CommonPrefixes>" ++
        "</ListBucketResult>";

    var backend: GCS = undefined;
    backend.allocator = std.testing.allocator;

    const objects = try backend.parseListObjectsResponse(xml, "LC08/01/001/002/");
    defer {
        for (objects) |object| std.testing.allocator.free(object);
        std.testing.allocator.free(objects);
    }

    try std.testing.expectEqual(@as(usize, 2), objects.len);
    try std.testing.expectEqualStrings("LC08/01/001/002/scene_$folder$", objects[0]);
    try std.testing.expectEqualStrings("LC08/01/001/002/scene/", objects[1]);
}

test "GCS splits bucket and object prefix" {
    var backend: GCS = undefined;

    const root_bucket, const root_prefix = backend.pathComponents("gcp-public-data-landsat/");
    try std.testing.expectEqualStrings("gcp-public-data-landsat", root_bucket);
    try std.testing.expectEqualStrings("", root_prefix);

    const bucket, const prefix = backend.pathComponents("gcp-public-data-landsat/LC08/01");
    try std.testing.expectEqualStrings("gcp-public-data-landsat", bucket);
    try std.testing.expectEqualStrings("LC08/01", prefix);
}

/// A token endpoint at `/token` and one object at `/bucket/object`, for the
/// token tests. The nth token request gets `answers[n]`, the last repeating;
/// a 200 grants `tok-<k>`, k counting the grants.
const TokenServer = struct {
    tcp: std.Io.net.Server,
    group: std.Io.Group = .init,
    object: []const u8,
    answers: []const std.http.Status,
    /// How long each token answer takes, so that readers queue behind a
    /// refresh.
    delay: std.Io.Duration = .fromMilliseconds(100),
    /// The body every token request must POST form-encoded, if any.
    form: ?[]const u8 = null,
    token_requests: std.atomic.Value(usize) = .init(0),
    granted: std.atomic.Value(usize) = .init(0),
    object_requests: std.atomic.Value(usize) = .init(0),
    /// Object requests without the last token granted.
    stale_requests: std.atomic.Value(usize) = .init(0),
    /// Token requests that did not POST `form`.
    bad_forms: std.atomic.Value(usize) = .init(0),
    first_error: stdx.FirstError(anyerror) = .{},

    fn init(io: std.Io, object: []const u8, answers: []const std.http.Status) !TokenServer {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        return .{ .tcp = try address.listen(io, .{ .reuse_address = true }), .object = object, .answers = answers };
    }

    fn deinit(self: *TokenServer, io: std.Io) void {
        self.group.cancel(io);
        self.tcp.deinit(io);
    }

    fn run(self: *TokenServer, io: std.Io) std.Io.Cancelable!void {
        var connections: std.Io.Group = .init;
        defer connections.cancel(io);
        while (true) {
            const stream = self.tcp.accept(io) catch |err| switch (err) {
                error.Canceled => return error.Canceled,
                else => {
                    _ = self.first_error.set(err);
                    return;
                },
            };
            connections.concurrent(io, onConnection, .{ self, stream, io }) catch |err| {
                stream.close(io);
                _ = self.first_error.set(err);
                return;
            };
        }
    }

    fn onConnection(self: *TokenServer, stream: std.Io.net.Stream, io: std.Io) std.Io.Cancelable!void {
        defer stream.close(io);
        self.answer(stream, io) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => _ = self.first_error.set(err),
        };
    }

    fn answer(self: *TokenServer, stream: std.Io.net.Stream, io: std.Io) !void {
        var read_buffer: [8 * 1024]u8 = undefined;
        var reader = stream.reader(io, &read_buffer);
        var write_buffer: [8 * 1024]u8 = undefined;
        var writer = stream.writer(io, &write_buffer);
        var server: std.http.Server = .init(&reader.interface, &writer.interface);
        var req = try server.receiveHead();

        if (std.mem.eql(u8, req.head.target, "/token")) {
            const n = self.token_requests.fetchAdd(1, .acq_rel);
            if (self.form) |form| {
                const content_type = req.head.content_type orelse "";
                const form_encoded = req.head.method == .POST and std.mem.eql(u8, content_type, "application/x-www-form-urlencoded");
                var body_buffer: [512]u8 = undefined;
                var body: std.Io.Writer = .fixed(&body_buffer);
                _ = try req.readerExpectNone(&.{}).streamRemaining(&body);
                if (!form_encoded or !std.mem.eql(u8, body.buffered(), form)) _ = self.bad_forms.fetchAdd(1, .acq_rel);
            }
            try io.sleep(self.delay, .awake);
            const status = self.answers[@min(n, self.answers.len - 1)];
            if (status != .ok) return req.respond("{\"error\":\"invalid_grant\"}", .{ .status = status, .keep_alive = false });
            var token_buffer: [64]u8 = undefined;
            const token = try std.fmt.bufPrint(&token_buffer, "{{\"access_token\":\"tok-{d}\",\"expires_in\":3600}}", .{self.granted.fetchAdd(1, .acq_rel) + 1});
            return req.respond(token, .{ .keep_alive = false });
        }

        if (!std.mem.eql(u8, req.head.target, "/bucket/object?alt=media")) return error.UnexpectedTarget;
        _ = self.object_requests.fetchAdd(1, .acq_rel);
        var authorization: []const u8 = "";
        var range: ?[]const u8 = null;
        var headers = req.iterateHeaders();
        while (headers.next()) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, "authorization")) authorization = header.value;
            if (std.ascii.eqlIgnoreCase(header.name, "range")) range = header.value;
        }
        var expected_buffer: [32]u8 = undefined;
        const expected = try std.fmt.bufPrint(&expected_buffer, "Bearer tok-{d}", .{self.granted.load(.acquire)});
        if (!std.mem.eql(u8, authorization, expected)) _ = self.stale_requests.fetchAdd(1, .acq_rel);
        if (req.head.method == .HEAD) return req.respond(self.object, .{ .keep_alive = false });

        const spec = (range orelse return error.MissingRange)["bytes=".len..];
        const dash = std.mem.indexOfScalar(u8, spec, '-') orelse return error.InvalidRange;
        const first = try std.fmt.parseInt(usize, spec[0..dash], 10);
        const last = try std.fmt.parseInt(usize, spec[dash + 1 ..], 10);
        var content_range_buffer: [64]u8 = undefined;
        return req.respond(self.object[first .. last + 1], .{
            .status = .partial_content,
            .keep_alive = false,
            .extra_headers = &.{.{
                .name = "Content-Range",
                .value = try std.fmt.bufPrint(&content_range_buffer, "bytes {d}-{d}/{d}", .{ first, last, self.object.len }),
            }},
        });
    }
};

/// A GCS whose tokens and object come from a `TokenServer`. It must not
/// move once set up.
const TokenFixture = struct {
    object: [4096]u8 = undefined,
    server: TokenServer = undefined,
    client: std.http.Client = undefined,
    gcs: GCS = undefined,
    endpoint_buffer: [32]u8 = undefined,
    token_url_buffer: [48]u8 = undefined,

    fn setUp(
        self: *TokenFixture,
        credentials: @FieldType(GCS.Options, "credentials"),
        answers: []const std.http.Status,
        max_retries: usize,
    ) !void {
        const io = std.testing.io;
        for (&self.object, 0..) |*byte, i| byte.* = @truncate(i *% 7 +% 3);
        self.server = try .init(io, &self.object, answers);
        errdefer self.server.deinit(io);
        try self.server.group.concurrent(io, TokenServer.run, .{ &self.server, io });
        self.client = .{ .allocator = std.testing.allocator, .io = io };
        errdefer self.client.deinit();
        const port = self.server.tcp.socket.address.getPort();
        const token_uri: std.Uri = try .parse(try std.fmt.bufPrint(&self.token_url_buffer, "http://127.0.0.1:{d}/token", .{port}));
        self.gcs = try .init(std.testing.allocator, io, &self.client, .{
            .credentials = credentials,
            .endpoint_url = try std.fmt.bufPrint(&self.endpoint_buffer, "http://127.0.0.1:{d}", .{port}),
            .retry = .{
                .max_retries = max_retries,
                .initial_delay = .fromMilliseconds(5),
                .max_delay = .fromMilliseconds(5),
                .max_hold = .fromMilliseconds(20),
            },
        });
        self.gcs.config.metadata_uri = token_uri;
        self.gcs.config.token_uri = token_uri;
    }

    fn tearDown(self: *TokenFixture) void {
        self.gcs.deinit();
        self.client.deinit();
        self.server.deinit(std.testing.io);
    }

    /// Stops the server and returns the first error it met.
    fn stop(self: *TokenFixture) !void {
        self.server.group.cancel(std.testing.io);
        if (self.server.first_error.get()) |err| return err;
    }

    /// `readers` concurrent reads of 256 bytes each, started together;
    /// returns how many failed. Every read that succeeds must match.
    fn readConcurrently(self: *TokenFixture, file: std.Io.File, comptime readers: usize) !usize {
        const io = std.testing.io;
        const Reader = struct {
            fn run(gcs_io: std.Io, file_: std.Io.File, output: []u8, offset: u64, go: *std.Io.Event, result: *anyerror!usize) void {
                go.wait(gcs_io) catch |err| {
                    result.* = err;
                    return;
                };
                result.* = file_.readPositional(gcs_io, &.{output}, offset);
            }
        };
        var outputs: [readers][256]u8 = undefined;
        var results: [readers]anyerror!usize = undefined;
        var go: std.Io.Event = .unset;
        var group: std.Io.Group = .init;
        defer group.cancel(io);
        for (&outputs, &results, 0..) |*output, *result, i| {
            try group.concurrent(io, Reader.run, .{ self.gcs.io(), file, output, i * output.len, &go, result });
        }
        go.set(io);
        try group.await(io);

        var failed: usize = 0;
        for (&outputs, results, 0..) |*output, result, i| {
            const n = result catch {
                failed += 1;
                continue;
            };
            try std.testing.expectEqual(output.len, n);
            try std.testing.expectEqualSlices(u8, self.object[i * output.len ..][0..output.len], output);
        }
        return failed;
    }
};

test "GCS refreshes an expired token once for concurrent readers, and each sends the new one" {
    var fixture: TokenFixture = .{};
    try fixture.setUp(.{ .metadata_server = {} }, &.{.ok}, 0);
    defer fixture.tearDown();
    const server = &fixture.server;
    const gcs_io = fixture.gcs.io();
    const file = try std.Io.Dir.openFile(.cwd(), gcs_io, "bucket/object", .{});
    defer file.close(gcs_io);
    try std.testing.expectEqual(1, server.token_requests.load(.acquire));

    fixture.gcs.token.expires_at = .zero;
    try std.testing.expectEqual(0, try fixture.readConcurrently(file, 16));
    try fixture.stop();
    try std.testing.expectEqual(2, server.token_requests.load(.acquire));
    try std.testing.expectEqual(17, server.object_requests.load(.acquire));
    try std.testing.expectEqual(0, server.stale_requests.load(.acquire));
}

test "GCS fails the readers queued behind a failed token refresh without repeating it" {
    const cases = [_]struct { status: std.http.Status, max_retries: usize, token_requests: usize }{
        // The open's request, then one refused refresh.
        .{ .status = .unauthorized, .max_retries = 0, .token_requests = 2 },
        // The open's request, then one refresh and its 2 retries.
        .{ .status = .internal_server_error, .max_retries = 2, .token_requests = 4 },
    };
    for (cases) |case| {
        var fixture: TokenFixture = .{};
        try fixture.setUp(.{ .metadata_server = {} }, &.{ .ok, case.status }, case.max_retries);
        defer fixture.tearDown();
        const server = &fixture.server;
        const gcs_io = fixture.gcs.io();
        const file = try std.Io.Dir.openFile(.cwd(), gcs_io, "bucket/object", .{});
        defer file.close(gcs_io);

        fixture.gcs.token.expires_at = .zero;
        try std.testing.expectEqual(16, try fixture.readConcurrently(file, 16));
        try fixture.stop();
        try std.testing.expectEqual(case.token_requests, server.token_requests.load(.acquire));
        try std.testing.expectEqual(1, server.object_requests.load(.acquire));
    }
}

test "GCS resends the ADC refresh form on a retry, and a throttled token endpoint holds the backend" {
    var json: std.Io.Reader = .fixed(
        \\{"type":"authorized_user","client_id":"cid.apps.googleusercontent.com","client_secret":"GOCSPX-sec","refresh_token":"1//0abc-def_ghi","account":"","universe_domain":"googleapis.com"}
    );
    var fixture: TokenFixture = .{};
    try fixture.setUp(.{ .json = &json }, &.{ .too_many_requests, .internal_server_error, .ok }, 2);
    defer fixture.tearDown();
    const server = &fixture.server;
    server.delay = .zero;
    server.form = "grant_type=refresh_token&client_id=cid.apps.googleusercontent.com&client_secret=GOCSPX-sec&refresh_token=1//0abc-def_ghi";
    const gcs_io = fixture.gcs.io();
    const file = try std.Io.Dir.openFile(.cwd(), gcs_io, "bucket/object", .{});
    defer file.close(gcs_io);

    try std.testing.expectEqual(0, try fixture.readConcurrently(file, 4));
    try fixture.stop();
    const stats = &fixture.gcs.governor.stats;
    try std.testing.expectEqual(1, stats.holds.load(.acquire));
    try std.testing.expectEqual(1, stats.retries.load(.acquire));
    try std.testing.expectEqual(3, server.token_requests.load(.acquire));
    try std.testing.expectEqual(0, server.bad_forms.load(.acquire));
    try std.testing.expectEqual(0, server.stale_requests.load(.acquire));
}
