const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/gcs");

const EndpointUrl = "https://storage.googleapis.com";
const MetadataUrl = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

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
        pub const JwtAssertion = struct {
            iss: []const u8,
            scope: []const u8 = "https://www.googleapis.com/auth/devstorage.read_only",
            aud: []const u8 = "https://oauth2.googleapis.com/token",
            exp: i64,
            iat: i64,
            sub: ?[]const u8 = null,
        };

        type: []const u8,
        client_email: []const u8,
        private_key: []const u8,
        token_uri: []const u8,
        quota_project_id: ?[]const u8 = null,

        pub fn makeAssertion(self: *const ServiceAccount, io_: std.Io) JwtAssertion {
            const now: std.Io.Timestamp = .now(io_, .real);
            return .{
                .iss = self.client_email,
                .iat = now.toSeconds(),
                .exp = now.addDuration(.fromSeconds(3600)).toSeconds(),
            };
        }
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
                unreachable;
            },
            else => std.json.ParseFromValueError.UnexpectedToken,
        };
    }
};

const ReadState = struct { index: usize, objects: [][]const u8 };

pub const GCS = struct {
    const Token = struct {
        header: []u8,
        expires_at: std.Io.Timestamp,

        fn expired(self: *const Token, io_: std.Io) bool {
            const now: std.Io.Timestamp = .now(io_, .real);
            return now.toSeconds() >= self.expires_at.toSeconds();
        }
    };

    pub const Config = struct {
        credentials: ?Credentials = null,
        endpoint_uri: std.Uri,
        region: []const u8 = "auto",
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
    client: *std.http.Client,
    config: Config,
    token: Token,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .empty,
    dir_read_states: std.AutoHashMapUnmanaged(*std.Io.Dir.Reader, ReadState) = .{},
    base: VFSBase,

    pub const InitArgs = struct {
        credentials: ?union(enum) {
            json: *std.Io.Reader,
            metadata_server: void,
        } = null,
        endpoint_url: []const u8 = "https://storage.googleapis.com",
        region: []const u8 = "auto",
    };

    pub const InitError = error{
        InvalidCredentialJson,
        RequestFailed,
        Unexpected,
    } || std.mem.Allocator.Error;

    pub fn init(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, args: InitArgs) InitError!GCS {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();

        const config: Config = .{
            .credentials = if (args.credentials) |creds| switch (creds) {
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
            .endpoint_uri = std.Uri.parse(try arena.allocator().dupe(u8, args.endpoint_url)) catch return InitError.Unexpected,
            .region = try arena.allocator().dupe(u8, args.region),
        };

        const token: Token = .{
            .header = try arena.allocator().alloc(u8, "Bearer ".len + OAuthToken.Size),
            .expires_at = .zero,
        };

        return .{
            .allocator = allocator,
            .arena = arena,
            .base = .init(inner),
            .client = http_client,
            .config = config,
            .token = token,
        };
    }

    pub fn auto(allocator: std.mem.Allocator, inner_io: std.Io, http_client: *std.http.Client, environ_map: *std.process.Environ.Map) !GCS {
        var jsonBuffer: [1024]u8 = undefined;

        if (environ_map.get("GOOGLE_APPLICATION_CREDENTIALS")) |json_path| {
            var f = try std.Io.Dir.openFile(.cwd(), inner_io, json_path, .{});
            defer f.close(inner_io);
            var reader = f.reader(inner_io, &jsonBuffer);
            return try .init(allocator, inner_io, http_client, .{ .credentials = .{ .json = &reader.interface } });
        }

        if (applicationDefaultCredentials(inner_io, environ_map)) |f| {
            defer f.close(inner_io);
            var reader = f.reader(inner_io, &jsonBuffer);
            return .init(allocator, inner_io, http_client, .{ .credentials = .{ .json = &reader.interface } });
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

    fn refreshMetadataServerToken(client: *std.http.Client, buffer: []u8) !?[]const u8 {
        var response_writer: std.Io.Writer = .fixed(buffer);
        const result = try client.fetch(.{
            .location = .{ .url = MetadataUrl },
            .method = .GET,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
            },
            .extra_headers = &.{
                .{ .name = "Metadata-Flavor", .value = "Google" },
            },
            .response_writer = &response_writer,
        });

        if (result.status != .ok) {
            return null;
        }

        return response_writer.buffered();
    }

    fn refreshAuthorizedUserToken(client: *std.http.Client, authorized_user: Credentials.AuthorizedUser, buffer: []u8) ![]const u8 {
        var response_writer: std.Io.Writer = .fixed(buffer);
        const result = try client.fetch(.{
            .location = .{ .url = "https://oauth2.googleapis.com/token" },
            .method = .POST,
            .payload = try std.fmt.bufPrint(buffer, "grant_type=refresh_token&client_id={f}&client_secret={f}&refresh_token={f}", .{
                std.fmt.alt(std.Uri.Component{ .raw = authorized_user.client_id }, .formatQuery),
                std.fmt.alt(std.Uri.Component{ .raw = authorized_user.client_secret }, .formatQuery),
                std.fmt.alt(std.Uri.Component{ .raw = authorized_user.refresh_token }, .formatQuery),
            }),
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .content_type = .{ .override = "application/x-www-form-urlencoded" },
            },
            .response_writer = &response_writer,
        });
        if (result.status != .ok) {
            log.err("Failed to refresh ADC token: {s}", .{response_writer.buffered()});
            return error.RequestFailed;
        }
        return response_writer.buffered();
    }

    fn refreshServiceAccountToken(io_: std.Io, client: *std.http.Client, service_account: Credentials.ServiceAccount, buffer: []u8) ![]const u8 {
        _ = io_;
        _ = client;
        _ = service_account;
        _ = buffer;
        return error.Unimplemented;

        // const assertion = service_account.makeAssertion(io_);

        // var tmp_buffer: [8 * 1024]u8 = undefined;

        // var buffer_writer: std.Io.Writer = .fixed(buffer);
        // var signing_input_buffer: [2 * 1024]u8 = undefined;
        // var signing_input_writer: std.Io.Writer = .fixed(signing_input_buffer);

        // const encoded_rs256_header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
        // try signing_input_writer.print("{s}.", .{encoded_rs256_header});
        // std.base64.url_safe_no_pad.Encoder.encodeWriter(
        //     signing_input_writer,
        //     try std.fmt.bufPrint(tmp_buffer, "{f}", .{
        //         std.json.fmt(assertion, .{}),
        //     }),
        // );
        // const signing_input = signing_input_writer.buffered();
        // _ = signing_input;

        // const result = try client.fetch(.{
        //     .location = .{ .url = "https://oauth2.googleapis.com/token" },
        //     .method = .POST,
        //     .payload = try std.fmt.bufPrint(buffer, "grant_type=refresh_token&client_id={f}&client_secret={f}&refresh_token={f}", .{
        //         std.fmt.alt(std.Uri.Component{ .raw = service_account.client_id }, .formatQuery),
        //         std.fmt.alt(std.Uri.Component{ .raw = authorized_user.client_secret }, .formatQuery),
        //         std.fmt.alt(std.Uri.Component{ .raw = authorized_user.refresh_token }, .formatQuery),
        //     }),
        //     .headers = .{
        //         .accept_encoding = .{ .override = "identity" },
        //         .content_type = .{ .override = "application/x-www-form-urlencoded" },
        //     },
        //     .response_writer = &response_writer,
        // });
        // if (result.status != .ok) {
        //     log.err("Failed to refresh ADC token: {s}", .{response_writer.buffered()});
        //     return error.RequestFailed;
        // }
    }

    fn refreshToken(self: *GCS) !void {
        var buffer: [4 * 1024]u8 = undefined;
        const payload = switch (self.config.credentials.?) {
            .authorized_user => |authorized_user| try refreshAuthorizedUserToken(self.client, authorized_user, &buffer),
            .service_account => |service_account| try refreshServiceAccountToken(self.base.inner, self.client, service_account, &buffer),
            .metadata_server => try refreshMetadataServerToken(self.client, &buffer) orelse return error.RequestFailed,
        };
        const oauth_token = try std.json.parseFromSlice(OAuthToken, self.allocator, payload, .{
            .allocate = .alloc_if_needed,
            .ignore_unknown_fields = true,
        });
        defer oauth_token.deinit();
        self.token = .{
            .header = try std.fmt.bufPrint(self.token.header, "Bearer {s}", .{oauth_token.value.access_token}),
            .expires_at = std.Io.Clock.now(.real, self.base.inner).addDuration(.fromSeconds(@intCast(oauth_token.value.expires_in))),
        };
    }

    fn getOrRefreshToken(self: *GCS) !std.http.Client.Request.Headers.Value {
        if (self.config.credentials == null) {
            return .omit;
        }
        if (self.token.expired(self.base.inner)) {
            try self.refreshToken();
        }
        return .{ .override = self.token.header };
    }

    pub fn deinit(self: *GCS) void {
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
        const size = self.fetchSize(dir, sub_path) catch |err| switch (err) {
            error.FileNotFound => return std.Io.File.OpenError.FileNotFound,
            error.PermissionDenied => return std.Io.File.OpenError.PermissionDenied,
            else => return std.Io.File.OpenError.Unexpected,
        };

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

    fn dirAccess(_: ?*anyopaque, _: std.Io.Dir, _: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {}

    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const size = self.fetchSize(dir, sub_path) catch |err| switch (err) {
            error.FileNotFound => return std.Io.File.OpenError.FileNotFound,
            error.PermissionDenied => return std.Io.File.OpenError.PermissionDenied,
            else => return std.Io.File.OpenError.Unexpected,
        };

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;
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
        _ = reader;
        _ = entries;
        _ = self;

        return 0;

        // if (reader.state == .finished) return 0;

        // if (reader.state == .reset) {
        //     if (self.dir_read_states.fetchRemove(reader)) |kv| {
        //         for (kv.value.objects) |obj| self.allocator.free(obj);
        //         self.allocator.free(kv.value.objects);
        //     }

        //     const handle = self.getDirHandle(reader.dir);
        //     const objects = self.listObjects(handle.uri) catch return std.Io.Dir.Reader.Error.Unexpected;

        //     self.dir_read_states.put(self.allocator, reader, .{
        //         .index = 0,
        //         .objects = objects,
        //     }) catch return std.Io.Dir.Reader.Error.Unexpected;

        //     reader.state = if (objects.len > 0) .reading else .finished;

        //     if (objects.len == 0) return 0;
        // }

        // const state = self.dir_read_states.getPtr(reader) orelse return std.Io.Dir.Reader.Error.Unexpected;

        // var count: usize = 0;
        // while (count < entries.len and state.index < state.objects.len) {
        //     const obj_key = state.objects[state.index];
        //     const kind: std.Io.File.Kind = if (std.mem.endsWith(u8, obj_key, "/")) .directory else .file;

        //     const name = if (std.mem.lastIndexOfScalar(u8, std.mem.trimEnd(u8, obj_key, "/"), '/')) |idx|
        //         obj_key[idx + 1 ..]
        //     else
        //         obj_key;

        //     entries[count] = .{
        //         .name = std.mem.trimEnd(u8, name, "/"),
        //         .kind = kind,
        //         .inode = state.index,
        //     };
        //     count += 1;
        //     state.index += 1;
        // }

        // if (state.index >= state.objects.len) {
        //     reader.state = .finished;
        // }

        // return count;
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
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return error.Unexpected;
        };
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *GCS = @alignCast(@fieldParentPtr("base", VFSBase.as(userdata)));
        const handle = self.getFileHandle(file);

        if (relative_offset >= 0) {
            handle.pos = handle.pos +| @as(u64, @intCast(relative_offset));
        } else {
            const abs_offset: u64 = @intCast(-relative_offset);
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

    fn getTimestamp(self: *GCS, buf: []u8) ![]const u8 {
        const now_ts = std.Io.Clock.now(.real, self.base.inner);

        const now: u64 = @intCast(@divFloor(now_ts.nanoseconds, std.time.ns_per_s));
        const epoch_secs: std.time.epoch.EpochSeconds = .{ .secs = now };
        const year_day = epoch_secs.getEpochDay().calculateYearDay();
        const month_day = year_day.calculateMonthDay();
        const day_seconds = epoch_secs.getDaySeconds();

        return try std.fmt.bufPrint(buf, "{d:0>4}{d:0>2}{d:0>2}T{d:0>2}{d:0>2}{d:0>2}Z", .{
            year_day.year,
            @intFromEnum(month_day.month),
            month_day.day_index + 1,
            day_seconds.getHoursIntoDay(),
            day_seconds.getMinutesIntoHour(),
            day_seconds.getSecondsIntoMinute(),
        });
    }

    // fn pathComponents(self: *GCS, path_: []const u8) struct { []const u8, []const u8, []const u8 } {
    //     const endpoint = std.mem.trimEnd(u8, self.config.endpoint_url, "/");
    //     const path = std.mem.trim(u8, path_, "/");

    //     if (std.mem.findScalar(u8, path, '/')) |idx| {
    //         return .{ endpoint, path[0..idx], if (idx + 1 < path.len) path[idx + 1 ..] else "" };
    //     } else {
    //         return .{ endpoint, path, "" };
    //     }
    // }

    fn gcsUri(self: *GCS, path: []const u8) std.Uri {
        var uri = self.config.endpoint_uri;
        uri.path = .{ .raw = path };
        uri.query = .{ .percent_encoded = "alt=media" };
        return uri;
    }

    // fn authHeaders(self: *GCS, _: std.http.Method, _: std.Uri, _: []const u8, authorization_buffer: []u8) !struct {
    //     authorization: ?[]const u8,
    //     extra_headers: [3]?std.http.Header,
    //     extra_len: usize,
    // } {
    //     if (self.config.oauth_access_token == null) {
    //         if (self.config.googleCredentialsJsonStr) |json| {
    //             const oauth = try refreshAuthorizedUserTokenFromJson(self.allocator, self.base.inner, json);
    //             if (self.config.oauth_access_token) |old_token| self.allocator.free(old_token);
    //             if (self.config.quota_project_id) |old_project| self.allocator.free(old_project);
    //             self.config.oauth_access_token = oauth.access_token;
    //             self.config.quota_project_id = oauth.quota_project_id;
    //         }
    //     }

    //     const token = self.config.oauth_access_token orelse return error.MissingCredentials;
    //     const authorization = try std.fmt.bufPrint(authorization_buffer, "Bearer {s}", .{token});
    //     var extra_headers: [3]?std.http.Header = .{ null, null, null };
    //     var extra_len: usize = 0;
    //     if (self.config.quota_project_id) |project| {
    //         extra_headers[0] = .{ .name = "x-goog-user-project", .value = project };
    //         extra_len = 1;
    //     }
    //     return .{
    //         .authorization = authorization,
    //         .extra_headers = extra_headers,
    //         .extra_len = extra_len,
    //     };
    // }

    // fn listObjects(self: *GCS, prefix: []const u8) ![][]const u8 {
    //     const endpoint, const bucket, const key_prefix = self.pathComponents(prefix);

    //     var query_buf: [4096]u8 = undefined;
    //     var query_writer = std.Io.Writer.fixed(&query_buf);

    //     try query_writer.writeAll("delimiter=");
    //     try std.Uri.Component.percentEncode(&query_writer, "/", gcsEncodeIsValid);

    //     if (key_prefix.len > 0) {
    //         try query_writer.writeAll("&prefix=");
    //         try std.Uri.Component.percentEncode(&query_writer, key_prefix, gcsEncodeIsValid);
    //         try std.Uri.Component.percentEncode(&query_writer, "/", gcsEncodeIsValid);
    //     }

    //     const endpoint_uri = try std.Uri.parse(endpoint);

    //     var path_buf: [256]u8 = undefined;

    //     const uri: std.Uri = .{
    //         .scheme = endpoint_uri.scheme,
    //         .host = endpoint_uri.host,
    //         .port = endpoint_uri.port,
    //         .path = .{ .percent_encoded = try std.fmt.bufPrint(&path_buf, "/{s}", .{bucket}) },
    //         .query = .{ .percent_encoded = query_writer.buffered() },
    //         .fragment = null,
    //     };

    //     var timestamp_buf: [16]u8 = undefined;
    //     const timestamp = try self.getTimestamp(&timestamp_buf);

    //     var authorization_buffer: [1024]u8 = undefined;
    //     const auth = try self.authHeaders(.GET, uri, timestamp, &authorization_buffer);

    //     var extra_headers_buf: [3]std.http.Header = undefined;
    //     for (0..auth.extra_len) |i| {
    //         extra_headers_buf[i] = auth.extra_headers[i].?;
    //     }

    //     var req = try self.client.request(.GET, uri, .{
    //         .redirect_behavior = .not_allowed,
    //         .headers = .{
    //             .accept_encoding = .{ .override = "identity" },
    //             .authorization = if (self.getOrRefreshToken()) |hdr| .{ .override = hdr } else .omit,
    //         },
    //         .extra_headers = if (self.config.quota_project_id) |project|
    //             &.{.{ .name = "x-goog-user-project", .value = project }}
    //         else
    //             &.{},
    //     });
    //     defer req.deinit();

    //     try req.sendBodiless();

    //     var redirect_buffer: [2 * 1024]u8 = undefined;
    //     var res = try req.receiveHead(&redirect_buffer);

    //     if (res.head.status != .ok) {
    //         log.err("Failed to list object {f}", .{uri});
    //         log.err("{s}", .{res.head.bytes});
    //         return error.RequestFailed;
    //     }

    //     const body = try res.reader(&.{}).readAlloc(self.allocator, res.head.content_length orelse 1024 * 1024);
    //     defer self.allocator.free(body);

    //     return try self.parseListObjectsResponse(body, key_prefix);
    // }

    // fn parseListObjectsResponse(self: *GCS, xml: []const u8, prefix: []const u8) ![][]const u8 {
    //     var results = std.ArrayListUnmanaged([]const u8){};
    //     errdefer {
    //         for (results.items) |item| self.allocator.free(item);
    //         results.deinit(self.allocator);
    //     }

    //     inline for (.{ "Key", "Prefix" }) |tag| {
    //         var pos: usize = 0;
    //         while (std.mem.indexOfPos(u8, xml, pos, "<" ++ tag ++ ">")) |start| {
    //             const v_start = start + tag.len + 2;
    //             const end = std.mem.indexOfPos(u8, xml, v_start, "</" ++ tag ++ ">") orelse break;
    //             const val = xml[v_start..end];

    //             const is_self = std.mem.eql(u8, std.mem.trimEnd(u8, val, "/"), std.mem.trimEnd(u8, prefix, "/"));
    //             if (val.len > 0 and !is_self) {
    //                 try results.append(self.allocator, try self.allocator.dupe(u8, val));
    //             }
    //             pos = end + tag.len + 3;
    //         }
    //     }

    //     return try results.toOwnedSlice(self.allocator);
    // }

    fn fetchSize(self: *GCS, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = try self.resolvePath(dir, sub_path, &path_buffer);
        const uri = self.gcsUri(path);
        var req = try self.client.request(.HEAD, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = try self.getOrRefreshToken(),
            },
            .extra_headers = &.{},
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        const res = try req.receiveHead(&redirect_buffer);

        const size = switch (res.head.status.class()) {
            .success => res.head.content_length.?,
            else => switch (res.head.status) {
                .not_found => return error.FileNotFound,
                .unauthorized, .forbidden => return error.PermissionDenied,
                else => blk: {
                    log.err("Failed to fetch size for {f}: {s}", .{ uri, res.head.bytes });
                    break :blk error.ServerError;
                },
            },
        };
        return size;
    }

    fn performRead(self: *GCS, handle: *Handle, data: []const []u8, offset: u64) !usize {
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

        const uri = self.gcsUri(handle.uri);
        var req = try self.client.request(
            .GET,
            uri,
            .{
                .headers = .{
                    .accept_encoding = .{ .override = "identity" },
                    .authorization = try self.getOrRefreshToken(),
                },
                .extra_headers = &.{.{ .name = "Range", .value = range_header }},
            },
        );
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .partial_content and res.head.status != .ok) {
            log.err("Failed to read {s}: {s}", .{ handle.uri, res.head.bytes });
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
