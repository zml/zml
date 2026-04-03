const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/gcs");

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
    access_token: []const u8,
    quota_project_id: ?[]const u8 = null,
    source: enum {
        adc_file,
        metadata_server,
    } = .adc_file,
};

pub const GcsSigV4 = struct {
    access_key: ?[]const u8,
    secret_key: ?[]const u8,
    region: []const u8,
    service: []const u8,

    const ALGORITHM = "GOOG4-HMAC-SHA256";
    const KEY_PREFIX = "GOOG4";
    const REQUEST_TYPE = "goog4_request";
    pub const UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD";

    pub fn generateAuthHeader(
        self: GcsSigV4,
        output_buffer: []u8,
        method: std.http.Method,
        uri: std.Uri,
        timestamp: []const u8,
        extra_headers: []const std.http.Header,
    ) !?[]const u8 {
        const access_key = self.access_key orelse return null;
        _ = self.secret_key orelse return null;

        const date = timestamp[0..8];

        var header_buf: [32]std.http.Header = undefined;
        var header_count: usize = 0;

        const host = uri.host orelse return error.MissingHost;
        const host_header: std.http.Header = blk: {
            const host_ = switch (host) {
                .raw, .percent_encoded => |h| h,
            };
            if (uri.port) |port| {
                var host_buf: [128]u8 = undefined;
                break :blk .{ .name = "host", .value = try std.fmt.bufPrint(&host_buf, "{s}:{d}", .{ host_, port }) };
            } else {
                break :blk .{ .name = "host", .value = host_ };
            }
        };

        header_buf[0] = host_header;
        header_buf[1] = .{ .name = "x-goog-content-sha256", .value = UNSIGNED_PAYLOAD };
        header_buf[2] = .{ .name = "x-goog-date", .value = timestamp };
        header_count = 3;

        for (extra_headers) |h| {
            if (header_count >= header_buf.len) return error.TooManyHeaders;
            header_buf[header_count] = h;
            header_count += 1;
        }

        const active_headers = header_buf[0..header_count];

        std.mem.sort(std.http.Header, active_headers, {}, struct {
            fn less(_: void, a: std.http.Header, b: std.http.Header) bool {
                return std.ascii.lessThanIgnoreCase(a.name, b.name);
            }
        }.less);

        var sha = std.crypto.hash.sha2.Sha256.init(.{});
        var sha_writer = stdx.crypto.hmacWriter(&sha);
        const sw = &sha_writer.interface;

        try sw.print("{s}\n", .{@tagName(method)});
        if (uri.path.isEmpty()) {
            try sw.writeAll("/\n");
        } else {
            try uri.path.formatPath(sw);
            try sw.writeByte('\n');
        }

        if (uri.query) |query| {
            try query.formatEscaped(sw);
            try sw.writeByte('\n');
        } else {
            try sw.writeAll("\n");
        }

        var signed_headers_buf: [512]u8 = undefined;
        var sh_writer = std.Io.Writer.fixed(&signed_headers_buf);
        for (active_headers, 0..) |h, i| {
            for (h.name) |char| try sw.writeByte(std.ascii.toLower(char));
            try sw.print(":{s}\n", .{h.value});
            if (i > 0) _ = try sh_writer.write(";");
            for (h.name) |char| try sh_writer.writeByte(std.ascii.toLower(char));
        }
        const signed_headers = sh_writer.buffered();

        try sw.print("\n{s}\n{s}", .{ signed_headers, UNSIGNED_PAYLOAD });

        var canonical_request_hash: [32]u8 = undefined;
        sha.final(&canonical_request_hash);

        var scope_buf: [128]u8 = undefined;
        const scope = try std.fmt.bufPrint(&scope_buf, "{s}/{s}/{s}/{s}", .{ date, self.region, self.service, REQUEST_TYPE });

        const signing_key = try self.deriveSigningKey(date);

        var h_state = std.crypto.auth.hmac.sha2.HmacSha256.init(&signing_key);
        var h_writer = stdx.crypto.hmacWriter(&h_state);
        const hw = &h_writer.interface;

        try hw.print("{s}\n{s}\n{s}\n{x}", .{ ALGORITHM, timestamp, scope, &canonical_request_hash });

        var signature: [32]u8 = undefined;
        h_state.final(&signature);

        return try std.fmt.bufPrint(output_buffer, "{s} Credential={s}/{s}, SignedHeaders={s}, Signature={x}", .{
            ALGORITHM,
            access_key,
            scope,
            signed_headers,
            &signature,
        });
    }

    fn deriveSigningKey(self: GcsSigV4, date: []const u8) ![32]u8 {
        var key_buf: [128]u8 = undefined;
        const k_init = try std.fmt.bufPrint(&key_buf, "{s}{s}", .{ KEY_PREFIX, self.secret_key.? });

        const k_date = hmacStatic(k_init, date);
        const k_region = hmacStatic(&k_date, self.region);
        const k_service = hmacStatic(&k_region, self.service);
        return hmacStatic(&k_service, REQUEST_TYPE);
    }

    fn hmacStatic(key: []const u8, data: []const u8) [32]u8 {
        var out: [32]u8 = undefined;
        std.crypto.auth.hmac.sha2.HmacSha256.create(&out, data, key);
        return out;
    }
};

const ReadState = struct { index: usize, objects: [][]const u8 };

pub const GCS = struct {
    pub const Config = struct {
        access_key: ?[]const u8 = null,
        secret_key: ?[]const u8 = null,
        oauth_access_token: ?[]const u8 = null,
        quota_project_id: ?[]const u8 = null,
        endpoint_url: []const u8,
        region: []const u8 = "auto",
        auth_service: []const u8 = "storage",
    };

    pub const HmacConfig = struct {
        access_key: []const u8,
        secret_key: []const u8,
        endpoint_url: []const u8 = "https://storage.googleapis.com",
        region: []const u8 = "auto",
        auth_service: []const u8 = "storage",
    };

    pub const OAuthConfig = struct {
        access_token: []const u8,
        quota_project_id: ?[]const u8 = null,
        endpoint_url: []const u8 = "https://storage.googleapis.com",
        region: []const u8 = "auto",
        auth_service: []const u8 = "storage",
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
    mutex: std.Io.Mutex = .init,
    client: *std.http.Client,
    config: Config,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .{},
    dir_read_states: std.AutoHashMapUnmanaged(*std.Io.Dir.Reader, ReadState) = .{},
    base: VFSBase,

    pub fn init(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        config: Config,
    ) !GCS {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .config = .{
                .access_key = if (config.access_key) |k| try allocator.dupe(u8, k) else null,
                .secret_key = if (config.secret_key) |k| try allocator.dupe(u8, k) else null,
                .oauth_access_token = if (config.oauth_access_token) |k| try allocator.dupe(u8, k) else null,
                .quota_project_id = if (config.quota_project_id) |k| try allocator.dupe(u8, k) else null,
                .endpoint_url = try allocator.dupe(u8, config.endpoint_url),
                .region = try allocator.dupe(u8, config.region),
                .auth_service = try allocator.dupe(u8, config.auth_service),
            },
        };
    }

    pub fn initHmac(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        config: HmacConfig,
    ) !GCS {
        return init(allocator, inner, http_client, .{
            .access_key = config.access_key,
            .secret_key = config.secret_key,
            .endpoint_url = config.endpoint_url,
            .region = config.region,
            .auth_service = config.auth_service,
        });
    }

    pub fn initOAuth(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        config: OAuthConfig,
    ) !GCS {
        return init(allocator, inner, http_client, .{
            .oauth_access_token = config.access_token,
            .quota_project_id = config.quota_project_id,
            .endpoint_url = config.endpoint_url,
            .region = config.region,
            .auth_service = config.auth_service,
        });
    }

    pub fn initAuthorizedUserADC(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        credential_path: []const u8,
        endpoint_url: []const u8,
        region: []const u8,
    ) !GCS {
        const oauth = try refreshAuthorizedUserTokenFromFile(allocator, inner, credential_path);
        defer {
            allocator.free(oauth.access_token);
            if (oauth.quota_project_id) |project| allocator.free(project);
        }

        return initOAuth(allocator, inner, http_client, .{
            .access_token = oauth.access_token,
            .quota_project_id = oauth.quota_project_id,
            .endpoint_url = endpoint_url,
            .region = region,
        });
    }

    pub fn initMetadataServer(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        endpoint_url: []const u8,
        region: []const u8,
    ) !GCS {
        const oauth = try fetchMetadataServerToken(allocator, inner) orelse return error.CredentialsNotFound;
        defer allocator.free(oauth.access_token);

        return initOAuth(allocator, inner, http_client, .{
            .access_token = oauth.access_token,
            .quota_project_id = oauth.quota_project_id,
            .endpoint_url = endpoint_url,
            .region = region,
        });
    }

    pub fn auto(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, environ_map: *std.process.Environ.Map) !GCS {
        var endpoint_is_default = false;
        const endpoint = environ_map.get("STORAGE_EMULATOR_HOST") orelse environ_map.get("GCS_ENDPOINT_URL") orelse environ_map.get("GOOGLE_CLOUD_STORAGE_ENDPOINT") orelse blk: {
            endpoint_is_default = true;
            break :blk "https://storage.googleapis.com";
        };

        const endpoint_with_scheme = if (std.mem.startsWith(u8, endpoint, "http://") or std.mem.startsWith(u8, endpoint, "https://"))
            endpoint
        else
            try std.fmt.allocPrint(allocator, "http://{s}", .{endpoint});
        defer if (endpoint_with_scheme.ptr != endpoint.ptr) allocator.free(endpoint_with_scheme);

        var region_is_default = false;
        const region = environ_map.get("GCS_REGION") orelse environ_map.get("GOOGLE_CLOUD_STORAGE_REGION") orelse blk: {
            region_is_default = true;
            break :blk "auto";
        };

        const Defaults = struct {
            endpoint_url: []const u8,
            region: []const u8,
            endpoint_is_default: bool,
            region_is_default: bool,
        };

        const defaults: Defaults = .{
            .endpoint_url = endpoint_with_scheme,
            .region = region,
            .endpoint_is_default = endpoint_is_default,
            .region_is_default = region_is_default,
        };

        if (environ_map.get("GOOGLE_APPLICATION_CREDENTIALS")) |path| {
            if (try credentialFileKind(allocator, inner, path)) |kind| switch (kind) {
                .hmac => |hmac| {
                    logAutoInit(.hmac_from_path, path, defaults);
                    return initHmac(allocator, inner, http_client, .{
                        .access_key = hmac.access_key,
                        .secret_key = hmac.secret_key,
                        .endpoint_url = endpoint_with_scheme,
                        .region = region,
                    });
                },
                .authorized_user => {
                    logAutoInit(.adc_oauth_from_path, path, defaults);
                    return initAuthorizedUserADC(allocator, inner, http_client, path, endpoint_with_scheme, region);
                },
            };
        }

        if (try defaultAdcPath(allocator, environ_map)) |path| {
            defer allocator.free(path);
            if (try credentialFileKind(allocator, inner, path)) |kind| switch (kind) {
                .hmac => |hmac| {
                    logAutoInit(.hmac_from_path, path, defaults);
                    return initHmac(allocator, inner, http_client, .{
                        .access_key = hmac.access_key,
                        .secret_key = hmac.secret_key,
                        .endpoint_url = endpoint_with_scheme,
                        .region = region,
                    });
                },
                .authorized_user => {
                    logAutoInit(.adc_oauth_from_path, path, defaults);
                    return initAuthorizedUserADC(allocator, inner, http_client, path, endpoint_with_scheme, region);
                },
            };
        }

        if ((isOnGCP(inner) catch false) and (fetchMetadataServerToken(allocator, inner) catch null != null)) {
            logAutoInit(.metadata_server_oauth, null, defaults);
            return initMetadataServer(allocator, inner, http_client, endpoint_with_scheme, region);
        }

        logAutoInit(.unauthenticated, null, defaults);

        return init(allocator, inner, http_client, .{
            .endpoint_url = endpoint_with_scheme,
            .region = region,
        });
    }

    fn logAutoInit(comptime kind: enum {
        hmac_from_path,
        adc_oauth_from_path,
        metadata_server_oauth,
        unauthenticated,
    }, path: ?[]const u8, defaults: anytype) void {
        switch (kind) {
            .hmac_from_path => log.info("GCS initialized with HMAC credentials from {s}.", .{path.?}),
            .adc_oauth_from_path => log.info("GCS initialized with ADC OAuth credentials from {s}.", .{path.?}),
            .metadata_server_oauth => log.info("GCS initialized with metadata server OAuth credentials.", .{}),
            .unauthenticated => log.info("GCS initialized without credentials. Access will be unauthenticated.", .{}),
        }

        if (defaults.endpoint_is_default) {
            log.info("Using default endpoint: {s}.", .{defaults.endpoint_url});
        }
        if (defaults.region_is_default) {
            log.info("Using default region: {s}.", .{defaults.region});
        }
    }

    const CredentialFileKind = union(enum) {
        hmac: struct {
            access_key: []const u8,
            secret_key: []const u8,
        },
        authorized_user,
    };

    fn credentialFileKind(allocator: std.mem.Allocator, runtime_io: std.Io, path: []const u8) !?CredentialFileKind {
        const file_data = std.Io.Dir.cwd().readFileAlloc(runtime_io, path, allocator, .limited(1024 * 1024)) catch return null;
        defer allocator.free(file_data);

        const Parsed = struct {
            type: ?[]const u8 = null,
            access_key_id: ?[]const u8 = null,
            secret_access_key: ?[]const u8 = null,
            client_id: ?[]const u8 = null,
            client_secret: ?[]const u8 = null,
            refresh_token: ?[]const u8 = null,
        };

        const parsed = std.json.parseFromSlice(Parsed, allocator, file_data, .{
            .ignore_unknown_fields = true,
        }) catch return null;
        defer parsed.deinit();

        if (parsed.value.access_key_id != null and parsed.value.secret_access_key != null) {
            return .{
                .hmac = .{
                    .access_key = try allocator.dupe(u8, parsed.value.access_key_id.?),
                    .secret_key = try allocator.dupe(u8, parsed.value.secret_access_key.?),
                },
            };
        }

        if (parsed.value.type != null and std.mem.eql(u8, parsed.value.type.?, "authorized_user") and
            parsed.value.client_id != null and
            parsed.value.client_secret != null and
            parsed.value.refresh_token != null)
        {
            return .authorized_user;
        }

        return null;
    }

    fn refreshAuthorizedUserTokenFromFile(
        allocator: std.mem.Allocator,
        runtime_io: std.Io,
        path: []const u8,
    ) !OAuthToken {
        const file_data = try std.Io.Dir.cwd().readFileAlloc(runtime_io, path, allocator, .limited(1024 * 1024));
        defer allocator.free(file_data);

        const Parsed = struct {
            type: ?[]const u8 = null,
            client_id: []const u8,
            client_secret: []const u8,
            refresh_token: []const u8,
            quota_project_id: ?[]const u8 = null,
        };

        const parsed = try std.json.parseFromSlice(Parsed, allocator, file_data, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();

        if (parsed.value.type == null or !std.mem.eql(u8, parsed.value.type.?, "authorized_user")) {
            return error.UnsupportedCredentialFile;
        }

        return refreshAuthorizedUserToken(
            allocator,
            runtime_io,
            parsed.value.client_id,
            parsed.value.client_secret,
            parsed.value.refresh_token,
            parsed.value.quota_project_id,
        );
    }

    fn refreshAuthorizedUserToken(
        allocator: std.mem.Allocator,
        runtime_io: std.Io,
        client_id: []const u8,
        client_secret: []const u8,
        refresh_token: []const u8,
        quota_project_id: ?[]const u8,
    ) !OAuthToken {
        var body_writer: std.Io.Writer.Allocating = .init(allocator);
        defer body_writer.deinit();

        try body_writer.writer.writeAll("grant_type=refresh_token&client_id=");
        try std.Uri.Component.percentEncode(&body_writer.writer, client_id, gcsEncodeIsValid);
        try body_writer.writer.writeAll("&client_secret=");
        try std.Uri.Component.percentEncode(&body_writer.writer, client_secret, gcsEncodeIsValid);
        try body_writer.writer.writeAll("&refresh_token=");
        try std.Uri.Component.percentEncode(&body_writer.writer, refresh_token, gcsEncodeIsValid);

        const uri = try std.Uri.parse("https://oauth2.googleapis.com/token");

        var client: std.http.Client = .{
            .allocator = allocator,
            .io = runtime_io,
        };
        defer client.deinit();

        const request_body = body_writer.written();

        var response_writer: std.Io.Writer.Allocating = .init(allocator);
        defer response_writer.deinit();

        const result = try client.fetch(.{
            .location = .{ .uri = uri },
            .method = .POST,
            .payload = request_body,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .content_type = .{ .override = "application/x-www-form-urlencoded" },
            },
            .response_writer = &response_writer.writer,
        });

        if (result.status != .ok) {
            log.err("Failed to refresh ADC token: {s}", .{response_writer.written()});
            return error.RequestFailed;
        }

        const TokenResponse = struct {
            access_token: []const u8,
        };

        const token = try std.json.parseFromSlice(TokenResponse, allocator, response_writer.written(), .{
            .ignore_unknown_fields = true,
        });
        defer token.deinit();

        return .{
            .access_token = try allocator.dupe(u8, token.value.access_token),
            .quota_project_id = if (quota_project_id) |project| try allocator.dupe(u8, project) else null,
            .source = .adc_file,
        };
    }

    fn fetchMetadataServerToken(allocator: std.mem.Allocator, runtime_io: std.Io) !?OAuthToken {
        const uri = std.Uri.parse("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token") catch return null;

        var client: std.http.Client = .{
            .allocator = allocator,
            .io = runtime_io,
        };
        defer client.deinit();

        var response_writer: std.Io.Writer.Allocating = .init(allocator);
        defer response_writer.deinit();

        const result = client.fetch(.{
            .location = .{ .uri = uri },
            .method = .GET,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
            },
            .extra_headers = &.{
                .{ .name = "Metadata-Flavor", .value = "Google" },
            },
            .response_writer = &response_writer.writer,
        }) catch return null;

        if (result.status != .ok) {
            return null;
        }

        const TokenResponse = struct {
            access_token: []const u8,
        };

        const token = std.json.parseFromSlice(TokenResponse, allocator, response_writer.written(), .{
            .ignore_unknown_fields = true,
        }) catch return null;
        defer token.deinit();

        return .{
            .access_token = try allocator.dupe(u8, token.value.access_token),
            .quota_project_id = null,
            .source = .metadata_server,
        };
    }

    fn defaultAdcPath(allocator: std.mem.Allocator, environ_map: *std.process.Environ.Map) !?[]const u8 {
        switch (builtin.os.tag) {
            .windows => {
                const appdata = environ_map.get("APPDATA") orelse return null;
                return try std.Io.Dir.path.join(allocator, &.{ appdata, "gcloud", "application_default_credentials.json" });
            },
            else => {
                const home = environ_map.get("HOME") orelse return null;
                return try std.Io.Dir.path.join(allocator, &.{ home, ".config", "gcloud", "application_default_credentials.json" });
            },
        }
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

        if (self.config.access_key) |access_key| self.allocator.free(access_key);
        if (self.config.secret_key) |secret_key| self.allocator.free(secret_key);
        if (self.config.oauth_access_token) |oauth_access_token| self.allocator.free(oauth_access_token);
        if (self.config.quota_project_id) |quota_project_id| self.allocator.free(quota_project_id);
        self.allocator.free(self.config.endpoint_url);
        self.allocator.free(self.config.region);
        self.allocator.free(self.config.auth_service);
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
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        switch (operation) {
            .file_read_streaming => |o| {
                const handle = self.getFileHandle(o.file);
                const total = self.performRead(handle, o.data, handle.pos) catch |err| {
                    log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, handle.pos, err });
                    return .{ .file_read_streaming = std.Io.File.ReadStreamingError.EndOfStream };
                };

                if (total == 0) {
                    return .{ .file_read_streaming = std.Io.File.ReadStreamingError.EndOfStream };
                }

                handle.pos += @intCast(total);
                return .{ .file_read_streaming = total };
            },
            .file_write_streaming, .device_io_control => {
                return self.base.inner.vtable.operate(self.base.inner.userdata, operation);
            },
        }
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, path, 0) catch return std.Io.Dir.OpenError.Unexpected;

        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
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
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch |err| switch (err) {
            error.FileNotFound => return std.Io.File.OpenError.FileNotFound,
            error.BadPathName => return std.Io.File.OpenError.BadPathName,
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
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch |err| switch (err) {
            error.FileNotFound => return std.Io.File.OpenError.FileNotFound,
            error.BadPathName => return std.Io.File.OpenError.BadPathName,
            else => return std.Io.File.OpenError.Unexpected,
        };

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;
        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, path, size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));

        if (reader.state == .finished) return 0;

        if (reader.state == .reset) {
            if (self.dir_read_states.fetchRemove(reader)) |kv| {
                for (kv.value.objects) |obj| self.allocator.free(obj);
                self.allocator.free(kv.value.objects);
            }

            const handle = self.getDirHandle(reader.dir);
            const objects = self.listObjects(handle.uri) catch return std.Io.Dir.Reader.Error.Unexpected;

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
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
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
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.Reader.Error.Unexpected;
        };
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
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
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *GCS = @fieldParentPtr("base", VFSBase.as(userdata));
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

    fn pathComponents(self: *GCS, path_: []const u8) struct { []const u8, []const u8, []const u8 } {
        const endpoint = std.mem.trimEnd(u8, self.config.endpoint_url, "/");
        const path = std.mem.trim(u8, path_, "/");

        if (std.mem.findScalar(u8, path, '/')) |idx| {
            return .{ endpoint, path[0..idx], if (idx + 1 < path.len) path[idx + 1 ..] else "" };
        } else {
            return .{ endpoint, path, "" };
        }
    }

    fn gcsUrl(self: *GCS, path: []const u8, buf: []u8) ![]const u8 {
        const endpoint, const bucket, const key = self.pathComponents(path);
        return try std.fmt.bufPrint(buf, "{s}/{s}/{s}", .{ endpoint, bucket, key });
    }

    fn authHeaders(self: *GCS, method: std.http.Method, uri: std.Uri, timestamp: []const u8, authorization_buffer: []u8) !struct {
        authorization: ?[]const u8,
        extra_headers: [3]?std.http.Header,
        extra_len: usize,
    } {
        if (self.config.oauth_access_token) |token| {
            const authorization = try std.fmt.bufPrint(authorization_buffer, "Bearer {s}", .{token});
            var extra_headers: [3]?std.http.Header = .{ null, null, null };
            var extra_len: usize = 0;
            if (self.config.quota_project_id) |project| {
                extra_headers[0] = .{ .name = "x-goog-user-project", .value = project };
                extra_len = 1;
            }
            return .{
                .authorization = authorization,
                .extra_headers = extra_headers,
                .extra_len = extra_len,
            };
        }

        const signer: GcsSigV4 = .{
            .access_key = self.config.access_key,
            .secret_key = self.config.secret_key,
            .region = self.config.region,
            .service = self.config.auth_service,
        };

        const authorization = try signer.generateAuthHeader(authorization_buffer, method, uri, timestamp, &.{});
        var extra_headers: [3]?std.http.Header = .{
            .{ .name = "x-goog-date", .value = timestamp },
            .{ .name = "x-goog-content-sha256", .value = GcsSigV4.UNSIGNED_PAYLOAD },
            null,
        };
        var extra_len: usize = 2;
        if (self.config.quota_project_id) |project| {
            extra_headers[2] = .{ .name = "x-goog-user-project", .value = project };
            extra_len = 3;
        }
        return .{
            .authorization = authorization,
            .extra_headers = extra_headers,
            .extra_len = extra_len,
        };
    }

    fn listObjects(self: *GCS, prefix: []const u8) ![][]const u8 {
        const endpoint, const bucket, const key_prefix = self.pathComponents(prefix);

        var query_buf: [4096]u8 = undefined;
        var query_writer = std.Io.Writer.fixed(&query_buf);

        try query_writer.writeAll("delimiter=");
        try std.Uri.Component.percentEncode(&query_writer, "/", gcsEncodeIsValid);

        if (key_prefix.len > 0) {
            try query_writer.writeAll("&prefix=");
            try std.Uri.Component.percentEncode(&query_writer, key_prefix, gcsEncodeIsValid);
            try std.Uri.Component.percentEncode(&query_writer, "/", gcsEncodeIsValid);
        }

        const endpoint_uri = try std.Uri.parse(endpoint);

        var path_buf: [256]u8 = undefined;

        const uri: std.Uri = .{
            .scheme = endpoint_uri.scheme,
            .host = endpoint_uri.host,
            .port = endpoint_uri.port,
            .path = .{ .percent_encoded = try std.fmt.bufPrint(&path_buf, "/{s}", .{bucket}) },
            .query = .{ .percent_encoded = query_writer.buffered() },
            .fragment = null,
        };

        var timestamp_buf: [16]u8 = undefined;
        const timestamp = try self.getTimestamp(&timestamp_buf);

        var authorization_buffer: [1024]u8 = undefined;
        const auth = try self.authHeaders(.GET, uri, timestamp, &authorization_buffer);

        var extra_headers_buf: [3]std.http.Header = undefined;
        for (0..auth.extra_len) |i| {
            extra_headers_buf[i] = auth.extra_headers[i].?;
        }

        var req = try self.client.request(.GET, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = if (auth.authorization) |value| .{ .override = value } else .omit,
            },
            .extra_headers = extra_headers_buf[0..auth.extra_len],
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [2 * 1024]u8 = undefined;
        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .ok) {
            log.err("Failed to list object {f}", .{uri});
            log.err("{s}", .{res.head.bytes});
            return error.RequestFailed;
        }

        const body = try res.reader(&.{}).readAlloc(self.allocator, res.head.content_length orelse 1024 * 1024);
        defer self.allocator.free(body);

        return try self.parseListObjectsResponse(body, key_prefix);
    }

    fn gcsEncodeIsValid(c: u8) bool {
        return switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '.', '_', '~' => true,
            else => false,
        };
    }

    fn parseListObjectsResponse(self: *GCS, xml: []const u8, prefix: []const u8) ![][]const u8 {
        var results = std.ArrayListUnmanaged([]const u8){};
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

    fn fetchSize(self: *GCS, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        var url_buf: [8 * 1024]u8 = undefined;

        const path = try self.resolvePath(dir, sub_path, &path_buffer);
        const url = try self.gcsUrl(path, &url_buf);

        const uri = std.Uri.parse(url) catch return error.BadPathName;

        var timestamp_buf: [16]u8 = undefined;
        const timestamp = try self.getTimestamp(&timestamp_buf);

        var authorization_buffer: [1024]u8 = undefined;
        const auth = try self.authHeaders(.HEAD, uri, timestamp, &authorization_buffer);

        var extra_headers_buf: [3]std.http.Header = undefined;
        for (0..auth.extra_len) |i| {
            extra_headers_buf[i] = auth.extra_headers[i].?;
        }

        var req = try self.client.request(.HEAD, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = if (auth.authorization) |value| .{ .override = value } else .omit,
            },
            .extra_headers = extra_headers_buf[0..auth.extra_len],
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        const res = try req.receiveHead(&redirect_buffer);

        const size = switch (res.head.status.class()) {
            .success => res.head.content_length.?,
            else => switch (res.head.status) {
                .not_found => return error.FileNotFound,
                else => blk: {
                    log.err("Failed to fetch size for {s}: {s}", .{ url, res.head.bytes });
                    break :blk error.ServerError;
                },
            },
        };
        return size;
    }

    fn performRead(self: *GCS, handle: *Handle, data: []const []u8, offset: u64) !usize {
        if (offset >= handle.size) return 0;

        var url_buf: [8 * 1024]u8 = undefined;
        const url = try self.gcsUrl(handle.uri, &url_buf);

        const uri = std.Uri.parse(url) catch return error.BadPathName;

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

        var timestamp_buf: [16]u8 = undefined;
        const timestamp = try self.getTimestamp(&timestamp_buf);

        var authorization_buffer: [1024]u8 = undefined;
        const auth = try self.authHeaders(.GET, uri, timestamp, &authorization_buffer);

        var extra_headers_buf: [4]std.http.Header = undefined;
        for (0..auth.extra_len) |i| {
            extra_headers_buf[i] = auth.extra_headers[i].?;
        }
        extra_headers_buf[auth.extra_len] = .{ .name = "Range", .value = range_header };

        var req = try self.client.request(.GET, uri, .{ .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = if (auth.authorization) |value| .{ .override = value } else .omit,
        }, .extra_headers = extra_headers_buf[0 .. auth.extra_len + 1] });
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
