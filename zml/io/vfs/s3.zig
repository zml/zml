const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/s3");

pub const AwsSigV4 = struct {
    access_key: []const u8,
    secret_key: []const u8,
    region: []const u8,
    service: []const u8 = "s3",

    const ALGORITHM = "AWS4-HMAC-SHA256";
    pub const UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD";

    pub fn generateAuthHeader(
        self: AwsSigV4,
        allocator: std.mem.Allocator,
        method: std.http.Method,
        endpoint: []const u8,
        uri: std.Uri,
        timestamp: []const u8,
        extra_headers: []const std.http.Header,
    ) ![]const u8 {
        const date = timestamp[0..8];

        var buf: [256]u8 = undefined;
        const host = getHostForEndpoint(endpoint, &buf);

        // Build sorted headers list
        var headers: std.ArrayList(std.http.Header) = .{};
        defer headers.deinit(allocator);

        try headers.append(allocator, .{ .name = "host", .value = host });
        try headers.append(allocator, .{ .name = "x-amz-content-sha256", .value = UNSIGNED_PAYLOAD });
        try headers.append(allocator, .{ .name = "x-amz-date", .value = timestamp });

        for (extra_headers) |h| {
            try headers.append(allocator, .{ .name = h.name, .value = h.value });
        }

        // Important: sort by header name
        std.mem.sort(
            std.http.Header,
            headers.items,
            {},
            struct {
                fn lessThan(_: void, a: std.http.Header, b: std.http.Header) bool {
                    return std.mem.order(u8, a.name, b.name) == .lt;
                }
            }.lessThan,
        );

        // Build canonical headers and signed headers
        var canonical_headers_list: std.ArrayList(u8) = .{};
        defer canonical_headers_list.deinit(allocator);

        var signed_headers_list: std.ArrayList(u8) = .{};
        defer signed_headers_list.deinit(allocator);

        for (headers.items, 0..) |header, i| {
            // Add to canonical headers: "name:value\n"
            try canonical_headers_list.appendSlice(allocator, header.name);
            try canonical_headers_list.append(allocator, ':');
            try canonical_headers_list.appendSlice(allocator, header.value);
            try canonical_headers_list.append(allocator, '\n');

            // Add to signed headers list: "name1;name2;name3"
            if (i > 0) try signed_headers_list.append(allocator, ';');
            try signed_headers_list.appendSlice(allocator, header.name);
        }

        const canonical_headers = canonical_headers_list.items;
        const signed_headers = signed_headers_list.items;

        // Build canonical request
        var canonical_request_list: std.ArrayList(u8) = .{};
        defer canonical_request_list.deinit(allocator);

        const canonical_uri = if (uri.path.percent_encoded.len > 0) uri.path.percent_encoded else "/";
        const canonical_query = ""; // todo

        // METHOD\nURI\nQUERY\nHEADERS\nSIGNED_HEADERS\nPAYLOAD_HASH
        try canonical_request_list.appendSlice(allocator, @tagName(method));
        try canonical_request_list.append(allocator, '\n');
        try canonical_request_list.appendSlice(allocator, canonical_uri);
        try canonical_request_list.append(allocator, '\n');
        try canonical_request_list.appendSlice(allocator, canonical_query);
        try canonical_request_list.append(allocator, '\n');
        try canonical_request_list.appendSlice(allocator, canonical_headers);
        try canonical_request_list.append(allocator, '\n');
        try canonical_request_list.appendSlice(allocator, signed_headers);
        try canonical_request_list.append(allocator, '\n');
        try canonical_request_list.appendSlice(allocator, UNSIGNED_PAYLOAD);

        // Hash canonical request
        const cr_hash = sha256(canonical_request_list.items);
        var cr_hash_hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&cr_hash_hex, "{x}", .{&cr_hash}) catch unreachable;

        // Build credential scope: date/region/service/aws4_request
        const credential_scope = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}/aws4_request", .{ date, self.region, self.service });
        defer allocator.free(credential_scope);

        // Build string to sign
        var string_to_sign_list: std.ArrayList(u8) = .{};
        defer string_to_sign_list.deinit(allocator);

        try string_to_sign_list.appendSlice(allocator, ALGORITHM);
        try string_to_sign_list.append(allocator, '\n');
        try string_to_sign_list.appendSlice(allocator, timestamp);
        try string_to_sign_list.append(allocator, '\n');
        try string_to_sign_list.appendSlice(allocator, credential_scope);
        try string_to_sign_list.append(allocator, '\n');
        try string_to_sign_list.appendSlice(allocator, &cr_hash_hex);

        // Calculate signing key
        const k_date = hmacSha256WithPrefix("AWS4", self.secret_key, date);
        const k_region = hmacSha256(&k_date, self.region);
        const k_service = hmacSha256(&k_region, self.service);
        const k_signing = hmacSha256(&k_service, "aws4_request");

        // Calculate signature
        const signature = hmacSha256(&k_signing, string_to_sign_list.items);
        var signature_hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&signature_hex, "{x}", .{&signature}) catch unreachable;

        // Build final Authorization header
        return try std.fmt.allocPrint(allocator, "{s} Credential={s}/{s}, SignedHeaders={s}, Signature={s}", .{
            ALGORITHM,
            self.access_key,
            credential_scope,
            signed_headers,
            signature_hex[0..64],
        });
    }

    fn getHostForEndpoint(endpoint: []const u8, buf: []u8) []const u8 {
        const uri = std.Uri.parse(endpoint) catch return endpoint;
        const host_raw = uri.host orelse return endpoint;
        const host = switch (host_raw) {
            .raw => |r| r,
            .percent_encoded => |p| p,
        };
        if (uri.port) |port| {
            const len = std.fmt.bufPrint(buf, "{s}:{d}", .{ host, port }) catch return endpoint;
            return buf[0..len.len];
        }
        return host;
    }

    fn hmacSha256WithPrefix(prefix: []const u8, secret: []const u8, data: []const u8) [32]u8 {
        var key_buf: [128]u8 = undefined;
        @memcpy(key_buf[0..prefix.len], prefix);
        @memcpy(key_buf[prefix.len..][0..secret.len], secret);
        const key = key_buf[0 .. prefix.len + secret.len];

        var out: [32]u8 = undefined;
        std.crypto.auth.hmac.sha2.HmacSha256.create(&out, data, key);
        return out;
    }

    fn sha256(data: []const u8) [32]u8 {
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(data, &hash, .{});
        return hash;
    }

    fn hmacSha256(key: []const u8, data: []const u8) [32]u8 {
        var out: [32]u8 = undefined;
        std.crypto.auth.hmac.sha2.HmacSha256.create(&out, data, key);
        return out;
    }
};

pub const S3 = struct {
    const Handle = struct {
        pub const Type = enum { file, directory };

        type: Type,
        uri: []const u8,
        pos: u64,
        size: u64,

        pub fn init(allocator: std.mem.Allocator, handle_type: Type, path: []const u8, size: u64) !Handle {
            return .{ .type = handle_type, .uri = try allocator.dupe(u8, path), .pos = 0, .size = size };
        }

        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.uri);
        }
    };

    pub const Credentials = struct {
        access_key: []const u8,
        secret_key: []const u8,
        endpoint_url: []const u8,
        region: []const u8 = "us-east-1", // todo: no default?

        pub fn fromEnv(allocator: std.mem.Allocator) !Credentials {
            const access_key = std.process.getEnvVarOwned(allocator, "AWS_ACCESS_KEY_ID") catch |err| {
                log.err("AWS_ACCESS_KEY_ID environment variable not set", .{});
                return err;
            };
            errdefer allocator.free(access_key);

            const secret_key = std.process.getEnvVarOwned(allocator, "AWS_SECRET_ACCESS_KEY") catch |err| {
                log.err("AWS_SECRET_ACCESS_KEY environment variable not set", .{});
                return err;
            };
            errdefer allocator.free(secret_key);

            const endpoint_url = std.process.getEnvVarOwned(allocator, "AWS_ENDPOINT_URL") catch
                std.process.getEnvVarOwned(allocator, "AWS_ENDPOINT_URL_S3") catch |err| {
                log.err("Neither AWS_ENDPOINT_URL nor AWS_ENDPOINT_URL_S3 environment variable is set", .{});
                return err;
            };
            errdefer allocator.free(endpoint_url);

            const region = std.process.getEnvVarOwned(allocator, "AWS_REGION") catch
                std.process.getEnvVarOwned(allocator, "AWS_DEFAULT_REGION") catch
                try allocator.dupe(u8, "us-east-1");

            return .{ .access_key = access_key, .secret_key = secret_key, .endpoint_url = endpoint_url, .region = region };
        }

        pub fn deinit(self: *Credentials, allocator: std.mem.Allocator) void {
            allocator.free(self.access_key);
            allocator.free(self.secret_key);
            allocator.free(self.endpoint_url);
            allocator.free(self.region);
        }
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    client: *std.http.Client,
    access_key: []const u8,
    secret_key: []const u8,
    endpoint_url: []const u8,
    region: []const u8,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .{},
    base: VFSBase,

    pub fn init(
        allocator: std.mem.Allocator,
        inner: std.Io,
        http_client: *std.http.Client,
        credentials: Credentials,
    ) !S3 {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .access_key = try allocator.dupe(u8, credentials.access_key),
            .secret_key = try allocator.dupe(u8, credentials.secret_key),
            .endpoint_url = try allocator.dupe(u8, credentials.endpoint_url),
            .region = try allocator.dupe(u8, credentials.region),
        };
    }

    pub fn fromEnv(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client) !S3 {
        var credentials = try Credentials.fromEnv(allocator);
        defer credentials.deinit(allocator);

        return init(allocator, inner, http_client, credentials);
    }

    pub fn deinit(self: *S3) void {
        var idx: usize = 0;
        while (idx < self.handles.len) : (idx += 1) {
            const is_closed = for (self.closed_handles.items) |closed_idx| {
                if (closed_idx == idx) break true;
            } else false;
            if (!is_closed) self.handles.at(idx).deinit(self.allocator);
        }
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);
        self.allocator.free(self.access_key);
        self.allocator.free(self.secret_key);
        self.allocator.free(self.endpoint_url);
        self.allocator.free(self.region);
    }

    pub fn io(self: *S3) std.Io {
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

    fn openHandle(self: *S3) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        if (self.closed_handles.pop()) |idx| return .{ idx, self.handles.at(idx) };
        return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
    }

    fn closeHandle(self: *S3, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        self.handles.at(idx).deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    fn getFileHandle(self: *S3, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(file.handle));
    }

    fn getDirHandle(self: *S3, dir: std.Io.Dir) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(dir.handle));
    }

    fn resolvePath(self: *S3, dir: std.Io.Dir, sub_path: []const u8, out_buffer: []u8) ![]u8 {
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

    // todo
    fn parseBucketKey(path: []const u8) struct { bucket: []const u8, key: []const u8 } {
        const trimmed = std.mem.trim(u8, path, "/");
        if (std.mem.indexOfScalar(u8, trimmed, '/')) |idx| {
            return .{ .bucket = trimmed[0..idx], .key = if (idx + 1 < trimmed.len) trimmed[idx + 1 ..] else "" };
        }
        return .{ .bucket = trimmed, .key = "" };
    }

    // todo
    const UrlStyle = enum {
        virtual_hosted, // AWS
        path_style, // MinIO
    };

    // todo
    fn buildS3Url(self: *S3, allocator: std.mem.Allocator, bucket: []const u8, key: []const u8) ![]const u8 {
        const trimmed_endpoint = std.mem.trimEnd(u8, self.endpoint_url, "/");

        if (key.len > 0) {
            return try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ trimmed_endpoint, bucket, key });
        }
        return try std.fmt.allocPrint(allocator, "{s}/{s}/", .{ trimmed_endpoint, bucket });
    }

    // todo
    fn getTimestamp(allocator: std.mem.Allocator, io_: std.Io) ![]const u8 {
        const now_ts = try std.Io.Clock.now(.real, io_);

        const now: u64 = @intCast(@divFloor(now_ts.nanoseconds, std.time.ns_per_s));
        const epoch_secs: std.time.epoch.EpochSeconds = .{ .secs = now };
        const year_day = epoch_secs.getEpochDay().calculateYearDay();
        const month_day = year_day.calculateMonthDay();
        const day_seconds = epoch_secs.getDaySeconds();

        return try std.fmt.allocPrint(allocator, "{d:0>4}{d:0>2}{d:0>2}T{d:0>2}{d:0>2}{d:0>2}Z", .{
            year_day.year,
            @intFromEnum(month_day.month),
            month_day.day_index + 1,
            day_seconds.getHoursIntoDay(),
            day_seconds.getMinutesIntoHour(),
            day_seconds.getSecondsIntoMinute(),
        });
    }

    fn fetchSize(self: *S3, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = try self.resolvePath(dir, sub_path, &path_buffer);
        const parsed = parseBucketKey(path);

        const url = try self.buildS3Url(self.allocator, parsed.bucket, parsed.key);
        defer self.allocator.free(url);

        const uri = std.Uri.parse(url) catch return error.BadPathName;

        const timestamp = try getTimestamp(self.allocator, self.base.inner);
        defer self.allocator.free(timestamp);

        const signer: AwsSigV4 = .{
            .access_key = self.access_key,
            .secret_key = self.secret_key,
            .region = self.region,
        };

        const auth_header = try signer.generateAuthHeader(self.allocator, .HEAD, self.endpoint_url, uri, timestamp, &.{});
        defer self.allocator.free(auth_header);

        var req = try self.client.request(.HEAD, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{
                .{ .name = "Authorization", .value = auth_header },
                .{ .name = "x-amz-date", .value = timestamp },
                .{ .name = "x-amz-content-sha256", .value = AwsSigV4.UNSIGNED_PAYLOAD },
            },
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        const res = try req.receiveHead(&redirect_buffer);

        switch (res.head.status.class()) {
            .success => return res.head.content_length.?,
            else => {
                log.err("Failed to fetch size for {s}: {s}", .{ url, res.head.bytes });
                return error.ServerError;
            },
        }
    }

    fn performRead(self: *S3, handle: *Handle, data: []const []u8, offset: u64) !usize {
        if (offset >= handle.size) return 0;

        const parsed = parseBucketKey(handle.uri);

        const url = try self.buildS3Url(self.allocator, parsed.bucket, parsed.key);
        defer self.allocator.free(url);

        const uri = std.Uri.parse(url) catch return error.BadPathName;

        // Calculate range
        var total_bytes: u64 = 0;
        for (data) |buf| total_bytes += buf.len;
        const remaining = handle.size - offset;
        const take = @min(remaining, total_bytes);
        const end = offset + take - 1;

        const range_header = try std.fmt.allocPrint(self.allocator, "bytes={d}-{d}", .{ offset, end });
        defer self.allocator.free(range_header);

        const timestamp = try getTimestamp(self.allocator, self.base.inner);
        defer self.allocator.free(timestamp);

        const signer: AwsSigV4 = .{
            .access_key = self.access_key,
            .secret_key = self.secret_key,
            .region = self.region,
        };

        const auth_header = try signer.generateAuthHeader(self.allocator, .GET, self.endpoint_url, uri, timestamp, &.{});
        defer self.allocator.free(auth_header);

        var req = try self.client.request(.GET, uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{
                .{ .name = "Authorization", .value = auth_header },
                .{ .name = "x-amz-date", .value = timestamp },
                .{ .name = "x-amz-content-sha256", .value = AwsSigV4.UNSIGNED_PAYLOAD },
                .{ .name = "Range", .value = range_header },
            },
        });
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
        // Format: "bytes start-end/total"
        const space = std.mem.indexOfScalar(u8, value, ' ') orelse return null;
        const dash = std.mem.indexOfScalar(u8, value, '-') orelse return null;
        const slash = std.mem.indexOfScalar(u8, value, '/') orelse return null;

        return .{
            .start = std.fmt.parseInt(u64, value[space + 1 .. dash], 10) catch return null,
            .end = std.fmt.parseInt(u64, value[dash + 1 .. slash], 10) catch return null,
            .total = std.fmt.parseInt(u64, value[slash + 1 ..], 10) catch return null,
        };
    }

    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;
        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, path, 0) catch return std.Io.Dir.OpenError.Unexpected;
        return .{ .handle = @intCast(idx) };
    }

    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        return .{
            .inode = 0,
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
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch return std.Io.Dir.StatFileError.Unexpected;
        return .{
            .inode = 0,
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
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch return std.Io.File.OpenError.Unexpected;
        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;
        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, path, size) catch return std.Io.File.OpenError.Unexpected;
        return .{ .handle = @intCast(idx) };
    }

    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| self.closeHandle(@intCast(dir.handle)) catch unreachable;
    }

    fn dirRead(_: ?*anyopaque, _: *std.Io.Dir.Reader, _: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        log.err("dirRead unsupported for S3 VFS", .{});
        return std.Io.Dir.Reader.Error.Unexpected;
    }

    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
    }

    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
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
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.getFileHandle(file).size;
    }

    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| self.closeHandle(@intCast(file.handle)) catch unreachable;
    }

    fn fileReadStreaming(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8) std.Io.File.Reader.Error!usize {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const total = self.performRead(handle, data, handle.pos) catch |err| {
            log.err("Failed to read {s} at {d}: {any}", .{ handle.uri, handle.pos, err });
            return std.Io.File.Reader.Error.Unexpected;
        };
        handle.pos += @intCast(total);
        return total;
    }

    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to read {s} at {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.ReadPositionalError.Unexpected;
        };
    }

    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
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
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.File.RealPathError.SystemResources;
        return path.len;
    }
};
