const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/io/vfs/s3");

// todo: move to zml.stdx.crypto
pub fn MacWriter(comptime T: type) type {
    return struct {
        const Self = @This();

        mac: *T,
        interface: std.Io.Writer,

        pub fn init(mac_: *T) MacWriter(T) {
            return .{
                .mac = mac_,
                .interface = .{
                    .buffer = &.{},
                    .vtable = &.{
                        .drain = drain,
                    },
                },
            };
        }

        pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
            const self: *Self = @alignCast(@fieldParentPtr("interface", w));
            var total: usize = 0;
            for (data) |chunk| {
                self.mac.update(chunk);
                total += chunk.len;
            }
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                self.mac.update(last);
                total += last.len;
            }
            return total;
        }
    };
}

pub fn hmacWriter(hmac: anytype) MacWriter(std.meta.Child(@TypeOf(hmac))) {
    return .init(hmac);
}

pub const AwsSigV4 = struct {
    access_key: ?[]const u8,
    secret_key: ?[]const u8,
    region: []const u8,
    service: []const u8,

    const ALGORITHM = "AWS4-HMAC-SHA256";
    pub const UNSIGNED_PAYLOAD = "UNSIGNED-PAYLOAD";

    pub fn generateAuthHeader(
        self: AwsSigV4,
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
            const host_slice = switch (host) {
                .raw, .percent_encoded => |h| h,
            };
            if (uri.port) |port| {
                var host_buf: [128]u8 = undefined;
                break :blk .{ .name = "host", .value = try std.fmt.bufPrint(&host_buf, "{s}:{d}", .{ host_slice, port }) };
            } else {
                break :blk .{ .name = "host", .value = host_slice };
            }
        };

        header_buf[0] = host_header;
        header_buf[1] = .{ .name = "x-amz-content-sha256", .value = UNSIGNED_PAYLOAD };
        header_buf[2] = .{ .name = "x-amz-date", .value = timestamp };
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
        var sha_writer = hmacWriter(&sha);
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
        const scope = try std.fmt.bufPrint(&scope_buf, "{s}/{s}/{s}/aws4_request", .{ date, self.region, self.service });

        const signing_key = try self.deriveSigningKey(date);

        var h_state = std.crypto.auth.hmac.sha2.HmacSha256.init(&signing_key);
        var h_writer = hmacWriter(&h_state);
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

    fn deriveSigningKey(self: AwsSigV4, date: []const u8) ![32]u8 {
        var key_buf: [128]u8 = undefined;
        const k_init = try std.fmt.bufPrint(&key_buf, "AWS4{s}", .{self.secret_key.?});

        const k_date = hmacStatic(k_init, date);
        const k_region = hmacStatic(&k_date, self.region);
        const k_service = hmacStatic(&k_region, self.service);
        return hmacStatic(&k_service, "aws4_request");
    }

    fn hmacStatic(key: []const u8, data: []const u8) [32]u8 {
        var out: [32]u8 = undefined;
        std.crypto.auth.hmac.sha2.HmacSha256.create(&out, data, key);
        return out;
    }
};

const ReadState = struct { index: usize, objects: [][]const u8 };

pub const S3 = struct {
    pub const Config = struct {
        access_key: ?[]const u8 = null,
        secret_key: ?[]const u8 = null,
        endpoint_url: []const u8,
        region: []const u8,
        auth_service: []const u8 = "s3",
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
    ) !S3 {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .config = .{
                .access_key = if (config.access_key) |k| try allocator.dupe(u8, k) else null,
                .secret_key = if (config.secret_key) |k| try allocator.dupe(u8, k) else null,
                .endpoint_url = try allocator.dupe(u8, config.endpoint_url),
                .region = try allocator.dupe(u8, config.region),
                .auth_service = try allocator.dupe(u8, config.auth_service),
            },
        };
    }

    pub fn auto(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client) !S3 {
        const access_key = std.process.getEnvVarOwned(allocator, "AWS_ACCESS_KEY_ID") catch null;
        const secret_key = std.process.getEnvVarOwned(allocator, "AWS_SECRET_ACCESS_KEY") catch null;

        var endpoint_is_default = false;
        const endpoint = std.process.getEnvVarOwned(allocator, "AWS_ENDPOINT_URL") catch
            std.process.getEnvVarOwned(allocator, "AWS_ENDPOINT_URL_S3") catch blk: {
            endpoint_is_default = true;
            break :blk try allocator.dupe(u8, "https://s3.amazonaws.com");
        };

        var region_is_default = false;
        const region = std.process.getEnvVarOwned(allocator, "AWS_REGION") catch
            std.process.getEnvVarOwned(allocator, "AWS_DEFAULT_REGION") catch blk: {
            region_is_default = true;
            break :blk try allocator.dupe(u8, "us-east-1");
        };

        if (access_key == null or secret_key == null or endpoint_is_default or region_is_default) {
            var buf: [512]u8 = undefined;
            var writer = std.Io.Writer.fixed(&buf);

            if (access_key == null or secret_key == null) {
                try writer.writeAll("S3 initialized without full credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY). Access will be unauthenticated.");
            } else {
                try writer.writeAll("S3 initialized with credentials.");
            }

            if (endpoint_is_default) {
                try writer.print(" Using default endpoint: {s} (set AWS_ENDPOINT_URL or AWS_ENDPOINT_URL_S3).", .{endpoint});
            }

            if (region_is_default) {
                try writer.print(" Using default region: {s} (set AWS_REGION or AWS_DEFAULT_REGION).", .{region});
            }

            log.warn("{s}", .{writer.buffered()});
        }

        defer {
            if (access_key) |k| allocator.free(k);
            if (secret_key) |k| allocator.free(k);
            allocator.free(endpoint);
            allocator.free(region);
        }

        return init(allocator, inner, http_client, .{
            .access_key = access_key,
            .secret_key = secret_key,
            .endpoint_url = endpoint,
            .region = region,
        });
    }

    pub fn deinit(self: *S3) void {
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
        self.allocator.free(self.config.endpoint_url);
        self.allocator.free(self.config.region);
        self.allocator.free(self.config.auth_service);
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
            .inode = @intCast(@intFromPtr(handle)),
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
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *S3 = @fieldParentPtr("base", VFSBase.as(userdata));

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
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
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
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.Reader.Error.Unexpected;
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

    fn getTimestamp(self: *S3, buf: []u8) ![]const u8 {
        const now_ts = try std.Io.Clock.now(.real, self.base.inner);

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

    fn s3EncodeIsValid(c: u8) bool {
        return switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '.', '_', '~' => true,
            else => false,
        };
    }

    fn pathComponents(self: *S3, path_: []const u8) struct { []const u8, []const u8, []const u8 } {
        const endpoint = std.mem.trimEnd(u8, self.config.endpoint_url, "/");
        const path = std.mem.trim(u8, path_, "/");

        if (std.mem.findScalar(u8, path, '/')) |idx| {
            return .{ endpoint, path[0..idx], if (idx + 1 < path.len) path[idx + 1 ..] else "" };
        } else {
            return .{ endpoint, path, "" };
        }
    }

    fn s3Url(self: *S3, path: []const u8, buf: []u8) ![]const u8 {
        const endpoint, const bucket, const key = self.pathComponents(path);
        return try std.fmt.bufPrint(buf, "{s}/{s}/{s}", .{ endpoint, bucket, key });
    }

    fn listObjects(self: *S3, prefix: []const u8) ![][]const u8 {
        const endpoint, const bucket, const key_prefix = self.pathComponents(prefix);

        var query_buf: [4096]u8 = undefined;
        var query_writer = std.Io.Writer.fixed(&query_buf);

        try query_writer.writeAll("delimiter=");
        try std.Uri.Component.percentEncode(&query_writer, "/", s3EncodeIsValid);
        try query_writer.writeAll("&list-type=2&max-keys=1000");

        if (key_prefix.len > 0) {
            try query_writer.writeAll("&prefix=");
            try std.Uri.Component.percentEncode(&query_writer, key_prefix, s3EncodeIsValid);
            try std.Uri.Component.percentEncode(&query_writer, "/", s3EncodeIsValid);
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

        const signer: AwsSigV4 = .{
            .access_key = self.config.access_key,
            .secret_key = self.config.secret_key,
            .region = self.config.region,
            .service = self.config.auth_service,
        };

        var authorization_buffer: [512]u8 = undefined;
        const authorization = try signer.generateAuthHeader(&authorization_buffer, .GET, uri, timestamp, &.{});

        var req = try self.client.request(.GET, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = if (authorization) |auth| .{ .override = auth } else .omit,
            },
            .extra_headers = &.{
                .{ .name = "x-amz-date", .value = timestamp },
                .{ .name = "x-amz-content-sha256", .value = AwsSigV4.UNSIGNED_PAYLOAD },
            },
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

    fn parseListObjectsResponse(self: *S3, xml: []const u8, prefix: []const u8) ![][]const u8 {
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

                // Filter out the directory itself (S3 often includes the prefix as a 0-byte key)
                const is_self = std.mem.eql(u8, std.mem.trimEnd(u8, val, "/"), std.mem.trimEnd(u8, prefix, "/"));
                if (val.len > 0 and !is_self) {
                    try results.append(self.allocator, try self.allocator.dupe(u8, val));
                }
                pos = end + tag.len + 3;
            }
        }

        return try results.toOwnedSlice(self.allocator);
    }

    fn fetchSize(self: *S3, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        var url_buf: [8 * 1024]u8 = undefined;

        const path = try self.resolvePath(dir, sub_path, &path_buffer);
        const url = try self.s3Url(path, &url_buf);

        const uri = std.Uri.parse(url) catch return error.BadPathName;

        var timestamp_buf: [16]u8 = undefined;
        const timestamp = try self.getTimestamp(&timestamp_buf);

        const signer: AwsSigV4 = .{
            .access_key = self.config.access_key,
            .secret_key = self.config.secret_key,
            .region = self.config.region,
            .service = self.config.auth_service,
        };

        var authorization_buffer: [512]u8 = undefined;
        const authorization = try signer.generateAuthHeader(&authorization_buffer, .HEAD, uri, timestamp, &.{});

        var req = try self.client.request(.HEAD, uri, .{
            .redirect_behavior = .not_allowed,
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = if (authorization) |auth| .{ .override = auth } else .omit,
            },
            .extra_headers = &.{
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

        var url_buf: [8 * 1024]u8 = undefined;
        const url = try self.s3Url(handle.uri, &url_buf);

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

        const signer: AwsSigV4 = .{
            .access_key = self.config.access_key,
            .secret_key = self.config.secret_key,
            .region = self.config.region,
            .service = self.config.auth_service,
        };

        var authorization_buffer: [512]u8 = undefined;
        const authorization = try signer.generateAuthHeader(&authorization_buffer, .GET, uri, timestamp, &.{});

        var req = try self.client.request(.GET, uri, .{ .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = if (authorization) |auth| .{ .override = auth } else .omit,
        }, .extra_headers = &.{
            .{ .name = "x-amz-date", .value = timestamp },
            .{ .name = "x-amz-content-sha256", .value = AwsSigV4.UNSIGNED_PAYLOAD },
            .{ .name = "Range", .value = range_header },
        } });
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
};
