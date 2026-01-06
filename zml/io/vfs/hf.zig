const json = @import("std").json;
const std = @import("std");

const log = std.log.scoped(.@"zml/io/vfs/hf");

const HFApi = struct {
    pub const Error = error{
        MalformedURI,
        RequestFailed,
        ResponseParsingFailed,
    } || std.mem.Allocator.Error;

    const PickleImport = struct {
        module: []const u8,
        name: []const u8,
        safety: []const u8,
    };

    const Scan = struct {
        status: ?[]const u8 = null,
        message: ?[]const u8 = null,
        reportLink: ?[]const u8 = null,
        reportLabel: ?[]const u8 = null,
        pickleImports: ?[]PickleImport = null,
        version: ?[]const u8 = null,
    };

    const SecurityFileStatus = struct {
        status: ?[]const u8 = null,
        jFrogScan: ?Scan = null,
        protectAiScan: ?Scan = null,
        avScan: ?Scan = null,
        pickleImportScan: ?Scan = null,
        virusTotalScan: ?Scan = null,
    };

    const LastCommit = struct {
        id: []const u8,
        title: []const u8,
        date: []const u8,
    };

    const Lfs = struct {
        oid: []const u8,
        size: u64,
        pointerSize: u64,
    };

    pub const TreeEntry = struct {
        type: []const u8,
        oid: []const u8,
        path: []const u8,
        size: u64,
        lfs: ?Lfs = null,
        lastCommit: ?LastCommit = null,
        xetHash: ?[]const u8 = null,
        securityFileStatus: ?SecurityFileStatus = null,
    };

    pub const TreeSize = struct {
        path: []const u8,
        size: u64,
    };

    pub const Model = struct {
        namespace: []const u8,
        repo: []const u8,
        revision: ?[]const u8,
    };

    pub fn modelFromUri(uri: []const u8) Error!Model {
        const parsed = std.Uri.parse(uri) catch {
            log.err("Failed to parse HF model URI: {s}", .{uri});
            return Error.MalformedURI;
        };

        const namespace = parsed.host.?.percent_encoded;

        var path = parsed.path.percent_encoded;
        // strip leading slash
        if (path.len > 0 and path[0] == '/') path = path[1..];
        // strip trailing slash
        if (path.len > 0 and path[path.len - 1] == '/') path = path[0 .. path.len - 1];

        var it = std.mem.splitSequence(u8, path, "/");
        const repo_and_rev = it.next();

        if (repo_and_rev == null) {
            log.err("Malformed HF model URI, missing repo: {s}", .{uri});
            return Error.MalformedURI;
        }

        // repo may contain an optional @revision suffix
        var rev: ?[]const u8 = null;
        var repo = repo_and_rev.?;
        const at_index = std.mem.indexOf(u8, repo_and_rev.?, "@");
        if (at_index) |idx| {
            repo = repo_and_rev.?[0..idx];
            rev = repo_and_rev.?[idx + 1 ..];
            if (rev.?.len == 0) rev = null;
        }

        return .{
            .namespace = namespace,
            .repo = repo,
            .revision = rev,
        };
    }

    pub fn filePathFromUri(uri: []const u8) Error![]const u8 {
        const parsed = std.Uri.parse(uri) catch {
            log.err("Failed to parse HF file URI: {s}", .{uri});
            return Error.MalformedURI;
        };

        var path = parsed.path.percent_encoded;
        // strip leading slash
        if (path.len > 0 and path[0] == '/') path = path[1..];
        // strip trailing slash
        if (path.len > 0 and path[path.len - 1] == '/') path = path[0 .. path.len - 1];

        // Remove the first segment (repo[@rev])
        var it = std.mem.splitSequence(u8, path, "/");
        _ = it.next(); // skip repo[@rev]
        const rest = it.rest();
        path = rest;

        return path;
    }

    pub fn isModelPath(uri: []const u8) bool {
        // Determine whether the directory URI represents an HF repo root.
        // Valid root examples:
        //   hf://user/repo
        //   hf://user/repo/
        //   hf://user/repo@rev
        //   hf://user/repo@rev/
        // Invalid examples:
        //   hf://
        //   hf://user
        //   hf://user/
        //   hf://user/repo/some/path

        const parsed_uri = std.Uri.parse(uri) catch return false;

        // Must be hf scheme with a host
        if (!std.mem.eql(u8, parsed_uri.scheme, "hf")) return false;
        if (parsed_uri.host == null) return false;

        var path = parsed_uri.path.percent_encoded;

        // No path or root-only path ("/") are not repo roots
        if (path.len == 0) return false;
        if (path.len == 1 and path[0] == '/') return false;

        // Strip leading slash if present
        if (path[0] == '/') path = path[1..];

        // Strip trailing slash if present
        if (path.len > 0 and path[path.len - 1] == '/') path = path[0 .. path.len - 1];

        // After trimming, there must be exactly one segment (no additional '/')
        if (path.len == 0) return false;

        // Check for additional '/' in the path
        var i: usize = 0;
        while (i < path.len) : (i += 1) {
            if (path[i] == '/') return false;
        }

        return true;
    }

    pub fn fetchModelSize(
        allocator: std.mem.Allocator,
        client: *std.http.Client,
        extra_headers: []std.http.Header,
        repo_uri: []const u8,
    ) Error!json.Parsed(TreeSize) {
        const model = try HFApi.modelFromUri(repo_uri);
        const rev = model.revision orelse blk: {
            log.debug("No revision specified in repo URI; defaulting to 'main'", .{});
            break :blk "main";
        };

        var redirect_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
        var url_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const api_url = std.fmt.bufPrint(
            &url_buf,
            "https://huggingface.co/api/models/{s}/{s}/treesize/{s}/",
            .{ model.namespace, model.repo, rev },
        ) catch {
            log.err("Failed to format HF model size API URL for repo URI {s}", .{repo_uri});
            return Error.OutOfMemory;
        };

        var req = try allocator.create(std.http.Client.Request);
        defer allocator.destroy(req);

        const api_uri = std.Uri.parse(api_url) catch {
            log.err("Failed to parse HF model size API URL {s}", .{api_url});
            return Error.MalformedURI;
        };

        req.* = client.request(.GET, api_uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = extra_headers,
        }) catch {
            log.err("Failed to create HTTP request for HF model size API URL {s}", .{api_url});
            return Error.RequestFailed;
        };
        defer req.deinit();

        req.sendBodiless() catch {
            log.err("Failed to send HTTP request for HF model size API URL {s}", .{api_url});
            return Error.RequestFailed;
        };

        var resp = req.receiveHead(&redirect_buffer) catch {
            log.err("Failed to receive HTTP response for HF model size API URL {s}", .{api_url});
            return Error.RequestFailed;
        };

        if (resp.head.status != .ok) {
            log.err("Failed to fetch HF model size, got {s} for {s}", .{ @tagName(resp.head.status), api_url });
            return Error.RequestFailed;
        }

        const content_length = resp.head.content_length orelse {
            log.err("Missing Content-Length in HF model size response for {s}", .{api_url});
            return Error.ResponseParsingFailed;
        };

        const body = resp.reader(&.{}).readAlloc(allocator, content_length) catch {
            log.err("Failed to read HF model size response body for {s} of content length {d}", .{ api_url, content_length });
            return Error.ResponseParsingFailed;
        };
        defer allocator.free(body);

        const parsed = json.parseFromSlice(
            HFApi.TreeSize,
            allocator,
            body,
            .{ .ignore_unknown_fields = true },
        ) catch {
            log.err("Failed to parse HF model size response JSON for {s}", .{api_url});
            return Error.ResponseParsingFailed;
        };

        return parsed;
    }
};

// todo: ref counters
const HFRecords = struct {
    pub const Error = HFApi.Error || std.mem.Allocator.Error;

    const Mapping = std.StringArrayHashMapUnmanaged(HFApi.TreeEntry);

    arena: std.heap.ArenaAllocator,
    mapping: Mapping,

    pub fn init(allocator: std.mem.Allocator) HFRecords {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .mapping = .{},
        };
    }

    pub fn get(self: *HFRecords, key: []const u8) ?HFApi.TreeEntry {
        // Normalize key by stripping directory trailing slash if present
        const normalized_key = if (key.len > 0 and key[key.len - 1] == '/') key[0 .. key.len - 1] else key;

        return self.mapping.get(normalized_key);
    }

    pub fn containsPathPrefix(self: *HFRecords, prefix: []const u8) bool {
        var it = self.mapping.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (key.len >= prefix.len and std.mem.startsWith(u8, key, prefix)) {
                return true;
            }
        }

        return false;
    }

    pub fn fetch(
        self: *HFRecords,
        client: *std.http.Client,
        limit: usize,
        extra_headers: []std.http.Header,
        repo_uri: []const u8,
        url: ?[]const u8, // used for pagination
    ) Error!void {
        const allocator = self.arena.allocator();

        var url_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        var redirect_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;

        const api_url = url orelse blk: {
            const model = try HFApi.modelFromUri(repo_uri);
            const rev = model.revision orelse blk_: {
                log.debug("No revision specified in repo URI; defaulting to 'main'", .{});
                break :blk_ "main";
            };

            break :blk std.fmt.bufPrint(
                &url_buf,
                "https://huggingface.co/api/models/{s}/{s}/tree/{s}/?expand=true&recursive=true&limit={d}",
                .{ model.namespace, model.repo, rev, limit },
            ) catch {
                log.err("Failed to format HF repo tree API URL for repo URI {s}", .{repo_uri});
                return Error.OutOfMemory;
            };
        };

        const api_uri = std.Uri.parse(api_url) catch {
            log.err("Failed to parse HF repo tree API URL {s}", .{api_url});
            return Error.MalformedURI;
        };

        var req = try allocator.create(std.http.Client.Request);
        defer allocator.destroy(req);

        req.* = client.request(.GET, api_uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = extra_headers,
        }) catch {
            log.err("Failed to create HTTP request for HF repo tree API URL {s}", .{api_url});
            return Error.RequestFailed;
        };
        defer req.deinit();

        req.sendBodiless() catch {
            log.err("Failed to send HTTP request for HF repo tree API URL {s}", .{api_url});
            return Error.RequestFailed;
        };

        var resp = req.receiveHead(&redirect_buffer) catch {
            log.err("Failed to receive HTTP response for HF repo tree API URL {s}", .{api_url});
            return Error.RequestFailed;
        };
        if (resp.head.status != .ok) {
            log.err("Failed to fetch HF repo tree, got {s} for {s}", .{ @tagName(resp.head.status), api_url });
            return Error.RequestFailed;
        }

        var next_url: ?[]const u8 = null;
        var it = resp.head.iterateHeaders();
        while (it.next()) |hdr| {
            if (std.mem.eql(u8, hdr.name, "Link") or std.mem.eql(u8, hdr.name, "link")) {
                const hv = hdr.value;
                var i: usize = 0;
                while (i < hv.len) : (i += 1) {
                    const lt = std.mem.indexOf(u8, hv[i..], "<");
                    if (lt == null) break;
                    const start = i + lt.? + 1;
                    const gt = std.mem.indexOf(u8, hv[start..], ">");
                    if (gt == null) break;
                    const end = start + gt.?;
                    const url_slice = hv[start..end];
                    const after = hv[end..];
                    if (std.mem.indexOf(u8, after, "rel=\"next\"") != null) {
                        next_url = url_slice;
                        break;
                    }
                    i = end + 1;
                }
            }
        }

        const content_length = resp.head.content_length orelse {
            log.err("Missing Content-Length in HF repo tree response for {s}", .{api_url});
            return Error.ResponseParsingFailed;
        };
        const body = resp.reader(&.{}).readAlloc(allocator, content_length) catch {
            log.err("Failed to read HF repo tree response body for {s} of content length {d}", .{ api_url, content_length });
            return Error.ResponseParsingFailed;
        };

        const entries = json.parseFromSliceLeaky(
            []HFApi.TreeEntry,
            allocator,
            body,
            .{ .ignore_unknown_fields = true },
        ) catch {
            log.err("Failed to parse HF repo tree response JSON for {s}", .{api_url});
            return Error.ResponseParsingFailed;
        };

        for (entries) |entry| {
            try self.add(repo_uri, entry);
        }

        if (next_url) |url_| {
            try self.fetch(client, limit, extra_headers, repo_uri, url_);
        }
    }

    fn add(self: *HFRecords, repo_uri: []const u8, entry: HFApi.TreeEntry) Error!void {
        const allocator = self.arena.allocator();

        // Ensure keys are built from the canonical repo root (hf://namespace/repo[@rev]/)
        const model = try HFApi.modelFromUri(repo_uri);
        const base = if (model.revision) |rev| blk: {
            break :blk try std.fmt.allocPrint(allocator, "hf://{s}/{s}@{s}/", .{ model.namespace, model.repo, rev });
        } else blk: {
            break :blk try std.fmt.allocPrint(allocator, "hf://{s}/{s}/", .{ model.namespace, model.repo });
        };
        const key = try std.fmt.allocPrint(allocator, "{s}{s}", .{ base, entry.path });
        try self.mapping.put(allocator, key, entry);
    }

    pub fn deinit(self: *HFRecords) void {
        self.arena.deinit();
    }
};

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
                AuthError.MissingHomePath, AuthError.MissingConfigFile => return fromEnv(allocator) catch |e| switch (e) {
                    AuthError.MissingHFToken => {
                        log.warn("No Hugging Face authentication token found in environment or home config; proceeding without authentication.", .{});
                        return .none;
                    },
                    else => return e,
                },
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

    const HandleId = i32;

    pub const InitHandleError = error{
        PathTooLong,
        PathUnresolvable,
        FailedToFetchRemoteResource,
        MissingRessource,
    } || std.mem.Allocator.Error;

    pub const FileHandle = struct {
        allocator: std.mem.Allocator,

        uri: []const u8,
        pos: u64,
        size: u64,
        redirect_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined,
        request: ?*std.http.Client.Request = null,
        response: ?std.http.Client.Response = null,
        body_reader: ?*std.Io.Reader = null,

        total_requests: u64 = 0,
        total_latency_ns: u64 = 0,
        min_latency_ns: u64 = std.math.maxInt(u64),
        max_latency_ns: u64 = 0,

        pub fn deinit(self: *FileHandle) void {
            self.releaseRequest();
            self.allocator.free(self.uri);
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
    };

    pub const DirHandle = struct {
        allocator: std.mem.Allocator,
        uri: []const u8,
        size: u64,

        pub fn deinit(self: *DirHandle) void {
            self.allocator.free(self.uri);
            self.allocator.destroy(self);
        }
    };

    pub const Config = struct {
        auth: Auth = .none,
        request_range_min: u64 = 8 * 1024,
        request_range_max: u64 = 128 * 1024 * 1024,
        hf_pagination_limit: usize = 100,
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex,

    file_handles: std.AutoHashMapUnmanaged(HandleId, *FileHandle),
    dir_handles: std.AutoHashMapUnmanaged(HandleId, *DirHandle),
    next_file_handle_id: HandleId,
    next_dir_handle_id: HandleId,

    hf_records: HFRecords,

    client: *std.http.Client,
    http_headers: []std.http.Header,
    config: Config,

    vtable: std.Io.VTable,

    pub fn init(allocator: std.mem.Allocator, base_io: std.Io, http_client: *std.http.Client, config: Config) std.mem.Allocator.Error!HF {
        var self: HF = .{
            .allocator = allocator,
            .mutex = .init,
            .file_handles = .{},
            .dir_handles = .{},
            .next_file_handle_id = 0,
            .next_dir_handle_id = 0,
            .hf_records = .init(allocator),
            .client = http_client,
            .http_headers = try allocator.alloc(std.http.Header, 2),
            .config = config,
            .vtable = makeVTable(base_io),
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
        self.hf_records.deinit();
        self.allocator.free(self.http_headers);
    }

    pub fn io(self: *HF) std.Io {
        return .{
            .vtable = &self.vtable,
            .userdata = self,
        };
    }

    const URIKind = union(enum) {
        dir,
        file,
    };

    fn resolveUri(self: *HF, parent_path: []const u8, sub_path: []const u8, kind: URIKind) InitHandleError![]const u8 {
        if (std.mem.startsWith(u8, sub_path, "hf://")) {
            if (kind == .dir and !std.mem.endsWith(u8, sub_path, "/")) {
                return std.fmt.allocPrint(self.allocator, "{s}/", .{sub_path});
            } else {
                return try self.allocator.dupe(u8, sub_path);
            }
        }

        const base_uri = std.Uri.parse(parent_path) catch return InitHandleError.PathUnresolvable;
        var aux_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;

        if (sub_path.len > aux_buf.len) return InitHandleError.PathTooLong;

        @memcpy(aux_buf[0..sub_path.len], sub_path);
        var aux_slice: []u8 = aux_buf[0..];
        const resolved = std.Uri.resolveInPlace(base_uri, sub_path.len, &aux_slice) catch return InitHandleError.PathUnresolvable;
        if (kind == .dir and !std.mem.endsWith(u8, resolved.path.percent_encoded, "/")) {
            return std.fmt.allocPrint(self.allocator, "{f}/", .{resolved.fmt(.{
                .scheme = true,
                .authority = true,
                .path = true,
                .query = false,
                .fragment = false,
            })});
        } else {
            return std.fmt.allocPrint(self.allocator, "{f}", .{resolved.fmt(.{
                .scheme = true,
                .authority = true,
                .path = true,
                .query = false,
                .fragment = false,
            })});
        }
    }

    fn fetchHFRecordsIfNeeded(
        self: *HF,
        uri: []const u8,
    ) HFRecords.Error!void {
        var extra_headers: []std.http.Header = &.{};
        switch (self.config.auth) {
            .hf_token => |_| {
                extra_headers = self.http_headers[0..1];
            },
            else => {},
        }

        if (!self.hf_records.containsPathPrefix(uri)) {
            log.debug("Fetching HF records for URI {s}", .{uri});
            try self.hf_records.fetch(self.client, self.config.hf_pagination_limit, extra_headers, uri, null);
        }
    }

    fn dirSize(self: *HF, uri: []const u8) HFApi.Error!u64 {
        var extra_headers: []std.http.Header = &.{};
        switch (self.config.auth) {
            .hf_token => |_| {
                extra_headers = self.http_headers[0..1];
            },
            else => {},
        }

        if (HFApi.isModelPath(uri)) {
            const model_size = HFApi.fetchModelSize(
                self.allocator,
                self.client,
                extra_headers,
                uri,
            ) catch |err| {
                log.err("Failed to fetch model size for dirSize with URI {s}", .{uri});
                return err;
            };
            defer model_size.deinit();

            return model_size.value.size;
        } else {
            const entry = self.hf_records.get(uri) orelse {
                log.err("Directory path not found in repo tree entries with URI {s}", .{uri});
                return HFApi.Error.ResponseParsingFailed;
            };

            return entry.size;
        }
    }

    fn initFileHandle(
        self: *HF,
        dir: DirHandle,
        sub_path: []const u8,
    ) InitHandleError!*FileHandle {
        const handle = try self.allocator.create(FileHandle);
        errdefer self.allocator.destroy(handle);

        const uri = try self.resolveUri(dir.uri, sub_path, .file);
        errdefer self.allocator.free(uri);

        self.fetchHFRecordsIfNeeded(uri) catch {
            log.err("Failed to fetch records in dirOpenFile for URI {s}", .{uri});
            return InitHandleError.FailedToFetchRemoteResource;
        };

        const entry = self.hf_records.get(uri) orelse {
            log.err("Directory path not found in repo tree entries with URI {s}", .{uri});
            return InitHandleError.MissingRessource;
        };

        handle.* = .{
            .allocator = self.allocator,
            .uri = uri,
            .pos = 0,
            .size = entry.size,
        };

        return handle;
    }

    fn registerFileHandle(self: *HF, handle: *FileHandle) std.mem.Allocator.Error!HandleId {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_id = self.next_file_handle_id;
        self.next_file_handle_id += 1;

        try self.file_handles.put(self.allocator, handle_id, handle);

        return handle_id;
    }

    fn fileHandle(self: *HF, file: std.Io.File) ?*FileHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        return self.file_handles.get(file.handle);
    }

    fn initDirHandle(
        self: *HF,
        parent_path: []const u8,
        sub_path: []const u8,
    ) InitHandleError!*DirHandle {
        const dir_handle = try self.allocator.create(DirHandle);
        errdefer self.allocator.destroy(dir_handle);

        const uri = try self.resolveUri(parent_path, sub_path, .dir);
        errdefer self.allocator.free(uri);

        self.fetchHFRecordsIfNeeded(uri) catch {
            log.err("Failed to fetch HF records in dirOpenFile for URI {s}", .{uri});
            return InitHandleError.FailedToFetchRemoteResource;
        };

        const dir_size = self.dirSize(uri) catch {
            log.err("Failed to get directory size in dirOpenFile for URI {s}", .{uri});
            return InitHandleError.FailedToFetchRemoteResource;
        };

        dir_handle.* = .{
            .allocator = self.allocator,
            .uri = uri,
            .size = dir_size,
        };

        return dir_handle;
    }

    fn registerDirHandle(self: *HF, handle: *DirHandle) std.mem.Allocator.Error!HandleId {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_id = self.next_dir_handle_id;
        self.next_dir_handle_id += 1;

        try self.dir_handles.put(self.allocator, handle_id, handle);

        return handle_id;
    }

    fn dirHandle(self: *HF, dir: std.Io.Dir) ?DirHandle {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return .{
                .allocator = self.allocator,
                .uri = "",
                .size = 0,
            };
        }

        if (self.dir_handles.get(dir.handle)) |handle_ptr| {
            return handle_ptr.*;
        } else {
            return null;
        }
    }

    fn openFile(
        self: *HF,
        dir: std.Io.Dir,
        sub_path: []const u8,
    ) std.Io.File.OpenError!std.Io.File {
        const dir_handle = self.dirHandle(dir) orelse {
            log.warn("Directory handle not found for dirOpenFile with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.File.OpenError.FileNotFound;
        };

        const handle = self.initFileHandle(dir_handle, sub_path) catch {
            log.warn("Failed to init FileHandle in openFile with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.File.OpenError.SystemResources;
        };

        const handle_id = self.registerFileHandle(handle) catch {
            log.warn("Failed to register FileHandle in openFile with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.File.OpenError.SystemResources;
        };

        return .{ .handle = @intCast(handle_id) };
    }

    fn openDir(
        self: *HF,
        dir: std.Io.Dir,
        sub_path: []const u8,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        const dir_handle = self.dirHandle(dir) orelse {
            log.warn("Directory handle not found for dirOpenFile", .{});
            return std.Io.Dir.OpenError.FileNotFound;
        };

        const handle = self.initDirHandle(dir_handle.uri, sub_path) catch {
            log.warn("Failed to init DirHandle in openDir with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.Dir.OpenError.SystemResources;
        };

        const handle_id = self.registerDirHandle(handle) catch {
            log.warn("Failed to register DirHandle in openDir with dir={any} sub_path={s}", .{ dir.handle, sub_path });
            return std.Io.Dir.OpenError.SystemResources;
        };

        return .{ .handle = @intCast(handle_id) };
    }

    fn closeDir(self: *HF, dir: std.Io.Dir) void {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_kv = self.dir_handles.fetchRemove(dir.handle) orelse {
            log.warn("Attempted to close non-existent dir handle: {d}", .{dir.handle});
            return;
        };

        handle_kv.value.deinit();
    }

    fn closeFile(self: *HF, file: std.Io.File) void {
        self.mutex.lockUncancelable(self.io());
        defer self.mutex.unlock(self.io());

        const handle_kv = self.file_handles.fetchRemove(file.handle) orelse {
            log.warn("Attempted to close non-existent file handle: {d}", .{file.handle});
            return;
        };

        handle_kv.value.deinit();
    }

    fn setAbsolutePos(self: *HF, file: std.Io.File, pos: u64) std.Io.File.SeekError!void {
        const handle = self.fileHandle(file) orelse return std.Io.File.SeekError.Unexpected;
        handle.releaseRequest();
        handle.pos = pos;
    }

    fn setRelativeOffset(self: *HF, file: std.Io.File, offset: i64) std.Io.File.SeekError!void {
        const handle = self.fileHandle(file) orelse return std.Io.File.SeekError.Unexpected;
        handle.releaseRequest();

        const new_pos = @as(i64, @intCast(handle.pos)) + offset;
        handle.pos = @intCast(new_pos);
    }

    fn performPositionalRead(self: *HF, file: std.Io.File, data: [][]u8, offset: u64) std.Io.File.ReadPositionalError!usize {
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

    fn performRead(self: *HF, file: std.Io.File, data: [][]u8) std.Io.File.Reader.Error!usize {
        const handle = self.fileHandle(file) orelse {
            log.err("Invalid file handle in VFS read", .{});
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

    fn requestRead(self: *HF, file: std.Io.File, start: u64, end: u64) std.Io.File.Reader.Error!void {
        const handle = self.fileHandle(file) orelse return std.Io.File.Reader.Error.Unexpected;

        if (handle.request) |_| return;

        std.debug.assert(start <= end);

        var range_buf: [64]u8 = undefined;
        const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ start, end }) catch unreachable;

        var timer = std.time.Timer.start() catch {
            log.err("Failed to start timer for request", .{});
            return std.Io.File.Reader.Error.SystemResources;
        };

        const request = self.allocator.create(std.http.Client.Request) catch {
            log.err("Failed to allocate request", .{});
            return std.Io.File.Reader.Error.SystemResources;
        };
        errdefer self.allocator.destroy(request);

        var url_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;

        const path = HFApi.filePathFromUri(handle.uri) catch |err| {
            log.err("Failed to extract file path from URI {s}: {}", .{ handle.uri, err });
            return std.Io.File.Reader.Error.Unexpected;
        };

        const api_url = blk: {
            const model = HFApi.modelFromUri(handle.uri) catch |err| {
                log.err("Failed to parse model from URI {s}: {}", .{ handle.uri, err });
                return std.Io.File.Reader.Error.Unexpected;
            };
            const rev = model.revision orelse blk_: {
                log.debug("No revision specified in repo URI; defaulting to 'main'", .{});
                break :blk_ "main";
            };

            break :blk std.fmt.bufPrint(
                &url_buf,
                "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
                .{ model.namespace, model.repo, rev, path },
            ) catch {
                log.err("Failed to format HF repo tree API URL for repo URI {s}", .{handle.uri});
                return std.Io.File.Reader.Error.SystemResources;
            };
        };

        const api_uri = std.Uri.parse(api_url) catch {
            log.err("Failed to parse URL during read request", .{});
            return std.Io.File.Reader.Error.Unexpected;
        };

        self.http_headers[1] = .{ .name = "Range", .value = range_header };

        var extra_headers: []std.http.Header = &.{};
        switch (self.config.auth) {
            .hf_token => |_| {
                extra_headers = self.http_headers[0..2];
            },
            else => {
                extra_headers = self.http_headers[1..2];
            },
        }

        request.* = self.client.request(.GET, api_uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = extra_headers,
        }) catch |err| {
            log.err("Failed to create GET request: {}", .{err});
            return std.Io.File.Reader.Error.SystemResources;
        };
        errdefer request.deinit();

        request.sendBodiless() catch |err| {
            log.err("Failed to send GET request: {}", .{err});
            return std.Io.File.Reader.Error.SystemResources;
        };

        const response = request.receiveHead(&handle.redirect_buffer) catch {
            log.err("Failed to receive header response for URL {s} ", .{api_url});
            return std.Io.File.Reader.Error.Unexpected;
        };

        const latency = timer.read();
        handle.recordLatency(latency);

        log.debug("GET call: handle={d} status={s} content_length={?} range={d}-{d} ({d}) latency={d}ms avg={d}ms min={d}ms max={d}ms for {s}", .{
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
            api_url,
        });

        if (response.head.status != .partial_content and response.head.status != .ok) {
            log.err("Unexpected HTTP status ({s}) for GET {s}", .{ @tagName(response.head.status), api_url });
            var it = response.head.iterateHeaders();
            while (it.next()) |hdr| {
                log.err("Header: {s}: {s}", .{ hdr.name, hdr.value });
            }

            if (response.head.status == .unauthorized) {
                return std.Io.File.Reader.Error.AccessDenied;
            } else {
                return std.Io.File.Reader.Error.Unexpected;
            }
        }

        handle.request = request;
        handle.response = response;
        handle.body_reader = handle.response.?.reader(&.{});
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
        log.err("VFS is read-only, dirMake not supported", .{});
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
        log.err("VFS is read-only, dirMakePath not supported", .{});
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
        log.err("VFS is read-only, dirMakeOpenPath not supported", .{});
        return std.Io.Dir.MakeOpenPathError.Unexpected;
    }

    fn dirStat(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
    ) std.Io.Dir.StatError!std.Io.Dir.Stat {
        _ = userdata;
        _ = dir;
        log.err("VFS does not support dirStat", .{});
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
        @panic("TODO: implement VFS dirStatPath");
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

        log.err("VFS does not support dirAccess", .{});
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
        log.err("VFS is read-only, dirCreateFile not supported", .{});
        return std.Io.File.OpenError.Unexpected;
    }

    fn dirOpenFile(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenError!std.Io.File {
        _ = flags;
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.openFile(dir, sub_path);
    }

    fn dirOpenDir(
        userdata: ?*anyopaque,
        dir: std.Io.Dir,
        sub_path: []const u8,
        options: std.Io.Dir.OpenOptions,
    ) std.Io.Dir.OpenError!std.Io.Dir {
        _ = options;
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.openDir(dir, sub_path);
    }

    fn dirClose(userdata: ?*anyopaque, dir: std.Io.Dir) void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        self.closeDir(dir);
    }

    fn fileStat(
        userdata: ?*anyopaque,
        file: std.Io.File,
    ) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        const handle = self.fileHandle(file) orelse {
            log.err("Invalid file handle in VFS stat", .{});
            return std.Io.File.StatError.Unexpected;
        };

        return .{
            .inode = @intCast(file.handle),
            .kind = .file,
            .size = handle.size,
            .mode = 0o444,
            .atime = .{ .nanoseconds = 0 },
            .mtime = .{ .nanoseconds = 0 },
            .ctime = .{ .nanoseconds = 0 },
        };
    }

    fn fileClose(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
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
        log.err("VFS is read-only, fileWriteStreaming not supported", .{});
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
        log.err("VFS is read-only, fileWritePositional not supported", .{});
        return std.Io.File.WritePositionalError.Unexpected;
    }

    fn fileReadStreaming(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
    ) std.Io.File.Reader.Error!usize {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.performRead(file, data);
    }

    fn fileReadPositional(
        userdata: ?*anyopaque,
        file: std.Io.File,
        data: [][]u8,
        offset: u64,
    ) std.Io.File.ReadPositionalError!usize {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        return try self.performPositionalRead(file, data, offset);
    }

    fn fileSeekBy(
        userdata: ?*anyopaque,
        file: std.Io.File,
        relative_offset: i64,
    ) std.Io.File.SeekError!void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        try self.setRelativeOffset(file, relative_offset);
    }

    fn fileSeekTo(
        userdata: ?*anyopaque,
        file: std.Io.File,
        absolute_offset: u64,
    ) std.Io.File.SeekError!void {
        const self: *HF = @ptrCast(@alignCast(userdata orelse unreachable));
        try self.setAbsolutePos(file, absolute_offset);
    }

    fn openSelfExe(
        userdata: ?*anyopaque,
        flags: std.Io.File.OpenFlags,
    ) std.Io.File.OpenSelfExeError!std.Io.File {
        _ = userdata;
        _ = flags;
        log.err("VFS does not support openSelfExe", .{});
        return std.Io.File.OpenSelfExeError.NotSupported;
    }
};
