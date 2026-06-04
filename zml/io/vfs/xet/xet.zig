const std = @import("std");
const reconstruction = @import("reconstruction.zig");
const tensor_registry = @import("tensor_registry.zig");
const xorb_frames = @import("xorb_frames.zig");
const stats = @import("stats.zig");

pub const DEFAULT_CONCURRENCY: usize = 32;
pub const MAX_CONCURRENCY: usize = DEFAULT_CONCURRENCY;
const BODY_PREFETCH: usize = 4;
pub const METADATA_PREFETCH_DISTANCE: usize = 20;

const AUTH_HEADER_BUFFER_LEN = 16 * 1024;
const CAS_URL_BUFFER_LEN = 8 * 1024;
const MAX_RECONSTRUCTION_JSON_LEN = 64 * 1024 * 1024;
const XET_TOKEN_URL_TEMPLATE = "https://huggingface.co/api/models/{[repo]s}/{[model]s}/xet-read-token/{[rev]s}";

pub const ReconstructionRange = reconstruction.ReconstructionRange;
pub const Stats = stats.Stats;
pub const TensorRangeRegistration = tensor_registry.RangeRegistration;
const AtomicStats = stats.AtomicStats;

fn parseEnabled(value: ?[]const u8) bool {
    const v = value orelse return true;
    return !(std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "0") or
        std.ascii.eqlIgnoreCase(v, "false"));
}

pub const Config = struct {
    enabled: bool = true,
};

pub const State = struct {
    config: Config = .{},
    mutex: std.Io.Mutex = .init,
    file_hashes: std.StringHashMapUnmanaged([]u8) = .{},
    tokens: std.StringHashMapUnmanaged(Auth.Token) = .{},
    reconstructions: std.StringHashMapUnmanaged(*reconstruction.Index) = .{},
    reconstruction_flights: std.StringHashMapUnmanaged(*Flight) = .{},
    tensor_registry: tensor_registry.Registry = .empty,
    prefetch_group: std.Io.Group = .init,
    prefetch_slots: std.Io.Semaphore = .{ .permits = DEFAULT_CONCURRENCY },
    xorb_slots: std.Io.Semaphore = .{ .permits = DEFAULT_CONCURRENCY },
    stats: AtomicStats = .{},

    pub fn initFromEnv(environ_map: *std.process.Environ.Map) State {
        return .{
            .config = .{
                .enabled = parseEnabled(environ_map.get("ZML_HF_XET")),
            },
        };
    }

    pub fn enabled(self: *const State) bool {
        return self.config.enabled;
    }

    pub fn deinit(self: *State, allocator: std.mem.Allocator, io: std.Io) void {
        self.prefetch_group.cancel(io);

        var file_hash_it = self.file_hashes.iterator();
        while (file_hash_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.file_hashes.deinit(allocator);

        var token_it = self.tokens.iterator();
        while (token_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(allocator);
        }
        self.tokens.deinit(allocator);

        var reconstruction_it = self.reconstructions.iterator();
        while (reconstruction_it.next()) |entry| {
            const index = entry.value_ptr.*;
            allocator.free(entry.key_ptr.*);
            index.deinit(allocator);
            allocator.destroy(index);
        }
        self.reconstructions.deinit(allocator);

        var flight_it = self.reconstruction_flights.iterator();
        while (flight_it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.destroy(entry.value_ptr.*);
        }
        self.reconstruction_flights.deinit(allocator);

        self.tensor_registry.deinit(allocator);
        self.* = .{};
    }

    pub fn snapshotStats(self: *const State) Stats {
        return self.stats.snapshot();
    }

    pub fn putFileHash(self: *State, allocator: std.mem.Allocator, io: std.Io, repo: Repo, path: []const u8, hash: ?[]const u8) !void {
        const value = hash orelse return;
        var key_buffer: [8 * 1024]u8 = undefined;
        const key = try repoPathKey(&key_buffer, repo, path);
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);
        const hash_copy = try allocator.dupe(u8, value);
        errdefer allocator.free(hash_copy);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        const gop = try self.file_hashes.getOrPut(allocator, key_copy);
        if (gop.found_existing) {
            allocator.free(key_copy);
            allocator.free(gop.value_ptr.*);
        }
        gop.value_ptr.* = hash_copy;
    }

    pub fn registerTensorRange(self: *State, allocator: std.mem.Allocator, io: std.Io, file_uri: []const u8, offset: u64, len: u64) !void {
        try self.registerTensorRanges(allocator, io, &.{.{ .file_uri = file_uri, .offset = offset, .len = len }});
    }

    pub fn registerTensorRanges(self: *State, allocator: std.mem.Allocator, io: std.Io, ranges: []const TensorRangeRegistration) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        try tensor_registry.registerRanges(&self.tensor_registry, allocator, ranges);
    }

    pub fn registerTensorStore(self: *State, allocator: std.mem.Allocator, io: std.Io, store: anytype) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        try tensor_registry.registerStore(&self.tensor_registry, allocator, store);
    }

    pub fn registerTensorStores(self: *State, allocator: std.mem.Allocator, io: std.Io, stores: anytype) !void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        try tensor_registry.registerStores(&self.tensor_registry, allocator, stores);
    }

    pub fn read(self: *State, request: ReadRequest) !?usize {
        if (!self.enabled()) return null;
        const xet_hash = self.fileHash(request.io, request.repo, request.path) orelse return null;
        var prefetch_buffer: [MAX_CONCURRENCY]ReconstructionRange = undefined;
        const range_lookup = self.reconstructionRange(request.io, request.file_uri, request.offset, request.take, prefetch_buffer[0..]);
        var authorization_buffer: [AUTH_HEADER_BUFFER_LEN]u8 = undefined;
        var cas_url_buffer: [CAS_URL_BUFFER_LEN]u8 = undefined;
        const auth = try Auth.ensure(self, request, &authorization_buffer, &cas_url_buffer);

        const index = try ReconstructionLoader.index(self, request, xet_hash, &auth, range_lookup.range);
        self.stats.addLogicalBytes(request.take);

        MetadataPrefetch.start(self, request, xet_hash, range_lookup.prefetch_ranges);
        var scratch = try ReadScratch.init(request.allocator, index.*, request.offset, request.take);
        defer scratch.deinit(request.allocator);

        var output = OutputSlicesWriter.init(request.data);
        var reader = OrderedTermReader.init(self, request, index.*, &scratch);
        defer reader.deinit();
        const streamed = try reader.interface.streamRemaining(&output.interface);
        try output.interface.flush();
        if (streamed != request.take or output.count != request.take) return error.InvalidReconstruction;
        return streamed;
    }

    fn fileHash(self: *State, io: std.Io, repo: Repo, path: []const u8) ?[]const u8 {
        var key_buffer: [8 * 1024]u8 = undefined;
        const key = repoPathKey(&key_buffer, repo, path) catch return null;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.file_hashes.get(key);
    }

    const ReconstructionLookup = struct {
        range: ReconstructionRange,
        prefetch_ranges: []const ReconstructionRange = &.{},
    };

    fn reconstructionRange(self: *State, io: std.Io, file_uri: []const u8, offset: u64, take: u64, prefetch_out: []ReconstructionRange) ReconstructionLookup {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        const read_end = offset + take;
        for (self.tensor_registry.items, 0..) |range, i| {
            if (!tensor_registry.sameFile(file_uri, range.file_uri)) continue;
            if (offset < range.offset or read_end > range.offset + range.len) continue;

            var count: usize = 0;
            if (offset == range.offset) {
                for (self.tensor_registry.items[i + 1 ..]) |next| {
                    if (count == prefetch_out.len or count == METADATA_PREFETCH_DISTANCE) break;
                    if (!tensor_registry.sameFile(file_uri, next.file_uri)) continue;
                    prefetch_out[count] = .{ .offset = next.offset, .len = next.len };
                    count += 1;
                }
            }
            return .{ .range = .{ .offset = range.offset, .len = range.len }, .prefetch_ranges = prefetch_out[0..count] };
        }

        return .{ .range = .{ .offset = offset, .len = take } };
    }
};

const Flight = struct {
    condition: std.Io.Condition = .init,
    waiters: usize = 1,
    complete: bool = false,
    err: ?anyerror = null,
};

fn repoKey(buffer: []u8, repo: Repo) ![]u8 {
    return try std.fmt.bufPrint(buffer, "{s}/{s}@{s}", .{ repo.repo, repo.model, repo.rev });
}

fn repoPathKey(buffer: []u8, repo: Repo, path: []const u8) ![]u8 {
    return try std.fmt.bufPrint(buffer, "{s}/{s}@{s}/{s}", .{ repo.repo, repo.model, repo.rev, path });
}

pub const Repo = struct {
    repo: []const u8,
    model: []const u8,
    rev: []const u8,
};

const ReadRequest = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    client: *std.http.Client,
    authorization: std.http.Client.Request.Headers.Value,
    repo: Repo,
    path: []const u8,
    file_uri: []const u8,
    data: []const []u8,
    offset: u64,
    take: u64,
};

const Auth = struct {
    const Headers = struct {
        authorization: []const u8,
        cas_url: []const u8,
        exp: i64,
    };

    const Token = struct {
        access_token: []u8,
        cas_url: []u8,
        exp: i64,

        fn deinit(self: *Token, allocator: std.mem.Allocator) void {
            allocator.free(self.access_token);
            allocator.free(self.cas_url);
        }

        fn expired(self: Token, io: std.Io) bool {
            const now = std.Io.Timestamp.now(io, .real).toSeconds();
            return now + 30 >= self.exp;
        }
    };

    const Response = struct {
        accessToken: []const u8,
        exp: i64,
        casUrl: []const u8,
    };

    fn ensure(state: *State, request: ReadRequest, authorization_buffer: []u8, cas_url_buffer: []u8) !Headers {
        var key_buffer: [512]u8 = undefined;
        const token_key = repoKey(&key_buffer, request.repo) catch return error.NameTooLong;

        if (try cached(state, request.io, token_key, authorization_buffer, cas_url_buffer)) |headers| return headers;

        var token = try fetchToken(state, request);
        errdefer token.deinit(request.allocator);

        const authorization = try std.fmt.bufPrint(authorization_buffer, "Bearer {s}", .{token.access_token});
        const cas_url = try std.fmt.bufPrint(cas_url_buffer, "{s}", .{token.cas_url});
        try put(state, request.allocator, request.io, request.repo, token);
        return .{ .authorization = authorization, .cas_url = cas_url, .exp = token.exp };
    }

    fn cached(state: *State, io: std.Io, token_key: []const u8, authorization_buffer: []u8, cas_url_buffer: []u8) !?Headers {
        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        const token = state.tokens.get(token_key) orelse return null;
        if (token.expired(io)) return null;
        return .{
            .authorization = try std.fmt.bufPrint(authorization_buffer, "Bearer {s}", .{token.access_token}),
            .cas_url = try std.fmt.bufPrint(cas_url_buffer, "{s}", .{token.cas_url}),
            .exp = token.exp,
        };
    }

    fn put(state: *State, allocator: std.mem.Allocator, io: std.Io, repo: Repo, token: Token) !void {
        var key_buffer: [512]u8 = undefined;
        const key = try repoKey(&key_buffer, repo);
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);

        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        const gop = try state.tokens.getOrPut(allocator, key_copy);
        if (gop.found_existing) {
            allocator.free(key_copy);
            gop.value_ptr.deinit(allocator);
        }
        gop.value_ptr.* = token;
    }

    fn fetchToken(_: *State, request: ReadRequest) !Token {
        var url_buffer: [8 * 1024]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, XET_TOKEN_URL_TEMPLATE, .{
            .repo = request.repo.repo,
            .model = request.repo.model,
            .rev = request.repo.rev,
        });

        var req = try request.client.request(.GET, try std.Uri.parse(url), .{
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = request.authorization,
            },
        });
        defer req.deinit();

        try req.sendBodiless();

        var head_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&head_buffer);
        if (res.head.status != .ok) return error.RequestFailed;

        const reader = res.reader(&.{});
        const content_length: usize = @intCast(res.head.content_length orelse return error.MissingContentLength);
        if (content_length > 256 * 1024) return error.ResponseTooLarge;
        const body = try reader.readAlloc(request.allocator, content_length);
        defer request.allocator.free(body);

        const parsed = try std.json.parseFromSlice(Response, request.allocator, body, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        return .{
            .access_token = try request.allocator.dupe(u8, parsed.value.accessToken),
            .cas_url = try request.allocator.dupe(u8, std.mem.trimEnd(u8, parsed.value.casUrl, "/")),
            .exp = parsed.value.exp,
        };
    }
};

const ReconstructionLoader = struct {
    fn index(state: *State, request: ReadRequest, xet_hash: []const u8, auth: *const Auth.Headers, range: ReconstructionRange) !*const reconstruction.Index {
        var key_buffer: [256]u8 = undefined;
        const cache_key = try range.formatCacheKey(&key_buffer, xet_hash);

        if (cachedIndex(state, request.io, xet_hash, cache_key, request.offset, request.take)) |cached| return cached;

        const flight = try beginFlight(state, request.allocator, request.io, cache_key);
        if (!flight.leader) {
            try waitForFlight(state, request.allocator, request.io, cache_key, flight.value);
            if (cachedIndex(state, request.io, xet_hash, cache_key, request.offset, request.take)) |cached| return cached;
            return error.RequestFailed;
        }
        errdefer |err| finishFlight(state, request.allocator, request.io, cache_key, flight.value, err);

        const body = try fetchBody(state, request, xet_hash, auth, range);
        defer request.allocator.free(body);

        const index_value = try reconstruction.parseIndexBody(request.allocator, range, auth.exp, body);

        const index_ptr = try request.allocator.create(reconstruction.Index);
        errdefer request.allocator.destroy(index_ptr);
        index_ptr.* = index_value;
        errdefer index_ptr.deinit(request.allocator);

        const key_copy = try request.allocator.dupe(u8, cache_key);
        errdefer request.allocator.free(key_copy);

        const cached_or_inserted = try insertIndex(state, request, key_copy, index_ptr);
        finishFlight(state, request.allocator, request.io, cache_key, flight.value, null);
        return cached_or_inserted;
    }

    const FlightToken = struct {
        value: *Flight,
        leader: bool,
    };

    fn beginFlight(state: *State, allocator: std.mem.Allocator, io: std.Io, key: []const u8) !FlightToken {
        const key_copy = try allocator.dupe(u8, key);
        errdefer allocator.free(key_copy);
        const new_flight = try allocator.create(Flight);
        errdefer allocator.destroy(new_flight);
        new_flight.* = .{};

        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        if (state.reconstruction_flights.get(key)) |existing| {
            existing.waiters += 1;
            allocator.free(key_copy);
            allocator.destroy(new_flight);
            return .{ .value = existing, .leader = false };
        }

        try state.reconstruction_flights.put(allocator, key_copy, new_flight);
        return .{ .value = new_flight, .leader = true };
    }

    fn finishFlight(state: *State, allocator: std.mem.Allocator, io: std.Io, key: []const u8, flight: *Flight, err: ?anyerror) void {
        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        flight.err = err;
        flight.complete = true;
        flight.condition.broadcast(io);
        releaseFlightLocked(state, allocator, key, flight);
    }

    fn waitForFlight(state: *State, allocator: std.mem.Allocator, io: std.Io, key: []const u8, flight: *Flight) !void {
        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        while (!flight.complete) flight.condition.waitUncancelable(io, &state.mutex);

        const err = flight.err;
        releaseFlightLocked(state, allocator, key, flight);
        if (err) |e| return e;
    }

    fn releaseFlightLocked(state: *State, allocator: std.mem.Allocator, key: []const u8, flight: *Flight) void {
        flight.waiters -= 1;
        if (!flight.complete or flight.waiters != 0) return;
        const removed = state.reconstruction_flights.fetchRemove(key).?;
        allocator.free(removed.key);
        allocator.destroy(flight);
    }

    fn insertIndex(state: *State, request: ReadRequest, key_copy: []u8, index_ptr: *reconstruction.Index) !*const reconstruction.Index {
        state.mutex.lockUncancelable(request.io);
        defer state.mutex.unlock(request.io);

        const gop = try state.reconstructions.getOrPut(request.allocator, key_copy);
        if (!gop.found_existing) {
            gop.value_ptr.* = index_ptr;
            return index_ptr;
        }

        request.allocator.free(key_copy);
        if (!gop.value_ptr.*.expired(request.io)) {
            index_ptr.deinit(request.allocator);
            request.allocator.destroy(index_ptr);
            return gop.value_ptr.*;
        }

        const old = gop.value_ptr.*;
        old.deinit(request.allocator);
        request.allocator.destroy(old);
        gop.value_ptr.* = index_ptr;
        return index_ptr;
    }

    fn cachedIndex(state: *State, io: std.Io, xet_hash: []const u8, exact_key: []const u8, offset: u64, take: u64) ?*const reconstruction.Index {
        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        if (state.reconstructions.get(exact_key)) |cached| {
            if (!cached.expired(io)) {
                return cached;
            }
        }

        var prefix_buffer: [128]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buffer, "{s}:", .{xet_hash}) catch return null;
        var it = state.reconstructions.iterator();
        while (it.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, prefix)) continue;
            const cached = entry.value_ptr.*;
            if (cached.expired(io)) continue;
            if (!cached.range.contains(offset, take)) continue;
            return cached;
        }

        return null;
    }

    fn fetchBody(_: *State, request: ReadRequest, xet_hash: []const u8, auth: *const Auth.Headers, range: ReconstructionRange) ![]u8 {
        var url_buffer: [8 * 1024]u8 = undefined;
        const reconstruction_url = try std.fmt.bufPrint(&url_buffer, "{s}/v2/reconstructions/{s}", .{
            auth.cas_url,
            xet_hash,
        });
        var range_buffer: [64]u8 = undefined;
        const end = range.offset + range.len - 1;
        const range_header = std.fmt.bufPrint(&range_buffer, "bytes={d}-{d}", .{ range.offset, end }) catch unreachable;

        var req = try request.client.request(.GET, try std.Uri.parse(reconstruction_url), .{
            .headers = .{
                .accept_encoding = .{ .override = "identity" },
                .authorization = .{ .override = auth.authorization },
            },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        });
        defer req.deinit();

        try req.sendBodiless();

        var head_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&head_buffer);
        if (res.head.status != .ok and res.head.status != .partial_content) return error.RequestFailed;

        const reader = res.reader(&.{});
        const content_length: usize = @intCast(res.head.content_length orelse return error.MissingContentLength);
        if (content_length > MAX_RECONSTRUCTION_JSON_LEN) return error.ResponseTooLarge;
        return try reader.readAlloc(request.allocator, content_length);
    }
};

const MetadataPrefetch = struct {
    const Task = struct {
        allocator: std.mem.Allocator,
        io: std.Io,
        client: *std.http.Client,
        authorization: std.http.Client.Request.Headers.Value,
        repo: Repo,
        xet_hash: []const u8,
        range: ReconstructionRange,

        fn init(request: ReadRequest, xet_hash: []const u8, range: ReconstructionRange) !*Task {
            const task = try request.allocator.create(Task);
            errdefer request.allocator.destroy(task);

            const repo_name = try request.allocator.dupe(u8, request.repo.repo);
            errdefer request.allocator.free(repo_name);
            const model = try request.allocator.dupe(u8, request.repo.model);
            errdefer request.allocator.free(model);
            const rev = try request.allocator.dupe(u8, request.repo.rev);
            errdefer request.allocator.free(rev);

            task.* = .{
                .allocator = request.allocator,
                .io = request.io,
                .client = request.client,
                .authorization = request.authorization,
                .repo = .{ .repo = repo_name, .model = model, .rev = rev },
                .xet_hash = xet_hash,
                .range = range,
            };
            return task;
        }

        fn destroy(self: *Task) void {
            self.allocator.free(self.repo.repo);
            self.allocator.free(self.repo.model);
            self.allocator.free(self.repo.rev);
            self.allocator.destroy(self);
        }
    };

    fn start(state: *State, request: ReadRequest, xet_hash: []const u8, ranges: []const ReconstructionRange) void {
        const limit = @min(ranges.len, MAX_CONCURRENCY);
        for (ranges[0..limit]) |range| {
            if (!shouldStart(state, request.io, xet_hash, range)) continue;
            if (!acquire(state, request.io)) continue;

            const task = Task.init(request, xet_hash, range) catch {
                release(state, request.io);
                continue;
            };

            state.prefetch_group.concurrent(request.io, run, .{ state, task }) catch |err| {
                task.destroy();
                release(state, request.io);
                switch (err) {
                    error.ConcurrencyUnavailable => {},
                }
                continue;
            };
        }
    }

    fn acquire(state: *State, io: std.Io) bool {
        if (!state.prefetch_slots.mutex.tryLock()) return false;
        defer state.prefetch_slots.mutex.unlock(io);
        if (state.prefetch_slots.permits == 0) return false;
        state.prefetch_slots.permits -= 1;
        return true;
    }

    fn release(state: *State, io: std.Io) void {
        state.prefetch_slots.post(io);
    }

    fn shouldStart(state: *State, io: std.Io, xet_hash: []const u8, range: ReconstructionRange) bool {
        var key_buffer: [256]u8 = undefined;
        const cache_key = range.formatCacheKey(&key_buffer, xet_hash) catch return false;

        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);

        if (state.reconstructions.get(cache_key)) |cached| {
            if (!cached.expired(io)) return false;
        }

        return state.reconstruction_flights.get(cache_key) == null;
    }

    fn run(state: *State, task: *Task) std.Io.Cancelable!void {
        defer {
            release(state, task.io);
            task.destroy();
        }

        const request: ReadRequest = .{
            .allocator = task.allocator,
            .io = task.io,
            .client = task.client,
            .authorization = task.authorization,
            .repo = task.repo,
            .path = &.{},
            .file_uri = &.{},
            .data = &.{},
            .offset = task.range.offset,
            .take = task.range.len,
        };

        var authorization_buffer: [AUTH_HEADER_BUFFER_LEN]u8 = undefined;
        var cas_url_buffer: [CAS_URL_BUFFER_LEN]u8 = undefined;
        const auth = Auth.ensure(state, request, &authorization_buffer, &cas_url_buffer) catch |err| {
            if (err == error.Canceled) return error.Canceled;
            return;
        };

        _ = ReconstructionLoader.index(state, request, task.xet_hash, &auth, task.range) catch |err| {
            if (err == error.Canceled) return error.Canceled;
            return;
        };
    }
};

const OutputSlicesWriter = struct {
    data: []const []u8,
    index: usize = 0,
    offset: usize = 0,
    count: usize = 0,
    interface: std.Io.Writer,

    fn init(data: []const []u8) OutputSlicesWriter {
        return .{
            .data = data,
            .interface = .{
                .buffer = &.{},
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    fn writeBytes(self: *OutputSlicesWriter, src: []const u8) !void {
        var src_offset: usize = 0;
        while (src_offset < src.len) {
            while (self.index < self.data.len and self.offset == self.data[self.index].len) {
                self.index += 1;
                self.offset = 0;
            }
            if (self.index >= self.data.len) return error.DestinationTooSmall;

            const dst = self.data[self.index];
            const n = @min(dst.len - self.offset, src.len - src_offset);
            if (n == 0) return error.DestinationTooSmall;
            @memcpy(dst[self.offset..][0..n], src[src_offset..][0..n]);
            self.offset += n;
            src_offset += n;
            self.count += n;
        }
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *OutputSlicesWriter = @alignCast(@fieldParentPtr("interface", w));
        var written: usize = 0;
        for (data) |chunk| {
            self.writeBytes(chunk) catch return error.WriteFailed;
            written += chunk.len;
        }
        const last = data[data.len - 1];
        for (0..splat - 1) |_| {
            self.writeBytes(last) catch return error.WriteFailed;
            written += last.len;
        }
        return written;
    }

    fn flush(_: *std.Io.Writer) std.Io.Writer.Error!void {}

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        _ = w;
        if (preserve != 0 or capacity != 0) return error.WriteFailed;
    }
};

const VisibleTerm = struct {
    term_index: usize,
    fetch_index: usize,
    skip: u64,
    len: usize,
};

const FetchResult = struct {
    order_index: usize,
    slot_index: usize,
    result: anyerror!usize,
};

const PipelineCompletion = union(enum) {
    fetch: FetchResult,
};

const PipelineSelect = std.Io.Select(PipelineCompletion);

const ReadScratch = struct {
    visible_terms: []VisibleTerm = &.{},
    visible_term_len: usize = 0,
    term_storage: []u8 = &.{},
    term_partial_storage: []u8 = &.{},
    term_grouped_storage: []u8 = &.{},
    fetch_order: []usize = &.{},
    fetch_order_len: usize = 0,
    fetch_term_counts: []u32 = &.{},
    select_buffer: []PipelineCompletion = &.{},
    body_slots: []XorbFetchWindow.BodySlot = &.{},
    body_storage: []u8 = &.{},

    fn init(allocator: std.mem.Allocator, index: reconstruction.Index, offset: u64, take: u64) !ReadScratch {
        var scratch: ReadScratch = .{};
        errdefer scratch.deinit(allocator);

        scratch.visible_terms = try allocator.alloc(VisibleTerm, index.terms.len);
        scratch.fetch_order = try allocator.alloc(usize, index.fetches.len);
        scratch.fetch_term_counts = try allocator.alloc(u32, index.fetches.len);
        @memset(scratch.fetch_term_counts, 0);

        var max_body_bytes: usize = 0;
        var max_term_bytes: usize = 0;
        var output_cursor: u64 = 0;
        const read_end = offset + take;
        for (index.terms, 0..) |term, term_index| {
            if (output_cursor == take) break;

            const term_start = term.file_start;
            const term_end = term_start + term.visible_length;
            const visible_start = @max(offset, term_start);
            const visible_end = @min(read_end, term_end);
            if (visible_start >= visible_end) continue;

            const visible_len: usize = @intCast(visible_end - visible_start);
            scratch.visible_terms[scratch.visible_term_len] = .{
                .term_index = term_index,
                .fetch_index = term.fetch_index,
                .skip = visible_start - term_start,
                .len = visible_len,
            };
            scratch.visible_term_len += 1;

            if (scratch.fetch_term_counts[term.fetch_index] == 0) {
                const info = index.fetches[term.fetch_index].info;
                const body_bytes: usize = @intCast(info.url_range.end - info.url_range.start + 1);
                scratch.fetch_order[scratch.fetch_order_len] = term.fetch_index;
                scratch.fetch_order_len += 1;
                max_body_bytes = @max(max_body_bytes, body_bytes);
            }
            scratch.fetch_term_counts[term.fetch_index] += 1;
            max_term_bytes = @max(max_term_bytes, visible_len);
            output_cursor += visible_len;
        }
        if (output_cursor != take) return error.InvalidReconstruction;

        const slot_count: usize = @min(BODY_PREFETCH, scratch.fetch_order_len);
        const select_count = @max(slot_count, @as(usize, 1));
        scratch.select_buffer = try allocator.alloc(PipelineCompletion, select_count);
        scratch.body_slots = try allocator.alloc(XorbFetchWindow.BodySlot, slot_count);

        scratch.body_storage = try allocator.alloc(u8, slot_count * max_body_bytes);
        for (scratch.body_slots, 0..) |*slot, i| {
            slot.* = .{ .body = scratch.body_storage[i * max_body_bytes ..][0..max_body_bytes] };
        }

        scratch.term_storage = try allocator.alloc(u8, max_term_bytes);
        scratch.term_partial_storage = try allocator.alloc(u8, xorb_frames.MAX_DECODED_FRAME_BYTES);
        scratch.term_grouped_storage = try allocator.alloc(u8, xorb_frames.MAX_DECODED_FRAME_BYTES);

        return scratch;
    }

    fn deinit(self: *ReadScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.term_grouped_storage);
        allocator.free(self.term_partial_storage);
        allocator.free(self.term_storage);
        allocator.free(self.body_storage);
        allocator.free(self.body_slots);
        allocator.free(self.select_buffer);
        allocator.free(self.fetch_term_counts);
        allocator.free(self.fetch_order);
        allocator.free(self.visible_terms);
        self.* = undefined;
    }

    fn terms(self: *const ReadScratch) []const VisibleTerm {
        return self.visible_terms[0..self.visible_term_len];
    }

    fn order(self: *const ReadScratch) []const usize {
        return self.fetch_order[0..self.fetch_order_len];
    }
};

const FetchTask = struct {
    fn run(state: *State, io: std.Io, client: *std.http.Client, info: reconstruction.FetchInfo, order_index: usize, slot_index: usize, slot: []u8) FetchResult {
        return .{
            .order_index = order_index,
            .slot_index = slot_index,
            .result = fetchInto(state, io, client, info, slot),
        };
    }

    fn fetchInto(state: *State, io: std.Io, client: *std.http.Client, info: reconstruction.FetchInfo, slot: []u8) !usize {
        var range_buffer: [64]u8 = undefined;
        const range_header = std.fmt.bufPrint(&range_buffer, "bytes={d}-{d}", .{ info.url_range.start, info.url_range.end }) catch unreachable;
        const request_bytes = info.url_range.end - info.url_range.start + 1;

        try state.xorb_slots.wait(io);
        defer state.xorb_slots.post(io);

        var req = try client.request(.GET, try std.Uri.parse(info.url), .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        });
        defer req.deinit();

        try req.sendBodiless();

        var head_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&head_buffer);
        if (res.head.status != .ok and res.head.status != .partial_content) return error.RequestFailed;

        const reader = res.reader(&.{});
        const content_length: usize = @intCast(res.head.content_length orelse request_bytes);
        if (content_length > request_bytes) return error.ResponseTooLarge;
        if (content_length > slot.len) return error.ResponseTooLarge;
        try reader.readSliceAll(slot[0..content_length]);
        state.stats.addXorbBytes(content_length);
        return content_length;
    }
};

const XorbFetchWindow = struct {
    const Body = struct {
        fetch_index: usize,
        slot_index: usize,
        bytes: []const u8,
    };

    const BodySlot = struct {
        body: []u8,
        fetch_index: ?usize = null,
        order_index: usize = 0,
        result: ?anyerror!usize = null,
        term_count: u32 = 0,
        remaining_terms: u32 = 0,
        len: usize = 0,

        fn begin(self: *BodySlot, fetch_index: usize, order_index: usize) void {
            self.fetch_index = fetch_index;
            self.order_index = order_index;
            self.result = null;
            self.term_count = 0;
            self.remaining_terms = 0;
            self.len = 0;
        }

        fn reset(self: *BodySlot) void {
            const body = self.body;
            self.* = .{ .body = body };
        }
    };

    state: *State,
    io: std.Io,
    client: *std.http.Client,
    fetches: []const reconstruction.FetchEntry,
    order: []const usize,
    body_slots: []BodySlot,
    next_start: usize = 0,
    in_flight: usize = 0,

    fn init(state: *State, request: ReadRequest, fetches: []const reconstruction.FetchEntry, scratch: *ReadScratch) XorbFetchWindow {
        return .{
            .state = state,
            .io = request.io,
            .client = request.client,
            .fetches = fetches,
            .order = scratch.order(),
            .body_slots = scratch.body_slots,
        };
    }

    fn fill(self: *XorbFetchWindow, select: *PipelineSelect) !void {
        while (self.in_flight < self.body_slots.len and self.next_start < self.order.len) {
            const order_index = self.next_start;
            const fetch_index = self.order[order_index];
            if (fetch_index >= self.fetches.len) return error.InvalidReconstruction;
            const slot_index = self.idleSlot() orelse return;
            self.body_slots[slot_index].begin(fetch_index, order_index);

            select.concurrent(.fetch, FetchTask.run, .{
                self.state,
                self.io,
                self.client,
                self.fetches[fetch_index].info,
                order_index,
                slot_index,
                self.body_slots[slot_index].body,
            }) catch |err| switch (err) {
                error.ConcurrencyUnavailable => {
                    const result = FetchTask.run(
                        self.state,
                        self.io,
                        self.client,
                        self.fetches[fetch_index].info,
                        order_index,
                        slot_index,
                        self.body_slots[slot_index].body,
                    );
                    self.next_start += 1;
                    try self.storeCompletion(result);
                    continue;
                },
            };
            self.next_start += 1;
            self.in_flight += 1;
        }
    }

    fn waitForBody(self: *XorbFetchWindow, select: *PipelineSelect, fetch_term_counts: []const u32, fetch_index: usize) !Body {
        try self.fill(select);
        var body = try self.readyBody(fetch_term_counts, fetch_index);
        while (body == null) {
            if (self.in_flight == 0) return error.InvalidReconstruction;
            try self.awaitFetch(select);
            try self.fill(select);
            body = try self.readyBody(fetch_term_counts, fetch_index);
        }
        return body.?;
    }

    fn complete(self: *XorbFetchWindow, result: FetchResult) !void {
        if (self.in_flight == 0) return error.InvalidReconstruction;
        self.in_flight -= 1;
        try self.storeCompletion(result);
    }

    fn storeCompletion(self: *XorbFetchWindow, result: FetchResult) !void {
        if (result.slot_index >= self.body_slots.len) return error.InvalidReconstruction;
        const slot = &self.body_slots[result.slot_index];
        if (slot.fetch_index == null or slot.order_index != result.order_index or slot.result != null) return error.InvalidReconstruction;
        if (result.result) |len| {
            slot.len = len;
        } else |err| {
            slot.len = 0;
            slot.result = err;
            return;
        }
        slot.result = result.result;
    }

    fn activateCompleted(self: *XorbFetchWindow, fetch_term_counts: []const u32) !void {
        for (self.body_slots, 0..) |*slot, slot_index| {
            const fetch_index = slot.fetch_index orelse continue;
            if (slot.result == null or slot.remaining_terms != 0) continue;

            const len = slot.result.? catch |err| {
                self.releaseSlot(slot_index);
                return err;
            };
            if (len != slot.len) return error.InvalidReconstruction;
            if (fetch_index >= fetch_term_counts.len) return error.InvalidReconstruction;
            const term_count = fetch_term_counts[fetch_index];
            if (term_count == 0) return error.InvalidReconstruction;
            slot.term_count = term_count;
            slot.remaining_terms = term_count;
        }
    }

    fn readyBody(self: *XorbFetchWindow, fetch_term_counts: []const u32, fetch_index: usize) !?Body {
        try self.activateCompleted(fetch_term_counts);
        return self.findBody(fetch_index);
    }

    fn finishBodyUse(self: *XorbFetchWindow, slot_index: usize) !void {
        const slot = &self.body_slots[slot_index];
        if (slot.remaining_terms == 0) return error.InvalidReconstruction;
        slot.remaining_terms -= 1;
        if (slot.remaining_terms == 0) self.releaseSlot(slot_index);
    }

    fn awaitFetch(self: *XorbFetchWindow, select: *PipelineSelect) !void {
        switch (try select.await()) {
            .fetch => |result| try self.complete(result),
        }
    }

    fn findBody(self: *const XorbFetchWindow, fetch_index: usize) ?Body {
        for (self.body_slots, 0..) |slot, slot_index| {
            const slot_fetch_index = slot.fetch_index orelse continue;
            if (slot_fetch_index != fetch_index or slot.remaining_terms == 0) continue;
            return .{
                .fetch_index = slot_fetch_index,
                .slot_index = slot_index,
                .bytes = slot.body[0..slot.len],
            };
        }
        return null;
    }

    fn idleSlot(self: *const XorbFetchWindow) ?usize {
        for (self.body_slots, 0..) |slot, slot_index| {
            if (slot.fetch_index == null) return slot_index;
        }
        return null;
    }

    fn releaseSlot(self: *XorbFetchWindow, slot_index: usize) void {
        self.body_slots[slot_index].reset();
    }
};

const OrderedTermReader = struct {
    state: *State,
    io: std.Io,
    index: reconstruction.Index,
    terms: []const VisibleTerm,
    fetch_term_counts: []const u32,
    term_buffer: []u8,
    partial: []u8,
    grouped: []u8,
    select: PipelineSelect,
    fetch_window: XorbFetchWindow,
    remaining: u64,
    next_term: usize = 0,
    buffered_len: usize = 0,
    buffered_offset: usize = 0,
    interface: std.Io.Reader,

    fn init(state: *State, request: ReadRequest, index: reconstruction.Index, scratch: *ReadScratch) OrderedTermReader {
        var reader: OrderedTermReader = .{
            .state = state,
            .io = request.io,
            .index = index,
            .terms = scratch.terms(),
            .fetch_term_counts = scratch.fetch_term_counts,
            .term_buffer = scratch.term_storage,
            .partial = scratch.term_partial_storage,
            .grouped = scratch.term_grouped_storage,
            .select = .init(request.io, scratch.select_buffer),
            .fetch_window = undefined,
            .remaining = request.take,
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
        };
        reader.fetch_window = XorbFetchWindow.init(state, request, index.fetches, scratch);
        return reader;
    }

    fn deinit(self: *OrderedTermReader) void {
        while (self.select.cancel()) |_| {}
    }

    fn stream(r: *std.Io.Reader, writer: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *OrderedTermReader = @alignCast(@fieldParentPtr("interface", r));
        if (self.remaining == 0) return error.EndOfStream;
        const take = limit.minInt64(self.remaining);
        if (take == 0) return error.EndOfStream;

        return self.drain(writer, @intCast(take)) catch |err| return switch (err) {
            error.WriteFailed, error.DestinationTooSmall => error.WriteFailed,
            else => error.ReadFailed,
        };
    }

    fn drain(self: *OrderedTermReader, writer: *std.Io.Writer, limit: usize) !usize {
        var written: usize = 0;
        while (written < limit and self.remaining != 0) {
            if (self.buffered_offset == self.buffered_len) {
                self.buffered_len = try self.decodeNextTerm();
                self.buffered_offset = 0;
            }

            const available = self.buffered_len - self.buffered_offset;
            const n = @min(available, limit - written);
            try writer.writeAll(self.term_buffer[self.buffered_offset..][0..n]);
            self.buffered_offset += n;
            written += n;
            self.remaining -= n;
        }
        return written;
    }

    fn decodeNextTerm(self: *OrderedTermReader) !usize {
        if (self.next_term >= self.terms.len) return error.EndOfStream;

        const plan = self.terms[self.next_term];
        if (plan.len > self.term_buffer.len) return error.ScratchTooSmall;
        const body = try self.fetch_window.waitForBody(&self.select, self.fetch_term_counts, plan.fetch_index);
        errdefer _ = self.fetch_window.finishBodyUse(body.slot_index) catch {};

        const term = self.index.terms[plan.term_index];
        const fetch = self.index.fetches[plan.fetch_index];
        if (term.range.start < fetch.info.range.start or term.range.end > fetch.info.range.end) return error.InvalidReconstruction;

        var fixed: std.Io.Reader = .fixed(body.bytes);
        var frames = xorb_frames.FrameReader.init(&fixed);
        try discardFetchPrefix(&frames, fetch.info.range.start, term.range.start);
        const len = try self.copyTermFrames(&frames, term, plan);
        try self.fetch_window.finishBodyUse(body.slot_index);
        self.next_term += 1;
        return len;
    }

    fn discardFetchPrefix(frames: *xorb_frames.FrameReader, first_chunk: u64, term_first_chunk: u64) !void {
        var chunk_index = first_chunk;
        while (chunk_index < term_first_chunk) : (chunk_index += 1) {
            const frame = try frames.next();
            try frames.reader.discardAll(frame.compressed_size);
        }
    }

    fn copyTermFrames(self: *OrderedTermReader, frames: *xorb_frames.FrameReader, term: reconstruction.TermEntry, plan: VisibleTerm) !usize {
        const visible_start = term.base_skip + plan.skip;
        const visible_end = visible_start + @as(u64, @intCast(plan.len));
        const dst = self.term_buffer[0..plan.len];
        var decoder: xorb_frames.FrameDecoder = .{ .stats = &self.state.stats, .io = self.io };

        var term_offset: u64 = 0;
        var written: usize = 0;
        var chunk_index = term.range.start;
        while (chunk_index < term.range.end and written < plan.len) : (chunk_index += 1) {
            if (term_offset >= visible_end) break;
            const frame = try frames.next();
            const chunk_start = term_offset;
            const chunk_end = chunk_start + frame.uncompressed_size;
            term_offset = chunk_end;

            if (chunk_end <= visible_start) {
                try frames.reader.discardAll(frame.compressed_size);
                continue;
            }

            written += try self.copyFramePart(frames, &decoder, frame, chunk_start, visible_start, visible_end, dst[written..]);
        }

        if (written != plan.len) return error.InvalidReconstruction;
        return written;
    }

    fn copyFramePart(
        self: *OrderedTermReader,
        frames: *xorb_frames.FrameReader,
        decoder: *xorb_frames.FrameDecoder,
        frame: xorb_frames.Frame,
        chunk_start: u64,
        visible_start: u64,
        visible_end: u64,
        dst: []u8,
    ) !usize {
        const chunk_end = chunk_start + frame.uncompressed_size;
        const start_in_chunk: usize = @intCast(@max(visible_start, chunk_start) - chunk_start);
        const end_in_chunk: usize = @intCast(@min(visible_end, chunk_end) - chunk_start);
        const len = end_in_chunk - start_in_chunk;

        if (start_in_chunk == 0 and end_in_chunk == frame.uncompressed_size) {
            try decoder.decode(frames.reader, frame, dst[0..len], self.grouped);
            return len;
        }

        if (frame.uncompressed_size > self.partial.len) return error.ChunkTooLarge;
        const decoded = self.partial[0..frame.uncompressed_size];
        try decoder.decode(frames.reader, frame, decoded, self.grouped);
        @memcpy(dst[0..len], decoded[start_in_chunk..end_in_chunk]);
        return len;
    }
};
