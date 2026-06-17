const std = @import("std");
const lz4 = @import("lz4.zig");
const bg4 = @import("bg4.zig");

const log = std.log.scoped(.@"zml/io/vfs/xet");

/// Block the calling worker thread for `ms` milliseconds via libc.
/// We don't have a `std.Io` handle in here and `std.Thread.sleep` is gone
/// in Zig 0.16; this is the most direct portable replacement.
fn sleepMs(ms: u64) void {
    const ts: std.posix.timespec = .{
        .sec = @intCast(ms / 1000),
        .nsec = @intCast((ms % 1000) * std.time.ns_per_ms),
    };
    _ = std.c.nanosleep(&ts, null);
}

pub const Repo = struct {
    repo: []const u8,
    model: []const u8,
    rev: []const u8,
    path: []const u8,
};

const Term = struct {
    hash: []const u8,
    unpacked_length: usize,
    range: struct { start: usize, end: usize },
};

const FetchUrl = struct {
    range: struct { start: usize, end: usize },
    url: []const u8,
    url_range: struct { start: usize, end: usize },
};

const ReconstructionResponse = struct {
    offset_into_first_range: usize,
    terms: []const Term,
    fetch_info: std.json.ArrayHashMap([]const FetchUrl) = .{},
};

const CasAuth = struct {
    casUrl: []const u8,
    accessToken: []const u8,
};

const chunk_header_size = 8;
const max_xorb_body_size = 64 * 1024 * 1024;

const Chunk = struct {
    compressed_size: u32,
    uncompressed_size: u32,
    compression_type: u8,
    compressed_data: []const u8,
};

/// Decompresses `chunk` into `dst[0..chunk.uncompressed_size]`. `dst` must
/// be at least that large. Composes via `lz4.BlockReader` / `bg4.DegroupWriter`
/// over `std.Io.Reader` / `std.Io.Writer`.
fn decompressChunk(chunk: Chunk, dst: []u8) ![]u8 {
    const n: usize = chunk.uncompressed_size;
    if (dst.len < n) return error.OutputTooSmall;
    const out = dst[0..n];
    switch (chunk.compression_type) {
        0 => {
            if (chunk.compressed_data.len != n) return error.SizeMismatch;
            @memcpy(out, chunk.compressed_data);
        },
        1 => {
            var src: std.Io.Reader = .fixed(chunk.compressed_data);
            var w: std.Io.Writer = .fixed(out);
            var rd = lz4.BlockReader.init(&src, chunk.compressed_size, n);
            _ = rd.interface.streamRemaining(&w) catch return error.CorruptedData;
        },
        2 => {
            var src: std.Io.Reader = .fixed(chunk.compressed_data);
            var dw = bg4.DegroupWriter.init(out);
            var rd = lz4.BlockReader.init(&src, chunk.compressed_size, n);
            _ = rd.interface.streamRemaining(&dw.interface) catch return error.CorruptedData;
            dw.interface.flush() catch return error.CorruptedData;
        },
        else => return error.InvalidCompressionType,
    }
    return out;
}

// ── HTTP helpers ────────────────────────────────────────────────────────────

fn bearerAuth(token: []const u8, buf: []u8) ![]u8 {
    return std.fmt.bufPrint(buf, "Bearer {s}", .{std.mem.trim(u8, token, " \t\n\r")}) catch error.TokenTooLong;
}

/// Fetch the `X-Xet-Hash` for the file at `repo`. Returns
/// `error.XetHashNotFound` for plain-LFS files. Caller owns the result.
pub fn fetchFileId(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: Repo,
    hf_token: []const u8,
) ![]const u8 {
    var auth_buf: [1024]u8 = undefined;
    const auth = try bearerAuth(hf_token, &auth_buf);
    var url_buf: [4096]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buf, "https://huggingface.co/{s}/{s}/resolve/{s}/{s}", .{ repo.repo, repo.model, repo.rev, repo.path });
    const uri: std.Uri = try .parse(url);
    var req = try client.request(.GET, uri, .{
        .redirect_behavior = .unhandled,
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    var it = res.head.iterateHeaders();
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "X-Xet-Hash")) return try allocator.dupe(u8, h.value);
    }
    return error.XetHashNotFound;
}

/// CAS auth (URL + bearer token) cached across `fetchRange` calls. HF rate-limits
/// `/xet-read-token/*` aggressively (3000 req / 5 min per IP); without caching, a
/// bulk-weight load on a fast box exhausts the quota mid-download.
pub const CasAuthCache = struct {
    mutex: std.Io.Mutex = .init,
    url: ?[]u8 = null,
    token: ?[]u8 = null,
    expires_at: std.Io.Timestamp = .zero,

    /// Conservative TTL: HF's xet-read tokens are valid much longer, but we don't
    /// parse the expiry. 4 min keeps us well clear of any reasonable lifetime while
    /// still amortizing the round-trip over many fetches.
    const ttl: std.Io.Duration = .fromSeconds(4 * 60);

    pub fn deinit(self: *CasAuthCache, allocator: std.mem.Allocator) void {
        if (self.url) |u| allocator.free(u);
        if (self.token) |t| allocator.free(t);
        self.* = .{};
    }

    /// Returns owned copies of (url, token). Caller frees both.
    fn get(
        self: *CasAuthCache,
        allocator: std.mem.Allocator,
        io: std.Io,
        client: *std.http.Client,
        repo: Repo,
        hf_token: []const u8,
    ) !struct { url: []u8, token: []u8 } {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        const now: std.Io.Timestamp = .now(io, .real);
        if (self.url == null or now.toSeconds() >= self.expires_at.toSeconds()) {
            const fresh = try fetchCasAuth(allocator, client, repo, hf_token);
            if (self.url) |u| allocator.free(u);
            if (self.token) |t| allocator.free(t);
            self.url = fresh.url;
            self.token = fresh.token;
            self.expires_at = now.addDuration(ttl);
        }
        return .{
            .url = try allocator.dupe(u8, self.url.?),
            .token = try allocator.dupe(u8, self.token.?),
        };
    }
};

fn fetchCasAuth(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: Repo,
    hf_token: []const u8,
) !struct { url: []u8, token: []u8 } {
    var auth_buf: [1024]u8 = undefined;
    const auth = try bearerAuth(hf_token, &auth_buf);
    var url_buf: [4096]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buf, "https://huggingface.co/api/models/{s}/{s}/xet-read-token/{s}", .{ repo.repo, repo.model, repo.rev });
    const uri: std.Uri = try .parse(url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [4 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok) return error.CasTokenFailed;
    const body = try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 128 * 1024);
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(CasAuth, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return .{
        .url = try allocator.dupe(u8, parsed.value.casUrl),
        .token = try allocator.dupe(u8, parsed.value.accessToken),
    };
}

fn fetchReconstruction(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    cas_url: []const u8,
    cas_auth: []const u8,
    file_id: []const u8,
    range_start: u64,
    range_end_inclusive: u64,
) !std.json.Parsed(ReconstructionResponse) {
    var url_buf: [4096]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buf, "{s}/v1/reconstructions/{s}", .{ cas_url, file_id });
    var range_buf: [64]u8 = undefined;
    const range = try std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ range_start, range_end_inclusive });
    const uri: std.Uri = try .parse(url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = cas_auth },
        },
        .extra_headers = &.{.{ .name = "Range", .value = range }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok and res.head.status != .partial_content) return error.ReconstructionFailed;
    const body = try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 64 * 1024 * 1024);
    defer allocator.free(body);
    return try std.json.parseFromSlice(ReconstructionResponse, allocator, body, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    });
}

// ── ChunkIterator ──────────────────────────────────────────────────────────

/// Walks a fully-buffered xorb body, yielding chunks one at a time.
pub const ChunkIterator = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn next(self: *ChunkIterator) !?Chunk {
        if (self.pos >= self.data.len) return null;
        if (self.pos + chunk_header_size > self.data.len) return error.TruncatedData;
        const hdr = self.data[self.pos..][0..chunk_header_size];
        if (hdr[0] != 0) return error.InvalidVersion;
        const csize: u32 = @as(u32, hdr[1]) | (@as(u32, hdr[2]) << 8) | (@as(u32, hdr[3]) << 16);
        const ctype = hdr[4];
        if (ctype > 2) return error.InvalidCompressionType;
        const usize_uncomp: u32 = @as(u32, hdr[5]) | (@as(u32, hdr[6]) << 8) | (@as(u32, hdr[7]) << 16);
        const body_start = self.pos + chunk_header_size;
        if (body_start + csize > self.data.len) return error.TruncatedData;
        self.pos = body_start + csize;
        return .{
            .compressed_size = csize,
            .uncompressed_size = usize_uncomp,
            .compression_type = ctype,
            .compressed_data = self.data[body_start..][0..csize],
        };
    }
};

// ── HTTP range GET into a slice (with retry) ───────────────────────────────

fn httpRangeGetIntoSlice(
    client: *std.http.Client,
    url: []const u8,
    start: usize,
    end_inclusive: usize,
    dst: []u8,
) !void {
    const max_attempts: u32 = 6;
    var backoff_ms: u64 = 250;
    var attempt: u32 = 0;
    while (true) : (attempt += 1) {
        tryHttpRangeGetIntoSlice(client, url, start, end_inclusive, dst) catch |err| {
            if (err == error.XorbFetchPermanent or attempt + 1 >= max_attempts) {
                log.err("xorb fetch failed after {d} attempt(s): {s} (url={s})", .{ attempt + 1, @errorName(err), url });
                return error.XorbFetchFailed;
            }
            log.warn("xorb fetch attempt {d}/{d} failed: {s} — retrying in {d}ms", .{ attempt + 1, max_attempts, @errorName(err), backoff_ms });
            sleepMs(backoff_ms);
            backoff_ms *= 2;
            continue;
        };
        return;
    }
}

/// Per-socket read deadline. CDN edges occasionally go silent mid-stream
/// without sending FIN; without this, readv() would block forever.
const xorb_socket_recv_timeout_s: i64 = 30;

fn tryHttpRangeGetIntoSlice(
    client: *std.http.Client,
    url: []const u8,
    start: usize,
    end_inclusive: usize,
    dst: []u8,
) !void {
    const uri: std.Uri = try .parse(url);
    var range_buf: [64]u8 = undefined;
    const range = try std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ start, end_inclusive });

    var req = try client.request(.GET, uri, .{
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
        .extra_headers = &.{.{ .name = "Range", .value = range }},
    });
    defer req.deinit();

    if (req.connection) |conn| {
        const fd = conn.stream_reader.stream.socket.handle;
        const tv: std.posix.timeval = .{ .sec = xorb_socket_recv_timeout_s, .usec = 0 };
        std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&tv)) catch {};
    }

    try req.sendBodiless();

    var redirect_buf: [8 * 1024]u8 = undefined;
    var res = req.receiveHead(&redirect_buf) catch |err| {
        log.warn("xorb receiveHead failed: {s}", .{@errorName(err)});
        return error.XorbFetchTransient;
    };
    if (res.head.status != .ok and res.head.status != .partial_content) {
        const code = @intFromEnum(res.head.status);
        if (code == 408 or code == 425 or code == 429 or (code >= 500 and code < 600)) {
            log.warn("xorb HTTP {d} (transient)", .{code});
            return error.XorbFetchTransient;
        }
        log.err("xorb HTTP {d} (permanent) from {s}", .{ code, url });
        return error.XorbFetchPermanent;
    }
    var transfer_buf: [16 * 1024]u8 = undefined;
    const reader = res.reader(&transfer_buf);
    reader.readSliceAll(dst) catch |err| {
        log.warn("xorb body read failed: {s}", .{@errorName(err)});
        return error.XorbFetchTransient;
    };
}

// ── Range fetch (worker-based, fills dst synchronously) ────────────────────

/// One term's contribution to the output: a contiguous slice of `dst[]`
/// produced by decompressing a chunk-range of one FetchUrl's body.
const TermPlan = struct {
    chunk_start: usize,
    chunk_end: usize, // exclusive
    byte_skip: usize, // bytes to drop inside the first contributing chunk
    wanted_len: usize, // bytes to copy into dst
    dst_off: usize, // absolute offset in caller's dst[]
    bytes_written: usize = 0,
};

/// One HTTP body to download. All TermPlans share the same FetchUrl.
const FetchTask = struct {
    xorb_hash: []const u8,
    url: []const u8,
    url_range_start: usize,
    url_range_end: usize, // inclusive
    byte_len: usize,
    chunk_idx_start: usize,
    term_plans: std.ArrayList(TermPlan),
};

fn freeFetchTasks(allocator: std.mem.Allocator, tasks: *std.ArrayList(FetchTask)) void {
    for (tasks.items) |*t| t.term_plans.deinit(allocator);
    tasks.deinit(allocator);
}

/// Walk `resp.terms` once. For each term, find its single covering FetchUrl
/// in `resp.fetch_info`, bucket a TermPlan under the matching FetchTask.
fn buildFetchTasks(
    allocator: std.mem.Allocator,
    resp: ReconstructionResponse,
    dst_len: usize,
) !std.ArrayList(FetchTask) {
    var tasks: std.ArrayList(FetchTask) = .empty;
    errdefer freeFetchTasks(allocator, &tasks);

    var dst_off: usize = 0;
    var skip_head: usize = resp.offset_into_first_range;

    for (resp.terms) |term| {
        if (dst_off >= dst_len) break;
        const avail = term.unpacked_length - skip_head;
        const wanted = @min(avail, dst_len - dst_off);

        const fus = resp.fetch_info.map.get(term.hash) orelse return error.NoFetchUrlsForXorb;
        const fu = for (fus) |*u| {
            if (term.range.start >= u.range.start and term.range.end <= u.range.end) break u;
        } else return error.NoFetchUrlCoversTerm;

        // Linear scan: N is small (tens or hundreds of terms per fetchRange call) and the
        // cost is dwarfed by the HTTP round-trip per task.
        var task_idx: usize = tasks.items.len;
        for (tasks.items, 0..) |*t, i| {
            if (t.url_range_start == fu.url_range.start and
                t.url_range_end == fu.url_range.end and
                std.mem.eql(u8, t.xorb_hash, term.hash))
            {
                task_idx = i;
                break;
            }
        }
        if (task_idx == tasks.items.len) {
            try tasks.append(allocator, .{
                .xorb_hash = term.hash,
                .url = fu.url,
                .url_range_start = fu.url_range.start,
                .url_range_end = fu.url_range.end,
                .byte_len = fu.url_range.end - fu.url_range.start + 1,
                .chunk_idx_start = fu.range.start,
                .term_plans = .empty,
            });
        }

        try tasks.items[task_idx].term_plans.append(allocator, .{
            .chunk_start = term.range.start,
            .chunk_end = term.range.end,
            .byte_skip = skip_head,
            .wanted_len = wanted,
            .dst_off = dst_off,
        });

        dst_off += wanted;
        skip_head = 0;
    }
    return tasks;
}

pub const FetchPool = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    client: *std.http.Client,
    workers: []Worker,
    group: std.Io.Group,
    job_queue: std.Io.Queue(*Job),
    queue_buf: []*Job,

    pub const Options = struct {
        workers: usize,
        queue_capacity: usize,
        body_size: usize = max_xorb_body_size,
        scratch_size: usize = 256 * 1024, // 2x the largest chunk size in practice (128 KiB) - Encountered a few chunks >128 KiB in the wild, so this is a safe upper bound.
    };

    const Worker = struct {
        body: []u8,
        scratch: []u8,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        client: *std.http.Client,
        opts: Options,
    ) !*FetchPool {
        const workers = try allocator.alloc(Worker, opts.workers);
        errdefer allocator.free(workers);

        var made: usize = 0;
        errdefer for (workers[0..made]) |*w| {
            allocator.free(w.body);
            allocator.free(w.scratch);
        };
        for (workers) |*w| {
            const body = try allocator.alloc(u8, opts.body_size);
            errdefer allocator.free(body);
            const scratch = try allocator.alloc(u8, opts.scratch_size);
            errdefer allocator.free(scratch);
            w.* = .{ .body = body, .scratch = scratch };
            made += 1;
        }

        const queue_buf = try allocator.alloc(*Job, opts.queue_capacity);
        errdefer allocator.free(queue_buf);

        const self = try allocator.create(FetchPool);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .client = client,
            .workers = workers,
            .group = .init,
            .job_queue = std.Io.Queue(*Job).init(queue_buf),
            .queue_buf = queue_buf,
        };
        errdefer self.group.cancel(io);
        for (workers) |*w| try self.group.concurrent(io, fetchWorker, .{ self, w });
        return self;
    }

    pub fn deinit(self: *FetchPool) void {
        self.job_queue.close(self.io);
        self.group.await(self.io) catch {};
        for (self.workers) |*w| {
            self.allocator.free(w.body);
            self.allocator.free(w.scratch);
        }
        self.allocator.free(self.workers);
        self.allocator.free(self.queue_buf);
        self.allocator.destroy(self);
    }
};

pub const Batch = struct {
    dst: []u8,
    pending: usize,
    err: ?anyerror = null,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,

    fn finishOne(self: *Batch, io: std.Io, e: ?anyerror) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (e) |x| if (self.err == null) {
            self.err = x;
        };
        self.pending -= 1;
        if (self.pending == 0) self.cond.signal(io);
    }

    fn wait(self: *Batch, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.pending != 0) self.cond.wait(io, &self.mutex) catch {};
    }
};

const Job = struct {
    task: *FetchTask,
    batch: *Batch,
};

fn fetchWorker(pool: *FetchPool, w: *FetchPool.Worker) void {
    while (true) {
        const job = pool.job_queue.getOne(pool.io) catch break;
        if (job.batch.err != null) {
            job.batch.finishOne(pool.io, null);
            continue;
        }
        const err = processJob(pool, w, job);
        job.batch.finishOne(pool.io, err);
    }
}

fn processJob(pool: *FetchPool, w: *FetchPool.Worker, job: *Job) ?anyerror {
    const task = job.task;
    if (task.byte_len > w.body.len) return error.XorbTooLarge;
    // log.warn("fetching {d} bytes from {s} (range={d}-{d})", .{ task.byte_len, task.url, task.url_range_start, task.url_range_end });
    const body = w.body[0..task.byte_len];
    runTask(pool.client, task, job.batch.dst, body, w.scratch) catch |e| return e;
    return null;
}

fn runTask(
    client: *std.http.Client,
    task: *FetchTask,
    dst: []u8,
    body: []u8,
    scratch: []u8,
) !void {
    try httpRangeGetIntoSlice(client, task.url, task.url_range_start, task.url_range_end, body);

    var it: ChunkIterator = .{ .data = body };
    var chunk_idx: usize = task.chunk_idx_start;
    while (try it.next()) |chunk| : (chunk_idx += 1) {
        // Find applicable plans for this chunk.
        var only_plan: ?*TermPlan = null;
        var multiple_plans = false;
        for (task.term_plans.items) |*tp| {
            if (chunk_idx < tp.chunk_start or chunk_idx >= tp.chunk_end) continue;
            if (tp.bytes_written >= tp.wanted_len) continue;
            if (only_plan != null) {
                multiple_plans = true;
                break;
            }
            only_plan = tp;
        }
        if (only_plan == null) continue;

        // Fast path: a single plan needs this chunk in full: decompress straight into dst,
        // skipping the scratch+memcpy round-trip.
        if (!multiple_plans) {
            const tp = only_plan.?;
            const needs_skip = chunk_idx == tp.chunk_start and tp.byte_skip != 0;
            const remaining = tp.wanted_len - tp.bytes_written;
            if (!needs_skip and chunk.uncompressed_size <= remaining) {
                const slot = dst[tp.dst_off + tp.bytes_written ..][0..chunk.uncompressed_size];
                _ = try decompressChunk(chunk, slot);
                tp.bytes_written += chunk.uncompressed_size;
                continue;
            }
        }

        // Fallback: decode once into scratch, then fan out @memcpy to each overlapping plan
        // (handles boundary byte_skip, partial-tail truncation, and multi-plan chunks).
        const decoded = try decompressChunk(chunk, scratch);
        for (task.term_plans.items) |*tp| {
            if (chunk_idx < tp.chunk_start or chunk_idx >= tp.chunk_end) continue;
            if (tp.bytes_written >= tp.wanted_len) continue;
            const src_off: usize = if (chunk_idx == tp.chunk_start) tp.byte_skip else 0;
            const avail = decoded.len - src_off;
            const remaining = tp.wanted_len - tp.bytes_written;
            const take = @min(avail, remaining);
            const dst_slice = dst[tp.dst_off + tp.bytes_written ..][0..take];
            @memcpy(dst_slice, decoded[src_off..][0..take]);
            tp.bytes_written += take;
        }
    }
}

/// Synchronously fill `dst[0..]` with `dst.len` bytes starting at file `offset`
/// of the XET-backed file `(repo, file_id)`. Submits one `Job` per HTTP range
/// to the long-lived `pool`; returns when every job in the batch completes
/// (or any worker errors).
pub fn fetchRange(
    pool: *FetchPool,
    cas_cache: *CasAuthCache,
    hf_token: []const u8,
    repo: Repo,
    file_id: []const u8,
    offset: u64,
    dst: []u8,
) !void {
    if (dst.len == 0) return;

    const cas = try cas_cache.get(pool.allocator, pool.io, pool.client, repo, hf_token);
    defer pool.allocator.free(cas.url);
    defer pool.allocator.free(cas.token);

    var auth_buf: [4096]u8 = undefined;
    const cas_auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

    const parsed = try fetchReconstruction(pool.allocator, pool.client, cas.url, cas_auth, file_id, offset, offset + dst.len - 1);
    defer parsed.deinit();

    var tasks = try buildFetchTasks(pool.allocator, parsed.value, dst.len);
    defer freeFetchTasks(pool.allocator, &tasks);

    if (tasks.items.len == 0) return;

    const jobs = try pool.allocator.alloc(Job, tasks.items.len);
    defer pool.allocator.free(jobs);

    //Create the batch and assign each task to a job. The batch will track how many jobs are pending and will wait for all of them to finish.
    var batch: Batch = .{ .dst = dst, .pending = tasks.items.len };
    for (jobs, tasks.items) |*j, *t| j.* = .{ .task = t, .batch = &batch };

    for (jobs) |*j| try pool.job_queue.putOne(pool.io, j);
    batch.wait(pool.io);
    if (batch.err) |e| return e;
}
