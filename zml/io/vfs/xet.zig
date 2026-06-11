const std = @import("std");
const stdx = @import("stdx");
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
    const max_attempts: u32 = 4;
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
    try req.sendBodiless();

    var redirect_buf: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buf);
    if (res.head.status != .ok and res.head.status != .partial_content) {
        const code = @intFromEnum(res.head.status);
        if (code == 429 or (code >= 500 and code < 600)) {
            log.warn("xorb HTTP {d} (transient)", .{code});
            return error.XorbFetchTransient;
        }
        log.err("xorb HTTP {d} (permanent) from {s}", .{ code, url });
        return error.XorbFetchPermanent;
    }
    var transfer_buf: [16 * 1024]u8 = undefined;
    const reader = res.reader(&transfer_buf);
    reader.readSliceAll(dst) catch return error.XorbFetchTransient;
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

const FetchCtx = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    dst: []u8,
    err_mutex: std.Io.Mutex = .init,
    err: ?anyerror = null,

    fn setError(self: *FetchCtx, e: anyerror) void {
        self.err_mutex.lockUncancelable(self.io);
        defer self.err_mutex.unlock(self.io);
        if (self.err == null) self.err = e;
    }
};

fn fetchWorker(ctx: *FetchCtx, task: *FetchTask) void {
    fetchWorkerImpl(ctx, task) catch |e| ctx.setError(e);
}

fn fetchWorkerImpl(ctx: *FetchCtx, task: *FetchTask) !void {
    const body = try ctx.allocator.alloc(u8, task.byte_len);
    defer ctx.allocator.free(body);
    try httpRangeGetIntoSlice(ctx.client, task.url, task.url_range_start, task.url_range_end, body);

    var scratch: [256 * 1024]u8 = undefined;
    var it: ChunkIterator = .{ .data = body };
    var chunk_idx: usize = task.chunk_idx_start;
    while (try it.next()) |chunk| : (chunk_idx += 1) {
        var decoded: []const u8 = &.{};
        var have_decoded = false;
        for (task.term_plans.items) |*tp| {
            if (chunk_idx < tp.chunk_start or chunk_idx >= tp.chunk_end) continue;
            if (tp.bytes_written >= tp.wanted_len) continue;
            if (!have_decoded) {
                decoded = try decompressChunk(chunk, &scratch);
                have_decoded = true;
            }
            const src_off: usize = if (chunk_idx == tp.chunk_start) tp.byte_skip else 0;
            const avail = decoded.len - src_off;
            const remaining = tp.wanted_len - tp.bytes_written;
            const take = @min(avail, remaining);
            const dst_slice = ctx.dst[tp.dst_off + tp.bytes_written ..][0..take];
            @memcpy(dst_slice, decoded[src_off..][0..take]);
            tp.bytes_written += take;
        }
    }
}

/// Synchronously fill `dst[0..]` with `dst.len` bytes starting at file `offset`
/// of the XET-backed file `(repo, file_id)`. Spawns up to `workers` concurrent
/// HTTP fetches via `stdx.Io.LimitedGroup`; returns when every byte is written
/// or any worker errors.
pub fn fetchRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    client: *std.http.Client,
    hf_token: []const u8,
    repo: Repo,
    file_id: []const u8,
    offset: u64,
    dst: []u8,
    workers: usize,
) !void {
    if (dst.len == 0) return;

    const cas = try fetchCasAuth(allocator, client, repo, hf_token);
    defer allocator.free(cas.url);
    defer allocator.free(cas.token);

    var auth_buf: [4096]u8 = undefined;
    const cas_auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

    const parsed = try fetchReconstruction(allocator, client, cas.url, cas_auth, file_id, offset, offset + dst.len - 1);
    defer parsed.deinit();

    var tasks = try buildFetchTasks(allocator, parsed.value, dst.len);
    defer freeFetchTasks(allocator, &tasks);

    var ctx: FetchCtx = .{
        .io = io,
        .allocator = allocator,
        .client = client,
        .dst = dst,
    };
    var group: stdx.Io.LimitedGroup = .init(workers);
    for (tasks.items) |*task| {
        try group.concurrent(io, fetchWorker, .{ &ctx, task });
    }
    try group.await(io);

    if (ctx.err) |e| return e;
}
