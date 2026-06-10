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

// ── Repo / wire types ───────────────────────────────────────────────────────

pub const Repo = struct {
    repo: []const u8,
    model: []const u8,
    rev: []const u8,
    path: []const u8,
};

const Term = struct {
    hash: []const u8,
    unpacked_length: u64,
    range: struct { start: u64, end: u64 },
};

const FetchUrl = struct {
    range: struct { start: u64, end: u64 },
    url: []const u8,
    url_range: struct { start: u64, end: u64 },
};

const ReconstructionResponse = struct {
    offset_into_first_range: u64,
    terms: []const Term,
    fetch_info: std.json.ArrayHashMap([]const FetchUrl) = .{},
};

const CasAuth = struct {
    casUrl: []const u8,
    accessToken: []const u8,
};

// ── Chunk codec (xorb wire format) ──────────────────────────────────────────

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

// ── FileRangeReader (std.Io.Reader implementation) ─────────────────────────

/// Streams the bytes of the file range `[offset, offset+length)` of
/// `(repo, file_id)` through the XET CAS. Drive it with
/// `interface.streamRemaining(writer)` or any other `std.Io.Reader` consumer.
///
/// Internally walks the reconstruction response: a sequence of *terms*
/// (each a sub-range of one xorb), each xorb materialized via a single
/// HTTP range GET, then walked as a sequence of compressed *chunks*
/// emitted one at a time into the destination writer.
pub const FileRangeReader = struct {
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    parsed: std.json.Parsed(ReconstructionResponse),

    /// Total unpacked bytes still to emit before EOF.
    remaining: u64,
    /// Bytes to drop from the head of the next decompressed chunk
    /// (decreases as the first chunk is emitted).
    skip_head: u64,

    term_idx: usize = 0,
    /// Number of output chunks left to read from the current term (0 when no
    /// term is active).
    chunks_remaining_in_term: u64 = 0,

    // Live HTTP state for the current xorb. `http_req` owns the connection;
    // `http_reader` is `&http_req.reader.interface` (so http_req must stay at
    // a stable address while http_reader is in use). Both null between terms.
    http_req: ?std.http.Client.Request = null,
    http_reader: ?*std.Io.Reader = null,

    /// Body-reader scratch (used by stdlib for chunked-encoding/content-length
    /// parsing). One per FileRangeReader; reused across terms.
    xorb_transfer_buf: [16 * 1024]u8 = undefined,
    /// One compressed chunk pulled off the xorb body. Xet caps chunks at
    /// 64 KiB uncompressed; 128 KiB covers worst-case LZ4 inflation.
    xorb_chunk_comp_buf: [128 * 1024]u8 = undefined,
    /// Redirect-tracking scratch for receiveHead.
    xorb_redirect_buf: [8 * 1024]u8 = undefined,

    /// Scratch for one decompressed chunk. Only used when the destination
    /// writer can't fit the chunk directly (head trim, tail clip, or a
    /// writer whose `writableSliceGreedy` is too small).
    chunk_buf: [256 * 1024]u8 = undefined,
    /// `chunk_buf[chunk_pos..chunk_end]` is the slice still pending emission.
    chunk_pos: usize = 0,
    chunk_end: usize = 0,

    interface: std.Io.Reader,

    pub fn init(
        allocator: std.mem.Allocator,
        client: *std.http.Client,
        hf_token: []const u8,
        repo: Repo,
        file_id: []const u8,
        offset: u64,
        length: u64,
    ) !FileRangeReader {
        const cas = try fetchCasAuth(allocator, client, repo, hf_token);
        defer allocator.free(cas.url);
        defer allocator.free(cas.token);

        var cas_auth_buf: [4096]u8 = undefined;
        const cas_auth = std.fmt.bufPrint(&cas_auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

        const parsed = try fetchReconstruction(allocator, client, cas.url, cas_auth, file_id, offset, offset + length - 1);
        errdefer parsed.deinit();

        return .{
            .allocator = allocator,
            .client = client,
            .parsed = parsed,
            .remaining = length,
            .skip_head = parsed.value.offset_into_first_range,
            .interface = .{
                .vtable = &.{ .stream = stream },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
        };
    }

    pub fn deinit(self: *FileRangeReader) void {
        if (self.http_req) |*req| req.deinit();
        self.parsed.deinit();
    }

    /// Tear down any in-flight HTTP request. Safe to call repeatedly.
    fn closeXorb(self: *FileRangeReader) void {
        if (self.http_req) |*req| {
            req.deinit();
            self.http_req = null;
            self.http_reader = null;
        }
    }

    /// Advance to the next term: tear down the previous xorb request, issue
    /// a fresh range GET for the new term's xorb window, then read+discard
    /// chunks until the term's first chunk is at the head of the stream.
    /// Returns `false` when no terms remain.
    fn openNextTerm(self: *FileRangeReader) !bool {
        self.closeXorb();

        const terms = self.parsed.value.terms;
        if (self.term_idx >= terms.len) return false;
        const term = terms[self.term_idx];
        self.term_idx += 1;

        const fetch_urls = self.parsed.value.fetch_info.map.get(term.hash) orelse return error.NoFetchUrlsForXorb;
        const fu = for (fetch_urls) |*u| {
            if (term.range.start >= u.range.start and term.range.end <= u.range.end) break u;
        } else return error.ChunkOutsideFetchUrls;

        try self.openXorbBody(fu);
        errdefer self.closeXorb();

        var i: u64 = fu.range.start;
        while (i < term.range.start) : (i += 1) try self.discardOneChunk();
        self.chunks_remaining_in_term = term.range.end - term.range.start;
        return true;
    }

    /// Issue the xorb range GET with bounded retry on transient errors
    /// (HTTP 5xx, 429, and connection-level failures from request /
    /// sendBodiless / receiveHead). 4xx (other than 429) fails immediately.
    fn openXorbBody(self: *FileRangeReader, fu: *const FetchUrl) !void {
        const max_attempts: u32 = 4;
        var backoff_ms: u64 = 250;
        var attempt: u32 = 0;
        while (true) : (attempt += 1) {
            self.tryOpenXorbBody(fu) catch |err| {
                self.closeXorb();
                if (err == error.XorbFetchPermanent or attempt + 1 >= max_attempts) {
                    log.err("xorb fetch failed after {d} attempt(s): {s} (url={s})", .{ attempt + 1, @errorName(err), fu.url });
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

    fn tryOpenXorbBody(self: *FileRangeReader, fu: *const FetchUrl) !void {
        const uri: std.Uri = try .parse(fu.url);
        var range_buf: [64]u8 = undefined;
        const range = try std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ fu.url_range.start, fu.url_range.end });

        self.http_req = try self.client.request(.GET, uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Range", .value = range }},
        });
        try self.http_req.?.sendBodiless();
        var res = try self.http_req.?.receiveHead(&self.xorb_redirect_buf);
        if (res.head.status != .ok and res.head.status != .partial_content) {
            const code = @intFromEnum(res.head.status);
            if (code == 429 or (code >= 500 and code < 600)) {
                log.warn("xorb HTTP {d} (transient)", .{code});
                return error.XorbFetchTransient;
            }
            log.err("xorb HTTP {d} (permanent) from {s}", .{ code, fu.url });
            return error.XorbFetchPermanent;
        }
        self.http_reader = res.reader(&self.xorb_transfer_buf);
    }

    /// Read one chunk header off the live HTTP body and discard `csize`
    /// payload bytes. Used to skip chunks before `term.range.start`.
    fn discardOneChunk(self: *FileRangeReader) !void {
        var hdr: [chunk_header_size]u8 = undefined;
        self.http_reader.?.readSliceAll(&hdr) catch return error.UnexpectedXorbEnd;
        if (hdr[0] != 0) return error.InvalidVersion;
        const csize: u32 = @as(u32, hdr[1]) | (@as(u32, hdr[2]) << 8) | (@as(u32, hdr[3]) << 16);
        self.http_reader.?.discardAll(csize) catch return error.UnexpectedXorbEnd;
    }

    /// Read one full chunk (header + compressed payload) off the live HTTP
    /// body into `xorb_chunk_comp_buf`.
    fn readOneChunk(self: *FileRangeReader) !Chunk {
        var hdr: [chunk_header_size]u8 = undefined;
        self.http_reader.?.readSliceAll(&hdr) catch return error.UnexpectedXorbEnd;
        if (hdr[0] != 0) return error.InvalidVersion;
        const csize: u32 = @as(u32, hdr[1]) | (@as(u32, hdr[2]) << 8) | (@as(u32, hdr[3]) << 16);
        const ctype = hdr[4];
        if (ctype > 2) return error.InvalidCompressionType;
        const usize_uncomp: u32 = @as(u32, hdr[5]) | (@as(u32, hdr[6]) << 8) | (@as(u32, hdr[7]) << 16);
        if (csize > self.xorb_chunk_comp_buf.len) return error.ChunkTooLarge;
        self.http_reader.?.readSliceAll(self.xorb_chunk_comp_buf[0..csize]) catch return error.UnexpectedXorbEnd;
        return .{
            .compressed_size = csize,
            .uncompressed_size = usize_uncomp,
            .compression_type = ctype,
            .compressed_data = self.xorb_chunk_comp_buf[0..csize],
        };
    }

    /// Pull the next chunk that contributes to the output (advancing
    /// terms as needed). Returns `null` at EOF.
    fn nextOutputChunk(self: *FileRangeReader) !?Chunk {
        while (true) {
            if (self.http_reader != null and self.chunks_remaining_in_term > 0) {
                self.chunks_remaining_in_term -= 1;
                return try self.readOneChunk();
            }
            if (!try self.openNextTerm()) return null;
        }
    }

    fn stream(reader: *std.Io.Reader, writer: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *FileRangeReader = @alignCast(@fieldParentPtr("interface", reader));
        if (self.remaining == 0) return error.EndOfStream;

        // 1. Drain any leftover bytes from `chunk_buf` from a previous call
        //    (head-trimmed first chunk or partial last chunk).
        if (self.chunk_pos < self.chunk_end) {
            const avail = self.chunk_buf[self.chunk_pos..self.chunk_end];
            const cap: usize = @intCast(@min(@as(u64, avail.len), @min(self.remaining, @intFromEnum(limit))));
            const n = writer.write(avail[0..cap]) catch return error.WriteFailed;
            if (n == 0) return error.WriteFailed;
            self.chunk_pos += n;
            self.remaining -= n;
            return n;
        }

        // 2. Pull the next chunk that contributes to the output.
        const chunk = (self.nextOutputChunk() catch return error.ReadFailed) orelse return error.EndOfStream;

        // 3. Zero-copy fast path: no head trim, full chunk fits within the
        //    remaining byte budget and the writer's contiguous writable
        //    slot. Decompress straight into the writer's buffer.
        const uncomp: usize = chunk.uncompressed_size;
        if (self.skip_head == 0 and self.remaining >= uncomp and @intFromEnum(limit) >= uncomp) {
            const slot = writer.writableSliceGreedy(uncomp) catch return error.WriteFailed;
            if (slot.len >= uncomp) {
                _ = decompressChunk(chunk, slot) catch return error.ReadFailed;
                writer.advance(uncomp);
                self.remaining -= uncomp;
                return uncomp;
            }
            // Writer can't take a contiguous slot big enough — fall through.
        }

        // 4. Fallback: decompress into the scratch `chunk_buf`, apply head
        //    trim, then emit as much as fits in this call. Remainder stays
        //    in `chunk_buf` for the next stream() call.
        const decoded = decompressChunk(chunk, &self.chunk_buf) catch return error.ReadFailed;
        var start: usize = 0;
        if (self.skip_head > 0) {
            const drop: usize = @intCast(@min(self.skip_head, @as(u64, decoded.len)));
            self.skip_head -= drop;
            start = drop;
        }
        self.chunk_pos = start;
        self.chunk_end = decoded.len;

        const avail = self.chunk_buf[self.chunk_pos..self.chunk_end];
        if (avail.len == 0) return 0; // entire chunk was head-trim — caller loops.
        const cap: usize = @intCast(@min(@as(u64, avail.len), @min(self.remaining, @intFromEnum(limit))));
        const n = writer.write(avail[0..cap]) catch return error.WriteFailed;
        if (n == 0) return error.WriteFailed;
        self.chunk_pos += n;
        self.remaining -= n;
        return n;
    }
};
