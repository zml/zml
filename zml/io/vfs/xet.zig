//! Dumb, stateless XET reader for the HF VFS, exposed as `std.Io.Reader`.
//!
//! `xet.FileRangeReader.init(...)` issues the CAS token + per-range reconstruction
//! requests, then `interface.stream(writer, limit)` emits the file bytes
//! incrementally, one xorb chunk at a time. Per-chunk LZ4 / BG4+LZ4
//! decompression goes directly into the destination writer's writable
//! slice (`writableSliceGreedy` + `advance`) — same pattern as
//! `lz4.BlockReader`. An internal `chunk_buf` is only used for the very
//! first chunk (when `offset_into_first_range` requires trimming the head)
//! and for the very last chunk (when the writer doesn't have room for the
//! full decoded size).
//!
//! No caches, no plans, no dedup.

const std = @import("std");
const lz4 = @import("lz4.zig");
const bg4 = @import("bg4.zig");

const log = std.log.scoped(.@"zml/io/vfs/xet");

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

const ChunkIterator = struct {
    data: []const u8,
    pos: usize = 0,

    const header_size = 8;

    const Chunk = struct {
        compressed_size: u32,
        uncompressed_size: u32,
        compression_type: u8,
        compressed_data: []const u8,
    };

    fn next(self: *ChunkIterator) !?Chunk {
        if (self.pos == self.data.len) return null;
        if (self.data.len - self.pos < header_size) return error.TruncatedHeader;
        const h = self.data[self.pos..][0..header_size];
        if (h[0] != 0) return error.InvalidVersion;
        const csize: u32 = @as(u32, h[1]) | (@as(u32, h[2]) << 8) | (@as(u32, h[3]) << 16);
        const ctype = h[4];
        if (ctype > 2) return error.InvalidCompressionType;
        const usize_uncomp: u32 = @as(u32, h[5]) | (@as(u32, h[6]) << 8) | (@as(u32, h[7]) << 16);
        const data_start = self.pos + header_size;
        const data_end = data_start + csize;
        if (data_end > self.data.len) return error.TruncatedData;
        self.pos = data_end;
        return .{
            .compressed_size = csize,
            .uncompressed_size = usize_uncomp,
            .compression_type = ctype,
            .compressed_data = self.data[data_start..data_end],
        };
    }
};

/// Decompresses `chunk` into `dst[0..chunk.uncompressed_size]`. `dst` must
/// be at least that large. Composes via `lz4.BlockReader` / `bg4.DegroupWriter`
/// over `std.Io.Reader` / `std.Io.Writer`.
fn decompressChunk(chunk: ChunkIterator.Chunk, dst: []u8) ![]u8 {
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

fn httpRangeGet(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    url: []const u8,
    start: u64,
    end_inclusive: u64,
) ![]u8 {
    const uri: std.Uri = try .parse(url);
    var range_buf: [64]u8 = undefined;
    const range = try std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ start, end_inclusive });
    var req = try client.request(.GET, uri, .{
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
        .extra_headers = &.{.{ .name = "Range", .value = range }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok and res.head.status != .partial_content) return error.XorbFetchFailed;
    const expected: usize = @intCast(end_inclusive - start + 1);
    return try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse expected);
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
    /// `term.range.end` cached for the active term (0 when no term active).
    term_range_end: u64 = 0,

    // Current xorb materialized in memory (allocated; freed on advance or
    // deinit). `chunk_it` walks it; `next_chunk_index` is the wire-index of
    // the chunk `chunk_it.next()` will return next.
    xorb_bytes: ?[]u8 = null,
    chunk_it: ChunkIterator = .{ .data = &.{} },
    next_chunk_index: u64 = 0,

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
        if (self.xorb_bytes) |b| self.allocator.free(b);
        self.parsed.deinit();
    }

    /// Advance to the next term (freeing the current xorb), download its
    /// xorb byte window, reset the chunk iterator, and skip any chunks
    /// before `term.range.start`. Returns `false` when no terms remain.
    fn openNextTerm(self: *FileRangeReader) !bool {
        if (self.xorb_bytes) |b| {
            self.allocator.free(b);
            self.xorb_bytes = null;
        }
        const terms = self.parsed.value.terms;
        if (self.term_idx >= terms.len) return false;
        const term = terms[self.term_idx];
        self.term_idx += 1;

        const fetch_urls = self.parsed.value.fetch_info.map.get(term.hash) orelse return error.NoFetchUrlsForXorb;
        const fu = for (fetch_urls) |*u| {
            if (term.range.start >= u.range.start and term.range.end <= u.range.end) break u;
        } else return error.ChunkOutsideFetchUrls;

        self.xorb_bytes = try httpRangeGet(self.allocator, self.client, fu.url, fu.url_range.start, fu.url_range.end);
        self.chunk_it = .{ .data = self.xorb_bytes.? };
        self.next_chunk_index = fu.range.start;
        self.term_range_end = term.range.end;
        while (self.next_chunk_index < term.range.start) : (self.next_chunk_index += 1) {
            _ = (try self.chunk_it.next()) orelse return error.UnexpectedXorbEnd;
        }
        return true;
    }

    /// Pull the next chunk that contributes to the output (advancing
    /// terms as needed). Returns `null` at EOF.
    fn nextOutputChunk(self: *FileRangeReader) !?ChunkIterator.Chunk {
        while (true) {
            if (self.xorb_bytes != null and self.next_chunk_index < self.term_range_end) {
                const chunk = (try self.chunk_it.next()) orelse return error.UnexpectedXorbEnd;
                self.next_chunk_index += 1;
                return chunk;
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
