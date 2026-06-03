const std = @import("std");
const lz4 = @import("lz4.zig");

const log = std.log.scoped(.@"zml/io/xet");

/// JSON-compatible term from the CAS reconstruction response.
pub const Term = struct {
    hash: []const u8,
    unpacked_length: u64,
    range: struct { start: u64, end: u64 },
};

/// JSON-compatible fetch_info entry: where to download a (sub)range of a xorb.
pub const FetchUrl = struct {
    range: struct { start: u64, end: u64 },
    url: []const u8,
    url_range: struct { start: u64, end: u64 },
};

/// JSON-compatible top-level reconstruction response for one group.
pub const ReconstructionResponse = struct {
    offset_into_first_range: u64,
    terms: []const Term,
    /// Map xorb_hash → list of presigned URL entries. Default empty so older
    /// fixtures and tests without fetch_info still parse.
    fetch_info: std.json.ArrayHashMap([]const FetchUrl) = .{},
};

// ── Chunk Iterator ──────────────────────────────────────────────────────────

pub const ChunkIterator = struct {
    data: []const u8,
    pos: usize = 0,
    chunk_index: u32 = 0,

    pub const header_size = 8;

    pub const Chunk = struct {
        index: u32,
        compressed_size: u32,
        uncompressed_size: u32,
        compression_type: u8,
        compressed_data: []const u8,
    };

    /// Returns the next chunk, or null at EOF.
    /// Returns error on truncation or invalid version.
    pub fn next(self: *ChunkIterator) !?Chunk {
        if (self.pos == self.data.len) return null;
        if (self.data.len - self.pos < header_size) return error.TruncatedHeader;

        const h = self.data[self.pos..][0..header_size];
        if (h[0] != 0) return error.InvalidVersion;

        const compressed_size: u32 = @as(u32, h[1]) | (@as(u32, h[2]) << 8) | (@as(u32, h[3]) << 16);
        const compression_type = h[4];
        if (compression_type > 3) return error.InvalidCompressionType;
        const uncompressed_size: u32 = @as(u32, h[5]) | (@as(u32, h[6]) << 8) | (@as(u32, h[7]) << 16);

        const data_start = self.pos + header_size;
        const data_end = data_start + compressed_size;
        if (data_end > self.data.len) return error.TruncatedData;

        const chunk = Chunk{
            .index = self.chunk_index,
            .compressed_size = compressed_size,
            .uncompressed_size = uncompressed_size,
            .compression_type = compression_type,
            .compressed_data = self.data[data_start..data_end],
        };
        self.pos = data_end;
        self.chunk_index += 1;
        return chunk;
    }
};

// ── Chunk Decompression ─────────────────────────────────────────────────────

/// Decompresses a single chunk into caller-provided buffers.
/// `dst` must be at least `chunk.uncompressed_size` bytes.
/// `tmp` must be at least `chunk.uncompressed_size` bytes (used as intermediate
/// buffer for types 2 and 3; ignored for types 0 and 1).
/// Returns `dst[0..chunk.uncompressed_size]`.
pub fn decompressChunk(chunk: ChunkIterator.Chunk, dst: []u8, tmp: []u8) ![]u8 {
    const usize_uncomp: usize = chunk.uncompressed_size;
    if (dst.len < usize_uncomp) return error.OutputTooSmall;

    switch (chunk.compression_type) {
        0 => {
            // None: raw data, just copy.
            if (chunk.compressed_data.len != usize_uncomp) return error.SizeMismatch;
            @memcpy(dst[0..usize_uncomp], chunk.compressed_data);
            return dst[0..usize_uncomp];
        },
        1 => {
            // LZ4 block/frame.
            const n = lz4.decompress(chunk.compressed_data, dst) catch return error.CorruptedData;
            if (n != usize_uncomp) return error.SizeMismatch;
            return dst[0..n];
        },
        2 => {
            // ByteGrouping4 + LZ4.
            if (tmp.len < usize_uncomp) return error.OutputTooSmall;
            const n = lz4.decompressBg4(chunk.compressed_data, dst, tmp) catch return error.CorruptedData;
            if (n != usize_uncomp) return error.SizeMismatch;
            return dst[0..n];
        },
        3 => {
            // FullBitslice + LZ4.
            if (tmp.len < usize_uncomp) return error.OutputTooSmall;
            const n = lz4.decompressFbs(chunk.compressed_data, dst, tmp) catch return error.CorruptedData;
            if (n != usize_uncomp) return error.SizeMismatch;
            return dst[0..n];
        },
        else => return error.InvalidCompressionType,
    }
}

// ── HF CAS Client ───────────────────────────────────────────────────────────

/// High-level Xet/HF CAS client. Holds the user's HF token and caches the
/// per-file Xet id and per-(repo, rev) CAS access token so repeated
/// `reconstruct` calls against the same model don't re-issue the handshake
/// round-trips. Not thread-safe: serialize calls externally.
pub const Client = struct {
    pub const Repo = struct {
        repo: []const u8,
        model: []const u8,
        rev: []const u8,
        path: []const u8,
    };

    pub const CasAuth = struct {
        url: []const u8,
        token: []const u8,
    };

    allocator: std.mem.Allocator,
    http: *std.http.Client,
    /// User's HF token, raw (no "Bearer " prefix). Whitespace is trimmed on use.
    hf_token: []const u8,

    file_id_cache: std.StringHashMapUnmanaged([]const u8) = .{},
    cas_cache: std.StringHashMapUnmanaged(CasAuth) = .{},

    pub fn init(allocator: std.mem.Allocator, http: *std.http.Client, hf_token: []const u8) Client {
        return .{ .allocator = allocator, .http = http, .hf_token = hf_token };
    }

    pub fn deinit(self: *Client) void {
        var fit = self.file_id_cache.iterator();
        while (fit.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.*);
        }
        self.file_id_cache.deinit(self.allocator);
        var cit = self.cas_cache.iterator();
        while (cit.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.url);
            self.allocator.free(e.value_ptr.token);
        }
        self.cas_cache.deinit(self.allocator);
    }

    /// Returns the cached Xet file id for `repo` (or fetches+caches on miss).
    /// The returned slice is owned by the client and valid until `deinit`.
    pub fn fileId(self: *Client, repo: Repo) ![]const u8 {
        var key_buf: [4096]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}/{s}@{s}/{s}", .{ repo.repo, repo.model, repo.rev, repo.path });
        if (self.file_id_cache.get(key)) |fid| return fid;

        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);
        const fid = try fetchFileId(self.allocator, self.http, repo, self.hf_token);
        errdefer self.allocator.free(fid);
        try self.file_id_cache.put(self.allocator, owned_key, fid);
        return fid;
    }

    /// Returns the cached CAS endpoint + access token for the (repo, rev)
    /// pair (or fetches+caches on miss). Slices are owned by the client.
    pub fn casAuth(self: *Client, repo: Repo) !CasAuth {
        var key_buf: [4096]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}/{s}@{s}", .{ repo.repo, repo.model, repo.rev });
        if (self.cas_cache.get(key)) |c| return c;

        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);
        const c = try fetchCasToken(self.allocator, self.http, repo, self.hf_token);
        errdefer {
            self.allocator.free(c.url);
            self.allocator.free(c.token);
        }
        try self.cas_cache.put(self.allocator, owned_key, c);
        return c;
    }

    /// Fetch + parse the CAS reconstruction plan for `repo`'s file range
    /// `[range_start, range_end_exclusive)`. Caller owns the returned
    /// `Parsed` and must call `.deinit()` when done.
    pub fn reconstruct(
        self: *Client,
        repo: Repo,
        range_start: u64,
        range_end_exclusive: u64,
    ) !std.json.Parsed(ReconstructionResponse) {
        const file_id = try self.fileId(repo);
        const cas = try self.casAuth(repo);

        var cas_auth_buf: [65536]u8 = undefined;
        const cas_auth = std.fmt.bufPrint(&cas_auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

        const body = try fetchReconstruction(self.allocator, self.http, cas.url, cas_auth, file_id, range_start, range_end_exclusive);
        defer self.allocator.free(body);

        // .alloc_always: copy string fields into the Parsed arena so they
        // outlive `body` (default for Scanner-backed parseFromSlice is
        // .alloc_if_needed, which would leave dangling pointers into `body`).
        return try std.json.parseFromSlice(ReconstructionResponse, self.allocator, body, .{
            .ignore_unknown_fields = true,
            .allocate = .alloc_always,
        });
    }
};

fn bearerAuth(hf_token: []const u8, buf: []u8) ![]u8 {
    return std.fmt.bufPrint(buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch error.TokenTooLong;
}

fn fetchFileId(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: Client.Repo,
    hf_token: []const u8,
) ![]const u8 {
    var auth_buf: [1024]u8 = undefined;
    const auth = try bearerAuth(hf_token, &auth_buf);
    var url_buf: [4096]u8 = undefined;
    const resolve_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
        .{ repo.repo, repo.model, repo.rev, repo.path },
    );
    const uri: std.Uri = try .parse(resolve_url);
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
    var header_it = res.head.iterateHeaders();
    while (header_it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "X-Xet-Hash")) {
            return try allocator.dupe(u8, header.value);
        }
    }
    log.err("X-Xet-Hash not found (status={}).", .{res.head.status});
    return error.XetHashNotFound;
}

fn fetchCasToken(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: Client.Repo,
    hf_token: []const u8,
) !Client.CasAuth {
    var auth_buf: [1024]u8 = undefined;
    const auth = try bearerAuth(hf_token, &auth_buf);
    var url_buf: [4096]u8 = undefined;
    const token_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/api/models/{s}/{s}/xet-read-token/{s}",
        .{ repo.repo, repo.model, repo.rev },
    );
    const uri: std.Uri = try .parse(token_url);
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
    if (res.head.status != .ok) {
        log.err("CAS token request failed: status={}", .{res.head.status});
        return error.CasTokenFailed;
    }
    const body = try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 128 * 1024);
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(struct {
        accessToken: []const u8,
        casUrl: []const u8,
    }, allocator, body, .{ .ignore_unknown_fields = true });
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
    range_end_exclusive: u64,
) ![]const u8 {
    var url_buf: [4096]u8 = undefined;
    const recon_url = try std.fmt.bufPrint(&url_buf, "{s}/v1/reconstructions/{s}", .{ cas_url, file_id });
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ range_start, range_end_exclusive - 1 }) catch unreachable;
    const uri: std.Uri = try .parse(recon_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = cas_auth },
        },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok and res.head.status != .partial_content) {
        log.err("Reconstruction failed: status={}", .{res.head.status});
        return error.ReconstructionFailed;
    }
    return try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 64 * 1024 * 1024);
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Helper: write a chunk header into a buffer.
fn writeChunkHeader(
    buf: []u8,
    compressed_size: u24,
    compression_type: u8,
    uncompressed_size: u24,
) void {
    buf[0] = 0; // version
    buf[1] = @truncate(compressed_size);
    buf[2] = @truncate(compressed_size >> 8);
    buf[3] = @truncate(compressed_size >> 16);
    buf[4] = compression_type;
    buf[5] = @truncate(uncompressed_size);
    buf[6] = @truncate(uncompressed_size >> 8);
    buf[7] = @truncate(uncompressed_size >> 16);
}

test "ChunkIterator: synthetic 3-chunk stream" {
    // Chunk 0: compression=0(None), 4 bytes payload, uncompressed=4
    // Chunk 1: compression=1(LZ4), 6 bytes payload, uncompressed=100
    // Chunk 2: compression=2(BG4LZ4), 2 bytes payload, uncompressed=50
    const payloads = [_][]const u8{ &.{ 0xAA, 0xBB, 0xCC, 0xDD }, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 0xFF, 0x01 } };
    const comp_types = [_]u8{ 0, 1, 2 };
    const uncomp_sizes = [_]u24{ 4, 100, 50 };

    var buf: [3 * 8 + 4 + 6 + 2]u8 = undefined;
    var off: usize = 0;
    for (0..3) |i| {
        writeChunkHeader(buf[off..][0..8], @intCast(payloads[i].len), comp_types[i], uncomp_sizes[i]);
        off += 8;
        @memcpy(buf[off..][0..payloads[i].len], payloads[i]);
        off += payloads[i].len;
    }
    try std.testing.expectEqual(buf.len, off);

    var it = ChunkIterator{ .data = &buf };
    var count: usize = 0;
    var total_wire: usize = 0;
    while (try it.next()) |chunk| {
        try std.testing.expectEqual(@as(u32, @intCast(count)), chunk.index);
        try std.testing.expectEqual(@as(u32, @intCast(payloads[count].len)), chunk.compressed_size);
        try std.testing.expectEqual(@as(u32, uncomp_sizes[count]), chunk.uncompressed_size);
        try std.testing.expectEqual(comp_types[count], chunk.compression_type);
        try std.testing.expectEqualSlices(u8, payloads[count], chunk.compressed_data);
        total_wire += 8 + chunk.compressed_size;
        count += 1;
    }

    // Correct number of chunks
    try std.testing.expectEqual(@as(usize, 3), count);
    // Wire size matches input
    try std.testing.expectEqual(buf.len, total_wire);
    // next() returns null again
    try std.testing.expectEqual(@as(?ChunkIterator.Chunk, null), try it.next());
}

test "ChunkIterator: truncated header" {
    const buf = [_]u8{ 0, 0, 0, 0 }; // only 4 bytes, need 8
    var it = ChunkIterator{ .data = &buf };
    try std.testing.expectError(error.TruncatedHeader, it.next());
}

test "ChunkIterator: invalid version" {
    var buf: [8]u8 = undefined;
    writeChunkHeader(&buf, 0, 0, 0);
    buf[0] = 1; // bad version
    var it = ChunkIterator{ .data = &buf };
    try std.testing.expectError(error.InvalidVersion, it.next());
}

test "ChunkIterator: truncated data" {
    var buf: [10]u8 = undefined; // header says 5 bytes payload but only 2 available
    writeChunkHeader(buf[0..8], 5, 0, 5);
    buf[8] = 0;
    buf[9] = 0;
    var it = ChunkIterator{ .data = &buf };
    try std.testing.expectError(error.TruncatedData, it.next());
}

test "ChunkIterator: empty input" {
    var it = ChunkIterator{ .data = &.{} };
    try std.testing.expectEqual(@as(?ChunkIterator.Chunk, null), try it.next());
}

test "decompressChunk: type 0 (None)" {
    const data = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF };
    const chunk = ChunkIterator.Chunk{
        .index = 0,
        .compressed_size = 4,
        .uncompressed_size = 4,
        .compression_type = 0,
        .compressed_data = &data,
    };
    var dst: [4]u8 = undefined;
    var tmp: [4]u8 = undefined;
    const result = try decompressChunk(chunk, &dst, &tmp);
    try std.testing.expectEqualSlices(u8, &data, result);
}

test "decompressChunk: type 0 size mismatch" {
    const data = [_]u8{ 0xDE, 0xAD };
    const chunk = ChunkIterator.Chunk{
        .index = 0,
        .compressed_size = 2,
        .uncompressed_size = 5, // mismatch
        .compression_type = 0,
        .compressed_data = &data,
    };
    var dst: [5]u8 = undefined;
    var tmp: [5]u8 = undefined;
    try std.testing.expectError(error.SizeMismatch, decompressChunk(chunk, &dst, &tmp));
}

test "decompressChunk: type 1 (LZ4)" {
    // Hand-encoded LZ4 block: 5 literals "Hello"
    // token = (5 << 4) = 0x50, then 5 literal bytes
    const compressed = [_]u8{ 0x50, 'H', 'e', 'l', 'l', 'o' };
    const chunk = ChunkIterator.Chunk{
        .index = 0,
        .compressed_size = compressed.len,
        .uncompressed_size = 5,
        .compression_type = 1,
        .compressed_data = &compressed,
    };
    var dst: [5]u8 = undefined;
    var tmp: [5]u8 = undefined;
    const result = try decompressChunk(chunk, &dst, &tmp);
    try std.testing.expectEqualStrings("Hello", result);
}

test "decompressChunk: type 2 (BG4+LZ4)" {
    // Original: [1,2,3,4,5,6,7,8]
    // BG4 grouped (round-robin into 4 groups of 2):
    //   Group 0: pos 0,4 → 1,5
    //   Group 1: pos 1,5 → 2,6
    //   Group 2: pos 2,6 → 3,7
    //   Group 3: pos 3,7 → 4,8
    //   Grouped: [1,5,2,6,3,7,4,8]
    // LZ4 all-literals block: token = (8 << 4) = 0x80, then 8 bytes
    const compressed = [_]u8{ 0x80, 1, 5, 2, 6, 3, 7, 4, 8 };
    const chunk = ChunkIterator.Chunk{
        .index = 0,
        .compressed_size = compressed.len,
        .uncompressed_size = 8,
        .compression_type = 2,
        .compressed_data = &compressed,
    };
    var dst: [8]u8 = undefined;
    var tmp: [8]u8 = undefined;
    const result = try decompressChunk(chunk, &dst, &tmp);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 }, result);
}

test "decompressChunk: type 3 (FBS+LZ4)" {
    // Take a known 4-byte original, apply forward bitslice, encode as LZ4 literals.
    const original = [_]u8{ 0xA5, 0x3C, 0xF0, 0x0F };

    // Compute forward bitslice (N=4):
    var bitsliced: [4]u8 = .{ 0, 0, 0, 0 };
    for (0..4) |orig_byte_idx| {
        for (0..8) |orig_bit_idx| {
            const k = orig_bit_idx * 4 + orig_byte_idx;
            const out_byte_idx = k / 8;
            const out_bit_idx = k % 8;
            if (out_byte_idx >= 4) continue;
            const bit: u8 = (original[orig_byte_idx] >> @intCast(orig_bit_idx)) & 1;
            bitsliced[out_byte_idx] |= bit << @intCast(out_bit_idx);
        }
    }

    // LZ4 all-literals: token = (4 << 4) = 0x40, then 4 bytes
    const compressed = [_]u8{0x40} ++ bitsliced;
    const chunk = ChunkIterator.Chunk{
        .index = 0,
        .compressed_size = compressed.len,
        .uncompressed_size = 4,
        .compression_type = 3,
        .compressed_data = &compressed,
    };
    var dst: [4]u8 = undefined;
    var tmp: [4]u8 = undefined;
    const result = try decompressChunk(chunk, &dst, &tmp);
    try std.testing.expectEqualSlices(u8, &original, result);
}

test "parse fetch_info: synthetic" {
    const json_str =
        \\{
        \\  "offset_into_first_range": 0,
        \\  "terms": [],
        \\  "fetch_info": {
        \\    "aaa": [
        \\      {"range": {"start": 0, "end": 10},
        \\       "url": "https://example.com/aaa",
        \\       "url_range": {"start": 0, "end": 1024}}
        \\    ],
        \\    "bbb": [
        \\      {"range": {"start": 0, "end": 5},
        \\       "url": "https://example.com/bbb/0",
        \\       "url_range": {"start": 0, "end": 512}},
        \\      {"range": {"start": 5, "end": 12},
        \\       "url": "https://example.com/bbb/1",
        \\       "url_range": {"start": 512, "end": 2048}}
        \\    ]
        \\  }
        \\}
    ;
    const parsed = try std.json.parseFromSlice(ReconstructionResponse, std.testing.allocator, json_str, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();

    const fi = parsed.value.fetch_info.map;
    try std.testing.expectEqual(@as(usize, 2), fi.count());

    const aaa = fi.get("aaa").?;
    try std.testing.expectEqual(@as(usize, 1), aaa.len);
    try std.testing.expectEqualStrings("https://example.com/aaa", aaa[0].url);
    try std.testing.expectEqual(@as(u64, 1024), aaa[0].url_range.end);

    const bbb = fi.get("bbb").?;
    try std.testing.expectEqual(@as(usize, 2), bbb.len);
    try std.testing.expectEqual(@as(u64, 512), bbb[1].url_range.start);
    try std.testing.expectEqual(@as(u64, 5), bbb[1].range.start);
}
