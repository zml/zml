const std = @import("std");
const lz4 = @import("lz4.zig");

const log = std.log.scoped(.@"zml/io/xet");

pub const TermRange = struct {
    file_start: u64,
    file_end: u64,
};

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

/// Computes the [file_start, file_end) byte range for each term.
///
/// `out` must have the same length as `response.terms`.
/// `group_range_start` is the absolute file byte offset where this group's
/// requested range begins (from the manifest / HTTP Range request).
pub fn computeTermRanges(
    response: ReconstructionResponse,
    group_range_start: u64,
    out: []TermRange,
) void {
    std.debug.assert(out.len == response.terms.len);
    var pos: u64 = group_range_start - response.offset_into_first_range;
    for (response.terms, out) |term, *range| {
        range.file_start = pos;
        range.file_end = pos + term.unpacked_length;
        pos = range.file_end;
    }
}

// ── Xorb Map ────────────────────────────────────────────────────────────────

pub const XorbMap = struct {
    /// One entry per unique xorb hash.
    entries: []const Entry,

    pub const Entry = struct {
        hash: []const u8,
        /// Indexed by chunk_index. Each element is a slice of term indices
        /// that reference this chunk. Empty slice = chunk not needed (skip).
        /// Length = max(term.range.end) for all terms referencing this xorb.
        chunk_to_terms: []const []const u32,
    };
};

/// Builds a map from xorb hash to chunk→term-indices lookup table.
///
/// For each unique xorb, allocates a flat array of size max(range.end) where
/// `chunk_to_terms[i]` gives the list of term indices needing chunk `i`.
/// All backing memory is allocated from `allocator` (use an arena).
pub fn buildXorbMap(allocator: std.mem.Allocator, terms: []const Term) !XorbMap {
    if (terms.len == 0) return .{ .entries = &.{} };

    // Sort term indices by hash to group terms by xorb.
    const sorted_idx = try allocator.alloc(u32, terms.len);
    defer allocator.free(sorted_idx);
    for (sorted_idx, 0..) |*s, i| s.* = @intCast(i);
    const Ctx = struct {
        terms: []const Term,
        fn lessThan(ctx: @This(), a: u32, b: u32) bool {
            return std.mem.order(u8, ctx.terms[a].hash, ctx.terms[b].hash) == .lt;
        }
    };
    std.mem.sort(u32, sorted_idx, Ctx{ .terms = terms }, Ctx.lessThan);

    // Pass 1: compute sizes for each xorb's chunk map and total refs.
    var num_xorbs: usize = 0;
    var total_chunk_slots: usize = 0;
    var total_refs: usize = 0;
    var max_max_end: usize = 0;
    {
        var i: usize = 0;
        while (i < sorted_idx.len) {
            const hash = terms[sorted_idx[i]].hash;
            var max_end: u64 = 0;
            while (i < sorted_idx.len and std.mem.eql(u8, terms[sorted_idx[i]].hash, hash)) : (i += 1) {
                const t = terms[sorted_idx[i]];
                max_end = @max(max_end, t.range.end);
                total_refs += @intCast(t.range.end - t.range.start);
            }
            num_xorbs += 1;
            total_chunk_slots += @intCast(max_end);
            max_max_end = @max(max_max_end, @as(usize, @intCast(max_end)));
        }
    }

    // Allocate output buffers.
    const entries = try allocator.alloc(XorbMap.Entry, num_xorbs);
    const chunk_map_backing = try allocator.alloc([]const u32, total_chunk_slots);
    const term_idx_backing = try allocator.alloc(u32, total_refs);

    // Temp: per-chunk counts, reused across xorbs.
    const counts = try allocator.alloc(u32, max_max_end);
    defer allocator.free(counts);

    const empty: []const u32 = &.{};

    // Pass 2: fill entries.
    var chunk_offset: usize = 0;
    var ref_offset: usize = 0;
    var entry_idx: usize = 0;
    {
        var i: usize = 0;
        while (i < sorted_idx.len) {
            const hash = terms[sorted_idx[i]].hash;
            const group_start = i;
            var max_end: u64 = 0;
            while (i < sorted_idx.len and std.mem.eql(u8, terms[sorted_idx[i]].hash, hash)) : (i += 1) {
                max_end = @max(max_end, terms[sorted_idx[i]].range.end);
            }
            const me: usize = @intCast(max_end);

            // Count per-chunk references.
            @memset(counts[0..me], 0);
            for (sorted_idx[group_start..i]) |ti| {
                const t = terms[ti];
                for (@intCast(t.range.start)..@intCast(t.range.end)) |c| {
                    counts[c] += 1;
                }
            }

            // Assign chunk_to_terms slices from backing buffer.
            const chunk_map = chunk_map_backing[chunk_offset..][0..me];
            chunk_offset += me;
            for (chunk_map, counts[0..me]) |*slot, cnt| {
                if (cnt > 0) {
                    slot.* = term_idx_backing[ref_offset..][0..cnt];
                    ref_offset += cnt;
                } else {
                    slot.* = empty;
                }
            }

            // Fill term indices (write through @constCast — safe, we own the backing).
            @memset(counts[0..me], 0);
            for (sorted_idx[group_start..i]) |ti| {
                const t = terms[ti];
                for (@intCast(t.range.start)..@intCast(t.range.end)) |c| {
                    @constCast(chunk_map[c])[counts[c]] = ti;
                    counts[c] += 1;
                }
            }

            entries[entry_idx] = .{
                .hash = hash,
                .chunk_to_terms = chunk_map,
            };
            entry_idx += 1;
        }
    }

    return .{ .entries = entries };
}

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

// ── Xorb Processing ─────────────────────────────────────────────────────────

/// Iterates all chunks in a xorb, decompresses those referenced by `xorb_entry`,
/// and calls `onChunk(context, term_idx, decompressed_data)` for each referencing term.
///
/// `dst` and `tmp` are caller-provided scratch buffers (>= 128 KiB each).
/// Zero allocations on the hot path.
pub fn processXorb(
    xorb_data: []const u8,
    xorb_entry: XorbMap.Entry,
    start_chunk_index: u32,
    dst: []u8,
    tmp: []u8,
    context: anytype,
    comptime onChunk: fn (@TypeOf(context), term_idx: u32, data: []const u8) anyerror!void,
) !void {
    var iter = ChunkIterator{ .data = xorb_data, .chunk_index = start_chunk_index };
    while (try iter.next()) |chunk| {
        // Chunks beyond the map are not referenced by any term.
        if (chunk.index >= xorb_entry.chunk_to_terms.len) continue;
        const term_refs = xorb_entry.chunk_to_terms[chunk.index];
        if (term_refs.len == 0) continue;

        const decompressed = try decompressChunk(chunk, dst, tmp);
        for (term_refs) |term_idx| {
            try onChunk(context, term_idx, decompressed);
        }
    }
}

// ── Destination Offset Resolver ─────────────────────────────────────────────

/// Returns the destination file offset for `chunk_len` bytes about to be
/// pushed for `term_idx`, then advances `bytes_pushed_per_term[term_idx]`.
///
/// `bytes_pushed_per_term` is caller-provided scratch sized to `term_ranges.len`
/// and zeroed once before processing. No allocation, no memcpy.
pub inline fn nextDestOffset(
    term_ranges: []const TermRange,
    bytes_pushed_per_term: []u64,
    term_idx: u32,
    chunk_len: u64,
) u64 {
    const dest = term_ranges[term_idx].file_start + bytes_pushed_per_term[term_idx];
    bytes_pushed_per_term[term_idx] += chunk_len;
    return dest;
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "computeTermRanges: small synthetic case" {
    const json_str =
        \\{
        \\  "offset_into_first_range": 100,
        \\  "terms": [
        \\    {"hash": "aaa", "unpacked_length": 500, "range": {"start": 0, "end": 5}},
        \\    {"hash": "bbb", "unpacked_length": 300, "range": {"start": 0, "end": 3}},
        \\    {"hash": "ccc", "unpacked_length": 200, "range": {"start": 0, "end": 2}}
        \\  ]
        \\}
    ;

    const parsed = try std.json.parseFromSlice(ReconstructionResponse, std.testing.allocator, json_str, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const response = parsed.value;

    const group_range_start: u64 = 1000;
    const group_range_end: u64 = 1900;

    var out: [3]TermRange = undefined;
    computeTermRanges(response, group_range_start, &out);

    // Invariant 1: sum(unpacked_lengths) - offset == range_end - range_start
    var total: u64 = 0;
    for (response.terms) |t| total += t.unpacked_length;
    try std.testing.expectEqual(group_range_end - group_range_start, total - response.offset_into_first_range);

    // Invariant 2: contiguity
    for (0..out.len - 1) |i| {
        try std.testing.expectEqual(out[i].file_end, out[i + 1].file_start);
    }

    // Spot-check values
    try std.testing.expectEqual(@as(u64, 900), out[0].file_start);
    try std.testing.expectEqual(@as(u64, 1400), out[0].file_end);
    try std.testing.expectEqual(@as(u64, 1700), out[1].file_end);
    try std.testing.expectEqual(@as(u64, 1900), out[2].file_end);
}

test "computeTermRanges: llama-70B group_000 (real data)" {
    // Values extracted from xet-llama-debug/group_000.json and manifest.json.
    // Single group covering the entire file, so invariant 1 holds exactly.
    const group_range_start: u64 = 1768;
    const group_range_end: u64 = 4584408808;
    const offset_into_first_range: u64 = 1768;
    const num_terms: usize = 4109;
    const sum_unpacked_lengths: u64 = 4584408808;

    // Invariant 1: sum - offset == range_end - range_start
    try std.testing.expectEqual(
        group_range_end - group_range_start,
        sum_unpacked_lengths - offset_into_first_range,
    );

    // Verify the function produces correct first-term position for this data.
    // We don't embed all 4109 terms, but we can verify the arithmetic with a
    // representative prefix.
    const prefix_lengths = [_]u64{ 10591, 64606297, 61198288, 47477, 5851501 };
    const terms = makeTerms(&prefix_lengths);
    var out: [prefix_lengths.len]TermRange = undefined;
    computeTermRanges(.{
        .offset_into_first_range = offset_into_first_range,
        .terms = terms[0..prefix_lengths.len],
    }, group_range_start, &out);

    // term[0].file_start == group_range_start - offset == 0 (start of file)
    try std.testing.expectEqual(@as(u64, 0), out[0].file_start);
    try std.testing.expectEqual(@as(u64, 10591), out[0].file_end);

    // Contiguity
    for (0..out.len - 1) |i| {
        try std.testing.expectEqual(out[i].file_end, out[i + 1].file_start);
    }

    // Last prefix term ends at sum of prefix lengths
    var prefix_sum: u64 = 0;
    for (prefix_lengths) |l| prefix_sum += l;
    try std.testing.expectEqual(prefix_sum, out[out.len - 1].file_end);

    _ = num_terms;
}

test "computeTermRanges: ltx group_000 (real data, multi-group)" {
    // Values from xet-ltx-debug/group_000.json and manifest.json.
    // Multi-group file: last term extends past group_range_end (83327 trailing bytes
    // that belong to the next group's overlap). Invariant 1 does NOT hold exactly;
    // instead sum - offset > range_end - range_start. Contiguity still holds.
    const group_range_start: u64 = 872440;
    const offset_into_first_range: u64 = 2543;
    const sum_unpacked_lengths: u64 = 10721839186;
    const group_range_end: u64 = 10722625756;
    const num_terms: usize = 161;

    // For multi-group: sum - offset >= range_size (trailing bytes at end)
    const range_size = group_range_end - group_range_start;
    const decoded_size = sum_unpacked_lengths - offset_into_first_range;
    try std.testing.expect(decoded_size >= range_size);

    // Verify first term position
    const prefix_lengths = [_]u64{ 66199766, 67074138, 37485, 66947766 };
    const terms = makeTerms(&prefix_lengths);
    var out: [prefix_lengths.len]TermRange = undefined;
    computeTermRanges(.{
        .offset_into_first_range = offset_into_first_range,
        .terms = terms[0..prefix_lengths.len],
    }, group_range_start, &out);

    // term[0].file_start = 872440 - 2543 = 869897
    try std.testing.expectEqual(@as(u64, 869897), out[0].file_start);
    try std.testing.expectEqual(@as(u64, 869897 + 66199766), out[0].file_end);

    // Contiguity
    for (0..out.len - 1) |i| {
        try std.testing.expectEqual(out[i].file_end, out[i + 1].file_start);
    }

    _ = num_terms;
}

test "buildXorbMap: small synthetic case" {
    // Two xorbs: "aaa" referenced by terms 0 and 2, "bbb" by term 1.
    // term 0: aaa [0,3), term 1: bbb [0,2), term 2: aaa [3,5)
    const json_str =
        \\{
        \\  "offset_into_first_range": 0,
        \\  "terms": [
        \\    {"hash": "aaa", "unpacked_length": 300, "range": {"start": 0, "end": 3}},
        \\    {"hash": "bbb", "unpacked_length": 200, "range": {"start": 0, "end": 2}},
        \\    {"hash": "aaa", "unpacked_length": 200, "range": {"start": 3, "end": 5}}
        \\  ]
        \\}
    ;
    const parsed = try std.json.parseFromSlice(ReconstructionResponse, std.testing.allocator, json_str, .{
        .ignore_unknown_fields = true,
    });
    defer parsed.deinit();
    const terms = parsed.value.terms;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const map = try buildXorbMap(arena.allocator(), terms);

    // 2 unique xorbs
    try std.testing.expectEqual(@as(usize, 2), map.entries.len);

    // Find "aaa" and "bbb" entries (order depends on sort).
    var aaa_entry: ?XorbMap.Entry = null;
    var bbb_entry: ?XorbMap.Entry = null;
    for (map.entries) |entry| {
        if (std.mem.eql(u8, entry.hash, "aaa")) aaa_entry = entry;
        if (std.mem.eql(u8, entry.hash, "bbb")) bbb_entry = entry;
    }

    // aaa: max_end = 5, chunks 0,1,2 → term 0; chunks 3,4 → term 2
    const aaa = aaa_entry.?;
    try std.testing.expectEqual(@as(usize, 5), aaa.chunk_to_terms.len);
    for (0..3) |c| {
        try std.testing.expectEqual(@as(usize, 1), aaa.chunk_to_terms[c].len);
        try std.testing.expectEqual(@as(u32, 0), aaa.chunk_to_terms[c][0]);
    }
    for (3..5) |c| {
        try std.testing.expectEqual(@as(usize, 1), aaa.chunk_to_terms[c].len);
        try std.testing.expectEqual(@as(u32, 2), aaa.chunk_to_terms[c][0]);
    }

    // bbb: max_end = 2, chunks 0,1 → term 1
    const bbb = bbb_entry.?;
    try std.testing.expectEqual(@as(usize, 2), bbb.chunk_to_terms.len);
    for (0..2) |c| {
        try std.testing.expectEqual(@as(usize, 1), bbb.chunk_to_terms[c].len);
        try std.testing.expectEqual(@as(u32, 1), bbb.chunk_to_terms[c][0]);
    }

    // Total refs = 3 + 2 + 2 = 7
    var total_refs: usize = 0;
    for (map.entries) |entry| {
        for (entry.chunk_to_terms) |cts| total_refs += cts.len;
    }
    try std.testing.expectEqual(@as(usize, 7), total_refs);
}

/// Helper: create a fixed-size array of Term structs from unpacked_lengths.
fn makeTerms(lengths: []const u64) [128]Term {
    var terms: [128]Term = @splat(Term{
        .hash = "0" ** 64,
        .unpacked_length = 0,
        .range = .{ .start = 0, .end = 0 },
    });
    for (lengths, 0..) |len, i| {
        terms[i].unpacked_length = len;
    }
    return terms;
}

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

// ── processXorb tests ───────────────────────────────────────────────────────

/// Helper: build a synthetic xorb binary from raw (uncompressed) payloads.
fn buildSyntheticXorb(payloads: []const []const u8, buf: []u8) []const u8 {
    var off: usize = 0;
    for (payloads) |payload| {
        writeChunkHeader(buf[off..][0..8], @intCast(payload.len), 0, @intCast(payload.len));
        off += 8;
        @memcpy(buf[off..][0..payload.len], payload);
        off += payload.len;
    }
    return buf[0..off];
}

test "processXorb: basic routing and byte counts" {
    // 4 chunks, 3 terms:
    //   term 0 ("x"): chunks [0, 2) → chunks 0, 1
    //   term 1 ("x"): chunks [2, 4) → chunks 2, 3
    //   term 2 ("x"): chunks [1, 3) → chunks 1, 2 (overlaps both)
    const payloads = [_][]const u8{ &.{ 0xAA, 0xBB }, &.{ 0xCC, 0xDD, 0xEE }, &.{0xFF}, &.{ 0x11, 0x22, 0x33, 0x44 } };
    var xorb_buf: [4 * (8 + 4)]u8 = undefined;
    const xorb_data = buildSyntheticXorb(&payloads, &xorb_buf);

    const terms = [_]Term{
        .{ .hash = "x", .unpacked_length = 5, .range = .{ .start = 0, .end = 2 } },
        .{ .hash = "x", .unpacked_length = 5, .range = .{ .start = 2, .end = 4 } },
        .{ .hash = "x", .unpacked_length = 4, .range = .{ .start = 1, .end = 3 } },
    };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const map = try buildXorbMap(arena.allocator(), &terms);
    try std.testing.expectEqual(@as(usize, 1), map.entries.len);

    const Ctx = struct {
        counters: [3]u64,
        // Fixed buffers to collect received data (max 10 bytes each is plenty).
        data: [3][10]u8,
        data_len: [3]usize,

        fn onChunk(self: *@This(), term_idx: u32, data: []const u8) !void {
            self.counters[term_idx] += data.len;
            const off = self.data_len[term_idx];
            @memcpy(self.data[term_idx][off..][0..data.len], data);
            self.data_len[term_idx] = off + data.len;
        }
    };
    var ctx = Ctx{
        .counters = .{ 0, 0, 0 },
        .data = undefined,
        .data_len = .{ 0, 0, 0 },
    };

    var dst: [128]u8 = undefined;
    var tmp: [128]u8 = undefined;
    try processXorb(xorb_data, map.entries[0], 0, &dst, &tmp, &ctx, Ctx.onChunk);

    // Verify byte counts match unpacked_length.
    try std.testing.expectEqual(@as(u64, 5), ctx.counters[0]); // 2 + 3
    try std.testing.expectEqual(@as(u64, 5), ctx.counters[1]); // 1 + 4
    try std.testing.expectEqual(@as(u64, 4), ctx.counters[2]); // 3 + 1

    // Verify actual data: term 0 gets chunks 0+1, term 1 gets chunks 2+3, term 2 gets chunks 1+2.
    try std.testing.expectEqualSlices(u8, &.{ 0xAA, 0xBB, 0xCC, 0xDD, 0xEE }, ctx.data[0][0..ctx.data_len[0]]);
    try std.testing.expectEqualSlices(u8, &.{ 0xFF, 0x11, 0x22, 0x33, 0x44 }, ctx.data[1][0..ctx.data_len[1]]);
    try std.testing.expectEqualSlices(u8, &.{ 0xCC, 0xDD, 0xEE, 0xFF }, ctx.data[2][0..ctx.data_len[2]]);
}

test "processXorb: unreferenced chunks are skipped" {
    // 3 chunks, but only chunk 1 is referenced.
    const payloads = [_][]const u8{ &.{0x00}, &.{ 0xAA, 0xBB }, &.{0x00} };
    var xorb_buf: [3 * (8 + 4)]u8 = undefined;
    const xorb_data = buildSyntheticXorb(&payloads, &xorb_buf);

    const terms = [_]Term{
        .{ .hash = "y", .unpacked_length = 2, .range = .{ .start = 1, .end = 2 } },
    };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const map = try buildXorbMap(arena.allocator(), &terms);

    const Ctx = struct {
        call_count: u32,
        chunks_seen: [3]bool,

        fn onChunk(self: *@This(), _: u32, data: []const u8) !void {
            self.call_count += 1;
            // Only chunk 1 data (0xAA, 0xBB) should arrive.
            if (data.len == 2 and data[0] == 0xAA and data[1] == 0xBB) {
                self.chunks_seen[1] = true;
            }
        }
    };
    var ctx = Ctx{ .call_count = 0, .chunks_seen = .{ false, false, false } };

    var dst: [128]u8 = undefined;
    var tmp: [128]u8 = undefined;
    try processXorb(xorb_data, map.entries[0], 0, &dst, &tmp, &ctx, Ctx.onChunk);

    // Only 1 callback call (chunk 1 → term 0).
    try std.testing.expectEqual(@as(u32, 1), ctx.call_count);
    try std.testing.expect(!ctx.chunks_seen[0]);
    try std.testing.expect(ctx.chunks_seen[1]);
    try std.testing.expect(!ctx.chunks_seen[2]);
}

test "processXorb: deduplication — shared chunk calls callback twice" {
    // 1 chunk referenced by 2 terms.
    const payloads = [_][]const u8{&.{ 0xDE, 0xAD }};
    var xorb_buf: [1 * (8 + 2)]u8 = undefined;
    const xorb_data = buildSyntheticXorb(&payloads, &xorb_buf);

    const terms = [_]Term{
        .{ .hash = "z", .unpacked_length = 2, .range = .{ .start = 0, .end = 1 } },
        .{ .hash = "z", .unpacked_length = 2, .range = .{ .start = 0, .end = 1 } },
    };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const map = try buildXorbMap(arena.allocator(), &terms);

    const Ctx = struct {
        call_count: u32,
        term_seen: [2]bool,
        last_data: [2]u8,

        fn onChunk(self: *@This(), term_idx: u32, data: []const u8) !void {
            self.call_count += 1;
            self.term_seen[term_idx] = true;
            @memcpy(&self.last_data, data[0..2]);
        }
    };
    var ctx = Ctx{ .call_count = 0, .term_seen = .{ false, false }, .last_data = .{ 0, 0 } };

    var dst: [128]u8 = undefined;
    var tmp: [128]u8 = undefined;
    try processXorb(xorb_data, map.entries[0], 0, &dst, &tmp, &ctx, Ctx.onChunk);

    // Callback called twice (once per term), both see same data.
    try std.testing.expectEqual(@as(u32, 2), ctx.call_count);
    try std.testing.expect(ctx.term_seen[0]);
    try std.testing.expect(ctx.term_seen[1]);
    try std.testing.expectEqualSlices(u8, &.{ 0xDE, 0xAD }, &ctx.last_data);
}

test "processXorb: chunks beyond map length are skipped" {
    // 3 chunks in xorb, but map only covers 1 chunk.
    const payloads = [_][]const u8{ &.{0x01}, &.{0x02}, &.{0x03} };
    var xorb_buf: [3 * (8 + 1)]u8 = undefined;
    const xorb_data = buildSyntheticXorb(&payloads, &xorb_buf);

    const terms = [_]Term{
        .{ .hash = "w", .unpacked_length = 1, .range = .{ .start = 0, .end = 1 } },
    };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const map = try buildXorbMap(arena.allocator(), &terms);

    const Ctx = struct {
        call_count: u32,
        total_bytes: u64,

        fn onChunk(self: *@This(), _: u32, data: []const u8) !void {
            self.call_count += 1;
            self.total_bytes += data.len;
        }
    };
    var ctx = Ctx{ .call_count = 0, .total_bytes = 0 };

    var dst: [128]u8 = undefined;
    var tmp: [128]u8 = undefined;
    try processXorb(xorb_data, map.entries[0], 0, &dst, &tmp, &ctx, Ctx.onChunk);

    // Only chunk 0 triggers the callback.
    try std.testing.expectEqual(@as(u32, 1), ctx.call_count);
    try std.testing.expectEqual(@as(u64, 1), ctx.total_bytes);
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

test "nextDestOffset: contiguous push per term" {
    // Two terms in the file: term 0 at file offset 1000 (length 300),
    // term 1 at file offset 1300 (length 200).
    const term_ranges = [_]TermRange{
        .{ .file_start = 1000, .file_end = 1300 },
        .{ .file_start = 1300, .file_end = 1500 },
    };
    var bytes_pushed = [_]u64{ 0, 0 };

    // Push 3 chunks of 100 bytes each into term 0, interleaved with one
    // 200-byte chunk for term 1 in the middle (simulating term 1 sharing
    // the same xorb pass).
    try std.testing.expectEqual(@as(u64, 1000), nextDestOffset(&term_ranges, &bytes_pushed, 0, 100));
    try std.testing.expectEqual(@as(u64, 1100), nextDestOffset(&term_ranges, &bytes_pushed, 0, 100));
    try std.testing.expectEqual(@as(u64, 1300), nextDestOffset(&term_ranges, &bytes_pushed, 1, 200));
    try std.testing.expectEqual(@as(u64, 1200), nextDestOffset(&term_ranges, &bytes_pushed, 0, 100));

    // Each term is now fully consumed.
    try std.testing.expectEqual(@as(u64, 300), bytes_pushed[0]);
    try std.testing.expectEqual(@as(u64, 200), bytes_pushed[1]);
}
