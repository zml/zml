const std = @import("std");

const JSON_PARSE_SCRATCH_LEN = 512 * 1024;

pub const ReconstructionRange = struct {
    offset: u64,
    len: u64,

    pub fn contains(self: ReconstructionRange, offset: u64, len: u64) bool {
        return self.offset <= offset and offset + len <= self.offset + self.len;
    }

    pub fn formatCacheKey(self: ReconstructionRange, buffer: []u8, file_hash: []const u8) ![]u8 {
        return try std.fmt.bufPrint(buffer, "{s}:{d}:{d}", .{ file_hash, self.offset, self.len });
    }
};

pub const Range = struct {
    start: u64,
    end: u64,

    fn contains(self: Range, other: Range) bool {
        return self.start <= other.start and self.end >= other.end;
    }
};

const Term = struct {
    hash: []const u8,
    unpacked_length: u64,
    range: Range,
};

pub const FetchInfo = struct {
    range: Range,
    url: []const u8,
    url_range: Range,
};

const RangeDescriptor = struct {
    chunks: Range,
    bytes: Range,
};

const MultiRangeFetch = struct {
    url: []const u8,
    ranges: []RangeDescriptor,
};

const JsonV2 = struct {
    offset_into_first_range: u64,
    terms: []Term,
    xorbs: std.json.ArrayHashMap([]MultiRangeFetch),
};

pub const TermEntry = struct {
    file_start: u64,
    visible_length: u64,
    base_skip: u64,
    range: Range,
    fetch_index: usize,
};

pub const FetchEntry = struct {
    hash: []u8,
    info: FetchInfo,

    fn deinit(self: *FetchEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.hash);
        allocator.free(self.info.url);
    }
};

pub const Index = struct {
    terms: []TermEntry = &.{},
    fetches: []FetchEntry = &.{},
    range: ReconstructionRange = .{ .offset = 0, .len = 0 },
    expires_at: i64 = 0,

    fn init(
        allocator: std.mem.Allocator,
        reconstruction: JsonV2,
        range: ReconstructionRange,
        expires_at: i64,
    ) !Index {
        var terms: std.ArrayList(TermEntry) = .empty;
        errdefer terms.deinit(allocator);

        var fetches: std.ArrayList(FetchEntry) = .empty;
        errdefer {
            for (fetches.items) |*fetch| fetch.deinit(allocator);
            fetches.deinit(allocator);
        }

        var file_cursor: u64 = range.offset;
        var remaining = range.len;
        for (reconstruction.terms, 0..) |term, i| {
            if (remaining == 0) break;
            const base_skip = if (i == 0) reconstruction.offset_into_first_range else 0;
            if (base_skip >= term.unpacked_length) continue;
            const visible_length = @min(term.unpacked_length - base_skip, remaining);
            const info = findFetchInfo(reconstruction.xorbs, term.hash, term.range) orelse return error.InvalidReconstruction;
            const fetch_index = try appendFetch(allocator, &fetches, term.hash, info);
            try terms.append(allocator, .{
                .file_start = file_cursor,
                .visible_length = visible_length,
                .base_skip = base_skip,
                .range = term.range,
                .fetch_index = fetch_index,
            });
            file_cursor += visible_length;
            remaining -= visible_length;
        }

        const term_slice = try terms.toOwnedSlice(allocator);
        errdefer allocator.free(term_slice);

        return .{
            .terms = term_slice,
            .fetches = try fetches.toOwnedSlice(allocator),
            .range = range,
            .expires_at = expires_at,
        };
    }

    pub fn expired(self: Index, io: std.Io) bool {
        const now = std.Io.Timestamp.now(io, .real).toSeconds();
        return now + 30 >= self.expires_at;
    }

    pub fn deinit(self: *Index, allocator: std.mem.Allocator) void {
        for (self.fetches) |*fetch| fetch.deinit(allocator);
        allocator.free(self.fetches);
        allocator.free(self.terms);
        self.* = .{};
    }
};

pub fn parseIndexBody(allocator: std.mem.Allocator, range: ReconstructionRange, expires_at: i64, body: []const u8) !Index {
    var json_parse_buffer: [JSON_PARSE_SCRATCH_LEN]u8 = undefined;
    var json_parse_fba = std.heap.FixedBufferAllocator.init(&json_parse_buffer);
    var parsed = try std.json.parseFromSlice(JsonV2, json_parse_fba.allocator(), body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return try Index.init(allocator, parsed.value, range, expires_at);
}

fn appendFetch(allocator: std.mem.Allocator, fetches: *std.ArrayList(FetchEntry), hash: []const u8, info: FetchInfo) !usize {
    for (fetches.items, 0..) |fetch, i| {
        if (fetchInfoEql(fetch.hash, fetch.info, hash, info)) return i;
    }

    const hash_copy = try allocator.dupe(u8, hash);
    errdefer allocator.free(hash_copy);
    const url_copy = try allocator.dupe(u8, info.url);
    errdefer allocator.free(url_copy);

    try fetches.append(allocator, .{
        .hash = hash_copy,
        .info = .{
            .range = info.range,
            .url = url_copy,
            .url_range = info.url_range,
        },
    });
    return fetches.items.len - 1;
}

fn fetchInfoEql(a_hash: []const u8, a_info: FetchInfo, b_hash: []const u8, b_info: FetchInfo) bool {
    return std.mem.eql(u8, a_hash, b_hash) and
        a_info.range.start == b_info.range.start and
        a_info.range.end == b_info.range.end and
        a_info.url_range.start == b_info.url_range.start and
        a_info.url_range.end == b_info.url_range.end;
}

fn findFetchInfo(map: std.json.ArrayHashMap([]MultiRangeFetch), hash: []const u8, term_range: Range) ?FetchInfo {
    var it = map.map.iterator();
    while (it.next()) |entry| {
        if (!std.mem.eql(u8, entry.key_ptr.*, hash)) continue;
        for (entry.value_ptr.*) |fetch| {
            for (fetch.ranges) |range| {
                if (!range.chunks.contains(term_range)) continue;
                return .{
                    .range = range.chunks,
                    .url = fetch.url,
                    .url_range = range.bytes,
                };
            }
        }
        return null;
    }
    return null;
}
