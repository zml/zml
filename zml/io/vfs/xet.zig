const std = @import("std");
const lz4 = @import("lz4.zig");
const bg4 = @import("bg4.zig");

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

// ── Reconstruction Plan ─────────────────────────────────────────────────────

/// Translates file-relative offsets into `(term_index, intra_term_offset)`.
/// Owns the parsed CAS response (when built from JSON) plus a prefix-sum of
/// unpacked term lengths for O(log N) `locate` with no hot-path allocation.
pub const Plan = struct {
    response: ReconstructionResponse,
    /// `term_prefix[i] = sum(response.terms[0..i].unpacked_length)`.
    /// Length `response.terms.len + 1`.
    term_prefix: []u64,
    /// Backing arena when built from JSON; null when constructed from
    /// caller-owned components (test seam).
    parsed_arena: ?*std.heap.ArenaAllocator,

    pub const Located = struct {
        term_index: usize,
        intra_term_offset: u64,
    };

    /// Wraps a parsed reconstruction response; takes ownership of `parsed`.
    pub fn fromParsed(allocator: std.mem.Allocator, parsed: std.json.Parsed(ReconstructionResponse)) !Plan {
        const prefix = try buildTermPrefix(allocator, parsed.value.terms);
        return .{
            .response = parsed.value,
            .term_prefix = prefix,
            .parsed_arena = parsed.arena,
        };
    }

    /// Test seam: builds a Plan over caller-owned slices. The provided
    /// `terms` / `fetch_info` must outlive the returned Plan.
    fn fromComponents(
        allocator: std.mem.Allocator,
        terms: []const Term,
        fetch_info: std.json.ArrayHashMap([]const FetchUrl),
        offset_into_first_range: u64,
    ) !Plan {
        const prefix = try buildTermPrefix(allocator, terms);
        return .{
            .response = .{
                .offset_into_first_range = offset_into_first_range,
                .terms = terms,
                .fetch_info = fetch_info,
            },
            .term_prefix = prefix,
            .parsed_arena = null,
        };
    }

    pub fn deinit(self: Plan, allocator: std.mem.Allocator) void {
        allocator.free(self.term_prefix);
        if (self.parsed_arena) |arena| {
            const child = arena.child_allocator;
            arena.deinit();
            child.destroy(arena);
        }
    }

    pub fn fileSize(self: Plan) u64 {
        return self.term_prefix[self.term_prefix.len - 1] - self.response.offset_into_first_range;
    }

    /// Resolves a file-relative byte offset to the owning term plus the
    /// number of unpacked bytes to skip inside that term. Returns
    /// `error.OutOfRange` for `file_offset >= fileSize()`.
    pub fn locate(self: Plan, file_offset: u64) !Located {
        if (file_offset >= self.fileSize()) return error.OutOfRange;
        const stream_pos = self.response.offset_into_first_range + file_offset;
        // Largest i such that term_prefix[i] <= stream_pos.
        var lo: usize = 0;
        var hi: usize = self.response.terms.len;
        while (lo + 1 < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.term_prefix[mid] <= stream_pos) lo = mid else hi = mid;
        }
        return .{ .term_index = lo, .intra_term_offset = stream_pos - self.term_prefix[lo] };
    }

    pub fn fetchFor(self: Plan, xorb_hash: []const u8) ?[]const FetchUrl {
        return self.response.fetch_info.map.get(xorb_hash);
    }
};

fn buildTermPrefix(allocator: std.mem.Allocator, terms: []const Term) ![]u64 {
    const prefix = try allocator.alloc(u64, terms.len + 1);
    prefix[0] = 0;
    for (terms, 0..) |t, i| prefix[i + 1] = prefix[i] + t.unpacked_length;
    return prefix;
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
        if (compression_type > 2) return error.InvalidCompressionType;
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

/// Decompresses a single chunk into a caller-provided buffer.
/// `dst` must be at least `chunk.uncompressed_size` bytes.
/// Returns `dst[0..chunk.uncompressed_size]`.
pub fn decompressChunk(chunk: ChunkIterator.Chunk, dst: []u8) ![]u8 {
    const usize_uncomp: usize = chunk.uncompressed_size;
    if (dst.len < usize_uncomp) return error.OutputTooSmall;
    const out = dst[0..usize_uncomp];

    switch (chunk.compression_type) {
        0 => {
            // None: raw data, just copy.
            if (chunk.compressed_data.len != usize_uncomp) return error.SizeMismatch;
            @memcpy(out, chunk.compressed_data);
            return out;
        },
        1 => {
            // LZ4 block/frame.
            var src_reader: std.Io.Reader = .fixed(chunk.compressed_data);
            var dst_writer: std.Io.Writer = .fixed(out);
            var lz4_reader = lz4.BlockReader.init(&src_reader, chunk.compressed_size, usize_uncomp);
            _ = lz4_reader.interface.streamRemaining(&dst_writer) catch return error.CorruptedData;
            return out;
        },
        2 => {
            // ByteGrouping4 + LZ4: LZ4 decompresses into dst in grouped layout,
            // then DegroupWriter.flush() permutes in place.
            var src_reader: std.Io.Reader = .fixed(chunk.compressed_data);
            var grouped_writer = bg4.DegroupWriter.init(out);
            var lz4_reader = lz4.BlockReader.init(&src_reader, chunk.compressed_size, usize_uncomp);
            _ = lz4_reader.interface.streamRemaining(&grouped_writer.interface) catch return error.CorruptedData;
            grouped_writer.interface.flush() catch return error.CorruptedData;
            return out;
        },
        else => return error.InvalidCompressionType,
    }
}

// ── Chunk Ring ──────────────────────────────────────────────────────────────

/// Per-Handle cache of recently decompressed XET chunks, keyed by
/// `(xorb_hash, chunk_index)`. `xorb_hash` is borrowed from a `Plan` and
/// must outlive every cached slot — the ring never copies the hash.
/// Round-robin eviction. No allocation on the hot path.
pub const ChunkRing = struct {
    /// Protocol cap: observed chunks ≤ ~100 KiB uncompressed on llama-3-70B,
    /// but AWQ-quantized shards have larger chunks. Reference
    /// `test_file_to_device` uses 128 KiB; we bump to 256 KiB to cover
    /// hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4.
    pub const chunk_max_bytes: usize = 256 * 1024;
    /// Holds the last `slot_count` decompressed chunks; sized to cover
    /// safetensors length prefix + JSON header + first tensor body plus
    /// one slack slot for round-robin eviction.
    pub const slot_count: usize = 4;

    pub const Slot = struct {
        valid: bool = false,
        xorb_hash: []const u8 = &.{},
        chunk_index: u32 = 0,
        len: u32 = 0,
        bytes: [chunk_max_bytes]u8 = undefined,
    };

    /// Fills `out` with raw (compressed) xorb bytes whose first byte is the
    /// header of `chunk_index`. Must write at least one complete chunk and
    /// return its on-wire byte count.
    pub const Fetcher = struct {
        ctx: *anyopaque,
        fetchFn: *const fn (ctx: *anyopaque, xorb_hash: []const u8, chunk_index: u32, out: []u8) anyerror!usize,
    };

    slots: [slot_count]Slot = @splat(.{}),
    next_victim: u8 = 0,

    pub fn init() ChunkRing {
        return .{};
    }

    /// Returns a borrowed slice of the decompressed chunk's bytes. The slice
    /// is valid until the next `get` call evicts this slot.
    pub fn get(
        self: *ChunkRing,
        xorb_hash: []const u8,
        chunk_index: u32,
        fetcher: Fetcher,
        scratch: []u8,
    ) ![]const u8 {
        for (&self.slots) |*s| {
            if (s.valid and s.chunk_index == chunk_index and std.mem.eql(u8, s.xorb_hash, xorb_hash)) {
                return s.bytes[0..s.len];
            }
        }
        const n = try fetcher.fetchFn(fetcher.ctx, xorb_hash, chunk_index, scratch);
        var it: ChunkIterator = .{ .data = scratch[0..n] };
        const chunk = (try it.next()) orelse return error.EmptyFetch;

        const victim = &self.slots[self.next_victim];
        self.next_victim = (self.next_victim + 1) % @as(u8, @intCast(slot_count));
        const out = try decompressChunk(chunk, &victim.bytes);
        victim.valid = true;
        victim.xorb_hash = xorb_hash;
        victim.chunk_index = chunk_index;
        victim.len = @intCast(out.len);
        return out;
    }
};

// ── HF CAS State ────────────────────────────────────────────────────────────

/// Per-process Xet/HF CAS state. Holds the user's HF token and caches the
/// per-file Xet id and per-(repo, rev) CAS access token so repeated
/// `reconstruct` calls against the same model don't re-issue the handshake
/// round-trips. Not thread-safe: serialize calls externally.
pub const State = struct {
    pub const Repo = struct {
        repo: []const u8,
        model: []const u8,
        rev: []const u8,
        path: []const u8,
    };

    pub const CasAuth = struct {
        url: []const u8,
        token: []const u8,
        /// Unix seconds at which the HF-issued token expires.
        exp: i64,
    };

    /// LRU cache of compressed xorb windows (`FetchUrl.url_range` byte
    /// ranges) shared across all open handles on a single State. Keyed by
    /// `(xorb_hash, url_range_start, url_range_end)`. Entries hold the raw
    /// bytes of an HTTP Range GET; consumers walk them with
    /// `ChunkIterator`. Bounded by `max_bytes` (default 2 GiB) — evicts LRU
    /// entries to fit. Thread-safe: callers may invoke `acquire`/`release`
    /// concurrently. A non-zero `pin` count prevents eviction of an entry
    /// while another caller holds its slice.
    pub const WindowCache = struct {
        pub const default_max_bytes: u64 = 2 * 1024 * 1024 * 1024;

        const Entry = struct {
            valid: bool = false,
            /// Borrowed from a `Plan`; outlives every entry.
            hash: []const u8 = &.{},
            url_start: u64 = 0,
            url_end: u64 = 0,
            bytes: []u8 = &.{},
            last_used: u64 = 0,
            pin: u32 = 0,
        };

        pub const Acquired = struct {
            slice: []u8,
            hit: bool,
            index: u32,
        };

        allocator: std.mem.Allocator,
        mutex: std.Io.Mutex = .init,
        entries: std.ArrayListUnmanaged(Entry) = .empty,
        used_bytes: u64 = 0,
        max_bytes: u64 = default_max_bytes,
        counter: u64 = 0,

        pub fn deinit(self: *WindowCache) void {
            for (self.entries.items) |*e| {
                if (e.valid) self.allocator.free(e.bytes);
            }
            self.entries.deinit(self.allocator);
        }

        /// Returns a hit (read-only borrowed slice) or allocates a fresh
        /// `byte_len`-sized writable slot for the caller to fill. The
        /// entry is pinned for the caller; pair every `acquire` with a
        /// matching `release(io, index)` (use `defer`).
        pub fn acquire(
            self: *WindowCache,
            io: std.Io,
            hash: []const u8,
            url_start: u64,
            url_end: u64,
            byte_len: usize,
        ) !Acquired {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.counter += 1;
            for (self.entries.items, 0..) |*e, i| {
                if (e.valid and e.url_start == url_start and e.url_end == url_end and std.mem.eql(u8, e.hash, hash)) {
                    e.last_used = self.counter;
                    e.pin += 1;
                    return .{ .slice = e.bytes, .hit = true, .index = @intCast(i) };
                }
            }
            // Miss: evict LRU among unpinned entries until there is room.
            while (self.used_bytes + byte_len > self.max_bytes) {
                var lru_idx: usize = 0;
                var lru_val: u64 = std.math.maxInt(u64);
                var found = false;
                for (self.entries.items, 0..) |*e, i| {
                    if (e.valid and e.pin == 0 and e.last_used < lru_val) {
                        lru_val = e.last_used;
                        lru_idx = i;
                        found = true;
                    }
                }
                if (!found) return error.CacheFull;
                const victim = &self.entries.items[lru_idx];
                self.used_bytes -= victim.bytes.len;
                self.allocator.free(victim.bytes);
                victim.* = .{};
            }
            const buf = try self.allocator.alloc(u8, byte_len);
            errdefer self.allocator.free(buf);
            var slot_idx: ?usize = null;
            for (self.entries.items, 0..) |*e, i| {
                if (!e.valid) {
                    slot_idx = i;
                    break;
                }
            }
            const idx = slot_idx orelse blk: {
                try self.entries.append(self.allocator, .{});
                break :blk self.entries.items.len - 1;
            };
            self.entries.items[idx] = .{
                .valid = true,
                .hash = hash,
                .url_start = url_start,
                .url_end = url_end,
                .bytes = buf,
                .last_used = self.counter,
                .pin = 1,
            };
            self.used_bytes += byte_len;
            return .{ .slice = buf, .hit = false, .index = @intCast(idx) };
        }

        pub fn release(self: *WindowCache, io: std.Io, index: u32) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            const e = &self.entries.items[index];
            if (e.pin > 0) e.pin -= 1;
        }
    };

    /// Skew applied to the cached token's expiry: treat a token as expired
    /// this many seconds before its real `exp` to avoid races with in-flight requests.
    const token_skew_seconds: i64 = 30;

    allocator: std.mem.Allocator,
    http: *std.http.Client,
    /// User's HF token, raw (no "Bearer " prefix). Whitespace is trimmed on use.
    hf_token: []const u8,

    file_id_cache: std.StringHashMapUnmanaged([]const u8) = .{},
    cas_cache: std.StringHashMapUnmanaged(CasAuth) = .{},
    plan_cache: std.StringHashMapUnmanaged(*Plan) = .{},
    /// Total bytes pulled from the CAS over HTTP across this State's lifetime.
    /// Bumped by xorb-window fetchers; used to measure dedup effectiveness.
    bytes_fetched: std.atomic.Value(u64) = .init(0),
    window_cache: WindowCache,

    pub fn init(allocator: std.mem.Allocator, http: *std.http.Client, hf_token: []const u8) State {
        return .{
            .allocator = allocator,
            .http = http,
            .hf_token = hf_token,
            .window_cache = .{ .allocator = allocator },
        };
    }

    pub fn deinit(self: *State) void {
        self.window_cache.deinit();
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
        var pit = self.plan_cache.iterator();
        while (pit.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            e.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(e.value_ptr.*);
        }
        self.plan_cache.deinit(self.allocator);
    }

    /// Returns the cached Xet file id for `repo` (or fetches+caches on miss).
    /// The returned slice is owned by the client and valid until `deinit`.
    pub fn fileId(self: *State, repo: Repo) ![]const u8 {
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
    /// Refreshes the cached entry when it is within `token_skew_seconds` of expiry.
    pub fn casAuth(self: *State, repo: Repo) !CasAuth {
        var key_buf: [4096]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}/{s}@{s}", .{ repo.repo, repo.model, repo.rev });
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.REALTIME, &ts);
        const now: i64 = @intCast(ts.sec);
        if (self.cas_cache.getEntry(key)) |entry| {
            if (now + token_skew_seconds < entry.value_ptr.exp) return entry.value_ptr.*;
            // Token within skew window: refresh in place. Fetch first so a
            // network failure leaves the (stale-but-allocated) entry untouched.
            const fresh = try fetchCasToken(self.allocator, self.http, repo, self.hf_token);
            self.allocator.free(entry.value_ptr.url);
            self.allocator.free(entry.value_ptr.token);
            entry.value_ptr.* = fresh;
            return fresh;
        }

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
        self: *State,
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

    /// Returns the cached `Plan` for `repo`'s whole file, building it via
    /// `reconstruct(repo, 0, file_size)` on miss. The returned pointer is
    /// stable for the lifetime of `State` (the map stores `*Plan`).
    pub fn getOrBuildPlan(self: *State, repo: Repo, file_size: u64) !*const Plan {
        var key_buf: [4096]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}/{s}@{s}/{s}", .{ repo.repo, repo.model, repo.rev, repo.path });
        if (self.plan_cache.get(key)) |p| return p;

        const parsed = try self.reconstruct(repo, 0, file_size);
        errdefer parsed.deinit();
        var plan = try Plan.fromParsed(self.allocator, parsed);
        errdefer plan.deinit(self.allocator);

        const owned_key = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(owned_key);
        const slot = try self.allocator.create(Plan);
        errdefer self.allocator.destroy(slot);
        slot.* = plan;
        try self.plan_cache.put(self.allocator, owned_key, slot);
        return slot;
    }
};

fn bearerAuth(hf_token: []const u8, buf: []u8) ![]u8 {
    return std.fmt.bufPrint(buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch error.TokenTooLong;
}

fn fetchFileId(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: State.Repo,
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
    repo: State.Repo,
    hf_token: []const u8,
) !State.CasAuth {
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
        exp: i64,
    }, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return .{
        .url = try allocator.dupe(u8, parsed.value.casUrl),
        .token = try allocator.dupe(u8, parsed.value.accessToken),
        .exp = parsed.value.exp,
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
    const result = try decompressChunk(chunk, &dst);
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
    try std.testing.expectError(error.SizeMismatch, decompressChunk(chunk, &dst));
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
    const result = try decompressChunk(chunk, &dst);
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
    const result = try decompressChunk(chunk, &dst);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 }, result);
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

test "Plan.locate: single term, offset_into_first_range = 0" {
    const terms = [_]Term{
        .{ .hash = "aaa", .unpacked_length = 100, .range = .{ .start = 0, .end = 4 } },
    };
    var plan = try Plan.fromComponents(std.testing.allocator, &terms, .{}, 0);
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 100), plan.fileSize());

    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 0 }, try plan.locate(0));
    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 50 }, try plan.locate(50));
    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 99 }, try plan.locate(99));
    try std.testing.expectError(error.OutOfRange, plan.locate(100));
    try std.testing.expectError(error.OutOfRange, plan.locate(1_000_000));
}

test "Plan.locate: multi-term boundaries and intra-term offsets" {
    const terms = [_]Term{
        .{ .hash = "aaa", .unpacked_length = 100, .range = .{ .start = 0, .end = 4 } },
        .{ .hash = "bbb", .unpacked_length = 200, .range = .{ .start = 0, .end = 8 } },
        .{ .hash = "ccc", .unpacked_length = 50, .range = .{ .start = 4, .end = 6 } },
    };
    var plan = try Plan.fromComponents(std.testing.allocator, &terms, .{}, 0);
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 350), plan.fileSize());

    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 99 }, try plan.locate(99));
    try std.testing.expectEqual(Plan.Located{ .term_index = 1, .intra_term_offset = 0 }, try plan.locate(100));
    try std.testing.expectEqual(Plan.Located{ .term_index = 1, .intra_term_offset = 199 }, try plan.locate(299));
    try std.testing.expectEqual(Plan.Located{ .term_index = 2, .intra_term_offset = 0 }, try plan.locate(300));
    try std.testing.expectEqual(Plan.Located{ .term_index = 2, .intra_term_offset = 49 }, try plan.locate(349));
    try std.testing.expectError(error.OutOfRange, plan.locate(350));
}

test "Plan.locate: nonzero offset_into_first_range" {
    const terms = [_]Term{
        .{ .hash = "aaa", .unpacked_length = 100, .range = .{ .start = 0, .end = 4 } },
        .{ .hash = "bbb", .unpacked_length = 200, .range = .{ .start = 0, .end = 8 } },
    };
    // Server returned terms covering [0, 300) but client asked for file bytes
    // starting at stream position 30: file_offset 0 ↔ stream_pos 30.
    var plan = try Plan.fromComponents(std.testing.allocator, &terms, .{}, 30);
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 270), plan.fileSize());

    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 30 }, try plan.locate(0));
    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 99 }, try plan.locate(69));
    try std.testing.expectEqual(Plan.Located{ .term_index = 1, .intra_term_offset = 0 }, try plan.locate(70));
    try std.testing.expectEqual(Plan.Located{ .term_index = 1, .intra_term_offset = 199 }, try plan.locate(269));
    try std.testing.expectError(error.OutOfRange, plan.locate(270));
}

test "Plan.fetchFor: returns urls keyed by xorb hash" {
    const terms = [_]Term{
        .{ .hash = "aaa", .unpacked_length = 10, .range = .{ .start = 0, .end = 1 } },
    };
    const aaa_urls = [_]FetchUrl{
        .{ .range = .{ .start = 0, .end = 10 }, .url = "https://example.com/aaa", .url_range = .{ .start = 0, .end = 1024 } },
    };
    var fi: std.json.ArrayHashMap([]const FetchUrl) = .{};
    defer fi.map.deinit(std.testing.allocator);
    try fi.map.put(std.testing.allocator, "aaa", &aaa_urls);

    var plan = try Plan.fromComponents(std.testing.allocator, &terms, fi, 0);
    defer plan.deinit(std.testing.allocator);

    const got = plan.fetchFor("aaa").?;
    try std.testing.expectEqual(@as(usize, 1), got.len);
    try std.testing.expectEqualStrings("https://example.com/aaa", got[0].url);
    try std.testing.expectEqual(@as(u64, 1024), got[0].url_range.end);
    try std.testing.expect(plan.fetchFor("missing") == null);
}

test "Plan.fromParsed: builds prefix and owns arena" {
    const json_str =
        \\{
        \\  "offset_into_first_range": 5,
        \\  "terms": [
        \\    {"hash": "aaa", "unpacked_length": 20, "range": {"start": 0, "end": 1}},
        \\    {"hash": "bbb", "unpacked_length": 30, "range": {"start": 0, "end": 2}}
        \\  ]
        \\}
    ;
    const parsed = try std.json.parseFromSlice(ReconstructionResponse, std.testing.allocator, json_str, .{
        .ignore_unknown_fields = true,
        .allocate = .alloc_always,
    });
    var plan = try Plan.fromParsed(std.testing.allocator, parsed);
    defer plan.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 45), plan.fileSize());
    try std.testing.expectEqual(Plan.Located{ .term_index = 0, .intra_term_offset = 5 }, try plan.locate(0));
    try std.testing.expectEqual(Plan.Located{ .term_index = 1, .intra_term_offset = 0 }, try plan.locate(15));
}

// ── ChunkRing tests ─────────────────────────────────────────────────────────

const TestXorb = struct {
    hash: []const u8,
    blob: []const u8,
    chunk_offsets: []const usize,
};

const TestFetcher = struct {
    xorbs: []const TestXorb,
    call_count: u32 = 0,

    fn fetch(ctx: *anyopaque, hash: []const u8, chunk_index: u32, out: []u8) anyerror!usize {
        const self: *TestFetcher = @ptrCast(@alignCast(ctx));
        self.call_count += 1;
        for (self.xorbs) |x| {
            if (std.mem.eql(u8, x.hash, hash)) {
                const start = x.chunk_offsets[chunk_index];
                const n = x.blob.len - start;
                @memcpy(out[0..n], x.blob[start..]);
                return n;
            }
        }
        return error.UnknownXorb;
    }

    fn fetcher(self: *TestFetcher) ChunkRing.Fetcher {
        return .{ .ctx = self, .fetchFn = fetch };
    }
};

/// Build a synthetic xorb of N type-0 (None) chunks with the given payloads.
/// Writes chunk-header byte offsets into `offsets_out`. Returns the populated
/// slice of `blob_buf`.
fn buildTestXorb(blob_buf: []u8, payloads: []const []const u8, offsets_out: []usize) []const u8 {
    var off: usize = 0;
    for (payloads, 0..) |p, i| {
        offsets_out[i] = off;
        writeChunkHeader(blob_buf[off..][0..8], @intCast(p.len), 0, @intCast(p.len));
        off += 8;
        @memcpy(blob_buf[off..][0..p.len], p);
        off += p.len;
    }
    return blob_buf[0..off];
}

test "ChunkRing: hit returns cached bytes without re-fetch" {
    var blob_buf: [128]u8 = undefined;
    var offsets: [2]usize = undefined;
    const blob = buildTestXorb(&blob_buf, &.{ &.{ 1, 2, 3, 4 }, &.{ 5, 6, 7, 8 } }, &offsets);

    var f: TestFetcher = .{ .xorbs = &.{.{ .hash = "A", .blob = blob, .chunk_offsets = &offsets }} };
    var ring: ChunkRing = .init();
    var scratch: [128]u8 = undefined;

    const a = try ring.get("A", 0, f.fetcher(), &scratch);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, a);
    try std.testing.expectEqual(@as(u32, 1), f.call_count);

    const b = try ring.get("A", 0, f.fetcher(), &scratch);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, b);
    try std.testing.expectEqual(@as(u32, 1), f.call_count);
}

test "ChunkRing: distinct chunks of same xorb each trigger one fetch" {
    var blob_buf: [128]u8 = undefined;
    var offsets: [2]usize = undefined;
    const blob = buildTestXorb(&blob_buf, &.{ &.{ 1, 2, 3, 4 }, &.{ 5, 6, 7, 8 } }, &offsets);

    var f: TestFetcher = .{ .xorbs = &.{.{ .hash = "A", .blob = blob, .chunk_offsets = &offsets }} };
    var ring: ChunkRing = .init();
    var scratch: [128]u8 = undefined;

    const a0 = try ring.get("A", 0, f.fetcher(), &scratch);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, a0);
    const a1 = try ring.get("A", 1, f.fetcher(), &scratch);
    try std.testing.expectEqualSlices(u8, &.{ 5, 6, 7, 8 }, a1);
    try std.testing.expectEqual(@as(u32, 2), f.call_count);

    // Both still cached.
    _ = try ring.get("A", 0, f.fetcher(), &scratch);
    _ = try ring.get("A", 1, f.fetcher(), &scratch);
    try std.testing.expectEqual(@as(u32, 2), f.call_count);
}

test "ChunkRing: two xorbs interleaved within capacity, one fetch per (xorb, chunk)" {
    var blob_a: [128]u8 = undefined;
    var off_a: [1]usize = undefined;
    const ba = buildTestXorb(&blob_a, &.{&.{ 0xAA, 0xAA, 0xAA }}, &off_a);

    var blob_b: [128]u8 = undefined;
    var off_b: [1]usize = undefined;
    const bb = buildTestXorb(&blob_b, &.{&.{ 0xBB, 0xBB, 0xBB }}, &off_b);

    var f: TestFetcher = .{ .xorbs = &.{
        .{ .hash = "A", .blob = ba, .chunk_offsets = &off_a },
        .{ .hash = "B", .blob = bb, .chunk_offsets = &off_b },
    } };
    var ring: ChunkRing = .init();
    var scratch: [128]u8 = undefined;

    _ = try ring.get("A", 0, f.fetcher(), &scratch);
    _ = try ring.get("B", 0, f.fetcher(), &scratch);
    _ = try ring.get("A", 0, f.fetcher(), &scratch);
    const last = try ring.get("B", 0, f.fetcher(), &scratch);
    try std.testing.expectEqualSlices(u8, &.{ 0xBB, 0xBB, 0xBB }, last);
    try std.testing.expectEqual(@as(u32, 2), f.call_count);
}

test "ChunkRing: eviction past capacity re-fetches the victim" {
    // 5 distinct chunks, ring holds 4 → first one is evicted by the 5th.
    var blob_buf: [256]u8 = undefined;
    var offsets: [5]usize = undefined;
    const blob = buildTestXorb(&blob_buf, &.{
        &.{0x10}, &.{0x20}, &.{0x30}, &.{0x40}, &.{0x50},
    }, &offsets);

    var f: TestFetcher = .{ .xorbs = &.{.{ .hash = "A", .blob = blob, .chunk_offsets = &offsets }} };
    var ring: ChunkRing = .init();
    var scratch: [256]u8 = undefined;

    for (0..5) |i| _ = try ring.get("A", @intCast(i), f.fetcher(), &scratch);
    try std.testing.expectEqual(@as(u32, 5), f.call_count);

    // Chunks 1..4 still cached (re-reading them stays at 5 fetches).
    for (1..5) |i| _ = try ring.get("A", @intCast(i), f.fetcher(), &scratch);
    try std.testing.expectEqual(@as(u32, 5), f.call_count);

    // Chunk 0 was the round-robin victim of the 5th insert → miss now.
    _ = try ring.get("A", 0, f.fetcher(), &scratch);
    try std.testing.expectEqual(@as(u32, 6), f.call_count);
}

test "WindowCache: parallel acquire/release stays consistent" {
    // Sized so 8 threads contend heavily: only ~3 of the 6 keys fit at once
    // (avg ~3 KiB per entry, 16 KiB cap). Verifies that:
    //   - no torn write is ever observed (writer-while-pinned safety),
    //   - acquire/release pin counts balance out (no leaked pins),
    //   - used_bytes accounting stays consistent under concurrency.
    const Worker = struct {
        cache: *State.WindowCache,
        io: std.Io,
        seed: u64,
        iters: u32,
        hits: std.atomic.Value(u32) = .init(0),
        misses: std.atomic.Value(u32) = .init(0),
        full: std.atomic.Value(u32) = .init(0),
        corrupt: std.atomic.Value(u32) = .init(0),

        fn run(self: *@This()) void {
            const keys = [_][]const u8{ "A", "B", "C", "D", "E", "F" };
            var rng: std.Random.DefaultPrng = .init(self.seed);
            var i: u32 = 0;
            while (i < self.iters) : (i += 1) {
                const k = keys[rng.random().uintLessThan(usize, keys.len)];
                const len: usize = 1024 + rng.random().uintLessThan(usize, 4096);
                const acq = self.cache.acquire(self.io, k, 0, len - 1, len) catch {
                    _ = self.full.fetchAdd(1, .monotonic);
                    continue;
                };
                if (acq.hit) {
                    for (acq.slice, 0..) |b, j| {
                        const want: u8 = @truncate(j ^ @as(usize, k[0]));
                        if (b != want) {
                            _ = self.corrupt.fetchAdd(1, .monotonic);
                            break;
                        }
                    }
                    _ = self.hits.fetchAdd(1, .monotonic);
                } else {
                    for (acq.slice, 0..) |*b, j| b.* = @truncate(j ^ @as(usize, k[0]));
                    _ = self.misses.fetchAdd(1, .monotonic);
                }
                self.cache.release(self.io, acq.index);
            }
        }
    };

    var threaded: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var cache: State.WindowCache = .{ .allocator = std.testing.allocator, .max_bytes = 16 * 1024 };
    defer cache.deinit();

    const num_workers = 8;
    const iters_per_worker: u32 = 2000;
    var workers: [num_workers]Worker = undefined;
    for (&workers, 0..) |*w, i| w.* = .{ .cache = &cache, .io = io, .seed = @intCast(i + 1), .iters = iters_per_worker };

    var threads: [num_workers]std.Thread = undefined;
    for (&threads, &workers) |*t, *w| t.* = try std.Thread.spawn(.{}, Worker.run, .{w});
    for (&threads) |t| t.join();

    var total_hits: u32 = 0;
    var total_misses: u32 = 0;
    var total_full: u32 = 0;
    var total_corrupt: u32 = 0;
    for (&workers) |*w| {
        total_hits += w.hits.load(.monotonic);
        total_misses += w.misses.load(.monotonic);
        total_full += w.full.load(.monotonic);
        total_corrupt += w.corrupt.load(.monotonic);
    }
    try std.testing.expectEqual(@as(u32, 0), total_corrupt);
    try std.testing.expectEqual(@as(u32, num_workers * iters_per_worker), total_hits + total_misses + total_full);
    try std.testing.expect(total_hits > 0); // contention path actually exercised

    var live_bytes: u64 = 0;
    for (cache.entries.items) |e| {
        try std.testing.expectEqual(@as(u32, 0), e.pin);
        if (e.valid) live_bytes += e.bytes.len;
    }
    try std.testing.expectEqual(live_bytes, cache.used_bytes);
}
