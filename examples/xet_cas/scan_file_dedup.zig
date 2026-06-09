// Static dedup / reuse analysis for a single safetensors shard.
//
// Asks the HF CAS for ONE whole-file reconstruction, then computes:
//   • file-level totals (terms, unique xorbs, unique chunks)
//   • per-file chunk reuse: how many times each (xorb, chunk) position is
//     cited by the reconstruction plan's terms (= byte-stream repetition
//     within the file, independent of tensor accounting)
//   • cross-tensor sharing (xorbs and (xorb, chunk) pairs referenced by ≥2
//     tensors of the file)
//   • bytes-on-wire comparison:
//        - minimal (whole-file): union of FetchUrl byte spans covering all
//          referenced chunks, per unique xorb
//        - naive (per-tensor):   sum over tensors of the FetchUrl byte spans
//          each tensor alone would need
//
// Usage:
//   scan_file_dedup --model hf://meta-llama/Meta-Llama-3-70B \
//                   --file model-00030-of-00030.safetensors
//   scan_file_dedup --model hf://...  # scans every shard in the repo

const std = @import("std");
const zml = @import("zml");
const xet = @import("io").xet;

const log = std.log.scoped(.scan_file_dedup);

pub const std_options: std.Options = .{ .log_level = .info };

const ChunkSet = std.bit_set.DynamicBitSetUnmanaged;

const TensorEntry = struct {
    name: []const u8,
    offset: u64,
    size: u64,
};

const XorbStats = struct {
    hash: []const u8,
    tensors: std.AutoArrayHashMapUnmanaged(u16, void) = .empty,
    // Exact set of chunks of this xorb referenced by the file's reconstruction.
    needed: ChunkSet = .{},
};

// Slim FetchUrl copy for cross-shard accounting. The original FetchUrl borrows
// a URL slice from the parsed response, which is freed at the end of scanFile;
// the aggregator only needs chunk ranges + byte sizes.
const FetchSpan = struct {
    chunk_start: u32,
    chunk_end: u32, // exclusive
    bytes: u64,
};

const GlobalXorb = struct {
    fetch_spans: []const FetchSpan, // arena-owned
    needed: ChunkSet = .{},
    // Per-chunk count of how many shards reference this chunk of this xorb.
    // Sized to `needed.bit_length`; gpa-owned (so it can be resized).
    chunk_shard_counts: []u16 = &.{},
    shards: u32 = 0,
};

const GlobalAgg = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    xorbs: std.StringHashMap(GlobalXorb),
    total_tensor_bytes: u64 = 0,
    total_file_size: u64 = 0,
    sum_per_shard_minimal: u64 = 0,
    sum_per_shard_naive: u64 = 0,
    shards_scanned: u32 = 0,

    fn init(allocator: std.mem.Allocator) GlobalAgg {
        return .{
            .allocator = allocator,
            .arena = std.heap.ArenaAllocator.init(allocator),
            .xorbs = std.StringHashMap(GlobalXorb).init(allocator),
        };
    }

    fn deinit(self: *GlobalAgg) void {
        var it = self.xorbs.valueIterator();
        while (it.next()) |v| {
            v.needed.deinit(self.allocator);
            self.allocator.free(v.chunk_shard_counts);
        }
        self.xorbs.deinit();
        self.arena.deinit();
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var file_arg: []const u8 = "";
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--file")) {
            file_arg = args_it.next() orelse return error.MissingFileArg;
        }
    }
    if (model_arg.len == 0) {
        std.debug.print("Usage: scan_file_dedup --model <uri> [--file <suffix>]\n", .{});
        std.process.exit(1);
    }

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    const io = vfs.io();

    log.info("Resolving model repo: {s}", .{model_arg});
    // When a single shard is requested, build the registry from just that
    // file. Using `fromRepo` would open every shard listed in the index
    // (and probe each for XET), which is wasted work for a single-file scan.
    const repo_dir = try zml.safetensors.resolveModelRepo(io, model_arg);
    var registry: zml.safetensors.TensorRegistry = if (file_arg.len != 0) blk: {
        log.info("Loading single shard: {s} (in repo {s})", .{ file_arg, model_arg });
        const shard_file = try repo_dir.openFile(io, file_arg, .{ .mode = .read_only });
        break :blk try zml.safetensors.fetchRegistry(allocator, io, repo_dir, shard_file);
    } else try zml.safetensors.TensorRegistry.fromRepo(allocator, io, repo_dir);
    defer registry.deinit();

    const hf_token = init.environ_map.get("HF_TOKEN") orelse {
        log.err("HF_TOKEN env var must be set", .{});
        return error.MissingToken;
    };

    var xet_state: xet.State = .init(allocator, &http_client, hf_token);
    defer xet_state.deinit();

    // ── Enumerate target files ─────────────────────────────────────────────
    // If --file is given, scan only the matching shard. Otherwise enumerate
    // every distinct file_uri in the registry.
    var seen_files = std.StringHashMap(void).init(allocator);
    defer seen_files.deinit();
    var files = std.ArrayList([]const u8).empty;
    defer files.deinit(allocator);
    for (registry.tensors.values()) |t| {
        if (file_arg.len != 0 and !std.mem.endsWith(u8, t.file_uri, file_arg)) continue;
        if (seen_files.contains(t.file_uri)) continue;
        try seen_files.put(t.file_uri, {});
        try files.append(allocator, t.file_uri);
    }
    if (files.items.len == 0) {
        log.err("No tensor file_uri matches --file={s}", .{file_arg});
        return error.FileNotFound;
    }
    log.info("Scanning {d} file(s)", .{files.items.len});

    // Cross-shard aggregator. Skipped for single-file scans where there is
    // nothing to aggregate.
    var maybe_agg: ?GlobalAgg = null;
    if (files.items.len > 1) maybe_agg = .init(allocator);
    defer if (maybe_agg != null) maybe_agg.?.deinit();
    const agg_ptr: ?*GlobalAgg = if (maybe_agg != null) &maybe_agg.? else null;

    for (files.items) |file_uri| {
        scanFile(allocator, &xet_state, &registry, file_uri, agg_ptr) catch |e| {
            log.err("file {s}: {s}", .{ file_uri, @errorName(e) });
        };
    }

    if (agg_ptr) |a| try printGlobalSummary(allocator, a);
}

fn scanFile(
    allocator: std.mem.Allocator,
    xet_state: *xet.State,
    registry: *const zml.safetensors.TensorRegistry,
    file_uri: []const u8,
    agg: ?*GlobalAgg,
) !void {
    // ── Collect every tensor belonging to this file ────────────────────────
    var tensors = std.ArrayList(TensorEntry).empty;
    defer tensors.deinit(allocator);
    var file_size: u64 = 0;
    var tensor_bytes_total: u64 = 0;
    for (registry.tensors.values()) |t| {
        if (!std.mem.eql(u8, t.file_uri, file_uri)) continue;
        const end = t.offset + t.byteSize();
        if (end > file_size) file_size = end;
        tensor_bytes_total += t.byteSize();
        try tensors.append(allocator, .{
            .name = t.name,
            .offset = t.offset,
            .size = t.byteSize(),
        });
    }
    if (tensors.items.len == 0) return error.NoTensorsInFile;
    if (tensors.items.len > std.math.maxInt(u16)) return error.TooManyTensors;

    // ── Reconstruct the whole file in ONE call ─────────────────────────────
    const hf_path = if (std.mem.startsWith(u8, file_uri, "hf://")) file_uri["hf://".len..] else file_uri;
    const hf_repo = try zml.io.VFS.HF.Repo.parse(hf_path);
    const repo: xet.State.Repo = .{
        .repo = hf_repo.repo,
        .model = hf_repo.model,
        .rev = hf_repo.rev,
        .path = hf_repo.path,
    };

    const parsed = try xet_state.reconstruct(repo, 0, file_size);
    defer parsed.deinit();
    const resp = parsed.value;

    // ── Build unique-xorb index and per-term stream offsets ────────────────
    const term_offs = try allocator.alloc(u64, resp.terms.len + 1);
    defer allocator.free(term_offs);
    term_offs[0] = 0;
    for (resp.terms, 0..) |t, i| term_offs[i + 1] = term_offs[i] + t.unpacked_length;

    // First pass: compute the highest chunk index seen per xorb (across terms
    // and fetch_info), so each xorb's `needed` bitset can be sized correctly.
    var xorb_chunk_count = std.StringHashMap(u32).init(allocator);
    defer xorb_chunk_count.deinit();
    for (resp.terms) |t| {
        const ce: u32 = @intCast(t.range.end);
        const gop = try xorb_chunk_count.getOrPut(t.hash);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        if (ce > gop.value_ptr.*) gop.value_ptr.* = ce;
    }
    var fi_it = resp.fetch_info.map.iterator();
    while (fi_it.next()) |e| {
        var m: u32 = 0;
        for (e.value_ptr.*) |f| {
            const fe: u32 = @intCast(f.range.end);
            if (fe > m) m = fe;
        }
        const gop = try xorb_chunk_count.getOrPut(e.key_ptr.*);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        if (m > gop.value_ptr.*) gop.value_ptr.* = m;
    }

    var xorb_idx_of = std.StringHashMap(u16).init(allocator);
    defer xorb_idx_of.deinit();
    var xorbs = std.ArrayList(XorbStats).empty;
    defer {
        for (xorbs.items) |*x| {
            x.tensors.deinit(allocator);
            x.needed.deinit(allocator);
        }
        xorbs.deinit(allocator);
    }
    const term_xorb = try allocator.alloc(u16, resp.terms.len);
    defer allocator.free(term_xorb);
    for (resp.terms, 0..) |t, i| {
        const gop = try xorb_idx_of.getOrPut(t.hash);
        if (!gop.found_existing) {
            if (xorbs.items.len >= std.math.maxInt(u16)) return error.TooManyXorbs;
            gop.value_ptr.* = @intCast(xorbs.items.len);
            const max_chunk = xorb_chunk_count.get(t.hash) orelse 0;
            try xorbs.append(allocator, .{
                .hash = t.hash,
                .needed = try ChunkSet.initEmpty(allocator, max_chunk),
            });
        }
        term_xorb[i] = gop.value_ptr.*;
    }
    const n_xorbs: u16 = @intCast(xorbs.items.len);

    // ── Walk each tensor's window and accumulate per-chunk references ──────
    // chunk_refs: (xorb_idx<<32 | chunk_idx) → set of tensor_idx
    var chunk_refs = std.AutoHashMap(u64, std.AutoArrayHashMapUnmanaged(u16, void)).init(allocator);
    defer {
        var it = chunk_refs.valueIterator();
        while (it.next()) |s| s.deinit(allocator);
        chunk_refs.deinit();
    }

    // tx_local lives per tensor (see loop below): per-xorb bitset of chunks
    // this tensor alone references. Used to compute naive (per-tensor) bytes
    // exactly.

    var chunks_referenced_total: u64 = 0;
    var naive_fetch_bytes: u64 = 0;
    const win_offset: u64 = resp.offset_into_first_range; // 0 when start==0

    // ── Per-file chunk reuse (independent of tensors) ──────────────────────
    // Count how many times each (xorb, chunk) position is cited by the
    // reconstruction plan's terms. A count >1 means the chunk's bytes appear
    // multiple times in the reconstructed file's byte stream.
    var chunk_plan_refs = std.AutoHashMap(u64, u32).init(allocator);
    defer chunk_plan_refs.deinit();
    var plan_chunks_total: u64 = 0;
    for (resp.terms, 0..) |t, i| {
        const xi = term_xorb[i];
        const cs: u32 = @intCast(t.range.start);
        const ce: u32 = @intCast(t.range.end);
        var ci: u32 = cs;
        while (ci < ce) : (ci += 1) {
            plan_chunks_total += 1;
            const key: u64 = (@as(u64, xi) << 32) | ci;
            const gop = try chunk_plan_refs.getOrPut(key);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
        }
    }

    for (tensors.items, 0..) |tn, ji| {
        const tidx: u16 = @intCast(ji);
        const win_start: u64 = win_offset + tn.offset;
        const win_end: u64 = win_start + tn.size;

        // tx_local: per-xorb bitset of chunks this tensor alone references.
        var tx_local: std.AutoArrayHashMapUnmanaged(u16, ChunkSet) = .empty;
        defer {
            for (tx_local.values()) |*bs| bs.deinit(allocator);
            tx_local.deinit(allocator);
        }

        var ti: usize = 0;
        // Skip leading terms that end before the window starts.
        while (ti < resp.terms.len and term_offs[ti + 1] <= win_start) : (ti += 1) {}
        while (ti < resp.terms.len and term_offs[ti] < win_end) : (ti += 1) {
            const t = resp.terms[ti];
            const xi = term_xorb[ti];
            const cs: u32 = @intCast(t.range.start);
            const ce: u32 = @intCast(t.range.end);

            // Mark tensor on this xorb and union the term's chunks into the
            // xorb's file-global needed-chunks bitset (the per-file plan
            // genuinely needs every chunk of every term it cites, including
            // bytes that are alignment padding rather than tensor payload).
            _ = try xorbs.items[xi].tensors.getOrPut(allocator, tidx);
            xorbs.items[xi].needed.setRangeValue(.{ .start = cs, .end = ce }, true);

            // Clip the term to this tensor's byte window, then translate that
            // sub-range into a sub-range of chunks. We don't have per-chunk
            // byte sizes, so assume bytes are uniformly distributed across
            // the term's chunks — accurate to a chunk near term ends.
            const term_start = term_offs[ti];
            const term_end = term_offs[ti + 1];
            const overlap_start = @max(win_start, term_start);
            const overlap_end = @min(win_end, term_end);
            if (overlap_end <= overlap_start) continue;
            const term_bytes = term_end - term_start;
            const chunk_count: u64 = ce - cs;
            if (term_bytes == 0 or chunk_count == 0) continue;
            const a = overlap_start - term_start;
            const b = overlap_end - term_start;
            const tcs: u32 = cs + @as(u32, @intCast((a * chunk_count) / term_bytes));
            const tce: u32 = cs + @as(u32, @intCast((b * chunk_count + term_bytes - 1) / term_bytes));

            var ci: u32 = tcs;
            while (ci < tce) : (ci += 1) {
                chunks_referenced_total += 1;
                const key: u64 = (@as(u64, xi) << 32) | ci;
                const cr_gop = try chunk_refs.getOrPut(key);
                if (!cr_gop.found_existing) cr_gop.value_ptr.* = .empty;
                _ = try cr_gop.value_ptr.getOrPut(allocator, tidx);
            }

            // Update per-tensor per-xorb chunk bitset (for naive bytes).
            const gop = try tx_local.getOrPut(allocator, xi);
            if (!gop.found_existing) {
                gop.value_ptr.* = try ChunkSet.initEmpty(allocator, xorbs.items[xi].needed.bit_length);
            }
            gop.value_ptr.setRangeValue(.{ .start = tcs, .end = tce }, true);
        }

        // Naive (per-tensor) fetch bytes: for each xorb this tensor touches,
        // sum FetchUrl spans that overlap at least one chunk in the bitset.
        var nit = tx_local.iterator();
        while (nit.next()) |e| {
            const fetch_list = resp.fetch_info.map.get(xorbs.items[e.key_ptr.*].hash) orelse continue;
            naive_fetch_bytes += fetchBytesForBits(fetch_list, e.value_ptr.*);
        }
    }

    // ── Compute minimal (whole-file) fetch bytes ───────────────────────────
    // For each unique xorb, sum FetchUrl spans that overlap at least one
    // chunk in the xorb's file-global needed-chunks bitset.
    var minimal_fetch_bytes: u64 = 0;
    for (xorbs.items) |x| {
        if (x.needed.count() == 0) continue;
        const fetch_list = resp.fetch_info.map.get(x.hash) orelse continue;
        minimal_fetch_bytes += fetchBytesForBits(fetch_list, x.needed);
    }

    // ── Aggregate chunk/xorb sharing stats ─────────────────────────────────
    // Histogram by exact tensor-reference count (1, 2, 3, ...).
    var chunk_hist: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
    defer chunk_hist.deinit(allocator);
    var unique_chunks: u32 = 0;
    var cr_it = chunk_refs.valueIterator();
    while (cr_it.next()) |s| {
        unique_chunks += 1;
        const n: u32 = @intCast(s.count());
        const gop = try chunk_hist.getOrPut(allocator, n);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    var xorb_hist: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
    defer xorb_hist.deinit(allocator);
    for (xorbs.items) |x| {
        const n: u32 = @intCast(x.tensors.count());
        const gop = try xorb_hist.getOrPut(allocator, n);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    // ── Print summary block ────────────────────────────────────────────────
    log.info("──────────────────────────────────────────────────────────────────", .{});
    log.info("FILE  {s}", .{file_uri});
    log.info("  tensors={d}  tensor_bytes={d:.2} MiB  file_size={d:.2} MiB", .{
        tensors.items.len, mib(tensor_bytes_total), mib(file_size),
    });
    log.info("  terms={d}  xorbs(unique)={d}", .{ resp.terms.len, n_xorbs });
    log.info("  chunks(refs)={d}  chunks(unique)={d}", .{ chunks_referenced_total, unique_chunks });

    // Per-file chunk reuse histogram (across plan terms, regardless of tensors).
    var chunk_plan_hist: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
    defer chunk_plan_hist.deinit(allocator);
    var plan_unique_chunks: u32 = 0;
    var cpr_it = chunk_plan_refs.valueIterator();
    while (cpr_it.next()) |v| {
        plan_unique_chunks += 1;
        const gop = try chunk_plan_hist.getOrPut(allocator, v.*);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }
    log.info("  chunks plan(refs)={d}  plan(unique)={d}", .{ plan_chunks_total, plan_unique_chunks });
    printHistogram("  chunks reuse (per file):  ", &chunk_plan_hist, plan_unique_chunks);
    printHistogram("  chunks reuse (per tensor):", &chunk_hist, unique_chunks);
    printHistogram("  xorbs  reuse (per tensor):", &xorb_hist, n_xorbs);
    log.info("  fetch bytes minimal (whole-file): {d:.2} MiB", .{mib(minimal_fetch_bytes)});
    log.info("  fetch bytes naive   (per-tensor): {d:.2} MiB", .{mib(naive_fetch_bytes)});
    const ratio_naive_to_min = ratio(naive_fetch_bytes, @max(minimal_fetch_bytes, 1));
    log.info("  amortization ratio (naive/minimal) = {d:.3}x", .{ratio_naive_to_min});
    const ratio_min_to_tensor = ratio(minimal_fetch_bytes, @max(tensor_bytes_total, 1));
    log.info("  on-wire / tensor-bytes (minimal): {d:.3}", .{ratio_min_to_tensor});
    log.info("  on-wire / file-size    (minimal): {d:.3}", .{ratio(minimal_fetch_bytes, @max(file_size, 1))});

    // ── Merge into cross-shard aggregator (if any) ────────────────────────
    if (agg) |a| try mergeIntoAgg(a, xorbs.items, resp.fetch_info, tensor_bytes_total, file_size, minimal_fetch_bytes, naive_fetch_bytes);
}

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

fn pct(num: u64, denom: u64) f64 {
    if (denom == 0) return 0.0;
    return 100.0 * @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(denom));
}

fn ratio(num: u64, denom: u64) f64 {
    return @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(denom));
}

fn printHistogram(
    label: []const u8,
    hist: *std.AutoArrayHashMapUnmanaged(u32, u32),
    total: u32,
) void {
    // Sort keys ascending so the output reads 1×, 2×, 3×, ...
    const keys = hist.keys();
    var sorted: [256]u32 = undefined;
    const n = @min(keys.len, sorted.len);
    @memcpy(sorted[0..n], keys[0..n]);
    std.mem.sort(u32, sorted[0..n], {}, std.sort.asc(u32));
    for (sorted[0..n]) |k| {
        const count = hist.get(k).?;
        log.info("{s} {d}x -> {d}  ({d:.2}%)", .{ label, k, count, pct(count, total) });
    }
}

fn anySetInRange(bits: ChunkSet, start: usize, end_excl: usize) bool {
    const clamped = @min(end_excl, bits.bit_length);
    if (start >= clamped) return false;
    var i = start;
    while (i < clamped) : (i += 1) {
        if (bits.isSet(i)) return true;
    }
    return false;
}

fn fetchBytesForBits(fetch_list: []const xet.FetchUrl, bits: ChunkSet) u64 {
    var total: u64 = 0;
    for (fetch_list) |f| {
        if (anySetInRange(bits, @intCast(f.range.start), @intCast(f.range.end))) {
            total += f.url_range.end - f.url_range.start + 1;
        }
    }
    return total;
}

fn fetchBytesForSpans(spans: []const FetchSpan, bits: ChunkSet) u64 {
    var total: u64 = 0;
    for (spans) |s| {
        if (anySetInRange(bits, s.chunk_start, s.chunk_end)) total += s.bytes;
    }
    return total;
}

fn mergeIntoAgg(
    agg: *GlobalAgg,
    xorbs: []const XorbStats,
    fetch_info: anytype,
    tensor_bytes: u64,
    file_bytes: u64,
    minimal_bytes: u64,
    naive_bytes: u64,
) !void {
    agg.total_tensor_bytes += tensor_bytes;
    agg.total_file_size += file_bytes;
    agg.sum_per_shard_minimal += minimal_bytes;
    agg.sum_per_shard_naive += naive_bytes;
    agg.shards_scanned += 1;

    const arena_alloc = agg.arena.allocator();

    for (xorbs) |x| {
        if (x.needed.bit_length == 0) continue;
        if (agg.xorbs.getPtr(x.hash)) |existing| {
            if (x.needed.bit_length > existing.needed.bit_length) {
                try existing.needed.resize(agg.allocator, x.needed.bit_length, false);
                const old = existing.chunk_shard_counts;
                const new = try agg.allocator.alloc(u16, x.needed.bit_length);
                @memcpy(new[0..old.len], old);
                @memset(new[old.len..], 0);
                agg.allocator.free(old);
                existing.chunk_shard_counts = new;
            }
            var sit = x.needed.iterator(.{});
            while (sit.next()) |i| {
                existing.needed.set(i);
                existing.chunk_shard_counts[i] += 1;
            }
            existing.shards += 1;
        } else {
            const fetch_list = fetch_info.map.get(x.hash) orelse continue;
            const dup_hash = try arena_alloc.dupe(u8, x.hash);
            const spans = try arena_alloc.alloc(FetchSpan, fetch_list.len);
            for (fetch_list, spans) |f, *s| {
                s.* = .{
                    .chunk_start = @intCast(f.range.start),
                    .chunk_end = @intCast(f.range.end),
                    .bytes = f.url_range.end - f.url_range.start + 1,
                };
            }
            var needed_copy = try ChunkSet.initEmpty(agg.allocator, x.needed.bit_length);
            const counts = try agg.allocator.alloc(u16, x.needed.bit_length);
            @memset(counts, 0);
            var sit = x.needed.iterator(.{});
            while (sit.next()) |i| {
                needed_copy.set(i);
                counts[i] = 1;
            }
            try agg.xorbs.put(dup_hash, .{
                .fetch_spans = spans,
                .needed = needed_copy,
                .chunk_shard_counts = counts,
                .shards = 1,
            });
        }
    }
}

fn printGlobalSummary(allocator: std.mem.Allocator, agg: *GlobalAgg) !void {
    var repo_minimal_fetch_bytes: u64 = 0;
    var xorb_shard_hist: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
    defer xorb_shard_hist.deinit(allocator);
    var chunk_shard_hist: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;
    defer chunk_shard_hist.deinit(allocator);
    var n_xorbs_global: u32 = 0;
    var unique_chunks_global: u64 = 0;
    var chunk_refs_global: u64 = 0;

    var it = agg.xorbs.valueIterator();
    while (it.next()) |x| {
        n_xorbs_global += 1;
        repo_minimal_fetch_bytes += fetchBytesForSpans(x.fetch_spans, x.needed);
        const xs_gop = try xorb_shard_hist.getOrPut(allocator, x.shards);
        if (!xs_gop.found_existing) xs_gop.value_ptr.* = 0;
        xs_gop.value_ptr.* += 1;

        for (x.chunk_shard_counts) |c| {
            if (c == 0) continue;
            unique_chunks_global += 1;
            chunk_refs_global += c;
            const cs_gop = try chunk_shard_hist.getOrPut(allocator, c);
            if (!cs_gop.found_existing) cs_gop.value_ptr.* = 0;
            cs_gop.value_ptr.* += 1;
        }
    }

    log.info("══════════════════════════════════════════════════════════════════", .{});
    log.info("REPO SUMMARY ({d} shards scanned)", .{agg.shards_scanned});
    log.info("  tensor_bytes_total={d:.2} MiB  file_size_total={d:.2} MiB", .{
        mib(agg.total_tensor_bytes), mib(agg.total_file_size),
    });
    log.info("  unique xorbs (global)={d}", .{n_xorbs_global});
    printHistogram("  xorbs reuse (cross-shard):", &xorb_shard_hist, n_xorbs_global);
    log.info("  chunks (refs across shards)={d}  chunks (unique global)={d}", .{
        chunk_refs_global, unique_chunks_global,
    });
    const chunk_redundancy = ratio(chunk_refs_global, @max(unique_chunks_global, 1));
    log.info("  chunks redundancy (refs / unique) = {d:.3}x", .{chunk_redundancy});
    const u32_unique = std.math.cast(u32, unique_chunks_global) orelse std.math.maxInt(u32);
    printHistogram("  chunks reuse (cross-shard):", &chunk_shard_hist, u32_unique);
    log.info("  fetch bytes sum per-shard minimal: {d:.2} MiB", .{mib(agg.sum_per_shard_minimal)});
    log.info("  fetch bytes global  minimal:       {d:.2} MiB", .{mib(repo_minimal_fetch_bytes)});
    const ratio_amort = ratio(agg.sum_per_shard_minimal, @max(repo_minimal_fetch_bytes, 1));
    log.info("  cross-shard amortization (per-shard-sum / global) = {d:.3}x", .{ratio_amort});
    log.info("  on-wire / tensor-bytes (global minimal): {d:.3}", .{
        ratio(repo_minimal_fetch_bytes, @max(agg.total_tensor_bytes, 1)),
    });
    log.info("  on-wire / file-size    (global minimal): {d:.3}", .{
        ratio(repo_minimal_fetch_bytes, @max(agg.total_file_size, 1)),
    });
}
