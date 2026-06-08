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

const TensorEntry = struct {
    name: []const u8,
    offset: u64,
    size: u64,
};

const TxRange = struct {
    tensor_idx: u16,
    xorb_idx: u16,
    chunk_start: u32,
    chunk_end: u32, // exclusive
};

const XorbStats = struct {
    hash: []const u8,
    tensors: std.AutoArrayHashMapUnmanaged(u16, void) = .empty,
    needed_chunk_start: u32 = std.math.maxInt(u32),
    needed_chunk_end: u32 = 0, // exclusive
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
    const repo_dir = try zml.safetensors.resolveModelRepo(io, model_arg);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo_dir);
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

    for (files.items) |file_uri| {
        scanFile(allocator, &xet_state, &registry, file_uri) catch |e| {
            log.err("file {s}: {s}", .{ file_uri, @errorName(e) });
        };
    }
}

fn scanFile(
    allocator: std.mem.Allocator,
    xet_state: *xet.State,
    registry: *const zml.safetensors.TensorRegistry,
    file_uri: []const u8,
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

    var xorb_idx_of = std.StringHashMap(u16).init(allocator);
    defer xorb_idx_of.deinit();
    var xorbs = std.ArrayList(XorbStats).empty;
    defer {
        for (xorbs.items) |*x| x.tensors.deinit(allocator);
        xorbs.deinit(allocator);
    }
    const term_xorb = try allocator.alloc(u16, resp.terms.len);
    defer allocator.free(term_xorb);
    for (resp.terms, 0..) |t, i| {
        const gop = try xorb_idx_of.getOrPut(t.hash);
        if (!gop.found_existing) {
            if (xorbs.items.len >= std.math.maxInt(u16)) return error.TooManyXorbs;
            gop.value_ptr.* = @intCast(xorbs.items.len);
            try xorbs.append(allocator, .{ .hash = t.hash });
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

    // tx_ranges: per (tensor, xorb) the [chunk_start, chunk_end) range that
    // tensor alone would have to fetch from that xorb. Used for naive bytes.
    var tx_ranges = std.ArrayList(TxRange).empty;
    defer tx_ranges.deinit(allocator);

    var chunks_referenced_total: u64 = 0;
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

        // tx_local: aggregate per-xorb chunk ranges for this tensor.
        var tx_local: std.AutoArrayHashMapUnmanaged(u16, struct { s: u32, e: u32 }) = .empty;
        defer tx_local.deinit(allocator);

        var ti: usize = 0;
        // Skip leading terms that end before the window starts.
        while (ti < resp.terms.len and term_offs[ti + 1] <= win_start) : (ti += 1) {}
        while (ti < resp.terms.len and term_offs[ti] < win_end) : (ti += 1) {
            const t = resp.terms[ti];
            const xi = term_xorb[ti];
            const cs: u32 = @intCast(t.range.start);
            const ce: u32 = @intCast(t.range.end);

            // Mark tensor on this xorb.
            _ = try xorbs.items[xi].tensors.getOrPut(allocator, tidx);
            // Record this xorb's globally-needed chunk range.
            if (cs < xorbs.items[xi].needed_chunk_start) xorbs.items[xi].needed_chunk_start = cs;
            if (ce > xorbs.items[xi].needed_chunk_end) xorbs.items[xi].needed_chunk_end = ce;

            // For each chunk in this term, record that tensor `tidx` referenced it.
            var ci: u32 = cs;
            while (ci < ce) : (ci += 1) {
                chunks_referenced_total += 1;
                const key: u64 = (@as(u64, xi) << 32) | ci;
                const cr_gop = try chunk_refs.getOrPut(key);
                if (!cr_gop.found_existing) cr_gop.value_ptr.* = .empty;
                _ = try cr_gop.value_ptr.getOrPut(allocator, tidx);
            }

            // Update per-tensor per-xorb chunk range (for naive bytes).
            const gop = try tx_local.getOrPut(allocator, xi);
            if (!gop.found_existing) {
                gop.value_ptr.* = .{ .s = cs, .e = ce };
            } else {
                if (cs < gop.value_ptr.s) gop.value_ptr.s = cs;
                if (ce > gop.value_ptr.e) gop.value_ptr.e = ce;
            }
        }

        // Flush per-tensor ranges.
        var lit = tx_local.iterator();
        while (lit.next()) |e| {
            try tx_ranges.append(allocator, .{
                .tensor_idx = tidx,
                .xorb_idx = e.key_ptr.*,
                .chunk_start = e.value_ptr.s,
                .chunk_end = e.value_ptr.e,
            });
        }
    }

    // ── Compute fetch bytes ────────────────────────────────────────────────
    // Minimal (whole-file): union of FetchUrl byte spans covering the global
    // needed chunk range of each unique xorb.
    var minimal_fetch_bytes: u64 = 0;
    for (xorbs.items) |x| {
        if (x.needed_chunk_end == 0) continue; // unused (shouldn't happen)
        const fetch_list = resp.fetch_info.map.get(x.hash) orelse continue;
        for (fetch_list) |f| {
            if (f.range.end <= x.needed_chunk_start) continue;
            if (f.range.start >= x.needed_chunk_end) continue;
            minimal_fetch_bytes += f.url_range.end - f.url_range.start + 1;
        }
    }

    // Naive (per-tensor): each tensor independently fetches its needed
    // FetchUrl spans, with no cross-tensor sharing.
    var naive_fetch_bytes: u64 = 0;
    for (tx_ranges.items) |r| {
        const x = xorbs.items[r.xorb_idx];
        const fetch_list = resp.fetch_info.map.get(x.hash) orelse continue;
        for (fetch_list) |f| {
            if (f.range.end <= r.chunk_start) continue;
            if (f.range.start >= r.chunk_end) continue;
            naive_fetch_bytes += f.url_range.end - f.url_range.start + 1;
        }
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
