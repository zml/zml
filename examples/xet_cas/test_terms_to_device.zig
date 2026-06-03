// Standalone "term-routed xorb → device buffer" smoke test.
//
// Given a CAS reconstruction JSON and a local xorb byte stream covering one
// fetch_info entry of a target xorb, this binary:
//   1. selects every term that references TARGET_HASH and whose chunk range
//      [range.start, range.end) is fully contained in the matching fetch
//      entry's range,
//   2. decompresses the xorb's chunks into an in-memory host reference at the
//      per-term file offsets (shifted by coverage_min),
//   3. pushes only the "wanted" chunks to a single device buffer via
//      transferData(offset) at the same shifted offsets,
//   4. reads back and compares per-plan slices.
//
// Static BSS only (apart from JSON / CLI which use init.gpa).

const std = @import("std");
const zml = @import("zml");
const pjrt = @import("pjrt");
const pjrtx = zml.pjrtx;
const xet = @import("io").xet;

const log = std.log.scoped(.test_terms_to_device);

pub const std_options: std.Options = .{ .log_level = .info };

// Fixed-size BSS buffers — no runtime allocations from this file.
const MAX_XORB: usize = 1 * 1024 * 1024; // 1 MiB compressed
const MAX_UNCOMP: usize = 8 * 1024 * 1024; // 8 MiB decompressed (coverage window)
const MAX_CHUNK: usize = 128 * 1024; // xorb chunk uncompressed cap
const MAX_PLANS: usize = 64;

var xorb_buf: [MAX_XORB]u8 = undefined;
var host_ref: [MAX_UNCOMP]u8 = undefined;
var readback: [MAX_UNCOMP]u8 = undefined;
var dec_tmp: [MAX_CHUNK]u8 = undefined; // scratch for compression types 2/3
var dec_sink: [MAX_CHUNK]u8 = undefined; // scratch for unwanted chunks

const TermPlan = struct {
    file_offset: u64, // absolute file byte offset (prefix sum of unpacked_length)
    unpacked_length: u32,
    byte_skip: u32, // head bytes to drop (only term[0] may be non-zero)
    chunk_start: u32, // inclusive
    chunk_end: u32, // exclusive
    bytes_written: u32, // running counter during chunk walks
};

var plans: [MAX_PLANS]TermPlan = undefined;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var json_path: []const u8 = "";
    var xorb_path: []const u8 = "";
    var xorb_hash: []const u8 = "";
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--json")) {
            json_path = args_it.next() orelse return error.MissingJsonArg;
        } else if (std.mem.eql(u8, arg, "--xorb")) {
            xorb_path = args_it.next() orelse return error.MissingXorbArg;
        } else if (std.mem.eql(u8, arg, "--xorb-hash")) {
            xorb_hash = args_it.next() orelse return error.MissingXorbHashArg;
        }
    }
    if (json_path.len == 0 or xorb_path.len == 0 or xorb_hash.len == 0) {
        std.debug.print("Usage: test_terms_to_device --json <path> --xorb <path> --xorb-hash <hex>\n", .{});
        std.process.exit(1);
    }

    // ── 1. Load xorb bytes into BSS via std.Io.Reader ───────────────────────
    var xfile = try std.Io.Dir.openFileAbsolute(io, xorb_path, .{ .mode = .read_only });
    defer xfile.close(io);
    const xorb_len = try xfile.length(io);
    if (xorb_len > xorb_buf.len) return error.XorbTooLarge;
    var xreader = xfile.reader(io, &.{});
    try xreader.interface.readSliceAll(xorb_buf[0..xorb_len]);
    log.info("loaded {d} bytes from {s}", .{ xorb_len, xorb_path });

    // ── 2. Parse JSON (allocator allowed) ───────────────────────────────────
    var jfile = try std.Io.Dir.openFileAbsolute(io, json_path, .{ .mode = .read_only });
    defer jfile.close(io);
    const jlen = try jfile.length(io);
    var jreader = jfile.reader(io, &.{});
    const json_bytes = try jreader.interface.readAlloc(allocator, jlen);
    defer allocator.free(json_bytes);

    const parsed = try std.json.parseFromSlice(
        xet.ReconstructionResponse,
        allocator,
        json_bytes,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const resp = parsed.value;

    // ── 3. Find fetch_info entry for TARGET whose size matches our xorb ─────
    const fetch_list = resp.fetch_info.map.get(xorb_hash) orelse return error.TargetXorbNotInFetchInfo;
    var picked: ?xet.FetchUrl = null;
    for (fetch_list) |f| {
        const size = f.url_range.end - f.url_range.start;
        if (size == xorb_len or size + 1 == xorb_len or size == xorb_len + 1) {
            picked = f;
            break;
        }
    }
    const fetch = picked orelse return error.NoMatchingFetchEntry;
    log.info("matched fetch entry: chunks [{d},{d}) url_range size={d}", .{
        fetch.range.start,
        fetch.range.end,
        fetch.url_range.end - fetch.url_range.start,
    });

    // ── 4. Build plans: terms whose chunk range ⊆ fetch range, hash=target ──
    var n_plans: u32 = 0;
    var file_pos: u64 = 0;
    for (resp.terms, 0..) |t, i| {
        const t_off = file_pos;
        file_pos += t.unpacked_length;
        if (!std.mem.eql(u8, t.hash, xorb_hash)) continue;
        if (t.range.start < fetch.range.start or t.range.end > fetch.range.end) continue;
        if (n_plans >= plans.len) return error.TooManyPlans;
        plans[n_plans] = .{
            .file_offset = t_off,
            .unpacked_length = @intCast(t.unpacked_length),
            .byte_skip = if (i == 0) @intCast(resp.offset_into_first_range) else 0,
            .chunk_start = @intCast(t.range.start),
            .chunk_end = @intCast(t.range.end),
            .bytes_written = 0,
        };
        n_plans += 1;
    }
    if (n_plans == 0) return error.NoTermsSelected;
    log.info("selected {d} term(s) referencing TARGET", .{n_plans});

    // ── 5. Coverage window for the device buffer ────────────────────────────
    var cov_min: u64 = std.math.maxInt(u64);
    var cov_max: u64 = 0;
    for (plans[0..n_plans]) |p| {
        if (p.file_offset < cov_min) cov_min = p.file_offset;
        const e = p.file_offset + p.unpacked_length;
        if (e > cov_max) cov_max = e;
    }
    const dev_size: usize = @intCast(cov_max - cov_min);
    if (dev_size > host_ref.len) return error.CoverageTooLarge;
    log.info("coverage [{d},{d}) → device buffer size {d}", .{ cov_min, cov_max, dev_size });

    // ── 6. Pass 1: decompress every chunk; route wanted ones into host_ref ──
    var it: xet.ChunkIterator = .{ .data = xorb_buf[0..xorb_len] };
    var chunk_idx: u32 = @intCast(fetch.range.start);
    var n_chunks: u32 = 0;
    while (try it.next()) |chunk| {
        const p_idx = findPlan(plans[0..n_plans], chunk_idx);
        if (p_idx) |pi| {
            const p = &plans[pi];
            const dst_off: usize = @intCast(p.file_offset - cov_min);
            const slot_off = dst_off + p.bytes_written;
            // Head-skip only for the very first chunk of a term[0] selection.
            if (p.byte_skip > 0 and p.bytes_written == 0) {
                _ = try xet.decompressChunk(chunk, dec_sink[0..chunk.uncompressed_size], &dec_tmp);
                const keep = chunk.uncompressed_size - p.byte_skip;
                @memcpy(host_ref[slot_off .. slot_off + keep], dec_sink[p.byte_skip .. p.byte_skip + keep]);
                p.bytes_written += @intCast(keep);
            } else {
                _ = try xet.decompressChunk(chunk, host_ref[slot_off .. slot_off + chunk.uncompressed_size], &dec_tmp);
                p.bytes_written += @intCast(chunk.uncompressed_size);
            }
        } else {
            _ = try xet.decompressChunk(chunk, dec_sink[0..chunk.uncompressed_size], &dec_tmp);
        }
        chunk_idx += 1;
        n_chunks += 1;
    }
    log.info("pass1: walked {d} chunks", .{n_chunks});
    for (plans[0..n_plans]) |p| {
        if (p.bytes_written != p.unpacked_length) {
            log.err("plan mismatch: written={d} expected={d}", .{ p.bytes_written, p.unpacked_length });
            return error.PlanByteCountMismatch;
        }
    }

    // ── 7. Boot platform & create device buffer ─────────────────────────────
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    if (platform.devices.len == 0) return error.NoDevices;
    const memory = platform.devices[0].memory(.default);

    const dims: [1]i64 = .{@intCast(dev_size)};
    const shape_spec: pjrt.ShapeSpec = .init(&dims, pjrtx.bufferTypeFromDtype(.u8));
    const tm = try platform.pjrt_client.createBuffersForAsyncHostToDevice(
        platform.pjrt_api,
        .{ .shape_specs = &.{shape_spec}, .memory = memory.pjrt_memory },
    );
    defer tm.deinit(platform.pjrt_api);
    const device_buffer = try tm.retrieveBuffer(platform.pjrt_api, 0);

    // ── 8. Pass 2: push wanted chunks at shifted device offsets ─────────────
    //      One-step look-back so the final wanted push is marked is_last=true.
    for (plans[0..n_plans]) |*p| p.bytes_written = 0;
    it = .{ .data = xorb_buf[0..xorb_len] };
    chunk_idx = @intCast(fetch.range.start);

    var have_pending = false;
    var pending_off: usize = 0;
    var pending_len: usize = 0;

    while (try it.next()) |chunk| {
        const p_idx = findPlan(plans[0..n_plans], chunk_idx);
        chunk_idx += 1;
        if (p_idx == null) continue;
        const p = &plans[p_idx.?];
        const dst_off: usize = @intCast(p.file_offset - cov_min);
        const slot_off = dst_off + p.bytes_written;
        const slot_len: usize = if (p.byte_skip > 0 and p.bytes_written == 0)
            chunk.uncompressed_size - p.byte_skip
        else
            chunk.uncompressed_size;

        if (have_pending) {
            const ev = try tm.transferData(
                platform.pjrt_api,
                0,
                host_ref[pending_off .. pending_off + pending_len],
                @intCast(pending_off),
                false,
            );
            defer ev.deinit(platform.pjrt_api);
            try ev.await(platform.pjrt_api, io);
        }
        pending_off = slot_off;
        pending_len = slot_len;
        have_pending = true;
        p.bytes_written += @intCast(slot_len);
    }
    if (!have_pending) return error.NoChunksPushed;
    {
        const ev = try tm.transferData(
            platform.pjrt_api,
            0,
            host_ref[pending_off .. pending_off + pending_len],
            @intCast(pending_off),
            true,
        );
        defer ev.deinit(platform.pjrt_api);
        try ev.await(platform.pjrt_api, io);
    }

    // ── 9. Read back & compare per-plan slices ──────────────────────────────
    if (try device_buffer.toHostBuffer(platform.pjrt_api, readback[0..dev_size])) |ev| {
        defer ev.deinit(platform.pjrt_api);
        try ev.await(platform.pjrt_api, io);
    }

    var total_checked: usize = 0;
    for (plans[0..n_plans], 0..) |p, pi| {
        const dst_off: usize = @intCast(p.file_offset - cov_min);
        const len: usize = p.bytes_written;
        if (!std.mem.eql(u8, host_ref[dst_off .. dst_off + len], readback[dst_off .. dst_off + len])) {
            for (host_ref[dst_off .. dst_off + len], readback[dst_off .. dst_off + len], 0..) |a, b, j| {
                if (a != b) {
                    log.err("plan {d} mismatch at byte {d}: host=0x{x:0>2} dev=0x{x:0>2}", .{ pi, j, a, b });
                    break;
                }
            }
            return error.ReadbackMismatch;
        }
        total_checked += len;
    }

    log.info("OK: {d} plan(s), {d} bytes verified via term-routed transferData", .{ n_plans, total_checked });
}

fn findPlan(ps: []const TermPlan, chunk_idx: u32) ?u32 {
    for (ps, 0..) |p, i| {
        if (chunk_idx >= p.chunk_start and chunk_idx < p.chunk_end) return @intCast(i);
    }
    return null;
}
