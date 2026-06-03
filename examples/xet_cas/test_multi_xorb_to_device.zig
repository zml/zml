// Standalone "multi-xorb → one device buffer" smoke test.
//
// Same shape as test_terms_to_device but accepts MULTIPLE xorbs and stitches
// their selected terms together into a shared coverage window / device buffer.
//
// Usage:
//   test_multi_xorb_to_device \
//     --json <path> \
//     --xorb <hash>=<path> [--xorb <hash>=<path> ...]
//
// For each supplied xorb:
//   1. load its bytes into a BSS slot,
//   2. auto-pick the fetch_info entry whose byte size matches file size,
//   3. select every term referencing that hash whose chunk range fits inside
//      the matching fetch entry.
// Then build the combined coverage window over ALL selected plans, route
// chunks into host_ref, push wanted chunks via transferData(offset), and
// verify per-plan slices.

const std = @import("std");
const zml = @import("zml");
const pjrt = @import("pjrt");
const pjrtx = zml.pjrtx;
const xet = @import("io").xet;

const log = std.log.scoped(.test_multi_xorb_to_device);

pub const std_options: std.Options = .{ .log_level = .info };

const MAX_XORB: usize = 1 * 1024 * 1024; // 1 MiB per xorb
const MAX_XORBS: usize = 4;
const MAX_UNCOMP: usize = 8 * 1024 * 1024; // 8 MiB coverage window
const MAX_CHUNK: usize = 128 * 1024;
const MAX_PLANS: usize = 64;

var xorb_bufs: [MAX_XORBS][MAX_XORB]u8 = undefined;
var host_ref: [MAX_UNCOMP]u8 = undefined;
var readback: [MAX_UNCOMP]u8 = undefined;
var dec_tmp: [MAX_CHUNK]u8 = undefined;
var dec_sink: [MAX_CHUNK]u8 = undefined;

const XorbSlot = struct {
    hash: []const u8, // borrowed from argv
    len: usize,
    fetch_chunk_start: u32,
    fetch_chunk_end: u32,
};

var xorbs: [MAX_XORBS]XorbSlot = undefined;

const TermPlan = struct {
    xorb_idx: u8,
    file_offset: u64,
    unpacked_length: u32,
    byte_skip: u32,
    chunk_start: u32,
    chunk_end: u32,
    bytes_written: u32,
};

var plans: [MAX_PLANS]TermPlan = undefined;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var json_path: []const u8 = "";
    var n_xorbs: u8 = 0;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--json")) {
            json_path = args_it.next() orelse return error.MissingJsonArg;
        } else if (std.mem.eql(u8, arg, "--xorb")) {
            const pair = args_it.next() orelse return error.MissingXorbArg;
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return error.XorbArgMustBeHashEqPath;
            if (n_xorbs >= MAX_XORBS) return error.TooManyXorbs;
            const hash = pair[0..eq];
            const path = pair[eq + 1 ..];
            // Load file into BSS slot.
            var f = try std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only });
            defer f.close(io);
            const flen = try f.length(io);
            if (flen > xorb_bufs[n_xorbs].len) return error.XorbTooLarge;
            var r = f.reader(io, &.{});
            try r.interface.readSliceAll(xorb_bufs[n_xorbs][0..flen]);
            xorbs[n_xorbs] = .{
                .hash = hash,
                .len = flen,
                .fetch_chunk_start = 0,
                .fetch_chunk_end = 0,
            };
            log.info("loaded xorb[{d}]: hash={s}.. {d} bytes from {s}", .{ n_xorbs, hash[0..16], flen, path });
            n_xorbs += 1;
        }
    }
    if (json_path.len == 0 or n_xorbs == 0) {
        std.debug.print("Usage: test_multi_xorb_to_device --json <path> --xorb <hash>=<path> [--xorb ...]\n", .{});
        std.process.exit(1);
    }

    // ── Parse JSON ──────────────────────────────────────────────────────────
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

    // ── Resolve fetch_info entry per supplied xorb (size-match) ─────────────
    for (xorbs[0..n_xorbs]) |*x| {
        const fetch_list = resp.fetch_info.map.get(x.hash) orelse return error.XorbNotInFetchInfo;
        var picked: ?xet.FetchUrl = null;
        for (fetch_list) |f| {
            // url_range is inclusive on both ends → byte count = end - start + 1
            const size = f.url_range.end - f.url_range.start + 1;
            if (size == x.len) {
                picked = f;
                break;
            }
        }
        const fetch = picked orelse return error.NoMatchingFetchEntry;
        x.fetch_chunk_start = @intCast(fetch.range.start);
        x.fetch_chunk_end = @intCast(fetch.range.end);
        log.info("xorb {s}..: matched fetch chunks=[{d},{d})", .{ x.hash[0..16], fetch.range.start, fetch.range.end });
    }

    // ── Build plans ─────────────────────────────────────────────────────────
    var n_plans: u32 = 0;
    var file_pos: u64 = 0;
    for (resp.terms, 0..) |t, i| {
        const t_off = file_pos;
        file_pos += t.unpacked_length;
        // Find which (if any) supplied xorb this term routes through.
        var xi: ?u8 = null;
        for (xorbs[0..n_xorbs], 0..) |x, j| {
            if (!std.mem.eql(u8, t.hash, x.hash)) continue;
            if (t.range.start < x.fetch_chunk_start or t.range.end > x.fetch_chunk_end) continue;
            xi = @intCast(j);
            break;
        }
        if (xi == null) continue;
        if (n_plans >= plans.len) return error.TooManyPlans;
        plans[n_plans] = .{
            .xorb_idx = xi.?,
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
    log.info("selected {d} term(s) across {d} xorb(s)", .{ n_plans, n_xorbs });

    // ── Coverage window ─────────────────────────────────────────────────────
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

    // ── Pass 1: decompress every chunk of every xorb; route into host_ref ──
    for (xorbs[0..n_xorbs], 0..) |x, xi| {
        var it: xet.ChunkIterator = .{ .data = xorb_bufs[xi][0..x.len] };
        var chunk_idx: u32 = x.fetch_chunk_start;
        while (try it.next()) |chunk| {
            const p_idx = findPlan(plans[0..n_plans], @intCast(xi), chunk_idx);
            if (p_idx) |pi| {
                const p = &plans[pi];
                const dst_off: usize = @intCast(p.file_offset - cov_min);
                const slot_off = dst_off + p.bytes_written;
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
        }
    }
    for (plans[0..n_plans]) |p| {
        if (p.bytes_written != p.unpacked_length) {
            log.err("plan mismatch: written={d} expected={d}", .{ p.bytes_written, p.unpacked_length });
            return error.PlanByteCountMismatch;
        }
    }

    // ── Boot platform & create device buffer ────────────────────────────────
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

    // ── Pass 2: push wanted chunks (all xorbs), look-back across xorbs ─────
    for (plans[0..n_plans]) |*p| p.bytes_written = 0;
    var have_pending = false;
    var pending_off: usize = 0;
    var pending_len: usize = 0;

    for (xorbs[0..n_xorbs], 0..) |x, xi| {
        var it: xet.ChunkIterator = .{ .data = xorb_bufs[xi][0..x.len] };
        var chunk_idx: u32 = x.fetch_chunk_start;
        while (try it.next()) |chunk| {
            const p_idx = findPlan(plans[0..n_plans], @intCast(xi), chunk_idx);
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

    // ── Readback & per-plan verification ───────────────────────────────────
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
                    log.err("plan {d} (xorb {d}) mismatch at byte {d}: host=0x{x:0>2} dev=0x{x:0>2}", .{ pi, p.xorb_idx, j, a, b });
                    break;
                }
            }
            return error.ReadbackMismatch;
        }
        total_checked += len;
    }

    log.info("OK: {d} plan(s) from {d} xorb(s), {d} bytes verified", .{ n_plans, n_xorbs, total_checked });
}

fn findPlan(ps: []const TermPlan, xorb_idx: u8, chunk_idx: u32) ?u32 {
    for (ps, 0..) |p, i| {
        if (p.xorb_idx == xorb_idx and chunk_idx >= p.chunk_start and chunk_idx < p.chunk_end) return @intCast(i);
    }
    return null;
}
