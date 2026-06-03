// Standalone "xorb → device buffer" smoke test.
//
// Reads /tmp/xorb_test.bin (raw xorb byte stream — concat of complete chunks),
// decompresses each chunk in-place into a host reference buffer at a running
// offset, and pushes that same slice to a single device buffer via
// transferData(offset). Reads back and asserts byte-for-byte equality.
//
// No CAS JSON, no term routing, no HTTP. Validates that real xorb chunks flow
// correctly through the push-at-offset PJRT path.

const std = @import("std");
const zml = @import("zml");
const pjrt = @import("pjrt");
const pjrtx = zml.pjrtx;
const xet = @import("io").xet;

const log = std.log.scoped(.test_xorb_to_device);

pub const std_options: std.Options = .{ .log_level = .info };

const XORB_PATH = "/tmp/xorb_test.bin";

// Fixed-size BSS buffers — no runtime allocations from this file.
const MAX_XORB: usize = 1 * 1024 * 1024; // 1 MiB compressed (file is ~545 KiB)
const MAX_UNCOMP: usize = 8 * 1024 * 1024; // 8 MiB decompressed
const MAX_CHUNK: usize = 128 * 1024; // xorb chunk uncompressed cap

var xorb_buf: [MAX_XORB]u8 = undefined;
var host_ref: [MAX_UNCOMP]u8 = undefined;
var readback: [MAX_UNCOMP]u8 = undefined;
var dec_tmp: [MAX_CHUNK]u8 = undefined; // scratch for compression types 2/3

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // ── 1. Load xorb into fixed buffer via std.Io.Reader ────────────────────
    var file = try std.Io.Dir.openFileAbsolute(io, XORB_PATH, .{ .mode = .read_only });
    defer file.close(io);
    const file_len = try file.length(io);
    if (file_len > xorb_buf.len) return error.XorbTooLarge;
    var freader = file.reader(io, &.{});
    try freader.interface.readSliceAll(xorb_buf[0..file_len]);
    log.info("loaded {d} bytes from {s}", .{ file_len, XORB_PATH });

    // ── 2. First pass: decompress every chunk in-place into host_ref ────────
    //      Records total uncompressed length N. Also doubles as the reference
    //      we'll compare device readback against.
    var it: xet.ChunkIterator = .{ .data = xorb_buf[0..file_len] };
    var total: usize = 0;
    var n_chunks: u32 = 0;
    while (try it.next()) |chunk| {
        const need = total + chunk.uncompressed_size;
        if (need > host_ref.len) return error.HostRefTooSmall;
        _ = try xet.decompressChunk(chunk, host_ref[total..need], &dec_tmp);
        total = need;
        n_chunks += 1;
    }
    log.info("decompressed {d} chunks → {d} bytes", .{ n_chunks, total });

    // ── 3. Boot platform, create one device buffer of [total]u8 ─────────────
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    if (platform.devices.len == 0) return error.NoDevices;
    const memory = platform.devices[0].memory(.default);

    const dims: [1]i64 = .{@intCast(total)};
    const shape_spec: pjrt.ShapeSpec = .init(&dims, pjrtx.bufferTypeFromDtype(.u8));
    const tm = try platform.pjrt_client.createBuffersForAsyncHostToDevice(
        platform.pjrt_api,
        .{ .shape_specs = &.{shape_spec}, .memory = memory.pjrt_memory },
    );
    defer tm.deinit(platform.pjrt_api);
    const device_buffer = try tm.retrieveBuffer(platform.pjrt_api, 0);

    // ── 4. Second pass: walk chunks again, push each at its running offset ──
    //      One-chunk look-back so the final transferData uses is_last=true.
    it = .{ .data = xorb_buf[0..file_len] };
    var off: usize = 0;
    var pending_off: usize = 0;
    var pending_len: usize = 0;
    var have_pending = false;
    while (try it.next()) |chunk| {
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
        pending_off = off;
        pending_len = chunk.uncompressed_size;
        have_pending = true;
        off += chunk.uncompressed_size;
    }
    if (!have_pending) return error.EmptyXorb;
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

    // ── 5. Read device buffer back into fixed readback buffer, compare ──────
    if (try device_buffer.toHostBuffer(platform.pjrt_api, readback[0..total])) |ev| {
        defer ev.deinit(platform.pjrt_api);
        try ev.await(platform.pjrt_api, io);
    }

    if (!std.mem.eql(u8, host_ref[0..total], readback[0..total])) {
        for (host_ref[0..total], readback[0..total], 0..) |a, b, i| {
            if (a != b) {
                log.err("mismatch at byte {d}: host=0x{x:0>2} dev=0x{x:0>2}", .{ i, a, b });
                break;
            }
        }
        return error.ReadbackMismatch;
    }

    log.info("OK: {d} bytes ({d} chunks) round-tripped via per-chunk transferData", .{ total, n_chunks });
}
