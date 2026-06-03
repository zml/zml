// Standalone PJRT "push-at-offset" smoke test.
//
// Allocates one device buffer of 1024 bytes via createBuffersForAsyncHostToDevice,
// pushes two non-adjacent halves of a fixed pattern at distinct offsets using
// transferData(offset), reads back, and asserts byte-for-byte equality.
//
// No XET, no safetensors, no networking. Validates the exact PJRT path the
// real reconstruction pipeline will use.

const std = @import("std");
const zml = @import("zml");
const pjrt = @import("pjrt");
const pjrtx = zml.pjrtx;

const log = std.log.scoped(.test_pjrt_push);

pub const std_options: std.Options = .{ .log_level = .info };

const N: usize = 1024;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("platform: {f}", .{platform.fmtVerbose()});

    if (platform.devices.len == 0) return error.NoDevices;
    const memory = platform.devices[0].memory(.default);

    // Fixed-size shape and host pattern.
    const shape: zml.Shape = .init(.{N}, .u8);
    var pattern: [N]u8 = undefined;
    for (&pattern, 0..) |*b, i| b.* = @intCast(i & 0xff);

    // Create transfer manager + device buffer.
    const shape_spec: pjrt.ShapeSpec = .init(shape.dims(), pjrtx.bufferTypeFromDtype(shape.dtype()));
    const tm = try platform.pjrt_client.createBuffersForAsyncHostToDevice(
        platform.pjrt_api,
        .{ .shape_specs = &.{shape_spec}, .memory = memory.pjrt_memory },
    );
    defer tm.deinit(platform.pjrt_api);
    const device_buffer = try tm.retrieveBuffer(platform.pjrt_api, 0);

    // Push in two non-adjacent halves at explicit offsets.
    const half: i64 = @intCast(N / 2);
    {
        const ev1 = try tm.transferData(platform.pjrt_api, 0, pattern[0 .. N / 2], 0, false);
        defer ev1.deinit(platform.pjrt_api);
        try ev1.await(platform.pjrt_api, io);
    }
    {
        const ev2 = try tm.transferData(platform.pjrt_api, 0, pattern[N / 2 ..], half, true);
        defer ev2.deinit(platform.pjrt_api);
        try ev2.await(platform.pjrt_api, io);
    }

    // Read back.
    var readback: [N]u8 = undefined;
    if (try device_buffer.toHostBuffer(platform.pjrt_api, &readback)) |ev| {
        defer ev.deinit(platform.pjrt_api);
        try ev.await(platform.pjrt_api, io);
    }

    if (!std.mem.eql(u8, &pattern, &readback)) {
        for (pattern, readback, 0..) |a, b, i| {
            log.info("Original: 0x{x:0>2}, Readback: 0x{x:0>2}", .{ a, b });
            if (a != b) {
                log.err("mismatch at byte {d}: pattern=0x{x:0>2} readback=0x{x:0>2}", .{ i, a, b });
                break;
            }
        }
        return error.ReadbackMismatch;
    }

    for (pattern, readback) |a, b| {
        log.info("Original: 0x{x:0>2}, Readback: 0x{x:0>2}", .{ a, b });
    }

    log.info("OK: {d} bytes round-tripped through device buffer via positional transferData", .{N});
}
