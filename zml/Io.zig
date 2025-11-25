const builtin = @import("builtin");
const std = @import("std");

const stdx = @import("stdx");
const pjrt = @import("pjrtx.zig");
const tracer = @import("context.zig").Context.tracer;
const log = std.log.scoped(.@"zml.io");

pub const ShardReader = struct {
    pub const NUM_SLOTS = 2;

    pjrt_api: *const pjrt.Api,
    pjrt_buffer: *const pjrt.Buffer,
    total_size: u64,
    chunk_size: usize,

    slots: [NUM_SLOTS]?*pjrt.Event,
    dma_buffer: []u8,
    bytes_requested: u64,
    bytes_activated: u64,
    next_request_slot: u1,
    next_consume_slot: u1,

    is_primed: bool,
    interface: std.io.Reader,

    pub fn init(pjrt_api: *const pjrt.Api, pjrt_buffer: *const pjrt.Buffer, dma_buffer: []u8) ShardReader {
        const chunk_size = dma_buffer.len / NUM_SLOTS;
        std.debug.assert(chunk_size > 0);

        return .{
            .pjrt_api = pjrt_api,
            .pjrt_buffer = pjrt_buffer,
            .total_size = pjrt_buffer.getOnDeviceSizeInBytes(pjrt_api) catch unreachable,
            .chunk_size = chunk_size,
            .slots = .{ null, null },
            .dma_buffer = dma_buffer,
            .bytes_requested = 0,
            .bytes_activated = 0,
            .next_request_slot = 0,
            .next_consume_slot = 0,
            .is_primed = false,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn requestNextChunk(self: *ShardReader) std.io.Reader.StreamError!void {
        const trace = tracer.frameStart("ShardReader.requestNextChunk");
        defer tracer.frameEnd(trace, "ShardReader.requestNextChunk");

        const slot_to_fill = self.next_request_slot;
        std.debug.assert(self.slots[slot_to_fill] == null);

        if (self.bytes_requested >= self.total_size) return;

        const offset_in_dma = slot_to_fill * self.chunk_size;
        const chunk_dma_buf = self.dma_buffer[offset_in_dma .. offset_in_dma + self.chunk_size];
        const remaining_on_device = self.total_size - self.bytes_requested;
        const transfer_size = @min(remaining_on_device, chunk_dma_buf.len);
        const dest_slice = chunk_dma_buf[0..transfer_size];
        const offset: i64 = @intCast(self.bytes_requested);

        log.debug("ShardReader.requestNextChunk: slot={d}, device_offset={d}B, transfer_size={d}B, remaining={d}B", .{
            slot_to_fill,
            offset,
            transfer_size,
            self.total_size - (self.bytes_requested + transfer_size),
        });

        const event = self.pjrt_buffer.copyRawToHost(self.pjrt_api, dest_slice, offset) catch |err| {
            log.err("PJRT copyRawToHost failed: {}", .{err});
            return error.ReadFailed;
        };

        if (event) |ev| self.slots[slot_to_fill] = ev;
        self.bytes_requested += transfer_size;
        self.next_request_slot = 1 - self.next_request_slot;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const trace = tracer.frameStart("ShardReader.stream(vtable)");
        defer tracer.frameEnd(trace, "ShardReader.stream(vtable)");

        // From Io.Writer.vtable documentation:
        // > In addition to, or instead of writing to `w`,
        // > the implementation may choose to store data in `buffer`.
        // This is what we do since we need to write in DMA memory.
        _ = w;
        _ = limit;
        const self = @as(*ShardReader, @alignCast(@fieldParentPtr("interface", r)));

        std.debug.assert(r.seek == r.end);

        if (!self.is_primed) {
            self.is_primed = true;
            if (self.total_size == 0) return error.EndOfStream;
            log.debug("ShardReader: Priming pipeline for {d} bytes...", .{self.total_size});
            for (0..NUM_SLOTS) |_| try self.requestNextChunk();
        } else {
            try self.requestNextChunk();
        }

        const slot_to_consume = self.next_consume_slot;

        if (self.slots[slot_to_consume]) |event| {
            const trace_await = tracer.frameStart("ShardReader.awaitEvent");
            defer tracer.frameEnd(trace_await, "ShardReader.awaitEvent");

            event.awaitBlocking(self.pjrt_api) catch |err| {
                log.err("Error awaiting event in stream: {}", .{err});
                return error.ReadFailed;
            };
            self.slots[slot_to_consume] = null;
        } else {
            if (self.bytes_activated >= self.total_size) {
                return error.EndOfStream;
            } else {
                log.err("ShardReader stalled: waiting for slot {d} which has no event.", .{slot_to_consume});
                return error.ReadFailed;
            }
        }

        const offset_in_dma = slot_to_consume * self.chunk_size;
        const remaining_total = self.total_size - self.bytes_activated;
        const actual_chunk_size = @min(self.chunk_size, remaining_total);

        self.bytes_activated += actual_chunk_size;

        r.buffer = self.dma_buffer[offset_in_dma .. offset_in_dma + actual_chunk_size];
        r.seek = 0;
        r.end = r.buffer.len;

        self.next_consume_slot = 1 - self.next_consume_slot;

        return 0;
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

test "ShardReader: streamRemaining" {
    const zml = @import("zml.zig");

    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const trace_test = tracer.frameStart("ShardReader.test.streamRemaining");
    defer tracer.frameEnd(trace_test, "ShardReader.test.streamRemaining");

    const Local = struct {
        fn forward() zml.Tensor {
            return .arange(.{ .end = 256 * MB }, .u32);
        }
    };

    const x_d = try zml.testing.compileAndCall(platform, Local.forward, .{});
    defer x_d.deinit();

    const dma_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(4 * KB), 1 * MB);
    defer allocator.free(dma_buffer);

    if (platform.target != .cpu) platform.pjrt_client.dmaMap(platform.pjrt_api, dma_buffer) catch {};
    defer if (platform.target != .cpu) platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_buffer);

    var device_reader: ShardReader = .init(platform.pjrt_api, x_d._shards.get(0), dma_buffer);

    const x_h = try allocator.alignedAlloc(u32, .fromByteUnits(16 * KB), x_d.shape().count());
    defer allocator.free(x_h);
    var x_h_writer: std.Io.Writer = .fixed(@ptrCast(x_h));

    const trace_stream = tracer.frameStart("ShardReader.streamRemaining");
    const bytes_read = try device_reader.interface.streamRemaining(&x_h_writer);
    tracer.frameEnd(trace_stream, "ShardReader.streamRemaining");

    std.log.warn("Device: {f}, host: {d}, read: {d}", .{
        x_d,
        256 * MB * @sizeOf(u32),
        bytes_read,
    });
    try std.testing.expectEqual(x_d.shape().byteSize(), bytes_read);
    try std.testing.expectEqual(256 * MB * @sizeOf(u32), bytes_read);
    for (x_h, 0..) |actual, expected| {
        errdefer log.err("Mismatch at offset {d}, expected {x}, got {x}", .{ expected, expected, actual });
        try std.testing.expectEqual(expected, actual);
    }
}

const KB = 1024;
const MB = 1024 * KB;
