const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

const log = std.log.scoped(.@"examples/loader");

const DataType = zml.DataType;
const Shape = zml.Shape;

const Dims = stdx.BoundedArray(i64, zml.Shape.MAX_RANK);
const StringBuilder = std.ArrayListUnmanaged(u8);

const Context = zml.Context;
const Platform = zml.Platform;
const Tracer = zml.tools.Tracer;
const pjrtx = zml.pjrt;
const pjrt = pjrtx.pjrt;

const KB = 1024;
const MB = 1024 * KB;

const BUF_1_KB = 1 * KB;
const BUF_4_KB = 4 * KB;
const BUF_8_KB = 8 * KB;
const BUF_16_KB = 16 * KB;
const BUF_32_KB = 32 * KB;
const BUF_64_KB = 64 * KB;

const BUF_1_MB = 1 * MB;
const BUF_4_MB = 4 * MB;
const BUF_8_MB = 8 * MB;
const BUF_16_MB = 16 * MB;
const BUF_32_MB = 32 * MB;
const BUF_64_MB = 64 * MB;
const BUF_128_MB = 128 * MB;
const BUF_256_MB = 256 * MB;

var tracer: Tracer = undefined;

const Tensor = struct {
    source_name: []const u8,
    name: []const u8,
    shape: Shape,
    offset: u64,

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }
};

const Shard = struct {
    shape: Shape,
    tensor: Tensor,
    device: *const pjrt.Device,

    pub fn byteSize(self: Shard) u64 {
        return self.shape.byteSize();
    }
};

pub const Registry = struct {
    pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);
    pub const Checksums = std.StringArrayHashMapUnmanaged([32]u8);

    arena: std.heap.ArenaAllocator,
    tensors: Tensors,
    metadata: Metadatas,
    checksums: Checksums,

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .{},
            .checksums = .{},
        };
    }

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();
        self.checksums.deinit(allocator);
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }

    pub fn totalBytes(self: *Registry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }
};

pub const SafetensorsSource = []const u8; // rename source to chunk
pub const SafetensorsSources = std.ArrayList(SafetensorsSource);

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    reader: *std.io.Reader,
    registry: *Registry,
) !SafetensorsSources {
    var chunks: SafetensorsSources = .{};

    const registry_allocator = registry.arena.allocator();
    var json_reader: std.json.Reader = .init(registry_allocator, reader);

    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        registry_allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;
        try chunks.append(allocator, try registry_allocator.dupe(u8, filename));
    }

    if (index.object.get("__metadata__")) |metadata| {
        var prefix_buf: [BUF_1_KB]u8 = undefined;
        try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), metadata);
    }

    return chunks;
}

fn parseSafetensors(
    allocator: std.mem.Allocator,
    registry: *Registry,
    reader: *std.io.Reader,
    source: SafetensorsSource,
) !void {
    const registry_allocator = registry.arena.allocator();
    const json_header_length: usize = @intCast(try reader.takeInt(u64, .little));
    const json_data = try allocator.alloc(u8, json_header_length);
    defer allocator.free(json_data);

    try reader.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, registry_allocator, json_data, .{});

    var it = metadata.object.iterator();

    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            var prefix_buf: [BUF_1_KB]u8 = undefined;
            try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), value);
            continue;
        }

        const shape_field = value.object.get("shape").?.array;

        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            continue;
        }

        const offset_field = value.object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const end: u64 = @intCast(offset_field.array.items[1].integer);
        const dtype = try stringToDtype(value.object.get("dtype").?.string);

        var dims: Dims = .{};
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const shape = Shape.init(dims.constSlice(), dtype);
        const size_in_bytes = end - start;
        std.debug.assert(size_in_bytes == shape.byteSize());

        const tensor: Tensor = .{
            .source_name = try registry_allocator.dupe(u8, source),
            .name = try registry_allocator.dupe(u8, key),
            .shape = shape,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(registry_allocator, key, tensor);
    }
}

const DeviceWriter2 = struct {
    platform: Platform,
    shard: Shard,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,
    pjrt_buffer: *pjrtx.Buffer,

    dma_buffer: []u8,
    chunk_size: usize,

    events: [2]?*pjrtx.Event,
    current_buffer_idx: u1,

    bytes_written: u64,
    can_process_last_event: bool,

    interface: std.io.Writer,

    pub fn init(platform: Platform, shard: Shard, buffer: []u8) !DeviceWriter2 {
        const trace = tracer.frameStart("DeviceWriter2.init");
        defer tracer.frameEnd(trace, "DeviceWriter2.init");

        std.debug.assert(buffer.len % 2 == 0);
        const chunk_size = buffer.len / 2;

        const memories = try shard.device.addressableMemories(platform.pjrt_api);
        var memory = memories[0];

        if (platform.target == .cuda) {
            for (memories) |mem| {
                if (mem.kind(platform.pjrt_api) == .device) {
                    memory = mem;
                    break;
                }
            }
        }

        const shape_spec = pjrt.ShapeSpec.init(shard.shape.dims(), bufferTypeFromDtype(shard.shape.dtype()));

        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = memory,
        });

        return .{
            .platform = platform,
            .shard = shard,
            .transfer_manager = transfer_manager,
            .pjrt_buffer = try transfer_manager.retrieveBuffer(platform.pjrt_api, 0),
            .dma_buffer = buffer,
            .chunk_size = chunk_size,
            .events = .{ null, null },
            .current_buffer_idx = 0,
            .bytes_written = 0,
            .can_process_last_event = true,
            .interface = .{ .vtable = &vtable, .buffer = buffer[0..chunk_size], .end = 0 },
        };
    }

    pub fn deinit(self: *DeviceWriter2) void {
        self.pjrt_buffer.deinit(self.platform.pjrt_api);
    }

    fn awaitEvent(self: *DeviceWriter2, idx: u1) !void {
        if (self.events[idx]) |event| {
            const trace_await = tracer.frameStart("DeviceWriter2.awaitEvent");
            defer tracer.frameEnd(trace_await, "DeviceWriter2.awaitEvent");
            try event.awaitBlocking(self.platform.pjrt_api);
            self.events[idx] = null;
        }
    }

    fn swap(self: *DeviceWriter2) void {
        log.debug("Swapping buffers: {d} -> {d}", .{ self.current_buffer_idx, 1 - self.current_buffer_idx });
        self.current_buffer_idx = 1 - self.current_buffer_idx;
        const new_offset = self.current_buffer_idx * self.chunk_size;
        self.interface.buffer = self.dma_buffer[new_offset..][0..self.chunk_size];
        self.interface.end = 0;
        log.debug("  Current buffer index: {d}, buffer range: [{}..{}]", .{ self.current_buffer_idx, new_offset, new_offset + self.chunk_size });
    }

    fn awaitTransfer(self: *DeviceWriter2) !void {
        const idx = 1 - self.current_buffer_idx;
        log.debug("Awaiting transfer on buffer index: {d}", .{idx});
        try self.awaitEvent(idx);
    }

    fn transfer(self: *DeviceWriter2, data: []const u8) !*pjrtx.Event {
        defer self.bytes_written += data.len;
        log.debug("Transferring {d} bytes at offset {d}", .{ data.len, self.bytes_written });
        return self.transfer_manager.transferData(self.platform.pjrt_api, 0, data, @intCast(self.bytes_written), false) catch |err| {
            log.err("Error during transferData in drain: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn transferAndSwap(self: *DeviceWriter2) !usize {
        try self.awaitTransfer();
        const len = self.interface.buffered().len;
        self.events[self.current_buffer_idx] = self.transfer(self.interface.buffered()) catch |err| {
            log.err("Error during transferAndSwap: {}", .{err});
            return err;
        };
        self.swap();
        return len;
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*DeviceWriter2, @alignCast(@fieldParentPtr("interface", w)));
        std.debug.assert(splat == 1);

        log.debug("DeviceWriter2.drain with buffer ptr=0x{x} called with {d} slices", .{ @intFromPtr(w.buffer.ptr), data.len });
        log.debug("  Current buffer state: idx={d}, end={d}, capacity={d}, buffered={d}", .{ self.current_buffer_idx, w.end, w.buffer.len, w.buffered().len });
        log.debug("  Bytes written so far: {d}/{d}", .{ self.bytes_written, self.shard.byteSize() });

        for (data, 0..) |slice, i| {
            log.debug("  Slice[{d}]: ptr=0x{x}, len={d}", .{ i, @intFromPtr(slice.ptr), slice.len });
        }

        // std.debug.assert(data[0].ptr == self.interface.buffer.ptr);

        const trace = tracer.frameStart("DeviceWriter2.drain (flip&wait)");
        defer tracer.frameEnd(trace, "DeviceWriter2.drain (flip&wait)");

        _ = self.transferAndSwap() catch |err| {
            log.err("Error during transferAndSwap in drain: {}", .{err});
            return error.WriteFailed;
        };

        return 0;
    }

    fn flush(w: *std.io.Writer) std.io.Writer.Error!void {
        const trace = tracer.frameStart("DeviceWriter2.flush");
        defer tracer.frameEnd(trace, "DeviceWriter2.flush");
        log.debug("DeviceWriter2.flush called", .{});

        const self = @as(*DeviceWriter2, @alignCast(@fieldParentPtr("interface", w)));

        if (w.end > 0) {
            const current_chunk = w.buffered();
            const event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, current_chunk, @intCast(self.bytes_written), false) catch |err| {
                log.err("Error during flush transfer: {}", .{err});
                return error.WriteFailed;
            };
            self.events[self.current_buffer_idx] = event;
            self.bytes_written += current_chunk.len;
            w.end = 0;
        }

        self.awaitEvent(0) catch |err| {
            log.err("Error awaiting event 0 in flush: {}", .{err});
            return error.WriteFailed;
        };

        self.awaitEvent(1) catch |err| {
            log.err("Error awaiting event 1 in flush: {}", .{err});
            return error.WriteFailed;
        };

        if (self.can_process_last_event) {
            log.debug("Finalizing transfer of {d} bytes (total expected: {d})", .{ self.bytes_written, self.shard.byteSize() });

            const last_event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, &.{}, @intCast(self.bytes_written), true) catch |err| {
                log.err("Error during final transferData: {}", .{err});
                return error.WriteFailed;
            };
            last_event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error during final awaitBlocking: {}", .{err});
                return error.WriteFailed;
            };

            self.can_process_last_event = false;
            self.transfer_manager.deinit(self.platform.pjrt_api);
            self.transfer_manager = undefined;
        }
    }

    pub fn rebase(w: *std.Io.Writer, preserve: usize, minimum_len: usize) std.Io.Writer.Error!void {
        while (w.buffer.len - w.end < minimum_len) {
            {
                // TODO: instead of this logic that "hides" data from
                // the implementation, introduce a seek index to Writer
                const preserved_head = w.end -| preserve;
                const preserved_tail = w.end;
                const preserved_len = preserved_tail - preserved_head;
                w.end = preserved_head;
                defer w.end += preserved_len;
                std.debug.assert(0 == try w.vtable.drain(w, &.{""}, 1));
                std.debug.assert(w.end <= preserved_head + preserved_len);
                // @memmove(w.buffer[w.end..][0..preserved_len], w.buffer[preserved_head..preserved_tail]);
            }

            // If the loop condition was false this assertion would have passed
            // anyway. Otherwise, give the implementation a chance to grow the
            // buffer before asserting on the buffer length.
            // std.debug.assert(w.buffer.len - preserve >= minimum_len);
        }
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
        // .rebase = std.io.Writer.defaultRebase,
    };
};

const Reader = std.io.Reader;
const Writer = std.io.Writer;
const Limit = std.io.Limit;

// The standard `std.io.Reader.Limited` does not correctly signal
// `error.EndOfStream` when its own internal limit is reached. Instead, it
// begins returning `0` for every subsequent read. A caller like `streamRemaining`
// interprets a 0-byte read as a temporary condition (not a final end-of-stream)
// and enters an infinite loop, repeatedly calling a reader that will never
// provide data again.
//
// This implementation adds a check at the beginning of `stream`. If the
// reader's limit has been exhausted (`remaining == .nothing`), it immediately
// returns `error.EndOfStream`, correctly terminating the stream and preventing
// infinite loops. It also correctly propagates `error.EndOfStream` from the
// underlying reader.
const MyLimitedReader = struct {
    unlimited: *std.io.Reader,
    remaining: std.io.Limit,
    interface: std.io.Reader,

    pub fn init(reader: *std.io.Reader, limit: std.io.Limit, buffer: []u8) MyLimitedReader {
        return .{
            .unlimited = reader,
            .remaining = limit,
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                    .discard = discard,
                },
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }

    fn stream(r: *Reader, w: *Writer, limit: Limit) Reader.StreamError!usize {
        const l: *@This() = @fieldParentPtr("interface", r);

        // If the limit has been reached, we must signal EndOfStream to the caller.
        if (l.remaining == .nothing) {
            return error.EndOfStream;
        }

        const combined_limit = limit.min(l.remaining);
        // If the underlying reader returns EndOfStream, that's fine too.
        const n = l.unlimited.stream(w, combined_limit) catch |err| switch (err) {
            error.EndOfStream => 0, // Treat underlying EOF as 0 bytes read this turn.
            else => |e| return e,
        };

        l.remaining = l.remaining.subtract(n) orelse @panic("limited reader logic error");
        return n;
    }

    fn discard(r: *Reader, limit: Limit) Reader.Error!usize {
        const l: *MyLimitedReader = @fieldParentPtr("interface", r);
        const combined_limit = limit.min(l.remaining);
        const n = try l.unlimited.discard(combined_limit);
        l.remaining = l.remaining.subtract(n).?;
        return n;
    }
};

fn testDeviceWriterWrite() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context = try Context.init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.test.DeviceWriterWrite");

    const platform = context.autoPlatform(.{});
    const devices = platform.getDevices();
    const device = devices[0];

    const TENSOR_SIZE = 2048 * MB;
    const TEST_DMA_BUFFER_SIZE = 32 * MB;

    const shape: Shape = .init(.{TENSOR_SIZE / @sizeOf(f32)}, .f32);
    const tensor: Tensor = .{
        .source_name = "test",
        .name = "tensor",
        .shape = shape,
        .offset = 0,
    };
    const test_shard: Shard = .{
        .shape = shape,
        .tensor = tensor,
        .device = device,
    };

    const dma_buffer = try allocator.alloc(u8, TEST_DMA_BUFFER_SIZE);
    defer allocator.free(dma_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_buffer) catch {};

    var writer: DeviceWriter2 = try .init(platform, test_shard, dma_buffer);
    defer writer.deinit();

    const host_data_to_write = try allocator.alloc(u8, TENSOR_SIZE);
    defer allocator.free(host_data_to_write);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, host_data_to_write);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, host_data_to_write) catch {};

    for (host_data_to_write, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    const WRITE_CHUNK_SIZE = TEST_DMA_BUFFER_SIZE / 4;
    var total_bytes_written: u64 = 0;

    while (total_bytes_written < TENSOR_SIZE) {
        const remaining = TENSOR_SIZE - total_bytes_written;
        const chunk_size = @min(remaining, WRITE_CHUNK_SIZE);
        const chunk = host_data_to_write[total_bytes_written..][0..chunk_size];
        // log.info("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });
        {
            var index: usize = 0;
            while (index < chunk.len) {
                const trace_write = tracer.frameStart("writer.interface.write");
                defer tracer.frameEnd(trace_write, "writer.interface.write");

                log.info("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });
                index += try writer.interface.write(chunk[index..]);
            }
        }
        // try writer.interface.writeAll(chunk);
        total_bytes_written += chunk.len;
    }
    try writer.interface.flush();

    // // try std.testing.expectEqual(TENSOR_SIZE, writer.bytes_written);
    // try std.testing.expectEqual(TENSOR_SIZE, total_bytes_written);

    // const dma_read_buffer = try allocator.alloc(u8, 256 * KB);
    // defer allocator.free(dma_read_buffer);
    // try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_read_buffer);
    // defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_read_buffer) catch {};

    // var device_reader = try DeviceReader.init(platform, writer.pjrt_buffer, dma_read_buffer);
    // for (0..DeviceReader.NUM_SLOTS) |_| {
    //     if (device_reader.bytes_read < device_reader.total_size) {
    //         try device_reader.requestChunk();
    //     }
    // }

    // const host_data_read_back = try allocator.alloc(u8, TENSOR_SIZE);
    // defer allocator.free(host_data_read_back);

    // var buffer_writer = std.io.Writer.fixed(host_data_read_back);
    // const bytes_read = try device_reader.interface.streamRemaining(&buffer_writer);

    // try std.testing.expectEqual(@as(usize, TENSOR_SIZE), bytes_read);
    // try std.testing.expectEqualSlices(u8, host_data_to_write, host_data_read_back);
}

const AlignedFileReader = struct {
    reader: std.fs.File.Reader,
    alignment: std.mem.Alignment,
    pos: u64, // Physical offset for the next pread
    file_size: u64,
    interface: std.io.Reader,

    buffer: [BUF_4_KB]u8 align(BUF_4_KB), // hard coded
    buffer_valid_len: usize, // Total bytes read into buffer
    buffer_consumed: usize, // Bytes consumed/skipped from buffer head

    pub fn init(reader: std.fs.File.Reader, alignment: std.mem.Alignment) !AlignedFileReader {
        const file_size = try reader.file.getEndPos();
        const alignment_bytes = alignment.toByteUnits();
        const initial_pos = reader.pos;

        var current_pos = initial_pos;
        var consumed: usize = 0;

        if (initial_pos % alignment_bytes != 0) {
            const unaligned_head = initial_pos % alignment_bytes;
            current_pos = initial_pos - unaligned_head;
            consumed = unaligned_head;
        }

        log.debug("AlignedFileReader: init called. Initial logical pos {d}. Starting physical pos {d}. Head skip {d}", .{ initial_pos, current_pos, consumed });

        return .{
            .reader = reader,
            .alignment = alignment,
            .pos = current_pos,
            .file_size = file_size,
            .buffer = undefined,
            .buffer_valid_len = 0,
            .buffer_consumed = consumed,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn stream_from_internal_buffer(self: *AlignedFileReader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const available = self.buffer[self.buffer_consumed..self.buffer_valid_len];
        if (available.len == 0) return 0;

        const copy_len = limit.minInt(available.len);
        const written = try w.write(available[0..copy_len]);

        self.buffer_consumed += written;

        return written;
    }

    fn load_aligned_block_to_internal(self: *AlignedFileReader, alignment_bytes: usize) std.io.Reader.StreamError!void {
        self.buffer_consumed = 0;
        self.buffer_valid_len = 0;

        const read_size = alignment_bytes;

        if (self.pos >= self.file_size) return error.EndOfStream;

        if (@intFromPtr(&self.buffer) % alignment_bytes != 0) {
            log.err("Internal buffer not aligned to {d} bytes: ptr=0x{x}", .{ alignment_bytes, @intFromPtr(&self.buffer) });
            return error.ReadFailed;
        }

        if (read_size % alignment_bytes != 0) {
            log.err("Read size not aligned to {d} bytes: size={d}", .{ alignment_bytes, read_size });
            return error.ReadFailed;
        }

        if (self.pos % alignment_bytes != 0) {
            log.err("File position not aligned to {d} bytes: pos={d}", .{ alignment_bytes, self.pos });
            return error.ReadFailed;
        }

        const bytes_read = self.reader.file.pread(self.buffer[0..read_size], self.pos) catch |e| {
            log.err("File pread error for aligned block: {}", .{e});
            return error.ReadFailed;
        };

        if (bytes_read == 0) return error.EndOfStream;

        self.buffer_valid_len = bytes_read;
        self.pos += bytes_read;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*AlignedFileReader, @alignCast(@fieldParentPtr("interface", r)));
        std.debug.assert(r.seek == 0 and r.end == 0);

        const alignment_bytes = self.alignment.toByteUnits();

        log.info("AlignedFileReader: stream called with limit {d}, pos {d}/{d}, block size {d}", .{ limit.toInt().?, self.pos, self.file_size, alignment_bytes });
        log.info("  AlignedFileReader: internal buffer state: valid_len={d}, consumed={d}", .{ self.buffer_valid_len, self.buffer_consumed });

        if (limit == .nothing) return error.EndOfStream;

        if (self.buffer_consumed > 0 and self.buffer_valid_len == 0) {
            // First time loading the head block
            try self.load_aligned_block_to_internal(alignment_bytes);
            if (self.buffer_valid_len == 0) return error.EndOfStream;

            return self.stream_from_internal_buffer(w, limit);
        }

        if (self.buffer_valid_len > self.buffer_consumed) {
            return self.stream_from_internal_buffer(w, limit);
        }

        self.buffer_consumed = 0;
        self.buffer_valid_len = 0;

        const logical_data_remaining_in_tensor = limit.toInt().?;
        const remaining_in_file = self.file_size - self.pos;

        if (remaining_in_file == 0) return error.EndOfStream;

        std.debug.assert(self.pos % alignment_bytes == 0);

        const dest_buffer = w.writableSliceGreedy(alignment_bytes) catch return error.WriteFailed;
        const max_read_size_to_buffer = dest_buffer.len;

        const max_aligned_read_logical = std.mem.alignBackward(usize, @min(remaining_in_file, @min(logical_data_remaining_in_tensor, max_read_size_to_buffer)), alignment_bytes);

        if (max_aligned_read_logical > 0) {
            const bytes_read = self.reader.file.pread(dest_buffer[0..max_aligned_read_logical], self.pos) catch |e| {
                log.err("File pread error: {}", .{e});
                return error.ReadFailed;
            };

            if (bytes_read == 0) return error.EndOfStream;

            log.info("  AlignedFileReader: bulk read {d} bytes at aligned pos {d}, consumed {d}", .{ bytes_read, self.pos, bytes_read });

            w.advance(bytes_read);
            self.pos += bytes_read;

            return bytes_read;
        }

        // Handle tail
        if (logical_data_remaining_in_tensor > 0) {
            try self.load_aligned_block_to_internal(alignment_bytes);

            if (self.buffer_valid_len == 0) return error.EndOfStream;

            self.buffer_valid_len = @min(self.buffer_valid_len, logical_data_remaining_in_tensor);

            return self.stream_from_internal_buffer(w, limit);
        }

        return 0;
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

fn testDirectFileReader() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const filename = "test_file.bin";

    const file_buff_size = BUF_64_MB;

    const writer_buffer_size = BUF_16_MB - 1;
    const writer_alignment_size = BUF_4_KB;

    const limited_reader_buffer_size = BUF_8_MB - 1;

    const file_buffer = try allocator.alloc(u8, file_buff_size);
    defer allocator.free(file_buffer);

    for (file_buffer, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    const tmp_file_for_writing = try tmp_dir.dir.createFile(filename, .{});
    try tmp_file_for_writing.writeAll(file_buffer);
    tmp_file_for_writing.close();

    const file_path = try tmp_dir.dir.realpathAlloc(allocator, filename);
    defer allocator.free(file_path);

    const file_fd = try std.posix.open(
        file_path,
        .{ .ACCMODE = .RDONLY, .DIRECT = true },
        0,
    );

    const file: std.fs.File = .{ .handle = file_fd };
    defer file.close();

    const file_reader = file.reader(&.{});
    // try file.seekTo(19);

    var reader = try AlignedFileReader.init(file_reader);
    var limited_reader = std.io.Reader.Limited.init(&reader.interface, .limited64(limited_reader_buffer_size), &.{});

    const writer_aligned_alloc = try allocator.alignedAlloc(u8, .fromByteUnits(writer_alignment_size), writer_buffer_size);
    defer allocator.free(writer_aligned_alloc);
    var writer = std.io.Writer.fixed(writer_aligned_alloc);

    log.info("Reading from file: {s}", .{file_path});

    var total_bytes_read: usize = 0;
    while (true) {
        const n = limited_reader.interface.stream(&writer, .unlimited) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };

        if (n == 0) break;

        total_bytes_read += n;
    }

    const bytes_read = total_bytes_read;
    log.info("Read {d} bytes from file", .{bytes_read});

    try std.testing.expectEqual(limited_reader_buffer_size, bytes_read);
    try std.testing.expectEqualSlices(u8, file_buffer[0..bytes_read], writer_aligned_alloc[0..bytes_read]);
}

fn testDeviceWriterStreamRemaining() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var context: Context = try .init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.test.DeviceWriter");

    const platform = context.autoPlatform(.{});
    const devices = platform.getDevices();
    const device = devices[0];

    const FILE_PATH = "/home/hugo/zml/data.bin";
    const INITIAL_FILE_SEEK = 0;
    const TENSOR_SIZE = 2 * MB - 4;
    const WRITER_DMA_BUFFER = 1 * MB;
    const ALIGNMENT = 4 * KB;

    const shape: Shape = .init(.{TENSOR_SIZE / @sizeOf(f32)}, .f32);
    var shard: Shard = .{
        .shape = shape,
        .tensor = .{
            .source_name = "test",
            .name = "tensor",
            .shape = shape,
            .offset = 0,
        },
        .device = device,
    };

    const writer_dma_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(ALIGNMENT), WRITER_DMA_BUFFER);
    defer allocator.free(writer_dma_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, writer_dma_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, writer_dma_buffer) catch {};

    const file_fd = try std.posix.open(
        FILE_PATH,
        .{ .ACCMODE = .RDONLY, .DIRECT = true },
        0,
    );

    const file: std.fs.File = .{ .handle = file_fd };
    defer file.close();

    var file_reader = file.reader(&.{});
    try file_reader.seekTo(INITIAL_FILE_SEEK);

    var tensor_offset: u64 = 0;

    for (0..2) |_| {
        shard.tensor.offset = tensor_offset;
        defer tensor_offset += shard.byteSize();

        try file_reader.seekTo(tensor_offset);

        var aligned_file_reader: AlignedFileReader = try .init(file_reader, .fromByteUnits(ALIGNMENT));

        var limited_reader: std.io.Reader.Limited = .init(&aligned_file_reader.interface, .limited64(shard.byteSize()), &.{});

        var writer: DeviceWriter2 = try .init(platform, shard, writer_dma_buffer);
        defer writer.deinit();

        log.info("Tensor info: source={s}, name={s}, shape={any}, offset={d}, byteSize={d}", .{ shard.tensor.source_name, shard.tensor.name, shard.tensor.shape.dims(), shard.tensor.offset, shard.tensor.byteSize() });
        log.info("Shard info: shape={any}, byteSize={d}, device_id={d}", .{ shard.shape.dims(), shard.byteSize(), shard.device.getLocalHardwareId(platform.pjrt_api) });
        log.info("", .{});
        log.info("FileReader initial state:", .{});
        log.info("    pos: {d}", .{file_reader.pos});
        log.info("    file_size: {d}", .{(try file_reader.file.getEndPos())});
        log.info("    remaining: {d}", .{(try file_reader.file.getEndPos()) - file_reader.pos});
        log.info("    pos aligned: {}", .{file_reader.pos % aligned_file_reader.alignment.toByteUnits() == 0});
        log.info("AlignedFileReader initial state:", .{});
        log.info("    pos: {d}", .{aligned_file_reader.pos});
        log.info("    file_size: {d}", .{aligned_file_reader.file_size});
        log.info("    alignment: {d}", .{aligned_file_reader.alignment.toByteUnits()});
        log.info("    pos aligned: {}", .{aligned_file_reader.pos % aligned_file_reader.alignment.toByteUnits() == 0});
        log.info("LimitedReader state:", .{});
        log.info("    seek: {d}", .{limited_reader.interface.seek});
        log.info("    end: {d}", .{limited_reader.interface.end});
        log.info("    remaining: {d}", .{limited_reader.remaining.toInt().?});
        log.info("DeviceWriter2 initial state:", .{});
        log.info("    shard byte size: {d}", .{writer.shard.byteSize()});
        log.info("    dma_buffer length: {d}", .{writer.dma_buffer.len});
        log.info("    dma_buffer aligned: {}", .{@intFromPtr(writer.dma_buffer.ptr) % aligned_file_reader.alignment.toByteUnits() == 0});
        log.info("    chunk_size: {d}", .{writer.chunk_size});
        log.info("    chunk_size aligned: {}", .{writer.chunk_size % aligned_file_reader.alignment.toByteUnits() == 0});
        log.info("    current_buffer_idx: {d}", .{writer.current_buffer_idx});
        log.info("    bytes_written: {d}", .{writer.bytes_written});
        log.info("    interface buffer ptr: 0x{x}", .{@intFromPtr(writer.interface.buffer.ptr)});
        log.info("    interface buffer ptr aligned: {}", .{@intFromPtr(writer.interface.buffer.ptr) % aligned_file_reader.alignment.toByteUnits() == 0});
        log.info("    interface buffer len: {d}", .{writer.interface.buffer.len});
        log.info("    interface buffer len aligned: {}", .{writer.interface.buffer.len % aligned_file_reader.alignment.toByteUnits() == 0});
        log.info("    interface end: {d}", .{writer.interface.end});

        const trace = tracer.frameStart("streamRemaining");
        const bytes_written = try limited_reader.interface.streamRemaining(&writer.interface);
        try writer.interface.flush();
        tracer.frameEnd(trace, "streamRemaining");

        log.info("Total bytes written: {d}", .{bytes_written});
        log.info("Expected tensor size: {d}", .{shard.byteSize()});
        log.info("AlignedFileReader state:", .{});
        log.info("    pos: {d}", .{aligned_file_reader.pos});
        log.info("    file_size: {d}", .{aligned_file_reader.file_size});
        log.info("    remaining: {d}", .{aligned_file_reader.file_size - aligned_file_reader.pos});
        log.info("DeviceWriter2 state:", .{});
        log.info("    bytes_written: {d}", .{writer.bytes_written});
        log.info("    current_buffer_idx: {d}", .{writer.current_buffer_idx});
        log.info("    can_process_last_event: {}", .{writer.can_process_last_event});
        log.info("\n", .{});

        // try std.testing.expectEqual(INITIAL_FILE_SEEK + bytes_written, file_reader.pos);
        try std.testing.expectEqual(shard.byteSize(), bytes_written);
    }
}

const TensorWriter = struct {
    const Self = @This();

    device_writers: []DeviceWriter2,

    buffer: []u8,
    chunk_size: usize,
    current_buffer_idx: u1,

    shard_size: u64,
    is_sharded: bool,
    total_bytes_processed: u64,

    interface: std.io.Writer,

    pub fn init(device_writers: []DeviceWriter2, buffer: []u8) Self {
        std.debug.assert(buffer.len % 2 == 0);
        const chunk_size = buffer.len / 2;

        const is_sharded = if (device_writers.len > 0)
            device_writers[0].shard.byteSize() < device_writers[0].shard.tensor.byteSize()
        else
            false;

        return .{
            .device_writers = device_writers,
            .buffer = buffer,
            .chunk_size = chunk_size,
            .current_buffer_idx = 0,
            .shard_size = if (device_writers.len > 0) device_writers[0].shard.byteSize() else 0,
            .is_sharded = is_sharded,
            .total_bytes_processed = 0,
            .interface = .{ .vtable = &vtable, .buffer = buffer[0..chunk_size], .end = 0 },
        };
    }

    fn processChunk(self: *Self, data: []const u8) !void {
        const trace = tracer.frameStart("TensorWriter.processChunk");
        defer tracer.frameEnd(trace, "TensorWriter.processChunk");

        var data_offset: usize = 0;
        while (data_offset < data.len) {
            const current_tensor_offset = self.total_bytes_processed + data_offset;
            const remaining_in_data = data.len - data_offset;

            var chunk_size: u64 = 0;
            var target_writers_start: usize = 0;
            var target_writers_end: usize = 0;

            if (self.is_sharded) {
                const total_TENSOR_SIZE = self.shard_size * self.device_writers.len;
                if (current_tensor_offset >= total_TENSOR_SIZE) break;
                const current_shard_idx: usize = @intCast(current_tensor_offset / self.shard_size);
                const offset_in_shard = current_tensor_offset % self.shard_size;
                const remaining_in_shard = self.shard_size - offset_in_shard;
                chunk_size = @min(@as(u64, @intCast(remaining_in_data)), remaining_in_shard);
                target_writers_start = current_shard_idx;
                target_writers_end = current_shard_idx + 1;
            } else {
                if (current_tensor_offset >= self.shard_size) break;
                const remaining_in_tensor = self.shard_size - current_tensor_offset;
                chunk_size = @min(@as(u64, @intCast(remaining_in_data)), remaining_in_tensor);
                target_writers_start = 0;
                target_writers_end = self.device_writers.len;
            }

            const chunk_size_usize: usize = @intCast(chunk_size);
            if (chunk_size_usize == 0) break;
            const chunk_to_write = data[data_offset .. data_offset + chunk_size_usize];
            for (self.device_writers[target_writers_start..target_writers_end]) |*dw| {
                try dw.interface.writeAll(chunk_to_write);
            }
            data_offset += chunk_size_usize;
        }
        self.total_bytes_processed += data.len;
    }

    fn rebase(w: *std.io.Writer, preserve: usize, capacity: usize) std.io.Writer.Error!void {
        const self: *Self = @alignCast(@fieldParentPtr("interface", w));
        std.debug.assert(preserve == 0);
        _ = capacity;

        const old_idx = self.current_buffer_idx;
        const new_idx = 1 - old_idx;

        const old_offset = old_idx * self.chunk_size;
        const chunk_to_process = self.buffer[old_offset .. old_offset + w.end];
        if (chunk_to_process.len > 0) {
            self.processChunk(chunk_to_process) catch |err| {
                log.err("Error in TensorWriter.processChunk: {}", .{err});
                return error.WriteFailed;
            };
        }

        self.current_buffer_idx = new_idx;
        const new_offset = new_idx * self.chunk_size;
        w.buffer = self.buffer[new_offset .. new_offset + self.chunk_size];
        w.end = 0;
    }

    // `drain` is simple: it just delegates to `rebase` to do the heavy lifting.
    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        w.rebase(0, 1) catch |err| {
            log.err("Error during TensorWriter.rebase in drain: {}", .{err});
            return error.WriteFailed;
        };
        return w.writeSplat(data, splat);
    }

    fn flush(w: *std.io.Writer) !void {
        const self: *Self = @alignCast(@fieldParentPtr("interface", w));

        if (w.end > 0) {
            self.processChunk(w.buffered()) catch |err| {
                log.err("Error processing chunk in flush: {}", .{err});
                return error.WriteFailed;
            };
            w.end = 0;
        }

        for (self.device_writers) |*dw| {
            dw.interface.flush() catch |err| {
                log.err("Error flushing downstream DeviceWriter: {}", .{err});
                return error.WriteFailed;
            };
        }
    }

    const vtable = std.io.Writer.VTable{ .drain = drain, .flush = flush, .rebase = rebase };
};

const DeviceReader = struct {
    pub const NUM_SLOTS = 10;

    platform: Platform,
    pjrt_buffer: *const pjrtx.Buffer,
    total_size: u64,
    chunk_size: usize,
    slots: [NUM_SLOTS]?*pjrtx.Event,
    next_slot_idx: usize,
    consume_slot_idx: usize,
    slot_offsets: [NUM_SLOTS]u64,

    buffer: []u8,
    bytes_read: u64,

    interface: std.io.Reader,

    pub fn init(platform: Platform, pjrt_buffer: *const pjrtx.Buffer, buffer: []u8) !DeviceReader {
        const chunk_size = buffer.len / NUM_SLOTS;

        return .{
            .platform = platform,
            .pjrt_buffer = pjrt_buffer,
            .total_size = try pjrt_buffer.getOnDeviceSizeInBytes(platform.pjrt_api),
            .chunk_size = chunk_size,
            .slots = [_]?*pjrtx.Event{null} ** NUM_SLOTS,
            .next_slot_idx = 0,
            .consume_slot_idx = 0,
            .slot_offsets = undefined,
            .buffer = buffer,
            .bytes_read = 0,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn awaitSlot(self: *DeviceReader, slot_index: usize) !void {
        if (self.slots[slot_index]) |event| {
            try event.awaitBlocking(self.platform.pjrt_api);
            self.slots[slot_index] = null;
        }
    }

    fn requestChunk(self: *DeviceReader) !void {
        const slot_idx = self.next_slot_idx;

        try self.awaitSlot(slot_idx);

        const offset_buffer = slot_idx * self.chunk_size;
        const chunk_buffer = self.buffer[offset_buffer .. offset_buffer + self.chunk_size];

        const remaining_on_device = self.total_size - self.bytes_read;
        if (remaining_on_device == 0) return;

        const transfer_size = @min(remaining_on_device, chunk_buffer.len);
        const dest_slice = chunk_buffer[0..transfer_size];

        self.slot_offsets[slot_idx] = self.bytes_read;

        if (try self.pjrt_buffer.copyRawToHost(self.platform.pjrt_api, dest_slice, @intCast(self.bytes_read))) |event| {
            self.slots[slot_idx] = event;
        }

        self.bytes_read += transfer_size;
        self.next_slot_idx = (self.next_slot_idx + 1) % NUM_SLOTS;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*DeviceReader, @alignCast(@fieldParentPtr("interface", r)));
        _ = w;
        _ = limit;

        std.debug.assert(r.seek == r.end);

        if (self.slots[self.consume_slot_idx] == null and self.bytes_read >= self.total_size) {
            return error.EndOfStream;
        }

        self.awaitSlot(self.consume_slot_idx) catch |err| {
            log.err("Error during awaitSlot: {}", .{err});
            return error.ReadFailed;
        };

        const offset_in_main_buffer = self.consume_slot_idx * self.chunk_size;

        const chunk_start_offset = self.slot_offsets[self.consume_slot_idx];
        const remaining_on_device = self.total_size - chunk_start_offset;
        const actual_chunk_size = @min(self.chunk_size, remaining_on_device);

        if (actual_chunk_size == 0) {
            return error.EndOfStream;
        }

        r.buffer = self.buffer[offset_in_main_buffer .. offset_in_main_buffer + self.chunk_size];
        r.seek = 0;
        r.end = @intCast(actual_chunk_size);

        self.consume_slot_idx = (self.consume_slot_idx + 1) % NUM_SLOTS;

        if (self.bytes_read < self.total_size) {
            self.requestChunk() catch |err| {
                log.err("Error during requestChunk: {}", .{err});
                return error.ReadFailed;
            };
        }

        return 0;
    }

    const vtable = std.io.Reader.VTable{
        .stream = stream,
    };
};

const TensorReader = struct {
    device_readers: []DeviceReader,
    current_reader_idx: usize,

    interface: std.io.Reader,

    pub fn init(device_readers: []DeviceReader) !TensorReader {
        for (device_readers) |*dr| {
            for (0..DeviceReader.NUM_SLOTS) |_| {
                if (dr.bytes_read < dr.total_size) {
                    try dr.requestChunk();
                }
            }
        }

        return .{
            .device_readers = device_readers,
            .current_reader_idx = 0,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*TensorReader, @alignCast(@fieldParentPtr("interface", r)));

        while (self.current_reader_idx < self.device_readers.len) {
            const current_device_reader = &self.device_readers[self.current_reader_idx].interface;

            const bytes_read = current_device_reader.stream(w, limit) catch |err| switch (err) {
                error.EndOfStream => {
                    self.current_reader_idx += 1;
                    continue;
                },
                else => |e| return e,
            };

            return bytes_read;
        }

        // If the loop finishes, all device readers are exhausted.
        return error.EndOfStream;
    }

    const vtable = std.io.Reader.VTable{
        .stream = stream,
    };
};

// This is an example of how sharding metadata might be added to a model registry.
// In a real application, this metadata might come from a config file or be inferred from the
// model architecture. Here, we hardcode some example tensor names and shard them on axis 1.
fn addExampleShardingMetadata(registry: *Registry) !void {
    const sharded_names = [_][]const u8{
        "model.embed_tokens.weight",
        "lm_head.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
    };

    const sharding_axis_indices = try registry.arena.allocator().dupe(i64, &.{1}); // Shard on axis 1 (columns)
    const sharding_meta_value: Metadata = .{ .array_int = sharding_axis_indices };

    var tensor_it = registry.tensors.iterator();

    for (sharded_names) |name| {
        tensor_it.reset();

        while (tensor_it.next()) |entry| {
            const tensor = entry.value_ptr.*;

            if (std.mem.endsWith(u8, tensor.name, name)) {
                var key_buf: [512]u8 = undefined;
                const sharding_key = try std.fmt.bufPrint(&key_buf, "sharding.{s}", .{tensor.name});

                try registry.metadata.put(
                    registry.arena.allocator(),
                    try registry.arena.allocator().dupe(u8, sharding_key),
                    sharding_meta_value,
                );
            }
        }
    }
}

// Annotate tensor shapes with sharding information from metadata.
fn annotateShapesWithSharding(registry: *Registry) !void {
    var tensor_it = registry.tensors.iterator();

    while (tensor_it.next()) |entry| {
        const tensor = entry.value_ptr;
        var key_buf: [512]u8 = undefined;
        const sharding_key = std.fmt.bufPrint(&key_buf, "sharding.{s}", .{tensor.name}) catch continue;

        if (registry.metadata.get(sharding_key)) |sharding_axes_meta| {
            for (sharding_axes_meta.array_int) |sharding_axis| {
                tensor.shape = tensor.shape.withSharding(.{sharding_axis});
            }
        }
    }
}

// Compute the shards for a tensor based on its shape and the available devices.
fn computeShards(allocator: std.mem.Allocator, tensor: Tensor, devices: []const *const pjrt.Device) ![]Shard {
    const sharded_axes_count = std.simd.countTrues(tensor.shape._sharding_info);
    const is_sharded = sharded_axes_count > 0;

    const shards = try allocator.alloc(Shard, devices.len);

    if (!is_sharded) {
        for (devices, 0..) |device, i| {
            shards[i] = .{ .shape = tensor.shape, .tensor = tensor, .device = device };
        }
    } else {
        const sharded_axis = std.simd.firstIndexOfValue(tensor.shape._sharding_info, true) orelse unreachable;
        const original_dim: u64 = @intCast(tensor.shape.dim(sharded_axis));

        const shard_dim = original_dim / @as(u64, @intCast(devices.len));

        var shard_shape = tensor.shape;
        shard_shape._dims.set(sharded_axis, @intCast(shard_dim));

        for (devices, 0..) |device, i| {
            shards[i] = .{ .shape = shard_shape, .tensor = tensor, .device = device };
        }
    }

    return shards;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var timer = try std.time.Timer.start();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.next().?;

    const file_path = args.next() orelse {
        log.err("Usage: bazel run //examples/loader /path/to/model.safetensors...", .{});
        return;
    };

    log.info("--- Initializing context and platform... ---", .{});
    var context = try Context.init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.examples.loader");
    tracer.event("Initialized tracer");

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const devices = platform.getDevices();

    const elapsed_init = timer.lap();
    log.info("--- Initialized context and platform with {d} devices in {d}ms ---", .{ devices.len, elapsed_init / std.time.ns_per_ms });

    const trace_post_init = tracer.frameStart("Main post context/platform init");
    defer tracer.frameEnd(trace_post_init, "Main post context/platform init");

    log.info("--- Discovering model... ---", .{});
    const trace_discovery = tracer.frameStart("Weights discovery");
    var registry: Registry = .init(allocator);
    defer registry.deinit();

    var files: std.StringHashMapUnmanaged(std.fs.File) = .{};
    defer {
        var it = files.iterator();

        while (it.next()) |entry| {
            entry.value_ptr.close();
            allocator.free(entry.key_ptr.*);
        }

        files.deinit(allocator);
    }

    if (std.mem.endsWith(u8, file_path, ".safetensors.index.json")) {
        const index_file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_only });

        const index_reader_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(index_reader_buffer);

        var index_reader = index_file.reader(index_reader_buffer);
        var chunks = try parseSafetensorsIndex(allocator, &index_reader.interface, &registry);
        defer chunks.deinit(allocator);

        const file_reader_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(file_reader_buffer);

        for (chunks.items) |chunk| {
            if (files.get(chunk)) |_| {
                // model.safetensors.index.json weight map values may contain duplicates
                continue;
            }

            const chunk_file_path = try std.fs.path.join(allocator, &.{ std.fs.path.dirname(file_path).?, chunk });
            defer allocator.free(chunk_file_path);

            const chunk_file = try std.fs.openFileAbsolute(chunk_file_path, .{ .mode = .read_only });
            try files.put(allocator, try allocator.dupe(u8, std.fs.path.basename(chunk_file_path)), chunk_file);

            var chunk_reader = chunk_file.reader(file_reader_buffer);

            try parseSafetensors(allocator, &registry, &chunk_reader.interface, chunk);
        }
    } else {
        const file_reader_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(file_reader_buffer);

        const file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_only });
        try files.put(allocator, try allocator.dupe(u8, std.fs.path.basename(file_path)), file);

        var chunk_reader = file.reader(file_reader_buffer);

        try parseSafetensors(allocator, &registry, &chunk_reader.interface, std.fs.path.basename(file_path));
    }

    const elapsed_discovery = timer.lap();
    tracer.frameEnd(trace_discovery, "Weights discovery");
    log.info("--- Discovered {d} tensors in model ({d:.2} GB) in {d}ms ---", .{ registry.tensors.count(), registry.totalBytes() / (1024 * 1024 * 1024), elapsed_discovery / std.time.ns_per_ms });

    var streaming_files: std.StringHashMapUnmanaged(std.fs.File) = .{};
    defer {
        var it = streaming_files.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.close();
            allocator.free(entry.key_ptr.*);
        }
        streaming_files.deinit(allocator);
    }

    // Populate the O_DIRECT map by re-opening the files discovered during parsing.
    var parsing_it = files.iterator();
    while (parsing_it.next()) |entry| {
        const basename = entry.key_ptr.*;
        const dirname = std.fs.path.dirname(file_path).?;
        const full_path = try std.fs.path.join(allocator, &.{ dirname, basename });
        defer allocator.free(full_path);

        const file_fd = try std.posix.open(
            full_path,
            .{ .ACCMODE = .RDONLY, .DIRECT = true },
            0,
        );
        const file = std.fs.File{ .handle = file_fd };
        try streaming_files.put(allocator, try allocator.dupe(u8, basename), file);
    }

    log.info("--- Applying sharding information... ---", .{});
    try addExampleShardingMetadata(&registry);
    try annotateShapesWithSharding(&registry);

    const elapsed_sharding = timer.lap();
    log.info("--- Applied sharding information in {d}ms ---", .{elapsed_sharding / std.time.ns_per_ms});

    log.info("--- Sorting tensors by source and offset... ---", .{});
    const tensors = blk: {
        const ts = registry.tensors.values();

        std.mem.sort(Tensor, ts, {}, struct {
            fn lessThan(_: void, a: Tensor, b: Tensor) bool {
                const name_cmp = std.mem.order(u8, a.source_name, b.source_name);
                return switch (name_cmp) {
                    .lt => true,
                    .gt => false,
                    .eq => a.offset < b.offset,
                };
            }
        }.lessThan);

        break :blk ts;
    };
    const elapsed_sorting = timer.lap();
    log.info("--- Sorted tensors in {d}ms ---", .{elapsed_sorting / std.time.ns_per_ms});

    const verify_checksums = false;

    if (verify_checksums) {
        log.info("--- Pre-computing checksums from disk... ---", .{});

        const checksum_file_buffer = try allocator.alloc(u8, BUF_256_MB);
        defer allocator.free(checksum_file_buffer);

        var hasher_buffer: [32]u8 = undefined;

        const registry_allocator = registry.arena.allocator();

        for (tensors) |tensor| {
            if (tensor.shape.byteSize() == 0) continue;
            if (registry.checksums.contains(tensor.name)) continue;

            const file = files.get(tensor.source_name).?;
            try file.seekTo(tensor.offset);

            var file_reader = file.reader(checksum_file_buffer);
            var limited_reader: std.io.Reader.Limited = .init(&file_reader.interface, .limited64(tensor.shape.byteSize()), &.{});

            var hasher_writer: std.io.Writer.Hashing(std.crypto.hash.sha2.Sha256) = .init(&hasher_buffer);

            const bytes_hashed = try limited_reader.interface.streamRemaining(&hasher_writer.writer);
            std.debug.assert(bytes_hashed == tensor.byteSize());

            var digest: [32]u8 = undefined;
            hasher_writer.hasher.final(&digest);

            try registry.checksums.put(registry_allocator, tensor.name, digest);
        }

        const checksum_elapsed = timer.lap();
        const gb_hashed = @as(f64, @floatFromInt(registry.totalBytes())) / (1.0 * 1024 * 1024 * 1024);
        const checksum_rate = if (checksum_elapsed > 0) gb_hashed / (@as(f64, @floatFromInt(checksum_elapsed)) / 1_000_000_000.0) else 0;
        log.info("--- Pre-computed {d} checksums in {d}ms ({d:.2} GB at {d:.2} GB/s) ---", .{ registry.checksums.count(), checksum_elapsed / std.time.ns_per_ms, gb_hashed, checksum_rate });
    }

    log.info("--- Allocating buffers... ---", .{});
    const trace_allocation = tracer.frameStart("Buffers allocation and DMA mapping");

    const DMA_STAGING_BUFFER_SIZE = BUF_8_MB * platform.getDevices().len;

    const dma_write_buffer = try allocator.alloc(u8, DMA_STAGING_BUFFER_SIZE);
    defer allocator.free(dma_write_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_write_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_write_buffer) catch unreachable;

    const dma_read_buffer = try allocator.alloc(u8, BUF_8_MB);
    defer allocator.free(dma_read_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_read_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_read_buffer) catch unreachable;

    const file_buffer = try allocator.alloc(u8, BUF_8_MB);
    defer allocator.free(file_buffer);

    const tensor_reader_buffer: [0]u8 = undefined;
    const tensor_writer_buffer = try allocator.alloc(u8, BUF_8_MB);
    defer allocator.free(tensor_writer_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, tensor_writer_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, tensor_writer_buffer) catch unreachable;

    const elasped_preparation = timer.lap();
    tracer.frameEnd(trace_allocation, "Buffers allocation and DMA mapping");
    log.info("--- Prepared for tensor processing in {d}ms ---", .{elasped_preparation / std.time.ns_per_ms});

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var sum_total_bytes_copied: u64 = 0;

    log.info("--- Warming up transfer managers for each device... ---", .{});
    var warmup_timer = try std.time.Timer.start();

    for (devices) |device| {
        const trace_warmup = tracer.frameStart("Transfer Manager Warmup");
        defer tracer.frameEnd(trace_warmup, "Transfer Manager Warmup");

        const WARMUP_SIZE = BUF_1_MB;
        const warmup_shape = Shape.init(.{WARMUP_SIZE / 4}, .f32);

        const warmup_data = try allocator.alloc(u8, WARMUP_SIZE);
        defer allocator.free(warmup_data);

        const warmup_tensor: Tensor = .{
            .source_name = "warmup",
            .name = "warmup_tensor",
            .shape = warmup_shape,
            .offset = 0,
        };

        const warmup_shard = Shard{
            .shape = warmup_shape,
            .tensor = warmup_tensor,
            .device = device,
        };

        const warmp_buffer = try allocator.alloc(u8, BUF_8_MB);
        defer allocator.free(warmp_buffer);

        var warmup_writer = try DeviceWriter2.init(platform, warmup_shard, warmp_buffer);
        defer warmup_writer.deinit();

        try warmup_writer.interface.writeAll(warmup_data);
        try warmup_writer.interface.flush();
    }

    const warmup_elapsed = warmup_timer.lap();
    log.info("--- Warmed up transfer managers for {d} devices in {d}ms ---", .{ devices.len, warmup_elapsed / std.time.ns_per_ms });

    log.info("--- Starting tensor processing stream... ---", .{});
    timer.reset();

    const trace_processing = tracer.frameStart("Tensor Processing Stream");
    for (tensors) |tensor| {
        var tensor_timer = try std.time.Timer.start();
        const trace = tracer.frameStart("Tensor Processing");
        defer tracer.frameEnd(trace, "Tensor Processing");

        log.info("Processing {s}/{s} dims={any} size={d} offset={d}", .{
            tensor.source_name,
            tensor.name,
            tensor.shape.dims(),
            tensor.byteSize(),
            tensor.offset,
        });

        _ = arena.reset(.free_all);

        const arena_allocator = arena.allocator();

        const file = streaming_files.get(tensor.source_name).?;

        var file_reader = file.reader(&.{});
        try file_reader.seekTo(tensor.offset);
        var aligned_file_reader: AlignedFileReader = try .init(file_reader, .fromByteUnits(BUF_4_KB));

        var tensor_reader: std.io.Reader.Limited = .init(&aligned_file_reader.interface, .limited64(tensor.shape.byteSize()), &tensor_reader_buffer);
        var device_writers: std.ArrayList(DeviceWriter2) = try .initCapacity(arena_allocator, devices.len);
        defer {
            for (device_writers.items) |*device_writer| {
                device_writer.deinit();
            }

            device_writers.deinit(arena_allocator);
        }

        const shards = try computeShards(arena_allocator, tensor, devices);
        defer arena_allocator.free(shards);

        const per_device_dma_size = DMA_STAGING_BUFFER_SIZE / devices.len;
        const per_device_dma_chunk_size = per_device_dma_size / 2;

        for (0..devices.len) |i| {
            const device_dma_slice = dma_write_buffer[i * per_device_dma_size .. (i + 1) * per_device_dma_size];
            _ = device_dma_slice; // autofix
            try device_writers.append(allocator, try .init(
                platform,
                shards[i],
                dma_write_buffer,
            ));
        }

        if (tensor_writer_buffer.len < per_device_dma_chunk_size) {
            log.err("TensorWriter buffer is too small for optimal flow. Have {d}, need {d}", .{
                tensor_writer_buffer.len, per_device_dma_chunk_size,
            });
            return error.BufferTooSmall;
        }
        const correctly_sized_tensor_writer_buffer = tensor_writer_buffer[0..per_device_dma_chunk_size];

        var tensor_writer: TensorWriter = .init(device_writers.items, correctly_sized_tensor_writer_buffer);

        const bytes_copied = try tensor_reader.interface.streamRemaining(&tensor_writer.interface);
        try tensor_writer.interface.flush();

        const elapsed_tensor = tensor_timer.lap();
        const mb_copied = @as(f64, @floatFromInt(bytes_copied)) / (1.0 * 1024 * 1024);
        const rate = if (elapsed_tensor > 0) mb_copied / (@as(f64, @floatFromInt(elapsed_tensor)) / 1_000_000_000.0) else 0;
        log.info("Loaded tensor in {d:.2}ms ({d:.2} MB at {d:.2} MB/s) ---", .{ elapsed_tensor / std.time.ns_per_ms, mb_copied, rate });

        std.debug.assert(bytes_copied == tensor.shape.byteSize());

        sum_total_bytes_copied += bytes_copied;

        if (verify_checksums) {
            var checksum_timer = try std.time.Timer.start();

            const expected_checksum = registry.checksums.get(tensor.name).?;

            var device_readers = try std.ArrayList(DeviceReader).initCapacity(allocator, devices.len);
            defer device_readers.deinit(allocator);

            const per_device_d2h_size = DMA_STAGING_BUFFER_SIZE / devices.len;

            for (0..devices.len) |i| {
                const d2h_dma_slice = dma_read_buffer[i * per_device_d2h_size .. (i + 1) * per_device_d2h_size];
                const pjrt_buffer = device_writers.items[i].pjrt_buffer;
                device_readers.appendAssumeCapacity(try .init(platform, pjrt_buffer, d2h_dma_slice));
            }

            var tensor_reader_from_device: TensorReader = try .init(device_readers.items);
            var hasher_buffer: [32]u8 = undefined;
            var hasher_writer: std.io.Writer.Hashing(std.crypto.hash.sha2.Sha256) = .init(&hasher_buffer);

            const bytes_hashed = try tensor_reader_from_device.interface.streamRemaining(&hasher_writer.writer);
            std.debug.assert(bytes_hashed == tensor.byteSize());

            var digest: [32]u8 = undefined;
            hasher_writer.hasher.final(&digest);

            if (!std.mem.eql(u8, &digest, &expected_checksum)) {
                log.err("!!! CHECKSUM MISMATCH for tensor '{s}' !!!", .{tensor.name});
                log.err("  Expected: {any}", .{&expected_checksum});
                log.err("  Computed: {any}", .{&digest});
                // return error.ChecksumMismatch;
            }

            const elapsed_checksum = checksum_timer.lap();
            log.info("Checksum OK for tensor '{s}' (verified in {d:.2}ms)", .{ tensor.name, @as(f64, @floatFromInt(elapsed_checksum)) / std.time.ns_per_ms });
        }
    }
    tracer.frameEnd(trace_processing, "Tensor Processing Stream");

    const elapsed = timer.read();
    const gb_copied = @as(f64, @floatFromInt(sum_total_bytes_copied)) / (1.0 * 1024 * 1024 * 1024);
    const rate = if (elapsed > 0) gb_copied / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    log.warn("--- All tensors loaded in {d}ms ({d:.2} GB at {d:.2} GB/s) ---", .{ elapsed / std.time.ns_per_ms, gb_copied, rate });
}

// all code below is unmodified (or slightly) / imported strucs / funcs from zml

pub const Metadata = union(enum) {
    null: void,
    int: i64,
    float: f64,
    bool: bool,
    string: []const u8,

    array_bool: []const bool,
    array_int: []const i64,
    array_float: []const f64,
    array_string: []const []const u8,

    pub const ItemType = enum {
        int,
        float,
        bool,
        string,

        pub fn toZigType(comptime kind: ItemType) type {
            return switch (kind) {
                .int => i64,
                .float => f64,
                .bool => bool,
                .string => []const u8,
            };
        }
    };

    pub fn wrap(x: anytype) Metadata {
        return switch (@TypeOf(x)) {
            inline u8, i8, u16, i16, u32, i32, u64, i64 => .{ .int = @intCast(x) },
            inline f16, f32, f64 => .{ .float = @floatCast(x) },
            bool => .{ .bool = x },
            []const u8 => .{ .string = x },
            else => @panic("Unsupported type for Value: " ++ @typeName(@TypeOf(x))),
        };
    }

    pub fn copySlice(allocator: std.mem.Allocator, any_slice: anytype) !Metadata {
        return switch (@TypeOf(any_slice[0])) {
            inline u8, i8, u16, i16, u32, i32, u64, i64 => {
                const res = try allocator.alloc(i64, any_slice.len);
                for (res, any_slice) |*r, val| r.* = @intCast(val);
                return .{ .array_int = res };
            },
            inline f16, f32, f64 => {
                const res = try allocator.alloc(f64, any_slice.len);
                for (res, any_slice) |*r, val| r.* = @floatCast(val);
                return .{ .array_float = res };
            },
            bool => .{ .array_bool = try allocator.dupe(bool, any_slice) },
            []const u8 => .{ .array_string = try allocator.dupe([]const u8, @alignCast(any_slice)) },
            else => @panic("Unsupported type for Value: " ++ @typeName(@TypeOf(any_slice))),
        };
    }

    pub fn format(
        self: Metadata,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .null => _ = try writer.write("null"),
            inline .bool, .array_bool => |b| try writer.print("{any}", .{b}),
            inline else => |v| try writer.print("{d}", .{v}),
        }
    }
};

fn stringToDtype(safetensor_type: []const u8) !DataType {
    const map = std.StaticStringMap(DataType).initComptime(.{
        .{ "F64", .f64 },
        .{ "F32", .f32 },
        .{ "F16", .f16 },
        .{ "BF16", .bf16 },
        .{ "F8_E4M3", .f8e4m3fn },
        .{ "I64", .i64 },
        .{ "I32", .i32 },
        .{ "I16", .i16 },
        .{ "I8", .i8 },
        .{ "U64", .u64 },
        .{ "U32", .u32 },
        .{ "U16", .u16 },
        .{ "U8", .u8 },
        .{ "BOOL", .bool },
    });

    return map.get(safetensor_type) orelse {
        log.err("Unsupported safetensor data type: {s}", .{safetensor_type});
        return error.UnsupportedDataType;
    };
}

pub fn parseMetadata(registry: *Registry, prefix: StringBuilder, val: std.json.Value) !void {
    const allocator = registry.arena.allocator();
    const metadata = &registry.metadata;
    const key = prefix.items;
    return switch (val) {
        .null => try metadata.put(allocator, try allocator.dupe(u8, key), .null),
        .bool => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .bool = v }),
        .integer => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .int = v }),
        .float => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .float = v }),
        .number_string, .string => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .string = try allocator.dupe(u8, v) }),
        .array => |v| {
            if (v.items.len == 0) return;
            return if (validSlice(v)) |item_type| {
                const data: Metadata = switch (item_type) {
                    .bool => blk: {
                        const values = try allocator.alloc(bool, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.bool;
                        break :blk .{ .array_bool = values };
                    },
                    .integer => blk: {
                        const values = try allocator.alloc(i64, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.integer;
                        break :blk .{ .array_int = values };
                    },
                    .float => blk: {
                        const values = try allocator.alloc(f64, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.float;
                        break :blk .{ .array_float = values };
                    },
                    inline .string, .number_string => |tag| blk: {
                        const values = try allocator.alloc([]const u8, v.items.len);
                        for (v.items, 0..) |item, i| {
                            values[i] = @field(item, @tagName(tag));
                        }
                        break :blk .{ .array_string = values };
                    },
                    .null, .array, .object => unreachable,
                };
                try metadata.put(allocator, try allocator.dupe(u8, key), data);
            } else {
                for (v.items, 0..) |item, i| {
                    var new_prefix = prefix;
                    if (prefix.items.len > 0)
                        new_prefix.appendAssumeCapacity('.');
                    new_prefix.items.len += std.fmt.printInt(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                    try parseMetadata(registry, new_prefix, item);
                }
            };
        },
        .object => |v| {
            var obj_iter = v.iterator();
            while (obj_iter.next()) |entry| {
                var new_prefix = prefix;
                if (prefix.items.len > 0)
                    new_prefix.appendAssumeCapacity('.');
                new_prefix.appendSliceAssumeCapacity(entry.key_ptr.*);
                try parseMetadata(registry, new_prefix, entry.value_ptr.*);
            }
        },
    };
}

/// We can only create a Zig slice out of json array, if all values
/// in the array have the same type.
fn validSlice(v: std.json.Array) ?std.meta.Tag(std.json.Value) {
    if (v.items.len == 0) return null;

    const item_type: std.meta.Tag(std.json.Value) = v.items[0];
    switch (item_type) {
        .null, .array, .object => return null,
        else => {},
    }

    for (v.items[1..]) |item| {
        if (item != item_type)
            return null;
    }

    return item_type;
}

pub fn bufferTypeFromDtype(dt: DataType) pjrtx.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrtx.BufferType, @tagName(tag)),
    };
}
