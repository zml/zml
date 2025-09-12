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
const pjrtx = zml.pjrt;
const pjrt = pjrtx.pjrt;

const BUF_1_KB = 1 * 1024;
const BUF_8_KB = 8 * 1024;
const BUF_16_KB = 16 * 1024;
const BUF_32_KB = 32 * 1024;
const BUF_64_KB = 64 * 1024;

// I/O Utilities
const io = struct {
    pub fn copy(
        reader: *std.io.Reader,
        writer: *std.io.Writer,
        pump_buffer: []u8,
    ) std.io.Reader.StreamRemainingError!u64 {
        var total_bytes_copied: u64 = 0;

        while (true) {
            const bytes_read = reader.readSliceShort(pump_buffer) catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            };

            if (bytes_read == 0) break;
            try writer.writeAll(pump_buffer[0..bytes_read]);
            total_bytes_copied += bytes_read;
        }

        return total_bytes_copied;
    }
};

// Primitives

const Tensor = struct {
    source_name: []const u8, // Logical name of the source file/chunk this tensor resides in.
    name: []const u8,
    shape: Shape,
    offset: u64,

    // todo: remove allocator and interface return
    pub fn reader(self: Tensor, allocator: std.mem.Allocator, source: *Source) !*std.io.Reader {
        var source_reader = try source.reader(self.source_name);
        try source_reader.discardAll(self.offset);

        const limited_reader_buffer = try allocator.alloc(u8, BUF_16_KB);
        const limited_reader = try allocator.create(LimitedReader);

        limited_reader.* = .init(
            source_reader,
            std.io.Limit.limited64(self.shape.byteSize()),
            limited_reader_buffer,
        );

        return &limited_reader.interface;
    }
};

pub const Registry = struct {
    pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);

    arena: std.heap.ArenaAllocator,

    tensors: Tensors,
    metadata: Metadatas,

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();

        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }
};

// Safetensors parser

pub fn registerSafetensors(allocator: std.mem.Allocator, source: *Source, path: []const u8) !Registry {
    var registry: Registry = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .tensors = .{},
        .metadata = .{},
    };
    errdefer registry.deinit();

    var processing_arena = std.heap.ArenaAllocator.init(allocator);
    defer processing_arena.deinit();

    const processing_allocator = processing_arena.allocator();

    const source_reader = try source.reader(path);

    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        try parseSafetensorsIndex(processing_allocator, &registry, source, source_reader);
    } else {
        try parseSafetensors(processing_allocator, &registry, source_reader, path);
    }

    return registry;
}

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: *Source,
    reader: *std.io.Reader,
) !void {
    var json_reader: std.json.Reader = .init(allocator, reader);
    const index = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, &json_reader, .{ .allocate = .alloc_if_needed });

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;
        const chunk_reader = try source.reader(filename);

        try parseSafetensors(allocator, registry, chunk_reader, filename);
    }

    if (index.object.get("__metadata__")) |metadata| {
        var prefix_buf: [BUF_1_KB]u8 = undefined;
        try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), metadata);
    }
}

fn parseSafetensors(
    allocator: std.mem.Allocator,
    registry: *Registry,
    reader: *std.io.Reader,
    source_name: []const u8,
) !void {
    const registry_allocator = registry.arena.allocator();
    const json_header_length: usize = @intCast(try reader.takeInt(u64, .little));
    const json_data = try allocator.alloc(u8, json_header_length);
    defer allocator.free(json_data);

    try reader.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{});

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
            .source_name = try registry_allocator.dupe(u8, source_name),
            .name = try registry_allocator.dupe(u8, key),
            .shape = shape,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(registry_allocator, key, tensor);
    }
}

// Source Providers

pub const Source = union(enum) {
    fs: *FsSource,
    // note: future sources could be added here

    pub fn reader(self: Source, path: []const u8) !*std.io.Reader {
        return switch (self) {
            .fs => |fs_source| fs_source.reader(path),
        };
    }
};

pub const FsSource = struct {
    const ManagedFile = struct {
        file: std.fs.File,
        buffer: *[BUF_16_KB]u8,
    };

    allocator: std.mem.Allocator,
    file_reader: std.fs.File.Reader,
    base_dir: []const u8,
    path_to_file_map: std.StringHashMapUnmanaged(ManagedFile),

    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) FsSource {
        return .{
            .allocator = allocator,
            .file_reader = undefined,
            .base_dir = base_dir,
            .path_to_file_map = .{},
        };
    }

    pub fn deinit(self: *FsSource) void {
        var it = self.path_to_file_map.iterator();

        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.file.close();
            self.allocator.destroy(entry.value_ptr.buffer);
        }

        self.path_to_file_map.deinit(self.allocator);
    }

    // todo: no alloc in reader
    pub fn reader(self: *FsSource, path: []const u8) !*std.io.Reader {
        const managed_file: ManagedFile = if (self.path_to_file_map.get(path)) |mf| mf else blk: {
            const full_path = try std.fs.path.join(self.allocator, &.{ self.base_dir, path });
            defer self.allocator.free(full_path);

            const file = try std.fs.openFileAbsolute(full_path, .{ .mode = .read_only });
            errdefer file.close();

            const buffer = try self.allocator.create([BUF_16_KB]u8);
            errdefer self.allocator.destroy(buffer);

            const path_dupe = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(path_dupe);

            try self.path_to_file_map.put(self.allocator, path_dupe, .{
                .file = file,
                .buffer = buffer,
            });

            break :blk self.path_to_file_map.get(path).?;
        };

        try managed_file.file.seekTo(0);

        self.file_reader = managed_file.file.reader(managed_file.buffer);

        return &self.file_reader.interface;
    }

    pub fn getFile(self: *FsSource, name: []const u8) !std.fs.File {
        return self.path_to_file_map.get(name).?.file;
    }
};

// Reusable Stream Processors

pub const LimitedReader = std.io.Reader.Limited;

pub const Quantizer = struct {
    pub const Writer = QuantizingWriter;

    target_bits: u8,

    pub fn writer(self: Quantizer, next: *std.io.Writer) Writer {
        return .init(next, self);
    }
};

const QuantizingWriter = struct {
    config: Quantizer,
    next_writer: *std.io.Writer,
    interface: std.io.Writer,

    pub fn init(next_writer: *std.io.Writer, config: Quantizer) QuantizingWriter {
        return .{
            .next_writer = next_writer,
            .config = config,
            .interface = .{ .vtable = &vtable, .buffer = &[_]u8{}, .end = 0 },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*QuantizingWriter, @alignCast(@fieldParentPtr("interface", w)));

        log.debug("Quantizing data to {d} bits...", .{self.config.target_bits});

        return self.next_writer.writeSplat(data, splat);
    }

    fn flush(w: *std.io.Writer) !void {
        const self = @as(*QuantizingWriter, @alignCast(@fieldParentPtr("interface", w)));
        try self.next_writer.flush();
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

pub const Checksumer = struct {
    pub const Writer = ChecksummingWriter;

    digest: *[32]u8,

    pub fn writer(self: Checksumer, next: *std.io.Writer) Writer {
        return .init(next, self);
    }
};

const ChecksummingWriter = struct {
    config: Checksumer,
    hasher: std.crypto.hash.sha2.Sha256,
    next_writer: *std.io.Writer,
    interface: std.io.Writer,

    pub fn init(next_writer: *std.io.Writer, config: Checksumer) ChecksummingWriter {
        return .{
            .config = config,
            .hasher = std.crypto.hash.sha2.Sha256.init(.{}),
            .next_writer = next_writer,
            .interface = .{ .vtable = &vtable, .buffer = &[_]u8{}, .end = 0 },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*ChecksummingWriter, @alignCast(@fieldParentPtr("interface", w)));

        for (data) |d| self.hasher.update(d);

        if (splat > 1 and data.len > 0) {
            const last = data[data.len - 1];
            for (0..splat - 1) |_| self.hasher.update(last);
        }

        return self.next_writer.writeSplat(data, splat);
    }

    fn flush(w: *std.io.Writer) !void {
        const self = @as(*ChecksummingWriter, @alignCast(@fieldParentPtr("interface", w)));

        self.hasher.final(self.config.digest);

        try self.next_writer.flush();
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

const MemoryWriter = struct {
    api: *const pjrt.Api,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,

    buffer_index: usize,
    tensor: Tensor,

    bytes_written: u64,
    is_flushed: bool = false,

    interface: std.io.Writer,

    pub fn init(
        api: *const pjrt.Api,
        transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,
        index: usize,
        tensor: Tensor,
    ) MemoryWriter {
        return .{
            .api = api,
            .transfer_manager = transfer_manager,
            .buffer_index = index,
            .tensor = tensor,
            .bytes_written = 0,
            .interface = .{ .vtable = &vtable, .buffer = &[_]u8{}, .end = 0 },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*MemoryWriter, @alignCast(@fieldParentPtr("interface", w)));

        var total_bytes_drained: usize = 0;

        for (data[0..data.len -| 1]) |chunk| {
            try self.transferChunk(chunk);
            total_bytes_drained += chunk.len;
        }

        if (data.len > 0) {
            const last_chunk = data[data.len - 1];
            for (0..splat) |_| {
                try self.transferChunk(last_chunk);
            }
            total_bytes_drained += last_chunk.len * splat;
        }

        return total_bytes_drained;
    }

    fn transferChunk(self: *MemoryWriter, chunk: []const u8) !void {
        if (chunk.len == 0) return;

        std.debug.assert(!self.is_flushed);
        std.debug.assert(self.bytes_written + chunk.len <= self.tensor.shape.byteSize());

        // Data transfers are never the last transfer. `flush` is.
        const is_last = false;

        // todo : revert to log.debug
        std.debug.print("Queueing data for '{s}' {d} (bytes) : offset={d} index={d}, chunk_size={d}, bytes_written={d}, is_last={}\n", .{
            self.tensor.name,
            self.tensor.shape.byteSize(),
            self.tensor.offset,
            self.buffer_index,
            chunk.len,
            self.bytes_written,
            is_last,
        });

        const event = self.transfer_manager.transferData(
            self.api,
            self.buffer_index,
            chunk,
            @intCast(self.bytes_written),
            is_last,
        ) catch |err| {
            log.err("[{s}] PJRT transferData failed to queue: {d} - {any}", .{ self.tensor.name, self.buffer_index, err });
            return error.WriteFailed;
        };
        _ = event; // autofix

        self.bytes_written += chunk.len;
    }

    fn flush(w: *std.io.Writer) !void {
        const self = @as(*MemoryWriter, @alignCast(@fieldParentPtr("interface", w)));

        if (self.is_flushed) return;

        self.is_flushed = true;

        if (self.bytes_written != self.tensor.shape.byteSize()) {
            log.err("[{s}] flush called but stream was incomplete. Wrote {d} of {d} bytes.", .{ self.tensor.name, self.bytes_written, self.tensor.shape.byteSize() });
            return error.WriteFailed;
        }

        // todo : revert to log.debug
        std.debug.print("Queueing finalization for '{s}' {d} (bytes) : index={d}, chunk_size={d}, bytes_written={d}, is_last={}\n", .{
            self.tensor.name,
            self.tensor.shape.byteSize(),
            self.buffer_index,
            0,
            self.bytes_written,
            true,
        });

        const event = self.transfer_manager.transferData(
            self.api,
            self.buffer_index,
            &[_]u8{},
            @intCast(self.bytes_written),
            true,
        ) catch |err| {
            log.err("[{s}] PJRT final transferData failed for tensor: {any}", .{ self.tensor.name, err });
            return error.WriteFailed;
        };
        _ = event; // autofix
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

const MemoryReader = struct {
    api: *const pjrt.Api,
    device_buffer: *const pjrtx.Buffer,

    bytes_read: u64,
    total_size: u64,

    interface: std.io.Reader,

    pub fn init(
        api: *const pjrt.Api,
        device_buffer: *const pjrtx.Buffer,
        host_buffer: []u8,
    ) !MemoryReader {
        return .{
            .api = api,
            .device_buffer = device_buffer,
            .bytes_read = 0,
            .total_size = try device_buffer.getOnDeviceSizeInBytes(api),
            .interface = .{
                .vtable = &vtable,
                .buffer = host_buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }

    fn fillBuffer(self: *MemoryReader) std.io.Reader.Error!void {
        const r = &self.interface;

        const unread_len = r.end - r.seek;

        if (unread_len > 0 and r.seek > 0) {
            @memmove(r.buffer[0..unread_len], r.buffer[r.seek..r.end]);
        }

        r.seek = 0;
        r.end = unread_len;

        if (self.bytes_read >= self.total_size) return;

        const remaining_on_device = self.total_size - self.bytes_read;
        const free_space_in_buffer = r.buffer.len - r.end;
        const bytes_to_fetch: usize = @min(remaining_on_device, free_space_in_buffer);

        if (bytes_to_fetch == 0) return;

        const dest_slice = r.buffer[r.end .. r.end + bytes_to_fetch];

        // todo: do not use inner
        const event = self.device_buffer.inner().copyRawToHost(self.api, dest_slice, @intCast(self.bytes_read)) catch |err| {
            log.err("PJRT copyRawToHost failed: {any}", .{err});
            return error.ReadFailed;
        };

        if (event) |e| {
            const event_: *pjrtx.Event = @ptrCast(e);
            event_.awaitBlocking(self.api) catch |err| {
                log.err("PJRT event await failed on copyRawToHost: {any}", .{err});
                return error.ReadFailed;
            };
        }

        self.bytes_read += bytes_to_fetch;
        r.end += @intCast(bytes_to_fetch);
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*MemoryReader, @alignCast(@fieldParentPtr("interface", r)));

        const buffered_data = limit.slice(r.buffered());

        if (buffered_data.len > 0) {
            const written = try w.write(buffered_data);
            r.toss(written);
            return written;
        }

        if (self.bytes_read >= self.total_size) {
            return error.EndOfStream;
        }

        try self.fillBuffer();

        const newly_buffered = limit.slice(r.buffered());
        if (newly_buffered.len == 0) {
            return error.EndOfStream;
        }

        const written = try w.write(newly_buffered);
        r.toss(written);

        return written;
    }

    fn readVec(r: *std.io.Reader, data: [][]u8) std.io.Reader.Error!usize {
        const self = @as(*MemoryReader, @alignCast(@fieldParentPtr("interface", r)));

        if (self.interface.seek == self.interface.end) {
            try self.fillBuffer();
        }

        var bytes_copied: usize = 0;

        for (data) |dest_slice| {
            const buffered = self.interface.buffered();
            if (buffered.len == 0) break;

            const amount_to_copy = @min(dest_slice.len, buffered.len);
            @memcpy(dest_slice[0..amount_to_copy], buffered[0..amount_to_copy]);

            self.interface.toss(amount_to_copy);
            bytes_copied += amount_to_copy;

            if (amount_to_copy < dest_slice.len) break;
        }

        if (bytes_copied == 0 and self.bytes_read >= self.total_size and self.interface.buffered().len == 0) {
            return error.EndOfStream;
        }

        return bytes_copied;
    }

    const vtable: std.io.Reader.VTable = .{ .stream = stream, .readVec = readVec };
};

const VerifyingWriter = struct {
    expected_data: []const u8,
    bytes_verified: u64 = 0,
    interface: std.io.Writer,

    pub fn init(expected: []const u8, buffer: []u8) VerifyingWriter {
        return .{
            .expected_data = expected,
            .interface = .{ .vtable = &vtable, .buffer = buffer, .end = 0 },
        };
    }

    fn verifyChunk(self: *VerifyingWriter, chunk: []const u8) !void {
        if (chunk.len == 0) return;

        const current_offset = self.bytes_verified;
        const next_offset = current_offset + chunk.len;

        if (next_offset > self.expected_data.len) {
            log.err(
                "Verification failed: received more data than expected. Expected total {d}, but received at least {d}",
                .{ self.expected_data.len, next_offset },
            );
            return error.WriteFailed;
        }

        const expected_chunk = self.expected_data[current_offset..next_offset];
        if (!std.mem.eql(u8, expected_chunk, chunk)) {
            log.err("Verification failed: data mismatch at offset {d} for chunk of size {d}", .{ current_offset, chunk.len });
            return error.WriteFailed;
        }
        self.bytes_verified = next_offset;
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*VerifyingWriter, @alignCast(@fieldParentPtr("interface", w)));

        try self.verifyChunk(w.buffered());
        w.end = 0;

        var bytes_from_data: usize = 0;
        for (data[0..data.len -| 1]) |chunk| {
            try self.verifyChunk(chunk);
            bytes_from_data += chunk.len;
        }

        if (data.len > 0) {
            const last_chunk = data[data.len - 1];
            for (0..splat) |_| {
                try self.verifyChunk(last_chunk);
            }
            bytes_from_data += last_chunk.len * splat;
        }

        return bytes_from_data;
    }

    fn flush(w: *std.io.Writer) !void {
        const self = @as(*VerifyingWriter, @alignCast(@fieldParentPtr("interface", w)));
        try self.verifyChunk(w.buffered());
        w.end = 0;

        if (self.bytes_verified != self.expected_data.len) {
            log.err("Verification failed: incomplete data. Expected {d} bytes, but received {d}", .{
                self.expected_data.len,
                self.bytes_verified,
            });
            return error.WriteFailed;
        }
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

// Sinks

pub const DiscardingSink = struct {
    discarder: std.io.Writer.Discarding,

    pub fn init() DiscardingSink {
        return .{ .discarder = .init(&[_]u8{}) };
    }

    pub fn writer(self: *DiscardingSink, tensor: Tensor) *std.io.Writer {
        _ = tensor;
        return @constCast(&self.discarder.writer);
    }
};

// Executor

const IoManager = struct {
    allocator: std.mem.Allocator,

    api: *const pjrt.Api,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,

    tensor_name_to_index: std.StringHashMapUnmanaged(usize),

    pub fn init(
        allocator: std.mem.Allocator,
        platform: Platform,
        memory: *const pjrt.Memory,
        tensors: []const Tensor,
    ) !IoManager {
        var self: IoManager = .{
            .allocator = allocator,
            .api = platform.pjrt_api,
            .transfer_manager = undefined,
            .tensor_name_to_index = .{},
        };
        errdefer self.deinit();

        var shape_specs_list = try std.ArrayList(pjrt.ShapeSpec).initCapacity(allocator, tensors.len);
        defer shape_specs_list.deinit(allocator);

        for (tensors, 0..) |*tensor, i| {
            shape_specs_list.appendAssumeCapacity(.init(tensor.shape.dims(), bufferTypeFromDtype(tensor.shape.dtype())));
            try self.tensor_name_to_index.put(allocator, try allocator.dupe(u8, tensor.name), i);
        }

        self.transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = shape_specs_list.items,
            .memory = memory,
        });

        return self;
    }

    pub fn deinit(self: *IoManager) void {
        self.transfer_manager.deinit(self.api);

        var it = self.tensor_name_to_index.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }

        self.tensor_name_to_index.deinit(self.allocator);
    }

    pub fn writer(self: *IoManager, tensor: Tensor) MemoryWriter {
        const index = self.tensor_name_to_index.get(tensor.name).?;

        return .init(
            self.api,
            self.transfer_manager,
            index,
            tensor,
        );
    }

    pub fn reader(self: *IoManager, tensor: Tensor, buffer: []u8) !MemoryReader {
        const index = self.tensor_name_to_index.get(tensor.name).?;
        const device_buffer = try self.transfer_manager.retrieveBuffer(self.api, index);

        return try .init(self.api, device_buffer, buffer);
    }
};

test "IoManager Integration" {
    const platform = zml.testing.env();
    const heap_allocator = std.testing.allocator;
    const device = platform.getDevices()[0];
    const memory = (try device.addressableMemories(platform.pjrt_api))[0];

    var arena_buffer: [BUF_16_KB]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&arena_buffer);
    const stack_allocator = fba.allocator();

    var tensors = [_]Tensor{
        .{ .source_name = "src", .name = "tensor_a", .shape = Shape.init(.{1024}, .f32), .offset = 0 },
        .{ .source_name = "src", .name = "tensor_b", .shape = Shape.init(.{ 1024, 4096 }, .f32), .offset = 0 },
        .{ .source_name = "src", .name = "tensor_c", .shape = Shape.init(.{8192}, .f32), .offset = 0 },
        .{ .source_name = "src", .name = "tensor_d", .shape = Shape.init(.{ 2048, 4096 }, .bf16), .offset = 0 },
        .{ .source_name = "src", .name = "zero_byte_tensor", .shape = Shape.init(.{0}, .f32), .offset = 0 },
    };

    var blob_size: usize = 0;

    for (tensors) |tensor| {
        blob_size += tensor.shape.byteSize();
    }

    const blob = try heap_allocator.alloc(u8, blob_size);
    defer heap_allocator.free(blob);

    var current_offset: u64 = 0;
    for (&tensors, 0..) |*tensor, i| {
        const tensor_size = tensor.shape.byteSize();
        tensor.offset = current_offset;

        const tensor_slice = blob[tensor.offset .. tensor.offset + tensor_size];
        @memset(tensor_slice, @intCast(0xAA + i));

        current_offset += tensor_size;
    }

    var io_manager = try IoManager.init(stack_allocator, platform, memory, &tensors);
    defer io_manager.deinit();

    std.debug.print("--- Writing tensors... ---\n", .{});

    var pump_buffer: [BUF_8_KB]u8 = undefined;

    for (tensors) |tensor| {
        var reader: std.io.Reader = .fixed(blob[tensor.offset .. tensor.offset + tensor.shape.byteSize()]);
        var writer = io_manager.writer(tensor);
        var total_bytes_copied: u64 = 0;

        while (true) {
            const bytes_read = reader.readSliceShort(&pump_buffer) catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            };

            if (bytes_read == 0) break;

            try writer.interface.writeAll(pump_buffer[0..bytes_read]);

            total_bytes_copied += bytes_read;
        }

        try std.testing.expectEqual(tensor.shape.byteSize(), total_bytes_copied);
        try writer.interface.flush();
    }

    std.debug.print("--- Reading and verifying tensors... ---\n", .{});

    var reader_buffer: [BUF_16_KB]u8 = undefined;
    var verification_buffer: [BUF_8_KB]u8 = undefined;

    for (tensors) |tensor| {
        const original_data = blob[tensor.offset .. tensor.offset + tensor.shape.byteSize()];

        var reader = try io_manager.reader(tensor, &reader_buffer);
        var bytes_verified: u64 = 0;

        while (bytes_verified < original_data.len) {
            const bytes_to_read = @min(verification_buffer.len, original_data.len - bytes_verified);
            const chunk_to_verify = verification_buffer[0..bytes_to_read];

            try reader.interface.readSliceAll(chunk_to_verify);

            const original_chunk = original_data[bytes_verified .. bytes_verified + chunk_to_verify.len];
            try std.testing.expectEqualSlices(u8, original_chunk, chunk_to_verify);

            bytes_verified += chunk_to_verify.len;
        }

        const final_read = reader.interface.readSliceShort(&verification_buffer) catch |err| {
            try std.testing.expect(err == error.EndOfStream);
            break;
        };

        try std.testing.expectEqual(0, final_read);
        try std.testing.expectEqual(original_data.len, bytes_verified);
        std.debug.print("Verification successful for tensor '{s}'\n", .{tensor.name});
    }
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.next().?;

    const file_path = args.next() orelse {
        log.err("Usage: bazel run //examples/loader /path/to/model.safetensors or /path/to/model.safetensors.index.json", .{});
        return;
    };

    var context = try Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    log.info("--- Loading model metadata... ---", .{});
    var fs_source = FsSource.init(allocator, std.fs.path.dirname(file_path) orelse ".");
    defer fs_source.deinit();

    var source: Source = .{ .fs = &fs_source };

    var registry = try registerSafetensors(allocator, &source, std.fs.path.basename(file_path));
    defer registry.deinit();

    log.info("Registry loaded with {d} tensors from {d} source files.", .{ registry.tensors.count(), fs_source.path_to_file_map.count() });

    log.info("--- Planning transfers ---", .{});

    const device = platform.getDevices()[0];
    const memory = (try device.addressableMemories(platform.pjrt_api))[0];

    var io_manager = try IoManager.init(
        allocator,
        platform,
        memory,
        registry.tensors.values(),
    );
    defer io_manager.deinit();

    log.info("IO Manager created for {d} tensors.", .{io_manager.tensor_name_to_index.count()});

    log.info("--- Starting tensor processing stream... ---", .{});

    var arena_buffer: [BUF_64_KB]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&arena_buffer);

    var pump_buffer: [BUF_32_KB]u8 = undefined;

    var digest: [32]u8 = undefined;

    var tensor_it = registry.tensors.iterator();

    while (tensor_it.next()) |entry| {
        fba.reset();
        const stack_allocator = fba.allocator();

        const tensor = entry.value_ptr.*;
        log.info("--- Processing tensor: {s} ---", .{tensor.name});

        const reader = try tensor.reader(stack_allocator, &source);

        _ = &digest;

        const pipeline_stages = .{
            Quantizer{ .target_bits = 8 },
            // Checksumer{ .digest = &digest },
        };

        var final_memory_writer = io_manager.writer(tensor);
        var processor_chain: *std.io.Writer = &final_memory_writer.interface;

        inline for (pipeline_stages) |stage_config| {
            const WriterType = @TypeOf(stage_config).Writer;
            const new_writer = try stack_allocator.create(WriterType);
            new_writer.* = stage_config.writer(processor_chain);
            processor_chain = &new_writer.interface;
            log.info("  [Pipeline] Added '{s}' to writer chain", .{@typeName(WriterType)});
        }

        var total_bytes_copied: u64 = 0;

        while (true) {
            const bytes_read = reader.readSliceShort(&pump_buffer) catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            };
            if (bytes_read == 0) break;

            try processor_chain.writeAll(pump_buffer[0..bytes_read]);
            total_bytes_copied += bytes_read;
        }

        try processor_chain.flush();

        std.debug.assert(total_bytes_copied == tensor.shape.byteSize());

        log.info("--- Finished tensor: {s} ({d} bytes) ---", .{ tensor.name, total_bytes_copied });
    }

    log.info("--- All tensors processed successfully. ---", .{});
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
