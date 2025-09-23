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
const BUF_8_MB = 8 * MB;
const BUF_16_MB = 16 * MB;
const BUF_32_MB = 32 * MB;
const BUF_64_MB = 64 * MB;
const BUF_128_MB = 128 * MB;
const BUF_256_MB = 256 * MB;

var tracer: Tracer = undefined;

// I/O Utilities
// todo" check usage
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
};

pub const Registry = struct {
    pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);
    pub const Checksums = std.StringArrayHashMapUnmanaged([32]u8);

    arena: std.heap.ArenaAllocator,

    tensors: Tensors,
    metadata: Metadatas,
    checksums: Checksums,

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();

        self.checksums.deinit(allocator);
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
        .checksums = .{},
    };
    errdefer registry.deinit();

    var processing_arena = std.heap.ArenaAllocator.init(allocator);
    defer processing_arena.deinit();

    const processing_allocator = processing_arena.allocator();

    var io_buffer: [BUF_64_KB]u8 = undefined;
    var source_reader: Source.Reader = undefined;

    try source.initReader(path, &io_buffer, &source_reader);
    const source_iface = sourceInterface(&source_reader);

    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        try parseSafetensorsIndex(
            processing_allocator,
            &registry,
            source,
            &source_reader,
            source_iface,
            &io_buffer,
        );
    } else {
        try parseSafetensors(processing_allocator, &registry, source_iface, path);
    }

    return registry;
}

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: *Source,
    source_reader: *Source.Reader,
    io_reader: *std.io.Reader,
    io_buffer: []u8,
) !void {
    var json_reader: std.json.Reader = .init(allocator, io_reader);
    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;

        try source.initReader(filename, io_buffer, source_reader);
        const chunk_reader = sourceInterface(source_reader);

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

    pub const Reader = union(enum) {
        fs: std.fs.File.Reader,
        direct_fs: DirectFileReader,
    };

    pub fn initReader(self: Source, path: []const u8, buffer: []u8, reader_mem: *Reader) !void {
        return switch (self) {
            .fs => |fs_source| try fs_source.initReader(path, buffer, reader_mem),
        };
    }
};

pub fn sourceInterface(reader: *Source.Reader) *std.io.Reader {
    return switch (reader.*) {
        .fs => |*r| &r.interface,
        .direct_fs => |*r| &r.interface,
    };
}
pub const FsSource = struct {
    const ManagedFile = struct {
        file: std.fs.File,
    };

    allocator: std.mem.Allocator,
    base_dir: []const u8,
    path_to_file_map: std.StringHashMapUnmanaged(ManagedFile),
    path_to_direct_file_map: std.StringHashMapUnmanaged(ManagedFile),

    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) FsSource {
        return .{
            .allocator = allocator,
            .base_dir = base_dir,
            .path_to_file_map = .{},
            .path_to_direct_file_map = .{},
        };
    }

    pub fn deinit(self: *FsSource) void {
        var it = self.path_to_file_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.file.close();
        }
        self.path_to_file_map.deinit(self.allocator);

        var it2 = self.path_to_direct_file_map.iterator();
        while (it2.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.file.close();
        }
        self.path_to_direct_file_map.deinit(self.allocator);
    }

    pub fn initReader(self: *FsSource, path: []const u8, buffer: []u8, source_reader: *Source.Reader) !void {
        const managed_file = if (self.path_to_file_map.get(path)) |mf| mf else blk: {
            const full_path = try std.fs.path.join(self.allocator, &.{ self.base_dir, path });
            defer self.allocator.free(full_path);

            const file_fd = try std.posix.open(
                full_path,
                .{ .ACCMODE = .RDONLY, .DIRECT = false },
                0,
            );

            const file: std.fs.File = .{ .handle = file_fd };
            errdefer file.close();

            const path_dupe = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(path_dupe);

            try self.path_to_file_map.put(self.allocator, path_dupe, .{ .file = file });

            break :blk self.path_to_file_map.get(path).?;
        };

        try managed_file.file.seekTo(0);

        source_reader.* = .{ .fs = managed_file.file.reader(buffer) };
    }

    pub fn initDirectReader(self: *FsSource, path: []const u8, buffer: []u8, source_reader: *Source.Reader) !void {
        const managed_file = if (self.path_to_direct_file_map.get(path)) |mf| mf else blk: {
            const full_path = try std.fs.path.join(self.allocator, &.{ self.base_dir, path });
            defer self.allocator.free(full_path);

            const file_fd = try std.posix.open(
                full_path,
                .{ .ACCMODE = .RDONLY, .DIRECT = true },
                0,
            );

            const file: std.fs.File = .{ .handle = file_fd };
            errdefer file.close();

            const path_dupe = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(path_dupe);

            try self.path_to_direct_file_map.put(self.allocator, path_dupe, .{ .file = file });

            break :blk self.path_to_direct_file_map.get(path).?;
        };

        try managed_file.file.seekTo(0);

        source_reader.* = .{ .direct_fs = .init(managed_file.file, buffer) };
    }
};

test FsSource {
    const allocator = std.testing.allocator;
    const temp_dir = std.testing.tmpDir(.{});

    const file_name = "weights.json";

    var file = try temp_dir.dir.createFile(file_name, .{});
    defer file.close();

    const file_path = try temp_dir.dir.realpathAlloc(std.testing.allocator, file_name);
    defer std.testing.allocator.free(file_path);

    const data: [BUF_32_MB]u8 = undefined;
    var io_buffer: [BUF_8_KB]u8 = undefined;

    var writer_buffer: [BUF_16_KB]u8 = undefined;

    var file_writer = file.writer(&writer_buffer);
    try file_writer.interface.writeAll(&data);
    try file_writer.interface.flush();

    var fs_source = FsSource.init(allocator, std.fs.path.dirname(file_path).?);
    defer fs_source.deinit();

    var source = Source{ .fs = &fs_source };

    var reader_instance: Source.Reader = undefined;

    try source.initReader(file_name, &io_buffer, &reader_instance);
    const reader = sourceInterface(&reader_instance);

    var reader_buffer: [BUF_32_KB]u8 = undefined;
    var offset: usize = 0;

    while (true) {
        const bytes_read = try reader.readSliceShort(&reader_buffer);

        if (bytes_read == 0) break;

        try std.testing.expectEqualSlices(u8, data[offset .. offset + bytes_read], reader_buffer[0..bytes_read]);
        offset += bytes_read;
    }

    try std.testing.expectEqual(offset, data.len);
}

// Reusable Stream Processors

pub const Checksumer = struct {
    pub const Writer = ChecksummingWriter;

    registry: *Registry,
    tensor_name: []const u8,

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

    fn flush(w: *std.io.Writer) std.io.Writer.Error!void {
        const self = @as(*ChecksummingWriter, @alignCast(@fieldParentPtr("interface", w)));

        var digest: [32]u8 = undefined;
        self.hasher.final(&digest);

        const allocator = self.config.registry.arena.allocator();

        self.config.registry.checksums.put(allocator, self.config.tensor_name, digest) catch |err| {
            log.err("Failed to store checksum for tensor '{s}': {any}", .{ self.config.tensor_name, err });
            return error.WriteFailed;
        };

        try self.next_writer.flush();
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

const Verifier = struct {
    pub const Writer = ChecksumVerifyingWriter;

    registry: *const Registry,
    tensor_name: []const u8,

    pub fn writer(self: Verifier) Writer {
        return .init(self);
    }
};

const ChecksumVerifyingWriter = struct {
    config: Verifier,
    hasher: std.crypto.hash.sha2.Sha256,

    interface: std.io.Writer,

    pub fn init(config: Verifier) ChecksumVerifyingWriter {
        return .{
            .config = config,
            .hasher = std.crypto.hash.sha2.Sha256.init(.{}),
            .interface = .{ .vtable = &vtable, .buffer = &[_]u8{}, .end = 0 },
        };
    }
    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*ChecksumVerifyingWriter, @alignCast(@fieldParentPtr("interface", w)));

        var total_written: usize = 0;

        for (data) |d| {
            self.hasher.update(d);
            total_written += d.len;
        }

        if (splat > 1 and data.len > 0) {
            const last = data[data.len - 1];

            for (0..splat - 1) |_| {
                self.hasher.update(last);
            }

            total_written += last.len * (splat - 1);
        }

        return total_written;
    }
    fn flush(w: *std.io.Writer) !void {
        const self = @as(*ChecksumVerifyingWriter, @alignCast(@fieldParentPtr("interface", w)));

        var actual_checksum: [32]u8 = undefined;

        self.hasher.final(&actual_checksum);

        const expected_checksum = self.config.registry.checksums.get(self.config.tensor_name) orelse {
            log.err("Checksum not found for tensor '{s}' during verification", .{self.config.tensor_name});
            return error.WriteFailed;
        };

        if (!std.mem.eql(u8, &expected_checksum, &actual_checksum)) {
            log.err("Checksum MISMATCH for tensor '{s}. Expected {x} got {x}'!", .{ self.config.tensor_name, &expected_checksum, &actual_checksum });

            return error.WriteFailed;
        }

        log.info("Tensor '{s}' verified successfully.", .{self.config.tensor_name});
    }

    const vtable: std.io.Writer.VTable = .{ .drain = drain, .flush = flush };
};

const MemoryWriter = struct {
    const InflightRequest = struct {
        event: *pjrtx.Event,
        buffer_idx: u16,
    };

    allocator: std.mem.Allocator,
    platform: Platform,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,

    // Buffer pool resources (owned by this struct)
    buffers: [][]u8,
    free_queue: std.ArrayList(u16),
    inflight_queue: std.ArrayList(InflightRequest),

    pub fn init(
        allocator: std.mem.Allocator,
        platform: Platform,
        transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,
        num_buffers: u16,
        buffer_size: usize,
    ) !MemoryWriter {
        var self: MemoryWriter = .{
            .allocator = allocator,
            .platform = platform,
            .transfer_manager = transfer_manager,
            .buffers = try allocator.alloc([]u8, num_buffers),
            .free_queue = .{},
            .inflight_queue = .{},
        };
        errdefer self.deinit();

        try self.free_queue.ensureTotalCapacity(allocator, num_buffers);
        try self.inflight_queue.ensureTotalCapacity(allocator, num_buffers);

        for (0..num_buffers) |i| {
            self.buffers[i] = try allocator.alloc(u8, buffer_size);
            errdefer allocator.free(self.buffers[i]);
            try platform.pjrt_client.dmaMap(platform.pjrt_api, self.buffers[i]);
            self.free_queue.appendAssumeCapacity(@intCast(i));
        }
        return self;
    }

    pub fn deinit(self: *MemoryWriter) void {
        _ = self.flushAllBlocking() catch unreachable;
        for (self.buffers) |buf| {
            self.platform.pjrt_client.dmaUnmap(self.platform.pjrt_api, buf) catch unreachable;
            self.allocator.free(buf);
        }
        self.allocator.free(self.buffers);
        self.free_queue.deinit(self.allocator);
        self.inflight_queue.deinit(self.allocator);
    }

    fn checkCompletions(self: *MemoryWriter) !void {
        const trace = tracer.frameStart("MemoryWriter checkCompletions");
        defer tracer.frameEnd(trace, "MemoryWriter checkCompletions");

        var i: usize = 0;
        while (i < self.inflight_queue.items.len) {
            const req = self.inflight_queue.items[i];
            if (req.event.isReady(self.platform.pjrt_api)) {
                if (req.event.getEventError(self.platform.pjrt_api)) |e| {
                    log.err("PJRT transfer event failed", .{});
                    e.deinit(self.platform.pjrt_api);
                    return error.WriteFailed;
                }
                req.event.deinit(self.platform.pjrt_api);
                _ = self.inflight_queue.swapRemove(i);
                self.free_queue.appendAssumeCapacity(req.buffer_idx);
            } else {
                i += 1;
            }
        }
    }

    pub fn getFreeBuffer(self: *MemoryWriter) ![]u8 {
        const trace = tracer.frameStart("MemoryWriter getFreeBuffer");
        defer tracer.frameEnd(trace, "MemoryWriter getFreeBuffer");

        try self.checkCompletions();
        if (self.free_queue.items.len > 0) {
            return self.buffers[self.free_queue.pop().?];
        }

        while (self.free_queue.items.len == 0) {
            if (self.inflight_queue.items.len == 0) return error.NoMoreBuffers;
            const oldest_req = self.inflight_queue.items[0];
            try oldest_req.event.awaitBlocking(self.platform.pjrt_api);
            try self.checkCompletions();
        }
        return self.buffers[self.free_queue.pop().?];
    }

    fn bufferIndexFromSlice(self: *const MemoryWriter, slice: []u8) u16 {
        for (self.buffers, 0..) |buf, i| {
            if (buf.ptr == slice.ptr) {
                return @intCast(i);
            }
        }
        unreachable;
    }

    pub fn submitBuffer(self: *MemoryWriter, buffer: []u8, buffer_index: usize, bytes_submitted: u64, tensor_byte_size: u64) !void {
        const trace = tracer.frameStart("MemoryWriter submitBuffer");
        defer tracer.frameEnd(trace, "MemoryWriter submitBuffer");

        const buffer_idx = self.bufferIndexFromSlice(buffer);
        const is_last_chunk = (bytes_submitted + buffer.len) >= tensor_byte_size;

        const event = try self.transfer_manager.transferData(
            self.platform.pjrt_api,
            buffer_index,
            buffer,
            @intCast(bytes_submitted),
            is_last_chunk,
        );

        self.inflight_queue.appendAssumeCapacity(.{ .event = event, .buffer_idx = buffer_idx });
    }

    pub fn hasInflight(self: *const MemoryWriter) bool {
        return self.inflight_queue.items.len > 0;
    }

    pub fn blockOnOldest(self: *MemoryWriter) !void {
        const trace = tracer.frameStart("MemoryWriter blockOnOldest");
        defer tracer.frameEnd(trace, "MemoryWriter blockOnOldest");

        if (self.inflight_queue.items.len > 0) {
            const oldest_req = self.inflight_queue.items[0];
            try oldest_req.event.awaitBlocking(self.platform.pjrt_api);
        }
    }

    pub fn flushAllBlocking(self: *MemoryWriter) !void {
        const trace = tracer.frameStart("MemoryWriter flushAllBlocking");
        defer tracer.frameEnd(trace, "MemoryWriter flushAllBlocking ");

        while (self.inflight_queue.items.len > 0) {
            const oldest_req = self.inflight_queue.items[0];
            try oldest_req.event.awaitBlocking(self.platform.pjrt_api);
            try self.checkCompletions();
        }
    }
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

        const event = self.device_buffer.copyRawToHost(self.api, dest_slice, @intCast(self.bytes_read)) catch |err| {
            log.err("PJRT copyRawToHost failed: {any}", .{err});
            return error.ReadFailed;
        };

        if (event) |e| {
            e.awaitBlocking(self.api) catch |err| {
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

const DirectFileReader = struct {
    file: std.fs.File,
    pos: u64,

    interface: std.io.Reader,

    pub const block_size = 4096; // todo: from zig std?

    pub fn init(file: std.fs.File, buffer: []u8) DirectFileReader {
        std.debug.assert(buffer.len % block_size == 0);
        std.debug.assert(@intFromPtr(buffer.ptr) % block_size == 0);

        return .{
            .file = file,
            .pos = 0,
            .interface = .{
                .vtable = &vtable,
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };
    }

    fn fillBuffer(self: *DirectFileReader) std.io.Reader.Error!void {
        const r = &self.interface;
        const unread_len = r.end - r.seek;

        if (unread_len > 0 and r.seek > 0) {
            @memmove(r.buffer[0..unread_len], r.buffer[r.seek..r.end]);
        }
        r.seek = 0;
        r.end = unread_len;

        const free_space = r.buffer.len - r.end;
        if (free_space < block_size) return;

        const read_offset = (self.pos + r.end) / block_size * block_size;
        const read_size = (free_space / block_size) * block_size;
        if (read_size == 0) return;

        const dest_slice = r.buffer[r.end .. r.end + read_size];
        const bytes_read = std.posix.pread(self.file.handle, dest_slice, @intCast(read_offset)) catch |err| {
            log.err("O_DIRECT pread failed: {any}", .{err});
            return error.ReadFailed;
        };

        if (bytes_read == 0) return;

        const prefix_to_skip = (self.pos + r.end) - read_offset;

        r.seek = prefix_to_skip;
        r.end += bytes_read;
    }

    fn discard(r: *std.io.Reader, limit: std.io.Limit) std.io.Reader.Error!usize {
        const self = @as(*DirectFileReader, @alignCast(@fieldParentPtr("interface", r)));

        const buffered_len = r.end - r.seek;
        const requested_discard = limit.minInt(std.math.maxInt(usize));

        if (requested_discard <= buffered_len) {
            r.seek += requested_discard;
            self.pos += requested_discard;
            return requested_discard;
        }

        const to_discard_after_buffer = requested_discard - buffered_len;

        self.pos += buffered_len + to_discard_after_buffer;
        r.seek = 0;
        r.end = 0;

        return requested_discard;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*DirectFileReader, @alignCast(@fieldParentPtr("interface", r)));

        const buffered_data = limit.slice(r.buffered());

        if (buffered_data.len > 0) {
            const written = try w.write(buffered_data);
            r.toss(written);
            self.pos += written;
            return written;
        }

        try self.fillBuffer();
        const newly_buffered = limit.slice(r.buffered());

        if (newly_buffered.len == 0) return error.EndOfStream;

        const written = try w.write(newly_buffered);

        r.toss(written);
        self.pos += written;

        return written;
    }

    fn readVec(r: *std.io.Reader, data: [][]u8) std.io.Reader.Error!usize {
        const self = @as(*DirectFileReader, @alignCast(@fieldParentPtr("interface", r)));

        if (r.seek == r.end) {
            try self.fillBuffer();
        }

        var bytes_copied: usize = 0;
        for (data) |dest_slice| {
            const buffered = r.buffered();

            if (buffered.len == 0) break;

            const amount_to_copy = @min(dest_slice.len, buffered.len);

            @memcpy(dest_slice[0..amount_to_copy], buffered[0..amount_to_copy]);

            r.toss(amount_to_copy);
            self.pos += amount_to_copy;
            bytes_copied += amount_to_copy;

            if (amount_to_copy < dest_slice.len) break;
        }

        if (bytes_copied == 0 and r.buffered().len == 0) {
            return error.EndOfStream;
        }

        return bytes_copied;
    }

    const vtable: std.io.Reader.VTable = .{ .stream = stream, .discard = discard, .readVec = readVec };
};

// Executor

const Shards = std.ArrayListUnmanaged(Shard);
const Shard = struct {
    tensor: *const Tensor,
    shape: Shape, // Shape of the shard
    offset: usize,
    device: *const pjrt.Device,

    pub fn byteSize(self: Shard) usize {
        return self.shape.byteSize();
    }
};

fn planShards(
    allocator: std.mem.Allocator,
    tensors: []const Tensor,
    devices: []const *const pjrt.Device,
) !Shards {
    var shards: Shards = .{};
    errdefer shards.deinit(allocator);

    const num_devices = devices.len;

    for (tensors) |*tensor| {
        const sharding_info = tensor.shape._sharding_info;
        const sharded_axes_count = std.simd.countTrues(sharding_info);

        if (sharded_axes_count == 0) {
            // Not sharded, replicate to all devices.
            for (devices) |device| {
                try shards.append(allocator, .{
                    .tensor = tensor,
                    .offset = tensor.offset,
                    .shape = tensor.shape,
                    .device = device,
                });
            }
        } else {
            // Sharded. For now, we only support sharding on one axis.
            std.debug.assert(sharded_axes_count == 1);
            const sharded_axis = std.simd.firstIndexOfValue(sharding_info, true) orelse unreachable;

            const original_dim: usize = @intCast(tensor.shape.dim(sharded_axis));
            std.debug.assert(original_dim % num_devices == 0);
            const shard_dim = original_dim / num_devices;

            var shard_shape = tensor.shape;
            shard_shape._dims.set(sharded_axis, @intCast(shard_dim));

            const total_size = tensor.shape.byteSize();
            const shard_size = total_size / num_devices;
            std.debug.assert(shard_size == shard_shape.byteSize());

            for (devices, 0..) |device, i| {
                try shards.append(allocator, .{
                    .tensor = tensor,
                    .offset = tensor.offset + i * shard_size,
                    .shape = shard_shape,
                    .device = device,
                });
            }
        }
    }

    return shards;
}

const TransferManagerIterator = struct {
    manager: *TransferManager,
    index: usize,

    pub fn next(self: *TransferManagerIterator) ?TransferManager.Item {
        if (self.index >= self.manager.items.items.len) return null;
        const item = self.manager.items.items[self.index];
        self.index += 1;
        return item;
    }

    pub fn reset(self: *TransferManagerIterator) void {
        self.index = 0;
    }
};

const TransferManager = struct {
    pub const Items = std.ArrayListUnmanaged(Item);
    const TransferManagers = std.ArrayListUnmanaged(*pjrtx.AsyncHostToDeviceTransferManager);
    const MemoryWriters = std.ArrayListUnmanaged(*MemoryWriter);

    pub const Item = struct {
        shard: *const Shard,
        memory_writer: *MemoryWriter,
        buffer_index: usize,

        pub fn byteSize(self: Item) usize {
            return self.shard.byteSize();
        }
    };

    allocator: std.mem.Allocator,
    api: *const pjrt.Api,

    transfer_managers: TransferManagers,
    memory_writers: MemoryWriters,
    items: Items,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: Platform,
        dest_memory: pjrt.Memory.Kind,
        shards: Shards,
        staging_queue_depth: u16,
        staging_buffer_size: usize,
    ) !TransferManager {
        var self: TransferManager = .{
            .allocator = allocator,
            .api = platform.pjrt_api,
            .transfer_managers = .{},
            .memory_writers = .{},
            .items = .{},
        };
        errdefer self.deinit();

        const devices = platform.getDevices();
        const num_devices = devices.len;

        var shards_per_device = try allocator.alloc(std.ArrayListUnmanaged(*const Shard), num_devices);
        defer allocator.free(shards_per_device);

        for (0..num_devices) |i| shards_per_device[i] = .{};
        defer for (0..num_devices) |i| shards_per_device[i].deinit(allocator);

        for (shards.items) |*shard_info| {
            const device_idx = shard_info.device.getLocalHardwareId(self.api); // assumes devices are indexed 0..N
            try shards_per_device[device_idx].append(allocator, shard_info);
        }

        try self.items.ensureTotalCapacity(allocator, shards.items.len);

        for (devices, 0..) |device, device_idx| {
            const device_shard_infos = shards_per_device[device_idx].items;
            if (device_shard_infos.len == 0) {
                continue;
            }

            var shape_specs = try std.ArrayList(pjrt.ShapeSpec).initCapacity(allocator, device_shard_infos.len);
            defer shape_specs.deinit(allocator);

            for (device_shard_infos) |info| {
                shape_specs.appendAssumeCapacity(.init(info.shape.dims(), bufferTypeFromDtype(info.shape.dtype())));
            }

            const memories = try device.addressableMemories(platform.pjrt_api);
            var memory = memories[0];

            if (platform.target == .cuda) {
                for (memories) |mem| {
                    if (mem.kind(platform.pjrt_api) == dest_memory) {
                        memory = mem;
                        break;
                    }
                }
            }

            const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
                .shape_specs = shape_specs.items,
                .memory = memory,
            });
            errdefer transfer_manager.deinit(self.api);

            const memory_writer = try allocator.create(MemoryWriter);
            errdefer allocator.destroy(memory_writer);
            memory_writer.* = try MemoryWriter.init(allocator, platform, transfer_manager, staging_queue_depth, staging_buffer_size);

            try self.transfer_managers.append(allocator, transfer_manager);
            try self.memory_writers.append(allocator, memory_writer);

            for (device_shard_infos, 0..) |info, buffer_idx| {
                self.items.appendAssumeCapacity(.{
                    .shard = info,
                    .memory_writer = memory_writer,
                    .buffer_index = buffer_idx,
                });
            }
        }

        std.sort.block(Item, self.items.items, {}, struct {
            fn lessThan(_: void, a: Item, b: Item) bool {
                const name_cmp = std.mem.order(u8, a.shard.tensor.source_name, b.shard.tensor.source_name);
                if (name_cmp == .eq) return a.shard.offset < b.shard.offset;
                return name_cmp == .lt;
            }
        }.lessThan);

        return self;
    }

    pub fn deinit(self: *TransferManager) void {
        for (self.transfer_managers.items) |transfer_manager| {
            transfer_manager.deinit(self.api);
        }
        for (self.memory_writers.items) |writer| {
            writer.deinit();
            self.allocator.destroy(writer);
        }
        self.memory_writers.deinit(self.allocator);
        self.transfer_managers.deinit(self.allocator);
        self.items.deinit(self.allocator);
    }

    pub fn iterator(self: *TransferManager) TransferManagerIterator {
        return .{ .manager = self, .index = 0 };
    }
};

test TransferManager {
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

    var transfer_manager = try TransferManager.init(stack_allocator, platform, memory, &tensors);
    defer transfer_manager.deinit();

    std.debug.print("--- Writing tensors... ---\n", .{});

    var pump_buffer: [BUF_8_KB]u8 = undefined;

    for (tensors) |tensor| {
        var reader: std.io.Reader = .fixed(blob[tensor.offset .. tensor.offset + tensor.shape.byteSize()]);
        var writer = transfer_manager.writer(tensor);
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

        var reader = try transfer_manager.reader(tensor, &reader_buffer);
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

    const verify_checksums = true;
    _ = verify_checksums; // autofix

    var context = try Context.init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.examples.loader");

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    log.info("--- Loading model metadata... ---", .{});
    var fs_source = FsSource.init(allocator, std.fs.path.dirname(file_path) orelse ".");
    defer fs_source.deinit();

    var source: Source = .{ .fs = &fs_source };
    var registry = try registerSafetensors(allocator, &source, std.fs.path.basename(file_path));
    defer registry.deinit();

    log.info("Registry loaded with {d} tensors.", .{registry.tensors.count()});

    log.info("--- Applying sharding information... ---", .{});
    try addExampleShardingMetadata(&registry);
    try annotateShapesWithSharding(&registry);

    log.info("--- Planning transfers ---", .{});
    var shards = try planShards(allocator, registry.tensors.values(), platform.getDevices());
    defer shards.deinit(allocator);

    const STAGING_QUEUE_DEPTH = 20;
    const STAGING_BUFFER_SIZE = BUF_8_MB;

    var transfer_manager = try TransferManager.init(allocator, platform, .device, shards, STAGING_QUEUE_DEPTH, STAGING_BUFFER_SIZE);
    defer transfer_manager.deinit();

    var itt = transfer_manager.iterator();

    while (itt.next()) |item| {
        log.info(
            "Planned shard: {s}/{s} -> dev {d} (idx {d}). Offset: {d}, Size: {d}, Shape: {any}. Original tensor shape: {any}",
            .{
                item.shard.tensor.source_name,
                item.shard.tensor.name,
                item.shard.device.getLocalHardwareId(platform.pjrt_api),
                item.buffer_index,
                item.shard.offset,
                item.byteSize(),
                item.shard.shape.dims(),
                item.shard.tensor.shape.dims(),
            },
        );
    }

    log.info("--- Starting tensor processing stream... ---", .{});

    const FILE_IO_BUFFER_SIZE = BUF_128_MB;
    const file_io_buffer = try allocator.alloc(u8, FILE_IO_BUFFER_SIZE);
    defer allocator.free(file_io_buffer);

    var shards_it = transfer_manager.iterator();
    var current_source_name: ?[]const u8 = null;
    var source_reader: Source.Reader = undefined;
    var reader_iface: *std.io.Reader = undefined;
    var file_pos: u64 = 0;

    var timer = try std.time.Timer.start();
    var sum_total_bytes_copied: u64 = 0;

    while (shards_it.next()) |item| {
        const trace = tracer.frameStart("Processing shard");
        defer tracer.frameEnd(trace, "Processing shard");

        const shard = item.shard;
        if (shard.byteSize() == 0) continue;

        log.info("Processing shard for tensor: {s} (device {d})", .{ shard.tensor.name, shard.device.getLocalHardwareId(platform.pjrt_api) });

        const needs_rewind = if (current_source_name) |name|
            !std.mem.eql(u8, name, shard.tensor.source_name) or (shard.offset < file_pos)
        else
            true;

        if (needs_rewind) {
            current_source_name = shard.tensor.source_name;
            try fs_source.initReader(shard.tensor.source_name, file_io_buffer, &source_reader);
            reader_iface = sourceInterface(&source_reader);
            file_pos = 0;
        }

        std.debug.assert(shard.offset >= file_pos);
        const bytes_to_discard = shard.offset - file_pos;
        if (bytes_to_discard > 0) {
            // try reader_iface.discardAll(bytes_to_discard);
        }
        file_pos = shard.offset;

        var limited_reader = std.io.Reader.Limited.init(reader_iface, .limited64(@intCast(shard.byteSize())), &.{});
        const reader = &limited_reader.interface;
        const writer = item.memory_writer;

        var bytes_processed: u64 = 0;
        while (bytes_processed < shard.byteSize()) {
            var free_buffer = try writer.getFreeBuffer();

            const read_limit: u64 = @min(shard.byteSize() - bytes_processed, free_buffer.len);
            const bytes_read = try reader.readSliceShort(free_buffer[0..@intCast(read_limit)]);

            if (bytes_read == 0) break;

            try writer.submitBuffer(
                free_buffer[0..bytes_read],
                item.buffer_index,
                bytes_processed,
                @intCast(shard.byteSize()),
            );

            bytes_processed += bytes_read;
            sum_total_bytes_copied += bytes_read;
        }

        file_pos += @intCast(shard.byteSize());
    }

    log.info("All shards submitted. Waiting for final transfers to complete.", .{});
    for (transfer_manager.memory_writers.items) |writer| {
        try writer.flushAllBlocking();
    }

    const elapsed = timer.read();
    const gb_copied = @as(f64, @floatFromInt(sum_total_bytes_copied)) / (1024.0 * 1024.0 * 1024.0);
    const rate = if (elapsed > 0) gb_copied / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;

    log.warn(
        "--- All tensors processed successfully in {d} ms ({d:.4} GB at {d:.2} GB/s) ---",
        .{ elapsed / std.time.ns_per_ms, gb_copied, rate },
    );
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
