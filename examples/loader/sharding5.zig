const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = asynk.logFn(std.log.defaultLog),
    .log_scope_levels = &.{
        .{ .scope = .@"zml/async", .level = .info },
    },
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

// Utility to create a binary file with a simple byte pattern for testing.
fn createBinFile(tmp_dir: std.testing.TmpDir, filename: []const u8, size: usize, alignment: ?usize) !usize {
    var file = try tmp_dir.dir.createFile(filename, .{});
    defer file.close();

    var writer_buffer: [BUF_64_KB]u8 = undefined;
    var file_writer = file.writer(&writer_buffer);

    var pattern_chunk: [BUF_64_KB]u8 = undefined;
    for (&pattern_chunk, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    var remaining_bytes_to_write = if (alignment) |a| std.mem.alignForward(usize, size, a) else size;
    while (remaining_bytes_to_write > 0) {
        const chunk_len = @min(remaining_bytes_to_write, pattern_chunk.len);
        try file_writer.interface.writeAll(pattern_chunk[0..chunk_len]);
        remaining_bytes_to_write -= chunk_len;
    }
    try file_writer.interface.flush();

    return try file_writer.file.getEndPos();
}

const Tensor = struct {
    source: []const u8,
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

pub const SafetensorsSource = []const u8;
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
            .source = try registry_allocator.dupe(u8, source),
            .name = try registry_allocator.dupe(u8, key),
            .shape = shape,
            .offset = data_start_offset + start,
        };

        log.debug("Parsed tensor {s}, shape: {any}, dtype: {any}, offset: {d}, size: {d} bytes", .{
            tensor.name,
            tensor.shape.dims(),
            tensor.shape.dtype(),
            tensor.offset,
            size_in_bytes,
        });

        try registry.tensors.put(registry_allocator, key, tensor);
    }
}

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

const AlignedFileReader = struct {
    reader: std.fs.File.Reader,
    alignment: std.mem.Alignment,
    pos: u64, // Physical offset for the next pread
    file_size: u64,

    buffer: [BUF_4_KB]u8 align(BUF_4_KB), // Hard coded
    buffer_valid_len: usize, // Total bytes read into buffer
    buffer_consumed: usize, // Bytes consumed/skipped from buffer head

    interface: std.io.Reader,

    pub fn init(reader: std.fs.File.Reader, alignment: std.mem.Alignment) !AlignedFileReader {
        const trace = tracer.frameStart("AlignedFileReader.init");
        defer tracer.frameEnd(trace, "AlignedFileReader.init");

        const file_size = try reader.file.getEndPos();
        const alignment_bytes = alignment.toByteUnits();
        const initial_pos = reader.pos;

        var current_pos = initial_pos;
        var consumed: usize = 0;

        if (initial_pos % alignment_bytes != 0) {
            const unaligned_head = initial_pos % alignment_bytes;
            current_pos = initial_pos - unaligned_head;
            consumed = @intCast(unaligned_head);
        }

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

    fn streamFromInternalBuffer(self: *AlignedFileReader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const available = self.buffer[self.buffer_consumed..self.buffer_valid_len];
        if (available.len == 0) return 0;

        const copy_len = limit.minInt(available.len);
        const written = try w.write(available[0..copy_len]);

        self.buffer_consumed += written;
        return written;
    }

    fn loadAlignedBlockToInternal(self: *AlignedFileReader) std.io.Reader.StreamError!void {
        self.buffer_valid_len = 0;
        const alignment_bytes = self.alignment.toByteUnits();

        if (self.pos >= self.file_size) return error.EndOfStream;

        const bytes_read = self.reader.file.pread(self.buffer[0..alignment_bytes], self.pos) catch |e| {
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

        if (limit == .nothing) return error.EndOfStream;

        if (self.buffer_consumed > 0 and self.buffer_valid_len == 0) {
            try self.loadAlignedBlockToInternal();
            return self.streamFromInternalBuffer(w, limit);
        }

        if (self.buffer_consumed < self.buffer_valid_len) {
            return self.streamFromInternalBuffer(w, limit);
        }

        self.buffer_consumed = 0;
        self.buffer_valid_len = 0;

        if (self.pos >= self.file_size) return error.EndOfStream;

        const alignment_bytes = self.alignment.toByteUnits();
        const logical_data_remaining = limit.toInt() orelse (self.file_size - self.pos);

        if (w.writableSliceGreedy(alignment_bytes)) |dest_buffer| {
            const max_read_size = @min(dest_buffer.len, self.file_size - self.pos);
            const max_aligned_read = std.mem.alignBackward(usize, @min(max_read_size, logical_data_remaining), alignment_bytes);

            if (max_aligned_read > 0) {
                const bytes_read = self.reader.file.pread(dest_buffer[0..max_aligned_read], self.pos) catch |e| {
                    log.err("File pread error (fast path): {}", .{e});
                    return error.ReadFailed;
                };
                if (bytes_read == 0) return error.EndOfStream;

                w.advance(bytes_read);
                self.pos += bytes_read;
                return bytes_read;
            }
        } else |err| {
            log.err("Writer error (fast path): {}", .{err});
            return err;
        }

        try self.loadAlignedBlockToInternal();
        return self.streamFromInternalBuffer(w, limit);
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

// test "AlignedFileReader: read from aligned offset" {
//     const allocator = std.testing.allocator;
//     var tmp_dir = std.testing.tmpDir(.{});
//     defer tmp_dir.cleanup();

//     const filename = "aligned.bin";
//     const file_size = BUF_1_MB;
//     const alignment = BUF_4_KB;
//     _ = try createBinFile(tmp_dir, filename, file_size, alignment);

//     const file = try tmp_dir.dir.openFile(filename, .{ .mode = .read_only });
//     defer file.close();

//     const o_direct_fd = try std.posix.open(
//         try tmp_dir.dir.realpathAlloc(allocator, filename),
//         .{ .ACCMODE = .RDONLY, .DIRECT = true },
//         0,
//     );

//     const o_direct_file: std.fs.File = .{ .handle = o_direct_fd };
//     defer o_direct_file.close();

//     const file_reader = o_direct_file.reader(&.{});
//     var reader: AlignedFileReader = try .init(file_reader, .fromByteUnits(alignment));

//     const read_size = 10 * KB;
//     const dest_buffer = try allocator.alloc(u8, read_size);
//     defer allocator.free(dest_buffer);

//     var writer = std.io.Writer.fixed(dest_buffer);
//     var total_read: usize = 0;
//     while (total_read < read_size) {
//         const n = try reader.interface.stream(&writer, .limited(read_size - total_read));
//         if (n == 0) break;
//         total_read += n;
//     }

//     try std.testing.expectEqual(@as(usize, read_size), total_read);

//     // Verify content
//     const expected_content = try allocator.alloc(u8, read_size);
//     defer allocator.free(expected_content);
//     _ = try file.readAll(expected_content);

//     try std.testing.expectEqualSlices(u8, expected_content, dest_buffer);
// }

// test "AlignedFileReader: read from unaligned offset" {
//     const allocator = std.testing.allocator;
//     var tmp_dir = std.testing.tmpDir(.{});
//     defer tmp_dir.cleanup();

//     const filename = "unaligned.bin";
//     const file_size = BUF_1_MB;
//     const alignment = BUF_4_KB;
//     const unaligned_offset = 123;
//     _ = try createBinFile(tmp_dir, filename, file_size, alignment);

//     const file = try tmp_dir.dir.openFile(filename, .{ .mode = .read_only });
//     defer file.close();

//     const o_direct_fd = try std.posix.open(
//         try tmp_dir.dir.realpathAlloc(allocator, filename),
//         .{ .ACCMODE = .RDONLY, .DIRECT = true },
//         0,
//     );
//     const o_direct_file = std.fs.File{ .handle = o_direct_fd };
//     defer o_direct_file.close();

//     var file_reader = o_direct_file.reader(&.{});
//     try file_reader.seekTo(unaligned_offset);

//     var reader = try AlignedFileReader.init(file_reader, .fromByteUnits(alignment));

//     // Internal state check
//     try std.testing.expectEqual(@as(u64, 0), reader.pos);
//     try std.testing.expectEqual(@as(usize, unaligned_offset), reader.buffer_consumed);

//     const read_size = 10 * KB;
//     const dest_buffer = try allocator.alloc(u8, read_size);
//     defer allocator.free(dest_buffer);

//     var writer = std.io.Writer.fixed(dest_buffer);
//     var total_read: usize = 0;
//     while (total_read < read_size) {
//         const n = try reader.interface.stream(&writer, .limited(read_size - total_read));
//         if (n == 0) break;
//         total_read += n;
//     }
//     try std.testing.expectEqual(@as(usize, read_size), total_read);

//     // Verify content
//     const expected_content = try allocator.alloc(u8, read_size);
//     defer allocator.free(expected_content);
//     _ = try file.seekTo(unaligned_offset);
//     _ = try file.readAll(expected_content);

//     try std.testing.expectEqualSlices(u8, expected_content, dest_buffer);
// }

// test "AlignedFileReader: fast path direct read" {
//     const allocator = std.testing.allocator;
//     var tmp_dir = std.testing.tmpDir(.{});
//     defer tmp_dir.cleanup();

//     const filename = "fastpath.bin";
//     const file_size = BUF_4_MB;
//     const alignment = BUF_4_KB;
//     _ = try createBinFile(tmp_dir, filename, file_size, alignment);

//     const file = try tmp_dir.dir.openFile(filename, .{ .mode = .read_only });
//     defer file.close();

//     const o_direct_fd = try std.posix.open(
//         try tmp_dir.dir.realpathAlloc(allocator, filename),
//         .{ .ACCMODE = .RDONLY, .DIRECT = true },
//         0,
//     );
//     const o_direct_file = std.fs.File{ .handle = o_direct_fd };
//     defer o_direct_file.close();

//     const file_reader = o_direct_file.reader(&.{});
//     var reader = try AlignedFileReader.init(file_reader, .fromByteUnits(alignment));

//     const read_size = BUF_1_MB; // Multiple of alignment
//     const dest_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), read_size);
//     defer allocator.free(dest_buffer);

//     var writer = std.io.Writer.fixed(dest_buffer);
//     const bytes_read = try reader.interface.stream(&writer, .limited(read_size));

//     try std.testing.expectEqual(@as(usize, read_size), bytes_read);
//     try std.testing.expectEqual(@as(u64, read_size), reader.pos); // Should have advanced via fast path

//     const expected_content = try allocator.alloc(u8, read_size);
//     defer allocator.free(expected_content);
//     _ = try file.readAll(expected_content);

//     try std.testing.expectEqualSlices(u8, expected_content, dest_buffer);
// }

// test "AlignedFileReader: mixed path (fast then slow)" {
//     const allocator = std.testing.allocator;
//     var tmp_dir = std.testing.tmpDir(.{});
//     defer tmp_dir.cleanup();

//     const filename = "mixedpath.bin";
//     const file_size = BUF_4_MB;
//     const alignment = BUF_4_KB;
//     const read_size = BUF_1_MB + 123; // Not a multiple of alignment
//     _ = try createBinFile(tmp_dir, filename, file_size, alignment);

//     const file = try tmp_dir.dir.openFile(filename, .{ .mode = .read_only });
//     defer file.close();

//     const o_direct_fd = try std.posix.open(
//         try tmp_dir.dir.realpathAlloc(allocator, filename),
//         .{ .ACCMODE = .RDONLY, .DIRECT = true },
//         0,
//     );
//     const o_direct_file = std.fs.File{ .handle = o_direct_fd };
//     defer o_direct_file.close();

//     const file_reader = o_direct_file.reader(&.{});
//     var reader = try AlignedFileReader.init(file_reader, .fromByteUnits(alignment));

//     const dest_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), read_size);
//     defer allocator.free(dest_buffer);

//     var writer = std.io.Writer.fixed(dest_buffer);

//     var total_read: usize = 0;
//     while (total_read < read_size) {
//         const n = reader.interface.stream(&writer, .limited(read_size - total_read)) catch |err| switch (err) {
//             error.EndOfStream => break,
//             else => |e| return e,
//         };
//         if (n == 0) break;
//         total_read += n;
//     }

//     try std.testing.expectEqual(@as(usize, read_size), total_read);

//     const expected_content = try allocator.alloc(u8, read_size);
//     defer allocator.free(expected_content);
//     _ = try file.readAll(expected_content);

//     try std.testing.expectEqualSlices(u8, expected_content, dest_buffer);
// }

// test "AlignedFileReader: file smaller than alignment" {
//     const allocator = std.testing.allocator;
//     var tmp_dir = std.testing.tmpDir(.{});
//     defer tmp_dir.cleanup();

//     const filename = "small.bin";
//     const file_size = 1 * KB;
//     const alignment = BUF_4_KB;
//     _ = try createBinFile(tmp_dir, filename, file_size, alignment);

//     const file = try tmp_dir.dir.openFile(filename, .{ .mode = .read_only });
//     defer file.close();

//     const o_direct_fd = try std.posix.open(
//         try tmp_dir.dir.realpathAlloc(allocator, filename),
//         .{ .ACCMODE = .RDONLY, .DIRECT = true },
//         0,
//     );
//     const o_direct_file = std.fs.File{ .handle = o_direct_fd };
//     defer o_direct_file.close();

//     const file_reader = o_direct_file.reader(&.{});
//     var reader = try AlignedFileReader.init(file_reader, .fromByteUnits(alignment));

//     const dest_buffer = try allocator.alloc(u8, file_size);
//     defer allocator.free(dest_buffer);

//     var writer = std.io.Writer.fixed(dest_buffer);
//     var total_read: usize = 0;
//     while (true) {
//         const n = reader.interface.stream(&writer, .unlimited) catch |err| switch (err) {
//             error.EndOfStream => break,
//             else => |e| return e,
//         };
//         if (n == 0) break;
//         total_read += n;
//     }

//     try std.testing.expectEqual(@as(usize, file_size), total_read);

//     const expected_content = try allocator.alloc(u8, file_size);
//     defer allocator.free(expected_content);
//     _ = try file.readAll(expected_content);

//     try std.testing.expectEqualSlices(u8, expected_content, dest_buffer);
// }

// test "AlignedFileReader: read until end of stream" {
//     const allocator = std.testing.allocator;
//     var tmp_dir = std.testing.tmpDir(.{});
//     defer tmp_dir.cleanup();

//     const filename = "eos.bin";
//     const file_size = 10 * KB + 5;
//     const alignment = BUF_4_KB;
//     _ = try createBinFile(tmp_dir, filename, file_size, alignment);

//     const file = try tmp_dir.dir.openFile(filename, .{ .mode = .read_only });
//     defer file.close();

//     const o_direct_fd = try std.posix.open(
//         try tmp_dir.dir.realpathAlloc(allocator, filename),
//         .{ .ACCMODE = .RDONLY, .DIRECT = true },
//         0,
//     );
//     const o_direct_file = std.fs.File{ .handle = o_direct_fd };
//     defer o_direct_file.close();

//     const file_reader = o_direct_file.reader(&.{});
//     var reader = try AlignedFileReader.init(file_reader, .fromByteUnits(alignment));

//     var writer = std.io.Writer.Allocating.init(allocator);

//     const bytes_read = try reader.interface.streamRemaining(&writer.writer);

//     try std.testing.expectEqual(@as(usize, file_size), bytes_read);

//     const expected_content = try file.readToEndAlloc(allocator, file_size + 1);
//     defer allocator.free(expected_content);

//     try std.testing.expectEqualSlices(u8, expected_content, try writer.toOwnedSlice());

//     // Check EndOfStream is sticky
//     var dummy_writer = std.io.Writer.fixed(&[_]u8{});
//     try std.testing.expectError(error.EndOfStream, reader.interface.stream(&dummy_writer, .unlimited));
// }

const DeviceWriter = struct {
    platform: Platform,
    shard: Shard,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,

    events: [2]?*pjrtx.Event,
    next_slot_idx: u1,

    bytes_written: u64,
    can_process_last_event: bool,

    interface: std.io.Writer,

    pub fn init(platform: Platform, shard: Shard, memory_kind: pjrt.Memory.Kind) !DeviceWriter {
        const trace = tracer.frameStart("DeviceWriter.init");
        defer tracer.frameEnd(trace, "DeviceWriter.init");

        const trace_memory = tracer.frameStart("DeviceWriter.init.memory");
        const memories = try shard.device.addressableMemories(platform.pjrt_api);
        var memory = memories[0];

        if (platform.target == .cuda) {
            for (memories) |mem| {
                if (mem.kind(platform.pjrt_api) == memory_kind) {
                    memory = mem;
                    break;
                }
            }
        }
        tracer.frameEnd(trace_memory, "DeviceWriter.init.memory");

        const shape_spec = pjrt.ShapeSpec.init(shard.shape.dims(), bufferTypeFromDtype(shard.shape.dtype()));
        const trace_transfer_manager = tracer.frameStart("DeviceWriter.init.transfer_manager");
        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = memory,
        });
        tracer.frameEnd(trace_transfer_manager, "DeviceWriter.init.transfer_manager");

        return .{
            .platform = platform,
            .shard = shard,
            .transfer_manager = transfer_manager,
            .events = .{ null, null },
            .next_slot_idx = 0,
            .bytes_written = 0,
            .can_process_last_event = true,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .end = 0 },
        };
    }

    pub fn deinit(self: *DeviceWriter) void {
        self.transfer_manager.deinit(self.platform.pjrt_api);
        self.transfer_manager = undefined;
    }

    pub fn buffer(self: *DeviceWriter) !*pjrtx.Buffer {
        return try self.transfer_manager.retrieveBuffer(self.platform.pjrt_api, 0);
    }

    fn deviceDescription(self: *DeviceWriter) []const u8 {
        return self.shard.device.getDescription(self.platform.pjrt_api).toString(self.platform.pjrt_api);
    }

    fn awaitEvent(self: *DeviceWriter, idx: u1) !void {
        if (self.events[idx]) |event| {
            const trace = tracer.frameStart("DeviceWriter.awaitEvent");
            defer tracer.frameEnd(trace, "DeviceWriter.awaitEvent");

            try event.awaitBlocking(self.platform.pjrt_api);
            self.events[idx] = null;
        }
    }

    fn transfer(self: *DeviceWriter, data: []const u8, is_last: bool) !*pjrtx.Event {
        const trace = tracer.frameStart("DeviceWriter.transfer");
        defer tracer.frameEnd(trace, "DeviceWriter.transfer");

        const offset = self.bytes_written;

        defer {
            if (!is_last) self.bytes_written += data.len;
        }

        log.debug("DeviceWriter({s}).transfer: {d}B, offset: {d}, is_last: {} - progress: {d}/{d}B", .{
            self.deviceDescription(),
            data.len,
            offset,
            is_last,
            offset + data.len,
            self.shard.byteSize(),
        });

        std.debug.assert(offset + data.len <= self.shard.byteSize());

        return self.transfer_manager.transferData(self.platform.pjrt_api, 0, data, @intCast(offset), is_last) catch |err| {
            log.err("PJRT transferData failed: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("DeviceWriter.drain");
        defer tracer.frameEnd(trace, "DeviceWriter.drain");

        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        std.debug.assert(w.end == 0);
        std.debug.assert(splat == 1);
        std.debug.assert(data.len == 1);

        const chunk = data[0];

        log.debug("DeviceWriter({s}).drain: chunk={d}B, progress={d}/{d}, slot={d}, pending=[{s},{s}], last_event_ready={}", .{
            self.deviceDescription(),
            chunk.len,
            self.bytes_written,
            self.shard.byteSize(),
            self.next_slot_idx,
            if (self.events[0] != null) "busy" else "free",
            if (self.events[1] != null) "busy" else "free",
            self.can_process_last_event,
        });

        if (chunk.len == 0) return 0;

        const slot_to_use = self.next_slot_idx;

        self.awaitEvent(slot_to_use) catch |err| {
            log.err("Error awaiting event in drain: {}", .{err});
            return error.WriteFailed;
        };

        self.events[slot_to_use] = try self.transfer(chunk, false);
        self.next_slot_idx = 1 - self.next_slot_idx;

        return chunk.len;
    }

    fn flush(w: *std.io.Writer) std.io.Writer.Error!void {
        const trace = tracer.frameStart("DeviceWriter.flush");
        defer tracer.frameEnd(trace, "DeviceWriter.flush");

        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        log.debug("DeviceWriter({s}).flush: awaiting pending transfers - progress={d}/{d}B, pending_slots=[{s},{s}], will_finalize={}", .{
            self.deviceDescription(),
            self.bytes_written,
            self.shard.byteSize(),
            if (self.events[0] != null) "busy" else "free",
            if (self.events[1] != null) "busy" else "free",
            self.can_process_last_event,
        });

        self.awaitEvent(0) catch return error.WriteFailed;
        self.awaitEvent(1) catch return error.WriteFailed;

        if (self.can_process_last_event) {
            std.debug.assert(self.bytes_written == self.shard.byteSize());

            const last_event = try self.transfer(&.{}, true);
            last_event.awaitBlocking(self.platform.pjrt_api) catch return error.WriteFailed;

            self.can_process_last_event = false;
        }
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

test "DeviceWriter: writeAll and read back" {
    const allocator = std.testing.allocator;

    var context: Context = try .init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.test.DeviceWriter.writeAll");
    const trace = tracer.frameStart("DeviceWriter.test.writeAllandReadBack");
    defer tracer.frameEnd(trace, "DeviceWriter.test.writeAllandReadBack");

    const platform = context.autoPlatform(.{});

    const device = platform.getDevices()[0];

    const tensor_size = 10 * 4 * BUF_256_MB;
    const shape: Shape = .init(.{tensor_size / @sizeOf(u8)}, .u8);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{ .source = "test", .name = "tensor1", .shape = shape, .offset = 0 },
        .device = device,
    };

    const original_data = try allocator.alloc(u8, tensor_size);
    defer allocator.free(original_data);

    for (original_data, 0..) |*byte, i| byte.* = @intCast(i % 256);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, original_data);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, original_data) catch {};

    var writer: DeviceWriter = try .init(platform, shard, .device);
    defer writer.deinit();

    try writer.interface.writeAll(original_data);
    try writer.interface.flush();

    try std.testing.expectEqual(shard.byteSize(), writer.bytes_written);
    try std.testing.expect(!writer.can_process_last_event);

    const pjrt_buffer = try writer.buffer();
    defer pjrt_buffer.deinit(platform.pjrt_api);

    const reader_dma_buffer_size = BUF_32_MB;
    const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), reader_dma_buffer_size);
    defer allocator.free(reader_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

    var reader: DeviceReader = try .init(platform, pjrt_buffer, reader_buffer);

    var read_back_data = std.io.Writer.Allocating.init(allocator);
    defer read_back_data.deinit();

    _ = try reader.interface.streamRemaining(&read_back_data.writer);

    const read_back_data_slice = try read_back_data.toOwnedSlice();
    defer allocator.free(read_back_data_slice);

    try std.testing.expectEqualSlices(u8, original_data, read_back_data_slice);
}

// test "DeviceWriter: write in chunks and read back" {
//     const allocator = std.testing.allocator;

//     var context: Context = try .init();
//     defer context.deinit();
//     tracer = Tracer.init("ai.zml.test.DeviceWriter.chunks");

//     const platform = context.autoPlatform(.{});
//     if (platform.getDevices().len == 0) {
//         std.log.warn("Skipping test, no devices found", .{});
//         return;
//     }
//     const device = platform.getDevices()[0];

//     const tensor_size: usize = 32 * MB;
//     const chunk_size: usize = 1 * MB;
//     const shape: Shape = .init(.{tensor_size / @sizeOf(u8)}, .u8);
//     const shard: Shard = .{
//         .shape = shape,
//         .tensor = .{ .source = "test", .name = "tensor2", .shape = shape, .offset = 0 },
//         .device = device,
//     };

//     var original_data = try allocator.alloc(u8, tensor_size);
//     defer allocator.free(original_data);
//     for (original_data, 0..) |*byte, i| byte.* = @intCast(i % 256);

//     try platform.pjrt_client.dmaMap(platform.pjrt_api, original_data);
//     defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, original_data) catch {};

//     var writer = try DeviceWriter.init(platform, shard, .device);
//     defer writer.deinit();

//     var written: usize = 0;
//     while (written < tensor_size) {
//         const to_write = @min(chunk_size, tensor_size - written);
//         const chunk = original_data[written .. written + to_write];
//         try writer.interface.writeAll(chunk);
//         written += to_write;
//     }
//     try writer.interface.flush();

//     const pjrt_buffer = try writer.buffer();
//     defer pjrt_buffer.deinit(platform.pjrt_api);

//     const reader_dma_buffer_size = 32 * MB;
//     const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), reader_dma_buffer_size);
//     defer allocator.free(reader_buffer);
//     try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
//     defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

//     var reader: DeviceReader = try .init(platform, pjrt_buffer, reader_buffer);

//     var read_back_writer: std.io.Writer.Allocating = try .initCapacity(allocator, original_data.len);
//     defer read_back_writer.deinit();

//     _ = try reader.interface.streamRemaining(&read_back_writer.writer);

//     const read_back_data = try read_back_writer.toOwnedSlice();
//     defer allocator.free(read_back_data);

//     try std.testing.expectEqualSlices(u8, original_data, read_back_data);
// }

test "DeviceWriter: flush finalizes transfer" {
    const platform = zml.testing.env();
    const device = platform.getDevices()[0];

    // Use zero size to test only the finalization logic
    const tensor_size: usize = 0;
    const shape = Shape.init(.{tensor_size}, .u8);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{ .source = "test", .name = "tensor_flush", .shape = shape, .offset = 0 },
        .device = device,
    };

    var writer = try DeviceWriter.init(platform, shard, .device);
    defer writer.deinit();

    try std.testing.expect(writer.can_process_last_event);
    try writer.interface.flush();
    try std.testing.expect(!writer.can_process_last_event);

    try writer.interface.flush();
    try std.testing.expect(!writer.can_process_last_event);
}

// test "DeviceWriter + Write" {
//     const allocator = std.testing.allocator;

//     var context: Context = try .init();
//     defer context.deinit();

//     tracer = Tracer.init("ai.zml.test.DeviceWriter+Write");

//     const platform = context.autoPlatform(.{});
//     const devices = platform.getDevices();
//     const device = devices[0];

//     const TENSOR_SIZE = 1024 * MB;
//     const DMA_BUFFER_SIZE = 32 * MB;

//     const shape: Shape = .init(.{TENSOR_SIZE / @sizeOf(f32)}, .f32);
//     const tensor: Tensor = .{
//         .source = "test",
//         .name = "tensor",
//         .shape = shape,
//         .offset = 0,
//     };
//     const shard: Shard = .{
//         .shape = shape,
//         .tensor = tensor,
//         .device = device,
//     };

//     const dma_buffer = try allocator.alloc(u8, DMA_BUFFER_SIZE);
//     defer allocator.free(dma_buffer);

//     try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_buffer);
//     defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_buffer) catch {};

//     var writer: DeviceWriter = try .init(platform, shard, .device);
//     defer writer.deinit();

//     const data = try allocator.alloc(u8, TENSOR_SIZE);
//     defer allocator.free(data);

//     try platform.pjrt_client.dmaMap(platform.pjrt_api, data);
//     defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, data) catch {};

//     for (data, 0..) |*byte, i| {
//         byte.* = @intCast(i % 256);
//     }

//     const WRITE_CHUNK_SIZE = DMA_BUFFER_SIZE / 4;
//     var total_bytes_written: u64 = 0;

//     while (total_bytes_written < TENSOR_SIZE) {
//         const remaining = TENSOR_SIZE - total_bytes_written;
//         const chunk_size = @min(remaining, WRITE_CHUNK_SIZE);
//         const chunk = data[total_bytes_written..][0..chunk_size];
//         log.debug("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });

//         var index: usize = 0;
//         while (index < chunk.len) {
//             const trace_write = tracer.frameStart("writer.interface.write");
//             defer tracer.frameEnd(trace_write, "writer.interface.write");

//             log.debug("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });
//             index += try writer.interface.write(chunk[index..]);
//         }

//         // try writer.interface.writeAll(chunk);
//         total_bytes_written += chunk.len;
//     }
//     try writer.interface.flush();
// }

const TensorWriter = struct {
    device_writers: []DeviceWriter,

    chunk_size: usize,
    current_buffer_idx: u1,
    shard_size: u64,
    is_sharded: bool,
    total_bytes_processed: u64,

    buffer: []u8,

    interface: std.io.Writer,

    pub fn init(device_writers: []DeviceWriter, buffer: []u8) TensorWriter {
        const chunk_size = buffer.len / 2;

        return .{
            .device_writers = device_writers,
            .buffer = buffer,
            .chunk_size = chunk_size,
            .shard_size = if (device_writers.len > 0) device_writers[0].shard.byteSize() else 0,
            .is_sharded = if (device_writers.len > 0)
                device_writers[0].shard.byteSize() < device_writers[0].shard.tensor.byteSize()
            else
                false,
            .current_buffer_idx = 0,
            .total_bytes_processed = 0,
            .interface = .{ .vtable = &vtable, .buffer = buffer[0..chunk_size], .end = 0 },
        };
    }

    fn process(self: *TensorWriter, chunk_to_process: []u8) !void {
        const trace = tracer.frameStart("TensorWriter.process");
        defer tracer.frameEnd(trace, "TensorWriter.process");

        if (self.device_writers.len == 0 or chunk_to_process.len == 0) return;

        var data_offset: usize = 0;
        while (data_offset < chunk_to_process.len) {
            const current_tensor_offset = self.total_bytes_processed + data_offset;
            const remaining_in_data = chunk_to_process.len - data_offset;

            const current_shard_idx: usize = @intCast(current_tensor_offset / self.shard_size);
            if (current_shard_idx >= self.device_writers.len) break;

            const offset_in_shard = current_tensor_offset % self.shard_size;
            const chunk_limit_by_boundary = self.shard_size - offset_in_shard;
            const chunk_to_write_len = @min(remaining_in_data, chunk_limit_by_boundary);
            if (chunk_to_write_len == 0) break;

            const chunk_to_move = chunk_to_process[data_offset .. data_offset + chunk_to_write_len];

            log.debug("TensorWriter.process: tensor_offset={d}B/{d}B, shard[{d}]@{d}B+{d}B, remaining_data={d}B", .{
                current_tensor_offset,
                self.shard_size * self.device_writers.len,
                current_shard_idx,
                offset_in_shard,
                chunk_to_write_len,
                remaining_in_data - chunk_to_write_len,
            });

            if (self.is_sharded) {
                try self.device_writers[current_shard_idx].interface.writeAll(chunk_to_move);
            } else {
                for (self.device_writers) |*dw| {
                    try dw.interface.writeAll(chunk_to_move);
                }
            }

            data_offset += chunk_to_write_len;
        }

        self.total_bytes_processed += chunk_to_process.len;
    }

    pub fn swap(self: *TensorWriter) void {
        self.current_buffer_idx = 1 - self.current_buffer_idx;
        const new_offset = self.current_buffer_idx * self.chunk_size;
        self.interface.buffer = self.buffer[new_offset .. new_offset + self.chunk_size];
        self.interface.end = 0;
    }

    fn processAndSwap(self: *TensorWriter) !void {
        const trace = tracer.frameStart("TensorWriter.processAndSwap");
        defer tracer.frameEnd(trace, "TensorWriter.processAndSwap");

        const chunk_to_process = self.interface.buffered();

        log.debug("TensorWriter.processAndSwap: switching from buffer #{d} to #{d}, processing {d}B chunk, total processed: {d}B", .{
            self.current_buffer_idx,
            1 - self.current_buffer_idx,
            chunk_to_process.len,
            self.total_bytes_processed,
        });

        if (chunk_to_process.len > 0) {
            try self.process(chunk_to_process);
        }

        self.swap();
    }

    fn rebase(w: *std.io.Writer, preserve: usize, minimum_len: usize) std.io.Writer.Error!void {
        const trace = tracer.frameStart("TensorWriter.rebase");
        defer tracer.frameEnd(trace, "TensorWriter.rebase");

        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));
        std.debug.assert(preserve == 0);

        log.debug("TensorWriter.rebase: preserve={d}, minimum_len={d}, buffered={d}B, processed={d}B, will_flip_buffer={}", .{
            preserve,
            minimum_len,
            w.end,
            self.total_bytes_processed,
            w.end + minimum_len >= w.buffer.len,
        });

        self.processAndSwap() catch |err| {
            log.err("Error processing chunk during rebase: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("TensorWriter.drain");
        defer tracer.frameEnd(trace, "TensorWriter.drain");

        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));

        const total_incoming_bytes = blk: {
            var sum: usize = 0;
            for (data) |chunk| sum += chunk.len;
            break :blk sum;
        };

        log.debug("TensorWriter.drain: incoming={d}B (splat={d}), buffered={d}B, processed={d}B, will_flip_buffer={}", .{
            total_incoming_bytes,
            splat,
            w.end,
            self.total_bytes_processed,
            w.end + total_incoming_bytes >= w.buffer.len,
        });

        try self.processAndSwap();

        return w.writeSplat(data, splat);
    }

    fn flush(w: *std.io.Writer) !void {
        const trace = tracer.frameStart("TensorWriter.flush");
        defer tracer.frameEnd(trace, "TensorWriter.flush");

        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));

        log.debug("TensorWriter.flush: finalizing tensor - buffered={d}B, processed={d}B, remaining_batches={d}, will_double_flush={}", .{
            w.end,
            self.total_bytes_processed,
            if (w.end > 0) @as(u32, 2) else @as(u32, 1),
            w.end > 0,
        });

        try self.processAndSwap();

        if (w.end > 0) {
            try self.processAndSwap();
        }

        for (self.device_writers) |*dw| {
            try dw.interface.flush();
        }
    }

    const vtable = std.io.Writer.VTable{
        .drain = drain,
        .flush = flush,
        .rebase = rebase,
    };
};

const DeviceReader = struct {
    pub const NUM_SLOTS = 2;

    platform: Platform,
    pjrt_buffer: *const pjrtx.Buffer,
    total_size: u64,
    chunk_size: usize,

    slots: [NUM_SLOTS]?*pjrtx.Event,
    dma_buffer: []u8,
    bytes_requested: u64,
    bytes_activated: u64,
    next_request_slot: u1,
    next_consume_slot: u1,

    is_primed: bool,
    interface: std.io.Reader,

    pub fn init(platform: Platform, pjrt_buffer: *const pjrtx.Buffer, dma_buffer: []u8) !DeviceReader {
        const chunk_size = dma_buffer.len / NUM_SLOTS;
        std.debug.assert(chunk_size > 0);

        return .{
            .platform = platform,
            .pjrt_buffer = pjrt_buffer,
            .total_size = try pjrt_buffer.getOnDeviceSizeInBytes(platform.pjrt_api),
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

    fn requestNextChunk(self: *DeviceReader) std.io.Reader.StreamError!void {
        const trace = tracer.frameStart("DeviceReader.requestNextChunk");
        defer tracer.frameEnd(trace, "DeviceReader.requestNextChunk");

        const slot_to_fill = self.next_request_slot;
        std.debug.assert(self.slots[slot_to_fill] == null);

        if (self.bytes_requested >= self.total_size) return;

        const offset_in_dma = slot_to_fill * self.chunk_size;
        const chunk_dma_buf = self.dma_buffer[offset_in_dma .. offset_in_dma + self.chunk_size];
        const remaining_on_device = self.total_size - self.bytes_requested;
        const transfer_size = @min(remaining_on_device, chunk_dma_buf.len);
        const dest_slice = chunk_dma_buf[0..transfer_size];
        const offset: i64 = @intCast(self.bytes_requested);

        log.debug("DeviceReader.requestNextChunk: slot={d}, device_offset={d}B, transfer_size={d}B, remaining={d}B", .{
            slot_to_fill,
            offset,
            transfer_size,
            self.total_size - (self.bytes_requested + transfer_size),
        });

        const event = self.pjrt_buffer.copyRawToHost(self.platform.pjrt_api, dest_slice, offset) catch |err| {
            log.err("PJRT copyRawToHost failed: {}", .{err});
            return error.ReadFailed;
        };

        if (event) |ev| self.slots[slot_to_fill] = ev;
        self.bytes_requested += transfer_size;
        self.next_request_slot = 1 - self.next_request_slot;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const trace = tracer.frameStart("DeviceReader.stream(vtable)");
        defer tracer.frameEnd(trace, "DeviceReader.stream(vtable)");

        _ = w;
        _ = limit;
        const self = @as(*DeviceReader, @alignCast(@fieldParentPtr("interface", r)));

        std.debug.assert(r.seek == r.end);

        if (!self.is_primed) {
            self.is_primed = true;
            if (self.total_size == 0) return error.EndOfStream;
            log.debug("DeviceReader: Priming pipeline for {d} bytes...", .{self.total_size});
            for (0..NUM_SLOTS) |_| try self.requestNextChunk();
        }

        const slot_to_consume = self.next_consume_slot;

        if (self.slots[slot_to_consume]) |event| {
            const trace_await = tracer.frameStart("DeviceReader.awaitEvent");
            defer tracer.frameEnd(trace_await, "DeviceReader.awaitEvent");

            event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error awaiting event in stream: {}", .{err});
                return error.ReadFailed;
            };
            self.slots[slot_to_consume] = null;
        } else {
            if (self.bytes_activated >= self.total_size) {
                return error.EndOfStream;
            } else {
                log.err("DeviceReader stalled: waiting for slot {d} which has no event.", .{slot_to_consume});
                return error.ReadFailed;
            }
        }

        try self.requestNextChunk();

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

    const vtable = std.io.Reader.VTable{
        .stream = stream,
    };
};

test "DeviceReader: streamRemaining" {
    const allocator = std.testing.allocator;

    tracer = Tracer.init("ai.zml.test.DeviceReader");
    const trace_test = tracer.frameStart("DeviceReader.test.streamRemaining");
    defer tracer.frameEnd(trace_test, "DeviceReader.test.streamRemaining");

    const platform = zml.testing.env();

    const memory_kind: pjrtx.Memory.Kind = if (platform.target == .cuda) .device else .pinned_host;
    const memories = try platform.getDevices()[0].addressableMemories(platform.pjrt_api);
    const memory = for (memories) |m| {
        const kind = m.kind(platform.pjrt_api);
        if (kind == memory_kind) break m;
    } else return error.NotFound;

    const shape: Shape = .init(.{4 * BUF_256_MB / @sizeOf(f32)}, .f32);

    const buffer = try platform.pjrt_client.createUnitializedBuffer(platform.pjrt_api, .{
        .dims = shape.dims(),
        .element_type = bufferTypeFromDtype(shape.dtype()),
        .layout = .{
            .tiled = .{
                .minor_to_major = minor_to_major[Shape.MAX_RANK - shape.rank() ..],
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
        .memory = memory,
    });
    defer buffer.deinit(platform.pjrt_api);

    const reader_buffer_size = BUF_64_MB;
    const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), reader_buffer_size);
    defer allocator.free(reader_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

    var device_reader: DeviceReader = try .init(platform, buffer, reader_buffer);

    const read_back_buffer = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(read_back_buffer);

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    random.bytes(read_back_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, read_back_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, read_back_buffer) catch {};

    var read_back_writer: std.io.Writer = .fixed(read_back_buffer);

    const trace_stream = tracer.frameStart("DeviceReader.streamRemaining");
    const bytes_read = try device_reader.interface.streamRemaining(&read_back_writer);
    tracer.frameEnd(trace_stream, "DeviceReader.streamRemaining");

    try std.testing.expectEqual(shape.byteSize(), bytes_read);

    const expected_zeros = try allocator.alloc(u8, read_back_buffer.len);
    defer allocator.free(expected_zeros);
    @memset(expected_zeros, 0);

    try std.testing.expectEqualSlices(u8, expected_zeros, read_back_buffer);
}

test "DeviceReader: discard writer" {
    const allocator = std.testing.allocator;

    tracer = Tracer.init("ai.zml.test.DeviceReader");
    const trace_test = tracer.frameStart("DeviceReader.test.discardWriter");
    defer tracer.frameEnd(trace_test, "DeviceReader.test.discardWriter");

    const platform = zml.testing.env();

    const memory_kind: pjrtx.Memory.Kind = if (platform.target == .cuda) .device else .pinned_host;
    const memories = try platform.getDevices()[0].addressableMemories(platform.pjrt_api);
    const memory = for (memories) |m| {
        const kind = m.kind(platform.pjrt_api);
        if (kind == memory_kind) break m;
    } else return error.NotFound;

    const shape: Shape = .init(.{4 * BUF_256_MB / @sizeOf(f32)}, .f32);

    const buffer = try platform.pjrt_client.createUnitializedBuffer(platform.pjrt_api, .{
        .dims = shape.dims(),
        .element_type = bufferTypeFromDtype(shape.dtype()),
        .layout = .{
            .tiled = .{
                .minor_to_major = minor_to_major[Shape.MAX_RANK - shape.rank() ..],
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
        .memory = memory,
    });
    defer buffer.deinit(platform.pjrt_api);

    const reader_buffer_size = BUF_64_MB;
    const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), reader_buffer_size);
    defer allocator.free(reader_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

    var device_reader: DeviceReader = try .init(platform, buffer, reader_buffer);

    var total_bytes_consumed: u64 = 0;
    var writer_buffer: [1]u8 = undefined;
    var writer: std.io.Writer = .fixed(&writer_buffer);

    while (true) {
        const chunk = device_reader.interface.buffered();
        if (chunk.len > 0) {
            total_bytes_consumed += chunk.len;
            device_reader.interface.tossBuffered();
        }

        const n = device_reader.interface.stream(&writer, .unlimited) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };

        std.debug.assert(n == 0);
    }

    try std.testing.expectEqual(shape.byteSize(), total_bytes_consumed);
}

const TensorReader = struct {
    device_readers: []DeviceReader,
    current_reader_idx: usize,
    interface: std.io.Reader,

    pub fn init(device_readers: []DeviceReader) !TensorReader {
        return .{
            .device_readers = device_readers,
            .current_reader_idx = 0,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*TensorReader, @alignCast(@fieldParentPtr("interface", r)));

        while (self.current_reader_idx < self.device_readers.len) {
            const current_dr_interface = &self.device_readers[self.current_reader_idx].interface;
            const bytes_read = current_dr_interface.stream(w, limit) catch |err| switch (err) {
                error.EndOfStream => {
                    self.current_reader_idx += 1;
                    continue;
                },
                else => |e| return e,
            };
            return bytes_read;
        }
        return error.EndOfStream;
    }

    const vtable = std.io.Reader.VTable{
        .stream = stream,
    };
};

test "Full Pipeline: TensorWriter -> GPU -> TensorReader" {
    const allocator = std.testing.allocator;

    tracer = Tracer.init("ai.zml.test.FullPipeline");

    const platform = zml.testing.env();
    const devices = platform.getDevices();

    if (devices.len < 2) {
        std.log.warn("Skipping test, requires at least 2 devices, found {d}", .{devices.len});
        return error.SkipZigTest;
    }

    const devices_to_use = devices[0..2];

    const tensor_size = 128 * MB;
    const tensor_writer_buffer_size = 16 * MB;
    const tensor_writer_chunk_size = 8 * MB;
    const device_reader_buffer_size = 128 * MB;
    const alignment = BUF_4_KB;

    const original_data = try allocator.alloc(u8, tensor_size);
    defer allocator.free(original_data);
    for (original_data, 0..) |*byte, i| byte.* = @intCast(i % 256);

    const shape = Shape.init(.{tensor_size}, .u8).withSharding(.{0});
    const tensor: Tensor = .{ .source = "test", .name = "full_pipeline_tensor", .shape = shape, .offset = 0 };

    const shards = try computeShards(allocator, tensor, devices_to_use);
    defer allocator.free(shards);

    const tensor_writer_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), tensor_writer_buffer_size);
    defer allocator.free(tensor_writer_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, tensor_writer_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, tensor_writer_buffer) catch {};

    var device_writers: std.ArrayList(DeviceWriter) = try .initCapacity(allocator, shards.len);
    errdefer for (device_writers.items) |*dw| dw.deinit();
    defer device_writers.deinit(allocator);

    for (shards) |s| device_writers.appendAssumeCapacity(try .init(platform, s, .device));

    var tensor_writer: TensorWriter = .init(device_writers.items, tensor_writer_buffer);

    var total_bytes_written: u64 = 0;
    while (total_bytes_written < tensor_size) {
        const remaining = tensor_size - total_bytes_written;
        const chunk_size = @min(remaining, tensor_writer_chunk_size);
        const chunk = original_data[total_bytes_written..][0..chunk_size];

        try tensor_writer.interface.writeAll(chunk);

        total_bytes_written += chunk.len;
    }
    try tensor_writer.interface.flush();

    var pjrt_buffers: [devices_to_use.len]*pjrtx.Buffer = undefined;
    for (device_writers.items, 0..) |*dw, i| {
        pjrt_buffers[i] = try dw.buffer();
        dw.deinit();
    }
    defer for (pjrt_buffers) |b| b.deinit(platform.pjrt_api);

    var device_readers_buffers: std.ArrayList([]u8) = .{};
    defer {
        for (device_readers_buffers.items) |b| {
            platform.pjrt_client.dmaUnmap(platform.pjrt_api, b) catch {};
            allocator.free(b);
        }
        device_readers_buffers.deinit(allocator);
    }

    var device_readers: std.ArrayList(DeviceReader) = .{};
    defer device_readers.deinit(allocator);

    for (pjrt_buffers) |buffer| {
        const device_reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), device_reader_buffer_size);
        try device_readers_buffers.append(allocator, device_reader_buffer);
        try platform.pjrt_client.dmaMap(platform.pjrt_api, device_reader_buffer);
        try device_readers.append(allocator, try .init(platform, buffer, device_reader_buffer));
    }

    var tensor_reader: TensorReader = try .init(device_readers.items);

    var read_back_writer: std.io.Writer.Allocating = try .initCapacity(allocator, tensor_size);
    defer read_back_writer.deinit();

    const bytes_read = try tensor_reader.interface.streamRemaining(&read_back_writer.writer);

    try std.testing.expectEqual(tensor_size, bytes_read);

    const read_back_data = try read_back_writer.toOwnedSlice();
    defer allocator.free(read_back_data);

    try std.testing.expectEqualSlices(u8, original_data, read_back_data);

    log.warn("Full pipeline test completed successfully, transferred {d} bytes", .{bytes_read});
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

// Compute the shards for a tensor based on its shape and the available devices.
fn computeShards(allocator: std.mem.Allocator, tensor: Tensor, devices: []const *const pjrt.Device) ![]Shard {
    const trace = tracer.frameStart("computeShards");
    defer tracer.frameEnd(trace, "computeShards");

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

    log.warn("--- Initializing context and platform... ---", .{});
    var context: Context = try .init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.examples.loader");

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);
    const devices = platform.getDevices();

    const elapsed_init = timer.lap();
    log.warn("--- Initialized context and platform with {d} devices in {d}ms ---", .{ devices.len, elapsed_init / std.time.ns_per_ms });

    const trace_post_init = tracer.frameStart("Main post context/platform init");
    defer tracer.frameEnd(trace_post_init, "Main post context/platform init");

    log.warn("--- Discovering model... ---", .{});
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
    log.warn("--- Discovered {d} tensors in model ({d:.2} GB) in {d}ms ---", .{ registry.tensors.count(), registry.totalBytes() / (1024 * 1024 * 1024), elapsed_discovery / std.time.ns_per_ms });

    log.warn("--- Preparing streaming file handles... ---", .{});
    var o_direct_files: std.StringHashMapUnmanaged(std.fs.File) = .{};
    defer {
        var it = o_direct_files.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.close();
            allocator.free(entry.key_ptr.*);
        }
        o_direct_files.deinit(allocator);
    }

    var files_it = files.iterator();
    while (files_it.next()) |entry| {
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
        try o_direct_files.put(allocator, try allocator.dupe(u8, basename), file);
    }

    const elapsed_streaming_prep = timer.lap();
    log.warn("--- Prepared {d} streaming file handles in {d}ms ---", .{ o_direct_files.count(), elapsed_streaming_prep / std.time.ns_per_ms });

    log.warn("--- Applying sharding information... ---", .{});
    try addExampleShardingMetadata(&registry);
    try annotateShapesWithSharding(&registry);

    const elapsed_sharding = timer.lap();
    log.warn("--- Applied sharding information in {d}ms ---", .{elapsed_sharding / std.time.ns_per_ms});

    log.warn("--- Sorting tensors by source and offset... ---", .{});
    const tensors = blk: {
        const ts = registry.tensors.values();

        std.mem.sort(Tensor, ts, {}, struct {
            fn lessThan(_: void, a: Tensor, b: Tensor) bool {
                const name_cmp = std.mem.order(u8, a.source, b.source);
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
    log.warn("--- Sorted tensors in {d}ms ---", .{elapsed_sorting / std.time.ns_per_ms});

    log.warn("--- Allocating DMA buffers", .{});
    const trace_allocation = tracer.frameStart("Buffers allocation and DMA mapping");

    const dma_writer_staging_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), BUF_32_MB);
    defer allocator.free(dma_writer_staging_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_writer_staging_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_writer_staging_buffer) catch unreachable;

    const file_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), BUF_64_MB);
    defer allocator.free(file_buffer);

    const tensor_reader_buffer: [0]u8 = undefined;

    const elasped_preparation = timer.lap();
    tracer.frameEnd(trace_allocation, "Buffers allocation and DMA mapping");
    log.warn("--- Prepared for tensor processing in {d}ms ---", .{elasped_preparation / std.time.ns_per_ms});

    log.warn("--- Warming up devices (GPU bfc allocation) ---", .{});
    var warmup_timer = try std.time.Timer.start();

    for (devices) |device| {
        const trace_warmup = tracer.frameStart("Transfer Manager Warmup");
        defer tracer.frameEnd(trace_warmup, "Transfer Manager Warmup");

        const warmup_shape = Shape.init(.{BUF_1_MB / @sizeOf(f32)}, .f32);

        const warmup_data = try allocator.alloc(u8, BUF_1_MB);
        defer allocator.free(warmup_data);

        const warmup_shard = Shard{
            .shape = warmup_shape,
            .tensor = .{
                .source = "warmup",
                .name = "warmup_tensor",
                .shape = warmup_shape,
                .offset = 0,
            },
            .device = device,
        };

        const warmp_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(warmp_buffer);

        var warmup_writer = try DeviceWriter.init(platform, warmup_shard, .device);
        defer warmup_writer.deinit();

        try warmup_writer.interface.writeAll(warmup_data);
        try warmup_writer.interface.flush();
    }

    const warmup_elapsed = warmup_timer.lap();
    log.warn("--- Warmed up transfer managers for {d} devices in {d}ms ---", .{ devices.len, warmup_elapsed / std.time.ns_per_ms });

    log.warn("--- Starting tensor processing stream... ---", .{});
    timer.reset();

    const trace_processing = tracer.frameStart("Tensor Processing Stream");

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    var sum_total_bytes_copied: u64 = 0;
    var sum_total_bytes_read: u64 = 0;

    for (tensors) |tensor| {
        var tensor_timer = try std.time.Timer.start();
        const trace = tracer.frameStart("Tensor Processing");
        defer tracer.frameEnd(trace, "Tensor Processing");

        log.info("Processing {s}/{s} dims={any} size={d} offset={d}", .{
            tensor.source,
            tensor.name,
            tensor.shape.dims(),
            tensor.byteSize(),
            tensor.offset,
        });

        const trace_arena_reset = tracer.frameStart("Arena reset");
        _ = arena.reset(.retain_capacity);
        tracer.frameEnd(trace_arena_reset, "Arena reset");

        const arena_allocator = arena.allocator();

        const file = o_direct_files.get(tensor.source).?;
        const shards = try computeShards(arena_allocator, tensor, devices);

        const trace_file_read = tracer.frameStart("File reader setup");
        var file_reader = file.reader(&.{});
        try file_reader.seekTo(tensor.offset);
        tracer.frameEnd(trace_file_read, "File reader setup");

        var aligned_file_reader: AlignedFileReader = try .init(file_reader, .fromByteUnits(BUF_4_KB));
        var tensor_reader: std.io.Reader.Limited = .init(&aligned_file_reader.interface, .limited64(tensor.shape.byteSize()), &tensor_reader_buffer);

        var device_writers: std.ArrayList(DeviceWriter) = try .initCapacity(arena_allocator, devices.len);
        errdefer {
            for (device_writers.items) |*device_writer| {
                device_writer.deinit();
            }
            device_writers.deinit(arena_allocator);
        }
        for (0..devices.len) |i| {
            device_writers.appendAssumeCapacity(try .init(platform, shards[i], .device));
        }

        var tensor_writer: TensorWriter = .init(device_writers.items, dma_writer_staging_buffer);
        const bytes_copied = try tensor_reader.interface.streamRemaining(&tensor_writer.interface);
        try tensor_writer.interface.flush();

        const elapsed_tensor = tensor_timer.lap();
        const mb_copied = @as(f64, @floatFromInt(bytes_copied)) / (BUF_1_MB);
        const rate = if (elapsed_tensor > 0) mb_copied / (@as(f64, @floatFromInt(elapsed_tensor)) / 1_000_000_000.0) else 0;
        log.info("Loaded tensor in {d:.2}ms ({d:.2} MB at {d:.2} MB/s)", .{ elapsed_tensor / std.time.ns_per_ms, mb_copied, rate });

        std.debug.assert(bytes_copied == tensor.shape.byteSize());

        for (shards) |shard| {
            sum_total_bytes_copied += shard.byteSize();
        }
        sum_total_bytes_read += tensor.shape.byteSize();
    }
    tracer.frameEnd(trace_processing, "Tensor Processing Stream");

    const elapsed = timer.read();
    const gb_copied = @as(f64, @floatFromInt(sum_total_bytes_copied)) / (1.0 * 1024 * 1024 * 1024);
    const gb_read = @as(f64, @floatFromInt(sum_total_bytes_read)) / (1.0 * 1024 * 1024 * 1024);
    const read_rate = if (elapsed > 0) gb_read / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    const copy_rate = if (elapsed > 0) gb_copied / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    log.warn("--- All tensors loaded in {d}ms ({d:.2} GB read at {d:.2} GB/s, {d:.2} GB copied at {d:.2} GB/s) ---", .{ elapsed / std.time.ns_per_ms, gb_read, read_rate, gb_copied, copy_rate });
}

// all code below is unmodified (or slightly) / imported strucs / funcs from zml

pub fn bufferTypeFromDtype(dt: DataType) pjrtx.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrtx.BufferType, @tagName(tag)),
    };
}

const minor_to_major: [Shape.MAX_RANK]i64 = blk: {
    var min_to_maj: [Shape.MAX_RANK]i64 = undefined;
    for (0..Shape.MAX_RANK) |i| {
        min_to_maj[i] = @intCast(Shape.MAX_RANK - i - 1);
    }
    break :blk min_to_maj;
};
