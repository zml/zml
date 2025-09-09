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
    name: []const u8,
    shape: Shape,
    source_index: u32,
    offset: u64,
};

pub const TensorReader = struct {
    tensor: Tensor,
    reader: *std.io.Reader,
};

pub const Registry = struct {
    pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);
    pub const SourcePaths = std.ArrayListUnmanaged([]const u8);

    arena: std.heap.ArenaAllocator,

    tensors: Tensors,
    metadata: Metadatas,
    source_paths: SourcePaths,

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();

        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.source_paths.deinit(allocator);

        self.arena.deinit();
    }
};

// Safetensors Loader

pub fn registerSafetensors(allocator: std.mem.Allocator, provider: anytype, path: []const u8) !Registry {
    var registry: Registry = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .tensors = .{},
        .metadata = .{},
        .source_paths = .{},
    };
    errdefer registry.deinit();

    var processing_arena = std.heap.ArenaAllocator.init(allocator);
    defer processing_arena.deinit();
    const processing_allocator = processing_arena.allocator();

    var path_map = std.StringHashMap(u32).init(processing_allocator);
    defer path_map.deinit();

    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        try registerFromIndex(processing_allocator, &registry, provider, path, &path_map);
    } else {
        try registerPath(processing_allocator, &registry, provider, path, &path_map);
    }

    return registry;
}

fn registerFromIndex(allocator: std.mem.Allocator, registry: *Registry, provider: anytype, path: []const u8, path_map: *std.StringHashMap(u32)) !void {
    const file = try provider.open(path);
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_reader = file.reader(&buffer);

    var json_reader: std.json.Reader = .init(allocator, &file_reader.interface);
    const index = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, &json_reader, .{ .allocate = .alloc_if_needed });

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;

        const full_filename = try std.fs.path.join(allocator, &.{ provider.base_dir, filename });

        if (path_map.contains(full_filename)) continue;

        log.info("Discovering file part: {s}", .{filename});

        try registerPath(allocator, registry, provider, full_filename, path_map);
    }

    if (index.object.get("__metadata__")) |metadata| {
        var prefix_buf: [1024]u8 = undefined;
        try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), metadata);
    }
}

fn registerPath(allocator: std.mem.Allocator, registry: *Registry, provider: anytype, path: []const u8, path_map: *std.StringHashMap(u32)) !void {
    const registry_allocator = registry.arena.allocator();

    const file = try provider.open(path);
    defer file.close();

    var reader_buffer: [16 * 1024]u8 = undefined;
    var file_reader = file.reader(&reader_buffer);

    const json_header_length: usize = @intCast(try file_reader.interface.takeInt(u64, .little));

    const json_data = try allocator.alloc(u8, json_header_length);
    defer allocator.free(json_data);
    try file_reader.interface.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;

    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{});

    const file_index: u32 = if (path_map.get(path)) |idx| idx else blk: {
        const new_index: u32 = @intCast(registry.source_paths.items.len);
        const registered_path = try registry_allocator.dupe(u8, path);

        try registry.source_paths.append(registry_allocator, registered_path);
        try path_map.put(registered_path, new_index);
        break :blk new_index;
    };

    var it = metadata.object.iterator();

    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            var prefix_buf: [1024]u8 = undefined;
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
            .name = try registry_allocator.dupe(u8, key),
            .shape = shape,
            .source_index = file_index,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(registry_allocator, key, tensor);
    }
}

// Source Providers

pub const SourceCallback = fn (ctx: anytype, stream: TensorReader) anyerror!void;

pub const FileSystemSourceProvider = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,

    pub fn open(_: FileSystemSourceProvider, path: []const u8) !std.fs.File {
        return std.fs.openFileAbsolute(path, .{ .mode = .read_only });
    }

    pub fn resolve(self: FileSystemSourceProvider, relative_path: []const u8) ![]const u8 {
        return std.fs.path.join(self.allocator, &.{ self.base_dir, relative_path });
    }
};

pub const FilesystemSource = struct {
    registry: *Registry,
    sources: []std.fs.File,

    pub fn iterator(self: FilesystemSource, context: anytype, callback: SourceCallback) !void {
        var it = self.registry.tensors.iterator();

        while (it.next()) |entry| {
            const tensor = entry.value_ptr.*;

            var source_file = self.sources[tensor.source_index];
            try source_file.seekTo(tensor.offset);

            var file_read_buffer: [16 * 1024]u8 = undefined;
            var file_reader = source_file.reader(&file_read_buffer);

            var limited_reader_storage = LimitingReader.init(&file_reader.interface, tensor.shape.byteSize());

            const stream: TensorReader = .{ .tensor = tensor, .reader = &limited_reader_storage.interface };

            try callback(context, stream);
        }
    }
};

// Reusable Stream Processors

pub const LimitingReader = struct {
    underlying_reader: *std.io.Reader,
    bytes_remaining: u64,
    interface: std.io.Reader,

    pub fn init(underlying_reader: *std.io.Reader, limit: u64) LimitingReader {
        return .{
            .underlying_reader = underlying_reader,
            .bytes_remaining = limit,
            .interface = .{ .vtable = &vtable, .buffer = &[_]u8{}, .seek = 0, .end = 0 },
        };
    }

    fn stream(r: *std.io.Reader, writer: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*@This(), @alignCast(@fieldParentPtr("interface", r)));

        if (self.bytes_remaining == 0) return error.EndOfStream;

        const effective_limit = std.io.Limit.limited64(self.bytes_remaining).min(limit);
        const bytes_read = try self.underlying_reader.stream(writer, effective_limit);
        self.bytes_remaining -= bytes_read;

        return bytes_read;
    }

    const vtable: std.io.Reader.VTable = .{ .stream = stream };
};

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
        const self = @as(*@This(), @alignCast(@fieldParentPtr("interface", w)));

        log.debug("Quantizing data to {d} bits...", .{self.config.target_bits});

        return self.next_writer.writeSplat(data, splat);
    }

    const vtable: std.io.Writer.VTable = .{ .drain = drain };
};

pub const Checksumer = struct {
    pub const Writer = ChecksummingWriter;

    digest: *[32]u8,

    pub fn writer(self: Checksumer, next: *std.io.Writer) Writer {
        return Writer.init(next, self);
    }
};

const ChecksummingWriter = struct {
    config: Checksumer,
    hasher: std.crypto.hash.sha2.Sha256,
    next_writer: *std.io.Writer,
    interface: std.io.Writer,

    pub fn init(next_writer: *std.io.Writer, config: Checksumer) @This() {
        return .{
            .config = config,
            .hasher = std.crypto.hash.sha2.Sha256.init(.{}),
            .next_writer = next_writer,
            .interface = .{ .vtable = &vtable, .buffer = &[_]u8{}, .end = 0 },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*@This(), @alignCast(@fieldParentPtr("interface", w)));

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
        log.warn("Checksum {any}", .{self.config.digest});
        try self.next_writer.flush();
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

pub const Executor = struct {
    pub fn process(context: anytype, stream: TensorReader) !void {
        const tensor = stream.tensor;
        log.info("--- Processing tensor: {s} ---", .{tensor.name});

        var arena_buffer: [1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&arena_buffer);
        const stack_allocator = fba.allocator();

        const final_writer = context.*.sink.writer(tensor);
        var writer_chain: *std.io.Writer = final_writer;

        inline for (context.*.pipeline_stages) |stage_config| {
            const WriterType = @TypeOf(stage_config).Writer;
            const new_writer = try stack_allocator.create(WriterType);
            new_writer.* = stage_config.writer(writer_chain);
            writer_chain = &new_writer.interface;
            log.info("  [Executor] Added '{s}' to pipeline", .{@typeName(WriterType)});
        }

        var pump_buffer: [64 * 1024]u8 = undefined;
        var total_bytes_copied: u64 = 0;

        while (true) {
            const bytes_read = stream.reader.readSliceShort(&pump_buffer) catch |err| {
                if (err == error.EndOfStream) break;
                return err;
            };

            if (bytes_read == 0) break;
            try writer_chain.writeAll(pump_buffer[0..bytes_read]);
            total_bytes_copied += bytes_read;
        }

        std.debug.assert(total_bytes_copied == tensor.shape.byteSize());

        try writer_chain.flush();

        log.info("--- Finished tensor: {s} ({d} bytes) ---", .{ tensor.name, total_bytes_copied });
    }
};

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

    log.info("--- Loading model metadata... ---", .{});
    const provider: FileSystemSourceProvider = .{
        .allocator = allocator,
        .base_dir = std.fs.path.dirname(file_path) orelse ".",
    };

    var registry = try registerSafetensors(allocator, provider, file_path);
    defer registry.deinit();

    log.info("Registry loaded with {d} tensors from {d} source files.", .{ registry.tensors.count(), registry.source_paths.items.len });

    log.info("--- Opening source files... ---", .{});
    var files: std.ArrayList(std.fs.File) = .{};
    defer {
        for (files.items) |file| file.close();
        files.deinit(allocator);
    }

    for (registry.source_paths.items) |source_path| {
        const file = try provider.open(source_path);
        try files.append(allocator, file);
    }

    var sink: DiscardingSink = .init();

    var digest: [32]u8 = undefined;

    const pipeline_stages = .{
        Quantizer{ .target_bits = 8 },
        Checksumer{ .digest = &digest },
    };

    const source = FilesystemSource{
        .registry = &registry,
        .sources = files.items,
    };

    const executor_context = .{
        .pipeline_stages = pipeline_stages,
        .sink = &sink,
    };

    log.info("--- Starting tensor processing stream... ---", .{});
    try source.iterator(&executor_context, Executor.process);
    log.info("--- Pipeline finished successfully. ---", .{});

    log.info("SHA256 of the last processed tensor was: {s}", .{std.fmt.bytesToHex(digest, .lower)});
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
