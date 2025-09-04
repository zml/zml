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

pub const Tensor = struct {
    shape: Shape,
    file_index: u32,
    offset: u64,
    size_in_bytes: u64,
};

pub const TensorReader = struct {
    underlying_reader: asynk.File.Reader,
    bytes_remaining: u64,
    interface: std.io.Reader,

    pub fn init(underlying_reader: asynk.File.Reader, size_in_bytes: u64) TensorReader {
        return TensorReader{
            .underlying_reader = underlying_reader,
            .bytes_remaining = size_in_bytes,
            .interface = .{
                .vtable = &vtable,
                .buffer = &[_]u8{},
                .seek = 0,
                .end = 0,
            },
        };
    }

    pub fn stream(r: *std.io.Reader, writer: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*TensorReader, @alignCast(@fieldParentPtr("interface", r)));

        if (self.bytes_remaining == 0) {
            return error.EndOfStream;
        }

        const effective_limit = std.io.Limit.limited64(self.bytes_remaining).min(limit);
        const bytes_read = try self.underlying_reader.interface.stream(writer, effective_limit);

        self.bytes_remaining -= bytes_read;
        return bytes_read;
    }

    const vtable = std.io.Reader.VTable{
        .stream = stream,
    };
};

pub const ChecksummingWriter = struct {
    hasher: std.crypto.hash.sha2.Sha256,
    next_writer: *std.io.Writer,
    interface: std.io.Writer,

    pub fn init(next_writer: *std.io.Writer) ChecksummingWriter {
        return ChecksummingWriter{
            .hasher = std.crypto.hash.sha2.Sha256.init(.{}),
            .next_writer = next_writer,
            .interface = .{
                .vtable = &vtable,
                .buffer = &[_]u8{},
                .end = 0,
            },
        };
    }

    pub fn final(self: *ChecksummingWriter, out_digest: *[32]u8) void {
        self.hasher.final(out_digest);
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*ChecksummingWriter, @alignCast(@fieldParentPtr("interface", w)));

        std.debug.assert(w.buffered().len == 0);

        for (data) |d| {
            self.hasher.update(d);
        }
        if (splat > 1 and data.len > 0) {
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                self.hasher.update(last);
            }
        }

        return self.next_writer.writeSplat(data, splat);
    }

    const vtable = std.io.Writer.VTable{
        .drain = drain,
    };
};

pub const Registry = struct {
    pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);

    arena: std.heap.ArenaAllocator,
    files: []asynk.File = &.{},
    tensors: Tensors = .{},
    metadata: Metadatas = .{},

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);

        for (self.files) |file| {
            file.close() catch |err| {
                log.warn("failed to close file: {}", .{err});
            };
        }
        self.arena.deinit();
    }

    pub fn tensorReader(
        self: *Registry,
        tensor: Tensor,
        read_buffer: []u8,
    ) !TensorReader {
        const file = &self.files[tensor.file_index];

        try file.seekTo(tensor.offset);

        const file_reader = file.reader(read_buffer);

        return TensorReader.init(file_reader, tensor.size_in_bytes);
    }
};

pub fn open(allocator: std.mem.Allocator, path: []const u8) !Registry {
    var registry: Registry = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer registry.deinit();

    const arena = registry.arena.allocator();
    registry.tensors = Registry.Tensors{};
    registry.metadata = Registry.Metadatas{};

    var files = std.array_list.Managed(asynk.File).init(arena);
    errdefer files.deinit();

    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        try loadFromIndex(arena, &registry, &files, path);
    } else {
        try loadFile(arena, &registry, &files, path);
    }

    registry.files = try files.toOwnedSlice();

    return registry;
}

fn loadFromIndex(allocator: std.mem.Allocator, registry: *Registry, files: *std.array_list.Managed(asynk.File), path: []const u8) !void {
    const file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };
    defer file.close() catch {};
    var buffer: [4096]u8 = undefined;
    var r = file.reader(&buffer);

    var json_reader = std.json.Reader.init(allocator, &r.interface);
    const index = try std.json.parseFromTokenSourceLeaky(std.json.Value, allocator, &json_reader, .{ .allocate = .alloc_if_needed });

    var loaded_files = std.StringHashMap(void).init(allocator);
    defer loaded_files.deinit();

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;
        if (loaded_files.contains(filename)) {
            continue;
        }

        log.info("Loading file part: {s}", .{filename});
        try loaded_files.put(filename, {});

        const full_filename = try std.fs.path.join(allocator, &.{ std.fs.path.dirname(path).?, filename });

        try loadFile(allocator, registry, files, full_filename);
    }

    if (index.object.get("__metadata__")) |metadata| {
        var prefix_buf: [1024]u8 = undefined;
        try parseMetadata(allocator, registry, StringBuilder.initBuffer(&prefix_buf), metadata);
    }
}

fn loadFile(allocator: std.mem.Allocator, registry: *Registry, files: *std.array_list.Managed(asynk.File), path: []const u8) !void {
    var file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };

    var r_buffer: [16 * 1024]u8 = undefined;
    var r = file.reader(&r_buffer);

    const json_header_length: usize = @intCast(try r.interface.takeInt(u64, .little));

    const json_data = try allocator.alloc(u8, json_header_length);
    defer allocator.free(json_data);
    try r.interface.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;

    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{});

    const file_index: u32 = @intCast(files.items.len);
    try files.append(file);

    var it = metadata.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) {
            var prefix_buf: [1024]u8 = undefined;
            try parseMetadata(allocator, registry, StringBuilder.initBuffer(&prefix_buf), entry.value_ptr.*);
            continue;
        }
        const val = entry.value_ptr.*;
        const shape_field = val.object.get("shape").?.array;
        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            continue;
        }
        const offset_field = val.object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const end: u64 = @intCast(offset_field.array.items[1].integer);
        const dtype = try stringToDtype(val.object.get("dtype").?.string);

        var dims: Dims = .{};
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const out_shape = Shape.init(dims.constSlice(), dtype);
        const size_in_bytes = end - start;
        std.debug.assert(size_in_bytes == out_shape.byteSize());

        const tensor_info = Tensor{
            .shape = out_shape,
            .file_index = file_index,
            .offset = data_start_offset + start,
            .size_in_bytes = size_in_bytes,
        };

        try registry.tensors.put(allocator, try allocator.dupe(u8, key), tensor_info);
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

    const file_path = if (args.next()) |path| path else {
        log.err("Usage: <program> /path/to/model.safetensors or /path/to/model.safetensors.index.json", .{});
        return;
    };

    log.info("Loading from: {s}", .{file_path});

    var registry = try open(allocator, file_path);
    defer registry.deinit();

    log.info("Registry loaded with {d} tensors.", .{registry.tensors.count()});

    var read_buffer: [64 * 1024]u8 = undefined;

    var it = registry.tensors.iterator();
    while (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        const tensor = entry.value_ptr.*;

        log.info("Processing tensor '{s}' with shape {any}", .{ tensor_name, tensor.shape });

        const dest_buffer = try allocator.alloc(u8, tensor.size_in_bytes);
        defer allocator.free(dest_buffer);

        var dest_writer = std.io.Writer.fixed(dest_buffer);

        var checksum_writer = ChecksummingWriter.init(&dest_writer);

        var tensor_reader = try registry.tensorReader(tensor, &read_buffer);

        log.info("Streaming {d} bytes...", .{tensor.size_in_bytes});

        // prefer         _ = try tensor_reader.interface.streamRemaining(&checksum_writer.interface);

        var pump_buffer: [64 * 1024]u8 = undefined;
        while (true) {
            const bytes_read = tensor_reader.interface.readSliceShort(&pump_buffer) catch |err| {
                if (err == error.EndOfStream) break; // Graceful end of tensor data
                return err;
            };
            if (bytes_read == 0) break; // Finished reading
            try checksum_writer.interface.writeAll(pump_buffer[0..bytes_read]);
        }

        var digest: [32]u8 = undefined;
        checksum_writer.final(&digest);
        log.info("...done streaming tensor '{s}'. SHA256: {s}", .{ tensor_name, &digest });
    }

    log.info("All tensors processed.", .{});
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

pub fn parseMetadata(allocator: std.mem.Allocator, registry: *Registry, prefix: StringBuilder, val: std.json.Value) !void {
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
                    try parseMetadata(allocator, registry, new_prefix, item);
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
                try parseMetadata(allocator, registry, new_prefix, entry.value_ptr.*);
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
