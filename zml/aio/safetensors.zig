const asynk = @import("async");
const std = @import("std");
const zml = @import("../zml.zig");
const helpers = @import("../helpers.zig");
const utils = @import("utils.zig");
const json = @import("json.zig");
const HostBuffer = @import("../hostbuffer.zig").HostBuffer;
const MemoryMappedFile = @import("../aio.zig").MemoryMappedFile;

const StringBuilder = std.ArrayListUnmanaged(u8);
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.zml_io);

fn stringToDtype(v: []const u8) !zml.DataType {
    const Case = enum { F64, F32, F16, BF16, F8_E4M3, I64, I32, I16, I8, U64, U32, U16, U8, BOOL };
    if (std.meta.stringToEnum(Case, v)) |case| {
        return switch (case) {
            .F64 => .f64,
            .F32 => .f32,
            .F16 => .f16,
            .BF16 => .bf16,
            .F8_E4M3 => .f8e4m3fn,
            .I64 => .i64,
            .I32 => .i32,
            .I16 => .i16,
            .I8 => .i8,
            .U64 => .u64,
            .U32 => .u32,
            .U16 => .u16,
            .U8 => .u8,
            .BOOL => .bool,
        };
    }
    std.log.err("Unsupported type-string: {s}\n", .{v});
    return error.UnsupportedDataType;
}

pub fn open(allocator: std.mem.Allocator, path: []const u8) !zml.aio.BufferStore {
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();
    const arena = res.arena.allocator();

    var files = std.ArrayList(MemoryMappedFile).init(arena);
    errdefer files.deinit();

    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        try loadFromIndex(arena, &res, &files, path);
    } else {
        try loadFile(arena, &res, &files, path);
    }
    res.files = try files.toOwnedSlice();
    return res;
}

fn loadFromIndex(allocator: Allocator, store: *zml.aio.BufferStore, files: *std.ArrayList(MemoryMappedFile), path: []const u8) !void {
    const file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };
    errdefer file.close() catch unreachable;
    var r = file.reader();

    const json_data = try allocator.alloc(u8, (try file.stat()).size);
    _ = try r.readAtLeast(json_data, json_data.len);
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{ .allocate = .alloc_if_needed });
    var loaded_files = std.StringHashMap(void).init(allocator);

    const weight_map = metadata.object.get("weight_map").?.object;
    var it = weight_map.iterator();
    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;
        if (loaded_files.contains(filename)) {
            continue;
        }

        log.debug("Loading shard: {s}", .{filename});
        try loaded_files.put(filename, {});

        const full_filename = try std.fs.path.join(allocator, &.{ std.fs.path.dirname(path).?, filename });
        try loadFile(allocator, store, files, full_filename);
    }
}

fn loadFile(allocator: Allocator, store: *zml.aio.BufferStore, files: *std.ArrayList(MemoryMappedFile), path: []const u8) !void {
    const file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };
    errdefer file.close() catch unreachable;
    var r = file.reader();

    const json_header_length: usize = @intCast(try r.readInt(u64, std.builtin.Endian.little));
    const json_data = try allocator.alloc(u8, json_header_length);
    _ = try r.readAtLeast(json_data, json_header_length);
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{ .allocate = .alloc_if_needed });

    var buffer_file = try MemoryMappedFile.init(file);
    errdefer buffer_file.deinit();
    buffer_file.data_offset = 8 + json_header_length;

    try files.append(buffer_file);
    errdefer _ = files.popOrNull();

    var it = metadata.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) {
            var prefix_buf: [1024]u8 = undefined;
            try json.parseMetadata(allocator, store, StringBuilder.initBuffer(&prefix_buf), entry.value_ptr.*);
            continue;
        }
        const val = entry.value_ptr.*;
        const shape_field = val.object.get("shape").?.array;
        if (shape_field.items.len > zml.Shape.MAX_RANK) {
            // Not an error until someone tries to read the tensor itself.
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            continue;
        }
        const offset_field = val.object.get("data_offsets").?;
        const start: usize = @intCast(offset_field.array.items[0].integer);
        const end: usize = @intCast(offset_field.array.items[1].integer);
        const dtype = try stringToDtype(val.object.get("dtype").?.string);
        var dims: std.BoundedArray(i64, zml.Shape.MAX_RANK) = .{};
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const out_shape = zml.Shape.init(dims.constSlice(), dtype);
        // We aren't storing 'end', so check we can infer it from the tensor shape.
        // This is fine cause safetensor only allow storing contiguous tensors.
        // https://github.com/huggingface/safetensors/blob/main/README.md#format
        // > The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents the creation of polyglot files.
        std.debug.assert(end - start == out_shape.byteSize());

        const buf = HostBuffer.fromBytes(out_shape, buffer_file.mappedSlice(start, out_shape.byteSize()));
        try store.buffers.put(allocator, try allocator.dupe(u8, key), buf);
    }
}
