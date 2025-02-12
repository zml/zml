const asynk = @import("async");
const core = @import("gguf/core.zig");
const std = @import("std");
const zml = @import("../zml.zig");

const HostBuffer = @import("../hostbuffer.zig").HostBuffer;

const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const log = std.log.scoped(.@"zml/io");

pub fn open(allocator: Allocator, path: []const u8) !zml.aio.BufferStore {
    var file = try core.GgufFile.open(path);
    errdefer file.close();

    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();
    const arena = res.arena.allocator();

    res.files = try arena.dupe(zml.aio.MemoryMappedFile, &.{file.file});

    // metadata must be read in order to read tensors
    try loadMetadata(arena, &res, &file);
    try loadBuffers(arena, &res, &file);
    if (res.buffers.count() != file.header.tensor_count) {
        log.warn("Expected to find {d} tensors in {s}, only found {d}", .{ file.header.tensor_count, path, res.buffers.count() });
    }
    return res;
}

fn loadMetadata(allocator: Allocator, store: *zml.aio.BufferStore, file: *core.GgufFile) !void {
    try store._metadata.ensureTotalCapacity(allocator, @intCast(file.header.metadata_kv_count));

    while (file.readMetadata(allocator)) |entry| {
        log.info("Loading MetaData: {s}", .{entry.name});
        const res = store._metadata.getOrPutAssumeCapacity(entry.name);
        if (res.found_existing) {
            // This file seems invalid. Since most metadatas aren't required, continue ahead.
            log.warn("Found duplicated metadata key: {s}", .{entry.name});
            continue;
        }
        res.value_ptr.* = switch (entry.val) {
            .array => |arr| switch (arr.child) {
                inline .uint8, .int8, .uint16, .int16, .uint32, .int32, .float32, .bool, .string, .uint64, .int64, .float64 => |tag| blk: {
                    const T = @FieldType(core.GgufValue, @tagName(tag));
                    break :blk try zml.aio.Metadata.copySlice(allocator, std.mem.bytesAsSlice(T, arr.data));
                },
                else => blk: {
                    log.warn("ignoring array metadata", .{});
                    break :blk .null;
                },
            },
            inline else => |v| zml.aio.Metadata.wrap(v),
        };
    } else |err| switch (err) {
        error.EndOfMetadata => {},
        else => return err,
    }
}

fn loadBuffers(allocator: Allocator, store: *zml.aio.BufferStore, file: *core.GgufFile) !void {
    try store.buffers.ensureTotalCapacity(allocator, @intCast(file.header.tensor_count));
    while (file.readTensorInfo(allocator)) |info| {
        const res = store.buffers.getOrPutAssumeCapacity(info.name);
        if (res.found_existing) {
            // This file seems invalid. Try to continue anyway.
            log.warn("Found duplicated tensor: {s}", .{info.name});
            continue;
        }

        // TODO: handle quantized types
        const dtype: zml.DataType = info.t.toDtype() orelse return error.UnsupportedGgufType;
        const buffer = HostBuffer.fromBytes(zml.Shape.init(info.shape(), dtype), file.file.mappedSlice(info.start, info.byte_len));
        res.value_ptr.* = buffer;
        // store the info index.
    } else |err| switch (err) {
        error.EndOfMetadata => {},
        else => return err,
    }
}
