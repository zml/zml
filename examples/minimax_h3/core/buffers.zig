const std = @import("std");

const zml = @import("zml");

pub fn fromItems(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, items: anytype) !zml.Buffer {
    return fromItemsSharded(io, platform, shape, .replicated, items);
}

pub fn fromItemsSharded(
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
    items: anytype,
) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, shape, sharding, std.mem.sliceAsBytes(items));
}

pub fn fromF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    values: []const f32,
) !zml.Buffer {
    switch (shape.dtype()) {
        .f32 => return fromItems(io, platform, shape, values),
        .bf16 => {
            const converted = try allocator.alloc(zml.floats.BFloat16, values.len);
            defer allocator.free(converted);
            for (converted, values) |*dst, src| dst.* = .fromF32(src);
            return fromItems(io, platform, shape, converted);
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

pub fn toF32(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) ![]f32 {
    const slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    switch (buffer.shape().dtype()) {
        .f32 => return allocator.dupe(f32, slice.items(f32)),
        .bf16 => {
            const source = slice.items(zml.floats.BFloat16);
            const converted = try allocator.alloc(f32, source.len);
            for (converted, source) |*dst, value| dst.* = value.toF32();
            return converted;
        },
        else => return error.UnsupportedEmbedDtype,
    }
}
