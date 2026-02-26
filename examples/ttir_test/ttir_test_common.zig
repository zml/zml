const std = @import("std");
const zml = @import("zml");

pub fn writeEmbeddedSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    bytes: []const u8,
    filename: []const u8,
) ![]const u8 {
    const file = try std.Io.Dir.createFile(.cwd(), io, filename, .{});
    defer file.close(io);

    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();

    var real_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const real_len = try file.realPath(io, &real_buf);
    return try allocator.dupe(u8, real_buf[0..real_len]);
}

pub fn loadBufferFromRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    registry: *zml.safetensors.TensorRegistry,
    key: []const u8,
) !zml.Buffer {
    const tensor_desc = registry.tensors.get(key) orelse return error.NotFound;
    const shape = tensor_desc.shape;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try registry.reader(io, key, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, host_bytes);
}

pub fn uninitBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    return zml.Buffer.fromSlice(io, platform, slice);
}

pub fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
) !zml.Buffer {
    var slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    @memset(slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, slice);
}

pub fn printExpectedMatch(io: std.Io, actual: zml.Buffer, expected: zml.Buffer) void {
    var matches = true;
    zml.testing.expectClose(io, actual, expected, .{}) catch {
        matches = false;
    };
    std.debug.print("\n\n", .{});
    if (matches) {
        std.debug.print("Output matches expected tensor\n", .{});
    } else {
        std.debug.print("Output does not match expected tensor\n", .{});
    }
}
