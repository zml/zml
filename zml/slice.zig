pub const std = @import("std");
pub const stdx = @import("stdx");

const Shape = @import("shape.zig").Shape;

const log = std.log.scoped(.@"zml/slice");

/// Creates a slice of type `T` representing the specified `shape`.
pub fn Shaped(T: type, shape: Shape, slice: []u8) []T {
    stdx.debug.assert(slice.len == shape.byteSize(), "Slice length does not match shape byte size: {} != {}", .{ slice.len, shape.byteSize() });
    const ptr: [*]T = @alignCast(@constCast(@ptrCast(slice.ptr)));
    return ptr[0..shape.count()];
}

/// Creates a slice of type `T` representing the specified `shape` and fills it with the provided `value`.
pub fn fill(T: type, allocator: std.mem.Allocator, shape: Shape, value: T) ![]u8 {
    const slice = try allocator.alloc(u8, shape.byteSize());
    errdefer allocator.free(slice);

    @memset(Shaped(T, shape, slice), value);

    log.debug("Created filled slice of size {d} ptr {*} bytes for {}", .{ slice.len, slice.ptr, shape });
    return slice;
}

/// Creates a slice of integers in the specified shape, starting from `start`
/// and incrementing by `step`. The slice is allocated using the provided allocator.
/// TODO: Add support for other data types in the future.
pub const ArangeOpts = struct {
    start: i64 = 0,
    step: i64 = 1,
};
pub fn arange(allocator: std.mem.Allocator, shape: Shape, opts: ArangeOpts) ![]u8 {
    const slice = try allocator.alloc(u8, shape.byteSize());
    errdefer allocator.free(slice);

    switch (shape.dtype()) {
        inline else => |d| if (comptime d.class() != .integer) {
            stdx.debug.assert(shape.dtype().class() == .integer, "arange expects type to be integer, got {} instead.", .{shape.dtype()});
        } else {
            const Zt = d.toZigType();
            var j: i64 = opts.start;
            for (Shaped(Zt, shape, slice)) |*val| {
                val.* = @intCast(j);
                j +%= opts.step;
            }
        },
    }
    log.debug("Created arange slice of size {d} ptr {*} bytes for {}", .{ slice.len, slice.ptr, shape });
    return slice;
}
