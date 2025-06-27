pub const std = @import("std");
pub const stdx = @import("stdx");

const floats = @import("floats.zig");
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

pub const Slice1DOpts = struct {
    start: i64 = 0,
    end: ?i64 = null,
};

/// Slices the input Tensor over the given axis using the given parameters.
pub fn slice1d(shape: Shape, slice: []u8, axis_: anytype, opts: Slice1DOpts) struct { Shape, []u8 } {
    const ax = shape.axis(axis_);
    const d = shape.dim(ax);
    const start: i64 = if (opts.start < 0) opts.start + d else opts.start;
    var end = opts.end orelse d;
    if (end < 0) end += d;

    stdx.debug.assert(start >= 0 and start < d, "slice1d({}, {*}, {}) expects the slice start to be between 0 and {} got: {}", .{ shape, slice.ptr, ax, d, opts });
    stdx.debug.assert(end >= 1 and end <= d, "slice1d({}, {*}, {}) expects the slice end to be between 1 and {} got: {}", .{ shape, slice.ptr, ax, d, opts });
    stdx.debug.assert(start < end, "slice1d({}, {*}, {}) expects the slice start ({}) to be smaller than the end ({}), got: {}", .{ shape, slice.ptr, ax, start, end, opts });

    const offset: usize = @intCast(start * shape.computeByteStrides().get(ax));
    const new_shape = shape.set(ax, end - start);
    const sliced = slice[offset .. offset + new_shape.byteSize()];

    return .{ new_shape, sliced };
}

// This module provides a way to pretty print a slice in a human-readable format.
pub fn pretty(shape: Shape, slice: []u8) PrettyPrinter {
    return .{ .shape = shape, .slice = slice };
}

pub const PrettyPrinter = struct {
    shape: Shape,
    slice: []u8,

    pub fn format(self: PrettyPrinter, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        const fmt_: stdx.fmt.Fmt = switch (self.shape.dtype().class()) {
            .integer => .parse(i32, fmt),
            .float => .parse(f32, fmt),
            else => .parse(void, fmt),
        };
        try prettyPrint(self.shape, self.slice, writer, .{ .fmt = fmt_, .options = options });
    }
};

pub fn prettyPrint(shape: Shape, slice: []u8, writer: anytype, options: stdx.fmt.FullFormatOptions) !void {
    return prettyPrintIndented(shape, slice, writer, 4, 0, options);
}

fn prettyPrintIndented(shape: Shape, slice: []u8, writer: anytype, num_rows: u8, indent_level: u8, options: stdx.fmt.FullFormatOptions) !void {
    if (shape.rank() == 0) {
        // Special case input tensor is a scalar
        return switch (shape.dtype()) {
            inline else => |dt| {
                const val: dt.toZigType() = Shaped(dt.toZigType(), shape, slice)[0];
                return switch (comptime dt.class()) {
                    // Since we have custom floats, we need to explicitly convert to float32 ourselves.
                    .float => stdx.fmt.formatFloatValue(floats.floatCast(f32, val), options, writer),
                    .integer => stdx.fmt.formatIntValue(val, options, writer),
                    .bool, .complex => stdx.fmt.formatAnyValue(val, options, writer),
                };
            },
        };
    }
    if (shape.rank() == 1) {
        // Print a contiguous slice of items from the buffer in one line.
        // The number of items printed is controlled by the user through format syntax.
        try writer.writeByteNTimes(' ', indent_level);
        switch (shape.dtype()) {
            inline else => |dt| {
                const values = Shaped(dt.toZigType(), shape, slice);
                switch (comptime dt.class()) {
                    .float => try stdx.fmt.formatFloatSlice(values, options, writer),
                    .integer => try stdx.fmt.formatIntSlice(values, options, writer),
                    .bool, .complex => try stdx.fmt.formatAnySlice(values, options, writer),
                }
            },
        }
        try writer.writeByte('\n');
        return;
    }
    // TODO: consider removing the \n if dim is 1 for this axis.
    try writer.writeByteNTimes(' ', indent_level);
    _ = try writer.write("{\n");
    defer {
        writer.writeByteNTimes(' ', indent_level) catch {};
        _ = writer.write("},\n") catch {};
    }

    // Write first rows
    const n: u64 = @intCast(shape.dim(0));
    for (0..@min(num_rows, n)) |d| {
        const di: i64 = @intCast(d);
        const new_shape, const sliced = slice1d(shape, slice, 0, .{ .start = di, .end = di + 1 });
        try prettyPrintIndented(new_shape.drop(0), sliced, writer, num_rows, indent_level + 2, options);
    }

    if (n < num_rows) return;
    // Skip middle rows
    if (n > 2 * num_rows) {
        try writer.writeByteNTimes(' ', indent_level + 2);
        _ = try writer.write("...\n");
    }
    // Write last rows
    for (@max(n - num_rows, num_rows)..n) |d| {
        const di: i64 = @intCast(d);
        const new_shape, const sliced = slice1d(shape, slice, 0, .{ .start = di, .end = di + 1 });
        try prettyPrintIndented(new_shape.drop(0), sliced, writer, num_rows, indent_level + 2, options);
    }
}
