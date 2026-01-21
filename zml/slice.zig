const std = @import("std");

const stdx = @import("stdx");
const floats = @import("floats.zig");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;

pub fn isBytes(comptime T: type) bool {
    const type_info = @typeInfo(T);
    if (type_info != .pointer) return false;
    if (type_info.pointer.child == u8) return true;
    const child_type_info = @typeInfo(type_info.pointer.child);
    if (child_type_info != .array) return false;
    if (child_type_info.array.child != u8) return false;
    return true;
}

pub const Slice = struct {
    inner_data: union(enum) {
        mutable: []u8,
        immutable: []const u8,
    },
    shape: Shape,

    pub fn init(shape: Shape, bytes: anytype) Slice {
        stdx.debug.assertComptime(isBytes(@TypeOf(bytes)), "Expected \"bytes\" to be a []u8 or []const u8, got {s}", .{@typeName(@TypeOf(bytes))});
        const type_info = @typeInfo(@TypeOf(bytes));
        return if (type_info.pointer.is_const)
            .{ .inner_data = .{ .immutable = bytes }, .shape = shape }
        else
            .{ .inner_data = .{ .mutable = bytes }, .shape = shape };
    }

    pub fn alloc(allocator: std.mem.Allocator, shape_: Shape) !Slice {
        const size = shape_.byteSize();
        const bytes: []u8 = switch (shape_.dtype().alignOf()) {
            1 => try allocator.alignedAlloc(u8, .@"1", size),
            2 => try allocator.alignedAlloc(u8, .@"2", size),
            4 => try allocator.alignedAlloc(u8, .@"4", size),
            8 => try allocator.alignedAlloc(u8, .@"8", size),
            16 => try allocator.alignedAlloc(u8, .@"16", size),
            32 => try allocator.alignedAlloc(u8, .@"32", size),
            64 => try allocator.alignedAlloc(u8, .@"64", size),
            else => |v| stdx.debug.panic("Unsupported alignment: {}", .{v}),
        };

        return .{ .inner_data = .{ .mutable = bytes }, .shape = shape_ };
    }

    pub fn free(slice: Slice, allocator: std.mem.Allocator) void {
        const d = slice.constData();

        switch (slice.shape.dtype().alignOf()) {
            1 => allocator.free(@as([]align(1) const u8, @alignCast(d))),
            2 => allocator.free(@as([]align(2) const u8, @alignCast(d))),
            4 => allocator.free(@as([]align(4) const u8, @alignCast(d))),
            8 => allocator.free(@as([]align(8) const u8, @alignCast(d))),
            16 => allocator.free(@as([]align(16) const u8, @alignCast(d))),
            32 => allocator.free(@as([]align(32) const u8, @alignCast(d))),
            64 => allocator.free(@as([]align(64) const u8, @alignCast(d))),
            else => |v| stdx.debug.panic("Unsupported alignment: {}", .{v}),
        }
    }

    pub fn dtype(slice: Slice) DataType {
        return slice.shape.dtype();
    }

    pub fn data(slice: Slice) []u8 {
        return switch (slice.inner_data) {
            .mutable => |d| d,
            else => stdx.debug.panic("Expected slice to be mutable but it's immutable", .{}),
        };
    }

    pub fn constData(slice: Slice) []const u8 {
        return switch (slice.inner_data) {
            inline else => |d| d,
        };
    }

    pub fn items(slice: Slice, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, slice.data()));
    }

    pub fn constItems(slice: Slice, comptime T: type) []const T {
        return @alignCast(std.mem.bytesAsSlice(T, slice.constData()));
    }

    pub fn format(
        slice: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        return writer.print("{any}", .{slice});
    }

    pub fn formatNumber(slice: Slice, writer: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
        return slice.prettyPrintIndented(writer, 4, 0, n);
    }

    pub fn prettyPrint(slice: Slice, writer: *std.Io.Writer, options: std.fmt.Number) !void {
        return slice.prettyPrintIndented(writer, 4, 0, options);
    }

    fn prettyPrintIndented(slice: Slice, writer: *std.Io.Writer, num_rows: u8, indent_level: u8, options: std.fmt.Number) !void {
        if (slice.shape.rank() == 0) {
            // Special case input tensor is a scalar
            return switch (slice.dtype()) {
                inline else => |dt| {
                    const val: dt.toZigType() = slice.constItems(dt.toZigType())[0];
                    return switch (comptime dt.class()) {
                        // Since we have custom floats, we need to explicitly convert to float32 ourselves.
                        .float => stdx.fmt.formatFloat(floats.floatCast(f32, val), options, writer),
                        .integer => stdx.fmt.formatInt(val, options, writer),
                        .bool => stdx.fmt.formatBool(val, options, writer),
                        .complex => stdx.fmt.formatComplex(val, options, writer),
                    };
                },
            };
        }

        if (slice.shape.rank() == 1) {
            // Print a contiguous slice of items from the buffer in one line.
            // The number of items printed is controlled by the user through format syntax.
            try writer.splatByteAll(' ', indent_level);
            switch (slice.dtype()) {
                inline else => |dt| {
                    const T = dt.toZigType();
                    // TODO: handle negative strides
                    const elem_strides: i64 = slice.shape.computeElementStrides().get(0);
                    const values = slice.constItems(T);
                    switch (comptime dt.class()) {
                        .float => try stdx.fmt.formatFloatSlice(values, options, elem_strides, writer),
                        .integer => try stdx.fmt.formatIntSlice(values, options, elem_strides, writer),
                        .complex => try stdx.fmt.formatComplexSlice(values, options, elem_strides, writer),
                        .bool => try stdx.fmt.formatBoolSlice(values, options, elem_strides, writer),
                    }
                },
            }
            return;
        }
        // TODO: consider removing the \n if dim is 1 for this axis.
        try writer.splatByteAll(' ', indent_level);
        _ = try writer.write("{\n");

        // Write first rows
        const n: u64 = @intCast(slice.shape.dim(0));
        for (0..@min(num_rows, n)) |d| {
            const byte_stride: usize = @intCast(slice.shape.computeByteStrides().get(0));
            const sub_slice: Slice = .init(slice.shape.drop(0), slice.constData()[d * byte_stride ..][0..byte_stride]);
            try sub_slice.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
            if (slice.shape.rank() > 1) {
                try writer.writeAll(",\n");
            }
        }

        if (n < num_rows) return;
        // Skip middle rows
        if (n > 2 * num_rows) {
            try writer.splatByteAll(' ', indent_level + 2);
            _ = try writer.write("...\n");
        }
        // Write last rows
        for (@max(n - num_rows, num_rows)..n) |d| {
            const byte_stride: usize = @intCast(slice.shape.computeByteStrides().get(0));
            const sub_slice: Slice = .init(slice.shape.drop(0), slice.constData()[d * byte_stride ..][0..byte_stride]);
            try sub_slice.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
            if (slice.shape.rank() > 1) {
                try writer.writeAll(",\n");
            }
        }

        try writer.splatByteAll(' ', indent_level);
        _ = try writer.write("}");
    }
};

test "slice pretty print rank 3" {
    const data: [4][4][4]i32 = .{
        .{
            .{ 0, 1, 2, 3 },
            .{ 4, 5, 6, 7 },
            .{ 8, 9, 10, 11 },
            .{ 12, 13, 14, 15 },
        },
        .{
            .{ 16, 17, 18, 19 },
            .{ 20, 21, 22, 23 },
            .{ 24, 25, 26, 27 },
            .{ 28, 29, 30, 31 },
        },
        .{
            .{ 32, 33, 34, 35 },
            .{ 36, 37, 38, 39 },
            .{ 40, 41, 42, 43 },
            .{ 44, 45, 46, 47 },
        },
        .{
            .{ 48, 49, 50, 51 },
            .{ 52, 53, 54, 55 },
            .{ 56, 57, 58, 59 },
            .{ 60, 61, 62, 63 },
        },
    };
    const slice = Slice.init(.init(.{ 4, 4, 4 }, .i32), std.mem.asBytes(&data));
    const expected =
        \\{
        \\  {
        \\    {0,1,2,3},
        \\    {4,5,6,7},
        \\    {8,9,10,11},
        \\    {12,13,14,15},
        \\  },
        \\  {
        \\    {16,17,18,19},
        \\    {20,21,22,23},
        \\    {24,25,26,27},
        \\    {28,29,30,31},
        \\  },
        \\  {
        \\    {32,33,34,35},
        \\    {36,37,38,39},
        \\    {40,41,42,43},
        \\    {44,45,46,47},
        \\  },
        \\  {
        \\    {48,49,50,51},
        \\    {52,53,54,55},
        \\    {56,57,58,59},
        \\    {60,61,62,63},
        \\  },
        \\}
    ;

    try std.testing.expectFmt(expected, "{d}", .{slice});
}

test "slice pretty print rank 1" {
    const data: [4]i32 = .{ 0, 1, 2, 3 };
    const slice = Slice.init(.init(.{4}, .i32), std.mem.asBytes(&data));
    const expected = "{0,1,2,3}";
    try std.testing.expectFmt(expected, "{d}", .{slice});
}

test "slice pretty print rank 0" {
    const data: [1]i32 = .{0};
    const slice = Slice.init(.init(.{}, .i32), std.mem.asBytes(&data));
    const expected = "0";
    try std.testing.expectFmt(expected, "{d}", .{slice});
}

test "slice pretty print ellipsis" {
    const data: [9][1]i32 = .{ .{0}, .{1}, .{2}, .{3}, .{4}, .{5}, .{6}, .{7}, .{8} };
    const slice = Slice.init(.init(.{ 9, 1 }, .i32), std.mem.asBytes(&data));
    const expected =
        \\{
        \\  {0},
        \\  {1},
        \\  {2},
        \\  {3},
        \\  ...
        \\  {5},
        \\  {6},
        \\  {7},
        \\  {8},
        \\}
    ;
    try std.testing.expectFmt(expected, "{d}", .{slice});
}
