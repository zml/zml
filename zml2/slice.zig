const std = @import("std");

const stdx = @import("stdx");
const floats = @import("floats.zig");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;

pub const Slice = struct {
    data: []u8,
    shape: Shape,

    pub fn init(shape: Shape, data: []u8) Slice {
        return .{ .data = data, .shape = shape };
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

        return .{ .data = bytes, .shape = shape_ };
    }

    pub fn free(slice: Slice, allocator: std.mem.Allocator) void {
        slice.constSlice().free(allocator);
    }

    pub fn constSlice(self: Slice) ConstSlice {
        return .{ .data = self.data, .shape = self.shape };
    }

    pub fn dtype(self: Slice) DataType {
        return self.shape.dtype();
    }

    pub fn items(self: Slice, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, self.data));
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        return writer.print("{any}", .{self});
    }

    pub fn formatNumber(self: @This(), writer: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
        return self.constSlice().formatNumber(writer, n);
    }
};

pub const ConstSlice = struct {
    data: []const u8,
    shape: Shape,

    pub fn init(shape: Shape, data: []const u8) ConstSlice {
        return .{ .data = data, .shape = shape };
    }

    pub fn free(slice: ConstSlice, allocator: std.mem.Allocator) void {
        switch (slice.shape.dtype().alignOf()) {
            1 => allocator.free(@as([]align(1) const u8, @alignCast(slice.data))),
            2 => allocator.free(@as([]align(2) const u8, @alignCast(slice.data))),
            4 => allocator.free(@as([]align(4) const u8, @alignCast(slice.data))),
            8 => allocator.free(@as([]align(8) const u8, @alignCast(slice.data))),
            16 => allocator.free(@as([]align(16) const u8, @alignCast(slice.data))),
            32 => allocator.free(@as([]align(32) const u8, @alignCast(slice.data))),
            64 => allocator.free(@as([]align(64) const u8, @alignCast(slice.data))),
            else => |v| stdx.debug.panic("Unsupported alignment: {}", .{v}),
        }
    }

    pub fn dtype(self: ConstSlice) DataType {
        return self.shape.dtype();
    }

    pub fn items(self: ConstSlice, comptime T: type) []const T {
        return @alignCast(std.mem.bytesAsSlice(T, self.data));
    }

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        return writer.print("{any}", .{self});
    }

    pub fn formatNumber(self: @This(), writer: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
        return self.prettyPrintIndented(writer, 4, 0, n);
    }

    pub fn prettyPrint(self: ConstSlice, writer: *std.Io.Writer, options: std.fmt.Number) !void {
        return self.prettyPrintIndented(writer, 4, 0, options);
    }

    fn prettyPrintIndented(self: ConstSlice, writer: *std.Io.Writer, num_rows: u8, indent_level: u8, options: std.fmt.Number) !void {
        if (self.shape.rank() == 0) {
            // Special case input tensor is a scalar
            return switch (self.dtype()) {
                inline else => |dt| {
                    const val: dt.toZigType() = self.items(dt.toZigType())[0];
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

        if (self.shape.rank() == 1) {
            // Print a contiguous slice of items from the buffer in one line.
            // The number of items printed is controlled by the user through format syntax.
            try writer.splatByteAll(' ', indent_level);
            switch (self.dtype()) {
                inline else => |dt| {
                    const T = dt.toZigType();
                    // TODO: handle negative strides
                    const elem_strides: i64 = self.shape.computeElementStrides().get(0);
                    const values = self.items(T);
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
        const n: u64 = @intCast(self.shape.dim(0));
        for (0..@min(num_rows, n)) |d| {
            const byte_stride: usize = @intCast(self.shape.computeByteStrides().get(0));
            const sub_slice: ConstSlice = .init(self.shape.drop(0), self.data[d * byte_stride ..][0..byte_stride]);
            try sub_slice.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
            if (self.shape.rank() > 1) {
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
            const byte_stride: usize = @intCast(self.shape.computeByteStrides().get(0));
            const sub_slice: ConstSlice = .init(self.shape.drop(0), self.data[d * byte_stride ..][0..byte_stride]);
            try sub_slice.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
            if (self.shape.rank() > 1) {
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
    const slice = ConstSlice.init(.init(.{ 4, 4, 4 }, .i32), std.mem.asBytes(&data));
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
    const slice = ConstSlice.init(.init(.{4}, .i32), std.mem.asBytes(&data));
    const expected = "{0,1,2,3}";
    try std.testing.expectFmt(expected, "{d}", .{slice});
}

test "slice pretty print rank 0" {
    const data: [1]i32 = .{ 0 };
    const slice = ConstSlice.init(.init(.{}, .i32), std.mem.asBytes(&data));
    const expected = "0";
    try std.testing.expectFmt(expected, "{d}", .{slice});
}

test "slice pretty print ellipsis" {
    const data: [9][1]i32 = .{ .{0}, .{1}, .{2}, .{3}, .{4}, .{5}, .{6}, .{7}, .{8} };
    const slice = ConstSlice.init(.init(.{ 9, 1 }, .i32), std.mem.asBytes(&data));
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
