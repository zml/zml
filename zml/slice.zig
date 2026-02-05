const std = @import("std");

const stdx = @import("stdx");

const constants = @import("constants.zig");
const DataType = @import("dtype.zig").DataType;
const floats = @import("floats.zig");
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
    offset_bytes: usize = 0,
    byte_strides: stdx.BoundedArray(i64, constants.MAX_RANK),

    pub fn init(shape: Shape, bytes: anytype) Slice {
        stdx.debug.assertComptime(isBytes(@TypeOf(bytes)), "Expected \"bytes\" to be a []u8 or []const u8, got {s}", .{@typeName(@TypeOf(bytes))});
        const type_info = @typeInfo(@TypeOf(bytes));
        const byte_strides = shape.computeByteStrides();
        return if (type_info.pointer.is_const)
            .{ .inner_data = .{ .immutable = bytes }, .shape = shape, .offset_bytes = 0, .byte_strides = byte_strides }
        else
            .{ .inner_data = .{ .mutable = bytes }, .shape = shape, .offset_bytes = 0, .byte_strides = byte_strides };
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

        return .{
            .inner_data = .{ .mutable = bytes },
            .shape = shape_,
            .offset_bytes = 0,
            .byte_strides = shape_.computeByteStrides(),
        };
    }

    pub fn free(slice: Slice, allocator: std.mem.Allocator) void {
        const d = slice.constData_();

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

    pub fn isContiguous(slice: Slice) bool {
        const expected = slice.shape.computeByteStrides();
        const rank = slice.shape.rank();
        for (0..rank) |ax| {
            if (slice.byte_strides.get(ax) != expected.get(ax)) return false;
        }
        return true;
    }

    /// Returns the actual element-wise strides for this specific slice.
    /// This respects transpositions and non-contiguous views.
    pub fn elementStrides(self: Slice) stdx.BoundedArray(i64, constants.MAX_RANK) {
        const element_size = @as(i64, @intCast(self.dtype().sizeOf()));
        var res = self.byte_strides;
        for (res.slice()) |*stride| {
            // We divide the actual byte strides by the size of the data type
            stride.* = @divExact(stride.*, element_size);
        }
        return res;
    }

    pub fn subSlice(slice: Slice, axis: anytype, start: i64, size: i64) Slice {
        const dim = slice.shape.dim(axis);

        stdx.debug.assert(start >= 0 and size >= 0 and start + size <= dim, "subSlice out of bounds: axis {d} start {d} size {d} dim {d}", .{ axis, start, size, dim });

        const stride_bytes = slice.byte_strides.get(axis);
        const new_offset = @as(i64, @intCast(slice.offset_bytes)) + start * stride_bytes;
        stdx.debug.assert(new_offset >= 0, "subSlice produced negative offset: {d}", .{new_offset});

        var res = slice;
        res.offset_bytes = @intCast(new_offset);
        res.shape = slice.shape.set(axis, size);
        return res;
    }

    fn dropAxis(slice: Slice, axis_: anytype) Slice {
        const axis = slice.shape.axis(axis_);
        var res = slice;
        res.shape = slice.shape.drop(axis);
        _ = res.byte_strides.orderedRemove(@intCast(axis));
        return res;
    }

    pub fn data(slice: Slice) []u8 {
        const base = slice.data_();
        stdx.debug.assert(slice.offset_bytes <= base.len, "Slice offset exceeds data length", .{});
        return base[slice.offset_bytes..];
    }

    fn data_(slice: Slice) []u8 {
        return switch (slice.inner_data) {
            .mutable => |d| d,
            else => stdx.debug.panic("Expected slice to be mutable but it's immutable", .{}),
        };
    }

    pub fn constData(slice: Slice) []const u8 {
        const base = slice.constData_();
        stdx.debug.assert(slice.offset_bytes <= base.len, "Slice offset exceeds data length", .{});
        return base[slice.offset_bytes..];
    }

    fn constData_(slice: Slice) []const u8 {
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
                    const n = slice.shape.dim(0);

                    const stride = @divExact(slice.byte_strides.get(0), @as(i64, @intCast(@sizeOf(T))));

                    const needed_len: usize = if (n == 0) 0 else @intCast(@abs((n - 1) * stride) + 1);
                    const values = slice.constItems(T)[0..needed_len];

                    switch (comptime dt.class()) {
                        .float => try stdx.fmt.formatFloatSlice(values, options, stride, writer),
                        .integer => try stdx.fmt.formatIntSlice(values, options, stride, writer),
                        .complex => try stdx.fmt.formatComplexSlice(values, options, stride, writer),
                        .bool => try stdx.fmt.formatBoolSlice(values, options, stride, writer),
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
            const sub_slice = slice.subSlice(0, @intCast(d), 1).dropAxis(0);
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
            const sub_slice = slice.subSlice(0, @intCast(d), 1).dropAxis(0);
            try sub_slice.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
            if (slice.shape.rank() > 1) {
                try writer.writeAll(",\n");
            }
        }

        try writer.splatByteAll(' ', indent_level);
        _ = try writer.write("}");
    }

    pub fn copyFromContiguous(destination: Slice, source: []const u8) void {
        const total_bytes = destination.shape.byteSize();
        stdx.debug.assert(source.len >= total_bytes, "Source buffer size ({}) smaller than destination shape ({})", .{ source.len, total_bytes });

        if (total_bytes == 0) return;

        if (destination.isContiguous()) {
            @memcpy(destination.data()[0..total_bytes], source[0..total_bytes]);
            return;
        }

        const rank = destination.shape.rank();
        stdx.debug.assert(rank > 0, "Non-contiguous slice with rank 0 is unexpected", .{});

        const outer_len: usize = @intCast(destination.shape.dim(0));
        stdx.debug.assert(outer_len > 0, "Invalid dimension size: {d}", .{outer_len});

        stdx.debug.assert(total_bytes % outer_len == 0, "Non-even split for contiguous source", .{});
        const outer_span_bytes = total_bytes / outer_len;

        for (0..outer_len) |i| {
            const destination_outer = destination.subSlice(0, @intCast(i), 1).dropAxis(0);
            const source_outer_start = i * outer_span_bytes;
            const source_outer_end = source_outer_start + outer_span_bytes;
            destination_outer.copyFromContiguous(source[source_outer_start..source_outer_end]);
        }
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
