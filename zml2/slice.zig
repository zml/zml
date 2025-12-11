const std = @import("std");

const stdx = @import("stdx");

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
        _ = self; // autofix
        _ = n; // autofix
        return writer.print("TODO", .{});
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
        _ = self; // autofix
        _ = n; // autofix
        return writer.print("TODO", .{});
    }
};
