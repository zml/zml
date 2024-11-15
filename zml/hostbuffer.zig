const std = @import("std");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

test {
    std.testing.refAllDecls(HostBuffer);
}

/// Represents a tensor with associated data allocated by user code.
/// If the memory is `.managed` it needs to be freed with `x.deinit(allocator)`
/// If the memory is `.unmanaged` it doesn't need to be freed (eg memory mapped, or tracked elsewhere).
pub const HostBuffer = struct {
    _shape: Shape,
    _strides: ?[Shape.MAX_RANK]i64 = null,
    data: []const u8,
    _memory: union(enum) {
        managed: u5,
        unmanaged,
    } = .unmanaged,

    /// Allocates a HostBuffer with the given shape.
    /// The memory is left undefined.
    /// The caller owns the memory, and need to call `deinit()`.
    pub fn empty(allocator: std.mem.Allocator, sh: Shape) !HostBuffer {
        return .{
            ._shape = sh,
            .data = try allocator.alignedAlloc(u8, std.atomic.cache_line, sh.byteSize()),
            ._memory = .{ .managed = std.math.log2_int(u16, std.atomic.cache_line) },
        };
    }

    /// Wraps an exisiting slice of bytes into a HostBuffer.
    /// The returned HostBuffer doesn't take ownership of the slice
    /// that will still need to be deallocated.
    pub fn fromBytes(shape_: Shape, data_: []const u8) HostBuffer {
        std.debug.assert(shape_.byteSize() == data_.len);
        return .{
            ._shape = shape_,
            .data = data_,
            ._memory = .unmanaged,
        };
    }

    /// Frees the underlying memory if we owned it, ie if we've been created with `HostBuffer.empty`.
    pub fn deinit(self: *const HostBuffer, allocator: std.mem.Allocator) void {
        // This means we don't own the data.
        if (self._memory == .unmanaged) return;
        const log2_align = self._memory.managed;
        allocator.rawFree(@constCast(self.data), log2_align, @returnAddress());
    }

    /// Wraps an exisiting slice into a HostBuffer.
    /// The element type is inferred from the slice type.
    /// The returned HostBuffer doesn't take ownership of the slice
    /// that will still need to be deallocated.
    pub fn fromSlice(sh: anytype, s: anytype) HostBuffer {
        const shape_ = Shape.init(sh, DataType.fromSliceElementType(s));
        std.debug.assert(shape_.count() == s.len);
        return .{
            ._shape = shape_,
            .data = @alignCast(std.mem.sliceAsBytes(s)),
            ._memory = .unmanaged,
        };
    }

    /// Wraps an exisiting slice into a HostBuffer.
    /// The element type is inferred from the slice type.
    /// The values in the slice doesn't need to be contiguous,
    /// strides can be specified.
    /// The returned HostBuffer doesn't take ownership of the slice.
    pub fn fromStridedSlice(sh: Shape, s: anytype, strides_: []const i64) HostBuffer {
        // std.debug.assert(sh.count() == s.len);
        var tmp: [Shape.MAX_RANK]i64 = undefined;
        @memcpy(tmp[0..strides_.len], strides_);
        return .{
            ._shape = sh,
            .data = @alignCast(std.mem.sliceAsBytes(s)),
            ._strides = tmp,
            ._memory = .unmanaged,
        };
    }

    /// Creates a tensor from a **pointer** to a "multi dimension" array.
    /// Note this doesn't copy, the pointee array need to survive the `HostBuffer` object.
    pub fn fromArray(arr_ptr: anytype) HostBuffer {
        const T = @TypeOf(arr_ptr.*);
        const sh = parseArrayInfo(T);
        return .{
            ._shape = sh,
            .data = @alignCast(std.mem.sliceAsBytes(arr_ptr)),
            // Array are typically stack allocated and don't need to be freed.
            ._memory = .unmanaged,
        };
    }

    pub const ArangeArgs = struct {
        start: i64 = 0,
        end: i64,
        step: i64 = 1,
    };

    /// Allocates a HostBuffer with the given shape.
    /// The memory is initialized with increasing numbers.
    /// The caller owns the memory, and need to call `deinit()`.
    pub fn arange(allocator: std.mem.Allocator, args: ArangeArgs, dt: DataType) !HostBuffer {
        stdx.debug.assert(args.start < args.end, "arange expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        stdx.debug.assert(args.step > 0, "arange expects 'args.step' to be positive, got {}", .{args.step});

        const n_steps = std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable;
        const b = dt.sizeOf();
        const res = try empty(allocator, Shape.init(.{n_steps}, dt));
        stdx.debug.assert(dt.class() == .integer, "arange expects type to be integer, got {} instead.", .{dt});
        var data_ = @constCast(res.data);
        switch (dt) {
            inline else => {
                var j: i64 = args.start;
                for (0..@intCast(n_steps)) |i| {
                    var v = Data.init(dt, j);
                    @memcpy(data_[i * b .. (i + 1) * b], v.constSlice());
                    j +%= args.step;
                }
            },
        }
        return res;
    }

    test arange {
        {
            var x = try arange(std.testing.allocator, .{ .end = 8 }, .i32);
            defer x.deinit(std.testing.allocator);
            try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, x.items(i32));
        }
        {
            var x = try arange(std.testing.allocator, .{ .start = -3, .end = 12, .step = 2 }, .i32);
            defer x.deinit(std.testing.allocator);
            try std.testing.expectEqualSlices(i32, &.{ -3, -1, 1, 3, 5, 7, 9, 11 }, x.items(i32));
        }
    }

    /// Copies this HostBuffer to the given accelerator.
    pub fn toDevice(self: HostBuffer, platform_: Platform) !Buffer {
        return try Buffer.from(platform_, self);
    }

    /// Interpret the underlying data as a contiguous slice.
    /// WARNING: It's only valid if the buffer is contiguous.
    /// Strided buffers can't use this method.
    pub fn items(self: HostBuffer, comptime T: type) []const T {
        if (DataType.fromZigType(T) != self.dtype()) {
            std.debug.panic("Can't reinterpret {} as {s}", .{ self, @typeName(T) });
        }
        if (!self.isContiguous()) {
            std.debug.panic("{} isn't contiguous", .{self});
        }
        const ptr: [*]const T = @alignCast(@ptrCast(self.data.ptr));
        return ptr[0..self._shape.count()];
    }

    pub fn shape(self: HostBuffer) Shape {
        return self._shape;
    }

    pub fn dtype(self: HostBuffer) DataType {
        return self._shape.dtype();
    }

    pub fn strides(self: *const HostBuffer) ?[]const i64 {
        // Pass strides per pointer otherwise we return a pointer to this stack frame.
        return if (self._strides) |*strd| strd[0..self.rank()] else null;
    }

    pub fn data(self: HostBuffer) []const u8 {
        return self.data;
    }

    pub inline fn rank(self: HostBuffer) u4 {
        return self._shape.rank();
    }

    pub inline fn count(self: HostBuffer) usize {
        return self._shape.count();
    }

    pub fn dim(self: HostBuffer, axis_: anytype) i64 {
        return self._shape.dim(axis_);
    }

    pub fn axis(self: HostBuffer, axis_: anytype) u3 {
        return self._shape.axis(axis_);
    }

    pub fn isContiguous(self: HostBuffer) bool {
        const strd = self._strides orelse return true;
        const cont_strides = self._shape.computeStrides();
        return std.mem.eql(i64, strd[0..self.rank()], cont_strides.constSlice());
    }

    pub fn reshape(self: HostBuffer, shape_: anytype) HostBuffer {
        stdx.debug.assert(self.isContiguous(), "reshape expects a contiguous tensor, got: {}", .{self});
        var res = self;
        res._shape = self._shape.reshape(shape_);
        return res;
    }

    pub const Slice = struct {
        start: i64 = 0,
        end: ?i64 = null,
    };

    /// Slices the input Tensor over the given axis_ using the given parameters.
    pub fn slice1d(self: HostBuffer, axis_: anytype, s: Slice) HostBuffer {
        const ax = self._shape.axis(axis_);
        const d = self.dim(ax);
        const start: i64 = if (s.start < 0) s.start + d else s.start;
        var end = s.end orelse d;
        if (end < 0) end += d;
        stdx.debug.assert(start >= 0 and start < d, "slice1d({}, {}) expects the slice start to be between 0 and {} got: {}", .{ self, ax, d, start });
        stdx.debug.assert(end >= 1 and end <= d, "slice1d({}, {}) expects the slice end to be between 1 and {} got: {}", .{ self, ax, d, end });
        stdx.debug.assert(start < end, "slice1d({}, {}) expects the slice start ({}) to be smaller than the end ({})", .{ self, ax, start, end });

        // If strides weren't set it means original buffer is contiguous.
        // But it won't be anymore after slicing. The strides don't change though.
        const _strides = self._strides orelse self._shape.computeStrides().buffer;
        const offset: usize = @intCast(start * _strides[ax]);
        return .{
            ._shape = self.shape().set(ax, end - start),
            .data = self.data[offset..],
            ._strides = _strides,
            ._memory = .unmanaged,
        };
    }

    pub fn format(
        self: HostBuffer,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("HostBuffer(.{_})", .{self._shape});
    }
};

fn parseArrayInfo(T: type) Shape {
    return switch (@typeInfo(T)) {
        .Array => |arr| {
            const s = parseArrayInfo(arr.child);
            return s.insert(0, .{arr.len});
        },
        else => .{ ._dtype = DataType.fromZigType(T) },
    };
}
