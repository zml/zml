const std = @import("std");

const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const DataType = @import("dtype.zig").DataType;
const floats = @import("floats.zig");
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
    _strides: [Shape.MAX_RANK]i64,
    _data: [*]const u8,
    _memory: union(enum) {
        managed: std.mem.Alignment,
        unmanaged,
    } = .unmanaged,

    /// Allocates a HostBuffer with the given shape.
    /// The memory is left undefined.
    /// The caller owns the memory, and need to call `deinit()`.
    pub fn empty(allocator: std.mem.Allocator, sh: Shape) error{OutOfMemory}!HostBuffer {
        return .{
            ._shape = sh,
            ._strides = sh.computeStrides().buffer,
            ._data = (try allocator.alignedAlloc(u8, 64, sh.byteSize())).ptr,
            ._memory = .{ .managed = .@"64" },
        };
    }

    /// Wraps an exisiting slice of bytes into a HostBuffer.
    /// The returned HostBuffer doesn't take ownership of the slice
    /// that will still need to be deallocated.
    pub fn fromBytes(shape_: Shape, data_: []const u8) HostBuffer {
        stdx.debug.assert(shape_.byteSize() == data_.len, "shape {} and data {} don't match", .{ shape_.byteSize(), data_.len });
        return .{
            ._shape = shape_,
            ._strides = shape_.computeStrides().buffer,
            ._data = data_.ptr,
            ._memory = .unmanaged,
        };
    }

    /// Frees the underlying memory if we owned it, ie if we've been created with `HostBuffer.empty`.
    pub fn deinit(self: *const HostBuffer, allocator: std.mem.Allocator) void {
        // This means we don't own the data.
        if (self._memory == .unmanaged) return;
        const log2_align = self._memory.managed;
        allocator.rawFree(self.mutBytes(), log2_align, @returnAddress());
    }

    /// Wraps an exisiting slice into a HostBuffer.
    /// The element type is inferred from the slice type.
    /// The returned HostBuffer doesn't take ownership of the slice
    /// that will still need to be deallocated.
    pub fn fromSlice(sh: anytype, s: anytype) HostBuffer {
        const shape_ = Shape.init(sh, DataType.fromSliceElementType(s));
        const raw_bytes = std.mem.sliceAsBytes(s);
        std.debug.assert(shape_.byteSize() == raw_bytes.len);
        return .{
            ._shape = shape_,
            ._strides = shape_.computeStrides().buffer,
            ._data = raw_bytes.ptr,
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
            ._data = @alignCast(std.mem.sliceAsBytes(s).ptr),
            ._strides = tmp,
            ._memory = .unmanaged,
        };
    }

    /// Creates a tensor from a **pointer** to a "multi dimension" array.
    /// Note this doesn't copy, the pointee array need to survive the `HostBuffer` object.
    /// Typically this is use with constant arrays.
    pub fn fromArray(arr_ptr: anytype) HostBuffer {
        const T = @TypeOf(arr_ptr.*);
        const sh = parseArrayInfo(T);
        std.debug.assert(sh.byteSize() == @sizeOf(T));
        return .{
            ._shape = sh,
            ._strides = sh.computeStrides().buffer,
            ._data = @ptrCast(arr_ptr),
            ._memory = .unmanaged,
        };
    }

    /// Returns a HostBuffer tagged with the tags in 'tagz'.
    pub fn withTags(self: HostBuffer, tagz: anytype) HostBuffer {
        var res = self;
        res._shape = self._shape.withTags(tagz);
        return res;
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
        const res = try empty(allocator, Shape.init(.{n_steps}, dt));
        switch (dt) {
            inline else => |d| if (comptime d.class() != .integer) {
                stdx.debug.assert(dt.class() == .integer, "arange expects type to be integer, got {} instead.", .{dt});
            } else {
                const Zt = d.toZigType();
                var j: i64 = args.start;
                for (res.mutItems(Zt)) |*val| {
                    val.* = @intCast(j);
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
        return try self.toDeviceOpts(platform_, .{});
    }

    /// Copies this HostBuffer to the given accelerator (with options).
    pub fn toDeviceOpts(self: HostBuffer, platform_: Platform, opts: Buffer.FromOptions) !Buffer {
        return try Buffer.from(platform_, self, opts);
    }

    /// Interpret the underlying data as a contiguous slice.
    /// WARNING: It's only valid if the buffer is contiguous.
    /// Strided buffers can't use this method.
    pub fn items(self: HostBuffer, comptime T: type) []const T {
        // TODO we should allow interpreting the output as @Vector(8, f32) when the tensor is f32.
        stdx.debug.assert(DataType.fromZigType(T) == self.dtype(), "Can't reinterpret {} as {s}", .{ self, @typeName(T) });
        stdx.debug.assert(self.isContiguous(), "{} isn't contiguous, can't interpret as []const u8", .{self});
        const ptr: [*]const T = @alignCast(@ptrCast(self._data));
        return ptr[0..self._shape.count()];
    }

    pub fn mutItems(self: HostBuffer, comptime T: type) []T {
        return @constCast(self.items(T));
    }

    pub fn bytes(self: HostBuffer) []const u8 {
        stdx.debug.assert(self.isContiguous(), "{} isn't contiguous, can't interpret as []const u8", .{self});
        return self._data[0..self._shape.byteSize()];
    }

    pub fn mutBytes(self: HostBuffer) []u8 {
        return @constCast(self.bytes());
    }

    pub fn shape(self: HostBuffer) Shape {
        return self._shape;
    }

    pub fn dtype(self: HostBuffer) DataType {
        return self._shape.dtype();
    }

    pub fn strides(self: *const HostBuffer) []const i64 {
        // Pass strides per pointer otherwise we return a pointer to this stack frame.
        return self._strides[0..self._shape.rank()];
    }

    // TODO: rename .data into ._data and make it a [*]u8
    // pub fn data(self: HostBuffer) []const u8 {
    //     return self.data;
    // }

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
        const _strides = self._strides;
        const cont_strides = self._shape.computeStrides();
        for (self._shape.dims(), _strides[0..self.rank()], cont_strides.constSlice()) |d, stride, cont_stride| {
            if (d != 1 and stride != cont_stride) return false;
        }
        return true;
    }

    pub fn reshape(self: HostBuffer, shape_: anytype) HostBuffer {
        stdx.debug.assert(self.isContiguous(), "reshape expects a contiguous tensor, got: {}", .{self});
        var res = self;
        res._shape = self._shape.reshape(shape_);
        res._strides = res._shape.computeStrides().buffer;
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
        stdx.debug.assert(start >= 0 and start < d, "slice1d({}, {}) expects the slice start to be between 0 and {} got: {}", .{ self, ax, d, s });
        stdx.debug.assert(end >= 1 and end <= d, "slice1d({}, {}) expects the slice end to be between 1 and {} got: {}", .{ self, ax, d, s });
        stdx.debug.assert(start < end, "slice1d({}, {}) expects the slice start ({}) to be smaller than the end ({}), got: {}", .{ self, ax, start, end, s });

        const offset: usize = @intCast(start * self._strides[ax]);
        const new_shape = self.shape().set(ax, end - start);
        return .{
            ._shape = new_shape,
            ._data = self._data[offset..],
            ._strides = self._strides,
            ._memory = .unmanaged,
        };
    }

    pub fn choose1d(self: HostBuffer, axis_: anytype, start: i64) HostBuffer {
        const ax = self.axis(axis_);
        return self.slice1d(ax, .{ .start = start, .end = start + 1 }).squeeze(ax);
    }

    pub fn choose(self: HostBuffer, offsets: anytype) HostBuffer {
        const off, const tags = Shape.parseDimensions(offsets);
        var sh = self._shape;
        var offset: i64 = 0;
        for (off.constSlice(), tags.constSlice()) |o, t| {
            const ax = sh.axis(t);
            offset += o * self._strides[ax];
            sh._dims.buffer[ax] = 0;
        }

        var new_strides: [Shape.MAX_RANK]i64 = @splat(self.dtype().sizeOf());

        // TODO rewrite with simd. This is a pshuf, but it's not supported by @shuffle.
        var res_ax: u32 = 0;
        for (0..self._shape.rank()) |ax| {
            if (sh._dims.buffer[ax] == 0) {
                continue;
            }

            sh._dims.buffer[res_ax] = self._shape._dims.buffer[ax];
            sh._tags.buffer[res_ax] = self._shape._tags.buffer[ax];
            new_strides[res_ax] = self._strides[ax];
            res_ax += 1;
        }
        sh._dims.len -= off.len;
        sh._tags.len -= off.len;

        return HostBuffer{
            ._shape = sh,
            ._strides = new_strides,
            ._data = self._data[@intCast(offset)..],
            ._memory = .unmanaged,
        };
    }

    pub fn squeeze(self: HostBuffer, axis_: anytype) HostBuffer {
        const ax = self._shape.axis(axis_);
        stdx.debug.assert(self.dim(ax) == 1, "squeeze expects a 1-d axis got {} in {}", .{ ax, self });

        var strd: std.BoundedArray(i64, Shape.MAX_RANK) = .{ .buffer = self._strides, .len = self.rank() };
        _ = strd.orderedRemove(ax);

        return .{
            ._shape = self.shape().drop(ax),
            ._data = self._data,
            ._strides = strd.buffer,
            ._memory = self._memory,
        };
    }

    pub fn format(
        self: HostBuffer,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        if (std.mem.eql(u8, fmt, "v")) {
            try writer.print("HostBuffer(.{_})@0x{x}", .{ self._shape, @intFromPtr(self._data) });
        } else {
            try writer.print("HostBuffer(.{_})", .{self._shape});
        }
    }

    /// Formatter for a HostBuffer that also print the values not just the shape.
    /// Usage: `std.log.info("my buffer: {}", .{buffer.pretty()});`
    pub fn pretty(self: HostBuffer) PrettyPrinter {
        return .{ .x = self };
    }

    pub const PrettyPrinter = struct {
        x: HostBuffer,

        pub fn format(self: PrettyPrinter, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            const fmt_: stdx.fmt.Fmt = switch (self.x.dtype().class()) {
                .integer => .parse(i32, fmt),
                .float => .parse(f32, fmt),
                else => .parse(void, fmt),
            };
            try prettyPrint(self.x, writer, .{ .fmt = fmt_, .options = options });
        }
    };

    pub fn prettyPrint(self: HostBuffer, writer: anytype, options: stdx.fmt.FullFormatOptions) !void {
        return self.prettyPrintIndented(writer, 4, 0, options);
    }

    fn prettyPrintIndented(self: HostBuffer, writer: anytype, num_rows: u8, indent_level: u8, options: stdx.fmt.FullFormatOptions) !void {
        if (self.rank() == 0) {
            // Special case input tensor is a scalar
            return switch (self.dtype()) {
                inline else => |dt| {
                    const val: dt.toZigType() = self.items(dt.toZigType())[0];
                    return switch (comptime dt.class()) {
                        // Since we have custom floats, we need to explicitly convert to float32 ourselves.
                        .float => stdx.fmt.formatFloatValue(floats.floatCast(f32, val), options, writer),
                        .integer => stdx.fmt.formatIntValue(val, options, writer),
                        .bool, .complex => stdx.fmt.formatAnyValue(val, options, writer),
                    };
                },
            };
        }
        if (self.rank() == 1) {
            // Print a contiguous slice of items from the buffer in one line.
            // The number of items printed is controlled by the user through format syntax.
            try writer.writeByteNTimes(' ', indent_level);
            switch (self.dtype()) {
                inline else => |dt| {
                    const values = self.items(dt.toZigType());
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
        const n: u64 = @intCast(self.dim(0));
        for (0..@min(num_rows, n)) |d| {
            const di: i64 = @intCast(d);
            const sliced_self = self.slice1d(0, .{ .start = di, .end = di + 1 }).squeeze(0);
            try sliced_self.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
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
            const sliced_self = self.slice1d(0, .{ .start = di, .end = di + 1 }).squeeze(0);
            try sliced_self.prettyPrintIndented(writer, num_rows, indent_level + 2, options);
        }
    }
};

fn parseArrayInfo(T: type) Shape {
    return switch (@typeInfo(T)) {
        .array => |arr| {
            const s = parseArrayInfo(arr.child);
            return s.insert(0, .{arr.len});
        },
        else => .{ ._dtype = DataType.fromZigType(T) },
    };
}
