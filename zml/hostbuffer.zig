const std = @import("std");

const Buffer = @import("buffer.zig").Buffer;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const Platform = @import("platform.zig").Platform;
const meta = @import("meta.zig");

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

    /// Allocates a HostBuffer with the given shape.
    /// The memory is initialized with increasing numbers.
    /// The caller owns the memory, and need to call `deinit()`.
    pub fn arange(allocator: std.mem.Allocator, args: Tensor.ArangeArgs, dt: DataType) !HostBuffer {
        meta.assert(args.start < args.end, "arange expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        meta.assert(args.step > 0, "arange expects 'args.step' to be positive, got {}", .{args.step});

        const n_steps = std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable;
        const b = dt.sizeOf();
        const res = try empty(allocator, Shape.init(.{n_steps}, dt));
        meta.assert(dt.class() == .integer, "arange expects type to be integer, got {} instead.", .{dt});
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

    /// Embeds a tensor with concrete values into an Mlir program.
    /// The content is copied, so the HostBuffer can be safely `deinit`.
    pub fn toStaticTensor(self: HostBuffer) Tensor {
        return Tensor.staticTensor(self.shape(), self.data);
    }

    /// Copies this HostBuffer to the given accelerator.
    pub fn toDevice(self: HostBuffer, platform_: Platform) !Buffer {
        return try Buffer.from(platform_, self);
    }

    pub fn items(self: HostBuffer, comptime T: type) []const T {
        if (DataType.fromZigType(T) != self.dtype()) {
            std.debug.panic("Can't reinterpret HostBuffer({_}) as {s}", .{ self.shape(), @typeName(T) });
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

    pub fn strides(self: HostBuffer) ?[]const i64 {
        return self._strides;
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

    pub fn dim(self: HostBuffer, axis: anytype) i64 {
        return self._shape.dim(axis);
    }

    pub fn reshape(self: HostBuffer, shape_: anytype) HostBuffer {
        meta.assert(self._strides == null, "reshape expects a contiguous tensor, got: {}", .{self});
        var res = self;
        res._shape = self._shape.reshape(shape_);
        return res;
    }

    pub const Slice = struct {
        single: ?i64 = null,
        start: i64 = 0,
        end: ?i64 = null,
        step: i64 = 1,
    };

    pub inline fn copySlice1d(self: HostBuffer, allocator: std.mem.Allocator, axis: i8, _args: Slice) !HostBuffer {
        var slices = [_]Slice{.{}} ** 5;
        slices[self._shape.axis(axis)] = _args;
        return copySlice(self, allocator, slices[0..self._shape.rank()]);
    }

    pub fn copySlice(self: HostBuffer, allocator: std.mem.Allocator, slices: []const Slice) !HostBuffer {
        const byte_size = self.dtype().sizeOf();
        var start_indices = [_]usize{0} ** 5;
        var strides_ = [_]usize{1} ** 5;
        const dims = self._shape.dims();
        var sh = self._shape;

        for (slices, 0..) |_args, a| {
            const args: Slice = .{
                .start = if (_args.start >= 0) _args.start else _args.start + dims[a],
                .end = _args.end orelse dims[a],
                .step = _args.step,
            };
            start_indices[a] = @intCast(args.start);
            strides_[a] = @intCast(args.step);
            sh._dims.set(a, b: {
                const range = args.end.? - args.start;
                const counts = @divFloor(range - 1, args.step) + 1;
                break :b counts;
            });
        }

        const rk = self.rank();
        meta.assert(rk <= 5, "copySlice only supports less than 5-D tensors. Received: {}", .{self});
        const raw_strides: [Shape.MAX_RANK]usize = blk: {
            var res: [Shape.MAX_RANK]usize = undefined;
            const _strides = self._shape.computeStrides(self.dtype().sizeOf());
            for (_strides.constSlice(), 0..rk) |stride, i| res[i] = @intCast(stride);
            break :blk res;
        };

        const result_tensor = try HostBuffer.empty(allocator, sh);

        const res_strides: [Shape.MAX_RANK]usize = blk: {
            var res: [Shape.MAX_RANK]usize = undefined;
            const _strides = self._shape.computeStrides(self.dtype().sizeOf());
            for (_strides.constSlice(), 0..rk) |stride, i| res[i] = @intCast(stride);
            break :blk res;
        };

        const src_data = self.data;
        const data_ = @constCast(result_tensor.data);
        for (0..@intCast(sh.dim(0))) |j0| {
            const off0 = (j0 * strides_[0] + start_indices[0]) * raw_strides[0];
            const res_off0 = j0 * res_strides[0];
            if (rk == 1) {
                @memcpy(data_[res_off0..][0..byte_size], src_data[off0..][0..byte_size]);
                continue;
            }
            for (0..@intCast(sh.dim(1))) |j1| {
                const off1 = off0 + (j1 * strides_[1] + start_indices[1]) * raw_strides[1];
                const res_off1 = res_off0 + j1 * res_strides[1];
                if (rk == 2) {
                    @memcpy(data_[res_off1..][0..byte_size], src_data[off1..][0..byte_size]);
                    continue;
                }
                for (0..@intCast(sh.dim(2))) |j2| {
                    const off2 = off1 + (j2 * strides_[2] + start_indices[2]) * raw_strides[2];
                    const res_off2 = res_off1 + j2 * res_strides[2];
                    if (rk == 3) {
                        @memcpy(data_[res_off2..][0..byte_size], src_data[off2..][0..byte_size]);
                        continue;
                    }
                    for (0..@intCast(sh.dim(3))) |j3| {
                        const off3 = off2 + (j3 * strides_[3] + start_indices[3]) * raw_strides[3];
                        const res_off3 = res_off2 + j3 * res_strides[3];
                        if (rk == 4) {
                            @memcpy(data_[res_off3..][0..byte_size], src_data[off3..][0..byte_size]);
                            continue;
                        }
                        for (0..@intCast(sh.dim(4))) |j4| {
                            const off4 = off3 + (j4 * strides_[4] + start_indices[4]) * raw_strides[4];
                            const res_off4 = res_off3 + j4 * res_strides[4];
                            @memcpy(data_[res_off4..][0..byte_size], src_data[off4..][0..byte_size]);
                        }
                    }
                }
            }
        }

        return result_tensor;
    }

    test copySlice {
        var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena_state.deinit();
        const allocator = arena_state.allocator();

        const x = HostBuffer.fromSlice(.{ 2, 5 }, &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
        {
            const res = try copySlice1d(x, allocator, 0, .{ .end = 1 });
            try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4 }, res.items(f32));
        }
        // { // failing
        //     const res = try copySlice1d(x, allocator, -1, .{ .start = -2 });
        //     try testing.expectEqualSlices(f32, &.{ 3, 4, 8, 9 }, res.items(f32));
        // }
        // {// failing
        //     const res = try copySlice1d(x, allocator, 1, .{ .start = 1, .step = 2 });
        //     try testing.expectEqualSlices(f32, &.{ 1, 3, 6, 8 }, res.items(f32));
        // }
        {
            const res = try copySlice(x, allocator, &.{ .{ .start = 1 }, .{ .start = 1, .step = 2 } });
            try std.testing.expectEqualSlices(f32, &.{ 6, 8 }, res.items(f32));
        }
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
