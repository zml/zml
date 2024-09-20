const std = @import("std");
const testing = std.testing;

const meta = @import("meta.zig");
const pjrt = @import("pjrt");
const pjrtx = @import("pjrtx.zig");

const Context = @import("context.zig").Context;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

test {
    std.testing.refAllDecls(Buffer);
}

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    _shape: Shape,
    _shards: Shape = undefined,
    _platform: Platform,
    _data: *pjrtx.Buffer,

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(platform: Platform, buf: HostBuffer) !Buffer {
        const pjrt_buffer = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, .{
            .data = buf.data,
            .buffer_type = pjrtx.Buffer.BufferTypeFromDType(buf.shape().dtype()),
            .dims = buf.shape().dims(),
            .byte_strides = buf.strides(),
            .device = platform.getDevices()[0],
            .host_buffer_semantics = .ImmutableUntilTransferCompletes,
        });
        return .{
            ._platform = platform,
            ._shape = buf.shape(),
            ._data = pjrt_buffer,
        };
    }

    /// Wraps a pre-exisiting `pjrt.Buffer` into a `zml.Buffer`.
    pub fn fromPjrtBuffer(platform: Platform, pjrt_buffer: *pjrtx.Buffer) Buffer {
        return .{
            ._platform = platform,
            ._shape = _shapeFromPjrtBuffer(platform, pjrt_buffer),
            ._data = pjrt_buffer,
        };
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromSlice(platform: Platform, dimz: anytype, s: anytype) !Buffer {
        const sh = Shape.init(dimz, DataType.fromSliceElementType(s));
        return from(platform, HostBuffer.fromBytes(sh, std.mem.sliceAsBytes(s)));
    }

    /// Copies the given Zig array to the accelerator memory and
    /// return a Buffer using the array shape.
    pub fn fromArray(platform: Platform, arr: anytype) !Buffer {
        const host_buffer = HostBuffer.fromArray(&arr);
        return try from(platform, host_buffer);
    }

    /// Creates a Buffer with a single element.
    pub fn scalar(platform: Platform, val: anytype, dtype_: DataType) !Buffer {
        const x = dtype_.constant(val);
        const host_buffer = HostBuffer.fromBytes(Shape.init(.{}, dtype_), x.constSlice());
        return try from(platform, host_buffer);
    }

    /// Creates a Buffer with a single element repeated manytime.
    pub fn constant(platform: Platform, shape_: Shape, val: anytype) !Buffer {
        const x = shape_.dtype().constant(val);
        const host_buffer: HostBuffer = .{
            ._shape = shape_,
            ._strides = [1]i64{0} ** Shape.MAX_RANK,
            .data = x.constSlice(),
        };
        return try from(platform, host_buffer);
    }

    test constant {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const x = try constant(platform, Shape.init(.{ 4, 3, 2 }, .u16), 42);
        const y = try x.getValue([4 * 3 * 2]u16);
        try std.testing.expectEqual([_]u16{42} ** (4 * 3 * 2), y);
    }

    /// Creates a Buffer as a view of memory visible from the device,
    /// thus avoiding a copy.
    ///
    /// On CUDA, it also allows you to specify a host allocated slice as they seem to be accessible.
    /// Be careful though, as it requires a specific alignment.
    /// Also note that it might not work on all platforms,
    /// could lead to crashes and is considerably slower.
    pub fn asViewOf(platform: Platform, buf: HostBuffer) !Buffer {
        const minor_to_major: [Shape.MAX_RANK]i64 = comptime blk: {
            var res: [Shape.MAX_RANK]i64 = undefined;
            for (0..Shape.MAX_RANK) |i| {
                res[i] = @intCast(Shape.MAX_RANK - i - 1);
            }
            break :blk res;
        };

        const pjrt_buffer = try platform.pjrt_client.createViewOfDeviceBuffer(platform.pjrt_api, .{
            .data = buf.data,
            .element_type = pjrtx.Buffer.BufferTypeFromDType(buf.shape().dtype()),
            .dims = buf.shape().dims(),
            .device = platform.getDevices()[0],
            .layout = .{
                .Tiled = .{
                    .minor_to_major = minor_to_major[Shape.MAX_RANK - buf.shape().rank() ..],
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
        });

        return .{
            ._platform = platform,
            ._shape = buf.shape(),
            ._data = pjrt_buffer,
        };
    }

    /// Fetches the content of the given buffer into a stack variable of the given type.
    pub fn getValue(self: Buffer, T: type) !T {
        meta.assert(self._shape.byteSize() == @sizeOf(T), "Buffer {} has {d} bytes of data, can't load it to a {s} with {d} bytes", .{ self, self._shape.byteSize(), @typeName(T), @sizeOf(T) });
        var res: T = undefined;
        try self._data.toHostBuffer(self._platform.pjrt_api, std.mem.asBytes(&res));
        return res;
    }

    /// Copies the content of the Buffer back to host, in the given buffer,
    /// and return a new `HostBuffer` object with the same shape.
    /// The returned `HostBuffer` doesn't own the memory.
    pub fn toHost(self: Buffer, output: []u8) !HostBuffer {
        try self._data.toHostBuffer(self._platform.pjrt_api, output);
        return HostBuffer.fromBytes(self.shape(), output);
    }

    /// Copies the content of the Buffer to the host.
    /// The returned `HostBuffer` does own the memory.
    pub fn toHostAlloc(self: Buffer, allocator: std.mem.Allocator) !HostBuffer {
        const output = try HostBuffer.empty(allocator, self.shape());
        try self._data.toHostBuffer(self._platform.pjrt_api, @constCast(output.data));
        return output;
    }

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *const Buffer) void {
        self._data.deinit(self._platform.pjrt_api);
    }

    /// This Buffer shape.
    pub fn shape(self: Buffer) Shape {
        return self._shape;
    }

    /// This Buffer shape as a slice of dims.
    pub fn dims(self: *const Buffer) []const i64 {
        return self._shape.dims();
    }

    /// This Buffer element type.
    pub fn dtype(self: Buffer) DataType {
        return self._shape.dtype();
    }

    /// This Buffer rank.
    pub fn rank(self: Buffer) u4 {
        return self._shape.rank();
    }

    /// Test helper: returns a new Buffer with the given tags.
    /// Allows to call `zml.testing.compileAndCall` when the tested
    /// functions requires tagged tensors.
    pub fn withTags(self: Buffer, tags_: anytype) Buffer {
        var res = self;
        res._shape = self._shape.withTags(tags_);
        return res;
    }

    pub fn format(
        self: Buffer,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Buffer({_})", .{self._shape});
    }

    fn _shapeFromPjrtBuffer(platform: Platform, buf: *pjrtx.Buffer) Shape {
        const dt: DataType = switch (buf.getElementType(platform.pjrt_api)) {
            // Please keep the list exhaustive and in the same order than in DataType.
            .PRED => .bool,
            .F8E4M3B11FNUZ => .f8e4m3b11fnuz,
            .F8E4M3FN => .f8e4m3fn,
            .F8E4M3FNUZ => .f8e4m3fnuz,
            .F8E5M2 => .f8e5m2,
            .F8E5M2FNUZ => .f8e5m2fnuz,
            .BF16 => .bf16,
            .F16 => .f16,
            .F32 => .f32,
            .F64 => .f64,
            .S4 => .i4,
            .S8 => .i8,
            .S16 => .i16,
            .S32 => .i32,
            .S64 => .i64,
            .U4 => .u4,
            .U8 => .u8,
            .U16 => .u16,
            .U32 => .u32,
            .U64 => .u64,
            .C64 => .c64,
            .C128 => .c128,
            .INVALID => @panic("Can't handle INVALID Pjrt buffers."),
        };

        return Shape.init(buf.getDimensions(platform.pjrt_api), dt);
    }
};
