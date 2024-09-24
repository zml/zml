const std = @import("std");
const testing = std.testing;

const meta = @import("meta.zig");
const pjrt = @import("pjrt");

const Context = @import("context.zig").Context;
const Data = @import("dtype.zig").Data;
const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Buffer);
}

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    _shape: Shape,
    _api: *const pjrt.Api,
    _shards: Shards,

    pub const MAX_NUM_SHARDS: u8 = 8;
    pub const Shards = std.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(platform: Platform, buf: HostBuffer) !Buffer {
        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._shape = buf.shape(),
            ._shards = .{},
        };

        for (platform.getDevices()) |dev| {
            const pjrt_buffer = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, .{
                .data = buf.data,
                .buffer_type = bufferTypeFromDtype(buf.shape().dtype()),
                .dims = buf.shape().dims(),
                .byte_strides = buf.strides(),
                .device = dev,
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            });

            res._shards.appendAssumeCapacity(pjrt_buffer);
        }
        return res;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: Platform, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        meta.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        meta.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});
        var shards: Shards = .{};
        shards.appendSliceAssumeCapacity(pjrt_buffers);
        return .{
            ._api = platform.pjrt_api,
            ._shape = Shape.init(
                // This isn't with sharded axes.
                pjrt_buffers[0].getDimensions(platform.pjrt_api),
                dtypeFromBufferType(pjrt_buffers[0].getElementType(platform.pjrt_api)),
            ),
            ._shards = shards,
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
            .element_type = bufferTypeFromDtype(buf.shape().dtype()),
            .dims = buf.shape().dims(),
            // TODO: split in shards
            .device = platform.getDevices()[0],
            .layout = .{
                .Tiled = .{
                    .minor_to_major = minor_to_major[Shape.MAX_RANK - buf.shape().rank() ..],
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
        });

        var shards: Shards = .{};
        shards.appendAssumeCapacity(pjrt_buffer);
        return .{
            ._api = platform.pjrt_api,
            ._shape = buf.shape(),
            ._shards = shards,
        };
    }

    /// Fetches the content of the given buffer into a stack variable of the given type.
    pub fn getValue(self: Buffer, T: type) !T {
        meta.assert(self._shape.byteSize() == @sizeOf(T), "Buffer {} has {d} bytes of data, can't load it to a {s} with {d} bytes", .{ self, self._shape.byteSize(), @typeName(T), @sizeOf(T) });
        var res: T = undefined;
        meta.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const event = try self._shards.get(0).toHostBuffer(self._api, std.mem.asBytes(&res));
        try event.await_(self._api);
        return res;
    }

    /// Copies the content of the Buffer back to host, in the given buffer,
    /// and return a new `HostBuffer` object with the same shape.
    /// The returned `HostBuffer` doesn't own the memory.
    pub fn toHost(self: Buffer, output: []u8) !HostBuffer {
        meta.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const event = try self._shards.get(0).toHostBuffer(self._api, output);
        try event.await_(self._api);
        return HostBuffer.fromBytes(self.shape(), output);
    }

    /// Copies the content of the Buffer to the host.
    /// The returned `HostBuffer` does own the memory.
    pub fn toHostAlloc(self: Buffer, allocator: std.mem.Allocator) !HostBuffer {
        const output = try HostBuffer.empty(allocator, self.shape());
        meta.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const event = try self._shards.get(0).toHostBuffer(self._api, @constCast(output.data));
        try event.await_(self._api);
        return output;
    }

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *const Buffer) void {
        for (self._shards.constSlice()) |buffer| {
            buffer.deinit(self._api);
        }
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

    fn hasShardedAxis(self: Buffer) bool {
        if (self._shards.len == 1) return false;
        return @reduce(.Or, self._shape._sharding_info);
    }
};

pub fn bufferTypeFromDtype(dt: DataType) pjrt.BufferType {
    return switch (dt) {
        .bool => .PRED,
        .f8e4m3b11fnuz => .F8E4M3B11FNUZ,
        .f8e4m3fn => .F8E4M3FN,
        .f8e4m3fnuz => .F8E4M3FNUZ,
        .f8e5m2 => .F8E5M2,
        .f8e5m2fnuz => .F8E5M2FNUZ,
        .bf16 => .BF16,
        .f16 => .F16,
        .f32 => .F32,
        .f64 => .F64,
        .i8 => .S8,
        .i4 => .S4,
        .i16 => .S16,
        .i32 => .S32,
        .i64 => .S64,
        .u4 => .U4,
        .u8 => .U8,
        .u16 => .U16,
        .u32 => .U32,
        .u64 => .U64,
        .c64 => .C64,
        .c128 => .C128,
    };
}

pub fn dtypeFromBufferType(pjrt_type: pjrt.BufferType) DataType {
    return switch (pjrt_type) {
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
        .S8 => .i8,
        .S4 => .i4,
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
        .INVALID => @panic("Found an invalid pjrt buffer"),
    };
}

test bufferTypeFromDtype {
    inline for (@typeInfo(DataType).Enum.fields) |field| {
        const dt: DataType = @enumFromInt(field.value);
        try std.testing.expectEqual(dt, dtypeFromBufferType(bufferTypeFromDtype(dt)));
    }

    inline for (@typeInfo(pjrt.BufferType).Enum.fields) |field| {
        const dt: pjrt.BufferType = @enumFromInt(field.value);
        if (dt == .INVALID) continue;
        try std.testing.expectEqual(dt, bufferTypeFromDtype(dtypeFromBufferType(dt)));
    }
}
