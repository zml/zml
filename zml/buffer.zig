const std = @import("std");

const pjrt = @import("pjrt");
const platforms = @import("platforms");
const stdx = @import("stdx");

const constants = @import("constants.zig");
const DataType = @import("dtype.zig").DataType;
const mem = @import("mem.zig");
const Memory = @import("platform.zig").Memory;
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Slice = @import("slice.zig").Slice;
const Target = @import("platform.zig").Target;
const testing = @import("testing.zig");

const log = std.log.scoped(.zml);

test {
    std.testing.refAllDecls(Buffer);
    std.testing.refAllDecls(Buffer.HostAccessible);
}

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    _platform: *const Platform,
    _shape: Shape,
    _shards: Shards,

    /// Whether executions may donate a buffer, writing their outputs into it.
    /// `HostAccessible.init` uses it to decide whether devices share allocations.
    pub const Donation = enum {
        /// Executions only read the buffer. Unlike XLA's undonatable buffers, the
        /// host may still rewrite a HostAccessible one between executions.
        undonatable,
        /// Executions may donate the buffer, replacing its handles with their outputs.
        donatable,
    };

    pub const HostAccessible = @import("buffer/HostAccessible.zig");

    pub const MAX_NUM_SHARDS: u16 = Platform.MAX_NUM_DEVICES;
    pub const Shards = stdx.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const Shard = struct {
        _platform: *const Platform,
        _pjrt_buffer: *pjrt.Buffer,

        pub fn devicePtr(self: *const Shard) *anyopaque {
            return self._pjrt_buffer.opaqueDeviceMemoryDataPointer(self._platform.pjrt_api) catch unreachable;
        }
    };

    pub const ShardIterator = struct {
        _platform: *const Platform,
        _shards: []const *pjrt.Buffer,
        _index: usize = 0,

        pub fn remaining(self: *ShardIterator) usize {
            return self._shards.len -| self._index;
        }

        pub fn next(self: *ShardIterator) ?Shard {
            defer self._index += 1;
            if (self._index >= self._shards.len) return null;

            return .{
                ._pjrt_buffer = self._shards[self._index],
                ._platform = self._platform,
            };
        }
    };

    pub const FromOptions = struct { wait: bool = true, memory: Memory.Kind = .default };

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *Buffer) void {
        for (self._shards.constSlice()) |buffer| {
            buffer.deinit(self._platform.pjrt_api);
        }
        self._shards = .empty;
    }

    /// Given a flat struct (static size, no slices) containing `zml.Buffer`, `deinit` each one of them.
    pub fn deinitAll(T: type, buffers: *mem.Bufferized(T)) void {
        meta.visitFlatStruct(struct {
            fn deinit(_: void, x: *Buffer) void {
                x.deinit();
            }
        }.deinit, {}, buffers);
    }

    /// Given an arbitrary struct `deinit` all `zml.Buffer` containing.
    /// If the struct contains slices of `zml.Buffer` the memory of the slices will NOT be freed,
    /// This only impacts device memory.
    pub fn freeDeviceMemoryButKeepHostMetadataMemory(T: type, buffers: *mem.Bufferized(T)) void {
        meta.visit(struct {
            fn deinit(_: void, x: *Buffer) void {
                x.deinit();
            }
        }.deinit, {}, buffers);
    }

    /// This Buffer shape.
    pub fn shape(self: Buffer) Shape {
        return self._shape;
    }

    pub fn numShards(self: Buffer) u32 {
        return @intCast(self._shards.len);
    }

    pub fn shards(self: *const Buffer) ShardIterator {
        return .{
            ._platform = self._platform,
            ._shards = self._shards.constSlice(),
        };
    }

    pub fn format(self: Buffer, writer: *std.Io.Writer) !void {
        const pl = Sharding.Placement.init(&self._shape) catch |err| {
            return try writer.print("Buffer({f}) with invalid sharding {} ({})", .{ self._shape, self._shape._sharding, err });
        };
        try writer.print("{f}", .{pl});
    }

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(
        io: std.Io,
        platform: *const Platform,
        shape_: Shape,
        data_: []const u8,
        opts: FromOptions,
    ) !Buffer {
        var pjrt_shards: Shards = .empty;
        var packed_shape = shape_.packedShape();
        if (packed_shape._sharding.mesh == null) packed_shape._sharding.mesh = platform.replicated_mesh;
        const byte_strides = packed_shape.computeByteStrides();
        const placement = Sharding.Placement.initWithByteStrides(&packed_shape, byte_strides.constSlice()) catch |err| {
            @branchHint(.cold);
            std.debug.panic("Buffer shape {f} is incompatible with its mesh: {t}", .{ packed_shape, err });
        };
        const ty = pjrtx.bufferTypeFromDtype(packed_shape.dtype());
        const layout = platform.defaultMemoryLayout(placement.local_dims.constSlice(), packed_shape.dtype());

        errdefer for (pjrt_shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});
        const byte_size: usize = if (packed_shape.rank() == 0)
            packed_shape.dtype().sizeOf()
        else
            @intCast(packed_shape.dim(0) * byte_strides.get(0));
        stdx.debug.assert(data_.len == byte_size, "Expected data length {}, got {}", .{ byte_size, data_.len });
        const base_ptr = data_.ptr;

        for (platform.physical_mesh.devices_in_canonical_order, 0..) |device, device_index| {
            const memory = platform.devices[device.id].memory(opts.memory).?;
            const args: pjrt.Client.BufferFromHostBufferArgs = .{
                // Change for each device
                .data = placement.shardPtr(device_index, base_ptr),
                .dst = .{ .memory = memory.pjrt_memory },
                // Constant across devices
                .layout = layout,
                .dims = placement.local_dims.constSlice(),
                .buffer_type = ty,
                .byte_strides = byte_strides.constSlice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            };

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);
            if (event) |ev| ev.deinit(platform.pjrt_api);

            pjrt_shards.appendAssumeCapacity(pjrt_buffer);
        }

        if (opts.wait) {
            try awaitShards(pjrt_shards.constSlice(), platform, io);
        }

        return .{ ._platform = platform, ._shape = shape_, ._shards = pjrt_shards };
    }

    /// Copies the given Zig bytes to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytes(io: std.Io, platform: *const Platform, sh: Shape, data: []const u8) !Buffer {
        return fromBytesOpts(io, platform, sh, data, .{});
    }

    pub fn fromBytesOpts(io: std.Io, platform: *const Platform, sh: Shape, data: []const u8, opts: FromOptions) !Buffer {
        return from(io, platform, sh, data, opts);
    }

    /// Copies the given zml.Slice to the accelerator memory and
    /// return a Buffer.
    pub fn fromSlice(io: std.Io, platform: *const Platform, slice: Slice) !Buffer {
        return fromSliceOpts(io, platform, slice, .{});
    }

    pub fn fromSliceOpts(io: std.Io, platform: *const Platform, slice: Slice, opts: FromOptions) !Buffer {
        return from(io, platform, slice.shape, std.mem.sliceAsBytes(slice.constData()), opts);
    }

    /// Creates a Buffer with a single element.
    pub fn scalar(io: std.Io, platform: *const Platform, val: anytype, dtype_: DataType) !Buffer {
        const x = dtype_.constant(val);
        return fromBytes(io, platform, .scalar(dtype_), x.asBytes());
    }

    pub fn await(self: Buffer, io: std.Io) !void {
        try awaitShards(self._shards.constSlice(), self._platform, io);
    }

    fn awaitShards(pjrt_shards: []const *pjrt.Buffer, platform: *const Platform, io: std.Io) !void {
        for (pjrt_shards) |shard| {
            const event = shard.readyEvent(platform.pjrt_api);
            defer event.deinit(platform.pjrt_api);
            try event.await(platform.pjrt_api, io);
        }
    }

    pub const UnitializedOptions = struct { memory: Memory.Kind = .default };

    pub fn uninitialized(
        _: std.Io,
        platform: *const Platform,
        shape_: Shape,
        opts: UnitializedOptions,
    ) !Buffer {
        std.log.debug("uninitialized {f}", .{shape_});
        var buffer: Buffer = .{
            ._platform = platform,
            ._shape = shape_,
            ._shards = .empty,
        };
        const metadata: PjrtBufferMetadata = .init(platform, shape_);
        errdefer for (buffer._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});

        for (platform.physical_mesh.devices_in_canonical_order) |device| {
            const memory = platform.devices[device.id].memory(opts.memory).?;
            const args: pjrt.Client.CreateUninitializedBufferArgs = .{
                // Change for each device
                .dst = .{ .memory = memory.pjrt_memory },
                // Constant across devices
                .layout = metadata.layout,
                .dims = metadata.placement.local_dims.constSlice(),
                .element_type = metadata.ty,
            };

            const shard_buffer = try platform.pjrt_client.createUninitializedBuffer(platform.pjrt_api, args);
            buffer._shards.appendAssumeCapacity(shard_buffer);
        }

        return buffer;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: *const Platform, sh: Shape, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        stdx.debug.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        stdx.debug.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});

        return .{
            ._platform = platform,
            ._shape = sh,
            ._shards = Shards.fromSlice(pjrt_buffers) catch unreachable,
        };
    }

    /// Fetches the content of the given buffer into a stack variable of the given type.
    pub fn getValue(self: Buffer, T: type, io: std.Io) !T {
        stdx.debug.assert(self._shape.byteSize() == @sizeOf(T), "Buffer {f} has {d} bytes of data, can't load it to a {s} with {d} bytes", .{ self, self._shape.byteSize(), @typeName(T), @sizeOf(T) });
        var res: T = undefined;

        try self.toSlice(io, .init(self.shape(), std.mem.asBytes(&res)));

        return res;
    }

    /// Copies the content of the Buffer to the provided slice.
    pub fn toSlice(self: Buffer, io: std.Io, slice: Slice) !void {
        stdx.debug.assert(self._shape.eql(slice.shape), "Buffer shape {f} doesn't match destination slice {f}", .{ self._shape, slice.shape });

        const placement = placementOrPanic(self._platform, self._shape);
        for (0..self._platform.physical_mesh.devices_in_canonical_order.len) |shard_index| {
            // TODO: handle replicated information, we shouldn't iterate over all the devices unless needed
            const sub_slice = placement.shardSlice(shard_index, slice);
            if (!sub_slice.isContiguous()) return error.NonContiguousShardRead;

            const size_bytes = placement.localShape(self._shape).byteSize();
            const destination = sub_slice.data()[0..size_bytes];
            const maybe_event = try self._shards.get(shard_index).toHostBuffer(self._platform.pjrt_api, destination);

            if (maybe_event) |event| {
                defer event.deinit(self._platform.pjrt_api);
                try event.await(self._platform.pjrt_api, io);
            }
        }
    }

    /// Copies the content of the Buffer to the provided slice.
    /// The returned slice owns the memory.
    pub fn toSliceAlloc(self: Buffer, allocator: std.mem.Allocator, io: std.Io) !Slice {
        const slice = try Slice.alloc(allocator, self.shape());
        errdefer slice.free(allocator);

        const placement = placementOrPanic(self._platform, self._shape);

        var shard_slice = try Slice.alloc(allocator, placement.localShape(self._shape));
        defer shard_slice.free(allocator);

        for (0..self._platform.physical_mesh.devices_in_canonical_order.len) |shard_index| {
            const sub_slice = placement.shardSlice(shard_index, slice);
            const maybe_event = try self._shards.get(shard_index).toHostBuffer(self._platform.pjrt_api, shard_slice.data());
            if (maybe_event) |event| {
                defer event.deinit(self._platform.pjrt_api);
                try event.await(self._platform.pjrt_api, io);
            }

            // TODO: why is this using Slice.copy while `toSlice` errors out ?
            // TODO: why is this writing in a copy rather than directly in place.
            sub_slice.copy(shard_slice.constData());
        }

        return slice;
    }

    /// The memory used by this Buffer across all devices
    /// ie: `num_devices * shard_byte_size`
    /// `shard_byte_size` can be up to `self.shape().byteSize()` when the buffer is fully replicated.
    pub fn byteSize(self: Buffer) usize {
        const placement = placementOrPanic(self._platform, self._shape);
        return placement.localShape(self._shape).byteSize() * self._shards.len;
    }

    pub fn opaqueDevicePtr(self: Buffer, device_id: usize) *anyopaque {
        return self._shards.get(device_id).opaqueDeviceMemoryDataPointer(self._platform.pjrt_api) catch unreachable;
    }

    /// Creates a view of the given buffer
    pub fn createView(self: Buffer) !Buffer {
        const platform = self._platform;

        var view_shards_buffer: [MAX_NUM_SHARDS]*pjrt.Buffer = undefined;
        var view_shard_list: std.ArrayList(*pjrt.Buffer) = .initBuffer(&view_shards_buffer);
        errdefer for (view_shard_list.items) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        for (self._shards.constSlice()) |shard| {
            const view_shard = try platform.pjrt_client.createViewOfDeviceBuffer(platform.pjrt_api, .{
                .device_buffer_ptr = try shard.opaqueDeviceMemoryDataPointer(platform.pjrt_api),
                .dims = shard.dimensions(platform.pjrt_api),
                .element_type = shard.elementType(platform.pjrt_api),
                .layout = try shard.memoryLayout(platform.pjrt_api),
                .device = try shard.device(platform.pjrt_api),
            });
            view_shard_list.appendAssumeCapacity(view_shard);
        }

        return fromPjrtBuffers(platform, self._shape, view_shard_list.items);
    }
};

test "device round-trip" {
    const zml = @import("zml.zig");
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const tp = zml.testing.tpMesh();

    const x: [8][8]u32 = .{
        .{ 0, 1, 2, 3, 4, 5, 6, 7 },
        .{ 8, 9, 10, 11, 12, 13, 14, 15 },
        .{ 16, 17, 18, 19, 20, 21, 22, 23 },
        .{ 24, 25, 26, 27, 28, 29, 30, 31 },
        .{ 32, 33, 34, 35, 36, 37, 38, 39 },
        .{ 40, 41, 42, 43, 44, 45, 46, 47 },
        .{ 48, 49, 50, 51, 52, 53, 54, 55 },
        .{ 56, 57, 58, 59, 60, 61, 62, 63 },
    };

    const x_h: zml.Slice = .init(.withPartitioning(
        .init(.{ .b = 8, .d = 8 }, .u32),
        tp,
        .{ .b = .model },
    ), std.mem.asBytes(&x));
    // no free: x_h is stack allocated
    const x_d: zml.Buffer = try .fromSlice(io, platform, x_h);
    try std.testing.expectEqual(platform.devices.len, x_d.numShards());

    {
        const x_h_reborn: zml.Slice = try x_d.toSliceAlloc(allocator, io);
        defer x_h_reborn.free(allocator);

        errdefer std.log.err(" - reference: {d}\n- actual: {d}", .{ x_h, x_h_reborn });
        try zml.testing.expectClose(io, x_h, x_h_reborn, .exact_match);
    }

    {
        var x_2: @TypeOf(x) = undefined;
        const x_h_reborn: zml.Slice = .init(x_h.shape, std.mem.asBytes(&x_2));
        // no free: x_h_reborn is stack allocated
        try x_d.toSlice(io, x_h_reborn);

        errdefer std.log.err(" - reference: {d}\n- actual: {d}", .{ x_h, x_h_reborn });
        try zml.testing.expectClose(io, x_h, x_h_reborn, .exact_match);
    }
}

pub const PjrtBufferMetadata = struct {
    shape: Shape,
    ty: pjrt.BufferType,
    placement: Sharding.Placement,
    layout: pjrt.MemoryLayout,

    /// Inline because the layout may point to stack memory, as with
    /// `Platform.defaultMemoryLayout`. Outside `Buffer` so that
    /// `Buffer.HostAccessible` can use it without it being part of the API:
    /// zml.zig only exports `Buffer` from this file.
    pub inline fn init(platform: *const Platform, shape_: Shape) PjrtBufferMetadata {

        // Shards use the PJRT shape, which packs sub-byte elements.
        const packed_shape = shape_.packedShape();
        const placement = placementOrPanic(platform, packed_shape);

        return .{
            .shape = packed_shape,
            .ty = pjrtx.bufferTypeFromDtype(packed_shape.dtype()),
            .placement = placement,
            .layout = platform.defaultMemoryLayout(placement.local_dims.constSlice(), packed_shape.dtype()),
        };
    }
};

fn placementOrPanic(platform: *const Platform, shape: Shape) Sharding.Placement {
    var sh = shape;
    if (sh._sharding.mesh == null) sh._sharding.mesh = platform.replicated_mesh;
    return Sharding.Placement.init(&sh) catch |err| {
        @branchHint(.cold);
        std.debug.panic("Buffer shape {f} is incompatible with its mesh: {t}", .{ shape, err });
    };
}
