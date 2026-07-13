const std = @import("std");

const pjrt = @import("pjrt");
const platforms = @import("platforms");
const stdx = @import("stdx");

const constants = @import("constants.zig");
const DataType = @import("dtype.zig").DataType;
const mem = @import("mem.zig");
const Memory = @import("platform.zig").Memory;
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
}

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    _platform: *const Platform,
    _shape: Shape,
    _sharding: Sharding,
    _shards: Shards,

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
        const placement = self._sharding.placement(self._shape) catch {
            return try writer.print("sharding error {} vs {}", .{ self._sharding, self._shape });
        };
        try writer.print("{f}", .{placement});
    }

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(
        io: std.Io,
        platform: *const Platform,
        sh: Shape,
        sharding: Sharding,
        data_: []const u8,
        opts: FromOptions,
    ) !Buffer {
        var res: Buffer = .{
            ._platform = platform,
            ._shape = sh,
            ._sharding = sharding.resolve(platform),
            ._shards = .empty,
        };
        errdefer for (res._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});
        const slice = Slice.init(sh, data_);
        const buffer_type = pjrtx.bufferTypeFromDtype(sh.dtype());

        const placement = placementOrPanic(res._sharding, sh);
        const shard_dims: []const i64 = placement.shape.dims();
        const layout = platform.defaultMemoryLayout(shard_dims, sh.dtype());

        for (platform.physical_mesh.devices_in_canonical_order) |device| {
            const memory = platform.devices[device.id].memory(opts.memory).?;
            const args: pjrt.Client.BufferFromHostBufferArgs = .{
                // Change for each device
                .data = placement.shardPtr(device.coords, slice),
                .dst = .{ .memory = memory.pjrt_memory },
                // Constant across devices
                .layout = layout,
                .dims = shard_dims,
                .buffer_type = buffer_type,
                .byte_strides = slice.byte_strides.constSlice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            };

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);
            if (event) |ev| ev.deinit(platform.pjrt_api);

            res._shards.appendAssumeCapacity(pjrt_buffer);
        }

        if (opts.wait) {
            try res.await(io);
        }

        return res;
    }

    /// Copies the given Zig bytes to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytes(io: std.Io, platform: *const Platform, sh: Shape, sharding: Sharding, data: []const u8) !Buffer {
        return fromBytesOpts(io, platform, sh, sharding, data, .{});
    }

    pub fn fromBytesOpts(io: std.Io, platform: *const Platform, sh: Shape, sharding: Sharding, data: []const u8, opts: FromOptions) !Buffer {
        return from(io, platform, sh, sharding, data, opts);
    }

    /// Copies the given zml.Slice to the accelerator memory and
    /// return a Buffer.
    pub fn fromSlice(io: std.Io, platform: *const Platform, slice: Slice, sharding: Sharding) !Buffer {
        return fromSliceOpts(io, platform, slice, sharding, .{});
    }

    pub fn fromSliceOpts(io: std.Io, platform: *const Platform, slice: Slice, sharding: Sharding, opts: FromOptions) !Buffer {
        return from(io, platform, slice.shape, sharding, std.mem.sliceAsBytes(slice.constData()), opts);
    }

    /// Creates a Buffer with a single element.
    pub fn scalar(io: std.Io, platform: *const Platform, val: anytype, dtype_: DataType) !Buffer {
        const x = dtype_.constant(val);
        return fromBytes(io, platform, .scalar(dtype_), .replicated, x.asBytes());
    }

    pub fn await(self: Buffer, io: std.Io) !void {
        for (self._shards.constSlice()) |buffer| {
            const ev = buffer.readyEvent(self._platform.pjrt_api);
            defer ev.deinit(self._platform.pjrt_api);
            try ev.await(self._platform.pjrt_api, io);
        }
    }

    pub const UnitializedOptions = struct { memory: Memory.Kind = .default };

    pub fn uninitialized(
        _: std.Io,
        platform: *const Platform,
        sh: Shape,
        sharding: Sharding,
        opts: UnitializedOptions,
    ) !Buffer {
        var res: Buffer = .{
            ._platform = platform,
            ._shape = sh,
            ._sharding = sharding.resolve(platform),
            ._shards = .empty,
        };
        errdefer for (res._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});
        const element_type = pjrtx.bufferTypeFromDtype(sh.dtype());
        const placement = placementOrPanic(res._sharding, sh);
        const shard_dims: []const i64 = placement.shape.dims();
        const layout = platform.defaultMemoryLayout(shard_dims, sh.dtype());

        for (platform.physical_mesh.devices_in_canonical_order) |device| {
            const memory = platform.devices[device.id].memory(opts.memory).?;
            const args: pjrt.Client.CreateUninitializedBufferArgs = .{
                // Change for each device
                .dst = .{ .memory = memory.pjrt_memory },
                // Constant across devices
                .layout = layout,
                .dims = shard_dims,
                .element_type = element_type,
            };

            const shard_buffer = try platform.pjrt_client.createUninitializedBuffer(platform.pjrt_api, args);
            res._shards.appendAssumeCapacity(shard_buffer);
        }

        return res;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: *const Platform, sh: Shape, sharding: Sharding, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        stdx.debug.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        stdx.debug.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});

        return .{
            ._platform = platform,
            ._shape = sh,
            ._sharding = sharding,
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

        const placement = placementOrPanic(self._sharding, self._shape);
        for (self._sharding.devicesInCanonicalOrder(), 0..) |device, shard_index| {
            // TODO: handle replicated information, we shouldn't iterate over all the devices unless needed
            const sub_slice = placement.shardSlice(device.coords, slice);
            if (!sub_slice.isContiguous()) return error.NonContiguousShardRead;

            const size_bytes = placement.shape.byteSize();
            const destination = sub_slice.data()[0..size_bytes];
            const maybe_event = try self._shards.get(shard_index).toHostBuffer(self._platform.pjrt_api, destination);

            if (maybe_event) |event| {
                try event.await(self._platform.pjrt_api, io);
            }
        }
    }

    /// Copies the content of the Buffer to the provided slice.
    /// The returned slice owns the memory.
    pub fn toSliceAlloc(self: Buffer, allocator: std.mem.Allocator, io: std.Io) !Slice {
        const slice = try Slice.alloc(allocator, self.shape());
        errdefer slice.free(allocator);

        const placement = placementOrPanic(self._sharding, self._shape);

        var shard_slice = try Slice.alloc(allocator, placement.shape);
        defer shard_slice.free(allocator);

        for (self._sharding.devicesInCanonicalOrder(), 0..) |device, shard_index| {
            const sub_slice = placement.shardSlice(device.coords, slice);
            const maybe_event = try self._shards.get(shard_index).toHostBuffer(self._platform.pjrt_api, shard_slice.data());
            if (maybe_event) |event| {
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
        const placement = placementOrPanic(self._sharding, self._shape);
        return placement.shape.byteSize() * self._sharding.devicesInCanonicalOrder().len;
    }

    pub fn opaqueDevicePtr(self: Buffer, device_id: usize) *anyopaque {
        return self._shards.get(device_id).opaqueDeviceMemoryDataPointer(self._platform.pjrt_api) catch unreachable;
    }
};

test "device round-trip" {
    const zml = @import("zml.zig");
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

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
        .{ .b = .model },
    ), std.mem.asBytes(&x));
    // no free: x_h is stack allocated
    const model_sharding: zml.Sharding = platform.shardings.get("model").?;
    const x_d: zml.Buffer = try .fromSlice(io, platform, x_h, model_sharding);
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

fn placementOrPanic(sharding: Sharding, shape: Shape) Sharding.Placement {
    return sharding.placement(shape) catch |err| {
        @branchHint(.cold);
        switch (err) {
            error.MissingLogicalBinding => {
                log.err(
                    \\Failed to shard Buffer of shape {f}, with sharding:
                    \\{f}
                    \\
                    \\The Buffer is probably inheriting a partitionned shape from a Tensor,
                    \\So Buffer creation must pass a Sharding, that maps the logical sharding of the Tensor to the physical mesh.
                , .{ shape, sharding });
                @panic("Failed to compute placement");
            },
            error.IncompatibleSharding => {
                log.err(
                    \\Failed to shard Buffer of shape {f}, with sharding:
                    \\{f}
                    \\
                    \\The Buffer dimension isn't properly divisible by the number of devices along the sharded axis.
                , .{ shape, sharding });
                @panic("Failed to compute placement");
            },
        }
    };
}
