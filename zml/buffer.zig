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
    _local_shards: LocalShards,

    pub const MAX_NUM_SHARDS: u16 = Platform.MAX_NUM_DEVICES;
    pub const LocalShard = struct {
        global_device_id: u32,
        buffer: *pjrt.Buffer,
    };
    pub const LocalShards = stdx.BoundedArray(LocalShard, MAX_NUM_SHARDS);

    pub const Shard = struct {
        _buffer: *const Buffer,
        _index: usize,

        fn local(self: Shard) LocalShard {
            return self._buffer._local_shards.get(self._index);
        }

        pub fn globalDeviceId(self: Shard) u32 {
            return self.local().global_device_id;
        }

        pub fn shape(self: Shard) Shape {
            return placementOrPanic(
                self._buffer._sharding,
                self._buffer._shape,
            ).shape;
        }

        pub fn globalSlices(self: Shard) Sharding.Placement.Slices {
            const placement = placementOrPanic(
                self._buffer._sharding,
                self._buffer._shape,
            );
            return placement.slices(
                self._buffer.shardingDevice(self.globalDeviceId()).coords,
            );
        }

        pub fn devicePtr(self: Shard) *anyopaque {
            return self.local().buffer.opaqueDeviceMemoryDataPointer(
                self._buffer._platform.pjrt_api,
            ) catch unreachable;
        }

        pub fn toSlice(
            self: Shard,
            io: std.Io,
            destination: Slice,
        ) !void {
            stdx.debug.assert(
                self.shape().eql(destination.shape),
                "Shard shape {f} doesn't match destination slice {f}",
                .{ self.shape(), destination.shape },
            );
            if (!destination.isContiguous()) {
                return error.NonContiguousShardRead;
            }

            const maybe_event = try self.local().buffer.toHostBuffer(
                self._buffer._platform.pjrt_api,
                destination.data(),
            );
            if (maybe_event) |event| {
                defer event.deinit(self._buffer._platform.pjrt_api);
                try event.await(self._buffer._platform.pjrt_api, io);
            }
        }

        pub fn toSliceAlloc(
            self: Shard,
            allocator: std.mem.Allocator,
            io: std.Io,
        ) !Slice {
            const result = try Slice.alloc(allocator, self.shape());
            errdefer result.free(allocator);
            try self.toSlice(io, result);
            return result;
        }
    };

    pub const ShardIterator = struct {
        _buffer: *const Buffer,
        _index: usize = 0,

        pub fn remaining(self: *ShardIterator) usize {
            return self._buffer._local_shards.len -| self._index;
        }

        pub fn next(self: *ShardIterator) ?Shard {
            defer self._index += 1;
            if (self._index >= self._buffer._local_shards.len) return null;

            return .{
                ._buffer = self._buffer,
                ._index = self._index,
            };
        }
    };

    pub const FromOptions = struct { wait: bool = true, memory: Memory.Kind = .default };

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *Buffer) void {
        for (self._local_shards.constSlice()) |shard| {
            shard.buffer.deinit(self._platform.pjrt_api);
        }
    }

    pub fn deinitAll(T: type, buffers: *mem.Bufferized(T)) void {
        meta.visitFlatStruct(struct {
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
        return @intCast(self._local_shards.len);
    }

    pub fn numGlobalShards(self: Buffer) u32 {
        return @intCast(self._sharding.devicesInCanonicalOrder().len);
    }

    pub fn shards(self: *const Buffer) ShardIterator {
        return .{
            ._buffer = self,
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
        shape_: Shape,
        sharding: Sharding,
        data_: []const u8,
        opts: FromOptions,
    ) !Buffer {
        // Use the PJRT shape for everything
        const sh = shape_.packedShape();
        var res: Buffer = .{
            ._platform = platform,
            ._shape = shape_,
            ._sharding = sharding.resolve(platform),
            ._local_shards = .empty,
        };
        errdefer for (res._local_shards.slice()) |shard| {
            shard.buffer.deinit(platform.pjrt_api);
        };

        const slice = Slice.init(sh, data_);
        const buffer_type = pjrtx.bufferTypeFromDtype(sh.dtype());

        const placement = placementOrPanic(res._sharding, sh);
        const shard_dims: []const i64 = placement.shape.dims();
        const layout = platform.defaultMemoryLayout(shard_dims, sh.dtype());

        for (res._sharding.devicesInCanonicalOrder()) |device| {
            const local_device = platform.addressableDeviceById(device.id) orelse
                continue;
            const memory = local_device.memory(opts.memory);
            stdx.debug.assert(
                memory != null,
                "Device {d} doesn't have {} memory",
                .{ device.id, opts.memory },
            );
            const args: pjrt.Client.BufferFromHostBufferArgs = .{
                // Change for each device
                .data = placement.shardPtr(device.coords, slice),
                .dst = .{ .memory = memory.?.pjrt_memory },
                // Constant across devices
                .layout = layout,
                .dims = shard_dims,
                .buffer_type = buffer_type,
                .byte_strides = slice.byte_strides.constSlice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            };

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);
            if (event) |ev| ev.deinit(platform.pjrt_api);

            res._local_shards.appendAssumeCapacity(.{
                .global_device_id = device.id,
                .buffer = pjrt_buffer,
            });
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
        for (self._local_shards.constSlice()) |shard| {
            const ev = shard.buffer.readyEvent(self._platform.pjrt_api);
            defer ev.deinit(self._platform.pjrt_api);
            try ev.await(self._platform.pjrt_api, io);
        }
    }

    pub const UnitializedOptions = struct { memory: Memory.Kind = .default };

    pub fn uninitialized(
        _: std.Io,
        platform: *const Platform,
        shape_: Shape,
        sharding: Sharding,
        opts: UnitializedOptions,
    ) !Buffer {
        const sh = shape_.packedShape();
        var res: Buffer = .{
            ._platform = platform,
            ._shape = shape_,
            ._sharding = sharding.resolve(platform),
            ._local_shards = .empty,
        };
        errdefer for (res._local_shards.slice()) |shard| {
            shard.buffer.deinit(platform.pjrt_api);
        };

        const element_type = pjrtx.bufferTypeFromDtype(sh.dtype());
        const placement = placementOrPanic(res._sharding, sh);
        const shard_dims: []const i64 = placement.shape.dims();
        const layout = platform.defaultMemoryLayout(shard_dims, sh.dtype());

        for (res._sharding.devicesInCanonicalOrder()) |device| {
            const local_device = platform.addressableDeviceById(device.id) orelse
                continue;
            const memory = local_device.memory(opts.memory);
            stdx.debug.assert(
                memory != null,
                "Device {d} doesn't have {} memory",
                .{ device.id, opts.memory },
            );
            const args: pjrt.Client.CreateUninitializedBufferArgs = .{
                // Change for each device
                .dst = .{ .memory = memory.?.pjrt_memory },
                // Constant across devices
                .layout = layout,
                .dims = shard_dims,
                .element_type = element_type,
            };

            const shard_buffer = try platform.pjrt_client.createUninitializedBuffer(platform.pjrt_api, args);
            res._local_shards.appendAssumeCapacity(.{
                .global_device_id = device.id,
                .buffer = shard_buffer,
            });
        }

        return res;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(
        platform: *const Platform,
        sh: Shape,
        sharding_: Sharding,
        local_shards: []const LocalShard,
    ) Buffer {
        stdx.debug.assert(
            local_shards.len <= MAX_NUM_SHARDS,
            "ZML doesn't support having more than {} shards. Received {} shards for one buffer.",
            .{ MAX_NUM_SHARDS, local_shards.len },
        );
        stdx.debug.assert(
            local_shards.len > 0,
            "fromPjrtBuffers expects at least one buffer, got 0.",
            .{},
        );

        const sharding = sharding_.resolve(platform);
        var result: Buffer = .{
            ._platform = platform,
            ._shape = sh,
            ._sharding = sharding,
            ._local_shards = .empty,
        };

        for (local_shards, 0..) |shard, i| {
            stdx.debug.assert(
                platform.addressableDeviceById(shard.global_device_id) != null,
                "Device {d} is not addressable on process {d}",
                .{ shard.global_device_id, platform.processIndex() },
            );
            for (local_shards[0..i]) |previous| {
                stdx.debug.assert(
                    previous.global_device_id != shard.global_device_id,
                    "Duplicate local shard for global device {d}",
                    .{shard.global_device_id},
                );
            }
        }

        for (sharding.devicesInCanonicalOrder()) |device| {
            for (local_shards) |shard| {
                if (shard.global_device_id == device.id) {
                    result._local_shards.appendAssumeCapacity(shard);
                    break;
                }
            }
        }
        stdx.debug.assert(
            result._local_shards.len == local_shards.len,
            "One or more local shard IDs do not belong to the Buffer sharding",
            .{},
        );
        return result;
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
        const selected = try self.readableLocalShards(placement);
        for (selected.constSlice()) |shard_index| {
            const shard: Shard = .{ ._buffer = &self, ._index = shard_index };
            const device = self.shardingDevice(shard.globalDeviceId());
            const sub_slice = placement.shardSlice(device.coords, slice);
            if (!sub_slice.isContiguous()) return error.NonContiguousShardRead;
        }

        for (selected.constSlice()) |shard_index| {
            const shard: Shard = .{ ._buffer = &self, ._index = shard_index };
            const device = self.shardingDevice(shard.globalDeviceId());
            try shard.toSlice(
                io,
                placement.shardSlice(device.coords, slice),
            );
        }
    }

    /// Copies the content of the Buffer to the provided slice.
    /// The returned slice owns the memory.
    pub fn toSliceAlloc(self: Buffer, allocator: std.mem.Allocator, io: std.Io) !Slice {
        const placement = placementOrPanic(self._sharding, self._shape);
        const selected = try self.readableLocalShards(placement);

        const slice = try Slice.alloc(allocator, self.shape());
        errdefer slice.free(allocator);

        var shard_slice = try Slice.alloc(allocator, placement.shape);
        defer shard_slice.free(allocator);

        for (selected.constSlice()) |shard_index| {
            const shard: Shard = .{ ._buffer = &self, ._index = shard_index };
            const device = self.shardingDevice(shard.globalDeviceId());
            const sub_slice = placement.shardSlice(device.coords, slice);
            try shard.toSlice(io, shard_slice);
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

    pub fn localByteSize(self: Buffer) usize {
        const placement = placementOrPanic(self._sharding, self._shape);
        return placement.shape.byteSize() * self._local_shards.len;
    }

    pub fn opaqueDevicePtr(
        self: Buffer,
        global_device_id: u32,
    ) *anyopaque {
        for (self._local_shards.constSlice()) |shard| {
            if (shard.global_device_id == global_device_id) {
                return shard.buffer.opaqueDeviceMemoryDataPointer(
                    self._platform.pjrt_api,
                ) catch unreachable;
            }
        }
        stdx.debug.panic(
            "Buffer has no local shard for global device {d}",
            .{global_device_id},
        );
    }

    fn shardingDevice(self: Buffer, global_device_id: u32) Sharding.Device {
        for (self._sharding.devicesInCanonicalOrder()) |device| {
            if (device.id == global_device_id) return device;
        }
        stdx.debug.panic(
            "Global device {d} does not belong to the Buffer sharding",
            .{global_device_id},
        );
    }

    fn readableLocalShards(
        self: Buffer,
        placement: Sharding.Placement,
    ) error{GlobalReadRequiresGather}!stdx.BoundedArray(
        usize,
        MAX_NUM_SHARDS,
    ) {
        var regions: stdx.BoundedArray(
            Sharding.Placement.Slices,
            MAX_NUM_SHARDS,
        ) = .empty;
        var result: stdx.BoundedArray(usize, MAX_NUM_SHARDS) = .empty;

        for (self._sharding.devicesInCanonicalOrder()) |device| {
            const region = placement.slices(device.coords);
            for (regions.constSlice()) |previous| {
                if (slicesEqual(previous, region)) break;
            } else {
                regions.appendAssumeCapacity(region);
                const local_index = for (
                    self._local_shards.constSlice(),
                    0..,
                ) |shard, index| {
                    const local_device = self.shardingDevice(
                        shard.global_device_id,
                    );
                    if (slicesEqual(
                        region,
                        placement.slices(local_device.coords),
                    )) break index;
                } else return error.GlobalReadRequiresGather;
                result.appendAssumeCapacity(local_index);
            }
        }
        return result;
    }
};

fn slicesEqual(
    left: Sharding.Placement.Slices,
    right: Sharding.Placement.Slices,
) bool {
    if (left.len != right.len) return false;
    for (left.constSlice(), right.constSlice()) |a, b| {
        if (a.start != b.start or a.size != b.size) return false;
    }
    return true;
}

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
    var x_d: zml.Buffer = try .fromSlice(io, platform, x_h, model_sharding);
    defer x_d.deinit();
    try std.testing.expectEqual(platform.devices.len, x_d.numShards());
    try std.testing.expectEqual(
        model_sharding.devicesInCanonicalOrder().len,
        x_d.numGlobalShards(),
    );
    try std.testing.expectEqual(x_d.byteSize(), x_d.localByteSize());

    var shards = x_d.shards();
    for (model_sharding.devicesInCanonicalOrder()) |device| {
        if (platform.addressableDeviceById(device.id) == null) continue;
        const shard = shards.next().?;
        try std.testing.expectEqual(device.id, shard.globalDeviceId());
        try std.testing.expectEqual(
            @as(usize, x_h.shape.rank()),
            shard.globalSlices().len,
        );
    }
    try std.testing.expect(shards.next() == null);

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

    var scalar = try zml.Buffer.scalar(io, platform, 42, .i32);
    defer scalar.deinit();
    try std.testing.expectEqual(42, try scalar.getValue(i32, io));
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
                @panic("Buffer shape and sharding should be consistent");
            },
            error.IncompatibleSharding => {
                log.err(
                    \\Failed to shard Buffer of shape {f}, with sharding:
                    \\{f}
                    \\
                    \\The Buffer dimension isn't properly divisible by the number of devices along the sharded axis.
                , .{ shape, sharding });
                @panic("Buffer shape should be divisible by the number of devices along the sharded axis.");
            },
        }
    };
}
