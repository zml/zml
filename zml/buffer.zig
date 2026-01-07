const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const constants = @import("constants.zig");
const ConstSlice = @import("slice.zig").ConstSlice;
const DataType = @import("dtype.zig").DataType;
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Slice = @import("slice.zig").Slice;
const Target = @import("platform.zig").Target;
const testing = @import("testing.zig");

const log = std.log.scoped(.zml);

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    _shape: Shape,
    _api: *const pjrt.Api,
    _target: Target,
    _shards: Shards,

    pub const MAX_NUM_SHARDS: u8 = Platform.MAX_NUM_DEVICES;
    pub const Shards = stdx.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const Memory = pjrt.Memory.Kind;
    pub const FromOptions = struct { wait: bool = true, memory: Memory = .device };

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *const Buffer) void {
        // log.warn("Unloading {f} {d} bytes", .{ self._shape, self._shape.byteSize() });
        for (self._shards.constSlice()) |buffer| {
            buffer.deinit(self._api);
        }
    }

    /// This Buffer shape.
    pub fn shape(self: Buffer) Shape {
        return self._shape;
    }

    pub fn format(self: Buffer, writer: *std.Io.Writer) !void {
        try writer.print("Buffer({f})@{x}", .{ self._shape, @intFromPtr(self.devicePtr()) });
    }

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(io: std.Io, platform: Platform, shape_: Shape, data_: []const u8, opts: FromOptions) !Buffer {
        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._target = platform.target,
            ._shape = shape_,
            ._shards = .{},
        };

        // We shard only on the first axis so that the chunks are still contiguous.
        // TODO: support more advanced sharding specs
        stdx.debug.assert(platform.sharding().num_replicas == 1, "ZML doesn't support num_replicas > 1 for now, got: {}", .{platform.sharding()});
        //const sharding_ax: ?u3 = std.simd.firstTrue(shape_._sharding_info);
        const n_partitions = platform.sharding().num_partitions;
        //const chunk_size = if (sharding_ax) |ax| cs: {
        //    // This kind of sharding error should be detected earlier on.
        //    stdx.debug.assert(@rem(shape_.dim(ax), n_partitions) == 0, "Buffer.from({f}) expects the sharding axis {} to have a dimension divisble by the number of devices ({}).", .{ shape_, ax, n_partitions });
        //    break :cs @divExact(shape_.dim(ax), n_partitions);
        //} else 0;
        //_ = chunk_size;

        const buffer_type = pjrtx.bufferTypeFromDtype(shape_.dtype());
        const byte_strides = shape_.computeByteStrides();

        const devices = platform.getDevices();
        for (0..n_partitions) |i| {
            // If no sharding if found, the given buffer is replicated on all devices.
            //const buf = if (sharding_ax) |ax| buf: {
            //    @panic("sharding not implemented");
            //    //const start: i64 = @as(i64, @intCast(i)) * chunk_size;
            //    //break :buf host_buffer.slice1d(ax, .{ .start = start, .end = start + chunk_size });
            //} else host_buffer;

            const args = pjrt.Client.BufferFromHostBufferArgs{
                .data = data_.ptr,
                .buffer_type = buffer_type,
                .dims = shape_.dims(),
                .byte_strides = byte_strides.slice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
                // CPU has no distinctions between memories.
                .dst = if (platform.target == .cpu or opts.memory == .device)
                    .{ .device = devices[i] }
                else
                    .{ .memory = platform.memoryForDevice(opts.memory, devices[i]) },
            };

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);

            if (event) |ev| ev.deinit(platform.pjrt_api);
            res._shards.appendAssumeCapacity(pjrt_buffer);
        }

        if (opts.wait) {
            res = try res.await(io);
        }

        return res;
    }

    /// Copies the given Zig bytes to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytes(io: std.Io, platform: Platform, sh: Shape, data: []const u8) !Buffer {
        return from(io, platform, sh, data, .{});
    }

    /// Copies the given zml.Slice or zml.ConstSlice to the accelerator memory and
    /// return a Buffer.
    pub fn fromSlice(io: std.Io, platform: Platform, slice: anytype) !Buffer {
        return fromSliceOpts(io, platform, slice, .{});
    }

    pub fn fromSliceOpts(io: std.Io, platform: Platform, slice: anytype, opts: FromOptions) !Buffer {
        if (@TypeOf(slice) != Slice and @TypeOf(slice) != ConstSlice) @compileError("Buffer.fromSlice expects a Slice or ConstSlice, got " ++ @typeName(@TypeOf(slice)));
        const real_slice: ConstSlice = if (@TypeOf(slice) == Slice) slice.constSlice() else slice;
        return from(io, platform, real_slice.shape, std.mem.sliceAsBytes(real_slice.data), opts);
    }

    /// Creates a Buffer with a single element.
    pub fn scalar(io: std.Io, platform: Platform, val: anytype, dtype_: DataType) !Buffer {
        const x = dtype_.constant(val);
        return fromBytes(io, platform, Shape.init(.{}, dtype_), x.asBytes());
    }

    pub fn await(self: Buffer, io: std.Io) !Buffer {
        for (self._shards.constSlice()) |buffer| {
            try buffer.getReadyEvent(self._api).await(self._api, io);
        }

        return self;
    }

    pub const UnitializedOptions = struct { memory: Memory = .device };

    pub fn uninitialized(io: std.Io, platform: Platform, shape_: Shape, opts: UnitializedOptions) !Buffer {
        if (opts.memory != .device) {
            // XLA uninitialized doesn't respect memory see https://github.com/openxla/xla/pull/31292
            // TODO: use uninitialized when it works again.
            const slice: Slice = try .alloc(std.heap.smp_allocator, shape_);
            defer slice.free(std.heap.smp_allocator);
            return .fromSliceOpts(io, platform, slice, .{ .memory = opts.memory });
        }

        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._shape = shape_,
            ._shards = .{},
            ._target = platform.target,
        };
        errdefer for (res._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        stdx.debug.assert(platform.sharding().num_replicas == 1, "ZML doesn't support num_replicas > 1 for now, got: {}", .{platform.sharding()});
        //const sharding_ax: ?u3 = std.simd.firstTrue(shape_._sharding_info);
        const n_partitions = platform.sharding().num_partitions;

        //const shard_shape = if (sharding_ax) |ax| s: {
        //    // This kind of sharding error should be detected earlier on.
        //    stdx.debug.assert(@rem(shape_.dim(ax), n_partitions) == 0, "Buffer.uninitialized() expects the sharding axis {} to have a dimension divisble by the number of devices ({}).", .{ ax, n_partitions });
        //    const shard_shape = shape_.set(ax, @divExact(shape_.dim(ax), n_partitions));
        //    break :s shard_shape;
        //} else shape_;

        var args = pjrt.Client.CreateUninitializedBufferArgs{
            .dims = shape_.dims(),
            .element_type = pjrtx.bufferTypeFromDtype(shape_.dtype()),
            .layout = .{
                .tiled = .{
                    .minor_to_major = constants.minorToMajor(shape_.rank()),
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            // set per device, see below.
            .dst = undefined,
        };

        const devices = platform.getDevices();
        for (0..n_partitions) |i| {
            args.dst = if (platform.target == .cpu or opts.memory == .device)
                .{ .device = devices[i] }
            else
                .{ .memory = platform.memoryForDevice(opts.memory, devices[i]) };

            const shard = try platform.pjrt_client.createUninitializedBuffer(platform.pjrt_api, args);
            res._shards.appendAssumeCapacity(shard);
        }

        return res;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: Platform, shape_: Shape, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        stdx.debug.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        stdx.debug.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});
        var shards: Shards = .{};
        shards.appendSliceAssumeCapacity(pjrt_buffers);
        return .{
            ._api = platform.pjrt_api,
            ._target = platform.target,
            ._shape = shape_,
            ._shards = shards,
        };
    }

    pub fn fromIoBufferize(platform: Platform, shape_: Shape) Buffer {
        return .{
            ._api = platform.pjrt_api,
            ._target = platform.target,
            ._shape = shape_,
            ._shards = .{},
        };
    }

    pub fn devicePtr(self: Buffer) *anyopaque {
        //stdx.debug.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer", .{});
        return self._shards.get(0).getOpaqueDeviceMemoryDataPointer(self._api) catch unreachable;
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
        //stdx.debug.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const maybe_event = try self._shards.get(0).toHostBuffer(self._api, slice.data);
        if (maybe_event) |event| {
            try event.await(self._api, io);
        }
    }

    /// Copies the content of the Buffer to the provided slice.
    /// The returned slice owns the memory.
    pub fn toSliceAlloc(self: Buffer, allocator: std.mem.Allocator, io: std.Io) !Slice {
        const slice = try Slice.alloc(allocator, self.shape());
        errdefer slice.free(allocator);

        try self.toSlice(io, slice);

        return slice;
    }
};
