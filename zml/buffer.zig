const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const constants = @import("constants.zig");
const DataType = @import("dtype.zig").DataType;
const Memory = @import("platform.zig").Memory;
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const sharding_ = @import("sharding.zig");
const Sharding = sharding_.Sharding;
const ShardAssignment = sharding_.ShardAssignment;
const slice_ = @import("slice.zig");
const Slice = slice_.Slice;
const SliceView = slice_.SliceView;
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
    platform: *const Platform,
    _shards: Shards,
    _sharding: Sharding,

    pub const MAX_NUM_SHARDS: u16 = Platform.MAX_NUM_DEVICES;
    pub const Shards = stdx.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const FromOptions = struct { wait: bool = true, memory: Memory.Kind = .default };

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *Buffer) void {
        // log.warn("Unloading {f} {d} bytes", .{ self._shape, self._shape.byteSize() });
        for (self._shards.constSlice()) |buffer| {
            buffer.deinit(self.platform.pjrt_api);
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
    pub fn from(
        io: std.Io,
        platform: *const Platform,
        shape_: Shape,
        sharding: Sharding,
        data_: []const u8,
        opts: FromOptions,
    ) !Buffer {
        var res: Buffer = .{
            .platform = platform,
            ._shape = shape_,
            ._shards = .{},
            ._sharding = sharding,
        };

        const devices = platform.devices;
        const buffer_type = pjrtx.bufferTypeFromDtype(shape_.dtype());

        var it = try sharding_.transferIterator(sharding, shape_);
        defer it.deinit();

        const slice = Slice.init(shape_, data_);

        while (it.next()) |shard| {
            if (shard.device_id >= devices.len) return error.InvalidDeviceId;

            const view = sliceViewFromShard(slice, shard);

            const args = pjrt.Client.BufferFromHostBufferArgs{
                .data = view.constData().ptr,
                .buffer_type = buffer_type,
                .dims = view.shape.dims(),
                .byte_strides = view.byte_strides.constSlice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
                .dst = .{ .memory = devices[shard.device_id].memory(opts.memory).pjrt_memory },
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
        return from(io, platform, sh, sharding, data, .{});
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
    pub fn scalar(io: std.Io, platform: *const Platform, val: anytype, dtype_: DataType, sharding: Sharding) !Buffer {
        const x = dtype_.constant(val);
        return fromBytes(io, platform, Shape.init(.{}, dtype_), sharding, x.asBytes());
    }

    pub fn await(self: Buffer, io: std.Io) !void {
        for (self._shards.constSlice()) |buffer| {
            const ev = buffer.readyEvent(self.platform.pjrt_api);
            defer ev.deinit(self.platform.pjrt_api);
            try ev.await(self.platform.pjrt_api, io);
        }
    }

    pub const UnitializedOptions = struct { memory: Memory = .device };

    pub fn uninitialized(
        _: std.Io,
        platform: *const Platform,
        shape_: Shape,
        sharding: Sharding,
        opts: UnitializedOptions,
    ) !Buffer {
        var res: Buffer = .{
            .platform = platform,
            ._shape = shape_,
            ._shards = .{},
            ._target = platform.target,
            ._sharding = sharding,
        };
        errdefer for (res._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

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
            .dst = undefined,
        };

        const devices = platform.devices;

        const assignment = try sharding.deviceAssignment(sharding.allocator); // todo
        defer sharding.allocator.free(assignment);

        for (assignment) |device_id| {
            args.dst = if (platform.target == .cpu or opts.memory == .device)
                .{ .device = devices[device_id] }
            else
                .{ .memory = platform.memoryForDevice(opts.memory, devices[device_id]) };

            const shard = try platform.pjrt_client.createUninitializedBuffer(platform.pjrt_api, args);
            res._shards.appendAssumeCapacity(shard);
        }

        return res;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: *const Platform, shape_: Shape, sharding: Sharding, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        stdx.debug.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        stdx.debug.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});
        var shards: Shards = .{};
        shards.appendSliceAssumeCapacity(pjrt_buffers);
        return .{
            .platform = platform,
            ._shape = shape_,
            ._shards = shards,
            ._sharding = sharding,
        };
    }

    pub fn devicePtr(self: Buffer) *anyopaque {
        return self._shards.get(0).opaqueDeviceMemoryDataPointer(self._api) catch unreachable;
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
        const maybe_event = try self._shards.get(0).toHostBuffer(self._api, slice.data());
        if (maybe_event) |event| {
            try event.await(self.platform.pjrt_api, io);
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

    fn sliceViewFromShard(
        src: Slice,
        shard: ShardAssignment.Shard,
    ) SliceView {
        const rank = src.shape.rank();
        var starts = [_]i64{0} ** constants.MAX_RANK;
        var sizes = [_]i64{0} ** constants.MAX_RANK;

        for (0..rank) |ax| {
            starts[ax] = 0;
            sizes[ax] = src.shape.dim(ax);
        }
        for (shard.slices.items) |s| {
            starts[s.axis] = s.start;
            sizes[s.axis] = s.size;
        }

        var out_shape = src.shape;
        for (0..rank) |ax| {
            out_shape = out_shape.set(ax, sizes[ax]);
        }

        const byte_strides = src.shape.computeByteStrides();
        var offset: i64 = 0;
        for (0..rank) |ax| {
            offset += starts[ax] * byte_strides.get(ax);
        }

        return .{
            .data = src.constData(),
            .offset_bytes = @intCast(offset),
            .shape = out_shape,
            .byte_strides = byte_strides,
        };
    }
};
