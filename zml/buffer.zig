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
    _platform: *const Platform,
    _shape: Shape,
    _sharding: Sharding,
    _shards: Shards,

    pub const MAX_NUM_SHARDS: u16 = Platform.MAX_NUM_DEVICES;
    pub const Shards = stdx.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const FromOptions = struct { wait: bool = true, memory: Memory.Kind = .default };

    /// Frees the accelerator memory.
    /// Depending on the platform, the memory is typically not released to the OS
    /// but just marked as available in the memory pool.
    pub fn deinit(self: *Buffer) void {
        // log.warn("Unloading {f} {d} bytes", .{ self._shape, self._shape.byteSize() });
        for (self._shards.constSlice()) |buffer| {
            buffer.deinit(self._platform.pjrt_api);
        }
    }

    /// This Buffer shape.
    pub fn shape(self: Buffer) Shape {
        return self._shape;
    }

    // todo: rework with sharding in mind
    pub fn format(self: Buffer, writer: *std.Io.Writer) !void {
        try writer.print("Buffer({f})@{x}", .{ self._shape, @intFromPtr(self.shardDevicePtr(0)) });
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
            ._platform = platform,
            ._shape = shape_,
            ._sharding = sharding,
            ._shards = .{},
        };

        const devices = platform.devices;
        const buffer_type = pjrtx.bufferTypeFromDtype(shape_.dtype());

        var it = try sharding_.transferIterator(sharding, shape_);
        defer it.deinit();

        const slice = Slice.init(shape_, data_);

        while (it.next()) |shard| {
            if (shard.device_id >= devices.len) return error.InvalidDeviceId;

            const view = shardSlice(slice, shard);

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
            const ev = buffer.readyEvent(self._platform.pjrt_api);
            defer ev.deinit(self._platform.pjrt_api);
            try ev.await(self._platform.pjrt_api, io);
        }
    }

    pub const UnitializedOptions = struct { memory: Memory = .device };

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
            ._sharding = sharding,
            ._shards = .{},
        };
        errdefer for (res._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        var args = pjrt.Client.CreateUninitializedBufferArgs{
            .dims = sh.dims(),
            .element_type = pjrtx.bufferTypeFromDtype(sh.dtype()),
            .layout = .{
                .tiled = .{
                    .minor_to_major = constants.minorToMajor(sh.rank()),
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = undefined,
        };

        const devices = platform.devices;

        const assignment = try sharding.deviceAssignment(sharding.allocator); // todo: sharding
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
            ._platform = platform,
            ._shape = shape_,
            ._sharding = sharding,
            ._shards = shards,
        };
    }

    pub fn shardDevicePtr(self: Buffer, shard_index: usize) *anyopaque {
        return self._shards.get(shard_index).opaqueDeviceMemoryDataPointer(self._api) catch unreachable;
    }

    /// Fetches the content of the given buffer into a stack variable of the given type.
    pub fn getValue(self: Buffer, T: type, io: std.Io) !T {
        stdx.debug.assert(self._shape.byteSize() == @sizeOf(T), "Buffer {f} has {d} bytes of data, can't load it to a {s} with {d} bytes", .{ self, self._shape.byteSize(), @typeName(T), @sizeOf(T) });
        var res: T = undefined;

        try self.toSlice(io, .init(self.shape(), std.mem.asBytes(&res)));

        return res;
    }

    /// Copies the content of the Buffer to the provided slice.
    /// No allocations; only supports contiguous shard views.
    pub fn toSlice(self: Buffer, io: std.Io, slice: Slice) !void {
        stdx.debug.assert(self._shape.eql(slice.shape), "Buffer shape {f} doesn't match destination slice {f}", .{ self._shape, slice.shape });

        var it = try sharding_.transferIterator(self._sharding, self._shape);
        defer it.deinit();

        var shard_index: usize = 0;
        while (it.next()) |shard| : (shard_index += 1) {
            if (shard_index >= self._shards.len) return error.ShardCountMismatch;

            const view = shardSlice(slice, shard);
            if (!view.isContiguous()) return error.NonContiguousShardRead;

            const size_bytes: usize = @intCast(view.shape.byteSize());
            const start: usize = view.offset_bytes;
            const end: usize = start + size_bytes;
            stdx.debug.assert(end <= slice.constData().len, "Shard view exceeds destination slice", .{});

            const dst = slice.data()[start..end];
            const maybe_event = try self._shards.get(shard_index).toHostBuffer(self._platform.pjrt_api, dst);

            if (maybe_event) |event| {
                try event.await(self._platform.pjrt_api, io);
            }
        }
    }

    /// Copies the content of the Buffer to the provided slice.
    /// The returned slice owns the memory.
    /// (This path supports non-contiguous shard views.)
    pub fn toSliceAlloc(self: Buffer, allocator: std.mem.Allocator, io: std.Io) !Slice {
        const slice = try Slice.alloc(allocator, self.shape());
        errdefer slice.free(allocator);

        var it = try sharding_.transferIterator(self._sharding, self._shape);
        defer it.deinit();

        var shard_index: usize = 0;
        while (it.next()) |shard| : (shard_index += 1) {
            if (shard_index >= self._shards.len) return error.ShardCountMismatch;

            const view = shardSlice(slice, shard);
            var shard_slice = try Slice.alloc(allocator, view.shape);
            defer shard_slice.free(allocator);

            const maybe_event = try self._shards.get(shard_index).toHostBuffer(self._platform.pjrt_api, shard_slice.data());
            if (maybe_event) |event| {
                try event.await(self._platform.pjrt_api, io);
            }

            view.copyFromContiguous(shard_slice.constData());
        }

        return slice;
    }

    /// Returns a Slice representing the sub-region of `total` defined by the shard assignment.
    fn shardSlice(total: Slice, shard: ShardAssignment.Shard) Slice {
        var res = total;
        for (shard.slices.items) |s| {
            res = res.subSlice(s.axis, s.start, s.size);
        }
        return res;
    }
};
