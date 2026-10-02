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
    std.testing.refAllDecls(HostStagedBuffer);
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
        sharding_: Sharding,
        data_: []const u8,
        opts: FromOptions,
    ) !Buffer {
        var buffer, const metadata = emptyShell(platform, shape_, sharding_);

        errdefer for (buffer._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});
        const slice = Slice.init(metadata.shape, data_);

        for (platform.physical_mesh.devices_in_canonical_order) |device| {
            const memory = platform.devices[device.id].memory(opts.memory).?;
            const args: pjrt.Client.BufferFromHostBufferArgs = .{
                // Change for each device
                .data = metadata.placement.shardPtr(device.coords, slice),
                .dst = .{ .memory = memory.pjrt_memory },
                // Constant across devices
                .layout = metadata.layout,
                .dims = metadata.placement.shape.dims(),
                .buffer_type = metadata.ty,
                .byte_strides = slice.byte_strides.constSlice(),
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            };

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);
            if (event) |ev| ev.deinit(platform.pjrt_api);

            buffer._shards.appendAssumeCapacity(pjrt_buffer);
        }

        if (opts.wait) {
            try buffer.await(io);
        }

        return buffer;
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
        shape_: Shape,
        sharding_: Sharding,
        opts: UnitializedOptions,
    ) !Buffer {
        std.log.debug("uninitialized {f}", .{shape_});
        var buffer, const metadata = emptyShell(platform, shape_, sharding_);
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
                .dims = metadata.placement.shape.dims(),
                .element_type = metadata.ty,
            };

            const shard_buffer = try platform.pjrt_client.createUninitializedBuffer(platform.pjrt_api, args);
            buffer._shards.appendAssumeCapacity(shard_buffer);
        }

        return buffer;
    }

    const PjrtBufferMetadata = struct {
        shape: Shape,
        ty: pjrt.BufferType,
        placement: Sharding.Placement,
        layout: pjrt.MemoryLayout,
    };

    /// Inline because the layout may point to stack memory, as with `Platform.defaultMemoryLayout`.
    inline fn emptyShell(
        platform: *const Platform,
        shape_: Shape,
        sharding: Sharding,
    ) struct { Buffer, PjrtBufferMetadata } {
        const buf: Buffer = .{
            ._platform = platform,
            ._shape = shape_,
            ._sharding = sharding.resolve(platform),
            ._shards = .empty,
        };

        // Shards use the PJRT shape, which packs sub-byte elements.
        const packed_shape = shape_.packedShape();
        const placement = placementOrPanic(buf._sharding, packed_shape);

        return .{
            buf,
            .{
                .shape = packed_shape,
                .ty = pjrtx.bufferTypeFromDtype(packed_shape.dtype()),
                .placement = placement,
                .layout = platform.defaultMemoryLayout(placement.shape.dims(), packed_shape.dtype()),
            },
        };
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

        const placement = placementOrPanic(self._sharding, self._shape);

        var shard_slice = try Slice.alloc(allocator, placement.shape);
        defer shard_slice.free(allocator);

        for (self._sharding.devicesInCanonicalOrder(), 0..) |device, shard_index| {
            const sub_slice = placement.shardSlice(device.coords, slice);
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
        const placement = placementOrPanic(self._sharding, self._shape);
        return placement.shape.byteSize() * self._sharding.devicesInCanonicalOrder().len;
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

        return fromPjrtBuffers(platform, self._shape, self._sharding, view_shard_list.items);
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

/// A `Buffer` in pinned host memory, which the host reads and writes in place.
/// Views address elements as PJRT stores them, with sub-byte elements packed in bytes.
/// Host access must happen between executable runs, after device use finishes.
/// View pointers are assumed stable during host access. Obtain new views after
/// execution replaces `buffer`; pointers are retrieved only when creating a view.
/// Execution must preserve shape, sharding and layout.
pub const HostStagedBuffer = struct {
    /// One PJRT buffer per device.
    buffer: Buffer,
    shape: Shape,
    shard_shape: Shape,
    /// Heap allocated, so views can borrow it while the buffer moves.
    prepared: *Prepared,
    shards: std.MultiArrayList(Shard),
    /// Heap allocated and borrowed by views, like `prepared`.
    hostPointers: *HostPointers,

    pub const Coordinates = [Shape.MAX_RANK]usize;

    /// Host pointers of the owning shards, read from their PJRT handles once:
    /// each read takes and releases a hold on the PJRT buffer.
    const HostPointers = struct {
        ptrs: stdx.BoundedArray([*]u8, Platform.MAX_NUM_DEVICES) = .empty,
        /// Execution replaces the handles of donated buffers with its outputs,
        /// which may not reuse the allocation, so `await` reads them again.
        handles: stdx.BoundedArray(*pjrt.Buffer, Platform.MAX_NUM_DEVICES) = .empty,

        fn matches(self: *const HostPointers, buffer: *const Buffer, owningBufferIndices: []const usize) bool {
            for (self.handles.constSlice(), owningBufferIndices) |handle, index| {
                if (handle != buffer._shards.get(index)) return false;
            }
            return true;
        }
    };

    /// Layout and shard placement prepared once and borrowed by views, which
    /// stay small enough to copy freely.
    pub const Prepared = struct {
        shape: Shape,
        shard_shape: Shape,
        preparedLayout: PreparedLayout,
        preparedShards: PreparedShards,
        /// Global coordinates of each shard's first element. Borrowed.
        origins: []const Coordinates,
        /// Every shard holds the whole array densely, so element `i` sits at
        /// `i * elementSize` in each of them.
        contiguous: bool,

        pub fn init(shape: Shape, shard_shape: Shape, layout: pjrt.MemoryLayout, origins: []const Coordinates) Prepared {
            const preparedLayout: PreparedLayout = .init(shape, shard_shape, layout);
            const preparedShards: PreparedShards = .init(origins, shard_shape.rank());
            return .{
                .shape = shape,
                .shard_shape = shard_shape,
                .preparedLayout = preparedLayout,
                .preparedShards = preparedShards,
                .origins = origins,
                .contiguous = preparedLayout.addressing == .dense and preparedShards.groups.len == 1 and shard_shape.count() == shape.count(),
            };
        }
    };

    const Shard = struct {
        origin: Coordinates,
        /// Index into buffer._shards, which execution may replace.
        owningBufferIndex: usize,
        /// Retains the allocation while other devices hold non-owning views.
        shared: bool = false,
    };

    /// An input that execution only reads. The first device holding a shard
    /// allocates it, and the other devices holding it read it through views.
    /// On plugins without such views, each device owns an allocation and
    /// writes reach every one of them. Shared inputs must not be donated.
    pub fn initDeviceReadOnly(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding) !HostStagedBuffer {
        return create(allocator, io, platform, shape, sharding, true);
    }

    /// A donated execution output. Each device owns an allocation: writing a
    /// shared one would expose the output of a device to the reads of another.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding) !HostStagedBuffer {
        return create(allocator, io, platform, shape, sharding, false);
    }

    fn create(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding, is_device_read_only: bool) !HostStagedBuffer {
        const can_share_allocations = switch (platform.target) {
            .cpu, .cuda, .rocm => is_device_read_only,
            else => false,
        };
        const api = platform.pjrt_api;
        var buffer, const metadata = Buffer.emptyShell(platform, shape, sharding);

        const devices = buffer._sharding.devicesInCanonicalOrder();
        var shards: std.MultiArrayList(Shard) = .empty;
        errdefer {
            deinitBuffer(&buffer, &shards);
            shards.deinit(allocator);
        }
        try shards.ensureTotalCapacity(allocator, devices.len);

        for (devices, 0..) |device, bufferIndex| {
            const memory = platform.devices[device.id].memory(.host_pinned).?.pjrt_memory;
            const origin = shardOrigin(&metadata.placement, device);
            const maybeExisting: ?usize = for (shards.items(.origin), 0..) |other, index| {
                if (can_share_allocations and std.mem.eql(usize, &other, &origin)) break index;
            } else null;

            if (maybeExisting) |index| {
                const existing = buffer._shards.get(shards.items(.owningBufferIndex)[index]);
                if (!shards.items(.shared)[index]) {
                    try existing.increaseExternalReferenceCount(api);
                    shards.items(.shared)[index] = true;
                }
                buffer._shards.appendAssumeCapacity(try platform.pjrt_client.createViewOfDeviceBuffer(api, .{
                    .device_buffer_ptr = try existing.opaqueDeviceMemoryDataPointer(api),
                    .dims = metadata.placement.shape.dims(),
                    .element_type = metadata.ty,
                    .layout = try existing.memoryLayout(api),
                    // The backing host allocation is accessible across devices
                    // (portable pinned memory on CUDA/ROCm). This memory space
                    // associates the view with the current device rather than
                    // the device that originally allocated the memory.
                    .memory = memory,
                }));
            } else {
                const allocation = try platform.pjrt_client.createUninitializedBuffer(api, .{
                    .dims = metadata.placement.shape.dims(),
                    .element_type = metadata.ty,
                    .layout = metadata.layout,
                    .dst = .{ .memory = memory },
                });
                buffer._shards.appendAssumeCapacity(allocation);
                shards.appendAssumeCapacity(.{ .origin = origin, .owningBufferIndex = bufferIndex });
            }
        }

        try buffer.await(io);

        const prepared = try allocator.create(Prepared);
        errdefer allocator.destroy(prepared);
        prepared.* = .init(metadata.shape, metadata.placement.shape, metadata.layout, shards.items(.origin));
        const hostPointers = try allocator.create(HostPointers);
        errdefer allocator.destroy(hostPointers);
        hostPointers.* = try queryHostPointers(&buffer, shards.items(.owningBufferIndex));

        return .{
            .buffer = buffer,
            .shape = metadata.shape,
            .shard_shape = metadata.placement.shape,
            .prepared = prepared,
            .shards = shards,
            .hostPointers = hostPointers,
        };
    }

    fn queryHostPointers(buffer: *const Buffer, owningBufferIndices: []const usize) !HostPointers {
        const api = buffer._platform.pjrt_api;
        var result: HostPointers = .{};
        for (owningBufferIndices) |index| {
            const handle = buffer._shards.get(index);
            result.ptrs.appendAssumeCapacity(@ptrCast(try handle.opaqueDeviceMemoryDataPointer(api)));
            result.handles.appendAssumeCapacity(handle);
        }
        return result;
    }

    /// Execution must be done with the buffer.
    pub fn deinit(self: *HostStagedBuffer, allocator: std.mem.Allocator) void {
        deinitBuffer(&self.buffer, &self.shards);
        self.shards.deinit(allocator);
        allocator.destroy(self.prepared);
        allocator.destroy(self.hostPointers);
    }

    fn deinitBuffer(buffer: *Buffer, shards: *const std.MultiArrayList(Shard)) void {
        const api = buffer._platform.pjrt_api;
        // Device views must be destroyed before the allocations they borrow.
        for (buffer._shards.constSlice(), 0..) |shard, index| {
            if (std.mem.indexOfScalar(usize, shards.items(.owningBufferIndex), index) == null) shard.deinit(api);
        }
        for (shards.items(.owningBufferIndex), shards.items(.shared)) |index, shared| {
            const shard = buffer._shards.get(index);
            if (shared) shard.decreaseExternalReferenceCount(api) catch unreachable;
            shard.deinit(api);
        }
        buffer._shards = .empty;
    }

    /// Wait for execution to finish, and read the host pointers of handles that
    /// execution replaced. Obtain new views before accessing the output.
    pub fn await(self: *HostStagedBuffer, io: std.Io) !void {
        try self.buffer.await(io);
        const owningBufferIndices = self.shards.items(.owningBufferIndex);
        if (!self.hostPointers.matches(&self.buffer, owningBufferIndices)) {
            self.hostPointers.* = try queryHostPointers(&self.buffer, owningBufferIndices);
        }
    }

    /// Borrows the buffer's metadata and host pointers. Execution must have
    /// completed, and `await` must follow any execution that replaced the handles.
    /// The view and its blocks must not be used after the next execution, nor
    /// after the buffer is deinitialized.
    pub fn view(self: *const HostStagedBuffer, comptime T: type) View(T) {
        stdx.debug.assert(self.hostPointers.matches(&self.buffer, self.shards.items(.owningBufferIndex)), "await the HostStagedBuffer after execution replaces its handles", .{});
        return .init(self.prepared, self.hostPointers.ptrs.constSlice());
    }

    pub fn format(self: HostStagedBuffer, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("HostStagedBuffer({f}) {{", .{self.shape});
        switch (self.shard_shape.dtype()) {
            inline else => |dt| {
                if (comptime dt.bitSizeOf() < 8) unreachable;
                const values = self.view(dt.toZigType());
                var blocks = values.blocks();
                while (blocks.next()) |block| {
                    for (block.items[0]) |value| try writer.print(" {any}", .{value});
                }
            },
        }
        try writer.writeAll(" }");
    }

    /// A borrowed logical range in row-major order, independent of PJRT handles.
    /// Reads use the first shard containing an element; writes update all replicas.
    /// Pointers and origins must remain valid throughout host access.
    /// Origins must remain unchanged, since replica groups are prepared at creation.
    pub fn View(comptime T: type) type {
        return struct {
            prepared: *const Prepared,
            /// One host pointer per origin of `prepared`. Borrowed.
            ptrs: []const [*]u8,
            start: usize = 0,
            len: usize,

            const Self = @This();

            /// A view without elements, which never accesses storage or metadata.
            pub const empty: Self = .{
                .prepared = undefined,
                .ptrs = &.{},
                .len = 0,
            };

            /// A view of raw storage without PJRT buffers, e.g. for tests.
            /// Borrows `prepared` and the pointers, one per origin of `prepared`.
            pub fn init(prepared: *const Prepared, ptrs: []const [*]u8) Self {
                std.debug.assert(@sizeOf(T) == prepared.shard_shape.dtype().sizeOf());
                std.debug.assert(ptrs.len == prepared.origins.len);
                return .{
                    .prepared = prepared,
                    .ptrs = ptrs,
                    .len = prepared.shape.count(),
                };
            }

            /// Select `len` elements starting at `start`, or the remainder for null.
            pub fn slice(self: Self, start: usize, len_: ?usize) Self {
                std.debug.assert(start <= self.len);
                const len = len_ orelse self.len - start;
                std.debug.assert(len <= self.len - start);
                var result = self;
                result.start += start;
                result.len = len;
                return result;
            }

            // Single elements are addressed directly, without slicing and iterating.
            pub fn get(self: Self, index: usize) T {
                const element = self.locateElement(index);
                return self.itemsAtByteOffset(self.prepared.preparedShards.shardIndices.get(element.group.start), element.byteOffset, 1)[0];
            }

            pub fn set(self: Self, index: usize, value: T) void {
                const element = self.locateElement(index);
                for (self.prepared.preparedShards.shardIndices.constSlice()[element.group.start..][0..element.group.len]) |shard| {
                    self.itemsAtByteOffset(shard, element.byteOffset, 1)[0] = value;
                }
            }

            fn locateElement(self: *const Self, index: usize) struct { group: PreparedShards.Group, byteOffset: usize } {
                std.debug.assert(index < self.len);
                if (self.prepared.contiguous) {
                    return .{ .group = self.prepared.preparedShards.groups.get(0), .byteOffset = (self.start + index) * @sizeOf(T) };
                }
                const location = self.locate(unflattenIndex(self.prepared.shape, self.start + index));
                return .{ .group = location.group, .byteOffset = self.prepared.preparedLayout.byteOffset(location.offset, location.index) };
            }

            pub fn fill(self: Self, value: T) void {
                var iterator = self.blocks();
                while (iterator.next()) |block| {
                    for (block.items) |items| @memset(items, value);
                }
            }

            /// Fill with `first + i * step`, where `i` is relative to this view.
            pub fn fillIota(self: Self, first: T, step: T) void {
                var iterator = self.blocks();
                while (iterator.next()) |block| {
                    const items = block.items[0];
                    for (items, block.offset..) |*value, i| value.* = first + @as(T, @intCast(i)) * step;
                    for (block.items[1..]) |replica| @memcpy(replica, items);
                }
            }

            pub fn copyFrom(self: Self, values: []const T) void {
                std.debug.assert(values.len == self.len);
                var iterator = self.blocks();
                while (iterator.next()) |block| {
                    for (block.items) |items| @memcpy(items, values[block.offset..][0..items.len]);
                }
            }

            pub fn copyTo(self: Self, values: []T) void {
                std.debug.assert(values.len == self.len);
                var iterator = self.blocks();
                while (iterator.next()) |block| {
                    const items = block.items[0];
                    @memcpy(values[block.offset..][0..items.len], items);
                }
            }

            /// Source and destination must not overlap.
            pub fn copyFromView(self: Self, src: Self) void {
                std.debug.assert(src.len == self.len);
                var iterator = self.blocks();
                while (iterator.next()) |block| {
                    const items = block.items[0];
                    src.slice(block.offset, items.len).copyTo(items);
                    for (block.items[1..]) |replica| @memcpy(replica, items);
                }
            }

            /// Return the consecutive run starting at `index` in the first replica
            /// holding it, stopping at a physical boundary or this view's end.
            pub fn readBlock(self: Self, index: usize) []T {
                std.debug.assert(index < self.len);
                var iterator = self.slice(index, null).blocks();
                return iterator.next().?.items[0];
            }

            /// Visit runs of consecutive logical elements in global row-major
            /// order, without padding, grouping replicas of each logical range.
            /// Shards must cover the view with disjoint regions or exact replicas.
            pub fn blocks(self: Self) BlockIterator {
                return .init(self);
            }

            /// Visit the same blocks in reverse order. Elements within each
            /// replica slice remain in ascending logical order.
            pub fn reverseBlocks(self: Self) ReverseBlockIterator {
                return .init(self);
            }

            pub const Block = struct {
                /// Logical position of the first element, relative to this view.
                offset: usize,
                /// Equal-length replicas in the supplied shard order. The iterator
                /// owns these descriptors until its next call; their elements
                /// borrow view storage.
                items: []const []T,

                pub fn len(self: *const Block) usize {
                    return self.items[0].len;
                }
            };

            pub const BlockIterator = OrderedBlockIterator(false);
            pub const ReverseBlockIterator = OrderedBlockIterator(true);

            fn OrderedBlockIterator(comptime reverse: bool) type {
                return struct {
                    view: Self,
                    /// Next logical offset, or one past it for reverse traversal.
                    offset: usize,
                    /// Coordinates of the next block's first (or reverse last) element.
                    global: Coordinates,
                    /// Replicas of the last block, kept here so blocks stay small.
                    replicas: stdx.BoundedArray([]T, Platform.MAX_NUM_DEVICES),

                    const Iterator = @This();

                    fn init(v: Self) Iterator {
                        return .{
                            .view = v,
                            .offset = if (reverse) v.len else 0,
                            .global = if (v.len == 0 or v.prepared.contiguous) @splat(0) else unflattenIndex(v.prepared.shape, v.start + (if (reverse) v.len - 1 else 0)),
                            .replicas = .{ .buffer = undefined, .len = 0 },
                        };
                    }

                    pub fn next(self: *Iterator) ?Block {
                        const v = &self.view;
                        if (if (reverse) self.offset == 0 else self.offset == v.len) return null;
                        self.replicas.clear();
                        if (v.prepared.contiguous) {
                            // The whole view is one run in every replica.
                            const group = v.prepared.preparedShards.groups.get(0);
                            for (v.prepared.preparedShards.shardIndices.constSlice()[group.start..][0..group.len]) |shard| {
                                self.replicas.appendAssumeCapacity(v.itemsAtByteOffset(shard, v.start * @sizeOf(T), v.len));
                            }
                            self.offset = if (reverse) 0 else v.len;
                            return .{ .offset = 0, .items = self.replicas.constSlice() };
                        }
                        const location = v.locate(self.global);
                        const bounds = v.blockBounds(location.offset);
                        const len = if (reverse)
                            @min(location.offset + 1 - bounds.start, self.offset)
                        else
                            @min(bounds.end - location.offset, v.len - self.offset);
                        // A block is physically contiguous, so its last element's
                        // address also gives its start without decoding coordinates again.
                        const byteOffset = v.prepared.preparedLayout.byteOffset(location.offset, location.index) -
                            (if (reverse) (len - 1) * v.prepared.preparedLayout.elementSize else 0);
                        for (v.prepared.preparedShards.shardIndices.constSlice()[location.group.start..][0..location.group.len]) |shard| {
                            self.replicas.appendAssumeCapacity(v.itemsAtByteOffset(shard, byteOffset, len));
                        }
                        const offset = if (reverse) self.offset - len else self.offset;
                        self.offset = if (reverse) offset else offset + len;
                        if (if (reverse) self.offset != 0 else self.offset != v.len) {
                            moveIndex(reverse, v.prepared.shape, &self.global, len);
                        }
                        return .{ .offset = offset, .items = self.replicas.constSlice() };
                    }
                };
            }

            const Bounds = struct { start: usize, end: usize };

            /// The block containing a shard-local element: elements contiguous in both
            /// physical memory and global row-major order. Bounds are shard-local
            /// row-major indices, not physical memory offsets.
            ///
            /// Example: an unsharded [2,5] array with row-major 2x2 tiles.
            ///
            /// 1. Logical indices, with tile boundaries. `_` denotes padding:
            /// ```text
            ///       tile 0    tile 1    tile 2
            ///       +-----+   +-----+   +-----+
            ///       | 0 1 |   | 2 3 |   | 4 _ |
            ///       | 5 6 |   | 7 8 |   | 9 _ |
            ///       +-----+   +-----+   +-----+
            /// ```
            /// 2. Tiles are stored consecutively, each in row-major order:
            /// ```text
            ///       [0 1 5 6] [2 3 7 8] [4 _ 9 _]
            /// ```
            /// Thus 0 and 1 are adjacent in memory, but 1 and 2 are not.
            /// `rowBlockSize = 2` is the maximum contiguous logical run.
            ///
            /// 3. Align runs within each logical row, clipping before padding:
            /// ```text
            ///       lastDimSize = 5
            ///       alignmentSize = max(5, 2) = 5
            ///
            ///       alignmentStart     blocks of logical elements      row end
            ///              0           [0 1] [2 3] [4]                    5
            ///              5           [5 6] [7 8] [9]                   10
            ///       actual block sizes:   2     2    1
            ///
            ///       local   alignmentStart   start   end   returned bounds
            ///         4            0           4      5       [4,5)
            ///         5            5           5      7       [5,7)
            ///         6            5           5      7       [5,7)
            ///         9            5           9     10       [9,10)
            /// ```
            /// For local=6, start = 6 - (6-5)%2 = 5, end = min(5+2, 5+5) = 7.
            /// For local=4, end = min(4+2, 0+5) = 5: the final run is shortened.
            ///
            /// Without tiles, the same dense row-major array has rowBlockSize=10
            /// and alignmentSize=10: one block [0,10) spanning both rows.
            fn blockBounds(self: *const Self, local: usize) Bounds {
                const blockSize = self.prepared.preparedLayout.blockSize;
                // Runs shorter than a row align within that row. Larger runs
                // span a whole number of rows, so alignment uses the run itself.
                const alignment_size = self.prepared.preparedLayout.alignmentSize;
                const alignment_start = local - local % alignment_size;
                // Round the position relative to alignmentStart down to a run boundary.
                const start = local - (local - alignment_start) % blockSize;
                // A row's final run may be shorter; stop at the alignment interval's end.
                return .{ .start = start, .end = @min(start + blockSize, alignment_start + alignment_size) };
            }

            fn itemsAtByteOffset(self: *const Self, shard: usize, byteOffset: usize, len: usize) []T {
                const ptr: [*]T = @ptrCast(@alignCast(self.ptrs[shard] + byteOffset));
                return ptr[0..len];
            }

            const Location = struct { group: PreparedShards.Group, index: Coordinates, offset: usize };

            /// Locate one logical region, retaining coordinates for physical addressing.
            fn locate(self: *const Self, index: Coordinates) Location {
                for (self.prepared.preparedShards.groups.constSlice()) |group| {
                    const origin = self.prepared.origins[group.first];
                    var local: Coordinates = @splat(0);
                    var offset: usize = 0;
                    for (self.prepared.shard_shape.dims(), 0..) |dim_, axis| {
                        const dim: usize = @intCast(dim_);
                        if (index[axis] < origin[axis] or index[axis] - origin[axis] >= dim) break;
                        local[axis] = index[axis] - origin[axis];
                        offset = offset * dim + local[axis];
                    } else {
                        return .{ .group = group, .index = local, .offset = offset };
                    }
                }
                unreachable;
            }
        };
    }
};

fn shardOrigin(placement: *const Sharding.Placement, device: Sharding.Device) HostStagedBuffer.Coordinates {
    var origin: HostStagedBuffer.Coordinates = @splat(0);
    for (placement.slices(device.coords).constSlice(), 0..) |slice, axis| origin[axis] = @intCast(slice.start);
    return origin;
}

fn unflattenIndex(shape: Shape, offset: usize) [Shape.MAX_RANK]usize {
    var index: [Shape.MAX_RANK]usize = @splat(0);
    var remaining = offset;
    var axis = shape.rank();
    while (axis > 0) {
        axis -= 1;
        const dim: usize = @intCast(shape.dim(axis));
        index[axis] = remaining % dim;
        remaining /= dim;
    }
    return index;
}

/// Advance row-major coordinates, decoding only axes that carry or borrow.
/// The resulting coordinates must still be within the shape.
fn moveIndex(comptime reverse: bool, shape: Shape, index: *HostStagedBuffer.Coordinates, count: usize) void {
    var carry = count;
    var axis = shape.rank();
    while (carry != 0 and axis > 0) {
        axis -= 1;
        const dim: usize = @intCast(shape.dim(axis));
        if (reverse) {
            if (carry <= index[axis]) {
                index[axis] -= carry;
                return;
            }
            carry -= index[axis] + 1;
            index[axis] = dim - 1 - carry % dim;
        } else {
            const available = dim - index[axis];
            if (carry < available) {
                index[axis] += carry;
                return;
            }
            carry -= available;
            index[axis] = carry % dim;
        }
        carry = carry / dim + 1;
    }
    std.debug.assert(carry == 0);
}

/// Group equal logical regions once, preserving the supplied order of replicas.
/// Indices address a view's pointer and origin arrays.
const PreparedShards = struct {
    const Group = struct { first: u8, start: u8 = 0, len: u8 = 0 };

    groups: stdx.BoundedArray(Group, Platform.MAX_NUM_DEVICES) = .empty,
    shardIndices: stdx.BoundedArray(u8, Platform.MAX_NUM_DEVICES) = .empty,

    const empty: PreparedShards = .{};

    fn init(origins: []const HostStagedBuffer.Coordinates, rank: usize) PreparedShards {
        std.debug.assert(origins.len <= Platform.MAX_NUM_DEVICES);
        var result: PreparedShards = .empty;
        for (origins, 0..) |origin, shard| {
            for (result.groups.constSlice()) |group| {
                if (std.mem.eql(usize, origin[0..rank], origins[group.first][0..rank])) break;
            } else {
                result.groups.appendAssumeCapacity(.{ .first = @intCast(shard) });
            }
        }
        for (result.groups.slice()) |*group| {
            group.start = @intCast(result.shardIndices.len);
            for (origins, 0..) |origin, shard| {
                if (std.mem.eql(usize, origin[0..rank], origins[group.first][0..rank])) {
                    result.shardIndices.appendAssumeCapacity(@intCast(shard));
                }
            }
            group.len = @intCast(result.shardIndices.len - group.start);
        }
        return result;
    }
};

/// Limit physical runs to the shard's consecutive global indices. This boundary
/// is a whole number of rows; physical tiles may impose a smaller run within a row.
fn rowBlockSize(shape: Shape, shard_shape: Shape, layout: pjrt.MemoryLayout) usize {
    var result: usize = 1;
    var axis = shape.rank();
    while (axis > 0) {
        axis -= 1;
        result *= @intCast(shard_shape.dim(axis));
        if (shape.dim(axis) != shard_shape.dim(axis)) break;
    }
    return @min(result, physicalContiguousBlockSize(shard_shape, layout));
}

/// A dense layout is one block. Otherwise, find the contiguous run along the
/// logical innermost axis, stopping when another physical axis introduces gaps.
fn physicalContiguousBlockSize(shard_shape: Shape, layout: pjrt.MemoryLayout) usize {
    if (shard_shape.count() == 0) return 0;
    if (shard_shape.rank() == 0) return 1;
    const last_axis = shard_shape.rank() - 1;
    const cols: usize = @intCast(shard_shape.dim(last_axis));
    switch (layout) {
        .strides => |strides| {
            const dense = shard_shape.computeByteStrides();
            if (std.mem.eql(i64, strides.byte_strides, dense.constSlice())) return shard_shape.count();
            return if (strides.byte_strides[last_axis] == shard_shape.dtype().sizeOf()) cols else 1;
        },
        .tiled => |tiled| {
            if (tiled.tile_dims.len == 0 and isMinorToMajorRowMajor(tiled.minor_to_major, shard_shape.rank())) return shard_shape.count();
            // From the most minor physical dimension, the run grows while each
            // dimension is the next digit of the innermost axis.
            const expanded: TiledDims = .init(shard_shape, tiled);
            var run: usize = 1;
            var i = expanded.dims.len;
            while (i > 0) {
                i -= 1;
                const dim = expanded.dims.get(i);
                if (dim.size == 1) continue;
                const axis = dim.axis orelse break;
                if (axis != last_axis or dim.divisor != run) break;
                if (dim.modulus == 0) return cols;
                run *= dim.modulus;
                // Padding after the digit's last value ends the run.
                if (dim.size != dim.modulus) break;
            }
            return @min(cols, run);
        },
    }
}

/// Physical addressing and logical block boundaries prepared once per pinned buffer.
/// All metadata is stored by value; the source layout is needed only during init.
const PreparedLayout = struct {
    const ByteStrides = [Shape.MAX_RANK]usize;

    /// One physical dimension of a tiled layout, as a digit of one coordinate.
    const Digit = struct {
        byteStride: usize,
        /// A shift when divShift is present; otherwise a general divisor.
        divisor: usize,
        /// A mask when isMask is set; otherwise a general modulus.
        modulus: usize,
        axis: u3,
        divShift: ?std.math.Log2Int(usize),
        isMask: bool,

        fn init(axis: u3, dim: TiledDims.Dim, byteStride: usize) Digit {
            // Unbounded digits keep every bit of the quotient.
            const modulus = if (dim.modulus == 0) std.math.maxInt(usize) else dim.modulus;
            const isMask = dim.modulus == 0 or std.math.isPowerOfTwo(modulus);
            return .{
                .byteStride = byteStride,
                .divisor = dim.divisor,
                .modulus = if (isMask and dim.modulus != 0) modulus - 1 else modulus,
                .axis = axis,
                .divShift = if (std.math.isPowerOfTwo(dim.divisor)) @intCast(@ctz(dim.divisor)) else null,
                .isMask = isMask,
            };
        }

        fn value(self: Digit, coordinate: usize) usize {
            const quotient = if (self.divShift) |shift| coordinate >> shift else coordinate / self.divisor;
            return if (self.isMask) quotient & self.modulus else quotient % self.modulus;
        }
    };

    const Tiled = struct {
        /// Unit and padding dimensions always hold index 0 and are left out.
        digits: stdx.BoundedArray(Digit, Shape.MAX_RANK + pjrt.DefaultMemoryLayout.MAX_TILE_DIMS) = .empty,
    };

    const Addressing = union(enum) {
        dense,
        strided: ByteStrides,
        tiled: Tiled,
    };

    rank: u4,
    elementSize: usize,
    blockSize: usize,
    alignmentSize: usize,
    addressing: Addressing = .dense,

    fn init(globalShape: Shape, shape: Shape, layout: pjrt.MemoryLayout) PreparedLayout {
        const blockSize = rowBlockSize(globalShape, shape, layout);
        const lastDimSize: usize = if (shape.rank() == 0) 1 else @intCast(shape.dim(-1));
        var result: PreparedLayout = .{
            .rank = shape.rank(),
            .elementSize = shape.dtype().sizeOf(),
            .blockSize = blockSize,
            .alignmentSize = @max(lastDimSize, blockSize),
        };
        switch (layout) {
            .strides => |strides| {
                std.debug.assert(strides.byte_strides.len == shape.rank());
                const dense = shape.computeByteStrides();
                if (std.mem.eql(i64, strides.byte_strides, dense.constSlice())) return result;

                var byteStrides: ByteStrides = @splat(0);
                for (strides.byte_strides, 0..) |stride, axis| {
                    stdx.debug.assert(stride >= 0, "negative destination byte strides are unsupported, got {}", .{stride});
                    byteStrides[axis] = @intCast(stride);
                }

                result.addressing = .{ .strided = byteStrides };
            },
            .tiled => |tiled| {
                stdx.debug.assert(tiled.minor_to_major.len == shape.rank(), "layout rank {} doesn't match shape rank {}", .{ tiled.minor_to_major.len, shape.rank() });
                if (tiled.tile_dims.len == 0 and isMinorToMajorRowMajor(tiled.minor_to_major, shape.rank())) return result;

                // The expanded dimensions are laid out row-major.
                const expanded: TiledDims = .init(shape, tiled);
                var prepared: Tiled = .{};
                var byteStride = result.elementSize;
                var i = expanded.dims.len;
                while (i > 0) {
                    i -= 1;
                    const dim = expanded.dims.get(i);
                    if (dim.size > 1) if (dim.axis) |axis| prepared.digits.appendAssumeCapacity(.init(axis, dim, byteStride));
                    byteStride *= dim.size;
                }

                result.addressing = .{ .tiled = prepared };
            },
        }
        return result;
    }

    fn byteOffset(self: *const PreparedLayout, local: usize, coordinates: HostStagedBuffer.Coordinates) usize {
        switch (self.addressing) {
            .dense => return local * self.elementSize,
            .strided => |*byteStrides| {
                var offset: usize = 0;
                for (coordinates[0..self.rank], byteStrides[0..self.rank]) |coord, byteStride| offset += coord * byteStride;
                return offset;
            },
            .tiled => |*tiled| {
                var offset: usize = 0;
                for (tiled.digits.constSlice()) |digit| offset += digit.value(coordinates[digit.axis]) * digit.byteStride;
                return offset;
            },
        }
    }
};

/// The physical dimensions of a tiled layout, major to minor. XLA lays a tiled array
/// out row-major over these dimensions; see https://openxla.org/xla/tiled_layout.
const TiledDims = struct {
    /// Every dimension holds the digit `(coordinate[axis] / divisor) % modulus`.
    const Dim = struct {
        /// Null for padding, which only holds index 0.
        axis: ?u3,
        size: usize,
        divisor: usize = 1,
        /// Zero for a digit without an upper bound, which holds `coordinate / divisor`.
        modulus: usize = 0,

        /// The index of the tile along this dimension.
        fn tileCount(self: Dim, tile: usize) Dim {
            const size = std.math.divCeil(usize, self.size, tile) catch unreachable;
            const axis = self.axis orelse return .{ .axis = null, .size = size };
            if (self.modulus == 0) return .{ .axis = axis, .size = size, .divisor = self.divisor * tile };
            if (self.modulus % tile == 0) return .{ .axis = axis, .size = size, .divisor = self.divisor * tile, .modulus = self.modulus / tile };
            // One tile holds every value of the digit.
            return .{ .axis = null, .size = size };
        }

        /// The index within the tile along this dimension.
        fn tileIndex(self: Dim, tile: usize) Dim {
            const axis = self.axis orelse return .{ .axis = null, .size = tile };
            stdx.debug.assert(
                self.modulus == 0 or self.modulus % tile == 0 or tile % self.modulus == 0,
                "nested tile {} must divide, or be a multiple of, the {} values it tiles",
                .{ tile, self.modulus },
            );
            if (self.modulus != 0 and self.modulus < tile) return .{ .axis = axis, .size = tile, .divisor = self.divisor, .modulus = self.modulus };
            return .{ .axis = axis, .size = tile, .divisor = self.divisor, .modulus = tile };
        }
    };

    dims: stdx.BoundedArray(Dim, Shape.MAX_RANK + 2 * pjrt.DefaultMemoryLayout.MAX_TILE_DIMS) = .empty,

    /// Follows XLA's LayoutUtil::LinearIndexForNestedTiling. The physical dimensions
    /// start in minor_to_major order. Each tile then splits the most minor ones,
    /// padded with leading unit dimensions if the tile has more, into tile counts
    /// followed by indices within the tile. Tile sizes are listed major to minor, so
    /// a later tile splits the previous tile's indices and possibly its counts.
    fn init(shape: Shape, layout: pjrt.MemoryLayout.Tiled) TiledDims {
        const rank = shape.rank();
        stdx.debug.assert(layout.minor_to_major.len == rank, "layout rank {} doesn't match shape rank {}", .{ layout.minor_to_major.len, rank });
        var result: TiledDims = .{};
        var i = rank;
        while (i > 0) {
            i -= 1;
            const axis: u3 = @intCast(layout.minor_to_major[i]);
            result.dims.appendAssumeCapacity(.{ .axis = axis, .size = @intCast(shape.dim(axis)) });
        }
        var cursor: usize = 0;
        for (layout.tile_dims_sizes) |tileRank| {
            const tile = layout.tile_dims[cursor..][0..tileRank];
            cursor += tileRank;
            while (result.dims.len < tileRank) result.dims.insert(0, .{ .axis = null, .size = 1 }) catch unreachable;
            const start = result.dims.len - tileRank;
            for (tile, start..) |size_, d| {
                // XLA marks dimensions combined with the next one by a negative size.
                stdx.debug.assert(size_ > 0, "unsupported tile dimension {}", .{size_});
                const size: usize = @intCast(size_);
                result.dims.appendAssumeCapacity(result.dims.get(d).tileIndex(size));
                result.dims.set(d, result.dims.get(d).tileCount(size));
            }
        }
        stdx.debug.assert(cursor == layout.tile_dims.len, "layout tile metadata is inconsistent", .{});
        return result;
    }
};

/// Unprepared reference calculation used to check cached layout strides.
fn referenceByteOffset(shape: Shape, layout: pjrt.MemoryLayout, index: []const usize) usize {
    return switch (layout) {
        .tiled => |tiled| tiledElementOffset(shape, tiled, index) * shape.dtype().sizeOf(),
        .strides => |strides| b: {
            var byte_offset: i64 = 0;
            for (index, strides.byte_strides) |coord, stride| {
                stdx.debug.assert(stride >= 0, "negative destination byte strides are unsupported, got {}", .{stride});
                byte_offset += @as(i64, @intCast(coord)) * stride;
            }
            stdx.debug.assert(byte_offset >= 0, "destination byte offset must be non-negative, got {}", .{byte_offset});
            break :b @as(usize, @intCast(byte_offset));
        },
    };
}

/// Computes the element offset for one logical index in XLA/PJRT tiled layout order,
/// as XLA's LayoutUtil::LinearIndexForNestedTiling does.
fn tiledElementOffset(shape: Shape, layout: pjrt.MemoryLayout.Tiled, index: []const usize) usize {
    const rank = shape.rank();
    stdx.debug.assert(layout.minor_to_major.len == rank, "layout rank {} doesn't match shape rank {}", .{ layout.minor_to_major.len, rank });

    // Physical dimensions and indices, major to minor.
    const Expanded = stdx.BoundedArray(usize, Shape.MAX_RANK + 2 * pjrt.DefaultMemoryLayout.MAX_TILE_DIMS);
    var dims: Expanded = .empty;
    var indices: Expanded = .empty;
    var i = rank;
    while (i > 0) {
        i -= 1;
        const axis: usize = @intCast(layout.minor_to_major[i]);
        dims.appendAssumeCapacity(@intCast(shape.dim(axis)));
        indices.appendAssumeCapacity(index[axis]);
    }

    var cursor: usize = 0;
    for (layout.tile_dims_sizes) |tile_rank| {
        const tile = layout.tile_dims[cursor..][0..tile_rank];
        cursor += tile_rank;
        while (dims.len < tile_rank) {
            dims.insert(0, 1) catch unreachable;
            indices.insert(0, 0) catch unreachable;
        }
        // Tiled dimensions become tile counts, followed by the indices within the tile.
        const start = dims.len - tile_rank;
        for (tile, start..) |tile_dim_, d| {
            const tile_dim: usize = @intCast(tile_dim_);
            dims.appendAssumeCapacity(tile_dim);
            indices.appendAssumeCapacity(indices.get(d) % tile_dim);
            dims.set(d, std.math.divCeil(usize, dims.get(d), tile_dim) catch unreachable);
            indices.set(d, indices.get(d) / tile_dim);
        }
    }
    stdx.debug.assert(cursor == layout.tile_dims.len, "layout tile metadata is inconsistent", .{});

    var offset: usize = 0;
    for (dims.constSlice(), indices.constSlice()) |dim, coord| offset = offset * dim + coord;
    return offset;
}

/// Checks whether PJRT minor-to-major order matches ZML's contiguous row-major shape order.
fn isMinorToMajorRowMajor(minor_to_major: []const i64, rank: usize) bool {
    if (minor_to_major.len != rank) return false;
    for (minor_to_major, 0..) |axis, i| {
        if (axis != @as(i64, @intCast(rank - i - 1))) return false;
    }
    return true;
}

test "prepared layout byte offsets match physical layouts" {
    const unitTiles: [pjrt.DefaultMemoryLayout.MAX_TILE_DIMS]i64 = @splat(1);
    const tileRanks: [pjrt.DefaultMemoryLayout.MAX_NUM_TILES]usize = @splat(Shape.MAX_RANK);
    const Case = struct { shape: Shape, layout: pjrt.MemoryLayout, blockSize: usize };
    const cases = [_]Case{
        .{ .shape = .init(.{}, .i32), .layout = .{ .strides = .{ .byte_strides = &.{} } }, .blockSize = 1 },
        .{ .shape = .init(.{ 3, 5 }, .i32), .layout = .{ .strides = .{ .byte_strides = &.{ 20, 4 } } }, .blockSize = 15 },
        .{ .shape = .init(.{ 3, 5 }, .f16), .layout = .{ .strides = .{ .byte_strides = &.{ 16, 2 } } }, .blockSize = 5 },
        .{ .shape = .init(.{ 3, 5 }, .i32), .layout = .{ .strides = .{ .byte_strides = &.{ 4, 16 } } }, .blockSize = 1 },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .blockSize = 15,
        },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .blockSize = 1,
        },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} } },
            .blockSize = 2,
        },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} } },
            .blockSize = 1,
        },
        .{
            .shape = .init(.{ 3, 5 }, .f16),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2, 2, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .blockSize = 1,
        },
        .{
            .shape = .init(.{ 4, 8 }, .f16),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 4, 2, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .blockSize = 1,
        },
        // The nested tile is larger than the values it tiles.
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2, 4, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .blockSize = 1,
        },
        // The nested tile also splits the tile counts.
        .{
            .shape = .init(.{ 4, 8 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 1, 4, 2, 1, 1 }, .tile_dims_sizes = &.{ 2, 3 } } },
            .blockSize = 1,
        },
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 4, 2 }, .tile_dims_sizes = &.{2} } },
            .blockSize = 2,
        },
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 3, 2 }, .tile_dims_sizes = &.{2} } },
            .blockSize = 2,
        },
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 3, 2, 3, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .blockSize = 1,
        },
        // The nested tile moves pairs of the innermost axis to the most minor dimension.
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 2, 1 }, .tile_dims = &.{ 2, 4, 2, 2, 1 }, .tile_dims_sizes = &.{ 3, 2 } } },
            .blockSize = 2,
        },
        // The tile has more dimensions than the shape.
        .{
            .shape = .init(.{5}, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{0}, .tile_dims = &.{ 2, 4 }, .tile_dims_sizes = &.{2} } },
            .blockSize = 4,
        },
        .{
            .shape = .init(.{ 1, 1, 1, 1, 1, 1, 2, 3 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 7, 6, 5, 4, 3, 2, 1, 0 }, .tile_dims = &unitTiles, .tile_dims_sizes = &tileRanks } },
            .blockSize = 3,
        },
    };
    for (cases) |case| {
        const prepared: PreparedLayout = .init(case.shape, case.shape, case.layout);
        try std.testing.expectEqual(case.blockSize, prepared.blockSize);
        for (0..case.shape.count()) |local| {
            const index = unflattenIndex(case.shape, local);
            const expected = referenceByteOffset(case.shape, case.layout, index[0..case.shape.rank()]);
            const actual = prepared.byteOffset(local, index);
            try std.testing.expectEqual(expected, actual);
            // Elements after the first of a block follow the previous one in memory.
            if ((local % prepared.alignmentSize) % prepared.blockSize != 0) {
                try std.testing.expectEqual(prepared.byteOffset(local - 1, unflattenIndex(case.shape, local - 1)) + prepared.elementSize, actual);
            }
        }
    }
}

test "tiled layout offsets follow XLA" {
    const Tiled = pjrt.MemoryLayout.Tiled;
    // Figure 1 of https://openxla.org/xla/tiled_layout: F32[3,5]{1,0:T(2,2)}.
    const tiles: Tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(17, tiledElementOffset(.init(.{ 3, 5 }, .f32), tiles, &.{ 2, 3 }));
    const shape = Shape.init(.{ 3, 4 }, .i32);
    try std.testing.expectEqual(0, tiledElementOffset(shape, tiles, &.{ 0, 0 }));
    try std.testing.expectEqual(1, tiledElementOffset(shape, tiles, &.{ 0, 1 }));
    try std.testing.expectEqual(2, tiledElementOffset(shape, tiles, &.{ 1, 0 }));
    try std.testing.expectEqual(3, tiledElementOffset(shape, tiles, &.{ 1, 1 }));
    try std.testing.expectEqual(4, tiledElementOffset(shape, tiles, &.{ 0, 2 }));
    try std.testing.expectEqual(8, tiledElementOffset(shape, tiles, &.{ 2, 0 }));

    // Figure 2: 4x8 tiled by (2,4)(2,1). The nested tile pairs rows within each
    // tile, as the (8,128)(2,1) tiling of bf16 on TPU does.
    const nested: Tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 4, 2, 1 }, .tile_dims_sizes = &.{ 2, 2 } };
    const figure2 = [4][8]usize{
        .{ 0, 2, 4, 6, 8, 10, 12, 14 },
        .{ 1, 3, 5, 7, 9, 11, 13, 15 },
        .{ 16, 18, 20, 22, 24, 26, 28, 30 },
        .{ 17, 19, 21, 23, 25, 27, 29, 31 },
    };
    for (figure2, 0..) |offsets, row| {
        for (offsets, 0..) |offset, col| try std.testing.expectEqual(offset, tiledElementOffset(.init(.{ 4, 8 }, .bf16), nested, &.{ row, col }));
    }

    // A tile lists sizes for the most minor physical dimensions, major to minor.
    // Like {2,1,0:T(8,128)} on TPU, this tiles rows by 2 and columns by 4:
    // offset = a*32 + (b/2)*16 + (c/4)*8 + (b%2)*4 + c%4.
    const rank3: Shape = .init(.{ 2, 3, 5 }, .f32);
    const partial: Tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 2, 4 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(4, tiledElementOffset(rank3, partial, &.{ 0, 1, 0 }));
    try std.testing.expectEqual(8, tiledElementOffset(rank3, partial, &.{ 0, 0, 4 }));
    try std.testing.expectEqual(12, tiledElementOffset(rank3, partial, &.{ 0, 1, 4 }));
    try std.testing.expectEqual(16, tiledElementOffset(rank3, partial, &.{ 0, 2, 0 }));
    try std.testing.expectEqual(51, tiledElementOffset(rank3, partial, &.{ 1, 2, 3 }));

    // Tiles follow the physical order: offset = (c/2)*8 + (r/2)*4 + (c%2)*2 + r%2.
    const transposed: Tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(1, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 1, 0 }));
    try std.testing.expectEqual(2, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 0, 1 }));
    try std.testing.expectEqual(4, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 2, 0 }));
    try std.testing.expectEqual(8, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 0, 2 }));
    try std.testing.expectEqual(14, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 2, 3 }));

    // A tile with more dimensions than the shape pads it with leading unit
    // dimensions: offset = (c/4)*8 + c%4.
    const wide: Tiled = .{ .minor_to_major = &.{0}, .tile_dims = &.{ 2, 4 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(3, tiledElementOffset(.init(.{5}, .f32), wide, &.{3}));
    try std.testing.expectEqual(8, tiledElementOffset(.init(.{5}, .f32), wide, &.{4}));

    // A later tile can also split the previous tile counts:
    // offset = r*8 + (c%4)*2 + (c/4)%2.
    const crossTile: Tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 1, 4, 2, 1, 1 }, .tile_dims_sizes = &.{ 2, 3 } };
    try std.testing.expectEqual(1, tiledElementOffset(.init(.{ 4, 8 }, .f32), crossTile, &.{ 0, 4 }));
    try std.testing.expectEqual(2, tiledElementOffset(.init(.{ 4, 8 }, .f32), crossTile, &.{ 0, 1 }));
    try std.testing.expectEqual(11, tiledElementOffset(.init(.{ 4, 8 }, .f32), crossTile, &.{ 1, 5 }));
    try std.testing.expectEqual(31, tiledElementOffset(.init(.{ 4, 8 }, .f32), crossTile, &.{ 3, 7 }));
}

test "HostStagedBuffer.View borrows its pointers through slicing" {
    const shape = Shape.init(.{4}, .i32);
    const layout: pjrt.MemoryLayout = .{ .strides = .{ .byte_strides = &.{4} } };
    var storage = [_]i32{ 0, 1, 2, 3 };
    var other = [_]i32{ 10, 11, 12, 13 };
    var ptrs = [_][*]u8{@ptrCast(&storage)};
    const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, layout, &.{@splat(0)});
    const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
    const nested = view.slice(1, null).slice(0, 2);

    // Slices read the pointers in place, which lets a buffer refresh them
    // without copying them into every view.
    try std.testing.expectEqual(@as([*]const [*]u8, &ptrs), nested.ptrs.ptr);
    ptrs[0] = @ptrCast(&other);
    nested.fill(99);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, &storage);
    try std.testing.expectEqualSlices(i32, &.{ 10, 99, 99, 13 }, &other);
}

test "HostStagedBuffer.View doesn't borrow source layout metadata" {
    const shape: Shape = .init(.{ 2, 3 }, .i32);
    for ([_]bool{ false, true }) |tiled| {
        var byteStrides = [_]i64{ 4, 8 };
        var minorToMajor = [_]i64{ 1, 0 };
        var tileDims = [_]i64{ 2, 2 };
        var tileRanks = [_]usize{2};
        const layout: pjrt.MemoryLayout = if (tiled)
            .{ .tiled = .{ .minor_to_major = &minorToMajor, .tile_dims = &tileDims, .tile_dims_sizes = &tileRanks } }
        else
            .{ .strides = .{ .byte_strides = &byteStrides } };
        var storage: [8]i32 = @splat(-1);
        const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, layout, &.{@splat(0)});
        const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &.{@ptrCast(&storage)});

        // Prepared addressing must survive changes to every source descriptor.
        @memset(&byteStrides, 0);
        @memset(&minorToMajor, 0);
        @memset(&tileDims, 0);
        @memset(&tileRanks, 0);

        view.fillIota(0, 1);
        const expected: [8]i32 = if (tiled) .{ 0, 1, 3, 4, 2, -1, 5, -1 } else .{ 0, 3, 1, 4, 2, 5, -1, -1 };
        try std.testing.expectEqualSlices(i32, &expected, &storage);
        var actual: [6]i32 = undefined;
        view.copyTo(&actual);
        try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5 }, &actual);
        try expectReverseBlocksMirrorForward(view.slice(1, 4));
    }
}

test "HostStagedBuffer.View blocks skip tile padding" {
    const shape = Shape.init(.{ 3, 5 }, .i32);
    const layout: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 1, 0 },
        .tile_dims = &.{ 2, 2 },
        .tile_dims_sizes = &.{2},
    } };
    var storage: [24]i32 = @splat(-1);
    const ptrs = [_][*]u8{@ptrCast(&storage)};
    const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, layout, &.{@splat(0)});
    const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
    try std.testing.expectEqual(2, view.prepared.preparedLayout.blockSize);
    try std.testing.expectEqual(HostStagedBuffer.View(i32).Bounds{ .start = 0, .end = 2 }, view.blockBounds(1));
    try std.testing.expectEqual(HostStagedBuffer.View(i32).Bounds{ .start = 4, .end = 5 }, view.blockBounds(4));
    var iterator = view.blocks();
    while (iterator.next()) |block| {
        for (block.items) |items| {
            for (items, block.offset..) |*value, i| value.* = @intCast(i + 1);
        }
    }
    try std.testing.expectEqualSlices(i32, &.{
        1,  2,  6,  7,  3,  4,  8,  9,  5,  -1, 10, -1,
        11, 12, -1, -1, 13, 14, -1, -1, 15, -1, -1, -1,
    }, &storage);
    view.slice(4, 3).fill(90);
    try std.testing.expectEqualSlices(i32, &.{
        1,  2,  90, 90, 3,  4,  8,  9,  90, -1, 10, -1,
        11, 12, -1, -1, 13, 14, -1, -1, 15, -1, -1, -1,
    }, &storage);
}

test "HostStagedBuffer.View nested tiles" {
    const shape = Shape.init(.{ 3, 5 }, .i32);
    const layout: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 1, 0 },
        .tile_dims = &.{ 2, 2, 2, 1 },
        .tile_dims_sizes = &.{ 2, 2 },
    } };
    var storage: [24]i32 = @splat(-1);
    const ptrs = [_][*]u8{@ptrCast(&storage)};
    const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, layout, &.{@splat(0)});
    const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
    // Pairs of rows are interleaved within each tile, so every block is one element.
    try std.testing.expectEqual(1, view.prepared.preparedLayout.blockSize);
    view.copyFrom(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
    try std.testing.expectEqualSlices(i32, &.{
        1,  6,  2,  7,  3,  8,  4,  9,  5,  10, -1, -1,
        11, -1, 12, -1, 13, -1, 14, -1, 15, -1, -1, -1,
    }, &storage);
}

test "HostStagedBuffer.View dense, transposed, strided and partial-rank tiled layouts" {
    const shape = Shape.init(.{ 2, 3 }, .i32);
    const Case = struct { layout: pjrt.MemoryLayout, row_block_size: usize, expected: []const i32 };
    const cases = [_]Case{
        .{
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .row_block_size = 6,
            .expected = &.{ 1, 2, 3, 4, 5, 6, -1, -1 },
        },
        .{
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .row_block_size = 1,
            .expected = &.{ 1, 4, 2, 5, 3, 6, -1, -1 },
        },
        .{
            .layout = .{ .strides = .{ .byte_strides = &.{ 12, 4 } } },
            .row_block_size = 6,
            .expected = &.{ 1, 2, 3, 4, 5, 6, -1, -1 },
        },
        .{
            .layout = .{ .strides = .{ .byte_strides = &.{ 20, 4 } } },
            .row_block_size = 3,
            .expected = &.{ 1, 2, 3, -1, -1, 4, 5, 6 },
        },
        .{
            .layout = .{ .strides = .{ .byte_strides = &.{ 4, 12 } } },
            .row_block_size = 1,
            .expected = &.{ 1, 4, -1, 2, 5, -1, 3, 6 },
        },
        .{
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{2}, .tile_dims_sizes = &.{1} } },
            .row_block_size = 3,
            .expected = &.{ 1, 2, 3, -1, 4, 5, 6, -1 },
        },
    };
    for (cases) |case| {
        var storage: [8]i32 = @splat(-1);
        const ptrs = [_][*]u8{@ptrCast(&storage)};
        const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, case.layout, &.{@splat(0)});
        const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
        try std.testing.expectEqual(case.row_block_size, view.prepared.preparedLayout.blockSize);
        view.copyFrom(&.{ 1, 2, 3, 4, 5, 6 });
        try std.testing.expectEqualSlices(i32, case.expected, &storage);
    }
}

test "HostStagedBuffer.View row-major layouts return one block across rows and replicas" {
    const shape: Shape = .init(.{ 3, 5 }, .i32);
    const layouts = [_]pjrt.MemoryLayout{
        .{ .strides = .{ .byte_strides = &.{ 20, 4 } } },
        .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
    };
    for (layouts) |layout| {
        var storage: [2][15]i32 = @splat(@splat(0));
        const ptrs = [_][*]u8{ @ptrCast(&storage[0]), @ptrCast(&storage[1]) };
        const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, layout, &.{ @splat(0), @splat(0) });
        const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
        for ([_]HostStagedBuffer.View(i32){ view, view.slice(1, null).slice(1, 10) }) |selected| {
            var forward = selected.blocks();
            var reverse = selected.reverseBlocks();
            for ([_]HostStagedBuffer.View(i32).Block{ forward.next().?, reverse.next().? }) |block| {
                try std.testing.expectEqual(0, block.offset);
                try std.testing.expectEqual(selected.len, block.len());
                try std.testing.expectEqual(2, block.items.len);
                for (block.items, &storage) |items, *replica| {
                    try std.testing.expectEqual(replica[selected.start..].ptr, items.ptr);
                    try std.testing.expectEqual(selected.len, items.len);
                }
            }
            try std.testing.expectEqual(null, forward.next());
            try std.testing.expectEqual(null, reverse.next());
            try std.testing.expectEqual(selected.len, selected.readBlock(0).len);
        }
    }
}

test "HostStagedBuffer.View dense shards stop at gaps in global row-major order" {
    const Case = struct {
        shape: Shape,
        shard_shape: Shape,
        row_block_size: usize,
        expected: [2][]const i32,
    };
    const cases = [_]Case{
        .{
            .shape = .init(.{ 2, 6 }, .i32),
            .shard_shape = .init(.{ 2, 3 }, .i32),
            .row_block_size = 3,
            .expected = .{ &.{ 0, 1, 2, 6, 7, 8 }, &.{ 3, 4, 5, 9, 10, 11 } },
        },
        .{
            .shape = .init(.{ 2, 6, 2 }, .i32),
            .shard_shape = .init(.{ 2, 3, 2 }, .i32),
            .row_block_size = 6,
            .expected = .{
                &.{ 0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17 },
                &.{ 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23 },
            },
        },
    };
    for (cases) |case| {
        const strides = case.shard_shape.computeByteStrides();
        var storage: [2][12]i32 = @splat(@splat(-1));
        var ptrs: [2][*]u8 = undefined;
        var origins: [2]HostStagedBuffer.Coordinates = @splat(@splat(0));
        for (&ptrs, &origins, &storage, 0..) |*ptr, *origin, *data, i| {
            ptr.* = @ptrCast(data);
            origin[1] = i * 3;
        }
        const viewPrepared: HostStagedBuffer.Prepared = .init(case.shape, case.shard_shape, .{ .strides = .{ .byte_strides = strides.constSlice() } }, &origins);
        const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
        try std.testing.expectEqual(case.row_block_size, view.prepared.preparedLayout.blockSize);
        try std.testing.expectEqual(case.row_block_size, view.blockBounds(0).end);
        try std.testing.expectEqual(@as(usize, 1), view.readBlock(case.row_block_size - 1).len);

        view.fillIota(0, 1);
        for (storage, case.expected) |data, expected| {
            try std.testing.expectEqualSlices(i32, expected, data[0..expected.len]);
        }
        var actual: [24]i32 = undefined;
        view.copyTo(actual[0..case.shape.count()]);
        for (actual[0..case.shape.count()], 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(i)), value);

        // A partial range crosses both a global gap and a physical shard boundary.
        const start = case.row_block_size - 1;
        const values = [_]i32{ 90, 91, 92, 93 };
        view.slice(start, values.len).copyFrom(&values);
        for (storage, case.expected) |data, expected| {
            for (data[0..expected.len], expected) |value, global| {
                const offset: usize = @intCast(global);
                try std.testing.expectEqual(if (offset >= start and offset < start + values.len) values[offset - start] else global, value);
            }
        }
        var copied: [4]i32 = undefined;
        view.slice(start, copied.len).copyTo(&copied);
        try std.testing.expectEqualSlices(i32, &values, &copied);
    }
}

/// A 4x6 array split into 2x3 shards. Shards 4 to 7 are separate copies of shards 0 to 3.
const TestShards = struct {
    storage: [8][8]i32 = @splat(@splat(-1)),
    ptrs: [8][*]u8 = undefined,
    origins: [8]HostStagedBuffer.Coordinates = @splat(@splat(0)),
    prepared: HostStagedBuffer.Prepared = undefined,

    const shape = Shape.init(.{ 4, 6 }, .i32);
    const shard_shape = Shape.init(.{ 2, 3 }, .i32);
    const tiled: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 1, 0 },
        .tile_dims = &.{ 2, 2 },
        .tile_dims_sizes = &.{2},
    } };

    /// The view borrows the storage, origins and prepared layout, which must stay in place.
    fn view(self: *TestShards, layout: pjrt.MemoryLayout) HostStagedBuffer.View(i32) {
        for (&self.ptrs, &self.origins, &self.storage, 0..) |*ptr, *origin, *data, i| {
            ptr.* = @ptrCast(data);
            origin[0] = ((i % 4) / 2) * 2;
            origin[1] = (i % 2) * 3;
        }
        self.prepared = .init(shape, shard_shape, layout, &self.origins);
        return .init(&self.prepared, &self.ptrs);
    }

    /// With the tiled layout, every copy of each element holds `expected[global]`
    /// and the tile padding is untouched.
    fn expectValues(self: *const TestShards, expected: [24]i32) !void {
        // Explicit offsets for the two rows of a 2x3 shard tiled by 2x2.
        const physical_offsets = [_]usize{ 0, 1, 4, 2, 3, 6 };
        for (self.storage, self.origins) |data, origin| {
            for (physical_offsets, 0..) |physical, local| {
                const global = (origin[0] + local / 3) * 6 + origin[1] + local % 3;
                try std.testing.expectEqual(expected[global], data[physical]);
            }
            try std.testing.expectEqual(-1, data[5]);
            try std.testing.expectEqual(-1, data[7]);
        }
    }
};

test "HostStagedBuffer.View blocks visit partial global ranges in every shard" {
    var fixture: TestShards = .{};
    const view = fixture.view(TestShards.tiled);
    var visits: [24]u8 = @splat(0);
    // Nested slicing must keep block offsets relative to the selected range.
    var iterator = view.slice(1, null).slice(1, 18).blocks();
    while (iterator.next()) |block| {
        try std.testing.expect(block.offset + block.len() <= 18);
        try std.testing.expectEqual(2, block.items.len);
        for (block.items) |items| {
            for (items, block.offset + 2..) |*value, global| {
                visits[global] += 1;
                value.* = @intCast(global);
            }
        }
    }
    var expected: [24]i32 = @splat(-1);
    for (visits, &expected, 0..) |count, *value, global| {
        const selected = global >= 2 and global < 20;
        try std.testing.expectEqual(@as(u8, if (selected) 2 else 0), count);
        if (selected) value.* = @intCast(global);
    }
    try fixture.expectValues(expected);
    // Starting after the first row partition must skip those shards entirely.
    view.set(17, 99);
    try std.testing.expectEqual(99, fixture.storage[3][4]);
    try std.testing.expectEqual(99, fixture.storage[7][4]);

    const Case = struct { step: i32, expected: [18]i32 };
    const cases = [_]Case{
        .{ .step = 3, .expected = .{ 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58 } },
        .{ .step = -2, .expected = .{ 7, 5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15, -17, -19, -21, -23, -25, -27 } },
        .{ .step = 0, .expected = @splat(7) },
    };
    for (cases) |case| {
        view.slice(1, null).slice(1, 18).fillIota(7, case.step);
        expected[2..20].* = case.expected;
        try fixture.expectValues(expected);
    }
}

test "HostStagedBuffer.View blocks merge shuffled shards and group replicas" {
    var fixture: TestShards = .{};
    const original = fixture.view(TestShards.tiled);
    const order = [_]usize{ 3, 4, 1, 6, 2, 0, 7, 5 };
    var ptrs: [8][*]u8 = undefined;
    var origins: [8]HostStagedBuffer.Coordinates = undefined;
    for (order, &ptrs, &origins) |index, *ptr, *origin| {
        ptr.* = original.ptrs[index];
        origin.* = original.prepared.origins[index];
    }
    const viewPrepared: HostStagedBuffer.Prepared = .init(original.prepared.shape, original.prepared.shard_shape, TestShards.tiled, &origins);
    const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
    const selected = view.slice(1, null).slice(1, 18);
    const offsets = [_]usize{ 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16 };
    const lengths = [_]usize{ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 };
    const replicas = [_][2]usize{ .{ 4, 0 }, .{ 1, 5 }, .{ 6, 2 }, .{ 3, 7 } };
    const physicalOffsets = [_]usize{ 0, 1, 4, 2, 3, 6 };
    var iterator = selected.blocks();
    for (offsets, lengths) |offset, len| {
        const block = iterator.next().?;
        try std.testing.expectEqual(offset, block.offset);
        try std.testing.expectEqual(len, block.len());
        try std.testing.expectEqual(2, block.items.len);
        const blockStart = offset + 2;
        const region = blockStart / 12 * 2 + blockStart % 6 / 3;
        const physical = physicalOffsets[blockStart / 6 % 2 * 3 + blockStart % 3];
        for (block.items, replicas[region]) |items, shard| {
            try std.testing.expectEqual(fixture.storage[shard][physical..].ptr, items.ptr);
        }
        try std.testing.expectEqual(block.items[0].ptr, selected.readBlock(offset).ptr);
        for (block.items) |items| {
            for (items, offset + 2..) |*value, global| value.* = @intCast(global);
        }
    }
    try std.testing.expectEqual(null, iterator.next());
    var expected: [24]i32 = @splat(-1);
    for (expected[2..20], 2..) |*value, global| value.* = @intCast(global);
    try fixture.expectValues(expected);
    try expectReverseBlocksMirrorForward(selected);
}

test "HostStagedBuffer.View blocks group the maximum number of replicas" {
    var storage: [Platform.MAX_NUM_DEVICES][4]i32 = @splat(@splat(-1));
    var ptrs: [Platform.MAX_NUM_DEVICES][*]u8 = undefined;
    const origins: [Platform.MAX_NUM_DEVICES]HostStagedBuffer.Coordinates = @splat(@splat(0));
    for (&ptrs, &storage) |*ptr, *items| ptr.* = @ptrCast(items);
    const shape: Shape = .init(.{4}, .i32);
    const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, .{ .strides = .{ .byte_strides = &.{4} } }, &origins);
    const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
    var iterator = view.blocks();
    const block = iterator.next().?;
    try std.testing.expectEqual(0, block.offset);
    try std.testing.expectEqual(4, block.len());
    try std.testing.expectEqual(Platform.MAX_NUM_DEVICES, block.items.len);
    try std.testing.expectEqual(null, iterator.next());
    // The descriptors remain valid after advancing the iterator.
    for (block.items, &storage) |items, *replica| {
        try std.testing.expectEqual(replica[0..].ptr, items.ptr);
        @memset(items, 42);
        try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42, 42 }, replica);
    }
    try expectReverseBlocksMirrorForward(view);
}

test "HostStagedBuffer.View visits the maximum number of distinct shuffled regions" {
    var storage: [Platform.MAX_NUM_DEVICES]i32 = @splat(-1);
    var ptrs: [Platform.MAX_NUM_DEVICES][*]u8 = undefined;
    var origins: [Platform.MAX_NUM_DEVICES]HostStagedBuffer.Coordinates = @splat(@splat(0));
    for (&ptrs, &origins, 0..) |*ptr, *origin, shard| {
        const index = storage.len - shard - 1;
        ptr.* = @ptrCast(&storage[index]);
        origin[0] = index;
    }
    const shape: Shape = .init(.{Platform.MAX_NUM_DEVICES}, .i32);
    const shardShape: Shape = .init(.{1}, .i32);
    const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shardShape, .{ .strides = .{ .byte_strides = &.{4} } }, &origins);
    const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &ptrs);
    view.fillIota(0, 1);
    var forward = view.blocks();
    var reverse = view.reverseBlocks();
    for (storage, 0..) |value, index| {
        try std.testing.expectEqual(@as(i32, @intCast(index)), value);
        const block = forward.next().?;
        try std.testing.expectEqual(index, block.offset);
        try std.testing.expectEqual(1, block.len());
        try std.testing.expectEqual(1, block.items.len);
        try std.testing.expectEqual(storage[index..].ptr, block.items[0].ptr);
        const reverseBlock = reverse.next().?;
        const reverseIndex = storage.len - index - 1;
        try std.testing.expectEqual(reverseIndex, reverseBlock.offset);
        try std.testing.expectEqual(storage[reverseIndex..].ptr, reverseBlock.items[0].ptr);
    }
    try std.testing.expectEqual(null, forward.next());
    try std.testing.expectEqual(null, reverse.next());
}

test "HostStagedBuffer.View multidimensional slices preserve coordinates across carries" {
    const shape: Shape = .init(.{ 2, 3, 5 }, .i32);
    const layouts = [_]pjrt.MemoryLayout{
        .{ .strides = .{ .byte_strides = &.{ 60, 20, 4 } } },
        .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 3, 2 }, .tile_dims_sizes = &.{2} } },
    };
    for (layouts) |layout| {
        var storage: [48]i32 = @splat(-1);
        const viewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, layout, &.{@splat(0)});
        const view: HostStagedBuffer.View(i32) = .init(&viewPrepared, &.{@ptrCast(&storage)});
        view.fillIota(0, 1);
        var expected: [48]i32 = @splat(-1);
        for (0..shape.count()) |local| {
            const index = unflattenIndex(shape, local);
            const physical = referenceByteOffset(shape, layout, index[0..shape.rank()]) / @sizeOf(i32);
            expected[physical] = @intCast(local);
            try std.testing.expectEqual(@as(i32, @intCast(local)), view.get(local));
        }
        try std.testing.expectEqualSlices(i32, &expected, &storage);
        for (0..shape.count() + 1) |start| {
            for (0..shape.count() - start + 1) |len| {
                try expectReverseBlocksMirrorForward(view.slice(start, len));
            }
        }
    }
}

test "HostStagedBuffer.View reverse blocks mirror forward blocks" {
    var fixture: TestShards = .{};
    const layouts = [_]pjrt.MemoryLayout{
        TestShards.tiled,
        .{ .strides = .{ .byte_strides = &.{ 12, 4 } } },
        .{ .strides = .{ .byte_strides = &.{ 4, 8 } } },
    };
    for (layouts) |layout| {
        const view = fixture.view(layout);
        for (0..view.prepared.shape.count() + 1) |start| {
            for (0..view.prepared.shape.count() - start + 1) |len| {
                try expectReverseBlocksMirrorForward(view.slice(start, len));
            }
        }
    }
    // Dense blocks can span multiple rows, and scalar blocks have no row axis.
    const densePrepared: HostStagedBuffer.Prepared = .init(TestShards.shard_shape, TestShards.shard_shape, layouts[1], fixture.origins[0..1]);
    const dense: HostStagedBuffer.View(i32) = .init(&densePrepared, fixture.ptrs[0..1]);
    try expectReverseBlocksMirrorForward(dense.slice(1, 4));
    const scalar_shape: Shape = .init(.{}, .i32);
    const scalarPrepared: HostStagedBuffer.Prepared = .init(scalar_shape, scalar_shape, .{ .strides = .{ .byte_strides = &.{} } }, fixture.origins[0..1]);
    const scalar: HostStagedBuffer.View(i32) = .init(&scalarPrepared, fixture.ptrs[0..1]);
    try expectReverseBlocksMirrorForward(scalar);
}

fn expectReverseBlocksMirrorForward(view: HostStagedBuffer.View(i32)) !void {
    // Blocks borrow their replicas from the iterator, so keep copies.
    const Expected = struct { offset: usize, items: stdx.BoundedArray([]i32, Platform.MAX_NUM_DEVICES) };
    var expected: [48]Expected = undefined;
    var len: usize = 0;
    var offset: usize = 0;
    var forward = view.blocks();
    while (forward.next()) |block| : (len += 1) {
        try std.testing.expectEqual(offset, block.offset);
        expected[len] = .{ .offset = block.offset, .items = try .fromSlice(block.items) };
        offset += block.len();
    }
    try std.testing.expectEqual(view.len, offset);
    var reverse = view.reverseBlocks();
    while (len > 0) {
        len -= 1;
        const block = reverse.next().?;
        try std.testing.expectEqual(expected[len].offset, block.offset);
        try std.testing.expectEqual(expected[len].items.len, block.items.len);
        for (expected[len].items.constSlice(), block.items) |forwardItems, reverseItems| {
            try std.testing.expectEqual(forwardItems.ptr, reverseItems.ptr);
            try std.testing.expectEqual(forwardItems.len, reverseItems.len);
        }
    }
    try std.testing.expectEqual(null, reverse.next());
}

test "HostStagedBuffer.View scalar, empty ranges and rank-one padding" {
    var scalar: i32 = 0;
    const scalarPtrs = [_][*]u8{@ptrCast(&scalar)};
    const shape = Shape.scalar(.i32);
    const scalar_layout: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{}, .tile_dims = &.{}, .tile_dims_sizes = &.{} } };
    const scalarViewPrepared: HostStagedBuffer.Prepared = .init(shape, shape, scalar_layout, &.{@splat(0)});
    const scalarView: HostStagedBuffer.View(i32) = .init(&scalarViewPrepared, &scalarPtrs);
    scalarView.copyFrom(&.{42});
    try std.testing.expectEqual(42, scalar);
    var past_end = scalarView.slice(1, null).blocks();
    try std.testing.expectEqual(null, past_end.next());

    var storage: [8]bool = @splat(false);
    const vectorPtrs = [_][*]u8{@ptrCast(&storage)};
    const vector_shape = Shape.init(.{5}, .bool);
    const vector_layout: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{0}, .tile_dims = &.{4}, .tile_dims_sizes = &.{1} } };
    const vectorPrepared: HostStagedBuffer.Prepared = .init(vector_shape, vector_shape, vector_layout, &.{@splat(0)});
    const vector: HostStagedBuffer.View(bool) = .init(&vectorPrepared, &vectorPtrs);
    vector.fill(true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false, false, false }, &storage);

    const empty_shape = Shape.init(.{0}, .bool);
    const emptyViewPrepared: HostStagedBuffer.Prepared = .init(empty_shape, empty_shape, vector_layout, &.{@splat(0)});
    const emptyView: HostStagedBuffer.View(bool) = .init(&emptyViewPrepared, &vectorPtrs);
    var empty_blocks = emptyView.blocks();
    try std.testing.expectEqual(null, empty_blocks.next());

    // The empty view has no storage or metadata to read.
    const none: HostStagedBuffer.View(u32) = .empty;
    none.fill(0);
    none.copyTo(&.{});
    var none_blocks = none.reverseBlocks();
    try std.testing.expectEqual(null, none_blocks.next());
}

test "HostStagedBuffer devices holding a shard read one allocation" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const api = platform.pjrt_api;
    const rows: i64 = @intCast(platform.devices.len);
    const Case = struct { shape: Shape, sharding: Sharding };
    var cases: stdx.BoundedArray(Case, 3) = .empty;
    cases.appendAssumeCapacity(.{ .shape = .init(.{ .b = rows, .d = 5 }, .i32), .sharding = .replicated });
    cases.appendAssumeCapacity(.{
        .shape = Shape.init(.{ .b = rows, .d = 5 }, .i32).withPartitioning(.{ .b = .model }),
        .sharding = platform.shardings.get("model").?,
    });

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    if (platform.devices.len == 4) {
        // Two data-parallel groups of two tensor-parallel devices.
        const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
            .axis(.link_y, .{ .mesh = .torus }, &.{
                .{ .leaf = .{ .id = 0, .coords = .{ 0, 0, 0, 0 } } },
                .{ .leaf = .{ .id = 1, .coords = .{ 0, 1, 0, 0 } } },
            }),
            .axis(.link_y, .{ .mesh = .torus }, &.{
                .{ .leaf = .{ .id = 2, .coords = .{ 1, 0, 0, 0 } } },
                .{ .leaf = .{ .id = 3, .coords = .{ 1, 1, 0, 0 } } },
            }),
        });
        const physical = try arena.allocator().create(Sharding.PhysicalMesh);
        physical.* = try .fromTree(arena.allocator(), .tpu, topology);
        const data = try arena.allocator().create(Sharding.Data);
        data.* = try .init(
            "data_parallel",
            physical,
            .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
            .parseBindings(.{ .batch = .link_x, .model = .link_y }),
        );
        cases.appendAssumeCapacity(.{
            .shape = Shape.init(.{ .b = 4, .d = 5 }, .i32).withPartitioning(.{ .b = .batch }),
            .sharding = .{ .data = data },
        });
    }

    for (cases.constSlice()) |case| {
        var pinned: HostStagedBuffer = try .initDeviceReadOnly(allocator, io, platform, case.shape, case.sharding);
        defer pinned.deinit(allocator);
        const placement = try pinned.buffer._sharding.placement(case.shape);
        const expectedAllocations = switch (platform.target) {
            .cpu, .cuda, .rocm => case.shape.count() / placement.shape.count(),
            else => platform.devices.len,
        };
        try std.testing.expectEqual(expectedAllocations, pinned.shards.len);

        // Rows are written once, and every device holding them reads them.
        pinned.view(i32).fillIota(0, 1);
        const values = try allocator.alloc(i32, placement.shape.count());
        defer allocator.free(values);
        for (pinned.buffer._shards.constSlice(), pinned.buffer._sharding.devicesInCanonicalOrder()) |shard, device| {
            @memset(values, -1);
            if (try shard.toHostBuffer(api, std.mem.sliceAsBytes(values))) |event| {
                defer event.deinit(api);
                try event.await(api, io);
            }
            const first_row = shardOrigin(&placement, device)[0];
            for (values, 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(first_row * 5 + i)), value);
        }
    }
}

test "HostStagedBuffer executions read shared inputs on every device" {
    const zml = @import("zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const api = platform.pjrt_api;
    const shape = Shape.init(.{ 3, 5 }, .i32);
    const Test = struct {
        fn increment(input: zml.Tensor) zml.Tensor {
            return input.onMemory(.host_pinned).toMemory(.device).addConstant(1);
        }
    };
    var exe = try platform.compileFn(allocator, io, Test.increment, .{zml.Tensor.fromShape(shape)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    var pinned: HostStagedBuffer = try .initDeviceReadOnly(allocator, io, platform, shape, .replicated);
    defer pinned.deinit(allocator);
    for (0..2) |iteration| {
        pinned.view(i32).fillIota(@intCast(iteration * 100), 1);
        var output: Buffer = undefined;
        runner.runOpts(io, .{pinned.buffer}, .{&output}, .{ .wait = false });
        defer output.deinit();
        try output.await(io);
        for (output._shards.constSlice()) |shard| {
            var values: [15]i32 = @splat(-1);
            if (try shard.toHostBuffer(api, std.mem.asBytes(&values))) |event| {
                defer event.deinit(api);
                try event.await(api, io);
            }
            for (values, 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(iteration * 100 + i + 1)), value);
        }
    }
}

test "HostStagedBuffer.View indexes replicas of a dense array directly" {
    const shape = Shape.init(.{ 2, 3 }, .i32);
    var storage: [2][6]i32 = @splat(@splat(-1));
    const ptrs = [_][*]u8{ @ptrCast(&storage[0]), @ptrCast(&storage[1]) };
    const prepared: HostStagedBuffer.Prepared = .init(shape, shape, .{ .strides = .{ .byte_strides = &.{ 12, 4 } } }, &.{ @splat(0), @splat(0) });
    try std.testing.expect(prepared.contiguous);
    const view: HostStagedBuffer.View(i32) = .init(&prepared, &ptrs);
    for (0..6) |i| view.set(i, @intCast(10 + i));
    view.slice(4, null).set(1, 99);
    for (storage) |replica| try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 99 }, &replica);
    try std.testing.expectEqual(99, view.get(5));
    try std.testing.expectEqual(13, view.slice(2, 3).get(1));

    inline for (.{ false, true }) |reverse| {
        const nested = view.slice(1, 4);
        var blocks = if (reverse) nested.reverseBlocks() else nested.blocks();
        const block = blocks.next().?;
        try std.testing.expectEqual(0, block.offset);
        try std.testing.expectEqual(2, block.items.len);
        for (block.items, &storage) |items, *replica| {
            try std.testing.expectEqualSlices(i32, &.{ 11, 12, 13, 14 }, items);
            try std.testing.expectEqual(replica[1..].ptr, items.ptr);
        }
        try std.testing.expectEqual(null, blocks.next());
    }
}

test "HostStagedBuffer.View set and get address single elements in every replica" {
    var fixture: TestShards = .{};
    const view = fixture.view(TestShards.tiled);
    var expected: [24]i32 = undefined;
    var i: usize = expected.len;
    while (i > 0) {
        i -= 1;
        expected[i] = @intCast(100 + i);
        view.set(i, expected[i]);
    }
    try fixture.expectValues(expected);
    for (expected, 0..) |value, index| try std.testing.expectEqual(value, view.get(index));

    // Indices of a nested view are relative to its start.
    const nested = view.slice(7, 10);
    nested.set(3, -5);
    expected[10] = -5;
    try fixture.expectValues(expected);
    try std.testing.expectEqual(-5, nested.get(3));
}

test "HostStagedBuffer.View copies across layouts, shards and replicas" {
    var fixture: TestShards = .{};
    const view = fixture.view(TestShards.tiled);
    view.fill(0);
    view.slice(2, 20).fillIota(10, 1);
    view.set(17, 99);
    const expected = [_]i32{ 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 99, 26, 27, 28, 29, 0, 0 };
    var actual: [24]i32 = undefined;
    view.copyTo(&actual);
    try std.testing.expectEqualSlices(i32, &expected, &actual);
    for (expected, 0..) |value, i| try std.testing.expectEqual(value, view.get(i));
    try fixture.expectValues(expected);

    // Blocks expose every replica, respect physical boundaries, and stop at
    // the end of a nested view even when the physical run continues.
    const nested = view.slice(1, null).slice(2, 4);
    var nestedBlocks = nested.blocks();
    const first = nestedBlocks.next().?;
    try std.testing.expectEqual(0, first.offset);
    try std.testing.expectEqual(2, first.items.len);
    for (first.items, [_]usize{ 1, 5 }) |items, shard| {
        try std.testing.expectEqualSlices(i32, expected[3..5], items);
        try std.testing.expectEqual(fixture.storage[shard][0..].ptr, items.ptr);
    }
    try std.testing.expectEqual(fixture.storage[1][0..].ptr, nested.readBlock(0).ptr);
    const middle = nestedBlocks.next().?;
    try std.testing.expectEqual(2, middle.offset);
    for (middle.items) |items| try std.testing.expectEqualSlices(i32, expected[5..6], items);
    const last = nestedBlocks.next().?;
    try std.testing.expectEqual(3, last.offset);
    for (last.items, [_]usize{ 0, 4 }) |items, shard| {
        try std.testing.expectEqualSlices(i32, expected[6..7], items);
        try std.testing.expectEqual(fixture.storage[shard][2..].ptr, items.ptr);
    }
    try std.testing.expectEqual(null, nestedBlocks.next());

    // Copy between differently shaped/tiled ranges, including a nonzero source
    // and destination start, and verify every destination replica physically.
    const dst_shape = Shape.init(.{ 3, 7 }, .i32);
    const dst_layout: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 0, 1 },
        .tile_dims = &.{},
        .tile_dims_sizes = &.{},
    } };
    var dst_storage: [2][21]i32 = @splat(@splat(-1));
    const dstPtrs = [_][*]u8{ @ptrCast(&dst_storage[0]), @ptrCast(&dst_storage[1]) };
    const dstPrepared: HostStagedBuffer.Prepared = .init(dst_shape, dst_shape, dst_layout, &.{ @splat(0), @splat(0) });
    const dst: HostStagedBuffer.View(i32) = .init(&dstPrepared, &dstPtrs);
    dst.slice(1, 19).copyFromView(view.slice(1, 22).slice(1, 19));
    for (dst_storage) |data| {
        for (0..21) |i| {
            const physical = (i % 7) * 3 + i / 7;
            try std.testing.expectEqual(if (i > 0 and i < 20) expected[i + 1] else -1, data[physical]);
        }
    }
    var empty: [0]i32 = .{};
    view.slice(view.len, 0).copyTo(&empty);
    view.slice(0, 0).copyFromView(dst.slice(0, 0));
    view.slice(view.len, null).fill(-1);
    view.slice(view.len, null).fillIota(-1, 1);
    view.slice(view.len, null).copyFrom(&empty);
    view.copyTo(&actual);
    try std.testing.expectEqualSlices(i32, &expected, &actual);
}

test "HostStagedBuffer await after executable donation permits the next host update" {
    const zml = @import("zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const shape = Shape.init(.{ 3, 5 }, .i32);
    const Test = struct {
        fn increment(input: zml.Tensor) zml.Tensor {
            const host_input = input.onMemory(.host_pinned);
            return host_input.toMemory(.device).addConstant(1).toMemory(.host_pinned).reuseBuffer(host_input);
        }
    };
    var exe = try platform.compileFn(allocator, io, Test.increment, .{zml.Tensor.fromShape(shape)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    var pinned: HostStagedBuffer = try .init(allocator, io, platform, shape, .replicated);
    defer pinned.deinit(allocator);
    try std.testing.expectEqual(pinned.buffer._shards.len, pinned.shards.len);
    const firstPtr = pinned.view(i32).readBlock(0).ptr;
    pinned.view(i32).fillIota(0, 1);

    // Reuse the same runner and output destination as the model execution paths.
    // Only await waits for the asynchronous executable before host access.
    for (0..3) |iteration| {
        runner.runOpts(io, .{pinned.buffer}, .{&pinned.buffer}, .{ .wait = false });
        try pinned.await(io);

        // Reacquire from the output handles, even when donation reuses memory.
        const view = pinned.view(i32);
        var actual: [15]i32 = undefined;
        view.copyTo(&actual);
        try std.testing.expectEqual(firstPtr, view.readBlock(0).ptr);
        for (actual, 0..) |value, i| {
            const expected: i32 = if (iteration == 0 or i < 3 or i >= 12)
                @intCast(i + iteration + 1)
            else
                @intCast(iteration * 100 + i - 3 + 1);
            try std.testing.expectEqual(expected, value);
        }

        // Also check the executable's logical output through PJRT, independently
        // of the host-visible layout used above.
        var device_values: [15]i32 = undefined;
        try pinned.buffer.toSlice(io, .init(shape, std.mem.asBytes(&device_values)));
        try std.testing.expectEqualSlices(i32, &actual, &device_values);

        view.slice(3, 9).fillIota(@intCast((iteration + 1) * 100), 1);
    }
}

test "HostStagedBuffer await follows an output that doesn't reuse the pinned allocation" {
    const zml = @import("zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const shape = Shape.init(.{ 3, 5 }, .i32);
    const Test = struct {
        fn increment(input: zml.Tensor) zml.Tensor {
            // Without reuseBuffer, execution writes a new host allocation.
            return input.onMemory(.host_pinned).toMemory(.device).addConstant(1).toMemory(.host_pinned);
        }
    };
    var exe = try platform.compileFn(allocator, io, Test.increment, .{zml.Tensor.fromShape(shape)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    var pinned: HostStagedBuffer = try .init(allocator, io, platform, shape, .replicated);
    defer pinned.deinit(allocator);
    pinned.view(i32).fillIota(0, 1);

    for (0..2) |iteration| {
        // The input isn't donated, so release it once execution replaced it.
        var input = pinned.buffer;
        defer input.deinit();
        const previous = pinned.view(i32).readBlock(0).ptr;
        runner.runOpts(io, .{input}, .{&pinned.buffer}, .{ .wait = false });
        try pinned.await(io);
        const view = pinned.view(i32);
        try std.testing.expect(view.readBlock(0).ptr != previous);

        var actual: [15]i32 = undefined;
        view.copyTo(&actual);
        for (actual, 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(iteration * 100 + i + 1)), value);
        var device_values: [15]i32 = undefined;
        try pinned.buffer.toSlice(io, .init(shape, std.mem.asBytes(&device_values)));
        try std.testing.expectEqualSlices(i32, &actual, &device_values);

        // Host writes now land in the output, which the next execution reads.
        view.fillIota(@intCast((iteration + 1) * 100), 1);
    }
}
