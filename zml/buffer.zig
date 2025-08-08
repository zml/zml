const std = @import("std");

const asynk = @import("async");
const runtimes = @import("runtimes");
const stdx = @import("stdx");

const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const pjrt = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const createDeviceMesh = @import("platform.zig").createDeviceMesh;
const Shape = @import("shape.zig").Shape;
const partitioning = @import("partitioning.zig");
const Mesh = partitioning.Mesh;
const Sharding = partitioning.Sharding;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Buffer);
}

const log = std.log.scoped(.zml);

/// Buffer is a multi-dimension array, whose memory is allocated on an accelerator.
///
/// * contains a handle that the ZML runtime can use to convert into a physical address, but there is no guarantee this address is visible from the CPU.
/// * loading weights from disk directly to the `device zml.aio.loadBuffers`
/// * can be created by calling `HostBuffer.toDevice(platform)`.
pub const Buffer = struct {
    pub const Memory = enum {
        host,
        host_pinned,
        device,

        pub fn toPjrtMemory(self: Memory) pjrt.Memory.Kind {
            return switch (self) {
                .host => .unpinned_host,
                .host_pinned => .pinned_host,
                .device => .device,
            };
        }

        pub fn pjrtName(self: Memory) []const u8 {
            return @tagName(self.toPjrtMemory());
        }
    };

    _shape: Shape,
    _api: *const pjrt.Api,
    _shards: Shards,
    _mesh: ?Mesh = null, // ? todo

    pub const MAX_NUM_SHARDS: u8 = Platform.MAX_NUM_DEVICES;
    pub const Shards = std.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const FromOptions = struct {
        wait: bool = true,
        memory: ?Memory = null,
    };

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(platform: Platform, sharding: Sharding, data: []const u8, opts: FromOptions) !Buffer {
        var device_list_buffer: [1024]u8 = undefined; // todo remove allocator in createDeviceMesh
        var fba = std.heap.FixedBufferAllocator.init(&device_list_buffer);
        const ordered_devices = try createDeviceMesh(fba.allocator(), sharding.mesh, platform);

        var sharding_devices = sharding.iterator(ordered_devices);

        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._shape = sharding.global_shape,
            ._shards = .{},
        };

        while (sharding_devices.next()) |shard| {
            const specs = shard.specs();

            var args = pjrt.Client.BufferFromHostBufferArgs{
                .data = data[specs.start_offset..].ptr,
                .buffer_type = bufferTypeFromDtype(shard.shard_shape.dtype()),
                .dims = specs.dims[0..specs.num_dims],
                .byte_strides = specs.byte_strides[0..specs.num_dims],
                .device = shard.device,
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            };

            if (opts.memory) |memory_kind| {
                const memories = try shard.device.addressableMemories(platform.pjrt_api);
                const memory = for (memories) |m| {
                    const kind = m.kind(platform.pjrt_api);
                    if (kind == memory_kind.toPjrtMemory()) break m;
                } else return error.NotFound;
                args.memory = memory;
            } else {
                args.device = shard.device;
            }

            const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);

            if (event) |ev| {
                ev.deinit(platform.pjrt_api);
            }

            res._shards.appendAssumeCapacity(pjrt_buffer);
        }

        if (opts.wait) {
            res = try res.awaitt();
        }

        return res;
    }
    pub fn awaitt(self: Buffer) !Buffer {
        for (self._shards.constSlice()) |buffer| {
            if (buffer.getReadyEvent(self._api)) |ev| {
                try ev.await_(self._api);
            }
        }

        return self;
    }

    pub const UnitializedOptions = struct {
        memory: ?Memory = null,
    };

    pub fn uninitialized(platform: Platform, sharding: Sharding, opts: UnitializedOptions) !Buffer {
        var sharding_devices = sharding.iterator(platform.getDevices());

        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._shape = sharding.global_shape,
            ._shards = .{},
        };

        errdefer for (res._shards.slice()) |shard| {
            shard.deinit(platform.pjrt_api);
        };

        const minor_to_major: [Shape.MAX_RANK]i64 = comptime blk: {
            var minor_to_major: [Shape.MAX_RANK]i64 = undefined;
            for (0..Shape.MAX_RANK) |i| {
                minor_to_major[i] = @intCast(Shape.MAX_RANK - i - 1);
            }
            break :blk minor_to_major;
        };

        while (sharding_devices.next()) |shard| {
            const specs = shard.specs();

            var args = pjrt.Client.CreateUninitializedBufferArgs{
                .dims = specs.dims[0..specs.num_dims],
                .element_type = bufferTypeFromDtype(shard.shard_shape.dtype()),
                .layout = .{
                    .tiled = .{
                        .minor_to_major = minor_to_major[Shape.MAX_RANK - shard.shard_shape.rank() ..],
                        .tile_dims = &.{},
                        .tile_dims_sizes = &.{},
                    },
                },
            };

            if (opts.memory) |memory_kind| {
                const memories = try shard.device.addressableMemories(platform.pjrt_api);
                const memory = for (memories) |m| {
                    const kind = m.kind(platform.pjrt_api);
                    if (kind == memory_kind.toPjrtMemory()) break m;
                } else return error.NotFound;
                args.memory = memory;
            } else {
                args.device = shard.device;
            }

            const pjrt_buffer = try platform.pjrt_client.createUnitializedBuffer(platform.pjrt_api, args);

            res._shards.appendAssumeCapacity(pjrt_buffer);
        }

        return res;
    }

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: Platform, sharding: Sharding, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        stdx.debug.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        stdx.debug.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});
        var shards: Shards = .{};
        shards.appendSliceAssumeCapacity(pjrt_buffers);
        return .{
            ._api = platform.pjrt_api,
            ._shape = sharding.global_shape,
            ._shards = shards,
            ._mesh = sharding.mesh,
        };
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromSlice(platform: Platform, mesh: Mesh, dimz: anytype, s: anytype) !Buffer {
        const sh: Shape = if (comptime @TypeOf(dimz) == Shape)
            dimz
        else
            Shape.init(dimz, DataType.fromSliceElementType(s));

        const sharding: Sharding = .init(mesh, sh);
        return from(platform, sharding, std.mem.sliceAsBytes(s), .{});
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytes(platform: Platform, sh: Shape, data: []const u8) !Buffer {
        const mesh = Mesh.single();
        const sharding: Sharding = .init(mesh, sh);
        return from(platform, sharding, data, .{});
    }

    /// Copies the given Zig array to the accelerator memory and
    /// return a Buffer using the array shape.
    pub fn fromArray(platform: Platform, arr: anytype) !Buffer {
        const host_buffer = HostBuffer.fromArray(&arr);
        const mesh = Mesh.single();
        const sharding: Sharding = .init(mesh, host_buffer.shape());
        return try from(platform, sharding, host_buffer.bytes(), .{ .wait = true });
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromSliceOpts(platform: Platform, dimz: anytype, s: anytype, opts: FromOptions) !Buffer {
        const sh = Shape.init(dimz, DataType.fromSliceElementType(s));
        return from(platform, HostBuffer.fromBytes(sh, std.mem.sliceAsBytes(s)), opts);
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytesOpts(platform: Platform, sh: Shape, data: []const u8, opts: FromOptions) !Buffer {
        const mesh = Mesh.single();
        const sharding: Sharding = .init(mesh, sh);
        return from(platform, sharding, data, opts);
    }

    /// Copies the given Zig array to the accelerator memory and
    /// return a Buffer using the array shape.
    pub fn fromArrayOpts(platform: Platform, arr: anytype, opts: FromOptions) !Buffer {
        const host_buffer = HostBuffer.fromArray(&arr);
        return try from(platform, host_buffer, opts);
    }

    pub fn asPinnedHostBuffer(self: Buffer) HostBuffer {
        // TODO restore assert
        // const memory = self.getMemory().kind(self._api);
        // stdx.debug.assert(memory == .pinned_host, "asPinnedHostBuffer({}) expects a buffer allocated on host memory, got {}. see `toMemory`", .{ self, memory });
        const ptr: [*]u8 = @ptrCast(self._shards.get(0).getOpaqueDeviceMemoryDataPointer(self._api) catch unreachable);
        return HostBuffer.fromBytes(self._shape, ptr[0..self._shape.byteSize()]);
    }

    /// Creates a Buffer with a single element.
    pub fn scalar(platform: Platform, val: anytype, dtype_: DataType) !Buffer {
        const x = dtype_.constant(val);
        const host_buffer = HostBuffer.fromBytes(Shape.init(.{}, dtype_), x.constSlice());
        return try from(platform, .init(Mesh.single(), host_buffer.shape()), host_buffer.bytes(), .{ .wait = true });
    }

    /// Creates a Buffer with a single element repeated manytime.
    pub fn constant(platform: Platform, sharding: Sharding, val: anytype) !Buffer {
        const shape_ = sharding.global_shape;
        var start = try std.time.Timer.start();
        defer {
            const duration_ms = stdx.math.divFloat(f32, start.read(), std.time.ns_per_ms);
            if (duration_ms > 100) {
                const size_gb = stdx.math.divFloat(f32, shape_.byteSize(), 1024 * 1024 * 1024);
                log.info("Wrote constant({_}) to device ({d:.2}Gb) in {d:.0}ms: {d:.2}Gb/s", .{ shape_, size_gb, duration_ms, size_gb / duration_ms * 1000 });
            }
        }

        // Convert val to the requested dtype.
        const x = shape_.dtype().constant(val);
        const byte_size = shape_.dtype().sizeOf();
        const max_bytes = 1024;

        // Naive version for scalars and buffers with long last axis.
        if (shape_.rank() < 1 or byte_size * shape_.dim(-1) > max_bytes) {
            return try from(platform, sharding, x.constSlice(), .{ .wait = true });
        }

        // To speed up copies, duplicate the scalar value into a vector,
        // so that PJRT can copy row by row.
        // Because this is respecting the shape, it won't work if the last axis is too big.
        // If this becomes an issue, we should create a new intermediary Buffer by splitting last axis into { n, max_bytes }
        // so that the trick works, and then reshape it
        // We could also handle sharded constant directly in this function to avoid having to create too big arrays.
        var bytes: [max_bytes]u8 align(64) = undefined;
        var strides = [1]i64{0} ** Shape.MAX_RANK;
        strides[shape_.rank() - 1] = byte_size;

        switch (byte_size) {
            inline 1, 2, 4, 8, 16 => |b| {
                const Int = std.meta.Int(.unsigned, b * 8);
                const x_as_int: Int = @bitCast(x.constSlice()[0..b].*);
                const bytes_as_int: [*]Int = @ptrCast(&bytes);
                @memset(bytes_as_int[0..@intCast(shape_.dim(-1))], x_as_int);
            },
            else => unreachable,
        }
        const host_buffer: HostBuffer = .{ ._shape = shape_, ._strides = strides, ._data = &bytes };
        return try from(platform, sharding, host_buffer.bytes(), .{ .wait = true });
    }

    test constant {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const x = try constant(platform, .init(zml.Mesh.single(), Shape.init(.{ 4, 3, 2 }, .u16)), 42);
        const y = try x.getValue([4 * 3 * 2]u16);
        try std.testing.expectEqual([_]u16{42} ** (4 * 3 * 2), y);
    }

    /// Creates a Buffer as a view of host memory visible from the device,
    /// thus avoiding a copy.
    ///
    /// Be careful though, as it requires a specific alignment
    /// and it might not work on all platforms,
    /// could lead to crashes and operations on the buffer will be slower.
    /// Tested on Cuda 12.4.
    pub fn asViewOfHostBuffer(platform: Platform, buf: HostBuffer) Buffer {
        return asViewOfDeviceBuffer(platform, buf.shape(), null, @constCast(buf._data));
    }

    /// Creates a Buffer from a pointer into device memory.
    /// This allows to interface with other libraries producing buffers.
    pub fn asViewOfDeviceBuffer(platform: Platform, shape_: Shape, stream: ?*const pjrt.Stream, device_data: *anyopaque) Buffer {
        const minor_to_major: [Shape.MAX_RANK]i64 = comptime blk: {
            var res: [Shape.MAX_RANK]i64 = undefined;
            for (0..Shape.MAX_RANK) |i| {
                res[i] = @intCast(Shape.MAX_RANK - i - 1);
            }
            break :blk res;
        };

        const pjrt_buffer = platform.pjrt_client.createViewOfDeviceBuffer(platform.pjrt_api, .{
            .data = device_data,
            .element_type = bufferTypeFromDtype(shape_.dtype()),
            .dims = shape_.dims(),
            // TODO: exposes sharding in the API.
            .device = platform.getDevices()[0],
            .layout = .{
                .tiled = .{
                    .minor_to_major = minor_to_major[Shape.MAX_RANK - shape_.rank() ..],
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .stream = stream,
        }) catch @panic("failed to createViewOfDeviceBuffer");

        var shards: Shards = .{};
        shards.appendAssumeCapacity(pjrt_buffer);
        return .{
            ._api = platform.pjrt_api,
            ._shape = shape_,
            ._shards = shards,
        };
    }

    /// Fetches the content of the given buffer into a stack variable of the given type.
    pub fn getValue(self: Buffer, T: type) !T {
        stdx.debug.assert(self._shape.byteSize() == @sizeOf(T), "Buffer {} has {d} bytes of data, can't load it to a {s} with {d} bytes", .{ self, self._shape.byteSize(), @typeName(T), @sizeOf(T) });
        var res: T = undefined;
        stdx.debug.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const maybe_event = try self._shards.get(0).toHostBuffer(self._api, std.mem.asBytes(&res), .{});
        if (maybe_event) |event| {
            try event.await_(self._api);
        }
        return res;
    }

    /// Copies the content of the Buffer to the host.
    /// The returned `HostBuffer` does own the memory.
    pub fn toHostBuffer(self: Buffer, sharding: Sharding, allocator: std.mem.Allocator) !HostBuffer {
        var self_with_mesh = self;
        self_with_mesh._mesh = sharding.mesh;
        const data = try self_with_mesh.toHost(allocator);
        return HostBuffer.fromBytes(self_with_mesh.shape(), data);
    }

    pub fn toHostBuffer2(self: Buffer, output: []u8) !HostBuffer {
        var fba = std.heap.FixedBufferAllocator.init(output);
        const allocator = fba.allocator();
        const buffer = try self.toHost(allocator);
        return HostBuffer.fromBytes(self.shape(), buffer);
    }

    pub fn toHost(self: Buffer, allocator: std.mem.Allocator) ![]u8 {
        const sharding: Sharding = .init(self._mesh.?, self.shape());
        stdx.debug.assert(self._shards.len == sharding.mesh.numPartitions(), "Expected {} PJRT buffers, got {}", .{ sharding.mesh.numPartitions(), self._shards.len });

        const global_buffer = try allocator.alloc(u8, sharding.global_shape.byteSize());
        errdefer allocator.free(global_buffer);

        if (sharding.getType() == .replicated) {
            if (self._shards.len > 0) {
                const event = try self._shards.get(0).toHostBuffer(self._api, global_buffer, .{});
                if (event) |e| try e.await_(self._api);
            }
            return global_buffer;
        }

        const shard_shape = sharding.shardShape();
        const temp_shard_buffer = try allocator.alloc(u8, shard_shape.byteSize());
        defer allocator.free(temp_shard_buffer);

        const element_size = sharding.global_shape.dtype().sizeOf();
        const global_element_strides = sharding.global_shape.computeElementStrides();
        const rk = sharding.global_shape.rank();

        var devices: [MAX_NUM_SHARDS]*const pjrt.Device = undefined;
        for (self._shards.constSlice(), 0..) |shard, i| {
            devices[i] = try shard.getDevice(self._api);
        }

        var sharding_iter = sharding.iterator(devices[0..self._shards.len]);
        while (sharding_iter.next()) |device_shard| {
            const shard_index: usize = @intCast(device_shard.index);
            const src_buffer = self._shards.get(shard_index);

            const event = try src_buffer.toHostBuffer(self._api, temp_shard_buffer, .{});
            if (event) |e| try e.await_(self._api);
            if (rk == 0) {
                if (global_buffer.len > 0) {
                    @memcpy(global_buffer[0..element_size], temp_shard_buffer[0..element_size]);
                }
                continue;
            }

            const last_dim_size = shard_shape.dim(rk - 1);
            const contiguous_row_bytes = @as(usize, @intCast(last_dim_size)) * element_size;
            const outer_dims_shape = shard_shape.remove(rk - 1);

            var iter = outer_dims_shape.iterator();

            while (iter.next()) |item| {
                const src_flat_index_start_of_row = item.flat_index * @as(usize, @intCast(last_dim_size));
                const src_byte_offset = src_flat_index_start_of_row * element_size;

                var dest_flat_index_start_of_row: i64 = 0;

                for (0..rk) |dim_idx| {
                    const shard_coord_in_dim = if (dim_idx < rk - 1) item.coords[dim_idx] else 0;

                    const global_coord_for_dim = blk: {
                        const part_spec = sharding.global_shape.partition(dim_idx);
                        if (part_spec != .axis) break :blk shard_coord_in_dim; // Replicated axis
                        const mesh_axis_tag = part_spec.toTag();
                        if (device_shard.indices.hasTag(mesh_axis_tag) != null) {
                            const coord = device_shard.indices.dim(mesh_axis_tag);
                            break :blk (coord * shard_shape.dim(dim_idx)) + shard_coord_in_dim;
                        } else {
                            break :blk shard_coord_in_dim;
                        }
                    };

                    dest_flat_index_start_of_row += global_coord_for_dim * global_element_strides.get(dim_idx);
                }

                const dest_byte_offset = @as(usize, @intCast(dest_flat_index_start_of_row)) * element_size;

                if (contiguous_row_bytes > 0) {
                    @memcpy(
                        global_buffer[dest_byte_offset..][0..contiguous_row_bytes],
                        temp_shard_buffer[src_byte_offset..][0..contiguous_row_bytes],
                    );
                }
            }
        }

        return global_buffer;
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

    pub fn getMemory(self: Buffer) *const pjrt.Memory {
        const shard = self._shards.get(0);
        return shard.memory(self._api);
    }

    fn hasShardedAxis(self: Buffer) bool {
        if (self._shards.len == 1) return false;
        return self._shape.hasAtLeastOnePartitionedAxis();
    }

    pub fn copyToMemory(self: Buffer, memory: *const pjrt.Memory) !Buffer {
        var new_shards: Buffer.Shards = .{};
        for (self._shards.slice()) |shard| {
            const new_shard = try shard.copyToMemory(self._api, memory);
            new_shards.appendAssumeCapacity(new_shard);
        }

        return Buffer{ ._shape = self._shape, ._shards = new_shards, ._api = self._api };
    }

    // pub const UnitializedOptions = struct {
    //     memory: ?pjrt.Memory.Kind = null,
    // };

    // pub fn uninitialized(platform: Platform, shape_: Shape, opts: UnitializedOptions) !Buffer {
    //     var res: Buffer = .{
    //         ._api = platform.pjrt_api,
    //         ._shape = shape_,
    //         ._shards = .{},
    //     };
    //     errdefer for (res._shards.slice()) |shard| {
    //         shard.deinit(platform.pjrt_api);
    //     };

    //     const minor_to_major: [Shape.MAX_RANK]i64 = comptime blk: {
    //         var minor_to_major: [Shape.MAX_RANK]i64 = undefined;
    //         for (0..Shape.MAX_RANK) |i| {
    //             minor_to_major[i] = @intCast(Shape.MAX_RANK - i - 1);
    //         }
    //         break :blk minor_to_major;
    //     };

    //     // TODO: support more advanced sharding specs
    //     stdx.debug.assert(platform.sharding().num_replicas == 1, "ZML doesn't support num_replicas > 1 for now, got: {}", .{platform.sharding()});
    //     const sharding_ax: ?u3 = std.simd.firstTrue(shape_._sharding_info);
    //     const n_partitions = platform.sharding().num_partitions;
    //     const shard_shape = if (sharding_ax) |ax| s: {
    //         // This kind of sharding error should be detected earlier on.
    //         stdx.debug.assert(@rem(shape_.dim(ax), n_partitions) == 0, "Buffer.uninitialized() expects the sharding axis {} to have a dimension divisble by the number of devices ({}).", .{ ax, n_partitions });
    //         const shard_shape = shape_.set(ax, @divExact(shape_.dim(ax), n_partitions));
    //         break :s shard_shape;
    //     } else shape_;

    //     const buffer_type = bufferTypeFromDtype(shape_.dtype());
    //     const devices = platform.getDevices();
    //     for (0..n_partitions) |i| {
    //         var args = pjrt.Client.CreateUninitializedBufferArgs{
    //             .dims = shard_shape.dims(),
    //             .element_type = buffer_type,
    //             .layout = .{
    //                 .tiled = .{
    //                     .minor_to_major = minor_to_major[Shape.MAX_RANK - shape_.rank() ..],
    //                     .tile_dims = &.{},
    //                     .tile_dims_sizes = &.{},
    //                 },
    //             },
    //         };
    //         if (opts.memory) |memory_kind| {
    //             const memories = try devices[i].addressableMemories(platform.pjrt_api);
    //             const memory = for (memories) |m| {
    //                 const kind = m.kind(platform.pjrt_api);
    //                 if (kind == memory_kind) break m;
    //             } else return error.NotFound;
    //             args.memory = memory;
    //         } else {
    //             args.device = devices[i];
    //         }
    //         const pjrt_buffer = try platform.pjrt_client.createUnitializedBuffer(platform.pjrt_api, args);

    //         res._shards.appendAssumeCapacity(pjrt_buffer);
    //     }

    //     return res;
    // }
};

pub fn bufferTypeFromDtype(dt: DataType) pjrt.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrt.BufferType, @tagName(tag)),
    };
}

pub fn dtypeFromBufferType(pjrt_type: pjrt.BufferType) DataType {
    return switch (pjrt_type) {
        .invalid => @panic("Found an invalid pjrt buffer"),
        inline else => |tag| @field(DataType, @tagName(tag)),
    };
}

test bufferTypeFromDtype {
    inline for (@typeInfo(DataType).@"enum".fields) |field| {
        const dt: DataType = @enumFromInt(field.value);
        try std.testing.expectEqual(dt, dtypeFromBufferType(bufferTypeFromDtype(dt)));
    }

    inline for (@typeInfo(pjrt.BufferType).@"enum".fields) |field| {
        const dt: pjrt.BufferType = @enumFromInt(field.value);
        if (dt == .invalid) continue;
        try std.testing.expectEqual(dt, bufferTypeFromDtype(dtypeFromBufferType(dt)));
    }
}
