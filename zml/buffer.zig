const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");

const DataType = @import("dtype.zig").DataType;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const pjrt = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

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

    pub const MAX_NUM_SHARDS: u8 = Platform.MAX_NUM_DEVICES;
    pub const Shards = std.BoundedArray(*pjrt.Buffer, MAX_NUM_SHARDS);

    pub const FromOptions = struct {
        wait: bool = true,
        memory: ?pjrt.Memory.Kind = null,
    };

    /// Copies the content of the given buffer from host memory to the accelerator memory.
    pub fn from(platform: Platform, host_buffer: HostBuffer, opts: FromOptions) !Buffer {
        var res: Buffer = .{
            ._api = platform.pjrt_api,
            ._shape = host_buffer.shape(),
            ._shards = .{},
        };

        // We shard only on the first axis so that the chunks are still contiguous.
        // TODO: support more advanced sharding specs
        stdx.debug.assert(platform.sharding().num_replicas == 1, "ZML doesn't support num_replicas > 1 for now, got: {}", .{platform.sharding()});
        const sharding_ax: ?u3 = std.simd.firstTrue(host_buffer.shape()._sharding_info);
        const n_partitions = platform.sharding().num_partitions;
        const chunk_size = if (sharding_ax) |ax| cs: {
            // This kind of sharding error should be detected earlier on.
            stdx.debug.assert(@rem(host_buffer.dim(ax), n_partitions) == 0, "Buffer.from({}) expects the sharding axis {} to have a dimension divisble by the number of devices ({}).", .{ host_buffer, ax, n_partitions });
            break :cs @divExact(host_buffer.dim(ax), n_partitions);
        } else 0;

        const buffer_type = bufferTypeFromDtype(host_buffer.shape().dtype());
        const byte_strides = host_buffer.strides();

        const devices = platform.getDevices();
        for (0..n_partitions) |i| {
            // If no sharding if found, the given buffer is replicated on all devices.
            const buf = if (sharding_ax) |ax| buf: {
                const start: i64 = @as(i64, @intCast(i)) * chunk_size;
                break :buf host_buffer.slice1d(ax, .{ .start = start, .end = start + chunk_size });
            } else host_buffer;

            var args = pjrt.Client.BufferFromHostBufferArgs{
                .data = buf._data,
                .buffer_type = buffer_type,
                .dims = buf.shape().dims(),
                .byte_strides = byte_strides,
                .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            };
            if (opts.memory) |memory_kind| {
                const memories = try devices[i].addressableMemories(platform.pjrt_api);
                const memory = for (memories) |m| {
                    const kind = m.kind(platform.pjrt_api);
                    if (kind == memory_kind) break m;
                } else return error.NotFound;
                args.memory = memory;
            } else {
                args.device = devices[i];
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

    /// Wraps pre-exisiting `pjrt.Buffer` shards into one `zml.Buffer`.
    pub fn fromPjrtBuffers(platform: Platform, shape_: Shape, pjrt_buffers: []const *pjrt.Buffer) Buffer {
        stdx.debug.assert(pjrt_buffers.len <= MAX_NUM_SHARDS, "ZML doesn't support having more than {} shards. Received {} shards for one buffer.", .{ MAX_NUM_SHARDS, pjrt_buffers.len });
        stdx.debug.assert(pjrt_buffers.len > 0, "fromPjrtBuffers expects at least one buffer, got 0.", .{});
        var shards: Shards = .{};
        shards.appendSliceAssumeCapacity(pjrt_buffers);
        return .{
            ._api = platform.pjrt_api,
            ._shape = shape_,
            ._shards = shards,
        };
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromSlice(platform: Platform, dimz: anytype, s: anytype) !Buffer {
        const sh = Shape.init(dimz, DataType.fromSliceElementType(s));
        return from(platform, HostBuffer.fromBytes(sh, std.mem.sliceAsBytes(s)), .{});
    }

    /// Copies the given Zig slice to the accelerator memory and
    /// return a Buffer with the given dimensions.
    pub fn fromBytes(platform: Platform, sh: Shape, data: []const u8) !Buffer {
        return from(platform, HostBuffer.fromBytes(sh, data), .{});
    }

    /// Copies the given Zig array to the accelerator memory and
    /// return a Buffer using the array shape.
    pub fn fromArray(platform: Platform, arr: anytype) !Buffer {
        const host_buffer = HostBuffer.fromArray(&arr);
        return try from(platform, host_buffer, .{ .wait = true });
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
        return from(platform, HostBuffer.fromBytes(sh, data), opts);
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
        return try from(platform, host_buffer, .{ .wait = true });
    }

    /// Creates a Buffer with a single element repeated manytime.
    pub fn constant(platform: Platform, shape_: Shape, val: anytype) !Buffer {
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
            const host_buffer: HostBuffer = .{
                ._shape = shape_,
                ._strides = @splat(0),
                ._data = x.constSlice().ptr,
            };
            return try from(platform, host_buffer, .{ .wait = true });
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
        return try from(platform, host_buffer, .{ .wait = true });
    }

    test constant {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();

        const x = try constant(platform, Shape.init(.{ 4, 3, 2 }, .u16), 42);
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
        const maybe_event = try self._shards.get(0).toHostBuffer(self._api, std.mem.asBytes(&res));
        if (maybe_event) |event| {
            try event.await_(self._api);
        }
        return res;
    }

    /// Copies the content of the Buffer back to host, in the given buffer,
    /// and return a new `HostBuffer` object with the same shape.
    /// The returned `HostBuffer` doesn't own the memory.
    pub fn toHost(self: Buffer, output: []u8) !HostBuffer {
        stdx.debug.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const maybe_event = try self._shards.get(0).toHostBuffer(self._api, output);
        if (maybe_event) |event| {
            try event.await_(self._api);
        }
        return HostBuffer.fromBytes(self.shape(), output);
    }

    /// Copies the content of the Buffer to the host.
    /// The returned `HostBuffer` does own the memory.
    pub fn toHostAlloc(self: Buffer, allocator: std.mem.Allocator) !HostBuffer {
        const output = try HostBuffer.empty(allocator, self.shape());
        stdx.debug.internalAssert(!self.hasShardedAxis(), "TODO: support sharded Buffer -> Host transfer", .{});
        const maybe_event = try self._shards.get(0).toHostBuffer(self._api, @constCast(output.bytes()));
        if (maybe_event) |event| {
            try event.await_(self._api);
        }
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

    pub fn getMemory(self: Buffer) *const pjrt.Memory {
        const shard = self._shards.get(0);
        return shard.memory(self._api);
    }

    fn hasShardedAxis(self: Buffer) bool {
        if (self._shards.len == 1) return false;
        return @reduce(.Or, self._shape._sharding_info);
    }

    pub fn copyToMemory(self: Buffer, memory: *const pjrt.Memory) !Buffer {
        var new_shards: Buffer.Shards = .{};
        for (self._shards.slice()) |shard| {
            const new_shard = try shard.copyToMemory(self._api, memory);
            new_shards.appendAssumeCapacity(new_shard);
        }

        return Buffer{ ._shape = self._shape, ._shards = new_shards, ._api = self._api };
    }
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
