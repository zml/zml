const builtin = @import("builtin");
const std = @import("std");
const stdx = @import("stdx");

const pjrtx = @import("pjrtx.zig");
const Context = @import("context.zig").Context;
const DataType = @import("dtype.zig").DataType;
const Device = @import("pjrtx.zig").Device;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

const log = std.log.scoped(.@"zml/partitioning");

test {
    std.testing.refAllDecls(Mesh);
    std.testing.refAllDecls(DeviceShard);
    std.testing.refAllDecls(Sharding);
}

pub const MaxMeshSize: u8 = 64;
pub const MaxMeshAxes: u8 = 8;

pub const TopologyIndicesIterator = struct {
    index: i64 = 0,
    len: i64,
    topology: Shape,
    indices: Shape,

    pub fn next(self: *TopologyIndicesIterator) ?Shape {
        if (self.index >= self.len) return null;

        defer {
            self.index += 1;
            self.computeNextIndices();
        }

        return self.indices;
    }

    fn computeNextIndices(self: *TopologyIndicesIterator) void {
        var next_indices = self.indices;

        // Change: Start from the last dimension and go to the first.
        var i: usize = self.topology.rank();
        while (i > 0) {
            i -= 1;
            const dim: u4 = @intCast(i);

            const current_value = next_indices.dim(dim);
            if (current_value < self.topology.dim(dim) - 1) {
                self.indices = next_indices.setDim(dim, current_value + 1);
                return; // Found the dimension to increment, we are done.
            } else {
                // This dimension has wrapped around. Reset it to 0 and let the
                // loop continue to the next (more major) dimension.
                next_indices = next_indices.setDim(dim, 0);
            }
        }

        self.indices = next_indices;
    }
};

pub const Mesh = struct {
    topology: Shape,

    pub fn init(topology: anytype) Mesh {
        const self: Mesh = .{
            .topology = .init(topology, .u8),
        };

        if (self.rank() == 0) {
            stdx.debug.panic("Mesh must have at least one tagged axis defined, got: {}", .{topology});
        }

        if (!self.topology.isFullyTagged()) {
            stdx.debug.panic("Mesh must be fully tagged, got: {}", .{topology});
        }

        return self;
    }

    pub fn rank(self: Mesh) i64 {
        return @intCast(self.topology.rank());
    }

    pub fn axis(self: Mesh, ax: anytype) i64 {
        return self.topology.dim(ax);
    }

    pub fn isPartitioned(self: Mesh) bool {
        return self.numPartitions() > 1;
    }

    pub fn isSinglePartition(self: Mesh) bool {
        return self.numPartitions() == 1;
    }

    pub fn iterator(self: Mesh) TopologyIndicesIterator {
        var indices = self.topology;

        for (0..indices.rank()) |dim| {
            indices = indices.setDim(dim, 0);
        }

        return .{
            .len = self.numPartitions(),
            .topology = self.topology,
            .indices = indices,
        };
    }

    pub fn numPartitions(self: Mesh) i64 {
        return @intCast(self.topology.count());
    }

    pub fn numReplicas(_: Mesh) i64 {
        return 1;
    }

    pub fn numRequiredDevices(self: Mesh) i64 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn format(
        self: Mesh,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt; // autofix
        _ = options;
        try writer.print("Mesh(topology={} rank={d} numRequiredDevices={d})", .{ self.topology, self.rank(), self.numRequiredDevices() });
    }
};

test Mesh {
    // 1D mesh with no partitions
    const mesh_1d_no_partitions: Mesh = .init(.{ .x = 1 });
    try std.testing.expect(mesh_1d_no_partitions.rank() == 1);
    try std.testing.expect(mesh_1d_no_partitions.axis(.x) == 1);
    try std.testing.expect(!mesh_1d_no_partitions.isPartitioned());
    try std.testing.expect(mesh_1d_no_partitions.isSinglePartition());
    try std.testing.expect(mesh_1d_no_partitions.numPartitions() == 1);
    try std.testing.expect(mesh_1d_no_partitions.numRequiredDevices() == 1);

    // 1D mesh with 8 partitions
    const mesh_1d_8_partitions: Mesh = .init(.{ .x = 8 });
    try std.testing.expect(mesh_1d_8_partitions.rank() == 1);
    try std.testing.expect(mesh_1d_8_partitions.axis(.x) == 8);
    try std.testing.expect(mesh_1d_8_partitions.isPartitioned());
    try std.testing.expect(!mesh_1d_8_partitions.isSinglePartition());
    try std.testing.expect(mesh_1d_8_partitions.numPartitions() == 8);
    try std.testing.expect(mesh_1d_8_partitions.numRequiredDevices() == 8);

    // 2D mesh with 6 partitions
    const mesh_2d: Mesh = .init(.{ .x = 2, .y = 3 });
    try std.testing.expect(mesh_2d.rank() == 2);
    try std.testing.expect(mesh_2d.axis(.x) == 2);
    try std.testing.expect(mesh_2d.axis(.y) == 3);
    try std.testing.expect(mesh_2d.isPartitioned());
    try std.testing.expect(!mesh_2d.isSinglePartition());
    try std.testing.expect(mesh_2d.numPartitions() == 6);
    try std.testing.expect(mesh_2d.numRequiredDevices() == 6);

    // 3D mesh with 64 partitions
    const mesh_3d: Mesh = .init(.{ .x = 4, .y = 4, .z = 4 });
    try std.testing.expect(mesh_3d.rank() == 3);
    try std.testing.expect(mesh_3d.axis(.x) == 4);
    try std.testing.expect(mesh_3d.axis(.y) == 4);
    try std.testing.expect(mesh_3d.axis(.z) == 4);
    try std.testing.expect(mesh_3d.isPartitioned());
    try std.testing.expect(!mesh_3d.isSinglePartition());
    try std.testing.expect(mesh_3d.numPartitions() == 64);
    try std.testing.expect(mesh_3d.numRequiredDevices() == 64);
}

pub const DeviceShard = struct {
    index: i64,
    len: i64,
    shape: Shape,
    shard: Shape,
    indices: Shape,
    topology: Shape,
    device: *const Device,

    /// Holds the layout information for a shard within a larger host buffer.
    pub const PjRTArgs = struct {
        /// The byte offset from the start of the host buffer to this shard's data.
        /// The caller adds this to the host buffer's base address to get the final pointer.
        start_offset: usize,

        /// The byte strides of the *global* tensor layout on the host.
        /// This is a fixed-size array; the caller should slice it using `num_dims`
        /// to get the `*const i64` and `size_t` for the PJRT C API.
        byte_strides: [Shape.MAX_RANK]i64,

        /// The dimensions of the data slice to be transferred to the device.
        /// For replicated axes, this will be the full global dimension.
        /// For partitioned axes, this will be the smaller shard dimension.
        dims: [Shape.MAX_RANK]i64,

        /// The number of valid dimensions (and strides) for this layout.
        num_dims: u4,
    };

    /// Calculates the necessary arguments for a PJRT_Client_BufferFromHostBuffer call.
    pub fn pjrtArgs(self: DeviceShard) PjRTArgs {
        const rank = self.shape.rank();
        const element_size_bytes: i64 = @intCast(self.shape.dtype().sizeOf());
        _ = element_size_bytes; // autofix

        // Step 1: Calculate the byte strides for the GLOBAL host buffer.
        // These strides describe how to navigate the full, unpartitioned tensor in host memory.
        // This layout is the same for all shards.
        // Example: For a global shape {m=16, k=16} of i32 (4 bytes), the strides are:
        //   - stride for 'k' (dim 1): 4 bytes
        //   - stride for 'm' (dim 0): 16 (dim k) * 4 bytes = 64 bytes
        const host_byte_strides_ba = self.shape.computeStrides();
        const host_byte_strides = host_byte_strides_ba.constSlice();

        // Step 2: Calculate the start offset for THIS specific shard.
        // We determine the starting coordinate of our shard's data slice within the global tensor
        // and use the global strides to find the byte offset.
        var shard_start_offset_bytes: i64 = 0;
        for (0..rank) |i| {
            // Check if the i-th dimension of the tensor is partitioned.
            const mesh_axis_tag = self.shape.partition(i);

            if (mesh_axis_tag != Shape.TagUnknown) {
                // This dimension IS partitioned. We need to calculate its contribution to the offset.

                // a) Get this device's coordinate along the relevant mesh axis.
                //    Example: for indices={x=1, y=2} and a dimension partitioned on 'y', this is 2.
                const device_coord_on_mesh_axis = self.indices.dim(mesh_axis_tag);

                // b) Get the size of the shard along this tensor dimension.
                //    Example: for k=16/x on an x=8 mesh, the shard size for 'k' is 16/8 = 2.
                const shard_dim_size = self.shard.dim(i);

                // c) The starting element for this dimension is `coord * size`.
                //    Example: For device with x=3 and shard size 2, the start is element 3 * 2 = 6.
                const start_element_in_dim = device_coord_on_mesh_axis * shard_dim_size;

                // d) Add the byte offset for this dimension to the total.
                shard_start_offset_bytes += start_element_in_dim * host_byte_strides[i];
            }
            // If the dimension is replicated, its starting coordinate is 0, so it adds 0 to the offset.
        }

        // Step 3: Determine the dimensions of the SLICE to transfer.
        // This is what PJRT will actually read, using the start_offset and host_byte_strides.
        var transfer_dims_buffer: [Shape.MAX_RANK]i64 = undefined;
        for (0..rank) |i| {
            const mesh_axis_tag = self.shape.partition(i);
            if (mesh_axis_tag == Shape.TagUnknown) {
                // This dimension is REPLICATED on this shard. The slice must span the
                // entire global dimension.
                transfer_dims_buffer[i] = self.shape.dim(i);
            } else {
                // This dimension is PARTITIONED. The slice is just the size of the
                // smaller shard dimension.
                transfer_dims_buffer[i] = self.shard.dim(i);
            }
        }

        // Step 4: Assemble and return the complete struct for the PJRT call.
        return .{
            .start_offset = @intCast(shard_start_offset_bytes),
            .byte_strides = host_byte_strides_ba.buffer,
            .dims = transfer_dims_buffer,
            .num_dims = rank,
        };
    }

    pub fn format(
        self: DeviceShard,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt; // autofix
        _ = options;
        const pjrt_args = self.pjrtArgs();
        try writer.print("DeviceShard(index={d}/{d} topology={} indices={} shard={} ({d}B) shape={} ({d}B) start_offset={d} byte_strides={any} device={})", .{
            self.index + 1,
            self.len,
            self.topology,
            self.indices,
            self.shard,
            self.shard.byteSize(),
            self.shape,
            self.shape.byteSize(),
            pjrt_args.start_offset,
            pjrt_args.byte_strides[0..pjrt_args.num_dims],
            self.device,
        });
    }
};

pub const DeviceShardIterator = struct {
    index: i64 = 0,
    len: i64,
    platform: Platform,
    indices_iterator: TopologyIndicesIterator,
    sharding: Sharding,

    pub fn next(self: *DeviceShardIterator) ?DeviceShard {
        if (self.index >= self.sharding.mesh.numPartitions()) return null;

        defer self.index += 1;

        return .{
            .index = self.index,
            .len = self.len,
            .shape = self.sharding.shape,
            .shard = self.sharding.shard(),
            .indices = self.indices_iterator.next().?,
            .topology = self.sharding.mesh.topology,
            .device = self.platform.getDevices()[@intCast(self.index)],
        };
    }

    fn devices(self: DeviceShardIterator) []const *const Device {
        // todo: compute mapping from indices to physical devices.
        // For now, we just return the devices from the platform.
        return self.platform.getDevices();
    }
};

pub const Sharding = struct {
    mesh: Mesh,
    shape: Shape,
    strides: []const i64, // todo : check usage of strides, maybe remove?

    pub const Type = enum {
        replicated,
        maximal,
        manual,
    };

    pub fn init(mesh: Mesh, shape: Shape) Sharding {
        return .{
            .mesh = mesh,
            .shape = shape,
            .strides = shape.computeStrides().constSlice(),
        };
    }

    pub fn getType(self: Sharding) Type {
        if (!self.shape.hasAtLeastOnePartitionedAxis()) {
            return .replicated;
        }

        if (self.shape.isFullyPartitioned()) {
            return .maximal;
        }

        return .manual;
    }

    pub fn shard(self: Sharding) Shape {
        var s: Shape = .init(.{}, self.shape.dtype());

        for (0..self.shape.rank()) |dim| {
            const mesh_axis = self.shape.partition(dim);

            if (mesh_axis == Shape.TagUnknown) {
                s = s.appendDim(self.shape.dim(dim), self.shape.tag(dim));
            } else {
                const mesh_dim = self.mesh.topology.dim(mesh_axis);
                const d = @divExact(self.shape.dim(dim), mesh_dim);
                s = s.appendDim(d, self.shape.tag(dim));
            }
        }

        return s;
    }

    pub fn iterator(self: Sharding, platform: Platform) DeviceShardIterator {
        const indices_iterator = self.mesh.iterator();

        return .{
            .len = indices_iterator.len,
            .platform = platform,
            .indices_iterator = indices_iterator,
            .sharding = self,
        };
    }

    pub fn shardingString(self: Sharding) []const u8 {
        var sharding_str: std.BoundedArray(u8, 128) = .{};
        self.writeShardingRepresentation(sharding_str.writer()) catch unreachable;
        return sharding_str.constSlice();
    }

    pub fn writeShardingRepresentation(self: Sharding, writer: anytype) @TypeOf(writer).Error!void {
        if (self.getType() == .replicated) {
            try writer.writeAll("{replicated}");
            return;
        }
        try writer.writeAll("{devices=[");
        for (0..self.shape.rank()) |i| {
            const mesh_axis = self.shape.partition(i);

            var dim: i64 = 1;

            if (mesh_axis != Shape.TagUnknown) {
                const mesh_dim = self.mesh.topology.dim(mesh_axis);
                dim = mesh_dim;
            }

            try writer.print("{d}", .{dim});
            if (i < self.shape.rank() - 1) try writer.writeByte(',');
        }
        try writer.print("]<=[{d}]}}", .{self.mesh.numPartitions()});
    }

    pub fn format(
        self: Sharding,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt; // autofix
        _ = options;
        try writer.print("Sharding(shape={} mesh={})", .{ self.shape, self.mesh });
    }
};

test "Sharding" {
    {
        // const mesh: Mesh = .init(.{ .x = 1 });
        // var shape: Shape = .init(.{ .m = 2, .k = 6 }, .i32);

        // const mesh: Mesh = .init(.{ .x = 8 });
        // var shape: Shape = .init(.{ .m = 16, .k = 16 }, .i32);

        const mesh: Mesh = .init(.{ .x = 2, .y = 3 });
        var shape: Shape = .init(.{ .m = 2, .k = 6 }, .i32);

        // const mesh: Mesh = .init(.{ .x = 2, .y = 2 });
        // var shape: Shape = .init(.{ .m = 2, .k = 2 }, .i32);

        const platform = env(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } });

        // Partially partitioned sharding
        const shape_partitioned = shape.withPartitionning(.{ .y = .k });
        // const shape_partitioned = shape.withPartitionning(.{});
        const sharding: Sharding = .init(mesh, shape_partitioned);

        std.debug.print("Sharding: {}\n", .{sharding});

        var iter = sharding.iterator(platform);

        const demo_slice = try allocateDemoSlice(std.testing.allocator, shape_partitioned);
        defer std.testing.allocator.free(demo_slice);

        std.debug.print("Full slice of data: {d}\n\n", .{items(demo_slice, shape_partitioned, i32)});

        while (iter.next()) |device_shard| {
            std.debug.print("{}\n", .{device_shard});

            const shard_on_device = try toDevice(platform, device_shard, demo_slice);
            defer shard_on_device.deinit();
            std.debug.print("{any} - shard data slice: {any}\n\n", .{ shard_on_device, items(shard_on_device.data, shard_on_device.shape(), i32) });
        }
    }
}

pub const ShardOnDevice = struct {
    dims: []const i64,
    buffer_type: pjrtx.BufferType,
    size: usize,
    data: []u8,

    pub fn format(
        self: ShardOnDevice,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt; // autofix
        _ = options;
        try writer.print("ShardOnDevice(dims={any} buffer_type={s} size={d})", .{
            self.dims,
            @tagName(self.buffer_type),
            self.size,
        });
    }

    pub fn shape(self: ShardOnDevice) Shape {
        return Shape.init(self.dims, dtypeFromBufferType(self.buffer_type));
    }

    pub fn deinit(self: ShardOnDevice) void {
        std.testing.allocator.free(self.data);
    }
};

pub fn toDevice(platform: Platform, shard: DeviceShard, slice: []u8) !ShardOnDevice {
    const shard_pjrt = shard.pjrtArgs();
    const args = pjrtx.Client.BufferFromHostBufferArgs{
        .data = slice[shard_pjrt.start_offset..].ptr,
        .buffer_type = bufferTypeFromDtype(shard.shard.dtype()),
        .dims = shard_pjrt.dims[0..shard_pjrt.num_dims], // Slice the fixed array
        .byte_strides = shard_pjrt.byte_strides[0..shard_pjrt.num_dims], // Slice the fixed array
        .host_buffer_semantics = .ImmutableUntilTransferCompletes,
        .device = shard.device,
    };

    // std.debug.print("Creating buffer on device {any} with args {any}\n", .{ shard.device, args });

    const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);

    if (event) |ev| {
        try ev.await_(platform.pjrt_api);
    }

    const dims = pjrt_buffer.getDimensions(platform.pjrt_api);
    const buffer_type = pjrt_buffer.getElementType(platform.pjrt_api);
    const size = try pjrt_buffer.getOnDeviceSizeInBytes(platform.pjrt_api);

    const data_buffer = try std.testing.allocator.alloc(u8, size);
    const to_host_event = try pjrt_buffer.toHostBuffer(platform.pjrt_api, data_buffer);

    if (to_host_event) |ev| {
        try ev.await_(platform.pjrt_api);
    }

    return .{
        .dims = dims,
        .buffer_type = buffer_type,
        .size = size,
        .data = data_buffer,
    };

    // pjrt_buffer.deinit(platform.pjrt_api);
    // pjrt_buffer.toHostBuffer(api: *const Api, dst: []u8)
}

pub fn allocateDemoSlice(allocator: std.mem.Allocator, shape: Shape) ![]u8 {
    const start: i64 = 0;
    const step: i64 = 1;
    const slice = try allocator.alloc(u8, shape.byteSize());

    switch (shape.dtype()) {
        inline else => |d| if (comptime d.class() != .integer) {
            stdx.debug.assert(shape.dtype().class() == .integer, "arange expects type to be integer, got {} instead.", .{shape.dtype()});
        } else {
            const Zt = d.toZigType();
            var j: i64 = start;
            for (items(slice, shape, Zt)) |*val| {
                val.* = @intCast(j);
                j +%= step;
            }
        },
    }

    return slice;
}

fn items(slice: []u8, shape: Shape, T: type) []T {
    const ptr: [*]T = @alignCast(@constCast(@ptrCast(slice.ptr)));
    return ptr[0..shape.count()];
}

pub fn bufferTypeFromDtype(dt: DataType) pjrtx.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrtx.BufferType, @tagName(tag)),
    };
}

pub fn dtypeFromBufferType(pjrt_type: pjrtx.BufferType) DataType {
    return switch (pjrt_type) {
        .invalid => @panic("Found an invalid pjrt buffer"),
        inline else => |tag| @field(DataType, @tagName(tag)),
    };
}

// todo: temp (zig deps and compilation story...)
var _platform: ?Platform = null;

pub fn env(opts: Platform.CreateOptions) Platform {
    if (!builtin.is_test) @compileError("Cannot use zml.testing.env outside of a test block");
    if (_platform == null) {
        var ctx = Context.init() catch unreachable;
        _platform = ctx.autoPlatform(opts).withCompilationOptions(.{
            .xla_dump_to = "/tmp/zml/tests-partitioning/",
            .sharding_enabled = true,
        });
    }

    return _platform.?;
}
