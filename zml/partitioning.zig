const builtin = @import("builtin");
const std = @import("std");
const stdx = @import("stdx");

const pjrtx = @import("pjrtx.zig");
const slice = @import("slice.zig");
const Shaped = slice.Shaped;
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

pub const MaxMeshAxes: u8 = 3;
pub const MaxMeshSize: u8 = MaxMeshAxes * Shape.MAX_RANK;

pub const TopologyIndicesIterator = struct {
    index: usize = 0,
    topology: Shape,
    indices: Shape,

    pub fn next(self: *TopologyIndicesIterator) ?Shape {
        if (self.index >= self.topology.count()) return null;

        defer {
            self.index += 1;
            self.computeNextIndices();
        }

        return self.indices;
    }

    fn computeNextIndices(self: *TopologyIndicesIterator) void {
        var next_indices = self.indices;
        var i: usize = self.topology.rank();

        while (i > 0) {
            i -= 1;
            const dim: u4 = @intCast(i);

            const current_value = next_indices.dim(dim);
            if (current_value < self.topology.dim(dim) - 1) {
                self.indices = next_indices.setDim(dim, current_value + 1);
                return;
            } else {
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

        if (self.rank() > MaxMeshAxes) {
            stdx.debug.panic("Mesh rank ({}) exceeds maximum allowed axes ({})", .{ self.rank(), MaxMeshAxes });
        }

        return self;
    }

    pub fn singlePartition() Mesh {
        const topology: Shape = .init(.{ .x = 1 }, .u8);
        return .init(topology);
    }

    pub fn auto(platform: Platform) Mesh {
        const num_devices = platform.getDevices().len;

        if (num_devices == 0) {
            stdx.debug.panic("No devices available in the platform: {}", .{platform.target});
        }

        if (num_devices > MaxMeshSize) {
            stdx.debug.panic("Too many devices ({}) for a mesh, max is {}", .{ num_devices, MaxMeshSize });
        }

        const topology: Shape = .init(.{ .x = num_devices }, .u8);

        return .init(topology);
    }

    // todo : better API?
    pub fn reshape(self: Mesh, new_shape: anytype) Mesh {
        const new_topology = self.topology.reshape(new_shape);
        return .init(new_topology);
    }

    pub fn rank(self: Mesh) i64 {
        return @intCast(self.topology.rank());
    }

    pub fn is1D(self: Mesh) bool {
        return self.rank() == 1;
    }

    pub fn is2D(self: Mesh) bool {
        return self.rank() == 2;
    }

    pub fn is3D(self: Mesh) bool {
        return self.rank() == 3;
    }

    pub fn axis(self: Mesh, ax: anytype) i64 {
        return self.topology.dim(ax);
    }

    pub fn hasManyPartitions(self: Mesh) bool {
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
            .topology = self.topology,
            .indices = indices,
        };
    }

    pub fn numPartitions(self: Mesh) u8 {
        return @intCast(self.topology.count());
    }

    pub fn numReplicas(_: Mesh) u8 {
        return 1;
    }

    pub fn numDevices(self: Mesh) u8 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn eql(self: Mesh, other: Mesh) bool {
        return self.topology.eql(other.topology);
    }

    pub fn format(
        self: Mesh,
        comptime _: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        try writer.print("Mesh(topology={} rank={d} numDevices={d})", .{ self.topology, self.rank(), self.numDevices() });
    }
};

test "Mesh / 1D mesh with 1 partition" {
    const mesh: Mesh = .init(.{ .x = 1 });
    try std.testing.expect(mesh.rank() == 1);
    try std.testing.expect(mesh.axis(.x) == 1);
    try std.testing.expect(mesh.is1D());
    try std.testing.expect(!mesh.is2D());
    try std.testing.expect(!mesh.is3D());
    try std.testing.expect(!mesh.hasManyPartitions());
    try std.testing.expect(mesh.isSinglePartition());
    try std.testing.expect(mesh.numPartitions() == 1);
    try std.testing.expect(mesh.numDevices() == 1);
}

test "Mesh / 1D mesh with 8 partitions" {
    const mesh: Mesh = .init(.{ .x = 8 });
    try std.testing.expect(mesh.rank() == 1);
    try std.testing.expect(mesh.axis(.x) == 8);
    try std.testing.expect(mesh.is1D());
    try std.testing.expect(!mesh.is2D());
    try std.testing.expect(!mesh.is3D());
    try std.testing.expect(mesh.hasManyPartitions());
    try std.testing.expect(!mesh.isSinglePartition());
    try std.testing.expect(mesh.numPartitions() == 8);
    try std.testing.expect(mesh.numDevices() == 8);
}

test "Mesh / 2D mesh with 1 partition" {
    const mesh: Mesh = .init(.{ .x = 1, .y = 1 });
    try std.testing.expect(mesh.rank() == 2);
    try std.testing.expect(mesh.axis(.x) == 1);
    try std.testing.expect(mesh.axis(.y) == 1);
    try std.testing.expect(!mesh.is1D());
    try std.testing.expect(mesh.is2D());
    try std.testing.expect(!mesh.is3D());
    try std.testing.expect(!mesh.hasManyPartitions());
    try std.testing.expect(mesh.isSinglePartition());
    try std.testing.expect(mesh.numPartitions() == 1);
    try std.testing.expect(mesh.numDevices() == 1);
}

test "Mesh / 2D mesh with 6 partitions" {
    const mesh_2d: Mesh = .init(.{ .x = 2, .y = 3 });
    try std.testing.expect(mesh_2d.rank() == 2);
    try std.testing.expect(mesh_2d.axis(.x) == 2);
    try std.testing.expect(mesh_2d.axis(.y) == 3);
    try std.testing.expect(!mesh_2d.is1D());
    try std.testing.expect(mesh_2d.is2D());
    try std.testing.expect(!mesh_2d.is3D());
    try std.testing.expect(mesh_2d.hasManyPartitions());
    try std.testing.expect(!mesh_2d.isSinglePartition());
    try std.testing.expect(mesh_2d.numPartitions() == 6);
    try std.testing.expect(mesh_2d.numDevices() == 6);

    // 3D mesh with 1 partition
    const mesh_3d_no_partitions: Mesh = .init(.{ .x = 1, .y = 1, .z = 1 });
    try std.testing.expect(mesh_3d_no_partitions.rank() == 3);
    try std.testing.expect(mesh_3d_no_partitions.axis(.x) == 1);
    try std.testing.expect(mesh_3d_no_partitions.axis(.y) == 1);
    try std.testing.expect(mesh_3d_no_partitions.axis(.z) == 1);
    try std.testing.expect(!mesh_3d_no_partitions.is1D());
    try std.testing.expect(!mesh_3d_no_partitions.is2D());
    try std.testing.expect(mesh_3d_no_partitions.is3D());
    try std.testing.expect(!mesh_3d_no_partitions.hasManyPartitions());
    try std.testing.expect(mesh_3d_no_partitions.isSinglePartition());
    try std.testing.expect(mesh_3d_no_partitions.numPartitions() == 1);
    try std.testing.expect(mesh_3d_no_partitions.numDevices() == 1);

    // 3D mesh with 64 partitions
    const mesh_3d: Mesh = .init(.{ .x = 4, .y = 4, .z = 4 });
    try std.testing.expect(mesh_3d.rank() == 3);
    try std.testing.expect(mesh_3d.axis(.x) == 4);
    try std.testing.expect(mesh_3d.axis(.y) == 4);
    try std.testing.expect(mesh_3d.axis(.z) == 4);
    try std.testing.expect(!mesh_3d.is1D());
    try std.testing.expect(!mesh_3d.is2D());
    try std.testing.expect(mesh_3d.is3D());
    try std.testing.expect(mesh_3d.hasManyPartitions());
    try std.testing.expect(!mesh_3d.isSinglePartition());
    try std.testing.expect(mesh_3d.numPartitions() == 64);
    try std.testing.expect(mesh_3d.numDevices() == 64);

    // single mesh
    const single_mesh: Mesh = .singlePartition();
    try std.testing.expect(single_mesh.rank() == 1);
    try std.testing.expect(single_mesh.axis(.x) == 1);
    try std.testing.expect(single_mesh.is1D());
    try std.testing.expect(!single_mesh.is2D());
    try std.testing.expect(!single_mesh.is3D());
    try std.testing.expect(!single_mesh.hasManyPartitions());
    try std.testing.expect(single_mesh.isSinglePartition());
    try std.testing.expect(single_mesh.numPartitions() == 1);
    try std.testing.expect(single_mesh.numDevices() == 1);

    // auto mesh with 4 devices
    const platform = env(.{ .cpu = .{ .cpu_device_count = 4 } });
    const auto_mesh: Mesh = .auto(platform);
    try std.testing.expect(auto_mesh.rank() == 1);
    try std.testing.expect(auto_mesh.axis(.x) == 4);
    try std.testing.expect(single_mesh.is1D());
    try std.testing.expect(!single_mesh.is2D());
    try std.testing.expect(!single_mesh.is3D());
    try std.testing.expect(auto_mesh.hasManyPartitions());
    try std.testing.expect(!auto_mesh.isSinglePartition());
    try std.testing.expect(auto_mesh.numPartitions() == 4);
    try std.testing.expect(auto_mesh.numDevices() == 4);
}

test "Mesh.iterator" {
    // 1D Mesh iterator
    const mesh_1d = Mesh.init(.{ .x = 4 });
    var iter_1d = mesh_1d.iterator();
    var count_1d: u8 = 0;

    while (iter_1d.next()) |indices| : (count_1d += 1) {
        const expected_shape = Shape.init(.{ .x = count_1d }, .u8);
        try std.testing.expect(indices.eqlWithTags(expected_shape));
    }
    try std.testing.expectEqual(4, count_1d);
    try std.testing.expect(iter_1d.next() == null);

    // 2D Mesh iterator
    const mesh_2d = Mesh.init(.{ .x = 2, .y = 3 });
    var iter_2d = mesh_2d.iterator();
    var count_2d: u8 = 0;
    const expected_indices_2d = [_]Shape{
        .init(.{ .x = 0, .y = 0 }, .u8),
        .init(.{ .x = 0, .y = 1 }, .u8),
        .init(.{ .x = 0, .y = 2 }, .u8),
        .init(.{ .x = 1, .y = 0 }, .u8),
        .init(.{ .x = 1, .y = 1 }, .u8),
        .init(.{ .x = 1, .y = 2 }, .u8),
    };

    while (iter_2d.next()) |indices| : (count_2d += 1) {
        try std.testing.expect(indices.eqlWithTags(expected_indices_2d[count_2d]));
    }

    try std.testing.expectEqual(6, count_2d);
    try std.testing.expect(iter_2d.next() == null);

    // 3D Mesh iterator
    const mesh_3d = Mesh.init(.{ .x = 2, .y = 2, .z = 2 });
    var iter_3d = mesh_3d.iterator();
    var count_3d: u8 = 0;
    const expected_indices_3d = [_]Shape{
        .init(.{ .x = 0, .y = 0, .z = 0 }, .u8),
        .init(.{ .x = 0, .y = 0, .z = 1 }, .u8),
        .init(.{ .x = 0, .y = 1, .z = 0 }, .u8),
        .init(.{ .x = 0, .y = 1, .z = 1 }, .u8),
        .init(.{ .x = 1, .y = 0, .z = 0 }, .u8),
        .init(.{ .x = 1, .y = 0, .z = 1 }, .u8),
        .init(.{ .x = 1, .y = 1, .z = 0 }, .u8),
        .init(.{ .x = 1, .y = 1, .z = 1 }, .u8),
    };

    while (iter_3d.next()) |indices| : (count_3d += 1) {
        try std.testing.expect(indices.eqlWithTags(expected_indices_3d[count_3d]));
    }

    try std.testing.expectEqual(8, count_3d);
    try std.testing.expect(iter_3d.next() == null);

    // Single device
    const mesh_single = Mesh.init(.{ .x = 1 });
    var iter_single = mesh_single.iterator();
    var count_single: u8 = 0;

    while (iter_single.next()) |indices| : (count_single += 1) {
        const expected_shape = Shape.init(.{ .x = 0 }, .u8);
        try std.testing.expect(indices.eqlWithTags(expected_shape));
    }
    try std.testing.expectEqual(1, count_single);
    try std.testing.expect(iter_single.next() == null);

    // Mesh with last dimension of size 1
    const mesh_mixed_one = Mesh.init(.{ .x = 3, .y = 1 });
    var iter_mixed = mesh_mixed_one.iterator();
    var count_mixed: u8 = 0;
    const expected_indices_mixed = [_]Shape{
        .init(.{ .x = 0, .y = 0 }, .u8),
        .init(.{ .x = 1, .y = 0 }, .u8),
        .init(.{ .x = 2, .y = 0 }, .u8),
    };

    while (iter_mixed.next()) |indices| : (count_mixed += 1) {
        try std.testing.expect(indices.eqlWithTags(expected_indices_mixed[count_mixed]));
    }

    try std.testing.expectEqual(3, count_mixed);
    try std.testing.expect(iter_mixed.next() == null);
}

pub const DeviceShard = struct {
    index: usize,
    topology: Shape,
    global_shape: Shape,
    indices: Shape,
    shard: Shape,
    device: *const Device,

    /// Holds the layout information for a shard within a larger host buffer.
    pub const SliceSpec = struct {
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

    /// Calculates the necessary arguments for a PJRT call to transfer this shard's data
    pub fn specs(self: DeviceShard) SliceSpec {
        const rank = self.global_shape.rank();

        // Step 1: Calculate the byte strides for the GLOBAL host buffer.
        // These strides describe how to navigate the full, unpartitioned tensor in host memory.
        // This layout is the same for all shards.
        // Example: For a global shape {m=16, k=16} of i32 (4 bytes), the strides are:
        //   - stride for 'k' (dim 1): 4 bytes
        //   - stride for 'm' (dim 0): 16 (dim k) * 4 bytes = 64 bytes
        const host_byte_strides_ba = self.global_shape.computeStrides();
        const host_byte_strides = host_byte_strides_ba.constSlice();

        // Step 2: Calculate the start offset for THIS specific shard.
        // We determine the starting coordinate of our shard's data slice within the global tensor
        // and use the global strides to find the byte offset.
        var shard_start_offset_bytes: i64 = 0;

        for (0..rank) |i| {
            // Check if the i-th dimension of the tensor is partitioned.
            const mesh_axis_tag = self.global_shape.partition(i);

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
            const mesh_axis_tag = self.global_shape.partition(i);
            if (mesh_axis_tag == Shape.TagUnknown) {
                // This dimension is REPLICATED on this shard. The slice must span the
                // entire global dimension.
                transfer_dims_buffer[i] = self.global_shape.dim(i);
            } else {
                // This dimension is PARTITIONED. The slice is just the size of the
                // smaller shard dimension.
                transfer_dims_buffer[i] = self.shard.dim(i);
            }
        }

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
        const specs_ = self.specs();
        try writer.print("DeviceShard(index={d}/{d} topology={} indices={} shard={} ({d}B) global_shape={} ({d}B) start_offset={d} byte_strides={any} device={})", .{
            self.index + 1,
            self.topology.count(),
            self.topology,
            self.indices,
            self.shard,
            self.shard.byteSize(),
            self.global_shape,
            self.global_shape.byteSize(),
            specs_.start_offset,
            specs_.byte_strides[0..specs_.num_dims],
            self.device,
        });
    }
};

pub const DeviceShardIterator = struct {
    index: usize = 0,
    platform: Platform,
    indices_iterator: TopologyIndicesIterator,
    sharding: Sharding,

    pub fn next(self: *DeviceShardIterator) ?DeviceShard {
        const num_partitions = self.sharding.mesh.numPartitions();

        if (self.index >= num_partitions) return null;

        defer self.index += 1;

        return .{
            .index = self.index,
            .topology = self.sharding.mesh.topology,
            .global_shape = self.sharding.global_shape,
            .indices = self.indices_iterator.next().?,
            .shard = self.sharding.shard(),
            .device = self.device(self.index),
        };
    }

    fn device(self: DeviceShardIterator, index: usize) *const Device {
        const devices = self.platform.getDevices();

        if (index < 0 or index >= devices.len) {
            stdx.debug.panic("DeviceShardIterator: index out of bounds: {} for devices of length {}", .{ index, devices.len });
        }

        return devices[@intCast(index)];
    }
};

pub const Sharding = struct {
    mesh: Mesh,
    global_shape: Shape,

    pub const Type = enum {
        replicated,
        maximal,
        manual,
    };

    pub fn init(mesh: Mesh, shape: Shape) Sharding {
        return .{
            .mesh = mesh,
            .global_shape = shape,
        };
    }

    pub fn getType(self: Sharding) Type {
        if (!self.global_shape.hasAtLeastOnePartitionedAxis()) {
            return .replicated;
        }

        if (self.global_shape.isFullyPartitioned()) {
            return .maximal;
        }

        return .manual;
    }

    pub fn shard(self: Sharding) Shape {
        var shard_: Shape = .init(.{}, self.global_shape.dtype());

        for (0..self.global_shape.rank()) |dim| {
            const mesh_axis = self.global_shape.partition(dim);

            if (mesh_axis == Shape.TagUnknown) {
                shard_ = shard_.appendDim(self.global_shape.dim(dim), self.global_shape.tag(dim));
            } else {
                const mesh_dim = self.mesh.topology.dim(mesh_axis);
                const d = @divExact(self.global_shape.dim(dim), mesh_dim);
                shard_ = shard_.appendDim(d, self.global_shape.tag(dim));
            }
        }

        return shard_;
    }

    pub fn iterator(self: Sharding, platform: Platform) DeviceShardIterator {
        const indices_iterator = self.mesh.iterator();

        return .{
            .platform = platform,
            .indices_iterator = indices_iterator,
            .sharding = self,
        };
    }

    /// A "packet of instructions" for reassembling a single shard.
    /// Contains all the dynamic information needed to construct the arguments for
    /// a `PJRT_Buffer_ToHostBuffer` call for one shard.
    pub const ReassemblyOp = struct {
        /// The byte offset from the start of the global host buffer where this
        /// shard's data begins. The caller adds this to the global buffer's base
        /// address to get the `dst` pointer for the PJRT call.
        start_offset_in_bytes: usize,

        /// The dimensions of this specific shard (e.g., {16, 2}).
        /// This is used to set the `dims` field in the `PJRT_Buffer_MemoryLayout`.
        shard_dims: [Shape.MAX_RANK]i64,

        /// Metadata about the shard this operation applies to. The `index` field
        /// is crucial for selecting the correct source `pjrt.Buffer`.
        shard_meta: DeviceShard,
    };

    /// An iterator that generates the necessary operations to reassemble a
    /// global tensor from its shards.
    pub const ReassemblyOpIterator = struct {
        sharding: Sharding,
        device_shard_iter: DeviceShardIterator,

        pub fn next(self: *ReassemblyOpIterator) !?ReassemblyOp {
            if (self.device_shard_iter.next()) |device_shard| {
                // This logic is identical to your pjrtArgs calculation, which is correct.
                // We are calculating the starting position of this shard's tile
                // within the larger global tensor layout.
                const global_strides_ba = self.sharding.global_shape.computeStrides();
                const global_strides = global_strides_ba.constSlice();
                var offset_in_bytes: i64 = 0;

                for (0..self.sharding.global_shape.rank()) |i| {
                    const mesh_axis_tag = self.sharding.global_shape.partition(i);
                    if (mesh_axis_tag != Shape.TagUnknown) {
                        const device_coord_on_mesh_axis = device_shard.indices.dim(mesh_axis_tag);
                        const shard_dim_size = device_shard.shard.dim(i);
                        const start_element_in_dim = device_coord_on_mesh_axis * shard_dim_size;
                        offset_in_bytes += start_element_in_dim * global_strides[i];
                    }
                }

                // The dimensions for the copy are simply the shard's dimensions.
                var dims_buffer: [Shape.MAX_RANK]i64 = .{0} ** Shape.MAX_RANK;
                const shard_dims = device_shard.shard.dims();
                @memcpy(dims_buffer[0..shard_dims.len], shard_dims);

                return ReassemblyOp{
                    .start_offset_in_bytes = @intCast(offset_in_bytes),
                    .shard_dims = dims_buffer,
                    .shard_meta = device_shard,
                };
            } else {
                return null;
            }
        }
    };

    /// Creates an iterator that yields the operations needed to reassemble the tensor.
    pub fn reassemblyOps(self: Sharding, platform: Platform) !ReassemblyOpIterator {
        return ReassemblyOpIterator{
            .sharding = self,
            .device_shard_iter = self.iterator(platform),
        };
    }

    /// Reassembles the full tensor from a slice of PJRT buffers into a host buffer.
    ///
    /// This function is the high-level entry point for retrieving sharded data.
    /// It orchestrates multiple asynchronous `PJRT_Buffer_ToHostBuffer` calls.
    ///
    /// - platform: The active platform containing the PJRT client and API pointers.
    /// - pjrt_buffers: A slice of pointers to the source `pjrt.Buffer` objects on the devices.
    ///   The order of these buffers MUST match the device order from the sharding iterator.
    /// - dest_buffer: The pre-allocated host buffer to write the result into.
    /// - allocator: Used to temporarily store PJRT_Event handles.
    pub fn reassembleFromPjrtBuffers(
        self: Sharding,
        platform: Platform,
        pjrt_buffers: []const *pjrtx.Buffer,
        dest_buffer: []u8,
        allocator: std.mem.Allocator,
    ) !void {
        stdx.debug.assert(pjrt_buffers.len == self.mesh.numPartitions(), "Expected {} PJRT buffers, got {}", .{ self.mesh.numPartitions(), pjrt_buffers.len });
        stdx.debug.assert(dest_buffer.len == self.global_shape.byteSize(), "Destination buffer size mismatch: expected {} bytes, got {}", .{ self.global_shape.byteSize(), dest_buffer.len });

        if (self.getType() == .replicated) {
            // If the data is replicated, each shard holds the full tensor.
            // We only need to copy the data from the first device.
            if (pjrt_buffers.len == 0) {
                // Nothing to do if there are no buffers.
                return;
            }

            const first_shard_buffer = pjrt_buffers[0];

            // The 'dest_buffer' is already allocated to the full global size.
            // We can copy directly into it.
            const event = try first_shard_buffer.toHostBuffer(platform.pjrt_api, dest_buffer, .{});
            if (event) |e| {
                // The pjrtx wrapper for await_ also handles deinit.
                try e.await_(platform.pjrt_api);
            }
            // The reassembly is complete.
            return;
        }

        const global_shape = self.global_shape;
        const element_size = global_shape.dtype().sizeOf(); // <-- DEFINED HERE

        // We need the GLOBAL strides in units of elements, not bytes.
        const global_element_strides = getGlobalElementStrides(global_shape);

        // Allocate a temporary buffer on the host, large enough for one shard.
        const temp_shard_buffer = try allocator.alloc(u8, self.shard().byteSize());
        defer allocator.free(temp_shard_buffer);

        // Get an iterator that provides metadata for each shard.
        var sharding_iter = self.iterator(platform);

        while (sharding_iter.next()) |device_shard| {
            const shard_index: usize = @intCast(device_shard.index);
            const src_buffer = pjrt_buffers[shard_index];
            const shard_shape = device_shard.shard;
            const shard_contiguous_slice = temp_shard_buffer[0..shard_shape.byteSize()];

            // STEP A: Device -> Temp Host Buffer (this is correct)
            const event = try src_buffer.toHostBuffer(platform.pjrt_api, shard_contiguous_slice, .{});
            if (event) |e| {
                try e.await_(platform.pjrt_api);
                // e.deinit(platform.pjrt_api);
            }

            // STEP B: Manually copy using a generic N-dimensional approach.
            // This replaces the flawed row-by-row copy.
            var inner_iter = MultiDimIterator.init(shard_shape);
            while (inner_iter.next()) |item| {
                const shard_coords = item.coords[0..item.rank];

                // Calculate destination flat index from shard_coords
                var dest_flat_index: usize = 0;
                for (shard_coords, 0..) |shard_coord_val, dim_idx| {
                    const mesh_axis_tag = self.global_shape.partition(dim_idx);
                    const global_coord_for_dim = if (mesh_axis_tag == Shape.TagUnknown)
                        shard_coord_val
                    else blk: {
                        const device_coord_on_mesh_axis = device_shard.indices.dim(mesh_axis_tag);
                        const shard_dim_size = shard_shape.dim(dim_idx);
                        break :blk (device_coord_on_mesh_axis * shard_dim_size) + shard_coord_val;
                    };
                    dest_flat_index += @as(usize, @intCast(global_coord_for_dim)) * global_element_strides[dim_idx];
                }

                const src_byte_offset = item.flat_index * element_size;
                const dest_byte_offset = dest_flat_index * element_size;

                @memcpy(dest_buffer[dest_byte_offset..][0..element_size], shard_contiguous_slice[src_byte_offset..][0..element_size]);
            }
        }
    }

    pub fn getShardingAttr(self: Sharding) []const u8 {
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
        for (0..self.global_shape.rank()) |i| {
            const mesh_axis = self.global_shape.partition(i);

            var dim: i64 = 1;

            if (mesh_axis != Shape.TagUnknown) {
                const mesh_dim = self.mesh.topology.dim(mesh_axis);
                dim = mesh_dim;
            }

            try writer.print("{d}", .{dim});
            if (i < self.global_shape.rank() - 1) try writer.writeByte(',');
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
        try writer.print("Sharding(global_shape={} mesh={})", .{ self.global_shape, self.mesh });
    }
};

test "Sharding All Cases" {
    const allocator = std.testing.allocator;
    const verbose = true; // <<< SET THIS TO `true` FOR DETAILED LOGGING!

    // Case 1: Fully Replicated on 1D Mesh
    // Tensor {8, 8} is copied in its entirety to all 4 devices.
    try testShardingCase(allocator, .init(.{ .x = 4 }), .init(.{ .m = 8, .k = 8 }, .i32), .{}, verbose);

    // Case 2: 1D Partitioning
    // Tensor {16, 16} partitioned on 'k' across an 8-device mesh.
    try testShardingCase(allocator, .init(.{ .x = 8 }), .init(.{ .m = 16, .k = 16 }, .i32), .{ .x = .k }, verbose);

    // Case 3: 2D Partitioning of 2D Tensor
    // Tensor {8, 12} partitioned on 'm' by 'x' and 'k' by 'y'.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 4 }), .init(.{ .m = 8, .k = 12 }, .i32), .{ .x = .m, .y = .k }, verbose);

    // Case 4: Partial Partitioning (Mixed Replicated/Partitioned)
    // 3D Tensor {4, 8, 6}. 'b' is replicated. 'h' and 'w' are partitioned.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .b = 4, .h = 8, .w = 6 }, .i32), .{ .x = .h, .y = .w }, verbose);

    // Case 5: Fully Partitioned 3D Tensor
    // The complex case from your previous successful test.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 4, .z = 3 }), .init(.{ .m = 4, .k = 8, .j = 6 }, .i32), .{ .x = .m, .y = .k, .z = .j }, verbose);

    // --- NEW TEST CASES ---

    // Case 6: 2D Tensor on 1D Mesh (Unused Mesh Axis)
    // Partition 'k' using mesh axis 'x'. Mesh axis 'y' is unused by this sharding spec.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 3 }), .init(.{ .m = 10, .k = 6 }, .i32), .{ .x = .k }, verbose);

    // Case 7: Replicated on a 2D Mesh
    // A sanity check. The logic should be identical to Case 1, but on a more complex mesh.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .m = 4, .k = 4 }, .i32), .{}, verbose);

    // Case 8: Non-Contiguous Partitioning
    // Partitioning the first and third dimensions of a 3D tensor, while the middle one is replicated.
    // This is a great test for the striding and offset logic.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .batch = 4, .seq_len = 10, .features = 8 }, .i32), .{ .x = .batch, .y = .features }, verbose);

    // Case 9: Single Device "Partitioning"
    // Partitioning a dimension by a mesh axis of size 1 should be a no-op.
    try testShardingCase(allocator, .init(.{ .x = 1, .y = 4 }), .init(.{ .m = 8, .k = 8 }, .i32), .{ .x = .m, .y = .k }, verbose);
}

fn testShardingCase(
    allocator: std.mem.Allocator,
    mesh: Mesh,
    shape: Shape,
    partition_spec: anytype,
    verbose: bool,
) !void {
    _platform = null; // Reset the global platform
    const platform = env(.{ .cpu = .{ .cpu_device_count = mesh.numDevices() } });

    // 1. Setup the sharding
    const shape_partitioned = shape.withPartitionning(partition_spec);
    const sharding = Sharding.init(mesh, shape_partitioned);
    std.debug.print("\n--- Testing Case (verbose={any}) --- {}\n", .{ verbose, sharding });

    // 2. Create original host data
    const demo_slice = try slice.arange(allocator, shape_partitioned, .{});
    defer allocator.free(demo_slice);
    if (verbose) {
        std.debug.print("Original full slice of data: {any}\n\n", .{Shaped(i32, shape_partitioned, demo_slice)});
    }

    // 3. Simulate sending to devices and collecting shards
    var shards_on_device = std.ArrayList(ShardOnDevice).init(allocator);
    defer {
        for (shards_on_device.items) |s| s.deinit();
        shards_on_device.deinit();
    }

    var device_shards = std.ArrayList(DeviceShard).init(allocator);
    defer device_shards.deinit();

    var iter = sharding.iterator(platform);
    while (iter.next()) |device_shard| {
        const shard_on_device = try transferAndFetch(platform, device_shard, demo_slice);
        try shards_on_device.append(shard_on_device);
        try device_shards.append(device_shard);

        if (verbose) {
            std.debug.print("{}\n", .{device_shard});
            std.debug.print("{any} - shard data slice: {any}\n\n", .{ shard_on_device, Shaped(i32, shard_on_device.shape(), shard_on_device.data) });
        }
    }

    // 4. Reassemble the data
    if (verbose) {
        std.debug.print("--- Reassembling from shards ---\n", .{});
    }
    const reassembled_slice = try reassembleFromShards(
        allocator,
        sharding,
        shards_on_device.items,
        device_shards.items,
    );
    defer allocator.free(reassembled_slice);

    if (verbose) {
        std.debug.print("Reassembled full slice of data: {any}\n\n", .{Shaped(i32, sharding.global_shape, reassembled_slice)});
    }

    // 5. Verify correctness
    try std.testing.expectEqualSlices(u8, demo_slice, reassembled_slice);
    std.debug.print("✅ Verification successful for: {}\n\n", .{sharding});
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
        return .init(self.dims, dtypeFromBufferType(self.buffer_type));
    }

    pub fn deinit(self: ShardOnDevice) void {
        std.testing.allocator.free(self.data);
    }
};

pub fn transferAndFetch(platform: Platform, shard: DeviceShard, data: []u8) !ShardOnDevice {
    const specs = shard.specs();
    const args = pjrtx.Client.BufferFromHostBufferArgs{
        .data = data[specs.start_offset..].ptr,
        .buffer_type = bufferTypeFromDtype(shard.shard.dtype()),
        .dims = specs.dims[0..specs.num_dims], // Slice the fixed array
        .byte_strides = specs.byte_strides[0..specs.num_dims], // Slice the fixed array
        .host_buffer_semantics = .ImmutableUntilTransferCompletes,
        .device = shard.device,
    };

    const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);

    if (event) |ev| {
        try ev.await_(platform.pjrt_api);
    }

    const dims = pjrt_buffer.getDimensions(platform.pjrt_api);
    const buffer_type = pjrt_buffer.getElementType(platform.pjrt_api);
    const size = try pjrt_buffer.getOnDeviceSizeInBytes(platform.pjrt_api);

    const data_buffer = try std.testing.allocator.alloc(u8, size);
    const to_host_event = try pjrt_buffer.toHostBuffer(platform.pjrt_api, data_buffer, .{});

    if (to_host_event) |ev| {
        try ev.await_(platform.pjrt_api);
    }

    return .{
        .dims = dims,
        .buffer_type = buffer_type,
        .size = size,
        .data = data_buffer,
    };
}

/// Reassembles a full global tensor on the host from a collection of device shards.
///
/// This version is hardened to ensure the rank of each shard matches the global rank,
/// which is a requirement for this reassembly logic.
///
/// - allocator: The memory allocator to use for the new global buffer.
/// - sharding: The Sharding object describing the global layout and mesh.
/// - shards_on_device: A slice of ShardOnDevice objects, containing the actual data from each device.
/// - device_shards: A slice of DeviceShard metadata, corresponding to each ShardOnDevice.
///
/// Returns a new `[]u8` slice containing the reassembled data, owned by the caller.
pub fn reassembleFromShards(
    allocator: std.mem.Allocator,
    sharding: Sharding,
    shards_on_device: []const ShardOnDevice,
    device_shards: []const DeviceShard,
) ![]u8 {
    stdx.debug.assert(shards_on_device.len == device_shards.len and shards_on_device.len == sharding.mesh.numPartitions(), "Mismatch in number of shards: expected {d}, got {d} and {d}.", .{
        sharding.mesh.numPartitions(),
        shards_on_device.len,
        device_shards.len,
    });

    const global_shape = sharding.global_shape;
    const global_rank = global_shape.rank();
    const element_size = global_shape.dtype().sizeOf();

    const global_buffer = try allocator.alloc(u8, global_shape.byteSize());
    errdefer allocator.free(global_buffer);

    // Calculate the strides of the global tensor in *elements*.
    var global_element_strides: [Shape.MAX_RANK]usize = undefined;
    if (global_rank > 0) {
        global_element_strides[global_rank - 1] = 1;
        var i: usize = global_rank - 1;
        while (i > 0) {
            i -= 1;
            global_element_strides[i] = global_element_strides[i + 1] * @as(usize, @intCast(global_shape.dim(i + 1)));
        }
    }

    // Iterate over each shard and "paint" its data onto the global buffer.
    for (shards_on_device, device_shards) |shard_on_device, device_shard| {
        const shard_shape = device_shard.shard;

        // --- HARDENED CHECK ---
        // Add a clear panic if the fundamental assumption of this function is violated.
        if (shard_shape.rank() != global_rank) {
            stdx.debug.panic(
                "Rank mismatch: Cannot reassemble shard with shape {} (rank={d}) into global shape {} (rank={d}). " ++
                    "This can happen if sharding involves splitting axes, which this function does not support.",
                .{ shard_shape, shard_shape.rank(), global_shape, global_rank },
            );
        }

        var shard_iterator = MultiDimIterator.init(shard_shape);
        while (shard_iterator.next()) |item| {
            const shard_coords = item.coords[0..item.rank];
            const src_flat_index = item.flat_index;

            var dest_flat_index: usize = 0;
            for (shard_coords, 0..) |shard_coord_val, dim_idx| {
                var global_coord_for_dim: i64 = 0;

                const mesh_axis_tag = global_shape.partition(dim_idx);

                if (mesh_axis_tag == Shape.TagUnknown) {
                    global_coord_for_dim = shard_coord_val;
                } else {
                    const device_coord_on_mesh_axis = device_shard.indices.dim(mesh_axis_tag);
                    const shard_dim_size = shard_shape.dim(dim_idx);
                    global_coord_for_dim = (device_coord_on_mesh_axis * shard_dim_size) + shard_coord_val;
                }

                dest_flat_index += @as(usize, @intCast(global_coord_for_dim)) * global_element_strides[dim_idx];
            }

            const src_byte_offset = src_flat_index * element_size;
            const dest_byte_offset = dest_flat_index * element_size;

            @memcpy(global_buffer[dest_byte_offset .. dest_byte_offset + element_size], shard_on_device.data[src_byte_offset .. src_byte_offset + element_size]);
        }
    }

    return global_buffer;
}

/// Iterates over all coordinates of a shape, yielding the multi-dimensional
/// coordinates and the corresponding flat index for each element.
const MultiDimIterator = struct {
    global_shape: Shape,
    coords: [Shape.MAX_RANK]i64,
    flat_index: usize,
    is_done: bool,

    pub const NextItem = struct {
        coords: [Shape.MAX_RANK]i64,
        rank: u4,
        flat_index: usize,
    };

    pub fn init(shape: Shape) MultiDimIterator {
        const initial_coords: [Shape.MAX_RANK]i64 = .{0} ** Shape.MAX_RANK;
        return .{
            .global_shape = shape,
            .coords = initial_coords,
            .flat_index = 0,
            .is_done = (shape.count() == 0),
        };
    }

    pub fn next(self: *MultiDimIterator) ?NextItem {
        if (self.is_done) {
            return null;
        }

        const rank = self.global_shape.rank();

        // 1. Create the item to return. The assignment from one array (`self.coords`)
        // to another (`item_to_return.coords`) performs a full value copy.
        const item_to_return = NextItem{
            .coords = self.coords, // This is now a value copy.
            .rank = rank,
            .flat_index = self.flat_index,
        };

        // 2. Now, we can safely advance the iterator's internal state.
        self.flat_index += 1;
        var i: usize = rank;
        while (i > 0) {
            i -= 1;
            self.coords[i] += 1;

            if (self.coords[i] < self.global_shape.dim(i)) {
                // The state is advanced. Return the pristine copy we made earlier.
                return item_to_return;
            }
            self.coords[i] = 0;
        }

        self.is_done = true;
        return item_to_return;
    }
};

test MultiDimIterator {
    const shape_2d = Shape.init(.{ .rows = 2, .cols = 3 }, .f32);
    var iter_2d = MultiDimIterator.init(shape_2d);

    var item = iter_2d.next().?;
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, item.coords[0..item.rank]);
    try std.testing.expectEqual(0, item.flat_index);

    item = iter_2d.next().?;
    try std.testing.expectEqualSlices(i64, &.{ 0, 1 }, item.coords[0..item.rank]);
    try std.testing.expectEqual(1, item.flat_index);

    item = iter_2d.next().?;
    try std.testing.expectEqualSlices(i64, &.{ 0, 2 }, item.coords[0..item.rank]);
    try std.testing.expectEqual(2, item.flat_index);

    item = iter_2d.next().?;
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, item.coords[0..item.rank]);
    try std.testing.expectEqual(3, item.flat_index);

    item = iter_2d.next().?;
    try std.testing.expectEqualSlices(i64, &.{ 1, 1 }, item.coords[0..item.rank]);
    try std.testing.expectEqual(4, item.flat_index);

    item = iter_2d.next().?;
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, item.coords[0..item.rank]);
    try std.testing.expectEqual(5, item.flat_index);

    try std.testing.expect(iter_2d.next() == null);
}

// todo: temp
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

/// Calculates the strides of a shape in units of elements.
/// For a shape {d0, d1, d2}, the strides would be {d1*d2, d2, 1}.
/// This is used to convert multi-dimensional coordinates to a flat 1D index.
fn getGlobalElementStrides(shape: Shape) [Shape.MAX_RANK]usize {
    const rank = shape.rank();
    var strides: [Shape.MAX_RANK]usize = .{0} ** Shape.MAX_RANK;

    if (rank == 0) {
        // A scalar has no strides, but returning {1} can be useful sometimes.
        // Let's stick to {0} for safety.
        return strides;
    }

    // The stride for the innermost (last) dimension is always 1 element.
    strides[rank - 1] = 1;

    // Work backwards from the second-to-last dimension.
    var i: usize = rank - 1;
    while (i > 0) {
        i -= 1;
        // The stride for dimension `i` is the product of all dimensions after it.
        // This is equivalent to `stride[i+1] * shape.dim(i+1)`.
        strides[i] = strides[i + 1] * @as(usize, @intCast(shape.dim(i + 1)));
    }

    return strides;
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
