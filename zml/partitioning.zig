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

/// Iterates through all possible multi-dimensional indices of a topology.
/// The iteration order is row-major.
pub const TopologyIndicesIterator = struct {
    index: usize = 0,
    topology: Shape,
    indices: Shape,

    pub fn next(self: *TopologyIndicesIterator) ?Shape {
        if (self.index >= self.topology.count()) return null;

        const current_indices = self.indices;

        // Prepare for the next iteration
        self.index += 1;

        // Use a standard row-major increment logic
        if (self.index < self.topology.count()) {
            const rank = self.topology.rank();
            var i: usize = rank;
            while (i > 0) {
                i -= 1;
                const dim: u4 = @intCast(i);

                const current_value = self.indices.dim(dim);
                if (current_value < self.topology.dim(dim) - 1) {
                    self.indices = self.indices.setDim(dim, current_value + 1);
                    // Found the dimension to increment, we are done.
                    return current_indices;
                } else {
                    // This dimension wrapped around, reset to 0 and carry over to the next.
                    self.indices = self.indices.setDim(dim, 0);
                }
            }
        }

        return current_indices;
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

    pub fn single() Mesh {
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

    pub fn reshape(self: Mesh, new_shape: anytype) Mesh {
        const new_topology = self.topology.reshape(new_shape);
        return .init(new_topology);
    }

    pub fn flatten(self: Mesh, new_axis_name: anytype) Mesh {
        const total_devices = self.numDevices();

        var new_topology = Shape.init(.{}, .u8);
        new_topology = new_topology.appendDim(total_devices, @tagName(new_axis_name));
        return Mesh.init(new_topology);
    }

    test flatten {
        {
            // Flatten a 2D mesh
            const mesh = Mesh.init(.{ .x = 2, .y = 4 });
            const flattened = mesh.flatten(.all);
            const expected = Mesh.init(.{ .all = 8 });

            try std.testing.expect(flattened.eql(expected));
            try std.testing.expectEqual(1, flattened.rank());
            try std.testing.expectEqual(8, flattened.numDevices());
        }

        {
            // Flatten a 3D mesh
            const mesh = Mesh.init(.{ .data = 2, .model = 3, .pipeline = 4 });
            const flattened = mesh.flatten(.devices);
            const expected = Mesh.init(.{ .devices = 24 });
            try std.testing.expect(flattened.eql(expected));
            try std.testing.expectEqual(1, flattened.rank());
            try std.testing.expectEqual(24, flattened.numDevices());
        }
        {
            // Flattening a 1D mesh should just rename the axis
            const mesh = Mesh.init(.{ .x = 16 });
            const flattened = mesh.flatten(.y);
            const expected = Mesh.init(.{ .y = 16 });
            try std.testing.expect(flattened.eql(expected));
            try std.testing.expectEqual(1, expected.rank());
        }
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
    const mesh: Mesh = .init(.{ .x = 2, .y = 3 });
    try std.testing.expect(mesh.rank() == 2);
    try std.testing.expect(mesh.axis(.x) == 2);
    try std.testing.expect(mesh.axis(.y) == 3);
    try std.testing.expect(!mesh.is1D());
    try std.testing.expect(mesh.is2D());
    try std.testing.expect(!mesh.is3D());
    try std.testing.expect(mesh.hasManyPartitions());
    try std.testing.expect(!mesh.isSinglePartition());
    try std.testing.expect(mesh.numPartitions() == 6);
    try std.testing.expect(mesh.numDevices() == 6);
}

test "Mesh / 3D mesh with 1 partition" {
    const mesh: Mesh = .init(.{ .x = 1, .y = 1, .z = 1 });
    try std.testing.expect(mesh.rank() == 3);
    try std.testing.expect(mesh.axis(.x) == 1);
    try std.testing.expect(mesh.axis(.y) == 1);
    try std.testing.expect(mesh.axis(.z) == 1);
    try std.testing.expect(!mesh.is1D());
    try std.testing.expect(!mesh.is2D());
    try std.testing.expect(mesh.is3D());
    try std.testing.expect(!mesh.hasManyPartitions());
    try std.testing.expect(mesh.isSinglePartition());
    try std.testing.expect(mesh.numPartitions() == 1);
    try std.testing.expect(mesh.numDevices() == 1);
}

test "Mesh / 3D mesh with 64 partitions" {
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
}

test "Mesh / single" {
    const mesh: Mesh = .single();
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

test "Mesh / auto with 4 devices" {
    const platform = env(.{ .cpu = .{ .cpu_device_count = 4 } });
    const mesh: Mesh = .auto(platform);
    try std.testing.expect(mesh.rank() == 1);
    try std.testing.expect(mesh.axis(.x) == 4);
    try std.testing.expect(mesh.is1D());
    try std.testing.expect(!mesh.is2D());
    try std.testing.expect(!mesh.is3D());
    try std.testing.expect(mesh.hasManyPartitions());
    try std.testing.expect(!mesh.isSinglePartition());
    try std.testing.expect(mesh.numPartitions() == 4);
    try std.testing.expect(mesh.numDevices() == 4);
}

test "Mesh / 1D mesh iterator" {
    const mesh = Mesh.init(.{ .x = 4 });
    var iter = mesh.iterator();
    var count: u8 = 0;

    while (iter.next()) |indices| : (count += 1) {
        const expected_shape = Shape.init(.{ .x = count }, .u8);
        try std.testing.expect(indices.eqlWithTags(expected_shape));
    }

    try std.testing.expectEqual(4, count);
    try std.testing.expect(iter.next() == null);
}

test "Mesh / 2D mesh iterator" {
    const mesh = Mesh.init(.{ .x = 2, .y = 3 });
    var iter = mesh.iterator();
    var count: u8 = 0;
    const expected_indices = [_]Shape{
        .init(.{ .x = 0, .y = 0 }, .u8),
        .init(.{ .x = 0, .y = 1 }, .u8),
        .init(.{ .x = 0, .y = 2 }, .u8),
        .init(.{ .x = 1, .y = 0 }, .u8),
        .init(.{ .x = 1, .y = 1 }, .u8),
        .init(.{ .x = 1, .y = 2 }, .u8),
    };

    while (iter.next()) |indices| : (count += 1) {
        try std.testing.expect(indices.eqlWithTags(expected_indices[count]));
    }
    try std.testing.expectEqual(6, count);
    try std.testing.expect(iter.next() == null);
}

test "Mesh / 3D mesh iterator" {
    const mesh = Mesh.init(.{ .x = 2, .y = 2, .z = 2 });
    var iter = mesh.iterator();
    var count: u8 = 0;
    const expected_indices = [_]Shape{
        .init(.{ .x = 0, .y = 0, .z = 0 }, .u8),
        .init(.{ .x = 0, .y = 0, .z = 1 }, .u8),
        .init(.{ .x = 0, .y = 1, .z = 0 }, .u8),
        .init(.{ .x = 0, .y = 1, .z = 1 }, .u8),
        .init(.{ .x = 1, .y = 0, .z = 0 }, .u8),
        .init(.{ .x = 1, .y = 0, .z = 1 }, .u8),
        .init(.{ .x = 1, .y = 1, .z = 0 }, .u8),
        .init(.{ .x = 1, .y = 1, .z = 1 }, .u8),
    };

    while (iter.next()) |indices| : (count += 1) {
        try std.testing.expect(indices.eqlWithTags(expected_indices[count]));
    }

    try std.testing.expectEqual(8, count);
    try std.testing.expect(iter.next() == null);
}

pub const DeviceShard = struct {
    index: u64,
    topology: Shape,
    global_shape: Shape,
    indices: Shape,
    shard_shape: Shape,
    device: *const Device,

    /// Holds the layout information for a shard within a larger host buffer.
    pub const SliceSpec = struct {
        /// The byte offset from the start of the host buffer to this shard's data.
        start_offset_bytes: usize,

        /// The byte strides of the *global* tensor layout on the host.
        /// This is a fixed-size array; the caller should slice it using `num_dims`
        /// to get the `*const i64` and `size_t` for the PJRT C API.
        host_byte_strides: [Shape.MAX_RANK]i64,

        /// The dimensions of the data slice to be transferred from the host.
        /// This describes the "view" into the host buffer.
        dims_on_host: [Shape.MAX_RANK]i64,

        /// The number of valid dimensions (and strides) for this layout.
        num_dims: u4,
    };

    /// Calculates the necessary arguments for a PJRT call to transfer this shard's data
    pub fn specs(self: DeviceShard) SliceSpec {
        const rank = self.global_shape.rank();

        // Step 1: Calculate the byte strides for the GLOBAL host buffer.
        // These strides describe how to navigate the full, unpartitioned tensor in host memory.
        const host_byte_strides_ba = self.global_shape.computeByteStrides();
        const host_byte_strides = host_byte_strides_ba.constSlice();

        // Step 2: Calculate the start offset for THIS specific shard.
        // We determine the starting coordinate of our shard's data slice within the global tensor
        // and use the global strides to find the byte offset.
        var shard_start_offset_bytes: i64 = 0;
        for (0..rank) |i| {
            const part_spec = self.global_shape.partition(i);
            if (part_spec == .axis) {
                const mesh_axis_tag = part_spec.toTag();
                const device_coord_on_mesh_axis = self.indices.dim(mesh_axis_tag);
                const shard_dim_size = self.shard_shape.dim(i);
                const start_element_in_dim = device_coord_on_mesh_axis * shard_dim_size;
                shard_start_offset_bytes += start_element_in_dim * host_byte_strides[i];
            }
        }

        // Step 3: Determine the dimensions of the SLICE to transfer.
        // This is what PJRT will actually read from the host buffer.
        // For a partitioned axis, we read a slice the size of the shard.
        // For a replicated axis, the shard on the device contains the full dimension,
        // so we must read the full dimension from the host.
        var transfer_dims_buffer: [Shape.MAX_RANK]i64 = undefined;
        for (0..rank) |i| {
            transfer_dims_buffer[i] = self.shard_shape.dim(i);
        }

        return .{
            .start_offset_bytes = @intCast(shard_start_offset_bytes),
            .host_byte_strides = host_byte_strides_ba.buffer,
            .dims_on_host = transfer_dims_buffer,
            .num_dims = rank,
        };
    }

    pub fn format(
        self: DeviceShard,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const specs_ = self.specs();
        try writer.print("DeviceShard(index={d}/{d} topology={} indices={} shard_shape={} ({d}B) global_shape={} ({d}B) start_offset={d} byte_strides={any} device={})", .{
            self.index + 1,
            self.topology.count(),
            self.topology,
            self.indices,
            self.shard_shape,
            self.shard_shape.byteSize(),
            self.global_shape,
            self.global_shape.byteSize(),
            specs_.start_offset_bytes,
            specs_.host_byte_strides[0..specs_.num_dims],
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
        const num_partitions = self.sharding.mesh.numPartitions(); // todo proxy numPartitions

        if (self.index >= num_partitions) return null;

        defer self.index += 1;

        return .{
            .index = self.index,
            .topology = self.sharding.mesh.topology,
            .global_shape = self.sharding.global_shape,
            .indices = self.indices_iterator.next().?,
            .shard_shape = self.sharding.shardShape(),
            .device = self.device(self.index),
        };
    }

    fn device(self: DeviceShardIterator, index: usize) *const Device {
        const devices = self.platform.getDevices();

        if (index >= devices.len) {
            stdx.debug.panic("DeviceShardIterator: index out of bounds: {} for devices of length {}", .{ index, devices.len });
        }

        return devices[index];
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
        // Validate that the shape is divisible by the mesh where partitioned.
        for (0..shape.rank()) |dim_idx| {
            const part_spec = shape.partition(dim_idx);
            if (part_spec == .axis) {
                const mesh_axis_tag = part_spec.toTag();
                const global_dim = shape.dim(dim_idx);
                const mesh_dim = mesh.topology.dim(mesh_axis_tag);
                if (@rem(global_dim, mesh_dim) != 0) {
                    stdx.debug.panic(
                        "Tensor dimension {s} (size={d}) is not divisible by mesh axis {s} (size={d})",
                        .{ shape.debugTag(dim_idx), global_dim, std.mem.span(mesh_axis_tag), mesh_dim },
                    );
                }
            }
        }
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

    pub fn shardShape(self: Sharding) Shape {
        var shard_: Shape = .init(.{}, self.global_shape.dtype());
        const rank = self.global_shape.rank();

        for (0..rank) |dim| {
            const part_spec = self.global_shape.partition(dim);
            const global_dim_size = self.global_shape.dim(dim);
            const tag = self.global_shape.tag(dim);

            if (part_spec == .axis) {
                const mesh_axis_tag = part_spec.toTag();
                const mesh_dim = self.mesh.topology.dim(mesh_axis_tag);
                const shard_dim_size = @divExact(global_dim_size, mesh_dim);
                shard_ = shard_.appendDim(shard_dim_size, tag);
            } else {
                // For replicated, open, or unknown, the shard dimension is the full global dimension.
                shard_ = shard_.appendDim(global_dim_size, tag);
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

    pub fn getShardingAttr(self: Sharding) std.BoundedArray(u8, 128) {
        var sharding_str: std.BoundedArray(u8, 128) = .{};
        self.writeShardingRepresentation(sharding_str.writer()) catch unreachable;
        return sharding_str;
    }

    pub fn writeShardingRepresentation(self: Sharding, writer: anytype) !void {
        if (self.getType() == .replicated) {
            try writer.writeAll("{replicated}");
            return;
        }

        try writer.writeAll("{devices=[");
        const rank = self.global_shape.rank();
        for (0..rank) |i| {
            const part_spec = self.global_shape.partition(i);

            var dim: i64 = 1; // Default to 1 for replicated dimensions.

            if (part_spec == .axis) {
                const mesh_axis_tag = part_spec.toTag();
                dim = self.mesh.topology.dim(mesh_axis_tag);
            }

            try writer.print("{d}", .{dim});
            if (i < rank - 1) {
                try writer.writeByte(',');
            }
        }
        try writer.print("]<=[{d}]}}", .{self.mesh.numPartitions()});
    }

    test "Sharding MLIR representation" {
        // Case 1: Fully Replicated
        const mesh_rep = Mesh.init(.{ .x = 4 });
        const shape_rep = Shape.init(.{ .m = 8, .k = 8 }, .i32);
        const sharding_rep = Sharding.init(mesh_rep, shape_rep);
        var attr_rep_ba = sharding_rep.getShardingAttr();
        try std.testing.expectEqualStrings("{replicated}", attr_rep_ba.constSlice());

        // Case 2: Partially partitioned
        const mesh_part = Mesh.init(.{ .x = 2, .y = 4 });
        const shape_part = Shape.init(.{ .m = 8, .k = 12 }, .i32).withPartitioning(.{ .m = .x });
        const sharding_part = Sharding.init(mesh_part, shape_part);
        var attr_part_ba = sharding_part.getShardingAttr();
        // dim 'm' is partitioned on 'x' (size 2), dim 'k' is replicated (size 1)
        try std.testing.expectEqualStrings("{devices=[2,1]<=[8]}", attr_part_ba.constSlice());

        // Case 3: Fully partitioned
        const mesh_full = Mesh.init(.{ .x = 2, .y = 4 });
        const shape_full = Shape.init(.{ .m = 8, .k = 12 }, .i32).withPartitioning(.{ .m = .x, .k = .y });
        const sharding_full = Sharding.init(mesh_full, shape_full);
        var attr_full_ba = sharding_full.getShardingAttr();
        // dim 'm' partitioned on 'x' (size 2), dim 'k' partitioned on 'y' (size 4)
        try std.testing.expectEqualStrings("{devices=[2,4]<=[8]}", attr_full_ba.constSlice());

        // Case 4: More complex 3D case with a replicated dimension in the middle
        const mesh_3d = Mesh.init(.{ .x = 2, .y = 2, .z = 3 });
        const shape_3d = Shape.init(.{ .b = 4, .h = 8, .w = 6 }, .i32).withPartitioning(.{ .b = .x, .w = .z });
        const sharding_3d = Sharding.init(mesh_3d, shape_3d);
        var attr_3d_ba = sharding_3d.getShardingAttr();
        // b -> x (size 2), h -> replicated (size 1), w -> z (size 3)
        try std.testing.expectEqualStrings("{devices=[2,1,3]<=[12]}", attr_3d_ba.constSlice());
    }

    /// Reassembles the full tensor from a slice of PJRT buffers into a host buffer.
    /// This is the "production" implementation. It performs a D2H copy for each shard
    /// into a temporary buffer, then copies that into the final destination.
    /// Reassembles the full tensor from a slice of PJRT buffers into a host buffer.
    pub fn reassembleFromPjrtBuffers(self: Sharding, platform: Platform, pjrt_buffers: []const *pjrtx.Buffer, allocator: std.mem.Allocator) ![]u8 {
        stdx.debug.assert(pjrt_buffers.len == self.mesh.numPartitions(), "Expected {} PJRT buffers, got {}", .{ self.mesh.numPartitions(), pjrt_buffers.len });

        const global_buffer = try allocator.alloc(u8, self.global_shape.byteSize());
        errdefer allocator.free(global_buffer);

        if (self.getType() == .replicated) {
            if (pjrt_buffers.len > 0) {
                const event = try pjrt_buffers[0].toHostBuffer(platform.pjrt_api, global_buffer, .{});
                if (event) |e| try e.await_(platform.pjrt_api);
            }
            return global_buffer;
        }

        const shard_shape = self.shardShape();
        const temp_shard_buffer = try allocator.alloc(u8, shard_shape.byteSize());
        defer allocator.free(temp_shard_buffer);

        const element_size = self.global_shape.dtype().sizeOf();
        const global_element_strides = self.global_shape.computeElementStrides();

        var sharding_iter = self.iterator(platform);
        while (sharding_iter.next()) |device_shard| {
            const shard_index: usize = @intCast(device_shard.index);
            const src_buffer = pjrt_buffers[shard_index];

            // Step 1: D2H copy of the contiguous shard to a temporary buffer.
            const event = try src_buffer.toHostBuffer(platform.pjrt_api, temp_shard_buffer, .{});
            if (event) |e| try e.await_(platform.pjrt_api);

            // Step 2: Manually copy from the temp buffer to the final strided buffer.
            var iter = shard_shape.iterator();
            while (iter.next()) |item| {
                const src_byte_offset = item.flat_index * element_size;

                var dest_flat_index: i64 = 0;
                for (0..self.global_shape.rank()) |dim_idx| {
                    const global_coord_for_dim = blk: {
                        const part_spec = self.global_shape.partition(dim_idx);
                        if (part_spec != .axis) break :blk item.coords[dim_idx];
                        const mesh_axis_tag = part_spec.toTag();
                        const device_coord = device_shard.indices.dim(mesh_axis_tag);
                        break :blk (device_coord * shard_shape.dim(dim_idx)) + item.coords[dim_idx];
                    };
                    dest_flat_index += global_coord_for_dim * global_element_strides.get(dim_idx);
                }
                const dest_byte_offset = @as(usize, @intCast(dest_flat_index)) * element_size;

                @memcpy(global_buffer[dest_byte_offset..][0..element_size], temp_shard_buffer[src_byte_offset..][0..element_size]);
            }
        }

        return global_buffer;
    }

    pub fn format(
        self: Sharding,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Sharding(global_shape={} mesh={})", .{ self.global_shape, self.mesh });
    }
};

fn testShardingCase(allocator: std.mem.Allocator, mesh: Mesh, shape: Shape, partition_spec: anytype, verbose: bool) !void {
    _platform = null; // Reset the global platform
    const platform = env(.{ .cpu = .{ .cpu_device_count = @intCast(mesh.numDevices()) } });

    const shape_partitioned = shape.withPartitioning(partition_spec);
    const sharding = Sharding.init(mesh, shape_partitioned);
    std.debug.print("\n--- Testing Case (verbose={any}) --- {}\n", .{ verbose, sharding });

    const original_data = try slice.arange(allocator, shape_partitioned, .{});
    defer allocator.free(original_data);
    if (verbose) {
        std.debug.print("Original full slice of data: {any}\n\n", .{Shaped(i32, shape_partitioned, original_data)});
    }

    // --- SCATTER (H2D) ---
    var pjrt_buffers = std.ArrayList(*pjrtx.Buffer).init(allocator);
    defer {
        for (pjrt_buffers.items) |b| b.deinit(platform.pjrt_api);
        pjrt_buffers.deinit();
    }

    var sharding_iter = sharding.iterator(platform);
    while (sharding_iter.next()) |device_shard| {
        if (verbose) std.debug.print("{}\n", .{device_shard});
        const specs = device_shard.specs();
        const args = pjrtx.Client.BufferFromHostBufferArgs{
            .data = original_data[specs.start_offset_bytes..].ptr,
            .buffer_type = bufferTypeFromDtype(device_shard.shard_shape.dtype()),
            .dims = specs.dims_on_host[0..specs.num_dims],
            .byte_strides = specs.host_byte_strides[0..specs.num_dims],
            .host_buffer_semantics = .ImmutableUntilTransferCompletes,
            .device = device_shard.device,
        };
        const pjrt_buffer, const event = try platform.pjrt_client.bufferFromHostBuffer(platform.pjrt_api, args);
        if (event) |e| try e.await_(platform.pjrt_api);
        try pjrt_buffers.append(pjrt_buffer);

        // Optional: verify the transferred data for debugging
        if (verbose) {
            const shard_shape = device_shard.shard_shape;
            const fetched_data = try allocator.alloc(u8, shard_shape.byteSize());
            defer allocator.free(fetched_data);
            const fetch_event = try pjrt_buffer.toHostBuffer(platform.pjrt_api, fetched_data, .{});
            if (fetch_event) |e| try e.await_(platform.pjrt_api);
            std.debug.print("Fetched shard data: {any}\n\n", .{Shaped(i32, shard_shape, fetched_data)});
        }
    }

    // --- GATHER (D2H) ---
    if (verbose) std.debug.print("--- Reassembling from shards ---\n", .{});
    const reassembled_data = try sharding.reassembleFromPjrtBuffers(platform, pjrt_buffers.items, allocator);
    defer allocator.free(reassembled_data);

    if (verbose) std.debug.print("Reassembled full slice of data: {any}\n\n", .{Shaped(i32, sharding.global_shape, reassembled_data)});

    // --- VERIFY ---
    try std.testing.expectEqualSlices(u8, original_data, reassembled_data);
    std.debug.print("✅ Verification successful for: {}\n\n", .{sharding});
}

test "Sharding All Cases" {
    const allocator = std.testing.allocator;
    const verbose = true;

    try testShardingCase(allocator, .init(.{ .x = 4 }), .init(.{ .m = 8, .k = 8 }, .i32), .{}, verbose);
    try testShardingCase(allocator, .init(.{ .x = 8 }), .init(.{ .m = 16, .k = 16 }, .i32), .{ .k = .x }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 4 }), .init(.{ .m = 8, .k = 12 }, .i32), .{ .m = .x, .k = .y }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .b = 4, .h = 8, .w = 6 }, .i32), .{ .h = .x, .w = .y }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 4, .z = 3 }), .init(.{ .m = 4, .k = 8, .j = 6 }, .i32), .{ .m = .x, .k = .y, .j = .z }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 3 }), .init(.{ .m = 10, .k = 6 }, .i32), .{ .k = .x }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .m = 4, .k = 4 }, .i32), .{}, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .batch = 4, .seq_len = 10, .features = 8 }, .i32), .{ .batch = .x, .features = .y }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 1, .y = 4 }), .init(.{ .m = 8, .k = 8 }, .i32), .{ .m = .x, .k = .y }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 3 }), .init(.{ .m = 8, .k = 10 }, .i32), .{ .m = .x }, verbose);
    try testShardingCase(allocator, .init(.{ .x = 1 }), .init(.{ .m = 8, .k = 10 }, .i32), .{ .m = .x }, verbose);
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
