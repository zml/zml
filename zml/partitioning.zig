const builtin = @import("builtin");
const std = @import("std");
const stdx = @import("stdx");

const pjrtx = @import("pjrtx.zig");
const slice = @import("slice.zig");
const Shaped = slice.Shaped;
const Buffer = @import("buffer.zig").Buffer;
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
                    return current_indices;
                } else {
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

    pub const SliceSpec = struct {
        start_offset: usize,
        byte_strides: [Shape.MAX_RANK]i64,
        dims: [Shape.MAX_RANK]i64,
        num_dims: u4,
    };

    pub fn specs(self: DeviceShard) SliceSpec {
        const rank = self.global_shape.rank();

        const host_byte_strides_ba = self.global_shape.computeByteStrides();

        var shard_start_offset_bytes: i64 = 0;
        for (0..rank) |i| {
            const part_spec = self.global_shape.partition(i);
            if (part_spec == .axis) {
                const mesh_axis_tag = part_spec.toTag();

                // Check if the device's indices actually have this axis.
                // If not, this dimension is effectively replicated for this device.
                if (self.indices.hasTag(mesh_axis_tag) != null) {
                    const device_coord_on_mesh_axis = self.indices.dim(mesh_axis_tag);
                    const shard_dim_size = self.shard_shape.dim(i);
                    const start_element_in_dim = device_coord_on_mesh_axis * shard_dim_size;
                    shard_start_offset_bytes += start_element_in_dim * host_byte_strides_ba.get(i);
                }
            }
        }

        var transfer_dims_buffer: [Shape.MAX_RANK]i64 = undefined;
        for (0..rank) |i| {
            transfer_dims_buffer[i] = self.shard_shape.dim(i);
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
        _ = fmt;
        _ = options;
        const slice_specs = self.specs();
        try writer.print("DeviceShard(index={d}/{d} topology={} indices={} shard_shape={} ({d}B) global_shape={} ({d}B) start_offset={d} byte_strides={any} device={})", .{
            self.index + 1,
            self.topology.count(),
            self.topology,
            self.indices,
            self.shard_shape,
            self.shard_shape.byteSize(),
            self.global_shape,
            self.global_shape.byteSize(),
            slice_specs.start_offset,
            slice_specs.byte_strides[0..slice_specs.num_dims],
            self.device,
        });
    }
};

pub const DeviceShardIterator = struct {
    index: usize = 0,
    indices_iterator: TopologyIndicesIterator,
    sharding: Sharding,
    devices: []const *const pjrtx.Device,

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
            .device = self.devices[self.index],
        };
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

                if (mesh.topology.hasTag(mesh_axis_tag) == null) {
                    // This partitioning is invalid for this mesh, so we skip the divisibility check.
                    // The dimension will effectively be replicated.
                    log.debug("Partitioning axis '{s}' not found in mesh {}. Treating dimension {s} as replicated.", .{
                        std.mem.span(mesh_axis_tag),
                        mesh,
                        shape.debugTag(dim_idx),
                    });
                    continue;
                }

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

    test "init / non existent mesh axis" {
        const mesh: Mesh = .init(.{ .x = 4 });
        const shape: Shape = .init(.{ .m = 8, .n = 12 }, .i32);
        const sharding = Sharding.init(mesh, shape.withPartitioning(.{ .m = .x, .n = .y }));
        const shard_shape = sharding.shardShape();
        try std.testing.expectEqual(2, shard_shape.dim(.m));
        try std.testing.expectEqual(12, shard_shape.dim(.n));
    }

    pub fn getType(self: Sharding) Type {
        var has_any_partitioned_axis = false;
        for (0..self.global_shape.rank()) |i| {
            const part_spec = self.global_shape.partition(i);
            if (part_spec == .axis) {
                if (self.mesh.topology.hasTag(part_spec.toTag()) != null) {
                    has_any_partitioned_axis = true;
                    break;
                }
            }
        }

        if (!has_any_partitioned_axis) {
            return .replicated;
        }

        var is_fully_partitioned = true;
        for (0..self.global_shape.rank()) |i| {
            const part_spec = self.global_shape.partition(i);
            switch (part_spec) {
                .replicated, .open, .unknown => {
                    is_fully_partitioned = false;
                    break;
                },
                .axis => |tag| {
                    if (self.mesh.topology.hasTag(tag) == null) {
                        is_fully_partitioned = false;
                        break;
                    }
                },
            }
        }

        if (is_fully_partitioned) {
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

                // If the specified mesh axis exists, partition. Otherwise, replicate.
                if (self.mesh.topology.hasTag(mesh_axis_tag) != null) {
                    const mesh_dim = self.mesh.topology.dim(mesh_axis_tag);
                    const shard_dim_size = if (mesh_dim > 1) @divExact(global_dim_size, mesh_dim) else global_dim_size;
                    shard_ = shard_.appendDim(shard_dim_size, tag);
                } else {
                    // Mesh axis does not exist, so this dimension is replicated.
                    shard_ = shard_.appendDim(global_dim_size, tag);
                }
            } else {
                // For replicated, open, or unknown, the shard dimension is the full global dimension.
                shard_ = shard_.appendDim(global_dim_size, tag);
            }
        }

        return shard_;
    }

    pub fn iterator(self: Sharding, devices: []const *const pjrtx.Device) DeviceShardIterator {
        const indices_iterator = self.mesh.iterator();

        return .{ .indices_iterator = indices_iterator, .sharding = self, .devices = devices };
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

        var product_of_tensor_tiles: i64 = 1;
        var num_partitioned_dims: u32 = 0;

        // Calculate the product of explicitly defined tile dimensions
        for (0..self.global_shape.rank()) |i| {
            const part_spec = self.global_shape.partition(i);
            if (part_spec == .axis) {
                const mesh_axis_tag = part_spec.toTag();
                if (self.mesh.topology.hasTag(mesh_axis_tag) != null) {
                    product_of_tensor_tiles *= self.mesh.topology.dim(mesh_axis_tag);
                    num_partitioned_dims += 1;
                }
            }
        }

        const is_fully_sharded_on_mesh = (product_of_tensor_tiles == self.mesh.numPartitions());

        if (is_fully_sharded_on_mesh) {
            // This is the easy case. The tensor is partitioned across all mesh devices.
            // The tile array must have the same rank as the tensor.
            try writer.writeAll("{devices=[");
            for (0..self.global_shape.rank()) |i| {
                const part_spec = self.global_shape.partition(i);
                var tile_dim: i64 = 1;
                if (part_spec == .axis) {
                    if (self.mesh.topology.hasTag(part_spec.toTag()) != null) {
                        tile_dim = self.mesh.topology.dim(part_spec.toTag());
                    }
                }
                try writer.print("{d}", .{tile_dim});
                if (i < self.global_shape.rank() - 1) {
                    try writer.writeByte(',');
                }
            }
            try writer.print("]<=[{d}]}}", .{self.mesh.numPartitions()});
        } else {
            // This is partial sharding. This is more complex.
            // The `devices` array lists the tiling only for the *partitioned* dimensions.
            // The remaining devices are handled by `last_tile_dim_replicate`.
            try writer.writeAll("{devices=[");
            var first = true;
            for (0..self.global_shape.rank()) |i| {
                const part_spec = self.global_shape.partition(i);
                if (part_spec == .axis) {
                    const mesh_axis_tag = part_spec.toTag();
                    if (self.mesh.topology.hasTag(mesh_axis_tag) != null) {
                        if (!first) try writer.writeByte(',');
                        try writer.print("{d}", .{self.mesh.topology.dim(mesh_axis_tag)});
                        first = false;
                    }
                }
            }
            try writer.print("]<=[{d}] last_tile_dim_replicate}}", .{self.mesh.numPartitions()});
        }
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

    // Scatter
    var pjrt_buffers = std.ArrayList(*pjrtx.Buffer).init(allocator);
    defer pjrt_buffers.deinit();

    var sharding_iter = sharding.iterator(platform.getDevices());
    while (sharding_iter.next()) |device_shard| {
        if (verbose) std.debug.print("{}\n", .{device_shard});
        const specs = device_shard.specs();
        const args = pjrtx.Client.BufferFromHostBufferArgs{
            .data = original_data[specs.start_offset..].ptr,
            .buffer_type = bufferTypeFromDtype(device_shard.shard_shape.dtype()),
            .dims = specs.dims[0..specs.num_dims],
            .byte_strides = specs.byte_strides[0..specs.num_dims],
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

    // Gather
    if (verbose) std.debug.print("--- Reassembling from shards ---\n", .{});
    const sharded_buffer = Buffer.fromPjrtBuffers(platform, sharding, pjrt_buffers.items);
    defer sharded_buffer.deinit(); // This will deinit all the underlying PJRT buffers.

    const reassembled_data = try sharded_buffer.toHost(allocator);
    defer allocator.free(reassembled_data);

    if (verbose) std.debug.print("Reassembled full slice of data: {any}\n\n", .{Shaped(i32, sharding.global_shape, reassembled_data)});

    // Verify
    try std.testing.expectEqualSlices(u8, original_data, reassembled_data);
    std.debug.print("✅ Verification successful for: {}\n\n", .{sharding});
}

test "Sharding All Cases" {
    const allocator = std.testing.allocator;
    const verbose = true;

    // Fully replicated
    try testShardingCase(allocator, .init(.{ .x = 4 }), .init(.{ .m = 8, .k = 8 }, .i32), .{}, verbose);
    // 1D sharding on a 1D mesh
    try testShardingCase(allocator, .init(.{ .x = 8 }), .init(.{ .m = 16, .k = 16 }, .i32), .{ .k = .x }, verbose);
    // 2D sharding on a 2D mesh
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 4 }), .init(.{ .m = 8, .k = 12 }, .i32), .{ .m = .x, .k = .y }, verbose);
    // 3D tensor on a 2D mesh, partitioned on non-contiguous dimensions
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .b = 4, .h = 8, .w = 6 }, .i32), .{ .b = .x, .w = .y }, verbose);
    // 3D sharding on a 3D mesh
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 4, .z = 3 }), .init(.{ .m = 4, .k = 8, .j = 6 }, .i32), .{ .m = .x, .k = .y, .j = .z }, verbose);
    // 2D tensor, 2D mesh, but only partitioned on one axis
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 3 }), .init(.{ .m = 10, .k = 6 }, .i32), .{ .k = .x }, verbose);
    // Replicated on a 2D mesh
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .m = 4, .k = 4 }, .i32), .{}, verbose);
    // 3D tensor, 2D mesh, one replicated dimension in the middle
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .batch = 4, .seq_len = 10, .features = 8 }, .i32), .{ .batch = .x, .features = .y }, verbose);
    // 2D mesh where one dimension is 1
    try testShardingCase(allocator, .init(.{ .x = 1, .y = 4 }), .init(.{ .m = 8, .k = 8 }, .i32), .{ .m = .x, .k = .y }, verbose);
    // Partially partitioned, non-divisible on replicated axis
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 3 }), .init(.{ .m = 8, .k = 10 }, .i32), .{ .m = .x }, verbose);
    // Single device mesh
    try testShardingCase(allocator, .init(.{ .x = 1 }), .init(.{ .m = 8, .k = 10 }, .i32), .{ .m = .x }, verbose);
    // NEW: 3D tensor, 2D mesh, partitioned on first two axes. Good test for new reassemble logic.
    try testShardingCase(allocator, .init(.{ .x = 2, .y = 2 }), .init(.{ .b = 4, .h = 8, .w = 6 }, .i32), .{ .b = .x, .h = .y }, verbose);
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
