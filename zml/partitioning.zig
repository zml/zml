const builtin = @import("builtin");
const std = @import("std");
const stdx = @import("stdx");

const Context = @import("context.zig").Context;
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

        var dim: usize = 0;
        while (dim < self.topology.rank()) : (dim += 1) {
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
    strides: []const i64,
    shard: Shape,
    indices: Shape,
    topology: Shape,
    device: *const Device,

    pub fn data(self: DeviceShard, slice: []const u8) []const u8 {
        return slice[@intCast(self.offset())..];
    }

    pub fn offset(self: DeviceShard) i64 {
        // Compute offset in row-major order (last axis changes fastest)
        const element_size: i64 = @intCast(self.shape.dtype().sizeOf());
        const shard_elems: i64 = @intCast(self.shard.count());

        var offset_: i64 = 0;
        var stride: i64 = 1;

        // Row-major: first axis is outermost, last axis is innermost
        for (0..self.topology.rank()) |dim| {
            const mesh_axis = self.shape.partition(dim);
            if (mesh_axis == Shape.TagUnknown) {
                offset_ = offset_;
            } else {
                offset_ += self.indices.dims()[dim] * stride;
                stride *= self.topology.dim(dim);
            }
        }

        offset_ = offset_ * shard_elems * element_size;

        return offset_;
    }

    pub fn format(
        self: DeviceShard,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt; // autofix
        _ = options;
        try writer.print("DeviceShard(index={d}/{d} topology={} indices={} shard={} ({d}B) shape={} ({d}B) offset={d} device={})", .{ self.index + 1, self.len, self.topology, self.indices, self.shard, self.shard.byteSize(), self.shape, self.shape.byteSize(), self.offset(), self.device });
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
            .strides = self.sharding.strides,
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
    strides: []const i64,

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

test Sharding {
    const mesh: Mesh = .init(.{ .x = 2, .y = 3 });
    var shape: Shape = .init(.{ .m = 4, .k = 6 }, .f16);

    {
        // Fully partitioned sharding
        const shape_fully_partitioned = shape.withPartitionning(.{ .x = .m, .y = .k });
        const sharding_fully_partitioned: Sharding = .init(mesh, shape_fully_partitioned);
        try std.testing.expect(sharding_fully_partitioned.getType() == .maximal);
        try std.testing.expect(sharding_fully_partitioned.shard().eql(Shape.init(.{ .m = 2, .k = 2 }, .f16)));
        try std.testing.expectEqualStrings(sharding_fully_partitioned.shardingString(), "{devices=[2,3]<=[6]}");

        var iter = sharding_fully_partitioned.iterator(env(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }));

        while (iter.next()) |device_shard| {
            std.debug.print("Fully partitioned >> {}\n", .{device_shard});
        }
        // var device_shard = iter.next().?;
        // std.debug.print("Device shard: {}\n", .{device_shard});
        // try std.testing.expect(device_shard.offset() == 0);
        // device_shard = iter.next().?;
        // std.debug.print("Device shard: {}\n", .{device_shard});

        // try std.testing.expect(device_shard.offset() == 24);
        // device_shard = iter.next().?;
        // device_shard = iter.next().?;
        // device_shard = iter.next().?;
        // try std.testing.expect(device_shard.offset() == 192);

        // while (iter.next()) |device_shard| {
        //     try std.testing.expect(device_shard.indices.dims() == &.{ .x = 0, .y = 0 });
        // }
    }

    {
        // Partially partitioned sharding
        const shape_partially_partitioned = shape.withPartitionning(.{ .y = .k });
        const sharding_partially_partitioned: Sharding = .init(mesh, shape_partially_partitioned);
        try std.testing.expect(sharding_partially_partitioned.getType() == .manual);
        try std.testing.expect(sharding_partially_partitioned.shard().eql(Shape.init(.{ .m = 4, .k = 2 }, .f16)));
        try std.testing.expectEqualStrings(sharding_partially_partitioned.shardingString(), "{devices=[1,3]<=[6]}");

        var iter = sharding_partially_partitioned.iterator(env(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }));

        while (iter.next()) |device_shard| {
            std.debug.print("Partially partitioned >> {}\n", .{device_shard});
        }
    }

    {
        // Replicated sharding
        const shape_replicated = shape.withPartitionning(.{});
        const sharding_replicated: Sharding = .init(mesh, shape_replicated);
        try std.testing.expect(sharding_replicated.getType() == .replicated);
        try std.testing.expect(sharding_replicated.shard().eql(shape_replicated));
        try std.testing.expectEqualStrings(sharding_replicated.shardingString(), "{replicated}");

        var iter = sharding_replicated.iterator(env(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }));

        while (iter.next()) |device_shard| {
            std.debug.print("Replicated >> {}\n", .{device_shard});
        }
    }
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
