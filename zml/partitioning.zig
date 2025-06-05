const std = @import("std");

const Device = @import("pjrtx.zig").Device;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;

const log = std.log.scoped(.@"zml/partitioning");

pub const MaxMeshSize: u8 = 64;
pub const MaxMeshAxes: u8 = 8;

pub const Mesh = struct {
    topology: Shape,

    pub fn init(topology: anytype) Mesh {
        return .{
            .topology = .init(topology, .u8),
        };
    }

    pub fn rank(self: Mesh) i64 {
        return @intCast(self.topology.rank());
    }

    pub fn axis(self: Mesh, ax: anytype) i64 {
        return self.topology.dim(ax);
    }

    pub fn isPartitioned(self: Mesh) bool {
        return self.numPartitions() > 0;
    }

    pub fn isSinglePartition(self: Mesh) bool {
        return self.numPartitions() == 0;
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
        try writer.print("Mesh(topology={} rank={d} num_devices={d})", .{ self.topology, self.rank(), self.numDevices() });
    }
};

pub const Partition = struct {};

pub const DeviceShard = struct {
    index: usize,
    device: *const Device,
    shape: Shape,
    shard: Shape,
    indices: Shape,
    offset: usize,

    pub fn data(self: DeviceShard, slice: [*]const u8) [*]const u8 {
        const offset = self.offset;
        return slice[offset..];
    }
};

// /// Slices the input Tensor over the given axis_ using the given parameters.
// pub fn slice1d(self: HostBuffer, axis_: anytype, s: Slice) HostBuffer {
//     const ax = self._shape.axis(axis_);
//     const d = self.dim(ax);
//     const start: i64 = if (s.start < 0) s.start + d else s.start;
//     var end = s.end orelse d;
//     if (end < 0) end += d;
//     stdx.debug.assert(start >= 0 and start < d, "slice1d({}, {}) expects the slice start to be between 0 and {} got: {}", .{ self, ax, d, s });
//     stdx.debug.assert(end >= 1 and end <= d, "slice1d({}, {}) expects the slice end to be between 1 and {} got: {}", .{ self, ax, d, s });
//     stdx.debug.assert(start < end, "slice1d({}, {}) expects the slice start ({}) to be smaller than the end ({}), got: {}", .{ self, ax, start, end, s });

//     const offset: usize = @intCast(start * self._strides[ax]);
//     const new_shape = self.shape().set(ax, end - start);
//     return .{
//         ._shape = new_shape,
//         ._data = self._data[offset..],
//         ._strides = self._strides,
//         ._memory = .unmanaged,
//     };
// }

pub const DeviceShardIterator = struct {
    platform: Platform,
    sharding: Sharding,
    indices: Shape,
    index: usize = 0,

    pub fn next(self: *DeviceShardIterator) ?DeviceShard {
        if (self.index >= self.sharding.mesh.numPartitions()) return null;

        defer {
            self.index += 1;
            self.indices = self.indices.advance(self.sharding.shape);
        }

        //     const mesh_axis = self.sharding.shape.partition(dim);
        //     if (mesh_axis == Shape.TagUnknown) {
        //         indices = indices.appendDim(self.sharding.shape.dim(dim), self.sharding.shape.tag(dim));
        //     } else {
        //         const mesh_dim = self.sharding.mesh.topology.dim(mesh_axis);
        //         const d = @divExact(self.sharding.shape.dim(dim), mesh_dim);
        //         indices = indices.setDim(d, self.sharding.shape.tag(dim));
        //     }
        // }

        log.warn("<<< indices: {}", .{self.indices});

        return .{
            .index = self.index,
            .device = self.platform.getDevices()[self.index],
            .shape = self.sharding.shape,
            .shard = self.sharding.shard(),
            .indices = self.indices,
            .offset = 0,
        };
    }

    fn advanceIndices(self: *DeviceShardIterator) void {
        const rank = self.sharding.shape.rank();
        for (0..rank) |dim| {
            const mesh_axis = self.sharding.shape.partition(dim);
            if (mesh_axis == Shape.TagUnknown) {
                self.indices = self.indices.appendDim(self.sharding.shape.dim(dim), self.sharding.shape.tag(dim));
            } else {
                const mesh_dim = self.sharding.mesh.topology.dim(mesh_axis);
                const d = @divExact(self.sharding.shape.dim(dim), mesh_dim);
                self.indices = self.indices.setDim(d, self.sharding.shape.tag(dim));
            }
        }
    }

    fn devices(self: DeviceShardIterator) []const *const Device {
        return self.platform.getDevices();
    }
};

pub const Sharding = struct {
    mesh: Mesh,
    shape: Shape,

    pub fn init(mesh: Mesh, shape: Shape) Sharding {
        return .{
            .mesh = mesh,
            .shape = shape,
        };
    }

    pub fn isSharded(self: Sharding) bool {
        return self.mesh.isPartitioned();
    }

    pub fn isReplicated(self: Sharding) bool {
        return self.isSharded() and !self.shape.hasAtLeastOnePartitionedAxis();
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
        var indices = self.mesh.topology;
        log.warn(">> indices: {}", .{indices});
        log.warn(">> shape: {}", .{self.shape});
        log.warn(">> shard: {}", .{self.shard()});

        for (0..indices.rank()) |dim| {
            // const mesh_axis = self.mesh.topology.partition(dim);
            // _ = mesh_axis; // autofix

            indices = indices.setDim(dim, 0);
        }

        return .{ .platform = platform, .sharding = self, .indices = indices };
    }

    pub fn shardingString(self: Sharding) []const u8 {
        _ = self; // autofix
        var sharding_str: std.BoundedArray(u8, 128) = .{};
        writeShardingRepresentation(sharding_str.writer()) catch unreachable;
        return sharding_str.constSlice();
    }

    pub fn writeShardingRepresentation(self: Sharding, writer: anytype) @TypeOf(writer).Error!void {
        if (self.isReplicated()) {
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

// test writeShardingRepresentation {
//     var rule: [64]u8 = undefined;
//     const x = Shape.init(.{ 16, 8 }, .f32);

//     // By default tensors are replicated.
//     {
//         var fbs = std.io.fixedBufferStream(&rule);
//         try writeShardingRepresentation(x, 4, fbs.writer());
//         try std.testing.expectEqualStrings("{replicated}", fbs.getWritten());
//     }
//     // Shard along first axis.
//     {
//         var fbs = std.io.fixedBufferStream(&rule);
//         try writeShardingRepresentation(x.withSharding(.{0}), 4, fbs.writer());
//         try std.testing.expectEqualStrings("{devices=[4,1]<=[4]}", fbs.getWritten());
//     }
//     // Also shard along second axis.
//     {
//         var fbs = std.io.fixedBufferStream(&rule);
//         try writeShardingRepresentation(x.withSharding(.{ 0, 1 }), 2, fbs.writer());
//         try std.testing.expectEqualStrings("{devices=[2,2]<=[2]}", fbs.getWritten());
//     }
// }
