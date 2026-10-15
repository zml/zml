//! Explain how to split a buffer across different devices.
const std = @import("std");
const builtin = @import("builtin");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const pjrt = @import("pjrt");
const platforms = @import("platforms");
const stdx = @import("stdx");

const Platform = @import("platform.zig").Platform;
const PlatformDevice = @import("platform.zig").Device;
const Shape = @import("shape.zig").Shape;
const Slice = @import("slice.zig").Slice;
const Target = @import("platform.zig").Target;

const Sharding = @This();

mesh: ?*const Mesh,
partition: Partitioning,

pub const MAX_MESH_RANK = 4;
pub const replicated: Sharding = .{ .mesh = null, .partition = .replicated(Shape.MAX_RANK) };

pub fn get(sharding: Sharding, ax: usize) PartitionSpec {
    return if (sharding.mesh == null) .replicated else sharding.partition.get(ax);
}

pub fn set(self: Sharding, ax: usize, spec: PartitionSpec) Sharding {
    var res = self;
    res.partition = res.partition.set(ax, spec);
    return res;
}

pub fn insert(self: Sharding, ax: usize, spec: PartitionSpec) Sharding {
    var res = self;
    res.partition = res.partition.insert(ax, spec);
    return res;
}

pub fn orderedRemove(self: Sharding, ax: usize) Sharding {
    var res = self;
    res.partition = res.partition.orderedRemove(ax);
    return res;
}

pub fn toArray(self: Sharding) [Shape.MAX_RANK]PartitionSpec {
    var result: [Shape.MAX_RANK]PartitionSpec = undefined;
    for (&result, 0..) |*spec, ax| spec.* = self.get(ax);
    return result;
}

pub fn isFullyReplicated(self: Sharding) bool {
    if (self.mesh == null) return true;
    for (self.partition.toArray()) |spec| {
        if (spec != .replicated and spec != .out_of_bound) return false;
    }
    return true;
}

/// An unspecified mesh is compatible with any sharding. On the same mesh,
/// open axes accept any in-bounds partition, while closed axes must match.
pub fn isCompatible(self: Sharding, other: Sharding) bool {
    if (self.mesh == null or other.mesh == null) return true;
    if (self.mesh != other.mesh) return false;
    for (self.partition.toArray(), other.partition.toArray()) |lhs, rhs| {
        if (lhs == rhs) continue;
        if (lhs == .out_of_bound or rhs == .out_of_bound) return false;
        if (lhs != .open and rhs != .open) return false;
    }
    return true;
}

test isCompatible {
    const tp: Mesh = .initTest(.tp, .{.model});
    const dp: Mesh = .initTest(.dp, .{ .data, .model });

    const sharded: Sharding = .{ .mesh = &tp, .partition = .init(&.{ .mesh_axis_0, .replicated }) };
    const cases = [_]struct { lhs: Sharding, rhs: Sharding, compatible: bool }{
        .{ .lhs = .replicated, .rhs = .replicated, .compatible = true },
        .{ .lhs = .replicated, .rhs = sharded, .compatible = true },
        .{ .lhs = sharded, .rhs = sharded, .compatible = true },
        .{ .lhs = sharded, .rhs = .{ .mesh = &tp, .partition = .open(2) }, .compatible = true },
        .{ .lhs = sharded.set(1, .open), .rhs = sharded.set(0, .open), .compatible = true },
        .{ .lhs = sharded, .rhs = sharded.set(0, .mesh_axis_1), .compatible = false },
        .{ .lhs = sharded, .rhs = sharded.set(0, .replicated), .compatible = false },
        .{ .lhs = sharded, .rhs = .{ .mesh = &dp, .partition = sharded.partition }, .compatible = false },
        .{ .lhs = sharded, .rhs = .{ .mesh = &dp, .partition = .open(2) }, .compatible = false },
        .{ .lhs = sharded, .rhs = sharded.set(1, .out_of_bound), .compatible = false },
        .{ .lhs = sharded, .rhs = sharded.set(2, .open), .compatible = false },
    };
    for (cases) |case| {
        errdefer std.log.err("{f}.isCompatible({f}) expected to be {}, got {}", .{ case.lhs, case.rhs, case.compatible, !case.compatible });

        try std.testing.expectEqual(case.compatible, case.lhs.isCompatible(case.rhs));
        try std.testing.expectEqual(case.compatible, case.rhs.isCompatible(case.lhs));
    }
}

pub fn format(self: Sharding, writer: *std.Io.Writer) !void {
    const mesh = self.mesh orelse return writer.writeAll("{replicated}");
    const specs = self.partition.toArray();
    var rank = specs.len;
    while (rank > 0 and specs[rank - 1] == .out_of_bound) rank -= 1;

    try writer.writeByte('{');
    for (specs[0..rank], 0..) |spec, i| {
        if (i > 0) try writer.writeByte(',');
        switch (spec) {
            .replicated => try writer.writeAll("replicated"),
            .open => try writer.writeByte('?'),
            .out_of_bound => try writer.writeByte('!'),
            else => try writer.print("{s}", .{mesh.logical.axes.get(spec.meshAxis().?)}),
        }
    }
    try writer.print("}}@{s}", .{mesh.name});
}

/// Describes how a given Shape axis behaves inside a mesh.
/// Is it replicated ? sharded along a specific logical axis ? or open to replication ?
pub const PartitionSpec = enum(u4) {
    mesh_axis_0 = 0,
    mesh_axis_1 = 1,
    mesh_axis_2 = 2,
    mesh_axis_3 = 3,
    mesh_axis_4 = 4,
    mesh_axis_5 = 5,
    mesh_axis_6 = 6,
    mesh_axis_7 = 7,
    // an 8D mesh seems already a lot, the max we know about is 3D.
    replicated = 8,

    // Partitioning don't encode the rank of corresponding tensor, so this is used to mark invalid positions.
    out_of_bound = 0xa,
    open = 15,

    pub fn sharded(mesh_axis: u3) PartitionSpec {
        return @fromBackingInt(@intCast(mesh_axis));
    }

    /// Extract the mesh axis along which we are sharded. Null if not sharded.
    pub fn meshAxis(self: PartitionSpec) ?u3 {
        const ax = @backingInt(self);
        return if (ax < @backingInt(PartitionSpec.replicated)) @intCast(ax) else null;
    }

    pub fn isSharded(self: PartitionSpec) bool {
        return @backingInt(self) < @backingInt(PartitionSpec.replicated);
    }

    pub fn isClosed(self: PartitionSpec) bool {
        return @backingInt(self) <= @backingInt(PartitionSpec.replicated);
    }

    test isClosed {
        try std.testing.expect(PartitionSpec.mesh_axis_2.isClosed());
        try std.testing.expect(!PartitionSpec.open.isClosed());

        try std.testing.expect(PartitionSpec.replicated.isClosed());

        try std.testing.expect(!PartitionSpec.out_of_bound.isClosed());
    }
};

pub const Partitioner = union(enum) {
    shardy,
    gspmd,

    pub fn fromTarget(target: Target) Partitioner {
        return switch (target) {
            .cpu, .cuda, .rocm, .tpu, .oneapi, .neuron, .metal, .furiosa => .shardy,
        };
    }
};

fn shapeHasAxisPartition(shape: Shape) bool {
    for (0..shape.rank()) |ax| {
        if (shape.partition(ax).meshAxis()) |_| return true;
    }
    return false;
}

pub const DimSharding = union(enum) {
    sharded: struct {
        dim: i64,
        factor: u32,
    },
    replicated,
};

pub fn sdyPerValueShardingAttr(
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    shapes: []const Shape,
    mesh: *const Mesh,
) error{OutOfMemory}!*const mlir.Attribute {
    const meshe_attrs = try allocator.alloc(*const dialects.shardy.TensorShardingAttribute, shapes.len);
    for (meshe_attrs, shapes) |*attr, shape| {
        attr.* = try mesh.sdyShardingAttrForShape(allocator, ctx, shape);
    }
    return dialects.shardy.TensorShardingPerValueAttribute.init(ctx, meshe_attrs).asAttr();
}

pub fn sdyManualAxesAttr(
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    in_shapes: []const Shape,
    out_shapes: []const Shape,
    mesh: *const Mesh,
) error{OutOfMemory}!*const mlir.Attribute {
    var axis_names = std.ArrayList([]const u8).empty;
    defer axis_names.deinit(allocator);

    const Collect = struct {
        fn appendUnique(list: *std.ArrayList([]const u8), allocator_: std.mem.Allocator, axis_name: []const u8) void {
            for (list.items) |existing| {
                if (std.mem.eql(u8, existing, axis_name)) return;
            }
            list.append(allocator_, axis_name) catch unreachable;
        }
    };

    for (in_shapes) |shape| {
        const attr = try mesh.sdyShardingAttrForShape(allocator, ctx, shape);
        for (0..attr.numReplicatedAxes()) |i| {
            Collect.appendUnique(&axis_names, allocator, attr.replicatedAxis(i).name());
        }
        for (0..attr.numDimensions()) |i| {
            const dim = attr.dimension(i);
            for (0..dim.numAxes()) |j| {
                Collect.appendUnique(&axis_names, allocator, dim.axis(j).name());
            }
        }
    }
    for (out_shapes) |shape| {
        const attr = try mesh.sdyShardingAttrForShape(allocator, ctx, shape);
        for (0..attr.numReplicatedAxes()) |i| {
            Collect.appendUnique(&axis_names, allocator, attr.replicatedAxis(i).name());
        }
        for (0..attr.numDimensions()) |i| {
            const dim = attr.dimension(i);
            for (0..dim.numAxes()) |j| {
                Collect.appendUnique(&axis_names, allocator, dim.axis(j).name());
            }
        }
    }

    const axes = try allocator.alloc(*const mlir.StringAttribute, axis_names.items.len);
    for (axis_names.items, 0..) |axis_name, i| {
        axes[i] = mlir.StringAttribute.init(ctx, axis_name);
    }

    return dialects.shardy.ManualAxesAttribute.init(ctx, axes).asAttr();
}

pub const Partitioning = packed struct {
    _0: PartitionSpec,
    _1: PartitionSpec,
    _2: PartitionSpec,
    _3: PartitionSpec,
    _4: PartitionSpec,
    _5: PartitionSpec,
    _6: PartitionSpec,
    _7: PartitionSpec,

    pub const out_of_bound: Partitioning = splat(.out_of_bound);

    const MAX_RANK = Shape.MAX_RANK;
    const Vec = @Vector(MAX_RANK, u4);

    pub fn init(specs: []const PartitionSpec) Partitioning {
        var res: Partitioning = out_of_bound;
        for (0.., specs) |i, spec| {
            res = res.set(i, spec);
        }
        return res;
    }

    pub fn replicated(rank_: usize) Partitioning {
        return repeat(.replicated, rank_);
    }

    pub fn open(rank_: usize) Partitioning {
        return repeat(.open, rank_);
    }

    pub fn splat(spec: PartitionSpec) Partitioning {
        const vec: Vec = @splat(@backingInt(spec));
        return @bitCast(vec);
    }

    pub fn repeat(spec: PartitionSpec, rank_: usize) Partitioning {
        std.debug.assert(rank_ <= MAX_RANK);
        const splatted: Vec = @splat(@backingInt(spec));
        const mask = std.simd.iota(u4, 8) < @as(Vec, @splat(@truncate(rank_)));
        return @bitCast(@select(u4, mask, splatted, @as(Vec, @bitCast(out_of_bound))));
    }

    pub fn get(p: Partitioning, ax: usize) PartitionSpec {
        std.debug.assert(ax < MAX_RANK);
        const pack: u32 = @bitCast(p);
        const shift: u5 = @intCast(4 * ax);
        return @fromBackingInt(@truncate(pack >> shift));
    }

    pub fn set(p: Partitioning, ax: usize, spec: PartitionSpec) Partitioning {
        std.debug.assert(ax < MAX_RANK);
        const pack: u32 = @bitCast(p);
        const shift: u5 = @intCast(4 * ax);
        const mask = @as(u32, 0xf) << shift;
        return @bitCast((pack & ~mask) | (@as(u32, @backingInt(spec)) << shift));
    }

    pub fn rank(p: Partitioning) u8 {
        for (0..MAX_RANK) |i_usize| {
            const i: u8 = @truncate(i_usize);
            if (p.get(i) == .out_of_bound) return i;
        }
        return MAX_RANK;
    }

    /// Inserts a spec, shifting subsequent slots right and discarding the last slot.
    pub fn insert(p: Partitioning, ax: usize, spec: PartitionSpec) Partitioning {
        std.debug.assert(ax < MAX_RANK);
        const pack: u32 = @bitCast(p);
        const shift: u5 = @intCast(4 * ax);
        const lower_mask = (@as(u32, 1) << shift) - 1;
        return @bitCast((pack & lower_mask) | ((pack & ~lower_mask) << 4) | (@as(u32, @backingInt(spec)) << shift));
    }

    /// Removes a spec, shifting subsequent slots left and filling the last with unknown.
    pub fn orderedRemove(p: Partitioning, ax: usize) Partitioning {
        std.debug.assert(ax < MAX_RANK);
        const pack: u32 = @bitCast(p);
        const shift: u5 = @intCast(4 * ax);
        const lower_mask = (@as(u32, 1) << shift) - 1;
        return @bitCast((pack & lower_mask) | ((pack >> 4) & ~lower_mask) | (@as(u32, @backingInt(PartitionSpec.out_of_bound)) << 28));
    }

    pub fn toArray(p: Partitioning) [MAX_RANK]PartitionSpec {
        var res: [MAX_RANK]PartitionSpec = undefined;
        for (&res, 0..) |*spec, ax| spec.* = p.get(ax);
        return res;
    }

    pub fn hasUniqueAxes(p: Partitioning) bool {
        var used_mesh_axes: std.StaticBitSet(MAX_RANK) = .empty;
        for (0..MAX_RANK) |shape_ax| {
            const spec = p.get(shape_ax);
            if (spec.meshAxis()) |mesh_axis| {
                if (used_mesh_axes.isSet(mesh_axis)) return false;
                used_mesh_axes.set(mesh_axis);
            }
        }
        return true;
    }

    test hasUniqueAxes {
        try std.testing.expect(hasUniqueAxes(.init(&.{ .mesh_axis_0, .mesh_axis_1, .replicated, .mesh_axis_2 })));
        try std.testing.expect(!hasUniqueAxes(.init(&.{ .mesh_axis_0, .mesh_axis_1, .replicated, .mesh_axis_0 })));
    }

    test "packed access and updates" {
        try std.testing.expectEqual(32, @bitSizeOf(Partitioning));
        try std.testing.expectEqual(4, @sizeOf(Partitioning));
        inline for (std.meta.tags(PartitionSpec)) |spec| {
            const filled = Partitioning.splat(spec);
            for (0..MAX_RANK) |ax| {
                try std.testing.expectEqual(spec, filled.get(ax));
                const updated = Partitioning.out_of_bound.set(ax, spec);
                for (0..MAX_RANK) |i| {
                    try std.testing.expectEqual(if (i == ax) spec else .out_of_bound, updated.get(i));
                }
            }
        }
        for (0..MAX_RANK + 1) |rank_| {
            const parts = Partitioning.replicated(rank_);
            for (0..MAX_RANK) |ax| {
                try std.testing.expectEqual(if (ax < rank_) PartitionSpec.replicated else .out_of_bound, parts.get(ax));
            }
        }
    }

    test "insertion and removal at every slot" {
        var parts: Partitioning = .out_of_bound;
        for (0..MAX_RANK) |ax| parts = parts.set(ax, @fromBackingInt(@intCast(ax)));
        const original = parts.toArray();
        for (0..MAX_RANK) |ax| {
            var inserted = original;
            std.mem.copyBackwards(PartitionSpec, inserted[ax + 1 ..], original[ax .. MAX_RANK - 1]);
            inserted[ax] = .open;
            try std.testing.expectEqualSlices(PartitionSpec, &inserted, &parts.insert(ax, .open).toArray());

            var removed = original;
            std.mem.copyForwards(PartitionSpec, removed[ax .. MAX_RANK - 1], original[ax + 1 ..]);
            removed[MAX_RANK - 1] = .out_of_bound;
            try std.testing.expectEqualSlices(PartitionSpec, &removed, &parts.orderedRemove(ax).toArray());
            try std.testing.expectEqual(parts, parts.orderedRemove(ax).insert(ax, parts.get(ax)));
        }
        try std.testing.expectEqual(Partitioning.out_of_bound, Partitioning.out_of_bound.orderedRemove(0));
        const comptime_parts = comptime Partitioning.out_of_bound.insert(7, .open).orderedRemove(0).set(0, .replicated);
        try std.testing.expectEqual(PartitionSpec.replicated, comptime_parts.get(0));
        try std.testing.expectEqual(PartitionSpec.open, comptime_parts.get(6));
        try std.testing.expectEqual(PartitionSpec.out_of_bound, comptime_parts.get(7));
    }

    /// Parse a struct literal into a partitioning object.
    /// The struct passed to partitioning must only have fields names present in `tags`.
    /// The ordering of tags matter, the return struct will map names to axis as specified by `tags`.
    /// The default value is important.
    /// In Tensor code unspecified axis will be inferred as `.open` while in shape code, it's `.replicated`.
    pub fn parse(tags: []const [*:0]const u8, mesh: *const Mesh, default_value: PartitionSpec, partitioning: anytype) Partitioning {
        const logical = if (@TypeOf(partitioning) == Logical)
            partitioning
        else
            // capture all the comptime logic in the Logical struct.
            comptime Logical.parse(partitioning);

        return logical.applyUseAll(tags, mesh, default_value);
    }

    pub const Logical = struct {
        src: Shape.TagsArray,
        tgt: Shape.TagsArray,

        const replicated: Shape.Tag = Shape.toTag(.replicated);
        const open: Shape.Tag = Shape.toTag(.open);

        pub fn parse(v: anytype) Logical {
            const V = @TypeOf(v);

            var src_: Shape.TagsArray = .empty;
            var tgt_: Shape.TagsArray = .empty;

            if (comptime stdx.meta.isStruct(V)) {
                const field_names = std.meta.fieldNames(V);
                stdx.debug.assertComptime(field_names.len <= @TypeOf(src_).max_len, "Too many tags in struct {} ({d})", .{ V, field_names.len });
                inline for (field_names) |field_name| {
                    const fv = @field(v, field_name);
                    tgt_.appendAssumeCapacity(@tagName(fv));

                    if (!comptime stdx.meta.isTuple(V)) {
                        src_.appendAssumeCapacity(field_name);
                    }
                }
                return .{ .src = src_, .tgt = tgt_ };
            }

            stdx.debug.compileError("Sharding.Partitioning.parse expects a struct of enum literals, eg {{ .b = .data, .d = .model }}, got: {any}", .{V});
        }

        test "Logical.parse" {
            const logical = comptime Logical.parse(.{ .a = .x, .b = .y });

            try std.testing.expectEqualSlices(Shape.Tag, &.{ "a".ptr, "b".ptr }, logical.src.constSlice());
            try std.testing.expectEqualSlices(Shape.Tag, &.{ "x".ptr, "y".ptr }, logical.tgt.constSlice());
        }

        pub fn apply(logical: *const Logical, tags: []const [*:0]const u8, mesh: *const Mesh, default_value: PartitionSpec) struct { Partitioning, ?u8 } {
            var partition_: Partitioning = .repeat(default_value, tags.len);
            var missing_src_axis: ?u8 = null;

            for (0.., logical.src.slice(), logical.tgt.slice()) |i, shape_tag, mesh_tag| {
                const shape_ax = Shape.axisFromTagMaybe(tags, shape_tag) orelse {
                    missing_src_axis = @intCast(i);
                    continue;
                };

                const spec: PartitionSpec = if (mesh_tag == Logical.replicated)
                    .replicated
                else if (mesh_tag == Logical.open)
                    .open
                else spec: {
                    const mesh_ax = mesh.resolveLogicalAxis(mesh_tag) orelse {
                        stdx.debug.panic("{f} has no logical axis '{s}'", .{ mesh, mesh_tag });
                    };
                    break :spec .sharded(@intCast(mesh_ax));
                };
                partition_ = partition_.set(shape_ax, spec);
            }
            return .{ partition_, missing_src_axis };
        }

        pub fn applyUseAll(logical: *const Logical, tags: []const [*:0]const u8, mesh: *const Mesh, default_value: PartitionSpec) Partitioning {
            const p, const missing_shape_axis = logical.apply(tags, mesh, default_value);

            if (missing_shape_axis) |ax| {
                std.debug.panic("{f} doesn't have an axis {s} to be partitioned on.", .{ stdx.fmt.stringsZ(tags), logical.src.get(ax) });
            }
            return p;
        }
    };
};

/// Device as part of a PhysicalMesh.
pub const Device = struct {
    /// Unique device identifier in the mesh
    id: u32,

    /// Coordinates in the physical mesh
    coords: Coords,

    const Coords = [MAX_MESH_RANK]u8;
    // Placeholder coords, that would most likely trigger panic if used like that.
    // Those are used when creating a mesh, before calling assignCoords
    // Private so it's kept as implementation detail
    const undefined_coords: Coords = @splat(0xff);

    fn hasValidCoords(device: Device) bool {
        return @as(u32, @bitCast(device.coords)) != @as(u32, @bitCast(undefined_coords));
    }
};

/// Physical axis tags represent hardware interconnect tiers.
pub const PhysicalAxisTag = enum {
    link, // Intra-device / island-local 1D interconnect
    link_x, // Torus/Mesh Dim X (TPU, Trainium)
    link_y, // Torus/Mesh Dim Y (TPU, Trainium)
    link_z, // Torus/Mesh Dim Z (TPU v3/v4/v5)
    bus, // PCIe / NUMA (Potential isolation boundary)
};

/// Ring geometry style for 1D neighbor-only interconnects.
pub const RingKind = enum {
    linear, // Neighbors only, open ends
    closed_ring, // Neighbors only, circular (wrap-around)
};

/// Mesh geometry style for 2D/3D neighbor-only interconnects.
pub const MeshKind = enum {
    grid, // 2D/3D Neighbors only, open ends
    torus, // 2D/3D Neighbors only, circular wrap-around
};

/// AxisGeometry describes the communication pattern for an axis.
pub const AxisGeometry = union(enum) {
    point_to_point: void, // Full all-to-all (NVLink Switch, TPU v5p)
    ring: RingKind, // Neighbor-only 1D
    mesh: MeshKind, // Neighbor-only 2D/3D
    tree: void, // Hierarchical (Standard PCIe)
    isolated: void, // No direct communication (AWS Inf2 islands)

    pub fn format(self: AxisGeometry, writer: *std.Io.Writer) !void {
        switch (self) {
            .point_to_point => try writer.writeAll("P2P"),
            .ring => |k| try writer.print("Ring({s})", .{@tagName(k)}),
            .mesh => |k| try writer.print("Mesh({s})", .{@tagName(k)}),
            .tree => try writer.writeAll("Tree"),
            .isolated => try writer.writeAll("Isolated"),
        }
    }
};

/// Represents the hierarchical physical topology of the hardware devices.
pub const PhysicalNode = union(enum) {
    /// A branch represents a physical axis
    branch: struct {
        tag: PhysicalAxisTag,
        geometry: AxisGeometry,
        children: []PhysicalNode,
    },
    /// A leaf represents the actual hardware device
    leaf: Device,

    pub fn axis(tag: PhysicalAxisTag, geometry_: AxisGeometry, children: []const PhysicalNode) PhysicalNode {
        return .{
            .branch = .{
                .tag = tag,
                .geometry = geometry_,
                .children = @constCast(children),
            },
        };
    }

    pub fn device(device_: PlatformDevice) PhysicalNode {
        return .{
            .leaf = .{
                .id = @intCast(device_.id()),
                .coords = Device.undefined_coords,
            },
        };
    }

    pub fn countDevices(self: PhysicalNode) usize {
        return switch (self) {
            .leaf => 1,
            .branch => |b| {
                var sum: usize = 0;
                for (b.children) |child| sum += child.countDevices();
                return sum;
            },
        };
    }
};

/// PhysicalMesh models hardware as a tree of axes.
/// Each branch = one physical axis (link_x/link_y/bus/etc),
/// each leaf = one device.
///
/// Diagram:
///   link_x
///    ├─ link_y
///    │   └─ device
///    └─ link_y
///        └─ device
///
/// The DFS axis order defines canonical coordinate ordering.
pub const PhysicalMesh = struct {
    /// Explicit topology tree API (canonical form)
    pub const Tree = PhysicalNode;

    /// AxisTraversal captures canonical axis order and depth mapping.
    pub const AxisTraversal = struct {
        pub const Order = stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK);
        pub const DepthByTag = std.EnumArray(PhysicalAxisTag, ?u8);

        /// Order of axes (DFS, first-child descent).
        order: Order,
        /// Depth (index) of each axis tag in the order.
        depth_by_tag: DepthByTag,

        /// Build axis order and depth mapping from a topology tree.
        pub fn init(root: PhysicalNode) AxisTraversal {
            var order: Order = .empty;
            axisOrderNode(root, &order);

            var depth_by_tag: DepthByTag = .initFill(null);
            for (order.slice(), 0..) |t, i| {
                depth_by_tag.set(t, @intCast(i));
            }

            return .{
                .order = order,
                .depth_by_tag = depth_by_tag,
            };
        }

        pub fn depth(self: AxisTraversal, tag: PhysicalAxisTag) ?u8 {
            return self.depth_by_tag.get(tag);
        }

        /// Canonical DFS axis order:
        ///
        /// Tree:
        ///   link_x
        ///    └─ link_y
        ///       └─ leaf
        ///
        /// Order = [link_x, link_y]
        ///
        /// We descend only the first child to keep a stable axis order that
        /// depends on structure (axes), not on device count.
        fn axisOrderNode(node: PhysicalNode, out: *Order) void {
            switch (node) {
                .leaf => {},
                .branch => |b| {
                    out.appendAssumeCapacity(b.tag);
                    if (b.children.len > 0) axisOrderNode(b.children[0], out);
                },
            }
        }
    };

    /// AxisInfo exposes the size and geometry for a tag.
    pub const AxisInfo = struct {
        tag: PhysicalAxisTag,
        size: i64,
        geometry: AxisGeometry,
    };

    target: Target,
    root: PhysicalNode,
    axis_traversal: AxisTraversal,
    devices_in_canonical_order: []Device,

    /// Build a PhysicalMesh from a tree:
    /// - clone the tree into `allocator`
    /// - validate geometry invariants
    /// - assign coordinates to leaves
    /// - compute axis traversal and canonical device order
    pub fn fromTree(allocator: std.mem.Allocator, target: Target, root: PhysicalNode) !PhysicalMesh {
        const cloned = try cloneNode(allocator, root);
        errdefer freeNode(allocator, cloned);

        return try fromOwnedTree(allocator, target, cloned);
    }

    fn fromOwnedTree(allocator: std.mem.Allocator, target: Target, root: PhysicalNode) !PhysicalMesh {
        try validateGeometry(root);

        var owned_root = root;
        var path: Device.Coords = @splat(0);
        assignCoords(&owned_root, &path, 0);

        var mesh: PhysicalMesh = .{
            .target = target,
            .root = owned_root,
            .axis_traversal = .init(owned_root),
            .devices_in_canonical_order = undefined,
        };
        try mesh.populateDevicesInCanonicalOrder(allocator);
        for (mesh.devices_in_canonical_order) |device| {
            std.debug.assert(device.hasValidCoords());
        }
        return mesh;
    }

    pub fn torus1d(
        allocator: std.mem.Allocator,
        target: Target,
        devices: []const PlatformDevice,
    ) !Sharding.PhysicalMesh {
        const nodes = try allocator.alloc(PhysicalNode, devices.len);
        for (nodes, devices) |*node, device| {
            node.* = .device(device);
        }

        return .fromOwnedTree(allocator, target, .axis(.link_x, .{ .mesh = .torus }, nodes));
    }

    pub fn torus2d(
        allocator: std.mem.Allocator,
        target: Target,
        devices: []const PlatformDevice,
    ) !Sharding.PhysicalMesh {
        // Find the best split: 4 -> 2x2, 6 -> 2x3, 8 -> 2x4
        var link_x: usize = 1;
        var link_y: usize = devices.len;
        if (devices.len < 2) return error.NotEnoughDevices;

        for (2..devices.len) |i| {
            if (@mod(devices.len, i) != 0) continue;
            const j = @divExact(devices.len, i);
            if (i <= j) {
                link_x = i;
                link_y = j;
            } else break;
        }

        // This only happen if number of devices is 1 or a prime number
        if (link_x == 1) return error.NotEnoughDevices;

        const x_axes = try allocator.alloc(PhysicalNode, link_x);
        var device_queue = devices;
        for (x_axes) |*x_ax| {
            const y_devices = try allocator.alloc(PhysicalNode, link_y);
            for (y_devices) |*y_dev| {
                y_dev.* = .device(device_queue[0]);
                device_queue = device_queue[1..];
            }

            x_ax.* = .axis(.link_y, .{ .mesh = .torus }, y_devices);
        }

        return .fromOwnedTree(allocator, target, .axis(.link_x, .{ .mesh = .torus }, x_axes));
    }

    pub fn torus3d(
        allocator: std.mem.Allocator,
        target: Target,
        devices: []const PlatformDevice,
    ) !Sharding.PhysicalMesh {
        if (devices.len < 8) return error.NotEnoughDevices;
        if (devices.len > 8) @panic("TODO: adapt logic from torus2D");
        const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
            .axis(.link_y, .{ .mesh = .torus }, &.{
                .axis(.link_z, .{ .mesh = .torus }, &.{
                    .device(devices[0]),
                    .device(devices[1]),
                }),
                .axis(.link_z, .{ .mesh = .torus }, &.{
                    .device(devices[2]),
                    .device(devices[3]),
                }),
            }),
            .axis(.link_y, .{ .mesh = .torus }, &.{
                .axis(.link_z, .{ .mesh = .torus }, &.{
                    .device(devices[4]),
                    .device(devices[5]),
                }),
                .axis(.link_z, .{ .mesh = .torus }, &.{
                    .device(devices[6]),
                    .device(devices[7]),
                }),
            }),
        });

        return .fromTree(allocator, target, topology);
    }

    fn populateDevicesInCanonicalOrder(self: *PhysicalMesh, allocator: std.mem.Allocator) !void {
        var devices_: std.ArrayList(Device) = .empty;
        errdefer devices_.deinit(allocator);
        try devicesNodeInto(allocator, self.root, &devices_);
        self.devices_in_canonical_order = try devices_.toOwnedSlice(allocator);

        const Order = struct {
            mesh: *const PhysicalMesh,
            fn lessThan(ctx: @This(), a: Device, b: Device) bool {
                const ia = ctx.mesh.linearIndexFromCoords(a.coords);
                const ib = ctx.mesh.linearIndexFromCoords(b.coords);
                return ia < ib;
            }
        };

        std.mem.sort(Device, self.devices_in_canonical_order, Order{ .mesh = self }, Order.lessThan);
    }

    fn cloneNode(allocator: std.mem.Allocator, node: PhysicalNode) !PhysicalNode {
        return switch (node) {
            .leaf => |d| .{
                .leaf = .{
                    .id = d.id,
                    .coords = d.coords,
                },
            },
            .branch => |b| blk: {
                const children = try allocator.alloc(PhysicalNode, b.children.len);
                errdefer allocator.free(children);
                for (b.children, 0..) |child, i| {
                    children[i] = try cloneNode(allocator, child);
                }
                break :blk .{
                    .branch = .{
                        .tag = b.tag,
                        .geometry = b.geometry,
                        .children = children,
                    },
                };
            },
        };
    }

    fn freeNode(allocator: std.mem.Allocator, node: PhysicalNode) void {
        switch (node) {
            .leaf => {},
            .branch => |b| {
                for (b.children) |child| freeNode(allocator, child);
                allocator.free(b.children);
            },
        }
    }

    pub fn deinit(self: *PhysicalMesh, allocator: std.mem.Allocator) void {
        allocator.free(self.devices_in_canonical_order);
        freeNode(allocator, self.root);
        self.* = undefined;
    }

    fn validateGeometry(node: PhysicalNode) !void {
        switch (node) {
            .leaf => return,
            .branch => |b| {
                if (b.children.len == 0) return error.InvalidPhysicalMesh;

                switch (b.geometry) {
                    .mesh => {
                        var has_leaf = false;
                        var has_branch = false;
                        var expected_tag: ?PhysicalAxisTag = null;

                        for (b.children) |child| {
                            switch (child) {
                                .leaf => has_leaf = true,
                                .branch => |cb| {
                                    has_branch = true;
                                    if (expected_tag == null) {
                                        expected_tag = cb.tag;
                                    } else if (cb.tag != expected_tag.?) {
                                        return error.IncompatibleGeometry;
                                    }
                                },
                            }
                        }

                        if (has_leaf and has_branch) return error.IncompatibleGeometry;
                    },
                    .ring => {
                        if (b.children.len < 2) return error.IncompatibleGeometry;
                    },
                    .point_to_point, .tree, .isolated => {},
                }

                for (b.children) |child| try validateGeometry(child);
            },
        }
    }

    /// Assign coords to each leaf using DFS path indices.
    ///
    /// Example (2x2):
    ///   link_x
    ///    └─ link_y
    ///
    /// coords:
    ///   (0,0) (0,1)
    ///   (1,0) (1,1)
    fn assignCoords(node: *PhysicalNode, path: *Device.Coords, depth: usize) void {
        switch (node.*) {
            .leaf => |*d| {
                // Coords can already been set by caller
                if (d.hasValidCoords()) return;

                d.coords = path.*;
            },
            .branch => |*b| {
                for (b.children, 0..) |*child, i| {
                    path[depth] = @intCast(i);
                    assignCoords(child, path, depth + 1);
                }
            },
        }
    }

    pub fn countDevices(self: PhysicalMesh) usize {
        return self.root.countDevices();
    }

    fn devicesNodeInto(allocator: std.mem.Allocator, node: PhysicalNode, out: *std.ArrayList(Device)) !void {
        switch (node) {
            .leaf => |d| try out.append(allocator, d),
            .branch => |b| {
                for (b.children) |child| {
                    try devicesNodeInto(allocator, child, out);
                }
            },
        }
    }

    /// Canonical DFS linearization for coordinate consensus.
    ///
    /// sizes: S0..Sk, coords: C0..Ck
    /// linear = (((C0 * S1) + C1) * S2 + C2) ...
    pub fn linearIndexFromCoords(self: PhysicalMesh, coords: Device.Coords) usize {
        const order = self.axisOrder();
        var idx: usize = 0;
        for (order.slice()) |tag| {
            const depth = self.axis_traversal.depth(tag);
            if (depth) |d| {
                const coord = coords[@intCast(d)];
                const size = self.axis(tag);
                idx = idx * @as(usize, @intCast(size)) + coord;
            }
        }
        return idx;
    }

    pub fn axisInfo(self: PhysicalMesh, tag: PhysicalAxisTag) ?AxisInfo {
        return axisInfoNode(self.root, tag);
    }

    fn axisInfoNode(node: PhysicalNode, tag: PhysicalAxisTag) ?AxisInfo {
        return switch (node) {
            .leaf => null,
            .branch => |b| {
                if (b.tag == tag) {
                    return .{
                        .tag = b.tag,
                        .size = @intCast(b.children.len),
                        .geometry = b.geometry,
                    };
                }

                for (b.children) |child| {
                    if (axisInfoNode(child, tag)) |info| return info;
                }

                return null;
            },
        };
    }

    pub fn hasAxis(self: PhysicalMesh, tag: PhysicalAxisTag) bool {
        return self.axisInfo(tag) != null;
    }

    pub fn axis(self: PhysicalMesh, tag: PhysicalAxisTag) i64 {
        if (self.axisInfo(tag)) |info| return info.size;

        return 1;
    }

    pub fn axisOrder(self: PhysicalMesh) AxisTraversal.Order {
        return self.axis_traversal.order;
    }

    pub fn geometry(self: PhysicalMesh, tag: PhysicalAxisTag) ?AxisGeometry {
        if (self.axisInfo(tag)) |info| return info.geometry;

        return null;
    }

    /// Preferred axis ordering by speed: fastest -> slowest.
    /// This is used by Strategy.suggest to match intent to bandwidth.
    pub fn shardableAxes(self: PhysicalMesh) []const PhysicalAxisTag {
        return switch (self.target) {
            .tpu => &.{ .link_x, .link_y, .link_z },
            .neuron => &.{ .link, .link_x, .link_y, .link_z },
            .cuda, .rocm => &.{.link},
            .cpu, .metal, .furiosa => &.{.bus},
            .oneapi => &.{ .link, .bus },
        };
    }

    pub fn isShardable(self: PhysicalMesh, tag: PhysicalAxisTag) bool {
        for (self.shardableAxes()) |t| {
            if (t == tag) return true;
        }

        return false;
    }

    pub fn format(self: PhysicalMesh, writer: *std.Io.Writer) !void {
        try writer.print("\nPhysicalMesh(platform={s} shardable_axes={any} num_devices={d})\n", .{ @tagName(self.target), self.shardableAxes(), self.countDevices() });
        try writer.print("│  \n", .{});

        try formatNode(self.root, writer, "", 0, true);
    }

    fn formatNode(node: PhysicalNode, writer: *std.Io.Writer, prefix: []const u8, depth: u8, is_last: bool) !void {
        try writer.print("{s}{s}", .{ prefix, if (is_last) "└── " else "├── " });
        switch (node) {
            .leaf => |device| {
                try writer.print("Device(id={d}, coords={any})\n", .{ device.id, device.coords[0..depth] });
            },
            .branch => |b| {
                try writer.print("[{s}] 0..{d} ({s})\n", .{ @tagName(b.tag), b.children.len, @tagName(b.geometry) });

                var buf: [256]u8 = undefined;
                const new_prefix = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, if (is_last) "    " else "│   " }) catch unreachable;

                for (b.children, 0..) |child, i| {
                    const child_is_last = (i == b.children.len - 1);
                    try formatNode(child, writer, new_prefix, depth +| 1, child_is_last);
                }
            },
        }
    }

    /// CoordsTopology builds a mesh from PJRT `coords` attributes.
    const CoordsTopology = struct {
        const CoordLayout = struct {
            rank: usize,
            axis_sizes: [MAX_MESH_RANK]usize,
        };

        const IndexedCoord = struct {
            placement: Device,
            linear: usize,
        };

        fn parseDevice(device: PlatformDevice) !Device {
            const api = device.platform.pjrt_api;
            const coords_attr = device.pjrt_desc.attribute(api, "coords") orelse return error.MissingDeviceCoords;
            const coords: Device.Coords = coords: switch (coords_attr) {
                .int64list => |values| {
                    var coords: Device.Coords = @splat(0);
                    if (values.len == 0 or values.len > MAX_MESH_RANK) return error.InvalidDeviceCoords;
                    for (coords[0..values.len], values) |*coord, value| {
                        if (value < 0) return error.InvalidDeviceCoords;
                        coord.* = @intCast(value);
                    }
                    break :coords coords;
                },
                else => return error.InvalidDeviceCoords,
            };

            return .{ .id = device.id(), .coords = coords };
        }

        pub fn collect(allocator: std.mem.Allocator, platform_devices: []const PlatformDevice) ![]Device {
            const placements = try allocator.alloc(Device, platform_devices.len);
            for (platform_devices, 0..) |device, i| {
                placements[i] = try parseDevice(device);
            }
            return placements;
        }

        pub fn layout(placements: []const Device, max_rank: usize) !CoordLayout {
            if (placements.len == 0) return error.InvalidPhysicalMesh;

            const rank = placements[0].coords.len;
            if (rank == 0) return error.InvalidDeviceCoords;
            if (rank > max_rank) return error.UnsupportedDeviceCoordsRank;

            for (placements[1..]) |coord_placement| {
                if (coord_placement.coords.len != rank) return error.InvalidDeviceCoordsRank;
            }

            var axis_sizes: [MAX_MESH_RANK]usize = @splat(1);
            for (0..rank) |ax_i| {
                var max_coord: usize = 0;
                for (placements) |coord_placement| {
                    max_coord = @max(max_coord, coord_placement.coords[ax_i]);
                }
                axis_sizes[ax_i] = max_coord + 1;
            }

            return .{
                .rank = rank,
                .axis_sizes = axis_sizes,
            };
        }

        fn linearIndex(coords: Device.Coords, rank: usize, axis_sizes: []const usize) usize {
            var linear: usize = 0;
            var stride: usize = 1;
            var ax = rank;
            while (ax > 0) {
                ax -= 1;
                linear += coords[ax] * stride;
                stride *= axis_sizes[ax];
            }
            return linear;
        }

        pub fn sorted(allocator: std.mem.Allocator, placements: []const Device, rank: usize, axis_sizes: []const usize) ![]IndexedCoord {
            const indexed = try allocator.alloc(IndexedCoord, placements.len);
            for (placements, indexed) |coord_placement, *out| {
                out.* = .{
                    .placement = coord_placement,
                    .linear = linearIndex(coord_placement.coords, rank, axis_sizes),
                };
            }

            const SortCtx = struct {
                fn lessThan(_: @This(), a: IndexedCoord, b: IndexedCoord) bool {
                    return a.linear < b.linear;
                }
            };
            std.mem.sort(IndexedCoord, indexed, SortCtx{}, SortCtx.lessThan);

            for (indexed[1..], 1..) |entry, i| {
                if (entry.linear == indexed[i - 1].linear) return error.InvalidDeviceTopology;
            }

            return indexed;
        }

        pub fn axisStrides(axis_sizes: []const usize) ![MAX_MESH_RANK]usize {
            var strides: [MAX_MESH_RANK]usize = @splat(0);
            var stride: usize = 1;

            var i = axis_sizes.len;
            while (i > 0) {
                i -= 1;
                strides[i] = stride;
                stride = std.math.mul(usize, stride, axis_sizes[i]) catch return error.InvalidDeviceTopology;
            }

            return strides;
        }

        pub fn buildMeshNode(
            allocator: std.mem.Allocator,
            indexed: []const IndexedCoord,
            axis_sizes: []const usize,
            axis_tags: []const PhysicalAxisTag,
            axis_strides: []const usize,
            depth: usize,
            base_offset: usize,
        ) !PhysicalNode {
            if (depth == axis_sizes.len) {
                const device = &indexed[base_offset].placement;
                return .{ .leaf = device.* };
            }

            const children = try allocator.alloc(PhysicalNode, axis_sizes[depth]);
            var built: usize = 0;
            errdefer {
                for (children[0..built]) |child| freeNode(allocator, child);
                allocator.free(children);
            }

            for (children, 0..) |*child, i| {
                child.* = try buildMeshNode(
                    allocator,
                    indexed,
                    axis_sizes,
                    axis_tags,
                    axis_strides,
                    depth + 1,
                    base_offset + i * axis_strides[depth],
                );
                built += 1;
            }

            return .{
                .branch = .{
                    .tag = axis_tags[depth],
                    .geometry = .{ .mesh = .torus },
                    .children = children,
                },
            };
        }
    };

    pub fn auto(allocator: std.mem.Allocator, target: Target, platform_devices: []const PlatformDevice) !PhysicalMesh {
        const root = try switch (target) {
            .cpu => cpu(allocator, platform_devices),
            .cuda, .rocm => gpu(allocator, platform_devices),
            .tpu => tpu(allocator, platform_devices),
            .neuron => return neuron(allocator, platform_devices),
            .oneapi, .metal, .furiosa => cpu(allocator, platform_devices),
        };
        errdefer freeNode(allocator, root);

        // We check a posteriori that the device id is a valid offset into platform.devices.
        // This is generally true because platform.devices is sorted by id and ids are 0-N.
        // For platforms where this isn't true like Neuron,
        // the platform specific builder need to correctly update the ids so that it's true.
        const mesh = try fromOwnedTree(allocator, target, root);
        for (mesh.devices_in_canonical_order) |pl| {
            const device = platform_devices[pl.id];
            std.debug.assert(pl.hasValidCoords());
            std.debug.assert(device.id() == pl.id);
        }

        return mesh;
    }

    pub fn cpu(allocator: std.mem.Allocator, platform_devices: []const PlatformDevice) !Tree {
        const nodes = try allocator.alloc(PhysicalNode, platform_devices.len);

        for (nodes, platform_devices) |*n, d| n.* = .device(d);

        return .{
            .branch = .{
                .tag = .bus,
                .geometry = .tree,
                .children = nodes,
            },
        };
    }

    pub fn gpu(allocator: std.mem.Allocator, platform_devices: []const PlatformDevice) !Tree {
        const placements = try CoordsTopology.collect(allocator, platform_devices);
        defer allocator.free(placements);

        const layout = try CoordsTopology.layout(placements, MAX_MESH_RANK);
        const indexed = try CoordsTopology.sorted(allocator, placements, layout.rank, layout.axis_sizes[0..layout.rank]);
        defer allocator.free(indexed);

        const nodes = try allocator.alloc(PhysicalNode, platform_devices.len);
        for (nodes, indexed, 0..) |*n, entry, i| {
            const linear_coord: Device.Coords = .{ @intCast(i), 0, 0, 0 };
            n.* = .{ .leaf = .{ .id = entry.placement.id, .coords = linear_coord } };
        }

        return .{
            .branch = .{
                .tag = .link,
                .geometry = .point_to_point,
                .children = nodes,
            },
        };
    }

    pub fn tpu(allocator: std.mem.Allocator, platform_devices: []const PlatformDevice) !Tree {
        if (platform_devices.len == 0) return error.InvalidPhysicalMesh;

        const placements = try CoordsTopology.collect(allocator, platform_devices);
        defer allocator.free(placements);

        const layout = try CoordsTopology.layout(placements, 4);
        var rank = layout.rank;
        var axis_sizes = layout.axis_sizes;

        while (rank > 1 and axis_sizes[rank - 1] == 1) : (rank -= 1) {}
        const indexed = try CoordsTopology.sorted(allocator, placements, rank, axis_sizes[0..rank]);
        defer allocator.free(indexed);

        var total_devices: usize = 1;
        for (axis_sizes[0..rank]) |axis_size| {
            total_devices = std.math.mul(usize, total_devices, axis_size) catch return error.InvalidDeviceTopology;
        }
        if (total_devices != indexed.len) return error.InvalidDeviceTopology;
        for (indexed, 0..) |entry, i| {
            if (entry.linear != i) return error.InvalidDeviceTopology;
        }

        const axis_tags: [3]PhysicalAxisTag = .{ .link_x, .link_y, .link_z };
        const axis_strides = try CoordsTopology.axisStrides(axis_sizes[0..rank]);

        return CoordsTopology.buildMeshNode(allocator, indexed, axis_sizes[0..rank], axis_tags[0..rank], axis_strides[0..rank], 0, 0);
    }

    pub fn neuron(allocator: std.mem.Allocator, platform_devices: []const PlatformDevice) !PhysicalMesh {
        if (comptime !platforms.isEnabled(.neuron)) {
            return error.UnsupportedPlatform;
        }

        const neuron_topology = @import("platforms/neuron/topology");

        const nc_ids = try allocator.alloc(usize, platform_devices.len);
        defer allocator.free(nc_ids);
        for (platform_devices, nc_ids) |device, *nc_id| {
            nc_id.* = @intCast(device.localHardwareId());
        }

        var topology = try neuron_topology.visibleMeshFromNcIds(allocator, nc_ids);
        defer topology.deinit(allocator);

        var axis_tags: stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK) = .empty;
        var axis_sizes: stdx.BoundedArray(usize, MAX_MESH_RANK) = .empty;
        var axis_geometries: stdx.BoundedArray(AxisGeometry, MAX_MESH_RANK) = .empty;

        for (topology.axes()) |mesh_axis| {
            axis_tags.appendAssumeCapacity(switch (mesh_axis.fabric) {
                .core => .link,
                .chip_x => .link_x,
                .chip_y => .link_y,
                .chip_z => .link_z,
            });
            axis_sizes.appendAssumeCapacity(mesh_axis.size);
            axis_geometries.appendAssumeCapacity(switch (mesh_axis.geometry) {
                .linear_ring => .{ .ring = .linear },
                .closed_ring => .{ .ring = .closed_ring },
                .torus => .{ .mesh = .torus },
                .point_to_point => .point_to_point,
            });
        }

        var coords_buf: [MAX_MESH_RANK]usize = @splat(0);
        var next_device: usize = 0;

        const Node = struct {
            fn build(
                allocator_: std.mem.Allocator,
                platform_devices_: []const PlatformDevice,
                placements: []const neuron_topology.Placement,
                axis_tags_: []const PhysicalAxisTag,
                axis_sizes_: []const usize,
                axis_geometries_: []const AxisGeometry,
                depth: usize,
                coords_buf_: *[MAX_MESH_RANK]usize,
                next_device_: *usize,
            ) !PhysicalNode {
                if (depth == axis_tags_.len) {
                    const placement_ = placements[next_device_.*];
                    next_device_.* += 1;

                    var coords: Device.Coords = @splat(0);
                    for (coords_buf_[0..axis_tags_.len], 0..) |coord, i| coords[i] = @intCast(coord);

                    return .{ .leaf = .{ .id = @intCast(placement_.nc_id), .coords = coords } };
                }

                const children = try allocator_.alloc(PhysicalNode, axis_sizes_[depth]);
                var built: usize = 0;
                errdefer {
                    for (children[0..built]) |child| freeNode(allocator_, child);
                    allocator_.free(children);
                }

                for (children, 0..) |*child, i| {
                    coords_buf_[depth] = i;
                    child.* = try @This().build(
                        allocator_,
                        platform_devices_,
                        placements,
                        axis_tags_,
                        axis_sizes_,
                        axis_geometries_,
                        depth + 1,
                        coords_buf_,
                        next_device_,
                    );
                    built += 1;
                }

                return .{
                    .branch = .{
                        .tag = axis_tags_[depth],
                        .geometry = axis_geometries_[depth],
                        .children = children,
                    },
                };
            }
        };

        const root = try Node.build(
            allocator,
            platform_devices,
            topology.placements,
            axis_tags.slice(),
            axis_sizes.slice(),
            axis_geometries.slice(),
            0,
            &coords_buf,
            &next_device,
        );
        const mesh: PhysicalMesh = try fromOwnedTree(allocator, .neuron, root);
        for (mesh.devices_in_canonical_order) |*dev| {
            // In Node.build we use nc_id as the Device id.
            // But now we want to use the id to get the offset into platform.devices.
            const dev_nc_id = dev.id;
            for (0.., topology.placements) |order, placement_| {
                if (placement_.nc_id == dev_nc_id) {
                    dev.id = @intCast(order);
                    std.debug.assert(platform_devices[order].localHardwareId() == dev_nc_id);
                    break;
                }
            }
        }
        return mesh;
    }
};

pub const LogicalAxisIntent = enum {
    /// Needs the fastest possible path (e.g. Weight mesh / All-Reduce).
    /// Requires high-bandwidth, low-latency interconnects (Cores/NVLink).
    high_bandwidth,

    /// Needs reliable throughput but can handle some latency (e.g. Shuffling / All-to-All).
    /// Suitable for chip-to-chip interconnects (NeuronLink/ICI).
    balanced,

    /// Can tolerate lower bandwidth and higher latency (e.g. Batch mesh).
    /// Suitable for system bus or network (PCIe/Ethernet).
    low_bandwidth,
};

/// Semantic axes, their communication intents, and optional device-count constraints.
pub const LogicalMesh = struct {
    pub const Axes = stdx.BoundedArray(Shape.Tag, MAX_MESH_RANK);
    pub const Intents = stdx.BoundedArray(LogicalAxisIntent, MAX_MESH_RANK);

    axes: Axes,
    intents: Intents,
    dimensions: [MAX_MESH_RANK]i32,

    /// Create a logical mesh.
    /// * names of the logical axes: .{ .a, .b },
    /// * intents: intent for each axis
    /// * dimensions: dimension for each axis, -1 infers the size from underlying topology
    pub fn init(names: anytype, intents: []const LogicalAxisIntent, dimensions: []const i32) !LogicalMesh {
        if (@TypeOf(names) != []const Shape.Tag) {
            const parsed_names = comptime Shape.parseTags(names);
            return .init(@as([]const Shape.Tag, parsed_names.slice()), intents, dimensions);
        }

        if (names.len == 0 or names.len > MAX_MESH_RANK or names.len != intents.len or names.len != dimensions.len) @panic("invalid logical mesh");

        var result: LogicalMesh = .{ .axes = .init(names), .intents = .init(intents), .dimensions = @splat(-1) };
        for (names, dimensions, 0..) |name, dimension, i| {
            if (dimension != -1 and dimension <= 0) return error.InvalidLogicalDimension;
            if (std.mem.findScalar(Shape.Tag, names[0..i], name)) |_| {
                @panic("invalid logical mesh");
            }

            result.dimensions[i] = dimension;
        }
        return result;
    }

    pub fn mesh(axes_: anytype) LogicalMesh {
        const T = @TypeOf(axes_);

        var axes: Axes = .empty;
        var intents: Intents = .empty;

        inline for (comptime std.meta.fieldNames(T)) |field_name| {
            const value = @field(axes_, field_name);
            axes.appendAssumeCapacity(Shape.toTag(field_name));
            intents.appendAssumeCapacity(intentFromValue(value));
        }

        if (axes.len == 0) {
            stdx.debug.panic("LogicalMesh must have at least one axis defined", .{});
        }

        return .{ .axes = axes, .intents = intents, .dimensions = @splat(-1) };
    }

    pub fn intent(self: LogicalMesh, tag: Shape.Tag) ?LogicalAxisIntent {
        const target = std.mem.span(tag);
        for (self.axes.slice(), self.intents.slice()) |t, i| {
            if (std.mem.eql(u8, std.mem.span(t), target)) return i;
        }

        return null;
    }

    fn intentFromValue(v: anytype) LogicalAxisIntent {
        const V = @TypeOf(v);
        if (V == LogicalAxisIntent) return v;

        if (V == @EnumLiteral()) {
            return @field(LogicalAxisIntent, @tagName(v));
        }

        stdx.debug.compileError("LogicalMesh intent must be LogicalAxisIntent, got {}", .{V});
    }

    pub fn format(self: LogicalMesh, writer: *std.Io.Writer) !void {
        try writer.writeAll("LogicalMesh(");
        for (self.axes.slice(), self.intents.slice(), 0..) |axis, int, i| {
            try writer.print(" {s}={s}", .{ axis, @tagName(int) });
            if (self.dimensions[i] != -1) try writer.print("[{d}]", .{self.dimensions[i]});
        }
        try writer.writeAll(")");
    }
};

pub const PhysicalList = stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK);

pub const Mesh = struct {
    pub const zml_no_meta_visit: void = {};

    name: []const u8,
    physical: *const PhysicalMesh,
    logical: LogicalMesh,

    grid: DeviceGrid,

    /// Compact binding table: logical axis -> physical axes
    bindings: stdx.BoundedArray(PhysicalList, MAX_MESH_RANK),

    pub fn shardableDim(mesh: *const Mesh, dim: i64, logical_axis: anytype, must_divide: i64) DimSharding {
        const partitions = mesh.numPartitionsForLogicalAxis(logical_axis);

        if (@mod(dim, partitions) == 0) {
            return if (@mod(must_divide, dim) == 0) .{ .sharded = .{ .dim = dim, .factor = 1 } } else .replicated;
        }

        const gcd: i64 = @intCast(std.math.gcd(@as(u64, @intCast(dim)), @as(u64, @intCast(partitions))));
        const repeat_factor: u32 = @intCast(@divExact(partitions, gcd));
        const materialized_dim = std.math.mul(i64, dim, @as(i64, repeat_factor)) catch return .replicated;
        if (@mod(must_divide, materialized_dim) == 0) {
            return .{ .sharded = .{ .dim = materialized_dim, .factor = repeat_factor } };
        } else {
            return .replicated;
        }
    }

    pub fn binding(self: *const Mesh, mesh_axis: usize) []const PhysicalAxisTag {
        return self.bindings.constSlice()[mesh_axis].constSlice();
    }

    pub fn resolveLogicalAxis(self: *const Mesh, tag: Shape.Tag) ?u8 {
        if (@inComptime()) {
            for (0.., self.logical.axes.slice()) |i, logical_axis| {
                if (std.mem.eql(u8, std.mem.span(logical_axis), std.mem.span(tag))) return @truncate(i);
            }
        } else {
            for (0.., self.logical.axes.slice()) |i, logical_axis| {
                if (logical_axis == tag) return @truncate(i);
            }
        }
        return null;
    }

    pub fn init(
        name_: []const u8,
        physical: *const PhysicalMesh,
        logical: LogicalMesh,
        strategy: Strategy,
    ) !Mesh {
        const num_devices = physical.countDevices();
        std.debug.assert(num_devices == physical.devices_in_canonical_order.len);
        const axis_order = physical.axisOrder().slice();
        if (axis_order.len == 0) return error.InvalidPhysicalMesh;
        if (strategy.bindings.len == 0) return error.InvalidStrategy;

        // Create one binding per logical axis
        var bindings: @FieldType(Mesh, "bindings") = .empty;
        for (logical.axes.slice()) |_| {
            bindings.appendAssumeCapacity(.empty);
        }

        for (strategy.bindings.slice()) |strat_binding| {
            for (logical.axes.slice(), bindings.slice()) |logical_axis, *binding_| {
                if (std.mem.eql(u8, std.mem.span(logical_axis), std.mem.span(strat_binding.logical))) {
                    for (strat_binding.physical.constSlice()) |part| binding_.appendAssumeCapacity(part.axis);
                    break;
                }
            } else {
                std.debug.panic("{f} expects strategy to only list known logical axes, got unknown {s}", .{ logical, strat_binding.logical });
            }
        }

        var mesh: Mesh = .{
            .name = name_,
            .logical = logical,
            .grid = undefined,
            .physical = physical,
            .bindings = bindings,
        };
        mesh.grid = try DeviceGrid.init(&mesh, &strategy);
        std.debug.assert(num_devices == physical.devices_in_canonical_order.len);
        std.debug.assert(@mod(mesh.numDevices(), mesh.numPartitions()) == 0);
        return mesh;
    }

    pub fn initTest(name_: @EnumLiteral(), tagz: anytype) Mesh {
        if (!builtin.is_test) @compileError("Cannot use zml.Sharding.Mesh.initTest outside of a test block");
        const axes_tags = Shape.parseTags(tagz);
        return .{
            .name = @tagName(name_),
            .logical = .{
                .axes = .init(axes_tags.slice()),
                .intents = .{ .buffer = @splat(.high_bandwidth), .len = axes_tags.len },
            },
            .physical = undefined,
            .grid = undefined,
            .bindings = .empty,
        };
    }

    pub fn devicesInCanonicalOrder(mesh: *const Mesh) []const Device {
        return mesh.physical.devices_in_canonical_order;
    }

    pub fn logicalIndexFromCoords(self: *const Mesh, coords: []const u8) usize {
        return self.physical.linearIndexFromCoords(coords);
    }

    pub fn numPartitions(self: *const Mesh) u32 {
        var partitions: u32 = 1;
        for (self.grid.axis_sizes[0..self.grid.rank]) |size| partitions *= size;
        return @intCast(partitions);
    }

    pub fn numPartitionsForLogicalAxis(self: *const Mesh, logical_axis: anytype) u32 {
        const logical_tag = Shape.toTag(logical_axis);

        const logical_ax = self.resolveLogicalAxis(logical_tag) orelse std.debug.panic("mesh {f} has no axis {s}", .{ self, logical_tag });
        return self.grid.axis_sizes[logical_ax];
    }

    pub fn numDevices(self: *const Mesh) u32 {
        return @intCast(self.physical.devices_in_canonical_order.len);
    }

    pub fn numReplicas(self: *const Mesh) u32 {
        return @divExact(self.numDevices(), self.numPartitions());
    }

    pub fn sdyMeshAttr(self: *const Mesh, allocator: std.mem.Allocator) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        try out.writer.writeAll("#sdy.mesh<[");
        var tile_count: usize = 1;
        for (self.logical.axes.slice(), 0..) |axis, i| {
            if (i > 0) try out.writer.writeAll(", ");
            const size = self.grid.axis_sizes[i];
            tile_count *= size;
            try out.writer.print("\"{s}\"={d}", .{ std.mem.span(axis), size });
        }
        const replicas = @divExact(self.grid.device_count, tile_count);
        if (replicas > 1) {
            try out.writer.print(", \"{s}\"={d}", .{ self.sdyReplicaAxis(), replicas });
        }
        try out.writer.writeAll("], device_ids=[");
        var replica_indices: [Platform.MAX_NUM_DEVICES]usize = @splat(0);
        var ids: [Platform.MAX_NUM_DEVICES]usize = undefined;
        for (0..self.grid.device_count) |partition_index| {
            const coords: [MAX_MESH_RANK]u8 = self.grid.logical_coords[partition_index];
            var tile_index: usize = 0;
            for (0..self.logical.axes.len) |axis| {
                tile_index = tile_index * self.grid.axis_sizes[axis] + coords[axis];
            }
            // Device IDs here are executable partition indices, not PJRT device IDs.
            ids[tile_index * replicas + replica_indices[tile_index]] = partition_index;
            replica_indices[tile_index] += 1;
        }
        for (ids[0..self.grid.device_count], 0..) |id, i| {
            if (i > 0) try out.writer.writeAll(", ");
            try out.writer.print("{d}", .{id});
        }
        try out.writer.writeAll("]>");
        return try out.toOwnedSlice();
    }

    fn sdyReplicaAxis(self: *const Mesh) []const u8 {
        // There are at most MAX_MESH_RANK logical names, so one candidate is free.
        inline for (0..MAX_MESH_RANK + 1) |i| {
            const name = std.fmt.comptimePrint("__replica_{d}", .{i});
            const taken = for (self.logical.axes.slice()) |axis| {
                if (std.mem.eql(u8, std.mem.span(axis), name)) break true;
            } else false;
            if (!taken) return name;
        }
        unreachable;
    }

    pub fn sdyShardingAttrForShape(
        data: *const Mesh,
        parent_allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        shape: Shape,
    ) error{OutOfMemory}!*const dialects.shardy.TensorShardingAttribute {
        var arena = try stdx.arenaWithCapacity(parent_allocator, 1024);
        defer arena.deinit();
        const allocator = arena.allocator();
        const dimensions = try allocator.alloc(*const dialects.shardy.DimensionShardingAttribute, shape.rank());
        var used: [MAX_MESH_RANK]bool = @splat(false);
        for (0.., dimensions) |ax, *d| {
            d.* = switch (shape.partition(ax)) {
                .out_of_bound => @panic("undefined"),
                .replicated => .replicated(ctx),
                .open => .open(ctx, &.{}),
                else => d: {
                    const axis = shape.partition(ax).meshAxis().?;
                    used[axis] = true;
                    break :d .closed(ctx, &.{.named(ctx, std.mem.span(data.logical.axes.get(axis)))});
                },
            };
        }

        var replicated_axes: stdx.BoundedArray(*const dialects.shardy.AxisRefAttribute, MAX_MESH_RANK + 1) = .empty;
        var tile_count: usize = 1;
        for (data.logical.axes.slice(), 0..) |axis, i| {
            tile_count *= data.grid.axis_sizes[i];
            if (!used[i]) replicated_axes.appendAssumeCapacity(.named(ctx, std.mem.span(axis)));
        }
        if (tile_count < data.grid.device_count) {
            replicated_axes.appendAssumeCapacity(.named(ctx, data.sdyReplicaAxis()));
        }

        return .init(ctx, .{ .mesh = data.name, .dimensions = dimensions, .replicated_axes = replicated_axes.slice() });
    }

    pub fn gspmdShardingAttrForShape(self: *const Mesh, parent_allocator: std.mem.Allocator, ctx: *mlir.Context, shape: Shape) !*const mlir.Attribute {
        var arena = try stdx.arenaWithCapacity(parent_allocator, 1024);
        defer arena.deinit();
        const allocator = arena.allocator();

        var tile_shape: stdx.BoundedArray(i64, Shape.MAX_RANK + 1) = .empty;
        var tile_count: usize = 1;
        for (0..shape.rank()) |ax| {
            const size = if (shape.partition(ax).meshAxis()) |axis| self.grid.axis_sizes[axis] else 1;
            tile_shape.appendAssumeCapacity(size);
            tile_count *= size;
        }
        if (tile_count == 1) return .string(ctx, "{replicated}");

        // All devices at the same logical tile coordinate are replicas, including
        // unused logical axes and physical axes that have no logical binding.
        const replicas = @divExact(self.grid.device_count, tile_count);
        const has_replication = replicas > 1;
        if (has_replication) tile_shape.appendAssumeCapacity(@intCast(replicas));

        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        const ids = try self.gspmdTileAssignment(allocator, shape);
        defer allocator.free(ids);

        try out.writer.writeAll("{devices=[");
        for (tile_shape.slice(), 0..) |s, i| {
            if (i > 0) try out.writer.writeAll(",");
            try out.writer.print("{d}", .{s});
        }
        try out.writer.writeAll("]");

        for (ids, 0..) |id, i| {
            if (i == 0) {
                try out.writer.print("{d}", .{id});
            } else {
                try out.writer.print(",{d}", .{id});
            }
        }

        if (has_replication) {
            try out.writer.writeAll(" last_tile_dim_replicate");
        }

        try out.writer.writeAll("}");

        return .string(ctx, try out.toOwnedSlice());
    }

    fn gspmdTileAssignment(self: *const Mesh, allocator: std.mem.Allocator, shape: Shape) ![]u32 {
        _ = try Sharding.Placement.init(&shape);
        const devices = self.devicesInCanonicalOrder();
        var tile_axes: stdx.BoundedArray(u8, MAX_MESH_RANK) = .empty;
        var used_axes: u8 = 0;
        var tile_count: usize = 1;
        for (0..shape.rank()) |ax| {
            const logical_axis = shape.partition(ax).meshAxis() orelse continue;
            const bit = @as(u8, 1) << @intCast(logical_axis);
            if (used_axes & bit != 0) continue;
            used_axes |= bit;
            tile_axes.appendAssumeCapacity(@intCast(logical_axis));
            tile_count *= self.grid.axis_sizes[logical_axis];
        }
        const replicas_per_tile = @divExact(devices.len, tile_count);
        const replica_indices = try allocator.alloc(usize, tile_count);
        defer allocator.free(replica_indices);
        @memset(replica_indices, 0);
        const ids = try allocator.alloc(u32, devices.len);

        for (0..devices.len) |partition_index| {
            const coords: [MAX_MESH_RANK]u8 = self.grid.logical_coords[partition_index];
            var tile_index: usize = 0;
            for (tile_axes.constSlice()) |logical_axis| {
                tile_index = tile_index * self.grid.axis_sizes[logical_axis] +
                    coords[logical_axis];
            }
            // HLO shardings refer to logical partitions; executable device assignment
            // separately maps those partitions to physical PJRT device IDs.
            ids[tile_index * replicas_per_tile + replica_indices[tile_index]] = @intCast(partition_index);
            replica_indices[tile_index] += 1;
        }
        return ids;
    }

    pub fn deviceAssignment(self: *const Mesh, allocator: std.mem.Allocator) ![]u32 {
        const count = self.physical.devices_in_canonical_order.len;

        var ids = try allocator.alloc(u32, count);
        @memset(ids, std.math.maxInt(u32));

        for (self.physical.devices_in_canonical_order) |d| {
            const coords = d.coords;
            const idx = self.physical.linearIndexFromCoords(coords);
            ids[idx] = d.id;
        }

        for (ids) |id| {
            if (id == std.math.maxInt(u32)) return error.MissingDeviceInTile;
        }

        return ids;
    }

    pub fn format(self: Mesh, writer: *std.Io.Writer) !void {
        try writer.print("Mesh(name={s}, partitions={d}, replicas={d})\n", .{ self.name, self.numPartitions(), self.numReplicas() });

        try writer.writeAll("Mappings (logical coordinates from physical coordinates):\n");
        for (self.logical.axes.slice(), self.logical.intents.slice(), 0..) |tag, intent, logical_axis| {
            try writer.print("  - {s} ({s}) -> ", .{ tag, @tagName(intent) });
            try self.formatAxisMapping(writer, logical_axis);
            try writer.writeAll("\n");
        }

        try writer.print("{f}\n", .{self.physical});

        try writer.writeAll("Logical coordinates:\n");
        var coords: [MAX_MESH_RANK]usize = @splat(0);
        try self.formatLogicalCoordinates(writer, &coords, 0);
    }

    /// Reverse engineer the mapping logical -> physical from the device grid.
    fn formatAxisMapping(self: Mesh, writer: *std.Io.Writer, logical_axis: usize) !void {
        const Part = struct {
            tag: PhysicalAxisTag,
            physical_size: usize,
            physical_stride: usize,
            size: usize,
            logical_stride: usize,
        };
        // A u8 coordinate has at most eight nontrivial mixed-radix digits.
        var parts: stdx.BoundedArray(Part, @bitSizeOf(u8)) = .empty;
        const logical_size = self.grid.axis_sizes[logical_axis];
        var logical_stride: usize = 1;
        while (logical_stride < logical_size) {
            const remaining = @divExact(logical_size, logical_stride);
            var size = remaining;
            const part: Part = found: {
                // Prefer the largest matching portion, so whole axes stay readable.
                while (size > 1) : (size -= 1) {
                    if (remaining % size != 0) continue;
                    for (self.physical.axisOrder().constSlice()) |tag| {
                        const physical_size: usize = @intCast(self.physical.axis(tag));
                        const depth = self.physical.axis_traversal.depth(tag).?;
                        var physical_stride: usize = 1;
                        while (physical_stride * size <= physical_size) : (physical_stride += 1) {
                            if (physical_size % (physical_stride * size) != 0) continue;
                            for (self.devicesInCanonicalOrder(), 0..) |device, i| {
                                const coords: [MAX_MESH_RANK]u8 = self.grid.logical_coords[i];
                                if ((coords[logical_axis] / logical_stride) % size !=
                                    (device.coords[depth] / physical_stride) % size) break;
                            } else {
                                break :found .{
                                    .tag = tag,
                                    .physical_size = physical_size,
                                    .physical_stride = physical_stride,
                                    .size = size,
                                    .logical_stride = logical_stride,
                                };
                            }
                        }
                    }
                }
                try writer.writeAll("device-dependent (see logical coordinates)");
                return;
            };
            parts.appendAssumeCapacity(part);
            logical_stride *= part.size;
        }
        if (parts.len == 0) {
            try writer.writeAll("0 (replicated)");
            return;
        }
        var i: usize = parts.len;
        while (i > 0) {
            i -= 1;
            const part = parts.constSlice()[i];
            if (i + 1 < parts.len) try writer.writeAll(" + ");
            const partial = part.size != part.physical_size;
            if (partial) try writer.writeAll("(");
            try writer.writeAll(@tagName(part.tag));
            if (part.physical_stride > 1) try writer.print(" / {d}", .{part.physical_stride});
            if (part.physical_stride * part.size < part.physical_size) try writer.print(" % {d}", .{part.size});
            if (partial) try writer.writeAll(")");
            if (part.logical_stride > 1) try writer.print(" * {d}", .{part.logical_stride});
        }
    }

    fn formatLogicalCoordinates(self: Mesh, writer: *std.Io.Writer, coords: *[MAX_MESH_RANK]usize, depth: usize) std.Io.Writer.Error!void {
        const axes = self.logical.axes.constSlice();
        if (axes.len == 0) return;
        const last = axes.len - 1;
        if (depth < last) {
            const size: usize = @intCast(self.numPartitionsForLogicalAxis(axes[depth]));
            for (0..size) |coord| {
                coords[depth] = coord;
                try writer.splatByteAll(' ', 2 + 4 * depth);
                try writer.print("- {s}={d}", .{ axes[depth], coord });
                if (depth + 1 < last) {
                    try writer.writeAll("\n");
                } else {
                    try writer.writeAll(", ");
                }
                try self.formatLogicalCoordinates(writer, coords, depth + 1);
            }
            return;
        }

        if (last == 0) try writer.writeAll("  - ");
        try writer.print("{s}=", .{axes[last]});
        const size: usize = @intCast(self.numPartitionsForLogicalAxis(axes[last]));
        for (0..size) |coord| {
            if (coord > 0) try writer.writeAll(",");
            try writer.print("{d}", .{coord});
        }
        try writer.writeAll(" -> devices=");
        for (0..size) |coord| {
            coords[last] = coord;
            if (coord > 0) try writer.writeAll(",");
            var count: usize = 0;
            for (0..self.grid.device_count) |device_index| {
                if (self.matchesLogicalCoordinates(device_index, coords[0..axes.len])) count += 1;
            }
            // Keep replicas grouped so each entry corresponds to one last-axis coordinate.
            if (count != 1) try writer.writeAll("[");
            var written: usize = 0;
            for (self.devicesInCanonicalOrder(), 0..) |device, device_index| {
                if (!self.matchesLogicalCoordinates(device_index, coords[0..axes.len])) continue;
                if (written > 0) try writer.writeAll(",");
                try writer.print("{d}", .{device.id});
                written += 1;
            }
            if (count != 1) try writer.writeAll("]");
        }
        try writer.writeAll("\n");
    }

    fn matchesLogicalCoordinates(self: Mesh, device_index: usize, coords: []const usize) bool {
        const actual: [MAX_MESH_RANK]u8 = self.grid.logical_coords[device_index];
        for (coords, 0..) |expected, logical_axis| {
            if (actual[logical_axis] != expected) return false;
        }
        return true;
    }
};

/// Resolved logical coordinates in canonical physical-device order.
/// Unused logical axes have coordinate zero.
pub const DeviceGrid = struct {
    pub const Coordinates = @Vector(MAX_MESH_RANK, u8);

    comptime {
        std.debug.assert(Platform.MAX_NUM_DEVICES <= 256);
    }

    rank: u8,
    device_count: usize,
    axis_sizes: [MAX_MESH_RANK]u32,
    logical_coords: [Platform.MAX_NUM_DEVICES]Coordinates align(16),

    fn init(mesh_: *const Mesh, strategy: *const Strategy) !DeviceGrid {
        for (strategy.folding.constSlice()) |entry| {
            for (entry.sources.constSlice()) |source| {
                if (!mesh_.physical.hasAxis(source)) return error.InvalidPhysicalAxis;
            }
        }
        const devices = mesh_.devicesInCanonicalOrder();
        if (devices.len > Platform.MAX_NUM_DEVICES) return error.InvalidPhysicalMesh;
        var grid: DeviceGrid = .{
            .rank = @intCast(mesh_.logical.axes.len),
            .device_count = devices.len,
            .axis_sizes = @splat(1),
            .logical_coords = @splat(@splat(0)),
        };
        for (mesh_.logical.axes.constSlice(), 0..) |logical_tag, logical_axis| {
            var used: std.EnumSet(PhysicalAxisTag) = .empty;
            for (strategy.bindings.constSlice()) |binding_| {
                if (!std.mem.eql(u8, std.mem.span(binding_.logical), std.mem.span(logical_tag))) continue;
                for (binding_.physical.constSlice()) |part| {
                    if (part.size) |size| {
                        const info = mesh_.physical.axisInfo(part.axis) orelse return error.InvalidPhysicalAxis;
                        const physical_size: u32 = @intCast(info.size);
                        if (size == 0 or part.stride == 0 or size > physical_size or part.stride > physical_size / size) return error.InvalidPartialBinding;
                        if (physical_size % (part.stride * size) != 0) return error.InvalidPartialBinding;
                        try grid.appendAxis(mesh_, logical_axis, part.axis, part.stride, size);
                    } else {
                        for (strategy.foldSources(part.axis) orelse &.{part.axis}) |source| {
                            if (used.contains(source)) continue;
                            used.insert(source);
                            const info = mesh_.physical.axisInfo(source) orelse continue;
                            try grid.appendAxis(mesh_, logical_axis, source, 1, @intCast(info.size));
                        }
                    }
                }
            }
            const dimension = mesh_.logical.dimensions[logical_axis];
            if (dimension != -1) {
                if (dimension <= 0) return error.InvalidLogicalDimension;
                const size: u32 = @intCast(dimension);
                if (grid.axis_sizes[logical_axis] % size != 0) return error.IncompatibleLogicalDimension;
                grid.axis_sizes[logical_axis] = size;
                // Keep the least-significant portion of the ordered binding; the rest replicates.
                for (grid.logical_coords[0..devices.len]) |*row| {
                    var coords: [MAX_MESH_RANK]u8 = row.*;
                    coords[logical_axis] = @intCast(coords[logical_axis] % size);
                    row.* = coords;
                }
            }
        }
        // Shared physical axes are valid only when their portions produce a complete
        // Cartesian logical grid with the same number of replicas at every coordinate.
        var partitions: usize = 1;
        for (grid.axis_sizes[0..grid.rank]) |size| {
            if (size > devices.len / partitions) return error.OverlappingLogicalAxes;
            partitions *= size;
        }
        if (devices.len % partitions != 0) return error.OverlappingLogicalAxes;
        var counts: [Platform.MAX_NUM_DEVICES]usize = @splat(0);
        for (grid.logical_coords[0..devices.len]) |row| {
            const coords: [MAX_MESH_RANK]u8 = row;
            var index: usize = 0;
            for (coords[0..grid.rank], grid.axis_sizes[0..grid.rank]) |coord, size| index = index * size + coord;
            counts[index] += 1;
        }
        for (counts[0..partitions]) |count| {
            if (count != devices.len / partitions) return error.OverlappingLogicalAxes;
        }
        return grid;
    }

    fn appendAxis(self: *DeviceGrid, mesh_: *const Mesh, logical_axis: usize, tag: PhysicalAxisTag, stride: u32, size: u32) !void {
        if (size > Platform.MAX_NUM_DEVICES / self.axis_sizes[logical_axis]) return error.InvalidPartialBinding;
        const depth = mesh_.physical.axis_traversal.depth(tag).?;
        self.axis_sizes[logical_axis] *= size;
        for (mesh_.devicesInCanonicalOrder(), 0..) |device, i| {
            var coords: [MAX_MESH_RANK]u8 = self.logical_coords[i];
            coords[logical_axis] = @intCast(@as(u32, coords[logical_axis]) * size + (device.coords[depth] / stride) % size);
            self.logical_coords[i] = coords;
        }
    }
};

pub const Strategy = struct {
    pub const Fold = struct {
        target: PhysicalAxisTag,
        sources: PhysicalList,
    };
    pub const PhysicalBinding = struct {
        axis: PhysicalAxisTag,
        stride: u32 = 1,
        /// null binds the whole axis (including any explicit fold).
        size: ?u32 = null,
    };
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: stdx.BoundedArray(PhysicalBinding, MAX_MESH_RANK),
    };

    pub const Bindings = stdx.BoundedArray(Binding, MAX_MESH_RANK);
    pub const Folding = stdx.BoundedArray(Strategy.Fold, MAX_MESH_RANK);

    bindings: Bindings,
    folding: Folding,

    // TODO: remove .init, promote .parseBindings to .init
    // It's currently an error to call .init and never calling `addBinding`
    pub const init: Strategy = .{ .bindings = .empty, .folding = .empty };

    pub fn parseBindings(bindings: anytype) Strategy {
        const err_msg = "Strategy.parseBindings excepts fields to be PhysicalAxisTag or tuple of PhysicalAxisTag, got {}";
        var res: Strategy = .{ .bindings = .empty, .folding = .empty };
        const info = @typeInfo(@TypeOf(bindings)).@"struct";
        if (info.field_names.len == 0) @compileError("Strategy.parseBindings requires at least one binding");
        inline for (info.field_names, info.field_types) |field_name, Field| {
            switch (@typeInfo(Field)) {
                .enum_literal, .@"enum" => res.addBinding(field_name, @field(bindings, field_name)),
                .@"struct" => |struct_info| {
                    stdx.debug.assertComptime(struct_info.is_tuple, err_msg, .{@TypeOf(bindings)});
                    inline for (@field(bindings, field_name)) |axis_tag| {
                        res.addBinding(field_name, axis_tag);
                    }
                },
                else => stdx.debug.compileError(err_msg, .{@TypeOf(bindings)}),
            }
        }
        return res;
    }

    pub fn addBinding(self: *Strategy, logical: anytype, physical: PhysicalAxisTag) void {
        self.appendBinding(Shape.toTag(logical), .{ .axis = physical });
    }

    /// Append the mixed-radix portion `(coordinate / stride) % size` of a physical axis.
    /// `stride * size` must divide the physical axis size. Partial bindings do not expand folds.
    /// Portions are concatenated in insertion order, most significant first.
    pub fn addPartialBinding(self: *Strategy, logical: anytype, physical: PhysicalAxisTag, stride: u32, size: u32) void {
        self.appendBinding(Shape.toTag(logical), .{ .axis = physical, .stride = stride, .size = size });
    }

    fn appendBinding(self: *Strategy, logical: Shape.Tag, physical: PhysicalBinding) void {
        for (self.bindings.slice()) |*binding_| {
            if (std.mem.eql(u8, std.mem.span(binding_.logical), std.mem.span(logical))) {
                binding_.physical.appendAssumeCapacity(physical);
                return;
            }
        }
        self.bindings.appendAssumeCapacity(.{ .logical = logical, .physical = .init(&.{physical}) });
    }

    fn foldSources(self: *const Strategy, tag: PhysicalAxisTag) ?[]const PhysicalAxisTag {
        for (self.folding.constSlice()) |*fold| {
            if (fold.target == tag) return fold.sources.constSlice();
        }
        return null;
    }

    /// Explicitly fold axes: `target` is the kept axis, `sources` define order.
    /// If `target` is missing from `sources`, it is prepended.
    pub fn addFold(self: *Strategy, target: PhysicalAxisTag, sources: []const PhysicalAxisTag) void {
        var list: PhysicalList = .empty;

        var has_target = false;
        for (sources) |s| {
            if (s == target) {
                has_target = true;
                break;
            }
        }

        if (!has_target) list.appendAssumeCapacity(target);
        for (sources) |s| list.appendAssumeCapacity(s);

        for (self.folding.slice()) |*f| {
            if (f.target == target) {
                f.sources = list;
                return;
            }
        }

        self.folding.appendAssumeCapacity(.{ .target = target, .sources = list });
    }

    /// suggest builds a Strategy from logical intents.
    pub fn suggest(logical: LogicalMesh, physical: *const PhysicalMesh) Strategy {
        var strategy: Strategy = .{ .bindings = .empty, .folding = .empty };

        const base_order = physical.shardableAxes();
        var available: stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK) = .empty;
        for (base_order) |tag| {
            if (physical.hasAxis(tag)) available.appendAssumeCapacity(tag);
        }

        if (available.len == 0) {
            for (physical.axisOrder().slice()) |tag| available.appendAssumeCapacity(tag);
        }

        const avail = available.constSlice();
        if (avail.len == 0) return strategy;
        var remaining: [MAX_MESH_RANK]u32 = @splat(1);
        for (avail, 0..) |tag, i| remaining[i] = @intCast(physical.axis(tag));

        // Reserve fixed dimensions before automatic axes can consume their capacity.
        // Carve fixed portions from the most-significant end, leaving a contiguous
        // low-order portion for automatic axes (e.g. model parallelism on one link).
        for ([_]bool{ true, false }) |fixed| {
            var counters: std.EnumArray(LogicalAxisIntent, u32) = .initFill(0);
            inline for (std.meta.tags(LogicalAxisIntent)) |target_intent| {
                for (logical.axes.constSlice(), logical.intents.constSlice(), 0..) |tag, intent, logical_axis| {
                    if (intent != target_intent) continue;
                    const i = counters.get(target_intent);
                    counters.set(target_intent, @intCast(i + 1));
                    const dimension = logical.dimensions[logical_axis];
                    if ((dimension != -1) != fixed) continue;
                    const preferred = switch (target_intent) {
                        .high_bandwidth => i % avail.len,
                        .balanced => (avail.len / 2 + i) % avail.len,
                        .low_bandwidth => avail.len - 1 - (i % avail.len),
                    };
                    if (fixed) {
                        if (dimension <= 1) {
                            strategy.addPartialBinding(tag, avail[preferred], 1, 1);
                            continue;
                        }
                        var needed: u32 = @intCast(dimension);
                        for (0..avail.len) |offset| {
                            const axis = if (target_intent == .low_bandwidth)
                                (preferred + avail.len - offset) % avail.len
                            else
                                (preferred + offset) % avail.len;
                            const size = std.math.gcd(needed, remaining[axis]);
                            if (size == 1) continue;
                            remaining[axis] = @divExact(remaining[axis], size);
                            strategy.addPartialBinding(tag, avail[axis], remaining[axis], size);
                            needed = @divExact(needed, size);
                            if (needed == 1) break;
                        }
                        // Mesh.init reports an incompatible dimension if capacity was insufficient.
                        if (needed == @as(u32, @intCast(dimension))) strategy.addPartialBinding(tag, avail[preferred], 1, 1);
                    } else if (remaining[preferred] > 1) {
                        strategy.addPartialBinding(tag, avail[preferred], 1, remaining[preferred]);
                        remaining[preferred] = 1;
                    }
                }
            }
        }
        if (strategy.bindings.len == 0) strategy.addPartialBinding(logical.axes.constSlice()[0], avail[0], 1, 1);

        return strategy;
    }
};

/// For a given shape, compute the shape of the slice each shard will receive.
pub fn shardedShape(shape: Shape) error{IncompatibleSharding}!Shape {
    if (shape._sharding.mesh != null) {
        const pl: Placement = try Sharding.Placement.init(&shape);
        var local_shape = pl.localShape(shape);
        local_shape._sharding = .replicated;
        return local_shape;
    } else {
        // fully replicated
        return shape;
    }
}

/// Owns per-device addressing metadata for a shape.
pub const Placement = struct {
    pub const Slices = stdx.BoundedArray(Slice1d, Shape.MAX_RANK);
    pub const Slice1d = struct {
        start: i64,
        size: i64,
    };

    local_dims: stdx.BoundedArray(i64, Shape.MAX_RANK),
    mesh_ptr: ?*const Mesh,
    byte_steps: [MAX_MESH_RANK]u64,

    pub fn init(shape: *const Shape) error{IncompatibleSharding}!Placement {
        const byte_strides = shape.computeByteStrides();
        return initWithByteStrides(shape, byte_strides.constSlice());
    }

    /// Reuse the caller's contiguous byte strides without recomputing them.
    pub fn initWithByteStrides(shape: *const Shape, byte_strides: []const i64) error{IncompatibleSharding}!Placement {
        std.debug.assert(byte_strides.len == shape.rank());
        if (builtin.mode == .debug) std.debug.assert(shape._sharding.partition.hasUniqueAxes());
        var result: Placement = .{
            .local_dims = .{ .buffer = undefined, .len = shape.rank() },
            .mesh_ptr = shape._sharding.mesh,
            .byte_steps = @splat(0),
        };
        for (shape.dims(), result.local_dims.slice(), 0..) |dim, *result_local_dim, ax| {
            result_local_dim.* = dim;
            const mesh_ = result.mesh_ptr orelse continue;
            const logical_axis = shape.partition(ax).meshAxis() orelse continue;
            const count = mesh_.grid.axis_sizes[logical_axis];
            if (@rem(dim, count) != 0) return error.IncompatibleSharding;
            const local_dim = @divExact(dim, count);
            result_local_dim.* = local_dim;
            result.byte_steps[logical_axis] = @intCast(local_dim * byte_strides[ax]);
        }
        return result;
    }

    pub fn mesh(pl: *const Placement) ?*const Mesh {
        return pl.mesh_ptr;
    }

    /// Reconstruct metadata only for consumers that need a full shard shape.
    pub fn localShape(pl: *const Placement, source: Shape) Shape {
        std.debug.assert(source.rank() == pl.local_dims.len);
        var result = source;
        @memcpy(result._dims.slice(), pl.local_dims.constSlice());
        result._sharding.mesh = pl.mesh_ptr;
        return result;
    }

    pub fn slices(pl: *const Placement, device_index: usize, source: Shape) Slices {
        std.debug.assert(source.rank() == pl.local_dims.len);
        var res: Slices = .{ .buffer = undefined, .len = pl.local_dims.len };
        if (pl.mesh_ptr == null) {
            for (res.slice(), pl.local_dims.constSlice()) |*slice, dim| slice.* = .{ .start = 0, .size = dim };
            return res;
        }
        const grid = &pl.mesh_ptr.?.grid;
        std.debug.assert(device_index < grid.device_count);
        const coords: [MAX_MESH_RANK]u8 = grid.logical_coords[device_index];
        var used_axes: u8 = 0;
        for (res.slice(), pl.local_dims.constSlice(), 0..) |*r, size, ax| {
            var coordinate: i64 = 0;
            if (source.partition(ax).meshAxis()) |logical_axis| {
                const bit = @as(u8, 1) << @intCast(logical_axis);
                if (used_axes & bit == 0) coordinate = coords[logical_axis];
                used_axes |= bit;
            }
            r.* = .{ .start = coordinate * size, .size = size };
        }
        return res;
    }

    /// Validate the contiguous source slice once before iterating over devices.
    /// The returned pointer includes the slice's byte offset.
    pub fn basePtr(pl: *const Placement, slice: Slice) [*]const u8 {
        if (builtin.mode == .debug) {
            std.debug.assert(pl.local_dims.len == slice.shape.rank());
        }
        return slice.constData().ptr;
    }

    /// device_index is a canonical ordinal, never a PJRT device ID.
    /// base_ptr must reference validated contiguous data for the global shape.
    pub fn shardPtr(pl: *const Placement, device_index: usize, base_ptr: [*]const u8) [*]const u8 {
        return base_ptr + pl.byteOffset(device_index);
    }

    fn byteOffset(pl: *const Placement, device_index: usize) usize {
        const mesh_ = pl.mesh() orelse return 0;
        std.debug.assert(device_index < mesh_.grid.device_count);
        const coords: [MAX_MESH_RANK]u8 = mesh_.grid.logical_coords[device_index];
        const steps = pl.byte_steps;
        // Placement coefficients and grid coordinates describe offsets within the
        // validated source slice. Keep arithmetic checks in debug only.
        return offset: {
            @setRuntimeSafety(builtin.mode == .debug);
            var offset: u64 = 0;
            inline for (0..MAX_MESH_RANK) |axis| {
                offset += @as(u64, coords[axis]) * steps[axis];
            }
            break :offset @intCast(offset);
        };
    }

    /// Given a Slice, returns the subslice corresponding to a specific device.
    /// Depending on the layout, the sub-slice may or may not be contiguous.
    pub fn shardSlice(pl: *const Placement, device_index: usize, slice: Slice) Slice {
        const shard_ptr = pl.shardPtr(device_index, pl.basePtr(slice));
        const offset_bytes = @intFromPtr(shard_ptr) - @intFromPtr(slice.bytes.ptr);
        return .{
            .bytes = slice.bytes,
            .mutable = slice.mutable,
            .shape = pl.localShape(slice.shape),
            .offset_bytes = offset_bytes,
            .byte_strides = slice.byte_strides,
        };
    }

    pub fn format(pl: *const Placement, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "Placement(local_dims={any}, mesh={*}, byte_steps={any})\n",
            .{ pl.local_dims.constSlice(), pl.mesh_ptr, pl.byte_steps },
        );
    }
};

const ShardingTest = struct {
    pub const ExpectedShard = struct {
        // TODO: remove device_id from here, we always have ids from 0 to N
        device_id: usize,
        slices: []const [2]i64,
    };

    pub const Scenario = struct {
        mesh: *const Mesh,
        shape: Shape,

        expected_num_partitions: u32,
        expected_num_replicas: u32,
        expect_error: ?anyerror = null,
        expected_sdy: ?[]const u8 = null,
        expected_shards: []const ExpectedShard = &.{},

        pub fn format(scenario: Scenario, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Scenario: partitions={d}, replicas={d}\n", .{ scenario.expected_num_partitions, scenario.expected_num_replicas });
            if (scenario.expect_error) |err| try writer.print("- error {}\n", .{err});
            if (scenario.expected_sdy) |sdy| try writer.print("- shardy: {s}\n", .{sdy});
            if (scenario.expected_shards.len > 0) try writer.print("- shards:\n", .{});
            for (scenario.expected_shards) |shard| {
                try writer.print("\t- device {d}: slices={any}\n", .{ shard.device_id, shard.slices });
            }
        }
    };

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ShardingTest {
        return .{ .allocator = allocator };
    }

    /// Creates a 1D-3D PhysicalMesh from simple dimensions.
    pub fn physical(self: ShardingTest, dims: anytype, geometry: AxisGeometry) !PhysicalMesh {
        const info = @typeInfo(@TypeOf(dims)).@"struct";
        const N = info.field_names.len;
        var sizes: [N]usize = undefined;
        inline for (info.field_names, sizes[0..]) |field_name, *s| {
            s.* = @intCast(@field(dims, field_name));
        }

        const tags: [3]PhysicalAxisTag = .{ .link_x, .link_y, .link_z };
        var next_id: u32 = 0;
        const root = try self.buildNode(tags[0..N], sizes[0..N], 0, &next_id, geometry);
        return try .fromTree(self.allocator, .tpu, root);
    }

    fn buildNode(self: ShardingTest, tags: []const PhysicalAxisTag, sizes: []const usize, depth: usize, next_id: *u32, geometry: AxisGeometry) !PhysicalNode {
        if (depth == tags.len) {
            const id = next_id.*;
            next_id.* += 1;
            return .{ .leaf = .{ .id = id, .coords = Device.undefined_coords } };
        }
        const count = sizes[depth];
        const children = try self.allocator.alloc(PhysicalNode, count);
        for (children) |*child| {
            child.* = try self.buildNode(tags, sizes, depth + 1, next_id, geometry);
        }
        return .{ .branch = .{ .tag = tags[depth], .geometry = geometry, .children = children } };
    }

    pub fn run(self: ShardingTest, s: Scenario) !void {
        const mesh = s.mesh;
        errdefer std.log.warn("Failed mesh {f}", .{s});

        try std.testing.expectEqual(s.expected_num_partitions, mesh.numPartitions());
        try std.testing.expectEqual(s.expected_num_replicas, mesh.numReplicas());
        try std.testing.expectEqual(mesh.grid.device_count, @as(usize, @intCast(mesh.numDevices())));

        // Verify MLIR String
        if (s.expected_sdy) |expected_attr| {
            errdefer std.log.warn("Expected shardy annotation failed", .{});
            const registry: *mlir.DialectRegistry = try .init();
            defer registry.deinit();
            registry.registerDialect("sdy");
            mlir.registerFuncExtensions(registry);

            var ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
            defer ctx.deinit();
            ctx.loadAllAvailableDialects();

            const actual_attr = try mesh.sdyShardingAttrForShape(self.allocator, ctx, s.shape);
            var writer: std.Io.Writer.Allocating = .init(self.allocator);
            defer writer.deinit();
            try actual_attr.asAttr().format(&writer.writer);
            try std.testing.expectEqualStrings(expected_attr, writer.written());
        }

        // Verify Placement logic / Error
        const ordered_devices = mesh.devicesInCanonicalOrder();
        if (s.expect_error) |err| {
            errdefer std.log.warn("Expected error failed", .{});
            try std.testing.expect(ordered_devices.len > 0);
            try std.testing.expectError(err, Sharding.Placement.init(&s.shape));
            return;
        }

        // Verify Shard Slices (Math)
        if (s.expected_shards.len > 0) {
            const pl: Placement = try Sharding.Placement.init(&s.shape);
            errdefer std.log.warn("Expected shards failed. Got {f}", .{pl});
            try std.testing.expectEqual(s.expected_shards.len, ordered_devices.len);
            for (s.expected_shards, ordered_devices, 0..) |expected, device, device_index| {
                {
                    try std.testing.expectEqual(expected.device_id, device.id);

                    // Check pl.slices(device)
                    const actual_slices = pl.slices(device_index, s.shape);

                    // Consider rewriting test cases to use Slice1D. This requires passing the axis
                    const actual_start_len = try self.allocator.alloc([2]i64, actual_slices.len);
                    defer self.allocator.free(actual_start_len);
                    for (actual_start_len, actual_slices.slice()) |*s_l, slice| {
                        s_l.* = .{ slice.start, slice.size };
                    }

                    try std.testing.expectEqualSlices([2]i64, expected.slices, actual_start_len);
                }
                {
                    // Check pl.shape
                    const expected_sizes = try self.allocator.alloc(i64, expected.slices.len);
                    defer self.allocator.free(expected_sizes);
                    for (expected_sizes, expected.slices) |*expected_size, slice| {
                        expected_size.* = slice[1];
                    }

                    try std.testing.expectEqualSlices(i64, expected_sizes, pl.local_dims.constSlice());
                }
            }
        }
    }
};

test "mesh: unknown partitioning implies replication" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .batch = .low_bandwidth });

    const strategy: Strategy = .parseBindings(.{ .batch = .link_x });
    const mesh: Sharding.Mesh = try .init("folded_mesh", &physical, logical, strategy);

    try runner.run(.{
        .expected_num_partitions = 2,
        .expected_num_replicas = 2,
        .mesh = &mesh,
        // No partitioning on shape
        .shape = Shape.init(.{ .batch = 8 }, .f32),
        .expected_sdy = "#sdy.sharding<@folded_mesh, [{}], replicated={\"batch\", \"__replica_0\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 1, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 2, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 3, .slices = &.{.{ 0, 8 }} },
        },
    });
}

test "suggest splits a one-dimensional topology for fixed data and automatic model axes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const runner: ShardingTest = .init(allocator);
    const logical = try LogicalMesh.init(.{ .data, .model }, &.{ .low_bandwidth, .high_bandwidth }, &.{ 2, -1 });
    for ([_]usize{ 2, 4, 8 }) |count| {
        const physical = try runner.physical(.{count}, .{ .mesh = .torus });
        const mesh_ = try Mesh.init("DP-MP", &physical, logical, .suggest(logical, &physical));
        try std.testing.expectEqual(2, mesh_.numPartitionsForLogicalAxis(.data));
        try std.testing.expectEqual(@as(u32, @intCast(count / 2)), mesh_.numPartitionsForLogicalAxis(.model));
        try std.testing.expectEqual(1, mesh_.numReplicas());
        for (0..count) |i| {
            const coords: [MAX_MESH_RANK]u8 = mesh_.grid.logical_coords[i];
            try std.testing.expectEqual(@as(u8, @intCast(i / (count / 2))), coords[0]);
            try std.testing.expectEqual(@as(u8, @intCast(i % (count / 2))), coords[1]);
        }
        const shape = Shape.init(.{ .b = 8, .m = 8 }, .f32).withPartitioning(&mesh_, .{ .b = .data, .m = .model });
        try runner.run(.{ .mesh = &mesh_, .shape = shape, .expected_num_partitions = @intCast(count), .expected_num_replicas = 1 });
    }
}

test "suggest reserves fixed dimensions across physical axes before automatic allocation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 4, 3 }, .{ .mesh = .torus });
    const logical = try LogicalMesh.init(.{ .model, .data }, &.{ .high_bandwidth, .low_bandwidth }, &.{ -1, 6 });
    const mesh_ = try Mesh.init("split", &physical, logical, .suggest(logical, &physical));
    try std.testing.expectEqual(2, mesh_.numPartitionsForLogicalAxis(.model));
    try std.testing.expectEqual(6, mesh_.numPartitionsForLogicalAxis(.data));
    try std.testing.expectEqual(1, mesh_.numReplicas());

    const fixed = try LogicalMesh.init(.{ .data, .model }, &.{ .low_bandwidth, .high_bandwidth }, &.{ 3, 4 });
    const fixed_mesh = try Mesh.init("fixed", &physical, fixed, .suggest(fixed, &physical));
    try std.testing.expectEqual(12, fixed_mesh.numPartitions());
    const impossible = try LogicalMesh.init(.{ .data, .model }, &.{ .low_bandwidth, .high_bandwidth }, &.{ 5, -1 });
    try std.testing.expectError(error.IncompatibleLogicalDimension, Mesh.init("impossible", &physical, impossible, .suggest(impossible, &physical)));
    const replica_logical = try LogicalMesh.init(.{ .data, .model }, &.{ .low_bandwidth, .high_bandwidth }, &.{ 1, 1 });
    const replica_mesh = try Mesh.init("replicated", &physical, replica_logical, .suggest(replica_logical, &physical));
    try std.testing.expectEqual(12, replica_mesh.numReplicas());
}

test "mesh: suggest strategy realization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });

    const logical: LogicalMesh = .mesh(.{
        .batch = .low_bandwidth,
        .model = .high_bandwidth,
    });

    const strategy: Strategy = .suggest(logical, &physical);
    const mesh: Mesh = try .init("suggested_mesh", &physical, logical, strategy);

    try runner.run(.{
        .expected_num_partitions = 4,
        .expected_num_replicas = 2,
        .mesh = &mesh,
        .shape = Shape.init(.{ .batch = 4, .model = 4 }, .f32)
            .withPartitioning(&mesh, .{ .batch = .batch, .model = .model }),
        .expected_sdy = "#sdy.sharding<@suggested_mesh, [{\"batch\"}, {\"model\"}], replicated={\"__replica_0\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 2 }, .{ 0, 2 } } }, // x0, y0, z0
            .{ .device_id = 1, .slices = &.{ .{ 2, 2 }, .{ 0, 2 } } }, // x0, y0, z1 (batch split)
            .{ .device_id = 2, .slices = &.{ .{ 0, 2 }, .{ 0, 2 } } }, // x0, y1, z0 (replicated y)
            .{ .device_id = 3, .slices = &.{ .{ 2, 2 }, .{ 0, 2 } } }, // x0, y1, z1
            .{ .device_id = 4, .slices = &.{ .{ 0, 2 }, .{ 2, 2 } } }, // x1, y0, z0 (model split)
            .{ .device_id = 5, .slices = &.{ .{ 2, 2 }, .{ 2, 2 } } },
            .{ .device_id = 6, .slices = &.{ .{ 0, 2 }, .{ 2, 2 } } },
            .{ .device_id = 7, .slices = &.{ .{ 2, 2 }, .{ 2, 2 } } },
        },
    });
}

test "mesh: suggest leaves conflicting logical axes replicated" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical = try runner.physical(.{4}, .point_to_point);

    const logical: LogicalMesh = .mesh(.{
        .model = .high_bandwidth,
        .experts = .high_bandwidth,
    });

    const strategy: Strategy = .suggest(logical, &physical);

    const mesh: Mesh = try .init("fold_mesh", &physical, logical, strategy);
    try runner.run(.{
        .expected_num_partitions = 4,
        .expected_num_replicas = 1,
        .mesh = &mesh,
        .shape = Shape.init(.{ .model = 4, .experts = 4 }, .f32)
            .withPartitioning(&mesh, .{ .model = .model, .experts = .experts }),
        .expected_sdy = "#sdy.sharding<@fold_mesh, [{\"model\"}, {\"experts\"}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 1 }, .{ 0, 4 } } },
            .{ .device_id = 1, .slices = &.{ .{ 1, 1 }, .{ 0, 4 } } },
            .{ .device_id = 2, .slices = &.{ .{ 2, 1 }, .{ 0, 4 } } },
            .{ .device_id = 3, .slices = &.{ .{ 3, 1 }, .{ 0, 4 } } },
        },
    });
    // Swapping tensor dimensions must not change which logical axis owns the devices.
    try runner.run(.{
        .expected_num_partitions = 4,
        .expected_num_replicas = 1,
        .mesh = &mesh,
        .shape = Shape.init(.{ .experts = 4, .model = 4 }, .f32)
            .withPartitioning(&mesh, .{ .experts = .experts, .model = .model }),
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 4 }, .{ 0, 1 } } },
            .{ .device_id = 1, .slices = &.{ .{ 0, 4 }, .{ 1, 1 } } },
            .{ .device_id = 2, .slices = &.{ .{ 0, 4 }, .{ 2, 1 } } },
            .{ .device_id = 3, .slices = &.{ .{ 0, 4 }, .{ 3, 1 } } },
        },
    });
}

test "mesh: multiple physical axes on one logical dimension (folding)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .model = .high_bandwidth });

    {
        const strategy: Strategy = .parseBindings(.{ .model = .{ .link_x, .link_y } });
        const mesh: Mesh = try .init("folded_mesh", &physical, logical, strategy);
        try runner.run(.{
            .expected_num_partitions = 4,
            .expected_num_replicas = 1,
            .mesh = &mesh,
            .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(&mesh, .{ .model = .model }),
            .expected_sdy = "#sdy.sharding<@folded_mesh, [{\"model\"}]>",
            .expected_shards = &.{
                .{ .device_id = 0, .slices = &.{.{ 0, 4 }} },
                .{ .device_id = 1, .slices = &.{.{ 4, 4 }} },
                .{ .device_id = 2, .slices = &.{.{ 8, 4 }} },
                .{ .device_id = 3, .slices = &.{.{ 12, 4 }} },
            },
        });
    }

    {
        // Swap link_x and link_y order:
        // -> the roles of devices 1 (0,1) and 2 (1,0) are swapped
        const strategy: Strategy = .parseBindings(.{ .model = .{ .link_y, .link_x } });
        const mesh: Mesh = try .init("folded_mesh", &physical, logical, strategy);
        try runner.run(.{
            .expected_num_partitions = 4,
            .expected_num_replicas = 1,
            .mesh = &mesh,
            .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(&mesh, .{ .model = .model }),
            .expected_sdy = "#sdy.sharding<@folded_mesh, [{\"model\"}]>",
            .expected_shards = &.{
                .{ .device_id = 0, .slices = &.{.{ 0, 4 }} },
                .{ .device_id = 1, .slices = &.{.{ 8, 4 }} },
                .{ .device_id = 2, .slices = &.{.{ 4, 4 }} },
                .{ .device_id = 3, .slices = &.{.{ 12, 4 }} },
            },
        });
    }
}

test "mesh: explicit strategy folding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .model = .high_bandwidth });

    var strategy: Strategy = .parseBindings(.{ .model = .link_x });
    strategy.addFold(.link_x, &.{ .link_x, .link_z });

    const mesh: Mesh = try .init("strategy_fold", &physical, logical, strategy);
    try runner.run(.{
        .expected_num_partitions = 4,
        .expected_num_replicas = 2,
        .mesh = &mesh,
        .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(&mesh, .{ .model = .model }),
        .expected_sdy = "#sdy.sharding<@strategy_fold, [{\"model\"}], replicated={\"__replica_0\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{.{ 0, 4 }} },
            .{ .device_id = 1, .slices = &.{.{ 4, 4 }} },
            .{ .device_id = 2, .slices = &.{.{ 0, 4 }} },
            .{ .device_id = 3, .slices = &.{.{ 4, 4 }} },
            .{ .device_id = 4, .slices = &.{.{ 8, 4 }} },
            .{ .device_id = 5, .slices = &.{.{ 12, 4 }} },
            .{ .device_id = 6, .slices = &.{.{ 8, 4 }} },
            .{ .device_id = 7, .slices = &.{.{ 12, 4 }} },
        },
    });
}

test "mesh: open and replicated dimension mix" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth });

    const strategy: Strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y });

    const mesh: Mesh = try .init("mix_mesh", &physical, logical, strategy);
    try runner.run(.{
        .expected_num_partitions = 4,
        .expected_num_replicas = 1,
        .mesh = &mesh,
        .shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(&mesh, .{ .batch = .open, .model = .replicated }),
        .expected_sdy = "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"batch\", \"model\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 1, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 2, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 3, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
        },
    });
}

test "mesh: full 3D cluster mesh" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth, .context = .balanced });

    const strategy: Strategy = .parseBindings(.{
        .batch = .link_x,
        .model = .link_y,
        .context = .link_z,
    });

    const mesh: Mesh = try .init("3d_mesh", &physical, logical, strategy);
    try runner.run(.{
        .expected_num_partitions = 8,
        .expected_num_replicas = 1,
        .mesh = &mesh,
        // Note: .context and .model are swapped compared to .link_x and .link_z
        .shape = Shape.init(.{ .batch = 4, .context = 4, .model = 4 }, .f32)
            .withPartitioning(&mesh, .{ .batch = .batch, .model = .model, .context = .context }),
        .expected_sdy = "#sdy.sharding<@\"3d_mesh\", [{\"batch\"}, {\"context\"}, {\"model\"}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 2 }, .{ 0, 2 }, .{ 0, 2 } } },
            .{ .device_id = 1, .slices = &.{ .{ 0, 2 }, .{ 2, 2 }, .{ 0, 2 } } },
            .{ .device_id = 2, .slices = &.{ .{ 0, 2 }, .{ 0, 2 }, .{ 2, 2 } } },
            .{ .device_id = 3, .slices = &.{ .{ 0, 2 }, .{ 2, 2 }, .{ 2, 2 } } },
            .{ .device_id = 4, .slices = &.{ .{ 2, 2 }, .{ 0, 2 }, .{ 0, 2 } } },
            .{ .device_id = 5, .slices = &.{ .{ 2, 2 }, .{ 2, 2 }, .{ 0, 2 } } },
            .{ .device_id = 6, .slices = &.{ .{ 2, 2 }, .{ 0, 2 }, .{ 2, 2 } } },
            .{ .device_id = 7, .slices = &.{ .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 } } },
        },
    });
}

test "mesh: num partitions for logical axis" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{
        .model = .high_bandwidth,
        .batch = .low_bandwidth,
    });

    var strategy: Strategy = .parseBindings(.{ .model = .link_x });
    strategy.addFold(.link_x, &.{ .link_x, .link_z });

    const mesh: Mesh = try .init("axis_parts_mesh", &physical, logical, strategy);

    try std.testing.expectEqual(4, mesh.numPartitionsForLogicalAxis(.model));
    try std.testing.expectEqual(1, mesh.numPartitionsForLogicalAxis(.batch));
}

test "mesh: size-one and unbound logical axes leave all devices replicated" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 1, 6 }, .{ .mesh = .torus });
    const mesh_ = try Mesh.init("replicas", &physical, .mesh(.{ .data = .balanced, .model = .balanced }), .parseBindings(.{ .data = .link_x }));
    try runner.run(.{
        .mesh = &mesh_,
        .shape = Shape.init(.{ .batch = 12 }, .f32).withPartitioning(&mesh_, .{ .batch = .data }),
        .expected_num_partitions = 1,
        .expected_num_replicas = 6,
    });
}

test "mesh: logical partitions and replicas can have different sizes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 2, 3, 2 }, .{ .mesh = .torus });
    var strategy: Strategy = .parseBindings(.{ .model = .link_x });
    strategy.addFold(.link_x, &.{ .link_z, .link_x });
    const mesh_ = try Mesh.init("folded", &physical, .mesh(.{ .model = .balanced }), strategy);
    try runner.run(.{
        .mesh = &mesh_,
        .shape = Shape.init(.{ .model = 8 }, .f32).withPartitioning(&mesh_, .{ .model = .model }),
        .expected_num_partitions = 4,
        .expected_num_replicas = 3,
    });
}

test "localized shapes do not retain global sharding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{2}, .{ .mesh = .torus });
    const mesh_ = try Mesh.init("local", &physical, .mesh(.{ .batch = .balanced }), .parseBindings(.{ .batch = .link_x }));
    try std.testing.expectEqual(mesh_.bindings.constSlice()[0].constSlice().ptr, mesh_.binding(0).ptr);
    const global = Shape.init(.{ .batch = 8 }, .f32).withPartitioning(&mesh_, .{ .batch = .batch });
    const local = try shardedShape(global);
    try std.testing.expectEqual(4, local.dim(.batch));
    try std.testing.expectEqual(null, local._sharding.mesh);
    try std.testing.expectEqual(8, global.dim(.batch));
    try std.testing.expectEqual(&mesh_, global._sharding.mesh);
}

test "GSPMD tiles use tensor axis order and logical partition indices" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    for (physical.devices_in_canonical_order, 0..) |*device, i| device.id = @intCast(7 - i);
    const mesh_ = try Mesh.init("tiles", &physical, .mesh(.{ .batch = .balanced, .model = .balanced }), .parseBindings(.{ .batch = .link_z, .model = .link_x }));
    const shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(&mesh_, .{ .batch = .batch, .model = .model });
    const ids = try mesh_.gspmdTileAssignment(arena.allocator(), shape);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 6, 1, 3, 5, 7 }, ids);
}

test "DeviceGrid stores aligned logical coordinates in canonical order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    for (physical.devices_in_canonical_order, 0..) |*device, i| device.id = @intCast(3 - i);
    const mesh_ = try Mesh.init("grid", &physical, .mesh(.{ .model = .balanced, .data = .balanced }), .parseBindings(.{ .model = .link_y, .data = .link_x }));
    const expected = [_][4]u8{ .{ 0, 0, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 1, 1, 0, 0 } };
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(&mesh_.grid.logical_coords) % 16);
    for (mesh_.grid.logical_coords[0..mesh_.grid.device_count], expected, 0..) |coords, want, i| {
        try std.testing.expectEqual(@intFromPtr(&mesh_.grid.logical_coords) + 4 * i, @intFromPtr(&mesh_.grid.logical_coords[i]));
        const actual: [4]u8 = coords;
        try std.testing.expectEqualSlices(u8, &want, &actual);
    }
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(DeviceGrid.Coordinates));
    const shape = Shape.init(.{ .m = 16, .n = 24 }, .f32).withPartitioning(&mesh_, .{ .m = .model, .n = .data });
    const placement = try Sharding.Placement.init(&shape);
    const offsets = [_]usize{ 0, 768, 48, 816 };
    var data: [16 * 24 * 4 + 16]u8 = undefined;
    var slice = Slice.init(shape, data[16..]);
    slice.bytes = &data;
    slice.offset_bytes = 16;
    const base_ptr = placement.basePtr(slice);
    try std.testing.expectEqual(@intFromPtr(&data) + 16, @intFromPtr(base_ptr));
    for (offsets, 0..) |offset, i| {
        try std.testing.expectEqual(offset, placement.byteOffset(i));
        try std.testing.expectEqual(@intFromPtr(&data) + 16 + offset, @intFromPtr(placement.shardPtr(i, base_ptr)));
    }
}

test "partial bindings share physical axes and combine portions in order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const runner: ShardingTest = .init(allocator);
    const physical = try runner.physical(.{ 4, 6 }, .{ .mesh = .torus });
    const logical = try LogicalMesh.init(.{ .model, .data }, &.{ .balanced, .low_bandwidth }, &.{ -1, -1 });
    var strategy: Strategy = .init;
    strategy.addPartialBinding(.model, .link_y, 2, 3);
    strategy.addPartialBinding(.model, .link_x, 1, 2);
    strategy.addPartialBinding(.data, .link_x, 2, 2);
    strategy.addPartialBinding(.data, .link_y, 1, 2);
    const mesh_ = try Mesh.init("partial", &physical, logical, strategy);
    try std.testing.expectEqual(6, mesh_.numPartitionsForLogicalAxis(.model));
    try std.testing.expectEqual(4, mesh_.numPartitionsForLogicalAxis(.data));
    try std.testing.expectEqual(1, mesh_.numReplicas());
    for (physical.devices_in_canonical_order, 0..) |device, i| {
        const coords: [MAX_MESH_RANK]u8 = mesh_.grid.logical_coords[i];
        try std.testing.expectEqual((device.coords[1] / 2) * 2 + device.coords[0] % 2, coords[0]);
        try std.testing.expectEqual((device.coords[0] / 2) * 2 + device.coords[1] % 2, coords[1]);
    }
    const formatted = try std.fmt.allocPrint(allocator, "{f}", .{mesh_});
    try std.testing.expect(std.mem.indexOf(u8, formatted, "model (balanced) -> (link_y / 2) * 2 + (link_x % 2)") != null);
    const shape = Shape.init(.{ .m = 12, .d = 8 }, .f32).withPartitioning(&mesh_, .{ .m = .model, .d = .data });
    try runner.run(.{ .mesh = &mesh_, .shape = shape, .expected_num_partitions = 24, .expected_num_replicas = 1 });
}

test "fixed logical dimensions constrain whole partial and folded bindings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 4, 6 }, .{ .mesh = .torus });
    const logical = try LogicalMesh.init(.{ .model, .data, .unused }, &.{ .balanced, .low_bandwidth, .balanced }, &.{ 2, -1, 1 });
    var strategy: Strategy = .parseBindings(.{ .model = .link_x, .data = .link_y });
    const mesh_ = try Mesh.init("constrained", &physical, logical, strategy);
    try std.testing.expectEqual(2, mesh_.numPartitionsForLogicalAxis(.model));
    try std.testing.expectEqual(6, mesh_.numPartitionsForLogicalAxis(.data));
    try std.testing.expectEqual(1, mesh_.numPartitionsForLogicalAxis(.unused));
    try std.testing.expectEqual(2, mesh_.numReplicas());
    for (physical.devices_in_canonical_order, 0..) |device, i| {
        try std.testing.expectEqual(device.coords[0] % 2, mesh_.grid.logical_coords[i][0]);
    }

    const fixed = try LogicalMesh.init(.{.model}, &.{.balanced}, &.{2});
    strategy = .init;
    strategy.addPartialBinding(.model, .link_x, 1, 4);
    const partial = try Mesh.init("partial", &physical, fixed, strategy);
    try std.testing.expectEqual(2, partial.numPartitions());
    try std.testing.expectEqual(12, partial.numReplicas());

    strategy = .parseBindings(.{ .model = .link_x });
    strategy.addFold(.link_x, &.{ .link_y, .link_x });
    const folded = try Mesh.init("folded", &physical, try LogicalMesh.init(.{.model}, &.{.balanced}, &.{8}), strategy);
    try std.testing.expectEqual(8, folded.numPartitions());
    try std.testing.expectEqual(3, folded.numReplicas());
    for (physical.devices_in_canonical_order, 0..) |device, i| {
        try std.testing.expectEqual((device.coords[1] * 4 + device.coords[0]) % 8, folded.grid.logical_coords[i][0]);
    }
    for ([_]i32{ 5, 32 }) |dimension| {
        try std.testing.expectError(error.IncompatibleLogicalDimension, Mesh.init("invalid", &physical, try LogicalMesh.init(.{.model}, &.{.balanced}, &.{dimension}), strategy));
    }
}

test "partial bindings reject invalid portions and correlated logical axes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{8}, .{ .mesh = .torus });
    for ([_][2]u32{ .{ 0, 2 }, .{ 1, 0 }, .{ 3, 2 }, .{ 1, 3 }, .{ 4, 4 }, .{ 1, std.math.maxInt(u32) } }) |portion| {
        var strategy: Strategy = .init;
        strategy.addPartialBinding(.model, .link_x, portion[0], portion[1]);
        try std.testing.expectError(error.InvalidPartialBinding, Mesh.init("invalid", &physical, .mesh(.{ .model = .balanced }), strategy));
    }
    var strategy: Strategy = .init;
    strategy.addPartialBinding(.model, .link_y, 1, 2);
    try std.testing.expectError(error.InvalidPhysicalAxis, Mesh.init("invalid", &physical, .mesh(.{ .model = .balanced }), strategy));
    strategy = .init;
    strategy.addPartialBinding(.model, .link_x, 1, 2);
    strategy.addPartialBinding(.data, .link_x, 1, 2);
    // Four claimed partitions fit on eight devices, but only two coordinates occur.
    try std.testing.expectError(error.OverlappingLogicalAxes, Mesh.init("overlap", &physical, .mesh(.{ .model = .balanced, .data = .balanced }), strategy));
}

test "DeviceGrid resolves folds once in strategy order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const runner: ShardingTest = .init(allocator);
    const physical = try runner.physical(.{ 2, 3, 2 }, .{ .mesh = .torus });
    for (physical.devices_in_canonical_order, 0..) |*device, i| device.id = @intCast(11 - i);
    var strategy: Strategy = .parseBindings(.{ .model = .{ .link_x, .link_z } });
    strategy.addFold(.link_x, &.{ .link_z, .link_x });
    const mesh_ = try Mesh.init("folded", &physical, .mesh(.{ .model = .balanced }), strategy);
    strategy.folding = .empty;

    try std.testing.expectEqual(4, mesh_.numPartitions());
    try std.testing.expectEqual(3, mesh_.numReplicas());
    const expected = [_]u8{ 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3 };
    for (expected, 0..) |coord, i| {
        try std.testing.expectEqual(coord, mesh_.grid.logical_coords[i][0]);
    }
    const formatted = try std.fmt.allocPrint(allocator, "{f}", .{mesh_});
    try std.testing.expect(std.mem.indexOf(u8, formatted, "model (balanced) -> link_z * 2 + link_x\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "model=0,1,2,3 -> devices=[11,9,7],[5,3,1],[10,8,6],[4,2,0]") != null);
}

test "DeviceGrid validates fold sources and overlapping folded axes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    var strategy: Strategy = .parseBindings(.{ .model = .link_x });
    strategy.addFold(.link_y, &.{ .link_y, .link_z });
    try std.testing.expectError(error.InvalidPhysicalAxis, Mesh.init("invalid", &physical, .mesh(.{ .model = .balanced }), strategy));

    strategy = .parseBindings(.{ .model = .link_x, .data = .link_y });
    strategy.addFold(.link_x, &.{ .link_x, .link_y });
    try std.testing.expectError(error.OverlappingLogicalAxes, Mesh.init("overlap", &physical, .mesh(.{ .model = .balanced, .data = .balanced }), strategy));
}

test "Placement owns dimensions and byte steps across copies" {
    const Helper = struct {
        fn create() !Placement {
            const shape = Shape.init(.{ .m = 3, .n = 5 }, .f32);
            return Placement.init(&shape);
        }
    };
    var original = try Helper.create();
    original.byte_steps[0] = 60;
    const copied = original;
    original.local_dims.buffer[0] = 99;
    original.byte_steps[0] = 100;
    try std.testing.expectEqualSlices(i64, &.{ 3, 5 }, copied.local_dims.constSlice());
    try std.testing.expectEqualSlices(u64, &.{ 60, 0, 0, 0 }, &copied.byte_steps);
    const source = Shape.init(.{ .m = 3, .n = 5 }, .f32);
    try std.testing.expect(copied.localShape(source).eqlWithTags(source));
    var data: [3 * 5 * 4]u8 = undefined;
    try std.testing.expectEqual(@intFromPtr(&data), @intFromPtr(copied.basePtr(.init(source, &data))));
}

test "Placement byte steps and dot product exceed 32 bits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{Platform.MAX_NUM_DEVICES}, .{ .mesh = .torus });
    const mesh_ = try Mesh.init("wide", &physical, .mesh(.{ .model = .balanced }), .parseBindings(.{ .model = .link_x }));
    const shape = Shape.init(.{ .m = 2 * Platform.MAX_NUM_DEVICES, .n = 1 << 30 }, .f32).withPartitioning(&mesh_, .{ .m = .model });
    const placement = try Sharding.Placement.init(&shape);
    try std.testing.expectEqual(@as(usize, 1) << 33, placement.byteOffset(1));
    const last_device = Platform.MAX_NUM_DEVICES - 1;
    const coords: [MAX_MESH_RANK]u8 = mesh_.grid.logical_coords[last_device];
    try std.testing.expectEqual(@as(u8, @intCast(last_device)), coords[0]);
    try std.testing.expectEqual(@as(usize, last_device) << 33, placement.byteOffset(last_device));
}

test "DeviceGrid rejects overlapping logical axes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    try std.testing.expectError(error.OverlappingLogicalAxes, Mesh.init("overlap", &physical, .mesh(.{ .model = .balanced, .data = .balanced }), .parseBindings(.{ .model = .{ .link_x, .link_y }, .data = .{ .link_y, .link_z } })));
}

test "Shardy meshes preserve logical order and unused physical replicas" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const runner: ShardingTest = .init(allocator);
    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const mesh_ = try Mesh.init("logical", &physical, .mesh(.{ .batch = .balanced, .model = .balanced }), .parseBindings(.{ .batch = .link_z, .model = .link_x }));
    const registry: *mlir.DialectRegistry = try .init();
    defer registry.deinit();
    registry.registerDialect("sdy");
    const ctx: *mlir.Context = try .init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const mesh_attr = try mesh_.sdyMeshAttr(allocator);
    _ = try mlir.Attribute.parse(ctx, mesh_attr);
    try std.testing.expectEqualStrings(
        "#sdy.mesh<[\"batch\"=2, \"model\"=2, \"__replica_0\"=2], device_ids=[0, 2, 4, 6, 1, 3, 5, 7]>",
        mesh_attr,
    );
}

test "Shardy replica axis avoids logical name collisions" {
    const mesh_ = Mesh.initTest(.collision, .{ .__replica_0, .__replica_1 });
    try std.testing.expectEqualStrings("__replica_2", mesh_.sdyReplicaAxis());
}

test "GSPMD attributes use logical tile sizes and replica groups" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const ctx: *mlir.Context = try .init(.{ .threading = false });
    defer ctx.deinit();
    const runner: ShardingTest = .init(allocator);
    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    for (physical.devices_in_canonical_order, 0..) |*device, i| device.id = @intCast(7 - i);
    const mesh_ = try Mesh.init("tiles", &physical, .mesh(.{ .batch = .balanced, .model = .balanced, .unused = .balanced }), .parseBindings(.{ .batch = .link_z, .model = .link_x }));

    const Case = struct { shape: Shape, expected: []const u8 };
    const cases = [_]Case{
        .{
            .shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(&mesh_, .{ .batch = .batch, .model = .model }),
            .expected = "{devices=[2,2,2]0,2,4,6,1,3,5,7 last_tile_dim_replicate}",
        },
        .{
            .shape = Shape.init(.{ .model = 8, .batch = 8 }, .f32).withPartitioning(&mesh_, .{ .batch = .batch, .model = .model }),
            .expected = "{devices=[2,2,2]0,2,1,3,4,6,5,7 last_tile_dim_replicate}",
        },
        .{
            .shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(&mesh_, .{ .batch = .batch }),
            .expected = "{devices=[2,1,4]0,2,4,6,1,3,5,7 last_tile_dim_replicate}",
        },
        .{
            .shape = Shape.init(.{ .batch = 8 }, .f32).withPartitioning(&mesh_, .{ .batch = .unused }),
            .expected = "{replicated}",
        },
        .{
            .shape = Shape.init(.{ .batch = 8 }, .f32),
            .expected = "{replicated}",
        },
        .{
            .shape = Shape.init(.{}, .f32),
            .expected = "{replicated}",
        },
    };
    for (cases) |case| {
        const actual = try mesh_.gspmdShardingAttrForShape(allocator, ctx, case.shape);
        try std.testing.expect(actual.eql(.string(ctx, case.expected)));
    }
}

test "GSPMD attributes preserve folded logical axis order" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const ctx: *mlir.Context = try .init(.{ .threading = false });
    defer ctx.deinit();
    const runner: ShardingTest = .init(allocator);
    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    var strategy: Strategy = .parseBindings(.{ .model = .link_x, .batch = .link_y });
    strategy.addFold(.link_x, &.{ .link_z, .link_x });
    const mesh_ = try Mesh.init("folded", &physical, .mesh(.{ .model = .balanced, .batch = .balanced }), strategy);
    const sharded = Shape.init(.{ .model = 8, .batch = 8 }, .f32).withPartitioning(&mesh_, .{ .model = .model, .batch = .batch });
    const actual = try mesh_.gspmdShardingAttrForShape(allocator, ctx, sharded);
    try std.testing.expect(actual.eql(.string(ctx, "{devices=[4,2]0,2,4,6,1,3,5,7}")));

    const partial = Shape.init(.{ .model = 8 }, .f32).withPartitioning(&mesh_, .{ .model = .model });
    const partial_attr = try mesh_.gspmdShardingAttrForShape(allocator, ctx, partial);
    try std.testing.expect(partial_attr.eql(.string(ctx, "{devices=[4,2]0,2,4,6,1,3,5,7 last_tile_dim_replicate}")));
}

test "GSPMD executes a reduction across a folded logical axis" {
    if (comptime !Target.cpu.isEnabled()) return error.SkipZigTest;
    const Tensor = @import("tensor.zig").Tensor;
    const Buffer = @import("buffer.zig").Buffer;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try Platform.init(allocator, io, .cpu, .{
        .cpu = .{ .device_count = 8 },
        .physical_mesh = .{ .custom = PhysicalMesh.torus3d },
    });
    defer platform.deinit(allocator, io);
    var strategy: Strategy = .parseBindings(.{ .model = .link_x, .data = .link_y });
    strategy.addFold(.link_x, &.{ .link_z, .link_x });
    const mesh_ = try platform.registerMeshWithStrategy("logical", .mesh(.{ .data = .balanced, .model = .balanced }), strategy);
    const Functions = struct {
        fn forward(input: Tensor) Tensor {
            return input.sum(.m).squeeze(.m).withPartitioning(input.shape()._sharding.mesh.?, .{ .n = .data });
        }
    };
    const input = Tensor.init(.{ .m = 8, .n = 4 }, .i32).withPartitioning(mesh_, .{ .m = .model, .n = .data });
    const exe = try platform.compileFn(allocator, io, Functions.forward, .{input}, .{
        .meshes = &.{mesh_},
        .partitioner = .gspmd,
    });
    defer exe.deinit();
    var values: [8][4]i32 = undefined;
    var expected: [4]i32 = @splat(0);
    for (&values, 0..) |*row, m| {
        for (row, 0..) |*value, n| {
            value.* = @intCast(10 * m + n);
            expected[n] += value.*;
        }
    }
    var input_buffer = try Buffer.fromBytes(io, platform, input.shape(), std.mem.asBytes(&values));
    defer input_buffer.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{input_buffer});
    exe.call(io, args, &results, .{});
    var output = results.get(Buffer);
    defer output.deinit();
    try std.testing.expectEqual(expected, try output.getValue([4]i32, io));
}
