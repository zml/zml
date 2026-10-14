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

    pub fn format(device: Device, writer: *std.Io.Writer) !void {
        try writer.print(
            "Device(id={d} coords={any})",
            .{ device.id, device.coords },
        );
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

        pub fn format(self: AxisTraversal, writer: *std.Io.Writer) !void {
            try writer.writeAll("AxisTraversal(order=[");

            for (self.order.slice(), 0..) |tag, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.writeAll(@tagName(tag));
            }

            try writer.writeAll("], depth_by_tag={");

            const field_names = comptime std.meta.fieldNames(PhysicalAxisTag);
            inline for (field_names, 0..) |field_name, i| {
                if (i > 0) try writer.writeAll(", ");
                const tag = @field(PhysicalAxisTag, field_name);
                try writer.writeAll(field_name);
                try writer.writeAll("=");
                if (self.depth_by_tag.get(tag)) |d| {
                    try writer.print("{d}", .{d});
                } else {
                    try writer.writeAll("null");
                }
            }

            try writer.writeAll("})");
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
        try writer.print("├── {f}\n", .{self.axis_traversal});
        try writer.print("│  \n", .{});

        try formatNode(self.root, writer, "", true);
    }

    fn formatNode(node: PhysicalNode, writer: *std.Io.Writer, prefix: []const u8, is_last: bool) !void {
        try writer.print("{s}{s}", .{ prefix, if (is_last) "└── " else "├── " });

        switch (node) {
            .leaf => |d| {
                try writer.print("{f}\n", .{d});
            },
            .branch => |b| {
                try writer.print("[{s}] x{d} ({s})\n", .{ @tagName(b.tag), b.children.len, @tagName(b.geometry) });

                var buf: [256]u8 = undefined;
                const new_prefix = std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, if (is_last) "    " else "│   " }) catch unreachable;

                for (b.children, 0..) |child, i| {
                    const child_is_last = (i == b.children.len - 1);
                    try formatNode(child, writer, new_prefix, child_is_last);
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

// Defines the logical mesh used for mesh the model and data.
// E.g., batch, model, context, experts, etc.
// Each axis represents a dimension along which the workload can be partitioned. The value is the intent of the axis.
/// LogicalMesh defines *semantic* axes only (batch/model/context/...).
/// It stores intent, not size.
///
/// Why:
/// - Logical axes represent model meaning, not hardware.
/// - Sizes come from tensors later, during assignment.
///
/// How it works:
/// - init() takes a struct of axis intents.
/// - each field name becomes a logical axis tag.
///
/// Limitations:
/// - No sizes here; no divisibility checks here.
pub const LogicalMesh = struct {
    pub const Axes = stdx.BoundedArray(Shape.Tag, MAX_MESH_RANK);
    pub const Intents = stdx.BoundedArray(LogicalAxisIntent, MAX_MESH_RANK);

    axes: Axes,
    intents: Intents,

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

        return .{
            .axes = axes,
            .intents = intents,
        };
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
        for (self.axes.slice(), self.intents.slice()) |axis, int| {
            try writer.print(" {s}={s}", .{ axis, @tagName(int) });
        }
        try writer.writeAll(")");
    }
};

pub const PhysicalList = stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK);

pub const Fold = struct {
    target: PhysicalAxisTag,
    sources: PhysicalList,
};
pub const Folds = stdx.BoundedArray(Fold, MAX_MESH_RANK);

pub const Axis = struct {
    tag: PhysicalAxisTag,
    size: i64,
    geometry: ?AxisGeometry,
    folded: stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK),

    pub fn contains(self: *const Axis, tag: PhysicalAxisTag) bool {
        if (self.tag == tag) return true;
        for (self.folded.slice()) |t| if (t == tag) return true;
        return false;
    }
};

pub const PhysicalView = struct {
    axes: stdx.BoundedArray(Axis, MAX_MESH_RANK),
    total_devices: i64,

    pub fn axisCoordFromLinearShard(self: *const PhysicalView, axis_index: usize, linear_idx: usize) usize {
        const stride = self.axisStrideForLinear(axis_index);
        const axis_size: usize = @intCast(self.axes.slice()[axis_index].size);
        return (linear_idx / stride) % axis_size;
    }

    pub fn axisStrideForLinear(self: *const PhysicalView, axis_index: usize) usize {
        var stride: usize = 1;
        var j = axis_index + 1;
        while (j < self.axes.len) : (j += 1) {
            stride *= @intCast(self.axes.slice()[j].size);
        }
        return stride;
    }
};

/// Resolved logical coordinates in canonical physical-device order. Each row is
/// one packed four-byte row; the contiguous table is 16-byte aligned.
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

    fn init(mesh_: *const Mesh) !DeviceGrid {
        const devices = mesh_.devicesInCanonicalOrder();
        if (devices.len > Platform.MAX_NUM_DEVICES) return error.InvalidPhysicalMesh;
        var grid: DeviceGrid = .{
            .rank = @intCast(mesh_.logical.axes.len),
            .device_count = devices.len,
            .axis_sizes = @splat(1),
            .logical_coords = @splat(@splat(0)),
        };
        var claimed: std.EnumSet(PhysicalAxisTag) = .empty;
        for (0..grid.rank) |logical_axis| {
            var used: std.EnumSet(PhysicalAxisTag) = .empty;
            for (mesh_.binding(logical_axis)) |tag| {
                for (mesh_.foldSources(tag) orelse &.{tag}) |source| {
                    if (used.contains(source)) continue;
                    used.insert(source);
                    if (claimed.contains(source)) return error.OverlappingLogicalAxes;
                    claimed.insert(source);
                    const info = mesh_.physical.axisInfo(source) orelse continue;
                    const depth = mesh_.physical.axis_traversal.depth(source) orelse continue;
                    const size: u32 = @intCast(info.size);
                    grid.axis_sizes[logical_axis] *= size;
                    for (devices, 0..) |device, i| {
                        var coords: [MAX_MESH_RANK]u8 = grid.logical_coords[i];
                        coords[logical_axis] = @intCast(@as(u32, coords[logical_axis]) * size + device.coords[depth]);
                        grid.logical_coords[i] = coords;
                    }
                }
            }
        }
        return grid;
    }
};

pub const Mesh = struct {
    pub const zml_no_meta_visit: void = {};

    name: []const u8,
    physical: *const PhysicalMesh,
    logical: LogicalMesh,
    grid: DeviceGrid,

    /// Compact binding table: logical axis -> physical axes
    bindings: stdx.BoundedArray(PhysicalList, MAX_MESH_RANK),

    /// Explicit folding rules: kept axis -> ordered source axes.
    folds: Folds,
    folds_consumed: std.EnumSet(PhysicalAxisTag),

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
                    binding_.* = strat_binding.physical;
                    break;
                }
            } else {
                std.debug.panic("{f} expects strategy to only list known logical axes, got unknown {s}", .{ logical, strat_binding.logical });
            }
        }

        var folds: Folds = .empty;
        var folds_consumed: std.EnumSet(PhysicalAxisTag) = .empty;
        for (strategy.folding.slice()) |entry| {
            for (entry.sources.slice()) |src| {
                if (!physical.hasAxis(src)) return error.InvalidPhysicalAxis;
                folds_consumed.insert(src);
            }
            folds.appendAssumeCapacity(.{ .target = entry.target, .sources = entry.sources });
        }

        var mesh_: Mesh = .{
            .name = name_,
            .logical = logical,
            .grid = undefined,
            .physical = physical,
            .bindings = bindings,
            .folds = folds,
            .folds_consumed = folds_consumed,
        };
        mesh_.grid = try DeviceGrid.init(&mesh_);
        return mesh_;
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
            .folds = .empty,
            .folds_consumed = .empty,
        };
    }

    fn foldSources(self: *const Mesh, tag: PhysicalAxisTag) ?[]const PhysicalAxisTag {
        for (self.folds.slice()) |*f| {
            if (f.target == tag) return f.sources.slice();
        }
        return null;
    }

    pub fn devicesInCanonicalOrder(mesh: *const Mesh) []const Device {
        return mesh.physical.devices_in_canonical_order;
    }

    pub fn physicalView(self: *const Mesh) PhysicalView {
        const physical = self.physical;
        var view: PhysicalView = .{
            .axes = .empty,
            .total_devices = 1,
        };

        const axis_order = physical.axisOrder().slice();

        for (axis_order) |tag| {
            if (!physical.hasAxis(tag)) continue;
            if (self.folds_consumed.contains(tag) and self.foldSources(tag) == null) continue;

            var folded: stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK) = .empty;
            var size: i64 = 1;

            if (self.foldSources(tag)) |sources| {
                for (sources) |src| {
                    folded.appendAssumeCapacity(src);
                    size *= physical.axis(src);
                }
            } else {
                folded.appendAssumeCapacity(tag);
                size = physical.axis(tag);
            }

            view.total_devices *= size;
            view.axes.appendAssumeCapacity(.{
                .tag = tag,
                .size = size,
                .geometry = physical.geometry(tag),
                .folded = folded,
            });
        }

        return view;
    }

    pub fn logicalIndexFromCoords(self: *const Mesh, coords: []const u8) usize {
        return self.physical.linearIndexFromCoords(coords);
    }

    pub fn numPartitions(self: *const Mesh) i32 {
        return @intCast(self.physicalView().total_devices);
    }

    pub fn numPartitionsForLogicalAxis(self: *const Mesh, logical_axis: anytype) i64 {
        const logical_tag = Shape.toTag(logical_axis);
        const logical_ax = self.resolveLogicalAxis(logical_tag) orelse std.debug.panic("mesh {f} has no axis {s}", .{ self, logical_tag });
        return self.grid.axis_sizes[logical_ax];
    }

    pub fn numReplicas(_: *const Mesh) i32 {
        return 1;
    }

    pub fn numDevices(self: *const Mesh) i32 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn sdyMeshAttr(self: *const Mesh, allocator: std.mem.Allocator) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        const view = self.physicalView();
        try out.writer.writeAll("#sdy.mesh<[");
        for (view.axes.slice(), 0..) |p, i| {
            if (i > 0) try out.writer.writeAll(", ");
            try out.writer.print("\"{s}\"={d}", .{ @tagName(p.tag), p.size });
        }
        try out.writer.writeAll("]>");
        return try out.toOwnedSlice();
    }

    const DimMapping = struct {
        /// For each tensor dimension, the indices of physical axes (in PhysicalView) mesh it.
        axes_per_dim: stdx.BoundedArray(stdx.BoundedArray(usize, Shape.MAX_RANK), Shape.MAX_RANK),
        /// Indices of physical axes used for replication.
        replicated_axes: stdx.BoundedArray(usize, Shape.MAX_RANK),
        /// Reference to the view for sizes.
        view: PhysicalView,
    };

    /// Common logic to map tensor dimensions to physical mesh indices.
    fn getDimMapping(self: *const Mesh, shape: Shape) DimMapping {
        const view = self.physicalView();
        var axes_per_dim: stdx.BoundedArray(stdx.BoundedArray(usize, Shape.MAX_RANK), Shape.MAX_RANK) = .empty;
        var used_mask: [Shape.MAX_RANK]bool = @splat(false);
        var globally_used: std.EnumSet(PhysicalAxisTag) = .empty;

        for (0..shape.rank()) |ax| {
            var dim_axes: stdx.BoundedArray(usize, Shape.MAX_RANK) = .empty;
            const spec = shape.partition(ax);

            if (spec.meshAxis()) |mesh_ax| {
                const binding_ = self.bindings.get(mesh_ax);
                for (binding_.slice()) |p_tag| {
                    for (view.axes.slice(), 0..) |v_ax, i| {
                        // Only use the axis if it's bound and hasn't been consumed by a previous dimension
                        if (v_ax.contains(p_tag) and !globally_used.contains(v_ax.tag)) {
                            dim_axes.appendAssumeCapacity(i);
                            globally_used.insert(v_ax.tag);
                            used_mask[i] = true;
                        }
                    }
                }
            }
            axes_per_dim.appendAssumeCapacity(dim_axes);
        }

        var replicated_axes: stdx.BoundedArray(usize, Shape.MAX_RANK) = .empty;
        for (0..view.axes.len) |i| {
            if (!used_mask[i]) replicated_axes.appendAssumeCapacity(i);
        }

        return .{
            .axes_per_dim = axes_per_dim,
            .replicated_axes = replicated_axes,
            .view = view,
        };
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
        const mapping = data.getDimMapping(shape);
        const dimensions = try allocator.alloc(*const dialects.shardy.DimensionShardingAttribute, shape.rank());

        // Note: here we map from the logical axis to physical. Is this a good idea ?
        // why not stay in logical realm ?
        for (0.., dimensions) |ax, *d| {
            d.* = switch (shape.partition(ax)) {
                .out_of_bound => @panic("undefined"),
                .replicated => .replicated(ctx),
                .open => .open(ctx, &.{}),
                else => d: {
                    const dim_phys_indices = mapping.axes_per_dim.get(ax);
                    if (dim_phys_indices.len == 0) {
                        break :d .replicated(ctx);
                    } else {
                        const axes = try allocator.alloc(*const dialects.shardy.AxisRefAttribute, dim_phys_indices.len);
                        for (dim_phys_indices.slice(), 0..) |p_idx, i| {
                            axes[i] = .named(ctx, @tagName(mapping.view.axes.get(p_idx).tag));
                        }
                        break :d .closed(ctx, axes);
                    }
                },
            };
        }

        const replicated_axes = try allocator.alloc(*const dialects.shardy.AxisRefAttribute, mapping.replicated_axes.len);
        for (replicated_axes, mapping.replicated_axes.slice()) |*r, p_idx| {
            r.* = .named(ctx, @tagName(mapping.view.axes.get(p_idx).tag));
        }

        return .init(ctx, .{ .mesh = data.name, .dimensions = dimensions, .replicated_axes = replicated_axes });
    }

    pub fn gspmdShardingAttrForShape(self: *const Mesh, parent_allocator: std.mem.Allocator, ctx: *mlir.Context, shape: Shape) !*const mlir.Attribute {
        var arena = try stdx.arenaWithCapacity(parent_allocator, 1024);
        defer arena.deinit();
        const allocator = arena.allocator();

        var has_sharding = false;
        for (0..shape.rank()) |ax| {
            if (shape.partition(ax).isSharded()) {
                has_sharding = true;
                break;
            }
        }
        if (!has_sharding) return .string(ctx, try allocator.dupe(u8, "{replicated}"));

        const mapping = self.getDimMapping(shape);
        var tile_shape: stdx.BoundedArray(i64, Shape.MAX_RANK + 1) = .empty;

        //  Calculate tile sizes per tensor dimension
        for (mapping.axes_per_dim.slice()) |dim_axes| {
            var combined: i64 = 1;
            for (dim_axes.slice()) |idx| {
                combined *= mapping.view.axes.get(idx).size;
            }
            tile_shape.appendAssumeCapacity(combined);
        }

        // Add replication dimension if physical axes are left over.
        const has_replication = mapping.replicated_axes.len > 0;
        if (has_replication) {
            var repl_size: i64 = 1;
            for (mapping.replicated_axes.slice()) |idx| {
                repl_size *= mapping.view.axes.get(idx).size;
            }
            tile_shape.appendAssumeCapacity(repl_size);
        }

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
        _ = try Placement.init(shape);
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
        const view = self.physicalView();
        const count: u32 = @intCast(view.total_devices);

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
        try writer.print("Mesh(name={s})\n", .{self.name});

        try writer.writeAll("Bindings:\n");
        for (self.logical.axes.slice(), self.logical.intents.slice(), self.bindings.slice()) |l_tag, l_intent, bind| {
            try writer.print("  - {s} ({s}) -> ", .{ l_tag, @tagName(l_intent) });

            if (bind.len == 0) {
                try writer.writeAll("unbound\n");
            } else {
                for (bind.slice(), 0..) |physical, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.writeAll(@tagName(physical));
                }
                try writer.writeAll("\n");
            }
        }

        const view = self.physicalView();

        try writer.print("Physical capacity: {d}\n", .{view.total_devices});

        try writer.writeAll("Physical axes: ");
        if (view.axes.len == 0) {
            try writer.writeAll("(none)\n");
        } else {
            for (view.axes.slice(), 0..) |axis, i| {
                if (i > 0) try writer.writeAll(" × ");
                try writer.print("{s}[{d}]", .{ @tagName(axis.tag), axis.size });
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("Geometries: ");
        if (view.axes.len == 0) {
            try writer.writeAll("(none)\n");
        } else {
            for (view.axes.slice(), 0..) |axis, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s}={s}", .{
                    @tagName(axis.tag),
                    @tagName(axis.geometry orelse .point_to_point),
                });
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("Logical coordinates:\n");
        var coords: [MAX_MESH_RANK]usize = @splat(0);
        try self.formatLogicalCoordinates(writer, &coords, 0);
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
        try writer.writeAll(", dev=");
        for (0..size) |coord| {
            coords[last] = coord;
            if (coord > 0) try writer.writeAll(",");
            var count: usize = 0;
            for (self.devicesInCanonicalOrder()) |device| {
                if (self.matchesLogicalCoordinates(device, coords[0..axes.len])) count += 1;
            }
            // Keep replicas grouped so each entry corresponds to one last-axis coordinate.
            if (count != 1) try writer.writeAll("[");
            var written: usize = 0;
            for (self.devicesInCanonicalOrder()) |device| {
                if (!self.matchesLogicalCoordinates(device, coords[0..axes.len])) continue;
                if (written > 0) try writer.writeAll(",");
                try writer.print("{d}", .{device.id});
                written += 1;
            }
            if (count != 1) try writer.writeAll("]");
        }
        try writer.writeAll("\n");
    }

    fn matchesLogicalCoordinates(self: Mesh, device: Device, coords: []const usize) bool {
        for (coords, 0..) |expected, logical_axis| {
            var coord: usize = 0;
            var used: std.EnumSet(PhysicalAxisTag) = .empty;
            for (self.binding(logical_axis)) |tag| {
                const sources = self.foldSources(tag) orelse &.{tag};
                for (sources) |source| {
                    if (used.contains(source)) continue;
                    used.insert(source);
                    const info = self.physical.axisInfo(source) orelse continue;
                    const depth = self.physical.axis_traversal.depth(source) orelse continue;
                    coord = coord * @as(usize, @intCast(info.size)) + device.coords[depth];
                }
            }
            if (coord != expected) return false;
        }
        return true;
    }
};

pub const Strategy = struct {
    pub const Fold = struct {
        target: PhysicalAxisTag,
        sources: PhysicalList,
    };
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: PhysicalList,
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
        const raw_tag = Shape.toTag(logical);
        const logical_tag: []const u8 = std.mem.span(raw_tag);

        for (self.bindings.slice()) |*b| {
            if (std.mem.eql(u8, std.mem.span(b.logical), logical_tag)) {
                b.physical.appendAssumeCapacity(physical);
                return;
            }
        }

        var list: PhysicalList = .empty;
        list.appendAssumeCapacity(physical);
        self.bindings.appendAssumeCapacity(.{ .logical = raw_tag, .physical = list });
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

        const avail = available.slice();
        var counters = std.EnumArray(LogicalAxisIntent, usize).initFill(0);
        var claimed: std.EnumSet(PhysicalAxisTag) = .empty;

        inline for (std.meta.tags(LogicalAxisIntent)) |target_intent| {
            for (logical.axes.slice(), logical.intents.slice()) |l_tag, l_intent| {
                if (l_intent != target_intent) continue;

                const i = counters.get(target_intent);
                const p_tag = switch (target_intent) {
                    .high_bandwidth => avail[i % avail.len],
                    .balanced => avail[(avail.len / 2 + i) % avail.len],
                    .low_bandwidth => avail[avail.len - 1 - (i % avail.len)],
                };

                // Leave a logical axis replicated when its preferred physical axis is taken.
                if (!claimed.contains(p_tag)) {
                    strategy.addBinding(l_tag, p_tag);
                    claimed.insert(p_tag);
                }
                counters.set(target_intent, i + 1);
            }
        }

        return strategy;
    }
};

/// For a given shape, compute the shape of the slice each shard will receive.
pub fn shardedShape(shape: Shape) error{IncompatibleSharding}!Shape {
    if (shape._sharding.mesh != null) {
        const pl: Placement = try .init(shape);
        var local_shape = pl.shape;
        local_shape._sharding = .replicated;
        return local_shape;
    } else {
        // fully replicated
        return shape;
    }
}

/// Precompute mesh information common to all devices for a given shape.
pub const Placement = struct {
    pub const Slices = stdx.BoundedArray(Slice1d, Shape.MAX_RANK);
    pub const Slice1d = struct {
        start: i64,
        size: i64,
    };

    // Note this shape is mainly use to indicate the dims of the sharded buffer.
    // We may want to only store that.
    shape: Shape,
    global_shape: if (builtin.mode == .debug) Shape else void,
    byte_steps: @Vector(MAX_MESH_RANK, u64),

    pub fn init(shape: Shape) error{IncompatibleSharding}!Placement {
        var pl: Placement = .{
            .shape = shape, // modified below
            .global_shape = if (builtin.mode == .debug) shape else {},
            .byte_steps = @splat(0),
        };
        // mesh == null => replicated => zero byte steps
        if (shape._sharding.mesh == null) return pl;
        const mesh_ = shape._sharding.mesh.?;
        var used_axes: u8 = 0;
        var byte_steps: [MAX_MESH_RANK]u64 = @splat(0);
        const byte_strides = shape.computeByteStrides();
        for (shape.dims(), 0..) |dim, ax| {
            const logical_axis = shape.partition(ax).meshAxis() orelse continue;
            const bit = @as(u8, 1) << @intCast(logical_axis);
            // Reusing a logical axis partitions only the first tensor dimension.
            if (used_axes & bit != 0) continue;
            used_axes |= bit;
            const count = mesh_.grid.axis_sizes[logical_axis];
            if (@rem(dim, count) != 0) return error.IncompatibleSharding;
            const local_dim = @divExact(dim, count);
            pl.shape._dims.set(ax, local_dim);
            byte_steps[logical_axis] = @intCast(local_dim * byte_strides.get(ax));
        }
        pl.byte_steps = byte_steps;
        return pl;
    }

    pub fn mesh(pl: *const Placement) ?*const Mesh {
        return pl.shape._sharding.mesh;
    }

    pub fn slices(pl: *const Placement, device_index: usize) Slices {
        var res: Slices = .{ .buffer = undefined, .len = pl.shape.rank() };
        if (pl.mesh() == null) {
            for (res.slice(), pl.shape.dims()) |*slice, dim| slice.* = .{ .start = 0, .size = dim };
            return res;
        }
        const grid = &pl.mesh().?.grid;
        std.debug.assert(device_index < grid.device_count);
        const coords: [MAX_MESH_RANK]u8 = grid.logical_coords[device_index];
        var used_axes: u8 = 0;
        for (res.slice(), pl.shape.dims(), 0..) |*r, size, ax| {
            var coordinate: i64 = 0;
            if (pl.shape.partition(ax).meshAxis()) |logical_axis| {
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
            std.debug.assert(pl.global_shape.eql(slice.shape));
        }
        return slice.constData().ptr;
    }

    /// device_index is a canonical ordinal, never a PJRT device ID.
    /// base_ptr must come from basePtr for this placement and source slice.
    pub fn shardPtr(pl: *const Placement, device_index: usize, base_ptr: [*]const u8) [*]const u8 {
        return base_ptr + pl.byteOffset(device_index);
    }

    fn byteOffset(pl: *const Placement, device_index: usize) usize {
        const mesh_ = pl.mesh() orelse return 0;
        std.debug.assert(device_index < mesh_.grid.device_count);
        const coords: [MAX_MESH_RANK]u8 = mesh_.grid.logical_coords[device_index];
        const steps: [MAX_MESH_RANK]u64 = pl.byte_steps;
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
            .shape = pl.shape,
            .offset_bytes = offset_bytes,
            .byte_strides = slice.byte_strides,
        };
    }

    pub fn format(pl: Placement, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print(
            "Placement(shape={f}, byte_steps={any})\n",
            .{ pl.shape, pl.byte_steps },
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

        expect_error: ?anyerror = null,
        expected_sdy: ?[]const u8 = null,
        expected_shards: []const ExpectedShard = &.{},

        pub fn format(scenario: Scenario, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Scenario:\n", .{});
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
            try std.testing.expectError(err, Sharding.Placement.init(s.shape));
            return;
        }

        // Verify Shard Slices (Math)
        if (s.expected_shards.len > 0) {
            const pl: Placement = try .init(s.shape);
            errdefer std.log.warn("Expected shards failed. Got {f}", .{pl});
            try std.testing.expectEqual(s.expected_shards.len, ordered_devices.len);
            for (s.expected_shards, ordered_devices, 0..) |expected, device, device_index| {
                {
                    try std.testing.expectEqual(expected.device_id, device.id);

                    // Check pl.slices(device)
                    const actual_slices = pl.slices(device_index);

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

                    try std.testing.expectEqualSlices(i64, expected_sizes, pl.shape.dims());
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
        .mesh = &mesh,
        // No partitioning on shape
        .shape = Shape.init(.{ .batch = 8 }, .f32),
        .expected_sdy = "#sdy.sharding<@folded_mesh, [{}], replicated={\"link_x\", \"link_y\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 1, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 2, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 3, .slices = &.{.{ 0, 8 }} },
        },
    });
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
        .mesh = &mesh,
        .shape = Shape.init(.{ .batch = 4, .model = 4 }, .f32)
            .withPartitioning(&mesh, .{ .batch = .batch, .model = .model }),
        .expected_sdy = "#sdy.sharding<@suggested_mesh, [{\"link_z\"}, {\"link_x\"}], replicated={\"link_y\"}>",
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
        .mesh = &mesh,
        .shape = Shape.init(.{ .model = 4, .experts = 4 }, .f32)
            .withPartitioning(&mesh, .{ .model = .model, .experts = .experts }),
        .expected_sdy = "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 1 }, .{ 0, 4 } } },
            .{ .device_id = 1, .slices = &.{ .{ 1, 1 }, .{ 0, 4 } } },
            .{ .device_id = 2, .slices = &.{ .{ 2, 1 }, .{ 0, 4 } } },
            .{ .device_id = 3, .slices = &.{ .{ 3, 1 }, .{ 0, 4 } } },
        },
    });
    // Swapping tensor dimensions must not change which logical axis owns the devices.
    try runner.run(.{
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
            .mesh = &mesh,
            .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(&mesh, .{ .model = .model }),
            .expected_sdy = "#sdy.sharding<@folded_mesh, [{\"link_x\", \"link_y\"}]>",
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
            .mesh = &mesh,
            .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(&mesh, .{ .model = .model }),
            .expected_sdy = "#sdy.sharding<@folded_mesh, [{\"link_y\", \"link_x\"}]>",
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
        .mesh = &mesh,
        .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(&mesh, .{ .model = .model }),
        .expected_sdy = "#sdy.sharding<@strategy_fold, [{\"link_x\"}], replicated={\"link_y\"}>",
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
        .mesh = &mesh,
        .shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(&mesh, .{ .batch = .open, .model = .replicated }),
        .expected_sdy = "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>",
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
        .mesh = &mesh,
        // Note: .context and .model are swapped compared to .link_x and .link_z
        .shape = Shape.init(.{ .batch = 4, .context = 4, .model = 4 }, .f32)
            .withPartitioning(&mesh, .{ .batch = .batch, .model = .model, .context = .context }),
        .expected_sdy = "#sdy.sharding<@\"3d_mesh\", [{\"link_x\"}, {\"link_z\"}, {\"link_y\"}]>",
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
    const placement = try Placement.init(shape);
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

test "Placement byte steps and dot product exceed 32 bits" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());
    const physical = try runner.physical(.{Platform.MAX_NUM_DEVICES}, .{ .mesh = .torus });
    const mesh_ = try Mesh.init("wide", &physical, .mesh(.{ .model = .balanced }), .parseBindings(.{ .model = .link_x }));
    const shape = Shape.init(.{ .m = 2 * Platform.MAX_NUM_DEVICES, .n = 1 << 30 }, .f32).withPartitioning(&mesh_, .{ .m = .model });
    const placement = try Placement.init(shape);
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
