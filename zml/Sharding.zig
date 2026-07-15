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

data: *const Data,

pub const Error = error{ MissingLogicalBinding, IncompatibleSharding };

pub const MAX_MESH_RANK = 4;

var _replicated: [11]u8 align(@alignOf(Data)) = "_replicated".*;

// special value to make public apis more fluent.
pub const replicated: Sharding = .{ .data = @ptrCast(&_replicated) };

pub fn resolve(sharding: Sharding, platform: *const Platform) Sharding {
    return if (sharding.data == replicated.data) platform.replicated_sharding else sharding;
}

pub fn numPartitionsForLogicalAxis(sharding: Sharding, logical_axis: anytype) i64 {
    return sharding.data.numPartitionsForLogicalAxis(logical_axis);
}

pub fn devicesInCanonicalOrder(sharding: Sharding) []const Device {
    return sharding.data.physical.devices_in_canonical_order;
}

pub fn format(sharding: Sharding, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    return sharding.data.format(writer);
}

pub const Partitioner = union(enum) {
    shardy,
    gspmd,

    pub fn fromTarget(target: Target) Partitioner {
        return switch (target) {
            .cpu, .cuda, .rocm, .tpu, .oneapi, .neuron, .metal => .shardy,
        };
    }
};

pub const Partitioning = struct {
    partitioner: Partitioner,
    shardings: []const Sharding,

    pub fn init(partitioner: Partitioner, shardings: []const Sharding) !Partitioning {
        stdx.debug.assert(shardings.len >= 1, "Waiting at leat 1 sharding strategy to be implemented", .{});
        var plan: Partitioning = .{ .partitioner = partitioner, .shardings = shardings };

        const first = plan.primarySharding();
        const partitions = first.data.numPartitions();
        const replicas = first.data.numReplicas();

        for (shardings[1..]) |s| {
            if (s.data.numPartitions() != partitions or s.data.numReplicas() != replicas) {
                // todo: deviceAssignments should also be checked for consistency here, but for simplicity we just check the cardinality numbers
                return error.InconsistentShardingCardinality;
            }
        }

        return plan;
    }

    pub fn numPartitions(self: Partitioning) i32 {
        const primary = self.primarySharding();
        return primary.data.numPartitions();
    }

    pub fn numReplicas(self: Partitioning) i32 {
        const primary = self.primarySharding();
        return primary.data.numReplicas();
    }

    pub fn numDevices(self: Partitioning) i32 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn deviceAssignment(self: Partitioning, allocator: std.mem.Allocator) ![]usize {
        return switch (self.partitioner) {
            .shardy, .gspmd => blk: {
                const sharding = self.primarySharding();
                break :blk sharding.data.deviceAssignment(allocator);
            },
        };
    }

    pub fn tensorShardingAttr(self: Partitioning, allocator: std.mem.Allocator, ctx: *mlir.Context, shape: Shape, sharding: ?Sharding) !*const mlir.Attribute {
        const selected_sharding = sharding orelse try self.selectSharding(shape);
        return switch (self.partitioner) {
            .shardy => (try selected_sharding.data.sdyShardingAttrForShape(allocator, ctx, shape)).asAttr(),
            .gspmd => try selected_sharding.data.gspmdShardingAttrForShape(allocator, ctx, shape),
        };
    }

    pub fn localShapeForShape(self: Partitioning, shape: Shape) !Shape {
        const sharding = try self.selectSharding(shape);
        const ordered_devices = sharding.devicesInCanonicalOrder();
        if (ordered_devices.len == 0) return error.InvalidPhysicalMesh;
        return sharding.shardedShape(shape);
    }

    pub fn numPartitionsForLogicalAxis(self: Partitioning, shape: Shape, logical_axis: anytype) !i64 {
        const sharding = try self.selectSharding(shape);
        return sharding.data.numPartitionsForLogicalAxis(logical_axis);
    }

    pub fn shardableDim(self: Partitioning, shape: Shape, axis: anytype, must_divide: i64) !DimSharding {
        const ax = shape.axis(axis);
        const spec = shape.partition(ax);
        if (spec != .axis) return .replicated;

        const sharding = try self.selectSharding(shape);
        return sharding.shardableDim(shape.dim(ax), spec.axis, must_divide);
    }

    pub fn sdyPerValueShardingAttr(self: Partitioning, allocator: std.mem.Allocator, ctx: *mlir.Context, shapes: []const Shape) !*const mlir.Attribute {
        stdx.debug.assert(self.partitioner == .shardy, "sdyPerValueShardingAttr requires shardy partitioner", .{});

        const shardings = try allocator.alloc(*const dialects.shardy.TensorShardingAttribute, shapes.len);
        for (shapes, 0..) |shape, i| {
            const sharding = try self.selectSharding(shape);
            shardings[i] = try sharding.data.sdyShardingAttrForShape(allocator, ctx, shape);
        }
        return dialects.shardy.TensorShardingPerValueAttribute.init(ctx, shardings).asAttr();
    }

    pub fn sdyManualAxesAttr(self: Partitioning, allocator: std.mem.Allocator, ctx: *mlir.Context, in_shapes: []const Shape, out_shapes: []const Shape) !*const mlir.Attribute {
        stdx.debug.assert(self.partitioner == .shardy, "sdyManualAxesAttr requires shardy partitioner", .{});

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
            const sharding = try self.selectSharding(shape);
            const attr = try sharding.data.sdyShardingAttrForShape(allocator, ctx, shape);
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
            const sharding = try self.selectSharding(shape);
            const attr = try sharding.data.sdyShardingAttrForShape(allocator, ctx, shape);
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

    pub fn selectSharding(self: Partitioning, shape: Shape) !Sharding {
        return pickSharding(self.shardings, shape, .any_covering) orelse error.NoSuitableSharding;
    }

    fn primarySharding(self: Partitioning) Sharding {
        return self.shardings[0];
    }
};

pub const SelectShardingMode = enum {
    any_covering,
    explicit_axis_binding,
};

pub fn pickSharding(shardings: []const Sharding, shape: Shape, mode: SelectShardingMode) ?Sharding {
    if (mode == .explicit_axis_binding and !shapeHasAxisPartition(shape)) return null;

    for (shardings) |sharding| {
        if (sharding.data.covers(shape)) return sharding;
    }
    return null;
}

fn shapeHasAxisPartition(shape: Shape) bool {
    for (0..shape.rank()) |ax| {
        if (shape.partition(ax) == .axis) return true;
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

pub fn shardableDim(sharding: Sharding, dim: i64, logical_axis: anytype, must_divide: i64) DimSharding {
    const partitions = sharding.numPartitionsForLogicalAxis(logical_axis);

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

            const fields = std.meta.fields(PhysicalAxisTag);
            inline for (fields, 0..) |field, i| {
                if (i > 0) try writer.writeAll(", ");
                const tag = @field(PhysicalAxisTag, field.name);
                try writer.writeAll(field.name);
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
            .cpu, .metal => &.{.bus},
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

            var axis_sizes = [_]usize{1} ** MAX_MESH_RANK;
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
            .oneapi, .metal => cpu(allocator, platform_devices),
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

        var coords_buf: [MAX_MESH_RANK]usize = [_]usize{0} ** MAX_MESH_RANK;
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
    /// Needs the fastest possible path (e.g. Weight sharding / All-Reduce).
    /// Requires high-bandwidth, low-latency interconnects (Cores/NVLink).
    high_bandwidth,

    /// Needs reliable throughput but can handle some latency (e.g. Shuffling / All-to-All).
    /// Suitable for chip-to-chip interconnects (NeuronLink/ICI).
    balanced,

    /// Can tolerate lower bandwidth and higher latency (e.g. Batch sharding).
    /// Suitable for system bus or network (PCIe/Ethernet).
    low_bandwidth,
};

// Defines the logical mesh used for sharding the model and data.
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

        inline for (std.meta.fields(T)) |field| {
            const value = @field(axes_, field.name);
            axes.appendAssumeCapacity(Shape.toTag(field));
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

pub const AxisList = stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK);
pub const Binding = struct {
    logical: Shape.Tag,
    physical: AxisList,
};
pub const Bindings = stdx.BoundedArray(Binding, MAX_MESH_RANK);

pub const Fold = struct {
    target: PhysicalAxisTag,
    sources: AxisList,
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

pub const Data = struct {
    name: []const u8,
    physical: *const PhysicalMesh,
    logical: LogicalMesh,

    /// Compact binding table: logical axis -> physical axes
    bindings: Bindings,

    /// Explicit folding rules: kept axis -> ordered source axes.
    folds: Folds,
    folds_consumed: std.EnumSet(PhysicalAxisTag),

    pub fn binding(self: *const Data, tag: Shape.Tag) ?[]const PhysicalAxisTag {
        for (self.bindings.slice()) |*b| {
            if (b.logical == tag) return b.physical.slice();
        }
        return null;
    }

    pub fn init(
        name: []const u8,
        physical: *const PhysicalMesh,
        logical: LogicalMesh,
        strategy: Strategy,
    ) !Data {
        const axis_order = physical.axisOrder().slice();
        if (axis_order.len == 0) return error.InvalidPhysicalMesh;
        if (strategy.bindings.len == 0) return error.InvalidStrategy;

        var bindings: Bindings = .empty;
        for (strategy.bindings.slice()) |bind| {
            bindings.appendAssumeCapacity(.{ .logical = bind.logical, .physical = bind.physical });
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

        return .{
            .name = name,
            .logical = logical,
            .physical = physical,
            .bindings = bindings,
            .folds = folds,
            .folds_consumed = folds_consumed,
        };
    }

    fn foldSources(self: *const Data, tag: PhysicalAxisTag) ?[]const PhysicalAxisTag {
        for (self.folds.slice()) |*f| {
            if (f.target == tag) return f.sources.slice();
        }
        return null;
    }

    pub fn physicalView(self: *const Data) PhysicalView {
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

    pub fn logicalIndexFromCoords(self: *const Data, coords: []const u8) usize {
        return self.physical.linearIndexFromCoords(coords);
    }

    pub fn numPartitions(self: *const Data) i32 {
        return @intCast(self.physicalView().total_devices);
    }

    pub fn numPartitionsForLogicalAxis(self: *const Data, logical_axis: anytype) i64 {
        const logical_tag = Shape.toTag(logical_axis);
        const bound_axes = self.binding(logical_tag) orelse return 1;

        var physical_axes: std.EnumSet(PhysicalAxisTag) = .empty;
        for (bound_axes) |bound_axis| {
            if (self.foldSources(bound_axis)) |sources| {
                for (sources) |source| physical_axes.insert(source);
            } else {
                physical_axes.insert(bound_axis);
            }
        }

        var partitions: i64 = 1;
        for (std.enums.values(PhysicalAxisTag)) |physical_axis| {
            if (physical_axes.contains(physical_axis)) {
                partitions *= self.physical.axis(physical_axis);
            }
        }
        return partitions;
    }

    pub fn numReplicas(_: *const Data) i32 {
        return 1;
    }

    pub fn numDevices(self: *const Data) i32 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn covers(sharding: *const Data, shape: Shape) bool {
        for (shape._partitioning.slice()) |partitioning| {
            switch (partitioning) {
                .axis => |tag| if (sharding.binding(tag) == null) return false,
                else => {},
            }
        }
        return true;
    }

    pub fn sdyMeshAttr(self: *const Data, allocator: std.mem.Allocator) ![]const u8 {
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
        /// For each tensor dimension, the indices of physical axes (in PhysicalView) sharding it.
        axes_per_dim: stdx.BoundedArray(stdx.BoundedArray(usize, Shape.MAX_RANK), Shape.MAX_RANK),
        /// Indices of physical axes used for replication.
        replicated_axes: stdx.BoundedArray(usize, Shape.MAX_RANK),
        /// Reference to the view for sizes.
        view: PhysicalView,
    };

    /// Common logic to map tensor dimensions to physical mesh indices.
    fn getDimMapping(self: *const Data, shape: Shape) DimMapping {
        const view = self.physicalView();
        var axes_per_dim: stdx.BoundedArray(stdx.BoundedArray(usize, Shape.MAX_RANK), Shape.MAX_RANK) = .empty;
        var used_mask: [Shape.MAX_RANK]bool = @splat(false);
        var globally_used: std.EnumSet(PhysicalAxisTag) = .empty;

        for (0..shape.rank()) |ax| {
            var dim_axes: stdx.BoundedArray(usize, Shape.MAX_RANK) = .empty;
            const spec = shape.partition(ax);

            if (spec == .axis) {
                if (self.binding(spec.axis)) |binding_| {
                    for (binding_) |p_tag| {
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
        data: *const Data,
        parent_allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        shape: Shape,
    ) !*const dialects.shardy.TensorShardingAttribute {
        var arena = try stdx.arenaWithCapacity(parent_allocator, 1024);
        defer arena.deinit();
        const allocator = arena.allocator();
        var any_explicit = false;
        for (0..shape.rank()) |ax| {
            if (shape.partition(ax) != .unknown) {
                any_explicit = true;
                break;
            }
        }
        const all_replicated = !any_explicit;

        const mapping = data.getDimMapping(shape);
        const dimensions = try allocator.alloc(*const dialects.shardy.DimensionShardingAttribute, shape.rank());

        for (0.., dimensions) |ax, *d| {
            const spec = shape.partition(ax);
            d.* = switch (spec) {
                .axis => |logical_tag| d: {
                    if (data.binding(logical_tag)) |_| {
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
                    } else {
                        break :d .open(ctx, &.{});
                    }
                },
                .replicated => .replicated(ctx),
                .open, .unknown => if (all_replicated) .replicated(ctx) else .open(ctx, &.{}),
            };
        }

        const replicated_axes = try allocator.alloc(*const dialects.shardy.AxisRefAttribute, mapping.replicated_axes.len);
        for (replicated_axes, mapping.replicated_axes.slice()) |*r, p_idx| {
            r.* = .named(ctx, @tagName(mapping.view.axes.get(p_idx).tag));
        }

        return .init(ctx, .{ .mesh = data.name, .dimensions = dimensions, .replicated_axes = replicated_axes });
    }

    pub fn gspmdShardingAttrForShape(self: *const Data, parent_allocator: std.mem.Allocator, ctx: *mlir.Context, shape: Shape) !*const mlir.Attribute {
        var arena = try stdx.arenaWithCapacity(parent_allocator, 1024);
        defer arena.deinit();
        const allocator = arena.allocator();

        var has_sharding = false;
        for (0..shape.rank()) |ax| {
            if (shape.partition(ax) == .axis) {
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

        // Emit explicit device assignment list to avoid XLA requiring
        // a sharding attribute on the custom-call instruction.
        const ids = try self.deviceAssignment(allocator);
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

    pub fn deviceAssignment(self: *const Data, allocator: std.mem.Allocator) ![]usize {
        const view = self.physicalView();
        const count: usize = @intCast(view.total_devices);

        var ids = try allocator.alloc(usize, count);
        @memset(ids, std.math.maxInt(usize));

        for (self.physical.devices_in_canonical_order) |d| {
            const coords = d.coords;
            const idx = self.physical.linearIndexFromCoords(coords);
            ids[idx] = d.id;
        }

        for (ids) |id| {
            if (id == std.math.maxInt(usize)) return error.MissingDeviceInTile;
        }

        return ids;
    }

    pub fn format(self: Data, writer: *std.Io.Writer) !void {
        try writer.print("Sharding(name={s})\n", .{self.name});

        try writer.writeAll("Bindings:\n");
        for (self.logical.axes.slice(), self.logical.intents.slice()) |l_tag, l_intent| {
            try writer.print("  - {s} ({s}) -> ", .{ l_tag, @tagName(l_intent) });

            if (self.binding(l_tag)) |axes| {
                if (axes.len == 0) {
                    try writer.writeAll("replicated\n");
                } else {
                    for (axes, 0..) |p, i| {
                        if (i > 0) try writer.writeAll(", ");
                        try writer.writeAll(@tagName(p));
                    }
                    try writer.writeAll("\n");
                }
            } else {
                try writer.writeAll("unbound\n");
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
    }
};

pub const Strategy = struct {
    pub const PhysicalList = stdx.BoundedArray(PhysicalAxisTag, MAX_MESH_RANK);
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: PhysicalList,
    };
    pub const Bindings = stdx.BoundedArray(Strategy.Binding, MAX_MESH_RANK);
    pub const Fold = struct {
        target: PhysicalAxisTag,
        sources: PhysicalList,
    };
    pub const Folding = stdx.BoundedArray(Strategy.Fold, MAX_MESH_RANK);

    bindings: Strategy.Bindings,
    folding: Folding,

    // TODO: remove .init, promote .parseBindings to .init
    // It's currently an error to call .init and never calling `addBinding`
    pub const init: Strategy = .{ .bindings = .empty, .folding = .empty };

    pub fn parseBindings(bindings: anytype) Strategy {
        const err_msg = "Strategy.parseBindings excepts fields to be PhysicalAxisTag or tuple of PhysicalAxisTag, got {}";
        var res: Strategy = .{ .bindings = .empty, .folding = .empty };
        const fields = @typeInfo(@TypeOf(bindings)).@"struct".fields;
        if (fields.len == 0) @compileError("Strategy.parseBindings requires at least one binding");
        inline for (fields) |field_info| {
            switch (@typeInfo(field_info.type)) {
                .enum_literal, .@"enum" => res.addBinding(field_info, @field(bindings, field_info.name)),
                .@"struct" => |struct_info| {
                    stdx.debug.assertComptime(struct_info.is_tuple, err_msg, .{@TypeOf(bindings)});
                    inline for (@field(bindings, field_info.name)) |axis_tag| {
                        res.addBinding(field_info, axis_tag);
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

        inline for (std.meta.tags(LogicalAxisIntent)) |target_intent| {
            for (logical.axes.slice(), logical.intents.slice()) |l_tag, l_intent| {
                if (l_intent != target_intent) continue;

                const i = counters.get(target_intent);
                const p_tag = switch (target_intent) {
                    .high_bandwidth => avail[i % avail.len],
                    .balanced => avail[(avail.len / 2 + i) % avail.len],
                    .low_bandwidth => avail[avail.len - 1 - (i % avail.len)],
                };

                strategy.addBinding(l_tag, p_tag);
                counters.set(target_intent, i + 1);
            }
        }

        return strategy;
    }
};

/// For a given shape, compute the shape of the slice each shard will receive.
pub fn shardedShape(sharding: Sharding, shape: Shape) Error!Shape {
    const pl = try sharding.placement(shape);
    return pl.shape;
}

pub fn placement(sharding: Sharding, shape: Shape) Error!Placement {
    return .init(sharding, shape);
}

/// Precompute sharding information common to all devices for a given shape.
pub const Placement = struct {
    pub const Slices = stdx.BoundedArray(Slice1d, Shape.MAX_RANK);
    pub const Slice1d = struct {
        start: i64,
        size: i64,
    };

    sharding: Sharding,
    // Note this shape is mainly use to indicate the dims of the sharded buffer.
    // We may want to only store that.
    shape: Shape,
    global_shape: if (builtin.mode == .Debug) Shape else void,

    axis_plans: stdx.BoundedArray(AxisSplit, Shape.MAX_RANK),

    pub fn init(sharding: Sharding, shape: Shape) Error!Placement {
        var pl: Placement = .{
            .sharding = sharding,
            .shape = shape, // modified below
            .global_shape = if (builtin.mode == .Debug) shape else {},
            .axis_plans = .empty, // set below
        };
        var used_axes: std.EnumSet(PhysicalAxisTag) = .empty;

        for (0..shape.rank()) |ax| {
            const axis_index: u8 = @intCast(ax);
            pl.axis_plans.appendAssumeCapacity(try axisSplit(sharding, shape, &used_axes, axis_index));
        }

        for (pl.shape._dims.slice(), shape.dims(), pl.axis_plans.slice()) |*shard_dim, dim, plan| {
            shard_dim.* = @divExact(dim, plan.num_devices);
        }
        return pl;
    }

    pub fn slices(pl: *const Placement, device: Device.Coords) Slices {
        var res: Slices = .{ .buffer = undefined, .len = pl.shape.rank() };
        for (res.slice(), pl.shape.dims(), pl.axis_plans.slice()) |*r, size, plan| {
            const start = plan.linearIndex(device) * size;
            r.* = .{
                .start = start,
                .size = size,
            };
        }
        return res;
    }

    pub fn shardPtr(pl: *const Placement, device: Device.Coords, slice: Slice) [*]const u8 {
        if (builtin.mode == .Debug) {
            // there is a bug in caller code that used a placement for a different shape
            std.debug.assert(pl.global_shape.eql(slice.shape));
        }

        var ptr: [*]const u8 = slice.constData().ptr;
        for (pl.shape.dims(), pl.axis_plans.slice(), slice.byte_strides.slice()) |size, plan, stride| {
            const start = plan.linearIndex(device) * size;
            ptr = ptr[@intCast(start * stride)..];
        }
        return ptr;
    }

    /// Given a Slice, returns the subslice corresponding to a specific device.
    /// Depending on the layout, the sub-slice may or may not be contiguous.
    pub fn shardSlice(pl: *const Placement, device: Device.Coords, slice: Slice) Slice {
        const shard_ptr = pl.shardPtr(device, slice);
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
        const ordered_devices = pl.sharding.devicesInCanonicalOrder();

        try pl.formatPlacementSummary(pl.shape, ordered_devices.len, writer);
        try writer.writeAll("Per-shard placement:\n");
        for (ordered_devices, 0..) |device, shard_index| {
            try writer.print("└─ Shard[{d}] device={d} coords={any}, slices=[", .{ shard_index, device.id, device.coords });
            for (0.., pl.slices(device.coords).slice()) |ax, s| {
                if (ax > 0) try writer.writeAll(", ");
                const axis_label = pl.shape.debugTag(ax);
                try writer.print("{s}:[{d}:{d}]", .{ axis_label, s.start, s.start + s.size });
            }
            try writer.writeAll("]\n");
        }

        for (0.., pl.axis_plans.slice()) |ax, plan| {
            try writer.print("- axis{d} -> plan {f}\n", .{ ax, plan });
        }
    }

    fn formatPlacementSummary(pl: Placement, shape: Shape, shard_count: usize, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Placement(shape={f} shards={d})\n", .{ shape, shard_count });

        try writer.writeAll("Axis partitioning summary:\n");
        for (0.., pl.slices(@splat(0)).slice()) |ax, slice| {
            const dim = shape.dim(ax);
            const spec = shape.partition(ax);
            const axis_label = shape.debugTag(ax);

            const shard_size: i64 = slice.size;
            const shards_count: i64 = if (shard_size > 0 and @rem(dim, shard_size) == 0)
                @divExact(dim, shard_size)
            else
                0;

            try writer.print(
                "└─ {s}: dim={d}, spec={s}, shard_size={d}, shards={d}\n",
                .{ axis_label, dim, @tagName(spec), shard_size, shards_count },
            );
        }
    }
};

const AxisSplit = struct {
    /// For each physical coordinate depth, the contribution of that coordinate
    /// to the linear shard index. Unused depths have stride 0.
    coord_strides: Device.Coords,
    counts: Device.Coords,
    num_devices: u32,

    pub const empty: AxisSplit = .{
        .coord_strides = @splat(0),
        .counts = @splat(0),
        .num_devices = 1,
    };

    pub fn add(split: *AxisSplit, size: u8, depth: u8) void {
        for (&split.coord_strides) |*stride| stride.* *= size;
        split.coord_strides[depth] = 1;
        split.counts[depth] = size;
        split.num_devices *= size;
    }

    pub fn linearIndex(split: AxisSplit, device_coords: Device.Coords) u32 {
        @setRuntimeSafety(false);
        const coords_u8: @Vector(MAX_MESH_RANK, u8) = device_coords;
        const strides: @Vector(MAX_MESH_RANK, u8) = split.coord_strides;
        return @reduce(.Add, coords_u8 * strides);
    }

    pub fn format(split: AxisSplit, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Plan(counts={any},strides={any})", .{ split.counts, split.coord_strides });
    }
};

fn axisSplit(
    sharding: Sharding,
    shape: Shape,
    used_axes: *std.EnumSet(PhysicalAxisTag),
    axis_index: u8,
) !AxisSplit {
    const dim = shape.dim(axis_index);
    const spec = shape.partition(axis_index);

    switch (spec) {
        .axis => |logical_tag| {
            const binding = sharding.data.binding(logical_tag) orelse return error.MissingLogicalBinding;

            // Calculate the split based on the physical coordinates of the current device.
            return try calculateSplit(sharding, dim, binding, used_axes);
        },
        else => return .empty,
    }
}

fn calculateSplit(
    sharding: Sharding,
    dim: i64,
    binding: []const PhysicalAxisTag,
    used_axes: *std.EnumSet(PhysicalAxisTag),
) !AxisSplit {
    var plan: AxisSplit = .empty;

    for (binding) |p_tag| {
        // Expand the physical tag: check if it's a target for folded source axes.
        if (sharding.data.foldSources(p_tag)) |sources| {
            for (sources) |src| {
                if (used_axes.contains(src)) continue;
                addPhysicalToSplit(sharding, &plan, src);
                used_axes.insert(src);
            }
        } else {
            // Standard case: directly bound, not part of an explicit fold.
            if (used_axes.contains(p_tag)) continue;
            addPhysicalToSplit(sharding, &plan, p_tag);
            used_axes.insert(p_tag);
        }
    }

    if (plan.num_devices > 0 and @rem(dim, plan.num_devices) != 0) {
        return error.IncompatibleSharding;
    }
    return plan;
}

/// Extract coordinate data from the physical mesh for a specific axis.
fn addPhysicalToSplit(sharding: Sharding, plan: *AxisSplit, tag: PhysicalAxisTag) void {
    const physical = sharding.data.physical;
    const info = physical.axisInfo(tag) orelse return;
    const depth = physical.axis_traversal.depth(tag) orelse return;
    plan.add(@intCast(info.size), depth);
}

const ShardingTest = struct {
    pub const ExpectedShard = struct {
        // TODO: remove device_id from here, we always have ids from 0 to N
        device_id: usize,
        slices: []const [2]i64,
    };

    pub const Scenario = struct {
        sharding: Sharding.Data,
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
        const N = info.fields.len;
        var sizes: [N]usize = undefined;
        inline for (info.fields, sizes[0..]) |field, *s| {
            s.* = @intCast(@field(dims, field.name));
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
        const sharding = s.sharding;
        errdefer std.log.warn("Failed sharding {f}", .{s});

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

            const actual_attr = try sharding.sdyShardingAttrForShape(self.allocator, ctx, s.shape);
            var writer: std.Io.Writer.Allocating = .init(self.allocator);
            defer writer.deinit();
            try actual_attr.asAttr().format(&writer.writer);
            try std.testing.expectEqualStrings(expected_attr, writer.written());
        }

        // Verify Placement logic / Error
        const sharding_ref: Sharding = .{ .data = &sharding };
        const ordered_devices = sharding_ref.devicesInCanonicalOrder();
        if (s.expect_error) |err| {
            errdefer std.log.warn("Expected error failed", .{});
            try std.testing.expect(ordered_devices.len > 0);
            try std.testing.expectError(err, sharding_ref.placement(s.shape));
            return;
        }

        // Verify Shard Slices (Math)
        if (s.expected_shards.len > 0) {
            const pl = try sharding_ref.placement(s.shape);
            errdefer std.log.warn("Expected shards failed. Got {f}", .{pl});
            try std.testing.expectEqual(s.expected_shards.len, ordered_devices.len);
            for (s.expected_shards, ordered_devices) |expected, device| {
                {
                    try std.testing.expectEqual(expected.device_id, device.id);

                    // Check pl.slices(device)
                    const actual_slices = pl.slices(device.coords);

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

test "sharding: unknown partitioning implies replication" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .batch = .low_bandwidth });

    const strategy: Strategy = .parseBindings(.{ .batch = .link_x });

    try runner.run(.{
        .sharding = try .init("folded_mesh", &physical, logical, strategy),
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

test "sharding: suggest strategy realization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });

    const logical: LogicalMesh = .mesh(.{
        .batch = .low_bandwidth,
        .model = .high_bandwidth,
    });

    const strategy: Strategy = .suggest(logical, &physical);

    try runner.run(.{
        .sharding = try .init("suggested_mesh", &physical, logical, strategy),
        .shape = Shape.init(.{ .batch = 4, .model = 4 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model }),
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

test "sharding: suggest folds logical axes with same intent" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical = try runner.physical(.{4}, .point_to_point);

    const logical: LogicalMesh = .mesh(.{
        .model = .high_bandwidth,
        .experts = .high_bandwidth,
    });

    const strategy: Strategy = .suggest(logical, &physical);

    try runner.run(.{
        .sharding = try .init("fold_mesh", &physical, logical, strategy),
        .shape = Shape.init(.{ .model = 4, .experts = 4 }, .f32)
            .withPartitioning(.{ .model = .model, .experts = .experts }),
        .expected_sdy = "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 1 }, .{ 0, 4 } } },
            .{ .device_id = 1, .slices = &.{ .{ 1, 1 }, .{ 0, 4 } } },
            .{ .device_id = 2, .slices = &.{ .{ 2, 1 }, .{ 0, 4 } } },
            .{ .device_id = 3, .slices = &.{ .{ 3, 1 }, .{ 0, 4 } } },
        },
    });
}

test "sharding: multiple physical axes on one logical dimension (folding)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .model = .high_bandwidth });

    {
        const strategy: Strategy = .parseBindings(.{ .model = .{ .link_x, .link_y } });
        try runner.run(.{
            .sharding = try .init("folded_mesh", &physical, logical, strategy),
            .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(.{ .model = .model }),
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
        try runner.run(.{
            .sharding = try .init("folded_mesh", &physical, logical, strategy),
            .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(.{ .model = .model }),
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

test "sharding: explicit strategy folding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .model = .high_bandwidth });

    var strategy: Strategy = .parseBindings(.{ .model = .link_x });
    strategy.addFold(.link_x, &.{ .link_x, .link_z });

    try runner.run(.{
        .sharding = try .init("strategy_fold", &physical, logical, strategy),
        .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(.{ .model = .model }),
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

test "sharding: open and replicated dimension mix" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const runner: ShardingTest = .init(arena.allocator());

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth });

    const strategy: Strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y });

    try runner.run(.{
        .sharding = try .init("mix_mesh", &physical, logical, strategy),
        .shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(.{ .batch = .open, .model = .replicated }),
        .expected_sdy = "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 1, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 2, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 3, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
        },
    });
}

test "sharding: full 3D cluster sharding" {
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

    try runner.run(.{
        .sharding = try .init("3d_mesh", &physical, logical, strategy),
        // Note: .context and .model are swapped compared to .link_x and .link_z
        .shape = Shape.init(.{ .batch = 4, .context = 4, .model = 4 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model, .context = .context }),
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

test "sharding: num partitions for logical axis" {
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

    const sharding: Sharding.Data = try .init("axis_parts_mesh", &physical, logical, strategy);

    try std.testing.expectEqual(4, sharding.numPartitionsForLogicalAxis(.model));
    try std.testing.expectEqual(1, sharding.numPartitionsForLogicalAxis(.batch));
}
