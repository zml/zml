const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Memory = @import("platform.zig").Memory;
const Platform = @import("platform.zig").Platform;
const PlatformDevice = @import("platform.zig").Device;
const Shape = @import("shape.zig").Shape;
const Slice = @import("slice.zig").Slice;
const Target = @import("platform.zig").Target;

const log = std.log.scoped(.@"zml/sharding");

// todo: check for disagregated prefill (compile on device #1)
// todo: default compilation opts 1 device, no logical mesh, no sharding, etc.
// todo: migrate tests

pub const Partitioning = struct {
    pub const Partitioner = union(enum) {
        shardy,
        gspmd,

        pub fn fromTarget(target: Target) Partitioner {
            return switch (target) {
                .cpu, .cuda, .rocm, .tpu => .shardy,
                .neuron => .gspmd,
            };
        }
    };

    partitioner: Partitioner,
    shardings: []const Sharding,

    pub fn init(partitioner: Partitioner, shardings: []const Sharding) !Partitioning {
        stdx.debug.assert(shardings.len >= 1, "Waiting at leat 1 sharding strategy to be implemented", .{});
        var plan: Partitioning = .{ .partitioner = partitioner, .shardings = shardings };

        const first = plan.primarySharding();
        const partitions = first.numPartitions();
        const replicas = first.numReplicas();

        for (shardings[1..]) |s| {
            if (s.numPartitions() != partitions or s.numReplicas() != replicas) {
                // todo: deviceAssignments should also be checked for consistency here, but for simplicity we just check the cardinality numbers
                return error.InconsistentShardingCardinality;
            }
        }

        return plan;
    }

    pub fn numPartitions(self: Partitioning) i32 {
        const primary = self.primarySharding();
        return primary.numPartitions();
    }

    pub fn numReplicas(self: Partitioning) i32 {
        const primary = self.primarySharding();
        return primary.numReplicas();
    }

    pub fn numDevices(self: Partitioning) i32 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn deviceAssignment(self: Partitioning, allocator: std.mem.Allocator) ![]usize {
        return switch (self.partitioner) {
            .shardy, .gspmd => blk: {
                const sharding = self.primarySharding();
                break :blk sharding.deviceAssignment(allocator);
            },
        };
    }

    pub fn tensorShardingAttr(self: Partitioning, allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) !?struct { name: []const u8, attr: []const u8 } {
        return switch (self.partitioner) {
            .shardy => if (try sharding.sdyShardingAttrForShape(allocator, shape)) |attr| .{
                .name = "sdy.sharding",
                .attr = attr,
            } else null,
            .gspmd => if (try sharding.gspmdShardingAttrForShape(allocator, shape)) |attr| .{
                .name = "mhlo.sharding",
                .attr = attr,
            } else null,
        };
    }

    pub fn tensorShardingAttr2(_: Partitioning, partitioner: Partitioner, allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) !?struct { name: []const u8, attr: []const u8 } {
        return switch (partitioner) {
            .shardy => if (try sharding.sdyShardingAttrForShape(allocator, shape)) |attr| .{
                .name = "sdy.sharding",
                .attr = attr,
            } else null,
            .gspmd => if (try sharding.gspmdShardingAttrForShape(allocator, shape)) |attr| .{
                .name = "mhlo.sharding",
                .attr = attr,
            } else null,
        };
    }

    pub fn selectSharding(self: Partitioning, shape: Shape) !Sharding {
        for (self.shardings) |sharding| {
            if (shardingCoversShape(sharding, shape)) return sharding;
        }

        return error.NoSuitableSharding;
    }

    fn primarySharding(self: Partitioning) Sharding {
        return self.shardings[0];
    }

    fn shardingCoversShape(sharding: Sharding, shape: Shape) bool {
        for (0..shape.rank()) |ax| {
            switch (shape.partition(ax)) {
                .axis => |tag| if (sharding.binding(tag) == null) return false,
                else => {},
            }
        }

        return true;
    }
};

/// Device is the leaf representation in a PhysicalMesh.
/// It carries identity, optional coordinates, and PJRT handle.
pub const Device = struct {
    /// Unique device identifier in the mesh
    id: usize,

    /// Coordinates in the physical mesh
    coords: stdx.BoundedArray(usize, Shape.MAX_RANK) = .{},

    /// Compute capacity local to this device (cores, SMs, etc.).
    compute_units: usize,

    /// Placeholder
    pjrt_device: ?*const pjrt.Device = null,

    pub fn coordsSlice(self: *const Device) ?[]const usize {
        if (self.coords.len == 0) return null;
        return self.coords.constSlice();
    }

    pub fn format(self: Device, writer: *std.Io.Writer) !void {
        try writer.print(
            "Device(id={d} compute_units={d} coords={any})",
            .{ self.id, self.compute_units, self.coords.constSlice() },
        );
    }
};

pub const Devices = stdx.BoundedArray(Device, Platform.MAX_NUM_DEVICES);

/// Physical axis tags represent hardware interconnect tiers.
pub const PhysicalAxisTag = enum {
    link, // 1D Interconnect (Inf1, Inf2 Islands)
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
        owned_children: bool,
    },
    /// A leaf represents the actual hardware device
    leaf: Device,

    pub fn axis(tag: PhysicalAxisTag, geometry_: AxisGeometry, children: []const PhysicalNode) PhysicalNode {
        return .{
            .branch = .{
                .tag = tag,
                .geometry = geometry_,
                .children = @constCast(children),
                .owned_children = false,
            },
        };
    }

    pub fn device(device_: PlatformDevice) PhysicalNode {
        return .{
            .leaf = .{
                .id = @intCast(device_.id()),
                .compute_units = 1,
                .coords = .{},
                .pjrt_device = device_.pjrt_device,
            },
        };
    }

    pub fn deinit(self: PhysicalNode, allocator: std.mem.Allocator) void {
        switch (self) {
            .branch => |b| {
                for (b.children) |child| child.deinit(allocator);
                allocator.free(b.children);
            },
            .leaf => {},
        }
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
        pub const Order = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK);
        pub const DepthByTag = std.EnumArray(PhysicalAxisTag, ?u8);

        /// Order of axes (DFS, first-child descent).
        order: Order,
        /// Depth (index) of each axis tag in the order.
        depth_by_tag: DepthByTag,

        /// Build axis order and depth mapping from a topology tree.
        pub fn init(root: PhysicalNode) !AxisTraversal {
            var order: Order = try .init(0);
            axisOrderNode(root, &order);

            var depth_by_tag: DepthByTag = .initFill(null);
            for (order.constSlice(), 0..) |t, i| {
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

            for (self.order.constSlice(), 0..) |tag, i| {
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

    allocator: std.mem.Allocator,
    target: Target,
    root: PhysicalNode,
    axis_traversal: AxisTraversal,

    /// Build a PhysicalMesh from a tree:
    /// - clone the tree (owning children)
    /// - validate geometry invariants
    /// - assign coordinates to leaves
    /// - compute axis traversal
    pub fn fromTree(allocator: std.mem.Allocator, target: Target, root: PhysicalNode) !PhysicalMesh {
        var cloned = try cloneNode(allocator, root);
        errdefer cloned.deinit(allocator);

        try validateGeometry(cloned);

        var path = [_]usize{0} ** Shape.MAX_RANK;
        try assignCoords(&cloned, &path, 0);

        return .{
            .allocator = allocator,
            .target = target,
            .root = cloned,
            .axis_traversal = try .init(cloned),
        };
    }

    fn cloneNode(allocator: std.mem.Allocator, node: PhysicalNode) !PhysicalNode {
        return switch (node) {
            .leaf => |d| .{
                .leaf = .{
                    .id = d.id,
                    .compute_units = d.compute_units,
                    .coords = d.coords,
                    .pjrt_device = d.pjrt_device,
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
                        .owned_children = true,
                    },
                };
            },
        };
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
    fn assignCoords(node: *PhysicalNode, path: *[Shape.MAX_RANK]usize, depth: usize) !void {
        switch (node.*) {
            .leaf => |*d| {
                var coords = stdx.BoundedArray(usize, Shape.MAX_RANK).init(0) catch unreachable;
                for (path[0..depth]) |c| coords.appendAssumeCapacity(c);
                d.coords = coords;
            },
            .branch => |*b| {
                for (b.children, 0..) |*child, i| {
                    path[depth] = i;
                    try assignCoords(child, path, depth + 1);
                }
            },
        }
    }

    pub fn deinit(self: *PhysicalMesh) void {
        self.root.deinit(self.allocator);
    }

    pub fn countDevices(self: PhysicalMesh) usize {
        return self.root.countDevices();
    }

    pub fn devices(self: PhysicalMesh, allocator: std.mem.Allocator) ![]Device {
        var list: Devices = .init(0) catch unreachable;
        try self.devicesInto(&list);

        const out = try allocator.alloc(Device, list.len);
        @memcpy(out, list.constSlice());
        return out;
    }

    pub fn devicesInto(self: PhysicalMesh, out: *Devices) !void {
        try devicesNodeInto(self.root, out);
    }

    fn devicesNodeInto(node: PhysicalNode, out: *Devices) !void {
        switch (node) {
            .leaf => |d| out.appendAssumeCapacity(d),
            .branch => |b| {
                for (b.children) |child| {
                    try devicesNodeInto(child, out);
                }
            },
        }
    }

    /// Canonical DFS linearization for coordinate consensus.
    ///
    /// sizes: S0..Sk, coords: C0..Ck
    /// linear = (((C0 * S1) + C1) * S2 + C2) ...
    pub fn linearIndexFromCoords(self: PhysicalMesh, coords: []const usize) usize {
        const order = self.axisOrder();
        var idx: usize = 0;
        for (order.constSlice()) |tag| {
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
            .tpu => &.{ .link_x, .link_y, .link_z, .bus },
            .neuron => &.{ .link_x, .link_y, .link_z, .link, .bus },
            .cuda, .rocm => &.{ .link, .bus },
            .cpu => &.{.bus},
        };
    }

    pub fn isShardable(self: PhysicalMesh, tag: PhysicalAxisTag) bool {
        for (self.shardableAxes()) |t| {
            if (t == tag) return true;
        }

        return false;
    }

    pub fn format(self: *const PhysicalMesh, writer: *std.Io.Writer) !void {
        try writer.print("\nPhysicalMesh(platform={s} num_devices={d})\n", .{ @tagName(self.target), self.countDevices() });
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

    pub fn auto(allocator: std.mem.Allocator, platform: *const Platform) !PhysicalMesh {
        var root = try switch (platform.target) {
            .cpu => cpu(allocator, platform),
            .cuda, .rocm => gpu(allocator, platform),
            .tpu => tpu(allocator, platform),
            .neuron => neuron(allocator, platform),
        };
        defer root.deinit(allocator);

        return try fromTree(allocator, platform.target, root);
    }

    pub fn cpu(allocator: std.mem.Allocator, platform: *const Platform) !Tree {
        const platform_devices = platform.devices;
        const nodes = try allocator.alloc(PhysicalNode, platform_devices.len);

        for (nodes, platform_devices) |*n, d| n.* = .device(d);

        return .{
            .branch = .{
                .tag = .bus,
                .geometry = .tree,
                .children = nodes,
                .owned_children = true,
            },
        };
    }

    pub fn gpu(allocator: std.mem.Allocator, platform: *const Platform) !Tree {
        const platform_devices = platform.devices;

        // todo: this is simplified I treat all GPUs as one P2P group for this example
        const nodes = try allocator.alloc(PhysicalNode, platform_devices.len);

        for (nodes, platform_devices) |*n, d| n.* = .device(d);

        return .{
            .branch = .{
                .tag = .link,
                .geometry = .point_to_point,
                .children = nodes,
                .owned_children = true,
            },
        };
    }

    pub fn tpu(allocator: std.mem.Allocator, platform: *const Platform) !Tree {
        const platform_devices = platform.devices;

        // Example: TPU v3-8 is 2x2x2 (8 devices)
        var z_branches = try allocator.alloc(PhysicalNode, 4);

        for (z_branches, 0..) |*z_branch, i| {
            const z_leaves = try allocator.alloc(PhysicalNode, 2);

            z_leaves[0] = .device(platform_devices[i * 2]);
            z_leaves[1] = .device(platform_devices[i * 2 + 1]);

            z_branch.* = .{
                .branch = .{
                    .tag = .link_z,
                    .geometry = .{ .mesh = .torus },
                    .children = z_leaves,
                    .owned_children = true,
                },
            };
        }

        var y_branches = try allocator.alloc(PhysicalNode, 2);

        y_branches[0] = .{
            .branch = .{
                .tag = .link_y,
                .geometry = .{ .mesh = .torus },
                .children = z_branches[0..2],
                .owned_children = true,
            },
        };

        y_branches[1] = .{
            .branch = .{
                .tag = .link_y,
                .geometry = .{ .mesh = .torus },
                .children = z_branches[2..4],
                .owned_children = true,
            },
        };

        return .{
            .branch = .{
                .tag = .link_x,
                .geometry = .{ .mesh = .torus },
                .children = y_branches,
                .owned_children = true,
            },
        };
    }

    pub fn neuron(allocator: std.mem.Allocator, platform: *const Platform) !PhysicalNode {
        const platform_devices = platform.devices;

        // AWS Inf2.48xlarge: 2 islands of 12 chips (24 devices)
        const islands = try allocator.alloc(PhysicalNode, 2);

        for (islands, 0..) |*island, i| {
            const start = i * 12;
            const ring_nodes = try allocator.alloc(PhysicalNode, 12);

            for (ring_nodes, 0..) |*rn, j| rn.* = .device(platform_devices[start + j]);

            island.* = .{
                .branch = .{
                    .tag = .link,
                    .geometry = .{ .ring = .closed_ring },
                    .children = ring_nodes,
                    .owned_children = true,
                },
            };
        }

        return .{
            .branch = .{
                .tag = .bus,
                .geometry = .isolated,
                .children = islands,
                .owned_children = true,
            },
        };
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
    pub const Axes = stdx.BoundedArray(Shape.Tag, Shape.MAX_RANK);
    pub const Intents = stdx.BoundedArray(LogicalAxisIntent, Shape.MAX_RANK);

    name: []const u8,
    axes: Axes,
    intents: Intents,

    pub fn init(name: []const u8, axes_: anytype) !LogicalMesh {
        const T = @TypeOf(axes_);

        var axes: Axes = try .init(0);
        var intents: Intents = try .init(0);

        inline for (std.meta.fields(T)) |field| {
            const value = @field(axes_, field.name);
            axes.appendAssumeCapacity(Shape.toTag(field));
            intents.appendAssumeCapacity(intentFromValue(value));
        }

        if (axes.len == 0) {
            stdx.debug.panic("LogicalMesh must have at least one axis defined", .{});
        }

        return .{
            .name = name,
            .axes = axes,
            .intents = intents,
        };
    }

    pub fn intent(self: LogicalMesh, tag: Shape.Tag) ?LogicalAxisIntent {
        const target = std.mem.span(tag);
        for (self.axes.constSlice(), self.intents.constSlice()) |t, i| {
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
        try writer.print("LogicalMesh(name={s}", .{self.name});
        for (self.axes.constSlice(), self.intents.constSlice()) |axis, int| {
            try writer.print(" {s}={s}", .{ axis, @tagName(int) });
        }
        try writer.print(")", .{});
    }
};

pub const Sharding = struct {
    pub const AxisList = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK);
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: AxisList,
    };
    pub const Bindings = stdx.BoundedArray(Binding, Shape.MAX_RANK);

    pub const Fold = struct {
        target: PhysicalAxisTag,
        sources: AxisList,
    };
    pub const Folds = stdx.BoundedArray(Fold, Shape.MAX_RANK);

    pub const Axis = struct {
        tag: PhysicalAxisTag,
        size: i64,
        geometry: ?AxisGeometry,
        folded: stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK),

        pub fn contains(self: *const Axis, tag: PhysicalAxisTag) bool {
            if (self.tag == tag) return true;
            for (self.folded.constSlice()) |t| if (t == tag) return true;
            return false;
        }
    };

    pub const PhysicalView = struct {
        axes: stdx.BoundedArray(Axis, Shape.MAX_RANK),
        total_devices: i64,

        pub fn axisCoordFromLinearShard(self: *const PhysicalView, axis_index: usize, linear_idx: usize) usize {
            const stride = self.axisStrideForLinear(axis_index);
            const axis_size: usize = @intCast(self.axes.constSlice()[axis_index].size);
            return (linear_idx / stride) % axis_size;
        }

        pub fn axisStrideForLinear(self: *const PhysicalView, axis_index: usize) usize {
            var stride: usize = 1;
            var j = axis_index + 1;
            while (j < self.axes.len) : (j += 1) {
                stride *= @intCast(self.axes.constSlice()[j].size);
            }
            return stride;
        }
    };

    logical: LogicalMesh,
    physical: PhysicalMesh,

    /// Compact binding table: logical axis -> physical axes
    bindings: Bindings,

    /// Explicit folding rules: kept axis -> ordered source axes.
    folds: Folds,
    folds_consumed: std.EnumSet(PhysicalAxisTag),

    pub fn binding(self: *const Sharding, tag: Shape.Tag) ?[]const PhysicalAxisTag {
        const target = std.mem.span(tag);
        for (self.bindings.constSlice()) |*b| {
            if (std.mem.eql(u8, std.mem.span(b.logical), target)) return b.physical.constSlice();
        }
        return null;
    }

    pub fn initFromStrategy(
        logical: LogicalMesh,
        physical: PhysicalMesh,
        strategy: Strategy,
    ) !Sharding {
        const axis_order = physical.axisOrder().constSlice();
        if (axis_order.len == 0) return error.InvalidPhysicalMesh;

        var bindings: Bindings = try .init(0);
        for (strategy.bindings.constSlice()) |bind| {
            var list: AxisList = try .init(0);
            for (bind.physical.constSlice()) |p_tag| {
                list.appendAssumeCapacity(p_tag);
            }
            bindings.appendAssumeCapacity(.{ .logical = bind.logical, .physical = list });
        }

        var folds: Folds = try .init(0);
        var folds_consumed = std.EnumSet(PhysicalAxisTag).initEmpty();
        for (strategy.folding.constSlice()) |entry| {
            for (entry.sources.constSlice()) |src| {
                if (!physical.hasAxis(src)) return error.InvalidPhysicalAxis;
                folds_consumed.insert(src);
            }
            folds.appendAssumeCapacity(.{ .target = entry.target, .sources = entry.sources });
        }

        return .{
            .logical = logical,
            .physical = physical,
            .bindings = bindings,
            .folds = folds,
            .folds_consumed = folds_consumed,
        };
    }

    fn foldSources(self: *const Sharding, tag: PhysicalAxisTag) ?[]const PhysicalAxisTag {
        for (self.folds.constSlice()) |*f| {
            if (f.target == tag) return f.sources.constSlice();
        }
        return null;
    }

    pub fn physicalView(self: Sharding) PhysicalView {
        var view: PhysicalView = .{
            .axes = stdx.BoundedArray(Axis, Shape.MAX_RANK).init(0) catch unreachable,
            .total_devices = 1,
        };

        const axis_order = self.physical.axisOrder().constSlice();

        for (axis_order) |tag| {
            if (!self.physical.hasAxis(tag)) continue;
            if (self.folds_consumed.contains(tag) and self.foldSources(tag) == null) continue;

            var folded = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
            var size: i64 = 1;

            if (self.foldSources(tag)) |sources| {
                for (sources) |src| {
                    folded.appendAssumeCapacity(src);
                    size *= self.physical.axis(src);
                }
            } else {
                folded.appendAssumeCapacity(tag);
                size = self.physical.axis(tag);
            }

            view.total_devices *= size;
            view.axes.appendAssumeCapacity(.{
                .tag = tag,
                .size = size,
                .geometry = self.physical.geometry(tag),
                .folded = folded,
            });
        }

        return view;
    }

    pub fn name(self: Sharding) []const u8 {
        return self.logical.name;
    }

    pub fn logicalIndexFromCoords(self: Sharding, coords: []const usize) usize {
        return self.physical.linearIndexFromCoords(coords);
    }

    pub fn numPartitions(self: Sharding) i32 {
        return @intCast(self.physicalView().total_devices);
    }

    pub fn numReplicas(_: Sharding) i32 {
        return 1;
    }

    pub fn numDevices(self: Sharding) i32 {
        return self.numPartitions() * self.numReplicas();
    }

    pub fn sdyMeshAttr(self: Sharding, allocator: std.mem.Allocator) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        const view = self.physicalView();
        try out.writer.writeAll("#sdy.mesh<[");
        for (view.axes.constSlice(), 0..) |p, i| {
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
    fn getDimMapping(self: *const Sharding, shape: Shape) DimMapping {
        const view = self.physicalView();
        var axes_per_dim = stdx.BoundedArray(stdx.BoundedArray(usize, Shape.MAX_RANK), Shape.MAX_RANK).init(0) catch unreachable;
        var used_mask = [_]bool{false} ** Shape.MAX_RANK;
        var globally_used = std.EnumSet(PhysicalAxisTag).initEmpty();

        for (0..shape.rank()) |ax| {
            var dim_axes = stdx.BoundedArray(usize, Shape.MAX_RANK).init(0) catch unreachable;
            const spec = shape.partition(ax);

            if (spec == .axis) {
                if (self.binding(spec.axis)) |binding_| {
                    for (binding_) |p_tag| {
                        for (view.axes.constSlice(), 0..) |v_ax, i| {
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

        var replicated_axes = stdx.BoundedArray(usize, Shape.MAX_RANK).init(0) catch unreachable;
        for (0..view.axes.len) |i| {
            if (!used_mask[i]) replicated_axes.appendAssumeCapacity(i);
        }

        return .{
            .axes_per_dim = axes_per_dim,
            .replicated_axes = replicated_axes,
            .view = view,
        };
    }

    pub fn sdyShardingAttrForShape(self: Sharding, allocator: std.mem.Allocator, shape: Shape) !?[]const u8 {
        var any_explicit = false;
        for (0..shape.rank()) |ax| {
            if (shape.partition(ax) != .unknown) {
                any_explicit = true;
                break;
            }
        }
        if (!any_explicit) return null;

        const mapping = self.getDimMapping(shape);
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        try out.writer.print("#sdy.sharding<@{s}, [", .{self.name()});
        for (0..shape.rank()) |ax| {
            if (ax > 0) try out.writer.writeAll(", ");
            const spec = shape.partition(ax);

            switch (spec) {
                .axis => |logical_tag| {
                    if (self.binding(logical_tag)) |_| {
                        const dim_phys_indices = mapping.axes_per_dim.get(ax);
                        if (dim_phys_indices.len == 0) {
                            try out.writer.writeAll("{}");
                        } else {
                            try out.writer.writeAll("{");
                            for (dim_phys_indices.constSlice(), 0..) |p_idx, i| {
                                if (i > 0) try out.writer.writeAll(", ");
                                try out.writer.print("\"{s}\"", .{@tagName(mapping.view.axes.get(p_idx).tag)});
                            }
                            try out.writer.writeAll("}");
                        }
                    } else {
                        try out.writer.writeAll("{?}");
                    }
                },
                .replicated => try out.writer.writeAll("{}"),
                .open, .unknown => try out.writer.writeAll("{?}"),
            }
        }
        try out.writer.writeAll("]");

        if (mapping.replicated_axes.len > 0) {
            try out.writer.writeAll(", replicated={");
            for (mapping.replicated_axes.constSlice(), 0..) |p_idx, i| {
                if (i > 0) try out.writer.writeAll(", ");
                try out.writer.print("\"{s}\"", .{@tagName(mapping.view.axes.get(p_idx).tag)});
            }
            try out.writer.writeAll("}");
        }
        try out.writer.writeAll(">");
        return try out.toOwnedSlice();
    }

    pub fn gspmdShardingAttrForShape(self: Sharding, allocator: std.mem.Allocator, shape: Shape) !?[]const u8 {
        var has_sharding = false;
        for (0..shape.rank()) |ax| {
            if (shape.partition(ax) == .axis) {
                has_sharding = true;
                break;
            }
        }
        // If the tensor is fully replicated, GSPMD usually prefers no attribute
        // or the explicit "{replicated}" string.
        if (!has_sharding) return null;

        const mapping = self.getDimMapping(shape);
        var tile_shape = stdx.BoundedArray(i64, Shape.MAX_RANK + 1).init(0) catch unreachable;
        var permutation = stdx.BoundedArray(usize, Shape.MAX_RANK).init(0) catch unreachable;

        //  Calculate tile sizes per tensor dimension
        for (mapping.axes_per_dim.constSlice()) |dim_axes| {
            var combined: i64 = 1;
            for (dim_axes.constSlice()) |idx| {
                combined *= mapping.view.axes.get(idx).size;
                permutation.appendAssumeCapacity(idx);
            }
            tile_shape.appendAssumeCapacity(combined);
        }

        // Add replication dimension if physical axes are left over
        const has_replication = mapping.replicated_axes.len > 0;
        if (has_replication) {
            var repl_size: i64 = 1;
            for (mapping.replicated_axes.constSlice()) |idx| {
                repl_size *= mapping.view.axes.get(idx).size;
                permutation.appendAssumeCapacity(idx);
            }
            tile_shape.appendAssumeCapacity(repl_size);
        }

        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        try out.writer.writeByte('"');

        try out.writer.writeAll("{devices=[");
        for (tile_shape.constSlice(), 0..) |s, i| {
            if (i > 0) try out.writer.writeAll(",");
            try out.writer.print("{d}", .{s});
        }
        try out.writer.writeAll("]<=[");
        for (mapping.view.axes.constSlice(), 0..) |ax, i| {
            if (i > 0) try out.writer.writeAll(",");
            try out.writer.print("{d}", .{ax.size});
        }
        try out.writer.writeAll("]");

        // Permutation T(...)
        // Only emit T if it's not the identity permutation to keep the string clean.
        var is_identity = (permutation.len == mapping.view.axes.len);
        if (is_identity) {
            for (permutation.constSlice(), 0..) |p, i| {
                if (p != i) {
                    is_identity = false;
                    break;
                }
            }
        }

        if (!is_identity and permutation.len > 0) {
            try out.writer.writeAll("T(");
            for (permutation.constSlice(), 0..) |p, i| {
                if (i > 0) try out.writer.writeAll(",");
                try out.writer.print("{d}", .{p});
            }
            try out.writer.writeAll(")");
        }

        if (has_replication) {
            try out.writer.writeAll(" last_tile_dim_replicate");
        }

        try out.writer.writeAll("}");
        try out.writer.writeByte('"');

        return try out.toOwnedSlice();
    }

    pub fn deviceAssignment(self: Sharding, allocator: std.mem.Allocator) ![]usize {
        const view = self.physicalView();
        const count: usize = @intCast(view.total_devices);

        var ids = try allocator.alloc(usize, count);
        @memset(ids, std.math.maxInt(usize));

        const ordered_devices = try self.devicesInCanonicalOrder();
        for (ordered_devices.constSlice()) |d| {
            const coords = d.coordsSlice() orelse return error.MissingDeviceCoords;
            const idx = self.physical.linearIndexFromCoords(coords);
            ids[idx] = d.id;
        }

        for (ids) |id| {
            if (id == std.math.maxInt(usize)) return error.MissingDeviceInTile;
        }

        return ids;
    }

    fn devicesInCanonicalOrder(self: Sharding) !Devices {
        var devices: Devices = try .init(0);
        try self.physical.devicesInto(&devices);

        const Order = struct {
            mesh: PhysicalMesh,
            fn lessThan(ctx: @This(), a: Device, b: Device) bool {
                const ca = a.coordsSlice() orelse unreachable;
                const cb = b.coordsSlice() orelse unreachable;
                const ia = ctx.mesh.linearIndexFromCoords(ca);
                const ib = ctx.mesh.linearIndexFromCoords(cb);
                return ia < ib;
            }
        };

        std.mem.sort(Device, devices.slice(), Order{ .mesh = self.physical }, Order.lessThan);
        return devices;
    }

    pub fn format(self: Sharding, writer: *std.Io.Writer) !void {
        try writer.print("Sharding(name={s})\n", .{self.logical.name});

        try writer.writeAll("Bindings:\n");
        for (self.logical.axes.constSlice(), self.logical.intents.constSlice()) |l_tag, l_intent| {
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
            for (view.axes.constSlice(), 0..) |axis, i| {
                if (i > 0) try writer.writeAll(" × ");
                try writer.print("{s}[{d}]", .{ @tagName(axis.tag), axis.size });
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("Geometries: ");
        if (view.axes.len == 0) {
            try writer.writeAll("(none)\n");
        } else {
            for (view.axes.constSlice(), 0..) |axis, i| {
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

pub fn replicatedSharding(physical_mesh: PhysicalMesh) !Sharding {
    const logical_mesh: LogicalMesh = try .init("replicated", .{ .x = .high_bandwidth });
    const strategy: Strategy = try .suggest(logical_mesh, physical_mesh);
    return try .initFromStrategy(logical_mesh, physical_mesh, strategy);
}

pub const Strategy = struct {
    pub const PhysicalList = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK);
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: PhysicalList,
    };
    pub const Bindings = stdx.BoundedArray(Binding, Shape.MAX_RANK);
    pub const Fold = struct {
        target: PhysicalAxisTag,
        sources: PhysicalList,
    };
    pub const Folding = stdx.BoundedArray(Fold, Shape.MAX_RANK);

    bindings: Bindings,
    folding: Folding,

    pub const init: Strategy = .{
        .bindings = .{},
        .folding = .{},
    };

    pub fn addBinding(self: *Strategy, logical: anytype, physical: PhysicalAxisTag) !void {
        const raw_tag = Shape.toTag(logical);
        const logical_tag: []const u8 = std.mem.span(raw_tag);

        for (self.bindings.slice()) |*b| {
            if (std.mem.eql(u8, std.mem.span(b.logical), logical_tag)) {
                b.physical.appendAssumeCapacity(physical);
                return;
            }
        }

        var list: PhysicalList = try .init(0);
        list.appendAssumeCapacity(physical);
        self.bindings.appendAssumeCapacity(.{ .logical = raw_tag, .physical = list });
    }

    /// Explicitly fold axes: `target` is the kept axis, `sources` define order.
    /// If `target` is missing from `sources`, it is prepended.
    pub fn addFold(self: *Strategy, target: PhysicalAxisTag, sources: []const PhysicalAxisTag) !void {
        var list: PhysicalList = try .init(0);

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
    pub fn suggest(logical: LogicalMesh, physical: PhysicalMesh) !Strategy {
        var strategy: Strategy = .init;

        const base_order = physical.shardableAxes();
        var available = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        for (base_order) |tag| {
            if (physical.hasAxis(tag)) available.appendAssumeCapacity(tag);
        }

        if (available.len == 0) {
            for (physical.axisOrder().constSlice()) |tag| available.appendAssumeCapacity(tag);
        }

        const avail = available.constSlice();
        var counters = std.EnumArray(LogicalAxisIntent, usize).initFill(0);

        inline for (std.meta.tags(LogicalAxisIntent)) |target_intent| {
            for (logical.axes.constSlice(), logical.intents.constSlice()) |l_tag, l_intent| {
                if (l_intent != target_intent) continue;

                const i = counters.get(target_intent);
                const p_tag = switch (target_intent) {
                    .high_bandwidth => avail[i % avail.len],
                    .balanced => avail[(avail.len / 2 + i) % avail.len],
                    .low_bandwidth => avail[avail.len - 1 - (i % avail.len)],
                };

                try strategy.addBinding(l_tag, p_tag);
                counters.set(target_intent, i + 1);
            }
        }

        return strategy;
    }
};

pub const Placement = struct {
    pub const Slice1d = struct {
        axis: u8,
        start: i64,
        size: i64,
    };

    pub const Shard = struct {
        device_id: usize,
        device_coords: stdx.BoundedArray(usize, Shape.MAX_RANK),
        shape: Shape,
        global_shape: Shape,
        slices: stdx.BoundedArray(Slice1d, Shape.MAX_RANK),

        pub fn device(self: *const Shard, platform: *const Platform) PlatformDevice {
            return blk: {
                for (platform.devices) |d| {
                    if (d.id() == self.device_id) break :blk d;
                }
                unreachable;
            };
        }

        pub fn memory(self: *const Shard, platform: *const Platform, kind: Memory.Kind) *const Memory {
            return self.device(platform).memory(kind);
        }

        pub fn shardSlice(self: *const Shard, slice: Slice) Slice {
            stdx.debug.assert(
                self.global_shape.eql(slice.shape),
                "Placement global shape {f} doesn't match slice shape {f}",
                .{ self.global_shape, slice.shape },
            );

            var res = slice;
            for (self.slices.constSlice()) |s| {
                res = res.subSlice(s.axis, s.start, s.size);
            }
            return res;
        }

        pub fn format(self: Shard, writer: *std.Io.Writer) !void {
            try writer.print("device_id={d} coords={any}", .{ self.device_id, self.device_coords.constSlice() });

            try writer.writeAll(" slices=[");
            for (self.slices.constSlice(), 0..) |s, i| {
                if (i > 0) try writer.writeAll(", ");
                const axis_label = self.global_shape.debugTag(s.axis);
                try writer.print("{s}:[{d}:{d}]", .{ axis_label, s.start, s.start + s.size });
            }
            try writer.writeAll("]");
        }
    };
    pub const Shards = stdx.BoundedArray(Shard, Platform.MAX_NUM_DEVICES);

    const AxisSplit = struct {
        product: i64,
        counts: stdx.BoundedArray(i64, Shape.MAX_RANK),
        indices: stdx.BoundedArray(i64, Shape.MAX_RANK),

        pub fn init() !AxisSplit {
            return .{
                .product = 1,
                .counts = try .init(0),
                .indices = try .init(0),
            };
        }

        pub fn linearIndex(self: AxisSplit) i64 {
            var linear: i64 = 0;
            for (self.counts.constSlice(), self.indices.constSlice()) |c, iidx| {
                linear = linear * c + iidx;
            }
            return linear;
        }
    };

    sharding: Sharding,
    shape: Shape,
    shards: Shards,

    pub fn init(
        sharding: Sharding,
        shape: Shape,
    ) !Placement {
        const ordered_devices = try sharding.devicesInCanonicalOrder();

        var shards: Shards = try .init(0);

        for (ordered_devices.constSlice()) |d| {
            if (d.coords.len == 0) return error.MissingDeviceCoords;

            shards.appendAssumeCapacity(.{
                .device_id = d.id,
                .device_coords = d.coords,
                .shape = undefined, // Calculated below
                .global_shape = shape,
                .slices = try .init(0),
            });
        }

        var self: Placement = .{
            .sharding = sharding,
            .shape = shape,
            .shards = shards,
        };

        for (self.shards.slice()) |*s| {
            var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();

            for (0..shape.rank()) |ax| {
                const axis_index: u8 = @intCast(ax);
                const slice = try self.sliceForDevice(s.device_coords.constSlice(), &used_axes, axis_index);
                s.slices.appendAssumeCapacity(slice);
            }

            s.shape = blk: {
                var sh = shape;
                for (s.slices.constSlice()) |slice| {
                    sh = sh.set(slice.axis, slice.size);
                }
                break :blk sh;
            };
        }

        return self;
    }

    fn sliceForDevice(
        self: *Placement,
        device_coords: []const usize,
        used_axes: *std.EnumSet(PhysicalAxisTag),
        axis_index: u8,
    ) !Slice1d {
        const dim = self.shape.dim(axis_index);
        const spec = self.shape.partition(axis_index);

        switch (spec) {
            .axis => |logical_tag| {
                const binding = self.sharding.binding(logical_tag) orelse return error.MissingLogicalBinding;

                // Calculate the split based on the physical coordinates of the current device
                const plan = try self.calculateSplit(dim, binding, device_coords, used_axes);

                const size = @divExact(dim, plan.product);
                const start = plan.linearIndex() * size;

                return .{
                    .axis = axis_index,
                    .start = start,
                    .size = size,
                };
            },
            else => return .{ .axis = axis_index, .start = 0, .size = dim },
        }
    }

    fn calculateSplit(
        self: *Placement,
        dim: i64,
        binding: []const PhysicalAxisTag,
        device_coords: []const usize,
        used_axes: *std.EnumSet(PhysicalAxisTag),
    ) !AxisSplit {
        var plan: AxisSplit = try .init();

        for (binding) |p_tag| {
            // Expand the physical tag: check if it's a target for folded source axes
            if (self.sharding.foldSources(p_tag)) |sources| {
                for (sources) |src| {
                    if (used_axes.contains(src)) continue;
                    try self.addPhysicalToSplit(&plan, src, device_coords);
                    used_axes.insert(src);
                }
            } else {
                // Standard case: directly bound, not part of an explicit fold
                if (used_axes.contains(p_tag)) continue;
                try self.addPhysicalToSplit(&plan, p_tag, device_coords);
                used_axes.insert(p_tag);
            }
        }

        // Validate divisibility
        if (plan.product > 0 and @rem(dim, plan.product) != 0) {
            return error.IncompatibleSharding;
        }
        return plan;
    }

    /// Extract coordinate data from the physical mesh for a specific axis.
    fn addPhysicalToSplit(self: *Placement, plan: *AxisSplit, tag: PhysicalAxisTag, device_coords: []const usize) !void {
        const info = self.sharding.physical.axisInfo(tag) orelse return;
        const depth = self.sharding.physical.axis_traversal.depth(tag) orelse return;
        const coord = device_coords[depth];

        plan.counts.appendAssumeCapacity(@intCast(info.size));
        plan.indices.appendAssumeCapacity(@intCast(coord));
        plan.product *= info.size;
    }

    pub fn format(self: Placement, writer: *std.Io.Writer) !void {
        try writer.print("Placement(shape={f} shards={d})\n", .{ self.shape, self.shards.len });

        try writer.writeAll("Axis partitioning summary:\n");
        for (0..self.shape.rank()) |ax| {
            const dim = self.shape.dim(ax);
            const spec = self.shape.partition(ax);
            const axis_label = self.shape.debugTag(ax);

            var shard_size: i64 = dim;
            if (self.shards.len > 0) {
                for (self.shards.constSlice()[0].slices.constSlice()) |s| {
                    if (s.axis == ax) {
                        shard_size = s.size;
                        break;
                    }
                }
            }

            const shards_count: i64 = if (shard_size > 0 and @rem(dim, shard_size) == 0)
                @divExact(dim, shard_size)
            else
                0;

            try writer.print(
                "└─ {s}: dim={d}, spec={s}, shard_size={d}, shards={d}\n",
                .{ axis_label, dim, @tagName(spec), shard_size, shards_count },
            );
        }

        try writer.writeAll("Per-shard placement:\n");
        for (self.shards.constSlice(), 0..) |shard, i| {
            try writer.print("└─ Shard[{d}] {f}\n", .{ i, shard });
        }
    }
};

const ShardingTest = struct {
    pub const ExpectedShard = struct {
        device_id: usize,
        slices: []const [2]i64,
    };

    pub const Scenario = struct {
        physical: PhysicalMesh,
        logical: LogicalMesh,
        strategy: Strategy,
        shape: Shape,

        expected_sdy: ?[]const u8 = null,
        expected_shards: []const ExpectedShard = &.{},

        expect_error: ?anyerror = null,
        expect_sdy_null: bool = false,

        pub fn deinit(self: *Scenario) void {
            self.physical.deinit();
        }
    };

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ShardingTest {
        return .{ .allocator = allocator };
    }

    /// Creates a 1D-3D PhysicalMesh from simple dimensions.
    pub fn physical(self: ShardingTest, dims: anytype, geometry: AxisGeometry) !PhysicalMesh {
        const info = @typeInfo(@TypeOf(dims)).@"struct";
        var tags = try stdx.BoundedArray(PhysicalAxisTag, 3).init(0);
        var sizes = try stdx.BoundedArray(usize, 3).init(0);

        const tag_names = [_]PhysicalAxisTag{ .link_x, .link_y, .link_z };
        inline for (info.fields, 0..) |field, i| {
            tags.appendAssumeCapacity(tag_names[i]);
            sizes.appendAssumeCapacity(@intCast(@field(dims, field.name)));
        }

        var next_id: usize = 0;
        var root = try self.buildNode(tags.constSlice(), sizes.constSlice(), 0, &next_id, geometry);
        defer root.deinit(self.allocator);
        return try .fromTree(self.allocator, .tpu, root);
    }

    fn buildNode(self: ShardingTest, tags: []const PhysicalAxisTag, sizes: []const usize, depth: usize, next_id: *usize, geometry: AxisGeometry) !PhysicalNode {
        if (depth == tags.len) {
            const id = next_id.*;
            next_id.* += 1;
            return .{ .leaf = .{ .id = id, .compute_units = 1 } };
        }
        const count = sizes[depth];
        const children = try self.allocator.alloc(PhysicalNode, count);
        for (children) |*child| {
            child.* = try self.buildNode(tags, sizes, depth + 1, next_id, geometry);
        }
        return .{ .branch = .{ .tag = tags[depth], .geometry = geometry, .children = children, .owned_children = true } };
    }

    pub fn run(self: ShardingTest, s: Scenario) !void {
        const sharding = try Sharding.initFromStrategy(s.logical, s.physical, s.strategy);

        // Verify MLIR String
        if (s.expected_sdy) |expected_attr| {
            const actual_attr = try sharding.sdyShardingAttrForShape(self.allocator, s.shape) orelse return error.ExpectedAnnotationButGotNull;
            defer self.allocator.free(actual_attr);
            try std.testing.expectEqualStrings(expected_attr, actual_attr);
        } else if (s.expect_sdy_null) {
            const actual_attr = try sharding.sdyShardingAttrForShape(self.allocator, s.shape);
            if (actual_attr) |a| {
                self.allocator.free(a);
                return error.ExpectedNullAnnotation;
            }
        }

        // Verify Placement logic / Error
        if (s.expect_error) |err| {
            try std.testing.expectError(err, Placement.init(sharding, s.shape));
            return;
        }

        // Verify Shard Slices (Math)
        if (s.expected_shards.len > 0) {
            const placement = try Placement.init(sharding, s.shape);
            try std.testing.expectEqual(s.expected_shards.len, placement.shards.len);
            for (s.expected_shards, 0..) |expected, i| {
                const actual = placement.shards.constSlice()[i];
                try std.testing.expectEqual(expected.device_id, actual.device_id);
                for (expected.slices, 0..) |exp_slice, dim| {
                    const act_slice = actual.slices.constSlice()[dim];
                    try std.testing.expectEqual(exp_slice[0], act_slice.start);
                    try std.testing.expectEqual(exp_slice[1], act_slice.size);
                }
            }
        }
    }
};

test "sharding: unknown partitioning implies replication" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = try .init("folded_mesh", .{ .batch = .low_bandwidth });

    var strategy: Strategy = .init;
    try strategy.addBinding(.batch, .link_x);

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
        .shape = Shape.init(.{ .batch = 8 }, .f32),
        .expect_sdy_null = true,
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 1, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 2, .slices = &.{.{ 0, 8 }} },
            .{ .device_id = 3, .slices = &.{.{ 0, 8 }} },
        },
    };
    defer scenario.deinit();
    try runner.run(scenario);
}

test "sharding: suggest strategy realization" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });

    const logical: LogicalMesh = try .init("suggested_mesh", .{
        .batch = .low_bandwidth,
        .model = .high_bandwidth,
    });

    const strategy: Strategy = try .suggest(logical, physical);

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
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
    };
    defer scenario.deinit();
    try runner.run(scenario);
}

test "sharding: suggest folds logical axes with same intent" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical = try runner.physical(.{4}, .point_to_point);

    const logical: LogicalMesh = try .init("fold_mesh", .{
        .model = .high_bandwidth,
        .experts = .high_bandwidth,
    });

    const strategy: Strategy = try .suggest(logical, physical);

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
        .shape = Shape.init(.{ .model = 4, .experts = 4 }, .f32)
            .withPartitioning(.{ .model = .model, .experts = .experts }),
        .expected_sdy = "#sdy.sharding<@fold_mesh, [{\"link_x\"}, {}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 1 }, .{ 0, 4 } } },
            .{ .device_id = 1, .slices = &.{ .{ 1, 1 }, .{ 0, 4 } } },
            .{ .device_id = 2, .slices = &.{ .{ 2, 1 }, .{ 0, 4 } } },
            .{ .device_id = 3, .slices = &.{ .{ 3, 1 }, .{ 0, 4 } } },
        },
    };
    defer scenario.deinit();
    try runner.run(scenario);
}

test "sharding: multiple physical axes on one logical dimension (folding)" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = try .init("folded_mesh", .{ .model = .high_bandwidth });

    var strategy: Strategy = .init;
    try strategy.addBinding(.model, .link_x);
    try strategy.addBinding(.model, .link_y);

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
        .shape = Shape.init(.{ .model = 16 }, .f32).withPartitioning(.{ .model = .model }),
        .expected_sdy = "#sdy.sharding<@folded_mesh, [{\"link_x\", \"link_y\"}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{.{ 0, 4 }} },
            .{ .device_id = 1, .slices = &.{.{ 4, 4 }} },
            .{ .device_id = 2, .slices = &.{.{ 8, 4 }} },
            .{ .device_id = 3, .slices = &.{.{ 12, 4 }} },
        },
    };
    defer scenario.deinit();
    try runner.run(scenario);
}

test "sharding: explicit strategy folding" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = try .init("strategy_fold", .{ .model = .high_bandwidth });

    var strategy: Strategy = .init;
    try strategy.addBinding(.model, .link_x);
    try strategy.addFold(.link_x, &.{ .link_x, .link_z });

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
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
    };
    defer scenario.deinit();
    try runner.run(scenario);
}

test "sharding: open and replicated dimension mix" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = try .init("mix_mesh", .{ .batch = .low_bandwidth, .model = .high_bandwidth });

    var strategy: Strategy = .init;
    try strategy.addBinding(.batch, .link_x);
    try strategy.addBinding(.model, .link_y);

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
        .shape = Shape.init(.{ .batch = 8, .model = 8 }, .f32).withPartitioning(.{ .batch = .open, .model = .replicated }),
        .expected_sdy = "#sdy.sharding<@mix_mesh, [{?}, {}], replicated={\"link_x\", \"link_y\"}>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 1, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 2, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
            .{ .device_id = 3, .slices = &.{ .{ 0, 8 }, .{ 0, 8 } } },
        },
    };
    defer scenario.deinit();
    try runner.run(scenario);
}

test "sharding: full 3D cluster sharding" {
    const allocator = std.testing.allocator;
    const runner: ShardingTest = .init(allocator);

    const physical: PhysicalMesh = try runner.physical(.{ 2, 2, 2 }, .{ .mesh = .torus });
    const logical: LogicalMesh = try .init("3d_mesh", .{ .batch = .low_bandwidth, .model = .high_bandwidth, .context = .balanced });

    var strategy: Strategy = .init;
    try strategy.addBinding(.batch, .link_x);
    try strategy.addBinding(.model, .link_y);
    try strategy.addBinding(.context, .link_z);

    var scenario: ShardingTest.Scenario = .{
        .physical = physical,
        .logical = logical,
        .strategy = strategy,
        .shape = Shape.init(.{ .batch = 4, .model = 4, .context = 4 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model, .context = .context }),
        .expected_sdy = "#sdy.sharding<@3d_mesh, [{\"link_x\"}, {\"link_y\"}, {\"link_z\"}]>",
        .expected_shards = &.{
            .{ .device_id = 0, .slices = &.{ .{ 0, 2 }, .{ 0, 2 }, .{ 0, 2 } } },
            .{ .device_id = 1, .slices = &.{ .{ 0, 2 }, .{ 0, 2 }, .{ 2, 2 } } },
            .{ .device_id = 2, .slices = &.{ .{ 0, 2 }, .{ 2, 2 }, .{ 0, 2 } } },
            .{ .device_id = 3, .slices = &.{ .{ 0, 2 }, .{ 2, 2 }, .{ 2, 2 } } },
            .{ .device_id = 4, .slices = &.{ .{ 2, 2 }, .{ 0, 2 }, .{ 0, 2 } } },
            .{ .device_id = 5, .slices = &.{ .{ 2, 2 }, .{ 0, 2 }, .{ 2, 2 } } },
            .{ .device_id = 6, .slices = &.{ .{ 2, 2 }, .{ 2, 2 }, .{ 0, 2 } } },
            .{ .device_id = 7, .slices = &.{ .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 } } },
        },
    };
    defer scenario.deinit();
    try runner.run(scenario);
}
