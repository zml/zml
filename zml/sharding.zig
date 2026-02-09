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

pub const Partitioner = union(enum) {
    shardy,
    gspmd,

    pub fn fromTarget(target: Target) Partitioner {
        return switch (target) {
            .cpu, .cuda, .rocm, .tpu => .shardy,
            .neuron => .gspmd,
        };
    }

    pub const Plan = struct {
        pub const MeshDescriptor = struct {
            name: []const u8,
            attr: []const u8,
        };

        pub const Cardinality = struct {
            num_partitions: i32,
            num_replicas: i32,
        };

        partitioner: Partitioner,
        shardings: []const Sharding,

        pub fn init(partitioner: Partitioner, shardings: []const Sharding) !Plan {
            stdx.debug.assert(shardings.len >= 1, "Waiting at leat 1 sharding strategy to be implemented", .{});
            var plan: Plan = .{ .partitioner = partitioner, .shardings = shardings };

            const first = plan.primarySharding();
            const partitions = first.numPartitions();
            const replicas = first.numReplicas();

            for (shardings[1..]) |s| {
                if (s.numPartitions() != partitions or s.numReplicas() != replicas) {
                    // todo deviceAssignments should also be checked for consistency here, but for simplicity we just check the cardinality numbers
                    return error.InconsistentShardingCardinality;
                }
            }

            return plan;
        }

        pub fn kind(self: Plan) Partitioner {
            return self.partitioner;
        }

        pub fn numPartitions(self: Plan) !i32 {
            return self.cardinality().num_partitions;
        }

        pub fn numReplicas(self: Plan) !i32 {
            return self.cardinality().num_replicas;
        }

        pub fn numDevices(self: Plan) !i32 {
            const card = self.cardinality();
            return card.num_partitions * card.num_replicas;
        }

        pub fn deviceAssignment(self: Plan, allocator: std.mem.Allocator) ![]usize {
            return switch (self.partitioner) {
                .shardy => blk: {
                    const sharding = self.primarySharding();
                    break :blk sharding.deviceAssignment(allocator);
                },
                .gspmd => error.UnimplementedPartitioner,
            };
        }

        pub fn tensorShardingAttr(self: Plan, allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) !?[]const u8 {
            return switch (self.partitioner) {
                .shardy => sharding.shardingAttrForShape(allocator, shape),
                .gspmd => null,
            };
        }

        pub fn meshDescriptors(self: Plan, allocator: std.mem.Allocator) ![]MeshDescriptor {
            return switch (self.partitioner) {
                .shardy => blk: {
                    var seen: std.StringHashMap(void) = .init(allocator);
                    defer seen.deinit();

                    var list = std.array_list.Managed(MeshDescriptor).init(allocator);
                    errdefer list.deinit();

                    for (self.shardings) |sharding| {
                        const name = sharding.meshName();
                        if (seen.contains(name)) continue;
                        try seen.put(name, {});
                        const attr = try sharding.meshAttrString(allocator);
                        try list.append(.{ .name = name, .attr = attr });
                    }

                    break :blk try list.toOwnedSlice();
                },
                .gspmd => error.UnimplementedPartitioner,
            };
        }

        pub fn selectSharding(self: Plan, shape: Shape) !Sharding {
            for (self.shardings) |s| {
                if (self.shardingCoversShape(s, shape)) return s;
            }

            return error.NoSuitableSharding;
        }

        fn cardinality(self: Plan) Cardinality {
            const first = self.primarySharding();

            return .{ .num_partitions = first.numPartitions(), .num_replicas = first.numReplicas() };
        }

        fn primarySharding(self: Plan) Sharding {
            return self.shardings[0];
        }

        fn shardingCoversShape(self: Plan, sharding: Sharding, shape: Shape) bool {
            _ = self;
            for (0..shape.rank()) |ax| {
                switch (shape.partition(ax)) {
                    .axis => |tag| if (sharding.binding(tag) == null) return false,
                    else => {},
                }
            }
            return true;
        }
    };
};

pub const Device = struct {
    /// Unique device identifier in the mesh
    id: usize,

    /// Coordinates in the physical mesh
    coords: ?[]const usize = null,

    /// Compute capacity local to this device (cores, SMs, etc.).
    compute_units: usize,

    /// Placeholder
    pjrt_device: ?*const pjrt.Device = null,

    pub fn format(self: Device, writer: *std.Io.Writer) !void {
        try writer.print(
            "Device(id={d} compute_units={d} coords={any})",
            .{ self.id, self.compute_units, self.coords },
        );
    }
};

pub const PhysicalAxisTag = enum {
    link, // 1D Interconnect (Inf1, Inf2 Islands)
    link_x, // Torus/Mesh Dim X (TPU, Trainium)
    link_y, // Torus/Mesh Dim Y (TPU, Trainium)
    link_z, // Torus/Mesh Dim Z (TPU v3/v4/v5)
    bus, // PCIe / NUMA (Potential isolation boundary)
};

pub const RingKind = enum {
    linear, // Neighbors only, open ends
    closed_ring, // Neighbors only, circular (wrap-around)
};

pub const MeshKind = enum {
    grid, // 2D/3D Neighbors only, open ends
    torus, // 2D/3D Neighbors only, circular wrap-around
};

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
                .coords = null,
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
            .leaf => |l| {
                if (l.coords) |coords| {
                    allocator.free(coords);
                }
            },
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

/// PhysicalMesh models the hardware as a tree of axes.
/// Each branch = one physical axis (link_x/link_y/bus/etc).
/// Each leaf = one device.
///
/// How it works:
/// - buildFromFields walks the axis list and assigns coords to each device.
/// - coords are ordered by axis traversal (root → leaf).
/// - axisOrder() later uses the same traversal to define axis depth.
///
/// Limitations:
/// - Assumes a single uniform topology (no heterogeneity).
/// - Only the axes that appear in the tree are available for sharding.
pub const PhysicalMesh = struct {
    /// Explicit topology tree API (canonical form)
    pub const Tree = PhysicalNode;

    pub const AxisTraversal = struct {
        pub const Order = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK);
        pub const DepthByTag = std.EnumArray(PhysicalAxisTag, ?u8);

        order: Order,
        depth_by_tag: DepthByTag,

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

    pub const AxisInfo = struct {
        tag: PhysicalAxisTag,
        size: i64,
        geometry: AxisGeometry,
    };

    allocator: std.mem.Allocator,
    target: Target,
    root: PhysicalNode,
    axis_traversal: AxisTraversal,

    pub fn fromTree(allocator: std.mem.Allocator, target: Target, root: PhysicalNode) !PhysicalMesh {
        var cloned = try cloneNode(allocator, root);
        errdefer cloned.deinit(allocator);

        try validateGeometry(cloned);

        var path = [_]usize{0} ** 16;
        try assignCoords(&cloned, allocator, &path, 0);

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
                    .coords = null,
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

    fn assignCoords(node: *PhysicalNode, allocator: std.mem.Allocator, path: *[16]usize, depth: usize) !void {
        switch (node.*) {
            .leaf => |*d| {
                if (d.coords) |coords| allocator.free(coords);
                d.coords = try allocator.dupe(usize, path[0..depth]);
            },
            .branch => |*b| {
                for (b.children, 0..) |*child, i| {
                    path[depth] = i;
                    try assignCoords(child, allocator, path, depth + 1);
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
        var list = stdx.BoundedArray(Device, Platform.MAX_NUM_DEVICES).init(0) catch unreachable;
        try self.devicesInto(&list);

        const out = try allocator.alloc(Device, list.len);
        @memcpy(out, list.constSlice());
        return out;
    }

    /// Append devices into a bounded array without allocations.
    pub fn devicesInto(self: PhysicalMesh, out: *stdx.BoundedArray(Device, Platform.MAX_NUM_DEVICES)) !void {
        try devicesNodeInto(self.root, out);
    }

    fn devicesNodeInto(node: PhysicalNode, out: *stdx.BoundedArray(Device, Platform.MAX_NUM_DEVICES)) !void {
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

    /// Fastest -> Slowest
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
        for (self.axes.constSlice(), self.intents.constSlice()) |t, i| {
            if (std.mem.eql(u8, std.mem.span(t), std.mem.span(tag))) return i;
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
    pub const Bindings = std.AutoArrayHashMapUnmanaged(Shape.Tag, AxisList);

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

    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    physical: PhysicalMesh,

    /// Compact binding table: logical axis -> physical axes
    bindings: Bindings,

    /// Folded axis mapping: kept axis -> ordered source axes
    folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, AxisList),

    pub fn binding(self: Sharding, tag: Shape.Tag) ?[]const PhysicalAxisTag {
        if (self.bindings.get(tag)) |axes| return axes.constSlice();
        return null;
    }

    pub fn deinit(self: *Sharding) void {
        self.bindings.deinit(self.allocator);
        self.folds.deinit(self.allocator);
    }

    pub fn initFromStrategy(
        allocator: std.mem.Allocator,
        logical: LogicalMesh,
        physical: PhysicalMesh,
        strategy: Strategy,
    ) !Sharding {
        const axis_order = physical.axisOrder().constSlice();
        if (axis_order.len == 0) return error.InvalidPhysicalMesh;

        var filtered = try Sharding.filterBindings(allocator, physical, strategy);
        errdefer filtered.bindings.deinit(allocator);

        const folds = if (strategy.folding.count() > 0)
            try Sharding.buildFoldsExplicit(allocator, physical, axis_order, strategy)
        else
            try Sharding.buildFoldsDefault(allocator, physical, logical, filtered.bindings, filtered.axis_used, axis_order);
        errdefer folds.deinit(allocator);

        return .{
            .allocator = allocator,
            .logical = logical,
            .physical = physical,
            .bindings = filtered.bindings,
            .folds = folds,
        };
    }

    pub fn physicalView(self: Sharding) PhysicalView {
        var view: PhysicalView = .{
            .axes = stdx.BoundedArray(Axis, Shape.MAX_RANK).init(0) catch unreachable,
            .total_devices = 1,
        };

        const axis_order = self.physical.axisOrder().constSlice();

        if (self.folds.count() > 0) {
            for (axis_order) |tag| {
                const sources = self.folds.get(tag) orelse continue;

                var folded = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
                var size: i64 = 1;
                for (sources.constSlice()) |src| {
                    folded.appendAssumeCapacity(src);
                    size *= self.physical.axis(src);
                }

                view.total_devices *= size;
                view.axes.appendAssumeCapacity(.{
                    .tag = tag,
                    .size = size,
                    .geometry = self.physical.geometry(tag),
                    .folded = folded,
                });
            }
        } else {
            for (axis_order) |tag| {
                var folded = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
                folded.appendAssumeCapacity(tag);

                const size = self.physical.axis(tag);
                view.total_devices *= size;
                view.axes.appendAssumeCapacity(.{
                    .tag = tag,
                    .size = size,
                    .geometry = self.physical.geometry(tag),
                    .folded = folded,
                });
            }
        }

        return view;
    }

    fn foldedAxisIndex(self: Sharding, tag: PhysicalAxisTag, coords: []const usize) usize {
        var buf = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        const sources = self.folds.get(tag) orelse blk: {
            buf.appendAssumeCapacity(tag);
            break :blk buf;
        };

        var idx: usize = 0;
        for (sources.constSlice()) |src| {
            const depth = self.physical.axis_traversal.depth(src) orelse unreachable;
            const coord = coords[@intCast(depth)];
            const size = self.physical.axis(src);
            idx = idx * @as(usize, @intCast(size)) + coord;
        }
        return idx;
    }

    fn foldedCoord(self: Sharding, tag: PhysicalAxisTag, coords: []const usize) usize {
        var buf = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        const sources = self.folds.get(tag) orelse blk: {
            buf.appendAssumeCapacity(tag);
            break :blk buf;
        };

        var idx: usize = 0;
        for (sources.constSlice()) |src| {
            const depth = self.physical.axis_traversal.depth(src) orelse unreachable;
            const coord = coords[@intCast(depth)];
            const size = self.physical.axis(src);
            idx = idx * @as(usize, @intCast(size)) + coord;
        }
        return idx;
    }

    // ---- private helpers moved inside Sharding ----

    const FilteredBindings = struct {
        bindings: Bindings,
        axis_used: std.EnumSet(PhysicalAxisTag),
    };

    fn filterBindings(
        allocator: std.mem.Allocator,
        physical: PhysicalMesh,
        strategy: Strategy,
    ) !FilteredBindings {
        var bindings: Bindings = .{};
        errdefer bindings.deinit(allocator);

        var axis_used = std.EnumSet(PhysicalAxisTag).initEmpty();

        var it = strategy.bindings.iterator();
        while (it.next()) |bind| {
            const l_tag = bind.key_ptr.*;

            var list: AxisList = try .init(0);
            for (bind.value_ptr.physical.constSlice()) |p_tag| {
                if (physical.hasAxis(p_tag)) {
                    list.appendAssumeCapacity(p_tag);
                    axis_used.insert(p_tag);
                }
            }

            try bindings.put(allocator, l_tag, list);
        }

        return .{ .bindings = bindings, .axis_used = axis_used };
    }

    fn buildFoldsExplicit(
        allocator: std.mem.Allocator,
        physical: PhysicalMesh,
        axis_order: []const PhysicalAxisTag,
        strategy: Strategy,
    ) !std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, AxisList) {
        var folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, AxisList) = .{};
        errdefer folds.deinit(allocator);

        var it = strategy.folding.iterator();
        while (it.next()) |entry| {
            const target = entry.key_ptr.*;
            if (!physical.hasAxis(target)) return error.InvalidPhysicalAxis;

            const sources = entry.value_ptr.*;
            for (sources.constSlice()) |src| {
                if (!physical.hasAxis(src)) return error.InvalidPhysicalAxis;
            }

            try folds.put(allocator, target, sources);
        }

        // Ensure every axis has a fold list (identity if missing).
        for (axis_order) |tag| {
            if (!folds.contains(tag)) {
                var one: AxisList = try .init(0);
                one.appendAssumeCapacity(tag);
                try folds.put(allocator, tag, one);
            }
        }

        return folds;
    }

    fn buildFoldsDefault(
        allocator: std.mem.Allocator,
        physical: PhysicalMesh,
        logical: LogicalMesh,
        bindings: Bindings,
        axis_used: std.EnumSet(PhysicalAxisTag),
        axis_order: []const PhysicalAxisTag,
    ) !std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, AxisList) {
        var folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, AxisList) = .{};
        errdefer folds.deinit(allocator);

        // Start with identity folds for used axes.
        for (axis_order) |tag| {
            if (axis_used.contains(tag)) {
                var one: AxisList = try .init(0);
                one.appendAssumeCapacity(tag);
                try folds.put(allocator, tag, one);
            }
        }

        const fold_target = Sharding.selectFoldTarget(logical, bindings, physical) orelse axis_order[0];

        // Collect unused axes (excluding fold_target).
        var unused = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        for (axis_order) |tag| {
            if (!axis_used.contains(tag) and tag != fold_target) {
                unused.appendAssumeCapacity(tag);
            }
        }

        // Ensure target exists.
        var res = try folds.getOrPut(allocator, fold_target);
        if (!res.found_existing) {
            var one: AxisList = try .init(0);
            one.appendAssumeCapacity(fold_target);
            res.value_ptr.* = one;
        }

        // Append unused axes if missing.
        for (unused.constSlice()) |tag| {
            if (!Sharding.listContains(res.value_ptr.*, tag)) {
                res.value_ptr.appendAssumeCapacity(tag);
            }
        }

        return folds;
    }

    fn listContains(list: AxisList, tag: PhysicalAxisTag) bool {
        for (list.constSlice()) |t| if (t == tag) return true;
        return false;
    }

    fn selectFoldTarget(
        logical: LogicalMesh,
        bindings: Bindings,
        physical: PhysicalMesh,
    ) ?PhysicalAxisTag {
        var best_tag: ?PhysicalAxisTag = null;
        var best_priority: i32 = 999;
        var best_size: i64 = -1;

        for (logical.axes.constSlice(), logical.intents.constSlice()) |l_tag, l_intent| {
            const binding_ = bindings.get(l_tag) orelse continue;
            if (binding_.len == 0) continue;

            const p_tag = binding_.constSlice()[0];
            if (!physical.hasAxis(p_tag)) continue;

            const priority = @intFromEnum(l_intent);
            const size = physical.axis(p_tag);

            if (priority < best_priority or (priority == best_priority and size > best_size)) {
                best_priority = priority;
                best_size = size;
                best_tag = p_tag;
            }
        }

        return best_tag;
    }

    fn devicesInCanonicalOrder(self: Sharding) !stdx.BoundedArray(Device, Platform.MAX_NUM_DEVICES) {
        var devices = stdx.BoundedArray(Device, Platform.MAX_NUM_DEVICES).init(0) catch unreachable;
        try self.physical.devicesInto(&devices);

        const Order = struct {
            mesh: PhysicalMesh,
            fn lessThan(ctx: @This(), a: Device, b: Device) bool {
                const ca = a.coords.?;
                const cb = b.coords.?;
                const ia = ctx.mesh.linearIndexFromCoords(ca);
                const ib = ctx.mesh.linearIndexFromCoords(cb);
                return ia < ib;
            }
        };

        std.mem.sort(Device, devices.slice(), Order{ .mesh = self.physical }, Order.lessThan);
        return devices;
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

    pub fn meshName(self: Sharding) []const u8 {
        return self.logical.name;
    }

    /// Preferred naming for SDY mesh attribute.
    pub fn sdyMeshAttrString(self: Sharding, allocator: std.mem.Allocator) ![]const u8 {
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

    pub fn meshAttrString(self: Sharding, allocator: std.mem.Allocator) ![]const u8 {
        return self.sdyMeshAttrString(allocator);
    }

    /// Preferred naming for SDY sharding attribute.
    pub fn sdyShardingAttrForShape(self: Sharding, allocator: std.mem.Allocator, shape: Shape) !?[]const u8 {
        var any_explicit = false;
        for (0..shape.rank()) |ax| {
            switch (shape.partition(ax)) {
                .axis => |logical_tag| {
                    if (self.binding(logical_tag) != null) {
                        any_explicit = true;
                        break;
                    }
                },
                .replicated, .open => {
                    any_explicit = true;
                    break;
                },
                .unknown => {},
            }
        }
        if (!any_explicit) return null;

        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        try out.writer.print("#sdy.sharding<@{s}, [", .{self.meshName()});

        const view = self.physicalView();
        var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();

        for (0..shape.rank()) |ax| {
            if (ax > 0) try out.writer.writeAll(", ");

            const spec = shape.partition(ax);
            switch (spec) {
                .axis => |logical_tag| {
                    const binding_opt = self.binding(logical_tag);
                    if (binding_opt == null) {
                        try out.writer.writeAll("{?}");
                        continue;
                    }
                    const binding_ = binding_opt.?;

                    var wrote_any = false;
                    var first = true;

                    for (binding_) |p_tag| {
                        for (view.axes.constSlice()) |axis| {
                            if (axis.contains(p_tag) and !used_axes.contains(axis.tag)) {
                                if (!wrote_any) {
                                    try out.writer.writeAll("{");
                                    wrote_any = true;
                                }
                                if (!first) try out.writer.writeAll(", ");
                                try out.writer.print("\"{s}\"", .{@tagName(axis.tag)});
                                used_axes.insert(axis.tag);
                                first = false;
                                break;
                            }
                        }
                    }

                    if (!wrote_any) {
                        try out.writer.writeAll("{?}");
                    } else {
                        try out.writer.writeAll("}");
                    }
                },
                .replicated => try out.writer.writeAll("{}"),
                .open => try out.writer.writeAll("{?}"),
                .unknown => try out.writer.writeAll("{?}"),
            }
        }

        try out.writer.writeAll("]");

        var replicated_axes = std.EnumSet(PhysicalAxisTag).initEmpty();
        for (view.axes.constSlice()) |axis| {
            if (!used_axes.contains(axis.tag)) {
                replicated_axes.insert(axis.tag);
            }
        }

        if (replicated_axes.count() > 0) {
            try out.writer.writeAll(", replicated={");
            var first = true;
            for (view.axes.constSlice()) |axis| {
                if (replicated_axes.contains(axis.tag)) {
                    if (!first) try out.writer.writeAll(", ");
                    try out.writer.print("\"{s}\"", .{@tagName(axis.tag)});
                    first = false;
                }
            }
            try out.writer.writeAll("}");
        }

        try out.writer.writeAll(">");
        return try out.toOwnedSlice();
    }

    pub fn shardingAttrForShape(self: Sharding, allocator: std.mem.Allocator, shape: Shape) !?[]const u8 {
        return self.sdyShardingAttrForShape(allocator, shape);
    }

    pub fn pjrtDeviceAssignment(self: Sharding, allocator: std.mem.Allocator) ![]usize {
        const view = self.physicalView();
        const count: usize = @intCast(view.total_devices);

        var ids = try allocator.alloc(usize, count);
        @memset(ids, std.math.maxInt(usize));

        const ordered_devices = try self.devicesInCanonicalOrder();
        for (ordered_devices.constSlice()) |d| {
            const coords = d.coords orelse return error.MissingDeviceCoords;
            const idx = self.physical.linearIndexFromCoords(coords);
            ids[idx] = d.id;
        }

        for (ids) |id| {
            if (id == std.math.maxInt(usize)) return error.MissingDeviceInTile;
        }

        return ids;
    }

    pub fn deviceAssignment(self: Sharding, allocator: std.mem.Allocator) ![]usize {
        return self.pjrtDeviceAssignment(allocator);
    }

    pub fn format(self: Sharding, writer: *std.Io.Writer) !void {
        try writer.print("Sharding(name={s})\n", .{self.logical.name});

        try writer.writeAll("Bindings:\n");
        for (self.logical.axes.constSlice(), self.logical.intents.constSlice()) |l_tag, l_intent| {
            try writer.print("  - {s} ({s}) -> ", .{ l_tag, @tagName(l_intent) });

            if (self.bindings.get(l_tag)) |axes| {
                if (axes.len == 0) {
                    try writer.writeAll("replicated\n");
                } else {
                    for (axes.constSlice(), 0..) |p, i| {
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

pub const Strategy = struct {
    pub const PhysicalList = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK);

    pub const Binding = struct {
        logical: Shape.Tag,
        physical: PhysicalList,
    };

    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, Binding),

    /// Explicit folding rules: kept axis -> ordered source axes.
    /// Example: fold link_x + link_z into link_x => { link_x: [link_x, link_z] }
    folding: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, PhysicalList),

    pub const init: Strategy = .{
        .bindings = .{},
        .folding = .{},
    };

    pub fn deinit(self: *Strategy, allocator: std.mem.Allocator) void {
        self.bindings.deinit(allocator);
        self.folding.deinit(allocator);
    }

    pub fn addBinding(self: *Strategy, allocator: std.mem.Allocator, logical: Shape.Tag, physical: PhysicalAxisTag) !void {
        var res = try self.bindings.getOrPut(allocator, logical);

        if (!res.found_existing) {
            var list: PhysicalList = try .init(0);
            list.appendAssumeCapacity(physical);
            res.value_ptr.* = .{ .logical = logical, .physical = list };
            return;
        }

        res.value_ptr.physical.appendAssumeCapacity(physical);
    }

    /// Explicitly fold axes: `target` is the kept axis, `sources` define order.
    /// If `target` is missing from `sources`, it is prepended.
    pub fn addFold(self: *Strategy, allocator: std.mem.Allocator, target: PhysicalAxisTag, sources: []const PhysicalAxisTag) !void {
        var has_target = false;
        for (sources) |s| {
            if (s == target) {
                has_target = true;
                break;
            }
        }

        var list: PhysicalList = try .init(0);
        if (!has_target) list.appendAssumeCapacity(target);
        for (sources) |s| list.appendAssumeCapacity(s);

        try self.folding.put(allocator, target, list);
    }

    /// suggest builds a Strategy from logical intents.
    pub fn suggest(
        allocator: std.mem.Allocator,
        logical: LogicalMesh,
        physical: PhysicalMesh,
    ) !Strategy {
        var strategy: Strategy = .init;
        errdefer strategy.deinit(allocator);

        var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();
        const base_order = physical.shardableAxes();

        inline for (.{ .high_bandwidth, .balanced, .low_bandwidth }) |intent| {
            var preferred_axes: PhysicalList = try .init(0);

            switch (intent) {
                .high_bandwidth => {
                    for (base_order) |t| Strategy.appendIfExists(physical, &preferred_axes, t);
                },
                .balanced => {
                    if (base_order.len > 1) {
                        for (base_order[1..]) |t| Strategy.appendIfExists(physical, &preferred_axes, t);
                        Strategy.appendIfExists(physical, &preferred_axes, base_order[0]);
                    } else if (base_order.len == 1) {
                        Strategy.appendIfExists(physical, &preferred_axes, base_order[0]);
                    }
                },
                .low_bandwidth => {
                    var i = base_order.len;
                    while (i > 0) : (i -= 1) {
                        Strategy.appendIfExists(physical, &preferred_axes, base_order[i - 1]);
                    }
                },
                else => unreachable,
            }

            if (preferred_axes.len == 0) {
                for (physical.axisOrder().constSlice()) |t| {
                    preferred_axes.appendAssumeCapacity(t);
                }
            }

            for (logical.axes.constSlice(), logical.intents.constSlice()) |logical_axis, logical_intent| {
                if (logical_intent != intent) continue;
                if (preferred_axes.len == 0) continue;

                var chosen_axis: ?PhysicalAxisTag = null;
                for (preferred_axes.constSlice()) |candidate| {
                    if (!used_axes.contains(candidate)) {
                        chosen_axis = candidate;
                        used_axes.insert(candidate);
                        break;
                    }
                }
                if (chosen_axis == null) {
                    chosen_axis = preferred_axes.constSlice()[0];
                }

                try strategy.addBinding(allocator, logical_axis, chosen_axis.?);
            }
        }

        return strategy;
    }

    fn appendIfExists(physical: PhysicalMesh, out: *PhysicalList, tag: PhysicalAxisTag) void {
        if (physical.hasAxis(tag)) out.appendAssumeCapacity(tag);
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
        device_coords: []const usize,
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
            try writer.print("device_id={d} coords={any}", .{ self.device_id, self.device_coords });

            if (self.slices.len == 0) {
                try writer.writeAll(" slices=replicated");
                return;
            }

            try writer.writeAll(" slices=[");
            for (self.slices.constSlice(), 0..) |s, i| {
                if (i > 0) try writer.writeAll(", ");
                const axis_label = self.global_shape.debugTag(s.axis);
                try writer.print("{s}:[{d}:{d}]", .{ axis_label, s.start, s.start + s.size });
            }
            try writer.writeAll("]");
        }
    };

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
    shards: stdx.BoundedArray(Shard, Platform.MAX_NUM_DEVICES),

    pub fn init(
        sharding: Sharding,
        shape: Shape,
    ) !Placement {
        const view = sharding.physicalView();
        const shards_len: usize = @intCast(view.total_devices);

        var shards = stdx.BoundedArray(Shard, Platform.MAX_NUM_DEVICES).init(0) catch unreachable;
        stdx.debug.assert(shards_len <= shards.capacity(), "Too many shards: {d} > {d}", .{ shards_len, shards.capacity() });

        // Pre-fill shards so we can index by device order later.
        for (0..shards_len) |_| {
            shards.appendAssumeCapacity(.{
                .device_id = 0,
                .device_coords = &.{},
                .shape = undefined,
                .global_shape = shape,
                .slices = stdx.BoundedArray(Slice1d, Shape.MAX_RANK).init(0) catch unreachable,
            });
        }

        const ordered_devices = try sharding.devicesInCanonicalOrder();
        for (ordered_devices.constSlice()) |d| {
            const coords = d.coords orelse return error.MissingDeviceCoords;
            const idx = sharding.physical.linearIndexFromCoords(coords);
            shards.slice()[idx].device_id = d.id;
            shards.slice()[idx].device_coords = coords;
        }

        var self: Placement = .{
            .sharding = sharding,
            .shape = shape,
            .shards = shards,
        };

        for (self.shards.slice(), 0..) |*s, shard_idx| {
            var slices = try stdx.BoundedArray(Slice1d, Shape.MAX_RANK).init(0);
            var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();

            for (0..shape.rank()) |ax| {
                const axis_index: u8 = @intCast(ax);
                const slice = try self.sliceForAxis(view, shard_idx, &used_axes, axis_index);
                slices.appendAssumeCapacity(slice);
            }

            s.slices = slices;
            s.shape = blk: {
                var sh = shape;
                for (slices.constSlice()) |slice| {
                    sh = sh.set(slice.axis, slice.size);
                }
                break :blk sh;
            };
        }

        return self;
    }

    fn sliceForAxis(
        self: *Placement,
        view: Sharding.PhysicalView,
        shard_idx: usize,
        used_axes: *std.EnumSet(PhysicalAxisTag),
        axis_index: u8,
    ) !Slice1d {
        const dim = self.shape.dim(axis_index);
        const spec = self.shape.partition(axis_index);

        switch (spec) {
            .axis => |logical_tag| {
                const binding = self.sharding.binding(logical_tag) orelse return error.MissingLogicalBinding;

                const plan = try selectPhysicalAxesForDim(dim, binding, view, shard_idx, used_axes);

                const size = @divExact(dim, plan.product);
                const start = plan.linearIndex() * size;

                return .{
                    .axis = axis_index,
                    .start = start,
                    .size = size,
                };
            },
            .replicated => return .{ .axis = axis_index, .start = 0, .size = dim },
            .open => return .{ .axis = axis_index, .start = 0, .size = dim },
            .unknown => return .{ .axis = axis_index, .start = 0, .size = dim },
        }
    }

    fn selectPhysicalAxesForDim(
        dim: i64,
        binding: []const PhysicalAxisTag,
        view: Sharding.PhysicalView,
        shard_idx: usize,
        used_axes: *std.EnumSet(PhysicalAxisTag),
    ) !AxisSplit {
        var plan: AxisSplit = try .init();

        for (binding) |p_tag| {
            for (view.axes.constSlice(), 0..) |axis, i| {
                if (axis.contains(p_tag) and !used_axes.contains(axis.tag)) {
                    const next_product = plan.product * axis.size;
                    if (@rem(dim, next_product) == 0) {
                        const coord = view.axisCoordFromLinearShard(i, shard_idx);
                        plan.counts.appendAssumeCapacity(axis.size);
                        plan.indices.appendAssumeCapacity(@intCast(coord));
                        plan.product = next_product;
                        used_axes.insert(axis.tag);
                    }
                    break;
                }
            }
        }

        return plan;
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
