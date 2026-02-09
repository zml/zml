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

    pub fn inputAttribute(self: Partitioner) ![]const u8 {
        return switch (self) {
            .shardy => "#shardy.partition",
            .gspmd => error.UnsupportedPartitioner,
        };
    }
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
        var list: std.ArrayList(Device) = .{};
        errdefer list.deinit(allocator);

        try devicesNode(allocator, self.root, &list);
        return try list.toOwnedSlice(allocator);
    }

    fn devicesNode(allocator: std.mem.Allocator, node: PhysicalNode, list: *std.ArrayList(Device)) !void {
        switch (node) {
            .leaf => |d| try list.append(allocator, d),
            .branch => |b| {
                for (b.children) |child| {
                    try devicesNode(allocator, child, list);
                }
            },
        }
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

/// Sharding is the program‑level mapping:
/// logical axes → physical axes → tile.
///
/// It is *not* per‑tensor.
/// It is *not* a concrete placement by itself.
///
/// What it contains:
/// - tiles: which physical axes are used and their sizes.
/// - bindings: logical axis → physical axis list.
///
/// This is the "rules of the road" for placement.
/// Per‑tensor assignment happens later.
pub const Sharding = struct {
    pub const Bindings = std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag);

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

    pub const AxisView = struct {
        axes: stdx.BoundedArray(Axis, Shape.MAX_RANK),
        total_devices: i64,

        pub fn axisCoordFromLinearIndex(self: *const AxisView, axis_index: usize, linear_idx: usize) usize {
            const stride = self.axisStride(axis_index);
            const axis_size: usize = @intCast(self.axes.constSlice()[axis_index].size);
            return (linear_idx / stride) % axis_size;
        }

        pub fn axisStride(self: *const AxisView, axis_index: usize) usize {
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
    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag),

    /// Folded axis mapping: kept axis -> ordered source axes
    folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag),

    pub fn binding(self: Sharding, tag: Shape.Tag) ?[]const PhysicalAxisTag {
        if (self.bindings.get(tag)) |axes| return axes;
        return null;
    }

    pub fn deinit(self: *Sharding) void {
        var it = self.bindings.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.bindings.deinit(self.allocator);

        var fit = self.folds.iterator();
        while (fit.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.folds.deinit(self.allocator);
    }

    pub fn axisView(self: Sharding) AxisView {
        var view: AxisView = .{
            .axes = stdx.BoundedArray(Axis, Shape.MAX_RANK).init(0) catch unreachable,
            .total_devices = 1,
        };

        const axis_order = self.physical.axisOrder().constSlice();

        if (self.folds.count() > 0) {
            for (axis_order) |tag| {
                const sources = self.folds.get(tag) orelse continue;

                var folded = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
                var size: i64 = 1;
                for (sources) |src| {
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

    fn foldedCoord(self: Sharding, tag: PhysicalAxisTag, coords: []const usize) usize {
        var buf = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        const sources = self.folds.get(tag) orelse blk: {
            buf.appendAssumeCapacity(tag);
            break :blk buf.constSlice();
        };

        var idx: usize = 0;
        for (sources) |src| {
            const depth = self.physical.axis_traversal.depth(src) orelse unreachable;
            const coord = coords[@intCast(depth)];
            const size = self.physical.axis(src);
            idx = idx * @as(usize, @intCast(size)) + coord;
        }
        return idx;
    }

    fn globalIndexFromCoords(self: Sharding, coords: []const usize) usize {
        const order = self.physical.axisOrder();
        var idx: usize = 0;
        for (order.constSlice()) |tag| {
            const depth = self.physical.axis_traversal.depth(tag);
            if (depth) |d| {
                const coord = coords[@intCast(d)];
                const size = self.physical.axis(tag);
                idx = idx * @as(usize, @intCast(size)) + coord;
            }
        }
        return idx;
    }

    fn deviceOrder(self: Sharding, allocator: std.mem.Allocator) ![]Device {
        const devices = try self.physical.devices(allocator);

        const Order = struct {
            sharding: Sharding,
            fn lessThan(ctx: @This(), a: Device, b: Device) bool {
                const ca = a.coords.?;
                const cb = b.coords.?;
                const ia = ctx.sharding.globalIndexFromCoords(ca);
                const ib = ctx.sharding.globalIndexFromCoords(cb);
                return ia < ib;
            }
        };

        std.mem.sort(Device, devices, Order{ .sharding = self }, Order.lessThan);
        return devices;
    }

    fn shardIndexAndCoords(
        self: Sharding,
        allocator: std.mem.Allocator,
        coords: []const usize,
    ) !struct { idx: usize, axis_coords: []usize } {
        const view = self.axisView();
        const axes_len = view.axes.len;

        var axis_coords = try allocator.alloc(usize, axes_len);

        for (view.axes.constSlice(), 0..) |axis, i| {
            const coord = self.foldedCoord(axis.tag, coords);
            axis_coords[i] = coord;
        }

        const idx = self.globalIndexFromCoords(coords);

        return .{ .idx = idx, .axis_coords = axis_coords };
    }

    pub fn numPartitions(self: Sharding) i32 {
        return @intCast(self.axisView().total_devices);
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

    /// Returns a *parsed* mesh attribute string.
    /// Example: #sdy.mesh<["x"=2, "y"=4]>
    pub fn meshAttrString(self: Sharding, allocator: std.mem.Allocator) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();

        const view = self.axisView();
        try out.writer.writeAll("#sdy.mesh<[");
        for (view.axes.constSlice(), 0..) |p, i| {
            if (i > 0) try out.writer.writeAll(", ");
            try out.writer.print("\"{s}\"={d}", .{ @tagName(p.tag), p.size });
        }
        try out.writer.writeAll("]>");

        return try out.toOwnedSlice();
    }

    /// Build a sharding attribute string for a given shape.
    /// Returns null when all dims are unknown/open (fully open tensor).
    pub fn shardingAttrForShape(self: Sharding, allocator: std.mem.Allocator, shape: Shape) !?[]const u8 {
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

        const view = self.axisView();
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

        // replicate unused axes
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

    pub fn deviceAssignment(self: Sharding, allocator: std.mem.Allocator) ![]usize {
        const view = self.axisView();
        const count: usize = @intCast(view.total_devices);

        var ids = try allocator.alloc(usize, count);
        @memset(ids, std.math.maxInt(usize));

        const ordered_devices = try self.deviceOrder(allocator);
        defer allocator.free(ordered_devices);

        for (ordered_devices) |d| {
            const coords = d.coords orelse return error.MissingDeviceCoords;
            const idx = self.globalIndexFromCoords(coords);
            ids[idx] = d.id;
        }

        for (ids) |id| {
            if (id == std.math.maxInt(usize)) return error.MissingDeviceInTile;
        }

        return ids;
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

        const view = self.axisView();

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

/// resolveStrategyConstraints builds a Sharding plan **without concrete sizes**.
///
/// What it does (and why):
/// - Filter Strategy bindings to only axes that exist on the physical mesh.
///   (Why: Strategies can be generic; this makes the plan mesh‑specific.)
/// - Compute folding rules:
///   - If user supplied explicit folds, use them (validated) and fill missing axes.
///   - Else, use a default fold that merges unused axes into a single target.
///   (Why: keeps all devices visible while allowing logical meshes with fewer axes.)
///
/// What it does NOT do:
/// - No divisibility checks (those happen in assignTensor).
/// - No explicit tile state (AxisView derives sizes later).
pub fn resolveStrategyConstraints(
    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    physical: PhysicalMesh,
    strategy: Strategy,
) !Sharding {
    const axis_order = physical.axisOrder().constSlice();
    if (axis_order.len == 0) return error.InvalidPhysicalMesh;

    var filtered = try filterBindings(allocator, physical, strategy);
    errdefer {
        var it = filtered.bindings.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        filtered.bindings.deinit(allocator);
    }

    const folds = if (strategy.folding.count() > 0)
        try buildFoldsExplicit(allocator, physical, axis_order, strategy)
    else
        try buildFoldsDefault(allocator, physical, logical, filtered.bindings, filtered.axis_used, axis_order);
    errdefer {
        var it = folds.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        folds.deinit(allocator);
    }

    return .{
        .allocator = allocator,
        .logical = logical,
        .physical = physical,
        .bindings = filtered.bindings,
        .folds = folds,
    };
}

const FilteredBindings = struct {
    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag),
    axis_used: std.EnumSet(PhysicalAxisTag),
};

/// Filter bindings to only axes that exist on the mesh.
/// Also tracks which axes are used by any logical axis.
fn filterBindings(
    allocator: std.mem.Allocator,
    physical: PhysicalMesh,
    strategy: Strategy,
) !FilteredBindings {
    var bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag) = .{};
    errdefer bindings.deinit(allocator);

    var axis_used = std.EnumSet(PhysicalAxisTag).initEmpty();

    var it = strategy.bindings.iterator();
    while (it.next()) |bind| {
        const l_tag = bind.key_ptr.*;

        // Count valid axes first to allocate exactly once.
        var count: usize = 0;
        for (bind.value_ptr.physical.items) |p_tag| {
            if (physical.hasAxis(p_tag)) count += 1;
        }

        var slice = try allocator.alloc(PhysicalAxisTag, count);
        var idx: usize = 0;
        for (bind.value_ptr.physical.items) |p_tag| {
            if (physical.hasAxis(p_tag)) {
                slice[idx] = p_tag;
                idx += 1;
                axis_used.insert(p_tag);
            }
        }

        try bindings.put(allocator, l_tag, slice);
    }

    return .{ .bindings = bindings, .axis_used = axis_used };
}

/// Build folds from explicit user rules.
/// - Validates all axes.
/// - Ensures every axis in the mesh has a fold entry (identity if missing).
fn buildFoldsExplicit(
    allocator: std.mem.Allocator,
    physical: PhysicalMesh,
    axis_order: []const PhysicalAxisTag,
    strategy: Strategy,
) !std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag) {
    var folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag) = .{};
    errdefer folds.deinit(allocator);

    var it = strategy.folding.iterator();
    while (it.next()) |entry| {
        const target = entry.key_ptr.*;
        if (!physical.hasAxis(target)) return error.InvalidPhysicalAxis;

        const sources = entry.value_ptr.*;
        for (sources) |src| {
            if (!physical.hasAxis(src)) return error.InvalidPhysicalAxis;
        }

        try putFoldSlice(allocator, &folds, target, sources);
    }

    // Ensure every axis has a fold list (identity if missing).
    for (axis_order) |tag| {
        if (!folds.contains(tag)) {
            var one = [_]PhysicalAxisTag{tag};
            try putFoldSlice(allocator, &folds, tag, one[0..]);
        }
    }

    return folds;
}

/// Build default folds (no explicit rules).
/// - Each used axis folds to itself.
/// - All unused axes are folded into a single “best‑fit” target axis.
fn buildFoldsDefault(
    allocator: std.mem.Allocator,
    physical: PhysicalMesh,
    logical: LogicalMesh,
    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag),
    axis_used: std.EnumSet(PhysicalAxisTag),
    axis_order: []const PhysicalAxisTag,
) !std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag) {
    var folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag) = .{};
    errdefer folds.deinit(allocator);

    // Start with identity folds for used axes.
    for (axis_order) |tag| {
        if (axis_used.contains(tag)) {
            var one = [_]PhysicalAxisTag{tag};
            try putFoldSlice(allocator, &folds, tag, one[0..]);
        }
    }

    const fold_target = selectFoldTarget(logical, bindings, physical) orelse axis_order[0];

    // Collect unused axes (excluding fold_target).
    var unused = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
    for (axis_order) |tag| {
        if (!axis_used.contains(tag) and tag != fold_target) {
            unused.appendAssumeCapacity(tag);
        }
    }

    // Ensure target has a fold list, then append unused axes.
    const existing = folds.get(fold_target) orelse blk: {
        var one = [_]PhysicalAxisTag{fold_target};
        try putFoldSlice(allocator, &folds, fold_target, one[0..]);
        break :blk folds.get(fold_target).?;
    };

    if (unused.len > 0) {
        // Append unused axes to the target fold list (avoid duplicates).
        var extra_count: usize = 0;
        for (unused.constSlice()) |tag| {
            if (!containsAxis(existing, tag)) extra_count += 1;
        }

        if (extra_count > 0) {
            const new_len = existing.len + extra_count;
            var buf = try allocator.alloc(PhysicalAxisTag, new_len);
            @memcpy(buf[0..existing.len], existing);

            var idx: usize = existing.len;
            for (unused.constSlice()) |tag| {
                if (!containsAxis(existing, tag)) {
                    buf[idx] = tag;
                    idx += 1;
                }
            }

            allocator.free(existing);
            try folds.put(allocator, fold_target, buf);
        }
    }

    return folds;
}

/// Replace or insert a fold slice, owning the memory.
fn putFoldSlice(
    allocator: std.mem.Allocator,
    folds: *std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag),
    target: PhysicalAxisTag,
    sources: []const PhysicalAxisTag,
) !void {
    const dup = try allocator.dupe(PhysicalAxisTag, sources);
    if (folds.get(target)) |old| {
        allocator.free(old);
    }
    try folds.put(allocator, target, dup);
}

fn containsAxis(list: []const PhysicalAxisTag, tag: PhysicalAxisTag) bool {
    for (list) |t| if (t == tag) return true;
    return false;
}

fn selectFoldTarget(
    logical: LogicalMesh,
    bindings: Sharding.Bindings,
    physical: PhysicalMesh,
) ?PhysicalAxisTag {
    var best_tag: ?PhysicalAxisTag = null;
    var best_priority: i32 = 999;
    var best_size: i64 = -1;

    for (logical.axes.constSlice(), logical.intents.constSlice()) |l_tag, l_intent| {
        const binding = bindings.get(l_tag) orelse continue;
        if (binding.len == 0) continue;
        const p_tag = binding[0];
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

/// Strategy is a compiled mapping plan:
/// logical axis → ordered physical axes.
/// It is built from logical intents + physical mesh hints.
///
/// `folding` lets you explicitly group/fold multiple physical axes into one
/// kept axis (used by mesh folding and axis view construction).
pub const Strategy = struct {
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: std.ArrayList(PhysicalAxisTag),
    };

    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, Binding),

    /// Explicit folding rules: kept axis -> ordered source axes.
    /// Example: fold link_x + link_z into link_x => { link_x: [link_x, link_z] }
    folding: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag),

    pub const init: Strategy = .{
        .bindings = .{},
        .folding = .{},
    };

    pub fn deinit(self: *Strategy, allocator: std.mem.Allocator) void {
        var it = self.bindings.iterator();
        while (it.next()) |entry| entry.value_ptr.physical.deinit(allocator);
        self.bindings.deinit(allocator);

        var fit = self.folding.iterator();
        while (fit.next()) |entry| allocator.free(entry.value_ptr.*);
        self.folding.deinit(allocator);
    }

    pub fn addBinding(self: *Strategy, allocator: std.mem.Allocator, logical: Shape.Tag, physical: PhysicalAxisTag) !void {
        var res = try self.bindings.getOrPut(allocator, logical);

        if (!res.found_existing) {
            res.value_ptr.* = .{ .logical = logical, .physical = .{} };
        }

        try res.value_ptr.physical.append(allocator, physical);
    }

    /// Explicitly fold axes: `target` is the kept axis, `sources` define order.
    /// If `target` is missing from `sources`, it is prepended.
    pub fn addFold(self: *Strategy, allocator: std.mem.Allocator, target: PhysicalAxisTag, sources: []const PhysicalAxisTag) !void {
        // Check if target is already present.
        var has_target = false;
        for (sources) |s| {
            if (s == target) {
                has_target = true;
                break;
            }
        }

        const new_len: usize = sources.len + @intFromBool(!has_target);
        var buf = try allocator.alloc(PhysicalAxisTag, new_len);

        if (!has_target) {
            buf[0] = target;
            @memcpy(buf[1..], sources);
        } else {
            @memcpy(buf, sources);
        }

        if (self.folding.get(target)) |old| {
            if (old.len == buf.len and std.mem.eql(PhysicalAxisTag, old, buf)) {
                allocator.free(buf);
                return;
            }
            allocator.free(old);
        }

        try self.folding.put(allocator, target, buf);
    }
};

/// suggestStrategy builds a Strategy from logical intents.
///
/// Specification:
/// - Produces a mapping: logical axis → **one** physical axis (at most).
/// - Selection is intent‑biased:
///   - high_bandwidth: prefer fastest physical axes (base order)
///   - balanced: prefer mid‑tier axes (base order rotated)
///   - low_bandwidth: prefer slowest axes (base order reversed)
/// - If none of the shardable axes exist in the mesh, fall back to
///   `physical.axisOrder()` so the strategy is never empty.
/// - No tensor sizes are considered.
/// - If more logical axes exist than available physical axes, axes are reused.
/// - A logical axis can remain unbound if the mesh has no axes at all.
pub fn suggestStrategy(
    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    physical: PhysicalMesh,
) !Strategy {
    var strategy: Strategy = .init;
    errdefer strategy.deinit(allocator);

    var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();
    const base_order = physical.shardableAxes();

    const pushIfExists = struct {
        fn f(physical_: PhysicalMesh, out_: *stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK), tag: PhysicalAxisTag) void {
            if (physical_.hasAxis(tag)) out_.appendAssumeCapacity(tag);
        }
    }.f;

    inline for (.{ .high_bandwidth, .balanced, .low_bandwidth }) |intent| {
        var preferred_axes = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;

        switch (intent) {
            .high_bandwidth => {
                for (base_order) |t| pushIfExists(physical, &preferred_axes, t);
            },
            .balanced => {
                if (base_order.len > 1) {
                    for (base_order[1..]) |t| pushIfExists(physical, &preferred_axes, t);
                    pushIfExists(physical, &preferred_axes, base_order[0]);
                } else if (base_order.len == 1) {
                    pushIfExists(physical, &preferred_axes, base_order[0]);
                }
            },
            .low_bandwidth => {
                var i = base_order.len;
                while (i > 0) : (i -= 1) {
                    pushIfExists(physical, &preferred_axes, base_order[i - 1]);
                }
            },
            else => unreachable,
        }

        if (preferred_axes.len == 0) {
            preferred_axes = physical.axisOrder();
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
        const view = sharding.axisView();
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

        const ordered_devices = try sharding.deviceOrder(sharding.allocator);
        defer sharding.allocator.free(ordered_devices);

        for (ordered_devices) |d| {
            const coords = d.coords orelse return error.MissingDeviceCoords;
            const idx = sharding.globalIndexFromCoords(coords);
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
        view: Sharding.AxisView,
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
        view: Sharding.AxisView,
        shard_idx: usize,
        used_axes: *std.EnumSet(PhysicalAxisTag),
    ) !AxisSplit {
        var plan: AxisSplit = try .init();

        for (binding) |p_tag| {
            for (view.axes.constSlice(), 0..) |axis, i| {
                if (axis.contains(p_tag) and !used_axes.contains(axis.tag)) {
                    const next_product = plan.product * axis.size;
                    if (@rem(dim, next_product) == 0) {
                        const coord = view.axisCoordFromLinearIndex(i, shard_idx);
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
