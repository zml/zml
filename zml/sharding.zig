const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Target = @import("platform.zig").Target;

const log = std.log.scoped(.@"zml/sharding");

// mock
fn getDevices(nb: usize) []Device {
    var devices: [54]Device = undefined;

    for (devices[0..nb], 0..) |*d, i| {
        d.* = .{
            .id = i,
            .compute_units = 16,
            .pjrt_device = null,
        };
    }

    return devices[0..nb];
}

pub const Device = struct {
    /// Unique device identifier in the mesh
    id: usize,

    /// Coordinates in the physical mesh
    coords: ?[]const usize = null,

    /// Compute capacity local to this device (cores, SMs, etc.).
    compute_units: usize,

    /// Placeholder
    pjrt_device: ?*anyopaque = null,

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
    },
    /// A leaf represents the actual hardware device
    leaf: Device,

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
    pub const AxisInfo = struct {
        tag: PhysicalAxisTag,
        size: i64,
        geometry: AxisGeometry,
    };

    allocator: std.mem.Allocator,
    target: Target,
    root: PhysicalNode,

    /// Usage: PhysicalMesh.init(alloc, .tpu, .{ .link_x = 4, .link_y = 8, .link_z = 8 }, .{ .mesh = .torus })
    pub fn init(
        allocator: std.mem.Allocator,
        target: Target,
        topology: anytype,
        geometry_: AxisGeometry,
    ) !PhysicalMesh {
        const T = @TypeOf(topology);
        const info = @typeInfo(T).@"struct";

        var total_devices: usize = 1;
        inline for (info.fields) |field| {
            total_devices *= @field(topology, field.name);
        }

        const devices = try allocator.alloc(Device, total_devices);
        defer allocator.free(devices);

        // mock
        for (devices, 0..) |*d, i| d.* = .{ .id = i, .compute_units = 2 };

        var path = [_]usize{0} ** 16;
        const root = try buildFromFields(allocator, topology, info.fields, devices, geometry_, &path, 0);

        return .{
            .target = target,
            .root = root,
            .allocator = allocator,
        };
    }

    fn buildFromFields(
        allocator: std.mem.Allocator,
        topology: anytype,
        comptime fields: []const std.builtin.Type.StructField,
        devices: []const Device,
        geometry_: AxisGeometry,
        path: *[16]usize,
        depth: usize,
    ) !PhysicalNode {
        const current_field = fields[0];
        const count = @field(topology, current_field.name);
        const stride = devices.len / count;

        const children = try allocator.alloc(PhysicalNode, count);
        errdefer allocator.free(children);

        for (children, 0..) |*child, i| {
            const slice = devices[i * stride .. (i + 1) * stride];

            path[depth] = i;

            if (fields.len > 1) {
                child.* = try buildFromFields(allocator, topology, fields[1..], slice, geometry_, path, depth + 1);
            } else {
                var dev = slice[0];
                dev.coords = try allocator.dupe(usize, path[0 .. depth + 1]);
                child.* = .{ .leaf = dev };
            }
        }

        return .{
            .branch = .{
                .tag = std.meta.stringToEnum(PhysicalAxisTag, current_field.name).?,
                .geometry = geometry_,
                .children = children,
            },
        };
    }

    pub fn deinit(self: *PhysicalMesh) void {
        self.root.deinit(self.allocator);
    }

    pub fn countDevices(self: PhysicalMesh) usize {
        return self.root.countDevices();
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

    pub fn axisOrder(self: PhysicalMesh) stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK) {
        var out = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        axisOrderNode(self.root, &out);
        return out;
    }

    fn axisOrderNode(node: PhysicalNode, out: *stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK)) void {
        switch (node) {
            .leaf => {},
            .branch => |b| {
                out.appendAssumeCapacity(b.tag);
                if (b.children.len > 0) axisOrderNode(b.children[0], out);
            },
        }
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

    pub fn auto(allocator: std.mem.Allocator, platform: *const Platform) !PhysicalMesh {
        const root = try switch (platform.target) {
            .cpu => cpu(allocator, platform),
            .cuda, .rocm => gpu(allocator, platform),
            .tpu => tpu(allocator, platform),
            .neuron => neuron(allocator, platform),
        };

        return .{
            .allocator = allocator,
            .target = platform.target,
            .root = root,
        };
    }

    fn cpu(allocator: std.mem.Allocator, platform: *const Platform) !PhysicalNode {
        const devices = platform.devices;
        const nodes = try allocator.alloc(PhysicalNode, devices.len);

        for (nodes, devices) |*n, d| n.* = .{ .leaf = d };

        return .{
            .branch = .{
                .tag = .bus,
                .geometry = .tree,
                .children = nodes,
            },
        };
    }

    fn gpu(allocator: std.mem.Allocator, platform: *const Platform) !PhysicalNode {
        const devices = platform.devices;
        // todo: this is simplified I treat all GPUs as one P2P group for this example
        const nodes = try allocator.alloc(PhysicalNode, devices.len);

        for (nodes, devices) |*n, d| n.* = .{ .leaf = d };

        return .{
            .branch = .{
                .tag = .link,
                .geometry = .point_to_point,
                .children = nodes,
            },
        };
    }

    fn tpu(allocator: std.mem.Allocator, _: *const Platform) !PhysicalNode {
        const devices = getDevices()[0..8];
        // Example: TPU v3-8 is 2x2x2

        var z_branches = try allocator.alloc(PhysicalNode, 4);

        for (z_branches, 0..) |*zb, i| {
            const z_leaves = try allocator.alloc(PhysicalNode, 2);
            z_leaves[0] = .{ .leaf = devices[i * 2] };
            z_leaves[1] = .{ .leaf = devices[i * 2 + 1] };
            zb.* = .{ .branch = .{ .tag = .link_z, .geometry = .{ .mesh = .torus }, .children = z_leaves } };
        }

        var y_branches = try allocator.alloc(PhysicalNode, 2);

        y_branches[0] = .{ .branch = .{ .tag = .link_y, .geometry = .{ .mesh = .torus }, .children = z_branches[0..2] } };
        y_branches[1] = .{ .branch = .{ .tag = .link_y, .geometry = .{ .mesh = .torus }, .children = z_branches[2..4] } };

        return .{
            .branch = .{
                .tag = .link_x,
                .geometry = .{ .mesh = .torus },
                .children = y_branches,
            },
        };
    }

    fn neuron(allocator: std.mem.Allocator, _: *const Platform) !PhysicalNode {
        const devices = getDevices()[0..24];

        if (devices.len == 24) { // AWS Inf2.48xlarge: 2 islands of 12 chips
            const islands = try allocator.alloc(PhysicalNode, 2);

            for (islands, 0..) |*island, i| {
                const start = i * 12;
                const ring_nodes = try allocator.alloc(PhysicalNode, 12);
                for (ring_nodes, 0..) |*rn, j| rn.* = .{ .leaf = devices[start + j] };

                island.* = .{ .branch = .{ .tag = .link, .geometry = .{ .ring = .closed_ring }, .children = ring_nodes } };
            }

            return .{ .branch = .{ .tag = .bus, .geometry = .isolated, .children = islands } };
        }

        // Default NeuronLink
        const nodes = try allocator.alloc(PhysicalNode, devices.len);
        defer allocator.free(nodes);
        for (nodes, devices) |*n, d| n.* = .{ .leaf = d };

        return .{ .branch = .{ .tag = .link, .geometry = .{ .ring = .closed_ring }, .children = try allocator.dupe(PhysicalNode, nodes) } };
    }

    pub fn format(self: *const PhysicalMesh, writer: *std.Io.Writer) !void {
        try writer.print("PhysicalMesh({s})\n", .{@tagName(self.target)});
        // try writer.print("├── Total Devices: {d}\n", .{self.countDevices()});
        try writer.print("└── Total Devices: {d}\n", .{self.countDevices()});
        try writer.writeAll("│\n");

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
    name: []const u8,
    axes: stdx.BoundedArray(Shape.Tag, Shape.MAX_RANK),
    intents: stdx.BoundedArray(LogicalAxisIntent, Shape.MAX_RANK),

    pub fn init(name: []const u8, axes_: anytype) LogicalMesh {
        const T = @TypeOf(axes_);
        // if (!stdx.meta.isStruct(T)) {
        //     stdx.debug.compileError("LogicalMesh.init expects a struct of axis intents, got {}", .{T});
        // }

        var tags = stdx.BoundedArray(Shape.Tag, Shape.MAX_RANK).init(0) catch unreachable;
        var intents = stdx.BoundedArray(LogicalAxisIntent, Shape.MAX_RANK).init(0) catch unreachable;

        inline for (std.meta.fields(T)) |field| {
            const value = @field(axes_, field.name);
            tags.appendAssumeCapacity(Shape.toTag(field));
            intents.appendAssumeCapacity(intentFromValue(value));
        }

        if (tags.len == 0) {
            stdx.debug.panic("LogicalMesh must have at least one axis defined", .{});
        }

        return .{
            .name = name,
            .axes = tags,
            .intents = intents,
        };
    }

    fn intentFromValue(v: anytype) LogicalAxisIntent {
        const V = @TypeOf(v);
        if (V == LogicalAxisIntent) return v;

        if (V == @EnumLiteral()) {
            return @field(LogicalAxisIntent, @tagName(v));
        }

        stdx.debug.compileError("LogicalMesh intent must be LogicalAxisIntent, got {}", .{V});
    }

    pub fn intent(self: LogicalMesh, tag: Shape.Tag) ?LogicalAxisIntent {
        for (self.axes.constSlice(), self.intents.constSlice()) |t, i| {
            if (std.mem.eql(u8, std.mem.span(t), std.mem.span(tag))) return i;
        }
        return null;
    }

    pub fn format(self: LogicalMesh, writer: *std.Io.Writer) !void {
        try writer.print("LogicalMesh(name='{s}'):\n", .{self.name});
        for (self.axes.constSlice(), self.intents.constSlice()) |t, i| {
            try writer.print("  - {s}: {s}\n", .{ t, @tagName(i) });
        }
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
    pub const PhysicalSlice = struct {
        tag: PhysicalAxisTag,
        start: i64,
        size: i64,
        geometry: ?AxisGeometry,
    };

    pub const Tile = struct {
        physical: std.ArrayListUnmanaged(PhysicalSlice),
        logical_axes: std.ArrayListUnmanaged(Shape.Tag),
        logical_demand: i64,
        physical_capacity: i64,
        virtual_factor: i64,
    };

    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    physical: PhysicalMesh,

    tiles: std.ArrayList(Tile),

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

        for (self.tiles.items) |*tile| {
            tile.physical.deinit(self.allocator);
            tile.logical_axes.deinit(self.allocator);
        }
        self.tiles.deinit(self.allocator);
    }

    pub fn numPartitions(self: Sharding) i32 {
        if (self.tiles.items.len == 0) return 1;
        return @intCast(self.tiles.items[0].physical_capacity);
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

        try out.writer.writeAll("#sdy.mesh<[");
        if (self.tiles.items.len > 0) {
            const tile = self.tiles.items[0];
            for (tile.physical.items, 0..) |p, i| {
                if (i > 0) try out.writer.writeAll(", ");
                try out.writer.print("\"{s}\"={d}", .{ @tagName(p.tag), p.size });
            }
        }
        try out.writer.writeAll("]>");

        return try out.toOwnedSlice();
    }

    /// Build a sharding attribute string for a given shape.
    /// Returns null when all dims are unknown/open (fully open tensor).
    pub fn shardingAttrForShape(self: Sharding, allocator: std.mem.Allocator, shape: Shape) !?[]const u8 {
        // Determine if this tensor has any explicit sharding for this mesh.
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

        var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();
        var has_replicated_dim = false;

        // per-dimension shardings
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

                    if (self.tiles.items.len > 0) {
                        const tile = self.tiles.items[0];
                        for (binding_) |p_tag| {
                            for (tile.physical.items) |p| {
                                if (p.tag == p_tag and !used_axes.contains(p.tag)) {
                                    if (!wrote_any) {
                                        try out.writer.writeAll("{");
                                        wrote_any = true;
                                    }
                                    if (!first) try out.writer.writeAll(", ");
                                    try out.writer.print("\"{s}\"", .{@tagName(p.tag)});
                                    used_axes.insert(p.tag);
                                    first = false;
                                    break;
                                }
                            }
                        }
                    }

                    if (!wrote_any) {
                        try out.writer.writeAll("{?}");
                    } else {
                        try out.writer.writeAll("}");
                    }
                },
                .replicated => {
                    has_replicated_dim = true;
                    try out.writer.writeAll("{}");
                },
                .open => try out.writer.writeAll("{?}"),
                .unknown => try out.writer.writeAll("{?}"),
            }
        }

        try out.writer.writeAll("]");

        // explicit replicated axes: only those not already used
        var replicated_axes = std.EnumSet(PhysicalAxisTag).initEmpty();
        if (has_replicated_dim) {
            if (self.tiles.items.len > 0) {
                const tile = self.tiles.items[0];
                for (tile.physical.items) |p| {
                    if (!used_axes.contains(p.tag)) {
                        replicated_axes.insert(p.tag);
                    }
                }
            }
        }

        if (replicated_axes.count() > 0) {
            try out.writer.writeAll(", replicated={");
            var first = true;

            if (self.tiles.items.len > 0) {
                const tile = self.tiles.items[0];
                for (tile.physical.items) |p| {
                    if (replicated_axes.contains(p.tag)) {
                        if (!first) try out.writer.writeAll(", ");
                        try out.writer.print("\"{s}\"", .{@tagName(p.tag)});
                        first = false;
                    }
                }
            }

            try out.writer.writeAll("}");
        }

        try out.writer.writeAll(">");

        return try out.toOwnedSlice();
    }

    pub fn deviceAssignment(self: Sharding, allocator: std.mem.Allocator) ![]usize {
        if (self.tiles.items.len == 0) {
            const ids = try allocator.alloc(usize, 1);
            ids[0] = 0;
            return ids;
        }

        const tile = self.tiles.items[0];
        const count: usize = @intCast(tile.physical_capacity);

        var ids = try allocator.alloc(usize, count);
        @memset(ids, std.math.maxInt(usize));

        const ordered_devices = try tileDeviceOrder(allocator, self);
        defer allocator.free(ordered_devices);

        const tag_to_depth = buildTagToDepth(self);

        if (tile.physical.items.len == 0) {
            if (ordered_devices.len == 0) return error.MissingDeviceInTile;
            ids[0] = ordered_devices[0].id;
            return ids;
        }

        for (ordered_devices) |d| {
            const coords = d.coords orelse return error.MissingDeviceCoords;
            const idx_and_coords = try shardIndexAndCoords(allocator, self, tile, tag_to_depth, coords);
            defer allocator.free(idx_and_coords.tile_coords);
            ids[idx_and_coords.idx] = d.id;
        }

        for (ids) |id| {
            if (id == std.math.maxInt(usize)) return error.MissingDeviceInTile;
        }

        return ids;
    }

    pub fn format(self: Sharding, writer: *std.Io.Writer) !void {
        try writer.print("Sharding(name={s}, tiles={d})\n", .{ self.logical.name, self.tiles.items.len });

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

        if (self.tiles.items.len == 0) return;

        for (self.tiles.items, 0..) |tile, i| {
            try writer.print("Tile #{d}:\n", .{i});

            try writer.writeAll("  Logical: ");
            if (tile.logical_axes.items.len == 0) {
                try writer.writeAll("(none)\n");
            } else {
                for (tile.logical_axes.items, 0..) |l, j| {
                    if (j > 0) try writer.writeAll(" ⊗ ");
                    try writer.writeAll(std.mem.span(l));
                }
                try writer.writeAll("\n");
            }

            try writer.print("  Physical capacity: {d}\n", .{tile.physical_capacity});
            try writer.print("  Virtual factor   : {d}x\n", .{tile.virtual_factor});

            try writer.writeAll("  Physical axes: ");
            if (tile.physical.items.len == 0) {
                try writer.writeAll("(none)\n");
            } else {
                for (tile.physical.items, 0..) |p, j| {
                    if (j > 0) try writer.writeAll(" × ");
                    try writer.print("{s}[{d}]", .{ @tagName(p.tag), p.size });
                }
                try writer.writeAll("\n");
            }

            try writer.writeAll("  Geometries: ");
            if (tile.physical.items.len == 0) {
                try writer.writeAll("(none)\n");
            } else {
                for (tile.physical.items, 0..) |p, j| {
                    if (j > 0) try writer.writeAll(", ");
                    try writer.print("{s}={s}", .{
                        @tagName(p.tag),
                        @tagName(p.geometry orelse .point_to_point),
                    });
                }
                try writer.writeAll("\n");
            }
        }
    }
};

/// resolveStrategyConstraints builds a Sharding without sizes.
///
/// Logic:
/// - Take Strategy bindings.
/// - Filter to physical axes that exist.
/// - Build a tile that spans the full size of those axes.
///
/// This gives a global plan that says:
/// "the program uses these axes of the mesh."
///
/// Limitations:
/// - No divisibility checks.
/// - No replication/virtualization computed here.
pub fn resolveStrategyConstraints(
    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    physical: PhysicalMesh,
    strategy: Strategy,
) !Sharding {
    var tiles: std.ArrayList(Sharding.Tile) = .{};
    errdefer {
        for (tiles.items) |*t| {
            t.physical.deinit(allocator);
            t.logical_axes.deinit(allocator);
        }
        tiles.deinit(allocator);
    }

    var bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag) = .{};
    errdefer {
        var it = bindings.iterator();
        while (it.next()) |entry| allocator.free(entry.value_ptr.*);
        bindings.deinit(allocator);
    }

    var axis_used = std.EnumSet(PhysicalAxisTag).initEmpty();

    var tile: Sharding.Tile = .{
        .physical = .{},
        .logical_axes = .{},
        .logical_demand = 1,
        .physical_capacity = 1,
        .virtual_factor = 1,
    };

    var seen_logical = std.AutoArrayHashMap(Shape.Tag, void).init(allocator);
    defer seen_logical.deinit();

    var it = strategy.bindings.iterator();
    while (it.next()) |bind| {
        const l_tag = bind.key_ptr.*;

        if (!seen_logical.contains(l_tag)) {
            try tile.logical_axes.append(allocator, l_tag);
            try seen_logical.put(l_tag, {});
        }

        // Filter to axes that exist on this physical mesh
        var filtered: std.ArrayList(PhysicalAxisTag) = .{};
        defer filtered.deinit(allocator);

        for (bind.value_ptr.physical.items) |p_tag| {
            if (physical.hasAxis(p_tag)) {
                try filtered.append(allocator, p_tag);
                axis_used.insert(p_tag);
            }
        }

        const slice = try allocator.dupe(PhysicalAxisTag, filtered.items);
        try bindings.put(allocator, l_tag, slice);
    }

    var fold_lists: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, std.ArrayListUnmanaged(PhysicalAxisTag)) = .{};
    defer {
        var it_lists = fold_lists.iterator();
        while (it_lists.next()) |entry| entry.value_ptr.deinit(allocator);
        fold_lists.deinit(allocator);
    }

    const order = physical.shardableAxes();

    if (strategy.folding.count() > 0) {
        var used = std.EnumSet(PhysicalAxisTag).initEmpty();
        var fit = strategy.folding.iterator();
        while (fit.next()) |entry| {
            if (!physical.hasAxis(entry.key_ptr.*)) return error.InvalidPhysicalAxis;

            var res = try fold_lists.getOrPut(allocator, entry.key_ptr.*);
            if (!res.found_existing) res.value_ptr.* = .{};
            for (entry.value_ptr.*) |src| {
                if (!physical.hasAxis(src)) return error.InvalidPhysicalAxis;
                try res.value_ptr.append(allocator, src);
                used.insert(src);
            }
        }
        for (order) |p| {
            if (!physical.hasAxis(p)) continue;
            if (!used.contains(p)) {
                var res = try fold_lists.getOrPut(allocator, p);
                if (!res.found_existing) res.value_ptr.* = .{};
                try res.value_ptr.append(allocator, p);
            }
        }
    } else {
        for (order) |p| {
            if (!physical.hasAxis(p)) continue;
            if (axis_used.contains(p)) {
                var res = try fold_lists.getOrPut(allocator, p);
                if (!res.found_existing) res.value_ptr.* = .{};
                try res.value_ptr.append(allocator, p);
            }
        }

        var fallback: ?PhysicalAxisTag = null;
        for (order) |p| {
            if (physical.hasAxis(p)) {
                fallback = p;
                break;
            }
        }
        if (fold_lists.count() == 0 and fallback != null) {
            var res = try fold_lists.getOrPut(allocator, fallback.?);
            if (!res.found_existing) res.value_ptr.* = .{};
            try res.value_ptr.append(allocator, fallback.?);
        }

        const fold_target = selectFoldTarget(logical, bindings, physical) orelse fallback.?;
        var target_entry = try fold_lists.getOrPut(allocator, fold_target);
        if (!target_entry.found_existing) target_entry.value_ptr.* = .{};
        if (target_entry.value_ptr.items.len == 0) {
            try target_entry.value_ptr.append(allocator, fold_target);
        }

        for (order) |p| {
            if (!physical.hasAxis(p)) continue;
            if (!axis_used.contains(p)) {
                try target_entry.value_ptr.append(allocator, p);
            }
        }
    }

    var folds: std.AutoArrayHashMapUnmanaged(PhysicalAxisTag, []const PhysicalAxisTag) = .{};
    errdefer {
        var fit2 = folds.iterator();
        while (fit2.next()) |entry| allocator.free(entry.value_ptr.*);
        folds.deinit(allocator);
    }

    var it_lists = fold_lists.iterator();
    while (it_lists.next()) |entry| {
        const dup = try allocator.dupe(PhysicalAxisTag, entry.value_ptr.items);
        try folds.put(allocator, entry.key_ptr.*, dup);
    }

    // Build tile axes from folded groups
    for (order) |p_tag| {
        const sources = folds.get(p_tag) orelse continue;

        var size: i64 = 1;
        for (sources) |src| {
            size *= physical.axis(src);
        }

        tile.physical_capacity *= size;

        try tile.physical.append(allocator, .{
            .tag = p_tag,
            .start = 0,
            .size = size,
            .geometry = physical.geometry(p_tag),
        });
    }

    try tiles.append(allocator, tile);

    return .{
        .allocator = allocator,
        .logical = logical,
        .physical = physical,
        .tiles = tiles,
        .bindings = bindings,
        .folds = folds,
    };
}

fn intentPriority(intent: LogicalAxisIntent) i32 {
    return switch (intent) {
        .high_bandwidth => 0,
        .balanced => 1,
        .low_bandwidth => 2,
    };
}

fn selectFoldTarget(
    logical: LogicalMesh,
    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, []const PhysicalAxisTag),
    physical: PhysicalMesh,
) ?PhysicalAxisTag {
    var best_tag: ?PhysicalAxisTag = null;
    var best_pri: i32 = 999;
    var best_size: i64 = -1;

    for (logical.axes.constSlice(), logical.intents.constSlice()) |l_tag, l_intent| {
        const binding = bindings.get(l_tag) orelse continue;
        if (binding.len == 0) continue;
        const p_tag = binding[0];
        if (!physical.hasAxis(p_tag)) continue;

        const pri = intentPriority(l_intent);
        const size = physical.axis(p_tag);

        if (pri < best_pri or (pri == best_pri and size > best_size)) {
            best_pri = pri;
            best_size = size;
            best_tag = p_tag;
        }
    }

    return best_tag;
}

fn largestFactorLE(n: i64, limit: i64) i64 {
    var f = if (limit < n) limit else n;

    while (f > 1) : (f -= 1) {
        if (@rem(n, f) == 0) return f;
    }

    return 1;
}

/// Split a logical axis size across multiple physical axes.
/// Logic:
/// - walk physical axes in binding order
/// - greedily take a factor that divides remaining size
/// - leftover becomes virtualization
/// Limitation: errors if a physical axis does not divide remaining size.
fn factorizeLogicalAxis(
    logical_size: i64,
    physical: PhysicalMesh,
    bindings: []const PhysicalAxisTag,
    axis_extents: *std.EnumArray(PhysicalAxisTag, i64),
    virtual_factor: *i64,
) !void {
    var remaining = logical_size;

    for (bindings) |p_tag| {
        const p_cap = physical.axis(p_tag);
        if (p_cap <= 0) return error.InvalidPhysicalAxis;

        if (remaining <= p_cap) {
            const extent = remaining;
            axis_extents.getPtr(p_tag).* *= extent;
            remaining = 1;
            break;
        }

        const extent = largestFactorLE(remaining, p_cap);
        if (extent == 1) {
            if (@rem(remaining, p_cap) != 0) return error.NonFactorableAxis;
            axis_extents.getPtr(p_tag).* *= p_cap;
            remaining = @divExact(remaining, p_cap);
        } else {
            axis_extents.getPtr(p_tag).* *= extent;
            remaining = @divExact(remaining, extent);
        }
    }

    if (remaining > 1) {
        virtual_factor.* *= remaining;
    }
}

fn tagsEqual(a: Shape.Tag, b: Shape.Tag) bool {
    return std.mem.eql(u8, std.mem.span(a), std.mem.span(b));
}

fn logicalDimFromShape(shape: Shape, logical_tag: Shape.Tag) ?i64 {
    var total: i64 = 1;
    var found = false;
    const rk = shape.rank();

    var ax: usize = 0;
    while (ax < rk) : (ax += 1) {
        const dim = shape.dim(ax);
        const tag = shape.tag(ax);

        if (tag != Shape.TagUnknown and tagsEqual(tag, logical_tag)) {
            total *= dim;
            found = true;
            continue;
        }

        switch (shape.partition(ax)) {
            .axis => |t| {
                if (tagsEqual(t, logical_tag)) {
                    total *= dim;
                    found = true;
                }
            },
            else => {},
        }
    }

    return if (found) total else null;
}

pub fn resolveStrategy(
    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    logical_shape: Shape,
    physical: PhysicalMesh,
    strategy: Strategy,
) !Sharding {
    var tiles: std.ArrayList(Sharding.Tile) = .{};
    errdefer {
        for (tiles.items) |*t| {
            t.physical.deinit(allocator);
            t.logical_axes.deinit(allocator);
        }
        tiles.deinit(allocator);
    }

    var axis_extents = std.EnumArray(PhysicalAxisTag, i64).initFill(1);
    var axis_used = std.EnumSet(PhysicalAxisTag).initEmpty();

    var tile: Sharding.Tile = .{
        .physical = .{},
        .logical_axes = .{},
        .logical_demand = 1,
        .physical_capacity = 1,
        .virtual_factor = 1,
    };

    var seen_logical = std.AutoArrayHashMap(Shape.Tag, void).init(allocator);
    defer seen_logical.deinit();

    var it = strategy.bindings.iterator();
    while (it.next()) |bind| {
        const l_tag = bind.key_ptr.*;

        const l_dim = logicalDimFromShape(logical_shape, l_tag) orelse return error.MissingLogicalAxisSize;

        if (!seen_logical.contains(l_tag)) {
            try tile.logical_axes.append(allocator, l_tag);
            try seen_logical.put(l_tag, {});
            tile.logical_demand *= l_dim;
        }

        try factorizeLogicalAxis(
            l_dim,
            physical,
            bind.value_ptr.physical.items,
            &axis_extents,
            &tile.virtual_factor,
        );
    }

    const order = physical.shardableAxes();
    for (order) |p_tag| {
        const used = axis_extents.get(p_tag);
        if (used <= 1) continue;

        const cap = physical.axis(p_tag);
        if (used > cap) return error.ExtentOverflow;

        axis_used.insert(p_tag);
        tile.physical_capacity *= used;

        try tile.physical.append(allocator, .{
            .tag = p_tag,
            .start = 0,
            .size = used,
            .geometry = physical.geometry(p_tag),
        });
    }

    try tiles.append(allocator, tile);

    return .{
        .allocator = allocator,
        .logical = logical,
        .physical = physical,
        .tiles = tiles,
    };
}

/// Strategy is a compiled mapping plan:
/// logical axis → ordered physical axes.
///
/// This is built from intents + physical mesh.
/// No tensor sizes are used here.
///
/// Think of it as: "if you shard logical axis X,
/// use physical axes in this order."
pub const Strategy = struct {
    pub const Binding = struct {
        logical: Shape.Tag,
        physical: std.ArrayList(PhysicalAxisTag),
    };

    bindings: std.AutoArrayHashMapUnmanaged(Shape.Tag, Binding),

    /// Optional explicit folding: target axis -> ordered source axes.
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
            res.value_ptr.* = .{
                .logical = logical,
                .physical = .{},
            };
        }

        try res.value_ptr.physical.append(allocator, physical);
    }

    /// Explicitly fold axes: `target` is the kept axis, `sources` define order.
    /// `sources` must include `target`; if missing, it will be prepended.
    pub fn addFold(self: *Strategy, allocator: std.mem.Allocator, target: PhysicalAxisTag, sources: []const PhysicalAxisTag) !void {
        var list = std.ArrayList(PhysicalAxisTag).init(allocator);
        defer list.deinit();

        var has_target = false;
        for (sources) |s| {
            if (s == target) has_target = true;
            try list.append(s);
        }
        if (!has_target) {
            try list.insert(0, target);
        }

        const dup = try allocator.dupe(PhysicalAxisTag, list.items);

        if (self.folding.get(target)) |old| {
            allocator.free(old);
        }
        try self.folding.put(allocator, target, dup);
    }
};

/// suggestStrategy builds a Strategy from intents.
///
/// Logic:
/// - High bandwidth axes get fastest physical axes first.
/// - Balanced axes get mid‑tier axes first.
/// - Low bandwidth axes get slower axes first.
///
/// It only binds axes that exist on the mesh.
///
/// Limitations:
/// - No tensor sizes.
/// - Multiple logical axes may map to the same physical axis.
pub fn suggestStrategy(
    allocator: std.mem.Allocator,
    logical: LogicalMesh,
    physical: PhysicalMesh,
) !Strategy {
    var strategy: Strategy = .init;
    errdefer strategy.deinit(allocator);

    var used_axes = std.EnumSet(PhysicalAxisTag).initEmpty();

    inline for (.{ .high_bandwidth, .balanced, .low_bandwidth }) |target_intent| {
        var order_buf = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
        const order = shardableAxesByIntent(physical, target_intent, &order_buf);

        for (logical.axes.constSlice(), logical.intents.constSlice()) |l_tag, l_intent| {
            if (l_intent != target_intent) continue;
            if (order.len == 0) continue;

            var selected: ?PhysicalAxisTag = null;
            for (order) |p_tag| {
                if (!used_axes.contains(p_tag)) {
                    selected = p_tag;
                    used_axes.insert(p_tag);
                    break;
                }
            }
            if (selected == null) {
                selected = order[0]; // reuse if we ran out
            }

            try strategy.addBinding(allocator, l_tag, selected.?);
        }
    }

    return strategy;
}

fn shardableAxesByIntent(
    physical: PhysicalMesh,
    intent: LogicalAxisIntent,
    out: *stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK),
) []const PhysicalAxisTag {
    out.* = stdx.BoundedArray(PhysicalAxisTag, Shape.MAX_RANK).init(0) catch unreachable;
    const base = physical.shardableAxes();

    switch (intent) {
        .high_bandwidth => {
            for (base) |t| {
                if (physical.hasAxis(t)) out.appendAssumeCapacity(t);
            }
        },
        .balanced => {
            if (base.len > 1) {
                for (base[1..]) |t| {
                    if (physical.hasAxis(t)) out.appendAssumeCapacity(t);
                }
                if (physical.hasAxis(base[0])) out.appendAssumeCapacity(base[0]);
            } else if (base.len == 1 and physical.hasAxis(base[0])) {
                out.appendAssumeCapacity(base[0]);
            }
        },
        .low_bandwidth => {
            var i = base.len;
            while (i > 0) : (i -= 1) {
                const t = base[i - 1];
                if (physical.hasAxis(t)) out.appendAssumeCapacity(t);
            }
        },
    }

    return out.constSlice();
}

/// ShardAssignment is the *per‑tensor* concrete plan.
///
/// It answers:
/// - which devices get data
/// - which slice of the tensor goes to each device
///
/// This is derived from:
/// Sharding (global plan) + tensor partition specs.
pub const ShardAssignment = struct {
    pub const Slice1d = struct {
        axis: u8,
        start: i64,
        size: i64,
    };

    pub const Shard = struct {
        device_id: usize,
        device_coords: []const usize,
        tile_coords: []usize,
        slices: std.ArrayListUnmanaged(Slice1d),
    };

    allocator: std.mem.Allocator,
    axes: std.ArrayListUnmanaged(Sharding.PhysicalSlice),
    shards: []Shard,

    pub fn deinit(self: *ShardAssignment) void {
        for (self.shards) |*s| {
            s.slices.deinit(self.allocator);
            if (s.tile_coords.len > 0) {
                self.allocator.free(s.tile_coords);
            }
        }
        self.allocator.free(self.shards);
        self.axes.deinit(self.allocator);
    }

    pub fn format(self: ShardAssignment, shape: Shape, writer: *std.Io.Writer) !void {
        try writer.print("ShardAssignment(shards={d})\n", .{self.shards.len});

        try writer.writeAll("Tensor axis summary:\n");
        for (0..shape.rank()) |ax| {
            const dim = shape.dim(ax);
            const tag = shape.tag(ax);
            const spec = shape.partition(ax);

            var shard_size: i64 = dim;
            if (self.shards.len > 0) {
                for (self.shards[0].slices.items) |s| {
                    if (s.axis == ax) {
                        shard_size = s.size;
                        break;
                    }
                }
            }

            const replicated = (shard_size == dim);

            var shard_count: ?i64 = null;
            if (shard_size > 0 and @rem(dim, shard_size) == 0) {
                shard_count = @divExact(dim, shard_size);
            }

            if (tag != Shape.TagUnknown) {
                try writer.print("  - {s}: dim={d}, shard={d}, shards=", .{ tag, dim, shard_size });
            } else {
                try writer.print("  - axis {d}: dim={d}, shard={d}, shards=", .{ ax, dim, shard_size });
            }

            if (shard_count) |c| {
                try writer.print("{d}", .{c});
            } else {
                try writer.writeAll("?");
            }

            try writer.print(", {s}, spec={s}\n", .{
                if (replicated) "replicated" else "sharded",
                @tagName(spec),
            });
        }

        try writer.writeAll("Per-shard placement:\n");
        for (self.shards, 0..) |shard, i| {
            try writer.print("└─ Shard[{d}]\n", .{i});
            try writer.print("   ├─ device_id: {d}\n", .{shard.device_id});
            try writer.print("   ├─ device_coords: {any}\n", .{shard.device_coords});
            try writer.print("   ├─ tile_coords: {any}\n", .{shard.tile_coords});
            try writer.writeAll("   └─ slices:\n");

            for (shard.slices.items) |s| {
                const tag = shape.tag(s.axis);
                if (tag != Shape.TagUnknown) {
                    try writer.print("       - {s}: [{d}:{d}] (size={d})\n", .{ tag, s.start, s.start + s.size, s.size });
                } else {
                    try writer.print("       - axis {d}: [{d}:{d}] (size={d})\n", .{ s.axis, s.start, s.start + s.size, s.size });
                }
            }
        }
    }
};

fn collectDevices(allocator: std.mem.Allocator, node: PhysicalNode, list: *std.ArrayList(Device)) !void {
    switch (node) {
        .leaf => |d| try list.append(allocator, d),
        .branch => |b| {
            for (b.children) |child| {
                try collectDevices(allocator, child, list);
            }
        },
    }
}

/// Precompute axis → depth index for device coords.
/// Logic:
/// - uses PhysicalMesh.axisOrder() (root→leaf).
fn buildTagToDepth(sharding: Sharding) std.EnumArray(PhysicalAxisTag, i8) {
    const axis_order = sharding.physical.axisOrder();
    var tag_to_depth = std.EnumArray(PhysicalAxisTag, i8).initFill(-1);
    for (axis_order.constSlice(), 0..) |t, i| {
        tag_to_depth.set(t, @intCast(i));
    }
    return tag_to_depth;
}

/// Returns true if device coords fall inside the tile.
/// Logic:
/// - for each tile axis, check coord in [start, start+size).
fn isDeviceInTile(sharding: Sharding, coords: []const usize, tile: Sharding.Tile, tag_to_depth: std.EnumArray(PhysicalAxisTag, i8)) bool {
    for (tile.physical.items) |p| {
        const depth = tag_to_depth.get(p.tag);
        if (depth < 0) return false;

        const coord = foldedCoord(sharding, p.tag, coords, tag_to_depth);
        if (coord < p.start or coord >= p.start + p.size) return false;
    }
    return true;
}

/// Returns devices sorted by tile coordinate order.
/// Logic:
/// - filter devices to tile
/// - sort by axis order: first axis changes slowest
/// Used by assignTensor and deviceAssignment for consistency.
fn tileDeviceOrder(allocator: std.mem.Allocator, sharding: Sharding) ![]Device {
    if (sharding.tiles.items.len == 0) return error.MissingTile;
    const tile = sharding.tiles.items[0];
    const tag_to_depth = buildTagToDepth(sharding);

    var devices: std.ArrayList(Device) = .{};
    defer devices.deinit(allocator);
    try collectDevices(allocator, sharding.physical.root, &devices);

    var filtered: std.ArrayList(Device) = .{};
    defer filtered.deinit(allocator);

    for (devices.items) |d| {
        const coords = d.coords orelse return error.MissingDeviceCoords;
        if (isDeviceInTile(sharding, coords, tile, tag_to_depth)) {
            try filtered.append(allocator, d);
        }
    }

    const TileOrder = struct {
        sharding: Sharding,
        tag_to_depth: std.EnumArray(PhysicalAxisTag, i8),

        fn lessThan(ctx: @This(), a: Device, b: Device) bool {
            const ca = a.coords.?;
            const cb = b.coords.?;
            const ia = globalIndexFromCoords(ctx.sharding, ca, ctx.tag_to_depth);
            const ib = globalIndexFromCoords(ctx.sharding, cb, ctx.tag_to_depth);
            return ia < ib;
        }
    };

    std.mem.sort(Device, filtered.items, TileOrder{
        .sharding = sharding,
        .tag_to_depth = tag_to_depth,
    }, TileOrder.lessThan);

    return try filtered.toOwnedSlice(allocator);
}

fn foldedCoord(
    sharding: Sharding,
    tag: PhysicalAxisTag,
    coords: []const usize,
    tag_to_depth: std.EnumArray(PhysicalAxisTag, i8),
) usize {
    const sources = sharding.folds.get(tag) orelse unreachable;
    var idx: usize = 0;
    for (sources) |src| {
        const depth = tag_to_depth.get(src);
        const coord = coords[@intCast(depth)];
        const size = sharding.physical.axis(src);
        idx = idx * @as(usize, @intCast(size)) + coord;
    }
    return idx;
}

fn globalIndexFromCoords(
    sharding: Sharding,
    coords: []const usize,
    tag_to_depth: std.EnumArray(PhysicalAxisTag, i8),
) usize {
    const order = sharding.physical.axisOrder();
    var idx: usize = 0;
    for (order.constSlice()) |tag| {
        const depth = tag_to_depth.get(tag);
        if (depth < 0) continue;
        const coord = coords[@intCast(depth)];
        const size = sharding.physical.axis(tag);
        idx = idx * @as(usize, @intCast(size)) + coord;
    }
    return idx;
}

/// Convert device coords into tile‑local coords + linear shard index.
/// Logic:
/// - map each tile axis coord to 0..size-1
/// - linear index = row‑major over tile axes
fn shardIndexAndCoords(
    allocator: std.mem.Allocator,
    sharding: Sharding,
    tile: Sharding.Tile,
    tag_to_depth: std.EnumArray(PhysicalAxisTag, i8),
    coords: []const usize,
) !struct { idx: usize, tile_coords: []usize } {
    const axes_len = tile.physical.items.len;
    var tile_coords = try allocator.alloc(usize, axes_len);

    for (tile.physical.items, 0..) |p, i| {
        const coord = foldedCoord(sharding, p.tag, coords, tag_to_depth);
        tile_coords[i] = coord;
    }

    const idx = globalIndexFromCoords(sharding, coords, tag_to_depth);

    return .{ .idx = idx, .tile_coords = tile_coords };
}

/// Select physical axes for one tensor axis.
/// Logic:
/// - walk binding order
/// - include an axis only if it keeps divisibility
/// - returns product of used axes
/// This allows partial usage when full usage is not divisible.
fn selectPhysicalAxesForDim(
    allocator: std.mem.Allocator,
    dim: i64,
    binding: []const PhysicalAxisTag,
    tile: Sharding.Tile,
    shard: ShardAssignment.Shard,
    counts: *std.ArrayListUnmanaged(i64),
    indices: *std.ArrayListUnmanaged(i64),
) !i64 {
    var product: i64 = 1;

    for (binding) |p_tag| {
        for (tile.physical.items, 0..) |p, i| {
            if (p.tag == p_tag) {
                const next_product = product * p.size;
                if (@rem(dim, next_product) == 0) {
                    try counts.append(allocator, p.size);
                    try indices.append(allocator, @intCast(shard.tile_coords[i]));
                    product = next_product;
                }
                break;
            }
        }
    }

    return product;
}

pub const TransferIterator = struct {
    assignment: ShardAssignment,
    index: usize = 0,

    pub fn next(self: *TransferIterator) ?ShardAssignment.Shard {
        defer self.index += 1;

        if (self.index >= self.assignment.shards.len) return null;

        return self.assignment.shards[self.index];
    }

    pub fn deinit(self: *TransferIterator) void {
        self.assignment.deinit();
    }
};

/// transferIterator is a small wrapper for data transfer.
///
/// It yields shard entries (device + slices) for a shape.
/// This is what Buffer uses to know where to send data.
pub fn transferIterator(sharding: Sharding, shape: Shape) !TransferIterator {
    const assignment = try assignTensor(sharding, shape);
    return .{ .assignment = assignment };
}

/// assignTensor builds a ShardAssignment for one tensor.
///
/// Logic (high level):
/// 1) Build the list of devices in tile order.
/// 2) For each device, compute its tile coords.
/// 3) For each tensor axis:
///    - if partitioned on logical axis:
///      follow logical→physical binding order,
///      take only physical axes that keep divisibility,
///      compute slice start/size.
///    - if replicated: slice = full axis.
///
/// Limitations:
/// - If a logical axis is unbound, this errors.
/// - If an axis cannot be divided by the chosen physical axes,
///   it will skip non‑divisible axes (partial usage).
pub fn assignTensor(
    sharding: Sharding,
    shape: Shape,
) !ShardAssignment {
    const allocator = sharding.allocator;

    if (sharding.tiles.items.len == 0) return error.MissingTile;
    const tile = sharding.tiles.items[0];

    var axes = std.ArrayListUnmanaged(Sharding.PhysicalSlice){};
    errdefer axes.deinit(allocator);
    try axes.appendSlice(allocator, tile.physical.items);

    const shards_len: usize = @intCast(tile.physical_capacity);
    var shards = try allocator.alloc(ShardAssignment.Shard, shards_len);
    errdefer {
        for (shards) |*s| {
            s.slices.deinit(allocator);
            if (s.tile_coords.len > 0) allocator.free(s.tile_coords);
        }
        allocator.free(shards);
    }

    @memset(shards, .{
        .device_id = 0,
        .device_coords = &.{},
        .tile_coords = &.{},
        .slices = .{},
    });

    const ordered_devices = try tileDeviceOrder(allocator, sharding);
    defer allocator.free(ordered_devices);

    const tag_to_depth = buildTagToDepth(sharding);
    const axes_len = tile.physical.items.len;

    if (axes_len == 0) {
        if (ordered_devices.len == 0) return error.MissingDeviceCoords;
        const d = ordered_devices[0];
        shards[0].device_id = d.id;
        shards[0].device_coords = d.coords orelse return error.MissingDeviceCoords;
        shards[0].tile_coords = &.{};
    } else {
        for (ordered_devices) |d| {
            const coords = d.coords orelse return error.MissingDeviceCoords;
            const idx_and_coords = try shardIndexAndCoords(allocator, sharding, tile, tag_to_depth, coords);

            shards[idx_and_coords.idx].device_id = d.id;
            shards[idx_and_coords.idx].device_coords = coords;
            shards[idx_and_coords.idx].tile_coords = idx_and_coords.tile_coords;
        }
    }

    for (shards) |*s| {
        var slices = std.ArrayListUnmanaged(ShardAssignment.Slice1d){};

        for (0..shape.rank()) |ax| {
            const dim = shape.dim(ax);
            const spec = shape.partition(ax);

            var start: i64 = 0;
            var size: i64 = dim;

            switch (spec) {
                .axis => |logical_tag| {
                    const binding = sharding.binding(logical_tag) orelse return error.MissingLogicalBinding;

                    var counts = std.ArrayListUnmanaged(i64){};
                    defer counts.deinit(allocator);

                    var indices = std.ArrayListUnmanaged(i64){};
                    defer indices.deinit(allocator);

                    const product = try selectPhysicalAxesForDim(allocator, dim, binding, tile, s.*, &counts, &indices);

                    size = @divExact(dim, product);

                    var linear: i64 = 0;
                    for (counts.items, indices.items) |c, iidx| {
                        linear = linear * c + iidx;
                    }

                    start = linear * size;
                },
                else => {},
            }

            try slices.append(allocator, .{
                .axis = @intCast(ax),
                .start = start,
                .size = size,
            });
        }

        s.slices = slices;
    }

    return .{
        .allocator = allocator,
        .axes = axes,
        .shards = shards,
    };
}

test "tile sharding virtualization uses tensor shape sizes" {
    const allocator = std.testing.allocator;

    var physical: PhysicalMesh = try .init(allocator, .cuda, .{ .link = 8 }, .point_to_point);
    defer physical.deinit();

    const logical = LogicalMesh.init("dp_virtualized", .{ .batch = .high_bandwidth });

    const tensor_shape = Shape.init(.{ .my_tensor_dim = 16 }, .u8).withPartitioning(.{ .my_tensor_dim = .batch });

    var strategy = try suggestStrategy(allocator, logical, physical);
    defer strategy.deinit(allocator);

    var sharding = try resolveStrategy(allocator, logical, tensor_shape, physical, strategy);
    defer sharding.deinit();

    try std.testing.expectEqual(1, sharding.tiles.items.len);

    const tile = sharding.tiles.items[0];
    try std.testing.expectEqual(16, tile.logical_demand);
    try std.testing.expectEqual(8, tile.physical_capacity);
    try std.testing.expectEqual(2, tile.virtual_factor);

    try std.testing.expectEqual(1, tile.physical.items.len);
    try std.testing.expectEqual(8, tile.physical.items[0].size);
}
