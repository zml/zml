const std = @import("std");

const neuron = @import("platforms/neuron");

// References:
// - `neuron.Instance` exposes the NRT instance family and size used below.
// - Neuron instance families and sizes are documented here:
//   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/index.html
// - Inf1:
//   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inf1-arch.html
// - Inf2:
//   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inf2-arch.html
// - Trn1 / Trn1n:
//   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn1-arch.html
// - Trn2 / Trn2 UltraServer:
//   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html
// - Trn3:
//   https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html
//
// The same instance-architecture pages are also the basis for the inter-chip
// topology inference below, including 2D torus and ring-over-torus fabrics.
pub const InterChipTopology = union(enum) {
    none: void,
    linear: usize,
    ring: usize,
    torus2d: struct {
        x: usize,
        y: usize,
    },
    torus2d_ring: struct {
        x: usize,
        y: usize,
        z: usize,
    },
    point_to_point: usize,
};

pub const Axis = struct {
    fabric: Fabric,
    size: usize,
    geometry: Geometry,
};

pub const Fabric = enum {
    core,
    chip_x,
    chip_y,
    chip_z,
};

pub const Geometry = enum {
    linear_ring,
    closed_ring,
    torus,
    point_to_point,
};

pub const Placement = struct {
    input_index: usize,
    nc_id: usize,
    chip: usize,
    core: usize,
};

pub const VisibleMesh = struct {
    instance: neuron.Instance,
    chips: usize,
    cores_per_chip: usize,
    inter_chip_topology: InterChipTopology,
    placements: []Placement,
    axes_buffer: [4]Axis,
    axes_len: usize,

    pub fn deinit(self: *VisibleMesh, allocator: std.mem.Allocator) void {
        allocator.free(self.placements);
        self.* = undefined;
    }

    pub fn axes(self: *const VisibleMesh) []const Axis {
        return self.axes_buffer[0..self.axes_len];
    }

    pub fn format(self: VisibleMesh, writer: *std.Io.Writer) !void {
        try writer.print(
            "VisibleMesh(family={s} size={s} chips={d} cores_per_chip={d} inter_chip_topology=",
            .{
                @tagName(self.instance.family),
                @tagName(self.instance.size),
                self.chips,
                self.cores_per_chip,
            },
        );
        switch (self.inter_chip_topology) {
            .none => try writer.writeAll("none"),
            .linear => |size| try writer.print("linear({d})", .{size}),
            .ring => |size| try writer.print("ring({d})", .{size}),
            .torus2d => |dims| try writer.print("torus2d({d}x{d})", .{ dims.x, dims.y }),
            .torus2d_ring => |dims| try writer.print("torus2d_ring({d}x{d}x{d})", .{ dims.x, dims.y, dims.z }),
            .point_to_point => |size| try writer.print("point_to_point({d})", .{size}),
        }
        try writer.writeAll(")");
    }
};

pub fn visibleMeshFromNcIds(allocator: std.mem.Allocator, nc_ids: []const usize) !VisibleMesh {
    if (nc_ids.len == 0) return error.InvalidDeviceTopology;

    const instance = try neuron.instance();
    const physical_cores_per_chip = try instance.coresPerChip();
    const placements = try allocator.alloc(Placement, nc_ids.len);
    errdefer allocator.free(placements);

    for (nc_ids, placements, 0..) |nc_id, *placement, input_index| {
        placement.* = .{
            .input_index = input_index,
            .nc_id = nc_id,
            .chip = chipOf(nc_id, physical_cores_per_chip),
            .core = coreOf(nc_id, physical_cores_per_chip),
        };
    }

    const Sort = struct {
        fn lessThan(_: void, a: Placement, b: Placement) bool {
            return if (a.chip == b.chip) a.core < b.core else a.chip < b.chip;
        }
    };
    std.mem.sort(Placement, placements, {}, Sort.lessThan);

    const first_chip = placements[0].chip;
    const last_chip = placements[placements.len - 1].chip;
    const selected_chip_count = last_chip - first_chip + 1;
    const selected_cores_per_chip = try std.math.divExact(usize, placements.len, selected_chip_count);
    const inter_chip_topology = interChipTopology(instance, selected_chip_count);

    var res: VisibleMesh = .{
        .instance = instance,
        .chips = selected_chip_count,
        .cores_per_chip = selected_cores_per_chip,
        .inter_chip_topology = inter_chip_topology,
        .placements = placements,
        .axes_buffer = undefined,
        .axes_len = 0,
    };

    switch (inter_chip_topology) {
        .none => {},
        .linear => |size| appendAxis(&res, .chip_x, size, .linear_ring),
        .ring => |size| appendAxis(&res, .chip_x, size, .closed_ring),
        .torus2d => |dims| {
            appendAxis(&res, .chip_x, dims.x, .torus);
            appendAxis(&res, .chip_y, dims.y, .torus);
        },
        .torus2d_ring => |dims| {
            appendAxis(&res, .chip_x, dims.x, .torus);
            appendAxis(&res, .chip_y, dims.y, .torus);
            appendAxis(&res, .chip_z, dims.z, .closed_ring);
        },
        .point_to_point => |size| appendAxis(&res, .chip_x, size, .point_to_point),
    }

    // Inter-chip topology only describes NeuronLink fabric between chips. Add a
    // local core axis for multi-core chips, and keep single-chip meshes non-empty.
    if (res.cores_per_chip > 1 or res.axes_len == 0) {
        appendAxis(&res, .core, res.cores_per_chip, .point_to_point);
    }

    return res;
}

pub fn interChipTopology(instance: neuron.Instance, chip_count: usize) InterChipTopology {
    if (chip_count == 1) return .none;

    // References for chip-count and fabric-shape inference:
    // - Inf1 instance sizes and 1/4/16-chip layouts
    // - Inf2 instance sizes and 1/6/12-chip layouts
    // - Trn1 / Trn1n 16-chip 2D torus
    // - Trn2 16-chip torus and Trn2 UltraServer 64-chip ring-over-torus
    // - Trn3 all-to-all / point-to-point scale-up
    return switch (instance.family) {
        .inf1 => switch (chip_count) {
            4 => .{ .linear = 4 },
            else => .{ .ring = chip_count },
        },
        .inf2, .inf2e => switch (chip_count) {
            6 => .{ .torus2d = .{ .x = 3, .y = 2 } },
            12 => .{ .torus2d = .{ .x = 4, .y = 3 } },
            else => switch (instance.size) {
                .xl24 => .{ .torus2d = .{ .x = 3, .y = 2 } },
                .xl48 => .{ .torus2d = .{ .x = 4, .y = 3 } },
                else => .{ .ring = chip_count },
            },
        },
        .trn1, .trn1n => switch (chip_count) {
            16 => .{ .torus2d = .{ .x = 4, .y = 4 } },
            else => .{ .ring = chip_count },
        },
        .trn2, .trn2n, .trn2p, .trn2u, .trn2e, .trn2eu, .trn2ac, .trn2uac => switch (chip_count) {
            16 => .{ .torus2d = .{ .x = 4, .y = 4 } },
            64 => .{ .torus2d_ring = .{ .x = 4, .y = 4, .z = 4 } },
            else => .{ .ring = chip_count },
        },
        .trn3, .trn3pds98 => .{ .point_to_point = chip_count },
        .unknown => .{ .ring = chip_count },
    };
}

pub fn chipOf(nc_id: usize, physical_cores_per_chip: usize) usize {
    return @divFloor(nc_id, physical_cores_per_chip);
}

pub fn coreOf(nc_id: usize, physical_cores_per_chip: usize) usize {
    return @mod(nc_id, physical_cores_per_chip);
}

fn appendAxis(self: *VisibleMesh, fabric: Fabric, size: usize, geometry: Geometry) void {
    self.axes_buffer[self.axes_len] = .{
        .fabric = fabric,
        .size = size,
        .geometry = geometry,
    };
    self.axes_len += 1;
}
