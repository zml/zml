const std = @import("std");

const c = @import("c");

// References:
// - NRT exposes `nrt_instance_info.family` and `.size`, which map to the
//   https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt.h
//   Neuron instance families and sizes documented here:
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
// The same instance-architecture pages are also the basis for:
// - `ChipTopology`, including the 2D torus and ring-over-torus fabrics.
// - `coresPerChip()`, using documented NeuronCore counts per chip generation:
//   Inferentia v1 = 4, Inferentia2 / Trainium v1 = 2, Trainium2 / Trainium3 = 8.
pub const Family = enum(u32) {
    unknown = 0,
    inf1 = 1,
    trn1 = 2,
    trn1n = 3,
    inf2 = 4,
    trn2 = 5,
    trn2n = 6,
    inf2e = 7,
    trn2p = 8,
    trn2u = 9,
    trn2e = 10,
    trn2eu = 11,
    trn2ac = 12,
    trn2uac = 13,
    trn3 = 14,
    _,
};

pub const Size = enum(u32) {
    xl1 = 0,
    xl2 = 1,
    xl4 = 2,
    xl6 = 3,
    xl8 = 4,
    xl24 = 5,
    xl32 = 6,
    xl48 = 7,
    xl3 = 8,
    unknown = 9,
    _,
};

pub const ChipTopology = union(enum) {
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

pub const TopologyInfo = struct {
    family: Family,
    size: Size,
    chips: usize,
    cores_per_chip: usize,
    chip_topology: ChipTopology,

    pub fn format(self: TopologyInfo, writer: *std.Io.Writer) !void {
        try writer.print(
            "TopologyInfo(family={s} size={s} chips={d} cores_per_chip={d} chip_topology=",
            .{
                @tagName(self.family),
                @tagName(self.size),
                self.chips,
                self.cores_per_chip,
            },
        );
        switch (self.chip_topology) {
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

pub fn instanceInfo() !struct { family: Family, size: Size } {
    var info = std.mem.zeroes(c.struct_nrt_instance_info);
    info.family = @intFromEnum(Family.unknown);
    info.size = @intFromEnum(Size.unknown);

    if (c.nrt_get_instance_info(&info, @sizeOf(c.struct_nrt_instance_info)) != c.NRT_SUCCESS) {
        return error.Unavailable;
    }

    return .{
        .family = @enumFromInt(info.family),
        .size = @enumFromInt(info.size),
    };
}

pub fn coresPerChip(family: Family) !usize {
    return switch (family) {
        .inf1 => 4,
        .trn1, .trn1n, .inf2, .inf2e => 2,
        .trn2, .trn2n, .trn2p, .trn2u, .trn2e, .trn2eu, .trn2ac, .trn2uac, .trn3 => 8,
        else => error.Unavailable,
    };
}

pub fn chipOf(nc_id: usize, physical_cores_per_chip: usize) usize {
    return @divFloor(nc_id, physical_cores_per_chip);
}

pub fn coreOf(nc_id: usize, physical_cores_per_chip: usize) usize {
    return @mod(nc_id, physical_cores_per_chip);
}

pub fn topologyFromSortedNcIds(sorted_nc_ids: []const usize) !TopologyInfo {
    const instance = try instanceInfo();
    const physical_cores_per_chip = try coresPerChip(instance.family);
    const first_chip = chipOf(sorted_nc_ids[0], physical_cores_per_chip);
    const last_chip = chipOf(sorted_nc_ids[sorted_nc_ids.len - 1], physical_cores_per_chip);
    const selected_chip_count = last_chip - first_chip + 1;
    const selected_cores_per_chip = std.math.divExact(usize, sorted_nc_ids.len, selected_chip_count) catch unreachable;

    return .{
        .family = instance.family,
        .size = instance.size,
        .chips = selected_chip_count,
        .cores_per_chip = selected_cores_per_chip,
        .chip_topology = try chipTopology(instance.family, instance.size, selected_chip_count),
    };
}

// References for chip-count and fabric-shape inference:
// - Inf1 instance sizes and 1/4/16-chip layouts
// - Inf2 instance sizes and 1/6/12-chip layouts
// - Trn1 / Trn1n 16-chip 2D torus
// - Trn2 16-chip torus and Trn2 UltraServer 64-chip ring-over-torus
// - Trn3 all-to-all / point-to-point scale-up
fn chipTopology(family: Family, size: Size, chip_count: usize) !ChipTopology {
    if (chip_count == 1) return .none;

    return switch (family) {
        .inf1 => switch (chip_count) {
            4 => .{ .linear = 4 },
            else => .{ .ring = chip_count },
        },
        .inf2, .inf2e => switch (chip_count) {
            6 => .{ .torus2d = .{ .x = 3, .y = 2 } },
            12 => .{ .torus2d = .{ .x = 4, .y = 3 } },
            else => switch (size) {
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
        .trn3 => .{ .point_to_point = chip_count },
        else => .{ .ring = chip_count },
    };
}
