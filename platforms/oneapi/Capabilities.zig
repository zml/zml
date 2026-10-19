//! Intel GPU hardware facts for kernels compiled for the local devices.
//!
//! Compute.supports() selects native instruction families; hardware.limits
//! describes subgroup, SLM and register constraints without choosing a tile,
//! subgroup size or register mode for the caller.
//!
//! zml/platform.zig caches this information in Device.capabilities and merges
//! devices into Platform.capabilities. Use the device view for device-bound
//! kernels and the platform view for a configuration shared by those devices.
//! A null oneAPI capability means the identity or common target is unresolved.
//!
//!     const caps: Capabilities = .init(.pvc, .{});
//!     const useMatrix = caps.compute.supports(.dpas);
//!     const widths = caps.hardware.limits.subgroupSizes;
//!
//! In the SPMD mapping, a subgroup occupies one hardware thread, not one SIMD
//! lane. Explicit SIMD backends must account for their own execution mapping.
//! Kernel backends must validate instruction shapes, operand layouts, compiler
//! support and compiled-kernel resource limits before launching a configuration.
//!
//! This model contains no cache geometry, bandwidth/latency estimates, atomic
//! capabilities, matrix shapes or compiled-kernel resource measurements.
//! Compute and the resource fields do not uniquely identify a physical device
//! or provide a performance-cache key. See KERNEL_TUNING.md for tuning workflow.
const std = @import("std");

const KiB = 1024;
const Capabilities = @This();

compute: Compute,
hardware: Hardware,

/// Builds limits from a known compute identity and restricts them with reported
/// properties. Does not query hardware; production callers use cached discovery.
pub fn init(compute: Compute, properties: Hardware.Properties) Capabilities {
    return .{
        .compute = compute,
        .hardware = .{
            .limits = properties.restrict(compute.limits()),
            .properties = properties,
        },
    };
}

/// Conservative view for a configuration shared across devices. Returns null
/// if either input is unknown or their compute identities differ.
/// Core count and memory capacity use the minimum when both are reported,
/// otherwise remain unknown. A reported SLM restriction survives an unreported
/// one; two reported restrictions use the minimum. Limits are then recomputed.
pub fn merge(lhs_: ?Capabilities, rhs_: ?Capabilities) ?Capabilities {
    const lhs = lhs_ orelse return null;
    const rhs = rhs_ orelse return null;
    if (lhs.compute != rhs.compute) return null;
    return .init(lhs.compute, lhs.hardware.properties.merge(rhs.hardware.properties));
}

pub const Hardware = struct {
    limits: Limits,
    properties: Properties,

    /// Compute-table geometry, with per-work-group SLM restricted by Properties.
    /// These are inputs to candidate filtering, not a complete launch contract:
    /// per-dimension limits and compiled-kernel constraints belong to the backend.
    /// Memory sizes are bytes; per-core values are independent of device counts.
    pub const Limits = struct {
        /// Hardware choices, not the width selected for a compiled kernel.
        /// The algorithm and backend may support only a subset.
        subgroupSizes: []const u32,
        /// Physical vector-engine width, not a preferred or required subgroup size.
        physicalSimdWidth: u32,
        /// Multiply by a register mode's thread slots for a per-core thread bound.
        vectorEnginesPerCore: u32,
        /// Total work-items per group. Compiler and register mode can lower this.
        maxWorkgroupSize: u32 = 1024,
        /// Schedulable SLM, separate from the combined L1/SLM cache capacity.
        /// Null when the compute target covers different product/stepping sizes.
        /// The per-work-group ceiling remains usable when this is unknown.
        sharedMemoryPerCore: ?u32,
        /// Ceiling for all simultaneously live allocations in one work-group,
        /// already clamped by Properties.sharedMemoryPerWorkgroupLimit in init.
        sharedMemoryPerWorkgroup: u32,
        /// Bytes per GRF. A mode's GRFs belong to the whole hardware thread
        /// (subgroup in SPMD), not to each work-item.
        grfBytes: u32,
        /// Available hardware modes; ordering does not designate a compiler default.
        registerModes: []const RegisterMode,

        pub fn supportsSubgroupSize(self: Limits, size: u32) bool {
            return std.mem.indexOfScalar(u32, self.subgroupSizes, size) != null;
        }
    };

    /// Hardware allocation modes, not a recommendation or a compiler default.
    /// Choosing a mode requires corresponding support in the kernel compiler.
    pub const RegisterMode = struct {
        /// GRFs per hardware thread; multiply by Limits.grfBytes for storage.
        grfCount: u32,
        /// Thread slots for this mode, not measured occupancy. Allocation and
        /// other resources can further restrict residency.
        maxThreadsPerVectorEngine: u32,
    };

    /// Properties supplied by PJRT for the visible device/partition.
    /// Null means unreported; product totals must not be inferred from Compute.
    /// zml/platform.zig ignores zero, negative, wrongly typed and out-of-range
    /// PJRT values. This path performs no separate oneAPI driver query.
    pub const Properties = struct {
        /// PJRT core_count: XLA's slices * subslices per slice, not an EU or lane
        /// count. Describes the visible device, not necessarily the entire card.
        coreCount: ?u32 = null,
        /// PJRT shared_memory_per_block_optin, in bytes. Null leaves the compute
        /// ceiling intact. Consumers use Limits.sharedMemoryPerWorkgroup, which
        /// already incorporates this restriction.
        sharedMemoryPerWorkgroupLimit: ?u32 = null,
        /// PJRT device_memory_bytes_limit, in bytes. Capacity, not currently free
        /// memory; allocation can still fail after checking this value.
        globalMemoryBytes: ?u64 = null,

        fn merge(self: Properties, other: Properties) Properties {
            return .{
                .coreCount = if (self.coreCount != null and other.coreCount != null) @min(self.coreCount.?, other.coreCount.?) else null,
                .globalMemoryBytes = if (self.globalMemoryBytes != null and other.globalMemoryBytes != null) @min(self.globalMemoryBytes.?, other.globalMemoryBytes.?) else null,
                // An unreported restriction retains the architectural ceiling.
                .sharedMemoryPerWorkgroupLimit = if (self.sharedMemoryPerWorkgroupLimit != null and other.sharedMemoryPerWorkgroupLimit != null)
                    @min(self.sharedMemoryPerWorkgroupLimit.?, other.sharedMemoryPerWorkgroupLimit.?)
                else
                    self.sharedMemoryPerWorkgroupLimit orelse other.sharedMemoryPerWorkgroupLimit,
            };
        }

        fn restrict(self: Properties, architectureLimits: Limits) Limits {
            var result = architectureLimits;
            if (self.sharedMemoryPerWorkgroupLimit) |bytes| result.sharedMemoryPerWorkgroup = @min(result.sharedMemoryPerWorkgroup, bytes);
            return result;
        }
    };
};

/// Native operations, not emulated datatype support or throughput guarantees.
/// Backend/toolchain support is separate. Tensor features describe operand types;
/// shapes, packing, subgroup widths and accumulator types remain the backend's
/// responsibility. These flags do not specify numerical policy or atomic support.
pub const Feature = enum {
    /// Native FP64 arithmetic; does not imply FP64 atomics.
    fp64,
    /// Native conversions; does not imply BF16 matrix support or a rounding mode.
    bf16Conversion,
    /// XMX matrix instruction family; validate the exact operation in the backend.
    dpas,
    /// Split DPAS with operands shared between paired vector engines.
    dpasw,
    tensorFp16,
    tensorBf16,
    tensorInt8,
    /// LSC two-dimensional block loads/stores (including load transforms).
    /// Alignment, pitch, shape and transform constraints require backend checks;
    /// this flag does not describe asynchronous-copy or prefetch support.
    block2dLoadStore,
};

/// Hardware target identity, not a device identifier or an ordered feature level.
/// Sources for OneAPI compute capabilities, product families, and
/// representative devices:
/// - https://github.com/openxla/xla/blob/main/xla/stream_executor/sycl/oneapi_compute_capability.h
/// - https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl
/// - https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/product_config_base.inl
pub const Compute = enum {
    /// Xe HPG: Arc A-series, Arc Pro A-series, Data Center GPU Flex.
    dg2,
    /// Xe HPC: Data Center GPU Max.
    pvc,
    /// Xe HPC: Data Center GPU Max 1550VG; no XMX.
    pvc_vg,
    /// Xe2 HPG: Arc B570/B580 and Arc Pro B-series.
    bmg,
    /// Xe LPG: Meteor Lake and Arrow Lake U/S; no XMX.
    mtl,
    /// Xe LPG+: Arrow Lake H Arc 130T/140T; has XMX.
    arl_h,
    /// Xe2 LPG: Lunar Lake Arc 130V/140V.
    lnl,
    /// Xe3 LPG: Panther Lake Arc B370/B390.
    ptl,
    /// Xe3 LPG: Wildcat Lake Core Series 3 integrated graphics.
    wcl,

    pub const Architecture = enum {
        xe_hpg,
        xe_hpc,
        xe2_hpg,
        xe_lpg,
        xe_lpg_plus,
        xe2_lpg,
        xe3_lpg,
    };

    pub fn architecture(self: Compute) Architecture {
        return switch (self) {
            .dg2 => .xe_hpg,
            .pvc, .pvc_vg => .xe_hpc,
            .bmg => .xe2_hpg,
            .mtl => .xe_lpg,
            .arl_h => .xe_lpg_plus,
            .lnl => .xe2_lpg,
            .ptl, .wcl => .xe3_lpg,
        };
    }

    /// Intel compute-runtime's release capability tables distinguish MTL from
    /// ARL-H and PVC from PVC-VG. Do not inherit features by generation order.
    /// https://github.com/intel/compute-runtime/tree/master/shared/source/release_helpers/caps
    /// FP64 follows the product hw_info tables; DG2's FP64 is emulated.
    pub fn supports(self: Compute, feature: Feature) bool {
        return switch (feature) {
            .fp64 => switch (self) {
                .dg2 => false,
                .pvc, .pvc_vg, .mtl, .arl_h, .bmg, .lnl, .ptl, .wcl => true,
            },
            .bf16Conversion => true,
            .dpas, .tensorFp16, .tensorBf16, .tensorInt8 => switch (self) {
                .mtl, .pvc_vg => false,
                .dg2, .pvc, .arl_h, .bmg, .lnl, .ptl, .wcl => true,
            },
            .dpasw => self == .dg2,
            .block2dLoadStore => switch (self) {
                .dg2, .mtl, .arl_h => false,
                .pvc, .pvc_vg, .bmg, .lnl, .ptl, .wcl => true,
            },
        };
    }

    /// Canonical hardware names and explicit compiler product aliases.
    /// Bare ARL is ambiguous: U/S use MTL graphics, H has XMX.
    pub fn parse(text: []const u8) ?Compute {
        inline for (std.meta.tags(Compute)) |arch| {
            if (std.ascii.eqlIgnoreCase(text, @tagName(arch))) return arch;
        }
        const aliases = .{
            .{ "PVC-VG", Compute.pvc_vg },
            .{ "DG2-G10", Compute.dg2 },
            .{ "DG2-G11", Compute.dg2 },
            .{ "DG2-G12", Compute.dg2 },
            .{ "BMG-G21", Compute.bmg },
            .{ "BMG-G31", Compute.bmg },
            .{ "MTL-H", Compute.mtl },
            .{ "MTL-U", Compute.mtl },
            .{ "ARL-U", Compute.mtl },
            .{ "ARL-S", Compute.mtl },
            .{ "ARL-H", Compute.arl_h },
            .{ "LNL-M", Compute.lnl },
            .{ "PTL-H", Compute.ptl },
            .{ "PTL-U", Compute.ptl },
        };
        inline for (aliases) |alias| {
            if (std.ascii.eqlIgnoreCase(text, alias[0])) return alias[1];
        }
        return null;
    }

    /// Level Zero IP version: architecture in bits 31:22, release in bits 21:14.
    /// Ignore revision only; an unknown release must not inherit capabilities.
    /// https://github.com/intel/compute-runtime/blob/master/third_party/aot_config_headers/platforms.h
    pub fn fromIpVersion(ip: u32) ?Compute {
        return switch (ip >> 14) {
            (12 << 8) | 55, (12 << 8) | 56, (12 << 8) | 57 => .dg2,
            (12 << 8) | 60 => .pvc,
            (12 << 8) | 61 => .pvc_vg,
            (12 << 8) | 70, (12 << 8) | 71 => .mtl,
            (12 << 8) | 74 => .arl_h,
            (20 << 8) | 1, (20 << 8) | 2 => .bmg,
            (20 << 8) | 4 => .lnl,
            (30 << 8) | 0, (30 << 8) | 1 => .ptl,
            (30 << 8) | 3 => .wcl,
            else => null,
        };
    }

    /// Intel PCI device IDs, not PJRT logical or local hardware ordinals.
    /// Product groups from compute-runtime's device_ids_configs headers.
    pub fn fromDeviceId(id: u32) ?Compute {
        return switch (id) {
            0x4F80, 0x4F81, 0x4F82, 0x4F83, 0x4F84, 0x5690, 0x5691, 0x5692, 0x56A0, 0x56A1, 0x56A2, 0x56C0, 0x56C2, 0x56BE, 0x56BF, 0x56AF, 0x4F87, 0x4F88, 0x5693, 0x5694, 0x5695, 0x56A5, 0x56A6, 0x56B0, 0x56B1, 0x56BA, 0x56BB, 0x56BC, 0x56BD, 0x56C1, 0x5696, 0x5697, 0x56A3, 0x56A4, 0x56B2, 0x56B3, 0x4F85, 0x4F86 => .dg2,
            0x0BD0, 0x0BD5, 0x0BD6, 0x0BD7, 0x0BD8, 0x0BD9, 0x0BDA, 0x0BDB, 0x0B69, 0x0B6E => .pvc,
            0x0BD4 => .pvc_vg,
            0xE202, 0xE209, 0xE20B, 0xE20C, 0xE20D, 0xE210, 0xE211, 0xE212, 0xE215, 0xE216, 0xE220, 0xE221, 0xE222, 0xE223 => .bmg,
            0x7D40, 0x7D45, 0x7D67, 0x7D41, 0x7D55, 0x7DD5 => .mtl,
            0x7D51, 0x7DD1 => .arl_h,
            0x6420, 0x64A0, 0x64B0 => .lnl,
            0xB080, 0xB081, 0xB082, 0xB083, 0xB084, 0xB085, 0xB086, 0xB087, 0xB08F, 0xB0A0, 0xB0B0, 0xB090 => .ptl,
            0xFD80, 0xFD81 => .wcl,
            else => null,
        };
    }

    /// Exact product names only: Arc B370/B390 are PTL, not BMG.
    /// Generic "Intel(R) Arc(TM) Graphics" is shared by multiple generations.
    pub fn parseDeviceKind(text: []const u8) ?Compute {
        const name = stripIntelBrand(text);
        const products = .{
            .{ "A770M", Compute.dg2 },
            .{ "A730M", Compute.dg2 },
            .{ "A550M", Compute.dg2 },
            .{ "A370M", Compute.dg2 },
            .{ "A350M", Compute.dg2 },
            .{ "A570M", Compute.dg2 },
            .{ "A530M", Compute.dg2 },
            .{ "Pro A30M", Compute.dg2 },
            .{ "Pro A40/A50", Compute.dg2 },
            .{ "Pro A40", Compute.dg2 },
            .{ "Pro A50", Compute.dg2 },
            .{ "Pro A60M", Compute.dg2 },
            .{ "Pro A60", Compute.dg2 },
            .{ "A380E", Compute.dg2 },
            .{ "A310E", Compute.dg2 },
            .{ "A370E", Compute.dg2 },
            .{ "A350E", Compute.dg2 },
            .{ "A750E", Compute.dg2 },
            .{ "A580E", Compute.dg2 },
            .{ "A770", Compute.dg2 },
            .{ "A750", Compute.dg2 },
            .{ "A580", Compute.dg2 },
            .{ "A380", Compute.dg2 },
            .{ "A310 LP", Compute.dg2 },
            .{ "A760A", Compute.dg2 },
            .{ "Data Center GPU Flex 170", Compute.dg2 },
            .{ "Data Center GPU Flex 140", Compute.dg2 },
            .{ "Data Center GPU Flex 170V", Compute.dg2 },
            .{ "Data Center GPU Max 1550", Compute.pvc },
            .{ "Data Center GPU Max 1350", Compute.pvc },
            .{ "Data Center GPU Max 1100", Compute.pvc },
            .{ "Data Center GPU Max 1450", Compute.pvc },
            .{ "Data Center GPU Max 1100C", Compute.pvc },
            .{ "Data Center GPU Max 1550VG", Compute.pvc_vg },
            .{ "B580", Compute.bmg },
            .{ "B570", Compute.bmg },
            .{ "Pro B60", Compute.bmg },
            .{ "Pro B50", Compute.bmg },
            .{ "Pro B65", Compute.bmg },
            .{ "Pro B70", Compute.bmg },
            .{ "B390", Compute.ptl },
            .{ "B370", Compute.ptl },
            .{ "130V", Compute.lnl },
            .{ "140V", Compute.lnl },
            .{ "130T", Compute.arl_h },
            .{ "140T", Compute.arl_h },
            .{ "Ponte Vecchio", Compute.pvc },
            .{ "Battlemage", Compute.bmg },
            .{ "BMG-G21", Compute.bmg },
            .{ "BMG-G31", Compute.bmg },
        };
        inline for (products) |product| {
            if (std.ascii.eqlIgnoreCase(name, product[0])) return product[1];
        }
        return null;
    }

    /// Subgroup and core geometry from Intel's occupancy guide:
    /// https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html
    /// Register modes/thread slots and SLM partitions from compute-runtime:
    /// shared/source/release_helpers/release_helper/ and
    /// shared/source/helpers/gfx_core_helper_{base,xe3_and_later}.inl.
    /// DG2's schedulable SLM varies (96/128 KiB), so leave it unknown.
    fn limits(self: Compute) Hardware.Limits {
        const simd8 = switch (self) {
            .dg2, .mtl, .arl_h => true,
            .pvc, .pvc_vg, .bmg, .lnl, .ptl, .wcl => false,
        };
        const registerModes: []const Hardware.RegisterMode = switch (self) {
            .dg2, .mtl, .arl_h, .pvc, .pvc_vg, .bmg, .lnl => &.{
                .{ .grfCount = 128, .maxThreadsPerVectorEngine = 8 },
                .{ .grfCount = 256, .maxThreadsPerVectorEngine = 4 },
            },
            .ptl, .wcl => &.{
                .{ .grfCount = 32, .maxThreadsPerVectorEngine = 10 },
                .{ .grfCount = 64, .maxThreadsPerVectorEngine = 10 },
                .{ .grfCount = 96, .maxThreadsPerVectorEngine = 10 },
                .{ .grfCount = 128, .maxThreadsPerVectorEngine = 8 },
                .{ .grfCount = 160, .maxThreadsPerVectorEngine = 6 },
                .{ .grfCount = 192, .maxThreadsPerVectorEngine = 5 },
                .{ .grfCount = 256, .maxThreadsPerVectorEngine = 4 },
            },
        };
        return .{
            .subgroupSizes = if (simd8) &.{ 8, 16, 32 } else &.{ 16, 32 },
            .physicalSimdWidth = if (simd8) 8 else 16,
            .vectorEnginesPerCore = if (simd8) 16 else 8,
            .sharedMemoryPerCore = switch (self) {
                .dg2 => null,
                .mtl => 96 * KiB,
                .arl_h, .pvc, .pvc_vg, .lnl => 128 * KiB,
                .bmg => 160 * KiB,
                .ptl, .wcl => 192 * KiB,
            },
            .sharedMemoryPerWorkgroup = if (simd8) 64 * KiB else 128 * KiB,
            .grfBytes = if (simd8) 32 else 64,
            .registerModes = registerModes,
        };
    }
};

fn stripIntelBrand(text: []const u8) []const u8 {
    var name = text;
    for ([_][]const u8{ "Intel(R) ", "Intel " }) |prefix| {
        if (std.ascii.startsWithIgnoreCase(name, prefix)) {
            name = name[prefix.len..];
            break;
        }
    }
    for ([_][]const u8{ " Graphics", " GPU" }) |suffix| {
        if (std.ascii.endsWithIgnoreCase(name, suffix)) {
            name = name[0 .. name.len - suffix.len];
            break;
        }
    }
    if (std.ascii.startsWithIgnoreCase(name, "Arc(TM) ")) return name[8..];
    if (std.ascii.startsWithIgnoreCase(name, "Arc ")) return name[4..];
    return name;
}

test "Intel product discovery distinguishes discrete and integrated graphics" {
    const cases = .{
        .{ "Intel(R) Arc(TM) Pro A60 Graphics", Compute.dg2 },
        .{ "Intel Arc Pro A40/A50", Compute.dg2 },
        .{ "Intel(R) Arc(TM) B390 GPU", Compute.ptl },
        .{ "Intel Arc B370", Compute.ptl },
        .{ "Intel(R) Arc(TM) B580 Graphics", Compute.bmg },
        .{ "Intel Arc Pro B70", Compute.bmg },
        .{ "Intel Arc 140V", Compute.lnl },
        .{ "Intel Arc 130T", Compute.arl_h },
        .{ "Intel Data Center GPU Max 1550VG", Compute.pvc_vg },
    };
    inline for (cases) |case| try std.testing.expectEqual(case[1], Compute.parseDeviceKind(case[0]));
    for ([_][]const u8{ "Intel(R) Arc(TM) Graphics", "Intel(R) Graphics", "Intel Arc B999 GPU", "Intel Arc Pro A999", "Intel Arc A770junk", "Intel Arc B580 engineering sample" }) |name| {
        try std.testing.expectEqual(null, Compute.parseDeviceKind(name));
    }
    try std.testing.expectEqual(null, Compute.parse("unknown"));
    try std.testing.expectEqual(null, Compute.parse("ARL"));
    try std.testing.expectEqual(Compute.mtl, Compute.parse("ARL-S"));
    try std.testing.expectEqual(Compute.arl_h, Compute.parse("ARL-H"));
    try std.testing.expectEqual(Compute.pvc_vg, Compute.parse("PVC-VG"));
}

test "Intel hardware identifiers preserve architecture and release boundaries" {
    const cases = .{
        .{ 0x56B3, 0x030dc008, Compute.dg2 },
        .{ 0x0BD5, 0x030f0007, Compute.pvc },
        .{ 0x0BD4, 0x030f4007, Compute.pvc_vg },
        .{ 0xE223, 0x05008000, Compute.bmg },
        .{ 0x7D55, 0x0311c004, Compute.mtl },
        .{ 0x7D51, 0x03128004, Compute.arl_h },
        .{ 0x64A0, 0x05010004, Compute.lnl },
        .{ 0xB080, 0x07800004, Compute.ptl },
        .{ 0xFD80, 0x0780c001, Compute.wcl },
    };
    inline for (cases) |case| {
        try std.testing.expectEqual(case[2], Compute.fromDeviceId(case[0]));
        try std.testing.expectEqual(case[2], Compute.fromIpVersion(case[1]));
    }
    try std.testing.expectEqual(Compute.mtl, Compute.fromDeviceId(0x7D67));
    try std.testing.expectEqual(null, Compute.fromDeviceId(0xB088));
    try std.testing.expectEqual(null, Compute.fromIpVersion(0x0500c000));
    try std.testing.expectEqual(null, Compute.fromIpVersion(0xffffffff));
}

test "Intel property merges preserve restrictions and unknown capacities" {
    const a: Capabilities = .init(.pvc, .{
        .coreCount = 64,
        .sharedMemoryPerWorkgroupLimit = 96 * KiB,
        .globalMemoryBytes = 64 * 1024 * 1024 * 1024,
    });
    const b: Capabilities = .init(.pvc, .{
        .coreCount = 32,
        .sharedMemoryPerWorkgroupLimit = 64 * KiB,
        .globalMemoryBytes = 32 * 1024 * 1024 * 1024,
    });
    const unknown: Capabilities = .init(.pvc, .{});
    const common = merge(a, b).?;
    try std.testing.expectEqual(32, common.hardware.properties.coreCount);
    try std.testing.expectEqual(32 * 1024 * 1024 * 1024, common.hardware.properties.globalMemoryBytes);
    try std.testing.expectEqual(64 * KiB, common.hardware.limits.sharedMemoryPerWorkgroup);
    try std.testing.expectEqualDeep(common, merge(b, a).?);
    try std.testing.expectEqualDeep(merge(merge(a, unknown), b), merge(a, merge(unknown, b)));
    const partial = merge(common, unknown).?;
    try std.testing.expectEqual(null, partial.hardware.properties.coreCount);
    try std.testing.expectEqual(null, partial.hardware.properties.globalMemoryBytes);
    try std.testing.expectEqual(64 * KiB, partial.hardware.limits.sharedMemoryPerWorkgroup);
    const excessive: Capabilities = .init(.pvc, .{ .sharedMemoryPerWorkgroupLimit = 256 * KiB });
    try std.testing.expectEqual(128 * KiB, excessive.hardware.limits.sharedMemoryPerWorkgroup);
    const zero: Capabilities = .init(.pvc, .{ .sharedMemoryPerWorkgroupLimit = 0 });
    try std.testing.expectEqual(0, zero.hardware.limits.sharedMemoryPerWorkgroup);
    try std.testing.expectEqual(null, merge(a, null));
    try std.testing.expectEqual(null, merge(a, .init(.bmg, .{})));
    try std.testing.expectEqual(null, merge(a, .init(.pvc_vg, .{})));
}
