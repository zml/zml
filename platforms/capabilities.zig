//! Hardware architecture identities, independent of runtimes and kernel support.
const std = @import("std");

const Platform = @import("platforms.zig").Platform;

pub const ComputeCapability = union(Platform) {
    cpu: void,
    cuda: Cuda,
    rocm: Rocm,
    tpu: void,
    neuron: void,
    oneapi: OneApi,
    metal: void,
    furiosa: void,

    pub fn unknown(target: Platform) ComputeCapability {
        return switch (target) {
            .cpu => .cpu,
            .cuda => .{ .cuda = .unknown },
            .rocm => .{ .rocm = .unknown },
            .tpu => .tpu,
            .neuron => .neuron,
            .oneapi => .{ .oneapi = .unknown },
            .metal => .metal,
            .furiosa => .furiosa,
        };
    }

    pub fn eql(self: ComputeCapability, other: ComputeCapability) bool {
        return std.meta.eql(self, other);
    }
};

/// Sources for CUDA compute capabilities, product architecture names, and the
/// representative devices listed below:
/// - Current GPUs: https://developer.nvidia.com/cuda/gpus
/// - Legacy GPUs: https://developer.nvidia.com/cuda/gpus/legacy
/// - Compiler target to architecture mapping:
///   https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-feature-list
///
/// The product pages are the source of truth for device-to-CC mappings. The
/// representative device on each enum field is illustrative rather than an
/// exhaustive list of products with that compute capability.
pub const Cuda = enum(u16) {
    unknown = 0,
    /// Volta: Tesla V100.
    sm70 = 70,
    /// Volta: Jetson AGX Xavier.
    sm72 = 72,
    /// Turing: GeForce RTX 2080.
    sm75 = 75,
    /// Ampere: A100.
    sm80 = 80,
    /// Ampere: GeForce RTX 3090.
    sm86 = 86,
    /// Ampere: Jetson AGX Orin.
    sm87 = 87,
    /// Ada Lovelace: GeForce RTX 4090.
    sm89 = 89,
    /// Hopper: H100.
    sm90 = 90,
    /// Blackwell: B200.
    sm100 = 100,
    /// Blackwell: Jetson Thor (pre-CUDA 13 name for sm110).
    sm101 = 101,
    /// Blackwell: B300.
    sm103 = 103,
    /// Blackwell: Jetson T5000 (Thor).
    sm110 = 110,
    /// Blackwell: GeForce RTX 5090.
    sm120 = 120,
    /// Blackwell: GB10 (DGX Spark).
    sm121 = 121,

    /// Accept PJRT's "major.minor" (or integer major) and canonical SM names.
    /// Unknown architectures are not assigned capabilities of a nearby model.
    pub fn parse(text: []const u8) ?Cuda {
        if (std.mem.startsWith(u8, text, "sm_")) {
            const value = parseDecimal(text[3..]) orelse return null;
            return std.enums.fromInt(Cuda, value);
        }
        if (std.mem.startsWith(u8, text, "sm")) return std.meta.stringToEnum(Cuda, text);
        var parts = std.mem.splitScalar(u8, text, '.');
        const major = parseDecimal(parts.first()) orelse return null;
        const minor = parseDecimal(parts.next() orelse "0") orelse return null;
        if (parts.next() != null or major > 6553 or minor > 9) return null;
        return std.enums.fromInt(Cuda, std.math.add(u16, major * 10, minor) catch return null);
    }

    pub const Architecture = enum {
        unknown,
        volta,
        turing,
        ampere,
        ada,
        hopper,
        blackwell,
    };

    /// Product architecture
    pub fn architecture(self: Cuda) Architecture {
        return switch (self) {
            .unknown => .unknown,
            .sm70, .sm72 => .volta,
            .sm75 => .turing,
            .sm80, .sm86, .sm87 => .ampere,
            .sm89 => .ada,
            .sm90 => .hopper,
            .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => .blackwell,
        };
    }
};

fn parseDecimal(text: []const u8) ?u16 {
    if (text.len == 0) return null;
    for (text) |c| if (!std.ascii.isDigit(c)) return null;
    return std.fmt.parseInt(u16, text, 10) catch null;
}

/// Sources for AMD GFX targets, product architecture names, and representative
/// devices:
/// - AMD GPU architecture specifications:
///   https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
/// - ROCm product-to-GFX lookup table:
///   https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table
/// - LLVM processor table (canonical target and architecture names):
///   https://llvm.org/docs/AMDGPUUsage.html#processors
///
/// The representative device on each enum field is illustrative. A GFX target
/// commonly covers several products in one family.
pub const Rocm = enum {
    unknown,
    /// CDNA1: MI100.
    gfx908,
    /// CDNA2: MI200 series.
    gfx90a,
    /// CDNA3: MI300 series.
    gfx942,
    /// CDNA4: MI350 series.
    gfx950,
    /// RDNA2: RX 6800/6900 series.
    gfx1030,
    /// RDNA2: RX 6700/6750 XT.
    gfx1031,
    /// RDNA2: RX 6600 series.
    gfx1032,
    /// RDNA2 APU: Van Gogh (Steam Deck).
    gfx1033,
    /// RDNA2: RX 6500 XT.
    gfx1034,
    /// RDNA2 APU: Radeon 680M.
    gfx1035,
    /// RDNA2 APU: Raphael.
    gfx1036,
    /// RDNA3 (Navi 31): RX 7900 series.
    gfx1100,
    /// RDNA3 (Navi 32): RX 7800 and RX 7700 series.
    gfx1101,
    /// RDNA3 (Navi 33): RX 7600 series.
    gfx1102,
    /// RDNA3 APU: Phoenix.
    gfx1103,
    /// RDNA3.5: Strix Point (Ryzen AI 9 HX 375).
    gfx1150,
    /// RDNA3.5: Strix Halo (Ryzen AI Max+ 395).
    gfx1151,
    /// RDNA3.5: Krackan Point (Ryzen AI 7 350).
    gfx1152,
    /// RDNA3.5: Radeon 820M.
    gfx1153,
    /// RDNA4 (Navi 44): RX 9060 family.
    gfx1200,
    /// RDNA4 (Navi 48): RX 9070 family.
    gfx1201,

    pub fn parse(text: []const u8) ?Rocm {
        return std.meta.stringToEnum(Rocm, std.mem.sliceTo(text, ':'));
    }

    pub const Architecture = enum {
        unknown,
        cdna1,
        cdna2,
        cdna3,
        cdna4,
        rdna2,
        rdna3,
        rdna3_5,
        rdna4,
    };

    /// Product architecture
    pub fn architecture(self: Rocm) Architecture {
        return switch (self) {
            .unknown => .unknown,
            .gfx908 => .cdna1,
            .gfx90a => .cdna2,
            .gfx942 => .cdna3,
            .gfx950 => .cdna4,
            .gfx1030, .gfx1031, .gfx1032, .gfx1033, .gfx1034, .gfx1035, .gfx1036 => .rdna2,
            .gfx1100, .gfx1101, .gfx1102, .gfx1103 => .rdna3,
            .gfx1150, .gfx1151, .gfx1152, .gfx1153 => .rdna3_5,
            .gfx1200, .gfx1201 => .rdna4,
        };
    }
};

/// Sources for OneAPI compute capabilities, product families, and
/// representative devices:
/// - https://github.com/openxla/xla/blob/main/xla/stream_executor/sycl/oneapi_compute_capability.h
/// - https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl
/// - https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/product_config_base.inl
pub const OneApi = enum {
    unknown,
    /// Xe HPG: Arc A-series, Arc Pro A-series, Data Center GPU Flex.
    dg2,
    /// Xe HPC: Data Center GPU Max.
    pvc,
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

    /// Canonical hardware names and explicit compiler product aliases.
    /// Bare ARL is ambiguous: U/S use MTL graphics, H has XMX.
    pub fn parse(text: []const u8) ?OneApi {
        inline for (std.meta.tags(OneApi)) |arch| {
            if (std.ascii.eqlIgnoreCase(text, @tagName(arch))) return arch;
        }
        const aliases = .{
            .{ "DG2-G10", OneApi.dg2 }, .{ "DG2-G11", OneApi.dg2 }, .{ "DG2-G12", OneApi.dg2 },
            .{ "BMG-G21", OneApi.bmg }, .{ "BMG-G31", OneApi.bmg }, .{ "MTL-H", OneApi.mtl },
            .{ "MTL-U", OneApi.mtl },   .{ "ARL-U", OneApi.mtl },   .{ "ARL-S", OneApi.mtl },
            .{ "ARL-H", OneApi.arl_h }, .{ "LNL-M", OneApi.lnl },   .{ "PTL-H", OneApi.ptl },
            .{ "PTL-U", OneApi.ptl },
        };
        inline for (aliases) |alias| {
            if (std.ascii.eqlIgnoreCase(text, alias[0])) return alias[1];
        }
        return null;
    }

    /// Level Zero IP version: architecture in bits 31:22, release in bits 21:14.
    /// Ignore revision only; an unknown release must not inherit capabilities.
    /// https://github.com/intel/compute-runtime/blob/master/third_party/aot_config_headers/platforms.h
    pub fn fromIpVersion(ip: u32) ?OneApi {
        return switch (ip >> 14) {
            (12 << 8) | 55, (12 << 8) | 56, (12 << 8) | 57 => .dg2,
            (12 << 8) | 60, (12 << 8) | 61 => .pvc,
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
    pub fn fromDeviceId(id: u32) ?OneApi {
        return switch (id) {
            0x4F80, 0x4F81, 0x4F82, 0x4F83, 0x4F84, 0x5690, 0x5691, 0x5692, 0x56A0, 0x56A1, 0x56A2, 0x56C0, 0x56C2, 0x56BE, 0x56BF, 0x56AF, 0x4F87, 0x4F88, 0x5693, 0x5694, 0x5695, 0x56A5, 0x56A6, 0x56B0, 0x56B1, 0x56BA, 0x56BB, 0x56BC, 0x56BD, 0x56C1, 0x5696, 0x5697, 0x56A3, 0x56A4, 0x56B2, 0x56B3, 0x4F85, 0x4F86 => .dg2,
            0x0BD0, 0x0BD5, 0x0BD6, 0x0BD7, 0x0BD8, 0x0BD9, 0x0BDA, 0x0BDB, 0x0B69, 0x0B6E, 0x0BD4 => .pvc,
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
    pub fn parseDeviceKind(text: []const u8) ?OneApi {
        const name = stripIntelBrand(text);
        const products = .{
            .{ "A770M", OneApi.dg2 },
            .{ "A730M", OneApi.dg2 },
            .{ "A550M", OneApi.dg2 },
            .{ "A370M", OneApi.dg2 },
            .{ "A350M", OneApi.dg2 },
            .{ "A570M", OneApi.dg2 },
            .{ "A530M", OneApi.dg2 },
            .{ "Pro A30M", OneApi.dg2 },
            .{ "Pro A40/A50", OneApi.dg2 },
            .{ "Pro A40", OneApi.dg2 },
            .{ "Pro A50", OneApi.dg2 },
            .{ "Pro A60M", OneApi.dg2 },
            .{ "Pro A60", OneApi.dg2 },
            .{ "A380E", OneApi.dg2 },
            .{ "A310E", OneApi.dg2 },
            .{ "A370E", OneApi.dg2 },
            .{ "A350E", OneApi.dg2 },
            .{ "A750E", OneApi.dg2 },
            .{ "A580E", OneApi.dg2 },
            .{ "A770", OneApi.dg2 },
            .{ "A750", OneApi.dg2 },
            .{ "A580", OneApi.dg2 },
            .{ "A380", OneApi.dg2 },
            .{ "A310 LP", OneApi.dg2 },
            .{ "A760A", OneApi.dg2 },
            .{ "Data Center GPU Flex 170", OneApi.dg2 },
            .{ "Data Center GPU Flex 140", OneApi.dg2 },
            .{ "Data Center GPU Flex 170V", OneApi.dg2 },
            .{ "Data Center GPU Max 1550", OneApi.pvc },
            .{ "Data Center GPU Max 1350", OneApi.pvc },
            .{ "Data Center GPU Max 1100", OneApi.pvc },
            .{ "Data Center GPU Max 1450", OneApi.pvc },
            .{ "Data Center GPU Max 1100C", OneApi.pvc },
            .{ "Data Center GPU Max 1550VG", OneApi.pvc },
            .{ "B580", OneApi.bmg },
            .{ "B570", OneApi.bmg },
            .{ "Pro B60", OneApi.bmg },
            .{ "Pro B50", OneApi.bmg },
            .{ "Pro B65", OneApi.bmg },
            .{ "Pro B70", OneApi.bmg },
            .{ "B390", OneApi.ptl },
            .{ "B370", OneApi.ptl },
            .{ "130V", OneApi.lnl },
            .{ "140V", OneApi.lnl },
            .{ "130T", OneApi.arl_h },
            .{ "140T", OneApi.arl_h },
            .{ "Ponte Vecchio", OneApi.pvc },
            .{ "Battlemage", OneApi.bmg },
            .{ "BMG-G21", OneApi.bmg },
            .{ "BMG-G31", OneApi.bmg },
        };
        inline for (products) |product| {
            if (std.ascii.eqlIgnoreCase(name, product[0])) return product[1];
        }
        return null;
    }

    pub const Architecture = enum {
        unknown,
        xe_hpg,
        xe_hpc,
        xe2_hpg,
        xe_lpg,
        xe_lpg_plus,
        xe2_lpg,
        xe3_lpg,
    };

    pub fn architecture(self: OneApi) Architecture {
        return switch (self) {
            .unknown => .unknown,
            .dg2 => .xe_hpg,
            .pvc => .xe_hpc,
            .bmg => .xe2_hpg,
            .mtl => .xe_lpg,
            .arl_h => .xe_lpg_plus,
            .lnl => .xe2_lpg,
            .ptl, .wcl => .xe3_lpg,
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
        .{ "Intel(R) Arc(TM) Pro A60 Graphics", OneApi.dg2 },
        .{ "Intel Arc Pro A40/A50", OneApi.dg2 },
        .{ "Intel(R) Arc(TM) B390 GPU", OneApi.ptl },
        .{ "Intel Arc B370", OneApi.ptl },
        .{ "Intel(R) Arc(TM) B580 Graphics", OneApi.bmg },
        .{ "Intel Arc Pro B70", OneApi.bmg },
        .{ "Intel Arc 140V", OneApi.lnl },
        .{ "Intel Arc 130T", OneApi.arl_h },
    };
    inline for (cases) |case| try std.testing.expectEqual(case[1], OneApi.parseDeviceKind(case[0]));
    for ([_][]const u8{ "Intel(R) Arc(TM) Graphics", "Intel(R) Graphics", "Intel Arc B999 GPU", "Intel Arc Pro A999", "Intel Arc A770junk", "Intel Arc B580 engineering sample" }) |name| {
        try std.testing.expectEqual(null, OneApi.parseDeviceKind(name));
    }
    try std.testing.expectEqual(null, OneApi.parse("ARL"));
    try std.testing.expectEqual(OneApi.mtl, OneApi.parse("ARL-S"));
    try std.testing.expectEqual(OneApi.arl_h, OneApi.parse("ARL-H"));
}

test "Intel hardware identifiers preserve architecture and release boundaries" {
    const cases = .{
        .{ 0x56B3, 0x030dc008, OneApi.dg2 },
        .{ 0x0BD5, 0x030f0007, OneApi.pvc },
        .{ 0xE223, 0x05008000, OneApi.bmg },
        .{ 0x7D55, 0x0311c004, OneApi.mtl },
        .{ 0x7D51, 0x03128004, OneApi.arl_h },
        .{ 0x64A0, 0x05010004, OneApi.lnl },
        .{ 0xB080, 0x07800004, OneApi.ptl },
        .{ 0xFD80, 0x0780c001, OneApi.wcl },
    };
    inline for (cases) |case| {
        try std.testing.expectEqual(case[2], OneApi.fromDeviceId(case[0]));
        try std.testing.expectEqual(case[2], OneApi.fromIpVersion(case[1]));
    }
    try std.testing.expectEqual(OneApi.mtl, OneApi.fromDeviceId(0x7D67));
    try std.testing.expectEqual(null, OneApi.fromDeviceId(0xB088));
    try std.testing.expectEqual(null, OneApi.fromIpVersion(0x0500c000));
    try std.testing.expectEqual(null, OneApi.fromIpVersion(0xffffffff));
}

test "architecture parsing is strict and future architectures stay unknown" {
    try std.testing.expectEqual(Cuda.sm100, Cuda.parse("10.0"));
    try std.testing.expectEqual(Cuda.sm100, Cuda.parse("10"));
    try std.testing.expectEqual(Cuda.sm103, Cuda.parse("sm_103"));
    try std.testing.expectEqual(Cuda.sm120, Cuda.parse("sm120"));
    for ([_][]const u8{ "", "10.0.1", "10.10", "10.", "-1", "+10", "10_0", "6553.9", "99999", "13.0", "sm_999", "sm_100a" }) |text| {
        try std.testing.expectEqual(null, Cuda.parse(text));
    }
    try std.testing.expectEqual(Rocm.gfx942, Rocm.parse("gfx942:sramecc+:xnack-"));
    try std.testing.expectEqual(null, Rocm.parse("gfx999"));
    try std.testing.expectEqual(null, Rocm.parse("gfx942junk"));
    try std.testing.expectEqual(Rocm.Architecture.cdna4, Rocm.gfx950.architecture());
}
