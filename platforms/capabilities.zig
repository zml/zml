//! Hardware architecture identities, independent of runtimes and kernel support.
const std = @import("std");

const Platform = @import("platforms.zig").Platform;

pub const ComputeCapability = union(Platform) {
    cpu: void,
    cuda: Cuda,
    rocm: Rocm,
    tpu: void,
    neuron: void,
    oneapi: void,
    metal: void,

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
