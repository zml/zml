//! ROCm capability contract for device-specialized kernel generation.
//!
//! Compute identifies a supported ISA target; supports() describes native
//! hardware instruction families. Hardware.limits is static resource geometry
//! narrowed by reported per-block restrictions. Neither certifies compiler
//! support, a legal instruction shape/layout, or a particular kernel's launch.
//! Keep tuning policy in generators and instruction restrictions in lowering.
//! See KERNEL_TUNING.md for the experiment and validation workflow.
//!
//! Population: ../../zml/platform.zig reads PJRT's string compute_capability,
//! parses it with Compute.parse(), and queries hardware_query.zig through
//! rocm.zig using localHardwareId (the HIP-visible ordinal). The query uses the
//! bundled HIP runtime on Linux when ROCm is enabled; failure leaves properties
//! unknown. Positive PJRT capacities then fill or narrow three properties (see
//! Properties). Device.capabilities is a snapshot for one visible device or
//! partition. Platform.capabilities merges all addressable devices once at
//! platform initialization; it is not a package total or a per-launch query.
//!
//! A missing/unrecognized target, or a merge of differing targets, is represented
//! outside this type as a null ROCm payload. A known target with Properties{}
//! still has static limits and features; it has no reported device capacities.
//! Do not substitute a product name, host GPU, or newer target for missing facts.
//!
//! Not modeled: compiler/plugin versions, instruction shape/type combinations,
//! per-kernel VGPR/SGPR/AGPR use, spills, LDS use, launch bounds or occupancy;
//! SGPR capacity, resident-workgroup limits, grid/per-axis block limits;
//! cache lines/associativity, cache/XCC connectivity, memory locality or workgroup
//! placement; instruction throughput, achieved bandwidth, live clocks, power,
//! free memory or other workloads. These require backend checks, compiled-kernel
//! inspection or device measurements. xccCount alone supplies no topology map.
const std = @import("std");

const KiB = 1024;
const Capabilities = @This();

/// ISA identity, normally parsed from PJRT; not a product or compiler version.
compute: Compute,
/// Static geometry and reported properties for the scope described above.
hardware: Hardware,

/// Construct without querying hardware or validating caller-supplied properties.
/// Copy properties unchanged; derive limits from compute and take the minimum
/// with each reported maxThreadsPerBlockLimit/sharedMemoryPerBlockLimit. Null
/// leaves that static ceiling intact, zero lowers it to zero, and a larger
/// report cannot raise it. No other property changes the static limits.
pub fn init(compute: Compute, properties: Hardware.Properties) Capabilities {
    return .{
        .compute = compute,
        .hardware = .{
            .limits = properties.restrict(compute.limits()),
            .properties = properties,
        },
    };
}

/// Combine views for code shared across their execution devices. Either null
/// input or unequal Compute values yields null, even within one Architecture;
/// there is no common-ISA inference. Once null, further merges remain null.
/// For equal targets, merge properties by the rules on Properties.merge(), then
/// call init() to recompute limits. Capacities are never summed. The result is a
/// conservative summary and need not describe any single physical device.
pub fn merge(lhs_: ?Capabilities, rhs_: ?Capabilities) ?Capabilities {
    const lhs = lhs_ orelse return null;
    const rhs = rhs_ orelse return null;
    if (lhs.compute != rhs.compute) return null;
    return .init(lhs.compute, lhs.hardware.properties.merge(rhs.hardware.properties));
}

pub const Hardware = struct {
    /// Effective ceilings/geometry after init() applies reported restrictions.
    limits: Limits,
    /// Reports retained by init(), or their merged summary; not live telemetry.
    properties: Properties,

    /// Static target table in Compute.limits(), except the two per-block ceilings
    /// narrowed by init(). Sources are linked at Compute.limits(). No field is
    /// optional: zero denotes an absent resource, not an unknown value.
    ///
    /// Geometry assumes HIP's native wave size and usual CU (CDNA) / WGP (RDNA)
    /// execution mode. PerCu means one physical CU; an LDS domain may span two.
    /// A backend selecting a different mode/wave size must redo the accounting;
    /// this API neither queries that mode nor provides alternate-mode limits.
    /// These are resource ceilings, not achievable occupancy or tuning targets.
    pub const Limits = struct {
        /// Threads (lanes) per native wavefront; not a memory vector width.
        waveSize: u32,
        /// Total threads per workgroup/block, possibly restricted by HIP.
        /// Per-axis dimensions and compiled-kernel bounds are not represented.
        maxThreadsPerBlock: u32 = 1024,
        /// SIMD execution units per physical CU, not per WGP or HIP processor.
        simdsPerCu: u32,
        /// Resident wave slots per SIMD at waveSize, before resource constraints.
        maxWavesPerSimd: u32,
        /// Physical CUs sharing one LDS allocation domain (CU on CDNA, WGP on
        /// RDNA). Does not specify how HIP counts multiprocessors or places work.
        computeUnitsPerLdsDomain: u32,
        /// Bytes of LDS shared by resident workgroups in one allocation domain.
        /// Runtime per-block restrictions do not reduce this static capacity.
        sharedMemoryPerLdsDomain: u32,
        /// LDS byte ceiling for one workgroup, possibly restricted by HIP/PJRT.
        /// Account for static and dynamic LDS together, including layout padding.
        sharedMemoryPerBlock: u32,
        /// LDS allocation granule in bytes for residency estimates. Round up by
        /// integer multiples; this granule is not always a power of two.
        sharedMemoryAllocationBytes: u32 = 512,
        /// Number of LDS banks; not a throughput or conflict-cycle prediction.
        sharedMemoryBanks: u32,
        /// Bytes per bank word for address-to-bank analysis. Instruction width,
        /// lane grouping and broadcast behavior still belong to the backend.
        sharedMemoryBankBytes: u32 = 4,
        /// Vector-register storage in bytes per SIMD. Includes the shared
        /// VGPR/AGPR file on CDNA2 and later: do not budget those files separately.
        /// This is physical capacity, not the register-index limit of a kernel.
        vgprFileBytesPerSimd: u32,
        /// Separate accumulator-register bytes per SIMD on CDNA1, unavailable
        /// to ordinary VALU. Zero elsewhere means no separate file, not no MMA.
        separateAccumulatorFileBytesPerSimd: u32 = 0,
        /// Allocation granule in 32-bit vector registers per lane at waveSize.
        /// Not bytes or a maximum count; not always a power of two. Combined
        /// VGPR/AGPR placement and actual allocation require compiler metadata.
        vgprAllocationPerThread: u32,
        /// Closest vector data-cache bytes per physical CU: L1 on CDNA, L0 on
        /// RDNA. Nominal capacity, not guaranteed usable space for a kernel.
        vectorCacheBytesPerCu: u32,

        /// Wave-slot ceiling per physical CU; ignores registers, LDS and groups.
        pub fn maxWavesPerCu(self: Limits) u32 {
            return self.simdsPerCu * self.maxWavesPerSimd;
        }

        /// Lane slots per physical CU at waveSize, not threads in one block.
        pub fn maxThreadsPerCu(self: Limits) u32 {
            return self.waveSize * self.maxWavesPerCu();
        }
    };

    /// HIP/PJRT reports for one visible device/partition, or a merged summary.
    /// Never infer package totals from them. All fields default to null (unknown,
    /// including query failure); a numeric zero or false is a reported fact.
    ///
    /// hardware_query.zig uses hipDeviceGetAttribute for the attributes named
    /// below. Failed/unsupported queries and negative numeric results become
    /// null; successful zero is preserved. Boolean reports are nonzero => true.
    /// globalMemoryBytes instead uses hipDeviceTotalMem, preserving zero too.
    ///
    /// ../../zml/platform.zig overlays PJRT core_count,
    /// shared_memory_per_block_optin and device_memory_bytes_limit on the three
    /// fields noted below, taking min(HIP, PJRT) when both exist. Only positive,
    /// representable int64 PJRT values participate: zero, negative, wrong-typed
    /// and out-of-range values are ignored. The optin name is only a PJRT key;
    /// this model does not expose a separate ROCm opt-in allocation mechanism.
    /// Query failures do not prevent a known Compute from retaining static limits.
    pub const Properties = struct {
        /// Count from hipDeviceAttributeMultiprocessorCount, narrowed/filled by
        /// PJRT core_count. RDNA may report WGPs; the count's CU/WGP unit is not
        /// normalized or separately recorded. Establish the runtime mode before
        /// combining it with per-physical-CU geometry or LDS-domain capacity.
        multiprocessorCount: ?u32 = null,
        /// Threads per block from hipDeviceAttributeMaxThreadsPerBlock. Used by
        /// init() as a restriction, not as a replacement for the static ceiling.
        maxThreadsPerBlockLimit: ?u32 = null,
        /// Bytes per block from hipDeviceAttributeMaxSharedMemoryPerBlock,
        /// narrowed/filled by PJRT shared_memory_per_block_optin. Restricts init().
        sharedMemoryPerBlockLimit: ?u32 = null,
        /// Aggregate L2 bytes reported by hipDeviceAttributeL2CacheSize. No cache
        /// domain count, sharing, partition locality or usable fraction is known.
        l2CacheBytes: ?u64 = null,
        /// Total device memory bytes from hipDeviceTotalMem, narrowed/filled by
        /// PJRT device_memory_bytes_limit. Not free memory or allocation success.
        globalMemoryBytes: ?u64 = null,
        /// Memory-interface width in bits from hipDeviceAttributeMemoryBusWidth.
        /// Does not supply memory technology, channel mapping or transfer rate.
        memoryBusWidthBits: ?u32 = null,
        /// Advertised core clock in kHz from hipDeviceAttributeClockRate; not a
        /// sampled operating frequency or an instruction-throughput guarantee.
        clockRateKHz: ?u32 = null,
        /// Advertised memory clock in kHz from hipDeviceAttributeMemoryClockRate.
        /// A bytes/s estimate also needs a transfer-per-clock convention absent
        /// here; bus width times this value alone is not measured bandwidth.
        memoryClockRateKHz: ?u32 = null,
        /// Asynchronous copy-engine count from hipDeviceAttributeAsyncEngineCount.
        /// Host/device transfer capability, not in-kernel direct-to-LDS support,
        /// pipeline stages or guaranteed overlap in the current execution path.
        asyncEngineCount: ?u32 = null,
        /// hipDeviceAttributeConcurrentKernels: runtime support for concurrent
        /// kernel execution, not a promise of overlap or available resources.
        concurrentKernels: ?bool = null,
        /// hipDeviceAttributeCooperativeLaunch: runtime cooperative-launch
        /// support. Does not establish that a ZML custom call uses that launch.
        cooperativeLaunch: ?bool = null,
        /// hipDeviceAttributeIntegrated: device reported as an integrated GPU.
        /// Does not establish pointer accessibility or zero-copy behavior.
        integrated: ?bool = null,
        /// hipDeviceAttributeEccEnabled: reported ECC configuration; not a
        /// capability to toggle it or a measured bandwidth/capacity penalty.
        eccEnabled: ?bool = null,
        /// XCC count in the visible compute partition from
        /// hipDeviceAttributeNumberOfXccs. No CU/cache/memory mapping is provided.
        xccCount: ?u32 = null,

        /// multiprocessorCount, l2CacheBytes, globalMemoryBytes, memoryBusWidthBits,
        /// clockRateKHz, memoryClockRateKHz and asyncEngineCount use min only when
        /// both inputs are known; either unknown makes the result unknown.
        /// maxThreadsPerBlockLimit/sharedMemoryPerBlockLimit instead preserve a
        /// sole known restriction (min if both known).
        /// concurrentKernels/cooperativeLaunch: any false => false, two true =>
        /// true, otherwise null. integrated/eccEnabled/xccCount survive only if
        /// equal, including null == null; topology is not ordered by capacity.
        /// Zero participates in all of these rules as a known value.
        fn merge(self: Properties, other: Properties) Properties {
            var result: Properties = .{};
            inline for (comptime std.meta.tags(std.meta.FieldEnum(Properties))) |field| {
                const name = comptime @tagName(field);
                const lhs = @field(self, name);
                const rhs = @field(other, name);
                @field(result, name) = switch (field) {
                    .multiprocessorCount,
                    .l2CacheBytes,
                    .globalMemoryBytes,
                    .memoryBusWidthBits,
                    .clockRateKHz,
                    .memoryClockRateKHz,
                    .asyncEngineCount,
                    => if (lhs != null and rhs != null) @min(lhs.?, rhs.?) else null,
                    .maxThreadsPerBlockLimit, .sharedMemoryPerBlockLimit => if (lhs != null and rhs != null) @min(lhs.?, rhs.?) else lhs orelse rhs,
                    .concurrentKernels, .cooperativeLaunch => if (lhs == false or rhs == false) false else if (lhs == true and rhs == true) true else null,
                    // Configuration and topology cannot be ordered as capacities.
                    .integrated, .eccEnabled, .xccCount => if (lhs == rhs) lhs else null,
                };
            }
            return result;
        }

        /// Apply only per-block restrictions; preserve all other static geometry.
        fn restrict(self: Properties, architectureLimits: Limits) Limits {
            var result = architectureLimits;
            if (self.maxThreadsPerBlockLimit) |threads| result.maxThreadsPerBlock = @min(result.maxThreadsPerBlock, threads);
            if (self.sharedMemoryPerBlockLimit) |bytes| result.sharedMemoryPerBlock = @min(result.sharedMemoryPerBlock, bytes);
            return result;
        }
    };
};

/// Native hardware instruction-family predicates in Compute.supports(). A true
/// value is necessary but insufficient for choosing a lowering: it does not
/// describe installed compiler support, every opcode/type/shape combination,
/// accumulator precision, fragment layout, alignment or numerical behavior.
/// A false tensor feature does not prohibit scalar or software-emulated math.
/// Packed VALU arithmetic, matrix arithmetic and data movement are independent.
pub const Feature = enum {
    /// Matrix fused multiply-add instruction family.
    mfma,
    /// Wave matrix multiply-accumulate instruction family.
    wmma,
    /// Packed FP16 vector arithmetic; not matrix arithmetic.
    packedFp16,
    /// Packed FP32 vector arithmetic; not matrix arithmetic.
    packedFp32,
    /// Native matrix arithmetic with FP16 inputs.
    tensorFp16,
    /// Native matrix arithmetic with BF16 inputs.
    tensorBf16,
    /// Native matrix arithmetic with FP32 inputs; distinct from XF32 below.
    tensorFp32,
    /// Native FP64 matrix arithmetic.
    tensorFp64,
    /// Native matrix arithmetic with 8-bit integer inputs.
    tensorInt8,
    /// Native matrix arithmetic with 4-bit integer inputs.
    tensorInt4,
    /// CDNA3's reduced-precision XF32 MFMA; not inherited by CDNA4. Requires an
    /// explicit numerical contract rather than substitution for full FP32 math.
    tensorXf32,
    /// Matrix arithmetic with E4M3/E5M2 FNUZ, distinct encodings from OCP FP8.
    tensorFp8Fnuz,
    /// Matrix arithmetic with OCP E4M3/E5M2 inputs; not FNUZ.
    tensorFp8,
    /// Native matrix arithmetic with FP6 inputs; check formats in lowering.
    tensorFp6,
    /// Native matrix arithmetic with FP4 inputs; check formats in lowering.
    tensorFp4,
    /// Native block-scaled matrix arithmetic. Scale encoding, grouping and
    /// supported operand combinations remain instruction-specific.
    blockScaledMma,
    /// Sparse WMMA family; sparsity pattern and metadata are backend constraints.
    sparseWmma,
    /// Buffer-addressed loads directly into LDS, bypassing VGPR staging.
    bufferLoadToLds,
    /// Global-addressed loads directly into LDS.
    globalLoadToLds,
    /// 12/16-byte direct-to-LDS loads per lane, in addition to 1/2/4 bytes.
    /// The addressing feature must also match. None of these load flags specifies
    /// alignment, completion/wait rules, barriers or automatic pipeline overlap.
    wideLoadToLds,
};

/// Supported GFX target identities, not an ordered feature/performance scale.
/// Targets in one Architecture can still have different features or geometry.
/// Sources for target names, architecture names and representative devices:
/// - AMD GPU architecture specifications:
///   https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
/// - ROCm product-to-GFX lookup table:
///   https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table
/// - LLVM processor table (canonical target and architecture names):
///   https://llvm.org/docs/AMDGPUUsage.html#processors
///
/// The representative device on each enum field is illustrative. A GFX target
/// commonly covers several products in one family.
pub const Compute = enum {
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

    /// Family grouping for description; use supports() for instruction selection.
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

    /// Map this exact target to its architecture family; no runtime query.
    pub fn architecture(self: Compute) Architecture {
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

    /// Look up a native instruction family for this exact target, independent of
    /// installed ROCm/compiler versions and product name. This is a static table,
    /// not a compiler probe or a promise that a particular kernel can use it.
    /// LLVM AMDGPU.td / VOP3PInstructions.td and AMD's ISA references are more
    /// precise than HIP's aggregate datatype-support tables (which may include
    /// emulation). In particular, gfx950 does not have gfx942's XF32 instructions.
    /// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPU.td
    /// https://rocm.docs.amd.com/en/latest/reference/precision-support.html
    /// https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/cross-arch/direct-to-lds-builtins.html
    pub fn supports(self: Compute, feature: Feature) bool {
        const arch = self.architecture();
        return switch (feature) {
            .packedFp16 => true,
            .mfma, .tensorFp32 => switch (arch) {
                .cdna1, .cdna2, .cdna3, .cdna4 => true,
                .rdna2, .rdna3, .rdna3_5, .rdna4 => false,
            },
            .wmma, .tensorInt4 => switch (arch) {
                .rdna3, .rdna3_5, .rdna4 => true,
                .cdna1, .cdna2, .cdna3, .cdna4, .rdna2 => false,
            },
            .tensorFp16, .tensorBf16, .tensorInt8 => switch (arch) {
                .cdna1, .cdna2, .cdna3, .cdna4, .rdna3, .rdna3_5, .rdna4 => true,
                .rdna2 => false,
            },
            .packedFp32, .tensorFp64 => switch (arch) {
                .cdna2, .cdna3, .cdna4 => true,
                .cdna1, .rdna2, .rdna3, .rdna3_5, .rdna4 => false,
            },
            .tensorXf32, .tensorFp8Fnuz => self == .gfx942,
            .tensorFp8 => switch (arch) {
                .cdna4, .rdna4 => true,
                .cdna1, .cdna2, .cdna3, .rdna2, .rdna3, .rdna3_5 => false,
            },
            .tensorFp6, .tensorFp4, .blockScaledMma, .wideLoadToLds => self == .gfx950,
            .sparseWmma => switch (arch) {
                .rdna4 => true,
                .cdna1, .cdna2, .cdna3, .cdna4, .rdna2, .rdna3, .rdna3_5 => false,
            },
            .bufferLoadToLds => switch (arch) {
                .cdna1, .cdna2, .cdna3, .cdna4, .rdna2 => true,
                .rdna3, .rdna3_5, .rdna4 => false,
            },
            .globalLoadToLds => switch (arch) {
                .cdna3, .cdna4 => true,
                .cdna1, .cdna2, .rdna2, .rdna3, .rdna3_5, .rdna4 => false,
            },
        };
    }

    /// Match an enum tag exactly and case-sensitively, using only text before the
    /// first ':'. Target suffixes such as sramecc+/xnack- are discarded, not
    /// validated or retained. Missing/unknown names return null; there is no
    /// product-name lookup or fallback to another target.
    pub fn parse(text: []const u8) ?Compute {
        return std.meta.stringToEnum(Compute, std.mem.sliceTo(text, ':'));
    }

    /// Static resource geometry for the native HIP wave size, not queried HIP
    /// attributes. init() applies the runtime restrictions afterward. Sources:
    /// wave/VGPR ceilings and allocation granules follow LLVM's IsaInfo;
    /// cache/register capacities and LDS organization follow AMD's references.
    /// These targets are not interchangeable:
    /// gfx908 has ten wave slots/SIMD; gfx90a/gfx942/gfx950 have eight.
    /// RDNA LDS capacity below is per WGP, while its workgroup limit is 64 KiB.
    /// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp
    /// https://llvm.org/docs/AMDGPUUsage.html#kernel-descriptor
    /// https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
    /// https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html#local-data-share-lds
    fn limits(self: Compute) Hardware.Limits {
        return switch (self) {
            .gfx908 => .{
                .waveSize = 64,
                .simdsPerCu = 4,
                .maxWavesPerSimd = 10,
                .computeUnitsPerLdsDomain = 1,
                .sharedMemoryPerLdsDomain = 64 * KiB,
                .sharedMemoryPerBlock = 64 * KiB,
                .sharedMemoryBanks = 32,
                .vgprFileBytesPerSimd = 64 * KiB,
                .separateAccumulatorFileBytesPerSimd = 64 * KiB,
                .vgprAllocationPerThread = 4,
                .vectorCacheBytesPerCu = 16 * KiB,
            },
            .gfx90a, .gfx942 => .{
                .waveSize = 64,
                .simdsPerCu = 4,
                .maxWavesPerSimd = 8,
                .computeUnitsPerLdsDomain = 1,
                .sharedMemoryPerLdsDomain = 64 * KiB,
                .sharedMemoryPerBlock = 64 * KiB,
                .sharedMemoryBanks = 32,
                .vgprFileBytesPerSimd = 128 * KiB,
                .vgprAllocationPerThread = 8,
                .vectorCacheBytesPerCu = if (self == .gfx90a) 16 * KiB else 32 * KiB,
            },
            .gfx950 => .{
                .waveSize = 64,
                .simdsPerCu = 4,
                .maxWavesPerSimd = 8,
                .computeUnitsPerLdsDomain = 1,
                .sharedMemoryPerLdsDomain = 160 * KiB,
                .sharedMemoryPerBlock = 160 * KiB,
                .sharedMemoryAllocationBytes = 1280,
                .sharedMemoryBanks = 64,
                .vgprFileBytesPerSimd = 128 * KiB,
                .vgprAllocationPerThread = 8,
                .vectorCacheBytesPerCu = 32 * KiB,
            },
            .gfx1030, .gfx1031, .gfx1032, .gfx1033, .gfx1034, .gfx1035, .gfx1036 => .{
                .waveSize = 32,
                .simdsPerCu = 2,
                .maxWavesPerSimd = 16,
                .computeUnitsPerLdsDomain = 2,
                .sharedMemoryPerLdsDomain = 128 * KiB,
                .sharedMemoryPerBlock = 64 * KiB,
                .sharedMemoryBanks = 64,
                .vgprFileBytesPerSimd = 128 * KiB,
                .vgprAllocationPerThread = 16,
                .vectorCacheBytesPerCu = 16 * KiB,
            },
            .gfx1100, .gfx1101, .gfx1102, .gfx1103, .gfx1150, .gfx1151, .gfx1152, .gfx1153, .gfx1200, .gfx1201 => blk: {
                const largeVgprFile = switch (self) {
                    .gfx1100, .gfx1101, .gfx1151, .gfx1200, .gfx1201 => true,
                    .gfx1102, .gfx1103, .gfx1150, .gfx1152, .gfx1153 => false,
                    .gfx908, .gfx90a, .gfx942, .gfx950, .gfx1030, .gfx1031, .gfx1032, .gfx1033, .gfx1034, .gfx1035, .gfx1036 => unreachable,
                };
                break :blk .{
                    .waveSize = 32,
                    .simdsPerCu = 2,
                    .maxWavesPerSimd = 16,
                    .computeUnitsPerLdsDomain = 2,
                    .sharedMemoryPerLdsDomain = 128 * KiB,
                    .sharedMemoryPerBlock = 64 * KiB,
                    .sharedMemoryBanks = 64,
                    .vgprFileBytesPerSimd = if (largeVgprFile) 192 * KiB else 128 * KiB,
                    .vgprAllocationPerThread = if (largeVgprFile) 24 else 16,
                    .vectorCacheBytesPerCu = 32 * KiB,
                };
            },
        };
    }
};

test "ROCm architecture parsing rejects missing and unrecognized architectures" {
    try std.testing.expectEqual(Compute.gfx942, Compute.parse("gfx942:sramecc+:xnack-"));
    try std.testing.expectEqual(null, Compute.parse(""));
    try std.testing.expectEqual(null, Compute.parse("gfx999"));
    try std.testing.expectEqual(null, Compute.parse("gfx942junk"));
    try std.testing.expectEqual(null, Compute.parse("unknown"));
    try std.testing.expectEqual(Compute.Architecture.cdna4, Compute.gfx950.architecture());
}

test "ROCm merges retain restrictions and require shared device facts" {
    const larger: Capabilities = .init(.gfx950, .{
        .multiprocessorCount = 256,
        .globalMemoryBytes = 288 * 1024 * 1024 * KiB,
        .sharedMemoryPerBlockLimit = 160 * KiB,
        .cooperativeLaunch = true,
        .concurrentKernels = true,
        .eccEnabled = true,
        .xccCount = 8,
    });
    const smaller: Capabilities = .init(.gfx950, .{
        .multiprocessorCount = 32,
        .globalMemoryBytes = 36 * 1024 * 1024 * KiB,
        .sharedMemoryPerBlockLimit = 64 * KiB,
        .maxThreadsPerBlockLimit = 512,
        .cooperativeLaunch = false,
        .concurrentKernels = true,
        .eccEnabled = false,
        .xccCount = 1,
    });
    const common = Capabilities.merge(larger, smaller).?;
    try std.testing.expectEqual(32, common.hardware.properties.multiprocessorCount);
    try std.testing.expectEqual(36 * 1024 * 1024 * KiB, common.hardware.properties.globalMemoryBytes);
    try std.testing.expectEqual(64 * KiB, common.hardware.limits.sharedMemoryPerBlock);
    try std.testing.expectEqual(512, common.hardware.limits.maxThreadsPerBlock);
    try std.testing.expectEqual(false, common.hardware.properties.cooperativeLaunch);
    try std.testing.expectEqual(true, common.hardware.properties.concurrentKernels);
    try std.testing.expectEqual(null, common.hardware.properties.eccEnabled);
    try std.testing.expectEqual(null, common.hardware.properties.xccCount);
    try std.testing.expectEqualDeep(common, Capabilities.merge(smaller, larger).?);
    try std.testing.expectEqualDeep(larger, Capabilities.merge(larger, larger).?);

    const unreported: Capabilities = .init(.gfx950, .{});
    const partial = Capabilities.merge(common, unreported).?;
    try std.testing.expectEqual(null, partial.hardware.properties.multiprocessorCount);
    try std.testing.expectEqual(null, partial.hardware.properties.globalMemoryBytes);
    try std.testing.expectEqual(null, partial.hardware.properties.concurrentKernels);
    try std.testing.expectEqual(false, partial.hardware.properties.cooperativeLaunch);
    try std.testing.expectEqual(64 * KiB, partial.hardware.limits.sharedMemoryPerBlock);
    try std.testing.expectEqual(512, partial.hardware.limits.maxThreadsPerBlock);
    try std.testing.expectEqualDeep(partial, Capabilities.merge(larger, Capabilities.merge(smaller, unreported)).?);
    try std.testing.expectEqualDeep(partial, Capabilities.merge(unreported, Capabilities.merge(smaller, larger)).?);

    const zero = Capabilities.merge(larger, .init(.gfx950, .{ .multiprocessorCount = 0, .sharedMemoryPerBlockLimit = 0 })).?;
    try std.testing.expectEqual(0, zero.hardware.properties.multiprocessorCount);
    try std.testing.expectEqual(0, zero.hardware.limits.sharedMemoryPerBlock);
}

test "ROCm merges cannot recover a missing or differing architecture" {
    const known: Capabilities = .init(.gfx942, .{});
    for ([_]?Capabilities{ null, .init(.gfx950, .{}) }) |other| {
        try std.testing.expectEqual(null, Capabilities.merge(known, other));
        try std.testing.expectEqual(null, Capabilities.merge(other, known));
        try std.testing.expectEqual(null, Capabilities.merge(Capabilities.merge(known, other), known));
    }
}
