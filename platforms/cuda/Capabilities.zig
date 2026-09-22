//! CUDA hardware contract for kernels specialized on the execution device before
//! use. Keep generation policy generic: query features and resources here, choose
//! workload-dependent configurations near the generator, and enforce instruction
//! encodings, target suffixes and toolchain restrictions in backend lowering.
//!
//! Population: zml/platform.zig parses PJRT's string `compute_capability`, queries
//! hardware_query.zig through cuda.queryHardwareProperties, applies the three
//! PJRT resource restrictions documented on Properties, then calls init. The
//! driver query uses PJRT's localHardwareId in the CUDA_VISIBLE_DEVICES ordinal
//! space, after loading the selected CUDA driver. It is available only with CUDA
//! enabled on Linux; query failures do not discard a recognized compute identity.
//!
//! Device.capabilities describes one visible device/partition. Platform.capabilities
//! merges every device in platform.devices, not just a kernel's execution subset.
//! The outer platforms.Capabilities.cuda is optional: a missing/unrecognized
//! compute identity or a merge of different identities produces null, even when
//! some resource properties are known. A known identity with missing properties
//! still supplies static limits. See merge for the exact aggregation rules.
//!
//! These are hardware facts, not compiler support or compiled-kernel attributes.
//! No API here reports actual register use, spills, static/dynamic kernel shared
//! memory, function launch limits, occupancy, cluster dimensions/residency, tensor
//! memory capacity, or backend instruction availability. There is no measured
//! bandwidth, live clock, memory/atomic/barrier latency, tensor throughput, free
//! memory, interconnect topology, or guarantee of exclusive device resources.
//! The optional arithmetic rates and latency below have deliberately limited
//! coverage. Unknown information must not become a guessed product specification.
//!
//! See [KERNEL_TUNING.md](KERNEL_TUNING.md) for the measurement workflow.
const std = @import("std");

const KiB = 1024;
const Capabilities = @This();

/// Recognized compute identity; does not encode compiler `a`/`f` target suffixes.
compute: Compute,
/// Static limits plus reported properties for the device or merged scope.
hardware: Hardware,

/// Resolve the private Compute.limits table, then apply Properties.restrict.
/// Only sharedMemoryPerBlockOptinLimit narrows limits, clamping both per-block
/// shared-memory ceilings. Other properties are retained without changing limits;
/// reservedSharedMemoryPerBlock is not subtracted automatically. No driver query,
/// validation of caller-supplied properties, or CUDA configuration occurs here.
/// With .{} properties, all static limits remain available.
pub fn init(compute: Compute, properties: Hardware.Properties) Capabilities {
    return .{
        .compute = compute,
        .hardware = .{
            .limits = properties.restrict(compute.limits()),
            .properties = properties,
        },
    };
}

/// Common view of two scopes with exactly the same Compute value. Null inputs
/// and differing identities return null; null is absorbing, not an empty scope.
/// No feature intersection or "nearest" architecture is inferred. For a matching
/// identity, merge properties and re-run init (do not merge existing limits).
/// Counts/capacities are never summed; see Properties.merge for unknown handling.
/// The result need not describe any single physical device, and missing reports
/// prevent it from guaranteeing that every runtime restriction has been captured.
pub fn merge(lhs_: ?Capabilities, rhs_: ?Capabilities) ?Capabilities {
    const lhs = lhs_ orelse return null;
    const rhs = rhs_ orelse return null;
    if (lhs.compute != rhs.compute) return null;
    return .init(lhs.compute, lhs.hardware.properties.merge(rhs.hardware.properties));
}

pub const Hardware = struct {
    /// Static model, narrowed only by Properties.sharedMemoryPerBlockOptinLimit.
    limits: Limits,
    /// Reports for one device, or the per-field common view produced by merge.
    properties: Properties,

    /// Concrete planning limits from Compute.limits, available without querying
    /// a device. Sources: the CUDA specifications and per-architecture cache and
    /// scheduler descriptions linked there. Allocation geometry follows CUDA 13.4
    /// cuda_occupancy.h. Memory transaction
    /// and bank geometry: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations
    ///
    /// Registers are counts of 32-bit registers, memory is bytes, and KiB = 1024.
    /// Block means CUDA thread block (CTA/work-group); SM means multiprocessor.
    /// These are independent ceilings, not resources simultaneously achievable
    /// by one kernel. Only init applies the reported per-block restriction; the
    /// other fields are static facts, not independently queried device limits.
    pub const Limits = struct {
        /// Threads per hardware warp; not a tile size or items per thread.
        warpSize: u32 = 32,
        /// Maximum total threads in one block, in addition to dimension limits.
        maxThreadsPerBlock: u32 = 1024,
        /// Maximum thread counts along block x/y/z, respectively.
        maxBlockDimensions: [3]u32 = .{ 1024, 1024, 64 },
        /// Maximum block counts along grid x/y/z, not resident block counts.
        maxGridDimensions: [3]u32 = .{ 2147483647, 65535, 65535 },
        /// Maximum resident warps on one SM before kernel resource constraints.
        maxWarpsPerSm: u32,
        /// Maximum resident blocks on one SM before kernel resource constraints.
        maxBlocksPerSm: u32,
        /// Warp schedulers per SM; does not specify issue rate or eligible warps.
        warpSchedulersPerSm: u32 = 4,
        /// Register-allocation subpartitions per SM. Total free registers alone
        /// do not determine residency; warp placement within subpartitions matters.
        smSubpartitions: u32 = 4,
        /// Allocation quantum in 32-bit registers per warp, not per thread/block.
        registerAllocationPerWarp: u32 = 256,
        /// Shared-memory allocation quantum in bytes per block.
        sharedMemoryAllocationBytes: u32 = 128,
        /// Total 32-bit register slots per SM, shared by resident blocks.
        registersPerSm: u32 = 64 * KiB,
        /// Maximum allocated 32-bit register slots for one block.
        registersPerBlock: u32 = 64 * KiB,
        /// Architectural 32-bit registers/thread ceiling, not a compiler cap
        /// recommendation or a prediction of the kernel's allocation.
        registersPerThread: u32 = 255,
        /// Kernel shared-memory bytes/block without explicit larger-memory opt-in.
        sharedMemoryPerBlock: u32 = 48 * KiB,
        /// Kernel shared-memory bytes/block with opt-in (static + dynamic total).
        /// This capacity does not opt a function in; its dynamic-memory allowance
        /// and the backend launch configuration must also permit the request.
        /// Extra capacity requires dynamic storage, not larger static arrays.
        sharedMemoryPerBlockOptin: u32,
        /// Unified L1/shared-memory pool bytes per SM. Not an available shared
        /// memory budget or a report of the current usable L1 cache capacity.
        unifiedL1SharedBytes: u32,
        /// Ascending, nonempty static list of shared-memory partitions in bytes/SM.
        /// Includes zero where supported. init does not trim this list when it
        /// narrows per-block limits; the runtime's chosen partition is not queried.
        sharedMemoryCarveouts: []const u32,
        /// Number of shared-memory banks used by the bank-address model.
        sharedMemoryBanks: u32 = 32,
        /// Bytes per successive bank word. Wide accesses and broadcasts require
        /// instruction-specific analysis; this is not shared bandwidth/clock.
        sharedMemoryBankBytes: u32 = 4,
        /// Bytes per global-memory coalescing sector, not a cache-line size,
        /// vector width, or guarantee that a sector reaches DRAM.
        globalMemorySectorBytes: u32 = 32,

        /// Derived resident-thread ceiling per SM: warpSize * maxWarpsPerSm.
        pub fn maxThreadsPerSm(self: Limits) u32 {
            return self.warpSize * self.maxWarpsPerSm;
        }

        /// Largest modeled shared partition in bytes/SM, not the active carveout
        /// or a per-block allowance. Assumes the static list's ordering invariant.
        pub fn sharedMemoryPerSm(self: Limits) u32 {
            return self.sharedMemoryCarveouts[self.sharedMemoryCarveouts.len - 1];
        }

        /// Smallest listed partition >= requiredBytes (aggregate bytes on one SM),
        /// or null if none fits. The caller accounts for allocation rounding and
        /// driver-reserved bytes for every proposed resident block. Unknown reserve
        /// prevents an exact budget. Zero selects the smallest listed partition.
        /// No CUDA setting is changed: carveout requests are preferences. This
        /// helper checks neither per-block limits nor registers/warps/occupancy.
        pub fn sharedMemoryCarveout(self: Limits, requiredBytes: u64) ?u32 {
            for (self.sharedMemoryCarveouts) |bytes| if (bytes >= requiredBytes) return bytes;
            return null;
        }
    };

    /// Optional device reports; defaults are all null. hardware_query.zig uses
    /// cuDeviceGetAttribute with the CU_DEVICE_ATTRIBUTE_ suffix named per field,
    /// except globalMemoryBytes, which uses cuDeviceTotalMem_v2. Definitions:
    /// https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html
    /// https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html
    ///
    /// Null means unavailable, failed, or unrepresentable, not zero/false. Driver
    /// queries preserve numeric zero (e.g. no persisting L2) and convert nonzero
    /// boolean reports to true. No product capacity is inferred from Compute.
    ///
    /// zml/platform.zig additionally accepts positive PJRT int64 attributes
    /// `core_count`, `shared_memory_per_block_optin`, `device_memory_bytes_limit`.
    /// Each supplies a missing value or narrows the corresponding driver report
    /// with min. Zero, negative, wrongly typed and out-of-range reports are ignored.
    /// Other fields have no PJRT fallback. Overall driver-query failure leaves
    /// just these fallbacks; individual attribute failures leave individual nulls.
    ///
    /// Scope is the visible device/partition at discovery, not all physical GPUs
    /// or a live resource reservation. PJRT may further restrict the three fields
    /// above; other driver reports are not rescaled to that PJRT partition.
    pub const Properties = struct {
        /// SM count: MULTIPROCESSOR_COUNT, narrowed by PJRT `core_count`.
        /// Counts multiprocessors, not scalar CUDA cores or currently idle SMs.
        multiprocessorCount: ?u32 = null,
        /// Bytes/block: MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, narrowed by PJRT
        /// `shared_memory_per_block_optin`. init clamps both shared-memory ceilings
        /// to this value; it does not increase them or configure a function.
        sharedMemoryPerBlockOptinLimit: ?u32 = null,
        /// Bytes/block reserved by the driver: RESERVED_SHARED_MEMORY_PER_BLOCK.
        /// Additional residency cost, separate from kernel static/dynamic storage.
        reservedSharedMemoryPerBlock: ?u32 = null,
        /// L2 bytes for the visible device: L2_CACHE_SIZE; not a per-SM cache.
        l2CacheBytes: ?u32 = null,
        /// Maximum persisting-L2 set-aside bytes: MAX_PERSISTING_L2_CACHE_SIZE.
        /// Neither the current setting nor a guarantee that a hot region stays cached.
        persistingL2CacheBytes: ?u32 = null,
        /// Maximum access-policy window bytes: MAX_ACCESS_POLICY_WINDOW_SIZE.
        /// Distinct from L2 capacity; this does not configure an access policy.
        accessPolicyWindowBytes: ?u32 = null,
        /// Total device memory bytes from cuDeviceTotalMem_v2, narrowed by PJRT
        /// `device_memory_bytes_limit`. Not free memory or allocatable scratch now.
        globalMemoryBytes: ?u64 = null,
        /// Global memory bus width in bits: GLOBAL_MEMORY_BUS_WIDTH.
        /// Memory technology/transfer multiplier and sustained bandwidth are absent.
        memoryBusWidthBits: ?u32 = null,
        /// Reported typical SM clock in kHz: CLOCK_RATE. Not sampled/boost clock.
        typicalClockRateKHz: ?u32 = null,
        /// Reported peak memory clock in kHz: MEMORY_CLOCK_RATE. Not a measured
        /// data rate; bus width times this clock alone is not a bandwidth model.
        peakMemoryClockRateKHz: ?u32 = null,
        /// Device asynchronous copy-engine count: ASYNC_ENGINE_COUNT. Does not
        /// count per-SM async-copy/TMA pipelines or set kernel pipeline depth.
        asyncEngineCount: ?u32 = null,
        /// Potential for concurrent kernel execution: CONCURRENT_KERNELS.
        /// Actual overlap depends on streams, dependencies and kernel resources.
        concurrentKernels: ?bool = null,
        /// Device supports cooperative launch: COOPERATIVE_LAUNCH. Does not
        /// establish that a particular grid can reside or that a backend exposes it.
        cooperativeLaunch: ?bool = null,
        /// Device supports cluster launch: CLUSTER_LAUNCH. Separate from the static
        /// clusters feature; cluster dimensions and function restrictions are absent.
        clusterLaunch: ?bool = null,
        /// Device is integrated with host memory: INTEGRATED. No bandwidth promise.
        integrated: ?bool = null,
        /// ECC is enabled: ECC_ENABLED. Configuration, not just ECC support.
        eccEnabled: ?bool = null,

        /// Counts, sizes, clocks and asyncEngineCount take min only if both are
        /// known; otherwise null. reservedSharedMemoryPerBlock takes max only if
        /// both are known. sharedMemoryPerBlockOptinLimit is special: keep a lone
        /// known restriction, or min of two; null adds no restriction to init.
        /// concurrentKernels/cooperativeLaunch/clusterLaunch are false if either
        /// is false, true only if both are true, otherwise null. integrated and
        /// eccEnabled survive only when the optional values agree.
        fn merge(self: Properties, other: Properties) Properties {
            var result: Properties = .{};
            inline for (comptime std.meta.tags(std.meta.FieldEnum(Properties))) |field| {
                const name = comptime @tagName(field);
                const lhs = @field(self, name);
                const rhs = @field(other, name);
                @field(result, name) = switch (field) {
                    .multiprocessorCount,
                    .l2CacheBytes,
                    .persistingL2CacheBytes,
                    .accessPolicyWindowBytes,
                    .globalMemoryBytes,
                    .memoryBusWidthBits,
                    .typicalClockRateKHz,
                    .peakMemoryClockRateKHz,
                    .asyncEngineCount,
                    => if (lhs != null and rhs != null) @min(lhs.?, rhs.?) else null,
                    .sharedMemoryPerBlockOptinLimit => if (lhs != null and rhs != null) @min(lhs.?, rhs.?) else lhs orelse rhs,
                    .reservedSharedMemoryPerBlock => if (lhs != null and rhs != null) @max(lhs.?, rhs.?) else null,
                    .concurrentKernels, .cooperativeLaunch, .clusterLaunch => if (lhs == false or rhs == false) false else if (lhs == true and rhs == true) true else null,
                    // These describe configuration, not an ordered capacity.
                    .integrated, .eccEnabled => if (lhs == rhs) lhs else null,
                };
            }
            return result;
        }

        /// Clamp both per-block shared-memory ceilings; no other field changes.
        /// A null restriction leaves the static ceilings intact. Reserve bytes,
        /// the SM carveout list and kernel-specific limits are handled elsewhere.
        fn restrict(self: Properties, architectureLimits: Hardware.Limits) Hardware.Limits {
            var result = architectureLimits;
            if (self.sharedMemoryPerBlockOptinLimit) |bytes| {
                result.sharedMemoryPerBlock = @min(result.sharedMemoryPerBlock, bytes);
                result.sharedMemoryPerBlockOptin = @min(result.sharedMemoryPerBlockOptin, bytes);
            }
            return result;
        }
    };
};

/// Product-family label from Compute.architecture; not an ordered feature level.
pub const Architecture = enum {
    volta,
    turing,
    ampere,
    ada,
    hopper,
    blackwell,
};

/// Hardware mechanisms queried by Compute.supports; not kernel-backend support.
/// Tensor flags describe native operand formats, not arbitrary tensor operations,
/// conversions, accumulator types or rates. Instruction shapes, layouts, alignment,
/// participation and synchronization rules must be enforced by backend lowering.
/// Presence does not relax the workload's numerical contract. This enum is not a
/// complete CUDA feature inventory (e.g. it does not describe atomic variants).
pub const Feature = enum {
    /// Hardware-assisted asynchronous global-to-shared copy.
    asyncCopy,
    /// Hardware-assisted split arrive/wait barriers.
    splitBarrier,
    /// Native warp reduction instructions; not arbitrary operators/dtypes.
    warpReduce,
    /// L2 persistence mechanism; usable capacity/settings are separate properties.
    persistingL2,
    /// Tensor Memory Accelerator transfers; descriptor/launch support is separate.
    tma,
    /// Thread-block cluster execution; see Properties.clusterLaunch as well.
    clusters,
    /// Access to shared memory across blocks of a cluster.
    distributedSharedMemory,
    /// Cluster launch cancellation/work scheduling mechanism.
    clusterLaunchControl,
    /// Native dynamic-programming instructions rather than software expansion.
    nativeDpx,
    /// Tensor arithmetic with FP16 operands.
    tensorFp16,
    /// Tensor arithmetic with BF16 operands.
    tensorBf16,
    /// Tensor arithmetic using TF32 precision; not full-precision FP32 products.
    tensorTf32,
    /// Tensor arithmetic with FP64 operands, independent of scalar FP64 support.
    tensorFp64,
    /// Tensor arithmetic with FP8 operands; exact encodings are instruction-specific.
    tensorFp8,
    /// Tensor arithmetic with FP6 operands; exact encodings are instruction-specific.
    tensorFp6,
    /// Tensor arithmetic with FP4 operands; exact encodings are instruction-specific.
    tensorFp4,
    /// Tensor arithmetic with INT8 operands; not a promise of tcgen05 INT8.
    tensorInt8,
    /// Tensor arithmetic with INT4 operands.
    tensorInt4,
    /// Structured-sparse MMA; not an arbitrary sparse matrix acceleration flag.
    sparseMma,
    /// MMA with block-scaled operands; scale layouts/formats remain instruction rules.
    blockScaledMma,
    /// Hopper's asynchronous warpgroup MMA, with 128 participating threads.
    wgmma,
    /// Blackwell's tensor-memory MMA. FP/TF32 kinds; INT8 is separate. Does not
    /// report tensor-memory capacity or imply support on all Blackwell devices.
    tcgen05,
    /// INT8 kind of tcgen05 MMA, queried independently of tensorInt8 and tcgen05.
    tcgen05Int8,
};

/// NVIDIA's peak native results/SM/clock, not warp instructions/clock or FLOP/s.
/// Scalar multiply-add produces one result from two FLOPs. These are table rates,
/// not measured kernel rates, and are not additive across shared execution pipes.
/// No tensor, load/store, atomic or synchronization throughput is supplied.
/// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-of-native-arithmetic-instructions
pub const InstructionThroughput = struct {
    /// Packed two-lane FP16 add/multiply/multiply-add result rate (f16x2).
    /// Do not reuse for scalar FP16, BF16 or tensor-core arithmetic.
    fp16x2: u32,
    /// Scalar FP32 add/multiply/multiply-add result rate.
    fp32: u32,
    /// Scalar FP64 add/multiply/multiply-add result rate.
    fp64: u32,
    /// Ordinary 32-bit integer add/subtract; not extended-precision operations.
    int32Add: u32 = 128,
    /// Ordinary 32-bit integer multiply/multiply-add; not extended-precision rate.
    int32MultiplyAdd: u32 = 64,
    /// Native approximate FP32 reciprocal/rsqrt/log2/exp2/sin/cos table rate,
    /// not the rate of full-accuracy library functions or arbitrary expressions.
    specialFunctions: u32 = 16,
    /// Per-lane shuffle result rate, assuming converged participating threads.
    warpShuffle: u32,
};

/// Recognized compute capabilities, encoded as 10 * major + minor. This is a
/// finite hardware table, not a list of targets accepted by every installed
/// compiler. Feature sets are not monotonic in this numeric value.
///
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
pub const Compute = enum(u16) {
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

    /// Product-family label only; use supports for instruction-path decisions.
    pub fn architecture(self: Compute) Architecture {
        return switch (self) {
            .sm70, .sm72 => .volta,
            .sm75 => .turing,
            .sm80, .sm86, .sm87 => .ampere,
            .sm89 => .ada,
            .sm90 => .hopper,
            .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => .blackwell,
        };
    }

    /// Native hardware support from the explicit table, independent of compiler
    /// version, emitted target and runtime configuration. A true result is only
    /// one prerequisite: backend lowering must support and enable the instruction
    /// (e.g. WGMMA requires sm_90a) and enforce its operand and execution rules.
    /// false describes this feature on this Compute, not a failed device query.
    /// CUDA 13.4 / PTX 9.4:
    /// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
    /// https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-module-directives-target
    pub fn supports(self: Compute, feature: Feature) bool {
        return switch (feature) {
            .tensorFp16 => switch (self) {
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .tensorInt8 => switch (self) {
                .sm70, .sm72 => false,
                .sm75, .sm80, .sm86, .sm87, .sm89, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .tensorInt4 => switch (self) {
                .sm75, .sm80, .sm86, .sm87, .sm89 => true,
                .sm70, .sm72, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => false,
            },
            .asyncCopy, .splitBarrier, .warpReduce, .persistingL2, .tensorBf16, .tensorTf32, .sparseMma => switch (self) {
                .sm70, .sm72, .sm75 => false,
                .sm80, .sm86, .sm87, .sm89, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .tensorFp64 => switch (self) {
                .sm80, .sm90, .sm100 => true,
                .sm70, .sm72, .sm75, .sm86, .sm87, .sm89, .sm101, .sm103, .sm110, .sm120, .sm121 => false,
            },
            .tensorFp8 => switch (self) {
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87 => false,
                .sm89, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .tma, .clusters, .distributedSharedMemory => switch (self) {
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89 => false,
                .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .nativeDpx => switch (self) {
                .sm90, .sm100, .sm103 => true,
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm101, .sm110, .sm120, .sm121 => false,
            },
            .wgmma => switch (self) {
                .sm90 => true,
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => false,
            },
            .clusterLaunchControl => switch (self) {
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm90 => false,
                .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .tensorFp6, .tensorFp4, .blockScaledMma => switch (self) {
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm90 => false,
                .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => true,
            },
            .tcgen05 => switch (self) {
                .sm100, .sm101, .sm103, .sm110 => true,
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm90, .sm120, .sm121 => false,
            },
            .tcgen05Int8 => switch (self) {
                .sm100, .sm101, .sm110 => true,
                .sm70, .sm72, .sm75, .sm80, .sm86, .sm87, .sm89, .sm90, .sm103, .sm120, .sm121 => false,
            },
        };
    }

    /// Optional static arithmetic rates, with units/limitations in
    /// InstructionThroughput. Null means no table entry, not absent arithmetic
    /// hardware. No interpolation from a nearby Compute or architecture family.
    pub fn throughput(self: Compute) ?InstructionThroughput {
        return switch (self) {
            .sm70, .sm72, .sm87, .sm101, .sm110 => null,
            .sm75 => .{ .fp16x2 = 64, .fp32 = 64, .fp64 = 2, .warpShuffle = 16 },
            .sm80 => .{ .fp16x2 = 128, .fp32 = 64, .fp64 = 32, .warpShuffle = 32 },
            .sm86, .sm89, .sm103, .sm120, .sm121 => .{ .fp16x2 = 64, .fp32 = 128, .fp64 = 2, .warpShuffle = 32 },
            .sm90 => .{ .fp16x2 = 128, .fp32 = 128, .fp64 = 64, .warpShuffle = 32 },
            .sm100 => .{ .fp16x2 = 64, .fp32 = 128, .fp64 = 64, .warpShuffle = 32 },
        };
    }

    /// Latency of a dependent FP32 FMA in SM clock cycles, only where the cited
    /// tuning guides supply a model. Null means unknown, not zero. Not instruction
    /// issue interval, memory latency, or a complete latency-hiding/occupancy model.
    /// https://docs.nvidia.com/cuda/volta-tuning-guide/#instruction-scheduling
    /// https://docs.nvidia.com/cuda/turing-tuning-guide/#instruction-scheduling
    pub fn dependentFp32FmaLatencyCycles(self: Compute) ?u32 {
        return switch (self) {
            .sm70, .sm72, .sm75 => 4,
            .sm80, .sm86, .sm87, .sm89, .sm90, .sm100, .sm101, .sm103, .sm110, .sm120, .sm121 => null,
        };
    }

    /// Parse a recognized PJRT "major.minor" (or integer major) or `smNN`/`sm_NN`
    /// name. For example, "10" means sm100, not sm10. No whitespace trimming,
    /// signs, compiler `a`/`f` suffixes, or unknown enum values are accepted.
    /// Null is a missing identity; no nearby architecture is substituted. The
    /// platform requires a string `compute_capability`, not an integer attribute.
    pub fn parse(text: []const u8) ?Compute {
        if (std.mem.startsWith(u8, text, "sm_")) {
            const value = parseDecimal(text[3..]) orelse return null;
            return std.enums.fromInt(Compute, value);
        }
        if (std.mem.startsWith(u8, text, "sm")) return std.meta.stringToEnum(Compute, text);
        var parts = std.mem.splitScalar(u8, text, '.');
        const major = parseDecimal(parts.first()) orelse return null;
        const minor = parseDecimal(parts.next() orelse "0") orelse return null;
        if (parts.next() != null or major > 6553 or minor > 9) return null;
        return std.enums.fromInt(Compute, std.math.add(u16, major * 10, minor) catch return null);
    }

    /// Internal static table. Public callers use init(...).hardware.limits to
    /// include reported restrictions; this function itself performs no query.
    /// CUDA 13.4 technical specifications; Volta from CUDA 12.9.1. The archived
    /// per-architecture chapters also describe schedulers and cache/carveout
    /// geometry; see the Blackwell guide for its shared-memory partitions.
    /// Carveouts and allocation geometry describe this model, not current settings.
    /// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications
    /// https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#compute-capabilities
    /// https://docs.nvidia.com/cuda/blackwell-tuning-guide/#unified-shared-memory-l1-texture-cache
    fn limits(self: Compute) Hardware.Limits {
        return switch (self) {
            .sm70, .sm72 => .{
                .maxWarpsPerSm = 64,
                .maxBlocksPerSm = 32,
                .sharedMemoryAllocationBytes = 256,
                .sharedMemoryPerBlockOptin = 96 * KiB,
                .unifiedL1SharedBytes = 128 * KiB,
                .sharedMemoryCarveouts = &.{ 0, 8 * KiB, 16 * KiB, 32 * KiB, 64 * KiB, 96 * KiB },
            },
            .sm75 => .{
                .maxWarpsPerSm = 32,
                .maxBlocksPerSm = 16,
                .sharedMemoryAllocationBytes = 256,
                .sharedMemoryPerBlockOptin = 64 * KiB,
                .unifiedL1SharedBytes = 96 * KiB,
                .sharedMemoryCarveouts = &.{ 32 * KiB, 64 * KiB },
            },
            .sm80 => .{
                .maxWarpsPerSm = 64,
                .maxBlocksPerSm = 32,
                .sharedMemoryPerBlockOptin = 163 * KiB,
                .unifiedL1SharedBytes = 192 * KiB,
                .sharedMemoryCarveouts = &.{ 0, 8 * KiB, 16 * KiB, 32 * KiB, 64 * KiB, 100 * KiB, 132 * KiB, 164 * KiB },
            },
            .sm87 => .{
                .maxWarpsPerSm = 48,
                .maxBlocksPerSm = 16,
                .sharedMemoryPerBlockOptin = 163 * KiB,
                .unifiedL1SharedBytes = 192 * KiB,
                .sharedMemoryCarveouts = &.{ 0, 8 * KiB, 16 * KiB, 32 * KiB, 64 * KiB, 100 * KiB, 132 * KiB, 164 * KiB },
            },
            .sm86, .sm89, .sm120, .sm121 => .{
                .maxWarpsPerSm = 48,
                .maxBlocksPerSm = switch (self) {
                    .sm86 => 16,
                    .sm89, .sm120, .sm121 => 24,
                    .sm70, .sm72, .sm75, .sm80, .sm87, .sm90, .sm100, .sm101, .sm103, .sm110 => unreachable,
                },
                .sharedMemoryPerBlockOptin = 99 * KiB,
                .unifiedL1SharedBytes = 128 * KiB,
                .sharedMemoryCarveouts = &.{ 0, 8 * KiB, 16 * KiB, 32 * KiB, 64 * KiB, 100 * KiB },
            },
            .sm90, .sm100, .sm101, .sm103, .sm110 => .{
                .maxWarpsPerSm = if (self == .sm101 or self == .sm110) 48 else 64,
                .maxBlocksPerSm = if (self == .sm101 or self == .sm110) 24 else 32,
                .sharedMemoryPerBlockOptin = 227 * KiB,
                .unifiedL1SharedBytes = 256 * KiB,
                .sharedMemoryCarveouts = &.{ 0, 8 * KiB, 16 * KiB, 32 * KiB, 64 * KiB, 100 * KiB, 132 * KiB, 164 * KiB, 196 * KiB, 228 * KiB },
            },
        };
    }
};

fn parseDecimal(text: []const u8) ?u16 {
    if (text.len == 0) return null;
    for (text) |char| if (!std.ascii.isDigit(char)) return null;
    return std.fmt.parseInt(u16, text, 10) catch null;
}

test "CUDA capabilities describe resources available on every device" {
    const larger: Capabilities = .init(.sm90, .{
        .multiprocessorCount = 132,
        .sharedMemoryPerBlockOptinLimit = 256 * KiB,
        .reservedSharedMemoryPerBlock = KiB,
        .l2CacheBytes = 50 * 1024 * KiB,
        .globalMemoryBytes = 80 * 1024 * 1024 * KiB,
        .asyncEngineCount = 3,
        .cooperativeLaunch = true,
        .clusterLaunch = true,
        .eccEnabled = true,
    });
    const smaller: Capabilities = .init(.sm90, .{
        .multiprocessorCount = 60,
        .sharedMemoryPerBlockOptinLimit = 99 * KiB,
        .reservedSharedMemoryPerBlock = 2 * KiB,
        .l2CacheBytes = 20 * 1024 * KiB,
        .globalMemoryBytes = 20 * 1024 * 1024 * KiB,
        .asyncEngineCount = 0,
        .cooperativeLaunch = true,
        .clusterLaunch = false,
        .eccEnabled = false,
    });
    try std.testing.expectEqual(227 * KiB, larger.hardware.limits.sharedMemoryPerBlockOptin);
    const common = Capabilities.merge(larger, smaller).?;
    try std.testing.expectEqual(Compute.sm90, common.compute);
    const hardware = common.hardware;
    try std.testing.expectEqual(60, hardware.properties.multiprocessorCount);
    try std.testing.expectEqual(99 * KiB, hardware.limits.sharedMemoryPerBlockOptin);
    try std.testing.expectEqual(2 * KiB, hardware.properties.reservedSharedMemoryPerBlock);
    try std.testing.expectEqual(20 * 1024 * KiB, hardware.properties.l2CacheBytes);
    try std.testing.expectEqual(20 * 1024 * 1024 * KiB, hardware.properties.globalMemoryBytes);
    try std.testing.expectEqual(0, hardware.properties.asyncEngineCount);
    try std.testing.expectEqual(true, hardware.properties.cooperativeLaunch);
    try std.testing.expectEqual(false, hardware.properties.clusterLaunch);
    try std.testing.expectEqual(null, hardware.properties.eccEnabled);
    try std.testing.expectEqualDeep(common, Capabilities.merge(smaller, larger).?);

    const unreported: Capabilities = .init(.sm90, .{});
    try std.testing.expectEqualDeep(Compute.sm90.limits(), unreported.hardware.limits);
    const partial = Capabilities.merge(common, unreported).?;
    try std.testing.expectEqual(null, partial.hardware.properties.multiprocessorCount);
    try std.testing.expectEqual(null, partial.hardware.properties.reservedSharedMemoryPerBlock);
    try std.testing.expectEqual(null, partial.hardware.properties.cooperativeLaunch);
    try std.testing.expectEqual(false, partial.hardware.properties.clusterLaunch);
    try std.testing.expectEqual(99 * KiB, partial.hardware.limits.sharedMemoryPerBlockOptin);
    try std.testing.expectEqualDeep(partial, Capabilities.merge(larger, Capabilities.merge(smaller, unreported)).?);
    try std.testing.expectEqualDeep(partial, Capabilities.merge(Capabilities.merge(unreported, smaller), larger).?);
}

test "CUDA capability merges cannot recover a missing or differing architecture" {
    const known: Capabilities = .init(.sm90, .{ .multiprocessorCount = 132 });
    for ([_]?Capabilities{ null, .init(.sm100, .{}) }) |other| {
        const common = Capabilities.merge(known, other);
        try std.testing.expectEqual(null, common);
        try std.testing.expectEqualDeep(common, Capabilities.merge(other, known));
        try std.testing.expectEqualDeep(common, Capabilities.merge(common, known));
        try std.testing.expectEqualDeep(common, Capabilities.merge(known, common));
    }
}

test "CUDA architecture parsing rejects missing and unrecognized architectures" {
    try std.testing.expectEqual(Capabilities.Compute.sm100, Capabilities.Compute.parse("10.0"));
    try std.testing.expectEqual(Capabilities.Compute.sm100, Capabilities.Compute.parse("10"));
    try std.testing.expectEqual(Capabilities.Compute.sm103, Capabilities.Compute.parse("sm_103"));
    try std.testing.expectEqual(Capabilities.Compute.sm120, Capabilities.Compute.parse("sm120"));
    for ([_][]const u8{ "", "unknown", "0", "sm0", "sm_0", "10.0.1", "10.10", "10.", "-1", "+10", "10_0", "6553.9", "99999", "13.0", "sm_999", "sm_100a" }) |text| {
        try std.testing.expectEqual(null, Capabilities.Compute.parse(text));
    }
}
