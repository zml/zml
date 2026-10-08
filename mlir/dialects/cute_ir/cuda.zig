//! Zig bindings of NVIDIA's CUDA host dialect of the CuTe DSL (nvidia-cutlass-dsl 4.8.0), generated
//! from the dialect's .td files; do not edit.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const dialect_namespace = "cuda";
/// The C dialect handle, `mlirGetDialectHandle__cuda__`.
pub const dialect_handle = "cuda";

// Enums, generated from the dialect's .td files; values are the compiler's.

/// Enumerated type describing portability.
pub const ArchPortability = enum(u32) {
    portable = 0,
    conditional = 1,
    family = 2,
};

/// Enumerated type describing CUDA device attributes can be set through Cuda Dialect.
pub const CUDeviceAttr = enum(u32) {
    maxThreadsPerBlock = 1,
    maxBlockDimX = 2,
    maxBlockDimY = 3,
    maxBlockDimZ = 4,
    maxGridDimX = 5,
    maxGridDimY = 6,
    maxGridDimZ = 7,
    maxSharedMemoryPerBlock = 8,
    totalConstantMemory = 9,
    warpSize = 10,
    maxPitch = 11,
    maxRegistersPerBlock = 12,
    clockRate = 13,
    textureAlignment = 14,
    gpuOverlap = 15,
    multiProcessorCount = 16,
    kernelExecTimeout = 17,
    integrated = 18,
    canMapHostMemory = 19,
    computeMode = 20,
    maxTexture1DWidth = 21,
    maxTexture2DWidth = 22,
    maxTexture2DHeight = 23,
    maxTexture3DWidth = 24,
    maxTexture3DHeight = 25,
    maxTexture3DDepth = 26,
    maxTexture2DLayeredWidth = 27,
    maxTexture2DLayeredHeight = 28,
    maxTexture2DLayeredLayers = 29,
    surfaceAlignment = 30,
    concurrentKernels = 31,
    eccEnabled = 32,
    pciBusId = 33,
    pciDeviceId = 34,
    tccDriver = 35,
    memoryClockRate = 36,
    globalMemoryBusWidth = 37,
    l2CacheSize = 38,
    maxThreadsPerMultiProcessor = 39,
    asyncEngineCount = 40,
    unifiedAddressing = 41,
    maxTexture1DLayeredWidth = 42,
    maxTexture1DLayeredLayers = 43,
    maxTexture2DGatherWidth = 45,
    maxTexture2DGatherHeight = 46,
    maxTexture3DWidthAlt = 47,
    maxTexture3DHeightAlt = 48,
    maxTexture3DDepthAlt = 49,
    pciDomainId = 50,
    texturePitchAlignment = 51,
    maxTextureCubemapWidth = 52,
    maxTextureCubemapLayeredWidth = 53,
    maxTextureCubemapLayeredLayers = 54,
    maxSurface1DWidth = 55,
    maxSurface2DWidth = 56,
    maxSurface2DHeight = 57,
    maxSurface3DWidth = 58,
    maxSurface3DHeight = 59,
    maxSurface3DDepth = 60,
    maxSurface1DLayeredWidth = 61,
    maxSurface1DLayeredLayers = 62,
    maxSurface2DLayeredWidth = 63,
    maxSurface2DLayeredHeight = 64,
    maxSurface2DLayeredLayers = 65,
    maxSurfaceCubemapWidth = 66,
    maxSurfaceCubemapLayeredWidth = 67,
    maxSurfaceCubemapLayeredLayers = 68,
    maxTexture1DLinearWidth = 69,
    maxTexture2DLinearWidth = 70,
    maxTexture2DLinearHeight = 71,
    maxTexture2DLinearPitch = 72,
    maxTexture2DMipmappedWidth = 73,
    maxTexture2DMipmappedHeight = 74,
    computeCapabilityMajor = 75,
    computeCapabilityMinor = 76,
    maxTexture1DMipmappedWidth = 77,
    streamPrioritiesSupported = 78,
    globalL1CacheSupported = 79,
    localL1CacheSupported = 80,
    maxSharedMemoryPerMultiprocessor = 81,
    maxRegistersPerMultiprocessor = 82,
    managedMemory = 83,
    isMultiGpuBoard = 84,
    multiGpuBoardGroupID = 85,
    hostNativeAtomicSupported = 86,
    singleToDoublePrecisionPerfRatio = 87,
    pageableMemoryAccess = 88,
    concurrentManagedAccess = 89,
    computePreemptionSupported = 90,
    canUseHostPointerForRegisteredMem = 91,
    reserved92 = 92,
    reserved93 = 93,
    reserved94 = 94,
    cooperativeLaunch = 95,
    reserved96 = 96,
    maxSharedMemoryPerBlockOptin = 97,
    canFlushRemoteWrites = 98,
    hostRegisterSupported = 99,
    pageableMemoryAccessUsesHostPageTables = 100,
    directManagedMemAccessFromHost = 101,
    maxBlocksPerMultiprocessor = 106,
    maxPersistingL2CacheSize = 108,
    maxAccessPolicyWindowSize = 109,
    reservedSharedMemoryPerBlock = 111,
    sparseCudaArraySupported = 112,
    hostRegisterReadOnlySupported = 113,
    timelineSemaphoreInteropSupported = 114,
    memoryPoolsSupported = 115,
    gpuDirectRdmaSupported = 116,
    gpuDirectRdmaFlushWritesOptions = 117,
    gpuDirectRdmaWritesOrdering = 118,
    memoryPoolSupportedHandleTypes = 119,
    clusterLaunch = 120,
    deferredMappingCudaArraySupported = 121,
    reserved122 = 122,
    reserved123 = 123,
    reserved124 = 124,
    ipcEventSupport = 125,
    memSyncDomainCount = 126,
    reserved127 = 127,
    reserved128 = 128,
    reserved129 = 129,
    numaConfig = 130,
    numaId = 131,
    reserved132 = 132,
    mpsEnabled = 133,
    hostNumaId = 134,
    d3d12CigSupported = 135,
    vulkanCigSupported = 138,
    gpuPciDeviceId = 139,
    gpuPciSubsystemId = 140,
    reserved141 = 141,
    hostNumaMemoryPoolsSupported = 142,
    hostNumaMultinodeIpcSupported = 143,
    hostMemoryPoolsSupported = 144,
    reserved145 = 145,
    onlyPartialHostNativeAtomicSupported = 147,
    cudaDevAttrOversizedSharedMemoryPerBlock = 150,
};

/// Enumerated type describing CUDA function attributes can be set through Cuda Dialect.
pub const CUFunctionAttribute = enum(u32) {
    max_threads_per_block = 0,
    shared_size_bytes = 1,
    const_size_bytes = 2,
    local_size_bytes = 3,
    num_regs = 4,
    ptx_version = 5,
    binary_version = 6,
    cache_mode_ca = 7,
    max_dynamic_shared_size_bytes = 8,
    preferred_shared_memory_carveout = 9,
    cluster_size_must_be_set = 10,
    required_cluster_width = 11,
    required_cluster_height = 12,
    required_cluster_depth = 13,
    non_portable_cluster_size_allowed = 14,
    cluster_scheduling_policy_preference = 15,
};

/// Enumerated of the host library ABI.
pub const CudaABI = enum(u32) {
    LP64 = 0,
    ILP64 = 1,
    LLP64 = 2,
};

/// Specifies performance hint with cudaAccessPolicyWindow for hitProp and missProp members.
pub const CudaAccessProperty = enum(u32) {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2,
};

/// Enumerated type describing CUDA cluster scheduling policy.
pub const CudaClusterSchedulingPolicy = enum(u32) {
    cudaClusterSchedulingPolicyDefault = 0,
    cudaClusterSchedulingPolicySpread = 1,
    cudaClusterSchedulingPolicyLoadBalancing = 2,
};

/// Enumerated type describing CUDA function attributes can be set through Cuda Dialect.
pub const CudaFuncAttribute = enum(u32) {
    cudaFuncAttributeSharedMemoryMode = 16,
};

/// Enumerated type mirroring CudaLaunchAttributeID from CUDA Runtime API.
pub const CudaLaunchAttributeID = enum(u32) {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributePreferredClusterDimension = 11,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
    cudaLaunchAttributeNvlinkUtilCentricScheduling = 16,
};

/// Enumerated type describing CUDA memory sync domain.
pub const CudaLaunchMemSyncDomain = enum(u32) {
    cudaLaunchMemSyncDomainDefault = 0,
    cudaLaunchMemSyncDomainRemote = 1,
};

/// Enumerated type describing CUDA shared memory modes.
pub const CudaSharedMemoryMode = enum(u32) {
    cudaSharedMemoryModeDefault = 0,
    cudaSharedMemoryModeRequirePortable = 1,
    cudaSharedMemoryModeAllowNonPortable = 2,
    cudaSharedMemoryModeAllowOversizedSharedMemory = 3,
    cudaSharedMemoryModePreferOversizedSharedMemory = 4,
};

/// Enumerated type describing CUDA synchronization policy.
pub const CudaSynchronizationPolicy = enum(u32) {
    cudaSyncPolicyAuto = 0,
    cudaSyncPolicySpin = 1,
    cudaSyncPolicyYield = 2,
    cudaSyncPolicyBlockingSync = 3,
};

/// Enumerated type describing the format of an executable.
pub const ExecutableFormat = enum(u32) {
    text = 0,
    elf = 1,
    bitcode = 2,
    fatbin = 3,
};

/// Enumerated type describing the representation of an executable.
pub const ExecutableRepresentation = enum(u32) {
    sass = 0,
    ptx = 2,
    nvvm = 3,
    ltoir = 4,
    mlir = 5,
    llvm = 6,
    hostobj = 7,
};

/// Enumerated type denoting a GPU Architecture.
pub const GpuArchitecture = enum(u32) {
    sm_80 = 80,
    sm_86 = 86,
    sm_87 = 87,
    sm_88 = 88,
    sm_89 = 89,
    sm_90 = 90,
    sm_100 = 100,
    sm_101 = 101,
    sm_103 = 103,
    sm_107 = 107,
    sm_110 = 110,
    sm_120 = 120,
    sm_121 = 121,
};

/// Enumerated type describing memcpy directions.
pub const MemcpyKind = enum(u32) {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
};

/// Enumerated type describing CUDA stream capture modes.
pub const StreamCaptureModeEnum = enum(u32) {
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2,
};

// Types and attributes, generated from the dialect's .td files.

/// `#cuda.assume_kernel_attr`: Whether `cuda.launch` may assume its callee is a kernel
pub const AssumeKernelAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaAssumeKernel;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "assume_kernel_attr";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        value: bool,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaAssumeKernelAttrGet(ctx.ptr(), args.value);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValue(self: *const Self) bool {
        return c.mlirCudaAssumeKernelAttrGetValue(self.ptr());
    }
};

/// `#cuda.compute_target`: Representation, portability and architectures of a compiled executable
pub const ComputeTargetAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaComputeTarget;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "compute_target";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        representation: ExecutableRepresentation,
        portability: ArchPortability,
        archs: []const GpuArchitecture,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaComputeTargetAttrGet(ctx.ptr(), @intFromEnum(args.representation), @intFromEnum(args.portability), @intCast(args.archs.len), @ptrCast(args.archs.ptr));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getRepresentation(self: *const Self) ExecutableRepresentation {
        return @enumFromInt(c.mlirCudaComputeTargetAttrGetRepresentation(self.ptr()));
    }
    pub fn getPortability(self: *const Self) ArchPortability {
        return @enumFromInt(c.mlirCudaComputeTargetAttrGetPortability(self.ptr()));
    }
    pub fn getNumArchs(self: *const Self) usize {
        return @intCast(c.mlirCudaComputeTargetAttrGetNumArchs(self.ptr()));
    }
    pub fn getArch(self: *const Self, pos: usize) GpuArchitecture {
        return @enumFromInt(c.mlirCudaComputeTargetAttrGetArchs(self.ptr(), @intCast(pos)));
    }
};

/// `#cuda.dev_max_shared_memory_optin`: The device's opt-in maximum of shared memory per block
pub const DevMaxSharedMemoryOptinAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaDevMaxSharedMemoryOptin;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dev_max_shared_memory_optin";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaDevMaxSharedMemoryOptinAttrGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `#cuda.device_attributes`: CUDA device attributes
pub const DeviceAttributesAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaDeviceAttributes;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "device_attributes";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        values: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaDeviceAttributesAttrGet(ctx.ptr(), args.values.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValues(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCudaDeviceAttributesAttrGetValues(self.ptr()).ptr.?);
    }
};

/// `#cuda.executable`: Format of a compiled executable
pub const ExecutableAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaExecutable;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "executable";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        format: ExecutableFormat,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaExecutableAttrGet(ctx.ptr(), @intFromEnum(args.format));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getFormat(self: *const Self) ExecutableFormat {
        return @enumFromInt(c.mlirCudaExecutableAttrGetFormat(self.ptr()));
    }
};

/// `#cuda.func_attributes`: CUDA function attributes of a kernel
pub const FuncAttributesAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaFuncAttributes;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "func_attributes";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        values: *const mlir.Attribute,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaFuncAttributesAttrGet(ctx.ptr(), args.values.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getValues(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCudaFuncAttributesAttrGetValues(self.ptr()).ptr.?);
    }
};

/// `!cuda.arch_set`: `!cuda.arch_set`
pub const ArchSetType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaArchSet;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch_set";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaArchSetTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.event`: `Event` represents CUDA event
pub const EventType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaEvent;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "event";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaEventTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.graph_device_node`: `GraphDeviceNode` represents CUDA device node handle for device-side node update
pub const GraphDeviceNodeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaGraphDeviceNode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "graph_device_node";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaGraphDeviceNodeTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.graph_exec`: CUDA Graph Execution
pub const GraphExecType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaGraphExec;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "graph_exec";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaGraphExecTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.graph_node`: CUDA Graph Node handle
pub const GraphNodeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaGraphNode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "graph_node";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaGraphNodeTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.graph`: CUDA Graph
pub const GraphType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaGraph;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "graph";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaGraphTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.int`: `cuda.int` is a integer type with a bitwidth matching the underlying ABI integer bitwidth
pub const IntType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaInt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "int";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaIntTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.kernel`: CUDA Kernel handle
pub const KernelType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaKernel;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "kernel";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaKernelTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.launch_cfg`: Representation of CUDA's extensible launch configuration mirroring cuda runtime API
pub const LaunchConfigType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaLaunchConfig;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "launch_cfg";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        maxNumAttrs: u32,
    };
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaLaunchConfigTypeGet(ctx.ptr(), args.maxNumAttrs);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getMaxNumAttrs(self: *const Self) u32 {
        return c.mlirCudaLaunchConfigTypeGetMaxNumAttrs(self.ptr());
    }
};

/// `!cuda.library`: CUDA Library handle
pub const LibraryType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaLibrary;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "library";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaLibraryTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.module`: CUDA Module handle
pub const ModuleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaModule;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "module";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaModuleTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.result`: CUDA Result type
pub const ResultType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaResult;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "result";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaResultTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.device_prop`: CUDA cudaDeviceProp struct type (value-typed).
pub const RuntimeDevicePropType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeDeviceProp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.device_prop";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeDevicePropTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.dim_3`: CUDA dim3 struct type (value-typed).
pub const RuntimeDim3Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeDim3;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.dim_3";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeDim3TypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.extent`: CUDA cudaExtent struct type (value-typed).
pub const RuntimeExtentType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeExtent;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.extent";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeExtentTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.kernel_node_params`: CUDA cudaKernelNodeParams struct type (value-typed).
pub const RuntimeKernelNodeParamsType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeKernelNodeParams;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.kernel_node_params";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeKernelNodeParamsTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference`: `!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference`
pub const RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreference;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.cluster_scheduling_policy_preference";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.cooperative`: `!cuda.runtime.launch_attribute_value.cooperative`
pub const RuntimeLaunchAttributeValueCooperativeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueCooperative;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.cooperative";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueCooperativeTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.mem_sync_domain`: `!cuda.runtime.launch_attribute_value.mem_sync_domain`
pub const RuntimeLaunchAttributeValueMemSyncDomainType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueMemSyncDomain;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.mem_sync_domain";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueMemSyncDomainTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling`: `!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling`
pub const RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueNvlinkUtilCentricScheduling;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.nvlink_util_centric_scheduling";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.portable_cluster_size_mode`: `!cuda.runtime.launch_attribute_value.portable_cluster_size_mode`
pub const RuntimeLaunchAttributeValuePortableClusterSizeModeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValuePortableClusterSizeMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.portable_cluster_size_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValuePortableClusterSizeModeTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.priority`: `!cuda.runtime.launch_attribute_value.priority`
pub const RuntimeLaunchAttributeValuePriorityType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValuePriority;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.priority";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValuePriorityTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed`: `!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed`
pub const RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.programmatic_stream_serialization_allowed";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.shared_mem_carveout`: `!cuda.runtime.launch_attribute_value.shared_mem_carveout`
pub const RuntimeLaunchAttributeValueSharedMemCarveoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemCarveout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.shared_mem_carveout";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueSharedMemCarveoutTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.shared_memory_mode`: `!cuda.runtime.launch_attribute_value.shared_memory_mode`
pub const RuntimeLaunchAttributeValueSharedMemoryModeType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemoryMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.shared_memory_mode";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueSharedMemoryModeTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.launch_attribute_value.sync_policy`: `!cuda.runtime.launch_attribute_value.sync_policy`
pub const RuntimeLaunchAttributeValueSyncPolicyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeLaunchAttributeValueSyncPolicy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.launch_attribute_value.sync_policy";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeLaunchAttributeValueSyncPolicyTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.memcpy_3d_params`: CUDA cudaMemcpy3DParms struct type (value-typed).
pub const RuntimeMemcpy3DParamsType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeMemcpy3DParams;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.memcpy_3d_params";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeMemcpy3DParamsTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.pitched_ptr`: CUDA cudaPitchedPtr struct type (value-typed).
pub const RuntimePitchedPtrType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimePitchedPtr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.pitched_ptr";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimePitchedPtrTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.pos`: CUDA cudaPos struct type (value-typed).
pub const RuntimePosType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimePos;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.pos";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimePosTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.runtime.uuid`: CUDA cudaUUID_t struct type (value-typed).
pub const RuntimeUuidType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaRuntimeUuid;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "runtime.uuid";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaRuntimeUuidTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.stream`: CUDA Stream
pub const StreamType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaStream;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "stream";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaStreamTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cuda.tensor_map`: CUDA Tensor Map type
pub const TensorMapType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaTensorMap;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tensor_map";
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCudaTensorMapTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Operation builders, generated from the operations' .td.
/// `cuda.addressof`. Result types are explicit; attributes use mlir.Attribute.
pub fn addressof(ctx: *mlir.Context, result_type: *const mlir.Type, symname: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.addressof", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "symname", symname),
        },
        .location = location,
    });
}

/// `cuda.binary`. Result types are explicit; attributes use mlir.Attribute. Not verified on creation: the blocks are usually filled afterwards.
pub fn binary(ctx: *mlir.Context, sym_name: *const mlir.Attribute, binary_: *const mlir.Attribute, blocks: [1]*mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.binary", .{
        .attributes = &.{
            .named(ctx, "sym_name", sym_name),
            .named(ctx, "binary", binary_),
        },
        .blocks = &blocks,
        .verify = false,
        .location = location,
    });
}

/// `cuda.cast`. Result types are explicit; attributes use mlir.Attribute.
pub fn cast(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cuda.compose`. Result types are explicit; attributes use mlir.Attribute.
pub fn compose(ctx: *mlir.Context, fields: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.compose", .{
        .operands = .{ .flat = fields },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.decompose`. Result types are explicit; attributes use mlir.Attribute.
pub fn decompose(ctx: *mlir.Context, source: *const mlir.Value, fields_type: ?*const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var results: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (fields_type) |value| results.appendAssumeCapacity(value);
    return mlir.Operation.make(ctx, "cuda.decompose", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = results.constSlice() },
        .location = location,
    });
}

/// `cuda.device.sync`. Result types are explicit; attributes use mlir.Attribute.
pub fn device_sync(ctx: *mlir.Context, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.device.sync", .{
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.event.create`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_create(ctx: *mlir.Context, Flags: *const mlir.Value, cuda_result_type: *const mlir.Type, phEvent_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.create", .{
        .operands = .{ .flat = &.{Flags} },
        .results = .{ .flat = &.{ cuda_result_type, phEvent_type } },
        .location = location,
    });
}

/// `cuda.driver.event.destroy`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_destroy(ctx: *mlir.Context, hEvent: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.destroy", .{
        .operands = .{ .flat = &.{hEvent} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.event.elapsed_time`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_elapsed_time(ctx: *mlir.Context, hStart: *const mlir.Value, hEnd: *const mlir.Value, cuda_result_type: *const mlir.Type, pMilliseconds_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.elapsed_time", .{
        .operands = .{ .flat = &.{ hStart, hEnd } },
        .results = .{ .flat = &.{ cuda_result_type, pMilliseconds_type } },
        .location = location,
    });
}

/// `cuda.driver.event.query`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_query(ctx: *mlir.Context, hEvent: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.query", .{
        .operands = .{ .flat = &.{hEvent} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.event.record`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_record(ctx: *mlir.Context, hEvent: *const mlir.Value, hStream: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.record", .{
        .operands = .{ .flat = &.{ hEvent, hStream } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.event.record_with_flags`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_record_with_flags(ctx: *mlir.Context, hEvent: *const mlir.Value, hStream: *const mlir.Value, flags: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.record_with_flags", .{
        .operands = .{ .flat = &.{ hEvent, hStream, flags } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.event.synchronize`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_event_synchronize(ctx: *mlir.Context, hEvent: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.event.synchronize", .{
        .operands = .{ .flat = &.{hEvent} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.library.get_global`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_get_global(ctx: *mlir.Context, library: *const mlir.Value, name_arg: *const mlir.Value, cuda_result_type: *const mlir.Type, dptr_type: *const mlir.Type, bytes_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.get_global", .{
        .operands = .{ .flat = &.{ library, name_arg } },
        .results = .{ .flat = &.{ cuda_result_type, dptr_type, bytes_type } },
        .location = location,
    });
}

/// `cuda.driver.library.get_kernel`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_get_kernel(ctx: *mlir.Context, library: *const mlir.Value, name_arg: *const mlir.Value, cuda_result_type: *const mlir.Type, pKernel_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.get_kernel", .{
        .operands = .{ .flat = &.{ library, name_arg } },
        .results = .{ .flat = &.{ cuda_result_type, pKernel_type } },
        .location = location,
    });
}

/// `cuda.driver.library.get_kernel_count`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_get_kernel_count(ctx: *mlir.Context, lib: *const mlir.Value, cuda_result_type: *const mlir.Type, count_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.get_kernel_count", .{
        .operands = .{ .flat = &.{lib} },
        .results = .{ .flat = &.{ cuda_result_type, count_type } },
        .location = location,
    });
}

/// `cuda.driver.library.get_managed`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_get_managed(ctx: *mlir.Context, library: *const mlir.Value, name_arg: *const mlir.Value, cuda_result_type: *const mlir.Type, dptr_type: *const mlir.Type, bytes_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.get_managed", .{
        .operands = .{ .flat = &.{ library, name_arg } },
        .results = .{ .flat = &.{ cuda_result_type, dptr_type, bytes_type } },
        .location = location,
    });
}

/// `cuda.driver.library.get_module`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_get_module(ctx: *mlir.Context, library: *const mlir.Value, cuda_result_type: *const mlir.Type, pMod_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.get_module", .{
        .operands = .{ .flat = &.{library} },
        .results = .{ .flat = &.{ cuda_result_type, pMod_type } },
        .location = location,
    });
}

/// `cuda.driver.library.get_unified_function`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_get_unified_function(ctx: *mlir.Context, library: *const mlir.Value, symbol: *const mlir.Value, cuda_result_type: *const mlir.Type, fptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.get_unified_function", .{
        .operands = .{ .flat = &.{ library, symbol } },
        .results = .{ .flat = &.{ cuda_result_type, fptr_type } },
        .location = location,
    });
}

/// `cuda.driver.library.load_data`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_load_data(ctx: *mlir.Context, code_arg: *const mlir.Value, jitOptions: *const mlir.Value, jitOptionsValues: *const mlir.Value, numJitOptions: *const mlir.Value, libraryOptions: *const mlir.Value, libraryOptionValues: *const mlir.Value, numLibraryOptions: *const mlir.Value, cuda_result_type: *const mlir.Type, library_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.load_data", .{
        .operands = .{ .flat = &.{ code_arg, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions } },
        .results = .{ .flat = &.{ cuda_result_type, library_type } },
        .location = location,
    });
}

/// `cuda.driver.library.load_from_file`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_load_from_file(ctx: *mlir.Context, fileName: *const mlir.Value, jitOptions: *const mlir.Value, jitOptionsValues: *const mlir.Value, numJitOptions: *const mlir.Value, libraryOptions: *const mlir.Value, libraryOptionValues: *const mlir.Value, numLibraryOptions: *const mlir.Value, cuda_result_type: *const mlir.Type, library_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.load_from_file", .{
        .operands = .{ .flat = &.{ fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions } },
        .results = .{ .flat = &.{ cuda_result_type, library_type } },
        .location = location,
    });
}

/// `cuda.driver.library.unload`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_library_unload(ctx: *mlir.Context, library: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.library.unload", .{
        .operands = .{ .flat = &.{library} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.driver.tensor_map.encode_im2col`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_tensor_map_encode_im2col(ctx: *mlir.Context, tensorDataType: *const mlir.Value, tensorRank: *const mlir.Value, globalAddress: *const mlir.Value, globalDim: *const mlir.Value, globalStrides: *const mlir.Value, pixelBoxLowerCorner: *const mlir.Value, pixelBoxUpperCorner: *const mlir.Value, channelsPerPixel: *const mlir.Value, pixelsPerColumn: *const mlir.Value, elementStrides: *const mlir.Value, interleave: *const mlir.Value, swizzle: *const mlir.Value, l2Promotion: *const mlir.Value, oobFill: *const mlir.Value, cuda_result_type: *const mlir.Type, tensorMap_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.tensor_map.encode_im2col", .{
        .operands = .{ .flat = &.{ tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion, oobFill } },
        .results = .{ .flat = &.{ cuda_result_type, tensorMap_type } },
        .location = location,
    });
}

/// `cuda.driver.tensor_map.encode_im2col_wide`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_tensor_map_encode_im2col_wide(ctx: *mlir.Context, tensorDataType: *const mlir.Value, tensorRank: *const mlir.Value, globalAddress: *const mlir.Value, globalDim: *const mlir.Value, globalStrides: *const mlir.Value, pixelBoxLowerCornerWidth: *const mlir.Value, pixelBoxUpperCornerWidth: *const mlir.Value, channelsPerPixel: *const mlir.Value, pixelsPerColumn: *const mlir.Value, elementStrides: *const mlir.Value, interleave: *const mlir.Value, mode: *const mlir.Value, swizzle: *const mlir.Value, l2Promotion: *const mlir.Value, oobFill: *const mlir.Value, cuda_result_type: *const mlir.Type, tensorMap_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.tensor_map.encode_im2col_wide", .{
        .operands = .{ .flat = &.{ tensorDataType, tensorRank, globalAddress, globalDim, globalStrides, pixelBoxLowerCornerWidth, pixelBoxUpperCornerWidth, channelsPerPixel, pixelsPerColumn, elementStrides, interleave, mode, swizzle, l2Promotion, oobFill } },
        .results = .{ .flat = &.{ cuda_result_type, tensorMap_type } },
        .location = location,
    });
}

/// `cuda.driver.tensor_map.replace_address`. Result types are explicit; attributes use mlir.Attribute.
pub fn driver_tensor_map_replace_address(ctx: *mlir.Context, globalAddress: *const mlir.Value, cuda_result_type: *const mlir.Type, tensorMap_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.driver.tensor_map.replace_address", .{
        .operands = .{ .flat = &.{globalAddress} },
        .results = .{ .flat = &.{ cuda_result_type, tensorMap_type } },
        .location = location,
    });
}

/// `cuda.event.create`. Result types are explicit; attributes use mlir.Attribute.
pub fn event_create(ctx: *mlir.Context, result_type: *const mlir.Type, event_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.event.create", .{
        .results = .{ .flat = &.{ result_type, event_type } },
        .location = location,
    });
}

/// `cuda.event.destroy`. Result types are explicit; attributes use mlir.Attribute.
pub fn event_destroy(ctx: *mlir.Context, event: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.event.destroy", .{
        .operands = .{ .flat = &.{event} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.event.elapsed_time`. Result types are explicit; attributes use mlir.Attribute.
pub fn event_elapsed_time(ctx: *mlir.Context, start_event: *const mlir.Value, end_event: *const mlir.Value, cuda_result_type: *const mlir.Type, elapsed_time_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.event.elapsed_time", .{
        .operands = .{ .flat = &.{ start_event, end_event } },
        .results = .{ .flat = &.{ cuda_result_type, elapsed_time_type } },
        .location = location,
    });
}

/// `cuda.event.record`. Result types are explicit; attributes use mlir.Attribute.
pub fn event_record(ctx: *mlir.Context, event: *const mlir.Value, stream: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.event.record", .{
        .operands = .{ .flat = if (stream) |value| &.{ event, value } else &.{event} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.func`. Result types are explicit; attributes use mlir.Attribute. Not verified on creation: the blocks are usually filled afterwards.
pub fn func(ctx: *mlir.Context, sym_name: *const mlir.Attribute, function_type: *const mlir.Attribute, arg_attrs: ?*const mlir.Attribute, res_attrs: ?*const mlir.Attribute, blocks: [1]*mlir.Block, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "sym_name", sym_name));
    attributes.appendAssumeCapacity(.named(ctx, "function_type", function_type));
    if (arg_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", value));
    if (res_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "res_attrs", value));
    return mlir.Operation.make(ctx, "cuda.func", .{
        .attributes = attributes.constSlice(),
        .blocks = &blocks,
        .verify = false,
        .location = location,
    });
}

/// `cuda.func.get_name`. Result types are explicit; attributes use mlir.Attribute.
pub fn func_get_name(ctx: *mlir.Context, kernel_: *const mlir.Value, result_type: *const mlir.Type, name_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.func.get_name", .{
        .operands = .{ .flat = &.{kernel_} },
        .results = .{ .flat = &.{ result_type, name_type } },
        .location = location,
    });
}

/// `cuda.graph.destroy`. Result types are explicit; attributes use mlir.Attribute.
pub fn graph_destroy(ctx: *mlir.Context, graph: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.graph.destroy", .{
        .operands = .{ .flat = &.{graph} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.graph.instantiate`. Result types are explicit; attributes use mlir.Attribute.
pub fn graph_instantiate(ctx: *mlir.Context, graph: *const mlir.Value, result_type: *const mlir.Type, graph_exec_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.graph.instantiate", .{
        .operands = .{ .flat = &.{graph} },
        .results = .{ .flat = &.{ result_type, graph_exec_type } },
        .location = location,
    });
}

/// `cuda.graph.launch`. Result types are explicit; attributes use mlir.Attribute.
pub fn graph_launch(ctx: *mlir.Context, graph_exec: *const mlir.Value, stream: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.graph.launch", .{
        .operands = .{ .flat = &.{ graph_exec, stream } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.graph_exec.destroy`. Result types are explicit; attributes use mlir.Attribute.
pub fn graph_exec_destroy(ctx: *mlir.Context, graph_exec: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.graph_exec.destroy", .{
        .operands = .{ .flat = &.{graph_exec} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.kernel`. Result types are explicit; attributes use mlir.Attribute. Not verified on creation: the blocks are usually filled afterwards.
pub fn kernel(ctx: *mlir.Context, sym_name: *const mlir.Attribute, function_type: *const mlir.Attribute, arg_attrs: ?*const mlir.Attribute, res_attrs: ?*const mlir.Attribute, cu_func_attrs: ?*const mlir.Attribute, blocks: [1]*mlir.Block, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "sym_name", sym_name));
    attributes.appendAssumeCapacity(.named(ctx, "function_type", function_type));
    if (arg_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", value));
    if (res_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "res_attrs", value));
    if (cu_func_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "cu_func_attrs", value));
    return mlir.Operation.make(ctx, "cuda.kernel", .{
        .attributes = attributes.constSlice(),
        .blocks = &blocks,
        .verify = false,
        .location = location,
    });
}

/// `cuda.launch`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch(ctx: *mlir.Context, grid: []const *const mlir.Value, block: []const *const mlir.Value, dynamicSmem: *const mlir.Value, cudaStream: *const mlir.Value, inputs: []const *const mlir.Value, cuda_result_type: *const mlir.Type, callee: *const mlir.Attribute, arg_attrs: ?*const mlir.Attribute, res_attrs: ?*const mlir.Attribute, assume_kernel_attr: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "callee", callee));
    if (arg_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", value));
    if (res_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "res_attrs", value));
    if (assume_kernel_attr) |value| attributes.appendAssumeCapacity(.named(ctx, "assume_kernel_attr", value));
    return mlir.Operation.make(ctx, "cuda.launch", .{
        .operands = .{ .variadic = &.{
            grid,
            block,
            &.{dynamicSmem},
            &.{cudaStream},
            inputs,
        } },
        .results = .{ .flat = &.{cuda_result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cuda.launch_cfg.access_policy_window`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_access_policy_window(ctx: *mlir.Context, launch_cfg: *const mlir.Value, base_ptr: *const mlir.Value, hitRatio: *const mlir.Value, num_bytes: *const mlir.Value, hitProp: *const mlir.Attribute, missProp: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.access_policy_window", .{
        .operands = .{ .flat = &.{ launch_cfg, base_ptr, hitRatio, num_bytes } },
        .attributes = &.{
            .named(ctx, "hitProp", hitProp),
            .named(ctx, "missProp", missProp),
        },
        .location = location,
    });
}

/// `cuda.launch_cfg.cluster_dim`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_cluster_dim(ctx: *mlir.Context, launch_cfg: *const mlir.Value, clusterDimX: *const mlir.Value, clusterDimY: *const mlir.Value, clusterDimZ: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.cluster_dim", .{
        .operands = .{ .flat = &.{ launch_cfg, clusterDimX, clusterDimY, clusterDimZ } },
        .location = location,
    });
}

/// `cuda.launch_cfg.cluster_scheduling_policy`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_cluster_scheduling_policy(ctx: *mlir.Context, launch_cfg: *const mlir.Value, policy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.cluster_scheduling_policy", .{
        .operands = .{ .flat = &.{launch_cfg} },
        .attributes = &.{
            .named(ctx, "policy", policy),
        },
        .location = location,
    });
}

/// `cuda.launch_cfg.cooperative`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_cooperative(ctx: *mlir.Context, launch_cfg: *const mlir.Value, cooperative: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.cooperative", .{
        .operands = .{ .flat = &.{ launch_cfg, cooperative } },
        .location = location,
    });
}

/// `cuda.launch_cfg.create`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_create(ctx: *mlir.Context, blockDimX: *const mlir.Value, blockDimY: *const mlir.Value, blockDimZ: *const mlir.Value, dynamicSmemBytes: *const mlir.Value, gridDimX: *const mlir.Value, gridDimY: *const mlir.Value, gridDimZ: *const mlir.Value, stream: *const mlir.Value, result_type: *const mlir.Type, maxNumAttrs: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.create", .{
        .operands = .{ .flat = &.{ blockDimX, blockDimY, blockDimZ, dynamicSmemBytes, gridDimX, gridDimY, gridDimZ, stream } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "maxNumAttrs", maxNumAttrs),
        },
        .location = location,
    });
}

/// `cuda.launch_cfg.create_from_stream`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_create_from_stream(ctx: *mlir.Context, stream: *const mlir.Value, result_type: *const mlir.Type, maxNumAttrs: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.create_from_stream", .{
        .operands = .{ .flat = &.{stream} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "maxNumAttrs", maxNumAttrs),
        },
        .location = location,
    });
}

/// `cuda.launch_cfg.device_updatable_kernel_node`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_device_updatable_kernel_node(ctx: *mlir.Context, launch_cfg: *const mlir.Value, devNode: *const mlir.Value, deviceUpdatable: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.device_updatable_kernel_node", .{
        .operands = .{ .flat = &.{ launch_cfg, devNode, deviceUpdatable } },
        .location = location,
    });
}

/// `cuda.launch_cfg.grid_dimensions`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_grid_dimensions(ctx: *mlir.Context, launch_cfg: *const mlir.Value, gridDimX: *const mlir.Value, gridDimY: *const mlir.Value, gridDimZ: *const mlir.Value, blockDimX: *const mlir.Value, blockDimY: *const mlir.Value, blockDimZ: *const mlir.Value, dynamicSmemBytes: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.grid_dimensions", .{
        .operands = .{ .flat = &.{ launch_cfg, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, dynamicSmemBytes } },
        .location = location,
    });
}

/// `cuda.launch_cfg.launch_completion_event`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_launch_completion_event(ctx: *mlir.Context, launch_cfg: *const mlir.Value, event: *const mlir.Value, flags: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.launch_completion_event", .{
        .operands = .{ .flat = &.{ launch_cfg, event, flags } },
        .location = location,
    });
}

/// `cuda.launch_cfg.mem_sync_domain`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_mem_sync_domain(ctx: *mlir.Context, launch_cfg: *const mlir.Value, memSyncDomain: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.mem_sync_domain", .{
        .operands = .{ .flat = &.{launch_cfg} },
        .attributes = &.{
            .named(ctx, "memSyncDomain", memSyncDomain),
        },
        .location = location,
    });
}

/// `cuda.launch_cfg.mem_sync_domain_map`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_mem_sync_domain_map(ctx: *mlir.Context, launch_cfg: *const mlir.Value, domain_default: *const mlir.Value, domain_remote: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.mem_sync_domain_map", .{
        .operands = .{ .flat = &.{ launch_cfg, domain_default, domain_remote } },
        .location = location,
    });
}

/// `cuda.launch_cfg.nvlink_util_centric_scheduling`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_nvlink_util_centric_scheduling(ctx: *mlir.Context, launch_cfg: *const mlir.Value, nvlinkSchedulingEnabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.nvlink_util_centric_scheduling", .{
        .operands = .{ .flat = &.{ launch_cfg, nvlinkSchedulingEnabled } },
        .location = location,
    });
}

/// `cuda.launch_cfg.preferred_cluster_dim`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_preferred_cluster_dim(ctx: *mlir.Context, launch_cfg: *const mlir.Value, preferredClusterDimX: *const mlir.Value, preferredClusterDimY: *const mlir.Value, preferredClusterDimZ: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.preferred_cluster_dim", .{
        .operands = .{ .flat = &.{ launch_cfg, preferredClusterDimX, preferredClusterDimY, preferredClusterDimZ } },
        .location = location,
    });
}

/// `cuda.launch_cfg.print`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_print(ctx: *mlir.Context, launch_cfg: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.print", .{
        .operands = .{ .flat = &.{launch_cfg} },
        .location = location,
    });
}

/// `cuda.launch_cfg.priority`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_priority(ctx: *mlir.Context, launch_cfg: *const mlir.Value, priority: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.priority", .{
        .operands = .{ .flat = &.{ launch_cfg, priority } },
        .location = location,
    });
}

/// `cuda.launch_cfg.programmatic_event`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_programmatic_event(ctx: *mlir.Context, launch_cfg: *const mlir.Value, event: *const mlir.Value, flags: *const mlir.Value, triggerAtBlockStart: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.programmatic_event", .{
        .operands = .{ .flat = &.{ launch_cfg, event, flags, triggerAtBlockStart } },
        .location = location,
    });
}

/// `cuda.launch_cfg.programmatic_stream_serialization_allowed`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_programmatic_stream_serialization_allowed(ctx: *mlir.Context, launch_cfg: *const mlir.Value, programmaticStreamSerializationAllowed: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.programmatic_stream_serialization_allowed", .{
        .operands = .{ .flat = &.{ launch_cfg, programmaticStreamSerializationAllowed } },
        .location = location,
    });
}

/// `cuda.launch_cfg.shared_mem_carveout`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_shared_mem_carveout(ctx: *mlir.Context, launch_cfg: *const mlir.Value, sharedMemCarveout: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.shared_mem_carveout", .{
        .operands = .{ .flat = &.{ launch_cfg, sharedMemCarveout } },
        .location = location,
    });
}

/// `cuda.launch_cfg.sync_policy`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_cfg_sync_policy(ctx: *mlir.Context, launch_cfg: *const mlir.Value, syncPolicy: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.launch_cfg.sync_policy", .{
        .operands = .{ .flat = &.{launch_cfg} },
        .attributes = &.{
            .named(ctx, "syncPolicy", syncPolicy),
        },
        .location = location,
    });
}

/// `cuda.launch_ex`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_ex(ctx: *mlir.Context, launch_cfg: *const mlir.Value, inputs: []const *const mlir.Value, cuda_result_type: *const mlir.Type, callee: *const mlir.Attribute, arg_attrs: ?*const mlir.Attribute, res_attrs: ?*const mlir.Attribute, assume_kernel_attr: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 256) = .empty;
    operands.appendAssumeCapacity(launch_cfg);
    operands.appendSliceAssumeCapacity(inputs);
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "callee", callee));
    if (arg_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", value));
    if (res_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "res_attrs", value));
    if (assume_kernel_attr) |value| attributes.appendAssumeCapacity(.named(ctx, "assume_kernel_attr", value));
    return mlir.Operation.make(ctx, "cuda.launch_ex", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{cuda_result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cuda.launch_kernel`. Result types are explicit; attributes use mlir.Attribute.
pub fn launch_kernel(ctx: *mlir.Context, kernel_: *const mlir.Value, grid: []const *const mlir.Value, block: []const *const mlir.Value, dynamicSmem: *const mlir.Value, cudaStream: *const mlir.Value, kernel_args: *const mlir.Value, cuda_result_type: *const mlir.Type, arg_attrs: ?*const mlir.Attribute, res_attrs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (arg_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "arg_attrs", value));
    if (res_attrs) |value| attributes.appendAssumeCapacity(.named(ctx, "res_attrs", value));
    return mlir.Operation.make(ctx, "cuda.launch_kernel", .{
        .operands = .{ .variadic = &.{
            &.{kernel_},
            grid,
            block,
            &.{dynamicSmem},
            &.{cudaStream},
            &.{kernel_args},
        } },
        .results = .{ .flat = &.{cuda_result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cuda.library.get_kernel`. Result types are explicit; attributes use mlir.Attribute.
pub fn library_get_kernel(ctx: *mlir.Context, library: *const mlir.Value, kernel_name: *const mlir.Value, result_type: *const mlir.Type, kernel_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.library.get_kernel", .{
        .operands = .{ .flat = &.{ library, kernel_name } },
        .results = .{ .flat = &.{ result_type, kernel_type } },
        .location = location,
    });
}

/// `cuda.library.load_data`. Result types are explicit; attributes use mlir.Attribute.
pub fn library_load_data(ctx: *mlir.Context, data: *const mlir.Value, result_type: *const mlir.Type, library_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.library.load_data", .{
        .operands = .{ .flat = &.{data} },
        .results = .{ .flat = &.{ result_type, library_type } },
        .location = location,
    });
}

/// `cuda.library.unload`. Result types are explicit; attributes use mlir.Attribute.
pub fn library_unload(ctx: *mlir.Context, library: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.library.unload", .{
        .operands = .{ .flat = &.{library} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.memory.alloc`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_alloc(ctx: *mlir.Context, bytes: *const mlir.Value, cuda_result_type: *const mlir.Type, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.alloc", .{
        .operands = .{ .flat = &.{bytes} },
        .results = .{ .flat = &.{ cuda_result_type, ptr_type } },
        .location = location,
    });
}

/// `cuda.memory.alloc_async`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_alloc_async(ctx: *mlir.Context, bytes: *const mlir.Value, stream: *const mlir.Value, cuda_result_type: *const mlir.Type, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.alloc_async", .{
        .operands = .{ .flat = &.{ bytes, stream } },
        .results = .{ .flat = &.{ cuda_result_type, ptr_type } },
        .location = location,
    });
}

/// `cuda.memory.alloc_host`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_alloc_host(ctx: *mlir.Context, bytes: *const mlir.Value, cuda_result_type: *const mlir.Type, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.alloc_host", .{
        .operands = .{ .flat = &.{bytes} },
        .results = .{ .flat = &.{ cuda_result_type, ptr_type } },
        .location = location,
    });
}

/// `cuda.memory.alloc_managed`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_alloc_managed(ctx: *mlir.Context, bytes: *const mlir.Value, cuda_result_type: *const mlir.Type, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.alloc_managed", .{
        .operands = .{ .flat = &.{bytes} },
        .results = .{ .flat = &.{ cuda_result_type, ptr_type } },
        .location = location,
    });
}

/// `cuda.memory.copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_copy(ctx: *mlir.Context, src: *const mlir.Value, dst: *const mlir.Value, bytes: *const mlir.Value, cuda_result_type: *const mlir.Type, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.copy", .{
        .operands = .{ .flat = &.{ src, dst, bytes } },
        .results = .{ .flat = &.{cuda_result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `cuda.memory.copy_async`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_copy_async(ctx: *mlir.Context, src: *const mlir.Value, dst: *const mlir.Value, bytes: *const mlir.Value, stream: ?*const mlir.Value, cuda_result_type: *const mlir.Type, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.copy_async", .{
        .operands = .{ .flat = if (stream) |value| &.{ src, dst, bytes, value } else &.{ src, dst, bytes } },
        .results = .{ .flat = &.{cuda_result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `cuda.memory.free`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_free(ctx: *mlir.Context, ptr: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.free", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.memory.free_async`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_free_async(ctx: *mlir.Context, ptr: *const mlir.Value, stream: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.free_async", .{
        .operands = .{ .flat = &.{ ptr, stream } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.memory.free_host`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_free_host(ctx: *mlir.Context, ptr: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.free_host", .{
        .operands = .{ .flat = &.{ptr} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.memory.set`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_set(ctx: *mlir.Context, ptr: *const mlir.Value, value_: *const mlir.Value, bytes: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.set", .{
        .operands = .{ .flat = &.{ ptr, value_, bytes } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.memory.set_async`. Result types are explicit; attributes use mlir.Attribute.
pub fn memory_set_async(ctx: *mlir.Context, ptr: *const mlir.Value, value_: *const mlir.Value, bytes: *const mlir.Value, stream: ?*const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.memory.set_async", .{
        .operands = .{ .flat = if (stream) |value| &.{ ptr, value_, bytes, value } else &.{ ptr, value_, bytes } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.occupancy_max_active_blocks_per_multiprocessor`. Result types are explicit; attributes use mlir.Attribute.
pub fn occupancy_max_active_blocks_per_multiprocessor(ctx: *mlir.Context, block_size: *const mlir.Value, dynamic_smem_size: *const mlir.Value, res_type: *const mlir.Type, num_blocks_type: *const mlir.Type, func_: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.occupancy_max_active_blocks_per_multiprocessor", .{
        .operands = .{ .flat = &.{ block_size, dynamic_smem_size } },
        .results = .{ .flat = &.{ res_type, num_blocks_type } },
        .attributes = &.{
            .named(ctx, "func", func_),
        },
        .location = location,
    });
}

/// `cuda.profiler.start`. Result types are explicit; attributes use mlir.Attribute.
pub fn profiler_start(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.profiler.start", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.profiler.stop`. Result types are explicit; attributes use mlir.Attribute.
pub fn profiler_stop(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.profiler.stop", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.result.assert_success`. Result types are explicit; attributes use mlir.Attribute.
pub fn result_assert_success(ctx: *mlir.Context, cuda_result: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.result.assert_success", .{
        .operands = .{ .flat = &.{cuda_result} },
        .location = location,
    });
}

/// `cuda.result.failed`. Result types are explicit; attributes use mlir.Attribute.
pub fn result_failed(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.result.failed", .{
        .location = location,
    });
}

/// `cuda.result.print_on_error`. Result types are explicit; attributes use mlir.Attribute.
pub fn result_print_on_error(ctx: *mlir.Context, cuda_result: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.result.print_on_error", .{
        .operands = .{ .flat = &.{cuda_result} },
        .location = location,
    });
}

/// `cuda.result.succeeded`. Result types are explicit; attributes use mlir.Attribute.
pub fn result_succeeded(ctx: *mlir.Context, cuda_result: *const mlir.Value, success_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.result.succeeded", .{
        .operands = .{ .flat = &.{cuda_result} },
        .results = .{ .flat = &.{success_type} },
        .location = location,
    });
}

/// `cuda.return`. Result types are explicit; attributes use mlir.Attribute. Not verified on creation: a terminator is only one in its block.
pub fn @"return"(ctx: *mlir.Context, inputs: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.return", .{
        .operands = .{ .flat = inputs },
        .verify = false,
        .location = location,
    });
}

/// `cuda.return_if_error`. Result types are explicit; attributes use mlir.Attribute.
pub fn return_if_error(ctx: *mlir.Context, inputs: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.return_if_error", .{
        .operands = .{ .flat = inputs },
        .location = location,
    });
}

/// `cuda.runtime.device.properties`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_properties(ctx: *mlir.Context, device: *const mlir.Value, cuda_result_type: *const mlir.Type, prop_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device.properties", .{
        .operands = .{ .flat = &.{device} },
        .results = .{ .flat = &.{ cuda_result_type, prop_type } },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_access_policy_max_window_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_access_policy_max_window_size(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_access_policy_max_window_size", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_async_engine_count`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_async_engine_count(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_async_engine_count", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_can_map_host_memory`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_can_map_host_memory(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_can_map_host_memory", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_can_use_host_pointer_for_registered_mem`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_can_use_host_pointer_for_registered_mem(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_can_use_host_pointer_for_registered_mem", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_cluster_launch`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_cluster_launch(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_cluster_launch", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_compute_preemption_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_compute_preemption_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_compute_preemption_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_concurrent_kernels`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_concurrent_kernels(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_concurrent_kernels", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_concurrent_managed_access`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_concurrent_managed_access(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_concurrent_managed_access", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_cooperative_launch`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_cooperative_launch(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_cooperative_launch", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_deferred_mapping_cuda_array_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_deferred_mapping_cuda_array_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_deferred_mapping_cuda_array_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_device_numa_config`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_device_numa_config(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_device_numa_config", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_device_numa_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_device_numa_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_device_numa_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_direct_managed_mem_access_from_host`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_direct_managed_mem_access_from_host(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_direct_managed_mem_access_from_host", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_ecc_enabled`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_ecc_enabled(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_ecc_enabled", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_global_l1cache_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_global_l1cache_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_global_l1cache_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_gpu_direct_rdma_flush_writes_options`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_gpu_direct_rdma_flush_writes_options(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_gpu_direct_rdma_flush_writes_options", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_gpu_direct_rdma_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_gpu_direct_rdma_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_gpu_direct_rdma_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_gpu_direct_rdma_writes_ordering`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_gpu_direct_rdma_writes_ordering(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_gpu_direct_rdma_writes_ordering", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_gpu_pci_device_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_gpu_pci_device_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_gpu_pci_device_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_gpu_pci_subsystem_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_gpu_pci_subsystem_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_gpu_pci_subsystem_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_host_native_atomic_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_host_native_atomic_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_host_native_atomic_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_host_numa_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_host_numa_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_host_numa_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_host_numa_multinode_ipc_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_host_numa_multinode_ipc_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_host_numa_multinode_ipc_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_host_register_read_only_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_host_register_read_only_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_host_register_read_only_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_host_register_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_host_register_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_host_register_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_integrated`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_integrated(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_integrated", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_ipc_event_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_ipc_event_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_ipc_event_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_is_multi_gpu_board`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_is_multi_gpu_board(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_is_multi_gpu_board", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_l_2cache_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_l_2cache_size(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_l_2cache_size", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_local_l1cache_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_local_l1cache_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_local_l1cache_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_luid`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_luid(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_luid", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_luid_device_node_mask`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_luid_device_node_mask(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_luid_device_node_mask", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_major`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_major(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_major", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_managed_memory`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_managed_memory(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_managed_memory", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_blocks_per_multi_processor`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_blocks_per_multi_processor(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_blocks_per_multi_processor", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_grid_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_grid_size(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_grid_size", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_1d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_1d(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_1d", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_1d_layered`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_1d_layered(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_1d_layered", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_2d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_2d(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_2d", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_2d_layered`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_2d_layered(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_2d_layered", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_3d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_3d(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_3d", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_cubemap`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_cubemap(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_cubemap", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_surface_cubemap_layered`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_surface_cubemap_layered(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_surface_cubemap_layered", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_1d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_1d(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_1d", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_1d_layered`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_1d_layered(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_1d_layered", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_1d_mipmap`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_1d_mipmap(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_1d_mipmap", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_2d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_2d(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_2d", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_2d_gather`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_2d_gather(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_2d_gather", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_2d_layered`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_2d_layered(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_2d_layered", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_2d_linear`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_2d_linear(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_2d_linear", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_2d_mipmap`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_2d_mipmap(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_2d_mipmap", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_3d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_3d(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_3d", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_3d_alt`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_3d_alt(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_3d_alt", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_cubemap`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_cubemap(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_cubemap", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_texture_cubemap_layered`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_texture_cubemap_layered(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_texture_cubemap_layered", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_threads_dim`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_threads_dim(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_threads_dim", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_threads_per_block`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_threads_per_block(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_threads_per_block", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_max_threads_per_multi_processor`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_max_threads_per_multi_processor(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_max_threads_per_multi_processor", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_mem_pitch`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_mem_pitch(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_mem_pitch", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_memory_bus_width`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_memory_bus_width(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_memory_bus_width", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_memory_pool_supported_handle_types`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_memory_pool_supported_handle_types(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_memory_pool_supported_handle_types", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_memory_pools_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_memory_pools_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_memory_pools_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_minor`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_minor(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_minor", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_mps_enabled`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_mps_enabled(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_mps_enabled", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_multi_gpu_board_group_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_multi_gpu_board_group_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_multi_gpu_board_group_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_multi_processor_count`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_multi_processor_count(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_multi_processor_count", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_name`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_name(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_name", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_pageable_memory_access`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_pageable_memory_access(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_pageable_memory_access", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_pageable_memory_access_uses_host_page_tables`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_pageable_memory_access_uses_host_page_tables(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_pageable_memory_access_uses_host_page_tables", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_pci_bus_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_pci_bus_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_pci_bus_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_pci_device_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_pci_device_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_pci_device_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_pci_domain_id`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_pci_domain_id(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_pci_domain_id", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_persisting_l2cache_max_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_persisting_l2cache_max_size(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_persisting_l2cache_max_size", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_regs_per_block`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_regs_per_block(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_regs_per_block", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_regs_per_multiprocessor`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_regs_per_multiprocessor(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_regs_per_multiprocessor", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_reserved_shared_mem_per_block`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_reserved_shared_mem_per_block(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_reserved_shared_mem_per_block", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_shared_mem_per_block`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_shared_mem_per_block(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_shared_mem_per_block", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_shared_mem_per_block_optin`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_shared_mem_per_block_optin(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_shared_mem_per_block_optin", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_shared_mem_per_multiprocessor`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_shared_mem_per_multiprocessor(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_shared_mem_per_multiprocessor", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_sparse_cuda_array_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_sparse_cuda_array_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_sparse_cuda_array_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_stream_priorities_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_stream_priorities_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_stream_priorities_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_surface_alignment`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_surface_alignment(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_surface_alignment", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_tcc_driver`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_tcc_driver(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_tcc_driver", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_texture_alignment`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_texture_alignment(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_texture_alignment", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_texture_pitch_alignment`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_texture_pitch_alignment(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_texture_pitch_alignment", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_timeline_semaphore_interop_supported`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_timeline_semaphore_interop_supported(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_timeline_semaphore_interop_supported", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_total_const_mem`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_total_const_mem(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_total_const_mem", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_total_global_mem`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_total_global_mem(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_total_global_mem", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_unified_addressing`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_unified_addressing(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_unified_addressing", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_unified_function_pointers`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_unified_function_pointers(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_unified_function_pointers", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_uuid`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_uuid(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_uuid", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.device_prop.get_warp_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_device_prop_get_warp_size(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.device_prop.get_warp_size", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.dim_3.get_x`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_dim_3_get_x(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.dim_3.get_x", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.dim_3.get_y`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_dim_3_get_y(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.dim_3.get_y", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.dim_3.get_z`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_dim_3_get_z(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.dim_3.get_z", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.event.create_with_flags`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_event_create_with_flags(ctx: *mlir.Context, flags: *const mlir.Value, cuda_result_type: *const mlir.Type, event_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.event.create_with_flags", .{
        .operands = .{ .flat = &.{flags} },
        .results = .{ .flat = &.{ cuda_result_type, event_type } },
        .location = location,
    });
}

/// `cuda.runtime.event.query`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_event_query(ctx: *mlir.Context, event: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.event.query", .{
        .operands = .{ .flat = &.{event} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.event.record_with_flags`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_event_record_with_flags(ctx: *mlir.Context, event: *const mlir.Value, stream: *const mlir.Value, flags: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.event.record_with_flags", .{
        .operands = .{ .flat = &.{ event, stream, flags } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.event.synchronize`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_event_synchronize(ctx: *mlir.Context, event: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.event.synchronize", .{
        .operands = .{ .flat = &.{event} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.extent.get_depth`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_extent_get_depth(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.extent.get_depth", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.extent.get_height`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_extent_get_height(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.extent.get_height", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.extent.get_width`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_extent_get_width(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.extent.get_width", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.graph.add_kernel_node`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_graph_add_kernel_node(ctx: *mlir.Context, graph: *const mlir.Value, pDependencies: *const mlir.Value, numDependencies: *const mlir.Value, pNodeParams: *const mlir.Value, cuda_result_type: *const mlir.Type, pGraphNode_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.graph.add_kernel_node", .{
        .operands = .{ .flat = &.{ graph, pDependencies, numDependencies, pNodeParams } },
        .results = .{ .flat = &.{ cuda_result_type, pGraphNode_type } },
        .location = location,
    });
}

/// `cuda.runtime.graph.create`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_graph_create(ctx: *mlir.Context, flags: *const mlir.Value, cuda_result_type: *const mlir.Type, pGraph_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.graph.create", .{
        .operands = .{ .flat = &.{flags} },
        .results = .{ .flat = &.{ cuda_result_type, pGraph_type } },
        .location = location,
    });
}

/// `cuda.runtime.graph.kernel_node_get_attribute`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_graph_kernel_node_get_attribute(ctx: *mlir.Context, hNode: *const mlir.Value, attr: *const mlir.Value, cuda_result_type: *const mlir.Type, value_out_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.graph.kernel_node_get_attribute", .{
        .operands = .{ .flat = &.{ hNode, attr } },
        .results = .{ .flat = &.{ cuda_result_type, value_out_type } },
        .location = location,
    });
}

/// `cuda.runtime.graph.kernel_node_set_attribute`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_graph_kernel_node_set_attribute(ctx: *mlir.Context, hNode: *const mlir.Value, attr: *const mlir.Value, value_: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.graph.kernel_node_set_attribute", .{
        .operands = .{ .flat = &.{ hNode, attr, value_ } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.kernel_node_params.get_block_dim`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_kernel_node_params_get_block_dim(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.kernel_node_params.get_block_dim", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.kernel_node_params.get_extra`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_kernel_node_params_get_extra(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.kernel_node_params.get_extra", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.kernel_node_params.get_func`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_kernel_node_params_get_func(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.kernel_node_params.get_func", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.kernel_node_params.get_grid_dim`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_kernel_node_params_get_grid_dim(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.kernel_node_params.get_grid_dim", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.kernel_node_params.get_kernel_params`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_kernel_node_params_get_kernel_params(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.kernel_node_params.get_kernel_params", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.kernel_node_params.get_shared_mem_bytes`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_kernel_node_params_get_shared_mem_bytes(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.kernel_node_params.get_shared_mem_bytes", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.library.get_global`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_get_global(ctx: *mlir.Context, library: *const mlir.Value, name_arg: *const mlir.Value, cuda_result_type: *const mlir.Type, dptr_type: *const mlir.Type, bytes_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.get_global", .{
        .operands = .{ .flat = &.{ library, name_arg } },
        .results = .{ .flat = &.{ cuda_result_type, dptr_type, bytes_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.get_kernel`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_get_kernel(ctx: *mlir.Context, library: *const mlir.Value, name_arg: *const mlir.Value, cuda_result_type: *const mlir.Type, pKernel_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.get_kernel", .{
        .operands = .{ .flat = &.{ library, name_arg } },
        .results = .{ .flat = &.{ cuda_result_type, pKernel_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.get_kernel_count`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_get_kernel_count(ctx: *mlir.Context, lib: *const mlir.Value, cuda_result_type: *const mlir.Type, count_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.get_kernel_count", .{
        .operands = .{ .flat = &.{lib} },
        .results = .{ .flat = &.{ cuda_result_type, count_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.get_managed`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_get_managed(ctx: *mlir.Context, library: *const mlir.Value, name_arg: *const mlir.Value, cuda_result_type: *const mlir.Type, dptr_type: *const mlir.Type, bytes_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.get_managed", .{
        .operands = .{ .flat = &.{ library, name_arg } },
        .results = .{ .flat = &.{ cuda_result_type, dptr_type, bytes_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.get_unified_function`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_get_unified_function(ctx: *mlir.Context, library: *const mlir.Value, symbol: *const mlir.Value, cuda_result_type: *const mlir.Type, fptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.get_unified_function", .{
        .operands = .{ .flat = &.{ library, symbol } },
        .results = .{ .flat = &.{ cuda_result_type, fptr_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.load_data`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_load_data(ctx: *mlir.Context, code_arg: *const mlir.Value, jitOptions: *const mlir.Value, jitOptionsValues: *const mlir.Value, numJitOptions: *const mlir.Value, libraryOptions: *const mlir.Value, libraryOptionValues: *const mlir.Value, numLibraryOptions: *const mlir.Value, cuda_result_type: *const mlir.Type, library_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.load_data", .{
        .operands = .{ .flat = &.{ code_arg, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions } },
        .results = .{ .flat = &.{ cuda_result_type, library_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.load_from_file`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_load_from_file(ctx: *mlir.Context, fileName: *const mlir.Value, jitOptions: *const mlir.Value, jitOptionsValues: *const mlir.Value, numJitOptions: *const mlir.Value, libraryOptions: *const mlir.Value, libraryOptionValues: *const mlir.Value, numLibraryOptions: *const mlir.Value, cuda_result_type: *const mlir.Type, library_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.load_from_file", .{
        .operands = .{ .flat = &.{ fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions } },
        .results = .{ .flat = &.{ cuda_result_type, library_type } },
        .location = location,
    });
}

/// `cuda.runtime.library.unload`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_library_unload(ctx: *mlir.Context, library: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.library.unload", .{
        .operands = .{ .flat = &.{library} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_dst_array`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_dst_array(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_dst_array", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_dst_pos`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_dst_pos(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_dst_pos", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_dst_ptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_dst_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_dst_ptr", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_extent`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_extent(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_extent", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_kind`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_kind(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_kind", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_src_array`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_src_array(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_src_array", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_src_pos`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_src_pos(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_src_pos", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memcpy_3d_params.get_src_ptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memcpy_3d_params_get_src_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memcpy_3d_params.get_src_ptr", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memory.cpy3d`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memory_cpy3d(ctx: *mlir.Context, p: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memory.cpy3d", .{
        .operands = .{ .flat = &.{p} },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.memory.cpy3d_async`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_memory_cpy3d_async(ctx: *mlir.Context, p: *const mlir.Value, stream: *const mlir.Value, cuda_result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.memory.cpy3d_async", .{
        .operands = .{ .flat = &.{ p, stream } },
        .results = .{ .flat = &.{cuda_result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pitched_ptr.get_pitch`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pitched_ptr_get_pitch(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pitched_ptr.get_pitch", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pitched_ptr.get_ptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pitched_ptr_get_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pitched_ptr.get_ptr", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pitched_ptr.get_xsize`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pitched_ptr_get_xsize(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pitched_ptr.get_xsize", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pitched_ptr.get_ysize`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pitched_ptr_get_ysize(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pitched_ptr.get_ysize", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pos.get_x`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pos_get_x(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pos.get_x", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pos.get_y`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pos_get_y(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pos.get_y", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.pos.get_z`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_pos_get_z(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.pos.get_z", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.runtime.uuid.get_bytes`. Result types are explicit; attributes use mlir.Attribute.
pub fn runtime_uuid_get_bytes(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.runtime.uuid.get_bytes", .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.stream.add_callback`. Result types are explicit; attributes use mlir.Attribute.
pub fn stream_add_callback(ctx: *mlir.Context, stream: *const mlir.Value, user_data: *const mlir.Value, result_type: *const mlir.Type, callback: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.stream.add_callback", .{
        .operands = .{ .flat = &.{ stream, user_data } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "callback", callback),
        },
        .location = location,
    });
}

/// `cuda.stream.begin_capture`. Result types are explicit; attributes use mlir.Attribute.
pub fn stream_begin_capture(ctx: *mlir.Context, stream: *const mlir.Value, result_type: *const mlir.Type, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.stream.begin_capture", .{
        .operands = .{ .flat = &.{stream} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

/// `cuda.stream.create`. Result types are explicit; attributes use mlir.Attribute.
pub fn stream_create(ctx: *mlir.Context, cuda_result_type: *const mlir.Type, stream_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.stream.create", .{
        .results = .{ .flat = &.{ cuda_result_type, stream_type } },
        .location = location,
    });
}

/// `cuda.stream.destroy`. Result types are explicit; attributes use mlir.Attribute.
pub fn stream_destroy(ctx: *mlir.Context, stream: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.stream.destroy", .{
        .operands = .{ .flat = &.{stream} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.stream.end_capture`. Result types are explicit; attributes use mlir.Attribute.
pub fn stream_end_capture(ctx: *mlir.Context, stream: *const mlir.Value, result_type: *const mlir.Type, graph_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.stream.end_capture", .{
        .operands = .{ .flat = &.{stream} },
        .results = .{ .flat = &.{ result_type, graph_type } },
        .location = location,
    });
}

/// `cuda.stream.synchronize`. Result types are explicit; attributes use mlir.Attribute.
pub fn stream_synchronize(ctx: *mlir.Context, stream: *const mlir.Value, event: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.stream.synchronize", .{
        .operands = .{ .flat = if (event) |value| &.{ stream, value } else &.{stream} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cuda.tensor_map_encode_tiled`. Result types are explicit; attributes use mlir.Attribute.
pub fn tensor_map_encode_tiled(ctx: *mlir.Context, tensorDataType: *const mlir.Value, tensorRank: *const mlir.Value, globalAddress: *const mlir.Value, globalDims: *const mlir.Value, globalStrides: *const mlir.Value, boxDims: *const mlir.Value, elementStrides: *const mlir.Value, interleave: *const mlir.Value, swizzle: *const mlir.Value, l2Promotion: *const mlir.Value, oobFill: *const mlir.Value, cuda_result_type: *const mlir.Type, cuTensorMap_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cuda.tensor_map_encode_tiled", .{
        .operands = .{ .flat = &.{ tensorDataType, tensorRank, globalAddress, globalDims, globalStrides, boxDims, elementStrides, interleave, swizzle, l2Promotion, oobFill } },
        .results = .{ .flat = &.{ cuda_result_type, cuTensorMap_type } },
        .location = location,
    });
}

/// Every operation bound here.
pub const operation_names: []const []const u8 = &.{
    "cuda.addressof",
    "cuda.binary",
    "cuda.cast",
    "cuda.compose",
    "cuda.decompose",
    "cuda.device.sync",
    "cuda.driver.event.create",
    "cuda.driver.event.destroy",
    "cuda.driver.event.elapsed_time",
    "cuda.driver.event.query",
    "cuda.driver.event.record",
    "cuda.driver.event.record_with_flags",
    "cuda.driver.event.synchronize",
    "cuda.driver.library.get_global",
    "cuda.driver.library.get_kernel",
    "cuda.driver.library.get_kernel_count",
    "cuda.driver.library.get_managed",
    "cuda.driver.library.get_module",
    "cuda.driver.library.get_unified_function",
    "cuda.driver.library.load_data",
    "cuda.driver.library.load_from_file",
    "cuda.driver.library.unload",
    "cuda.driver.tensor_map.encode_im2col",
    "cuda.driver.tensor_map.encode_im2col_wide",
    "cuda.driver.tensor_map.replace_address",
    "cuda.event.create",
    "cuda.event.destroy",
    "cuda.event.elapsed_time",
    "cuda.event.record",
    "cuda.func",
    "cuda.func.get_name",
    "cuda.graph.destroy",
    "cuda.graph.instantiate",
    "cuda.graph.launch",
    "cuda.graph_exec.destroy",
    "cuda.kernel",
    "cuda.launch",
    "cuda.launch_cfg.access_policy_window",
    "cuda.launch_cfg.cluster_dim",
    "cuda.launch_cfg.cluster_scheduling_policy",
    "cuda.launch_cfg.cooperative",
    "cuda.launch_cfg.create",
    "cuda.launch_cfg.create_from_stream",
    "cuda.launch_cfg.device_updatable_kernel_node",
    "cuda.launch_cfg.grid_dimensions",
    "cuda.launch_cfg.launch_completion_event",
    "cuda.launch_cfg.mem_sync_domain",
    "cuda.launch_cfg.mem_sync_domain_map",
    "cuda.launch_cfg.nvlink_util_centric_scheduling",
    "cuda.launch_cfg.preferred_cluster_dim",
    "cuda.launch_cfg.print",
    "cuda.launch_cfg.priority",
    "cuda.launch_cfg.programmatic_event",
    "cuda.launch_cfg.programmatic_stream_serialization_allowed",
    "cuda.launch_cfg.shared_mem_carveout",
    "cuda.launch_cfg.sync_policy",
    "cuda.launch_ex",
    "cuda.launch_kernel",
    "cuda.library.get_kernel",
    "cuda.library.load_data",
    "cuda.library.unload",
    "cuda.memory.alloc",
    "cuda.memory.alloc_async",
    "cuda.memory.alloc_host",
    "cuda.memory.alloc_managed",
    "cuda.memory.copy",
    "cuda.memory.copy_async",
    "cuda.memory.free",
    "cuda.memory.free_async",
    "cuda.memory.free_host",
    "cuda.memory.set",
    "cuda.memory.set_async",
    "cuda.occupancy_max_active_blocks_per_multiprocessor",
    "cuda.profiler.start",
    "cuda.profiler.stop",
    "cuda.result.assert_success",
    "cuda.result.failed",
    "cuda.result.print_on_error",
    "cuda.result.succeeded",
    "cuda.return",
    "cuda.return_if_error",
    "cuda.runtime.device.properties",
    "cuda.runtime.device_prop.get_access_policy_max_window_size",
    "cuda.runtime.device_prop.get_async_engine_count",
    "cuda.runtime.device_prop.get_can_map_host_memory",
    "cuda.runtime.device_prop.get_can_use_host_pointer_for_registered_mem",
    "cuda.runtime.device_prop.get_cluster_launch",
    "cuda.runtime.device_prop.get_compute_preemption_supported",
    "cuda.runtime.device_prop.get_concurrent_kernels",
    "cuda.runtime.device_prop.get_concurrent_managed_access",
    "cuda.runtime.device_prop.get_cooperative_launch",
    "cuda.runtime.device_prop.get_deferred_mapping_cuda_array_supported",
    "cuda.runtime.device_prop.get_device_numa_config",
    "cuda.runtime.device_prop.get_device_numa_id",
    "cuda.runtime.device_prop.get_direct_managed_mem_access_from_host",
    "cuda.runtime.device_prop.get_ecc_enabled",
    "cuda.runtime.device_prop.get_global_l1cache_supported",
    "cuda.runtime.device_prop.get_gpu_direct_rdma_flush_writes_options",
    "cuda.runtime.device_prop.get_gpu_direct_rdma_supported",
    "cuda.runtime.device_prop.get_gpu_direct_rdma_writes_ordering",
    "cuda.runtime.device_prop.get_gpu_pci_device_id",
    "cuda.runtime.device_prop.get_gpu_pci_subsystem_id",
    "cuda.runtime.device_prop.get_host_native_atomic_supported",
    "cuda.runtime.device_prop.get_host_numa_id",
    "cuda.runtime.device_prop.get_host_numa_multinode_ipc_supported",
    "cuda.runtime.device_prop.get_host_register_read_only_supported",
    "cuda.runtime.device_prop.get_host_register_supported",
    "cuda.runtime.device_prop.get_integrated",
    "cuda.runtime.device_prop.get_ipc_event_supported",
    "cuda.runtime.device_prop.get_is_multi_gpu_board",
    "cuda.runtime.device_prop.get_l_2cache_size",
    "cuda.runtime.device_prop.get_local_l1cache_supported",
    "cuda.runtime.device_prop.get_luid",
    "cuda.runtime.device_prop.get_luid_device_node_mask",
    "cuda.runtime.device_prop.get_major",
    "cuda.runtime.device_prop.get_managed_memory",
    "cuda.runtime.device_prop.get_max_blocks_per_multi_processor",
    "cuda.runtime.device_prop.get_max_grid_size",
    "cuda.runtime.device_prop.get_max_surface_1d",
    "cuda.runtime.device_prop.get_max_surface_1d_layered",
    "cuda.runtime.device_prop.get_max_surface_2d",
    "cuda.runtime.device_prop.get_max_surface_2d_layered",
    "cuda.runtime.device_prop.get_max_surface_3d",
    "cuda.runtime.device_prop.get_max_surface_cubemap",
    "cuda.runtime.device_prop.get_max_surface_cubemap_layered",
    "cuda.runtime.device_prop.get_max_texture_1d",
    "cuda.runtime.device_prop.get_max_texture_1d_layered",
    "cuda.runtime.device_prop.get_max_texture_1d_mipmap",
    "cuda.runtime.device_prop.get_max_texture_2d",
    "cuda.runtime.device_prop.get_max_texture_2d_gather",
    "cuda.runtime.device_prop.get_max_texture_2d_layered",
    "cuda.runtime.device_prop.get_max_texture_2d_linear",
    "cuda.runtime.device_prop.get_max_texture_2d_mipmap",
    "cuda.runtime.device_prop.get_max_texture_3d",
    "cuda.runtime.device_prop.get_max_texture_3d_alt",
    "cuda.runtime.device_prop.get_max_texture_cubemap",
    "cuda.runtime.device_prop.get_max_texture_cubemap_layered",
    "cuda.runtime.device_prop.get_max_threads_dim",
    "cuda.runtime.device_prop.get_max_threads_per_block",
    "cuda.runtime.device_prop.get_max_threads_per_multi_processor",
    "cuda.runtime.device_prop.get_mem_pitch",
    "cuda.runtime.device_prop.get_memory_bus_width",
    "cuda.runtime.device_prop.get_memory_pool_supported_handle_types",
    "cuda.runtime.device_prop.get_memory_pools_supported",
    "cuda.runtime.device_prop.get_minor",
    "cuda.runtime.device_prop.get_mps_enabled",
    "cuda.runtime.device_prop.get_multi_gpu_board_group_id",
    "cuda.runtime.device_prop.get_multi_processor_count",
    "cuda.runtime.device_prop.get_name",
    "cuda.runtime.device_prop.get_pageable_memory_access",
    "cuda.runtime.device_prop.get_pageable_memory_access_uses_host_page_tables",
    "cuda.runtime.device_prop.get_pci_bus_id",
    "cuda.runtime.device_prop.get_pci_device_id",
    "cuda.runtime.device_prop.get_pci_domain_id",
    "cuda.runtime.device_prop.get_persisting_l2cache_max_size",
    "cuda.runtime.device_prop.get_regs_per_block",
    "cuda.runtime.device_prop.get_regs_per_multiprocessor",
    "cuda.runtime.device_prop.get_reserved_shared_mem_per_block",
    "cuda.runtime.device_prop.get_shared_mem_per_block",
    "cuda.runtime.device_prop.get_shared_mem_per_block_optin",
    "cuda.runtime.device_prop.get_shared_mem_per_multiprocessor",
    "cuda.runtime.device_prop.get_sparse_cuda_array_supported",
    "cuda.runtime.device_prop.get_stream_priorities_supported",
    "cuda.runtime.device_prop.get_surface_alignment",
    "cuda.runtime.device_prop.get_tcc_driver",
    "cuda.runtime.device_prop.get_texture_alignment",
    "cuda.runtime.device_prop.get_texture_pitch_alignment",
    "cuda.runtime.device_prop.get_timeline_semaphore_interop_supported",
    "cuda.runtime.device_prop.get_total_const_mem",
    "cuda.runtime.device_prop.get_total_global_mem",
    "cuda.runtime.device_prop.get_unified_addressing",
    "cuda.runtime.device_prop.get_unified_function_pointers",
    "cuda.runtime.device_prop.get_uuid",
    "cuda.runtime.device_prop.get_warp_size",
    "cuda.runtime.dim_3.get_x",
    "cuda.runtime.dim_3.get_y",
    "cuda.runtime.dim_3.get_z",
    "cuda.runtime.event.create_with_flags",
    "cuda.runtime.event.query",
    "cuda.runtime.event.record_with_flags",
    "cuda.runtime.event.synchronize",
    "cuda.runtime.extent.get_depth",
    "cuda.runtime.extent.get_height",
    "cuda.runtime.extent.get_width",
    "cuda.runtime.graph.add_kernel_node",
    "cuda.runtime.graph.create",
    "cuda.runtime.graph.kernel_node_get_attribute",
    "cuda.runtime.graph.kernel_node_set_attribute",
    "cuda.runtime.kernel_node_params.get_block_dim",
    "cuda.runtime.kernel_node_params.get_extra",
    "cuda.runtime.kernel_node_params.get_func",
    "cuda.runtime.kernel_node_params.get_grid_dim",
    "cuda.runtime.kernel_node_params.get_kernel_params",
    "cuda.runtime.kernel_node_params.get_shared_mem_bytes",
    "cuda.runtime.library.get_global",
    "cuda.runtime.library.get_kernel",
    "cuda.runtime.library.get_kernel_count",
    "cuda.runtime.library.get_managed",
    "cuda.runtime.library.get_unified_function",
    "cuda.runtime.library.load_data",
    "cuda.runtime.library.load_from_file",
    "cuda.runtime.library.unload",
    "cuda.runtime.memcpy_3d_params.get_dst_array",
    "cuda.runtime.memcpy_3d_params.get_dst_pos",
    "cuda.runtime.memcpy_3d_params.get_dst_ptr",
    "cuda.runtime.memcpy_3d_params.get_extent",
    "cuda.runtime.memcpy_3d_params.get_kind",
    "cuda.runtime.memcpy_3d_params.get_src_array",
    "cuda.runtime.memcpy_3d_params.get_src_pos",
    "cuda.runtime.memcpy_3d_params.get_src_ptr",
    "cuda.runtime.memory.cpy3d",
    "cuda.runtime.memory.cpy3d_async",
    "cuda.runtime.pitched_ptr.get_pitch",
    "cuda.runtime.pitched_ptr.get_ptr",
    "cuda.runtime.pitched_ptr.get_xsize",
    "cuda.runtime.pitched_ptr.get_ysize",
    "cuda.runtime.pos.get_x",
    "cuda.runtime.pos.get_y",
    "cuda.runtime.pos.get_z",
    "cuda.runtime.uuid.get_bytes",
    "cuda.stream.add_callback",
    "cuda.stream.begin_capture",
    "cuda.stream.create",
    "cuda.stream.destroy",
    "cuda.stream.end_capture",
    "cuda.stream.synchronize",
    "cuda.tensor_map_encode_tiled",
};
