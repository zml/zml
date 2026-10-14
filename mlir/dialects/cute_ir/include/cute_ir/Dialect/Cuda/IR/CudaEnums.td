// The cuda dialect's enums, generated from the nvidia-cutlass-dsl 4.8.0 enum
// bindings (`_cuda_enum_gen.py`); do not edit.
//
// Case values and spellings are the compiler's. An enum the bindings build
// as a dialect attribute also has its EnumAttr, printed `#<mnemonic><case>`
// or, where the compiler prints it so, `#nvvm<mnemonic case>`.

#ifndef CUTE_IR_GENERATED_CUDA_ENUMS_TD
#define CUTE_IR_GENERATED_CUDA_ENUMS_TD

include "cute_ir/Dialect/Cuda/IR/CudaDialect.td"
include "mlir/IR/EnumAttr.td"

def Cuda_ArchPortabilityEnum : I32EnumAttr<"ArchPortability", "Enumerated type describing portability", [
    I32EnumAttrCase<"portable", 0, "portable">,
    I32EnumAttrCase<"conditional", 1, "conditional">,
    I32EnumAttrCase<"family", 2, "family">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CUDeviceAttrEnum : I32EnumAttr<"CUDeviceAttr", "Enumerated type describing CUDA device attributes can be set through Cuda Dialect", [
    I32EnumAttrCase<"maxThreadsPerBlock", 1, "maxThreadsPerBlock">,
    I32EnumAttrCase<"maxBlockDimX", 2, "maxBlockDimX">,
    I32EnumAttrCase<"maxBlockDimY", 3, "maxBlockDimY">,
    I32EnumAttrCase<"maxBlockDimZ", 4, "maxBlockDimZ">,
    I32EnumAttrCase<"maxGridDimX", 5, "maxGridDimX">,
    I32EnumAttrCase<"maxGridDimY", 6, "maxGridDimY">,
    I32EnumAttrCase<"maxGridDimZ", 7, "maxGridDimZ">,
    I32EnumAttrCase<"maxSharedMemoryPerBlock", 8, "maxSharedMemoryPerBlock">,
    I32EnumAttrCase<"totalConstantMemory", 9, "totalConstantMemory">,
    I32EnumAttrCase<"warpSize", 10, "warpSize">,
    I32EnumAttrCase<"maxPitch", 11, "maxPitch">,
    I32EnumAttrCase<"maxRegistersPerBlock", 12, "maxRegistersPerBlock">,
    I32EnumAttrCase<"clockRate", 13, "clockRate">,
    I32EnumAttrCase<"textureAlignment", 14, "textureAlignment">,
    I32EnumAttrCase<"gpuOverlap", 15, "gpuOverlap">,
    I32EnumAttrCase<"multiProcessorCount", 16, "multiProcessorCount">,
    I32EnumAttrCase<"kernelExecTimeout", 17, "kernelExecTimeout">,
    I32EnumAttrCase<"integrated", 18, "integrated">,
    I32EnumAttrCase<"canMapHostMemory", 19, "canMapHostMemory">,
    I32EnumAttrCase<"computeMode", 20, "computeMode">,
    I32EnumAttrCase<"maxTexture1DWidth", 21, "maxTexture1DWidth">,
    I32EnumAttrCase<"maxTexture2DWidth", 22, "maxTexture2DWidth">,
    I32EnumAttrCase<"maxTexture2DHeight", 23, "maxTexture2DHeight">,
    I32EnumAttrCase<"maxTexture3DWidth", 24, "maxTexture3DWidth">,
    I32EnumAttrCase<"maxTexture3DHeight", 25, "maxTexture3DHeight">,
    I32EnumAttrCase<"maxTexture3DDepth", 26, "maxTexture3DDepth">,
    I32EnumAttrCase<"maxTexture2DLayeredWidth", 27, "maxTexture2DLayeredWidth">,
    I32EnumAttrCase<"maxTexture2DLayeredHeight", 28, "maxTexture2DLayeredHeight">,
    I32EnumAttrCase<"maxTexture2DLayeredLayers", 29, "maxTexture2DLayeredLayers">,
    I32EnumAttrCase<"surfaceAlignment", 30, "surfaceAlignment">,
    I32EnumAttrCase<"concurrentKernels", 31, "concurrentKernels">,
    I32EnumAttrCase<"eccEnabled", 32, "eccEnabled">,
    I32EnumAttrCase<"pciBusId", 33, "pciBusId">,
    I32EnumAttrCase<"pciDeviceId", 34, "pciDeviceId">,
    I32EnumAttrCase<"tccDriver", 35, "tccDriver">,
    I32EnumAttrCase<"memoryClockRate", 36, "memoryClockRate">,
    I32EnumAttrCase<"globalMemoryBusWidth", 37, "globalMemoryBusWidth">,
    I32EnumAttrCase<"l2CacheSize", 38, "l2CacheSize">,
    I32EnumAttrCase<"maxThreadsPerMultiProcessor", 39, "maxThreadsPerMultiProcessor">,
    I32EnumAttrCase<"asyncEngineCount", 40, "asyncEngineCount">,
    I32EnumAttrCase<"unifiedAddressing", 41, "unifiedAddressing">,
    I32EnumAttrCase<"maxTexture1DLayeredWidth", 42, "maxTexture1DLayeredWidth">,
    I32EnumAttrCase<"maxTexture1DLayeredLayers", 43, "maxTexture1DLayeredLayers">,
    I32EnumAttrCase<"maxTexture2DGatherWidth", 45, "maxTexture2DGatherWidth">,
    I32EnumAttrCase<"maxTexture2DGatherHeight", 46, "maxTexture2DGatherHeight">,
    I32EnumAttrCase<"maxTexture3DWidthAlt", 47, "maxTexture3DWidthAlt">,
    I32EnumAttrCase<"maxTexture3DHeightAlt", 48, "maxTexture3DHeightAlt">,
    I32EnumAttrCase<"maxTexture3DDepthAlt", 49, "maxTexture3DDepthAlt">,
    I32EnumAttrCase<"pciDomainId", 50, "pciDomainId">,
    I32EnumAttrCase<"texturePitchAlignment", 51, "texturePitchAlignment">,
    I32EnumAttrCase<"maxTextureCubemapWidth", 52, "maxTextureCubemapWidth">,
    I32EnumAttrCase<"maxTextureCubemapLayeredWidth", 53, "maxTextureCubemapLayeredWidth">,
    I32EnumAttrCase<"maxTextureCubemapLayeredLayers", 54, "maxTextureCubemapLayeredLayers">,
    I32EnumAttrCase<"maxSurface1DWidth", 55, "maxSurface1DWidth">,
    I32EnumAttrCase<"maxSurface2DWidth", 56, "maxSurface2DWidth">,
    I32EnumAttrCase<"maxSurface2DHeight", 57, "maxSurface2DHeight">,
    I32EnumAttrCase<"maxSurface3DWidth", 58, "maxSurface3DWidth">,
    I32EnumAttrCase<"maxSurface3DHeight", 59, "maxSurface3DHeight">,
    I32EnumAttrCase<"maxSurface3DDepth", 60, "maxSurface3DDepth">,
    I32EnumAttrCase<"maxSurface1DLayeredWidth", 61, "maxSurface1DLayeredWidth">,
    I32EnumAttrCase<"maxSurface1DLayeredLayers", 62, "maxSurface1DLayeredLayers">,
    I32EnumAttrCase<"maxSurface2DLayeredWidth", 63, "maxSurface2DLayeredWidth">,
    I32EnumAttrCase<"maxSurface2DLayeredHeight", 64, "maxSurface2DLayeredHeight">,
    I32EnumAttrCase<"maxSurface2DLayeredLayers", 65, "maxSurface2DLayeredLayers">,
    I32EnumAttrCase<"maxSurfaceCubemapWidth", 66, "maxSurfaceCubemapWidth">,
    I32EnumAttrCase<"maxSurfaceCubemapLayeredWidth", 67, "maxSurfaceCubemapLayeredWidth">,
    I32EnumAttrCase<"maxSurfaceCubemapLayeredLayers", 68, "maxSurfaceCubemapLayeredLayers">,
    I32EnumAttrCase<"maxTexture1DLinearWidth", 69, "maxTexture1DLinearWidth">,
    I32EnumAttrCase<"maxTexture2DLinearWidth", 70, "maxTexture2DLinearWidth">,
    I32EnumAttrCase<"maxTexture2DLinearHeight", 71, "maxTexture2DLinearHeight">,
    I32EnumAttrCase<"maxTexture2DLinearPitch", 72, "maxTexture2DLinearPitch">,
    I32EnumAttrCase<"maxTexture2DMipmappedWidth", 73, "maxTexture2DMipmappedWidth">,
    I32EnumAttrCase<"maxTexture2DMipmappedHeight", 74, "maxTexture2DMipmappedHeight">,
    I32EnumAttrCase<"computeCapabilityMajor", 75, "computeCapabilityMajor">,
    I32EnumAttrCase<"computeCapabilityMinor", 76, "computeCapabilityMinor">,
    I32EnumAttrCase<"maxTexture1DMipmappedWidth", 77, "maxTexture1DMipmappedWidth">,
    I32EnumAttrCase<"streamPrioritiesSupported", 78, "streamPrioritiesSupported">,
    I32EnumAttrCase<"globalL1CacheSupported", 79, "globalL1CacheSupported">,
    I32EnumAttrCase<"localL1CacheSupported", 80, "localL1CacheSupported">,
    I32EnumAttrCase<"maxSharedMemoryPerMultiprocessor", 81, "maxSharedMemoryPerMultiprocessor">,
    I32EnumAttrCase<"maxRegistersPerMultiprocessor", 82, "maxRegistersPerMultiprocessor">,
    I32EnumAttrCase<"managedMemory", 83, "managedMemory">,
    I32EnumAttrCase<"isMultiGpuBoard", 84, "isMultiGpuBoard">,
    I32EnumAttrCase<"multiGpuBoardGroupID", 85, "multiGpuBoardGroupID">,
    I32EnumAttrCase<"hostNativeAtomicSupported", 86, "hostNativeAtomicSupported">,
    I32EnumAttrCase<"singleToDoublePrecisionPerfRatio", 87, "singleToDoublePrecisionPerfRatio">,
    I32EnumAttrCase<"pageableMemoryAccess", 88, "pageableMemoryAccess">,
    I32EnumAttrCase<"concurrentManagedAccess", 89, "concurrentManagedAccess">,
    I32EnumAttrCase<"computePreemptionSupported", 90, "computePreemptionSupported">,
    I32EnumAttrCase<"canUseHostPointerForRegisteredMem", 91, "canUseHostPointerForRegisteredMem">,
    I32EnumAttrCase<"reserved92", 92, "reserved92">,
    I32EnumAttrCase<"reserved93", 93, "reserved93">,
    I32EnumAttrCase<"reserved94", 94, "reserved94">,
    I32EnumAttrCase<"cooperativeLaunch", 95, "cooperativeLaunch">,
    I32EnumAttrCase<"reserved96", 96, "reserved96">,
    I32EnumAttrCase<"maxSharedMemoryPerBlockOptin", 97, "maxSharedMemoryPerBlockOptin">,
    I32EnumAttrCase<"canFlushRemoteWrites", 98, "canFlushRemoteWrites">,
    I32EnumAttrCase<"hostRegisterSupported", 99, "hostRegisterSupported">,
    I32EnumAttrCase<"pageableMemoryAccessUsesHostPageTables", 100, "pageableMemoryAccessUsesHostPageTables">,
    I32EnumAttrCase<"directManagedMemAccessFromHost", 101, "directManagedMemAccessFromHost">,
    I32EnumAttrCase<"maxBlocksPerMultiprocessor", 106, "maxBlocksPerMultiprocessor">,
    I32EnumAttrCase<"maxPersistingL2CacheSize", 108, "maxPersistingL2CacheSize">,
    I32EnumAttrCase<"maxAccessPolicyWindowSize", 109, "maxAccessPolicyWindowSize">,
    I32EnumAttrCase<"reservedSharedMemoryPerBlock", 111, "reservedSharedMemoryPerBlock">,
    I32EnumAttrCase<"sparseCudaArraySupported", 112, "sparseCudaArraySupported">,
    I32EnumAttrCase<"hostRegisterReadOnlySupported", 113, "hostRegisterReadOnlySupported">,
    I32EnumAttrCase<"timelineSemaphoreInteropSupported", 114, "timelineSemaphoreInteropSupported">,
    I32EnumAttrCase<"memoryPoolsSupported", 115, "memoryPoolsSupported">,
    I32EnumAttrCase<"gpuDirectRdmaSupported", 116, "gpuDirectRdmaSupported">,
    I32EnumAttrCase<"gpuDirectRdmaFlushWritesOptions", 117, "gpuDirectRdmaFlushWritesOptions">,
    I32EnumAttrCase<"gpuDirectRdmaWritesOrdering", 118, "gpuDirectRdmaWritesOrdering">,
    I32EnumAttrCase<"memoryPoolSupportedHandleTypes", 119, "memoryPoolSupportedHandleTypes">,
    I32EnumAttrCase<"clusterLaunch", 120, "clusterLaunch">,
    I32EnumAttrCase<"deferredMappingCudaArraySupported", 121, "deferredMappingCudaArraySupported">,
    I32EnumAttrCase<"reserved122", 122, "reserved122">,
    I32EnumAttrCase<"reserved123", 123, "reserved123">,
    I32EnumAttrCase<"reserved124", 124, "reserved124">,
    I32EnumAttrCase<"ipcEventSupport", 125, "ipcEventSupport">,
    I32EnumAttrCase<"memSyncDomainCount", 126, "memSyncDomainCount">,
    I32EnumAttrCase<"reserved127", 127, "reserved127">,
    I32EnumAttrCase<"reserved128", 128, "reserved128">,
    I32EnumAttrCase<"reserved129", 129, "reserved129">,
    I32EnumAttrCase<"numaConfig", 130, "numaConfig">,
    I32EnumAttrCase<"numaId", 131, "numaId">,
    I32EnumAttrCase<"reserved132", 132, "reserved132">,
    I32EnumAttrCase<"mpsEnabled", 133, "mpsEnabled">,
    I32EnumAttrCase<"hostNumaId", 134, "hostNumaId">,
    I32EnumAttrCase<"d3d12CigSupported", 135, "d3d12CigSupported">,
    I32EnumAttrCase<"vulkanCigSupported", 138, "vulkanCigSupported">,
    I32EnumAttrCase<"gpuPciDeviceId", 139, "gpuPciDeviceId">,
    I32EnumAttrCase<"gpuPciSubsystemId", 140, "gpuPciSubsystemId">,
    I32EnumAttrCase<"reserved141", 141, "reserved141">,
    I32EnumAttrCase<"hostNumaMemoryPoolsSupported", 142, "hostNumaMemoryPoolsSupported">,
    I32EnumAttrCase<"hostNumaMultinodeIpcSupported", 143, "hostNumaMultinodeIpcSupported">,
    I32EnumAttrCase<"hostMemoryPoolsSupported", 144, "hostMemoryPoolsSupported">,
    I32EnumAttrCase<"reserved145", 145, "reserved145">,
    I32EnumAttrCase<"onlyPartialHostNativeAtomicSupported", 147, "onlyPartialHostNativeAtomicSupported">,
    I32EnumAttrCase<"cudaDevAttrOversizedSharedMemoryPerBlock", 150, "cudaDevAttrOversizedSharedMemoryPerBlock">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CUFunctionAttributeEnum : I32EnumAttr<"CUFunctionAttribute", "Enumerated type describing CUDA function attributes can be set through Cuda Dialect", [
    I32EnumAttrCase<"max_threads_per_block", 0, "max_threads_per_block">,
    I32EnumAttrCase<"shared_size_bytes", 1, "shared_size_bytes">,
    I32EnumAttrCase<"const_size_bytes", 2, "const_size_bytes">,
    I32EnumAttrCase<"local_size_bytes", 3, "local_size_bytes">,
    I32EnumAttrCase<"num_regs", 4, "num_regs">,
    I32EnumAttrCase<"ptx_version", 5, "ptx_version">,
    I32EnumAttrCase<"binary_version", 6, "binary_version">,
    I32EnumAttrCase<"cache_mode_ca", 7, "cache_mode_ca">,
    I32EnumAttrCase<"max_dynamic_shared_size_bytes", 8, "max_dynamic_shared_size_bytes">,
    I32EnumAttrCase<"preferred_shared_memory_carveout", 9, "preferred_shared_memory_carveout">,
    I32EnumAttrCase<"cluster_size_must_be_set", 10, "cluster_size_must_be_set">,
    I32EnumAttrCase<"required_cluster_width", 11, "required_cluster_width">,
    I32EnumAttrCase<"required_cluster_height", 12, "required_cluster_height">,
    I32EnumAttrCase<"required_cluster_depth", 13, "required_cluster_depth">,
    I32EnumAttrCase<"non_portable_cluster_size_allowed", 14, "non_portable_cluster_size_allowed">,
    I32EnumAttrCase<"cluster_scheduling_policy_preference", 15, "cluster_scheduling_policy_preference">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaABIEnum : I32EnumAttr<"CudaABI", "Enumerated of the host library ABI", [
    I32EnumAttrCase<"LP64", 0, "LP64">,
    I32EnumAttrCase<"ILP64", 1, "ILP64">,
    I32EnumAttrCase<"LLP64", 2, "LLP64">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaAccessPropertyEnum : I32EnumAttr<"CudaAccessProperty", "Specifies performance hint with cudaAccessPolicyWindow for hitProp and missProp members", [
    I32EnumAttrCase<"cudaAccessPropertyNormal", 0, "cudaAccessPropertyNormal">,
    I32EnumAttrCase<"cudaAccessPropertyStreaming", 1, "cudaAccessPropertyStreaming">,
    I32EnumAttrCase<"cudaAccessPropertyPersisting", 2, "cudaAccessPropertyPersisting">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaClusterSchedulingPolicyEnum : I32EnumAttr<"CudaClusterSchedulingPolicy", "Enumerated type describing CUDA cluster scheduling policy", [
    I32EnumAttrCase<"cudaClusterSchedulingPolicyDefault", 0, "cudaClusterSchedulingPolicyDefault">,
    I32EnumAttrCase<"cudaClusterSchedulingPolicySpread", 1, "cudaClusterSchedulingPolicySpread">,
    I32EnumAttrCase<"cudaClusterSchedulingPolicyLoadBalancing", 2, "cudaClusterSchedulingPolicyLoadBalancing">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaFuncAttributeEnum : I32EnumAttr<"CudaFuncAttribute", "Enumerated type describing CUDA function attributes can be set through Cuda Dialect", [
    I32EnumAttrCase<"cudaFuncAttributeSharedMemoryMode", 16, "cudaFuncAttributeSharedMemoryMode">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaLaunchAttributeIDEnum : I32EnumAttr<"CudaLaunchAttributeID", "Enumerated type mirroring CudaLaunchAttributeID from CUDA Runtime API", [
    I32EnumAttrCase<"cudaLaunchAttributeIgnore", 0, "cudaLaunchAttributeIgnore">,
    I32EnumAttrCase<"cudaLaunchAttributeAccessPolicyWindow", 1, "cudaLaunchAttributeAccessPolicyWindow">,
    I32EnumAttrCase<"cudaLaunchAttributeCooperative", 2, "cudaLaunchAttributeCooperative">,
    I32EnumAttrCase<"cudaLaunchAttributeSynchronizationPolicy", 3, "cudaLaunchAttributeSynchronizationPolicy">,
    I32EnumAttrCase<"cudaLaunchAttributeClusterDimension", 4, "cudaLaunchAttributeClusterDimension">,
    I32EnumAttrCase<"cudaLaunchAttributeClusterSchedulingPolicyPreference", 5, "cudaLaunchAttributeClusterSchedulingPolicyPreference">,
    I32EnumAttrCase<"cudaLaunchAttributeProgrammaticStreamSerialization", 6, "cudaLaunchAttributeProgrammaticStreamSerialization">,
    I32EnumAttrCase<"cudaLaunchAttributeProgrammaticEvent", 7, "cudaLaunchAttributeProgrammaticEvent">,
    I32EnumAttrCase<"cudaLaunchAttributePriority", 8, "cudaLaunchAttributePriority">,
    I32EnumAttrCase<"cudaLaunchAttributeMemSyncDomainMap", 9, "cudaLaunchAttributeMemSyncDomainMap">,
    I32EnumAttrCase<"cudaLaunchAttributeMemSyncDomain", 10, "cudaLaunchAttributeMemSyncDomain">,
    I32EnumAttrCase<"cudaLaunchAttributePreferredClusterDimension", 11, "cudaLaunchAttributePreferredClusterDimension">,
    I32EnumAttrCase<"cudaLaunchAttributeLaunchCompletionEvent", 12, "cudaLaunchAttributeLaunchCompletionEvent">,
    I32EnumAttrCase<"cudaLaunchAttributeDeviceUpdatableKernelNode", 13, "cudaLaunchAttributeDeviceUpdatableKernelNode">,
    I32EnumAttrCase<"cudaLaunchAttributePreferredSharedMemoryCarveout", 14, "cudaLaunchAttributePreferredSharedMemoryCarveout">,
    I32EnumAttrCase<"cudaLaunchAttributeNvlinkUtilCentricScheduling", 16, "cudaLaunchAttributeNvlinkUtilCentricScheduling">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaLaunchMemSyncDomainEnum : I32EnumAttr<"CudaLaunchMemSyncDomain", "Enumerated type describing CUDA memory sync domain", [
    I32EnumAttrCase<"cudaLaunchMemSyncDomainDefault", 0, "cudaLaunchMemSyncDomainDefault">,
    I32EnumAttrCase<"cudaLaunchMemSyncDomainRemote", 1, "cudaLaunchMemSyncDomainRemote">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaSharedMemoryModeEnum : I32EnumAttr<"CudaSharedMemoryMode", "Enumerated type describing CUDA shared memory modes", [
    I32EnumAttrCase<"cudaSharedMemoryModeDefault", 0, "cudaSharedMemoryModeDefault">,
    I32EnumAttrCase<"cudaSharedMemoryModeRequirePortable", 1, "cudaSharedMemoryModeRequirePortable">,
    I32EnumAttrCase<"cudaSharedMemoryModeAllowNonPortable", 2, "cudaSharedMemoryModeAllowNonPortable">,
    I32EnumAttrCase<"cudaSharedMemoryModeAllowOversizedSharedMemory", 3, "cudaSharedMemoryModeAllowOversizedSharedMemory">,
    I32EnumAttrCase<"cudaSharedMemoryModePreferOversizedSharedMemory", 4, "cudaSharedMemoryModePreferOversizedSharedMemory">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_CudaSynchronizationPolicyEnum : I32EnumAttr<"CudaSynchronizationPolicy", "Enumerated type describing CUDA synchronization policy", [
    I32EnumAttrCase<"cudaSyncPolicyAuto", 0, "cudaSyncPolicyAuto">,
    I32EnumAttrCase<"cudaSyncPolicySpin", 1, "cudaSyncPolicySpin">,
    I32EnumAttrCase<"cudaSyncPolicyYield", 2, "cudaSyncPolicyYield">,
    I32EnumAttrCase<"cudaSyncPolicyBlockingSync", 3, "cudaSyncPolicyBlockingSync">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_ExecutableFormatEnum : I32EnumAttr<"ExecutableFormat", "Enumerated type describing the format of an executable", [
    I32EnumAttrCase<"text", 0, "text">,
    I32EnumAttrCase<"elf", 1, "elf">,
    I32EnumAttrCase<"bitcode", 2, "bitcode">,
    I32EnumAttrCase<"fatbin", 3, "fatbin">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_ExecutableRepresentationEnum : I32EnumAttr<"ExecutableRepresentation", "Enumerated type describing the representation of an executable", [
    I32EnumAttrCase<"sass", 0, "sass">,
    I32EnumAttrCase<"ptx", 2, "ptx">,
    I32EnumAttrCase<"nvvm", 3, "nvvm">,
    I32EnumAttrCase<"ltoir", 4, "ltoir">,
    I32EnumAttrCase<"mlir", 5, "mlir">,
    I32EnumAttrCase<"llvm", 6, "llvm">,
    I32EnumAttrCase<"hostobj", 7, "hostobj">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_GpuArchitectureEnum : I32EnumAttr<"GpuArchitecture", "Enumerated type denoting a GPU Architecture", [
    I32EnumAttrCase<"sm_80", 80, "sm_80">,
    I32EnumAttrCase<"sm_86", 86, "sm_86">,
    I32EnumAttrCase<"sm_87", 87, "sm_87">,
    I32EnumAttrCase<"sm_88", 88, "sm_88">,
    I32EnumAttrCase<"sm_89", 89, "sm_89">,
    I32EnumAttrCase<"sm_90", 90, "sm_90">,
    I32EnumAttrCase<"sm_100", 100, "sm_100">,
    I32EnumAttrCase<"sm_101", 101, "sm_101">,
    I32EnumAttrCase<"sm_103", 103, "sm_103">,
    I32EnumAttrCase<"sm_107", 107, "sm_107">,
    I32EnumAttrCase<"sm_110", 110, "sm_110">,
    I32EnumAttrCase<"sm_120", 120, "sm_120">,
    I32EnumAttrCase<"sm_121", 121, "sm_121">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_MemcpyKindEnum : I32EnumAttr<"MemcpyKind", "Enumerated type describing memcpy directions", [
    I32EnumAttrCase<"HostToHost", 0, "HostToHost">,
    I32EnumAttrCase<"HostToDevice", 1, "HostToDevice">,
    I32EnumAttrCase<"DeviceToHost", 2, "DeviceToHost">,
    I32EnumAttrCase<"DeviceToDevice", 3, "DeviceToDevice">,
    I32EnumAttrCase<"Default", 4, "Default">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

def Cuda_StreamCaptureModeEnumEnum : I32EnumAttr<"StreamCaptureModeEnum", "Enumerated type describing CUDA stream capture modes", [
    I32EnumAttrCase<"Global", 0, "Global">,
    I32EnumAttrCase<"ThreadLocal", 1, "ThreadLocal">,
    I32EnumAttrCase<"Relaxed", 2, "Relaxed">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let genSpecializedAttr = 0;
}

#endif // CUTE_IR_GENERATED_CUDA_ENUMS_TD
