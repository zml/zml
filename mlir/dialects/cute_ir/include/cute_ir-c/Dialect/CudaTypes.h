// C API for the cuda types, generated from the dialect's .td files.
#ifndef CUTE_IR_C_DIALECT_CUDA_TYPES_H
#define CUTE_IR_C_DIALECT_CUDA_TYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles. Enums are their integer
// values (the dialect's *Enums.td). Getters require a value accepted by the
// matching IsA.

// `!cuda.arch_set`: `!cuda.arch_set`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaArchSet(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaArchSetTypeGet(MlirContext context);

// `!cuda.event`: `Event` represents CUDA event
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaEvent(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaEventTypeGet(MlirContext context);

// `!cuda.graph_device_node`: `GraphDeviceNode` represents CUDA device node handle for device-side node update
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaGraphDeviceNode(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaGraphDeviceNodeTypeGet(MlirContext context);

// `!cuda.graph_exec`: CUDA Graph Execution
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaGraphExec(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaGraphExecTypeGet(MlirContext context);

// `!cuda.graph_node`: CUDA Graph Node handle
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaGraphNode(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaGraphNodeTypeGet(MlirContext context);

// `!cuda.graph`: CUDA Graph
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaGraph(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaGraphTypeGet(MlirContext context);

// `!cuda.int`: `cuda.int` is a integer type with a bitwidth matching the underlying ABI integer bitwidth
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaInt(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaIntTypeGet(MlirContext context);

// `!cuda.kernel`: CUDA Kernel handle
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaKernel(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaKernelTypeGet(MlirContext context);

// `!cuda.launch_cfg`: Representation of CUDA's extensible launch configuration mirroring cuda runtime API
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaLaunchConfig(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaLaunchConfigTypeGet(MlirContext context, uint32_t maxNumAttrs);
MLIR_CAPI_EXPORTED uint32_t mlirCudaLaunchConfigTypeGetMaxNumAttrs(MlirType type);

// `!cuda.library`: CUDA Library handle
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaLibrary(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaLibraryTypeGet(MlirContext context);

// `!cuda.module`: CUDA Module handle
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaModule(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaModuleTypeGet(MlirContext context);

// `!cuda.result`: CUDA Result type
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaResult(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaResultTypeGet(MlirContext context);

// `!cuda.runtime.device_prop`: CUDA cudaDeviceProp struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeDeviceProp(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeDevicePropTypeGet(MlirContext context);

// `!cuda.runtime.dim_3`: CUDA dim3 struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeDim3(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeDim3TypeGet(MlirContext context);

// `!cuda.runtime.extent`: CUDA cudaExtent struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeExtent(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeExtentTypeGet(MlirContext context);

// `!cuda.runtime.kernel_node_params`: CUDA cudaKernelNodeParams struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeKernelNodeParams(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeKernelNodeParamsTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference`: `!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreference(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.cooperative`: `!cuda.runtime.launch_attribute_value.cooperative`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueCooperative(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueCooperativeTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.mem_sync_domain`: `!cuda.runtime.launch_attribute_value.mem_sync_domain`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueMemSyncDomain(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueMemSyncDomainTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling`: `!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueNvlinkUtilCentricScheduling(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.portable_cluster_size_mode`: `!cuda.runtime.launch_attribute_value.portable_cluster_size_mode`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValuePortableClusterSizeMode(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValuePortableClusterSizeModeTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.priority`: `!cuda.runtime.launch_attribute_value.priority`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValuePriority(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValuePriorityTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed`: `!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowed(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.shared_mem_carveout`: `!cuda.runtime.launch_attribute_value.shared_mem_carveout`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemCarveout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueSharedMemCarveoutTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.shared_memory_mode`: `!cuda.runtime.launch_attribute_value.shared_memory_mode`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemoryMode(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueSharedMemoryModeTypeGet(MlirContext context);

// `!cuda.runtime.launch_attribute_value.sync_policy`: `!cuda.runtime.launch_attribute_value.sync_policy`
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeLaunchAttributeValueSyncPolicy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeLaunchAttributeValueSyncPolicyTypeGet(MlirContext context);

// `!cuda.runtime.memcpy_3d_params`: CUDA cudaMemcpy3DParms struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeMemcpy3DParams(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeMemcpy3DParamsTypeGet(MlirContext context);

// `!cuda.runtime.pitched_ptr`: CUDA cudaPitchedPtr struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimePitchedPtr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimePitchedPtrTypeGet(MlirContext context);

// `!cuda.runtime.pos`: CUDA cudaPos struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimePos(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimePosTypeGet(MlirContext context);

// `!cuda.runtime.uuid`: CUDA cudaUUID_t struct type (value-typed).
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaRuntimeUuid(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaRuntimeUuidTypeGet(MlirContext context);

// `!cuda.stream`: CUDA Stream
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaStream(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaStreamTypeGet(MlirContext context);

// `!cuda.tensor_map`: CUDA Tensor Map type
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaTensorMap(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaTensorMapTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUDA_TYPES_H
