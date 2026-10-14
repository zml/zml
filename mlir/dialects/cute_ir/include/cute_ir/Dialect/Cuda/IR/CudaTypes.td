//===- CudaTypes.td - CUDA host types of the CuTe DSL -----*- tablegen -*-===//
//
// Generated for nvidia-cutlass-dsl 4.8.0 from the DSL compiler (its typed
// Python type classes, its mnemonics, `_cutlass_ir.cu13` parser messages and
// the verifier's wording for each type); do not edit by hand.
//
// Every cuda type the compiler's parser knows. Only `launch_cfg` has a
// parameter; the C++ class names not given by the typed Python classes
// (`!cuda.int`, the graph handles, `arch_set`, `kernel`, `module` and the
// launch attribute values) follow the mnemonic.
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_IR_DIALECT_CUDA_IR_CUDA_TYPES_TD
#define CUTE_IR_DIALECT_CUDA_IR_CUDA_TYPES_TD

include "cute_ir/Dialect/Cuda/IR/CudaDialect.td"

def Cuda_ArchSetType : Cuda_Type<"ArchSet", "arch_set"> {
  let summary = "`!cuda.arch_set`";
  let description = "`!cuda.arch_set`";
  let assemblyFormat = "";
}

def Cuda_EventType : Cuda_Type<"Event", "event"> {
  let summary = "`Event` represents CUDA event";
  let description = "`!cuda.event`";
  let assemblyFormat = "";
}

def Cuda_GraphType : Cuda_Type<"Graph", "graph"> {
  let summary = "CUDA Graph";
  let description = "`!cuda.graph`";
  let assemblyFormat = "";
}

def Cuda_GraphDeviceNodeType : Cuda_Type<"GraphDeviceNode", "graph_device_node"> {
  let summary = "`GraphDeviceNode` represents CUDA device node handle for device-side node update";
  let description = "`!cuda.graph_device_node`";
  let assemblyFormat = "";
}

def Cuda_GraphExecType : Cuda_Type<"GraphExec", "graph_exec"> {
  let summary = "CUDA Graph Execution";
  let description = "`!cuda.graph_exec`";
  let assemblyFormat = "";
}

def Cuda_GraphNodeType : Cuda_Type<"GraphNode", "graph_node"> {
  let summary = "CUDA Graph Node handle";
  let description = "`!cuda.graph_node`";
  let assemblyFormat = "";
}

def Cuda_IntType : Cuda_Type<"Int", "int"> {
  let summary = "`cuda.int` is a integer type with a bitwidth matching the underlying ABI integer bitwidth";
  let description = "`!cuda.int`";
  let assemblyFormat = "";
}

def Cuda_KernelType : Cuda_Type<"Kernel", "kernel"> {
  let summary = "CUDA Kernel handle";
  let description = "`!cuda.kernel`";
  let assemblyFormat = "";
}

def Cuda_LaunchConfigType : Cuda_Type<"LaunchConfig", "launch_cfg"> {
  let summary = "Representation of CUDA's extensible launch configuration mirroring cuda runtime API";
  let description = "`!cuda.launch_cfg<max_attrs = 2>`";
  let parameters = (ins "uint32_t":$maxNumAttrs);
  let assemblyFormat = "`<` `max_attrs` `=` $maxNumAttrs `>`";
}

def Cuda_LibraryType : Cuda_Type<"Library", "library"> {
  let summary = "CUDA Library handle";
  let description = "`!cuda.library`";
  let assemblyFormat = "";
}

def Cuda_ModuleType : Cuda_Type<"Module", "module"> {
  let summary = "CUDA Module handle";
  let description = "`!cuda.module`";
  let assemblyFormat = "";
}

def Cuda_ResultType : Cuda_Type<"Result", "result"> {
  let summary = "CUDA Result type";
  let description = "`!cuda.result`";
  let assemblyFormat = "";
}

def Cuda_RuntimeDevicePropType : Cuda_Type<"RuntimeDeviceProp", "runtime.device_prop"> {
  let summary = "CUDA cudaDeviceProp struct type (value-typed).";
  let description = "`!cuda.runtime.device_prop`";
  let assemblyFormat = "";
}

def Cuda_RuntimeDim3Type : Cuda_Type<"RuntimeDim3", "runtime.dim_3"> {
  let summary = "CUDA dim3 struct type (value-typed).";
  let description = "`!cuda.runtime.dim_3`";
  let assemblyFormat = "";
}

def Cuda_RuntimeExtentType : Cuda_Type<"RuntimeExtent", "runtime.extent"> {
  let summary = "CUDA cudaExtent struct type (value-typed).";
  let description = "`!cuda.runtime.extent`";
  let assemblyFormat = "";
}

def Cuda_RuntimeKernelNodeParamsType : Cuda_Type<"RuntimeKernelNodeParams", "runtime.kernel_node_params"> {
  let summary = "CUDA cudaKernelNodeParams struct type (value-typed).";
  let description = "`!cuda.runtime.kernel_node_params`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType : Cuda_Type<"RuntimeLaunchAttributeValueClusterSchedulingPolicyPreference", "runtime.launch_attribute_value.cluster_scheduling_policy_preference"> {
  let summary = "`!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference`";
  let description = "`!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueCooperativeType : Cuda_Type<"RuntimeLaunchAttributeValueCooperative", "runtime.launch_attribute_value.cooperative"> {
  let summary = "`!cuda.runtime.launch_attribute_value.cooperative`";
  let description = "`!cuda.runtime.launch_attribute_value.cooperative`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueMemSyncDomainType : Cuda_Type<"RuntimeLaunchAttributeValueMemSyncDomain", "runtime.launch_attribute_value.mem_sync_domain"> {
  let summary = "`!cuda.runtime.launch_attribute_value.mem_sync_domain`";
  let description = "`!cuda.runtime.launch_attribute_value.mem_sync_domain`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType : Cuda_Type<"RuntimeLaunchAttributeValueNvlinkUtilCentricScheduling", "runtime.launch_attribute_value.nvlink_util_centric_scheduling"> {
  let summary = "`!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling`";
  let description = "`!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValuePortableClusterSizeModeType : Cuda_Type<"RuntimeLaunchAttributeValuePortableClusterSizeMode", "runtime.launch_attribute_value.portable_cluster_size_mode"> {
  let summary = "`!cuda.runtime.launch_attribute_value.portable_cluster_size_mode`";
  let description = "`!cuda.runtime.launch_attribute_value.portable_cluster_size_mode`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValuePriorityType : Cuda_Type<"RuntimeLaunchAttributeValuePriority", "runtime.launch_attribute_value.priority"> {
  let summary = "`!cuda.runtime.launch_attribute_value.priority`";
  let description = "`!cuda.runtime.launch_attribute_value.priority`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType : Cuda_Type<"RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowed", "runtime.launch_attribute_value.programmatic_stream_serialization_allowed"> {
  let summary = "`!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed`";
  let description = "`!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueSharedMemCarveoutType : Cuda_Type<"RuntimeLaunchAttributeValueSharedMemCarveout", "runtime.launch_attribute_value.shared_mem_carveout"> {
  let summary = "`!cuda.runtime.launch_attribute_value.shared_mem_carveout`";
  let description = "`!cuda.runtime.launch_attribute_value.shared_mem_carveout`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueSharedMemoryModeType : Cuda_Type<"RuntimeLaunchAttributeValueSharedMemoryMode", "runtime.launch_attribute_value.shared_memory_mode"> {
  let summary = "`!cuda.runtime.launch_attribute_value.shared_memory_mode`";
  let description = "`!cuda.runtime.launch_attribute_value.shared_memory_mode`";
  let assemblyFormat = "";
}

def Cuda_RuntimeLaunchAttributeValueSyncPolicyType : Cuda_Type<"RuntimeLaunchAttributeValueSyncPolicy", "runtime.launch_attribute_value.sync_policy"> {
  let summary = "`!cuda.runtime.launch_attribute_value.sync_policy`";
  let description = "`!cuda.runtime.launch_attribute_value.sync_policy`";
  let assemblyFormat = "";
}

def Cuda_RuntimeMemcpy3DParamsType : Cuda_Type<"RuntimeMemcpy3DParams", "runtime.memcpy_3d_params"> {
  let summary = "CUDA cudaMemcpy3DParms struct type (value-typed).";
  let description = "`!cuda.runtime.memcpy_3d_params`";
  let assemblyFormat = "";
}

def Cuda_RuntimePitchedPtrType : Cuda_Type<"RuntimePitchedPtr", "runtime.pitched_ptr"> {
  let summary = "CUDA cudaPitchedPtr struct type (value-typed).";
  let description = "`!cuda.runtime.pitched_ptr`";
  let assemblyFormat = "";
}

def Cuda_RuntimePosType : Cuda_Type<"RuntimePos", "runtime.pos"> {
  let summary = "CUDA cudaPos struct type (value-typed).";
  let description = "`!cuda.runtime.pos`";
  let assemblyFormat = "";
}

def Cuda_RuntimeUuidType : Cuda_Type<"RuntimeUuid", "runtime.uuid"> {
  let summary = "CUDA cudaUUID_t struct type (value-typed).";
  let description = "`!cuda.runtime.uuid`";
  let assemblyFormat = "";
}

def Cuda_StreamType : Cuda_Type<"Stream", "stream"> {
  let summary = "CUDA Stream";
  let description = "`!cuda.stream`";
  let assemblyFormat = "";
}

def Cuda_TensorMapType : Cuda_Type<"TensorMap", "tensor_map"> {
  let summary = "CUDA Tensor Map type";
  let description = "`!cuda.tensor_map`";
  let assemblyFormat = "";
}

#endif // CUTE_IR_DIALECT_CUDA_IR_CUDA_TYPES_TD
