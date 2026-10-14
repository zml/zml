//===- CudaAttrs.td - CUDA host attributes of the CuTe DSL --*- tablegen -*-===//
//
// Generated for nvidia-cutlass-dsl 4.8.0 from the DSL compiler
// (`_cutlass_ir.cu13` parser messages, its mnemonics, and probing its
// attribute parser and printer); do not edit by hand.
//
// Every cuda attribute the compiler's parser knows. `device_attributes` and
// `func_attributes` print as `#cuda<device_attributes{...}>`: the compiler
// parses any dictionary there and prints back only the entries it keeps,
// whose keys were not recovered; these keep the dictionary as written, and
// print a space before it, which the compiler parses too.
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_IR_DIALECT_CUDA_IR_CUDA_ATTRS_TD
#define CUTE_IR_DIALECT_CUDA_IR_CUDA_ATTRS_TD

include "cute_ir/Dialect/Cuda/IR/CudaDialect.td"
include "cute_ir/Dialect/Cuda/IR/CudaEnums.td"

defvar Cuda_Ns = "::mlir::cutlass_compiler::cuda";

def Cuda_AssumeKernelAttr : Cuda_Attr<"AssumeKernel", "assume_kernel_attr"> {
  let summary = "Whether `cuda.launch` may assume its callee is a kernel";
  let description = "`#cuda.assume_kernel_attr<true>`";
  let parameters = (ins "bool":$value);
  let assemblyFormat = "`<` $value `>`";
}

def Cuda_ComputeTargetAttr : Cuda_Attr<"ComputeTarget", "compute_target"> {
  let summary = "Representation, portability and architectures of a compiled executable";
  let description = "`#cuda.compute_target<sass, portable, [sm_90, sm_100]>`";
  let parameters = (ins
    AttrParameter<Cuda_Ns # "::ExecutableRepresentation", "">:$representation,
    AttrParameter<Cuda_Ns # "::ArchPortability", "">:$portability,
    ArrayRefParameter<Cuda_Ns # "::GpuArchitecture", "">:$archs
  );
  let assemblyFormat = "`<` $representation `,` $portability `,` `[` $archs `]` `>`";
}

def Cuda_DevMaxSharedMemoryOptinAttr
    : Cuda_Attr<"DevMaxSharedMemoryOptin", "dev_max_shared_memory_optin"> {
  let summary = "The device's opt-in maximum of shared memory per block";
  let description = "`#cuda.dev_max_shared_memory_optin`";
}

def Cuda_DeviceAttributesAttr : Cuda_Attr<"DeviceAttributes", "device_attributes"> {
  let summary = "CUDA device attributes";
  let description = "`#cuda<device_attributes {}>`";
  let parameters = (ins "::mlir::DictionaryAttr":$values);
  let assemblyFormat = "$values";
}

def Cuda_ExecutableAttr : Cuda_Attr<"Executable", "executable"> {
  let summary = "Format of a compiled executable";
  let description = "`#cuda.executable<elf>`";
  let parameters = (ins AttrParameter<Cuda_Ns # "::ExecutableFormat", "">:$format);
  let assemblyFormat = "`<` $format `>`";
}

def Cuda_FuncAttributesAttr : Cuda_Attr<"FuncAttributes", "func_attributes"> {
  let summary = "CUDA function attributes of a kernel";
  let description = "`#cuda<func_attributes {}>`";
  let parameters = (ins "::mlir::DictionaryAttr":$values);
  let assemblyFormat = "$values";
}

#endif // CUTE_IR_DIALECT_CUDA_IR_CUDA_ATTRS_TD
