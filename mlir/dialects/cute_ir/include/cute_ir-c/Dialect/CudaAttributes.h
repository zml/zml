// C API for the cuda attributes, generated from the dialect's .td files.
#ifndef CUTE_IR_C_DIALECT_CUDA_ATTRIBUTES_H
#define CUTE_IR_C_DIALECT_CUDA_ATTRIBUTES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles. Enums are their integer
// values (the dialect's *Enums.td). Getters require a value accepted by the
// matching IsA.

// `#cuda.assume_kernel_attr`: Whether `cuda.launch` may assume its callee is a kernel
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaAssumeKernel(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaAssumeKernelAttrGet(MlirContext context, bool value);
MLIR_CAPI_EXPORTED bool mlirCudaAssumeKernelAttrGetValue(MlirAttribute attr);

// `#cuda.compute_target`: Representation, portability and architectures of a compiled executable
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaComputeTarget(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaComputeTargetAttrGet(MlirContext context, uint32_t representation, uint32_t portability, intptr_t numArchs, const uint32_t *archs);
MLIR_CAPI_EXPORTED uint32_t mlirCudaComputeTargetAttrGetRepresentation(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t mlirCudaComputeTargetAttrGetPortability(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t mlirCudaComputeTargetAttrGetNumArchs(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t mlirCudaComputeTargetAttrGetArchs(MlirAttribute attr, intptr_t pos);

// `#cuda.dev_max_shared_memory_optin`: The device's opt-in maximum of shared memory per block
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaDevMaxSharedMemoryOptin(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaDevMaxSharedMemoryOptinAttrGet(MlirContext context);

// `#cuda.device_attributes`: CUDA device attributes
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaDeviceAttributes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaDeviceAttributesAttrGet(MlirContext context, MlirAttribute values);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaDeviceAttributesAttrGetValues(MlirAttribute attr);

// `#cuda.executable`: Format of a compiled executable
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaExecutable(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaExecutableAttrGet(MlirContext context, uint32_t format);
MLIR_CAPI_EXPORTED uint32_t mlirCudaExecutableAttrGetFormat(MlirAttribute attr);

// `#cuda.func_attributes`: CUDA function attributes of a kernel
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaFuncAttributes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaFuncAttributesAttrGet(MlirContext context, MlirAttribute values);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaFuncAttributesAttrGetValues(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUDA_ATTRIBUTES_H
