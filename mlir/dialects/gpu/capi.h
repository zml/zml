// C entry points for the `gpu` dialect that upstream's C API leaves out: its
// registration (outside `CAPIGPU`, which also brings the GPU passes) and the
// enum attributes kernels build.
#ifndef ZML_MLIR_DIALECTS_GPU_CAPI_H
#define ZML_MLIR_DIALECTS_GPU_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(GPU, gpu);

// Enum attributes take and return the C++ enum's integer value; `Get` returns
// a null attribute for a value outside the enum.

// #gpu<dim ...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsAGPUDimension(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirGPUDimensionAttrGet(MlirContext ctx, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirGPUDimensionAttrGetValue(MlirAttribute attr);

// #gpu<shuffle_mode ...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsAGPUShuffleMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirGPUShuffleModeAttrGet(MlirContext ctx, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirGPUShuffleModeAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_GPU_CAPI_H
