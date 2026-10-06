//===- Cuda.h - C API for NVIDIA's CUDA host dialect of the CuTe DSL -*- C -*-===//
//
// The dialect handle is `cuda`.

#ifndef CUTE_IR_C_DIALECT_CUDA_H
#define CUTE_IR_C_DIALECT_CUDA_H

#include "cute_ir-c/Dialect/CudaAttributes.h"
#include "cute_ir-c/Dialect/CudaTypes.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cuda, cuda);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUDA_H
