//===- Cuda.cpp - C API for NVIDIA's CUDA host dialect of the CuTe DSL ------===//

#include "cute_ir-c/Dialect/Cuda.h"

#include "cute_ir/Dialect/Cuda/IR/CudaDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    Cuda, cuda, mlir::cutlass_compiler::cuda::CudaDialect)
