//===- NVVM.cpp - C API for NVIDIA's NVVM dialect of the CuTe DSL ------===//

#include "cute_ir-c/Dialect/NVVM.h"

#include "cute_ir/Dialect/NVVM/IR/NVVMDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    CuteNVVM, cute_nvvm, mlir::cutlass_compiler::nvvm::NVVMDialect)
