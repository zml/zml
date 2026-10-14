//===- NVVM.h - C API for NVIDIA's NVVM dialect of the CuTe DSL -*- C -*-===//
//
// The dialect handle is `cute_nvvm`, not the upstream dialect's `nvvm`: both register a dialect named `nvvm`.

#ifndef CUTE_IR_C_DIALECT_NVVM_H
#define CUTE_IR_C_DIALECT_NVVM_H

#include "cute_ir-c/Dialect/NVVMAttributes.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CuteNVVM, cute_nvvm);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_NVVM_H
