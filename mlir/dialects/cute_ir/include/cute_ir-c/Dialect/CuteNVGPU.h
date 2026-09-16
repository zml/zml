//===- CuteNVGPU.h - C API for the CuTe NVIDIA GPU dialect ------*- C -*-===//

#ifndef CUTE_IR_C_DIALECT_CUTENVGPU_H
#define CUTE_IR_C_DIALECT_CUTENVGPU_H

#include "cute_ir-c/Dialect/CuteNVGPUTypes.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(CuteNVGPU, cute_nvgpu);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTENVGPU_H
