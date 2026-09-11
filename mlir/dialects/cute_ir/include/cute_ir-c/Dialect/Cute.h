//===- Cute.h - C API for the CuTe dialect ----------------------*- C -*-===//

#ifndef CUTE_IR_C_DIALECT_CUTE_H
#define CUTE_IR_C_DIALECT_CUTE_H

#include "cute_ir-c/Dialect/CuteAttributes.h"
#include "cute_ir-c/Dialect/CuteTypes.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cute, cute);

/// Returns true when `type` belongs to the CuTe dialect.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTE_H
