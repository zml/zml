//===- capi.h - C API for the ub dialect --------------------------*- C -*-===//

#ifndef MLIR_DIALECTS_UB_CAPI_H
#define MLIR_DIALECTS_UB_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(UB, ub);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_UB_CAPI_H
