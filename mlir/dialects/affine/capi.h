//===- capi.h - C API for the affine dialect ----------------------*- C -*-===//

#ifndef MLIR_DIALECTS_AFFINE_CAPI_H
#define MLIR_DIALECTS_AFFINE_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Affine, affine);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_AFFINE_CAPI_H
