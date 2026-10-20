#ifndef ZML_MLIR_DIALECTS_AFFINE_CAPI_H
#define ZML_MLIR_DIALECTS_AFFINE_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Affine, affine);

#ifdef __cplusplus
}
#endif

#endif
