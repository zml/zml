#ifndef ZML_MLIR_DIALECTS_AFFINE_AFFINE_CAPI_H
#define ZML_MLIR_DIALECTS_AFFINE_AFFINE_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Upstream MLIR has no mlir-c/Dialect/Affine.h; expose the registration
// symbol via this shim so the Zig side can resolve it by mnemonic.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Affine, affine);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_AFFINE_AFFINE_CAPI_H
