#ifndef MLIRX_CC_H
#define MLIRX_CC_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute mlirDenseArrayToElements(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // MLIRX_CC_H
