#ifndef ZML_MLIR_DIALECTS_FUNC_CAPI_H
#define ZML_MLIR_DIALECTS_FUNC_CAPI_H

#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

void mlirRegisterFuncExtensions(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif
