//===- capi.h - C API for the func dialect ------------------------*- C -*-===//

#ifndef MLIR_DIALECTS_FUNC_CAPI_H
#define MLIR_DIALECTS_FUNC_CAPI_H

#include "mlir-c/Dialect/Func.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void
mlirRegisterFuncExtensions(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_FUNC_CAPI_H
