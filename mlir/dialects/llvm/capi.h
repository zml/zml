//===- capi.h - C API for the llvm dialect ------------------------*- C -*-===//
//
// C entry points for the `llvm` dialect's loop annotations, which upstream's
// C API leaves out.

#ifndef MLIR_DIALECTS_LLVM_CAPI_H
#define MLIR_DIALECTS_LLVM_CAPI_H

#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Null parameters are unset.

MLIR_CAPI_EXPORTED bool mlirAttributeIsALLVMLoopUnroll(MlirAttribute attr);
/// `disable`, `runtimeDisable` and `full` are bool attributes, `count` an
/// integer attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMLoopUnrollAttrGet(
    MlirContext context, MlirAttribute disable, MlirAttribute count,
    MlirAttribute runtimeDisable, MlirAttribute full);

MLIR_CAPI_EXPORTED bool mlirAttributeIsALLVMLoopAnnotation(MlirAttribute attr);
/// `disableNonforced` and `mustProgress` are bool attributes, `unroll` a
/// loop unroll attribute; the other parameters are unset.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMLoopAnnotationAttrGet(
    MlirContext context, MlirAttribute disableNonforced, MlirAttribute unroll,
    MlirAttribute mustProgress);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_LLVM_CAPI_H
