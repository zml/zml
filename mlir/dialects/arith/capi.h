//===- capi.h - C API for the arith dialect -----------------------*- C -*-===//
//
// C entry points for the `arith` dialect's flag attributes, which upstream's
// C API leaves out.

#ifndef MLIR_DIALECTS_ARITH_CAPI_H
#define MLIR_DIALECTS_ARITH_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// #arith.fastmath<...>: `value` is the `FastMathFlags` bit set; `Get` returns
// a null attribute when it has bits outside the enum.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAArithFastMath(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirArithFastMathAttrGet(MlirContext context,
                                                          uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirArithFastMathAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAArithIntegerOverflow(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirArithIntegerOverflowAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirArithIntegerOverflowAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_ARITH_CAPI_H
