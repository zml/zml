// C entry points for the `arith` dialect's flag attributes, which upstream's
// C API leaves out.
#ifndef ZML_MLIR_DIALECTS_ARITH_CAPI_H
#define ZML_MLIR_DIALECTS_ARITH_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// #arith.fastmath<...>: `value` is the `FastMathFlags` bit set; `Get` returns
// a null attribute when it has bits outside the enum.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAArithFastMath(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirArithFastMathAttrGet(MlirContext ctx, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirArithFastMathAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_ARITH_CAPI_H
