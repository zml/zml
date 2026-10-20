// C entry points for the `vector` dialect's enum attributes, which upstream's
// C API leaves out.
#ifndef ZML_MLIR_DIALECTS_VECTOR_CAPI_H
#define ZML_MLIR_DIALECTS_VECTOR_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enum attributes take and return the C++ enum's integer value; `Get` returns
// a null attribute for a value outside the enum.

// #vector.kind<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsAVectorCombiningKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirVectorCombiningKindAttrGet(MlirContext ctx, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirVectorCombiningKindAttrGetValue(MlirAttribute attr);

// #vector.iterator_type<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsAVectorIteratorType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirVectorIteratorTypeAttrGet(MlirContext ctx, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirVectorIteratorTypeAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_VECTOR_CAPI_H
