//===- capi.h - C API for the vector dialect ----------------------*- C -*-===//
//
// C entry points for the `vector` dialect's enum attributes, which upstream's
// C API leaves out.

#ifndef MLIR_DIALECTS_VECTOR_CAPI_H
#define MLIR_DIALECTS_VECTOR_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Enum attributes take and return the C++ enum's integer value; `Get` returns
// a null attribute for a value outside the enum.

// #vector.kind<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsAVectorCombiningKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirVectorCombiningKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirVectorCombiningKindAttrGetValue(MlirAttribute attr);

// #vector.iterator_type<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsAVectorIteratorType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirVectorIteratorTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirVectorIteratorTypeAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_VECTOR_CAPI_H
