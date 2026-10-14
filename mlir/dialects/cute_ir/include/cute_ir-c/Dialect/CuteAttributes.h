// C API for CuTe algebra attributes.

#ifndef CUTE_IR_C_DIALECT_CUTE_ATTRIBUTES_H
#define CUTE_IR_C_DIALECT_CUTE_ATTRIBUTES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors accept raw algebra expressions, not MLIR attribute assembly.
// For example, LayoutAttrGet accepts "(16,32):(32,1)". Invalid expressions
// return a null attribute. GetValue returns canonical algebra text owned by
// the attribute's context; callers must not free it. Getters require an
// attribute accepted by the corresponding IsA predicate.

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteIntTupleAttrGet(MlirContext context,
                                                         MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteIntTuple(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirCuteIntTupleAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCoordAttrGet(MlirContext context,
                                                      MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteCoord(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteCoordAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteShapeAttrGet(MlirContext context,
                                                      MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteShapeAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteStrideAttrGet(MlirContext context,
                                                       MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteStride(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteStrideAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteLayoutAttrGet(MlirContext context,
                                                       MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteLayoutAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteTileAttrGet(MlirContext context,
                                                     MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteTile(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteTileAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteComposedLayoutAttrGet(MlirContext context, MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteComposedLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirCuteComposedLayoutAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteSwizzleAttrGet(MlirContext context,
                                                        MlirStringRef value);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteSwizzle(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirCuteSwizzleAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTE_ATTRIBUTES_H
