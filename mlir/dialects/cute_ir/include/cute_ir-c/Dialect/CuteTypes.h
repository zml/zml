// C API for CuTe types.
#ifndef CUTE_IR_C_DIALECT_CUTE_TYPES_H
#define CUTE_IR_C_DIALECT_CUTE_TYPES_H
#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif

// Constructors return a null type for invalid parameters. All handles must
// belong to context. Getters require a type accepted by the matching IsA.

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteArithTupleIterator(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteArithTupleIteratorTypeGet(MlirContext context, MlirType arithTuple);
MLIR_CAPI_EXPORTED MlirType
mlirCuteArithTupleIteratorTypeGetArithTuple(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteComposedLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteComposedLayoutTypeGet(MlirContext context,
                                                          MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteComposedLayoutTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteConstrainedInt32(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteConstrainedInt32TypeGet(MlirContext context, uint64_t divisibleBy);
MLIR_CAPI_EXPORTED uint64_t
mlirCuteConstrainedInt32TypeGetDivisibleBy(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteConstrainedInt64(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteConstrainedInt64TypeGet(MlirContext context, uint64_t divisibleBy);
MLIR_CAPI_EXPORTED uint64_t
mlirCuteConstrainedInt64TypeGetDivisibleBy(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteCoordTensor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTensorTypeGet(MlirContext context,
                                                       MlirType arithTuple,
                                                       MlirType layout);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTensorTypeGetArithTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTensorTypeGetLayout(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteCoord(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTypeGet(MlirContext context,
                                                 MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCoordTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteFastDivmodDivisor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteFastDivmodDivisorTypeGet(
    MlirContext context, unsigned width, bool isPow2);
MLIR_CAPI_EXPORTED unsigned
mlirCuteFastDivmodDivisorTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteFastDivmodDivisorTypeGetIsPow2(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteIntTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteIntTupleTypeGet(MlirContext context,
                                                    MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteIntTupleTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteLayoutTypeGet(MlirContext context,
                                                  MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteLayoutTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteMemRef(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteMemRefTypeGet(MlirContext context,
                                                  MlirType ptr,
                                                  MlirType layout);
MLIR_CAPI_EXPORTED MlirType mlirCuteMemRefTypeGetPtr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteMemRefTypeGetLayout(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACutePtr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCutePtrTypeGet(MlirContext context,
                                               MlirType valueType,
                                               MlirAttribute memorySpace,
                                               uint64_t alignment,
                                               MlirAttribute swizzle);
MLIR_CAPI_EXPORTED MlirType mlirCutePtrTypeGetValueType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirCutePtrTypeGetMemorySpace(MlirType type);
MLIR_CAPI_EXPORTED uint64_t mlirCutePtrTypeGetAlignment(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirCutePtrTypeGetSwizzle(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteShape(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteShapeTypeGet(MlirContext context,
                                                 MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteShapeTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteSparseElem(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteSparseElemTypeGet(MlirContext context,
                                                      int numLogical,
                                                      MlirType physicalType);
MLIR_CAPI_EXPORTED int mlirCuteSparseElemTypeGetNumLogical(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteSparseElemTypeGetPhysicalType(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteStride(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteStrideTypeGet(MlirContext context,
                                                  MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteStrideTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteSwizzle(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteSwizzleTypeGet(MlirContext context,
                                                   MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteSwizzleTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteTile(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteTileTypeGet(MlirContext context,
                                                MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteTileTypeGetAttr(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTE_TYPES_H
