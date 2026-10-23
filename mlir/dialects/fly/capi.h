//===- capi.h - C API for the fly dialect -------------------------*- C -*-===//

#ifndef MLIR_DIALECTS_FLY_CAPI_H
#define MLIR_DIALECTS_FLY_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Fly, fly);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyInt(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntAttrStaticGet(MlirContext context,
                                                         int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntAttrDynamicGet(MlirContext context,
                                                          int32_t width,
                                                          int32_t divisibility);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntAttrNoneGet(MlirContext context);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyIntTuple(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleAttrGet(MlirContext context,
                                                        MlirAttribute value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyIntTupleAttrBasisGet(MlirContext context, MlirAttribute value,
                            intptr_t nModes, const int32_t *modes);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleAttrTupleGet(
    MlirContext context, intptr_t nElements, MlirAttribute const *elements);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyLayoutAttrGet(MlirContext context,
                                                      MlirAttribute shape,
                                                      MlirAttribute stride);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyTile(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyTileAttrGet(MlirContext context,
                                                    MlirAttribute value);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyTileAttrModesGet(
    MlirContext context, intptr_t nModes, MlirAttribute const *modes);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlySwizzle(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlySwizzleAttrGet(MlirContext context,
                                                       int32_t mask,
                                                       int32_t base,
                                                       int32_t shift);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyAlign(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyAlignAttrGet(MlirContext context,
                                                     int32_t alignment);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyAddressSpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyAddressSpaceAttrGet(MlirContext context, int32_t addressSpace);

MLIR_CAPI_EXPORTED MlirAttribute mlirFlyMmaOperandAttrGet(MlirContext context,
                                                          int32_t operand);
MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyGemmTraversalOrderAttrGet(MlirContext context, int32_t order);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIntTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIntTupleTypeGet(MlirContext context,
                                                   MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleTypeGetAttr(MlirType type);
MLIR_CAPI_EXPORTED intptr_t mlirFlyIntTupleTypeGetNumElements(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIntTupleTypeGetElement(MlirType type,
                                                          intptr_t pos);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsLeaf(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsStatic(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsStaticLeaf(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsNoneLeaf(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsBasisLeaf(MlirType type);
MLIR_CAPI_EXPORTED int64_t mlirFlyIntTupleTypeGetStaticValue(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGet(MlirContext context,
                                                 MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyLayoutTypeGetAttr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGetShape(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGetStride(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyComposedLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyComposedLayoutTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyTile(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTileTypeGet(MlirContext context,
                                               MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyTileTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlySwizzle(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlySwizzleTypeGet(MlirContext context,
                                                  MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyPointer(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGet(MlirContext context,
                                                  MlirType elemTy,
                                                  MlirAttribute addressSpace,
                                                  MlirAttribute alignment,
                                                  MlirAttribute swizzle);
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGetElemTy(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyPointerTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyPointerTypeGetAlignment(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyPointerTypeGetSwizzle(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyMemRef(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGet(
    MlirContext context, MlirType elemTy, MlirAttribute addressSpace,
    MlirAttribute layout, MlirAttribute alignment, MlirAttribute swizzle);
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGetElemTy(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyMemRefTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyMemRefTypeGetLayout(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCoordTensor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCoordTensorTypeGet(MlirContext context,
                                                      MlirAttribute base,
                                                      MlirAttribute layout);

MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutLikeTypeGetShape(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCopyAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGet(MlirContext context,
                                                   MlirType copyOp,
                                                   int32_t valBits);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGetThrLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyCopyAtomTypeGetThrValLayoutSrc(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyCopyAtomTypeGetThrValLayoutDst(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyCopyAtomTypeGetThrValLayoutRef(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyMmaAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGet(MlirContext context,
                                                  MlirType mmaOp);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetShapeMNK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrValLayoutA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrValLayoutB(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrValLayoutC(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyTiledCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyTiledCopyTypeGetTiledThrValLayoutDst(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyTiledMma(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetTileSizeMNK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetThrLayoutVMNK(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyTiledMmaTypeGetTiledThrValLayoutA(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyTiledMmaTypeGetTiledThrValLayoutB(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyTiledMmaTypeGetTiledThrValLayoutC(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCopyOpUniversalCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirFlyCopyOpUniversalCopyTypeGet(MlirContext context, int32_t bitSize);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_FLY_CAPI_H
