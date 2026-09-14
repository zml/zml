#ifndef ZML_MLIR_DIALECTS_FLY_FLY_CAPI_H
#define ZML_MLIR_DIALECTS_FLY_FLY_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Fly, fly);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(UB, ub);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(GPU, gpu);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyInt(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntAttrGetStatic(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntAttrGetDynamic(MlirContext ctx, int32_t width, int32_t divisibility);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntAttrGetNone(MlirContext ctx);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyIntTuple(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleAttrGet(MlirContext ctx, MlirAttribute value);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleAttrGetBasis(MlirContext ctx, MlirAttribute value, intptr_t nModes, const int32_t *modes);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleAttrGetTuple(MlirContext ctx, intptr_t nElements, MlirAttribute const *elements);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyLayoutAttrGet(MlirContext ctx, MlirAttribute shape, MlirAttribute stride);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyTile(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyTileAttrGet(MlirContext ctx, MlirAttribute value);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyTileAttrGetModes(MlirContext ctx, intptr_t nModes, MlirAttribute const *modes);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlySwizzle(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlySwizzleAttrGet(MlirContext ctx, int32_t mask, int32_t base, int32_t shift);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyAlign(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyAlignAttrGet(MlirContext ctx, int32_t alignment);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAFlyAddressSpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyAddressSpaceAttrGet(MlirContext ctx, int32_t addressSpace);

MLIR_CAPI_EXPORTED MlirAttribute mlirFlyMmaOperandAttrGet(MlirContext ctx, int32_t operand);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyGemmTraversalOrderAttrGet(MlirContext ctx, int32_t order);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIntTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIntTupleTypeGet(MlirContext ctx, MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyIntTupleTypeGetAttr(MlirType type);
MLIR_CAPI_EXPORTED intptr_t mlirFlyIntTupleTypeGetNumElements(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIntTupleTypeGetElement(MlirType type, intptr_t pos);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsLeaf(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsStatic(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsStaticLeaf(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsNoneLeaf(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsBasisLeaf(MlirType type);
MLIR_CAPI_EXPORTED int64_t mlirFlyIntTupleTypeGetStaticValue(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGet(MlirContext ctx, MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyLayoutTypeGetAttr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGetShape(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGetStride(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyComposedLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyComposedLayoutTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyTile(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTileTypeGet(MlirContext ctx, MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyTileTypeGetAttr(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlySwizzle(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlySwizzleTypeGet(MlirContext ctx, MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyPointer(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGet(MlirContext ctx, MlirType elemTy, MlirAttribute addressSpace, MlirAttribute alignment, MlirAttribute swizzle);
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGetElemTy(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyPointerTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyPointerTypeGetAlignment(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyPointerTypeGetSwizzle(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyMemRef(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGet(MlirContext ctx, MlirType elemTy, MlirAttribute addressSpace, MlirAttribute layout, MlirAttribute alignment, MlirAttribute swizzle);
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGetElemTy(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyMemRefTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirFlyMemRefTypeGetLayout(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCoordTensor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCoordTensorTypeGet(MlirContext ctx, MlirAttribute base, MlirAttribute layout);

MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutLikeTypeGetShape(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCopyAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGet(MlirContext ctx, MlirType copyOp, int32_t valBits);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGetThrLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGetThrValLayoutSrc(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGetThrValLayoutDst(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomTypeGetThrValLayoutRef(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyMmaAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGet(MlirContext ctx, MlirType mmaOp);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetShapeMNK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrValLayoutA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrValLayoutB(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomTypeGetThrValLayoutC(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyTiledCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledCopyTypeGetTiledThrValLayoutDst(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyTiledMma(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetTileSizeMNK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetThrLayoutVMNK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetTiledThrValLayoutA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetTiledThrValLayoutB(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyTiledMmaTypeGetTiledThrValLayoutC(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCopyOpUniversalCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyOpUniversalCopyTypeGet(MlirContext ctx, int32_t bitSize);

MLIR_CAPI_EXPORTED MlirAttribute mlirFlyROCDLBufferDescAddressAttrGet(MlirContext ctx);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(MlirContext ctx, int32_t bitSize, int32_t cacheModifier);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopyLDS(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet(MlirContext ctx, int32_t bitSize);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaOpCDNA3MFMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaOpCDNA3MFMATypeGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaOpGFX11WMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaOpGFX11WMMATypeGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc, bool signA, bool signB, bool clamp);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaOpGFX120XWMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaOpGFX120XWMMATypeGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc, bool signA, bool signB, bool clamp);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_FLY_FLY_CAPI_H
