#ifndef ZML_MLIR_DIALECTS_FLY_FLY_CAPI_H
#define ZML_MLIR_DIALECTS_FLY_FLY_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Fly, fly);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(UB, ub);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(GPU, gpu);

// TODO(raph): I'm not really satisfied by those shims, but until we have a
// better solution, it is fine

// Layouts FlyDSL derives from an atom's traits. Each takes the atom / tiled
// type and returns a !fly.layout or !fly.int_tuple type, null on a type
// mismatch.
MlirType zmlFlyCopyAtomThrLayout(MlirType copy_atom);
MlirType zmlFlyCopyAtomTvLayoutSrc(MlirType copy_atom);
MlirType zmlFlyCopyAtomTvLayoutDst(MlirType copy_atom);
MlirType zmlFlyCopyAtomTvLayoutRef(MlirType copy_atom);
MlirType zmlFlyMmaAtomThrLayout(MlirType mma_atom);
MlirType zmlFlyMmaAtomShapeMNK(MlirType mma_atom);
MlirType zmlFlyMmaAtomTvLayoutA(MlirType mma_atom);
MlirType zmlFlyMmaAtomTvLayoutB(MlirType mma_atom);
MlirType zmlFlyMmaAtomTvLayoutC(MlirType mma_atom);
MlirType zmlFlyTiledCopyTiledTvLayoutSrc(MlirType tiled_copy);
MlirType zmlFlyTiledCopyTiledTvLayoutDst(MlirType tiled_copy);
MlirType zmlFlyTiledMmaTileSizeMNK(MlirType tiled_mma);
MlirType zmlFlyTiledMmaThrLayoutVMNK(MlirType tiled_mma);
MlirType zmlFlyTiledMmaTiledTvLayoutA(MlirType tiled_mma);
MlirType zmlFlyTiledMmaTiledTvLayoutB(MlirType tiled_mma);
MlirType zmlFlyTiledMmaTiledTvLayoutC(MlirType tiled_mma);

// Structural readers, so Zig never re-parses printed types.
  
int32_t zmlFlyTypeKind(MlirType type);
int32_t zmlFlyIntTupleRank(MlirType int_tuple);
bool zmlFlyIntTupleIsLeaf(MlirType int_tuple);
bool zmlFlyIntTupleIsStatic(MlirType int_tuple);
MlirType zmlFlyIntTupleAt(MlirType int_tuple, int32_t i);
int32_t zmlFlyIntTupleLeafKind(MlirType int_tuple, int64_t* value);
MlirType zmlFlyLayoutShape(MlirType layout);
MlirType zmlFlyLayoutStride(MlirType layout);
MlirType zmlFlyLayoutLikeShape(MlirType t);
MlirType zmlFlyMemRefElemType(MlirType memref);
MlirAttribute zmlFlyMemRefAddressSpace(MlirType memref);
MlirType zmlFlyPtrElemType(MlirType ptr);
MlirAttribute zmlFlyPtrAddressSpace(MlirType ptr);
int32_t zmlFlyPtrAlignment(MlirType ptr);
MlirAttribute zmlFlyPtrSwizzle(MlirType ptr);
MlirType zmlFlyPtrTypeGet(MlirType elem, MlirAttribute address_space, int32_t alignment, MlirAttribute swizzle);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_FLY_FLY_CAPI_H
