// C API for the CuTe types, generated from the dialect's .td files.
#ifndef CUTE_IR_C_DIALECT_CUTE_TYPES_H
#define CUTE_IR_C_DIALECT_CUTE_TYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles, and std::optional enums a
// negative value. Enums are their integer values (the dialect's *Enums.td).
// Getters require a value accepted by the matching IsA.

// `!cute.arith_tuple_iter`: Iterator over an arithmetic tuple.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteArithTupleIterator(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteArithTupleIteratorTypeGet(MlirContext context, MlirType arithTuple);
MLIR_CAPI_EXPORTED MlirType
mlirCuteArithTupleIteratorTypeGetArithTuple(MlirType type);

// `!cute.composed_layout`: CuTe composed layout: A ∘ offset ∘ B.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteComposedLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteComposedLayoutTypeGet(MlirContext context,
                                                          MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteComposedLayoutTypeGetAttr(MlirType type);

// `!cute.i<N>`: Integer of any width with known divisibility and power of two.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteConstrainedInt(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteConstrainedIntTypeGet(MlirContext context,
                                                          int64_t divisibility,
                                                          unsigned width,
                                                          bool isPow2);
MLIR_CAPI_EXPORTED int64_t
mlirCuteConstrainedIntTypeGetDivisibility(MlirType type);
MLIR_CAPI_EXPORTED unsigned mlirCuteConstrainedIntTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteConstrainedIntTypeGetIsPow2(MlirType type);

// `!cute.coord_tensor`: Coordinate tensor: an arithmetic tuple with a layout.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteCoordTensor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTensorTypeGet(MlirContext context,
                                                       MlirType arithTuple,
                                                       MlirType layout);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTensorTypeGetArithTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTensorTypeGetLayout(MlirType type);

// `!cute.coord`: Scalar integer, underscore wildcard, or recursive tuple of
// coordinates.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteCoord(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteCoordTypeGet(MlirContext context,
                                                 MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCoordTypeGetAttr(MlirType type);

// `!cute.fast_divmod_divisor`: Precomputed divisor of the fast divmod
// operations.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteFastDivmodDivisor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteFastDivmodDivisorTypeGet(
    MlirContext context, unsigned width, bool isPow2);
MLIR_CAPI_EXPORTED unsigned
mlirCuteFastDivmodDivisorTypeGetWidth(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteFastDivmodDivisorTypeGetIsPow2(MlirType type);

// `!cute.int_tuple`: Scalar integer or recursive tuple of integers.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteIntTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteIntTupleTypeGet(MlirContext context,
                                                    MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteIntTupleTypeGetAttr(MlirType type);

// `!cute.layout`: CuTe layout: a shape/stride pair.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteLayout(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteLayoutTypeGet(MlirContext context,
                                                  MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteLayoutTypeGetAttr(MlirType type);

// `!cute.memref`: Tensor view: a pointer with a layout.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteMemRef(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteMemRefTypeGet(MlirContext context,
                                                  MlirType ptr,
                                                  MlirType layout);
MLIR_CAPI_EXPORTED MlirType mlirCuteMemRefTypeGetPtr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteMemRefTypeGetLayout(MlirType type);

// `!cute.ptr`: Pointer with address space, alignment, swizzle and bit layout.
MLIR_CAPI_EXPORTED bool mlirTypeIsACutePtr(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCutePtrTypeGet(
    MlirContext context, MlirType valueType, uint32_t addressSpace,
    uint64_t alignment, MlirAttribute swizzle, MlirAttribute bitlayout);
MLIR_CAPI_EXPORTED MlirType mlirCutePtrTypeGetValueType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t mlirCutePtrTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED uint64_t mlirCutePtrTypeGetAlignment(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirCutePtrTypeGetSwizzle(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirCutePtrTypeGetBitlayout(MlirType type);

// `!cute.shape`: Scalar integer or recursive tuple of shape extents.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteShape(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteShapeTypeGet(MlirContext context,
                                                 MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteShapeTypeGetAttr(MlirType type);

// `!cute.sparse_elem`: Logical sparse element stored in a physical type.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteSparseElem(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteSparseElemTypeGet(MlirContext context,
                                                      int numLogical,
                                                      MlirType physicalType);
MLIR_CAPI_EXPORTED int mlirCuteSparseElemTypeGetNumLogical(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteSparseElemTypeGetPhysicalType(MlirType type);

// `!cute.stride`: Scalar or recursive tuple of stride elements.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteStride(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteStrideTypeGet(MlirContext context,
                                                  MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteStrideTypeGetAttr(MlirType type);

// `!cute.swizzle`: CuTe swizzle: S<num_bits, num_base, num_shift>.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteSwizzle(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteSwizzleTypeGet(MlirContext context,
                                                   MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteSwizzleTypeGetAttr(MlirType type);

// `!cute.tile`: CuTe tile: a recursive sequence of layouts and underscores.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteTile(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteTileTypeGet(MlirContext context,
                                                MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteTileTypeGetAttr(MlirType type);

// `!cute.tiled_copy`: Copy atom tiled over threads and values.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteTiledCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledCopyTypeGet(MlirContext context,
                                                     MlirType copyAtom,
                                                     MlirAttribute layoutCopyTv,
                                                     MlirAttribute tilerMn);
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledCopyTypeGetCopyAtom(MlirType type);
// Former name of mlirCuteTiledCopyTypeGetCopyAtom.
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledCopyTypeGetAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledCopyTypeGetLayoutCopyTv(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteTiledCopyTypeGetTilerMn(MlirType type);

// `!cute.tiled_copy_v2`: V2 copy atom tiled over threads and values.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteTiledCopyV2(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteTiledCopyV2TypeGet(MlirContext context, MlirType copyAtom,
                           MlirAttribute layoutCopyTv, MlirAttribute tilerMn);
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledCopyV2TypeGetCopyAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledCopyV2TypeGetLayoutCopyTv(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledCopyV2TypeGetTilerMn(MlirType type);

// `!cute.tiled_mma`: MMA atom tiled over MNK.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteTiledMma(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledMmaTypeGet(
    MlirContext context, MlirType mmaAtom, MlirAttribute atomLayoutMNK,
    MlirAttribute permutationMNK);
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledMmaTypeGetMmaAtom(MlirType type);
// Former name of mlirCuteTiledMmaTypeGetMmaAtom.
MLIR_CAPI_EXPORTED MlirType mlirCuteTiledMmaTypeGetAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledMmaTypeGetAtomLayoutMNK(MlirType type);
// Former name of mlirCuteTiledMmaTypeGetAtomLayoutMNK.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledMmaTypeGetAtomLayoutMnk(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledMmaTypeGetPermutationMNK(MlirType type);
// Former name of mlirCuteTiledMmaTypeGetPermutationMNK.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTiledMmaTypeGetPermutationMnk(MlirType type);

// `!cute.tuple`: Non-empty tuple of types.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteTuple(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteTupleTypeGet(MlirContext context,
                                                 intptr_t numTypes,
                                                 MlirType const *types);
MLIR_CAPI_EXPORTED intptr_t mlirCuteTupleTypeGetNumTypes(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteTupleTypeGetType(MlirType type,
                                                     intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTE_TYPES_H
