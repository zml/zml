//===- capi.h - C API for the cute dialect ------------------------*- C -*-===//

#ifndef MLIR_DIALECTS_CUTE_IR_CAPI_H
#define MLIR_DIALECTS_CUTE_IR_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registration

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cute, cute);

// Attributes

// Algebra attributes are built bottom up. A leaf is a static integer, a
// dynamic integer of `width` bits (1, 8, 16, 32 or 64) known to be a multiple
// of `divisibility`, a scaled basis (strides) or an underscore (coordinates
// and tiles); a tuple groups attributes of the same kind. A basis `modes` is
// its mode path, outermost first, which CuTe prints innermost first:
// `1@1@0` is modes {0, 1}. Constructors return null for invalid parameters or
// handles of another context. GetValue returns canonical algebra text owned
// by the attribute's context; callers must not free it. Getters require an
// attribute accepted by the corresponding IsA predicate.

MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteIntTupleAttrStaticGet(MlirContext context, int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteIntTupleAttrDynamicGet(
    MlirContext context, int32_t width, int64_t divisibility);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteIntTupleAttrTupleGet(
    MlirContext context, intptr_t numElements, MlirAttribute const *elements);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteIntTuple(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirCuteIntTupleAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCoordAttrStaticGet(MlirContext context,
                                                            int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCoordAttrDynamicGet(
    MlirContext context, int32_t width, int64_t divisibility);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCoordAttrUnderscoreGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCoordAttrTupleGet(
    MlirContext context, intptr_t numElements, MlirAttribute const *elements);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteCoord(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteCoordAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteShapeAttrStaticGet(MlirContext context,
                                                            int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteShapeAttrDynamicGet(
    MlirContext context, int32_t width, int64_t divisibility);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteShapeAttrTupleGet(
    MlirContext context, intptr_t numElements, MlirAttribute const *elements);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteShapeAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteStrideAttrStaticGet(MlirContext context, int64_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteStrideAttrDynamicGet(
    MlirContext context, int32_t width, int64_t divisibility);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteStrideAttrBasisGet(
    MlirContext context, int64_t numerator, int64_t denominator,
    intptr_t numModes, int32_t const *modes);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteStrideAttrTupleGet(
    MlirContext context, intptr_t numElements, MlirAttribute const *elements);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteStride(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteStrideAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteLayoutAttrGet(MlirContext context,
                                                       MlirAttribute shape,
                                                       MlirAttribute stride);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteLayoutAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTileAttrLayoutGet(MlirContext context, MlirAttribute layout);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteTileAttrUnderscoreGet(MlirContext context);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteTileAttrTupleGet(
    MlirContext context, intptr_t numElements, MlirAttribute const *elements);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteTile(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteTileAttrGetValue(MlirAttribute attr);

// `inner` is a swizzle or a layout attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteComposedLayoutAttrGet(MlirContext context, MlirAttribute inner,
                              MlirAttribute offset, MlirAttribute outer);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteComposedLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirCuteComposedLayoutAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirCuteSwizzleAttrGet(MlirContext context,
                                                        uint32_t bits,
                                                        uint32_t base,
                                                        int32_t shift);
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteSwizzle(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirCuteSwizzleAttrGetValue(MlirAttribute attr);

// Compiler attributes

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles, and std::optional enums a
// negative value. Enums are their integer values (the dialect's *Enums.td).
// Getters require a value accepted by the matching IsA.

// `#cute.bitlayout`: Bit layout of a pointer's elements within a storage chunk.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteBitLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteBitLayoutAttrGet(MlirContext context,
                                                          MlirAttribute layout);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteBitLayoutAttrGetLayout(MlirAttribute attr);

// `#cute.copy_atom`: Thread and value layouts of a copy atom.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteCopyAtom(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCopyAtomAttrGet(
    MlirContext context, MlirAttribute thrId, MlirAttribute layoutSrc,
    MlirAttribute layoutDst, MlirAttribute layoutRef);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomAttrGetThrId(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomAttrGetLayoutSrc(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomAttrGetLayoutDst(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomAttrGetLayoutRef(MlirAttribute attr);

// `#cute.copy_atom_v2`: Thread and value layouts and fragments of a V2 copy
// atom.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteCopyAtomV2(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteCopyAtomV2AttrGet(
    MlirContext context, MlirAttribute thrId, MlirAttribute layoutSrcTV,
    MlirAttribute layoutDstTV, MlirAttribute frgSrc, MlirAttribute frgDst);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomV2AttrGetThrId(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomV2AttrGetLayoutSrcTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomV2AttrGetLayoutDstTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomV2AttrGetFrgSrc(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteCopyAtomV2AttrGetFrgDst(MlirAttribute attr);

// `#cute.mma_atom`: Shapes, thread-value layouts and fragments of an MMA atom.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteMmaAtom(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutBTV, MlirAttribute layoutCTV,
    MlirAttribute shapeAMK, MlirAttribute shapeBNK, MlirAttribute shapeCMN,
    MlirAttribute frgA, MlirAttribute frgB, MlirAttribute frgC);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetShapeMNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetThrId(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetLayoutATV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetLayoutBTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetLayoutCTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetShapeAMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetShapeBNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMmaAtomAttrGetShapeCMN(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteMmaAtomAttrGetFrgA(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteMmaAtomAttrGetFrgB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteMmaAtomAttrGetFrgC(MlirAttribute attr);

// `#cute.mx_mma_atom`: MMA atom attribute with scale-factor operands.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteMxMmaAtom(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteMxMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutSFATV, MlirAttribute layoutBTV,
    MlirAttribute layoutSFBTV, MlirAttribute layoutCTV, MlirAttribute shapeAMK,
    MlirAttribute shapeSFAMK, MlirAttribute shapeBNK, MlirAttribute shapeSFBNK,
    MlirAttribute shapeCMN, MlirAttribute frgA, MlirAttribute frgSFA,
    MlirAttribute frgB, MlirAttribute frgSFB, MlirAttribute frgC);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetShapeMNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetThrId(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetLayoutATV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetLayoutSFATV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetLayoutBTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetLayoutSFBTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetLayoutCTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetShapeAMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetShapeSFAMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetShapeBNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetShapeSFBNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetShapeCMN(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetFrgA(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetFrgSFA(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetFrgB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetFrgSFB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteMxMmaAtomAttrGetFrgC(MlirAttribute attr);

// `#cute.reduction_op`: Op for cute reduce operations.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteReductionOp(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteReductionOpAttrGet(MlirContext context,
                                                            uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteReductionOpAttrGetValue(MlirAttribute attr);

// `#cute.sparse_mma_atom`: MMA atom attribute with a sparsity-metadata operand.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteSparseMmaAtom(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteSparseMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutBTV, MlirAttribute layoutCTV,
    MlirAttribute layoutETV, MlirAttribute shapeAMK, MlirAttribute shapeBNK,
    MlirAttribute shapeCMN, MlirAttribute shapeEMK, MlirAttribute frgA,
    MlirAttribute frgB, MlirAttribute frgC, MlirAttribute frgE);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetShapeMNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetThrId(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetLayoutATV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetLayoutBTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetLayoutCTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetLayoutETV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetShapeAMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetShapeBNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetShapeCMN(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetShapeEMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetFrgA(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetFrgB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetFrgC(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMmaAtomAttrGetFrgE(MlirAttribute attr);

// `#cute.sparse_mx_mma_atom`: MMA atom attribute with scale-factor and metadata
// operands.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteSparseMxMmaAtom(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteSparseMxMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutSFATV, MlirAttribute layoutBTV,
    MlirAttribute layoutSFBTV, MlirAttribute layoutCTV, MlirAttribute layoutETV,
    MlirAttribute shapeAMK, MlirAttribute shapeSFAMK, MlirAttribute shapeBNK,
    MlirAttribute shapeSFBNK, MlirAttribute shapeCMN, MlirAttribute shapeEMK,
    MlirAttribute frgA, MlirAttribute frgSFA, MlirAttribute frgB,
    MlirAttribute frgSFB, MlirAttribute frgC, MlirAttribute frgE);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeMNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetThrId(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetLayoutATV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetLayoutSFATV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetLayoutBTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetLayoutSFBTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetLayoutCTV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetLayoutETV(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeAMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeSFAMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeBNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeSFBNK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeCMN(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetShapeEMK(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetFrgA(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetFrgSFA(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetFrgB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetFrgSFB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetFrgC(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteSparseMxMmaAtomAttrGetFrgE(MlirAttribute attr);

// Types

/// Returns true when `type` belongs to the CuTe dialect.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteType(MlirType type);

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
                                                          uint32_t width,
                                                          bool isPow2);
MLIR_CAPI_EXPORTED int64_t
mlirCuteConstrainedIntTypeGetDivisibility(MlirType type);
MLIR_CAPI_EXPORTED uint32_t mlirCuteConstrainedIntTypeGetWidth(MlirType type);
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
    MlirContext context, uint32_t width, bool isPow2);
MLIR_CAPI_EXPORTED uint32_t
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

#endif // MLIR_DIALECTS_CUTE_IR_CAPI_H
