// C API for the CuTe attributes beyond the algebra, generated from the
// dialect's .td files.
#ifndef CUTE_IR_C_DIALECT_CUTE_COMPILER_ATTRIBUTES_H
#define CUTE_IR_C_DIALECT_CUTE_COMPILER_ATTRIBUTES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

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

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTE_COMPILER_ATTRIBUTES_H
