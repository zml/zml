// C API for the CuTe attributes beyond the algebra, generated from the
// dialect's .td files.
#include "cute_ir-c/Dialect/CuteCompilerAttributes.h"

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
#include "mlir/CAPI/IR.h"

namespace cute = mlir::cutlass_compiler::cute;

namespace {
// Unwraps `handle` into `out`: false when it is of another context or not a
// T, or when it is null and not optional.
template <typename T, typename H>
bool unwrapAs(mlir::MLIRContext *ctx, H handle, bool optional, T &out) {
  auto value = unwrap(handle);
  if (!value)
    return optional;
  if (value.getContext() != ctx)
    return false;
  out = llvm::dyn_cast<T>(value);
  return static_cast<bool>(out);
}
} // namespace

bool mlirAttributeIsACuteBitLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::BitLayoutAttr>(unwrap(attr));
}
MlirAttribute mlirCuteBitLayoutAttrGet(MlirContext context,
                                       MlirAttribute layout) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutValue;
  if (!unwrapAs(ctx, layout, false, layoutValue))
    return {nullptr};
  return wrap(cute::BitLayoutAttr::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      layoutValue));
}
MlirAttribute mlirCuteBitLayoutAttrGetLayout(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::BitLayoutAttr>(unwrap(attr)).getLayout()));
}

bool mlirAttributeIsACuteCopyAtom(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::CopyAtomAttr>(unwrap(attr));
}
MlirAttribute mlirCuteCopyAtomAttrGet(MlirContext context, MlirAttribute thrId,
                                      MlirAttribute layoutSrc,
                                      MlirAttribute layoutDst,
                                      MlirAttribute layoutRef) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::LayoutAttr thrIdValue;
  if (!unwrapAs(ctx, thrId, false, thrIdValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutSrcValue;
  if (!unwrapAs(ctx, layoutSrc, false, layoutSrcValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutDstValue;
  if (!unwrapAs(ctx, layoutDst, false, layoutDstValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutRefValue;
  if (!unwrapAs(ctx, layoutRef, false, layoutRefValue))
    return {nullptr};
  return wrap(cute::CopyAtomAttr::get(ctx, thrIdValue, layoutSrcValue,
                                      layoutDstValue, layoutRefValue));
}
MlirAttribute mlirCuteCopyAtomAttrGetThrId(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomAttr>(unwrap(attr)).getThrId()));
}
MlirAttribute mlirCuteCopyAtomAttrGetLayoutSrc(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomAttr>(unwrap(attr)).getLayoutSrc()));
}
MlirAttribute mlirCuteCopyAtomAttrGetLayoutDst(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomAttr>(unwrap(attr)).getLayoutDst()));
}
MlirAttribute mlirCuteCopyAtomAttrGetLayoutRef(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomAttr>(unwrap(attr)).getLayoutRef()));
}

bool mlirAttributeIsACuteCopyAtomV2(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::CopyAtomV2Attr>(unwrap(attr));
}
MlirAttribute
mlirCuteCopyAtomV2AttrGet(MlirContext context, MlirAttribute thrId,
                          MlirAttribute layoutSrcTV, MlirAttribute layoutDstTV,
                          MlirAttribute frgSrc, MlirAttribute frgDst) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::LayoutAttr thrIdValue;
  if (!unwrapAs(ctx, thrId, false, thrIdValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutSrcTVValue;
  if (!unwrapAs(ctx, layoutSrcTV, false, layoutSrcTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutDstTVValue;
  if (!unwrapAs(ctx, layoutDstTV, false, layoutDstTVValue))
    return {nullptr};
  mlir::Attribute frgSrcValue;
  if (!unwrapAs(ctx, frgSrc, false, frgSrcValue))
    return {nullptr};
  mlir::Attribute frgDstValue;
  if (!unwrapAs(ctx, frgDst, false, frgDstValue))
    return {nullptr};
  return wrap(cute::CopyAtomV2Attr::get(ctx, thrIdValue, layoutSrcTVValue,
                                        layoutDstTVValue, frgSrcValue,
                                        frgDstValue));
}
MlirAttribute mlirCuteCopyAtomV2AttrGetThrId(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomV2Attr>(unwrap(attr)).getThrId()));
}
MlirAttribute mlirCuteCopyAtomV2AttrGetLayoutSrcTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomV2Attr>(unwrap(attr)).getLayoutSrc_TV()));
}
MlirAttribute mlirCuteCopyAtomV2AttrGetLayoutDstTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomV2Attr>(unwrap(attr)).getLayoutDst_TV()));
}
MlirAttribute mlirCuteCopyAtomV2AttrGetFrgSrc(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomV2Attr>(unwrap(attr)).getFrgSrc()));
}
MlirAttribute mlirCuteCopyAtomV2AttrGetFrgDst(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CopyAtomV2Attr>(unwrap(attr)).getFrgDst()));
}

bool mlirAttributeIsACuteMmaAtom(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::MmaAtomAttr>(unwrap(attr));
}
MlirAttribute mlirCuteMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutBTV, MlirAttribute layoutCTV,
    MlirAttribute shapeAMK, MlirAttribute shapeBNK, MlirAttribute shapeCMN,
    MlirAttribute frgA, MlirAttribute frgB, MlirAttribute frgC) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMNKValue;
  if (!unwrapAs(ctx, shapeMNK, false, shapeMNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr thrIdValue;
  if (!unwrapAs(ctx, thrId, false, thrIdValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutATVValue;
  if (!unwrapAs(ctx, layoutATV, false, layoutATVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutBTVValue;
  if (!unwrapAs(ctx, layoutBTV, false, layoutBTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutCTVValue;
  if (!unwrapAs(ctx, layoutCTV, false, layoutCTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeAMKValue;
  if (!unwrapAs(ctx, shapeAMK, false, shapeAMKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeBNKValue;
  if (!unwrapAs(ctx, shapeBNK, false, shapeBNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeCMNValue;
  if (!unwrapAs(ctx, shapeCMN, false, shapeCMNValue))
    return {nullptr};
  mlir::Attribute frgAValue;
  if (!unwrapAs(ctx, frgA, false, frgAValue))
    return {nullptr};
  mlir::Attribute frgBValue;
  if (!unwrapAs(ctx, frgB, false, frgBValue))
    return {nullptr};
  mlir::Attribute frgCValue;
  if (!unwrapAs(ctx, frgC, false, frgCValue))
    return {nullptr};
  return wrap(cute::MmaAtomAttr::get(
      ctx, shapeMNKValue, thrIdValue, layoutATVValue, layoutBTVValue,
      layoutCTVValue, shapeAMKValue, shapeBNKValue, shapeCMNValue, frgAValue,
      frgBValue, frgCValue));
}
MlirAttribute mlirCuteMmaAtomAttrGetShapeMNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getShape_MNK()));
}
MlirAttribute mlirCuteMmaAtomAttrGetThrId(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getThrId()));
}
MlirAttribute mlirCuteMmaAtomAttrGetLayoutATV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getLayoutA_TV()));
}
MlirAttribute mlirCuteMmaAtomAttrGetLayoutBTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getLayoutB_TV()));
}
MlirAttribute mlirCuteMmaAtomAttrGetLayoutCTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getLayoutC_TV()));
}
MlirAttribute mlirCuteMmaAtomAttrGetShapeAMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getShapeA_MK()));
}
MlirAttribute mlirCuteMmaAtomAttrGetShapeBNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getShapeB_NK()));
}
MlirAttribute mlirCuteMmaAtomAttrGetShapeCMN(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getShapeC_MN()));
}
MlirAttribute mlirCuteMmaAtomAttrGetFrgA(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getFrgA()));
}
MlirAttribute mlirCuteMmaAtomAttrGetFrgB(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getFrgB()));
}
MlirAttribute mlirCuteMmaAtomAttrGetFrgC(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MmaAtomAttr>(unwrap(attr)).getFrgC()));
}

bool mlirAttributeIsACuteMxMmaAtom(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::MxMmaAtomAttr>(unwrap(attr));
}
MlirAttribute mlirCuteMxMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutSFATV, MlirAttribute layoutBTV,
    MlirAttribute layoutSFBTV, MlirAttribute layoutCTV, MlirAttribute shapeAMK,
    MlirAttribute shapeSFAMK, MlirAttribute shapeBNK, MlirAttribute shapeSFBNK,
    MlirAttribute shapeCMN, MlirAttribute frgA, MlirAttribute frgSFA,
    MlirAttribute frgB, MlirAttribute frgSFB, MlirAttribute frgC) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMNKValue;
  if (!unwrapAs(ctx, shapeMNK, false, shapeMNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr thrIdValue;
  if (!unwrapAs(ctx, thrId, false, thrIdValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutATVValue;
  if (!unwrapAs(ctx, layoutATV, false, layoutATVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutSFATVValue;
  if (!unwrapAs(ctx, layoutSFATV, false, layoutSFATVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutBTVValue;
  if (!unwrapAs(ctx, layoutBTV, false, layoutBTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutSFBTVValue;
  if (!unwrapAs(ctx, layoutSFBTV, false, layoutSFBTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutCTVValue;
  if (!unwrapAs(ctx, layoutCTV, false, layoutCTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeAMKValue;
  if (!unwrapAs(ctx, shapeAMK, false, shapeAMKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeSFAMKValue;
  if (!unwrapAs(ctx, shapeSFAMK, false, shapeSFAMKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeBNKValue;
  if (!unwrapAs(ctx, shapeBNK, false, shapeBNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeSFBNKValue;
  if (!unwrapAs(ctx, shapeSFBNK, false, shapeSFBNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeCMNValue;
  if (!unwrapAs(ctx, shapeCMN, false, shapeCMNValue))
    return {nullptr};
  mlir::Attribute frgAValue;
  if (!unwrapAs(ctx, frgA, false, frgAValue))
    return {nullptr};
  mlir::Attribute frgSFAValue;
  if (!unwrapAs(ctx, frgSFA, false, frgSFAValue))
    return {nullptr};
  mlir::Attribute frgBValue;
  if (!unwrapAs(ctx, frgB, false, frgBValue))
    return {nullptr};
  mlir::Attribute frgSFBValue;
  if (!unwrapAs(ctx, frgSFB, false, frgSFBValue))
    return {nullptr};
  mlir::Attribute frgCValue;
  if (!unwrapAs(ctx, frgC, false, frgCValue))
    return {nullptr};
  return wrap(cute::MxMmaAtomAttr::get(
      ctx, shapeMNKValue, thrIdValue, layoutATVValue, layoutSFATVValue,
      layoutBTVValue, layoutSFBTVValue, layoutCTVValue, shapeAMKValue,
      shapeSFAMKValue, shapeBNKValue, shapeSFBNKValue, shapeCMNValue, frgAValue,
      frgSFAValue, frgBValue, frgSFBValue, frgCValue));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetShapeMNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getShape_MNK()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetThrId(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getThrId()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetLayoutATV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getLayoutA_TV()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetLayoutSFATV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getLayoutSFA_TV()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetLayoutBTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getLayoutB_TV()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetLayoutSFBTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getLayoutSFB_TV()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetLayoutCTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getLayoutC_TV()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetShapeAMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getShapeA_MK()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetShapeSFAMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getShapeSFA_MK()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetShapeBNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getShapeB_NK()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetShapeSFBNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getShapeSFB_NK()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetShapeCMN(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getShapeC_MN()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetFrgA(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getFrgA()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetFrgSFA(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getFrgSFA()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetFrgB(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getFrgB()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetFrgSFB(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getFrgSFB()));
}
MlirAttribute mlirCuteMxMmaAtomAttrGetFrgC(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::MxMmaAtomAttr>(unwrap(attr)).getFrgC()));
}

bool mlirAttributeIsACuteReductionOp(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::ReductionOpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteReductionOpAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute::symbolizeReductionOp(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute::ReductionOpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteReductionOpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute::ReductionOpAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteSparseMmaAtom(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::SparseMmaAtomAttr>(unwrap(attr));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutBTV, MlirAttribute layoutCTV,
    MlirAttribute layoutETV, MlirAttribute shapeAMK, MlirAttribute shapeBNK,
    MlirAttribute shapeCMN, MlirAttribute shapeEMK, MlirAttribute frgA,
    MlirAttribute frgB, MlirAttribute frgC, MlirAttribute frgE) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMNKValue;
  if (!unwrapAs(ctx, shapeMNK, false, shapeMNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr thrIdValue;
  if (!unwrapAs(ctx, thrId, false, thrIdValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutATVValue;
  if (!unwrapAs(ctx, layoutATV, false, layoutATVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutBTVValue;
  if (!unwrapAs(ctx, layoutBTV, false, layoutBTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutCTVValue;
  if (!unwrapAs(ctx, layoutCTV, false, layoutCTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutETVValue;
  if (!unwrapAs(ctx, layoutETV, false, layoutETVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeAMKValue;
  if (!unwrapAs(ctx, shapeAMK, false, shapeAMKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeBNKValue;
  if (!unwrapAs(ctx, shapeBNK, false, shapeBNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeCMNValue;
  if (!unwrapAs(ctx, shapeCMN, false, shapeCMNValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeEMKValue;
  if (!unwrapAs(ctx, shapeEMK, false, shapeEMKValue))
    return {nullptr};
  mlir::Attribute frgAValue;
  if (!unwrapAs(ctx, frgA, false, frgAValue))
    return {nullptr};
  mlir::Attribute frgBValue;
  if (!unwrapAs(ctx, frgB, false, frgBValue))
    return {nullptr};
  mlir::Attribute frgCValue;
  if (!unwrapAs(ctx, frgC, false, frgCValue))
    return {nullptr};
  mlir::Attribute frgEValue;
  if (!unwrapAs(ctx, frgE, false, frgEValue))
    return {nullptr};
  return wrap(cute::SparseMmaAtomAttr::get(
      ctx, shapeMNKValue, thrIdValue, layoutATVValue, layoutBTVValue,
      layoutCTVValue, layoutETVValue, shapeAMKValue, shapeBNKValue,
      shapeCMNValue, shapeEMKValue, frgAValue, frgBValue, frgCValue,
      frgEValue));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetShapeMNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getShape_MNK()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetThrId(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getThrId()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetLayoutATV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getLayoutA_TV()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetLayoutBTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getLayoutB_TV()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetLayoutCTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getLayoutC_TV()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetLayoutETV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getLayoutE_TV()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetShapeAMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getShapeA_MK()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetShapeBNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getShapeB_NK()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetShapeCMN(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getShapeC_MN()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetShapeEMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getShapeE_MK()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetFrgA(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getFrgA()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetFrgB(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getFrgB()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetFrgC(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getFrgC()));
}
MlirAttribute mlirCuteSparseMmaAtomAttrGetFrgE(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMmaAtomAttr>(unwrap(attr)).getFrgE()));
}

bool mlirAttributeIsACuteSparseMxMmaAtom(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::SparseMxMmaAtomAttr>(unwrap(attr));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGet(
    MlirContext context, MlirAttribute shapeMNK, MlirAttribute thrId,
    MlirAttribute layoutATV, MlirAttribute layoutSFATV, MlirAttribute layoutBTV,
    MlirAttribute layoutSFBTV, MlirAttribute layoutCTV, MlirAttribute layoutETV,
    MlirAttribute shapeAMK, MlirAttribute shapeSFAMK, MlirAttribute shapeBNK,
    MlirAttribute shapeSFBNK, MlirAttribute shapeCMN, MlirAttribute shapeEMK,
    MlirAttribute frgA, MlirAttribute frgSFA, MlirAttribute frgB,
    MlirAttribute frgSFB, MlirAttribute frgC, MlirAttribute frgE) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMNKValue;
  if (!unwrapAs(ctx, shapeMNK, false, shapeMNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr thrIdValue;
  if (!unwrapAs(ctx, thrId, false, thrIdValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutATVValue;
  if (!unwrapAs(ctx, layoutATV, false, layoutATVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutSFATVValue;
  if (!unwrapAs(ctx, layoutSFATV, false, layoutSFATVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutBTVValue;
  if (!unwrapAs(ctx, layoutBTV, false, layoutBTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutSFBTVValue;
  if (!unwrapAs(ctx, layoutSFBTV, false, layoutSFBTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutCTVValue;
  if (!unwrapAs(ctx, layoutCTV, false, layoutCTVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutETVValue;
  if (!unwrapAs(ctx, layoutETV, false, layoutETVValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeAMKValue;
  if (!unwrapAs(ctx, shapeAMK, false, shapeAMKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeSFAMKValue;
  if (!unwrapAs(ctx, shapeSFAMK, false, shapeSFAMKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeBNKValue;
  if (!unwrapAs(ctx, shapeBNK, false, shapeBNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeSFBNKValue;
  if (!unwrapAs(ctx, shapeSFBNK, false, shapeSFBNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeCMNValue;
  if (!unwrapAs(ctx, shapeCMN, false, shapeCMNValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeEMKValue;
  if (!unwrapAs(ctx, shapeEMK, false, shapeEMKValue))
    return {nullptr};
  mlir::Attribute frgAValue;
  if (!unwrapAs(ctx, frgA, false, frgAValue))
    return {nullptr};
  mlir::Attribute frgSFAValue;
  if (!unwrapAs(ctx, frgSFA, false, frgSFAValue))
    return {nullptr};
  mlir::Attribute frgBValue;
  if (!unwrapAs(ctx, frgB, false, frgBValue))
    return {nullptr};
  mlir::Attribute frgSFBValue;
  if (!unwrapAs(ctx, frgSFB, false, frgSFBValue))
    return {nullptr};
  mlir::Attribute frgCValue;
  if (!unwrapAs(ctx, frgC, false, frgCValue))
    return {nullptr};
  mlir::Attribute frgEValue;
  if (!unwrapAs(ctx, frgE, false, frgEValue))
    return {nullptr};
  return wrap(cute::SparseMxMmaAtomAttr::get(
      ctx, shapeMNKValue, thrIdValue, layoutATVValue, layoutSFATVValue,
      layoutBTVValue, layoutSFBTVValue, layoutCTVValue, layoutETVValue,
      shapeAMKValue, shapeSFAMKValue, shapeBNKValue, shapeSFBNKValue,
      shapeCMNValue, shapeEMKValue, frgAValue, frgSFAValue, frgBValue,
      frgSFBValue, frgCValue, frgEValue));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeMNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShape_MNK()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetThrId(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getThrId()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetLayoutATV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getLayoutA_TV()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetLayoutSFATV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getLayoutSFA_TV()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetLayoutBTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getLayoutB_TV()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetLayoutSFBTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getLayoutSFB_TV()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetLayoutCTV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getLayoutC_TV()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetLayoutETV(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getLayoutE_TV()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeAMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShapeA_MK()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeSFAMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShapeSFA_MK()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeBNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShapeB_NK()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeSFBNK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShapeSFB_NK()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeCMN(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShapeC_MN()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetShapeEMK(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getShapeE_MK()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetFrgA(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getFrgA()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetFrgSFA(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getFrgSFA()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetFrgB(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getFrgB()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetFrgSFB(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getFrgSFB()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetFrgC(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getFrgC()));
}
MlirAttribute mlirCuteSparseMxMmaAtomAttrGetFrgE(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SparseMxMmaAtomAttr>(unwrap(attr)).getFrgE()));
}
