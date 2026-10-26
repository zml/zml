//===- capi.cc - C API for the cute dialect -------------------------------===//

#include "capi.h"

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

// Registration

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Cute, cute,
                                      mlir::cutlass_compiler::cute::CuteDialect)

// Attributes

namespace cute = mlir::cutlass_compiler::cute;

// Unwraps `handle` into `out`: false when it is of another context or not a
// T, or when it is null and not optional.
template <typename T, typename H>
static bool unwrapAs(mlir::MLIRContext *ctx, H handle, bool optional, T &out) {
  auto value = unwrap(handle);
  if (!value)
    return optional;
  if (value.getContext() != ctx)
    return false;
  out = llvm::dyn_cast<T>(value);
  return static_cast<bool>(out);
}

template <typename Attr>
static MlirAttribute getAlgebraAttr(MlirContext context,
                                    typename Attr::algebra_t algebra) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(Attr::get(ctx, std::move(algebra)));
}

template <typename Attr>
static MlirAttribute getStaticAttr(MlirContext context, int64_t value) {
  return getAlgebraAttr<Attr>(
      context, typename Attr::algebra_t(cutegen::static_int_t(value)));
}

template <typename Attr>
static MlirAttribute getDynamicAttr(MlirContext context, int32_t width,
                                    int64_t divisibility) {
  if (!llvm::is_contained({1, 8, 16, 32, 64}, width) || divisibility <= 0)
    return {nullptr};
  return getAlgebraAttr<Attr>(
      context, typename Attr::algebra_t(cutegen::mlir_dynamic_t(
                   cutegen::dynamic_int_properties_t(width, divisibility))));
}

template <typename Attr>
static MlirAttribute getUnderscoreAttr(MlirContext context) {
  return getAlgebraAttr<Attr>(
      context, typename Attr::algebra_t(cutegen::underscore_t()));
}

template <typename Attr>
static MlirAttribute getTupleAttr(MlirContext context, intptr_t numElements,
                                  MlirAttribute const *elements) {
  auto *ctx = unwrap(context);
  if (!ctx || numElements < 0 || (numElements > 0 && !elements))
    return {nullptr};
  typename Attr::algebra_t::vector_t children;
  children.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i) {
    Attr element;
    if (!unwrapAs(ctx, elements[i], /*optional=*/false, element))
      return {nullptr};
    children.push_back(element.getRef());
  }
  return getAlgebraAttr<Attr>(context,
                              typename Attr::algebra_t(std::move(children)));
}

template <typename Attr>
static MlirStringRef getAlgebraValue(MlirAttribute attr) {
  auto value = llvm::cast<Attr>(unwrap(attr));
  // Intern the serialization so the returned view outlives this function.
  auto text = mlir::StringAttr::get(value.getContext(),
                                    cutegen::to_string(value.getRef()));
  return wrap(text.getValue());
}

MlirAttribute mlirCuteIntTupleAttrStaticGet(MlirContext context,
                                            int64_t value) {
  return getStaticAttr<cute::IntTupleAttr>(context, value);
}

MlirAttribute mlirCuteIntTupleAttrDynamicGet(MlirContext context, int32_t width,
                                             int64_t divisibility) {
  return getDynamicAttr<cute::IntTupleAttr>(context, width, divisibility);
}

MlirAttribute mlirCuteIntTupleAttrTupleGet(MlirContext context,
                                           intptr_t numElements,
                                           MlirAttribute const *elements) {
  return getTupleAttr<cute::IntTupleAttr>(context, numElements, elements);
}

bool mlirAttributeIsACuteIntTuple(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::IntTupleAttr>(unwrap(attr));
}

MlirStringRef mlirCuteIntTupleAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::IntTupleAttr>(attr);
}

MlirAttribute mlirCuteCoordAttrStaticGet(MlirContext context, int64_t value) {
  return getStaticAttr<cute::CoordAttr>(context, value);
}

MlirAttribute mlirCuteCoordAttrDynamicGet(MlirContext context, int32_t width,
                                          int64_t divisibility) {
  return getDynamicAttr<cute::CoordAttr>(context, width, divisibility);
}

MlirAttribute mlirCuteCoordAttrUnderscoreGet(MlirContext context) {
  return getUnderscoreAttr<cute::CoordAttr>(context);
}

MlirAttribute mlirCuteCoordAttrTupleGet(MlirContext context,
                                        intptr_t numElements,
                                        MlirAttribute const *elements) {
  return getTupleAttr<cute::CoordAttr>(context, numElements, elements);
}

bool mlirAttributeIsACuteCoord(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::CoordAttr>(unwrap(attr));
}

MlirStringRef mlirCuteCoordAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::CoordAttr>(attr);
}

MlirAttribute mlirCuteShapeAttrStaticGet(MlirContext context, int64_t value) {
  return getStaticAttr<cute::ShapeAttr>(context, value);
}

MlirAttribute mlirCuteShapeAttrDynamicGet(MlirContext context, int32_t width,
                                          int64_t divisibility) {
  return getDynamicAttr<cute::ShapeAttr>(context, width, divisibility);
}

MlirAttribute mlirCuteShapeAttrTupleGet(MlirContext context,
                                        intptr_t numElements,
                                        MlirAttribute const *elements) {
  return getTupleAttr<cute::ShapeAttr>(context, numElements, elements);
}

bool mlirAttributeIsACuteShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::ShapeAttr>(unwrap(attr));
}

MlirStringRef mlirCuteShapeAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::ShapeAttr>(attr);
}

MlirAttribute mlirCuteStrideAttrStaticGet(MlirContext context, int64_t value) {
  return getStaticAttr<cute::StrideAttr>(context, value);
}

MlirAttribute mlirCuteStrideAttrDynamicGet(MlirContext context, int32_t width,
                                           int64_t divisibility) {
  return getDynamicAttr<cute::StrideAttr>(context, width, divisibility);
}

MlirAttribute mlirCuteStrideAttrBasisGet(MlirContext context, int64_t numerator,
                                         int64_t denominator, intptr_t numModes,
                                         int32_t const *modes) {
  if (denominator <= 0 || numModes <= 0 || !modes)
    return {nullptr};
  cutegen::scaled_basis::vector_t path(modes, modes + numModes);
  auto basis =
      denominator == 1
          ? cutegen::scaled_basis(std::move(path), numerator)
          : cutegen::scaled_basis(std::move(path),
                                  cutegen::ratio(numerator, denominator));
  return getAlgebraAttr<cute::StrideAttr>(context,
                                          cutegen::stride(std::move(basis)));
}

MlirAttribute mlirCuteStrideAttrTupleGet(MlirContext context,
                                         intptr_t numElements,
                                         MlirAttribute const *elements) {
  return getTupleAttr<cute::StrideAttr>(context, numElements, elements);
}

bool mlirAttributeIsACuteStride(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::StrideAttr>(unwrap(attr));
}

MlirStringRef mlirCuteStrideAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::StrideAttr>(attr);
}

MlirAttribute mlirCuteLayoutAttrGet(MlirContext context, MlirAttribute shape,
                                    MlirAttribute stride) {
  auto *ctx = unwrap(context);
  cute::ShapeAttr shapeAttr;
  cute::StrideAttr strideAttr;
  if (!ctx || !unwrapAs(ctx, shape, /*optional=*/false, shapeAttr) ||
      !unwrapAs(ctx, stride, /*optional=*/false, strideAttr))
    return {nullptr};
  cutegen::layout layout(shapeAttr.getRef(), strideAttr.getRef());
  if (cutegen::has_error(layout))
    return {nullptr};
  return getAlgebraAttr<cute::LayoutAttr>(context, std::move(layout));
}

bool mlirAttributeIsACuteLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::LayoutAttr>(unwrap(attr));
}

MlirStringRef mlirCuteLayoutAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::LayoutAttr>(attr);
}

MlirAttribute mlirCuteTileAttrLayoutGet(MlirContext context,
                                        MlirAttribute layout) {
  auto *ctx = unwrap(context);
  cute::LayoutAttr layoutAttr;
  if (!ctx || !unwrapAs(ctx, layout, /*optional=*/false, layoutAttr))
    return {nullptr};
  return getAlgebraAttr<cute::TileAttr>(context,
                                        cutegen::tile(layoutAttr.getRef()));
}

MlirAttribute mlirCuteTileAttrUnderscoreGet(MlirContext context) {
  return getUnderscoreAttr<cute::TileAttr>(context);
}

MlirAttribute mlirCuteTileAttrTupleGet(MlirContext context,
                                       intptr_t numElements,
                                       MlirAttribute const *elements) {
  return getTupleAttr<cute::TileAttr>(context, numElements, elements);
}

bool mlirAttributeIsACuteTile(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::TileAttr>(unwrap(attr));
}

MlirStringRef mlirCuteTileAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::TileAttr>(attr);
}

MlirAttribute mlirCuteComposedLayoutAttrGet(MlirContext context,
                                            MlirAttribute inner,
                                            MlirAttribute offset,
                                            MlirAttribute outer) {
  auto *ctx = unwrap(context);
  cute::IntTupleAttr offsetAttr;
  cute::LayoutAttr outerAttr;
  if (!ctx || !unwrapAs(ctx, offset, /*optional=*/false, offsetAttr) ||
      !unwrapAs(ctx, outer, /*optional=*/false, outerAttr))
    return {nullptr};
  cute::SwizzleAttr swizzleAttr;
  cute::LayoutAttr layoutAttr;
  std::optional<cutegen::composed_layout> composed;
  if (unwrapAs(ctx, inner, /*optional=*/false, swizzleAttr))
    composed.emplace(swizzleAttr.getRef(), offsetAttr.getRef(),
                     outerAttr.getRef());
  else if (unwrapAs(ctx, inner, /*optional=*/false, layoutAttr))
    composed.emplace(layoutAttr.getRef(), offsetAttr.getRef(),
                     outerAttr.getRef());
  if (!composed || cutegen::has_error(*composed))
    return {nullptr};
  return getAlgebraAttr<cute::ComposedLayoutAttr>(context,
                                                  std::move(*composed));
}

bool mlirAttributeIsACuteComposedLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::ComposedLayoutAttr>(unwrap(attr));
}

MlirStringRef mlirCuteComposedLayoutAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::ComposedLayoutAttr>(attr);
}

MlirAttribute mlirCuteSwizzleAttrGet(MlirContext context, uint32_t bits,
                                     uint32_t base, int32_t shift) {
  auto swizzle = cutegen::swizzle::get(bits, base, shift);
  if (!swizzle)
    return {nullptr};
  return getAlgebraAttr<cute::SwizzleAttr>(context, *swizzle);
}

bool mlirAttributeIsACuteSwizzle(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::SwizzleAttr>(unwrap(attr));
}

MlirStringRef mlirCuteSwizzleAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::SwizzleAttr>(attr);
}

// Compiler attributes

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

// Types

bool mlirTypeIsACuteType(MlirType type) {
  mlir::Type cppType = unwrap(type);
  return cppType &&
         cppType.getDialect().getNamespace() ==
             mlir::cutlass_compiler::cute::CuteDialect::getDialectNamespace();
}

bool mlirTypeIsACuteArithTupleIterator(MlirType type) {
  return llvm::isa_and_nonnull<cute::ArithTupleIteratorType>(unwrap(type));
}
MlirType mlirCuteArithTupleIteratorTypeGet(MlirContext context,
                                           MlirType arithTuple) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::IntTupleType arithTupleValue;
  if (!unwrapAs(ctx, arithTuple, false, arithTupleValue))
    return {nullptr};
  return wrap(cute::ArithTupleIteratorType::get(ctx, arithTupleValue));
}
MlirType mlirCuteArithTupleIteratorTypeGetArithTuple(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::ArithTupleIteratorType>(unwrap(type)).getArithTuple()));
}

bool mlirTypeIsACuteComposedLayout(MlirType type) {
  return llvm::isa_and_nonnull<cute::ComposedLayoutType>(unwrap(type));
}
MlirType mlirCuteComposedLayoutTypeGet(MlirContext context,
                                       MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::ComposedLayoutAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::ComposedLayoutType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteComposedLayoutTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::ComposedLayoutType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteConstrainedInt(MlirType type) {
  return llvm::isa_and_nonnull<cute::ConstrainedIntType>(unwrap(type));
}
MlirType mlirCuteConstrainedIntTypeGet(MlirContext context,
                                       int64_t divisibility, uint32_t width,
                                       bool isPow2) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(cute::ConstrainedIntType::get(ctx, divisibility, width, isPow2));
}
int64_t mlirCuteConstrainedIntTypeGetDivisibility(MlirType type) {
  return llvm::cast<cute::ConstrainedIntType>(unwrap(type)).getDivisibility();
}
uint32_t mlirCuteConstrainedIntTypeGetWidth(MlirType type) {
  return llvm::cast<cute::ConstrainedIntType>(unwrap(type)).getWidth();
}
bool mlirCuteConstrainedIntTypeGetIsPow2(MlirType type) {
  return llvm::cast<cute::ConstrainedIntType>(unwrap(type)).getIsPow2();
}

bool mlirTypeIsACuteCoordTensor(MlirType type) {
  return llvm::isa_and_nonnull<cute::CoordTensorType>(unwrap(type));
}
MlirType mlirCuteCoordTensorTypeGet(MlirContext context, MlirType arithTuple,
                                    MlirType layout) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::IntTupleType arithTupleValue;
  if (!unwrapAs(ctx, arithTuple, false, arithTupleValue))
    return {nullptr};
  mlir::Type layoutValue;
  if (!unwrapAs(ctx, layout, false, layoutValue))
    return {nullptr};
  return wrap(cute::CoordTensorType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      arithTupleValue, layoutValue));
}
MlirType mlirCuteCoordTensorTypeGetArithTuple(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::CoordTensorType>(unwrap(type)).getArithTuple()));
}
MlirType mlirCuteCoordTensorTypeGetLayout(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::CoordTensorType>(unwrap(type)).getLayout()));
}

bool mlirTypeIsACuteCoord(MlirType type) {
  return llvm::isa_and_nonnull<cute::CoordType>(unwrap(type));
}
MlirType mlirCuteCoordTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::CoordAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::CoordType::get(ctx, attrValue));
}
MlirAttribute mlirCuteCoordTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::CoordType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteFastDivmodDivisor(MlirType type) {
  return llvm::isa_and_nonnull<cute::FastDivmodDivisorType>(unwrap(type));
}
MlirType mlirCuteFastDivmodDivisorTypeGet(MlirContext context, uint32_t width,
                                          bool isPow2) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(cute::FastDivmodDivisorType::get(ctx, width, isPow2));
}
uint32_t mlirCuteFastDivmodDivisorTypeGetWidth(MlirType type) {
  return llvm::cast<cute::FastDivmodDivisorType>(unwrap(type)).getWidth();
}
bool mlirCuteFastDivmodDivisorTypeGetIsPow2(MlirType type) {
  return llvm::cast<cute::FastDivmodDivisorType>(unwrap(type)).getIsPow_2();
}

bool mlirTypeIsACuteIntTuple(MlirType type) {
  return llvm::isa_and_nonnull<cute::IntTupleType>(unwrap(type));
}
MlirType mlirCuteIntTupleTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::IntTupleAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::IntTupleType::get(ctx, attrValue));
}
MlirAttribute mlirCuteIntTupleTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::IntTupleType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteLayout(MlirType type) {
  return llvm::isa_and_nonnull<cute::LayoutType>(unwrap(type));
}
MlirType mlirCuteLayoutTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::LayoutAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::LayoutType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteLayoutTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::LayoutType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteMemRef(MlirType type) {
  return llvm::isa_and_nonnull<cute::MemRefType>(unwrap(type));
}
MlirType mlirCuteMemRefTypeGet(MlirContext context, MlirType ptr,
                               MlirType layout) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::PtrType ptrValue;
  if (!unwrapAs(ctx, ptr, false, ptrValue))
    return {nullptr};
  mlir::Type layoutValue;
  if (!unwrapAs(ctx, layout, false, layoutValue))
    return {nullptr};
  return wrap(cute::MemRefType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      ptrValue, layoutValue));
}
MlirType mlirCuteMemRefTypeGetPtr(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::MemRefType>(unwrap(type)).getPtr()));
}
MlirType mlirCuteMemRefTypeGetLayout(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::MemRefType>(unwrap(type)).getLayout()));
}

bool mlirTypeIsACutePtr(MlirType type) {
  return llvm::isa_and_nonnull<cute::PtrType>(unwrap(type));
}
MlirType mlirCutePtrTypeGet(MlirContext context, MlirType valueType,
                            uint32_t addressSpace, uint64_t alignment,
                            MlirAttribute swizzle, MlirAttribute bitlayout) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  mlir::Type valueTypeValue;
  if (!unwrapAs(ctx, valueType, true, valueTypeValue))
    return {nullptr};
  auto addressSpaceValue =
      ::mlir::cutlass_compiler::cute::symbolizeAddressSpace(addressSpace);
  if (!addressSpaceValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute::SwizzleAttr swizzleValue;
  if (!unwrapAs(ctx, swizzle, true, swizzleValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::BitLayoutAttr bitlayoutValue;
  if (!unwrapAs(ctx, bitlayout, true, bitlayoutValue))
    return {nullptr};
  return wrap(cute::PtrType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valueTypeValue, *addressSpaceValue, alignment, swizzleValue,
      bitlayoutValue));
}
MlirType mlirCutePtrTypeGetValueType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::PtrType>(unwrap(type)).getValueType()));
}
uint32_t mlirCutePtrTypeGetAddressSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute::PtrType>(unwrap(type)).getAddressSpace());
}
uint64_t mlirCutePtrTypeGetAlignment(MlirType type) {
  return llvm::cast<cute::PtrType>(unwrap(type)).getAlignment();
}
MlirAttribute mlirCutePtrTypeGetSwizzle(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::PtrType>(unwrap(type)).getSwizzle()));
}
MlirAttribute mlirCutePtrTypeGetBitlayout(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::PtrType>(unwrap(type)).getBitlayout()));
}

bool mlirTypeIsACuteShape(MlirType type) {
  return llvm::isa_and_nonnull<cute::ShapeType>(unwrap(type));
}
MlirType mlirCuteShapeTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::ShapeType::get(ctx, attrValue));
}
MlirAttribute mlirCuteShapeTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::ShapeType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteSparseElem(MlirType type) {
  return llvm::isa_and_nonnull<cute::SparseElemType>(unwrap(type));
}
MlirType mlirCuteSparseElemTypeGet(MlirContext context, int numLogical,
                                   MlirType physicalType) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  mlir::Type physicalTypeValue;
  if (!unwrapAs(ctx, physicalType, false, physicalTypeValue))
    return {nullptr};
  return wrap(cute::SparseElemType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      numLogical, physicalTypeValue));
}
int mlirCuteSparseElemTypeGetNumLogical(MlirType type) {
  return llvm::cast<cute::SparseElemType>(unwrap(type)).getNumLogical();
}
MlirType mlirCuteSparseElemTypeGetPhysicalType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::SparseElemType>(unwrap(type)).getPhysicalType()));
}

bool mlirTypeIsACuteStride(MlirType type) {
  return llvm::isa_and_nonnull<cute::StrideType>(unwrap(type));
}
MlirType mlirCuteStrideTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::StrideAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::StrideType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteStrideTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::StrideType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteSwizzle(MlirType type) {
  return llvm::isa_and_nonnull<cute::SwizzleType>(unwrap(type));
}
MlirType mlirCuteSwizzleTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::SwizzleAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::SwizzleType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteSwizzleTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::SwizzleType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteTile(MlirType type) {
  return llvm::isa_and_nonnull<cute::TileType>(unwrap(type));
}
MlirType mlirCuteTileTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  ::mlir::cutlass_compiler::cute::TileAttr attrValue;
  if (!unwrapAs(ctx, attr, false, attrValue))
    return {nullptr};
  return wrap(cute::TileType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteTileTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TileType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteTiledCopy(MlirType type) {
  return llvm::isa_and_nonnull<cute::TiledCopyType>(unwrap(type));
}
MlirType mlirCuteTiledCopyTypeGet(MlirContext context, MlirType copyAtom,
                                  MlirAttribute layoutCopyTv,
                                  MlirAttribute tilerMn) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  mlir::Type copyAtomValue;
  if (!unwrapAs(ctx, copyAtom, false, copyAtomValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutCopyTvValue;
  if (!unwrapAs(ctx, layoutCopyTv, false, layoutCopyTvValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::TileAttr tilerMnValue;
  if (!unwrapAs(ctx, tilerMn, false, tilerMnValue))
    return {nullptr};
  return wrap(cute::TiledCopyType::get(ctx, copyAtomValue, layoutCopyTvValue,
                                       tilerMnValue));
}
MlirType mlirCuteTiledCopyTypeGetCopyAtom(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::TiledCopyType>(unwrap(type)).getCopyAtom()));
}
MlirType mlirCuteTiledCopyTypeGetAtom(MlirType type) {
  return mlirCuteTiledCopyTypeGetCopyAtom(type);
}
MlirAttribute mlirCuteTiledCopyTypeGetLayoutCopyTv(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TiledCopyType>(unwrap(type)).getLayoutCopyTv()));
}
MlirAttribute mlirCuteTiledCopyTypeGetTilerMn(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TiledCopyType>(unwrap(type)).getTilerMn()));
}

bool mlirTypeIsACuteTiledCopyV2(MlirType type) {
  return llvm::isa_and_nonnull<cute::TiledCopyV2Type>(unwrap(type));
}
MlirType mlirCuteTiledCopyV2TypeGet(MlirContext context, MlirType copyAtom,
                                    MlirAttribute layoutCopyTv,
                                    MlirAttribute tilerMn) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  mlir::Type copyAtomValue;
  if (!unwrapAs(ctx, copyAtom, false, copyAtomValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutCopyTvValue;
  if (!unwrapAs(ctx, layoutCopyTv, false, layoutCopyTvValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::TileAttr tilerMnValue;
  if (!unwrapAs(ctx, tilerMn, false, tilerMnValue))
    return {nullptr};
  return wrap(cute::TiledCopyV2Type::get(ctx, copyAtomValue, layoutCopyTvValue,
                                         tilerMnValue));
}
MlirType mlirCuteTiledCopyV2TypeGetCopyAtom(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::TiledCopyV2Type>(unwrap(type)).getCopyAtom()));
}
MlirAttribute mlirCuteTiledCopyV2TypeGetLayoutCopyTv(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TiledCopyV2Type>(unwrap(type)).getLayoutCopyTv()));
}
MlirAttribute mlirCuteTiledCopyV2TypeGetTilerMn(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TiledCopyV2Type>(unwrap(type)).getTilerMn()));
}

bool mlirTypeIsACuteTiledMma(MlirType type) {
  return llvm::isa_and_nonnull<cute::TiledMmaType>(unwrap(type));
}
MlirType mlirCuteTiledMmaTypeGet(MlirContext context, MlirType mmaAtom,
                                 MlirAttribute atomLayoutMNK,
                                 MlirAttribute permutationMNK) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  mlir::Type mmaAtomValue;
  if (!unwrapAs(ctx, mmaAtom, false, mmaAtomValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr atomLayoutMNKValue;
  if (!unwrapAs(ctx, atomLayoutMNK, false, atomLayoutMNKValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::TileAttr permutationMNKValue;
  if (!unwrapAs(ctx, permutationMNK, true, permutationMNKValue))
    return {nullptr};
  return wrap(cute::TiledMmaType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      mmaAtomValue, atomLayoutMNKValue, permutationMNKValue));
}
MlirType mlirCuteTiledMmaTypeGetMmaAtom(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::TiledMmaType>(unwrap(type)).getMmaAtom()));
}
MlirType mlirCuteTiledMmaTypeGetAtom(MlirType type) {
  return mlirCuteTiledMmaTypeGetMmaAtom(type);
}
MlirAttribute mlirCuteTiledMmaTypeGetAtomLayoutMNK(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TiledMmaType>(unwrap(type)).getAtomLayout_MNK()));
}
MlirAttribute mlirCuteTiledMmaTypeGetAtomLayoutMnk(MlirType type) {
  return mlirCuteTiledMmaTypeGetAtomLayoutMNK(type);
}
MlirAttribute mlirCuteTiledMmaTypeGetPermutationMNK(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TiledMmaType>(unwrap(type)).getPermutation_MNK()));
}
MlirAttribute mlirCuteTiledMmaTypeGetPermutationMnk(MlirType type) {
  return mlirCuteTiledMmaTypeGetPermutationMNK(type);
}

bool mlirTypeIsACuteTuple(MlirType type) {
  return llvm::isa_and_nonnull<cute::TupleType>(unwrap(type));
}
MlirType mlirCuteTupleTypeGet(MlirContext context, intptr_t numTypes,
                              MlirType const *types) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  llvm::SmallVector<mlir::Type> typesValue;
  for (intptr_t i = 0; i < numTypes; ++i) {
    mlir::Type element;
    if (!unwrapAs(ctx, types[i], false, element))
      return {nullptr};
    typesValue.push_back(element);
  }
  return wrap(cute::TupleType::get(ctx, typesValue));
}
intptr_t mlirCuteTupleTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      llvm::cast<cute::TupleType>(unwrap(type)).getTypes().size());
}
MlirType mlirCuteTupleTypeGetType(MlirType type, intptr_t pos) {
  return wrap(llvm::cast<cute::TupleType>(unwrap(type)).getTypes()[pos]);
}
