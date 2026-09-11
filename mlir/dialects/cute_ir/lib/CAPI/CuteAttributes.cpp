// C++ to C bindings for CuTe algebra attributes.

#include "cute_ir-c/Dialect/CuteAttributes.h"

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

namespace cute = mlir::cutlass_compiler::cute;

namespace {
template <typename Attr>
MlirAttribute getAlgebraAttribute(MlirContext context, MlirStringRef value) {
  auto *ctx = unwrap(context);
  if (!ctx || (!value.data && value.length))
    return {nullptr};
  auto algebra =
      cutegen::from_string<typename Attr::algebra_t>(unwrap(value).str());
  if (!algebra)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(Attr::get(ctx, std::move(*algebra)));
}

template <typename Attr> MlirStringRef getAlgebraValue(MlirAttribute attr) {
  auto value = llvm::cast<Attr>(unwrap(attr));
  // Intern the serialization so the returned view outlives this function.
  auto text = mlir::StringAttr::get(value.getContext(),
                                    cutegen::to_string(value.getRef()));
  return wrap(text.getValue());
}
} // namespace

MlirAttribute mlirCuteIntTupleAttrGet(MlirContext context,
                                      MlirStringRef value) {
  return getAlgebraAttribute<cute::IntTupleAttr>(context, value);
}

bool mlirAttributeIsACuteIntTuple(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::IntTupleAttr>(unwrap(attr));
}

MlirStringRef mlirCuteIntTupleAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::IntTupleAttr>(attr);
}

MlirAttribute mlirCuteCoordAttrGet(MlirContext context, MlirStringRef value) {
  return getAlgebraAttribute<cute::CoordAttr>(context, value);
}

bool mlirAttributeIsACuteCoord(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::CoordAttr>(unwrap(attr));
}

MlirStringRef mlirCuteCoordAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::CoordAttr>(attr);
}

MlirAttribute mlirCuteShapeAttrGet(MlirContext context, MlirStringRef value) {
  return getAlgebraAttribute<cute::ShapeAttr>(context, value);
}

bool mlirAttributeIsACuteShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::ShapeAttr>(unwrap(attr));
}

MlirStringRef mlirCuteShapeAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::ShapeAttr>(attr);
}

MlirAttribute mlirCuteStrideAttrGet(MlirContext context, MlirStringRef value) {
  return getAlgebraAttribute<cute::StrideAttr>(context, value);
}

bool mlirAttributeIsACuteStride(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::StrideAttr>(unwrap(attr));
}

MlirStringRef mlirCuteStrideAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::StrideAttr>(attr);
}

MlirAttribute mlirCuteLayoutAttrGet(MlirContext context, MlirStringRef value) {
  return getAlgebraAttribute<cute::LayoutAttr>(context, value);
}

bool mlirAttributeIsACuteLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::LayoutAttr>(unwrap(attr));
}

MlirStringRef mlirCuteLayoutAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::LayoutAttr>(attr);
}

MlirAttribute mlirCuteTileAttrGet(MlirContext context, MlirStringRef value) {
  return getAlgebraAttribute<cute::TileAttr>(context, value);
}

bool mlirAttributeIsACuteTile(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::TileAttr>(unwrap(attr));
}

MlirStringRef mlirCuteTileAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::TileAttr>(attr);
}

MlirAttribute mlirCuteComposedLayoutAttrGet(MlirContext context,
                                            MlirStringRef value) {
  return getAlgebraAttribute<cute::ComposedLayoutAttr>(context, value);
}

bool mlirAttributeIsACuteComposedLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::ComposedLayoutAttr>(unwrap(attr));
}

MlirStringRef mlirCuteComposedLayoutAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::ComposedLayoutAttr>(attr);
}

MlirAttribute mlirCuteSwizzleAttrGet(MlirContext context, MlirStringRef value) {
  return getAlgebraAttribute<cute::SwizzleAttr>(context, value);
}

bool mlirAttributeIsACuteSwizzle(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute::SwizzleAttr>(unwrap(attr));
}

MlirStringRef mlirCuteSwizzleAttrGetValue(MlirAttribute attr) {
  return getAlgebraValue<cute::SwizzleAttr>(attr);
}
