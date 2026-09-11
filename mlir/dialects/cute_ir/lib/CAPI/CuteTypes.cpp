// C++ to C bindings for CuTe types.
#include "cute_ir-c/Dialect/CuteTypes.h"
#include "cute_ir/Dialect/Cute/IR/CuteDialectPrivate.h"
#include "mlir/CAPI/IR.h"

namespace cute = mlir::cutlass_compiler::cute;

bool mlirTypeIsACuteArithTupleIterator(MlirType type) {
  return llvm::isa_and_nonnull<cute::ArithTupleIteratorType>(unwrap(type));
}
MlirType mlirCuteArithTupleIteratorTypeGet(MlirContext context,
                                           MlirType arithTuple) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  auto arithTupleValue =
      llvm::dyn_cast_if_present<cute::IntTupleType>(unwrap(arithTuple));
  if (!arithTupleValue || arithTupleValue.getContext() != ctx)
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
  auto attrValue =
      llvm::dyn_cast_if_present<cute::ComposedLayoutAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute::ComposedLayoutType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteComposedLayoutTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::ComposedLayoutType>(unwrap(type)).getAttr()));
}

bool mlirTypeIsACuteConstrainedInt32(MlirType type) {
  return llvm::isa_and_nonnull<cute::ConstrainedInt32Type>(unwrap(type));
}
MlirType mlirCuteConstrainedInt32TypeGet(MlirContext context,
                                         uint64_t divisibleBy) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(cute::ConstrainedInt32Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      divisibleBy));
}
uint64_t mlirCuteConstrainedInt32TypeGetDivisibleBy(MlirType type) {
  return llvm::cast<cute::ConstrainedInt32Type>(unwrap(type)).getDivisibleBy();
}

bool mlirTypeIsACuteConstrainedInt64(MlirType type) {
  return llvm::isa_and_nonnull<cute::ConstrainedInt64Type>(unwrap(type));
}
MlirType mlirCuteConstrainedInt64TypeGet(MlirContext context,
                                         uint64_t divisibleBy) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(cute::ConstrainedInt64Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      divisibleBy));
}
uint64_t mlirCuteConstrainedInt64TypeGetDivisibleBy(MlirType type) {
  return llvm::cast<cute::ConstrainedInt64Type>(unwrap(type)).getDivisibleBy();
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
  auto arithTupleValue =
      llvm::dyn_cast_if_present<cute::IntTupleType>(unwrap(arithTuple));
  if (!arithTupleValue || arithTupleValue.getContext() != ctx)
    return {nullptr};
  auto layoutValue = unwrap(layout);
  if (!layoutValue || layoutValue.getContext() != ctx)
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
  auto attrValue = llvm::dyn_cast_if_present<cute::CoordAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
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
MlirType mlirCuteFastDivmodDivisorTypeGet(MlirContext context, unsigned width,
                                          bool isPow2) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(cute::FastDivmodDivisorType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx, width,
      isPow2));
}
unsigned mlirCuteFastDivmodDivisorTypeGetWidth(MlirType type) {
  return llvm::cast<cute::FastDivmodDivisorType>(unwrap(type)).getWidth();
}
bool mlirCuteFastDivmodDivisorTypeGetIsPow2(MlirType type) {
  return llvm::cast<cute::FastDivmodDivisorType>(unwrap(type)).getIsPow2();
}

bool mlirTypeIsACuteIntTuple(MlirType type) {
  return llvm::isa_and_nonnull<cute::IntTupleType>(unwrap(type));
}
MlirType mlirCuteIntTupleTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  auto attrValue = llvm::dyn_cast_if_present<cute::IntTupleAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
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
  auto attrValue = llvm::dyn_cast_if_present<cute::LayoutAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
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
  auto ptrValue = llvm::dyn_cast_if_present<cute::PtrType>(unwrap(ptr));
  if (!ptrValue || ptrValue.getContext() != ctx)
    return {nullptr};
  auto layoutValue = unwrap(layout);
  if (!layoutValue || layoutValue.getContext() != ctx)
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
                            MlirAttribute memorySpace, uint64_t alignment,
                            MlirAttribute swizzle) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  auto valueTypeValue = unwrap(valueType);
  if (valueType.ptr && (!valueTypeValue || valueTypeValue.getContext() != ctx))
    return {nullptr};
  auto memorySpaceValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(memorySpace));
  if (!memorySpaceValue || memorySpaceValue.getContext() != ctx)
    return {nullptr};
  auto swizzleValue =
      llvm::dyn_cast_if_present<cute::SwizzleAttr>(unwrap(swizzle));
  if (swizzle.ptr && (!swizzleValue || swizzleValue.getContext() != ctx))
    return {nullptr};
  return wrap(cute::PtrType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valueTypeValue, memorySpaceValue, alignment, swizzleValue));
}
MlirType mlirCutePtrTypeGetValueType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute::PtrType>(unwrap(type)).getValueType()));
}
MlirAttribute mlirCutePtrTypeGetMemorySpace(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::PtrType>(unwrap(type)).getMemorySpace()));
}
uint64_t mlirCutePtrTypeGetAlignment(MlirType type) {
  return llvm::cast<cute::PtrType>(unwrap(type)).getAlignment();
}
MlirAttribute mlirCutePtrTypeGetSwizzle(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::PtrType>(unwrap(type)).getSwizzle()));
}

bool mlirTypeIsACuteShape(MlirType type) {
  return llvm::isa_and_nonnull<cute::ShapeType>(unwrap(type));
}
MlirType mlirCuteShapeTypeGet(MlirContext context, MlirAttribute attr) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  auto attrValue = llvm::dyn_cast_if_present<cute::ShapeAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
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
  auto physicalTypeValue = unwrap(physicalType);
  if (!physicalTypeValue || physicalTypeValue.getContext() != ctx)
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
  auto attrValue = llvm::dyn_cast_if_present<cute::StrideAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
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
  auto attrValue = llvm::dyn_cast_if_present<cute::SwizzleAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
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
  auto attrValue = llvm::dyn_cast_if_present<cute::TileAttr>(unwrap(attr));
  if (!attrValue || attrValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute::TileType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      attrValue));
}
MlirAttribute mlirCuteTileTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute::TileType>(unwrap(type)).getAttr()));
}
