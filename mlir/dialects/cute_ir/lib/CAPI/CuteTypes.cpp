// C API for the CuTe types, generated from the dialect's .td files.
#include "cute_ir-c/Dialect/CuteTypes.h"

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
                                       int64_t divisibility, unsigned width,
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
unsigned mlirCuteConstrainedIntTypeGetWidth(MlirType type) {
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
MlirType mlirCuteFastDivmodDivisorTypeGet(MlirContext context, unsigned width,
                                          bool isPow2) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute::CuteDialect>();
  return wrap(cute::FastDivmodDivisorType::get(ctx, width, isPow2));
}
unsigned mlirCuteFastDivmodDivisorTypeGetWidth(MlirType type) {
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
