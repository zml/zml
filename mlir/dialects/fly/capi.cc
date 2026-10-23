//===- capi.cc - C API for the fly dialect --------------------------------===//

#include "capi.h"

#include "TiledOpTraits.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fly, fly, mlir::fly::FlyDialect)

using mlir::cast;
using mlir::isa;
using namespace mlir::fly;

static mlir::MLIRContext *loadFly(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx)
    ctx->getOrLoadDialect<FlyDialect>();
  return ctx;
}

/// `!attr` and `attr == other` are value arithmetic on a fly IntAttr, not a
/// null test: every handle check here goes through these two helpers.
template <typename T> static bool isNull(T value) {
  return !static_cast<bool>(value);
}

template <typename T, typename Handle>
static T checked(Handle handle, mlir::MLIRContext *ctx) {
  auto value = mlir::dyn_cast_if_present<T>(unwrap(handle));
  if (isNull(value) || value.getContext() != ctx)
    return T{};
  return value;
}

#define FLY_ATTR_ISA(Name, Attr)                                               \
  bool mlirAttributeIsAFly##Name(MlirAttribute attr) {                         \
    return mlir::isa_and_nonnull<Attr>(unwrap(attr));                          \
  }

FLY_ATTR_ISA(Int, IntAttr)
FLY_ATTR_ISA(IntTuple, IntTupleAttr)
FLY_ATTR_ISA(Layout, LayoutAttr)
FLY_ATTR_ISA(Tile, TileAttr)
FLY_ATTR_ISA(Swizzle, SwizzleAttr)
FLY_ATTR_ISA(Align, AlignAttr)
FLY_ATTR_ISA(AddressSpace, AddressSpaceAttr)

#undef FLY_ATTR_ISA

#define FLY_TYPE_ISA(Name, Type)                                               \
  bool mlirTypeIsAFly##Name(MlirType type) {                                   \
    return mlir::isa_and_nonnull<Type>(unwrap(type));                          \
  }

FLY_TYPE_ISA(IntTuple, IntTupleType)
FLY_TYPE_ISA(Layout, LayoutType)
FLY_TYPE_ISA(ComposedLayout, ComposedLayoutType)
FLY_TYPE_ISA(Tile, TileType)
FLY_TYPE_ISA(Swizzle, SwizzleType)
FLY_TYPE_ISA(Pointer, PointerType)
FLY_TYPE_ISA(MemRef, mlir::fly::MemRefType)
FLY_TYPE_ISA(CoordTensor, CoordTensorType)
FLY_TYPE_ISA(CopyAtom, CopyAtomType)
FLY_TYPE_ISA(MmaAtom, MmaAtomType)
FLY_TYPE_ISA(TiledCopy, TiledCopyType)
FLY_TYPE_ISA(TiledMma, TiledMmaType)
FLY_TYPE_ISA(CopyOpUniversalCopy, CopyOpUniversalCopyType)

#undef FLY_TYPE_ISA

MlirAttribute mlirFlyIntAttrStaticGet(MlirContext context, int32_t value) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntAttr::getStatic(ctx, value)));
}

MlirAttribute mlirFlyIntAttrDynamicGet(MlirContext context, int32_t width,
                                       int32_t divisibility) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      IntAttr::getDynamic(ctx, width, divisibility)));
}

MlirAttribute mlirFlyIntAttrNoneGet(MlirContext context) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntAttr::getNone(ctx)));
}

MlirAttribute mlirFlyIntTupleAttrGet(MlirContext context, MlirAttribute value) {
  mlir::MLIRContext *ctx = loadFly(context);
  mlir::Attribute inner = unwrap(value);
  if (!ctx || isNull(inner) || inner.getContext() != ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntTupleAttr::get(ctx, inner)));
}

MlirAttribute mlirFlyIntTupleAttrBasisGet(MlirContext context,
                                          MlirAttribute value, intptr_t nModes,
                                          const int32_t *modes) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  IntAttr intValue = checked<IntAttr>(value, ctx);
  if (isNull(intValue) || (nModes > 0 && !modes))
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      IntTupleAttr::get(intValue, llvm::ArrayRef<int32_t>(modes, nModes))));
}

MlirAttribute mlirFlyIntTupleAttrTupleGet(MlirContext context,
                                          intptr_t nElements,
                                          MlirAttribute const *elements) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx || (nElements > 0 && !elements))
    return {nullptr};
  llvm::SmallVector<mlir::Attribute> values;
  values.reserve(nElements);
  for (intptr_t i = 0; i < nElements; ++i) {
    IntTupleAttr element = checked<IntTupleAttr>(elements[i], ctx);
    if (isNull(element))
      return {nullptr};
    values.push_back(element);
  }
  return wrap(static_cast<mlir::Attribute>(
      IntTupleAttr::get(ctx, mlir::ArrayAttr::get(ctx, values))));
}

MlirAttribute mlirFlyLayoutAttrGet(MlirContext context, MlirAttribute shape,
                                   MlirAttribute stride) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  IntTupleAttr shapeAttr = checked<IntTupleAttr>(shape, ctx);
  IntTupleAttr strideAttr = checked<IntTupleAttr>(stride, ctx);
  if (isNull(shapeAttr) || isNull(strideAttr))
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      LayoutAttr::get(ctx, shapeAttr, strideAttr)));
}

MlirAttribute mlirFlyTileAttrGet(MlirContext context, MlirAttribute value) {
  mlir::MLIRContext *ctx = loadFly(context);
  mlir::Attribute inner = unwrap(value);
  if (!ctx || isNull(inner) || inner.getContext() != ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(TileAttr::get(ctx, inner)));
}

MlirAttribute mlirFlyTileAttrModesGet(MlirContext context, intptr_t nModes,
                                      MlirAttribute const *modes) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx || (nModes > 0 && !modes))
    return {nullptr};
  llvm::SmallVector<mlir::Attribute> values;
  values.reserve(nModes);
  for (intptr_t i = 0; i < nModes; ++i) {
    mlir::Attribute mode = unwrap(modes[i]);
    if (isNull(mode) || mode.getContext() != ctx)
      return {nullptr};
    values.push_back(mode);
  }
  return wrap(static_cast<mlir::Attribute>(
      TileAttr::get(ctx, mlir::ArrayAttr::get(ctx, values))));
}

MlirAttribute mlirFlySwizzleAttrGet(MlirContext context, int32_t mask,
                                    int32_t base, int32_t shift) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(
      static_cast<mlir::Attribute>(SwizzleAttr::get(ctx, mask, base, shift)));
}

MlirAttribute mlirFlyAlignAttrGet(MlirContext context, int32_t alignment) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(AlignAttr::get(ctx, alignment)));
}

MlirAttribute mlirFlyAddressSpaceAttrGet(MlirContext context,
                                         int32_t addressSpace) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      AddressSpaceAttr::get(ctx, static_cast<AddressSpace>(addressSpace))));
}

MlirAttribute mlirFlyMmaOperandAttrGet(MlirContext context, int32_t operand) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      MmaOperandAttr::get(ctx, static_cast<MmaOperand>(operand))));
}

MlirAttribute mlirFlyGemmTraversalOrderAttrGet(MlirContext context,
                                               int32_t order) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(GemmTraversalOrderAttr::get(
      ctx, static_cast<GemmTraversalOrder>(order))));
}

MlirType mlirFlyIntTupleTypeGet(MlirContext context, MlirAttribute attr) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  IntTupleAttr value = checked<IntTupleAttr>(attr, ctx);
  if (isNull(value))
    return {nullptr};
  return wrap(static_cast<mlir::Type>(IntTupleType::get(value)));
}

MlirAttribute mlirFlyIntTupleTypeGetAttr(MlirType type) {
  return wrap(
      static_cast<mlir::Attribute>(cast<IntTupleType>(unwrap(type)).getAttr()));
}

intptr_t mlirFlyIntTupleTypeGetNumElements(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).rank();
}

MlirType mlirFlyIntTupleTypeGetElement(MlirType type, intptr_t pos) {
  return wrap(
      static_cast<mlir::Type>(cast<IntTupleType>(unwrap(type)).at(pos)));
}

bool mlirFlyIntTupleTypeIsLeaf(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).isLeaf();
}

bool mlirFlyIntTupleTypeIsStatic(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).isStatic();
}

bool mlirFlyIntTupleTypeIsStaticLeaf(MlirType type) {
  IntTupleAttr attr = cast<IntTupleType>(unwrap(type)).getAttr();
  return attr.isLeafInt() && attr.getLeafAsInt().isStatic();
}

bool mlirFlyIntTupleTypeIsNoneLeaf(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).getAttr().isLeafNone();
}

bool mlirFlyIntTupleTypeIsBasisLeaf(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).getAttr().isLeafBasis();
}

int64_t mlirFlyIntTupleTypeGetStaticValue(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).getAttr().getLeafAsInt().getValue();
}

MlirType mlirFlyLayoutTypeGet(MlirContext context, MlirAttribute attr) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  LayoutAttr value = checked<LayoutAttr>(attr, ctx);
  if (isNull(value))
    return {nullptr};
  return wrap(static_cast<mlir::Type>(LayoutType::get(value)));
}

MlirAttribute mlirFlyLayoutTypeGetAttr(MlirType type) {
  return wrap(
      static_cast<mlir::Attribute>(cast<LayoutType>(unwrap(type)).getAttr()));
}

MlirType mlirFlyLayoutTypeGetShape(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      IntTupleType::get(cast<LayoutType>(unwrap(type)).getAttr().getShape())));
}

MlirType mlirFlyLayoutTypeGetStride(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      IntTupleType::get(cast<LayoutType>(unwrap(type)).getAttr().getStride())));
}

MlirAttribute mlirFlyComposedLayoutTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      cast<ComposedLayoutType>(unwrap(type)).getAttr()));
}

MlirType mlirFlyTileTypeGet(MlirContext context, MlirAttribute attr) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  TileAttr value = checked<TileAttr>(attr, ctx);
  if (isNull(value))
    return {nullptr};
  return wrap(static_cast<mlir::Type>(TileType::get(value)));
}

MlirAttribute mlirFlyTileTypeGetAttr(MlirType type) {
  return wrap(
      static_cast<mlir::Attribute>(cast<TileType>(unwrap(type)).getAttr()));
}

MlirType mlirFlySwizzleTypeGet(MlirContext context, MlirAttribute attr) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  SwizzleAttr value = checked<SwizzleAttr>(attr, ctx);
  if (isNull(value))
    return {nullptr};
  return wrap(static_cast<mlir::Type>(SwizzleType::get(value)));
}

MlirType mlirFlyPointerTypeGet(MlirContext context, MlirType elemTy,
                               MlirAttribute addressSpace,
                               MlirAttribute alignment, MlirAttribute swizzle) {
  mlir::MLIRContext *ctx = loadFly(context);
  mlir::Type element = unwrap(elemTy);
  mlir::Attribute space = unwrap(addressSpace);
  if (!ctx || isNull(element) || element.getContext() != ctx)
    return {nullptr};
  if (isNull(space) || space.getContext() != ctx)
    return {nullptr};
  AlignAttr align = alignment.ptr ? checked<AlignAttr>(alignment, ctx)
                                  : AlignAttr::getTrivialAlignment(element);
  SwizzleAttr sw = swizzle.ptr ? checked<SwizzleAttr>(swizzle, ctx)
                               : SwizzleAttr::getTrivialSwizzle(ctx);
  if (isNull(align) || isNull(sw))
    return {nullptr};
  return wrap(static_cast<mlir::Type>(
      PointerType::get(ctx, element, space, align, sw)));
}

MlirType mlirFlyPointerTypeGetElemTy(MlirType type) {
  return wrap(cast<PointerType>(unwrap(type)).getElemTy());
}

MlirAttribute mlirFlyPointerTypeGetAddressSpace(MlirType type) {
  return wrap(cast<PointerType>(unwrap(type)).getAddressSpace());
}

MlirAttribute mlirFlyPointerTypeGetAlignment(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      cast<PointerType>(unwrap(type)).getAlignment()));
}

MlirAttribute mlirFlyPointerTypeGetSwizzle(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      cast<PointerType>(unwrap(type)).getSwizzle()));
}

MlirType mlirFlyMemRefTypeGet(MlirContext context, MlirType elemTy,
                              MlirAttribute addressSpace, MlirAttribute layout,
                              MlirAttribute alignment, MlirAttribute swizzle) {
  mlir::MLIRContext *ctx = loadFly(context);
  mlir::Type element = unwrap(elemTy);
  mlir::Attribute space = unwrap(addressSpace);
  mlir::Attribute lay = unwrap(layout);
  if (!ctx || isNull(element) || element.getContext() != ctx)
    return {nullptr};
  if (isNull(space) || space.getContext() != ctx)
    return {nullptr};
  if (isNull(lay) || lay.getContext() != ctx)
    return {nullptr};
  AlignAttr align = alignment.ptr ? checked<AlignAttr>(alignment, ctx)
                                  : AlignAttr::getTrivialAlignment(element);
  SwizzleAttr sw = swizzle.ptr ? checked<SwizzleAttr>(swizzle, ctx)
                               : SwizzleAttr::getTrivialSwizzle(ctx);
  if (isNull(align) || isNull(sw))
    return {nullptr};
  return wrap(static_cast<mlir::Type>(
      mlir::fly::MemRefType::get(ctx, element, space, lay, align, sw)));
}

MlirType mlirFlyMemRefTypeGetElemTy(MlirType type) {
  return wrap(cast<mlir::fly::MemRefType>(unwrap(type)).getElemTy());
}

MlirAttribute mlirFlyMemRefTypeGetAddressSpace(MlirType type) {
  return wrap(cast<mlir::fly::MemRefType>(unwrap(type)).getAddressSpace());
}

MlirAttribute mlirFlyMemRefTypeGetLayout(MlirType type) {
  return wrap(cast<mlir::fly::MemRefType>(unwrap(type)).getLayout());
}

MlirType mlirFlyCoordTensorTypeGet(MlirContext context, MlirAttribute base,
                                   MlirAttribute layout) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  IntTupleAttr baseAttr = checked<IntTupleAttr>(base, ctx);
  mlir::Attribute lay = unwrap(layout);
  if (isNull(baseAttr) || isNull(lay) || lay.getContext() != ctx)
    return {nullptr};
  return wrap(
      static_cast<mlir::Type>(CoordTensorType::get(ctx, baseAttr, lay)));
}

static mlir::Attribute outerLayoutAttr(mlir::Attribute layout) {
  while (auto composed = mlir::dyn_cast<ComposedLayoutAttr>(layout))
    layout = composed.getOuter();
  return layout;
}

MlirType mlirFlyLayoutLikeTypeGetShape(MlirType type) {
  mlir::Type ty = unwrap(type);
  mlir::Attribute layout;
  if (auto l = mlir::dyn_cast<LayoutType>(ty)) {
    layout = l.getAttr();
  } else if (auto c = mlir::dyn_cast<ComposedLayoutType>(ty)) {
    layout = c.getAttr();
  } else if (auto m = mlir::dyn_cast<mlir::fly::MemRefType>(ty)) {
    layout = m.getLayout();
  } else {
    layout = cast<CoordTensorType>(ty).getLayout();
  }
  return wrap(static_cast<mlir::Type>(
      IntTupleType::get(cast<LayoutAttr>(outerLayoutAttr(layout)).getShape())));
}

MlirType mlirFlyCopyAtomTypeGet(MlirContext context, MlirType copyOp,
                                int32_t valBits) {
  mlir::MLIRContext *ctx = loadFly(context);
  mlir::Type op = unwrap(copyOp);
  if (!ctx || isNull(op) || op.getContext() != ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Type>(CopyAtomType::get(ctx, op, valBits)));
}

MlirType mlirFlyMmaAtomTypeGet(MlirContext context, MlirType mmaOp) {
  mlir::MLIRContext *ctx = loadFly(context);
  mlir::Type op = unwrap(mmaOp);
  if (!ctx || isNull(op) || op.getContext() != ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Type>(MmaAtomType::get(ctx, op)));
}

#define FLY_ATOM_LAYOUT(Name, AtomType, getter)                                \
  MlirType Name(MlirType type) {                                               \
    return wrap(static_cast<mlir::Type>(LayoutType::get(                       \
        cast<LayoutAttr>(cast<AtomType>(unwrap(type)).getter()))));            \
  }

FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrLayout, CopyAtomType, getThrLayout)
FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrValLayoutSrc, CopyAtomType,
                getThrValLayoutSrc)
FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrValLayoutDst, CopyAtomType,
                getThrValLayoutDst)
FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrValLayoutRef, CopyAtomType,
                getThrValLayoutRef)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrLayout, MmaAtomType, getThrLayout)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrValLayoutA, MmaAtomType,
                getThrValLayoutA)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrValLayoutB, MmaAtomType,
                getThrValLayoutB)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrValLayoutC, MmaAtomType,
                getThrValLayoutC)

#undef FLY_ATOM_LAYOUT

MlirType mlirFlyMmaAtomTypeGetShapeMNK(MlirType type) {
  return wrap(static_cast<mlir::Type>(IntTupleType::get(
      cast<IntTupleAttr>(cast<MmaAtomType>(unwrap(type)).getShapeMNK()))));
}

MlirType mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc(MlirType type) {
  auto ty = cast<TiledCopyType>(unwrap(type));
  return wrap(
      static_cast<mlir::Type>(LayoutType::get(tiledCopyGetTiledThrValLayoutSrc(
          cast<CopyAtomType>(ty.getCopyAtom()), ty.getLayoutThrVal().getAttr(),
          ty.getTileMN().getAttr()))));
}

MlirType mlirFlyTiledCopyTypeGetTiledThrValLayoutDst(MlirType type) {
  auto ty = cast<TiledCopyType>(unwrap(type));
  return wrap(
      static_cast<mlir::Type>(LayoutType::get(tiledCopyGetTiledThrValLayoutDst(
          cast<CopyAtomType>(ty.getCopyAtom()), ty.getLayoutThrVal().getAttr(),
          ty.getTileMN().getAttr()))));
}

MlirType mlirFlyTiledMmaTypeGetTileSizeMNK(MlirType type) {
  auto ty = cast<TiledMmaType>(unwrap(type));
  return wrap(static_cast<mlir::Type>(IntTupleType::get(tiledMmaGetTileSizeMNK(
      cast<MmaAtomType>(ty.getMmaAtom()), ty.getAtomLayout().getAttr(),
      ty.getPermutation().getAttr()))));
}

MlirType mlirFlyTiledMmaTypeGetThrLayoutVMNK(MlirType type) {
  auto ty = cast<TiledMmaType>(unwrap(type));
  return wrap(static_cast<mlir::Type>(LayoutType::get(tiledMmaGetThrLayoutVMNK(
      cast<MmaAtomType>(ty.getMmaAtom()), ty.getAtomLayout().getAttr()))));
}

static MlirType tiledMmaTiledThrVal(MlirType tiledMma, MmaOperand operand) {
  auto ty = cast<TiledMmaType>(unwrap(tiledMma));
  return wrap(
      static_cast<mlir::Type>(LayoutType::get(tiledMmaGetTiledThrValLayout(
          cast<MmaAtomType>(ty.getMmaAtom()), ty.getAtomLayout().getAttr(),
          ty.getPermutation().getAttr(), operand))));
}

MlirType mlirFlyTiledMmaTypeGetTiledThrValLayoutA(MlirType type) {
  return tiledMmaTiledThrVal(type, MmaOperand::A);
}

MlirType mlirFlyTiledMmaTypeGetTiledThrValLayoutB(MlirType type) {
  return tiledMmaTiledThrVal(type, MmaOperand::B);
}

MlirType mlirFlyTiledMmaTypeGetTiledThrValLayoutC(MlirType type) {
  return tiledMmaTiledThrVal(type, MmaOperand::C);
}

MlirType mlirFlyCopyOpUniversalCopyTypeGet(MlirContext context,
                                           int32_t bitSize) {
  mlir::MLIRContext *ctx = loadFly(context);
  if (!ctx)
    return {nullptr};
  return wrap(
      static_cast<mlir::Type>(CopyOpUniversalCopyType::get(ctx, bitSize)));
}
