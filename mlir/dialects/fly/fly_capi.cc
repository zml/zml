#include "fly_capi.h"

#include "TiledOpTraits.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fly, fly, mlir::fly::FlyDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl,
                                      mlir::fly_rocdl::FlyROCDLDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(UB, ub, mlir::ub::UBDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, mlir::gpu::GPUDialect)

using mlir::cast;
using mlir::isa;
using namespace mlir::fly;

namespace {

mlir::MLIRContext *loadFly(MlirContext ctx) {
  mlir::MLIRContext *context = unwrap(ctx);
  if (context) context->getOrLoadDialect<FlyDialect>();
  return context;
}

mlir::MLIRContext *loadFlyROCDL(MlirContext ctx) {
  mlir::MLIRContext *context = unwrap(ctx);
  if (context) context->getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>();
  return context;
}

/// `!attr` and `attr == other` are value arithmetic on a fly IntAttr, not a
/// null test: every handle check here goes through these two helpers.
template <typename T>
bool isNull(T value) {
  return !static_cast<bool>(value);
}

template <typename T, typename Handle>
T checked(Handle handle, mlir::MLIRContext *ctx) {
  auto value = mlir::dyn_cast_if_present<T>(unwrap(handle));
  if (isNull(value) || value.getContext() != ctx) return T{};
  return value;
}

template <typename Type, typename... Flags>
MlirType mmaOpGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc, Flags... flags) {
  mlir::MLIRContext *context = loadFlyROCDL(ctx);
  mlir::Type a = unwrap(elemTyA), b = unwrap(elemTyB), acc = unwrap(elemTyAcc);
  if (!context || isNull(a) || isNull(b) || isNull(acc)) return {nullptr};
  if (a.getContext() != context || b.getContext() != context ||
      acc.getContext() != context)
    return {nullptr};
  return wrap(static_cast<mlir::Type>(
      Type::get(context, m, n, k, a, b, acc, flags...)));
}


}  // namespace

extern "C" {

#define FLY_ATTR_ISA(Name, Attr)                    \
  bool mlirAttributeIsAFly##Name(MlirAttribute a) { \
    return mlir::isa_and_nonnull<Attr>(unwrap(a));  \
  }

FLY_ATTR_ISA(Int, IntAttr)
FLY_ATTR_ISA(IntTuple, IntTupleAttr)
FLY_ATTR_ISA(Layout, LayoutAttr)
FLY_ATTR_ISA(Tile, TileAttr)
FLY_ATTR_ISA(Swizzle, SwizzleAttr)
FLY_ATTR_ISA(Align, AlignAttr)
FLY_ATTR_ISA(AddressSpace, AddressSpaceAttr)

#undef FLY_ATTR_ISA

#define FLY_TYPE_ISA(Name, Type)          \
  bool mlirTypeIsAFly##Name(MlirType t) { \
    return mlir::isa_and_nonnull<Type>(unwrap(t)); \
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
FLY_TYPE_ISA(ROCDLCopyOpCDNA3BufferCopy, mlir::fly_rocdl::CopyOpCDNA3BufferCopyType)
FLY_TYPE_ISA(ROCDLCopyOpCDNA3BufferCopyLDS, mlir::fly_rocdl::CopyOpCDNA3BufferCopyLDSType)
FLY_TYPE_ISA(ROCDLMmaOpCDNA3MFMA, mlir::fly_rocdl::MmaOpCDNA3_MFMAType)
FLY_TYPE_ISA(ROCDLMmaOpGFX11WMMA, mlir::fly_rocdl::MmaOpGFX11_WMMAType)
FLY_TYPE_ISA(ROCDLMmaOpGFX120XWMMA, mlir::fly_rocdl::MmaOpGFX120X_WMMAType)

#undef FLY_TYPE_ISA

MlirAttribute mlirFlyIntAttrGetStatic(MlirContext ctx, int32_t value) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntAttr::getStatic(context, value)));
}

MlirAttribute mlirFlyIntAttrGetDynamic(MlirContext ctx, int32_t width, int32_t divisibility) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntAttr::getDynamic(context, width, divisibility)));
}

MlirAttribute mlirFlyIntAttrGetNone(MlirContext ctx) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntAttr::getNone(context)));
}

MlirAttribute mlirFlyIntTupleAttrGet(MlirContext ctx, MlirAttribute value) {
  mlir::MLIRContext *context = loadFly(ctx);
  mlir::Attribute inner = unwrap(value);
  if (!context || isNull(inner) || inner.getContext() != context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(IntTupleAttr::get(context, inner)));
}

MlirAttribute mlirFlyIntTupleAttrGetBasis(MlirContext ctx, MlirAttribute value, intptr_t nModes, const int32_t *modes) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  IntAttr intValue = checked<IntAttr>(value, context);
  if (isNull(intValue) || (nModes > 0 && !modes)) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      IntTupleAttr::get(intValue, llvm::ArrayRef<int32_t>(modes, nModes))));
}

MlirAttribute mlirFlyIntTupleAttrGetTuple(MlirContext ctx, intptr_t nElements, MlirAttribute const *elements) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context || (nElements > 0 && !elements)) return {nullptr};
  llvm::SmallVector<mlir::Attribute> values;
  values.reserve(nElements);
  for (intptr_t i = 0; i < nElements; ++i) {
    IntTupleAttr element = checked<IntTupleAttr>(elements[i], context);
    if (isNull(element)) return {nullptr};
    values.push_back(element);
  }
  return wrap(static_cast<mlir::Attribute>(
      IntTupleAttr::get(context, mlir::ArrayAttr::get(context, values))));
}

MlirAttribute mlirFlyLayoutAttrGet(MlirContext ctx, MlirAttribute shape, MlirAttribute stride) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  IntTupleAttr shapeAttr = checked<IntTupleAttr>(shape, context);
  IntTupleAttr strideAttr = checked<IntTupleAttr>(stride, context);
  if (isNull(shapeAttr) || isNull(strideAttr)) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(LayoutAttr::get(context, shapeAttr, strideAttr)));
}

MlirAttribute mlirFlyTileAttrGet(MlirContext ctx, MlirAttribute value) {
  mlir::MLIRContext *context = loadFly(ctx);
  mlir::Attribute inner = unwrap(value);
  if (!context || isNull(inner) || inner.getContext() != context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(TileAttr::get(context, inner)));
}

MlirAttribute mlirFlyTileAttrGetModes(MlirContext ctx, intptr_t nModes, MlirAttribute const *modes) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context || (nModes > 0 && !modes)) return {nullptr};
  llvm::SmallVector<mlir::Attribute> values;
  values.reserve(nModes);
  for (intptr_t i = 0; i < nModes; ++i) {
    mlir::Attribute mode = unwrap(modes[i]);
    if (isNull(mode) || mode.getContext() != context) return {nullptr};
    values.push_back(mode);
  }
  return wrap(static_cast<mlir::Attribute>(
      TileAttr::get(context, mlir::ArrayAttr::get(context, values))));
}

MlirAttribute mlirFlySwizzleAttrGet(MlirContext ctx, int32_t mask, int32_t base, int32_t shift) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(SwizzleAttr::get(context, mask, base, shift)));
}

MlirAttribute mlirFlyAlignAttrGet(MlirContext ctx, int32_t alignment) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(AlignAttr::get(context, alignment)));
}

MlirAttribute mlirFlyAddressSpaceAttrGet(MlirContext ctx, int32_t addressSpace) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      AddressSpaceAttr::get(context, static_cast<AddressSpace>(addressSpace))));
}

MlirAttribute mlirFlyMmaOperandAttrGet(MlirContext ctx, int32_t operand) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      MmaOperandAttr::get(context, static_cast<MmaOperand>(operand))));
}

MlirAttribute mlirFlyGemmTraversalOrderAttrGet(MlirContext ctx, int32_t order) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(GemmTraversalOrderAttr::get(
      context, static_cast<GemmTraversalOrder>(order))));
}

MlirType mlirFlyIntTupleTypeGet(MlirContext ctx, MlirAttribute attr) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  IntTupleAttr value = checked<IntTupleAttr>(attr, context);
  if (isNull(value)) return {nullptr};
  return wrap(static_cast<mlir::Type>(IntTupleType::get(value)));
}

MlirAttribute mlirFlyIntTupleTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(cast<IntTupleType>(unwrap(type)).getAttr()));
}

intptr_t mlirFlyIntTupleTypeGetNumElements(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).rank();
}

MlirType mlirFlyIntTupleTypeGetElement(MlirType type, intptr_t pos) {
  return wrap(static_cast<mlir::Type>(cast<IntTupleType>(unwrap(type)).at(pos)));
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

MlirType mlirFlyLayoutTypeGet(MlirContext ctx, MlirAttribute attr) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  LayoutAttr value = checked<LayoutAttr>(attr, context);
  if (isNull(value)) return {nullptr};
  return wrap(static_cast<mlir::Type>(LayoutType::get(value)));
}

MlirAttribute mlirFlyLayoutTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(cast<LayoutType>(unwrap(type)).getAttr()));
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
  return wrap(static_cast<mlir::Attribute>(cast<ComposedLayoutType>(unwrap(type)).getAttr()));
}

MlirType mlirFlyTileTypeGet(MlirContext ctx, MlirAttribute attr) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  TileAttr value = checked<TileAttr>(attr, context);
  if (isNull(value)) return {nullptr};
  return wrap(static_cast<mlir::Type>(TileType::get(value)));
}

MlirAttribute mlirFlyTileTypeGetAttr(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(cast<TileType>(unwrap(type)).getAttr()));
}

MlirType mlirFlySwizzleTypeGet(MlirContext ctx, MlirAttribute attr) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  SwizzleAttr value = checked<SwizzleAttr>(attr, context);
  if (isNull(value)) return {nullptr};
  return wrap(static_cast<mlir::Type>(SwizzleType::get(value)));
}

MlirType mlirFlyPointerTypeGet(MlirContext ctx, MlirType elemTy, MlirAttribute addressSpace, MlirAttribute alignment, MlirAttribute swizzle) {
  mlir::MLIRContext *context = loadFly(ctx);
  mlir::Type element = unwrap(elemTy);
  mlir::Attribute space = unwrap(addressSpace);
  if (!context || isNull(element) || element.getContext() != context) return {nullptr};
  if (isNull(space) || space.getContext() != context) return {nullptr};
  AlignAttr align = alignment.ptr ? checked<AlignAttr>(alignment, context)
                                  : AlignAttr::getTrivialAlignment(element);
  SwizzleAttr sw = swizzle.ptr ? checked<SwizzleAttr>(swizzle, context)
                               : SwizzleAttr::getTrivialSwizzle(context);
  if (isNull(align) || isNull(sw)) return {nullptr};
  return wrap(static_cast<mlir::Type>(
      PointerType::get(context, element, space, align, sw)));
}

MlirType mlirFlyPointerTypeGetElemTy(MlirType type) {
  return wrap(cast<PointerType>(unwrap(type)).getElemTy());
}

MlirAttribute mlirFlyPointerTypeGetAddressSpace(MlirType type) {
  return wrap(cast<PointerType>(unwrap(type)).getAddressSpace());
}

MlirAttribute mlirFlyPointerTypeGetAlignment(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(cast<PointerType>(unwrap(type)).getAlignment()));
}

MlirAttribute mlirFlyPointerTypeGetSwizzle(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(cast<PointerType>(unwrap(type)).getSwizzle()));
}

MlirType mlirFlyMemRefTypeGet(MlirContext ctx, MlirType elemTy, MlirAttribute addressSpace, MlirAttribute layout, MlirAttribute alignment, MlirAttribute swizzle) {
  mlir::MLIRContext *context = loadFly(ctx);
  mlir::Type element = unwrap(elemTy);
  mlir::Attribute space = unwrap(addressSpace);
  mlir::Attribute lay = unwrap(layout);
  if (!context || isNull(element) || element.getContext() != context) return {nullptr};
  if (isNull(space) || space.getContext() != context) return {nullptr};
  if (isNull(lay) || lay.getContext() != context) return {nullptr};
  AlignAttr align = alignment.ptr ? checked<AlignAttr>(alignment, context)
                                  : AlignAttr::getTrivialAlignment(element);
  SwizzleAttr sw = swizzle.ptr ? checked<SwizzleAttr>(swizzle, context)
                               : SwizzleAttr::getTrivialSwizzle(context);
  if (isNull(align) || isNull(sw)) return {nullptr};
  return wrap(static_cast<mlir::Type>(
      mlir::fly::MemRefType::get(context, element, space, lay, align, sw)));
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

MlirType mlirFlyCoordTensorTypeGet(MlirContext ctx, MlirAttribute base, MlirAttribute layout) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  IntTupleAttr baseAttr = checked<IntTupleAttr>(base, context);
  mlir::Attribute lay = unwrap(layout);
  if (isNull(baseAttr) || isNull(lay) || lay.getContext() != context) return {nullptr};
  return wrap(static_cast<mlir::Type>(CoordTensorType::get(context, baseAttr, lay)));
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

MlirType mlirFlyCopyAtomTypeGet(MlirContext ctx, MlirType copyOp, int32_t valBits) {
  mlir::MLIRContext *context = loadFly(ctx);
  mlir::Type op = unwrap(copyOp);
  if (!context || isNull(op) || op.getContext() != context) return {nullptr};
  return wrap(static_cast<mlir::Type>(CopyAtomType::get(context, op, valBits)));
}

MlirType mlirFlyMmaAtomTypeGet(MlirContext ctx, MlirType mmaOp) {
  mlir::MLIRContext *context = loadFly(ctx);
  mlir::Type op = unwrap(mmaOp);
  if (!context || isNull(op) || op.getContext() != context) return {nullptr};
  return wrap(static_cast<mlir::Type>(MmaAtomType::get(context, op)));
}

#define FLY_ATOM_LAYOUT(Name, AtomType, getter)                        \
  MlirType Name(MlirType type) {                                       \
    return wrap(static_cast<mlir::Type>(LayoutType::get(               \
        cast<LayoutAttr>(cast<AtomType>(unwrap(type)).getter()))));    \
  }

FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrLayout, CopyAtomType, getThrLayout)
FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrValLayoutSrc, CopyAtomType, getThrValLayoutSrc)
FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrValLayoutDst, CopyAtomType, getThrValLayoutDst)
FLY_ATOM_LAYOUT(mlirFlyCopyAtomTypeGetThrValLayoutRef, CopyAtomType, getThrValLayoutRef)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrLayout, MmaAtomType, getThrLayout)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrValLayoutA, MmaAtomType, getThrValLayoutA)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrValLayoutB, MmaAtomType, getThrValLayoutB)
FLY_ATOM_LAYOUT(mlirFlyMmaAtomTypeGetThrValLayoutC, MmaAtomType, getThrValLayoutC)

#undef FLY_ATOM_LAYOUT

MlirType mlirFlyMmaAtomTypeGetShapeMNK(MlirType type) {
  return wrap(static_cast<mlir::Type>(IntTupleType::get(
      cast<IntTupleAttr>(cast<MmaAtomType>(unwrap(type)).getShapeMNK()))));
}

MlirType mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc(MlirType type) {
  auto ty = cast<TiledCopyType>(unwrap(type));
  return wrap(static_cast<mlir::Type>(LayoutType::get(tiledCopyGetTiledThrValLayoutSrc(
      cast<CopyAtomType>(ty.getCopyAtom()), ty.getLayoutThrVal().getAttr(),
      ty.getTileMN().getAttr()))));
}

MlirType mlirFlyTiledCopyTypeGetTiledThrValLayoutDst(MlirType type) {
  auto ty = cast<TiledCopyType>(unwrap(type));
  return wrap(static_cast<mlir::Type>(LayoutType::get(tiledCopyGetTiledThrValLayoutDst(
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
  return wrap(static_cast<mlir::Type>(LayoutType::get(tiledMmaGetTiledThrValLayout(
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

MlirType mlirFlyCopyOpUniversalCopyTypeGet(MlirContext ctx, int32_t bitSize) {
  mlir::MLIRContext *context = loadFly(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Type>(CopyOpUniversalCopyType::get(context, bitSize)));
}

MlirAttribute mlirFlyROCDLBufferDescAddressAttrGet(MlirContext ctx) {
  mlir::MLIRContext *context = loadFlyROCDL(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      mlir::fly_rocdl::BufferDescAddressAttr::get(context)));
}

MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(MlirContext ctx, int32_t bitSize, int32_t cacheModifier) {
  mlir::MLIRContext *context = loadFlyROCDL(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Type>(
      mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(context, bitSize, cacheModifier)));
}

MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet(MlirContext ctx, int32_t bitSize) {
  mlir::MLIRContext *context = loadFlyROCDL(ctx);
  if (!context) return {nullptr};
  return wrap(static_cast<mlir::Type>(
      mlir::fly_rocdl::CopyOpCDNA3BufferCopyLDSType::get(context, bitSize)));
}

MlirType mlirFlyROCDLMmaOpCDNA3MFMATypeGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc) {
  return mmaOpGet<mlir::fly_rocdl::MmaOpCDNA3_MFMAType>(ctx, m, n, k, elemTyA, elemTyB, elemTyAcc);
}

MlirType mlirFlyROCDLMmaOpGFX11WMMATypeGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc, bool signA, bool signB, bool clamp) {
  return mmaOpGet<mlir::fly_rocdl::MmaOpGFX11_WMMAType>(ctx, m, n, k, elemTyA, elemTyB, elemTyAcc, signA, signB, clamp);
}

MlirType mlirFlyROCDLMmaOpGFX120XWMMATypeGet(MlirContext ctx, int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc, bool signA, bool signB, bool clamp) {
  return mmaOpGet<mlir::fly_rocdl::MmaOpGFX120X_WMMAType>(ctx, m, n, k, elemTyA, elemTyB, elemTyAcc, signA, signB, clamp);
}

}  // extern "C"
