#include "fly_capi.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "TiledOpTraits.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/TypeSwitch.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fly, fly, mlir::fly::FlyDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl,
                                      mlir::fly_rocdl::FlyROCDLDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(UB, ub, mlir::ub::UBDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, mlir::gpu::GPUDialect)

using mlir::cast;
using mlir::dyn_cast;
using namespace mlir::fly;

extern "C" {

#define ZML_FLY_ATOM_LAYOUT(fn, AtomTy, getter)                             \
  MlirType fn(MlirType t) {                                                 \
    auto atom = dyn_cast<AtomTy>(unwrap(t));                                \
    if (!atom) return {nullptr};                                            \
    return wrap(LayoutType::get(cast<LayoutAttr>(atom.getter())));          \
  }

ZML_FLY_ATOM_LAYOUT(zmlFlyCopyAtomThrLayout, CopyAtomType, getThrLayout)
ZML_FLY_ATOM_LAYOUT(zmlFlyCopyAtomTvLayoutSrc, CopyAtomType, getThrValLayoutSrc)
ZML_FLY_ATOM_LAYOUT(zmlFlyCopyAtomTvLayoutDst, CopyAtomType, getThrValLayoutDst)
ZML_FLY_ATOM_LAYOUT(zmlFlyCopyAtomTvLayoutRef, CopyAtomType, getThrValLayoutRef)
ZML_FLY_ATOM_LAYOUT(zmlFlyMmaAtomThrLayout, MmaAtomType, getThrLayout)
ZML_FLY_ATOM_LAYOUT(zmlFlyMmaAtomTvLayoutA, MmaAtomType, getThrValLayoutA)
ZML_FLY_ATOM_LAYOUT(zmlFlyMmaAtomTvLayoutB, MmaAtomType, getThrValLayoutB)
ZML_FLY_ATOM_LAYOUT(zmlFlyMmaAtomTvLayoutC, MmaAtomType, getThrValLayoutC)

#undef ZML_FLY_ATOM_LAYOUT

MlirType zmlFlyMmaAtomShapeMNK(MlirType t) {
  auto atom = dyn_cast<MmaAtomType>(unwrap(t));
  if (!atom) return {nullptr};
  return wrap(IntTupleType::get(cast<IntTupleAttr>(atom.getShapeMNK())));
}

MlirType zmlFlyTiledCopyTiledTvLayoutSrc(MlirType t) {
  auto ty = dyn_cast<TiledCopyType>(unwrap(t));
  if (!ty) return {nullptr};
  auto atom = cast<CopyAtomType>(ty.getCopyAtom());
  return wrap(LayoutType::get(tiledCopyGetTiledThrValLayoutSrc(
      atom, ty.getLayoutThrVal().getAttr(), ty.getTileMN().getAttr())));
}

MlirType zmlFlyTiledCopyTiledTvLayoutDst(MlirType t) {
  auto ty = dyn_cast<TiledCopyType>(unwrap(t));
  if (!ty) return {nullptr};
  auto atom = cast<CopyAtomType>(ty.getCopyAtom());
  return wrap(LayoutType::get(tiledCopyGetTiledThrValLayoutDst(
      atom, ty.getLayoutThrVal().getAttr(), ty.getTileMN().getAttr())));
}

MlirType zmlFlyTiledMmaTileSizeMNK(MlirType t) {
  auto ty = dyn_cast<TiledMmaType>(unwrap(t));
  if (!ty) return {nullptr};
  auto atom = cast<MmaAtomType>(ty.getMmaAtom());
  return wrap(IntTupleType::get(tiledMmaGetTileSizeMNK(
      atom, ty.getAtomLayout().getAttr(), ty.getPermutation().getAttr())));
}

MlirType zmlFlyTiledMmaThrLayoutVMNK(MlirType t) {
  auto ty = dyn_cast<TiledMmaType>(unwrap(t));
  if (!ty) return {nullptr};
  auto atom = cast<MmaAtomType>(ty.getMmaAtom());
  return wrap(LayoutType::get(
      tiledMmaGetThrLayoutVMNK(atom, ty.getAtomLayout().getAttr())));
}

static MlirType tiledMmaTiledTv(MlirType t, MmaOperand operand) {
  auto ty = dyn_cast<TiledMmaType>(unwrap(t));
  if (!ty) return {nullptr};
  auto atom = cast<MmaAtomType>(ty.getMmaAtom());
  return wrap(LayoutType::get(tiledMmaGetTiledThrValLayout(
      atom, ty.getAtomLayout().getAttr(), ty.getPermutation().getAttr(),
      operand)));
}

MlirType zmlFlyTiledMmaTiledTvLayoutA(MlirType t) {
  return tiledMmaTiledTv(t, MmaOperand::A);
}
MlirType zmlFlyTiledMmaTiledTvLayoutB(MlirType t) {
  return tiledMmaTiledTv(t, MmaOperand::B);
}
MlirType zmlFlyTiledMmaTiledTvLayoutC(MlirType t) {
  return tiledMmaTiledTv(t, MmaOperand::C);
}

}  // extern "C"

extern "C" {

int32_t zmlFlyTypeKind(MlirType t) {
  return llvm::TypeSwitch<mlir::Type, int32_t>(unwrap(t))
      .Case<MemRefType>([](auto) { return 0; })
      .Case<CoordTensorType>([](auto) { return 1; })
      .Case<PointerType>([](auto) { return 2; })
      .Case<IntTupleType>([](auto) { return 3; })
      .Case<LayoutType>([](auto) { return 4; })
      .Case<ComposedLayoutType>([](auto) { return 5; })
      .Case<TileType>([](auto) { return 6; })
      .Case<SwizzleType>([](auto) { return 7; })
      .Case<CopyAtomType>([](auto) { return 8; })
      .Case<MmaAtomType>([](auto) { return 9; })
      .Case<TiledCopyType>([](auto) { return 10; })
      .Case<TiledMmaType>([](auto) { return 11; })
      .Default([](auto) { return -1; });
}

int32_t zmlFlyIntTupleRank(MlirType t) {
  auto ty = dyn_cast<IntTupleType>(unwrap(t));
  return ty ? ty.rank() : -1;
}
bool zmlFlyIntTupleIsLeaf(MlirType t) {
  auto ty = dyn_cast<IntTupleType>(unwrap(t));
  return ty && ty.isLeaf();
}
bool zmlFlyIntTupleIsStatic(MlirType t) {
  auto ty = dyn_cast<IntTupleType>(unwrap(t));
  return ty && ty.isStatic();
}
MlirType zmlFlyIntTupleAt(MlirType t, int32_t i) {
  auto ty = dyn_cast<IntTupleType>(unwrap(t));
  if (!ty || ty.isLeaf() || i < 0 || i >= ty.rank()) return {nullptr};
  return wrap(ty.at(i));
}
int32_t zmlFlyIntTupleLeafKind(MlirType t, int64_t* value) {
  auto ty = dyn_cast<IntTupleType>(unwrap(t));
  if (!ty || !ty.isLeaf()) return -1;
  IntTupleAttr attr = ty.getAttr();
  if (attr.isLeafBasis()) return 3;
  if (attr.isLeafNone()) return 2;
  IntAttr leaf = attr.getLeafAsInt();
  if (!leaf.isStatic()) return 1;
  if (value) *value = leaf.getValue();
  return 0;
}
MlirType zmlFlyLayoutShape(MlirType t) {
  auto ty = dyn_cast<LayoutType>(unwrap(t));
  return ty ? wrap(IntTupleType::get(ty.getAttr().getShape())) : MlirType{nullptr};
}
MlirType zmlFlyLayoutStride(MlirType t) {
  auto ty = dyn_cast<LayoutType>(unwrap(t));
  return ty ? wrap(IntTupleType::get(ty.getAttr().getStride())) : MlirType{nullptr};
}

static mlir::Attribute outerLayoutAttr(mlir::Attribute layout) {
  while (auto composed = dyn_cast<ComposedLayoutAttr>(layout)) layout = composed.getOuter();
  return layout;
}

MlirType zmlFlyLayoutLikeShape(MlirType t) {
  mlir::Type ty = unwrap(t);
  mlir::Attribute layout;
  if (auto l = dyn_cast<LayoutType>(ty)) layout = l.getAttr();
  else if (auto c = dyn_cast<ComposedLayoutType>(ty)) layout = c.getAttr();
  else if (auto m = dyn_cast<mlir::fly::MemRefType>(ty)) layout = m.getLayout();
  else if (auto c = dyn_cast<CoordTensorType>(ty)) layout = c.getLayout();
  else return {nullptr};
  auto plain = dyn_cast<LayoutAttr>(outerLayoutAttr(layout));
  if (!plain) return {nullptr};
  return wrap(IntTupleType::get(plain.getShape()));
}

MlirType zmlFlyMemRefElemType(MlirType t) {
  auto ty = dyn_cast<mlir::fly::MemRefType>(unwrap(t));
  return ty ? wrap(ty.getElemTy()) : MlirType{nullptr};
}
MlirAttribute zmlFlyMemRefAddressSpace(MlirType t) {
  auto ty = dyn_cast<mlir::fly::MemRefType>(unwrap(t));
  return ty ? wrap(ty.getAddressSpace()) : MlirAttribute{nullptr};
}
MlirType zmlFlyPtrElemType(MlirType t) {
  auto ty = dyn_cast<PointerType>(unwrap(t));
  return ty ? wrap(ty.getElemTy()) : MlirType{nullptr};
}
MlirAttribute zmlFlyPtrAddressSpace(MlirType t) {
  auto ty = dyn_cast<PointerType>(unwrap(t));
  return ty ? wrap(ty.getAddressSpace()) : MlirAttribute{nullptr};
}
int32_t zmlFlyPtrAlignment(MlirType t) {
  auto ty = dyn_cast<PointerType>(unwrap(t));
  return ty ? ty.getAlignment().getAlignment() : 0;
}
MlirAttribute zmlFlyPtrSwizzle(MlirType t) {
  auto ty = dyn_cast<PointerType>(unwrap(t));
  return ty ? wrap(ty.getSwizzle()) : MlirAttribute{nullptr};
}
MlirType zmlFlyPtrTypeGet(MlirType elem, MlirAttribute address_space, int32_t alignment, MlirAttribute swizzle) {
  mlir::Type e = unwrap(elem);
  SwizzleAttr sw = swizzle.ptr ? cast<SwizzleAttr>(unwrap(swizzle)) : SwizzleAttr::getTrivialSwizzle(e.getContext());
  return wrap(PointerType::get(e.getContext(), e, unwrap(address_space), AlignAttr::get(e.getContext(), alignment), sw));
}

}  // extern "C"
