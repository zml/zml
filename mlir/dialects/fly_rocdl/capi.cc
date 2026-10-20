#include "capi.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl,
                                      mlir::fly_rocdl::FlyROCDLDialect)

namespace {
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

#define FLY_TYPE_ISA(Name, Type)          \
  bool mlirTypeIsAFly##Name(MlirType t) { \
    return mlir::isa_and_nonnull<Type>(unwrap(t)); \
  }

FLY_TYPE_ISA(ROCDLCopyOpCDNA3BufferCopy, mlir::fly_rocdl::CopyOpCDNA3BufferCopyType)
FLY_TYPE_ISA(ROCDLCopyOpCDNA3BufferCopyLDS, mlir::fly_rocdl::CopyOpCDNA3BufferCopyLDSType)
FLY_TYPE_ISA(ROCDLMmaOpCDNA3MFMA, mlir::fly_rocdl::MmaOpCDNA3_MFMAType)
FLY_TYPE_ISA(ROCDLMmaOpGFX11WMMA, mlir::fly_rocdl::MmaOpGFX11_WMMAType)
FLY_TYPE_ISA(ROCDLMmaOpGFX120XWMMA, mlir::fly_rocdl::MmaOpGFX120X_WMMAType)

#undef FLY_TYPE_ISA

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
