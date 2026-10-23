//===- capi.cc - C API for the fly_rocdl dialect --------------------------===//

#include "capi.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl,
                                      mlir::fly_rocdl::FlyROCDLDialect)

static mlir::MLIRContext *loadFlyROCDL(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx)
    ctx->getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>();
  return ctx;
}

/// Null test for unwrapped handles, kept in one place as in fly's C API, where
/// `!attr` is value arithmetic on a fly IntAttr rather than a null test.
template <typename T> static bool isNull(T value) {
  return !static_cast<bool>(value);
}

template <typename Type, typename... Flags>
static MlirType mmaOpGet(MlirContext context, int32_t m, int32_t n, int32_t k,
                         MlirType elemTyA, MlirType elemTyB, MlirType elemTyAcc,
                         Flags... flags) {
  mlir::MLIRContext *ctx = loadFlyROCDL(context);
  mlir::Type a = unwrap(elemTyA), b = unwrap(elemTyB), acc = unwrap(elemTyAcc);
  if (!ctx || isNull(a) || isNull(b) || isNull(acc))
    return {nullptr};
  if (a.getContext() != ctx || b.getContext() != ctx || acc.getContext() != ctx)
    return {nullptr};
  return wrap(
      static_cast<mlir::Type>(Type::get(ctx, m, n, k, a, b, acc, flags...)));
}

#define FLY_ROCDL_TYPE_ISA(Name, Type)                                         \
  bool mlirTypeIsAFlyROCDL##Name(MlirType type) {                              \
    return mlir::isa_and_nonnull<Type>(unwrap(type));                          \
  }

FLY_ROCDL_TYPE_ISA(CopyOpCDNA3BufferCopy,
                   mlir::fly_rocdl::CopyOpCDNA3BufferCopyType)
FLY_ROCDL_TYPE_ISA(CopyOpCDNA3BufferCopyLDS,
                   mlir::fly_rocdl::CopyOpCDNA3BufferCopyLDSType)
FLY_ROCDL_TYPE_ISA(MmaOpCDNA3MFMA, mlir::fly_rocdl::MmaOpCDNA3_MFMAType)
FLY_ROCDL_TYPE_ISA(MmaOpGFX11WMMA, mlir::fly_rocdl::MmaOpGFX11_WMMAType)
FLY_ROCDL_TYPE_ISA(MmaOpGFX120XWMMA, mlir::fly_rocdl::MmaOpGFX120X_WMMAType)

#undef FLY_ROCDL_TYPE_ISA

MlirAttribute mlirFlyROCDLBufferDescAddressAttrGet(MlirContext context) {
  mlir::MLIRContext *ctx = loadFlyROCDL(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Attribute>(
      mlir::fly_rocdl::BufferDescAddressAttr::get(ctx)));
}

MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(MlirContext context,
                                                  int32_t bitSize,
                                                  int32_t cacheModifier) {
  mlir::MLIRContext *ctx = loadFlyROCDL(context);
  if (!ctx)
    return {nullptr};
  return wrap(
      static_cast<mlir::Type>(mlir::fly_rocdl::CopyOpCDNA3BufferCopyType::get(
          ctx, bitSize, cacheModifier)));
}

MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet(MlirContext context,
                                                     int32_t bitSize) {
  mlir::MLIRContext *ctx = loadFlyROCDL(context);
  if (!ctx)
    return {nullptr};
  return wrap(static_cast<mlir::Type>(
      mlir::fly_rocdl::CopyOpCDNA3BufferCopyLDSType::get(ctx, bitSize)));
}

MlirType mlirFlyROCDLMmaOpCDNA3MFMATypeGet(MlirContext context, int32_t m,
                                           int32_t n, int32_t k,
                                           MlirType elemTyA, MlirType elemTyB,
                                           MlirType elemTyAcc) {
  return mmaOpGet<mlir::fly_rocdl::MmaOpCDNA3_MFMAType>(
      context, m, n, k, elemTyA, elemTyB, elemTyAcc);
}

MlirType mlirFlyROCDLMmaOpGFX11WMMATypeGet(MlirContext context, int32_t m,
                                           int32_t n, int32_t k,
                                           MlirType elemTyA, MlirType elemTyB,
                                           MlirType elemTyAcc, bool signA,
                                           bool signB, bool clamp) {
  return mmaOpGet<mlir::fly_rocdl::MmaOpGFX11_WMMAType>(
      context, m, n, k, elemTyA, elemTyB, elemTyAcc, signA, signB, clamp);
}

MlirType mlirFlyROCDLMmaOpGFX120XWMMATypeGet(MlirContext context, int32_t m,
                                             int32_t n, int32_t k,
                                             MlirType elemTyA, MlirType elemTyB,
                                             MlirType elemTyAcc, bool signA,
                                             bool signB, bool clamp) {
  return mmaOpGet<mlir::fly_rocdl::MmaOpGFX120X_WMMAType>(
      context, m, n, k, elemTyA, elemTyB, elemTyAcc, signA, signB, clamp);
}
