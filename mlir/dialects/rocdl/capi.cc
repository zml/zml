//===- capi.cc - C API for the rocdl dialect ------------------------------===//

#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Diagnostics.h"

// Introspection

intptr_t mlirROCDLGetOperationCount(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return 0;
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  intptr_t count = 0;
  for (auto op : ctx->getRegisteredOperations())
    if (op.getDialectNamespace() == "rocdl")
      ++count;
  return count;
}

// Attributes

bool mlirAttributeIsAROCDLBufferOOBMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::BufferOOBModeAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLBufferOOBModeAttrGet(MlirContext context,
                                            uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeBufferOOBMode(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::BufferOOBModeAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLBufferOOBModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::BufferOOBModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLBufferOOBModeModuleFlag(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::BufferOOBModeModuleFlagAttr>(
      unwrap(attr));
}

MlirAttribute mlirROCDLBufferOOBModeModuleFlagAttrGet(MlirContext context,
                                                      uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeBufferOOBMode(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::BufferOOBModeModuleFlagAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLBufferOOBModeModuleFlagAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::BufferOOBModeModuleFlagAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsAROCDLGfx12AtomicCachePolicy(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::Gfx12AtomicCachePolicyAttr>(
      unwrap(attr));
}

MlirAttribute mlirROCDLGfx12AtomicCachePolicyAttrGet(MlirContext context,
                                                     uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeGfx12AtomicCachePolicy(
      static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::Gfx12AtomicCachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLGfx12AtomicCachePolicyAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::Gfx12AtomicCachePolicyAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsAROCDLGfx12CachePolicy(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::Gfx12CachePolicyAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLGfx12CachePolicyAttrGet(MlirContext context,
                                               uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeGfx12CachePolicy(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::Gfx12CachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLGfx12CachePolicyAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::Gfx12CachePolicyAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLGfx942CachePolicy(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::Gfx942CachePolicyAttr>(
      unwrap(attr));
}

MlirAttribute mlirROCDLGfx942CachePolicyAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeGfx942CachePolicy(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::Gfx942CachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLGfx942CachePolicyAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::Gfx942CachePolicyAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLMFMANegModifier(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::MFMANegModifierAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLMFMANegModifierAttrGet(MlirContext context,
                                              uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeMFMANegModifier(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::MFMANegModifierAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLMFMANegModifierAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::MFMANegModifierAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLMFMAPermB(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::MFMAPermBAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLMFMAPermBAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeMFMAPermB(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::MFMAPermBAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLMFMAPermBAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::MFMAPermBAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLMatrixFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::MatrixFormatAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLMatrixFormatAttrGet(MlirContext context,
                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeMatrixFormat(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::MatrixFormatAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLMatrixFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::MatrixFormatAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLPreGfx12CachePolicy(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::PreGfx12CachePolicyAttr>(
      unwrap(attr));
}

MlirAttribute mlirROCDLPreGfx12CachePolicyAttrGet(MlirContext context,
                                                  uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizePreGfx12CachePolicy(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::PreGfx12CachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLPreGfx12CachePolicyAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::PreGfx12CachePolicyAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsAROCDLSchedGroupMask(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::SchedGroupMaskAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLSchedGroupMaskAttrGet(MlirContext context,
                                             uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeSchedGroupMask(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::SchedGroupMaskAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLSchedGroupMaskAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::SchedGroupMaskAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLTBufferOOBModeModuleFlag(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::TBufferOOBModeModuleFlagAttr>(
      unwrap(attr));
}

MlirAttribute mlirROCDLTBufferOOBModeModuleFlagAttrGet(MlirContext context,
                                                       uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeBufferOOBMode(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::TBufferOOBModeModuleFlagAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLTBufferOOBModeModuleFlagAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::TBufferOOBModeModuleFlagAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsAROCDLWMMACModifier(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::WMMACModifierAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLWMMACModifierAttrGet(MlirContext context,
                                            uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  if (value > 65535)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeWMMACModifier(static_cast<uint16_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::WMMACModifierAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLWMMACModifierAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::WMMACModifierAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLWMMAMatrixScale(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::WMMAMatrixScaleAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLWMMAMatrixScaleAttrGet(MlirContext context,
                                              uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeWMMAMatrixScale(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::WMMAMatrixScaleAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLWMMAMatrixScaleAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::WMMAMatrixScaleAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAROCDLWMMAMatrixScaleFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::WMMAMatrixScaleFormatAttr>(
      unwrap(attr));
}

MlirAttribute mlirROCDLWMMAMatrixScaleFormatAttrGet(MlirContext context,
                                                    uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  auto symbol =
      mlir::ROCDL::symbolizeWMMAMatrixScaleFormat(static_cast<uint32_t>(value));
  if (!symbol)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::WMMAMatrixScaleFormatAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLWMMAMatrixScaleFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::ROCDL::WMMAMatrixScaleFormatAttr>(unwrap(attr))
          .getValue());
}

// Target

bool mlirAttributeIsAROCDLTarget(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr));
}

MlirAttribute mlirROCDLTargetAttrGet(MlirContext context, int optLevel,
                                     MlirStringRef triple, MlirStringRef chip,
                                     MlirStringRef features, MlirStringRef abi,
                                     MlirAttribute flags, MlirAttribute link) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  for (auto s : {triple, chip, features, abi})
    if (!s.data && s.length)
      return {nullptr};
  auto flagsValue =
      llvm::dyn_cast_if_present<mlir::DictionaryAttr>(unwrap(flags));
  auto linkValue = llvm::dyn_cast_if_present<mlir::ArrayAttr>(unwrap(link));
  if (unwrap(flags) && (!flagsValue || flagsValue.getContext() != ctx))
    return {nullptr};
  if (unwrap(link) && (!linkValue || linkValue.getContext() != ctx))
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  // The upstream custom getChecked builder delegates to unchecked Base::get.
  // Verify first so invalid C API inputs return null instead of asserting.
  if (mlir::failed(mlir::ROCDL::ROCDLTargetAttr::verifyInvariants(
          [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, optLevel,
          unwrap(triple), unwrap(chip), unwrap(features), unwrap(abi),
          flagsValue, linkValue)))
    return {nullptr};
  return wrap(mlir::ROCDL::ROCDLTargetAttr::get(
      ctx, optLevel, unwrap(triple), unwrap(chip), unwrap(features),
      unwrap(abi), flagsValue, linkValue));
}

int mlirROCDLTargetAttrGetOptLevel(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getO();
}

MlirStringRef mlirROCDLTargetAttrGetTriple(MlirAttribute attr) {
  return wrap(
      llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getTriple());
}

MlirStringRef mlirROCDLTargetAttrGetChip(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getChip());
}

MlirStringRef mlirROCDLTargetAttrGetFeatures(MlirAttribute attr) {
  return wrap(
      llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getFeatures());
}

MlirStringRef mlirROCDLTargetAttrGetAbi(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getAbi());
}

MlirAttribute mlirROCDLTargetAttrGetFlags(MlirAttribute attr) {
  return wrap(
      llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getFlags());
}

MlirAttribute mlirROCDLTargetAttrGetLink(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).getLink());
}

bool mlirROCDLTargetAttrHasFlag(MlirAttribute attr, MlirStringRef flag) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr))
      .hasFlag(unwrap(flag));
}

bool mlirROCDLTargetAttrHasWave64(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).hasWave64();
}

bool mlirROCDLTargetAttrHasFastMath(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).hasFastMath();
}

bool mlirROCDLTargetAttrHasDaz(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).hasDaz();
}

bool mlirROCDLTargetAttrHasFiniteOnly(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).hasFiniteOnly();
}

bool mlirROCDLTargetAttrHasUnsafeMath(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr)).hasUnsafeMath();
}

bool mlirROCDLTargetAttrHasCorrectSqrt(MlirAttribute attr) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attr))
      .hasCorrectSqrt();
}
