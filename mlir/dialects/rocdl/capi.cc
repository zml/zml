#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Diagnostics.h"

// Introspection

intptr_t mlirROCDLGetOperationCount(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx) return 0;
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  intptr_t count = 0;
  for (auto op : ctx->getRegisteredOperations())
    if (op.getDialectNamespace() == "rocdl") ++count;
  return count;
}

// Attributes

bool mlirAttributeIsAROCDLBufferOOBModeAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::BufferOOBModeAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLBufferOOBModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeBufferOOBMode(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::BufferOOBModeAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLBufferOOBModeAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::BufferOOBModeAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLBufferOOBModeModuleFlagAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::BufferOOBModeModuleFlagAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLBufferOOBModeModuleFlagAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeBufferOOBMode(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::BufferOOBModeModuleFlagAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLBufferOOBModeModuleFlagAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::BufferOOBModeModuleFlagAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLGfx12AtomicCachePolicyAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::Gfx12AtomicCachePolicyAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLGfx12AtomicCachePolicyAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeGfx12AtomicCachePolicy(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::Gfx12AtomicCachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLGfx12AtomicCachePolicyAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::Gfx12AtomicCachePolicyAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLGfx12CachePolicyAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::Gfx12CachePolicyAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLGfx12CachePolicyAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeGfx12CachePolicy(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::Gfx12CachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLGfx12CachePolicyAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::Gfx12CachePolicyAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLGfx942CachePolicyAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::Gfx942CachePolicyAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLGfx942CachePolicyAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeGfx942CachePolicy(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::Gfx942CachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLGfx942CachePolicyAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::Gfx942CachePolicyAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLMFMANegModifierAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::MFMANegModifierAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLMFMANegModifierAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeMFMANegModifier(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::MFMANegModifierAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLMFMANegModifierAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::MFMANegModifierAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLMFMAPermBAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::MFMAPermBAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLMFMAPermBAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeMFMAPermB(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::MFMAPermBAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLMFMAPermBAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::MFMAPermBAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLMatrixFormatAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::MatrixFormatAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLMatrixFormatAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeMatrixFormat(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::MatrixFormatAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLMatrixFormatAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::MatrixFormatAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLPreGfx12CachePolicyAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::PreGfx12CachePolicyAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLPreGfx12CachePolicyAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizePreGfx12CachePolicy(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::PreGfx12CachePolicyAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLPreGfx12CachePolicyAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::PreGfx12CachePolicyAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLSchedGroupMaskAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::SchedGroupMaskAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLSchedGroupMaskAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeSchedGroupMask(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::SchedGroupMaskAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLSchedGroupMaskAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::SchedGroupMaskAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLTBufferOOBModeModuleFlagAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::TBufferOOBModeModuleFlagAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLTBufferOOBModeModuleFlagAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeBufferOOBMode(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::TBufferOOBModeModuleFlagAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLTBufferOOBModeModuleFlagAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::TBufferOOBModeModuleFlagAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLWMMACModifierAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::WMMACModifierAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLWMMACModifierAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  if (value > 65535) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeWMMACModifier(static_cast<uint16_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::WMMACModifierAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLWMMACModifierAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::WMMACModifierAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLWMMAMatrixScaleAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::WMMAMatrixScaleAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLWMMAMatrixScaleAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeWMMAMatrixScale(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::WMMAMatrixScaleAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLWMMAMatrixScaleAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::WMMAMatrixScaleAttr>(unwrap(attribute)).getValue());
}

bool mlirAttributeIsAROCDLWMMAMatrixScaleFormatAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::WMMAMatrixScaleFormatAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLWMMAMatrixScaleFormatAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  auto symbol = mlir::ROCDL::symbolizeWMMAMatrixScaleFormat(static_cast<uint32_t>(value));
  if (!symbol) return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  return wrap(mlir::ROCDL::WMMAMatrixScaleFormatAttr::get(ctx, *symbol));
}

uint32_t mlirROCDLWMMAMatrixScaleFormatAttrGetValue(MlirAttribute attribute) {
  return static_cast<uint32_t>(llvm::cast<mlir::ROCDL::WMMAMatrixScaleFormatAttr>(unwrap(attribute)).getValue());
}

// Target

bool mlirAttributeIsAROCDLTargetAttr(MlirAttribute attribute) {
  return llvm::isa_and_nonnull<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute));
}

MlirAttribute mlirROCDLTargetAttrGet(
    MlirContext context, int optLevel, MlirStringRef triple, MlirStringRef chip,
    MlirStringRef features, MlirStringRef abi, MlirAttribute flags,
    MlirAttribute link) {
  auto *ctx = unwrap(context);
  if (!ctx) return {nullptr};
  for (auto s : {triple, chip, features, abi})
    if (!s.data && s.length) return {nullptr};
  auto flagsValue = llvm::dyn_cast_if_present<mlir::DictionaryAttr>(unwrap(flags));
  auto linkValue = llvm::dyn_cast_if_present<mlir::ArrayAttr>(unwrap(link));
  if (unwrap(flags) && (!flagsValue || flagsValue.getContext() != ctx))
    return {nullptr};
  if (unwrap(link) && (!linkValue || linkValue.getContext() != ctx))
    return {nullptr};
  ctx->getOrLoadDialect<mlir::ROCDL::ROCDLDialect>();
  // The upstream custom getChecked builder delegates to unchecked Base::get.
  // Verify first so invalid C API inputs return null instead of asserting.
  if (mlir::failed(mlir::ROCDL::ROCDLTargetAttr::verifyInvariants(
          [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); },
          optLevel, unwrap(triple), unwrap(chip), unwrap(features), unwrap(abi),
          flagsValue, linkValue)))
    return {nullptr};
  return wrap(mlir::ROCDL::ROCDLTargetAttr::get(
      ctx, optLevel, unwrap(triple), unwrap(chip), unwrap(features), unwrap(abi),
      flagsValue, linkValue));
}

int mlirROCDLTargetAttrGetOptLevel(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getO();
}

MlirStringRef mlirROCDLTargetAttrGetTriple(MlirAttribute attribute) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getTriple());
}

MlirStringRef mlirROCDLTargetAttrGetChip(MlirAttribute attribute) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getChip());
}

MlirStringRef mlirROCDLTargetAttrGetFeatures(MlirAttribute attribute) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getFeatures());
}

MlirStringRef mlirROCDLTargetAttrGetAbi(MlirAttribute attribute) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getAbi());
}

MlirAttribute mlirROCDLTargetAttrGetFlags(MlirAttribute attribute) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getFlags());
}

MlirAttribute mlirROCDLTargetAttrGetLink(MlirAttribute attribute) {
  return wrap(llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).getLink());
}

bool mlirROCDLTargetAttrHasFlag(MlirAttribute attribute, MlirStringRef flag) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasFlag(unwrap(flag));
}

bool mlirROCDLTargetAttrHasWave64(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasWave64();
}

bool mlirROCDLTargetAttrHasFastMath(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasFastMath();
}

bool mlirROCDLTargetAttrHasDaz(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasDaz();
}

bool mlirROCDLTargetAttrHasFiniteOnly(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasFiniteOnly();
}

bool mlirROCDLTargetAttrHasUnsafeMath(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasUnsafeMath();
}

bool mlirROCDLTargetAttrHasCorrectSqrt(MlirAttribute attribute) {
  return llvm::cast<mlir::ROCDL::ROCDLTargetAttr>(unwrap(attribute)).hasCorrectSqrt();
}
