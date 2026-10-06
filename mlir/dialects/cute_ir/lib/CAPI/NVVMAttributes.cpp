// C API for the nvvm attributes, generated from the dialect's .td
// files.
#include "cute_ir-c/Dialect/NVVMAttributes.h"

#include "cute_ir/Dialect/NVVM/IR/NVVMDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

namespace nvvm = mlir::cutlass_compiler::nvvm;

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

bool mlirAttributeIsACuteNVVMAtomicOpKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::AtomicOpKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMAtomicOpKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeAtomicOpKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::AtomicOpKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMAtomicOpKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::AtomicOpKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMBarrierReduction(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::BarrierReductionAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMBarrierReductionAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeBarrierReduction(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::BarrierReductionAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMBarrierReductionAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::BarrierReductionAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMBarrierReduxKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::BarrierReduxKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMBarrierReduxKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeBarrierReduxKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::BarrierReduxKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMBarrierReduxKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::BarrierReduxKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMBlockScaleFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::BlockScaleFormatAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMBlockScaleFormatAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeBlockScaleFormat(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::BlockScaleFormatAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMBlockScaleFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::BlockScaleFormatAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMCTAGroupKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::CTAGroupKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMCTAGroupKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeCTAGroupKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::CTAGroupKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMCTAGroupKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::CTAGroupKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMCVTPackFloatKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::CVTPackFloatKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMCVTPackFloatKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeCVTPackFloatKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::CVTPackFloatKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMCVTPackFloatKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::CVTPackFloatKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMCacheEvictionPriority(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::CacheEvictionPriorityAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMCacheEvictionPriorityAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeCacheEvictionPriority(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::CacheEvictionPriorityAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMCacheEvictionPriorityAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::CacheEvictionPriorityAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMClusterLaunchControlQueryType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ClusterLaunchControlQueryTypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMClusterLaunchControlQueryTypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeClusterLaunchControlQueryType(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ClusterLaunchControlQueryTypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMClusterLaunchControlQueryTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ClusterLaunchControlQueryTypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMCompareOpKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::CompareOpKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMCompareOpKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeCompareOpKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::CompareOpKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMCompareOpKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::CompareOpKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMConvertFP4Type(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ConvertFP4TypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMConvertFP4TypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeConvertFP4Type(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ConvertFP4TypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMConvertFP4TypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ConvertFP4TypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMConvertFP8Type(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ConvertFP8TypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMConvertFP8TypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeConvertFP8Type(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ConvertFP8TypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMConvertFP8TypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ConvertFP8TypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMConvertScaleKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ConvertScaleKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMConvertScaleKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeConvertScaleKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ConvertScaleKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMConvertScaleKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ConvertScaleKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMDotAccumulateType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::DotAccumulateTypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMDotAccumulateTypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeDotAccumulateType(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::DotAccumulateTypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMDotAccumulateTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::DotAccumulateTypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMEvictKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::EvictKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMEvictKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeEvictKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::EvictKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMEvictKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::EvictKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMFPRoundingMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::FPRoundingModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMFPRoundingModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeFPRoundingMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::FPRoundingModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMFPRoundingModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::FPRoundingModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMGridDepActionKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::GridDepActionKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMGridDepActionKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeGridDepActionKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::GridDepActionKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMGridDepActionKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::GridDepActionKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMIntegerRoundingMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::IntegerRoundingModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMIntegerRoundingModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeIntegerRoundingMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::IntegerRoundingModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMIntegerRoundingModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::IntegerRoundingModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVML2PrefetchSize(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::L2PrefetchSizeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVML2PrefetchSizeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeL2PrefetchSize(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::L2PrefetchSizeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVML2PrefetchSizeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::L2PrefetchSizeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMLdStMatrixEltType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::LdStMatrixEltTypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMLdStMatrixEltTypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeLdStMatrixEltType(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::LdStMatrixEltTypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMLdStMatrixEltTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::LdStMatrixEltTypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMLoadCacheModifierExtKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::LoadCacheModifierExtKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMLoadCacheModifierExtKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeLoadCacheModifierExtKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::LoadCacheModifierExtKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMLoadCacheModifierExtKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::LoadCacheModifierExtKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMLoadCacheModifierKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::LoadCacheModifierKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMLoadCacheModifierKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeLoadCacheModifierKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::LoadCacheModifierKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMLoadCacheModifierKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::LoadCacheModifierKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMLoadShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::LoadShapeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMLoadShapeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeLoadShape(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::LoadShapeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMLoadShapeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::LoadShapeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMLoadSrcFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::LoadSrcFormatAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMLoadSrcFormatAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeLoadSrcFormat(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::LoadSrcFormatAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMLoadSrcFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::LoadSrcFormatAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMBarrierLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MBarrierLayoutAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMBarrierLayoutAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMBarrierLayout(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MBarrierLayoutAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMBarrierLayoutAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MBarrierLayoutAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMBarrierPhase(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MBarrierPhaseAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMBarrierPhaseAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMBarrierPhase(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MBarrierPhaseAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMBarrierPhaseAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MBarrierPhaseAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMBarrierScopeKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MBarrierScopeKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMBarrierScopeKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMBarrierScopeKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MBarrierScopeKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMBarrierScopeKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MBarrierScopeKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMBarrierSpaceKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MBarrierSpaceKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMBarrierSpaceKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMBarrierSpaceKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MBarrierSpaceKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMBarrierSpaceKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MBarrierSpaceKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMBarrierTxnKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MBarrierTxnKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMBarrierTxnKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMBarrierTxnKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MBarrierTxnKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMBarrierTxnKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MBarrierTxnKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMBarrierWaitKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MBarrierWaitKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMBarrierWaitKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMBarrierWaitKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MBarrierWaitKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMBarrierWaitKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MBarrierWaitKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMAB1Op(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMAB1OpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMAB1OpAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMAB1Op(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMAB1OpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMAB1OpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMAB1OpAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMABlockScaleKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMABlockScaleKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMABlockScaleKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMABlockScaleKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMABlockScaleKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMABlockScaleKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMABlockScaleKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMACtaCount(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMACtaCountAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMACtaCountAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMACtaCount(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMACtaCountAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMACtaCountAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMACtaCountAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMAFrag(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMAFragAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMAFragAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMAFrag(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMAFragAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMAFragAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMAFragAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMAIntOverflow(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMAIntOverflowAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMAIntOverflowAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMAIntOverflow(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMAIntOverflowAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMAIntOverflowAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMAIntOverflowAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMAKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMAKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMAKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMAKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMAKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMAKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMAKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMALayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMALayoutAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMALayoutAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMALayout(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMALayoutAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMALayoutAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMALayoutAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMMATypes(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMATypesAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMATypesAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMMATypes(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MMATypesAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMMATypesAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MMATypesAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMatchSyncKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MatchSyncKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMatchSyncKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMatchSyncKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MatchSyncKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMatchSyncKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MatchSyncKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMemOrderKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MemOrderKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMemOrderKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMemOrderKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MemOrderKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMemOrderKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MemOrderKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMemScopeKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MemScopeKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMemScopeKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMemScopeKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MemScopeKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMemScopeKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MemScopeKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMMulMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MulModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMulModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeMulMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::MulModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMMulModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::MulModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMNVVMMemorySpace(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::NVVMMemorySpaceAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMNVVMMemorySpaceAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeNVVMMemorySpace(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::NVVMMemorySpaceAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMNVVMMemorySpaceAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::NVVMMemorySpaceAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMPermuteMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::PermuteModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMPermuteModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizePermuteMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::PermuteModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMPermuteModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::PermuteModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMPrefetchCacheLevel(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::PrefetchCacheLevelAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMPrefetchCacheLevelAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizePrefetchCacheLevel(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::PrefetchCacheLevelAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMPrefetchCacheLevelAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::PrefetchCacheLevelAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMProxyKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ProxyKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMProxyKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeProxyKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ProxyKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMProxyKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ProxyKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMReductionKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ReductionKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMReductionKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeReductionKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ReductionKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMReductionKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ReductionKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMReductionOp(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ReductionOpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMReductionOpAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeReductionOp(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ReductionOpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMReductionOpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ReductionOpAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMReductionType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ReductionTypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMReductionTypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeReductionType(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ReductionTypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMReductionTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ReductionTypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPCompressElemSize(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPCompressElemSizeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPCompressElemSizeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPCompressElemSize(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPCompressElemSizeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPCompressElemSizeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPCompressElemSizeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPCompressFactorType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPCompressFactorTypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPCompressFactorTypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPCompressFactorType(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPCompressFactorTypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPCompressFactorTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPCompressFactorTypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPCompressIndexSize(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPCompressIndexSizeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPCompressIndexSizeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPCompressIndexSize(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPCompressIndexSizeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPCompressIndexSizeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPCompressIndexSizeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPCompressOpKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPCompressOpKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPCompressOpKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPCompressOpKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPCompressOpKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPCompressOpKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPCompressOpKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPCompressRepFactor(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPCompressRepFactorAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPCompressRepFactorAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPCompressRepFactor(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPCompressRepFactorAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPCompressRepFactorAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPCompressRepFactorAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPDecompressElemSize(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPDecompressElemSizeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPDecompressElemSizeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPDecompressElemSize(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPDecompressElemSizeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPDecompressElemSizeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPDecompressElemSizeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPDecompressFactorType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPDecompressFactorTypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPDecompressFactorTypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPDecompressFactorType(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPDecompressFactorTypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPDecompressFactorTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPDecompressFactorTypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPDecompressIndexSize(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPDecompressIndexSizeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPDecompressIndexSizeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPDecompressIndexSize(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPDecompressIndexSizeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPDecompressIndexSizeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPDecompressIndexSizeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSPDecompressRepFactor(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SPDecompressRepFactorAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSPDecompressRepFactorAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSPDecompressRepFactor(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SPDecompressRepFactorAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSPDecompressRepFactorAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SPDecompressRepFactorAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSaturationMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SaturationModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSaturationModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSaturationMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SaturationModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSaturationModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SaturationModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSaturationModeKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SaturationModeKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSaturationModeKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSaturationModeKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SaturationModeKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSaturationModeKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SaturationModeKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMScaleVecSize(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ScaleVecSizeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMScaleVecSizeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeScaleVecSize(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ScaleVecSizeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMScaleVecSizeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ScaleVecSizeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSetMaxRegisterAction(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SetMaxRegisterActionAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSetMaxRegisterActionAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSetMaxRegisterAction(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SetMaxRegisterActionAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSetMaxRegisterActionAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SetMaxRegisterActionAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSharedSpace(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SharedSpaceAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSharedSpaceAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSharedSpace(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SharedSpaceAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSharedSpaceAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SharedSpaceAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMShflKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ShflKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMShflKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeShflKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ShflKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMShflKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ShflKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMSparsityFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::SparsityFormatAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMSparsityFormatAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeSparsityFormat(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::SparsityFormatAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMSparsityFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::SparsityFormatAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMStateSpace(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::StateSpaceAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMStateSpaceAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeStateSpace(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::StateSpaceAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMStateSpaceAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::StateSpaceAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMStoreCacheModifierKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::StoreCacheModifierKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMStoreCacheModifierKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeStoreCacheModifierKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::StoreCacheModifierKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMStoreCacheModifierKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::StoreCacheModifierKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMStoreShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::StoreShapeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMStoreShapeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeStoreShape(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::StoreShapeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMStoreShapeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::StoreShapeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTCBarParam(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TCBarParamAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTCBarParamAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTCBarParam(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TCBarParamAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTCBarParamAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TCBarParamAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTMALoadMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TMALoadModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTMALoadModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTMALoadMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TMALoadModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTMALoadModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TMALoadModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTMAReduxKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TMAReduxKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTMAReduxKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTMAReduxKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TMAReduxKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTMAReduxKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TMAReduxKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTMAStoreMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TMAStoreModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTMAStoreModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTMAStoreMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TMAStoreModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTMAStoreModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TMAStoreModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05CpMulticast(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05CpMulticastAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05CpMulticastAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05CpMulticast(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05CpMulticastAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05CpMulticastAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05CpMulticastAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05CpShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05CpShapeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05CpShapeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05CpShape(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05CpShapeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05CpShapeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05CpShapeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05CpSrcFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05CpSrcFormatAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05CpSrcFormatAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05CpSrcFormat(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05CpSrcFormatAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05CpSrcFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05CpSrcFormatAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05FenceKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05FenceKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05FenceKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05FenceKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05FenceKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05FenceKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05FenceKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05LdStShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05LdStShapeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05LdStShapeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05LdStShape(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05LdStShapeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05LdStShapeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05LdStShapeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05MMABlockScale(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05MMABlockScaleAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05MMABlockScaleAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05MMABlockScale(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05MMABlockScaleAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05MMABlockScaleAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05MMABlockScaleAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05MMACollectorBBuffer(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05MMACollectorBBufferAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05MMACollectorBBufferAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05MMACollectorBBuffer(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05MMACollectorBBufferAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05MMACollectorBBufferAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05MMACollectorBBufferAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05MMACollectorOp(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05MMACollectorOpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05MMACollectorOpAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05MMACollectorOp(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05MMACollectorOpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05MMACollectorOpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05MMACollectorOpAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05MMAKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05MMAKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05MMAKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05MMAKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05MMAKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05MMAKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05MMAKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTcgen05WaitKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::Tcgen05WaitKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTcgen05WaitKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTcgen05WaitKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::Tcgen05WaitKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTcgen05WaitKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::Tcgen05WaitKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTensormapElemtype(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TensormapElemtypeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTensormapElemtypeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTensormapElemtype(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TensormapElemtypeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTensormapElemtypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TensormapElemtypeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTensormapField(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TensormapFieldAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTensormapFieldAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTensormapField(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TensormapFieldAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTensormapFieldAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TensormapFieldAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTensormapFillMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TensormapFillModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTensormapFillModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTensormapFillMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TensormapFillModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTensormapFillModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TensormapFillModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTensormapInterleaveLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TensormapInterleaveLayoutAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTensormapInterleaveLayoutAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTensormapInterleaveLayout(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TensormapInterleaveLayoutAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTensormapInterleaveLayoutAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TensormapInterleaveLayoutAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTensormapSwizzleAtomicity(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TensormapSwizzleAtomicityAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTensormapSwizzleAtomicityAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTensormapSwizzleAtomicity(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TensormapSwizzleAtomicityAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTensormapSwizzleAtomicityAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TensormapSwizzleAtomicityAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTensormapSwizzleMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TensormapSwizzleModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTensormapSwizzleModeAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTensormapSwizzleMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TensormapSwizzleModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTensormapSwizzleModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TensormapSwizzleModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMTmemLayout(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TmemLayoutAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTmemLayoutAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeTmemLayout(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::TmemLayoutAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMTmemLayoutAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::TmemLayoutAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMValidatePattern(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::ValidatePatternAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMValidatePatternAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeValidatePattern(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::ValidatePatternAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMValidatePatternAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::ValidatePatternAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMVoteSyncKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::VoteSyncKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMVoteSyncKindAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeVoteSyncKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::VoteSyncKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMVoteSyncKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::VoteSyncKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMWGMMAScaleIn(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::WGMMAScaleInAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMWGMMAScaleInAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeWGMMAScaleIn(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::WGMMAScaleInAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMWGMMAScaleInAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::WGMMAScaleInAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMWGMMAScaleOut(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::WGMMAScaleOutAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMWGMMAScaleOutAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeWGMMAScaleOut(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::WGMMAScaleOutAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMWGMMAScaleOutAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::WGMMAScaleOutAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMWGMMATypes(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::WGMMATypesAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMWGMMATypesAttrGet(MlirContext context, uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  auto valueValue = ::mlir::cutlass_compiler::nvvm::symbolizeWGMMATypes(value);
  if (!valueValue)
    return {nullptr};
  return wrap(nvvm::WGMMATypesAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVVMWGMMATypesAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<nvvm::WGMMATypesAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVVMLdStMatrixShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::LdStMatrixShapeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMLdStMatrixShapeAttrGet(MlirContext context, int m, int n) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  return wrap(nvvm::LdStMatrixShapeAttr::get(ctx, m, n));
}
int mlirCuteNVVMLdStMatrixShapeAttrGetM(MlirAttribute attr) {
  return llvm::cast<nvvm::LdStMatrixShapeAttr>(unwrap(attr)).getM();
}
int mlirCuteNVVMLdStMatrixShapeAttrGetN(MlirAttribute attr) {
  return llvm::cast<nvvm::LdStMatrixShapeAttr>(unwrap(attr)).getN();
}

bool mlirAttributeIsACuteNVVMMMAShape(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::MMAShapeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMMMAShapeAttrGet(MlirContext context, int m, int n, int k) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  return wrap(nvvm::MMAShapeAttr::get(ctx, m, n, k));
}
int mlirCuteNVVMMMAShapeAttrGetM(MlirAttribute attr) {
  return llvm::cast<nvvm::MMAShapeAttr>(unwrap(attr)).getM();
}
int mlirCuteNVVMMMAShapeAttrGetN(MlirAttribute attr) {
  return llvm::cast<nvvm::MMAShapeAttr>(unwrap(attr)).getN();
}
int mlirCuteNVVMMMAShapeAttrGetK(MlirAttribute attr) {
  return llvm::cast<nvvm::MMAShapeAttr>(unwrap(attr)).getK();
}

bool mlirAttributeIsACuteNVVMTarget(MlirAttribute attr) {
  return llvm::isa_and_nonnull<nvvm::TargetAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVVMTargetAttrGet(MlirContext context, int O, MlirStringRef triple, MlirStringRef chip, MlirStringRef features, MlirAttribute flags, MlirAttribute link, bool verifyTarget) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<nvvm::NVVMDialect>();
  ::mlir::DictionaryAttr flagsValue;
  if (!unwrapAs(ctx, flags, true, flagsValue))
    return {nullptr};
  ::mlir::ArrayAttr linkValue;
  if (!unwrapAs(ctx, link, true, linkValue))
    return {nullptr};
  return wrap(nvvm::TargetAttr::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx, O, unwrap(triple), unwrap(chip), unwrap(features), flagsValue, linkValue, verifyTarget));
}
int mlirCuteNVVMTargetAttrGetO(MlirAttribute attr) {
  return llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getO();
}
MlirStringRef mlirCuteNVVMTargetAttrGetTriple(MlirAttribute attr) {
  return wrap(llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getTriple());
}
MlirStringRef mlirCuteNVVMTargetAttrGetChip(MlirAttribute attr) {
  return wrap(llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getChip());
}
MlirStringRef mlirCuteNVVMTargetAttrGetFeatures(MlirAttribute attr) {
  return wrap(llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getFeatures());
}
MlirAttribute mlirCuteNVVMTargetAttrGetFlags(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getFlags()));
}
MlirAttribute mlirCuteNVVMTargetAttrGetLink(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getLink()));
}
bool mlirCuteNVVMTargetAttrGetVerifyTarget(MlirAttribute attr) {
  return llvm::cast<nvvm::TargetAttr>(unwrap(attr)).getVerifyTarget();
}
