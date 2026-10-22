//===- capi.cc - C API for the cute_nvgpu dialect -------------------------===//

#include "capi.h"

#include "mlir/dialects/cute_nvgpu/CuteNVGPUDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

// Registration

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    CuteNVGPU, cute_nvgpu, mlir::cutlass_compiler::cute_nvgpu::CuteNVGPUDialect)

// Attributes

namespace cute_nvgpu = mlir::cutlass_compiler::cute_nvgpu;

// Unwraps `handle` into `out`: false when it is of another context or not a
// T, or when it is null and not optional.
template <typename T, typename H>
static bool unwrapAs(mlir::MLIRContext *ctx, H handle, bool optional, T &out) {
  auto value = unwrap(handle);
  if (!value)
    return optional;
  if (value.getContext() != ctx)
    return false;
  out = llvm::dyn_cast<T>(value);
  return static_cast<bool>(out);
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyG2S(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldBulkCopyG2SAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGet(MlirContext context,
                                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldBulkCopyG2S(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldBulkCopyG2SAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldBulkCopyG2SAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2G(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldBulkCopyS2GAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGet(MlirContext context,
                                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldBulkCopyS2G(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldBulkCopyS2GAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldBulkCopyS2GAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2S(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldBulkCopyS2SAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGet(MlirContext context,
                                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldBulkCopyS2S(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldBulkCopyS2SAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldBulkCopyS2SAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldDsmemStore(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldDsmemStoreAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGet(MlirContext context,
                                                          uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldDsmemStore(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldDsmemStoreAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldDsmemStoreAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldLoadGlobal(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldLoadGlobalAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGet(MlirContext context,
                                                          uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldLoadGlobal(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldLoadGlobalAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldLoadGlobalAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoad(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExec2DGather4TmaLoadAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGet(MlirContext context,
                                                         uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExec2DGather4TmaLoad(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldNonExec2DGather4TmaLoadAttr::get(
      ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGetValue(
    MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExec2DGather4TmaLoadAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStore(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExec2DScatter4TmaStoreAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGet(MlirContext context,
                                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExec2DScatter4TmaStore(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldNonExec2DScatter4TmaStoreAttr::get(
      ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGetValue(
    MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExec2DScatter4TmaStoreAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoad(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExecIm2ColTmaLoadAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGet(MlirContext context,
                                                      uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExecIm2ColTmaLoad(value);
  if (!valueValue)
    return {nullptr};
  return wrap(
      cute_nvgpu::AtomCopyFieldNonExecIm2ColTmaLoadAttr::get(ctx, *valueValue));
}
uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExecIm2ColTmaLoadAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaStore(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExecIm2ColTmaStoreAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGet(MlirContext context,
                                                       uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExecIm2ColTmaStore(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldNonExecIm2ColTmaStoreAttr::get(
      ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGetValue(
    MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExecIm2ColTmaStoreAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaLoad(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExecTiledTmaLoadAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGet(MlirContext context,
                                                     uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExecTiledTmaLoad(value);
  if (!valueValue)
    return {nullptr};
  return wrap(
      cute_nvgpu::AtomCopyFieldNonExecTiledTmaLoadAttr::get(ctx, *valueValue));
}
uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExecTiledTmaLoadAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaReduce(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExecTiledTmaReduceAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGet(MlirContext context,
                                                       uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExecTiledTmaReduce(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldNonExecTiledTmaReduceAttr::get(
      ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGetValue(
    MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExecTiledTmaReduceAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaStore(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomCopyFieldNonExecTiledTmaStoreAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGet(MlirContext context,
                                                      uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomCopyFieldNonExecTiledTmaStore(value);
  if (!valueValue)
    return {nullptr};
  return wrap(
      cute_nvgpu::AtomCopyFieldNonExecTiledTmaStoreAttr::get(ctx, *valueValue));
}
uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldNonExecTiledTmaStoreAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldStoreGlobal(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldStoreGlobalAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGet(MlirContext context,
                                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldStoreGlobal(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldStoreGlobalAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldStoreGlobalAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldTmaLoad(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldTmaLoadAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGet(MlirContext context,
                                                       uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldTmaLoad(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldTmaLoadAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldTmaLoadAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldTmaReduce(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldTmaReduceAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGet(MlirContext context,
                                                         uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldTmaReduce(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldTmaReduceAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldTmaReduceAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomCopyFieldTmaStore(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomCopyFieldTmaStoreAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGet(MlirContext context,
                                                        uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomCopyFieldTmaStore(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomCopyFieldTmaStoreAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomCopyFieldTmaStoreAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM100(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomMmaFieldSM100Attr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomMmaFieldSM100AttrGet(MlirContext context,
                                                    uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomMmaFieldSM100(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomMmaFieldSM100Attr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomMmaFieldSM100AttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM100Attr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaled(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomMmaFieldSM100BlockScaledAttr>(
      unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGet(MlirContext context,
                                                 uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomMmaFieldSM100BlockScaled(value);
  if (!valueValue)
    return {nullptr};
  return wrap(
      cute_nvgpu::AtomMmaFieldSM100BlockScaledAttr::get(ctx, *valueValue));
}
uint32_t
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM100BlockScaledAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaledSparse(
    MlirAttribute attr) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::AtomMmaFieldSM100BlockScaledSparseAttr>(unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGet(MlirContext context,
                                                       uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomMmaFieldSM100BlockScaledSparse(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomMmaFieldSM100BlockScaledSparseAttr::get(
      ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGetValue(
    MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM100BlockScaledSparseAttr>(
          unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM100Sparse(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomMmaFieldSM100SparseAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGet(MlirContext context,
                                                          uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomMmaFieldSM100Sparse(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomMmaFieldSM100SparseAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM100SparseAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM120BlockScaled(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomMmaFieldSM120BlockScaledAttr>(
      unwrap(attr));
}
MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGet(MlirContext context,
                                                 uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute_nvgpu::
      symbolizeAtomMmaFieldSM120BlockScaled(value);
  if (!valueValue)
    return {nullptr};
  return wrap(
      cute_nvgpu::AtomMmaFieldSM120BlockScaledAttr::get(ctx, *valueValue));
}
uint32_t
mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM120BlockScaledAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM80Sparse(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomMmaFieldSM80SparseAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGet(MlirContext context,
                                                         uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomMmaFieldSM80Sparse(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomMmaFieldSM80SparseAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM80SparseAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUAtomMmaFieldSM90(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::AtomMmaFieldSM90Attr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUAtomMmaFieldSM90AttrGet(MlirContext context,
                                                   uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeAtomMmaFieldSM90(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::AtomMmaFieldSM90Attr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUAtomMmaFieldSM90AttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::AtomMmaFieldSM90Attr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUBinaryOp(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::BinaryOpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUBinaryOpAttrGet(MlirContext context,
                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeBinaryOp(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::BinaryOpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUBinaryOpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::BinaryOpAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUCopyS2TBroadcast(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyS2TBroadcastAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUCopyS2TBroadcastAttrGet(MlirContext context,
                                                   uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeCopyS2TBroadcast(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyS2TBroadcastAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUCopyS2TBroadcastAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyS2TBroadcastAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUGatherScatterTmaLoad(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::GatherScatterTmaLoadAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUGatherScatterTmaLoadAttrGet(MlirContext context,
                                                       uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeGatherScatterTmaLoad(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::GatherScatterTmaLoadAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUGatherScatterTmaLoadAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::GatherScatterTmaLoadAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPUIm2ColTmaLoad(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::Im2ColTmaLoadAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUIm2ColTmaLoadAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeIm2ColTmaLoad(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::Im2ColTmaLoadAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUIm2ColTmaLoadAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::Im2ColTmaLoadAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPULdReduceAccPrecisionKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::LdReduceAccPrecisionKindAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPULdReduceAccPrecisionKindAttrGet(MlirContext context,
                                                           uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeLdReduceAccPrecisionKind(
          value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::LdReduceAccPrecisionKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPULdReduceAccPrecisionKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::LdReduceAccPrecisionKindAttr>(unwrap(attr))
          .getValue());
}

bool mlirAttributeIsACuteNVGPULdsmSzPattern(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::LdsmSzPatternAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPULdsmSzPatternAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeLdsmSzPattern(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::LdsmSzPatternAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPULdsmSzPatternAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::LdsmSzPatternAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPULoadCacheMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::LoadCacheModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPULoadCacheModeAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeLoadCacheMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::LoadCacheModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPULoadCacheModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::LoadCacheModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUMMAIntOverflow(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::MMAIntOverflowAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUMMAIntOverflowAttrGet(MlirContext context,
                                                 uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMMAIntOverflow(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::MMAIntOverflowAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUMMAIntOverflowAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MMAIntOverflowAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUMajorMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::MajorModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUMajorModeAttrGet(MlirContext context,
                                            uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::MajorModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUMajorModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MajorModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUMmaCollectorOp(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaCollectorOpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUMmaCollectorOpAttrGet(MlirContext context,
                                                 uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::MmaCollectorOpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUMmaCollectorOpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaCollectorOpAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUMmaFragKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaFragKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUMmaFragKindAttrGet(MlirContext context,
                                              uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::MmaFragKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUMmaFragKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaFragKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUNotImplementedFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::NotImplementedFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUNotImplementedFrgAttrGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  return wrap(cute_nvgpu::NotImplementedFrgAttr::get(ctx));
}

bool mlirAttributeIsACuteNVGPUReductionKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::ReductionKindAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUReductionKindAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeReductionKind(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::ReductionKindAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUReductionKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::ReductionKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPURmemFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::RmemFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPURmemFrgAttrGet(MlirContext context,
                                          MlirType valueType, uint32_t operand,
                                          bool deriveElemType) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valueTypeValue;
  if (!unwrapAs(ctx, valueType, false, valueTypeValue))
    return {nullptr};
  auto operandValue =
      ::mlir::cutlass_compiler::cute::symbolizeMmaOperand(operand);
  if (!operandValue)
    return {nullptr};
  return wrap(cute_nvgpu::RmemFrgAttr::get(ctx, valueTypeValue, *operandValue,
                                           deriveElemType));
}
MlirType mlirCuteNVGPURmemFrgAttrGetValueType(MlirAttribute attr) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::RmemFrgAttr>(unwrap(attr)).getValueType()));
}
uint32_t mlirCuteNVGPURmemFrgAttrGetOperand(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::RmemFrgAttr>(unwrap(attr)).getOperand());
}
bool mlirCuteNVGPURmemFrgAttrGetDeriveElemType(MlirAttribute attr) {
  return llvm::cast<cute_nvgpu::RmemFrgAttr>(unwrap(attr)).getDeriveElemType();
}

bool mlirAttributeIsACuteNVGPUSM100CircularSmemFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM100CircularSmemFrgAttr>(
      unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM100CircularSmemFrgAttrGet(MlirContext context,
                                                       uint32_t majorMode) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto majorModeValue =
      ::mlir::cutlass_compiler::cute::symbolizeMajorMode(majorMode);
  if (!majorModeValue)
    return {nullptr};
  return wrap(cute_nvgpu::SM100CircularSmemFrgAttr::get(ctx, *majorModeValue));
}
uint32_t mlirCuteNVGPUSM100CircularSmemFrgAttrGetMajorMode(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::SM100CircularSmemFrgAttr>(unwrap(attr))
          .getMajorMode());
}

bool mlirAttributeIsACuteNVGPUSM100SmemFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM100SmemFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM100SmemFrgAttrGet(MlirContext context,
                                               uint32_t majorMode) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto majorModeValue =
      ::mlir::cutlass_compiler::cute::symbolizeMajorMode(majorMode);
  if (!majorModeValue)
    return {nullptr};
  return wrap(cute_nvgpu::SM100SmemFrgAttr::get(ctx, *majorModeValue));
}
uint32_t mlirCuteNVGPUSM100SmemFrgAttrGetMajorMode(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::SM100SmemFrgAttr>(unwrap(attr)).getMajorMode());
}

bool mlirAttributeIsACuteNVGPUSM100TmemEFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM100TmemEFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM100TmemEFrgAttrGet(MlirContext context,
                                                MlirType aType,
                                                MlirType eType) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type eTypeValue;
  if (!unwrapAs(ctx, eType, false, eTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::SM100TmemEFrgAttr::get(ctx, aTypeValue, eTypeValue));
}
MlirType mlirCuteNVGPUSM100TmemEFrgAttrGetAType(MlirAttribute attr) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::SM100TmemEFrgAttr>(unwrap(attr)).getAType()));
}
MlirType mlirCuteNVGPUSM100TmemEFrgAttrGetEType(MlirAttribute attr) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::SM100TmemEFrgAttr>(unwrap(attr)).getEType()));
}

bool mlirAttributeIsACuteNVGPUSM100TmemFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM100TmemFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM100TmemFrgAttrGet(MlirContext context,
                                               MlirType dataType,
                                               MlirType storageType,
                                               int ctaGroup,
                                               uint32_t tmemAllocMode) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type dataTypeValue;
  if (!unwrapAs(ctx, dataType, false, dataTypeValue))
    return {nullptr};
  mlir::Type storageTypeValue;
  if (!unwrapAs(ctx, storageType, false, storageTypeValue))
    return {nullptr};
  auto tmemAllocModeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmemAllocMode(
          tmemAllocMode);
  if (!tmemAllocModeValue)
    return {nullptr};
  return wrap(cute_nvgpu::SM100TmemFrgAttr::get(
      ctx, dataTypeValue, storageTypeValue, ctaGroup, *tmemAllocModeValue));
}
MlirType mlirCuteNVGPUSM100TmemFrgAttrGetDataType(MlirAttribute attr) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::SM100TmemFrgAttr>(unwrap(attr)).getDataType()));
}
MlirType mlirCuteNVGPUSM100TmemFrgAttrGetStorageType(MlirAttribute attr) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::SM100TmemFrgAttr>(unwrap(attr)).getStorageType()));
}
int mlirCuteNVGPUSM100TmemFrgAttrGetCtaGroup(MlirAttribute attr) {
  return llvm::cast<cute_nvgpu::SM100TmemFrgAttr>(unwrap(attr)).getCtaGroup();
}
uint32_t mlirCuteNVGPUSM100TmemFrgAttrGetTmemAllocMode(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::SM100TmemFrgAttr>(unwrap(attr))
          .getTmemAllocMode());
}

bool mlirAttributeIsACuteNVGPUSM100TmemSfFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM100TmemSfFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM100TmemSfFrgAttrGet(MlirContext context,
                                                 MlirType sfType, int sfVecSize,
                                                 int ctaGroup, bool isSfa,
                                                 uint32_t tmemAllocMode) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type sfTypeValue;
  if (!unwrapAs(ctx, sfType, false, sfTypeValue))
    return {nullptr};
  auto tmemAllocModeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmemAllocMode(
          tmemAllocMode);
  if (!tmemAllocModeValue)
    return {nullptr};
  return wrap(cute_nvgpu::SM100TmemSfFrgAttr::get(
      ctx, sfTypeValue, sfVecSize, ctaGroup, isSfa, *tmemAllocModeValue));
}
MlirType mlirCuteNVGPUSM100TmemSfFrgAttrGetSfType(MlirAttribute attr) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::SM100TmemSfFrgAttr>(unwrap(attr)).getSfType()));
}
int mlirCuteNVGPUSM100TmemSfFrgAttrGetSfVecSize(MlirAttribute attr) {
  return llvm::cast<cute_nvgpu::SM100TmemSfFrgAttr>(unwrap(attr))
      .getSfVecSize();
}
int mlirCuteNVGPUSM100TmemSfFrgAttrGetCtaGroup(MlirAttribute attr) {
  return llvm::cast<cute_nvgpu::SM100TmemSfFrgAttr>(unwrap(attr)).getCtaGroup();
}
bool mlirCuteNVGPUSM100TmemSfFrgAttrGetIsSfa(MlirAttribute attr) {
  return llvm::cast<cute_nvgpu::SM100TmemSfFrgAttr>(unwrap(attr)).getIsSfa();
}
uint32_t mlirCuteNVGPUSM100TmemSfFrgAttrGetTmemAllocMode(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::SM100TmemSfFrgAttr>(unwrap(attr))
          .getTmemAllocMode());
}

bool mlirAttributeIsACuteNVGPUSM107SmemFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM107SmemFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM107SmemFrgAttrGet(MlirContext context,
                                               uint32_t majorMode) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto majorModeValue =
      ::mlir::cutlass_compiler::cute::symbolizeMajorMode(majorMode);
  if (!majorModeValue)
    return {nullptr};
  return wrap(cute_nvgpu::SM107SmemFrgAttr::get(ctx, *majorModeValue));
}
uint32_t mlirCuteNVGPUSM107SmemFrgAttrGetMajorMode(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::SM107SmemFrgAttr>(unwrap(attr)).getMajorMode());
}

bool mlirAttributeIsACuteNVGPUSM90SmemFrg(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::SM90SmemFrgAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUSM90SmemFrgAttrGet(MlirContext context,
                                              uint32_t majorMode) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto majorModeValue =
      ::mlir::cutlass_compiler::cute::symbolizeMajorMode(majorMode);
  if (!majorModeValue)
    return {nullptr};
  return wrap(cute_nvgpu::SM90SmemFrgAttr::get(ctx, *majorModeValue));
}
uint32_t mlirCuteNVGPUSM90SmemFrgAttrGetMajorMode(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::SM90SmemFrgAttr>(unwrap(attr)).getMajorMode());
}

bool mlirAttributeIsACuteNVGPUTiledTmaLoad(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::TiledTmaLoadAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUTiledTmaLoadAttrGet(MlirContext context,
                                               uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTiledTmaLoad(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::TiledTmaLoadAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUTiledTmaLoadAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::TiledTmaLoadAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUTmaDataFormat(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmaDataFormatAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUTmaDataFormatAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::TmaDataFormatAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUTmaDataFormatAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::TmaDataFormatAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUTmaLoadMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmaLoadModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUTmaLoadModeAttrGet(MlirContext context,
                                              uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaLoadMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::TmaLoadModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUTmaLoadModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::TmaLoadModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUTmaStoreMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmaStoreModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUTmaStoreModeAttrGet(MlirContext context,
                                               uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaStoreMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::TmaStoreModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUTmaStoreModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::TmaStoreModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUTmemAllocMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmemAllocModeAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUTmemAllocModeAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmemAllocMode(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::TmemAllocModeAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUTmemAllocModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::TmemAllocModeAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsACuteNVGPUTmemLoadRedOp(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmemLoadRedOpAttr>(unwrap(attr));
}
MlirAttribute mlirCuteNVGPUTmemLoadRedOpAttrGet(MlirContext context,
                                                uint32_t value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto valueValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmemLoadRedOp(value);
  if (!valueValue)
    return {nullptr};
  return wrap(cute_nvgpu::TmemLoadRedOpAttr::get(ctx, *valueValue));
}
uint32_t mlirCuteNVGPUTmemLoadRedOpAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::TmemLoadRedOpAttr>(unwrap(attr)).getValue());
}

// Types

bool mlirTypeIsACuteNVGPUType(MlirType type) {
  mlir::Type cppType = unwrap(type);
  if (!cppType)
    return false;
  if (auto opaque = mlir::dyn_cast<mlir::OpaqueType>(cppType))
    return opaque.getDialectNamespace() == "cute_nvgpu";
  return cppType.getDialect().getNamespace() == "cute_nvgpu";
}

bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomBulkCopyG2SType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(MlirContext context,
                                                 MlirType valType, int copyBits,
                                                 bool mcast) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomBulkCopyG2SType::get(ctx, valTypeValue,
                                                       copyBits, mcast));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomBulkCopyG2SType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomBulkCopyG2SType>(unwrap(type))
      .getCopyBits();
}
bool mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetMcast(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomBulkCopyG2SType>(unwrap(type))
      .getMcast();
}

bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomBulkCopyS2GType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(MlirContext context,
                                                 MlirType valType, int copyBits,
                                                 bool mask) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomBulkCopyS2GType::get(ctx, valTypeValue,
                                                       copyBits, mask));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2GType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2GType>(unwrap(type))
      .getCopyBits();
}
bool mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetMask(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2GType>(unwrap(type))
      .getMask();
}

bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomBulkCopyS2SType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(MlirContext context,
                                                 MlirType valType,
                                                 int copyBits) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomBulkCopyS2SType::get(ctx, valTypeValue, copyBits));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2SType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2SType>(unwrap(type))
      .getCopyBits();
}

bool mlirTypeIsACuteNVGPUCopyAtomDsmemStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomDsmemStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(MlirContext context,
                                                MlirType valType,
                                                int copyBits) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomDsmemStoreType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits));
}
MlirType mlirCuteNVGPUCopyAtomDsmemStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomDsmemStoreType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomDsmemStoreTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomDsmemStoreType>(unwrap(type))
      .getCopyBits();
}

bool mlirTypeIsACuteNVGPUCopyAtomG2R(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomG2RType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomG2RTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t memOrder,
    uint32_t memScope, uint32_t l2PrefetchSize, uint32_t l1CacheEvictPriority,
    uint32_t loadCacheMode, uint32_t sharedSpace, bool invariant) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  auto l2PrefetchSizeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeL2PrefetchSize(
          l2PrefetchSize);
  if (!l2PrefetchSizeValue)
    return {nullptr};
  auto l1CacheEvictPriorityValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeCacheEvictionPriority(
          l1CacheEvictPriority);
  if (!l1CacheEvictPriorityValue)
    return {nullptr};
  auto loadCacheModeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeLoadCacheMode(
          loadCacheMode);
  if (!loadCacheModeValue)
    return {nullptr};
  auto sharedSpaceValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSharedSpace(sharedSpace);
  if (!sharedSpaceValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomG2RType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, *memOrderValue, *memScopeValue,
      *l2PrefetchSizeValue, *l1CacheEvictPriorityValue, *loadCacheModeValue,
      *sharedSpaceValue, invariant));
}
MlirType mlirCuteNVGPUCopyAtomG2RTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getValType()));
}
int mlirCuteNVGPUCopyAtomG2RTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomG2RTypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getMemOrder());
}
uint32_t mlirCuteNVGPUCopyAtomG2RTypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getMemScope());
}
uint32_t mlirCuteNVGPUCopyAtomG2RTypeGetL2PrefetchSize(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type))
          .getL2PrefetchSize());
}
uint32_t mlirCuteNVGPUCopyAtomG2RTypeGetL1CacheEvictPriority(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type))
          .getL1CacheEvictPriority());
}
uint32_t mlirCuteNVGPUCopyAtomG2RTypeGetLoadCacheMode(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getLoadCacheMode());
}
uint32_t mlirCuteNVGPUCopyAtomG2RTypeGetSharedSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getSharedSpace());
}
bool mlirCuteNVGPUCopyAtomG2RTypeGetInvariant(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getInvariant();
}

bool mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(MlirContext context,
                                                   MlirType valType,
                                                   int copyBits, int numCta,
                                                   MlirType gStride, bool mcast,
                                                   MlirType tmaGbasis) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::StrideType gStrideValue;
  if (!unwrapAs(ctx, gStride, false, gStrideValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, true, tmaGbasisValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomIm2ColTmaLoadType::get(
      ctx, valTypeValue, copyBits, numCta, gStrideValue, mcast,
      tmaGbasisValue));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
      .getCopyBits();
}
int mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
      .getNumCta();
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetGStride(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
          .getGStride()));
}
bool mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetMcast(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
      .getMcast();
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
          .getTmaGbasis()));
}

bool mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(MlirContext context,
                                                    MlirType valType,
                                                    int copyBits,
                                                    MlirType gStride,
                                                    MlirType tmaGbasis) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::StrideType gStrideValue;
  if (!unwrapAs(ctx, gStride, false, gStrideValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, true, tmaGbasisValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomIm2ColTmaStoreType::get(
      ctx, valTypeValue, copyBits, gStrideValue, tmaGbasisValue));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(unwrap(type))
      .getCopyBits();
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetGStride(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(unwrap(type))
          .getGStride()));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(unwrap(type))
          .getTmaGbasis()));
}

bool mlirTypeIsACuteNVGPUCopyAtomLdsm(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomLdsmType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomLdsmTypeGet(MlirContext context, MlirType valType,
                                          MlirAttribute mode,
                                          uint32_t szPattern, int numMatrices,
                                          bool transpose) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr modeValue;
  if (!unwrapAs(ctx, mode, false, modeValue))
    return {nullptr};
  auto szPatternValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeLdsmSzPattern(szPattern);
  if (!szPatternValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomLdsmType::get(
      ctx, valTypeValue, modeValue, *szPatternValue, numMatrices,
      transpose ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr()));
}
MlirType mlirCuteNVGPUCopyAtomLdsmTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomLdsmType>(unwrap(type)).getValType()));
}
MlirAttribute mlirCuteNVGPUCopyAtomLdsmTypeGetMode(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomLdsmType>(unwrap(type)).getMode()));
}
uint32_t mlirCuteNVGPUCopyAtomLdsmTypeGetSzPattern(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomLdsmType>(unwrap(type)).getSzPattern());
}
int mlirCuteNVGPUCopyAtomLdsmTypeGetNumMatrices(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomLdsmType>(unwrap(type))
      .getNumMatrices();
}
bool mlirCuteNVGPUCopyAtomLdsmTypeGetTranspose(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomLdsmType>(unwrap(type)).getTranspose());
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto kindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeGatherScatterTmaLoad(kind);
  if (!kindValue)
    return {nullptr};
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      *kindValue, valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
uint32_t
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(unwrap(type))
          .getKind());
}
MlirType
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(
             unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaFormat(MlirType type) {
  auto value =
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(unwrap(type))
          .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExec2DScatter4TmaStore(MlirType type) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGet(
    MlirContext context, MlirType valType, int copyBits, MlirType tmaGbasis,
    int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
MlirType
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>(
          unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetCopyBits(
    MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>(
             unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>(
          unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaFormat(MlirType type) {
  auto value = llvm::cast<cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>(
                   unwrap(type))
                   .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto kindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeIm2ColTmaLoad(kind);
  if (!kindValue)
    return {nullptr};
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      *kindValue, valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
uint32_t mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(unwrap(type))
          .getKind());
}
MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaFormat(MlirType type) {
  auto value =
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(unwrap(type))
          .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(MlirContext context,
                                                           MlirType valType,
                                                           int copyBits,
                                                           MlirType tmaGbasis,
                                                           int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaFormat(MlirType type) {
  auto value =
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(unwrap(type))
          .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto kindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTiledTmaLoad(kind);
  if (!kindValue)
    return {nullptr};
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExecTiledTmaLoadType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      *kindValue, valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
uint32_t mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(unwrap(type))
          .getKind());
}
MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaFormat(MlirType type) {
  auto value =
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(unwrap(type))
          .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto kindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeReductionKind(kind);
  if (!kindValue)
    return {nullptr};
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExecTiledTmaReduceType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      *kindValue, valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
uint32_t mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(unwrap(type))
          .getKind());
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaFormat(MlirType type) {
  auto value =
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(unwrap(type))
          .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(MlirContext context,
                                                          MlirType valType,
                                                          int copyBits,
                                                          MlirType tmaGbasis,
                                                          int64_t tmaFormat) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, false, tmaGbasisValue))
    return {nullptr};
  std::optional<::mlir::cutlass_compiler::cute_nvgpu::TmaDataFormat>
      tmaFormatValue;
  if (tmaFormat >= 0) {
    tmaFormatValue =
        ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaDataFormat(
            static_cast<uint32_t>(tmaFormat));
    if (!tmaFormatValue)
      return {nullptr};
  }
  return wrap(cute_nvgpu::CopyAtomNonExecTiledTmaStoreType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, tmaGbasisValue, tmaFormatValue));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(unwrap(type))
      .getCopyBits();
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(unwrap(type))
          .getTmaGbasis()));
}
int64_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaFormat(MlirType type) {
  auto value =
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(unwrap(type))
          .getTmaFormat();
  return value ? static_cast<int64_t>(*value) : -1;
}

bool mlirTypeIsACuteNVGPUCopyAtomR2G(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomR2GType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomR2GTypeGet(MlirContext context, MlirType valType,
                                         int copyBits, uint32_t memOrder,
                                         uint32_t memScope,
                                         uint32_t l1CacheEvictPriority,
                                         uint32_t storeCacheMode,
                                         uint32_t sharedSpace) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  auto l1CacheEvictPriorityValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeCacheEvictionPriority(
          l1CacheEvictPriority);
  if (!l1CacheEvictPriorityValue)
    return {nullptr};
  auto storeCacheModeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeStoreCacheMode(
          storeCacheMode);
  if (!storeCacheModeValue)
    return {nullptr};
  auto sharedSpaceValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSharedSpace(sharedSpace);
  if (!sharedSpaceValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomR2GType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, *memOrderValue, *memScopeValue,
      *l1CacheEvictPriorityValue, *storeCacheModeValue, *sharedSpaceValue));
}
MlirType mlirCuteNVGPUCopyAtomR2GTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type)).getValType()));
}
int mlirCuteNVGPUCopyAtomR2GTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type)).getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomR2GTypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type)).getMemOrder());
}
uint32_t mlirCuteNVGPUCopyAtomR2GTypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type)).getMemScope());
}
uint32_t mlirCuteNVGPUCopyAtomR2GTypeGetL1CacheEvictPriority(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type))
          .getL1CacheEvictPriority());
}
uint32_t mlirCuteNVGPUCopyAtomR2GTypeGetStoreCacheMode(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type))
          .getStoreCacheMode());
}
uint32_t mlirCuteNVGPUCopyAtomR2GTypeGetSharedSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type)).getSharedSpace());
}

bool mlirTypeIsACuteNVGPUCopyAtomR2S(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomR2SType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomR2STypeGet(MlirContext context, MlirType valType,
                                         int copyBits, uint32_t memOrder,
                                         uint32_t memScope,
                                         uint32_t sharedSpace) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  auto sharedSpaceValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSharedSpace(sharedSpace);
  if (!sharedSpaceValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomR2SType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, *memOrderValue, *memScopeValue,
      *sharedSpaceValue));
}
MlirType mlirCuteNVGPUCopyAtomR2STypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomR2SType>(unwrap(type)).getValType()));
}
int mlirCuteNVGPUCopyAtomR2STypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomR2SType>(unwrap(type)).getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomR2STypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2SType>(unwrap(type)).getMemOrder());
}
uint32_t mlirCuteNVGPUCopyAtomR2STypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2SType>(unwrap(type)).getMemScope());
}
uint32_t mlirCuteNVGPUCopyAtomR2STypeGetSharedSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomR2SType>(unwrap(type)).getSharedSpace());
}

bool mlirTypeIsACuteNVGPUCopyAtomS2R(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomS2RType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomS2RTypeGet(MlirContext context, MlirType valType,
                                         int copyBits, uint32_t memOrder,
                                         uint32_t memScope,
                                         uint32_t sharedSpace) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  auto sharedSpaceValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSharedSpace(sharedSpace);
  if (!sharedSpaceValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomS2RType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, *memOrderValue, *memScopeValue,
      *sharedSpaceValue));
}
MlirType mlirCuteNVGPUCopyAtomS2RTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomS2RType>(unwrap(type)).getValType()));
}
int mlirCuteNVGPUCopyAtomS2RTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomS2RType>(unwrap(type)).getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomS2RTypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomS2RType>(unwrap(type)).getMemOrder());
}
uint32_t mlirCuteNVGPUCopyAtomS2RTypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomS2RType>(unwrap(type)).getMemScope());
}
uint32_t mlirCuteNVGPUCopyAtomS2RTypeGetSharedSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomS2RType>(unwrap(type)).getSharedSpace());
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTAsyncCopyType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(MlirContext context,
                                                   MlirType valType,
                                                   uint32_t cache,
                                                   int copyBits) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto cacheValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeLoadCacheMode(cache);
  if (!cacheValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTAsyncCopyType::get(
      ctx, valTypeValue, *cacheValue, copyBits));
}
MlirType mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTAsyncCopyType>(unwrap(type))
          .getValType()));
}
uint32_t mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCache(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTAsyncCopyType>(unwrap(type))
          .getCache());
}
int mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSIMTAsyncCopyType>(unwrap(type))
      .getCopyBits();
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t reduction,
    MlirAttribute ldReduceAccPrecision, uint32_t memOrder, uint32_t memScope) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto reductionValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeReductionKind(reduction);
  if (!reductionValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::LdReduceAccPrecisionKindAttr
      ldReduceAccPrecisionValue;
  if (!unwrapAs(ctx, ldReduceAccPrecision, true, ldReduceAccPrecisionValue))
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTMultimemLdReduceType::get(
      ctx, valTypeValue, copyBits, *reductionValue, ldReduceAccPrecisionValue,
      *memOrderValue, *memScopeValue));
}
MlirType
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
      .getCopyBits();
}
uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetReduction(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
          .getReduction());
}
MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetLdReduceAccPrecision(
    MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
          .getLdReduceAccPrecision()));
}
uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
          .getMemOrder());
}
uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
          .getMemScope());
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTMultimemRedType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t reduction,
    uint32_t memOrder, uint32_t memScope) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto reductionValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeReductionKind(reduction);
  if (!reductionValue)
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTMultimemRedType::get(
      ctx, valTypeValue, copyBits, *reductionValue, *memOrderValue,
      *memScopeValue));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemRedType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemRedType>(unwrap(type))
      .getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetReduction(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemRedType>(unwrap(type))
          .getReduction());
}
uint32_t mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemRedType>(unwrap(type))
          .getMemOrder());
}
uint32_t mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemRedType>(unwrap(type))
          .getMemScope());
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTMultimemStType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(MlirContext context,
                                                    MlirType valType,
                                                    int copyBits,
                                                    uint32_t memOrder,
                                                    uint32_t memScope) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto memOrderValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemOrderKind(memOrder);
  if (!memOrderValue)
    return {nullptr};
  auto memScopeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMemScopeKind(memScope);
  if (!memScopeValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTMultimemStType::get(
      ctx, valTypeValue, copyBits, *memOrderValue, *memScopeValue));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemStType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemStType>(unwrap(type))
      .getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemOrder(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemStType>(unwrap(type))
          .getMemOrder());
}
uint32_t mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemScope(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemStType>(unwrap(type))
          .getMemScope());
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTSyncCopyType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(MlirContext context,
                                                  MlirType valType,
                                                  int copyBits,
                                                  uint32_t srcSpace,
                                                  uint32_t dstSpace) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto srcSpaceValue =
      ::mlir::cutlass_compiler::cute::symbolizeAddressSpace(srcSpace);
  if (!srcSpaceValue)
    return {nullptr};
  auto dstSpaceValue =
      ::mlir::cutlass_compiler::cute::symbolizeAddressSpace(dstSpace);
  if (!dstSpaceValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTSyncCopyType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, copyBits, *srcSpaceValue, *dstSpaceValue));
}
MlirType mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTSyncCopyType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSIMTSyncCopyType>(unwrap(type))
      .getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetSrcSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTSyncCopyType>(unwrap(type))
          .getSrcSpace());
}
uint32_t mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetDstSpace(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTSyncCopyType>(unwrap(type))
          .getDstSpace());
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100CopyS2TType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(MlirContext context,
                                                  MlirType valType, int numDp,
                                                  int numBit, int numCta,
                                                  uint32_t broadcast) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto broadcastValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeCopyS2TBroadcast(
          broadcast);
  if (!broadcastValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100CopyS2TType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, numDp, numBit, numCta, *broadcastValue));
}
MlirType mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumDp(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(unwrap(type))
      .getNumDp();
}
int mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumBit(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(unwrap(type))
      .getNumBit();
}
int mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(unwrap(type))
      .getNumCta();
}
uint32_t mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetBroadcast(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(unwrap(type))
          .getBroadcast());
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(MlirContext context,
                                                    MlirType valType, int numDp,
                                                    int numBit, int numCta,
                                                    uint32_t smemMajor,
                                                    uint32_t broadcast) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto smemMajorValue =
      ::mlir::cutlass_compiler::cute::symbolizeMajorMode(smemMajor);
  if (!smemMajorValue)
    return {nullptr};
  auto broadcastValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeCopyS2TBroadcast(
          broadcast);
  if (!broadcastValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100S2TCopyV2Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, numDp, numBit, numCta, *smemMajorValue, *broadcastValue));
}
MlirType mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumDp(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
      .getNumDp();
}
int mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumBit(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
      .getNumBit();
}
int mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
      .getNumCta();
}
uint32_t mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetSmemMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
          .getSmemMajor());
}
uint32_t mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetBroadcast(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
          .getBroadcast());
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100TmemLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(MlirContext context,
                                                   MlirType valType, int numDp,
                                                   int numBit, uint32_t numRep,
                                                   bool pack16b) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100TmemLoadType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, numDp, numBit, numRep,
      pack16b ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr()));
}
MlirType mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumDp(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(unwrap(type))
      .getNumDp();
}
int mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumBit(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(unwrap(type))
      .getNumBit();
}
uint32_t mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumRep(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(unwrap(type))
      .getNumRep();
}
bool mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPack16b(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(unwrap(type))
          .getPack16b());
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100TmemStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(MlirContext context,
                                                    MlirType valType, int numDp,
                                                    int numBit, uint32_t numRep,
                                                    bool expand16b) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100TmemStoreType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, numDp, numBit, numRep,
      expand16b ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr()));
}
MlirType mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumDp(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(unwrap(type))
      .getNumDp();
}
int mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumBit(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(unwrap(type))
      .getNumBit();
}
uint32_t mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumRep(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(unwrap(type))
      .getNumRep();
}
bool mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetExpand16b(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(unwrap(type))
          .getExpand16b());
}

bool mlirTypeIsACuteNVGPUCopyAtomSM107TmemLoadSPCompress(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit,
    uint32_t numRep, uint32_t redOp, bool red, bool nan) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto redOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmemLoadRedOp(redOp);
  if (!redOpValue)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, numDp, numBit, numRep, *redOpValue,
      red ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr(),
      nan ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr()));
}
MlirType
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumDp(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(
             unwrap(type))
      .getNumDp();
}
int mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumBit(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(
             unwrap(type))
      .getNumBit();
}
uint32_t
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumRep(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(
             unwrap(type))
      .getNumRep();
}
uint32_t
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRedOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(unwrap(type))
          .getRedOp());
}
bool mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRed(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(unwrap(type))
          .getRed());
}
bool mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNan(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>(unwrap(type))
          .getNan());
}

bool mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit,
    uint32_t numRep, uint32_t redOp, bool nan, MlirAttribute halfSplitOff) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto redOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmemLoadRedOp(redOp);
  if (!redOpValue)
    return {nullptr};
  ::mlir::IntegerAttr halfSplitOffValue;
  if (!unwrapAs(ctx, halfSplitOff, true, halfSplitOffValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM10xTmemLoadRedType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, numDp, numBit, numRep, *redOpValue,
      nan ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr(), halfSplitOffValue));
}
MlirType mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumDp(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
      .getNumDp();
}
int mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumBit(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
      .getNumBit();
}
uint32_t mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumRep(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
      .getNumRep();
}
uint32_t mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetRedOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
          .getRedOp());
}
bool mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNan(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
          .getNan());
}
MlirAttribute
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetHalfSplitOff(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
          .getHalfSplitOff()));
}

bool mlirTypeIsACuteNVGPUCopyAtomStsm(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomStsmType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomStsmTypeGet(MlirContext context, MlirType valType,
                                          MlirAttribute mode, int numMatrices,
                                          bool transpose) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::ShapeAttr modeValue;
  if (!unwrapAs(ctx, mode, false, modeValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomStsmType::get(
      ctx, valTypeValue, modeValue, numMatrices,
      transpose ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr()));
}
MlirType mlirCuteNVGPUCopyAtomStsmTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomStsmType>(unwrap(type)).getValType()));
}
MlirAttribute mlirCuteNVGPUCopyAtomStsmTypeGetMode(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomStsmType>(unwrap(type)).getMode()));
}
int mlirCuteNVGPUCopyAtomStsmTypeGetNumMatrices(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomStsmType>(unwrap(type))
      .getNumMatrices();
}
bool mlirCuteNVGPUCopyAtomStsmTypeGetTranspose(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::CopyAtomStsmType>(unwrap(type)).getTranspose());
}

bool mlirTypeIsACuteNVGPUCopyAtomTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomTmaLoadTypeGet(
    MlirContext context, MlirType valType, int sparsity, int copyBits,
    uint32_t mode, int numCta, MlirType gStride, bool mcast, MlirType tmaGbasis,
    bool override, bool noFullyOobTile) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto modeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaLoadMode(mode);
  if (!modeValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute::StrideType gStrideValue;
  if (!unwrapAs(ctx, gStride, false, gStrideValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, true, tmaGbasisValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomTmaLoadType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, sparsity, copyBits, *modeValue, numCta, gStrideValue, mcast,
      tmaGbasisValue, override, noFullyOobTile));
}
MlirType mlirCuteNVGPUCopyAtomTmaLoadTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type)).getValType()));
}
int mlirCuteNVGPUCopyAtomTmaLoadTypeGetSparsity(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type))
      .getSparsity();
}
int mlirCuteNVGPUCopyAtomTmaLoadTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type))
      .getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomTmaLoadTypeGetMode(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type)).getMode());
}
int mlirCuteNVGPUCopyAtomTmaLoadTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type)).getNumCta();
}
MlirType mlirCuteNVGPUCopyAtomTmaLoadTypeGetGStride(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type)).getGStride()));
}
bool mlirCuteNVGPUCopyAtomTmaLoadTypeGetMcast(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type)).getMcast();
}
MlirType mlirCuteNVGPUCopyAtomTmaLoadTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type))
          .getTmaGbasis()));
}
bool mlirCuteNVGPUCopyAtomTmaLoadTypeGetOverride(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type))
      .getOverride();
}
bool mlirCuteNVGPUCopyAtomTmaLoadTypeGetNoFullyOobTile(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type))
      .getNoFullyOobTile();
}

bool mlirTypeIsACuteNVGPUCopyAtomTmaReduce(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGet(MlirContext context,
                                               MlirType valType, int copyBits,
                                               uint32_t mode, uint32_t kind,
                                               MlirType gStride,
                                               MlirType tmaGbasis) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto modeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaStoreMode(mode);
  if (!modeValue)
    return {nullptr};
  auto kindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeReductionKind(kind);
  if (!kindValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute::StrideType gStrideValue;
  if (!unwrapAs(ctx, gStride, false, gStrideValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, true, tmaGbasisValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomTmaReduceType::get(
      ctx, valTypeValue, copyBits, *modeValue, *kindValue, gStrideValue,
      tmaGbasisValue));
}
MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type))
          .getValType()));
}
int mlirCuteNVGPUCopyAtomTmaReduceTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type))
      .getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomTmaReduceTypeGetMode(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type)).getMode());
}
uint32_t mlirCuteNVGPUCopyAtomTmaReduceTypeGetKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type)).getKind());
}
MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGetGStride(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type))
          .getGStride()));
}
MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type))
          .getTmaGbasis()));
}

bool mlirTypeIsACuteNVGPUCopyAtomTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGet(MlirContext context,
                                              MlirType valType, int sparsity,
                                              int copyBits, uint32_t mode,
                                              MlirType gStride,
                                              MlirType tmaGbasis, bool override,
                                              bool noFullyOobTile) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type valTypeValue;
  if (!unwrapAs(ctx, valType, false, valTypeValue))
    return {nullptr};
  auto modeValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeTmaStoreMode(mode);
  if (!modeValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute::StrideType gStrideValue;
  if (!unwrapAs(ctx, gStride, false, gStrideValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutType tmaGbasisValue;
  if (!unwrapAs(ctx, tmaGbasis, true, tmaGbasisValue))
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomTmaStoreType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      valTypeValue, sparsity, copyBits, *modeValue, gStrideValue,
      tmaGbasisValue, override, noFullyOobTile));
}
MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGetValType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type)).getValType()));
}
int mlirCuteNVGPUCopyAtomTmaStoreTypeGetSparsity(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type))
      .getSparsity();
}
int mlirCuteNVGPUCopyAtomTmaStoreTypeGetCopyBits(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type))
      .getCopyBits();
}
uint32_t mlirCuteNVGPUCopyAtomTmaStoreTypeGetMode(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type)).getMode());
}
MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGetGStride(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type)).getGStride()));
}
MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGetTmaGbasis(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type))
          .getTmaGbasis()));
}
bool mlirCuteNVGPUCopyAtomTmaStoreTypeGetOverride(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type))
      .getOverride();
}
bool mlirCuteNVGPUCopyAtomTmaStoreTypeGetNoFullyOobTile(MlirType type) {
  return llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type))
      .getNoFullyOobTile();
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse(MlirType type) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int vecSize, uint32_t archPromote, MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type sfTypeValue;
  if (!unwrapAs(ctx, sfType, false, sfTypeValue))
    return {nullptr};
  auto sparseMetadataFormatValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSparseMetadataFormat(
          sparseMetadataFormat);
  if (!sparseMetadataFormatValue)
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  auto archPromoteValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeArch(archPromote);
  if (!archPromoteValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, sfTypeValue, *sparseMetadataFormatValue, *aFragKindValue,
      vecSize, *archPromoteValue, intOverflowValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
             unwrap(type))
      .getNumCta();
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getAMajor());
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getBMajor());
}
MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getAType()));
}
MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getBType()));
}
MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getCType()));
}
MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSfType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getSfType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSparseMetadataFormat(
    MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getSparseMetadataFormat());
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetVecSize(
    MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
             unwrap(type))
      .getVecSize();
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetArchPromote(
    MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getArchPromote());
}
MlirAttribute mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetIntOverflow(
    MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t aFragKind, int vecSize, uint32_t archPromote,
    MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type sfTypeValue;
  if (!unwrapAs(ctx, sfType, false, sfTypeValue))
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  auto archPromoteValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeArch(archPromote);
  if (!archPromoteValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMABlockScaledType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, sfTypeValue, *aFragKindValue, vecSize, *archPromoteValue,
      intOverflowValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
      .getNumCta();
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getCType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetSfType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getSfType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetVecSize(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
      .getVecSize();
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetArchPromote(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getArchPromote());
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM100UMMASparseType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType eType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int cScaleExp, MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type eTypeValue;
  if (!unwrapAs(ctx, eType, false, eTypeValue))
    return {nullptr};
  auto sparseMetadataFormatValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSparseMetadataFormat(
          sparseMetadataFormat);
  if (!sparseMetadataFormatValue)
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMASparseType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, eTypeValue, *sparseMetadataFormatValue, *aFragKindValue,
      cScaleExp, intOverflowValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
      .getNumCta();
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getCType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetEType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getEType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetSparseMetadataFormat(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getSparseMetadataFormat());
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCScaleExp(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
      .getCScaleExp();
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMA(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    uint32_t aFragKind, int cScaleExp, MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMAType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, *aFragKindValue, cScaleExp, intOverflowValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM100UMMATypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM100UMMATypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getNumCta();
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMATypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMATypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getCType()));
}
uint32_t mlirCuteNVGPUMmaAtomSM100UMMATypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM100UMMATypeGetCScaleExp(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type))
      .getCScaleExp();
}
MlirAttribute mlirCuteNVGPUMmaAtomSM100UMMATypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaledSparse(MlirType type) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int vecSize, uint32_t aCollectorOp, uint32_t bCollectorOp,
    MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type sfTypeValue;
  if (!unwrapAs(ctx, sfType, false, sfTypeValue))
    return {nullptr};
  auto sparseMetadataFormatValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSparseMetadataFormat(
          sparseMetadataFormat);
  if (!sparseMetadataFormatValue)
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  auto aCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          aCollectorOp);
  if (!aCollectorOpValue)
    return {nullptr};
  auto bCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          bCollectorOp);
  if (!bCollectorOpValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, sfTypeValue, *sparseMetadataFormatValue, *aFragKindValue,
      vecSize, *aCollectorOpValue, *bCollectorOpValue, intOverflowValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
             unwrap(type))
      .getNumCta();
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getAMajor());
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getBMajor());
}
MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getAType()));
}
MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getBType()));
}
MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getCType()));
}
MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSfType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getSfType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSparseMetadataFormat(
    MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getSparseMetadataFormat());
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetVecSize(
    MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
             unwrap(type))
      .getVecSize();
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetACollectorOp(
    MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getACollectorOp());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBCollectorOp(
    MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getBCollectorOp());
}
MlirAttribute mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetIntOverflow(
    MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>(
          unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaled(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t aFragKind, int vecSize, uint32_t aCollectorOp,
    uint32_t bCollectorOp, MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type sfTypeValue;
  if (!unwrapAs(ctx, sfType, false, sfTypeValue))
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  auto aCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          aCollectorOp);
  if (!aCollectorOpValue)
    return {nullptr};
  auto bCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          bCollectorOp);
  if (!bCollectorOpValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM107UMMABlockScaledType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, sfTypeValue, *aFragKindValue, vecSize, *aCollectorOpValue,
      *bCollectorOpValue, intOverflowValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
      .getNumCta();
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getCType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetSfType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getSfType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetVecSize(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
      .getVecSize();
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetACollectorOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getACollectorOp());
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBCollectorOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getBCollectorOp());
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMABlockScaledType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM107UMMASparse(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM107UMMASparseType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType eType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int cScaleExp, uint32_t aCollectorOp, uint32_t bCollectorOp,
    MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type eTypeValue;
  if (!unwrapAs(ctx, eType, false, eTypeValue))
    return {nullptr};
  auto sparseMetadataFormatValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSparseMetadataFormat(
          sparseMetadataFormat);
  if (!sparseMetadataFormatValue)
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  auto aCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          aCollectorOp);
  if (!aCollectorOpValue)
    return {nullptr};
  auto bCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          bCollectorOp);
  if (!bCollectorOpValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM107UMMASparseType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, eTypeValue, *sparseMetadataFormatValue, *aFragKindValue,
      cScaleExp, *aCollectorOpValue, *bCollectorOpValue, intOverflowValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
      .getNumCta();
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getCType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetEType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getEType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetSparseMetadataFormat(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getSparseMetadataFormat());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCScaleExp(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
      .getCScaleExp();
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetACollectorOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getACollectorOp());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBCollectorOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getBCollectorOp());
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMASparseType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM107UMMA(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMATypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    uint32_t aFragKind, int cScaleExp, uint32_t aCollectorOp,
    uint32_t bCollectorOp, MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  auto aCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          aCollectorOp);
  if (!aCollectorOpValue)
    return {nullptr};
  auto bCollectorOpValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaCollectorOp(
          bCollectorOp);
  if (!bCollectorOpValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM107UMMAType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, numCta, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, *aFragKindValue, cScaleExp, *aCollectorOpValue,
      *bCollectorOpValue, intOverflowValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM107UMMATypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM107UMMATypeGetNumCta(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type)).getNumCta();
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMATypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type)).getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMATypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type)).getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMATypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMATypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM107UMMATypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type)).getCType()));
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMATypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type))
          .getAFragKind());
}
int mlirCuteNVGPUMmaAtomSM107UMMATypeGetCScaleExp(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type))
      .getCScaleExp();
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMATypeGetACollectorOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type))
          .getACollectorOp());
}
uint32_t mlirCuteNVGPUMmaAtomSM107UMMATypeGetBCollectorOp(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type))
          .getBCollectorOp());
}
MlirAttribute mlirCuteNVGPUMmaAtomSM107UMMATypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM107UMMAType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM120BlockScaledType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int vecSize, MlirType aType,
    MlirType bType, MlirType cType, MlirType sfType, bool useSfLayoutTV) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  mlir::Type sfTypeValue;
  if (!unwrapAs(ctx, sfType, false, sfTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM120BlockScaledType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, vecSize, aTypeValue, bTypeValue, cTypeValue, sfTypeValue,
      useSfLayoutTV));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
          .getShapeMnk()));
}
int mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetVecSize(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
      .getVecSize();
}
MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
          .getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
          .getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
          .getCType()));
}
MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetSfType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
          .getSfType()));
}
bool mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetUseSfLayoutTV(MlirType type) {
  return llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
      .getUseSfLayout_TV();
}

bool mlirTypeIsACuteNVGPUMmaAtomSM80Sparse(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, MlirType aType, MlirType bType,
    MlirType cType, uint32_t sparseMetadataFormat, MlirAttribute intOverflow) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  auto sparseMetadataFormatValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeSparseMetadataFormat(
          sparseMetadataFormat);
  if (!sparseMetadataFormatValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM80SparseType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, aTypeValue, bTypeValue, cTypeValue,
      *sparseMetadataFormatValue, intOverflowValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80SparseTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type))
          .getShapeMnk()));
}
MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type)).getCType()));
}
uint32_t
mlirCuteNVGPUMmaAtomSM80SparseTypeGetSparseMetadataFormat(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type))
          .getSparseMetadataFormat());
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80SparseTypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type))
          .getIntOverflow()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM80(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM80Type>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM80TypeGet(MlirContext context,
                                         MlirAttribute shapeMnk, MlirType aType,
                                         MlirType bType, MlirType cType,
                                         MlirAttribute intOverflow,
                                         MlirAttribute binaryOp) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::BinaryOpAttr binaryOpValue;
  if (!unwrapAs(ctx, binaryOp, true, binaryOpValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM80Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, aTypeValue, bTypeValue, cTypeValue, intOverflowValue,
      binaryOpValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80TypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getShapeMnk()));
}
MlirType mlirCuteNVGPUMmaAtomSM80TypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM80TypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM80TypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getCType()));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80TypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getIntOverflow()));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80TypeGetBinaryOp(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getBinaryOp()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM89(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM89Type>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM89TypeGet(MlirContext context,
                                         MlirAttribute shapeMnk, MlirType aType,
                                         MlirType bType, MlirType cType) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM89Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, aTypeValue, bTypeValue, cTypeValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM89TypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM89Type>(unwrap(type)).getShapeMnk()));
}
MlirType mlirCuteNVGPUMmaAtomSM89TypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM89Type>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM89TypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM89Type>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM89TypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM89Type>(unwrap(type)).getCType()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM90(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM90Type>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM90TypeGet(
    MlirContext context, MlirAttribute shapeMnk, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    uint32_t aFragKind, MlirAttribute intOverflow, bool aNeg, bool bNeg) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  auto aMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(aMajor);
  if (!aMajorValue)
    return {nullptr};
  auto bMajorValue = ::mlir::cutlass_compiler::cute::symbolizeMajorMode(bMajor);
  if (!bMajorValue)
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  auto aFragKindValue =
      ::mlir::cutlass_compiler::cute_nvgpu::symbolizeMmaFragKind(aFragKind);
  if (!aFragKindValue)
    return {nullptr};
  ::mlir::cutlass_compiler::cute_nvgpu::MMAIntOverflowAttr intOverflowValue;
  if (!unwrapAs(ctx, intOverflow, true, intOverflowValue))
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM90Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, *aMajorValue, *bMajorValue, aTypeValue, bTypeValue,
      cTypeValue, *aFragKindValue, intOverflowValue,
      aNeg ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr(),
      bNeg ? mlir::UnitAttr::get(ctx) : mlir::UnitAttr()));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM90TypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getShapeMnk()));
}
uint32_t mlirCuteNVGPUMmaAtomSM90TypeGetAMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getAMajor());
}
uint32_t mlirCuteNVGPUMmaAtomSM90TypeGetBMajor(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getBMajor());
}
MlirType mlirCuteNVGPUMmaAtomSM90TypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUMmaAtomSM90TypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUMmaAtomSM90TypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getCType()));
}
uint32_t mlirCuteNVGPUMmaAtomSM90TypeGetAFragKind(MlirType type) {
  return static_cast<uint32_t>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getAFragKind());
}
MlirAttribute mlirCuteNVGPUMmaAtomSM90TypeGetIntOverflow(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getIntOverflow()));
}
bool mlirCuteNVGPUMmaAtomSM90TypeGetANeg(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getANeg());
}
bool mlirCuteNVGPUMmaAtomSM90TypeGetBNeg(MlirType type) {
  return static_cast<bool>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getBNeg());
}

bool mlirTypeIsACuteNVGPUSmemDescCircularSM103(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::SmemDescCircularSM103Type>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUSmemDescCircularSM103TypeGet(MlirContext context,
                                          MlirAttribute blockLayoutBytes) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::LayoutAttr blockLayoutBytesValue;
  if (!unwrapAs(ctx, blockLayoutBytes, false, blockLayoutBytesValue))
    return {nullptr};
  return wrap(cute_nvgpu::SmemDescCircularSM103Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      blockLayoutBytesValue));
}
MlirAttribute
mlirCuteNVGPUSmemDescCircularSM103TypeGetBlockLayoutBytes(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::SmemDescCircularSM103Type>(unwrap(type))
          .getBlockLayoutBytes()));
}

bool mlirTypeIsACuteNVGPUSmemDescCircularSM107(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::SmemDescCircularSM107Type>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUSmemDescCircularSM107TypeGet(MlirContext context,
                                          MlirAttribute blockLayoutBytes) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::LayoutAttr blockLayoutBytesValue;
  if (!unwrapAs(ctx, blockLayoutBytes, false, blockLayoutBytesValue))
    return {nullptr};
  return wrap(cute_nvgpu::SmemDescCircularSM107Type::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      blockLayoutBytesValue));
}
MlirAttribute
mlirCuteNVGPUSmemDescCircularSM107TypeGetBlockLayoutBytes(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::SmemDescCircularSM107Type>(unwrap(type))
          .getBlockLayoutBytes()));
}

bool mlirTypeIsACuteNVGPUSmemDescSM107(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::SmemDescSM107Type>(unwrap(type));
}
MlirType mlirCuteNVGPUSmemDescSM107TypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  return wrap(cute_nvgpu::SmemDescSM107Type::get(ctx));
}

bool mlirTypeIsACuteNVGPUSmemDesc(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::SmemDescType>(unwrap(type));
}
MlirType mlirCuteNVGPUSmemDescTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  return wrap(cute_nvgpu::SmemDescType::get(ctx));
}

bool mlirTypeIsACuteNVGPUSmemDescView(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::SmemDescViewType>(unwrap(type));
}
MlirType mlirCuteNVGPUSmemDescViewTypeGet(MlirContext context, MlirType desc,
                                          MlirAttribute layout) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  mlir::Type descValue;
  if (!unwrapAs(ctx, desc, false, descValue))
    return {nullptr};
  ::mlir::cutlass_compiler::cute::LayoutAttr layoutValue;
  if (!unwrapAs(ctx, layout, false, layoutValue))
    return {nullptr};
  return wrap(cute_nvgpu::SmemDescViewType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      descValue, layoutValue));
}
MlirType mlirCuteNVGPUSmemDescViewTypeGetDesc(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::SmemDescViewType>(unwrap(type)).getDesc()));
}
MlirAttribute mlirCuteNVGPUSmemDescViewTypeGetLayout(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::SmemDescViewType>(unwrap(type)).getLayout()));
}

bool mlirTypeIsACuteNVGPUTmaDescriptorIm2Col(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmaDescriptorIm2ColType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUTmaDescriptorIm2ColTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  return wrap(cute_nvgpu::TmaDescriptorIm2ColType::get(ctx));
}

bool mlirTypeIsACuteNVGPUTmaDescriptorTiled(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::TmaDescriptorTiledType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUTmaDescriptorTiledTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  return wrap(cute_nvgpu::TmaDescriptorTiledType::get(ctx));
}

bool mlirTypeIsACuteNVGPUUniversalFmaAtom(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::UniversalFmaAtomType>(unwrap(type));
}
MlirType mlirCuteNVGPUUniversalFmaAtomTypeGet(MlirContext context,
                                              MlirAttribute shapeMnk,
                                              MlirType aType, MlirType bType,
                                              MlirType cType) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  ::mlir::cutlass_compiler::cute::ShapeAttr shapeMnkValue;
  if (!unwrapAs(ctx, shapeMnk, false, shapeMnkValue))
    return {nullptr};
  mlir::Type aTypeValue;
  if (!unwrapAs(ctx, aType, false, aTypeValue))
    return {nullptr};
  mlir::Type bTypeValue;
  if (!unwrapAs(ctx, bType, false, bTypeValue))
    return {nullptr};
  mlir::Type cTypeValue;
  if (!unwrapAs(ctx, cType, false, cTypeValue))
    return {nullptr};
  return wrap(cute_nvgpu::UniversalFmaAtomType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx,
      shapeMnkValue, aTypeValue, bTypeValue, cTypeValue));
}
MlirAttribute mlirCuteNVGPUUniversalFmaAtomTypeGetShapeMnk(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::UniversalFmaAtomType>(unwrap(type))
          .getShapeMnk()));
}
MlirType mlirCuteNVGPUUniversalFmaAtomTypeGetAType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::UniversalFmaAtomType>(unwrap(type)).getAType()));
}
MlirType mlirCuteNVGPUUniversalFmaAtomTypeGetBType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::UniversalFmaAtomType>(unwrap(type)).getBType()));
}
MlirType mlirCuteNVGPUUniversalFmaAtomTypeGetCType(MlirType type) {
  return wrap(static_cast<mlir::Type>(
      llvm::cast<cute_nvgpu::UniversalFmaAtomType>(unwrap(type)).getCType()));
}

bool mlirTypeIsACuteNVGPUWorkIdResponse(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::WorkIdResponseType>(unwrap(type));
}
MlirType mlirCuteNVGPUWorkIdResponseTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  return wrap(cute_nvgpu::WorkIdResponseType::get(ctx));
}
