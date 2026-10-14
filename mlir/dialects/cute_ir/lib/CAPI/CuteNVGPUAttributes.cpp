// C API for the CuTe NVIDIA GPU attributes, generated from the dialect's .td
// files.
#include "cute_ir-c/Dialect/CuteNVGPUAttributes.h"

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"
#include "mlir/CAPI/IR.h"

namespace cute_nvgpu = mlir::cutlass_compiler::cute_nvgpu;

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
