// C++ to C bindings for CuTe NVIDIA GPU types.
#include "cute_ir-c/Dialect/CuteNVGPUTypes.h"
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"
#include "mlir/CAPI/IR.h"

namespace cute_nvgpu = mlir::cutlass_compiler::cute_nvgpu;

bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomBulkCopyG2SType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(MlirContext context,
                                                 MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomBulkCopyG2SType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomBulkCopyG2SType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomBulkCopyS2GType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(MlirContext context,
                                                 MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomBulkCopyS2GType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2GType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomBulkCopyS2SType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(MlirContext context,
                                                 MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomBulkCopyS2SType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomBulkCopyS2SType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomDsmemStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomDsmemStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(MlirContext context,
                                                MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomDsmemStoreType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomDsmemStoreTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomDsmemStoreType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomG2R(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomG2RType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomG2RTypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomG2RType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomG2RTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomG2RType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(MlirContext context,
                                                   MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomIm2ColTmaLoadType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaLoadType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(MlirContext context,
                                                    MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomIm2ColTmaStoreType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomIm2ColTmaStoreType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomLdsm(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomLdsmType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomLdsmTypeGet(MlirContext context,
                                          MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomLdsmType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomLdsmTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomLdsmType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(MlirContext context,
                                                    MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(MlirContext context,
                                                 MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(MlirContext context,
                                                  MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(MlirContext context,
                                                MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomNonExecTiledTmaLoadType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(MlirContext context,
                                                  MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomNonExecTiledTmaReduceType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(MlirContext context,
                                                 MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomNonExecTiledTmaStoreType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomR2G(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomR2GType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomR2GTypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomR2GType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomR2GTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomR2GType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomR2S(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomR2SType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomR2STypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomR2SType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomR2STypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomR2SType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomS2R(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomS2RType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomS2RTypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomS2RType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomS2RTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomS2RType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTAsyncCopyType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(MlirContext context,
                                                   MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTAsyncCopyType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTAsyncCopyType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(MlirContext context,
                                                 MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::CopyAtomSIMTMultimemLdReduceType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTMultimemRedType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(MlirContext context,
                                                     MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTMultimemRedType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemRedType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTMultimemStType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(MlirContext context,
                                                    MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTMultimemStType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTMultimemStType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSIMTSyncCopyType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(MlirContext context,
                                                  MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSIMTSyncCopyType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSIMTSyncCopyType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(MlirContext context,
                                                      MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM10xTmemLoadRedType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100CopyS2TType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(MlirContext context,
                                                  MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100CopyS2TType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(MlirContext context,
                                                    MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100S2TCopyV2Type::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSM100S2TCopyV2Type>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100TmemLoadType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(MlirContext context,
                                                   MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100TmemLoadType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomSM100TmemStoreType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(MlirContext context,
                                                    MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomSM100TmemStoreType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomStsm(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomStsmType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomStsmTypeGet(MlirContext context,
                                          MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomStsmType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomStsmTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomStsmType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomTmaLoad(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomTmaLoadTypeGet(MlirContext context,
                                             MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomTmaLoadType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomTmaLoadTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomTmaLoadType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomTmaReduce(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGet(MlirContext context,
                                               MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomTmaReduceType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomTmaReduceTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomTmaReduceType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUCopyAtomTmaStore(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type));
}
MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGet(MlirContext context,
                                              MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::CopyAtomTmaStoreType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUCopyAtomTmaStoreTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::CopyAtomTmaStoreType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM80Sparse(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGet(MlirContext context,
                                               MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM80SparseType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80SparseTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80SparseType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM80(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM80Type>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM80TypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM80Type::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM80TypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM80Type>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM89(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM89Type>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM89TypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM89Type::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM89TypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM89Type>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM90(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM90Type>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM90TypeGet(MlirContext context,
                                         MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM90Type::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM90TypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM90Type>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse(MlirType type) {
  return llvm::isa_and_nonnull<
      cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(unwrap(type));
}
MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(MlirContext context,
                                                      MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType::get(
      ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>(
          unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(
      unwrap(type));
}
MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(MlirContext context,
                                                MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(
      cute_nvgpu::MmaAtomSM100UMMABlockScaledType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM100UMMASparseType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(MlirContext context,
                                                    MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMASparseType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMASparseType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMA(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGet(MlirContext context,
                                              MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM100UMMAType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUMmaAtomSM100UMMATypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM100UMMAType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::MmaAtomSM120BlockScaledType>(
      unwrap(type));
}
MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(MlirContext context,
                                                     MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::MmaAtomSM120BlockScaledType::get(ctx, payloadValue));
}
MlirAttribute
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::MmaAtomSM120BlockScaledType>(unwrap(type))
          .getPayload()));
}

bool mlirTypeIsACuteNVGPUSmemDescCircularSM103(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::SmemDescCircularSM103Type>(
      unwrap(type));
}
MlirType mlirCuteNVGPUSmemDescCircularSM103TypeGet(MlirContext context,
                                                   MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::SmemDescCircularSM103Type::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUSmemDescCircularSM103TypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::SmemDescCircularSM103Type>(unwrap(type))
          .getPayload()));
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
MlirType mlirCuteNVGPUSmemDescViewTypeGet(MlirContext context,
                                          MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::SmemDescViewType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUSmemDescViewTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::SmemDescViewType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUTiledCopy(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::TiledCopyType>(unwrap(type));
}
MlirType mlirCuteNVGPUTiledCopyTypeGet(MlirContext context,
                                       MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::TiledCopyType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUTiledCopyTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::TiledCopyType>(unwrap(type)).getPayload()));
}

bool mlirTypeIsACuteNVGPUTiledMma(MlirType type) {
  return llvm::isa_and_nonnull<cute_nvgpu::TiledMmaType>(unwrap(type));
}
MlirType mlirCuteNVGPUTiledMmaTypeGet(MlirContext context,
                                      MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::TiledMmaType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUTiledMmaTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::TiledMmaType>(unwrap(type)).getPayload()));
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
                                              MlirAttribute payload) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cute_nvgpu::CuteNVGPUDialect>();
  auto payloadValue =
      llvm::dyn_cast_if_present<mlir::StringAttr>(unwrap(payload));
  if (!payloadValue || payloadValue.getContext() != ctx)
    return {nullptr};
  return wrap(cute_nvgpu::UniversalFmaAtomType::get(ctx, payloadValue));
}
MlirAttribute mlirCuteNVGPUUniversalFmaAtomTypeGetPayload(MlirType type) {
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cute_nvgpu::UniversalFmaAtomType>(unwrap(type)).getPayload()));
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
