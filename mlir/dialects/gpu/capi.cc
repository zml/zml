//===- capi.cc - C API for the gpu dialect --------------------------------===//

#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, mlir::gpu::GPUDialect)

static mlir::MLIRContext *loadGPU(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx)
    ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  return ctx;
}

bool mlirAttributeIsAGPUDimension(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::gpu::DimensionAttr>(unwrap(attr));
}

MlirAttribute mlirGPUDimensionAttrGet(MlirContext context, uint32_t value) {
  mlir::MLIRContext *ctx = loadGPU(context);
  std::optional<mlir::gpu::Dimension> dimension =
      mlir::gpu::symbolizeDimension(value);
  if (!ctx || !dimension)
    return {nullptr};
  return wrap(mlir::gpu::DimensionAttr::get(ctx, *dimension));
}

uint32_t mlirGPUDimensionAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::gpu::DimensionAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAGPUShuffleMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::gpu::ShuffleModeAttr>(unwrap(attr));
}

MlirAttribute mlirGPUShuffleModeAttrGet(MlirContext context, uint32_t value) {
  mlir::MLIRContext *ctx = loadGPU(context);
  std::optional<mlir::gpu::ShuffleMode> mode =
      mlir::gpu::symbolizeShuffleMode(value);
  if (!ctx || !mode)
    return {nullptr};
  return wrap(mlir::gpu::ShuffleModeAttr::get(ctx, *mode));
}

uint32_t mlirGPUShuffleModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::gpu::ShuffleModeAttr>(unwrap(attr)).getValue());
}
