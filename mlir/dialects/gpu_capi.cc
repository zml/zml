#include "gpu_capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, mlir::gpu::GPUDialect)

namespace {

mlir::MLIRContext *loadGPU(MlirContext ctx) {
  mlir::MLIRContext *context = unwrap(ctx);
  if (context) context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  return context;
}

}  // namespace

bool mlirAttributeIsAGPUDimension(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::gpu::DimensionAttr>(unwrap(attr));
}

MlirAttribute mlirGPUDimensionAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *context = loadGPU(ctx);
  std::optional<mlir::gpu::Dimension> dimension = mlir::gpu::symbolizeDimension(value);
  if (!context || !dimension) return {nullptr};
  return wrap(mlir::gpu::DimensionAttr::get(context, *dimension));
}

uint32_t mlirGPUDimensionAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<mlir::gpu::DimensionAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAGPUShuffleMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::gpu::ShuffleModeAttr>(unwrap(attr));
}

MlirAttribute mlirGPUShuffleModeAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *context = loadGPU(ctx);
  std::optional<mlir::gpu::ShuffleMode> mode = mlir::gpu::symbolizeShuffleMode(value);
  if (!context || !mode) return {nullptr};
  return wrap(mlir::gpu::ShuffleModeAttr::get(context, *mode));
}

uint32_t mlirGPUShuffleModeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<mlir::gpu::ShuffleModeAttr>(unwrap(attr)).getValue());
}
