// C API for the cuda attributes, generated from the dialect's .td
// files.
#include "cute_ir-c/Dialect/CudaAttributes.h"

#include "cute_ir/Dialect/Cuda/IR/CudaDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

namespace cuda = mlir::cutlass_compiler::cuda;

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

bool mlirAttributeIsACudaAssumeKernel(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::AssumeKernelAttr>(unwrap(attr));
}
MlirAttribute mlirCudaAssumeKernelAttrGet(MlirContext context, bool value) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::AssumeKernelAttr::get(ctx, value));
}
bool mlirCudaAssumeKernelAttrGetValue(MlirAttribute attr) {
  return llvm::cast<cuda::AssumeKernelAttr>(unwrap(attr)).getValue();
}

bool mlirAttributeIsACudaComputeTarget(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::ComputeTargetAttr>(unwrap(attr));
}
MlirAttribute mlirCudaComputeTargetAttrGet(MlirContext context, uint32_t representation, uint32_t portability, intptr_t numArchs, const uint32_t *archs) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  auto representationValue = ::mlir::cutlass_compiler::cuda::symbolizeExecutableRepresentation(representation);
  if (!representationValue)
    return {nullptr};
  auto portabilityValue = ::mlir::cutlass_compiler::cuda::symbolizeArchPortability(portability);
  if (!portabilityValue)
    return {nullptr};
  llvm::SmallVector<::mlir::cutlass_compiler::cuda::GpuArchitecture> archsValue;
  for (intptr_t i = 0; i < numArchs; ++i) {
    auto value = ::mlir::cutlass_compiler::cuda::symbolizeGpuArchitecture(archs[i]);
    if (!value)
      return {nullptr};
    archsValue.push_back(*value);
  }
  return wrap(cuda::ComputeTargetAttr::get(ctx, *representationValue, *portabilityValue, archsValue));
}
uint32_t mlirCudaComputeTargetAttrGetRepresentation(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getRepresentation());
}
uint32_t mlirCudaComputeTargetAttrGetPortability(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getPortability());
}
intptr_t mlirCudaComputeTargetAttrGetNumArchs(MlirAttribute attr) {
  return llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getArchs().size();
}
uint32_t mlirCudaComputeTargetAttrGetArchs(MlirAttribute attr, intptr_t pos) {
  return static_cast<uint32_t>(llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getArchs()[pos]);
}

bool mlirAttributeIsACudaDevMaxSharedMemoryOptin(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::DevMaxSharedMemoryOptinAttr>(unwrap(attr));
}
MlirAttribute mlirCudaDevMaxSharedMemoryOptinAttrGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::DevMaxSharedMemoryOptinAttr::get(ctx));
}

bool mlirAttributeIsACudaDeviceAttributes(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::DeviceAttributesAttr>(unwrap(attr));
}
MlirAttribute mlirCudaDeviceAttributesAttrGet(MlirContext context, MlirAttribute values) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  ::mlir::DictionaryAttr valuesValue;
  if (!unwrapAs(ctx, values, false, valuesValue))
    return {nullptr};
  return wrap(cuda::DeviceAttributesAttr::get(ctx, valuesValue));
}
MlirAttribute mlirCudaDeviceAttributesAttrGetValues(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(llvm::cast<cuda::DeviceAttributesAttr>(unwrap(attr)).getValues()));
}

bool mlirAttributeIsACudaExecutable(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::ExecutableAttr>(unwrap(attr));
}
MlirAttribute mlirCudaExecutableAttrGet(MlirContext context, uint32_t format) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  auto formatValue = ::mlir::cutlass_compiler::cuda::symbolizeExecutableFormat(format);
  if (!formatValue)
    return {nullptr};
  return wrap(cuda::ExecutableAttr::get(ctx, *formatValue));
}
uint32_t mlirCudaExecutableAttrGetFormat(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<cuda::ExecutableAttr>(unwrap(attr)).getFormat());
}

bool mlirAttributeIsACudaFuncAttributes(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::FuncAttributesAttr>(unwrap(attr));
}
MlirAttribute mlirCudaFuncAttributesAttrGet(MlirContext context, MlirAttribute values) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  ::mlir::DictionaryAttr valuesValue;
  if (!unwrapAs(ctx, values, false, valuesValue))
    return {nullptr};
  return wrap(cuda::FuncAttributesAttr::get(ctx, valuesValue));
}
MlirAttribute mlirCudaFuncAttributesAttrGetValues(MlirAttribute attr) {
  return wrap(static_cast<mlir::Attribute>(llvm::cast<cuda::FuncAttributesAttr>(unwrap(attr)).getValues()));
}
