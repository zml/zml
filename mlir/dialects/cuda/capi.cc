//===- capi.cc - C API for NVIDIA's cuda dialect --------------------------===//

#include "capi.h"

#include "mlir/dialects/cuda/CudaDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

// Registration

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Cuda, cuda,
                                      mlir::cutlass_compiler::cuda::CudaDialect)

// Attributes

namespace cuda = mlir::cutlass_compiler::cuda;

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
MlirAttribute mlirCudaComputeTargetAttrGet(MlirContext context,
                                           uint32_t representation,
                                           uint32_t portability,
                                           intptr_t numArchs,
                                           const uint32_t *archs) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  auto representationValue =
      ::mlir::cutlass_compiler::cuda::symbolizeExecutableRepresentation(
          representation);
  if (!representationValue)
    return {nullptr};
  auto portabilityValue =
      ::mlir::cutlass_compiler::cuda::symbolizeArchPortability(portability);
  if (!portabilityValue)
    return {nullptr};
  llvm::SmallVector<::mlir::cutlass_compiler::cuda::GpuArchitecture> archsValue;
  for (intptr_t i = 0; i < numArchs; ++i) {
    auto value =
        ::mlir::cutlass_compiler::cuda::symbolizeGpuArchitecture(archs[i]);
    if (!value)
      return {nullptr};
    archsValue.push_back(*value);
  }
  return wrap(cuda::ComputeTargetAttr::get(ctx, *representationValue,
                                           *portabilityValue, archsValue));
}
uint32_t mlirCudaComputeTargetAttrGetRepresentation(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getRepresentation());
}
uint32_t mlirCudaComputeTargetAttrGetPortability(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getPortability());
}
intptr_t mlirCudaComputeTargetAttrGetNumArchs(MlirAttribute attr) {
  return llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getArchs().size();
}
uint32_t mlirCudaComputeTargetAttrGetArchs(MlirAttribute attr, intptr_t pos) {
  return static_cast<uint32_t>(
      llvm::cast<cuda::ComputeTargetAttr>(unwrap(attr)).getArchs()[pos]);
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
MlirAttribute mlirCudaDeviceAttributesAttrGet(MlirContext context,
                                              MlirAttribute values) {
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
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cuda::DeviceAttributesAttr>(unwrap(attr)).getValues()));
}

bool mlirAttributeIsACudaExecutable(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::ExecutableAttr>(unwrap(attr));
}
MlirAttribute mlirCudaExecutableAttrGet(MlirContext context, uint32_t format) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  auto formatValue =
      ::mlir::cutlass_compiler::cuda::symbolizeExecutableFormat(format);
  if (!formatValue)
    return {nullptr};
  return wrap(cuda::ExecutableAttr::get(ctx, *formatValue));
}
uint32_t mlirCudaExecutableAttrGetFormat(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<cuda::ExecutableAttr>(unwrap(attr)).getFormat());
}

bool mlirAttributeIsACudaFuncAttributes(MlirAttribute attr) {
  return llvm::isa_and_nonnull<cuda::FuncAttributesAttr>(unwrap(attr));
}
MlirAttribute mlirCudaFuncAttributesAttrGet(MlirContext context,
                                            MlirAttribute values) {
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
  return wrap(static_cast<mlir::Attribute>(
      llvm::cast<cuda::FuncAttributesAttr>(unwrap(attr)).getValues()));
}

// Types

bool mlirTypeIsACudaArchSet(MlirType type) {
  return llvm::isa_and_nonnull<cuda::ArchSetType>(unwrap(type));
}
MlirType mlirCudaArchSetTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::ArchSetType::get(ctx));
}

bool mlirTypeIsACudaEvent(MlirType type) {
  return llvm::isa_and_nonnull<cuda::EventType>(unwrap(type));
}
MlirType mlirCudaEventTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::EventType::get(ctx));
}

bool mlirTypeIsACudaGraphDeviceNode(MlirType type) {
  return llvm::isa_and_nonnull<cuda::GraphDeviceNodeType>(unwrap(type));
}
MlirType mlirCudaGraphDeviceNodeTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::GraphDeviceNodeType::get(ctx));
}

bool mlirTypeIsACudaGraphExec(MlirType type) {
  return llvm::isa_and_nonnull<cuda::GraphExecType>(unwrap(type));
}
MlirType mlirCudaGraphExecTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::GraphExecType::get(ctx));
}

bool mlirTypeIsACudaGraphNode(MlirType type) {
  return llvm::isa_and_nonnull<cuda::GraphNodeType>(unwrap(type));
}
MlirType mlirCudaGraphNodeTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::GraphNodeType::get(ctx));
}

bool mlirTypeIsACudaGraph(MlirType type) {
  return llvm::isa_and_nonnull<cuda::GraphType>(unwrap(type));
}
MlirType mlirCudaGraphTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::GraphType::get(ctx));
}

bool mlirTypeIsACudaInt(MlirType type) {
  return llvm::isa_and_nonnull<cuda::IntType>(unwrap(type));
}
MlirType mlirCudaIntTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::IntType::get(ctx));
}

bool mlirTypeIsACudaKernel(MlirType type) {
  return llvm::isa_and_nonnull<cuda::KernelType>(unwrap(type));
}
MlirType mlirCudaKernelTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::KernelType::get(ctx));
}

bool mlirTypeIsACudaLaunchConfig(MlirType type) {
  return llvm::isa_and_nonnull<cuda::LaunchConfigType>(unwrap(type));
}
MlirType mlirCudaLaunchConfigTypeGet(MlirContext context,
                                     uint32_t maxNumAttrs) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::LaunchConfigType::get(ctx, maxNumAttrs));
}
uint32_t mlirCudaLaunchConfigTypeGetMaxNumAttrs(MlirType type) {
  return llvm::cast<cuda::LaunchConfigType>(unwrap(type)).getMaxNumAttrs();
}

bool mlirTypeIsACudaLibrary(MlirType type) {
  return llvm::isa_and_nonnull<cuda::LibraryType>(unwrap(type));
}
MlirType mlirCudaLibraryTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::LibraryType::get(ctx));
}

bool mlirTypeIsACudaModule(MlirType type) {
  return llvm::isa_and_nonnull<cuda::ModuleType>(unwrap(type));
}
MlirType mlirCudaModuleTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::ModuleType::get(ctx));
}

bool mlirTypeIsACudaResult(MlirType type) {
  return llvm::isa_and_nonnull<cuda::ResultType>(unwrap(type));
}
MlirType mlirCudaResultTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::ResultType::get(ctx));
}

bool mlirTypeIsACudaRuntimeDeviceProp(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeDevicePropType>(unwrap(type));
}
MlirType mlirCudaRuntimeDevicePropTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeDevicePropType::get(ctx));
}

bool mlirTypeIsACudaRuntimeDim3(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeDim3Type>(unwrap(type));
}
MlirType mlirCudaRuntimeDim3TypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeDim3Type::get(ctx));
}

bool mlirTypeIsACudaRuntimeExtent(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeExtentType>(unwrap(type));
}
MlirType mlirCudaRuntimeExtentTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeExtentType::get(ctx));
}

bool mlirTypeIsACudaRuntimeKernelNodeParams(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeKernelNodeParamsType>(unwrap(type));
}
MlirType mlirCudaRuntimeKernelNodeParamsTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeKernelNodeParamsType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreference(
    MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType>(
      unwrap(type));
}
MlirType
mlirCudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceTypeGet(
    MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(
      cuda::RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType::
          get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueCooperative(MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValueCooperativeType>(unwrap(type));
}
MlirType
mlirCudaRuntimeLaunchAttributeValueCooperativeTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueCooperativeType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueMemSyncDomain(MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValueMemSyncDomainType>(unwrap(type));
}
MlirType
mlirCudaRuntimeLaunchAttributeValueMemSyncDomainTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueMemSyncDomainType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueNvlinkUtilCentricScheduling(
    MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType>(
      unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingTypeGet(
    MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(
      cuda::RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType::get(
          ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValuePortableClusterSizeMode(
    MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValuePortableClusterSizeModeType>(
      unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValuePortableClusterSizeModeTypeGet(
    MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(
      cuda::RuntimeLaunchAttributeValuePortableClusterSizeModeType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValuePriority(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValuePriorityType>(
      unwrap(type));
}
MlirType
mlirCudaRuntimeLaunchAttributeValuePriorityTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValuePriorityType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowed(
    MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::
          RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType>(
      unwrap(type));
}
MlirType
mlirCudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedTypeGet(
    MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(
      cuda::
          RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType::
              get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemCarveout(
    MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValueSharedMemCarveoutType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueSharedMemCarveoutTypeGet(
    MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueSharedMemCarveoutType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemoryMode(MlirType type) {
  return llvm::isa_and_nonnull<
      cuda::RuntimeLaunchAttributeValueSharedMemoryModeType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueSharedMemoryModeTypeGet(
    MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueSharedMemoryModeType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueSyncPolicy(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueSyncPolicyType>(
      unwrap(type));
}
MlirType
mlirCudaRuntimeLaunchAttributeValueSyncPolicyTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueSyncPolicyType::get(ctx));
}

bool mlirTypeIsACudaRuntimeMemcpy3DParams(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeMemcpy3DParamsType>(unwrap(type));
}
MlirType mlirCudaRuntimeMemcpy3DParamsTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeMemcpy3DParamsType::get(ctx));
}

bool mlirTypeIsACudaRuntimePitchedPtr(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimePitchedPtrType>(unwrap(type));
}
MlirType mlirCudaRuntimePitchedPtrTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimePitchedPtrType::get(ctx));
}

bool mlirTypeIsACudaRuntimePos(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimePosType>(unwrap(type));
}
MlirType mlirCudaRuntimePosTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimePosType::get(ctx));
}

bool mlirTypeIsACudaRuntimeUuid(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeUuidType>(unwrap(type));
}
MlirType mlirCudaRuntimeUuidTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeUuidType::get(ctx));
}

bool mlirTypeIsACudaStream(MlirType type) {
  return llvm::isa_and_nonnull<cuda::StreamType>(unwrap(type));
}
MlirType mlirCudaStreamTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::StreamType::get(ctx));
}

bool mlirTypeIsACudaTensorMap(MlirType type) {
  return llvm::isa_and_nonnull<cuda::TensorMapType>(unwrap(type));
}
MlirType mlirCudaTensorMapTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::TensorMapType::get(ctx));
}
