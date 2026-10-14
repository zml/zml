// C API for the cuda types, generated from the dialect's .td files.
#include "cute_ir-c/Dialect/CudaTypes.h"

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
MlirType mlirCudaLaunchConfigTypeGet(MlirContext context, uint32_t maxNumAttrs) {
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

bool mlirTypeIsACudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreference(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueCooperative(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueCooperativeType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueCooperativeTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueCooperativeType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueMemSyncDomain(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueMemSyncDomainType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueMemSyncDomainTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueMemSyncDomainType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueNvlinkUtilCentricScheduling(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValuePortableClusterSizeMode(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValuePortableClusterSizeModeType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValuePortableClusterSizeModeTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValuePortableClusterSizeModeType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValuePriority(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValuePriorityType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValuePriorityTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValuePriorityType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowed(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemCarveout(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueSharedMemCarveoutType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueSharedMemCarveoutTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueSharedMemCarveoutType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueSharedMemoryMode(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueSharedMemoryModeType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueSharedMemoryModeTypeGet(MlirContext context) {
  auto *ctx = unwrap(context);
  if (!ctx)
    return {nullptr};
  ctx->getOrLoadDialect<cuda::CudaDialect>();
  return wrap(cuda::RuntimeLaunchAttributeValueSharedMemoryModeType::get(ctx));
}

bool mlirTypeIsACudaRuntimeLaunchAttributeValueSyncPolicy(MlirType type) {
  return llvm::isa_and_nonnull<cuda::RuntimeLaunchAttributeValueSyncPolicyType>(unwrap(type));
}
MlirType mlirCudaRuntimeLaunchAttributeValueSyncPolicyTypeGet(MlirContext context) {
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
