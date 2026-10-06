#include "mosaic_tpu_capi.h"

#include <cstdint>
#include <optional>

#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

namespace {

mlir::MLIRContext *loadTPU(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx) ctx->getOrLoadDialect<mlir::tpu::TPUDialect>();
  return ctx;
}

template <typename AttrT, typename EnumT>
MlirAttribute enumAttrGet(MlirContext context, std::optional<EnumT> value) {
  mlir::MLIRContext *ctx = loadTPU(context);
  if (!ctx || !value) return {nullptr};
  return wrap(AttrT::get(ctx, *value));
}

template <typename AttrT>
uint32_t enumAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<AttrT>(unwrap(attr)).getValue());
}

llvm::ArrayRef<int64_t> arrayRef(intptr_t size, const int64_t *data) {
  return llvm::ArrayRef<int64_t>(data, static_cast<size_t>(size));
}

}  // namespace

bool mlirTypeIsATpuSemaphore(MlirType type) { return llvm::isa<mlir::tpu::SemaphoreType>(unwrap(type)); }

MlirType mlirTpuSemaphoreTypeGet(MlirContext context) {
  mlir::MLIRContext *ctx = loadTPU(context);
  if (!ctx) return {nullptr};
  return wrap(mlir::tpu::SemaphoreType::get(ctx));
}

bool mlirTypeIsATpuDMASemaphore(MlirType type) { return llvm::isa<mlir::tpu::DMASemaphoreType>(unwrap(type)); }

MlirType mlirTpuDMASemaphoreTypeGet(MlirContext context) {
  mlir::MLIRContext *ctx = loadTPU(context);
  if (!ctx) return {nullptr};
  return wrap(mlir::tpu::DMASemaphoreType::get(ctx));
}

bool mlirAttributeIsATpuReductionKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::ReductionKindAttr>(unwrap(attr));
}

MlirAttribute mlirTpuReductionKindAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::ReductionKindAttr>(context, mlir::tpu::symbolizeReductionKind(value));
}

uint32_t mlirTpuReductionKindAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::ReductionKindAttr>(attr);
}

bool mlirAttributeIsATpuContractPrecision(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::ContractPrecisionAttr>(unwrap(attr));
}

MlirAttribute mlirTpuContractPrecisionAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::ContractPrecisionAttr>(context, mlir::tpu::symbolizeContractPrecision(value));
}

uint32_t mlirTpuContractPrecisionAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::ContractPrecisionAttr>(attr);
}

bool mlirAttributeIsATpuRoundingMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::RoundingModeAttr>(unwrap(attr));
}

MlirAttribute mlirTpuRoundingModeAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::RoundingModeAttr>(context, mlir::tpu::symbolizeRoundingMode(value));
}

uint32_t mlirTpuRoundingModeAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::RoundingModeAttr>(attr);
}

bool mlirAttributeIsATpuCoreType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::CoreTypeAttr>(unwrap(attr));
}

MlirAttribute mlirTpuCoreTypeAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::CoreTypeAttr>(context, mlir::tpu::symbolizeCoreType(value));
}

uint32_t mlirTpuCoreTypeAttrGetValue(MlirAttribute attr) { return enumAttrGetValue<mlir::tpu::CoreTypeAttr>(attr); }

bool mlirAttributeIsATpuDimensionSemantics(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::DimensionSemanticsAttr>(unwrap(attr));
}

MlirAttribute mlirTpuDimensionSemanticsAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::DimensionSemanticsAttr>(context, mlir::tpu::symbolizeDimensionSemantics(value));
}

uint32_t mlirTpuDimensionSemanticsAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::DimensionSemanticsAttr>(attr);
}

bool mlirAttributeIsATpuPipelineMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::PipelineModeAttr>(unwrap(attr));
}

MlirAttribute mlirTpuPipelineModeAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::PipelineModeAttr>(context, mlir::tpu::symbolizePipelineMode(value));
}

uint32_t mlirTpuPipelineModeAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::PipelineModeAttr>(attr);
}

bool mlirAttributeIsATpuRevisitMode(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::RevisitModeAttr>(unwrap(attr));
}

MlirAttribute mlirTpuRevisitModeAttrGet(MlirContext context, uint32_t value) {
  return enumAttrGet<mlir::tpu::RevisitModeAttr>(context, mlir::tpu::symbolizeRevisitMode(value));
}

uint32_t mlirTpuRevisitModeAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::RevisitModeAttr>(attr);
}

bool mlirAttributeIsATpuMemorySpace(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::MemorySpaceAttr>(unwrap(attr));
}

MlirAttribute mlirTpuMemorySpaceAttrGet(MlirContext context, uint32_t value, int64_t coreType) {
  mlir::MLIRContext *ctx = loadTPU(context);
  std::optional<mlir::tpu::MemorySpace> memorySpace = mlir::tpu::symbolizeMemorySpace(value);
  if (!ctx || !memorySpace) return {nullptr};
  std::optional<mlir::tpu::CoreType> core;
  if (coreType >= 0) {
    if (coreType > UINT32_MAX) return {nullptr};
    core = mlir::tpu::symbolizeCoreType(static_cast<uint32_t>(coreType));
    if (!core) return {nullptr};
  }
  return wrap(mlir::tpu::MemorySpaceAttr::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx, *memorySpace, core));
}

uint32_t mlirTpuMemorySpaceAttrGetValue(MlirAttribute attr) {
  return enumAttrGetValue<mlir::tpu::MemorySpaceAttr>(attr);
}

int64_t mlirTpuMemorySpaceAttrGetCoreType(MlirAttribute attr) {
  std::optional<mlir::tpu::CoreType> core = llvm::cast<mlir::tpu::MemorySpaceAttr>(unwrap(attr)).getCoreType();
  return core ? static_cast<int64_t>(*core) : -1;
}

bool mlirAttributeIsATpuElementWindow(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::ElementWindowAttr>(unwrap(attr));
}

MlirAttribute mlirTpuElementWindowAttrGet(MlirContext context, intptr_t numPadLow, const int64_t *padLow,
                                         intptr_t numPadHigh, const int64_t *padHigh) {
  mlir::MLIRContext *ctx = loadTPU(context);
  if (!ctx) return {nullptr};
  return wrap(mlir::tpu::ElementWindowAttr::get(ctx, arrayRef(numPadLow, padLow), arrayRef(numPadHigh, padHigh)));
}

intptr_t mlirTpuElementWindowAttrGetNumPadLow(MlirAttribute attr) {
  return llvm::cast<mlir::tpu::ElementWindowAttr>(unwrap(attr)).getPadLow().size();
}

int64_t mlirTpuElementWindowAttrGetPadLow(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::tpu::ElementWindowAttr>(unwrap(attr)).getPadLow()[pos];
}

intptr_t mlirTpuElementWindowAttrGetNumPadHigh(MlirAttribute attr) {
  return llvm::cast<mlir::tpu::ElementWindowAttr>(unwrap(attr)).getPadHigh().size();
}

int64_t mlirTpuElementWindowAttrGetPadHigh(MlirAttribute attr, intptr_t pos) {
  return llvm::cast<mlir::tpu::ElementWindowAttr>(unwrap(attr)).getPadHigh()[pos];
}

bool mlirAttributeIsATpuDotDimensionNumbers(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::tpu::DotDimensionNumbersAttr>(unwrap(attr));
}

MlirAttribute mlirTpuDotDimensionNumbersAttrGet(
    MlirContext context, intptr_t numLhsContractingDims, const int64_t *lhsContractingDims,
    intptr_t numRhsContractingDims, const int64_t *rhsContractingDims, intptr_t numLhsNonContractingDims,
    const int64_t *lhsNonContractingDims, intptr_t numRhsNonContractingDims, const int64_t *rhsNonContractingDims,
    intptr_t numOutputDimOrder, const int64_t *outputDimOrder, intptr_t numLhsBatchDims, const int64_t *lhsBatchDims,
    intptr_t numRhsBatchDims, const int64_t *rhsBatchDims) {
  mlir::MLIRContext *ctx = loadTPU(context);
  if (!ctx) return {nullptr};
  return wrap(mlir::tpu::DotDimensionNumbersAttr::get(
      ctx, arrayRef(numLhsContractingDims, lhsContractingDims),
      arrayRef(numRhsContractingDims, rhsContractingDims), arrayRef(numLhsNonContractingDims, lhsNonContractingDims),
      arrayRef(numRhsNonContractingDims, rhsNonContractingDims), arrayRef(numOutputDimOrder, outputDimOrder),
      arrayRef(numLhsBatchDims, lhsBatchDims), arrayRef(numRhsBatchDims, rhsBatchDims)));
}

#define ZML_TPU_DOT_DIMS_GETTERS(Name)                                                                        \
  intptr_t mlirTpuDotDimensionNumbersAttrGetNum##Name(MlirAttribute attr) {                                    \
    return llvm::cast<mlir::tpu::DotDimensionNumbersAttr>(unwrap(attr)).get##Name().size();                   \
  }                                                                                                           \
  int64_t mlirTpuDotDimensionNumbersAttrGet##Name(MlirAttribute attr, intptr_t pos) {                          \
    return llvm::cast<mlir::tpu::DotDimensionNumbersAttr>(unwrap(attr)).get##Name()[pos];                     \
  }

ZML_TPU_DOT_DIMS_GETTERS(LhsContractingDims)
ZML_TPU_DOT_DIMS_GETTERS(RhsContractingDims)
ZML_TPU_DOT_DIMS_GETTERS(LhsNonContractingDims)
ZML_TPU_DOT_DIMS_GETTERS(RhsNonContractingDims)
ZML_TPU_DOT_DIMS_GETTERS(OutputDimOrder)
ZML_TPU_DOT_DIMS_GETTERS(LhsBatchDims)
ZML_TPU_DOT_DIMS_GETTERS(RhsBatchDims)

#undef ZML_TPU_DOT_DIMS_GETTERS
