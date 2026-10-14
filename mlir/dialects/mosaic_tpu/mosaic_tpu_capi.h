// C entry points for the `tpu` dialect that Mosaic's C API
// (`tpu_dialect.h`) leaves out: the semaphore types and the attributes
// kernels build.
#ifndef ZML_MLIR_DIALECTS_MOSAIC_TPU_CAPI_H
#define ZML_MLIR_DIALECTS_MOSAIC_TPU_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// !tpu.semaphore
MLIR_CAPI_EXPORTED bool mlirTypeIsATpuSemaphore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTpuSemaphoreTypeGet(MlirContext context);

// !tpu.dma_semaphore
MLIR_CAPI_EXPORTED bool mlirTypeIsATpuDMASemaphore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTpuDMASemaphoreTypeGet(MlirContext context);

// Enum attributes take and return the C++ enum's integer value (tpu_ops.td);
// `Get` returns a null attribute for a value outside the enum.

// #tpu.reduction_kind<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuReductionKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuReductionKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuReductionKindAttrGetValue(MlirAttribute attr);

// #tpu.contract_precision<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuContractPrecision(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuContractPrecisionAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuContractPrecisionAttrGetValue(MlirAttribute attr);

// #tpu.rounding_mode<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuRoundingMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuRoundingModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuRoundingModeAttrGetValue(MlirAttribute attr);

// #tpu.core_type<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuCoreType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuCoreTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuCoreTypeAttrGetValue(MlirAttribute attr);

// #tpu.dimension_semantics<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuDimensionSemantics(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuDimensionSemanticsAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuDimensionSemanticsAttrGetValue(MlirAttribute attr);

// #tpu.pipeline_mode<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuPipelineMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuPipelineModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuPipelineModeAttrGetValue(MlirAttribute attr);

// #tpu.revisit_mode<...>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuRevisitMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuRevisitModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirTpuRevisitModeAttrGetValue(MlirAttribute attr);

// #tpu.memory_space<value[, core_type]>: a negative `coreType` leaves the
// optional core type out, and `GetCoreType` returns -1 when it is absent.
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuMemorySpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuMemorySpaceAttrGet(MlirContext context, uint32_t value, int64_t coreType);
MLIR_CAPI_EXPORTED uint32_t mlirTpuMemorySpaceAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuMemorySpaceAttrGetCoreType(MlirAttribute attr);

// #tpu.element_window<[pad_low], [pad_high]>
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuElementWindow(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuElementWindowAttrGet(MlirContext context, intptr_t numPadLow, const int64_t *padLow,
                                                            intptr_t numPadHigh, const int64_t *padHigh);
MLIR_CAPI_EXPORTED intptr_t mlirTpuElementWindowAttrGetNumPadLow(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuElementWindowAttrGetPadLow(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuElementWindowAttrGetNumPadHigh(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuElementWindowAttrGetPadHigh(MlirAttribute attr, intptr_t pos);

// #tpu.dot_dimension_numbers<...>: the dimension lists in ODS order.
MLIR_CAPI_EXPORTED bool mlirAttributeIsATpuDotDimensionNumbers(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirTpuDotDimensionNumbersAttrGet(
    MlirContext context, intptr_t numLhsContractingDims, const int64_t *lhsContractingDims,
    intptr_t numRhsContractingDims, const int64_t *rhsContractingDims, intptr_t numLhsNonContractingDims,
    const int64_t *lhsNonContractingDims, intptr_t numRhsNonContractingDims, const int64_t *rhsNonContractingDims,
    intptr_t numOutputDimOrder, const int64_t *outputDimOrder, intptr_t numLhsBatchDims, const int64_t *lhsBatchDims,
    intptr_t numRhsBatchDims, const int64_t *rhsBatchDims);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumLhsContractingDims(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetLhsContractingDims(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumRhsContractingDims(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetRhsContractingDims(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumLhsNonContractingDims(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetLhsNonContractingDims(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumRhsNonContractingDims(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetRhsNonContractingDims(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumOutputDimOrder(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetOutputDimOrder(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumLhsBatchDims(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetLhsBatchDims(MlirAttribute attr, intptr_t pos);
MLIR_CAPI_EXPORTED intptr_t mlirTpuDotDimensionNumbersAttrGetNumRhsBatchDims(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirTpuDotDimensionNumbersAttrGetRhsBatchDims(MlirAttribute attr, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_MOSAIC_TPU_CAPI_H
