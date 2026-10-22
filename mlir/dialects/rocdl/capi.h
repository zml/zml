//===- capi.h - C API for the rocdl dialect -----------------------*- C -*-===//

#ifndef MLIR_DIALECTS_ROCDL_CAPI_H
#define MLIR_DIALECTS_ROCDL_CAPI_H

#include "mlir-c/Dialect/ROCDL.h"

#ifdef __cplusplus
extern "C" {
#endif

// Introspection

MLIR_CAPI_EXPORTED intptr_t mlirROCDLGetOperationCount(MlirContext context);

// Attributes

MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDLBufferOOBMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLBufferOOBModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLBufferOOBModeAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLBufferOOBModeModuleFlag(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLBufferOOBModeModuleFlagAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLBufferOOBModeModuleFlagAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLGfx12AtomicCachePolicy(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLGfx12AtomicCachePolicyAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLGfx12AtomicCachePolicyAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLGfx12CachePolicy(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLGfx12CachePolicyAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLGfx12CachePolicyAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLGfx942CachePolicy(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLGfx942CachePolicyAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLGfx942CachePolicyAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLMFMANegModifier(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLMFMANegModifierAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLMFMANegModifierAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDLMFMAPermB(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirROCDLMFMAPermBAttrGet(MlirContext context,
                                                           uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirROCDLMFMAPermBAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDLMatrixFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLMatrixFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLMatrixFormatAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLPreGfx12CachePolicy(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLPreGfx12CachePolicyAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLPreGfx12CachePolicyAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDLSchedGroupMask(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLSchedGroupMaskAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLSchedGroupMaskAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLTBufferOOBModeModuleFlag(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLTBufferOOBModeModuleFlagAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLTBufferOOBModeModuleFlagAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDLWMMACModifier(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLWMMACModifierAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLWMMACModifierAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLWMMAMatrixScale(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLWMMAMatrixScaleAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLWMMAMatrixScaleAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool
mlirAttributeIsAROCDLWMMAMatrixScaleFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLWMMAMatrixScaleFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirROCDLWMMAMatrixScaleFormatAttrGetValue(MlirAttribute attr);

// Target

MLIR_CAPI_EXPORTED bool mlirAttributeIsAROCDLTarget(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirROCDLTargetAttrGet(
    MlirContext context, int optLevel, MlirStringRef triple, MlirStringRef chip,
    MlirStringRef features, MlirStringRef abi, MlirAttribute flags,
    MlirAttribute link);
MLIR_CAPI_EXPORTED int mlirROCDLTargetAttrGetOptLevel(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirROCDLTargetAttrGetTriple(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirROCDLTargetAttrGetChip(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirROCDLTargetAttrGetFeatures(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirROCDLTargetAttrGetAbi(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirROCDLTargetAttrGetFlags(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirROCDLTargetAttrGetLink(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasFlag(MlirAttribute attr,
                                                   MlirStringRef flag);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasWave64(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasFastMath(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasDaz(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasFiniteOnly(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasUnsafeMath(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirROCDLTargetAttrHasCorrectSqrt(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_ROCDL_CAPI_H
