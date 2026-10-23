//===- capi.h - C API for the fly_rocdl dialect -------------------*- C -*-===//

#ifndef MLIR_DIALECTS_FLY_ROCDL_CAPI_H
#define MLIR_DIALECTS_FLY_ROCDL_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl);

MLIR_CAPI_EXPORTED MlirAttribute
mlirFlyROCDLBufferDescAddressAttrGet(MlirContext context);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(
    MlirContext context, int32_t bitSize, int32_t cacheModifier);

MLIR_CAPI_EXPORTED bool
mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopyLDS(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet(
    MlirContext context, int32_t bitSize);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaOpCDNA3MFMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaOpCDNA3MFMATypeGet(
    MlirContext context, int32_t m, int32_t n, int32_t k, MlirType elemTyA,
    MlirType elemTyB, MlirType elemTyAcc);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaOpGFX11WMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaOpGFX11WMMATypeGet(
    MlirContext context, int32_t m, int32_t n, int32_t k, MlirType elemTyA,
    MlirType elemTyB, MlirType elemTyAcc, bool signA, bool signB, bool clamp);

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaOpGFX120XWMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaOpGFX120XWMMATypeGet(
    MlirContext context, int32_t m, int32_t n, int32_t k, MlirType elemTyA,
    MlirType elemTyB, MlirType elemTyAcc, bool signA, bool signB, bool clamp);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_FLY_ROCDL_CAPI_H
