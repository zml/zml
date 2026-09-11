// C API for CuTe NVIDIA GPU types.
#ifndef CUTE_IR_C_DIALECT_CUTENVGPU_TYPES_H
#define CUTE_IR_C_DIALECT_CUTENVGPU_TYPES_H
#include "mlir-c/IR.h"
#ifdef __cplusplus
extern "C" {
#endif

// Constructors return a null type for invalid parameters. All handles must
// belong to context. Getters require a type accepted by the matching IsA.

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomDsmemStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomDsmemStoreTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomG2R(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomG2RTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomG2RTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomLdsm(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomLdsmTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomLdsmTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomR2G(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomR2GTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomR2GTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomR2S(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomR2STypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomR2STypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomS2R(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomS2RTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomS2RTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomStsm(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomStsmTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomStsmTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaLoadTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomTmaLoadTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomTmaReduce(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomTmaReduceTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomTmaStoreTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM80Sparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80SparseTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM80(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM80TypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80TypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM89(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM89TypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM89TypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM90(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM90TypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM90TypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(MlirContext context,
                                                      MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMATypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUSmemDescCircularSM103(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescCircularSM103TypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSmemDescCircularSM103TypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUSmemDesc(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUSmemDescView(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSmemDescViewTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSmemDescViewTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUTiledCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUTiledCopyTypeGet(MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTiledCopyTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUTiledMma(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUTiledMmaTypeGet(MlirContext context,
                                                         MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTiledMmaTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUTmaDescriptorIm2Col(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUTmaDescriptorIm2ColTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUTmaDescriptorTiled(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUTmaDescriptorTiledTypeGet(MlirContext context);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUUniversalFmaAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUUniversalFmaAtomTypeGet(
    MlirContext context, MlirAttribute payload);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUUniversalFmaAtomTypeGetPayload(MlirType type);

MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUWorkIdResponse(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUWorkIdResponseTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTENVGPU_TYPES_H
