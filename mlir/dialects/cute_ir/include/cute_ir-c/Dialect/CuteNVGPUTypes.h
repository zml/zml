// C API for the CuTe NVIDIA GPU types, generated from the dialect's .td files.
#ifndef CUTE_IR_C_DIALECT_CUTENVGPU_TYPES_H
#define CUTE_IR_C_DIALECT_CUTENVGPU_TYPES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles, and std::optional enums a
// negative value. Enums are their integer values (the dialect's *Enums.td).
// Getters require a value accepted by the matching IsA.

// `!cute_nvgpu.atom.bulk_copy_g2s`: Bulk (TMA non-tensor) copy atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(
    MlirContext context, MlirType valType, int copyBits, bool mcast);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetMcast(MlirType type);

// `!cute_nvgpu.atom.bulk_copy_s2g`: Bulk (TMA non-tensor) copy atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(
    MlirContext context, MlirType valType, int copyBits, bool mask);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetMask(MlirType type);

// `!cute_nvgpu.atom.bulk_copy_s2s`: Bulk (TMA non-tensor) copy atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(
    MlirContext context, MlirType valType, int copyBits);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetCopyBits(MlirType type);

// `!cute_nvgpu.atom.dsmem_store`: Distributed shared-memory store atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomDsmemStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(
    MlirContext context, MlirType valType, int copyBits);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomDsmemStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomDsmemStoreTypeGetCopyBits(MlirType type);

// `!cute_nvgpu.atom.g2r`: Global-to-register load atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomG2R(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomG2RTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t memOrder,
    uint32_t memScope, uint32_t l2PrefetchSize, uint32_t l1CacheEvictPriority,
    uint32_t loadCacheMode, uint32_t sharedSpace, bool invariant);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomG2RTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int mlirCuteNVGPUCopyAtomG2RTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomG2RTypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomG2RTypeGetMemScope(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomG2RTypeGetL2PrefetchSize(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomG2RTypeGetL1CacheEvictPriority(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomG2RTypeGetLoadCacheMode(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomG2RTypeGetSharedSpace(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteNVGPUCopyAtomG2RTypeGetInvariant(MlirType type);

// `!cute_nvgpu.atom.im2col_tma_load`: Executable im2col TMA load atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(
    MlirContext context, MlirType valType, int copyBits, int numCta,
    MlirType gStride, bool mcast, MlirType tmaGbasis);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetGStride(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetMcast(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetTmaGbasis(MlirType type);

// `!cute_nvgpu.atom.im2col_tma_store`: Executable im2col TMA store atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(
    MlirContext context, MlirType valType, int copyBits, MlirType gStride,
    MlirType tmaGbasis);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetGStride(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetTmaGbasis(MlirType type);

// `!cute_nvgpu.atom.ldsm`: ldmatrix atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomLdsm(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomLdsmTypeGet(
    MlirContext context, MlirType valType, MlirAttribute mode,
    uint32_t szPattern, int numMatrices, bool transpose);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomLdsmTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomLdsmTypeGetMode(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomLdsmTypeGetSzPattern(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomLdsmTypeGetNumMatrices(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomLdsmTypeGetTranspose(MlirType type);

// `!cute_nvgpu.atom.non_exec_2d_gather4_tma_load`: Non-executable 2-D gather4
// TMA load atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetKind(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store`: Non-executable 2-D
// scatter4 TMA store atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExec2DScatter4TmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGet(MlirContext context,
                                                      MlirType valType,
                                                      int copyBits,
                                                      MlirType tmaGbasis,
                                                      int64_t tmaFormat);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.non_exec_im2col_tma_load`: Non-executable im2col TMA load
// atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetKind(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.non_exec_im2col_tma_store`: Non-executable im2col TMA store
// atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(
    MlirContext context, MlirType valType, int copyBits, MlirType tmaGbasis,
    int64_t tmaFormat);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.non_exec_tiled_tma_load`: Non-executable tiled TMA load
// atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetKind(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.non_exec_tiled_tma_reduce`: Non-executable tiled TMA reduce
// atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(
    MlirContext context, uint32_t kind, MlirType valType, int copyBits,
    MlirType tmaGbasis, int64_t tmaFormat);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetKind(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.non_exec_tiled_tma_store`: Non-executable tiled TMA store
// atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(
    MlirContext context, MlirType valType, int copyBits, MlirType tmaGbasis,
    int64_t tmaFormat);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED int64_t
mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaFormat(MlirType type);

// `!cute_nvgpu.atom.r2g`: Register-to-global store atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomR2G(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomR2GTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t memOrder,
    uint32_t memScope, uint32_t l1CacheEvictPriority, uint32_t storeCacheMode,
    uint32_t sharedSpace);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomR2GTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int mlirCuteNVGPUCopyAtomR2GTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2GTypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2GTypeGetMemScope(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2GTypeGetL1CacheEvictPriority(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2GTypeGetStoreCacheMode(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2GTypeGetSharedSpace(MlirType type);

// `!cute_nvgpu.atom.r2s`: Register-to-shared store atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomR2S(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomR2STypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t memOrder,
    uint32_t memScope, uint32_t sharedSpace);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomR2STypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int mlirCuteNVGPUCopyAtomR2STypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2STypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2STypeGetMemScope(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomR2STypeGetSharedSpace(MlirType type);

// `!cute_nvgpu.atom.s2r`: Shared-to-register load atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomS2R(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomS2RTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t memOrder,
    uint32_t memScope, uint32_t sharedSpace);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomS2RTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int mlirCuteNVGPUCopyAtomS2RTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomS2RTypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomS2RTypeGetMemScope(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomS2RTypeGetSharedSpace(MlirType type);

// `!cute_nvgpu.atom.simt_async_copy`: cp.async copy atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(
    MlirContext context, MlirType valType, uint32_t cache, int copyBits);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCache(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCopyBits(MlirType type);

// `!cute_nvgpu.atom.simt_multimem_ld_reduce`: multimem.ld_reduce atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t reduction,
    MlirAttribute ldReduceAccPrecision, uint32_t memOrder, uint32_t memScope);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetReduction(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetLdReduceAccPrecision(
    MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemScope(MlirType type);

// `!cute_nvgpu.atom.simt_multimem_red`: multimem.red atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t reduction,
    uint32_t memOrder, uint32_t memScope);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetReduction(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemScope(MlirType type);

// `!cute_nvgpu.atom.simt_multimem_st`: multimem.st atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t memOrder,
    uint32_t memScope);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemOrder(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemScope(MlirType type);

// `!cute_nvgpu.atom.universal_copy`: Universal (SIMT) copy atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t srcSpace,
    uint32_t dstSpace);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetSrcSpace(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetDstSpace(MlirType type);

// `!cute_nvgpu.atom.s2t_copy`: tcgen05.cp (shared to tensor memory) atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit, int numCta,
    uint32_t broadcast);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumDp(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumBit(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetBroadcast(MlirType type);

// `!cute_nvgpu.atom.sm100_s2t_copy_v2`: tcgen05.cp atom with the shared-memory
// major mode.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit, int numCta,
    uint32_t smemMajor, uint32_t broadcast);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumDp(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumBit(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetSmemMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetBroadcast(MlirType type);

// `!cute_nvgpu.atom.tmem_load`: tcgen05.ld atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit,
    unsigned numRep, bool pack16b);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumDp(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumBit(MlirType type);
MLIR_CAPI_EXPORTED unsigned
mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumRep(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPack16b(MlirType type);

// `!cute_nvgpu.atom.tmem_store`: tcgen05.st atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit,
    unsigned numRep, bool expand16b);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumDp(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumBit(MlirType type);
MLIR_CAPI_EXPORTED unsigned
mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumRep(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetExpand16b(MlirType type);

// `!cute_nvgpu.atom.tmem_load_spcompress`: SM107 tensor-memory load with sparse
// compression.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM107TmemLoadSPCompress(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit,
    unsigned numRep, uint32_t redOp, bool red, bool nan);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumDp(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumBit(MlirType type);
MLIR_CAPI_EXPORTED unsigned
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumRep(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRedOp(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRed(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNan(MlirType type);

// `!cute_nvgpu.atom.tmem_load_red`: tcgen05.ld.red atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(
    MlirContext context, MlirType valType, int numDp, int numBit,
    unsigned numRep, uint32_t redOp, bool nan, MlirAttribute halfSplitOff);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumDp(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumBit(MlirType type);
MLIR_CAPI_EXPORTED unsigned
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumRep(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetRedOp(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNan(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetHalfSplitOff(MlirType type);

// `!cute_nvgpu.atom.stsm`: stmatrix atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomStsm(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomStsmTypeGet(
    MlirContext context, MlirType valType, MlirAttribute mode, int numMatrices,
    bool transpose);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomStsmTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyAtomStsmTypeGetMode(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomStsmTypeGetNumMatrices(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomStsmTypeGetTranspose(MlirType type);

// `!cute_nvgpu.atom.tma_load`: Executable TMA load atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomTmaLoad(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomTmaLoadTypeGet(
    MlirContext context, MlirType valType, int sparsity, int copyBits,
    uint32_t mode, int numCta, MlirType gStride, bool mcast, MlirType tmaGbasis,
    bool override, bool noFullyOobTile);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaLoadTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomTmaLoadTypeGetSparsity(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomTmaLoadTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomTmaLoadTypeGetMode(MlirType type);
MLIR_CAPI_EXPORTED int mlirCuteNVGPUCopyAtomTmaLoadTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaLoadTypeGetGStride(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteNVGPUCopyAtomTmaLoadTypeGetMcast(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaLoadTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomTmaLoadTypeGetOverride(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomTmaLoadTypeGetNoFullyOobTile(MlirType type);

// `!cute_nvgpu.atom.tma_reduce`: Executable TMA reduce atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomTmaReduce(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomTmaReduceTypeGet(
    MlirContext context, MlirType valType, int copyBits, uint32_t mode,
    uint32_t kind, MlirType gStride, MlirType tmaGbasis);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaReduceTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomTmaReduceTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomTmaReduceTypeGetMode(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomTmaReduceTypeGetKind(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaReduceTypeGetGStride(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaReduceTypeGetTmaGbasis(MlirType type);

// `!cute_nvgpu.atom.tma_store`: Executable TMA store atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUCopyAtomTmaStore(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUCopyAtomTmaStoreTypeGet(
    MlirContext context, MlirType valType, int sparsity, int copyBits,
    uint32_t mode, MlirType gStride, MlirType tmaGbasis, bool override,
    bool noFullyOobTile);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaStoreTypeGetValType(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomTmaStoreTypeGetSparsity(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUCopyAtomTmaStoreTypeGetCopyBits(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyAtomTmaStoreTypeGetMode(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaStoreTypeGetGStride(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUCopyAtomTmaStoreTypeGetTmaGbasis(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomTmaStoreTypeGetOverride(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUCopyAtomTmaStoreTypeGetNoFullyOobTile(MlirType type);

// `!cute_nvgpu.sm100.mma_bs_sp`: SM100 block-scaled sparse UMMA MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int vecSize, uint32_t archPromote, MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSfType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSparseMetadataFormat(
    MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetVecSize(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetArchPromote(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm100.mma_bs`: SM100 block-scaled UMMA MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t aFragKind, int vecSize, uint32_t archPromote,
    MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetSfType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetVecSize(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetArchPromote(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm100.mma_sp`: SM100 sparse UMMA MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType eType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int cScaleExp, MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetEType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetSparseMetadataFormat(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCScaleExp(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm100.mma`: SM100 UMMA MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM100UMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM100UMMATypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    uint32_t aFragKind, int cScaleExp, MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMATypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMATypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMATypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMATypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMATypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMATypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM100UMMATypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM100UMMATypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM100UMMATypeGetCScaleExp(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM100UMMATypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm107.mma_bs_sp`: SM107 block-scaled sparse UMMA MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaledSparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int vecSize, uint32_t aCollectorOp, uint32_t bCollectorOp,
    MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSfType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSparseMetadataFormat(
    MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetVecSize(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetACollectorOp(
    MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBCollectorOp(
    MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm107.mma_bs`: SM107 block-scaled UMMA MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaled(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType sfType, uint32_t aFragKind, int vecSize, uint32_t aCollectorOp,
    uint32_t bCollectorOp, MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetSfType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetVecSize(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetACollectorOp(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBCollectorOp(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm107.mma_sp`: SM107 sparse UMMA MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM107UMMASparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    MlirType eType, uint32_t sparseMetadataFormat, uint32_t aFragKind,
    int cScaleExp, uint32_t aCollectorOp, uint32_t bCollectorOp,
    MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetEType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetSparseMetadataFormat(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCScaleExp(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetACollectorOp(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBCollectorOp(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm107.mma`: SM107 UMMA MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM107UMMA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM107UMMATypeGet(
    MlirContext context, MlirAttribute shapeMnk, int numCta, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    uint32_t aFragKind, int cScaleExp, uint32_t aCollectorOp,
    uint32_t bCollectorOp, MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMATypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMATypeGetNumCta(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMATypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMATypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMATypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMATypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM107UMMATypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMATypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM107UMMATypeGetCScaleExp(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMATypeGetACollectorOp(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM107UMMATypeGetBCollectorOp(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM107UMMATypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.SM120.mma_bs`: SM120 block-scaled mma.sync MMA atom.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(
    MlirContext context, MlirAttribute shapeMnk, int vecSize, MlirType aType,
    MlirType bType, MlirType cType, MlirType sfType, bool useSfLayoutTV);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetVecSize(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetSfType(MlirType type);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetUseSfLayoutTV(MlirType type);

// `!cute_nvgpu.sm80.sparse_mma`: SM80 sparse mma.sync MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM80Sparse(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM80SparseTypeGet(
    MlirContext context, MlirAttribute shapeMnk, MlirType aType, MlirType bType,
    MlirType cType, uint32_t sparseMetadataFormat, MlirAttribute intOverflow);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80SparseTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM80SparseTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM80SparseTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM80SparseTypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM80SparseTypeGetSparseMetadataFormat(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80SparseTypeGetIntOverflow(MlirType type);

// `!cute_nvgpu.sm80.mma`: SM80 mma.sync MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM80(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM80TypeGet(
    MlirContext context, MlirAttribute shapeMnk, MlirType aType, MlirType bType,
    MlirType cType, MlirAttribute intOverflow, MlirAttribute binaryOp);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80TypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM80TypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM80TypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM80TypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80TypeGetIntOverflow(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM80TypeGetBinaryOp(MlirType type);

// `!cute_nvgpu.sm89.mma`: SM89 fp8 mma.sync MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM89(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUMmaAtomSM89TypeGet(MlirContext context, MlirAttribute shapeMnk,
                                MlirType aType, MlirType bType, MlirType cType);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM89TypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM89TypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM89TypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM89TypeGetCType(MlirType type);

// `!cute_nvgpu.sm90.mma`: SM90 wgmma MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUMmaAtomSM90(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM90TypeGet(
    MlirContext context, MlirAttribute shapeMnk, uint32_t aMajor,
    uint32_t bMajor, MlirType aType, MlirType bType, MlirType cType,
    uint32_t aFragKind, MlirAttribute intOverflow, bool aNeg, bool bNeg);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM90TypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM90TypeGetAMajor(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM90TypeGetBMajor(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM90TypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM90TypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUMmaAtomSM90TypeGetCType(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaAtomSM90TypeGetAFragKind(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaAtomSM90TypeGetIntOverflow(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteNVGPUMmaAtomSM90TypeGetANeg(MlirType type);
MLIR_CAPI_EXPORTED bool mlirCuteNVGPUMmaAtomSM90TypeGetBNeg(MlirType type);

// `!cute_nvgpu.sm103.smem_desc_circular`: Circular shared-memory descriptor
// over a block layout in bytes.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUSmemDescCircularSM103(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescCircularSM103TypeGet(
    MlirContext context, MlirAttribute blockLayoutBytes);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSmemDescCircularSM103TypeGetBlockLayoutBytes(MlirType type);

// `!cute_nvgpu.sm107.smem_desc_circular`: Circular shared-memory descriptor
// over a block layout in bytes.
MLIR_CAPI_EXPORTED bool
mlirTypeIsACuteNVGPUSmemDescCircularSM107(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescCircularSM107TypeGet(
    MlirContext context, MlirAttribute blockLayoutBytes);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSmemDescCircularSM107TypeGetBlockLayoutBytes(MlirType type);

// `!cute_nvgpu.sm107.smem_desc`: SM107 shared-memory descriptor.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUSmemDescSM107(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSmemDescSM107TypeGet(MlirContext context);

// `!cute_nvgpu.smem_desc`: UMMA/GMMA shared-memory descriptor.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUSmemDesc(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescTypeGet(MlirContext context);

// `!cute_nvgpu.smem_desc_view`: Shared-memory descriptor iterator with a
// layout.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUSmemDescView(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescViewTypeGet(
    MlirContext context, MlirType desc, MlirAttribute layout);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUSmemDescViewTypeGetDesc(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSmemDescViewTypeGetLayout(MlirType type);

// `!cute_nvgpu.tma_descriptor_im2col`: Im2col TMA descriptor.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUTmaDescriptorIm2Col(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUTmaDescriptorIm2ColTypeGet(MlirContext context);

// `!cute_nvgpu.tma_descriptor_tiled`: Tiled TMA descriptor.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUTmaDescriptorTiled(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUTmaDescriptorTiledTypeGet(MlirContext context);

// `!cute_nvgpu.atom.universal_fma`: Universal FMA MMA atom.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUUniversalFmaAtom(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCuteNVGPUUniversalFmaAtomTypeGet(
    MlirContext context, MlirAttribute shapeMnk, MlirType aType, MlirType bType,
    MlirType cType);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUUniversalFmaAtomTypeGetShapeMnk(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUUniversalFmaAtomTypeGetAType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUUniversalFmaAtomTypeGetBType(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUUniversalFmaAtomTypeGetCType(MlirType type);

// `!cute_nvgpu.workid_response`: Cluster launch control work-id response.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteNVGPUWorkIdResponse(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUWorkIdResponseTypeGet(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTENVGPU_TYPES_H
