// C API for the CuTe NVIDIA GPU attributes, generated from the dialect's .td
// files.
#ifndef CUTE_IR_C_DIALECT_CUTENVGPU_ATTRIBUTES_H
#define CUTE_IR_C_DIALECT_CUTENVGPU_ATTRIBUTES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles, and std::optional enums a
// negative value. Enums are their integer values (the dialect's *Enums.td).
// Getters require a value accepted by the matching IsA.

// `#cute_nvgpu.atom_copy_field_bulkg2s`: Fields stored in Bulk Load Copy Atom
// type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyG2S(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_bulks2g`: Fields stored in Bulk Store Copy Atom
// type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2G(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_bulks2s`: Fields stored in Bulk CTA to Cluster
// Copy Atom type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2S(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_dsmem_store`: Fields stored in
// CopyAtomDsmemStoreType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldDsmemStore(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_g2r`: Fields stored in CopyAtomG2RType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldLoadGlobal(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_2d_gather4_tma_load`: Fields stored in
// CopyAtomNonExec2DGather4TmaLoadType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoad(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGet(MlirContext context,
                                                         uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGetValue(
    MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_2d_scatter4_tma_store`: Fields stored
// in CopyAtomNonExec2DScatter4TmaStoreType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStore(
    MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGet(MlirContext context,
                                                           uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGetValue(
    MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_load`: Fields stored in
// CopyAtomNonExecIm2ColTmaLoadType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoad(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGet(MlirContext context,
                                                      uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_store`: Fields stored in
// CopyAtomNonExecIm2ColTmaStoreType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaStore(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGet(MlirContext context,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_tma_load`: Fields stored in
// CopyAtomNonExecTiledTmaLoadType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaLoad(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGet(MlirContext context,
                                                     uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_tma_reduce`: Fields stored in
// CopyAtomNonExecTiledTmaReduceType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaReduce(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGet(MlirContext context,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_non_exec_tma_store`: Fields stored in
// CopyAtomNonExecTiledTmaStoreType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaStore(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGet(MlirContext context,
                                                      uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_r2g`: Fields stored in CopyAtomR2GType.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldStoreGlobal(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_tmaload`: Fields stored in Tma Load Copy Atom
// type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldTmaLoad(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_tmareduce`: Fields stored in Tma Reduce Copy
// Atom type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldTmaReduce(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_copy_field_tmastore`: Fields stored in Tma Store Copy Atom
// type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomCopyFieldTmaStore(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm100`: Fields stored in MMA Atom types for
// SM100.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM100(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM100AttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM100AttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm100_block_scaled`: Fields stored in MMA Atom
// types for SM100 (block scaled).
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaled(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGet(MlirContext context,
                                                 uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm100_block_scaled_sparse`: Fields stored in MMA
// Atom types for SM100 (block scaled).
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaledSparse(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGet(MlirContext context,
                                                       uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm100_sparse`: Fields stored in sparse MMA Atom
// types for SM100.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM100Sparse(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm120_block_scaled`: Fields stored in MMA Atom
// types for SM120 (block scaled).
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM120BlockScaled(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGet(MlirContext context,
                                                 uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm80_sparse`: Fields stored in MMA Atom types for
// SM80.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM80Sparse(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.atom_mma_field_sm90`: Fields stored in MMA Atom types for SM90.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUAtomMmaFieldSM90(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUAtomMmaFieldSM90AttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUAtomMmaFieldSM90AttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.bin_op`: Binary operation for single-bit MMA operations.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVGPUBinaryOp(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUBinaryOpAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUBinaryOpAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.copy_s2t_broadcast_mode`: Broadcast modes for the different
// utccp instructions.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUCopyS2TBroadcast(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUCopyS2TBroadcastAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUCopyS2TBroadcastAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.gather_scatter_tma_load`: The various kinds of TMA loads in
// gather/scatter mode.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUGatherScatterTmaLoad(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUGatherScatterTmaLoadAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUGatherScatterTmaLoadAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.im2col_tma_load`: The various kinds of TMA loads in im2col mode.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUIm2ColTmaLoad(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUIm2ColTmaLoadAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUIm2ColTmaLoadAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.ld_reduce_acc_precision_kind`: multimem ld_reduce accumulation
// precision kind.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPULdReduceAccPrecisionKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPULdReduceAccPrecisionKindAttrGet(
    MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPULdReduceAccPrecisionKindAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.ldsm_sz_pattern`: LDSM's sz pattern, describing the bit size.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPULdsmSzPattern(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPULdsmSzPatternAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPULdsmSzPatternAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.load_cache_mode`: Cache modes for the load instructions.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPULoadCacheMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPULoadCacheModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPULoadCacheModeAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.mma_int_overflow`: MMA overflow options.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUMMAIntOverflow(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMMAIntOverflowAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMMAIntOverflowAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.major`: Major mode for MMA operations.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVGPUMajorMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMajorModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMajorModeAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.mma_collector_op`: Enums for the mma collector op.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUMmaCollectorOp(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaCollectorOpAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaCollectorOpAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.mma_frag_kind`: Enums for the mma frag type.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUMmaFragKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUMmaFragKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUMmaFragKindAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.not_implemented_frg`: Fragment of an atom that has none.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUNotImplementedFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUNotImplementedFrgAttrGet(MlirContext context);

// `#cute_nvgpu.tma_reduce_kind`: Op for the TMASTORE instruction.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUReductionKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUReductionKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUReductionKindAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.rmem_frg`: Register-memory fragment of an MMA operand.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVGPURmemFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPURmemFrgAttrGet(MlirContext context, MlirType valueType,
                            uint32_t operand, bool deriveElemType);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPURmemFrgAttrGetValueType(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPURmemFrgAttrGetOperand(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPURmemFrgAttrGetDeriveElemType(MlirAttribute attr);

// `#cute_nvgpu.arch.sm100.circular_smem_frg`: SM100 circular shared-memory
// fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM100CircularSmemFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUSM100CircularSmemFrgAttrGet(
    MlirContext context, uint32_t majorMode);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUSM100CircularSmemFrgAttrGetMajorMode(MlirAttribute attr);

// `#cute_nvgpu.arch.sm100.smem_frg`: SM100 shared-memory fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM100SmemFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSM100SmemFrgAttrGet(MlirContext context, uint32_t majorMode);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUSM100SmemFrgAttrGetMajorMode(MlirAttribute attr);

// `#cute_nvgpu.arch.sm100.tmem_e_frg`: SM100 tensor-memory sparse-metadata
// fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM100TmemEFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUSM100TmemEFrgAttrGet(
    MlirContext context, MlirType aType, MlirType eType);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSM100TmemEFrgAttrGetAType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSM100TmemEFrgAttrGetEType(MlirAttribute attr);

// `#cute_nvgpu.arch.sm100.tmem_frg`: SM100 tensor-memory fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM100TmemFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUSM100TmemFrgAttrGet(
    MlirContext context, MlirType dataType, MlirType storageType, int ctaGroup,
    uint32_t tmemAllocMode);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSM100TmemFrgAttrGetDataType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSM100TmemFrgAttrGetStorageType(MlirAttribute attr);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUSM100TmemFrgAttrGetCtaGroup(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUSM100TmemFrgAttrGetTmemAllocMode(MlirAttribute attr);

// `#cute_nvgpu.arch.sm100.tmem_sf_frg`: SM100 tensor-memory scale-factor
// fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM100TmemSfFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVGPUSM100TmemSfFrgAttrGet(
    MlirContext context, MlirType sfType, int sfVecSize, int ctaGroup,
    bool isSfa, uint32_t tmemAllocMode);
MLIR_CAPI_EXPORTED MlirType
mlirCuteNVGPUSM100TmemSfFrgAttrGetSfType(MlirAttribute attr);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUSM100TmemSfFrgAttrGetSfVecSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED int
mlirCuteNVGPUSM100TmemSfFrgAttrGetCtaGroup(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool
mlirCuteNVGPUSM100TmemSfFrgAttrGetIsSfa(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUSM100TmemSfFrgAttrGetTmemAllocMode(MlirAttribute attr);

// `#cute_nvgpu.arch.sm107.smem_frg`: SM107 shared-memory fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM107SmemFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSM107SmemFrgAttrGet(MlirContext context, uint32_t majorMode);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUSM107SmemFrgAttrGetMajorMode(MlirAttribute attr);

// `#cute_nvgpu.arch.sm90.smem_frg`: SM90 shared-memory fragment.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUSM90SmemFrg(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUSM90SmemFrgAttrGet(MlirContext context, uint32_t majorMode);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUSM90SmemFrgAttrGetMajorMode(MlirAttribute attr);

// `#cute_nvgpu.tiled_tma_load`: The various kinds of TMA loads in tiled mode.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUTiledTmaLoad(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTiledTmaLoadAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUTiledTmaLoadAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.tma_data_format`: Bits 74-71 of the TMA descriptor.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUTmaDataFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTmaDataFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUTmaDataFormatAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.tma_load_mode`: Modes for the TMA load instruction.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUTmaLoadMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTmaLoadModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUTmaLoadModeAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.tma_store_mode`: Modes for the TMA store instruction.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUTmaStoreMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTmaStoreModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUTmaStoreModeAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.tmem_alloc_mode`: Modes for the Tmem allocation.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUTmemAllocMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTmemAllocModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUTmemAllocModeAttrGetValue(MlirAttribute attr);

// `#cute_nvgpu.tmem_load_red_op`: Reduce operation for the Tmem load
// instruction.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACuteNVGPUTmemLoadRedOp(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCuteNVGPUTmemLoadRedOpAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirCuteNVGPUTmemLoadRedOpAttrGetValue(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTENVGPU_ATTRIBUTES_H
