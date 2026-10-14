// C API for the nvvm attributes, generated from the dialect's .td files.
#ifndef CUTE_IR_C_DIALECT_NVVM_ATTRIBUTES_H
#define CUTE_IR_C_DIALECT_NVVM_ATTRIBUTES_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters or handles of another
// context; optional parameters take null handles. Enums are their integer
// values (the dialect's *Enums.td). Getters require a value accepted by the
// matching IsA.

// `#nvvm.atomic_op`: operations supported by atom instruction
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMAtomicOpKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMAtomicOpKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMAtomicOpKindAttrGetValue(MlirAttribute attr);

// `#nvvm.reduction`: NVVM barrier reduction operation
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMBarrierReduction(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMBarrierReductionAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMBarrierReductionAttrGetValue(MlirAttribute attr);

// `#nvvm.barrier_redux_kind`: NVVM barrier redux kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMBarrierReduxKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMBarrierReduxKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMBarrierReduxKindAttrGetValue(MlirAttribute attr);

// `#nvvm.block_scale_format`: MMA Block Scale Format
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMBlockScaleFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMBlockScaleFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMBlockScaleFormatAttrGetValue(MlirAttribute attr);

// `#nvvm.cta_group`: NVVM CTA group kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMCTAGroupKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMCTAGroupKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMCTAGroupKindAttrGetValue(MlirAttribute attr);

// `#nvvm.packfloat_type`: NVVM CVT Pack Float kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMCVTPackFloatKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMCVTPackFloatKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMCVTPackFloatKindAttrGetValue(MlirAttribute attr);

// `#nvvm.cache_eviction_priority`: NVVM Cache Eviction Priority
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMCacheEvictionPriority(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMCacheEvictionPriorityAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMCacheEvictionPriorityAttrGetValue(MlirAttribute attr);

// `#nvvm.cluster_launch_control_query_type`: NVVM ClusterLaunchControlQueryType
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMClusterLaunchControlQueryType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMClusterLaunchControlQueryTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMClusterLaunchControlQueryTypeAttrGetValue(MlirAttribute attr);

// `#nvvm.op`: Comparison operator encoding
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMCompareOpKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMCompareOpKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMCompareOpKindAttrGetValue(MlirAttribute attr);

// `#nvvm.convert_fp4_type`: NVVM ConvertFP4Type kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMConvertFP4Type(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMConvertFP4TypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMConvertFP4TypeAttrGetValue(MlirAttribute attr);

// `#nvvm.convert_fp8_type`: NVVM ConvertFP8Type kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMConvertFP8Type(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMConvertFP8TypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMConvertFP8TypeAttrGetValue(MlirAttribute attr);

// `#nvvm.convert_scale_kind`: NVVM ConvertScale kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMConvertScaleKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMConvertScaleKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMConvertScaleKindAttrGetValue(MlirAttribute attr);

// `#nvvm.dot_accumulate_type`: NVVM DotAccumulateType
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMDotAccumulateType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMDotAccumulateTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMDotAccumulateTypeAttrGetValue(MlirAttribute attr);

// `#nvvm.evict_kind`: NVVM L2 Prefetch Size
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMEvictKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMEvictKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMEvictKindAttrGetValue(MlirAttribute attr);

// `#nvvm.fp_rnd_mode`: NVVM FPRoundingMode kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMFPRoundingMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMFPRoundingModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMFPRoundingModeAttrGetValue(MlirAttribute attr);

// `#nvvm.grid_dep_action`: Action kind for grid dependency control
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMGridDepActionKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMGridDepActionKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMGridDepActionKindAttrGetValue(MlirAttribute attr);

// `#nvvm.int_rnd_mode`: NVVM IntegerRoundingMode kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMIntegerRoundingMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMIntegerRoundingModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMIntegerRoundingModeAttrGetValue(MlirAttribute attr);

// `#nvvm.l2_prefetch`: NVVM L2 Prefetch Size
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVML2PrefetchSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVML2PrefetchSizeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVML2PrefetchSizeAttrGetValue(MlirAttribute attr);

// `#nvvm.ld_st_matrix_elt_type`: Element type for ldmatrix and stmatrix
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMLdStMatrixEltType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMLdStMatrixEltTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMLdStMatrixEltTypeAttrGetValue(MlirAttribute attr);

// `#nvvm.load_cache_modifier_ext`: NVVM load cache modifier kind(Ext)
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMLoadCacheModifierExtKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMLoadCacheModifierExtKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMLoadCacheModifierExtKindAttrGetValue(MlirAttribute attr);

// `#nvvm.load_cache_modifier`: NVVM load cache modifier kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMLoadCacheModifierKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMLoadCacheModifierKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMLoadCacheModifierKindAttrGetValue(MlirAttribute attr);

// `#nvvm.load_shape`: shape attribute for ldmatrix
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMLoadShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMLoadShapeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMLoadShapeAttrGetValue(MlirAttribute attr);

// `#nvvm.load_src_format`: source format for ldmatrix
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMLoadSrcFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMLoadSrcFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMLoadSrcFormatAttrGetValue(MlirAttribute attr);

// `#nvvm.mbarrier_layout`: NVVM MBarrier Layout
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMBarrierLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMBarrierLayoutAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMBarrierLayoutAttrGetValue(MlirAttribute attr);

// `#nvvm.mbarrier_phase`: NVVM mbarrier phase type
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMBarrierPhase(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMBarrierPhaseAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMBarrierPhaseAttrGetValue(MlirAttribute attr);

// `#nvvm.mbar_scope`: NVVM MBarrier scope kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMBarrierScopeKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMBarrierScopeKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMBarrierScopeKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mbar_space`: NVVM MBarrier space kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMBarrierSpaceKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMBarrierSpaceKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMBarrierSpaceKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mbar_txn_kind`: NVVM MBarrier Transaction kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMBarrierTxnKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMBarrierTxnKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMBarrierTxnKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mbar_wait`: NVVM MBarrier wait kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMBarrierWaitKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMBarrierWaitKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMBarrierWaitKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_b1op`: MMA binary operations
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMAB1Op(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMAB1OpAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMAB1OpAttrGetValue(MlirAttribute attr);

// `#nvvm.block_scale_kind`: Block Scale Kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMABlockScaleKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMABlockScaleKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMABlockScaleKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_cta_count`: MMA CTA count
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMACtaCount(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMACtaCountAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMACtaCountAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_frag`: NVVM MMA frag type
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMAFrag(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMAFragAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMAFragAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_int_overflow`: MMA overflow options
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMAIntOverflow(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMAIntOverflowAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMAIntOverflowAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_kind`: MMA operation kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMAKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMAKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMAKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_layout`: NVVM MMA layout
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMALayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMALayoutAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMALayoutAttrGetValue(MlirAttribute attr);

// `#nvvm.mma_type`: NVVM MMA types
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMATypes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMATypesAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMMATypesAttrGetValue(MlirAttribute attr);

// `#nvvm.match_sync_kind`: NVVM match sync kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMatchSyncKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMatchSyncKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMatchSyncKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mem_order`: NVVM Memory Ordering kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMemOrderKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMemOrderKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMemOrderKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mem_scope`: NVVM Memory Scope kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMemScopeKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMemScopeKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMemScopeKindAttrGetValue(MlirAttribute attr);

// `#nvvm.mul_mode`: multiply mode attribute
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMulMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMulModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMMulModeAttrGetValue(MlirAttribute attr);

// `#nvvm.memory_space`: NVVM Memory Space
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMNVVMMemorySpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMNVVMMemorySpaceAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMNVVMMemorySpaceAttrGetValue(MlirAttribute attr);

// `#nvvm.permute_mode`: NVVM permute mode
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMPermuteMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMPermuteModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMPermuteModeAttrGetValue(MlirAttribute attr);

// `#nvvm.prefetch_cache_level`: NVVM Prefetch Cache Level
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMPrefetchCacheLevel(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMPrefetchCacheLevelAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMPrefetchCacheLevelAttrGetValue(MlirAttribute attr);

// `#nvvm.proxy_kind`: Proxy kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMProxyKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMProxyKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMProxyKindAttrGetValue(MlirAttribute attr);

// `#nvvm.reduction_kind`: NVVM Reduction Kind attribute
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMReductionKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMReductionKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMReductionKindAttrGetValue(MlirAttribute attr);

// `#nvvm.red_op`: Ops supported by red instruction
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMReductionOp(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMReductionOpAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMReductionOpAttrGetValue(MlirAttribute attr);

// `#nvvm.red_type`: types supported by red instruction
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMReductionType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMReductionTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMReductionTypeAttrGetValue(MlirAttribute attr);

// `#nvvm.spcomp_elem_size`: Sparse tensor compression element size
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPCompressElemSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPCompressElemSizeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPCompressElemSizeAttrGetValue(MlirAttribute attr);

// `#nvvm.spcomp_factor`: Sparse tensor compression factor type
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPCompressFactorType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPCompressFactorTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPCompressFactorTypeAttrGetValue(MlirAttribute attr);

// `#nvvm.spcomp_index_size`: Sparse tensor compression index size
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPCompressIndexSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPCompressIndexSizeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPCompressIndexSizeAttrGetValue(MlirAttribute attr);

// `#nvvm.spcomp_op_kind`: spcompress operation kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPCompressOpKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPCompressOpKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPCompressOpKindAttrGetValue(MlirAttribute attr);

// `#nvvm.spcomp_rep_factor`: Sparse tensor compression repetition factor
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPCompressRepFactor(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPCompressRepFactorAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPCompressRepFactorAttrGetValue(MlirAttribute attr);

// `#nvvm.spdecomp_elem_size`: Sparse tensor decompression element size
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPDecompressElemSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPDecompressElemSizeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPDecompressElemSizeAttrGetValue(MlirAttribute attr);

// `#nvvm.spdecomp_factor`: Sparse tensor decompression factor type
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPDecompressFactorType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPDecompressFactorTypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPDecompressFactorTypeAttrGetValue(MlirAttribute attr);

// `#nvvm.spdecomp_index_size`: Sparse tensor decompression index size
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPDecompressIndexSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPDecompressIndexSizeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPDecompressIndexSizeAttrGetValue(MlirAttribute attr);

// `#nvvm.spdecomp_rep_factor`: Sparse tensor decompression repetition factor
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSPDecompressRepFactor(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSPDecompressRepFactorAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSPDecompressRepFactorAttrGetValue(MlirAttribute attr);

// `#nvvm.sat_mode`: NVVM SaturationMode kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSaturationMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSaturationModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSaturationModeAttrGetValue(MlirAttribute attr);

// `#nvvm.sat`: NVVM SaturationMode kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSaturationModeKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSaturationModeKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSaturationModeKindAttrGetValue(MlirAttribute attr);

// `#nvvm.scale_vec_size`: MMA Scale Vector Sizes
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMScaleVecSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMScaleVecSizeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMScaleVecSizeAttrGetValue(MlirAttribute attr);

// `#nvvm.action`: NVVM set max register action
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSetMaxRegisterAction(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSetMaxRegisterActionAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSetMaxRegisterActionAttrGetValue(MlirAttribute attr);

// `#nvvm.shared_space`: Shared memory space
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSharedSpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSharedSpaceAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSharedSpaceAttrGetValue(MlirAttribute attr);

// `#nvvm.shfl_kind`: NVVM shuffle kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMShflKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMShflKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMShflKindAttrGetValue(MlirAttribute attr);

// `#nvvm.sparsity_format`: MMA Sparsity Format
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMSparsityFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMSparsityFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMSparsityFormatAttrGetValue(MlirAttribute attr);

// `#nvvm.state_space`: NVVM State Space
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMStateSpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMStateSpaceAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMStateSpaceAttrGetValue(MlirAttribute attr);

// `#nvvm.store_cache_modifier`: NVVM store cache modifier kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMStoreCacheModifierKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMStoreCacheModifierKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMStoreCacheModifierKindAttrGetValue(MlirAttribute attr);

// `#nvvm.store_shape`: shape attribute for stmatrix
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMStoreShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMStoreShapeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMStoreShapeAttrGetValue(MlirAttribute attr);

// `#nvvm.TCBarParam`: Cluster MMA Barrier Parameter Type
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTCBarParam(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTCBarParamAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTCBarParamAttrGetValue(MlirAttribute attr);

// `#nvvm.tma_load_mode`: NVVM TMA Load Mode
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTMALoadMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTMALoadModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTMALoadModeAttrGetValue(MlirAttribute attr);

// `#nvvm.tma_redux_kind`: NVVM TMA redux kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTMAReduxKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTMAReduxKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTMAReduxKindAttrGetValue(MlirAttribute attr);

// `#nvvm.tma_store_mode`: NVVM TMA Store Mode
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTMAStoreMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTMAStoreModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTMAStoreModeAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_cp_multicast`: tcgen05 cp multicast
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05CpMulticast(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05CpMulticastAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05CpMulticastAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_cp_shape`: tcgen05 cp shapes
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05CpShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05CpShapeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05CpShapeAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_cp_src_fmt`: tcgen05 cp source format
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05CpSrcFormat(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05CpSrcFormatAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05CpSrcFormatAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_fence`: NVVM Tcgen05 fence kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05FenceKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05FenceKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05FenceKindAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_ldst_shape`: allowed 32-bit signless integer cases: 0, 1, 2, 3, 4
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05LdStShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05LdStShapeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05LdStShapeAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_mma_block_scale`: tcgen05.mma block scale attribute
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05MMABlockScale(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05MMABlockScaleAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05MMABlockScaleAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_mma_collectorb`: tcgen05 MMA Collector Buffer B Attribute
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05MMACollectorBBuffer(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05MMACollectorBBufferAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05MMACollectorBBufferAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_mma_collectorop`: tcgen05.mma Collector Buffer Operation
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05MMACollectorOp(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05MMACollectorOpAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05MMACollectorOpAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_mma_kind`: tcgen05 MMA Supported Types
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05MMAKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05MMAKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05MMAKindAttrGetValue(MlirAttribute attr);

// `#nvvm.tcgen05_wait`: NVVM Tcgen05 wait kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTcgen05WaitKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTcgen05WaitKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTcgen05WaitKindAttrGetValue(MlirAttribute attr);

// `#nvvm.tensormap_elemtype`: NVVM Tensormap Elemtype
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTensormapElemtype(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTensormapElemtypeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTensormapElemtypeAttrGetValue(MlirAttribute attr);

// `#nvvm.tensormap_field`: NVVM Tensormap Field Kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTensormapField(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTensormapFieldAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTensormapFieldAttrGetValue(MlirAttribute attr);

// `#nvvm.tensormap_fill_mode`: NVVM Tensormap Fill Mode
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTensormapFillMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTensormapFillModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTensormapFillModeAttrGetValue(MlirAttribute attr);

// `#nvvm.tensormap_interleave_layout`: NVVM Tensormap Interleave Layout
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTensormapInterleaveLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTensormapInterleaveLayoutAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTensormapInterleaveLayoutAttrGetValue(MlirAttribute attr);

// `#nvvm.tensormap_swizzle_atomicity`: NVVM Tensormap Swizzle Atomicity
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTensormapSwizzleAtomicity(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTensormapSwizzleAtomicityAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTensormapSwizzleAtomicityAttrGetValue(MlirAttribute attr);

// `#nvvm.tensormap_swizzle_mode`: NVVM Tensormap Swizzle Mode
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTensormapSwizzleMode(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTensormapSwizzleModeAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTensormapSwizzleModeAttrGetValue(MlirAttribute attr);

// `#nvvm.TmemLayout`: Tensor Memory Layout Enumerated Type
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTmemLayout(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTmemLayoutAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMTmemLayoutAttrGetValue(MlirAttribute attr);

// `#nvvm.validate_pattern`: NVVM validate data pattern
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMValidatePattern(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMValidatePatternAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMValidatePatternAttrGetValue(MlirAttribute attr);

// `#nvvm.vote_sync_kind`: NVVM vote sync kind
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMVoteSyncKind(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMVoteSyncKindAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMVoteSyncKindAttrGetValue(MlirAttribute attr);

// `#nvvm.wgmma_scale_in`: WGMMA overflow options
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMWGMMAScaleIn(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMWGMMAScaleInAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMWGMMAScaleInAttrGetValue(MlirAttribute attr);

// `#nvvm.wgmma_scale_out`: WGMMA input predicate
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMWGMMAScaleOut(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMWGMMAScaleOutAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMWGMMAScaleOutAttrGetValue(MlirAttribute attr);

// `#nvvm.wgmma_type`: NVVM WGMMA types
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMWGMMATypes(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMWGMMATypesAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirCuteNVVMWGMMATypesAttrGetValue(MlirAttribute attr);

// `#nvvm.ld_st_matrix_shape`: Matrix shape of ldmatrix, stmatrix and movmatrix
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMLdStMatrixShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMLdStMatrixShapeAttrGet(MlirContext context, int m, int n);
MLIR_CAPI_EXPORTED int mlirCuteNVVMLdStMatrixShapeAttrGetM(MlirAttribute attr);
MLIR_CAPI_EXPORTED int mlirCuteNVVMLdStMatrixShapeAttrGetN(MlirAttribute attr);

// `#nvvm.shape`: Shape of an MMA operation
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMMMAShape(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMMMAShapeAttrGet(MlirContext context, int m, int n, int k);
MLIR_CAPI_EXPORTED int mlirCuteNVVMMMAShapeAttrGetM(MlirAttribute attr);
MLIR_CAPI_EXPORTED int mlirCuteNVVMMMAShapeAttrGetN(MlirAttribute attr);
MLIR_CAPI_EXPORTED int mlirCuteNVVMMMAShapeAttrGetK(MlirAttribute attr);

// `#nvvm.target`: GPU target of an NVVM module
MLIR_CAPI_EXPORTED bool mlirAttributeIsACuteNVVMTarget(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTargetAttrGet(MlirContext context, int O, MlirStringRef triple, MlirStringRef chip, MlirStringRef features, MlirAttribute flags, MlirAttribute link, bool verifyTarget);
MLIR_CAPI_EXPORTED int mlirCuteNVVMTargetAttrGetO(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteNVVMTargetAttrGetTriple(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteNVVMTargetAttrGetChip(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef mlirCuteNVVMTargetAttrGetFeatures(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTargetAttrGetFlags(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCuteNVVMTargetAttrGetLink(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirCuteNVVMTargetAttrGetVerifyTarget(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_NVVM_ATTRIBUTES_H
