// The cute_nvgpu dialect's enums, generated from the nvidia-cutlass-dsl 4.8.0
// enum bindings (`_cute_nvgpu_enum_gen.py`); do not edit.
//
// Case values and spellings are the compiler's. An enum the bindings build
// as a dialect attribute also has its EnumAttr, printed `#<mnemonic><case>`.
// The cute enums the nvgpu bindings repeat are the cute dialect's; their
// nvgpu attributes are in CuteNVGPUAttrs.td, so that -gen-enum-decls
// over this file emits the nvgpu enums only.

#ifndef CUTE_IR_GENERATED_CUTENVGPU_ENUMS_TD
#define CUTE_IR_GENERATED_CUTENVGPU_ENUMS_TD

include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.td"
include "mlir/IR/EnumAttr.td"

def CuteNVGPU_BinaryOpEnum : I32EnumAttr<"BinaryOp", "Binary operation for single-bit MMA operations", [
    I32EnumAttrCase<"none", 0, "none">,
    I32EnumAttrCase<"xor_popc", 1, "xor_popc">,
    I32EnumAttrCase<"and_popc", 2, "and_popc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldBulkCopyG2SEnum : I32EnumAttr<"AtomCopyFieldBulkCopyG2S", "Fields stored in Bulk Load Copy Atom type", [
    I32EnumAttrCase<"tma_bar", 1, "tma_bar">,
    I32EnumAttrCase<"mcast_mask", 2, "mcast_mask">,
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldBulkCopyS2GEnum : I32EnumAttr<"AtomCopyFieldBulkCopyS2G", "Fields stored in Bulk Store Copy Atom type", [
    I32EnumAttrCase<"byte_mask", 4, "byte_mask">,
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldBulkCopyS2SEnum : I32EnumAttr<"AtomCopyFieldBulkCopyS2S", "Fields stored in Bulk CTA to Cluster Copy Atom type", [
    I32EnumAttrCase<"tma_bar", 1, "tma_bar">,
    I32EnumAttrCase<"cta_rank", 5, "cta_rank">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldDsmemStoreEnum : I32EnumAttr<"AtomCopyFieldDsmemStore", "Fields stored in CopyAtomDsmemStoreType", [
    I32EnumAttrCase<"mbar_ptr", 8, "mbar_ptr">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldLoadGlobalEnum : I32EnumAttr<"AtomCopyFieldLoadGlobal", "Fields stored in CopyAtomG2RType", [
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExec2DGather4TmaLoadEnum : I32EnumAttr<"AtomCopyFieldNonExec2DGather4TmaLoad", "Fields stored in CopyAtomNonExec2DGather4TmaLoadType", [
    I32EnumAttrCase<"tma_desc", 3, "tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExec2DScatter4TmaStoreEnum : I32EnumAttr<"AtomCopyFieldNonExec2DScatter4TmaStore", "Fields stored in CopyAtomNonExec2DScatter4TmaStoreType", [
    I32EnumAttrCase<"tma_desc", 3, "tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExecIm2ColTmaLoadEnum : I32EnumAttr<"AtomCopyFieldNonExecIm2ColTmaLoad", "Fields stored in CopyAtomNonExecIm2ColTmaLoadType", [
    I32EnumAttrCase<"im2col_tma_desc", 6, "im2col_tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExecIm2ColTmaStoreEnum : I32EnumAttr<"AtomCopyFieldNonExecIm2ColTmaStore", "Fields stored in CopyAtomNonExecIm2ColTmaStoreType", [
    I32EnumAttrCase<"im2col_tma_desc", 6, "im2col_tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExecTiledTmaLoadEnum : I32EnumAttr<"AtomCopyFieldNonExecTiledTmaLoad", "Fields stored in CopyAtomNonExecTiledTmaLoadType", [
    I32EnumAttrCase<"tma_desc", 3, "tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExecTiledTmaReduceEnum : I32EnumAttr<"AtomCopyFieldNonExecTiledTmaReduce", "Fields stored in CopyAtomNonExecTiledTmaReduceType", [
    I32EnumAttrCase<"tma_desc", 3, "tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldNonExecTiledTmaStoreEnum : I32EnumAttr<"AtomCopyFieldNonExecTiledTmaStore", "Fields stored in CopyAtomNonExecTiledTmaStoreType", [
    I32EnumAttrCase<"tma_desc", 3, "tma_desc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldStoreGlobalEnum : I32EnumAttr<"AtomCopyFieldStoreGlobal", "Fields stored in CopyAtomR2GType", [
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldTmaLoadEnum : I32EnumAttr<"AtomCopyFieldTmaLoad", "Fields stored in Tma Load Copy Atom type", [
    I32EnumAttrCase<"tma_descriptor_ptr", 0, "tma_descriptor_ptr">,
    I32EnumAttrCase<"tma_bar", 1, "tma_bar">,
    I32EnumAttrCase<"mcast_mask", 2, "mcast_mask">,
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">,
    I32EnumAttrCase<"g_stride", 9, "g_stride">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldTmaReduceEnum : I32EnumAttr<"AtomCopyFieldTmaReduce", "Fields stored in Tma Reduce Copy Atom type", [
    I32EnumAttrCase<"tma_descriptor_ptr", 0, "tma_descriptor_ptr">,
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">,
    I32EnumAttrCase<"g_stride", 9, "g_stride">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomCopyFieldTmaStoreEnum : I32EnumAttr<"AtomCopyFieldTmaStore", "Fields stored in Tma Store Copy Atom type", [
    I32EnumAttrCase<"tma_descriptor_ptr", 0, "tma_descriptor_ptr">,
    I32EnumAttrCase<"cache_policy", 7, "cache_policy">,
    I32EnumAttrCase<"g_stride", 9, "g_stride">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM80SparseEnum : I32EnumAttr<"AtomMmaFieldSM80Sparse", "Fields stored in MMA Atom types for SM80", [
    I32EnumAttrCase<"e", 0, "e">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM90Enum : I32EnumAttr<"AtomMmaFieldSM90", "Fields stored in MMA Atom types for SM90", [
    I32EnumAttrCase<"accum_c", 0, "accum_c">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM100BlockScaledEnum : I32EnumAttr<"AtomMmaFieldSM100BlockScaled", "Fields stored in MMA Atom types for SM100 (block scaled)", [
    I32EnumAttrCase<"accum_c", 0, "accum_c">,
    I32EnumAttrCase<"neg_a", 1, "neg_a">,
    I32EnumAttrCase<"neg_b", 2, "neg_b">,
    I32EnumAttrCase<"sf_a", 3, "sf_a">,
    I32EnumAttrCase<"sf_b", 4, "sf_b">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM100BlockScaledSparseEnum : I32EnumAttr<"AtomMmaFieldSM100BlockScaledSparse", "Fields stored in MMA Atom types for SM100 (block scaled)", [
    I32EnumAttrCase<"accum_c", 0, "accum_c">,
    I32EnumAttrCase<"neg_a", 1, "neg_a">,
    I32EnumAttrCase<"neg_b", 2, "neg_b">,
    I32EnumAttrCase<"sf_a", 3, "sf_a">,
    I32EnumAttrCase<"sf_b", 4, "sf_b">,
    I32EnumAttrCase<"e", 5, "e">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM100Enum : I32EnumAttr<"AtomMmaFieldSM100", "Fields stored in MMA Atom types for SM100", [
    I32EnumAttrCase<"accum_c", 0, "accum_c">,
    I32EnumAttrCase<"neg_a", 1, "neg_a">,
    I32EnumAttrCase<"neg_b", 2, "neg_b">,
    I32EnumAttrCase<"disable_output_lane", 6, "disable_output_lane">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM100SparseEnum : I32EnumAttr<"AtomMmaFieldSM100Sparse", "Fields stored in sparse MMA Atom types for SM100", [
    I32EnumAttrCase<"accum_c", 0, "accum_c">,
    I32EnumAttrCase<"neg_a", 1, "neg_a">,
    I32EnumAttrCase<"neg_b", 2, "neg_b">,
    I32EnumAttrCase<"e", 5, "e">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_AtomMmaFieldSM120BlockScaledEnum : I32EnumAttr<"AtomMmaFieldSM120BlockScaled", "Fields stored in MMA Atom types for SM120 (block scaled)", [
    I32EnumAttrCase<"sf_a", 0, "sf_a">,
    I32EnumAttrCase<"sf_b", 1, "sf_b">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_CacheEvictionPriorityEnum : I32EnumAttr<"CacheEvictionPriority", "Cute Cache Eviction Priority kind", [
    I32EnumAttrCase<"EVICT_NORMAL", 0, "evict_normal">,
    I32EnumAttrCase<"EVICT_FIRST", 1, "evict_first">,
    I32EnumAttrCase<"EVICT_LAST", 2, "evict_last">,
    I32EnumAttrCase<"EVICT_UNCHANGED", 3, "evict_unchanged">,
    I32EnumAttrCase<"NO_ALLOCATE", 4, "no_allocate">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_CopyS2TBroadcastEnum : I32EnumAttr<"CopyS2TBroadcast", "Broadcast modes for the different utccp instructions", [
    I32EnumAttrCase<"none", 0, "none">,
    I32EnumAttrCase<"lw_0213", 1, "lw_0213">,
    I32EnumAttrCase<"lw_0123", 2, "lw_0123">,
    I32EnumAttrCase<"x4", 3, "x4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_GatherScatterTmaLoadEnum : I32EnumAttr<"GatherScatterTmaLoad", "The various kinds of TMA loads in gather/scatter mode", [
    I32EnumAttrCase<"sm_100", 0, "sm_100">,
    I32EnumAttrCase<"sm_100_multicast", 1, "sm_100_multicast">,
    I32EnumAttrCase<"sm_100_2sm", 2, "sm_100_2sm">,
    I32EnumAttrCase<"sm_100_2sm_multicast", 3, "sm_100_2sm_multicast">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_ArchEnum : I32EnumAttr<"Arch", "NVGpu arch number like sm_103", [
    I32EnumAttrCase<"sm_52", 520, "sm_52">,
    I32EnumAttrCase<"sm_70", 700, "sm_70">,
    I32EnumAttrCase<"sm_75", 750, "sm_75">,
    I32EnumAttrCase<"sm_80", 800, "sm_80">,
    I32EnumAttrCase<"sm_86", 860, "sm_86">,
    I32EnumAttrCase<"sm_90", 900, "sm_90">,
    I32EnumAttrCase<"sm_100", 1000, "sm_100">,
    I32EnumAttrCase<"sm_103", 1030, "sm_103">,
    I32EnumAttrCase<"sm_107", 1070, "sm_107">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_Im2ColTmaLoadEnum : I32EnumAttr<"Im2ColTmaLoad", "The various kinds of TMA loads in im2col mode", [
    I32EnumAttrCase<"sm_90", 0, "sm_90">,
    I32EnumAttrCase<"sm_90_multicast", 1, "sm_90_multicast">,
    I32EnumAttrCase<"sm_100_2sm", 2, "sm_100_2sm">,
    I32EnumAttrCase<"sm_100_2sm_multicast", 3, "sm_100_2sm_multicast">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_L2PrefetchSizeEnum : I32EnumAttr<"L2PrefetchSize", "CuTe L2 prefetch size options", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"RESERVED", 1, "reserved">,
    I32EnumAttrCase<"SIZE_64B", 2, "size_64b">,
    I32EnumAttrCase<"SIZE_128B", 3, "size_128b">,
    I32EnumAttrCase<"SIZE_256B", 4, "size_256b">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_LdReduceAccPrecisionKindEnum : I32EnumAttr<"LdReduceAccPrecisionKind", "multimem ld_reduce accumulation precision kind", [
    I32EnumAttrCase<"F16", 0, "F16">,
    I32EnumAttrCase<"F32", 1, "F32">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_LdsmSzPatternEnum : I32EnumAttr<"LdsmSzPattern", "LDSM's sz pattern, describing the bit size", [
    I32EnumAttrCase<"u16", 0, "u16">,
    I32EnumAttrCase<"u4to8", 1, "u4to8">,
    I32EnumAttrCase<"s4to8", 2, "s4to8">,
    I32EnumAttrCase<"u2to4", 3, "u2to4">,
    I32EnumAttrCase<"s2to4", 4, "s2to4">,
    I32EnumAttrCase<"u4x16p64to8", 5, "u4x16p64to8">,
    I32EnumAttrCase<"u6x16p32to8", 6, "u6x16p32to8">,
    I32EnumAttrCase<"u8", 7, "u8">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_LoadCacheModeEnum : I32EnumAttr<"LoadCacheMode", "Cache modes for the load instructions", [
    I32EnumAttrCase<"always", 0, "always">,
    I32EnumAttrCase<"global", 1, "global">,
    I32EnumAttrCase<"streaming", 2, "streaming">,
    I32EnumAttrCase<"last_use", 3, "last_use">,
    I32EnumAttrCase<"none", 4, "none">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_MMAIntOverflowEnum : I32EnumAttr<"MMAIntOverflow", "MMA overflow options", [
    I32EnumAttrCase<"satfinite", 1, "satfinite">,
    I32EnumAttrCase<"wrapped", 0, "wrapped">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_MemOrderKindEnum : I32EnumAttr<"MemOrderKind", "Memory Ordering kind", [
    I32EnumAttrCase<"WEAK", 0, "weak">,
    I32EnumAttrCase<"RELAXED", 1, "relaxed">,
    I32EnumAttrCase<"ACQUIRE", 2, "acquire">,
    I32EnumAttrCase<"RELEASE", 3, "release">,
    I32EnumAttrCase<"ACQ_REL", 4, "acq_rel">,
    I32EnumAttrCase<"SC", 5, "sc">,
    I32EnumAttrCase<"MMIO", 6, "mmio">,
    I32EnumAttrCase<"CONSTANT", 7, "constant">,
    I32EnumAttrCase<"VOLATILE", 8, "volatile">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_MemScopeKindEnum : I32EnumAttr<"MemScopeKind", "Cute Memory Scope kind", [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"CLUSTER", 1, "cluster">,
    I32EnumAttrCase<"GPU", 2, "gpu">,
    I32EnumAttrCase<"SYS", 3, "sys">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_MmaCollectorOpEnum : I32EnumAttr<"MmaCollectorOp", "Enums for the mma collector op", [
    I32EnumAttrCase<"discard", 0, "discard">,
    I32EnumAttrCase<"lastuse", 1, "lastuse">,
    I32EnumAttrCase<"fill", 2, "fill">,
    I32EnumAttrCase<"use", 3, "use">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_MmaFragKindEnum : I32EnumAttr<"MmaFragKind", "Enums for the mma frag type", [
    I32EnumAttrCase<"smem_desc", 0, "smem_desc">,
    I32EnumAttrCase<"sparse_smem_desc", 1, "sparse_smem_desc">,
    I32EnumAttrCase<"tmem", 2, "tmem">,
    I32EnumAttrCase<"tmem_ws", 3, "tmem_ws">,
    I32EnumAttrCase<"tmem_e", 4, "tmem_e">,
    I32EnumAttrCase<"tmem_sf", 5, "tmem_sf">,
    I32EnumAttrCase<"rmem", 6, "rmem">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_ReductionKindEnum : I32EnumAttr<"ReductionKind", "Op for the TMASTORE instruction", [
    I32EnumAttrCase<"ADD", 0, "ADD">,
    I32EnumAttrCase<"MIN", 1, "MIN">,
    I32EnumAttrCase<"MAX", 2, "MAX">,
    I32EnumAttrCase<"INC", 3, "INC">,
    I32EnumAttrCase<"DEC", 4, "DEC">,
    I32EnumAttrCase<"AND", 5, "AND">,
    I32EnumAttrCase<"OR", 6, "OR">,
    I32EnumAttrCase<"XOR", 7, "XOR">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_SharedSpaceEnum : I32EnumAttr<"SharedSpace", "CuTe Shared Space kind", [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"CLUSTER", 1, "cluster">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_SparseMetadataFormatEnum : I32EnumAttr<"SparseMetadataFormat", "Metadata format used in sparse MMA", [
    I32EnumAttrCase<"tid", 0, "tid">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_StoreCacheModeEnum : I32EnumAttr<"StoreCacheMode", "Cache modes for the store instructions", [
    I32EnumAttrCase<"write_back", 0, "write_back">,
    I32EnumAttrCase<"global", 1, "global">,
    I32EnumAttrCase<"streaming", 2, "streaming">,
    I32EnumAttrCase<"write_through", 3, "write_through">,
    I32EnumAttrCase<"none", 4, "none">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_TiledTmaLoadEnum : I32EnumAttr<"TiledTmaLoad", "The various kinds of TMA loads in tiled mode", [
    I32EnumAttrCase<"sm_90", 0, "sm_90">,
    I32EnumAttrCase<"sm_90_multicast", 1, "sm_90_multicast">,
    I32EnumAttrCase<"sm_100_2sm", 2, "sm_100_2sm">,
    I32EnumAttrCase<"sm_100_2sm_multicast", 3, "sm_100_2sm_multicast">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_TmaDataFormatEnum : I32EnumAttr<"TmaDataFormat", "Bits 74-71 of the TMA descriptor", [
    I32EnumAttrCase<"U8", 0, "U8">,
    I32EnumAttrCase<"U16", 1, "U16">,
    I32EnumAttrCase<"U32", 2, "U32">,
    I32EnumAttrCase<"S32", 3, "S32">,
    I32EnumAttrCase<"U64", 4, "U64">,
    I32EnumAttrCase<"S64", 5, "S64">,
    I32EnumAttrCase<"F16_RN", 6, "F16_RN">,
    I32EnumAttrCase<"F32_RN", 7, "F32_RN">,
    I32EnumAttrCase<"F32_FTZ_RN", 8, "F32_FTZ_RN">,
    I32EnumAttrCase<"F64_RN", 9, "F64_RN">,
    I32EnumAttrCase<"BF16_RN", 10, "BF16_RN">,
    I32EnumAttrCase<"U4", 11, "U4">,
    I32EnumAttrCase<"U4_UNPACK_U8", 12, "U4_UNPACK_U8">,
    I32EnumAttrCase<"U6_UNPACK_U8", 13, "U6_UNPACK_U8">,
    I32EnumAttrCase<"TF32_RN", 14, "TF32_RN">,
    I32EnumAttrCase<"TF32_FTZ_RN", 15, "TF32_FTZ_RN">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_TmaLoadModeEnum : I32EnumAttr<"TmaLoadMode", "Modes for the TMA load instruction", [
    I32EnumAttrCase<"tiled", 0, "tiled">,
    I32EnumAttrCase<"im2col", 1, "im2col">,
    I32EnumAttrCase<"w128", 2, "w128">,
    I32EnumAttrCase<"w", 3, "w">,
    I32EnumAttrCase<"gather4", 4, "gather4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_TmaStoreModeEnum : I32EnumAttr<"TmaStoreMode", "Modes for the TMA store instruction", [
    I32EnumAttrCase<"tiled", 0, "tiled">,
    I32EnumAttrCase<"im2col", 1, "im2col">,
    I32EnumAttrCase<"scatter4", 2, "scatter4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_TmemAllocModeEnum : I32EnumAttr<"TmemAllocMode", "Modes for the Tmem allocation", [
    I32EnumAttrCase<"Interleaved", 0, "Interleaved">,
    I32EnumAttrCase<"NonInterleaved", 1, "NonInterleaved">,
    I32EnumAttrCase<"Duplicated", 2, "Duplicated">,
    I32EnumAttrCase<"SfDuplicated_4x1", 3, "SfDuplicated_4x1">,
    I32EnumAttrCase<"SfDuplicated_2x2", 4, "SfDuplicated_2x2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_TmemLoadRedOpEnum : I32EnumAttr<"TmemLoadRedOp", "Reduce operation for the Tmem load instruction", [
    I32EnumAttrCase<"max", 0, "max">,
    I32EnumAttrCase<"maxabs", 1, "maxabs">,
    I32EnumAttrCase<"min", 2, "min">,
    I32EnumAttrCase<"minabs", 3, "minabs">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let genSpecializedAttr = 0;
}

def CuteNVGPU_CopyS2TBroadcastAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_CopyS2TBroadcastEnum, "copy_s2t_broadcast_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldBulkCopyG2SAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldBulkCopyG2SEnum, "atom_copy_field_bulkg2s"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldBulkCopyS2GAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldBulkCopyS2GEnum, "atom_copy_field_bulks2g"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldBulkCopyS2SAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldBulkCopyS2SEnum, "atom_copy_field_bulks2s"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldDsmemStoreAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldDsmemStoreEnum, "atom_copy_field_dsmem_store"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldLoadGlobalAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldLoadGlobalEnum, "atom_copy_field_g2r"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExec2DGather4TmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExec2DGather4TmaLoadEnum, "atom_copy_field_non_exec_2d_gather4_tma_load"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExec2DScatter4TmaStoreAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExec2DScatter4TmaStoreEnum, "atom_copy_field_non_exec_2d_scatter4_tma_store"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExecIm2ColTmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExecIm2ColTmaLoadEnum, "atom_copy_field_non_exec_im2col_tma_load"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExecIm2ColTmaStoreAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExecIm2ColTmaStoreEnum, "atom_copy_field_non_exec_im2col_tma_store"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExecTiledTmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExecTiledTmaLoadEnum, "atom_copy_field_non_exec_tma_load"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExecTiledTmaReduceAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExecTiledTmaReduceEnum, "atom_copy_field_non_exec_tma_reduce"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldNonExecTiledTmaStoreAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldNonExecTiledTmaStoreEnum, "atom_copy_field_non_exec_tma_store"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldStoreGlobalAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldStoreGlobalEnum, "atom_copy_field_r2g"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldTmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldTmaLoadEnum, "atom_copy_field_tmaload"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldTmaReduceAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldTmaReduceEnum, "atom_copy_field_tmareduce"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomCopyFieldTmaStoreAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomCopyFieldTmaStoreEnum, "atom_copy_field_tmastore"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM80SparseAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM80SparseEnum, "atom_mma_field_sm80_sparse"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM90Attr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM90Enum, "atom_mma_field_sm90"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM100Attr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM100Enum, "atom_mma_field_sm100"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM100BlockScaledAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM100BlockScaledEnum, "atom_mma_field_sm100_block_scaled"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM100BlockScaledSparseAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM100BlockScaledSparseEnum, "atom_mma_field_sm100_block_scaled_sparse"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM100SparseAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM100SparseEnum, "atom_mma_field_sm100_sparse"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_AtomMmaFieldSM120BlockScaledAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_AtomMmaFieldSM120BlockScaledEnum, "atom_mma_field_sm120_block_scaled"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_BinaryOpAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_BinaryOpEnum, "bin_op"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_GatherScatterTmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_GatherScatterTmaLoadEnum, "gather_scatter_tma_load"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_Im2ColTmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_Im2ColTmaLoadEnum, "im2col_tma_load"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_LdReduceAccPrecisionKindAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_LdReduceAccPrecisionKindEnum, "ld_reduce_acc_precision_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_LdsmSzPatternAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_LdsmSzPatternEnum, "ldsm_sz_pattern"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_LoadCacheModeAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_LoadCacheModeEnum, "load_cache_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_MMAIntOverflowAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_MMAIntOverflowEnum, "mma_int_overflow"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_MmaCollectorOpAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_MmaCollectorOpEnum, "mma_collector_op"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_MmaFragKindAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_MmaFragKindEnum, "mma_frag_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_ReductionKindAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_ReductionKindEnum, "tma_reduce_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_TiledTmaLoadAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_TiledTmaLoadEnum, "tiled_tma_load"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_TmaDataFormatAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_TmaDataFormatEnum, "tma_data_format"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_TmaLoadModeAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_TmaLoadModeEnum, "tma_load_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_TmaStoreModeAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_TmaStoreModeEnum, "tma_store_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_TmemAllocModeAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_TmemAllocModeEnum, "tmem_alloc_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

def CuteNVGPU_TmemLoadRedOpAttr : EnumAttr<CuteNVGPU_Dialect, CuteNVGPU_TmemLoadRedOpEnum, "tmem_load_red_op"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let assemblyFormat = "`<` $value `>`";
}

#endif // CUTE_IR_GENERATED_CUTENVGPU_ENUMS_TD
