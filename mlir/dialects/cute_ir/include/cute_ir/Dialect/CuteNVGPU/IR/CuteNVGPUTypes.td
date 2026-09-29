//===- CuteNVGPUTypes.td - CuTe NVIDIA GPU types -----------*- tablegen -*-===//
//
// Generated for nvidia-cutlass-dsl 4.8.0 from the DSL compiler
// (`_cutlass_ir.cu13` parser messages and verifier strings, its typed Python
// type classes, and probing its type parser and printer); do not edit by hand.
//
// Every cute_nvgpu type of the compiler is here, with its parameters in the
// compiler's order and C++ types. Parameters the compiler parses with custom
// directives (the `MxNxK` shape, `frag_kind`, `D DP`, `xR`, `t|n`, the sm90
// negation flags) are named after the typed getters or the directive.
//
// Syntax is the compiler's, byte for byte, except where its printer emits text
// its own parser rejects; the parsers here also accept that text, and the
// printers here then emit the form the compiler parses back to the same type:
//   - atom.universal_copy with copy_bits 0 and an address space prints `0 b`;
//   - atom.simt_multimem_* print a default field when a later one is not;
//   - sm100.mma_bs(_sp) print `arch_promote` when `int_overflow` is present.
//
// Verifiers carry the compiler's rules on parameter values where they are
// known, with its messages; instruction legality (the MMA shape and type
// tables, multimem PTX forms, ldmatrix/stmatrix configurations) is left to the
// compiler.
//
// Left out: the hidden `atom` attribute of the mma atoms' typed get()s. It
// overrides the atom's layouts, is never printed, and so cannot survive text;
// every mma atom parsed from text has it unset, which is what these types
// model.
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_TYPES_TD
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_TYPES_TD

include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.td"

class CuteNVGPU_CustomType<string name, string typeMnemonic>
    : CuteNVGPU_Type<name, typeMnemonic> {
  let hasCustomAssemblyFormat = 1;
}

defvar CuteNVGPU_Ty = "::mlir::Type";
defvar CuteNVGPU_Shape = "::mlir::cutlass_compiler::cute::ShapeAttr";
defvar CuteNVGPU_Layout = "::mlir::cutlass_compiler::cute::LayoutAttr";
defvar CuteNVGPU_LayoutTy = "::mlir::cutlass_compiler::cute::LayoutType";
defvar CuteNVGPU_StrideTy = "::mlir::cutlass_compiler::cute::StrideType";
defvar CuteNVGPU_Major = "::mlir::cutlass_compiler::cute::MajorMode";
defvar CuteNVGPU_Space = "::mlir::cutlass_compiler::cute::AddressSpace";

//===----------------------------------------------------------------------===//
// Descriptors
//===----------------------------------------------------------------------===//

def CuteNVGPU_SmemDescType : CuteNVGPU_Type<"SmemDesc", "smem_desc"> {
  let summary = "UMMA/GMMA shared-memory descriptor";
  let description = "`!cute_nvgpu.smem_desc`";
  let assemblyFormat = "";
}

def CuteNVGPU_SmemDescSM107Type
    : CuteNVGPU_Type<"SmemDescSM107", "sm107.smem_desc"> {
  let summary = "SM107 shared-memory descriptor";
  let description = "`!cute_nvgpu.sm107.smem_desc`";
  let assemblyFormat = "";
}

class CuteNVGPU_SmemDescCircular<string name, string typeMnemonic>
    : CuteNVGPU_Type<name, typeMnemonic> {
  let summary = "Circular shared-memory descriptor over a block layout in bytes";
  let description = [{
    The block layout is static, has three modes, a 128-byte mode 1 and is
    mode-1-major compact, with at least two blocks in mode 2:

    `!cute_nvgpu.}] # typeMnemonic # [{<<"(8,128,2):(128,1,1024)">>`
  }];
  let parameters = (ins CuteNVGPU_Layout:$block_layout_bytes);
  let assemblyFormat = "`<` $block_layout_bytes `>`";
  let genVerifyDecl = 1;
}

def CuteNVGPU_SmemDescCircularSM103Type
    : CuteNVGPU_SmemDescCircular<"SmemDescCircularSM103",
                                 "sm103.smem_desc_circular">;
def CuteNVGPU_SmemDescCircularSM107Type
    : CuteNVGPU_SmemDescCircular<"SmemDescCircularSM107",
                                 "sm107.smem_desc_circular">;

def CuteNVGPU_SmemDescViewType
    : CuteNVGPU_CustomType<"SmemDescView", "smem_desc_view"> {
  let summary = "Shared-memory descriptor iterator with a layout";
  let description = [{
    The descriptor is a smem_desc, sm107.smem_desc or smem_desc_circular type;
    the layout is a plain layout, as a string.

    `!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(2,3):(1,2)">`
  }];
  let parameters = (ins CuteNVGPU_Ty:$desc, CuteNVGPU_Layout:$layout);
  let genVerifyDecl = 1;
}

def CuteNVGPU_TmaDescriptorTiledType
    : CuteNVGPU_Type<"TmaDescriptorTiled", "tma_descriptor_tiled"> {
  let summary = "Tiled TMA descriptor";
  let description = "`!cute_nvgpu.tma_descriptor_tiled`";
  let assemblyFormat = "";
}

def CuteNVGPU_TmaDescriptorIm2ColType
    : CuteNVGPU_Type<"TmaDescriptorIm2Col", "tma_descriptor_im2col"> {
  let summary = "Im2col TMA descriptor";
  let description = "`!cute_nvgpu.tma_descriptor_im2col`";
  let assemblyFormat = "";
}

def CuteNVGPU_WorkIdResponseType
    : CuteNVGPU_Type<"WorkIdResponse", "workid_response"> {
  let summary = "Cluster launch control work-id response";
  let description = "`!cute_nvgpu.workid_response`";
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// SIMT copy atoms
//===----------------------------------------------------------------------===//

def CuteNVGPU_CopyAtomSIMTSyncCopyType
    : CuteNVGPU_CustomType<"CopyAtomSIMTSyncCopy", "atom.universal_copy"> {
  let summary = "Universal (SIMT) copy atom";
  let description = [{
    `copy_bits` 0 means the element width and is kept as 0; the address
    spaces default to generic. Each field needs the previous ones.

    `!cute_nvgpu.atom.universal_copy<f32, 128 b, src_space = gmem, dst_space = smem>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    DefaultValuedParameter<"int", "0">:$copy_bits,
    DefaultValuedParameter<CuteNVGPU_Space, CuteNVGPU_Space # "::generic">:$src_space,
    DefaultValuedParameter<CuteNVGPU_Space, CuteNVGPU_Space # "::generic">:$dst_space
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomSIMTAsyncCopyType
    : CuteNVGPU_CustomType<"CopyAtomSIMTAsyncCopy", "atom.simt_async_copy"> {
  let summary = "cp.async copy atom";
  let description = [{
    `copy_bits` absent or 0 parses as the element width and is always printed.

    `!cute_nvgpu.atom.simt_async_copy<bf16, cache = global, 128 b>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "LoadCacheMode":$cache,
    "int":$copy_bits
  );
}

def CuteNVGPU_CopyAtomDsmemStoreType
    : CuteNVGPU_CustomType<"CopyAtomDsmemStore", "atom.dsmem_store"> {
  let summary = "Distributed shared-memory store atom";
  let description = [{
    `copy_bits` absent or 0 parses as the element width and is always printed.

    `!cute_nvgpu.atom.dsmem_store<f32, 64 b>`
  }];
  let parameters = (ins CuteNVGPU_Ty:$val_type, "int":$copy_bits);
  let genVerifyDecl = 1;
}

class CuteNVGPU_BulkCopy<string name, string typeMnemonic, string flag,
                         string example>
    : CuteNVGPU_CustomType<name, typeMnemonic> {
  let summary = "Bulk (TMA non-tensor) copy atom";
  let description = [{
    Space-separated groups, printed when not default (`copy_bits` 0).

    `!cute_nvgpu.}] # typeMnemonic # "<" # example # ">`";
  let parameters = !con(
      (ins CuteNVGPU_Ty:$val_type, DefaultValuedParameter<"int", "0">:$copy_bits),
      !if(!empty(flag), (ins),
          !dag(ins, [DefaultValuedParameter<"bool", "false">], [flag])));
}

def CuteNVGPU_CopyAtomBulkCopyG2SType
    : CuteNVGPU_BulkCopy<"CopyAtomBulkCopyG2S", "atom.bulk_copy_g2s", "mcast",
                         "f32 copy_bits = 128 mcast = true">;
def CuteNVGPU_CopyAtomBulkCopyS2GType
    : CuteNVGPU_BulkCopy<"CopyAtomBulkCopyS2G", "atom.bulk_copy_s2g", "mask",
                         "f32 copy_bits = 128 mask = true">;
def CuteNVGPU_CopyAtomBulkCopyS2SType
    : CuteNVGPU_BulkCopy<"CopyAtomBulkCopyS2S", "atom.bulk_copy_s2s", "",
                         "f32 copy_bits = 128">;

// g2r, r2g, r2s, s2r: `<T copy_bits = N key = value ...>`, space separated,
// fields in parameter order and printed when not default; copy_bits absent or
// 0 parses as the element width and is always printed.

def CuteNVGPU_CopyAtomG2RType
    : CuteNVGPU_CustomType<"CopyAtomG2R", "atom.g2r"> {
  let summary = "Global-to-register load atom";
  let description = [{
    `!cute_nvgpu.atom.g2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu l2_prefetch_size = size_128b l1_cache_evict_priority = evict_last shared_space = cluster>`,
    `!cute_nvgpu.atom.g2r<f32 copy_bits = 64 invariant = true>` (invariant only with
    the default memory features)
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    DefaultValuedParameter<"MemOrderKind", "MemOrderKind::WEAK">:$mem_order,
    DefaultValuedParameter<"MemScopeKind", "MemScopeKind::CTA">:$mem_scope,
    DefaultValuedParameter<"L2PrefetchSize", "L2PrefetchSize::NONE">:$l2_prefetch_size,
    DefaultValuedParameter<"CacheEvictionPriority", "CacheEvictionPriority::EVICT_NORMAL">:$l1_cache_evict_priority,
    DefaultValuedParameter<"LoadCacheMode", "LoadCacheMode::always">:$load_cache_mode,
    DefaultValuedParameter<"SharedSpace", "SharedSpace::CTA">:$shared_space,
    DefaultValuedParameter<"bool", "false">:$invariant
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomR2GType
    : CuteNVGPU_CustomType<"CopyAtomR2G", "atom.r2g"> {
  let summary = "Register-to-global store atom";
  let description = [{
    `!cute_nvgpu.atom.r2g<f16 copy_bits = 128 l1_cache_evict_priority = no_allocate>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    DefaultValuedParameter<"MemOrderKind", "MemOrderKind::WEAK">:$mem_order,
    DefaultValuedParameter<"MemScopeKind", "MemScopeKind::CTA">:$mem_scope,
    DefaultValuedParameter<"CacheEvictionPriority", "CacheEvictionPriority::EVICT_NORMAL">:$l1_cache_evict_priority,
    DefaultValuedParameter<"StoreCacheMode", "StoreCacheMode::write_back">:$store_cache_mode,
    DefaultValuedParameter<"SharedSpace", "SharedSpace::CTA">:$shared_space
  );
  let genVerifyDecl = 1;
}

class CuteNVGPU_SharedCopy<string name, string typeMnemonic, string what>
    : CuteNVGPU_CustomType<name, typeMnemonic> {
  let summary = what # " atom";
  let description = [{
    `!cute_nvgpu.}] # typeMnemonic # [{<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    DefaultValuedParameter<"MemOrderKind", "MemOrderKind::WEAK">:$mem_order,
    DefaultValuedParameter<"MemScopeKind", "MemScopeKind::CTA">:$mem_scope,
    DefaultValuedParameter<"SharedSpace", "SharedSpace::CTA">:$shared_space
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomR2SType
    : CuteNVGPU_SharedCopy<"CopyAtomR2S", "atom.r2s", "Register-to-shared store">;
def CuteNVGPU_CopyAtomS2RType
    : CuteNVGPU_SharedCopy<"CopyAtomS2R", "atom.s2r", "Shared-to-register load">;

// Multimem atoms: positional `<T, N b, RED, <ACC>, ORDER, SCOPE>`; a trailing
// run of defaults is omitted (copy_bits 0, ADD, no accumulation precision,
// relaxed, sys).

def CuteNVGPU_CopyAtomSIMTMultimemStType
    : CuteNVGPU_CustomType<"CopyAtomSIMTMultimemSt", "atom.simt_multimem_st"> {
  let summary = "multimem.st atom";
  let description = [{
    `copy_bits` 0 (the element width) is kept as 0.

    `!cute_nvgpu.atom.simt_multimem_st<f16, 64 b, release, gpu>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    DefaultValuedParameter<"int", "0">:$copy_bits,
    DefaultValuedParameter<"MemOrderKind", "MemOrderKind::RELAXED">:$mem_order,
    DefaultValuedParameter<"MemScopeKind", "MemScopeKind::SYS">:$mem_scope
  );
}

def CuteNVGPU_CopyAtomSIMTMultimemRedType
    : CuteNVGPU_CustomType<"CopyAtomSIMTMultimemRed", "atom.simt_multimem_red"> {
  let summary = "multimem.red atom";
  let description = [{
    `copy_bits` absent or 0 parses as the element width and is always printed.

    `!cute_nvgpu.atom.simt_multimem_red<f16, 64 b, MAX, release, gpu>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    DefaultValuedParameter<"ReductionKind", "ReductionKind::ADD">:$reduction,
    DefaultValuedParameter<"MemOrderKind", "MemOrderKind::RELAXED">:$mem_order,
    DefaultValuedParameter<"MemScopeKind", "MemScopeKind::SYS">:$mem_scope
  );
}

def CuteNVGPU_CopyAtomSIMTMultimemLdReduceType
    : CuteNVGPU_CustomType<"CopyAtomSIMTMultimemLdReduce",
                           "atom.simt_multimem_ld_reduce"> {
  let summary = "multimem.ld_reduce atom";
  let description = [{
    `copy_bits` 0 (the element width) is kept as 0. The accumulation
    precision must be given to give a memory order or scope.

    `!cute_nvgpu.atom.simt_multimem_ld_reduce<bf16, 64 b, MIN, <F32>, acquire, gpu>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    DefaultValuedParameter<"int", "0">:$copy_bits,
    DefaultValuedParameter<"ReductionKind", "ReductionKind::ADD">:$reduction,
    OptionalParameter<"LdReduceAccPrecisionKindAttr">:$ld_reduce_acc_precision,
    DefaultValuedParameter<"MemOrderKind", "MemOrderKind::RELAXED">:$mem_order,
    DefaultValuedParameter<"MemScopeKind", "MemScopeKind::SYS">:$mem_scope
  );
}

//===----------------------------------------------------------------------===//
// Tensor-memory copy atoms: `<T, D DP, B bit, xR ...>`
//===----------------------------------------------------------------------===//

class CuteNVGPU_TmemCopy<string name, string typeMnemonic, string what,
                         string flag>
    : CuteNVGPU_CustomType<name, typeMnemonic> {
  let summary = what;
  let description = [{
    `!cute_nvgpu.}] # typeMnemonic # "<f32, 32 DP, 32 bit, x4, " # flag # ">`";
  let parameters = !con(
      (ins CuteNVGPU_Ty:$val_type, "int":$num_dp, "int":$num_bit,
           "unsigned":$num_rep),
      !dag(ins, [OptionalParameter<"::mlir::UnitAttr">], [flag]));
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomSM100TmemLoadType
    : CuteNVGPU_TmemCopy<"CopyAtomSM100TmemLoad", "atom.tmem_load",
                         "tcgen05.ld atom", "pack16b">;
def CuteNVGPU_CopyAtomSM100TmemStoreType
    : CuteNVGPU_TmemCopy<"CopyAtomSM100TmemStore", "atom.tmem_store",
                         "tcgen05.st atom", "expand16b">;

def CuteNVGPU_CopyAtomSM10xTmemLoadRedType
    : CuteNVGPU_CustomType<"CopyAtomSM10xTmemLoadRed", "atom.tmem_load_red"> {
  let summary = "tcgen05.ld.red atom";
  let description = [{
    `half_split_off` prints as an integer attribute without spaces around `=`.

    `!cute_nvgpu.atom.tmem_load_red<f32, 16 DP, 32 bit, x4, maxabs, nan, half_split_off=3 : i64>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$num_dp,
    "int":$num_bit,
    "unsigned":$num_rep,
    "TmemLoadRedOp":$red_op,
    OptionalParameter<"::mlir::UnitAttr">:$nan,
    OptionalParameter<"::mlir::IntegerAttr">:$half_split_off
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomSM107TmemLoadSPCompressType
    : CuteNVGPU_CustomType<"CopyAtomSM107TmemLoadSPCompress",
                           "atom.tmem_load_spcompress"> {
  let summary = "SM107 tensor-memory load with sparse compression";
  let description = [{
    `!cute_nvgpu.atom.tmem_load_spcompress<f32, 32 DP, 32 bit, x4, maxabs, red, nan>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$num_dp,
    "int":$num_bit,
    "unsigned":$num_rep,
    "TmemLoadRedOp":$red_op,
    OptionalParameter<"::mlir::UnitAttr">:$red,
    OptionalParameter<"::mlir::UnitAttr">:$nan
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomSM100CopyS2TType
    : CuteNVGPU_CustomType<"CopyAtomSM100CopyS2T", "atom.s2t_copy"> {
  let summary = "tcgen05.cp (shared to tensor memory) atom";
  let description = [{
    The broadcast mode `none` is omitted.

    `!cute_nvgpu.atom.s2t_copy<f8E4M3FN, 32 DP, 128 bit, 1 cta, x4>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$num_dp,
    "int":$num_bit,
    "int":$num_cta,
    DefaultValuedParameter<"CopyS2TBroadcast", "CopyS2TBroadcast::none">:$broadcast
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomSM100S2TCopyV2Type
    : CuteNVGPU_CustomType<"CopyAtomSM100S2TCopyV2", "atom.sm100_s2t_copy_v2"> {
  let summary = "tcgen05.cp atom with the shared-memory major mode";
  let description = [{
    The broadcast mode `none` is omitted.

    `!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$num_dp,
    "int":$num_bit,
    "int":$num_cta,
    CuteNVGPU_Major:$smem_major,
    DefaultValuedParameter<"CopyS2TBroadcast", "CopyS2TBroadcast::none">:$broadcast
  );
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// ldmatrix / stmatrix
//===----------------------------------------------------------------------===//

def CuteNVGPU_CopyAtomLdsmType
    : CuteNVGPU_CustomType<"CopyAtomLdsm", "atom.ldsm"> {
  let summary = "ldmatrix atom";
  let description = [{
    The last field is `t` (transposed) or `n`.

    `!cute_nvgpu.atom.ldsm<val_type = f16, mode = <"(8,8)">, sz_pattern = u16, num_matrices = 4, t>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    CuteNVGPU_Shape:$mode,
    "LdsmSzPattern":$sz_pattern,
    "int":$num_matrices,
    OptionalParameter<"::mlir::UnitAttr">:$transpose
  );
}

def CuteNVGPU_CopyAtomStsmType
    : CuteNVGPU_CustomType<"CopyAtomStsm", "atom.stsm"> {
  let summary = "stmatrix atom";
  let description = [{
    The last field is `t` (transposed) or `n`.

    `!cute_nvgpu.atom.stsm<f16, mode = <"(8,8)">, num_matrices = 4, n>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    CuteNVGPU_Shape:$mode,
    "int":$num_matrices,
    OptionalParameter<"::mlir::UnitAttr">:$transpose
  );
}

//===----------------------------------------------------------------------===//
// TMA atoms
//===----------------------------------------------------------------------===//

// Executable: comma-separated fields up to g_stride, then space-separated
// groups printed when set (booleans when true, sparsity when not 1).

def CuteNVGPU_CopyAtomTmaLoadType
    : CuteNVGPU_CustomType<"CopyAtomTmaLoad", "atom.tma_load"> {
  let summary = "Executable TMA load atom";
  let description = [{
    `!cute_nvgpu.atom.tma_load<bf16, copy_bits = 1024, mode = tiled, num_cta = 2, g_stride = <"()"> mcast = true tma_gbasis = <"(64,1):(1@0,1@1)">>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    DefaultValuedParameter<"int", "1">:$sparsity,
    "int":$copy_bits,
    "TmaLoadMode":$mode,
    "int":$num_cta,
    CuteNVGPU_StrideTy:$g_stride,
    DefaultValuedParameter<"bool", "false">:$mcast,
    OptionalParameter<CuteNVGPU_LayoutTy>:$tma_gbasis,
    DefaultValuedParameter<"bool", "false">:$override,
    DefaultValuedParameter<"bool", "false">:$no_fully_oob_tile
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomTmaStoreType
    : CuteNVGPU_CustomType<"CopyAtomTmaStore", "atom.tma_store"> {
  let summary = "Executable TMA store atom";
  let description = [{
    `!cute_nvgpu.atom.tma_store<bf16, copy_bits = 16384, mode = tiled, g_stride = <"()"> tma_gbasis = <"(64,16):(1@0,1@1)">>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    DefaultValuedParameter<"int", "1">:$sparsity,
    "int":$copy_bits,
    "TmaStoreMode":$mode,
    CuteNVGPU_StrideTy:$g_stride,
    OptionalParameter<CuteNVGPU_LayoutTy>:$tma_gbasis,
    DefaultValuedParameter<"bool", "false">:$override,
    DefaultValuedParameter<"bool", "false">:$no_fully_oob_tile
  );
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomTmaReduceType
    : CuteNVGPU_CustomType<"CopyAtomTmaReduce", "atom.tma_reduce"> {
  let summary = "Executable TMA reduce atom";
  let description = [{
    `!cute_nvgpu.atom.tma_reduce<f32, copy_bits = 131072, mode = tiled, kind = ADD, g_stride = <"()"> tma_gbasis = <"(32,128):(1@1,1@0)">>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    "TmaStoreMode":$mode,
    "ReductionKind":$kind,
    CuteNVGPU_StrideTy:$g_stride,
    OptionalParameter<CuteNVGPU_LayoutTy>:$tma_gbasis
  );
}

def CuteNVGPU_CopyAtomIm2ColTmaLoadType
    : CuteNVGPU_CustomType<"CopyAtomIm2ColTmaLoad", "atom.im2col_tma_load"> {
  let summary = "Executable im2col TMA load atom";
  let description = [{
    `!cute_nvgpu.atom.im2col_tma_load<f16, copy_bits = 131072, num_cta = 1, g_stride = <"()"> mcast = true tma_gbasis = <"(64,128):(1@1,1@0)">>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    "int":$num_cta,
    CuteNVGPU_StrideTy:$g_stride,
    DefaultValuedParameter<"bool", "false">:$mcast,
    OptionalParameter<CuteNVGPU_LayoutTy>:$tma_gbasis
  );
}

def CuteNVGPU_CopyAtomIm2ColTmaStoreType
    : CuteNVGPU_CustomType<"CopyAtomIm2ColTmaStore", "atom.im2col_tma_store"> {
  let summary = "Executable im2col TMA store atom";
  let description = [{
    `!cute_nvgpu.atom.im2col_tma_store<f16, copy_bits = 65536, g_stride = <"()"> tma_gbasis = <"(32,128):(1@1,1@0)">>`
  }];
  let parameters = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    CuteNVGPU_StrideTy:$g_stride,
    OptionalParameter<CuteNVGPU_LayoutTy>:$tma_gbasis
  );
}

// Non-executable: `<[kind, ]T, copy_bits = N, tma_gbasis = <"L">[, tma_format = F]>`.
// An omitted format is the element type's (F16_RN, BF16_RN, F32_RN, TF32_RN,
// F64_RN, U4 for 4-bit, U8 up to 8 bits, U16, U32/S32, U64/S64); it is always
// printed, and empty when the element type has none.

class CuteNVGPU_NonExecTma<string name, string typeMnemonic, string what,
                           string kindType, string example>
    : CuteNVGPU_CustomType<name, typeMnemonic> {
  let summary = "Non-executable " # what # " atom";
  let description = "`!cute_nvgpu." # typeMnemonic # "<" # example # ">`";
  defvar common = (ins
    CuteNVGPU_Ty:$val_type,
    "int":$copy_bits,
    CuteNVGPU_LayoutTy:$tma_gbasis,
    "std::optional<TmaDataFormat>":$tma_format
  );
  let parameters = !if(!empty(kindType), common,
                       !con((ins kindType:$kind), common));
  let genVerifyDecl = 1;
}

def CuteNVGPU_CopyAtomNonExecTiledTmaLoadType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExecTiledTmaLoad",
          "atom.non_exec_tiled_tma_load", "tiled TMA load", "TiledTmaLoad",
          "sm_90, bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN">;
def CuteNVGPU_CopyAtomNonExecTiledTmaStoreType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExecTiledTmaStore",
          "atom.non_exec_tiled_tma_store", "tiled TMA store", "",
          "bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN">;
def CuteNVGPU_CopyAtomNonExecTiledTmaReduceType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExecTiledTmaReduce",
          "atom.non_exec_tiled_tma_reduce", "tiled TMA reduce", "ReductionKind",
          "ADD, f32, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F32_RN">;
def CuteNVGPU_CopyAtomNonExecIm2ColTmaLoadType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExecIm2ColTmaLoad",
          "atom.non_exec_im2col_tma_load", "im2col TMA load", "Im2ColTmaLoad",
          "sm_90, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN">;
def CuteNVGPU_CopyAtomNonExecIm2ColTmaStoreType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExecIm2ColTmaStore",
          "atom.non_exec_im2col_tma_store", "im2col TMA store", "",
          "f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN">;
def CuteNVGPU_CopyAtomNonExec2DGather4TmaLoadType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExec2DGather4TmaLoad",
          "atom.non_exec_2d_gather4_tma_load", "2-D gather4 TMA load",
          "GatherScatterTmaLoad",
          "sm_100, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN">;
def CuteNVGPU_CopyAtomNonExec2DScatter4TmaStoreType
    : CuteNVGPU_NonExecTma<"CopyAtomNonExec2DScatter4TmaStore",
          "atom.non_exec_2d_scatter4_tma_store", "2-D scatter4 TMA store", "",
          "f16, copy_bits = 1024, tma_gbasis = <\"(4,64):(1@0,1@1)\">, tma_format = F16_RN">;

//===----------------------------------------------------------------------===//
// MMA atoms. The shape is `MxNxK` (a static rank-3 shape); `elem_type` lists
// A, B and C; `frag_kind` is the A fragment kind: `ss` (smem_desc), `rs`
// (rmem, sm90) or `ts` (tmem, sm100 and sm107).
//===----------------------------------------------------------------------===//

class CuteNVGPU_MmaAtom<string name, string typeMnemonic, string what,
                        string example>
    : CuteNVGPU_CustomType<name, typeMnemonic> {
  let summary = what # " MMA atom";
  let description = "`!cute_nvgpu." # typeMnemonic # "<" # example # ">`";
  let genVerifyDecl = 1;
}

def CuteNVGPU_UniversalFmaAtomType
    : CuteNVGPU_MmaAtom<"UniversalFmaAtom", "atom.universal_fma", "Universal FMA",
                        "1x1x1, (f32, f32) -> f32 "> {
  let parameters = (ins CuteNVGPU_Shape:$shape_mnk, CuteNVGPU_Ty:$a_type,
                        CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type);
}

def CuteNVGPU_MmaAtomSM80Type
    : CuteNVGPU_MmaAtom<"MmaAtomSM80", "sm80.mma", "SM80 mma.sync",
          "16x8x32, elem_type = (i8, i8, i32), int_overflow = <satfinite> "> {
  let description = [{
    `binary_op` needs `int_overflow` before it.

    `!cute_nvgpu.sm80.mma<16x8x32, elem_type = (i8, i8, i32), int_overflow = <wrapped>, binary_op = <none> >`
  }];
  let parameters = (ins
    CuteNVGPU_Shape:$shape_mnk,
    CuteNVGPU_Ty:$a_type, CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type,
    OptionalParameter<"MMAIntOverflowAttr">:$int_overflow,
    OptionalParameter<"BinaryOpAttr">:$binary_op
  );
}

def CuteNVGPU_MmaAtomSM80SparseType
    : CuteNVGPU_MmaAtom<"MmaAtomSM80Sparse", "sm80.sparse_mma",
          "SM80 sparse mma.sync",
          "16x8x64, elem_type = (i8, i8, i32), md_format = tid, int_overflow = <satfinite> "> {
  let parameters = (ins
    CuteNVGPU_Shape:$shape_mnk,
    CuteNVGPU_Ty:$a_type, CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type,
    "SparseMetadataFormat":$sparse_metadata_format,
    OptionalParameter<"MMAIntOverflowAttr">:$int_overflow
  );
}

def CuteNVGPU_MmaAtomSM89Type
    : CuteNVGPU_MmaAtom<"MmaAtomSM89", "sm89.mma", "SM89 fp8 mma.sync",
          "16x8x32, elem_type = (f8E4M3FN, f8E4M3FN, f32) "> {
  let parameters = (ins CuteNVGPU_Shape:$shape_mnk, CuteNVGPU_Ty:$a_type,
                        CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type);
}

def CuteNVGPU_MmaAtomSM90Type
    : CuteNVGPU_MmaAtom<"MmaAtomSM90", "sm90.mma", "SM90 wgmma",
          "64x128x16, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = rs, a_neg, b_neg"> {
  let description = [{
    The trailing `int_overflow`, `a_neg` and `b_neg` parse in any order and
    print in this one.

    `!cute_nvgpu.sm90.mma<64x128x32, ab_major = (k, k), elem_type = (i8, i8, i32), frag_kind = ss, int_overflow = <satfinite>, a_neg, b_neg>`
  }];
  let parameters = (ins
    CuteNVGPU_Shape:$shape_mnk,
    CuteNVGPU_Major:$a_major, CuteNVGPU_Major:$b_major,
    CuteNVGPU_Ty:$a_type, CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type,
    "MmaFragKind":$a_frag_kind,
    OptionalParameter<"MMAIntOverflowAttr">:$int_overflow,
    OptionalParameter<"::mlir::UnitAttr">:$a_neg,
    OptionalParameter<"::mlir::UnitAttr">:$b_neg
  );
}

def CuteNVGPU_MmaAtomSM120BlockScaledType
    : CuteNVGPU_MmaAtom<"MmaAtomSM120BlockScaled", "SM120.mma_bs",
          "SM120 block-scaled mma.sync",
          "16x8x64, vec_size = 32, elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, use_sf_layout_TV = true"> {
  let parameters = (ins
    CuteNVGPU_Shape:$shape_mnk,
    "int":$vec_size,
    CuteNVGPU_Ty:$a_type, CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type,
    CuteNVGPU_Ty:$sf_type,
    "bool":$use_sf_layout_TV
  );
}

// UMMA (tcgen05.mma): `<MxNxK, num_cta = N, ab_major = (A, B),
// elem_type = (A, B, C), [e_type = T, ][sf_type = T, ]
// [sparse_metadata_format = tid, ]frag_kind = ss|ts, c_scale_exp = N |
// vec_size = N[, arch_promote = sm_103]
// [, ab_collector_op = (A, B)][, int_overflow = <o>]>`. arch_promote (sm100
// block-scaled only) defaults to sm_100 and is omitted then; the sm107 atoms
// require ab_collector_op.

class CuteNVGPU_Umma<string name, string typeMnemonic, string what,
                     string example, bit sf, bit sp, bit bs, bit sm107>
    : CuteNVGPU_MmaAtom<name, typeMnemonic, what, example> {
  let parameters = !con(
      (ins CuteNVGPU_Shape:$shape_mnk, "int":$num_cta,
           CuteNVGPU_Major:$a_major, CuteNVGPU_Major:$b_major,
           CuteNVGPU_Ty:$a_type, CuteNVGPU_Ty:$b_type, CuteNVGPU_Ty:$c_type),
      !if(sf, (ins CuteNVGPU_Ty:$sf_type), (ins)),
      !if(!and(sp, !not(sf)), (ins CuteNVGPU_Ty:$e_type), (ins)),
      !if(sp, (ins "SparseMetadataFormat":$sparse_metadata_format), (ins)),
      (ins "MmaFragKind":$a_frag_kind),
      !if(bs, (ins "int":$vec_size), (ins "int":$c_scale_exp)),
      !if(!and(bs, !not(sm107)),
          (ins DefaultValuedParameter<"Arch", "Arch::sm_100">:$arch_promote), (ins)),
      !if(sm107, (ins "MmaCollectorOp":$a_collector_op,
                      "MmaCollectorOp":$b_collector_op), (ins)),
      (ins OptionalParameter<"MMAIntOverflowAttr">:$int_overflow));
}

def CuteNVGPU_MmaAtomSM100UMMAType
    : CuteNVGPU_Umma<"MmaAtomSM100UMMA", "sm100.mma", "SM100 UMMA",
          "128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0",
          0, 0, 0, 0>;
def CuteNVGPU_MmaAtomSM100UMMASparseType
    : CuteNVGPU_Umma<"MmaAtomSM100UMMASparse", "sm100.mma_sp", "SM100 sparse UMMA",
          "128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0",
          0, 1, 0, 0>;
def CuteNVGPU_MmaAtomSM100UMMABlockScaledType
    : CuteNVGPU_Umma<"MmaAtomSM100UMMABlockScaled", "sm100.mma_bs",
          "SM100 block-scaled UMMA",
          "128x128x96, num_cta = 1, ab_major = (k, k), elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 16, arch_promote = sm_103",
          1, 0, 1, 0>;
def CuteNVGPU_MmaAtomSM100UMMABlockScaledSparseType
    : CuteNVGPU_Umma<"MmaAtomSM100UMMABlockScaledSparse", "sm100.mma_bs_sp",
          "SM100 block-scaled sparse UMMA",
          "128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32",
          1, 1, 1, 0>;
def CuteNVGPU_MmaAtomSM107UMMAType
    : CuteNVGPU_Umma<"MmaAtomSM107UMMA", "sm107.mma", "SM107 UMMA",
          "128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0, ab_collector_op = (lastuse, fill)",
          0, 0, 0, 1>;
def CuteNVGPU_MmaAtomSM107UMMASparseType
    : CuteNVGPU_Umma<"MmaAtomSM107UMMASparse", "sm107.mma_sp", "SM107 sparse UMMA",
          "128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0, ab_collector_op = (discard, discard)",
          0, 1, 0, 1>;
def CuteNVGPU_MmaAtomSM107UMMABlockScaledType
    : CuteNVGPU_Umma<"MmaAtomSM107UMMABlockScaled", "sm107.mma_bs",
          "SM107 block-scaled UMMA",
          "128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, use), int_overflow = <satfinite>",
          1, 0, 1, 1>;
def CuteNVGPU_MmaAtomSM107UMMABlockScaledSparseType
    : CuteNVGPU_Umma<"MmaAtomSM107UMMABlockScaledSparse", "sm107.mma_bs_sp",
          "SM107 block-scaled sparse UMMA",
          "128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, discard)",
          1, 1, 1, 1>;

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_TYPES_TD
