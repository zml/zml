// The nvvm dialect's enums, generated from the nvidia-cutlass-dsl 4.8.0 enum
// bindings (`_nvvm_enum_gen.py`); do not edit.
//
// Case values and spellings are the compiler's. An enum the bindings build
// as a dialect attribute also has its EnumAttr, printed `#<mnemonic><case>`
// or, where the compiler prints it so, `#nvvm<mnemonic case>`.

#ifndef CUTE_IR_GENERATED_NVVM_ENUMS_TD
#define CUTE_IR_GENERATED_NVVM_ENUMS_TD

include "cute_ir/Dialect/NVVM/IR/NVVMDialect.td"
include "mlir/IR/EnumAttr.td"

def NVVM_AtomicOpKindEnum : I32EnumAttr<"AtomicOpKind", "operations supported by atom instruction", [
    I32EnumAttrCase<"AND", 0, "and">,
    I32EnumAttrCase<"OR", 1, "or">,
    I32EnumAttrCase<"XOR", 2, "xor">,
    I32EnumAttrCase<"CAS", 3, "cas">,
    I32EnumAttrCase<"EXCH", 4, "exch">,
    I32EnumAttrCase<"ADD", 5, "add">,
    I32EnumAttrCase<"INC", 6, "inc">,
    I32EnumAttrCase<"DEC", 7, "dec">,
    I32EnumAttrCase<"MIN", 8, "min">,
    I32EnumAttrCase<"MAX", 9, "max">,
    I32EnumAttrCase<"UMIN", 10, "umin">,
    I32EnumAttrCase<"UMAX", 11, "umax">,
    I32EnumAttrCase<"FADD", 12, "fadd">,
    I32EnumAttrCase<"SUB", 13, "sub">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_BarrierReductionEnum : I32EnumAttr<"BarrierReduction", "NVVM barrier reduction operation", [
    I32EnumAttrCase<"POPC", 0, "popc">,
    I32EnumAttrCase<"AND", 1, "and">,
    I32EnumAttrCase<"OR", 2, "or">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_BarrierReduxKindEnum : I32EnumAttr<"BarrierReduxKind", "NVVM barrier redux kind", [
    I32EnumAttrCase<"AND", 0, "and">,
    I32EnumAttrCase<"OR", 1, "or">,
    I32EnumAttrCase<"POPC", 2, "popc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_BlockScaleFormatEnum : I32EnumAttr<"BlockScaleFormat", "MMA Block Scale Format", [
    I32EnumAttrCase<"UE8M0", 0, "ue8m0">,
    I32EnumAttrCase<"UE4M3", 1, "ue4m3">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_CTAGroupKindEnum : I32EnumAttr<"CTAGroupKind", "NVVM CTA group kind", [
    I32EnumAttrCase<"CTA_1", 0, "cta_1">,
    I32EnumAttrCase<"CTA_2", 1, "cta_2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_CVTPackFloatKindEnum : I32EnumAttr<"CVTPackFloatKind", "NVVM CVT Pack Float kind", [
    I32EnumAttrCase<"F32", 0, "f32">,
    I32EnumAttrCase<"F16x2", 1, "f16x2">,
    I32EnumAttrCase<"E4M3x2", 2, "e4m3x2">,
    I32EnumAttrCase<"E5M2x2", 3, "e5m2x2">,
    I32EnumAttrCase<"BF16x2", 4, "bf16x2">,
    I32EnumAttrCase<"E2M1x2", 5, "e2m1x2">,
    I32EnumAttrCase<"E2M3x2", 6, "e2m3x2">,
    I32EnumAttrCase<"E3M2x2", 7, "e3m2x2">,
    I32EnumAttrCase<"UE8M0x2", 8, "ue8m0x2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_CacheEvictionPriorityEnum : I32EnumAttr<"CacheEvictionPriority", "NVVM Cache Eviction Priority", [
    I32EnumAttrCase<"EvictNormal", 0, "evict_normal">,
    I32EnumAttrCase<"EvictFirst", 1, "evict_first">,
    I32EnumAttrCase<"EvictLast", 2, "evict_last">,
    I32EnumAttrCase<"EvictUnchanged", 3, "evict_unchanged">,
    I32EnumAttrCase<"NoAllocate", 4, "no_allocate">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ClusterLaunchControlQueryTypeEnum : I32EnumAttr<"ClusterLaunchControlQueryType", "NVVM ClusterLaunchControlQueryType", [
    I32EnumAttrCase<"IS_CANCELED", 0, "is_canceled">,
    I32EnumAttrCase<"GET_FIRST_CTA_ID_X", 1, "get_first_cta_id_x">,
    I32EnumAttrCase<"GET_FIRST_CTA_ID_Y", 2, "get_first_cta_id_y">,
    I32EnumAttrCase<"GET_FIRST_CTA_ID_Z", 3, "get_first_cta_id_z">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_CompareOpKindEnum : I32EnumAttr<"CompareOpKind", "Comparison operator encoding", [
    I32EnumAttrCase<"EQ", 0, "eq">,
    I32EnumAttrCase<"NE", 1, "ne">,
    I32EnumAttrCase<"UGT", 2, "ugt">,
    I32EnumAttrCase<"UGE", 3, "uge">,
    I32EnumAttrCase<"ULT", 4, "ult">,
    I32EnumAttrCase<"ULE", 5, "ule">,
    I32EnumAttrCase<"SGT", 6, "sgt">,
    I32EnumAttrCase<"SGE", 7, "sge">,
    I32EnumAttrCase<"SLT", 8, "slt">,
    I32EnumAttrCase<"SLE", 9, "sle">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ConvertFP4TypeEnum : I32EnumAttr<"ConvertFP4Type", "NVVM ConvertFP4Type kind", [
    I32EnumAttrCase<"E2M1", 0, "e2m1">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ConvertFP8TypeEnum : I32EnumAttr<"ConvertFP8Type", "NVVM ConvertFP8Type kind", [
    I32EnumAttrCase<"E4M3", 0, "e4m3">,
    I32EnumAttrCase<"E5M2", 1, "e5m2">,
    I32EnumAttrCase<"UE8M0", 2, "ue8m0">,
    I32EnumAttrCase<"UE5M3", 3, "ue5m3">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ConvertScaleKindEnum : I32EnumAttr<"ConvertScaleKind", "NVVM ConvertScale kind", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"UE8M0", 1, "ue8m0">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_DotAccumulateTypeEnum : I32EnumAttr<"DotAccumulateType", "NVVM DotAccumulateType", [
    I32EnumAttrCase<"SIGNED", 1, "signed">,
    I32EnumAttrCase<"UNSIGNED", 0, "unsigned">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_EvictKindEnum : I32EnumAttr<"EvictKind", "NVVM L2 Prefetch Size", [
    I32EnumAttrCase<"NORMAL", 0, "evict_normal">,
    I32EnumAttrCase<"FIRST", 1, "evict_first">,
    I32EnumAttrCase<"LAST", 2, "evict_last">,
    I32EnumAttrCase<"UNCHANGED", 3, "evict_unchanged">,
    I32EnumAttrCase<"NOALLOCATE", 4, "evict_no_allocate">,
    I32EnumAttrCase<"NONE", 5, "none">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_FPRoundingModeEnum : I32EnumAttr<"FPRoundingMode", "NVVM FPRoundingMode kind", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"RN", 1, "rn">,
    I32EnumAttrCase<"RM", 2, "rm">,
    I32EnumAttrCase<"RP", 3, "rp">,
    I32EnumAttrCase<"RZ", 4, "rz">,
    I32EnumAttrCase<"RNA", 5, "rna">,
    I32EnumAttrCase<"RS", 6, "rs">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_GridDepActionKindEnum : I32EnumAttr<"GridDepActionKind", "Action kind for grid dependency control", [
    I32EnumAttrCase<"wait", 0, "wait">,
    I32EnumAttrCase<"launch_dependents", 1, "launch_dependents">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_IntegerRoundingModeEnum : I32EnumAttr<"IntegerRoundingMode", "NVVM IntegerRoundingMode kind", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"RNI", 1, "rni">,
    I32EnumAttrCase<"RZI", 2, "rzi">,
    I32EnumAttrCase<"RMI", 3, "rmi">,
    I32EnumAttrCase<"RPI", 4, "rpi">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_L2PrefetchSizeEnum : I32EnumAttr<"L2PrefetchSize", "NVVM L2 Prefetch Size", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"RESERVED", 1, "reserved">,
    I32EnumAttrCase<"SIZE_64B", 2, "size_64b">,
    I32EnumAttrCase<"SIZE_128B", 3, "size_128b">,
    I32EnumAttrCase<"SIZE_256B", 4, "size_256b">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_LdStMatrixEltTypeEnum : I32EnumAttr<"LdStMatrixEltType", "Element type for ldmatrix and stmatrix", [
    I32EnumAttrCase<"B16", 0, "b16">,
    I32EnumAttrCase<"B8", 1, "b8">,
    I32EnumAttrCase<"B8X16_B6X16_P32", 2, "b8x16.b6x16_p32">,
    I32EnumAttrCase<"B8X16_B4X16_P64", 3, "b8x16.b4x16_p64">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_LoadCacheModifierExtKindEnum : I32EnumAttr<"LoadCacheModifierExtKind", "NVVM load cache modifier kind(Ext)", [
    I32EnumAttrCase<"CA", 0, "ca">,
    I32EnumAttrCase<"CG", 1, "cg">,
    I32EnumAttrCase<"CS", 2, "cs">,
    I32EnumAttrCase<"LU", 3, "lu">,
    I32EnumAttrCase<"CV", 4, "cv">,
    I32EnumAttrCase<"NONE", 5, "none">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_LoadCacheModifierKindEnum : I32EnumAttr<"LoadCacheModifierKind", "NVVM load cache modifier kind", [
    I32EnumAttrCase<"CA", 0, "ca">,
    I32EnumAttrCase<"CG", 1, "cg">,
    I32EnumAttrCase<"CS", 2, "cs">,
    I32EnumAttrCase<"LU", 3, "lu">,
    I32EnumAttrCase<"CV", 4, "cv">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_LoadShapeEnum : I32EnumAttr<"LoadShape", "shape attribute for ldmatrix", [
    I32EnumAttrCase<"M8N8", 0, "m8n8">,
    I32EnumAttrCase<"M8N16", 1, "m8n16">,
    I32EnumAttrCase<"M16N16", 3, "m16n16">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_LoadSrcFormatEnum : I32EnumAttr<"LoadSrcFormat", "source format for ldmatrix", [
    I32EnumAttrCase<"B6x16_P32", 0, "b6x16_p32">,
    I32EnumAttrCase<"B4x16_P64", 1, "b4x16_p64">,
    I32EnumAttrCase<"B8", 2, "b8">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MBarrierLayoutEnum : I32EnumAttr<"MBarrierLayout", "NVVM MBarrier Layout", [
    I32EnumAttrCase<"V0", 0, "v0">,
    I32EnumAttrCase<"V1", 1, "v1">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MBarrierPhaseEnum : I32EnumAttr<"MBarrierPhase", "NVVM mbarrier phase type", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"PRIMARY", 1, "primary">,
    I32EnumAttrCase<"CONDITIONAL", 2, "conditional">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MBarrierScopeKindEnum : I32EnumAttr<"MBarrierScopeKind", "NVVM MBarrier scope kind", [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"CLUSTER", 1, "cluster">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MBarrierSpaceKindEnum : I32EnumAttr<"MBarrierSpaceKind", "NVVM MBarrier space kind", [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"CLUSTER", 1, "cluster">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MBarrierTxnKindEnum : I32EnumAttr<"MBarrierTxnKind", "NVVM MBarrier Transaction kind", [
    I32EnumAttrCase<"ARRIVE", 0, "arrive">,
    I32EnumAttrCase<"ARRIVE_DROP", 1, "arrive_drop">,
    I32EnumAttrCase<"ARRIVE_EXPECT_TX", 2, "arrive_expect_tx">,
    I32EnumAttrCase<"ARRIVE_DROP_EXPECT_TX", 3, "arrive_drop_expect_tx">,
    I32EnumAttrCase<"EXPECT_TX", 4, "expect_tx">,
    I32EnumAttrCase<"COMPLETE_TX", 5, "complete_tx">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MBarrierWaitKindEnum : I32EnumAttr<"MBarrierWaitKind", "NVVM MBarrier wait kind", [
    I32EnumAttrCase<"TEST", 0, "test">,
    I32EnumAttrCase<"TRY", 1, "try">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMAB1OpEnum : I32EnumAttr<"MMAB1Op", "MMA binary operations", [
    I32EnumAttrCase<"none", 0, "none">,
    I32EnumAttrCase<"xor_popc", 1, "xor_popc">,
    I32EnumAttrCase<"and_popc", 2, "and_popc">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMABlockScaleKindEnum : I32EnumAttr<"MMABlockScaleKind", "Block Scale Kind", [
    I32EnumAttrCase<"MXF8F6F4", 0, "mxf8f6f4">,
    I32EnumAttrCase<"MXF4", 1, "mxf4">,
    I32EnumAttrCase<"MXF4NVF4", 2, "mxf4nvf4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMACtaCountEnum : I32EnumAttr<"MMACtaCount", "MMA CTA count", [
    I32EnumAttrCase<"cta1", 1, "cta1">,
    I32EnumAttrCase<"cta2", 2, "cta2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMAFragEnum : I32EnumAttr<"MMAFrag", "NVVM MMA frag type", [
    I32EnumAttrCase<"a", 0, "a">,
    I32EnumAttrCase<"b", 1, "b">,
    I32EnumAttrCase<"c", 2, "c">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMAIntOverflowEnum : I32EnumAttr<"MMAIntOverflow", "MMA overflow options", [
    I32EnumAttrCase<"satfinite", 1, "satfinite">,
    I32EnumAttrCase<"wrapped", 0, "wrapped">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMAKindEnum : I32EnumAttr<"MMAKind", "MMA operation kind", [
    I32EnumAttrCase<"f8f6f4", 0, "f8f6f4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMALayoutEnum : I32EnumAttr<"MMALayout", "NVVM MMA layout", [
    I32EnumAttrCase<"row", 0, "row">,
    I32EnumAttrCase<"col", 1, "col">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MMATypesEnum : I32EnumAttr<"MMATypes", "NVVM MMA types", [
    I32EnumAttrCase<"f16", 0, "f16">,
    I32EnumAttrCase<"f32", 1, "f32">,
    I32EnumAttrCase<"tf32", 2, "tf32">,
    I32EnumAttrCase<"bf16", 9, "bf16">,
    I32EnumAttrCase<"s8", 4, "s8">,
    I32EnumAttrCase<"u8", 3, "u8">,
    I32EnumAttrCase<"s32", 5, "s32">,
    I32EnumAttrCase<"s4", 8, "s4">,
    I32EnumAttrCase<"u4", 7, "u4">,
    I32EnumAttrCase<"b1", 6, "b1">,
    I32EnumAttrCase<"f64", 10, "f64">,
    I32EnumAttrCase<"e4m3", 11, "e4m3">,
    I32EnumAttrCase<"e5m2", 12, "e5m2">,
    I32EnumAttrCase<"e3m2", 13, "e3m2">,
    I32EnumAttrCase<"e2m3", 14, "e2m3">,
    I32EnumAttrCase<"e2m1", 15, "e2m1">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MatchSyncKindEnum : I32EnumAttr<"MatchSyncKind", "NVVM match sync kind", [
    I32EnumAttrCase<"any", 0, "any">,
    I32EnumAttrCase<"all", 1, "all">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MemOrderKindEnum : I32EnumAttr<"MemOrderKind", "NVVM Memory Ordering kind", [
    I32EnumAttrCase<"WEAK", 0, "weak">,
    I32EnumAttrCase<"RELAXED", 1, "relaxed">,
    I32EnumAttrCase<"ACQUIRE", 2, "acquire">,
    I32EnumAttrCase<"RELEASE", 3, "release">,
    I32EnumAttrCase<"ACQ_REL", 4, "acq_rel">,
    I32EnumAttrCase<"SC", 5, "sc">,
    I32EnumAttrCase<"MMIO", 6, "mmio">,
    I32EnumAttrCase<"VOLATILE", 8, "volatile">,
    I32EnumAttrCase<"CONSTANT", 7, "constant">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MemScopeKindEnum : I32EnumAttr<"MemScopeKind", "NVVM Memory Scope kind", [
    I32EnumAttrCase<"CTA", 0, "cta">,
    I32EnumAttrCase<"CLUSTER", 1, "cluster">,
    I32EnumAttrCase<"GPU", 2, "gpu">,
    I32EnumAttrCase<"SYS", 3, "sys">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_MulModeEnum : I32EnumAttr<"MulMode", "multiply mode attribute", [
    I32EnumAttrCase<"HI", 0, "hi">,
    I32EnumAttrCase<"LO", 1, "lo">,
    I32EnumAttrCase<"WIDE", 2, "wide">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_NVVMMemorySpaceEnum : I32EnumAttr<"NVVMMemorySpace", "NVVM Memory Space", [
    I32EnumAttrCase<"Generic", 0, "generic">,
    I32EnumAttrCase<"Global", 1, "global">,
    I32EnumAttrCase<"Shared", 3, "shared">,
    I32EnumAttrCase<"Constant", 4, "constant">,
    I32EnumAttrCase<"Local", 5, "local">,
    I32EnumAttrCase<"Tensor", 6, "tensor">,
    I32EnumAttrCase<"SharedCluster", 7, "shared_cluster">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_PermuteModeEnum : I32EnumAttr<"PermuteMode", "NVVM permute mode", [
    I32EnumAttrCase<"DEFAULT", 0, "default">,
    I32EnumAttrCase<"F4E", 1, "f4e">,
    I32EnumAttrCase<"B4E", 2, "b4e">,
    I32EnumAttrCase<"RC8", 3, "rc8">,
    I32EnumAttrCase<"ECL", 4, "ecl">,
    I32EnumAttrCase<"ECR", 5, "ecr">,
    I32EnumAttrCase<"RC16", 6, "rc16">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_PrefetchCacheLevelEnum : I32EnumAttr<"PrefetchCacheLevel", "NVVM Prefetch Cache Level", [
    I32EnumAttrCase<"L1", 0, "L1">,
    I32EnumAttrCase<"L2", 1, "L2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ProxyKindEnum : I32EnumAttr<"ProxyKind", "Proxy kind", [
    I32EnumAttrCase<"alias", 0, "alias">,
    I32EnumAttrCase<"async_", 1, "async">,
    I32EnumAttrCase<"async_global", 2, "async.global">,
    I32EnumAttrCase<"async_shared", 3, "async.shared">,
    I32EnumAttrCase<"TENSORMAP", 4, "tensormap">,
    I32EnumAttrCase<"GENERIC", 5, "generic">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ReductionKindEnum : I32EnumAttr<"ReductionKind", "NVVM Reduction Kind attribute", [
    I32EnumAttrCase<"ADD", 1, "add">,
    I32EnumAttrCase<"AND", 2, "and">,
    I32EnumAttrCase<"MAX", 3, "max">,
    I32EnumAttrCase<"MIN", 4, "min">,
    I32EnumAttrCase<"OR", 5, "or">,
    I32EnumAttrCase<"UMAX", 6, "umax">,
    I32EnumAttrCase<"UMIN", 7, "umin">,
    I32EnumAttrCase<"XOR", 8, "xor">,
    I32EnumAttrCase<"FMIN", 9, "fmin">,
    I32EnumAttrCase<"FMAX", 10, "fmax">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ReductionOpEnum : I32EnumAttr<"ReductionOp", "Ops supported by red instruction", [
    I32EnumAttrCase<"AND", 0, "and">,
    I32EnumAttrCase<"OR", 1, "or">,
    I32EnumAttrCase<"XOR", 2, "xor">,
    I32EnumAttrCase<"ADD", 3, "add">,
    I32EnumAttrCase<"INC", 4, "inc">,
    I32EnumAttrCase<"DEC", 5, "dec">,
    I32EnumAttrCase<"MIN", 6, "min">,
    I32EnumAttrCase<"MAX", 7, "max">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ReductionTypeEnum : I32EnumAttr<"ReductionType", "types supported by red instruction", [
    I32EnumAttrCase<"B32", 0, "b32">,
    I32EnumAttrCase<"B64", 1, "b64">,
    I32EnumAttrCase<"U32", 2, "u32">,
    I32EnumAttrCase<"U64", 3, "u64">,
    I32EnumAttrCase<"S32", 4, "s32">,
    I32EnumAttrCase<"S64", 5, "s64">,
    I32EnumAttrCase<"F32", 6, "f32">,
    I32EnumAttrCase<"F64", 7, "f64">,
    I32EnumAttrCase<"F16", 8, "f16">,
    I32EnumAttrCase<"F16X2", 9, "f16x2">,
    I32EnumAttrCase<"BF16", 10, "bf16">,
    I32EnumAttrCase<"BF16X2", 11, "bf16x2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPCompressElemSizeEnum : I32EnumAttr<"SPCompressElemSize", "Sparse tensor compression element size", [
    I32EnumAttrCase<"B4", 4, "b4">,
    I32EnumAttrCase<"B8", 8, "b8">,
    I32EnumAttrCase<"B16", 16, "b16">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPCompressFactorTypeEnum : I32EnumAttr<"SPCompressFactorType", "Sparse tensor compression factor type", [
    I32EnumAttrCase<"SP2TO4", 0, "sp2to4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPCompressIndexSizeEnum : I32EnumAttr<"SPCompressIndexSize", "Sparse tensor compression index size", [
    I32EnumAttrCase<"B2", 2, "b2">,
    I32EnumAttrCase<"B4", 4, "b4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPCompressOpKindEnum : I32EnumAttr<"SPCompressOpKind", "spcompress operation kind", [
    I32EnumAttrCase<"MIN", 0, "min">,
    I32EnumAttrCase<"MAX", 1, "max">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPCompressRepFactorEnum : I32EnumAttr<"SPCompressRepFactor", "Sparse tensor compression repetition factor", [
    I32EnumAttrCase<"X1", 1, "x1">,
    I32EnumAttrCase<"X2", 2, "x2">,
    I32EnumAttrCase<"X4", 4, "x4">,
    I32EnumAttrCase<"X8", 8, "x8">,
    I32EnumAttrCase<"X16", 16, "x16">,
    I32EnumAttrCase<"X32", 32, "x32">,
    I32EnumAttrCase<"X64", 64, "x64">,
    I32EnumAttrCase<"X128", 128, "x128">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPDecompressElemSizeEnum : I32EnumAttr<"SPDecompressElemSize", "Sparse tensor decompression element size", [
    I32EnumAttrCase<"B4", 4, "b4">,
    I32EnumAttrCase<"B8", 8, "b8">,
    I32EnumAttrCase<"B16", 16, "b16">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPDecompressFactorTypeEnum : I32EnumAttr<"SPDecompressFactorType", "Sparse tensor decompression factor type", [
    I32EnumAttrCase<"SP1TO2", 0, "sp1to2">,
    I32EnumAttrCase<"SP1TO4", 1, "sp1to4">,
    I32EnumAttrCase<"SP1TO8", 2, "sp1to8">,
    I32EnumAttrCase<"SP1TO16", 3, "sp1to16">,
    I32EnumAttrCase<"SP2TO4", 4, "sp2to4">,
    I32EnumAttrCase<"SP2TO8", 5, "sp2to8">,
    I32EnumAttrCase<"SP2TO16", 6, "sp2to16">,
    I32EnumAttrCase<"SP4TO8", 7, "sp4to8">,
    I32EnumAttrCase<"SP4TO16", 8, "sp4to16">,
    I32EnumAttrCase<"SP8TO16", 9, "sp8to16">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPDecompressIndexSizeEnum : I32EnumAttr<"SPDecompressIndexSize", "Sparse tensor decompression index size", [
    I32EnumAttrCase<"B2", 2, "b2">,
    I32EnumAttrCase<"B4", 4, "b4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SPDecompressRepFactorEnum : I32EnumAttr<"SPDecompressRepFactor", "Sparse tensor decompression repetition factor", [
    I32EnumAttrCase<"X1", 1, "x1">,
    I32EnumAttrCase<"X2", 2, "x2">,
    I32EnumAttrCase<"X4", 4, "x4">,
    I32EnumAttrCase<"X8", 8, "x8">,
    I32EnumAttrCase<"X16", 16, "x16">,
    I32EnumAttrCase<"X32", 32, "x32">,
    I32EnumAttrCase<"X64", 64, "x64">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SaturationModeEnum : I32EnumAttr<"SaturationMode", "NVVM SaturationMode kind", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"SATFINITE", 1, "satfinite">,
    I32EnumAttrCase<"SAT", 2, "sat">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SaturationModeKindEnum : I32EnumAttr<"SaturationModeKind", "NVVM SaturationMode kind", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"SATFINITE", 1, "satfinite">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ScaleVecSizeEnum : I32EnumAttr<"ScaleVecSize", "MMA Scale Vector Sizes", [
    I32EnumAttrCase<"X1", 0, "x1">,
    I32EnumAttrCase<"X2", 1, "x2">,
    I32EnumAttrCase<"X4", 2, "x4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SetMaxRegisterActionEnum : I32EnumAttr<"SetMaxRegisterAction", "NVVM set max register action", [
    I32EnumAttrCase<"decrease", 1, "decrease">,
    I32EnumAttrCase<"increase", 0, "increase">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SharedSpaceEnum : I32EnumAttr<"SharedSpace", "Shared memory space", [
    I32EnumAttrCase<"shared_cta", 0, "cta">,
    I32EnumAttrCase<"shared_cluster", 1, "cluster">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ShflKindEnum : I32EnumAttr<"ShflKind", "NVVM shuffle kind", [
    I32EnumAttrCase<"bfly", 0, "bfly">,
    I32EnumAttrCase<"up", 1, "up">,
    I32EnumAttrCase<"down", 2, "down">,
    I32EnumAttrCase<"idx", 3, "idx">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_SparsityFormatEnum : I32EnumAttr<"SparsityFormat", "MMA Sparsity Format", [
    I32EnumAttrCase<"THREAD", 0, "thread">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_StateSpaceEnum : I32EnumAttr<"StateSpace", "NVVM State Space", [
    I32EnumAttrCase<"GENERIC", 0, "generic">,
    I32EnumAttrCase<"GLOBAL", 1, "global">,
    I32EnumAttrCase<"SHARED_CTA", 2, "shared_cta">,
    I32EnumAttrCase<"SHARED_CLUSTER", 3, "shared_cluster">,
    I32EnumAttrCase<"CONSTANT", 4, "constant">,
    I32EnumAttrCase<"LOCAL", 5, "local">,
    I32EnumAttrCase<"TENSOR", 6, "tensor">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_StoreCacheModifierKindEnum : I32EnumAttr<"StoreCacheModifierKind", "NVVM store cache modifier kind", [
    I32EnumAttrCase<"WB", 0, "wb">,
    I32EnumAttrCase<"CG", 1, "cg">,
    I32EnumAttrCase<"CS", 2, "cs">,
    I32EnumAttrCase<"WT", 3, "wt">,
    I32EnumAttrCase<"NONE", 4, "none">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_StoreShapeEnum : I32EnumAttr<"StoreShape", "shape attribute for stmatrix", [
    I32EnumAttrCase<"M8N8", 0, "m8n8">,
    I32EnumAttrCase<"M16N8", 2, "m16n8">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TCBarParamEnum : I32EnumAttr<"TCBarParam", "Cluster MMA Barrier Parameter Type", [
    I32EnumAttrCase<"a1t0", 0, "a1t0">,
    I32EnumAttrCase<"a0tx", 1, "a0tx">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TMALoadModeEnum : I32EnumAttr<"TMALoadMode", "NVVM TMA Load Mode", [
    I32EnumAttrCase<"TILE", 0, "tile">,
    I32EnumAttrCase<"IM2COL", 1, "im2col">,
    I32EnumAttrCase<"IM2COL_W", 2, "im2col_w">,
    I32EnumAttrCase<"IM2COL_W_128", 3, "im2col_w_128">,
    I32EnumAttrCase<"TILE_GATHER4", 4, "tile_gather4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TMAReduxKindEnum : I32EnumAttr<"TMAReduxKind", "NVVM TMA redux kind", [
    I32EnumAttrCase<"ADD", 0, "add">,
    I32EnumAttrCase<"MAX", 2, "max">,
    I32EnumAttrCase<"MIN", 1, "min">,
    I32EnumAttrCase<"INC", 3, "inc">,
    I32EnumAttrCase<"DEC", 4, "dec">,
    I32EnumAttrCase<"AND", 5, "and">,
    I32EnumAttrCase<"OR", 6, "or">,
    I32EnumAttrCase<"XOR", 7, "xor">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TMAStoreModeEnum : I32EnumAttr<"TMAStoreMode", "NVVM TMA Store Mode", [
    I32EnumAttrCase<"TILE", 0, "tile">,
    I32EnumAttrCase<"IM2COL", 1, "im2col">,
    I32EnumAttrCase<"TILE_SCATTER4", 2, "tile_scatter4">,
    I32EnumAttrCase<"IM2COL_W", 3, "im2col_w">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05CpMulticastEnum : I32EnumAttr<"Tcgen05CpMulticast", "tcgen05 cp multicast", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"WARPX2_02_13", 1, "warpx2_02_13">,
    I32EnumAttrCase<"WARPX2_01_23", 2, "warpx2_01_23">,
    I32EnumAttrCase<"WARPX4", 3, "warpx4">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05CpShapeEnum : I32EnumAttr<"Tcgen05CpShape", "tcgen05 cp shapes", [
    I32EnumAttrCase<"SHAPE_128x256b", 0, "shape_128x256b">,
    I32EnumAttrCase<"SHAPE_4x256b", 1, "shape_4x256b">,
    I32EnumAttrCase<"SHAPE_128x128b", 2, "shape_128x128b">,
    I32EnumAttrCase<"SHAPE_64x128b", 3, "shape_64x128b">,
    I32EnumAttrCase<"SHAPE_32x128b", 4, "shape_32x128b">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05CpSrcFormatEnum : I32EnumAttr<"Tcgen05CpSrcFormat", "tcgen05 cp source format", [
    I32EnumAttrCase<"B6x16_P32", 0, "b6x16_p32">,
    I32EnumAttrCase<"B4x16_P64", 1, "b4x16_p64">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05FenceKindEnum : I32EnumAttr<"Tcgen05FenceKind", "NVVM Tcgen05 fence kind", [
    I32EnumAttrCase<"BEFORE_THREAD_SYNC", 0, "before">,
    I32EnumAttrCase<"AFTER_THREAD_SYNC", 1, "after">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05LdStShapeEnum : I32EnumAttr<"Tcgen05LdStShape", "allowed 32-bit signless integer cases: 0, 1, 2, 3, 4", [
    I32EnumAttrCase<"SHAPE_16X64B", 0, "shape_16x64b">,
    I32EnumAttrCase<"SHAPE_16X128B", 1, "shape_16x128b">,
    I32EnumAttrCase<"SHAPE_16X256B", 2, "shape_16x256b">,
    I32EnumAttrCase<"SHAPE_32X32B", 3, "shape_32x32b">,
    I32EnumAttrCase<"SHAPE_16X32BX2", 4, "shape_16x32bx2">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05MMABlockScaleEnum : I32EnumAttr<"Tcgen05MMABlockScale", "tcgen05.mma block scale attribute", [
    I32EnumAttrCase<"DEFAULT", 0, "default">,
    I32EnumAttrCase<"BLOCK16", 1, "block16">,
    I32EnumAttrCase<"BLOCK32", 2, "block32">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05MMACollectorBBufferEnum : I32EnumAttr<"Tcgen05MMACollectorBBuffer", "tcgen05 MMA Collector Buffer B Attribute", [
    I32EnumAttrCase<"B0", 0, "b0">,
    I32EnumAttrCase<"B1", 1, "b1">,
    I32EnumAttrCase<"B2", 2, "b2">,
    I32EnumAttrCase<"B3", 3, "b3">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05MMACollectorOpEnum : I32EnumAttr<"Tcgen05MMACollectorOp", "tcgen05.mma Collector Buffer Operation", [
    I32EnumAttrCase<"DISCARD", 0, "discard">,
    I32EnumAttrCase<"LASTUSE", 1, "lastuse">,
    I32EnumAttrCase<"FILL", 2, "fill">,
    I32EnumAttrCase<"USE", 3, "use">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05MMAKindEnum : I32EnumAttr<"Tcgen05MMAKind", "tcgen05 MMA Supported Types", [
    I32EnumAttrCase<"F16", 0, "f16">,
    I32EnumAttrCase<"TF32", 1, "tf32">,
    I32EnumAttrCase<"F8F6F4", 2, "f8f6f4">,
    I32EnumAttrCase<"I8", 3, "i8">,
    I32EnumAttrCase<"MXF8F6F4", 4, "mxf8f6f4">,
    I32EnumAttrCase<"MXF4", 5, "mxf4">,
    I32EnumAttrCase<"MXF4NVF4", 6, "mxf4nvf4">,
    I32EnumAttrCase<"TI16", 7, "ti16">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_Tcgen05WaitKindEnum : I32EnumAttr<"Tcgen05WaitKind", "NVVM Tcgen05 wait kind", [
    I32EnumAttrCase<"LOAD", 0, "load">,
    I32EnumAttrCase<"STORE", 1, "store">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TensormapElemtypeEnum : I32EnumAttr<"TensormapElemtype", "NVVM Tensormap Elemtype", [
    I32EnumAttrCase<"U8", 0, "u8">,
    I32EnumAttrCase<"U16", 1, "u16">,
    I32EnumAttrCase<"U32", 2, "u32">,
    I32EnumAttrCase<"S32", 3, "s32">,
    I32EnumAttrCase<"U64", 4, "u64">,
    I32EnumAttrCase<"S64", 5, "s64">,
    I32EnumAttrCase<"F16", 6, "f16">,
    I32EnumAttrCase<"F32", 7, "f32">,
    I32EnumAttrCase<"F32_FTZ", 8, "f32.ftz">,
    I32EnumAttrCase<"F64", 9, "f64">,
    I32EnumAttrCase<"BF16", 10, "bf16">,
    I32EnumAttrCase<"TF32", 11, "tf32">,
    I32EnumAttrCase<"TF32_FTZ", 12, "tf32.ftz">,
    I32EnumAttrCase<"B4x16", 13, "b4x16">,
    I32EnumAttrCase<"B4x16P64", 14, "b4x16_p64">,
    I32EnumAttrCase<"B6x16P32", 15, "b6x16_p32">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TensormapFieldEnum : I32EnumAttr<"TensormapField", "NVVM Tensormap Field Kind", [
    I32EnumAttrCase<"GLOBAL_ADDRESS", 0, "global_address">,
    I32EnumAttrCase<"RANK", 1, "rank">,
    I32EnumAttrCase<"BOX_DIM", 2, "box_dim">,
    I32EnumAttrCase<"GLOBAL_DIM", 3, "global_dim">,
    I32EnumAttrCase<"GLOBAL_STRIDE", 4, "global_stride">,
    I32EnumAttrCase<"ELEMENT_STRIDE", 5, "element_stride">,
    I32EnumAttrCase<"ELEMTYPE", 6, "elemtype">,
    I32EnumAttrCase<"INTERLEAVE_LAYOUT", 7, "interleave_layout">,
    I32EnumAttrCase<"SWIZZLE_MODE", 8, "swizzle_mode">,
    I32EnumAttrCase<"SWIZZLE_ATOMICITY", 9, "swizzle_atomicity">,
    I32EnumAttrCase<"FILL_MODE", 10, "fill_mode">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TensormapFillModeEnum : I32EnumAttr<"TensormapFillMode", "NVVM Tensormap Fill Mode", [
    I32EnumAttrCase<"ZERO", 0, "zero">,
    I32EnumAttrCase<"OOB_NAN", 1, "oob_nan">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TensormapInterleaveLayoutEnum : I32EnumAttr<"TensormapInterleaveLayout", "NVVM Tensormap Interleave Layout", [
    I32EnumAttrCase<"NO_INTERLEAVE", 0, "no_interleave">,
    I32EnumAttrCase<"B16", 1, "b16">,
    I32EnumAttrCase<"B32", 2, "b32">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TensormapSwizzleAtomicityEnum : I32EnumAttr<"TensormapSwizzleAtomicity", "NVVM Tensormap Swizzle Atomicity", [
    I32EnumAttrCase<"B16", 0, "b16">,
    I32EnumAttrCase<"B32", 1, "b32">,
    I32EnumAttrCase<"B32_FLIP_8B", 2, "b32_flip_b8">,
    I32EnumAttrCase<"B64", 3, "b64">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TensormapSwizzleModeEnum : I32EnumAttr<"TensormapSwizzleMode", "NVVM Tensormap Swizzle Mode", [
    I32EnumAttrCase<"NO_SWIZZLING", 0, "no_swizzling">,
    I32EnumAttrCase<"B32", 1, "b32">,
    I32EnumAttrCase<"B64", 2, "b64">,
    I32EnumAttrCase<"B128", 3, "b128">,
    I32EnumAttrCase<"B96", 4, "b96">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_TmemLayoutEnum : I32EnumAttr<"TmemLayout", "Tensor Memory Layout Enumerated Type", [
    I32EnumAttrCase<"tmem_16dp_128bit", 0, "tmem_16dp_128bit">,
    I32EnumAttrCase<"tmem_16dp_256bit", 1, "tmem_16dp_256bit">,
    I32EnumAttrCase<"tmem_32dp_32bit", 2, "tmem_32dp_32bit">,
    I32EnumAttrCase<"tmem_16dp_64bit", 3, "tmem_16dp_64bit">,
    I32EnumAttrCase<"tmem_16dp_32bit_t0_t15", 4, "tmem_16dp_32bit_t0_t15">,
    I32EnumAttrCase<"tmem_16dp_32bit_t16_t31", 5, "tmem_16dp_32bit_t16_t31">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_ValidatePatternEnum : I32EnumAttr<"ValidatePattern", "NVVM validate data pattern", [
    I32EnumAttrCase<"NONE", 0, "none">,
    I32EnumAttrCase<"PER_16BYTES_80000000", 1, "per_16bytes_80000000">,
    I32EnumAttrCase<"PER_16BYTES_8000", 2, "per_16bytes_8000">,
    I32EnumAttrCase<"PER_16BYTES_80", 3, "per_16bytes_80">,
    I32EnumAttrCase<"PER_16BYTES_8", 4, "per_16bytes_8">,
    I32EnumAttrCase<"PER_ELEMENT_FF", 5, "per_element_ff">,
    I32EnumAttrCase<"PER_ELEMENT_80000000", 6, "per_element_80000000">,
    I32EnumAttrCase<"PER_ELEMENT_8000", 7, "per_element_8000">,
    I32EnumAttrCase<"PER_ELEMENT_80", 8, "per_element_80">,
    I32EnumAttrCase<"PER_ELEMENT_8", 9, "per_element_8">,
    I32EnumAttrCase<"PER_16BYTES_80000000_PAIRED", 10, "per_16bytes_80000000_paired">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_VoteSyncKindEnum : I32EnumAttr<"VoteSyncKind", "NVVM vote sync kind", [
    I32EnumAttrCase<"any", 0, "any">,
    I32EnumAttrCase<"all", 1, "all">,
    I32EnumAttrCase<"ballot", 2, "ballot">,
    I32EnumAttrCase<"uni", 3, "uni">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_WGMMAScaleInEnum : I32EnumAttr<"WGMMAScaleIn", "WGMMA overflow options", [
    I32EnumAttrCase<"one", 1, "one">,
    I32EnumAttrCase<"neg", 2, "neg">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_WGMMAScaleOutEnum : I32EnumAttr<"WGMMAScaleOut", "WGMMA input predicate", [
    I32EnumAttrCase<"zero", 0, "zero">,
    I32EnumAttrCase<"one", 1, "one">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_WGMMATypesEnum : I32EnumAttr<"WGMMATypes", "NVVM WGMMA types", [
    I32EnumAttrCase<"f16", 0, "f16">,
    I32EnumAttrCase<"tf32", 1, "tf32">,
    I32EnumAttrCase<"u8", 2, "u8">,
    I32EnumAttrCase<"s8", 3, "s8">,
    I32EnumAttrCase<"b1", 4, "b1">,
    I32EnumAttrCase<"bf16", 5, "bf16">,
    I32EnumAttrCase<"e4m3", 6, "e4m3">,
    I32EnumAttrCase<"e5m2", 7, "e5m2">,
    I32EnumAttrCase<"f32", 8, "f32">,
    I32EnumAttrCase<"s32", 9, "s32">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let genSpecializedAttr = 0;
}

def NVVM_AtomicOpKindAttr : EnumAttr<NVVM_Dialect, NVVM_AtomicOpKindEnum, "atomic_op"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_BarrierReductionAttr : EnumAttr<NVVM_Dialect, NVVM_BarrierReductionEnum, "reduction"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_BarrierReduxKindAttr : EnumAttr<NVVM_Dialect, NVVM_BarrierReduxKindEnum, "barrier_redux_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_BlockScaleFormatAttr : EnumAttr<NVVM_Dialect, NVVM_BlockScaleFormatEnum, "block_scale_format"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_CTAGroupKindAttr : EnumAttr<NVVM_Dialect, NVVM_CTAGroupKindEnum, "cta_group"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_CVTPackFloatKindAttr : EnumAttr<NVVM_Dialect, NVVM_CVTPackFloatKindEnum, "packfloat_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_CacheEvictionPriorityAttr : EnumAttr<NVVM_Dialect, NVVM_CacheEvictionPriorityEnum, "cache_eviction_priority"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_ClusterLaunchControlQueryTypeAttr : EnumAttr<NVVM_Dialect, NVVM_ClusterLaunchControlQueryTypeEnum, "cluster_launch_control_query_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_CompareOpKindAttr : EnumAttr<NVVM_Dialect, NVVM_CompareOpKindEnum, "op"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ConvertFP4TypeAttr : EnumAttr<NVVM_Dialect, NVVM_ConvertFP4TypeEnum, "convert_fp4_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ConvertFP8TypeAttr : EnumAttr<NVVM_Dialect, NVVM_ConvertFP8TypeEnum, "convert_fp8_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ConvertScaleKindAttr : EnumAttr<NVVM_Dialect, NVVM_ConvertScaleKindEnum, "convert_scale_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_DotAccumulateTypeAttr : EnumAttr<NVVM_Dialect, NVVM_DotAccumulateTypeEnum, "dot_accumulate_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_EvictKindAttr : EnumAttr<NVVM_Dialect, NVVM_EvictKindEnum, "evict_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_FPRoundingModeAttr : EnumAttr<NVVM_Dialect, NVVM_FPRoundingModeEnum, "fp_rnd_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_GridDepActionAttr : EnumAttr<NVVM_Dialect, NVVM_GridDepActionKindEnum, "grid_dep_action"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_IntegerRoundingModeAttr : EnumAttr<NVVM_Dialect, NVVM_IntegerRoundingModeEnum, "int_rnd_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_L2PrefetchSizeAttr : EnumAttr<NVVM_Dialect, NVVM_L2PrefetchSizeEnum, "l2_prefetch"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_LdStMatrixEltTypeAttr : EnumAttr<NVVM_Dialect, NVVM_LdStMatrixEltTypeEnum, "ld_st_matrix_elt_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_LoadCacheModifierAttr : EnumAttr<NVVM_Dialect, NVVM_LoadCacheModifierKindEnum, "load_cache_modifier"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_LoadCacheModifierAttrExt : EnumAttr<NVVM_Dialect, NVVM_LoadCacheModifierExtKindEnum, "load_cache_modifier_ext"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_LoadShapeAttr : EnumAttr<NVVM_Dialect, NVVM_LoadShapeEnum, "load_shape"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_LoadSrcFormatAttr : EnumAttr<NVVM_Dialect, NVVM_LoadSrcFormatEnum, "load_src_format"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MBarrierLayoutAttr : EnumAttr<NVVM_Dialect, NVVM_MBarrierLayoutEnum, "mbarrier_layout"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MBarrierScopeKindAttr : EnumAttr<NVVM_Dialect, NVVM_MBarrierScopeKindEnum, "mbar_scope"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MBarrierSpaceKindAttr : EnumAttr<NVVM_Dialect, NVVM_MBarrierSpaceKindEnum, "mbar_space"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MBarrierTxnKindAttr : EnumAttr<NVVM_Dialect, NVVM_MBarrierTxnKindEnum, "mbar_txn_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MBarrierWaitKindAttr : EnumAttr<NVVM_Dialect, NVVM_MBarrierWaitKindEnum, "mbar_wait"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMAB1OpAttr : EnumAttr<NVVM_Dialect, NVVM_MMAB1OpEnum, "mma_b1op"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMABlockScaleKindAttr : EnumAttr<NVVM_Dialect, NVVM_MMABlockScaleKindEnum, "block_scale_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMACtaCountAttr : EnumAttr<NVVM_Dialect, NVVM_MMACtaCountEnum, "mma_cta_count"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMAFragAttr : EnumAttr<NVVM_Dialect, NVVM_MMAFragEnum, "mma_frag"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMAIntOverflowAttr : EnumAttr<NVVM_Dialect, NVVM_MMAIntOverflowEnum, "mma_int_overflow"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMAKindAttr : EnumAttr<NVVM_Dialect, NVVM_MMAKindEnum, "mma_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMALayoutAttr : EnumAttr<NVVM_Dialect, NVVM_MMALayoutEnum, "mma_layout"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MMATypesAttr : EnumAttr<NVVM_Dialect, NVVM_MMATypesEnum, "mma_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MatchSyncKindAttr : EnumAttr<NVVM_Dialect, NVVM_MatchSyncKindEnum, "match_sync_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_MemOrderKindAttr : EnumAttr<NVVM_Dialect, NVVM_MemOrderKindEnum, "mem_order"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MemScopeKindAttr : EnumAttr<NVVM_Dialect, NVVM_MemScopeKindEnum, "mem_scope"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_MulModeAttr : EnumAttr<NVVM_Dialect, NVVM_MulModeEnum, "mul_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_NVVMMemorySpaceAttr : EnumAttr<NVVM_Dialect, NVVM_NVVMMemorySpaceEnum, "memory_space"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_NVVM_MBarrierPhaseAttr : EnumAttr<NVVM_Dialect, NVVM_MBarrierPhaseEnum, "mbarrier_phase"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_PermuteModeAttr : EnumAttr<NVVM_Dialect, NVVM_PermuteModeEnum, "permute_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_PrefetchCacheLevelAttr : EnumAttr<NVVM_Dialect, NVVM_PrefetchCacheLevelEnum, "prefetch_cache_level"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_ProxyKindAttr : EnumAttr<NVVM_Dialect, NVVM_ProxyKindEnum, "proxy_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ReductionKindAttr : EnumAttr<NVVM_Dialect, NVVM_ReductionKindEnum, "reduction_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_ReductionOpAttr : EnumAttr<NVVM_Dialect, NVVM_ReductionOpEnum, "red_op"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ReductionTypeAttr : EnumAttr<NVVM_Dialect, NVVM_ReductionTypeEnum, "red_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPCompressElemSizeAttr : EnumAttr<NVVM_Dialect, NVVM_SPCompressElemSizeEnum, "spcomp_elem_size"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPCompressFactorTypeAttr : EnumAttr<NVVM_Dialect, NVVM_SPCompressFactorTypeEnum, "spcomp_factor"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPCompressIndexSizeAttr : EnumAttr<NVVM_Dialect, NVVM_SPCompressIndexSizeEnum, "spcomp_index_size"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPCompressOpKindAttr : EnumAttr<NVVM_Dialect, NVVM_SPCompressOpKindEnum, "spcomp_op_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPCompressRepFactorAttr : EnumAttr<NVVM_Dialect, NVVM_SPCompressRepFactorEnum, "spcomp_rep_factor"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPDecompressElemSizeAttr : EnumAttr<NVVM_Dialect, NVVM_SPDecompressElemSizeEnum, "spdecomp_elem_size"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPDecompressFactorTypeAttr : EnumAttr<NVVM_Dialect, NVVM_SPDecompressFactorTypeEnum, "spdecomp_factor"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPDecompressIndexSizeAttr : EnumAttr<NVVM_Dialect, NVVM_SPDecompressIndexSizeEnum, "spdecomp_index_size"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SPDecompressRepFactorAttr : EnumAttr<NVVM_Dialect, NVVM_SPDecompressRepFactorEnum, "spdecomp_rep_factor"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SaturationModeAttr : EnumAttr<NVVM_Dialect, NVVM_SaturationModeEnum, "sat_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SaturationModeKindAttr : EnumAttr<NVVM_Dialect, NVVM_SaturationModeKindEnum, "sat"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ScaleVecSizeAttr : EnumAttr<NVVM_Dialect, NVVM_ScaleVecSizeEnum, "scale_vec_size"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_SetMaxRegisterActionAttr : EnumAttr<NVVM_Dialect, NVVM_SetMaxRegisterActionEnum, "action"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_SharedSpaceAttr : EnumAttr<NVVM_Dialect, NVVM_SharedSpaceEnum, "shared_space"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ShflKindAttr : EnumAttr<NVVM_Dialect, NVVM_ShflKindEnum, "shfl_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_SparsityFormatAttr : EnumAttr<NVVM_Dialect, NVVM_SparsityFormatEnum, "sparsity_format"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_StateSpaceAttr : EnumAttr<NVVM_Dialect, NVVM_StateSpaceEnum, "state_space"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_StoreCacheModifierAttr : EnumAttr<NVVM_Dialect, NVVM_StoreCacheModifierKindEnum, "store_cache_modifier"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_StoreShapeAttr : EnumAttr<NVVM_Dialect, NVVM_StoreShapeEnum, "store_shape"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TCBarParamEnumAttr : EnumAttr<NVVM_Dialect, NVVM_TCBarParamEnum, "TCBarParam"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TMALoadModeAttr : EnumAttr<NVVM_Dialect, NVVM_TMALoadModeEnum, "tma_load_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TMAReduxKindAttr : EnumAttr<NVVM_Dialect, NVVM_TMAReduxKindEnum, "tma_redux_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TMAStoreModeAttr : EnumAttr<NVVM_Dialect, NVVM_TMAStoreModeEnum, "tma_store_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05CpMulticastAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05CpMulticastEnum, "tcgen05_cp_multicast"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05CpShapeAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05CpShapeEnum, "tcgen05_cp_shape"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05CpSrcFormatAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05CpSrcFormatEnum, "tcgen05_cp_src_fmt"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05FenceKindAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05FenceKindEnum, "tcgen05_fence"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05LdStShapeAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05LdStShapeEnum, "tcgen05_ldst_shape"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05MMABlockScaleAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05MMABlockScaleEnum, "tcgen05_mma_block_scale"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05MMACollectorBBufferAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05MMACollectorBBufferEnum, "tcgen05_mma_collectorb"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05MMACollectorOpAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05MMACollectorOpEnum, "tcgen05_mma_collectorop"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05MMAKindAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05MMAKindEnum, "tcgen05_mma_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_Tcgen05WaitKindAttr : EnumAttr<NVVM_Dialect, NVVM_Tcgen05WaitKindEnum, "tcgen05_wait"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TensormapElemtypeAttr : EnumAttr<NVVM_Dialect, NVVM_TensormapElemtypeEnum, "tensormap_elemtype"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TensormapFieldAttr : EnumAttr<NVVM_Dialect, NVVM_TensormapFieldEnum, "tensormap_field"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_TensormapFillModeAttr : EnumAttr<NVVM_Dialect, NVVM_TensormapFillModeEnum, "tensormap_fill_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TensormapInterleaveLayoutAttr : EnumAttr<NVVM_Dialect, NVVM_TensormapInterleaveLayoutEnum, "tensormap_interleave_layout"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TensormapSwizzleAtomicityAttr : EnumAttr<NVVM_Dialect, NVVM_TensormapSwizzleAtomicityEnum, "tensormap_swizzle_atomicity"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TensormapSwizzleModeAttr : EnumAttr<NVVM_Dialect, NVVM_TensormapSwizzleModeEnum, "tensormap_swizzle_mode"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_TmemLayoutEnumAttr : EnumAttr<NVVM_Dialect, NVVM_TmemLayoutEnum, "TmemLayout"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_ValidatePatternAttr : EnumAttr<NVVM_Dialect, NVVM_ValidatePatternEnum, "validate_pattern"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_VoteSyncKindAttr : EnumAttr<NVVM_Dialect, NVVM_VoteSyncKindEnum, "vote_sync_kind"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let assemblyFormat = "$value";
}

def NVVM_WGMMAScaleInAttr : EnumAttr<NVVM_Dialect, NVVM_WGMMAScaleInEnum, "wgmma_scale_in"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_WGMMAScaleOutAttr : EnumAttr<NVVM_Dialect, NVVM_WGMMAScaleOutEnum, "wgmma_scale_out"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

def NVVM_WGMMATypesAttr : EnumAttr<NVVM_Dialect, NVVM_WGMMATypesEnum, "wgmma_type"> {
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
}

#endif // CUTE_IR_GENERATED_NVVM_ENUMS_TD
