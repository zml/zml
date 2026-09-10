//===- CuteNVGPUTypes.td - CuTe NVIDIA GPU types -----------*- tablegen -*-===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_TYPES_TD
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_TYPES_TD

include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.td"

def CuteNVGPU_SmemDescType : CuteNVGPU_Type<"SmemDesc", "smem_desc"> {
  let summary = "Unparameterized shared-memory descriptor";
  let assemblyFormat = "";
}

def CuteNVGPU_TmaDescriptorTiledType
    : CuteNVGPU_Type<"TmaDescriptorTiled", "tma_descriptor_tiled"> {
  let summary = "Unparameterized tiled TMA descriptor";
  let assemblyFormat = "";
}

def CuteNVGPU_TmaDescriptorIm2ColType
    : CuteNVGPU_Type<"TmaDescriptorIm2Col", "tma_descriptor_im2col"> {
  let summary = "Unparameterized im2col TMA descriptor";
  let assemblyFormat = "";
}

def CuteNVGPU_WorkIdResponseType
    : CuteNVGPU_Type<"WorkIdResponse", "workid_response"> {
  let summary = "Tensor-memory work identifier response";
  let assemblyFormat = "";
}

class CuteNVGPU_PayloadType<string name, string mnemonic>
    : CuteNVGPU_Type<name, mnemonic> {
  let summary = "CuTe-DSL NVIDIA GPU type with lossless architecture payload";
  let parameters = (ins
    "mlir::StringAttr":$payload
  );
  let hasCustomAssemblyFormat = 1;
}

def CuteNVGPU_CopyAtomBulkCopyG2SType
    : CuteNVGPU_PayloadType<"CopyAtomBulkCopyG2S", "atom.bulk_copy_g2s">;
def CuteNVGPU_CopyAtomBulkCopyS2GType
    : CuteNVGPU_PayloadType<"CopyAtomBulkCopyS2G", "atom.bulk_copy_s2g">;
def CuteNVGPU_CopyAtomBulkCopyS2SType
    : CuteNVGPU_PayloadType<"CopyAtomBulkCopyS2S", "atom.bulk_copy_s2s">;
def CuteNVGPU_CopyAtomDsmemStoreType
    : CuteNVGPU_PayloadType<"CopyAtomDsmemStore", "atom.dsmem_store">;
def CuteNVGPU_CopyAtomG2RType
    : CuteNVGPU_PayloadType<"CopyAtomG2R", "atom.g2r">;
def CuteNVGPU_CopyAtomLdsmType
    : CuteNVGPU_PayloadType<"CopyAtomLdsm", "atom.ldsm">;
def CuteNVGPU_CopyAtomNonExec2DGather4TmaLoadType
    : CuteNVGPU_PayloadType<"CopyAtomNonExec2DGather4TmaLoad",
                            "atom.non_exec_2d_gather4_tma_load">;
def CuteNVGPU_CopyAtomNonExecIm2ColTmaLoadType
    : CuteNVGPU_PayloadType<"CopyAtomNonExecIm2ColTmaLoad",
                            "atom.non_exec_im2col_tma_load">;
def CuteNVGPU_CopyAtomNonExecIm2ColTmaStoreType
    : CuteNVGPU_PayloadType<"CopyAtomNonExecIm2ColTmaStore",
                            "atom.non_exec_im2col_tma_store">;
def CuteNVGPU_CopyAtomNonExecTiledTmaLoadType
    : CuteNVGPU_PayloadType<"CopyAtomNonExecTiledTmaLoad",
                            "atom.non_exec_tiled_tma_load">;
def CuteNVGPU_CopyAtomNonExecTiledTmaReduceType
    : CuteNVGPU_PayloadType<"CopyAtomNonExecTiledTmaReduce",
                            "atom.non_exec_tiled_tma_reduce">;
def CuteNVGPU_CopyAtomNonExecTiledTmaStoreType
    : CuteNVGPU_PayloadType<"CopyAtomNonExecTiledTmaStore",
                            "atom.non_exec_tiled_tma_store">;
def CuteNVGPU_CopyAtomR2GType
    : CuteNVGPU_PayloadType<"CopyAtomR2G", "atom.r2g">;
def CuteNVGPU_CopyAtomR2SType
    : CuteNVGPU_PayloadType<"CopyAtomR2S", "atom.r2s">;
def CuteNVGPU_CopyAtomS2RType
    : CuteNVGPU_PayloadType<"CopyAtomS2R", "atom.s2r">;
def CuteNVGPU_CopyAtomSIMTAsyncCopyType
    : CuteNVGPU_PayloadType<"CopyAtomSIMTAsyncCopy", "atom.simt_async_copy">;
def CuteNVGPU_CopyAtomSIMTSyncCopyType
    : CuteNVGPU_PayloadType<"CopyAtomSIMTSyncCopy", "atom.universal_copy">;
def CuteNVGPU_CopyAtomSM100CopyS2TType
    : CuteNVGPU_PayloadType<"CopyAtomSM100CopyS2T", "atom.s2t_copy">;
def CuteNVGPU_CopyAtomSM100TmemLoadType
    : CuteNVGPU_PayloadType<"CopyAtomSM100TmemLoad", "atom.tmem_load">;
def CuteNVGPU_CopyAtomSM100TmemStoreType
    : CuteNVGPU_PayloadType<"CopyAtomSM100TmemStore", "atom.tmem_store">;
def CuteNVGPU_CopyAtomSM10xTmemLoadRedType
    : CuteNVGPU_PayloadType<"CopyAtomSM10xTmemLoadRed", "atom.tmem_load_red">;
def CuteNVGPU_CopyAtomStsmType
    : CuteNVGPU_PayloadType<"CopyAtomStsm", "atom.stsm">;
def CuteNVGPU_MmaAtomSM100UMMABlockScaledType
    : CuteNVGPU_PayloadType<"MmaAtomSM100UMMABlockScaled", "sm100.mma_bs">;
def CuteNVGPU_MmaAtomSM100UMMAType
    : CuteNVGPU_PayloadType<"MmaAtomSM100UMMA", "sm100.mma">;
def CuteNVGPU_MmaAtomSM120BlockScaledType
    : CuteNVGPU_PayloadType<"MmaAtomSM120BlockScaled", "SM120.mma_bs">;
def CuteNVGPU_MmaAtomSM80Type
    : CuteNVGPU_PayloadType<"MmaAtomSM80", "sm80.mma">;
def CuteNVGPU_MmaAtomSM89Type
    : CuteNVGPU_PayloadType<"MmaAtomSM89", "sm89.mma">;
def CuteNVGPU_MmaAtomSM90Type
    : CuteNVGPU_PayloadType<"MmaAtomSM90", "sm90.mma">;
def CuteNVGPU_SmemDescViewType
    : CuteNVGPU_PayloadType<"SmemDescView", "smem_desc_view">;
def CuteNVGPU_TiledCopyType
    : CuteNVGPU_PayloadType<"TiledCopy", "tiled_copy">;
def CuteNVGPU_TiledMmaType
    : CuteNVGPU_PayloadType<"TiledMma", "tiled_mma">;
def CuteNVGPU_UniversalFmaAtomType
    : CuteNVGPU_PayloadType<"UniversalFmaAtom", "atom.universal_fma">;
def CuteNVGPU_CopyAtomIm2ColTmaLoadType
    : CuteNVGPU_PayloadType<"CopyAtomIm2ColTmaLoad", "atom.im2col_tma_load">;
def CuteNVGPU_CopyAtomIm2ColTmaStoreType
    : CuteNVGPU_PayloadType<"CopyAtomIm2ColTmaStore", "atom.im2col_tma_store">;
def CuteNVGPU_CopyAtomSIMTMultimemLdReduceType
    : CuteNVGPU_PayloadType<"CopyAtomSIMTMultimemLdReduce",
                            "atom.simt_multimem_ld_reduce">;
def CuteNVGPU_CopyAtomSIMTMultimemRedType
    : CuteNVGPU_PayloadType<"CopyAtomSIMTMultimemRed", "atom.simt_multimem_red">;
def CuteNVGPU_CopyAtomSIMTMultimemStType
    : CuteNVGPU_PayloadType<"CopyAtomSIMTMultimemSt", "atom.simt_multimem_st">;
def CuteNVGPU_CopyAtomSM100S2TCopyV2Type
    : CuteNVGPU_PayloadType<"CopyAtomSM100S2TCopyV2", "atom.sm100_s2t_copy_v2">;
def CuteNVGPU_CopyAtomTmaLoadType
    : CuteNVGPU_PayloadType<"CopyAtomTmaLoad", "atom.tma_load">;
def CuteNVGPU_CopyAtomTmaReduceType
    : CuteNVGPU_PayloadType<"CopyAtomTmaReduce", "atom.tma_reduce">;
def CuteNVGPU_CopyAtomTmaStoreType
    : CuteNVGPU_PayloadType<"CopyAtomTmaStore", "atom.tma_store">;
def CuteNVGPU_MmaAtomSM100UMMABlockScaledSparseType
    : CuteNVGPU_PayloadType<"MmaAtomSM100UMMABlockScaledSparse",
                            "sm100.mma_bs_sp">;
def CuteNVGPU_MmaAtomSM100UMMASparseType
    : CuteNVGPU_PayloadType<"MmaAtomSM100UMMASparse", "sm100.mma_sp">;
def CuteNVGPU_MmaAtomSM80SparseType
    : CuteNVGPU_PayloadType<"MmaAtomSM80Sparse", "sm80.sparse_mma">;
def CuteNVGPU_SmemDescCircularSM103Type
    : CuteNVGPU_PayloadType<"SmemDescCircularSM103",
                            "sm103.smem_desc_circular">;

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_TYPES_TD
