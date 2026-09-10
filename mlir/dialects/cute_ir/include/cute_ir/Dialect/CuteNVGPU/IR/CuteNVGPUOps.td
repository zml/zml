//===- CuteNVGPUOps.td - CuTe NVIDIA GPU operations --------*- tablegen -*-===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_OPS_TD
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_OPS_TD

include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.td"

class CuteNVGPU_SchemaOp<string mnemonic, list<Trait> traits = []>
    : CuteNVGPU_Op<mnemonic, traits> {
  let summary = "Operation recovered from the CuTe-DSL operation registry";
}

def CuteNVGPU_ArchAllocRmemOp : CuteNVGPU_SchemaOp<"arch.alloc_rmem"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$ptr);
}

def CuteNVGPU_ArchAllocSmemOp : CuteNVGPU_SchemaOp<"arch.alloc_smem"> {
  let arguments = (ins I32Attr:$input);
  let results = (outs AnyType:$ptr);
}

def CuteNVGPU_ArchCopySM90BulkCopyG2SOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM90.bulk_copy_g2s",
                         [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$gmem_data_addr, AnyType:$dsmem_data_addr,
                       AnyType:$dsmem_bar_addr,
                       Optional<AnyInteger>:$multicast_mask,
                       Optional<AnyInteger>:$cache_policy, I32Attr:$size);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM90BulkCopyS2GOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM90.bulk_copy_s2g",
                         [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$smem_data_addr, AnyType:$gmem_data_addr,
                       Optional<AnyInteger>:$byte_mask,
                       Optional<AnyInteger>:$cache_policy, I32Attr:$size);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM90BulkCopyS2SOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM90.bulk_copy_s2s"> {
  let arguments = (ins AnyType:$smem_data_addr, AnyType:$dsmem_data_addr,
                       AnyType:$dsmem_bar_addr, AnyInteger:$cta_rank,
                       I32Attr:$size);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM100CopyS2TOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM100.copy_s2t"> {
  let arguments = (ins AnyType:$src, AnyType:$dst, I32Attr:$dp, I32Attr:$bits,
                       I32Attr:$cta, AnyAttr:$broadcast);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM100TmaLoadOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM100.tma_load",
                         [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$src_desc, AnyType:$dsmem_data_addr,
                       AnyType:$dsmem_bar_addr, Variadic<AnyType>:$coord,
                       Optional<AnyInteger>:$multicast_mask,
                       Variadic<AnyType>:$offsets,
                       Optional<AnyInteger>:$cache_policy,
                       AnyAttr:$mode, I32Attr:$num_cta);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM100TmaReduceOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM100.tma_reduce",
                         [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$dst_desc, AnyType:$src_smem_data_addr,
                       Variadic<AnyType>:$coord,
                       Optional<AnyInteger>:$cache_policy,
                       AnyAttr:$mode, AnyAttr:$kind);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM100TmaStoreOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM100.tma_store",
                         [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$dst_desc, AnyType:$src_smem_data_addr,
                       Variadic<AnyType>:$coord,
                       Optional<AnyInteger>:$cache_policy, AnyAttr:$mode);
  let results = (outs);
}

def CuteNVGPU_ArchCopySM100TmemLoadOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM100.tmem_load"> {
  let arguments = (ins AnyType:$src, I32Attr:$num_dp, I32Attr:$num_b,
                       I32Attr:$num_rep, OptionalAttr<UnitAttr>:$pack_16);
  let results = (outs AnyType:$dst);
}

def CuteNVGPU_ArchCopySM100TmemStoreOp
    : CuteNVGPU_SchemaOp<"arch.copy.SM100.tmem_store"> {
  let arguments = (ins AnyType:$dst, AnyType:$src, I32Attr:$num_dp,
                       I32Attr:$num_b, I32Attr:$num_rep,
                       OptionalAttr<UnitAttr>:$expand_16);
  let results = (outs);
}

def CuteNVGPU_ArchGetDynSmemOp
    : CuteNVGPU_SchemaOp<"arch.get_dyn_smem"> {
  let arguments = (ins);
  let results = (outs AnyType:$ptr);
}

def CuteNVGPU_ArchGetDynSmemSizeOp
    : CuteNVGPU_SchemaOp<"arch.get_dyn_smem_size"> {
  let arguments = (ins);
  let results = (outs I32:$size);
}

def CuteNVGPU_ArchMakeWarpUniformOp
    : CuteNVGPU_SchemaOp<"arch.make_warp_uniform"> {
  let arguments = (ins AnyInteger:$val);
  let results = (outs AnyInteger:$res);
}

def CuteNVGPU_ArchMmaSM120BlockScaledOp
    : CuteNVGPU_SchemaOp<"arch.mma.SM120.block_scaled",
                         [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<AnyType>:$opA, Variadic<AnyType>:$opB,
                       Variadic<AnyType>:$opC, AnyType:$sfA, AnyType:$sfB,
                       Optional<AnyInteger>:$byte_id_a,
                       Optional<AnyInteger>:$byte_id_b,
                       AnyAttr:$shape_MNK, I32Attr:$vec_size,
                       OptionalAttr<I16Attr>:$thread_id_a,
                       OptionalAttr<I16Attr>:$thread_id_b,
                       TypeAttr:$a_type, TypeAttr:$b_type, TypeAttr:$sf_type);
  let results = (outs Variadic<AnyType>:$res);
}

def CuteNVGPU_ArchPrefetchTmaDescOp
    : CuteNVGPU_SchemaOp<"arch.prefetch_tma_desc"> {
  let arguments = (ins AnyType:$tma_descriptor_ptr);
  let results = (outs);
}

def CuteNVGPU_ArchSM100AllocTmemOp
    : CuteNVGPU_SchemaOp<"arch.sm100.alloc_tmem"> {
  let arguments = (ins AnyInteger:$n_cols, AnyType:$dst_ptr,
                       OptionalAttr<UnitAttr>:$is_two_cta,
                       OptionalAttr<UnitAttr>:$exclusive);
  let results = (outs);
}

def CuteNVGPU_ArchSM100DeallocTmemOp
    : CuteNVGPU_SchemaOp<"arch.sm100.dealloc_tmem"> {
  let arguments = (ins AnyType:$tmem_ptr, AnyInteger:$n_cols,
                       OptionalAttr<UnitAttr>:$is_two_cta,
                       OptionalAttr<UnitAttr>:$exclusive);
  let results = (outs);
}

def CuteNVGPU_ArchSM100RelinquishTmemAllocPermitOp
    : CuteNVGPU_SchemaOp<"arch.sm100.relinquish_tmem_alloc_permit"> {
  let arguments = (ins OptionalAttr<UnitAttr>:$is_two_cta);
  let results = (outs);
}

def CuteNVGPU_ArchSM100RetrieveTmemPtrOp
    : CuteNVGPU_SchemaOp<"arch.sm100.retrieve_tmem_ptr"> {
  let arguments = (ins AnyType:$smem_ptr);
  let results = (outs AnyType:$tmem_ptr);
}

def CuteNVGPU_AtomMakeExecTmaOp
    : CuteNVGPU_SchemaOp<"atom.make_exec_tma"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomMakeNonExec2DGather4TmaLoadOp
    : CuteNVGPU_SchemaOp<"atom.make_non_exec_2d_gather4_tma_load"> {
  let arguments = (ins AnyType:$gmem_tensor, AnyType:$gmem_coord_layout,
                       AnyType:$smem_layout, AnyType:$cta_v_map,
                       AnyAttr:$kind, OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$non_exec_atom, AnyType:$tma_tensor);
}

def CuteNVGPU_AtomMakeNonExecIm2ColTmaLoadOp
    : CuteNVGPU_SchemaOp<"atom.make_non_exec_im2col_tma_load"> {
  let arguments = (ins AnyType:$gmem_tensor, AnyType:$smem_layout,
                       AnyType:$cta_v_map, AnyType:$lower_corner_whd,
                       AnyType:$upper_corner_whd, AnyType:$lower_padding_whd,
                       AnyType:$upper_padding_whd, AnyType:$stride_whd,
                       AnyType:$lower_srt, AnyType:$stride_srt,
                       AnyAttr:$kind, OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$non_exec_atom, AnyType:$tma_tensor);
}

def CuteNVGPU_AtomMakeNonExecIm2ColTmaStoreOp
    : CuteNVGPU_SchemaOp<"atom.make_non_exec_im2col_tma_store"> {
  let arguments = (ins AnyType:$gmem_tensor, AnyType:$smem_layout,
                       AnyType:$cta_v_map,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$non_exec_atom, AnyType:$tma_tensor);
}

def CuteNVGPU_AtomMakeNonExecTiledTmaLoadOp
    : CuteNVGPU_SchemaOp<"atom.make_non_exec_tiled_tma_load"> {
  let arguments = (ins AnyType:$gmem_tensor, AnyType:$smem_layout,
                       AnyType:$cta_v_map, AnyAttr:$kind,
                       OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$non_exec_atom, AnyType:$tma_tensor);
}

def CuteNVGPU_AtomMakeNonExecTiledTmaReduceOp
    : CuteNVGPU_SchemaOp<"atom.make_non_exec_tiled_tma_reduce"> {
  let arguments = (ins AnyType:$gmem_tensor, AnyType:$smem_layout,
                       AnyType:$cta_v_map, AnyAttr:$kind,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$non_exec_atom, AnyType:$tma_tensor);
}

def CuteNVGPU_AtomMakeNonExecTiledTmaStoreOp
    : CuteNVGPU_SchemaOp<"atom.make_non_exec_tiled_tma_store"> {
  let arguments = (ins AnyType:$gmem_tensor, AnyType:$smem_layout,
                       AnyType:$cta_v_map,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$non_exec_atom, AnyType:$tma_tensor);
}

def CuteNVGPU_AtomMakeS2TCopyOp
    : CuteNVGPU_SchemaOp<"atom.make_s2t_copy"> {
  let arguments = (ins AnyType:$copy_s2t_atom, AnyType:$tmem_memref);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomMakeTmemCopyOp
    : CuteNVGPU_SchemaOp<"atom.make_tmem_copy"> {
  let arguments = (ins AnyType:$atom, AnyType:$tmem_memref);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomGetCoordTensorOp
    : CuteNVGPU_SchemaOp<"atom.get_coord_tensor",
                         [SameVariadicOperandSize]> {
  let arguments = (ins AnyType:$atom, AnyType:$shape,
                       Variadic<AnyType>:$offset_b,
                       Variadic<AnyType>:$layout_b);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomGetCopyS2TSmemDescViewOp
    : CuteNVGPU_SchemaOp<"atom.get_copy_s2t_smem_desc_view"> {
  let arguments = (ins AnyType:$atom, AnyType:$view);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomGetValueOp
    : CuteNVGPU_SchemaOp<"atom.get_value"> {
  let arguments = (ins AnyType:$atom, AnyAttr:$field);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomSetValueOp
    : CuteNVGPU_SchemaOp<"atom.set_value"> {
  let arguments = (ins AnyType:$atom, AnyType:$value, AnyAttr:$field);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_AtomTmaPartitionOp
    : CuteNVGPU_SchemaOp<"atom.tma_partition"> {
  let arguments = (ins AnyType:$tma_atom, AnyType:$cta_coord,
                       AnyType:$cta_layout, AnyType:$smem_tensor,
                       Variadic<AnyType>:$target_tensors);
  let results = (outs AnyType:$res_smem_tensor,
                      Variadic<AnyType>:$res_target_tensors);
}

def CuteNVGPU_CastTmaDescToIntegerOp
    : CuteNVGPU_SchemaOp<"cast_tma_desc_to_integer"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyInteger:$dst);
}

def CuteNVGPU_CopyTmaDescOp : CuteNVGPU_SchemaOp<"copy_tma_desc"> {
  let arguments = (ins AnyType:$atom, AnyType:$tma_desc_ptr);
  let results = (outs);
}

def CuteNVGPU_GetGridConstantPointerOp
    : CuteNVGPU_SchemaOp<"get_grid_constant_pointer"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_GetTmaDescAddrOp
    : CuteNVGPU_SchemaOp<"get_tma_desc_addr"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$ptr);
}

def CuteNVGPU_MakeGmmaSmemDescOp
    : CuteNVGPU_SchemaOp<"make_gmma_smem_desc"> {
  let arguments = (ins AnyType:$src, AnyAttr:$layout, AnyAttr:$major);
  let results = (outs AnyType:$res);
}

def CuteNVGPU_MakeTmaDescIm2ColAtOp
    : CuteNVGPU_SchemaOp<"make_tma_desc_im2col_at"> {
  let arguments = (ins AnyType:$gmem_view, AnyType:$smem_layout,
                       AnyType:$cta_value_tile, AnyType:$lower_corner_whd,
                       AnyType:$upper_corner_whd, AnyType:$lower_padding_whd,
                       AnyType:$upper_padding_whd, AnyType:$stride_whd,
                       AnyType:$lower_srt, AnyType:$stride_srt,
                       AnyType:$tma_desc_addr,
                       OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs);
}

def CuteNVGPU_MakeTmaDescIm2ColOp
    : CuteNVGPU_SchemaOp<"make_tma_desc_im2col"> {
  let arguments = (ins AnyType:$gmem_view, AnyType:$smem_layout,
                       AnyType:$cta_value_tile, AnyType:$lower_corner_whd,
                       AnyType:$upper_corner_whd, AnyType:$lower_padding_whd,
                       AnyType:$upper_padding_whd, AnyType:$stride_whd,
                       AnyType:$lower_srt, AnyType:$stride_srt,
                       OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_MakeTmaDescTiledAtOp
    : CuteNVGPU_SchemaOp<"make_tma_desc_tiled_at"> {
  let arguments = (ins AnyType:$gmem_view, AnyType:$smem_layout,
                       AnyType:$cta_value_tile, AnyType:$tma_desc_addr,
                       Optional<AnyType>:$traversal_stride, AnyAttr:$mode,
                       OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs);
}

def CuteNVGPU_MakeTmaDescTiledOp
    : CuteNVGPU_SchemaOp<"make_tma_desc_tiled"> {
  let arguments = (ins AnyType:$gmem_view, AnyType:$smem_layout,
                       AnyType:$cta_value_tile,
                       Optional<AnyType>:$traversal_stride, AnyAttr:$mode,
                       OptionalAttr<I32Attr>:$num_multicast,
                       OptionalAttr<AnyAttr>:$tma_format);
  let results = (outs AnyType:$result);
}

def CuteNVGPU_SM103MakeUmmaSmemDescCircularOp
    : CuteNVGPU_SchemaOp<"sm103.make_umma_smem_desc_circular"> {
  let arguments = (ins AnyType:$src, Optional<AnyType>:$nextSrc,
                       AnyAttr:$layout, AnyAttr:$major);
  let results = (outs AnyType:$res);
}

def CuteNVGPU_MakeUmmaSmemDescOp
    : CuteNVGPU_SchemaOp<"make_umma_smem_desc"> {
  let arguments = (ins AnyType:$src, Optional<AnyType>:$nextSrc,
                       AnyAttr:$layout, AnyAttr:$major);
  let results = (outs AnyType:$res);
}

def CuteNVGPU_PrefetchTmaDescOp
    : CuteNVGPU_SchemaOp<"prefetch_tma_desc"> {
  let arguments = (ins AnyType:$atom);
  let results = (outs);
}

def CuteNVGPU_UpdateTmaDescOp
    : CuteNVGPU_SchemaOp<"update_tma_desc"> {
  let arguments = (ins AnyType:$tma_atom, AnyType:$gmem_tensor,
                       AnyType:$tma_desc_ptr);
  let results = (outs);
}

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_OPS_TD
