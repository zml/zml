//===- CutePrivateCompatOps.td - proprietary CuTe op schemas -*- tablegen -*-===//

// Operation contracts recovered from the generated bindings distributed with
// nvidia-cutlass-dsl 4.6.2. These declarations intentionally encode only facts
// present in that registry: operand/result group structure, attribute names,
// and builtin type constraints. The proprietary compiler remains the authority
// for semantic verification and lowering.

#ifndef CUTE_IR_DIALECT_CUTE_IR_CUTE_PRIVATE_COMPAT_OPS_TD
#define CUTE_IR_DIALECT_CUTE_IR_CUTE_PRIVATE_COMPAT_OPS_TD

include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"

class Cute_PrivateSchemaOp<string mnemonic, list<Trait> traits = []>
    : Cute_Op<mnemonic, traits> {
  let summary = "Operation recovered from the CuTe-DSL operation registry";
}

def Cute_PrivateAddOffsetOp : Cute_PrivateSchemaOp<"add_offset"> {
  let arguments = (ins AnyType:$src, AnyType:$offset);
  let results = (outs AnyType:$dst);
}

def Cute_PrivateApplySwizzleOp : Cute_PrivateSchemaOp<"apply_swizzle"> {
  let arguments = (ins AnyType:$ptr, OptionalAttr<Cute_SwizzleAttr>:$swizzle);
  let results = (outs AnyType:$swizzled_ptr);
}

def Cute_PrivateAssumeOp : Cute_PrivateSchemaOp<"assume"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$dst);
}

def Cute_PrivateCopyOp
    : Cute_PrivateSchemaOp<"copy", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$copy_atom, Variadic<AnyType>:$src,
                       Variadic<AnyType>:$dst, Optional<AnyType>:$pred);
  let results = (outs);
}

def Cute_PrivateCopyMakeFragmentOp
    : Cute_PrivateSchemaOp<"copy.make_fragment"> {
  let arguments = (ins AnyType:$atom, AnyType:$input, I32Attr:$operand_id);
  let results = (outs AnyType:$result);
}

def Cute_PrivateCopyAtomCallOp
    : Cute_PrivateSchemaOp<"copy_atom_call", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$atom, Variadic<AnyType>:$src,
                       Variadic<AnyType>:$dst, Optional<AnyType>:$pred);
  let results = (outs);
}

def Cute_PrivateDerefArithTupleIterOp
    : Cute_PrivateSchemaOp<"deref_arith_tuple_iter"> {
  let arguments = (ins AnyType:$iter);
  let results = (outs AnyType:$arith_tuple);
}

def Cute_PrivateDerefDescIterOp : Cute_PrivateSchemaOp<"deref_desc_iter"> {
  let arguments = (ins AnyType:$iter);
  let results = (outs AnyType:$value);
}

def Cute_PrivateDerefineOp : Cute_PrivateSchemaOp<"derefine"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$dst);
}

def Cute_PrivateFastDivmodComputeOp
    : Cute_PrivateSchemaOp<"fast_divmod.compute"> {
  let arguments = (ins AnyType:$dividend, AnyType:$divisor);
  let results = (outs AnyType:$quotient, AnyType:$remainder);
}

def Cute_PrivateFastDivmodCreateDivisorOp
    : Cute_PrivateSchemaOp<"fast_divmod.create_divisor"> {
  let arguments = (ins AnyType:$divisor);
  let results = (outs AnyType:$fast_divmod_divisor);
}

def Cute_PrivateFastDivmodDivideOp
    : Cute_PrivateSchemaOp<"fast_divmod.divide"> {
  let arguments = (ins AnyType:$dividend, AnyType:$divisor);
  let results = (outs AnyType:$quotient);
}

def Cute_PrivateFastDivmodGetAuxOp
    : Cute_PrivateSchemaOp<"fast_divmod.get_aux"> {
  let arguments = (ins AnyType:$divisor);
  let results = (outs AnyType:$multiplier, AnyInteger:$shift1,
                     AnyInteger:$shift2);
}

def Cute_PrivateFastDivmodGetDivisorOp
    : Cute_PrivateSchemaOp<"fast_divmod.get_divisor"> {
  let arguments = (ins AnyType:$fast_divmod_divisor);
  let results = (outs AnyType:$divisor);
}

def Cute_PrivateFastDivmodMakeDivisorOp
    : Cute_PrivateSchemaOp<"fast_divmod.make_divisor"> {
  let arguments = (ins AnyType:$divisor, AnyType:$multiplier,
                       AnyType:$sh1, AnyType:$sh2);
  let results = (outs AnyType:$fast_divmod_divisor);
}

def Cute_PrivateFilterOp : Cute_PrivateSchemaOp<"filter"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateFilterZerosOp : Cute_PrivateSchemaOp<"filter_zeros"> {
  let arguments = (ins AnyType:$input, Optional<AnyType>:$target_profile);
  let results = (outs AnyType:$res);
}

def Cute_PrivateGemmOp
    : Cute_PrivateSchemaOp<"gemm", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$mma_atom, AnyType:$d, Variadic<AnyType>:$a,
                       Variadic<AnyType>:$b, AnyType:$c);
  let results = (outs);
}

def Cute_PrivateGetFlatCoordOp : Cute_PrivateSchemaOp<"get_flat_coord"> {
  let arguments = (ins AnyType:$index, AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateGetHierCoordOp : Cute_PrivateSchemaOp<"get_hier_coord"> {
  let arguments = (ins AnyType:$index, AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateGetIntegralCoordOp
    : Cute_PrivateSchemaOp<"get_integral_coord"> {
  let arguments = (ins AnyType:$index, AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateGetIterOp : Cute_PrivateSchemaOp<"get_iter"> {
  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$ptr);
}

def Cute_PrivateGetLayoutOp : Cute_PrivateSchemaOp<"get_layout"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$layout);
}

def Cute_PrivateIntToPtrOp : Cute_PrivateSchemaOp<"inttoptr"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$dst);
}

def Cute_PrivateKernelSmemSizeOp : Cute_PrivateSchemaOp<"kernel_smem_size"> {
  let arguments = (ins SymbolRefAttr:$kernel_name);
  let results = (outs I64:$size);
}

def Cute_PrivateLoadScaledIndexOp
    : Cute_PrivateSchemaOp<"load_scaled_index"> {
  let arguments = (ins AnyType:$ptr_a, AnyType:$ptr_b, AnyType:$index,
                       AnyType:$stride);
  let results = (outs AnyType:$result);
}

def Cute_PrivateLocalPartitionOp : Cute_PrivateSchemaOp<"local_partition"> {
  let arguments = (ins AnyType:$input, AnyType:$tiler, AnyType:$index);
  let results = (outs AnyType:$result);
}

def Cute_PrivateLocalTileOp : Cute_PrivateSchemaOp<"local_tile"> {
  let arguments = (ins AnyType:$input, AnyType:$tile, AnyType:$coord,
                       OptionalAttr<Cute_CoordAttr>:$proj);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeArithTupleIterOp
    : Cute_PrivateSchemaOp<"make_arith_tuple_iter"> {
  let arguments = (ins Optional<AnyType>:$value);
  let results = (outs AnyType:$iter);
}

def Cute_PrivateMakeAtomOp : Cute_PrivateSchemaOp<"make_atom"> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$atom);
}

def Cute_PrivateMakeDescIterOp : Cute_PrivateSchemaOp<"make_desc_iter"> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$iter);
}

def Cute_PrivateMakeFragmentLikeOp
    : Cute_PrivateSchemaOp<"make_fragment_like"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$dst);
}

def Cute_PrivateMakeIdentityTensorOp
    : Cute_PrivateSchemaOp<"make_identity_tensor"> {
  let arguments = (ins AnyType:$shape);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeSparseElemOp : Cute_PrivateSchemaOp<"make_sparse_elem"> {
  let arguments = (ins AnyType:$physical_storage);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeTiledCopyOp : Cute_PrivateSchemaOp<"make_tiled_copy"> {
  let arguments = (ins AnyType:$atom);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeTiledCopyV2Op
    : Cute_PrivateSchemaOp<"make_tiled_copy_v2"> {
  let arguments = (ins AnyType:$atom);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeTiledMmaOp : Cute_PrivateSchemaOp<"make_tiled_mma"> {
  let arguments = (ins AnyType:$atom);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeTupleOp : Cute_PrivateSchemaOp<"make_tuple"> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMakeViewOp : Cute_PrivateSchemaOp<"make_view"> {
  let arguments = (ins AnyType:$iter, Optional<AnyType>:$layout);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMemrefAllocSmemOp
    : Cute_PrivateSchemaOp<"memref.alloc_smem"> {
  let arguments = (ins);
  let results = (outs AnyType:$memref);
}

def Cute_PrivateMemrefAllocaOp : Cute_PrivateSchemaOp<"memref.alloca"> {
  let arguments = (ins Optional<AnyType>:$layout);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMemrefLoadOp : Cute_PrivateSchemaOp<"memref.load"> {
  let arguments = (ins AnyType:$src, AnyType:$coord);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMemrefLoadVecOp
    : Cute_PrivateSchemaOp<"memref.load_vec", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$src, Optional<AnyVectorOfAnyRank>:$mask,
                       Optional<AnyVectorOfAnyRank>:$pass_thru);
  let results = (outs AnyVectorOfAnyRank:$result);
}

def Cute_PrivateMemrefStoreOp : Cute_PrivateSchemaOp<"memref.store"> {
  let arguments = (ins AnyType:$dst, AnyType:$coord, AnyType:$value);
  let results = (outs);
}

def Cute_PrivateMemrefStoreVecOp
    : Cute_PrivateSchemaOp<"memref.store_vec"> {
  let arguments = (ins AnyVectorOfAnyRank:$value, AnyType:$dst,
                       Optional<AnyVectorOfAnyRank>:$mask);
  let results = (outs);
}

def Cute_PrivateMmaMakeFragmentOp
    : Cute_PrivateSchemaOp<"mma.make_fragment"> {
  let arguments = (ins AnyType:$atom, AnyType:$input, I32Attr:$operand_id);
  let results = (outs AnyType:$result);
}

def Cute_PrivateMmaAtomCallOp
    : Cute_PrivateSchemaOp<"mma_atom_call", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$atom, AnyType:$d, Variadic<AnyType>:$a,
                       Variadic<AnyType>:$b, AnyType:$c);
  let results = (outs);
}

def Cute_PrivatePrefetchOp : Cute_PrivateSchemaOp<"prefetch"> {
  let arguments = (ins AnyType:$prefetch_atom, Variadic<AnyType>:$src);
  let results = (outs);
}

def Cute_PrivatePrefetchAtomCallOp
    : Cute_PrivateSchemaOp<"prefetch_atom_call"> {
  let arguments = (ins AnyType:$atom, Variadic<AnyType>:$src);
  let results = (outs);
}

def Cute_PrivatePrintTmaDescIm2colOp
    : Cute_PrivateSchemaOp<"print_tma_desc_im2col"> {
  let arguments = (ins AnyType:$input, AnyType:$fd);
  let results = (outs);
}

def Cute_PrivatePrintTmaDescTiledOp
    : Cute_PrivateSchemaOp<"print_tma_desc_tiled"> {
  let arguments = (ins AnyType:$input, AnyType:$fd);
  let results = (outs);
}

def Cute_PrivatePrintViewOp : Cute_PrivateSchemaOp<"print_view"> {
  let arguments = (ins AnyType:$src, Optional<AnyType>:$coord,
                       OptionalAttr<BoolAttr>:$verbose,
                       OptionalAttr<BoolAttr>:$is_signed);
  let results = (outs);
}

def Cute_PrivatePtrLoadOp : Cute_PrivateSchemaOp<"ptr.load"> {
  let arguments = (ins AnyType:$ptr);
  let results = (outs AnyType:$value);
}

def Cute_PrivatePtrStoreOp : Cute_PrivateSchemaOp<"ptr.store"> {
  let arguments = (ins AnyType:$value, AnyType:$ptr);
  let results = (outs);
}

def Cute_PrivatePtrToIntOp : Cute_PrivateSchemaOp<"ptrtoint"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyInteger:$dst);
}

def Cute_PrivateRandomDelayOp : Cute_PrivateSchemaOp<"random_delay"> {
  let arguments = (ins);
  let results = (outs);
}

def Cute_PrivateRecastIterOp : Cute_PrivateSchemaOp<"recast_iter"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$dst);
}

def Cute_PrivateReduceOp : Cute_PrivateSchemaOp<"reduce"> {
  let arguments = (ins AnyVectorOfAnyRank:$input, AnyType:$shape,
                       AnyType:$init_val,
                       Optional<AnyVectorOfAnyRank>:$predication,
                       AnyAttr:$reduction_op,
                       OptionalAttr<Cute_CoordAttr>:$reduction_profile);
  let results = (outs AnyType:$output, Optional<AnyType>:$reduced_shape);
}

def Cute_PrivateSmemPartitionSizeOp
    : Cute_PrivateSchemaOp<"smem_partition_size"> {
  let arguments = (ins I32Attr:$partition_id,
                       OptionalAttr<UnitAttr>:$cumulative);
  let results = (outs I64:$size);
}

def Cute_PrivateStencilDivideOp
    : Cute_PrivateSchemaOp<"stencil_divide", [AttrSizedOperandSegments]> {
  let arguments = (ins AnyType:$input, AnyType:$stencil,
                       Optional<AnyType>:$padding_upper,
                       Optional<AnyType>:$padding_lower,
                       Optional<AnyType>:$traversal_stride);
  let results = (outs AnyType:$result);
}

def Cute_PrivateSymbolicOp : Cute_PrivateSchemaOp<"symbolic"> {
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$symbolic);
}

def Cute_PrivateTiledCopyPartitionDOp
    : Cute_PrivateSchemaOp<"tiled.copy.partition_D"> {
  let arguments = (ins AnyType:$tiled_copy, AnyType:$input, AnyType:$coord);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTiledCopyPartitionSOp
    : Cute_PrivateSchemaOp<"tiled.copy.partition_S"> {
  let arguments = (ins AnyType:$tiled_copy, AnyType:$input, AnyType:$coord);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTiledCopyRetileOp
    : Cute_PrivateSchemaOp<"tiled.copy.retile"> {
  let arguments = (ins AnyType:$tiled_copy, AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTiledMmaPartitionOp
    : Cute_PrivateSchemaOp<"tiled.mma.partition"> {
  let arguments = (ins AnyType:$tiled_mma, AnyType:$input, AnyType:$coord,
                       I32Attr:$operand_id);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTiledMmaPartitionShapeOp
    : Cute_PrivateSchemaOp<"tiled.mma.partition_shape"> {
  let arguments = (ins AnyType:$tiled_mma, AnyType:$input,
                       I32Attr:$operand_id);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTupleProductOp : Cute_PrivateSchemaOp<"tuple.product"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTupleProductEachOp
    : Cute_PrivateSchemaOp<"tuple.product_each"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTupleDivOp : Cute_PrivateSchemaOp<"tuple_div"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTupleModOp : Cute_PrivateSchemaOp<"tuple_mod"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def Cute_PrivateTupleMulOp : Cute_PrivateSchemaOp<"tuple_mul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def Cute_PrivateUnpackTupleOp : Cute_PrivateSchemaOp<"unpack_tuple"> {
  let arguments = (ins AnyType:$tuple);
  let results = (outs Variadic<AnyType>:$result);
}

#endif // CUTE_IR_DIALECT_CUTE_IR_CUTE_PRIVATE_COMPAT_OPS_TD
