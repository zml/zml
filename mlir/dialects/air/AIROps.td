//===- AIROps.td - Metal AIR operations --------------------*- tablegen -*-===//

#ifndef MLIR_DIALECTS_AIR_AIROPS_TD
#define MLIR_DIALECTS_AIR_AIROPS_TD

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/dialects/air/AIRAttributes.td"

class AIR_VectorOf<list<Type> types>
    : FixedVectorOfRankAndLengthAndType<[1], [2, 3, 4], types>;
class AIR_ScalarOrVectorOf<list<Type> types>
    : AnyTypeOf<!listconcat(types, [AIR_VectorOf<types>])>;

def AIR_FloatLike : AIR_ScalarOrVectorOf<[F16, F32]>;
def AIR_F32Like : AIR_ScalarOrVectorOf<[F32]>;
def AIR_IntLike : AIR_ScalarOrVectorOf<[I8, I16, I32, I64]>;
def AIR_SimdIntLike : AIR_ScalarOrVectorOf<[I8, I16, I32]>;
def AIR_ConvertLike : AIR_ScalarOrVectorOf<[F16, BF16, F32, I8, I16, I32, I64]>;
def AIR_Matrix : FixedVectorOfRankAndLengthAndType<[1], [64], [F16, BF16, F32]>;
def AIR_MatrixVector : FixedVectorOfShapeAndType<[2], I64>;
def AIR_MatrixPointer : AnyTypeOf<[LLVM_PointerInAddressSpace<1>,
                                   LLVM_PointerInAddressSpace<3>]>;

def AIR_FastMath : DefaultValuedAttr<Arith_FastMathAttr,
                                     "::mlir::arith::FastMathFlags::none">;

//===----------------------------------------------------------------------===//
// Math
//===----------------------------------------------------------------------===//

class AIR_ElementwiseOp<string mnemonic, list<Trait> traits = []>
    : AIR_Op<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let summary = "`" # mnemonic # "` on each element";
  let description = "Calls `@air." # mnemonic # ".<type>`.";
}

class AIR_FloatUnaryOp<string mnemonic, Type type = AIR_FloatLike>
    : AIR_ElementwiseOp<mnemonic,
                        [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins type:$x, AIR_FastMath:$fastmath);
  let results = (outs type:$result);
  let assemblyFormat =
      "$x (`fastmath` `` $fastmath^)? attr-dict `:` type($result)";
}

class AIR_FloatBinaryOp<string mnemonic, Type type = AIR_FloatLike>
    : AIR_ElementwiseOp<mnemonic,
                        [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins type:$x, type:$y, AIR_FastMath:$fastmath);
  let results = (outs type:$result);
  let assemblyFormat =
      "$x `,` $y (`fastmath` `` $fastmath^)? attr-dict `:` type($result)";
}

class AIR_FloatTernaryOp<string mnemonic, Type type = AIR_FloatLike>
    : AIR_ElementwiseOp<mnemonic,
                        [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins type:$x, type:$y, type:$z, AIR_FastMath:$fastmath);
  let results = (outs type:$result);
  let assemblyFormat = [{
    $x `,` $y `,` $z (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
  }];
}

class AIR_IntUnaryOp<string mnemonic> : AIR_ElementwiseOp<mnemonic> {
  let arguments = (ins AIR_IntLike:$x);
  let results = (outs AIR_IntLike:$result);
  let assemblyFormat = "$x attr-dict `:` type($result)";
}

class AIR_IntBinaryOp<string mnemonic> : AIR_ElementwiseOp<mnemonic> {
  let arguments = (ins AIR_IntLike:$x, AIR_IntLike:$y);
  let results = (outs AIR_IntLike:$result);
  let assemblyFormat = "$x `,` $y attr-dict `:` type($result)";
}

class AIR_IntTernaryOp<string mnemonic> : AIR_ElementwiseOp<mnemonic> {
  let arguments = (ins AIR_IntLike:$x, AIR_IntLike:$y, AIR_IntLike:$z);
  let results = (outs AIR_IntLike:$result);
  let assemblyFormat = "$x `,` $y `,` $z attr-dict `:` type($result)";
}

// `exp` is `metal::precise::exp`; `fast_exp` is `metal::fast::exp`, which
// `exp` selects under fast math. Only f32 has fast variants.
def AIR_ExpOp : AIR_FloatUnaryOp<"exp">;
def AIR_FastExpOp : AIR_FloatUnaryOp<"fast_exp", AIR_F32Like>;
def AIR_Exp2Op : AIR_FloatUnaryOp<"exp2">;
def AIR_FastExp2Op : AIR_FloatUnaryOp<"fast_exp2", AIR_F32Like>;
def AIR_LogOp : AIR_FloatUnaryOp<"log">;
def AIR_FastLogOp : AIR_FloatUnaryOp<"fast_log", AIR_F32Like>;
def AIR_Log2Op : AIR_FloatUnaryOp<"log2">;
def AIR_FastLog2Op : AIR_FloatUnaryOp<"fast_log2", AIR_F32Like>;
def AIR_TanhOp : AIR_FloatUnaryOp<"tanh">;
def AIR_FastTanhOp : AIR_FloatUnaryOp<"fast_tanh", AIR_F32Like>;
def AIR_SqrtOp : AIR_FloatUnaryOp<"sqrt">;
def AIR_FastSqrtOp : AIR_FloatUnaryOp<"fast_sqrt", AIR_F32Like>;
def AIR_RsqrtOp : AIR_FloatUnaryOp<"rsqrt">;
def AIR_FastRsqrtOp : AIR_FloatUnaryOp<"fast_rsqrt", AIR_F32Like>;
def AIR_FabsOp : AIR_FloatUnaryOp<"fabs">;
def AIR_FastFabsOp : AIR_FloatUnaryOp<"fast_fabs", AIR_F32Like>;
def AIR_CeilOp : AIR_FloatUnaryOp<"ceil">;
def AIR_FastCeilOp : AIR_FloatUnaryOp<"fast_ceil", AIR_F32Like>;
def AIR_FloorOp : AIR_FloatUnaryOp<"floor">;
def AIR_FastFloorOp : AIR_FloatUnaryOp<"fast_floor", AIR_F32Like>;
def AIR_RoundOp : AIR_FloatUnaryOp<"round">;
def AIR_FastRoundOp : AIR_FloatUnaryOp<"fast_round", AIR_F32Like>;
def AIR_TruncOp : AIR_FloatUnaryOp<"trunc">;
def AIR_FastTruncOp : AIR_FloatUnaryOp<"fast_trunc", AIR_F32Like>;
def AIR_SinOp : AIR_FloatUnaryOp<"sin">;
def AIR_FastSinOp : AIR_FloatUnaryOp<"fast_sin", AIR_F32Like>;
def AIR_CosOp : AIR_FloatUnaryOp<"cos">;
def AIR_FastCosOp : AIR_FloatUnaryOp<"fast_cos", AIR_F32Like>;

def AIR_FmaxOp : AIR_FloatBinaryOp<"fmax">;
def AIR_FastFmaxOp : AIR_FloatBinaryOp<"fast_fmax", AIR_F32Like>;
def AIR_FminOp : AIR_FloatBinaryOp<"fmin">;
def AIR_FastFminOp : AIR_FloatBinaryOp<"fast_fmin", AIR_F32Like>;
def AIR_PowOp : AIR_FloatBinaryOp<"pow">;
def AIR_FastPowOp : AIR_FloatBinaryOp<"fast_pow", AIR_F32Like>;
def AIR_FmodOp : AIR_FloatBinaryOp<"fmod">;
def AIR_FastFmodOp : AIR_FloatBinaryOp<"fast_fmod", AIR_F32Like>;

def AIR_FmaOp : AIR_FloatTernaryOp<"fma">;
def AIR_ClampOp : AIR_FloatTernaryOp<"clamp">;
def AIR_FastClampOp : AIR_FloatTernaryOp<"fast_clamp", AIR_F32Like>;

def AIR_DotOp : AIR_Op<"dot", [Pure, AllTypesMatch<["x", "y"]>,
    DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let summary = "Dot product of two float vectors";
  let description = "Calls `@air.dot.<vector type>`.";
  let arguments = (ins AIR_VectorOf<[F16, F32]>:$x, AIR_VectorOf<[F16, F32]>:$y,
                       AIR_FastMath:$fastmath);
  let results = (outs AnyTypeOf<[F16, F32]>:$result);
  let assemblyFormat = [{
    $x `,` $y (`fastmath` `` $fastmath^)? attr-dict `:` type($x) `->`
    type($result)
  }];
  let hasVerifier = 1;
}

// `.s` and `.u` are the signed and unsigned overloads of MSL's integer
// functions.
def AIR_MaxSOp : AIR_IntBinaryOp<"max.s">;
def AIR_MaxUOp : AIR_IntBinaryOp<"max.u">;
def AIR_MinSOp : AIR_IntBinaryOp<"min.s">;
def AIR_MinUOp : AIR_IntBinaryOp<"min.u">;
def AIR_AbsSOp : AIR_IntUnaryOp<"abs.s">;
def AIR_ClampSOp : AIR_IntTernaryOp<"clamp.s">;
def AIR_ClampUOp : AIR_IntTernaryOp<"clamp.u">;

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

class AIR_ConvertBaseOp<string mnemonic, string summaryText>
    : AIR_Op<mnemonic, [Pure,
        DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let summary = summaryText;
  let description = [{
    Calls `@air.convert.<result kind>.<result type>.<input kind>.<input type>`,
    where floats are `f` and the integer side is `s` or `u` as the mnemonic
    says. Scalar f32 to and from f16 or bf16, and integer to integer, are
    LLVM casts in Apple's compiler.
  }];
  let arguments = (ins AIR_ConvertLike:$x, AIR_FastMath:$fastmath);
  let results = (outs AIR_ConvertLike:$result);
  let assemblyFormat = [{
    $x (`fastmath` `` $fastmath^)? attr-dict `:` type($x) `to` type($result)
  }];
  let hasVerifier = 1;
}

def AIR_ConvertOp : AIR_ConvertBaseOp<"convert", "Float to float conversion">;
def AIR_ConvertSOp : AIR_ConvertBaseOp<"convert.s",
    "Conversion between floats and signed integers">;
def AIR_ConvertUOp : AIR_ConvertBaseOp<"convert.u",
    "Conversion between floats and unsigned integers">;

//===----------------------------------------------------------------------===//
// SIMD-group functions
//===----------------------------------------------------------------------===//

// SIMD-group functions read the other lanes, so they are not Pure: unknown
// effects keep them from being speculated or moved across divergent control.
class AIR_SimdReduceOp<string mnemonic, Type type, list<Trait> traits = []>
    : AIR_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let summary = "`" # mnemonic # "` across the SIMD-group";
  let description = "Calls `@air." # mnemonic # ".<type>`.";
  let results = (outs type:$result);
}

class AIR_SimdFloatReduceOp<string mnemonic>
    : AIR_SimdReduceOp<mnemonic, AIR_FloatLike,
                       [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins AIR_FloatLike:$x, AIR_FastMath:$fastmath);
  let assemblyFormat =
      "$x (`fastmath` `` $fastmath^)? attr-dict `:` type($result)";
}

class AIR_SimdIntReduceOp<string mnemonic>
    : AIR_SimdReduceOp<mnemonic, AIR_SimdIntLike> {
  let arguments = (ins AIR_SimdIntLike:$x);
  let assemblyFormat = "$x attr-dict `:` type($result)";
}

class AIR_SimdShuffleOp<string mnemonic, Type type, list<Trait> traits = []>
    : AIR_Op<mnemonic, traits # [AllTypesMatch<["data", "result"]>]> {
  let summary = "`" # mnemonic # "` across the SIMD-group";
  let description = [{
    Calls `@air.}] # mnemonic # [{.<type>`. `lane` is the source lane, the
    XOR mask or the lane delta, as for the MSL function.
  }];
  let results = (outs type:$result);
}

class AIR_SimdFloatShuffleOp<string mnemonic>
    : AIR_SimdShuffleOp<mnemonic, AIR_FloatLike,
                        [DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let arguments = (ins AIR_FloatLike:$data, I16:$lane, AIR_FastMath:$fastmath);
  let assemblyFormat = [{
    $data `,` $lane (`fastmath` `` $fastmath^)? attr-dict `:` type($result)
  }];
}

class AIR_SimdIntShuffleOp<string mnemonic>
    : AIR_SimdShuffleOp<mnemonic, AIR_SimdIntLike> {
  let arguments = (ins AIR_SimdIntLike:$data, I16:$lane);
  let assemblyFormat = "$data `,` $lane attr-dict `:` type($result)";
}

def AIR_SimdSumOp : AIR_SimdFloatReduceOp<"simd_sum">;
def AIR_SimdSumSOp : AIR_SimdIntReduceOp<"simd_sum.s">;
def AIR_SimdSumUOp : AIR_SimdIntReduceOp<"simd_sum.u">;
def AIR_SimdMaxOp : AIR_SimdFloatReduceOp<"simd_max">;
def AIR_SimdMaxSOp : AIR_SimdIntReduceOp<"simd_max.s">;
def AIR_SimdMaxUOp : AIR_SimdIntReduceOp<"simd_max.u">;
def AIR_SimdMinOp : AIR_SimdFloatReduceOp<"simd_min">;
def AIR_SimdMinSOp : AIR_SimdIntReduceOp<"simd_min.s">;
def AIR_SimdMinUOp : AIR_SimdIntReduceOp<"simd_min.u">;
def AIR_SimdPrefixExclusiveSumOp
    : AIR_SimdFloatReduceOp<"simd_prefix_exclusive_sum">;

def AIR_SimdShuffleOp : AIR_SimdFloatShuffleOp<"simd_shuffle">;
def AIR_SimdShuffleSOp : AIR_SimdIntShuffleOp<"simd_shuffle.s">;
def AIR_SimdShuffleUOp : AIR_SimdIntShuffleOp<"simd_shuffle.u">;
def AIR_SimdShuffleXorOp : AIR_SimdFloatShuffleOp<"simd_shuffle_xor">;
def AIR_SimdShuffleXorSOp : AIR_SimdIntShuffleOp<"simd_shuffle_xor.s">;
def AIR_SimdShuffleXorUOp : AIR_SimdIntShuffleOp<"simd_shuffle_xor.u">;
def AIR_SimdShuffleDownOp : AIR_SimdFloatShuffleOp<"simd_shuffle_down">;
def AIR_SimdShuffleDownSOp : AIR_SimdIntShuffleOp<"simd_shuffle_down.s">;
def AIR_SimdShuffleDownUOp : AIR_SimdIntShuffleOp<"simd_shuffle_down.u">;
def AIR_SimdBroadcastOp : AIR_SimdFloatShuffleOp<"simd_broadcast">;
def AIR_SimdBroadcastSOp : AIR_SimdIntShuffleOp<"simd_broadcast.s">;
def AIR_SimdBroadcastUOp : AIR_SimdIntShuffleOp<"simd_broadcast.u">;

class AIR_SimdVoteOp<string mnemonic, string summaryText> : AIR_Op<mnemonic> {
  let summary = summaryText;
  let description = "Calls `@air." # mnemonic # "`.";
  let arguments = (ins I1:$x);
  let results = (outs I1:$result);
  let assemblyFormat = "$x attr-dict";
}

def AIR_SimdAnyOp : AIR_SimdVoteOp<"simd_any",
    "Whether `x` is true in any active lane">;
def AIR_SimdAllOp : AIR_SimdVoteOp<"simd_all",
    "Whether `x` is true in every active lane">;

//===----------------------------------------------------------------------===//
// Barriers
//===----------------------------------------------------------------------===//

def AIR_WgBarrierOp : AIR_Op<"wg.barrier"> {
  let summary = "threadgroup_barrier";
  let description = "Calls `@air.wg.barrier(i32 mem_flags, i32 1)`.";
  let arguments = (ins AIR_MemFlagsAttr:$mem_flags);
  let assemblyFormat = "$mem_flags attr-dict";
}

def AIR_SimdgroupBarrierOp : AIR_Op<"simdgroup.barrier"> {
  let summary = "simdgroup_barrier";
  let description = "Calls `@air.simdgroup.barrier(i32 mem_flags, i32 4)`.";
  let arguments = (ins AIR_MemFlagsAttr:$mem_flags);
  let assemblyFormat = "$mem_flags attr-dict";
}

//===----------------------------------------------------------------------===//
// SIMD-group matrices
//===----------------------------------------------------------------------===//

// A simdgroup_matrix<T, 8, 8> is a vector<64xT>; thread_elements() are its
// elements 0 and 1.
def AIR_SimdgroupMatrixLoadOp : AIR_Op<"simdgroup_matrix_8x8_load", [
    DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let summary = "simdgroup_load";
  let description = [{
    Calls `@air.simdgroup_matrix_8x8_load.v64<T>.p<address space><T>`.
    `simdgroup_load(m, p, ld, origin, transpose)` passes `extents` (ld, 8)
    and `strides` (1, ld), swapped when transposed.
  }];
  let arguments = (ins AIR_MatrixPointer:$pointer,
                       AIR_MatrixVector:$extents,
                       AIR_MatrixVector:$strides,
                       AIR_MatrixVector:$origin,
                       AIR_FastMath:$fastmath);
  let results = (outs AIR_Matrix:$result);
  let assemblyFormat = [{
    $pointer `,` $extents `,` $strides `,` $origin (`fastmath` `` $fastmath^)?
    attr-dict `:` qualified(type($pointer)) `->` type($result)
  }];
}

def AIR_SimdgroupMatrixStoreOp : AIR_Op<"simdgroup_matrix_8x8_store"> {
  let summary = "simdgroup_store";
  let description = [{
    Calls `@air.simdgroup_matrix_8x8_store.v64<T>.p<address space><T>`, with
    the operands of `simdgroup_matrix_8x8_load`.
  }];
  let arguments = (ins AIR_Matrix:$value,
                       AIR_MatrixPointer:$pointer,
                       AIR_MatrixVector:$extents,
                       AIR_MatrixVector:$strides,
                       AIR_MatrixVector:$origin);
  let assemblyFormat = [{
    $value `,` $pointer `,` $extents `,` $strides `,` $origin attr-dict `:`
    type($value) `,` qualified(type($pointer))
  }];
}

def AIR_SimdgroupMatrixMultiplyAccumulateOp
    : AIR_Op<"simdgroup_matrix_8x8_multiply_accumulate", [
        DeclareOpInterfaceMethods<ArithFastMathInterface>]> {
  let summary = "simdgroup_multiply_accumulate";
  let description = [{
    Calls `@air.simdgroup_matrix_8x8_multiply_accumulate.v64<D>.v64<A>.v64<B>.v64<C>`:
    `result = a * b + c`. `simdgroup_multiply` passes a zero `c`.
  }];
  let arguments = (ins AIR_Matrix:$a, AIR_Matrix:$b, AIR_Matrix:$c,
                       AIR_FastMath:$fastmath);
  let results = (outs AIR_Matrix:$result);
  let assemblyFormat = [{
    $a `,` $b `,` $c (`fastmath` `` $fastmath^)? attr-dict `:` type($a) `,`
    type($b) `,` type($c) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // MLIR_DIALECTS_AIR_AIROPS_TD
