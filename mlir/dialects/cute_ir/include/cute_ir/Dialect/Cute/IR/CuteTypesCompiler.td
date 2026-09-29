// The cute dialect's types beyond the release, generated from the
// nvidia-cutlass-dsl 4.8.0 release; do not edit.
//
// Sources: the compiler library (`_cutlass_ir`): its generated parser
// messages give the classes with a declarative syntax, their parameters in
// declaration order and their C++ types; the typed Python bindings of the
// types (`get` signatures and getters) give the parameters of the others; the
// live compiler, loaded from Python, gives the syntax (what it prints for what
// it parses, including every default it leaves out) and the verifier
// messages.
//
// Differences from the compiler, where its C++ types are not defined here:
//   - The atoms of tiled_copy / tiled_copy_v2 / tiled_mma are
//     `CopyAtomTypeInterface` / `CopyAtomTypeInterfaceV2` /
//     `MmaAtomTypeInterface` in the compiler; plain types here.
//   - ptr, memref and i<N> have bespoke parsers in the compiler, so their
//     parameter names and order are not recorded; the order here follows the
//     Python `get` of each.
//   - coord_tensor and memref layouts are `cutegen::layout` in the compiler's
//     messages but accept composed layouts; they are a layout or composed
//     layout type here.
// Verifier rules not reproduced: the bitlayout's cosize and element-count
// rules and TiledMMA's "requires static permutations of MNK" (neither
// triggered by any probe). The memref's sparse density rule ("layout must be
// dense but cannot be divided by sparsity") is inferred from probes: every
// probe agrees with it.

#ifndef CUTE_IR_GENERATED_CUTE_TYPES_COMPILER_TD
#define CUTE_IR_GENERATED_CUTE_TYPES_COMPILER_TD

include "cute_ir/Dialect/Cute/IR/CuteTypes.td"
include "cute_ir/Dialect/Cute/IR/CuteAttrsCompiler.td"
include "cute_ir/Dialect/Cute/IR/CuteEnums.td"

def Cute_PtrType : Cute_Type<"Ptr", "ptr"> {
  let summary = "Pointer with address space, alignment, swizzle and bit layout";
  let description = [{
    `<[value_type, ][space][, align<N>][, S<B,M,S>][, bitlayout<"L">]>`, the
    fields in this order. Defaults are left out when printing and filled in
    when parsing:

    - the address space is `generic` (always printed);
    - the alignment is the value type's natural one: its width in bytes for
      an integer or float that is a whole number of bytes, 1 for `i1`, 0
      (none) for other sub-byte widths, the physical type's bytes for a
      `!cute.sparse_elem`, and 1 for any other type or no type;
    - the swizzle is the identity `S<0,4,3>`;
    - the bit layout is `(b,n):(1,s)` for an integer or float of `b` bits
      (`s` = 8 for `i1`, else `b`) with `n` the least count making a chunk of
      whole bytes (32-bit chunks in `tmem`), and
      `(pb,(L,n)):(1,(0,pb))` for a `!cute.sparse_elem<L, P>` of `pb` bits.

    A sparse pointer (the Python `sparsity`) is a `!cute.sparse_elem` value
    type.

    ```mlir
    !cute.ptr<bf16, smem, align<1024>, S<2,4,3>>
    !cute.ptr<f32, generic>
    !cute.ptr<i8, gmem, bitlayout<"(8,3):(1,8)">>
    ```
  }];
  let parameters = (ins
    Cute_OptionalParam<"mlir::Type", "pointee type">:$value_type,
    Cute_Param<"AddressSpace", "address space">:$address_space,
    Cute_Param<"uint64_t", "alignment in bytes; 0 only as a sub-byte type's natural alignment">:$alignment,
    Cute_OptionalParam<"SwizzleAttr", "swizzle; null for the identity">:$swizzle,
    Cute_OptionalParam<"BitLayoutAttr", "bit layout; null for the default">:$bitlayout
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    /// The alignment a pointer to `valueType` has when none is spelled.
    static uint64_t getNaturalAlignment(mlir::Type valueType);
    /// The bit layout a pointer has when none is spelled (null if none).
    static BitLayoutAttr getDefaultBitLayout(mlir::Type valueType,
                                             AddressSpace addressSpace);
  }];
}

def Cute_MemRefType : Cute_Type<"MemRef", "memref"> {
  let summary = "Tensor view: a pointer with a layout";
  let description = [{
    `<value_type, [space][, align<N>][, S<B,M,S>], "layout"[, bitlayout<"L">]>`:
    the pointer fields as in `!cute.ptr` (the value type is required), then the
    layout or composed layout, then a non-default bit layout. A composed
    layout whose inner part is the identity swizzle is read as its outer
    layout.

    ```mlir
    !cute.memref<bf16, smem, align<1024>, S<2,4,3>, "((8,16),(32,1),(1,3)):((32,256),(1,0),(0,4096))">
    !cute.memref<i8, gmem, "(4,8):(8,1)", bitlayout<"(8,3):(1,8)">>
    ```
  }];
  let parameters = (ins
    Cute_Param<"PtrType", "pointer">:$ptr,
    Cute_Param<"mlir::Type", "layout or composed-layout type">:$layout
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    mlir::Type getValueType() const { return getPtr().getValueType(); }
    AddressSpace getAddressSpace() const { return getPtr().getAddressSpace(); }
    uint64_t getAlignment() const { return getPtr().getAlignment(); }
    SwizzleAttr getSwizzle() const { return getPtr().getSwizzle(); }
  }];
}

def Cute_CoordTensorType : Cute_Type<"CoordTensor", "coord_tensor"> {
  let summary = "Coordinate tensor: an arithmetic tuple with a layout";
  let description = [{
    ```mlir
    !cute.coord_tensor<"(?,?,?)", "((512,1),1):((1/2@0,0),0)">
    ```
  }];
  let parameters = (ins
    Cute_Param<"IntTupleType", "origin arithmetic tuple">:$arith_tuple,
    Cute_Param<"mlir::Type", "layout or composed-layout type">:$layout
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Cute_ArithTupleIteratorType
    : Cute_Type<"ArithTupleIterator", "arith_tuple_iter"> {
  let summary = "Iterator over an arithmetic tuple";
  let description = [{
    ```mlir
    !cute.arith_tuple_iter<"(?,?{div=16})">
    ```
  }];
  let parameters = (ins
    Cute_Param<"IntTupleType", "current arithmetic tuple">:$arith_tuple
  );
  // `$arith_tuple` alone would print a space after the mnemonic.
  let hasCustomAssemblyFormat = 1;
}

// `!cute.i<width>`: one mnemonic per width, which the generated type dispatch
// cannot match, so the class has no mnemonic and CuteTypes.cpp routes
// `i<digits>` keywords to it.
def Cute_ConstrainedIntType : TypeDef<Cute_Dialect, "ConstrainedInt"> {
  let summary = "Integer of any width with known divisibility and power of two";
  let description = [{
    `!cute.i<width>[<[pow2, ]divby D>]`; `divby 1` is left out (the default),
    `divby 0` and negative divisors are kept. The compiler also reads
    `!cute_nvgpu.i<width>` as this type.

    ```mlir
    !cute.i32<divby 8>
    !cute.i16<pow2, divby 8>
    !cute.i64
    ```
  }];
  // The compiler's type name for it.
  let typeName = "cute.ConstrainedInt";
  let parameters = (ins
    Cute_Param<"int64_t", "known divisor">:$divisibility,
    Cute_Param<"unsigned", "integer width">:$width,
    Cute_Param<"bool", "known to be a power of two">:$is_pow2
  );
  let extraClassDeclaration = [{
    static mlir::Type parse(mlir::AsmParser &parser, unsigned width);
    void print(mlir::AsmPrinter &printer) const;
  }];
}

def Cute_FastDivmodDivisorType
    : Cute_Type<"FastDivmodDivisor", "fast_divmod_divisor"> {
  let summary = "Precomputed divisor of the fast divmod operations";
  let description = [{
    ```mlir
    !cute.fast_divmod_divisor<32>
    !cute.fast_divmod_divisor<64, is_pow_2>
    ```
  }];
  let parameters = (ins
    Cute_Param<"unsigned", "integer width">:$width,
    Cute_Param<"bool", "whether the divisor is a power of two">:$is_pow_2
  );
  let hasCustomAssemblyFormat = 1;
}

def Cute_SparseElemType : Cute_Type<"SparseElem", "sparse_elem"> {
  let summary = "Logical sparse element stored in a physical type";
  let description = [{
    `num_logical` logical elements stored as `physical_type` (an integer,
    float or vector of them).

    ```mlir
    !cute.sparse_elem<2, vector<1xf16>>
    ```
  }];
  let parameters = (ins
    Cute_Param<"int", "number of logical elements">:$num_logical,
    Cute_Param<"mlir::Type", "physical storage type">:$physical_type
  );
  let assemblyFormat = "`<` $num_logical `,` $physical_type `>`";
  let genVerifyDecl = 1;
}

def Cute_TiledCopyType : Cute_Type<"TiledCopy", "tiled_copy"> {
  let summary = "Copy atom tiled over threads and values";
  let description = [{
    ```mlir
    !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32>, layout_copy_tv = <"(32,4):(4,1)">, tiler_mn = <"[4:1;32:1]">>
    ```
  }];
  let parameters = (ins
    Cute_Param<"mlir::Type", "copy atom">:$copy_atom,
    Cute_Param<"LayoutAttr", "thread-value layout">:$layout_copy_tv,
    Cute_Param<"TileAttr", "MN tiler">:$tiler_mn
  );
  let assemblyFormat = [{
    `<` $copy_atom `,` `layout_copy_tv` `=` $layout_copy_tv `,`
    `tiler_mn` `=` $tiler_mn `>`
  }];
}

def Cute_TiledCopyV2Type : Cute_Type<"TiledCopyV2", "tiled_copy_v2"> {
  let summary = "V2 copy atom tiled over threads and values";
  let description = [{
    ```mlir
    !cute.tiled_copy_v2<!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>, layout_copy_tv = <"(1,8):(0,1)">, tiler_mn = <"[8:1]">>
    ```
  }];
  let parameters = (ins
    Cute_Param<"mlir::Type", "V2 copy atom">:$copy_atom,
    Cute_Param<"LayoutAttr", "thread-value layout">:$layout_copy_tv,
    Cute_Param<"TileAttr", "MN tiler">:$tiler_mn
  );
  let assemblyFormat = [{
    `<` $copy_atom `,` `layout_copy_tv` `=` $layout_copy_tv `,`
    `tiler_mn` `=` $tiler_mn `>`
  }];
}

def Cute_TiledMmaType : Cute_Type<"TiledMma", "tiled_mma"> {
  let summary = "MMA atom tiled over MNK";
  let description = [{
    The permutation is optional (absent, not `[_;_;_]`, when not given).

    ```mlir
    !cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <"(16,16,1):(16,1,0)">, permutation_MNK = <"[(16,4):(4,1);(16,4):(4,1);_]">>
    ```
  }];
  let parameters = (ins
    Cute_Param<"mlir::Type", "mma atom">:$mma_atom,
    Cute_Param<"LayoutAttr", "rank-3 atom layout over MNK">:$atom_layout_MNK,
    Cute_OptionalParam<"TileAttr", "rank-3 permutation over MNK">:$permutation_MNK
  );
  let assemblyFormat = [{
    `<` $mma_atom `,` `atom_layout_MNK` `=` $atom_layout_MNK
    (`,` `permutation_MNK` `=` $permutation_MNK^)? `>`
  }];
  let genVerifyDecl = 1;
}

def Cute_TupleType : Cute_Type<"Tuple", "tuple"> {
  let summary = "Non-empty tuple of types";
  let description = [{
    ```mlir
    !cute.tuple<i32, f32>
    ```
  }];
  let parameters = (ins ArrayRefParameter<"mlir::Type", "element types">:$types);
  let assemblyFormat = "`<` $types `>`";
}

#endif // CUTE_IR_GENERATED_CUTE_TYPES_COMPILER_TD
