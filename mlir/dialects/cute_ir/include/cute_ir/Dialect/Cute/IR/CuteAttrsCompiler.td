// The cute dialect's attributes beyond the release, generated from the
// nvidia-cutlass-dsl 4.8.0 release; do not edit.
//
// Sources: the compiler library (`_cutlass_ir`): its generated parser
// messages give every class, its parameters in declaration order and their
// C++ types; the live compiler, loaded from Python, gives the syntax (what it
// prints for what it parses) and the verifier messages.
//
// The fragment parameters are `FragmentAttrInterface` in the compiler, an
// interface neither the release nor this tree defines: they are plain
// attributes here (the cute_nvgpu `*_frg` attributes).
//
// The reduction_op enum attribute is in CuteEnums.td.

#ifndef CUTE_IR_GENERATED_CUTE_ATTRS_COMPILER_TD
#define CUTE_IR_GENERATED_CUTE_ATTRS_COMPILER_TD

include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"

def Cute_BitLayoutAttr : Cute_Attr<"BitLayout", "bitlayout"> {
  let summary = "Bit layout of a pointer's elements within a storage chunk";
  let description = [{
    A two-mode layout from (element bit, element index) to the bit offset in
    a storage chunk. Used by `!cute.ptr` (`bitlayout<"...">`).

    ```mlir
    #cute.bitlayout<<"(8,4):(1,8)">>
    ```
  }];
  let parameters = (ins
    Cute_Param<"LayoutAttr", "(element bit, element index) -> bit">:$layout
  );
  let assemblyFormat = "`<` $layout `>`";
  let genVerifyDecl = 1;
}

def Cute_CopyAtomAttr : Cute_Attr<"CopyAtom", "copy_atom"> {
  let summary = "Thread and value layouts of a copy atom";
  let description = [{
    ```mlir
    #cute.copy_atom<thr_id = <"1:0">, layout_tv = (<"(1,8):(0,1)">, <"(1,8):(0,1)">, <"(1,8):(0,1)">)>
    ```
  }];
  let parameters = (ins
    Cute_Param<"LayoutAttr", "thread id layout">:$thr_id,
    Cute_Param<"LayoutAttr", "source thread-value layout">:$layout_src,
    Cute_Param<"LayoutAttr", "destination thread-value layout">:$layout_dst,
    Cute_Param<"LayoutAttr", "reference thread-value layout">:$layout_ref
  );
  let assemblyFormat = [{
    `<` `thr_id` `=` $thr_id `,` `layout_tv` `=`
    `(` $layout_src `,` $layout_dst `,` $layout_ref `)` `>`
  }];
}

def Cute_CopyAtomV2Attr : Cute_Attr<"CopyAtomV2", "copy_atom_v2"> {
  let summary = "Thread and value layouts and fragments of a V2 copy atom";
  let description = [{
    ```mlir
    #cute.copy_atom_v2<thr_id = <"1:0">, layout_tv = (<"(1,8):(0,1)">, <"(1,8):(0,1)">), frg = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>
    ```
  }];
  let parameters = (ins
    Cute_Param<"LayoutAttr", "thread id layout">:$thr_id,
    Cute_Param<"LayoutAttr", "source thread-value layout">:$layoutSrc_TV,
    Cute_Param<"LayoutAttr", "destination thread-value layout">:$layoutDst_TV,
    Cute_Param<"mlir::Attribute", "source fragment">:$frgSrc,
    Cute_Param<"mlir::Attribute", "destination fragment">:$frgDst
  );
  let assemblyFormat = [{
    `<` `thr_id` `=` $thr_id `,` `layout_tv` `=`
    `(` $layoutSrc_TV `,` $layoutDst_TV `)` `,`
    `frg` `=` `(` $frgSrc `,` $frgDst `)` `>`
  }];
}

def Cute_MmaAtomAttr : Cute_Attr<"MmaAtom", "mma_atom"> {
  let summary = "Shapes, thread-value layouts and fragments of an MMA atom";
  let description = [{
    ```mlir
    #cute.mma_atom<<"(16,8,16)">, <"32:1">, layout_TV = (<"1:0">, <"1:0">, <"1:0">), shapes = (<"(16,16)">, <"(8,16)">, <"(16,8)">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>
    ```
  }];
  let parameters = (ins
    Cute_Param<"ShapeAttr", "MNK shape">:$shape_MNK,
    Cute_Param<"LayoutAttr", "thread id layout">:$thr_id,
    Cute_Param<"LayoutAttr", "A thread-value layout">:$layoutA_TV,
    Cute_Param<"LayoutAttr", "B thread-value layout">:$layoutB_TV,
    Cute_Param<"LayoutAttr", "C thread-value layout">:$layoutC_TV,
    Cute_Param<"ShapeAttr", "A shape (MK)">:$shapeA_MK,
    Cute_Param<"ShapeAttr", "B shape (NK)">:$shapeB_NK,
    Cute_Param<"ShapeAttr", "C shape (MN)">:$shapeC_MN,
    Cute_Param<"mlir::Attribute", "A fragment">:$frgA,
    Cute_Param<"mlir::Attribute", "B fragment">:$frgB,
    Cute_Param<"mlir::Attribute", "C fragment">:$frgC
  );
  let assemblyFormat = [{
    `<` $shape_MNK `,` $thr_id `,`
    `layout_TV` `=` `(` $layoutA_TV `,` $layoutB_TV `,` $layoutC_TV `)` `,`
    `shapes` `=` `(` $shapeA_MK `,` $shapeB_NK `,` $shapeC_MN `)`
    `frgs` `=` `(` $frgA `,` $frgB `,` $frgC `)` `>`
  }];
}

def Cute_MxMmaAtomAttr : Cute_Attr<"MxMmaAtom", "mx_mma_atom"> {
  let summary = "MMA atom attribute with scale-factor operands";
  let description = [{
    Operands in the order A, SFA, B, SFB, C.

    ```mlir
    #cute.mx_mma_atom<<"(16,8,16)">, <"32:1">, layout_TV = (<"1:0">, <"1:0">, <"1:0">, <"1:0">, <"1:0">), shapes = (<"(16,16)">, <"(8,16)">, <"(16,8)">, <"1">, <"2">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>
    ```
  }];
  let parameters = (ins
    Cute_Param<"ShapeAttr", "MNK shape">:$shape_MNK,
    Cute_Param<"LayoutAttr", "thread id layout">:$thr_id,
    Cute_Param<"LayoutAttr", "A thread-value layout">:$layoutA_TV,
    Cute_Param<"LayoutAttr", "SFA thread-value layout">:$layoutSFA_TV,
    Cute_Param<"LayoutAttr", "B thread-value layout">:$layoutB_TV,
    Cute_Param<"LayoutAttr", "SFB thread-value layout">:$layoutSFB_TV,
    Cute_Param<"LayoutAttr", "C thread-value layout">:$layoutC_TV,
    Cute_Param<"ShapeAttr", "A shape (MK)">:$shapeA_MK,
    Cute_Param<"ShapeAttr", "SFA shape (MK)">:$shapeSFA_MK,
    Cute_Param<"ShapeAttr", "B shape (NK)">:$shapeB_NK,
    Cute_Param<"ShapeAttr", "SFB shape (NK)">:$shapeSFB_NK,
    Cute_Param<"ShapeAttr", "C shape (MN)">:$shapeC_MN,
    Cute_Param<"mlir::Attribute", "A fragment">:$frgA,
    Cute_Param<"mlir::Attribute", "SFA fragment">:$frgSFA,
    Cute_Param<"mlir::Attribute", "B fragment">:$frgB,
    Cute_Param<"mlir::Attribute", "SFB fragment">:$frgSFB,
    Cute_Param<"mlir::Attribute", "C fragment">:$frgC
  );
  let assemblyFormat = [{
    `<` $shape_MNK `,` $thr_id `,`
    `layout_TV` `=` `(` $layoutA_TV `,` $layoutSFA_TV `,` $layoutB_TV `,`
      $layoutSFB_TV `,` $layoutC_TV `)` `,`
    `shapes` `=` `(` $shapeA_MK `,` $shapeSFA_MK `,` $shapeB_NK `,`
      $shapeSFB_NK `,` $shapeC_MN `)`
    `frgs` `=` `(` $frgA `,` $frgSFA `,` $frgB `,` $frgSFB `,` $frgC `)` `>`
  }];
}

def Cute_SparseMmaAtomAttr : Cute_Attr<"SparseMmaAtom", "sparse_mma_atom"> {
  let summary = "MMA atom attribute with a sparsity-metadata operand";
  let description = [{
    Operands in the order A, B, C, E.

    ```mlir
    #cute.sparse_mma_atom<<"(16,8,16)">, <"32:1">, layout_TV = (<"1:0">, <"1:0">, <"1:0">, <"1:0">), shapes = (<"(16,16)">, <"(8,16)">, <"(16,8)">, <"1">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>
    ```
  }];
  let parameters = (ins
    Cute_Param<"ShapeAttr", "MNK shape">:$shape_MNK,
    Cute_Param<"LayoutAttr", "thread id layout">:$thr_id,
    Cute_Param<"LayoutAttr", "A thread-value layout">:$layoutA_TV,
    Cute_Param<"LayoutAttr", "B thread-value layout">:$layoutB_TV,
    Cute_Param<"LayoutAttr", "C thread-value layout">:$layoutC_TV,
    Cute_Param<"LayoutAttr", "E thread-value layout">:$layoutE_TV,
    Cute_Param<"ShapeAttr", "A shape (MK)">:$shapeA_MK,
    Cute_Param<"ShapeAttr", "B shape (NK)">:$shapeB_NK,
    Cute_Param<"ShapeAttr", "C shape (MN)">:$shapeC_MN,
    Cute_Param<"ShapeAttr", "E shape (MK)">:$shapeE_MK,
    Cute_Param<"mlir::Attribute", "A fragment">:$frgA,
    Cute_Param<"mlir::Attribute", "B fragment">:$frgB,
    Cute_Param<"mlir::Attribute", "C fragment">:$frgC,
    Cute_Param<"mlir::Attribute", "E fragment">:$frgE
  );
  let assemblyFormat = [{
    `<` $shape_MNK `,` $thr_id `,`
    `layout_TV` `=` `(` $layoutA_TV `,` $layoutB_TV `,` $layoutC_TV `,`
      $layoutE_TV `)` `,`
    `shapes` `=` `(` $shapeA_MK `,` $shapeB_NK `,` $shapeC_MN `,` $shapeE_MK `)`
    `frgs` `=` `(` $frgA `,` $frgB `,` $frgC `,` $frgE `)` `>`
  }];
}

def Cute_SparseMxMmaAtomAttr
    : Cute_Attr<"SparseMxMmaAtom", "sparse_mx_mma_atom"> {
  let summary = "MMA atom attribute with scale-factor and metadata operands";
  let description = [{
    Operands in the order A, SFA, B, SFB, C, E.

    ```mlir
    #cute.sparse_mx_mma_atom<<"(16,8,16)">, <"32:1">, layout_TV = (<"1:0">, <"2:0">, <"3:0">, <"4:0">, <"5:0">, <"6:0">), shapes = (<"1">, <"2">, <"3">, <"4">, <"5">, <"6">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>
    ```
  }];
  let parameters = (ins
    Cute_Param<"ShapeAttr", "MNK shape">:$shape_MNK,
    Cute_Param<"LayoutAttr", "thread id layout">:$thr_id,
    Cute_Param<"LayoutAttr", "A thread-value layout">:$layoutA_TV,
    Cute_Param<"LayoutAttr", "SFA thread-value layout">:$layoutSFA_TV,
    Cute_Param<"LayoutAttr", "B thread-value layout">:$layoutB_TV,
    Cute_Param<"LayoutAttr", "SFB thread-value layout">:$layoutSFB_TV,
    Cute_Param<"LayoutAttr", "C thread-value layout">:$layoutC_TV,
    Cute_Param<"LayoutAttr", "E thread-value layout">:$layoutE_TV,
    Cute_Param<"ShapeAttr", "A shape (MK)">:$shapeA_MK,
    Cute_Param<"ShapeAttr", "SFA shape (MK)">:$shapeSFA_MK,
    Cute_Param<"ShapeAttr", "B shape (NK)">:$shapeB_NK,
    Cute_Param<"ShapeAttr", "SFB shape (NK)">:$shapeSFB_NK,
    Cute_Param<"ShapeAttr", "C shape (MN)">:$shapeC_MN,
    Cute_Param<"ShapeAttr", "E shape (MK)">:$shapeE_MK,
    Cute_Param<"mlir::Attribute", "A fragment">:$frgA,
    Cute_Param<"mlir::Attribute", "SFA fragment">:$frgSFA,
    Cute_Param<"mlir::Attribute", "B fragment">:$frgB,
    Cute_Param<"mlir::Attribute", "SFB fragment">:$frgSFB,
    Cute_Param<"mlir::Attribute", "C fragment">:$frgC,
    Cute_Param<"mlir::Attribute", "E fragment">:$frgE
  );
  let assemblyFormat = [{
    `<` $shape_MNK `,` $thr_id `,`
    `layout_TV` `=` `(` $layoutA_TV `,` $layoutSFA_TV `,` $layoutB_TV `,`
      $layoutSFB_TV `,` $layoutC_TV `,` $layoutE_TV `)` `,`
    `shapes` `=` `(` $shapeA_MK `,` $shapeSFA_MK `,` $shapeB_NK `,`
      $shapeSFB_NK `,` $shapeC_MN `,` $shapeE_MK `)`
    `frgs` `=` `(` $frgA `,` $frgSFA `,` $frgB `,` $frgSFB `,` $frgC `,`
      $frgE `)` `>`
  }];
}

#endif // CUTE_IR_GENERATED_CUTE_ATTRS_COMPILER_TD
