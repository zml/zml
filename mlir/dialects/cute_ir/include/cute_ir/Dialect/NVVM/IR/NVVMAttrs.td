//===- NVVMAttrs.td - NVVM attributes of the CuTe DSL ------*- tablegen -*-===//
//
// Generated for nvidia-cutlass-dsl 4.8.0 from the DSL compiler
// (`_cutlass_ir.cu13` parser messages, its mnemonics, and probing its
// attribute parser and printer); do not edit by hand.
//
// The enum attributes are NVVMEnums.td. These are the compiler's other nvvm
// attributes: every mnemonic its parser knows is one of the two.
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_IR_DIALECT_NVVM_IR_NVVM_ATTRS_TD
#define CUTE_IR_DIALECT_NVVM_IR_NVVM_ATTRS_TD

include "cute_ir/Dialect/NVVM/IR/NVVMDialect.td"
include "cute_ir/Dialect/NVVM/IR/NVVMEnums.td"

def NVVM_MMAShapeAttr : NVVM_Attr<"MMAShape", "shape"> {
  let summary = "Shape of an MMA operation";
  let description = "`#nvvm.shape<m = 16, n = 8, k = 16>`";
  let parameters = (ins "int":$m, "int":$n, "int":$k);
  let assemblyFormat = "`<` struct(params) `>`";
}

def NVVM_LdStMatrixShapeAttr : NVVM_Attr<"LdStMatrixShape", "ld_st_matrix_shape"> {
  let summary = "Matrix shape of ldmatrix, stmatrix and movmatrix";
  let description = "`#nvvm.ld_st_matrix_shape<m = 8, n = 8>`";
  let parameters = (ins "int":$m, "int":$n);
  let assemblyFormat = "`<` struct(params) `>`";
}

// Defaulted fields print only when set; the defaults and the verifier's rules
// and messages are the compiler's.
def NVVM_TargetAttr : NVVM_Attr<"Target", "target"> {
  let summary = "GPU target of an NVVM module";
  let description = [{
    `#nvvm.target<O = 3, triple = "nvptx64-nvidia-cuda", chip = "sm_100a", features = "+ptx88", flags = {}, link = [], verifyTarget = true>`
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2">:$O,
    StringRefParameter<"target triple", "\"nvptx64-nvidia-cuda\"">:$triple,
    StringRefParameter<"target chip", "\"sm_75\"">:$chip,
    StringRefParameter<"target features", "\"\"">:$features,
    OptionalParameter<"::mlir::DictionaryAttr">:$flags,
    OptionalParameter<"::mlir::ArrayAttr">:$link,
    DefaultValuedParameter<"bool", "false">:$verifyTarget
  );
  let assemblyFormat = "(`<` struct(params)^ `>`)?";
  let genVerifyDecl = 1;
}

#endif // CUTE_IR_DIALECT_NVVM_IR_NVVM_ATTRS_TD
