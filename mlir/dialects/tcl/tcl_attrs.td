#ifndef ZML_MLIR_DIALECTS_TCL_TCL_ATTRS
#define ZML_MLIR_DIALECTS_TCL_TCL_ATTRS
include "mlir/IR/AttrTypeBase.td"
include "mlir/dialects/tcl/tcl_dialect.td"

def Tcl_Symbol : AttrDef<Tcl_Dialect, "Symbol"> {
  let mnemonic = "symbol";
  let parameters = (ins StringRefParameter<>:$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}
def Tcl_Expr : AttrDef<Tcl_Dialect, "Expr"> {
  let mnemonic = "expr";
  let summary = "Checked symbolic arithmetic or DRAM mapping expression tree";
  let parameters = (ins StringRefParameter<>:$kind, "::mlir::ArrayAttr":$args);
  let assemblyFormat = "`<` $kind `,` $args `>`";
  let genVerifyDecl = 1;
}
def Tcl_Tactic : AttrDef<Tcl_Dialect, "Tactic"> {
  let mnemonic = "tactic";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}
def Tcl_VeOpcode : AttrDef<Tcl_Dialect, "VeOpcode"> {
  let mnemonic = "ve_opcode";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}
def Tcl_ReduceMode : AttrDef<Tcl_Dialect, "ReduceMode"> {
  let mnemonic = "reduce_mode";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Tcl_Predicate : AttrDef<Tcl_Dialect, "Predicate"> {
  let mnemonic = "predicate";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}
def Tcl_Context : AttrDef<Tcl_Dialect, "Context"> {
  let mnemonic = "context";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}
def Tcl_ReadOptions : AttrDef<Tcl_Dialect, "ReadOptions"> {
  let mnemonic = "read_options";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}

def Tcl_Config : AttrDef<Tcl_Dialect, "Config"> {
  let mnemonic = "config";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}
def Tcl_DramMapping : AttrDef<Tcl_Dialect, "DramMapping"> {
  let mnemonic = "dram_mapping";
  let parameters = (ins "::mlir::ArrayAttr":$chip, "::mlir::ArrayAttr":$inner,
                         "::mlir::ArrayAttr":$original);
  let assemblyFormat = "`<` $chip `,` $inner `,` $original `>`";
  let genVerifyDecl = 1;
}
#endif
