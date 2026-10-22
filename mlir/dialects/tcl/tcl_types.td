#ifndef ZML_MLIR_DIALECTS_TCL_TCL_TYPES
#define ZML_MLIR_DIALECTS_TCL_TCL_TYPES
include "mlir/IR/AttrTypeBase.td"
include "mlir/dialects/tcl/tcl_dialect.td"

def Tcl_Logical : TypeDef<Tcl_Dialect, "Logical"> {
  let mnemonic = "logical";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::ArrayAttr":$axes);
  let assemblyFormat = "`<` $elementType `,` $axes `>`";
  let genVerifyDecl = 1;
}
def Tcl_Mapped : TypeDef<Tcl_Dialect, "Mapped"> {
  let mnemonic = "mapped";
  let parameters = (ins "::mlir::Type":$elementType,
                        "::xla::furiosa::tcl::DramMappingAttr":$mapping);
  let assemblyFormat = "`<` $elementType `,` $mapping `>`";
  let genVerifyDecl = 1;
}
#endif
