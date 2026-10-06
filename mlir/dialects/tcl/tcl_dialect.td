#ifndef ZML_MLIR_DIALECTS_TCL_TCL_DIALECT
#define ZML_MLIR_DIALECTS_TCL_TCL_DIALECT
include "mlir/IR/DialectBase.td"

// Native TCL: tensors over named axes, `tk.*` tensor operations and graph
// operations, one-to-one with the TCL source the Furiosa compiler takes. The
// subset of xla/backends/furiosa/codegen/tcl/ir that ZML emits.
def Tcl_Dialect : Dialect {
  let name = "tcl";
  let cppNamespace = "::xla::furiosa::tcl";
  let summary = "Typed Furiosa Tensor Compute Language graph";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}
#endif
