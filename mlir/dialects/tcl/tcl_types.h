#ifndef ZML_MLIR_DIALECTS_TCL_TCL_TYPES_H_
#define ZML_MLIR_DIALECTS_TCL_TCL_TYPES_H_
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/dialects/tcl/tcl_attrs.h"
#include "mlir/dialects/tcl/tcl_dialect.h"
#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/tcl/tcl_types.h.inc"
namespace xla::furiosa::tcl {
// Empty name denotes a type outside the SDK's 17 element types.
llvm::StringRef ElementName(mlir::Type type);
}  // namespace xla::furiosa::tcl
#endif  // ZML_MLIR_DIALECTS_TCL_TCL_TYPES_H_
