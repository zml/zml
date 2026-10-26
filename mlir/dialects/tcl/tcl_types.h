//===- tcl_types.h - Furiosa TCL types --------------------------*- C++ -*-===//

#ifndef MLIR_DIALECTS_TCL_TCL_TYPES_H
#define MLIR_DIALECTS_TCL_TCL_TYPES_H

#include "mlir/dialects/tcl/tcl_attrs.h"
#include "mlir/dialects/tcl/tcl_dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/tcl/tcl_types.h.inc"
namespace xla::furiosa::tcl {
// Empty name denotes a type outside the SDK's 17 element types.
llvm::StringRef ElementName(mlir::Type type);
} // namespace xla::furiosa::tcl

#endif // MLIR_DIALECTS_TCL_TCL_TYPES_H
