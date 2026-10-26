//===- tcl_ops.h - Furiosa TCL operations -----------------------*- C++ -*-===//

#ifndef MLIR_DIALECTS_TCL_TCL_OPS_H
#define MLIR_DIALECTS_TCL_TCL_OPS_H

#include "mlir/dialects/tcl/tcl_attrs.h"   // IWYU pragma: export
#include "mlir/dialects/tcl/tcl_dialect.h" // IWYU pragma: export
#include "mlir/dialects/tcl/tcl_types.h"   // IWYU pragma: export

#include "mlir/Bytecode/BytecodeOpInterface.h"    // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"            // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"                 // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"                 // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"             // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h" // IWYU pragma: keep
#define GET_OP_CLASSES
#include "mlir/dialects/tcl/tcl_ops.h.inc"

#endif // MLIR_DIALECTS_TCL_TCL_OPS_H
