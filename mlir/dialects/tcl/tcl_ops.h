#ifndef ZML_MLIR_DIALECTS_TCL_TCL_OPS_H_
#define ZML_MLIR_DIALECTS_TCL_TCL_OPS_H_
#include "mlir/Bytecode/BytecodeOpInterface.h"     // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"             // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"                  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"                  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"              // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"  // IWYU pragma: keep
#include "mlir/dialects/tcl/tcl_attrs.h"           // IWYU pragma: export
#include "mlir/dialects/tcl/tcl_dialect.h"         // IWYU pragma: export
#include "mlir/dialects/tcl/tcl_types.h"           // IWYU pragma: export
#define GET_OP_CLASSES
#include "mlir/dialects/tcl/tcl_ops.h.inc"
#endif  // ZML_MLIR_DIALECTS_TCL_TCL_OPS_H_
