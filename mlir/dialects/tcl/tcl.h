#ifndef ZML_MLIR_DIALECTS_TCL_TCL_H_
#define ZML_MLIR_DIALECTS_TCL_TCL_H_
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/dialects/tcl/tcl_dialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/tcl/tcl_attrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/tcl/tcl_types.h.inc"
#define GET_OP_CLASSES
#include "mlir/dialects/tcl/tcl_ops.h.inc"

namespace xla::furiosa::tcl {
struct VeInstructionSpec {
  llvm::StringRef name;
  llvm::StringRef spelling;
  unsigned arity;
};

// Empty name denotes a type outside the SDK's 17 element types.
llvm::StringRef ElementName(mlir::Type type);
llvm::ArrayRef<VeInstructionSpec> VeInstructions();
}  // namespace xla::furiosa::tcl
#endif
