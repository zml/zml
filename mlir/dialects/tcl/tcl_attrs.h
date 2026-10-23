//===- tcl_attrs.h - Furiosa TCL attributes ---------------------*- C++ -*-===//

#ifndef MLIR_DIALECTS_TCL_TCL_ATTRS_H
#define MLIR_DIALECTS_TCL_TCL_ATTRS_H

#include "mlir/dialects/tcl/tcl_dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/tcl/tcl_attrs.h.inc"
namespace xla::furiosa::tcl {
struct VeInstructionSpec {
  llvm::StringRef name;
  llvm::StringRef spelling;
  unsigned arity;
};
llvm::ArrayRef<VeInstructionSpec> VeInstructions();

// Schema predicates shared by attribute, type and operation verifiers.
bool IsNumber(mlir::Attribute a);
bool IsExpression(mlir::Attribute a);
bool IsAxis(mlir::Attribute a);
bool IsAxes(mlir::Attribute a);
mlir::LogicalResult
VerifyKeys(llvm::function_ref<mlir::InFlightDiagnostic()> error,
           mlir::DictionaryAttr d, llvm::ArrayRef<llvm::StringRef> keys);
mlir::LogicalResult
VerifyAxes(llvm::function_ref<mlir::InFlightDiagnostic()> error,
           mlir::ArrayAttr axes);
} // namespace xla::furiosa::tcl

#endif // MLIR_DIALECTS_TCL_TCL_ATTRS_H
