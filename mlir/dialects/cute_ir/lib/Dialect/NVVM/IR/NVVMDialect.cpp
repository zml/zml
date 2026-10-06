//===- NVVMDialect.cpp - NVIDIA's nvvm dialect of the CuTe DSL -----------===//

#include "cute_ir/Dialect/NVVM/IR/NVVMDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::nvvm;

#include "cute_ir/Dialect/NVVM/IR/NVVMDialect.cpp.inc"

#include "cute_ir/Dialect/NVVM/IR/NVVMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cute_ir/Dialect/NVVM/IR/NVVMAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/NVVM/IR/NVVMOps.cpp.inc"

void NVVMDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cute_ir/Dialect/NVVM/IR/NVVMAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cute_ir/Dialect/NVVM/IR/NVVMOps.cpp.inc"
      >();
}

// The compiler's rules and messages.
LogicalResult TargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int O, StringRef triple, StringRef chip,
                                 StringRef, DictionaryAttr, ArrayAttr link, bool) {
  if (O < 0 || O > 3)
    return emitError() << "The optimization level must be a number between 0 and 3.";
  if (triple.empty())
    return emitError() << "The target triple cannot be empty.";
  if (chip.empty())
    return emitError() << "The target chip cannot be empty.";
  if (link && !llvm::all_of(link, [](Attribute attr) { return attr && isa<StringAttr>(attr); }))
    return emitError() << "All the elements in the `link` array must be strings.";
  return success();
}
