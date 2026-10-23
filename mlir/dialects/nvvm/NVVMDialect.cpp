//===- NVVMDialect.cpp - NVIDIA's nvvm dialect of the CuTe DSL ------------===//

#include "mlir/dialects/nvvm/NVVMDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::nvvm;

#include "mlir/dialects/nvvm/NVVMDialect.cpp.inc"

#include "mlir/dialects/nvvm/NVVMEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/nvvm/NVVMAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/dialects/nvvm/NVVMOps.cpp.inc"

void NVVMDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/dialects/nvvm/NVVMAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/dialects/nvvm/NVVMOps.cpp.inc"
      >();
}

// The compiler's rules and messages.
LogicalResult TargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int O, StringRef triple, StringRef chip,
                                 StringRef, DictionaryAttr, ArrayAttr link,
                                 bool) {
  if (O < 0 || O > 3)
    return emitError()
           << "The optimization level must be a number between 0 and 3.";
  if (triple.empty())
    return emitError() << "The target triple cannot be empty.";
  if (chip.empty())
    return emitError() << "The target chip cannot be empty.";
  if (link && !llvm::all_of(link, [](Attribute attr) {
        return attr && isa<StringAttr>(attr);
      }))
    return emitError()
           << "All the elements in the `link` array must be strings.";
  return success();
}
