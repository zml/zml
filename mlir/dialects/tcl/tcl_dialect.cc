#include "mlir/dialects/tcl/tcl_dialect.h"

#include "llvm/ADT/TypeSwitch.h"            // IWYU pragma: keep
#include "mlir/IR/Builders.h"               // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/dialects/tcl/tcl_attrs.h"    // IWYU pragma: keep
#include "mlir/dialects/tcl/tcl_ops.h"      // IWYU pragma: keep
#include "mlir/dialects/tcl/tcl_types.h"    // IWYU pragma: keep

// Storage definitions must be complete where the dialect registers them.
#include "mlir/dialects/tcl/tcl_dialect.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/tcl/tcl_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/tcl/tcl_types.cc.inc"
namespace xla::furiosa::tcl {
void TclDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/dialects/tcl/tcl_attrs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/dialects/tcl/tcl_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/dialects/tcl/tcl_ops.cc.inc"
      >();
}
}  // namespace xla::furiosa::tcl
