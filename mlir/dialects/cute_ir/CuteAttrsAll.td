//===- CuteAttrsAll.td - all cute attributes ---------------*- tablegen -*-===//
//
// The dialect has one generated attribute parser and printer, so the
// release's attributes, the compiler's and its enum attributes are generated
// together from here.

#ifndef MLIR_DIALECTS_CUTE_IR_CUTEATTRSALL_TD
#define MLIR_DIALECTS_CUTE_IR_CUTEATTRSALL_TD

include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"
include "mlir/dialects/cute_ir/CuteAttrsCompiler.td"
include "mlir/dialects/cute_ir/CuteEnums.td"

#endif // MLIR_DIALECTS_CUTE_IR_CUTEATTRSALL_TD
