// The dialect has one generated attribute parser and printer, so the
// release's attributes, the compiler's and its enum attributes are generated
// together from here.

include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"
include "mlir/dialects/cute_ir/CuteAttrsCompiler.td"
include "mlir/dialects/cute_ir/CuteEnums.td"
