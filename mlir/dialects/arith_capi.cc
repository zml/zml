#include "arith_capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

bool mlirAttributeIsAArithFastMath(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::arith::FastMathFlagsAttr>(unwrap(attr));
}

MlirAttribute mlirArithFastMathAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *context = unwrap(ctx);
  std::optional<mlir::arith::FastMathFlags> flags = mlir::arith::symbolizeFastMathFlags(value);
  if (!context || !flags) return {nullptr};
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  return wrap(mlir::arith::FastMathFlagsAttr::get(context, *flags));
}

uint32_t mlirArithFastMathAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<mlir::arith::FastMathFlagsAttr>(unwrap(attr)).getValue());
}
