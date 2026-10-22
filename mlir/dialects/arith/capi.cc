//===- capi.cc - C API for the arith dialect ------------------------------===//

#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

bool mlirAttributeIsAArithFastMath(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::arith::FastMathFlagsAttr>(unwrap(attr));
}

MlirAttribute mlirArithFastMathAttrGet(MlirContext context, uint32_t value) {
  mlir::MLIRContext *ctx = unwrap(context);
  std::optional<mlir::arith::FastMathFlags> flags =
      mlir::arith::symbolizeFastMathFlags(value);
  if (!ctx || !flags)
    return {nullptr};
  ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
  return wrap(mlir::arith::FastMathFlagsAttr::get(ctx, *flags));
}

uint32_t mlirArithFastMathAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::arith::FastMathFlagsAttr>(unwrap(attr)).getValue());
}
