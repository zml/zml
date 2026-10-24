//===- AIROps.cpp - Metal AIR operations ----------------------------------===//

#include "mlir/dialects/air/AIROps.h"

#include "mlir/dialects/air/AIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::metal::air;

#define GET_OP_CLASSES
#include "mlir/dialects/air/AIROps.cpp.inc"

LogicalResult DotOp::verify() {
  if (getElementTypeOrSelf(getX().getType()) != getResult().getType())
    return emitOpError("expects the operands' element type as result");
  return success();
}

static LogicalResult verifyConvert(Operation *op, bool toOrFromInteger) {
  Type input = op->getOperand(0).getType();
  Type result = op->getResult(0).getType();
  if (failed(verifyCompatibleShape(input, result)) ||
      isa<VectorType>(input) != isa<VectorType>(result))
    return op->emitOpError("expects operands and results of the same shape");
  bool inputInt = isa<IntegerType>(getElementTypeOrSelf(input));
  bool resultInt = isa<IntegerType>(getElementTypeOrSelf(result));
  if (!toOrFromInteger && (inputInt || resultInt || input == result))
    return op->emitOpError("converts between distinct float types");
  if (toOrFromInteger && inputInt == resultInt)
    return op->emitOpError("converts between an integer and a float type");
  if (resultInt && cast<arith::ArithFastMathInterface>(op)
                           .getFastMathFlagsAttr()
                           .getValue() != arith::FastMathFlags::none)
    return op->emitOpError("integer results take no fastmath flags");
  return success();
}

LogicalResult ConvertOp::verify() { return verifyConvert(*this, false); }
LogicalResult ConvertSOp::verify() { return verifyConvert(*this, true); }
LogicalResult ConvertUOp::verify() { return verifyConvert(*this, true); }

LogicalResult SimdgroupMatrixMultiplyAccumulateOp::verify() {
  auto element = [](Value value) {
    return cast<VectorType>(value.getType()).getElementType();
  };
  Type d = element(getResult()), a = element(getA()), b = element(getB()),
       c = element(getC());
  // The combinations Apple's compiler emits for simdgroup_multiply_accumulate.
  bool mixed = d.isF32() && c.isF32() && (a.isF32() || a == b) &&
               (b.isF16() || b.isBF16() || b.isF32());
  bool half = d.isF16() && a.isF16() && b.isF16() && c.isF16();
  if (!mixed && !half)
    return emitOpError("unsupported element types ")
           << a << " * " << b << " + " << c << " -> " << d;
  return success();
}
