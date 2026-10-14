//===- CudaInference.cpp - result types the compiler infers ------------===//
//
// Generated from the nvidia-cutlass-dsl 4.8.0 compiler (its inference
// probed by varying operand types); do not edit. An operation without a
// recovered rule fails inference, so the result types written stand.

#include "cute_ir/Dialect/Cuda/IR/CudaDialect.h"

using namespace mlir;
using namespace ::mlir::cutlass_compiler::cuda;

LogicalResult LaunchCfgCreateFromStreamOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}
