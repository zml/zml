//===- CudaDialect.cpp - NVIDIA's cuda dialect of the CuTe DSL -----------===//

#include "mlir/dialects/cuda/CudaDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::cuda;

#include "mlir/dialects/cuda/CudaDialect.cpp.inc"

#include "mlir/dialects/cuda/CudaEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/cuda/CudaAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/cuda/CudaTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/dialects/cuda/CudaOps.cpp.inc"

void CudaDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/dialects/cuda/CudaAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/dialects/cuda/CudaTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/dialects/cuda/CudaOps.cpp.inc"
      >();
}
