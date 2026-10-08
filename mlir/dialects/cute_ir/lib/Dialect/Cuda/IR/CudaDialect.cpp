//===- CudaDialect.cpp - NVIDIA's cuda dialect of the CuTe DSL -----------===//

#include "cute_ir/Dialect/Cuda/IR/CudaDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::cuda;

#include "cute_ir/Dialect/Cuda/IR/CudaDialect.cpp.inc"

#include "cute_ir/Dialect/Cuda/IR/CudaEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cute_ir/Dialect/Cuda/IR/CudaAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/Cuda/IR/CudaTypes.cpp.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/Cuda/IR/CudaOps.cpp.inc"

void CudaDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cute_ir/Dialect/Cuda/IR/CudaAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "cute_ir/Dialect/Cuda/IR/CudaTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cute_ir/Dialect/Cuda/IR/CudaOps.cpp.inc"
      >();
}
