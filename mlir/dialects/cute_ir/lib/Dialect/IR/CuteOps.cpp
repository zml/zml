//===- CuteOps.cpp - the compiler's CuTe operations -----------------------===//

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
// Operand constraints name cute_nvgpu and LLVM types.
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/Cute/IR/CuteOps.cpp.inc"
