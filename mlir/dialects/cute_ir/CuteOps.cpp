//===- CuteOps.cpp - the compiler's CuTe operations -----------------------===//

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
// Operand constraints name cute_nvgpu and LLVM types.
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/dialects/cute_nvgpu/CuteNVGPUDialect.h"

#define GET_OP_CLASSES
#include "mlir/dialects/cute_ir/CuteOps.cpp.inc"
