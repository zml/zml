//===- CuteNVGPUDialect.h - CuTe NVIDIA GPU dialect ------------*- C++ -*-===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_H
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUTypes.h.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUOps.h.inc"

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_H
