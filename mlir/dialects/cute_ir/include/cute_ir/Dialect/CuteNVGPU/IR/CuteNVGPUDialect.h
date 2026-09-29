//===- CuteNVGPUDialect.h - CuTe NVIDIA GPU dialect ------------*- C++ -*-===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_H
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_H

#include <optional>

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h.inc"

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUEnums.h.inc"

namespace mlir::cutlass_compiler::cute_nvgpu {
/// Parses `true` or `false`.
FailureOr<bool> parseBool(AsmParser &parser);
/// The TMA data format a non-executable TMA atom takes for an element type
/// when none is given, if there is one.
std::optional<TmaDataFormat> defaultTmaDataFormat(Type elementType);
}  // namespace mlir::cutlass_compiler::cute_nvgpu

#define GET_ATTRDEF_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUTypes.h.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUOps.h.inc"

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_H
