//===- CuteNVGPUDialect.h - CuTe NVIDIA GPU dialect -------------*- C++ -*-===//

#ifndef MLIR_DIALECTS_CUTE_NVGPU_CUTENVGPUDIALECT_H
#define MLIR_DIALECTS_CUTE_NVGPU_CUTENVGPUDIALECT_H

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

#include "mlir/dialects/cute_nvgpu/CuteNVGPUDialect.h.inc"

#include "mlir/dialects/cute_nvgpu/CuteNVGPUEnums.h.inc"

namespace mlir::cutlass_compiler::cute_nvgpu {
/// Parses `true` or `false`.
FailureOr<bool> parseBool(AsmParser &parser);
/// The TMA data format a non-executable TMA atom takes for an element type
/// when none is given, if there is one.
std::optional<TmaDataFormat> defaultTmaDataFormat(Type elementType);
} // namespace mlir::cutlass_compiler::cute_nvgpu

#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/cute_nvgpu/CuteNVGPUAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/cute_nvgpu/CuteNVGPUTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/dialects/cute_nvgpu/CuteNVGPUOps.h.inc"

#endif // MLIR_DIALECTS_CUTE_NVGPU_CUTENVGPUDIALECT_H
