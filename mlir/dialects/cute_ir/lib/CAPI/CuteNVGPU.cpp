//===- CuteNVGPU.cpp - C API for the CuTe NVIDIA GPU dialect ------------===//

#include "cute_ir-c/Dialect/CuteNVGPU.h"

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    CuteNVGPU, cute_nvgpu,
    mlir::cutlass_compiler::cute_nvgpu::CuteNVGPUDialect)

MlirType mlirCuteNVGPUTypeParse(MlirContext context, MlirStringRef assembly) {
  return mlirTypeParseGet(context, assembly);
}

bool mlirTypeIsACuteNVGPUType(MlirType type) {
  mlir::Type cppType = unwrap(type);
  if (!cppType)
    return false;
  if (auto opaque = mlir::dyn_cast<mlir::OpaqueType>(cppType))
    return opaque.getDialectNamespace() == "cute_nvgpu";
  return cppType.getDialect().getNamespace() == "cute_nvgpu";
}

