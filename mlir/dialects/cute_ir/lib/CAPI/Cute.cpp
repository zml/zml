//===- Cute.cpp - C API for the CuTe dialect -----------------------------===//

#include "cute_ir-c/Dialect/Cute.h"

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(
    Cute, cute, mlir::cutlass_compiler::cute::CuteDialect)

MlirType mlirCuteTypeParse(MlirContext context, MlirStringRef assembly) {
  return mlirTypeParseGet(context, assembly);
}

bool mlirTypeIsACuteType(MlirType type) {
  mlir::Type cppType = unwrap(type);
  return cppType &&
         cppType.getDialect().getNamespace() ==
             mlir::cutlass_compiler::cute::CuteDialect::getDialectNamespace();
}
