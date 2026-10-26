//===- capi.cc - C API for the ub dialect ---------------------------------===//

#include "capi.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(UB, ub, mlir::ub::UBDialect)
