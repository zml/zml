//===- capi.cc - C API for the func dialect -------------------------------===//

#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

void mlirRegisterFuncExtensions(MlirDialectRegistry registry) {
  mlir::func::registerAllExtensions(*unwrap(registry));
}
