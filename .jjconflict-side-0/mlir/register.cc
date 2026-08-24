#include "mlir.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>

extern "C" void mlirRegisterFuncExtensions(MlirDialectRegistry registry) {
  mlir::func::registerAllExtensions(*unwrap(registry));
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Affine, affine,
                                      mlir::affine::AffineDialect)
