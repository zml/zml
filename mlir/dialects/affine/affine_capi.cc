#include "affine_capi.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Affine, affine, mlir::affine::AffineDialect)
