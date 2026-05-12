#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/MemRef.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/Vector.h>
#include <mlir-c/IR.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Pass.h>
#include <mlir-c/Transforms.h>

#ifdef __cplusplus
extern "C" {
#endif

void mlirRegisterFuncExtensions(MlirDialectRegistry registry);

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Affine, affine);

#ifdef __cplusplus
}
#endif
