#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>

extern "C" void mlirRegisterFuncExtensions(MlirDialectRegistry registry) {
  mlir::func::registerAllExtensions(*unwrap(registry));
}
