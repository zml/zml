#include "ttkernel_capi.h"

#include <mutex>

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "ttlang/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Target/TTKernel/TTKernelToCpp.h"

using namespace mlir::tt;

void mlirTTLangRegisterUpstreamDialects(MlirDialectRegistry registry) { ttlangRegisterUpstreamDialects(registry); }

void mlirTTLangRegisterPasses(void) {
  static std::once_flag once;
  std::call_once(once, [] { ttlangRegisterPasses(); });
}

MlirType mlirTTCoreTileTypeGet(MlirContext ctx, int64_t height, int64_t width, uint32_t data_type) {
  return wrap(ttcore::TileType::get(unwrap(ctx), {height, width}, static_cast<ttcore::DataType>(data_type)));
}

bool mlirTypeIsATTCoreTile(MlirType type) { return mlir::isa<ttcore::TileType>(unwrap(type)); }

uint32_t mlirTTCoreTileTypeGetDataType(MlirType type) {
  return static_cast<uint32_t>(mlir::cast<ttcore::TileType>(unwrap(type)).getDataType());
}

int64_t mlirTTCoreTileTypeGetSizeBytes(MlirType type) {
  return static_cast<int64_t>(mlir::cast<ttcore::TileType>(unwrap(type)).getSizeBytes());
}

MlirAttribute mlirTTCoreArchAttrGet(MlirContext ctx, uint32_t arch) {
  return wrap(ttcore::ArchAttr::get(unwrap(ctx), static_cast<ttcore::Arch>(arch)));
}

MlirType mlirTTKernelCBTypeGet(MlirContext ctx, int64_t num_elements, MlirType element_type) {
  return wrap(ttkernel::CBType::get(unwrap(ctx), num_elements, unwrap(element_type)));
}

bool mlirTypeIsATTKernelCB(MlirType type) { return mlir::isa<ttkernel::CBType>(unwrap(type)); }

int64_t mlirTTKernelCBTypeGetNumElements(MlirType type) {
  return mlir::cast<ttkernel::CBType>(unwrap(type)).getNumElements();
}

MlirType mlirTTKernelCBTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<ttkernel::CBType>(unwrap(type)).getElementType());
}

MlirAttribute mlirTTKernelBcastTypeAttrGet(MlirContext ctx, uint32_t value) {
  return wrap(ttkernel::BcastTypeAttr::get(unwrap(ctx), static_cast<ttkernel::BcastType>(value)));
}

MlirLogicalResult mlirTTKernelConvertToEmitC(MlirModule module) {
  mlir::ModuleOp op = unwrap(module);
  mlir::PassManager pm(op->getName());
  pm.addPass(createConvertTTKernelToEmitC());
  return wrap(pm.run(op));
}

MlirLogicalResult mlirTTKernelTranslateToCpp(MlirModule module, MlirStringRef name, MlirStringCallback callback,
                                             void *user_data) {
  mlir::detail::CallbackOstream stream(callback, user_data);
  return wrap(ttkernel::translateTopLevelKernelToCpp(unwrap(module), stream, unwrap(name)));
}
