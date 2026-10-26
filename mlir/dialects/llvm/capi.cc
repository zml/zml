//===- capi.cc - C API for the llvm dialect -------------------------------===//

#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

static MLIRContext *loadLLVM(MlirContext context) {
  MLIRContext *ctx = unwrap(context);
  if (ctx)
    ctx->getOrLoadDialect<LLVM::LLVMDialect>();
  return ctx;
}

template <typename T> static T optional(MlirAttribute attr) {
  return llvm::dyn_cast_or_null<T>(unwrap(attr));
}

bool mlirAttributeIsALLVMLoopUnroll(MlirAttribute attr) {
  return llvm::isa_and_nonnull<LLVM::LoopUnrollAttr>(unwrap(attr));
}

MlirAttribute mlirLLVMLoopUnrollAttrGet(MlirContext context,
                                        MlirAttribute disable,
                                        MlirAttribute count,
                                        MlirAttribute runtimeDisable,
                                        MlirAttribute full) {
  MLIRContext *ctx = loadLLVM(context);
  if (!ctx)
    return {nullptr};
  return wrap(LLVM::LoopUnrollAttr::get(ctx, optional<BoolAttr>(disable),
                                        optional<IntegerAttr>(count),
                                        optional<BoolAttr>(runtimeDisable),
                                        optional<BoolAttr>(full), {}, {}, {}));
}

bool mlirAttributeIsALLVMLoopAnnotation(MlirAttribute attr) {
  return llvm::isa_and_nonnull<LLVM::LoopAnnotationAttr>(unwrap(attr));
}

MlirAttribute mlirLLVMLoopAnnotationAttrGet(MlirContext context,
                                            MlirAttribute disableNonforced,
                                            MlirAttribute unroll,
                                            MlirAttribute mustProgress) {
  MLIRContext *ctx = loadLLVM(context);
  if (!ctx)
    return {nullptr};
  return wrap(LLVM::LoopAnnotationAttr::get(
      ctx, optional<BoolAttr>(disableNonforced), {}, {},
      optional<LLVM::LoopUnrollAttr>(unroll), {}, {}, {}, {}, {}, {},
      optional<BoolAttr>(mustProgress), {}, {}, {}, {}));
}
