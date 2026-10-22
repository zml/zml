//===- capi.cc - C API for the vector dialect -----------------------------===//

#include "capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

static mlir::MLIRContext *loadVector(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx)
    ctx->getOrLoadDialect<mlir::vector::VectorDialect>();
  return ctx;
}

bool mlirAttributeIsAVectorCombiningKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::vector::CombiningKindAttr>(unwrap(attr));
}

MlirAttribute mlirVectorCombiningKindAttrGet(MlirContext context,
                                             uint32_t value) {
  mlir::MLIRContext *ctx = loadVector(context);
  std::optional<mlir::vector::CombiningKind> kind =
      mlir::vector::symbolizeCombiningKind(value);
  if (!ctx || !kind)
    return {nullptr};
  return wrap(mlir::vector::CombiningKindAttr::get(ctx, *kind));
}

uint32_t mlirVectorCombiningKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::vector::CombiningKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAVectorIteratorType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::vector::IteratorTypeAttr>(unwrap(attr));
}

MlirAttribute mlirVectorIteratorTypeAttrGet(MlirContext context,
                                            uint32_t value) {
  mlir::MLIRContext *ctx = loadVector(context);
  std::optional<mlir::vector::IteratorType> type =
      mlir::vector::symbolizeIteratorType(value);
  if (!ctx || !type)
    return {nullptr};
  return wrap(mlir::vector::IteratorTypeAttr::get(ctx, *type));
}

uint32_t mlirVectorIteratorTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(
      llvm::cast<mlir::vector::IteratorTypeAttr>(unwrap(attr)).getValue());
}
