#include "vector_capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace {

mlir::MLIRContext *loadVector(MlirContext ctx) {
  mlir::MLIRContext *context = unwrap(ctx);
  if (context) context->getOrLoadDialect<mlir::vector::VectorDialect>();
  return context;
}

}  // namespace

bool mlirAttributeIsAVectorCombiningKind(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::vector::CombiningKindAttr>(unwrap(attr));
}

MlirAttribute mlirVectorCombiningKindAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *context = loadVector(ctx);
  std::optional<mlir::vector::CombiningKind> kind = mlir::vector::symbolizeCombiningKind(value);
  if (!context || !kind) return {nullptr};
  return wrap(mlir::vector::CombiningKindAttr::get(context, *kind));
}

uint32_t mlirVectorCombiningKindAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<mlir::vector::CombiningKindAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAVectorIteratorType(MlirAttribute attr) {
  return llvm::isa_and_nonnull<mlir::vector::IteratorTypeAttr>(unwrap(attr));
}

MlirAttribute mlirVectorIteratorTypeAttrGet(MlirContext ctx, uint32_t value) {
  mlir::MLIRContext *context = loadVector(ctx);
  std::optional<mlir::vector::IteratorType> type = mlir::vector::symbolizeIteratorType(value);
  if (!context || !type) return {nullptr};
  return wrap(mlir::vector::IteratorTypeAttr::get(context, *type));
}

uint32_t mlirVectorIteratorTypeAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(llvm::cast<mlir::vector::IteratorTypeAttr>(unwrap(attr)).getValue());
}
