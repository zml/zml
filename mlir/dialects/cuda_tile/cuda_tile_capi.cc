#include "cuda_tile_capi.h"

#include <optional>

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir::cuda_tile;

namespace {

mlir::MLIRContext *loadCudaTile(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  if (ctx) ctx->getOrLoadDialect<CudaTileDialect>();
  return ctx;
}

std::optional<int64_t> optional(bool has, int64_t value) {
  return has ? std::optional<int64_t>(value) : std::nullopt;
}

}  // namespace

//===----------------------------------------------------------------------===//
// TileView
//===----------------------------------------------------------------------===//

bool mlirTypeIsACudaTileTileView(MlirType type) {
  return llvm::isa<TileView>(unwrap(type));
}

MlirType mlirCudaTileTileViewTypeGetViewTileType(MlirType type) {
  return wrap(llvm::cast<TileView>(unwrap(type)).getViewTileType());
}

intptr_t mlirCudaTileTileViewTypeGetViewIndexRank(MlirType type) {
  return static_cast<intptr_t>(
      llvm::cast<TileView>(unwrap(type)).getViewIndexRank());
}

//===----------------------------------------------------------------------===//
// GatherScatterViewType
//===----------------------------------------------------------------------===//

bool mlirTypeIsACudaTileGatherScatterView(MlirType type) {
  return llvm::isa<GatherScatterViewType>(unwrap(type));
}

MlirType mlirCudaTileGatherScatterViewTypeGet(MlirContext context,
                                              intptr_t rank,
                                              const int32_t *tileShape,
                                              MlirType tensorView,
                                              uint32_t sparseDim,
                                              MlirAttribute paddingValue) {
  mlir::MLIRContext *ctx = loadCudaTile(context);
  if (!ctx) return {nullptr};
  auto view = llvm::dyn_cast_or_null<TensorViewType>(unwrap(tensorView));
  if (!view) return {nullptr};
  PaddingValueAttr padding;
  if (!mlirAttributeIsNull(paddingValue)) {
    padding = llvm::dyn_cast<PaddingValueAttr>(unwrap(paddingValue));
    if (!padding) return {nullptr};
  }
  auto shape = mlir::DenseI32ArrayAttr::get(
      ctx, llvm::ArrayRef<int32_t>(tileShape, rank));
  return wrap(GatherScatterViewType::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx, shape,
      view, sparseDim, padding));
}

MlirAttribute mlirCudaTileGatherScatterViewTypeGetTileShape(MlirType type) {
  return wrap(llvm::cast<GatherScatterViewType>(unwrap(type)).getTileShape());
}

MlirType mlirCudaTileGatherScatterViewTypeGetTensorView(MlirType type) {
  return wrap(llvm::cast<GatherScatterViewType>(unwrap(type)).getTensorView());
}

uint32_t mlirCudaTileGatherScatterViewTypeGetSparseDim(MlirType type) {
  return llvm::cast<GatherScatterViewType>(unwrap(type)).getSparseDim();
}

MlirAttribute mlirCudaTileGatherScatterViewTypeGetPaddingValue(MlirType type) {
  return wrap(llvm::cast<GatherScatterViewType>(unwrap(type)).getPaddingValue());
}

//===----------------------------------------------------------------------===//
// OptimizationHintsAttr
//===----------------------------------------------------------------------===//

MlirAttribute mlirCudaTileOptimizationHintsAttrGet(MlirContext context,
                                                   MlirAttribute value) {
  mlir::MLIRContext *ctx = loadCudaTile(context);
  if (!ctx) return {nullptr};
  auto dict = llvm::dyn_cast_or_null<mlir::DictionaryAttr>(unwrap(value));
  if (!dict) return {nullptr};
  return wrap(OptimizationHintsAttr::getChecked(
      [&] { return mlir::emitError(mlir::UnknownLoc::get(ctx)); }, ctx, dict));
}

MlirAttribute mlirCudaTileOptimizationHintsAttrGetValue(MlirAttribute attr) {
  return wrap(llvm::cast<OptimizationHintsAttr>(unwrap(attr)).getValue());
}

//===----------------------------------------------------------------------===//
// DivByAttr
//===----------------------------------------------------------------------===//

bool mlirAttributeIsACudaTileDivBy(MlirAttribute attr) {
  return llvm::isa<DivByAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileDivByAttrGet(MlirContext context, uint64_t divisor,
                                       bool hasEvery, int64_t every,
                                       bool hasAlong, int64_t along) {
  mlir::MLIRContext *ctx = loadCudaTile(context);
  // The printer writes `every` and `along` together.
  if (!ctx || hasEvery != hasAlong) return {nullptr};
  return wrap(DivByAttr::get(ctx, divisor, optional(hasEvery, every),
                             optional(hasAlong, along)));
}

uint64_t mlirCudaTileDivByAttrGetDivisor(MlirAttribute attr) {
  return llvm::cast<DivByAttr>(unwrap(attr)).getDivisor();
}

bool mlirCudaTileDivByAttrHasEvery(MlirAttribute attr) {
  return llvm::cast<DivByAttr>(unwrap(attr)).getEvery().has_value();
}

int64_t mlirCudaTileDivByAttrGetEvery(MlirAttribute attr) {
  return *llvm::cast<DivByAttr>(unwrap(attr)).getEvery();
}

bool mlirCudaTileDivByAttrHasAlong(MlirAttribute attr) {
  return llvm::cast<DivByAttr>(unwrap(attr)).getAlong().has_value();
}

int64_t mlirCudaTileDivByAttrGetAlong(MlirAttribute attr) {
  return *llvm::cast<DivByAttr>(unwrap(attr)).getAlong();
}

//===----------------------------------------------------------------------===//
// SameElementsAttr
//===----------------------------------------------------------------------===//

bool mlirAttributeIsACudaTileSameElements(MlirAttribute attr) {
  return llvm::isa<SameElementsAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileSameElementsAttrGet(MlirContext context,
                                              intptr_t rank,
                                              const int64_t *values) {
  mlir::MLIRContext *ctx = loadCudaTile(context);
  if (!ctx) return {nullptr};
  return wrap(SameElementsAttr::get(
      ctx, mlir::DenseI64ArrayAttr::get(ctx,
                                        llvm::ArrayRef<int64_t>(values, rank))));
}

MlirAttribute mlirCudaTileSameElementsAttrGetValues(MlirAttribute attr) {
  return wrap(llvm::cast<SameElementsAttr>(unwrap(attr)).getValues());
}

//===----------------------------------------------------------------------===//
// BoundedAttr
//===----------------------------------------------------------------------===//

bool mlirAttributeIsACudaTileBounded(MlirAttribute attr) {
  return llvm::isa<BoundedAttr>(unwrap(attr));
}

MlirAttribute mlirCudaTileBoundedAttrGet(MlirContext context, bool hasLb,
                                         int64_t lb, bool hasUb, int64_t ub) {
  mlir::MLIRContext *ctx = loadCudaTile(context);
  if (!ctx) return {nullptr};
  return wrap(BoundedAttr::get(ctx, optional(hasLb, lb), optional(hasUb, ub)));
}

bool mlirCudaTileBoundedAttrHasLb(MlirAttribute attr) {
  return llvm::cast<BoundedAttr>(unwrap(attr)).getLb().has_value();
}

int64_t mlirCudaTileBoundedAttrGetLb(MlirAttribute attr) {
  return *llvm::cast<BoundedAttr>(unwrap(attr)).getLb();
}

bool mlirCudaTileBoundedAttrHasUb(MlirAttribute attr) {
  return llvm::cast<BoundedAttr>(unwrap(attr)).getUb().has_value();
}

int64_t mlirCudaTileBoundedAttrGetUb(MlirAttribute attr) {
  return *llvm::cast<BoundedAttr>(unwrap(attr)).getUb();
}
