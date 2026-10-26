//===- capi.h - C API for the cuda_tile dialect -------------------*- C -*-===//
//
// C entry points for what NVIDIA's cuda_tile C API leaves out: the
// `TileView` type interface, shared by the partition, strided and
// gather/scatter views, the gather/scatter view type, the general
// `optimization_hints` constructor and the `assume` predicates.

#ifndef MLIR_DIALECTS_CUDA_TILE_CAPI_H
#define MLIR_DIALECTS_CUDA_TILE_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters; optional attribute
// parameters take null handles, and std::optional integers a `has` flag.
// Getters require a value accepted by the matching IsA.

MLIR_CAPI_EXPORTED bool mlirTypeIsACudaTileTileView(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCudaTileTileViewTypeGetViewTileType(MlirType type);
MLIR_CAPI_EXPORTED intptr_t
mlirCudaTileTileViewTypeGetViewIndexRank(MlirType type);

// `!cuda_tile.gather_scatter_view`. `paddingValue` is a
// `#cuda_tile.padding_value` attribute, or null for none.
MLIR_CAPI_EXPORTED bool mlirTypeIsACudaTileGatherScatterView(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirCudaTileGatherScatterViewTypeGet(
    MlirContext context, intptr_t rank, const int32_t *tileShape,
    MlirType tensorView, uint32_t sparseDim, MlirAttribute paddingValue);
// A DenseI32ArrayAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileGatherScatterViewTypeGetTileShape(MlirType type);
MLIR_CAPI_EXPORTED MlirType
mlirCudaTileGatherScatterViewTypeGetTensorView(MlirType type);
MLIR_CAPI_EXPORTED uint32_t
mlirCudaTileGatherScatterViewTypeGetSparseDim(MlirType type);
// Null when the view has no padding value.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileGatherScatterViewTypeGetPaddingValue(MlirType type);

// `#cuda_tile.optimization_hints` over its dictionary of per-architecture
// dictionaries (IsA: NVIDIA's mlirCudaTileAttributeIsAOptimizationHintsAttr).
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileOptimizationHintsAttrGet(MlirContext context, MlirAttribute value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileOptimizationHintsAttrGetValue(MlirAttribute attr);

// `#cuda_tile.div_by`. `every` and `along` are both present or both absent.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaTileDivBy(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileDivByAttrGet(MlirContext context, uint64_t divisor, bool hasEvery,
                         int64_t every, bool hasAlong, int64_t along);
MLIR_CAPI_EXPORTED uint64_t mlirCudaTileDivByAttrGetDivisor(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirCudaTileDivByAttrHasEvery(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirCudaTileDivByAttrGetEvery(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirCudaTileDivByAttrHasAlong(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirCudaTileDivByAttrGetAlong(MlirAttribute attr);

// `#cuda_tile.same_elements`.
MLIR_CAPI_EXPORTED bool
mlirAttributeIsACudaTileSameElements(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaTileSameElementsAttrGet(
    MlirContext context, intptr_t rank, const int64_t *values);
// A DenseI64ArrayAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirCudaTileSameElementsAttrGetValues(MlirAttribute attr);

// `#cuda_tile.bounded`, each bound optional.
MLIR_CAPI_EXPORTED bool mlirAttributeIsACudaTileBounded(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirCudaTileBoundedAttrGet(
    MlirContext context, bool hasLb, int64_t lb, bool hasUb, int64_t ub);
MLIR_CAPI_EXPORTED bool mlirCudaTileBoundedAttrHasLb(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirCudaTileBoundedAttrGetLb(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirCudaTileBoundedAttrHasUb(MlirAttribute attr);
MLIR_CAPI_EXPORTED int64_t mlirCudaTileBoundedAttrGetUb(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_CUDA_TILE_CAPI_H
