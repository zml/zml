// C entry points for what NVIDIA's cuda_tile C API leaves out: the
// `TileView` type interface, shared by the partition, strided and
// gather/scatter views, the gather/scatter view type, the general
// `optimization_hints` constructor and the `assume` predicates.
#ifndef ZML_MLIR_DIALECTS_CUDA_TILE_CAPI_H_
#define ZML_MLIR_DIALECTS_CUDA_TILE_CAPI_H_

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Constructors return null for invalid parameters; optional attribute
// parameters take null handles, and std::optional integers a `has` flag.
// Getters require a value accepted by the matching IsA.

bool mlirTypeIsACudaTileTileView(MlirType type);
MlirType mlirCudaTileTileViewTypeGetViewTileType(MlirType type);
intptr_t mlirCudaTileTileViewTypeGetViewIndexRank(MlirType type);

// `!cuda_tile.gather_scatter_view`. `paddingValue` is a
// `#cuda_tile.padding_value` attribute, or null for none.
bool mlirTypeIsACudaTileGatherScatterView(MlirType type);
MlirType mlirCudaTileGatherScatterViewTypeGet(MlirContext context,
                                              intptr_t rank,
                                              const int32_t *tileShape,
                                              MlirType tensorView,
                                              uint32_t sparseDim,
                                              MlirAttribute paddingValue);
// A DenseI32ArrayAttr.
MlirAttribute mlirCudaTileGatherScatterViewTypeGetTileShape(MlirType type);
MlirType mlirCudaTileGatherScatterViewTypeGetTensorView(MlirType type);
uint32_t mlirCudaTileGatherScatterViewTypeGetSparseDim(MlirType type);
// Null when the view has no padding value.
MlirAttribute mlirCudaTileGatherScatterViewTypeGetPaddingValue(MlirType type);

// `#cuda_tile.optimization_hints` over its dictionary of per-architecture
// dictionaries (IsA: NVIDIA's mlirCudaTileAttributeIsAOptimizationHintsAttr).
MlirAttribute mlirCudaTileOptimizationHintsAttrGet(MlirContext context,
                                                   MlirAttribute value);
MlirAttribute mlirCudaTileOptimizationHintsAttrGetValue(MlirAttribute attr);

// `#cuda_tile.div_by`. `every` and `along` are both present or both absent.
bool mlirAttributeIsACudaTileDivBy(MlirAttribute attr);
MlirAttribute mlirCudaTileDivByAttrGet(MlirContext context, uint64_t divisor,
                                       bool hasEvery, int64_t every,
                                       bool hasAlong, int64_t along);
uint64_t mlirCudaTileDivByAttrGetDivisor(MlirAttribute attr);
bool mlirCudaTileDivByAttrHasEvery(MlirAttribute attr);
int64_t mlirCudaTileDivByAttrGetEvery(MlirAttribute attr);
bool mlirCudaTileDivByAttrHasAlong(MlirAttribute attr);
int64_t mlirCudaTileDivByAttrGetAlong(MlirAttribute attr);

// `#cuda_tile.same_elements`.
bool mlirAttributeIsACudaTileSameElements(MlirAttribute attr);
MlirAttribute mlirCudaTileSameElementsAttrGet(MlirContext context,
                                              intptr_t rank,
                                              const int64_t *values);
// A DenseI64ArrayAttr.
MlirAttribute mlirCudaTileSameElementsAttrGetValues(MlirAttribute attr);

// `#cuda_tile.bounded`, each bound optional.
bool mlirAttributeIsACudaTileBounded(MlirAttribute attr);
MlirAttribute mlirCudaTileBoundedAttrGet(MlirContext context, bool hasLb,
                                         int64_t lb, bool hasUb, int64_t ub);
bool mlirCudaTileBoundedAttrHasLb(MlirAttribute attr);
int64_t mlirCudaTileBoundedAttrGetLb(MlirAttribute attr);
bool mlirCudaTileBoundedAttrHasUb(MlirAttribute attr);
int64_t mlirCudaTileBoundedAttrGetUb(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_CUDA_TILE_CAPI_H_
