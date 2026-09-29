// C entry points for what NVIDIA's cuda_tile C API leaves out: the
// `TileView` type interface, shared by the partition, strided and
// gather/scatter views.
#ifndef ZML_MLIR_DIALECTS_CUDA_TILE_CAPI_H_
#define ZML_MLIR_DIALECTS_CUDA_TILE_CAPI_H_

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

bool zmlCudaTileTypeIsATileView(MlirType type);
MlirType zmlCudaTileTileViewGetViewTileType(MlirType type);
intptr_t zmlCudaTileTileViewGetViewIndexRank(MlirType type);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_CUDA_TILE_CAPI_H_
