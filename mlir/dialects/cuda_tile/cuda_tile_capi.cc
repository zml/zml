#include "cuda_tile_capi.h"

#include "cuda_tile/Dialect/CudaTile/IR/Types.h"
#include "mlir/CAPI/IR.h"

bool zmlCudaTileTypeIsATileView(MlirType type) {
  return llvm::isa<mlir::cuda_tile::TileView>(unwrap(type));
}

MlirType zmlCudaTileTileViewGetViewTileType(MlirType type) {
  return wrap(llvm::cast<mlir::cuda_tile::TileView>(unwrap(type)).getViewTileType());
}

intptr_t zmlCudaTileTileViewGetViewIndexRank(MlirType type) {
  return static_cast<intptr_t>(
      llvm::cast<mlir::cuda_tile::TileView>(unwrap(type)).getViewIndexRank());
}
