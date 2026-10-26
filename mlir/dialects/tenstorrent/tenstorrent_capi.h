#ifndef ZML_MLIR_DIALECTS_TENSTORRENT_CAPI_H
#define ZML_MLIR_DIALECTS_TENSTORRENT_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Do not use TT-MLIR's upstream `tt` C handle: Triton already exports it.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TenstorrentIR, ttir);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TenstorrentCore, ttcore);

#ifdef __cplusplus
}
#endif
#endif
