#include "tenstorrent_capi.h"

#include "mlir/CAPI/Registration.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TenstorrentIR, ttir,
                                    mlir::tt::ttir::TTIRDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TenstorrentCore, ttcore,
                                    mlir::tt::ttcore::TTCoreDialect)
