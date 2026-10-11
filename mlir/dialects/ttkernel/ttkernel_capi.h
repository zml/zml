#ifndef ZML_MLIR_DIALECTS_TTKERNEL_TTKERNEL_CAPI_H
#define ZML_MLIR_DIALECTS_TTKERNEL_TTKERNEL_CAPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "ttlang-c/Dialects.h"
#include "ttlang-c/TTKernelTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the dialects a tt-lang kernel module may hold besides ttcore,
/// ttkernel and ttl (emitc, memref, tensor, index, ub...).
MLIR_CAPI_EXPORTED void mlirTTLangRegisterUpstreamDialects(MlirDialectRegistry registry);

/// Registers tt-lang's passes and pipelines once per process.
MLIR_CAPI_EXPORTED void mlirTTLangRegisterPasses(void);

/// `!ttcore.tile<height x width, data_type>`; `data_type` is a `ttcore::DataType`.
MLIR_CAPI_EXPORTED MlirType mlirTTCoreTileTypeGet(MlirContext ctx, int64_t height, int64_t width, uint32_t data_type);
MLIR_CAPI_EXPORTED bool mlirTypeIsATTCoreTile(MlirType type);
MLIR_CAPI_EXPORTED uint32_t mlirTTCoreTileTypeGetDataType(MlirType type);
MLIR_CAPI_EXPORTED int64_t mlirTTCoreTileTypeGetSizeBytes(MlirType type);

/// `#ttcore.arch<arch>`; `arch` is a `ttcore::Arch`.
MLIR_CAPI_EXPORTED MlirAttribute mlirTTCoreArchAttrGet(MlirContext ctx, uint32_t arch);

/// `!ttkernel.cb<num_elements, element_type>`.
MLIR_CAPI_EXPORTED MlirType mlirTTKernelCBTypeGet(MlirContext ctx, int64_t num_elements, MlirType element_type);
MLIR_CAPI_EXPORTED bool mlirTypeIsATTKernelCB(MlirType type);
MLIR_CAPI_EXPORTED int64_t mlirTTKernelCBTypeGetNumElements(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTTKernelCBTypeGetElementType(MlirType type);

/// `#ttkernel.bcast_type<value>`.
MLIR_CAPI_EXPORTED MlirAttribute mlirTTKernelBcastTypeAttrGet(MlirContext ctx, uint32_t value);

/// Runs `convert-ttkernel-to-emitc` on the module.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTTKernelConvertToEmitC(MlirModule module);

/// Writes the C++ source of the kernel `func.func` named `name` (already
/// lowered to EmitC) through `callback`.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTTKernelTranslateToCpp(MlirModule module, MlirStringRef name, MlirStringCallback callback, void *user_data);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_TTKERNEL_TTKERNEL_CAPI_H
