#ifndef ZML_MLIR_DIALECTS_TTL_TTL_CAPI_H
#define ZML_MLIR_DIALECTS_TTL_TTL_CAPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "ttlang-c/TTLAttrs.h"

#ifdef __cplusplus
extern "C" {
#endif

/// `!ttl.cb<[shape], element_type, block_count>`.
MLIR_CAPI_EXPORTED MlirType mlirTTLCircularBufferTypeGet(MlirContext ctx, intptr_t rank, const int64_t *shape,
                                                         MlirType element_type, int64_t block_count);
MLIR_CAPI_EXPORTED bool mlirTypeIsATTLCircularBuffer(MlirType type);
MLIR_CAPI_EXPORTED intptr_t mlirTTLCircularBufferTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED int64_t mlirTTLCircularBufferTypeGetDim(MlirType type, intptr_t pos);
MLIR_CAPI_EXPORTED MlirType mlirTTLCircularBufferTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED int64_t mlirTTLCircularBufferTypeGetBlockCount(MlirType type);

/// `!ttl.transfer_handle<kind>`; `kind` is a `ttl::TransferKind`, or -1 for none.
MLIR_CAPI_EXPORTED MlirType mlirTTLTransferHandleTypeGet(MlirContext ctx, int32_t kind);

/// `#ttl.layout<shape = [...], element_type = T, buffer = b, grid = [...], memory = m>`;
/// `buffer_type` and `memory_layout` are `ttl::BufferType` and `ttl::TensorMemoryLayout`.
MLIR_CAPI_EXPORTED MlirAttribute mlirTTLLayoutAttrGet(MlirContext ctx, intptr_t rank, const int64_t *shape,
                                                      MlirType element_type, uint32_t buffer_type, intptr_t grid_rank,
                                                      const int64_t *grid, uint32_t memory_layout);

/// Runs the textual pass pipeline `pipeline` (`builtin.module(...)`) on the
/// module; tt-lang's passes must be registered.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTTLRunPassPipeline(MlirModule module, MlirStringRef pipeline);

/// Describes a module lowered by tt-lang's TTL pipeline as JSON, written
/// through `callback`: its dataflow buffer allocations, pipe resources and,
/// for each kernel function, the attributes the host program is built from.
/// Read before translating the kernels to C++.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirTTLDescribeKernels(MlirModule module, MlirStringCallback callback,
                                                            void *user_data);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_TTL_TTL_CAPI_H
