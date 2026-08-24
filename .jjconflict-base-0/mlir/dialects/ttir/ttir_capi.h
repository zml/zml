#ifndef ZML_MLIR_DIALECTS_TTIR_TTIR_CAPI_H
#define ZML_MLIR_DIALECTS_TTIR_TTIR_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, tt);

// !tt.ptr<T, addr_space>
MlirType mlirTritonPointerTypeGet(MlirType pointee, int32_t address_space);
MlirType mlirTritonPointerTypeGetPointee(MlirType ptr);
int32_t mlirTritonPointerTypeGetAddressSpace(MlirType ptr);
bool mlirTritonTypeIsAPointer(MlirType t);

// !tt.tensordesc<tensor<SHAPExELEMENT>> — result of tt.make_tensor_descriptor.
// `shape` is a `rank`-element array of int64_t block dims; `element_type` is
// the scalar element type; `shared_layout` may be a null MlirAttribute for none.
MlirType mlirTritonTensorDescTypeGet(intptr_t rank, const int64_t* shape,
                                     MlirType element_type,
                                     MlirAttribute shared_layout);
bool mlirTritonTypeIsATensorDesc(MlirType t);

// I32EnumAttr getters
MlirAttribute mlirTritonProgramDimGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonCacheModifierGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonEvictionPolicyGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonPaddingOptionGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonRoundingModeGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonInputPrecisionGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonRMWOpGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonMemSemanticGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonMemSyncScopeGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonPropagateNanGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonScaleDotElemTypeGet(MlirContext ctx, int32_t value);
MlirAttribute mlirTritonDescriptorReduceKindGet(MlirContext ctx, int32_t value);

#ifdef __cplusplus
}
#endif

#endif // ZML_MLIR_DIALECTS_TTIR_TTIR_CAPI_H
