//===- capi.h - C API for the Triton tt dialect -------------------*- C -*-===//

#ifndef MLIR_DIALECTS_TTIR_CAPI_H
#define MLIR_DIALECTS_TTIR_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, tt);

// !tt.ptr<T, addr_space>
MLIR_CAPI_EXPORTED MlirType mlirTritonPointerTypeGet(MlirType pointee,
                                                     int32_t addressSpace);
MLIR_CAPI_EXPORTED MlirType mlirTritonPointerTypeGetPointee(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirTritonPointerTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED bool mlirTypeIsATritonPointer(MlirType type);

// !tt.tensordesc<tensor<SHAPExELEMENT>> — result of tt.make_tensor_descriptor.
// `shape` is a `rank`-element array of int64_t block dims; `elementType` is
// the scalar element type; `sharedLayout` may be a null MlirAttribute for none.
MLIR_CAPI_EXPORTED MlirType
mlirTritonTensorDescTypeGet(intptr_t rank, const int64_t *shape,
                            MlirType elementType, MlirAttribute sharedLayout);
MLIR_CAPI_EXPORTED bool mlirTypeIsATritonTensorDesc(MlirType type);

// I32EnumAttr constructors
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonProgramIDDimAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonCacheModifierAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonEvictionPolicyAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonPaddingOptionAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonRoundingModeAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonInputPrecisionAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTritonRMWOpAttrGet(MlirContext context,
                                                        int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonMemSemanticAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonMemSyncScopeAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonPropagateNanAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonScaleDotElemTypeAttrGet(MlirContext context, int32_t value);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTritonDescriptorReduceKindAttrGet(MlirContext context, int32_t value);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_TTIR_CAPI_H
