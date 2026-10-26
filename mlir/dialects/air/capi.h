//===- capi.h - C API for the Metal AIR dialect -------------------*- C -*-===//

#ifndef MLIR_DIALECTS_AIR_CAPI_H
#define MLIR_DIALECTS_AIR_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registration

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(MetalAIR, air);

// Attributes. Invalid parameters emit a diagnostic and return a null handle;
// typed getters require a handle of their attribute kind.

MLIR_CAPI_EXPORTED bool mlirAttributeIsAMetalAIRBuiltin(MlirAttribute attr);
/// `value` is a `mlir::metal::air::Builtin`.
MLIR_CAPI_EXPORTED MlirAttribute mlirMetalAIRBuiltinAttrGet(MlirContext context,
                                                            uint32_t value);
MLIR_CAPI_EXPORTED uint32_t mlirMetalAIRBuiltinAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAMetalAIRMemFlags(MlirAttribute attr);
/// `value` is a `mlir::metal::air::MemFlags` bit set.
MLIR_CAPI_EXPORTED MlirAttribute
mlirMetalAIRMemFlagsAttrGet(MlirContext context, uint32_t value);
MLIR_CAPI_EXPORTED uint32_t
mlirMetalAIRMemFlagsAttrGetValue(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAMetalAIRStructField(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirMetalAIRStructFieldAttrGet(
    MlirContext context, uint64_t offset, uint64_t size, uint64_t count,
    MlirStringRef typeName, MlirStringRef name);
MLIR_CAPI_EXPORTED uint64_t
mlirMetalAIRStructFieldAttrGetOffset(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
mlirMetalAIRStructFieldAttrGetSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
mlirMetalAIRStructFieldAttrGetCount(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirMetalAIRStructFieldAttrGetTypeName(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirMetalAIRStructFieldAttrGetName(MlirAttribute attr);

MLIR_CAPI_EXPORTED bool mlirAttributeIsAMetalAIRBuffer(MlirAttribute attr);
/// `addressSpace` is a `mlir::metal::air::AddressSpace` and `access` a
/// `mlir::metal::air::Access`. Sizes are bytes; a zero `bufferSize` omits it.
/// `elementType` and `fields` must belong to `context`.
MLIR_CAPI_EXPORTED MlirAttribute mlirMetalAIRBufferAttrGet(
    MlirContext context, uint32_t locationIndex, uint32_t addressSpace,
    uint32_t access, MlirType elementType, MlirStringRef typeName,
    uint64_t typeSize, uint64_t typeAlignment, uint64_t bufferSize,
    bool noalias, intptr_t numFields, MlirAttribute const *fields);
MLIR_CAPI_EXPORTED uint32_t
mlirMetalAIRBufferAttrGetLocationIndex(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t
mlirMetalAIRBufferAttrGetAddressSpace(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint32_t mlirMetalAIRBufferAttrGetAccess(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirType
mlirMetalAIRBufferAttrGetElementType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirStringRef
mlirMetalAIRBufferAttrGetTypeName(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
mlirMetalAIRBufferAttrGetTypeSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
mlirMetalAIRBufferAttrGetTypeAlignment(MlirAttribute attr);
MLIR_CAPI_EXPORTED uint64_t
mlirMetalAIRBufferAttrGetBufferSize(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirMetalAIRBufferAttrGetNoalias(MlirAttribute attr);
MLIR_CAPI_EXPORTED intptr_t
mlirMetalAIRBufferAttrGetNumFields(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute
mlirMetalAIRBufferAttrGetField(MlirAttribute attr, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_AIR_CAPI_H
