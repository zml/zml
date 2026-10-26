//===- capi.cc - C API for the Metal AIR dialect --------------------------===//

#include "capi.h"

#include "mlir/dialects/air/AIRAttributes.h"
#include "mlir/dialects/air/AIRDialect.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;
using namespace mlir::metal::air;

// Registration

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MetalAIR, air, AIRDialect)

static MLIRContext *loadAIR(MlirContext context) {
  MLIRContext *ctx = unwrap(context);
  if (ctx)
    ctx->getOrLoadDialect<AIRDialect>();
  return ctx;
}

static MlirAttribute invalid(MLIRContext *ctx, const Twine &message) {
  emitError(UnknownLoc::get(ctx), message);
  return {nullptr};
}

// Attributes

bool mlirAttributeIsAMetalAIRBuiltin(MlirAttribute attr) {
  return isa_and_nonnull<BuiltinAttr>(unwrap(attr));
}

MlirAttribute mlirMetalAIRBuiltinAttrGet(MlirContext context, uint32_t value) {
  MLIRContext *ctx = loadAIR(context);
  if (!ctx)
    return {nullptr};
  std::optional<Builtin> builtin = symbolizeBuiltin(value);
  if (!builtin)
    return invalid(ctx, "unknown AIR builtin " + Twine(value));
  return wrap(BuiltinAttr::get(ctx, *builtin));
}

uint32_t mlirMetalAIRBuiltinAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(cast<BuiltinAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAMetalAIRMemFlags(MlirAttribute attr) {
  return isa_and_nonnull<MemFlagsAttr>(unwrap(attr));
}

MlirAttribute mlirMetalAIRMemFlagsAttrGet(MlirContext context, uint32_t value) {
  MLIRContext *ctx = loadAIR(context);
  if (!ctx)
    return {nullptr};
  std::optional<MemFlags> flags = symbolizeMemFlags(value);
  if (!flags)
    return invalid(ctx, "unknown AIR memory flags " + Twine(value));
  return wrap(MemFlagsAttr::get(ctx, *flags));
}

uint32_t mlirMetalAIRMemFlagsAttrGetValue(MlirAttribute attr) {
  return static_cast<uint32_t>(cast<MemFlagsAttr>(unwrap(attr)).getValue());
}

bool mlirAttributeIsAMetalAIRStructField(MlirAttribute attr) {
  return isa_and_nonnull<StructFieldAttr>(unwrap(attr));
}

MlirAttribute mlirMetalAIRStructFieldAttrGet(MlirContext context,
                                             uint64_t offset, uint64_t size,
                                             uint64_t count,
                                             MlirStringRef typeName,
                                             MlirStringRef name) {
  MLIRContext *ctx = loadAIR(context);
  if (!ctx)
    return {nullptr};
  return wrap(StructFieldAttr::getChecked(UnknownLoc::get(ctx), ctx, offset,
                                          size, count, unwrap(typeName),
                                          unwrap(name)));
}

uint64_t mlirMetalAIRStructFieldAttrGetOffset(MlirAttribute attr) {
  return cast<StructFieldAttr>(unwrap(attr)).getOffset();
}

uint64_t mlirMetalAIRStructFieldAttrGetSize(MlirAttribute attr) {
  return cast<StructFieldAttr>(unwrap(attr)).getSize();
}

uint64_t mlirMetalAIRStructFieldAttrGetCount(MlirAttribute attr) {
  return cast<StructFieldAttr>(unwrap(attr)).getCount();
}

MlirStringRef mlirMetalAIRStructFieldAttrGetTypeName(MlirAttribute attr) {
  return wrap(cast<StructFieldAttr>(unwrap(attr)).getTypeName());
}

MlirStringRef mlirMetalAIRStructFieldAttrGetName(MlirAttribute attr) {
  return wrap(cast<StructFieldAttr>(unwrap(attr)).getName());
}

bool mlirAttributeIsAMetalAIRBuffer(MlirAttribute attr) {
  return isa_and_nonnull<BufferAttr>(unwrap(attr));
}

MlirAttribute mlirMetalAIRBufferAttrGet(
    MlirContext context, uint32_t locationIndex, uint32_t addressSpace,
    uint32_t access, MlirType elementType, MlirStringRef typeName,
    uint64_t typeSize, uint64_t typeAlignment, uint64_t bufferSize,
    bool noalias, intptr_t numFields, MlirAttribute const *fields) {
  MLIRContext *ctx = loadAIR(context);
  if (!ctx)
    return {nullptr};
  std::optional<AddressSpace> space = symbolizeAddressSpace(addressSpace);
  std::optional<Access> mode = symbolizeAccess(access);
  if (!space || !mode)
    return invalid(ctx, "unknown AIR buffer address space or access");
  Type element = unwrap(elementType);
  if (!element || element.getContext() != ctx)
    return invalid(ctx,
                   "buffer element type must belong to the attribute context");
  if (numFields < 0 || (numFields > 0 && !fields))
    return invalid(ctx, "invalid AIR struct fields");
  SmallVector<StructFieldAttr> structFields;
  for (intptr_t i = 0; i < numFields; ++i) {
    auto field = dyn_cast_or_null<StructFieldAttr>(unwrap(fields[i]));
    if (!field || field.getContext() != ctx)
      return invalid(ctx, "AIR struct fields must be #air.struct_field of "
                          "the attribute context");
    structFields.push_back(field);
  }
  return wrap(BufferAttr::getChecked(
      UnknownLoc::get(ctx), ctx, locationIndex, *space, *mode, element,
      unwrap(typeName), typeSize, typeAlignment, bufferSize, noalias,
      ArrayRef<StructFieldAttr>(structFields)));
}

uint32_t mlirMetalAIRBufferAttrGetLocationIndex(MlirAttribute attr) {
  return cast<BufferAttr>(unwrap(attr)).getLocationIndex();
}

uint32_t mlirMetalAIRBufferAttrGetAddressSpace(MlirAttribute attr) {
  return static_cast<uint32_t>(
      cast<BufferAttr>(unwrap(attr)).getAddressSpace());
}

uint32_t mlirMetalAIRBufferAttrGetAccess(MlirAttribute attr) {
  return static_cast<uint32_t>(cast<BufferAttr>(unwrap(attr)).getAccess());
}

MlirType mlirMetalAIRBufferAttrGetElementType(MlirAttribute attr) {
  return wrap(cast<BufferAttr>(unwrap(attr)).getElementType());
}

MlirStringRef mlirMetalAIRBufferAttrGetTypeName(MlirAttribute attr) {
  return wrap(cast<BufferAttr>(unwrap(attr)).getTypeName());
}

uint64_t mlirMetalAIRBufferAttrGetTypeSize(MlirAttribute attr) {
  return cast<BufferAttr>(unwrap(attr)).getTypeSize();
}

uint64_t mlirMetalAIRBufferAttrGetTypeAlignment(MlirAttribute attr) {
  return cast<BufferAttr>(unwrap(attr)).getTypeAlignment();
}

uint64_t mlirMetalAIRBufferAttrGetBufferSize(MlirAttribute attr) {
  return cast<BufferAttr>(unwrap(attr)).getBufferSize();
}

bool mlirMetalAIRBufferAttrGetNoalias(MlirAttribute attr) {
  return cast<BufferAttr>(unwrap(attr)).getNoalias();
}

intptr_t mlirMetalAIRBufferAttrGetNumFields(MlirAttribute attr) {
  return cast<BufferAttr>(unwrap(attr)).getFields().size();
}

MlirAttribute mlirMetalAIRBufferAttrGetField(MlirAttribute attr, intptr_t pos) {
  return wrap(cast<BufferAttr>(unwrap(attr)).getFields()[pos]);
}
