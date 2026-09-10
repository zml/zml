//===- CuteNVGPUDialect.cpp - CuTe NVIDIA GPU dialect --------------------===//

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::cute_nvgpu;

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUTypes.cpp.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUOps.cpp.inc"

#define CUTE_NVGPU_PAYLOAD_TYPES(M)                                           \
  M(CopyAtomBulkCopyG2SType)                                                  \
  M(CopyAtomBulkCopyS2GType)                                                  \
  M(CopyAtomBulkCopyS2SType)                                                  \
  M(CopyAtomDsmemStoreType)                                                   \
  M(CopyAtomG2RType)                                                          \
  M(CopyAtomLdsmType)                                                         \
  M(CopyAtomNonExec2DGather4TmaLoadType)                                      \
  M(CopyAtomNonExecIm2ColTmaLoadType)                                         \
  M(CopyAtomNonExecIm2ColTmaStoreType)                                        \
  M(CopyAtomNonExecTiledTmaLoadType)                                          \
  M(CopyAtomNonExecTiledTmaReduceType)                                        \
  M(CopyAtomNonExecTiledTmaStoreType)                                         \
  M(CopyAtomR2GType)                                                          \
  M(CopyAtomR2SType)                                                          \
  M(CopyAtomS2RType)                                                          \
  M(CopyAtomSIMTAsyncCopyType)                                                \
  M(CopyAtomSIMTSyncCopyType)                                                 \
  M(CopyAtomSM100CopyS2TType)                                                 \
  M(CopyAtomSM100TmemLoadType)                                                \
  M(CopyAtomSM100TmemStoreType)                                               \
  M(CopyAtomSM10xTmemLoadRedType)                                             \
  M(CopyAtomStsmType)                                                         \
  M(MmaAtomSM100UMMABlockScaledType)                                          \
  M(MmaAtomSM100UMMAType)                                                     \
  M(MmaAtomSM120BlockScaledType)                                              \
  M(MmaAtomSM80Type)                                                          \
  M(MmaAtomSM89Type)                                                          \
  M(MmaAtomSM90Type)                                                          \
  M(SmemDescViewType)                                                         \
  M(TiledCopyType)                                                            \
  M(TiledMmaType)                                                             \
  M(UniversalFmaAtomType)                                                     \
  M(CopyAtomIm2ColTmaLoadType)                                                \
  M(CopyAtomIm2ColTmaStoreType)                                               \
  M(CopyAtomSIMTMultimemLdReduceType)                                         \
  M(CopyAtomSIMTMultimemRedType)                                              \
  M(CopyAtomSIMTMultimemStType)                                               \
  M(CopyAtomSM100S2TCopyV2Type)                                               \
  M(CopyAtomTmaLoadType)                                                      \
  M(CopyAtomTmaReduceType)                                                    \
  M(CopyAtomTmaStoreType)                                                     \
  M(MmaAtomSM100UMMABlockScaledSparseType)                                    \
  M(MmaAtomSM100UMMASparseType)                                               \
  M(MmaAtomSM80SparseType)                                                    \
  M(SmemDescCircularSM103Type)

#define DEFINE_PAYLOAD_TYPE(TYPE)                                             \
  Type TYPE::parse(AsmParser &parser) {                                       \
    parser.emitError(parser.getCurrentLocation(),                             \
                     "payload types are parsed by CuteNVGPUDialect");         \
    return {};                                                                \
  }                                                                           \
  void TYPE::print(AsmPrinter &printer) const {                               \
    printer << getPayload().getValue();                                       \
  }
CUTE_NVGPU_PAYLOAD_TYPES(DEFINE_PAYLOAD_TYPE)
#undef DEFINE_PAYLOAD_TYPE

template <typename TypeT>
static Type maybeParsePayloadType(MLIRContext *context, StringRef spec) {
  StringRef remainder = spec;
  if (!remainder.consume_front(TypeT::getMnemonic()))
    return {};
  if (!remainder.empty() && !remainder.starts_with("<"))
    return {};
  return TypeT::get(context, StringAttr::get(context, remainder));
}

Type CuteNVGPUDialect::parseType(DialectAsmParser &parser) const {
  StringRef spec = parser.getFullSymbolSpec();
  if (spec == SmemDescType::getMnemonic() ||
      spec == TmaDescriptorTiledType::getMnemonic() ||
      spec == TmaDescriptorIm2ColType::getMnemonic() ||
      spec == WorkIdResponseType::getMnemonic()) {
    StringRef mnemonic;
    Type type;
    OptionalParseResult result = generatedTypeParser(parser, &mnemonic, type);
    return result.has_value() && succeeded(*result) ? type : Type();
  }

#define TRY_PAYLOAD_TYPE(TYPE)                                                \
  if (Type type = maybeParsePayloadType<TYPE>(getContext(), spec))            \
    return type;
  CUTE_NVGPU_PAYLOAD_TYPES(TRY_PAYLOAD_TYPE)
#undef TRY_PAYLOAD_TYPE

  StringAttr dialectNamespace =
      StringAttr::get(getContext(), getDialectNamespace());
  return OpaqueType::get(dialectNamespace, spec);
}

void CuteNVGPUDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected non-CuteNVGPU type");
}

void CuteNVGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUOps.cpp.inc"
      >();

  // Preserve every not-yet-structured private type spelling exactly. MLIR's
  // default hook stores the complete symbol specification in OpaqueType.
  allowUnknownTypes();
}

#undef CUTE_NVGPU_PAYLOAD_TYPES
