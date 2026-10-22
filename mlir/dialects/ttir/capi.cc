//===- capi.cc - C API for the Triton tt dialect --------------------------===//

#include "capi.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

// Note: the second arg `tt` controls the generated symbol name
// `mlirGetDialectHandle__tt__`, which `mlir.DialectHandle.fromString("tt")` on
// the Zig side resolves at comptime (see `DialectHandle.fromString` in
// mlir/mlir.zig). Must match the dialect's declared name (`let name = "tt"` in
// TritonDialect.td).
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, tt, mlir::triton::TritonDialect)

using mlir::cast;
using mlir::isa;

MlirType mlirTritonPointerTypeGet(MlirType pointee, int32_t addressSpace) {
  return wrap(mlir::triton::PointerType::get(
      unwrap(pointee), static_cast<mlir::triton::PtrAddrSpace>(addressSpace)));
}

MlirType mlirTritonPointerTypeGetPointee(MlirType type) {
  return wrap(cast<mlir::triton::PointerType>(unwrap(type)).getPointeeType());
}

int32_t mlirTritonPointerTypeGetAddressSpace(MlirType type) {
  return static_cast<int32_t>(
      cast<mlir::triton::PointerType>(unwrap(type)).getAddressSpace());
}

bool mlirTypeIsATritonPointer(MlirType type) {
  return isa<mlir::triton::PointerType>(unwrap(type));
}

MlirType mlirTritonTensorDescTypeGet(intptr_t rank, const int64_t *shape,
                                     MlirType elementType,
                                     MlirAttribute sharedLayout) {
  mlir::Attribute layoutAttr;
  if (!mlirAttributeIsNull(sharedLayout)) {
    layoutAttr = unwrap(sharedLayout);
  }
  // TensorDescType::get is a TypeBuilderWithInferredContext: the context is
  // inferred from elementType. Signature: (shape, elementType, sharedLayout).
  return wrap(mlir::triton::TensorDescType::get(
      llvm::ArrayRef<int64_t>(shape, rank), unwrap(elementType), layoutAttr));
}

bool mlirTypeIsATritonTensorDesc(MlirType type) {
  return isa<mlir::triton::TensorDescType>(unwrap(type));
}

MlirAttribute mlirTritonProgramIDDimAttrGet(MlirContext context,
                                            int32_t value) {
  return wrap(mlir::triton::ProgramIDDimAttr::get(
      unwrap(context), static_cast<mlir::triton::ProgramIDDim>(value)));
}

MlirAttribute mlirTritonCacheModifierAttrGet(MlirContext context,
                                             int32_t value) {
  return wrap(mlir::triton::CacheModifierAttr::get(
      unwrap(context), static_cast<mlir::triton::CacheModifier>(value)));
}

MlirAttribute mlirTritonEvictionPolicyAttrGet(MlirContext context,
                                              int32_t value) {
  return wrap(mlir::triton::EvictionPolicyAttr::get(
      unwrap(context), static_cast<mlir::triton::EvictionPolicy>(value)));
}

MlirAttribute mlirTritonPaddingOptionAttrGet(MlirContext context,
                                             int32_t value) {
  return wrap(mlir::triton::PaddingOptionAttr::get(
      unwrap(context), static_cast<mlir::triton::PaddingOption>(value)));
}

MlirAttribute mlirTritonRoundingModeAttrGet(MlirContext context,
                                            int32_t value) {
  return wrap(mlir::triton::RoundingModeAttr::get(
      unwrap(context), static_cast<mlir::triton::RoundingMode>(value)));
}

MlirAttribute mlirTritonInputPrecisionAttrGet(MlirContext context,
                                              int32_t value) {
  return wrap(mlir::triton::InputPrecisionAttr::get(
      unwrap(context), static_cast<mlir::triton::InputPrecision>(value)));
}

MlirAttribute mlirTritonRMWOpAttrGet(MlirContext context, int32_t value) {
  return wrap(mlir::triton::RMWOpAttr::get(
      unwrap(context), static_cast<mlir::triton::RMWOp>(value)));
}

MlirAttribute mlirTritonMemSemanticAttrGet(MlirContext context, int32_t value) {
  return wrap(mlir::triton::MemSemanticAttr::get(
      unwrap(context), static_cast<mlir::triton::MemSemantic>(value)));
}

MlirAttribute mlirTritonMemSyncScopeAttrGet(MlirContext context,
                                            int32_t value) {
  return wrap(mlir::triton::MemSyncScopeAttr::get(
      unwrap(context), static_cast<mlir::triton::MemSyncScope>(value)));
}

MlirAttribute mlirTritonPropagateNanAttrGet(MlirContext context,
                                            int32_t value) {
  return wrap(mlir::triton::PropagateNanAttr::get(
      unwrap(context), static_cast<mlir::triton::PropagateNan>(value)));
}

MlirAttribute mlirTritonScaleDotElemTypeAttrGet(MlirContext context,
                                                int32_t value) {
  return wrap(mlir::triton::ScaleDotElemTypeAttr::get(
      unwrap(context), static_cast<mlir::triton::ScaleDotElemType>(value)));
}

MlirAttribute mlirTritonDescriptorReduceKindAttrGet(MlirContext context,
                                                    int32_t value) {
  return wrap(mlir::triton::DescriptorReduceKindAttr::get(
      unwrap(context), static_cast<mlir::triton::DescriptorReduceKind>(value)));
}
