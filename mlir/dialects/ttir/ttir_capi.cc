#include "ttir_capi.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

// Note: the second arg `tt` controls the generated symbol name
// `mlirGetDialectHandle__tt__`, which `mlir.DialectHandle.fromString("tt")` on
// the Zig side resolves at comptime (see mlir/mlir.zig:696-698). Must match the
// dialect's declared name (`let name = "tt"` in TritonDialect.td).
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, tt, mlir::triton::TritonDialect)

using mlir::cast;
using mlir::isa;

extern "C" {

MlirType mlirTritonPointerTypeGet(MlirType pointee, int32_t address_space) {
  return wrap(mlir::triton::PointerType::get(unwrap(pointee), address_space));
}

MlirType mlirTritonPointerTypeGetPointee(MlirType ptr) {
  return wrap(cast<mlir::triton::PointerType>(unwrap(ptr)).getPointeeType());
}

int32_t mlirTritonPointerTypeGetAddressSpace(MlirType ptr) {
  return cast<mlir::triton::PointerType>(unwrap(ptr)).getAddressSpace();
}

bool mlirTritonTypeIsAPointer(MlirType t) {
  return isa<mlir::triton::PointerType>(unwrap(t));
}

MlirType mlirTritonTensorDescTypeGet(intptr_t rank, const int64_t* shape,
                                     MlirType element_type,
                                     MlirAttribute shared_layout) {
  mlir::Attribute layout_attr;
  if (!mlirAttributeIsNull(shared_layout)) {
    layout_attr = unwrap(shared_layout);
  }
  // TensorDescType::get is a TypeBuilderWithInferredContext: the context is
  // inferred from elementType. Signature: (shape, elementType, sharedLayout).
  return wrap(mlir::triton::TensorDescType::get(
      llvm::ArrayRef<int64_t>(shape, rank),
      unwrap(element_type), layout_attr));
}

bool mlirTritonTypeIsATensorDesc(MlirType t) {
  return isa<mlir::triton::TensorDescType>(unwrap(t));
}

MlirAttribute mlirTritonProgramDimGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::ProgramIDDimAttr::get(
      unwrap(ctx), static_cast<mlir::triton::ProgramIDDim>(value)));
}

MlirAttribute mlirTritonCacheModifierGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::CacheModifierAttr::get(
      unwrap(ctx), static_cast<mlir::triton::CacheModifier>(value)));
}

MlirAttribute mlirTritonEvictionPolicyGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::EvictionPolicyAttr::get(
      unwrap(ctx), static_cast<mlir::triton::EvictionPolicy>(value)));
}

MlirAttribute mlirTritonPaddingOptionGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::PaddingOptionAttr::get(
      unwrap(ctx), static_cast<mlir::triton::PaddingOption>(value)));
}

MlirAttribute mlirTritonRoundingModeGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::RoundingModeAttr::get(
      unwrap(ctx), static_cast<mlir::triton::RoundingMode>(value)));
}

MlirAttribute mlirTritonInputPrecisionGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::InputPrecisionAttr::get(
      unwrap(ctx), static_cast<mlir::triton::InputPrecision>(value)));
}

MlirAttribute mlirTritonRMWOpGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::RMWOpAttr::get(
      unwrap(ctx), static_cast<mlir::triton::RMWOp>(value)));
}

MlirAttribute mlirTritonMemSemanticGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::MemSemanticAttr::get(
      unwrap(ctx), static_cast<mlir::triton::MemSemantic>(value)));
}

MlirAttribute mlirTritonMemSyncScopeGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::MemSyncScopeAttr::get(
      unwrap(ctx), static_cast<mlir::triton::MemSyncScope>(value)));
}

MlirAttribute mlirTritonPropagateNanGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::PropagateNanAttr::get(
      unwrap(ctx), static_cast<mlir::triton::PropagateNan>(value)));
}

MlirAttribute mlirTritonScaleDotElemTypeGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::ScaleDotElemTypeAttr::get(
      unwrap(ctx), static_cast<mlir::triton::ScaleDotElemType>(value)));
}

MlirAttribute mlirTritonDescriptorReduceKindGet(MlirContext ctx, int32_t value) {
  return wrap(mlir::triton::DescriptorReduceKindAttr::get(
      unwrap(ctx), static_cast<mlir::triton::DescriptorReduceKind>(value)));
}

} // extern "C"
