//===- NVVMInference.cpp - result types the compiler infers ------------===//
//
// Generated from the nvidia-cutlass-dsl 4.8.0 compiler (its inference
// probed by varying operand types); do not edit. An operation without a
// recovered rule fails inference, so the result types written stand.

#include "cute_ir/Dialect/NVVM/IR/NVVMDialect.h"

using namespace mlir;
using namespace ::mlir::cutlass_compiler::nvvm;

LogicalResult AddFOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(0);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult AddPackedF32x2Op::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult AtomicRMWOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult BarrierOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult ClusterLaunchControlQueryCancelOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult CompareAndSetOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(0);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult FAbsOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(0);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult FMAPackedF32x2Op::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult FmaOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult InternalMmaBlockScaleOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(8);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult InternalMmaSparseBlockScaleOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(8);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult LdMatrixOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MBarrierArriveDropExpectTxOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MBarrierArriveDropOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MBarrierArriveExpectTxOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MBarrierArriveOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MBarrierTestWaitOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MBarrierTryWaitOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MatchSyncOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult MulPackedF32x2Op::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult ReduxOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult RsqrtOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(0);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult ShflOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult SubFOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  Adaptor adaptor(operands, attrs, props, regions);
  auto group = adaptor.getODSOperands(0);
  if (group.empty()) return failure();
  out.push_back(group.front().getType());
  return success();
}

LogicalResult SubPackedF32x2Op::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}

LogicalResult VoteSyncOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange regions,
    SmallVectorImpl<Type> &out) {
  return failure();
}
