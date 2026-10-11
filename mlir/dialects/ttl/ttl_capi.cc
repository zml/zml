#include "ttl_capi.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

using namespace mlir::tt;

MlirType mlirTTLCircularBufferTypeGet(MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType element_type,
                                      int64_t block_count) {
  return wrap(ttl::CircularBufferType::get(unwrap(ctx), llvm::ArrayRef<int64_t>(shape, rank), unwrap(element_type),
                                           block_count));
}

bool mlirTypeIsATTLCircularBuffer(MlirType type) { return mlir::isa<ttl::CircularBufferType>(unwrap(type)); }

intptr_t mlirTTLCircularBufferTypeGetRank(MlirType type) {
  return mlir::cast<ttl::CircularBufferType>(unwrap(type)).getShape().size();
}

int64_t mlirTTLCircularBufferTypeGetDim(MlirType type, intptr_t pos) {
  return mlir::cast<ttl::CircularBufferType>(unwrap(type)).getShape()[pos];
}

MlirType mlirTTLCircularBufferTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<ttl::CircularBufferType>(unwrap(type)).getElementType());
}

int64_t mlirTTLCircularBufferTypeGetBlockCount(MlirType type) {
  return mlir::cast<ttl::CircularBufferType>(unwrap(type)).getBlockCount();
}

MlirType mlirTTLTransferHandleTypeGet(MlirContext ctx, int32_t kind) {
  std::optional<ttl::TransferKind> k;
  if (kind >= 0) k = static_cast<ttl::TransferKind>(kind);
  return wrap(ttl::TransferHandleType::get(unwrap(ctx), k));
}

MlirAttribute mlirTTLLayoutAttrGet(MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType element_type,
                                   uint32_t buffer_type, intptr_t grid_rank, const int64_t *grid,
                                   uint32_t memory_layout) {
  return wrap(ttl::LayoutAttr::get(unwrap(ctx), llvm::ArrayRef<int64_t>(shape, rank), unwrap(element_type),
                                   static_cast<ttl::BufferType>(buffer_type),
                                   llvm::ArrayRef<int64_t>(grid, grid_rank),
                                   static_cast<ttl::TensorMemoryLayout>(memory_layout)));
}

MlirLogicalResult mlirTTLRunPassPipeline(MlirModule module, MlirStringRef pipeline) {
  mlir::ModuleOp op = unwrap(module);
  // The pipeline names its anchor (`builtin.module(...)`): parsing it into a
  // module pass manager would nest it one level down.
  mlir::FailureOr<mlir::OpPassManager> parsed = mlir::parsePassPipeline(unwrap(pipeline), llvm::errs());
  if (mlir::failed(parsed)) return mlirLogicalResultFailure();
  mlir::PassManager pm(op->getContext(), parsed->getOpAnchorName());
  static_cast<mlir::OpPassManager &>(pm) = std::move(*parsed);
  return wrap(pm.run(op));
}

namespace {

llvm::json::Value i32Array(mlir::Operation *op, llvm::StringRef name) {
  llvm::json::Array out;
  if (auto array = op->getAttrOfType<mlir::ArrayAttr>(name)) {
    for (mlir::Attribute a : array)
      if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(a)) out.push_back(i.getInt());
  } else if (auto dense = op->getAttrOfType<mlir::DenseI32ArrayAttr>(name)) {
    for (int32_t v : dense.asArrayRef()) out.push_back(v);
  }
  return out;
}

llvm::json::Value optionalInt(mlir::Operation *op, llvm::StringRef name) {
  if (auto i = op->getAttrOfType<mlir::IntegerAttr>(name)) return i.getInt();
  return nullptr;
}

llvm::json::Value optionalBool(mlir::Operation *op, llvm::StringRef name) {
  if (auto b = op->getAttrOfType<mlir::BoolAttr>(name)) return b.getValue();
  return nullptr;
}

int64_t dictInt(mlir::DictionaryAttr dict, llvm::StringRef name) {
  if (auto i = mlir::dyn_cast_or_null<mlir::IntegerAttr>(dict.get(name))) return i.getInt();
  return -1;
}

}  // namespace

MlirLogicalResult mlirTTLDescribeKernels(MlirModule module, MlirStringCallback callback, void *user_data) {
  mlir::ModuleOp op = unwrap(module);
  llvm::json::Object root;

  llvm::json::Array dfbs;
  if (auto allocations = op->getAttrOfType<mlir::ArrayAttr>("ttl.dfb_allocations")) {
    for (mlir::Attribute a : allocations) {
      auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(a);
      if (!dict) return mlirLogicalResultFailure();
      llvm::json::Object dfb;
      dfb["dfb_index"] = dictInt(dict, "dfb_index");
      dfb["storage_index"] = dictInt(dict, "storage_index");
      dfb["num_tiles"] = dictInt(dict, "num_tiles");
      dfb["block_count"] = dictInt(dict, "block_count");
      dfb["page_size"] = dictInt(dict, "page_size");
      auto element = mlir::dyn_cast_or_null<mlir::TypeAttr>(dict.get("element_type"));
      auto tile = element ? mlir::dyn_cast<ttcore::TileType>(element.getValue()) : nullptr;
      if (!tile) return mlirLogicalResultFailure();
      dfb["data_type"] = static_cast<int64_t>(tile.getDataType());
      dfb["tile_height"] = tile.getHeight();
      dfb["tile_width"] = tile.getWidth();
      dfb["storage_segments"] = dict.get("storage_segments") != nullptr;
      dfb["remote"] = dict.get("address_scope") != nullptr;
      dfbs.push_back(std::move(dfb));
    }
  }
  root["dfbs"] = std::move(dfbs);
  root["pipe_sync_semaphores"] = optionalInt(op, "ttl.pipe_sync_semaphore_count");
  root["pipe_global_semaphores"] = optionalInt(op, "ttl.pipe_global_semaphore_count");
  root["pipe_sram_scratch_bytes"] = optionalInt(op, "ttl.pipe_sram_scratch_bytes");
  root["dfb_resets"] = optionalInt(op, "ttl.dfb_reset_count");
  root["dfb_reconfiguration"] = op->hasAttr("ttl.dfb_reconfiguration_plan");

  llvm::json::Array kernels;
  for (auto func : op.getOps<mlir::func::FuncOp>()) {
    auto thread = func->getAttrOfType<ttkernel::ThreadTypeAttr>("ttkernel.thread");
    if (!thread) continue;
    llvm::json::Object kernel;
    kernel["name"] = func.getSymName().str();
    kernel["thread"] = thread.getValue() == ttkernel::ThreadType::Compute ? "compute" : "noc";
    kernel["noc_index"] = optionalInt(func, "ttl.noc_index");
    kernel["base_cta_index"] = optionalInt(func, "ttl.base_cta_index");
    kernel["crta_indices"] = i32Array(func, "ttl.crta_indices");
    kernel["fp32_dest_acc_en"] = optionalBool(func, "fp32_dest_acc_en");
    kernel["dst_full_sync_en"] = optionalBool(func, "dst_full_sync_en");
    kernel["unpack_to_dest_fp32"] = i32Array(func, "ttl.unpack_to_dest_fp32");
    kernel["pipe_computed_address_dfb_indices"] = i32Array(func, "ttl.pipe_computed_address_dfb_indices");
    kernel["specialized"] = func->hasAttr("ttl.core_coord");
    kernel["local_tensors"] = func->hasAttr("ttl.local_tensor_indices");
    kernel["fabric"] = func->hasAttr("ttl.fabric_runtime_arg_base_common_index");
    kernels.push_back(std::move(kernel));
  }
  root["kernels"] = std::move(kernels);

  mlir::detail::CallbackOstream stream(callback, user_data);
  stream << llvm::json::Value(std::move(root));
  stream.flush();
  return mlirLogicalResultSuccess();
}
