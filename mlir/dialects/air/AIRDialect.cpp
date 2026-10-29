//===- AIRDialect.cpp - Metal AIR dialect ---------------------------------===//

#include "mlir/dialects/air/AIRDialect.h"

#include "mlir/dialects/air/AIRAttributes.h"
#include "mlir/dialects/air/AIROps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseSet.h"

#include <utility>

using namespace mlir;
using namespace mlir::metal::air;

#include "mlir/dialects/air/AIRDialect.cpp.inc"

void AIRDialect::initialize() {
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "mlir/dialects/air/AIROps.cpp.inc"
      >();
}

static bool isKernelArgumentAttribute(StringRef name) {
  return name == AIRDialect::getBuiltinAttrName() ||
         name == AIRDialect::getBufferAttrName() ||
         name == AIRDialect::getArgNameAttrName();
}

// Position and size builtins may be 2- or 3-element vectors; index builtins
// are scalars.
static bool isVectorBuiltin(Builtin builtin) {
  switch (builtin) {
  case Builtin::thread_position_in_grid:
  case Builtin::threadgroup_position_in_grid:
  case Builtin::thread_position_in_threadgroup:
  case Builtin::threads_per_threadgroup:
  case Builtin::threadgroups_per_grid:
  case Builtin::threads_per_grid:
    return true;
  default:
    return false;
  }
}

static LogicalResult verifyKernelArgument(FunctionOpInterface function,
                                          unsigned index) {
  Operation *op = function.getOperation();
  Attribute builtin =
      function.getArgAttr(index, AIRDialect::getBuiltinAttrName());
  Attribute buffer =
      function.getArgAttr(index, AIRDialect::getBufferAttrName());
  if (bool(builtin) == bool(buffer))
    return op->emitOpError()
           << "argument " << index
           << " must have exactly one of air.builtin or air.buffer";
  if (Attribute name =
          function.getArgAttr(index, AIRDialect::getArgNameAttrName()))
    if (auto string = dyn_cast<StringAttr>(name);
        !string || string.getValue().empty())
      return op->emitOpError("air.arg_name must be a nonempty string");
  Type type = function.getArgumentTypes()[index];
  if (builtin) {
    auto metadata = dyn_cast<BuiltinAttr>(builtin);
    if (!metadata)
      return op->emitOpError("air.builtin must be an #air.builtin");
    if (auto vector = dyn_cast<VectorType>(type)) {
      if (!isVectorBuiltin(metadata.getValue()) || vector.isScalable() ||
          vector.getRank() != 1 ||
          (vector.getNumElements() != 2 && vector.getNumElements() != 3))
        return op->emitOpError()
               << "builtin " << stringifyBuiltin(metadata.getValue())
               << " cannot be a " << type;
      type = vector.getElementType();
    }
    if (!type.isSignlessInteger(32) && !type.isSignlessInteger(16))
      return op->emitOpError("AIR builtins are uint or ushort scalars or "
                             "vectors");
    return success();
  }
  auto metadata = dyn_cast<BufferAttr>(buffer);
  if (!metadata)
    return op->emitOpError("air.buffer must be an #air.buffer");
  auto pointer = dyn_cast<LLVM::LLVMPointerType>(type);
  if (!pointer ||
      pointer.getAddressSpace() != unsigned(metadata.getAddressSpace()))
    return op->emitOpError()
           << "argument " << index
           << " must be an !llvm.ptr in the address space of its air.buffer";
  return success();
}

LogicalResult AIRDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  StringRef name = attr.getName().getValue();
  if (isKernelArgumentAttribute(name))
    return op->emitOpError() << name << " belongs on a kernel argument";
  if (name == AIRDialect::getFastMathEnableAttrName()) {
    if (!isa<ModuleOp>(op) || !isa<UnitAttr>(attr.getValue()))
      return op->emitOpError() << name << " is a unit attribute of a module";
    return success();
  }
  if (name != AIRDialect::getKernelAttrName())
    return op->emitOpError() << "unknown AIR attribute " << name;
  auto function = dyn_cast<FunctionOpInterface>(op);
  if (!function || !isa<UnitAttr>(attr.getValue()))
    return op->emitOpError(
        "air.kernel requires a function and a unit attribute");
  if (failed(op->getName().verifyInvariants(op)))
    return failure();
  if (function.isExternal())
    return op->emitOpError("AIR kernels need a body");
  if (function.getNumResults() != 0)
    return op->emitOpError("AIR kernels cannot return values");
  if (auto llvmFunction = dyn_cast<LLVM::LLVMFuncOp>(op))
    if (llvmFunction.isVarArg())
      return op->emitOpError("AIR kernels cannot be variadic");
  llvm::DenseSet<Builtin> builtins;
  llvm::DenseSet<std::pair<unsigned, unsigned>> bindings;
  for (unsigned i = 0; i < function.getNumArguments(); ++i) {
    if (failed(verifyKernelArgument(function, i)))
      return failure();
    if (auto builtin = function.getArgAttrOfType<BuiltinAttr>(
            i, AIRDialect::getBuiltinAttrName())) {
      if (!builtins.insert(builtin.getValue()).second)
        return op->emitOpError() << "duplicate AIR builtin "
                                 << stringifyBuiltin(builtin.getValue());
      continue;
    }
    auto buffer = function.getArgAttrOfType<BufferAttr>(
        i, AIRDialect::getBufferAttrName());
    // Device and constant buffers share bindings; threadgroup buffers do not.
    auto key = std::make_pair(
        unsigned(buffer.getAddressSpace() == AddressSpace::threadgroup),
        buffer.getLocationIndex());
    if (!bindings.insert(key).second)
      return op->emitOpError()
             << "duplicate AIR buffer binding " << buffer.getLocationIndex();
  }
  return success();
}

LogicalResult AIRDialect::verifyRegionArgAttribute(Operation *op,
                                                   unsigned regionIndex,
                                                   unsigned argIndex,
                                                   NamedAttribute attr) {
  StringRef name = attr.getName().getValue();
  if (!isKernelArgumentAttribute(name))
    return op->emitOpError() << name << " is not an AIR argument attribute";
  auto function = dyn_cast<FunctionOpInterface>(op);
  if (!function || regionIndex != 0 ||
      !op->getAttrOfType<UnitAttr>(AIRDialect::getKernelAttrName()))
    return op->emitOpError()
           << name << " requires an air.kernel function argument";
  return verifyKernelArgument(function, argIndex);
}

LogicalResult AIRDialect::verifyRegionResultAttribute(Operation *op, unsigned,
                                                      unsigned,
                                                      NamedAttribute attr) {
  return op->emitOpError() << attr.getName()
                           << " cannot annotate a function result";
}
