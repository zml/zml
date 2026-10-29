//===- AIRAttributes.cpp - Metal AIR attributes ---------------------------===//

#include "mlir/dialects/air/AIRAttributes.h"

#include "mlir/dialects/air/AIRDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::metal::air;

#include "mlir/dialects/air/AIREnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/air/AIRAttributes.cpp.inc"

void AIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/dialects/air/AIRAttributes.cpp.inc"
      >();
}

static bool isValidName(StringRef name) {
  return !name.empty() && !name.contains('\0');
}

// The size of a Metal scalar or vector, where 3-element vectors take the
// space of 4. A bool vector is an i8 vector in memory, as LLVM packs i1
// vectors into bits.
static std::optional<uint64_t> metalSize(Type type) {
  uint64_t elements = 1;
  if (auto vector = dyn_cast<VectorType>(type)) {
    if (vector.isScalable() || vector.getRank() != 1 ||
        vector.getElementType().isInteger(1))
      return std::nullopt;
    elements = vector.getNumElements();
    if (elements < 2 || elements > 4)
      return std::nullopt;
    if (elements == 3)
      elements = 4;
    type = vector.getElementType();
  }
  // bool, char, short, int, long, half, bfloat and float.
  if (!type.isInteger(1) && !type.isInteger(8) && !type.isInteger(16) &&
      !type.isInteger(32) && !type.isInteger(64) && !type.isF16() &&
      !type.isBF16() && !type.isF32())
    return std::nullopt;
  return elements * llvm::divideCeil(type.getIntOrFloatBitWidth(), 8);
}

LogicalResult
StructFieldAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                        uint64_t offset, uint64_t size, uint64_t count,
                        StringRef typeName, StringRef name) {
  if (!size)
    return emitError() << "expected a positive struct field size";
  if (!isValidName(typeName) || !isValidName(name))
    return emitError() << "expected nonempty field and type names";
  return success();
}

static LogicalResult
verifyStructFields(llvm::function_ref<InFlightDiagnostic()> emitError,
                   LLVM::LLVMStructType type, uint64_t typeSize,
                   ArrayRef<StructFieldAttr> fields) {
  if (type.isOpaque() || type.isPacked())
    return emitError() << "expected a sized, unpacked struct";
  ArrayRef<Type> body = type.getBody();
  if (body.size() != fields.size())
    return emitError() << "expected one field per struct element";
  uint64_t end = 0;
  for (auto [member, field] : llvm::zip_equal(body, fields)) {
    Type element = member;
    uint64_t count = 0;
    if (auto array = dyn_cast<LLVM::LLVMArrayType>(element)) {
      count = array.getNumElements();
      element = array.getElementType();
    }
    std::optional<uint64_t> size = metalSize(element);
    if (!size || *size != field.getSize() || count != field.getCount())
      return emitError() << "field " << field.getName()
                         << " does not match the struct element " << element;
    if (field.getOffset() < end || field.getOffset() % *size != 0)
      return emitError() << "field " << field.getName()
                         << " is misplaced at offset " << field.getOffset();
    end = field.getOffset() + *size * std::max<uint64_t>(count, 1);
  }
  if (end > typeSize)
    return emitError() << "struct fields overflow the buffer type size";
  return success();
}

LogicalResult BufferAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, unsigned locationIndex,
    AddressSpace addressSpace, Access access, Type elementType,
    StringRef typeName, uint64_t typeSize, uint64_t typeAlignment,
    uint64_t bufferSize, bool noalias, ArrayRef<StructFieldAttr> fields) {
  if (addressSpace == AddressSpace::thread)
    return emitError() << "buffers are not in thread memory";
  if (addressSpace == AddressSpace::constant && access != Access::read)
    return emitError() << "constant buffers are read-only";
  if (addressSpace == AddressSpace::threadgroup && access != Access::read_write)
    return emitError() << "threadgroup buffers are read_write";
  if (!elementType)
    return emitError() << "expected a buffer element type";
  if (!isValidName(typeName))
    return emitError() << "expected a nonempty Metal type name";
  if (!typeSize || !llvm::isPowerOf2_64(typeAlignment) ||
      typeSize % typeAlignment != 0)
    return emitError() << "expected a positive ABI size divisible by a "
                          "power-of-two alignment";
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(elementType)) {
    if (addressSpace != AddressSpace::constant)
      return emitError() << "struct buffers must be constant";
    if (failed(verifyStructFields(emitError, structType, typeSize, fields)))
      return failure();
  } else {
    std::optional<uint64_t> size = metalSize(elementType);
    if (!size)
      return emitError() << "expected a scalar, vector or struct element type";
    if (*size != typeSize)
      return emitError() << "ABI size " << typeSize << " does not match "
                         << elementType;
    if (!fields.empty())
      return emitError() << "only struct buffers have fields";
  }
  if (bufferSize && (addressSpace != AddressSpace::constant ||
                     bufferSize < typeSize || bufferSize % typeAlignment))
    return emitError() << "only constant buffers have a buffer size, which "
                          "must cover the element and respect its alignment";
  return success();
}
