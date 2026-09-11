#include "cute_ir/Dialect/Cute/IR/CuteDialectPrivate.h"

#include "mlir/IR/DialectImplementation.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::cute;

//===----------------------------------------------------------------------===//
// CuTe-DSL runtime and tensor-view types
//===----------------------------------------------------------------------===//

static IntTupleType parseIntTuplePayload(AsmParser &parser) {
  std::string text;
  if (failed(parser.parseString(&text)))
    return {};
  auto value = cutegen::from_string<cutegen::int_tuple>(text);
  if (!value) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse arithmetic tuple from \"")
        << text << '"';
    return {};
  }
  return IntTupleType::get(parser.getContext(), std::move(*value));
}

static Type parseLayoutPayload(AsmParser &parser) {
  std::string text;
  if (failed(parser.parseString(&text)))
    return {};
  if (auto value = cutegen::from_string<cutegen::layout>(text))
    return LayoutType::get(parser.getContext(), std::move(*value));
  if (auto value = cutegen::from_string<cutegen::composed_layout>(text))
    return ComposedLayoutType::get(parser.getContext(), std::move(*value));
  parser.emitError(parser.getCurrentLocation(), "failed to parse layout from \"")
      << text << '"';
  return {};
}

static void printLayoutPayload(AsmPrinter &printer, Type layout) {
  if (auto plain = llvm::dyn_cast<LayoutType>(layout)) {
    printer.printString(cutegen::to_string(plain.getRef()));
    return;
  }
  auto composed = llvm::cast<ComposedLayoutType>(layout);
  printer.printString(cutegen::to_string(composed.getRef()));
}

Type ArithTupleIteratorType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};
  IntTupleType tuple = parseIntTuplePayload(parser);
  if (!tuple || failed(parser.parseGreater()))
    return {};
  return get(parser.getContext(), tuple);
}

void ArithTupleIteratorType::print(AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getArithTuple().getRef()));
  printer << '>';
}

template <typename ConstrainedType>
static Type parseConstrainedInt(AsmParser &parser) {
  uint64_t divisibleBy = 1;
  if (succeeded(parser.parseOptionalLess())) {
    if (failed(parser.parseKeyword("divby")) ||
        failed(parser.parseInteger(divisibleBy)) ||
        failed(parser.parseGreater()))
      return {};
  }
  return ConstrainedType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), divisibleBy);
}

template <typename ConstrainedType>
static void printConstrainedInt(ConstrainedType type, AsmPrinter &printer) {
  if (type.getDivisibleBy() != 1)
    printer << "<divby " << type.getDivisibleBy() << '>';
}

Type ConstrainedInt32Type::parse(AsmParser &parser) {
  return parseConstrainedInt<ConstrainedInt32Type>(parser);
}

void ConstrainedInt32Type::print(AsmPrinter &printer) const {
  printConstrainedInt(*this, printer);
}

LogicalResult ConstrainedInt32Type::verify(
    function_ref<InFlightDiagnostic()> emitError, uint64_t divisibleBy) {
  return divisibleBy > 0 ? success()
                         : emitError() << "expects divisible_by > 0";
}

Type ConstrainedInt64Type::parse(AsmParser &parser) {
  return parseConstrainedInt<ConstrainedInt64Type>(parser);
}

void ConstrainedInt64Type::print(AsmPrinter &printer) const {
  printConstrainedInt(*this, printer);
}

LogicalResult ConstrainedInt64Type::verify(
    function_ref<InFlightDiagnostic()> emitError, uint64_t divisibleBy) {
  return divisibleBy > 0 ? success()
                         : emitError() << "expects divisible_by > 0";
}

Type FastDivmodDivisorType::parse(AsmParser &parser) {
  unsigned width;
  if (failed(parser.parseLess()) || failed(parser.parseInteger(width)))
    return {};
  bool isPow2 = false;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseKeyword("is_pow_2")))
      return {};
    isPow2 = true;
  }
  if (failed(parser.parseGreater()))
    return {};
  return getChecked([&] { return parser.emitError(parser.getCurrentLocation()); },
                    parser.getContext(), width, isPow2);
}

void FastDivmodDivisorType::print(AsmPrinter &printer) const {
  printer << '<' << getWidth();
  if (getIsPow2())
    printer << ", is_pow_2";
  printer << '>';
}

LogicalResult FastDivmodDivisorType::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned width,
    bool /*isPow2*/) {
  return width == 32 || width == 64
             ? success()
             : emitError() << "expects width to be 32 or 64, got " << width;
}

LogicalResult SparseElemType::verify(
    function_ref<InFlightDiagnostic()> emitError, int numLogical,
    Type physicalType) {
  if (numLogical <= 0)
    return emitError() << "expects num_logical > 0";
  if (!physicalType)
    return emitError() << "expects a physical storage type";
  return success();
}

namespace {
struct ParsedPtrFields {
  Type valueType;
  StringAttr memorySpace;
  uint64_t alignment = 0;
  SwizzleAttr swizzle;
};
} // namespace

static FailureOr<SwizzleAttr> parseInlineSwizzle(AsmParser &parser) {
  int64_t bits, base, shift;
  if (failed(parser.parseLess()) || failed(parser.parseInteger(bits)) ||
      failed(parser.parseComma()) || failed(parser.parseInteger(base)) ||
      failed(parser.parseComma()) || failed(parser.parseInteger(shift)) ||
      failed(parser.parseGreater()))
    return failure();
  std::string text = "S<" + std::to_string(bits) + "," +
                     std::to_string(base) + "," + std::to_string(shift) + ">";
  auto value = cutegen::from_string<cutegen::swizzle>(text);
  if (!value) {
    parser.emitError(parser.getCurrentLocation(), "invalid pointer swizzle ")
        << text;
    return failure();
  }
  return SwizzleAttr::get(parser.getContext(), std::move(*value));
}

static FailureOr<ParsedPtrFields> parsePtrFields(AsmParser &parser,
                                                  bool expectLayout,
                                                  Type *layout) {
  ParsedPtrFields fields;
  OptionalParseResult valueResult = parser.parseOptionalType(fields.valueType);
  if (valueResult.has_value() && failed(*valueResult))
    return failure();
  if (valueResult.has_value() && failed(parser.parseComma()))
    return failure();

  StringRef memorySpace;
  if (failed(parser.parseKeyword(&memorySpace)))
    return failure();
  fields.memorySpace = StringAttr::get(parser.getContext(), memorySpace);

  while (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword("align"))) {
      if (failed(parser.parseLess()) ||
          failed(parser.parseInteger(fields.alignment)) ||
          failed(parser.parseGreater()))
        return failure();
      continue;
    }
    if (succeeded(parser.parseOptionalKeyword("S"))) {
      FailureOr<SwizzleAttr> swizzle = parseInlineSwizzle(parser);
      if (failed(swizzle))
        return failure();
      fields.swizzle = *swizzle;
      continue;
    }
    if (expectLayout && layout && !*layout) {
      *layout = parseLayoutPayload(parser);
      if (!*layout)
        return failure();
      continue;
    }
    parser.emitError(parser.getCurrentLocation(),
                     "expected align, swizzle, or layout");
    return failure();
  }
  return fields;
}

static void printPtrFields(AsmPrinter &printer, PtrType pointer) {
  if (pointer.getValueType()) {
    printer.printType(pointer.getValueType());
    printer << ", ";
  }
  printer << pointer.getMemorySpace().getValue();
  if (pointer.getAlignment() != 0)
    printer << ", align<" << pointer.getAlignment() << '>';
  if (pointer.getSwizzle())
    printer << ", " << cutegen::to_string(pointer.getSwizzle().getRef());
}

Type PtrType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};
  FailureOr<ParsedPtrFields> fields = parsePtrFields(parser, false, nullptr);
  if (failed(fields) || failed(parser.parseGreater()))
    return {};
  return getChecked([&] { return parser.emitError(parser.getCurrentLocation()); },
                    parser.getContext(), fields->valueType,
                    fields->memorySpace, fields->alignment, fields->swizzle);
}

void PtrType::print(AsmPrinter &printer) const {
  printer << '<';
  printPtrFields(printer, *this);
  printer << '>';
}

LogicalResult PtrType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Type /*valueType*/, StringAttr memorySpace,
                              uint64_t alignment, SwizzleAttr /*swizzle*/) {
  if (!memorySpace)
    return emitError() << "expects an address space";
  StringRef space = memorySpace.getValue();
  if (space != "generic" && space != "gmem" && space != "cmem" &&
      space != "smem" && space != "rmem" && space != "tmem" &&
      space != "dsmem")
    return emitError() << "invalid address space '" << space << "'";
  if (alignment != 0 && !llvm::isPowerOf2_64(alignment))
    return emitError() << "expects power-of-two alignment, got " << alignment;
  return success();
}

Type mlir::cutlass_compiler::cute::MemRefType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};
  Type layout;
  FailureOr<ParsedPtrFields> fields = parsePtrFields(parser, true, &layout);
  if (failed(fields) || !layout || failed(parser.parseGreater()))
    return {};
  PtrType pointer = PtrType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), fields->valueType, fields->memorySpace,
      fields->alignment, fields->swizzle);
  if (!pointer)
    return {};
  return getChecked([&] { return parser.emitError(parser.getCurrentLocation()); },
                    parser.getContext(), pointer, layout);
}

void mlir::cutlass_compiler::cute::MemRefType::print(
    AsmPrinter &printer) const {
  printer << '<';
  printPtrFields(printer, getPtr());
  printer << ", ";
  printLayoutPayload(printer, getLayout());
  printer << '>';
}

LogicalResult mlir::cutlass_compiler::cute::MemRefType::verify(
    function_ref<InFlightDiagnostic()> emitError, PtrType pointer,
    Type layout) {
  if (!pointer)
    return emitError() << "expects a CuTe pointer";
  if (!llvm::isa<LayoutType, ComposedLayoutType>(layout))
    return emitError() << "expects a CuTe layout or composed layout";
  return success();
}

Type CoordTensorType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return {};
  IntTupleType tuple = parseIntTuplePayload(parser);
  if (!tuple || failed(parser.parseComma()))
    return {};
  Type layout = parseLayoutPayload(parser);
  if (!layout || failed(parser.parseGreater()))
    return {};
  return getChecked([&] { return parser.emitError(parser.getCurrentLocation()); },
                    parser.getContext(), tuple, layout);
}

void CoordTensorType::print(AsmPrinter &printer) const {
  printer << '<';
  printer.printString(cutegen::to_string(getArithTuple().getRef()));
  printer << ", ";
  printLayoutPayload(printer, getLayout());
  printer << '>';
}

LogicalResult CoordTensorType::verify(
    function_ref<InFlightDiagnostic()> emitError, IntTupleType tuple,
    Type layout) {
  if (!tuple)
    return emitError() << "expects an arithmetic tuple";
  if (!llvm::isa<LayoutType, ComposedLayoutType>(layout))
    return emitError() << "expects a CuTe layout or composed layout";
  return success();
}
