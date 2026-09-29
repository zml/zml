//===- CuteNVGPUDialect.cpp - CuTe NVIDIA GPU dialect --------------------===//
//
// Parsers, printers and verifiers of the cute_nvgpu types that the declarative
// format cannot express. The syntax and the verifier messages are the DSL
// compiler's (see CuteNVGPUTypes.td).

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"

#include <optional>
#include <string>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::cute_nvgpu;
namespace cute = mlir::cutlass_compiler::cute;

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.cpp.inc"

#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUTypes.cpp.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUOps.cpp.inc"

FailureOr<bool> mlir::cutlass_compiler::cute_nvgpu::parseBool(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  StringRef kw;
  if (p.parseKeyword(&kw)) return failure();
  if (kw == "true" || kw == "false") return kw == "true";
  return p.emitError(loc, "expected `true` or `false`");
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

using EmitFn = llvm::function_ref<InFlightDiagnostic()>;

template <typename T>
ParseResult field(AsmParser &p, T &out) {
  FailureOr<T> v = FieldParser<T>::parse(p);
  if (failed(v)) return failure();
  out = *v;
  return success();
}
ParseResult field(AsmParser &p, Type &out) { return p.parseType(out); }
ParseResult field(AsmParser &p, int &out) { return p.parseInteger(out); }
ParseResult field(AsmParser &p, bool &out) {
  FailureOr<bool> v = parseBool(p);
  if (failed(v)) return failure();
  out = *v;
  return success();
}

// `key = value`.
template <typename T>
ParseResult keyField(AsmParser &p, StringRef key, T &out) {
  return failure(p.parseKeyword(key) || p.parseEqual() || field(p, out));
}
// `, key = value`.
template <typename T>
ParseResult nextKeyField(AsmParser &p, StringRef key, T &out) {
  return failure(p.parseComma() || keyField(p, key, out));
}
// An optional ` key = value` group.
template <typename T>
ParseResult optKeyField(AsmParser &p, StringRef key, T &out) {
  if (failed(p.parseOptionalKeyword(key))) return success();
  return failure(p.parseEqual() || field(p, out));
}
// `N word`, as in `128 b` or `32 DP`.
ParseResult countOf(AsmParser &p, StringRef word, int &out) {
  return failure(p.parseInteger(out) || p.parseKeyword(word));
}
// `(a, b)`.
template <typename T>
ParseResult pairOf(AsmParser &p, T &a, T &b) {
  return failure(p.parseLParen() || field(p, a) || p.parseComma() ||
                 field(p, b) || p.parseRParen());
}
// `elem_type = (A, B, C)`.
ParseResult elemTypes(AsmParser &p, Type &a, Type &b, Type &c) {
  return failure(p.parseKeyword("elem_type") || p.parseEqual() ||
                 p.parseLParen() || p.parseType(a) || p.parseComma() ||
                 p.parseType(b) || p.parseComma() || p.parseType(c) ||
                 p.parseRParen());
}
void printElemTypes(AsmPrinter &p, Type a, Type b, Type c) {
  p << "elem_type = (" << a << ", " << b << ", " << c << ")";
}
// `xR`.
ParseResult repeat(AsmParser &p, unsigned &out) {
  SMLoc loc = p.getCurrentLocation();
  StringRef kw;
  if (p.parseKeyword(&kw)) return failure();
  if (!kw.consume_front("x"))
    return p.emitError(loc, "failed to parse parameter 'num_rep' which is to start with `x`");
  if (kw.getAsInteger(10, out))
    return p.emitError(loc, "failed to parse parameter 'num_rep' which is to be a `unsigned`");
  return success();
}

template <typename E>
void printIfNot(AsmPrinter &p, StringRef key, E value, E dflt) {
  if (value != dflt) p << ' ' << key << " = " << stringifyEnum(value);
}
template <typename A>
void printAttrField(AsmPrinter &p, StringRef key, A attr) {
  if (!attr) return;
  p << ", " << key << " = ";
  p.printStrippedAttrOrType(attr);
}

// The width of an element: integers and floats, 64 for index and pointers,
// a sparse element's share of its physical storage.
int elementBits(Type t) {
  if (auto s = llvm::dyn_cast<cute::SparseElemType>(t))
    return s.getNumLogical() ? elementBits(s.getPhysicalType()) / s.getNumLogical() : 0;
  if (t.isTF32()) return 32;
  if (t.isIntOrFloat()) return t.getIntOrFloatBitWidth();
  if (llvm::isa<IndexType, cute::PtrType>(t)) return 64;
  return 0;
}
int orElementBits(int bits, Type t) { return bits ? bits : elementBits(t); }

// The integers of a flat tuple `(a,b,c)`.
std::optional<SmallVector<int64_t>> flatInts(StringRef text) {
  if (!text.consume_front("(") || !text.consume_back(")")) return std::nullopt;
  SmallVector<int64_t> out;
  for (StringRef part : llvm::split(text, ',')) {
    int64_t v;
    if (part.getAsInteger(10, v)) return std::nullopt;
    out.push_back(v);
  }
  return out;
}

// `MxNxK`.
ParseResult parseShapeMNK(AsmParser &p, cute::ShapeAttr &out) {
  SMLoc loc = p.getCurrentLocation();
  SmallVector<int64_t> dims;
  if (p.parseDimensionList(dims, /*allowDynamic=*/false, /*withTrailingX=*/false))
    return p.emitError(loc, "can not parse shape_MNK");
  auto shape = cutegen::from_string<cutegen::shape>(
      "(" + llvm::join(llvm::map_range(dims, [](int64_t d) { return std::to_string(d); }), ",") + ")");
  if (!shape) return p.emitError(loc, "can not parse shape_MNK");
  out = cute::ShapeAttr::get(p.getContext(), *shape);
  return success();
}
void printShapeMNK(AsmPrinter &p, cute::ShapeAttr shape) {
  std::string text = cutegen::to_string(shape.getRef());
  StringRef body(text);
  body.consume_front("(");
  body.consume_back(")");
  for (char c : body) p << (c == ',' ? 'x' : c);
}
LogicalResult verifyShapeMNK(EmitFn emitError, cute::ShapeAttr shape) {
  std::string text = cutegen::to_string(shape.getRef());
  auto dims = flatInts(text);
  if (!dims || dims->size() != 3)
    return emitError() << "expects flatten rank-3 tuple of shape, but got " << text;
  return success();
}

// The A fragment kind: `ss` (shared memory), `rs` (registers) or `ts`
// (tensor memory).
ParseResult parseFragKind(AsmParser &p, MmaFragKind other, MmaFragKind &out) {
  SMLoc loc = p.getCurrentLocation();
  StringRef kw;
  bool ts = other == MmaFragKind::tmem;
  if (p.parseKeyword("frag_kind") || p.parseEqual() || p.parseKeyword(&kw))
    return failure();
  if (kw == "ss") out = MmaFragKind::smem_desc;
  else if (kw == (ts ? "ts" : "rs")) out = other;
  else return p.emitError(loc, ts ? "expects `ts` or `ss`" : "expects `rs` or `ss`");
  return success();
}
StringRef fragKindText(MmaFragKind k) {
  switch (k) {
    case MmaFragKind::smem_desc: return "ss";
    case MmaFragKind::rmem: return "rs";
    case MmaFragKind::tmem: return "ts";
    default: return stringifyEnum(k);
  }
}

// `t` or `n`.
ParseResult parseTranspose(AsmParser &p, UnitAttr &out) {
  SMLoc loc = p.getCurrentLocation();
  StringRef kw;
  if (p.parseKeyword(&kw)) return failure();
  if (kw != "t" && kw != "n") return p.emitError(loc, "expects `t` or `n`");
  out = kw == "t" ? UnitAttr::get(p.getContext()) : UnitAttr();
  return success();
}

bool isPow2AtLeast8(int v) { return v >= 8 && llvm::isPowerOf2_32(v); }

}  // namespace

//===----------------------------------------------------------------------===//
// Descriptors
//===----------------------------------------------------------------------===//

// sm103/sm107.smem_desc_circular: static `(B,128,N):(128,1,128*B)`, N >= 2.
static LogicalResult verifyCircular(EmitFn emitError, cute::LayoutAttr layout) {
  std::string text = cutegen::to_string(layout.getRef());
  StringRef shapeText, strideText;
  std::tie(shapeText, strideText) = StringRef(text).split(':');
  if (StringRef(text).contains('?'))
    return emitError() << "block layout must be fully static but got " << text;
  auto shape = flatInts(shapeText), stride = flatInts(strideText);
  if (!shape || shape->size() != 3 || !stride || stride->size() != 3)
    return emitError() << "expects block layout to have 3 modes, but got " << text;
  if ((*shape)[1] != 128)
    return emitError() << "expects block layout to have 128B 1st mode, but got " << text;
  if ((*stride)[1] != 1 || (*stride)[0] != 128 || (*stride)[2] != 128 * (*shape)[0])
    return emitError() << "expects mode-1 major compact layout, but got " << text;
  if ((*shape)[2] < 2)
    return emitError() << "Circular SMEM descriptor requires at least 2 128B blocks since "
                          "MMA may span 2 buffers, but got " << text;
  return success();
}
LogicalResult SmemDescCircularSM103Type::verify(EmitFn emitError, cute::LayoutAttr layout) {
  return verifyCircular(emitError, layout);
}
LogicalResult SmemDescCircularSM107Type::verify(EmitFn emitError, cute::LayoutAttr layout) {
  return verifyCircular(emitError, layout);
}

Type SmemDescViewType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type desc;
  std::string text;
  if (p.parseLess() || p.parseType(desc) || p.parseComma()) return {};
  SMLoc layoutLoc = p.getCurrentLocation();
  if (p.parseString(&text) || p.parseGreater()) return {};
  auto layout = cutegen::from_string<cutegen::layout>(text);
  if (!layout) {
    p.emitError(layoutLoc, "failed to parse SmemDescViewType parameter 'layout' as `cutegen::layout`");
    return {};
  }
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), desc,
                    cute::LayoutAttr::get(p.getContext(), *layout));
}
void SmemDescViewType::print(AsmPrinter &p) const {
  p << '<' << getDesc() << ", ";
  p.printString(cutegen::to_string(getLayout().getRef()));
  p << '>';
}
LogicalResult SmemDescViewType::verify(EmitFn emitError, Type desc, cute::LayoutAttr) {
  if (!llvm::isa<SmemDescType, SmemDescCircularSM103Type, SmemDescCircularSM107Type,
                 SmemDescSM107Type>(desc))
    return emitError() << "expected SmemDescType, SmemDescCircularSM103Type, "
                          "SmemDescCircularSM107Type or SmemDescSM107Type";
  return success();
}

//===----------------------------------------------------------------------===//
// SIMT copy atoms
//===----------------------------------------------------------------------===//

// `<T[, N b[, src_space = S[, dst_space = D]]]>`; the compiler prints
// `<T, src_space = S>` for copy_bits 0, which is accepted too.
Type CopyAtomSIMTSyncCopyType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int bits = 0;
  auto src = cute::AddressSpace::generic, dst = src;
  if (p.parseLess() || p.parseType(t)) return {};
  // `src_space` / `dst_space` groups; one alone is what the compiler prints
  // when the other is generic (or copy_bits is 0), so it is read too.
  auto spaces = [&]() -> ParseResult {
    if (succeeded(p.parseOptionalKeyword("dst_space")))
      return failure(p.parseEqual() || field(p, dst));
    if (p.parseKeyword("src_space") || p.parseEqual() || field(p, src)) return failure();
    if (succeeded(p.parseOptionalComma())) return keyField(p, "dst_space", dst);
    return success();
  };
  if (succeeded(p.parseOptionalComma())) {
    OptionalParseResult n = p.parseOptionalInteger(bits);
    if (n.has_value() && (failed(*n) || p.parseKeyword("b"))) return {};
    if ((!n.has_value() || succeeded(p.parseOptionalComma())) && spaces()) return {};
  }
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, bits, src, dst);
}
void CopyAtomSIMTSyncCopyType::print(AsmPrinter &p) const {
  auto generic = cute::AddressSpace::generic;
  bool spaces = getSrcSpace() != generic || getDstSpace() != generic;
  p << '<' << getValType();
  if (getCopyBits() || spaces) p << ", " << getCopyBits() << " b";
  if (spaces) p << ", src_space = " << stringifyEnum(getSrcSpace());
  if (getDstSpace() != generic) p << ", dst_space = " << stringifyEnum(getDstSpace());
  p << '>';
}
LogicalResult CopyAtomSIMTSyncCopyType::verify(EmitFn emitError, Type, int bits,
                                               cute::AddressSpace, cute::AddressSpace) {
  if (bits != 0 && !isPow2AtLeast8(bits))
    return emitError() << "expected copy_bits to be 0 (auto) or a power of 2 (>= 8), but got "
                       << bits;
  return success();
}

// `<T, cache = C[, N b]>`.
Type CopyAtomSIMTAsyncCopyType::parse(AsmParser &p) {
  Type t;
  int bits = 0;
  LoadCacheMode cache;
  if (p.parseLess() || p.parseType(t) || nextKeyField(p, "cache", cache)) return {};
  if (succeeded(p.parseOptionalComma()) && countOf(p, "b", bits)) return {};
  if (p.parseGreater()) return {};
  return get(p.getContext(), t, cache, orElementBits(bits, t));
}
void CopyAtomSIMTAsyncCopyType::print(AsmPrinter &p) const {
  p << '<' << getValType() << ", cache = " << stringifyEnum(getCache()) << ", "
    << getCopyBits() << " b>";
}

// `<T[, N b]>`.
Type CopyAtomDsmemStoreType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int bits = 0;
  if (p.parseLess() || p.parseType(t)) return {};
  if (succeeded(p.parseOptionalComma()) && countOf(p, "b", bits)) return {};
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t,
                    orElementBits(bits, t));
}
void CopyAtomDsmemStoreType::print(AsmPrinter &p) const {
  p << '<' << getValType() << ", " << getCopyBits() << " b>";
}
LogicalResult CopyAtomDsmemStoreType::verify(EmitFn emitError, Type, int bits) {
  if (bits != 0 && bits != 32 && bits != 64 && bits != 128)
    return emitError() << "expects copy_bits to be one of (0, 32, 64, 128), but got " << bits;
  return success();
}

// Bulk copies: `<T[ copy_bits = N][ flag = B]>`.
Type CopyAtomBulkCopyG2SType::parse(AsmParser &p) {
  Type t;
  int bits = 0;
  bool mcast = false;
  if (p.parseLess() || p.parseType(t) || optKeyField(p, "copy_bits", bits) ||
      optKeyField(p, "mcast", mcast) || p.parseGreater())
    return {};
  return get(p.getContext(), t, bits, mcast);
}
void CopyAtomBulkCopyG2SType::print(AsmPrinter &p) const {
  p << '<' << getValType();
  if (getCopyBits()) p << " copy_bits = " << getCopyBits();
  if (getMcast()) p << " mcast = true";
  p << '>';
}
Type CopyAtomBulkCopyS2GType::parse(AsmParser &p) {
  Type t;
  int bits = 0;
  bool mask = false;
  if (p.parseLess() || p.parseType(t) || optKeyField(p, "copy_bits", bits) ||
      optKeyField(p, "mask", mask) || p.parseGreater())
    return {};
  return get(p.getContext(), t, bits, mask);
}
void CopyAtomBulkCopyS2GType::print(AsmPrinter &p) const {
  p << '<' << getValType();
  if (getCopyBits()) p << " copy_bits = " << getCopyBits();
  if (getMask()) p << " mask = true";
  p << '>';
}
Type CopyAtomBulkCopyS2SType::parse(AsmParser &p) {
  Type t;
  int bits = 0;
  if (p.parseLess() || p.parseType(t) || optKeyField(p, "copy_bits", bits) ||
      p.parseGreater())
    return {};
  return get(p.getContext(), t, bits);
}
void CopyAtomBulkCopyS2SType::print(AsmPrinter &p) const {
  p << '<' << getValType();
  if (getCopyBits()) p << " copy_bits = " << getCopyBits();
  p << '>';
}

// g2r, r2g, r2s, s2r: `<T copy_bits = N key = value ...>`.
static LogicalResult verifyCopyBits(EmitFn emitError, int bits) {
  if (!isPow2AtLeast8(bits))
    return emitError() << "expects copy_bits to be a power of 2 (>= 8), but got " << bits;
  return success();
}

Type CopyAtomG2RType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int bits = 0;
  auto order = MemOrderKind::WEAK;
  auto scope = MemScopeKind::CTA;
  auto prefetch = L2PrefetchSize::NONE;
  auto evict = CacheEvictionPriority::EVICT_NORMAL;
  auto cache = LoadCacheMode::always;
  auto shared = SharedSpace::CTA;
  bool invariant = false;
  if (p.parseLess() || p.parseType(t) || optKeyField(p, "copy_bits", bits) ||
      optKeyField(p, "mem_order", order) || optKeyField(p, "mem_scope", scope) ||
      optKeyField(p, "l2_prefetch_size", prefetch) ||
      optKeyField(p, "l1_cache_evict_priority", evict) ||
      optKeyField(p, "load_cache_mode", cache) || optKeyField(p, "shared_space", shared) ||
      optKeyField(p, "invariant", invariant) || p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t,
                    orElementBits(bits, t), order, scope, prefetch, evict, cache, shared,
                    invariant);
}
void CopyAtomG2RType::print(AsmPrinter &p) const {
  p << '<' << getValType() << " copy_bits = " << getCopyBits();
  printIfNot(p, "mem_order", getMemOrder(), MemOrderKind::WEAK);
  printIfNot(p, "mem_scope", getMemScope(), MemScopeKind::CTA);
  printIfNot(p, "l2_prefetch_size", getL2PrefetchSize(), L2PrefetchSize::NONE);
  printIfNot(p, "l1_cache_evict_priority", getL1CacheEvictPriority(),
             CacheEvictionPriority::EVICT_NORMAL);
  printIfNot(p, "load_cache_mode", getLoadCacheMode(), LoadCacheMode::always);
  printIfNot(p, "shared_space", getSharedSpace(), SharedSpace::CTA);
  if (getInvariant()) p << " invariant = true";
  p << '>';
}
LogicalResult CopyAtomG2RType::verify(EmitFn emitError, Type, int bits, MemOrderKind order,
                                      MemScopeKind scope, L2PrefetchSize prefetch,
                                      CacheEvictionPriority evict, LoadCacheMode cache,
                                      SharedSpace, bool invariant) {
  if (failed(verifyCopyBits(emitError, bits))) return failure();
  if (invariant && (order != MemOrderKind::WEAK || scope != MemScopeKind::CTA ||
                    prefetch != L2PrefetchSize::NONE ||
                    evict != CacheEvictionPriority::EVICT_NORMAL ||
                    cache != LoadCacheMode::always))
    return emitError() << "invariant_load is not allowed when the other non-default memory "
                          "features (e.g., mem_order) are set to non-default values";
  if (order != MemOrderKind::WEAK && cache != LoadCacheMode::always)
    return emitError() << "expects load_cache_mode to be 'always' when mem_order is not "
                          "'weak', but got '"
                       << stringifyEnum(cache) << "' and '" << stringifyEnum(order) << "'";
  return success();
}

Type CopyAtomR2GType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int bits = 0;
  auto order = MemOrderKind::WEAK;
  auto scope = MemScopeKind::CTA;
  auto evict = CacheEvictionPriority::EVICT_NORMAL;
  auto cache = StoreCacheMode::write_back;
  auto shared = SharedSpace::CTA;
  if (p.parseLess() || p.parseType(t) || optKeyField(p, "copy_bits", bits) ||
      optKeyField(p, "mem_order", order) || optKeyField(p, "mem_scope", scope) ||
      optKeyField(p, "l1_cache_evict_priority", evict) ||
      optKeyField(p, "store_cache_mode", cache) || optKeyField(p, "shared_space", shared) ||
      p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t,
                    orElementBits(bits, t), order, scope, evict, cache, shared);
}
void CopyAtomR2GType::print(AsmPrinter &p) const {
  p << '<' << getValType() << " copy_bits = " << getCopyBits();
  printIfNot(p, "mem_order", getMemOrder(), MemOrderKind::WEAK);
  printIfNot(p, "mem_scope", getMemScope(), MemScopeKind::CTA);
  printIfNot(p, "l1_cache_evict_priority", getL1CacheEvictPriority(),
             CacheEvictionPriority::EVICT_NORMAL);
  printIfNot(p, "store_cache_mode", getStoreCacheMode(), StoreCacheMode::write_back);
  printIfNot(p, "shared_space", getSharedSpace(), SharedSpace::CTA);
  p << '>';
}
LogicalResult CopyAtomR2GType::verify(EmitFn emitError, Type, int bits, MemOrderKind order,
                                      MemScopeKind, CacheEvictionPriority,
                                      StoreCacheMode cache, SharedSpace) {
  if (failed(verifyCopyBits(emitError, bits))) return failure();
  if (order != MemOrderKind::WEAK && cache != StoreCacheMode::write_back)
    return emitError() << "expects store_cache_mode to be 'write_back' when mem_order is "
                          "not 'weak', but got '"
                       << stringifyEnum(cache) << "' and '" << stringifyEnum(order) << "'";
  return success();
}

template <typename T>
static Type parseSharedCopy(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int bits = 0;
  auto order = MemOrderKind::WEAK;
  auto scope = MemScopeKind::CTA;
  auto shared = SharedSpace::CTA;
  if (p.parseLess() || p.parseType(t) || optKeyField(p, "copy_bits", bits) ||
      optKeyField(p, "mem_order", order) || optKeyField(p, "mem_scope", scope) ||
      optKeyField(p, "shared_space", shared) || p.parseGreater())
    return {};
  return T::getChecked([&] { return p.emitError(loc); }, p.getContext(), t,
                       orElementBits(bits, t), order, scope, shared);
}
template <typename T>
static void printSharedCopy(AsmPrinter &p, T type) {
  p << '<' << type.getValType() << " copy_bits = " << type.getCopyBits();
  printIfNot(p, "mem_order", type.getMemOrder(), MemOrderKind::WEAK);
  printIfNot(p, "mem_scope", type.getMemScope(), MemScopeKind::CTA);
  printIfNot(p, "shared_space", type.getSharedSpace(), SharedSpace::CTA);
  p << '>';
}
Type CopyAtomR2SType::parse(AsmParser &p) { return parseSharedCopy<CopyAtomR2SType>(p); }
void CopyAtomR2SType::print(AsmPrinter &p) const { printSharedCopy(p, *this); }
LogicalResult CopyAtomR2SType::verify(EmitFn emitError, Type, int bits, MemOrderKind,
                                      MemScopeKind, SharedSpace) {
  return verifyCopyBits(emitError, bits);
}
Type CopyAtomS2RType::parse(AsmParser &p) { return parseSharedCopy<CopyAtomS2RType>(p); }
void CopyAtomS2RType::print(AsmPrinter &p) const { printSharedCopy(p, *this); }
LogicalResult CopyAtomS2RType::verify(EmitFn emitError, Type, int bits, MemOrderKind,
                                      MemScopeKind, SharedSpace) {
  return verifyCopyBits(emitError, bits);
}

//===----------------------------------------------------------------------===//
// Multimem atoms: `<T, N b, RED, <ACC>, ORDER, SCOPE>`, positional. Parsing
// takes each field by its kind, which also reads what the compiler prints
// when it drops a default field before a set one.
//===----------------------------------------------------------------------===//

namespace {
struct Multimem {
  int bits = 0;
  ReductionKind red = ReductionKind::ADD;
  LdReduceAccPrecisionKindAttr acc;
  MemOrderKind order = MemOrderKind::RELAXED;
  MemScopeKind scope = MemScopeKind::SYS;
};

ParseResult parseMultimem(AsmParser &p, Type &t, Multimem &m, bool hasRed, bool hasAcc) {
  if (p.parseLess() || p.parseType(t)) return failure();
  while (succeeded(p.parseOptionalComma())) {
    SMLoc loc = p.getCurrentLocation();
    StringRef kw;
    OptionalParseResult bits = p.parseOptionalInteger(m.bits);
    if (bits.has_value()) {
      if (failed(*bits) || p.parseKeyword("b")) return failure();
      continue;
    }
    if (hasAcc && succeeded(p.parseOptionalLess())) {
      LdReduceAccPrecisionKind kind;
      if (field(p, kind) || p.parseGreater()) return failure();
      m.acc = LdReduceAccPrecisionKindAttr::get(p.getContext(), kind);
      continue;
    }
    if (p.parseKeyword(&kw)) return failure();
    if (auto r = symbolizeReductionKind(kw); r && hasRed) m.red = *r;
    else if (auto o = symbolizeMemOrderKind(kw)) m.order = *o;
    else if (auto s = symbolizeMemScopeKind(kw)) m.scope = *s;
    else return p.emitError(loc, "unexpected `") << kw << "`";
  }
  return p.parseGreater();
}

// Prints the fields up to the last one that is not default.
void printMultimem(AsmPrinter &p, Type t, const Multimem &m, bool bitsSet, bool hasRed,
                   bool hasAcc) {
  int last = m.scope != MemScopeKind::SYS          ? 4
             : m.order != MemOrderKind::RELAXED    ? 3
             : m.acc                               ? 2
             : hasRed && m.red != ReductionKind::ADD ? 1
             : bitsSet                             ? 0
                                                   : -1;
  p << '<' << t;
  if (last >= 0) p << ", " << m.bits << " b";
  if (hasRed && last >= 1) p << ", " << stringifyEnum(m.red);
  if (hasAcc && last >= 2 && m.acc) {
    p << ", ";
    p.printStrippedAttrOrType(m.acc);
  }
  if (last >= 3) p << ", " << stringifyEnum(m.order);
  if (last >= 4) p << ", " << stringifyEnum(m.scope);
  p << '>';
}
}  // namespace

Type CopyAtomSIMTMultimemStType::parse(AsmParser &p) {
  Type t;
  Multimem m;
  if (parseMultimem(p, t, m, /*hasRed=*/false, /*hasAcc=*/false)) return {};
  return get(p.getContext(), t, m.bits, m.order, m.scope);
}
void CopyAtomSIMTMultimemStType::print(AsmPrinter &p) const {
  Multimem m{getCopyBits(), ReductionKind::ADD, {}, getMemOrder(), getMemScope()};
  printMultimem(p, getValType(), m, getCopyBits() != 0, false, false);
}

Type CopyAtomSIMTMultimemRedType::parse(AsmParser &p) {
  Type t;
  Multimem m;
  if (parseMultimem(p, t, m, /*hasRed=*/true, /*hasAcc=*/false)) return {};
  // An unset width is the element's, at least 32 bits (the f16x2 form).
  int bits = m.bits ? m.bits : std::max(elementBits(t), 32);
  return get(p.getContext(), t, bits, m.red, m.order, m.scope);
}
void CopyAtomSIMTMultimemRedType::print(AsmPrinter &p) const {
  Multimem m{getCopyBits(), getReduction(), {}, getMemOrder(), getMemScope()};
  printMultimem(p, getValType(), m, /*bitsSet=*/true, true, false);
}

Type CopyAtomSIMTMultimemLdReduceType::parse(AsmParser &p) {
  Type t;
  Multimem m;
  if (parseMultimem(p, t, m, /*hasRed=*/true, /*hasAcc=*/true)) return {};
  return get(p.getContext(), t, m.bits, m.red, m.acc, m.order, m.scope);
}
void CopyAtomSIMTMultimemLdReduceType::print(AsmPrinter &p) const {
  Multimem m{getCopyBits(), getReduction(), getLdReduceAccPrecision(), getMemOrder(),
             getMemScope()};
  printMultimem(p, getValType(), m, getCopyBits() != 0, true, true);
}

//===----------------------------------------------------------------------===//
// Tensor-memory atoms: `<T, D DP, B bit, xR ...>`
//===----------------------------------------------------------------------===//

static ParseResult parseTmemHead(AsmParser &p, Type &t, int &dp, int &bit, unsigned &rep) {
  return failure(p.parseLess() || p.parseType(t) || p.parseComma() ||
                 countOf(p, "DP", dp) || p.parseComma() || countOf(p, "bit", bit) ||
                 p.parseComma() || repeat(p, rep));
}
static void printTmemHead(AsmPrinter &p, Type t, int dp, int bit, unsigned rep) {
  p << '<' << t << ", " << dp << " DP, " << bit << " bit, x" << rep;
}
// An optional `, word`.
static ParseResult parseFlag(AsmParser &p, StringRef word, UnitAttr &out) {
  if (failed(p.parseOptionalComma())) return success();
  if (p.parseKeyword(word)) return failure();
  out = UnitAttr::get(p.getContext());
  return success();
}

static LogicalResult verifyTmemCopy(EmitFn emitError, int dp, int bit, unsigned rep) {
  if (dp != 16 && dp != 32)
    return emitError() << "expects the number of data paths to be one of [16,32], but got " << dp;
  if (bit != 32 && bit != 64 && bit != 128 && bit != 256)
    return emitError() << "expects the number of bits to be one of [32,64,128,256], but got " << bit;
  if (rep < 1 || rep > 128 || !llvm::isPowerOf2_32(rep))
    return emitError() << "expects the number of repetitions to be one of "
                          "[1,2,4,8,16,32,64,128], but got " << rep;
  if (dp == 32 && bit != 32)
    return emitError() << "only 32b patterns are supported when the number of data paths "
                          "is 32, but got " << bit;
  if (dp == 16 && bit == 256 && rep >= 64)
    return emitError() << "a repetition of 64 or 128 is not supported when the number of "
                          "data paths is 16 and the number of bits is 256";
  if (dp == 16 && bit == 128 && rep == 128)
    return emitError() << "a repetition of 128 is not supported when the number of data "
                          "paths is 16 and the number of bits is 128";
  return success();
}

Type CopyAtomSM100TmemLoadType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int dp, bit;
  unsigned rep;
  UnitAttr pack;
  if (parseTmemHead(p, t, dp, bit, rep) || parseFlag(p, "pack16b", pack) || p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, dp, bit, rep, pack);
}
void CopyAtomSM100TmemLoadType::print(AsmPrinter &p) const {
  printTmemHead(p, getValType(), getNumDp(), getNumBit(), getNumRep());
  p << (getPack16b() ? ", pack16b>" : ">");
}
LogicalResult CopyAtomSM100TmemLoadType::verify(EmitFn emitError, Type, int dp, int bit,
                                                unsigned rep, UnitAttr) {
  return verifyTmemCopy(emitError, dp, bit, rep);
}

Type CopyAtomSM100TmemStoreType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int dp, bit;
  unsigned rep;
  UnitAttr expand;
  if (parseTmemHead(p, t, dp, bit, rep) || parseFlag(p, "expand16b", expand) ||
      p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, dp, bit, rep, expand);
}
void CopyAtomSM100TmemStoreType::print(AsmPrinter &p) const {
  printTmemHead(p, getValType(), getNumDp(), getNumBit(), getNumRep());
  p << (getExpand16b() ? ", expand16b>" : ">");
}
LogicalResult CopyAtomSM100TmemStoreType::verify(EmitFn emitError, Type, int dp, int bit,
                                                 unsigned rep, UnitAttr) {
  return verifyTmemCopy(emitError, dp, bit, rep);
}

// `<..., OP[, nan][, half_split_off = N[ : T]]>`.
Type CopyAtomSM10xTmemLoadRedType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int dp, bit;
  unsigned rep;
  TmemLoadRedOp op;
  UnitAttr nan;
  IntegerAttr split;
  if (parseTmemHead(p, t, dp, bit, rep) || p.parseComma() || field(p, op)) return {};
  while (succeeded(p.parseOptionalComma())) {
    if (!nan && !split && succeeded(p.parseOptionalKeyword("nan"))) {
      nan = UnitAttr::get(p.getContext());
      continue;
    }
    if (p.parseKeyword("half_split_off") || p.parseEqual() || p.parseAttribute(split))
      return {};
  }
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, dp, bit, rep, op, nan,
                    split);
}
void CopyAtomSM10xTmemLoadRedType::print(AsmPrinter &p) const {
  printTmemHead(p, getValType(), getNumDp(), getNumBit(), getNumRep());
  p << ", " << stringifyEnum(getRedOp());
  if (getNan()) p << ", nan";
  if (getHalfSplitOff()) p << ", half_split_off=" << getHalfSplitOff();
  p << '>';
}
LogicalResult CopyAtomSM10xTmemLoadRedType::verify(EmitFn emitError, Type t, int dp, int bit,
                                                   unsigned rep, TmemLoadRedOp op, UnitAttr nan,
                                                   IntegerAttr split) {
  if (!(t.isF32() || t.isInteger(32)))
    return emitError() << "expect val_type to be u32, s32, or f32, but got '" << t << "'";
  if (nan && !t.isF32())
    return emitError() << "expect val_type to be f32 when nan is specified";
  if ((op == TmemLoadRedOp::minabs || op == TmemLoadRedOp::maxabs) && !t.isF32())
    return emitError() << "expect val_type to be f32 when red_op is minabs or maxabs";
  if (bit != 32) return emitError() << "only 32b patterns are supported, but got " << bit;
  if (rep < 2 || rep > 128 || !llvm::isPowerOf2_32(rep))
    return emitError() << "expects the number of repetitions to be one of "
                          "[2,4,8,16,32,64,128], but got " << rep;
  if (split && dp != 16)
    return emitError() << "expect half_split_off to be set only when mode is 16dp 32bit";
  return success();
}

// `<..., OP[, red[, nan]]>`.
Type CopyAtomSM107TmemLoadSPCompressType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int dp, bit;
  unsigned rep;
  TmemLoadRedOp op;
  UnitAttr red, nan;
  if (parseTmemHead(p, t, dp, bit, rep) || p.parseComma() || field(p, op) ||
      parseFlag(p, "red", red) || (red && parseFlag(p, "nan", nan)) || p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, dp, bit, rep, op, red,
                    nan);
}
void CopyAtomSM107TmemLoadSPCompressType::print(AsmPrinter &p) const {
  printTmemHead(p, getValType(), getNumDp(), getNumBit(), getNumRep());
  p << ", " << stringifyEnum(getRedOp());
  if (getRed()) p << ", red";
  if (getNan()) p << ", nan";
  p << '>';
}
LogicalResult CopyAtomSM107TmemLoadSPCompressType::verify(EmitFn emitError, Type t, int dp,
                                                          int bit, unsigned rep, TmemLoadRedOp,
                                                          UnitAttr red, UnitAttr nan) {
  if (dp != 32 || bit != 32)
    return emitError() << "expect `num_dp` and `num_bit` to be 32, but got " << dp << " and "
                       << bit;
  if (!t.isF32()) return emitError() << "expect val_type to be f32, but got '" << t << "'";
  if (rep < 4 || rep > 128 || !llvm::isPowerOf2_32(rep))
    return emitError() << "expect the number of repetitions to be one of "
                          "[4,8,16,32,64,128], but got " << rep;
  if (nan && !red) return emitError() << "expect `red` to be true when `nan` is true";
  return success();
}

// The UTCCP shapes of tcgen05.cp.
static bool isUtccp(int dp, int bit, CopyS2TBroadcast b) {
  using B = CopyS2TBroadcast;
  return (b == B::none && (dp == 4 || dp == 128) && (bit == 128 || bit == 256)) ||
         (b == B::x4 && dp == 32 && bit == 128) ||
         ((b == B::lw_0213 || b == B::lw_0123) && dp == 64 && bit == 128);
}

// `<T, D DP, B bit, C cta[, BCAST]>`.
Type CopyAtomSM100CopyS2TType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int dp, bit, cta;
  auto bcast = CopyS2TBroadcast::none;
  if (p.parseLess() || p.parseType(t) || p.parseComma() || countOf(p, "DP", dp) ||
      p.parseComma() || countOf(p, "bit", bit) || p.parseComma() || countOf(p, "cta", cta))
    return {};
  if (succeeded(p.parseOptionalComma()) && field(p, bcast)) return {};
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, dp, bit, cta, bcast);
}
void CopyAtomSM100CopyS2TType::print(AsmPrinter &p) const {
  p << '<' << getValType() << ", " << getNumDp() << " DP, " << getNumBit() << " bit, "
    << getNumCta() << " cta";
  if (getBroadcast() != CopyS2TBroadcast::none) p << ", " << stringifyEnum(getBroadcast());
  p << '>';
}
LogicalResult CopyAtomSM100CopyS2TType::verify(EmitFn emitError, Type, int dp, int bit, int cta,
                                               CopyS2TBroadcast b) {
  if (cta != 1 && cta != 2) return emitError() << "Expect cta is 1/2 but got: " << cta;
  if (!isUtccp(dp, bit, b))
    return emitError() << "The given UTCCP params: dp " << dp << ", bits " << bit
                       << ", broadcast " << stringifyEnum(b) << " is unsupported or illegal.";
  return success();
}

// `<T, num_dp = D, num_bit = B, num_cta = C, smem_major = M[, broadcast = X]>`.
Type CopyAtomSM100S2TCopyV2Type::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int dp, bit, cta;
  cute::MajorMode major;
  auto bcast = CopyS2TBroadcast::none;
  if (p.parseLess() || p.parseType(t) || nextKeyField(p, "num_dp", dp) ||
      nextKeyField(p, "num_bit", bit) || nextKeyField(p, "num_cta", cta) ||
      nextKeyField(p, "smem_major", major))
    return {};
  if (succeeded(p.parseOptionalComma()) && keyField(p, "broadcast", bcast)) return {};
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, dp, bit, cta, major,
                    bcast);
}
void CopyAtomSM100S2TCopyV2Type::print(AsmPrinter &p) const {
  p << '<' << getValType() << ", num_dp = " << getNumDp() << ", num_bit = " << getNumBit()
    << ", num_cta = " << getNumCta() << ", smem_major = " << stringifyEnum(getSmemMajor());
  if (getBroadcast() != CopyS2TBroadcast::none)
    p << ", broadcast = " << stringifyEnum(getBroadcast());
  p << '>';
}
LogicalResult CopyAtomSM100S2TCopyV2Type::verify(EmitFn emitError, Type, int dp, int bit,
                                                 int cta, cute::MajorMode,
                                                 CopyS2TBroadcast b) {
  if (cta != 1 && cta != 2) return emitError() << "expects cta to be 1 or 2, but got " << cta;
  if (!isUtccp(dp, bit, b))
    return emitError() << "expects valid UTCCP params, but got dp " << dp << ", bits " << bit
                       << ", broadcast " << stringifyEnum(b);
  return success();
}

//===----------------------------------------------------------------------===//
// ldmatrix / stmatrix
//===----------------------------------------------------------------------===//

Type CopyAtomLdsmType::parse(AsmParser &p) {
  Type t;
  cute::ShapeAttr mode;
  LdsmSzPattern sz;
  int n;
  UnitAttr trans;
  if (p.parseLess() || keyField(p, "val_type", t) || nextKeyField(p, "mode", mode) ||
      nextKeyField(p, "sz_pattern", sz) || nextKeyField(p, "num_matrices", n) ||
      p.parseComma() || parseTranspose(p, trans) || p.parseGreater())
    return {};
  return get(p.getContext(), t, mode, sz, n, trans);
}
void CopyAtomLdsmType::print(AsmPrinter &p) const {
  p << "<val_type = " << getValType() << ", mode = ";
  p.printStrippedAttrOrType(getMode());
  p << ", sz_pattern = " << stringifyEnum(getSzPattern()) << ", num_matrices = "
    << getNumMatrices() << (getTranspose() ? ", t>" : ", n>");
}

Type CopyAtomStsmType::parse(AsmParser &p) {
  Type t;
  cute::ShapeAttr mode;
  int n;
  UnitAttr trans;
  if (p.parseLess() || p.parseType(t) || nextKeyField(p, "mode", mode) ||
      nextKeyField(p, "num_matrices", n) || p.parseComma() || parseTranspose(p, trans) ||
      p.parseGreater())
    return {};
  return get(p.getContext(), t, mode, n, trans);
}
void CopyAtomStsmType::print(AsmPrinter &p) const {
  p << '<' << getValType() << ", mode = ";
  p.printStrippedAttrOrType(getMode());
  p << ", num_matrices = " << getNumMatrices() << (getTranspose() ? ", t>" : ", n>");
}

//===----------------------------------------------------------------------===//
// TMA atoms
//===----------------------------------------------------------------------===//

static LogicalResult verifySparsity(EmitFn emitError, int sparsity) {
  if (sparsity != 1 && sparsity != 2 && sparsity != 4 && sparsity != 8 && sparsity != 16)
    return emitError() << "expects sparsity to be one of (1, 2, 4, 8, 16), but got " << sparsity;
  return success();
}
// `<T[, sparsity = S], copy_bits = N, `
static ParseResult parseTmaHead(AsmParser &p, Type &t, int *sparsity, int &bits) {
  if (p.parseLess() || p.parseType(t) || p.parseComma()) return failure();
  if (sparsity && succeeded(p.parseOptionalKeyword("sparsity")) &&
      (p.parseEqual() || p.parseInteger(*sparsity) || p.parseComma()))
    return failure();
  return keyField(p, "copy_bits", bits);
}
static void printTmaHead(AsmPrinter &p, Type t, int sparsity, int bits) {
  p << '<' << t;
  if (sparsity != 1) p << ", sparsity = " << sparsity;
  p << ", copy_bits = " << bits;
}
static void printTmaTail(AsmPrinter &p, cute::StrideType stride, cute::LayoutType gbasis) {
  p << ", g_stride = ";
  p.printStrippedAttrOrType(stride);
  if (gbasis) {
    p << " tma_gbasis = ";
    p.printStrippedAttrOrType(gbasis);
  }
}
static void printTrue(AsmPrinter &p, StringRef key, bool value) {
  if (value) p << ' ' << key << " = true";
}

Type CopyAtomTmaLoadType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int sparsity = 1, bits, ctas;
  TmaLoadMode mode;
  cute::StrideType stride;
  cute::LayoutType gbasis;
  bool mcast = false, override = false, noOob = false;
  if (parseTmaHead(p, t, &sparsity, bits) || nextKeyField(p, "mode", mode) ||
      nextKeyField(p, "num_cta", ctas) || nextKeyField(p, "g_stride", stride) ||
      optKeyField(p, "mcast", mcast) || optKeyField(p, "tma_gbasis", gbasis) ||
      optKeyField(p, "override", override) || optKeyField(p, "no_fully_oob_tile", noOob) ||
      p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, sparsity, bits, mode,
                    ctas, stride, mcast, gbasis, override, noOob);
}
void CopyAtomTmaLoadType::print(AsmPrinter &p) const {
  printTmaHead(p, getValType(), getSparsity(), getCopyBits());
  p << ", mode = " << stringifyEnum(getMode()) << ", num_cta = " << getNumCta()
    << ", g_stride = ";
  p.printStrippedAttrOrType(getGStride());
  printTrue(p, "mcast", getMcast());
  if (getTmaGbasis()) {
    p << " tma_gbasis = ";
    p.printStrippedAttrOrType(getTmaGbasis());
  }
  printTrue(p, "override", getOverride());
  printTrue(p, "no_fully_oob_tile", getNoFullyOobTile());
  p << '>';
}
LogicalResult CopyAtomTmaLoadType::verify(EmitFn emitError, Type, int sparsity, int,
                                          TmaLoadMode, int, cute::StrideType, bool,
                                          cute::LayoutType, bool, bool) {
  return verifySparsity(emitError, sparsity);
}

Type CopyAtomTmaStoreType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Type t;
  int sparsity = 1, bits;
  TmaStoreMode mode;
  cute::StrideType stride;
  cute::LayoutType gbasis;
  bool override = false, noOob = false;
  if (parseTmaHead(p, t, &sparsity, bits) || nextKeyField(p, "mode", mode) ||
      nextKeyField(p, "g_stride", stride) || optKeyField(p, "tma_gbasis", gbasis) ||
      optKeyField(p, "override", override) || optKeyField(p, "no_fully_oob_tile", noOob) ||
      p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, sparsity, bits, mode,
                    stride, gbasis, override, noOob);
}
void CopyAtomTmaStoreType::print(AsmPrinter &p) const {
  printTmaHead(p, getValType(), getSparsity(), getCopyBits());
  p << ", mode = " << stringifyEnum(getMode());
  printTmaTail(p, getGStride(), getTmaGbasis());
  printTrue(p, "override", getOverride());
  printTrue(p, "no_fully_oob_tile", getNoFullyOobTile());
  p << '>';
}
LogicalResult CopyAtomTmaStoreType::verify(EmitFn emitError, Type, int sparsity, int,
                                           TmaStoreMode, cute::StrideType, cute::LayoutType,
                                           bool, bool) {
  return verifySparsity(emitError, sparsity);
}

Type CopyAtomTmaReduceType::parse(AsmParser &p) {
  Type t;
  int bits;
  TmaStoreMode mode;
  ReductionKind kind;
  cute::StrideType stride;
  cute::LayoutType gbasis;
  if (parseTmaHead(p, t, nullptr, bits) || nextKeyField(p, "mode", mode) ||
      nextKeyField(p, "kind", kind) || nextKeyField(p, "g_stride", stride) ||
      optKeyField(p, "tma_gbasis", gbasis) || p.parseGreater())
    return {};
  return get(p.getContext(), t, bits, mode, kind, stride, gbasis);
}
void CopyAtomTmaReduceType::print(AsmPrinter &p) const {
  printTmaHead(p, getValType(), 1, getCopyBits());
  p << ", mode = " << stringifyEnum(getMode()) << ", kind = " << stringifyEnum(getKind());
  printTmaTail(p, getGStride(), getTmaGbasis());
  p << '>';
}

Type CopyAtomIm2ColTmaLoadType::parse(AsmParser &p) {
  Type t;
  int bits, ctas;
  cute::StrideType stride;
  cute::LayoutType gbasis;
  bool mcast = false;
  if (parseTmaHead(p, t, nullptr, bits) || nextKeyField(p, "num_cta", ctas) ||
      nextKeyField(p, "g_stride", stride) || optKeyField(p, "mcast", mcast) ||
      optKeyField(p, "tma_gbasis", gbasis) || p.parseGreater())
    return {};
  return get(p.getContext(), t, bits, ctas, stride, mcast, gbasis);
}
void CopyAtomIm2ColTmaLoadType::print(AsmPrinter &p) const {
  printTmaHead(p, getValType(), 1, getCopyBits());
  p << ", num_cta = " << getNumCta() << ", g_stride = ";
  p.printStrippedAttrOrType(getGStride());
  printTrue(p, "mcast", getMcast());
  if (getTmaGbasis()) {
    p << " tma_gbasis = ";
    p.printStrippedAttrOrType(getTmaGbasis());
  }
  p << '>';
}

Type CopyAtomIm2ColTmaStoreType::parse(AsmParser &p) {
  Type t;
  int bits;
  cute::StrideType stride;
  cute::LayoutType gbasis;
  if (parseTmaHead(p, t, nullptr, bits) || nextKeyField(p, "g_stride", stride) ||
      optKeyField(p, "tma_gbasis", gbasis) || p.parseGreater())
    return {};
  return get(p.getContext(), t, bits, stride, gbasis);
}
void CopyAtomIm2ColTmaStoreType::print(AsmPrinter &p) const {
  printTmaHead(p, getValType(), 1, getCopyBits());
  printTmaTail(p, getGStride(), getTmaGbasis());
  p << '>';
}

// The TMA data format of an element type when none is given.
std::optional<TmaDataFormat> mlir::cutlass_compiler::cute_nvgpu::defaultTmaDataFormat(Type t) {
  if (auto s = llvm::dyn_cast<cute::SparseElemType>(t)) t = s.getPhysicalType();
  if (t.isF16()) return TmaDataFormat::F16_RN;
  if (t.isBF16()) return TmaDataFormat::BF16_RN;
  if (t.isF32()) return TmaDataFormat::F32_RN;
  if (t.isTF32()) return TmaDataFormat::TF32_RN;
  if (t.isF64()) return TmaDataFormat::F64_RN;
  if (!t.isIntOrFloat()) return std::nullopt;
  unsigned bits = t.getIntOrFloatBitWidth();
  bool isSigned = t.isSignedInteger();
  if (bits == 4) return TmaDataFormat::U4;
  if (bits <= 8) return TmaDataFormat::U8;
  if (bits == 16) return TmaDataFormat::U16;
  if (bits == 32) return isSigned ? TmaDataFormat::S32 : TmaDataFormat::U32;
  if (bits == 64) return isSigned ? TmaDataFormat::S64 : TmaDataFormat::U64;
  return std::nullopt;
}

namespace {
// `<[kind, ]T, copy_bits = N, tma_gbasis = L[, tma_format = F]>`.
template <typename Kind>
ParseResult parseNonExecTma(AsmParser &p, Kind *kind, Type &t, int &bits,
                            cute::LayoutType &gbasis, std::optional<TmaDataFormat> &format) {
  if (p.parseLess()) return failure();
  if (kind && (field(p, *kind) || p.parseComma())) return failure();
  if (p.parseType(t) || nextKeyField(p, "copy_bits", bits) ||
      nextKeyField(p, "tma_gbasis", gbasis))
    return failure();
  format = defaultTmaDataFormat(t);
  if (succeeded(p.parseOptionalComma())) {
    // Empty for an element type without a format, as the compiler prints it.
    if (p.parseKeyword("tma_format") || p.parseEqual()) return failure();
    if (succeeded(p.parseOptionalGreater())) {
      format = std::nullopt;
      return success();
    }
    TmaDataFormat f;
    if (field(p, f)) return failure();
    format = f;
  }
  return p.parseGreater();
}
template <typename T>
void printNonExecTma(AsmPrinter &p, T type) {
  p << type.getValType() << ", copy_bits = " << type.getCopyBits() << ", tma_gbasis = ";
  p.printStrippedAttrOrType(type.getTmaGbasis());
  p << ", tma_format = ";
  if (type.getTmaFormat()) p << stringifyEnum(*type.getTmaFormat());
  p << '>';
}
LogicalResult verifyNonExecTma(EmitFn emitError, int bits) {
  if (bits == 0) return emitError() << "invalid copy_bits, got 0";
  return success();
}
}  // namespace

#define NON_EXEC_TMA_WITH_KIND(TYPE, KIND)                                          \
  Type TYPE::parse(AsmParser &p) {                                                  \
    SMLoc loc = p.getCurrentLocation();                                             \
    KIND kind;                                                                      \
    Type t;                                                                         \
    int bits;                                                                       \
    cute::LayoutType gbasis;                                                        \
    std::optional<TmaDataFormat> format;                                            \
    if (parseNonExecTma(p, &kind, t, bits, gbasis, format)) return {};              \
    return getChecked([&] { return p.emitError(loc); }, p.getContext(), kind, t,    \
                      bits, gbasis, format);                                        \
  }                                                                                 \
  void TYPE::print(AsmPrinter &p) const {                                           \
    p << '<' << stringifyEnum(getKind()) << ", ";                                   \
    printNonExecTma(p, *this);                                                      \
  }                                                                                 \
  LogicalResult TYPE::verify(EmitFn emitError, KIND, Type, int bits,                \
                             cute::LayoutType, std::optional<TmaDataFormat>) {      \
    return verifyNonExecTma(emitError, bits);                                       \
  }
#define NON_EXEC_TMA(TYPE)                                                          \
  Type TYPE::parse(AsmParser &p) {                                                  \
    SMLoc loc = p.getCurrentLocation();                                             \
    Type t;                                                                         \
    int bits;                                                                       \
    cute::LayoutType gbasis;                                                        \
    std::optional<TmaDataFormat> format;                                            \
    if (parseNonExecTma<int>(p, nullptr, t, bits, gbasis, format)) return {};       \
    return getChecked([&] { return p.emitError(loc); }, p.getContext(), t, bits,    \
                      gbasis, format);                                              \
  }                                                                                 \
  void TYPE::print(AsmPrinter &p) const {                                           \
    p << '<';                                                                       \
    printNonExecTma(p, *this);                                                      \
  }                                                                                 \
  LogicalResult TYPE::verify(EmitFn emitError, Type, int bits, cute::LayoutType,    \
                             std::optional<TmaDataFormat>) {                        \
    return verifyNonExecTma(emitError, bits);                                       \
  }
NON_EXEC_TMA_WITH_KIND(CopyAtomNonExecTiledTmaLoadType, TiledTmaLoad)
NON_EXEC_TMA_WITH_KIND(CopyAtomNonExecTiledTmaReduceType, ReductionKind)
NON_EXEC_TMA_WITH_KIND(CopyAtomNonExecIm2ColTmaLoadType, Im2ColTmaLoad)
NON_EXEC_TMA_WITH_KIND(CopyAtomNonExec2DGather4TmaLoadType, GatherScatterTmaLoad)
NON_EXEC_TMA(CopyAtomNonExecTiledTmaStoreType)
NON_EXEC_TMA(CopyAtomNonExecIm2ColTmaStoreType)
NON_EXEC_TMA(CopyAtomNonExec2DScatter4TmaStoreType)
#undef NON_EXEC_TMA_WITH_KIND
#undef NON_EXEC_TMA

//===----------------------------------------------------------------------===//
// MMA atoms
//===----------------------------------------------------------------------===//

Type UniversalFmaAtomType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  cute::ShapeAttr shape;
  Type a, b, c;
  if (p.parseLess() || parseShapeMNK(p, shape) || p.parseComma() || p.parseLParen() ||
      p.parseType(a) || p.parseComma() || p.parseType(b) || p.parseRParen() ||
      p.parseArrow() || p.parseType(c) || p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), shape, a, b, c);
}
void UniversalFmaAtomType::print(AsmPrinter &p) const {
  p << '<';
  printShapeMNK(p, getShapeMnk());
  p << ", (" << getAType() << ", " << getBType() << ") -> " << getCType() << " >";
}
LogicalResult UniversalFmaAtomType::verify(EmitFn emitError, cute::ShapeAttr shape, Type, Type,
                                           Type) {
  return verifyShapeMNK(emitError, shape);
}

// `<MNK, elem_type = (A, B, C)[, int_overflow = O[, binary_op = B]] >`.
Type MmaAtomSM80Type::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  cute::ShapeAttr shape;
  Type a, b, c;
  MMAIntOverflowAttr overflow;
  BinaryOpAttr op;
  if (p.parseLess() || parseShapeMNK(p, shape) || p.parseComma() || elemTypes(p, a, b, c))
    return {};
  if (succeeded(p.parseOptionalComma()) &&
      (keyField(p, "int_overflow", overflow) ||
       (succeeded(p.parseOptionalComma()) && keyField(p, "binary_op", op))))
    return {};
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), shape, a, b, c, overflow,
                    op);
}
void MmaAtomSM80Type::print(AsmPrinter &p) const {
  p << '<';
  printShapeMNK(p, getShapeMnk());
  p << ", ";
  printElemTypes(p, getAType(), getBType(), getCType());
  printAttrField(p, "int_overflow", getIntOverflow());
  printAttrField(p, "binary_op", getBinaryOp());
  p << " >";
}
LogicalResult MmaAtomSM80Type::verify(EmitFn emitError, cute::ShapeAttr shape, Type, Type, Type,
                                      MMAIntOverflowAttr, BinaryOpAttr) {
  return verifyShapeMNK(emitError, shape);
}

// `<MNK, elem_type = (A, B, C), md_format = F[, int_overflow = O] >`.
Type MmaAtomSM80SparseType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  cute::ShapeAttr shape;
  Type a, b, c;
  SparseMetadataFormat format;
  MMAIntOverflowAttr overflow;
  if (p.parseLess() || parseShapeMNK(p, shape) || p.parseComma() || elemTypes(p, a, b, c) ||
      nextKeyField(p, "md_format", format))
    return {};
  if (succeeded(p.parseOptionalComma()) && keyField(p, "int_overflow", overflow)) return {};
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), shape, a, b, c, format,
                    overflow);
}
void MmaAtomSM80SparseType::print(AsmPrinter &p) const {
  p << '<';
  printShapeMNK(p, getShapeMnk());
  p << ", ";
  printElemTypes(p, getAType(), getBType(), getCType());
  p << ", md_format = " << stringifyEnum(getSparseMetadataFormat());
  printAttrField(p, "int_overflow", getIntOverflow());
  p << " >";
}
LogicalResult MmaAtomSM80SparseType::verify(EmitFn emitError, cute::ShapeAttr shape, Type, Type,
                                            Type, SparseMetadataFormat, MMAIntOverflowAttr) {
  return verifyShapeMNK(emitError, shape);
}

Type MmaAtomSM89Type::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  cute::ShapeAttr shape;
  Type a, b, c;
  if (p.parseLess() || parseShapeMNK(p, shape) || p.parseComma() || elemTypes(p, a, b, c) ||
      p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), shape, a, b, c);
}
void MmaAtomSM89Type::print(AsmPrinter &p) const {
  p << '<';
  printShapeMNK(p, getShapeMnk());
  p << ", ";
  printElemTypes(p, getAType(), getBType(), getCType());
  p << " >";
}
LogicalResult MmaAtomSM89Type::verify(EmitFn emitError, cute::ShapeAttr shape, Type, Type,
                                      Type) {
  return verifyShapeMNK(emitError, shape);
}

// `<MNK, ab_major = (A, B), elem_type = (A, B, C), frag_kind = ss|rs
//  {, int_overflow = O | , a_neg | , b_neg}>`.
Type MmaAtomSM90Type::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  cute::ShapeAttr shape;
  cute::MajorMode aMajor, bMajor;
  Type a, b, c;
  MmaFragKind frag;
  MMAIntOverflowAttr overflow;
  UnitAttr aNeg, bNeg;
  if (p.parseLess() || parseShapeMNK(p, shape) || p.parseComma() ||
      p.parseKeyword("ab_major") || p.parseEqual() || pairOf(p, aMajor, bMajor) ||
      p.parseComma() || elemTypes(p, a, b, c) || p.parseComma() ||
      parseFragKind(p, MmaFragKind::rmem, frag))
    return {};
  while (succeeded(p.parseOptionalComma())) {
    SMLoc at = p.getCurrentLocation();
    if (succeeded(p.parseOptionalKeyword("a_neg"))) aNeg = UnitAttr::get(p.getContext());
    else if (succeeded(p.parseOptionalKeyword("b_neg"))) bNeg = UnitAttr::get(p.getContext());
    else if (succeeded(p.parseOptionalKeyword("int_overflow"))) {
      if (p.parseEqual() || field(p, overflow)) return {};
    } else {
      p.emitError(at, "expected `int_overflow`, `a_neg`, or `b_neg`");
      return {};
    }
  }
  if (p.parseGreater()) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), shape, aMajor, bMajor, a,
                    b, c, frag, overflow, aNeg, bNeg);
}
void MmaAtomSM90Type::print(AsmPrinter &p) const {
  p << '<';
  printShapeMNK(p, getShapeMnk());
  p << ", ab_major = (" << stringifyEnum(getAMajor()) << ", " << stringifyEnum(getBMajor())
    << "), ";
  printElemTypes(p, getAType(), getBType(), getCType());
  p << ", frag_kind = " << fragKindText(getAFragKind());
  printAttrField(p, "int_overflow", getIntOverflow());
  if (getANeg()) p << ", a_neg";
  if (getBNeg()) p << ", b_neg";
  p << '>';
}
LogicalResult MmaAtomSM90Type::verify(EmitFn emitError, cute::ShapeAttr shape, cute::MajorMode,
                                      cute::MajorMode, Type, Type, Type, MmaFragKind,
                                      MMAIntOverflowAttr, UnitAttr, UnitAttr) {
  return verifyShapeMNK(emitError, shape);
}

// `<MNK, vec_size = N, elem_type = (A, B, C), sf_type = T, use_sf_layout_TV = B>`.
Type MmaAtomSM120BlockScaledType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  cute::ShapeAttr shape;
  int vec;
  Type a, b, c, sf;
  bool useTV;
  if (p.parseLess() || parseShapeMNK(p, shape) || nextKeyField(p, "vec_size", vec) ||
      p.parseComma() || elemTypes(p, a, b, c) || nextKeyField(p, "sf_type", sf) ||
      nextKeyField(p, "use_sf_layout_TV", useTV) || p.parseGreater())
    return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), shape, vec, a, b, c, sf,
                    useTV);
}
void MmaAtomSM120BlockScaledType::print(AsmPrinter &p) const {
  p << '<';
  printShapeMNK(p, getShapeMnk());
  p << ", vec_size = " << getVecSize() << ", ";
  printElemTypes(p, getAType(), getBType(), getCType());
  p << ", sf_type = " << getSfType() << ", use_sf_layout_TV = "
    << (getUseSfLayout_TV() ? "true" : "false") << '>';
}
LogicalResult MmaAtomSM120BlockScaledType::verify(EmitFn emitError, cute::ShapeAttr shape, int,
                                                  Type, Type, Type, Type, bool) {
  return verifyShapeMNK(emitError, shape);
}

// UMMA atoms: one grammar, fields present by kind (see CuteNVGPUTypes.td).
namespace {
struct Umma {
  bool sf, sp, bs, sm107;
  cute::ShapeAttr shape;
  int numCta = 1;
  cute::MajorMode aMajor, bMajor;
  Type a, b, c, sfType, eType;
  SparseMetadataFormat metadata = SparseMetadataFormat::tid;
  MmaFragKind frag = MmaFragKind::smem_desc;
  int scaleOrVec = 0;
  Arch arch = Arch::sm_100;
  MmaCollectorOp aCollector = MmaCollectorOp::discard, bCollector = MmaCollectorOp::discard;
  MMAIntOverflowAttr overflow;
};

ParseResult parseUmma(AsmParser &p, Umma &u) {
  if (p.parseLess() || parseShapeMNK(p, u.shape) || nextKeyField(p, "num_cta", u.numCta) ||
      p.parseComma() || p.parseKeyword("ab_major") || p.parseEqual() ||
      pairOf(p, u.aMajor, u.bMajor) || p.parseComma() || elemTypes(p, u.a, u.b, u.c))
    return failure();
  if (u.sf && nextKeyField(p, "sf_type", u.sfType)) return failure();
  if (u.sp && !u.sf && nextKeyField(p, "e_type", u.eType)) return failure();
  if (u.sp && nextKeyField(p, "sparse_metadata_format", u.metadata)) return failure();
  if (p.parseComma() || parseFragKind(p, MmaFragKind::tmem, u.frag) ||
      nextKeyField(p, u.bs ? "vec_size" : "c_scale_exp", u.scaleOrVec))
    return failure();
  bool comma = succeeded(p.parseOptionalComma());
  if (comma && u.bs && !u.sm107 && succeeded(p.parseOptionalKeyword("arch_promote"))) {
    if (p.parseEqual() || field(p, u.arch)) return failure();
    comma = succeeded(p.parseOptionalComma());
  }
  if (u.sm107) {
    if (!comma) return p.parseComma();
    if (p.parseKeyword("ab_collector_op") || p.parseEqual() ||
        pairOf(p, u.aCollector, u.bCollector))
      return failure();
    comma = succeeded(p.parseOptionalComma());
  }
  if (comma && keyField(p, "int_overflow", u.overflow)) return failure();
  return p.parseGreater();
}

void printUmma(AsmPrinter &p, const Umma &u) {
  p << '<';
  printShapeMNK(p, u.shape);
  p << ", num_cta = " << u.numCta << ", ab_major = (" << stringifyEnum(u.aMajor) << ", "
    << stringifyEnum(u.bMajor) << "), ";
  printElemTypes(p, u.a, u.b, u.c);
  if (u.sf) p << ", sf_type = " << u.sfType;
  if (u.sp && !u.sf) p << ", e_type = " << u.eType;
  if (u.sp) p << ", sparse_metadata_format = " << stringifyEnum(u.metadata);
  p << ", frag_kind = " << fragKindText(u.frag) << (u.bs ? ", vec_size = " : ", c_scale_exp = ")
    << u.scaleOrVec;
  if (u.bs && !u.sm107 && (u.arch != Arch::sm_100 || u.overflow))
    p << ", arch_promote = " << stringifyEnum(u.arch);
  if (u.sm107)
    p << ", ab_collector_op = (" << stringifyEnum(u.aCollector) << ", "
      << stringifyEnum(u.bCollector) << ")";
  printAttrField(p, "int_overflow", u.overflow);
  p << '>';
}

// `sourced` picks the atom's wording of the tensor-memory A rule.
LogicalResult verifyUmma(EmitFn emitError, cute::ShapeAttr shape, int numCta, MmaFragKind frag,
                         cute::MajorMode aMajor, int cScaleExp = 0, bool sourced = false) {
  if (failed(verifyShapeMNK(emitError, shape))) return failure();
  if (numCta != 1 && numCta != 2)
    return emitError() << "expects `num_cta` equals to 1 or 2, but got " << numCta;
  if (cScaleExp > 15)
    return emitError() << "expects `c_scale_exp` in range [0, 15], but got " << cScaleExp;
  if (frag == MmaFragKind::tmem && aMajor != cute::MajorMode::k)
    return emitError() << (sourced ? "a operand must be k-major when sourced from tmem"
                                   : "a operand must be of major-mode k when coming from tmem");
  return success();
}
}  // namespace

Type MmaAtomSM100UMMAType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{false, false, false, false};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.frag, u.scaleOrVec, u.overflow);
}
void MmaAtomSM100UMMAType::print(AsmPrinter &p) const {
  Umma u{false, false, false, false};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.frag = getAFragKind(), u.scaleOrVec = getCScaleExp(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM100UMMAType::verify(EmitFn emitError, cute::ShapeAttr shape, int numCta,
                                           cute::MajorMode aMajor, cute::MajorMode, Type, Type,
                                           Type, MmaFragKind frag, int cScaleExp,
                                           MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, cScaleExp);
}

Type MmaAtomSM100UMMASparseType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{false, true, false, false};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.eType, u.metadata, u.frag,
                    u.scaleOrVec, u.overflow);
}
void MmaAtomSM100UMMASparseType::print(AsmPrinter &p) const {
  Umma u{false, true, false, false};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.eType = getEType(), u.metadata = getSparseMetadataFormat(), u.frag = getAFragKind(),
  u.scaleOrVec = getCScaleExp(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM100UMMASparseType::verify(EmitFn emitError, cute::ShapeAttr shape,
                                                 int numCta, cute::MajorMode aMajor, cute::MajorMode,
                                                 Type, Type, Type, Type, SparseMetadataFormat,
                                                 MmaFragKind frag, int cScaleExp,
                                                 MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, cScaleExp);
}

Type MmaAtomSM100UMMABlockScaledType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{true, false, true, false};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.sfType, u.frag, u.scaleOrVec, u.arch,
                    u.overflow);
}
void MmaAtomSM100UMMABlockScaledType::print(AsmPrinter &p) const {
  Umma u{true, false, true, false};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.sfType = getSfType(), u.frag = getAFragKind(), u.scaleOrVec = getVecSize(),
  u.arch = getArchPromote(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM100UMMABlockScaledType::verify(EmitFn emitError, cute::ShapeAttr shape,
                                                      int numCta, cute::MajorMode aMajor,
                                                      cute::MajorMode, Type, Type, Type, Type,
                                                      MmaFragKind frag, int, Arch,
                                                      MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, 0, /*sourced=*/true);
}

Type MmaAtomSM100UMMABlockScaledSparseType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{true, true, true, false};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.sfType, u.metadata, u.frag,
                    u.scaleOrVec, u.arch, u.overflow);
}
void MmaAtomSM100UMMABlockScaledSparseType::print(AsmPrinter &p) const {
  Umma u{true, true, true, false};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.sfType = getSfType(), u.metadata = getSparseMetadataFormat(), u.frag = getAFragKind(),
  u.scaleOrVec = getVecSize(), u.arch = getArchPromote(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM100UMMABlockScaledSparseType::verify(
    EmitFn emitError, cute::ShapeAttr shape, int numCta, cute::MajorMode aMajor, cute::MajorMode,
    Type, Type, Type, Type, SparseMetadataFormat, MmaFragKind frag, int, Arch,
    MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor);
}

Type MmaAtomSM107UMMAType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{false, false, false, true};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.frag, u.scaleOrVec, u.aCollector,
                    u.bCollector, u.overflow);
}
void MmaAtomSM107UMMAType::print(AsmPrinter &p) const {
  Umma u{false, false, false, true};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.frag = getAFragKind(), u.scaleOrVec = getCScaleExp(), u.aCollector = getACollectorOp(),
  u.bCollector = getBCollectorOp(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM107UMMAType::verify(EmitFn emitError, cute::ShapeAttr shape, int numCta,
                                           cute::MajorMode aMajor, cute::MajorMode, Type, Type,
                                           Type, MmaFragKind frag, int cScaleExp,
                                           MmaCollectorOp, MmaCollectorOp, MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, cScaleExp);
}

Type MmaAtomSM107UMMASparseType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{false, true, false, true};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.eType, u.metadata, u.frag,
                    u.scaleOrVec, u.aCollector, u.bCollector, u.overflow);
}
void MmaAtomSM107UMMASparseType::print(AsmPrinter &p) const {
  Umma u{false, true, false, true};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.eType = getEType(), u.metadata = getSparseMetadataFormat(), u.frag = getAFragKind(),
  u.scaleOrVec = getCScaleExp(), u.aCollector = getACollectorOp(),
  u.bCollector = getBCollectorOp(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM107UMMASparseType::verify(EmitFn emitError, cute::ShapeAttr shape,
                                                 int numCta, cute::MajorMode aMajor, cute::MajorMode,
                                                 Type, Type, Type, Type, SparseMetadataFormat,
                                                 MmaFragKind frag, int cScaleExp, MmaCollectorOp,
                                                 MmaCollectorOp, MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, cScaleExp);
}

Type MmaAtomSM107UMMABlockScaledType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{true, false, true, true};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.sfType, u.frag, u.scaleOrVec,
                    u.aCollector, u.bCollector, u.overflow);
}
void MmaAtomSM107UMMABlockScaledType::print(AsmPrinter &p) const {
  Umma u{true, false, true, true};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.sfType = getSfType(), u.frag = getAFragKind(), u.scaleOrVec = getVecSize(),
  u.aCollector = getACollectorOp(), u.bCollector = getBCollectorOp(),
  u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM107UMMABlockScaledType::verify(EmitFn emitError, cute::ShapeAttr shape,
                                                      int numCta, cute::MajorMode aMajor,
                                                      cute::MajorMode, Type, Type, Type, Type,
                                                      MmaFragKind frag, int, MmaCollectorOp,
                                                      MmaCollectorOp, MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, 0, /*sourced=*/true);
}

Type MmaAtomSM107UMMABlockScaledSparseType::parse(AsmParser &p) {
  SMLoc loc = p.getCurrentLocation();
  Umma u{true, true, true, true};
  if (parseUmma(p, u)) return {};
  return getChecked([&] { return p.emitError(loc); }, p.getContext(), u.shape, u.numCta,
                    u.aMajor, u.bMajor, u.a, u.b, u.c, u.sfType, u.metadata, u.frag,
                    u.scaleOrVec, u.aCollector, u.bCollector, u.overflow);
}
void MmaAtomSM107UMMABlockScaledSparseType::print(AsmPrinter &p) const {
  Umma u{true, true, true, true};
  u.shape = getShapeMnk(), u.numCta = getNumCta(), u.aMajor = getAMajor(),
  u.bMajor = getBMajor(), u.a = getAType(), u.b = getBType(), u.c = getCType(),
  u.sfType = getSfType(), u.metadata = getSparseMetadataFormat(), u.frag = getAFragKind(),
  u.scaleOrVec = getVecSize(), u.aCollector = getACollectorOp(),
  u.bCollector = getBCollectorOp(), u.overflow = getIntOverflow();
  printUmma(p, u);
}
LogicalResult MmaAtomSM107UMMABlockScaledSparseType::verify(
    EmitFn emitError, cute::ShapeAttr shape, int numCta, cute::MajorMode aMajor, cute::MajorMode,
    Type, Type, Type, Type, SparseMetadataFormat, MmaFragKind frag, int, MmaCollectorOp,
    MmaCollectorOp, MMAIntOverflowAttr) {
  return verifyUmma(emitError, shape, numCta, frag, aMajor, 0, /*sourced=*/true);
}

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

// `!cute_nvgpu.iN[<divby D>]` is the compiler's alias of `!cute.iN[<divby D>]`.
Type CuteNVGPUDialect::parseType(DialectAsmParser &parser) const {
  SMLoc loc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type type;
  OptionalParseResult result = generatedTypeParser(parser, &mnemonic, type);
  if (result.has_value()) return succeeded(*result) ? type : Type();
  StringRef width = mnemonic;
  if (width.consume_front("i") && !width.empty() && llvm::all_of(width, llvm::isDigit)) {
    std::string spec = ("!cute." + mnemonic).str();
    int64_t divisor;
    if (succeeded(parser.parseOptionalLess())) {
      if (parser.parseKeyword("divby") || parser.parseInteger(divisor) || parser.parseGreater())
        return {};
      spec += "<divby " + std::to_string(divisor) + ">";
    }
    return mlir::parseType(spec, getContext());
  }
  parser.emitError(loc) << "unknown  type `" << mnemonic << "` in dialect `"
                        << getNamespace() << "`";
  return {};
}

void CuteNVGPUDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer))) return;
  llvm_unreachable("unexpected non-CuteNVGPU type");
}

void CuteNVGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUOps.cpp.inc"
      >();
}
