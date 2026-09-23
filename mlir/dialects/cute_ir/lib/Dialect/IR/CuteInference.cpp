//===- CuteInference.cpp - result types the DSL compiler would infer -----===//
//
// Every operation the DSL compiler infers result types for, inferred here
// over cutegen, so a module built in this tree carries the types the compiler
// expects. Where the compiler's rule is not recovered for some inputs (other
// atoms, tilings or layouts than those covered), inference fails rather than
// guesses. test:dialect_inference_test holds these to the compiler's choice.

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "cute_ir/Dialect/Cute/IR/CuteInference.h"
#include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUDialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::cutlass_compiler::cute;
namespace cg = cutegen;
namespace cute_nvgpu = mlir::cutlass_compiler::cute_nvgpu;
// The dialect's memref, not the builtin one.
using CuteMemRefType = ::mlir::cutlass_compiler::cute::MemRefType;

namespace {

// Composed layouts that turned out plain are typed as plain layouts, as the
// compiler types them.
Type layoutType(MLIRContext *ctx, cg::layout l) {
  return LayoutType::get(ctx, std::move(l));
}
Type layoutType(MLIRContext *ctx, cg::composed_layout l) {
  if (l.is_normal_layout()) return LayoutType::get(ctx, l.layout_b());
  return ComposedLayoutType::get(ctx, std::move(l));
}

// `f` runs over the algebra value of a layout or composed layout type; the
// result keeps the input's kind (see rebuild for views).
template <class F>
Type mapLayout(MLIRContext *ctx, Type layout, F &&f) {
  if (auto l = llvm::dyn_cast<LayoutType>(layout)) {
    auto r = f(l.getRef());
    if (!cg::is_valid(r)) return {};
    return layoutType(ctx, std::move(r));
  }
  if (auto l = llvm::dyn_cast<ComposedLayoutType>(layout)) {
    auto r = f(l.getRef());
    if (!cg::is_valid(r)) return {};
    return ComposedLayoutType::get(ctx, std::move(r));
  }
  return {};
}

// What carries a layout: the layout itself, a memref (pointer + layout) or a
// coordinate tensor (arithmetic tuple + layout).
struct Carrier {
  Type layout;
  PtrType ptr;             // memref
  IntTupleType iter;       // coord_tensor
  Type desc;               // cute_nvgpu.smem_desc_view
};

std::optional<Carrier> carrierOf(Type t) {
  if (llvm::isa<LayoutType, ComposedLayoutType>(t)) return Carrier{t, {}, {}, {}};
  if (auto m = llvm::dyn_cast<CuteMemRefType>(t))
    return Carrier{m.getLayout(), m.getPtr(), {}, {}};
  if (auto c = llvm::dyn_cast<CoordTensorType>(t))
    return Carrier{c.getLayout(), {}, c.getArithTuple(), {}};
  if (auto v = llvm::dyn_cast<cute_nvgpu::SmemDescViewType>(t))
    return Carrier{LayoutType::get(t.getContext(), v.getLayout()), {}, {}, v.getDesc()};
  return std::nullopt;
}

Type rebuild(MLIRContext *ctx, const Carrier &c, Type layout) {
  if (!layout) return {};
  // A view's layout is plain when its composed layout turned out plain.
  if (auto l = llvm::dyn_cast<ComposedLayoutType>(layout); l && (c.ptr || c.iter || c.desc))
    layout = layoutType(ctx, l.getRef());
  if (c.ptr) return CuteMemRefType::get(ctx, c.ptr, layout);
  if (c.iter) return CoordTensorType::get(ctx, c.iter, layout);
  if (c.desc) {
    auto l = llvm::dyn_cast<LayoutType>(layout);
    return l ? cute_nvgpu::SmemDescViewType::get(ctx, c.desc, l.getAttr()) : Type();
  }
  return layout;
}

// The bit width of an element type; tf32 counts as its 32-bit storage.
int64_t bitsOfType(Type t) {
  if (t.isTF32()) return 32;
  return t.isIntOrFloat() ? t.getIntOrFloatBitWidth() : 0;
}

// The alignment a pointer keeps after moving by `offset` elements: the gcd
// with what is known of the offset, a static value or a divisibility; the
// element's natural alignment when less than a byte is known.
uint64_t movedAlignment(PtrType ptr, const cg::int_tuple &offset) {
  Type elem = ptr.getValueType();
  uint64_t bits = elem && elem.isIntOrFloat() ? bitsOfType(elem) : 8;
  uint64_t natural = PtrType::getNaturalAlignment(elem);
  uint64_t align = ptr.getAlignment() ? ptr.getAlignment() : natural;
  uint64_t known = 0;
  if (cg::holds_int(offset)) {
    int64_t v = offset.as_int64();
    if (v == 0) return ptr.getAlignment();
    known = static_cast<uint64_t>(v < 0 ? -v : v);
  } else if (cg::holds_dynamic_int(offset)) {
    known = static_cast<uint64_t>(
        std::get<cg::mlir_dynamic_t>(offset).get_properties().divisibility);
  } else {
    return natural;
  }
  uint64_t result = std::gcd(align * 8, known * bits) / 8;
  return result ? result : natural;
}

PtrType movedPtr(PtrType ptr, const cg::int_tuple &offset) {
  return PtrType::get(ptr.getContext(), ptr.getValueType(), ptr.getAddressSpace(),
                      movedAlignment(ptr, offset), ptr.getSwizzle(),
                      ptr.getBitlayout());
}

// A coordinate with its underscores read as zero, for evaluating a layout.
std::optional<cg::coord> zeroUnderscores(const cg::coord &c) {
  std::string text = cg::to_string(c);
  std::replace(text.begin(), text.end(), '_', '0');
  return cg::from_string<cg::coord>(text);
}

// Slicing a carrier: the layout is sliced and the iterator moves by the
// layout evaluated at the coordinate.
Type sliceCarrier(MLIRContext *ctx, const Carrier &c, const cg::coord &coord) {
  cg::int_tuple offset;
  Type layout;
  if (auto l = llvm::dyn_cast<LayoutType>(c.layout)) {
    auto [sliced, off] = cg::slice_and_offset(coord, l.getRef());
    if (!cg::is_valid(sliced)) return {};
    layout = layoutType(ctx, std::move(sliced));
    offset = off;
  } else if (auto l = llvm::dyn_cast<ComposedLayoutType>(c.layout)) {
    // The composed layout's offset takes the move; the iterator stays.
    auto sliced = cg::slice(coord, l.getRef());
    if (!cg::is_valid(sliced)) return {};
    return rebuild(ctx, c, layoutType(ctx, std::move(sliced)));
  } else {
    return {};
  }
  if (c.ptr) return CuteMemRefType::get(ctx, movedPtr(c.ptr, offset), layout);
  if (c.iter) {
    auto sum = cg::arith_tuple_sum<cg::int_tuple>(c.iter.getRef(), offset);
    if (!cg::is_valid(sum)) return {};
    return CoordTensorType::get(ctx, IntTupleType::get(ctx, std::move(sum)),
                                layout);
  }
  // A descriptor iterator does not move.
  return rebuild(ctx, c, layout);
}

// The compiler writes a stride of 0 on every mode of extent 1 that a divide
// produces; cutegen keeps the stride the algebra gave it.
cg::layout zeroUnitModes(cg::layout l) {
  if (cg::holds_leaf(l.shape())) {
    if (cg::holds_int(l.shape()) && l.shape().as_int64() == 1) {
      if (auto unit = cg::from_string<cg::layout>("1:0")) return *unit;
    }
    return l;
  }
  // Shapes and strides are congruent, so walk both spellings together.
  auto shape = cg::to_string(l.shape());
  auto stride = cg::to_string(l.stride());
  auto fs = cg::flatten(l.shape());
  auto ft = cg::flatten(l.stride());
  if (cg::rank(fs) != cg::rank(ft)) return l;
  bool changed = false;
  for (size_t i = 0; i < cg::rank(fs); ++i) {
    if (cg::holds_int(fs[i]) && fs[i].as_int64() == 1 &&
        !(cg::holds_int(ft[i]) && ft[i].as_int64() == 0)) {
      changed = true;
    }
  }
  if (!changed) return l;
  // Rewrite the stride text leaf by leaf, in flattened order.
  std::string out;
  size_t leaf = 0;
  auto stridesAreLeaves = [&](size_t i) {
    return cg::holds_int(fs[i]) && fs[i].as_int64() == 1;
  };
  size_t i = 0;
  while (i < stride.size()) {
    char ch = stride[i];
    if (ch == '(' || ch == ')' || ch == ',') {
      out += ch;
      ++i;
      continue;
    }
    size_t j = i;
    int depth = 0;
    while (j < stride.size() &&
           !((stride[j] == ',' || stride[j] == ')') && depth == 0)) {
      if (stride[j] == '<' || stride[j] == '{') ++depth;
      if (stride[j] == '>' || stride[j] == '}') --depth;
      ++j;
    }
    out += stridesAreLeaves(leaf) ? std::string("0") : stride.substr(i, j - i);
    ++leaf;
    i = j;
  }
  auto rebuilt = cg::from_string<cg::layout>(shape + ":" + out);
  return rebuilt ? *rebuilt : l;
}

const cg::layout &outerOf(const cg::layout &l) { return l; }
cg::layout outerOf(const cg::composed_layout &l) { return l.layout_b(); }

cg::composed_layout zeroUnitModes(cg::composed_layout l) {
  if (l.is_a_swizzle())
    return cg::composed_layout(l.swizzle_a(), l.offset(), zeroUnitModes(l.layout_b()));
  return cg::composed_layout(l.layout_a(), l.offset(), zeroUnitModes(l.layout_b()));
}

template <class F>
LogicalResult infer(std::optional<Location> loc, SmallVectorImpl<Type> &out,
                    F &&f) {
  Type t = f();
  if (!t) return emitOptionalError(loc, "cannot infer the result type");
  out.push_back(t);
  return success();
}

// `like`'s nesting with its leaves replaced, in order, by `leaves`.
template <class T>
std::string nestLike(const T &like, llvm::ArrayRef<std::string> leaves, size_t &i) {
  if (!cg::holds_vector(like)) return i < leaves.size() ? leaves[i++] : "x";
  std::string s = "(";
  for (size_t k = 0; k < cg::rank(like); ++k)
    s += (k ? "," : "") + nestLike(like[k], leaves, i);
  return s + ")";
}

template <class T>
void leavesOf(const T &t, std::vector<std::string> &out) {
  if (!cg::holds_vector(t)) return out.push_back(cg::to_string(t));
  for (size_t k = 0; k < cg::rank(t); ++k) leavesOf(t[k], out);
}

bool isOne(const std::string &leaf) { return leaf == "1"; }

// The plain layout of a layout, or the outer layout of a composed one.
std::optional<cg::layout> outerLayout(Type t) {
  if (auto l = llvm::dyn_cast<LayoutType>(t)) return l.getRef();
  if (auto c = llvm::dyn_cast<ComposedLayoutType>(t)) return c.getRef().layout_b();
  return std::nullopt;
}

// The plain layout of a layout or swizzled composed layout.
std::optional<cg::layout> swizzledOuter(Type t) {
  if (auto l = llvm::dyn_cast<LayoutType>(t)) return l.getRef();
  auto c = llvm::dyn_cast<ComposedLayoutType>(t);
  if (c && c.getRef().is_a_swizzle()) return c.getRef().layout_b();
  return std::nullopt;
}

// A dynamic integer's spelling: `?`, with its width when 64 and its
// divisibility when above 1.
std::string dynamicInt(int64_t div, bool wide) {
  std::string props = std::string(wide ? "i64" : "") + (wide && div > 1 ? " " : "") +
                      (div > 1 ? "div=" + std::to_string(div) : "");
  return props.empty() ? "?" : "?{" + props + "}";
}

// idx2crd of a scalar `idx` over `shape` (column-major), keeping what a
// dynamic index's divisibility tells of each coordinate, as the compiler
// does (cutegen drops it): a divisible mode's coordinate is 0; a mode whose
// extent the divisibility divides keeps it; the rest know nothing.
std::optional<std::string> splitIndex(const cg::int_tuple &idx, const cg::shape &shape) {
  std::vector<std::string> leaves, crd;
  leavesOf(shape, leaves);
  bool known = cg::holds_int(idx);
  if (!known && !cg::holds_dynamic_int(idx)) return std::nullopt;
  int64_t v = known ? idx.as_int64() : 0, div = 1;
  bool wide = false;
  if (!known) {
    const auto &p = std::get<cg::mlir_dynamic_t>(idx).get_properties();
    div = p.divisibility;
    wide = p.width == 64;
  }
  for (size_t i = 0; i < leaves.size(); ++i) {
    int64_t n = 0;
    bool staticExtent = !StringRef(leaves[i]).getAsInteger(10, n) && n > 0;
    bool last = i + 1 == leaves.size();
    if (known) {
      if (!staticExtent) return std::nullopt;
      crd.push_back(std::to_string(last ? v : v % n));
      v /= n;
    } else if (last) {
      crd.push_back(dynamicInt(div, wide));
    } else if (staticExtent && div % n == 0) {
      crd.push_back("0");
      div /= n;
    } else {
      crd.push_back(dynamicInt(staticExtent && n % div == 0 ? div : 1, wide));
      div = 1;
    }
  }
  size_t i = 0;
  return nestLike(shape, crd, i);
}
}  // namespace

// ---------------------------------------------------------------- cute

LogicalResult AddOffsetOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto offset = llvm::dyn_cast<IntTupleType>(a.getOffset().getType());
  if (!offset) return emitOptionalError(loc, "offset must be an int tuple");
  return infer(loc, out, [&]() -> Type {
    Type src = a.getSrc().getType();
    if (auto p = llvm::dyn_cast<PtrType>(src)) return movedPtr(p, offset.getRef());
    if (auto it = llvm::dyn_cast<ArithTupleIteratorType>(src)) {
      auto sum = cg::arith_tuple_sum<cg::int_tuple>(it.getArithTuple().getRef(),
                                                    offset.getRef());
      if (!cg::is_valid(sum)) return {};
      return ArithTupleIteratorType::get(ctx, IntTupleType::get(ctx, sum));
    }
    return {};
  });
}

namespace {
// Pads `input` to `rank` modes with `element`, after (append) or before.
template <class Op>
LogicalResult inferToRank(MLIRContext *ctx, std::optional<Location> loc,
                          ValueRange operands, DictionaryAttr attrs,
                          PropertyRef props, SmallVectorImpl<Type> &out,
                          bool append) {
  typename Op::Adaptor a(operands, attrs, props);
  int32_t rank = a.getRank();
  Type in = a.getInput().getType(), e = a.getElement().getType();
  return infer(loc, out, [&]() -> Type {
    if (rank <= 0) return {};
    return llvm::TypeSwitch<Type, Type>(in)
        .Case<IntTupleType, ShapeType, StrideType, CoordType, LayoutType,
              ComposedLayoutType>([&](auto ty) -> Type {
          using T = decltype(ty);
          // A composed layout is padded with plain layouts.
          using E = std::conditional_t<std::is_same_v<T, ComposedLayoutType>, LayoutType, T>;
          auto el = llvm::dyn_cast<E>(e);
          if (!el) return {};
          auto r = append ? cg::append_to_rank_N(rank, ty.getRef(), el.getRef())
                          : cg::prepend_to_rank_N(rank, ty.getRef(), el.getRef());
          if (!cg::is_valid(r)) return {};
          return T::get(ctx, std::move(r));
        })
        .Default([](Type) -> Type { return {}; });
  });
}
}  // namespace

LogicalResult AppendToRankOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  return inferToRank<AppendToRankOp>(ctx, loc, operands, attrs, props, out, true);
}

LogicalResult PrependToRankOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  return inferToRank<PrependToRankOp>(ctx, loc, operands, attrs, props, out, false);
}

Type mlir::cutlass_compiler::cute::assumeResultType(Type src) {
  auto constrained = llvm::dyn_cast<ConstrainedIntType>(src);
  for (unsigned width : {64u, 32u})
    if (src.isInteger(width) || (constrained && constrained.getWidth() == width))
      return ConstrainedIntType::get(src.getContext(), 1, width, false);
  return {};
}

LogicalResult CeilDivOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  Type in = a.getInput().getType(), tiler = a.getTiler().getType();
  return infer(loc, out, [&]() -> Type {
    // cutegen divides shapes; int tuples go through shape and back.
    cg::shape shape;
    bool tuple = false;
    if (auto s = llvm::dyn_cast<ShapeType>(in)) {
      shape = s.getRef();
    } else if (auto t = llvm::dyn_cast<IntTupleType>(in)) {
      shape = cg::rec_var_cast<cg::shape>(t.getRef());
      tuple = true;
    } else {
      return {};
    }
    cg::shape r;
    if (auto t = llvm::dyn_cast<TileType>(tiler)) {
      r = cg::ceil_div(shape, t.getRef());
    } else if (auto t = llvm::dyn_cast<ShapeType>(tiler)) {
      r = cg::ceil_div(shape, t.getRef());
    } else if (auto t = llvm::dyn_cast<IntTupleType>(tiler)) {
      r = cg::ceil_div(shape, cg::rec_var_cast<cg::shape>(t.getRef()));
    } else {
      return {};
    }
    if (!cg::is_valid(r)) return {};
    if (tuple) return IntTupleType::get(ctx, cg::rec_var_cast<cg::int_tuple>(r));
    return ShapeType::get(ctx, std::move(r));
  });
}

LogicalResult CoalesceOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getInput().getType());
  if (!c) return emitOptionalError(loc, "input carries no layout");
  CoordType profile;
  if (a.getTargetProfile())
    profile = llvm::dyn_cast<CoordType>(a.getTargetProfile().getType());
  return infer(loc, out, [&]() -> Type {
    return rebuild(ctx, *c, mapLayout(ctx, c->layout, [&](const auto &l) {
      return profile ? cg::coalesce(l, profile.getRef()) : cg::coalesce(l);
    }));
  });
}

LogicalResult ComposedGetOuterOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto c = llvm::dyn_cast<ComposedLayoutType>(operands[0].getType());
    if (!c) return {};
    return LayoutType::get(ctx, c.getRef().layout_b());
  });
}

LogicalResult CompositionOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getLhs().getType());
  if (!c) return emitOptionalError(loc, "lhs carries no layout");
  Type rhs = a.getRhs().getType();
  return infer(loc, out, [&]() -> Type {
    return rebuild(ctx, *c, llvm::TypeSwitch<Type, Type>(rhs)
        .Case<LayoutType, ShapeType, TileType>([&](auto r) {
          return mapLayout(ctx, c->layout, [&](const auto &l) {
            return cg::composition(l, r.getRef());
          });
        })
        .Default([](Type) -> Type { return {}; }));
  });
}

LogicalResult Crd2IdxOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  if (!coord) return emitOptionalError(loc, "coord must be a coord");
  return infer(loc, out, [&]() -> Type {
    auto c = zeroUnderscores(coord.getRef());
    if (!c) return {};
    Type layout = a.getLayout().getType();
    cg::int_tuple idx;
    if (auto l = llvm::dyn_cast<LayoutType>(layout)) {
      idx = cg::layout_eval(*c, l.getRef());
    } else if (auto s = llvm::dyn_cast<ShapeType>(layout)) {
      idx = cg::crd2idx(*c, s.getRef());
    } else {
      return {};
    }
    if (!cg::is_valid(idx)) return {};
    return IntTupleType::get(ctx, std::move(idx));
  });
}

LogicalResult DereferenceArithTupleIteratorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto it = llvm::dyn_cast<ArithTupleIteratorType>(operands[0].getType());
    return it ? it.getArithTuple() : Type{};
  });
}

LogicalResult DiceOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  const cg::coord &coord = a.getCoordAttr().getRef();
  Type in = a.getInput().getType();
  return infer(loc, out, [&]() -> Type {
    if (auto c = carrierOf(in)) {
      return rebuild(ctx, *c, mapLayout(ctx, c->layout, [&](const auto &l) {
        return cg::dice(coord, l);
      }));
    }
    return llvm::TypeSwitch<Type, Type>(in)
        .Case<IntTupleType, ShapeType, StrideType, CoordType>([&](auto ty) -> Type {
          auto r = cg::dice(coord, ty.getRef());
          if (!cg::is_valid(r)) return {};
          return decltype(ty)::get(ctx, std::move(r));
        })
        .Default([](Type) -> Type { return {}; });
  });
}

Type mlir::cutlass_compiler::cute::fastDivmodComputeResultType(Type divisor) {
  auto d = llvm::dyn_cast<FastDivmodDivisorType>(divisor);
  return d ? IntegerType::get(divisor.getContext(), d.getWidth()) : Type();
}

Type mlir::cutlass_compiler::cute::fastDivmodCreateDivisorResultType(Type divisor) {
  auto i = llvm::dyn_cast<IntegerType>(divisor);
  return i ? FastDivmodDivisorType::get(divisor.getContext(), i.getWidth(), false) : Type();
}

LogicalResult FilterZerosOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getInput().getType());
  if (!c) return emitOptionalError(loc, "input carries no layout");
  return infer(loc, out, [&]() -> Type {
    return rebuild(ctx, *c, mapLayout(ctx, c->layout, [&](const auto &l) {
      return cg::filter_zeros(l);
    }));
  });
}

namespace {
// zeroUnitModes on all but the last `keep` modes (of mode 1 when `zipped`):
// the modes a divide leaves untouched keep their strides.
cg::layout zeroUnitModesHead(const cg::layout &l, size_t keep, bool zipped) {
  if (!keep || !cg::holds_vector(l.shape())) return zeroUnitModes(l);
  if (zipped) {
    if (cg::rank(l.shape()) != 2) return zeroUnitModes(l);
    return cg::make_layout(std::vector<cg::layout>{
        zeroUnitModes(cg::get(l, 0)), zeroUnitModesHead(cg::get(l, 1), keep, false)});
  }
  size_t n = cg::rank(l.shape());
  if (keep > n) return l;
  std::vector<cg::layout> modes;
  for (size_t i = 0; i < n; ++i)
    modes.push_back(i < n - keep ? zeroUnitModes(cg::get(l, i)) : cg::get(l, i));
  return cg::make_layout(modes);
}
cg::composed_layout zeroUnitModesHead(const cg::composed_layout &l, size_t keep, bool zipped) {
  cg::layout b = zeroUnitModesHead(l.layout_b(), keep, zipped);
  if (l.is_a_swizzle()) return cg::composed_layout(l.swizzle_a(), l.offset(), b);
  return cg::composed_layout(l.layout_a(), l.offset(), b);
}

template <class Op, class F>
LogicalResult inferDivide(MLIRContext *ctx, std::optional<Location> loc,
                          ValueRange operands, DictionaryAttr attrs,
                          PropertyRef props, SmallVectorImpl<Type> &out,
                          F &&divide, bool zipped = false) {
  typename Op::Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getInput().getType());
  if (!c) return emitOptionalError(loc, "input carries no layout");
  Type tiler = a.getTiler().getType();
  return infer(loc, out, [&]() -> Type {
    return rebuild(ctx, *c, llvm::TypeSwitch<Type, Type>(tiler)
        .template Case<TileType, LayoutType, ShapeType>([&](auto t) {
          // Unit modes the tiler passes through (`_` or extent 1) keep their
          // nonzero strides, which zeroUnitModes does not model: not covered.
          if constexpr (std::is_same_v<decltype(t), TileType>) {
            auto outer = outerLayout(c->layout);
            for (size_t i = 0; outer && cg::holds_vector(t.getRef()) && i < cg::rank(t.getRef()) &&
                               cg::holds_vector(outer->shape()) && i < cg::rank(outer->shape());
                 ++i) {
              std::string m = cg::to_string(t.getRef()[i]);
              auto tl = cg::from_string<cg::layout>(m);
              bool pass = m == "_" || (tl && cg::holds_int(cg::size(*tl)) &&
                                       cg::size(*tl).as_int64() == 1 && cg::to_string(tl->stride()) != "0");
              std::vector<std::string> shapes, strides;
              leavesOf(outer->shape()[i], shapes);
              leavesOf(outer->stride()[i], strides);
              for (size_t k = 0; pass && k < shapes.size() && k < strides.size(); ++k)
                if (shapes[k] == "1" && strides[k] != "0") return Type();
            }
          }
          return mapLayout(ctx, c->layout, [&](const auto &l) {
            // A tile or shape tiler leaves the input's trailing modes alone.
            size_t keep = 0;
            if constexpr (!std::is_same_v<decltype(t), LayoutType>) {
              size_t in = cg::rank(outerOf(l).shape()), r = cg::rank(t.getRef());
              if (cg::holds_vector(outerOf(l).shape()) && r < in) keep = in - r;
            }
            return zeroUnitModesHead(divide(l, t.getRef()), keep, zipped);
          });
        })
        .Default([](Type) -> Type { return {}; }));
  });
}
}  // namespace

LogicalResult FlatDivideOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  return inferDivide<FlatDivideOp>(ctx, loc, operands, attrs, props, out,
      [](const auto &l, const auto &t) { return cg::flat_divide(l, t); });
}

LogicalResult LogicalDivideOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  return inferDivide<LogicalDivideOp>(ctx, loc, operands, attrs, props, out,
      [](const auto &l, const auto &t) { return cg::logical_divide(l, t); });
}

LogicalResult TiledDivideOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  return inferDivide<TiledDivideOp>(ctx, loc, operands, attrs, props, out,
      [](const auto &l, const auto &t) { return cg::tiled_divide(l, t); });
}

LogicalResult ZippedDivideOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  return inferDivide<ZippedDivideOp>(ctx, loc, operands, attrs, props, out,
      [](const auto &l, const auto &t) { return cg::zipped_divide(l, t); }, true);
}

Type mlir::cutlass_compiler::cute::getResultType(Type in, ArrayRef<int32_t> mode) {
  MLIRContext *ctx = in.getContext();
  return llvm::TypeSwitch<Type, Type>(in)
      .Case<IntTupleType, ShapeType, StrideType, CoordType, TileType,
            LayoutType, ComposedLayoutType>([&](auto ty) -> Type {
        auto r = cg::get(ty.getRef(), mode);
        if (!cg::is_valid(r)) return {};
        if constexpr (std::is_same_v<decltype(ty), ComposedLayoutType>)
          return layoutType(ctx, std::move(r));
        else
          return decltype(ty)::get(ctx, std::move(r));
      })
      .Default([](Type) -> Type { return {}; });
}

LogicalResult GetIterOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    Type src = operands[0].getType();
    if (auto m = llvm::dyn_cast<CuteMemRefType>(src)) return m.getPtr();
    if (auto c = llvm::dyn_cast<CoordTensorType>(src))
      return ArithTupleIteratorType::get(ctx, c.getArithTuple());
    if (auto v = llvm::dyn_cast<cute_nvgpu::SmemDescViewType>(src)) return v.getDesc();
    return {};
  });
}

LogicalResult GetLayoutOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto c = carrierOf(operands[0].getType());
    return c ? c->layout : Type{};
  });
}

LogicalResult GetLeavesOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return llvm::TypeSwitch<Type, LogicalResult>(operands[0].getType())
      .Case<ShapeType, StrideType, CoordType, IntTupleType>(
          [&](auto ty) -> LogicalResult {
            auto flat = cg::flatten(ty.getRef());
            for (size_t i = 0; i < cg::rank(flat); ++i)
              out.push_back(decltype(ty)::get(ctx, flat[i]));
            return success();
          })
      .Case<TileType>([&](TileType ty) -> LogicalResult {
        // A tile's leaves: its layouts, and `_` as a tile.
        auto flat = cg::flatten(ty.getRef());
        for (size_t i = 0; i < cg::rank(flat); ++i) {
          flat[i].visit([&](const auto &e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (cg::is_layout<T>::value)
              out.push_back(LayoutType::get(ctx, e));
            else if constexpr (!std::is_same_v<T, std::vector<cg::tile>>)
              out.push_back(TileType::get(ctx, flat[i]));
          });
        }
        return success();
      })
      .Default([&](Type t) -> LogicalResult {
        return emitOptionalError(loc, "cannot take the leaves of ", t);
      });
}

LogicalResult GetScalarsOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  bool onlyDynamic = a.getOnlyDynamic();
  return llvm::TypeSwitch<Type, LogicalResult>(a.getCuteValue().getType())
      .Case<IntTupleType, ShapeType, StrideType, CoordType, TileType,
            LayoutType, ComposedLayoutType>([&](auto ty) -> LogicalResult {
        SmallVector<Type> scalars;
        cg::collect_scalar_types(ctx, ty.getRef(), scalars, onlyDynamic);
        out.append(scalars.begin(), scalars.end());
        return success();
      })
      .Default([&](Type t) -> LogicalResult {
        return emitOptionalError(loc, "cannot take the scalars of ", t);
      });
}

LogicalResult GetShapeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    Type in = operands[0].getType();
    if (auto t = llvm::dyn_cast<TileType>(in))
      return ShapeType::get(ctx, cg::extract_shape_from_tile(t.getRef()));
    auto c = carrierOf(in);
    if (!c) return {};
    if (auto l = llvm::dyn_cast<LayoutType>(c->layout))
      return ShapeType::get(ctx, l.getRef().shape());
    if (auto l = llvm::dyn_cast<ComposedLayoutType>(c->layout))
      return ShapeType::get(ctx, l.getRef().shape());
    return {};
  });
}

LogicalResult GroupModesOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getInput().getType());
  if (!c) return emitOptionalError(loc, "input carries no layout");
  int begin = static_cast<int>(a.getBegin()), end = static_cast<int>(a.getEnd());
  return infer(loc, out, [&]() -> Type {
    return rebuild(ctx, *c, mapLayout(ctx, c->layout, [&](const auto &l) {
      return cg::group(begin, end, l);
    }));
  });
}

LogicalResult LocalTileOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getInput().getType());
  auto tile = llvm::dyn_cast<TileType>(a.getTile().getType());
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  if (!c || !tile || !coord)
    return emitOptionalError(loc, "expects a layout carrier, a tile and a coord");
  return infer(loc, out, [&]() -> Type {
    // CuTe's inner_partition: divide by the tile, take the coordinate in the
    // rest modes, and flatten the two modes into one rank.
    cg::tile tiler = tile.getRef();
    cg::coord crd = coord.getRef();
    if (CoordAttr proj = a.getProjAttr()) {
      tiler = cg::dice(proj.getRef(), tiler);
      crd = cg::dice(proj.getRef(), crd);
    }
    auto layout = llvm::dyn_cast<LayoutType>(c->layout);
    if (!layout) return {};
    cg::layout zipped = cg::zipped_divide(layout.getRef(), tiler);
    if (!cg::is_valid(zipped)) return {};
    cg::layout tiles = cg::get(zipped, 0), rest = cg::get(zipped, 1);
    auto [sliced, offset] = cg::slice_and_offset(crd, rest);
    if (!cg::is_valid(sliced)) return {};
    // One rank: the tile modes, then what is left of the rest modes.
    std::vector<cg::layout> modes;
    for (size_t i = 0; i < cg::rank(tiles.shape()); ++i)
      modes.push_back(cg::get(tiles, i));
    if (cg::holds_leaf(sliced.shape())) {
      modes.push_back(sliced);
    } else {
      for (size_t i = 0; i < cg::rank(sliced.shape()); ++i)
        modes.push_back(cg::get(sliced, i));
    }
    // The input's modes beyond the tiler that the coordinate keeps are
    // untouched and keep their strides.
    size_t keep = 0, in = cg::rank(layout.getRef().shape()), r = cg::rank(tiler);
    for (size_t i = r; cg::holds_vector(crd) && i < in && i < cg::rank(crd); ++i)
      keep += cg::to_string(crd[i]) == "_";
    cg::layout flat = zeroUnitModesHead(cg::make_layout(modes), keep, false);
    Type result = layoutType(ctx, std::move(flat));
    if (c->ptr) return CuteMemRefType::get(ctx, movedPtr(c->ptr, offset), result);
    if (c->iter) {
      auto sum = cg::arith_tuple_sum<cg::int_tuple>(c->iter.getRef(), offset);
      if (!cg::is_valid(sum)) return {};
      return CoordTensorType::get(ctx, IntTupleType::get(ctx, std::move(sum)),
                                  result);
    }
    return result;
  });
}

Type mlir::cutlass_compiler::cute::makeArithTupleIterResultType(Type tuple) {
  auto t = llvm::dyn_cast<IntTupleType>(tuple);
  return t ? ArithTupleIteratorType::get(tuple.getContext(), t) : Type();
}

LogicalResult MakeComposedLayoutOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto offset = llvm::dyn_cast<IntTupleType>(a.getOffset().getType());
  auto outer = llvm::dyn_cast<LayoutType>(a.getOuter().getType());
  if (!offset || !outer)
    return emitOptionalError(loc, "expects an int tuple offset and a layout");
  return infer(loc, out, [&]() -> Type {
    Type inner = a.getInner().getType();
    if (auto l = llvm::dyn_cast<LayoutType>(inner))
      return ComposedLayoutType::get(ctx, cg::composed_layout(l.getRef(), offset.getRef(), outer.getRef()));
    if (auto s = llvm::dyn_cast<SwizzleType>(inner))
      return ComposedLayoutType::get(ctx, cg::composed_layout(s.getRef(), offset.getRef(), outer.getRef()));
    return {};
  });
}

LogicalResult MakeIdentityLayoutOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto s = llvm::dyn_cast<ShapeType>(operands[0].getType());
    if (!s) return {};
    cg::layout l = cg::make_identity_layout(s.getRef());
    return cg::is_valid(l) ? LayoutType::get(ctx, std::move(l)) : Type{};
  });
}

LogicalResult MakeOrderedLayoutOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto shape = llvm::dyn_cast<ShapeType>(a.getShape().getType());
  auto order = llvm::dyn_cast<IntTupleType>(a.getOrder().getType());
  if (!shape || !order) return emitOptionalError(loc, "expects a shape and an order");
  return infer(loc, out, [&]() -> Type {
    cg::layout l(shape.getRef(),
                 cg::compact_order<cg::stride>(shape.getRef(), order.getRef()));
    return cg::is_valid(l) ? LayoutType::get(ctx, std::move(l)) : Type{};
  });
}

Type mlir::cutlass_compiler::cute::makeViewResultType(Type iter, Type layout) {
  MLIRContext *ctx = iter.getContext();
  if (!llvm::isa<LayoutType, ComposedLayoutType>(layout)) return {};
  if (auto p = llvm::dyn_cast<PtrType>(iter)) return CuteMemRefType::get(ctx, p, layout);
  if (auto it = llvm::dyn_cast<ArithTupleIteratorType>(iter))
    return CoordTensorType::get(ctx, it.getArithTuple(), layout);
  auto plain = llvm::dyn_cast<LayoutType>(layout);
  if (plain && llvm::isa<cute_nvgpu::SmemDescType>(iter))
    return cute_nvgpu::SmemDescViewType::get(ctx, iter, plain.getAttr());
  return {};
}

LogicalResult MemRefLoadOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto m = llvm::dyn_cast<CuteMemRefType>(operands[0].getType());
    return m ? m.getValueType() : Type{};
  });
}

LogicalResult MemRefLoadVecOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  return infer(loc, out, [&]() -> Type {
    auto m = llvm::dyn_cast<CuteMemRefType>(a.getSrc().getType());
    if (!m) return {};
    auto l = llvm::dyn_cast<LayoutType>(m.getLayout());
    if (!l) return {};
    auto n = cg::size(l.getRef());
    if (!cg::holds_int(n)) return {};
    return VectorType::get({n.as_int64()}, m.getValueType());
  });
}

LogicalResult SizeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  ArrayRef<int32_t> mode;
  if (auto m = a.getModeAttr()) mode = m.asArrayRef();
  Type in = a.getInput().getType();
  if (auto c = carrierOf(in)) in = c->layout;
  return infer(loc, out, [&]() -> Type {
    return llvm::TypeSwitch<Type, Type>(in)
        .Case<ShapeType, IntTupleType, LayoutType, ComposedLayoutType>(
            [&](auto ty) -> Type {
              auto r = cg::rec_var_cast<cg::int_tuple>(
                  cg::size(cg::get(ty.getRef(), mode)));
              if (!cg::is_valid(r)) return {};
              return IntTupleType::get(ctx, std::move(r));
            })
        .Default([](Type) -> Type { return {}; });
  });
}

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  if (!coord) return emitOptionalError(loc, "coord must be a coord");
  Type in = a.getInput().getType();
  return infer(loc, out, [&]() -> Type {
    if (auto c = carrierOf(in)) return sliceCarrier(ctx, *c, coord.getRef());
    return llvm::TypeSwitch<Type, Type>(in)
        .Case<IntTupleType, ShapeType, StrideType, CoordType>([&](auto ty) -> Type {
          auto r = cg::slice(coord.getRef(), ty.getRef());
          if (!cg::is_valid(r)) return {};
          return decltype(ty)::get(ctx, std::move(r));
        })
        .Default([](Type) -> Type { return {}; });
  });
}

LogicalResult TileToShapeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto shape = llvm::dyn_cast<ShapeType>(a.getTrgShape().getType());
  auto order = llvm::dyn_cast<IntTupleType>(a.getOrdShape().getType());
  if (!shape || !order) return emitOptionalError(loc, "expects a shape and an order");
  return infer(loc, out, [&]() -> Type {
    return mapLayout(ctx, a.getBlock().getType(), [&](const auto &l) {
      return cg::tile_to_shape(l, shape.getRef(), order.getRef());
    });
  });
}

LogicalResult ToIntTupleOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    return llvm::TypeSwitch<Type, Type>(operands[0].getType())
        .Case<ShapeType, StrideType, CoordType, IntTupleType>([&](auto ty) -> Type {
          return IntTupleType::get(ctx, cg::rec_var_cast<cg::int_tuple>(ty.getRef()));
        })
        .Default([](Type) -> Type { return {}; });
  });
}

namespace {
template <class F>
LogicalResult inferTupleArith(MLIRContext *ctx, std::optional<Location> loc,
                              ValueRange operands, SmallVectorImpl<Type> &out,
                              F &&f) {
  return infer(loc, out, [&]() -> Type {
    return llvm::TypeSwitch<Type, Type>(operands[0].getType())
        .Case<IntTupleType, ShapeType>([&](auto lhs) -> Type {
          using T = decltype(lhs);
          auto rhs = llvm::dyn_cast<T>(operands[1].getType());
          if (!rhs) return {};
          auto r = f.template operator()<typename T::algebra_t>(lhs.getRef(), rhs.getRef());
          if (!cg::is_valid(r)) return {};
          return T::get(ctx, std::move(r));
        })
        .Default([](Type) -> Type { return {}; });
  });
}
}  // namespace

LogicalResult TupleAddOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return inferTupleArith(ctx, loc, operands, out, []<class T>(const auto &a, const auto &b) {
    return cg::arith_tuple_sum<T>(a, b);
  });
}

LogicalResult TupleMulOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return inferTupleArith(ctx, loc, operands, out, []<class T>(const auto &a, const auto &b) {
    return cg::arith_tuple_mul<T>(a, b);
  });
}

LogicalResult TupleDivOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return inferTupleArith(ctx, loc, operands, out, []<class T>(const auto &a, const auto &b) {
    return cg::arith_tuple_div<T>(a, b);
  });
}

// Without a stride the layout is compact column-major, unit modes at 0.
Type mlir::cutlass_compiler::cute::makeLayoutResultType(Type shape, Type stride) {
  auto s = llvm::dyn_cast<ShapeType>(shape);
  if (!s) return {};
  MLIRContext *ctx = shape.getContext();
  if (stride) {
    auto st = llvm::dyn_cast<StrideType>(stride);
    if (!st) return {};
    return LayoutType::get(ctx, cg::layout(s.getRef(), st.getRef()));
  }
  cg::layout l(s.getRef(), cg::compact_col_major<cg::shape, cg::stride>(s.getRef()));
  return LayoutType::get(ctx, zeroUnitModes(l));
}

namespace {
// `MxNxK` and `num_cta` of an SM100 block-scaled or plain mma atom.
struct MmaAtom {
  int64_t m = 0, n = 0, k = 0, ctas = 1;
  Type c;              // accumulator element
  bool aTmem = false;  // frag_kind = ts
};
std::optional<MmaAtom> mmaAtomOf(Type tiledMma) {
  auto tiled = llvm::dyn_cast<TiledMmaType>(tiledMma);
  if (!tiled) return std::nullopt;
  // One atom, not permuted: other tilings are not covered here.
  auto n = cg::size(tiled.getAtomLayout_MNK().getRef());
  TileAttr perm = tiled.getPermutation_MNK();
  if (!cg::holds_int(n) || n.as_int64() != 1 ||
      (perm && cg::to_string(perm.getRef()) != "[_;_;_]"))
    return std::nullopt;
  MmaAtom atom;
  ShapeAttr shape;
  if (auto bs = llvm::dyn_cast<cute_nvgpu::MmaAtomSM100UMMABlockScaledType>(tiled.getMmaAtom())) {
    shape = bs.getShapeMnk();
    atom.ctas = bs.getNumCta();
    atom.c = bs.getCType();
    atom.aTmem = bs.getAFragKind() == cute_nvgpu::MmaFragKind::tmem;
  } else if (auto mma = llvm::dyn_cast<cute_nvgpu::MmaAtomSM100UMMAType>(tiled.getMmaAtom())) {
    shape = mma.getShapeMnk();
    atom.ctas = mma.getNumCta();
    atom.c = mma.getCType();
    atom.aTmem = mma.getAFragKind() == cute_nvgpu::MmaFragKind::tmem;
  } else {
    return std::nullopt;
  }
  if (cg::rank(shape.getRef()) != 3) return std::nullopt;
  int64_t *mnk[] = {&atom.m, &atom.n, &atom.k};
  for (int i = 0; i < 3; ++i) {
    auto e = cg::get(shape.getRef(), i);
    if (!cg::holds_int(e)) return std::nullopt;
    *mnk[i] = e.as_int64();
  }
  return atom;
}
}  // namespace

// partition_A/B/C of the trivial tiled mma: the leading two modes are divided
// by the per-CTA atom shape, the CTA's share of the first is taken at the
// thread coordinate, and what remains follows the atom modes.
LogicalResult TiledMmaPartitionOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto atom = mmaAtomOf(a.getTiledMma().getType());
  auto c = carrierOf(a.getInput().getType());
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  auto which = llvm::dyn_cast_or_null<IntegerAttr>(a.getOperandIdAttr());
  if (!atom || !c || !coord || !which)
    return emitOptionalError(loc, "expects a tiled mma, a tensor, a coord and an operand");
  return infer(loc, out, [&]() -> Type {
    auto layout = llvm::dyn_cast<LayoutType>(c->layout);
    if (!layout) return {};
    int64_t a0, a1;
    switch (which.getInt()) {
      case 0: a0 = atom->m / atom->ctas; a1 = atom->k; break;  // A: (M, K)
      case 1: a0 = atom->n / atom->ctas; a1 = atom->k; break;  // B: (N, K)
      case 2: a0 = atom->m / atom->ctas; a1 = atom->n; break;  // C: (M, N)
      default: return {};
    }
    const cg::layout &l = layout.getRef();
    if (cg::rank(l.shape()) < 2) return {};
    auto tile = cg::from_string<cg::tile>("[" + std::to_string(a0) + ":1;" +
                                         std::to_string(a1) + ":1]");
    if (!tile) return {};
    cg::layout two = cg::make_layout(std::vector<cg::layout>{cg::get(l, 0), cg::get(l, 1)});
    cg::layout divided = cg::zipped_divide(two, *tile);
    if (!cg::is_valid(divided)) return {};
    cg::layout atoms = cg::get(divided, 0), rest = cg::get(divided, 1);
    // The first rest mode holds the CTA index innermost.
    auto ctaTile = cg::from_string<cg::tile>("[" + std::to_string(atom->ctas) + ":1]");
    if (!ctaTile) return {};
    cg::layout rest0 = cg::zipped_divide(cg::get(rest, 0), *ctaTile);
    if (!cg::is_valid(rest0) || cg::rank(rest0.shape()) != 2) return {};
    // A lone CTA has nothing to select: its coordinate is 0 whatever the
    // thread index says.
    std::string which_cta = atom->ctas == 1 ? "0" : cg::to_string(coord.getRef());
    auto [rest0Sliced, offset] = cg::slice_and_offset(
        *cg::from_string<cg::coord>("(" + which_cta + ",_)"), rest0);
    if (!cg::is_valid(rest0Sliced)) return {};
    // What the slice leaves is one mode, not a tuple around it.
    while (cg::holds_vector(rest0Sliced.shape()) && cg::rank(rest0Sliced.shape()) == 1)
      rest0Sliced = cg::get(rest0Sliced, 0);
    std::vector<cg::layout> modes{atoms, rest0Sliced, cg::get(rest, 1)};
    for (size_t i = 2; i < cg::rank(l.shape()); ++i) modes.push_back(cg::get(l, i));
    // Modes beyond M and K (or N) are untouched.
    Type result = layoutType(ctx, zeroUnitModesHead(cg::make_layout(modes), modes.size() - 3, false));
    if (c->ptr) return CuteMemRefType::get(ctx, movedPtr(c->ptr, offset), result);
    if (c->iter) {
      auto sum = cg::arith_tuple_sum<cg::int_tuple>(c->iter.getRef(), offset);
      if (!cg::is_valid(sum)) return {};
      return CoordTensorType::get(ctx, IntTupleType::get(ctx, std::move(sum)), result);
    }
    return result;
  });
}

// The fragment an mma reads: for A and B a descriptor view over the staged
// smem tile, atom modes collapsed and strides in 16-byte units; for C the
// accumulator in tensor memory, one lane per row and one column per element.
LogicalResult MmaMakeFragmentOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto which = llvm::dyn_cast_or_null<IntegerAttr>(a.getOperandIdAttr());
  auto atom = mmaAtomOf(a.getAtom().getType());
  // A in tensor memory (frag_kind = ts) is not covered.
  if (!which || !atom || (which.getInt() == 0 && atom->aTmem))
    return emitOptionalError(loc, "expects an SM100 tiled mma and an operand id");
  Type in = a.getInput().getType();
  return infer(loc, out, [&]() -> Type {
    if (which.getInt() == 2) {
      auto shape = llvm::dyn_cast<ShapeType>(in);
      if (!shape) return {};
      const cg::shape &s = shape.getRef();
      if (cg::rank(s) < 1 || !cg::holds_vector(cg::get(s, 0))) return {};
      auto m = cg::size(cg::get(s, 0), 0), n = cg::size(cg::get(s, 0), 1);
      if (!cg::holds_int(m) || !cg::holds_int(n)) return {};
      // ((M,N), rest...): rows on lanes, columns contiguous, stages by N
      // columns; unit modes at 0. A 64-row share of a 2-CTA atom puts the
      // second half of the columns on lanes 64 and up.
      int64_t cols = n.as_int64();
      // Mode 0 is flattened to its (M,N) extents.
      std::string text = "((" + cg::to_string(m) + "," + cg::to_string(n) + ")",
                  stride = "((65536,1)";
      if (atom->m / atom->ctas == 64) {
        if (atom->ctas != 2 || m.as_int64() != 64 || cols % 2) return {};
        cols /= 2;
        text = "((64,(" + std::to_string(cols) + ",2))";
        stride = "((65536,(1,4194304))";
      } else if (atom->m / atom->ctas != 128) {
        return {};
      }
      for (size_t i = 1; i < cg::rank(s); ++i) {
        auto e = cg::size(s, i);
        text += "," + cg::to_string(cg::get(s, i));
        stride += "," + ((cg::holds_int(e) && e.as_int64() == 1)
                             ? std::string("0") : std::to_string(cols));
      }
      auto l = cg::from_string<cg::layout>(text + "):" + stride + ")");
      if (!l) return {};
      auto ptr = PtrType::get(ctx, atom->c, AddressSpace::tmem, 1, {}, {});
      return CuteMemRefType::get(ctx, ptr, LayoutType::get(ctx, *l));
    }
    auto m = llvm::dyn_cast<CuteMemRefType>(in);
    if (!m) return {};
    cg::layout l;
    if (auto plain = llvm::dyn_cast<LayoutType>(m.getLayout())) l = plain.getRef();
    else if (auto composed = llvm::dyn_cast<ComposedLayoutType>(m.getLayout())) l = composed.getRef().layout_b();
    else return {};
    // The atom mode collapses to one; the rest modes follow.
    int64_t bits = bitsOfType(m.getValueType());
    std::string shape = "(1", stride = "(0";
    for (size_t i = 1; i < cg::rank(l.shape()); ++i) {
      auto e = cg::size(l.shape(), i);
      auto st = cg::get(l.stride(), i);
      if (!cg::holds_int(e) || !cg::holds_int(st)) return {};
      shape += "," + std::to_string(e.as_int64());
      stride += "," + std::to_string(st.as_int64() * bits / 128);
    }
    auto view = cg::from_string<cg::layout>(shape + "):" + stride + ")");
    if (!view) return {};
    return cute_nvgpu::SmemDescViewType::get(
        ctx, cute_nvgpu::SmemDescType::get(ctx), LayoutAttr::get(ctx, *view));
  });
}

LogicalResult TupleSubOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return inferTupleArith(ctx, loc, operands, out, []<class T>(const auto &a, const auto &b) {
    return cg::arith_tuple_sub<T>(a, b);
  });
}

LogicalResult TupleModOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  if (failed(inferTupleArith(ctx, loc, operands, out, []<class T>(const auto &a, const auto &b) {
        return cg::arith_tuple_mod<T>(a, b);
      })))
    return failure();
  // A dynamic leaf modulo a static one: 0 when its divisibility is a
  // multiple, itself when a divisor, else unknown (cutegen drops this).
  std::vector<std::string> lhs, rhs, res;
  auto leaves = [](Type t, std::vector<std::string> &l) {
    if (auto i = llvm::dyn_cast<IntTupleType>(t)) leavesOf(i.getRef(), l);
    if (auto s = llvm::dyn_cast<ShapeType>(t)) leavesOf(s.getRef(), l);
  };
  leaves(operands[0].getType(), lhs);
  leaves(operands[1].getType(), rhs);
  leaves(out[0], res);
  if (lhs.size() != rhs.size() || lhs.size() != res.size()) return success();
  for (size_t i = 0; i < res.size(); ++i) {
    int64_t n, div = 1;
    if (lhs[i].front() != '?' || StringRef(rhs[i]).getAsInteger(10, n) || n <= 0) continue;
    size_t d = lhs[i].find("div=");
    if (d != std::string::npos) StringRef(lhs[i]).substr(d + 4).take_while(llvm::isDigit).getAsInteger(10, div);
    res[i] = div % n == 0 ? "0" : n % div == 0 ? lhs[i] : dynamicInt(1, lhs[i].find("i64") != std::string::npos);
  }
  size_t i = 0;
  Type t = out[0];
  std::string text;
  if (auto it = llvm::dyn_cast<IntTupleType>(t)) text = nestLike(it.getRef(), res, i);
  else text = nestLike(llvm::cast<ShapeType>(t).getRef(), res, i);
  if (llvm::isa<IntTupleType>(t)) {
    if (auto r = cg::from_string<cg::int_tuple>(text)) out[0] = IntTupleType::get(ctx, *r);
  } else if (auto r = cg::from_string<cg::shape>(text)) {
    out[0] = ShapeType::get(ctx, *r);
  }
  return success();
}

// ---------------------------------------------------------------- more cute

namespace {

template <class Op, class F>
LogicalResult inferProduct(MLIRContext *ctx, std::optional<Location> loc,
                           ValueRange operands, DictionaryAttr attrs,
                           PropertyRef props, SmallVectorImpl<Type> &out, F &&f) {
  typename Op::Adaptor a(operands, attrs, props);
  auto tiler = llvm::dyn_cast<LayoutType>(a.getTiler().getType());
  if (!tiler || !cg::is_int_or_dynamic_int_only(tiler.getRef().stride()))
    return emitOptionalError(loc, "expects an integer-strided layout tiler");
  Type in = a.getInput().getType();
  return infer(loc, out, [&]() -> Type {
    auto outer = outerLayout(in);
    if (!outer || !cg::is_static(*outer) ||
        (!cg::is_int_or_dynamic_int_only(outer->stride()) && !cg::is_static(tiler.getRef())))
      return {};
    return mapLayout(ctx, in, [&](const auto &l) { return f(l, tiler.getRef()); });
  });
}

// A coordinate per leaf of `shape`: 0 where the extent is 1, else dynamic.
std::optional<cg::coord> dynamicCoordLike(const cg::shape &shape) {
  std::vector<std::string> leaves;
  leavesOf(shape, leaves);
  for (auto &leaf : leaves) leaf = isOne(leaf) ? "0" : "?";
  size_t i = 0;
  return cg::from_string<cg::coord>(nestLike(shape, leaves, i));
}

// Moves a carrier's iterator by `offset` and gives it `layout`.
Type moved(MLIRContext *ctx, const Carrier &c, const cg::int_tuple &offset, Type layout) {
  if (c.ptr) return CuteMemRefType::get(ctx, movedPtr(c.ptr, offset), layout);
  if (c.iter) {
    auto sum = cg::arith_tuple_sum<cg::int_tuple>(c.iter.getRef(), offset);
    if (!cg::is_valid(sum)) return {};
    return CoordTensorType::get(ctx, IntTupleType::get(ctx, std::move(sum)), layout);
  }
  return rebuild(ctx, c, layout);
}

}  // namespace

#define CUTE_PRODUCT(Op, fn)                                                   \
  LogicalResult Op::inferReturnTypes(                                          \
      MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,      \
      DictionaryAttr attrs, PropertyRef props, RegionRange,                    \
      SmallVectorImpl<Type> &out) {                                            \
    return inferProduct<Op>(ctx, loc, operands, attrs, props, out,            \
                            [](const auto &l, const auto &t) { return cg::fn(l, t); }); \
  }
CUTE_PRODUCT(LogicalProductOp, logical_product)
CUTE_PRODUCT(ZippedProductOp, zipped_product)
CUTE_PRODUCT(TiledProductOp, tiled_product)
CUTE_PRODUCT(FlatProductOp, flat_product)
CUTE_PRODUCT(BlockedProductOp, blocked_product)
#undef CUTE_PRODUCT

// Unlike the other products, the raked one leaves unit modes at stride 0.
LogicalResult RakedProductOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange, SmallVectorImpl<Type> &out) {
  return inferProduct<RakedProductOp>(ctx, loc, operands, attrs, props, out,
      [](const auto &l, const auto &t) { return zeroUnitModes(cg::raked_product(l, t)); });
}

// The pointer's type is kept: the compiler records no swizzle on it.
LogicalResult ApplySwizzleOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  out.push_back(operands[0].getType());
  return success();
}

LogicalResult ComplementOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto in = llvm::dyn_cast<LayoutType>(a.getInput().getType());
  Type co = a.getCotarget().getType();
  return infer(loc, out, [&]() -> Type {
    if (!in) return {};
    const cg::layout &l = in.getRef();
    if (!cg::is_int_or_dynamic_int_only(l.stride()) ||
        (!cg::is_static(l.stride()) && cg::rank(l.shape()) > 1))
      return {};
    cg::shape shape;
    if (auto s = llvm::dyn_cast<ShapeType>(co)) shape = s.getRef();
    else if (auto c = llvm::dyn_cast<LayoutType>(co)) shape = c.getRef().shape();
    else return {};
    cg::layout r = cg::complement(l, shape);
    return cg::is_valid(r) ? LayoutType::get(ctx, std::move(r)) : Type();
  });
}

LogicalResult ComposedGetInnerOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto c = llvm::dyn_cast<ComposedLayoutType>(operands[0].getType());
    if (!c) return {};
    if (c.getRef().is_a_swizzle()) return SwizzleType::get(ctx, c.getRef().swizzle_a());
    return LayoutType::get(ctx, c.getRef().layout_a());
  });
}

LogicalResult ComposedGetOffsetOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto c = llvm::dyn_cast<ComposedLayoutType>(operands[0].getType());
    return c ? IntTupleType::get(ctx, c.getRef().offset()) : Type();
  });
}

LogicalResult CosizeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  ArrayRef<int32_t> mode = a.getMode();
  auto c = carrierOf(a.getInput().getType());
  return infer(loc, out, [&]() -> Type {
    if (!c) return {};
    cg::int_tuple r;
    if (auto l = llvm::dyn_cast<LayoutType>(c->layout)) {
      r = cg::rec_var_cast<cg::int_tuple>(cg::cosize(cg::get(l.getRef(), mode)));
    } else {
      // An affine inner layout has no cosize.
      auto cl = llvm::cast<ComposedLayoutType>(c->layout);
      if (!cl.getRef().is_a_swizzle()) return {};
      r = cg::rec_var_cast<cg::int_tuple>(cg::cosize(cg::get(cl.getRef(), mode)));
    }
    return cg::is_valid(r) ? IntTupleType::get(ctx, std::move(r)) : Type();
  });
}

// The multiplier and the two shifts of the divisor's magic numbers.
LogicalResult FastDivmodGetAuxOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  Type i = fastDivmodComputeResultType(operands[0].getType());
  if (!i) return emitOptionalError(loc, "expects a fast divmod divisor");
  out.append({i, IntegerType::get(ctx, 8), IntegerType::get(ctx, 8)});
  return success();
}

LogicalResult FastDivmodGetDivisorOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&] { return fastDivmodComputeResultType(operands[0].getType()); });
}

LogicalResult FilterOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  auto c = carrierOf(operands[0].getType());
  if (!c) return emitOptionalError(loc, "input carries no layout");
  return infer(loc, out, [&]() -> Type {
    Type layout;
    if (auto l = llvm::dyn_cast<LayoutType>(c->layout)) {
      layout = layoutType(ctx, cg::filter(l.getRef()));
    } else {
      // The outer layout is filtered; the inner function is kept.
      const cg::composed_layout &cl = llvm::cast<ComposedLayoutType>(c->layout).getRef();
      cg::layout f = cg::filter(cl.layout_b());
      if (!cg::is_valid(f)) return {};
      layout = layoutType(ctx, cl.is_a_swizzle() ? cg::composed_layout(cl.swizzle_a(), cl.offset(), f)
                                                 : cg::composed_layout(cl.layout_a(), cl.offset(), f));
    }
    return rebuild(ctx, *c, layout);
  });
}

namespace {
// Layout::get_hier_coord: each leaf's coordinate is (index / stride) % shape.
// A constant index gives static coordinates where the leaf is static; any
// other index gives dynamic ones, 0 on leaves of extent 1.
std::optional<std::vector<std::string>> hierCoord(Value index, const cg::layout &l) {
  std::vector<std::string> shapes, strides;
  leavesOf(l.shape(), shapes);
  leavesOf(l.stride(), strides);
  if (shapes.size() != strides.size()) return std::nullopt;
  APInt value;
  bool constant = matchPattern(index, m_ConstantInt(&value));
  std::vector<std::string> crd;
  for (size_t i = 0; i < shapes.size(); ++i) {
    int64_t n = 0, d = 0;
    bool staticShape = !StringRef(shapes[i]).getAsInteger(10, n);
    if (staticShape && n == 1) {
      crd.push_back("0");
    } else if (!constant) {
      crd.push_back("?");
    } else if (StringRef(strides[i]).getAsInteger(10, d) || d <= 0) {
      return std::nullopt;  // not recovered for dynamic or zero strides
    } else {
      int64_t q = value.getSExtValue() / d;
      crd.push_back(staticShape ? std::to_string(q % n) : "?");
    }
  }
  return crd;
}

// Column-major index of static coordinates in `shape`, or "?".
std::string colMajor(const std::vector<std::string> &crd, const std::vector<std::string> &shapes,
                     size_t begin, size_t end) {
  int64_t idx = 0, scale = 1;
  for (size_t i = begin; i < end; ++i) {
    int64_t c, n;
    if (StringRef(crd[i]).getAsInteger(10, c)) return "?";
    idx += c * scale;
    if (i + 1 < end && StringRef(shapes[i]).getAsInteger(10, n)) return "?";
    scale *= n;
  }
  return std::to_string(idx);
}
}  // namespace

LogicalResult GetHierCoordOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto layout = llvm::dyn_cast<LayoutType>(a.getInput().getType());
  return infer(loc, out, [&]() -> Type {
    auto crd = layout ? hierCoord(a.getIndex(), layout.getRef()) : std::nullopt;
    if (!crd) return {};
    size_t i = 0;
    auto c = cg::from_string<cg::coord>(nestLike(layout.getRef().shape(), *crd, i));
    return c ? CoordType::get(ctx, std::move(*c)) : Type();
  });
}

// One coordinate per mode: the mode's hierarchical coordinate, column-major.
LogicalResult GetFlatCoordOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto layout = llvm::dyn_cast<LayoutType>(a.getInput().getType());
  return infer(loc, out, [&]() -> Type {
    auto crd = layout ? hierCoord(a.getIndex(), layout.getRef()) : std::nullopt;
    if (!crd) return {};
    const cg::shape &shape = layout.getRef().shape();
    std::vector<std::string> shapes, modes;
    leavesOf(shape, shapes);
    if (!cg::holds_vector(shape)) {
      modes.push_back(colMajor(*crd, shapes, 0, shapes.size()));
    } else {
      size_t begin = 0;
      for (size_t m = 0; m < cg::rank(shape); ++m) {
        std::vector<std::string> sub;
        leavesOf(shape[m], sub);
        modes.push_back(colMajor(*crd, shapes, begin, begin + sub.size()));
        begin += sub.size();
      }
    }
    auto c = cg::from_string<cg::coord>(cg::holds_vector(shape) ? "(" + llvm::join(modes, ",") + ")"
                                                                : modes[0]);
    return c ? CoordType::get(ctx, std::move(*c)) : Type();
  });
}

LogicalResult GetIntegralCoordOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto layout = llvm::dyn_cast<LayoutType>(a.getInput().getType());
  return infer(loc, out, [&]() -> Type {
    auto crd = layout ? hierCoord(a.getIndex(), layout.getRef()) : std::nullopt;
    if (!crd) return {};
    std::vector<std::string> shapes;
    leavesOf(layout.getRef().shape(), shapes);
    auto c = cg::from_string<cg::coord>(colMajor(*crd, shapes, 0, shapes.size()));
    return c ? CoordType::get(ctx, std::move(*c)) : Type();
  });
}

LogicalResult GetLayoutsFromTileOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  auto tile = llvm::dyn_cast<TileType>(operands[0].getType());
  if (!tile) return emitOptionalError(loc, "expects a tile");
  auto flat = cg::flatten(tile.getRef());
  for (size_t i = 0; i < cg::rank(flat); ++i) {
    flat[i].visit([&](const auto &e) {
      if constexpr (cg::is_layout<std::decay_t<decltype(e)>>::value)
        out.push_back(LayoutType::get(ctx, e));
    });
  }
  if (out.empty()) return emitOptionalError(loc, "the tile holds no layout");
  return success();
}

LogicalResult GetStrideOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto l = llvm::dyn_cast<LayoutType>(operands[0].getType());
    return l ? StrideType::get(ctx, l.getRef().stride()) : Type();
  });
}

LogicalResult Idx2CrdOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto index = llvm::dyn_cast<IntTupleType>(a.getIndex().getType());
  auto shape = llvm::dyn_cast<ShapeType>(a.getShape().getType());
  return infer(loc, out, [&]() -> Type {
    if (!index || !shape) return {};
    size_t r = cg::rank(index.getRef());
    if (r > 1 && r != cg::rank(shape.getRef())) return {};
    if (auto split = cg::holds_dynamic_int(index.getRef())
                         ? splitIndex(index.getRef(), shape.getRef()) : std::nullopt) {
      auto c = cg::from_string<cg::coord>(*split);
      return c ? CoordType::get(ctx, std::move(*c)) : Type();
    }
    auto c = cg::rec_var_cast<cg::coord>(cg::idx2crd(index.getRef(), shape.getRef()));
    return cg::is_valid(c) ? CoordType::get(ctx, std::move(c)) : Type();
  });
}

LogicalResult IncrementCoordOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  auto shape = llvm::dyn_cast<ShapeType>(a.getShape().getType());
  return infer(loc, out, [&]() -> Type {
    if (!coord || !shape || !cg::is_congruent(coord.getRef(), shape.getRef()) ||
        cg::has_underscore(coord.getRef()))
      return {};
    auto c = cg::increment_coord(coord.getRef(), shape.getRef());
    return cg::is_valid(c) ? CoordType::get(ctx, std::move(c)) : Type();
  });
}

namespace {
template <class F>
LogicalResult inferInverse(MLIRContext *ctx, std::optional<Location> loc,
                           Type input, bool staticStride, SmallVectorImpl<Type> &out,
                           F &&f) {
  return infer(loc, out, [&]() -> Type {
    auto l = llvm::dyn_cast<LayoutType>(input);
    if (!l || !cg::is_static(l.getRef().shape()) ||
        (staticStride && !cg::is_static(l.getRef().stride())))
      return {};
    cg::layout r = f(l.getRef());
    return cg::is_valid(r) ? LayoutType::get(ctx, std::move(r)) : Type();
  });
}
}  // namespace

LogicalResult LeftInverseOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return inferInverse(ctx, loc, operands[0].getType(), true, out,
                      [](const cg::layout &l) { return cg::left_inverse(l); });
}

LogicalResult RightInverseOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return inferInverse(ctx, loc, operands[0].getType(), false, out,
                      [](const cg::layout &l) { return cg::right_inverse(l); });
}

LogicalResult LoadScaledIndexOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  out.push_back(operands[0].getType());
  return success();
}

// CuTe's local_partition: the tensor divided by the tiler's shape, sliced at
// the thread's (dynamic) coordinate in the tile.
LogicalResult LocalPartitionOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto c = carrierOf(a.getInput().getType());
  auto tiler = llvm::dyn_cast<LayoutType>(a.getTiler().getType());
  return infer(loc, out, [&]() -> Type {
    auto layout = c ? llvm::dyn_cast<LayoutType>(c->layout) : LayoutType();
    if (!layout || !tiler || !cg::is_static(tiler.getRef())) return {};
    cg::shape tile = cg::product_each(tiler.getRef().shape());
    cg::layout zipped = cg::zipped_divide(layout.getRef(), tile);
    auto thread = dynamicCoordLike(tile);
    if (!cg::is_valid(zipped) || !thread) return {};
    auto crd = cg::from_string<cg::coord>("(" + cg::to_string(*thread) + ",_)");
    if (!crd) return {};
    auto [sliced, offset] = cg::slice_and_offset(*crd, zipped);
    if (!cg::is_valid(sliced)) return {};
    // The slice keeps the rest mode's own modes, not a tuple around them.
    if (cg::holds_vector(sliced.shape()) && cg::rank(sliced.shape()) == 1)
      sliced = cg::get(sliced, 0);
    return moved(ctx, *c, offset, LayoutType::get(ctx, zeroUnitModes(sliced)));
  });
}

// CuTe's make_fragment_like: the first mode compact column-major with its
// stride-0 modes kept, the other modes ordered by their strides (dynamic
// ones last) after it; column-major when there is one mode or the first
// mode's shape is dynamic.
LogicalResult MakeFragmentLikeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  auto src = swizzledOuter(operands[0].getType());
  return infer(loc, out, [&]() -> Type {
    if (!src) return {};
    const cg::shape &shape = src->shape();
    std::vector<std::string> shapes, strides;
    leavesOf(shape, shapes);
    leavesOf(src->stride(), strides);
    if (shapes.size() != strides.size()) return {};
    size_t first = shapes.size();  // leaves of mode 0
    bool ordered = cg::holds_vector(shape) && cg::rank(shape) > 1 && cg::is_static(shape[0]);
    if (ordered) {
      std::vector<std::string> m0;
      leavesOf(shape[0], m0);
      first = m0.size();
    }
    // The order the leaves are laid out in.
    std::vector<size_t> order(shapes.size());
    std::iota(order.begin(), order.end(), 0);
    auto key = [&](size_t i) -> int64_t {
      int64_t v;
      return StringRef(strides[i]).getAsInteger(10, v) ? INT64_MAX : v;
    };
    if (ordered)
      std::stable_sort(order.begin() + first, order.end(),
                       [&](size_t x, size_t y) { return key(x) < key(y); });
    std::vector<std::string> result(shapes.size());
    cg::int_tuple step = *cg::from_string<cg::int_tuple>("1");
    for (size_t i : order) {
      if (isOne(shapes[i]) || (ordered && i < first && strides[i] == "0")) {
        result[i] = "0";
        continue;
      }
      result[i] = cg::to_string(step);
      auto n = cg::from_string<cg::int_tuple>(shapes[i]);
      if (!n) return {};
      step = cg::arith_tuple_mul<cg::int_tuple>(step, *n);
    }
    size_t i = 0;
    auto l = cg::from_string<cg::layout>(cg::to_string(shape) + ":" + nestLike(shape, result, i));
    return l ? LayoutType::get(ctx, std::move(*l)) : Type();
  });
}

LogicalResult MakeIdentityTensorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto s = llvm::dyn_cast<ShapeType>(operands[0].getType());
    if (!s) return {};
    std::vector<std::string> leaves;
    leavesOf(s.getRef(), leaves);
    for (auto &leaf : leaves) leaf = "0";
    size_t i = 0;
    auto zero = cg::from_string<cg::int_tuple>(nestLike(s.getRef(), leaves, i));
    cg::layout l = cg::make_identity_layout(s.getRef());
    if (!zero || !cg::is_valid(l)) return {};
    return CoordTensorType::get(ctx, IntTupleType::get(ctx, *zero), LayoutType::get(ctx, l));
  });
}

LogicalResult MakeLayoutLikeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    Type src = operands[0].getType();
    cg::layout r;
    if (auto l = llvm::dyn_cast<LayoutType>(src)) {
      if (cg::has_scaled_basis(l.getRef().stride())) return {};
      r = cg::make_layout_like(l.getRef());
    } else if (auto c = llvm::dyn_cast<ComposedLayoutType>(src)) {
      if (!c.getRef().is_a_swizzle() || cg::has_scaled_basis(c.getRef().layout_b().stride()))
        return {};
      r = cg::make_layout_like(c.getRef());
    } else {
      return {};
    }
    return cg::is_valid(r) ? LayoutType::get(ctx, std::move(r)) : Type();
  });
}

LogicalResult PtrLoadOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto p = llvm::dyn_cast<PtrType>(operands[0].getType());
    return p ? p.getValueType() : Type();
  });
}

LogicalResult RecastLayoutOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  int newBits = a.getNewTypeBits(), oldBits = a.getOldTypeBits();
  return infer(loc, out, [&]() -> Type {
    if (newBits <= 0 || oldBits <= 0) return {};
    auto outer = outerLayout(a.getSrc().getType());
    if (!outer || !cg::is_int_or_dynamic_int_only(outer->stride())) return {};
    return mapLayout(ctx, a.getSrc().getType(),
                     [&](const auto &l) { return cg::recast(newBits, oldBits, l); });
  });
}

// Without a profile the whole vector reduces to a scalar; with one, the
// modes it marks reduce and the others (`_`) remain, as a vector and shape.
LogicalResult ReduceOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto vec = llvm::dyn_cast<VectorType>(a.getInput().getType());
  if (!vec) return emitOptionalError(loc, "expects a vector input");
  if (!a.getReductionProfileAttr()) {
    out.push_back(vec.getElementType());
    return success();
  }
  auto shape = llvm::dyn_cast<ShapeType>(a.getShape().getType());
  if (!shape) return emitOptionalError(loc, "expects a shape");
  cg::shape rest = cg::slice(a.getReductionProfileAttr().getRef(), shape.getRef());
  auto n = cg::size(rest);
  if (!cg::is_valid(rest) || !cg::holds_int(n))
    return emitOptionalError(loc, "cannot reduce the shape");
  out.push_back(VectorType::get({n.as_int64()}, vec.getElementType()));
  out.push_back(ShapeType::get(ctx, std::move(rest)));
  return success();
}

LogicalResult SelectOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  ArrayRef<int32_t> mode = a.getMode();
  return infer(loc, out, [&]() -> Type {
    return llvm::TypeSwitch<Type, Type>(a.getInput().getType())
        .Case<ShapeType, StrideType, CoordType, IntTupleType, TileType, LayoutType,
              ComposedLayoutType>([&](auto ty) -> Type {
          size_t r = cg::rank(ty.getRef());
          for (int32_t m : mode)
            if (m < 0 || static_cast<size_t>(m) >= r) return {};
          auto res = cg::select(ty.getRef(), mode);
          if (!cg::is_valid(res)) return {};
          return decltype(ty)::get(ctx, std::move(res));
        })
        .Default([](Type) -> Type { return {}; });
  });
}

LogicalResult ShapeDivOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    auto x = llvm::dyn_cast<ShapeType>(operands[0].getType());
    auto y = llvm::dyn_cast<ShapeType>(operands[1].getType());
    if (!x || !y) return {};
    cg::shape r = cg::shape_div(x.getRef(), y.getRef());
    return cg::is_valid(r) ? ShapeType::get(ctx, std::move(r)) : Type();
  });
}

// The compiler aborts on every input tried (bad_alloc, segfault): no rule.
LogicalResult StencilDivideOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &) {
  return emitOptionalError(loc, "cute.stencil_divide inference is not recovered");
}

// No type of the compiler's registry implements the descriptor iterator
// interface this needs, so the compiler rejects every input.
LogicalResult DereferenceDescriptorIteratorOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &) {
  return emitOptionalError(loc, "expects a descriptor iterator");
}

LogicalResult TupleProductOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    return llvm::TypeSwitch<Type, Type>(operands[0].getType())
        .Case<IntTupleType, ShapeType>([&](auto ty) -> Type {
          auto r = cg::product(ty.getRef());
          return cg::is_valid(r) ? decltype(ty)::get(ctx, std::move(r)) : Type();
        })
        .Default([](Type) -> Type { return {}; });
  });
}

LogicalResult TupleProductEachOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  return infer(loc, out, [&]() -> Type {
    return llvm::TypeSwitch<Type, Type>(operands[0].getType())
        .Case<IntTupleType, ShapeType>([&](auto ty) -> Type {
          auto r = cg::product_each(ty.getRef());
          return cg::is_valid(r) ? decltype(ty)::get(ctx, std::move(r)) : Type();
        })
        .Default([](Type) -> Type { return {}; });
  });
}

// A SIMT copy's fragment is the tensor itself; the rule for other atoms is
// not recovered.
LogicalResult CopyMakeFragmentOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto tiled = llvm::dyn_cast<TiledCopyType>(a.getAtom().getType());
  Type in = a.getInput().getType();
  if (!tiled || !llvm::isa<cute_nvgpu::CopyAtomSIMTSyncCopyType>(tiled.getCopyAtom()) ||
      !llvm::isa<CuteMemRefType, LayoutType>(in))
    return emitOptionalError(loc, "copy fragment inference covers tiled SIMT copies only");
  out.push_back(in);
  return success();
}

// The atom's shape per CTA, then how often the atom repeats over the tile.
LogicalResult TiledMmaPartitionShapeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto atom = mmaAtomOf(a.getTiledMma().getType());
  auto shape = llvm::dyn_cast<ShapeType>(a.getInput().getType());
  auto which = llvm::dyn_cast_or_null<IntegerAttr>(a.getOperandIdAttr());
  return infer(loc, out, [&]() -> Type {
    if (!atom || !shape || !which || !cg::holds_vector(shape.getRef()) ||
        cg::rank(shape.getRef()) != 2)
      return {};
    int64_t full[2], part[2];
    switch (which.getInt()) {
      case 0: full[0] = atom->m; full[1] = atom->k; part[0] = atom->m / atom->ctas; break;
      case 1: full[0] = atom->n; full[1] = atom->k; part[0] = atom->n / atom->ctas; break;
      case 2: full[0] = atom->m; full[1] = atom->n; part[0] = atom->m / atom->ctas; break;
      default: return {};
    }
    part[1] = full[1];
    std::string rest;
    for (int i = 0; i < 2; ++i) {
      auto n = cg::size(shape.getRef(), i);
      if (!cg::holds_int(n) || n.as_int64() % full[i]) return {};
      rest += "," + std::to_string(n.as_int64() / full[i]);
    }
    auto r = cg::from_string<cg::shape>("((" + std::to_string(part[0]) + "," +
                                        std::to_string(part[1]) + ")" + rest + ")");
    return r ? ShapeType::get(ctx, std::move(*r)) : Type();
  });
}

// ---------------------------------------------------------------- copy atoms

namespace {

std::optional<cg::layout> layoutOf(const std::string &text) {
  return cg::from_string<cg::layout>(text);
}

template <class V>
int64_t asInt(const V &v) {
  return cg::holds_int(v) ? v.as_int64() : -1;
}


// TMEM addresses count lanes in the upper half-word: 2^21 bits per lane.
constexpr int64_t kTmemLaneBits = int64_t(1) << 21;

// A copy atom's traits as CuTe spells them (cute/atom/copy_traits_*.hpp):
// the thread count and the (thr,val) -> bit layouts, and for tensor memory
// atoms the bit -> address layout.
struct CopyTraits {
  int64_t threads = 1;
  int64_t bits = 8;
  cg::layout src, dst, ref, valId;
};

std::optional<CopyTraits> copyTraitsOf(Type atom) {
  CopyTraits t;
  auto set = [&](std::string src, std::string dst, bool refIsSrc,
                 std::string valId = "") -> std::optional<CopyTraits> {
    auto s = layoutOf(src), d = layoutOf(dst);
    if (!s || !d) return std::nullopt;
    t.src = *s;
    t.dst = *d;
    t.ref = refIsSrc ? *s : *d;
    if (!valId.empty()) {
      auto v = layoutOf(valId);
      if (!v) return std::nullopt;
      t.valId = *v;
    }
    return t;
  };
  auto n = [](int64_t v) { return std::to_string(v); };
  if (auto u = llvm::dyn_cast<cute_nvgpu::CopyAtomSIMTSyncCopyType>(atom)) {
    t.bits = bitsOfType(u.getValType());
    std::string l = "(1," + n(u.getCopyBits() ? u.getCopyBits() : t.bits) + "):(0,1)";
    return set(l, l, true);
  }
  if (auto s = llvm::dyn_cast<cute_nvgpu::CopyAtomStsmType>(atom)) {
    // SM90_U32xN_STSM_N and SM90_U16x2N_STSM_T: the SM75 ldmatrix layouts
    // with source and destination swapped.
    t.bits = bitsOfType(s.getValType());
    t.threads = 32;
    int64_t matrices = s.getNumMatrices();
    bool transposed = bool(s.getTranspose());
    std::string ldsmSrc, ldsmDst;
    if (!transposed) {
      switch (matrices) {
        case 1: ldsmSrc = "((8,4),128):((128,0),1)"; ldsmDst = "(32,32):(32,1)"; break;
        case 2: ldsmSrc = "((16,2),128):((128,0),1)"; ldsmDst = "(32,(32,2)):(32,(1,1024))"; break;
        case 4: ldsmSrc = "(32,128):(128,1)"; ldsmDst = "(32,(32,4)):(32,(1,1024))"; break;
        default: return std::nullopt;
      }
    } else {
      switch (matrices) {
        case 1: ldsmSrc = "((8,4),128):((128,0),1)"; ldsmDst = "((4,8),(16,2)):((256,16),(1,128))"; break;
        case 2: ldsmSrc = "((16,2),128):((128,0),1)"; ldsmDst = "((4,8),(16,2,2)):((256,16),(1,128,1024))"; break;
        case 4: ldsmSrc = "(32,128):(128,1)"; ldsmDst = "((4,8),(16,2,4)):((256,16),(1,128,1024))"; break;
        default: return std::nullopt;
      }
    }
    return set(ldsmDst, ldsmSrc, true);
  }
  // Tensor memory loads and stores: the tmem side is the reference; a store
  // is a load with source and destination swapped.
  auto tmem = [&](Type val, int64_t dp, int64_t bit, int64_t x, bool load) -> std::optional<CopyTraits> {
    t.bits = bitsOfType(val);
    t.threads = 32;
    std::string mem, reg, valId;
    if (dp == 32 && bit == 32) {
      mem = "(32," + n(1024 * x) + "):(0,1)";
      reg = "(32," + n(32 * x) + "):(" + n(32 * x) + ",1)";
      valId = "(" + n(32 * x) + ",32):(1," + n(kTmemLaneBits) + ")";
    } else if (dp == 16 && bit == 256) {
      mem = "(32," + n(4096 * x) + "):(0,1)";
      reg = x == 1 ? "((4,8),(64,2)):((64,256),(1,2048))"
                   : "((4,8),(64,2," + n(x) + ")):((64," + n(256 * x) + "),(1," + n(2048 * x) + ",256))";
      valId = "(" + n(256 * x) + ",16):(1," + n(kTmemLaneBits) + ")";
    } else {
      return std::nullopt;
    }
    return load ? set(mem, reg, true, valId) : set(reg, mem, false, valId);
  };
  // 16-bit packing and split reductions are not covered.
  if (auto l = llvm::dyn_cast<cute_nvgpu::CopyAtomSM100TmemLoadType>(atom))
    if (!l.getPack16b())
      return tmem(l.getValType(), l.getNumDp(), l.getNumBit(), l.getNumRep(), true);
  if (auto l = llvm::dyn_cast<cute_nvgpu::CopyAtomSM10xTmemLoadRedType>(atom))
    if (!l.getHalfSplitOff())
      return tmem(l.getValType(), l.getNumDp(), l.getNumBit(), l.getNumRep(), true);
  if (auto st = llvm::dyn_cast<cute_nvgpu::CopyAtomSM100TmemStoreType>(atom))
    if (!st.getExpand16b())
      return tmem(st.getValType(), st.getNumDp(), st.getNumBit(), st.getNumRep(), false);
  if (auto s = llvm::dyn_cast<cute_nvgpu::CopyAtomSM100CopyS2TType>(atom)) {
    // <T, 32 DP, 128 bit, C cta, x4>: SM100_UTCCP_4x32dp128bit_{C}cta.
    t.bits = bitsOfType(s.getValType());
    int64_t dp = s.getNumDp(), bit = s.getNumBit();
    t.threads = s.getNumCta();
    if (dp != 32 || bit != 128 || s.getBroadcast() != cute_nvgpu::CopyS2TBroadcast::x4 ||
        t.threads < 1)
      return std::nullopt;
    return set("(" + n(t.threads) + ",(32,128,4)):(0,(1,32,0))",
               "(" + n(t.threads) + ",16384):(0,1)", false,
               "(32,128,4):(" + n(kTmemLaneBits) + ",1," + n(32 * kTmemLaneBits) + ")");
  }
  return std::nullopt;
}

// A tiled copy: its atom in element units, the (thr,val) -> tile layout and
// the tiler over MN.
struct TiledCopy {
  Type atom;
  CopyTraits traits;
  cg::layout tv;
  cg::tile tiler;
  int64_t atomThreads = 0, atomVals = 0, tiledThreads = 0;
};

std::optional<TiledCopy> tiledCopyOf(Type t) {
  auto tiled = llvm::dyn_cast<TiledCopyType>(t);
  if (!tiled) return std::nullopt;
  auto traits = copyTraitsOf(tiled.getCopyAtom());
  if (!traits) return std::nullopt;
  TiledCopy tc;
  tc.atom = tiled.getCopyAtom();
  tc.traits = *traits;
  // Copy_Atom recasts the bit layouts to the value type.
  tc.traits.src = cg::upcast(traits->bits, traits->src);
  tc.traits.dst = cg::upcast(traits->bits, traits->dst);
  tc.traits.ref = cg::upcast(traits->bits, traits->ref);
  if (!cg::is_valid(tc.traits.src) || !cg::is_valid(tc.traits.dst) ||
      !cg::is_valid(tc.traits.ref))
    return std::nullopt;
  tc.tv = tiled.getLayoutCopyTv().getRef();
  tc.tiler = tiled.getTilerMn().getRef();
  tc.atomThreads = asInt(cg::size(tc.traits.ref, 0));
  tc.atomVals = asInt(cg::size(tc.traits.ref, 1));
  tc.tiledThreads = asInt(cg::size(tc.tv, 0));
  if (tc.atomThreads <= 0 || tc.atomVals <= 0 || tc.tiledThreads <= 0)
    return std::nullopt;
  return tc;
}

// The modes of a layout, one leaf mode as itself.
std::vector<cg::layout> modesOf(const cg::layout &l) {
  std::vector<cg::layout> out;
  if (!cg::holds_vector(l.shape())) return {l};
  for (size_t i = 0; i < cg::rank(l.shape()); ++i) out.push_back(cg::get(l, i));
  return out;
}

// `l` with its first mode composed with `f` and the others kept: CuTe's
// tensor.compose(f, _).
cg::layout composeFirst(const cg::layout &l, const cg::layout &f) {
  std::vector<cg::layout> modes = modesOf(l);
  modes[0] = cg::composition(modes[0], f);
  return cg::make_layout(modes);
}

// ((a,b),(c,d)) -> ((a,c),(b,d))
cg::layout zip2(const cg::layout &l) {
  return cg::make_layout(std::vector<cg::layout>{
      cg::make_layout(std::vector<cg::layout>{cg::get(l, 0, 0), cg::get(l, 1, 0)}),
      cg::make_layout(std::vector<cg::layout>{cg::get(l, 0, 1), cg::get(l, 1, 1)})});
}

cg::coord profileOf(const char *text) {
  return *cg::from_string<cg::coord>(text);
}

// TiledCopy::tile2thrfrg: ((Tile),(Rest...)) -> (Thr, (FrgV, FrgX), Rest...)
// through the atom's (src or dst) thread-value map `ref2trg`.
cg::layout tile2thrfrg(const TiledCopy &tc, const cg::layout &tiled,
                       const cg::layout &ref2trg, bool flatRest = true) {
  auto atomShape = cg::from_string<cg::shape>(
      "(" + std::to_string(tc.atomThreads) + "," + std::to_string(tc.atomVals) + ")");
  if (!atomShape) return cg::layout(cg::cg_error_t{});
  cg::layout atomTV = cg::zipped_divide(tc.tv, *atomShape);
  if (!cg::is_valid(atomTV)) return atomTV;
  cg::layout trgTV = composeFirst(atomTV, ref2trg);
  cg::layout thrval2mn = cg::coalesce(zip2(trgTV), profileOf("(1,(1,1))"));
  if (!cg::is_valid(thrval2mn)) return thrval2mn;
  cg::layout tv = cg::composition(cg::get(tiled, 0), thrval2mn);
  if (!cg::is_valid(tv)) return tv;
  std::vector<cg::layout> modes{cg::get(tv, 0), cg::get(tv, 1)};
  // The rest modes follow; a rank-1 tensor keeps its one rest mode whole.
  if (!flatRest) modes.push_back(cg::get(tiled, 1));
  else for (const cg::layout &m : modesOf(cg::get(tiled, 1))) modes.push_back(m);
  return cg::make_layout(modes);
}

// TiledCopy::tidfrg_S/D: (M,N,...) -> (Thr, (FrgV, FrgX), (RestM, RestN, ...)).
cg::layout tidfrg(const TiledCopy &tc, const cg::layout &tensor, bool src) {
  cg::layout tiled = cg::zipped_divide(tensor, tc.tiler);
  if (!cg::is_valid(tiled)) return tiled;
  cg::layout ref2trg = cg::composition(cg::right_inverse(tc.traits.ref),
                                       src ? tc.traits.src : tc.traits.dst);
  bool flatRest = cg::holds_vector(tensor.shape()) && cg::rank(tensor.shape()) > 1;
  return tile2thrfrg(tc, tiled, ref2trg, flatRest);
}

// ThrCopy::partition_S/D: the thread's slice of tidfrg.
Type partitionCopy(MLIRContext *ctx, const TiledCopy &tc, const Carrier &c,
                   const cg::coord &thread, bool src) {
  auto layout = llvm::dyn_cast<LayoutType>(c.layout);
  if (!layout) return {};
  cg::layout tv = tidfrg(tc, layout.getRef(), src);
  if (!cg::is_valid(tv)) return {};
  std::string coord = "(" + cg::to_string(thread);
  for (size_t i = 1; i < cg::rank(tv.shape()); ++i) coord += ",_";
  coord += ")";
  auto crd = cg::from_string<cg::coord>(coord);
  if (!crd) return {};
  auto [sliced, offset] = cg::slice_and_offset(*crd, tv);
  if (!cg::is_valid(sliced)) return {};
  // The tensor's modes beyond the tiler are untouched and keep their strides.
  size_t in = cg::rank(layout.getRef().shape()), r = cg::rank(tc.tiler);
  size_t keep = cg::holds_vector(layout.getRef().shape()) && in > r ? in - r : 0;
  Type result = LayoutType::get(ctx, zeroUnitModesHead(sliced, keep, false));
  if (c.ptr) return CuteMemRefType::get(ctx, movedPtr(c.ptr, offset), result);
  if (c.iter) {
    auto sum = cg::arith_tuple_sum<cg::int_tuple>(c.iter.getRef(), offset);
    if (!cg::is_valid(sum)) return {};
    return CoordTensorType::get(ctx, IntTupleType::get(ctx, std::move(sum)), result);
  }
  return rebuild(ctx, c, result);
}

// The shapes of a tiler's modes, as one shape.
std::optional<cg::shape> tilerShape(const cg::tile &tiler) {
  std::string text = cg::to_string(tiler);
  if (text.size() < 2 || text.front() != '[') return std::nullopt;
  SmallVector<StringRef> parts;
  StringRef(text).drop_front().drop_back().split(parts, ';');
  std::string shape = "(";
  for (size_t i = 0; i < parts.size(); ++i) {
    auto l = cg::from_string<cg::layout>(parts[i].str());
    if (!l) return std::nullopt;
    shape += (i ? "," : "") + cg::to_string(l->shape());
  }
  return cg::from_string<cg::shape>(shape + ")");
}

// TiledCopy::retile: (V, RestM, RestN, ...) of one thread's values regrouped
// into the atom's values.
Type retileCopy(MLIRContext *ctx, const TiledCopy &tc, const Carrier &c) {
  auto layout = llvm::dyn_cast<LayoutType>(c.layout);
  if (!layout) return {};
  const cg::layout &l = layout.getRef();
  size_t R = cg::rank(l.shape());
  int64_t V = asInt(cg::size(l, 0));
  auto shape = tilerShape(tc.tiler);
  if (V <= 0 || !shape) return {};
  cg::layout inv = cg::right_inverse(tc.tv).with_shape(*shape);
  cg::layout frgMN = cg::upcast(tc.tiledThreads * V, inv);
  if (!cg::is_valid(frgMN)) return {};
  auto vLayout = layoutOf(std::to_string(V) + ":1");
  auto atomVals = layoutOf(std::to_string(tc.atomVals) + ":1");
  if (!vLayout || !atomVals) return {};
  cg::layout frgV = cg::zipped_divide(
      cg::logical_product(*vLayout, cg::right_inverse(frgMN)), *atomVals);
  if (!cg::is_valid(frgV)) return {};
  std::string tiler = "(" + std::to_string(V);
  auto each = cg::product_each(frgMN.shape());
  for (size_t i = 0; i < cg::rank(each); ++i) tiler += "," + cg::to_string(each[i]);
  auto tilerShape = cg::from_string<cg::shape>(tiler + ")");
  if (!tilerShape) return {};
  cg::layout tiled = cg::zipped_divide(l, *tilerShape);
  if (!cg::is_valid(tiled)) return {};
  cg::layout v = composeFirst(tiled, frgV);
  if (!cg::is_valid(v)) return {};
  // v(_, (0, _, _...)): the first rest mode is the one the tile added.
  std::vector<cg::layout> modes{cg::get(v, 0)};
  std::vector<cg::layout> rest = modesOf(cg::get(v, 1));
  for (size_t i = 1; i < rest.size() && i < R; ++i) modes.push_back(rest[i]);
  return rebuild(ctx, c, LayoutType::get(ctx, zeroUnitModes(cg::make_layout(modes))));
}

// make_cotiled_copy: the (thr,val) -> tile layout and the tiler of a copy
// whose atom addresses `data` directly.
std::optional<std::pair<cg::layout, cg::tile>> cotiledCopy(const cg::layout &atomTV,
                                                          const cg::layout &data) {
  auto unit = layoutOf("1:0");
  cg::layout invData = cg::make_layout(std::vector<cg::layout>{cg::left_inverse(data), *unit});
  cg::layout tvData = cg::composition(invData, atomTV);
  if (!cg::is_valid(tvData)) return std::nullopt;
  auto flat = cg::product_each(data.shape());
  size_t R = cg::rank(flat);
  std::string flatText = cg::to_string(flat);
  std::string tilerText = "[";
  for (size_t i = 0; i < R; ++i) {
    std::string stride = "(";
    for (size_t j = 0; j < R; ++j) stride += std::string(j ? "," : "") + (i == j ? "1" : "0");
    auto pick = layoutOf(flatText + ":" + stride + ")");
    if (!pick) return std::nullopt;
    cg::layout mode = cg::filter(cg::composition(*pick, tvData));
    if (!cg::is_valid(mode)) return std::nullopt;
    tilerText += (i ? ";" : "") + cg::to_string(mode);
  }
  auto tiler = cg::from_string<cg::tile>(tilerText + "]");
  auto flatShape = cg::from_string<cg::shape>(flatText);
  if (!tiler || !flatShape) return std::nullopt;
  cg::layout tile2data = cg::composition(cg::layout(*flatShape), *tiler);
  cg::layout tv = cg::composition(cg::left_inverse(tile2data), tvData);
  if (!cg::is_valid(tv)) return std::nullopt;
  return std::make_pair(tv, *tiler);
}

Type tiledCopyType(MLIRContext *ctx, Type atom, const cg::layout &tv, const cg::tile &tiler) {
  return TiledCopyType::get(ctx, atom, LayoutAttr::get(ctx, tv),
                            TileAttr::get(ctx, tiler));
}

// ---------------------------------------------------------------- TMA

struct TmaFormat {
  std::string name;
  int64_t bits = 0;
};

// The descriptor's data format: the attribute when given, else the element's.
std::optional<TmaFormat> tmaFormatOf(Type elem, Attribute attr) {
  TmaFormat f;
  if (attr) {
    auto format = llvm::dyn_cast<cute_nvgpu::TmaDataFormatAttr>(attr);
    if (!format) return std::nullopt;
    f.name = cute_nvgpu::stringifyTmaDataFormat(format.getValue()).str();
    StringRef n = f.name;
    if (n == "F16_RN" || n == "BF16_RN") f.bits = 16;
    else if (n == "F32_RN" || n == "F32_FTZ_RN" || n == "INT32" || n == "TF32") f.bits = 32;
    else if (n == "F64_RN" || n == "INT64") f.bits = 64;
    else if (!n.drop_until(llvm::isDigit).take_while(llvm::isDigit).getAsInteger(10, f.bits)) {}
    else return std::nullopt;
    return f;
  }
  f.bits = bitsOfType(elem);
  if (elem.isTF32()) f.name = "TF32_RN";
  else if (elem.isF16()) f.name = "F16_RN";
  else if (elem.isBF16()) f.name = "BF16_RN";
  else if (elem.isF32()) f.name = "F32_RN";
  else if (elem.isF64()) f.name = "F64_RN";
  else f.name = "U" + std::to_string(f.bits);
  return f;
}

// CuTe's coalesce_256: which adjacent modes of a flattened layout merge into
// one TMA box dimension, the merged extent staying within 256.
std::vector<std::vector<size_t>> coalesce256Groups(const cg::layout &l) {
  auto fs = cg::flatten(l.shape());
  auto ft = cg::flatten(l.stride());
  if (!cg::holds_vector(fs)) return {{0}};
  std::vector<std::vector<size_t>> groups;
  std::vector<size_t> cur{0};
  int64_t curS = asInt(fs[0]), curD = asInt(ft[0]);
  for (size_t i = 1; i < cg::rank(fs); ++i) {
    int64_t s = asInt(fs[i]), d = asInt(ft[i]);
    if (curS == 1) {
      cur = {i}; curS = s; curD = d;
    } else if (s == 1) {
      continue;
    } else if (curS > 0 && curD >= 0 && s > 0 && d >= 0 && curS * curD == d &&
               s * curS <= 256) {
      cur.push_back(i);
      curS *= s;
    } else {
      groups.push_back(cur);
      cur = {i}; curS = s; curD = d;
    }
  }
  groups.push_back(cur);
  return groups;
}

// The layout with `shape`'s leaves regrouped by `groups`, over `stride`'s
// leaves: what composing with the box shape does, keeping every basis.
std::optional<cg::layout> regroup(const cg::layout &shape, const cg::layout &stride,
                                  const std::vector<std::vector<size_t>> &groups) {
  auto fs = cg::flatten(shape.shape());
  auto ft = cg::flatten(stride.stride());
  auto leaf = [&](const auto &flat, size_t i) {
    return cg::holds_vector(flat) ? cg::to_string(flat[i]) : cg::to_string(flat);
  };
  std::string shapes, strides;
  for (size_t g = 0; g < groups.size(); ++g) {
    if (g) { shapes += ","; strides += ","; }
    if (groups[g].size() == 1) {
      shapes += leaf(fs, groups[g][0]);
      strides += leaf(ft, groups[g][0]);
      continue;
    }
    shapes += "("; strides += "(";
    for (size_t k = 0; k < groups[g].size(); ++k) {
      if (k) { shapes += ","; strides += ","; }
      shapes += leaf(fs, groups[g][k]);
      strides += leaf(ft, groups[g][k]);
    }
    shapes += ")"; strides += ")";
  }
  return layoutOf("(" + shapes + "):(" + strides + ")");
}

// A layout composed with a coordinate layout: each mode of `rhs` selects,
// through its basis stride, the mode of `lhs` it indexes.
cg::layout composeCoord(const cg::layout &lhs, const cg::layout &rhs) {
  std::vector<cg::layout> modes;
  for (const cg::layout &m : modesOf(rhs)) {
    const auto &d = m.stride();
    if (cg::holds_int(d)) {
      modes.push_back(m);
      continue;
    }
    if (!cg::holds_scaled_basis(d)) return cg::layout(cg::cg_error_t{});
    const auto &basis = std::get<cg::scaled_basis>(d);
    if (!basis.value_holds_int()) return cg::layout(cg::cg_error_t{});
    auto scale = layoutOf(cg::to_string(m.shape()) + ":" +
                          std::to_string(basis.static_integral_value()));
    if (!scale) return cg::layout(cg::cg_error_t{});
    cg::layout sub = cg::get(lhs, basis.modes());
    cg::layout composed = cg::composition(sub, *scale);
    if (!cg::is_valid(composed)) return composed;
    modes.push_back(composed);
  }
  return cg::make_layout(modes);
}

// Leaves of congruent shape, stride and basis, with the path to each.
template <class F>
void walkLeaves(const cg::shape &s, const cg::stride &d, const cg::stride &e,
                std::vector<int> &path, F &&f) {
  if (cg::holds_vector(s)) {
    for (size_t i = 0; i < cg::rank(s); ++i) {
      path.push_back(i);
      walkLeaves(s[i], d[i], e[i], path, f);
      path.pop_back();
    }
    return;
  }
  f(s, d, e, path);
}

bool basisAt(const cg::stride &leaf, const std::vector<int> &path) {
  return cg::holds_scaled_basis(leaf) &&
         std::get<cg::scaled_basis>(leaf).modes() == path;
}

// Which mode of a stride tuple carries the basis `path`, if any.
std::optional<size_t> modeWithBasis(const cg::stride &strides, const std::vector<int> &path) {
  for (size_t j = 0; j < cg::rank(strides); ++j) {
    auto flat = cg::flatten(strides[j]);
    if (!cg::holds_vector(flat)) {
      if (basisAt(flat, path)) return j;
      continue;
    }
    for (size_t k = 0; k < cg::rank(flat); ++k)
      if (basisAt(flat[k], path)) return j;
  }
  return std::nullopt;
}

// construct_tma_gbasis: the TMA box shape with, per mode, the gmem modes it
// runs over.
cg::layout tmaGbasis(const cg::layout &gmem, int64_t elemBits, int64_t tmaBits,
                     const cg::layout &smem, const cg::layout &ctaVMap) {
  auto fail = [] { return cg::layout(cg::cg_error_t{}); };
  cg::layout inv = cg::right_inverse(smem);
  cg::layout full = cg::coalesce(cg::composition(ctaVMap, inv));
  if (!cg::is_valid(full)) return full;
  std::vector<cg::layout> fullModes = modesOf(full);
  // Keep the leading modes that step through gmem one element at a time.
  size_t smemRank = 0;
  for (; smemRank < fullModes.size(); ++smemRank) {
    const auto &d = fullModes[smemRank].stride();
    if (!cg::holds_scaled_basis(d)) break;
    const auto &b = std::get<cg::scaled_basis>(d);
    if (!b.value_holds_int() || b.static_integral_value() != 1) break;
  }
  if (smemRank == 0) return fail();
  fullModes.resize(smemRank);
  cg::layout sidx2gmode = cg::make_layout(fullModes);
  cg::layout tileGstride = cg::recast(tmaBits, elemBits, composeCoord(gmem, sidx2gmode));
  if (!cg::is_valid(tileGstride)) return tileGstride;
  cg::layout gbasis = cg::make_identity_layout(gmem.shape());
  cg::layout tileGbasisTmp = composeCoord(gbasis, sidx2gmode);
  if (!cg::is_valid(tileGbasisTmp)) return tileGbasisTmp;
  auto regrouped = regroup(tileGstride, tileGbasisTmp, coalesce256Groups(tileGstride));
  if (!regrouped) return fail();
  cg::layout tmaGbasisTile = *regrouped;
  // Gmem modes the tile misses still shape the descriptor, as size-1 modes.
  cg::layout gmemT = cg::recast(tmaBits, elemBits, gmem);
  if (!cg::is_valid(gmemT)) return gmemT;
  std::vector<cg::layout> modes = modesOf(tmaGbasisTile);
  auto flatTile = cg::flatten(tmaGbasisTile.stride());
  std::vector<int> path;
  bool ok = true;
  walkLeaves(gmemT.shape(), gmemT.stride(), gbasis.stride(), path,
             [&](const auto &s, const auto &d, const auto &e, const std::vector<int> &p) {
               if (asInt(s) == 1 || asInt(d) == 0) return;
               bool found = false;
               if (!cg::holds_vector(flatTile)) {
                 found = basisAt(flatTile, p);
               } else {
                 for (size_t k = 0; k < cg::rank(flatTile) && !found; ++k)
                   found = basisAt(flatTile[k], p);
               }
               if (found) return;
               auto extra = layoutOf("1:" + cg::to_string(e));
               if (!extra) ok = false;
               else modes.push_back(*extra);
             });
  if (!ok) return fail();
  cg::layout result = cg::make_layout(modes);
  if (modes.size() > 4) result = cg::group(4, modes.size(), result);
  return result;
}

// The TMA tensor's layout: gmem shape over TMA coordinate bases, scaled to
// the descriptor's element.
std::optional<cg::layout> tmaTensorLayout(const cg::layout &gmem, int64_t elemBits,
                                          int64_t tmaBits, const cg::layout &gbasis) {
  cg::layout basis = cg::make_identity_layout(gmem.shape());
  const auto &tmaStrides = gbasis.stride();
  bool ok = true;
  std::string text;
  std::vector<int> path;
  // Written in the shape's own nesting.
  std::function<void(const cg::shape &, const cg::stride &, const cg::stride &)> write =
      [&](const cg::shape &s, const cg::stride &d, const cg::stride &e) {
        if (cg::holds_vector(s)) {
          text += "(";
          for (size_t i = 0; i < cg::rank(s); ++i) {
            if (i) text += ",";
            path.push_back(i);
            write(s[i], d[i], e[i]);
            path.pop_back();
          }
          text += ")";
          return;
        }
        if (asInt(s) == 1 || asInt(d) == 0) { text += "0"; return; }
        auto j = cg::holds_vector(tmaStrides) ? modeWithBasis(tmaStrides, path)
                                              : (basisAt(tmaStrides, path) ? std::optional<size_t>(0) : std::nullopt);
        if (!j) { text += "0"; return; }
        if (*j == 0) {
          int64_t stride = asInt(d);
          if (stride <= 0) { ok = false; return; }
          int64_t num = stride * elemBits, den = tmaBits;
          int64_t g = std::gcd(num, den);
          num /= g; den /= g;
          text += std::to_string(num) + (den == 1 ? "" : "/" + std::to_string(den)) + "@0";
          return;
        }
        auto mode = cg::holds_vector(tmaStrides) ? tmaStrides[*j] : tmaStrides;
        auto flat = cg::flatten(mode);
        if (cg::holds_vector(flat) && cg::rank(flat) != 1) { ok = false; return; }
        text += "1@" + std::to_string(*j);
      };
  write(gmem.shape(), gmem.stride(), basis.stride());
  if (!ok) return std::nullopt;
  return layoutOf(cg::to_string(gmem.shape()) + ":" + text);
}

// The smem tile layout without its swizzle.
std::optional<cg::layout> plainLayoutOf(Type t) {
  if (auto l = llvm::dyn_cast<LayoutType>(t)) return l.getRef();
  if (auto c = llvm::dyn_cast<ComposedLayoutType>(t)) return c.getRef().layout_b();
  return std::nullopt;
}

struct TmaAtomInfo {
  int64_t elemBits = 0, copyBits = 0;
};

std::optional<TmaAtomInfo> tmaAtomInfoOf(Type atom) {
  TmaAtomInfo info;
  if (auto l = llvm::dyn_cast<cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>(atom)) {
    info.elemBits = bitsOfType(l.getValType());
    info.copyBits = l.getCopyBits();
  } else if (auto s = llvm::dyn_cast<cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>(atom)) {
    info.elemBits = bitsOfType(s.getValType());
    info.copyBits = s.getCopyBits();
  } else if (auto r = llvm::dyn_cast<cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>(atom)) {
    info.elemBits = bitsOfType(r.getValType());
    info.copyBits = r.getCopyBits();
  } else if (auto l = llvm::dyn_cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>(atom)) {
    info.elemBits = bitsOfType(l.getValType());
    info.copyBits = l.getCopyBits();
  } else if (auto s = llvm::dyn_cast<cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>(atom)) {
    info.elemBits = bitsOfType(s.getValType());
    info.copyBits = s.getCopyBits();
  } else {
    return std::nullopt;
  }
  if (info.elemBits <= 0 || info.copyBits <= 0) return std::nullopt;
  return info;
}


// make_non_exec_tiled_tma_load/store: the atom and the TMA tensor.
LogicalResult inferTmaAtom(MLIRContext *ctx, std::optional<Location> loc,
                           Type gmemType, Type smemType, Type ctaVMapType,
                           Attribute kind, Attribute format, SmallVectorImpl<Type> &out) {
  auto gmem = llvm::dyn_cast<CuteMemRefType>(gmemType);
  auto smem = plainLayoutOf(smemType);
  auto ctaVMap = llvm::dyn_cast<LayoutType>(ctaVMapType);
  if (!gmem || !smem || !ctaVMap)
    return emitOptionalError(loc, "expects a gmem tensor, an smem layout and a CTA value map");
  auto glayout = llvm::dyn_cast<LayoutType>(gmem.getLayout());
  if (!glayout) return emitOptionalError(loc, "gmem tensor needs a plain layout");
  Type elem = gmem.getValueType();
  auto fmt = tmaFormatOf(elem, format);
  if (!fmt || fmt->bits <= 0) return emitOptionalError(loc, "unknown TMA data format");
  int64_t elemBits = bitsOfType(elem);
  cg::layout gbasis = tmaGbasis(glayout.getRef(), elemBits, fmt->bits, *smem, ctaVMap.getRef());
  if (!cg::is_valid(gbasis)) return emitOptionalError(loc, "cannot build the TMA basis");
  auto tensor = tmaTensorLayout(glayout.getRef(), elemBits, fmt->bits, gbasis);
  if (!tensor) return emitOptionalError(loc, "cannot build the TMA tensor");
  int64_t copyBits = asInt(cg::size(gbasis)) * fmt->bits;
  auto gbasisType = LayoutType::get(ctx, gbasis);
  std::optional<cute_nvgpu::TmaDataFormat> dataFormat =
      cute_nvgpu::symbolizeTmaDataFormat(fmt->name);
  Type atom;
  if (auto k = llvm::dyn_cast_or_null<cute_nvgpu::TiledTmaLoadAttr>(kind)) {
    atom = cute_nvgpu::CopyAtomNonExecTiledTmaLoadType::get(ctx, k.getValue(), elem, copyBits,
                                                            gbasisType, dataFormat);
  } else if (auto k = llvm::dyn_cast_or_null<cute_nvgpu::ReductionKindAttr>(kind)) {
    atom = cute_nvgpu::CopyAtomNonExecTiledTmaReduceType::get(ctx, k.getValue(), elem, copyBits,
                                                              gbasisType, dataFormat);
  } else {
    atom = cute_nvgpu::CopyAtomNonExecTiledTmaStoreType::get(ctx, elem, copyBits, gbasisType,
                                                             dataFormat);
  }
  std::string zeros = "(";
  for (size_t i = 0; i < cg::rank(gbasis.shape()); ++i) zeros += i ? ",0" : "0";
  auto iter = cg::from_string<cg::int_tuple>(zeros + ")");
  if (!iter) return emitOptionalError(loc, "cannot build the TMA coordinate");
  out.push_back(atom);
  out.push_back(CoordTensorType::get(ctx, IntTupleType::get(ctx, *iter),
                                     LayoutType::get(ctx, *tensor)));
  return success();
}

}  // namespace

// ---------------------------------------------------------------- cute copy ops

// TiledCopy::get_layoutD_TV over the reference tile, then the tiled copy of
// another atom with that layout (make_tiled_copy_D).
Type mlir::cutlass_compiler::cute::tiledCopyD(Type atom, TiledCopyType tiled) {
  auto tc = tiledCopyOf(tiled);
  if (!tc) return {};
  auto shape = tilerShape(tc->tiler);
  if (!shape) return {};
  auto refShape = cg::from_string<cg::shape>("(" + cg::to_string(*shape) + ",1)");
  if (!refShape) return {};
  cg::layout refD(*refShape);
  cg::layout ref2trg = cg::composition(cg::right_inverse(tc->traits.ref), tc->traits.dst);
  cg::layout tv = tile2thrfrg(*tc, refD, ref2trg);
  if (!cg::is_valid(tv) || cg::rank(tv.shape()) < 2) return {};
  cg::layout layoutTV =
      zeroUnitModes(cg::make_layout(std::vector<cg::layout>{cg::get(tv, 0), cg::get(tv, 1)}));
  MLIRContext *ctx = atom.getContext();
  return TiledCopyType::get(ctx, atom, LayoutAttr::get(ctx, layoutTV),
                            tiled.getTilerMn());
}

LogicalResult TiledCopyPartitionSOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto tc = tiledCopyOf(a.getTiledCopy().getType());
  auto c = carrierOf(a.getInput().getType());
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  if (!tc || !c || !coord)
    return emitOptionalError(loc, "expects a tiled copy, a tensor and a thread coord");
  return infer(loc, out, [&] { return partitionCopy(ctx, *tc, *c, coord.getRef(), true); });
}

LogicalResult TiledCopyPartitionDOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto tc = tiledCopyOf(a.getTiledCopy().getType());
  auto c = carrierOf(a.getInput().getType());
  auto coord = llvm::dyn_cast<CoordType>(a.getCoord().getType());
  if (!tc || !c || !coord)
    return emitOptionalError(loc, "expects a tiled copy, a tensor and a thread coord");
  return infer(loc, out, [&] { return partitionCopy(ctx, *tc, *c, coord.getRef(), false); });
}

LogicalResult TiledCopyRetileOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto tc = tiledCopyOf(a.getTiledCopy().getType());
  auto c = carrierOf(a.getInput().getType());
  if (!tc || !c) return emitOptionalError(loc, "expects a tiled copy and a tensor");
  return infer(loc, out, [&] { return retileCopy(ctx, *tc, *c); });
}

// ---------------------------------------------------------------- cute_nvgpu


namespace mlir::cutlass_compiler::cute_nvgpu {

LogicalResult AtomSetValueOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  out.push_back(operands[0].getType());
  return success();
}

// The executable atom of a TMA descriptor: the element, copy size and basis
// carry over; the kind becomes the mode and, for loads, the CTA count.
LogicalResult AtomCopyMakeExecTmaOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  bool over = a.getOverride(), noOob = a.getNoFullyOobTile();
  Type in = a.getInput().getType();
  auto stride = cg::from_string<cg::stride>("()");
  if (!stride) return emitOptionalError(loc, "cannot build the TMA stride");
  auto gStride = StrideType::get(ctx, *stride);
  if (auto load = llvm::dyn_cast<CopyAtomNonExecTiledTmaLoadType>(in)) {
    TiledTmaLoad kind = load.getKind();
    bool twoSm = kind == TiledTmaLoad::sm_100_2sm || kind == TiledTmaLoad::sm_100_2sm_multicast;
    bool mcast = kind == TiledTmaLoad::sm_90_multicast || kind == TiledTmaLoad::sm_100_2sm_multicast;
    out.push_back(CopyAtomTmaLoadType::get(ctx, load.getValType(), /*sparsity=*/1,
                                           load.getCopyBits(), TmaLoadMode::tiled, twoSm ? 2 : 1,
                                           gStride, mcast, load.getTmaGbasis(), over, noOob));
    return success();
  }
  if (auto store = llvm::dyn_cast<CopyAtomNonExecTiledTmaStoreType>(in)) {
    out.push_back(CopyAtomTmaStoreType::get(ctx, store.getValType(), /*sparsity=*/1,
                                            store.getCopyBits(), TmaStoreMode::tiled, gStride,
                                            store.getTmaGbasis(), over, noOob));
    return success();
  }
  if (auto red = llvm::dyn_cast<CopyAtomNonExecTiledTmaReduceType>(in)) {
    out.push_back(CopyAtomTmaReduceType::get(ctx, red.getValType(), red.getCopyBits(),
                                             TmaStoreMode::tiled, red.getKind(), gStride,
                                             red.getTmaGbasis()));
    return success();
  }
  if (auto load = llvm::dyn_cast<CopyAtomNonExecIm2ColTmaLoadType>(in)) {
    Im2ColTmaLoad kind = load.getKind();
    bool twoSm = kind == Im2ColTmaLoad::sm_100_2sm || kind == Im2ColTmaLoad::sm_100_2sm_multicast;
    bool mcast = kind == Im2ColTmaLoad::sm_90_multicast || kind == Im2ColTmaLoad::sm_100_2sm_multicast;
    out.push_back(CopyAtomIm2ColTmaLoadType::get(ctx, load.getValType(), load.getCopyBits(),
                                                 twoSm ? 2 : 1, gStride, mcast, load.getTmaGbasis()));
    return success();
  }
  if (auto store = llvm::dyn_cast<CopyAtomNonExecIm2ColTmaStoreType>(in)) {
    out.push_back(CopyAtomIm2ColTmaStoreType::get(ctx, store.getValType(), store.getCopyBits(),
                                                  gStride, store.getTmaGbasis()));
    return success();
  }
  // The gather/scatter atoms' executable forms are not recovered.
  return emitOptionalError(loc, "expects a non-executable tiled or im2col TMA atom");
}


LogicalResult AtomCopyMakeNonExecTiledTmaLoadOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  return inferTmaAtom(ctx, loc, a.getGmemTensor().getType(),
                      a.getSmemLayout().getType(), a.getCtaVMap().getType(),
                      a.getKindAttr(), a.getTmaFormatAttr(), out);
}

LogicalResult AtomCopyMakeNonExecTiledTmaStoreOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  return inferTmaAtom(ctx, loc, a.getGmemTensor().getType(),
                      a.getSmemLayout().getType(), a.getCtaVMap().getType(),
                      /*kind=*/nullptr, a.getTmaFormatAttr(), out);
}

LogicalResult AtomCopyMakeNonExecTiledTmaReduceOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  return inferTmaAtom(ctx, loc, a.getGmemTensor().getType(),
                      a.getSmemLayout().getType(), a.getCtaVMap().getType(),
                      a.getKindAttr(), a.getTmaFormatAttr(), out);
}

// tma_partition: the smem tile's contiguous vectors, one instruction each,
// read through every tensor; a multicast CTA starts at its share.
LogicalResult AtomTmaPartitionOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto info = tmaAtomInfoOf(a.getTmaAtom().getType());
  auto coord = llvm::dyn_cast<CoordType>(a.getCtaCoord().getType());
  auto ctaLayout = llvm::dyn_cast<LayoutType>(a.getCtaLayout().getType());
  auto smem = carrierOf(a.getSmemTensor().getType());
  if (!info || !coord || !ctaLayout || !smem)
    return emitOptionalError(loc, "expects a TMA atom, a CTA coord and layout, and tensors");
  auto smemLayout = llvm::dyn_cast_or_null<LayoutType>(smem->layout);
  if (!smemLayout || !cg::holds_vector(smemLayout.getRef().shape()))
    return emitOptionalError(loc, "smem tensor needs a plain layout of rank >= 1");
  int64_t numVal = info->copyBits / info->elemBits;
  cg::layout inv = cg::right_inverse(cg::get(smemLayout.getRef(), 0));
  int64_t b = asInt(cg::size(inv));
  int64_t n = asInt(cg::size(smemLayout.getRef(), 0));
  if (b <= 0 || n <= 0 || n % b || numVal <= 0)
    return emitOptionalError(loc, "cannot tile the smem vector");
  // tile_to_shape: the vector repeated over the tile, broadcast modes at 0.
  cg::layout layoutV = inv;
  if (n != b) {
    auto rest = layoutOf(std::to_string(n / b) + ":1");
    layoutV = cg::logical_product(inv, *rest);
    if (!cg::is_valid(layoutV)) return emitOptionalError(loc, "cannot tile the smem vector");
  }
  auto tmaV = layoutOf(std::to_string(numVal) + ":1");
  layoutV = cg::logical_divide(layoutV, *tmaV);
  if (!cg::is_valid(layoutV)) return emitOptionalError(loc, "cannot divide the smem vector");
  // The multicast offset inside the instruction's vector.
  int64_t ctas = asInt(cg::cosize(ctaLayout.getRef()));
  std::optional<cg::int_tuple> multicast;
  if (ctas > 1) {
    // The CTA layout scaled by the share: (cosize:share) o cta_layout.
    auto share = layoutOf(std::to_string(ctas) + ":" + std::to_string(numVal / ctas));
    cg::layout scaled = cg::composition(*share, ctaLayout.getRef());
    if (!cg::is_valid(scaled)) return emitOptionalError(loc, "cannot scale the CTA layout");
    multicast = cg::layout_eval(coord.getRef(), scaled);
  }
  SmallVector<Type> results;
  auto partition = [&](Type tensor) -> Type {
    auto c = carrierOf(tensor);
    if (!c) return {};
    auto layout = llvm::dyn_cast<LayoutType>(c->layout);
    if (!layout) return {};
    cg::layout v = cg::coalesce(composeFirst(layout.getRef(), layoutV), profileOf("((1,1))"));
    if (!cg::is_valid(v)) return {};
    // Only the vector's unit modes get a zero stride; the rest keep theirs.
    Type result = LayoutType::get(ctx, zeroUnitModesHead(v, cg::rank(v.shape()) - 1, false));
    if (!multicast) return rebuild(ctx, *c, result);
    // The index split over the vector's modes, column-major, keeping what
    // its divisibility says of each coordinate.
    auto vec = cg::get(v.shape(), 0);
    if (!cg::holds_vector(vec)) return {};
    auto split = splitIndex(*multicast, vec[0]);
    if (!split) return {};
    std::string crd = "((" + *split + ",0)";
    for (size_t i = 1; i < cg::rank(v.shape()); ++i) crd += ",0";
    auto at = cg::from_string<cg::coord>(crd + ")");
    if (!at) return {};
    cg::int_tuple offset = cg::layout_eval(*at, v);
    if (c->ptr) return CuteMemRefType::get(ctx, movedPtr(c->ptr, offset), result);
    if (c->iter) {
      auto sum = cg::arith_tuple_sum<cg::int_tuple>(c->iter.getRef(), offset);
      if (!cg::is_valid(sum)) return {};
      return CoordTensorType::get(ctx, IntTupleType::get(ctx, std::move(sum)), result);
    }
    return rebuild(ctx, *c, result);
  };
  results.push_back(partition(a.getSmemTensor().getType()));
  for (Value target : a.getTargetTensors()) results.push_back(partition(target.getType()));
  for (Type t : results)
    if (!t) return emitOptionalError(loc, "cannot partition for TMA");
  out.append(results.begin(), results.end());
  return success();
}

namespace {
// make_cotiled_copy over a tensor memory tensor for an atom whose thread
// layout is `atomT`.
Type tmemTiledCopy(MLIRContext *ctx, std::optional<Location> loc, Type atom,
                   Type tmem, const std::string &atomT, SmallVectorImpl<Type> &out) {
  auto traits = copyTraitsOf(atom);
  auto m = llvm::dyn_cast<CuteMemRefType>(tmem);
  auto layout = m ? llvm::dyn_cast<LayoutType>(m.getLayout()) : LayoutType();
  if (!traits || !layout || !cg::is_valid(traits->valId)) return {};
  auto t = layoutOf(atomT);
  if (!t) return {};
  cg::layout v = cg::coalesce(cg::upcast(traits->bits, traits->valId));
  if (!cg::is_valid(v)) return {};
  auto tiled = cotiledCopy(cg::make_layout(std::vector<cg::layout>{*t, v}), layout.getRef());
  if (!tiled) return {};
  return tiledCopyType(ctx, atom, tiled->first, tiled->second);
}
}  // namespace

// make_tmem_copy: four warps, each at the same place in its own lane quarter.
LogicalResult AtomCopyMakeTmemCopyOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto traits = copyTraitsOf(a.getAtom().getType());
  if (!traits || traits->bits <= 0) return emitOptionalError(loc, "expects a tmem atom");
  int64_t lane = kTmemLaneBits / traits->bits;
  return infer(loc, out, [&] {
    return tmemTiledCopy(ctx, loc, a.getAtom().getType(), a.getTmemMemref().getType(),
                         "(32,4):(0," + std::to_string(32 * lane) + ")", out);
  });
}

// make_s2t_copy (make_utccp_copy): one thread per CTA.
LogicalResult AtomCopyMakeS2TCopyOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  auto traits = copyTraitsOf(a.getCopyS2tAtom().getType());
  if (!traits) return emitOptionalError(loc, "expects an s2t atom");
  return infer(loc, out, [&] {
    return tmemTiledCopy(ctx, loc, a.getCopyS2tAtom().getType(),
                         a.getTmemMemref().getType(),
                         std::to_string(traits->threads) + ":0", out);
  });
}

namespace {
// The smem side of an s2t copy as descriptor units: the tile recast to 128
// bits, viewed through `desc`.
LogicalResult s2tSmemDescView(std::optional<Location> loc, Type atom, Type view,
                              Type desc, SmallVectorImpl<Type> &out) {
  if (auto tiled = llvm::dyn_cast<TiledCopyType>(atom)) atom = tiled.getCopyAtom();
  auto m = llvm::dyn_cast<CuteMemRefType>(view);
  auto layout = m ? llvm::dyn_cast<LayoutType>(m.getLayout()) : LayoutType();
  if (!llvm::isa<CopyAtomSM100CopyS2TType>(atom) || !layout)
    return emitOptionalError(loc, "expects an s2t atom and an smem tensor");
  // The compiler wants mode 0 as (values, 1).
  const cg::layout &l = layout.getRef();
  auto m0 = cg::get(l.shape(), 0);
  if (!cg::holds_vector(m0) || cg::rank(m0) != 2 || !cg::holds_int(cg::size(m0, 1)) ||
      cg::size(m0, 1).as_int64() != 1)
    return emitOptionalError(loc, "expects mode 0 as (values, 1)");
  return infer(loc, out, [&]() -> Type {
    cg::layout v = cg::recast(128, bitsOfType(m.getValueType()), l);
    if (!cg::is_valid(v)) return {};
    return SmemDescViewType::get(atom.getContext(), desc, LayoutAttr::get(atom.getContext(), v));
  });
}

// An operation whose inference rule is not recovered: the compiler either
// aborts on the inputs tried or needs inputs no probe produced.
LogicalResult unrecovered(std::optional<Location> loc, StringRef op) {
  return emitOptionalError(loc, op, " result inference is not recovered");
}
}  // namespace

LogicalResult AtomGetCopyS2TSmemDescViewOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  return s2tSmemDescView(loc, a.getAtom().getType(), a.getView().getType(),
                         SmemDescType::get(ctx), out);
}

LogicalResult AtomGetCopyS2TSmemDescViewSM107Op::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  return s2tSmemDescView(loc, a.getAtom().getType(), a.getView().getType(),
                         SmemDescSM107Type::get(ctx), out);
}

// A generic-space pointer to the parameter, at its type's natural alignment.
LogicalResult GetGridConstantPointerOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr, PropertyRef, RegionRange, SmallVectorImpl<Type> &out) {
  Type t = operands[0].getType();
  bool desc = llvm::isa<TmaDescriptorTiledType, TmaDescriptorIm2ColType>(t);
  out.push_back(PtrType::get(ctx, t, AddressSpace::generic,
                             desc ? 64 : PtrType::getNaturalAlignment(t), {}, {}));
  return success();
}

// The residues of the TMA tensor's coordinates: one per box mode, starting
// at the extent of the gmem mode it runs over and counting down.
LogicalResult AtomMakeTmaResidueTensorOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, PropertyRef props, RegionRange,
    SmallVectorImpl<Type> &out) {
  Adaptor a(operands, attrs, props);
  Type atom = a.getTmaAtom().getType(), gbasis;
  if (auto l = llvm::dyn_cast<CopyAtomNonExecTiledTmaLoadType>(atom)) gbasis = l.getTmaGbasis();
  if (auto s = llvm::dyn_cast<CopyAtomNonExecTiledTmaStoreType>(atom)) gbasis = s.getTmaGbasis();
  auto basis = llvm::dyn_cast_or_null<LayoutType>(gbasis);
  auto m = llvm::dyn_cast<CuteMemRefType>(a.getGmemTensor().getType());
  auto gmem = m ? llvm::dyn_cast<LayoutType>(m.getLayout()) : LayoutType();
  return infer(loc, out, [&]() -> Type {
    if (!basis || !gmem || !cg::holds_vector(basis.getRef().stride())) return {};
    const cg::stride &modes = basis.getRef().stride();
    // Each box mode's one basis: the gmem leaf it runs over.
    std::vector<std::vector<int>> paths;
    for (size_t j = 0; j < cg::rank(modes); ++j) {
      auto flat = cg::flatten(modes[j]);
      const cg::stride &leaf = cg::holds_vector(flat) && cg::rank(flat) == 1 ? flat[0] : flat;
      if (!cg::holds_scaled_basis(leaf)) return {};
      paths.push_back(std::get<cg::scaled_basis>(leaf).modes());
    }
    std::vector<std::string> iter(paths.size()), strides;
    std::vector<int> path;
    const cg::layout &g = gmem.getRef();
    walkLeaves(g.shape(), g.stride(), g.stride(), path,
               [&](const auto &s, const auto &, const auto &, const std::vector<int> &p) {
                 auto j = std::find(paths.begin(), paths.end(), p) - paths.begin();
                 if (j == static_cast<long>(paths.size())) return strides.push_back("0");
                 iter[j] = cg::to_string(s);
                 strides.push_back("-1@" + std::to_string(j));
               });
    for (const std::string &e : iter)
      if (e.empty()) return {};
    size_t i = 0;
    auto layout = cg::from_string<cg::layout>(cg::to_string(g.shape()) + ":" +
                                              nestLike(g.shape(), strides, i));
    auto start = cg::from_string<cg::int_tuple>("(" + llvm::join(iter, ",") + ")");
    if (!layout || !start) return {};
    return CoordTensorType::get(ctx, IntTupleType::get(ctx, *start), LayoutType::get(ctx, *layout));
  });
}

LogicalResult AtomGetCoordTensorOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> loc, ValueRange, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &) {
  return unrecovered(loc, "cute_nvgpu.atom.get_coord_tensor");
}

// The im2col and gather/scatter TMA atoms and the circular descriptors
// validate inputs (canonical UMMA layouts, im2col corners) whose rules are
// not recovered.
#define CUTE_UNRECOVERED(Op, name)                                             \
  LogicalResult Op::inferReturnTypes(                                          \
      MLIRContext *, std::optional<Location> loc, ValueRange, DictionaryAttr,  \
      PropertyRef, RegionRange, SmallVectorImpl<Type> &) {                     \
    return unrecovered(loc, name);                                             \
  }
CUTE_UNRECOVERED(AtomCopyMakeNonExec2DGather4TmaLoadOp, "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load")
CUTE_UNRECOVERED(AtomCopyMakeNonExec2DScatter4TmaStoreOp, "cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store")
CUTE_UNRECOVERED(AtomCopyMakeNonExecIm2ColTmaLoadOp, "cute_nvgpu.atom.make_non_exec_im2col_tma_load")
CUTE_UNRECOVERED(AtomCopyMakeNonExecIm2ColTmaStoreOp, "cute_nvgpu.atom.make_non_exec_im2col_tma_store")
CUTE_UNRECOVERED(MakeUMMASmemDescCircularSM103Op, "cute_nvgpu.sm103.make_umma_smem_desc_circular")
CUTE_UNRECOVERED(MakeUMMASmemDescCircularSM107Op, "cute_nvgpu.sm107.make_umma_smem_desc_circular")
#undef CUTE_UNRECOVERED

}  // namespace mlir::cutlass_compiler::cute_nvgpu
