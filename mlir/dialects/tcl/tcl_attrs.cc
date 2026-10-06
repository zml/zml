#include "mlir/dialects/tcl/tcl_attrs.h"

#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/dialects/tcl/tcl_dialect.h"
#include "mlir/dialects/tcl/tcl_types.h"
namespace xla::furiosa::tcl {
using namespace mlir;  // NOLINT
namespace {
using Error = llvm::function_ref<InFlightDiagnostic()>;
bool Identifier(llvm::StringRef s) {
  if (s.empty() || (!llvm::isAlpha(s.front()) && s.front() != '_'))
    return false;
  return llvm::all_of(s, [](char c) { return llvm::isAlnum(c) || c == '_'; });
}
bool Symbol(llvm::StringRef s) {
  auto [base, view] = s.split('.');
  return Identifier(base) &&
         (view.empty() ? !s.contains('.') : Identifier(view));
}
bool Mapping(Attribute a) {
  if (!a) return false;
  if (isa<IntegerAttr>(a)) return IsNumber(a);
  if (isa<SymbolAttr>(a)) return true;
  auto e = dyn_cast<ExprAttr>(a);
  return e &&
         llvm::is_contained(
             {"stride", "modulo", "padding", "resize", "pair", "broadcast"},
             e.getKind()) &&
         llvm::all_of(e.getArgs(), Mapping);
}
bool Json(Attribute a) {
  if (auto n = dyn_cast<FloatAttr>(a)) return n.getValue().isFinite();
  if (IsNumber(a) || isa<StringAttr, UnitAttr>(a)) return true;
  if (auto arr = dyn_cast<ArrayAttr>(a)) return llvm::all_of(arr, Json);
  if (auto dict = dyn_cast<DictionaryAttr>(a))
    return llvm::all_of(dict,
                        [](NamedAttribute f) { return Json(f.getValue()); });
  return false;
}
LogicalResult Enum(Error error, llvm::StringRef value,
                   llvm::ArrayRef<llvm::StringRef> values) {
  return llvm::is_contained(values, value)
             ? success()
             : error() << "invalid enumerator: " << value;
}
}  // namespace

bool IsNumber(Attribute a) {
  if (!a) return false;
  if (auto i = dyn_cast<IntegerAttr>(a))
    return i.getValue().getBitWidth() <= 64;
  return isa<FloatAttr>(a);
}
bool IsExpression(Attribute a) {
  if (!a) return false;
  if (isa<IntegerAttr>(a)) return IsNumber(a);
  if (isa<SymbolAttr>(a)) return true;
  if (auto e = dyn_cast<ExprAttr>(a))
    return e.getKind() != "broadcast" && e.getKind() != "pair" &&
           e.getKind() != "padding" && e.getKind() != "resize";
  return false;
}
bool IsAxis(Attribute a) {
  if (!a) return false;
  auto s = dyn_cast<SymbolAttr>(a);
  return s && llvm::isUpper(s.getName().front());
}
bool IsAxes(Attribute a) {
  if (!a) return false;
  auto array = dyn_cast<ArrayAttr>(a);
  return array && llvm::all_of(array, IsAxis);
}
LogicalResult VerifyKeys(Error error, DictionaryAttr d,
                         llvm::ArrayRef<llvm::StringRef> keys) {
  for (auto f : d)
    if (!llvm::is_contained(keys, f.getName().strref()))
      return error() << "unknown field: " << f.getName();
  return success();
}
LogicalResult VerifyAxes(Error error, ArrayAttr axes) {
  if (!IsAxes(axes)) return error() << "axes must be uppercase TCL symbols";
  llvm::SmallDenseSet<Attribute, 8> seen;
  for (auto a : axes)
    if (!seen.insert(a).second) return error() << "duplicate logical axis";
  return success();
}

// opcode, emitted spelling, arity. Conversion widths are handled below.
static constexpr VeInstructionSpec kVeInstructions[] = {
#define VE(name, text, arity) {name, text, arity},
#include "mlir/dialects/tcl/ve_instructions.inc"
#undef VE
};
llvm::ArrayRef<VeInstructionSpec> VeInstructions() { return kVeInstructions; }

LogicalResult SymbolAttr::verify(Error error, llvm::StringRef name) {
  return Symbol(name) ? success()
                      : error() << "expected a TCL identifier, optionally with "
                                   "one view suffix";
}
LogicalResult ExprAttr::verify(Error error, llvm::StringRef kind,
                               ArrayAttr args) {
  if (kind == "broadcast")
    return args.empty() ? success() : error() << "broadcast takes no arguments";
  if (kind == "arg") {
    if (args.size() != 1 || !isa<IntegerAttr>(args[0]) || !IsNumber(args[0]) ||
        cast<IntegerAttr>(args[0]).getInt() < 0)
      return error() << "arg requires one nonnegative operand index";
    return success();
  }
  bool map = llvm::is_contained(
      {"stride", "modulo", "padding", "resize", "pair"}, kind);
  bool arithmetic =
      llvm::is_contained({"add", "sub", "mul", "div", "rem", "exact_div", "eq",
                          "ne", "lt", "le", "gt", "ge"},
                         kind);
  if ((!map && !arithmetic) || args.size() != 2)
    return error() << "expected a known binary expression";
  for (auto a : args)
    if (!(map ? Mapping(a) : IsExpression(a)))
      return error() << "invalid expression operand";
  if (llvm::is_contained({"div", "exact_div", "rem", "stride", "modulo"}, kind))
    if (auto n = dyn_cast<IntegerAttr>(args[1]); n && n.getInt() == 0)
      return error() << "division by zero";
  return success();
}
LogicalResult TacticAttr::verify(Error e, llvm::StringRef v) {
  return Enum(e, v,
              {"EinsumByDpe", "ReduceByVe", "Interleaving", "Elementwise",
               "TensorOperation", "EinsumByVe", "FilterCompaction"});
}
LogicalResult VeOpcodeAttr::verify(Error e, llvm::StringRef v) {
  for (const auto& spec : VeInstructions())
    if (spec.name == v) return success();
  if (v.consume_front("to_f") || v.consume_front("to_i")) {
    unsigned width;
    if (!v.getAsInteger(10, width) && width <= 31 && std::to_string(width) == v)
      return success();
  }
  return e() << "unknown VE opcode";
}
LogicalResult ReduceModeAttr::verify(Error e, llvm::StringRef v) {
  return Enum(e, v, {"Addi", "Addf", "Maxi", "Maxf", "Mini", "Minf", "Cumsum"});
}

LogicalResult PredicateAttr::verify(Error e, llvm::StringRef v) {
  return Enum(e, v, {"eq", "ne", "lt", "le", "gt", "ge"});
}
LogicalResult ConfigAttr::verify(Error e, DictionaryAttr d) {
  return Json(d) ? success()
                 : e() << "compiler/auto configuration requires "
                          "JSON-compatible attributes";
}

LogicalResult ReadOptionsAttr::verify(Error e, DictionaryAttr d) {
  if (failed(VerifyKeys(e, d,
                        {"subtraction", "table_lookup", "broadcast_to",
                         "typecast_to", "pad", "slide"})))
    return failure();
  for (auto field : d) {
    auto k = field.getName().strref();
    auto v = field.getValue();
    if (k == "subtraction" || k == "table_lookup") {
      auto i = dyn_cast<IntegerAttr>(v);
      if (!i || !IsNumber(i) || i.getInt() < 1)
        return e() << k << " must index an auxiliary read operand (>=1)";
    } else if (k == "broadcast_to") {
      if (!IsAxes(v)) return e() << "broadcast_to requires axes";
    } else if (k == "typecast_to") {
      auto t = dyn_cast<TypeAttr>(v);
      if (!t || ElementName(t.getValue()).empty())
        return e() << "invalid typecast element type";
    } else if (k == "slide") {
      auto entries = dyn_cast<DictionaryAttr>(v);
      if (!entries) return e() << "slide requires an axis dictionary";
      for (auto item : entries) {
        auto spec = dyn_cast<DictionaryAttr>(item.getValue());
        if (!Symbol(item.getName()) ||
            !llvm::isUpper(item.getName().strref().front()) || !spec ||
            failed(VerifyKeys(e, spec,
                              {"undilated_window", "frame_axis", "window_axis",
                               "stride", "dilation"})) ||
            !IsExpression(spec.get("undilated_window")) ||
            !IsAxis(spec.get("frame_axis")) || !IsAxis(spec.get("window_axis")))
          return e() << "slide requires undilated_window, frame_axis and "
                        "window_axis";
        for (auto key : {"stride", "dilation"})
          if (auto a = spec.get(key); a && !IsExpression(a))
            return e() << "invalid slide " << key;
      }
    } else {
      auto entries = dyn_cast<DictionaryAttr>(v);
      if (!entries) return e() << k << " requires an axis dictionary";
      for (auto item : entries) {
        if (!Symbol(item.getName()) ||
            !llvm::isUpper(item.getName().strref().front()))
          return e() << "invalid axis";
        auto p = dyn_cast<ArrayAttr>(item.getValue());
        if (!p || p.size() != 3 || !IsExpression(p[0]) || !IsExpression(p[1]) ||
            !IsNumber(p[2]))
          return e() << "padding requires [left, right, fill]";
      }
    }
  }
  return success();
}
LogicalResult ContextAttr::verify(Error e, DictionaryAttr d) {
  if (failed(VerifyKeys(e, d, {"operator", "heuristic_hint"})))
    return failure();
  if (auto a = d.get("operator")) {
    auto layout = dyn_cast<DictionaryAttr>(a);
    if (!layout || failed(VerifyKeys(e, layout, {"Chip", "Cluster", "Split"})))
      return e() << "invalid operator layout";
    if (!layout.get("Chip")) return e() << "operator layout requires Chip";
    for (auto f : layout) {
      if (f.getName() == "Split") {
        if (!IsAxes(f.getValue())) return e() << "Split requires axes";
      } else if (!IsAxis(f.getValue()) &&
                 !(isa<ExprAttr>(f.getValue()) &&
                   cast<ExprAttr>(f.getValue()).getKind() == "broadcast"))
        return e() << "Chip/Cluster requires axis or Broadcast";
    }
  }
  if (auto a = d.get("heuristic_hint")) {
    auto hints = dyn_cast<DictionaryAttr>(a);
    if (!hints) return e() << "heuristic_hint requires a dictionary";
    for (auto f : hints)
      if (!Identifier(f.getName()) ||
          !(IsNumber(f.getValue()) || IsAxis(f.getValue())))
        return e() << "invalid heuristic hint";
  }
  return success();
}

LogicalResult DramMappingAttr::verify(Error e, ArrayAttr chip, ArrayAttr inner,
                                      ArrayAttr original) {
  if (!llvm::all_of(chip, Mapping) || !llvm::all_of(inner, Mapping))
    return e() << "invalid DRAM mapping";
  return VerifyAxes(e, original);
}
}  // namespace xla::furiosa::tcl
