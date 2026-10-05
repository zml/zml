#include "mlir/dialects/tcl/tcl.h"

#include <cctype>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/dialects/tcl/tcl_dialect.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/tcl/tcl_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/tcl/tcl_types.cc.inc"
#define GET_OP_CLASSES
#include "mlir/dialects/tcl/tcl_ops.cc.inc"

namespace xla::furiosa::tcl {
using namespace mlir;  // NOLINT

void TclDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/dialects/tcl/tcl_attrs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/dialects/tcl/tcl_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/dialects/tcl/tcl_ops.cc.inc"
      >();
}

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

bool Number(Attribute a) {
  if (!a) return false;
  if (auto i = dyn_cast<IntegerAttr>(a))
    return i.getValue().getBitWidth() <= 64;
  return isa<FloatAttr>(a);
}

bool Expression(Attribute a) {
  if (!a) return false;
  if (isa<IntegerAttr>(a)) return Number(a);
  if (isa<SymbolAttr>(a)) return true;
  if (auto e = dyn_cast<ExprAttr>(a))
    return e.getKind() != "broadcast" && e.getKind() != "pair" &&
           e.getKind() != "padding" && e.getKind() != "resize";
  return false;
}

bool Axis(Attribute a) {
  if (!a) return false;
  auto s = dyn_cast<SymbolAttr>(a);
  return s && llvm::isUpper(s.getName().front());
}

bool Axes(Attribute a) {
  if (!a) return false;
  auto array = dyn_cast<ArrayAttr>(a);
  return array && llvm::all_of(array, Axis);
}

bool Mapping(Attribute a) {
  if (!a) return false;
  if (isa<IntegerAttr>(a)) return Number(a);
  if (isa<SymbolAttr>(a)) return true;
  auto e = dyn_cast<ExprAttr>(a);
  return e &&
         llvm::is_contained(
             {"stride", "modulo", "padding", "resize", "pair", "broadcast"},
             e.getKind()) &&
         llvm::all_of(e.getArgs(), Mapping);
}

bool Tensor(Type t) { return isa<LogicalType, MappedType>(t); }
bool ScalarI32(Type t) {
  auto l = dyn_cast<LogicalType>(t);
  return l && l.getAxes().empty() && l.getElementType().isInteger(32);
}
bool LoopIndex(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && arg.getArgNumber() == 0 &&
         isa_and_nonnull<GraphForOp>(arg.getOwner()->getParentOp());
}

LogicalResult Keys(Error error, DictionaryAttr d,
                   llvm::ArrayRef<llvm::StringRef> keys) {
  for (auto f : d)
    if (!llvm::is_contained(keys, f.getName().strref()))
      return error() << "unknown field: " << f.getName();
  return success();
}

LogicalResult CheckAxes(Error error, ArrayAttr axes) {
  if (!Axes(axes)) return error() << "axes must be uppercase TCL symbols";
  llvm::SmallDenseSet<Attribute, 8> seen;
  for (auto a : axes)
    if (!seen.insert(a).second) return error() << "duplicate logical axis";
  return success();
}

bool Json(Attribute a) {
  if (auto n = dyn_cast<FloatAttr>(a)) return n.getValue().isFinite();
  if (Number(a) || isa<StringAttr, UnitAttr>(a)) return true;
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

LogicalResult InKernel(Operation* op) {
  if (!op->getParentOfType<KernelOp>())
    return op->emitOpError("requires a tcl.kernel region");
  return success();
}

LogicalResult SameAxes(Operation* op, Value input, Type output) {
  auto in = dyn_cast<LogicalType>(input.getType());
  auto out = dyn_cast<LogicalType>(output);
  if (!in || !out || in.getAxes() != out.getAxes())
    return op->emitOpError("logical axes must match");
  return success();
}
}  // namespace

llvm::StringRef ElementName(Type t) {
  if (t.isBF16()) return "bf16";
  if (t.isF16()) return "f16";
  if (t.isF32()) return "f32";
  if (t.isF64()) return "f64";
  if (isa<Float4E2M1FNType>(t)) return "f4_e2";
  if (isa<Float8E4M3FNType>(t)) return "f8_e4";
  if (isa<Float8E5M2Type>(t)) return "f8_e5";
  if (auto i = dyn_cast<IntegerType>(t)) {
    if (i.getWidth() == 1 && i.isSignless()) return "bool";
    if (i.isUnsigned()) {
      switch (i.getWidth()) {
        case 8:
          return "u8";
        case 16:
          return "u16";
        case 32:
          return "u32";
        case 64:
          return "u64";
      }
    } else {
      switch (i.getWidth()) {
        case 4:
          return "i4";
        case 8:
          return "i8";
        case 16:
          return "i16";
        case 32:
          return "i32";
        case 64:
          return "i64";
      }
    }
  }
  return {};
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
    if (args.size() != 1 || !isa<IntegerAttr>(args[0]) || !Number(args[0]) ||
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
    if (!(map ? Mapping(a) : Expression(a)))
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
  if (failed(Keys(e, d,
                  {"subtraction", "table_lookup", "broadcast_to", "typecast_to",
                   "pad", "slide"})))
    return failure();
  for (auto field : d) {
    auto k = field.getName().strref();
    auto v = field.getValue();
    if (k == "subtraction" || k == "table_lookup") {
      auto i = dyn_cast<IntegerAttr>(v);
      if (!i || !Number(i) || i.getInt() < 1)
        return e() << k << " must index an auxiliary read operand (>=1)";
    } else if (k == "broadcast_to") {
      if (!Axes(v)) return e() << "broadcast_to requires axes";
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
            failed(Keys(e, spec,
                        {"undilated_window", "frame_axis", "window_axis",
                         "stride", "dilation"})) ||
            !Expression(spec.get("undilated_window")) ||
            !Axis(spec.get("frame_axis")) || !Axis(spec.get("window_axis")))
          return e() << "slide requires undilated_window, frame_axis and "
                        "window_axis";
        for (auto key : {"stride", "dilation"})
          if (auto a = spec.get(key); a && !Expression(a))
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
        if (!p || p.size() != 3 || !Expression(p[0]) || !Expression(p[1]) ||
            !Number(p[2]))
          return e() << "padding requires [left, right, fill]";
      }
    }
  }
  return success();
}

LogicalResult ContextAttr::verify(Error e, DictionaryAttr d) {
  if (failed(Keys(e, d, {"operator", "heuristic_hint"}))) return failure();
  if (auto a = d.get("operator")) {
    auto layout = dyn_cast<DictionaryAttr>(a);
    if (!layout || failed(Keys(e, layout, {"Chip", "Cluster", "Split"})))
      return e() << "invalid operator layout";
    if (!layout.get("Chip")) return e() << "operator layout requires Chip";
    for (auto f : layout) {
      if (f.getName() == "Split") {
        if (!Axes(f.getValue())) return e() << "Split requires axes";
      } else if (!Axis(f.getValue()) &&
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
          !(Number(f.getValue()) || Axis(f.getValue())))
        return e() << "invalid heuristic hint";
  }
  return success();
}

LogicalResult DramMappingAttr::verify(Error e, ArrayAttr chip, ArrayAttr inner,
                                      ArrayAttr original) {
  if (!llvm::all_of(chip, Mapping) || !llvm::all_of(inner, Mapping))
    return e() << "invalid DRAM mapping";
  return CheckAxes(e, original);
}

LogicalResult LogicalType::verify(Error e, Type element, ArrayAttr axes) {
  if (ElementName(element).empty())
    return e() << "unsupported TCL element type";
  return CheckAxes(e, axes);
}

LogicalResult MappedType::verify(Error e, Type element,
                                 DramMappingAttr mapping) {
  return ElementName(element).empty() ? e() << "unsupported TCL element type"
                                      : success();
}

namespace {
// Exact graph operation schema. Optional operands occupy documented positions;
// the options dictionary never changes operand ordering.
LogicalResult VerifyGraph(Operation* op, DictionaryAttr options) {
  auto e = [&] { return op->emitOpError(); };
  llvm::StringRef n = op->getName().getStringRef().drop_front(10);
  int min = 1, max = 1, results = 1;
  llvm::SmallVector<llvm::StringRef> fields;
  if (n == "all_gather") {
    fields = {"axis"};
  } else if (n == "gather" || n == "scatter") {
    min = 2;
    max = 3;
    fields = {"axis", "batch_axis", "table_dram_layout"};
  } else if (n == "concat") {
    max = -1;
  } else if (n == "slice") {
    fields = {"axis", "offset"};
  } else if (n == "arange") {
    min = max = 0;
    fields = {"start", "end", "step"};
  } else if (n == "vector") {
    min = max = 0;
    fields = {"values", "repeat_to"};
  } else if (n == "index_read") {
    min = max = 2;
  } else if (n == "index_write") {
    min = max = 3;
  } else if (n == "scratchpad") {
    min = max = 0;
  } else if (n == "full") {
    min = max = 0;
    fields = {"value"};
  } else if (n == "sym_expr") {
    min = 0;
    max = -1;
    fields = {"expr"};
  }
  if (op->getNumOperands() < min || (max >= 0 && op->getNumOperands() > max) ||
      op->getNumResults() != results)
    return e() << "incorrect operand or result count";
  if (failed(Keys(e, options, fields))) return failure();
  if (n == "gather" && options.get("batch_axis") && op->getNumOperands() == 3)
    return e() << "SDK sparse gather cannot also use batch_axis";
  for (Type t : op->getOperandTypes())
    if (!Tensor(t)) return e() << "expected a native TCL tensor operand";
  for (Type t : op->getResultTypes())
    if (!Tensor(t)) return e() << "expected a native TCL tensor result";
  for (auto f : options) {
    auto k = f.getName().strref();
    auto v = f.getValue();
    bool ok = false;
    if (k.ends_with("axis") || k == "repeat_to")
      ok = Axis(v);
    else if (k == "axes")
      ok = Axes(v);
    else if (k == "value")
      ok = Number(v);
    else if (k == "values") {
      auto a = dyn_cast<ArrayAttr>(v);
      ok = a && !a.empty() && llvm::all_of(a, Number);
    } else if (k == "table_dram_layout")
      ok = isa<TypeAttr>(v) && isa<MappedType>(cast<TypeAttr>(v).getValue());
    else
      ok = Expression(v);
    if (!ok) return e() << "invalid field " << k;
  }
  for (auto k : fields) {
    bool required =
        llvm::is_contained({"axis", "offset", "end", "values", "expr", "value"}, k);
    if (required && !options.get(k))
      return e() << "missing required field " << k;
  }
  if (n == "as_logical" && (!isa<MappedType>(op->getOperand(0).getType()) ||
                            !isa<LogicalType>(op->getResult(0).getType())))
    return e() << "as_logical requires storage -> logical";
  if (n == "as_dram" && (!isa<LogicalType>(op->getOperand(0).getType()) ||
                         !isa<MappedType>(op->getResult(0).getType())))
    return e() << "as_dram requires logical -> mapped";
  if (n == "index_read" || n == "index_write") {
    if (!LoopIndex(op->getOperand(1)))
      return e() << "requires the index of an enclosing loop";
    auto table = dyn_cast<LogicalType>(op->getOperand(0).getType());
    auto slice = dyn_cast<LogicalType>(
        (n == "index_read" ? op->getResult(0) : op->getOperand(2)).getType());
    if (!table || !slice || table.getAxes().empty() ||
        table.getElementType() != slice.getElementType() ||
        table.getAxes().getValue().drop_front() != slice.getAxes().getValue())
      return e() << "slice must be the table without its outermost axis";
    if (n == "index_write" &&
        (op->getResult(0).getType() != table || !op->getOperand(0).hasOneUse()))
      return e() << "writes in place: the table must have no other use";
  }
  if ((n == "reduce_max_i32" || n == "sym_expr") &&
      !ScalarI32(op->getResult(0).getType()))
    return e() << "requires a scalar i32 result";
  if (n == "sym_expr") {
    for (Value v : op->getOperands())
      if (!ScalarI32(v.getType()))
        return e() << "requires scalar i32 operands";
    llvm::SmallVector<Attribute> pending{options.get("expr")};
    while (!pending.empty()) {
      auto x = dyn_cast<ExprAttr>(pending.pop_back_val());
      if (!x) continue;
      if (x.getKind() == "arg") {
        if (cast<IntegerAttr>(x.getArgs()[0]).getInt() >= op->getNumOperands())
          return e() << "expression operand out of range";
        continue;
      }
      pending.append(x.getArgs().begin(), x.getArgs().end());
    }
  }
  return success();
}
}  // namespace

LogicalResult KernelOp::verify() {
  if (getBody().empty() || getBody().front().empty() ||
      getBody().front().getNumArguments())
    return emitOpError("requires one body without block arguments");
  auto write = dyn_cast<WriteOp>(getBody().front().getTerminator());
  if (!write) return emitOpError("requires tcl.write terminator");
  for (auto input : getInputs())
    if (!Tensor(input.getType()))
      return emitOpError("requires native tensor inputs");
  for (Operation& op : getBody().front()) {
    if (!isa<ReadOp, DpeOp, VeOp, VeReduceOp, VeSelectOp, WriteOp>(op) &&
        op.getName().getStringRef() != "arith.constant")
      return emitOpError("unsupported kernel instruction ") << op.getName();
  }
  WalkResult captured = getBody().walk([&](Operation* op) {
    for (Value v : op->getOperands()) {
      if (getBody().isAncestor(v.getParentRegion())) continue;
      if (!isa<ReadOp>(op) || !llvm::is_contained(getInputs(), v)) {
        op->emitOpError(
            "external values must be explicit kernel inputs consumed by read");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (captured.wasInterrupted()) return failure();
  return success();
}

LogicalResult ReadOp::verify() {
  if (failed(InKernel(*this))) return failure();
  if (getInputs().empty() ||
      !llvm::all_of(getInputs(), [](Value v) { return Tensor(v.getType()); }))
    return emitOpError("requires tensor inputs");
  if (getOptions())
    for (auto k : {"subtraction", "table_lookup"})
      if (auto a = getOptions()->getFields().getAs<IntegerAttr>(k);
          a && a.getInt() >= getInputs().size())
        return emitOpError("auxiliary read operand index out of bounds");
  if (auto input = dyn_cast<LogicalType>(getInputs()[0].getType())) {
    SmallVector<Attribute> axes(input.getAxes().begin(), input.getAxes().end());
    Type element = input.getElementType();
    if (getOptions()) {
      auto fields = getOptions()->getFields();
      if (auto table = fields.getAs<IntegerAttr>("table_lookup")) {
        auto t = dyn_cast<LogicalType>(getInputs()[table.getInt()].getType());
        if (!t || t.getAxes().size() != 1)
          return emitOpError("lookup table must be a one-axis logical tensor");
        element = t.getElementType();
      }
      if (auto cast = fields.getAs<TypeAttr>("typecast_to"))
        element = cast.getValue();
      if (auto pads = fields.getAs<DictionaryAttr>("pad"))
        for (auto pad : pads)
          if (!llvm::is_contained(axes,
                                  SymbolAttr::get(getContext(), pad.getName())))
            return emitOpError("padding axis missing from input");
      // A slide replaces its axis with the window and frame axes.
      if (auto slides = fields.getAs<DictionaryAttr>("slide"))
        for (auto slide : slides) {
          auto it = llvm::find(axes,
                               SymbolAttr::get(getContext(), slide.getName()));
          if (it == axes.end())
            return emitOpError("slide axis missing from input");
          auto spec = cast<DictionaryAttr>(slide.getValue());
          *it = spec.get("window_axis");
          axes.insert(it + 1, spec.get("frame_axis"));
        }
      if (auto broadcast = fields.getAs<ArrayAttr>("broadcast_to")) {
        for (auto axis : axes)
          if (!llvm::is_contained(broadcast, axis))
            return emitOpError("broadcast must retain all read axes");
        axes.assign(broadcast.begin(), broadcast.end());
      }
    }
    if (getOutput().getType().getAxes().getValue() !=
            ArrayRef<Attribute>(axes) ||
        getOutput().getType().getElementType() != element)
      return emitOpError("read result disagrees with its modifiers");
  }
  return success();
}

LogicalResult DpeOp::verify() {
  if (failed(InKernel(*this))) return failure();
  auto out = getOutput().getType();
  if (!out.getElementType().isF32() &&
      !out.getElementType().isSignlessInteger(32))
    return emitOpError("DPE accumulators require f32 or i32");
  for (auto axis : out.getAxes())
    if (!llvm::is_contained(getLhs().getType().getAxes(), axis) &&
        !llvm::is_contained(getRhs().getType().getAxes(), axis))
      return emitOpError("DPE result axis missing from both inputs");
  return success();
}

LogicalResult VeOp::verify() {
  if (failed(InKernel(*this))) return failure();
  unsigned arity = 1;
  for (auto spec : VeInstructions())
    if (spec.name == getOpcode().getValue()) arity = spec.arity;
  if (getInputs().size() != arity)
    return emitOpError("incorrect VE operand count");
  llvm::SmallDenseSet<Attribute, 8> axes;
  for (Value v : getInputs()) {
    if (auto type = dyn_cast<LogicalType>(v.getType())) {
      for (auto a : type.getAxes()) axes.insert(a);
    } else if (!isa<IntegerType, FloatType>(v.getType()) ||
               !matchPattern(v, m_Constant()))
      return emitOpError("VE scalar operands must be constants");
  }
  auto out = getOutput().getType();
  if (axes.size() != out.getAxes().size() ||
      !llvm::all_of(out.getAxes(),
                    [&](Attribute a) { return axes.contains(a); }))
    return emitOpError("VE output axes must be the union of operand axes");
  auto code = getOpcode().getValue();
  llvm::StringRef spelling = code;
  for (auto spec : VeInstructions())
    if (spec.name == code) spelling = spec.spelling;
  bool fp = spelling.starts_with("to_f") || spelling == "as_f" ||
            spelling.ends_with("_f") ||
            llvm::is_contained({"+", "-", "*", "/", "exp", "neg_exp", "sqrt",
                                "tanh", "sigmoid", "erf", "log", "sin", "cos"},
                               spelling);
  if (fp ? !out.getElementType().isF32()
         : !out.getElementType().isSignlessInteger(32))
    return emitOpError("VE registers produce f32 or i32 according to opcode");
  return success();
}

LogicalResult VeReduceOp::verify() {
  if (failed(InKernel(*this)) ||
      failed(CheckAxes([&] { return emitOpError(); }, getAxes())))
    return failure();
  // An integer cumulative sum runs along its axes and keeps them.
  if (getMode().getValue() == "Cumsum") {
    if (getAxes().empty() ||
        getInput().getType().getAxes() != getOutput().getType().getAxes() ||
        !llvm::all_of(getAxes(),
                      [&](Attribute a) {
                        return llvm::is_contained(
                            getInput().getType().getAxes(), a);
                      }) ||
        !getInput().getType().getElementType().isInteger(32))
      return emitOpError("cumsum keeps the axes of an i32 value");
    return success();
  }
  llvm::SmallVector<Attribute> remaining;
  for (Attribute a : getInput().getType().getAxes())
    if (!llvm::is_contained(getAxes(), a)) remaining.push_back(a);
  if (getAxes().empty() ||
      remaining.size() + getAxes().size() !=
          getInput().getType().getAxes().size() ||
      remaining != getOutput().getType().getAxes().getValue())
    return emitOpError("reduction result must remove exactly the reduced axes");
  return success();
}

LogicalResult VeSelectOp::verify() {
  if (failed(InKernel(*this)) || !Number(getThreshold()))
    return emitOpError("requires kernel and numeric threshold");
  llvm::SmallDenseSet<Attribute, 8> axes;
  for (Value v :
       SmallVector<Value>{getCondition(), getTrueValue(), getFalseValue()}) {
    if (auto t = dyn_cast<LogicalType>(v.getType())) {
      for (auto a : t.getAxes()) axes.insert(a);
    } else if (!matchPattern(v, m_Constant()))
      return emitOpError("scalar branches require constants");
  }
  auto output = getOutput().getType().getAxes();
  if (axes.size() != output.size() ||
      !llvm::all_of(output, [&](Attribute a) { return axes.contains(a); }))
    return emitOpError("select output axes must cover its operands");
  return success();
}

LogicalResult WriteOp::verify() {
  auto kernel = dyn_cast<KernelOp>((*this)->getParentOp());
  if (!kernel) return emitOpError("requires a kernel parent");
  auto in = getInput().getType(), out = kernel.getOutput().getType();
  if (in.getAxes().size() != out.getAxes().size())
    return emitOpError("write must preserve the axis set");
  for (auto a : in.getAxes())
    if (!llvm::is_contained(out.getAxes(), a))
      return emitOpError("write must preserve the axis set");
  // TCL write.to implicitly consumes the preceding instruction.
  auto previous = (*this)->getPrevNode();
  while (previous && previous->getName().getStringRef() == "arith.constant")
    previous = previous->getPrevNode();
  if (getInput().getDefiningOp() != previous)
    return emitOpError(
        "write must consume the immediately preceding instruction");
  return success();
}

LogicalResult GraphGatherOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphAllGatherOp::verify() {
  if (failed(VerifyGraph(*this, getOptions()))) return failure();
  auto input = dyn_cast<LogicalType>(getInputs()[0].getType());
  auto output = dyn_cast<LogicalType>(getOutputs()[0].getType());
  if (!input || !output || input != output)
    return emitOpError("all_gather requires matching logical tensor types");
  if (!llvm::is_contained(input.getAxes(), getOptions().get("axis")))
    return emitOpError("collective axis must be present in the input");
  return success();
}

LogicalResult GraphScatterOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphReshapeOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphTransmuteOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphConcatOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphSliceOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphArangeOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphVectorOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphAsLogicalOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphAsDramOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphIndexReadOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphIndexWriteOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphScratchpadOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphFullOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphReduceMaxI32Op::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphSymExprOp::verify() {
  return VerifyGraph(*this, getOptions());
}

LogicalResult GraphForOp::verify() {
  if (!getLimit() == !getBound())
    return emitOpError("requires exactly one of a limit tensor or a bound");
  if (getLimit() && !ScalarI32(getLimit().getType()))
    return emitOpError("limit must be a scalar i32 tensor");
  if (auto bound = getBoundAttr()) {
    auto n = dyn_cast<IntegerAttr>(bound);
    if (!Axis(bound) && !(n && Number(n) && n.getInt() > 0))
      return emitOpError("bound must be a positive integer or an axis");
  }
  Block& body = getBody().front();
  if (body.getNumArguments() != getInits().size() + 1 ||
      !ScalarI32(body.getArgument(0).getType()))
    return emitOpError("requires a scalar i32 index and one accumulator each");
  auto yield = dyn_cast<GraphYieldOp>(body.getTerminator());
  if (!yield) return emitOpError("requires a tcl.graph.yield terminator");
  if (getOutputs().size() != getInits().size() ||
      yield.getValues().size() != getInits().size())
    return emitOpError("requires one result and one yield per accumulator");
  for (auto [i, init] : llvm::enumerate(getInits())) {
    Type t = init.getType();
    if (!Tensor(t) || body.getArgument(i + 1).getType() != t ||
        getOutputs()[i].getType() != t || yield.getValues()[i].getType() != t)
      return emitOpError("accumulator ") << i << " changes type";
  }
  return success();
}

}  // namespace xla::furiosa::tcl
