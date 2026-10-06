#include "mlir/dialects/tcl/tcl_ops.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/IR/Matchers.h"
#define GET_OP_CLASSES
#include "mlir/dialects/tcl/tcl_ops.cc.inc"
namespace xla::furiosa::tcl {
using namespace mlir;  // NOLINT
namespace {
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
LogicalResult InKernel(Operation* op) {
  if (!op->getParentOfType<KernelOp>())
    return op->emitOpError("requires a tcl.kernel region");
  return success();
}
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
  if (failed(VerifyKeys(e, options, fields))) return failure();
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
      ok = IsAxis(v);
    else if (k == "axes")
      ok = IsAxes(v);
    else if (k == "value")
      ok = IsNumber(v);
    else if (k == "values") {
      auto a = dyn_cast<ArrayAttr>(v);
      ok = a && !a.empty() && llvm::all_of(a, IsNumber);
    } else if (k == "table_dram_layout")
      ok = isa<TypeAttr>(v) && isa<MappedType>(cast<TypeAttr>(v).getValue());
    else
      ok = IsExpression(v);
    if (!ok) return e() << "invalid field " << k;
  }
  for (auto k : fields) {
    bool required = llvm::is_contained(
        {"axis", "offset", "end", "values", "expr", "value"}, k);
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
      if (!ScalarI32(v.getType())) return e() << "requires scalar i32 operands";
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
          auto it =
              llvm::find(axes, SymbolAttr::get(getContext(), slide.getName()));
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
      failed(VerifyAxes([&] { return emitOpError(); }, getAxes())))
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
  if (failed(InKernel(*this)) || !IsNumber(getThreshold()))
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
LogicalResult GraphFullOp::verify() { return VerifyGraph(*this, getOptions()); }
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
    if (!IsAxis(bound) && !(n && IsNumber(n) && n.getInt() > 0))
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
