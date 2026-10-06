#ifndef ZML_MLIR_DIALECTS_TCL_TCL_OPS
#define ZML_MLIR_DIALECTS_TCL_TCL_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/dialects/tcl/tcl_attrs.td"
include "mlir/dialects/tcl/tcl_dialect.td"
include "mlir/dialects/tcl/tcl_types.td"

class Tcl_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tcl_Dialect, mnemonic, traits>;

// Options contain typed attributes, never snippets of TCL source. Their exact
// per-operation schema is verified in tcl_ops.cc. Generic assembly is
// intentional: all native constructs support standard MLIR parsing, printing
// and bytecode.
class Tcl_GraphOp<string name, list<Trait> traits = [MemoryEffects<[MemRead]>]>
    : Tcl_Op<!strconcat("graph.", name), traits> {
  let arguments = (ins Variadic<AnyType>:$inputs,
                       DefaultValuedAttr<DictionaryAttr, "{}">:$options,
                       OptionalAttr<Tcl_Context>:$context);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasVerifier = 1;
}
def Tcl_GraphGatherOp : Tcl_GraphOp<"gather">;
def Tcl_GraphAllGatherOp : Tcl_GraphOp<"all_gather">;
def Tcl_GraphScatterOp : Tcl_GraphOp<"scatter">;
def Tcl_GraphReshapeOp : Tcl_GraphOp<"reshape">;
def Tcl_GraphTransmuteOp : Tcl_GraphOp<"transmute">;

def Tcl_GraphConcatOp : Tcl_GraphOp<"concat">;

def Tcl_GraphSliceOp : Tcl_GraphOp<"slice">;
def Tcl_GraphArangeOp : Tcl_GraphOp<"arange">;
def Tcl_GraphVectorOp : Tcl_GraphOp<"vector">;

def Tcl_GraphAsLogicalOp : Tcl_GraphOp<"as_logical">;
def Tcl_GraphAsDramOp : Tcl_GraphOp<"as_dram">;

// Loop-level graph constructs. `x[i]` reads and `acc[i] = v` writes the slice
// of the outermost axis at the loop index; `sym_expr` evaluates an integer
// expression (`#tcl.expr<arg, ...>` names its operands) on the device.
def Tcl_GraphIndexReadOp : Tcl_GraphOp<"index_read">;
def Tcl_GraphIndexWriteOp : Tcl_GraphOp<"index_write">;
def Tcl_GraphScratchpadOp : Tcl_GraphOp<"scratchpad">;
def Tcl_GraphFullOp : Tcl_GraphOp<"full">;
def Tcl_GraphReduceMaxI32Op : Tcl_GraphOp<"reduce_max_i32">;
def Tcl_GraphSymExprOp : Tcl_GraphOp<"sym_expr">;

// `outputs = for i in 0..limit (acc = init, ...): body; yield ...;`. The limit
// is a static bound (integer or axis) or a scalar i32 tensor computed in the
// function. The block arguments are the loop index, then the accumulators.
def Tcl_GraphForOp : Tcl_Op<"graph.for",
    [AttrSizedOperandSegments, RecursiveMemoryEffects, SingleBlock]> {
  let arguments = (ins Optional<AnyType>:$limit, Variadic<AnyType>:$inits,
                       OptionalAttr<AnyAttr>:$bound);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}
def Tcl_GraphYieldOp : Tcl_Op<"graph.yield",
    [Pure, Terminator, HasParent<"GraphForOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
}

def Tcl_KernelOp : Tcl_Op<"kernel", [MemoryEffects<[MemRead]>, SingleBlock]> {
  let arguments = (ins Variadic<AnyType>:$inputs, Tcl_Tactic:$tactic,
                       OptionalAttr<Tcl_Context>:$context);
  let results = (outs Tcl_Logical:$output);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}
def Tcl_ReadOp : Tcl_Op<"read", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs,
                       OptionalAttr<Tcl_ReadOptions>:$options);
  let results = (outs Tcl_Logical:$output);
  let hasVerifier = 1;
}
def Tcl_DpeOp : Tcl_Op<"dpe", [Pure]> {
  let arguments = (ins Tcl_Logical:$lhs, Tcl_Logical:$rhs);
  let results = (outs Tcl_Logical:$output);
  let hasVerifier = 1;
}
def Tcl_VeOp : Tcl_Op<"ve", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$inputs, Tcl_VeOpcode:$opcode);
  let results = (outs Tcl_Logical:$output);
  let hasVerifier = 1;
}
def Tcl_VeReduceOp : Tcl_Op<"ve_reduce", [Pure]> {
  let arguments = (ins Tcl_Logical:$input, Tcl_ReduceMode:$mode,
                       ArrayAttr:$axes);
  let results = (outs Tcl_Logical:$output);
  let hasVerifier = 1;
}

def Tcl_VeSelectOp : Tcl_Op<"ve_select", [Pure]> {
  let arguments = (ins Tcl_Logical:$condition, Tcl_Predicate:$predicate,
                       AnyAttr:$threshold, AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs Tcl_Logical:$output);
  let hasVerifier = 1;
}

def Tcl_WriteOp : Tcl_Op<"write", [Pure, Terminator]> {
  let arguments = (ins Tcl_Logical:$input);
  let hasVerifier = 1;
}
#endif
