#ifndef ZML_MLIR_DIALECTS_TCL_TCL
#define ZML_MLIR_DIALECTS_TCL_TCL

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Native TCL: tensors over named axes, `tk.*` tensor operations and graph
// operations, one-to-one with the TCL source the Furiosa compiler takes.
def Tcl_Dialect : Dialect {
  let name = "tcl";
  let cppNamespace = "::xla::furiosa::tcl";
  let summary = "Typed Furiosa Tensor Compute Language graph";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Tcl_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tcl_Dialect, mnemonic, traits>;

def Tcl_Symbol : AttrDef<Tcl_Dialect, "Symbol"> {
  let mnemonic = "symbol";
  let parameters = (ins StringRefParameter<>:$name);
  let assemblyFormat = "`<` $name `>`";
  let genVerifyDecl = 1;
}

def Tcl_Expr : AttrDef<Tcl_Dialect, "Expr"> {
  let mnemonic = "expr";
  let summary = "Checked symbolic arithmetic or DRAM mapping expression tree";
  let parameters = (ins StringRefParameter<>:$kind, "::mlir::ArrayAttr":$args);
  let assemblyFormat = "`<` $kind `,` $args `>`";
  let genVerifyDecl = 1;
}

def Tcl_Tactic : AttrDef<Tcl_Dialect, "Tactic"> {
  let mnemonic = "tactic";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Tcl_VeOpcode : AttrDef<Tcl_Dialect, "VeOpcode"> {
  let mnemonic = "ve_opcode";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Tcl_ReduceMode : AttrDef<Tcl_Dialect, "ReduceMode"> {
  let mnemonic = "reduce_mode";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Tcl_Predicate : AttrDef<Tcl_Dialect, "Predicate"> {
  let mnemonic = "predicate";
  let parameters = (ins StringRefParameter<>:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

def Tcl_Context : AttrDef<Tcl_Dialect, "Context"> {
  let mnemonic = "context";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}

def Tcl_ReadOptions : AttrDef<Tcl_Dialect, "ReadOptions"> {
  let mnemonic = "read_options";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}

def Tcl_Config : AttrDef<Tcl_Dialect, "Config"> {
  let mnemonic = "config";
  let parameters = (ins "::mlir::DictionaryAttr":$fields);
  let assemblyFormat = "`<` $fields `>`";
  let genVerifyDecl = 1;
}

def Tcl_DramMapping : AttrDef<Tcl_Dialect, "DramMapping"> {
  let mnemonic = "dram_mapping";
  let parameters = (ins "::mlir::ArrayAttr":$chip, "::mlir::ArrayAttr":$inner,
                         "::mlir::ArrayAttr":$original);
  let assemblyFormat = "`<` $chip `,` $inner `,` $original `>`";
  let genVerifyDecl = 1;
}

def Tcl_Logical : TypeDef<Tcl_Dialect, "Logical"> {
  let mnemonic = "logical";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::ArrayAttr":$axes);
  let assemblyFormat = "`<` $elementType `,` $axes `>`";
  let genVerifyDecl = 1;
}

def Tcl_Mapped : TypeDef<Tcl_Dialect, "Mapped"> {
  let mnemonic = "mapped";
  let parameters = (ins "::mlir::Type":$elementType,
                        "::xla::furiosa::tcl::DramMappingAttr":$mapping);
  let assemblyFormat = "`<` $elementType `,` $mapping `>`";
  let genVerifyDecl = 1;
}

// Options contain typed attributes, never snippets of TCL source. Their exact
// per-operation schema is verified in native.cc. Generic assembly is intentional:
// all native constructs support standard MLIR parsing, printing and bytecode.
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
