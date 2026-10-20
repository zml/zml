// The cute dialect's enums, generated from the nvidia-cutlass-dsl 4.8.0
// enum bindings (`_cute_enum_gen.py`); do not edit.
//
// Case values and spellings are the compiler's. An enum the bindings build
// as a dialect attribute also has its EnumAttr, printed `#<mnemonic><case>`.

#ifndef CUTE_IR_GENERATED_CUTE_ENUMS_TD
#define CUTE_IR_GENERATED_CUTE_ENUMS_TD

include "cute_ir/Dialect/Cute/IR/CuteDialect.td"
include "mlir/IR/EnumAttr.td"

def Cute_AddressSpaceEnum : I32EnumAttr<"AddressSpace", "Address spaces for CuTe memrefs and pointers", [
    I32EnumAttrCase<"generic", 0, "generic">,
    I32EnumAttrCase<"gmem", 1, "gmem">,
    I32EnumAttrCase<"smem", 3, "smem">,
    I32EnumAttrCase<"rmem", 5, "rmem">,
    I32EnumAttrCase<"tmem", 6, "tmem">,
    I32EnumAttrCase<"dsmem", 7, "dsmem">,
    I32EnumAttrCase<"cmem", 4, "cmem">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute";
  let genSpecializedAttr = 0;
}

def Cute_CopyOperandEnum : I32EnumAttr<"CopyOperand", "Copy operand ID", [
    I32EnumAttrCase<"S", 0, "S">,
    I32EnumAttrCase<"D", 1, "D">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute";
  let genSpecializedAttr = 0;
}

def Cute_MajorModeEnum : I32EnumAttr<"MajorMode", "Major mode for MMA operations", [
    I32EnumAttrCase<"k", 0, "k">,
    I32EnumAttrCase<"mn", 1, "mn">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute";
  let genSpecializedAttr = 0;
}

def Cute_MmaOperandEnum : I32EnumAttr<"MmaOperand", "MMA operand ID", [
    I32EnumAttrCase<"A", 0, "A">,
    I32EnumAttrCase<"B", 1, "B">,
    I32EnumAttrCase<"C", 2, "C">,
    I32EnumAttrCase<"D", 3, "D">,
    I32EnumAttrCase<"SFA", 4, "SFA">,
    I32EnumAttrCase<"SFB", 5, "SFB">,
    I32EnumAttrCase<"E", 6, "E">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute";
  let genSpecializedAttr = 0;
}

def Cute_ReductionOpEnum : I32EnumAttr<"ReductionOp", "Op for cute reduce operations", [
    I32EnumAttrCase<"ADD", 0, "ADD">,
    I32EnumAttrCase<"MUL", 1, "MUL">,
    I32EnumAttrCase<"MIN", 2, "MIN">,
    I32EnumAttrCase<"MAX", 3, "MAX">
  ]> {
  let cppNamespace = "::mlir::cutlass_compiler::cute";
  let genSpecializedAttr = 0;
}

def Cute_ReductionOpAttr : EnumAttr<Cute_Dialect, Cute_ReductionOpEnum, "reduction_op"> {
  let cppNamespace = "::mlir::cutlass_compiler::cute";
  let assemblyFormat = "`<` $value `>`";
}

#endif // CUTE_IR_GENERATED_CUTE_ENUMS_TD
