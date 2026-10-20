//===- NVVMDialect.td - NVIDIA's NVVM dialect of the CuTe DSL -*- tablegen -*-===//

#ifndef CUTE_IR_DIALECT_NVVM_IR_NVVM_DIALECT_TD
#define CUTE_IR_DIALECT_NVVM_IR_NVVM_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "::mlir::cutlass_compiler::nvvm";
  let summary = "NVIDIA's NVVM dialect as the CuTe-DSL compiler registers it";
  let description = [{
    This dialect mirrors the operation, attribute and enum registry of the
    `nvvm` dialect embedded in the nvidia-cutlass-dsl 4.8.0 compiler, built on
    NVIDIA's own LLVM revision (23.0.0git 271b151610907). It differs from the
    upstream NVVM dialect (`mlir::NVVM`) and is not meant to be registered
    next to it: both are named `nvvm`.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let useDefaultAttributePrinterParser = 1;
}

class NVVM_Op<string mnemonic, list<Trait> traits = []>
    : Op<NVVM_Dialect, mnemonic, traits>;

class NVVM_Attr<string name, string attrMnemonic>
    : AttrDef<NVVM_Dialect, name> {
  let mnemonic = attrMnemonic;
}

#endif // CUTE_IR_DIALECT_NVVM_IR_NVVM_DIALECT_TD
