//===- CudaDialect.td - NVIDIA's CUDA host dialect of the CuTe DSL -*- tablegen -*-===//

#ifndef CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_TD
#define CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Cuda_Dialect : Dialect {
  let name = "cuda";
  let cppNamespace = "::mlir::cutlass_compiler::cuda";
  let summary = "NVIDIA's CUDA host-program dialect used by CuTe-DSL";
  let description = [{
    This dialect mirrors the operation, type, attribute and enum registry of
    the proprietary `cuda` dialect embedded in the nvidia-cutlass-dsl 4.8.0
    compiler: kernels, launch configurations and the CUDA runtime and driver
    calls a host function makes.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Cuda_Op<string mnemonic, list<Trait> traits = []>
    : Op<Cuda_Dialect, mnemonic, traits>;

class Cuda_Type<string name, string typeMnemonic>
    : TypeDef<Cuda_Dialect, name> {
  let mnemonic = typeMnemonic;
}

class Cuda_Attr<string name, string attrMnemonic>
    : AttrDef<Cuda_Dialect, name> {
  let mnemonic = attrMnemonic;
}

#endif // CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_TD
