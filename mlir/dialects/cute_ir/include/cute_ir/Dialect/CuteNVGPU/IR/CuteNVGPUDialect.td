//===- CuteNVGPUDialect.td - CuTe NVIDIA GPU dialect -------*- tablegen -*-===//

#ifndef CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_TD
#define CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def CuteNVGPU_Dialect : Dialect {
  let name = "cute_nvgpu";
  let cppNamespace = "::mlir::cutlass_compiler::cute_nvgpu";
  let summary = "NVIDIA GPU operations and atom types used by CuTe-DSL";
  let description = [{
    This dialect mirrors the operation and type registry embedded in the
    proprietary CuTe-DSL compiler. Proven types receive concrete TypeDefs;
    parameterized types remain lossless opaque types until their storage and
    verification contracts are fully recovered.
  }];
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type,
                   ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

class CuteNVGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<CuteNVGPU_Dialect, mnemonic, traits>;

class CuteNVGPU_Type<string name, string typeMnemonic>
    : TypeDef<CuteNVGPU_Dialect, name> {
  let mnemonic = typeMnemonic;
}

#endif // CUTE_IR_DIALECT_CUTENVGPU_IR_CUTENVGPU_DIALECT_TD
