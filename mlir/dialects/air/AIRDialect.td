//===- AIRDialect.td - Metal AIR dialect -------------------*- tablegen -*-===//

#ifndef MLIR_DIALECTS_AIR_AIRDIALECT_TD
#define MLIR_DIALECTS_AIR_AIRDIALECT_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def AIR_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "::mlir::metal::air";
  let summary = "Apple Metal AIR intrinsics and kernel ABI metadata";
  let description = [{
    The Metal-specific part of the LLVM IR Apple's Metal compiler emits (AIR),
    not AMD's asynchronous AIR dialect. Each operation is an `air.*`
    intrinsic named as Apple names it, without the type suffix its operand
    and result types determine: `air.fast_exp %x : f32` is
    `@air.fast_exp.f32`, `air.max.s %a, %b : vector<2xi32>` is
    `@air.max.s.v2i32`. Kernels are `func.func` with `air.kernel`, and their
    arguments carry the `!air.kernel` descriptors as `air.builtin` or
    `air.buffer`. Arithmetic, memory and control flow use the arith, llvm
    and scf dialects; pointers are `!llvm.ptr` in AIR's address spaces.
  }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
  let extraClassDeclaration = [{
    void registerAttributes();

    static constexpr ::llvm::StringLiteral getKernelAttrName() {
      return ::llvm::StringLiteral("air.kernel");
    }
    static constexpr ::llvm::StringLiteral getBuiltinAttrName() {
      return ::llvm::StringLiteral("air.builtin");
    }
    static constexpr ::llvm::StringLiteral getBufferAttrName() {
      return ::llvm::StringLiteral("air.buffer");
    }
    static constexpr ::llvm::StringLiteral getArgNameAttrName() {
      return ::llvm::StringLiteral("air.arg_name");
    }
    static constexpr ::llvm::StringLiteral getFastMathEnableAttrName() {
      return ::llvm::StringLiteral("air.fast_math_enable");
    }

    static constexpr unsigned kPrivateAddressSpace = 0;
    static constexpr unsigned kDeviceAddressSpace = 1;
    static constexpr unsigned kConstantAddressSpace = 2;
    static constexpr unsigned kThreadgroupAddressSpace = 3;
  }];
}

class AIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIR_Dialect, mnemonic, traits>;

#endif // MLIR_DIALECTS_AIR_AIRDIALECT_TD
