// The nvvm dialect's operations, generated from the nvidia-cutlass-dsl 4.8.0 release; do not edit.
//
// Sources: the operation bindings (`_nvvm_ops_gen.py`: names, classes,
// docstrings, operand/result/attribute structure, inference) and the live
// compiler (`_cutlass_ir` loaded from Python): each operand and result
// constraint carries the verifier's wording and accepts every type class the
// verifier let through when probed with one instance of every type; traits
// and interfaces are the registered operation's; memory effects come from
// loop-invariant code motion (hoisted: Pure) and CSE (merged: at most
// reads). Operations whose effects include writes or could not be told
// apart carry none (unknown effects). The compiler's own verifiers (operand
// combinations, shapes, instruction legality) are left to it.
//
// Operations keep the generic assembly form.
//
// Result types the compiler infers without a recovered rule (written types stand):
//   nvvm.add.packed.f32x2
//   nvvm.atomicrmw
//   nvvm.barrier
//   nvvm.clusterlaunchcontrol.query.cancel
//   nvvm.fma
//   nvvm.fma.packed.f32x2
//   nvvm.ldmatrix
//   nvvm.match.sync
//   nvvm.mbarrier.arrive
//   nvvm.mbarrier.arrive.expect_tx
//   nvvm.mbarrier.arrive_drop
//   nvvm.mbarrier.arrive_drop.expect_tx
//   nvvm.mbarrier.test.wait
//   nvvm.mbarrier.try_wait
//   nvvm.mul (`mode = wide` doubles the operand width)
//   nvvm.mul.packed.f32x2
//   nvvm.redux.sync
//   nvvm.shfl.sync (`return_value_and_is_valid` returns `!llvm.struct<(T, i1)>`)
//   nvvm.sub.packed.f32x2
//   nvvm.vote.sync
//
// Constraints not recovered:
//   nvvm.convert.bf16x2.to.f8x2 `dstTy`: `fnv8E5M3FNU` has no type here and is not accepted

#ifndef CUTE_IR_GENERATED_NVVM_OPS_TD
#define CUTE_IR_GENERATED_NVVM_OPS_TD

include "cute_ir/Dialect/NVVM/IR/NVVMAttrs.td"
include "cute_ir/Dialect/NVVM/IR/NVVMEnums.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Operand and result constraints, in the compiler's words.
def NVVM_AnyType : Type<CPred<"true">, "any type">;
def NVVM_FixedLengthVectorOf32Bit : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 32-bit signless integer values of length 4/8">;
def NVVM_FixedLengthVectorOf8Bit : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 8-bit signless integer values of length 4 or fixed-length vector of 16-bit signless integer values of length 2">;
def NVVM_FloatingPointOrFixedLengthVector : Type<CPred<"::llvm::isa<::mlir::FloatType>($_self) || ::llvm::isa<::mlir::VectorType>($_self)">, "floating-point or fixed-length vector of floating-point values of length 2">;
def NVVM_LLVMDialectCompatibleType : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::Float16Type>($_self) || ::llvm::isa<::mlir::BFloat16Type>($_self) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::VectorType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMArrayType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMFunctionType>($_self)">, "LLVM dialect-compatible type">;
def NVVM_LLVMPointerInAddressSpace0 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 0">;
def NVVM_LLVMPointerInAddressSpace02 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 0 or LLVM pointer in address space 3">;
def NVVM_LLVMPointerInAddressSpace03 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 0 or LLVM pointer in address space 7">;
def NVVM_LLVMPointerInAddressSpace04 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 0 or LLVM pointer in address space 3 or LLVM pointer in address space 7">;
def NVVM_LLVMPointerInAddressSpace05 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 0 or LLVM pointer in address space 3 or LLVM pointer in address space 1 or LLVM pointer in address space 7">;
def NVVM_LLVMPointerInAddressSpace1 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 1">;
def NVVM_LLVMPointerInAddressSpace12 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 1 or LLVM pointer in address space 0 or LLVM pointer in address space 3 or LLVM pointer in address space 7">;
def NVVM_LLVMPointerInAddressSpace13 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 1 or LLVM pointer in address space 5 or LLVM pointer in address space 0 or LLVM pointer in address space 4">;
def NVVM_LLVMPointerInAddressSpace14 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 1 or LLVM pointer in address space 3">;
def NVVM_LLVMPointerInAddressSpace3 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 3">;
def NVVM_LLVMPointerInAddressSpace32 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 3 or LLVM pointer in address space 7">;
def NVVM_LLVMPointerInAddressSpace6 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 6">;
def NVVM_LLVMPointerInAddressSpace62 : Type<CPred<"$_self.isSignlessInteger(64) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 6 or 64-bit signless integer">;
def NVVM_LLVMPointerInAddressSpace7 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 7">;
def NVVM_LLVMPointerInAddressSpace72 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 7 or LLVM pointer in address space 3">;
def NVVM_LLVMPointerType : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer type">;
def NVVM_LLVMPointerTypeOrLLVMPointer : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer type or LLVM pointer in address space 3">;
def NVVM_LLVMStructureType : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)">, "LLVM structure type">;
def NVVM_LLVMStructureTypeOr64Bit : Type<CPred<"::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)">, "LLVM structure type or 64-bit float">;
def NVVM_LLVMTypeWithSize : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::Float16Type>($_self) || ::llvm::isa<::mlir::BFloat16Type>($_self) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::VectorType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMArrayType>($_self)">, "LLVM type with size">;
def NVVM_T128BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(128)">, "128-bit signless integer">;
def NVVM_T16BitFloatOrBfloat16Type : Type<CPred<"::llvm::isa<::mlir::Float16Type>($_self) || ::llvm::isa<::mlir::BFloat16Type>($_self) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::VectorType>($_self)">, "16-bit float or bfloat16 type or 32-bit float or 64-bit float or vector of 16-bit float or bfloat16 type or 32-bit float or 64-bit float values of length 2">;
def NVVM_T16BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(16)">, "16-bit signless integer">;
def NVVM_T16BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(16)">, "16-bit signless integer">, BuildableType<"$_builder.getIntegerType(16)">;
def NVVM_T16BitSignlessIntegerOr32 : Type<CPred<"$_self.isSignlessInteger(16) || $_self.isSignlessInteger(32) || $_self.isSignlessInteger(64) || $_self.isSignlessInteger(128) || ::llvm::isa<::mlir::Float16Type>($_self) || ::llvm::isa<::mlir::BFloat16Type>($_self) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::VectorType>($_self)">, "16-bit signless integer or 32-bit signless integer or 64-bit signless integer or 128-bit signless integer or 16-bit float or bfloat16 type or 32-bit float or 64-bit float or 128-bit float or vector of 32-bit float values of length 2/4 or vector of 16-bit float or bfloat16 type values of length 2/4/8">;
def NVVM_T16BitSignlessIntegerOr322 : Type<CPred<"$_self.isSignlessInteger(16) || $_self.isSignlessInteger(32)">, "16-bit signless integer or 32-bit signless integer">;
def NVVM_T16BitSignlessIntegerOr323 : Type<CPred<"$_self.isSignlessInteger(16) || $_self.isSignlessInteger(32) || $_self.isSignlessInteger(64)">, "16-bit signless integer or 32-bit signless integer or 64-bit signless integer">;
def NVVM_T16BitSignlessIntegerOrVector : Type<CPred<"$_self.isSignlessInteger(16) || ::llvm::isa<::mlir::VectorType>($_self)">, "16-bit signless integer or vector of 8-bit signless integer values of length 2">;
def NVVM_T1BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(1)">, "1-bit signless integer">;
def NVVM_T1BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(1)">, "1-bit signless integer">, BuildableType<"$_builder.getIntegerType(1)">;
def NVVM_T1BitSignlessIntegerOr32 : Type<CPred<"$_self.isSignlessInteger(1) || $_self.isSignlessInteger(32)">, "1-bit signless integer or 32-bit signless integer">;
def NVVM_T1BitSignlessIntegerOrLLVM : Type<CPred<"$_self.isSignlessInteger(1) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)">, "1-bit signless integer or LLVM structure type">;
def NVVM_T32BitFloat : Type<CPred<"::llvm::isa<::mlir::Float32Type>($_self)">, "32-bit float">;
def NVVM_T32BitFloat2 : Type<CPred<"::llvm::isa<::mlir::Float32Type>($_self)">, "32-bit float">, BuildableType<"::mlir::Float32Type::get($_builder.getContext())">;
def NVVM_T32BitFloatOr64Bit : Type<CPred<"::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self)">, "32-bit float or 64-bit float">;
def NVVM_T32BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(32)">, "32-bit signless integer">;
def NVVM_T32BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(32)">, "32-bit signless integer">, BuildableType<"$_builder.getIntegerType(32)">;
def NVVM_T32BitSignlessIntegerOr1 : Type<CPred<"$_self.isSignlessInteger(1) || $_self.isSignlessInteger(32)">, "32-bit signless integer or 1-bit signless integer">;
def NVVM_T32BitSignlessIntegerOr32 : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::Float32Type>($_self)">, "32-bit signless integer or 32-bit float">;
def NVVM_T32BitSignlessIntegerOr322 : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)">, "32-bit signless integer or 32-bit float or LLVM struct type">;
def NVVM_T32BitSignlessIntegerOr64 : Type<CPred<"$_self.isSignlessInteger(32) || $_self.isSignlessInteger(64)">, "32-bit signless integer or 64-bit signless integer">;
def NVVM_T32BitSignlessIntegerOr642 : Type<CPred<"$_self.isSignlessInteger(32) || $_self.isSignlessInteger(64) || ::llvm::isa<::mlir::Float16Type>($_self) || ::llvm::isa<::mlir::BFloat16Type>($_self) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::VectorType>($_self)">, "32-bit signless integer or 64-bit signless integer or 64-bit float or bfloat16 type or 16-bit float or 32-bit float or vector of 32-bit float or 32-bit signless integer values of length 2/4 or vector of 16-bit float or bfloat16 type values of length 2/4/8">;
def NVVM_T32BitSignlessIntegerOr643 : Type<CPred<"$_self.isSignlessInteger(32) || $_self.isSignlessInteger(64) || ::llvm::isa<::mlir::VectorType>($_self)">, "32-bit signless integer or 64-bit signless integer or vector of 32-bit signless integer or 64-bit signless integer values of length 2/4">;
def NVVM_T32BitSignlessIntegerOrLLVM : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self)">, "32-bit signless integer or LLVM struct type">;
def NVVM_T32BitSignlessIntegerOrVector : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::VectorType>($_self)">, "32-bit signless integer or vector of 32-bit signless integer values of length 2/4/8/16/32/64">;
def NVVM_T32BitSignlessIntegerOrVector2 : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::VectorType>($_self)">, "32-bit signless integer or vector of 32-bit signless integer values of length 2/4/8/16/32/64/128">;
def NVVM_T32BitSignlessIntegerOrVector3 : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::VectorType>($_self)">, "32-bit signless integer or vector of 32-bit signless integer values of length 2/4">;
def NVVM_T64BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(64)">, "64-bit signless integer">;
def NVVM_T64BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(64)">, "64-bit signless integer">, BuildableType<"$_builder.getIntegerType(64)">;
def NVVM_T64BitSignlessIntegerOr32 : Type<CPred<"$_self.isSignlessInteger(32) || $_self.isSignlessInteger(64)">, "64-bit signless integer or 32-bit signless integer">;
def NVVM_T8BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(8)">, "8-bit signless integer">, BuildableType<"$_builder.getIntegerType(8)">;
def NVVM_T8BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(8)">, "8-bit signless integer">;
def NVVM_T8BitSignlessIntegerOr32 : Type<CPred<"$_self.isSignlessInteger(8) || $_self.isSignlessInteger(32)">, "8-bit signless integer or 32-bit signless integer">;
def NVVM_VariadicOf16BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(16)">, "variadic of 16-bit signless integer">;
def NVVM_VariadicOf32BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(32)">, "variadic of 32-bit signless integer">;
def NVVM_VariadicOfLLVMDialectCompatibleType : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::Float16Type>($_self) || ::llvm::isa<::mlir::BFloat16Type>($_self) || ::llvm::isa<::mlir::Float32Type>($_self) || ::llvm::isa<::mlir::Float64Type>($_self) || ::llvm::isa<::mlir::VectorType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMArrayType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMFunctionType>($_self)">, "variadic of LLVM dialect-compatible type">;
def NVVM_VectorOf16BitFloatValues : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 16-bit float values of length 2">;
def NVVM_VectorOf16BitSignlessInteger : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 16-bit signless integer values of length 2">;
def NVVM_VectorOf32BitFloatValues : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit float values of length 2">;
def NVVM_VectorOf32BitFloatValues2 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit float values">;
def NVVM_VectorOf32BitFloatValues3 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit float values of length 4">;
def NVVM_VectorOf32BitSignlessInteger : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit signless integer values of length 2/4/8/16/32/64/128">;
def NVVM_VectorOf32BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(32) || ::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit signless integer values of length 2/4/8/16/32/64/128 or 32-bit signless integer">;
def NVVM_VectorOf32BitSignlessInteger3 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit signless integer or 32-bit float values of length 2/4/8/16/32/64/128">;
def NVVM_VectorOf32BitSignlessInteger4 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 32-bit signless integer values of length 2/4/8/16/32/64">;
def NVVM_VectorOf8BitSignlessInteger : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 8-bit signless integer values of length 4">;
def NVVM_VectorOf8BitSignlessInteger2 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of 8-bit signless integer values of length 2">;
def NVVM_VectorOfAnyTypeValues : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of any type values">;
def NVVM_VectorOfBfloat16TypeValuesOf : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "vector of bfloat16 type values of length 2">;

// Attribute constraints the bindings name but do not describe.
def NVVM_ConvertBF16x2ToF4x2DstTyAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float4E2M1FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of f4E2M1FN type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_ConvertBF16x2ToF6x2DstTyAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float6E2M3FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float6E3M2FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of f6E2M3FN type or f6E3M2FN type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_ConvertBF16x2ToF8x2DstTyAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float8E8M0FNUType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of f8E8M0FNU type or f8E4M3FN type or f8E5M2 type or fnv8E5M3FNU type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_ConvertF16x2ToF4x2DstTyAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float4E2M1FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of f4E2M1FN type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_ConvertF16x2ToF6x2DstTyAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float6E2M3FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float6E3M2FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of f6E2M3FN type or f6E3M2FN type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_FmaRndAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::FPRoundingModeAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<::mlir::cutlass_compiler::nvvm::FPRoundingMode>{::mlir::cutlass_compiler::nvvm::FPRoundingMode::NONE, ::mlir::cutlass_compiler::nvvm::FPRoundingMode::RN, ::mlir::cutlass_compiler::nvvm::FPRoundingMode::RM, ::mlir::cutlass_compiler::nvvm::FPRoundingMode::RP, ::mlir::cutlass_compiler::nvvm::FPRoundingMode::RZ}, ::llvm::cast<::mlir::cutlass_compiler::nvvm::FPRoundingModeAttr>($_self).getValue())">, "NVVM FPRoundingMode kind whose value is one of {none, rm, rn, rp, rz}"> {
  let storageType = "::mlir::cutlass_compiler::nvvm::FPRoundingModeAttr";
  let returnType = "::mlir::cutlass_compiler::nvvm::FPRoundingModeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_MultimemLdReduceOrderingAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::MemOrderKindAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<::mlir::cutlass_compiler::nvvm::MemOrderKind>{::mlir::cutlass_compiler::nvvm::MemOrderKind::WEAK, ::mlir::cutlass_compiler::nvvm::MemOrderKind::RELAXED, ::mlir::cutlass_compiler::nvvm::MemOrderKind::ACQUIRE}, ::llvm::cast<::mlir::cutlass_compiler::nvvm::MemOrderKindAttr>($_self).getValue())">, "NVVM Memory Ordering kind whose value is one of {weak, relaxed, acquire}"> {
  let storageType = "::mlir::cutlass_compiler::nvvm::MemOrderKindAttr";
  let returnType = "::mlir::cutlass_compiler::nvvm::MemOrderKindAttr";
  let convertFromStorage = "$_self";
}

def NVVM_MultimemLdReduceOpAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::ReductionOpAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<::mlir::cutlass_compiler::nvvm::ReductionOp>{::mlir::cutlass_compiler::nvvm::ReductionOp::AND, ::mlir::cutlass_compiler::nvvm::ReductionOp::OR, ::mlir::cutlass_compiler::nvvm::ReductionOp::XOR, ::mlir::cutlass_compiler::nvvm::ReductionOp::ADD, ::mlir::cutlass_compiler::nvvm::ReductionOp::MIN, ::mlir::cutlass_compiler::nvvm::ReductionOp::MAX}, ::llvm::cast<::mlir::cutlass_compiler::nvvm::ReductionOpAttr>($_self).getValue())">, "Ops supported by red instruction whose value is none of {inc, dec}"> {
  let storageType = "::mlir::cutlass_compiler::nvvm::ReductionOpAttr";
  let returnType = "::mlir::cutlass_compiler::nvvm::ReductionOpAttr";
  let convertFromStorage = "$_self";
}

def NVVM_MultimemRedOrderingAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::MemOrderKindAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<::mlir::cutlass_compiler::nvvm::MemOrderKind>{::mlir::cutlass_compiler::nvvm::MemOrderKind::RELAXED, ::mlir::cutlass_compiler::nvvm::MemOrderKind::RELEASE}, ::llvm::cast<::mlir::cutlass_compiler::nvvm::MemOrderKindAttr>($_self).getValue())">, "NVVM Memory Ordering kind whose value is one of {relaxed, release}"> {
  let storageType = "::mlir::cutlass_compiler::nvvm::MemOrderKindAttr";
  let returnType = "::mlir::cutlass_compiler::nvvm::MemOrderKindAttr";
  let convertFromStorage = "$_self";
}

def NVVM_MultimemRedOpAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::ReductionOpAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<::mlir::cutlass_compiler::nvvm::ReductionOp>{::mlir::cutlass_compiler::nvvm::ReductionOp::AND, ::mlir::cutlass_compiler::nvvm::ReductionOp::OR, ::mlir::cutlass_compiler::nvvm::ReductionOp::XOR, ::mlir::cutlass_compiler::nvvm::ReductionOp::ADD, ::mlir::cutlass_compiler::nvvm::ReductionOp::INC, ::mlir::cutlass_compiler::nvvm::ReductionOp::DEC, ::mlir::cutlass_compiler::nvvm::ReductionOp::MIN, ::mlir::cutlass_compiler::nvvm::ReductionOp::MAX}, ::llvm::cast<::mlir::cutlass_compiler::nvvm::ReductionOpAttr>($_self).getValue())">, "ReductionOp attribute"> {
  let storageType = "::mlir::cutlass_compiler::nvvm::ReductionOpAttr";
  let returnType = "::mlir::cutlass_compiler::nvvm::ReductionOpAttr";
  let convertFromStorage = "$_self";
}

def NVVM_MultimemStOrderingAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::MemOrderKindAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<::mlir::cutlass_compiler::nvvm::MemOrderKind>{::mlir::cutlass_compiler::nvvm::MemOrderKind::WEAK, ::mlir::cutlass_compiler::nvvm::MemOrderKind::RELAXED, ::mlir::cutlass_compiler::nvvm::MemOrderKind::RELEASE}, ::llvm::cast<::mlir::cutlass_compiler::nvvm::MemOrderKindAttr>($_self).getValue())">, "NVVM Memory Ordering kind whose value is one of {weak, relaxed, release}"> {
  let storageType = "::mlir::cutlass_compiler::nvvm::MemOrderKindAttr";
  let returnType = "::mlir::cutlass_compiler::nvvm::MemOrderKindAttr";
  let convertFromStorage = "$_self";
}

def NVVM_LLVMConstantRangeAttr : Attr<CPred<"::llvm::isa<::mlir::LLVM::ConstantRangeAttr>($_self)">, "A range of two integers, corresponding to LLVM's ConstantRange"> {
  let storageType = "::mlir::LLVM::ConstantRangeAttr";
  let returnType = "::mlir::LLVM::ConstantRangeAttr";
  let convertFromStorage = "$_self";
}

def NVVM_TensormapFieldValueAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::nvvm::TensormapElemtypeAttr, ::mlir::cutlass_compiler::nvvm::TensormapInterleaveLayoutAttr, ::mlir::cutlass_compiler::nvvm::TensormapSwizzleModeAttr, ::mlir::cutlass_compiler::nvvm::TensormapSwizzleAtomicityAttr, ::mlir::cutlass_compiler::nvvm::TensormapFillModeAttr>($_self)">, "NVVM Tensormap Elemtype or NVVM Tensormap Interleave Layout or NVVM Tensormap Swizzle Mode or NVVM Tensormap Swizzle Atomicity or NVVM Tensormap Fill Mode"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
}

// Operations whose result types are inferred in C++ (NVVMInference.cpp).
// Verification compares them with the inferred ones where the rule is
// recovered; elsewhere the types written stand, as the compiler accepted them.
class NVVM_InferredOp<string mnemonic, list<Trait> traits = []>
    : NVVM_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferTypeOpInterface, ["refineReturnTypes"]>]> {
  let extraClassDefinition = [{
    ::llvm::LogicalResult $cppClass::refineReturnTypes(
        ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
        ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
        ::mlir::PropertyRef properties, ::mlir::RegionRange regions,
        ::llvm::SmallVectorImpl<::mlir::Type> &returnTypes) {
      ::llvm::SmallVector<::mlir::Type, 4> inferred;
      if (::mlir::failed(inferReturnTypes(context, ::std::nullopt, operands,
                                          attributes, properties, regions,
                                          inferred)))
        return ::mlir::success();
      if (!isCompatibleReturnTypes(inferred, returnTypes))
        return ::mlir::emitOptionalError(
            location, "'", getOperationName(), "' op inferred type(s) ",
            inferred, " are incompatible with return type(s) of operation ",
            returnTypes);
      return ::mlir::success();
    }
  }];
}

def NVVM_AddPacked_BF16x2_F32x2_F32x2Op : NVVM_Op<"add.packed.bf16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs the add operation on a pair of f32x2 vectors.
    The inputs and output are of the form v2f32 and v2bf16 respectively.
    The inputs are converted to a scalar of packed i64 where the 0th element of the
    vector forms the low-32 bits and the 1st element occupies the high-32 bits. The round to zero
    is the only rounding mode supported for this operation.

    Examples:
    ```mlir
    // Round to zero (only rounding mode supported)
    %0 = nvvm.add .packed.bf16x2.f32x2.f32x2 %srcA, %srcB : vector<2xf32>, vector<2xf32> -> vector<2xbf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$res
  );
}
def NVVM_AddPacked_F16x2_F32x2_F32x2Op : NVVM_Op<"add.packed.f16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs the add operation on a pair of f32x2 vectors.
    The inputs and output are of the form v2f32 and v2f16 respectively.
    The inputs are converted to a scalar of packed i64 where the 0th element of the
    vector forms the low-32 bits and the 1st element occupies the high-32 bits. The round to zero
    is the only rounding mode supported for this operation.

    Examples:
    ```mlir
    // Round to zero (only rounding mode supported)
    %0 = nvvm.add .packed.f16x2.f32x2.f32x2 %srcA, %srcB : vector<2xf32>, vector<2xf32> -> vector<2xf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$res
  );
}
def NVVM_AddPackedF32x2Op : NVVM_InferredOp<"add.packed.f32x2"> {
  let description = [{
    This Op performs addition of floating-point vectors.
    The inputs and outputs are of the form <N x f32>. The `rnd` attribute
    specifies the rounding mode. The `ftz` attribute, when set, flushes the
    subnormal inputs and results to sign-preserving zero.
    This Op is supported from SM100 and onwards.
    [For more information, see PTX ISA]
    (TODO: Add public PTX link when available)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues2:$srcA,
    NVVM_VectorOf32BitFloatValues2:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<I1Attr>:$ftz
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues2:$res
  );
}
def NVVM_AddPacked_F32x2_BF16x2_F32x2Op : NVVM_Op<"add.packed.f32x2.bf16x2.f32x2"> {
  let description = [{
    This Op performs the add operation on a pair of floating-point values.
    The inputs are of the form v2bf16 and v2f32. The first input `srcA` is
    converted to a scalar of packed i32 where the 0th element of the vector forms
    the low-16 bits and the 1st element occupies the high-16 bits. Similarly,
    the second input `srcB` is converted to a scalar of packed i64 where the 0th
    element of the vector forms the low-32 bits and the 1st element occupies the
    high-32 bits. The `rnd` attribute specifies the rounding mode.

    The rounding mode can be one of the following:
    - rn: round to nearest even
    - rz: round toward zero
    - rm: round toward negative infinity
    - rp: round toward positive infinity

    Examples:
    ```mlir
    // Round to nearest even (default)
    %0 = nvvm.add .packed.f32x2.bf16x2.f32x2 %srcA, %srcB : vector<2xbf16>, vector<2xf32> -> vector<2xf32>

    // Round toward negative infinity
    %1 = nvvm.add .packed.f32x2.bf16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>

    // Round toward zero with flush-to-zero
    %2 = nvvm.add .packed.f32x2.bf16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues:$res
  );
}
def NVVM_AddPacked_F32x2_F16x2_F32x2Op : NVVM_Op<"add.packed.f32x2.f16x2.f32x2"> {
  let description = [{
    This Op performs the add operation on a pair of floating-point values.
    The inputs are of the form v2f16 and v2f32. The first input `srcA` is
    converted to a scalar of packed i32 where the 0th element of the vector forms
    the low-16 bits and the 1st element occupies the high-16 bits. Similarly,
    the second input `srcB` is converted to a scalar of packed i64 where the 0th
    element of the vector forms the low-32 bits and the 1st element occupies the
    high-32 bits. The `rnd` attribute specifies the rounding mode.

    The rounding mode can be one of the following:
    - rn: round to nearest even
    - rz: round toward zero
    - rm: round toward negative infinity
    - rp: round toward positive infinity

    Examples:
    ```mlir
    // Round to nearest even (default)
    %0 = nvvm.add .packed.f32x2.f16x2.f32x2 %srcA, %srcB : vector<2xf16>, vector<2xf32> -> vector<2xf32>

    // Round toward negative infinity
    %1 = nvvm.add .packed.f32x2.f16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>

    // Round toward zero with flush-to-zero
    %2 = nvvm.add .packed.f32x2.f16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues:$res
  );
}
def NVVM_AddFOp : NVVM_InferredOp<"addf", [Pure]> {
  let description = [{
    The `nvvm.addf` operation performs floating point addition of two floating
    point operands of the same type.

    The rounding mode is specified by the `rnd` attribute, saturation mode by
    the `sat` attribute, and flush-to-zero by the `ftz` attribute.

    For more information, see PTX ISA:
    - [floating point addition](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add)
    - [half-precision floating point addition](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add)
  }];
  let arguments = (ins
    NVVM_T16BitFloatOrBfloat16Type:$lhs,
    NVVM_T16BitFloatOrBfloat16Type:$rhs,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T16BitFloatOrBfloat16Type:$res
  );
}
def NVVM_ApplyPriorityAsyncBulkOp : NVVM_Op<"applypriority.async.bulk"> {
  let description = [{
    Sets the eviction priority to evict normal for cache lines within the specified
    region of L2 cache that are previously present and marked as evict last.
    Other L2 cache lines in the specified region are not affected.

    Arguments:
    - `ptr`: Global memory pointer.
    - `size`: Address range of length.
    - `evict`: Eviction priority.

    Example:
    ```mlir
    nvvm.applypriority.async.bulk %ptr, %size evict_priority = evict_normal
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace1:$ptr,
    NVVM_T32BitSignlessInteger:$size,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evict
  );
  let results = (outs);
}
def NVVM_ApplyPriorityAsyncBulkTensorOp : NVVM_Op<"applypriority.async.bulk.tensor", [AttrSizedOperandSegments]> {
  let description = [{
    Sets the eviction priority to evict normal for cache lines within the
    specified region of L2 cache that are previously present and marked as
    evict last. Other L2 cache lines in the specified region are not affected.

    Arguments:
    - `tmaDesc`: The TMA descriptor.
    - `box/coordinates`: The coordinates of the tensor.
    - `im2col`: The im2col arguments.
    - `mode`: Tensor load mode.
    - `evict`: Eviction priority.

    Example:
    ```mlir
    nvvm.applypriority.async.bulk.tensor %tma_desc, box [ %d0] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr
    nvvm.applypriority.async.bulk.tensor %tma_desc, box [ %d0, %d1, %d2] im2col [ %im2col0] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
    nvvm.applypriority.async.bulk.tensor %tma_desc, box [ %d0, %d1, %d2] im2col [ %im2w_halo, %im2w_offset] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<im2col_w>} : !llvm.ptr
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$tmaDesc,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$im2colArgs,
    OptionalAttr<NVVM_TMALoadModeAttr>:$mode,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evict
  );
  let results = (outs);
}
def NVVM_ApplyPriorityAsyncBulkTensorOverrideOp : NVVM_Op<"applypriority.async.bulk.tensor.override", [AttrSizedOperandSegments]> {
  let description = [{
    Sets the eviction priority to evict normal for cache lines within the
    specified region of L2 cache that are previously present and marked as
    evict last. Other L2 cache lines in the specified region are not affected.
    It supports a way to override the global address of the tensor.

    Arguments:
    - `tmaDesc`: The TMA descriptor.
    - `overrideAdrr`: The global address to override.
    - `box/coordinates`: The coordinates of the tensor.
    - `tensorSize`: The size of the tensor.
    - `lowerStride`: The lower stride of the tensor.
    - `upperStride`: The upper stride of the tensor.
    - `im2col`: The im2col arguments.
    - `mode`: Tensor load mode.
    - `evict`: Eviction priority.

    Example:
    ```mlir
    nvvm.applypriority.async.bulk.tensor.override %tma_desc, %gbl_addr, box [ %d0] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr
    nvvm.applypriority.async.bulk.tensor.override %tma_desc, %gbl_addr, box [ %d0, %d1, %d2] im2col [ %im2col0] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<im2col>} : !llvm.ptr
    nvvm.applypriority.async.bulk.tensor.override %tma_desc, %gbl_addr, box [ %d0] tensor_size [ %ts0] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr
    nvvm.applypriority.async.bulk.tensor.override %tma_desc, %gbl_addr, box [ %d0, %d1] tensor_size [ %ts0, %ts1] lower_stride [ %lstrd0] upper_stride [ %ustrd] evict_priority = evict_normal {mode = #nvvm.tma_load_mode<tile>} : !llvm.ptr
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$tmaDesc,
    NVVM_LLVMPointerInAddressSpace1:$overrideAdrr,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$tensorSize,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$lowerStride,
    Optional<NVVM_T16BitSignlessInteger>:$upperStride,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$im2colArgs,
    OptionalAttr<NVVM_TMALoadModeAttr>:$mode,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evict
  );
  let results = (outs);
}
def NVVM_AtomicRMWOp : NVVM_InferredOp<"atomicrmw"> {
  let description = [{
    Atomically loads the original value at location `a` into destination register `res`,
    performs a reduction operation with operand `b` and the value in location `a`, and
    stores the result of the specified operation at location `a`, overwriting the
    original value. Operand `a` specifies a location in the specified state space.
    If no state space is given, perform the memory accesses using `Generic Addressing`.
    For more information, refer [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace12:$ptr,
    NVVM_T16BitSignlessIntegerOr32:$a,
    Optional<NVVM_T16BitSignlessIntegerOr32>:$b,
    NVVM_AtomicOpKindAttr:$op,
    OptionalAttr<NVVM_MemOrderKindAttr>:$memOrder,
    OptionalAttr<NVVM_MemScopeKindAttr>:$syncscope
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOr32:$res
  );
}
def NVVM_SyncWarpOp : NVVM_Op<"bar.warp.sync"> {
  let description = [{
    The `nvvm.bar.warp.sync` operation performs barrier synchronization for threads
    within a warp.

    This operation causes the executing thread to wait until all threads corresponding
    to the `mask` operand have executed a `bar.warp.sync` with the same mask value
    before resuming execution.

    The `mask` operand specifies the threads participating in the barrier, where each
    bit position corresponds to the thread's lane ID within the warp. Only threads with
    their corresponding bit set in the mask participate in the barrier synchronization.

    **Important constraints**:
    - The behavior is undefined if the executing thread is not included in the mask
      (i.e., the bit corresponding to the thread's lane ID is not set)
    - For compute capability sm_6x or below, all threads in the mask must execute
      the same `bar.warp.sync` instruction in convergence

    This operation also guarantees memory ordering among participating threads.
    Threads within the warp that wish to communicate via memory can store to memory,
    execute `bar.warp.sync`, and then safely read values stored by other threads
    in the warp.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$mask
  );
  let results = (outs);
}
def NVVM_BarrierOp : NVVM_InferredOp<"barrier", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.barrier` operation performs barrier synchronization and communication
    within a CTA (Cooperative Thread Array). It causes executing threads to wait for
    all non-exited threads participating in the barrier to arrive.

    The operation takes two optional operands:

    - `barrierId`: Specifies a logical barrier resource with value 0 through 15.
      Each CTA instance has sixteen barriers numbered 0..15. Defaults to 0 if not specified.
    - `numberOfThreads`: Specifies the number of threads participating in the barrier.
      When specified, the value must be a multiple of the warp size. If not specified,
      all threads in the CTA participate in the barrier.
    - `reductionOp`: specifies the reduction operation (`popc`, `and`, `or`).
    - `reductionPredicate`: specifies the predicate to be used with the
      `reductionOp`.

    The barrier operation guarantees that when the barrier completes, prior memory
    accesses requested by participating threads are performed relative to all threads
    participating in the barrier. It also ensures that no new memory access is
    requested by participating threads before the barrier completes.

    When a barrier completes, the waiting threads are restarted without delay, and
    the barrier is reinitialized so that it can be immediately reused.

    This operation generates an aligned barrier, indicating that all threads in the CTA
    will execute the same barrier instruction. Behavior is undefined if all threads in the
    CTA do not reach this instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
  }];
  let arguments = (ins
    Optional<NVVM_T32BitSignlessInteger>:$barrierId,
    Optional<NVVM_T32BitSignlessInteger>:$numberOfThreads,
    Optional<NVVM_T32BitSignlessInteger>:$reductionPredicate,
    OptionalAttr<NVVM_BarrierReductionAttr>:$reductionOp
  );
  let results = (outs
    Optional<NVVM_T32BitSignlessInteger>:$res
  );
}
def NVVM_BarrierArriveOp : NVVM_Op<"barrier.arrive"> {
  let description = [{
    Thread that executes this op announces their arrival at the barrier with
    given id and continue their execution.

    The default barrier id is 0 that is similar to `nvvm.barrier` Op. When
    `barrierId` is not present, the default barrier id is used.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
  }];
  let arguments = (ins
    Optional<NVVM_T32BitSignlessInteger>:$barrierId,
    NVVM_T32BitSignlessInteger:$numberOfThreads
  );
  let results = (outs);
}
def NVVM_BarrierCTAArriveOp : NVVM_Op<"barrier.cta.arrive"> {
  let description = [{
    The `nvvm.barrier.cta.arrive` is used by the threads within the CTA for synchronization and communication.
    The `barrierId` specifies a logical barrier resource and the `threadCount` specifies the number of threads
    participating in the barrier. The `barrier.cta.arrive` does not cause the executing thread to wait for
    threads of other participating warps. The `aligned` attribute, when set to 1, generates the .aligned
    version of the PTX instruction.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$barrierId,
    NVVM_T32BitSignlessInteger:$threadCount,
    OptionalAttr<I1Attr>:$aligned
  );
  let results = (outs);
}
def NVVM_BarrierCTAReduxOp : NVVM_Op<"barrier.cta.red"> {
  let description = [{
    The `nvvm.barrier.cta.red` is used to perform a reduction operation across threads in the CTA.
    The `pred` from all the participating threads are combined using the specified reduction operator.
    AND, OR, POPC (population-count) are available as reduction operators. The `barrierId` specifies
    a logical barrier resource and the `threadCount` specifies the number of threads participating
    in the barrier. When the `threadCount` is not specified, all threads in the CTA participate in
    the reduction operation. The `aligned` attribute, when set to 1, generates the .aligned version
    of the PTX instruction.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)
  }];
  let arguments = (ins
    NVVM_T1BitSignlessInteger:$pred,
    NVVM_T32BitSignlessInteger:$barrierId,
    Optional<NVVM_T32BitSignlessInteger>:$threadCount,
    NVVM_BarrierReduxKindAttr:$kind,
    OptionalAttr<I1Attr>:$aligned
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_BarrierCTASyncOp : NVVM_Op<"barrier.cta.sync"> {
  let description = [{
    The `nvvm.barrier.cta.sync` is used by the threads within the CTA for synchronization and communication.
    The `barrierId` specifies a logical barrier resource and the `threadCount` specifies the number of
    threads participating in the barrier. When the `threadCount` is not specified, all threads in the CTA
    participate in the barrier. The `aligned` attribute, when set to 1, generates the .aligned version of
    the PTX instruction.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$barrierId,
    Optional<NVVM_T32BitSignlessInteger>:$threadCount,
    OptionalAttr<I1Attr>:$aligned
  );
  let results = (outs);
}
def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  let description = [{
    The `nvvm.barrier0` operation is a convenience operation that performs barrier
    synchronization and communication within a CTA (Cooperative Thread Array) using
    barrier ID 0. It is functionally equivalent to `nvvm.barrier` or `nvvm.barrier id=0`.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_Breakpoint : NVVM_Op<"breakpoint"> {
  let description = [{
    Breakpoint suspends execution of the program for debugging.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_ClmadOp : NVVM_Op<"clmad", [Pure]> {
  let description = [{
    Performs a carryless multiplication of `a` and `b`, followed by a carryless
    addition of `c`.
    When `hi` is true, the result is the upper 64 bits of the product, with the
    addition of `c`. Otherwise, the result is the lower 64 bits of the product,
    with the addition of `c`.
  }];
  let arguments = (ins
    NVVM_T64BitSignlessInteger:$a,
    NVVM_T64BitSignlessInteger:$b,
    NVVM_T64BitSignlessInteger:$c,
    OptionalAttr<BoolAttr>:$high
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_ClusterArriveOp : NVVM_Op<"cluster.arrive"> {
  let description = [{
    The `cluster.arrive` can be used by the threads within the cluster for synchronization and
    communication. The `cluster.arrive` instruction marks the warps' arrival at the barrier
    without causing the executing thread to wait for other participating threads.

    The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
  }];
  let arguments = (ins
    UnitAttr:$aligned
  );
  let results = (outs);
}
def NVVM_ClusterArriveRelaxedOp : NVVM_Op<"cluster.arrive.relaxed"> {
  let description = [{
    The `cluster.arrive` can be used by the threads within the cluster for synchronization and
    communication. The `cluster.arrive` instruction marks the warps' arrival at the barrier
    without causing the executing thread to wait for other participating threads.

    The `aligned` attribute, when provided, generates the .aligned version of the PTX instruction.
    The .relaxed qualifier on `cluster.arrive` specifies that there are no memory
    ordering and visibility guarantees provided for the memory accesses performed prior to
    `cluster.arrive`.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
  }];
  let arguments = (ins
    UnitAttr:$aligned
  );
  let results = (outs);
}
def NVVM_ClusterWaitOp : NVVM_Op<"cluster.wait"> {
  let description = [{
    The `cluster.wait` causes the executing thread to wait for all non-exited threads
    of the cluster to perform `cluster.arrive`. The `aligned` attribute, when provided,
    generates the .aligned version of the PTX instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)
  }];
  let arguments = (ins
    UnitAttr:$aligned
  );
  let results = (outs);
}
def NVVM_ClusterLaunchControlQueryCancelOp : NVVM_InferredOp<"clusterlaunchcontrol.query.cancel"> {
  let description = [{
    `clusterlaunchcontrol.query.cancel` queries the response of a
    `clusterlaunchcontrol.try.cancel` operation specified by operand
    `try_cancel_response`.

    Operand `query_type` specifies the type of query to perform and can be one
    of the following:
    - `is_canceled` : Returns true if the try cancel request succeeded,
    and false otherwise.
    - `get_first_cta_id_{x/y/z}` : Returns the x, y, or z coordinate of the
    first CTA in the canceled cluster. Behaviour is defined only if the try
    cancel request succeeded.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel)
  }];
  let arguments = (ins
    NVVM_T128BitSignlessInteger:$try_cancel_response,
    NVVM_ClusterLaunchControlQueryTypeAttr:$query_type
  );
  let results = (outs
    NVVM_T1BitSignlessIntegerOr32:$res
  );
}
def NVVM_ClusterLaunchControlTryCancelOp : NVVM_Op<"clusterlaunchcontrol.try.cancel"> {
  let description = [{
    `clusterlaunchcontrol.try.cancel` requests atomically canceling the launch
    of a cluster that has not started running yet. It asynchronously writes an
    opaque response to shared memory indicating whether the operation succeeded
    or failed.

    Operand `smemAddress` specifies the naturally aligned address of the
    16-byte wide shared memory location where the request's response is written.

    Operand `mbarrier` specifies the mbarrier object used to track the
    completion of the asynchronous operation.

    If `multicast` is specified, the response is asynchronously written to the
    corresponding local shared memory location (specifed by `addr`) of each CTA
    in the requesting cluster.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace3:$smemAddress,
    NVVM_LLVMPointerInAddressSpace3:$mbarrier,
    UnitAttr:$multicast
  );
  let results = (outs);
}
def NVVM_CompareAndSetOp : NVVM_InferredOp<"compare.and.set"> {
  let description = [{
    Compare two values and return boolean mask of 1's or 0's. Return value
    will be of the same type as the operands (vector<4xi8> or vector<2xi16>).

    Arguments:
    - `a`: The first value.
    - `b`: The second value.
    - `op`: The comparison operator. (eq, ne, ugt, uge, ult, ule, sgt, sge, slt, sle)

    Example:
    ```mlir
    %res = nvvm.compare.and.set <ugt> %a, %b : vector<4xi8>
    %res = nvvm.compare.and.set <slt> %a, %b : vector<2xi16>
    ```
  }];
  let arguments = (ins
    NVVM_FixedLengthVectorOf8Bit:$a,
    NVVM_FixedLengthVectorOf8Bit:$b,
    NVVM_CompareOpKindAttr:$op
  );
  let results = (outs
    NVVM_FixedLengthVectorOf8Bit:$result
  );
}
def NVVM_ConvertPackIntegerOp : NVVM_Op<"convert.and.pack.integer"> {
  let description = [{
    This Op converts the given integer inputs to the specified integer type
    with saturation and packs the result into the lower bits of the result.

    The result `dst` is represented as an i32 type.

    The optional operand `srcC` must be specified when the converted types add
    up to less than 32 bits. The remaining bits of the result are filled with
    the lower bits of `srcC`.

    The `convert_type` attribute specifies the integer type to convert to
    before packing. The `signed` attribute can be specified to indicate that
    the `convert_type` is signed.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt-pack)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$srcA,
    NVVM_T32BitSignlessInteger:$srcB,
    Optional<NVVM_T32BitSignlessInteger>:$srcC,
    UnitAttr:$is_signed,
    TypeAttr:$convert_type
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$dst
  );
}
def NVVM_ConvertBF16x2ToF4x2Op : NVVM_Op<"convert.bf16x2.to.f4x2"> {
  let description = [{
    This Op converts each of the given BF16 inputs in an bf16x2 vector to the specified fp4 type.
    The result `dst` is returned as an i8 type where the converted values are
    packed such that the value converted from the first element of `a` is
    stored in the lower 4 bits of `dst` and the value converted from the second
    element of `a` is stored in the upper 4 bits of `dst`.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.
    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<BoolAttr>:$relu,
    NVVM_ConvertBF16x2ToF4x2DstTyAttr:$dstTy
  );
  let results = (outs
    NVVM_T8BitSignlessInteger:$dst
  );
}
def NVVM_ConvertBF16x2ToF6x2Op : NVVM_Op<"convert.bf16x2.to.f6x2"> {
  let description = [{
    This Op converts each of the given BF16 inputs in an bf16x2 vector to the specified fp6 type. The result `dst` is
    represented either as an i16 type or as a vector of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from the first element of `a` is stored in the
    lower 8 bits of `dst` with 2 MSB bits padded with zeros and the value
    converted from the second element of `a` is stored in the upper 8 bits of
    `dst` with 2 MSB bits padded with zeros.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector with 2 MSB bits padded with zeros.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction. The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<BoolAttr>:$relu,
    NVVM_ConvertBF16x2ToF6x2DstTyAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertBF16x2ToF8x2Op : NVVM_Op<"convert.bf16x2.to.f8x2"> {
  let description = [{
    This Op converts the given bf16 inputs in a bf16x2 vector to the specified
    f8 type. The result `dst` is represented either as a packed i16 type or as
    a vector of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from the first element of `a` is stored in the
    lower 8 bits of `dst` and the value converted from the second element of
    `a` is stored in the upper 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector.
    The `rnd` and `sat` attributes specify the rounding and saturation modes
    respectively.

    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.


    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu,
    NVVM_ConvertBF16x2ToF8x2DstTyAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertBF16x2ToS2F6x2Op : NVVM_Op<"convert.bf16x2.to.s2f6x2"> {
  let description = [{
    This Op converts each of the given BF16 inputs in a bf16x2 vector to the
    S2F6x2 type. The result `dst` can be either a packed i16 type or a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from the first element of `a` is stored in the
    lower 8 bits of `dst` and the value converted from the second element of
    `a` is stored in the upper 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector.
    The `relu` attribute, when set, lowers to the '.relu' variant
    of the cvt instruction.
    The optional scaling-factors for each of the inputs are provided through
    the operand `scaleFactor` as a packed i16 type. Only `ue8m0` is supported
    as the type of the scale-factor currently.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scaleFactor,
    OptionalAttr<BoolAttr>:$relu
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertF16x2ToF4x2Op : NVVM_Op<"convert.f16x2.to.f4x2"> {
  let description = [{
    This Op converts each of the given F16 inputs in an f16x2 vector to the specified fp4 type.
    The result `dst` is returned as an i8 type where the converted values are
    packed such that the value converted from the first element of `a` is
    stored in the lower 4 bits of `dst` and the value converted from the second
    element of `a` is stored in the upper 4 bits of `dst`.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.
    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<BoolAttr>:$relu,
    NVVM_ConvertF16x2ToF4x2DstTyAttr:$dstTy
  );
  let results = (outs
    NVVM_T8BitSignlessInteger:$dst
  );
}
def NVVM_ConvertF16x2ToF6x2Op : NVVM_Op<"convert.f16x2.to.f6x2"> {
  let description = [{
    This Op converts each of the given F16 inputs in an f16x2 vector to the specified fp6 type. The result `dst` is
    represented either as an i16 type or as a vector of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from the first element of `a` is stored in the
    lower 8 bits of `dst` with 2 MSB bits padded with zeros and the value
    converted from the second element of `a` is stored in the upper 8 bits of
    `dst` with 2 MSB bits padded with zeros.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector with 2 MSB bits padded with zeros.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction. The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<BoolAttr>:$relu,
    NVVM_ConvertF16x2ToF6x2DstTyAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertF16x2ToF8x2Op : NVVM_Op<"convert.f16x2.to.f8x2"> {
  let description = [{
    This Op converts the given f16 inputs in an f16x2 vector to the specified
    f8 type.
    The result `dst` is represented as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values from `a`
    are packed such that the value converted from the first element of `a`
    is stored in the upper 8 bits of `dst` and the value converted from the
    second element of `a` is stored in the lower 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.
    The `rnd` and `sat` attributes specify the rounding and saturation modes
    respectively.
    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$a,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertF32x2ToBF16x2Op : NVVM_Op<"convert.f32x2.to.bf16x2"> {
  let description = [{
    Converts two F32 values to packed bf16x2 format with
    the specified rounding mode. The `src_hi` and `src_lo` parameters
    correspond to operands `a` and `b` in the PTX ISA, respectively.

    The `random_bits` parameter is required for stochastic rounding and
    provides the [random bits](https://docs.nvidia.com/cuda/parallel-thread-execution/#cvt-rs-rbits-layout-bf16) to be used for the conversion.

    The `relu` attribute clamps negative results to 0.

    The `sat` attribute determines saturation behavior.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src_hi,
    NVVM_T32BitFloat:$src_lo,
    Optional<NVVM_T32BitSignlessInteger>:$random_bits,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$dst
  );
}
def NVVM_ConvertF32x2ToF16x2Op : NVVM_Op<"convert.f32x2.to.f16x2"> {
  let description = [{
    Converts two F32 values to packed f16x2 format with
    the specified rounding mode. The `src_hi` and `src_lo` parameters
    correspond to operands `a` and `b` in the PTX ISA, respectively.

    The `random_bits` parameter is required for stochastic rounding and
    provides the [random bits](https://docs.nvidia.com/cuda/parallel-thread-execution/#cvt-rs-rbits-layout-f16) to be used for the conversion.

    The `relu` attribute clamps negative results to 0.

    The `sat` attribute determines saturation behavior.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src_hi,
    NVVM_T32BitFloat:$src_lo,
    Optional<NVVM_T32BitSignlessInteger>:$random_bits,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$dst
  );
}
def NVVM_ConvertF32x2ToF4x2Op : NVVM_Op<"convert.f32x2.to.f4x2"> {
  let description = [{
    This Op converts each of the given float inputs to the specified fp4 type.

    The result `dst` is returned as an i8 type where the converted values are
    packed such that the value converted from `a` is stored in the upper 4 bits
    of `dst` and the value converted from `b` is stored in the lower 4 bits of
    `dst`.

    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.

    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.


    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$a,
    NVVM_T32BitFloat:$b,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstType
  );
  let results = (outs
    NVVM_T8BitSignlessInteger:$dst
  );
}
def NVVM_ConvertF32x2ToF6x2Op : NVVM_Op<"convert.f32x2.to.f6x2"> {
  let description = [{
    This Op converts each of the given float inputs to the specified fp6 type.
    The result `dst` is represented either as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from `a` is stored in the upper 8 bits of `dst`
    with 2 MSB bits padded with zeros and the value converted from `b` is
    stored in the lower 8 bits of `dst` with 2 MSB bits padded with zeros.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.

    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.


    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$a,
    NVVM_T32BitFloat:$b,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertF32x2ToF8x2Op : NVVM_Op<"convert.f32x2.to.f8x2"> {
  let description = [{
    This Op converts each of the given float inputs to the specified fp8 type.
    The result `dst` is represented as an i16 type or as a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from `a` is stored in the upper 8 bits of `dst`
    and the value converted from `b` is stored in the lower 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector.
    The `rnd` and `sat` attributes specify the rounding and saturation modes respectively.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction.

    The `pzo` (positive zero only), when provided, forces negative zero to be positive zero.


    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$a,
    NVVM_T32BitFloat:$b,
    Optional<NVVM_T16BitSignlessInteger>:$scale_factor,
    UnitAttr:$isPZO,
    OptionalAttr<NVVM_ConvertScaleKindAttr>:$scale_factor_kind,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertF32x2ToS2F6x2Op : NVVM_Op<"convert.f32x2.to.s2f6x2"> {
  let description = [{
    This Op converts each of the given f32 inputs to the
    S2F6x2 type. The result `dst` can be either a packed i16 type or a vector
    of two i8 types.
    If `dst` is returned as an i16 type, the converted values are packed such
    that the value converted from `a` is stored in the upper 8 bits of `dst`
    and the value converted from `b` is stored in the lower 8 bits of `dst`.
    If `dst` is returned as a vector type, each converted value is stored as an
    i8 element in the vector.
    The `relu` attribute, when set, lowers to the '.relu' variant
    of the cvt instruction.
    The optional scaling-factors for each of the inputs are provided through
    the operand `scaleFactor` as a packed i16 type. Only `ue8m0` is supported
    as the type of the scale-factor currently.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$a,
    NVVM_T32BitFloat:$b,
    Optional<NVVM_T16BitSignlessInteger>:$scaleFactor,
    OptionalAttr<BoolAttr>:$relu
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOrVector:$dst
  );
}
def NVVM_ConvertF32x4ToF4x4Op : NVVM_Op<"convert.f32x4.to.f4x4"> {
  let description = [{
    Converts a vector<4xf32> to packed f4x4 format using
    stochastic rounding (.rs) mode with SATFINITE saturation. Randomness is
    provided by the `rbits` parameter. The `dstTy` attribute specifies the
    target floating-point format. The `relu` attribute clamps negative results to 0.

    Note: These operations always use RS rounding mode and SATFINITE saturation mode.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues3:$src,
    NVVM_T32BitSignlessInteger:$rbits,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstTy
  );
  let results = (outs
    NVVM_T16BitSignlessInteger2:$dst
  );
}
def NVVM_ConvertF32x4ToF6x4Op : NVVM_Op<"convert.f32x4.to.f6x4"> {
  let description = [{
    Converts a vector<4xf32> to packed f6x4 format using
    stochastic rounding (.rs) mode with SATFINITE saturation. Randomness is
    provided by the `rbits` parameter. The `dstTy` attribute specifies the
    target floating-point format. The `relu` attribute clamps negative results to 0.

    Note: These operations always use RS rounding mode and SATFINITE saturation mode.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues3:$src,
    NVVM_T32BitSignlessInteger:$rbits,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstTy
  );
  let results = (outs
    NVVM_VectorOf8BitSignlessInteger:$dst
  );
}
def NVVM_ConvertF32x4ToF8x4Op : NVVM_Op<"convert.f32x4.to.f8x4"> {
  let description = [{
    Converts a vector<4xf32> to packed f8x4 format using
    stochastic rounding (.rs) mode with SATFINITE saturation. Randomness is
    provided by the `rbits` parameter. The `dstTy` attribute specifies the
    target floating-point format. The `relu` attribute clamps negative results to 0.

    Note: These operations always use RS rounding mode and SATFINITE saturation mode.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues3:$src,
    NVVM_T32BitSignlessInteger:$rbits,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$dstTy
  );
  let results = (outs
    NVVM_VectorOf8BitSignlessInteger:$dst
  );
}
def NVVM_ConvertF4x2ToF16x2Op : NVVM_Op<"convert.f4x2.to.f16x2"> {
  let description = [{
    This Op converts the given f4 inputs in a packed i8 to f16.

    The result `dst` is represented as a vector of f16 elements.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction."

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T8BitSignlessInteger2:$src,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$srcType
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$dst
  );
}
def NVVM_ConvertF6x2ToF16x2Op : NVVM_Op<"convert.f6x2.to.f16x2"> {
  let description = [{
    This Op converts the given f6 inputs in a i8x2 vector to f16.

    The result `dst` is represented as a vector of f16 elements.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction."

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf8BitSignlessInteger2:$src,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$srcType
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$dst
  );
}
def NVVM_ConvertF8x2ToBF16x2Op : NVVM_Op<"convert.f8x2.to.bf16x2"> {
  let description = [{
    This Op converts the given f8 inputs in a i8x2 vector to bf16.

    The result `dst` is represented as a vector of bf16 elements.
    The `sat` attribute specifies the saturation mode.


    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf8BitSignlessInteger2:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scaleFactor,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    TypeAttr:$srcType
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$dst
  );
}
def NVVM_ConvertF8x2ToF16x2Op : NVVM_Op<"convert.f8x2.to.f16x2"> {
  let description = [{
    This Op converts the given f8 inputs in a i8x2 vector to f16.

    The result `dst` is represented as a vector of f16 elements.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction."

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf8BitSignlessInteger2:$src,
    OptionalAttr<BoolAttr>:$relu,
    TypeAttr:$srcType
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$dst
  );
}
def NVVM_ConvertFloatToIntegerOp : NVVM_Op<"convert.float.to.integer"> {
  let description = [{
    This Op converts the given float input to the specified integer type.

    The `rnd` attribute specifies the rounding mode.
    The `sat` atttribute, when set, limits the result to the range of the
    integer type.
    The `ftz` attribute, when set, converts denormal inputs to zero.
    The `signed` attribute, when set, converts the input to a signed integer.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src,
    OptionalAttr<NVVM_IntegerRoundingModeAttr>:$rnd,
    UnitAttr:$sat,
    UnitAttr:$ftz,
    UnitAttr:$is_signed
  );
  let results = (outs
    NVVM_T8BitSignlessIntegerOr32:$dst
  );
}
def NVVM_ConvertFloatToTF32Op : NVVM_Op<"convert.float.to.tf32"> {
  let description = [{
    This Op converts the given f32 input to tf32.
    The result `res` is represented as an i32 type.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction. The `rnd` and `sat` attributes specify the
    the rounding and saturation modes respectively.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_ConvertS2F6x2ToBF16x2Op : NVVM_Op<"convert.s2f6x2.to.bf16x2"> {
  let description = [{
    This Op converts a pair of s2f6x2 inputs to bf16x2 type. The result `dst`
    is represented as a vector of two bf16 elements.

    The `relu` attribute, when set, lowers to the '.relu' variant
    of the cvt instruction.

    The optional scaling-factors for each of the inputs are provided through
    the operand `scaleFactor` as a packed i16 type. Only `ue8m0` is supported
    as the type of the scale-factor currently.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_VectorOf8BitSignlessInteger2:$src,
    Optional<NVVM_T16BitSignlessInteger>:$scaleFactor,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$relu
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$dst
  );
}
def NVVM_CosOp : NVVM_Op<"cos", [Pure]> {
  let description = [{
    Computes a fast approximation of the cosine of the input value (in
    radians). The `ftz` attribute, when set, flushes subnormal inputs
    and results to sign-preserving zero.
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_CpAsyncBulkCommitGroupOp : NVVM_Op<"cp.async.bulk.commit.group"> {
  let description = [{
    This Op commits all prior initiated but uncommitted cp.async.bulk
    instructions into a cp.async.bulk-group.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_CpAsyncBulkSharedCTAToGlobalOp : NVVM_Op<"cp.async.bulk.global.shared.cta", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous copy operation from Shared CTA memory to
    global memory. The 32-bit operand `size` specifies the amount of
    memory to be copied, in terms of number of bytes. `size` must be a
    multiple of 16. The `l2CacheHint` operand is optional, and it is used
    to specify cache eviction policy that may be used during the memory
    access. The `byteMask` operand is optional. The i-th bit in the 16-bit
    wide `byteMask` specifies whether the i-th byte of each 16-byte wide
    chunk of source data is copied to the destination. If the bit is set,
    the byte is copied.

    Example:
    ```mlir
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size
          : !llvm.ptr<1>, !llvm.ptr<3>

      // with l2_cache_hint
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch
          : !llvm.ptr<1>, !llvm.ptr<3>

      // with byte_mask
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size byte_mask = %mask
          : !llvm.ptr<1>, !llvm.ptr<3>

      // with both l2_cache_hint and byte_mask
      nvvm.cp.async.bulk.global.shared.cta %dst, %src, %size l2_cache_hint = %ch byte_mask = %mask
          : !llvm.ptr<1>, !llvm.ptr<3>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace1:$dstMem,
    NVVM_LLVMPointerInAddressSpace3:$srcMem,
    NVVM_T32BitSignlessInteger:$size,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    Optional<NVVM_T16BitSignlessInteger>:$byteMask
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkPrefetchOp : NVVM_Op<"cp.async.bulk.prefetch"> {
  let description = [{
    Initiates an asynchronous prefetch of data from the location
    specified by `srcMem` to the L2 cache.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.
    The `evict` operand is optional, and it is used to specify cache eviction
    priority. The `evict` operand is mutually exclusive with the `l2CacheHint`
    operand.

    Example:
    ```mlir
      nvvm.cp.async.bulk.prefetch %src, %size : !llvm.ptr<1>

      // with l2_cache_hint
      nvvm.cp.async.bulk.prefetch %src, %size l2_cache_hint = %ch : !llvm.ptr<1>
    ```
    Example:
    ```mlir
      // with evict_priority
      nvvm.cp.async.bulk.prefetch %src, %size evict_priority = evict_last : !llvm.ptr<1>
    ```
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace1:$srcMem,
    NVVM_T32BitSignlessInteger:$size,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evict
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkGlobalToSharedClusterOp : NVVM_Op<"cp.async.bulk.shared.cluster.global", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous copy operation from global memory to shared
    memory or shared_cluster memory.

    The `multicastMask` operand is optional and can be used only when the
    destination is shared::cluster memory. When it is present, this Op copies
    data from global memory to shared memory of multiple CTAs in the cluster.
    Operand `multicastMask` specifies the destination CTAs in the cluster such
    that each bit position in the 16-bit `multicastMask` operand corresponds to
    the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace32:$dstMem,
    NVVM_LLVMPointerInAddressSpace1:$srcMem,
    NVVM_LLVMPointerInAddressSpace3:$mbar,
    NVVM_T32BitSignlessInteger:$size,
    Optional<NVVM_T16BitSignlessIntegerOr322>:$multicastMask,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_ValidatePatternAttr>:$validatePattern
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkSharedCTAToSharedClusterOp : NVVM_Op<"cp.async.bulk.shared.cluster.shared.cta"> {
  let description = [{
    Initiates an asynchronous copy operation from Shared CTA memory to Shared
    cluster memory.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace7:$dstMem,
    NVVM_LLVMPointerInAddressSpace3:$srcMem,
    NVVM_LLVMPointerInAddressSpace3:$mbar,
    NVVM_T32BitSignlessInteger:$size
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorSharedCTAToGlobalOp : NVVM_Op<"cp.async.bulk.tensor.global.shared.cta", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous copy of the tensor data from shared::cta
    memory to global memory. This Op supports all the store modes specified in
    `TMAStoreMode`.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace0:$tmaDescriptor,
    NVVM_LLVMPointerInAddressSpace3:$srcMem,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    OptionalAttr<NVVM_TMAStoreModeAttr>:$mode
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorSharedCTAToGlobalOverrideAddrOp : NVVM_Op<"cp.async.bulk.tensor.global.shared.cta.override", [AttrSizedOperandSegments]> {
  let description = [{
      Initiates an asynchronous bulk tensor operation that transfers data from shared
      memory to global memory overriding one or many of the fields: addr, size, and stride.

      The operation:
      - Reads data from shared memory location specified by `srcMem`
      - Uses a TMA descriptor `tmaDesc` to control the data access pattern
      - Overrides the default destination address with `overrideAdrr`
      - Specifies box coordinates via `coordinates` array
      - Can specify tensor size via `tensorSize` (optional)
      - Controls memory access patterns with `lowerStride` and `upperStride` (optional)
      - Supports an optional L2 cache hint with `l2CacheHint` to support cache eviction policy (optional)
      - Supports different store modes (`TILE`, `IM2COL`, `IM2COL_W`, and `TILE_SCATTER4`) via the `mode` attribute

    Examples:

    // Basic usage with 1D box coordinates (TILE mode)
    ```mlir
    nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0] {
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // With L2 cache hint
    ```mlir
    nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2]
      l2_cache_hint = %ch {
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // IM2COL mode
    ```mlir
    nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] {
      mode = #nvvm.tma_store_mode<im2col>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // With tensor size
    ```mlir
    nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr,
      box[%d0] tensor_size[%ts0] {
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // With stride
    ```mlir
    nvvm.cp.async.bulk.tensor.global.shared.cta.override %tma_desc, %src, %override_addr,
      box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] {
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$tmaDesc,
    NVVM_LLVMPointerInAddressSpace3:$srcMem,
    NVVM_LLVMPointerInAddressSpace1:$overrideAdrr,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$tensorSize,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$lowerStride,
    Optional<NVVM_T16BitSignlessInteger>:$upperStride,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_TMAStoreModeAttr>:$mode
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorPrefetchOp : NVVM_Op<"cp.async.bulk.tensor.prefetch", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous prefetch operation on the tensor data from global
    memory to L2 cache. This Op supports all the load modes specified in
    `TMALoadMode`.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace0:$tmaDescriptor,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$im2colOffsets,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_TMALoadModeAttr>:$mode,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evict
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorPrefetchOverrideAddrOp : NVVM_Op<"cp.async.bulk.tensor.prefetch.override", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous bulk tensor operation that prefetches data from
    global memory to L2 cache overriding one or many of the fields: addr, size, and stride.

    The operation:
    - Reads data from global memory location specified by `overrideAdrr`
    - Uses a TMA descriptor `tmaDesc` to control the data access pattern
    - Specifies box coordinates via `coordinates` array
    - Can specify tensor size via `tensorSize` (optional)
    - Controls memory access patterns with `lowerStride` and `upperStride` (optional)
    - Can specify im2col offsets via `im2colOffsets` (optional)
    - Supports an optional L2 cache hint with `l2CacheHint` to support cache eviction policy (optional)
    - Supports an optional cache eviction priority with `evict` (optional)
    - Supports different load modes (`TILE`/`IM2COL`/`IM2COL_W`/`IM2COL_W_128`/`TILE_GATHER4`) via the `mode` attribute (default = `TILE`)

    Examples:

    // Basic prefetch operations
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0] : !llvm.ptr, !llvm.ptr<1>
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1, %d2] : !llvm.ptr, !llvm.ptr<1>
    ```

    // With L2 cache hint
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<1>
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1, %d2] l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<1>
    ```

    // With evict_priority
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0] evict_priority = evict_last : !llvm.ptr, !llvm.ptr<1>
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1, %d2] evict_priority = evict_last : !llvm.ptr, !llvm.ptr<1>
    ```

    // With tensor size specification
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0] tensor_size[%ts0] : !llvm.ptr, !llvm.ptr<1>
    ```

    // With tensor size and stride specifications
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1]
    tensor_size[%ts0, %ts1] lower_stride[%ls0] upper_stride[%us0] : !llvm.ptr, !llvm.ptr<1>
    ```

    // With im2col offsets
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1, %d2]
    im2col[%off0] : !llvm.ptr, !llvm.ptr<1>
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1, %d2, %d3]
    im2col[%off0, %off1] : !llvm.ptr, !llvm.ptr<1>
    ```

    // Dim stride with cache hint
    ```mlir
    nvvm.cp.async.bulk.tensor.prefetch.override %tma_desc, %override_addr, box[%d0, %d1, %d2, %d3, %d4]
    tensor_size[%ts0, %ts1, %ts2, %ts3, %ts4]
    lower_stride[%ls0, %ls1, %ls2, %ls3]
    upper_stride[%us0]
    l2_cache_hint = %ch : !llvm.ptr, !llvm.ptr<1>
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$tmaDesc,
    NVVM_LLVMPointerInAddressSpace1:$overrideAdrr,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$tensorSize,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$lowerStride,
    Optional<NVVM_T16BitSignlessInteger>:$upperStride,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$im2colOffsets,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evict,
    OptionalAttr<NVVM_TMALoadModeAttr>:$mode
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorReduceOp : NVVM_Op<"cp.async.bulk.tensor.reduce", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous reduction operation of tensor data in
    global memory with tensor data in shared memory.

    The `mode` attribute indicates whether the copy mode is tile or im2col.
    The `redOp` attribute specifies the reduction operations applied.
    The supported reduction operations are:
    {add, min, max, inc, dec, and, or, xor}

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor)
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$tmaDescriptor,
    NVVM_LLVMPointerInAddressSpace3:$srcMem,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    NVVM_TMAReduxKindAttr:$redKind,
    OptionalAttr<NVVM_TMAStoreModeAttr>:$mode
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorReduceOverrideAddrOp : NVVM_Op<"cp.async.bulk.tensor.reduce.override", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous bulk tensor operation that performs a reduction operation
    overriding one or many of the fields: addr, size, and stride.

    The operation:
    - Reads data from shared memory location specified by `srcMem`
    - Uses a TMA descriptor `tmaDesc` to control the data access pattern
    - Overrides the default destination address with `overrideAdrr`
    - Specifies box coordinates via `coordinates` array
    - Can specify tensor size via `tensorSize` (optional)
    - Controls memory access patterns with `lowerStride` and `upperStride` (optional)
    - Supports an optional L2 cache hint with `l2CacheHint` to support cache eviction policy (optional)
    - Supports different store modes (`TILE` or `IM2COL`) via the `mode` attribute
    - Performs a reduction operation specified by the `redKind` attribute (supports ADD, MIN, MAX,
      INC, DEC, AND, OR, XOR) that combines the source data stored at the destination location

    Examples:

    // Override address with ADD reduction (TILE mode)
    ```mlir
    nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0] {
      redKind = #nvvm.tma_redux_kind<add>,
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // With L2 cache hint
    ```mlir
    nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1]
      l2_cache_hint = %ch {
      redKind = #nvvm.tma_redux_kind<min>,
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // IM2COL mode with XOR reduction
    ```mlir
    nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr, box[%d0, %d1, %d2] {
      redKind = #nvvm.tma_redux_kind<xor>,
      mode = #nvvm.tma_store_mode<im2col>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // With tensor size (dim variant)
    ```mlir
    nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr,
      box[%d0] tensor_size[%ts0] {
      redKind = #nvvm.tma_redux_kind<add>,
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```

    // With stride information (dim_stride variant)
    ```mlir
    nvvm.cp.async.bulk.tensor.reduce.override %tma_desc, %src, %override_addr,
      box[%d0, %d1] tensor_size[%ts0, %ts1] lower_stride[%lstrd0] upper_stride[%ustrd] {
      redKind = #nvvm.tma_redux_kind<and>,
      mode = #nvvm.tma_store_mode<tile>
    } : !llvm.ptr, !llvm.ptr<3>, !llvm.ptr<1>
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$tmaDesc,
    NVVM_LLVMPointerInAddressSpace3:$srcMem,
    NVVM_LLVMPointerInAddressSpace1:$overrideAdrr,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$tensorSize,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$lowerStride,
    Optional<NVVM_T16BitSignlessInteger>:$upperStride,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    NVVM_TMAReduxKindAttr:$redKind,
    OptionalAttr<NVVM_TMAStoreModeAttr>:$mode
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorGlobalToSharedClusterOp : NVVM_Op<"cp.async.bulk.tensor.shared.cluster.global", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous copy operation on the tensor data from global
    memory to shared::cluster (or) shared::cta memory. This Op supports all
    the load modes specified in `TMALoadMode`.

    The `multicastMask` operand is optional. When it is present, the Op copies
    data from global memory to shared memory of multiple CTAs in the cluster.
    Operand `multicastMask` specifies the destination CTAs in the cluster such
    that each bit position in the 16-bit `multicastMask` operand corresponds to
    the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.

    When the `isCTAOnly` attribute is set to true, the destination is
    shared::cta only. Hence, `multicastMask` and `CTAGroup` are not applicable
    when `isCTAOnly` is true.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace32:$dstMem,
    NVVM_LLVMPointerInAddressSpace0:$tmaDescriptor,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    NVVM_LLVMPointerInAddressSpace3:$mbar,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$im2colOffsets,
    Optional<NVVM_T16BitSignlessIntegerOr322>:$multicastMask,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    OptionalAttr<NVVM_ValidatePatternAttr>:$validatePattern,
    OptionalAttr<NVVM_TMALoadModeAttr>:$mode,
    OptionalAttr<BoolAttr>:$isCTAOnly,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkTensorGlobalToSharedClusterOverrideAddrOp : NVVM_Op<"cp.async.bulk.tensor.shared.cluster.global.override", [AttrSizedOperandSegments]> {
  let description = [{
    Initiates an asynchronous bulk tensor operation that transfers data from global
    memory to shared::cluster or shared::cta memory overriding one or many of the fields: addr, size, and stride.

    The operation:
    - Writes data to shared::cluster or shared::cta memory location specified by `dstMem`
    - Uses a TMA descriptor `tmaDesc` to specify the properties of the tensor copy operation
    - Uses `mbar` to specify the memory barrier for the operation
    - Overrides the default source address with `overrideAddr`
    - Specifies box coordinates via `coordinates` array
    - Specifies im2col arguments via `im2colArgs` (optional)
    - Can specify tensor size via `tensorSize` (optional)
    - Controls memory access patterns with `lowerStride` and `upperStride` (optional)
    - Supports multicast mask via `multicastMask` (optional)
    - Supports an optional L2 cache hint with `l2CacheHint` to support cache eviction policy (optional)
    - Supports different load modes (`TILE`/`IM2COL`/`IM2COL_W`/`IM2COL_W_128`/`TILE_GATHER4`) via the `mode` attribute (default = `TILE`)
    - Supports different group kinds (`CTA_1` or `CTA_2`) via the `group` attribute (optional)
    - Supports different validate load data patterns via the `validatePattern` attribute (optional)

    Examples:
    // Basic usage with box coordinates
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0] {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0, %d1] {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // With L2 cache hint
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0]
      l2_cache_hint = %ch {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // With multicast mask (i16)
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0]
      multicast_mask = %mc {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>, i16
    ```

    // With multicast mask (i32)
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0]
      multicast_mask = %mc32 {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>, i32
    ```

    // With both multicast mask and L2 cache hint
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0]
      multicast_mask = %mc
      l2_cache_hint = %ch {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>, i16
    ```

    // With CTA_2 group
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0]
      l2_cache_hint = %ch {
      group = #nvvm.cta_group<cta_2>,
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // With tensor size and strides
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0]
      tensor_size[%ts0] {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0, %d1]
      tensor_size[%ts0, %ts1]
      lower_stride[%lstrd0]
      upper_stride[%ustrd] {
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // Using IM2COL mode with im2col arguments
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0, %d1, %d2]
      im2col[%im2col0] {
      mode = #nvvm.tma_load_mode<im2col>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // Using IM2COL_W mode
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0, %d1, %d2]
      im2col[%wHalo, %wOffset] {
      mode = #nvvm.tma_load_mode<im2col_w>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // Using IM2COL_W_128 mode
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0, %d1, %d2]
      im2col[%wHalo, %wOffset] {
      mode = #nvvm.tma_load_mode<im2col_w_128>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>
    ```

    // Comprehensive example with all optional parameters
    ```mlir
    nvvm.cp.async.bulk.tensor.shared.cluster.global.override %dst, %tma_desc, %mbar, %override_addr, box[%d0, %d1]
      tensor_size[%ts0, %ts1]
      lower_stride[%lstrd0]
      upper_stride[%ustrd]
      multicast_mask = %mc32
      l2_cache_hint = %ch {
      group = #nvvm.cta_group<cta_2>,
      mode = #nvvm.tma_load_mode<tile>
    } : !llvm.ptr<3>, !llvm.ptr, !llvm.ptr<1>, i32
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace72:$dstMem,
    NVVM_LLVMPointerType:$tmaDesc,
    NVVM_LLVMPointerInAddressSpace3:$mbar,
    NVVM_LLVMPointerInAddressSpace1:$overrideAddr,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$coordinates,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$im2colArgs,
    Variadic<NVVM_VariadicOf16BitSignlessInteger>:$tensorSize,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$lowerStride,
    Optional<NVVM_T16BitSignlessInteger>:$upperStride,
    Optional<NVVM_T16BitSignlessIntegerOr322>:$multicastMask,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group,
    OptionalAttr<NVVM_ValidatePatternAttr>:$validatePattern,
    OptionalAttr<NVVM_TMALoadModeAttr>:$mode
  );
  let results = (outs);
}
def NVVM_CpAsyncBulkWaitGroupOp : NVVM_Op<"cp.async.bulk.wait_group"> {
  let description = [{
    Op waits for completion of the most recent bulk async-groups.

    The `$group` operand tells waiting has to be done until for $group or fewer
    of the most recent bulk async-groups. If `$group` is 0, the op wait until
    all the most recent bulk async-groups have completed.

    The `$read` indicates that the waiting has to be done until all the bulk
    async operations in the specified bulk async-group have completed reading
    from their source locations.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
  }];
  let arguments = (ins
    I32Attr:$group,
    UnitAttr:$read
  );
  let results = (outs);
}
def NVVM_CpAsyncCommitGroupOp : NVVM_Op<"cp.async.commit.group"> {
  let arguments = (ins);
  let results = (outs);
}
def NVVM_CpAsyncMBarrierArriveOp : NVVM_Op<"cp.async.mbarrier.arrive"> {
  let description = [{
    The `cp.async.mbarrier.arrive` Op makes the *mbarrier object* track
    all prior cp.async operations initiated by the executing thread.
    The `addr` operand specifies the address of the *mbarrier object*
    in generic or shared::cta address space. When it is generic, the
    underlying memory should fall within the shared::cta space;
    otherwise the behavior is undefined. The `noinc` attr impacts
    how the mbarrier's state is updated.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    OptionalAttr<I1Attr>:$noinc
  );
  let results = (outs);
}
def NVVM_CpAsyncOp : NVVM_Op<"cp.async.shared.global"> {
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace3:$dst,
    NVVM_LLVMPointerInAddressSpace1:$src,
    Optional<NVVM_T32BitSignlessInteger>:$cpSize,
    I32Attr:$size,
    NVVM_LoadCacheModifierAttr:$modifier
  );
  let results = (outs);
}
def NVVM_CpAsyncWaitGroupOp : NVVM_Op<"cp.async.wait.group"> {
  let arguments = (ins
    I32Attr:$n
  );
  let results = (outs);
}
def NVVM_CvtPackFloatOp : NVVM_Op<"cvt.packfloat"> {
  let description = [{
    This Op converts the given floating point input (`srcA`) to another
    packed floating point type (`res`). The operand `from` represents the
    source type and `to` represents the destination type.
    The result is merged with the given `srcC`. Both the input and result
    are represented as 'i32' type.

    Below table describes the format of the result:
    ```
    |--------------|-------------------------------------------------------|
    | Dst Types    | Description                                           |
    |--------------|-------------------------------------------------------|
    |   f16x2      |  The result contains two fp16 values, in the low      |
    |   bf16x2     |  and high bits.                                       |
    |--------------|-------------------------------------------------------|
    |   e4m3x2     |  The result contains two fp8 values in the low or     |
    |   e5m2x2     |  high 16-bits of `srcC` depending on `ExtractHi`.     |
    |   ue8m0x2    |  When `ExtractHi` is 0, the result is packed in       |
    |              |  bits `[7:0]` and `[15:8]` of `srcC` and the high     |
    |              |  16-bits of srcC remain unchanged.                    |
    |              |  When `ExtractHi` is 1, the result is packed in       |
    |              |  bits `[23:16]` and `[31:24]` of `srcC` and the low   |
    |              |  16-bits of srcC remain unchanged.                    |
    |--------------|-------------------------------------------------------|
    |   e2m3x2     |  The result contains two 6-bit floating point values  |
    |   e3m2x2     |  with the two MSB's padded with zeroes. The 16-bits   |
    |              |  of the result are then packed in the same format     |
    |              |  described above.                                     |
    |--------------|-------------------------------------------------------|
    |   e2m1x2     |  The result contains two 4-bit floating point values  |
    |              |  packed in bits `[7:0]` or `[23:16]` of `srcC`        |
    |              |  depending on `ExtractHi`. The other bits of `srcC`   |
    |              |  remain unchanged.                                    |
    |--------------|-------------------------------------------------------|
    ```

    Below table describes the format of the source operand:
    ```
    |--------------|-------------------------------------------------------|
    | Src Types    | Description                                           |
    |--------------|-------------------------------------------------------|
    |   f16x2      |  The two fp16 values are taken from the low and high  |
    |   bf16x2     |  high bits of operand `srcA`.                         |
    |--------------|-------------------------------------------------------|
    |   e4m3x2     |  The two fp8 values are taken from the low or high    |
    |   e5m2x2     |  16-bits of `srcA` depending on `ExtractHi`.          |
    |   ue8m0x2    |  When `ExtractHi` is 0, the inputs are taken from     |
    |              |  bits `[7:0]` and `[15:8]` of `srcA`.                 |
    |              |  When `ExtractHi` is 1, the inputs are sourced from   |
    |              |  bits `[23:16]` and `[31:24]` of `srcA`.              |
    |--------------|-------------------------------------------------------|
    |   e2m3x2     |  Similar to the row above.                            |
    |   e3m2x2     |                                                       |
    |--------------|-------------------------------------------------------|
    |   e2m1x2     |  The two 4-bit floating point input values are taken  |
    |              |  from bits `[7:0]` or `[23:16]` of `srcA` depending   |
    |              |  on `ExtractHi`.                                      |
    |--------------|-------------------------------------------------------|
    ```
    The `relu` attribute, when set, lowers to the '.relu' variant of the
    cvt instruction. The `rnd` and `sat` attributes specify the rounding
    and saturation respectively. The ue8m0x2, e2m3x2, e3m2x2 and e2m1x2
    types, either as source or destination types, are supported only in SM100a.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$srcA,
    NVVM_T32BitSignlessInteger:$srcC,
    NVVM_CVTPackFloatKindAttr:$from,
    NVVM_CVTPackFloatKindAttr:$to,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeKindAttr>:$sat,
    OptionalAttr<I1Attr>:$relu,
    OptionalAttr<I1Attr>:$extractHi
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_CvtPackFloatF32Op : NVVM_Op<"cvt.packfloat.f32"> {
  let description = [{
    This Op converts two given F32 inputs (in `srcA` and `srcB`) to a
    packed floating point type requested in `to`, and the
    result is merged with the given `srcC`.
    `extractHi`= 0: the result is packed into the low 16-bits of `srcC`.
    (and the high-16 bits of `srcC` remain as is)
    `extractHi`= 1: the result is packed into the high 16-bits of `srcC`.
    (and the low-16 bits of `srcC` remain as is)
    The result `res` is represented as an i32 type.
    The `relu` attribute, when set, lowers to the '.relu' variant of
    the cvt instruction. The `rnd` and `sat` attributes specify the
    the rounding and saturation modes respectively. The ue8m0x2,
    e2m3x2, e3m2x2 and e2m1x2 types are supported only in SM100a.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$srcA,
    NVVM_T32BitFloat:$srcB,
    NVVM_T32BitSignlessInteger:$srcC,
    NVVM_CVTPackFloatKindAttr:$to,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeKindAttr>:$sat,
    OptionalAttr<I1Attr>:$relu,
    OptionalAttr<I1Attr>:$extractHi
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_CvtToF4x8PackedOp : NVVM_Op<"cvt.to.f4x8.packed"> {
  let description = [{
    Fused conversion + packing: performs 4 pairwise conversions to f4 and
    packs the 4 byte-sized results into a single i32 using inline PTX
    `{ .reg .b8; 4x cvt; mov.b32 }`.

    This avoids the NVPTX backend's lack of a B8 register class, which
    causes 3x `prmt.b32` instructions instead of a single `mov.b32` when
    packing 4x i8 values via `BUILD_VECTOR v4i8`.

    For f32 source: takes 8 f32 values (4 high-low pairs).
    For f16 source: takes 4 i32 values (packed f16x2).
    For bf16 source: takes 4 i32 values (packed bf16x2).

    The `srcType` attribute specifies the source element type (f32, f16,
    or bf16) and is used to select the correct PTX cvt instruction suffix.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
  }];
  let arguments = (ins
    Variadic<NVVM_AnyType>:$srcs,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    TypeAttr:$srcType,
    TypeAttr:$dstType
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$dst
  );
}
def NVVM_DotAccumulate2WayOp : NVVM_Op<"dot.accumulate.2way"> {
  let description = [{
    Performs a two-way 16-bit to 8-bit dot-product which is accumulated in a
    32-bit result.
    Operand `a` is a vector of two 16-bit elements and operand `b` a vector
    of four 8-bit elements between which the dot product is computed.

    The `a_type` and `b_type` attributes specify the type of the elements in `a`
    and `b` respectively.
    If `a_type` or `b_type` is `s`, then the elements in the corresponding
    vector are sign-extended to 32-bit before the dot product is computed.
    If `a_type` or `b_type` is `u`, then the elements in the corresponding
    vector are zero-extended to 32-bit instead.

    The `b_hi` boolean attribute specifies which two bytes of `b` are used for
    the dot product. If `b_hi` is true, then the dot product is computed
    between  `a` and elements at indices 2 and 3 of `b`. If `b_hi` is false,
    then the dot product is computed between `a` and elements at indices 0 and
    1 of `b`.

    Operand `c` is a 32-bit integer to which the result is accumulated. It is
    treated as holding a signed integer if any of `a_type` or `b_type` is
    signed.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp2a)
  }];
  let arguments = (ins
    NVVM_VectorOf16BitSignlessInteger:$a,
    NVVM_VectorOf8BitSignlessInteger:$b,
    NVVM_T32BitSignlessInteger:$c,
    NVVM_DotAccumulateTypeAttr:$a_type,
    NVVM_DotAccumulateTypeAttr:$b_type,
    BoolAttr:$b_hi
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_DotAccumulate4WayOp : NVVM_Op<"dot.accumulate.4way"> {
  let description = [{
    Performs a four-way byte dot-product which is accumulated in a 32-bit
    result.
    Operand `a` and `b` are vectors of 4 bytes between which the dot product is
    computed.

    The `a_type` and `b_type` attributes specify the type of the elements in `a`
    and `b` respectively.
    If `a_type` or `b_type` is `signed`, then the elements in the corresponding
    vector are sign-extended to 32-bit before the dot product is computed.
    If `a_type` or `b_type` is `unsigned`, then the elements in the
    corresponding vector are zero-extended to 32-bit instead.

    Operand `c` is a 32-bit integer to which the result is accumulated. It is
    treated as holding a signed integer if any of `a_type` or `b_type` is `s8`.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp4a)
  }];
  let arguments = (ins
    NVVM_VectorOf8BitSignlessInteger:$a,
    NVVM_VectorOf8BitSignlessInteger:$b,
    NVVM_T32BitSignlessInteger:$c,
    NVVM_DotAccumulateTypeAttr:$a_type,
    NVVM_DotAccumulateTypeAttr:$b_type
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_ElectSyncOp : NVVM_Op<"elect.sync"> {
  let description = [{
    The `elect.sync` instruction elects one predicated active leader
    thread from among a set of threads specified in the `membermask`.
    When the `membermask` is not provided explicitly, a default value
    of `0xFFFFFFFF` is used. The predicate result is set to `True` for
    the leader thread, and `False` for all other threads.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync)
  }];
  let arguments = (ins
    Optional<NVVM_T32BitSignlessInteger>:$membermask
  );
  let results = (outs
    NVVM_T1BitSignlessInteger2:$pred
  );
}
def NVVM_Ex2Op : NVVM_Op<"ex2", [Pure]> {
  let description = [{
    Computes a fast approximation of 2 raised to the power of the input
    value. The `ftz` attribute, when set, flushes subnormal inputs and
    results to sign-preserving zero.
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_Exit : NVVM_Op<"exit"> {
  let description = [{
    Ends execution of a thread.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_FAbsOp : NVVM_InferredOp<"fabs"> {
  let description = [{
    Returns the absolute value of the floating point operand. The Op also has an
    optional `ftz` attribute which flushes subnormal inputs and returns to
    sign-preserving zero

    [For more information, see PTX ISA]
    (http://sw-mobile-docs/CUDA/GPGPU/parallel-thread-execution/index.html#floating-point-instructions-abs)
  }];
  let arguments = (ins
    NVVM_FloatingPointOrFixedLengthVector:$value,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_FloatingPointOrFixedLengthVector:$res
  );
}
def NVVM_FenceMbarrierInitOp : NVVM_Op<"fence.mbarrier.init"> {
  let description = [{
    Fence operation that applies on the prior nvvm.mbarrier.init

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_FenceProxyOp : NVVM_Op<"fence.proxy"> {
  let description = [{
    Fence operation with proxy to establish an ordering between memory accesses
    that may happen through different proxies.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins
    NVVM_ProxyKindAttr:$kind,
    OptionalAttr<NVVM_SharedSpaceAttr>:$space
  );
  let results = (outs);
}
def NVVM_FenceProxyAcquireOp : NVVM_Op<"fence.proxy.acquire"> {
  let description = [{
    `fence.proxy.acquire` is a uni-directional fence used to establish ordering
    between a prior memory access performed via the generic proxy and a
    subsequent memory access performed via the tensormap proxy

    The address operand `addr` and the operand `size` together specify the
    memory range `[addr, addr+size)` on which the ordering guarantees on the
    memory accesses across the proxies is to be provided. The only supported
    value for the `size` operand is 128 and must be an immediate. Generic Addressing
    is used unconditionally, and the address specified by the operand `addr` must
    fall within the `.global` state space. Otherwise, the behavior is undefined

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace0:$addr,
    NVVM_T32BitSignlessInteger:$size,
    NVVM_MemScopeKindAttr:$scope,
    OptionalAttr<NVVM_ProxyKindAttr>:$fromProxy,
    OptionalAttr<NVVM_ProxyKindAttr>:$toProxy
  );
  let results = (outs);
}
def NVVM_FenceProxyReleaseOp : NVVM_Op<"fence.proxy.release"> {
  let description = [{
    `fence.proxy.release` is a uni-directional fence used to establish ordering
    between a prior memory access performed via the generic proxy and a
    subsequent memory access performed via the tensormap proxy. `fence.proxy.release`
    operation can form a release sequence that synchronizes with an acquire
    sequence that contains the fence.proxy.acquire proxy fence operation

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins
    NVVM_MemScopeKindAttr:$scope,
    OptionalAttr<NVVM_ProxyKindAttr>:$fromProxy,
    OptionalAttr<NVVM_ProxyKindAttr>:$toProxy
  );
  let results = (outs);
}
def NVVM_FenceProxySyncRestrictOp : NVVM_Op<"fence.proxy.sync_restrict"> {
  let description = [{
    The `nvvm.fence.proxy.sync_restrict` Op used to establish
    ordering between a prior memory access performed between proxies. Currently,
    the ordering is only supported between async and generic proxies. `sync_restrict`
    restricts `acquire` memory semantics to `shared_cluster` and `release` memory
    semantics to `shared_cta` with cluster scope.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins
    NVVM_MemOrderKindAttr:$order,
    OptionalAttr<NVVM_ProxyKindAttr>:$fromProxy,
    OptionalAttr<NVVM_ProxyKindAttr>:$toProxy
  );
  let results = (outs);
}
def NVVM_FenceScClusterOp : NVVM_Op<"fence.sc.cluster"> {
  let arguments = (ins);
  let results = (outs);
}
def NVVM_FenceSyncRestrictOp : NVVM_Op<"fence.sync_restrict"> {
  let description = [{
    The `nvvm.fence.sync_restrict` Op restricts the class of memory
    operations for which the fence instruction provides the memory ordering guarantees.
    `sync_restrict` restricts `acquire` memory semantics to `shared_cluster` and
    `release` memory semantics to `shared_cta` with cluster scope.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins
    NVVM_MemOrderKindAttr:$order
  );
  let results = (outs);
}
def NVVM_FmaOp : NVVM_InferredOp<"fma"> {
  let description = [{
    The `nvvm.fma` operation performs floating point fused multiply-add of
    three operands of the same type.

    The rounding mode is specified by the `rnd` attribute, saturation mode by
    the `sat` attribute, flush-to-zero by the `ftz` attribute, and ReLU by the
    `relu` attribute.

    Out-of-bounds (OOB) behavior is controlled by the `oob` attribute. `oob`
    clamps the result to 0 if either of the operands is `OOB NaN` (see [Tensors](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensors)).

    For more information, see PTX ISA:
    - [floating point fused multiply-add](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma)
    - [half-precision floating point fused multiply-add](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma)
  }];
  let arguments = (ins
    NVVM_T16BitFloatOrBfloat16Type:$a,
    NVVM_T16BitFloatOrBfloat16Type:$b,
    NVVM_T16BitFloatOrBfloat16Type:$c,
    NVVM_FmaRndAttr:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$ftz,
    OptionalAttr<BoolAttr>:$relu,
    OptionalAttr<BoolAttr>:$oob
  );
  let results = (outs
    NVVM_T16BitFloatOrBfloat16Type:$res
  );
}
def NVVM_FMAPackedF32x2Op : NVVM_InferredOp<"fma.packed.f32x2"> {
  let description = [{
    This Op performs FMA of floating-point vectors.
    The inputs and outputs are of the form <N x f32>. The `rnd` attribute
    specifies the rounding mode. The `ftz` attribute, when set, flushes the
    subnormal inputs and results to sign-preserving zero.
    This Op is supported from SM100 and onwards.
    [For more information, see PTX ISA]
    (TODO: Add public PTX link when available)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues2:$srcA,
    NVVM_VectorOf32BitFloatValues2:$srcB,
    NVVM_VectorOf32BitFloatValues2:$srcC,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<I1Attr>:$ftz
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues2:$res
  );
}
def NVVM_FMAPacked_F32x2_BF16x2_F32x2_F32x2Op : NVVM_Op<"fma.packed.f32x2.bf16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs FMA of a pair of floating-point values.
    The first operand is a vector<2xbf16>, and the second and third operands
    are vector<2xf32>. The result is a vector<2xf32>. The first input is converted to
    a scalar of packed i32 where the 0th element of the vector forms the low-16 bits
    and the 1st element occupies the high-16 bits. The second and third inputs are
    converted to a scalar of packed i64 where the 0th element of the vector forms the
    low-32 bits and the 1st element occupies the high-32 bits.

    The rounding mode can be one of the following:
    - rn: round to nearest even
    - rz: round toward zero
    - rm: round toward negative infinity
    - rp: round toward positive infinity

    Examples:
    ```mlir
    // Round to nearest even (default)
    %0 = nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2 %srcA, %srcB, %srcC : vector<2xbf16>, vector<2xf32>, vector<2xf32> -> vector<2xf32>

    // Round toward negative infinity
    %1 = nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2 %srcA, %srcB, %srcC {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>, vector<2xf32>, vector<2xf32> -> vector<2xf32>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB,
    NVVM_VectorOf32BitFloatValues:$srcC,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues:$res
  );
}
def NVVM_FMAPacked_F32x2_F16x2_F32x2_F32x2Op : NVVM_Op<"fma.packed.f32x2.f16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs FMA of a pair of floating-point values.
    The first operand is a vector<2xf16>, and the second and third operands
    are vector<2xf32>. The result is a vector<2xf32>. The first input is converted to
    a scalar of packed i32 where the 0th element of the vector forms the low-16 bits
    and the 1st element occupies the high-16 bits. The second and third inputs are
    converted to a scalar of packed i64 where the 0th element of the vector forms the
    low-32 bits and the 1st element occupies the high-32 bits.

    The rounding mode can be one of the following:
    - rn: round to nearest even
    - rz: round toward zero
    - rm: round toward negative infinity
    - rp: round toward positive infinity

    Examples:
    ```mlir
    // Round to nearest even (default)
    %0 = nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2 %srcA, %srcB, %srcC : vector<2xf16>, vector<2xf32>, vector<2xf32> -> vector<2xf32>

    // Round toward negative infinity
    %1 = nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2 %srcA, %srcB, %srcC {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xf32>, vector<2xf32> -> vector<2xf32>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB,
    NVVM_VectorOf32BitFloatValues:$srcC,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues:$res
  );
}
def NVVM_MaxOp : NVVM_Op<"fmax"> {
  let description = [{
    `nvvm.fmax` Op calculates maximum of given values. if `nan` attributes is
    used, then the result is a canonical nan if any of the input is nan.
    if `abs` attribute is used, the magnitude of destination operation d is
    the maximum of absolute values of input arguments and for two operand
    version, `xorsign` is enabled by default which sets the sign bit of the
    result by XOR'ing the sign bits of the inputs. `ftz` modifiers flushes
    the subnormal inputs and the results to sign preserving zero
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$a,
    NVVM_T32BitFloat:$b,
    Optional<NVVM_T32BitFloat>:$c,
    UnitAttr:$ftz,
    UnitAttr:$nan,
    UnitAttr:$abs
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_MinOp : NVVM_Op<"fmin"> {
  let description = [{
    `nvvm.fmin` Op calculates minimum of given values. if `nan` attributes is
    used, then the result is a canonical nan if any of the input is nan.
    if `abs` attribute is used, the magnitude of destination operation d is
    the minimum of absolute values of input arguments and for two operand
    version, `xorsign` is enabled by default which sets the sign bit of the
    result by XOR'ing the sign bits of the inputs. `ftz` modifiers flushes
    the subnormal inputs and the results to sign preserving zero.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$a,
    NVVM_T32BitFloat:$b,
    Optional<NVVM_T32BitFloat>:$c,
    UnitAttr:$ftz,
    UnitAttr:$nan,
    UnitAttr:$abs
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_GriddepcontrolOp : NVVM_Op<"griddepcontrol"> {
  let description = [{
    If the $kind attribute is set to `wait`, it causes the
    executing thread to wait until all prerequisite grids in flight
    have completed and all the memory operations from the prerequisite grids
    are performed and made visible to the current grid.

    When the $kind is launch_dependents, it signals that specific dependents
    the runtime system designated to react to this instruction can be scheduled
    as soon as all other CTAs in the grid issue the same instruction or have
    completed.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)
  }];
  let arguments = (ins
    NVVM_GridDepActionAttr:$kind
  );
  let results = (outs);
}
def NVVM_InlinePtxOp : NVVM_Op<"inline_ptx", [AttrSizedOperandSegments]> {
  let description = [{
    This op allows using PTX directly within the NVVM
        dialect, while greatly simplifying llvm.inline_asm generation. It
        automatically handles register size selection and sets the correct
        read/write access for each operand. The operation leverages the
        `BasicPtxBuilderInterface` to abstract away low-level details of
        PTX assembly formatting.

        The `predicate` attribute is used to specify a predicate for the
        PTX instruction.

        Example 1: Read-only Parameters
        ```mlir
        nvvm.inline_ptx "mbarrier.init.b64 [$0], $1;" (%barrier_gen, %count) : !llvm.ptr, i32

        // Lowers to:
        llvm.inline_asm has_side_effects asm_dialect = att
          "mbarrier.init.b64 [$0], $1;", "l,r" %arg0, %arg2 : (!llvm.ptr, i32) -> ()
        ```

        Example 2: Read-only and Write-only Parameters
        ```mlir
        %0 = nvvm.inline_ptx "ex2.approx.ftz.f32 $0, $1;" (%input) : f32 -> f32

        // Lowers to:
        %0 = llvm.inline_asm has_side_effects asm_dialect = att
          "ex2.approx.ftz.f32 $0, $1;", "=f,f" %arg0 : (f32) -> f32
        ```

        Example 3: Predicate Usage
        ```mlir
        nvvm.inline_ptx "mbarrier.init.b64 [$0], $1;" (%barrier_gen, %count),
          predicate = %pred : !llvm.ptr, i32, i1

        // Lowers to:
        llvm.inline_asm has_side_effects asm_dialect = att
          "@$2 mbarrier.init.b64 [$0], $1;", "l,r,b" %arg0, %arg2, %arg3
          : (!llvm.ptr, i32, i1) -> ()
        ```
  }];
  let arguments = (ins
    Variadic<NVVM_AnyType>:$readOnlyArgs,
    Variadic<NVVM_AnyType>:$readWriteArgs,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    StrAttr:$ptxCode
  );
  let results = (outs
    Optional<NVVM_AnyType>:$writeOnlyArgs
  );
}
def NVVM_LdMatrixOp : NVVM_InferredOp<"ldmatrix"> {
  let arguments = (ins
    NVVM_LLVMPointerType:$ptr,
    I32Attr:$num,
    NVVM_MMALayoutAttr:$layout,
    OptionalAttr<NVVM_LoadShapeAttr>:$shape,
    OptionalAttr<NVVM_LoadSrcFormatAttr>:$srcFormat
  );
  let results = (outs
    NVVM_AnyType:$res
  );
}
def NVVM_ExtLoadOp : NVVM_Op<"load.ext"> {
  let description = [{
    The `load.ext` operation is used to read from memory in generic,
    global or shared address space. This Op explicitly supports
    various cache properties and memory orderings associated with
    the load operation.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
  }];
  let arguments = (ins
    NVVM_LLVMPointerType:$addr,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<NVVM_L2PrefetchSizeAttr>:$prefetch,
    OptionalAttr<NVVM_EvictKindAttr>:$evict,
    OptionalAttr<NVVM_LoadCacheModifierAttrExt>:$cacheModifier,
    OptionalAttr<NVVM_SharedSpaceAttr>:$sharedSpace,
    OptionalAttr<I1Attr>:$unified
  );
  let results = (outs
    NVVM_LLVMTypeWithSize:$res
  );
}
def NVVM_Log2Op : NVVM_Op<"log2", [Pure]> {
  let description = [{
    Computes a fast approximation of the base-2 logarithm of the input
    value. The `ftz` attribute, when set, flushes subnormal inputs and
    results to sign-preserving zero.
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_MapaOp : NVVM_Op<"mapa"> {
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$a,
    NVVM_T32BitSignlessInteger:$b
  );
  let results = (outs
    NVVM_LLVMPointerInAddressSpace03:$res
  );
}
def NVVM_MatchSyncOp : NVVM_InferredOp<"match.sync"> {
  let description = [{
    The `match.sync` op performs broadcast and compare of operand `val` across
    all non-exited threads in `thread_mask` and returns a mask depending on the
    kind and an optional predicate.

    The matching operation kinds are:
    - `any`: Returns a mask corresponding to the non-exited threads in the
    `thread_mask` that have the same value of operand `val`.
    - `all`: Returns a mask and a predicate. If all non-exited threads in the
    `thread_mask` have the same value of operand `val`, the predicate is set to
    true and the mask corresponds to the non-exited threads in the
    `thread_mask`. Otherwise, the predicate is set to false and the mask is 0.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$thread_mask,
    NVVM_T32BitSignlessIntegerOr64:$val,
    NVVM_MatchSyncKindAttr:$kind
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOrLLVM:$res
  );
}
def NVVM_MBarrierArriveOp : NVVM_InferredOp<"mbarrier.arrive", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mbarrier.arrive` operation performs an arrive-on operation on the
    *mbarrier object* at the specified address. Uses the default `.release.cta` semantics.
    This release pattern establishes memory ordering for operations occurring in program
    order before this arrive instruction by making operations from the current thread
    visible to subsequent operations in other threads within the CTA. When other threads
    perform corresponding acquire operations (like 'mbarrier.test.wait'), they synchronize
    with this release pattern.

    This operation causes the executing thread to signal its arrival at the barrier.

    - `res`: When the `space` is not shared_cluster, this operation returns an
      opaque 64-bit value capturing the phase of the *mbarrier object* prior to
      the arrive-on operation. The contents of this return value are
      implementation-specific. An *mbarrier object* located in the shared_cluster
      space cannot return a value.

    The operation takes the following operands:
    - `addr`: A pointer to the memory location of the *mbarrier object*. The `addr`
      must be a pointer to generic or shared_cta or shared_cluster memory. When it
      is generic, the underlying address must be within the shared_cta memory space;
      otherwise the behavior is undefined.
    - `count`: This specifies the amount by which the pending arrival count is
      decremented. If the `count` argument is not specified, the pending arrival
      count is decremented by 1.
    - `scope`: This specifies the set of threads that directly observe the memory
      synchronizing effect of the `mbarrier.arrive` operation.
    - `space`: This indicates the memory space where the mbarrier object resides.
    - `relaxed`: When set to true, the `arrive` operation has relaxed memory semantics
      and does not provide any ordering or visibility guarantees.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace04:$addr,
    Optional<NVVM_T32BitSignlessInteger>:$count,
    Optional<NVVM_T32BitSignlessInteger>:$multicastMask,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<BoolAttr>:$relaxed
  );
  let results = (outs
    Optional<NVVM_T64BitSignlessInteger>:$res
  );
}
def NVVM_MBarrierArriveExpectTxOp : NVVM_InferredOp<"mbarrier.arrive.expect_tx", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mbarrier.arrive.expect_tx` operation performs an expect-tx operation
    followed by an arrive-on operation on the *mbarrier object*. Uses the default
    `.release.cta` semantics. This release pattern establishes memory ordering for
    operations occurring in program order before this arrive instruction by making
    operations from the current thread visible to subsequent operations in other
    threads within the CTA. When other threads perform corresponding acquire operations
    (like 'mbarrier.test.wait'), they synchronize with this release pattern.

    This operation first performs an expect-tx operation with the specified transaction
    count, then performs an arrive-on operation with an implicit count of 1. The
    expect-tx operation increases the expect-count of the *mbarrier object* by the
    specified value (i.e. `txcount`), setting the current phase to expect and track
    the completion of additional asynchronous transactions.

    The operation takes the following operands:
    - `addr`: A pointer to the memory location of the *mbarrier object*. Uses generic
      addressing, but the address must still be in the shared memory space.
    - `txcount`: An unsigned integer specifying the expected transaction count
      for the expect-tx operation. This represents the number of asynchronous transactions
      expected to complete before the barrier phase completes.
    - `scope`: This specifies the set of threads that directly observe the memory
      synchronizing effect of the `mbarrier.test.wait` operation.
    - `relaxed`: When set to true, the `arrive` operation has relaxed memory semantics
      and does not provide any ordering or visibility guarantees.
    - `predicate`: Optional predicate for conditional execution used only when lowering to
      inline-ptx.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace04:$addr,
    NVVM_T32BitSignlessInteger:$txcount,
    Optional<NVVM_T32BitSignlessInteger>:$multicastMask,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<BoolAttr>:$relaxed
  );
  let results = (outs
    Optional<NVVM_T64BitSignlessInteger>:$res
  );
}
def NVVM_MBarrierArriveNocompleteOp : NVVM_Op<"mbarrier.arrive.nocomplete"> {
  let description = [{
    The `nvvm.mbarrier.arrive.nocomplete` operation performs an arrive-on operation
    on the *mbarrier object* with the guarantee that it will not cause the barrier to
    complete its current phase. Uses the default `.release.cta` semantics. This release
    pattern establishes memory ordering for operations occurring in program order before
    this arrive instruction by making operations from the current thread visible to
    subsequent operations in other threads within the CTA. When other threads perform
    corresponding acquire operations (like 'mbarrier.test.wait'), they synchronize with
    this release pattern.

    This operation causes the executing thread to signal its arrival at the barrier
    with a specified count, but ensures that the barrier phase will not complete as
    a result of this operation. The operation returns an opaque value that
    captures the phase of the *mbarrier object* prior to the arrive-on operation.

    The operation takes the following operands:
    - `addr`: A pointer to the memory location of the *mbarrier object*. The `addr`
      must be a pointer to generic or shared::cta memory. When it is generic, the
      underlying address must be within the shared::cta memory space; otherwise
      the behavior is undefined.
    - `count`: Integer specifying the count argument to the arrive-on operation.
      Must be in the valid range as specified in the *mbarrier object* contents.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T32BitSignlessInteger:$count
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierArriveDropOp : NVVM_InferredOp<"mbarrier.arrive_drop", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mbarrier.arrive_drop` operation decrements the expected arrival
    count of the *mbarrier object* by `count` and then performs an arrive-on
    operation. When `count` is not specified, it defaults to 1. The decrement
    of the expected arrival count applies to all the subsequent phases of the
    *mbarrier object*. The remaining semantics are identical to those of the
    `nvvm.mbarrier.arrive` operation.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace04:$addr,
    Optional<NVVM_T32BitSignlessInteger>:$count,
    Optional<NVVM_T32BitSignlessInteger>:$multicastMask,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<BoolAttr>:$relaxed
  );
  let results = (outs
    Optional<NVVM_T64BitSignlessInteger>:$res
  );
}
def NVVM_MBarrierArriveDropExpectTxOp : NVVM_InferredOp<"mbarrier.arrive_drop.expect_tx"> {
  let description = [{
    The `nvvm.mbarrier.arrive_drop.expect_tx` operation is similar to the
    `nvvm.mbarrier.arrive.expect_tx` operation except that it performs an
    `arrive_drop` operation instead of only an `arrive` operation.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace04:$addr,
    NVVM_T32BitSignlessInteger:$txcount,
    Optional<NVVM_T32BitSignlessInteger>:$multicastMask,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<BoolAttr>:$relaxed
  );
  let results = (outs
    Optional<NVVM_T64BitSignlessInteger>:$res
  );
}
def NVVM_MBarrierArriveDropNocompleteOp : NVVM_Op<"mbarrier.arrive_drop.nocomplete"> {
  let description = [{
    The `nvvm.mbarrier.arrive_drop.nocomplete` operation decrements the expected
    arrival count of the *mbarrier object* by the amount `count` and then performs
    an arrive-on operation on the *mbarrier object* with the guarantee that it
    will not cause the barrier to complete its current phase.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T32BitSignlessInteger:$count
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierCheckLayoutOp : NVVM_Op<"mbarrier.check.layout"> {
  let description = [{
    The `nvvm.mbarrier.check.layout` operation returns `true` if the *mbarrier object* at `addr`
    uses the given `layout`; otherwise it returns `false`.
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace3:$addr,
    NVVM_MBarrierLayoutAttr:$layout
  );
  let results = (outs
    NVVM_T1BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierCompleteTxOp : NVVM_Op<"mbarrier.complete_tx", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mbarrier.complete_tx` operation decrements the transaction
    count of the *mbarrier object* at `addr` by `txcount`. It also signals
    the completion of asynchronous transactions that were tracked by the
    current phase. The `scope` specifies the set of threads that can directly
    observe the memory synchronizing effect of the `mbarrier.complete_tx`
    operation. `CTA` and `CLUSTER` are the only allowed values for `scope`.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace32:$addr,
    NVVM_T32BitSignlessInteger:$txcount,
    Optional<NVVM_T32BitSignlessInteger>:$report,
    Optional<NVVM_T32BitSignlessInteger>:$multicastMask,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope
  );
  let results = (outs);
}
def NVVM_MBarrierExpectTxOp : NVVM_Op<"mbarrier.expect_tx"> {
  let description = [{
    The `nvvm.mbarrier.expect_tx` operation increases the transaction count
    of the mbarrier located at `addr` by `txcount` amount. The `scope`
    specifies the set of threads that can directly observe the memory
    synchronizing effect of the `mbarrier.expect_tx` operation. `CTA`
    and `CLUSTER` are the only allowed values for `scope`.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace32:$addr,
    NVVM_T32BitSignlessInteger:$txcount,
    Optional<NVVM_T32BitSignlessInteger>:$multicastMask,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope
  );
  let results = (outs);
}
def NVVM_MBarrierInitOp : NVVM_Op<"mbarrier.init"> {
  let description = [{
    The `nvvm.mbarrier.init` operation initializes an *mbarrier object* at the specified
    memory location.

    This operation initializes the *mbarrier object* with the following state:
    - Current phase: 0
    - Expected arrival count: `count`
    - Pending arrival count: `count`
    - Transaction count (tx-count): 0

    The operation takes the following operands:
    - `addr`: A pointer to the memory location of the *mbarrier object*. The `addr`
      must be a pointer to generic or shared::cta memory. When it is generic, the
      underlying address must be within the shared::cta memory space; otherwise
      the behavior is undefined.
    - `count`: Integer specifying the number of threads that will participate in barrier
      synchronization. Must be in the range [1, 2²⁰ - 1].
    - `predicate`: Optional predicate for conditional execution.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T32BitSignlessInteger:$count,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    OptionalAttr<NVVM_MBarrierLayoutAttr>:$layout
  );
  let results = (outs);
}
def NVVM_MBarrierInvalOp : NVVM_Op<"mbarrier.inval"> {
  let description = [{
    The `nvvm.mbarrier.inval` operation invalidates an *mbarrier object* at the
    specified memory location.

    This operation marks the *mbarrier object* as invalid, making it safe to repurpose
    the memory location for other uses or to reinitialize it as a new *mbarrier object*.
    It is undefined behavior if the *mbarrier object* is already invalid.

    The operation takes the following operand:
    - `addr`: A pointer to the memory location of the *mbarrier object*. The `addr`
      must be a pointer to generic or shared::cta memory. When it is generic, the
      underlying address must be within the shared::cta memory space; otherwise
      the behavior is undefined.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr
  );
  let results = (outs);
}
def NVVM_MBarrierTestWaitOp : NVVM_InferredOp<"mbarrier.test.wait"> {
  let description = [{
    The `nvvm.mbarrier.test.wait` operation performs a non-blocking test for the
    completion of a specific phase of an *mbarrier object*. It uses the default
    `.acquire.cta` semantics. This acquire pattern establishes memory ordering for
    operations occurring in program order after this wait instruction by making
    operations from other threads in the CTA visible to subsequent operations in the current
    thread. When this wait completes, it synchronizes with the corresponding release
    pattern from the `mbarrier.arrive` operation, establishing memory ordering within
    the CTA.

    This operation tests whether the mbarrier phase specified by the state operand
    has completed. It is a non-blocking instruction that immediately returns the
    completion status without suspending the executing thread.

    The operation takes the following operands:
    - `addr`: A pointer to the memory location of the *mbarrier object*. Uses generic
      addressing, but the address must still be in the shared memory space.
    - `stateOrPhase`: This argument represents a `state` when it is a 64-bit value
      and represents a `phase` when it is a 32-bit value. The `state` is an opaque
      value returned by a previous `mbarrier.arrive` operation on the same
      *mbarrier object* during the current or immediately preceding phase.
      The `phase` is an integer specifying the phase parity (0 or 1).
      Even phases have parity 0, odd phases have parity 1.
    - `scope`: This specifies the set of threads that directly observe the memory
      synchronizing effect of the `mbarrier.test.wait` operation.
    - `relaxed`: When set to true, the `arrive` operation has relaxed memory semantics
      and does not provide any ordering or visibility guarantees.

    The operation returns a boolean value indicating whether the specified phase
    has completed:
    - `true`: The immediately preceding phase has completed
    - `false`: The phase is still incomplete (current phase)

    The primary-phase variants the result is an LLVM struct whose first
    field (`res`) is the `i1` completion flag. An optional second field
    (`isReportSeen`, `i1`) indicates whether a report was observed, and an
    optional third field (`report`, `i32`) carries the report payload.

    **Memory ordering guarantees**: When this wait returns true, the following
    ordering guarantees hold:

    1. All memory accesses (except async operations) requested prior to
       `mbarrier.arrive` having release semantics by participating CTA threads
       are visible to the executing thread.
    2. All `cp.async` operations requested prior to `cp.async.mbarrier.arrive`
       by participating CTA threads are visible to the executing thread.
    3. All `cp.async.bulk` operations using the same *mbarrier object* requested
       prior to `mbarrier.arrive` having release semantics by participating CTA
       threads are visible to the executing thread.
    4. Memory accesses requested after this wait are not visible to memory
       accesses performed prior to `mbarrier.arrive` by other participating
       threads.
    5. No ordering guarantee exists for memory accesses by the same thread
       between `mbarrier.arrive` and this wait.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T64BitSignlessIntegerOr32:$stateOrPhase,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<BoolAttr>:$relaxed,
    OptionalAttr<NVVM_NVVM_MBarrierPhaseAttr>:$mbarrierPhase
  );
  let results = (outs
    NVVM_T1BitSignlessIntegerOrLLVM:$res
  );
}
def NVVM_MBarrierTryWaitOp : NVVM_InferredOp<"mbarrier.try_wait"> {
  let description = [{
    The `nvvm.mbarrier.try_wait` operation checks whether the specified
    *mbarrier object* at `addr` has completed the given phase. Note that
    unlike the `nvvm.mbarrier.test.wait` operation, the try_wait operation
    is a potentially-blocking one. If the phase is not yet complete, the
    calling thread may be suspended. A suspended thread resumes execution
    once the phase completes or when a system-defined timeout occurs.
    Optionally, the `ticks` operand can be used to provide a custom timeout
    (in nanoseconds), overriding the system-defined one. The semantics of
    this operation and its operands are otherwise similar to those of the
    `nvvm.mbarrier.test.wait` Op.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)

    The primary-phase variants the result is an LLVM struct whose first
    field (`res`) is the `i1` completion flag. An optional second field
    (`isReportSeen`, `i1`) indicates whether a report was observed, and an
    optional third field (`report`, `i32`) carries the report payload.
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T64BitSignlessIntegerOr32:$stateOrPhase,
    Optional<NVVM_T32BitSignlessInteger>:$ticks,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<BoolAttr>:$relaxed,
    OptionalAttr<NVVM_NVVM_MBarrierPhaseAttr>:$mbarrierPhase
  );
  let results = (outs
    NVVM_T1BitSignlessIntegerOrLLVM:$res
  );
}
def NVVM_MBarrierTryWaitParityOp : NVVM_Op<"mbarrier.try_wait.parity"> {
  let description = [{
    The `nvvm.mbarrier.try_wait.parity` operation performs a potentially-blocking
    test for the completion of a specific phase of an *mbarrier object* using phase
    parity. It uses the default `.acquire.cta` semantics. This acquire pattern
    establishes memory ordering for operations occurring in program order after this
    wait instruction by making operations from other threads in the CTA visible to subsequent
    operations in the current thread. When this wait completes, it synchronizes with
    the corresponding release pattern from the `mbarrier.arrive` operation, establishing
    memory ordering within the CTA.

    This operation waits for the completion of the mbarrier phase indicated by the
    phase parity. While it uses the underlying PTX `mbarrier.try_wait.parity`
    instruction, this MLIR operation generates a loop that enforces the test to
    complete before continuing execution, ensuring the barrier phase is actually
    completed rather than potentially timing out.

    The operation takes the following operands:
    - `addr`: A pointer to the memory location of the *mbarrier object*. Uses generic
      addressing, but the address must still be in the shared memory space.
    - `phase`: An integer specifying the phase parity (0 or 1). Even phases
      have parity 0, odd phases have parity 1.
    - `ticks`: An unsigned integer specifying the suspend time hint in
      nanoseconds. This may be used instead of the system-dependent time limit.

    **Memory ordering guarantees**: When this wait returns true, the following
    ordering guarantees hold:

    1. All memory accesses (except async operations) requested prior to
       `mbarrier.arrive` having release semantics by participating CTA threads
       are visible to the executing thread.
    2. All `cp.async` operations requested prior to `cp.async.mbarrier.arrive`
       by participating CTA threads are visible to the executing thread.
    3. All `cp.async.bulk` operations using the same *mbarrier object* requested
       prior to `mbarrier.arrive` having release semantics by participating CTA
       threads are visible to the executing thread.
    4. Memory accesses requested after this wait are not visible to memory
       accesses performed prior to `mbarrier.arrive` by other participating
       threads.
    5. No ordering guarantee exists for memory accesses by the same thread
       between `mbarrier.arrive` and this wait.

    **Implementation behavior**:
    This operation generates a PTX loop that repeatedly calls the underlying
    `mbarrier.try_wait.parity` instruction until the barrier phase completes.
    Unlike the raw PTX instruction which may return without completion after a
    timeout, this MLIR operation guarantees completion by continuing to loop until
    the specified phase is reached.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T32BitSignlessInteger:$phase,
    NVVM_T32BitSignlessInteger:$ticks,
    OptionalAttr<I1Attr>:$useIntrinsic
  );
  let results = (outs);
}
def NVVM_MBarrierTryWaitParityTimeLimitOp : NVVM_Op<"mbarrier.try_wait.parity.timelimit"> {
  let description = [{
    The `mbarrier.try_wait.parity.timelimit` is a potentially blocking
    instruction which tests for the completion of the phase. If
    the phase is not complete, the executing thread may be suspended.
    Suspended thread resumes execution when the specified phase completes
    or a system-dependent time limit elapses. The 32-bit `timeLimit` operand
    specifies the time limit, in nanoseconds, that may be used instead of
    the system-dependent limit. The `addr` operand specifies the address
    of the mbarrier in generic or shared memory space. The `phase` operand
    is the integer parity of either the current phase or the immediately
    preceding phase of the mbarrier object. The `order` denotes the
    ordering semantics for the memory consistency model. This Op only
    supports 'acquire' or 'relaxed' ordering.
    [For more information refer PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T32BitSignlessInteger:$phase,
    NVVM_T32BitSignlessInteger:$timeLimit,
    OptionalAttr<NVVM_MBarrierScopeKindAttr>:$scope,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order
  );
  let results = (outs
    NVVM_T1BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierTryWaitTimeLimitOp : NVVM_Op<"mbarrier.try_wait.timelimit"> {
  let description = [{
    The `mbarrier.try_wait.timelimit` is a potentially blocking
    instruction which tests for the completion of the phase. If
    the phase is not complete, the executing thread may be suspended.
    Suspended thread resumes execution when the specified phase completes
    or a system-dependent time limit elapses. The 32-bit `timeLimit` operand
    specifies the time limit, in nanoseconds, that may be used instead of
    the system-dependent limit. The `addr` operand specifies the address
    of the mbarrier in generic or shared memory space. The `state` operand
    is the captured state of the mbarrier object from prior operations.
    The `order` denotes the ordering semantics for the memory consistency
    model. This Op only supports 'acquire' or 'relaxed' ordering.
    [For more information refer PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T64BitSignlessInteger:$state,
    NVVM_T32BitSignlessInteger:$timeLimit,
    OptionalAttr<NVVM_MBarrierScopeKindAttr>:$scope,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order
  );
  let results = (outs
    NVVM_T1BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierTransactionOp : NVVM_Op<"mbarrier.txn"> {
  let description = [{
    The `mbarrier.txn` Op exposes the transactions that can be done on an MBarrier object.

    Operands:
    - `addr`: Address of the mbarrier in generic or shared memory space
    - `count`: Transaction count for the given kind

    - `multicast`: (Optional) Multicast mask for cluster space operations


    Attributes:
    - `kind`: One of the transaction kinds from MBarrierTxnKind
    - `space`: Memory space (CTA or CLUSTER) where MBarrier object is located
    - `scope`: Scope of synchronization (CTA or CLUSTER)
    - `order`: Memory ordering semantics (release or relaxed only)

    This Op supports MBarrier objects in both shared::CTA and shared::CLUSTER spaces.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T32BitSignlessInteger:$count,
    Optional<NVVM_T32BitSignlessInteger>:$multicast,
    NVVM_MBarrierTxnKindAttr:$kind,
    OptionalAttr<NVVM_MBarrierSpaceKindAttr>:$space,
    OptionalAttr<NVVM_MBarrierScopeKindAttr>:$scope,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order
  );
  let results = (outs);
}
def NVVM_MBarrierTransactionCTASpaceOp : NVVM_Op<"mbarrier.txn.cta"> {
  let description = [{
    The `mbarrier.txn.cta` Op exposes the transactions that can be done
    using an MBarrier object. The `addr` operand specifies the address of
    the mbarrier in generic or shared memory space. The `kind` is one of
    the various transaction kinds supported in `MBarrierTxnKind`. The
    `count` argument is the transaction count for a given `kind`. The
    `order` denotes the ordering semantics for the memory consistency
    model. This Op only supports 'release' or 'relaxed' ordering. This
    Op supports only MBarrier objects located in shared::CTA space (but
    not in shared::CLUSTER space) and hence the 'CTASpace' suffix in the
    naming. This Op returns an opaque 64-bit value capturing the phase of
    the mbarrier object prior to the transaction.
    [For more information on 'order/space/scope/kind-of-transactions' refer PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T32BitSignlessInteger:$count,
    NVVM_MBarrierTxnKindAttr:$kind,
    OptionalAttr<NVVM_MBarrierScopeKindAttr>:$scope,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierWaitOp : NVVM_Op<"mbarrier.wait"> {
  let description = [{
    The `mbarrier.wait` Op exposes the try/test wait operations that
    can be performed on MBarrier object. The `addr` operand specifies
    the address of the mbarrier in generic or shared memory space.
    The `state` operand is the captured state of the mbarrier object
    from prior operations.  The `kind` is one of TEST_WAIT or TRY_WAIT.
    The `order` denotes the ordering semantics for the memory consistency
    model. This Op only supports 'acquire' or 'relaxed' ordering.
    [For more information refer PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T64BitSignlessInteger:$state,
    NVVM_MBarrierWaitKindAttr:$kind,
    OptionalAttr<NVVM_MBarrierScopeKindAttr>:$scope,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order
  );
  let results = (outs
    NVVM_T1BitSignlessInteger2:$res
  );
}
def NVVM_MBarrierWaitParityOp : NVVM_Op<"mbarrier.wait.parity"> {
  let description = [{
    The `mbarrier.wait` Op exposes the try/test wait operations that
    can be performed on MBarrier object. The `addr` operand specifies
    the address of the mbarrier in generic or shared memory space.
    The `phase` operand is the integer parity of either the current
    phase or the immediately preceding phase of the mbarrier object.
    The `kind` is one of TEST_WAIT or TRY_WAIT.
    The `order` denotes the ordering semantics for the memory consistency
    model. This Op only supports 'acquire' or 'relaxed' ordering.
    [For more information refer PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T32BitSignlessInteger:$phase,
    NVVM_MBarrierWaitKindAttr:$kind,
    OptionalAttr<NVVM_MBarrierScopeKindAttr>:$scope,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order
  );
  let results = (outs
    NVVM_T1BitSignlessInteger2:$res
  );
}
def NVVM_MembarOp : NVVM_Op<"memory.barrier"> {
  let description = [{
    `membar` operation guarantees that prior memory accesses requested by this
    thread are performed at the specified `scope`, before later memory
    operations requested by this thread following the membar instruction.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar)
  }];
  let arguments = (ins
    NVVM_MemScopeKindAttr:$scope
  );
  let results = (outs);
}
def NVVM_MmaBlockScaleOp : NVVM_Op<"mma.block_scale", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mma.block_scale` operation collectively performs the operation
    `D = matmul(A * SF_A, B * SF_B) + C` using all threads in a warp.

    A, B, C and D are dense matrices and SF_A and SF_B are scaling factors.
    Dimensions of SF_A and SF_B are based on scale vector sizes (x1, x2, x4),
    and the data type must be either ue8m0 or ue4m3.

    All the threads in the warp must execute the same `mma.block_scale` operation.

    This operation follows the same design pattern as `nvvm.mma.sync`, with additional
    scaling operands for both A and B matrices.

    Example:
    ```mlir
    %d = nvvm.mma.block_scale A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                              scaleA[%scaleAData, %byteIdA, %threadIdA]
                              scaleB[%scaleBData, %byteIdB, %threadIdB]
                              {shape = #nvvm.shape<m = 16, n = 8, k = 64>,
                               multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                               multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                               scaleVecSize = #nvvm.scale_vec_size<x2>,
                               blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                               kind = #nvvm.block_scale_kind<mxf4nvf4>}
        : (vector<4xf16>, vector<2xf16>, vector<2xf32>) -> !llvm.struct<(f32, f32)>
    ```
  }];
  let arguments = (ins
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandA,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandB,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandC,
    NVVM_T32BitSignlessInteger:$scaleAData,
    NVVM_T16BitSignlessInteger:$byteIdA,
    NVVM_T16BitSignlessInteger:$threadIdA,
    NVVM_T32BitSignlessInteger:$scaleBData,
    NVVM_T16BitSignlessInteger:$byteIdB,
    NVVM_T16BitSignlessInteger:$threadIdB,
    NVVM_MMAShapeAttr:$shape,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandAPtxType,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandBPtxType,
    NVVM_ScaleVecSizeAttr:$scaleVecSize,
    NVVM_BlockScaleFormatAttr:$blockScaleFormat,
    NVVM_MMABlockScaleKindAttr:$kind
  );
  let results = (outs
    NVVM_LLVMStructureType:$res
  );
}
def NVVM_InternalMmaBlockScaleOp : NVVM_InferredOp<"mma.block_scale.internal"> {
  let description = [{
    The `nvvm.mma.blockscale.internal` operation collectively performs the operation
    `D = matmul(A * SF_A, B * SF_B) + C` using all threads in a warp.

    A, B, C and D are dense matrices and SF_A and SF_B are scaling factor.
    Dimensions of SF_A and SF_B are based on scale vector sizes (X1, X2, X4),
    the data type of must be either ue8m0 or ue4m3

    Example:
    ```mlir

    %0 = nvvm.mma.block_scale.internal %operandA, %scaleAData, %byteIdA, %threadIdA,
                                       %operandB, %scaleBData, %byteIdB, %threadIdB,
                                       %operandC
    {
      layoutA = #nvvm.mma_layout<row>,
      layoutB = #nvvm.mma_layout<col>,
      shape = #nvvm.shape<m = 16, n = 8, k = 64>,
      aType = #nvvm.mma_type<e2m1>,
      bType = #nvvm.mma_type<e2m1>,
      cType = #nvvm.mma_type<f32>,
      scaleVecSize = #nvvm.scale_vec_size<x2>,
      blockScaleFormat = #nvvm.block_scale_format<ue8m0>
      kind = #nvvm.block_scale_kind<mxf4nvf4>
    }
    ```
  }];
  let arguments = (ins
    NVVM_VectorOfAnyTypeValues:$operandA,
    NVVM_T32BitSignlessInteger:$scaleAData,
    NVVM_T16BitSignlessInteger:$byteIdA,
    NVVM_T16BitSignlessInteger:$threadIdA,
    NVVM_VectorOfAnyTypeValues:$operandB,
    NVVM_T32BitSignlessInteger:$scaleBData,
    NVVM_T16BitSignlessInteger:$byteIdB,
    NVVM_T16BitSignlessInteger:$threadIdB,
    NVVM_VectorOfAnyTypeValues:$operandC,
    NVVM_MMAShapeAttr:$shape,
    NVVM_MMALayoutAttr:$layoutA,
    NVVM_MMALayoutAttr:$layoutB,
    NVVM_MMATypesAttr:$aType,
    NVVM_MMATypesAttr:$bType,
    NVVM_MMATypesAttr:$cType,
    NVVM_ScaleVecSizeAttr:$scaleVecSize,
    NVVM_BlockScaleFormatAttr:$blockScaleFormat,
    OptionalAttr<NVVM_MMABlockScaleKindAttr>:$kind
  );
  let results = (outs
    NVVM_VectorOfAnyTypeValues:$res
  );
}
def NVVM_MmaSpBlockScaleOp : NVVM_Op<"mma.sp.block_scale", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mma.sp.block_scale` operation collectively performs the operation
    `D = matmul(A_sparse * SF_A, B * SF_B) + C` using all threads in a warp.

    A is a sparse matrix, and B, C and D are dense matrices.
    SF_A and SF_B are scaling factors.
    Dimensions of SF_A and SF_B are based on scale vector sizes (x1, x2, x4),
    and the data type must be either ue8m0 or ue4m3.

    This operation is similar to `nvvm.mma.block_scale` but with structured sparsity
    in the A operand. The sparsity follows the 2:4 structured sparse pattern
    where 2 out of every 4 elements are non-zero.

    All the threads in the warp must execute the same `mma.sp.block_scale` operation.

    The `sparseMetadata` operand provides the sparsity indices that indicate
    which elements in the A operand are non-zero. The `sparsitySelector`
    controls how the indices are distributed among threads in the warp and
    should typically be 0 or 1.

    This operation follows the same design pattern as `nvvm.mma.sp.sync`, with additional
    scaling operands for both A and B matrices. Note that sparse block scale operations
    always use ordered metadata (sm_90+).

    Example:
    ```mlir
    %d = nvvm.mma.sp.block_scale A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                                 sparseMetadata[%meta] selector[%sel]
                                 scaleA[%scaleAData, %byteIdA, %threadIdA]
                                 scaleB[%scaleBData, %byteIdB, %threadIdB]
                                 {shape = #nvvm.shape<m = 16, n = 8, k = 128>,
                                  multiplicandAPtxType = #nvvm.mma_type<e2m1>,
                                  multiplicandBPtxType = #nvvm.mma_type<e2m1>,
                                  scaleVecSize = #nvvm.scale_vec_size<x2>,
                                  blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
                                  kind = #nvvm.block_scale_kind<mxf4>}
        : (vector<2xf16>, vector<2xf16>, vector<2xf32>) -> !llvm.struct<(f32, f32)>
    ```
  }];
  let arguments = (ins
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandA,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandB,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandC,
    NVVM_T32BitSignlessInteger:$sparseMetadata,
    NVVM_T32BitSignlessInteger:$sparsitySelector,
    NVVM_T32BitSignlessInteger:$scaleAData,
    NVVM_T16BitSignlessInteger:$byteIdA,
    NVVM_T16BitSignlessInteger:$threadIdA,
    NVVM_T32BitSignlessInteger:$scaleBData,
    NVVM_T16BitSignlessInteger:$byteIdB,
    NVVM_T16BitSignlessInteger:$threadIdB,
    NVVM_MMAShapeAttr:$shape,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandAPtxType,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandBPtxType,
    NVVM_ScaleVecSizeAttr:$scaleVecSize,
    NVVM_BlockScaleFormatAttr:$blockScaleFormat,
    NVVM_MMABlockScaleKindAttr:$kind,
    UnitAttr:$orderedMetadata
  );
  let results = (outs
    NVVM_LLVMStructureType:$res
  );
}
def NVVM_InternalMmaSparseBlockScaleOp : NVVM_InferredOp<"mma.sp.block_scale.internal"> {
  let description = [{
    The `nvvm.mma.sparse.blockscale` operation collectively performs the
    operation `D = matmul(A * SF_A, B * SF_B) + C` using all threads in a warp.

    Where A is structued sparse with atleast 50% zeros and B, C and D are
    dense and SF_A and SF_B are scaling factor. Dimensions of SF_A and SF_B
    are based on scale vector sizes (X1, X2, X4), the data type of
    must be either ue8m0 or ue4m3

    Example:
    ```mlir

    %selector = llvm.mlir.constant(0)
    %0 = nvvm.mma.sparse.block_scale
                    %operandA, %scaleAData, %byteIdA, %threadIdA,
                    %operandB, %scaleBData, %byteIdB, %threadIdB,
                    %operandC, %spMetadata, %selector
    {
      layoutA = #nvvm.mma_layout<row>,
      layoutB = #nvvm.mma_layout<col>,
      shape = #nvvm.shape<m = 16, n = 8, k = 64>,
      aType = #nvvm.mma_type<e2m1>,
      bType = #nvvm.mma_type<e2m1>,
      cType = #nvvm.mma_type<f32>,
      scaleVecSize = #nvvm.scale_vec_size<x2>,
      blockScaleFormat = #nvvm.block_scale_format<ue8m0>,
      sparityFormat = #nvvm.sparsity_format<thread>
    }
    ```
  }];
  let arguments = (ins
    NVVM_VectorOfAnyTypeValues:$operandA,
    NVVM_T32BitSignlessInteger:$scaleAData,
    NVVM_T16BitSignlessInteger:$byteIdA,
    NVVM_T16BitSignlessInteger:$threadIdA,
    NVVM_VectorOfAnyTypeValues:$operandB,
    NVVM_T32BitSignlessInteger:$scaleBData,
    NVVM_T16BitSignlessInteger:$byteIdB,
    NVVM_T16BitSignlessInteger:$threadIdB,
    NVVM_VectorOfAnyTypeValues:$operandC,
    NVVM_T32BitSignlessInteger:$sp_metadata,
    NVVM_T32BitSignlessInteger:$selector,
    NVVM_MMAShapeAttr:$shape,
    OptionalAttr<NVVM_MMAB1OpAttr>:$b1Op,
    OptionalAttr<NVVM_MMAIntOverflowAttr>:$intOverflowBehavior,
    NVVM_MMALayoutAttr:$layoutA,
    NVVM_MMALayoutAttr:$layoutB,
    NVVM_MMATypesAttr:$aType,
    NVVM_MMATypesAttr:$bType,
    NVVM_MMATypesAttr:$cType,
    NVVM_ScaleVecSizeAttr:$scaleVecSize,
    NVVM_BlockScaleFormatAttr:$blockScaleFormat,
    NVVM_SparsityFormatAttr:$sparsityFormat
  );
  let results = (outs
    NVVM_VectorOfAnyTypeValues:$res
  );
}
def NVVM_MmaSpOp : NVVM_Op<"mma.sp.sync", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mma.sp.sync` operation collectively performs the sparse operation
    `D = matmul(A_sparse, B) + C` using all threads in a warp.

    This operation is similar to `nvvm.mma.sync` but with structured sparsity
    in the A operand. The sparsity follows the 2:4 structured sparse pattern
    where 2 out of every 4 elements are non-zero.

    All the threads in the warp must execute the same `mma.sp.sync` operation.

    The `sparseMetadata` operand provides the sparsity indices that indicate
    which elements in the A operand are non-zero. The `sparsitySelector`
    controls how the indices are distributed among threads in the warp and
    should typically be 0 or 1.

    The optional `orderedMetadata` attribute specifies the metadata ordering:
    - Absence (default): Uses standard sparse metadata ordering
    - Presence: Uses ordered metadata (PTX ISA 8.5+, sm_90+)

    The optional `kind` attribute specifies mixed-precision modes for FP8 operations:
    - `f8f6f4`: Enables e3m2, e2m3, e2m1 FP8 types and f16 accumulator (PTX ISA 8.7+, sm_90+)
    - Only valid with ordered metadata and m16n8k64 shape

    The shapes, layouts, and data types follow the same constraints as the
    regular `nvvm.mma.sync` operation, but the A operand contains only the
    non-zero elements in compressed format.

    Example:
    ```mlir
    %d = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                          sparseMetadata[%meta] selector[%sel]
                          {shape = {k = 32 : i32, m = 16 : i32, n = 8 : i32}}
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>

    // With ordered metadata:
    %d = nvvm.mma.sp.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1]
                          sparseMetadata[%meta] selector[%sel]
                          {orderedMetadata, shape = {k = 32 : i32, m = 16 : i32, n = 8 : i32}}
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    ```
  }];
  let arguments = (ins
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandA,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandB,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandC,
    NVVM_T32BitSignlessInteger:$sparseMetadata,
    NVVM_T32BitSignlessInteger:$sparsitySelector,
    NVVM_MMAShapeAttr:$shape,
    OptionalAttr<NVVM_MMAIntOverflowAttr>:$intOverflowBehavior,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandAPtxType,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandBPtxType,
    UnitAttr:$orderedMetadata,
    OptionalAttr<NVVM_MMAKindAttr>:$kind
  );
  let results = (outs
    NVVM_LLVMStructureType:$res
  );
}
def NVVM_MmaOp : NVVM_Op<"mma.sync", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.mma.sync` operation collectively performs the operation
    `D = matmul(A, B) + C` using all threads in a warp.

    All the threads in the warp must execute the same `mma.sync` operation.

    For each possible multiplicand PTX data type, there are one or more possible
    instruction shapes given as "mMnNkK". The below table describes the posssibilities
    as well as the types required for the operands. Note that the data type for
    C (the accumulator) and D (the result) can vary independently when there are
    multiple possibilities in the "C/D Type" column.

    When an optional attribute cannot be immediately inferred from the types of
    the operands and the result during parsing or validation, an error will be
    raised.

    `b1Op` is only relevant when the binary (b1) type is given to
    `multiplicandDataType`. It specifies how the multiply-and-acumulate is
    performed and is either `xor_popc` or `and_poc`. The default is `xor_popc`.

    `intOverflowBehavior` is only relevant when the `multiplicandType` attribute
    is one of `u8, s8, u4, s4`, this attribute describes how overflow is handled
    in the accumulator. When the attribute is `satfinite`, the accumulator values
    are clamped in the int32 range on overflow. This is the default behavior.
    Alternatively, accumulator behavior `wrapped` can also be specified, in
    which case overflow wraps from one end of the range to the other.

    `layoutA` and `layoutB` are required and should generally be set to
    `#nvvm.mma_layout<row>` and `#nvvm.mma_layout<col>` respectively, but other
    combinations are possible for certain layouts according to the table below.

    ```
    | A/B Type | Shape     | ALayout | BLayout | A Type   | B Type   | C/D Type          |
    |----------|-----------|---------|---------|----------|----------|-------------------|
    | f64      | .m8n8k4   | row     | col     | 1x f64   | 1x f64   | 2x f64            |
    | f16      | .m8n8k4   | row/col | row/col | 2x f16x2 | 2x f16x2 | 4x f16x2 or 8xf32 |
    |          | .m16n8k8  | row     | col     | 2x f16x2 | 1x f16x2 | 2x f16x2 or 4 f32 |
    |          | .m16n8k16 | row     | col     | 4x f16x2 | 2x f16x2 | 2x f16x2 or 4 f32 |
    | bf16     | .m16n8k8  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
    |          | .m16n8k16 | row     | col     | 4x i32   | 2x i32   | 4x f32            |
    | tf32     | .m16n8k4  | row     | col     | 2x i32   | 1x i32   | 4x f32            |
    |          | .m16n8k8  | row     | col     | 4x i32   | 2x i32   | 2x f16x2 or 4 f32 |
    | u8/s8    | .m8n8k16  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | .m16n8k16 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    |          | .m16n8k32 | row     | col     | 4x i32   | 2x i32   | 4x i32            |
    | u4/s4    | .m8n8k32  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | m16n8k32  | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    |          | m16n8k64  | row     | col     | 4x i32   | 2x i32   | 4x i32            |
    | b1       | m8n8k128  | row     | col     | 1x i32   | 1x i32   | 2x i32            |
    |          | m16n8k128 | row     | col     | 2x i32   | 1x i32   | 4x i32            |
    ```

    Example:
    ```mlir

    %128 = nvvm.mma.sync A[%120, %121, %122, %123]
                         B[%124, %125]
                         C[%126, %127]
                         {layoutA = #nvvm.mma_layout<row>,
                          layoutB = #nvvm.mma_layout<col>,
                          shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}}
        : (vector<2xf16>, vector<2xf16>, vector<2xf16>)
           -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
    ```
  }];
  let arguments = (ins
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandA,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandB,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$operandC,
    NVVM_MMAShapeAttr:$shape,
    OptionalAttr<NVVM_MMAB1OpAttr>:$b1Op,
    OptionalAttr<NVVM_MMAIntOverflowAttr>:$intOverflowBehavior,
    NVVM_MMALayoutAttr:$layoutA,
    NVVM_MMALayoutAttr:$layoutB,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandAPtxType,
    OptionalAttr<NVVM_MMATypesAttr>:$multiplicandBPtxType
  );
  let results = (outs
    NVVM_LLVMStructureType:$res
  );
}
def NVVM_MmaSmemDescOp : NVVM_Op<"mma_smem_desc", [MemoryEffects<[MemRead]>]> {
  let description = [{
    The `nvvm.mma_smem_desc` constructs a Shared Memory descriptor for use as the MMA Shared Memory operand.
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$pointer,
    NVVM_T32BitSignlessInteger:$ldm,
    NVVM_T32BitSignlessInteger:$stride,
    NVVM_T8BitSignlessInteger2:$baseOffset,
    NVVM_T8BitSignlessInteger2:$swizzle,
    OptionalAttr<I32Attr>:$mmaDescVersion
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_MovMatrixOp : NVVM_Op<"movmatrix"> {
  let description = [{
    Moves a row-major matrix across all threads in a warp, reading elements
    from source `$src`, and writing the transposed elements to destination
    `$dst`.

    The `shape` attribute indicates the dimensions of the matrix being
    transposed. Each matrix element holds 16-bit data as indicated by the
    `eltType` attribute.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-movmatrix-instruction)

    Example:
    ```mlir
    %dst = nvvm.movmatrix %src {shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>,
                                eltType = #nvvm.ld_st_matrix_elt_type<b16>} : i32
    ```
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$src,
    NVVM_LdStMatrixShapeAttr:$shape,
    OptionalAttr<NVVM_MMALayoutAttr>:$layout,
    NVVM_LdStMatrixEltTypeAttr:$eltType
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$dst
  );
}
def NVVM_MulOp : NVVM_InferredOp<"mul"> {
  let description = [{
    nvvm.mul Op multiplies two values with a given `type` and `mode`.
    The type of the operation represents the types of the `a` and `b` operands.
    If `.hi` or `.lo` is specified, then `res` is the same size as `a` and `b`,
    and either the upper or lower half of the result is written to the
    destination register. If `.wide` is specified, then `res` is twice as
    wide as `a` and `b` to receive the full result of the multiplication. The
    `.wide` suffix is supported only for 16- and 32-bit integer types.
    For more information, refer [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mul)
  }];
  let arguments = (ins
    NVVM_T16BitSignlessIntegerOr323:$a,
    NVVM_T16BitSignlessIntegerOr323:$b,
    NVVM_MulModeAttr:$mode,
    OptionalAttr<BoolAttr>:$isSigned
  );
  let results = (outs
    NVVM_T16BitSignlessIntegerOr323:$res
  );
}
def NVVM_MulPacked_BF16x2_BF16x2_F16x2Op : NVVM_Op<"mul.packed.bf16x2.bf16x2.f16x2"> {
  let description = [{
    This Op performs the multiplication operation on a pair of "bf16"x2 and "f16"x2 vectors.
    The inputs are of the form v2bf16 and v2f16, and the output is of the form v2bf16.
    The inputs are converted to a scalar of packed i32 where the 0th element of the
    vector forms the low-16 bits and the 1st element occupies the high-16 bits.

    Examples:
    ```mlir
    %0 = nvvm.mul.packed.bf16x2.bf16x2.f16x2 %srcA, %srcB : vector<2xbf16>, vector<2xf16> -> vector<2xbf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$srcA,
    NVVM_VectorOf16BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$res
  );
}
def NVVM_MulPacked_BF16x2_F32x2_F32x2Op : NVVM_Op<"mul.packed.bf16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs the multiplication operation on a pair of f32x2 vectors.
    The inputs are of the form v2f32 and the output is of the form v2bf16.
    The inputs are converted to a scalar of packed i64 where the 0th element of the
    vector forms the low-32 bits and the 1st element occupies the high-32 bits. The round to zero
    is the only rounding mode supported for this operation.

    Examples:
    ```mlir
    // Round to zero (only rounding mode supported)
    %0 = nvvm.mul.packed.bf16x2.f32x2.f32x2 %srcA, %srcB : vector<2xf32>, vector<2xf32> -> vector<2xbf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$res
  );
}
def NVVM_MulPacked_F16x2_F16x2_BF16x2Op : NVVM_Op<"mul.packed.f16x2.f16x2.bf16x2"> {
  let description = [{
    This Op performs the multiplication operation on a pair of "f16"x2 and "bf16"x2 vectors.
    The inputs are of the form v2f16 and v2bf16, and the output is of the form v2f16.
    The inputs are converted to a scalar of packed i32 where the 0th element of the
    vector forms the low-16 bits and the 1st element occupies the high-16 bits.

    Examples:
    ```mlir
    %0 = nvvm.mul.packed.f16x2.f16x2.bf16x2 %srcA, %srcB : vector<2xf16>, vector<2xbf16> -> vector<2xf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$srcA,
    NVVM_VectorOfBfloat16TypeValuesOf:$srcB
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$res
  );
}
def NVVM_MulPacked_F16x2_F32x2_F32x2Op : NVVM_Op<"mul.packed.f16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs the multiplication operation on a pair of f32x2 vectors.
    The inputs are of the form v2f32 and the output is of the form v2f16.
    The inputs are converted to a scalar of packed i64 where the 0th element of the
    vector forms the low-32 bits and the 1st element occupies the high-32 bits. The round to zero
    is the only rounding mode supported for this operation.

    Examples:
    ```mlir
    // Round to zero (only rounding mode supported)
    %0 = nvvm.mul.packed.f16x2.f32x2.f32x2 %srcA, %srcB : vector<2xf32>, vector<2xf32> -> vector<2xf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$res
  );
}
def NVVM_MulPackedF32x2Op : NVVM_InferredOp<"mul.packed.f32x2"> {
  let description = [{
    This Op multiplies floating-point vectors.
    The inputs and outputs are of the form <N x f32>. The `rnd` attribute
    specifies the rounding mode. The `ftz` attribute, when set, flushes the
    subnormal inputs and results to sign-preserving zero.
    This Op is supported from SM100 and onwards.
    [For more information, see PTX ISA]
    (TODO: Add public PTX link when available)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues2:$srcA,
    NVVM_VectorOf32BitFloatValues2:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<I1Attr>:$ftz
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues2:$res
  );
}
def NVVM_MultimemLdReduceOp : NVVM_Op<"multimem.ld.reduce"> {
  let description = [{
    The `nvvm.multimem.ld.reduce` operation performs a multimem load-reduce
    instruction. It reads from a multimem address and reduces partial
    results across participating threads.

    The `ordering` attribute specifies the memory ordering semantics (weak,
    relaxed, acquire).

    The `op` attribute specifies the reduction operation (and, or,
    xor, add, min, max).

    The `dataType` attribute specifies the PTX data type:
    - Integer: b32, s32, u32, b64, s64, u64
    - Floating-point: f16, bf16, f32, f64
    - FP8: e5m2, e4m3

    The optional `accPrec` attribute specifies the accumulator precision type
    for sub-single-precision types (f16/bf16/f8E5M2/f8E4M3FN): f16 or f32.

    Example:
    ```mlir
    %res = nvvm.multimem.ld.reduce %addr
        {ordering = #nvvm.mem_order<relaxed>,
         op = #nvvm.red_op<add>,
         dataType = f32}
        : !llvm.ptr<1> -> f32
    ```

    See PTX ISA:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace1:$addr,
    NVVM_MultimemLdReduceOrderingAttr:$ordering,
    NVVM_MultimemLdReduceOpAttr:$op,
    TypeAttr:$dataType,
    OptionalAttr<TypeAttr>:$accPrec
  );
  let results = (outs
    NVVM_AnyType:$res
  );
}
def NVVM_MultimemRedOp : NVVM_Op<"multimem.red"> {
  let description = [{
    The `nvvm.multimem.red` operation performs a multimem atomic reduction,
    applying a reduction operation between a source value and the value at a
    multimem address.

    The `ordering` attribute specifies the memory ordering semantics.
    The `scope` attribute specifies the scope of the operation.
    The `op` attribute specifies the reduction operation.
    The `dataType` attribute specifies the PTX data type (b32/s32/u32/b64/s64/
    u64 for integer types, f16/bf16/f32/f64 for floating-point types).

    Example:
    ```mlir
    nvvm.multimem.red %val, %addr
        {ordering = #nvvm.mem_order<relaxed>,
         scope = #nvvm.mem_scope<cta>,
         op = #nvvm.red_op<add>,
         dataType = f32}
        : f32, !llvm.ptr<1>
    ```

    See PTX ISA:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem
  }];
  let arguments = (ins
    NVVM_AnyType:$val,
    NVVM_AnyType:$addr,
    NVVM_MultimemRedOrderingAttr:$ordering,
    NVVM_MemScopeKindAttr:$scope,
    NVVM_MultimemRedOpAttr:$op,
    TypeAttr:$dataType
  );
  let results = (outs);
}
def NVVM_MultimemStOp : NVVM_Op<"multimem.st"> {
  let description = [{
    The `nvvm.multimem.st` operation performs a multimem store instruction,
    broadcasting a value to a multimem address.

    The `ordering` attribute specifies the memory ordering semantics.
    The `scope` attribute specifies the scope of the operation.
    The `dataType` attribute specifies the PTX data type (b32/s32/u32/b64/s64/
    u64 for integer types, f16/bf16/f32/f64 for floating-point, e5m2/e4m3 for
    FP8 types).

    Example:
    ```mlir
    nvvm.multimem.st %val, %addr
        {ordering = #nvvm.mem_order<relaxed>,
         scope = #nvvm.mem_scope<cta>,
         dataType = f32}
        : f32, !llvm.ptr<1>
    ```

    See PTX ISA:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem
  }];
  let arguments = (ins
    NVVM_AnyType:$val,
    NVVM_LLVMPointerInAddressSpace1:$addr,
    NVVM_MultimemStOrderingAttr:$ordering,
    NVVM_MemScopeKindAttr:$scope,
    TypeAttr:$dataType
  );
  let results = (outs);
}
def NVVM_NanosleepOp : NVVM_Op<"nanosleep"> {
  let description = [{
    The op suspends the thread for a sleep duration approximately close to the
    delay `$duration`, specified in nanoseconds.

    The sleep duration is approximated, but guaranteed to be in the
    interval [0, 2*t]. The maximum sleep duration is 1 millisecond.
    The implementation may reduce the sleep duration for individual threads
    within a warp such that all sleeping threads in the warp wake up together.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$duration
  );
  let results = (outs);
}
def NVVM_PMEventOp : NVVM_Op<"pmevent"> {
  let description = [{
    Triggers one or more of a fixed number of performance monitor events, with
    event index or mask specified by immediate operand.

    Without `mask` it triggers a single performance monitor event indexed by
    immediate operand a, in the range 0..15.

    With `mask` it triggers one or more of the performance monitor events. Each
    bit in the 16-bit immediate operand controls an event.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent)
  }];
  let arguments = (ins
    OptionalAttr<I16Attr>:$maskedEventId,
    OptionalAttr<I32Attr>:$eventId
  );
  let results = (outs);
}
def NVVM_PrefetchOp : NVVM_Op<"prefetch"> {
  let description = [{
    Prefetches the cache line containing the address given by `addr`. The
    operand may be a global, local, or generic pointer. When `tensormap` is
    specified, the operand may instead be a constant or generic pointer. If the
    address maps to shared memory, the operation has no effect.

    At most one of `cacheLevel` or `tensormap` may be present. The `cacheLevel`
    attribute selects the target cache level. When combined with `uniform`, the
    prefetch is performed to the uniform cache, in which case `addr` must be a
    generic pointer.

    When `tensormap` is used, the line containing `addr` is brought from the
    constant or parameter state space for later use by `cp.async.bulk.tensor`.
    If `in_param_space` is specified, the generic pointer is interpreted as
    referring to the parameter state space.

    `uniform` can be specified after the `cacheLevel` to indicate that the
    prefetch is performed to the specified uniform cache level. If `uniform` is
    specified, `addr` must be a generic address pointer and no operation is
    performed if `addr` maps to a `const`, `local`, or `shared` memory location.

    The `evictPriority` attribute is optional and specifies the cache eviction
    priority when `cacheLevel` is L2.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace13:$addr,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    OptionalAttr<NVVM_PrefetchCacheLevelAttr>:$cacheLevel,
    OptionalAttr<NVVM_CacheEvictionPriorityAttr>:$evictPriority,
    UnitAttr:$tensormap,
    UnitAttr:$uniform,
    UnitAttr:$in_param_space
  );
  let results = (outs);
}
def NVVM_PermuteOp : NVVM_Op<"prmt", [Pure]> {
  let description = [{
    The `nvvm.prmt` operation constructs a permutation of the
    bytes of the first one or two operands, selecting based on
    the 2 least significant bits of the final operand.

    The bytes in the first one or two source operands are numbered.
    The first source operand (%lo) is numbered {b3, b2, b1, b0},
    in the case of the '``default``', '``f4e``' and '``b4e``' variants,
    the second source operand (%hi) is numbered {b7, b6, b5, b4}.

    Modes:
    - `default`: Index mode         - each nibble in `selector` selects a byte from the 8-byte pool
    - `f4e`    : Forward 4 extract  - extracts 4 contiguous bytes starting from position in `selector`
    - `b4e`    : Backward 4 extract - extracts 4 contiguous bytes in reverse order
    - `rc8`    : Replicate 8        - replicates the lower 8 bits across the 32-bit result
    - `ecl`    : Edge clamp left    - clamps out-of-range indices to the leftmost valid byte
    - `ecr`    : Edge clamp right   - clamps out-of-range indices to the rightmost valid byte
    - `rc16`   : Replicate 16       - replicates the lower 16 bits across the 32-bit result

    Depending on the 2 least significant bits of the %selector operand, the result
    of the permutation is defined as follows:

    +------------+----------------+--------------+
    |    Mode    | %selector[1:0] |    Output    |
    +------------+----------------+--------------+
    | '``f4e``'  | 0              | {3, 2, 1, 0} |
    |            +----------------+--------------+
    |            | 1              | {4, 3, 2, 1} |
    |            +----------------+--------------+
    |            | 2              | {5, 4, 3, 2} |
    |            +----------------+--------------+
    |            | 3              | {6, 5, 4, 3} |
    +------------+----------------+--------------+
    | '``b4e``'  | 0              | {5, 6, 7, 0} |
    |            +----------------+--------------+
    |            | 1              | {6, 7, 0, 1} |
    |            +----------------+--------------+
    |            | 2              | {7, 0, 1, 2} |
    |            +----------------+--------------+
    |            | 3              | {0, 1, 2, 3} |
    +------------+----------------+--------------+
    | '``rc8``'  | 0              | {0, 0, 0, 0} |
    |            +----------------+--------------+
    |            | 1              | {1, 1, 1, 1} |
    |            +----------------+--------------+
    |            | 2              | {2, 2, 2, 2} |
    |            +----------------+--------------+
    |            | 3              | {3, 3, 3, 3} |
    +------------+----------------+--------------+
    | '``ecl``'  | 0              | {3, 2, 1, 0} |
    |            +----------------+--------------+
    |            | 1              | {3, 2, 1, 1} |
    |            +----------------+--------------+
    |            | 2              | {3, 2, 2, 2} |
    |            +----------------+--------------+
    |            | 3              | {3, 3, 3, 3} |
    +------------+----------------+--------------+
    | '``ecr``'  | 0              | {0, 0, 0, 0} |
    |            +----------------+--------------+
    |            | 1              | {1, 1, 1, 0} |
    |            +----------------+--------------+
    |            | 2              | {2, 2, 1, 0} |
    |            +----------------+--------------+
    |            | 3              | {3, 2, 1, 0} |
    +------------+----------------+--------------+
    | '``rc16``' | 0              | {1, 0, 1, 0} |
    |            +----------------+--------------+
    |            | 1              | {3, 2, 3, 2} |
    |            +----------------+--------------+
    |            | 2              | {1, 0, 1, 0} |
    |            +----------------+--------------+
    |            | 3              | {3, 2, 3, 2} |
    +------------+----------------+--------------+

    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$lo,
    Optional<NVVM_T32BitSignlessInteger>:$hi,
    NVVM_T32BitSignlessInteger:$selector,
    NVVM_PermuteModeAttr:$mode
  );
  let results = (outs
    NVVM_T32BitSignlessInteger2:$res
  );
}
def NVVM_RcpApproxFtzF32Op : NVVM_Op<"rcp.approx.ftz.f", [Pure]> {
  let arguments = (ins
    NVVM_T32BitFloat:$arg
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_AggrSmemSize : NVVM_Op<"read.ptx.sreg.aggr.smem.size", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClockOp : NVVM_Op<"read.ptx.sreg.clock"> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_Clock64Op : NVVM_Op<"read.ptx.sreg.clock64"> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockInClusterIdXOp : NVVM_Op<"read.ptx.sreg.cluster.ctaid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockInClusterIdYOp : NVVM_Op<"read.ptx.sreg.cluster.ctaid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockInClusterIdZOp : NVVM_Op<"read.ptx.sreg.cluster.ctaid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterId : NVVM_Op<"read.ptx.sreg.cluster.ctarank", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDimBlocksXOp : NVVM_Op<"read.ptx.sreg.cluster.nctaid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDimBlocksYOp : NVVM_Op<"read.ptx.sreg.cluster.nctaid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDimBlocksZOp : NVVM_Op<"read.ptx.sreg.cluster.nctaid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDim : NVVM_Op<"read.ptx.sreg.cluster.nctarank", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterIdXOp : NVVM_Op<"read.ptx.sreg.clusterid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterIdYOp : NVVM_Op<"read.ptx.sreg.clusterid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterIdZOp : NVVM_Op<"read.ptx.sreg.clusterid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockIdXOp : NVVM_Op<"read.ptx.sreg.ctaid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockIdYOp : NVVM_Op<"read.ptx.sreg.ctaid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockIdZOp : NVVM_Op<"read.ptx.sreg.ctaid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_DynamicSmemSize : NVVM_Op<"read.ptx.sreg.dynamic.smem.size", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg0Op : NVVM_Op<"read.ptx.sreg.envreg0", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg1Op : NVVM_Op<"read.ptx.sreg.envreg1", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg10Op : NVVM_Op<"read.ptx.sreg.envreg10", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg11Op : NVVM_Op<"read.ptx.sreg.envreg11", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg12Op : NVVM_Op<"read.ptx.sreg.envreg12", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg13Op : NVVM_Op<"read.ptx.sreg.envreg13", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg14Op : NVVM_Op<"read.ptx.sreg.envreg14", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg15Op : NVVM_Op<"read.ptx.sreg.envreg15", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg16Op : NVVM_Op<"read.ptx.sreg.envreg16", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg17Op : NVVM_Op<"read.ptx.sreg.envreg17", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg18Op : NVVM_Op<"read.ptx.sreg.envreg18", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg19Op : NVVM_Op<"read.ptx.sreg.envreg19", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg2Op : NVVM_Op<"read.ptx.sreg.envreg2", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg20Op : NVVM_Op<"read.ptx.sreg.envreg20", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg21Op : NVVM_Op<"read.ptx.sreg.envreg21", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg22Op : NVVM_Op<"read.ptx.sreg.envreg22", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg23Op : NVVM_Op<"read.ptx.sreg.envreg23", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg24Op : NVVM_Op<"read.ptx.sreg.envreg24", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg25Op : NVVM_Op<"read.ptx.sreg.envreg25", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg26Op : NVVM_Op<"read.ptx.sreg.envreg26", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg27Op : NVVM_Op<"read.ptx.sreg.envreg27", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg28Op : NVVM_Op<"read.ptx.sreg.envreg28", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg29Op : NVVM_Op<"read.ptx.sreg.envreg29", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg3Op : NVVM_Op<"read.ptx.sreg.envreg3", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg30Op : NVVM_Op<"read.ptx.sreg.envreg30", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg31Op : NVVM_Op<"read.ptx.sreg.envreg31", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg4Op : NVVM_Op<"read.ptx.sreg.envreg4", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg5Op : NVVM_Op<"read.ptx.sreg.envreg5", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg6Op : NVVM_Op<"read.ptx.sreg.envreg6", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg7Op : NVVM_Op<"read.ptx.sreg.envreg7", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg8Op : NVVM_Op<"read.ptx.sreg.envreg8", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_EnvReg9Op : NVVM_Op<"read.ptx.sreg.envreg9", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_GlobalTimerOp : NVVM_Op<"read.ptx.sreg.globaltimer"> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_GlobalTimerLoOp : NVVM_Op<"read.ptx.sreg.globaltimer.lo"> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_GridIdOp : NVVM_Op<"read.ptx.sreg.gridid", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_LaneIdOp : NVVM_Op<"read.ptx.sreg.laneid", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_LaneMaskEqOp : NVVM_Op<"read.ptx.sreg.lanemask.eq", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_LaneMaskGeOp : NVVM_Op<"read.ptx.sreg.lanemask.ge", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_LaneMaskGtOp : NVVM_Op<"read.ptx.sreg.lanemask.gt", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_LaneMaskLeOp : NVVM_Op<"read.ptx.sreg.lanemask.le", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_LaneMaskLtOp : NVVM_Op<"read.ptx.sreg.lanemask.lt", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDimXOp : NVVM_Op<"read.ptx.sreg.nclusterid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDimYOp : NVVM_Op<"read.ptx.sreg.nclusterid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ClusterDimZOp : NVVM_Op<"read.ptx.sreg.nclusterid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_GridDimXOp : NVVM_Op<"read.ptx.sreg.nctaid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_GridDimYOp : NVVM_Op<"read.ptx.sreg.nctaid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_GridDimZOp : NVVM_Op<"read.ptx.sreg.nctaid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_SmDimOp : NVVM_Op<"read.ptx.sreg.nsmid", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockDimXOp : NVVM_Op<"read.ptx.sreg.ntid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockDimYOp : NVVM_Op<"read.ptx.sreg.ntid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_BlockDimZOp : NVVM_Op<"read.ptx.sreg.ntid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_WarpDimOp : NVVM_Op<"read.ptx.sreg.nwarpid", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_SmIdOp : NVVM_Op<"read.ptx.sreg.smid", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ThreadIdXOp : NVVM_Op<"read.ptx.sreg.tid.x", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ThreadIdYOp : NVVM_Op<"read.ptx.sreg.tid.y", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_ThreadIdZOp : NVVM_Op<"read.ptx.sreg.tid.z", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_TotalSmemSize : NVVM_Op<"read.ptx.sreg.total.smem.size", [Pure]> {
  let arguments = (ins);
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_WarpIdOp : NVVM_Op<"read.ptx.sreg.warpid", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_WarpSizeOp : NVVM_Op<"read.ptx.sreg.warpsize", [Pure]> {
  let arguments = (ins
    OptionalAttr<NVVM_LLVMConstantRangeAttr>:$range
  );
  let results = (outs
    NVVM_LLVMDialectCompatibleType:$res
  );
}
def NVVM_RedOp : NVVM_Op<"red"> {
  let description = [{
    Performs a reduction operation with operand `b` and the value in
    location `a`, and stores the result of the specified operation at
    location `a`. Operand `a` specifies a location in the specified state space.
    If no state space is given, perform the memory accesses
    using Generic Addressing

    [For more information, see PTX ISA]
    (http://sw-mobile-docs/CUDA/GPGPU/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace05:$a,
    NVVM_T32BitSignlessIntegerOr642:$b,
    Optional<NVVM_T64BitSignlessInteger>:$cacheHint,
    OptionalAttr<NVVM_MemOrderKindAttr>:$memOrder,
    OptionalAttr<NVVM_MemScopeKindAttr>:$memScope,
    NVVM_ReductionOpAttr:$op,
    NVVM_ReductionTypeAttr:$type
  );
  let results = (outs);
}
def NVVM_ReduxOp : NVVM_InferredOp<"redux.sync"> {
  let description = [{
    `redux.sync` performs a reduction operation `kind` of the 32 bit source
    register across all non-exited threads in the membermask.

    The `abs` and `nan` attributes can be used in the case of f32 input type,
    where the `abs` attribute causes the absolute value of the input to be used
    in the reduction operation, and the `nan` attribute causes the reduction
    operation to return NaN if any of the inputs to participating threads are
    NaN.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessIntegerOr32:$val,
    NVVM_T32BitSignlessInteger:$mask_and_clamp,
    NVVM_ReductionKindAttr:$kind,
    OptionalAttr<BoolAttr>:$abs,
    OptionalAttr<BoolAttr>:$nan
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOr32:$res
  );
}
def NVVM_RsqrtOp : NVVM_InferredOp<"rsqrt", [Pure]> {
  let description = [{
    Computes an approximation of the reciprocal of the square root of the
    input value: `d = 1 / sqrt(a)`. Supports both f32 and f64. The maximum
    relative error for the f32 form over the entire positive finite range
    is 2^-22.9.

    The `ftz` attribute, when set, flushes subnormal inputs and results to
    sign-preserving zero. For f64 inputs, `ftz=true` selects a coarser
    approximation that uses only the upper 32 bits of the input (the lower
    32 bits of the result are zeroed).

    For more information, see PTX ISA:
    [rsqrt](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt)
  }];
  let arguments = (ins
    NVVM_T32BitFloatOr64Bit:$src,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T32BitFloatOr64Bit:$res
  );
}
def NVVM_SetMaxRegisterOp : NVVM_Op<"setmaxregister"> {
  let arguments = (ins
    I32Attr:$regCount,
    NVVM_SetMaxRegisterActionAttr:$action
  );
  let results = (outs);
}
def NVVM_ShflOp : NVVM_InferredOp<"shfl.sync"> {
  let description = [{
    The `shfl.sync` Op implements data shuffle within threads of a warp.
    The `thread_mask` denotes the threads participating in the Op where
    the bit position corresponds to a particular thread's laneid.
    The `offset` specifies a source lane or source lane offset
    (depending on `kind`). The `val` is the input value to be copied from
    the source. The `mask_and_clamp` contains two packed values specifying
    a mask for logically splitting warps into sub-segments and an upper bound
    for clamping the source lane index.

    The `return_value_and_is_valid` unit attribute can be specified to indicate
    that the return value is a two-element struct, where the first element is
    the result value and the second element is a predicate indicating if the
    computed source lane index is valid.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$thread_mask,
    NVVM_T32BitSignlessIntegerOr32:$val,
    NVVM_T32BitSignlessInteger:$offset,
    NVVM_T32BitSignlessInteger:$mask_and_clamp,
    NVVM_ShflKindAttr:$kind,
    UnitAttr:$return_value_and_is_valid
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOr322:$res
  );
}
def NVVM_SinOp : NVVM_Op<"sin", [Pure]> {
  let description = [{
    Computes a fast approximation of the sine of the input value (in radians).
    The `ftz` attribute, when set, flushes subnormal inputs and results to
    sign-preserving zero.

    For more information, see PTX ISA:
    [sin](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin)
  }];
  let arguments = (ins
    NVVM_T32BitFloat:$src,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T32BitFloat2:$res
  );
}
def NVVM_SPCompressOp : NVVM_Op<"spcompress"> {
  let description = [{
    Compresses elements from a dense vector `data` into a structured sparse vector `compressed_data`.
    The metadata information is stored in `metadata`.

    Arguments:
    - `spDesc`: Sparsity descriptor specifying the selection criteria for elements
    - `data`: Input dense vector containing elements to be compressed
    - `compFactor`: Compression factor attribute (only `sp2to4` is allowed, resulting in 2:4 compression)
    - `indexSize`: Size of metadata index for each selected element (can only be `b2` or `b4` given 2:4 compression)
    - `elemSize`: Size in bits for each element from input vector `data`
    - `repFactor`: Repeat factor for the operation

    Results:
    - `metadata`: Vector storing indices for chosen elements
    - `compressed_data`: Output compressed vector containing selected elements

    Examples:
    // Example 1: Basic compression with 4-bit elements, 2-bit indices, repeat factor of 1
    ```mlir
    %metadata, %compressed_data = nvvm.spcompress %spdesc, %data {
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      indexSize = #nvvm.spcomp_index_size<b2>,
      elemSize = #nvvm.spcomp_elem_size<b4>,
      repFactor = #nvvm.spcomp_rep_factor<x1>
    } : i32, vector<2xi32> -> i32, i32
    ```

    // Example 2: Compression with 8-bit elements, 4-bit indices, repeat factor of 2
    ```mlir
    %metadata, %compressed_data = nvvm.spcompress %spdesc, %data {
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      indexSize = #nvvm.spcomp_index_size<b4>,
      elemSize = #nvvm.spcomp_elem_size<b8>,
      repFactor = #nvvm.spcomp_rep_factor<x2>
    } : i32, vector<4xi32> -> i32, vector<2xi32>
    ```

    // Example 3: Compression with 16-bit elements, 4-bit indices, larger repeat factor
    ```mlir
    %metadata, %compressed_data = nvvm.spcompress %spdesc, %data {
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      indexSize = #nvvm.spcomp_index_size<b4>,
      elemSize = #nvvm.spcomp_elem_size<b16>,
      repFactor = #nvvm.spcomp_rep_factor<x8>
    } : i32, vector<16xi32> -> vector<2xi32>, vector<8xi32>
    ```
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$spDesc,
    NVVM_VectorOf32BitSignlessInteger:$data,
    NVVM_SPCompressFactorTypeAttr:$compFactor,
    NVVM_SPCompressIndexSizeAttr:$indexSize,
    NVVM_SPCompressElemSizeAttr:$elemSize,
    NVVM_SPCompressRepFactorAttr:$repFactor
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOrVector:$metadata,
    NVVM_T32BitSignlessIntegerOrVector:$compressed_data
  );
}
def NVVM_SPDecompressOp : NVVM_Op<"spdecompress"> {
  let description = [{
    Decompresses elements from a structured sparse vector `compressed_data` into a dense vector `data`.
    The metadata information is stored in `metadata`.

    Arguments:
    - `metadata`: Vector storing indices for chosen elements
    - `compressed_data`: Compressed vector containing selected elements
    - `compFactor`: Decompression factor attribute (sp1to2, sp1to4, sp1to8, sp1to16, sp2to4, sp2to8, sp2to16, sp4to8, sp4to16, sp8to16)
    - `indexSize`: Size of metadata index for each selected element (b2 or b4)
    - `elemSize`: Size in bits for each element from output vector `data` (b4, b8, b16)
    - `repFactor`: Repeat factor for the operation (x1, x2, x4, x8, x16, x32, x64)

    Results:
    - `data`: Output dense vector containing decompressed elements

    Examples:
    // Example 1: Decompression with 4-bit elements, 2-bit indices, repeat factor of 2
    ```mlir
    %res = nvvm.spdecompress %metadata, %compdata {
      compFactor = #nvvm.spdecomp_factor<sp2to4>,
      indexSize = #nvvm.spdecomp_index_size<b2>,
      elemSize = #nvvm.spdecomp_elem_size<b4>,
      repFactor = #nvvm.spdecomp_rep_factor<x2>
    } : i32, i32 -> i32
    ```

    // Example 2: Decompression with 8-bit elements, 4-bit indices, repeat factor of 2
    ```mlir
    %res = nvvm.spdecompress %metadata, %compdata {
      compFactor = #nvvm.spdecomp_factor<sp2to4>,
      indexSize = #nvvm.spdecomp_index_size<b4>,
      elemSize = #nvvm.spdecomp_elem_size<b8>,
      repFactor = #nvvm.spdecomp_rep_factor<x2>
    } : i32, i32 -> vector<2 x i32>
    ```

    // Example 3: Decompression with 16-bit elements, 4-bit indices, repeat factor of 64
    ```mlir
    %res = nvvm.spdecompress %metadata, %compdata {
      compFactor = #nvvm.spdecomp_factor<sp2to4>,
      indexSize = #nvvm.spdecomp_index_size<b4>,
      elemSize = #nvvm.spdecomp_elem_size<b16>,
      repFactor = #nvvm.spdecomp_rep_factor<x64>
    } : vector<16 x i32>, vector<64 x i32> -> vector<128 x i32>
    ```
  }];
  let arguments = (ins
    NVVM_T32BitSignlessIntegerOrVector:$metadata,
    NVVM_T32BitSignlessIntegerOrVector:$compressed_data,
    NVVM_SPDecompressFactorTypeAttr:$compFactor,
    NVVM_SPDecompressIndexSizeAttr:$indexSize,
    NVVM_SPDecompressElemSizeAttr:$elemSize,
    NVVM_SPDecompressRepFactorAttr:$repFactor
  );
  let results = (outs
    NVVM_VectorOf32BitSignlessInteger2:$data
  );
}
def NVVM_BulkStoreOp : NVVM_Op<"st.bulk"> {
  let description = [{
    Initializes a region of shared memory at the address given by `addr`.
    The `size` operand specifies the number of bytes to initialize and must be
    a multiple of 8.
    The `initVal` operand specifies the value to initialize the memory to. The
    only supported value is 0.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace02:$addr,
    NVVM_T64BitSignlessInteger:$size,
    OptionalAttr<I64Attr>:$initVal
  );
  let results = (outs);
}
def NVVM_StMatrixOp : NVVM_Op<"stmatrix"> {
  let description = [{
    Collectively store one or more matrices across all threads in a warp to the
    location indicated by the address operand $ptr in shared memory.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace3:$ptr,
    Variadic<NVVM_VariadicOf32BitSignlessInteger>:$sources,
    NVVM_MMALayoutAttr:$layout,
    OptionalAttr<NVVM_StoreShapeAttr>:$shape
  );
  let results = (outs);
}
def NVVM_AsyncStoreOp : NVVM_Op<"store.async"> {
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace7:$addr,
    NVVM_T32BitSignlessIntegerOr643:$value,
    NVVM_LLVMPointerInAddressSpace7:$mbarrier
  );
  let results = (outs);
}
def NVVM_ExtStoreOp : NVVM_Op<"store.ext"> {
  let description = [{
    The `store.ext` operation is used to write to memory in generic,
    global or shared address space. This Op explicitly supports various
    cache properties and memory orderings associated with the store operation.

    The `l2CacheHint` operand is optional, and it is used to specify cache
    eviction policy that may be used during the memory access.
    [For more information, see PTX ISA]
    (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st)
  }];
  let arguments = (ins
    NVVM_LLVMTypeWithSize:$value,
    NVVM_LLVMPointerType:$addr,
    Optional<NVVM_T64BitSignlessInteger>:$l2CacheHint,
    OptionalAttr<NVVM_MemOrderKindAttr>:$order,
    OptionalAttr<NVVM_MemScopeKindAttr>:$scope,
    OptionalAttr<NVVM_EvictKindAttr>:$evict,
    OptionalAttr<NVVM_StoreCacheModifierAttr>:$cacheModifier,
    OptionalAttr<NVVM_SharedSpaceAttr>:$sharedSpace
  );
  let results = (outs);
}
def NVVM_SubPacked_BF16x2_F32x2_F32x2Op : NVVM_Op<"sub.packed.bf16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs the sub operation on a pair of f32x2 vectors.
    The inputs and output are of the form v2f32 and v2bf16 respectively.
    The inputs are converted to a scalar of packed i64 where the 0th element of the
    vector forms the low-32 bits and the 1st element occupies the high-32 bits. The round to zero
    is the only rounding mode supported for this operation.

    Examples:
    ```mlir
    // Round to zero (only rounding mode supported)
    %0 = nvvm.sub .packed.bf16x2.f32x2.f32x2 %srcA, %srcB : vector<2xf32>, vector<2xf32> -> vector<2xbf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOfBfloat16TypeValuesOf:$res
  );
}
def NVVM_SubPacked_F16x2_F32x2_F32x2Op : NVVM_Op<"sub.packed.f16x2.f32x2.f32x2"> {
  let description = [{
    This Op performs the sub operation on a pair of f32x2 vectors.
    The inputs and output are of the form v2f32 and v2f16 respectively.
    The inputs are converted to a scalar of packed i64 where the 0th element of the
    vector forms the low-32 bits and the 1st element occupies the high-32 bits. The round to zero
    is the only rounding mode supported for this operation.

    Examples:
    ```mlir
    // Round to zero (only rounding mode supported)
    %0 = nvvm.sub .packed.f16x2.f32x2.f32x2 %srcA, %srcB : vector<2xf32>, vector<2xf32> -> vector<2xf16>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB
  );
  let results = (outs
    NVVM_VectorOf16BitFloatValues:$res
  );
}
def NVVM_SubPackedF32x2Op : NVVM_InferredOp<"sub.packed.f32x2"> {
  let description = [{
    This Op performs subtraction of floating-point vectors.
    The inputs and outputs are of the form <N x f32>. The `rnd` attribute
    specifies the rounding mode. The `ftz` attribute, when set, flushes the
    subnormal inputs and results to sign-preserving zero.
    This Op is supported from SM100 and onwards.
    [For more information, see PTX ISA]
    (TODO: Add public PTX link when available)
  }];
  let arguments = (ins
    NVVM_VectorOf32BitFloatValues2:$srcA,
    NVVM_VectorOf32BitFloatValues2:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<I1Attr>:$ftz
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues2:$res
  );
}
def NVVM_SubPacked_F32x2_BF16x2_F32x2Op : NVVM_Op<"sub.packed.f32x2.bf16x2.f32x2"> {
  let description = [{
    This Op performs the sub operation on a pair of floating-point values.
    The inputs are of the form v2bf16 and v2f32. The first input `srcA` is
    converted to a scalar of packed i32 where the 0th element of the vector forms
    the low-16 bits and the 1st element occupies the high-16 bits. Similarly,
    the second input `srcB` is converted to a scalar of packed i64 where the 0th
    element of the vector forms the low-32 bits and the 1st element occupies the
    high-32 bits. The `rnd` attribute specifies the rounding mode.

    The rounding mode can be one of the following:
    - rn: round to nearest even
    - rz: round toward zero
    - rm: round toward negative infinity
    - rp: round toward positive infinity

    Examples:
    ```mlir
    // Round to nearest even (default)
    %0 = nvvm.sub .packed.f32x2.bf16x2.f32x2 %srcA, %srcB : vector<2xbf16>, vector<2xf32> -> vector<2xf32>

    // Round toward negative infinity
    %1 = nvvm.sub .packed.f32x2.bf16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>

    // Round toward zero with flush-to-zero
    %2 = nvvm.sub .packed.f32x2.bf16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xbf16>, vector<2xf32> -> vector<2xf32>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOfBfloat16TypeValuesOf:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues:$res
  );
}
def NVVM_SubPacked_F32x2_F16x2_F32x2Op : NVVM_Op<"sub.packed.f32x2.f16x2.f32x2"> {
  let description = [{
    This Op performs the sub operation on a pair of floating-point values.
    The inputs are of the form v2f16 and v2f32. The first input `srcA` is
    converted to a scalar of packed i32 where the 0th element of the vector forms
    the low-16 bits and the 1st element occupies the high-16 bits. Similarly,
    the second input `srcB` is converted to a scalar of packed i64 where the 0th
    element of the vector forms the low-32 bits and the 1st element occupies the
    high-32 bits. The `rnd` attribute specifies the rounding mode.

    The rounding mode can be one of the following:
    - rn: round to nearest even
    - rz: round toward zero
    - rm: round toward negative infinity
    - rp: round toward positive infinity

    Examples:
    ```mlir
    // Round to nearest even (default)
    %0 = nvvm.sub .packed.f32x2.f16x2.f32x2 %srcA, %srcB : vector<2xf16>, vector<2xf32> -> vector<2xf32>

    // Round toward negative infinity
    %1 = nvvm.sub .packed.f32x2.f16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rm>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>

    // Round toward zero with flush-to-zero
    %2 = nvvm.sub .packed.f32x2.f16x2.f32x2 %srcA, %srcB {rnd = #nvvm.fp_rnd_mode<rz>} : vector<2xf16>, vector<2xf32> -> vector<2xf32>
    ```
  }];
  let arguments = (ins
    NVVM_VectorOf16BitFloatValues:$srcA,
    NVVM_VectorOf32BitFloatValues:$srcB,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd
  );
  let results = (outs
    NVVM_VectorOf32BitFloatValues:$res
  );
}
def NVVM_SubFOp : NVVM_InferredOp<"subf", [Pure]> {
  let description = [{
    The `nvvm.subf` operation performs floating point subtraction of two
    operands.

    It supports the same type combinations and modifiers as `nvvm.addf`.
    This is equivalent to `nvvm.addf(lhs, -rhs)`.

    For more information, see PTX ISA:
    - [floating point subtraction](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sub)
    - [half-precision floating point subtraction](https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub)
  }];
  let arguments = (ins
    NVVM_T16BitFloatOrBfloat16Type:$lhs,
    NVVM_T16BitFloatOrBfloat16Type:$rhs,
    OptionalAttr<NVVM_FPRoundingModeAttr>:$rnd,
    OptionalAttr<NVVM_SaturationModeAttr>:$sat,
    OptionalAttr<BoolAttr>:$ftz
  );
  let results = (outs
    NVVM_T16BitFloatOrBfloat16Type:$res
  );
}
def NVVM_Tcgen05AllocOp : NVVM_Op<"tcgen05.alloc"> {
  let description = [{
    The `tcgen05.alloc` Op allocates tensor core memory for
    the amount specified by `nCols` and writes the destination
    address to the `addr` argument. The `nCols` operand specifies the
    number of columns to be allocated and it must be a power-of-two.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    NVVM_T32BitSignlessInteger:$nCols,
    UnitAttr:$isExclusive,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group
  );
  let results = (outs);
}
def NVVM_Tcgen05CommitOp : NVVM_Op<"tcgen05.commit"> {
  let description = [{
    The `tcgen05.commit` makes the *mbarrier object*, specified by
    the operand `addr`, track the completion of all the prior
    async-tcgen05 operations initiated by the executing thread.
    The multicast variants allow signaling on the *mbarrier objects*
    of multiple CTAs within the cluster. Operand `multicastMask`,
    when present, specifies the destination CTAs in the cluster such
    that each bit position in the 16-bit or 32-bit `multicastMask` operand
    corresponds to the `nvvm.read.ptx.sreg.ctaid` of the destination CTA.
    Operand `smem_a_read` when present, specifies track only shared memory
    read of "Matrix A for the prior tcgen05.mma Ops"

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit)
  }];
  let arguments = (ins
    NVVM_LLVMPointerTypeOrLLVMPointer:$addr,
    Optional<NVVM_T16BitSignlessIntegerOr322>:$multicastMask,
    UnitAttr:$smem_a_read,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group
  );
  let results = (outs);
}
def NVVM_Tcgen05CpOp : NVVM_Op<"tcgen05.cp"> {
  let description = [{
    Instruction tcgen05.cp initiates an asynchronous copy operation from
    shared memory to the location specified by the address operand `taddr`
    in the Tensor Memory. The 64-bit register operand `smem_desc` specifies
    the matrix descriptor representing the source matrix in the shared memory
    that needs to be copied.

    Example:
    ```mlir
      nvvm.tcgen05.cp %taddr, %smem_desc {
        group = #nvvm.cta_group<cta_2>,
        shape = #nvvm.tcgen05_cp_shape<shape_64x128b>,
        multicast = #nvvm.tcgen05_cp_multicast<warpx2_01_23>,
        srcFormat = #nvvm.tcgen05_cp_src_fmt<b6x16_p32>
      }
    ```
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-cp)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$taddr,
    NVVM_T64BitSignlessInteger:$smem_desc,
    NVVM_Tcgen05CpShapeAttr:$shape,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group,
    OptionalAttr<NVVM_Tcgen05CpMulticastAttr>:$multicast,
    OptionalAttr<NVVM_Tcgen05CpSrcFormatAttr>:$srcFormat
  );
  let results = (outs);
}
def NVVM_Tcgen05DeallocOp : NVVM_Op<"tcgen05.dealloc"> {
  let description = [{
    The `tcgen05.dealloc` Op de-allocates the tensor core memory
    specified by `tmemAddr`, which must be from a previous tensor
    memory allocation. The `nCols` operand specifies the number
    of columns to be de-allocated, and it must be a power-of-two.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$taddr,
    NVVM_T32BitSignlessInteger:$nCols,
    UnitAttr:$isExclusive,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group
  );
  let results = (outs);
}
def NVVM_Tcgen05FenceOp : NVVM_Op<"tcgen05.fence"> {
  let description = [{
    The `tcgen05.fence<before>` orders all prior async tcgen05 operations
    with respect to the subsequent tcgen05 and execution ordering operations.
    The `tcgen05.fence<after>` orders all subsequent async tcgen05 operations
    with respect to the prior tcgen05 and execution ordering operations.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-fence)
  }];
  let arguments = (ins
    NVVM_Tcgen05FenceKindAttr:$kind
  );
  let results = (outs);
}
def NVVM_Tcgen05LdOp : NVVM_Op<"tcgen05.ld"> {
  let description = [{
    Instruction `tcgen05.ld` asynchronously loads data from the Tensor Memory at
    the location specified by the 32-bit address operand `tmemAddr` into the
    destination register `res`, collectively across all threads of the warps.

    The `shape` and the `num` attribute together determines the total
    dimension of the data which is loaded from the Tensor Memory. The `shape`
    attribute indicates the base dimension of data to be accessed as described
    in the Data Movement Shape. The `num` attribute indicates the repeat
    factor on the base dimension resulting in the total dimension of the data
    that is accessed.

    The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
    `16x32b`. The base address of the first access is specified by `tmemAddr`
    and the base address of the second access is specified by
    `tmemAddr + offset`, where `offset` is an immediate argument.

    The unit attribute `pack` can be used to pack two 16-bit
    elements from adjacent columns into a single 32-bit element during the load.

    The following table describes the size of the vector for various combinations
    of `num` and `shape` attributes:
    ```
    |=====================================================================|
    | num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  |
    |=====================================================================|
    | x1             |          1                 |    2       |    4     |
    | x2             |          2                 |    4       |    8     |
    | x4             |          4                 |    8       |    16    |
    | x8             |          8                 |    16      |    32    |
    | x16            |          16                |    32      |    64    |
    | x32            |          32                |    64      |    128   |
    | x64            |          64                |    128     |    NA    |
    | x128           |          128               |    NA      |    NA    |
    |=====================================================================|
    ```

    Example:
    ```mlir
      nvvm.tcgen05.ld %tmemAddr, %offset pack {
        shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
      } : <2xi32>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$tmemAddr,
    Optional<NVVM_T64BitSignlessInteger>:$offset,
    UnitAttr:$pack,
    NVVM_Tcgen05LdStShapeAttr:$shape
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOrVector2:$res
  );
}
def NVVM_Tcgen05LdRedOp : NVVM_Op<"tcgen05.ld.red"> {
  let description = [{
    Instruction `tcgen05.ld.red` asynchronously loads data from the Tensor
    Memory at the location specified by the 32-bit address operand `addr` into
    the destination register `data`, collectively across all threads of the
    warp. The operation also performs reduction operation specified by `op` on
    the loaded data across columns in each lane and stored into `redVal`

    The `shape` and the `num` attribute together determines the total
    dimension of the data which is loaded from the Tensor Memory. The `shape`
    attribute indicates the base dimension of data to be accessed as described
    in the Data Movement Shape. The `num` attribute indicates the repeat
    factor on the base dimension resulting in the total dimension of the data
    that is accessed.

    The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
    `16x32b`. The base address of the first access is specified by `addr`
    and the base address of the second access is specified by
    `addr + offset`, where `offset` is an immediate argument.

      `isSigned` attribute is used to specify whether the integer reduction value is is signed, it is to false by default

    The following table describes the size of the vector for various combinations
    of `num` and `shape` attributes:
    ```
    |=============================================|
    | num/shape      |     16x32bx2/32x32b        |
    |=============================================|
    | x2             |             2              |
    | x4             |             4              |
    | x8             |             8              |
    | x16            |             16             |
    | x32            |             32             |
    | x64            |             64             |
    | x128           |             128            |
    |=============================================|
    ```

    Example:
    ```mlir
      // u32
      %data, %redval = nvvm.tcgen05.ld,red %addr, %offset {
        shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
      } : <2xi32>, i32

      // s32
      %data, %redval = nvvm.tcgen05.ld,red %addr, %offset {
        shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
        isSigned = true
      } : <2xi32>, i32

      // f32
      %data, %redval = nvvm.tcgen05.ld,red %addr {
        shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      } : <2xf32>, f32
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$addr,
    Optional<NVVM_T64BitSignlessInteger>:$offset,
    NVVM_Tcgen05LdStShapeAttr:$shape,
    NVVM_ReductionKindAttr:$op,
    UnitAttr:$abs,
    UnitAttr:$nan,
    OptionalAttr<BoolAttr>:$isSigned
  );
  let results = (outs
    NVVM_VectorOf32BitSignlessInteger3:$data,
    NVVM_T32BitSignlessIntegerOr32:$redVal
  );
}
def NVVM_Tcgen05LdRedSpcompressOp : NVVM_Op<"tcgen05.ld.red.spcompress"> {
  let description = [{
    The `tcgen05.ld.red.spcompress` performs in-line sparse compression when loading data from Tensor Memory.
    The operation uses a combination of `abs` and `opKind` parameters to select values for compression.
    The NaN detection mode is always used for the compression operation, while the .NAN flag only applies
    to the reduction portion.

    Arguments:
    - `tmemAddr`: Tensor memory address to load from
    - `abs`: Whether to use absolute value comparison (used with opKind for value selection)
    - `opKind`: Comparison operator kind (min/max) for selecting values (used with abs for value selection)
    - `shape`: Pattern used to construct the Matrix (used with repFactor for Matrix construction)
    - `repFactor`: Repeat factor, number of times the pattern has to be repeated to form the Matrix (used with shape for Matrix construction)
    - `compFactor`: Compression factor (2:4 sparsity, choose 2 pair of elems from every 4 pairs of elems)
    - `nan`: Whether to use NaN detection mode

    Results:
    - `metadata`: Vector storing sparse metadata
    - `compressedData`: Vector storing compressed data
    - `redVal`: Reduction value

    Examples:
    ```mlir
    // MIN operation with x4 repetition factor
    %metadata, %compressed_data, %red_val = nvvm.tcgen05.ld.red.spcompress %taddr {
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<min>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x4>
    } : i32, vector<2xi32>

    // Max operation with absolute value comparison
    %metadata, %compressed_data, %red_val = nvvm.tcgen05.ld.red.spcompress %taddr {
      abs,
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<max>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x8>
    } : i32, vector<4xi32>

    // MIN operation with NaN detection for reduction
    %metadata, %compressed_data, %red_val = nvvm.tcgen05.ld.red.spcompress %taddr {
      nan,
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<min>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x16>
    } : i32, vector<8xi32>

    // MAX operation with abs and nan options
    %metadata, %compressed_data, %red_val = nvvm.tcgen05.ld.red.spcompress %taddr {
      abs,
      nan,
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<max>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x64>
    } : vector<2xi32>, vector<32xi32>

    // MIN operation with maximum repetition factor
    %metadata, %compressed_data, %red_val = nvvm.tcgen05.ld.red.spcompress %taddr {
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<min>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x128>
    } : vector<4xi32>, vector<64xi32>
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$tmemAddr,
    UnitAttr:$abs,
    UnitAttr:$nan,
    NVVM_Tcgen05LdStShapeAttr:$shape,
    NVVM_SPCompressOpKindAttr:$opKind,
    NVVM_SPCompressFactorTypeAttr:$compFactor,
    NVVM_SPCompressRepFactorAttr:$repFactor
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOrVector3:$metadata,
    NVVM_VectorOf32BitSignlessInteger4:$compressedData,
    NVVM_T32BitSignlessInteger:$redVal
  );
}
def NVVM_Tcgen05LdSpcompressOp : NVVM_Op<"tcgen05.ld.spcompress"> {
  let description = [{
    The `tcgen05.ld.spcompress` performs an in-line sparse compression when loading data from Tensor Memory.
    The operation uses a combination of `abs` and `opKind` parameters to select values for compression.

    With a 2-to-4 sparsity factor (specified by `compFactor`), half as much data is written to the output
    as is read from Tensor Memory. The sequence of sparse metadata is written to the `metadata` result,
    while the compressed data is written to `compressedData`.

    Arguments:
    - `tmemAddr`: Tensor memory address to load from
    - `abs`: Whether to use absolute value comparison (used with opKind for value selection)
    - `opKind`: Comparison operator kind (min/max) for selecting values (used with abs for value selection)
    - `shape`: Pattern used to construct the Matrix (used with repFactor for Matrix construction)
    - `repFactor`: Repeat factor, number of times the pattern has to be repeated to form the Matrix (used with shape for Matrix construction)
    - `compFactor`: Compression factor (2:4 sparsity, choose 2 pair of elems from every 4 pairs of elems)

    Results:
    - `metadata`: Vector storing sparse metadata
    - `compressedData`: Vector storing compressed data

    Example:
    ```mlir
    // MIN operation with x4 repetition factor
    %metadata, %compressed_data = nvvm.tcgen05.ld.spcompress %taddr {
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<min>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x4>
    } : i32, vector<2xi32>

    // MAX operation with abs option
    %metadata, %compressed_data = nvvm.tcgen05.ld.spcompress %taddr {
      abs,
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<max>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x8>
    } : i32, vector<4xi32>

    // MIN operation with x64 repetition factor
    %metadata, %compressed_data = nvvm.tcgen05.ld.spcompress %taddr {
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<min>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x64>
    } : vector<2xi32>, vector<32xi32>

    // MAX operation with abs and x128 repetition factor
    %metadata, %compressed_data = nvvm.tcgen05.ld.spcompress %taddr {
      abs,
      shape = #nvvm.tcgen05_ldst_shape<shape_32x32b>,
      opKind = #nvvm.spcomp_op_kind<max>,
      compFactor = #nvvm.spcomp_factor<sp2to4>,
      repFactor = #nvvm.spcomp_rep_factor<x128>
    } : vector<4xi32>, vector<64xi32>
    ```
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$tmemAddr,
    UnitAttr:$abs,
    NVVM_Tcgen05LdStShapeAttr:$shape,
    NVVM_SPCompressOpKindAttr:$opKind,
    NVVM_SPCompressFactorTypeAttr:$compFactor,
    NVVM_SPCompressRepFactorAttr:$repFactor
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOrVector3:$metadata,
    NVVM_VectorOf32BitSignlessInteger4:$compressedData
  );
}
def NVVM_Tcgen05MMAOp : NVVM_Op<"tcgen05.mma", [AttrSizedOperandSegments]> {
  let description = [{
    The `tcgen05.mma` operation is an asynchronous tensor core instruction that
    performs matrix multiplication, accumulation in a single fused operation. It
    targets 5th-generation tensor cores, providing developers with fine-grained
    control over execution and scheduling.

    ```
    D = A * B + (D * 2^ -scaleInputD)    // if `scaleInputD` is provided
    D = A * B                            // if `enableInputD` is false
    D = A * B + D                        // otherwise
    ```

    where:
    - A is an `M x K` matrix in tensor memory or described using shared memory descriptor
    - B is a `K x N` matrix described using shared memory descriptor
    - D is an `M x N` accumulator matrix in tensor memory

    The `shared memory descriptor` can be generated using `tcgen05.mma_smem_desc` Op

    - idesc is a 32-bit value representing the [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)

    Optional Operands:
    - `scaleInputD` is an Immediate value operand used for scaling D matrix by 2 ^ (-scaleInputD). The valid range is [0, 15]

    - `disableOutputLane` is a vector mask for selective output
      * vector<4 x i32> when ctaGroup is CTA_1
      * vector<8 x i32> when ctaGroup is CTA_2

    Required Attributes:
    - `kind` is a Tcgen05MMAKind attribute

    - `ctaGroup` specifies CTA group configuration
      * cta_1: MMA will be performed on the current thread's CTA
      * cta_2: MMA will be performed on the current thread and it's peer CTA

    Default Attributes:
    - collectorOp is a Tcgen05MMACollectorOp attribute with matrix A as the collector buffer

    - `aShift` shifts the rows of the A matrix down by one row and can only be
       applied if A is in tensor memory

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$matrixD,
    NVVM_LLVMPointerInAddressSpace62:$matrixA,
    NVVM_T64BitSignlessInteger:$matrixB,
    NVVM_T32BitSignlessInteger:$idesc,
    NVVM_T1BitSignlessInteger:$enableInputD,
    Optional<NVVM_T64BitSignlessInteger>:$scaleInputD,
    Optional<NVVM_FixedLengthVectorOf32Bit>:$disableOutputLane,
    NVVM_Tcgen05MMAKindAttr:$kind,
    NVVM_CTAGroupKindAttr:$ctaGroup,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOp,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpB,
    UnitAttr:$aShift
  );
  let results = (outs);
}
def NVVM_Tcgen05MMABlockScaleOp : NVVM_Op<"tcgen05.mma.block_scale"> {
  let description = [{
    The `tcgen05.mma.block_scale` operation is an asynchronous tensor core instruction
    that performs matrix multiplication, accumulation with block scaling in a
    single fused operation. It targets 5th-generation tensor cores, providing
    developers with fine-grained control over execution and scheduling.

    ```
    D = (A * scale_a)  * (B * scale_b)`      // if `enableInputD` is false
    D = (A * scale_a)  * (B * scale_b) + D`
    ```

    where:
    - A is an M x (K / 2) matrix in tensor memory or described using shared memory descriptor
    - B is a K x N matrix described using shared memory descriptor
    - D is an M x N accumulator matrix in tensor memory
    - `scale_a` and `scale_b` are matrices in tensor memory used to scale `A` and `B` respectively

    The `shared memory descriptor` can be generated using `tcgen05.mma_smem_desc` Op

    - `idesc` is a 32 bit value representing the [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)

    Required Attributes:
    - `kind` is a Tcgen05MMAKind attribute restricted to mxf8f6f4, mxf4, or mxf4nvf4

    - `ctaGroup` specifies CTA group configuration
      * cta_1: MMA will be performed on the current thread's CTA
      * cta_2: MMA will be performed on the current thread and it's peer CTA

    Default Attributes:
    - collectorOp is a Tcgen05MMACollectorOp attribute with matrix A as the collector buffer

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)
  }];
  let arguments = (ins
    NVVM_AnyType:$matrixD,
    NVVM_AnyType:$matrixA,
    NVVM_AnyType:$matrixB,
    NVVM_AnyType:$idesc,
    NVVM_AnyType:$enableInputD,
    NVVM_AnyType:$scaleA,
    NVVM_AnyType:$scaleB,
    NVVM_Tcgen05MMAKindAttr:$kind,
    NVVM_CTAGroupKindAttr:$ctaGroup,
    OptionalAttr<NVVM_Tcgen05MMABlockScaleAttr>:$blockScale,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOp,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpB
  );
  let results = (outs);
}
def NVVM_Tcgen05MMABlockScaleDecompressBOp : NVVM_Op<"tcgen05.mma.block_scale.decompress_b"> {
  let description = [{
    The `tcgen05.mma.block_scale.decompress_b` operation is an asynchronous tensor core
    instruction that decompresses the B matrix and performs matrix
    multiplication, accumulation with block scaling in a single fused operation.
    It targets 5th-generation tensor cores, providing developers with
    fine-grained control over execution and scheduling.

    ```
    D = (A * scale_a)  * (B * scale_b)`      // if `enableInputD` is false
    D = (A * scale_a)  * (B * scale_b) + D`
    ```

    where:
    - A is an M x (K / 2) matrix in tensor memory or described using shared memory descriptor
    - B is a `K x N` matrix described using shared memory descriptor
    - D is an `M x N` accumulator matrix in tensor memory
    - `scale_a` and `scale_b` are matrices in tensor memory used to scale `A` and `B` respectively

    The `shared memory descriptor` can be generated using `tcgen05.mma_smem_desc` Op

    - `idesc` is a 32-bit value representing the [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)

    - `decompressBMetadata` is a decompress metadata of B matrix

    Required Attributes:
    - `ctaGroup` specifies CTA group configuration
      * cta_1: MMA will be performed on the current thread's CTA
      * cta_2: MMA will be performed on the current thread and it's peer CTA

    Default Attributes:
    - collectorOpA is a Tcgen05MMACollectorOp attribute with matrix A as the collector buffer
    - collectorOpB is a Tcgen05MMACollectorOp attribute with matrix B as the collector buffer

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)
  }];
  let arguments = (ins
    NVVM_AnyType:$matrixD,
    NVVM_AnyType:$matrixA,
    NVVM_AnyType:$matrixB,
    NVVM_AnyType:$idesc,
    NVVM_AnyType:$enableInputD,
    NVVM_AnyType:$scaleA,
    NVVM_AnyType:$scaleB,
    NVVM_AnyType:$decompressBMetadata,
    NVVM_Tcgen05MMAKindAttr:$kind,
    NVVM_CTAGroupKindAttr:$ctaGroup,
    OptionalAttr<NVVM_Tcgen05MMABlockScaleAttr>:$blockScale,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpA,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpB
  );
  let results = (outs);
}
def NVVM_Tcgen05MMADecompressBOp : NVVM_Op<"tcgen05.mma.decompress_b"> {
  let description = [{
    The `tcgen05.mma.decompress_b` operation is an asynchronous tensor core
    instruction that decompresses the B matrix and performs matrix
    multiplication, accumulation in a single fused operation. It targets
    5th-generation tensor cores, providing developers with fine-grained
    control over execution and scheduling.

    ```
    D = A * B                            // if `enableInputD` is false
    D = A * B + D                        // otherwise
    ```

    where:
    - A is an `M x K` matrix in tensor memory or described using shared memory descriptor
    - B is a `K x N` matrix described using shared memory descriptor
    - D is an `M x N` accumulator matrix in tensor memory

    The `shared memory descriptor` can be generated using `tcgen05.mma_smem_desc` Op

    - `idesc` is a 32-bit value representing the [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)

    - `decompressBMetadata` is a decompress metadata of B matrix

    Optional Operands:
    - `disableOutputLane` is a vector mask for selective output
      * vector<4 x i32> when ctaGroup is CTA_1
      * vector<8 x i32> when ctaGroup is CTA_2

    Required Attributes:
    - `ctaGroup` specifies CTA group configuration
      * cta_1: MMA will be performed on the current thread's CTA
      * cta_2: MMA will be performed on the current thread and it's peer CTA

    Default Attributes:
    - collectorOAp is a Tcgen05MMACollectorOp attribute with matrix A as the collector buffer
    - collectorOpB is a Tcgen05MMACollectorOp attribute with matrix B as the collector buffer

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma)
  }];
  let arguments = (ins
    NVVM_AnyType:$matrixD,
    NVVM_AnyType:$matrixA,
    NVVM_AnyType:$matrixB,
    NVVM_AnyType:$idesc,
    NVVM_AnyType:$enableInputD,
    NVVM_AnyType:$decompressBMetadata,
    Optional<NVVM_AnyType>:$disableOutputLane,
    NVVM_Tcgen05MMAKindAttr:$kind,
    NVVM_CTAGroupKindAttr:$ctaGroup,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpA,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpB
  );
  let results = (outs);
}
def NVVM_Tcgen05MMASparseOp : NVVM_Op<"tcgen05.mma.sp", [AttrSizedOperandSegments]> {
  let description = [{
    The `tcgen05.mma.sp` operation is an asynchronous tensor core instruction
    that performs matrix multiplication, accumulation with sparse `A` matrix in
    a single fused operation. It targets 5th-generation tensor cores, providing
    developers with fine-grained control over execution and scheduling.

    ```
    D = A * B + (D * 2^ -scaleInputD)    // if `scaleInputD` is provided
    D = A * B                            // if `enableInputD` is false
    D = A * B + D                        // otherwise
    ```

    where:
    - A is an `M x (K / 2)` matrix in tensor memory or described using shared memory descriptor
    - B is a `K x N` matrix described using shared memory descriptor
    - D is an `M x N` accumulator matrix in tensor memory
    - sparseMetadata located in tensor memory specifies the mapping of the `K / 2`
    non-zero elements to the K elements before performing the MMA operation

    Other attributes and operands are similar to that of tcgen05.mma Op

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-sp)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$matrixD,
    NVVM_LLVMPointerInAddressSpace62:$matrixA,
    NVVM_T64BitSignlessInteger:$matrixB,
    NVVM_T32BitSignlessInteger:$idesc,
    NVVM_T1BitSignlessInteger:$enableInputD,
    NVVM_LLVMPointerInAddressSpace6:$sparseMetadata,
    Optional<NVVM_T64BitSignlessInteger>:$scaleInputD,
    Optional<NVVM_FixedLengthVectorOf32Bit>:$disableOutputLane,
    NVVM_Tcgen05MMAKindAttr:$kind,
    NVVM_CTAGroupKindAttr:$ctaGroup,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOp,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpB,
    UnitAttr:$aShift
  );
  let results = (outs);
}
def NVVM_Tcgen05MMASparseBlockScaleOp : NVVM_Op<"tcgen05.mma.sp.block_scale"> {
  let description = [{
    The `tcgen05.mma.sp.block_scale` operation is an asynchronous tensor core
    instruction that performs matrix multiplication, accumulation with block
    scaling, and sparse `A` matrix in a single fused operation. It targets
    5th-generation tensor cores, providing developers with fine-grained control
    over execution, and scheduling.

    ```
    D = (A * scale_a)  * (B * scale_b)      // if `enableInputD` is specified
    D = (A * scale_a)  * (B * scale_b) + D  // otherwise
    ```

    where:
    - A is an M x (K / 2) matrix in tensor memory or described using shared memory descriptor
    - B is a K x N matrix described using shared memory descriptor
    - D is an M x N accumulator matrix in tensor memory
    - `scale_a` and `scale_b` are matrices in tensor memory used to scale `A` and `B` respectively

    Other attributes and operands are similar to that of tcgen05.mma.block_scale Op

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-sp)
  }];
  let arguments = (ins
    NVVM_AnyType:$matrixD,
    NVVM_AnyType:$matrixA,
    NVVM_AnyType:$matrixB,
    NVVM_AnyType:$idesc,
    NVVM_AnyType:$enableInputD,
    NVVM_AnyType:$sparseMetadata,
    NVVM_AnyType:$scaleA,
    NVVM_AnyType:$scaleB,
    NVVM_Tcgen05MMAKindAttr:$kind,
    NVVM_CTAGroupKindAttr:$ctaGroup,
    OptionalAttr<NVVM_Tcgen05MMABlockScaleAttr>:$blockScale,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOp,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOpB
  );
  let results = (outs);
}
def NVVM_Tcgen05MMAWsOp : NVVM_Op<"tcgen05.mma.ws"> {
  let description = [{
    The `tcgen05.mma.ws` operation is an asynchronous tensor core instruction
    that performs weight stationary convolution matrix multiplication, accumulation
    in a single fused operation. It targets 5th-generation tensor cores, providing
    developers with fine-grained control over execution, and scheduling.

    ```
    D = A * B`      // if `enableInputD` is false
    D = A * B + D`  // otherwise
    ```

    where:
    - A is an `M x K` matrix in tensor memory or described using shared memory descriptor
    - B is a `K x N` matrix described using shared memory descriptor
    - D is an `M x N` accumulator matrix in tensor memory

    The `shared memory descriptor` can be generated using `tcgen05.mma_smem_desc` Op

    - idesc is a 32-bit value representing the [Instruction Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor)

    Optional Operands:
    - zeroColMask is a 64 bit value representing the [Zero-column mask descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-zero-column-mask-descriptor)

    Required Attributes:
    - `kind` is a Tcgen05MMAKind attribute

    Default Valued Attributes:
    - collectorBBuffer specifies collector buffer for matrix B: b0 (default), b1, b2, b3

    - collectorOp is a Tcgen05MMACollectorOp attribute with matrix B as the collector buffer

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-ws)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$matrixD,
    NVVM_LLVMPointerInAddressSpace62:$matrixA,
    NVVM_T64BitSignlessInteger:$matrixB,
    NVVM_T32BitSignlessInteger:$idesc,
    NVVM_T1BitSignlessInteger:$enableInputD,
    Optional<NVVM_T64BitSignlessInteger>:$zeroColMask,
    NVVM_Tcgen05MMAKindAttr:$kind,
    OptionalAttr<NVVM_Tcgen05MMACollectorBBufferAttr>:$collectorBBuffer,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOp
  );
  let results = (outs);
}
def NVVM_Tcgen05MMAWsSparseOp : NVVM_Op<"tcgen05.mma.ws.sp"> {
  let description = [{
    The `tcgen05.mma.ws.sp` operation is an asynchronous tensor core instruction
    that performs weight stationary convolution matrix multiplication, accumulation
    with sparse `A` matrix in a single fused operation. It targets 5th-generation
    tensor cores, providing developers with fine-grained control over execution,
    and scheduling.

    ```
    D = A * B`      // if `enableInputD` is false
    D = A * B + D`  // otherwise
    ```

    where:
    - A is an M x (K / 2) matrix in memory or descriptor format
    - B is a K x N matrix
    - D is an M x N accumulator matrix
    - sparseMetadata located in tensor memory specifies the mapping of the `K / 2`
    non-zero elements to the K elements before performing the MMA operation

    Other attributes and operands are similar to that of tcgen05.mma.ws Op

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-ws-sp)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$matrixD,
    NVVM_LLVMPointerInAddressSpace62:$matrixA,
    NVVM_T64BitSignlessInteger:$matrixB,
    NVVM_T32BitSignlessInteger:$idesc,
    NVVM_T1BitSignlessInteger:$enableInputD,
    NVVM_LLVMPointerInAddressSpace6:$sparseMetadata,
    Optional<NVVM_T64BitSignlessInteger>:$zeroColMask,
    NVVM_Tcgen05MMAKindAttr:$kind,
    OptionalAttr<NVVM_Tcgen05MMACollectorBBufferAttr>:$collectorBBuffer,
    OptionalAttr<NVVM_Tcgen05MMACollectorOpAttr>:$collectorOp
  );
  let results = (outs);
}
def NVVM_Tcgen05MmaSmemDescOp : NVVM_Op<"tcgen05.mma_smem_desc", [MemoryEffects<[MemRead]>]> {
  let description = [{
    The `nvvm.tcgen05_mma_smem_desc` constructs a Shared Memory descriptor
    for tcgen05.mma. This descriptor is a 64-bit value which describes the
    properties of multiplicand matrix in shared memory including its location
    in the shared memory of the current CTA.

    ```
    +-----------+------+------------------------------------------------------+
    | Bit-field | Size | Description                                          |
    +-----------+------+------------------------------------------------------+
    | 0-13      | 14   | Matrix start address                                 |
    | 14-15     | 2    | Reserved                                             |
    | 16-29     | 14   | Leading dim relative-offset (or) absolute-address    |
    | 30-31     | 2    | Reserved                                             |
    | 32-45     | 14   | Stride dimension byte offset                         |
    | 46-48     | 3    | Fixed constant value of 0b001                        |
    | 49-51     | 3    | Matrix base offset                                   |
    | 52        | 1    | Leading dimension stride mode:                       |
    |           |      |   0: byte offset relative                            |
    |           |      |   1: byte address absolute                           |
    | 53-60     | 8    | Fixed constant value of 0xb00000000                  |
    | 61-63     | 3    | Swizzling mode:                                      |
    |           |      |   0: No swizzling                                    |
    |           |      |   1: 128-Byte with 32B atomic swizzling              |
    |           |      |   2: 128-Byte swizzling                              |
    |           |      |   4: 64-Byte swizzling                               |
    |           |      |   6: 32-Byte swizzling                               |
    |           |      |   (Values 3, 5 and 7 are invalid)                    |
    +-----------+------+------------------------------------------------------+
    ```

    Example:
    ```mlir
      %desc = nvvm.tcgen05.mma_smem_desc (%startAddr, %leadingDimOffset, %strideDimOffset,
                                          %baseOffset, %leadingDimMode, %swizzleMode) : (i32, i32, i32, i8, i1, i8) -> i64
    ```
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$startAddr,
    NVVM_T32BitSignlessInteger:$leadingDimOffset,
    NVVM_T32BitSignlessInteger:$strideDimOffset,
    NVVM_T8BitSignlessInteger2:$baseOffset,
    NVVM_T1BitSignlessInteger:$leadingDimMode,
    NVVM_T8BitSignlessInteger2:$swizzleMode
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_Tcgen05RelinquishAllocPermitOp : NVVM_Op<"tcgen05.relinquish_alloc_permit"> {
  let description = [{
    The `tcgen05.relinquish_alloc_permit` Op specifies that the CTA
    of the executing thread is relinquishing the right to allocate
    Tensor Memory. So, it is illegal for a CTA to perform `tcgen05.alloc`
    after any of its constituent threads execute `tcgen05.relinquish_alloc_permit`.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-alloc-manage-instructions)
  }];
  let arguments = (ins
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group
  );
  let results = (outs);
}
def NVVM_Tcgen05ShiftOp : NVVM_Op<"tcgen05.shift"> {
  let description = [{
    The `tcgen05.shift` is an asynchronous instruction which initiates
    the shifting of 32-byte elements downwards across all the rows,
    except the last, by one row. The operand `taddr` specifies the base
    address of the matrix in Tensor Memory whose rows must be down shifted.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$taddr,
    OptionalAttr<NVVM_CTAGroupKindAttr>:$group
  );
  let results = (outs);
}
def NVVM_Tcgen05StOp : NVVM_Op<"tcgen05.st"> {
  let description = [{
    Instruction `tcgen05.st` asynchronously stores data from the source register `r`
    into the Tensor Memory at the location specified by the 32-bit address operand
    `tmemAddr`, collectively across all threads of the warps.

    The `shape` and the `num` attribute together determines the total dimension of
    the data which is stored to the Tensor Memory. The `shape` indicates the base
    dimension of data to be accessed. The `num` attribute indicates the repeat
    factor on the base dimension resulting in the total dimension of the data that
    is accessed.

    The shape `16x32bx2` performs two accesses into Tensor Memory of the shape
    `16x32b`. The base address of the first access is specified by `tmemAddr`
    and the base address of the second access is specified by
    `tmemAddr + offset`, where `offset` is an immediate argument.

    The unit attribute `unpack` can be used to unpack a 32-bit element
    in the register into two 16-bit elements and store them in adjacent columns.

    The following table describes the size of the vector for various combinations
    of `num` and `shape` attributes:
    ```
    |=====================================================================|
    | num/shape      |     16x32bx2/16x64b/32x32b |  16x128b   | 16x256b  |
    |=====================================================================|
    | x1             |          1                 |    2       |    4     |
    | x2             |          2                 |    4       |    8     |
    | x4             |          4                 |    8       |    16    |
    | x8             |          8                 |    16      |    32    |
    | x16            |          16                |    32      |    64    |
    | x32            |          32                |    64      |    128   |
    | x64            |          64                |    128     |    NA    |
    | x128           |          128               |    NA      |    NA    |
    |=====================================================================|
    ```

    Example:
    ```mlir
      nvvm.tcgen05.st %tmemAddr, %val, %offset unpack {
        shape = #nvvm.tcgen05_ldst_shape<shape_16x32bx2>,
      } : <2xi32>
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace6:$tmemAddr,
    NVVM_T32BitSignlessIntegerOrVector2:$val,
    Optional<NVVM_T64BitSignlessInteger>:$offset,
    UnitAttr:$unpack,
    NVVM_Tcgen05LdStShapeAttr:$shape
  );
  let results = (outs);
}
def NVVM_Tcgen05WaitOp : NVVM_Op<"tcgen05.wait"> {
  let description = [{
    The `tcgen05.wait<load>` causes the executing thread to block until
    all prior `tcgen05.ld` operations issued by the executing thread
    have completed. Similarly, the `tcgen05.wait<store>` causes the executing
    thread to block until all prior `tcgen05.st` operations issued by the
    executing thread have completed.
    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait)
  }];
  let arguments = (ins
    NVVM_Tcgen05WaitKindAttr:$kind
  );
  let results = (outs);
}
def NVVM_Tcgen05MmaSmemDescOpV2 : NVVM_Op<"tcgen05_mma_smem_desc_v2", [MemoryEffects<[MemRead]>]> {
  let description = [{
    The `nvvm.tcgen05_mma_smem_desc_v2` constructs a Shared Memory descriptor for use as the MMA
    Shared Memory operand. It configures the memory layout according to the provided parameters.

    The descriptor contains the following bitfields:
    - Bits 63-61 (3 bits): Swizzle descriptor type
      0 = SWIZZLE_NONE matrix descriptor
      1 = SWIZZLE_128B_ATOM_32B
      2 = SWIZZLE_128B matrix descriptor
      3 = RESERVED
      4 = SWIZZLE_64B descriptor
      5 = RESERVED
      6 = SWIZZLE_32B descriptor
      7 = RESERVED
    - Bits 60-54 (7 bits): Reserved
    - Bit 53 (1 bit): UTCQMMA.LUTB Matrix B K_Segment start offset in K=128 (48B) block
      0: K[0:63] = GDESC.start_address + 0B
      1: K[64:127] = GDESC.start_address + 24B
    - Bit 52 (1 bit): Matrix A or B leading dimension mode
      0 = BYTE_OFFSET (legacy)
      1 = NEXT_BUFFER_ADDR (circular buffer)
    - Bits 51-49 (3 bits): Matrix A or B base offset (valid only for matrix descriptor 1, 2 or 3, 4)
    - Bits 48-46 (3 bits): Matrix A or B descriptor version, needs to be set to 1
    - Bits 45-32 (14 bits): Matrix A or B stride dimension byte offset, 16 byte aligned, 4 LSBs not included
    - Bit 31 (1 bit): Reserved
    - Bits 30-16 (15 bits): Matrix A or B leading dimension byte offset or next buffer start address
      If LEADING_DIM_MODE == BYTE_OFFSET: Matrix A or B leading dimension byte offset
      If LEADING_DIM_MODE == NEXT_BUFFER_ADDR: Matrix A or B next buffer start address
      Offset or address is 16 byte aligned, 4 LSBs not included
    - Bit 15 (1 bit): Reserved
    - Bits 14-0 (15 bits): Matrix A or B start address, 16 byte aligned, 4 LSBs not included
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$startAddress,
    NVVM_T32BitSignlessInteger:$leadingDimOffset,
    NVVM_T32BitSignlessInteger:$strideDimOffset,
    NVVM_T8BitSignlessInteger2:$descriptorVersion,
    NVVM_T8BitSignlessInteger2:$baseOffset,
    NVVM_T1BitSignlessInteger:$leadingDimMode,
    NVVM_T1BitSignlessInteger:$kSegmentOffset,
    NVVM_T8BitSignlessInteger2:$swizzleType
  );
  let results = (outs
    NVVM_T64BitSignlessInteger2:$res
  );
}
def NVVM_TensormapCpFenceProxyOp : NVVM_Op<"tensormap.cp_fenceproxy"> {
  let description = [{
    The `tensormap.cp_fenceproxy` is a fused copy and fence operation.
    Instructions perform the following operations in order:

    Copies data of size specified by the `size` argument, in bytes, from the
    location specified by the address operand `src` in shared memory to the
    location specified by the address operand `dst` in the global memory, in
    the generic proxy.

    Establishes a uni-directional proxy release pattern on the ordering from
    the copy operation to the subsequent access performed in the tensormap
    proxy on the address `dst`

    The valid value of `size` operand is 128 and must be an immediate.

    The operands `src` and `dst` specify non-generic addresses in shared::cta and
    global state space respectively
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace1:$dst,
    NVVM_LLVMPointerInAddressSpace3:$src,
    NVVM_T32BitSignlessInteger:$size,
    NVVM_MemScopeKindAttr:$scope
  );
  let results = (outs);
}
def NVVM_TensormapReplaceOp : NVVM_Op<"tensormap.replace"> {
  let description = [{
    The `nvvm.tensormap.replace` replaces the specified field of the tensor-map
    object at the location specified by `addr` with a new value (specified by
    `new_value` or `new_value_attr`).

    The `field` argument specifies the field of the tensor-map object to
    replace.

    `new_value` is an `i32`/`i64` argument that specifies the new value to
    replace the `field` with for the `global_address`, `rank`, `box_dim`,
    `global_dim`, `global_stride`, and `element_stride` fields. It must be an
    `i64` for the `global_address` and `global_stride` fields and `i32` for the
    remaining fields.

    For `rank`, `new_value` must be one less than the desired tensor rank as
    this field uses zero-based numbering.

    `new_value_attr` is an attribute that specifies the new value to replace
    the `field` with for the `elemtype`, `interleave_layout`, `swizzle_mode`,
    `swizzle_atomicity`, and `fill_mode` fields. It takes the place of
    `new_value` for these fields. It must be a valid attribute corresponding to
    the `field` type.

    The ordinal `ord` is an immediate integer argument that specifies the
    ordinal of the `field` across the tensor which needs to be replaced and is
    required only for the `box_dim`, `global_dim`, `global_stride`, and
    `element_stride` fields.

    [For more information, see PTX ISA.](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-tensormap-replace)
  }];
  let arguments = (ins
    NVVM_LLVMPointerInAddressSpace14:$addr,
    Optional<NVVM_T64BitSignlessIntegerOr32>:$new_value,
    NVVM_TensormapFieldAttr:$field,
    OptionalAttr<I32Attr>:$ord,
    OptionalAttr<NVVM_TensormapFieldValueAttr>:$new_value_attr
  );
  let results = (outs);
}
def NVVM_TraceMarkOp : NVVM_Op<"trace.mark", [AttrSizedOperandSegments]> {
  let description = [{
    The `nvvm.trace.mark` operation introduces trace events for profiling
    programs.

    It takes `event_type`, a 12-bit value in the range 0-4095, which denotes
    the type of the event. Additionally, it takes `domain` and `event`, which
    are strings that identify the domain and event name of the trace event.
    Finally, it takes an optional 32-bit or 64-bit payload, an optional
    payload descriptor (required when payload is provided), and a predicate to
    conditionally execute the trace event.

    Examples:

    ```mlir
    // Basic trace event
    nvvm.trace.mark event_type = 42, domain = "myDomain", event = "myEvent"

    // With 32-bit payload
    nvvm.trace.mark event_type = 42, payload = %p,
        domain = "myDomain", event = "myEvent", payload_descriptor = "counter"
        : i32

    // Conditional trace event
    nvvm.trace.mark event_type = 42, predicate = %pred,
        domain = "myDomain", event = "myEvent"
    ```
  }];
  let arguments = (ins
    Optional<NVVM_T32BitSignlessIntegerOr64>:$payload,
    Optional<NVVM_T1BitSignlessInteger>:$predicate,
    I32Attr:$eventType,
    StrAttr:$domain,
    StrAttr:$event,
    OptionalAttr<StrAttr>:$payloadDescriptor
  );
  let results = (outs);
}
def NVVM_VoteSyncOp : NVVM_InferredOp<"vote.sync"> {
  let description = [{
    The `vote.sync` op will cause executing thread to wait until all non-exited
    threads corresponding to membermask have executed `vote.sync` with the same
    qualifiers and same membermask value before resuming execution.

    The vote operation kinds are:
    - `any`: True if source predicate is True for some thread in membermask.
    - `all`: True if source predicate is True for all non-exited threads in
      membermask.
    - `uni`: True if source predicate has the same value in all non-exited
      threads in membermask.
    - `ballot`: In the ballot form, the destination result is a 32 bit integer.
      In this form, the predicate from each thread in membermask are copied into
      the corresponding bit position of the result, where the bit position
      corresponds to the thread's lane id.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync)
  }];
  let arguments = (ins
    NVVM_T32BitSignlessInteger:$mask,
    NVVM_T1BitSignlessInteger:$pred,
    NVVM_VoteSyncKindAttr:$kind
  );
  let results = (outs
    NVVM_T32BitSignlessIntegerOr1:$res
  );
}
def NVVM_WgmmaGroupSyncAlignedOp : NVVM_Op<"wgmma.commit.group.sync.aligned"> {
  let description = [{
    Commits all prior uncommitted warpgroup level matrix multiplication operations.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_WgmmaFenceAlignedOp : NVVM_Op<"wgmma.fence.aligned"> {
  let description = [{
    Enforce an ordering of register accesses between warpgroup level matrix
    multiplication and other operations.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)
  }];
  let arguments = (ins);
  let results = (outs);
}
def NVVM_WgmmaMmaAsyncOp : NVVM_Op<"wgmma.mma_async"> {
  let description = [{
    The warpgroup (128 threads) level matrix multiply and accumulate operation
    has either of the following forms, where matrix D is called accumulator:
      D = A * B + D
      D = A * B, where the input from accumulator D is disabled.

    Supported shapes:
    ```
    |--------------|--------------|------------|--------------|---------------|
    |              |              |            |              |f16+=e4m3*e4m3 |
    |              |              |            |              |f16+=e5m2*e5m2 |
    |f32+=tf32*tf32|f16+=f16 *f16 | s32+=s8*s8 |s32 += b1 * b1|f16+=e5m2*e4m3 |
    |              |f32+=f16 *f16 | s32+=u8*u8 |              |f16+=e4m3*e5m2 |
    |              |f32+=bf16*bf16| s32+=u8*u8 |              |f16+=e4m3*e5m2 |
    |              |f32+=bf16*bf16| s32+=s8*u8 |              |f32+=e4m3*e4m3 |
    |              |              | s32+=u8*s8 |              |f32+=e5m2*e5m2 |
    |              |              |            |              |f32+=e4m3*e5m2 |
    |              |              |            |              |f32+=e4m3*e5m2 |
    |--------------|--------------|------------|--------------|---------------|
    |   .m64n8k8   |  .m64n8k16   | .m64n8k32  | .m64n8k256   | .m64n8k32     |
    |   .m64n16k8  |  .m64n16k16  | .m64n16k32 | .m64n16k256  | .m64n16k32    |
    |   .m64n24k8  |  .m64n24k16  | .m64n24k32 | .m64n24k256  | .m64n24k32    |
    |   .m64n32k8  |  .m64n32k16  | .m64n32k32 | .m64n32k256  | .m64n32k32    |
    |   .m64n40k8  |  .m64n40k16  | .m64n48k32 | .m64n48k256  | .m64n40k32    |
    |   .m64n48k8  |  .m64n48k16  | .m64n64k32 | .m64n64k256  | .m64n48k32    |
    |   .m64n56k8  |  .m64n56k16  | .m64n80k32 | .m64n80k256  | .m64n56k32    |
    |   .m64n64k8  |  .m64n64k16  | .m64n96k32 | .m64n96k256  | .m64n64k32    |
    |   .m64n72k8  |  .m64n72k16  | .m64n112k32| .m64n112k256 | .m64n72k32    |
    |   .m64n80k8  |  .m64n80k16  | .m64n128k32| .m64n128k256 | .m64n80k32    |
    |   .m64n88k8  |  .m64n88k16  | .m64n144k32| .m64n144k256 | .m64n88k32    |
    |   .m64n96k8  |  .m64n96k16  | .m64n160k32| .m64n160k256 | .m64n96k32    |
    |   .m64n104k8 |  .m64n104k16 | .m64n176k32| .m64n176k256 | .m64n104k32   |
    |   .m64n112k8 |  .m64n112k16 | .m64n192k32| .m64n192k256 | .m64n112k32   |
    |   .m64n120k8 |  .m64n120k16 | .m64n208k32| .m64n208k256 | .m64n120k32   |
    |   .m64n128k8 |  .m64n128k16 | .m64n224k32| .m64n224k256 | .m64n128k32   |
    |   .m64n136k8 |  .m64n136k16 | .m64n240k32| .m64n240k256 | .m64n136k32   |
    |   .m64n144k8 |  .m64n144k16 | .m64n256k32| .m64n256k256 | .m64n144k32   |
    |   .m64n152k8 |  .m64n152k16 |            |              | .m64n152k32   |
    |   .m64n160k8 |  .m64n160k16 |            |              | .m64n160k32   |
    |   .m64n168k8 |  .m64n168k16 |            |              | .m64n168k32   |
    |   .m64n176k8 |  .m64n176k16 |            |              | .m64n176k32   |
    |   .m64n184k8 |  .m64n184k16 |            |              | .m64n184k32   |
    |   .m64n192k8 |  .m64n192k16 |            |              | .m64n192k32   |
    |   .m64n200k8 |  .m64n200k16 |            |              | .m64n200k32   |
    |   .m64n208k8 |  .m64n208k16 |            |              | .m64n208k32   |
    |   .m64n216k8 |  .m64n216k16 |            |              | .m64n216k32   |
    |   .m64n224k8 |  .m64n224k16 |            |              | .m64n224k32   |
    |   .m64n232k8 |  .m64n232k16 |            |              | .m64n232k32   |
    |   .m64n240k8 |  .m64n240k16 |            |              | .m64n240k32   |
    |   .m64n248k8 |  .m64n248k16 |            |              | .m64n248k32   |
    |   .m64n256k8 |  .m64n256k16 |            |              | .m64n256k32   |
    |--------------|--------------|------------|--------------|---------------|
    ```

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)
  }];
  let arguments = (ins
    NVVM_LLVMStructureType:$inouts,
    NVVM_T64BitSignlessInteger:$descriptorA,
    NVVM_T64BitSignlessInteger:$descriptorB,
    NVVM_MMAShapeAttr:$shape,
    NVVM_WGMMATypesAttr:$typeA,
    NVVM_WGMMATypesAttr:$typeB,
    NVVM_WGMMATypesAttr:$typeD,
    NVVM_WGMMAScaleOutAttr:$scaleD,
    NVVM_WGMMAScaleInAttr:$scaleA,
    NVVM_WGMMAScaleInAttr:$scaleB,
    NVVM_MMALayoutAttr:$layoutA,
    NVVM_MMALayoutAttr:$layoutB,
    OptionalAttr<NVVM_MMAIntOverflowAttr>:$satfinite
  );
  let results = (outs
    NVVM_LLVMStructureType:$results_
  );
}
def NVVM_WgmmaWaitGroupSyncOp : NVVM_Op<"wgmma.wait.group.sync.aligned"> {
  let description = [{
    Signal the completion of a preceding warpgroup operation.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group)
  }];
  let arguments = (ins
    I64Attr:$group
  );
  let results = (outs);
}
def NVVM_WMMALoadOp : NVVM_Op<"wmma.load"> {
  let arguments = (ins
    NVVM_LLVMPointerType:$ptr,
    NVVM_T32BitSignlessInteger:$stride,
    I32Attr:$m,
    I32Attr:$n,
    I32Attr:$k,
    NVVM_MMALayoutAttr:$layout,
    NVVM_MMATypesAttr:$eltype,
    NVVM_MMAFragAttr:$frag
  );
  let results = (outs
    NVVM_LLVMStructureTypeOr64Bit:$res
  );
}
def NVVM_WMMAMmaOp : NVVM_Op<"wmma.mma"> {
  let arguments = (ins
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$args,
    I32Attr:$m,
    I32Attr:$n,
    I32Attr:$k,
    NVVM_MMALayoutAttr:$layoutA,
    NVVM_MMALayoutAttr:$layoutB,
    NVVM_MMATypesAttr:$eltypeA,
    NVVM_MMATypesAttr:$eltypeB
  );
  let results = (outs
    NVVM_LLVMStructureType:$res
  );
}
def NVVM_WMMAStoreOp : NVVM_Op<"wmma.store"> {
  let arguments = (ins
    NVVM_LLVMPointerType:$ptr,
    Variadic<NVVM_VariadicOfLLVMDialectCompatibleType>:$args,
    NVVM_T32BitSignlessInteger:$stride,
    I32Attr:$m,
    I32Attr:$n,
    I32Attr:$k,
    NVVM_MMALayoutAttr:$layout,
    NVVM_MMATypesAttr:$eltype
  );
  let results = (outs);
}

#endif // CUTE_IR_GENERATED_NVVM_OPS_TD
