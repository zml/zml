// The cute dialect's operations, generated from the nvidia-cutlass-dsl 4.8.0 release; do not edit.
//
// Sources: the operation bindings (`_cute_ops_gen.py`: names, classes,
// docstrings, operand/result/attribute structure, inference) and the live
// compiler (`_cutlass_ir` loaded from Python): each operand and result
// constraint carries the verifier's wording and accepts every type class the
// verifier let through when probed with one instance of every type; traits
// and interfaces are the registered operation's; memory effects come from
// loop-invariant code motion (hoisted: Pure), canonicalize (dead: no write)
// and CSE (merged: at most reads). Operations whose effects include writes
// or could not be told apart carry none (unknown effects).
//
// Operations keep the generic assembly form; the compiler's custom syntax is
// shown in each description.
//
// Constraints not recovered:
//   cute.deref_desc_iter operand #0: no probe type accepted; any type allowed
//   cute.make_desc_iter operand #0: no probe type accepted; any type allowed

#ifndef CUTE_IR_GENERATED_CUTE_OPS_TD
#define CUTE_IR_GENERATED_CUTE_OPS_TD

include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"
include "cute_ir/Dialect/Cute/IR/CuteEnums.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Operand and result constraints, in the compiler's words.
def Cute_32BitSignlessIntegerOr64BitSignless : Type<CPred<"($_self.isSignlessInteger(32) || $_self.isSignlessInteger(64))">, "32-bit signless integer or 64-bit signless integer">;
def Cute_AMemoryReferenceTypeThatRepresentsATensor : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self))">, "A memory reference type that represents a tensor view with an iterator and layout or CuTe Layout Type or Cute shape type">;
def Cute_AMemoryReferenceTypeThatRepresentsATensor2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "A memory reference type that represents a tensor view with an iterator and layout or Cute coord_tensor type">;
def Cute_AMemoryReferenceTypeThatRepresentsATensor3 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "A memory reference type that represents a tensor view with an iterator and layout">;
def Cute_ATupleIsAFixedLengthOrderedList : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TupleType>($_self))">, "A tuple is a fixed-length, ordered list consisting of multiple elements.">;
def Cute_ATypeInterfaceForDescriptorIteratorTypes : Type<CPred<"true">, "A type interface for descriptor iterator types.">;  // not recovered: no pool type passed
def Cute_ATypeInterfaceForMemoryDescriptorsSuchAs : Type<CPred<"true">, "A type interface for memory descriptors such as smem desc etc.">;  // not recovered: no pool type passed
def Cute_ArithmeticTupleType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self))">, "Arithmetic tuple type">;
def Cute_AtomTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledMmaType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM120BlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomLdsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomS2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTAsyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemStType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100S2TCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomStsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM10xTmemLoadRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTSyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::UniversalFmaAtomType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80SparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM89Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM90Type>($_self))">, "AtomTypeInterface instance">;
def Cute_CopyAtomTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomLdsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomS2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTAsyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemStType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomStsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM10xTmemLoadRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTSyncCopyType>($_self))">, "CopyAtomTypeInterface instance">;
def Cute_CopyAtomTypeInterfaceInstanceOrCopyAtomTypeInterfaceV2Instance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomLdsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomS2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTAsyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemStType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100S2TCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomStsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM10xTmemLoadRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTSyncCopyType>($_self))">, "CopyAtomTypeInterface instance or CopyAtomTypeInterfaceV2 instance">;
def Cute_CopyAtomTypeInterfaceV2Instance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100S2TCopyV2Type>($_self))">, "CopyAtomTypeInterfaceV2 instance">;
def Cute_CuTeComposedLayoutType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self))">, "CuTe composed layout type">;
def Cute_CuTeIteratorTypeWithIntTupleAsIts : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self))">, "CuTe iterator type with int_tuple as its underlying implementation">;
def Cute_CuTeLayoutType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self))">, "CuTe Layout Type">;
def Cute_CuTeLayoutTypeOrCuteSwizzleType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self))">, "CuTe Layout Type or Cute swizzle type">;
def Cute_CuTeLayoutTypeOrSwizzleWithValidParameters : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self))">, "CuTe Layout Type or swizzle with valid parameters that can be represented by 32-bit masks">;
def Cute_CuTePointerTo32BitSignlessInteger : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to 32-bit signless integer">;
def Cute_CuTePointerType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, " CuTe pointer type ">;
def Cute_CuTePointerTypeOrAMemoryReferenceType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::FastDivmodDivisorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledMmaType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM120BlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100S2TCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80SparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM90Type>($_self) || ::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::FloatType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMArrayType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMStructType>($_self) || ($_self.getDialect().getNamespace() == \"ptr\" || (::llvm::isa<::mlir::OpaqueType>($_self) && ::llvm::cast<::mlir::OpaqueType>($_self).getDialectNamespace() == \"ptr\")) || ::llvm::isa<::mlir::VectorType>($_self))">, " CuTe pointer type  or A memory reference type that represents a tensor view with an iterator and layout or  Integer type with constraints.  or CuTe fast divmod divisor type or loadable maybe-static CuTe type with runtime storage or LLVM type with size">;
def Cute_CuTePointerTypeOrLLVMPointerType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self))">, " CuTe pointer type  or LLVM pointer type">;
def Cute_CuTeSparseElementType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::SparseElemType>($_self))">, " CuTe sparse element type">;
def Cute_CuteCoordTensorType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self))">, "Cute coord_tensor type">;
def Cute_CuteCoordType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self))">, "Cute coord type">;
def Cute_CuteShapeType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self))">, "Cute shape type">;
def Cute_CuteShapeTypeOrIntTupleAsIntegerOr : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self))">, "Cute shape type or IntTuple as integer or tuple of IntTuple(s) or Cute coord type">;
def Cute_CuteShapeTypeOrIntTupleAsIntegerOr2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "Cute shape type or IntTuple as integer or tuple of IntTuple(s) or Cute coord type or CuTe Layout Type or CuTe composed layout type or ViewTypeInterface instance">;
def Cute_CuteStrideType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self))">, "Cute stride type">;
def Cute_CuteTileType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self))">, "Cute tile type">;
def Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::FastDivmodDivisorType>($_self))">, "Divisor for fast unsigned integer divmods by invariant integers.">;
def Cute_GeneralizedIteratorTypeOfCuTe : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescType>($_self))">, " Generalized iterator type of CuTe ">;
def Cute_GeneralizedLayoutTypeOfCuTe : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self))">, " Generalized layout type of CuTe ">;
def Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, " Generalized layout type of CuTe  or ViewTypeInterface instance">;
def Cute_I32I64OrConstrainedInteger : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || $_self.isSignlessInteger(32) || $_self.isSignlessInteger(64))">, "i32, i64 or Constrained integer">;
def Cute_IntTupleAsIntegerOrTupleOfIntTupleS : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self))">, "IntTuple as integer or tuple of IntTuple(s)">;
def Cute_IntTupleAsIntegerOrTupleOfIntTupleS2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self))">, "IntTuple as integer or tuple of IntTuple(s) or Cute shape type">;
def Cute_IntTupleTypeThatDepthIs0 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self))">, "Int tuple type that depth is 0">;
def Cute_Integer : Type<CPred<"(::llvm::isa<::mlir::IntegerType>($_self))">, "integer">;
def Cute_IntegerOrIntegerTypeWithConstraints : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::IntegerType>($_self))">, "integer or  Integer type with constraints. ">;
def Cute_IntegerTypeWithConstraints : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self))">, " Integer type with constraints. ">;
def Cute_MaybeStaticTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledMmaType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM120BlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomLdsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomS2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTAsyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemStType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100S2TCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomStsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM10xTmemLoadRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTSyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::UniversalFmaAtomType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80SparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM89Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM90Type>($_self))">, "MaybeStaticTypeInterface instance">;
def Cute_MemrefInGenericOrRegisterMemOrSmem : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "memref in generic or register mem or smem">;
def Cute_MemrefInSmemWithStaticAndNonComposed : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "memref in smem with static and non-composed layout">;
def Cute_MmaAtomTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledMmaType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM120BlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::UniversalFmaAtomType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80SparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM89Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM90Type>($_self))">, "MmaAtomTypeInterface instance">;
def Cute_OfRanks1 : Type<CPred<"(::llvm::isa<::mlir::VectorType>($_self))">, " of ranks 1">;
def Cute_OfRanks1OrFloatingPointOrInteger : Type<CPred<"(::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::FloatType>($_self) || ::llvm::isa<::mlir::VectorType>($_self))">, " of ranks 1 or floating-point or integer">;
def Cute_PointerTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "PointerTypeInterface instance">;
def Cute_PrefetchAtomTypeInterfaceOrTiledCopyTypeWithPrefetchAtomTypeInterfaceCopyAtom : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self))">, "PrefetchAtomTypeInterface or TiledCopyType with PrefetchAtomTypeInterface copy atom">;
def Cute_RefinableTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "RefinableTypeInterface instance">;
def Cute_RefinableTypeInterfaceInstanceOrInteger : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self) || ::llvm::isa<::mlir::IntegerType>($_self))">, "RefinableTypeInterface instance or integer">;
def Cute_StaticLayout : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self))">, "static layout">;
def Cute_StaticShape : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "static shape">;
def Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self))">, "Tile-like types, e.g. LayoutType, ShapeType, TileType">;
def Cute_TiledCopyAtomType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self))">, "Tiled copy atom type.">;
def Cute_TiledCopyAtomTypeOrTiledCopyV2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self))">, "Tiled copy atom type. or Tiled copy V2 atom type.">;
def Cute_TiledCopyV2AtomType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self))">, "Tiled copy V2 atom type.">;
def Cute_TiledMMAAtomType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledMmaType>($_self))">, "Tiled MMA atom type.">;
def Cute_TypeInterfaceForCuteTypesCanBePrinted : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ArithTupleIteratorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self) || ::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::FloatType>($_self))">, "Type interface for cute types can be printed by `cute.print`">;
def Cute_ValueTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self))">, "ValueTypeInterface instance">;
def Cute_ValueTypeInterfaceInstanceOrIntegerTypeWithConstraints : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self))">, "ValueTypeInterface instance or  Integer type with constraints. ">;
def Cute_ValueTypeInterfaceInstanceOrIntegerTypeWithConstraintsOr : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ConstrainedIntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::SwizzleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || $_self.isSignlessInteger(8) || $_self.isSignlessInteger(16) || $_self.isSignlessInteger(32) || $_self.isSignlessInteger(64) || ::llvm::isa<::mlir::IndexType>($_self))">, "ValueTypeInterface instance or  Integer type with constraints.  or 8/16/32/64-bit integer or index">;
def Cute_VectorOf1BitSignlessIntegerValuesOf : Type<CPred<"(::llvm::isa<::mlir::VectorType>($_self))">, "vector of 1-bit signless integer values of ranks 1">;
def Cute_ViewTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "ViewTypeInterface instance">;
def Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "ViewTypeInterface instance or  Generalized layout type of CuTe ">;
def Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "ViewTypeInterface instance or  Generalized layout type of CuTe  or Cute tile type">;
def Cute_XTupleLikeType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self))">, "XTuple-like type">;
def Cute_XTupleLikeTypeOrCuTeLayoutType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::StrideType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TileType>($_self))">, "XTuple-like type or CuTe Layout Type">;

// Operations whose result types are inferred in C++. Verification compares
// them with the inferred ones where the rule is recovered for the operands;
// elsewhere the types written stand, as the compiler accepted them.
class Cute_InferredOp<string mnemonic, list<Trait> traits = []>
    : Cute_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferTypeOpInterface, ["refineReturnTypes"]>]> {
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

def Cute_AddOffsetOp : Cute_InferredOp<"add_offset", [Pure]> {
  let description = [{
    The `cute.add_offset` operation accepts an input IteratorTypeInterface and
    a corresponding offset type which must be an
    arithmetic tuple. It returns a new SSA value as type IteratorTypeInterface.

    When src operand is a sparse pointer, the logical offset must be a multiple
    of the sparsity factor. The actual offset added to the src operand is
    equal to the logical offset divided by the sparsity factor.

    `add_offset(%it, N)` returns a new iterator pointing to an object with an
    offset of`N` elements. It is similar to C++'s iterator arithmetic.

    Example:

    ```mlir
    // Add offset to 1-D linear pointer
    %ptr = cute.add_offset(%0, %1) : (!cute.ptr<f16, smem, align<128>>, !cute.int_tuple<"?">) -> !cute.ptr<f16, smem, align<128>>

    // Add offset to 1-D linear pointer with static offset
    %off = cute.static : !cute.int_tuple<"1012">
    %ptr = cute.add_offset(%0, %off) : (!cute.ptr<f16, smem, align<128>>) -> !cute.ptr<f16, smem, align<128>>

    // Add offset as arithmetic tuple to N-D nested arithmetic tuple
    %arith_tuple = cute.add_offset(%0, %1) : (!cute.int_tuple<"(?,(3,?))">, !cute.int_tuple<"(3,(1,?))">) -> !cute.int_tuple<"(?,(4,?))">

    // Add offset as arithmetic tuple to N-D nested arithmetic tuple with static offset
    %arith_tuple = cute.add_offset(%0, "(3,(1,3))") : (!cute.int_tuple<"(?,(3,?))">) -> !cute.int_tuple<"(?,(4,?))">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedIteratorTypeOfCuTe:$src,
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$offset
  );
  let results = (outs
    Cute_GeneralizedIteratorTypeOfCuTe:$dst
  );
}
def Cute_AlgorithmCopyOp : Cute_Op<"copy", [AttrSizedOperandSegments]> {
  let description = [{
    CuTe's copy algorithm copies the elements of a source Tensor into the elements
    of a destination Tensor.

    See the file cutlass/media/docs/cute/04_algorithms.md in the CUTLASS repository
    for more information.

    A Tensor (cute.memref) encapsulates the data type, data location, and possibly
    also the shape and stride of the tensor at compile time. As a result, copy can
    and does dispatch, based on the types of its arguments, to use any of various
    synchronous or asynchronous hardware copy instructions.

    An .copy accepts CuteCopyAtomTypeInterface as first parameter. To keep consistent
    interface with CUTLASS C++, it could be either atom.tiled_copy or atom.universal_copy.

    Optional argument 'pred' is used to guard memory accesses performed in copy.
    Predication works in granularity of atom_v. For src tensor whose shape is:
    `((atom_v, rest_v), rest_m, rest_n, ...)`, the pred tensor shape should be:
    `((rest_v), rest_m, rest_n, ...)`, we denoted this as the canonical format.
    If the pred tensor's value equals 0, the corresponding copy will be skipped.

    When using atom.universal_copy instead of atom.tiled_copy, we allow the layout of
    `pred` to deviate from the canonical format. Instead, the layout shape of `pred`
    must match the layout shape of `src` or `dst`. Additionally, the `copy_bits`
    parameter cannot be set for atom.universal_copy when predicate is present.

    The operands `src` and `dst` are variadic, each containing a variable number of tensors:
    - For regular copy, `src` and `dst` each contain a single tensor.
    - For copy with auxiliary operands, they contain the main tensor followed by
      auxiliary tensors. For example:
      - For load with row-wise reduction, `dst` = [data, stat].
      - For SPARSIFY (structured sparsity compression), `dst` = [data, metadata].
      - For TMA gather4, `src` = [coord0, coord1, coord2, coord3] (four 2D coordinate tensors).
      - For TMA scatter4, `dst` = [coord0, coord1, coord2, coord3] (four 2D coordinate tensors).

    Example:

    ```mlir
    cute.copy(%tiled_copy, %src, %dst) : (
      !cute.tiled_copy<!cute_nvgpu.atom.tma_load<mode = tiled, num_cta = 1, mcast>,
                       layout_copy_tv = <"((4,2),(64,128)):((0,64),(128,1))">,
                       tiler_mn = <"[2048:1;1:1;4:1;1:1]">>,
      !cute.coord_tensor<"(0,0)", "(((64, 128), 1), 1, 1, 2):(((1@0, 1@1), 0), 0, 0, 64@0)">,
      !cute.memref<f16, smem, align<128>, "(((64,128),1),1,1,2):(((1,64),0),0,0,8192)">)

    // atom.universal_copy with canonicalize layout format
    cute.copy(%copy_atom, %src, %dst, %pred) : (
      !cute_nvgpu.atom.universal_copy<i32>,
      !cute.memref<i32, gmem, align<4>, "((1,1),2,4):((0,0),1,2)">,
      !cute.memref<i32, rmem, align<4>, "((1,1),2,4):((0,0),1,2)">,
      !cute.memref<i8,  rmem, align<1>, "((1),2,4):((0),1,2)">)

    // atom.universal_copy with relaxed layout format
    cute.copy(%atom, %arg0, %arg1, %arg2) : (
      !cute_nvgpu.atom.universal_copy<f32>,
      !cute.memref<f32, rmem, align<8>, "(4,4):(1,4)">,
      !cute.memref<f32, gmem, "(4,4):(?,?{div=512})">,
      !cute.memref<i1, rmem, align<8>, "(4,4):(1,4)">)

    ```
  }];
  let arguments = (ins
    Cute_CopyAtomTypeInterfaceInstanceOrCopyAtomTypeInterfaceV2Instance:$copy_atom,
    Variadic<Cute_ViewTypeInterfaceInstance>:$src,
    Variadic<Cute_ViewTypeInterfaceInstance>:$dst,
    Optional<Cute_ViewTypeInterfaceInstance>:$pred
  );
  let results = (outs
  );
}
def Cute_AlgorithmGemmOp : Cute_Op<"gemm", [AttrSizedOperandSegments]> {
  let description = [{
    CuTe's gemm algorithm takes 1 mma atom and 4 inputs(cute.memref/cute_nvgpu.smem_desc_view)
    as input.
    It performs the general matrix multiplication:
                               `D += A * B + C`.
    Layout carried by operads determines the exact form of the gemm computation.
    Generally speaking, there are 5 valid forms:
    1. `(V) x (V) => (V)`.
         The element-wise product of vectors. Dispatches to FMA or MMA.
         In this case, `rank(A) = rank(B) = rank(C) = rank(D) = 1`.
    2. `(M) x (N) => (M,N)`.
         The outer product of vectors. Dispatches to [3] with new mode K=(1).
         In this case, `rank(A) = rank(B) = 1` and `rank(C) = rank(D) = 2`.
    3. `(M,K) x (N,K) => (M,N)`.
         The product of matrices. Dispatches to [5] with MMA vector-mode V.
         In this case, `rank(A) = rank(B) = rank(C) = rank(D) = 2`.
    4. `(V,M) x (V,N) => (V,M,N)`.
         The batched outer product of vectors. Accounts for register reuse and dispatches to [1] for each (m,n).
         In this case, `rank(A) = rank(B) = 2` and `rank(C) = rank(D) = 3`.
    5. `(V,M,K) x (V,N,K) => (V,M,N)`.
         The batched product of matrices. Dispatches to [4] for each (k).
         In this case, `rank(A) = rank(B) = rank(C) = rank(D) = 3`.

    Currently, only the form in item 5 is supported.

    The operands `a` and `b` are variadic, each containing a variable number of tensors:
    - For regular GEMM, `a` and `b` contain the GEMM A and B tensors respectively.
    - For GEMM with auxiliary operands, `a` and `b` contain the GEMM A and B tensors followed by their respective auxiliary tensors. For example:
      - For BlockScaledGemm, `a` = [A, SFA] and `b` = [B, SFB].
      - For SparseGemm, `a` = [A, E] and `b` = [B].
      - For BlockScaledSparseGemm, `a` = [A, SFA, E] and `b` = [B, SFB].
    See the file cutlass/media/docs/cute/04_algorithms.md and include/cute/algorithm/gemm.hpp in the CUTLASS repository for more information.

    Example (regular GEMM):
    ```mlir
    cute.gemm(%mma, %d, %a, %b, %c) {
    } : (!cute.tiled_mma<!cute_nvgpu.sm100.mma<128x128x16,
                                               num_cta = 1,
                                               ab_major = (k,k),
                                               elem_type = (f16,f16,f16),
                                               frag_kind = ss,
                                               c_scale_exp = 1>,
                         atom_layout_MNK = <"(1,1,1)">>,
         !cute.memref<f16, tmem, align<16>, "((128,128),1,1):((131072,1),0,0)">,
         !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "S<3,4,3> o 0 o ((128,16),1,4):((64,1),0,16)">,
         !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "S<3,4,3> o 0 o ((128,16),1,4):((64,1),0,16)">,
         !cute.memref<f16, tmem, align<16>, "((128,128),1,1):((131072,1),0,0)">)
    ```

    Example (block-scaled GEMM):
    ```mlir
    cute.gemm(%mma, %d, [%a, %sfa], [%b, %sfb], %c) : (
      !cute.tiled_mma<!cute_nvgpu.sm100.mma_bs<256x128x64,
                                               num_cta = 2,
                                               ab_major = (k, k),
                                               elem_type = (f4E2M1FN, f4E2M1FN, f32),
                                               sf_type = f8E4M3FN,
                                               frag_kind = ss,
                                               vec_size = 16>,
                       atom_layout_MNK = <"(1,2,1):(0,1,0)">>,
      !cute.memref<f32, tmem, align<16>, "((128,128),1,1):((65536,1),0,0)">,
      [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">,
      !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4)),1):((((262144,4),8388608),(0,1)),0)">],
      [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1,1):(0,0)">,
      !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4)),1):((((262144,4),8388608),(0,1)),0)">],
      !cute.memref<f32, tmem, align<16>, "((128,128),1,1):((65536,1),0,0)">)
    ```
  }];
  let arguments = (ins
    Cute_MmaAtomTypeInterfaceInstance:$mma_atom,
    Cute_ViewTypeInterfaceInstance:$d,
    Variadic<Cute_ViewTypeInterfaceInstance>:$a,
    Variadic<Cute_ViewTypeInterfaceInstance>:$b,
    Cute_ViewTypeInterfaceInstance:$c
  );
  let results = (outs
  );
}
def Cute_AlgorithmPrefetchOp : Cute_Op<"prefetch"> {
  let summary = "CuTe's prefetch algorithm prefetches the elements of source Tensor(s) into cache.";
  let description = [{
    A Tensor encapsulates the data type, data location, and possibly
    also the shape and stride of the tensor at compile time. As a result, prefetch can
    and does dispatch, based on the types of its arguments, to use any of various
    synchronous or asynchronous hardware prefetch instructions.

    An .prefetch accepts CutePrefetchAtomTypeinterface as first parameter.

    The `src` operand is variadic:
    - For regular TMA prefetch, `src` is a single tensor.
    - For 2D `tile::gather4` mode, `src` is `[data_coord_tensor, gmem_index_tensor]`,
      mirroring `cute.copy` for the same mode.

    Example:

    ```mlir
    cute.prefetch(%tiled_copy, %src) : (
      !cute.tiled_copy<!cute_nvgpu.atom.tma_load<mode = tiled, num_cta = 1, mcast>,
                       layout_copy_tv = <"((4,2),(64,128)):((0,64),(128,1))">,
                       tiler_mn = <"[2048:1;1:1;4:1;1:1]">>,
      !cute.coord_tensor<"(0,0)", "(((64, 128), 1), 1, 1, 2):(((1@0, 1@1), 0), 0, 0, 64@0)">)

    // gather4 mode: data tensor + gmem index tensor
    cute.prefetch(%tiled_copy, %src_data, %src_idx) : (
      !cute.tiled_copy<!cute_nvgpu.atom.tma_load<f16, mode = gather4, num_cta = 1>, ...>,
      !cute.coord_tensor<...>,
      !cute.coord_tensor<...>)
    ```
  }];
  let arguments = (ins
    Cute_PrefetchAtomTypeInterfaceOrTiledCopyTypeWithPrefetchAtomTypeInterfaceCopyAtom:$prefetch_atom,
    Variadic<Cute_ViewTypeInterfaceInstance>:$src
  );
  let results = (outs
  );
}
def Cute_AppendToRankOp : Cute_InferredOp<"append_to_rank", [Pure]> {
  let summary = "Append element to input to rank N.";
  let description = [{
    Example:

    ```mlir
    %shp = cute.make_shape() : !cute.shape<"(5,4,3)">
    %elem = cute.make_shape() : !cute.shape<"(2,1)">
    %new_layout = cute.append_to_rank<5> (%shp, %elem)
      : (..., ...) -> !cute.shape<"(5,4,3,(2,1),(2,1))">
    ```
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstance:$input,
    Cute_ValueTypeInterfaceInstance:$element,
    SI32Attr:$rank
  );
  let results = (outs
    Cute_ValueTypeInterfaceInstance:$result
  );
}
def Cute_ApplySwizzleOp : Cute_InferredOp<"apply_swizzle", [Pure]> {
  let description = [{
    The `apply_swizzle` operation applies a swizzle transformation to a pointer's
    address, producing a new pointer with the swizzled address. This is necessary
    for dereferencing pointers that use swizzled memory layouts, such as those used
    for improved memory access patterns on GPUs.

    Two forms are supported:

    1. **CutePtrType form**: The swizzle is encoded in the pointer type itself.
       No explicit swizzle operand is required or allowed.

       ```mlir
       %swizzled = cute.apply_swizzle(%ptr) : !cute.ptr<f16, smem, align<16>, S<3,4,3>>
       ```

    2. **llvm.ptr form**: The swizzle is passed as an optional attribute in the
       attribute dictionary. If the swizzle attribute is absent, the operation
       is a no-op (identity swizzle).

       ```mlir
       // With explicit swizzle:
       %swizzled = cute.apply_swizzle(%ptr) { S<3,4,3> } : !llvm.ptr<3>
       // Identity (no-op):
       %swizzled = cute.apply_swizzle(%ptr) : !llvm.ptr<3>
       ```

    If the input pointer type has a non-identity swizzle attribute, this operation
    computes the swizzled address by applying the swizzle transformation to the
    pointer's address bits. The transformation is typically an XOR-based permutation
    of certain address bits, as defined by the swizzle parameters.

    Swizzle is parameterized by three values:
    - MBase: The number of least-significant bits to keep constant (not affected by the swizzle)
    - BBits: The number of bits in the mask (the width of the swizzle region)
    - SShift: The distance to shift the mask (positive values shift right, negative values shift left)

    The swizzle transformation operates on the address as follows:

        0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                      ^--^ MBase: least-significant bits kept constant
                         ^-^       ^-^     BBits: width of the mask (YYY and ZZZ)
                           ^---------^     SShift: shift distance for the YYY mask

    The swizzled address is computed by XOR'ing the ZZZ and YYY bitfields:

    Example:

    ```
        0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

        the result is
        0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
    ```

    This operation is typically used just before dereferencing a pointer,
    ensuring that the correct physical memory location is accessed according
    to the swizzle pattern. If the pointer type has an identity swizzle,
    this operation is a no-op and returns the original pointer.
  }];
  let arguments = (ins
    Cute_CuTePointerTypeOrLLVMPointerType:$ptr,
    OptionalAttr<Cute_SwizzleAttr>:$swizzle
  );
  let results = (outs
    Cute_CuTePointerTypeOrLLVMPointerType:$swizzled_ptr
  );
}
def Cute_AssumeOp : Cute_Op<"assume", [Pure]> {
  let description = [{
    `cute.assume` represents assumptions such as divisibility for an SSA value.
    unlike `llvm.assume`, `cute.assume` produces a strong typed value as output.
    It is limited to support all constrains defined by `ConstrainedIntType`.

    Please see [constrained integer type](### ConstrainedIntType) for details.

    Example:
    ```mlir
    // Claim %1 is divisible by 8
    %1 = cute.assume(%0 : i32) : !cute.i32<divby 8>

    ```

    Note:

    // xxx: need to revisit

    ```mlir
    %0 = ... -> !cute.i32
    if (%cond) {
      %1 = cute.assume(%0 : !cute.i32) : !cute.i32<divby 8>
      ... // %1 has the assumption.

      // %0 is not assumed here
      %data = load %ptr[%0] : ?
    }

    // %0 here does not have any assumption.
    ```
  }];
  let arguments = (ins
    Cute_Integer:$src
  );
  let results = (outs
    Cute_IntegerTypeWithConstraints:$dst
  );
}
def Cute_BlockedProductOp : Cute_InferredOp<"blocked_product", [Pure]> {
  let description = [{
    This operation performs "blocked_product" on the input and produce a blocked_product result,
    and it automatically infer the "blocked_product" layout for the result type.

    Example:

    ```mlir
    %input = cute.make_layout() : !cute.layout<"(3,4):(4,1)">
    %tiler = cute.make_layout() : !cute.layout<"(2,5):(1,2)">
    %bprod = cute.blocked_product(%input, %tiler)
      : (!cute.layout<"(3,4):(4,1)">, !cute.layout<"(2,5):(1,2)">)
      -> !cute.layout<"((3,2),(4,5)):((4,12),(1,24))">
    ```

    Example with dynamic composed_layout:

    ```
    %bprod = cute.blocked_product(%input, %tiler)
      : (!cute.composed_layout<"(?):(?) o ? o (5,4):(4,1)">, !cute.layout<"(?,?):(?,?)">)
      -> !cute.composed_layout<"(?):(?) o ? o ((5,?),(4,?)):((4,?{div=20}),(1,?{div=20}))">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuTeLayoutType:$tiler
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_CeilDivOp : Cute_InferredOp<"ceil_div", [Pure]> {
  let arguments = (ins
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS2:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$tiler
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS2:$result
  );
}
def Cute_CoalesceOp : Cute_InferredOp<"coalesce", [Pure]> {
  let summary = "`coalesce` op simplifies layout by combining modes that are possible to combine.";
  let description = [{
    It does not respect the shape of the layout but preserve total size.

    ```mlir
    // coalesce modes with contiguous/packed strides from left to right
    %0 = cute.coalesce(%arg0)
      : (!cute.layout<"((4,5),3):((1,4),20)">) -> !cute.layout<"60:1">

    // coalesce from left to right but not right to left
    %0 = cute.coalesce(%arg0)
      : (!cute.layout<"(3,(4,5)):(20,(1,4))">) -> !cute.layout<"(3,20):(20,1)">
    %0 = cute.coalesce(%arg0, %arg1)
      : (!cute.layout<"(3,(4,5)):(20,(1,4))">, !cute.coord<"(1,1)">) -> !cute.layout<"(3,20):(20,1)">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance:$input,
    Optional<Cute_CuteCoordType>:$target_profile
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance:$result
  );
}
def Cute_ComplementOp : Cute_InferredOp<"complement", [Pure]> {
  let summary = "Produces a disjoint layout.";
  let description = [{
    The complement of a layout L is a layout L* such that (L,L*) achieves greater surjectivity.
    That is, L* maps to elements in the codomain of L that are not in the image of L.

    It's worth to note that (L,L*) isn't alway surjective.

    Examples:
    ```
    // trivial complement of L under it's domain is 1:0
    %res = cute.complement(%arg0, %arg1)
      : (!cute.layout<"3:1">, !cute.shape<"3">) -> !cute.layout<"1:0">

    // 3:2 is strided layout. complement of 3:2 under 6 is 2:1 filling stride
    // (L,L*) => (3,2):(2,1) is a surjective layout
    %res = cute.complement(%arg0, %arg1)
      : (!cute.layout<"3:2">, !cute.shape<"6">) -> !cute.layout<"2:1">

    %res = cute.complement(%arg0, %arg1)
      : (!cute.layout<"(3,(5,2)):(1,(3,15))">, !cute.shape<"(6,20)">) -> !cute.layout<"4:30">
    ```
  }];
  let arguments = (ins
    Cute_CuTeLayoutType:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$cotarget
  );
  let results = (outs
    Cute_CuTeLayoutType:$res
  );
}
def Cute_ComposedGetInnerOp : Cute_InferredOp<"composed_get_inner", [Pure]> {
  let summary = "Gets inner layout from a layout.";
  let description = [{
    Example:

    ```mlir
    %layout = cute.make_layout(.., ..): !cute.layout<"(32,64):(1,32)>">
    %layout_a = cute.composed_get_layout_a(%layout) : (!cute.composed_layout<"(64,32):(1,64) o 0 o (32,64):(1,32)>">) -> !cute.layout<"(64,32):(1,64)">
    %offset = cute.composed_get_offset(%layout) : (!cute.composed_layout<"S<3,4,3> o 7 o (32,64):(1,32)>">) -> !cute.shape<"7">
    %layout_b = cute.composed_get_layout_b(%layout) : (!cute.composed_layout<"(64,32):(1,64) o 0 o (32,64):(1,32)>">) -> !cute.layout<"(32,64):(1,32)">
    ```
  }];
  let arguments = (ins
    Cute_CuTeComposedLayoutType:$input
  );
  let results = (outs
    Cute_CuTeLayoutTypeOrCuteSwizzleType:$result
  );
}
def Cute_ComposedGetOffsetOp : Cute_InferredOp<"composed_get_offset", [Pure]> {
  let summary = "Gets offset from a layout.";
  let description = [{
    Example:

    ```mlir
    %layout = cute.make_layout(.., ..): !cute.layout<"(32,64):(1,32)>">
    %layout_a = cute.composed_get_layout_a(%layout) : (!cute.composed_layout<"(64,32):(1,64) o 0 o (32,64):(1,32)>">) -> !cute.layout<"(64,32):(1,64)">
    %offset = cute.composed_get_offset(%layout) : (!cute.composed_layout<"S<3,4,3> o 7 o (32,64):(1,32)>">) -> !cute.shape<"7">
    %layout_b = cute.composed_get_layout_b(%layout) : (!cute.composed_layout<"(64,32):(1,64) o 0 o (32,64):(1,32)>">) -> !cute.layout<"(32,64):(1,32)">
    ```
  }];
  let arguments = (ins
    Cute_CuTeComposedLayoutType:$input
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$result
  );
}
def Cute_ComposedGetOuterOp : Cute_InferredOp<"composed_get_outer", [Pure]> {
  let summary = "Gets layout outer from composed layout.";
  let description = [{
    Example:

    ```mlir
    %layout_outer = cute.composed_get_outer(%layout)
      : (!cute.composed_layout<"(64,32):(1,64) o 0 o (32,64):(1,32)>">) -> !cute.layout<"(32,64):(1,32)">
    ```
  }];
  let arguments = (ins
    Cute_CuTeComposedLayoutType:$input
  );
  let results = (outs
    Cute_CuTeLayoutType:$result
  );
}
def Cute_CompositionOp : Cute_InferredOp<"composition", [Pure]> {
  let description = [{
    The `composition` operation computes the composition of two layouts
    (`!cute.layout`), in the following called `lhs` and `rhs`.

    Examples:
    ```
    // Composition of two static layouts.
    %0 = cute.composition(%arg0, %arg1)
        : (!cute.layout<"(20):(2)">, !cute.layout<"(5,4):(4,1)">) -> !cute.layout<"(5,4):(8,2)">

    // Composition of a static and a dynamic layout.
    %1 = cute.composition(%arg0, %arg1)
        : (!cute.layout<"(20):(2)">, !cute.layout<"(?,?):(?,?)">) -> !cute.layout<"(?,?):(?{div=2},?{div=2})">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$lhs,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$rhs
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$res
  );
}
def Cute_CopyAtomCallOp : Cute_Op<"copy_atom_call", [AttrSizedOperandSegments]> {
  let summary = "`cute.copy_atom_call` op executes a copy atom operation with the given operands.";
  let description = [{
    Following src/dst layout of atom are valid:
    * ((atom_v, rest_v))
    * ((atom_v))
    * (atom_v)

    Both `src` and `dst` operands are variadic, containing a variable number of tensors:

    - For regular copy, `src` and `dst` each contain a single tensor.
    - For copy with auxiliary operands, they contain the main tensor followed by
      auxiliary tensors. For example:

      - For load with row-wise reduction, `dst` = [data, stat].
      - For SPARSIFY (structured sparsity compression), `dst` = [data, metadata].
      - For TMA gather4, `src` = [coord0, coord1, coord2, coord3] (four 2D coordinate tensors).
      - For TMA scatter4, `dst` = [coord0, coord1, coord2, coord3] (four 2D coordinate tensors).

    Examples:

    ```mlir
    // Regular copy atom operation
    cute.copy_atom_call(%copy_load, %src_v, %dst_v)
      : (!cute_nvgpu.atom.tmem_load<f16, 32 DP, 32 bit, x1>,
         !cute.memref<f16, tmem, align<16>, "(((2,32),1)):(((1,131072),0))">,
         !cute.memref<f16, rmem, align<16>, "((2,1)):((1,0))">)
      -> ()
    ```

    ```mlir
    // TMA gather4: combine four 2D coordinate tensors into single destination
    cute.copy_atom_call(%tma_gather4_atom, [%coord0, %coord1, %coord2, %coord3], %dst_v)
      : (!cute_nvgpu.atom.tma_gather4<...>,
         [!cute.coord_tensor<...>, !cute.coord_tensor<...>,
          !cute.coord_tensor<...>, !cute.coord_tensor<...>],
         !cute.memref<f16, smem, ...>)
      -> ()
    ```
  }];
  let arguments = (ins
    Cute_CopyAtomTypeInterfaceInstanceOrCopyAtomTypeInterfaceV2Instance:$atom,
    Variadic<Cute_ViewTypeInterfaceInstance>:$src,
    Variadic<Cute_ViewTypeInterfaceInstance>:$dst,
    Optional<Cute_ViewTypeInterfaceInstance>:$pred
  );
  let results = (outs
  );
}
def Cute_CopyMakeFragmentOp : Cute_InferredOp<"copy.make_fragment", [Pure]> {
  let description = [{
    Construct a copy fragment for the S or D side from either a partitioned view or an abstract TV
    description. Intended for use between `cute.tiled.copy.partition_{S,D}` and `cute.copy`, but
    also supports layout/shape inputs for RMEM/TMEM fragment construction similar to MMA.

    The op accepts:
      1. an operand id `S` or `D` (printed as a keyword);
      2. a copy atom value;
      3. either a partitioned view or a layout/shape to build the fragment.

    Returns a new ViewTypeInterface derived value to be consumed by `cute.copy`.

    Examples:
    ```mlir
    // Load from global memory (gmem -> rmem): build S and D fragments from S-partition
    %atom_load = cute.make_atom() : () -> !cute_nvgpu.atom.universal_copy<f16, 128b, src_space=gmem, dst_space=rmem>
    %tiled_copy_load = cute.make_tiled_copy(%atom_load)
      : !cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f16, 128b, src_space=gmem, dst_space=rmem>,
                        layout_copy_tv = <"(128,8):(1,128)">, tiler_mn = <"[128;8]">>
    %tGSrc_load = cute.tiled.copy.partition_S(%tiled_copy_load, %gSrc, %tid_coord)
      : (!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f16, 128b, src_space=gmem, dst_space=rmem>,
                          layout_copy_tv = <"(128,8):(1,128)">, tiler_mn = <"[128;8]">>,
         !cute.memref<f16, gmem, align<128>, "(128,128):(?{div=8},1)">, !cute.coord<"?">)
      -> !cute.memref<f16, gmem, align<16>, "((8,1),1,16):((1,0),0,8)">

    %frgS_load = cute.copy.make_fragment S (%atom_load, %tGSrc_load) : (
        !cute_nvgpu.atom.universal_copy<f16, 128b, src_space=gmem, dst_space=rmem>,
        !cute.memref<f16, gmem, align<16>, "((8,1),1,16):((1,0),0,8)">
      ) -> !cute.memref<f16, gmem, align<16>, "((8,1),1,16):((1,0),0,8)">

    %rLoad = cute.copy.make_fragment D (%atom_load, %tGSrc_load) : (
        !cute_nvgpu.atom.universal_copy<f16, 128b, src_space=gmem, dst_space=rmem>,
        !cute.memref<f16, gmem, align<16>, "((8,1),1,16):((1,0),0,8)"
      ) -> !cute.memref<f16, rmem, align<16>, "((8,1),1,16):((1,0),0,8)"
    ```
  }];
  let arguments = (ins
    Cute_CopyAtomTypeInterfaceInstanceOrCopyAtomTypeInterfaceV2Instance:$atom,
    Cute_AMemoryReferenceTypeThatRepresentsATensor:$input,
    Cute_CopyOperandEnum:$operand_id
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_CosizeOp : Cute_InferredOp<"cosize", [Pure]> {
  let description = [{
    This operation calculates the cosize of a layout.
    * For a layout `l`, cosize(l) = l(size(l) - 1) + 1.
    * For a layout `l` with mode `mode`, cosize(l) = cosize(get<mode>(l)).
    * For a composed layout 'l = a o b', cosize(l) = cosize(b).

    The `mode` attribute specifies the nested mode in the type for cosize calculation.
    It indexes a sub-scope within the input type and returns the cosize of that sub-scope.
    The rank of `mode` must be less than or equal to the depth of the type.
    If `mode` is empty, the operation calculates the cosize of the entire type.

    ### Examples

    ```mlir
    %input: !cute.composed_layout<"(100):(1) o 0 o (3,4):(4,1)">
    %0 = cute.cosize(%input)
      : !cute.composed_layout<"(100):(1) o 0 o (3,4):(4,1)">
    // cosize(%input) = cosize((3,4):(4,1)), ret is 12
    ```

    Example with mode:
    ```mlir
    %input: !cute.layout<"((4,3),((3,(2,2)),2)):((1,4),((12,(36,72)),144))">
    %0 = cute.cosize(%input) <{mode=[1,0,1]}>
      : (!cute.layout<"((4,3),((3,(2,2)),2)):((1,4),((12,(36,72)),144))">) -> !cute.int_tuple<"109">
    // cosize(%input) = cosize((2,2):(36,72)), ret is 109
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$mode
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$result
  );
}
def Cute_Crd2IdxOp : Cute_InferredOp<"crd2idx", [Pure]> {
  let description = [{
    Compute the index of a coordinate with respect to the given layout. I.e.,
    compute the natural coordinate of `coord` wrt. to the shape
    and then compute the dot product with the strides.

    Examples:
    ```mlir
    // %crd0 is already a natural coordinate for %shape.
    %0 = cute.crd2idx(%crd, %layout): (!cute.coord<"(?,(?,?))">, !cute.layout<"(?,(?,?)):(?,(?,?))">) -> (i32)
    // Compute the natural coordinate, then dot product.
    %1 = cute.crd2idx(%crd, %layout): (!cute.coord<"(?,?)">, !cute.layout<"(?,(?,?)):(?,(?,?))">) -> (i32)
    // Static and dynamic shape/strides are supported.
    %3 = cute.crd2idx(%crd, %layout): (!cute.coord<"(?,?)">, !cute.layout<"(4,2):(1,2)">) -> (i32)
    // Strides with scaled basis produces a tuple.
    %5 = cute.crd2idx(%crd, %layout) : (!cute.coord<"(?,?,?)">, !cute.layout<"(10,20,30):(1@0,1@1,20@1)">) -> (!cute.int_tuple<"(?,?)">)
    ```
  }];
  let arguments = (ins
    Cute_CuteCoordType:$coord,
    Cute_GeneralizedLayoutTypeOfCuTe:$layout
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$index
  );
}
def Cute_DereferenceArithTupleIteratorOp : Cute_InferredOp<"deref_arith_tuple_iter", [Pure]> {
  let summary = "Dereferences an arithmetic tuple iterator to return the int_tuple value it holds.";
  let description = [{
    Example:

    ```mlir
    %0 = cute.deref_arith_tuple_iter(%arg0) : !cute.arith_tuple_iter<"(?,4)"> -> !cute.int_tuple<"(?,4)">
    ```
  }];
  let arguments = (ins
    Cute_CuTeIteratorTypeWithIntTupleAsIts:$iter
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$arith_tuple
  );
}
def Cute_DereferenceDescriptorIteratorOp : Cute_InferredOp<"deref_desc_iter", [Pure]> {
  let description = [{
    This Operation dereferences a descriptor iterator to return the descriptor
    value it holds.
  }];
  let arguments = (ins
    Cute_ATypeInterfaceForDescriptorIteratorTypes:$iter
  );
  let results = (outs
    AnyType:$value
  );
}
def Cute_DerefineOp : Cute_Op<"derefine", [Pure]> {
  let description = [{
    The `cute.derefine` operation discard static information from modes
    of a given cute algebra type like shape, stride, layout, coord or tile into
    modes with less static information.

    For each mode, it either cast it into dynamic mode or keep it unchanged. In future,
    it can be extended to support removal of `divby N` from dynamic mode.

    Casting dynamic mode into static mode is illegal.

    E.g.

    ```mlir
    %dyn_0 = cute.derefine(%0) : !cute.shape<"(3,5)">  to !cute.shape<"(?,5)">
    %dyn_1 = cute.derefine(%1) : !cute.stride<"(?,5)"> to !cute.stride<"(?,?)">
    %dyn_2 = cute.derefine(%2) : !cute.coord<"(8,_)">  to !cute.coord<"(?,_)">
    ```
  }];
  let arguments = (ins
    Cute_RefinableTypeInterfaceInstance:$src
  );
  let results = (outs
    Cute_RefinableTypeInterfaceInstanceOrInteger:$dst
  );
}
def Cute_DiceOp : Cute_InferredOp<"dice", [Pure]> {
  let description = [{
    This operation dices a type implementing the `CuteValueTypeInterface` (ShapeType, IntTupleType,
    LayoutType,...) using a static coordinate attribute. The profile of said coordinate must be
    weakly congruent with that of `input`. Modes associated with an underscore are discarded in the
    result.
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstance:$input,
    Cute_CoordAttr:$coord
  );
  let results = (outs
    Cute_ValueTypeInterfaceInstance:$result
  );
}
def Cute_ElemLessOp : Cute_Op<"elem_less", [Pure]> {
  let description = [{
    The cute.elem_less operation compares every elements in Lhs to the
    corresponding elements in Rhs. The result is 1 if every Lhs elements
    are less than corresponding Rhs elements, otherwise result is 0.

    `ElemLessOp` is useful when we want to know if a coordinate is out of bound:

    ```mlir
    %is_in_bound = cute.elem_less(%coord, %shape) : !cute.coord<"(?,?)">, !cute.shape<"(?,?)">
    ```
  }];
  let arguments = (ins
    Cute_CuteShapeTypeOrIntTupleAsIntegerOr:$lhs,
    Cute_CuteShapeTypeOrIntTupleAsIntegerOr:$rhs
  );
  let results = (outs
    I1:$result
  );
}
def Cute_EqualOp : Cute_Op<"equal", [Pure]> {
  let description = [{
    Checks two dynamic cute values for equality. Full static values are considered
    equal only when their types are the same.

    Example:

    ```mlir
    %1 = cute.make_shape : !cute.shape<"(32, 64)">
    %2 = cute.make_shape : !cute.shape<"(32, 64)">
    %result = cute.equal %1, %2 : !cute.shape<"(32,64)">, !cute.shape<"(32,64)">
    ```

    The generated code only checks profile positions in which at least one side has a dynamic value.
    Static comparisons are performed at compile time by the operation folder.

    ```mlir
    %1 = cute.make_shape : !cute.shape<"(?, 64)">
    %2 = cute.make_shape : !cute.shape<"(?, 64)">
    %result = cute.equal %1, %2 : !cute.shape<"(?,64)">, !cute.shape<"(?,64)">
    ```

    This would result in the first element being checked for equality in the runtime representation of the shapes.

    Note that this is dynamic if either of the operands will force that operand to be checked. So,

    ```mlir
    %1 = cute.make_shape : !cute.shape<"(?, 64)">
    %2 = cute.make_shape : !cute.shape<"(64, ?)">
    %result = cute.equal %1, %2 : !cute.shape<"(?,64)">, !cute.shape<"(64,?)">
    ```

    will result in both dimensions being checked.
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstance:$lhs,
    Cute_ValueTypeInterfaceInstance:$rhs
  );
  let results = (outs
    I1:$result
  );
}
def Cute_FastDivmodComputeOp : Cute_Op<"fast_divmod.compute", [Pure]> {
  let description = [{
    This operation performs a fast divmod using a fast divmod divisor. It computes the quotient and
    the remainder without emitting division instructions. The quotient and the remainder are of the
    same width as the divisor.
  }];
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$dividend,
    Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant:$divisor
  );
  let results = (outs
    Cute_32BitSignlessIntegerOr64BitSignless:$quotient,
    Cute_32BitSignlessIntegerOr64BitSignless:$remainder
  );
}
def Cute_FastDivmodCreateDivisorOp : Cute_Op<"fast_divmod.create_divisor", [Pure]> {
  let description = [{
    This operation creates a fast divmod divisor from an integer. It is meant to be pre-computed,
    ideally in host code, and then used throughout the lifetime of a device kernel for performing
    divmods without emitting division instructions.
  }];
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$divisor
  );
  let results = (outs
    Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant:$fast_divmod_divisor
  );
}
def Cute_FastDivmodDivideOp : Cute_Op<"fast_divmod.divide", [Pure]> {
  let description = [{
    This operation performs a fast divide using a fast divmod divisor. It computes the quotient
    without emitting division instructions. The quotient is of the same width as the divisor.
  }];
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$dividend,
    Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant:$divisor
  );
  let results = (outs
    Cute_32BitSignlessIntegerOr64BitSignless:$quotient
  );
}
def Cute_FastDivmodGetAuxOp : Cute_InferredOp<"fast_divmod.get_aux", [Pure]> {
  let arguments = (ins
    Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant:$divisor
  );
  let results = (outs
    Cute_32BitSignlessIntegerOr64BitSignless:$multiplier,
    I8:$shift1,
    I8:$shift2
  );
}
def Cute_FastDivmodGetDivisorOp : Cute_InferredOp<"fast_divmod.get_divisor", [Pure]> {
  let arguments = (ins
    Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant:$fast_divmod_divisor
  );
  let results = (outs
    Cute_32BitSignlessIntegerOr64BitSignless:$divisor
  );
}
def Cute_FastDivmodMakeDivisorOp : Cute_Op<"fast_divmod.make_divisor", [Pure]> {
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$divisor,
    Cute_32BitSignlessIntegerOr64BitSignless:$multiplier,
    I8:$sh1,
    I8:$sh2
  );
  let results = (outs
    Cute_DivisorForFastUnsignedIntegerDivmodsByInvariant:$fast_divmod_divisor
  );
}
def Cute_FilterOp : Cute_InferredOp<"filter", [Pure]> {
  let summary = "This operation filters out dimensions with 0-strides and 1-sizes from a layout or view.";
  let description = [{
    When working with tensor layouts, some dimensions may have stride 0 (indicating
    broadcasting) or size 1 (singleton dimensions). This operation creates a new layout
    or view with these dimensions removed, which can simplify subsequent operations.

    OptionalOperand `target_profile` is the user indicated filter control, with this operand,
    the original layout's stride is ignored in the zero-stride selection.
    But the `target_profile` will not take effect currently. This is a placeholder for future use.

    This operation accepts both layout and view types as valid input.
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance:$input
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance:$result
  );
}
def Cute_FilterZerosOp : Cute_InferredOp<"filter_zeros", [Pure]> {
  let description = [{
    This operation produces a transformed layout or view by replacing the shape
    of any mode with stride zero and non-unit extent with 1. This is commonly
    applied prior to copy operations to prevent redundant or duplicated
    data movement arising from broadcasted dimensions.

    The optional operand `target_profile` allows the user to specify a stride
    filter profile; when provided, the selection of zero-stride dimensions is
    determined by this stride profile rather than the original layout's stride
    information.

    This operation accepts both layout and view types as input.
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance:$input,
    Optional<Cute_CuteStrideType>:$target_profile
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTeOrViewTypeInterfaceInstance:$res
  );
}
def Cute_FlatDivideOp : Cute_InferredOp<"flat_divide", [Pure]> {
  let summary = "This operation performs a flat_divide on a view or a layout by a tiler.";
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$tiler
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_FlatProductOp : Cute_InferredOp<"flat_product", [Pure]> {
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuTeLayoutType:$tiler
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_GetFlatCoordOp : Cute_InferredOp<"get_flat_coord", [Pure]> {
  let description = [{
    This OP accept a scalar index and a normal layout,
    returns a flat coord.


    The "flat" here means the reversed coordinate will be depth=1 coordinate.
    This is useful when we just care about the rank semantic of the layout and don't care about the detail profile.


    Example:

    ```mlir
    %layout = cute.make_layout() : !cute.layout<"(5,(4,3),2):(1,(5,20),60)">
    %index = arith.constant 117 : i32
    %flat_coord = cute.get_flat_coord(%index, %layout) : (..., ...) -> !cute.coord<"(2,11,1)">
    %hier_coord = cute.get_hire_coord(%index, %layout) : (..., ...) -> !cute.coord<"(2,(3,2),1)">
    %integral_coord = cute.get_integral_coord(%index, %layout) : (..., ...) -> !cute.coord<"117">
    ```
  }];
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$index,
    Cute_CuTeLayoutType:$input
  );
  let results = (outs
    Cute_CuteCoordType:$result
  );
}
def Cute_GetHierCoordOp : Cute_InferredOp<"get_hier_coord", [Pure]> {
  let description = [{
    This OP accept a scalar index and a normal layout,
    returns a hier coord.


    The "hier" here means the reversed coordinate will have the same profile with the given layout.
    This is useful when we care about both the rank semantic and the detail layout profile.


    Example:

    ```mlir
    %layout = cute.make_layout() : !cute.layout<"(5,(4,3),2):(1,(5,20),60)">
    %index = arith.constant 117 : i32
    %flat_coord = cute.get_flat_coord(%index, %layout) : (..., ...) -> !cute.coord<"(2,11,1)">
    %hier_coord = cute.get_hire_coord(%index, %layout) : (..., ...) -> !cute.coord<"(2,(3,2),1)">
    %integral_coord = cute.get_integral_coord(%index, %layout) : (..., ...) -> !cute.coord<"117">
    ```
  }];
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$index,
    Cute_CuTeLayoutType:$input
  );
  let results = (outs
    Cute_CuteCoordType:$result
  );
}
def Cute_GetIntegralCoordOp : Cute_InferredOp<"get_integral_coord", [Pure]> {
  let description = [{
    This OP accept a scalar index and a normal layout,
    returns a integral coord.


    The "integral" here means the reversed coordinate will be a "scalar".
    This is useful when we don't care about the layout's semantic and profile and we just want to know the offset in "domain".


    Example:

    ```mlir
    %layout = cute.make_layout() : !cute.layout<"(5,(4,3),2):(1,(5,20),60)">
    %index = arith.constant 117 : i32
    %flat_coord = cute.get_flat_coord(%index, %layout) : (..., ...) -> !cute.coord<"(2,11,1)">
    %hier_coord = cute.get_hire_coord(%index, %layout) : (..., ...) -> !cute.coord<"(2,(3,2),1)">
    %integral_coord = cute.get_integral_coord(%index, %layout) : (..., ...) -> !cute.coord<"117">
    ```
  }];
  let arguments = (ins
    Cute_32BitSignlessIntegerOr64BitSignless:$index,
    Cute_CuTeLayoutType:$input
  );
  let results = (outs
    Cute_CuteCoordType:$result
  );
}
def Cute_GetIterOp : Cute_InferredOp<"get_iter", [Pure]> {
  let description = [{
    The "cute.get_iter" gets the underlying iterator from memref or coord_tensor.
    It is solely as step during lowering, it has no side effects.

    Example:

    ```mlir
      %ptr = cute.get_iter(%0) : !cute.memref<f16, smem, align<128>, "((64,128),2):((1,1),64)">

      // Or for TMA tensor

      %arith_tuple = cute.get_iter(%0) : !cute.coord_tensor<"(128, 64)", "((64,64),1):((1@0,1@0),64@1)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstance:$source
  );
  let results = (outs
    Cute_GeneralizedIteratorTypeOfCuTe:$ptr
  );
}
def Cute_GetLayoutOp : Cute_InferredOp<"get_layout", [Pure]> {
  let summary = "Gets the layout out of a `CuteViewTypeInterface`.";
  let description = [{
    Example:

    ```mlir
    %0 = cute.get_layout(%arg0) : !cute.memref<f16, smem, "(256,?):(?,1)">

    %0 = cute.get_layout(%arg0) : !cute.memref<f16, smem, "S<3,4,3> o 0 o (256,?):(?,1)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstance:$input
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$layout
  );
}
def Cute_GetLayoutsFromTileOp : Cute_InferredOp<"get_layouts_from_tile", [Pure]> {
  let description = [{
    This operation extracts the layout elements out of a tile. A tile is a
    layout, an underscore, or a tuple of tiles.

    Examples:

    - the type of the tile is !cute.tile<"[1:0;2:1]">
      2 layouts are extracted of type !cute.layout<"1:0"> and !cute.layout<"2:1">, respectively
    - the type of the tile is !cute.tile<[_;[1:0;2:1];3:1]>
      3 layouts are extracted of type !cute.layout<"1:0">, !cute.layout<"2:1">, and
      !cute.layout<"3:1">, respectively
  }];
  let arguments = (ins
    Cute_CuteTileType:$tile
  );
  let results = (outs
    Variadic<Cute_CuTeLayoutType>:$layouts
  );
}
def Cute_GetLeavesOp : Cute_InferredOp<"get_leaves", [Pure]> {
  let description = [{
    The `cute.get_leaves` operation extracts all leaf values from a hierarchical tuple into
    a flat sequence of results. Each result corresponds to a leaf value found in the input tuple,
    preserving the order of a depth-first traversal.

    The operation handles nested tuples of any depth, including tuples containing other tuples.
    The type of each result matches the type of the corresponding leaf value in the input.

    Example:
    ```mlir
    // Extract leaves from a hierarchical shape tuple
    %0:3 = cute.get_leaves(%hier) : !cute.shape<"(3,?{div=4},5)">
        -> (!cute.shape<"3">, !cute.shape<"?{div=4}">, !cute.shape<"5">)

    // Extract leaves from a nested int tuple
    %1:4 = cute.get_leaves(%t) : !cute.int_tuple<"((1,2),(3,4))">
        -> (!cute.int_tuple<"1">, !cute.int_tuple<"2">,
            !cute.int_tuple<"3">, !cute.int_tuple<"4">)
    ```
  }];
  let arguments = (ins
    Cute_XTupleLikeType:$input
  );
  let results = (outs
    Variadic<Cute_XTupleLikeTypeOrCuTeLayoutType>:$results_
  );
}
def Cute_GetOp : Cute_Op<"get", [Pure]> {
  let description = [{
    This operation returns a potentially nested mode out of a type implementing
    the `CuteValueTypeInterface`, e.g. a shape, a tuple, a layout,...

    The optional `mode` attribute indices are used to select the portion of the
    input type. The number of indices in the `mode` attribute should be less
    than or equal to the depth of type. If `mode` is empty, a copy of the input
    value is returned.

    Example:

    ```mlir
    %shape = cute.make_shape() : () -> !cute.shape<"(1, (2, 4), 8)">
    %mode0 = cute.get(%shape) <{mode=[1]}>
      : !cute.shape<"(1,(2,4),8)"> -> !cute.shape<"(2,4)">
    %mode1 = cute.get(%shape) <{mode=[1,1]}>
      : !cute.shape<"(1,(2,4),8)"> -> !cute.shape<"4">
    ```
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstance:$input,
    OptionalAttr<DenseI32ArrayAttr>:$mode
  );
  let results = (outs
    Cute_ValueTypeInterfaceInstanceOrIntegerTypeWithConstraintsOr:$result
  );
}
def Cute_GetScalarsOp : Cute_InferredOp<"get_scalars", [Pure]> {
  let summary = "Gets the scalars out of a value with CuteValueTypeInterface.";
  let description = [{
    This operation returns only the dynamic scalars when `only_dynamic` is set.

    A "scalar" in this context refers to a single value used as part of the
    state of a type. In general, each state value held by a type will be
    considered a scalar if it is printed in the text format. This includes:
    - static integers
    - dynamic integers
    - swizzle parameters
    - scaled basis value (static integer, ratio (num + denom), dynamic integer)
    - scaled basis modes
    - indexed pointer address (always a dynamic value)
    - indexed pointer stride (dynamic integer or dynamic)

    Note that underscore values in coordinate types are NOT considered scalars.

    Example:

    ```mlir
    %1 = cute.get_shape(%0) : (!cute.layout<"(256,?):(?,1)">) -> !cute.shape<"(256,?)">
    %2 = cute.get_scalars(%1) <{only_dynamic}> : (!cute.shape<"(256,?)">) -> i32
    ```
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstanceOrIntegerTypeWithConstraints:$cute_value,
    UnitAttr:$only_dynamic
  );
  let results = (outs
    Variadic<Cute_32BitSignlessIntegerOr64BitSignless>:$scalars
  );
}
def Cute_GetShapeOp : Cute_InferredOp<"get_shape", [Pure]> {
  let summary = "get shape from view, layout or tile";
  let description = [{
    Example:

    ```mlir
      %shape = cute.get_shape(%layout) : (!cute.layout<"(32,64):(1,32)>">) -> !cute.shape<"(32,64)">
      %shape = cute.get_shape(%tile)
        : (!cute.tile<"[(32,64):(1,32);3:1]>">) -> !cute.shape<"((32,64),3)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe2:$input
  );
  let results = (outs
    Cute_CuteShapeType:$result
  );
}
def Cute_GetStrideOp : Cute_InferredOp<"get_stride", [Pure]> {
  let summary = "get stride from layout.";
  let description = [{
    Please be noted that stride is not generalized for view and layout because composed layout
    doesn't have generalized layout concept.

    Example:

    ```mlir
      %stride = cute.get_stride(%layout) : (!cute.layout<"(32,64):(1,32)>">) -> !cute.stride<"(1, 32)">
    ```
  }];
  let arguments = (ins
    Cute_CuTeLayoutType:$input
  );
  let results = (outs
    Cute_CuteStrideType:$result
  );
}
def Cute_GroupModesOp : Cute_InferredOp<"group_modes", [Pure]> {
  let description = [{
    This operation will make the input view/layout's [B,E) modes
    into one new mode with one-depth deeper.

    Example:

    ```mlir
    %res = cute.group_modes(%arg0) <0,1>
      : (!cute.coord_tensor<"(0,0)", "(512,64,2):(1@1,1@0,64@0)">)
      -> !cute.coord_tensor<"(0,0)", "((512),64,2):((1@1),1@0,64@0)">
    %res = cute.group_modes(%arg0) <0,2>
      : (!cute.composed_layout<"(3,4,5):(6,7,8) o (0,1,2) o (9,9,9):(1@1,1@0,1@2)">)
      -> !cute.composed_layout<"(3,4,5):(6,7,8) o (0,1,2) o ((9,9),9):((1@1,1@0),1@2)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    I32Attr:$begin,
    I32Attr:$end
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_Idx2CrdOp : Cute_InferredOp<"idx2crd", [Pure]> {
  let description = [{
    compute the coordinate corresponding to an index with respect to the given shape. I.e.,
    compute the coordinate that maps to the given index using given shape and default stride.
    Note:  This only works for compact layout.

    Examples:
    ```mlir
      // Convert a linear index to coordinate using shape and default stride.
      %0 = cute.idx2crd(%index, %shape): (!cute.int_tuple<"?">, !cute.shape<"(4,2)">) -> (!cute.coord<"(?,?)">)
      // Static and dynamic shape are supported.
      %1 = cute.idx2crd(%index, %shape): (!cute.int_tuple<"?">, !cute.shape<"(?,(?,?))">) -> (!cute.coord<"(?,(?,?))">)
    ```
  }];
  let arguments = (ins
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$index,
    Cute_CuteShapeType:$shape
  );
  let results = (outs
    Cute_CuteCoordType:$coord
  );
}
def Cute_IncrementCoordOp : Cute_InferredOp<"increment_coord", [Pure]> {
  let summary = "This operation colexicographically increments a coord within a coord space defined by a shape.";
  let description = [{
    Example:

    ```mlir
    // With static coord and shape
    %coord1 = cute.increment_coord(%coord0, %shape) :
    (!cute.coord<"(0,0)">, !cute.shape<"(2,2)">) -> !cute.coord<"(1,0)">
    %coord2 = cute.increment_coord(%coord1, %shape) :
    (!cute.coord<"(1,0)">, !cute.shape<"(2,2)">) -> !cute.coord<"(0,1)">

    // With dynamic coord or shape
    %coord1 = cute.increment_coord(%coord0, %shape) :
    (!cute.coord<"(?,?)">, !cute.shape<"(?,8)">) -> !cute.coord<"(?,?)">
    ```
  }];
  let arguments = (ins
    Cute_CuteCoordType:$coord,
    Cute_CuteShapeType:$shape
  );
  let results = (outs
    Cute_CuteCoordType:$result
  );
}
def Cute_IntToPtrOp : Cute_Op<"inttoptr", [Pure]> {
  let description = [{
    The `cute.inttoptr` operation accepts an integer value and cast to cute
    pointer value with align=1 (byte) by default.

    To create ptr with align > 1, we must use constrained int type with divby > 1.
    Alignment of result ptr is the same as `divby` of constrained int type

    Example:

    ```mlir
    // Convert to cute.ptr with unrestricted integer type. Can only have align<1>
    %ptr = cute.inttoptr(%0) : i64 to !cute.ptr<f16, smem, align<1>>

    // Convert to cute.ptr with constrained integer type.
    %aligned_ptr = cute.inttoptr(%0) : !cute.i64<divby 8> to !cute.ptr<f16, smem, align<8>>
    ```
  }];
  let arguments = (ins
    Cute_IntegerOrIntegerTypeWithConstraints:$src
  );
  let results = (outs
    Cute_PointerTypeInterfaceInstance:$dst
  );
}
def Cute_KernelSmemSizeOp : Cute_Op<"kernel_smem_size", [Pure]> {
  let description = [{
    This host-side operation gets the static smem size from a kernel function 
    and returns the size as an i64 integer.
  }];
  let arguments = (ins
    SymbolRefAttr:$kernel_name
  );
  let results = (outs
    I64:$size
  );
}
def Cute_LeftInverseOp : Cute_InferredOp<"left_inverse", [Pure]> {
  let arguments = (ins
    Cute_CuTeLayoutType:$input
  );
  let results = (outs
    Cute_CuTeLayoutType:$result
  );
}
def Cute_LoadScaledIndexOp : Cute_InferredOp<"load_scaled_index", [MemoryEffects<[MemRead]>]> {
  let description = [{
    The `cute.load_scaled_index` operation adds the offset (%ptr_b[%index] * %stride) to a
    cute pointer %ptr_a. %ptr_a can be any cute pointer type, while %ptr_b must be an I32 cute pointer.
    The behavior is undefined if %ptr_b[%index] results in an out-of-bound access.
    The result is a cute pointer that has identical type as %ptr_a except that its alignment is derived
    from ptr_a's alignment and the stride.

    Example:

    ```mlir
      %ptr = cute.load_scaled_index(%src_ptr, %table_ptr, %index, %stride)
        : (!cute.ptr<f16, gmem>, !cute.ptr<i32, gmem>, !cute.int_tuple<"?">, !cute.int_tuple<"?{div=16}">)
        -> !cute.ptr<f16, gmem>
    ```
  }];
  let arguments = (ins
    Cute_CuTePointerType:$ptr_a,
    Cute_CuTePointerTo32BitSignlessInteger:$ptr_b,
    Cute_IntTupleTypeThatDepthIs0:$index,
    Cute_IntTupleTypeThatDepthIs0:$stride
  );
  let results = (outs
    Cute_CuTePointerType:$result
  );
}
def Cute_LocalPartitionOp : Cute_InferredOp<"local_partition", [Pure]> {
  let description = [{
    The `local_partition` operation tiles a tensor according to the flat shape of a layout
    that provides the coordinate of the target index.
    This is typical at the Thread level where data is partitioned across repeated patterns
    of threads.

    Example:

    ```mlir
    %local_partition = cute.local_partition(%input, %tile, %index)
      : (!cute.memref<f32, smem, align<32>, "(16,?):(1,16)">, !cute.layout<"(2,4)">, i32)
      -> !cute.memref<f32, smem, "(8,?):(2,64)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstance:$input,
    Cute_StaticLayout:$tiler,
    Cute_32BitSignlessIntegerOr64BitSignless:$index
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_LocalTileOp : Cute_InferredOp<"local_tile", [Pure]> {
  let description = [{
    The `local_tile` operation tiles a tensor according to a tiler and use a coord to
    index into the remainder, keeping the tile.

    This is typical at the CTA level where tiles of data are extracted:

    ```mlir
    %data =                                                                   // (  M,  N)
    %crd = cute.make_coord(blockIdx.x,blockIdx.y) : !cute.coord<"(?,?)">
    %cta_data = cute.local_tile(%data, %tile, %crd) : ...                    // (_32,_64)
    ```

    Example:

    ```mlir
    %0 = cute.local_tile(%alloc, %tile, %coord)
      : (!cute.memref<f32, tmem, align<32>, "(128,128):(65536,1)">,
         !cute.tile<"[128:1;32:1]">,
         !cute.coord<"(_,_)">)
      -> !cute.memref<f32, tmem, align<32>, "(128,32,1,4):(65536,1,0,32)">

    %1 = cute.local_tile(%arg0, %shape, %coord_1)
      : (!cute.coord_tensor<"(0,0,0)", "(256,64,1):(1@1,1@0,1@2)">,
         !cute.shape<"(256,64)">,
         !cute.coord<"(0,_,_)">)
      -> !cute.coord_tensor<"(0,0,0)", "(256,64,1,1):(1@1,1@0,0,1@2)">
    ```

    Utilizing a flat coordinate '(_)' will flatten the resulting tile.

    ```mlir
    %0 = cute.local_tile(%alloc, %tile, %coord)
      : (!cute.memref<f32, tmem, align<32>, "(128,128):(65536,1)">,
         !cute.tile<"[128:1;32:1]">,
         !cute.coord<"(_)">)
      -> !cute.memref<f32, tmem, align<32>, "(128,32,1,4):(65536,1,0,32)">

    %1 = cute.local_tile(%arg0, %shape, %coord_1)
      : (!cute.coord_tensor<"(0,0,0)", "(256,64,1):(1@1,1@0,1@2)">,
         !cute.shape<"(256,64)">,
         !cute.coord<"(_)">)
      -> !cute.coord_tensor<"(0,0,0)", "(256,64,1,1,1):(1@1,1@0,0,0,1@2)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstance:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$tile,
    Cute_CuteCoordType:$coord,
    OptionalAttr<Cute_CoordAttr>:$proj
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_LogicalDivideOp : Cute_InferredOp<"logical_divide", [Pure]> {
  let description = [{
    This operation performs "logical_divide" on the input and produces a logical_divided result,
    and it automatically infers the "logical_divided" layout for the result type.

    Example:

    ```mlir
    %alloc = cute.memref.alloc() : !cute.memref<f16, generic, "(12,32,6):(1,128,0)">
    %shape = cute.make_shape() : () -> !cute.shape<"(4,8)">
    %zdiv = cute.logical_divide(%alloc, %shape)
      : (!cute.memref<f16, generic, "(12,32,6):(1,128,0)">, !cute.shape<"(4,8)">)
      -> !cute.memref<f16, generic, "((4,3),(8,4),6):((1,4),(128,1024),0)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$tiler
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_LogicalProductOp : Cute_InferredOp<"logical_product", [Pure]> {
  let description = [{
    This operation performs "logical_product" on the input and produce a logical_product result,
    and it automatically infer the "logical_product" layout for the result type.

    Example:

    ```mlir
    %input = cute.make_layout() : !cute.layout<"(3,4):(4,1)">
    %tiler = cute.make_layout() : !cute.layout<"(2,5):(1,2)">
    %prod = cute.logical_product(%input, %tiler)
      : (!cute.layout<"(3,4):(4,1)">, !cute.layout<"(2,5):(1,2)">)
      -> !cute.layout<"((3,4),(2,5)):((4,1),(12,24))">
    ```

    Example with dynamic composed_layout:

    ```
    %prod = cute.logical_product(%input, %tiler)
      : (!cute.composed_layout<"(?):(?) o ? o (5,4):(4,1)">,
         !cute.layout<"(?,?):(?,?)">)
      -> !cute.composed_layout<"(?):(?) o ? o ((5,4),(?,?)):((4,1),(?{div=20},?{div=20}))">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuTeLayoutType:$tiler
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_MakeArithTupleIteratorOp : Cute_Op<"make_arith_tuple_iter", [Pure]> {
  let description = [{
    This Operation makes an arithmetic tuple iterator (`!cute.arith_tuple_iter<...>`)
    from an int_tuple value (`!cute.int_tuple<...>`) if provided, or it is empty
    when all the modes are static.

    Its verifier checks that the value type of the result iterator matches the
    operand value type if provided.
  }];
  let arguments = (ins
    Optional<Cute_IntTupleAsIntegerOrTupleOfIntTupleS>:$value
  );
  let results = (outs
    Cute_CuTeIteratorTypeWithIntTupleAsIts:$iter
  );
}
def Cute_MakeAtomOp : Cute_Op<"make_atom", [Pure]> {
  let summary = "`cute.make_atom` op is generic constructor of SSA value of copy or mma atom type.";
  let description = [{
    It optionally takes variadic list of SSA values as well as named attribute to
    initialize corresponding fields of copy or mma atom.

    Examples:

    ```mlir
    // create copy atom which doesn't require initialization
    %0 = cute.make_atom() : ()
      -> !cute_nvgpu.atom.tmem_load<f32, 16 DP, 256 bit, x1>

    // create mma atom and initialize it with SSA values
    %1 = cute.make_atom(%accum_c, %a_neg, %b_neg) : (i1, i1, i1)
      -> !cute_nvgpu.sm100.mma<128x128x16,
                              num_cta = 1,
                              ab_major = (k,k),
                              elem_type = (f16,f16,f32),
                              frag_kind = ss,
                              c_scale_exp = 0>
    ```
  }];
  let arguments = (ins
    Variadic<AnyType>:$values
  );
  let results = (outs
    Cute_AtomTypeInterfaceInstance:$atom
  );
}
def Cute_MakeComposedLayoutOp : Cute_InferredOp<"make_composed_layout", [Pure]> {
  let summary = "The `make_composed_layout` operation constructs composed layout with";
  let description = [{
      * (inner) affined layout or swizzle layout
      * offset
      * (outer) layout

    In CuTe algebra, composed layout represent composition of

    ```math
      L_{inner} o offset o L_{outer}
    ```

    where outer layout defined the domain of the layout which is visible to user. For given input coordinate
    in domain of outer layout, composed layout maps coordinate to codomain by

    ```math
      L_{inner}(L_{outer}(coordinate) + offset)
    ```

    Composed Layout is useful in cases where composition() does not or cannot apply to LayoutA and LayoutB.
    For example, when the "divisibility condition" in shape_div is violated in composition(LayoutA, LayoutB).

    This ComposedLayout provides similar functionality to Layout including tiling, partitioning,
    coordinate-to-index mapping and layout manipulations, but is not considered a "normal" layout.
    For example, this layout provides shape() and size() functions, but does not provide stride() functions.
    Mostly, the similar functionality is accomplished by applying each operation to LayoutB only
    as LayoutB defines the domain.

    ### Examples:

    ```mlir
    // Swizzle composed layout
    %inner : !cute.swizzle<"S<3,4,3>">
    %layout = cute.make_composed_layout(%inner, %offset, %outer)
      : !cute.composed_layout<"S<3,4,3> o 5 o (?,?):(?,?)">

    // Linearized composed layout
    %inner : !cute.layout<"((?,?),(?,?)):((?,?),(?,?))">
    %layout = cute.make_composed_layout(%inner, %offset, %outer)
      : !cute.composed_layout<"((?,?),(?,?)):((?,?),(?,?)) o ? o (1,2):(3,4)">
    ```
  }];
  let arguments = (ins
    Cute_CuTeLayoutTypeOrSwizzleWithValidParameters:$inner,
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$offset,
    Cute_CuTeLayoutType:$outer
  );
  let results = (outs
    Cute_CuTeComposedLayoutType:$result
  );
}
def Cute_MakeCoordOp : Cute_Op<"make_coord", [Pure]> {
  let description = [{
    Constructs a coord. The operands correspond to the
    coord's dynamic elements, ordered from left to right.
    The number of operands should match the count of dynamic elements in the
    resulting coord type.

    Example:

    ```mlir
    // Definitions for %0, %1 come earlier
    %shape = cute.make_coord(%0, %1) : (i32, i32)
        -> !cute.coord<"((?,32),?)">
    ```
  }];
  let arguments = (ins
    Variadic<Cute_I32I64OrConstrainedInteger>:$dynamicElements
  );
  let results = (outs
    Cute_CuteCoordType:$result
  );
}
def Cute_MakeDescriptorIteratorOp : Cute_Op<"make_desc_iter", [Pure]> {
  let description = [{
    This Operation makes a descriptor iterator from a descriptor value. Its
    verifier verifies that the value Type of the result Type implementing the
    iterator Type Interface is the same as the Type as the input operand.
  }];
  let arguments = (ins
    Cute_ATypeInterfaceForMemoryDescriptorsSuchAs:$value
  );
  let results = (outs
    AnyType:$iter
  );
}
def Cute_MakeFragmentLikeOp : Cute_InferredOp<"make_fragment_like", [Pure]> {
  let summary = "A deprecated operation but used by collective DSL (keep it here to avoid breaking changes)";
  let description = [{
    TODO: remove this operation in the future
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$src
  );
  let results = (outs
    Cute_CuTeLayoutType:$dst
  );
}
def Cute_MakeIdentityLayoutOp : Cute_InferredOp<"make_identity_layout", [Pure]> {
  let summary = "This operation makes an identity layout with the given shape.";
  let description = [{
    Example:
    ```mlir
    %sha = cute.make_shape() : () -> !cute.shape<"(2,(3,4),5)">
    %lay = cute.make_identity_layout(%sha) : !cute.layout<"(2,(3,4),5):(1@0,(1@0@1,1@1@1),1@2)">
    ```
  }];
  let arguments = (ins
    Cute_CuteShapeType:$shape
  );
  let results = (outs
    Cute_CuTeLayoutType:$result
  );
}
def Cute_MakeIdentityTensorOp : Cute_InferredOp<"make_identity_tensor", [Pure]> {
  let description = [{
    The cute.make_identity_tensor operation construct an identity_tensor
    based on the given shape.

    Example to make an identity_tensor from a static shape:

    ```mlir
    %identity_tensor = cute.make_identity_tensor(%shape)
      : !cute.coord_tensor<"(0,(0,0))", "(2,(3,4)):(1@0,(1@0@1,1@1@1))">
    ```

    Example to make an identity_tensor from a dynamic shape:

    ```mlir
    %identity_tensor = cute.make_identity_tensor(%shape)
      : !cute.coord_tensor<"((0,0),(0,0))", "((?,?),(?,?)):((1@0@0,1@1@0),(1@0@1,1@1@1))">
    ```
  }];
  let arguments = (ins
    Cute_CuteShapeType:$shape
  );
  let results = (outs
    Cute_CuteCoordTensorType:$result
  );
}
def Cute_MakeIntTupleOp : Cute_Op<"make_int_tuple", [Pure]> {
  let description = [{
    Constructs a int_tuple. The operands correspond to the
    int_tuple's dynamic elements, ordered from left to right.
    The number of operands should match the count of dynamic elements in the
    resulting int_tuple type.

    Example:

    ```mlir
    // Definitions for %0, %1 come earlier
    %shape = cute.make_int_tuple(%0, %1) : (i32, i32)
        -> !cute.int_tuple<"((?,32),?)">
    ```
  }];
  let arguments = (ins
    Variadic<Cute_I32I64OrConstrainedInteger>:$dynamicElements
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$result
  );
}
def Cute_MakeLayoutLikeOp : Cute_InferredOp<"make_layout_like", [Pure]> {
  let description = [{
    Creates a compact memory layout that:
    1. Has a congruent shape with the source layout
    2. Has strides in the same order of major-ness as the input layout
    3. Mode 0 strides are `LayoutLeft` (column major) with zero strides preserved from the input layout

    Mode-0 constraints are applied due to its common usage with MMA_Atoms and Copy_Atoms
    that need to create RMEM fragments.

    Example:
    ```mlir
      %0 = cute.make_layout_like(%layout)
        : !cute.layout<"(4,8):(1,1024)"> to !cute.layout<"(4,8):(1,4)">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$src
  );
  let results = (outs
    Cute_CuTeLayoutType:$dst
  );
}
def Cute_MakeLayoutOp : Cute_Op<"make_layout", [AttrSizedOperandSegments, Pure]> {
  let description = [{
    The `make_layout` operation generates a layout based on provided shape and stride.
    If the stride component is omitted in the operation's result type
    (e.g., `!cute.layout<"(?,(?,?))">`), the operation defaults to a column-major
    layout, and missing stride details are automatically inferred to complete the type
    of the resulting layout.

    The shape or stride operand must be supplied if they contain dynamic elements.
    However, if the shape or stride is entirely static, corresponding operands can be
    neglected.

    When the result type does not specify stride information (like `!cute.layout<"(?,?)">`
    ), it is not necessary to provide a stride operand, even if the stride to be inferred
    includes dynamic components (as in the inferred type `!cute.layout<"(?,?):(1,?)">`).

    ### Examples:

    ```mlir
    // Shape and stride are both dynamic
    %layout = cute.make_layout(%shape, %stride) : !cute.layout<"(?,64):(1,?)">

    // Shape and stride are both static
    %layout = cute.make_layout() : !cute.layout<"(32,64):(1,32)">

    // Stride is omitted in the result type, implying it should be inferred
    %layout = cute.make_layout(%shape) : !cute.layout<"(?,(?,?))">
    ```
  }];
  let arguments = (ins
    Optional<Cute_CuteShapeType>:$shape,
    Optional<Cute_CuteStrideType>:$stride
  );
  let results = (outs
    Cute_CuTeLayoutType:$layout
  );
}
def Cute_MakeOrderedLayoutOp : Cute_InferredOp<"make_ordered_layout", [Pure]> {
  let description = [{
    The `make_ordered_layout` operation generates a compact layout based on provided `shape`
    and stride is calculated by `shape` and `order`.

    Different from `make_layout`, `shape` operand is required, `order` is optional. When `order`
    isn't set, default stride growth order is from left to right. Supports dynamic values in `shape`,
    while currently `order` is restricted to static. Rank of `order` should be equal to rank of
    shape but no need for congruent.

    Note that numbers in `order` are just used for relative order relationship, e.g. (67,46,23,32)
    is equivalent to (4,3,1,2), (3,2,4) is equivalent to (2,1,3).

    ### Examples:

    ```mlir
    %shape = cute.make_shape() : !cute.shape<"(32,64)">
    %shape_dyn = cute.make_shape() : !cute.shape<"(32,?)">
    %order = cute.make_shape() : !cute.shape<"(1,2)">
    %layout = cute.make_ordered_layout(%shape, %order)
      : (!cute.shape<"(32,64)">, !cute.shape<"(1,2)">) -> !cute.layout<"(32,64):(1,32)">
    %layout_dyn = cute.make_ordered_layout(%shape_dyn, %order)
      : (!cute.shape<"(32,?)">, !cute.shape<"(1,2)">) -> !cute.layout<"(32,?):(1,32)">

    %order = cute.make_shape() : !cute.shape<"(2,1)">
    %layout = cute.make_ordered_layout(%shape, %order)
      : (!cute.shape<"(32,64)">, !cute.shape<"(2,1)">) -> !cute.layout<"(32,64):(64,1)">
    %layout_dyn = cute.make_ordered_layout(%shape_dyn, %order)
      : (!cute.shape<"(32,?)">, !cute.shape<"(2,1)">) -> !cute.layout<"(32,?):(?,1)">

    %shape = cute.make_shape() : !cute.shape<"(2,(2,2),2)">
    %order = cute.make_shape() : !cute.shape<"(0,(2,3),1)">
    %layout = cute.make_ordered_layout(%shape, %order) : !cute.layout<"(2,(2,2),2):(1,(4,8),2)">

    %order = cute.make_shape() : !cute.shape<"(2,1,3)">
    %layout = cute.make_ordered_layout(%shape, %order) : !cute.layout<"(2,(2,2),2):(4,(1,2),8)">
  }];
  let arguments = (ins
    Cute_CuteShapeType:$shape,
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$order
  );
  let results = (outs
    Cute_CuTeLayoutType:$layout
  );
}
def Cute_MakeShapeOp : Cute_Op<"make_shape", [Pure]> {
  let description = [{
    Constructs a shape. The operands correspond to the
    shape's dynamic elements, ordered from left to right.
    The number of operands should match the count of dynamic elements in the
    resulting shape type.

    Example:

    ```mlir
    // Definitions for %0, %1 come earlier
    %shape = cute.make_shape(%0, %1) : (i32, i32)
        -> !cute.shape<"((?,32),?)">
    ```
  }];
  let arguments = (ins
    Variadic<Cute_I32I64OrConstrainedInteger>:$dynamicElements
  );
  let results = (outs
    Cute_CuteShapeType:$result
  );
}
def Cute_MakeSparseElemOp : Cute_Op<"make_sparse_elem", [Pure]> {
  let summary = "Creates a sparse element from a physical storage.";
  let description = [{
    Example:

    ```mlir
    %v0 = arith.constant 8 : i8
    %v1 = arith.constant 31 : i8
    %physical_storage = vector.from_elements(vector<2xf16>, %v0, %v1)
    %sparse_elem = cute.make_sparse_elem(%physical_storage) : (!cute.sparse_elem<4, vector<2xf16>>)
    ```
  }];
  let arguments = (ins
    AnyType:$physical_storage
  );
  let results = (outs
    Cute_CuTeSparseElementType:$result
  );
}
def Cute_MakeStrideOp : Cute_Op<"make_stride", [Pure]> {
  let description = [{
    Constructs a stride. The operands correspond to the
    stride's dynamic elements, ordered from left to right.
    The number of operands should match the count of dynamic elements in the
    resulting stride type.

    Example:

    ```mlir
    // Definitions for %0, %1 come earlier
    %shape = cute.make_stride(%0, %1) : (i32, i32)
        -> !cute.stride<"((?,32),?)">
    ```
  }];
  let arguments = (ins
    Variadic<Cute_I32I64OrConstrainedInteger>:$dynamicElements
  );
  let results = (outs
    Cute_CuteStrideType:$result
  );
}
def Cute_MakeTileOp : Cute_Op<"make_tile", [Pure]> {
  let description = [{
    Constructs a tile. The operands correspond to the
    tile's dynamic elements, ordered from left to right.
    The number of operands should match the count of dynamic elements in the
    resulting tile type.

    Example:

    ```mlir
    // Definitions for %0, %1 come earlier
    %shape = cute.make_tile(%0, %1) : (i32, i32)
        -> !cute.tile<"((?,32),?)">
    ```
  }];
  let arguments = (ins
    Variadic<Cute_I32I64OrConstrainedInteger>:$dynamicElements
  );
  let results = (outs
    Cute_CuteTileType:$result
  );
}
def Cute_MakeTiledCopyOp : Cute_Op<"make_tiled_copy", [Pure]> {
  let description = [{
    This operation creates a new copy atom instance by tiling a "base" copy atom
    instance. Copy atoms are used by the CuTe algorithm layer during lowering to
    generate the appropriate `cute_nvgpu.arch` operations.

    Example:
    The below values are origin from cutlass SM100_f16f16_UMMA kernel, the original code:
    auto tAgA_mkl = cta_tma_a.partition_S(cta_mma.partition_A(gA_mkl))(_,_0{},_0{},_0{},_,_,_);
    All the values are from the "cta_tma_a".
    ```mlir
    %0 = cute.make_atom(%tma_desc_ptr, %tma_bar) : (!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>!cute.ptr<smem, align<8>>)
         -> !cute_nvgpu.atom.tma_load<f16, copy_bits = 131072, mode=tiled, num_cta = 1>
    %1 = cute.make_tiled_copy(%0)
      : !cute.tiled_copy<!cute_nvgpu.atom.tma_load<f16, copy_bits = 131072, mode=tiled, num_cta = 1>,
                         layout_copy_tv = <"((4,2),(64,128)):((0,64),(128,1))">,
                         tiler_mn = <"[2048:1;1:1;4:1;1:1]">>
    ```
  }];
  let arguments = (ins
    Cute_CopyAtomTypeInterfaceInstance:$atom
  );
  let results = (outs
    Cute_TiledCopyAtomType:$result
  );
}
def Cute_MakeTiledCopyV2Op : Cute_Op<"make_tiled_copy_v2", [Pure]> {
  let description = [{
    This operation creates a V2 tiled copy instance by tiling a V2 copy atom.
    V2 copy atoms use fragment attributes for operand-specific behavior.

    Example:
    ```mlir
    %0 = cute.make_atom() : () -> !cute_nvgpu.atom.sm100_s2t_copy_v2<f16, num_dp=128, num_bit=128, num_cta=1, smem_major=k>
    %1 = cute.make_tiled_copy_v2(%0)
      : !cute.tiled_copy_v2<!cute_nvgpu.atom.sm100_s2t_copy_v2<f16, num_dp=128, num_bit=128, num_cta=1, smem_major=k>,
                            layout_copy_tv = <"((32,4),(2,32)):((0,1),(128,4))">,
                            tiler_mn = <"[(4,32):(32,1);2:1]">>
    ```
  }];
  let arguments = (ins
    Cute_CopyAtomTypeInterfaceV2Instance:$atom
  );
  let results = (outs
    Cute_TiledCopyV2AtomType:$result
  );
}
def Cute_MakeTiledMmaOp : Cute_Op<"make_tiled_mma", [Pure]> {
  let description = [{
    This operation creates a new MMA atom instance by tiling a "base" MMA atom
    instance. MMA atoms are used by the CuTe algorithm layer during lowering to
    generate the appropriate `cute.arch` operations.

    Example:
    ```mlir
    %0 = cute.make_atom(%accum, %a_neg, %b_neg) : (i1, i1, i1)
      -> !cute_nvgpu.sm100.mma<64x64x16, num_cta = 1, ab_major = (k,mn), elem_type = (f16,f16,f16), frag_kind = ss, c_scale_exp = 0>
    %1 = cute.make_tiled_mma(%0)
      : !cute.tiled_mma<!cute_nvgpu.sm100.mma<64x64x16, num_cta = 1, ab_major = (k,mn), elem_type = (f16,f16,f16), frag_kind = ss, c_scale_exp = 0>,
                        atom_layout_MNK = <"(1,1,1)">>
    ```
  }];
  let arguments = (ins
    Cute_MmaAtomTypeInterfaceInstance:$atom
  );
  let results = (outs
    Cute_TiledMMAAtomType:$result
  );
}
def Cute_MakeTupleOp : Cute_Op<"make_tuple", [Pure]> {
  let description = [{
    Creates a tuple from multiple values. The number and types of the
    parameters must match the number and types of the result tuple.
  }];
  let arguments = (ins
    Variadic<AnyType>:$values
  );
  let results = (outs
    Cute_ATupleIsAFixedLengthOrderedList:$result
  );
}
def Cute_MakeViewOp : Cute_Op<"make_view", [Pure]> {
  let description = [{
    The cute.make_view operation construct `CuteViewTypeInterface`
    (i.e., cute.memref or cute.coord_tensor) with given `iter` and `layout`.
    Internally, we infer the exact return type by checking `iter`'s type:
    - if `iter` is cute.ptr, return type is `CuteMemRefType`
    - if `iter` is cute.tuple, return type is `CuteCoordTensorType`

    `iter` and `layout` could contain both static and dynamic values.

    `MakeViewOp` usually is the result of a previous operation
    (e.g., cute.slice, cute.zipped_divide), where creating a new
    `CuteViewTypeInterface` is needed.

    `MakeViewOp` does not involve allocating buffers of any kind.

    Example to make cute.coord_tensor with an iter of cute.tuple type:

    ```mlir
    %coord_tensor= cute.make_view(%iter, %layout) : !cute.coord_tensor<"(0,2)", "(30):(20@1)">
    ```

    Example to make cute.memref with an iter of cute.ptr type:

    ```mlir
    %memref= cute.make_view(%iter, %layout) : !cute.memref<f16, generic, "(30):(20)">
    ```

    Static Layout

    ```mlir
    %memref= cute.make_view(%iter) : !cute.memref<f16, generic, "(30):(20)">
    ```

    Composed+Static Layout

    ```mlir
    %memref= cute.make_view(%iter) : !cute.memref<f16, generic, "S<3,4,3> o 9 o (64):(16)">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedIteratorTypeOfCuTe:$iter,
    Optional<Cute_GeneralizedLayoutTypeOfCuTe>:$layout
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_MemRefAllocSmemOp : Cute_Op<"memref.alloc_smem"> {
  let description = [{
    The memref.alloc_smem operation is utilized for the allocation of static
    shared memory. The cosize of the memref layout, which must be static,
    determines the size of the memory allocated by this operation.

    ```mlir
    %0 = cute.memref.alloc_smem() : !cute.memref<f32, smem, "(32,24)">
    ```
  }];
  let arguments = (ins
  );
  let results = (outs
    Cute_MemrefInSmemWithStaticAndNonComposed:$memref
  );
}
def Cute_MemRefAllocaOp : Cute_Op<"memref.alloca"> {
  let description = [{
    The memref.alloca operation is used to allocate shared/stack memory, which is automatically
    released when control returns from the closest surrounding operation's region. This
    operation only supports memref with the address space reg, and attempting to allocate
    memory with other address spaces will result in an error.

    The size of memory allocated is determined by the layout operand's cosize, which can
    be either static or dynamic. The layout operand can be omitted when the memref layout
    is static.

    If the memref is sparse, the allocated memory size will be layout operand's cosize
    divided by sparsity factor.

    ```mlir
    // %dynamic_layout : !cute.layout<"(?,24)">
    %0 = cute.memref.alloca(%dynamic_layout) : !cute.memref<f32, rmem, "(?,24)">
    // static layout
    %1 = cute.memref.alloca() : !cute.memref<f32, rmem, "(32,24)">
    ```
  }];
  let arguments = (ins
    Optional<Cute_GeneralizedLayoutTypeOfCuTe>:$layout
  );
  let results = (outs
    Cute_MemrefInGenericOrRegisterMemOrSmem:$result
  );
}
def Cute_MemRefLoadOp : Cute_InferredOp<"memref.load", [MemoryEffects<[MemRead]>]> {
  let description = [{
    The `memref.load` operation loads one element at the specified logical coordinate from a memref
    in gmem/rmem/smem/generic memory spaces to a register. This operation performs the following steps:

    1. Converts the logical coordinate to a linear index using `cute.crd2idx`
    2. Retrieves the iterator from the memref using `cute.get_iter`
    3. Adds the offset to the iterator using `cute.add_offset`
    4. For CuteMemRefType, applies swizzling if needed, converts to LLVM pointer, and performs the load
    5. For CoordTensorType, returns the tuple directly

    The element type generally cannot be a sub-byte type (no f4, f6, i1, i2 etc.), with the exception
    of i1 which has special handling.

    NOTE: For tf32 types, the operation treat it as f32 even bitwidth of MLIR type is 19.

    Example:

    ```mlir
    // Load a fp32
    %coord = cute.make_coord(%c1_i32, %c2_i32) : (i32, i32) -> !cute.coord<"(?,?)">
    %0 = cute.memref.load(%memref, %coord)  : (!cute.memref<f32, rmem, "(16,4):(1,16)">, !cute.coord<"(?,?)">) -> f32

    // Load a tf32 value (will be handled with special bitwise casting)
    %1 = cute.memref.load(%tf32_memref, %coord) : (!cute.memref<tf32, rmem, "(16,4):(1,16)">, !cute.coord<"(?,?)">)  -> tf32
    ```
  }];
  let arguments = (ins
    Cute_AMemoryReferenceTypeThatRepresentsATensor2:$src,
    Cute_CuteCoordType:$coord
  );
  let results = (outs
    AnyType:$result
  );
}
def Cute_MemRefLoadVecOp : Cute_InferredOp<"memref.load_vec", [AttrSizedOperandSegments, MemoryEffects<[MemRead]>]> {
  let description = [{
    The `memref.load_vec` operation loads all elements in the logical coordinate
    space from a memref generic or register memory to a 1D vector. Only memrefs with
    static layouts are accepted.

    The resulting vector has the same element type as the memref, with a size
    equal to the **size** (not **cosize**) of the memref layout.

    An optional mask vector (of the same size as the output vector) can be
    provided to mask out elements. When the mask is present, an optional
    pass_thru vector can also be provided to fill in values for elements that
    are masked out, and thus the result is equivalent to a regular memref load
    followed by a select between the loaded and the passthru values, predicated
    on the same mask, except that the masked-off lanes are not accessed.

    Example:

    ```mlir
    // Load vector
    %0 = cute.memref.load_vec(%memref) : (!cute.memref<f32, rmem, "((32,1),1,1):((1,0),0,0)">) -> vector<32xf32>

    // Load vector from swizzled memory
    %0 = cute.memref.load_vec %memref : !cute.memref<f32, smem, S<1,3,1>, "((32,1),1,1):((1,0),0,0)">

    // Load vector with mask and pass_thru
    %0 = cute.memref.load_vec(%memref, %mask, %pass_thru)
          : (!cute.memref<f32, gmem, "((32,1),1,1):((1,0),0,0)">, vector<32xi1>, vector<32xf32>) -> vector<32xf32>
    ```
  }];
  let arguments = (ins
    Cute_StaticShape:$src,
    Optional<Cute_VectorOf1BitSignlessIntegerValuesOf>:$mask,
    Optional<Cute_OfRanks1>:$pass_thru
  );
  let results = (outs
    Cute_OfRanks1:$result
  );
}
def Cute_MemRefStoreOp : Cute_Op<"memref.store"> {
  let description = [{
    The `memref.store` operation stores a register value to a memref in gmem/rmem/smem/generic at given logical coordinate.
    This operation performs the following steps:

    1. Converts the logical coordinate to a linear index using `cute.crd2idx`
    2. Retrieves the iterator from the memref using `cute.get_iter`
    3. Adds the offset to the iterator using `cute.add_offset`
    4. Converts the pointer to the appropriate LLVM pointer type and performs the store

    The element type generally cannot be a sub-byte type (no f4, f6, i1, i2 etc.), with the exception
    of i1 which has special handling.

    NOTE: For tf32 types, the operation treats it as f32 even though the bitwidth of the MLIR type is 19.
    This special handling allows for proper conversion between tf32 and f32 representations during store operations.

    Example:

    ```mlir
    // Store a fp32
    %coord = cute.make_coord(%c1_i32, %c2_i32) : (i32, i32) -> !cute.coord<"(?,?)">
    cute.memref.store(%memref, %coord, %value) : (!cute.memref<f32, rmem, "(16,4):(1,16)">, !cute.coord<"(?,?)">, f32)

    // Store a tf32 value (will be handled with special bitwise casting)
    cute.memref.store(%tf32_memref, %coord, %tf32_value) : (!cute.memref<tf32, rmem, "(16,4):(1,16)">, !cute.coord<"(?,?)">, tf32)
    ```
  }];
  let arguments = (ins
    Cute_AMemoryReferenceTypeThatRepresentsATensor3:$dst,
    Cute_CuteCoordType:$coord,
    AnyType:$value
  );
  let results = (outs
  );
}
def Cute_MemRefStoreVecOp : Cute_Op<"memref.store_vec"> {
  let description = [{
    The `memref.store_vec` operation stores all elements of a 1D vector to a to the logical coordinate
    space of memref in generic or register memory. Only memrefs with static layouts are accepted.

    The vector has the same element type as the memref, with a size equal to
    the **size** (not **cosize**) of the memref layout.

    An optional mask vector (of the same size as the input vector) can be
    provided to mask out elements.

    ```mlir
    // Store vector
    %0 = cute.memref.store_vec(%value, %memref) : (vector<32xf32>, !cute.memref<f32, rmem, "((32,1),1,1):((1,0),0,0)">) -> ()

    // Store vector to swizzled memory
    %0 = cute.memref.store_vec %value, %memref : !cute.memref<f32, smem, S<1,3,1>, "((32,1),1,1):((1,0),0,0)">

    // Store vector with mask
    %0 = cute.memref.store_vec(%value, %memref, %mask)
          : (vector<32xf32>, !cute.memref<f32, gmem, "((32,1),1,1):((1,0),0,0)">, vector<32xi1>) -> ()
    ```
  }];
  let arguments = (ins
    Cute_OfRanks1:$value,
    Cute_StaticShape:$dst,
    Optional<Cute_VectorOf1BitSignlessIntegerValuesOf>:$mask
  );
  let results = (outs
  );
}
def Cute_MmaAtomCallOp : Cute_Op<"mma_atom_call", [AttrSizedOperandSegments]> {
  let summary = "`cute.mma_atom_call` op executes a mma atom operation with the given operands.";
  let description = [{
    The operands `a` and `b` are variadic, each containing a variable number of tensors:
    - For regular MMA, `a` and `b` contain the MMA A and B tensors respectively.
    - For MMA with auxiliary operands, `a` and `b` contain the MMA A and B tensors followed by their respective auxiliary tensors. For example:
      - For BlockScaledMMA, `a` = [A, SFA] and `b` = [B, SFB].
      - For SparseMMA, `a` = [A, E] and `b` = [B].
      - For BlockScaledSparseMMA, `a` = [A, SFA, E] and `b` = [B, SFB].
    Example (regular MMA):

    ```mlir
    // Call a mma atom operation
    cute.mma_atom_call(%mma_atom, %d, %a, %b, %c)
      : (!cute_nvgpu.sm100.mma<128x128x16, num_cta = 1, ab_major = (k,k),
                               elem_type = (f16,f16,f32),
                               frag_kind = ss,
                               c_scale_exp = 0>,
         !cute.memref<f16, tmem, align<16>, "((128,128)):((131072,1))">,
         !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
         !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
         !cute.memref<f16, tmem, align<16>, "((128,128)):((131072,1))">)
      -> ()
    ```

    Example (block-scaled MMA):

    ```mlir
    cute.mma_atom_call(%mma_atom, %d, [%a, %sfa], [%b, %sfb], %c)
      : (!cute_nvgpu.sm100.mma_bs<256x128x64, num_cta = 2, ab_major = (k,k),
                                  elem_type = (f4E2M1FN, f4E2M1FN, f32),
                                  sf_type = f8E4M3FN, frag_kind = ss, vec_size = 16>,
         !cute.memref<f32, tmem, align<16>, "((128,128)):((65536,1))">,
         [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
          !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4))):((((262144,4),8388608),(0,1)))">],
         [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
          !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4))):((((262144,4),8388608),(0,1)))">],
         !cute.memref<f32, tmem, align<16>, "((128,128)):((65536,1))">)
      -> ()
    ```

    Example (block-scaled MMA):

    ```mlir
    cute.mma_atom_call(%mma_atom, %d, [%a, %sfa], [%b, %sfb], %c)
      : (!cute_nvgpu.sm100.mma_bs<256x128x64, num_cta = 2, ab_major = (k,k),
                                  elem_type = (f4E2M1FN, f4E2M1FN, f32),
                                  sf_type = f8E4M3FN, frag_kind = ss, vec_size = 16>,
         !cute.memref<f32, tmem, align<16>, "((128,128)):((65536,1))">,
         [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
          !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4))):((((262144,4),8388608),(0,1)))">],
         [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
          !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4))):((((262144,4),8388608),(0,1)))">],
         !cute.memref<f32, tmem, align<16>, "((128,128)):((65536,1))">)
      -> ()
    ```

    Example (block-scaled MMA):

    ```mlir
    cute.mma_atom_call(%mma_atom, %d, [%a, %sfa], [%b, %sfb], %c)
      : (!cute_nvgpu.sm100.mma_bs<256x128x64, num_cta = 2, ab_major = (k,k),
                                  elem_type = (f4E2M1FN, f4E2M1FN, f32),
                                  sf_type = f8E4M3FN, frag_kind = ss, vec_size = 16>,
         !cute.memref<f32, tmem, align<16>, "((128,128)):((65536,1))">,
         [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
          !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4))):((((262144,4),8388608),(0,1)))">],
         [!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(1):(0)">,
          !cute.memref<f8E4M3FN, tmem, align<16>, "((((32,4),4),(16,4))):((((262144,4),8388608),(0,1)))">],
         !cute.memref<f32, tmem, align<16>, "((128,128)):((65536,1))">)
      -> ()
    ```
  }];
  let arguments = (ins
    Cute_MmaAtomTypeInterfaceInstance:$atom,
    Cute_ViewTypeInterfaceInstance:$d,
    Variadic<Cute_ViewTypeInterfaceInstance>:$a,
    Variadic<Cute_ViewTypeInterfaceInstance>:$b,
    Cute_ViewTypeInterfaceInstance:$c
  );
  let results = (outs
  );
}
def Cute_MmaMakeFragmentOp : Cute_InferredOp<"mma.make_fragment", [Pure]> {
  let description = [{
    Given a partitioned tensor, the `cute.mma.make_fragment` operation returns
    a tensor with the correct memory space for the specified operand and dot
    atom. This op is usually used between partition and gemm. It is the
    "prepare" step before gemm.

    `cute.tiled.mma.partition` returns a view into the specified tensor, but it
    does not copy the data to the correct memory space. This is done by
    `cute.mma.make_fragment` and explicit memcpy operations. Whether a memcpy
    is needed or not depend on the result memory space.

    * RMEM: The result tensor is a newly allocated tensor with unspecified
      contents. A memcpy is needed.
    * SMEM: The result tensor is a view into the input tensor in the form of a
      CuteSmemDescViewType. A memcpy is not needed.
    * TMEM: The result tensor uses a dummy TMEM engine (fake pointer 0x0) and
      must be set correctly later, aligned with Cute C++.

    Instead of a tensor, this op also accepts cute.layout (for RMEM/TMEM) and
    cute.shape (for RMEM/TMEM).

    Example:

    ```mlir
    // Example 1: Full example for SM80.
    // Partition a tensor for operand A, which is stored in SMEM.
    %partitioned = cute.tiled.mma.partition A(%atom, %smem_A, %tid_coord)
      : (!cute.tiled_mma<!cute_nvgpu.sm80.mma<16x8x16,
                                              elem_type = (f16,f16,f32)>,
                                              atom_layout_MNK = <"(2,2,1)">>,
         !cute.memref<f16, smem, align<16>, S<2,4,3>, "(128,32):(32,1)">,
         !cute.coord<"?">)
      -> !cute.memref<f16, smem, align<4>, S<2,4,3>, "((2,2,2),4,2):((1,256,8),1024,16)">

    // The tensor is still in SMEM, but sm80.mma expects A in registers.
    // Prepare a tensor with the correct shape and strides in RMEM.
    %fragment = cute.mma.make_fragment A (%3, %partitioned)
      : (!cute.tiled_mma<!cute_nvgpu.sm80.mma<16x8x16,
                                              elem_type = (f16,f16,f32)>,
                                              atom_layout_MNK = <"(2,2,1)">>,
         !cute.memref<f16, smem, align<4>, S<2,4,3>, "((2,2,2),4,2):((1,256,8),1024,16)">)
      -> !cute.memref<f16, rmem, align<16>, "((2,2,2),4,2):((1,2,4),16,8)">

    // Copy the data from SMEM to RMEM.
    cute.copy(..., %partitioned, %fragment)

    // Example 2: Make fragment for SM100 (SMEM).
    %fragment = cute.mma.make_fragment A (%0, %1)
      : (!cute.tiled_mma<!cute_nvgpu.sm100.mma<128x128x16,
                                               num_cta = 1,
                                               ab_major = (k,mn),
                                               elem_type = (f16,f16,f16),
                                               frag_kind = ss,
                                               c_scale_exp = 0>,
                         atom_layout_MNK = <"(1,1,1):(0,0,0)">>,
         !cute.memref<f16, smem, align<16>, "((128,16),2,32):((512,1),65536,16)">)
      -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "((128,16),2,32):((1,128),65536,2048)">

    // Example 3: Make fragment for SM100 (TMEM) from layout.
    %fragment = cute.mma.make_fragment A (%0, %1)
      : (!cute.tiled_mma<!cute_nvgpu.sm100.mma<128x128x16,
                                               num_cta = 1,
                                               ab_major = (k,mn),
                                               elem_type = (f16,f16,f16),
                                               frag_kind = ts,
                                               c_scale_exp = 0>,
                         atom_layout_MNK = <"(1,1,1):(0,0,0)">>,
         !cute.layout<"((64,16),1,4):((1,64),0,1024)">)
      -> !cute.memref<f16, tmem, align<1>, "((128,16),1,4):((131072,1),0,16)">

    // Example 3: Make fragment for SM100 (TMEM) from shape.
    %fragment = cute.mma.make_fragment A (%0, %1)
      : (!cute.tiled_mma<!cute_nvgpu.sm100.mma<128x128x16,
                                               num_cta = 1,
                                               ab_major = (k,mn),
                                               elem_type = (f16,f16,f16),
                                               frag_kind = ts,
                                               c_scale_exp = 0>,
                         atom_layout_MNK = <"(1,1,1):(0,0,0)">>,
         !cute.shape<"((64,16),1,4)">)
      -> !cute.memref<f16, tmem, align<1>, "((128,16),1,4):((131072,1),0,16)">
    ```
  }];
  let arguments = (ins
    Cute_MmaAtomTypeInterfaceInstance:$atom,
    Cute_AMemoryReferenceTypeThatRepresentsATensor:$input,
    Cute_MmaOperandEnum:$operand_id
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_PrefetchAtomCallOp : Cute_Op<"prefetch_atom_call", [MemoryEffects<[MemRead]>]> {
  let summary = "`cute.exec_prefetch_atom` op executes a prefetch atom operation with the given operands.";
  let description = [{
    Following src layout of atom are valid
    * ((atom_v, rest_v))
    * ((atom_v))
    * (atom_v)

    The `src` operand is variadic:
    - For regular TMA prefetch, `src` is a single tensor.
    - For 2D `tile::gather4` mode, `src` is `[data_coord_tensor, gmem_index_tensor]`.

    Examples:

    ```mlir
    // Call a prefetch atom operation
    cute.prefetch_atom_call(%copy_tma_load, %src_v)
      : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 131072, mode=tiled, num_cta = 1, g_stride=<"()">,
         %src: !cute.coord_tensor<"(0,0)", "(((64,128),1),1,1,2):(((1@0,1@1),0),0,0,64@0)">)
      -> ()

    // gather4 prefetch: data tensor + gmem index tensor
    cute.prefetch_atom_call(%gather4_atom, %src_v, %idx_v)
      : (!cute_nvgpu.atom.tma_load<f16, copy_bits = ..., mode=gather4, num_cta = 1, g_stride=<"()">>,
         !cute.coord_tensor<...>,
         !cute.coord_tensor<...>)
      -> ()
    ```
  }];
  let arguments = (ins
    Cute_PrefetchAtomTypeInterfaceOrTiledCopyTypeWithPrefetchAtomTypeInterfaceCopyAtom:$atom,
    Variadic<Cute_ViewTypeInterfaceInstance>:$src
  );
  let results = (outs
  );
}
def Cute_PrependToRankOp : Cute_InferredOp<"prepend_to_rank", [Pure]> {
  let summary = "Prepend element to input to rank N.";
  let description = [{
    Example:

    ```mlir
    %shp = cute.make_shape() : !cute.shape<"(3,4,5)">
    %elem = cute.make_shape() : !cute.shape<"(1,2)">
    %new_layout = cute.prepend_to_rank<5> (%shp, %elem)
      : (!cute.shape<"(3,4,5)">, !cute.shape<"1">) -> !cute.shape<"((1,2),(1,2),3,4,5)">
    ```
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstance:$input,
    Cute_ValueTypeInterfaceInstance:$element,
    SI32Attr:$rank
  );
  let results = (outs
    Cute_ValueTypeInterfaceInstance:$result
  );
}
def Cute_PrintOp : Cute_Op<"print"> {
  let summary = "Debug operation to print out value of cute data type.";
  let description = [{
    Supported following cute data types
    * shape, stride, layout, tile
    * View: memref, coord_tensor
    * Iterator: ptr, tuple

    Examples:

    ```mlir
    cute.print(%0) : !cute.shape<"(3,5)">

    cute.print(%1) : !cute.layout<"(5,(3,?)):(?,(4,5))">

    // variadic operands without format string
    cute.print(%0, %1) : !cute.shape<"(3,5)">, !cute.layout<"(5,(3,?)):(?,(4,5))">

    // with format string
    cute.print("{} -> {}", %0, %1) : !cute.shape<"(3,5)">, !cute.layout<"(5,(3,?)):(?,(4,5))">
    ```

    The optional `operand_signed` attribute carries the DSL-level signedness of
    each operand, one entry per value. It exists because DSL integers lower to
    signless MLIR integer types, so an unsigned value is indistinguishable from
    a signed one by type alone. When an entry is false, a generic `{}`
    placeholder for that integer operand is rendered with an unsigned specifier
    (`%u`/`%llu`) instead of `%d`/`%lld`. When absent, all operands are treated
    as signed, and an explicit `%u` in the format string still forces unsigned.
  }];
  let arguments = (ins
    Variadic<Cute_TypeInterfaceForCuteTypesCanBePrinted>:$values,
    OptionalAttr<StrAttr>:$fmt,
    OptionalAttr<DenseBoolArrayAttr>:$operand_signed
  );
  let results = (outs
  );
}
def Cute_PrintTMADescIm2ColOp : Cute_Op<"print_tma_desc_im2col"> {
  let summary = "This operation interprets the input i64 as a pointer to a TmaDescriptorInternal.";
  let description = [{
    Example:
    ```mlir
        %tma_desc_a_host = cute_nvgpu.make_tma_desc_im2col(
            %tensor,
            %slayout,
            %cta_v_tile,
            %lower_corner_whd,
            %upper_corner_whd,
            %lower_padding_whd,
            %upper_padding_whd,
            %stride_whd,
            %lower_srt,
            %stride_srt)
            <{host = true}> : (
            !cute.memref<f16, gmem, align<16>, "((8,8,1,4),64,1):((64,512,4096,4096),1,0)">,
            !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1)):((32,256),(1,0))">,
            !cute.layout<"(128,32):(1@0,1@1)">,
            !cute.int_tuple<"(0,0,0)">,
            !cute.int_tuple<"(0,0,0)">,
            !cute.int_tuple<"(0,0,0)">,
            !cute.int_tuple<"(0,0,0)">,
            !cute.int_tuple<"(1,1,1)">,
            !cute.int_tuple<"(0,0,0)">,
            !cute.int_tuple<"(1,1,1)">) -> !cute.ptr<!cute_nvgpu.tma_descriptor_im2col, generic, align<64>>
    %tma_desc_a_host_ptr_i64 = cute_nvgpu.cast_tma_desc_to_integer(%tma_desc_a_host: !cute.ptr<!cute_nvgpu.tma_descriptor_im2col, generic>) -> i64
    %fd = arith.constant 1 : i32 // stdout
    cute.print_tma_desc_im2col(%tma_desc_a_host_ptr_i64, %fd)
  }];
  let arguments = (ins
    I64:$input,
    I32:$fd
  );
  let results = (outs
  );
}
def Cute_PrintTMADescTiledOp : Cute_Op<"print_tma_desc_tiled"> {
  let summary = "This operation interprets the input i64 as a pointer to a TmaDescriptorInternal.";
  let description = [{
    Example:
    ```mlir
    %tma_desc_a_host = cute_nvgpu.make_tma_desc_tiled(%a_tensor, %smem_ablayout_128_64_k_major, %cta_v_tile_128_64_k_major)
      <{mode = <tiled> host = true}> :
      (
          !cute.memref<bf16, gmem, align<16>, "(?,?,?) : (?,1,?)">,
          !cute.composed_layout<"S<3, 4, 3> o 0 o ((128,64),1,1):((64,1),0,0)">,
          !cute.layout<"((128,64),1,1):((1@0,1@1),0,0)">
      ) -> !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic>
    %tma_desc_a_host_ptr_i64 = cute_nvgpu.cast_tma_desc_to_integer(%tma_desc_a_host: !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic>) -> i64
    %fd = arith.constant 1 : i32 // stdout
    cute.print_tma_desc_tiled(%tma_desc_a_host_ptr_i64, %fd)
    ```
  }];
  let arguments = (ins
    I64:$input,
    I32:$fd
  );
  let results = (outs
  );
}
def Cute_PrintViewOp : Cute_Op<"print_view"> {
  let summary = "Debug operation to print cute view types.";
  let description = [{
    Currently supports cute.memref and cute.tensor. MemRefs with both static layout
    and dynamic layout are accepted. An optional coord parameter can be used to
    specify the element to be printed. If no coord is provided, all elements will
    be printed.

    The `verbose` attribute controls the output format:
    - When `verbose` is true (default), it prints the full details of the view
      including memory address, shape, and strides.
    - When `verbose` is false, it prints in a more compact torch-style matrix format.

    The `signed` attribute is used for integer data types to control whether values
    should be interpreted as signed or unsigned. This attribute is ignored for
    floating point types and coord tensors.

    Examples:

    ```mlir
    cute.print_view(%0) : !cute.memref<f32,rmem,"(2,?):(?,1)">
    output:
      Memref raw_ptr(0x0000152397fffc80: f32, rmem, align<4>) o (2,2):(2,1) = (
        (0,0)=1.000000
        (0,1)=2.000000
        (1,0)=3.000000
        (1,1)=4.000000
      )

    cute.print_view(%1) : !cute.memref<f32,rmem,"(2,(?,2)):(?,(2,1))">
    output:
      Memref raw_ptr(0x0000148eabfffbf0: f32, rmem, align<4>) o (2,(2,2)):(4,(2,1)) = (
        (0,(0,0))=1.000000
        (0,(0,1))=2.000000
        (0,(1,0))=3.000000
        (0,(1,1))=4.000000
        (1,(0,0))=5.000000
        (1,(0,1))=6.000000
        (1,(1,0))=7.000000
        (1,(1,1))=8.000000
      )

    cute.print_view(%3,%coord) : !cute.memref<f32,rmem,"(2,(?,2)):(?,(2,1))">,!cute.coord<"(?,(?,?))">
    output:
      Memref raw_ptr(0x0000150ce3fffc90: f32, rmem, align<4>) o (2,(2,2)):(4,(2,1)), coord (1,(1,0)) = (7.000000)

    // Using verbose=false to print in torch-style format
    cute.print_view(%0) {verbose = false} : !cute.memref<f32,rmem,"(8,5,3):(15,3,1)">

    output:
      tensor(raw_ptr(0x000000003746a2e0: f64, generic, align<8>) o (8,5,3):(15,3,1), data=
             [[[ 1.426091, -0.279034, -1.635684, -0.026360, -0.732545, ],
               [-0.733555,  0.886134, -1.418841, -0.203192, -1.836270, ],
               ...
               [-1.172012,  1.678573, -0.004324,  0.441605,  1.484385, ],
               [-0.550848,  1.455505, -0.968547,  1.233722,  0.812084, ]],

              [[ 0.762814, -0.641657,  0.843695, -1.380960,  0.261200, ],
               [ 1.500137, -0.921287,  1.001725,  0.106540, -1.326231, ],
               ...
               [-0.878001, -0.465694, -1.204548, -0.407944, -0.092967, ],
               [ 1.035407,  0.043012, -1.959517,  0.075913,  0.893392, ]],

              [[-0.129989,  0.750885,  0.219408,  0.442481, -0.508588, ],
               [ 0.791268, -0.547978,  0.194177,  0.652729,  2.642380, ],
               ...
               [-0.674993, -1.017359, -1.272791,  0.829529,  0.852879, ],
               [-0.256070, -0.287006,  1.900199, -0.492318, -0.658134, ]]])

    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstance:$src,
    Optional<Cute_CuteCoordType>:$coord,
    DefaultValuedAttr<BoolAttr, "true">:$verbose,
    DefaultValuedAttr<BoolAttr, "true">:$is_signed
  );
  let results = (outs
  );
}
def Cute_PtrLoadOp : Cute_InferredOp<"ptr.load", [MemoryEffects<[MemRead]>]> {
  let summary = "Loads an LLVM or CuTe value from the given `!cute.ptr`.";
  let arguments = (ins
    Cute_CuTePointerType:$ptr
  );
  let results = (outs
    Cute_CuTePointerTypeOrAMemoryReferenceType:$value
  );
}
def Cute_PtrStoreOp : Cute_Op<"ptr.store"> {
  let description = [{
    Stores the given CuTe value at the provided pointer. Handles any
    LLVM-storable type, including those that can be stored with
    `llvm.store` *and* CuTe types.
  }];
  let arguments = (ins
    Cute_CuTePointerTypeOrAMemoryReferenceType:$value,
    Cute_CuTePointerType:$ptr
  );
  let results = (outs
  );
}
def Cute_PtrToIntOp : Cute_Op<"ptrtoint", [Pure]> {
  let description = [{
    The `cute.ptrtoint` operation accepts a cute pointer value and casts it to an
    integer value. The pointer must have byte alignment, meaning subbyte types
    must have explicit alignment specified (e.g., align<1> or higher).

    Example:

    ```mlir
    %int_aligned = cute.ptrtoint(%ptr) : !cute.ptr<f16, smem, align<8>> to i64
    ```
  }];
  let arguments = (ins
    Cute_PointerTypeInterfaceInstance:$src
  );
  let results = (outs
    Cute_Integer:$dst
  );
}
def Cute_RakedProductOp : Cute_InferredOp<"raked_product", [Pure]> {
  let description = [{
    This operation performs "raked_product" on the input and produce a raked_product result,
    and it automatically infer the "raked_product" layout for the result type.

    Example:

    ```mlir
    %input = cute.make_layout() : !cute.layout<"(3,4):(4,1)">
    %tiler = cute.make_layout() : !cute.layout<"(2,5):(1,2)">
    %prod = cute.raked_product(%arg0, %arg1)
      : (!cute.layout<"(3,4):(4,1)">, !cute.layout<"(2,5):(1,2)">)
      -> !cute.layout<"((2,3),(5,4)):((12,4),(24,1))">
    ```

    Example with dynamic composed_layout:

    ```
    %prod = cute.raked_product(%input, %tiler)
      : (!cute.composed_layout<"(?):(?) o ? o (5,4):(4,1)">,
         !cute.layout<"(?,?):(?,?)">)
      -> !cute.composed_layout<"(?):(?) o ? o ((?,5),(?,4)):((?{div=20},4),(?{div=20},1))">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuTeLayoutType:$tiler
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_RandomDelayOp : Cute_Op<"random_delay"> {
  let arguments = (ins
  );
  let results = (outs
  );
}
def Cute_RecastIterOp : Cute_Op<"recast_iter", [Pure]> {
  let description = [{
    This `cute.recast_iter` is similar to CuTe C++'s recast_ptr, convert two iterators
    in certain aspects like type/swizzle for convenience.
    But this operation has some usage limitation for correctness,
    like it's not allowed for TMA tensor's iterator and it's not allowed to convert align attribute.
    In general, it's not allowed to convert two different iterator kinds (PtrType <-> IntTupleType)

    Example:

    ```mlir
      %ptr = cute.recast_iter(%0) : !cute.ptr<f16, smem, align<128>, S<3,4,3>>
          to !cute.ptr<f32, smem, align<128>, S<2,4,3>>
    ```
  }];
  let arguments = (ins
    Cute_CuTePointerType:$src
  );
  let results = (outs
    Cute_CuTePointerType:$dst
  );
}
def Cute_RecastLayoutOp : Cute_InferredOp<"recast_layout", [Pure]> {
  let summary = "Example:";
  let description = [{
    ```mlir
      # upcast<8> operation
      %0 = cute.recast_layout<32, 4>(%arg0)
        : !cute.layout<"(128,64):(1,128)"> to !cute.layout<"(16,64):(1,16)">
      # upcast<2> for composed layout
      %0 = cute.recast_layout<32, 16>(%arg0)
        : !cute.composed_layout<"((6,7),8):((10,12),1) o 6 o (8,2):(1,4)">
        to !cute.composed_layout<"((6,7),4):((5,6),1) o 3 o (4,2):(1,2)">
      # downcast<8> operation
      %0 = cute.recast_layout<8, 32>(%arg0)
        : !!cute.layout<"(16,64):(1,16)"> to cute.layout<"(128,64):(1,128)">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$src,
    I32Attr:$new_type_bits,
    I32Attr:$old_type_bits
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$dst
  );
}
def Cute_ReduceOp : Cute_InferredOp<"reduce", [Pure]> {
  let description = [{
    The `cute.reduce` operation reduces elements of a vector according to a given
    shape and reduction operator, optionally guided by a reduction profile and/or a
    predication mask.

    The reduction is performed over the dimensions of the input vector as described
    by the provided shape. By default, all dimensions are reduced to a scalar. If a
    reduction profile is specified, it determines which dimensions are kept (marked
    with '_') and which are reduced (marked with an integer or tuple structure). The
    result is either a scalar or a lower-rank vector, along with the reduced shape
    if applicable.

    An optional predication vector (of the same size as the input vector) can be
    provided to mask out elements from the reduction, enabling conditional
    reduction.

    The reduction operator (e.g., MIN, MAX, ADD, MUL) is specified via the `op`
    attribute.

    Examples:
    ```mlir
    // Reduce all elements to a scalar (minimum)
    %reduced = cute.reduce(%input, %shape, %init)
      { op = #cute.reduction_op<MIN> }
      : (vector<20xf32>, !cute.shape<"(4,5)">, f32) -> f32

    // Reduce all elements to a scalar (maximum) with predication
    %reduced = cute.reduce(%input, %shape, %init, %pred)
      { op = #cute.reduction_op<MAX> }
      : (vector<20xf32>, !cute.shape<"(4,5)">, f32, vector<20xi1>) -> f32

    // Reduce all but the second dimension (profile keeps dimension 1)
    %reduced, %red_shp = cute.reduce(%input, %shape, %init)
      { op = #cute.reduction_op<ADD>, profile = #cute.coord<"(1,_,1)"> }
      : (vector<60xf32>, !cute.shape<"(5,4,3)">, f32) -> (vector<4xf32>, !cute.shape<"(4)>")

    // Reduce a nested dimension (profile keeps outer and first inner dimension)
    %reduced, %red_shp = cute.reduce(%input, %shape, %init)
      { op = #cute.reduction_op<MUL>, profile = #cute.coord<"(_,(_,1))"> }
      : (vector<30xf32>, !cute.shape<"(5,(3,2))">, f32) -> (vector<15xf32>, !cute.shape<"(5,3)>")
    ```
    The operation supports both flat and nested shapes and profiles, enabling reductions over complex multidimensional data layouts.
  }];
  let arguments = (ins
    Cute_OfRanks1:$input,
    AnyType:$shape,
    AnyType:$init_val,
    Optional<Cute_VectorOf1BitSignlessIntegerValuesOf>:$predication,
    Cute_ReductionOpAttr:$reduction_op,
    OptionalAttr<Cute_CoordAttr>:$reduction_profile
  );
  let results = (outs
    Cute_OfRanks1OrFloatingPointOrInteger:$output,
    Optional<AnyType>:$reduced_shape
  );
}
def Cute_RightInverseOp : Cute_InferredOp<"right_inverse", [Pure]> {
  let description = [{
    This operation builds the right inverse of a layout.
    The current implementation only supports static layout.
    An error is returned when the layout contains dynamic values.

    Example:

    ```mlir
    %layout = cute.make_layout(.., ..): !cute.layout<"(8, 4):(4, 1)">
    %inv_layout = cute.right_inverse %layout : (!cute.layout<"(8, 4):(4, 1)">) -> !cute.layout<"(4, 8):(8, 1)">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_SelectOp : Cute_InferredOp<"select", [Pure]> {
  let description = [{
    This operation extracts specific dimensions out of a type implementing
    the `CuteValueTypeInterface`, e.g. a shape, a coordinate, a tuple, a layout, ...

    The `mode` attribute indices are used to select the portion of the input type.
    The indices in the `mode` attribute can be permutated or repeated, but should
    be less than the rank of type.

    Example:

    ```mlir
    %shape = cute.make_shape() : () -> !cute.shape<"(1, (2, 4), 8)">
    %mode0 = cute.select<[1]>(%shape)
      : (!cute.shape<"(1,(2,4),8)">) -> !cute.shape<"((2, 4))">
    %mode1 = cute.select<[0,1]>(%shape)
      : (!cute.shape<"(1,(2,4),8)">) -> !cute.shape<"(1, (2,4))">
    %mode2 = cute.select<[2,0]>(%shape)
      : (!cute.shape<"(1,(2,4),8)">) -> !cute.shape<"(8, 1)">
    %mode3 = cute.select<[2,2]>(%shape)
      : (!cute.shape<"(1,(2,4),8)">) -> !cute.shape<"(8, 8)">
    ```
  }];
  let arguments = (ins
    Cute_ValueTypeInterfaceInstance:$input,
    DenseI32ArrayAttr:$mode
  );
  let results = (outs
    Cute_ValueTypeInterfaceInstance:$result
  );
}
def Cute_ShapeDivOp : Cute_InferredOp<"shape_div", [Pure]> {
  let description = [{
    Division for Shapes.
    Case Tuple Tuple:
      Perform shape_div element-wise
    Case Tuple Int:
      Fold the division of b across each element of a
      Example: shape_div((4,5,6),40) -> shape_div((1,5,6),10) -> shape_div((1,1,6),2) -> (1,1,3)
    Case Int Tuple:
      Return shape_div(a, product(b))
    Case Int Int:
      Enforce the divisibility condition a % b == 0 || b % a == 0 when possible
      Return a / b with rounding away from 0 (that is, 1 or -1 when a < b)

    ```mlir
     %res = cute.shape_div(%a, %b)
        : (!cute.int_tuple<"(4,5)">, !cute.int_tuple<"(40)">) ->  !cute.int_tuple<"(1,1)">
    ```
  }];
  let arguments = (ins
    Cute_CuteShapeType:$a,
    Cute_CuteShapeType:$b
  );
  let results = (outs
    Cute_CuteShapeType:$result
  );
}
def Cute_SizeOp : Cute_InferredOp<"size", [Pure]> {
  let summary = "This operation computes the size of a cute value type by multiplying its dimensions.";
  let description = [{
    The `mode` attribute specifies the nested mode in the type for size calculation.
    It indexes a sub-scope within the input type and returns the size of that sub-scope.
    The rank of `mode` must be less than or equal to the depth of the type.
    If `mode` is empty, the operation calculates the size of the entire type.

    ### Examples

    ```mlir
    %shape = cute.make_shape() : () -> !cute.shape<"(1,(2,4),9)">
    %size = cute.size(%shape) <{mode=[1]}> : (!cute.shape<"(1,(2,4),9)">) -> !cute.int_tuple<"8">
    ```

    In this example, the mode attribute is set to [1], which selects the second dimension of the
    shape (1,(2,4),9). The size of this dimension is calculated as 2 * 4 = 8.

    ```mlir
    %shape = cute.make_shape() : () -> !cute.shape<"(1,(2,4),8)">
    %size = cute.size(%shape) : (!cute.shape<"(1,(2,4),8)">) -> !cute.int_tuple<"64">
    ```

    In this example, the mode attribute is empty, so the operation calculates the size of the entire
    shape (1,(2,4),8). The size is calculated as 1 * 2 * 4 * 8 = 64.
  }];
  let arguments = (ins
    Cute_CuteShapeTypeOrIntTupleAsIntegerOr2:$input,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$mode
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$result
  );
}
def Cute_SliceOp : Cute_InferredOp<"slice", [Pure]> {
  let description = [{
       This operation takes slices of a input memref according to the `coord`. Note that
       `!coord<"(_,_,k)">` means "create a slice that views (i, j, k) for all valid i,
       all valid j, and a specific value of k.

       The `coord` elements can be static, dynamic (represented by "?"), or underscore
       (similar to the ":" (the colon punctuation) in Python or Fortran array slices).
       However, the mode of `coord` should be less than or equal to the mode of `input`
       memref layout; otherwise, a runtime error will occur.

       Example:

       ```mlir
       %slice = cute.slice(%alloc, %coord)
         : !cute.memref<f16, generic, "(1,(16384,1),1):(0,(1,0),0)">, !cute.coord<"(0, (_, _), 0)">
       ```

       Example with dynamic coord:

       ```mlir
       %slice = cute.slice(%alloc, %coord)
         : !cute.memref<f16, generic, "(32,(16384,64),1):(1,(32,524288),0)">, !cute.coord<"(_,(64,?),0)">
       ```

       Example with scaled basis in layout:

       ```mlir
       %slice = cute.slice(%coord_tensor, %coord)
         : !cute.coord_tensor<"(0,0)", "(10,20,30):(1@0,1@1,20@1)">, !cute.coord<"(0,2,_)">
       ```
       Example with swizzled layout:

       ```mlir
       %slice = cute.slice(%swizzle_layout, %coord)
        : (!cute.composed_layout<"S<1,5,1> o 0 o ((2,4),(8,4)):((1,16),(2,64))">,
    !cute.coord<"((?,?),(_,_))">) -> !cute.composed_layout<"(8,(2,2)):(2,(?,128))">
       ```

       Example with PDSL memref slice (swizzle in units of bytes):

       ```mlir
       %res_memref = !cute.slice(%memref, %crd) : !cute.memref<i8, S<1,4,3>, "(64,32):(32,1)">, !cute.coord<?,_>
       ```

       Example with PISL memref slice (swizzle in units of elements):

       ```mlir
       %res_memref = !cute.slice(%memref, %crd) : !cute.memref<i8, "S<1,4,3> o 0 o (64,32):(32,1)">, !cute.coord<?,_>
       ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuteCoordType:$coord
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_SmemPartitionSizeOp : Cute_Op<"smem_partition_size", [Pure]> {
  let description = [{
    This operation queries the static smem layout for one partition in the
    current kernel and returns the partition size as an i64 integer. It is only
    valid on the kernel side; host-side total smem queries should use
    `cute.kernel_smem_size`.
  }];
  let arguments = (ins
    I32Attr:$partition_id,
    UnitAttr:$cumulative
  );
  let results = (outs
    I64:$size
  );
}
def Cute_StaticOp : Cute_Op<"static", [Pure]> {
  let description = [{
    Creates a compile-time constant value with a fully static CuTe type. This operation takes no
    operands and produces a single result of one of the following CuTe types:

    - !cute.int_tuple: A tuple of static integers
    - !cute.coord: A coordinate with static dimensions
    - !cute.stride: A stride with static values
    - !cute.shape: A shape with static dimensions
    - !cute.tile: A tile with static dimensions and layout
    - !cute.layout: A layout with static shape and strides
    - !cute.composed_layout: A composed layout with static inner/outer layouts

    The type of the result must be fully static (contain no dynamic/unknown values).

    Some atom types can be fully static with no runtime states, e.g. `UniversalFma`.

    ### Examples:

    ```mlir
    // Create a static shape with value 5
    %0 = cute.static : !cute.shape<"5">

    // Create a static 2D layout with shape (32,64) and strides (64,1)
    %1 = cute.static : !cute.layout<"(32,64):(64,1)">
    ```
  }];
  let arguments = (ins
  );
  let results = (outs
    Cute_MaybeStaticTypeInterfaceInstance:$result
  );
}
def Cute_StencilDivideOp : Cute_InferredOp<"stencil_divide", [AttrSizedOperandSegments, Pure]> {
  let summary = "A convolution has the following input tensors:";
  let description = [{
    * Activation tensor (Act): `((N,(D,H,W,...)), (C,(1,1,1)))`
    * Filter tensor (Flt): `(K,(C,(T,R,S, ...)))`
    * Output tensor (Out): `((N,(Z,P,Q, ...)), K)`

    Where

    * `N` := number of samples in the signal
    * `D`, `H`, `W`, ... := spatial dimensions of the activation tensor
    * `C` := channel dimension of the activation tensor
    * `K` := channel dimension of the filter and output tensor
    * `T`,`R`,`S`,... := spoke dimensions of the filter tensor
    * `Z`,`P`,`Q`,... := spatial dimensions of the output tensor

    The number of spatial dimensions must DHW... must match both TRS... and ZPQ...

    As is evident in the tensor shapes, these cannot be issued to a GEMM without transformation, since
    there are no logical M, N, and K modes into which we can group the tensor modes.

    Every spoke of the filter tensor (TRS) will be applied to some (offset)
    view of the activation tensor, thus expanding the logical size of the activation tensor.
    Additionally, a similar logical transform of the spatial dimensions can be encoded as
    a function of the padding, dilations, traversal strides, and filter spokes.
    This gets us to our im2col transform:

    Im2col transform affects the component shapes/strides of the activation tensor in the following way:

    * `ZPQ` Shape : changes `DHW` domain with formula `(1 + (DHW + pad - (((TRS-1) * dilation) + 1)) / traversal_stride)`
    * `TRS` Shape : `TRS` domain instead of `(1,1,1)`
    * `ZPQ` Strides : Original `DHW` strides get scaled by traversal strides `DHW`
    * `TRS` Strides : Original `DHW` strides get scaled by dilation `DHW`.

    With this transform applied, we end up with a set of input and output tensors that are logically
    consistent in their `MNK` dimensions, thus allowing us to dispatch to a GEMM.

    * im2col activation layout: `((N,(Z,P,Q)), (C,(T,R,S)))` // logical (M,K)
    * filter layout :    `( K, (C,(T,R,S)))` // logical (N,K)
    * output layout : `((N,(Z,P,Q)), K )` // logical (M,N)

    `stencil_divide` performs this operation given these parameters.
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuteTileType:$stencil,
    Optional<Cute_CuteShapeType>:$padding_upper,
    Optional<Cute_CuteShapeType>:$padding_lower,
    Optional<Cute_CuteStrideType>:$traversal_stride
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_SymbolicOp : Cute_Op<"symbolic", [Pure]> {
  let description = [{
    A symbol in CuTe IR represent an abstracted value which could be materialized
    later in a pass. If it's not materialized, the behavior is undefined.

    Example:

    ```mlir
    %0 = cute.symbolic("tile_m") : i32
    %1 = cute.symbolic("cluster_size") : !cute.shape<"(?,2)">
    ```
  }];
  let arguments = (ins
    StrAttr:$name
  );
  let results = (outs
    AnyType:$symbolic
  );
}
def Cute_TileToShapeOp : Cute_InferredOp<"tile_to_shape", [Pure]> {
  let description = [{
    The `tile_to_shape` operation is similar to blocked_product, but specifies the
    result shape instead of the product shape, which is more convenient in certain
    circumstances.

    The op has three operands:
    1. `block`: The layout to repeat.
    2. `trg_shape`: The target shape of the result
    3. `ord_shape`: The order of the modes of `trg_shape` to tile layout `block` with.
       E.g. (_2,_1,_3) will cause @a layout to repeat across the second mode first,
       the first mode second, and the third mode last. Currently only supports static order.

    Example:

    ```mlir
    %0 = cute.tile_to_shape(%block, %trg_shape, %ord_shape)
      : (!cute.layout<"(128,128):(128,1)">,
         !cute.shape<"(1024,512)">,
         !cute.shape<"(2,1)">)
      -> !cute.layout<"((128,8),(128,4)):((128,65536),(1,16384))">

    %0 = cute.tile_to_shape(%block, %trg_shape, %ord_shape)
      : (!cute.layout<"((32,4),(16,4)):((16,4),(0,1))">,
         !cute.shape<"(256,512,2)">,
         !cute.shape<"(1,2,3)">)
      -> !cute.layout<"((32,4,2),(16,4,8),2):((16,4,512),(0,1,1024),8192)">

    %0 = cute.tile_to_shape(%block, %trg_shape, %ord_shape)
      : (!cute.layout<"((32,4),(16,4)):((16,4),(0,1))">,
         !cute.shape<"(256,512,2)">,
         !cute.shape<"(2,1,3)">)
      -> !cute.layout<"((32,4,2),(16,4,8),2):((16,4,4096),(0,1,512),8192)">

    %0 = cute.tile_to_shape(%block, %trg_shape, %ord_shape)
      : (!cute.composed_layout<"S<3,4,3> o 0 o ((32,4),(?,4)):((16,4),(0,1))">,
         !cute.shape<"(256,?,2)">,
         !cute.shape<"(3,2,1)">)
      -> !cute.composed_layout<"S<3,4,3> o 0 o ((32,4,2),(?,4,?),2):((16,4,?{div=1024}),(0,1,1024),512)">
    ```
  }];
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$block,
    Cute_CuteShapeType:$trg_shape,
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$ord_shape
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_TiledCopyPartitionDOp : Cute_InferredOp<"tiled.copy.partition_D", [Pure]> {
  let description = [{
    The `cute.tiled.copy.partition_X` operation accepts:
        1. cute.tiled_copy;
        2. A ViewTypeInterface;
        3. cute.coord.
    returns a new ViewTypeInterface type (MemrefType/CoordTensorType) with partitioned layout
    and sliced offset (sliced offset will be implemented in the lowering).

    The new partitioned layout has the following meaning: ((vector_mode), rest_m, rest_n). vector_mode
    represents the number of instructions issued and the number of elements processed per
    instruction; in our example below, we have (2, 1) : (1, 0), meaning that each
    instruction is processing two data. Rest_m is the number of iterations
    along the first mode, a single iteration in our example. Finally, rest_n is the
    number of iterations along the second mode, 128.

    Example:

    ```mlir
      %dst_partitioned = cute.tiled.copy.partition_D(...)
        : (!cute.tiled_copy<!cute_nvgpu.atom.tmem_load<f32, 32 DP, 32 bit, x2>,
                            layout_copy_tv = <"((32,4),(2,32)):((0,1),(128,4))">,
                            tiler_mn = <"[(4,32):(32,1);2:1]">>,
           !cute.memref<f32, rmem, align<16>, "(128,256):(256,1)">,
           !cute.coord<"?">)
        -> !cute.memref<f32, rmem, align<16>, "((2,1),1,128):((1,0),0,2)">
    ```
  }];
  let arguments = (ins
    Cute_TiledCopyAtomTypeOrTiledCopyV2:$tiled_copy,
    Cute_ViewTypeInterfaceInstance:$input,
    Cute_CuteCoordType:$coord
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_TiledCopyPartitionSOp : Cute_InferredOp<"tiled.copy.partition_S", [Pure]> {
  let description = [{
    The `cute.tiled.copy.partition_X` operation accepts:
        1. cute.tiled_copy;
        2. A ViewTypeInterface;
        3. cute.coord.
    returns a new ViewTypeInterface type (MemrefType/CoordTensorType) with partitioned layout
    and sliced offset (sliced offset will be implemented in the lowering).

    The new partitioned layout has the following meaning: ((vector_mode), rest_m, rest_n). vector_mode
    represents the number of instructions issued and the number of elements processed per
    instruction; in our example below, we have (2, 1) : (1, 0), meaning that each
    instruction is processing two data. Rest_m is the number of iterations
    along the first mode, a single iteration in our example. Finally, rest_n is the
    number of iterations along the second mode, 128.

    Example:

    ```mlir
      %dst_partitioned = cute.tiled.copy.partition_D(...)
        : (!cute.tiled_copy<!cute_nvgpu.atom.tmem_load<f32, 32 DP, 32 bit, x2>,
                            layout_copy_tv = <"((32,4),(2,32)):((0,1),(128,4))">,
                            tiler_mn = <"[(4,32):(32,1);2:1]">>,
           !cute.memref<f32, rmem, align<16>, "(128,256):(256,1)">,
           !cute.coord<"?">)
        -> !cute.memref<f32, rmem, align<16>, "((2,1),1,128):((1,0),0,2)">
    ```
  }];
  let arguments = (ins
    Cute_TiledCopyAtomTypeOrTiledCopyV2:$tiled_copy,
    Cute_ViewTypeInterfaceInstance:$input,
    Cute_CuteCoordType:$coord
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_TiledCopyRetileOp : Cute_InferredOp<"tiled.copy.retile", [Pure]> {
  let description = [{
    The cute.tiled.copy.retile operation is used to reorganize the layout of a tensor
    that has already been tiled or partitioned. This is typically necessary when
    two TiledMma or TiledCopy instances are connected, accessing the same memory,
    but require different logical views of that memory.

    ### Use Case

    Consider a scenario where a TiledMma instance is the consumer, and a TiledCopy
    instance is the producer. The TiledMma instance partitions the memory, creating
    a logical view that suits its needs. However, this view may not be compatible with
    the TiledCopy instance. To resolve this, the cute.tiled.copy.retile operation is used
    to retile the TiledCopy instance, creating a new logical view that is compatible with
    the TiledMma instance.

    ```
        // consumer
        partition_A(%tiled_mma, ...)
        %mma_view = partition_A(%tiled_mma, %sa_tensor, %idx)

        // producer
        %tiled_copy = make_tiled_copy_A(%copy_atom, %tiled_mma)
        // retile %mma_view to meet needs of tiled_copy
        %lds_view = retile(%tiled_copy, %mma_view)
    ```

    ### Operation

    The cute.tiled.copy.retile operation takes two inputs:
      * A cute.tiled_copy instance, which represents the tensor to be retiled.
      * A CuTeTypeWithLayout derived type, which represents the tensor that has already been tiled or partitioned.


    Example:

    ```mlir
    %retiled = cute.tiled.copy.retile(%tiled_copy, %src_tensor)
      : (!cute.tiled_copy<!cute_nvgpu.atom.stsm<mode=<"(8,8)">, num_matrices=4>,
                          layout_copy_tv = <"((4,8,4),((2,2,4),1)):((128,4,1),((64,32,512),0))">,
                          tiler_mn = <"[(4,16):(32,1);32:1]">>,
         !cute.memref<f32, rmem, "(((2,2,4),1),2,1):(((1,2,4),0),16,0)">)
      -> !cute.memref<f32, rmem, "((8,2),2,1):((1,8),16,0)">
    ```
  }];
  let arguments = (ins
    Cute_TiledCopyAtomTypeOrTiledCopyV2:$tiled_copy,
    Cute_ViewTypeInterfaceInstance:$input
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_TiledDivideOp : Cute_InferredOp<"tiled_divide", [Pure]> {
  let summary = "This operation performs a tiled_divide on a view or a layout by a tiler.";
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$tiler
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_TiledMmaPartitionOp : Cute_InferredOp<"tiled.mma.partition", [Pure]> {
  let description = [{
    `cute.tiled.mma.partition` reorganizes a tensor's layout to match what the
    MMA atom expects. The operation takes 4 arguments:

    1. The type of operand. E.g., `CUTE_MMA_OPERAND_A` or `CUTE_MMA_OPERAND_B`.
    2. A `!cute.tiled_mma` value that specifies the kind of MMA.
    3. The unpartitioned operand tensor (ViewTypeInterface-derived SSA value).
    4. The `!cute.coord` value of the execution unit for which to partition the
       the input tensor.

    The coordinate type depends on the MMA atom and the operand memory space.
    Some examples:
    * SM80: thread ID
    * SM90: It depends...
      - When partitioning A with "frag_kind = ss": warp-group ID
      - When partitioning A with "frag_kind = rs": thread ID
      - When partitioning B: warp-group ID
      - When partitioning C/D: thread ID
    * SM100: CTA ID within the cluster (always 0 for 1-CTA mode)

    The operation returns a new ViewTypeInterface-derived value
    (CuteMemrefType/CuteCoordTensorType) with partitioned layout and sliced
    offset (sliced offset will be implemented in the lowering).

    Example:

    ```mlir
    // Partition an operand A tensor of shape 128x32 (residing in shared
    // memory) for a !tiled_mma atom that operates on shape 32x16x16. (I.e.,
    // four SM80 MMA atoms with shape 16x8x16.) The result tensor is a view
    // into the A tensor that represents the part that thread %tid_coord is
    // operating on.
    //
    // The last two modes of the result tensor are 4,2. This is because the
    // tiled atom is applied 4 times in M direction and 2 times in K direction.
    //
    //                  K=0:15                K=16:31
    //          ┌────────────────────┬────────────────────┐
    //  M=0:31  │  tiled atom (0,0)  │   tiled atom (0,1) │
    //          ├────────────────────┼────────────────────┤
    // M=32:63  │  tiled atom (1,0)  │   tiled atom (1,1) │
    //          ├────────────────────┼────────────────────┤
    // M=64:95  │  tiled atom (2,0)  │   tiled atom (2,1) │
    //          ├────────────────────┼────────────────────┤
    // M=96:127 │  tiled atom (3,0)  │   tiled atom (3,1) │
    //          └────────────────────┴────────────────────┘
    //
    // For each tiled atom application of size 32x16x16, thread %tid_coord gets
    // a subview of size 2,2,2 of operand A. (128 threads are cooperatively
    // computing the !tiled_mma of shape 32x16x16, but an individual thread
    // does not need access to the full 32x16 matrix A part.)
    //
    %partitioned = cute.tiled.mma.partition A(%0, %1, %tid_coord)
      : (!cute.tiled_mma<!cute_nvgpu.sm80.mma<16x8x16,
                                              elem_type = (f16,f16,f32)>,
                                              atom_layout_MNK = <"(2,2,1)">>,
         !cute.memref<f16, smem, align<16>, S<2,4,3>, "(128,32):(32,1)">,
         !cute.coord<"?">)
      -> !cute.memref<f16, smem, align<4>, S<2,4,3>, "((2,2,2),4,2):((1,256,8),1024,16)">

    // Partition an operand A tensor of shape 256x512 for an SM100 !tiled_mma.
    // There are 256/128=2 tiled atom applications in M direction and
    // 512/16=32 tiled atom applications in K direction. The atom specifies a
    // single CTA, so that CTA get the entire A tensor. The coordinate is not
    // really need in this case, but we pass it anyway to keep the API uniform.
    %partitioned = cute.tiled.mma.partition A (%0, %1, %cta_coord)
      : (!cute.tiled_mma<!cute_nvgpu.sm100.mma<128x128x16,
                                               num_cta = 1,
                                               ab_major = (k,mn),
                                               elem_type = (f16,f16,f16),
                                               frag_kind = ss,
                                               c_scale_exp = 0>,
                                               atom_layout_MNK = <"(1,1,1):(0,0,0)">>,
         !cute.memref<f16, generic, align<16>, "(256,512):(512,1)">,
         !cute.coord<"0">)
      -> !cute.memref<f16, generic, align<16>, "((128,16),2,32):((512,1),65536,16)">
    ```
  }];
  let arguments = (ins
    Cute_TiledMMAAtomType:$tiled_mma,
    Cute_ViewTypeInterfaceInstance:$input,
    Cute_CuteCoordType:$coord,
    Cute_MmaOperandEnum:$operand_id
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstance:$result
  );
}
def Cute_TiledMmaPartitionShapeOp : Cute_InferredOp<"tiled.mma.partition_shape", [Pure]> {
  let description = [{
    This operation partitions a shape into an MMA shape. For example, it takes a logical (M,K,...)
    shape and tiles it into a shape like (MMA,MMA_M,MMA_K,...) where
    - MMA is the MMA mode, for example (128,16) for an MMA instruction of shape 128x128x16
    - MMA_M is the number of times the MMA repeats in the M-mode
    - MMA_K is the number of times the MMA repeats in the K-mode
    - ... denotes the "rest"
  }];
  let arguments = (ins
    Cute_TiledMMAAtomType:$tiled_mma,
    Cute_CuteShapeType:$input,
    Cute_MmaOperandEnum:$operand_id
  );
  let results = (outs
    Cute_CuteShapeType:$result
  );
}
def Cute_TiledProductOp : Cute_InferredOp<"tiled_product", [Pure]> {
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuTeLayoutType:$tiler
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_ToIntTupleOp : Cute_InferredOp<"to_int_tuple", [Pure]> {
  let description = [{
    The `cute.cast_to_int_tuple` operation casts from any x-tuple type (shape, stride, coord) to an int-tuple type.
    The operation preserves the values but changes their type to int-tuple.

    cast will fail if input types contains modes not supported by int-tuple, e.g. _, scaled based, etc.

    Example:

    ```mlir
    // Cast from shape to int_tuple
    %0 = cute.to_int_tuple(%shape) : !cute.shape<"(8,4)"> to !cute.int_tuple<"(8,4)">

    // Cast from stride to int_tuple
    %1 = cute.to_int_tuple(%stride) : !cute.stride<"(8,4)"> to !cute.int_tuple<"(8,4)">

    // Cast from coord to int_tuple
    %2 = cute.to_int_tuple(%coord) : !cute.coord<"(8,4)"> to !cute.int_tuple<"(8,4)">

    // Cast from stride to int_tuple preserving divisibility
    %3 = cute.to_int_tuple(%stride) : !cute.stride<"(8,?{div=4})"> to !cute.int_tuple<"(8,?{div=4})">
    ```
  }];
  let arguments = (ins
    Cute_XTupleLikeType:$src
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS:$dst
  );
}
def Cute_TupleAddOp : Cute_InferredOp<"tuple_add", [Pure]> {
  let description = [{
    The `cute.tuple_add` operation performs element-wise addition between two CuTe x-tuple types.
    It accepts two input operands and returns a result of the same kind, where each element in the result
    is the sum of the corresponding elements in the input tuples.

    This operation applies subtraction recursively for nested tuples, matching corresponding elements
    at each level of the tuple hierarchy.

    Example:

    ```mlir
    // Element-wise add two arithmetic tuples
    %result = cute.tuple_add(%a, %b)
      : (!cute.int_tuple<"(?,(3,?))">, !cute.int_tuple<"(3,(1,?))">) -> !cute.int_tuple<"(?,(4,?))">

    // Element-wise add with static values
    %b = cute.static : !cute.int_tuple<"(3,(1,3))">
    %result = cute.tuple_add(%a, %b) : (!cute.int_tuple<"(?,(3,?))">) -> !cute.int_tuple<"(?,(4,?))">
    ```
  }];
  let arguments = (ins
    Cute_ArithmeticTupleType:$lhs,
    Cute_ArithmeticTupleType:$rhs
  );
  let results = (outs
    Cute_ArithmeticTupleType:$result
  );
}
def Cute_TupleDivOp : Cute_InferredOp<"tuple_div", [Pure]> {
  let description = [{
    The `cute.tuple_div` operation performs element-wise division between two CuTe x-tuple types.
    It accepts two input operands and returns a result of the same kind, where each element in the result
    is the quotient of the corresponding elements in the input tuples.

    This operation applies division recursively for nested tuples, matching corresponding elements
    at each level of the tuple hierarchy.

    Example:

    ```mlir
    // Element-wise divide two arithmetic tuples
    %result = cute.tuple_div(%a, %b)
      : (!cute.int_tuple<"(?,(3,?))">, !cute.int_tuple<"(3,(1,?))">) -> !cute.int_tuple<"(?,(3,?))">

    // Element-wise divide with static values
    %b = cute.static : !cute.int_tuple<"(3,(1,3))">
    %result = cute.tuple_div(%a, %b) : (!cute.int_tuple<"(?,(3,?))">) -> !cute.int_tuple<"(?,(0,?))">
    ```
  }];
  let arguments = (ins
    Cute_ArithmeticTupleType:$lhs,
    Cute_ArithmeticTupleType:$rhs
  );
  let results = (outs
    Cute_ArithmeticTupleType:$result
  );
}
def Cute_TupleModOp : Cute_InferredOp<"tuple_mod", [Pure]> {
  let description = [{
    The `cute.tuple_mod` operation computes the element-wise modulo (remainder)
    between two CuTe x-tuple types. Both operands must be of the same tuple kind
    (e.g., `!cute.int_tuple`), and the result is a tuple of the same structure
    and type.

    For each corresponding element in the input tuples, the operation computes
    the remainder of the division of the left-hand side element by the right-hand
    side element. If the tuples are nested, the operation is applied recursively,
    matching elements at each level of the tuple hierarchy.

    This operation supports both static and dynamic tuple elements. When static
    values are provided, the modulo is computed at compile time; otherwise, it is
    performed at runtime.

    **Examples:**

    ```mlir
    // Element-wise modulo of two arithmetic tuples
    %result = cute.tuple_mod(%a, %b)
      : (!cute.int_tuple<"(?,(3,?))">, !cute.int_tuple<"(3,(1,?))">)
        -> !cute.int_tuple<"(?,(0,?))">

    // Element-wise modulo with static values
    %b = cute.static : !cute.int_tuple<"(3,(1,3))">
    %result = cute.tuple_mod(%a, %b)
      : (!cute.int_tuple<"(?,(3,?))">) -> !cute.int_tuple<"(?,(1,?))">
    ```

    **Semantics:**
    - The operation traverses the tuple structure recursively.
    - For each pair of corresponding elements `(x, y)`, the result is `x % y`.
    - The structure and rank of the result tuple matches that of the operands.
    - If the tuple contains nested tuples, the modulo is applied to each
      corresponding sub-tuple.

    This operation is useful for broadcasting, tiling, and other tensor algebra
    applications where modular arithmetic over structured tuples is required.
  }];
  let arguments = (ins
    Cute_ArithmeticTupleType:$lhs,
    Cute_ArithmeticTupleType:$rhs
  );
  let results = (outs
    Cute_ArithmeticTupleType:$result
  );
}
def Cute_TupleMulOp : Cute_InferredOp<"tuple_mul", [Pure]> {
  let description = [{
    The `cute.tuple_mul` operation performs element-wise multiplication between two CuTe x-tuple types.
    It accepts two input operands and returns a result of the same kind, where each element in the result
    is the product of the corresponding elements in the input tuples.

    This operation applies multiplication recursively for nested tuples, matching corresponding elements
    at each level of the tuple hierarchy.

    Example:

    ```mlir
    // Element-wise multiply two arithmetic tuples
    %result = cute.tuple_mul(%a, %b)
      : (!cute.int_tuple<"(?,(3,?))">, !cute.int_tuple<"(3,(1,?))">) -> !cute.int_tuple<"(?,(3,?))">

    // Element-wise multiply with static values
    %b = cute.static : !cute.int_tuple<"(3,(1,3))">
    %result = cute.tuple_mul(%a, %b) : (!cute.int_tuple<"(?,(3,?))">) -> !cute.int_tuple<"(?,(3,?))">
    ```
  }];
  let arguments = (ins
    Cute_ArithmeticTupleType:$lhs,
    Cute_ArithmeticTupleType:$rhs
  );
  let results = (outs
    Cute_ArithmeticTupleType:$result
  );
}
def Cute_TupleProductEachOp : Cute_InferredOp<"tuple.product_each", [Pure]> {
  let summary = "This operation performs \"product_each\" on each element in a tuple";
  let description = [{
    Example:

    ```mlir
    %tuple = cute.make_int_tuple() : !cute.int_tuple<"(2,(5,4))">
    %prod = cute.tuple.product_each(%tuple) : (!cute.int_tuple<"(2,(5,4))">) -> !cute.int_tuple<"(2, 20)">
    ```
  }];
  let arguments = (ins
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS2:$input
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS2:$result
  );
}
def Cute_TupleProductOp : Cute_InferredOp<"tuple.product", [Pure]> {
  let summary = "This operation performs \"product\" on all elements in a tuple";
  let description = [{
    Example:

    ```mlir
    %tuple = cute.make_int_tuple() : !cute.int_tuple<"(2,(5,4))">
    %prod = cute.product(%tuple) : (!cute.int_tuple<"(2,(5,4))">) -> !cute.int_tuple<"40">
    ```
  }];
  let arguments = (ins
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS2:$input
  );
  let results = (outs
    Cute_IntTupleAsIntegerOrTupleOfIntTupleS2:$result
  );
}
def Cute_TupleSubOp : Cute_InferredOp<"tuple_sub", [Pure]> {
  let description = [{
    The `cute.tuple_sub` operation performs element-wise subtraction between two CuTe x-tuple types.
    It accepts two input operands and returns a result of the same kind, where each element in the result
    is the difference of the corresponding elements in the input tuples.

    This operation applies subtraction recursively for nested tuples, matching corresponding elements
    at each level of the tuple hierarchy.

    Example:

    ```mlir
    // Element-wise subtract two arithmetic tuples
    %result = cute.tuple_sub(%a, %b)
      : (!cute.int_tuple<"(?,(3,?))">, !cute.int_tuple<"(3,(1,?))">) -> !cute.int_tuple<"(?,(2,?))">

    // Element-wise subtract with static values
    %b = cute.static : !cute.int_tuple<"(3,(1,3))">
    %result = cute.tuple_sub(%a, %b) : (!cute.int_tuple<"(?,(3,?))">) -> !cute.int_tuple<"(?,(2,?))">
    ```
  }];
  let arguments = (ins
    Cute_ArithmeticTupleType:$lhs,
    Cute_ArithmeticTupleType:$rhs
  );
  let results = (outs
    Cute_ArithmeticTupleType:$result
  );
}
def Cute_UnpackTupleOp : Cute_Op<"unpack_tuple", [Pure]> {
  let description = [{
    Unpacks a tuple into its individual components. The number and types
    of the components must match the number and types of the results.
  }];
  let arguments = (ins
    Cute_ATupleIsAFixedLengthOrderedList:$tuple
  );
  let results = (outs
    Variadic<AnyType>:$result
  );
}
def Cute_ZippedDivideOp : Cute_InferredOp<"zipped_divide", [Pure]> {
  let description = [{
    This operation performs "zipped_divide" on the input and produce a zipped_divided result,
    and it automatically infer the "zipped_divided" layout for the result type.

    Example:

    ```mlir
    %alloc = cute.memref.alloc() : !cute.memref<f16, generic, "(12,32,6):(1,128,0)">
    %shape = cute.make_shape() : () -> !cute.shape<"(4,8)">
    %zdiv = cute.zipped_divide(%alloc, %shape)
      : !cute.memref<f16, generic, "(12,32,6):(1,128,0)">, !cute.shape<"(4,8)">
    ```
  }];
  let arguments = (ins
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$input,
    Cute_TileLikeTypesEGLayoutTypeShapeTypeTileType:$tiler
  );
  let results = (outs
    Cute_ViewTypeInterfaceInstanceOrGeneralizedLayoutTypeOfCuTe:$result
  );
}
def Cute_ZippedProductOp : Cute_InferredOp<"zipped_product", [Pure]> {
  let arguments = (ins
    Cute_GeneralizedLayoutTypeOfCuTe:$input,
    Cute_CuTeLayoutType:$tiler
  );
  let results = (outs
    Cute_GeneralizedLayoutTypeOfCuTe:$result
  );
}

#endif // CUTE_IR_GENERATED_CUTE_OPS_TD
