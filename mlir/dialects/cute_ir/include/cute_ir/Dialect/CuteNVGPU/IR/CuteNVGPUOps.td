// The cute_nvgpu dialect's operations, generated from the nvidia-cutlass-dsl 4.8.0 release; do not edit.
//
// Sources: the operation bindings (`_cute_nvgpu_ops_gen.py`: names, classes,
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
//   cute_nvgpu.arch.mma.SM107.umma_block_scaled operand #3: no probe type accepted; any type allowed
//   cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse operand #3: no probe type accepted; any type allowed
//   cute_nvgpu.arch.mma.SM107.umma_sparse operand #3: no probe type accepted; any type allowed

#ifndef CUTE_IR_GENERATED_CUTENVGPU_OPS_TD
#define CUTE_IR_GENERATED_CUTENVGPU_OPS_TD

include "cute_ir/Dialect/CuteNVGPU/IR/CuteNVGPUAttrs.td"
include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"
include "cute_ir/Dialect/Cute/IR/CuteEnums.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Operand and result constraints, in the compiler's words.
def CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "A memory reference type that represents a tensor view with an iterator and layout">;
def CuteNVGPU_AllowableDescriptorTypesForS2TSrcOperand : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescType>($_self))">, "allowable descriptor types for S2T src operand">;
def CuteNVGPU_AllowableDescriptorTypesForSM107BlockscaledUMMAA : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self))">, "allowable descriptor types for SM107 Blockscaled UMMA A operand">, BuildableType<"::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type::get($_builder.getContext())">;
def CuteNVGPU_AllowableDescriptorTypesForSM107UMMAAOperand : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self))">, "allowable descriptor types for SM107 UMMA A operand">, BuildableType<"::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type::get($_builder.getContext())">;
def CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA : Type<CPred<"($_self.isSignlessInteger(32))">, "allowable register types for SM120 block scaled MMA operations">;
def CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA2 : Type<CPred<"(::llvm::isa<::mlir::FloatType>($_self))">, "allowable register types for SM120 block scaled MMA operations">;
def CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA3 : Type<CPred<"($_self.isSignlessInteger(8) || $_self.isSignlessInteger(16) || $_self.isSignlessInteger(32))">, "allowable register types for SM120 block scaled MMA operations">;
def CuteNVGPU_AtomTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::TiledMmaType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM120BlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomLdsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomS2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTAsyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemStType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100S2TCopyV2Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomStsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM10xTmemLoadRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTSyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::UniversalFmaAtomType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM100UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMAType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMABlockScaledSparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM107UMMASparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM80SparseType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM89Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::MmaAtomSM90Type>($_self))">, "AtomTypeInterface instance">;
def CuteNVGPU_CopyAtomTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyG2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomBulkCopyS2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomDsmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomG2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomLdsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2GType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomR2SType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomS2RType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTAsyncCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemLdReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTMultimemStType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomStsmType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM10xTmemLoadRedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM107TmemLoadSPCompressType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100TmemStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSIMTSyncCopyType>($_self))">, "CopyAtomTypeInterface instance">;
def CuteNVGPU_CuTeComposedLayoutType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self))">, "CuTe composed layout type">;
def CuteNVGPU_CuTeLayoutType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self))">, "CuTe Layout Type">;
def CuteNVGPU_CuTeMemrefToMemorySpacesSmem : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self))">, "CuTe memref to memory spaces (smem)">;
def CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to allowable types for SM100 UMMA C/D operands in spaces (tmem)">;
def CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA2 : Type<CPred<"true">, "CuTe pointer to allowable types for SM100 UMMA E operand in spaces (tmem)">;  // not recovered: no pool type passed
def CuteNVGPU_CuTePointerToAllowableTypesForSM107UMMA : Type<CPred<"true">, "CuTe pointer to allowable types for SM107 UMMA SF A/B operands in spaces (tmem)">;  // not recovered: no pool type passed
def CuteNVGPU_CuTePointerToAllowableTypesForSM90MMA : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to allowable types for SM90 MMA A/B operands in spaces (smem) with byte alignment 1">;
def CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to global memory with 64-byte alignment">;
def CuteNVGPU_CuTePointerToGlobalMemoryWithByteAlignment : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to global memory with byte alignment 1">;
def CuteNVGPU_CuTePointerToMemorySpacesRmem : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to memory spaces (rmem)">;
def CuteNVGPU_CuTePointerToMemorySpacesSmem : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to memory spaces (smem)">;
def CuteNVGPU_CuTePointerToMemorySpacesSmemOrCuTe : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to memory spaces (smem) or CuTe pointer to memory spaces (gmem)">;
def CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to memory spaces (smem) with byte alignment 1">;
def CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to memory spaces (smem) with byte alignment 128">;
def CuteNVGPU_CuTePointerToMemorySpacesTmem : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to memory spaces (tmem)">;
def CuteNVGPU_CuTePointerToSharedMemoryWith8Byte : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to shared memory with 8-byte alignment">;
def CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to shared memory with swizzle base atom compatible alignment with byte alignment 1">;
def CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to shared memory with swizzle base atom compatible alignment">;
def CuteNVGPU_CuTePointerToSharedOrDistributedSharedMemory : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "CuTe pointer to shared or distributed shared memory with 8-byte alignment">;
def CuteNVGPU_CuTePointerType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, " CuTe pointer type ">;
def CuteNVGPU_CuteCoordTensorType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self))">, "Cute coord_tensor type">;
def CuteNVGPU_CuteCoordType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordType>($_self))">, "Cute coord type">;
def CuteNVGPU_CuteShapeType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ShapeType>($_self))">, "Cute shape type">;
def CuteNVGPU_CuteSm107SmemDescType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type>($_self))">, "Cute sm107.smem_desc type">, BuildableType<"::mlir::cutlass_compiler::cute_nvgpu::SmemDescSM107Type::get($_builder.getContext())">;
def CuteNVGPU_CuteSmemDescType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescType>($_self))">, "Cute smem_desc type">, BuildableType<"::mlir::cutlass_compiler::cute_nvgpu::SmemDescType::get($_builder.getContext())">;
def CuteNVGPU_CuteSmemDescViewType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "Cute smem_desc_view type">;
def CuteNVGPU_GeneralizedLayoutTypeOfCuTe : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::ComposedLayoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self))">, " Generalized layout type of CuTe ">;
def CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::IntTupleType>($_self))">, "IntTuple as integer or tuple of IntTuple(s)">;
def CuteNVGPU_Integer : Type<CPred<"(::llvm::isa<::mlir::IntegerType>($_self))">, "integer">;
def CuteNVGPU_SM100BlackwellSharedToTensorMemoryCopyAtom : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self))">, "SM100 (Blackwell) shared-to-tensor memory copy atom type via tcgen05.cp.">;
def CuteNVGPU_SM100BlackwellSharedToTensorMemoryCopyAtom2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomSM100CopyS2TType>($_self))">, "SM100 (Blackwell) shared-to-tensor memory copy atom type via tcgen05.cp. or Tiled copy atom type.">;
def CuteNVGPU_SM90PointerToTMATensorDescriptor : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::PtrType>($_self))">, "SM90+ pointer to TMA tensor descriptor">;
def CuteNVGPU_SM90TMATensorMemoryAcceleratorLoadAtomType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomTmaStoreType>($_self))">, "SM90+ TMA (Tensor Memory Accelerator) load atom type via cp.async.bulk. or SM90+ TMA im2col mode load atom type via cp.async.bulk.tensor. or SM90+ TMA (Tensor Memory Accelerator) store atom type via cp.async.bulk. or SM90+ TMA im2col mode store atom type via cp.async.bulk.tensor. or SM90+ TMA store with atomic reduction atom type via cp.reduce.async.bulk. or Type interface for non-executable TMA atom types">;
def CuteNVGPU_SignlessIntegerOrIndexOrCuTeLayoutType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::LayoutType>($_self) || ::llvm::isa<::mlir::IntegerType>($_self) || ::llvm::isa<::mlir::IndexType>($_self))">, "signless integer or index or CuTe Layout Type">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self))">, "The non-executable copy atom type for TMA loads in 2D gather4 mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA2 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self))">, "The non-executable copy atom type for TMA stores in 2D scatter4 mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA3 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self))">, "The non-executable copy atom type for TMA loads in im2col mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA4 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self))">, "The non-executable copy atom type for TMA stores in im2col mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA5 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self))">, "The non-executable copy atom type for TMA loads in tiled mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA6 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self))">, "The non-executable copy atom type for TMA reduce in tiled mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA7 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self))">, "The non-executable copy atom type for TMA stores in tiled mode">;
def CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA8 : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self))">, "The non-executable copy atom type for TMA loads in tiled mode or The non-executable copy atom type for TMA stores in tiled mode">;
def CuteNVGPU_TiledCopyAtomType : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::TiledCopyType>($_self))">, "Tiled copy atom type.">;
def CuteNVGPU_TypeInterfaceForNonExecutableTMAAtomTypes : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DGather4TmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExec2DScatter4TmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecIm2ColTmaStoreType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaLoadType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaReduceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::CopyAtomNonExecTiledTmaStoreType>($_self))">, "Type interface for non-executable TMA atom types">;
def CuteNVGPU_VectorOf32BitSignlessIntegerValuesOf : Type<CPred<"(::llvm::isa<::mlir::VectorType>($_self))">, "vector of 32-bit signless integer values of length 1/2/4/8/16/32/64/128">;
def CuteNVGPU_VectorOf32BitSignlessIntegerValuesOf2 : Type<CPred<"(::llvm::isa<::mlir::VectorType>($_self))">, "vector of 32-bit signless integer values of length 4/8">;
def CuteNVGPU_ViewTypeInterfaceInstance : Type<CPred<"(::llvm::isa<::mlir::cutlass_compiler::cute::CoordTensorType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute::MemRefType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::SmemDescViewType>($_self))">, "ViewTypeInterface instance">;

// Attribute constraints the bindings name but do not describe.
def CuteNVGPU_ArchCopySM100CopyS2TBitAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{128, 256}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_ArchCopySM100CopyS2TCtaAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{1, 2}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_ArchCopySM100CopyS2TDpAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{4, 32, 64, 128}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_ArchCopySM100TMEMNumBAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{32, 64, 128, 256}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_ArchCopySM100TMEMNumDpAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{16, 32}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_ArchCopySM100TMEMNumRepAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{1, 2, 4, 8, 16, 32, 64, 128}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_ArchCopySM100TmaLoadOpNumCTAAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self) && ::llvm::is_contained(::llvm::ArrayRef<int64_t>{1, 2}, ::llvm::cast<::mlir::IntegerAttr>($_self).getInt())">, "arbitrary integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_SM120BlockScaledABTypeAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float8E5M2Type>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float4E2M1FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float6E2M3FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float6E3M2FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of allowable types for SM120 block scaled MMA A/B operands"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_SM120BlockScaledSFTypeAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::Float8E4M3FNType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()) || ::llvm::isa<::mlir::Float8E8M0FNUType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of allowable types for SM120 block scaled MMA A/B scale factors"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}
def CuteNVGPU_TmaLoadOrStoreModeAttr : Attr<CPred<"::llvm::isa<::mlir::cutlass_compiler::cute_nvgpu::TmaLoadModeAttr, ::mlir::cutlass_compiler::cute_nvgpu::TmaStoreModeAttr>($_self)">, "TMA load or store mode attribute"> {
  let storageType = "::mlir::Attribute";
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
}

// Operations whose result types are inferred in C++. Verification compares
// them with the inferred ones where the rule is recovered for the operands;
// elsewhere the types written stand, as the compiler accepted them.
class CuteNVGPU_InferredOp<string mnemonic, list<Trait> traits = []>
    : CuteNVGPU_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferTypeOpInterface, ["refineReturnTypes"]>]> {
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

def CuteNVGPU_ArchAllocRmemOp : CuteNVGPU_Op<"arch.alloc_rmem", [MemoryEffects<[MemAlloc]>]> {
  let description = [{
    This operation allocate a per-thread private buffer on the stack, with its size
    determined by either the provided element count or the static layout.

    Examples:

    ```mlir
    %c32_i32 = arith.constant 32 : i32
    %0 = cute_nvgpu.arch.alloc_rmem(%c32_i32) : i32 -> !cute.ptr<f32, rmem>

    // %static_layout : !cute.layout<"(8,4):(1,8)">
    %1 = cute_nvgpu.arch.alloc_rmem(%static_layout) : !cute.layout<"(8,4):(1,8)"> -> !cute.ptr<f32, rmem>

    // %dynamic_layout : !cute.layout<"(?,4):(1,?)">
    %2 = cute_nvgpu.arch.alloc_rmem(%dynamic_layout) : !cute.layout<"(?,4):(1,?)"> -> !cute.ptr<f32, rmem>
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SignlessIntegerOrIndexOrCuTeLayoutType:$input
  );
  let results = (outs
    CuteNVGPU_CuTePointerToMemorySpacesRmem:$ptr
  );
}
def CuteNVGPU_ArchAllocSmemOp : CuteNVGPU_Op<"arch.alloc_smem", [MemoryEffects<[MemAlloc]>]> {
  let description = [{
    This operation allocate static shared memory based on either a specified
    element count or a static layout.

    Examples:

    ```mlir
    // size provided by element count
    %0 = cute_nvgpu.arch.alloc_smem(1024) : !cute.ptr<f16, smem, align<32>>

    // size provided by static layout
    %1 = cute_nvgpu.arch.alloc_smem("(8,4):(1,8)") : !cute.ptr<f16, smem, align<32>>
    ```
  }];
  let arguments = (ins
    I32Attr:$input
  );
  let results = (outs
    CuteNVGPU_CuTePointerToMemorySpacesSmem:$ptr
  );
}
def CuteNVGPU_ArchCopySM90BulkCopyG2SOp : CuteNVGPU_Op<"arch.copy.SM90.bulk_copy_g2s", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM90.bulk_copy_g2s` operation performs a bulk load from global
    to shared memory.

    The `gmem_data_addr` operand is a 16-byte aligned pointer to the source data
    in global memory.

    The `dsmem_data_addr` operand is a 16-byte aligned pointer to the destination data
    in shared cluster memory.

    The `dsmem_bar_addr` operand is a 8-byte aligned pointer to the barrier in shared
    memory.

    The `size` operand specifies the bytes copied by BLKCP, aligned to 16 bytes.

    The `multicast_mask` operand is used when multicasting.

    The `cache_policy` operand is used to indicate the eviction priority of the L2 cache.

    Examples:
    ```mlir
    %gmem = ... : !cute.ptr<f32, gmem, align<16>>
    %smem = ... : !cute.ptr<f32, smem, align<16>>
    %bar  = ... : !cute.ptr<i64, smem, align<8>>

    // Without multicast mask
    %0 = cute_nvgpu.arch.copy.SM90.bulk_copy_g2s(
      %gmem : !cute.ptr<f32, gmem, align<16>>,
      %smem : !cute.ptr<f32, smem, align<16>>,
      %bar  : !cute.ptr<i64, smem, align<8>>
    ) size = 128

    // With multicast mask
    %1 = cute_nvgpu.arch.copy.SM90.bulk_copy_g2s(
      %gmem : !cute.ptr<f32, gmem, align<16>>,
      %smem : !cute.ptr<f32, smem, align<16>>,
      %bar  : !cute.ptr<i64, smem, align<8>>
    ) size = 128 mask = %mask
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToGlobalMemoryWithByteAlignment:$gmem_data_addr,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase:$dsmem_data_addr,
    CuteNVGPU_CuTePointerToSharedMemoryWith8Byte:$dsmem_bar_addr,
    Optional<I16>:$multicast_mask,
    Optional<I64>:$cache_policy,
    I32Attr:$size
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM90BulkCopyS2GOp : CuteNVGPU_Op<"arch.copy.SM90.bulk_copy_s2g", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM90.bulk_copy_s2g` operation performs a bulk load from shared
    to global memory.

    The `smem_data_addr` operand is a 16-byte aligned pointer to the sourced data
    in shared memory.

    The `gmem_data_addr` operand is a 16-byte aligned pointer to the estination data
    in global memory.

    The `size` operand specifies the bytes copied by BLKCP, aligned to 16 bytes.

    The `byte_mask` operand is specifies whether the i-th byte of each 16-byte wide
    chunk of source data is copied to the destination.

    The `cache_policy` operand is used to indicate the eviction priority of the L2 cache.

    Examples:
    ```mlir
    %smem = ... : !cute.ptr<f32, smem, align<16>>
    %gmem = ... : !cute.ptr<f32, gmem, align<16>>

    // Without byte mask
    %0 = cute_nvgpu.arch.copy.SM90.bulk_copy_s2g(
      %smem : !cute.ptr<f32, smem, align<16>>,
      %gmem : !cute.ptr<f32, gmem, align<16>>,
    ) size = 128

    // With byte mask
    %1 = cute_nvgpu.arch.copy.SM90.bulk_copy_s2g(
      %smem : !cute.ptr<f32, smem, align<16>>,
      %gmem : !cute.ptr<f32, gmem, align<16>>,
    ) size = 128 byte_mask = %mask
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase:$smem_data_addr,
    CuteNVGPU_CuTePointerToGlobalMemoryWithByteAlignment:$gmem_data_addr,
    Optional<I16>:$byte_mask,
    Optional<I64>:$cache_policy,
    I32Attr:$size
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM90BulkCopyS2SOp : CuteNVGPU_Op<"arch.copy.SM90.bulk_copy_s2s"> {
  let description = [{
    The `arch.copy.SM90.bulk_copy_s2s` operation performs a bulk load from shared
    cta to shared cluster.

    The `smem_data_addr` operand is a 16-byte aligned pointer to the source data 
    in shared memory.

    The `dsmem_data_addr` operand is a 16-byte aligned pointer to the destination data
    in shared cluster memory.

    The `dsmem_bar_addr` operand is a 8-byte aligned pointer to the barrier in shared
    memory.

    The `cta_rank` operand specifies the dst cta rank in cluster for BLKCP.

    The `size` attr specifies the bytes copied by BLKCP, aligned to 16 bytes.

    Examples:
    ```mlir
    %smem  = ... : !cute.ptr<f32, smem, align<16>>
    %dsmem = ... : !cute.ptr<f32, smem, align<16>>
    %bar   = ... : !cute.ptr<i64, smem, align<8>>
    %cta   = ... : i32

    %0 = cute_nvgpu.arch.copy.SM90.bulk_copy_s2s(
      %smem  : !cute.ptr<f32, smem, align<16>>,
      %dsmem : !cute.ptr<f32, smem, align<16>>,
      %bar   : !cute.ptr<i64, smem, align<8>>,
      %cta   : i32
    ) size = 128
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase:$smem_data_addr,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase:$dsmem_data_addr,
    CuteNVGPU_CuTePointerToSharedMemoryWith8Byte:$dsmem_bar_addr,
    I32:$cta_rank,
    I32Attr:$size
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM100CopyS2TOp : CuteNVGPU_Op<"arch.copy.SM100.copy_s2t"> {
  let description = [{
    See the file cutlass/include/cute/arch/copy_sm100.hpp in the CUTLASS
    repository for more information.
  }];
  let arguments = (ins
    CuteNVGPU_AllowableDescriptorTypesForS2TSrcOperand:$src,
    CuteNVGPU_CuTePointerToMemorySpacesTmem:$dst,
    CuteNVGPU_ArchCopySM100CopyS2TDpAttr:$dp,
    CuteNVGPU_ArchCopySM100CopyS2TBitAttr:$bits,
    CuteNVGPU_ArchCopySM100CopyS2TCtaAttr:$cta,
    CuteNVGPU_CopyS2TBroadcastAttr:$broadcast
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM100TmaLoadOp : CuteNVGPU_Op<"arch.copy.SM100.tma_load", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM100.tma_load` operation performs a tensor load from global
    to shared memory.

    The attribute `mode` indicates which mode is selected. It can be:

    - tiled
    - im2col for image-to-column loads
    - w128 for activation-stationary convolution MMA mode (requires multicast)
    - w for weight-stationary convolution MMA mode (requires multicast)
    - gather4 for gather mode

    The attribute `num_cta` specifies the scope of the instruction.

    The `src_desc` operand is a 64-byte aligned pointer to the source tensor in
    global memory.

    The `dsmem_data_addr` is a 128-byte aligned pointer to the destination data
    address in shared memory.

    The `dsmem_bar_addr` is an 8-byte aligned pointer to the barrier in shared
    or distributed shared memory.

    The `coord` operand holds the tensor coordinates (up to 5 dimensions).

    The `multicast_mask` operand is used when multicasting.

    The `offsets` operand holds offsets for im2col, w128, or w modes.

    The `cache_policy` operand is used to indicate the eviction priority of the L2 cache.
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte:$src_desc,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase:$dsmem_data_addr,
    CuteNVGPU_CuTePointerToSharedOrDistributedSharedMemory:$dsmem_bar_addr,
    Variadic<I32>:$coord,
    Optional<I16>:$multicast_mask,
    Variadic<I16>:$offsets,
    Optional<I64>:$cache_policy,
    CuteNVGPU_TmaLoadModeAttr:$mode,
    CuteNVGPU_ArchCopySM100TmaLoadOpNumCTAAttr:$num_cta
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM100TmaReduceOp : CuteNVGPU_Op<"arch.copy.SM100.tma_reduce", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM100.tma_reduce` operation performs a tensor store from shared memory
    to global with reduction.

    The attribute `mode` indicates which mode is selected. It can be:

    - tiled
    - im2col for image-to-column loads

    The attribute `Kind` indicates which kind of reduce op is selected. It can be:

    - ADD
    - MIN
    - MAX
    - INC
    - DEC
    - AND
    - OR
    - XOR

    The `dst_desc` is a 64-byte aligned pointer to the destination data
    address in global memory.

    The `src_smem_data_addr` operand is a 128-byte aligned pointer to the source address in
    shared memory.

    The `coord` operand holds the tensor coordinates (up to 5 dimensions).

    The `cache_policy` operand is used to indicate the eviction priority of the L2 cache.
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte:$dst_desc,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase2:$src_smem_data_addr,
    Variadic<I32>:$coord,
    Optional<I64>:$cache_policy,
    CuteNVGPU_TmaStoreModeAttr:$mode,
    CuteNVGPU_ReductionKindAttr:$kind
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM100TmaStoreOp : CuteNVGPU_Op<"arch.copy.SM100.tma_store", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM100.tma_store` operation performs a tensor store from shared memory
    to global.

    The attribute `mode` indicates which mode is selected. It can be:

    - tiled
    - im2col for image-to-column loads
    - scatter4 for scatter mode

    The `dst_desc` is a 64-byte aligned pointer to the destination data
    address in global memory.

    The `src_smem_data_addr` operand is a 128-byte aligned pointer to the source address in
    shared memory.

    The `coord` operand holds the tensor coordinates (up to 5 dimensions).

    The `cache_policy` operand is used to indicate the eviction priority of the L2 cache.
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte:$dst_desc,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase2:$src_smem_data_addr,
    Variadic<I32>:$coord,
    Optional<I64>:$cache_policy,
    CuteNVGPU_TmaStoreModeAttr:$mode
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM100TmemLoadOp : CuteNVGPU_Op<"arch.copy.SM100.tmem_load", [MemoryEffects<[MemRead]>]> {
  let description = [{
    See the file cutlass/include/cute/arch/copy_sm100.hpp in the CUTLASS
    repository for more information.
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesTmem:$src,
    CuteNVGPU_ArchCopySM100TMEMNumDpAttr:$num_dp,
    CuteNVGPU_ArchCopySM100TMEMNumBAttr:$num_b,
    CuteNVGPU_ArchCopySM100TMEMNumRepAttr:$num_rep,
    UnitAttr:$pack_16
  );
  let results = (outs
    CuteNVGPU_VectorOf32BitSignlessIntegerValuesOf:$dst
  );
}
def CuteNVGPU_ArchCopySM100TmemStoreOp : CuteNVGPU_Op<"arch.copy.SM100.tmem_store"> {
  let description = [{
    See the file cutlass/include/cute/arch/copy_sm100.hpp in the CUTLASS
    repository for more information.
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesTmem:$dst,
    CuteNVGPU_VectorOf32BitSignlessIntegerValuesOf:$src,
    CuteNVGPU_ArchCopySM100TMEMNumDpAttr:$num_dp,
    CuteNVGPU_ArchCopySM100TMEMNumBAttr:$num_b,
    CuteNVGPU_ArchCopySM100TMEMNumRepAttr:$num_rep,
    UnitAttr:$expand_16
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM107TmaLoadOverrideOp : CuteNVGPU_Op<"arch.copy.SM107.tma_load_override", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM107.tma_load_override` operation performs a TMA tensor load
    from global to shared memory with runtime override of address, dimensions,
    and strides.

    In addition to the normal TMA load operands, this op carries:
    - `override_addr`: global pointer overriding the descriptor's base address
    - `tensor_size`: per-mode sizes (encoded as size-1, i16)
    - `lower_stride`: per-mode strides in 16B units (i32, modes 1..N-1)
    - `upper_stride`: packed upper 4 bits of up to 4 strides (i16)

    Coordinates are passed as zeros (override form ignores them).

    Example:

    Suppose we want to load a 2D tile from a tensor with logical sizes
    `{S0, S1}` and per-mode element strides `{1, ES1}` (mode 0 is contiguous,
    so its stride is not passed). Let `EBITS` be the value type's bit-width
    and `elemsPer16B = 128 / EBITS`. Mode 0's size is in `tensor_size[0]`,
    mode 1's size is in `tensor_size[1]`, and mode 1's stride is split between
    `lower_stride[0]` and the nibble at bits `[0:4]` of `upper_stride`.

    The encoding (matching `CopyAtomTmaLoadType::lowerOverrideToArch`) is:

    ```mlir
    // ---- tensor_size: trunc<i16>(S_i - 1), one entry per mode ----
    %s0_m1 = arith.subi  %S0_i32, %c1_i32 : i32          // S0 - 1
    %s1_m1 = arith.subi  %S1_i32, %c1_i32 : i32          // S1 - 1
    %ts0   = arith.trunci %s0_m1 : i32 to i16
    %ts1   = arith.trunci %s1_m1 : i32 to i16

    // ---- stride in 16B units: ES_i (element stride, i64) >> log2(elemsPer16B)
    %sh    = arith.constant log2(elemsPer16B) : i64
    %s1_16b = arith.shrui %ES1_i64, %sh : i64            // mode 1, stride in 16B

    // ---- lower_stride[idx]: low 32 bits of stride_in_16B (idx = mode - 1) ----
    %ls1   = arith.trunci %s1_16b : i64 to i32           // idx = 0 (mode 1)

    // ---- upper_stride: packed nibbles, OR-accumulated, truncated to i16 ----
    //   nibble_i = (stride_in_16B[mode i] >> 32) & 0xF
    //   upper   |= nibble_i << (4 * (i - 1))
    %c32   = arith.constant 32 : i64
    %c0xF  = arith.constant 15 : i64
    %hi_1  = arith.shrui %s1_16b, %c32  : i64            // upper bits of mode 1
    %nib_1 = arith.andi  %hi_1,   %c0xF : i64
    %sh_1  = arith.constant 0 : i64                      // 4 * (1 - 1) = 0
    %shf_1 = arith.shli  %nib_1,  %sh_1 : i64
    %acc   = arith.ori   %c0_i64, %shf_1 : i64           // accumulate over modes
    %us    = arith.trunci %acc : i64 to i16

    // ---- the override TMA load itself ----
    cute_nvgpu.arch.copy.SM107.tma_load_override(
        %tma_desc      : !cute.ptr<generic, align<64>>,    // src TMA descriptor
        %dst_smem      : !cute.ptr<i32, smem, align<128>>, // dst data buffer
        %dst_bar       : !cute.ptr<smem, align<8>>,        // mbarrier
        %override_addr : !cute.ptr<i32, gmem, align<16>>,  // runtime base addr override
        [%c0_i32, %c0_i32] : i32, i32)                     // coords (ignored: zeros)
      tensor_size  = %ts0, %ts1   // (size - 1) per mode, i16
      lower_stride = %ls1         // strides for modes 1..N-1, in 16B units, low 32b (i32)
      upper_stride = %us          // packed upper 4 bits of up to 4 strides (i16)
      cache_policy = %policy      // L2 eviction policy (i64)
      mode = <tiled> num_cta = 1
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte:$src_desc,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase:$dsmem_data_addr,
    CuteNVGPU_CuTePointerToSharedOrDistributedSharedMemory:$dsmem_bar_addr,
    CuteNVGPU_CuTePointerType:$override_addr,
    Variadic<I16>:$tensor_size,
    Variadic<I32>:$lower_stride,
    Optional<I16>:$upper_stride,
    Variadic<I32>:$coord,
    Optional<I32>:$multicast_mask,
    Optional<I64>:$cache_policy,
    CuteNVGPU_TmaLoadModeAttr:$mode,
    CuteNVGPU_ArchCopySM100TmaLoadOpNumCTAAttr:$num_cta
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchCopySM107TmaStoreOverrideOp : CuteNVGPU_Op<"arch.copy.SM107.tma_store_override", [AttrSizedOperandSegments]> {
  let description = [{
    The `arch.copy.SM107.tma_store_override` operation performs a TMA tensor store
    from shared memory to global with runtime override of address, dimensions,
    and strides.

    In addition to the normal TMA store operands, this op carries:
    - `override_addr`: global pointer overriding the descriptor's base address
    - `tensor_size`: per-mode sizes (encoded as size-1, i16)
    - `lower_stride`: per-mode strides in 16B units (i32, modes 1..N-1)
    - `upper_stride`: packed upper 4 bits of up to 4 strides (i16)

    Example
    The example of tma_load_override applies here except that tma_store_override
    doesn't have the mbar argument.

    Coordinates are passed as zeros (override form ignores them).
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte:$dst_desc,
    CuteNVGPU_CuTePointerToSharedMemoryWithSwizzleBase2:$src_smem_data_addr,
    CuteNVGPU_CuTePointerType:$override_addr,
    Variadic<I16>:$tensor_size,
    Variadic<I32>:$lower_stride,
    Optional<I16>:$upper_stride,
    Variadic<I32>:$coord,
    Optional<I64>:$cache_policy,
    CuteNVGPU_TmaStoreModeAttr:$mode
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchGetDynamicSmemOp : CuteNVGPU_Op<"arch.get_dyn_smem", [MemoryEffects<[MemAlloc]>]> {
  let description = [{
    This operation acquires the pointer to the dynamic shared memory, whose size
    is determined at the kernel launch time.
  }];
  let arguments = (ins
  );
  let results = (outs
    CuteNVGPU_CuTePointerToMemorySpacesSmem:$ptr
  );
}
def CuteNVGPU_ArchGetDynamicSmemSizeOp : CuteNVGPU_Op<"arch.get_dyn_smem_size", [Pure]> {
  let description = [{
    This operation gets the size in bytes of the dynamic shared memory that was
    specified at kernel launch time. This can be used for bounds checking during
    shared memory allocation.
  }];
  let arguments = (ins
  );
  let results = (outs
    I32:$size
  );
}
def CuteNVGPU_ArchMakeWarpUniformOp : CuteNVGPU_Op<"arch.make_warp_uniform", [Pure]> {
  let description = [{
    This operation is to reach a similar goal as canonical_warp_idx_sync() in cutlass C++, which is
    typically used to hint to ptxas to use uniform register to store current value so as to avoid
    redundant R2URs in mainloop. It's actually a shortcut usage of `nvvm.shfl.sync idx 31, val, 0, -1`.

    Example:

    ```mlir
    %thread_id = nvvm.read.ptx.sreg.tid.x : i32
    %warp_size_log2 = arith.constant 5 : i32
    %warp_id = arith.shrsi %thread_id, %warp_size_log2 : i32
    %warp_id_uniform = cute_nvgpu.arch.make_warp_uniform %warp_id : i32
    ```
    And this IR will be lowered to:
    ```mlir
    %thread_id = nvvm.read.ptx.sreg.tid.x : i32
    %warp_size_log2 = arith.constant 5 : i32
    %warp_id = arith.shrsi %thread_id, %warp_size_log2 : i32
    %warp_full_mask = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %warp_id_uniform = nvvm.shfl.sync idx %c0_i32, %warp_id, %c0_i32, %warp_full_mask : i32 -> i32
    ```
  }];
  let arguments = (ins
    I32:$val
  );
  let results = (outs
    I32:$res
  );
}
def CuteNVGPU_ArchMmaSM107UMMABlockScaledOp : CuteNVGPU_Op<"arch.mma.SM107.umma_block_scaled"> {
  let description = [{
    The `cute_nvgpu.arch.mma.SM107.umma_block_scaled` operation performs a matrix
    multiply-accumulate operation. This arch op contains scaling factor for A/B.
    Matrix elements for operand A can reside in tensor memory or shared memory.
    Operand B always reside in shared memory. Elements for the accumulator input
    and output (CD) always reside in tensor memory. The `accum` operand controls
    whether the matrix multiplication result is added to data in `C` (`accum` = `true`)
    or overwrites the data in `C` (`accum` = `false`). The `sf_a/b` operands indicate
    the scaling factor of A/B and they always reside in tensor memory. The `vec_size`
    attribute indicates the scaling vector size.
    See the file cutlass/include/cute/arch/mma_sm100_umma.hpp in the CUTLASS
    repository for more information.
    Examples:
    ```mlir
    cute_nvgpu.arch.mma.SM107.umma_block_scaled(%A, %B, %C, %sf_a, %sf_b, %idesc, %accum)
        { num_cta = 1, vec_size = 32, a_type = f8E5M2, shape_k = 32 }
        : (!cute_nvgpu.smem_desc, !cute_nvgpu.smem_desc, !cute.ptr<f32, tmem>, !cute.ptr<f8E8M0FNU, tmem>,
        !cute.ptr<f8E8M0FNU, tmem>, i32, i1)
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AllowableDescriptorTypesForSM107BlockscaledUMMAA:$a,
    CuteNVGPU_CuteSm107SmemDescType:$b,
    CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA:$cd,
    CuteNVGPU_CuTePointerToAllowableTypesForSM107UMMA:$sf_a,
    AnyType:$sf_b,
    AnyType:$idesc,
    AnyType:$accum,
    APIntAttr:$num_cta,
    I32Attr:$vec_size,
    TypeAttr:$a_type,
    TypeAttr:$b_type,
    I32Attr:$shape_k,
    CuteNVGPU_MmaCollectorOpAttr:$a_collector_op,
    CuteNVGPU_MmaCollectorOpAttr:$b_collector_op
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchMmaSM107UMMABlockScaledSparseOp : CuteNVGPU_Op<"arch.mma.SM107.umma_block_scaled_sparse"> {
  let description = [{
    The `cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse` operation performs a matrix
    multiply-accumulate operation. This arch op contains scaling factor for A/B and supports
    structured sparsity. Matrix elements for operand A can reside in tensor memory or shared memory.
    Operand B always reside in shared memory. Elements for the accumulator input
    and output (CD) always reside in tensor memory. The `accum` operand controls
    whether the matrix multiplication result is added to data in `C` (`accum` = `true`)
    or overwrites the data in `C` (`accum` = `false`). The `sf_a/b` operands indicate
    the scaling factor of A/B and they always reside in tensor memory. The `e_op` operand
    provides the sparse metadata that describes the sparsity positional information of the matrix,
    enabling efficient computation with structured sparse matrices. The `vec_size`
    attribute indicates the scaling vector size.
    See the file cutlass/include/cute/arch/mma_sm100_umma.hpp in the CUTLASS
    repository for more information.
    Examples:
    ```mlir
    cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse(%A, %B, %C, %sf_a, %sf_b, %E, %idesc, %accum)
        { num_cta = 1, vec_size = 32, a_type = f8E5M2, shape_k = 32 }
        : (!cute_nvgpu.sm107.smem_desc, !cute_nvgpu.sm107.smem_desc, !cute.ptr<f32, tmem>, !cute.ptr<f8E8M0FNU, tmem>,
        !cute.ptr<f8E8M0FNU, tmem>, !cute.ptr<ui8, tmem, align<8>>, i32, i1)
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AllowableDescriptorTypesForSM107BlockscaledUMMAA:$a,
    CuteNVGPU_CuteSm107SmemDescType:$b,
    CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA:$cd,
    CuteNVGPU_CuTePointerToAllowableTypesForSM107UMMA:$sf_a,
    AnyType:$sf_b,
    AnyType:$e_op,
    AnyType:$idesc,
    AnyType:$accum,
    APIntAttr:$num_cta,
    I32Attr:$vec_size,
    TypeAttr:$a_type,
    TypeAttr:$b_type,
    I32Attr:$shape_k,
    CuteNVGPU_MmaCollectorOpAttr:$a_collector_op,
    CuteNVGPU_MmaCollectorOpAttr:$b_collector_op
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchMmaSM107UMMAOp : CuteNVGPU_Op<"arch.mma.SM107.umma"> {
  let description = [{
    The `cute_nvgpu.arch.mma.SM107.umma` operation performs a matrix multiply-
    accumulate operation.
    Matrix elements for operand A can reside in either shared memory or tensor
    memory (tmem). As such, the argument for operand A can be either a shared
    memory descriptor or a pointer to tensor memory.
    Elements for operand B always reside in shared memory. Elements for the
    accumulator input and output (CD) always reside in tensor memory. The
    `accum` operand controls whether the matrix multiplication result is added
    to data in `C` (`accum` = `true`) or overwrites the data in `C` (`accum` =
    `false`). Needs instruction desc according to nvvm backend and an additional
    `a_type` for mmaKind inference.
    See the file cutlass/include/cute/arch/mma_sm100_umma.hpp in the CUTLASS
    repository for more information.
    Examples:
    ```mlir
    cute_nvgpu.arch.mma.SM107.umma(%A, %B, %C, %idesc, %accum)
        { num_cta = 1, ab_type = (f8E5M2, f8E5M2), shape_k = 64, ab_collector_op = (discard, discard) }
        : (!cute_nvgpu.smem_desc, !cute_nvgpu.smem_desc, !cute.ptr<f32, tmem>, i32, i1) -> ()
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AllowableDescriptorTypesForSM107UMMAAOperand:$a,
    CuteNVGPU_CuteSm107SmemDescType:$b,
    CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA:$cd,
    I32:$idesc,
    I1:$accum,
    CuteNVGPU_VectorOf32BitSignlessIntegerValuesOf2:$disable_output_lane,
    APIntAttr:$num_cta,
    APIntAttr:$scale,
    TypeAttr:$a_type,
    TypeAttr:$b_type,
    I32Attr:$shape_k,
    CuteNVGPU_MmaCollectorOpAttr:$a_collector_op,
    CuteNVGPU_MmaCollectorOpAttr:$b_collector_op
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchMmaSM107UMMASparseOp : CuteNVGPU_Op<"arch.mma.SM107.umma_sparse"> {
  let description = [{
    The `cute_nvgpu.arch.mma.SM107.umma_sparse` operation performs a matrix
    multiply-accumulate operation where matrix A is structured sparse.
    Matrix elements for operand A can reside in either shared memory or tensor
    memory (tmem). As such, the argument for operand A can be either a shared
    memory descriptor or a pointer to tensor memory. Operand E holds sparse
    metadata that reside in tensor memory.
    Elements for operand B always reside in shared memory. Elements for the
    accumulator input and output (CD) always reside in tensor memory. The
    `accum` operand controls whether the matrix multiplication result is added
    to data in `C` (`accum` = `true`) or overwrites the data in `C` (`accum` =
    `false`). Needs instruction desc according to nvvm backend and an additional
    `a_type` for mmaKind inference.
    See the file cutlass/include/cute/arch/mma_sm100_umma.hpp in the CUTLASS
    repository for more information.
    Examples:
    ```mlir
    cute_nvgpu.arch.mma.SM107.umma_sparse(%A, %B, %C, %E, %idesc, %accum)
        {num_cta = 1 : i32, scale = 0 : i32, ab_type = (f16, f16), shape_k = 32 : i32, ab_collector_op = (discard, discard)}
        : (!cute_nvgpu.sm107.smem_desc, !cute_nvgpu.sm107.smem_desc, !cute.ptr<f16, tmem>,
           !cute.ptr<ui8, tmem>, i32, i1) -> ()
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AllowableDescriptorTypesForSM107UMMAAOperand:$a,
    CuteNVGPU_CuteSm107SmemDescType:$b,
    CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA:$cd,
    CuteNVGPU_CuTePointerToAllowableTypesForSM100UMMA2:$e,
    AnyType:$idesc,
    AnyType:$accum,
    APIntAttr:$num_cta,
    APIntAttr:$scale,
    TypeAttr:$a_type,
    TypeAttr:$b_type,
    I32Attr:$shape_k,
    CuteNVGPU_MmaCollectorOpAttr:$a_collector_op,
    CuteNVGPU_MmaCollectorOpAttr:$b_collector_op
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchMmaSM120BlockScaledOp : CuteNVGPU_Op<"arch.mma.SM120.block_scaled", [AttrSizedOperandSegments]> {
  let description = [{
    The `cute_nvgpu.arch.mma.SM120.block_scaled` operation performs a matrix
    multiply-accumulate operation. This arch op contains scaling factor for A/B.
    Matrix elements for A/B/C/D/SFA/SFB and accumulator are always in registers.

    C is always F32 under current SM120 arch, so there is no `c_type` here.

    See the file cutlass/include/cute/arch/mma_sm120.hpp in the CUTLASS repository
    for more information.

    Examples:
    ```mlir
    cute_nvgpu.arch.mma.SM120.block_scaled
              A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
              sfA(%scaleAData) sfB(%scaleBData)
              {shape_MNK=<"(16,8,32)">, vec_size = 32, a_type = f4E2M1FN, b_type = f4E2M1FN, sf_type = f8E8M0FNU}
              : A[i32, i32, i32, i32] B[i32, i32] C[f32, f32, f32, f32]
                sfA(i8) sfB(i8) -> (f32, f32, f32, f32)
    ```
  }];
  let arguments = (ins
    Variadic<CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA>:$opA,
    Variadic<CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA>:$opB,
    Variadic<CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA2>:$opC,
    CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA3:$sfA,
    CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA3:$sfB,
    Optional<I16>:$byte_id_a,
    Optional<I16>:$byte_id_b,
    Cute_ShapeAttr:$shape_MNK,
    I32Attr:$vec_size,
    DefaultValuedAttr<I16Attr, "0">:$thread_id_a,
    DefaultValuedAttr<I16Attr, "0">:$thread_id_b,
    CuteNVGPU_SM120BlockScaledABTypeAttr:$a_type,
    CuteNVGPU_SM120BlockScaledABTypeAttr:$b_type,
    CuteNVGPU_SM120BlockScaledSFTypeAttr:$sf_type
  );
  let results = (outs
    Variadic<CuteNVGPU_AllowableRegisterTypesForSM120BlockScaledMMA2>:$res
  );
}
def CuteNVGPU_ArchPrefetchTmaDescOp : CuteNVGPU_Op<"arch.prefetch_tma_desc"> {
  let description = [{
    The `arch.prefetch_tma_desc` operation does prefetch TMA descriptor using generic addressing
    (i.e. no specific state space: const or param).

    Example:

    ```mlir
    cute_nvgpu.arch.prefetch_tma_desc(%tma_a) : (!cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>) -> ()
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SM90PointerToTMATensorDescriptor:$tma_descriptor_ptr
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchSM100AllocTmemOp : CuteNVGPU_Op<"arch.sm100.alloc_tmem"> {
  let summary = "The `arch.sm100.alloc_tmem` operation performs dynamic allocation of TMEM.";
  let description = [{
    It dynamically allocates storage on TMEM with requested `n_cols` and stores
    the allocated TMEM pointer into the shared memory buffer at the shared memory
    location specified by `dst_ptr`.

    This operation is blocking, meaning it will wait until enough memory is available.
    The action writting pointer value into shared memory is not synchronized. It's
    user's responsibility to synchronize to make sure it's visible to all threads in the CTA.

    The allocation may fail if one of following conditions is true:
    1. The requested allocation size is not a multiple of 32 columns or greater
       than 512 columns or not a power of 2, or
    2. If an attempt is made to allocate after executing the
       `cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit` operation.

    This operation must be issued by a single fully active warp of the CTA and
    must not be issued by more than one warp simultaneously. For repeated
    allocations, the same warp must be used to issue all allocations. All
    `alloc_tmem` operations must be issued by the same warp.

    For the allocation of 2 CTAs, one warp from each of the peer CTAs must
    collectively perform the allocation, and the issuing warp must make sure that
    peer CTA is launched and is still active. The participating warps from both
    CTAs should have the same logical warp ID. There is no explicit need for
    synchronization between these 2 CTAs before executing this operation.

    The behavior of this operation is undefined if all the threads in the warp do
    not use the same values of `n_cols`, or if any thread in the warp has exited.

    The meaning of each argument is as follows:
    - `n_cols`: The number of columns to be allocated. The unit of allocation is
    32 columns and the number of columns must be a power of 2. It must be within
    the range [32, 512]. The number of columns allocated should not increase
    between any two allocations in the execution order within the CTA. Failing to
    meet these conditions may lead to undefined behavior.
    - `dst_ptr`: Pointer to shared memory where the result TMEM pointer will be
    written. It should have a natural alignment of 32-bits.
    - `is_two_cta`: Specifies the number of CTAs involved in the allocation.

    Examples:

    ```mlir
    cute_nvgpu.arch.sm100.alloc_tmem(%n_cols, %res_shm_ptr) [cta_1]
        : i32, !cute.ptr<i32, smem align<4>>

    cute_nvgpu.arch.sm100.alloc_tmem(%n_cols, %res_shm_ptr) [cta_2]
        : i32, !cute.ptr<i32, smem align<4>>
    ```
  }];
  let arguments = (ins
    I32:$n_cols,
    CuteNVGPU_CuTePointerToMemorySpacesSmem:$dst_ptr,
    UnitAttr:$is_two_cta,
    UnitAttr:$exclusive
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchSM100DeallocTmemOp : CuteNVGPU_Op<"arch.sm100.dealloc_tmem"> {
  let description = [{
    The `arch.sm100.dealloc_tmem` operation deallocates the Tensor Memory
    specified by the TMEM address `tmem_ptr` and slice count. The address
    `tmem_ptr` must point to a previously allocated TMEM. If the
    deallocation fails, a runtime error occurs.

    This operation should only be performed by a single fully active warp within
    the CTA. It should not be carried out by multiple warps simultaneously. The
    warp ID that does alloc and dealloc can be different, but all `dealloc_tmem`
    operations must be issued by the same warp.

    For the 2 CTAs deallocation, one warp from each of the peer CTAs must
    collectively perform the de-allocation, and the issuing warp must make sure
    that peer CTA is launched and is still active. The warps from both CTAs should
    have the same logical warp ID. Prior explicit synchronization between the two
    CTAs is not required before executing this operation.

    The behavior of this operation is undefined if all the threads in the warp do
    not use the same values of `n_cols`, or if any thread in the warp has exited.

    The meaning of each argument is as follows:
    - `tmem_ptr`: The base address of the TMEM buffer being deallocated.
    - `n_cols`: The number of columns to be deallocated. The unit of deallocation
    is 32 columns and the number of columns must be a power of 2. It must be within
    the range [32, 512]. Failing to meet these conditions may lead to undefined
    behavior.
    - `is_two_cta`: Specifies the number of CTAs involved in the deallocation.

    Examples:

    ```mlir
    cute_nvgpu.arch.sm100.dealloc_tmem(%tmem_ptr, %n_cols) [cta_1]
        : !cute.ptr<f32, tmem, align<16>>, i32

    cute_nvgpu.arch.sm100.dealloc_tmem(%tmem_ptr, %n_cols) [cta_2]
        : !cute.ptr<f32, tmem, align<16>>, i32
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesTmem:$tmem_ptr,
    I32:$n_cols,
    UnitAttr:$is_two_cta,
    UnitAttr:$exclusive
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchSM100RelinquishTmemAllocPermitOp : CuteNVGPU_Op<"arch.sm100.relinquish_tmem_alloc_permit"> {
  let description = [{
    This `arch.sm100.relinquish_tmem_alloc_permit` specifies that the CTA of the
    executing thread is relinquishing the right to allocate Tensor Memory so that
    other CTAs can allocate.

    Once this operation is executed, any subsequent attempts by the same CTA to
    allocate TMEM are considered illegal.

    Example:

    ```mlir
    cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit [cta_1]
    ```
  }];
  let arguments = (ins
    UnitAttr:$is_two_cta
  );
  let results = (outs
  );
}
def CuteNVGPU_ArchSM100RetrieveTmemPtrOp : CuteNVGPU_Op<"arch.sm100.retrieve_tmem_ptr", [MemoryEffects<[MemRead]>]> {
  let description = [{
    The `arch.sm100.retrieve_tmem_ptr` operation is a convenient shorthand for
    loading the allocated TMEM pointer from shared memory and casting it into a
    TMEM pointer. It includes a correctness check for the TMEM pointer and
    returns a null pointer if the TMEM pointer is invalid.

    Example:

    ```mlir
    %tmem_ptr = cute_nvgpu.arch.sm100.retrieve_tmem_ptr(%res_shm_ptr)
        : !cute.ptr<i32, smem align<4>> -> !cute.ptr<f32, tmem, align<16>>
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesSmem:$smem_ptr
  );
  let results = (outs
    CuteNVGPU_CuTePointerToMemorySpacesTmem:$tmem_ptr
  );
}
def CuteNVGPU_AtomCopyMakeExecTmaOp : CuteNVGPU_InferredOp<"atom.make_exec_tma", [Pure]> {
  let arguments = (ins
    CuteNVGPU_TypeInterfaceForNonExecutableTMAAtomTypes:$input,
    UnitAttr:$override,
    UnitAttr:$no_fully_oob_tile
  );
  let results = (outs
    CuteNVGPU_CopyAtomTypeInterfaceInstance:$result
  );
}
def CuteNVGPU_AtomCopyMakeNonExec2DGather4TmaLoadOp : CuteNVGPU_InferredOp<"atom.make_non_exec_2d_gather4_tma_load", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA loads in tiled gather4 mode (.tile::gather4)
    2. the TMA tensor mapping the logical coordinates of gmem_tensor to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for
    safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma. It can also be used directly for TiledCopy-based
    partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_CuTeLayoutType:$gmem_coord_layout,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    CuteNVGPU_GatherScatterTmaLoadAttr:$kind,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeNonExec2DScatter4TmaStoreOp : CuteNVGPU_InferredOp<"atom.make_non_exec_2d_scatter4_tma_store", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA stores in tiled scatter4 mode (.tile::scatter4)
    2. the TMA tensor mapping the logical coordinates of the scatter index layout to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for
    safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma. It can also be used directly for TiledCopy-based
    partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_CuTeLayoutType:$gmem_index_layout,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA2:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeNonExecIm2ColTmaLoadOp : CuteNVGPU_InferredOp<"atom.make_non_exec_im2col_tma_load", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA loads in im2col mode
    2. the TMA tensor mapping the logical coordinates of gmem_tensor to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for
    safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma_load. It can also be used directly for TiledCopy-based
    partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_CuTeComposedLayoutType:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_corner_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$upper_corner_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_padding_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$upper_padding_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$stride_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_srt,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$stride_srt,
    CuteNVGPU_Im2ColTmaLoadAttr:$kind,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA3:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeNonExecIm2ColTmaStoreOp : CuteNVGPU_InferredOp<"atom.make_non_exec_im2col_tma_store", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA stores in im2col mode
    2. the TMA tensor mapping the logical coordinates of gmem_tensor to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for
    safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma_store. It can also be used directly for TiledCopy-based
    partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_CuTeComposedLayoutType:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA4:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeNonExecTiledTmaLoadOp : CuteNVGPU_InferredOp<"atom.make_non_exec_tiled_tma_load", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA loads in tiled mode
    2. the TMA tensor mapping the logical coordinates of gmem_tensor to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for
    safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma. It can also be used directly for TiledCopy-based
    partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    CuteNVGPU_TiledTmaLoadAttr:$kind,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA5:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeNonExecTiledTmaReduceOp : CuteNVGPU_InferredOp<"atom.make_non_exec_tiled_tma_reduce", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA reduce in tiled mode
    2. the TMA tensor mapping the logical coordinates of gmem_tensor to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma. It can also be used directly for TiledCopy-based partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    CuteNVGPU_ReductionKindAttr:$kind,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA6:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeNonExecTiledTmaStoreOp : CuteNVGPU_InferredOp<"atom.make_non_exec_tiled_tma_store", [Pure]> {
  let summary = "This operation creates";
  let description = [{
    1. a non-executable atom for TMA store in tiled mode
    2. the TMA tensor mapping the logical coordinates of gmem_tensor to TMA coordinate offsets

    The non-executable atom, which wraps a TMA descriptor and attaches type information to it for
    safety, can then be used to create an executable one using
    cute_nvgpu.atom.copy.make_exec_tma. It can also be used directly for TiledCopy-based
    partitioning of source and destination tensors.
  }];
  let arguments = (ins
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_v_map,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA7:$non_exec_atom,
    CuteNVGPU_CuteCoordTensorType:$tma_tensor
  );
}
def CuteNVGPU_AtomCopyMakeS2TCopyOp : CuteNVGPU_InferredOp<"atom.make_s2t_copy", [Pure]> {
  let description = [{
    This operation creates a new copy atom instance by tiling an utccp copy atom
    instance with a tmem memref.

    Example:
    ```mlir
    %copy_s2t_atom = cute_nvgpu.atom.make_s2t_copy() : !cute_nvgpu.atom.s2t_copy<i32, 128 DP, 256 bit, 1cta>
    %tmem_memref = cute.memref.alloca() : !cute.memref<i32, tmem, align<128>, "(128,32):(65536,1)">
    %0 = cute_nvgpu.atom.make_s2t_copy(%copy_s2t_atom, %tmem_memref)
      : (!cute_nvgpu.atom.s2t_copy<i32, 128 DP, 256 bit, 1cta>,
         !cute.memref<i32, tmem, align<128>, "(128,32):(65536,1)">)
      -> !cute.tiled_copy<!cute_nvgpu.atom.s2t_copy<i32, 128 DP, 256 bit, 1cta>,
                          layout_copy_tv = <"(1,(128,8)):(0,(1,128))">,
                          tiler_mn = <"[128:1;8:1]">>
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SM100BlackwellSharedToTensorMemoryCopyAtom:$copy_s2t_atom,
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$tmem_memref
  );
  let results = (outs
    CuteNVGPU_TiledCopyAtomType:$result
  );
}
def CuteNVGPU_AtomCopyMakeTmemCopyOp : CuteNVGPU_InferredOp<"atom.make_tmem_copy", [Pure]> {
  let description = [{
    This operation creates a new copy atom instance by tiling a load tmem copy atom
    instance with a tmem memref.

    Example:
    ```mlir
    // %tmem_memref : !cute.memref<i32, tmem, align<128>, "(128,32):(65536,1)">
    %load_atom = cute.make_atom() : !cute_nvgpu.atom.tmem_load<i32, 32 DP, 32 bit, x32>
    %0 = cute_nvgpu.atom.make_tmem_copy(%load_atom, %tmem_memref)
      : (!cute_nvgpu.atom.tmem_load<i32, 32 DP, 32 bit, x32>,
         !cute.memref<i32, tmem, align<128>, "(128,32):(65536,1)">)
      -> !cute.tiled_copy<!cute_nvgpu.atom.tmem_load<i32, 32 DP, 32 bit, x32>,
                          layout_copy_tv = <"((32,4),(32,32)):((0,1),(128,4))">,
                          tiler_mn = <"[(4,32):(32,1);32:1]">>
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CopyAtomTypeInterfaceInstance:$atom,
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$tmem_memref
  );
  let results = (outs
    CuteNVGPU_TiledCopyAtomType:$result
  );
}
def CuteNVGPU_AtomGetCoordTensorOp : CuteNVGPU_InferredOp<"atom.get_coord_tensor", [SameVariadicOperandSize, Pure]> {
  let description = [{
    This operation creates a get tma coord atom instance from copy atom and shapes.
    The atom will be called before algo copy(tma copy) to provide tma coord for algo copy.
    In Cute C++, copy_atom cache stride info during make_tiled_copy and the stride
    info is associated with shapes. shapes in argument will make a congruent comparison
    with cache first and then construct tma coord with stride.

    Example:
    ```mlir
    %5 = cute_nvgpu.atom.get_coord_tensor(%4, %5)
      : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 131072, mode=tiled, num_cta=1, g_stride=<"(1@1,1@0,64@0)">>,
         !cute.shape<128, 64>)
      -> !cute.coord_tensor<"(0,0)", "(512,64,2):(1@1,1@0,64@0)">
    ```

    ```mlir
    %6 = cute_nvgpu.atom.get_coord_tensor(%4, %5, %2, %3)
      : (!cute_nvgpu.atom.tma_load<f16, copy_bits = 131072, mode=im2col, num_cta=1, g_stride=<"(1@1, 1@0)">>,
         !cute.shape<"(500, 128)">,
         !cute.int_tuple<"(0, 0)">,
         !cute.layout<"(128, 256): (1@0, 1@1)
      -> !cute.coord_tensor<"(0,0)", "(500,128):(1@1,1@0) o (0,0) o (128,256):(1@0,1@1)">
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AtomTypeInterfaceInstance:$atom,
    CuteNVGPU_CuteShapeType:$shape,
    Optional<CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS>:$offset_b,
    Optional<CuteNVGPU_CuTeLayoutType>:$layout_b
  );
  let results = (outs
    CuteNVGPU_CuteCoordTensorType:$result
  );
}
def CuteNVGPU_AtomGetCopyS2TSmemDescViewOp : CuteNVGPU_InferredOp<"atom.get_copy_s2t_smem_desc_view", [Pure]> {
  let description = [{
    This operation creates a SmemDescView instance from the given UtccpAtomType and
    the SmemView. The given SmemView must match with the Utccp's TiledCopy src partitioned layout.
    The result will be used as the src of algo.copy.

    Example:
    ```mlir
    %res_0 = cute_nvgpu.atom.get_copy_s2t_smem_desc_view(%atom_0, %src_0)
    : (!cute_nvgpu.atom.s2t_copy<f32, 128 DP, 128 bit, 1cta>,
       !cute.memref<f32, smem, align<16>, S<2,4,3>, "(((128,4),1),2,(4,2)):(((16,1),0),2048,(4,4096))">)
    -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, "(((128,1),1),2,(4,2)):(((4,1),0),512,(1,1024))">
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SM100BlackwellSharedToTensorMemoryCopyAtom2:$atom,
    CuteNVGPU_CuTeMemrefToMemorySpacesSmem:$view
  );
  let results = (outs
    CuteNVGPU_CuteSmemDescViewType:$result
  );
}
def CuteNVGPU_AtomGetCopyS2TSmemDescViewSM107Op : CuteNVGPU_InferredOp<"atom.sm107.get_copy_s2t_smem_desc_view", [Pure]> {
  let description = [{
    This operation creates a SmemDescView instance from the given UtccpAtomType and
    the SmemView. The given SmemView must match with the Utccp's TiledCopy src partitioned layout.
    The result will be used as the src of algo.copy.

    Example:
    ```mlir
    %res_0 = cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view(%atom_0, %src_0)
    : (!cute_nvgpu.atom.s2t_copy<f32, 128 DP, 128 bit, 1cta>,
       !cute.memref<f32, smem, align<16>, S<2,4,3>, "(((128,4),1),2,(4,2)):(((16,1),0),2048,(4,4096))">)
    -> !cute_nvgpu.smem_desc_view<!cute_nvgpu.sm107.smem_desc, "(((128,1),1),2,(4,2)):(((4,1),0),512,(1,1024))">
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SM100BlackwellSharedToTensorMemoryCopyAtom2:$atom,
    CuteNVGPU_CuTeMemrefToMemorySpacesSmem:$view
  );
  let results = (outs
    CuteNVGPU_CuteSmemDescViewType:$result
  );
}
def CuteNVGPU_AtomGetValueOp : CuteNVGPU_Op<"atom.get_value", [Pure]> {
  let description = [{
    CuTe atoms represent both compile-time and runtime information for an
    architecture-specific `copy` or `MMA` operation. For runtime information,
    an atom can be thought of as similar to a C/C++ `struct` with predefined
    member variables. Instead of defining operations to get and set the member
    variables of each architecture atom, the CuTe dialect uses `get_value` and
    `set_value` operations to uniformly handle retrieving and updating atom
    member variables (hereafter referred to as "fields").
    The fields of each architecture-specific atom are identified by an enum
    attribute. See the documentation for the operations that create an atom
    or the attribute associated with the atom type for the expected enum values
    and associated operand types.
    The `field` attribute must correspond to the type of the `atom` operand.
    For example, if the `atom` type is `!cute_nvgpu.sm100.mma`, the enum
    attribute must be `#cute_nvgpu.atom_mma_field_sm100`. The verifier for this
    operation will also compare the value `Type` to that expected by the
    specific field.

    Example:
    ```mlir
    %5 = cute_nvgpu.atom.get_value<accum_c>(%4) : (!cute_nvgpu.sm100.mma<64x64x16,
                                                                          num_cta = 1,
                                                                          ab_major = (k,mn),
                                                                          elem_type = (f16,f16,f16),
                                                                          frag_kind = ss,
                                                                          c_scale_exp = 0>) -> i1
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AtomTypeInterfaceInstance:$atom,
    AnyAttr:$field
  );
  let results = (outs
    AnyType:$result
  );
}
def CuteNVGPU_AtomMakeTmaResidueTensorOp : CuteNVGPU_InferredOp<"atom.make_tma_residue_tensor", [Pure]> {
  let description = [{
    Builds the per-TMA-mode "remaining elements" coord tensor that
    `cute.copy` consumes on the override path. The op takes a non-executable
    TMA atom (whose type carries `tma_gbasis`) and the corresponding gmem
    tensor, and produces a coord tensor whose

    - iter is gmem tensor's shape combined and ordered according to tma gbasis.
    - layout has the same shape as the tma tensor while the stride is negated.

    The residue tensor tracks the remaining shape to copy during TMA override,
    e.g, slicing the residue tensor with a cta coordinate gets the remaining 
    shape for it given the negated stride.

    Example:
    ```mlir
    %residue = cute_nvgpu.atom.make_tma_residue_tensor(%non_exec_atom, %gmem)
      : (!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, i32, copy_bits=262144,
           tma_gbasis=<"(128,64):(1@1,1@0)">>,
         !cute.memref<i32, gmem, align<16>, "(?,?):(?,1)">)
      -> !cute.coord_tensor<"(?,?)", "(?,?):(-1@1,-1@0)">
    ```
  }];
  let arguments = (ins
    CuteNVGPU_TheNonExecutableCopyAtomTypeForTMA8:$tma_atom,
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor
  );
  let results = (outs
    CuteNVGPU_CuteCoordTensorType:$residue
  );
}
def CuteNVGPU_AtomSetValueOp : CuteNVGPU_InferredOp<"atom.set_value", [Pure]> {
  let description = [{
    CuTe atoms represent both compile-time and runtime information for an
    architecture-specific `copy` or `MMA` operation. For runtime information,
    an atom can be thought of as similar to a C/C++ `struct` with predefined
    member variables. Instead of defining operations to get and set the member
    variables of each architecture atom, the CuTe dialect uses `get_value` and
    `set_value` operations to uniformly handle retrieving and updating atom
    member variables (hereafter referred to as "fields").
    The fields of each architecture-specific atom are identified by an enum
    attribute. See the documentation for the operations that create an atom
    or the attribute associated with the atom type for the expected enum values
    and associated operand types.
    The `field` attribute must correspond to the type of the `atom` operand.
    For example, if the `atom` type is `!cute_nvgpu.sm100.mma`, the enum
    attribute must be `#cute_nvgpu.atom_mma_field_sm100`. The verifier for this
    operation will also compare the value `Type` to that expected by the
    specific field.

    Example:
    ```mlir
    %2 = cute_nvgpu.atom.set_value<accum_c>(%0, %no_accum) :
      (!cute_nvgpu.sm100.mma<64x64x16,
                             num_cta = 1,
                             ab_major = (k,mn),
                             elem_type = (f16,f16,f16),
                             frag_kind = ss,
                             c_scale_exp = 0>, i1)
    ```
  }];
  let arguments = (ins
    CuteNVGPU_AtomTypeInterfaceInstance:$atom,
    AnyType:$value,
    AnyAttr:$field
  );
  let results = (outs
    CuteNVGPU_AtomTypeInterfaceInstance:$result
  );
}
def CuteNVGPU_AtomTmaPartitionOp : CuteNVGPU_InferredOp<"atom.tma_partition", [Pure]> {
  let description = [{
    tma_partition is the new TMA routine for the dynamic and preferred cluster feature.
    tma_partition op requires below arguments:
      1. tma_atom;
      2. cta_coord in the multicast dimension;
      3. cta_layout in the multicast dimension;
      4. smem_memref;
      5. variadic CuteViewTypeInterface tensors (one or more coord_tensor / memref),
         covering the following cases:
         5.a: for regular TMA load/store, a single TMA tensor as input
         5.b: for tile::gather4 mode, a 2D TMA coord tensor plus the corresponding
              gmem index tensor
    and it produces:
      1. partitioned_smem_memref;
      2. one partitioned coord_tensor / memref per input, with the following cases:
         2.a: for regular TMA load/store, one partitioned TMA tensor
         2.b: for tile::gather4 mode, the partitioned TMA coord tensor plus the
              partitioned gmem index tensor

    The return values are the src and dst arguments of algo.copy(atom, src, dst).
    The second and the third input arguments can be runtime values, so the multicast
    can be runtime determined.

    This operation is also compatible for the old static cluster TMA routines.
    You only need to feed the cta_coord and cta_layout with static values,
    then you can get compile-time multicast computation.

    The target_tensors operand is variadic and accepts any CuteViewTypeInterface
    type (coord_tensor or memref). When a single tensor is provided, brackets
    are omitted for backward compatibility. When multiple tensors are provided,
    they are grouped in brackets.

    Example (single tensor):
    ```mlir
    %res_smem, %res_gmem = cute_nvgpu.atom.tma_partition(
      %tma_atom, %cta_coord, %cta_layout, %stensor, %gtensor
    ) : (!cute_nvgpu.atom.tma_load<f16, copy_bits=131072, mode=tiled, num_cta=2>,
         !cute.coord<"?">,
         !cute.layout<"?:1">,
         !cute.memref<f16, smem, align<128>, S<3,4,3>, "(((128,16),1,4),8):(((64,1),0,16),8192)">,
         !cute.coord_tensor<"(0,?,0)", "(((128,16),1,4),?,?,?):(((1@1,1@0),0,16@0),256@1,64@0,1@2)">)
    ```

    Example (multiple tensors):
    ```mlir
    %res_smem, %res_gmem0, %res_gmem1 = cute_nvgpu.atom.tma_partition(
      %tma_atom, %cta_coord, %cta_layout, %stensor, [%gtensor0, %gtensor1]
    ) : (!cute_nvgpu.atom.tma_load<...>,
         !cute.coord<"?">,
         !cute.layout<"?:1">,
         !cute.memref<f16, smem, ...>,
         [!cute.coord_tensor<...>, !cute.memref<f16, gmem, ...>])
      -> (!cute.memref<f16, smem, ...>,
          [!cute.coord_tensor<...>, !cute.memref<f16, gmem, ...>])
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SM90TMATensorMemoryAcceleratorLoadAtomType:$tma_atom,
    CuteNVGPU_CuteCoordType:$cta_coord,
    CuteNVGPU_CuTeLayoutType:$cta_layout,
    CuteNVGPU_CuTeMemrefToMemorySpacesSmem:$smem_tensor,
    Variadic<CuteNVGPU_ViewTypeInterfaceInstance>:$target_tensors
  );
  let results = (outs
    CuteNVGPU_CuTeMemrefToMemorySpacesSmem:$res_smem_tensor,
    Variadic<CuteNVGPU_ViewTypeInterfaceInstance>:$res_target_tensors
  );
}
def CuteNVGPU_CastTMADescToIntegerOp : CuteNVGPU_Op<"cast_tma_desc_to_integer", [Pure]> {
  let summary = "cast_tma_desc_to_integer cast a tma_desc into an integer.";
  let description = [{
    Example:
    ```mlir
      %23 = cute_nvgpu.cast_tma_desc_to_integer(%11 : !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>) -> i64
    ```
  }];
  let arguments = (ins
    CuteNVGPU_SM90PointerToTMATensorDescriptor:$src
  );
  let results = (outs
    CuteNVGPU_Integer:$dst
  );
}
def CuteNVGPU_CopyTMADescriptorOp : CuteNVGPU_Op<"copy_tma_desc"> {
  let description = [{
    This operation copies the TMA descriptor from a non-executable TMA atom to the pointer.
    The pointer could be a smem or gmem pointer and should be aligned to at least the size of TMA descriptor.
  }];
  let arguments = (ins
    CuteNVGPU_TypeInterfaceForNonExecutableTMAAtomTypes:$atom,
    CuteNVGPU_CuTePointerToMemorySpacesSmemOrCuTe:$tma_desc_ptr
  );
  let results = (outs
  );
}
def CuteNVGPU_GetGridConstantPointerOp : CuteNVGPU_InferredOp<"get_grid_constant_pointer", [Pure]> {
  let description = [{
    'cute_nvgpu.get_grid_constant_pointer' takes a grid constant value argument
    from a kernel and returns a `cute.ptr` whose pointee type matches the
    input. 

    Before device ABI conversion, the input may be an intermediate result that
    function finalization will promote to a grid-constant kernel argument.
    Direct kernel arguments are verified immediately; intermediate provenance
    is validated when the physical kernel signature is finalized. The result
    type and alignment are fully inferred from the input type by
    `InferTypeOpInterface`.
    Example (TMA descriptor):
      func.func @kernel(
        %tma_desc_a : !cute_nvgpu.tma_descriptor_tiled {cute_nvgpu.grid_constant})
        attributes {cute.kernel} {
          %tma_desc_a_ptr = cute_nvgpu.get_grid_constant_pointer (%tma_desc_a) :
                !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>;
          ...
      }

    Example (arbitrary value type):
      func.func @kernel(
        %params : !llvm.struct<(i64, i64)> {cute_nvgpu.grid_constant})
        attributes {cute.kernel} {
          %params_ptr = cute_nvgpu.get_grid_constant_pointer (%params) :
                !cute.ptr<!llvm.struct<(i64, i64)>, generic>;
          ...
      }
  }];
  let arguments = (ins
    AnyType:$input
  );
  let results = (outs
    CuteNVGPU_CuTePointerType:$result
  );
}
def CuteNVGPU_GetTmaDescAddrOp : CuteNVGPU_Op<"get_tma_desc_addr", [Pure]> {
  let description = [{
    The `cute_nvgpu.get_tma_desc_addr` operation accepts an input copy atom,
    extracts the address within copy atom, then returns it as the address
    of current copy atom's TMA desc in a single SSA value. The address within
    input copy atom should have global/generic address space, and should be aligned to 64B.
    Example:

    ```mlir
    %1 = cute_nvgpu.get_tma_desc_addr(%cpy_atom : !cute_nvgpu.atom.tma_load<f16, copy_bits = 131072, mode=tiled, num_cta=1>) -> !cute.ptr<generic, align<64>>
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CopyAtomTypeInterfaceInstance:$src
  );
  let results = (outs
    CuteNVGPU_CuTePointerToGlobalMemoryWith64Byte:$ptr
  );
}
def CuteNVGPU_MakeGMMASmemDescOp : CuteNVGPU_Op<"make_gmma_smem_desc", [Pure]> {
  let description = [{
    The `cute_nvgpu.make_gmma_smem_desc` operation accepts an input cute.ptr alongside a cute.layout and a major attr,
    then constructs the shared memory descriptor and returns it. The layout must be describing
    the buffer pointed to by the input pointer and must carry valid swizzle information.
    For GMMA, there are 4 supported swizzle variants:

    ```
    swizzle    | cooresponding enum in gmma smem desc
    S<0, 4, 3> | SWIZZLE_NONE
    S<1, 4, 3> | SWIZZLE_32B
    S<2, 4, 3> | SWIZZLE_64B
    S<3, 4, 3> | SWIZZLE_128B
    ```
    The cute.ptr must carry shared address space and must be aligned to 16B.

    Example:

    ```mlir
    %1 = cute_nvgpu.make_gmma_smem_desc(%arg1 : !cute.ptr<f16, smem, align<16>, S<3,4,3>>)
      layout = <"((64,16),1):((64,1),0)">, major = #cute_nvgpu.major<k>
      -> !cute_nvgpu.smem_desc
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToAllowableTypesForSM90MMA:$src,
    Cute_LayoutAttr:$layout,
    CuteNVGPU_MajorModeAttr:$major
  );
  let results = (outs
    CuteNVGPU_CuteSmemDescType:$res
  );
}
def CuteNVGPU_MakeTmaDescIm2ColAtOp : CuteNVGPU_Op<"make_tma_desc_im2col_at"> {
  let description = [{
    In-place variant of Cute_MakeTmaDescIm2ColOp. The descriptor is stored
    into the address passed as the last operand.

    Operands                 (identical semantics to the im2col op)
    - All operands of MakeTmaDescIm2ColOp (see above).
    - Value  tma_desc_addr – destination address for the descriptor.

    Optional attributes      (identical semantics to the im2col op)
    - num_multicast, tma_format

    Results: none – descriptor is written to the supplied address.

    Example:
    ```mlir
    cute_nvgpu.make_tma_desc_im2col_at(
        %tensor,
        %slayout,
        %cta_v_tile,
        %lower_corner_whd,
        %upper_corner_whd,
        %lower_padding_whd,
        %upper_padding_whd,
        %stride_whd,
        %lower_srt,
        %stride_srt,
        %tma_desc_addr)
        <{num_multicast = 1 tma_format = #cute_nvgpu.tma_data_format<F16_RN>}> : (
        !cute.memref<f16, gmem, align<16>, "((8,8,1,4),64,1):((64,512,4096,4096),1,0)">,
        !cute.composed_layout<"S<2,4,3> o 0 o ((8,16),(32,1)):((32,256),(1,0))">,
        !cute.layout<"(128,32):(1@0,1@1)">,
        !cute.int_tuple<"(0,0,0)">,
        !cute.int_tuple<"(0,0,0)">,
        !cute.int_tuple<"(0,0,0)">,
        !cute.int_tuple<"(0,0,0)">,
        !cute.int_tuple<"(1,1,1)">,
        !cute.int_tuple<"(0,0,0)">,
        !cute.int_tuple<"(1,1,1)">,
        !cute.ptr<!cute_nvgpu.tma_descriptor_im2col, generic, align<64>>)
    ```
  }];
  let arguments = (ins
    CuteNVGPU_ViewTypeInterfaceInstance:$gmem_view,
    CuteNVGPU_CuTeComposedLayoutType:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_value_tile,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_corner_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$upper_corner_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_padding_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$upper_padding_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$stride_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_srt,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$stride_srt,
    CuteNVGPU_CuTePointerType:$tma_desc_addr,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
  );
}
def CuteNVGPU_MakeTmaDescIm2ColOp : CuteNVGPU_Op<"make_tma_desc_im2col", [Pure]> {
  let summary = "Create a TMA descriptor in im2col mode, suitable for convolutional forward passes.";
  let description = [{
    Operands
    - Value  gmem_view, smem_layout, cta_value_tile   – as in the tiled op.
    - Value  lower_corner_whd / upper_corner_whd      – start & end (W,H,D)
                                                        positions of the
                                                        receptive-field window.
    - Value  lower_padding_whd / upper_padding_whd    – padding applied
                                                        before/after each dim.
    - Value  stride_whd                               – traversal stride in WHD.
    - Value  lower_srt / stride_srt                   – split-K traversal
                                                        parameters (SRT).

    Optional attributes
    - num_multicast : i32
    - tma_format    : CuteTmaDataFormatAttr

    Result
    - Cute pointer to the generated descriptor.

    Example:
    ```mlir
    %desc = cute_nvgpu.make_tma_desc_im2col(
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
        <{num_multicast = 1 tma_format = #cute_nvgpu.tma_data_format<F16_RN>}> : (
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
    ```
  }];
  let arguments = (ins
    CuteNVGPU_ViewTypeInterfaceInstance:$gmem_view,
    CuteNVGPU_CuTeComposedLayoutType:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_value_tile,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_corner_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$upper_corner_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_padding_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$upper_padding_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$stride_whd,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$lower_srt,
    CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS:$stride_srt,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_SM90PointerToTMATensorDescriptor:$result
  );
}
def CuteNVGPU_MakeTmaDescTiledAtOp : CuteNVGPU_Op<"make_tma_desc_tiled_at"> {
  let summary = "In-place version of Cute_MakeTmaDescTiledOp.";
  let description = [{
    Operands                   (identical semantics to the Tiled op)
    - Value  gmem_view       – Global-memory tensor view.
    - Value  smem_layout     – SMEM box layout.
    - Value  cta_value_tile  – CTA-value-tile mapping.
    - Value  tma_desc_addr   – Cute pointer that names the memory location
                               where the TMA descriptor should be written.
    - Value  traversal_stride – Cute int tuple that defines the traversal stride.
                                If omitted, the traversal stride is set to 1 for all dimensions.

    Optional attributes        (identical semantics to the Tiled op)
    - mode, num_multicast, tma_format

    Results: none – descriptor is written to the supplied address.

    Example:
    ```mlir
    cute_nvgpu.make_tma_desc_tiled_at(%a_tensor, %a_smem_clayout, %a_cta_v_tile, %tma_desc_addr)
      <{mode = <tiled>}> : (!cute.memref<f32, gmem, align<16>, "(?,?,?) : (1,?,?)">,
      !cute.composed_layout<"S<3,4,3> o 0 o (((64,64),2)):(((1,64,4096))">,
      !cute.layout<"(((64,64),2)):(((1@0,1@1),64@0))">,
      !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>)

    cute_nvgpu.make_tma_desc_tiled_at(%sfa_tensor, %sfa_smem_clayout, %sfa_cta_v_tile, %existing_tma_desc)
      <{mode = <tiled> num_multicast = 2 tma_format = #cute_nvgpu.tma_data_format<U16>}> : (
      !cute.memref<f8E4M3FN, global, align<1>, "((32,4,?),(32,4,?)):((16,4,?),(0,1,512))">,
      !cute.composed_layout<"S<0,4,3> o 0 o ((((32,4),1),(32,1)),1,(4,1))">,
      !cute.layout<"(((32,4),32),1,4):(((1@0@0,1@1@0),1@0@1),0,1@1@1)">,
      !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>)

    cute_nvgpu.make_tma_desc_tiled(%view, %slayout, %cta_v_tile, %tma_desc_addr %traversal_stride)
      <{mode = <tiled>}> : (
      !cute.memref<f16, gmem, align<16>, "((?,?,?,?),?):((?,?,?,?),?)">,
      !cute.composed_layout<"S<2,4,3> o 0 o (128,32):(32,1)">,
      !cute.layout<"((128),32):((1@0@0),1@1)">,
      !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>,
      !cute.int_tuple<"(1,?,?,?)">)
    ```
  }];
  let arguments = (ins
    CuteNVGPU_ViewTypeInterfaceInstance:$gmem_view,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_value_tile,
    CuteNVGPU_CuTePointerType:$tma_desc_addr,
    Optional<CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS>:$traversal_stride,
    CuteNVGPU_TmaLoadModeAttr:$mode,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
  );
}
def CuteNVGPU_MakeTmaDescTiledOp : CuteNVGPU_Op<"make_tma_desc_tiled", [Pure]> {
  let summary = "Create a TMA descriptor in tiled load mode.";
  let description = [{
    Operands
    - Value  gmem_view       – Cute view that describes the tensor in global memory.
    - Value  smem_layout     – Composed/regular Cute layout that defines the
                               rectangular box in shared memory (SMEM tile).
    - Value  cta_value_tile  – Cute layout that maps CTA threads to elements
                               within the SMEM tile.
    - Value  traversal_stride – Cute int tuple that defines the traversal stride.
                                If omitted, the traversal stride is set to 1 for all dimensions.

    Optional attributes
    - mode          : CuteTmaLoadModeAttr or CuteTmaStoreModeAttr (defaults to #cute_nvgpu.tma_load_mode<tiled>)
    - num_multicast : i32                    (defaults to 1; 1 ⇒ no multicast)
    - tma_format    : CuteTmaDataFormatAttr  (element format written to
                                              bits 74–71 of the HW TMA
                                              descriptor; if omitted an
                                              appropriate default is deduced)

    Result
    - Cute pointer to the freshly created TMA-descriptor object.

    Example:
    ```mlir
    %tma_desc_a = cute_nvgpu.make_tma_desc_tiled(%a_tensor, %a_smem_clayout, %a_cta_v_tile)
      <{mode = #cute_nvgpu.tma_load_mode<tiled>}> : (!cute.memref<f32, gmem, align<16>, "(?,?,?) : (1,?,?)">,
      !cute.composed_layout<"S<3,4,3> o 0 o (((64,64),2)):(((1,64),4096))">,
      !cute.layout<"(((64,64),2)):(((1@0,1@1),64@0))">) -> !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>

    %tma_desc_sfa = cute_nvgpu.make_tma_desc_tiled(%sfa_tensor, %sfa_smem_clayout, %sfa_cta_v_tile)
      <{mode = #cute_nvgpu.tma_load_mode<tiled> num_multicast = 2 tma_format = #cute_nvgpu.tma_data_format<U16>}> : (
      !cute.memref<f8E4M3FN, global, align<1>, "((32,4,?),(32,4,?)):((16,4,?),(0,1,512))">,
      !cute.composed_layout<"S<0,4,3> o 0 o ((((32,4),1),(32,1)),1,(4,1)):((((16,4),512),(0,1)),0,(1,512))">,
      !cute.layout<"(((32,4),32),1,4):(((1@0@0,1@1@0),1@0@1),0,1@1@1)">) -> !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>

    %tma_desc_strided_c = cute_nvgpu.make_tma_desc_tiled(%view, %slayout, %cta_v_tile, %traversal_stride)
      <{mode = #cute_nvgpu.tma_load_mode<tiled>}> : (
      !cute.memref<f16, gmem, align<16>, "((?,?,?,?),?):((?,?,?,?),?)">,
      !cute.composed_layout<"S<2,4,3> o 0 o (128,32):(32,1)">,
      !cute.layout<"((128),32):((1@0@0),1@1)">,
      !cute.int_tuple<"(1,?,?,?)">) -> !cute.ptr<!cute_nvgpu.tma_descriptor_tiled, generic, align<64>>
    ```
  }];
  let arguments = (ins
    CuteNVGPU_ViewTypeInterfaceInstance:$gmem_view,
    CuteNVGPU_GeneralizedLayoutTypeOfCuTe:$smem_layout,
    CuteNVGPU_CuTeLayoutType:$cta_value_tile,
    Optional<CuteNVGPU_IntTupleAsIntegerOrTupleOfIntTupleS>:$traversal_stride,
    CuteNVGPU_TmaLoadOrStoreModeAttr:$mode,
    OptionalAttr<I32Attr>:$num_multicast,
    OptionalAttr<CuteNVGPU_TmaDataFormatAttr>:$tma_format
  );
  let results = (outs
    CuteNVGPU_SM90PointerToTMATensorDescriptor:$result
  );
}
def CuteNVGPU_MakeUMMASM107SmemDescOp : CuteNVGPU_Op<"sm107.make_umma_smem_desc", [Pure]> {
  let summary = "Example:";
  let description = [{
    ```mlir
    %1 = cute_nvgpu.sm107.make_umma_smem_desc(%arg1 : !cute.ptr<f16, smem, align<16>)
      layout = <"S<3, 4, 3> o 0 o ((64,16),1):((64,1),0)">, major = #cute_nvgpu.major<k>
      -> !cute_nvgpu.sm107.smem_desc
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesSmem:$src,
    Optional<CuteNVGPU_CuTePointerToMemorySpacesSmem>:$nextSrc,
    Cute_LayoutAttr:$layout,
    CuteNVGPU_MajorModeAttr:$major
  );
  let results = (outs
    CuteNVGPU_CuteSm107SmemDescType:$res
  );
}
def CuteNVGPU_MakeUMMASmemDescCircularSM103Op : CuteNVGPU_InferredOp<"sm103.make_umma_smem_desc_circular", [Pure]> {
  let summary = "Example:";
  let description = [{
    ```mlir
    %1 = cute_nvgpu.sm103.make_umma_smem_desc_circular(%arg1 : !cute.ptr<f16, smem, align<16>)
      layout = <"S<3, 4, 3> o 0 o ((64,16),1):((64,1),0)">, major = #cute_nvgpu.major<k>
      -> !cute_nvgpu.smem_desc_circular
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte:$src,
    Optional<CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte>:$nextSrc,
    Cute_LayoutAttr:$layout,
    CuteNVGPU_MajorModeAttr:$major
  );
  let results = (outs
    AnyType:$res
  );
}
def CuteNVGPU_MakeUMMASmemDescCircularSM107Op : CuteNVGPU_InferredOp<"sm107.make_umma_smem_desc_circular", [Pure]> {
  let summary = "Example:";
  let description = [{
    ```mlir
    %1 = cute_nvgpu.sm103.make_umma_smem_desc_circular(%arg1 : !cute.ptr<f16, smem, align<16>)
      layout = <"S<3, 4, 3> o 0 o ((64,16),1):((64,1),0)">, major = #cute_nvgpu.major<k>
      -> !cute_nvgpu.smem_desc_circular
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte:$src,
    Optional<CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte>:$nextSrc,
    Cute_LayoutAttr:$layout,
    CuteNVGPU_MajorModeAttr:$major
  );
  let results = (outs
    AnyType:$res
  );
}
def CuteNVGPU_MakeUMMASmemDescOp : CuteNVGPU_Op<"make_umma_smem_desc", [Pure]> {
  let summary = "Example:";
  let description = [{
    ```mlir
    %1 = cute_nvgpu.make_umma_smem_desc(%arg1 : !cute.ptr<f16, smem, align<16>)
      layout = <"S<3, 4, 3> o 0 o ((64,16),1):((64,1),0)">, major = #cute_nvgpu.major<k>
      -> !cute_nvgpu.smem_desc
    ```
  }];
  let arguments = (ins
    CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte:$src,
    Optional<CuteNVGPU_CuTePointerToMemorySpacesSmemWithByte2>:$nextSrc,
    Cute_LayoutAttr:$layout,
    CuteNVGPU_MajorModeAttr:$major
  );
  let results = (outs
    CuteNVGPU_CuteSmemDescType:$res
  );
}
def CuteNVGPU_PrefetchTmaDescriptorOp : CuteNVGPU_Op<"prefetch_tma_desc"> {
  let arguments = (ins
    CuteNVGPU_TypeInterfaceForNonExecutableTMAAtomTypes:$atom
  );
  let results = (outs
  );
}
def CuteNVGPU_UpdateTMADescOp : CuteNVGPU_Op<"update_tma_desc"> {
  let description = [{
    This operation updates the TMA descriptor in smem or gmem with the information from given gmem tensor.
    Specifically, the following fields of the TMA descriptor will be updated:
      1. the global tensor base address will be updated with the base address of the global memory tensor.
      2. the tensor shapes will be updated with the shape of the global memory tensor.
      3. the tensor strides will be updated with the stride of the global memory tensor.
    Other fields of the TMA descriptor will be left unchanged.
    The TMA descriptor pointer specifies the address of the tma descriptor to be updated.
    It could be a smem or gmem pointer. The pointer should be aligned to at least 64B.
    Example:
    ```mlir
    cute_nvgpu.update_tma_desc(%tma_atom, %gmem_tensor, %smem_ptr) : (
      !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90_multicast, f16, copy_bits=131072, tma_gbasis=<"(64,128,1):(1@0,1@1,1@2)">>,
      !cute.memref<f16, gmem, align<128>, "(?,?,?) : (1,?,?)">,
      !cute.ptr<i64, smem, align<128>>)
    -> ()

    cute_nvgpu.update_tma_desc(%tma_atom, %gmem_tensor, %gmem_ptr) : (
      !cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90_multicast, f16, copy_bits=131072, tma_gbasis=<"(128,64,1):(1@1,1@0,1@2)">>,
      !cute.memref<f16, gmem, align<128>, "(?,?,?) : (?,1,?)">,
      !cute.ptr<i64, gmem, align<128>>)
    -> ()
    ```
  }];
  let arguments = (ins
    CuteNVGPU_TypeInterfaceForNonExecutableTMAAtomTypes:$tma_atom,
    CuteNVGPU_AMemoryReferenceTypeThatRepresentsATensor:$gmem_tensor,
    CuteNVGPU_CuTePointerToMemorySpacesSmemOrCuTe:$tma_desc_ptr
  );
  let results = (outs
  );
}

#endif // CUTE_IR_GENERATED_CUTENVGPU_OPS_TD
