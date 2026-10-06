// The cuda dialect's operations, generated from the nvidia-cutlass-dsl 4.8.0 release; do not edit.
//
// Sources: the operation bindings (`_cuda_ops_gen.py`: names, classes,
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
//   cuda.launch_cfg.create_from_stream
//
// Constraints not recovered:

#ifndef CUTE_IR_GENERATED_CUDA_OPS_TD
#define CUTE_IR_GENERATED_CUDA_OPS_TD

include "cute_ir/Dialect/Cuda/IR/CudaAttrs.td"
include "cute_ir/Dialect/Cuda/IR/CudaEnums.td"
include "cute_ir/Dialect/Cuda/IR/CudaTypes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Operand and result constraints, in the compiler's words.
def Cuda_AnyCUDAStructType : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeDevicePropType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeDim3Type>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeExtentType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeKernelNodeParamsType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeMemcpy3DParamsType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimePitchedPtrType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimePosType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeUuidType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueCooperativeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueMemSyncDomainType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValuePortableClusterSizeModeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValuePriorityType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueSharedMemCarveoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueSharedMemoryModeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueSyncPolicyType>($_self)">, "any CUDA struct type">;
def Cuda_AnyType : Type<CPred<"true">, "any type">;
def Cuda_CUDACudaDevicePropStructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeDevicePropType>($_self)">, "CUDA cudaDeviceProp struct type (value-typed).">, BuildableType<"::mlir::cutlass_compiler::cuda::RuntimeDevicePropType::get($_builder.getContext())">;
def Cuda_CUDACudaDevicePropStructTypeValueTyped2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeDevicePropType>($_self)">, "CUDA cudaDeviceProp struct type (value-typed).">;
def Cuda_CUDACudaExtentStructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeExtentType>($_self)">, "CUDA cudaExtent struct type (value-typed).">;
def Cuda_CUDACudaExtentStructTypeValueTyped2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeExtentType>($_self)">, "CUDA cudaExtent struct type (value-typed).">, BuildableType<"::mlir::cutlass_compiler::cuda::RuntimeExtentType::get($_builder.getContext())">;
def Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeKernelNodeParamsType>($_self)">, "CUDA cudaKernelNodeParams struct type (value-typed).">;
def Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeMemcpy3DParamsType>($_self)">, "CUDA cudaMemcpy3DParms struct type (value-typed).">;
def Cuda_CUDACudaPitchedPtrStructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimePitchedPtrType>($_self)">, "CUDA cudaPitchedPtr struct type (value-typed).">, BuildableType<"::mlir::cutlass_compiler::cuda::RuntimePitchedPtrType::get($_builder.getContext())">;
def Cuda_CUDACudaPitchedPtrStructTypeValueTyped2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimePitchedPtrType>($_self)">, "CUDA cudaPitchedPtr struct type (value-typed).">;
def Cuda_CUDACudaPosStructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimePosType>($_self)">, "CUDA cudaPos struct type (value-typed).">, BuildableType<"::mlir::cutlass_compiler::cuda::RuntimePosType::get($_builder.getContext())">;
def Cuda_CUDACudaPosStructTypeValueTyped2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimePosType>($_self)">, "CUDA cudaPos struct type (value-typed).">;
def Cuda_CUDACudaUUIDTStructTypeValue : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeUuidType>($_self)">, "CUDA cudaUUID_t struct type (value-typed).">, BuildableType<"::mlir::cutlass_compiler::cuda::RuntimeUuidType::get($_builder.getContext())">;
def Cuda_CUDACudaUUIDTStructTypeValue2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeUuidType>($_self)">, "CUDA cudaUUID_t struct type (value-typed).">;
def Cuda_CUDADim3StructTypeValueTyped : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeDim3Type>($_self)">, "CUDA dim3 struct type (value-typed).">;
def Cuda_CUDADim3StructTypeValueTyped2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeDim3Type>($_self)">, "CUDA dim3 struct type (value-typed).">, BuildableType<"::mlir::cutlass_compiler::cuda::RuntimeDim3Type::get($_builder.getContext())">;
def Cuda_CUDAGraph : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphType>($_self)">, "CUDA Graph">;
def Cuda_CUDAGraph2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphType>($_self)">, "CUDA Graph">, BuildableType<"::mlir::cutlass_compiler::cuda::GraphType::get($_builder.getContext())">;
def Cuda_CUDAGraphExecution : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphExecType>($_self)">, "CUDA Graph Execution">, BuildableType<"::mlir::cutlass_compiler::cuda::GraphExecType::get($_builder.getContext())">;
def Cuda_CUDAGraphExecution2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphExecType>($_self)">, "CUDA Graph Execution">;
def Cuda_CUDAGraphNodeHandle : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphNodeType>($_self)">, "CUDA Graph Node handle">, BuildableType<"::mlir::cutlass_compiler::cuda::GraphNodeType::get($_builder.getContext())">;
def Cuda_CUDAGraphNodeHandle2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphNodeType>($_self)">, "CUDA Graph Node handle">;
def Cuda_CUDAKernelHandle : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::KernelType>($_self)">, "CUDA Kernel handle">, BuildableType<"::mlir::cutlass_compiler::cuda::KernelType::get($_builder.getContext())">;
def Cuda_CUDAKernelHandle2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::KernelType>($_self)">, "CUDA Kernel handle">;
def Cuda_CUDALibraryHandle : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::LibraryType>($_self)">, "CUDA Library handle">;
def Cuda_CUDALibraryHandle2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::LibraryType>($_self)">, "CUDA Library handle">, BuildableType<"::mlir::cutlass_compiler::cuda::LibraryType::get($_builder.getContext())">;
def Cuda_CUDAModuleHandle : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::ModuleType>($_self)">, "CUDA Module handle">, BuildableType<"::mlir::cutlass_compiler::cuda::ModuleType::get($_builder.getContext())">;
def Cuda_CUDAResultType : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::ResultType>($_self)">, "CUDA Result type">, BuildableType<"::mlir::cutlass_compiler::cuda::ResultType::get($_builder.getContext())">;
def Cuda_CUDAResultType2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::ResultType>($_self)">, "CUDA Result type">;
def Cuda_CUDAResultTypeOrCUDAStream : Type<CPred<"$_self.isSignlessInteger(32) || $_self.isSignlessInteger(64) || ::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::StreamType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::ResultType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::EventType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::GraphDeviceNodeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::TensorMapType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::IntType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::LaunchConfigType>($_self) || ($_self.getDialect().getNamespace() == \"ptr\" || (::llvm::isa<::mlir::OpaqueType>($_self) && ::llvm::cast<::mlir::OpaqueType>($_self).getDialectNamespace() == \"ptr\"))">, "CUDA Result type or CUDA Stream or `Event` represents CUDA event or CUDA Tensor Map type or Representation of CUDA's extensible launch configuration mirroring cuda runtime API or `GraphDeviceNode` represents CUDA device node handle for device-side node update or `cuda.int` is a integer type with a bitwidth matching the underlying ABI integer bitwidth or 32-bit signless integer or 64-bit signless integer or pointer type or LLVM pointer type">;
def Cuda_CUDAStream : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::StreamType>($_self)">, "CUDA Stream">;
def Cuda_CUDAStream2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::StreamType>($_self)">, "CUDA Stream">, BuildableType<"::mlir::cutlass_compiler::cuda::StreamType::get($_builder.getContext())">;
def Cuda_CUDATensorMapType : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::TensorMapType>($_self)">, "CUDA Tensor Map type">, BuildableType<"::mlir::cutlass_compiler::cuda::TensorMapType::get($_builder.getContext())">;
def Cuda_CudaIntIsAIntegerType : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::IntType>($_self)">, "`cuda.int` is a integer type with a bitwidth matching the underlying ABI integer bitwidth">;
def Cuda_EventRepresentsCUDAEvent : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::EventType>($_self)">, "`Event` represents CUDA event">, BuildableType<"::mlir::cutlass_compiler::cuda::EventType::get($_builder.getContext())">;
def Cuda_EventRepresentsCUDAEvent2 : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::EventType>($_self)">, "`Event` represents CUDA event">;
def Cuda_FixedLengthVectorOf32Bit : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 32-bit signless integer values of length 3">;
def Cuda_FixedLengthVectorOf32Bit2 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 32-bit signless integer values of length 2">;
def Cuda_FixedLengthVectorOf8Bit : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 8-bit signless integer values of length 8">;
def Cuda_FixedLengthVectorOf8Bit2 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 8-bit signless integer values of length 256">;
def Cuda_FixedLengthVectorOf8Bit3 : Type<CPred<"::llvm::isa<::mlir::VectorType>($_self)">, "fixed-length vector of 8-bit signless integer values of length 16">;
def Cuda_GraphDeviceNodeRepresentsCUDADeviceNodeHandle : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::GraphDeviceNodeType>($_self)">, "`GraphDeviceNode` represents CUDA device node handle for device-side node update">;
def Cuda_LLVMPointerInAddressSpace0 : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer in address space 0">;
def Cuda_LLVMPointerType : Type<CPred<"::llvm::isa<::mlir::LLVM::LLVMPointerType>($_self)">, "LLVM pointer type">;
def Cuda_PointerType : Type<CPred<"($_self.getDialect().getNamespace() == \"ptr\" || (::llvm::isa<::mlir::OpaqueType>($_self) && ::llvm::cast<::mlir::OpaqueType>($_self).getDialectNamespace() == \"ptr\"))">, "pointer type">;
def Cuda_RepresentationOfCUDASExtensibleLaunch : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::LaunchConfigType>($_self)">, "Representation of CUDA's extensible launch configuration mirroring cuda runtime API">;
def Cuda_T1BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(1)">, "1-bit signless integer">, BuildableType<"$_builder.getIntegerType(1)">;
def Cuda_T32BitFloat : Type<CPred<"::llvm::isa<::mlir::Float32Type>($_self)">, "32-bit float">, BuildableType<"::mlir::Float32Type::get($_builder.getContext())">;
def Cuda_T32BitFloat2 : Type<CPred<"::llvm::isa<::mlir::Float32Type>($_self)">, "32-bit float">;
def Cuda_T32BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(32)">, "32-bit signless integer">;
def Cuda_T32BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(32)">, "32-bit signless integer">, BuildableType<"$_builder.getIntegerType(32)">;
def Cuda_T64BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(64)">, "64-bit signless integer">;
def Cuda_T64BitSignlessInteger2 : Type<CPred<"$_self.isSignlessInteger(64)">, "64-bit signless integer">, BuildableType<"$_builder.getIntegerType(64)">;
def Cuda_VariadicOf32BitSignlessInteger : Type<CPred<"$_self.isSignlessInteger(32)">, "variadic of 32-bit signless integer">;
def Cuda_VariantCooperativeOfCUDACudaLaunchAttributeValueUnion : Type<CPred<"::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueCooperativeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueMemSyncDomainType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValuePortableClusterSizeModeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValuePriorityType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueSharedMemCarveoutType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueSharedMemoryModeType>($_self) || ::llvm::isa<::mlir::cutlass_compiler::cuda::RuntimeLaunchAttributeValueSyncPolicyType>($_self)">, "Variant `cooperative` of CUDA `cudaLaunchAttributeValue` union (payload: int). or Variant `syncPolicy` of CUDA `cudaLaunchAttributeValue` union (payload: enum cudaSynchronizationPolicy). or Variant `clusterSchedulingPolicyPreference` of CUDA `cudaLaunchAttributeValue` union (payload: enum cudaClusterSchedulingPolicy). or Variant `programmaticStreamSerializationAllowed` of CUDA `cudaLaunchAttributeValue` union (payload: int). or Variant `priority` of CUDA `cudaLaunchAttributeValue` union (payload: int). or Variant `memSyncDomain` of CUDA `cudaLaunchAttributeValue` union (payload: cudaLaunchMemSyncDomain). or Variant `sharedMemCarveout` of CUDA `cudaLaunchAttributeValue` union (payload: unsigned int). or Variant `nvlinkUtilCentricScheduling` of CUDA `cudaLaunchAttributeValue` union (payload: unsigned int). or Variant `portableClusterSizeMode` of CUDA `cudaLaunchAttributeValue` union (payload: cudaLaunchAttributePortableClusterMode). or Variant `sharedMemoryMode` of CUDA `cudaLaunchAttributeValue` union (payload: cudaSharedMemoryMode).">;

// Attribute constraints the bindings name but do not describe.
def Cuda_DenseResourceElementsAttr : Attr<CPred<"::llvm::isa<::mlir::DenseResourceElementsAttr>($_self)">, "dense resource elements attribute"> {
  let storageType = "::mlir::DenseResourceElementsAttr";
  let returnType = "::mlir::DenseResourceElementsAttr";
  let convertFromStorage = "$_self";
}

def Cuda_FuncFunction_typeAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of function type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def Cuda_KernelFunction_typeAttr : Attr<CPred<"::llvm::isa<::mlir::TypeAttr>($_self) && (::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>($_self).getValue()))">, "type attribute of function type"> {
  let storageType = "::mlir::TypeAttr";
  let returnType = "::mlir::TypeAttr";
  let convertFromStorage = "$_self";
}

def Cuda_IntegerAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">, "integer attribute"> {
  let storageType = "::mlir::IntegerAttr";
  let returnType = "::mlir::IntegerAttr";
  let convertFromStorage = "$_self";
}

// Operations whose result types are inferred in C++ (CudaInference.cpp).
// Verification compares them with the inferred ones where the rule is
// recovered; elsewhere the types written stand, as the compiler accepted them.
class Cuda_InferredOp<string mnemonic, list<Trait> traits = []>
    : Cuda_Op<mnemonic, traits # [DeclareOpInterfaceMethods<InferTypeOpInterface, ["refineReturnTypes"]>]> {
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

def Cuda_AddressOfOp : Cuda_Op<"addressof"> {
  let arguments = (ins
    SymbolRefAttr:$symname
  );
  let results = (outs
    Cuda_PointerType:$result
  );
}
def Cuda_BinaryOp : Cuda_Op<"binary", [NoTerminator]> {
  let description = [{
    This operation represents a CUDA binary, which contains a number of functions and
    a serialized representation of the binary.

    Examples:

    ```mlir

    cuda.binary @binary_name <...> {
      cuda.kernel @function_name function_type
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Cuda_DenseResourceElementsAttr:$binary
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def Cuda_CastOp : Cuda_Op<"cast", [Pure]> {
  let description = [{
    Casts between CUDA types and standard dialect or built-in types.
    The source/destination types are statically checked and the IR
    is invalid if the cast is not possible.

    Examples:
    ```mlir

    %res_i32 = cuda.cast %res : cuda.result -> i32
    %res1    = cuda.cast %res_i32 : i32 -> cuda.result

    %s_ptr     = cuda.cast %hstream : cuda.stream -> ptr.ptr
    %s_i64     = cuda.cast %hstream : cuda.stream -> i64
    ```
  }];
  let arguments = (ins
    Cuda_CUDAResultTypeOrCUDAStream:$src
  );
  let results = (outs
    Cuda_CUDAResultTypeOrCUDAStream:$dst
  );
}
def Cuda_ComposeOp : Cuda_Op<"compose"> {
  let description = [{
    Generic constructor for any auto-generated CUDA struct type. The result
    type identifies which struct is being built; operands are the struct's
    non-padding fields in declaration order. The verifier checks operand
    count and per-operand type against the result type's
    `CudaStructTypeInterface::getOperandFieldTypes()`.

    Lowers to `llvm.mlir.undef` + a chain of `llvm.insertvalue` (one per
    operand-mapped slot from `getOperandFieldIndices()`). Pure value SSA;
    no memory.

    Example:
    ```mlir
    %pos = cuda.compose %x, %y, %z
         : i64, i64, i64 -> !cuda.runtime.pos
    ```
  }];
  let arguments = (ins
    Variadic<Cuda_AnyType>:$fields
  );
  let results = (outs
    Cuda_AnyCUDAStructType:$result
  );
}
def Cuda_DecomposeOp : Cuda_Op<"decompose"> {
  let description = [{
    Generic accessor: returns each non-padding field of a CUDA struct value
    as an SSA value, in declaration order. The verifier checks result count
    and per-result type against the source type's
    `CudaStructTypeInterface::getOperandFieldTypes()`.

    Lowers to a chain of `llvm.extractvalue` (one per operand-mapped slot
    from `getOperandFieldIndices()`). Pure value SSA; no memory.

    Example:
    ```mlir
    %x, %y, %z = cuda.decompose %pos
               : !cuda.runtime.pos -> i64, i64, i64
    ```
  }];
  let arguments = (ins
    Cuda_AnyCUDAStructType:$source
  );
  let results = (outs
    Optional<Cuda_AnyType>:$fields
  );
}
def Cuda_DeviceSyncOp : Cuda_Op<"device.sync"> {
  let description = [{
    Operation to synchronize with the device. See cudaDeviceSynchronize()
  }];
  let arguments = (ins);
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverEventCreateOp : Cuda_Op<"driver.event.create"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$Flags
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_EventRepresentsCUDAEvent:$phEvent
  );
}
def Cuda_DriverEventDestroyOp : Cuda_Op<"driver.event.destroy"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$hEvent
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverEventElapsedTimeOp : Cuda_Op<"driver.event.elapsed_time"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$hStart,
    Cuda_EventRepresentsCUDAEvent2:$hEnd
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_T32BitFloat:$pMilliseconds
  );
}
def Cuda_DriverEventQueryOp : Cuda_Op<"driver.event.query", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$hEvent
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverEventRecordOp : Cuda_Op<"driver.event.record"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$hEvent,
    Cuda_CUDAStream:$hStream
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverEventRecordWithFlagsOp : Cuda_Op<"driver.event.record_with_flags"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$hEvent,
    Cuda_CUDAStream:$hStream,
    Cuda_T32BitSignlessInteger:$flags
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverEventSynchronizeOp : Cuda_Op<"driver.event.synchronize"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$hEvent
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverLibraryGetGlobalOp : Cuda_Op<"driver.library.get_global", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$name_arg
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$dptr,
    Cuda_T64BitSignlessInteger:$bytes
  );
}
def Cuda_DriverLibraryGetKernelOp : Cuda_Op<"driver.library.get_kernel", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$name_arg
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDAKernelHandle:$pKernel
  );
}
def Cuda_DriverLibraryGetKernelCountOp : Cuda_Op<"driver.library.get_kernel_count", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$lib
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_T32BitSignlessInteger2:$count
  );
}
def Cuda_DriverLibraryGetManagedOp : Cuda_Op<"driver.library.get_managed", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$name_arg
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$dptr,
    Cuda_T64BitSignlessInteger:$bytes
  );
}
def Cuda_DriverLibraryGetModuleOp : Cuda_Op<"driver.library.get_module", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDAModuleHandle:$pMod
  );
}
def Cuda_DriverLibraryGetUnifiedFunctionOp : Cuda_Op<"driver.library.get_unified_function", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$symbol
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$fptr
  );
}
def Cuda_DriverLibraryLoadDataOp : Cuda_Op<"driver.library.load_data"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_PointerType:$code_arg,
    Cuda_PointerType:$jitOptions,
    Cuda_PointerType:$jitOptionsValues,
    Cuda_T32BitSignlessInteger:$numJitOptions,
    Cuda_PointerType:$libraryOptions,
    Cuda_PointerType:$libraryOptionValues,
    Cuda_T32BitSignlessInteger:$numLibraryOptions
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDALibraryHandle2:$library
  );
}
def Cuda_DriverLibraryLoadFromFileOp : Cuda_Op<"driver.library.load_from_file"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_PointerType:$fileName,
    Cuda_PointerType:$jitOptions,
    Cuda_PointerType:$jitOptionsValues,
    Cuda_T32BitSignlessInteger:$numJitOptions,
    Cuda_PointerType:$libraryOptions,
    Cuda_PointerType:$libraryOptionValues,
    Cuda_T32BitSignlessInteger:$numLibraryOptions
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDALibraryHandle2:$library
  );
}
def Cuda_DriverLibraryUnloadOp : Cuda_Op<"driver.library.unload"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_DriverTensorMapEncodeIm2colOp : Cuda_Op<"driver.tensor_map.encode_im2col"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$tensorDataType,
    Cuda_T32BitSignlessInteger:$tensorRank,
    Cuda_PointerType:$globalAddress,
    Cuda_PointerType:$globalDim,
    Cuda_PointerType:$globalStrides,
    Cuda_PointerType:$pixelBoxLowerCorner,
    Cuda_PointerType:$pixelBoxUpperCorner,
    Cuda_T32BitSignlessInteger:$channelsPerPixel,
    Cuda_T32BitSignlessInteger:$pixelsPerColumn,
    Cuda_PointerType:$elementStrides,
    Cuda_T32BitSignlessInteger:$interleave,
    Cuda_T32BitSignlessInteger:$swizzle,
    Cuda_T32BitSignlessInteger:$l2Promotion,
    Cuda_T32BitSignlessInteger:$oobFill
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDATensorMapType:$tensorMap
  );
}
def Cuda_DriverTensorMapEncodeIm2colWideOp : Cuda_Op<"driver.tensor_map.encode_im2col_wide"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$tensorDataType,
    Cuda_T32BitSignlessInteger:$tensorRank,
    Cuda_PointerType:$globalAddress,
    Cuda_PointerType:$globalDim,
    Cuda_PointerType:$globalStrides,
    Cuda_T32BitSignlessInteger:$pixelBoxLowerCornerWidth,
    Cuda_T32BitSignlessInteger:$pixelBoxUpperCornerWidth,
    Cuda_T32BitSignlessInteger:$channelsPerPixel,
    Cuda_T32BitSignlessInteger:$pixelsPerColumn,
    Cuda_PointerType:$elementStrides,
    Cuda_T32BitSignlessInteger:$interleave,
    Cuda_T32BitSignlessInteger:$mode,
    Cuda_T32BitSignlessInteger:$swizzle,
    Cuda_T32BitSignlessInteger:$l2Promotion,
    Cuda_T32BitSignlessInteger:$oobFill
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDATensorMapType:$tensorMap
  );
}
def Cuda_DriverTensorMapReplaceAddressOp : Cuda_Op<"driver.tensor_map.replace_address"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_PointerType:$globalAddress
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDATensorMapType:$tensorMap
  );
}
def Cuda_EventCreateOp : Cuda_Op<"event.create"> {
  let description = [{
    Operation to create a CUDA event.
  }];
  let arguments = (ins);
  let results = (outs
    Cuda_CUDAResultType:$result,
    Cuda_EventRepresentsCUDAEvent:$event
  );
}
def Cuda_EventDestroyOp : Cuda_Op<"event.destroy"> {
  let description = [{
    Operation to destroy a CUDA event.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$event
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_EventElapsedTimeOp : Cuda_Op<"event.elapsed_time"> {
  let description = [{
    Operation to compute the elapsed time between events
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$start_event,
    Cuda_EventRepresentsCUDAEvent2:$end_event
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_T32BitFloat:$elapsed_time
  );
}
def Cuda_EventRecordOp : Cuda_Op<"event.record"> {
  let description = [{
    Operation to record an event in a CUDA stream.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$event,
    Optional<Cuda_CUDAStream>:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_FuncOp : Cuda_Op<"func"> {
  let description = [{
    This operation represents a CUDA device function which can be called from
    another `cuda.func` or a `cuda.kernel`. It cannot be launched from the host.

    Examples:

    ```mlir

    cuda.module @module_name {

    cuda.func @device_func(%arg0 : i32) -> i32 {
      %0 = arith.constant 0 : i32
      cuda.return %0 : i32
    }

    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Cuda_FuncFunction_typeAttr:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def Cuda_FuncGetNameOp : Cuda_Op<"func.get_name"> {
  let arguments = (ins
    Cuda_CUDAKernelHandle2:$kernel
  );
  let results = (outs
    Cuda_CUDAResultType2:$result,
    Cuda_PointerType:$name
  );
}
def Cuda_GraphDestroyOp : Cuda_Op<"graph.destroy"> {
  let description = [{
    Operation to destroy a CUDA graph.
  }];
  let arguments = (ins
    Cuda_CUDAGraph:$graph
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_GraphInstantiateOp : Cuda_Op<"graph.instantiate"> {
  let description = [{
    Operation to instantiate a CUDA graph.
  }];
  let arguments = (ins
    Cuda_CUDAGraph:$graph
  );
  let results = (outs
    Cuda_CUDAResultType:$result,
    Cuda_CUDAGraphExecution:$graph_exec
  );
}
def Cuda_GraphLaunchOp : Cuda_Op<"graph.launch"> {
  let description = [{
    Operation to launch a CUDA graph.
  }];
  let arguments = (ins
    Cuda_CUDAGraphExecution2:$graph_exec,
    Cuda_CUDAStream:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_GraphExecDestroyOp : Cuda_Op<"graph_exec.destroy"> {
  let description = [{
    Operation to destroy a CUDA graph execution.
  }];
  let arguments = (ins
    Cuda_CUDAGraphExecution2:$graph_exec
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_KernelOp : Cuda_Op<"kernel"> {
  let description = [{
    This operation represents a CUDA kernel which may be launched as a grid. It
    differs from a `cuda.func`.

    Examples:

    ```mlir

    cuda.module @module_name {

    cuda.func @kernel(%arg0 : i32)
      attributes {
        cu_attrs = {
          num_regs = 128,
          max_dynamic_shared_size_bytes = 2048
          // ... add more cuda function attributes here
        }
      }
    }

    ...

    %res = cuda.launch @module_name::@device_func <<< %grid, %block, %cluster, %smem, %stream >>>(%0)

    %0 = cuda.result.succeeded(%res)
    scf.if %0 {

    }
    else {

    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Cuda_KernelFunction_typeAttr:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<Cuda_FuncAttributesAttr>:$cu_func_attrs
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
}
def Cuda_LaunchOp : Cuda_Op<"launch", [AttrSizedOperandSegments]> {
  let description = [{
    Operation to launch a `cuda.kernel` on the device.

    Examples:

    ```mlir

    cuda.module @module_name {

    cuda.func @kernel(%arg0 : i32) {

    }

    }

    ...

    %0 = ... // kernel arguments

    %res = cuda.launch @module_name::@device_func<<< %grid, %block, %smem, %stream >>>(%0) : !cuda.launch_cfg<max_attr = ...>, ... -> !cuda.result

    // If calling with an Op that implements a FunctionOpInterface
    %res = cuda.launch @module_name::@device_func<<< %grid, %block, %smem, %stream >>>(%0) : !cuda.launch_cfg<max_attr = ...>, ... <assume_kernel_attr = true> -> !cuda.result

    %res2 = cuda.result.succeeded(%res)
    scf.if %res2 {

    }
    else {

    }
    ```
  }];
  let arguments = (ins
    Variadic<Cuda_VariadicOf32BitSignlessInteger>:$grid,
    Variadic<Cuda_VariadicOf32BitSignlessInteger>:$block,
    Cuda_T64BitSignlessInteger:$dynamicSmem,
    Cuda_CUDAStream:$cudaStream,
    Variadic<Cuda_AnyType>:$inputs,
    SymbolRefAttr:$callee,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<Cuda_AssumeKernelAttr>:$assume_kernel_attr
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_LaunchCfgAccessPolicyWindowOp : Cuda_Op<"launch_cfg.access_policy_window"> {
  let description = [{
    Sets accessPolicyWindow for a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create<max_attrs = 3>(...)
      cuda.launch_cfg.access_policy_window [%cfg0] (
        base_ptr=%basePtr,
        hitProp=%hitProp,
        hitRatio=%hitRatio,
        missProp=%missProp,
        numBytes=%numBytes
      ) : !cuda.launch_cfg<max_attrs = 3>, !ptr.ptr<#ptr.generic_space>, f32, i64
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_PointerType:$base_ptr,
    Cuda_T32BitFloat2:$hitRatio,
    Cuda_T64BitSignlessInteger:$num_bytes,
    Cuda_CudaAccessPropertyEnum:$hitProp,
    Cuda_CudaAccessPropertyEnum:$missProp
  );
  let results = (outs);
}
def Cuda_LaunchCfgClusterDimOp : Cuda_Op<"launch_cfg.cluster_dim"> {
  let description = [{
    Sets clusterDim for a launch_config

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create<max_attrs = 3>(...)
      cuda.launch_cfg.cluster_dim [%cfg0] (1, 2, 8) : !cuda.launch_cfg<max_attrs = 3>, i32, i32, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$clusterDimX,
    Cuda_T32BitSignlessInteger:$clusterDimY,
    Cuda_T32BitSignlessInteger:$clusterDimZ
  );
  let results = (outs);
}
def Cuda_LaunchCfgClusterSchedulingPolicyOp : Cuda_Op<"launch_cfg.cluster_scheduling_policy"> {
  let description = [{
    Sets clusterSchedulingPolicyPreference for a launch_config

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.cluster_scheduling_policy [%cfg0] cudaClusterSchedulingPolicyLoadBalancing
        : !cuda.launch_cfg<max_attrs = ...>
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_CudaClusterSchedulingPolicyEnum:$policy
  );
  let results = (outs);
}
def Cuda_LaunchCfgCooperativeOp : Cuda_Op<"launch_cfg.cooperative"> {
  let description = [{
    Sets cooperative attribute to a launch_config. Nonzero indicates a cooperative kernel.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.cooperative [%cfg0] 1
        : !cuda.launch_cfg<max_attr = ...>, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$cooperative
  );
  let results = (outs);
}
def Cuda_LaunchCfgCreateOp : Cuda_Op<"launch_cfg.create"> {
  let description = [{
    Creates a launch_config object with the specified runtime values.
    The result is a Cuda_LaunchConfig type that contains all the runtime data.

    Example:
    ```mlir
    %cfg = cuda.launch_cfg.create<max_attrs = 3>(
      blockDim=(%block_x, %block_y, %block_z),
      dynamicSmemBytes=%dynamic_smem_bytes,
      gridDim=(%grid_x, %grid_y, %grid_z),
      stream=%hstream
    ) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 3>
    ```
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$blockDimX,
    Cuda_T32BitSignlessInteger:$blockDimY,
    Cuda_T32BitSignlessInteger:$blockDimZ,
    Cuda_T64BitSignlessInteger:$dynamicSmemBytes,
    Cuda_T32BitSignlessInteger:$gridDimX,
    Cuda_T32BitSignlessInteger:$gridDimY,
    Cuda_T32BitSignlessInteger:$gridDimZ,
    Cuda_CUDAStream:$stream,
    Cuda_IntegerAttr:$maxNumAttrs
  );
  let results = (outs
    Cuda_RepresentationOfCUDASExtensibleLaunch:$result
  );
}
def Cuda_LaunchCfgCreateFromStreamOp : Cuda_InferredOp<"launch_cfg.create_from_stream"> {
  let description = [{
    Creates a launch_config object initialized with a CUDA stream and space for
    the specified number of launch attributes. Grid dimensions, block
    dimensions, and dynamic shared memory bytes can be supplied later with
    `cuda.launch_cfg.grid_dimensions`.

    Example:
    ```mlir
    %cfg = cuda.launch_cfg.create_from_stream<max_attrs = 16>(
      stream=%hstream
    ) : !cuda.stream
    ```
  }];
  let arguments = (ins
    Cuda_CUDAStream:$stream,
    Cuda_IntegerAttr:$maxNumAttrs
  );
  let results = (outs
    Cuda_RepresentationOfCUDASExtensibleLaunch:$result
  );
}
def Cuda_LaunchCfgDeviceUpdatableKernelNodeOp : Cuda_Op<"launch_cfg.device_updatable_kernel_node"> {
  let description = [{
    Sets deviceUpdatableKernelNode attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.device_updatable_kernel_node [%cfg0] (devNode=%devNode, deviceUpdatable=%deviceUpdatable)
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_GraphDeviceNodeRepresentsCUDADeviceNodeHandle:$devNode,
    Cuda_T32BitSignlessInteger:$deviceUpdatable
  );
  let results = (outs);
}
def Cuda_LaunchCfgGridDimensionsOp : Cuda_Op<"launch_cfg.grid_dimensions"> {
  let description = [{
    Sets the fixed launch dimensions for a launch_config object. This updates
    the direct `cudaLaunchConfig_t` fields, not the launch attribute array.

    Example:
    ```mlir
    cuda.launch_cfg.grid_dimensions [%cfg0]
      grid(%gx, %gy, %gz), block(%bx, %by, %bz), dynamicSmem(%smem)
      : !cuda.launch_cfg<max_attrs = 16>, i32, i32, i32, i32, i32, i32, i64
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$gridDimX,
    Cuda_T32BitSignlessInteger:$gridDimY,
    Cuda_T32BitSignlessInteger:$gridDimZ,
    Cuda_T32BitSignlessInteger:$blockDimX,
    Cuda_T32BitSignlessInteger:$blockDimY,
    Cuda_T32BitSignlessInteger:$blockDimZ,
    Cuda_T64BitSignlessInteger:$dynamicSmemBytes
  );
  let results = (outs);
}
def Cuda_LaunchCfgLaunchCompletionEventOp : Cuda_Op<"launch_cfg.launch_completion_event"> {
  let description = [{
    Sets launchCompletionEvent attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.launch_completion_event [%cfg0] (event=%event, flags=%flags)
        : !cuda.launch_cfg<max_attr = ...>, !cuda.event, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_EventRepresentsCUDAEvent2:$event,
    Cuda_T32BitSignlessInteger:$flags
  );
  let results = (outs);
}
def Cuda_LaunchCfgMemSyncDomainOp : Cuda_Op<"launch_cfg.mem_sync_domain"> {
  let description = [{
    Sets memSyncDomain attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.mem_sync_domain [%cfg0] cudaMemSyncDomainRemote
        : !cuda.launch_cfg<max_attr = ...>
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_CudaLaunchMemSyncDomainEnum:$memSyncDomain
  );
  let results = (outs);
}
def Cuda_LaunchCfgMemSyncDomainMapOp : Cuda_Op<"launch_cfg.mem_sync_domain_map"> {
  let description = [{
    Sets memSyncDomainMap attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.mem_sync_domain_map [%cfg0] (domain_default=0, domain_remote=1)
        : !cuda.launch_cfg<max_attr = ...>, i32, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$domain_default,
    Cuda_T32BitSignlessInteger:$domain_remote
  );
  let results = (outs);
}
def Cuda_LaunchCfgNvlinkUtilCentricSchedulingOp : Cuda_Op<"launch_cfg.nvlink_util_centric_scheduling"> {
  let description = [{
    Sets nvlinkUtilCentricScheduling attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.nvlink_util_centric_scheduling [%cfg0] 1 : !cuda.launch_cfg<max_attr = ...>, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$nvlinkSchedulingEnabled
  );
  let results = (outs);
}
def Cuda_LaunchCfgPreferredClusterDimOp : Cuda_Op<"launch_cfg.preferred_cluster_dim"> {
  let description = [{
    Sets preferredClusterDim attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.preferred_cluster_dim [%cfg0] (1, 1, 8)
        : !cuda.launch_cfg<max_attr = ...>, i32, i32, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$preferredClusterDimX,
    Cuda_T32BitSignlessInteger:$preferredClusterDimY,
    Cuda_T32BitSignlessInteger:$preferredClusterDimZ
  );
  let results = (outs);
}
def Cuda_LaunchCfgPrintOp : Cuda_Op<"launch_cfg.print"> {
  let description = [{
    Prints a cuda launch config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create<max_attrs = 3>(...)
      cuda.launch_cfg.print [%cfg0] : !cuda.launch_cfg<max_attrs = 3>
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg
  );
  let results = (outs);
}
def Cuda_LaunchCfgPriorityOp : Cuda_Op<"launch_cfg.priority"> {
  let description = [{
    Sets priority attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.priority [%cfg0] 1
        : !cuda.launch_cfg<max_attr = ...>, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$priority
  );
  let results = (outs);
}
def Cuda_LaunchCfgProgrammaticEventOp : Cuda_Op<"launch_cfg.programmatic_event"> {
  let description = [{
    Sets programmaticEvent attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.programmatic_event [%cfg0] (
        event=%event,
        flags=%flags,
        triggerAtBlockStart=%triggerAtBlockStart)
        : !cuda.launch_cfg<max_attr = ...>, !cuda.event, i32, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_EventRepresentsCUDAEvent2:$event,
    Cuda_T32BitSignlessInteger:$flags,
    Cuda_T32BitSignlessInteger:$triggerAtBlockStart
  );
  let results = (outs);
}
def Cuda_LaunchCfgProgrammaticStreamSerializationAllowedOp : Cuda_Op<"launch_cfg.programmatic_stream_serialization_allowed"> {
  let description = [{
    Sets programmaticStreamSerializationAllowed attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.programmatic_stream_serialization_allowed [%cfg0] 1
        : !cuda.launch_cfg<max_attr = ...>, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$programmaticStreamSerializationAllowed
  );
  let results = (outs);
}
def Cuda_LaunchCfgSharedMemCarveoutOp : Cuda_Op<"launch_cfg.shared_mem_carveout"> {
  let description = [{
    Sets sharedMemCarveout attribute to a launch_config.

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.shared_mem_carveout [%cfg0] 1024
        : !cuda.launch_cfg<max_attr = ...>, i32
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_T32BitSignlessInteger:$sharedMemCarveout
  );
  let results = (outs);
}
def Cuda_LaunchCfgSyncPolicyOp : Cuda_Op<"launch_cfg.sync_policy"> {
  let description = [{
    Sets syncPolicy for a launch_config

    Example:
    ```mlir
      %cfg0 = cuda.launch_cfg.create...
      cuda.launch_cfg.sync_policy [%cfg0] cudaSynchronizationPolicyDefault
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Cuda_CudaSynchronizationPolicyEnum:$syncPolicy
  );
  let results = (outs);
}
def Cuda_LaunchExOp : Cuda_Op<"launch_ex"> {
  let description = [{
    Operation to launch a `cuda.kernel` on the device using extensible launch API.

    Examples:

    ```mlir

    cuda.module @module_name {

    cuda.func @kernel(%arg0 : i32) {

    }

    }

    ...
    %0 = ... // kernel arguments

    %res = cuda.launch_ex @module_name::@device_func<%cfg>(%0) : !cuda.launch_cfg<max_attr = ...>, ... -> !cuda.result

    // If calling with an Op that implements a FunctionOpInterface
    %res = cuda.launch_ex @module_name::@device_func<%cfg>(%0) : !cuda.launch_cfg<max_attr = ...>, ... <assume_kernel_attr = true> -> !cuda.result

    %res2 = cuda.result.succeeded(%res)
    scf.if %res2 {

    }
    else {

    }
    ```
  }];
  let arguments = (ins
    Cuda_RepresentationOfCUDASExtensibleLaunch:$launch_cfg,
    Variadic<Cuda_AnyType>:$inputs,
    SymbolRefAttr:$callee,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    OptionalAttr<Cuda_AssumeKernelAttr>:$assume_kernel_attr
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_LaunchKernelOp : Cuda_Op<"launch_kernel", [AttrSizedOperandSegments]> {
  let description = [{
    Operation to launch a kernel from a kernel pointer on the device. This launch API
    directly follows the CUDA C++ runtime API, passing all kernel arguments as a pointer
    to a struct.

    Example:

    ```mlir
      %res2 = cuda.launch_kernel %kernel, (%one32, %one32, %one32), (%one32, %one32, %one32), %zero64, %stream, %zeroPtr : !ptr.ptr<#ptr.generic_space> -> !cuda.result

    ```
  }];
  let arguments = (ins
    Cuda_CUDAKernelHandle2:$kernel,
    Variadic<Cuda_VariadicOf32BitSignlessInteger>:$grid,
    Variadic<Cuda_VariadicOf32BitSignlessInteger>:$block,
    Cuda_T64BitSignlessInteger:$dynamicSmem,
    Cuda_CUDAStream:$cudaStream,
    Cuda_PointerType:$kernel_args,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_LibraryGetKernelOp : Cuda_Op<"library.get_kernel"> {
  let description = [{
    Return a kernel handle object if found in the given library.

    ## Example

    ```
    %result_0, %kernel = cuda.library.get_kernel %l1, %name : !cuda.library, !ptr.ptr<#ptr.generic_space>  ->  !cuda.result, !cuda.kernel
    ```

    ## Behavior

    This op executes the CUDA API function cudaGetKernel. If the kernel is not found the cuda result will be `cudaErrorSymbolNotFound`. If the `kernel_name` is not a valid, null-terminated ASCII string, the behavior is UNDEFINED. Upon a result of `cudaSuccess`, `kernel` contains a valid kernel handle object. Is the result is not `cudaSuccess`, and use of the `kernel_name` handle is UNDEFINED.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$kernel_name
  );
  let results = (outs
    Cuda_CUDAResultType:$result,
    Cuda_CUDAKernelHandle:$kernel
  );
}
def Cuda_LibraryLoadDataOp : Cuda_Op<"library.load_data"> {
  let description = [{
    Load a library from a pointer.

    ## Example

    ```
    %result,%l1 = cuda.library.load_data %modptr : !ptr.ptr<#ptr.generic_space> -> !cuda.result, !cuda.library
    ```
    ## Behavior

    This op execute the CUDA API function cudaLibraryLoadData. Whether the library is loaded immediately depends on the module load environment.
  }];
  let arguments = (ins
    Cuda_PointerType:$data
  );
  let results = (outs
    Cuda_CUDAResultType:$result,
    Cuda_CUDALibraryHandle2:$library
  );
}
def Cuda_LibraryUnloadOp : Cuda_Op<"library.unload"> {
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_MemoryAllocOp : Cuda_Op<"memory.alloc"> {
  let description = [{
    Operation to allocate device memory. Calls `cuMemAlloc()`.

    Examples:
    ```mlir

    %bytes = arith.constant 0 : i64
    %res, %device_mem = cuda.memory.alloc(%bytes) : i64 -> cuda.result, cuda.ptr

    %res0 = cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%device_mem)

    %res1 = cuda.memory.free(%device_mem) : i64 -> cuda.result
    ```
  }];
  let arguments = (ins
    Cuda_T64BitSignlessInteger:$bytes
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$ptr
  );
}
def Cuda_MemoryAllocAsyncOp : Cuda_Op<"memory.alloc_async"> {
  let description = [{
    Operation to allocate device memory asynchronously. Calls `cuMemAllocAsync()`.

    Examples:
    ```mlir

    %res, %device_mem = cuda.memory.alloc_async(%bytes, %stream) : i64, cuda.ptr -> cuda.result

    cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%device_mem)

    $res1 = cuda.memory.free_async(%device_mem, %stream) : i64, !cuda.stream -> cuda.result
    ```
  }];
  let arguments = (ins
    Cuda_T64BitSignlessInteger:$bytes,
    Cuda_CUDAStream:$stream
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$ptr
  );
}
def Cuda_MemoryAllocHostOp : Cuda_Op<"memory.alloc_host"> {
  let description = [{
    Operation to allocate host memory. Calls `cuMemAllocHost()`.

    Examples:
    ```mlir

    %bytes = arith.constant 0 : i64
    %host_mem, %res = cuda.memory.alloc_host(%bytes) : (i64) -> (cuda.ptr, cuda.result)

    %res0 = cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%host_mem)

    %res1 = cuda.memory.free_host(%host_mem)
    ```
  }];
  let arguments = (ins
    Cuda_T64BitSignlessInteger:$bytes
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$ptr
  );
}
def Cuda_MemoryAllocManagedOp : Cuda_Op<"memory.alloc_managed"> {
  let description = [{
    Operation to allocate managed memory.

    Examples:
    ```mlir

    %bytes = arith.constant 0 : i64
    %managed_mem, %res = cuda.memory.alloc_managed(%bytes) : (i64) -> (cuda.ptr, cuda.result)

    %res0 = cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%managed_mem)

    %res1 = cuda.memory.free(%managed_mem)
    ```
  }];
  let arguments = (ins
    Cuda_T64BitSignlessInteger:$bytes
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$ptr
  );
}
def Cuda_MemoryCopyOp : Cuda_Op<"memory.copy"> {
  let description = [{
    Operation to copy memory between host and device.

    Examples:
    ```mlir

    %bytes = arith.constant 1024 : i64

    %device_mem, %res = cuda.memory.alloc %bytes : i64 -> (!cuda.result, !ptr.ptr)

    %res = cuda.memory.copy %device_mem, %host_mem, %bytes [HostToDevice]

    %res = cuda.launch @module::@kernel<<< (%gridX, %gridY, %gridZ), (%blockX, %blockY, %blockZ), %smem, %stream >>>(%device_mem)

    %res = cuda.memory.copy %host_mem, %device_mem, %bytes [DeviceToHost]

    %res = cuda.memory.free %device_mem : !ptr.ptr -> !cuda.result
    ```
  }];
  let arguments = (ins
    Cuda_PointerType:$src,
    Cuda_PointerType:$dst,
    Cuda_T64BitSignlessInteger:$bytes,
    Cuda_MemcpyKindEnum:$kind
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_MemoryCopyAsyncOp : Cuda_Op<"memory.copy_async"> {
  let description = [{
    Operation to copy memory between host and device asynchronously.

    Examples:
    ```mlir

    %bytes = arith.constant 1024 : i64

    %device_mem, %res = cuda.memory.alloc %bytes : i64 -> (!cuda.result, !ptr.ptr)

    %res = cuda.memory.copy_async %device_mem, %host_mem, %bytes [HostToDevice]

    %res = cuda.launch @module::@kernel<<< (%gridX, %gridY, %gridZ), (%blockX, %blockY, %blockZ), %smem, %stream >>>(%device_mem)

    %res = cuda.memory.copy_async %host_mem, %device_mem, %bytes [DeviceToHost]

    %res = cuda.memory.free %device_mem : !ptr.ptr -> !cuda.result
    ```
  }];
  let arguments = (ins
    Cuda_PointerType:$src,
    Cuda_PointerType:$dst,
    Cuda_T64BitSignlessInteger:$bytes,
    Optional<Cuda_CUDAStream>:$stream,
    Cuda_MemcpyKindEnum:$kind
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_MemoryFreeOp : Cuda_Op<"memory.free"> {
  let description = [{
    Operation to free allocated device memory.

    Examples:
    ```mlir

    %bytes = arith.constant 0 : i64
    %res, %device_mem = cuda.memory.alloc(%bytes) : i64 -> cuda.result, cuda.ptr

    %res0 = cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%device_mem)

    %res1 = cuda.memory.free(%device_mem) : i64 -> cuda.result
    ```
  }];
  let arguments = (ins
    Cuda_PointerType:$ptr
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_MemoryFreeAsyncOp : Cuda_Op<"memory.free_async"> {
  let description = [{
    Operation to free allocated device memory asynchronously.

    Examples:
    ```mlir

    %res, %device_mem = cuda.memory.alloc_async(%bytes, %stream) : i64, cuda.ptr -> cuda.result

    cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%device_mem)

    $res1 = cuda.memory.free_async(%device_mem, %stream) : i64, !cuda.stream -> cuda.result
    ```

    Preserves semantics of [cuMemAllocAsync](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f)
  }];
  let arguments = (ins
    Cuda_PointerType:$ptr,
    Cuda_CUDAStream:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_MemoryFreeHostOp : Cuda_Op<"memory.free_host"> {
  let description = [{
    Operation to free allocated host memory.

    Examples:
    ```mlir

    %bytes = arith.constant 0 : i64
    %host_mem, %res = cuda.memory.alloc_host(%bytes) : (i64) -> (cuda.ptr, cuda.result)

    %res0 = cuda.launch @module::@kernel<<< %grid, %block, %cluster, %smem, %stream >>>(%host_mem)

    %res1 = cuda.memory.free_host(%host_mem)
    ```
  }];
  let arguments = (ins
    Cuda_PointerType:$ptr
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_MemorySetOp : Cuda_Op<"memory.set"> {
  let description = [{
    Operation to fill memory with a uniform value.
  }];
  let arguments = (ins
    Cuda_PointerType:$ptr,
    Cuda_T32BitSignlessInteger:$value,
    Cuda_T64BitSignlessInteger:$bytes
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_MemorySetAsyncOp : Cuda_Op<"memory.set_async"> {
  let description = [{
    Operation to fill memory with a uniform value asynchronously.
  }];
  let arguments = (ins
    Cuda_PointerType:$ptr,
    Cuda_T32BitSignlessInteger:$value,
    Cuda_T64BitSignlessInteger:$bytes,
    Optional<Cuda_CUDAStream>:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_OccupancyMaxActiveBlocks : Cuda_Op<"occupancy_max_active_blocks_per_multiprocessor"> {
  let description = [{
    Returns in `num_blocks` the maximum number of active blocks per streaming multiplrocessor for the device function.

    Example:

    ```mlir
        %one64 = arith.constant 1 : i64
        %one32 = arith.constant 1 : i32
        %res, %num_blocks = cuda.occupancy_max_active_blocks_per_multiprocessor @funcname, %one32, %one64 : (i32, i64) -> (!cuda.result, i32)
    ```
  }];
  let arguments = (ins
    Cuda_CudaIntIsAIntegerType:$block_size,
    Cuda_T64BitSignlessInteger:$dynamic_smem_size,
    SymbolRefAttr:$func
  );
  let results = (outs
    Cuda_CUDAResultType:$res,
    Cuda_T32BitSignlessInteger2:$num_blocks
  );
}
def Cuda_ProfilerStart : Cuda_Op<"profiler.start"> {
  let description = [{
    Starts the CUDA profiler.
  }];
  let arguments = (ins);
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_ProfilerStop : Cuda_Op<"profiler.stop"> {
  let description = [{
    Stops the CUDA profiler.
  }];
  let arguments = (ins);
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_ResultAssertSuccessOp : Cuda_Op<"result.assert_success"> {
  let description = [{
    Asserts that a `cuda.result` indicates success.
  }];
  let arguments = (ins
    Cuda_CUDAResultType2:$cuda_result
  );
  let results = (outs);
}
def Cuda_ResultFailedOp : Cuda_Op<"result.failed"> {
  let description = [{
    Determines if a `cuda.result` indicates failure.
  }];
  let arguments = (ins);
  let results = (outs);
}
def Cuda_ResultPrintOnErrorOp : Cuda_Op<"result.print_on_error"> {
  let description = [{
    Prints a message on error.
  }];
  let arguments = (ins
    Cuda_CUDAResultType2:$cuda_result
  );
  let results = (outs);
}
def Cuda_ResultSucceededOp : Cuda_Op<"result.succeeded"> {
  let description = [{
    Determines if a `cuda.result` indicates success.
  }];
  let arguments = (ins
    Cuda_CUDAResultType2:$cuda_result
  );
  let results = (outs
    Cuda_T1BitSignlessInteger:$success
  );
}
def Cuda_ReturnOp : Cuda_Op<"return", [Terminator]> {
  let description = [{
    Returns a value from a `cuda.kernel` or `cuda.func` region.

    Note, the emitter seems to omit the dialect and simply prints `return` in the
    custom assembly syntax.

    Examples:

    ```mlir

    cuda.module @module_name {

    cuda.func @device_func(%arg0 : i32) -> i32 {
      %0 = arith.constant 0 : i32
      cuda.return %0 : i32
    }

    }
    ```
  }];
  let arguments = (ins
    Variadic<Cuda_AnyType>:$inputs
  );
  let results = (outs);
}
def Cuda_ReturnIfErrorOp : Cuda_Op<"return_if_error"> {
  let description = [{
    The first operand must be the !cuda.result to check.
    The subsequent operands are the values to be returned if an error exists.
    This op is first lowered to an intermediate op `cuda.return_with_error`,
    before final conversion to LLVM in a subsequent pass `cuda-finalize-error-handling`.

    Examples:
    ```mlir
    func.func @host_func() -> i32 {
      %ten = arith.constant 10 : i32
      %cuda_result = cuda.launch @module::@kernel...
      // Create an undefined/poison value for the error return path.
      // Users can also input any arbitrary value for the error return path.
      %poison = llvm.mlir.poison : i32
      // return poison if cuda_result is not cudaSuccess
      %res_i32 = cuda.cast %cuda_result : !cuda.result -> i32
      cuda.return_if_error %res_i32, %poison : i32, i32

      // return 10 if execution reaches here
      return %ten : i32
    }
    ```
  }];
  let arguments = (ins
    Variadic<Cuda_AnyType>:$inputs
  );
  let results = (outs);
}
def Cuda_RuntimeGetDevicePropertiesOp : Cuda_Op<"runtime.device.properties"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$device
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDACudaDevicePropStructTypeValueTyped:$prop
  );
}
def Cuda_RuntimeDevicePropGetAccessPolicyMaxWindowSizeOp : Cuda_Op<"runtime.device_prop.get_access_policy_max_window_size", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetAsyncEngineCountOp : Cuda_Op<"runtime.device_prop.get_async_engine_count", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetCanMapHostMemoryOp : Cuda_Op<"runtime.device_prop.get_can_map_host_memory", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetCanUseHostPointerForRegisteredMemOp : Cuda_Op<"runtime.device_prop.get_can_use_host_pointer_for_registered_mem", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetClusterLaunchOp : Cuda_Op<"runtime.device_prop.get_cluster_launch", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetComputePreemptionSupportedOp : Cuda_Op<"runtime.device_prop.get_compute_preemption_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetConcurrentKernelsOp : Cuda_Op<"runtime.device_prop.get_concurrent_kernels", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetConcurrentManagedAccessOp : Cuda_Op<"runtime.device_prop.get_concurrent_managed_access", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetCooperativeLaunchOp : Cuda_Op<"runtime.device_prop.get_cooperative_launch", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetDeferredMappingCudaArraySupportedOp : Cuda_Op<"runtime.device_prop.get_deferred_mapping_cuda_array_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetDeviceNumaConfigOp : Cuda_Op<"runtime.device_prop.get_device_numa_config", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetDeviceNumaIdOp : Cuda_Op<"runtime.device_prop.get_device_numa_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetDirectManagedMemAccessFromHostOp : Cuda_Op<"runtime.device_prop.get_direct_managed_mem_access_from_host", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetECCEnabledOp : Cuda_Op<"runtime.device_prop.get_ecc_enabled", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetGlobalL1CacheSupportedOp : Cuda_Op<"runtime.device_prop.get_global_l1cache_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetGpuDirectRDMAFlushWritesOptionsOp : Cuda_Op<"runtime.device_prop.get_gpu_direct_rdma_flush_writes_options", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetGpuDirectRDMASupportedOp : Cuda_Op<"runtime.device_prop.get_gpu_direct_rdma_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetGpuDirectRDMAWritesOrderingOp : Cuda_Op<"runtime.device_prop.get_gpu_direct_rdma_writes_ordering", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetGpuPciDeviceIDOp : Cuda_Op<"runtime.device_prop.get_gpu_pci_device_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetGpuPciSubsystemIDOp : Cuda_Op<"runtime.device_prop.get_gpu_pci_subsystem_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetHostNativeAtomicSupportedOp : Cuda_Op<"runtime.device_prop.get_host_native_atomic_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetHostNumaIdOp : Cuda_Op<"runtime.device_prop.get_host_numa_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetHostNumaMultinodeIpcSupportedOp : Cuda_Op<"runtime.device_prop.get_host_numa_multinode_ipc_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetHostRegisterReadOnlySupportedOp : Cuda_Op<"runtime.device_prop.get_host_register_read_only_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetHostRegisterSupportedOp : Cuda_Op<"runtime.device_prop.get_host_register_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetIntegratedOp : Cuda_Op<"runtime.device_prop.get_integrated", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetIpcEventSupportedOp : Cuda_Op<"runtime.device_prop.get_ipc_event_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetIsMultiGpuBoardOp : Cuda_Op<"runtime.device_prop.get_is_multi_gpu_board", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetL2CacheSizeOp : Cuda_Op<"runtime.device_prop.get_l_2cache_size", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetLocalL1CacheSupportedOp : Cuda_Op<"runtime.device_prop.get_local_l1cache_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetLuidOp : Cuda_Op<"runtime.device_prop.get_luid", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf8Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetLuidDeviceNodeMaskOp : Cuda_Op<"runtime.device_prop.get_luid_device_node_mask", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMajorOp : Cuda_Op<"runtime.device_prop.get_major", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetManagedMemoryOp : Cuda_Op<"runtime.device_prop.get_managed_memory", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxBlocksPerMultiProcessorOp : Cuda_Op<"runtime.device_prop.get_max_blocks_per_multi_processor", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxGridSizeOp : Cuda_Op<"runtime.device_prop.get_max_grid_size", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurface1DOp : Cuda_Op<"runtime.device_prop.get_max_surface_1d", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurface1DLayeredOp : Cuda_Op<"runtime.device_prop.get_max_surface_1d_layered", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurface2DOp : Cuda_Op<"runtime.device_prop.get_max_surface_2d", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurface2DLayeredOp : Cuda_Op<"runtime.device_prop.get_max_surface_2d_layered", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurface3DOp : Cuda_Op<"runtime.device_prop.get_max_surface_3d", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurfaceCubemapOp : Cuda_Op<"runtime.device_prop.get_max_surface_cubemap", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxSurfaceCubemapLayeredOp : Cuda_Op<"runtime.device_prop.get_max_surface_cubemap_layered", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture1DOp : Cuda_Op<"runtime.device_prop.get_max_texture_1d", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture1DLayeredOp : Cuda_Op<"runtime.device_prop.get_max_texture_1d_layered", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture1DMipmapOp : Cuda_Op<"runtime.device_prop.get_max_texture_1d_mipmap", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture2DOp : Cuda_Op<"runtime.device_prop.get_max_texture_2d", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture2DGatherOp : Cuda_Op<"runtime.device_prop.get_max_texture_2d_gather", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture2DLayeredOp : Cuda_Op<"runtime.device_prop.get_max_texture_2d_layered", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture2DLinearOp : Cuda_Op<"runtime.device_prop.get_max_texture_2d_linear", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture2DMipmapOp : Cuda_Op<"runtime.device_prop.get_max_texture_2d_mipmap", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture3DOp : Cuda_Op<"runtime.device_prop.get_max_texture_3d", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTexture3DAltOp : Cuda_Op<"runtime.device_prop.get_max_texture_3d_alt", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTextureCubemapOp : Cuda_Op<"runtime.device_prop.get_max_texture_cubemap", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxTextureCubemapLayeredOp : Cuda_Op<"runtime.device_prop.get_max_texture_cubemap_layered", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxThreadsDimOp : Cuda_Op<"runtime.device_prop.get_max_threads_dim", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf32Bit:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxThreadsPerBlockOp : Cuda_Op<"runtime.device_prop.get_max_threads_per_block", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMaxThreadsPerMultiProcessorOp : Cuda_Op<"runtime.device_prop.get_max_threads_per_multi_processor", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMemPitchOp : Cuda_Op<"runtime.device_prop.get_mem_pitch", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMemoryBusWidthOp : Cuda_Op<"runtime.device_prop.get_memory_bus_width", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMemoryPoolSupportedHandleTypesOp : Cuda_Op<"runtime.device_prop.get_memory_pool_supported_handle_types", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMemoryPoolsSupportedOp : Cuda_Op<"runtime.device_prop.get_memory_pools_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMinorOp : Cuda_Op<"runtime.device_prop.get_minor", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMpsEnabledOp : Cuda_Op<"runtime.device_prop.get_mps_enabled", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMultiGpuBoardGroupIDOp : Cuda_Op<"runtime.device_prop.get_multi_gpu_board_group_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetMultiProcessorCountOp : Cuda_Op<"runtime.device_prop.get_multi_processor_count", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetNameOp : Cuda_Op<"runtime.device_prop.get_name", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf8Bit2:$result
  );
}
def Cuda_RuntimeDevicePropGetPageableMemoryAccessOp : Cuda_Op<"runtime.device_prop.get_pageable_memory_access", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetPageableMemoryAccessUsesHostPageTablesOp : Cuda_Op<"runtime.device_prop.get_pageable_memory_access_uses_host_page_tables", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetPciBusIDOp : Cuda_Op<"runtime.device_prop.get_pci_bus_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetPciDeviceIDOp : Cuda_Op<"runtime.device_prop.get_pci_device_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetPciDomainIDOp : Cuda_Op<"runtime.device_prop.get_pci_domain_id", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetPersistingL2CacheMaxSizeOp : Cuda_Op<"runtime.device_prop.get_persisting_l2cache_max_size", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetRegsPerBlockOp : Cuda_Op<"runtime.device_prop.get_regs_per_block", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetRegsPerMultiprocessorOp : Cuda_Op<"runtime.device_prop.get_regs_per_multiprocessor", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetReservedSharedMemPerBlockOp : Cuda_Op<"runtime.device_prop.get_reserved_shared_mem_per_block", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetSharedMemPerBlockOp : Cuda_Op<"runtime.device_prop.get_shared_mem_per_block", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetSharedMemPerBlockOptinOp : Cuda_Op<"runtime.device_prop.get_shared_mem_per_block_optin", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetSharedMemPerMultiprocessorOp : Cuda_Op<"runtime.device_prop.get_shared_mem_per_multiprocessor", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetSparseCudaArraySupportedOp : Cuda_Op<"runtime.device_prop.get_sparse_cuda_array_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetStreamPrioritiesSupportedOp : Cuda_Op<"runtime.device_prop.get_stream_priorities_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetSurfaceAlignmentOp : Cuda_Op<"runtime.device_prop.get_surface_alignment", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetTccDriverOp : Cuda_Op<"runtime.device_prop.get_tcc_driver", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetTextureAlignmentOp : Cuda_Op<"runtime.device_prop.get_texture_alignment", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetTexturePitchAlignmentOp : Cuda_Op<"runtime.device_prop.get_texture_pitch_alignment", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetTimelineSemaphoreInteropSupportedOp : Cuda_Op<"runtime.device_prop.get_timeline_semaphore_interop_supported", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetTotalConstMemOp : Cuda_Op<"runtime.device_prop.get_total_const_mem", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetTotalGlobalMemOp : Cuda_Op<"runtime.device_prop.get_total_global_mem", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetUnifiedAddressingOp : Cuda_Op<"runtime.device_prop.get_unified_addressing", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetUnifiedFunctionPointersOp : Cuda_Op<"runtime.device_prop.get_unified_function_pointers", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDevicePropGetUuidOp : Cuda_Op<"runtime.device_prop.get_uuid", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_CUDACudaUUIDTStructTypeValue:$result
  );
}
def Cuda_RuntimeDevicePropGetWarpSizeOp : Cuda_Op<"runtime.device_prop.get_warp_size", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaDevicePropStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDim3GetXOp : Cuda_Op<"runtime.dim_3.get_x", [Pure]> {
  let arguments = (ins
    Cuda_CUDADim3StructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDim3GetYOp : Cuda_Op<"runtime.dim_3.get_y", [Pure]> {
  let arguments = (ins
    Cuda_CUDADim3StructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeDim3GetZOp : Cuda_Op<"runtime.dim_3.get_z", [Pure]> {
  let arguments = (ins
    Cuda_CUDADim3StructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeEventCreateWithFlagsOp : Cuda_Op<"runtime.event.create_with_flags"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$flags
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_EventRepresentsCUDAEvent:$event
  );
}
def Cuda_RuntimeEventQueryOp : Cuda_Op<"runtime.event.query", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$event
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimeEventRecordWithFlagsOp : Cuda_Op<"runtime.event.record_with_flags"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$event,
    Cuda_CUDAStream:$stream,
    Cuda_T32BitSignlessInteger:$flags
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimeEventSynchronizeOp : Cuda_Op<"runtime.event.synchronize"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_EventRepresentsCUDAEvent2:$event
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimeExtentGetDepthOp : Cuda_Op<"runtime.extent.get_depth", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaExtentStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeExtentGetHeightOp : Cuda_Op<"runtime.extent.get_height", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaExtentStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeExtentGetWidthOp : Cuda_Op<"runtime.extent.get_width", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaExtentStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeGraphAddKernelNodeOp : Cuda_Op<"runtime.graph.add_kernel_node"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDAGraph:$graph,
    Cuda_PointerType:$pDependencies,
    Cuda_T64BitSignlessInteger:$numDependencies,
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$pNodeParams
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDAGraphNodeHandle:$pGraphNode
  );
}
def Cuda_RuntimeGraphCreateOp : Cuda_Op<"runtime.graph.create"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$flags
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDAGraph2:$pGraph
  );
}
def Cuda_RuntimeGraphKernelNodeGetAttributeOp : Cuda_Op<"runtime.graph.kernel_node_get_attribute", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDAGraphNodeHandle2:$hNode,
    Cuda_T32BitSignlessInteger:$attr
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_VariantCooperativeOfCUDACudaLaunchAttributeValueUnion:$value_out
  );
}
def Cuda_RuntimeGraphKernelNodeSetAttributeOp : Cuda_Op<"runtime.graph.kernel_node_set_attribute"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDAGraphNodeHandle2:$hNode,
    Cuda_T32BitSignlessInteger:$attr,
    Cuda_VariantCooperativeOfCUDACudaLaunchAttributeValueUnion:$value
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimeKernelNodeParamsGetBlockDimOp : Cuda_Op<"runtime.kernel_node_params.get_block_dim", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDADim3StructTypeValueTyped2:$result
  );
}
def Cuda_RuntimeKernelNodeParamsGetExtraOp : Cuda_Op<"runtime.kernel_node_params.get_extra", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_LLVMPointerType:$result
  );
}
def Cuda_RuntimeKernelNodeParamsGetFuncOp : Cuda_Op<"runtime.kernel_node_params.get_func", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_LLVMPointerType:$result
  );
}
def Cuda_RuntimeKernelNodeParamsGetGridDimOp : Cuda_Op<"runtime.kernel_node_params.get_grid_dim", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDADim3StructTypeValueTyped2:$result
  );
}
def Cuda_RuntimeKernelNodeParamsGetKernelParamsOp : Cuda_Op<"runtime.kernel_node_params.get_kernel_params", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_LLVMPointerType:$result
  );
}
def Cuda_RuntimeKernelNodeParamsGetSharedMemBytesOp : Cuda_Op<"runtime.kernel_node_params.get_shared_mem_bytes", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaKernelNodeParamsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeLibraryGetGlobalOp : Cuda_Op<"runtime.library.get_global", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$name_arg
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$dptr,
    Cuda_T64BitSignlessInteger:$bytes
  );
}
def Cuda_RuntimeLibraryGetKernelOp : Cuda_Op<"runtime.library.get_kernel", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$name_arg
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDAKernelHandle:$pKernel
  );
}
def Cuda_RuntimeLibraryGetKernelCountOp : Cuda_Op<"runtime.library.get_kernel_count", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$lib
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_T32BitSignlessInteger2:$count
  );
}
def Cuda_RuntimeLibraryGetManagedOp : Cuda_Op<"runtime.library.get_managed", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$name_arg
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$dptr,
    Cuda_T64BitSignlessInteger:$bytes
  );
}
def Cuda_RuntimeLibraryGetUnifiedFunctionOp : Cuda_Op<"runtime.library.get_unified_function", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library,
    Cuda_PointerType:$symbol
  );
  let results = (outs
    Cuda_CUDAResultType2:$cuda_result,
    Cuda_PointerType:$fptr
  );
}
def Cuda_RuntimeLibraryLoadDataOp : Cuda_Op<"runtime.library.load_data"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_PointerType:$code_arg,
    Cuda_PointerType:$jitOptions,
    Cuda_PointerType:$jitOptionsValues,
    Cuda_T32BitSignlessInteger:$numJitOptions,
    Cuda_PointerType:$libraryOptions,
    Cuda_PointerType:$libraryOptionValues,
    Cuda_T32BitSignlessInteger:$numLibraryOptions
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDALibraryHandle2:$library
  );
}
def Cuda_RuntimeLibraryLoadFromFileOp : Cuda_Op<"runtime.library.load_from_file"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_PointerType:$fileName,
    Cuda_PointerType:$jitOptions,
    Cuda_PointerType:$jitOptionsValues,
    Cuda_T32BitSignlessInteger:$numJitOptions,
    Cuda_PointerType:$libraryOptions,
    Cuda_PointerType:$libraryOptionValues,
    Cuda_T32BitSignlessInteger:$numLibraryOptions
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDALibraryHandle2:$library
  );
}
def Cuda_RuntimeLibraryUnloadOp : Cuda_Op<"runtime.library.unload"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDALibraryHandle:$library
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetDstArrayOp : Cuda_Op<"runtime.memcpy_3d_params.get_dst_array", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_PointerType:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetDstPosOp : Cuda_Op<"runtime.memcpy_3d_params.get_dst_pos", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDACudaPosStructTypeValueTyped:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetDstPtrOp : Cuda_Op<"runtime.memcpy_3d_params.get_dst_ptr", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDACudaPitchedPtrStructTypeValueTyped:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetExtentOp : Cuda_Op<"runtime.memcpy_3d_params.get_extent", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDACudaExtentStructTypeValueTyped2:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetKindOp : Cuda_Op<"runtime.memcpy_3d_params.get_kind", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_T32BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetSrcArrayOp : Cuda_Op<"runtime.memcpy_3d_params.get_src_array", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_PointerType:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetSrcPosOp : Cuda_Op<"runtime.memcpy_3d_params.get_src_pos", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDACudaPosStructTypeValueTyped:$result
  );
}
def Cuda_RuntimeMemcpy3DParamsGetSrcPtrOp : Cuda_Op<"runtime.memcpy_3d_params.get_src_ptr", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$source
  );
  let results = (outs
    Cuda_CUDACudaPitchedPtrStructTypeValueTyped:$result
  );
}
def Cuda_RuntimeMemcpy3DOp : Cuda_Op<"runtime.memory.cpy3d"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$p
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimeMemcpy3DAsyncOp : Cuda_Op<"runtime.memory.cpy3d_async"> {
  let description = [{
    Base class for all CUDA driver call builder operations. It provides a common
    interface for building CUDA driver calls, including the ability to specify
    the API to be called, the arguments, and the expected results.
    This class is not intended to be used directly, but rather as a base class
    for specific CUDA driver call builder operations.
  }];
  let arguments = (ins
    Cuda_CUDACudaMemcpy3DParmsStructTypeValueTyped:$p,
    Cuda_CUDAStream:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result
  );
}
def Cuda_RuntimePitchedPtrGetPitchOp : Cuda_Op<"runtime.pitched_ptr.get_pitch", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPitchedPtrStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimePitchedPtrGetPtrOp : Cuda_Op<"runtime.pitched_ptr.get_ptr", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPitchedPtrStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_LLVMPointerType:$result
  );
}
def Cuda_RuntimePitchedPtrGetXsizeOp : Cuda_Op<"runtime.pitched_ptr.get_xsize", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPitchedPtrStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimePitchedPtrGetYsizeOp : Cuda_Op<"runtime.pitched_ptr.get_ysize", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPitchedPtrStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimePosGetXOp : Cuda_Op<"runtime.pos.get_x", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPosStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimePosGetYOp : Cuda_Op<"runtime.pos.get_y", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPosStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimePosGetZOp : Cuda_Op<"runtime.pos.get_z", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaPosStructTypeValueTyped2:$source
  );
  let results = (outs
    Cuda_T64BitSignlessInteger2:$result
  );
}
def Cuda_RuntimeUuidGetBytesOp : Cuda_Op<"runtime.uuid.get_bytes", [Pure]> {
  let arguments = (ins
    Cuda_CUDACudaUUIDTStructTypeValue2:$source
  );
  let results = (outs
    Cuda_FixedLengthVectorOf8Bit3:$result
  );
}
def Cuda_StreamAddCallbackOp : Cuda_Op<"stream.add_callback"> {
  let description = [{
    Enqueues a host-side callback on the given CUDA stream. The callback is provided
    as a symbol to a `func.func` with ABI:
      `func @cb(!ptr.ptr<#ptr.generic_space>, i32, !ptr.ptr<#ptr.generic_space>) -> ()`
    i.e., (stream, status, userData) -> void.
  }];
  let arguments = (ins
    Cuda_CUDAStream:$stream,
    Cuda_PointerType:$user_data,
    FlatSymbolRefAttr:$callback
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_StreamBeginCaptureOp : Cuda_Op<"stream.begin_capture"> {
  let description = [{
    Operation to begin capturing a CUDA stream.
  }];
  let arguments = (ins
    Cuda_CUDAStream:$stream,
    Cuda_StreamCaptureModeEnumEnum:$mode
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_StreamCreateOp : Cuda_Op<"stream.create"> {
  let description = [{
    Operation to create a CUDA stream.
  }];
  let arguments = (ins);
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDAStream2:$stream
  );
}
def Cuda_StreamDestroyOp : Cuda_Op<"stream.destroy"> {
  let description = [{
    Operation to destroy a CUDA stream.
  }];
  let arguments = (ins
    Cuda_CUDAStream:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_StreamEndCaptureOp : Cuda_Op<"stream.end_capture"> {
  let description = [{
    Operation to end capturing a CUDA stream.
  }];
  let arguments = (ins
    Cuda_CUDAStream:$stream
  );
  let results = (outs
    Cuda_CUDAResultType:$result,
    Cuda_CUDAGraph2:$graph
  );
}
def Cuda_StreamSynchronizeOp : Cuda_Op<"stream.synchronize", [MemoryEffects<[MemRead]>]> {
  let description = [{
    Operation to synchronize with a CUDA Stream.
  }];
  let arguments = (ins
    Cuda_CUDAStream:$stream,
    Optional<Cuda_EventRepresentsCUDAEvent2>:$event
  );
  let results = (outs
    Cuda_CUDAResultType:$result
  );
}
def Cuda_TensorMapEncodeTiledOp : Cuda_Op<"tensor_map_encode_tiled"> {
  let description = [{
    Creates a descriptor for Tensor Memory Access (TMA) object specified by the parameters
    describing a tiled region and returns it in tensorMap.

    This operation corresponds to the CUDA Driver API function:
    `cuTensorMapEncodeTiled`

    Tensor map objects are only supported on devices of compute capability 9.0 or higher.
    The tensorMap address must be aligned to 64 bytes.

    All parameters are provided as runtime SSA values.
  }];
  let arguments = (ins
    Cuda_T32BitSignlessInteger:$tensorDataType,
    Cuda_T32BitSignlessInteger:$tensorRank,
    Cuda_PointerType:$globalAddress,
    Cuda_LLVMPointerInAddressSpace0:$globalDims,
    Cuda_LLVMPointerInAddressSpace0:$globalStrides,
    Cuda_LLVMPointerInAddressSpace0:$boxDims,
    Cuda_LLVMPointerInAddressSpace0:$elementStrides,
    Cuda_T32BitSignlessInteger:$interleave,
    Cuda_T32BitSignlessInteger:$swizzle,
    Cuda_T32BitSignlessInteger:$l2Promotion,
    Cuda_T32BitSignlessInteger:$oobFill
  );
  let results = (outs
    Cuda_CUDAResultType:$cuda_result,
    Cuda_CUDATensorMapType:$cuTensorMap
  );
}

#endif // CUTE_IR_GENERATED_CUDA_OPS_TD
