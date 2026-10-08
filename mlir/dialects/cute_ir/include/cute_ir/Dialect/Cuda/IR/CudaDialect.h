//===- CudaDialect.h - NVIDIA's cuda dialect of the CuTe DSL ------*- C++ -*-===//

#ifndef CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_H
#define CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cute_ir/Dialect/Cuda/IR/CudaDialect.h.inc"

#include "cute_ir/Dialect/Cuda/IR/CudaEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cute_ir/Dialect/Cuda/IR/CudaAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/Cuda/IR/CudaTypes.h.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/Cuda/IR/CudaOps.h.inc"

#endif // CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_H
