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

#include "mlir/dialects/cuda/CudaDialect.h.inc"

#include "mlir/dialects/cuda/CudaEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/cuda/CudaAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/dialects/cuda/CudaTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/dialects/cuda/CudaOps.h.inc"

#endif // CUTE_IR_DIALECT_CUDA_IR_CUDA_DIALECT_H
