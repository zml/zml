//===- NVVMDialect.h - NVIDIA's nvvm dialect of the CuTe DSL ------*- C++ -*-===//

#ifndef CUTE_IR_DIALECT_NVVM_IR_NVVM_DIALECT_H
#define CUTE_IR_DIALECT_NVVM_IR_NVVM_DIALECT_H

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

#include "cute_ir/Dialect/NVVM/IR/NVVMDialect.h.inc"

#include "cute_ir/Dialect/NVVM/IR/NVVMEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cute_ir/Dialect/NVVM/IR/NVVMAttrs.h.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/NVVM/IR/NVVMOps.h.inc"

#endif // CUTE_IR_DIALECT_NVVM_IR_NVVM_DIALECT_H
