//===- NVVMDialect.h - NVIDIA's nvvm dialect of the CuTe DSL ------*- C++ -*-===//

#ifndef MLIR_DIALECTS_NVVM_NVVMDIALECT_H
#define MLIR_DIALECTS_NVVM_NVVMDIALECT_H

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

#include "mlir/dialects/nvvm/NVVMDialect.h.inc"

#include "mlir/dialects/nvvm/NVVMEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/dialects/nvvm/NVVMAttrs.h.inc"

#define GET_OP_CLASSES
#include "mlir/dialects/nvvm/NVVMOps.h.inc"

#endif // MLIR_DIALECTS_NVVM_NVVMDIALECT_H
