//===- AIROps.h - Metal AIR operations --------------------------*- C++ -*-===//

#ifndef MLIR_DIALECTS_AIR_AIROPS_H
#define MLIR_DIALECTS_AIR_AIROPS_H

#include "mlir/dialects/air/AIRAttributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#define GET_OP_CLASSES
#include "mlir/dialects/air/AIROps.h.inc"

#endif // MLIR_DIALECTS_AIR_AIROPS_H
