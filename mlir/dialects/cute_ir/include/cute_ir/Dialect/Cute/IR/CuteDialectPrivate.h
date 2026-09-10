//===- CuteDialectPrivate.h - recovered CuTe declarations -----*- C++ -*-===//

#ifndef CUTE_IR_DIALECT_CUTE_IR_CUTE_DIALECT_PRIVATE_H
#define CUTE_IR_DIALECT_CUTE_IR_CUTE_DIALECT_PRIVATE_H

// The OSS header owns CuteDialect and the public CuTe declarations.  The
// generated declarations below extend that same C++ dialect class; they do not
// define a second MLIR dialect.
#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"

#define GET_TYPEDEF_CLASSES
#include "cute_ir/Dialect/Cute/IR/CuteTypesPrivate.h.inc"

#define GET_OP_CLASSES
#include "cute_ir/Dialect/Cute/IR/CuteOpsPrivate.h.inc"

#endif // CUTE_IR_DIALECT_CUTE_IR_CUTE_DIALECT_PRIVATE_H
