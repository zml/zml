//===- CuteTypes.cpp - combined OSS and recovered CuTe types -------------===//

#include "cute_ir/Dialect/Cute/IR/CuteDialectPrivate.h"

// Compile the unmodified OSS implementation against the combined generated
// TypeDef definitions. This gives the one CuteDialect type parser visibility
// of both the OSS and recovered type mnemonics.
#include "cute_ir/lib/Dialect/IR/CuteTypes.cpp"
