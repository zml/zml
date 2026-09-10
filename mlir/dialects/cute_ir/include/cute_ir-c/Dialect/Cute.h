//===- Cute.h - C API for the CuTe dialect ----------------------*- C -*-===//

#ifndef CUTE_IR_C_DIALECT_CUTE_H
#define CUTE_IR_C_DIALECT_CUTE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cute, cute);

/// Parses a CuTe type from its complete assembly spelling.
///
/// This is intentionally assembly based: CuTe's algebra types contain recursive
/// tuples, dynamic leaves, ratios, and basis strides. The dialect parser is the
/// canonical validator for that structure and avoids duplicating it in the C ABI.
MLIR_CAPI_EXPORTED MlirType mlirCuteTypeParse(MlirContext context,
                                              MlirStringRef assembly);

/// Returns true when `type` belongs to the CuTe dialect.
MLIR_CAPI_EXPORTED bool mlirTypeIsACuteType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // CUTE_IR_C_DIALECT_CUTE_H
