#ifndef ZML_MLIR_DIALECTS_TCL_TCL_CAPI_H
#define ZML_MLIR_DIALECTS_TCL_TCL_CAPI_H

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Tcl, tcl);

// Constructors verify their parameters and return a null handle when invalid.
MLIR_CAPI_EXPORTED MlirAttribute mlirTclSymbolAttrGet(MlirContext ctx,
                                                      MlirStringRef name);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclExprAttrGet(MlirContext ctx,
                                                    MlirStringRef kind,
                                                    MlirAttribute args);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclTacticAttrGet(MlirContext ctx,
                                                      MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclVeOpcodeAttrGet(MlirContext ctx,
                                                        MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclReduceModeAttrGet(MlirContext ctx,
                                                          MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclPredicateAttrGet(MlirContext ctx,
                                                         MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclContextAttrGet(MlirContext ctx,
                                                       MlirAttribute fields);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTclReadOptionsAttrGet(MlirContext ctx, MlirAttribute fields);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclConfigAttrGet(MlirContext ctx,
                                                      MlirAttribute fields);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTclDramMappingAttrGet(MlirContext ctx, MlirAttribute chip,
                          MlirAttribute inner, MlirAttribute original);

MLIR_CAPI_EXPORTED MlirType mlirTclLogicalTypeGet(MlirContext ctx,
                                                  MlirType element,
                                                  MlirAttribute axes);
MLIR_CAPI_EXPORTED bool mlirTypeIsATclLogical(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTclLogicalTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclLogicalTypeGetAxes(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTclMappedTypeGet(MlirContext ctx,
                                                 MlirType element,
                                                 MlirAttribute mapping);

#ifdef __cplusplus
}
#endif

#endif  // ZML_MLIR_DIALECTS_TCL_TCL_CAPI_H
