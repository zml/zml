//===- capi.h - C API for the Furiosa TCL dialect -----------------*- C -*-===//

#ifndef MLIR_DIALECTS_TCL_CAPI_H
#define MLIR_DIALECTS_TCL_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Tcl, tcl);

// Constructors verify their parameters and return a null handle when invalid.
MLIR_CAPI_EXPORTED MlirAttribute mlirTclSymbolAttrGet(MlirContext context,
                                                      MlirStringRef name);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclExprAttrGet(MlirContext context,
                                                    MlirStringRef kind,
                                                    MlirAttribute args);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclTacticAttrGet(MlirContext context,
                                                      MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclVeOpcodeAttrGet(MlirContext context,
                                                        MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclReduceModeAttrGet(MlirContext context,
                                                          MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclPredicateAttrGet(MlirContext context,
                                                         MlirStringRef value);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclContextAttrGet(MlirContext context,
                                                       MlirAttribute fields);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTclReadOptionsAttrGet(MlirContext context, MlirAttribute fields);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclConfigAttrGet(MlirContext context,
                                                      MlirAttribute fields);
MLIR_CAPI_EXPORTED MlirAttribute
mlirTclDramMappingAttrGet(MlirContext context, MlirAttribute chip,
                          MlirAttribute inner, MlirAttribute original);

MLIR_CAPI_EXPORTED MlirType mlirTclLogicalTypeGet(MlirContext context,
                                                  MlirType element,
                                                  MlirAttribute axes);
MLIR_CAPI_EXPORTED bool mlirTypeIsATclLogical(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirTclLogicalTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirTclLogicalTypeGetAxes(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirTclMappedTypeGet(MlirContext context,
                                                 MlirType element,
                                                 MlirAttribute mapping);

#ifdef __cplusplus
}
#endif

#endif // MLIR_DIALECTS_TCL_CAPI_H
