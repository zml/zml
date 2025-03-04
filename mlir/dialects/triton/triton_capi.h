#ifndef TRITON_INTEGRATIONS_C_TRITON_DIALECT_H
#define TRITON_INTEGRATIONS_C_TRITON_DIALECT_H

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"

#ifdef __cplusplus
extern "C"
{
#endif

    MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Triton, triton);

#ifdef __cplusplus
}
#endif

#endif // TRITON_INTEGRATIONS_C_TRITON_DIALECT_H
