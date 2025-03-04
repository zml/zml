#include "triton_capi.h"

#include "include/triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Triton, triton, mlir::triton::TritonDialect)

MlirAttribute tritonProgramDimGet(MlirContext ctx, int32_t axis)
{
    return wrap(mlir::triton::TT_ProgramDim::get(unwrap(ctx), axis));
}

MlirAttribute tritonCacheModifierGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_CacheModifierAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonMemSemanticGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_MemSemanticAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonEvictionPolicyGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_EvictionPolicyAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonPaddingOptionGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_PaddingOptionAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonAtomicRMWGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_AtomicRMWAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonMemSyncScopeGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_MemSyncScopeAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonRoundingModeGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_RoundingModeAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonPropagateNanGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_PropagateNanAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonInputPrecisionGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_InputPrecisionAttr::get(unwrap(ctx), kind));
}

MlirAttribute tritonScaleDotElemTypeGet(MlirContext ctx, int32_t kind)
{
    return wrap(mlir::triton::TT_ScaleDotElemTypeAttr::get(unwrap(ctx), kind));
}
