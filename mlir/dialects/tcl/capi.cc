//===- capi.cc - C API for the Furiosa TCL dialect ------------------------===//

#include "capi.h"

#include "mlir/dialects/tcl/tcl_ops.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

#include <utility>

using namespace mlir;
using namespace xla::furiosa::tcl;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tcl, tcl, TclDialect)

// `getChecked` with its diagnostic silenced: an invalid parameter is a null
// handle, which the caller reports.
template <typename T, typename... Args>
static auto checked(MlirContext context, Args &&...args) {
  MLIRContext *ctx = unwrap(context);
  if (!ctx)
    return T();
  ctx->getOrLoadDialect<TclDialect>();
  ScopedDiagnosticHandler silence(ctx, [](Diagnostic &) { return success(); });
  return T::getChecked([ctx] { return emitError(UnknownLoc::get(ctx)); }, ctx,
                       std::forward<Args>(args)...);
}

template <typename T> static T as(MlirAttribute attr) {
  return dyn_cast_if_present<T>(unwrap(attr));
}

MlirAttribute mlirTclSymbolAttrGet(MlirContext context, MlirStringRef name) {
  return wrap(checked<SymbolAttr>(context, unwrap(name)));
}

MlirAttribute mlirTclTacticAttrGet(MlirContext context, MlirStringRef value) {
  return wrap(checked<TacticAttr>(context, unwrap(value)));
}

MlirAttribute mlirTclVeOpcodeAttrGet(MlirContext context, MlirStringRef value) {
  return wrap(checked<VeOpcodeAttr>(context, unwrap(value)));
}

MlirAttribute mlirTclReduceModeAttrGet(MlirContext context,
                                       MlirStringRef value) {
  return wrap(checked<ReduceModeAttr>(context, unwrap(value)));
}

MlirAttribute mlirTclPredicateAttrGet(MlirContext context,
                                      MlirStringRef value) {
  return wrap(checked<PredicateAttr>(context, unwrap(value)));
}

MlirAttribute mlirTclContextAttrGet(MlirContext context, MlirAttribute fields) {
  auto dict = as<DictionaryAttr>(fields);
  return dict ? wrap(checked<ContextAttr>(context, dict))
              : MlirAttribute{nullptr};
}

MlirAttribute mlirTclReadOptionsAttrGet(MlirContext context,
                                        MlirAttribute fields) {
  auto dict = as<DictionaryAttr>(fields);
  return dict ? wrap(checked<ReadOptionsAttr>(context, dict))
              : MlirAttribute{nullptr};
}

MlirAttribute mlirTclConfigAttrGet(MlirContext context, MlirAttribute fields) {
  auto dict = as<DictionaryAttr>(fields);
  return dict ? wrap(checked<ConfigAttr>(context, dict))
              : MlirAttribute{nullptr};
}

MlirAttribute mlirTclExprAttrGet(MlirContext context, MlirStringRef kind,
                                 MlirAttribute args) {
  auto array = as<ArrayAttr>(args);
  return array ? wrap(checked<ExprAttr>(context, unwrap(kind), array))
               : MlirAttribute{nullptr};
}

MlirAttribute mlirTclDramMappingAttrGet(MlirContext context, MlirAttribute chip,
                                        MlirAttribute inner,
                                        MlirAttribute original) {
  auto c = as<ArrayAttr>(chip), i = as<ArrayAttr>(inner),
       o = as<ArrayAttr>(original);
  return c && i && o ? wrap(checked<DramMappingAttr>(context, c, i, o))
                     : MlirAttribute{nullptr};
}

MlirType mlirTclLogicalTypeGet(MlirContext context, MlirType element,
                               MlirAttribute axes) {
  auto array = as<ArrayAttr>(axes);
  if (!array || mlirTypeIsNull(element))
    return {nullptr};
  return wrap(checked<LogicalType>(context, unwrap(element), array));
}

bool mlirTypeIsATclLogical(MlirType type) {
  return isa_and_nonnull<LogicalType>(unwrap(type));
}

MlirType mlirTclLogicalTypeGetElementType(MlirType type) {
  return wrap(cast<LogicalType>(unwrap(type)).getElementType());
}

MlirAttribute mlirTclLogicalTypeGetAxes(MlirType type) {
  return wrap(cast<LogicalType>(unwrap(type)).getAxes());
}

MlirType mlirTclMappedTypeGet(MlirContext context, MlirType element,
                              MlirAttribute mapping) {
  auto m = as<DramMappingAttr>(mapping);
  if (!m || mlirTypeIsNull(element))
    return {nullptr};
  return wrap(checked<MappedType>(context, unwrap(element), m));
}
