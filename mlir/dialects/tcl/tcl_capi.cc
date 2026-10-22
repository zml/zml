#include "tcl_capi.h"

#include <utility>

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/dialects/tcl/tcl_ops.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tcl, tcl, xla::furiosa::tcl::TclDialect)

using namespace xla::furiosa::tcl;  // NOLINT

namespace {

// `getChecked` with its diagnostic silenced: an invalid parameter is a null
// handle, which the caller reports.
template <typename T, typename... Args>
auto checked(MlirContext ctx, Args &&...args) {
  mlir::MLIRContext *context = unwrap(ctx);
  context->getOrLoadDialect<TclDialect>();
  mlir::ScopedDiagnosticHandler silence(
      context, [](mlir::Diagnostic &) { return mlir::success(); });
  return T::getChecked(
      [context] { return mlir::emitError(mlir::UnknownLoc::get(context)); },
      context, std::forward<Args>(args)...);
}

template <typename T>
T as(MlirAttribute attr) {
  return mlir::dyn_cast_if_present<T>(unwrap(attr));
}

}  // namespace

extern "C" {

MlirAttribute mlirTclSymbolAttrGet(MlirContext ctx, MlirStringRef name) {
  return wrap(checked<SymbolAttr>(ctx, unwrap(name)));
}

MlirAttribute mlirTclTacticAttrGet(MlirContext ctx, MlirStringRef value) {
  return wrap(checked<TacticAttr>(ctx, unwrap(value)));
}

MlirAttribute mlirTclVeOpcodeAttrGet(MlirContext ctx, MlirStringRef value) {
  return wrap(checked<VeOpcodeAttr>(ctx, unwrap(value)));
}

MlirAttribute mlirTclReduceModeAttrGet(MlirContext ctx, MlirStringRef value) {
  return wrap(checked<ReduceModeAttr>(ctx, unwrap(value)));
}

MlirAttribute mlirTclPredicateAttrGet(MlirContext ctx, MlirStringRef value) {
  return wrap(checked<PredicateAttr>(ctx, unwrap(value)));
}

MlirAttribute mlirTclContextAttrGet(MlirContext ctx, MlirAttribute fields) {
  auto dict = as<mlir::DictionaryAttr>(fields);
  return dict ? wrap(checked<ContextAttr>(ctx, dict)) : MlirAttribute{nullptr};
}

MlirAttribute mlirTclReadOptionsAttrGet(MlirContext ctx, MlirAttribute fields) {
  auto dict = as<mlir::DictionaryAttr>(fields);
  return dict ? wrap(checked<ReadOptionsAttr>(ctx, dict))
              : MlirAttribute{nullptr};
}

MlirAttribute mlirTclConfigAttrGet(MlirContext ctx, MlirAttribute fields) {
  auto dict = as<mlir::DictionaryAttr>(fields);
  return dict ? wrap(checked<ConfigAttr>(ctx, dict)) : MlirAttribute{nullptr};
}

MlirAttribute mlirTclExprAttrGet(MlirContext ctx, MlirStringRef kind,
                                 MlirAttribute args) {
  auto array = as<mlir::ArrayAttr>(args);
  return array ? wrap(checked<ExprAttr>(ctx, unwrap(kind), array))
               : MlirAttribute{nullptr};
}

MlirAttribute mlirTclDramMappingAttrGet(MlirContext ctx, MlirAttribute chip,
                                        MlirAttribute inner,
                                        MlirAttribute original) {
  auto c = as<mlir::ArrayAttr>(chip), i = as<mlir::ArrayAttr>(inner),
       o = as<mlir::ArrayAttr>(original);
  return c && i && o ? wrap(checked<DramMappingAttr>(ctx, c, i, o))
                     : MlirAttribute{nullptr};
}

MlirType mlirTclLogicalTypeGet(MlirContext ctx, MlirType element,
                               MlirAttribute axes) {
  auto array = as<mlir::ArrayAttr>(axes);
  if (!array || mlirTypeIsNull(element)) return {nullptr};
  return wrap(checked<LogicalType>(ctx, unwrap(element), array));
}

bool mlirTypeIsATclLogical(MlirType type) {
  return mlir::isa_and_nonnull<LogicalType>(unwrap(type));
}

MlirType mlirTclLogicalTypeGetElementType(MlirType type) {
  return wrap(mlir::cast<LogicalType>(unwrap(type)).getElementType());
}

MlirAttribute mlirTclLogicalTypeGetAxes(MlirType type) {
  return wrap(mlir::cast<LogicalType>(unwrap(type)).getAxes());
}

MlirType mlirTclMappedTypeGet(MlirContext ctx, MlirType element,
                              MlirAttribute mapping) {
  auto m = as<DramMappingAttr>(mapping);
  if (!m || mlirTypeIsNull(element)) return {nullptr};
  return wrap(checked<MappedType>(ctx, unwrap(element), m));
}

}  // extern "C"
