#include "mlir/dialects/tcl/tcl_types.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/dialects/tcl/tcl_attrs.h"
#include "mlir/dialects/tcl/tcl_dialect.h"
namespace xla::furiosa::tcl {
using namespace mlir;  // NOLINT
using Error = llvm::function_ref<InFlightDiagnostic()>;
llvm::StringRef ElementName(Type t) {
  if (t.isBF16()) return "bf16";
  if (t.isF16()) return "f16";
  if (t.isF32()) return "f32";
  if (t.isF64()) return "f64";
  if (isa<Float4E2M1FNType>(t)) return "f4_e2";
  if (isa<Float8E4M3FNType>(t)) return "f8_e4";
  if (isa<Float8E5M2Type>(t)) return "f8_e5";
  if (auto i = dyn_cast<IntegerType>(t)) {
    if (i.getWidth() == 1 && i.isSignless()) return "bool";
    if (i.isUnsigned()) {
      switch (i.getWidth()) {
        case 8:
          return "u8";
        case 16:
          return "u16";
        case 32:
          return "u32";
        case 64:
          return "u64";
      }
    } else {
      switch (i.getWidth()) {
        case 4:
          return "i4";
        case 8:
          return "i8";
        case 16:
          return "i16";
        case 32:
          return "i32";
        case 64:
          return "i64";
      }
    }
  }
  return {};
}
LogicalResult LogicalType::verify(Error e, Type element, ArrayAttr axes) {
  if (ElementName(element).empty())
    return e() << "unsupported TCL element type";
  return VerifyAxes(e, axes);
}
LogicalResult MappedType::verify(Error e, Type element,
                                 DramMappingAttr mapping) {
  return ElementName(element).empty() ? e() << "unsupported TCL element type"
                                      : success();
}
}  // namespace xla::furiosa::tcl
