#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlirx.h"

namespace mlirx {

    static mlir::Attribute ArrayToElements(mlir::Attribute attr) {
        if (auto array = attr.dyn_cast<mlir::DenseI64ArrayAttr>()) {
            return mlir::DenseIntElementsAttr::get(
                mlir::RankedTensorType::get(array.size(), array.getElementType()),
                array.asArrayRef());
        }
        if (auto array = attr.dyn_cast<mlir::DenseBoolArrayAttr>()) {
            return mlir::DenseIntElementsAttr::get(
                mlir::RankedTensorType::get(array.size(), array.getElementType()),
                array.asArrayRef());
        }
        return attr;
    }

}

MlirAttribute mlirDenseArrayToElements(MlirAttribute attr) {
    return wrap(mlirx::ArrayToElements(unwrap(attr)));
}
