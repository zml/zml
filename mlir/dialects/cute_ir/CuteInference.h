//===- CuteInference.h - result types the compiler infers --------*- C++ -*-===//

#ifndef CUTE_IR_DIALECT_CUTE_IR_CUTE_INFERENCE_H
#define CUTE_IR_DIALECT_CUTE_IR_CUTE_INFERENCE_H

#include "cute_ir/Dialect/Cute/IR/CuteDialect.h"

namespace mlir::cutlass_compiler::cute {
// make_tiled_copy_D: `atom` tiled over `tiled`'s destination thread-value
// layout and tiler, or null when the atom is not one this tree knows.
Type tiledCopyD(Type atom, TiledCopyType tiled);

// Result types of operations the compiler does not infer (their builders
// take the type): what the DSL passes, or null when not computable here.
Type makeLayoutResultType(Type shape, Type stride);  // stride may be null
Type makeViewResultType(Type iter, Type layout);
Type getResultType(Type input, ArrayRef<int32_t> mode);
Type assumeResultType(Type src);
Type makeArithTupleIterResultType(Type tuple);
Type fastDivmodCreateDivisorResultType(Type divisor);
Type fastDivmodComputeResultType(Type divisor);  // both results
} // namespace mlir::cutlass_compiler::cute

#endif // CUTE_IR_DIALECT_CUTE_IR_CUTE_INFERENCE_H
