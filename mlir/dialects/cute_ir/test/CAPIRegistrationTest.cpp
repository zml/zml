//===- CAPIRegistrationTest.cpp - CuTe C API smoke test ------------------===//

#include "cute_ir-c/Dialect/Cute.h"
#include "cute_ir-c/Dialect/CuteNVGPU.h"
#include "mlir-c/IR.h"

#include <cstdio>
#include <string>

namespace {

int fail(MlirContext context, const char *message) {
  std::fprintf(stderr, "%s\n", message);
  mlirContextDestroy(context);
  return 1;
}

void appendString(MlirStringRef chunk, void *userData) {
  static_cast<std::string *>(userData)->append(chunk.data, chunk.length);
}

} // namespace

int main() {
  MlirContext context = mlirContextCreate();

  MlirDialectHandle cute = mlirGetDialectHandle__cute__();
  MlirDialectHandle cuteNVGPU = mlirGetDialectHandle__cute_nvgpu__();
  mlirDialectHandleRegisterDialect(cute, context);
  mlirDialectHandleRegisterDialect(cuteNVGPU, context);

  if (mlirDialectIsNull(mlirDialectHandleLoadDialect(cute, context)))
    return fail(context, "failed to load the cute dialect");
  if (mlirDialectIsNull(mlirDialectHandleLoadDialect(cuteNVGPU, context)))
    return fail(context, "failed to load the cute_nvgpu dialect");

  MlirType cutePublicType =
      mlirCuteTypeParse(context, mlirStringRefCreateFromCString(
                                     "!cute.int_tuple<\"1\">"));
  if (mlirTypeIsNull(cutePublicType) ||
      !mlirTypeIsACuteType(cutePublicType))
    return fail(context, "failed to parse an OSS cute type");

  MlirType cutePrivateType = mlirCuteTypeParse(
      context, mlirStringRefCreateFromCString("!cute.i32"));
  if (mlirTypeIsNull(cutePrivateType) ||
      !mlirTypeIsACuteType(cutePrivateType))
    return fail(context, "failed to parse a recovered cute type");

  MlirType cuteNVGPUType = mlirCuteNVGPUTypeParse(
      context, mlirStringRefCreateFromCString("!cute_nvgpu.smem_desc"));
  if (mlirTypeIsNull(cuteNVGPUType) ||
      !mlirTypeIsACuteNVGPUType(cuteNVGPUType))
    return fail(context, "failed to parse a cute_nvgpu type");

  if (!mlirContextIsRegisteredOperation(
          context, mlirStringRefCreateFromCString("cute.add_offset")))
    return fail(context, "recovered cute operation is not registered");
  if (!mlirContextIsRegisteredOperation(
          context,
          mlirStringRefCreateFromCString("cute_nvgpu.arch.alloc_rmem")))
    return fail(context, "cute_nvgpu operation is not registered");

  const char *privateOpsModule = R"mlir(
module {
  %alloc = "cute.memref.alloc_smem"() : () -> !cute.i32
  %assumed = "cute.assume"(%alloc) : (!cute.i32) -> !cute.i32
  %tuple = "cute.make_tuple"(%alloc, %assumed)
      : (!cute.i32, !cute.i32) -> !cute.i32
  "cute.copy"(%tuple, %alloc, %assumed)
      <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}>
      : (!cute.i32, !cute.i32, !cute.i32) -> ()
}
)mlir";
  MlirModule module = mlirModuleCreateParse(
      context, mlirStringRefCreateFromCString(privateOpsModule));
  if (mlirModuleIsNull(module))
    return fail(context, "failed to parse recovered cute operations");
  bool privateOpsAreValid = mlirOperationVerify(mlirModuleGetOperation(module));
  std::string bytecode;
  mlirOperationWriteBytecode(mlirModuleGetOperation(module), appendString,
                             &bytecode);
  mlirModuleDestroy(module);
  if (!privateOpsAreValid)
    return fail(context, "recovered cute operations failed verification");

  MlirOperation roundTripped = mlirOperationCreateParse(
      context, mlirStringRefCreate(bytecode.data(), bytecode.size()),
      mlirStringRefCreateFromCString("private-ops.mlirbc"));
  if (mlirOperationIsNull(roundTripped))
    return fail(context, "failed to read recovered operations from bytecode");
  bool bytecodeIsValid = mlirOperationVerify(roundTripped);
  mlirOperationDestroy(roundTripped);
  if (!bytecodeIsValid)
    return fail(context, "recovered operations failed after bytecode round trip");

  mlirContextDestroy(context);
  return 0;
}
