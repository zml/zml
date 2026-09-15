#include "cute_ir-c/Dialect/Cute.h"
#include "cute_ir-c/Dialect/CuteNVGPU.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "failed at line %d: %s\n", __LINE__, #condition);        \
      mlirContextDestroy(context);                                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  MlirContext context = mlirContextCreate();
  MlirType nullType = {NULL};
  MlirAttribute nullAttr = {NULL};
  MlirType f32 = mlirF32TypeGet(context);
  MlirAttribute space =
      mlirStringAttrGet(context, mlirStringRefCreateFromCString("smem"));
  MlirAttribute layoutAttr =
      mlirCuteLayoutAttrGet(context, mlirStringRefCreateFromCString("4:1"));
  CHECK(mlirAttributeIsACuteLayout(layoutAttr));
  CHECK(!mlirAttributeIsACuteLayout(nullAttr));
  CHECK(!mlirAttributeIsACuteLayout(space));
  MlirStringRef layoutValue = mlirCuteLayoutAttrGetValue(layoutAttr);
  CHECK(layoutValue.length == 3 && memcmp(layoutValue.data, "4:1", 3) == 0);
  CHECK(mlirAttributeIsNull(mlirCuteLayoutAttrGet(
      context, mlirStringRefCreateFromCString("invalid"))));

  // Constructors load their dialect and build storage without parsing a type.
  MlirType pointer = mlirCutePtrTypeGet(context, f32, space, 16, nullAttr);
  CHECK(mlirTypeIsACutePtr(pointer));
  CHECK(mlirTypeEqual(mlirCutePtrTypeGetValueType(pointer), f32));
  CHECK(mlirAttributeEqual(mlirCutePtrTypeGetMemorySpace(pointer), space));
  CHECK(mlirCutePtrTypeGetAlignment(pointer) == 16);
  CHECK(mlirAttributeIsNull(mlirCutePtrTypeGetSwizzle(pointer)));
  CHECK(!mlirTypeIsACutePtr(f32));
  CHECK(!mlirTypeIsACutePtr(nullType));
  CHECK(
      mlirTypeIsNull(mlirCutePtrTypeGet(context, f32, nullAttr, 16, nullAttr)));

  MlirType integer = mlirCuteConstrainedInt32TypeGet(context, 8);
  CHECK(mlirTypeIsACuteConstrainedInt32(integer));
  CHECK(mlirCuteConstrainedInt32TypeGetDivisibleBy(integer) == 8);

  MlirType layout = mlirCuteLayoutTypeGet(context, layoutAttr);
  CHECK(mlirTypeIsACuteLayout(layout));
  CHECK(mlirAttributeEqual(mlirCuteLayoutTypeGetAttr(layout), layoutAttr));
  CHECK(mlirTypeIsNull(mlirCuteLayoutTypeGet(context, space)));
  MlirType memref = mlirCuteMemRefTypeGet(context, pointer, layout);
  CHECK(mlirTypeIsACuteMemRef(memref));
  CHECK(mlirTypeEqual(mlirCuteMemRefTypeGetPtr(memref), pointer));
  CHECK(mlirTypeEqual(mlirCuteMemRefTypeGetLayout(memref), layout));

  MlirType descriptor = mlirCuteNVGPUSmemDescTypeGet(context);
  CHECK(mlirTypeIsACuteNVGPUSmemDesc(descriptor));
  MlirAttribute payload =
      mlirStringAttrGet(context, mlirStringRefCreateFromCString("<f32>"));
  MlirType atom = mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(context, payload);
  CHECK(mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy(atom));
  CHECK(mlirAttributeEqual(
      mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetPayload(atom), payload));
  mlirContextDestroy(context);
  return 0;
}
