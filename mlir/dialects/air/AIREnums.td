//===- AIREnums.td - Metal AIR enums -----------------------*- tablegen -*-===//

#ifndef MLIR_DIALECTS_AIR_AIRENUMS_TD
#define MLIR_DIALECTS_AIR_AIRENUMS_TD

include "mlir/IR/EnumAttr.td"

def AIR_Builtin : I32EnumAttr<"Builtin", "Kernel argument builtin", [
  I32EnumAttrCase<"thread_position_in_grid", 0>,
  I32EnumAttrCase<"threadgroup_position_in_grid", 1>,
  I32EnumAttrCase<"thread_position_in_threadgroup", 2>,
  I32EnumAttrCase<"threads_per_threadgroup", 3>,
  I32EnumAttrCase<"threadgroups_per_grid", 4>,
  I32EnumAttrCase<"threads_per_grid", 5>,
  I32EnumAttrCase<"thread_index_in_simdgroup", 6>,
  I32EnumAttrCase<"simdgroup_index_in_threadgroup", 7>,
  I32EnumAttrCase<"threads_per_simdgroup", 8>,
  I32EnumAttrCase<"simdgroups_per_threadgroup", 9>,
  I32EnumAttrCase<"thread_index_in_threadgroup", 10>,
]> {
  let cppNamespace = "::mlir::metal::air";
  let genSpecializedAttr = 0;
}

def AIR_AddressSpace : I32EnumAttr<"AddressSpace", "Address space", [
  I32EnumAttrCase<"thread", 0>,
  I32EnumAttrCase<"device", 1>,
  I32EnumAttrCase<"constant", 2>,
  I32EnumAttrCase<"threadgroup", 3>,
]> {
  let cppNamespace = "::mlir::metal::air";
  let genSpecializedAttr = 0;
}

def AIR_Access : I32EnumAttr<"Access", "Buffer access", [
  I32EnumAttrCase<"read", 0>,
  I32EnumAttrCase<"read_write", 1>,
]> {
  let cppNamespace = "::mlir::metal::air";
  let genSpecializedAttr = 0;
}

def AIR_MemFlags : I32BitEnumAttr<"MemFlags", "Barrier memory flags", [
  I32BitEnumAttrCaseNone<"none">,
  I32BitEnumAttrCaseBit<"device", 0>,
  I32BitEnumAttrCaseBit<"threadgroup", 1>,
  I32BitEnumAttrCaseBit<"texture", 2>,
]> {
  let separator = "|";
  let cppNamespace = "::mlir::metal::air";
  let genSpecializedAttr = 0;
}

#endif // MLIR_DIALECTS_AIR_AIRENUMS_TD
