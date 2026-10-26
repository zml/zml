//===- AIRAttributes.td - Metal AIR attributes -------------*- tablegen -*-===//

#ifndef MLIR_DIALECTS_AIR_AIRATTRIBUTES_TD
#define MLIR_DIALECTS_AIR_AIRATTRIBUTES_TD

include "mlir/dialects/air/AIRDialect.td"
include "mlir/dialects/air/AIREnums.td"

def AIR_BuiltinAttr : EnumAttr<AIR_Dialect, AIR_Builtin, "builtin"> {
  let summary = "Kernel argument builtin";
  let description = [{
    Attach as `air.builtin` to a kernel argument. The position and size
    builtins are `uint` or `ushort` scalars or 2- and 3-element vectors
    (`i32`, `i16`, `vector<3xi32>`, ...); the index builtins are `uint` or
    `ushort` scalars.
  }];
}

def AIR_MemFlagsAttr : EnumAttr<AIR_Dialect, AIR_MemFlags, "mem_flags"> {
  let summary = "Memory a barrier orders";
}

def AIR_StructField : AttrDef<AIR_Dialect, "StructField"> {
  let mnemonic = "struct_field";
  let summary = "Field of a constant buffer struct";
  let description = [{
    One `air.struct_type_info` entry: the byte offset, the element size,
    the array length (zero for a non-array field), the Metal type name and
    the field name.
  }];
  let parameters = (ins "uint64_t":$offset,
                         "uint64_t":$size,
                         DefaultValuedParameter<"uint64_t", "0">:$count,
                         StringRefParameter<>:$typeName,
                         StringRefParameter<>:$name);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def AIR_Buffer : AttrDef<AIR_Dialect, "Buffer"> {
  let mnemonic = "buffer";
  let summary = "Kernel buffer argument";
  let description = [{
    Attach as `air.buffer` to a kernel argument of type `!llvm.ptr` in
    `addressSpace`. `elementType` is the pointee (a scalar, a vector, or an
    `!llvm.struct` described by `fields`), `typeName` its Metal spelling,
    and `typeSize` and `typeAlignment` its size and alignment in bytes.
    `locationIndex` is the `[[buffer(n)]]` or `[[threadgroup(n)]]` index.
    A constant reference records its size as `bufferSize`. `noalias` is a
    `__restrict__` pointer.
  }];
  let parameters = (ins "unsigned":$locationIndex,
                         EnumParameter<AIR_AddressSpace>:$addressSpace,
                         EnumParameter<AIR_Access>:$access,
                         "::mlir::Type":$elementType,
                         StringRefParameter<>:$typeName,
                         "uint64_t":$typeSize,
                         "uint64_t":$typeAlignment,
                         DefaultValuedParameter<"uint64_t", "0">:$bufferSize,
                         DefaultValuedParameter<"bool", "false">:$noalias,
                         OptionalArrayRefParameter<"StructFieldAttr">:$fields);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

#endif // MLIR_DIALECTS_AIR_AIRATTRIBUTES_TD
