#ifndef CUTE_IR_DIALECT_CUTE_IR_CUTE_PRIVATE_COMPAT_TYPES_TD
#define CUTE_IR_DIALECT_CUTE_IR_CUTE_PRIVATE_COMPAT_TYPES_TD

include "cute_ir/Dialect/Cute/IR/CuteAttrs.td"

def Cute_ArithTupleIteratorType
    : Cute_Type<"ArithTupleIterator", "arith_tuple_iter"> {
  let summary = "Iterator over an arithmetic tuple";
  let parameters = (ins
    Cute_Param<"IntTupleType", "arithmetic tuple carried by the iterator">:$arithTuple
  );
  let hasCustomAssemblyFormat = 1;
}

def Cute_ConstrainedInt32Type : Cute_Type<"ConstrainedInt32", "i32"> {
  let summary = "32-bit integer with a divisibility constraint";
  let parameters = (ins
    Cute_Param<"uint64_t", "known positive divisor">:$divisibleBy
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Cute_ConstrainedInt64Type : Cute_Type<"ConstrainedInt64", "i64"> {
  let summary = "64-bit integer with a divisibility constraint";
  let parameters = (ins
    Cute_Param<"uint64_t", "known positive divisor">:$divisibleBy
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Cute_FastDivmodDivisorType
    : Cute_Type<"FastDivmodDivisor", "fast_divmod_divisor"> {
  let summary = "Precomputed divisor used by fast divmod operations";
  let parameters = (ins
    Cute_Param<"unsigned", "integer width">:$width,
    Cute_Param<"bool", "whether the divisor is a power of two">:$isPow2
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Cute_SparseElemType : Cute_Type<"SparseElem", "sparse_elem"> {
  let summary = "Logical sparse element represented by physical storage";
  let parameters = (ins
    Cute_Param<"int", "number of logical elements">:$numLogical,
    Cute_Param<"mlir::Type", "physical storage type">:$physicalType
  );
  let assemblyFormat = "`<` $numLogical `,` qualified($physicalType) `>`";
  let genVerifyDecl = 1;
}

def Cute_PtrType : Cute_Type<"Ptr", "ptr"> {
  let summary = "CuTe pointer with memory-space, alignment, and swizzle metadata";
  let parameters = (ins
    Cute_OptionalParam<"mlir::Type", "optional pointed-to value type">:$valueType,
    Cute_Param<"mlir::StringAttr", "CuTe address-space mnemonic">:$memorySpace,
    Cute_Param<"uint64_t", "byte alignment">:$alignment,
    Cute_OptionalParam<"SwizzleAttr", "optional pointer swizzle">:$swizzle
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def Cute_MemRefType : Cute_Type<"MemRef", "memref"> {
  let summary = "CuTe tensor view pairing a pointer with a static layout";
  let parameters = (ins
    Cute_Param<"PtrType", "underlying pointer">:$ptr,
    Cute_Param<"mlir::Type", "layout or composed-layout type">:$layout
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    mlir::Type getValueType() const { return getPtr().getValueType(); }
    mlir::StringAttr getMemorySpace() const { return getPtr().getMemorySpace(); }
    uint64_t getAlignment() const { return getPtr().getAlignment(); }
    SwizzleAttr getSwizzle() const { return getPtr().getSwizzle(); }
  }];
}

def Cute_CoordTensorType : Cute_Type<"CoordTensor", "coord_tensor"> {
  let summary = "Coordinate iterator paired with a CuTe layout";
  let parameters = (ins
    Cute_Param<"IntTupleType", "coordinate arithmetic tuple">:$arithTuple,
    Cute_Param<"mlir::Type", "layout or composed-layout type">:$layout
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // CUTE_IR_DIALECT_CUTE_IR_CUTE_PRIVATE_COMPAT_TYPES_TD
