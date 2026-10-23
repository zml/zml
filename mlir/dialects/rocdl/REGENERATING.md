# Regenerating the ROCDL bindings

These are instructions for an agent. They describe what the generated ROCDL
bindings must contain after an LLVM bump, where to derive them from, and how to
check the result. Scripts and intermediate files are not committed; only the
generated bindings are.

## Sources

Use the LLVM revision Bazel pins, and read TableGen's resolved records rather
than the `.td` text: multiclasses instantiate many ROCDL operations.

```sh
bazel build @llvm-project//llvm:llvm-tblgen
llvm_project="$(bazel info output_base)/external/+llvm+llvm-project"
bazel-bin/external/+llvm+llvm-project/llvm/llvm-tblgen --dump-json \
  -I "$llvm_project/mlir/include" -I "$llvm_project/llvm/include" \
  "$llvm_project/mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td" \
  -o /tmp/rocdl-records.json
```

Every `Op`, `AttrDef` and `EnumInfo` record whose dialect is `ROCDL_Dialect`
belongs to the bindings.

## What each file must contain

Names, builder parameters, `opts`, wrappers, C API names and comments follow
`mlir/dialects/README.md`. Specific to ROCDL:

- `ops.zig`: `operation_names` and one builder per operation. Follow argument
  constraints through `Arg` wrappers.
- `enums.zig`: one enum per ROCDL enum, with its bit width and values. Unlike
  the README rule, a case whose IR spelling is not an identifier or is a
  keyword takes the C++ symbol (`volatile` → `volatile_op`). Bit enums are
  non-exhaustive and provide `combine` and `contains`.
- `attributes.zig`: one `<X>Attr` wrapper per enum attribute, and
  `enum_attributes` listing them.
- `types.zig`: `AddressSpace`, `pointerType`, and a `v<N><element>` constructor
  for every vector type the records use, named or inline.
- `capi.h`/`capi.cc`: `mlirROCDLGetOperationCount` under `// Introspection`
  and, per enum attribute, the predicate, the getter (null for out-of-range
  values) and `GetValue` under `// Attributes`. The `// Target` sections are
  maintained by hand: in the
  pinned revision `ROCDLTargetAttr::getChecked` delegates to `Base::get`, so the
  C wrapper verifies target inputs first.

Each generated Zig file starts with the "Generated from LLVM ROCDL TableGen
records" header. `target.zig`, `annotations.zig` and `rocdl.zig` are maintained
by hand; update their aliases and helpers when LLVM's surface changes. If the
dialect gains variadic operands, regions, successors, its own types, an enum
without an attribute, or new target fields or discardable attributes, extend
the bindings and this file.

## Checks

- Format and run the conformance checks of `mlir/dialects/README.md`.
- `bazel test //mlir/dialects/rocdl:test //dsl/fly:test //zml:test`. Beyond the
  README's tests, the operation list is compared to
  `mlirROCDLGetOperationCount`, and attribute tests exercise every named enum
  case, named bit groups and combined bitfields. Add fixtures when a change
  introduces a new operand layout or attribute representation.
- A regeneration without an LLVM change is byte-identical.
