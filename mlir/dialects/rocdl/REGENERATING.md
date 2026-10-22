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

- `ops.zig`: `names`, every operation sorted by name, and one builder
  per operation named by its mnemonic with dots replaced by underscores.
  Parameters, in order: `ctx`, the operands, the result types (`<name>_type`),
  the required attributes, an `opts` struct of the optional, defaulted and unit
  attributes (each `= null`), then `location`. Follow argument constraints
  through `Arg` wrappers. Names that are Zig keywords or primitives are quoted
  (`@"..."`); names that collide with the builder's locals or the file's
  declarations take a trailing underscore.
- `enums.zig`: one enum per ROCDL enum, with its bit width and values. Fields
  use the IR spelling, or the C++ symbol when that spelling is not an identifier
  or is a keyword. Bit enums are non-exhaustive and provide `combine` and
  `contains`.
- `attributes.zig`: one `<X>Attr` wrapper per enum attribute (`get`, `value`,
  `attribute`), and `enum_attributes` listing them.
- `types.zig`: `AddressSpace`, `pointerType`, and a `v<N><element>` constructor
  for every vector type the records use, named or inline.
- `capi.h`/`capi.cc`: `mlirROCDLGetOperationCount` under `// Introspection`
  and, per enum attribute, `mlirAttributeIsAROCDL<X>`, `mlirROCDL<X>AttrGet`
  (null for out-of-range values) and `mlirROCDL<X>AttrGetValue` under
  `// Attributes`. The `// Target` sections are maintained by hand: in the
  pinned revision `ROCDLTargetAttr::getChecked` delegates to `Base::get`, so the
  C wrapper verifies target inputs first.

Each generated Zig file starts with the "Generated from LLVM ROCDL TableGen
records" header. `target.zig`, `annotations.zig` and `rocdl.zig` are maintained
by hand; update their aliases and helpers when LLVM's surface changes. If the
dialect gains variadic operands, regions, successors, its own types, an enum
without an attribute, or new target fields or discardable attributes, extend
the bindings and this file.

## Checks

- Format with the Zig toolchain Bazel uses (CI runs its `zig fmt --check`); an
  older `zig fmt` leaves builtins that 0.17 rewrites. The C API follows
  `mlir/.clang-format`.
- `bazel test //mlir/dialects/rocdl:test //kernels/fly:test //zml:test`: every
  generated declaration is referenced, the operation list is compared to the
  native registry, and attribute tests exercise every named enum case, named
  bit groups and combined bitfields. Add fixtures when a change introduces a
  new operand layout or attribute representation.
- A regeneration without an LLVM change is byte-identical.
