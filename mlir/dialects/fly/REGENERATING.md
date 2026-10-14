# Regenerating the operation builders

These are instructions for an agent. They describe what `ops.zig` and
`rocdl_ops.zig` must contain after a FlyDSL bump, where to derive them from,
and how to check the result. Scripts and intermediate files are not
committed; only the generated builders are.

Paths below are relative to this directory.

## Goal

`ops.zig` binds every operation of the `fly` dialect and `rocdl_ops.zig` every
operation of the `fly_rocdl` dialect, one builder per operation, so that no
code builds a FlyDSL operation from its name. The attributes and types
(`attributes.zig`, `types.zig`, `rocdl.zig`, `fly_capi.*`) are maintained by
hand and are not covered here.

## Sources

The FlyDSL commit is the one `third_party/flydsl/repo.bzl` pins, with its
patches applied: read the `.td` files from the fetched repository
(`$(bazel info output_base)/external/+non_module_deps+flydsl`), never from an
unpatched checkout.

- `include/flydsl/Dialect/Fly/IR/FlyOps.td` for `ops.zig`.
- `include/flydsl/Dialect/FlyROCDL/IR/Ops.td` for `rocdl_ops.zig`.

Dump their records with the LLVM the workspace builds:

```sh
bazel build @llvm-project//llvm:llvm-tblgen
llvm-tblgen --dump-json \
  -I <flydsl>/include -I <output_base>/external/+llvm+llvm-project/mlir/include \
  <flydsl>/include/flydsl/Dialect/Fly/IR/FlyOps.td -o FlyOps.json
```

Every record in `!instanceof.Op` whose `opDialect` names the dialect is an
operation. Do not regex the `.td` files.

## What each builder must be

- One `pub fn` per operation, named by its mnemonic with dots replaced by
  underscores (`fly.tiled_copy.partition_src` → `tiled_copy_partition_src`),
  sorted by mnemonic, with a doc comment naming the operation, its `summary`,
  and whether result types are inferred, explicit, or absent.
- Parameters, in order: `ctx`, the operands, the result types (only when the
  operation does not implement `InferTypeOpInterface`), the attributes, then
  `location`. Names are the ODS names in snake case, with a trailing
  underscore when they collide with a Zig keyword, a builder name or a
  reserved local (`ctx`, `location`, `make`, `operands`, `attributes`,
  `present`).
- Operands: single → `*const mlir.Value`, `Optional` → `?*const mlir.Value`,
  `Variadic` → `[]const *const mlir.Value`. Operations with
  `AttrSizedOperandSegments` pass every operand group through
  `Operation.make`'s `.variadic` operands, which writes `operandSegmentSizes`;
  others pass a flat list (an optional operand may only be last).
- Results: single → `*const mlir.Type`, `Variadic` →
  `[]const *const mlir.Type` named `<name>_types`. Inferred operations set
  `.result_type_inference = true` and take no result types. Inference comes
  from the flattened trait list (trait lists such as `Pure` expand to their
  members); an `OpInterface` record whose `cppInterfaceName` is
  `InferTypeOpInterface` marks it.
- Attributes: `*const mlir.Attribute`, optional (`?`) when the attribute is
  `OptionalAttr`, `UnitAttr` or has a default value; set under its ODS name.
- Builders call `make` from `fly.zig`, which verifies the operation and names
  the operand types when creation fails.
- Each file starts with a header naming the FlyDSL commit and the `.td` it
  was generated from, and exports `names`, the list of operation names, for
  the registration test.

Regions, successors and `VariadicOfVariadic` operands have no use in either
dialect today; when one appears, extend the generator and this file.

## Checks

- `zig fmt` the generated files.
- `bazel test //mlir/dialects/fly:test //kernels/fly:test`: every name in
  `ops.names` and `rocdl.ops.names` must be a registered operation.
- Callers that no longer compile are the point of generating: an operation
  whose signature changed in FlyDSL now fails at the call site. Fix them in
  the same change (`kernels/fly/builder.zig`, `zml/*/fly_kernels`).
- Emit the IR of the fly kernels before and after (for instance
  `sparse_mla.Main2D.emit` and its siblings with representative configs) and
  diff it: a regeneration without an upstream change must be byte-identical.
