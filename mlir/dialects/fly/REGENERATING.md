# Regenerating the operation builders

These are instructions for an agent. They describe what `ops.zig` and
`../fly_rocdl/ops.zig` must contain after a FlyDSL bump, where to derive them from,
and how to check the result. Scripts and intermediate files are not
committed; only the generated builders are.

Paths below are relative to this directory.

## Goal

`ops.zig` binds every operation of the `fly` dialect and `../fly_rocdl/ops.zig` every
operation of the `fly_rocdl` dialect, one builder per operation, so that no
code builds a FlyDSL operation from its name. The attributes and types
(`attributes.zig`, `types.zig`, `../fly_rocdl/fly_rocdl.zig`, `capi.*`, `../fly_rocdl/capi.*`) are maintained by
hand and are not covered here.

## Sources

The FlyDSL commit is the one `third_party/flydsl/repo.bzl` pins, with its
patches applied: read the `.td` files from the fetched repository
(`$(bazel info output_base)/external/+non_module_deps+flydsl`), never from an
unpatched checkout.

- `include/flydsl/Dialect/Fly/IR/FlyOps.td` for `ops.zig`.
- `include/flydsl/Dialect/FlyROCDL/IR/Ops.td` for `../fly_rocdl/ops.zig`.

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

Builder names, parameter order, `opts`, operand and result types, doc comments
and `operation_names` follow `mlir/dialects/README.md`. Specific to FlyDSL:

- Builders are sorted by mnemonic. The doc comment gives the `summary` and
  whether result types are inferred, explicit or absent.
- Unlike the README rule, parameter names are the ODS names in snake case. The
  reserved locals `ctx`, `opts`, `location`, `make`, `operands`, `attributes`
  and `present` also take a trailing underscore.
- Operations with `AttrSizedOperandSegments` pass every operand group through
  `Operation.make`'s `.variadic` operands, which writes `operandSegmentSizes`;
  others pass a flat list (an optional operand may only be last).
- Inferred operations set `.result_type_inference = true`. Inference comes
  from the flattened trait list (trait lists such as `Pure` expand to their
  members); an `OpInterface` record whose `cppInterfaceName` is
  `InferTypeOpInterface` marks it.
- Attributes are `*const mlir.Attribute`, set under their ODS name.
- Builders call `make` from `fly.zig`, which verifies the operation and names
  the operand types when creation fails.
- Each file starts with a header naming the FlyDSL commit and the `.td` it
  was generated from.

Regions, successors and `VariadicOfVariadic` operands have no use in either
dialect today; when one appears, extend the generator and this file.

## Checks

- Format and run the conformance checks of `mlir/dialects/README.md`.
- `bazel test //mlir/dialects/fly:test //dsl/fly:test`, which includes the
  registration tests of the `operation_names` lists in both packages'
  `test.zig`.
- Callers that no longer compile are the point of generating: an operation
  whose signature changed in FlyDSL now fails at the call site. Fix them in
  the same change (`dsl/fly/builder.zig`, `zml/kernel.zig`, `zml/kernels/*/fly/`,
  and downstream llmd's `llmd/kernels/*/fly/`).
- Emit the IR of the fly kernels before and after (for instance
  llmd's `sparse_mla.Main2D.emit` and the `zml/kernels/moe/fly/` kernels with
  representative configs) and
  diff it: a regeneration without an upstream change must be byte-identical.
