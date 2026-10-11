# Regenerating the operation builders

These are instructions for an agent. `ops.zig` (`ttkernel`),
`ttcore_ops.zig` (`ttcore`) and `../ttl/ops.zig` (`ttl`) are generated from
tt-lang's operation definitions after a tt-lang bump; the attributes and
types (`ttkernel.zig`, `ttcore.zig`, `../ttl/ttl.zig`, the `*_capi.*` files)
are maintained by hand. Scripts and intermediate files are not committed;
only the generated builders are.

## Sources

The tt-lang commit is the one `third_party/tt_lang/repo.bzl` pins, with its
patches applied: read the `.td` files from the fetched repository
(`$(bazel info output_base)/external/+non_module_deps+tt_lang`).

- `include/ttlang/Dialect/TTKernel/IR/TTKernelOps.td` for `ops.zig`.
- `include/ttlang/Dialect/TTCore/IR/TTCoreOps.td` for `ttcore_ops.zig`.
- `include/ttlang/Dialect/TTL/IR/TTLOps.td` for `../ttl/ops.zig`.

Dump their records with the LLVM the workspace builds:

```sh
bazel build @llvm-project//llvm:llvm-tblgen
llvm-tblgen --dump-json \
  -I <tt_lang>/include -I <output_base>/external/+llvm+llvm-project/mlir/include \
  <tt_lang>/include/ttlang/Dialect/TTKernel/IR/TTKernelOps.td -o TTKernelOps.json
```

Every record in `!instanceof.Op` whose `opDialect` is `TTKernel_Dialect`
(`TTCore_Dialect`, `TTL_Dialect`) is an operation of that file.

## What each builder must be

The rules of `../fly/REGENERATING.md`, with these additions:

- Parameters also may not shadow a declaration of the file: the builder
  names, `names`, `make`, `std`, `mlir`, `stdx`. Such names get a trailing
  underscore.
- An operation with regions (`ttl.compute`, `ttl.pipenet_foreach_src`...)
  takes each region's entry block after its result types, named
  `<region>_block` (`[]const *mlir.Block` `<region>_blocks` for a variadic
  region).
- Operands of operations without `AttrSizedOperandSegments` that mix
  optional or variadic groups are concatenated in order.
- A single result whose ODS type has a `builderCall` (`I32`, `Index`,
  `TTKernel_NocAddr`...) is built by the builder, which takes no type for
  it: `mlir.Type.int`/`mlir.Type.index` for integer and index types,
  `helpers.parseType(ctx, "!<typeName>")` for dialect types. The doc
  comment says "Result types are fixed".
- `helpers` is `ttkernel.zig` (`ttl.zig` for `../ttl/ops.zig`), which
  provides `make` and `parseType`.
- Each file ends with `num_results`, one `comptime_int` per builder: the
  number of results of its operation, -1 when a result group is variadic or
  optional. The kernel DSLs use it to type their generic operation call.
- The header names the tt-lang commit and the `.td`; each file exports
  `names` for the registration tests.

## Checks

- `zig fmt` the generated files.
- `bazel test //mlir/dialects/ttkernel:test //mlir/dialects/ttl:test`: every
  builder compiles and every name in `names` is a registered operation.
- Fix the callers that no longer compile (`kernels/ttkernel`, `kernels/ttl`)
  in the same change.
