# Regenerating ROCDL bindings

Run these commands from the repository root against the LLVM revision selected
by Bazel. The input must be TableGen's resolved records, rather than a textual
scan of `.td` definitions; multiclasses instantiate many ROCDL operations.

```sh
bazel build @llvm-project//llvm:llvm-tblgen
llvm_project="$(bazel info output_base)/external/+llvm+llvm-project"
bazel-bin/external/+llvm+llvm-project/llvm/llvm-tblgen --dump-json \
  -I "$llvm_project/mlir/include" -I "$llvm_project/llvm/include" \
  "$llvm_project/mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td" \
  -o /tmp/rocdl-records.json
python3 mlir/dialects/rocdl/generate.py /tmp/rocdl-records.json
zig fmt mlir/dialects/rocdl/*.zig
bazel test //mlir/dialects/rocdl:test //kernels/fly:test //zml:test
```

`generate.py` writes `ops.zig`, `enums.zig`, `attributes.zig`, `types.zig`, and
the registration and enum-attribute sections of `capi.h` and `capi.cc`. It keeps
the handwritten `// Target` sections in those C API files. `target.zig`,
`annotations.zig`, and the `rocdl.zig` entry point and tests are maintained by
hand. Update their exported aliases and helpers when LLVM's surface changes.

The generator follows argument constraints through `Arg` wrappers, distinguishes
operands from attributes, preserves required/defaulted/optional attributes,
and emits explicit result types. This revision has no variadic operands,
regions, successors, or independent type definitions. Generation stops if those
unsupported constructs appear, or if target fields, enum coverage, or discardable
attributes change.

Keep the C API in `capi.cc`/`capi.h`, with section comments between registration,
attributes, and target configuration. In the pinned LLVM revision, the custom
`ROCDLTargetAttr::getChecked` builder delegates to `Base::get`. The C wrapper
explicitly verifies target inputs first to prevent invalid inputs from asserting.

All generated declarations are referenced during tests, and the operation list
is compared to the native registry. Attribute tests exercise every named enum
case, including named bit groups, and combined bitfields. Add runtime fixtures
when a change introduces a new operand layout or attribute representation.
