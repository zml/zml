# ZML Source-to-HLO Explorer POC

This proof of concept compiles the bundled `forward` function on CPU and emits
a self-contained artifact bundle containing the Zig source, post-ZML-pass
StableHLO, XLA's pre-optimization HLO, and a provenance graph connecting them.

Generate an artifact bundle:

```sh
bazel run //tools/source_to_hlo_explorer:poc -- --output=/tmp/zml-hlo-poc
```

Launch the dependency-free viewer:

```sh
python3 tools/source_to_hlo_explorer/serve.py /tmp/zml-hlo-poc --open
```

Omit the artifact directory to open the checked-in fixture instead. The server
uses only Python's standard library and listens on `127.0.0.1:8000` by default.
Use `--port 0` to select an available port.

The model source is ordinary ZML code with no explicit provenance arguments:

```zig
const sum = x.add(y);
const doubled = sum.mulConstant(2);
```

Before compilation, a dependency-free Zig AST tool creates a formatting-
preserving shadow source. It rewrites recognized `zml.Tensor` calls to the
internal `addAt` and `mulConstantAt` provenance APIs and supplies locations from
the untouched source. Bazel compiles the generated module while the artifact
bundle and viewer continue to use the original `source.zig`.

The POC deliberately limits type tracking to parameters spelled `zml.Tensor`
and local results of registered tensor operations. Extending it to arbitrary
aliases, shadowing, and interprocedural flow will require semantic type data.

The generated bundle contains:

- `source.zig`
- `stablehlo.mlir`
- `hlo.before_optimizations.txt`
- `hlo.before_optimizations.pb`
- `provenance.json`
- `mapping.json`

For this narrow POC, add and multiply retain exact XLA metadata. The current XLA
lowering drops operation metadata for scalar constants and broadcasts, so those
two links are recovered by following matching operand positions from the mapped
multiply through broadcast to constant. `mapping.json` records that recovery as
`dataflow_operand`, and artifact generation fails if any StableHLO operation is
left unresolved.
