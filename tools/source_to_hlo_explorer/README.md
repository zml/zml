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

The explicit provenance API uses `addAt` and `mulConstantAt`. Zig does not
support overloads or default arguments, so these methods preserve compatibility
with the existing `add` and `scale` call sites while accepting `@src()`:

```zig
const sum = x.addAt(y, @src());
const doubled = sum.mulConstantAt(2, @src());
```

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
