# ZML Source-to-HLO Explorer POC

The explorer is a producer-agnostic artifact consumer. Its packer and web
viewer do not invoke Zig, ZML, StableHLO, or XLA compilation; they consume
already-produced source and IR files that follow the provenance contract
described below. The bundled ZML POC is one optional way to produce those
inputs, not a requirement of the viewer.

## Pack compiler outputs

The packer requires four inputs:

- `--source`: the untouched source displayed in the source pane.
- `--source-map`: the JSON sidecar mapping source-expression spans to the
  provenance locations attached to StableHLO operations.
- `--stablehlo`: post-lowering StableHLO text containing the surviving
  `zml.stable_op.N` provenance markers and source locations.
- `--hlo`: XLA pre-optimization HLO text corresponding to that StableHLO.

The source map uses one-based positions. Each expression records the source
span to highlight and the provenance anchor carried by its StableHLO
operations:

```json
{
  "version": 2,
  "original_file": "model.zig",
  "expressions": [
    {
      "file": "model.zig",
      "line": 4,
      "column": 17,
      "end_line": 4,
      "end_column": 25,
      "start_byte": 103,
      "end_byte": 111,
      "provenance_line": 4,
      "provenance_column": 19,
      "method": "add"
    }
  ]
}
```

`file`, `provenance_line`, and `provenance_column` must match the source
location attached to the corresponding `zml.stable_op.N` operation.

Use absolute paths because the working directory of a program launched by
`bazel run` is not guaranteed to be the workspace:

```sh
bazel run //tools/source_to_hlo_explorer:pack -- \
  --source=/abs/path/model.zig \
  --source-map=/abs/path/model.source-map.json \
  --stablehlo=/abs/path/module.mlir \
  --hlo=/abs/path/module.before_optimizations.txt \
  --output=/abs/path/bundle
```

The resulting viewer bundle contains exactly four required files:

- `source.zig`
- `stablehlo.mlir`
- `hlo.before_optimizations.txt`
- `mapping.json`

Neither an HLO protobuf nor a separate `provenance.json` is required by the
viewer. The packer derives surviving provenance from the StableHLO text,
enriches it with the source-map spans, and correlates those operations with the
pre-optimization HLO.

Launch the dependency-free viewer with the packed directory:

```sh
python3 tools/source_to_hlo_explorer/serve.py /abs/path/bundle --open
```

The server uses only Python's standard library and listens on `127.0.0.1:8000`
by default. Use `--port 0` to select an available port. Omitting the bundle
directory serves the checked-in fixture.

## Optional ZML input production

ZML can transparently generate the source-map input without changing model
source. For example, the bundled model remains ordinary ZML code:

```zig
const sum = x.add(y);
const doubled = sum.mulConstant(2);
```

Before compilation, the dependency-free Zig AST tool creates a
formatting-preserving shadow source. It rewrites recognized `zml.Tensor` calls
to internal provenance APIs, supplies locations from the untouched source, and
emits the JSON source-map sidecar. Bazel compiles the generated module; the
source pane continues to display the original file.

StableHLO and HLO production is a separate step. A compiler driver must emit
post-lowering StableHLO with the stable-operation provenance markers and XLA's
matching pre-optimization HLO text. Those outputs, the original source, and the
AST sidecar can then be passed to the generic pack command above. Another
compiler or build system can supply the same four inputs without using the ZML
AST transformer.

The bundled end-to-end POC still produces a directly serveable artifact bundle:

```sh
bazel run //tools/source_to_hlo_explorer:poc -- --output=/tmp/zml-hlo-poc
python3 tools/source_to_hlo_explorer/serve.py /tmp/zml-hlo-poc --open
```

The current AST registry covers `add`, `mulConstant`, `flatten`, `convert`,
`dot`, `relu`, and `argMax`. Syntax-level type flow follows parameters spelled
`zml.Tensor`, explicitly typed tensor fields, local tensor results, `withTags`,
and the tensor-valued `values` and `indices` projections from `argMax`. Chained
calls use distinct method-token provenance anchors and non-overlapping source
segments, so each operation remains independently selectable.

MNIST exercises that broader producer path without modifying `mnist.zig`:

```sh
bazel test \
  //examples/mnist:mnist_ast_compile_test \
  //examples/mnist:mnist_ast_generated_source_test \
  //examples/mnist:mnist_ast_source_map_test
```

The generated shadow source and source map are available through
`//examples/mnist:mnist_ast_source`. Full aliases, lexical shadowing,
pointer-qualified container types, and interprocedural return-type inference
still require Zig semantic type data rather than syntax alone.

For this narrow POC, add and multiply retain exact XLA metadata. The current XLA
lowering drops operation metadata for scalar constants and broadcasts, so those
links are recovered by following matching operand positions from the mapped
multiply through broadcast to constant. `mapping.json` records that recovery as
`dataflow_operand`, and packing fails if a surviving StableHLO operation is left
unresolved.
