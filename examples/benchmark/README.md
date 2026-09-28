# Projection benchmark

`--operation=projection --rows=M --size=K --dtype=bf16` computes
`[M,K] @ [K,K]^T` with physically row-major projection weights. This is the
contracting-dimensions `{1},{1}` form used by Llama. The default `matmul`
benchmark remains unchanged.

Prepare the override and SDK using the [Furiosa guide](../../platforms/furiosa/README.md).
The benchmark uses the default four-PE client:

```sh
bazel run --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  //examples/benchmark -- \
  --operation=projection --rows=128 --size=4096 --dtype=bf16 --iterations=100
```

The measurement warms up once, then awaits each execution. It includes PJRT
dispatch, completion, and output-buffer lifecycle, but excludes compilation,
input upload, and the final correctness check. The existing input generator
fills each float buffer with a single random value. Projection checks this
premise, computes the analytical BF16 reference, and checks every output.
Varied dense operands are covered separately by the plugin hardware tests.
This is a projection microbenchmark, not Llama generation tok/s.
