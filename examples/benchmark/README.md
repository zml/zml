# Projection benchmark

`--operation=projection --rows=M --size=K --dtype=bf16` computes
`[M,K] @ [K,K]^T` with physically row-major projection weights. This is the
contracting-dimensions `{1},{1}` form used by Llama. The default `matmul`
benchmark remains unchanged.

For the local Furiosa2 checkout:

```sh
bazel build //examples/benchmark --@zml//platforms:furiosa2=true
XLA_FURIOSA2_PJRT_LIBRARY=/home/steeve/xla-private/bazel-bin/xla/pjrt/furiosa2/libpjrt_c_api_furiosa2_plugin.so \
XLA_FURIOSA2_COMPILER=/home/steeve/xla-private/xla/pjrt/furiosa2/compile_visa.sh \
XLA_FURIOSA_RUNTIME_LIBRARY=/home/steeve/.local/share/xla-rngd-sdk/libdevice_runtime.so \
XLA_FURIOSA_COMPILER=/nonexistent/tcl-compiler \
  bazel-bin/examples/benchmark/benchmark \
    --operation=projection --rows=128 --size=4096 --dtype=bf16 --iterations=100
```

The measurement warms up once, then awaits each execution. It includes PJRT
dispatch, completion, and output-buffer lifecycle, but excludes compilation,
input upload, and the final correctness check. The existing input generator
fills each float buffer with a single random value. Projection checks this
premise, computes the analytical BF16 reference, and checks every output.
Varied dense operands are covered separately by the plugin hardware tests.
This is a projection microbenchmark, not Llama generation tok/s.

Validation on RNGD pe0-3 with the vISA backend (2026-09-27): build and Zig
format checks passed; all twelve projection runs checked every result. Three
warm repeats for N=K=4096 measured 130.744–136.332 us at M=1 (1000 iterations),
5070.903–5092.203 us at M=128 (100 iterations), and
157944.292–158161.051 us at M=4096 (10 iterations). The initial emitter is
unoptimized; its prefill/square throughput is only 0.84–0.87 TFLOP/s.
Detailed SDK failures, corrections, and logs live in the companion XLA
`xla/pjrt/furiosa2/experiments/2026-09-27-projection` directory.
