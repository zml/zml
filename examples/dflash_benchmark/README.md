# DFlash Benchmark

Build the benchmark binary with:

```sh
bazel build //examples/dflash_benchmark:benchmark
```

Run it with a dataset path supplied at runtime:

```sh
bazel run //examples/dflash_benchmark:benchmark -- \
  --model=<dflash model path or hf://...> \
  --target-model=<target model path or hf://...> \
  --dataset=<math500|sharegpt|alpaca|swe_bench_lite|mt_bench> \
  --dataset-path=<path to dataset file or directory> \
  --split=<test|dev|train> \
  --samples=<n>
```

This package does not declare Bazel external dataset repositories. Dataset files are
expected to be available through the path passed on the command line.
