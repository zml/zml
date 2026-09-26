# Llama experiment evidence

These are captured logs, not golden files or tests run automatically by Bazel.
The model checkpoint and generated executables are not included. Commands,
acceptance criteria and interpretation are in [WHOLE_FORWARD.md](../WHOLE_FORWARD.md).

| Log | Why the experiment was run | Outcome |
|---|---|---|
| `packed-cpu-comparison.log` | Compare the former packed whole-forward graph to the independently composed reference on CPU, isolating graph construction from the Furiosa backend | Pass: exact argmax, KV tolerance and unchanged cache bits; includes repeated libunwind warnings |
| `packed-decode-benchmark.log` | Measure the former packed single-executable decode and then compare it against CPU | 52.93/52.73/52.81 tok/s; **failed** exact argmax, 323 versus 311; not a validated generation rate |
| `packed-decode-profile.log` | Determine whether the packed program's slowdown was host submission or native execution | Device wait dominates; profiling run is separate from the unprofiled throughput measurement |
| `diagnostic-build.log` | Compile revised full-forward diagnostics that retain KV checks after argmax failure | Build passed; failure-reporting execution not yet verified at this commit |

The packed implementation was removed in `bad504f`. These historical logs explain
why packing removal and better failure diagnostics were investigated; they do
not establish the correctness or speed of the current separate-weight graph.
The separate-weight results are recorded in `separate-*.log` and the corresponding
section of `WHOLE_FORWARD.md`: position 0 passes, position 127 fails, and the
unprofiled median decode rate is provisionally 62.34 tok/s.

The XLA repository records bridge/11 implementation details, experiments and
hardware evidence in `xla/stream_executor/furiosa/opt_runtime/INDIRECT_ARGUMENTS.md`
and its adjacent `testdata/indirect` directory (commit `18826ff670`).
