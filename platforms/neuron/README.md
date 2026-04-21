# Neuron Backend Notes

`platforms/neuron` contains the PJRT loader plus the compile hook that translates
ZML HLO into Neuron-compatible custom calls.

## Compile flow

`libneuronxla.so` is loaded by the Neuron Python stack and exposes
`neuronx_cc(...)`.

1. ZML emits HLO containing either:
   - regular StableHLO lowered for Neuron, or
   - synthetic `zml$neuron$nki` custom-calls for inline Python NKI kernels.
2. `platforms/neuron/nki/` owns embedded custom-kernel materialization.
   `nki/hlo_rewriter.zig` rewrites synthetic `zml$neuron$nki` calls into
   `AwsNeuronCustomNativeKernel` calls by invoking `nki-cc`, and
   `nki/zml_compiler.py` is the Python-side kernel compiler shim used by that
   path.
3. `libneuronxla.zig` logs the Python boundary arguments and owns the whole
   StableHLO program compile: target resolution, temp workspace lifecycle,
   `neuronx-cc` invocation, and collection of the produced NEFF.
4. `libneuronxla.zig` wraps the produced NEFF back into an `AwsNeuronNeff`
   custom-call so the outer runtime sees a normal Neuron executable.

## Target And Verbosity

The compile hook no longer uses environment-variable overrides for target or
compiler verbosity.

- Target selection prefers the explicit target hint supplied by the Neuron
  Python bridge.
- If that hint is absent, the hook falls back to the Neuron platform version
  (`1.0 -> inf1`, `2.0 -> trn1`, `3.0 -> trn2`).
- `neuronx-cc --verbose` and `--logfile-verbose` are derived from
  `std.options.log_level`:
  - `.debug -> debug`
  - `.info -> info`
  - `.warn -> warning`
  - `.err -> error`
