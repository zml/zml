# Neuron Backend Notes

Architecture and contract documentation for this package now lives inline with
the implementation:

- `libneuronxla.zig` documents the whole-program Neuron compile hook boundary.
- `whole_program_compiler.zig` documents the StableHLO -> NEFF ->
  `AwsNeuronNeff` flow.
- `nki/embedded_kernel_rewriter.zig` documents the embedded-kernel lowering
  boundary and points to the split NKI helper modules.
