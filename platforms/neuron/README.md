# Neuron Backend Notes

Architecture and contract documentation for this package now lives inline with
the implementation:

- `libneuronxla.zig` documents the whole-program Neuron compile hook boundary.
- `nki/hlo_rewriter.zig` documents the embedded-kernel lowering boundary and the
  Zig/Python request-result contract used by `nki-cc`.
