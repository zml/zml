# Tenstorrent tensor kernels

`//mlir/dialects/tenstorrent` registers TT-MLIR's TTIR dialect in ZML's MLIR
context. `//kernels/tenstorrent:builder` builds and verifies pure tensor programs.
It is separate from `//mlir/dialects/ttir`, which is Triton.

Use `zml.kernel.tenstorrent.Kernel(Config, spec)` with named input and output
fields, as with the other kernel builders. The `run` callback receives a builder
whose arguments and result types come from the call's tensor shapes. Obtain
inputs with `b.arg(index)`, construct tensor operations, and end with
`b.returns(&.{ ... })`. `Kernel.call(inputs, output_shapes, config)` verifies and
serializes the function into the `__tt$ttmlir` custom call. No launch grid or device
pointer arguments are needed. The XLA plugin imports TTIR before its layout and
optimization passes, so the payload participates in the surrounding graph.

`b.attention(q, k, v, mask, options)` emits FlashAttention-2 with tensors in
`[B,H,S,D]` order. A missing mask and `is_causal=true` select square causal
prefill; an optional `[1|B,1|Hq,Sq,Sk]` mask requires `is_causal=false`. An explicit
scale supports padded head widths. `b.attentionDecode` provides the specialized
one-query path. The public attention wrapper constructs offset-aware masks,
pads tiles, and restores tagged layouts according to its shared contract.

The payload must remain pure and single-block. Buffer donation is expressed on
the surrounding PJRT executable, rather than through custom-call operand aliases.

`gated_delta_net.zig` implements the recurrence used by
`zml.nn.GatedDeltaNet.forward` on Tenstorrent. It computes in f32, uses batched
matrix-vector products and an outer-product state update, and masks inactive
positions to preserve state and produce zero outputs. ZML normalizes tagged
batched/unbatched tensor layouts before the call and restores the original tags
and dtypes afterward.

For head-partitioned execution across cards, the wrapper builds TTIR inside a
local manual computation. Queries, keys, values, gates, and state share the
head partition; the active length is replicated. This makes static slices and
reshapes use the local head count. Emitting a global 48-head payload before
partitioning previously left slice bounds at 48 after inputs became 24 heads
per card, causing compilation to fail. Computed inputs with open partitioning
resolve the model axis before entering the local computation. Donation stays
on the enclosing executable.

Sequence positions are currently unrolled into TTIR. This is a functional
prefill/decode implementation, not a chunk-parallel prefill algorithm or a
single fused Tensix kernel. Large-prefill compilation and kernel fusion remain
performance work.

Validation:

- `//mlir/dialects/tenstorrent:test`: registered dialect verification.
- `//kernels/tenstorrent:test`: builder and recurrent payload round trips.
- `//zml:tenstorrent_test`: native execution against an independent scalar
  recurrence, with one/three tokens, one/two batches, and negative, zero, partial,
  full, and oversized active lengths. State and outputs use 1e-4 absolute and
  relative tolerances.

`//zml:tenstorrent_gated_delta_net_test` runs the recurrence tests independently
of attention. Additional Qwen-sized cases use 48 heads of width 128 with BF16
queries/keys/values and FP32 gates/state, on one and two P150a cards. One-token
decode and three-token prefill each perform six donated state updates with
negative, zero, partial, full, and oversized active lengths. Every state element
is compared with an independent f64 scalar recurrence; BF16 outputs, retained
outputs, and unchanged query/key/value/gate inputs are checked as well. Tests
verify the executable's state-donation index and pass with tracing enabled and
disabled. The two-card three-token operation trace contains no all-gather,
all-reduce, or reduce-scatter operations. These kernel tests do not establish
full Qwen 27B weight loading, generation correctness, or model throughput.
