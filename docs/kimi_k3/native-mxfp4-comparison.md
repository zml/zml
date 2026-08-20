# Kimi K3 native grouped MXFP4 comparison

This document is the line-by-line comparison map for Milestone 17. All source
revisions are immutable in `docs/kimi_k3/revisions.lock.json`.

## Revisions and retained code

- Moonshot Kimi K3: `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`.
- Grouped A16W4 donor: `brabier/glm5.2` at
  `b4f0af76e4c464c0f533420b94fdb1fba838c5e3`, Apache-2.0.
- Retained donor kernel: `zml/moe/triton_kernels/a16w4_kernel.zig`, SHA-256
  `735cf28078063e86978530a3ca5909302cea8853f4f851736a4319297c6c3d93`.
- Kimi orchestration: `examples/llm/models/kimi_k3/grouped_mxfp4.zig`.

The retained kernel is byte-identical to the locked donor. Kimi routing,
route-order restoration, invalid-route masking, SiTU, and route-weighted
reduction are implemented outside that kernel.

## Semantic map

| Operation | Pinned Moonshot/reference behavior | ZML implementation | Validation |
| --- | --- | --- | --- |
| Expert grouping | Flatten top-k IDs, argsort, run contiguous token groups, restore with `new_x[idxs] = outs` in `modeling_kimi_linear.py` | StableHLO sort/histogram/offset/block map in `grouped_mxfp4.zig`; scatter restores original route order | Official 64-route fixture; duplicate and empty expert cases |
| Packed E2M1 | Low nibble then high nibble, sign bit 3, magnitudes `[0,.5,1,1.5,2,3,4,6]` in `export_moe_reference.py` | Triton `dotScaledOpts(... .e2m1 ...)` reads two values per byte | Primitive, partial-K, and 183 real matrix paths |
| E8M0 scale | `exp2(uint8(scale) - 127)`, repeated for 32 K values | Native E8M0 operand passed directly to scaled dot, block size 32 | Primitive and real checkpoint fixture parity |
| Weight layout | Checkpoint `[expert, out, packed_k]`, scale `[expert, out, k/32]` | Kernel strides use packed K as the minor dimension and never dequantize globally | Donor SHA check and real w1/w2/w3 execution |
| w1/w3 | Per-routed-expert linear projections | Two unfused grouped native linears | 61 selected experts, 64 routes |
| SiTU | `4*tanh(gate/4)*sigmoid(gate) * 25*tanh(up/25)` | `primitives.situGlu` after native w1/w3 | Official route outputs and every layer-family activation |
| w2 | Per-route down projection after SiTU | Route-input grouped native linear | Token-input/route-input edge suite and official fixture |
| Route weights | Multiply restored routes, then sum route axis | FP32 multiply/reduction in `moe.forward` | Synthetic weighted reduction and official combined latent |
| Invalid/non-local ID | Not executed by that expert partition | One-past-local-end sentinel, masked to exact zero | Exact sentinel and two logical partition tests |

## Reproduce the comparison

From `/dev/shm/kimi-k3`:

```bash
./scripts/reproduce.sh 17
```

The runner is CUDA-only, uses `/dev/shm/kimi-k3/.venv`, keeps Hugging Face
offline, reads only `/dev/shm/kimi-k3/moonshot/kimi-k3`, and does not download
model files. Its primary machine-readable output is
`artifacts/reports/milestone-17-comparisons.json`.

For an individual activation-debug run:

```bash
cd /dev/shm/kimi-k3/zml
bazel-bin/examples/llm/kimi_k3_grouped_mxfp4_tests \
  --fixture=/dev/shm/kimi-k3/artifacts/fixtures/milestone-5/primitive-reference.safetensors
```

Each case prints shape, max/mean absolute error, RMSE, close fraction, and
percentile errors. The selected-expert and layer-family logs add official
Moonshot boundary parity and synchronized execution time.

## Performance interpretation

The accepted H100 budget is at least 2x over the slow selected-expert ZML
oracle under the same 61-expert/64-route/13-boundary scope. The retained
`split_k=1` configuration measured 12,652 microseconds warm versus 37,909
microseconds for the immutable slow baseline (3.00x). A donor-style
`split_k=4` experiment measured 19,704 microseconds and was rejected.

HBM comparison is exact static temporary-tensor accounting rather than a
process-wide allocator watermark: the slow path may materialize 5,637,144,576
bytes of route-expanded FP32 gate/up weights, while the native path's largest
route output is 458,752 bytes and weights remain in their packed shared form.
Persistent input weights and allocator slack are excluded equally.
