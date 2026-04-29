# GatedDeltaNet NKI Kernel — Round 2 Optimization Followup

## TL;DR

| Stage | Wall avg | Speedup vs ZML ref (0.96 ms) |
|---|---|---|
| Round 1 final (items 1 + 3) | 2.56 ms | 0.367× |
| **+ Item A** (HBM output layout) | **2.36 ms** | **0.412×** |
| **+ Item E** (fused normalize, dropped decay copy) | **2.15 ms** | **0.454×** |
| Item D (DMA / compute pipeline) | — | rejected: compiler ICE |

Net round-2 improvement: **2.56 → 2.15 ms** (−16 %), bringing speedup
from 0.367× to **0.454×**. Implementation lives in
[examples/neuron_nki/gated_deltanet.py](examples/neuron_nki/gated_deltanet.py)
and the matching layout change in
[examples/neuron_nki/gated_deltanet.zig](examples/neuron_nki/gated_deltanet.zig).

## Methodology

Each accepted optimization was: (1) implemented in isolation, (2)
benchmarked end-to-end with `NEURON_RT_VISIBLE_CORES=1`, (3) profiled
under `NEURON_RT_INSPECT_*` and inspected with `neuron-explorer view
--output-format=summary-text`, and (4) kept only if both wall-clock
and profile metrics improved.

Build / run command:

```bash
NEURON_RT_VISIBLE_CORES=1 bazel run --config=remote \
  --@zml//platforms:neuron=true --@zml//platforms:cpu=false \
  //examples/neuron_nki:gated_deltanet
```

Profile command:

```bash
NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=/tmp/profX \
NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1 \
neuron-explorer inspect -o /tmp/profX/ -- <bazel run cmd>
neuron-explorer view -n <neff> -s <ntff> --output-format=summary-text
```

## Item A — HBM output layout `[b, s, vh, vd]`

### Change

- Reshape kernel output from `[b, s, num_q_heads, qk_rep, vd]` (which
  required a per-step `nc_transpose` + PSUM-to-SBUF `tensor_copy` +
  fragmented store into a `[head, value]` layout) to a single flat
  `[b, s, vh, vd]` HBM buffer where every `(qh, ir)` pair owns a
  contiguous `(1, value_dim)` row.
- Drop `out_t_psum` and `out_v_slot` SBUF scratch.
- Per-step output path becomes one `nc_matmul` into `out_psum`, one
  `tensor_copy` into `out_v_T`, then `qk_rep` direct
  `nl.store(output[..., vh:vh+1, :], out_v_T[..., ir*Dv:(ir+1)*Dv])`.
- Updated `outputShape` and `ReferenceProgram` rename in
  [gated_deltanet.zig](examples/neuron_nki/gated_deltanet.zig) to
  `{.b, .s, .vh, .vd}` so downstream graph matches by tag.

### Profile delta (post-A vs post-1+3)

| Metric | Before | After | Δ |
|---|---|---|---|
| `matmul_instruction_count` | 5120 | 2560 | −50 % |
| `sync_engine_instruction_count` | 2888 | 51 | −98 % |
| `dma_transfer_count` | 1050 | 375 | −64 % |
| `vector_engine_instruction_count` | 11 749 | 9 295 | −21 % |
| `total_active_time` | 2.45 ms | 1.91 ms | −22 % |
| Wall | 2.56 ms | 2.36 ms | −7.8 % |

The huge drop in `sync_engine_instruction_count` confirms the kernel
is no longer bottlenecked on per-store cross-engine synchronization.

## Item E — fused L2-normalize + drop decay-scalar copy

### Change

- Replace the unfused per-step
  `tensor_tensor(square) + tensor_reduce(sum)` pair on the Vector
  engine with a single
  `nisa.activation_reduce(op=nl.square, reduce_op=nl.add)` on the
  Scalar engine. The bias-fused `rsqrt` and the
  `nisa.tensor_scalar(op0=mul, op1=mul)` for `q` (folded with
  `1/sqrt(...)` and `scale`) and `k` are unchanged.
- Drop the previously needed `tensor_copy` that materialized
  `decay_scalar` from PSUM to SBUF; we now read straight from
  `decay_rows` via `nl.broadcast_to`.

### Profile delta (post-E vs post-A)

| Metric | Before | After | Δ |
|---|---|---|---|
| `vector_engine_active_time` | 1.67 ms | 1.42 ms | −15 % |
| `scalar_engine_instruction_count` | 588 | 1 836 | +212 % (work shifted) |
| `total_active_time` | 1.91 ms | 1.74 ms | −9 % |
| `throttle` | 92 % | 94 % | (tighter packing) |
| Wall | 2.36 ms | 2.15 ms | −9 % |

The work-shift pattern (Scalar instr count tripled while Vector
active-time dropped) is the expected signature of moving the
square-and-reduce from Vector to Scalar.

## Item D — software-pipelined paired q-head loop (rejected)

### Intent

Process q-heads two at a time and double SBUF preload tiles so that
the compiler may overlap pair-member B's bulk DMAs / normalize / Dk
transpose with pair-member A's sequential scan, exposing
DMA-vs-compute and Vector-vs-Scalar parallelism that the strictly
sequential `affine_range(num_q_heads)` schedule does not.

### Three implementations attempted

1. **Python `for i_q_head in range(num_q_heads)`** with parity-based
   selection of two nested-function-allocated buffer dicts.
2. **Paired `nl.affine_range(num_q_heads // 2)`** with module-level
   helpers `_preload_qh` / `_scan_qh` for clarity.
3. **Fully inlined paired body** (~1.2k lines, no closures, no nested
   defs, no Python control flow, no conditionals — only doubled SBUF
   tile count and a 2×-unrolled body).

### Result

All three failed in the Penguin frontend with the same internal
error:

```
File "neuronxcc/starfish/penguin/ModuleGen.py", line 184,
  in load_xla_function.read
KeyError: 'ir'
```

That (3) — pure linear code with no exotic constructs, only larger
in absolute size and SBUF allocation count — also ICEs strongly
suggests a Penguin frontend bug triggered by kernel source size or
allocation count, not a misuse on our side. Reverted to the post-E
state.

### Fallback ideas not yet tried

- **DMA-only doubling.** Double *only* `v_rows`, `g_rows`,
  `beta_rows`, `q_all`, `k_all` (the inputs) but reuse a single set
  of normalize / transpose / scan tiles. Tests whether DMA
  double-buffering alone is enough to win, without restructuring the
  compute pipeline.
- **Drop kernel size below the ICE threshold.** If the bug is
  size-driven, factoring the bulk transpose+scan body into an
  equivalent but tighter `affine_range`-based loop may compile.

Both fall out of scope for this round; recorded for round 3.

## Items rejected or subsumed

| Item | Status | Reason |
|---|---|---|
| 2 — combined `pred_v` + `pred_q` matmul, stationary `(Dk, 2)` | rejected | Same `KeyError: 'ir'` ICE class as item D |
| B — PE-array transpose mode for q/k | rejected | Incompatible with rank-1 update's required `(1, Dk)` partition-0 stationary operand |
| C — drop redundant per-step transpose | subsumed | Folded into item A |
| F | not attempted | Out of scope |
| G | not attempted | Out of scope |

## Final state

- File: [examples/neuron_nki/gated_deltanet.py](examples/neuron_nki/gated_deltanet.py)
- Wall-clock: **2.15 ms / iter** (was 2.56 ms at start of round 2;
  baseline was ~2.9 ms).
- Speedup vs ZML reference: **0.454×** (was 0.367×).
- Numerics: pass at `absolute_tolerance=5e-3`, `relative_tolerance=5e-2`.

## Suggested next round (items for round 3)

In rough order of expected impact / risk ratio:

1. **DMA-only double-buffer** (the item D fallback above). Smallest
   structural change; tests the pipelining hypothesis without
   triggering the Penguin ICE.
2. **fp16 / bf16 storage path for q / k / v / g / beta in SBUF**
   (compute still fp32 in PSUM). Halves DMA bytes and SBUF residency
   for the bulk preload tiles. Risk: numerics drift on rsqrt path.
3. **Multi-core sharding across `num_q_heads`.** The kernel's
   q-head loop is fully independent — `nl.spmd_kernel` over
   `inf2.8xlarge`'s 2 NeuronCores would give a near-2× speedup at
   the example level. (Currently we pin `NEURON_RT_VISIBLE_CORES=1`
   for benchmarking parity; lifting that with proper sharding would
   put the kernel ahead of the ZML reference.)
4. **Reduce `num_q_heads` loop overhead.** Profile shows ~250 µs of
   the 2.15 ms is loop preamble / state init from `h0`. A
   `load_transpose2d` of all `qk_rep` slots in one bulk op would
   trim this.
