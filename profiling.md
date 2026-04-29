# GatedDeltaNet NKI Kernel — Profile Analysis

## Methodology

```bash
bazel run --config=remote --@zml//platforms:neuron=true \
  --run_under="NEURON_RT_INSPECT_ENABLE=1 \
    NEURON_RT_INSPECT_OUTPUT_DIR=/home/kevin/profiling \
    NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_FRAMEWORK_DEBUG=1 \
    NEURON_RT_INSPECT_DEVICE_PROFILE=1 XLA_HLO_DEBUG=1 XLA_IR_DEBUG=1 \
    neuron-explorer inspect -o /home/kevin/profiling/ -- " \
  --@zml//platforms:cpu=false //examples/neuron_nki:gated_deltanet

neuron-explorer view \
  -n neff_<id>.neff -s <id>_instid_0_vnc_0.ntff \
  --output-format=summary-text
```

Two NEFFs are produced and both were profiled:

- `neff_998636308723560.neff` → **NKI kernel** (Program.forward)
- `neff_173232423153411.neff` → **ZML reference** (`zml.nn.GatedDeltaNet.forward`)

Wall-clock benchmark from the same run:

| Build | avg / iter |
|---|---|
| ZML nn reference | **0.95 ms** |
| Neuron NKI | **2.95 ms** |
| **Speedup** | **0.32×** |

## Side-by-side hardware metrics

> All metrics are for ONE invocation of the NEFF.

| Metric | NKI | Reference | NKI / Ref |
|---|---|---|---|
| `total_active_time` | 2.45 ms | **0.83 ms** | **2.95×** |
| `total_active_time_percent` | 87.5 % | 30.9 % | — |
| `tensor_engine_instruction_count` | **10 264** | 4 592 | **2.23×** |
| `matmul_instruction_count` | **5 120** | 2 325 | **2.20×** |
| `tensor_engine_active_time` | 950 µs | 445 µs | 2.14× |
| `tensor_engine_active_time_percent` | 34 % | 16.6 % | — |
| `vector_engine_instruction_count` | **11 749** | 1 000 | **11.7×** |
| `vector_engine_active_time` | 1.89 ms | 648 µs | 2.92× |
| `vector_engine_active_time_percent` | **67.7 %** | 24.1 % | — |
| `scalar_engine_instruction_count` | 620 | 258 | 2.4× |
| `gpsimd_engine_instruction_count` | 2 068 | 86 | 24× |
| `event_count` | **46 811** | 3 337 | **14×** |
| `trace_count` | 127 345 | 11 806 | 10.8× |
| `sync_engine_instruction_count` | **2 888** | **20** | **144×** |
| `psum_read_sbuf_write_count` | 800 | 162 | 4.9× |
| `psum_write_bytes` | 1.58 MB | 1.42 MB | 1.11× |
| `dma_transfer_count` | **1 050** | **2** | **525×** |
| `software_dynamic_dma_packet_count` | **32 768** | 2 416 | **13.6×** |
| `hbm_read_bytes` | 2.36 MB | 3.26 MB | 0.72× |
| `hbm_write_bytes` | 2.36 MB | 2.38 MB | ≈ |
| `transpose_flops` | 262 144 | **815 M** | 0.0003× |
| `adjusted_hardware_flops` | 202 M | **983 M** | 0.21× |
| `mm_arithmetic_intensity` | 10.66 | 7.46 | — |
| `throttle_avg_util_limit_nc0_percent` | **91.4 %** | 28.5 % | — |
| `mfu_max_achievable_estimated_percent` | 4.79 % | 2.66 % | — |

(`mfu_estimated_percent` is < 0.001% in both cases — this workload is
nowhere near compute-bound on Inferentia2. Peak FP32 PE-array
throughput is ≈ 13 TFLOPS; we use < 100 GFLOPS.)

## What the numbers say

### Finding 1 — Per-instruction overhead is the bottleneck

The PE-array does **2.2× more matmul instructions** in the NKI
build (5120 vs 2325) but only **0.21× the total FLOPs** (202 M vs
983 M). That means each NKI matmul carries on average

```
202 M / 5120 ≈ 39 KFLOPs / instruction
```

vs

```
983 M / 2325 ≈ 423 KFLOPs / instruction
```

— the reference packs **10× more useful arithmetic** into each PE
launch. NeuronCore-v2 has a 128×128 PE array, so a *peak* matmul
launch executes 32 KFLOPs/cycle. At 39 KFLOPs/instr the NKI
matmuls finish their useful work in ≈ 1 cycle and then spend the
rest of the launch on stationary fill / drain / sync.

### Finding 2 — Sync engine is being asked to do 144× more

`sync_engine_instruction_count = 2888` (NKI) vs `20` (reference).
The sync engine emits one instruction per inter-engine fence:
PSUM → SBUF copy completes → next matmul into same bank can start,
DMA done → activation can read the tile, etc. The reference
achieves **20 syncs total** by issuing one giant matmul whose
outputs are streamed through the PE-array pipeline; NKI emits
thousands of fine-grained syncs because each `nisa.*` call is its
own MLIR op with explicit dataflow.

### Finding 3 — DMA is fragmented 525×

`dma_transfer_count = 1050` (NKI) vs `2` (reference). Even with
the "bulk preload" of v/g/beta in the current kernel, the compiler
is materializing the `nl.affine_range(seq_len) × nl.affine_range(qk_rep)`
DMA loop as 16 × 2 = 32 separate transfers per q-head × 16 q-heads
= 512 v-DMAs alone, plus q/k/g/β. Each transfer carries ~2.5 KB.
`software_dynamic_dma_packet_count = 32 768` confirms the
descriptor count.

The reference does **2 transfers** of ~50 KB each (the entire q+k
tensor and the entire v+α+β tensor) and lets the on-chip Vector
engine slice them up.

### Finding 4 — Vector engine is over-active (67% NKI vs 24% ref)

Vector engine instruction count: **11 749** (NKI) vs **1 000**
(reference) — **11.7×** more vector ops. These are the chains of
`tensor_tensor`, `tensor_scalar`, `tensor_reduce`, `activation`
(rsqrt / exp), `nc_transpose` (vector-engine for small tiles)
that implement the L2-normalize, decay, diff and per-slot output
transposes. The reference fuses all of them — the kernel does not.

### Finding 5 — Throttling is the dominant wall-time component

`throttle_avg_util_limit_nc0_percent = 91.4 %` for the NKI build
(28.5 % for reference). The Neuron runtime reports the kernel as
spending **91.4 % of its execution time waiting** for some
per-instruction throttle limit to lift — not because it's
overheating, but because *the SchedulerSyncEngine is back-pressuring
on PSUM availability and inter-engine sync*. With 5120 matmuls
sharing 16 PSUM banks and 2888 explicit syncs, the PE array
finishes a matmul, must drain to PSUM, must be copied or consumed,
then the bank can be reused. That round-trip dominates.

### Finding 6 — Transposes via PE-array vs vector engine

`transpose_flops`: NKI = 0.26 M, reference = **815 M**. The
reference uses `nc_matmul(is_transpose=True)` mode which runs the
transpose on the PE-array as part of a larger matmul flow, getting
counted as transpose-flops at PE-array rates. NKI does small
isolated `nc_transpose` calls on the Vector engine, generating
near-zero FLOP counters but burning vector-engine time.

### Finding 7 — Memory bandwidth is fine

`hbm_read_bytes` 2.36 MB (NKI) ≈ 3.26 MB (reference). At HBM peak
of ~440 GB/s, both load all weights in well under 10 µs. The
problem is not bandwidth — `mbu_estimated_percent` < 0.5 % for
both. **HBM is not the bottleneck.**

## Diagnosis

The kernel is **per-instruction-issue bound**, not memory bound,
not arithmetic bound. The Neuron runtime spends 91 % of its time
on instruction sequencing / inter-engine sync, churning through
**~22 000 instructions across PE + Vector + Scalar + GpSimd
engines**, vs the reference's ~7 000.

The reference wins because XLA → neuronx-cc:

1. **Fuses the 16 q-heads** into wider PE-array launches (each
   matmul does ~10× more arithmetic per instruction issue).
2. **Generates 2 huge DMAs** instead of 1050 small ones.
3. **Folds the L2-normalize / decay / delta math** directly into
   the matmul bias / scale operands, eliminating thousands of
   vector-engine ops.
4. **Uses the PE-array transpose mode** rather than vector-engine
   transposes, freeing the vector engine for elementwise math
   *concurrent with* PE work.
5. **Emits ~20 cross-engine syncs** for the entire kernel instead
   of 2888 — its dataflow stays inside dedicated execution modules
   (the XLA `while` loop body lowers to a tight BIR loop with
   per-iteration register-resident state).

## Concrete optimization targets, ranked by expected payoff

These are *NKI-level* changes — for an XLA-level rewrite see
[followup.md](followup.md).

### 1. Collapse per-vh DMAs into one DMA per (batch, q-head) — **easy, ~30 % expected**

Today's preload:

```python
for it in nl.affine_range(seq_len):
    for ir in nl.affine_range(qk_rep):
        nisa.dma_copy(dst=v_rows[..., slice(...)], src=v[i_batch, it, vh:vh+1, :])
```

→ 32 768 DMA descriptors. Replace with one `nl.load` of the whole
`v[i_batch, :, vh_start:vh_end, :]` slab per q-head. Same for q,
k, g, β. The HBM source is contiguous in the right axes, so the
hardware DMA engine should produce ~16 transfers (one per q-head)
instead of 1050.

### 2. Combine the predicted_v + output matmuls — **medium, ~15 % expected**

Today both `predicted_v = k_kp.T @ state` and `out = q_kp.T @ state`
launch their own matmul, each with stationary `(Dk, 1)`. Pack them
into one `nc_matmul` with stationary `(Dk, 2)` (free-dim cap is
128, so well within budget), output PSUM tile `(2, pack_w)`, then
slice rows. Halves the predicted_v / output matmul count from 512
to 256 (=16 q-heads × 16 timesteps).

### 3. Increase `v_chunk` from 64 to 128 (or 256) — **easy, ~10 % expected**

Currently the rank-1 update runs 4 chunks of `v_chunk=64`. PSUM
budget for one chunk: `128 × 64 × 4 B = 32 KB` — the whole PSUM.
But if we move predicted_v / output / out_t_psum to **smaller**
PSUM tiles freed by the new matmul layout (item 2), or if we drop
unused PSUM tiles, we may be able to fit `v_chunk=128`, halving
the update matmul count from 4 to 2 per timestep (256 → 128).

### 4. Replace separate `tensor_tensor` + `tensor_reduce` with `activation` fused-form — **easy, ~5 % expected**

Today's L2-normalize hoisted into preload:

```
tensor_tensor(q_sq = q_row * q_row)
tensor_reduce(q_sumsq = sum(q_sq))
activation(q_inv = rsqrt(q_sumsq + eps))
tensor_scalar(q_scaled = q_row * q_inv * scale)
```

→ 4 vector-engine ops × 16 timesteps × 2 (q & k) = 128 vector ops.
`nisa.activation` can take `data * data` and a reduce in a single
fused launch on Scalar engine. Cuts the per-q-head normalize cost
in half.

### 5. Batch the per-slot output `nc_transpose` — **easy, ~3 % expected**

Today: 2 small `nc_transpose(1×128 → 128×1)` per timestep × 256 =
512 transposes on the Vector engine. Pack the qk_rep slots into a
single `(qk_rep, Dv)` SBUF tile and transpose in one shot to
`(Dv, qk_rep)`, then DMA-store with stride. Cuts 512 to 256.

### 6. Move sync-heavy decay into the matmul bias — **medium, ~5 % expected**

Today `state *= exp(g)` is a separate `tensor_scalar` per packed
slot. `nisa.nc_matmul` accepts an `accumulate` mode and a bias
broadcast. If we restructure `predicted_v = k_kp.T @ (decay ⊙
state)` so the decay multiplies the state on its way through
the matmul (via `tensor_scalar` fused into the `state` read), we
collapse 32 vector ops into 0 per timestep.

### 7. Multi-buffered q-heads (pipeline) — **harder, ~25 % if successful**

Allocate 2 sets of bulk-preload tiles. While q-head N's scan runs,
DMA-preload q-head N+1. The compiler's instruction scheduler will
overlap the two only if the data dependencies are clean. Empirical
upside: ~25 % wall-time reduction (the preload phase is ~30 % of
the per-q-head time according to the trace).

## Expected best-case after all NKI-level optimizations

Best case (items 1-6 stacked, optimistic):

```
2.95 ms × 0.70 (item 1) × 0.85 (item 2) × 0.90 (item 3)
       × 0.95 (item 4) × 0.97 (item 5) × 0.95 (item 6)
≈ 1.45 ms / iter
```

That is `~1.5×` the reference (still 0.65×), assuming none of the
optimizations interact badly. Item 7 could push us under
1 ms / iter, but the success is contingent on the compiler's
scheduling choices.

**To actually beat the reference at this exact shape (S=16,
single-core), the algorithmic structure has to change.** See
[followup.md](followup.md) for the discussion of chunkwise
DeltaNet and multi-core sharding.

## Raw profile artifacts

- `/home/kevin/profiling/i-017d54c80d8411272_pid_859537/`
  - `neff_998636308723560.neff` — NKI kernel
  - `998636308723560_instid_0_vnc_0.ntff` — NKI device profile
  - `neff_173232423153411.neff` — reference
  - `173232423153411_instid_0_vnc_0.ntff` — reference device
    profile
  - `ntrace.pb`, `trace_info.pb` — full event trace (load with
    `--output-format=perfetto` for visual inspection)
