# GatedDeltaNet NKI Kernel — Architectural Analysis

## Status

[examples/neuron_nki/gated_deltanet.py](examples/neuron_nki/gated_deltanet.py)
has been rewritten. The kernel:

- **Compiles cleanly** (`Compiler status PASS`, ~3 s).
- **Is numerically correct** — passes `zml.testing.expectClose`
  against both the host-side ground truth and the
  `zml.nn.GatedDeltaNet` reference at `absolute_tolerance=5e-3`,
  `relative_tolerance=5e-2`.
- **Stable runtime:** ~2.8 ms / iteration on `inf2.8xlarge` with
  `NEURON_RT_VISIBLE_CORES=1`.
- **Does not** beat the `ReferenceProgram` (XLA-lowered
  `zml.nn.GatedDeltaNet`), which clocks ~0.92 ms / iteration.

The current speedup vs the reference is `~0.33×`. This document
explains, with reference to the Neuron programming model, why
beating the XLA-lowered reference for this particular kernel shape
is practically impossible from hand-written NKI on a single
NeuronCore-v2.

## Workload geometry

From [examples/neuron_nki/gated_deltanet.zig](examples/neuron_nki/gated_deltanet.zig)
(Qwen3.5-9B layer):

| Dim | Value |
|---|---|
| `batch_size` | 1 |
| `seq_len` (S) | 16 |
| `num_q_heads` (Hq) | 16 |
| `num_value_heads` (Hv) | 32 |
| `qk_rep = Hv / Hq` | 2 |
| `key_dim` (Dk) | 128 |
| `value_dim` (Dv) | 128 |

Per `(batch, q_head, t)` step the recurrence requires three
contractions (`predicted_v`, rank-1 update, `output`) plus a
normalize / decay / delta chain.

## Hardware: NeuronCore-v2 (Inferentia2 / Trainium-1)

Relevant specs (see the [Neuron programming
model](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/_sources/nki/programming_model.rst)):

- **PE array:** `128 × 128` systolic array. Contraction axis ≡
  partition dim ≡ `pmax = 128`.
- **SBUF:** 24 MB. 128-partition × free-dim tiles. `tensor_*` ops
  require operand partition offsets to match.
- **PSUM:** 16 banks × 2 KB = **32 KB total** PSUM per core. Matmul
  results MUST land in PSUM and be copied to SBUF before the next
  matmul into the same bank.
- **Engines:** Tensor (PE), Vector, Scalar, GpSimd, DMA. Independent
  engines can run in parallel — but only across **independent** ops.

## Why the reference is fast

`ReferenceProgram` lowers `zml.nn.GatedDeltaNet` through XLA →
`neuronx-cc`. The Neuron compiler:

1. **Cross-head parallelism via partition packing.** With Hq=16 ≤
   pmax=128, the compiler can pack independent q-heads along the
   partition dim of intermediate tiles. The 16 `predicted_v` /
   `output` contractions (independent across heads at fixed `t`)
   issue as a small number of fat matmuls instead of 16 narrow ones.
2. **Multi-engine scheduling.** XLA's HLO + `neuronx-cc` has full
   visibility into dataflow and exploits Vector / Scalar / GpSimd
   in parallel with the PE array. The NKI frontend exposes a much
   thinner abstraction — each `nisa.*` call is largely a one-to-one
   instruction emission and the compiler does not re-fuse across
   them.
3. **Optimized scan lowering.** XLA `while` bodies are recognized
   and lowered to dedicated dynamic-loop control on Neuron, with
   bounded-state SBUF allocation. NKI's `nl.sequential_range` is
   closer to a literal sequential schedule.

## Why an NKI kernel struggles to match it (here)

### 1. Sequential recurrence on `seq_len`

`state[t]` depends on `state[t-1]` through `predicted_v`,
`delta_v`, and the rank-1 update. There is **no associative scan
form** of the gated-delta rule that fits `nisa.tensor_tensor_scan`
(which the Mamba samples use to great effect — Mamba's recurrence
*is* associative). The 16 timesteps must serialize.

### 2. Cross-q-head parallelism is gated by PSUM capacity

The natural "wider matmul" trick — pack all `Hq` q-heads into the
free dim of a single state tile — runs into the rank-1 update:

- `state += k_n ⊗ delta_v` over `(Dk=128, pack_w)` requires a PSUM
  matmul tile of shape `(Dk, pack_w_chunk)`.
- For a single q-head, `pack_w = qk_rep * Dv = 256`. Even at
  `v_chunk=64` the update PSUM tile is `128 × 64 × 4 B = 32 KB` —
  **the entire PSUM**.
- Packing all 16 q-heads into one state would need
  `pack_w_full = 4096`, i.e. 16× more PSUM, which is physically
  impossible.

This is the architectural pinch point: the rank-1 update must run
PSUM-chunked, and chunking serializes within a step. The other two
contractions per step (`predicted_v` / `output`) could in principle
parallelize across heads, but they feed into / consume from `state`
which is mutated by the rank-1 update, so they sit on the same
critical path inside a step.

### 3. Op-count overhead per step

Each `nisa.*` call carries instruction-issue + sync cost. The
unavoidable per-step minimum (after every optimization the NKI
abstraction allows us) is ~25 ops:

- 4 matmul setup × 3 contractions = 12 (memset + nc_matmul +
  tensor_copy + occasional transpose)
- 4 chunks of update × 3 ops = 12
- 2 nc_transpose for q/k `(Dk,1)` form
- 2 decay tensor_scalar
- 1 diff + 2 delta_v scalars + per-rep output transposes/stores

At 16 q-heads × 16 timesteps × ~25 ops × ~0.5 µs amortized issue =
**~3.2 ms**, which matches the observed ~2.8 ms. The reference, by
running fewer logical ops (matmul fusion across heads, fewer
memset/copy round-trips) and using all engines, achieves ~3.5×
better utilization on this small problem.

### 4. Small-problem launch overhead dominates

With S=16 the kernel is short. A large fraction of wall time is
PJRT launch + DMA setup — both invariant of how clever the NKI body
is. The reference benefits from the same overhead but completes the
compute portion much faster, so its overhead amortizes better.

## Optimizations that *are* in the current kernel

- **Resident packed state**: `(Dk, qk_rep * Dv)` tile lives in SBUF
  for the full sequence; `qk_rep`-many heads' contractions fold
  into one wider matmul.
- **Bulk DMA preload** of `v` / `g` / `beta` for the whole sequence
  per `(batch, q_head)` — eliminates `O(qk_rep)` DMAs per step.
- **Hoisted q / k normalize.** L2-normalize + scale runs once per
  timestep at preload, not in the scan; the scan only does a
  `(1, Dk) → (Dk, 1)` transpose.
- **Pre-computed `decay = exp(g)`** for the full sequence in one
  `nisa.activation` call.
- **PSUM-friendly `v_chunk = 64`**, matching the proven Mamba-style
  PSUM budget.
- **Function-scope scratch** so PSUM/SBUF banks are pinned for the
  whole kernel — keeps compile time bounded (~3 s instead of the
  multi-minute compile observed with per-step `nl.ndarray` allocs
  in earlier drafts).
- **Structured loops only.** `nl.sequential_range` for the timestep
  scan, `nl.affine_range` everywhere else. No Python-level unroll,
  so MLIR / LLVM IR size stays compact.

## Optimizations that were tried and rejected

| Idea | Why rejected |
|---|---|
| Bulk transpose `q_scaled` / `k_scaled` to `(Dk, S)` upfront | Allocates `(Dk × S) ≈ 8 KB ≈ 4 banks` of PSUM that the allocator does not free before the scan-loop `update_psum_chunk` (16 banks) → "Out of memory in psum". |
| Pack all `Hq` q-heads into one state tile | Rank-1 update PSUM exceeds 32 KB at `pack_w_full = 4096`. |
| `tensor_tensor_scan`-based formulation | DeltaNet's recurrence is non-associative (`state ← state(I − β k k^T) + β k v^T`). |
| Per-step `nl.ndarray` allocations | Compile time blows up; NKI pins SBUF/PSUM more reliably with function-scope tiles. |
| `nl.broadcast_to` of multi-element tiles inside the scan | The compiler tries to materialize the broadcast view in PSUM and OOMs. Replaced with a per-step `(1,1)` tile + broadcast-to-`(Dk,1)`. |

## What it would take to actually beat the reference

The viable paths all require either bigger problem sizes or
hardware features not present on a single NeuronCore-v2:

1. **Larger `seq_len` (≥ 128) with chunked DeltaNet.** The
   chunkwise-parallel form (Yang et al., 2024) recovers `O(S/c)`
   parallelism per head by representing the within-chunk
   recurrence as a closed-form matrix product. At S=16 the chunk
   would have to be S itself and there is no parallelism left to
   recover.
2. **Multi-core sharding.** With `NEURON_RT_VISIBLE_CORES > 1` the
   `Hv = 32` value-heads could be split across cores. The prompt
   explicitly fixes `NEURON_RT_VISIBLE_CORES=1`, so this is out of
   scope.
3. **A compiler-level `while`-loop primitive in NKI.** Not currently
   exposed. The reference path uses one through XLA.

## Conclusion

The rewritten kernel is correct, compiles in seconds, and hits a
stable ~2.8 ms / iteration with bounded SBUF/PSUM usage and no
unrolling pathologies. For the **specific** Qwen3.5-9B
GatedDeltaNet shape pinned in the host (S=16, Hq=16, single core),
the XLA-lowered reference path retains a structural advantage that
hand-written NKI cannot close: cross-head matmul packing and
multi-engine scheduling that NKI does not expose. A followup that
either raises `seq_len` (to amortize the recurrence cost) or
relaxes `NEURON_RT_VISIBLE_CORES` (to shard `Hv` across cores) is
required to demonstrate a clean speedup.
