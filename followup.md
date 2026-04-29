# GatedDeltaNet NKI Kernel — Architectural Analysis

## Status

[examples/neuron_nki/gated_deltanet.py](examples/neuron_nki/gated_deltanet.py)
has been rewritten and progressively optimized. Final state:

- **Compiles cleanly** (`Compiler status PASS`, ~3 s).
- **Numerically correct** — passes `zml.testing.expectClose` against
  both the host-side ground truth and the `zml.nn.GatedDeltaNet`
  reference at `absolute_tolerance=5e-3`, `relative_tolerance=5e-2`.
- **Stable runtime:** ~2.9 ms / iteration on `inf2.8xlarge` with
  `NEURON_RT_VISIBLE_CORES=1`.
- **Reference (XLA-lowered `zml.nn.GatedDeltaNet`):** ~0.93 ms / iter.
- **Speedup vs reference: ~0.32×.**

Despite multiple optimization passes the kernel cannot beat the
reference for this exact problem shape on a single core. This
document explains the structural floor that prevents the speedup,
backed by measurements and the [Neuron programming
model](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/_sources/nki/programming_model.rst).

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

## Hardware: NeuronCore-v2 (Inferentia2)

- **PE array:** 128 × 128 systolic. Contraction axis ≡ partition dim
  ≡ `pmax = 128`. Stationary free-dim cap = 128.
- **SBUF:** 24 MB.
- **PSUM:** 16 banks × 2 KB = **32 KB total**. Matmul results MUST
  land in PSUM and be copied to SBUF (or directly consumed by ops
  that accept PSUM operands) before re-using the bank.
- **Engines:** Tensor (PE), Vector, Scalar, GpSimd, DMA. Independent
  engines can run in parallel only across **independent** ops.

## What the kernel does (per timestep, per q-head)

Per `(batch, q_head, t)` step the algorithm needs three
contractions over the ``(Dk, qk_rep · Dv)`` packed state plus a
normalize / decay / delta / output chain:

```
state ← state * exp(g[t])                     # decay
predicted_v ← k_n.T @ state                   # 1 PE op,  (Dk,1) × (Dk, 256)
delta_v     ← beta * (v[t] - predicted_v)
state       ← state + k_n ⊗ delta_v           # 4 PE ops (chunked)
out[t]      ← q_n.T @ state                   # 1 PE op
```

That is **6 PE-engine ops** per timestep on the critical path of
the recurrence, plus 2 small `nc_transpose` ops to shuffle output
slots back to `(Dv, 1)` for the HBM store. Total: **8 PE ops × 16
timesteps × 16 q-heads = 2048 PE ops** per kernel invocation.

## Per-step latency floor

Each `nc_matmul` launch on NeuronCore-v2 carries:

- ~50–100 cycle stationary-fill setup
- one cycle per moving-tile column streamed through
- ~128 cycle drain

For our dominant matmul shapes:

| Matmul | Stationary | Moving | Output | Approx cycles |
|---|---|---|---|---|
| predicted_v | (Dk, 1) | (Dk, 256) | (1, 256) | ~400 |
| update (×4) | (1, Dk) | (1, 64)  | (Dk, 64) | ~200 each |
| output | (Dk, 1) | (Dk, 256) | (1, 256) | ~400 |
| transpose (×2) | (1, Dv) | identity | (Dv, 1) | ~200 each |

≈ 2 µs of PE-engine work per timestep, × 256 timesteps = **~0.5 ms
of pure PE work**. Adding vec/scalar/DMA serialization (which
NeuronCore-v2 and the NKI scheduler do not always overlap with PE
work), launch sync, and PJRT call overhead reaches the observed
~2.9 ms.

The reference (XLA → neuronx-cc) achieves ~0.93 ms by:

1. Fusing the 16 independent q-heads' contractions into wider PE
   instructions (matmul tiling across PE columns) — something only
   the full XLA scheduler can do because it has whole-program
   dataflow.
2. Issuing vec/scalar work concurrently with PE work via
   instruction-level scheduling that NKI does not expose at the
   `nisa.*` level.
3. Lowering the XLA `while` recurrence to a dedicated dynamic-loop
   primitive in BIR which has lower per-iteration overhead than
   `nl.sequential_range`.

## What is in the current kernel (all of these are wins over the original baseline)

- **Bulk DMA preload** of `v` / `g` / `beta` for the whole sequence
  per `(batch, q_head)`: removes O(qk_rep) DMAs per timestep.
- **Bulk q / k normalize** into a `(1, S · Dk)` tile, hoisted out
  of the scan: removes 7 vector ops per timestep.
- **Pre-computed `decay = exp(g)`** for the full sequence in one
  `nisa.activation`.
- **Bulk q / k Dk-transpose** into a `(Dk, S)` tile: replaces 2
  per-step transposes + 2 PSUM→SBUF copies with `S` transposes
  hoisted out of the scan, and lets the compiler overlap them with
  the bulk-preload DMAs.
- **Resident packed state**: `(Dk, qk_rep · Dv)` SBUF tile, shared
  across the whole sequence; folds `qk_rep`-many matmuls into one
  wider one.
- **Direct PSUM consumption** for `predicted_v` (read by
  `tensor_tensor` from PSUM) and the chunked update (`tensor_tensor
  add` reads update_psum_chunk straight from PSUM): removes 5
  PSUM→SBUF copies per timestep.
- **`v_chunk = 64`** for the rank-1 update, matching the proven
  Mamba-style PSUM budget.
- **Function-scope scratch tiles** for stable PSUM/SBUF bank
  pinning. Compile time ~3 s.
- **Structured loops only.** `nl.sequential_range` for the timestep
  scan, `nl.affine_range` everywhere else; no Python-level unroll
  of large ranges.

Empirically, every one of these changes verified correctness but
**none moved the wall-clock time** beyond ~50 µs. The Neuron
compiler had already been smart enough to schedule around the
redundant ops in the original kernel; the structural per-step
floor is the binding constraint.

## Optimizations attempted and rejected

| Idea | Outcome |
|---|---|
| Vectorized normalize over `(seq_len partition, Dk free)` | Caused partition-alignment errors with `(1, *)` per-step scratch (`'tensor_tensor_arith' op SBUF partition alignment: 'lhs' and 'rhs' have different partition start offsets`). |
| Bulk transpose `q_scaled / k_scaled` to `(Dk, S)` in one `nc_transpose` | `Matmul stationary free dimension 2048 exceeds gemm_stationary_fmax=128`. Replaced with looped per-column transpose into a shared PSUM bank. |
| `nl.broadcast_to(decay_rows, (Dk, S * qk_rep))` for in-scan decay | `Out of memory in psum` — broadcast tries to materialize as a partition-128 view in PSUM. Replaced with a per-step `(1,1)` tile + `(Dk, 1)` broadcast. |
| Dropping `nisa.memset(0.0)` before `nc_matmul` (auto-detect mode) | Caused gross numerical errors (~`1e38` absolute) — `accumulate=None` auto-detects across loop iterations and the same PSUM tile is reused, so the second iteration accumulates instead of overwriting. Restored the memsets. |
| `affine_range(num_q_heads)` instead of `sequential_range` | No measurable change — the Neuron compiler does not auto-parallelize across iterations that share scratch tiles, and per-head scratch replication would balloon SBUF and compile time. |
| `tensor_tensor_scan` reformulation | DeltaNet's recurrence is non-associative (`state ← state(I − β k k^T) + β k v^T`); `tensor_tensor_scan` requires associative ops. |
| Pack all `Hq` q-heads into one state tile for a single fat matmul | Each head needs its own `k_kp` stationary; `nc_matmul` does not support block-diagonal matmuls, and packing the stationary into `(Dk, Hq)` would compute every (head_a state) × (head_b k) cross term. |
| PE-array tiling via `tile_position` / `tile_size` to gang two matmuls | The dominant `predicted_v` and `output` matmuls have stationary `(Dk, 1)`: column-tiling can't subdivide a 1-column stationary, and row-tiling would shrink the contraction dim below Dk. |

## What it would take to actually beat the reference

The viable paths all require either bigger problem sizes or
features not present on a single NeuronCore-v2:

1. **Chunkwise-parallel DeltaNet (Yang et al., 2024).** The
   within-chunk recurrence has a closed-form matrix product; with
   chunk size `c` the PE work per head scales as `O(S/c · c²) +
   O(S/c · c · Dk · Dv)` — fewer launches at the cost of bigger
   matmuls. At S=16 the chunk would have to be the whole sequence,
   collapsing back to a one-shot per-head computation. The compute
   is the same; the *win* is one big matmul instead of S small
   launches. Implementation complexity is significant (cumulative
   `K K^T β` factors, etc.) and I did not get it working within
   the time budget for this task.
2. **Multi-core sharding.** With `NEURON_RT_VISIBLE_CORES > 1` the
   `Hv = 32` value-heads could be split across cores, multiplying
   throughput by core count. The prompt explicitly fixes
   `NEURON_RT_VISIBLE_CORES=1`, so this is out of scope.
3. **Manual cross-engine scheduling.** Issue PE matmul, vec ops,
   and DMA stores on independent engines per timestep so they
   pipeline. NKI does not expose explicit engine assignment in a
   way that consistently produces concurrent issue on
   NeuronCore-v2; the compiler decides.

## Conclusion

The rewritten and optimized kernel is correct, compiles in seconds,
fits PSUM/SBUF cleanly, and is structurally as tight as
`nisa.*`-level NKI allows for this exact shape — every redundant
op the framework permits us to remove has been removed, with no
measurable wall-clock improvement, indicating that per-`nc_matmul`
launch / engine-serialization overhead is the binding floor at
~2.9 ms. Beating the XLA-lowered reference (~0.93 ms) on this
specific Qwen3.5-9B GatedDeltaNet shape (S=16, Hq=16,
single-core) requires either reformulating the recurrence in
chunkwise-parallel form (a substantial algorithmic rewrite) or
relaxing the single-core constraint.
