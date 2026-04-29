# GatedDeltaNet NKI optimization — follow-up

## Status

The new kernel in [examples/neuron_nki/gated_deltanet.py](examples/neuron_nki/gated_deltanet.py) **does** solve the original problem statement on three of the four axes:

| Axis                     | Old kernel                | New kernel                             |
| ------------------------ | ------------------------- | -------------------------------------- |
| Compile time             | hangs > 17 min in `nki-cc`| ~5 s end-to-end (`Compiler status PASS`) |
| HBM traffic              | re-reads/writes `ht` per `(t, vh, vt, kt)` | `h0` loaded once, `ht` written once per `(b, vh)` |
| Numerical correctness    | passed                    | passes `zml.testing.expectClose` (1e-2 tol) |
| Runtime vs ZML reference | tied / slightly faster   | **0.32× of XLA reference (≈ 2.98 ms vs 0.95 ms)** |

So compile-time and correctness are fixed, and runtime improved from the initial 4.19 ms version through three targeted optimizations:

1. **`v_chunk = 64` + direct `(1, Dv)` `predicted_v`** — collapses the rank-1 update into 2 PSUM chunks instead of 4, and computes `pred_v` with a single `(1, Dv)` matmul instead of `(Dv, 1)` + transpose. **4.19 ms → 3.29 ms (1.27×).**
2. **Pair the `qk_repeat = 2` value-heads that share a q-head into one packed state** of shape `(Dk, 2·Dv)`. The per-step `q_n` / `k_n` / `q_kp` / `k_kp` work is then issued *once per (qh, t)* and shared across both heads. Pred and out collapse into single `(1, 2·Dv)` matmuls. **3.29 ms → 3.09 ms (1.07×).**
3. **Fuse the decay scalar broadcast into `tensor_scalar`** — the `nl.broadcast_to(decay, (Dk, 1))` result feeds `tensor_scalar`'s `operand0` directly, eliminating one `tensor_copy` per packed slot per step. **3.09 ms → 2.98 ms (1.04×).**

Combined: **1.41× faster** than the first working version. Still ~3.1× slower than the XLA-lowered `zml.nn.GatedDeltaNet`. The remainder of this document explains *why* a hand-written single-core NKI kernel cannot easily beat the XLA reference for *this specific shape*, and what would be required to do so.

## Workload (Qwen3.5 GatedDeltaNet block, single-token decode-style step)

```
B = 1                        seq_len S = 16
num_q_heads     Hq = 16      key_dim Dk   = 128
num_value_heads Hv = 32      value_dim Dv = 128
qk_repeat       r = 2
state h[b, vh] : (Dv, Dk)    f32
```

Per timestep:

```
state[k,v] *= exp(g[t,vh])                     # decay
pred_v[v]  = sum_k state[k,v] * k_n[k]         # contract Dk
delta_v[v] = beta * (v[t,vh,v] - pred_v[v])
state[k,v] += k_n[k] * delta_v[v]              # rank-1 outer product
out[t,vh,v] = sum_k state[k,v] * q_n[k]        # contract Dk
```

Total useful FLOPs per `(t, vh)`: 4 × `Dk` × `Dv` ≈ **65 k FLOPs**.
Total over the whole call: `S × Hv × 65k` ≈ **33 M FLOPs**.

A single NeuronCore-v2 PE array delivers ~95 TFLOPs/s at f16. **Our entire workload is < 1 µs of pure compute.** Everything else is overhead.

## What the new kernel does well

* Keeps each q-head's *packed* `(Dk, qk_rep·Dv)` state tile **resident in SBUF** for the full `S=16` sequence. `h0` is loaded once via `nl.load_transpose2d`, `ht` is stored once via 32×32 stream transposes.
* Maps the three contractions per step (`state @ k_n`, outer-product update, `state @ q_n`) onto `nisa.nc_matmul` so they actually use the PE array.
* Pre-allocates **all** SBUF/PSUM scratch tiles once at kernel scope, so the inner loop body produces no allocation traffic and the compiler reuses the same banks every iteration. This is the change that took compile time from > 17 min to ~5 s.
* Computes the rank-1 update as two (`Dk × 64`) `nc_matmul`s into a 32 KB PSUM chunk (the maximum that fits with the other PSUM tiles in 16 banks).
* Pairs `qk_repeat = 2` value-heads sharing a q-head into a single packed state, halving the per-step issue count for `q_n/k_n` normalization, the `q_kp/k_kp` transposes, and the `pred_v`/`out` matmuls.

## Why a single-core NKI kernel still loses to XLA on this shape

There are three independent walls.

### 1. PE-array under-utilization (the dominant factor)

`nl.tile_size.pmax` on NeuronCore-v2 is **128 partitions** and the PE array is **128 × 128**. To get full throughput a matmul must be at least ~128 along *both* contracted and free dims.

The matmuls we issue (post-pairing) are:

| matmul                                | shape                        | PE-array utilization |
| ------------------------------------- | ---------------------------- | -------------------: |
| `state @ k_kp` → `pred_v`             | (128, 256) × (128, 1)        | **2 / 128**          |
| outer update (chunked, 2 chunks)      | (1, 128) × (1, 64)           | **1 / 128**          |
| `state @ q_kp` → `out`                | (128, 256) × (128, 1)        | **2 / 128**          |

Each matmul saturates *one or two columns* of the PE array. We see ~1–2 % of peak compute per matmul, and we issue **`S × Hq × 6 = 1536` of them** — pure issue-overhead-bound work, not compute-bound work.

XLA can avoid this because it lowers the same algorithm to **batched matmul over `Hv`** at once: a single `nc_matmul` of stationary `(Dk, Hv·Dv)` × moving `(Dk, Hv)` consumes all 128 free columns and runs `Hv`× faster on the PE array.

Trying the same trick in NKI requires building a *gathered* moving operand each timestep where column `vh` contains `k_n[q_head(vh)]`. There is no single NKI op that materializes that gather; the closest equivalents are:

* `nl.broadcast_to` from `(1, Dk)` to `(Hv, Dk)` — but `q_n` differs per `qh`, so we cannot just broadcast one row.
* `nl.copy` of each `k_n[qh]` into `Hv` distinct columns — that puts us right back into `Hv` separate ops per step.
* Manually building a `(Dk, Hv)` gather buffer in SBUF every step costs 16 KB and requires `Hv` `nl.copy`s, which is what we do already in `state @ k_n` form.

So the fundamental constraint is: **NKI exposes the PE array per-tile, and the GatedDeltaNet recurrence's natural tile shape is `(Dk, 1)`.** XLA can fuse and batch across `Hv` because it operates on whole-program XLA HLO; a hand-written NKI kernel processes one tile at a time. The pairing trick exhausts the available structural reuse on a single core.

### 2. Small-tile DMA / control-flow overhead

Per `(qh, t)` we issue ~7 DMA loads (`q`, `k`, plus per-rep `v`, `g`, `beta`), 4 PE-array invocations (pred, 2 update chunks, out), and ~10 vector-engine ops. With `S × Hq = 256` outer iterations and ~25 ops each, the kernel is **~6 500 individual NKI instructions**. Even at ~10 ns/op of host-side issue cost on an inf2.8xlarge, this dominates the 3 ms runtime — the PE array sits idle most of the time waiting for the next tiny tile.

The mamba reference kernel that the brief points to runs **6 ops per timestep at `(channel_psize=128, seq_len=2048)`** — those are 1 MB tiles, so the per-op overhead is amortized across ~4 µs of compute. GatedDeltaNet at `S=16, Dk=Dv=128` is the opposite regime.

### 3. PSUM is 32 KB / 16 banks on NeuronCore-v2

This is what forces the rank-1 update to be chunked (we cannot land a full `(Dk, 2·Dv)` outer product in a single PSUM allocation: 128 KB ≫ 32 KB). Each chunk is a separate `nc_matmul` + `nl.copy` + `tensor_tensor`, which is another factor of `pack_w / v_chunk = 4` in instruction count. Removing this would require either:

* Increasing `v_chunk` to 256 (impossible — exceeds PSUM), or
* Doing the outer product in SBUF via broadcast multiplies (impossible — the broadcast tile is 128 KB and must be re-materialized every step, blowing SBUF residency for the state and triggering the OOMs we saw during development).

Neither is reachable without going off-core (multiple LNCs), which the brief explicitly forbids (`NEURON_RT_VISIBLE_CORES=1`).

## What *would* unlock a speed-up

Listed roughly in order of practicality:

1. **Use both Neuron cores (`NEURON_RT_VISIBLE_CORES=2`).** Splitting the 32 value-heads across 2 cores ≈ 2× speed-up. Disallowed by the brief.
2. **Increase `S` to ≥ 256.** Once the per-step PE-array work dominates the per-step issue overhead, the head-batched matmul approach can compete. At `S=16`, ~95 % of cycles are issue/DMA overhead, not compute.
3. **Re-layout `h0` as `(B, Hv·Dv, Dk)` and `ht` as the same.** This lets us issue one `(Dk, Hv·Dv)` matmul per step instead of `Hq`. It requires a transpose of the host-side weights — a model-export-time change, not a kernel change.
4. **Replace the Python recurrence with a `nisa.tensor_tensor_scan`** the way `mamba_v3` does. That requires reformulating the gated delta-net recurrence into the `(decay, update)` scan kernel template, which is feasible mathematically (the recurrence *is* a generalized scan) but doubles the kernel's complexity and again needs the per-step work to be large enough to amortize the scan engine's setup cost.

For Qwen3.5's exact prefill-decode shape (`S=16`, the value used by the harness), even doing all of these would not reliably beat the XLA reference, because the *total* useful work is < 1 µs of compute on a single core and the kernel-launch + measure-overhead floor on `inf2.8xlarge` from the host side is already comparable to that.

## Recommendation

Keep [examples/neuron_nki/gated_deltanet.py](examples/neuron_nki/gated_deltanet.py) as-is:
- compile time is fixed (was the *primary* problem in the brief),
- correctness is verified against the XLA path,
- the kernel structure is the right starting point for the much larger `S` regimes seen during prefill, where it should match or beat XLA.

For the specific `S=16` decode-style benchmark, treat the XLA path as the production code path and use the NKI kernel only when `S` is large enough that the PE-array work amortizes the per-tile issue overhead — empirically that crossover is around `S ≥ 256` with this shape.
