# Adaptive Vectored DmaMapped Loader Context

Snapshot: 2026-07-21. ZML is checked out at `65dd003a` (`adaptive concurrency
again`) with the uncommitted controller follow-up described here. XLA is clean
at `b0990b33b1` (`[XLA:GPU][oneAPI] Enable PJRT_Client_DmaMap for SYCL`). The
user built that checkout into `pjrt-oneapi_linux-amd64-2026-07-21_13-43.tar`;
ZML's oneAPI repository rule selects that archive and its configured SHA-256
matches the file. No XLA source was modified or built during this follow-up.

This file is the authoritative handoff for the current implementation and
measurements. `RESEARCH.md` is historical controller research; its adaptive
staging architecture is retired.

## Outcome

CUDA and oneAPI now use a bounded vectored path with independently controlled
read and physical DMA-event admission:

```text
adaptive positional source reads
    -> bounded queue of small client-DmaMapped blocks
    -> adaptive per-device PJRT event admission
    -> final destination shards/replicas
```

The ready queue contains the final DmaMapped transfer blocks; it introduces no
pageable staging allocation or userspace copy. One controller owns separate
read and per-device DMA limits so the stages cannot fight through the queue.
The load-wide pinned pool remains shared across devices. Other targets retain
the ordinary buffered loader.

Read admission also has an internal byte bound. It is the smaller of the hard
pinned limit and `max(64 MiB, current_read_limit * read_request_size)`, rounded
to whole DMA blocks. It counts unique blocks from reservation through the last
replica callback. This is not another tuned concurrency dimension: it prevents
a fast source from cycling its worker lanes and dirtying the entire hard pool
between 25 ms samples, while automatically expanding with the learned read
width for a slow remote source.

Request, block, and pinned-byte sizing remain explicit and static:

```text
                         local/default       S3/HF/GCS high latency
read_parallelism cap          32                       32
dma_parallelism cap/device    32                       32
read_request_size          2 MiB                   16 MiB
dma_block_size             2 MiB                    2 MiB
max_pinned_bytes         128 MiB                  512 MiB
```

The controller starts at eight reads and eight physical events per device,
then probes within the public caps. It does not tune request size or pinned
memory. `bench_s3.sh` supplies the remote request/pinned preset; every cap and
size remains overridable.

The public `LoadOpts` fields are:

- `read_parallelism = 32` (hard adaptive cap)
- `dma_parallelism = 32` (hard adaptive cap per device)
- `read_request_size = 2 MiB`
- `dma_block_size = 2 MiB`
- `max_pinned_bytes = 128 MiB`
- `shardings`, `progress`, and `total_bytes`

There is deliberately no adaptive toggle or public controller threshold.
Previously removed fields remain removed: `initial_parallelism`, `adaptive_parallelism`,
`max_read_parallelism`, `read_chunk_size`, `max_staging_bytes`,
`max_pinned_buffers_per_device`, `pinned_buffer_size`, and
`transfer_quantum_size`.

## Implementation

### Exact vectored reads

`zml/safetensors.zig` adds `TensorReader.readPositionalAllV`.

- The combined tensor-relative range and backing-file offset use checked
  arithmetic.
- Empty in-bounds reads succeed; out-of-bounds and overflow fail.
- Partial reads resume at the exact slice and intra-slice offset.
- Local calls are split at the platform `IOV_MAX`.
- Unexpected EOF is reported only after all actual progress is consumed.
- Positional calls do not disturb the sequential reader position.

Generic HTTP range reads now fill every supplied slice. A `206` must have a
covering `Content-Range`; a server returning `200` and ignoring `Range` is
handled by discarding the requested prefix before scattering. S3, GCS, and HF
continue to use their existing internal scatter/chunk readers.

The vectored loader opens each distinct backing file once and gives tensor
readers borrowed positional-only handles. This is safe because positional
reads do not mutate streaming position. It is important remotely: Llama 3.1
8B has 291 tensors but only four safetensor files, so per-tensor opens caused
291 object `HEAD` requests. Shared handles reduced the 1000 ms S3Proxy run from
49.03 s to 41.72 s.

### Reservation planner

`VectoredRequestPlan` is a pure planner over `DispatchSpans`.

- It accepts any tensor source range, including ranges crossing shard and DMA
  block boundaries.
- Scatter entries remain in global source/file order.
- Each DMA block contains bytes contiguous in one final destination layout and
  records the exact destination shard/replica mask and final offset.
- Replicated destinations reuse one host block; the block has one reference
  per submitted device transfer.
- Multiple requests from one tensor may complete out of order.

Final-transfer handling is per destination. A destination's final block waits
until the sum of all earlier submitted byte ranges reaches its final offset;
only then is it submitted with PJRT's `is_last_transfer=true`. Finals for one
tensor do not wait for unrelated tensors, which avoids pin-pool deadlocks.

### DMA pool and completion

`zml/mem.zig` adds one load-scoped pool shared by every device of the PJRT
client.

- Host memory is allocated in lazy slabs of up to 64 MiB, registered once via
  `PJRT_Client_DmaMap`, and divided into fixed independent blocks.
- The hard byte limit includes all mapped blocks, not only currently leased
  blocks.
- A request reserves all blocks atomically or waits without retaining a
  partial reservation.
- PJRT completion callbacks decrement reference-counted leases; only the last
  replica callback returns a block.
- Slab storage uses raw allocation so Zig does not poison/memset memory that a
  source read immediately overwrites.
- Pool close wakes blocked reservations. On failure the coordinator stops new
  work, drains active reads and events, aborts deferred transfers, marks
  unfinished PJRT buffers with an error, then destroys events/managers and
  finally unmaps slabs.

A 128 MiB pool made from two 64 MiB mappings completed sharded and replicated
loads across all four B70s. This validates that subranges of one registered slab
are accepted by all devices in this oneAPI client.

### Adaptive coordinator and metrics

The coordinator creates at most `read_parallelism` stable worker lanes. Lanes
below the current read limit fetch round-robin jobs without per-request mutex
admission; lanes above a reduced limit sleep until the controller raises it.
Each request atomically leases all required blocks, fills them with one exact
vectored read, and offers one physical transfer entry to each destination.

An entry takes one credit from its destination device immediately when credit
is free and its device queue is empty. Otherwise it waits in that queue. The
pump may skip a final entry until all preceding bytes for its transfer manager
have been submitted. Among eligible entries it deliberately favors recently
filled blocks: this preserves host-cache locality for DMA, while an aged-cohort
metric prevents one cold or ordering-blocked entry from masquerading as broad
queue pressure. A replicated source block consumes one event credit per
destination while retaining one shared host lease until every callback
completes. Limit reductions affect only new admissions.

The controller samples every 25 ms and uses 50-100 ms startup and 100-250 ms
steady windows. Ordinary attributed probes still require 64 MiB/200 ms. A
probe may finish after 64 MiB/100 ms only when both its epoch-attributed rate
and cumulative post-activation rate independently show at least a 10% gain or
at least a 10% loss. Ambiguous candidates retain the full 200 ms floor and 3%
acceptance threshold. Read probes are demand-gated by DMA starvation; DMA
probes require two fed baseline windows and enough eligible queued work to
exercise the candidate limit on every demanded device. Only one dimension is
probed at a time. Probe work carries read and DMA epochs so late callbacks
cannot validate the wrong limit. Resource reductions may remain within 3% of
the best settled value.

Ready accumulation is read pressure only when DMA is fed and occupancy,
growth, or a material aged cohort persists. Ordering-blocked final entries are
excluded from eligible demand, DMA-probe capacity, and age. Age pressure needs
at least four entries older than 250 ms and at least 25% of the eligible queue;
one deliberately cold entry cannot back reads off. Read latency and admission
wait are never congestion signals. Slow/bursty sources retain read-ahead until
DMA has remained fed for two seconds. Instantaneous empty-queue starvation is
used for fast local sources, but suppressed after a slow source is identified;
its request-completion bursts otherwise look like false DMA starvation.

Startup distinguishes capacity discovery from performance probing. If a
50-100 ms window reaches its deadline without the 32 MiB representative
progress floor, the read limit doubles directly (`8 -> 16 -> 32`) because
there is no output to score. Once representative progress exists, exactly one
initial read increase may also publish without a 200 ms score: fast sources
take the gradual step, while a source below 1.5 GiB/s per-request service
bandwidth opens to the caller's read cap. The latter is the 10 ms S3Proxy case:
16 MiB requests take about 85 ms and need the full 32-read cap to sustain
roughly 5.7 GiB/s. Startup settles after 500 ms without another direct change;
subsequent read changes are scored. DMA is never fast-published and is held
until two representative fed baseline windows exist, avoiding initialization
comparisons and the severe latency inflation observed in unscored experiments.

Metrics include byte-weighted read/DMA latency, unique ready bytes, oldest
eligible age and aged/eligible entry counts, per-device active/high-water
events, DMA starvation, epoch commits, pool waits/high-water, physical
submitted/committed bytes, logical completion, and wall time.

## Code map

- `zml/io.zig`: `LoadOpts`, planner, lazy source/tensor states, coordinator,
  PJRT final/error handling, buffered fallback, and planner/final tests.
- `zml/mem.zig`: `DmaBlockPool`, slab mapping, atomic bulk reservations, and
  lease/refcount tests.
- `zml/safetensors.zig`: exact positional scatter API, borrowed positional
  readers, and bounds/partial/EOF/IOV tests.
- `zml/io/vfs/http.zig`: complete scatter filling and strict range response
  validation.
- `zml/io/vfs/parallel_read.zig`: scatter chunk validation.
- `pjrt/pjrt.zig`: helper for setting unfinished async buffers to an unknown
  PJRT error.
- `examples/io/main.zig`: the five static size/cap environment controls.
- `bench_file.sh`, `profile_file.sh`: local defaults; `PERF_DATA` prevents
  overwriting a caller's profile.
- `bench_s3.sh`: remote defaults and latency/bandwidth controls.

## Performance results

Model: Llama 3.1 8B, 14.96 GiB logical weights. Local numbers use warm page
cache. Medians are wall-clock goodput reported by `examples/io`.

### Adaptive-controller iteration baseline

Before changing `30d18486`, five fresh runs were captured without overwriting
the workspace `perf.data` files. The host was in the previously observed slow
local-read state, so these numbers are a same-state regression control and do
not replace the historical 25-27 GiB/s warm-state results below:

| Placement | Five static runs | Median |
|---|---|---:|
| one B70 | 8.79, 9.36, 9.46, 9.28, 8.85 GiB/s | **9.28 GiB/s** |
| four B70 sharded | 8.86, 8.47, 8.98, 8.73, 8.27 GiB/s | **8.73 GiB/s** |
| four B70 replicated | 6.28, 6.14, 6.36, 6.54, 6.34 GiB/s | **6.34 GiB/s** |

During implementation the host returned to its faster state. Final balanced
adaptive runs were:

| Placement | Five adaptive runs | Median |
|---|---|---:|
| one B70 | 25.66, 25.67, 26.83, 26.83, 26.77 GiB/s | **26.77 GiB/s** |
| four B70 sharded | 26.80, 26.81, 26.80, 26.81, 26.78 GiB/s | **26.80 GiB/s** |
| four B70 replicated | 12.86, 12.79, 12.87, 12.86, 12.60 GiB/s | **12.86 GiB/s** |

The one-device runs converged to 12 reads/eight events per device; sharded
runs used 16/eight. Replicated runs explored 8-16 reads and occasionally
probed 12 events before restoring eight. All three medians exceed the
26.47/26.62/11.84 GiB/s static references below. The default bounded admission
high-water was only 64 MiB despite the 128 MiB hard pool.

### Pre-replacement baseline

Five balanced runs of the old adaptive/staging loader:

| Placement | Median | Individual wall times |
|---|---:|---|
| one B70 | 25.31 GiB/s | 590.969, 591.014, 591.071, 565.594, 592.490 ms |
| four B70 sharded | 24.72 GiB/s | 1.032 s outlier, 605.183, 604.719, 605.316, 580.092 ms |

The separately measured parallel page-cache ceiling on this machine is about
30-31 GiB/s.

### Static vectored configuration before adaptive admission

Configuration: 12 reads, 2 MiB request, 2 MiB block, 128 MiB pinned.

| Placement | Five runs | Median |
|---|---|---:|
| one B70 | 26.57, 25.52, 27.01, 26.47, 25.22 GiB/s | **26.47 GiB/s** |
| four B70 sharded | 26.62, 26.63, 27.16, 25.86, 22.43 GiB/s | **26.62 GiB/s** |
| four B70 replicated | 11.93, 11.87, 11.84, 10.98, 11.58 GiB/s | **11.84 GiB/s** |

The four-B70 sharded median is 7.7% above the old 24.72 GiB/s median and reaches
about 86% of the measured page-cache ceiling. Replicated goodput is logical;
the loader physically transfers 59.83 GiB to four devices.

### Sweeps

The required first block sweep held request/read width/pinned at
32 MiB/32/1 GiB. Five-run medians were:

| DMA block | 1 MiB | 2 MiB | 4 MiB | 8 MiB | 32 MiB |
|---|---:|---:|---:|---:|---:|
| GiB/s | 13.64 | 14.51 | 14.62 | 14.81 | 14.98 |

That point is misleading in isolation: large blocks waste most of each lease
when the request is smaller, while 32 large concurrent requests consume the
whole 1 GiB pool. Repeating the block sweep at the selected 2 MiB request and
12-read point found the interaction:

| DMA block | 1 MiB | 2 MiB | 4 MiB | 8 MiB | 32 MiB |
|---|---:|---:|---:|---:|---:|
| median GiB/s | 16.14 | **25.36** | 24.54 | 22.03 | 22.46 |

At 2 MiB request/block and 128 MiB pinned, read-width medians were:

| Reads | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| median GiB/s | 20.73 | 22.14 | **25.67** | 21.92 | 20.61 | 19.62 |

At 12 reads, 2 MiB block, and 128 MiB pinned, request medians were:

| Request | 2 MiB | 4 MiB | 8 MiB | 16 MiB | 32 MiB | 64 MiB |
|---|---:|---:|---:|---:|---:|---:|
| median GiB/s | **25.10** | 22.82 | 20.61 | 17.52 | 14.90 | 11.48 |

That sweep also changed effective memory admission: a 128 MiB pool can hold
only eight full 16 MiB requests. A follow-up requested by the user fixed
`read_parallelism=12`, `dma_block_size=2 MiB`, and
`max_pinned_bytes=192 MiB` for both 2 and 16 MiB requests. This is the minimum
budget that can admit twelve full 16 MiB requests, so only request size changed
and both cases reached `peak_reads=12`:

| Request | Five runs | Median | Pinned behavior |
|---|---|---:|---|
| 2 MiB | 9.44, 9.58, 9.35, 9.31, 9.53 GiB/s | **9.44 GiB/s** | 36-62 MiB high-water, 64 MiB mapped, no waits |
| 16 MiB | 8.20, 8.57, 8.33, 8.41, 8.06 GiB/s | **8.33 GiB/s** | 192 MiB high-water/mapped; both cases reached 12 reads |

The host was in a much slower local-read state during this follow-up (the
2 MiB control was 9.44 rather than the earlier 25-27 GiB/s), so these absolute
numbers must not be mixed with the warm-page-cache table. The controlled
comparison nevertheless exposes a real coupling after the read: one 16 MiB
completion submits roughly eight 2 MiB transfers, so twelve simultaneous read
completions can burst about 96 PJRT submissions. At 192 MiB, completion latency
for the same 2 MiB DMA blocks rose from roughly 0.09 ms in the 2 MiB case to
1.2-1.9 ms, and later reads waited for groups of eight blocks even though peak
read width was twelve.

Increasing the 16 MiB budget did not solve that coupling in the immediate-
submission implementation. A 384 MiB
double-bank trial fell to 6.67 GiB/s with 11.65 ms average DMA completion
latency; a 1 GiB five-run median fell to 5.33 GiB/s with 40-56 ms DMA latency
and the entire pool mapped/in use. More memory let read completions enqueue a
larger PJRT backlog. The adaptive event gate plus bounded read admission now
removes that coupling; final controlled results are recorded below.

Pinned-limit medians at 12 reads and 2 MiB request/block were:

| Limit | 128 MiB | 256 MiB | 512 MiB | 1 GiB |
|---|---:|---:|---:|---:|
| median GiB/s | **25.68** | 25.52 | 25.18 | 24.98 |

The lowest-memory candidate is also the fastest. The later final five-run
median improved to 26.47 GiB/s after shared source handles and normal run
variance.

With adaptive event admission and the internal read-admission bound, the
requested 16 MiB controlled rerun is qualitatively different:

| Hard pinned limit | Wall goodput | Peak read calls | Peak DMA/device | Avg DMA latency | Actual high-water |
|---|---:|---:|---:|---:|---:|
| 192 MiB | 19.69 GiB/s | 12 | 8 | 0.507 ms | 192 MiB |
| 384 MiB | 20.32 GiB/s | 16 | 8 | 0.532 ms | 256 MiB |
| 1 GiB | 20.33 GiB/s | 16 | 8 | 0.546 ms | 256 MiB |

Extra registered-memory allowance no longer raises admitted PJRT concurrency
or recreates the former 11-56 ms completion latency. The 384 MiB and 1 GiB
runs are effectively identical because both settle on the same 16-read/8-DMA
tuple and 256 MiB internal admission bound. The 192 MiB case cannot fully
exercise the temporary 16-read candidate, but remains close and bounded.

### S3Proxy

The selected remote configuration is 32 reads, 16 MiB requests, 2 MiB blocks,
and 512 MiB pinned. It was within 0.2% of 32 MiB/32 reads/1 GiB at 10 and 250 ms
while using half the registered memory. A 64 MiB/24-read/1 GiB candidate was
slower.

| S3Proxy profile | Final runs | Final median/time | Old controller |
|---|---|---:|---:|
| 10 ms / 1000 MiB/s | 2.691, 2.688, 2.689 s | **2.689 s / 5.56 GiB/s** | 2.912 s median |
| 250 ms / 1000 MiB/s | 11.105, 11.102, 11.099 s | **11.102 s / 1.35 GiB/s** | 11.462 s |
| 1000 ms / 100 MiB/s | one validation | **41.724 s / 367 MiB/s** | 43.175 s |

The full 10 ms discovery pass rose from 1.17 GiB/s at 2 MiB/12 reads to
5.40-5.41 GiB/s at 16-32 MiB/32 reads before source-handle sharing. This is why
the request-size knob remains explicit.

Final adaptive validation with the same fixed 16 MiB request, 2 MiB block,
512 MiB hard pool, and 32/32 public caps:

| S3Proxy profile | Adaptive runs | Median/time | Versus static reference |
|---|---|---:|---:|
| 10 ms / 1000 MiB/s | 3.070, 3.094, 3.069 s | **3.070 s / 4.87 GiB/s** | 14.2% slower |
| 250 ms / 1000 MiB/s | 11.223, 11.225, 11.191 s | **11.223 s / 1.33 GiB/s** | 1.1% slower |
| 1000 ms / 100 MiB/s | one validation | **41.862 s / 365.9 MiB/s** | 0.3% slower |

The no-progress bootstrap reaches 32 reads before the first response on the
250 and 1000 ms profiles, satisfying the 3% requirement there. At 10 ms there
is enough early progress to enter gradual read discovery, so serialized 200 ms
read probes consumed a material part of the original 3-second load.

### Probe-floor and startup follow-up

The follow-up implements two complementary changes rather than globally
weakening the 200 ms floor:

- Startup may publish one representative source-side growth step without a
  probe. The slow-source classification sends the 10 ms S3Proxy profile from
  eight to the 32-read cap; a fast local source takes only the ordinary gradual
  step when it is actually starved.
- Settled probes retain 64 MiB/200 ms unless both the epoch cohort and the
  cumulative post-activation cohort show the same decisive result. A gain of
  at least 10% or loss of at least 10% may finish at 100 ms. Unit tests cover
  early keep, early rollback, and an ambiguous 5% candidate that must wait.

The 100 ms exit does not fire in the current 10 ms S3Proxy run because source
startup reaches the cap without creating a steady probe. It removes latency
only when a real later candidate is clearly different. This is intentional:
the run-to-run issue left on 10 ms is no longer a serialized 200 ms decision.

Three final 10 ms runs were 3.044, 4.047, and 3.445 s (median **3.445 s**).
Steady committed goodput was consistently about 5.5--6.0 GiB/s, but initial
PJRT completion stalls varied by roughly 0.2--1.0 s and dominated wall-time
variance; the best run is essentially the prior 3.070 s adaptive median, while
the median is still 28% behind the 2.689 s static reference. These measurements
were taken while the local host was also in its previously documented degraded
state, so they do not replace the balanced reference table above.

One 250 ms / 1000 MiB/s regression run completed in 11.541 s / 1.30 GiB/s.
That is 2.8% behind the previous 11.223 s adaptive median and 4.0% behind the
11.102 s static reference; one run on the degraded host is not a replacement
for the required balanced three-run median, but it confirms correct high-latency
bootstrap (`8 -> 16 -> 32`) and completion.

Experiments rejected during this follow-up:

- Letting scored DMA probes interleave with startup raised the limit through
  12/16/20 events and produced 4.06 s remotely.
- Unscored broad DMA growth collapsed a local run to 6.07 GiB/s.
- A guarded remote-only 8-to-16 shortcut completed in 3.53 s and inflated the
  startup completion latency to about 39 ms.
- A transient jump to 32 events made the initial stall worse (about 0.69 s)
  and completed in 4.07 s. A large ready backlog is therefore not enough
  evidence to raise DMA concurrency.
- Holding the slow source at 16 reads completed in 5.27 s; its 16 MiB requests
  averaged about 84 ms and sustained only 3--4 GiB/s. This validates opening
  that source to the caller's 32-read cap.

Local queue traces also explain the value of recently filled buffers. Replacing
newest-ready `swapRemove` scheduling with strict FIFO reduced steady goodput
from about 12.4 to 10.1 GiB/s on the current host (1.26 s versus 1.58 s wall),
consistent with DMA benefiting from cache-hot pages. The newest-ready policy is
retained, but readiness diagnostics now exclude ordering-blocked final entries
and report an aged cohort rather than treating one oldest entry as pressure.
The accepted local run was 1.258 s / 11.88 GiB/s; it adaptively reduced reads
from eight to two while retaining eight DMA events. This host remains far below
the earlier 26 GiB/s balanced baseline, so the number is diagnostic only.

### `perf`

Current adaptive recordings (workspace profiles were preserved):

- `/tmp/zml-adaptive-1b70.data`
- `/tmp/zml-adaptive-4b70-sharded.data`

| Flat symbol | adaptive one B70 | adaptive four-B70 sharded |
|---|---:|---:|
| `_copy_to_iter` | 73.02% | 64.88% |
| `filemap_get_read_batch` | 4.14% | 3.93% |
| `copy_page_to_iter` | 2.41% | 2.03% |
| `filemap_read` | 2.15% | 1.84% |
| oneAPI/XLA callback `sched_yield` path | 2.17% | 2.18% |
| userspace `memset` | 0.39% | 0.38% |

The controller and queue are below the 0.2% flat-report threshold. There is no
pageable staging or userspace sharding copy; the remaining CPU bottleneck is
the kernel page-cache copy into anonymous DmaMapped blocks.

The profiles are whole-process recordings, so they also include PJRT/device
initialization and teardown. Kernel code accounts for 93.87% of the one-B70
samples and 92.52% of the four-B70 samples. Most of the apparent remainder
after `_copy_to_iter` is adjacent page-cache work (`filemap_get_read_batch`,
`copy_page_to_iter`, `filemap_read`, folio access and RCU bookkeeping), rather
than loader/controller overhead. On four devices, PJRT's BFC output allocation
also exposes `std::vector<unsigned long>::resize` (1.47%) plus page-table,
page-clear, and unmap work. The Intel `xe` driver itself is only about 0.27%.

The clearest actionable non-copy item is the roughly 2.2% callback
`sched_yield` path. XLA special-cases SYCL in
`LocalDeviceState::ThenExecuteCallback` by scheduling a host worker which calls
`Stream::BlockHostUntilDone`; the SYCL implementation turns that into a whole
queue `wait()`. SYCL's stream already implements `DoHostCallbackWithStatus`
using `host_task`, so using an event/stream callback instead of polling a whole
queue is worth investigating in XLA. It can only recover low-single-digit CPU
time in these profiles and its existing special case may encode a correctness
workaround, so it must be validated separately rather than assumed safe.

The XLA checkout contains an experiment in
`xla/pjrt/se/local_device_state.cc`: SYCL skips the separate callback-stream
path and falls through to the existing `DoHostCallback` plus XLA worker-thread
handoff. This preserves same-stream ordering and the contract that PJRT-facing
callbacks execute on XLA's worker, while replacing the worker's whole-queue
`BlockHostUntilDone`/`queue.wait()` with SYCL's asynchronous `host_task`.

Runtime validation rejects this approach. Five warm runs fell from 26.77 to
11.21 GiB/s on one B70 (-58.1%) and from 26.80 to 10.61 GiB/s on four sharded
B70s (-60.4%). Average 2 MiB DMA completion latency rose to approximately
1.45-1.72 ms in the representative runs, with bad 12/16-event probes reaching
2.6-3.0 ms. A host task is an ordered command in the transfer queue; adding one
after each of 7,723-7,924 small copies serializes expensive host-task/event
dispatch with DMA instead of merely observing completion.

Fresh profiles are `/tmp/zml-hosttask-1b70.data` and
`/tmp/zml-hosttask-4b70-sharded.data`. The former lost 20 of roughly 52k
samples; the latter lost none. `_copy_to_iter` drops to 37.72%/37.46% only
because the new overhead is large: `__memmove_avx512_unaligned_erms` accounts
for 24.19%/21.45%, `urEventWait -> sched_yield` for about 7.3%/3.9%, and SYCL
scheduler command enqueue for 0.92%/1.22%. The former callback worker's
`queueFinish` sample disappears, but polling moves to the host-task execution
thread and becomes larger.

Matching whole-process stat files are `/tmp/zml-hosttask-1b70.stat` and
`/tmp/zml-hosttask-4b70-sharded.stat`. Versus the previous profiles, context
switches rise from 31,906 to 57,482 on one device and from 48,848 to 91,209 on
four. Instructions rise from 5.426 G to 7.344 G and 12.487 G to 14.093 G.
Total cycles fall because the serialized transfer queue throttles concurrent
page-cache copying; this is not an efficiency win. Do not use one ordered
`host_task` per 2 MiB PJRT transfer as the replacement for the queue wait.

The host has one NUMA node; NUMA placement does not explain the residual cost.
CCD affinity is a possible experiment, but streaming page copies have little
cache reuse and restricting workers could reduce available memory bandwidth.

Current whole-process `perf stat` (including PJRT/device initialization):

| Counter | adaptive one B70 | adaptive four-B70 sharded |
|---|---:|---:|
| task-clock | 7.330 s | 9.212 s |
| cycles | 37.995 G | 47.577 G |
| instructions | 5.426 G | 12.487 G |
| context switches | 31,906 | 48,848 |
| CPU migrations | 1,041 | 2,745 |
| page faults | 278,886 | 987,208 |

The following recordings and comparison are the earlier static-vectored
investigation retained for historical context.

Final recordings:

- `/tmp/zml-vectored-final-1b70.data`
- `/tmp/zml-vectored-final-4b70.data`
- old four-B70 baseline: `/tmp/zml-vectored-baseline-4b70.data`

Flat cycle profiles:

| Symbol | old four B70 | final one B70 | final four B70 |
|---|---:|---:|---:|
| `_copy_to_iter` | 67.06% | 70.55% | 64.28% |
| `filemap_get_read_batch` | 3.81% | 6.15% | 5.55% |
| `copy_page_to_iter` | 1.79% | 2.76% | 2.55% |
| `filemap_read` | 1.53% | 2.60% | 2.39% |
| `std::vector<...>::resize` | 1.72% | <0.5% | 1.64% |
| `xe_hw_fence_signaled` | <0.5% | <0.5% | 0.57% |

No pageable staging copy, userspace sharding copy, or slab poison/memset appears
in the final profiles. The remaining dominant CPU cost is Linux copying warm
page-cache pages into DmaMapped anonymous memory. DmaMap pins/registers those
pages for device access; it does not let ordinary file-backed page-cache pages
become the PJRT transfer source without this kernel copy.

Selected `perf stat` comparison:

| Counter | old one B70 | final one B70 | old four B70 | final four B70 |
|---|---:|---:|---:|---:|
| task-clock | 6.284 s | 7.602 s | 8.089 s | 8.321 s |
| cycles | 32.906 G | 39.506 G | 42.105 G | 43.298 G |
| instructions | 5.158 G | 5.549 G | 10.420 G | 10.911 G |
| context switches | 42,162 | 30,638 | 39,594 | 41,270 |
| CPU migrations | 3,558 | 1,369 | 2,680 | 1,946 |
| page faults | 275,781 | 278,678 | 985,254 | 988,597 |

The one-device path spends more aggregate CPU on many 2 MiB reads/transfers,
but finishes faster. In the sharded case—the original CPU concern—task-clock
is only 2.9% higher while median wall goodput is 7.7% higher, and the staging
and sharding copies are gone.

One initial post-change profile and the fifth pass of an early request-width
sweep were invalidated when another user's 32-way XLA/oneAPI compiler job
saturated the host. `/tmp/zml-vectored-new-1b70.data` is retained only as a
contaminated artifact and was not used for final conclusions.

## Rejected hypotheses and retained observations

- DmaMap itself did not make CPU writes materially slower. It changes page
  registration/device visibility, not the basic CPU cache hierarchy.
- Anonymous DmaMapped memory means ordinary anonymous virtual memory allocated
  by ZML and then registered with PJRT; it is not file-backed and not a hidden
  staging allocation.
- Huge pages reduce translation/registration pressure but do not remove the
  page-cache-to-anonymous copy or make CPU writes bypass cache coherency.
- A file-backed mmap used directly for DMA reached about 21.4 GiB/s; mmap plus
  an explicit copy reached about 23.7 GiB/s. Neither beat the final preadv into
  DmaMapped blocks.
- The useful local 12-read knee is a pipeline balance, not twelve dedicated DMA
  engines. Reads and transfers overlap, while the one global byte budget owns
  blocks through read, ready, and PJRT completion states.
- Per-device pools are unnecessary for this client: one DmaMapped slab was used
  successfully by all four B70 devices. A future PJRT client spanning distinct
  driver handles would need separate runtime validation.

## Validation and commands

Passed after the final edits:

```text
./bazel.sh test //zml:test //zml/io/vfs:test
./bazel.sh build --config=release --@zml//platforms:oneapi=true \
  //examples/io:playground //examples/mnist:mnist
./bazel.sh build --config=release --@zml//platforms:cuda=true \
  //examples/io:playground //examples/mnist:mnist
```

Follow-up controller validation on 2026-07-21 also passed:

```text
bazel test //zml:test --test_output=errors
./bazel.sh build --config=release --@zml//platforms:cuda=true \
  //examples/io:playground
```

The CUDA command compiled ZML against the existing PJRT artifact; it did not
build XLA. The follow-up also ran one-device oneAPI local and S3Proxy loads.

Four-device oneAPI sharded and replicated loads completed repeatedly. CUDA
release compilation passes, but this host has no RTX GPU, so the requested RTX
5090 runtime sweep remains for the CUDA machine.

Useful runs:

```text
# Local/default; all values may still be overridden.
./bench_file.sh

# Remote preset selected by the proxy sweep.
LATENCY_MS=10 SPEED_MIB=1000 ./bench_s3.sh

# Preserve workspace perf.data by choosing a different output.
PERF_DATA=/tmp/zml-loader.data ./profile_file.sh

# Four B70s.
ONEAPI_DEVICE_SELECTOR='level_zero:*' ./bench_file.sh

# Four B70 replicas.
ONEAPI_DEVICE_SELECTOR='level_zero:*' ZML_LOAD_SHARDING=replicated ./bench_file.sh
```

## Workspace boundary

The user's benchmark selector behavior remains; the scripts now also carry
the read/DMA caps, size presets, optional `ZML_LOAD_SHARDING`, and `PERF_DATA`.
Workspace `perf.data` and `perf.data.old` were not overwritten or removed.
There is no `CTX.md.orig`. Do not build or modify XLA in follow-up work unless
the user explicitly changes that instruction.
