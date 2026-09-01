# DMA-Calibrated, Source-Adaptive Loader Redesign

Status: design plan, not yet implemented
Repository baseline: detached HEAD `ee306456` (`faster dma bench`)
Historical repackaging reviewed: `brabier/io-rework-static-concurrency`,
`cd6a8973^..d8854b00`
Context reviewed: all of `CTX.md`; lowercase `plan.md` and `RESEARCH.md` are
historical inputs, not this plan

## Decision

Split configuration by when it can be learned:

1. Calibrate the DMA data plane before reading weights. `benchmarkDma` must
   produce an immutable, loader-shaped calibration artifact and the caller
   must pass its configuration into the loader after compile/warm-up.
2. Hold DMA block size, per-device event limits, and any global DMA cap fixed
   for the load.
3. Hold the logical source read size at 32 MiB for this iteration. Tensor tails
   remain smaller. Do not adapt request size.
4. Adapt only the number of active positional source reads. Optimize ordinary
   operation from completed source bytes per wall-clock time, with source-read
   latency as a congestion/plateau guardrail.
5. Keep a minimal, optional, load-scoped side channel only for explicit source
   backpressure such as HTTP 429, a classified overload response, and
   Retry-After. It is a safety brake, not the normal estimator of bandwidth or
   concurrency.
6. Continue monitoring after convergence. Recent measurements must decay and
   periodic bounded probes must reopen the search when the path, cache state,
   network route, or service load changes.

The target flow is:

```text
model metadata + sharding
        |
        v
DMA preflight (same DmaMapped/PJRT transfer path and fair admission policy)
        |
        v
immutable DmaLoadConfig
  - one common DMA block size
  - one event limit per destination device
  - optional fair global event cap
        |
        v
round-robin 32 MiB source jobs
        |
        v
retained-lifecycle/worker admission
        |
        v
build request plan + atomically reserve every DmaBlockPool block
        |
        v
adaptive active-read gate <----- completed read bytes + read latency
        ^                                  |
        |                                  v
        +---- optional load-scoped throttle / retry deadline
        |
        v
positional source read -> per-device ready queues
        |
        v
fixed fair DMA admission -> PJRT callbacks -> release lifecycle/blocks
        |
        v
final buffers
```

This is deliberately not another tuple optimizer. There is one live control
variable: source read parallelism.

## Scope and non-goals

In scope:

- CUDA, ROCm, and oneAPI vectored loading, selected today in
  `zml/io.zig:7211-7231`.
- The current DMA benchmark at `zml/io.zig:4288-6864` and its handoff to the
  loader.
- The active-read controller, its measurements, source backpressure, bounded
  retained work, and fixed DMA admission.
- Local files and arbitrary `std.Io` implementations with no ZML-specific
  feedback, as well as the HTTP/HF/S3/GCS VFS backends.
- A changing source during one load. The selected width is not treated as a
  permanent property of a URI scheme.
- Loads containing more than one source handle. Because separate files can
  silently share disk, page cache, NIC, endpoint, or service quotas, the first
  implementation conservatively gives the whole load one controller, gate,
  measurement stream, and backpressure scope.

Out of scope for this iteration:

- Adapting the 32 MiB logical read size.
- Adapting DMA after the preflight has produced a configuration.
- Changing PJRT/XLA, direct-I/O policy, NUMA placement, or the DmaMapped memory
  mechanism.
- Guaranteeing the mathematical global optimum for an adversarial source that
  changes faster than it can be measured. No online controller can provide
  that guarantee. The goal is bounded exploration and tracking of the best
  recent operating region.
- Independently optimizing multiple source domains. Splitting the load-wide
  controller is a follow-up only when a backend/caller can explicitly prove
  independence and the scheduler has fair pre-reservation worker/pinned
  budgets. Never infer independence from distinct handles or URI schemes.

The fixed 32 MiB read size is an explicit simplification, not an assertion
that 32 MiB is universally optimal. `CTX.md:257-298` shows it is the useful
ROCm source size around the measured 8 MiB/eight-event DMA tuple, while the
historical oneAPI local measurements in `CTX.md:1238-1319` favored smaller
reads. Acceptance must therefore compare the controller with a static 32 MiB
oracle and separately report any change from the current production default.
If 32 MiB itself is unacceptable on a platform, revisit it in a later,
independent design; do not put request-size search back into this controller.

## Context: what is already proven

### Historical branch lineage

The requested branch is a linear reconstruction of earlier work. The literal
chain is:

```text
405a3813 benchmark setup
  -> cd6a8973 io/oneapi: activate DmaMapped loading baseline
  -> 069f190d zml/mem: add bounded DmaMapped block pool
  -> 3df2c9b7 zml/io: batch positional reads through shared sources
  -> 67fc1343 zml/io: bound retained reads and device transfers
  -> dcdcc43a zml/io: adapt source concurrency and request size
  -> d8854b00 zml/io: make each remote admission one Range GET
```

`CTX.md` in that branch mentions `4ad4dd3b` as a base in one historical note,
but `cd6a8973` actually has parent `405a3813`; the new implementation should
not use the former as its literal branch base. `405a3813` and `4ad4dd3b` share
a parent and subject but not a tree (`bench_aws.sh` is absent from the former).
The repackaged series records original squash
`9e87bc400ff6031635b890d63b332e864a94b01a` and maps the earlier work items as
`ktmqvrwz` -> `cd6a8973`, `nwmkrmrk` -> `069f190d`, `rwwxukqk` -> `3df2c9b7`,
`soxlrqyu` -> `67fc1343`, `uxkzvsok` -> `dcdcc43a`, and `otpylnlv` ->
`d8854b00`. Lowercase `plan.md` belongs to the original squash lineage; it is
not present in the final repackaged tree.

| Commit | Durable contribution | Treatment in this redesign |
|---|---|---|
| `cd6a8973` | Activates DmaMapped loading on oneAPI and moves device allocator warm-up before the measured load. | Preserve. DMA preflight also runs after compilation and allocator warm-up. |
| `069f190d` | Adds `DmaBlockPool`: lazy registered slabs (up to the default 64 MiB), fixed blocks, atomic whole-request reservation, a hard mapped-byte cap, and replica-aware leases. | Preserve as the memory safety substrate. |
| `3df2c9b7` | Adds the global positional batch, shared source handles, direct scatter into pooled DmaMapped transfer blocks, submission to final shard offsets, and replica reuse. | Preserve the data plane and model-shape-based placement. |
| `67fc1343` | Separates active source calls, retained request lifecycles, and per-device DMA events. This stopped a large pinned allowance from becoming an enormous PJRT backlog. | Preserve the three distinct constraints; change only how their limits are supplied. |
| `dcdcc43a` | Adds the current multi-axis adaptive policy, resizable gates, dynamic range scheduler, VFS profiles, and typed statistics. | Preserve resizable drain semantics and scheduler; replace the coupled policy. |
| `d8854b00` | Removes hidden backend `parallel_read` fan-out. One admitted remote positional call is one Range GET; retries are serial and retain the credit. It also makes exact-fill/timing capabilities explicit and validates 206/Content-Range/Content-Length, including scatter lists above `IOV_MAX`. | Preserve as a non-negotiable observability, capability, and admission contract. |

The branch demonstrates why this is not a ground-up loader rewrite. The
DmaMapped pool, exact positional reads, shared-source scheduler, per-device
ready queues, callback ownership, and failure drain behavior already solve the
hard data-plane problems. The redesign should narrow and simplify the control
plane around them.

The performance progression supports that split. In the repackaged branch,
the oneAPI DmaMapped baseline improved local loading from about 5.24 to 9.44
GiB/s; the shared positional batch then moved local loading from about 9.61 to
25.97 GiB/s and AWS from about 161 to 314 MiB/s, but an unconstrained 2 GiB
case did not complete. Separating source calls, retained lifecycles, and DMA
events fixed the runaway backlog: the ordinary 2 GiB-default smoke retained
about 40 MiB. In a separate hostile 16 MiB-request/1 GiB-ceiling case, retained
memory stopped at about 312-320 MiB and DMA latency fell from roughly 40-56 ms
to 0.52-0.59 ms. The later adaptive and exact-one-GET changes were
throughput-neutral within the branch's 3% criterion. Preserve those data-plane
wins; do not preserve the coupled search merely because it arrived in the same
series
(`d8854b00:CTX.md:31-71,81-124,145-167`).

### Current data path

The current loader performs the following sequence:

```text
stable maximum worker set
  -> adaptive worker gate
  -> retained-lifecycle gate
  -> round-robin range claim
  -> atomic reservation of every DMA block for the request
  -> active source-read gate
  -> readPositionalAllV
  -> per-device ready queues
  -> adaptive scalar event limit per device
  -> transferData / onReady
  -> release block leases and retained lifecycle
```

Relevant code:

| Area | Current location | Consequence for the redesign |
|---|---|---|
| Loader options | `Parallelism`, `ReadRequestSize`, and `LoadOpts`, `zml/io.zig:5155-5240` | Replace adaptive DMA and request-size fields with an immutable DMA configuration and a fixed 32 MiB source request. Keep a fixed/adaptive read-width override for tests and operators. |
| Vectored orchestration | `loadVectored`, `zml/io.zig:3786-4286` | This is the integration point for DMA settings, source feedback discovery, gates, controller runtime, and logging. |
| Dynamic jobs | `VectoredReadScheduler`, `zml/io.zig:1940-2076` | Keep round-robin claims, but remove tuple/request-size changes. Every non-tail claim is 32 MiB. |
| Read execution | `VectoredReadRequest.run`, `zml/io.zig:1834-1937` | Record controller samples when `readPositionalAllV` returns, before enqueueing DMA. Assign trial generation when the source permit is acquired, not when a job is claimed. |
| Exact scatter read | `TensorReader.readPositionalAllV`, `zml/safetensors.zig:206-257` | Preserve exact fill, bounds checks, local `IOV_MAX` batching, and one-call remote mode. |
| Resizable gate | `AdaptiveRequestGate`, `zml/io.zig:1073-1134` | Preserve natural draining: a lower limit blocks only new work and never cancels an active read. |
| Pinned feasibility | `PinnedGateLimits`, `zml/io.zig:1136-1152` | Recompute with the fixed 32 MiB reservation footprint. Do not confuse feasibility with a source performance signal. |
| Request lifetime and DMA queues | `VectoredLoadPipeline`, `zml/io.zig:1154-1832` | Replace one scalar `dma_limit` with immutable per-device limits plus an optional fair global cap. Remove DMA probe state. |
| Registered pool | `DmaBlockPool`, `zml/mem.zig:212-402` | Keep lazy slabs, atomic `acquireMany`, hard byte cap, and reference-counted release. |
| VFS telemetry | `ReadStats`/`AtomicReadStats`/`ReadStatsProvider`, `zml/io/vfs/base.zig:52-209,256-269` | Keep diagnostics initially, but stop using TTFB/body buckets and scheme hints to choose concurrency. Add a smaller load-scoped backpressure channel. |
| VFS lookup | `VFS.readProfileForPath`, `zml/io/vfs.zig:117-137` | Replace backend-instance policy lookup with a data-plane capability and feedback provider tied to the opened source handle/load. |
| DMA benchmark | result types at `zml/io.zig:4349-4406`; `benchmarkDma` at `zml/io.zig:6553-6864` | Add a loader-shaped recommendation and share the actual fixed DMA admission/submission machinery with the loader. |
| Playground | separate `dma-bench` and `load` commands, `examples/io/main.zig:159-445` | Make `load` run/pass preflight settings by default, with an explicit fixed configuration path for controlled tests. |
| Production-shaped integration point | compile, warm-up, then load at `examples/llm/main.zig:135-140` | Run/resolve DMA calibration after compile and allocator warm-up, immediately before `loadBuffers`. |

### Why the current controller should be replaced

The current `AdaptiveVectoredController` (`zml/io.zig:2104-3063`) jointly
controls read width, request size, and DMA width. Its runtime
(`zml/io.zig:3134-3783`) derives source, lifecycle, queue, and DMA signals and
serializes one probe across all three dimensions. This has concrete problems,
not merely aesthetic complexity:

- DMA block size is outside its search space because `DmaBlockPool` is built
  once at `zml/io.zig:3813`. `CTX.md:310-370` records a ROCm default making
  30,650 2 MiB PJRT transactions while the useful 8 MiB block made 8,390.
- Request-size probes also replace read width through
  `modeledReadConcurrency` (`zml/io.zig:2352-2369,2656-2717`). The measured
  2-to-4 MiB probe compared `12/2 MiB` against `4/4 MiB`, so it could not
  attribute the result to either dimension.
- Source probes are scored from logical request retirement. Retirement occurs
  only after every DMA block and replica callback in
  `RequestContext.completeOne` (`zml/io.zig:1187-1213`), even though raw read
  bytes and time are already recorded at `zml/io.zig:1902-1911`. DMA callback
  latency, replication, and downstream queues therefore contaminate source
  learning.
- Only one dimension may probe, and cooldown/tail rules consume much of a
  finite load. The documented default ROCm load ended at six reads, 4 MiB
  requests, and eight DMA events before it could reach the useful source
  region.
- Monotonic peak goodput and slowly moving baselines retain stale regimes.
  Static `high_latency` and request-minimum hints describe a registered scheme,
  not a path that can move between cache, disk, proxy, network, or an
  overloaded service during the load.
- Current read reduction is driven mainly by downstream ready pressure. A
  source that becomes overloaded while DMA is starved can cause the controller
  to ask for more reads rather than fewer unless a typed throttle happens to
  arrive.
- The stats provider is backend-instance global and sources are deduplicated
  by profile id at `zml/io.zig:3932-3947`. Unrelated traffic through the same
  VFS can be attributed to this load, and Retry-After is not propagated to the
  admission gate.

The static AWS grid in `CTX.md:799-833` also states the resource objective
clearly: widths 24 through 128 at 16 MiB stayed within about 0.7% throughput,
while average request latency grew from roughly 0.39 to 2.02 seconds and
pinned memory grew substantially. At the 32 MiB size proposed here, widths
32/48/64 measured about 949.10/950.53/951.38 MiB/s while pinned memory grew
from 1.06 to 1.57 to 2.00 GiB (`CTX.md:816-818`). The controller should select
the smallest width in the near-peak band rather than treat more concurrency as
success.

The branch also records why bounded exploration and paired confirmation are
structural requirements. A one-shot search stranded dimensions; unbounded
rearming ran roughly ten experiments and achieved 895.23 MiB/s against a
953.74 MiB/s fixed run; short upward windows then accepted implausible
+42% to +479% gains, filled the 2 GiB pool, and caused 42 pool waits. Conversely,
all-high 64-read/32-MiB/16-DMA starts stayed at the initial tuple and consumed
the full 2 GiB because the `64 -> 48` transition was scored badly; attempted
drain/refill fixes reduced ordinary AWS runs to about 853.89-898.30 MiB/s.
This redesign therefore needs an explicit experiment budget, clean transition
accounting, confirmation, and a regression starting deliberately above the
knee (`d8854b00:CTX.md:150-164,244-274,359-406,713-740`).

## Design invariants

The implementation must preserve these invariants throughout migration:

1. One admitted logical remote read causes one physical Range GET unless that
   same call retries. Retries are serial within that call, and the load retry
   coordinator prevents simultaneous retry amplification across calls.
2. A read starts only after all DmaMapped blocks needed by that 32 MiB request
   have been reserved atomically. It must not hold a source permit while
   waiting for pinned memory.
3. Active source permits are released as soon as the positional read returns.
   Retained-lifecycle credits and block leases remain until every destination
   callback completes.
4. Reducing any count limit drains existing work naturally. It never cancels a
   source read or PJRT transfer.
5. Pinned bytes remain hard-bounded independently of every adaptive decision.
6. DMA limits are immutable during a load and are addressed by stable PJRT
   device id, not by an unchecked array position.
7. Replicas reuse one host block until every destination callback completes.
8. Final transfers retain the current per-destination ordering rule and
   `is_last_transfer=true` semantics.
9. Ordinary read optimization works with a plain `std.Io` and no VFS hints,
   timing provider, rate-limit provider, local/remote label, or URI scheme.
10. Explicit source backpressure can only reduce/pause admission. A backend
    cannot prescribe a higher concurrency or override the hard cap.
11. Transfer planning uses the tagged model tensor shape, not the raw
    safetensor descriptor shape; sharding and dispatch-span distinctions must
    survive extraction.
12. All source handles in the first implementation intentionally share one
    load-wide controller because hidden bottlenecks may be shared. Statistics
    and backpressure from other loads/backend users never enter it.
13. Installing a source limit and generation is atomic with gate admission:
    `acquire` returns the generation and monotonic admission id observed under
    the same gate lock as its permit. A caller cannot acquire under one limit
    and be scored as another.

## 1. DMA preflight and loader handoff

### Loader-shaped configuration

Introduce an immutable configuration with approximately this shape (names are
provisional, semantics are not):

```zig
pub const DmaLoadConfig = struct {
    block_size: usize,
    devices: []const DeviceLimit,
    global_max_in_flight: ?usize = null,

    pub const DeviceLimit = struct {
        device_id: u32,
        max_in_flight: usize,
    };
};

pub const DmaCalibrationArtifact = struct {
    schema_version: u32,
    fingerprint: DmaCalibrationFingerprint,
    confidence: DmaBenchmarkConfidence,
    measured_mapped_budget: usize,
    minimum_feeding_bytes: usize,
    config: DmaLoadConfig,
};
```

Keep runtime configuration separate from measured provenance.
`DmaBenchmarkResult` owns the device-limit storage and exposes one
`DmaCalibrationArtifact`; its storage must outlive the loader call. The
low-level vectored loader borrows either form, so caller-owned fixed slices
must likewise outlive the call:

- `.calibrated = &artifact`, the normal explicit handoff; or
- `.fixed = config`, an operator/test override that bypasses measurement but
  goes through the same structural validation and fixed admission.

Structural validation applies to both forms. Schema, fingerprint, and
confidence/provenance validation applies only to calibrated or cached
artifacts; an operator-fixed configuration makes no claim that it was measured.
The eventual Zig API should encode this distinction rather than accepting two
identical config values with different names.

Automatic/caller-produced `.calibrated` handoff requires
`DmaBenchmarkConfidence.confident`. A `budget_exhausted` result remains useful
diagnostics but returns `error.DmaCalibrationInconclusive` at calibrated
handoff; the caller may rerun with a larger benchmark budget or deliberately
copy a reviewed tuple into `.fixed`. Never silently promote an inconclusive
artifact or substitute device zero's result.

DMA detection is caller-owned: compile, warm device allocators, call
`benchmarkDma`, keep its result alive, then pass the artifact to the loader.
A production wrapper may automate that sequence, but generic `zml.io.load`
must not silently benchmark because it cannot know that compilation and warm-up
have happened. Resolve DMA only on the CUDA/ROCm/oneAPI vectored branch.
Buffered targets neither run the benchmark nor consume the config; reject a
supplied DMA option there with a typed invalid-option error rather than
silently pretending it was enforced.

There must not be an `.adaptive` DMA mode in the loader. Environment overrides
in `examples/io/main.zig` construct `.fixed`; they do not revive live DMA
search.

Before allocating the pool, validate:

- every destination device used by the chosen shardings appears exactly once;
- no unknown or duplicate device id is present;
- every width and the block size are nonzero and within absolute safety caps;
- `global_max_in_flight`, when present, is nonzero and does not exceed the sum
  of per-device limits;
- the block size is at most 32 MiB and is compatible with the pool/planner;
- the pinned budget is at least `max(planner's worst full-request reservation,
  unique host-block bytes needed to feed the effective DMA tuple)`, using
  `min(global cap, sum(device widths))` when capped and replica-sharing rules;
  do not add these terms because the same leased blocks feed DMA;
- for a calibrated/cached artifact, its fingerprint matches the current PJRT
  plugin, device kinds/ids, visible topology, and workload distribution, and
  the caller's mapped budget is compatible with the recorded
  `minimum_feeding_bytes`/measured budget.

Return a typed configuration mismatch instead of silently using the first
device recommendation or a stale array index.

### One common block size

The benchmark currently reports a block size per device, but the loader has
one `DmaBlockPool`, one block layout per request, and shared replica blocks.
Arbitrary per-device block sizes are therefore not representable without
duplicating or repacking host data.

Make the loader recommendation choose one common block size explicitly:

1. Keep per-device isolated block results as diagnostics.
2. Coarsely screen common candidates (currently 2/4/8/16/32 MiB) across all
   used devices with the model's full-block/tensor-tail distribution at a
   documented small reference-width set, not one hidden width.
3. For every block candidate within the coarse finalist band, tune each
   device's width, including widths below eight. This evaluates finalist
   `(common block, per-device width vector)` tuples rather than assuming block
   and width are independent.
4. Verify finalist tuples concurrently. Reject one that materially starves any
   device; among tuples in the accepted aggregate/per-device throughput band,
   choose the smallest common block and then the smallest qualifying widths.
5. If concurrent verification changes the block ranking or leaves the accepted
   band, promote the next coarse finalist and repeat. Only a stable verified
   tuple proceeds to global-cap consideration.

Authoritative tuning must be allowed to select below eight events. Today
`DmaBenchmarkOpts.initial_parallelism = 8` treats smaller widths as
diagnostic-only (`zml/io.zig:4413-4416,5491,5881,5961`). Remove that selection
floor or justify it with platform evidence; otherwise calibration can encode a
known-nonminimal width forever.

Do not derive a common block after the fact with `min`, `max`, or “device 0
wins”; that would make a synthetic result look representable without measuring
the represented configuration.

Start with a 3-5% common-block/per-device throughput band and the current 2%
global-cap band. The current `DmaBenchmarkOpts.block_selection_tolerance =
0.15` at `zml/io.zig:4430-4436` is intentionally loose and should be
revalidated for a configuration that is now authoritative rather than merely
a source-headroom hint.

### Per-device and global fixed admission

Replace `VectoredLoadPipeline.dma_limit` with immutable `limits_by_device`.
The existing `active_by_device` and ready queues already provide most of the
mechanics.

When `global_max_in_flight` is absent, bypass global admission entirely. When
it is present, use the same weighted max-min semantics proven by the current
benchmark-local `DmaBenchmarkFairGate` (`zml/io.zig:4537-4644`):

- balance active grants relative to each device's calibrated width;
- rotate ties;
- time-share a cap below the number of active devices without starvation;
- lend unused slots when a device has no eligible waiter.

Extract this selection rule into shared code used by both benchmark and
loader. A benchmark-only fair gate followed by an ordinary loader gate would
not enforce the measured policy.

Do not always search or install a global cap. Enter the cap search only when
the concurrently verified aggregate scales materially worse (start at about
10%) than isolated device results, or callback latency grows dramatically
(start at about 2x). For candidate caps, preserve the benchmark's current
guardrails: choose in the 2% aggregate-peak band only when each device retains
at least 95% of its reference share and fairness is at least 0.98. Emit a cap
only when it produces a material throughput gain or roughly 2x latency
reduction; otherwise leave `global_max_in_flight = null` and bypass the gate
(`CTX.md:20-72,388-408`).

Use the recorded hardware neighborhoods as regression oracles, not universal
defaults: the observed ROCm candidate is one common 8 MiB block, about eight
DMA events, 16 reads, and roughly 24.59 GiB/s median under that host regime;
the four-B70 oneAPI case is one common 4 MiB block, eight events per device,
and a fair global cap of four. Host contention was material in the ROCm runs,
so neither tuple should be hard-coded.

### Benchmark fidelity

The current benchmark uses DmaMapped memory and
`AsyncHostToDeviceTransferManager.transferData`, which is the correct API, but
its reusable lanes and completion pattern do not exactly match the loader's
manager-per-tensor destinations, callback pump, offsets, or final ordering.
Before making automatic production preflight the default:

- factor a common fixed DMA submission/admission engine out of
  `VectoredLoadPipeline`;
- have the benchmark feed synthetic ready blocks through that engine;
- use the model/sharding-derived distribution already built by
  `benchmarkDma`;
- exercise asynchronous callbacks and representative destination offsets;
- keep setup and device allocator warm-up outside sampling.

The benchmark need not read weights. The first explicit handoff may use the
existing exact `transferData` path with representative tails, but compare its
selection against the loader callback path. Make the larger shared-submission
extraction a prerequisite for automatic production preflight if that
difference changes selection or confidence; the fair admission selector is
shared in either case.

### Placement and caching

Run preflight after model compilation and `Platform.warmupDeviceAllocators`,
immediately before `loadBuffers`; `examples/llm/main.zig:135-140` is the
current natural boundary. Report calibration time separately from load time.

An in-memory cache may be added after correctness. Its key must include at
least PJRT/plugin build identity, backend/device identity, visible topology,
driver/runtime identity when available, host identity, candidate policy, and
a fingerprint of the model transfer-size/sharding distribution. A cached
entry still gets a short neighbor/health confirmation. A failed confirmation
reruns full calibration. Persistent caching is not required for the first
implementation, but local subsecond loads make it important before caller-owned
automatic preflight becomes universally enabled.

## 2. Fixed 32 MiB source requests

Set one internal constant for normal claims:

```text
read_request_size = 32 MiB
```

Only tensor tails are smaller. Remove request-size transitions from
`VectoredReadScheduler`; it continues to round-robin unscheduled tensor ranges
and to permit concurrent positional ranges from one large tensor.

Remove `ReadRequestSize.adaptive`, exact-size timing buckets, source-minimum
selection, and `modeledReadConcurrency` from controller policy. Static VFS
minimums may remain temporarily as diagnostics, but they do not change the
chunk or starting width.

Do not use `high_latency` to decide how a read is executed. Instead expose a
pure data-plane capability on an opened source:

- local/resumable positional mode may batch at `IOV_MAX` and resume short
  reads;
- exact remote positional mode passes the complete scatter list once, even
  above `IOV_MAX`, so one logical admission remains one Range GET.

This capability says how to satisfy the call; it is not a concurrency hint.
The safe default for an arbitrary plain `std.Io` is resumable `IOV_MAX`
batching. Only an explicit opened-source capability may select the
single-call exact-fill mode.

The feasible read cap is:

```text
min(
    operator hard count cap,
    floor(max_pinned_bytes / full_request_reservation_bytes),
    remaining schedulable full requests,
)
```

Compute `full_request_reservation_bytes` from the selected DMA block size (the
maximum mapped bytes required by an actual 32 MiB claim in the model/sharding
request plan), not from `ceil(32 MiB / block_size)`. `VectoredRequestPlan.init`
builds blocks separately across writer-mask/dispatch-span boundaries, so
boundary fragmentation can require extra partially filled blocks. The
scheduler may use an exact per-claim footprint for admission and the controller
cap must use the maximum over still-possible full claims.

## 3. One-dimensional read-parallelism controller

### Objective and inputs

The primary objective is aggregate source completion throughput for this load:

```text
read_goodput = source bytes completed during an eligible wall-clock interval
               / eligible interval duration
```

This is deliberately measured when `readPositionalAllV` returns, before DMA
enqueue. The controller may also consume byte-weighted source service latency.
With fixed full-size calls, operation-weighted and byte-weighted full-request
latency are equivalent; tails must not dominate the latency statistic.

Ordinary policy inputs are limited to:

- source bytes and full requests completed;
- wall-clock interval;
- source-call service latency;
- current/peak active source calls;
- whether the gate actually exercised the candidate while full requests were
  still schedulable;
- remaining full requests (unscheduled plus already claimed/pre-source), solely
  to decide whether a probe can finish.

The following current signals must not decide source performance:

- GPU-committed or request-retired bytes;
- DMA starvation, DMA callback latency, or a DMA probe epoch;
- ready-queue age/bytes;
- VFS minimum request size, `high_latency`, URI scheme, TTFB, or modeled body
  bandwidth;
- an all-time peak from a previous source regime.

Pinned/lifecycle/DMA state can mark a measurement **ineligible** when the
candidate could not run, but it cannot be interpreted as evidence that the
source width is faster or slower. This keeps structural backpressure separate
from source estimation.

### Trial measurements

Make the controller a pure state machine in a new
`zml/io/read_controller.zig`. Feed it explicit samples and return an optional
new limit. Keep clocking, atomics, gates, and logging in the runtime adapter.

Each trial has a generation assigned when a read acquires the source permit.
Do not assign it when the scheduler claims work, because pool or lifecycle
waits can otherwise attribute a call to a configuration it never ran under.
Extend the gate so installing `{ limit, generation }` and granting a permit
share one lock/critical section, and make `acquire` return that generation plus
a monotonic admission id. Separate atomics for the limit and generation permit
impossible mixed snapshots.

Every limit change has an unscored transition window followed by a clean scored
window. During transition, count **all** source completions and elapsed time for
probe-cost/tail accounting, regardless of generation; old calls consume real
bandwidth and must not disappear from the numerator while remaining in the
denominator. When the last old-generation call retires, snapshot the highest
candidate admission id as the rollover cutoff. Wait for every candidate call at
or below that cutoff to retire; only calls admitted after the cutoff experienced
the candidate for their whole service lifetime. Reach capacity with that clean
cohort before opening the scored interval. New clean calls may keep the
pipeline full, so this is an admission-cohort barrier rather than a global
empty/drain barrier.

For an increase:

- install the new gate limit and trial generation;
- account transition/fill cost from installation in the experiment budget;
- after the old cohort retires, snapshot and drain the rollover admission set,
  then require the post-cutoff active peak to reach the candidate before
  scoring;
- score a representative interval from post-cutoff completions only; do not
  compare a partial rollover cohort to a settled baseline.

For a decrease:

- lower the gate and let old calls drain;
- account the drain time when deciding whether the finite tail can afford the
  experiment;
- after no old-generation call remains, apply the same admission-id cutoff and
  start measurement only when the rollover set is gone and active calls are at
  or below the new limit, so neither old-width contention nor one-time drain is
  mistaken for the candidate's steady throughput.

A representative first threshold set is:

- after clean activation, one full candidate-width turnover as warm-up, then at
  least `max(8, candidate_width)` complete 32 MiB calls in the scored window;
  use more/paired windows when noise demands it rather than scoring eight early
  completions while dozens of long calls remain right-censored;
- at least 100 ms of wall time;
- extend a window for a slow source until the sample floor is met, bounded by
  remaining work and an observed-latency-derived deadline rather than a
  local/remote classification;
- no decision from an application-limited or downstream-blocked window;
- no probe whose estimated cost exceeds 25% of the useful remaining load.

These are starting values and belong in one internal policy definition, not
as a large public threshold surface.

### Search policy

Use a bounded, reversible, bracketed hill climb over one count dimension.
This is easier to attribute than the present coordinate controller and does
not assume that latency alone identifies congestion.

#### Startup

1. Start at `min(12, feasible_cap)` unless an explicit fixed/test initial value
   is supplied. Retain the branch's measured prior until fixed-32-MiB oracles
   justify changing it; this is still a warm-start prior, not a source
   classification.
2. While the gate is saturated but no read has completed, use generic timed
   no-response steps `12 -> 24 -> 32`, clipped by feasibility. Stop blind growth
   at 32. This covers high-bandwidth-delay paths without a `high_latency` hint
   and bounds damage before feedback is possible.
3. Once representative completions exist, probe multiplicatively upward on a
   ladder centered on the observed point (for example 12, 24, 32, 48, 64,
   clipped by the feasible cap).
4. Continue while the candidate has a statistically credible throughput gain
   and no material latency inflation. Stop at the first confirmed plateau,
   regression, explicit backpressure, cap, or finite tail.

#### Selection and refinement

- Keep recent per-width trial results, not one monotonic global peak.
- Define the accepted throughput band as within 3% of the best comparable
  recent width.
- Select the smallest width in that band. Latency breaks an otherwise
  ambiguous tie in favor of the lower-latency/lower-width point.
- After bracketing a knee, test one useful intermediate width when it can
  change the answer (for example 24 between 16 and 32). Do not run a full
  integer sweep.
- Interleave a best/candidate confirmation pair when noise makes the result
  ambiguous and enough work remains. Reuse/extract the paired-ratio confidence
  logic already used by the adaptive DMA benchmark rather than comparing one
  lucky short window with a stale EMA.
- A candidate that cannot exercise capacity is infeasible, not slow. Restore
  the last settled width without poisoning its throughput estimate.
- Bound each search episode (start with six scored candidates for bootstrap and
  four after a detected regime change) and cap all exploratory work at 25% of
  useful load bytes/time. Steady operation earns at most one adjacent-probe
  credit after several representative settled windows; a regime reset spends
  only remaining global budget. Exhaustion settles at the best confirmed
  recent width and never rearms an effectively unbounded probe sequence.

#### Steady tracking

Settling is not terminal:

- maintain short and long decaying estimates at the selected width;
- periodically probe one adjacent lower or higher candidate while substantial
  work remains, alternating direction and preferring a lower-resource check;
- invalidate stale width estimates after two representative windows show a
  material regime shift, such as a large throughput change or latency
  inflation with flat/regressing throughput;
- if latency inflates while throughput is flat/regressing, probe down first;
  if available throughput changes without queue-like latency inflation, probe
  the neighboring direction that can recover bandwidth;
- after invalidation, restart a bounded search centered on the current safe
  width rather than comparing against the best rate seen at the start of the
  load.

Throughput decides whether a width is useful. Latency is a veto/direction
signal around a plateau, not proof that a service is rate-limiting. This is
important because a genuine route or device slowdown can raise latency without
being caused by excess concurrency.

#### Tail behavior

Do not start a trial unless it has enough full 32 MiB ranges for candidate
activation, the sample floor, and any required confirmation. This accounting
must include unscheduled ranges **and** jobs already claimed but waiting for a
request plan, lifecycle credit, blocks, or a source permit. Those jobs have not
yet received a generation and can still become valid trial samples. When the
tail begins:

- finish at the last settled width;
- roll back an unscored candidate;
- do not leave the largest startup width installed merely because the load
  ended before it could be judged;
- report that convergence was tail-limited.

Short loads may finish on the warm-start prior. That is preferable to turning
every finite load into a benchmark suite; DMA caching and a future read-width
cache can improve the prior independently.

### Suggested controller states

Keep the implementation explicit and small:

```text
bootstrap_no_response
  -> measure_baseline
  -> probe_up
  -> refine_or_settle
  -> steady
       | periodic adjacent probe
       | regime change -> measure_baseline
       | explicit throttle -> backoff
  -> tail
```

There is no request-size state, DMA state, source-tuple settlement state, TTFB
model, or coordination turn between dimensions.

## 4. Minimal explicit source backpressure

### Why a side channel remains justified

A successful 32 MiB call that internally received HTTP 429, slept, retried,
and eventually completed is observationally similar to an ordinarily slow
read. Throughput/latency control can eventually react, but it cannot promptly
honor a server deadline or distinguish provider enforcement from path
variance. Because retries intentionally remain inside the backend call, the
ordinary `!void` read result also hides a recovered throttle.

Retain one optional typed exception for that semantic information. Do not use
it to report a preferred concurrency, source bandwidth, “high latency,” or a
request-size minimum.

### Contract

Expose feedback from every opened source handle into this load's sink, not
cumulative statistics for the whole registered backend. This must be an
event-driven sink, not a controller-polled backend snapshot, so a first-call
throttle can close load admission before more work starts. A minimal event is:

```zig
pub const ReadBackpressureEvent = struct {
    kind: enum { rate_limited, overloaded },
    retry_not_before_ns: ?u64,
};
```

The load-owned sink assigns its own monotonic event sequence, queues the event,
and wakes the controller/gate. Provider-local sequence values are never
compared. Required semantics:

- events are scoped to calls made through this load's opened handles;
- a backend publishes synchronously before sleeping/retrying, and publication
  immediately pauses load admission;
- `retry_not_before_ns` uses the controller's monotonic time domain and
  reflects an explicit header when available, otherwise the backend's actual
  selected retry delay;
- concurrent events in this load take the maximum retry deadline;
- unrelated users of the same HTTP client/backend cannot trigger this load's
  controller;
- the provider is optional. Plain `std.Io` needs no implementation.

For an HTTP-date `Retry-After`, compute a nonnegative duration from the wall
clock at receipt, then add that duration to the monotonic timestamp captured at
the same point. Add standard Retry-After parsing and preserve HF's existing
`RateLimit: ... t=` delay. A generic HTTP 503/5xx is not automatically an
authoritative overload signal: emit `.overloaded` only when the backend has
classified it as such or it carries explicit retry semantics. Other server
failures remain diagnostic/error input, not controller policy.

Attach every opened handle to the load sink before its first positional read.
With the current lazy `SourceSlot.ensure` path, either eagerly open/register
sources before starting controller workers or make `ensure` synchronously
attach the sink before returning the reader; path-level profile discovery is
too early and backend-global provider discovery is too broad.

Already-admitted calls need coordinated physical retries. Give the load a
shared `RetryAdmission` used by backend retry loops. On backpressure it closes
until the maximum deadline, begins recovery with one retry grant, and paces
additional grants after clean retry completions up to the controller's reduced
limit. Apply jitter so calls released at the same advertised deadline do not
form a retry herd. This remains serial inside each logical call and does not
create hidden Range GET fan-out; it adds coordination across calls that already
hold source permits.

The existing serial retry loops are the producer integration points:

- HTTP: `zml/io/vfs/http.zig:401-429,555-563`;
- HF and its rate-limit delay: `zml/io/vfs/hf.zig:780-810,895-910,950-962`;
- S3: `zml/io/vfs/s3.zig:806-836,942-954`;
- GCS: `zml/io/vfs/gcs.zig:972-1001,1080-1090`.

### Controller reaction

On a new explicit backpressure event:

1. Cancel/rollback any performance probe.
2. Immediately lower the admission limit by the branch's measured starting
   policy, `max(1, floor(current * 0.7))`, and require at least a five-second
   clean cooldown before another upward probe. Treat other factors as explicit
   experiments, not silent defaults.
3. Pause **new** admissions until `retry_not_before_ns` when a deadline is
   present. Existing reads retain their source permits and blocks, but every
   physical retry passes through the shared retry coordinator; do not cancel
   them or release them as a simultaneous retry wave.
4. Discard trial samples spanning the event.
5. Resume from the reduced width in a recovery mode and grow only after clean,
   representative source windows.

Without a provider, nothing fails: serialized retries naturally occupy
permits, read goodput falls/latency grows, and ordinary bounded probes can move
down. Reaction is slower but the data plane and hard bounds remain correct.

Keep the broader `ReadStats` counters for logging during migration if useful,
but the controller consumes only scoped explicit backpressure. Remove its
dependency on exact-size TTFB/body timing and backend-global failure cohorts.

## 5. Admission and backpressure separation

Continue to distinguish three resources:

1. **Active source calls**: the only adaptive count.
2. **Retained request lifecycles / pinned bytes**: a bounded read-ahead safety
   buffer from read start through all DMA callbacks.
3. **DMA events**: fixed per-device limits and an optional fixed global fair
   cap from preflight.

Keep the current ordering in `VectoredReadRequest.run`: reserve blocks before
taking a source permit. The stable worker-prefix gate is still useful because
it prevents all maximum workers from reserving 32 MiB each while waiting for a
smaller source limit. Derive its limit deterministically from the installed
read/lifecycle limit (for example `min(max_workers, lifecycle_limit)`); it is a
pre-reservation mirror/safety bound, not a second learned concurrency axis.

Replace the current `high_latency ? 8 : 0` lifecycle slack decision at
`zml/io.zig:4013-4020` with one source-agnostic fixed allowance. For the first
implementation, use the conservative historically remote-capable value:

```text
retained_slack = 8,
lifecycle_limit = min(pinned-feasible requests, read_limit + retained_slack).
```

This is a bounded decoupling allowance, not a learned source property. Before
shipping it, compare fixed values 0/1/2/4/8 across local, remote, and mid-load
low-latency <-> high-latency transitions, then select one repository-wide
constant by the end-to-end acceptance criterion. Zero must remain in that
study because `CTX.md` shows completion-aware zero-slack pacing helping local
oneAPI, while remote S3 needed roughly eight retained slots and four regressed.
If no single fixed value satisfies both regimes, redesign structural pacing in
a separate decision before enabling this controller; do not reintroduce
`high_latency`, scheme inference, or live slack adaptation. Replica sharing and
the optional global DMA cap still enter hard pinned-feasibility calculation,
but not this fixed source read-ahead policy.

If pool/lifecycle pressure prevents a read candidate from exercising its
limit, mark the trial infeasible and hold/rollback. Do not feed ready bytes,
DMA starvation, or pool-wait duration into the read throughput comparison.

## Public API migration

Target the following conceptual surface:

```zig
pub const ReadParallelism = union(enum) {
    adaptive: struct {
        initial: usize = 12,
        maximum: usize = max_load_read_parallelism,
    },
    fixed: usize,
};

pub const DmaConfigSource = union(enum) {
    calibrated: *const DmaCalibrationArtifact,
    fixed: DmaLoadConfig,
};

pub const LoadOpts = struct {
    read_parallelism: ReadParallelism = .{ .adaptive = .{} },
    dma: ?DmaConfigSource = null,
    max_pinned_bytes: usize = 2 * GiB,
    shardings: []const Sharding = &.{},
    progress: ?*std.Progress.Node = null,
    total_bytes: ?*usize = null,
};
```

In the final API, a vectored load requires `dma != null`; a buffered load
requires `dma == null`. During migration only, the current fixed defaults may
be a compatibility fallback with a loud log. Production `loadBuffers` wrappers
perform the explicit benchmark-after-warm-up sequence and pass
`.calibrated = &result.artifact`; operator and oracle paths pass `.fixed`.
There is no hidden benchmark inside generic `load`.

Adaptive and fixed widths are load-wide across all its source handles. Worker
count, lifecycles, and pinned bytes remain separate hard bounds. A candidate
that cannot reach its width because a structural bound is occupied is
capacity-ineligible, not a low-throughput result.

The fixed 32 MiB read size is intentionally not a public adaptive union. A
temporary internal/test override is acceptable for static-oracle regression
work, but production callers should not grow another configuration matrix.

During migration, keep compatibility shims long enough to update all in-tree
callers in one series. Log a clear distinction among:

- DMA calibration time and confidence;
- immutable selected common block/per-device/global limits;
- read-controller initial, peak, settled, and tail widths;
- source read goodput/latency and explicit backpressure events;
- pinned high-water, physical submissions, and total load wall time.

## File-level implementation map

### `zml/io.zig`

- Keep `VectoredRequestPlan`, `VectoredTensorTransfer`, the read scheduler,
  request/block ownership, exact final ordering, and failure drain.
- Replace `VectoredLoadMetrics`' probe/DMA/source-tuple fields with a smaller
  load-scoped source-read metrics snapshot plus diagnostic DMA counters.
- Replace `AdaptiveVectoredController` and `AdaptiveVectoredRuntime` with a
  runtime adapter around one load-wide `ReadParallelismController`.
- Extend the source gate so `{ limit, generation }` installation and permit
  acquisition are atomic and `acquire` returns the admitted generation and
  monotonic admission id.
- Make the scheduler claim fixed 32 MiB jobs and remove `setTuple`,
  `trySetCandidateTuple`, request-size candidate counts, and timing buckets.
- Change `VectoredLoadPipeline` to consume `DmaLoadConfig`, per-device limits,
  and optional global fair admission. Delete DMA-probe epoch/capacity state and
  `setDmaLimit`.
- Accept and validate the caller-produced DMA artifact/config before
  `DmaBlockPool.init`; do not run a hidden benchmark here.
- Score source reads at `readPositionalAllV` completion, never at
  `RequestContext` retirement.

### New `zml/io/read_controller.zig`

- Pure state machine, trial/result records, recent-width estimates, regime
  invalidation, tail logic, and explicit backpressure reaction.
- No PJRT, VFS, allocator, mutex, event, or logging dependency.
- Inline deterministic tests for startup, plateau selection, noisy paired
  comparisons, downward probes, change detection, tail rollback, capacity
  failure, and throttle recovery.

### New `zml/io/dma.zig` (or equivalently named internal module)

- `DmaLoadConfig` structural validation plus calibrated-artifact
  schema/fingerprint/mapped-budget validation and ownership helpers.
- Shared per-device/global fair selector used by benchmark and loader.
- Fixed DMA admission accounting and its focused fairness tests.

### DMA benchmark extraction

Move the large benchmark implementation from `zml/io.zig:4288-7067` into
`zml/io/dma_benchmark.zig` after behavior is covered. The extraction may be a
separate mechanical commit. Add common-block selection, loader configuration,
and the shared callback/submission engine there.

### `zml/mem.zig`

Keep `DmaMapAllocator` and `DmaBlockPool`. Add only a helper for calculating a
full 32 MiB request's block reservation if that avoids duplicate overflow
logic. Preserve atomic whole-request acquisition and replica-aware leases.

### `zml/safetensors.zig`

Keep `TensorReader.readPositionalAllV`. Replace the policy-derived
`batch_iovecs` choice with the opened-source positional capability. Preserve
all bounds, partial-read, EOF, and `IOV_MAX` tests.

### `zml/io/vfs/base.zig` and `zml/io/vfs.zig`

- Separate positional-read capability from adaptive policy.
- Add opened-handle registration with a load-scoped, event-driven
  `ReadBackpressureSink` and shared `RetryAdmission`.
- Keep broad statistics only for observability while callers migrate; remove
  controller dependence on backend-global timing/failure snapshots.
- Stop using `minimum_request_size` and `high_latency` as controller inputs.

### HTTP/HF/S3/GCS backends

- Preserve one positional call/one Range GET and serial in-call retries.
- Publish scoped throttle/overload/deadline feedback before sleeping and take
  coordinated retry admission before every subsequent physical attempt.
- Preserve HF `RateLimit: ... t=`, add standard Retry-After parsing, and do not
  newly treat every generic 503/5xx as semantic overload.
- Preserve strict Range validation and authentication/redirect behavior.

### Examples and builds

- `examples/io/main.zig`: keep `dma-bench`; make `load` demonstrate
  benchmark-to-loader handoff and print the selected loader config. Retain
  explicit fixed controls for oracle measurements.
- `examples/llm/main.zig` and model wrappers: resolve DMA config after compile
  and allocator warm-up and pass it through `loadBuffers` to `zml.io.load`.
- Update MNIST, model tests, `zml/testing.zig`, and every direct `zml.io.load`
  caller.
- Add new source files to `zml/BUILD.bazel`; update VFS build lists as needed.
- Leave lowercase `plan.md` and `RESEARCH.md` as historical documents. Update
  `CTX.md` only with implementation results and measurements after the work is
  complete.

## Implementation sequence

Keep each step buildable and separately reviewable.

### Phase 0: baselines and deterministic harness

- Record the current HEAD and preserve user-owned `perf.data` recordings.
- Add a deterministic synthetic source harness whose service curve can change
  by read count and time. It must observe requested peak concurrency and allow
  injected throttle deadlines.
- Record fixed-32-MiB static width oracles on available local, ROCm, oneAPI,
  and remote fixtures before replacing the controller.
- Compare fixed lifecycle slack 0/1/2/4/8 on steady local/remote and a
  controlled low-latency <-> high-latency transition; retain eight as the
  conservative starting value and block enablement if no one fixed value meets
  the end-to-end criterion.

### Phase 1: represent and enforce DMA results

- Add `DmaLoadConfig`, `DmaCalibrationArtifact`, ownership, and separate
  structural/provenance validation.
- Add common-block selection to `benchmarkDma` and return the loader config.
- Remove the recommendation floor that prevents widths below eight.
- Add immutable per-device limits and optional global fair cap to the loader.
- Gate global-cap search on the scaling/latency trigger and preserve the
  per-device retention/fairness acceptance rules.
- Wire `examples/io load` through explicit benchmark -> config -> load.
- Keep the old source controller temporarily, but force its DMA dimension
  fixed so it cannot move away from calibration.

### Phase 2: share DMA scheduling fidelity

- Extract the fair selector unconditionally. Compare benchmark selections on
  its current exact `transferData` path with the loader callback path; extract
  and share the full fixed submission engine if the difference is material or
  before enabling automatic production preflight.
- If extracted, run benchmark synthetic blocks through that engine with
  callback completion.
- Revalidate current ROCm/oneAPI recommendations and global fairness.

### Phase 3: fix source size and remove two adaptive axes

- Set scheduler requests to 32 MiB.
- Remove request-size adaptation/modeling and DMA adaptation/probes.
- Remove `high_latency`/minimum-size policy inputs while preserving positional
  capabilities and one-GET semantics.
- Reduce options, metrics, logs, and obsolete controller tests.

### Phase 4: source-only controller without a side channel

- Add the pure controller, atomic limit/generation gate, and load-scoped
  source-read metric attribution.
- Implement startup, bracket/refine, smallest-near-peak selection, steady
  reprobes, regime invalidation, and finite-tail rollback.
- Prove it operates correctly with plain `std.Io` and an absent feedback
  provider.

### Phase 5: scoped rate-limit backpressure

- Add per-load event sinks and coordinated retry admission for all its opened
  sources.
- Wire 429, explicitly classified overloads, HF rate-limit delay, and
  explicit/selected retry deadlines from HTTP/HF/S3/GCS.
- Add immediate admission pause, coordinated/jittered retry recovery,
  multiplicative backoff, sample invalidation, and clean recovery.
- Prove concurrent unrelated backend traffic cannot affect the load.

### Phase 6: production integration and cleanup

- Make production-shaped call sites explicitly run preflight after warm-up and
  pass the live artifact to the vectored loader.
- Add in-memory caching/health confirmation if calibration overhead warrants
  it before default enablement.
- Extract modules from `zml/io.zig`, remove compatibility fields, refresh docs,
  and record final `CTX.md` traces/medians.

## Test plan

### Pure read-controller tests

Use deterministic samples rather than sleeps:

- throughput grows through widths 1/2/4/8/16 and plateaus at 16/32: select 16;
- 16, 24, 32, and 64 are within 3% while latency/memory rises: select the
  smallest demonstrated width;
- an upward point regresses throughput: roll back and bracket below it;
- one lucky short window cannot keep a candidate; paired confirmation rejects
  it;
- an upward generation rollover includes all transition traffic in probe cost
  but compares only calls beyond the clean admission-id cutoff;
- a long-latency width-64 candidate cannot score after eight early
  completions; it waits for the required full turnover or becomes tail-limited;
- a candidate never reaches active capacity: mark infeasible and restore;
- a limit decrease drains without cancelling active calls and is measured only
  after activation;
- an intentionally high initial width (64) converges downward to the smallest
  near-peak width without a drain/refill throughput collapse;
- no response causes bounded blind growth only through 32;
- source bandwidth/latency changes mid-load: stale estimates are invalidated
  and the controller converges around the new region;
- a short tail suppresses a probe and restores the last settled width;
- explicit rate limit with deadline pauses new admission and backs off;
- the same slowdown without a provider remains correct and eventually probes
  down from throughput/latency alone;
- fixed mode never changes width.

### DMA configuration/admission tests

- benchmark result converts to a common-block loader config;
- calibrated artifact ownership, schema/fingerprint, confidence, and mapped
  budget survive through the load; fixed config requires only structural
  validation;
- `budget_exhausted` calibrated handoff is rejected until rerun or explicitly
  converted to an operator-fixed tuple;
- heterogeneous devices are either measured at one common block or rejected,
  never silently coerced;
- a block that wins at one reference width but loses after per-block width
  tuning/concurrent verification is re-ranked rather than frozen early;
- authoritative selection can choose widths below eight;
- stable device-id mapping detects missing, duplicate, reordered, and stale
  configurations;
- active events never exceed each device width;
- optional global cap never exceeds its limit;
- a present zero global cap is rejected before constructing the fair gate;
- global-cap search is bypassed below its scaling/latency trigger, and a cap is
  emitted only when throughput/latency plus per-device retention and fairness
  criteria pass;
- fair selection balances calibrated shares, rotates ties, supports caps below
  device count, and lends idle slots;
- uncapped mode bypasses the global gate;
- sharded and replicated final ordering and lease release remain correct.
- planner boundary fragmentation and replica sharing produce the correct
  worst-case mapped-byte feasibility check.

Carry forward the existing DMA distribution/selection/fairness tests at
`zml/io.zig:6866-7067` and pool tests at `zml/mem.zig:448-537`, relocating them
with the extracted modules.

### VFS/data-plane tests

Carry forward and extend:

- one 32 MiB scatter admission, including a scatter list above `IOV_MAX`, is
  one physical GET;
- a retry is serial and physical high-water never exceeds caller admission;
- a throttle/backoff event is visible before the backend sleeps;
- Retry-After reaches the admission pause deadline;
- HTTP-date Retry-After and HF `RateLimit: ... t=` convert to the correct
  monotonic deadline;
- simultaneous 429s pause immediately and resume through paced/jittered retry
  admission rather than one deadline-aligned wave;
- an unclassified generic 503 remains a failure/diagnostic, while an explicitly
  classified overload event applies backpressure;
- feedback is scoped to this load's opened sources, not a backend-global
  counter; an unrelated load cannot consume its events;
- absent feedback works with generic `std.Io`;
- generic `std.Io` defaults to resumable `IOV_MAX` batching; only an explicit
  exact-fill capability uses one scatter call;
- strict Content-Range/exact-fill and credential/redirect rules do not regress.

Relevant existing coverage is in
`zml/io/vfs/http_acceptance_test.zig:205-385`,
`zml/io/vfs/base.zig:211-254`, and
`zml/safetensors.zig:426-665`.

### End-to-end loader tests

Add a controllable fake positional backend and test the real scheduler/gates:

- completed-source metrics are recorded before DMA retirement;
- wrong/old trial generations cannot score a new candidate;
- racing `setLimit`/`acquire` returns a consistent limit generation;
- pool/lifecycle pressure makes a sample ineligible rather than a false source
  regression;
- a full job claimed while blocked on planning/pool admission remains visible
  to tail feasibility and may receive the next generation;
- lowering read width drains naturally;
- fixed preflight DMA limits remain unchanged for the full load;
- peak mapped memory never exceeds `max_pinned_bytes`;
- failure still closes gates, drains events, errors unfinished buffers, and
  unmaps only after callbacks complete;
- multi-device sharding and replication obey per-device/global limits and make
  progress fairly.
- multiple handles in one load share the conservative width/backpressure
  scope, while activity through the same backend in another load cannot affect
  it;
- fixed lifecycle-slack candidates are exercised across local, remote, and
  mid-load low-latency <-> high-latency transitions; the runtime never changes
  slack or consults a source class;
- extraction retains tagged model tensor shapes and their sharding distinctions
  rather than substituting raw safetensor descriptor shapes.

## Performance and acceptance

For each available platform/source, first build a fixed 32 MiB width oracle
under the exact calibrated DMA configuration. Sweep a bounded width set such
as 1/2/4/8/12/16/24/32/48/64 plus the feasible cap when larger, clipped by
pinned feasibility. Use interleaved repeats and report medians; do not mix
measurements from different host-load, cache, NUMA, or backend regimes.

Required acceptance:

- DMA preflight settings are exactly representable and exactly enforced by the
  loader. No live controller changes them.
- On comparable hardware, calibration reproduces the measured ROCm
  8-MiB/eight-event neighborhood and four-B70 4-MiB/eight-per-device/global-4
  neighborhood, or the report identifies host/regime evidence for the change;
  these are regression oracles, never hard-coded defaults.
- On loads with enough full requests to pay the documented transition,
  turnover, confirmation, and exploration budgets, the adaptive result is
  within 3% of the best fixed-32-MiB static source median and selects the
  smallest demonstrated width inside that band. Its end-to-end wall-load
  goodput must also remain within 3% of the best fixed-32-MiB wall-load median
  under the same DMA config. If source-only selection fails the wall-load
  guard, fix structural pacing/feasibility; do not silently feed DMA metrics
  back into the source estimator.
- A tail-limited load that cannot afford convergence is evaluated against the
  fixed warm-start prior, must respect the 25% exploration budget, roll back
  unscored probes, and report that no near-oracle claim was established. It is
  not required to discover an unmeasurable width-64 knee from a short model.
- Calibration time is reported separately. Cached and uncached paths are both
  visible in results.
- A changing-source test recovers to the new near-peak band without exceeding
  hard read, pinned, per-device DMA, or global DMA bounds.
- An explicit throttle stops new admission through the advertised deadline,
  reduces width, and resumes already-admitted calls through bounded,
  paced/jittered retry admission rather than a retry herd.
- Generic `std.Io` with no side channel passes the same correctness suite.
- One admitted remote read remains one physical request, except serial retries.
- Local, sharded, and replicated loads complete without final-transfer or
  ownership regressions.
- One fixed lifecycle-slack policy meets the end-to-end band across steady and
  changing local/remote fixtures; otherwise production enablement is blocked
  pending a separate structural-pacing redesign.
- Controller runtime remains negligible in a load-window profile; source and
  DMA metrics must not introduce a new material flat-profile symbol.

Measurement matrix, where hardware/access is available:

- one-device warm and cold local file;
- oneAPI B70 single-device and four-device sharded/replicated;
- ROCm MI300X single-device and multi-device;
- CUDA build and runtime on the CUDA host;
- real AWS S3/HF and a conforming deterministic HTTP fixture;
- a mid-load source-regime change and rate-limit fixture.

Run narrow tests first, then:

```text
bazel test //zml:test --test_output=errors
bazel test //zml/io/vfs:test --test_output=errors
bazel test //stdx:test --test_output=errors
bazel build //examples/io:playground //examples/mnist
bazel build --config=release --@zml//platforms:oneapi=true //examples/io:playground
bazel build --config=release --@zml//platforms:cuda=true //examples/io:playground
bazel build --config=release --@zml//platforms:rocm=true //examples/io:playground
```

Do not build or modify XLA for this work. Do not edit/build the adjacent
user-owned `llmd` checkout or change its `log.txt` without separate
authorization. Preserve existing `perf.data` recordings, benchmark scripts,
and environment-selector behavior as required by `CTX.md:1916-1976`, including
`ONEAPI_DEVICE_SELECTOR`, `ZML_LOAD_SHARDING`, and caller-selected `PERF_DATA`.
Keep the inherited request-size override only as the explicit test/oracle hook
during migration; production behavior remains fixed at 32 MiB.

## Risks and explicit follow-ups

- **32 MiB source regression:** known possible on warm oneAPI/local paths.
  Measure and report it, but keep size fixed during this controller redesign.
- **Synthetic DMA mismatch:** close it by sharing the submission/callback
  engine before enabling automatic production preflight if callback-path
  comparison shows a material selection difference.
- **Calibration overhead:** local loads can be shorter than a cold preflight.
  Keep explicit handoff and add keyed caching/short confirmation before making
  automatic calibration universal.
- **Mixed sources:** one load-wide controller may be suboptimal for genuinely
  independent sources, but is safe when independence is unknowable. A future
  split requires explicit independence plus domain-aware scheduling and fair
  worker/pinned budgets before jobs claim or reserve resources.
- **Fixed lifecycle slack:** historical local and remote optima disagree. Use
  eight as the conservative starting constant, test regime transitions, and
  block enablement rather than smuggling source classification or another live
  adaptive axis back into the design.
- **No feedback provider:** an opaque rate limit cannot be recognized
  semantically. The throughput/latency controller still adapts, but less
  promptly; logs must say explicit backpressure was unavailable.
- **Fast regime churn:** probing consumes real finite work. Bound search cost
  and prefer a safe settled width over chasing every short fluctuation.
- **Global DMA cap semantics:** the loader and benchmark must share fairness
  code. Merely adding a scalar semaphore repeats the starvation defect
  documented in `CTX.md:20-72`.

## Definition of done

The redesign is complete when:

1. `benchmarkDma` returns an owned, fingerprinted calibration artifact with one
   common block size, per-device event limits, and an optional fair global cap.
2. The loader consumes that configuration immutably and enforces it exactly.
3. Every normal non-tail source job is 32 MiB.
4. The only independently learned live value is active source read
   parallelism; worker/lifecycle bounds may mirror it through fixed formulas
   but are never separately optimized.
5. Its normal decisions use source completion throughput and source latency,
   not DMA/request retirement or static source classification.
6. Explicit rate-limit feedback is optional, scoped to this load, published
   before retry sleep, wakes admission immediately, and coordinates
   already-active retries.
7. The controller tracks a mid-load regime change, selects the smallest width
   near recent peak, and rolls back unfinished tail probes.
8. Existing DmaMapped ownership, exact remote-read, pinned-bound, replica,
   final-transfer, failure-drain, and fairness guarantees remain covered.
9. Fixed-oracle and real-platform results meet the acceptance criteria, with
   calibration overhead and the known 32 MiB tradeoff reported separately.
