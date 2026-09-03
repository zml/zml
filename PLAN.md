# Loader simplification plan

`CTX.md` is the authoritative description of the loader's current design and
measured behavior. This file is the execution checklist. After every task:

1. run the task's focused validation;
2. update the result and status here;
3. update `CTX.md` so it describes the code that now exists;
4. do not begin the next task until those updates are complete.

The compatibility target is behavioral, based on the callers in
`~/github/zml/monorepo`: repeated synchronous `loadExecute`, followed by at
most one `load`/`await` epoch, multi-source `TensorStore` bindings, and
cumulative byte accounting. The old constructor options and exact method
signatures are not constraints; that checkout is intentionally not being
migrated yet.

## Second-pass status

- [x] 0. Record the second-pass plan and refresh the `CTX.md` baseline.
- [x] 1. Flatten epoch plans and remove dead pipeline state.
- [x] 2. Make the lifecycle gate the epoch-completion mechanism.
- [x] 3. Simplify adaptive-runtime and probe-state representation.
- [ ] 4. Specialize DMA calibration for its single measured lane.
- [ ] 5. Remove redundant platform/device identity state.
- [ ] 6. Make `DmaBlockPool` provider-only and simplify block leases.
- [ ] 7. Unify loader request preparation and exact positional scatter.
- [ ] 8. Split the loader implementation into focused modules.
- [ ] 9. Run final validation and reconcile all documentation.

## Second-pass task details

### 0. Baseline — completed

Record the remaining simplifications found after the first pass. Preserve the
two admission gates, generation-safe width changes, NUMA matching, final DMA
ordering, persistent workers, per-tensor PJRT buffers, and physical-byte fair
ordering. No loader policy changes are part of this pass.

Validation: repository and adjacent-monorepo usage audit; clean worktree.

Result: the only required behavioral compatibility remains repeated
synchronous `loadExecute`, then at most one `load`/`await`, multi-source
bindings, ready outputs, and cumulative loaded bytes. Current repository users
also require public adaptive/fixed read parallelism and DMA benchmark options.

### 1. Flat epoch plan and pipeline cleanup

- Publish jobs directly in final fair order instead of retaining a separate
  order array and planning-only predecessor/tensor identity.
- Replace suffix metadata with the scalar source-byte total and remaining job
  count actually consumed by production.
- Remove test-only optionals and redundant maximum-job metadata.
- Remove dead pipeline fields and assertion-only submission bookkeeping.
- Make a DMA lease completion report whether it released the final reference.

Validation: planner/fairness/failure tests and the focused build/test set.

Result: the published epoch now owns jobs physically arranged in final fair
order; planning-only predecessors and the separate order array are discarded.
The source-byte total is scalar and remaining adaptive work is derived directly
from the atomic cursor because every coalesced production job is sampleable.
Removed production-dead tensor IDs, optional source slots, suffix arrays,
per-batch maximum-block metadata, peak-DMA state, the unused DMA-done event,
ready-transfer tensor pointers, and assertion-only pending-submission counts.
`DmaBlockPool.Lease.complete` now reports the final reference, removing a
second completion atomic. Per-device queues use unordered removal. The full
loader-inclusive 233-test suite passes with CUDA dependencies and CPU runtime;
the IO playground builds and VFS/stdx tests pass.

### 2. Lifecycle completion

- Add an empty/drained wait to the lifecycle gate.
- Use scheduler exhaustion plus an empty lifecycle gate as epoch completion.
- Remove the parallel `epoch_jobs` counter/event and tracking flags.
- Preserve controller synchronization and failure draining.

Validation: epoch reuse, active-epoch rejection, callback failure, and focused
integration tests.

Result: `await` now first waits until the immutable scheduler has handed out
every job, then waits until the lifecycle gate has no active requests. Because
a lifecycle credit spans claim through the final DMA callback, that pair is
the epoch-completion condition. Removed the parallel epoch job atomic, drained
event, tracking flag, abandoned-job accounting, and request-local completion
flag. Scheduler failure directly exhausts the plan and wakes waiters; the
controller generation barrier and worker rendezvous still run before metadata
is reaped. Added focused waits-for-final-claim and waits-for-final-request
tests. The loader-inclusive ZML suite, IO playground build, and VFS/stdx tests
pass.

### 3. Adaptive runtime state

- Replace mutually dependent probe booleans with an explicit tagged state.
- Move generation/evidence validation into the measurement runtime so the
  controller remains a width-selection policy.
- Make mutex-protected probe counters non-atomic and remove redundant telemetry
  structures and reporting atomics.
- Remove nonpersistent controller branches unused by production.

Validation: controller, generation, tail, blind bootstrap, settled backoff,
and allocation-failure tests.

Result: the runtime's four mutually dependent probe booleans plus separate
pending limit/evidence storage are now one tagged measurement state:
inactive, transitioning, blind, measuring, or scoring. The measurement layer
validates generation, duration, request count, and exercised concurrency before
the controller sees evidence; the controller is only width-selection policy.
Probe counters are plain fields under their existing mutex, aggregate feedback
is reduced directly to a backpressure bit, and the selected width is published
through the epoch barrier without an extra atomic. Removed the unused
nonpersistent finalization/tail branches. The loader-inclusive ZML suite, IO
playground build, and VFS/stdx tests pass.

### 4. Single-lane DMA calibration

- Specialize benchmark windows and candidate measurement for the one
  representative device that is now measured.
- Remove lane slices, one-element result allocations, nullable fixed width,
  cloned unmeasured recommendations, and dynamic three-sample storage.
- Correct public documentation to say that one representative device is
  measured while every device allocator is warmed.

Validation: benchmark decision tests and focused builds.

### 5. Platform settings identity

- Remove device kind/ID ownership and canonicalization from settings already
  owned by a stable `Platform`.
- Retain topology, block size, per-device DMA width, mapped budget, platform
  identity, and heterogeneous-device rejection.

Validation: settings validation and platform-focused tests.

### 6. Provider-only DMA pool

- Convert allocator-backed pool tests to the existing arena provider fixture.
- Remove the optional provider/slab allocator mode and owned-slab branches.
- Remove the externally refreshed-arena path, which has no production or
  monorepo caller after staged calibration.
- Carry the NUMA node in leased block handles so release does not reverse-scan
  arenas.

Validation: pool growth, matching, close, reuse, and allocation-failure tests.

### 7. Shared loader front end and scatter helper

- Prepare model traversal, source lookup, sharding, and `loadExecute` inputs
  once above the buffered/direct backend split.
- Extract the duplicated exact positional scatter loop into one shared helper.
- Bundle epoch diagnostics and remove unused public writer surface.
- Define and test consistent successful-epoch byte accounting.

Validation: public compatibility workflow, safetensor positional reads, VFS,
and both loader implementations where supported.

### 8. Module split

After state cleanup, split calibration, dispatch/planning, and direct-loader
implementation out of the monolithic `zml/io.zig`, keeping tests beside the
code they exercise and preserving the public `zml.io` surface.

Validation: formatting, focused builds, and all affected Zig tests.

### 9. Final validation

Run formatting, the IO playground build, VFS/stdx tests, and the loader-inclusive
ZML suite with the platform flags needed to avoid the unrelated missing
FlashInfer module. Record any independent default-configuration blocker.

## Status

- [x] 0. Record the plan and reconcile `CTX.md` with the current implementation.
- [x] 1. Correct coalesced DMA block accounting and pinned-width feasibility.
- [x] 2. Simplify admission gates and make settled backoff generation-safe.
- [x] 3. Remove the tensor-local loader path and decision-dead calibration code.
- [x] 4. Make loader epochs immutable and precompute fair job order.
- [x] 5. Flatten final DMA transfer records once during planning.
- [x] 6. Simplify adaptive-controller and runtime bookkeeping.
- [x] 7. Run final validation and reconcile all documentation.

## Task details and completion log

### 0. Plan and documentation baseline — completed

Write the current simplification sequence down and correct stale claims in
`CTX.md`: the global DMA cap and old writer/pool APIs are gone, calibration
arenas are staged, settled source backoff exists, and the current planner
stores logical source pieces rather than final block transfer records.

Validation: documentation inspection and repository-wide symbol search.

### 1. Exact DMA block accounting — completed

- Replace `ceil(request/block) + device_count - 1` and writer-group estimates
  with the coalesced path's exact `ceil(max_job_len/block)` bound.
- Use one calculation consistently for workspace validation, arena reserves,
  worker scratch, aggregate feasibility, and strict-NUMA feasibility.
- Add regression coverage demonstrating that device count does not inflate
  the number of blocks needed by one coalesced source job.

Validation: focused Zig tests, `//examples/io:playground`, and `//vfs:test`.

Result: added `maximumCoalescedJobBlocks` as the shared bound and removed all
device/writer-count inflation from platform workspace validation, retained
arena growth, loader feasibility, and worker scratch sizing. Added a regression
that an eight-device shared-NUMA configuration needs eight feed blocks rather
than the obsolete nine-block request estimate. `//examples/io:playground`
builds and `//vfs:test` passes. The inline Zig test is compiled by the broad
`//zml:test` target, which remains independently blocked by the missing CUDA
FlashInfer module recorded in Task 7.

### 2. Admission and backpressure — completed

- Remove `worker_gate`; retain `request_gate` for complete request lifecycles
  and `read_gate` for source calls/generation drains.
- Make mutex-protected gate fields non-atomic where possible.
- Apply every changed source width, including settled backoff, through a clean
  close/drain/telemetry-reset/reopen transition.
- Prevent old-generation feedback from stepping down more than once before
  the new width has handled work.

Validation: gate/controller unit tests plus the focused build/test set.

Result: removed `worker_gate` from the loader, pipeline, controller runtime,
workers, failure shutdown, and tests. All persistent workers now compete for
the lifecycle gate, whose one extra credit can stage work while the read gate
limits source calls. Gate limit/closed fields are plain mutex-protected values.
Changed settled widths now close and drain the read gate, reset telemetry at
the boundary, and require a new-generation admission before another backoff;
feedback consumed during the drain cannot cause another step. Added focused
backoff-boundary coverage. `//examples/io:playground` builds and `//vfs:test`
passes.

### 3. Dead paths and calibration — completed

- Remove the unused tensor-local `VectoredReadRequest.run` path,
  `FairVectoredReadScheduler.init`, and their private support machinery.
- Move valuable sharding/replication assertions from `VectoredRequestPlan`
  tests to the production planning representation before deleting it.
- Remove the aggregate DMA timing phase now that it cannot affect selection;
  retain all-device workspace preparation and a cheap correctness warm-up if
  required by the PJRT path.
- Remove private identity wrappers and unreachable compatibility helpers found
  during the same pass.

Validation: planner/calibration tests plus the focused build/test set.

Result: deleted `VectoredRequestPlan`, the tensor-local request runner and its
allocation/enqueue helpers, the unused non-appendable scheduler constructor,
and the no-longer-needed borrowed tensor reader state. Reworked the existing
replicated, packed, 1D, 2D, and 3D tests to validate the production
`DispatchSpans` traversal across request and block boundaries. Removed the
aggregate calibration phase, per-device aggregate source carving, unused
nonrepresentative cohorts, the synthetic distribution wrapper, and several
private dead/identity helpers. Calibration now tunes one representative device,
warms allocators on all devices, clones the selected tuple, and grows the
all-device retained working set. `zml/io.zig` fell from 8,495 lines at review
start to 7,600. `//examples/io:playground` builds; `//vfs:test` and
`//stdx:test` pass.

### 4. Immutable epochs — completed

- Reject a second `load` while an epoch is active. Keep persistent workers for
  inexpensive sequential `loadExecute` epochs.
- Build and publish one immutable epoch plan.
- Simulate the deterministic physical-byte fairness policy during planning and
  store a predecessor-safe job order.
- Replace runtime queues/cursors/claimed bookkeeping and claim mutex work with
  an atomic epoch cursor and immutable remaining-work metadata.
- Add the real compatibility workflow test:
  `loadExecute -> loadExecute -> load -> await -> cumulative bytesLoaded`.

Validation: scheduler and loader integration tests plus the focused build/test
set.

Result: both loader backends now reject a second `load` while an epoch is
active. The direct planner computes the physical-byte fair, predecessor-safe
order once and publishes one owned immutable plan. Runtime queue scans,
per-device cursors/debt, claimed flags, append/seal/reopen states, piece-batch
ownership, and the claim mutex were replaced by one atomic position plus
precomputed remaining-work suffixes. Persistent workers rendezvous at the epoch
barrier before the plan is released, then wait for the next sequential epoch.
Added the compatibility workflow test covering two synchronous `loadExecute`
calls followed by `load`/`await`, ready output, active-epoch rejection, and
cumulative `bytesLoaded`. It also exposed and fixed structural placement
comparison that had relied on unsupported struct `!=`. The full 233-test ZML
suite passes with CUDA dependencies and the CPU runtime enabled; the focused
playground build and VFS/stdx tests pass.

### 5. Final transfer planning — completed

- During the planner's existing dispatch-span walk, emit final records carrying
  item, block index/offset, writer mask, destination offset, and length.
- Make workers initialize destinations, lease/read blocks, and enqueue those
  records without rebuilding transfers or re-walking dispatch spans.
- Remove logical/source transfer state that becomes redundant.

Validation: 1D/2D/3D, sharded, replicated, overlap, duplicate, packed-dtype,
and failure-drain tests plus the focused build/test set.

Result: the coalescing planner now emits the final item/block/writer-mask/
destination records while it already has each tensor's `DispatchSpans`. Those
same records drive physical-byte fairness and are retained in the immutable
epoch plan. Workers only initialize referenced destinations, derive per-block
reference/NUMA/queue counts, acquire and read blocks, then publish the records;
their transfer `ArrayList`, dispatch traversal, block splitting, and associated
allocation/error branches are gone. Runtime tensor state no longer owns a
second `DispatchSpans`, and an unused single-block enqueue path was removed.
The multidimensional, mirrored/folded, replicated, and packed-dtype tests now
exercise the production final-record helper directly. All 233 ZML tests pass
with CUDA dependencies plus the CPU runtime enabled.

### 6. Controller and bookkeeping cleanup — completed

- Reduce the adaptive source controller to ramp-up, refine-down, and settled,
  retaining the smallest-within-3%-of-peak rule and finite-tail handling.
- Use at most one adjacent confirmation rather than the six-phase alternating
  pair protocol.
- Remove write-only metrics and the request/block timestamps and branches that
  exist solely to maintain them.
- Replace singleton read-stat arrays with one optional cursor.
- Move `DmaBlockPool.acquireMany` matching arrays into reusable scratch so a
  source job performs no allocator calls in the steady state.
- Remove productionless pool modes only where repository and monorepo searches
  confirm they are not public compatibility requirements.

Validation: controller, pool, allocation-failure, and focused integration
tests; compare planning/load diagnostics against the recorded baseline.

Result: replaced the baseline/upward/downward/pair-reference/pair-candidate
controller with ramp-up, refine-down, and settled phases. It still chooses the
smallest measured width within 3% of peak, handles finite tails, and now permits
only one extra measurement of an adjacent borderline candidate. Removed
write-only latency, byte-residency, high-water, and pool-wait metrics together
with the request/block/event timestamps and success branches maintained solely
for those metrics. Aggregate VFS feedback now uses one optional cursor rather
than one-element arrays. Each worker owns reusable pool matching scratch, so
`acquireMany` has no steady-state allocator calls and remains safe when several
callers wait concurrently. Removed the unused direct-DMA construction mode;
production pools are arena-provider backed and the allocator-backed path is
test-only. Added reuse and exhaustive allocation-failure tests for pool
scratch. The playground builds, VFS/stdx tests pass, and the full ZML suite
passes with CUDA dependencies plus the CPU runtime enabled.

### 7. Final validation — completed

- Run `zig fmt` on changed Zig files.
- Run `bazel build //examples/io:playground`, `bazel test //vfs:test`,
  `bazel test //stdx:test`, and `bazel test //zml:test`.
- If the broad ZML test remains blocked by the existing missing CUDA
  FlashInfer module, record the exact failure without treating it as loader
  validation.
- Ensure `CTX.md` contains only the final current design, measurements,
  compatibility contract, and genuinely open work.

Result: `zig fmt --check` passes for both changed Zig files;
`//examples/io:playground` builds; `//vfs:test` and `//stdx:test` pass. The
loader-inclusive ZML suite passes all 233 tests (230 passed, three skipped)
with `--@zml//platforms:cuda=true --@zml//platforms:cpu=true`. As before this
work, the default `//zml:test` configuration does not compile because
`zml/moe/cutlass_flashinfer.zig` imports the unavailable
`platforms/cuda/flashinfer_cutlass_moe` module; this is unrelated to the loader.
The final symbol and adjacent-monorepo audit found no removed loader state still
referenced by current code or by the behavioral compatibility callers.
