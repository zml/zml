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

## Status

- [x] 0. Record the plan and reconcile `CTX.md` with the current implementation.
- [x] 1. Correct coalesced DMA block accounting and pinned-width feasibility.
- [x] 2. Simplify admission gates and make settled backoff generation-safe.
- [x] 3. Remove the tensor-local loader path and decision-dead calibration code.
- [x] 4. Make loader epochs immutable and precompute fair job order.
- [x] 5. Flatten final DMA transfer records once during planning.
- [x] 6. Simplify adaptive-controller and runtime bookkeeping.
- [ ] 7. Run final validation and reconcile all documentation.

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

### 7. Final validation — pending

- Run `zig fmt` on changed Zig files.
- Run `bazel build //examples/io:playground`, `bazel test //vfs:test`,
  `bazel test //stdx:test`, and `bazel test //zml:test`.
- If the broad ZML test remains blocked by the existing missing CUDA
  FlashInfer module, record the exact failure without treating it as loader
  validation.
- Ensure `CTX.md` contains only the final current design, measurements,
  compatibility contract, and genuinely open work.
