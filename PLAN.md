# Adaptive-concurrency merge plan

Status: merge completed; conflicts and in-repository correctness checks resolved;
hardware-specific runtime/performance acceptance remains a follow-up

Merge inputs:

- feature: `brabier/adaptive-concurrency` at `87d7df89` (`simpler dma`)
- target: `master` at `e1e983c8`
- fork point: `d6489fbb` (`zml/attention: implement batching in cu_fa2 backend (#641)`)
- audit merge working change: `mkkktrwz`, created with
  `jj new brabier/adaptive-concurrency master`

The merge began with 20 conflicted paths, but a correct integration requires
more than selecting conflict sides. Master
has replaced or moved several APIs that the feature branch was built around,
and several branch-only files merge cleanly into the wrong location or retain
behavior superseded by master.

## Outcome

This is not a mechanical merge. The feature data plane is worth preserving:

- fixed-size positional source jobs;
- shared source handles and fair, vectorized read scheduling;
- bounded `DmaBlockPool` ownership;
- one admitted remote positional read mapping to one physical Range GET, with
  serial retries;
- separate source, retained-lifecycle, and DMA limits;
- callback-driven final transfers and failure draining;
- DMA calibration and fair global admission;
- source-only adaptive read width.

Master must remain authoritative for:

- the current `TensorStore` and `zml.io.Loader` public surface;
- multi-source/fused tensor support;
- tagged model shapes and packed device shapes;
- `Compiler.zig` and current model implementations;
- the promoted `//vfs` package;
- current CUDA/ROCm/PJRT artifacts and platform state;
- current GCS credential handling;
- current compiler, quantization, donation, and MoE behavior.

The recommended integration strategy is therefore to port the feature
subsystems onto master, not to accept either complete side of `zml/io.zig`,
`zml/platform.zig`, the model conflicts, or the old VFS deletion conflicts.

## Merge decisions

The following choices are fixed for this merge. They are no longer alternatives
for conflict resolution; follow-up designs may change them in later commits.

### D1. Loader API and unit of scheduling

Master commit `577d6452` replaced the one-shot top-level loader with
`zml.io.Loader`. Master callers pre-bufferize models, call `loader.load`, and
then `loader.await`; `Loader` also owns `loadExecute` for multi-source/fused
tensors. The feature head reintroduces a synchronous top-level `zml.io.load`
that bufferizes internally and schedules the whole model through one adaptive
controller.

The feature pipeline needs the complete tensor set to build its fair job
schedule, mapped-memory bound, DMA device set, and one load-wide controller.
Trying to hide it behind master's current per-tensor asynchronous callback
without changing the contract would either fragment one load into many
controllers or defer all real work until `await`.

Decision:

1. Keep master's `TensorStore` and `Loader` public types for compatibility.
2. Extract the feature implementation into one synchronous model-wide
   `loadInto` operation that accepts preallocated buffers and returns errors.
3. Keep a one-shot `load` convenience wrapper that bufferizes once and calls
   `loadInto`; do not duplicate the vectorized implementation.
4. Migrate in-tree production model loaders to the model-wide operation while
   preserving their current buffer ownership and unload behavior.
5. Keep `Loader.load`, `Loader.await`, and `Loader.loadExecute` for source
   compatibility. Keep the progress support from `031b2a56` and the
   `.wait = true` lifetime fix from `d539f6bd`. Preserve master's accelerator
   `DirectMemoryWriter` path and its device tests; the model-wide vectorized
   path supplements rather than replaces it.

Do not defer a model-wide operation into `Loader.await`: the merge uses an
explicit synchronous borrow lifetime and direct error propagation.

### D2. Multi-source/fused tensors

Master's `TensorStore` maps a tensor id to a slice of copied source descriptors
and has `maybeCreateBinding`. The feature branch maps an id to one descriptor;
its vectorized path calls `getPtrFromId` and
`getBorrowedPositionalReaderById`. Simply keeping master's `TensorStore` makes
the feature code fail to compile, and using master's `getPtrFromId` assertion
would panic for a multi-source tensor.

Decision for this merged version:

- preserve master's store without mutation;
- vectorize single-source tensors;
- skip multi-source tensors in `loadInto` without calling a single-source
  accessor, and leave explicit `loadExecute` callers in charge of materializing
  them;
- require model-wide loading to complete before an external caller invokes
  `loadExecute`, and retain its synchronous `.wait = true` barrier;
- account `loadExecute` bytes/progress exactly once;
- add a later planner extension only if fused tensors need vectorized loading.

Never restore the feature store's behavior of mutating registry descriptor
shapes in place. Master intentionally copies descriptors before applying tags
and partitioning.

### D3. DMA settings ownership and calibration policy

The old plan and the feature head disagree:

- The old plan required a caller-owned, fingerprinted calibration artifact
  produced after compilation/warm-up and passed explicitly to the loader.
- The feature head installs conservative settings in `Platform.init`, retains
  one platform-owned mapped workspace, lets loads consume defaults
  automatically, and makes `Platform.benchTransfer` an optional atomic
  replacement. The README documents this latter behavior.

Decision: keep the feature-head platform-owned contract. Calibration remains
optional, defaults are retained by `Platform`, and successful calibration
atomically replaces them. The accepted consequences are:

- calibration is optional rather than required;
- uncalibrated 4 MiB/eight-event defaults become production behavior;
- retained mapped arenas live until platform destruction or replacement;
- only one calibration/load/inspection operation may borrow the workspace at
  once; another operation returns `error.DmaWorkspaceBusy`;
- a configuration is tied to the platform pointer and current device kind,
  but has no persistent plugin/driver/model fingerprint;
- the benchmark calibrates all addressable devices using a full-block
  synthetic distribution, not the devices and tail distribution of one
  model.

DMA workspace setup is best effort during `Platform.init`: a setup failure is
logged and leaves settings absent, but must not make `Platform.auto` silently
skip an otherwise usable accelerator. A later direct model-wide load then
returns the explicit `DmaResourcesRequired` error; buffered loading and the
legacy `Loader` remain usable, and `benchTransfer` may publish working settings.
No second caller-owned calibration artifact is introduced in this merge.

### D4. Uniform versus per-device DMA limits

The old plan called for one common block size plus per-device widths. The
feature head deliberately stores one uniform `max_in_flight_per_device` and
chooses a uniform block size/width across all participating devices. The fair
global cap is optional.

Decision: accept the uniform tuple as the supported contract, validate that
every used device is represented, and retain the fair global cap. Per-device
widths require changing config ownership, workspace reserves, admission
weighting, logging, and tests; they are not part of this merge.

Document that heterogeneous device kinds remain rejected and re-run
multi-device calibration on master before treating uniform settings as
authoritative.

### D5. Which adaptive controller is being merged

The feature head implements a smaller source-width controller, but it is not
the controller described by the old plan. At `87d7df89` it:

- drains active source reads to zero between scored generations;
- becomes terminal once `phase == .settled` and does not track a mid-load
  regime change;
- reads backend-global cumulative VFS statistics to invalidate trials;
- uses static `high_latency` hints for blind bootstrap;
- uses lifecycle slack eight for a high-latency source and zero otherwise;
- has no load-scoped Retry-After/backpressure sink or coordinated retry gate;
- changes the config epoch separately from gate admission, relying on full
  drains (except deliberately unscored blind growth) for attribution.

Those are known behavioral choices, not merge artifacts. Decision: merge this
controller as documented v1, retain the feature's adaptive default for the new
model-wide API, and keep the fixed-width escape hatch. Do not claim load-scoped
backpressure or post-settlement regime tracking. Fixed mode must pass first in
validation, and MNIST remains fixed at width one.

### D6. Platform-wide workspace concurrency

The branch adds `Platform._dma`; master adds the `Platform.state` union used by
CUDA FlashInfer/CUTLASS MoE resources. Both must exist. DMA teardown must run
before PJRT client teardown, and CUDA state teardown must also run before the
arena disappears.

Decision: the advanced model-wide operation remains synchronous. Its exclusive
platform DMA borrow starts only after used-device discovery and ends after all
source workers, PJRT callbacks, failure drains, and block leases have completed.
Calibration and settings inspection use the same operation gate. The legacy
`Loader` retains its existing independent lifetime and does not borrow this
workspace.

### D7. oneAPI command-buffer override

The only feature change to deleted `zml/module.zig` comments out the oneAPI
override that disables XLA GPU command buffers. Master moved the code to
`zml/Compiler.zig` and still sets the override to an empty string.

Decision: keep master's `Compiler.zig` and its current oneAPI command-buffer
override. Do not port this unrelated one-line feature change without a separate
oneAPI compile/runtime regression. Branch performance measurements may have
used command buffers, so they are not merge acceptance numbers.

## Required correctness adaptations

These are not optional design choices.

### Model shape, sharding, and packed dtypes

Master commit `f4ffbbc0` fixed loading to use the tagged model tensor shape
rather than the raw checkpoint descriptor. The feature head currently assigns
`const shape = reader.tensor.shape` in `VectoredTensorTransfer.init`, losing
master's tags and partitioning. Use `tensor.shape()` for dispatch spans,
sharding selection, output metadata, progress totals, and logical placement.

Master also supports packed sub-byte dtypes. `Buffer.from` creates PJRT buffers
from `shape.packedShape()`, while the feature vectorized path currently builds
`pjrt.ShapeSpec` directly from `placement.shape`. The vectorized manager must
use the packed placement shape for PJRT allocation while retaining the logical
model shape on the `Buffer`. Verify byte offsets and final-transfer lengths for
FP4/I2/I4/U2/U4 rather than assuming an element shape is a byte shape.

Cover at least:

- tagged/partitioned tensor shape differing from the safetensor descriptor;
- sharded and replicated placement;
- packed FP4 and another sub-byte dtype;
- scalar and tensor tails;
- master quantization models that have an optional/fused `lm_head`.

### Request reservation bound

`requiredDmaWorkspaceBytes` and the initial `ensureLoadBlockReserves` call
approximate the worst request as
`ceil(32 MiB / block_size) + device_count - 1`. The real
`VectoredRequestPlan` can split at repeated writer-mask/dispatch-span
boundaries. Although the scheduler computes `maximumBlocksPerJob`, the feature
currently does so only after reserve growth. Grow/refresh the workspace again
from the exact maximum before admissions and use that value for load-specific
validation. Otherwise a configuration can validate and still fail at runtime
with `DmaMappedBudgetExceeded`, especially for fragmented sharding.

The feature also computes `minimumStrictAffinityRequestWidth` but only logs
it. When strict NUMA affinity is active, admission must use the minimum of the
aggregate and strict widths; an aggregate block count can report a width that
no eligible node can satisfy.

### PJRT completion errors and ownership

Master commit `49a12235` propagates ready-event failures and validates opaque
pointers. Preserve the feature callback behavior that records a callback
error, and keep every returned `pjrt.Error` and `pjrt.Event` alive until it is
deinitialized exactly once. Preserve the failure path that calls
`setBufferErrorUnknown` for managers that never received a final transfer.

Test synchronous `transferData` failure, asynchronous event failure,
callback-registration failure, allocation failure while retaining the event,
and cancellation/failure drain. No host block or final buffer may be released
before all relevant callbacks complete.

### Current master platform behavior

While merging `zml/platform.zig`, retain:

- `State` initialization and CUDA MoE registration/deinitialization;
- `zml.Compiler.Options` and current compile entry points;
- CUDA `gpu_async_dispatch = true` and its `use_tfrt_gpu_client` named value;
- current physical mesh/sharding setup;
- current allocator/client/arena teardown order.

Per D3, `initPlatformDma` failure is logged and leaves DMA settings absent; it
does not unwind or reject the otherwise usable platform. All ordinary fatal
initialization paths must still deinitialize CUDA state, the physical mesh,
PJRT client, and arena in their ownership order.

The branch predates CUDA async dispatch, so CUDA calibration and load results
must be re-measured with master's setting.

ROCm needs a functional check, not merely a successful link. The branch points
at a local ROCm PJRT archive, and `CTX.md` records that the configured SHA did
not match the local archive during at least one measurement, so Bazel used a
cached artifact with DmaMap-range tracking. Master uses a different public
ROCm 7.14 artifact. Confirm that `PJRT_Client_DmaMap` is implemented and that
the direct loader still bypasses PJRT host linearization/staging; otherwise
the apparent merge can be correct at the Zig level and lose the intended ROCm
data path.

## Textual conflict resolution map

| Paths | Resolution | Risk |
|---|---|---|
| `zml/io.zig` | Keep master's `TensorStore`, `Loader`, `loadExecute`, imports, and current utility fixes. Port the branch's vectorized planner/pipeline, DMA benchmark/state helpers, fixed 32 MiB scheduler, and controller behind the D1/D2 API decision. Do not select an entire side. | Highest; API, ownership, shapes, and failures intersect here. |
| `zml/platform.zig` | Compose master's `State`/Compiler/CUDA changes with the branch's transfer config, `_dma` state, warm-up, benchmark, settings inspection, init, and teardown. | High; platform lifetime and concurrent borrowing. |
| `examples/io/main.zig` | Start from master so `tree`, current writers, and promoted VFS remain. Port `dma-bench`, environment parsing, warm-up, adaptive/fixed read controls, and the chosen model-wide load API. Keep one internally consistent streaming-writer construction/flush API. Deinitialize the returned buffer tree before `Platform`. | Medium; it is also the calibration/benchmark harness. |
| Four `examples/llm/models/*/model.zig` conflicts | Keep all master model, quantization, optional `lm_head`, unload, and compiler changes. Replace only the old `Loader.init/load/await` blocks with the chosen model-wide load call and accounting. | High if an entire feature side is selected; it would resurrect obsolete model code. |
| `examples/mnist/mnist.zig` | Keep master model/compile code and adapt only loading. Preserve fixed read width one for the smoke test unless an adaptive test is intentional. | Low after D1. |
| `platforms/rocm/rocm.bzl`, `packages.lock.json` | Keep master completely. Do not port the branch's local `file:///home/brabier/github/openxla/...` PJRT URL/SHA or old ROCm 7.2.2 lockfile edits. | Critical supply-chain/reproducibility issue. |
| `zml/module.zig` deletion conflict | Keep deleted; master uses `zml/Compiler.zig`. Review the oneAPI command-buffer line under D7 as a separate patch. | Medium performance/stability decision. |
| old `zml/io/vfs/*` deletion conflicts and `zml/io/vfs.zig` | Keep the old paths deleted. Port behavior to root `vfs/*` and `vfs/vfs.zig`; never resolve these by restoring the old package. | High because a clean-looking merge can leave duplicate VFS stacks. |

The merge initially reported these 20 conflicts; all are now resolved:

```text
examples/io/main.zig
examples/llm/models/lfm2/model.zig
examples/llm/models/llama/model.zig
examples/llm/models/qwen3_5/model.zig
examples/llm/models/qwen3_5_moe/model.zig
examples/mnist/mnist.zig
platforms/rocm/packages.lock.json
platforms/rocm/rocm.bzl
zml/io/vfs/BUILD.bazel
zml/io/vfs/base.zig
zml/io/vfs/file.zig
zml/io/vfs/gcs.zig
zml/io/vfs/hf.zig
zml/io/vfs/http.zig
zml/io/vfs/index.zig
zml/io/vfs/s3.zig
zml/io/vfs.zig
zml/io.zig
zml/module.zig
zml/platform.zig
```

### Post-resolution semantic audit findings

Four correctness problems were not represented by unresolved conflict markers
and required manual integration work:

- The first `zml/io.zig` resolution retained the public `Loader` but silently
  omitted master's CUDA/oneAPI `MemoryWriter`, `DirectShardWriter`, and
  `DirectMemoryWriter` implementation and tests. They are restored, and
  `Loader.loadSingleInner` again selects that path exactly as master did.
- `examples/io/main.zig` combined the constructor half of one stdout writer API
  with the flush half of another. The result built but failed at runtime with
  `NotOpenForWriting`/`WriteFailed`. All commands now use a consistent
  streaming writer and explicitly flush before returning.
- A late asynchronous PJRT event error could leave queued managers waiting
  forever after the completion callback had already begun draining. The
  callback now aborts the queues, the completion signal is emitted only after
  draining, and managers without a completed final transfer are marked failed.
- `mem.bufferize` could leak already-created buffers when a later recursive
  field failed. Rollback now covers structs, slices (including const slices),
  arrays, unions, and optionals; `mem.deinitBufferized` is the matching generic
  cleanup API used by model-wide loading, examples, and tests.

The audit also removed an `unreachable` in uncancelable admission: a closed
`LimitedGroup` now returns its declared `Cancelable` error, with regression
coverage.

## VFS relocation and behavioral merge

Master commit `2b9320fe` promoted `zml/io/vfs` to `//vfs`. Jujutsu does not
carry the branch modifications through that move: the audit merge adds a
second old-path package while leaving master's root package largely unchanged.
This is the largest silent merge problem.

Port the following into root `vfs/`:

- `Backend`, `ReadHints`, typed `ReadStats`, and provider plumbing from the
  feature `base.zig`;
- `registerBackend` and `readProfileForPath` in `vfs/vfs.zig`;
- backend `backend()` constructors;
- `range_read.zig` and exact Range validation;
- the HTTP acceptance test and a root `//vfs:test` target;
- HTTP, HF, S3, and GCS serial retry/statistics changes;
- the single-call exact-fill path used by remote positional reads.

Then remove the old root `vfs/parallel_read.zig` and its BUILD entry. Because
the branch deletion occurred at the pre-promotion path, it otherwise survives
the merge without a conflict and can silently restore hidden fan-out.

Preserve master commit `de134385` when porting GCS:

- unknown credential types return an error instead of `unreachable`;
- missing or invalid credential files fall back safely;
- diagnostics retain the current behavior.

Use `pub const VFS = @import("vfs")` from `zml/io.zig`. Remove the branch's
`@import("io").VFS`, old `index.zig`, old BUILD package, and old log-scope path
names where practical.

### VFS/controller scope problem

The branch's `ReadStatsProvider` is backend-instance global. A load snapshots
it by profile id, so concurrent unrelated traffic using the same registered
backend can mark a probe dirty. Retry-After is consumed inside backend retries
and cannot immediately close admissions for the affected load.

For the selected v1 controller, backend-global statistics are diagnostic only;
they must not invalidate trials or change the selected width. The robust
follow-up is the old plan's load-scoped event sink and retry admission gate.

## Non-conflicting changes that still need review

These changes auto-merge or add files cleanly, but are not automatically safe.

### `zml/mem.zig`

Retain the feature's DmaMapped allocator, transparent huge-page alignment,
NUMA-aware `DmaBlockPool`, atomic whole-request acquisition, and
reference-counted leases. Re-run all allocation-failure tests against master's
current platform and packed-shape code. Verify that retained provider arenas
are not unmapped by a per-load pool and that platform teardown happens only
after all leases are returned.

### `zml/safetensors.zig`

Retain exact positional scatter reads, bounds/overflow checks, local
`IOV_MAX` batching with short-read resumption, and remote single-exact-fill
mode. Retain the current v1 policy choice
(`batch_iovecs = !high_latency`) for this merge and document that static
latency is standing in for an opened-source capability. Replacing it with an
explicit capability is a follow-up.

### `stdx/Io.zig`

The resizable `LimitedGroup`, uncancelable admission, and direct-call helper
merge cleanly and also affect master's existing Loader. Run `//stdx:test` and
add stress coverage for concurrent limit changes/cancellation. Ensure no
master caller assumes `limit` is a plain field or relies on all waiters being
woken by a growth.

### `pjrt/pjrt.zig`

`setBufferErrorUnknown` merges beside master's event-error changes. Keep it as
a narrow helper and validate it against every enabled PJRT C API version.

### oneAPI package files

The feature and master versions of the oneAPI package/Bazel files are
identical, so no special conflict resolution is needed. Runtime results still
need revalidation because D7 and the loader API differ.

### Documentation and benchmark artifacts

`CTX.md`, `RESEARCH.md`, lowercase `plan.md`, and four shell scripts are
branch-only additions and therefore merge without review pressure. Decide
whether they belong on master:

- `CTX.md` is useful performance provenance but is large and names historical
  code locations.
- lowercase `plan.md` and `RESEARCH.md` are superseded design inputs.
- `bench_s3.sh` hard-codes a Nix-store JRE and `$HOME/s3proxy` layout.
- `bench_file.sh` and `profile_file.sh` assume a local oneAPI/S3 proxy setup.
- `bench_aws.sh` targets a specific bucket and computes `ZML_LOAD_SHARDING`
  without passing it to the playground command, so the override is currently
  ineffective.

Decision: retain `CTX.md`, `RESEARCH.md`, lowercase `plan.md`, and the benchmark
scripts as historical/experimental branch provenance for this merge, but do
not treat them as current API documentation or acceptance results. Fix the
missing sharding propagation and replace machine-specific Java/model paths
with required environment variables before calling the scripts supported.
Update README examples to the selected optional platform-owned calibration and
model-wide API.

`examples/llm/main.zig` also auto-merges the branch's `registerBackend` calls
and allocator warm-up. It will not compile until the root VFS port supplies
that API. Preserve the warm-up before production loading, and make the final
calibration policy explicit there rather than relying on the playground to be
the only calibrated caller.

## Master changes whose behavior must survive

Use these commits as semantic checks while resolving and reviewing:

| Master commit | Behavior to preserve |
|---|---|
| `577d6452` | current `Loader`, model call sites, and buffer ownership |
| `031b2a56` | progress feedback for `loadExecute` |
| `f4ffbbc0` | use model tensor shape, not source descriptor shape |
| `d539f6bd` | wait for `loadExecute` before freeing input buffers |
| `2b9320fe` | root `//vfs` package and imports |
| `de134385` | robust invalid GCS credential handling |
| `cd477b85`, `fcb0d2a9` | `Compiler.zig`, current compiler/sharding APIs |
| `067602b2`, `a46e4cac` | device/packed byte-size and packed FP4 behavior |
| `49a12235` | PJRT ready-event error propagation and pointer validation |
| `62a4d3c6` | CUDA async dispatch default |
| `3e589c97` | ROCm 7.14 packaging and public PJRT artifact |
| `e1e983c8` | CUDA FlashInfer/CUTLASS MoE platform-owned state |

## Integration sequence

Keep the merge reviewable even if the final result remains one merge commit.

### 1. Resolve package and platform scaffolding

- keep master ROCm files and delete the old VFS paths;
- port feature VFS behavior to root `vfs/`;
- compose master `Platform.state` with feature `_dma` state;
- retain current Compiler namespace and CUDA named values;
- build `//vfs` and the platform-neutral ZML library before touching callers.

### 2. Apply D1-D7

- implement the selected model-wide loader entry point;
- define multi-source fallback;
- preserve platform-owned defaults and optional calibration;
- enforce uniform DMA width;
- document controller v1 and keep backend-global stats diagnostic-only;
- define workspace borrow lifetime;
- retain the oneAPI command-buffer override.

No caller migration should precede these decisions.

### 3. Port the vectorized core

- integrate feature planner, scheduler, gates, pool, callbacks, and DMA
  benchmark with master's `TensorStore` and buffers;
- use tagged logical shapes and packed PJRT shapes;
- compute reservation feasibility from actual plans;
- preserve fixed 32 MiB full requests and tensor tails;
- preserve one-GET remote semantics;
- retain a fixed read-width mode;
- test fixed mode before adaptive mode.

### 4. Migrate callers without reverting master model work

- update the playground and `dma-bench` command;
- update MNIST and all four LLM model loaders at the smallest possible code
  span;
- retain optional/fused weights and current unload helpers;
- update README and logging after the final API is fixed;
- apply the historical docs/scripts disposition.

### 5. Enable and validate adaptive mode

- run deterministic controller tests;
- run local and mock-remote exact-read tests;
- exercise failures and cancellation;
- compare fixed widths with adaptive selection;
- only then make adaptive mode the in-tree default.

## Verification plan

Validation followed the narrowest targets first, then the complete CPU and
cross-platform build matrix below.

### Formatting and static checks

```text
zig fmt --check zml stdx vfs examples
./tools/buildifier.sh
git diff --check
```

### Unit and package tests

```text
bazel test //stdx:test --test_output=errors
bazel test //vfs:test --test_output=errors
bazel test //zml:test --test_output=errors
bazel test //zml/tokenizer:test --test_output=errors
```

Required focused coverage:

- local positional scatter over `IOV_MAX`, partial reads, EOF, and overflow;
- one remote logical call/one physical GET, with only serial retries;
- strict 206, Content-Range, and Content-Length validation;
- HTTP/HF/S3/GCS retry statistics and the chosen feedback scope;
- fair scheduler behavior for sharded and replicated jobs;
- actual maximum block reservation at repeated writer-mask boundaries;
- NUMA strict and replicated affinity feasibility;
- pool cancellation, allocation failure, high-water bound, and final release;
- DMA fairness with a cap below device count and idle-slot lending;
- synchronous and callback PJRT failures;
- ROCm 7.14 DmaMap support and absence of the old linearization/staging path;
- tagged source/model shape mismatch;
- packed FP4/sub-byte transfer allocation and byte offsets;
- multi-source fallback and `loadExecute` input lifetime;
- fixed read width and all adaptive state transitions being shipped;
- concurrent platform load/calibration behavior and busy errors.

### CPU and example builds

```text
bazel build //zml
bazel build //examples/io:playground //examples/mnist //examples/llm
bazel run //examples/mnist
```

### Accelerator builds

```text
bazel build --config=release --@zml//platforms:cuda=true //examples/io:playground //examples/llm
bazel build --config=release --@zml//platforms:rocm=true //examples/io:playground //examples/llm
bazel build --config=release --@zml//platforms:oneapi=true //examples/io:playground //examples/llm
```

Run the corresponding `//zml:test` target on available CUDA, ROCm, and oneAPI
hosts. A build-only result is not sufficient for DMA callback, mapped-memory,
or adaptive scheduling behavior.

### Performance revalidation

Old numbers are diagnostic, not merge acceptance, because master changes the
ROCm artifact, CUDA async dispatch, model shapes/quantization, Compiler, and
loader API. Re-measure under the final merged binaries:

- uncalibrated defaults and calibrated settings separately;
- calibration time separately from load time;
- one-device and multi-device sharded/replicated loads;
- local warm/cold files and deterministic remote service curves;
- real S3/HF only after deterministic correctness;
- fixed 32 MiB read widths `1,2,4,8,12,16,24,32,48,64` clipped by memory;
- adaptive result against the best fixed result;
- packed quantized and ordinary models;
- mid-load source change if D5 claims steady tracking;
- concurrent unrelated VFS traffic if stats affect policy.

At minimum, adaptive mode should remain within 3% of the best comparable
fixed-32-MiB source and end-to-end median on a sufficiently long load, select
the smallest demonstrated width in that band, and never exceed pinned or DMA
hard limits. If the merged controller remains terminal after settlement, do
not claim changing-source recovery.

## Validation results (2026-09-01)

| Scope | Result |
|---|---|
| Conflicts and provenance | `jj resolve --list` reports no conflicts; `@` has exactly feature `87d7df89` and master `e1e983c8` as parents. The temporary master-baseline workspace was forgotten and removed. |
| Static checks | Repository Zig sources pass `zig fmt --check` using the Bazel-provisioned Zig 0.16 toolchain; `./tools/buildifier.sh` and `git diff --check` pass. |
| Unit/package tests | `bazel test //stdx:test //vfs:test //zml:test //zml/tokenizer:test --test_output=errors` passes all four targets. |
| CPU build/runtime | `bazel build //zml //examples/io:playground //examples/mnist //examples/llm` passes, and `bazel run //examples/mnist` compiles, loads, and executes inference successfully. |
| Accelerator builds | Release builds of `//examples/io:playground` and `//examples/llm` pass with each of CUDA, ROCm, and oneAPI enabled. These are compile/link checks for CUDA and ROCm, not runtime DMA validation. |
| oneAPI fixed runtime | Release `//examples/mnist` passes on four Intel Arc Pro B70 devices, validating the fixed model-wide direct loader before adaptive mode. |
| oneAPI adaptive runtime | `playground load` of the MNIST safetensors in replicated mode passes with vectorization and adaptive reads enabled (four tensors, 16 DMA submissions). |
| oneAPI calibration smoke | A deliberately short 4 MiB/width-one `playground dma-bench` completes on all four devices and atomically publishes calibrated settings. This proves the control/lifetime path only; it is not a performance result. |
| Playground output paths | `playground safetensors` and the adaptive load command print and exit successfully after the streaming-writer fix. |
| oneAPI full test baseline | `bazel test --config=release --@zml//platforms:oneapi=true //zml:test` aborts in `attention: q=1,qh=64,kh=8`. The identical command at unmodified master `e1e983c8` aborts at the same test with the same native XLA stack header, so this is a confirmed pre-existing baseline failure rather than a merge regression. |

Still external to this merge audit: CUDA and ROCm runtime DMA/callback tests,
the ROCm 7.14 `PJRT_Client_DmaMap`/no-staging confirmation, and the full
performance matrix above. No performance claim is made from the short oneAPI
smoke run.

## Merge completion checklist

Merge correctness is complete; unchecked entries are explicit hardware or
performance follow-ups rather than hidden conflict work:

- [x] D1-D7 are explicitly resolved in this plan.
- [x] No `zml/io/vfs` package or `vfs/parallel_read.zig` remains.
- [x] No local `file:///home/...` dependency or stale ROCm lockfile content
      remains.
- [x] Master `TensorStore`, multi-source behavior, legacy direct writer,
      Compiler, model code, CUDA state, and GCS fix are preserved.
- [x] The vectorized loader uses model tensor shapes and packed PJRT shapes.
- [x] Workspace validation uses the real maximum request plan and NUMA
      affinity constraints.
- [x] Every load/calibration/platform teardown path releases the DMA borrow,
      events, blocks, buffers, and mapped arenas exactly once.
- [x] The IO playground deinitializes loaded PJRT buffers before its platform.
- [x] Remote positional admission still means one physical GET except serial
      retry.
- [x] Fixed-width mode passes before adaptive mode is enabled.
- [x] CPU, VFS, stdx, CUDA, ROCm, and oneAPI build/test results are recorded,
      including the master-baselined oneAPI failure.
- [ ] CUDA and ROCm runtime DMA/callback behavior and ROCm 7.14 DmaMap are
      validated on suitable hosts.
- [ ] Performance is re-measured on master's PJRT/runtime settings.
- [x] README and examples describe the API and calibration policy actually
      merged.
- [x] Historical documents and benchmark scripts have an intentional
      keep/parameterize decision.
