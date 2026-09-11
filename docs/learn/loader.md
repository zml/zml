# Loading checkpoints

`zml.io.Loader` turns checkpoint sources into device buffers. `load` submits
the single-source tensors of a model; `loadExecute` submits the sources of
executable bindings. Reads and transfers may start before submission returns.
Submissions are retired in submission order: the loader waits for the reads,
then for `loadExecute` runs each executable in binding order on the calling
task, writes its output and frees its inputs.

```zig
var buffers = try zml.mem.bufferize(allocator, Model, &model);
defer zml.mem.deinitBufferized(allocator, Model, &buffers);

var loader = try zml.io.Loader.init(allocator, io, platform, .{
    .load_profile = profile,
});
defer loader.deinit();

try loader.load(Model, &model, &buffers, &store, shardings, null);
try loader.awaitAll();
```

`load` accepts shardings for each submission; an empty slice
selects replicated placement. `loadExecute` uses the executable’s input and
output shardings. The loader does not retain the shardings slice.

The store is passed to each `load` or `loadExecute` submission.
Initialization needs no store. These submission calls also accept an optional
progress node as their final argument (`null` disables reporting). Keep it
alive until `awaitAll` or `deinit` returns.
The caller owns the estimated total. For a whole checkpoint loaded once,
`store.view().count()` estimates the number of source tensors. Each loaded
source completes one item; bulk loads skip already delivered transformed
tensors, and executable loads count their input sources, not their outputs.
Unused or repeatedly loaded sources can make the checkpoint estimate inexact.
Source progress completes before the executable runs; keep the enclosing node
alive until `awaitAll` returns to cover the full loading lifecycle.
The caller owns the store, platform, model buffers, and executable outputs.
Each submission borrows the store’s source metadata until it is retired.
The loader owns its backend and its pending submissions. Declare cleanup in
the order above so the loader finishes using buffers before they are freed.
Submit and await serially on the owning task; the backend provides the read
and transfer concurrency. A loader may have any number of pending submissions.

`loadExecute` admits its submission itself. Before publishing, it retires the
oldest pending submissions until the new one fits the room the devices report
(their memory limit minus what is in use, minus the loader’s own unlanded
weights), at a cost of inputs, temporaries and output per device; it always
admits when nothing is pending. Where a device reports no memory limit (CPU)
or the backend cannot account its allocations, every pending submission is
retired first. `load` is never gated: submitted after `loadExecute`, it queues
behind the packs, and a transformed tensor counts as delivered once a
`loadExecute` naming it was submitted. Outputs of earlier bindings may
therefore be written by a later `loadExecute` or by `awaitAll`.

`awaitAll` retires everything and returns the first error; a failure is
sticky: later submissions are refused and `awaitAll` keeps returning it.
`bytesLoaded` counts the logical bytes of submissions retired with execution.
`deinit` awaits what is still pending without executing it (outputs
unwritten, inputs freed), then destroys the backend. There is no
per-submission handle and no memory knob.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `zml/io.zig` | Public IO facade |
| `zml/io/loader.zig` | Loader options, FIFO submissions, admission, source preparation and executable ownership |
| `zml/io/execute_admission.zig` | Pure per-device room and cost arithmetic behind `loadExecute` admission |
| `zml/io/backend.zig` | Backend selection, submission dispatch and the shared `LoadSpec` contract |
| `zml/io/TensorStore.zig` | Checkpoint lookup, source bindings and prefixed model views |
| `zml/io/direct_loader.zig` | Planning, FIFO scheduling, source workers, transfer completion and the throttle watch |
| `zml/io/DispatchSpans.zig` | Pure expansion of sharding into source ranges and destination offsets |
| `zml/io/buffered_loader.zig` | Whole-tensor staging and bounded positional reads |
| `zml/io/dma_calibration.zig` | Representative-device measurement and DMA block selection |
| `zml/io/host_memory.zig` | Internal host arenas, block leases and placement |
| `zml/mem.zig` | Generic buffer conversion and public host-memory placement policy |

The shared front end resolves sources and shardings once. Each `LoadSpec`
contains a source, target shape, resolved sharding, and caller-owned output.
Backends consume those specs without knowing about model traversal or
executable bindings.

The direct backend reads coalesced source ranges into reusable host blocks.
Its `Planner` produces immutable jobs and transfer records, one plan per file.
Touching or overlapping source ranges share reads; gaps and file boundaries
remain separate. The planner also determines fair job order across devices.
The `Scheduler` publishes and claims those jobs in submission/file order.
Workers read them, and per-device pumps submit the preplanned transfer pieces.

One block may feed several outputs or devices. Its lease stays alive until
every consuming transfer completes or is abandoned. A batch owns its plans
and callback contexts until every job completes. The last-transfer flag goes
on the submission that completes a destination's byte count, allowing reads
and DMA to complete out of source order. Failure retires unclaimed work and
drains existing ownership before batch teardown.

The buffered backend instead stages a whole tensor for `Buffer.from`. It
uses separate read permits and a byte budget for host staging. CPU, CUDA,
ROCm and oneAPI use the direct backend; TPU, neuron and metal use the buffered
backend. CPU's direct arenas are ordinary pages.

## Initialization and host memory

`Loader.init` selects the transfer path from the platform target, and the
backend takes the same `Loader.Options` the caller passed. That decision
describes validated loading behavior, independently of whether host memory is
pinned or transfers use DMA.

The direct backend owns its workspace for its entire lifetime. Initialization
allocates that workspace, calibrates transfer sizing, and prepares the block
pool before returning. Calibration arenas become the load's initial capacity;
all arenas are released by `Loader.deinit`. Workspace and block-pool types
are internal to `io/host_memory.zig`.

Callers configure calibration through `Loader.Options.dma`. The direct
backend caps its pinned arenas at a fixed 16 GiB, a safety guard rather than
an allocation target; callers never supply a workspace, a budget, a NUMA
policy or a calibration result. Page-backed arenas automatically interleave across the
host's memory-bearing NUMA nodes when more than one is discoverable.
`Loader.calibration()` reports the sizing selected during initialization, or
null for buffered loading. There is no separate public benchmark or
recalibration operation.

CPU uses default transfer sizing without measurement. Buffered backends skip
calibration. CUDA and oneAPI register host pages with PJRT; ROCm obtains pinned
host buffers from PJRT; CPU uses unregistered pages. The arena allocation
strategy and the loader's transfer path are separate decisions.

The source profile supplies a minimum read size. The effective request size
is the larger of that minimum and the selected DMA block, within the supported
limit. The source width is fixed for the load: `Options.read_parallelism`, or
the profile's default when null (`limits.defaultReadParallelism`: 16 reads for
local files, 32 for a high-latency source), clipped to what the pre-grown
pinned set holds. The direct backend pre-grows that many requests plus the
DMA reserve before the first read, so nothing maps during a load. A read gate
enforces the width; the lifecycle credits are the whole pre-grown capacity, so
transfers that still hold host blocks keep the DMA stage fed without ever
needing a block the reads did not leave free. The one change during a load is a step down: when a remote
source reports a throttle or timeout, the width is halved once the reads in
flight at the previous step have returned. Nothing raises it again.
