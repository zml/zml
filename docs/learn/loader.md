# Loading checkpoints

`zml.io.Loader` turns checkpoint sources into device buffers. `load` submits
the single-source tensors of a model; `loadExecute` submits the sources of
executable bindings. Both return a `Handle`. Reads and transfers may start
before submission returns, while executable bindings run when their handle
is awaited.

```zig
var buffers = try zml.mem.bufferize(allocator, Model, &model);
defer zml.mem.deinitBufferized(allocator, Model, &buffers);

var loader = try zml.io.Loader.init(allocator, io, platform, .{
    .load_profile = profile,
});
defer loader.deinit();

const handle = try loader.load(Model, &model, &buffers, &store, shardings, null);
try handle.await();
```

`load` and `loadBuffer` accept shardings for each submission; an empty slice
selects replicated placement. `loadExecute` uses the executable’s input and
output shardings. The loader does not retain the shardings slice.

The store is passed to each `load`, `loadBuffer`, or `loadExecute` submission;
`Window.submit` forwards it to `loadExecute`. Initialization needs no store.
These submission calls also accept an optional progress node as their final
argument (`null` disables reporting). Keep it alive until the handle completes.
The caller owns the estimated total. For a whole checkpoint loaded once,
`store.view().count()` estimates the number of source tensors. Each loaded
source completes one item; bulk loads skip already delivered transformed
tensors, and executable loads count their input sources, not their outputs.
Unused or repeatedly loaded sources can make the checkpoint estimate inexact.
Source progress completes before the executable runs; keep the enclosing node
alive until all handles finish to cover the full loading lifecycle.
The caller owns the store, platform, model buffers, and executable outputs.
Each submission borrows the store’s source metadata until its handle completes.
The loader owns its backend and handles. Declare cleanup in
the order above so the loader finishes using buffers before they are freed.
Submit and await serially on the owning task; the backend provides the read
and transfer concurrency. A loader may have any number of outstanding handles.

`Handle.await` is idempotent and caches its outcome. For `loadExecute`, it
runs each executable in binding order and frees the inputs. Loaded logical
bytes are counted only after a successful await. `Handle.isDone` reports
completion of the source reads and transfers; awaiting may still execute the
bindings. Handles remain valid until `Loader.deinit`, which waits for pending
transfers and frees inputs without executing pending bindings.

`zml.io.Window` adds a caller-side budget for executable inputs per device and
a maximum outstanding-handle count. It awaits the oldest handle before a new
submission would exceed either limit. An otherwise empty window always admits
one submission, even when its inputs exceed the budget. `Window.drain` awaits
everything and reports the first error; `Window.deinit` drains and drops errors.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `zml/io.zig` | Public IO facade |
| `zml/io/loader.zig` | Loader options, handles, execution window, source preparation and executable ownership |
| `zml/io/backend.zig` | Backend selection, submission dispatch and the shared `LoadSpec` contract |
| `zml/io/TensorStore.zig` | Checkpoint lookup, source bindings and prefixed model views |
| `zml/io/direct_loader.zig` | Planning, FIFO scheduling, source workers and transfer completion |
| `zml/io/source_concurrency.zig` | Pure adaptive source-width policy and its evidence |
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

`Loader.init` selects the transfer path with `Loader.backendFor(target)`.
That decision describes validated loading behavior, independently of whether
host memory is pinned or transfers use DMA.

The direct backend owns its workspace for its entire lifetime. Initialization
allocates that workspace, calibrates transfer sizing, and prepares the block
pool before returning. Calibration arenas become the load's initial capacity;
all arenas are released by `Loader.deinit`. Workspace and block-pool types
are internal to `io/host_memory.zig`.

Callers configure calibration through `Loader.Options.dma`, host memory limits
through `.max_host_bytes`, and NUMA placement through `.numa`. They never
supply a workspace or calibration result. `Loader.calibration()` reports the
sizing selected during initialization, or null for buffered loading. There is
no separate public benchmark or recalibration operation.

CPU uses default transfer sizing without measurement. Buffered backends skip
calibration. CUDA and oneAPI register host pages with PJRT; ROCm obtains pinned
host buffers from PJRT; CPU uses unregistered pages. The arena allocation
strategy and the loader's transfer path are separate decisions.

The source profile supplies a minimum read size. The effective request size
is the larger of that minimum and the selected DMA block, within the supported
limit. `source_concurrency.Controller` receives completed-read evidence and
backpressure, then returns a width and measurement generation. Runtime gates
enforce that width without draining requests on each decision. Request
lifecycle credits separately cover transfers that still hold host blocks.
