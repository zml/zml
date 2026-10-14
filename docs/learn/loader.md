# Loading checkpoints

`zml.io.Loader` turns checkpoint sources into device buffers. `load` submits
the tensors of a model, each from one source or, for a composite tensor, from
several sources assembled in its destination; `loadExecute` submits the
sources of executable bindings. Reads and transfers may start before submission returns.
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

try loader.load(Model, &model, &buffers, &store, null);
try loader.awaitAll();
```

Placement comes from each tensor's shape, not from the submission: the mesh
and partitioning the tensor was created with (`View.createTensor(subkey, tags,
.mesh_name, partitioning)` resolves the mesh among those passed to
`TensorStore.fromRegistry`; `createReplicatedTensor` replicates it on every
device). A binding's memory kind picks the destination memory space on each
device: a tensor created with `createHostPinnedTensor` lands in the device's
`host_pinned` memory, every other tensor in its default memory.
`TensorStore.setMemory(tensor, kind)` (or `View.setMemory`) changes it after
the tensor was created, for a placement decided later.
`loadExecute` places the inputs and checks the outputs against the
executable's `input_shapes` and `output_shapes`, whose shardings carry the
placement; a binding's tensor must match the executable's output placement.

```zig
const tp = try platform.registerMesh("tp", .mesh(.{ .model = .high_bandwidth }));
var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{tp});
const w = store.view().createTensor("w", .{ .out, .in }, .tp, .{ .out = .model });
const b = store.view().createReplicatedTensor("b", .{.out});
```

The store is passed to each `load` or `loadExecute` submission.
Initialization needs no store. These submission calls also accept an optional
progress node as their final argument (`null` disables reporting). Keep it
alive until `awaitAll` or `deinit` returns.
The caller owns the estimated total. For a whole checkpoint loaded once,
`store.view().count()` estimates the number of source tensors. Each loaded
tensor completes one item, a composite tensor once for all its sources; bulk
loads skip already delivered transformed tensors, and executable loads count
their input sources, not their outputs. Unused, repeatedly loaded or
concatenated sources can make the checkpoint estimate inexact.
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
`bytesLoaded` counts the logical bytes of submissions retired with execution:
each destination's shape once, so a composite tensor counts its padding and
not each of its sources.
`deinit` awaits what is still pending without executing it (outputs
unwritten, inputs freed), then destroys the backend. There is no
per-submission handle and no memory knob.

## Composite tensors

A composite tensor is one destination filled by several source byte ranges,
with every byte no range covers set to zero. The loader assembles it with
the ordinary `load`: no program, no `loadExecute`, no buffer per source. The
destination is placed like any tensor, by its shape's sharding, into the
memory space its binding names (`.default` or `.host_pinned`), and it is
allocated once per device whatever its parts: in `.default` that is device
memory, while a `.host_pinned` destination takes none.

```zig
// a, b and c end to end along dimension 0, padded to a multiple of 8 rows,
// row-sharded, in host pinned memory.
const table = store.view().createConcatenated(&.{ "a", "b", "c" }, .{ .row, .channel }, .tp, .{ .row = .model }, .{
    .pad_rows_to_multiple = 8,
    .memory = .host_pinned,
});
// One source padded to 8 rows: the engram-style table.
const padded = store.view().createConcatenated(&.{"weight"}, .{ .row, .channel }, .tp, .{ .row = .model }, .{ .pad_rows_to_multiple = 8 });
```

`createConcatenated` (and `maybeCreateConcatenated`, null when a source is
missing) takes the sources' dtype and dimensions past the first; dimension 0
is their total, rounded up to `pad_rows_to_multiple`, or `rows` when given.
Concatenating along the outermost axis concatenates the byte streams, so
each source becomes one contiguous range of the destination.
`createComposite(subkeys, shape, layout, memory)` is the general form: the
binding's `TensorStore.Layout` holds one `Part` per source, a byte map into
the destination's logical bytes (row-major bytes of its packed shape, before
the sharding splits them). A `Part` already has the fields of a strided map
(`rows`, `src_pitch`, `dst_pitch`) so interleaved or broadcast layouts can be
added without changing it.

`load` checks a composite binding before submitting anything. It must have
at least one source, a part must be non-empty and inside its source and the
destination, parts must not overlap, and every source must have the
destination's dtype, with its dimensions past the first equal to the
destination's innermost ones and the part cutting whole rows of them;
otherwise `InvalidLayout`. A zero-byte source is `EmptyTensor`. Strided parts and nonzero fills are valid
layouts that this loader does not assemble yet: `UnsupportedLayout`.

The front end turns a composite binding into one `LoadSpec` whose parts all
share the destination. The direct backend plans each part like a tensor: the
part's source bytes map to the destination's logical offsets, which the
destination's dispatch spans place on the devices. Neighbouring pieces of one
destination merge across parts when block, writers and both offsets are
contiguous. The parts of one destination may come from different files and
plans; each device's target still closes on the submission that completes its
placement bytes. The zero fill goes out with the plan of the destination's
first part, as one last job of that plan. That job reads nothing; it submits
pieces from one pinned zero block that the loader detaches from its block
pool when it is created and keeps until it is destroyed. The buffered backend
zeroes the gaps of its staging buffer, reads each part at its offset, then
places the whole tensor as usual.

What stays a program: anything that computes rather than moves bytes,
such as a dtype conversion, dequantization, a block-scale swizzle or a row
interleave. These remain transformed bindings delivered by `loadExecute`.

## Errors and diagnostics

The public operations declare `Loader.LoadError`,
`Loader.LoadExecuteError`, and `Loader.AwaitError`; `Loader.init` returns an
inferred error set. Their finite error sets
include the underlying file I/O, allocation, cancellation, sharding, and PJRT
errors. For example, missing files still return `FileNotFound`, truncated
reads return `UnexpectedEndOfFile`. Host-memory allocation and DMA registration
log the original PJRT failure and return `OutOfMemory`.
Exceeding the host-memory mapped ceiling also logs the budget details at `err`
and returns `OutOfMemory`.

A failure is logged once, where it is detected. Validation and failures
before publication leave the loader usable and log at `debug`; an
initialization failure and the first terminal failure log at `err` with the
values that explain it. Frames that only propagate an error do not log it
again, and subsequent calls return the stored error. Worker cancellation and
normal shutdown use `debug`. A planning failure before the first file is
published is recoverable; the same failure after partial publication is
terminal. The direct backend validates source sizes and parts before
publication, while the buffered backend checks them in its workers.

Loader-specific errors use these shared names:

| Error | Meaning |
| --- | --- |
| `InvalidOptions` | Invalid profile, parallelism, or alignment |
| `UnsupportedPlatform` | Unsupported host-memory target or device configuration |
| `SourceSizeMismatch` | Source bytes differ from the destination shape in either backend |
| `InvalidLayout` | A composite binding's parts do not fit its tensor |
| `UnsupportedLayout` | A composite layout this loader does not assemble yet |
| `InvalidTensorRange` | A source range cannot be represented |
| `Internal` | A planning or transfer bookkeeping invariant failed |
| `Closed` | An internal pool or scheduler has shut down |
| `TensorNotFound` | A tensor has no checkpoint binding |

`EmptyTensor`, `TransformedTensorNotDelivered`, and executable platform,
shape, and placement mismatch errors remain distinct. Executable arity
failures are `ExecutableInputCountMismatch` and
`ExecutableOutputCountMismatch`. Internal pool requests that cannot fit
still return `RequestExceedsCapacity`.

These names replace the former loader-specific profile, DMA benchmark,
DMA budget, loader-job, and shutdown names; callers matching those errors
must update their switches. They do not change how submissions are admitted,
retired, or cleaned up.

Error-path tests assert returned errors and cleanup rather than capturing logs.
The loader's log scopes (`zml/io/log.zig`) demote `err` to `debug` in test
builds because the Zig test runner counts logged errors as failures.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `zml/io.zig` | Public IO facade |
| `zml/io/loader.zig` | Loader options, FIFO submissions, admission, source preparation and executable ownership |
| `zml/io/execute_admission.zig` | Pure per-device room and cost arithmetic behind `loadExecute` admission |
| `zml/io/backend.zig` | Backend selection, submission dispatch and the shared `LoadSpec` contract |
| `zml/io/TensorStore.zig` | Checkpoint lookup, source bindings and prefixed model views |
| `zml/io/direct_loader.zig` | Planning, FIFO scheduling, source workers and transfer completion |
| `vfs/request.zig` | The governed request loop: retries and the backend-wide hold on rate limiting |
| `zml/io/DispatchSpans.zig` | Pure expansion of sharding into source ranges and destination offsets |
| `zml/io/buffered_loader.zig` | Whole-tensor staging and bounded positional reads |
| `zml/io/dma_calibration.zig` | Representative-device measurement and DMA block selection |
| `zml/io/host_memory.zig` | Internal host arenas, block leases and placement |
| `zml/io/log.zig` | The loader's log scopes: one log per failure, `err` demoted in test builds |
| `zml/io/testing.zig` | Test support: device counts, safetensors files, per-shard read-back |
| `zml/mem.zig` | Generic buffer conversion and public host-memory placement policy |

The shared front end resolves sources and placements once, from each tensor's
shape (or the executable's input shapes). Each `LoadSpec` is one destination:
the target shape with its placement, the destination memory kind, the
caller-owned output, and the parts that fill it, one whole source for a plain
tensor.
Backends consume those specs without knowing about model traversal or
executable bindings.

The direct backend reads coalesced source ranges into reusable host blocks.
Its `Planner` produces immutable jobs and transfer records, one plan per file.
Touching or overlapping source ranges share reads; gaps and file boundaries
remain separate. Jobs keep their planning order, which is file order.
The `Scheduler` publishes and claims those jobs in submission/file order.
Workers read them, and per-device pumps submit the preplanned transfer pieces.

One block may feed several outputs or devices. Its lease stays alive until
every consuming transfer completes or is abandoned. A fill job's one block is
the zero block, which nothing leases or releases. A batch owns its plans
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

Callers configure calibration through `Loader.Options.dma`. Calibration options
are checked with assertions; every candidate block size at the configured
parallelism must fit within the workspace mapped ceiling. The direct
backend caps its pinned arenas at a fixed 16 GiB, a safety guard rather than
an allocation target; callers never supply a workspace, a budget, a NUMA
policy or a calibration result. Page-backed arenas automatically interleave across the
host's memory-bearing NUMA nodes when more than one is discoverable.
The direct backend keeps the sizing selected during initialization
(`loader.backend.direct.calibration`, which the `examples/io` `dma-bench`
command prints); buffered loading has none. There is no separate public
benchmark or recalibration operation.

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
needing a block the reads did not leave free. The width never changes during
a load: a source that answers 429 (or 503 on the object stores) is handled
one layer down, where the VFS holds every request of that backend until the
delay it named has passed, so the traffic falls without the loader's memory
bounds moving.
