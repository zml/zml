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

A composite tensor is one destination filled from several checkpoint byte
ranges, with every byte no range covers set to zero. The loader assembles it
with the ordinary `load`: no program, no `loadExecute`, no buffer per source.
The destination is placed like any tensor, by its shape's sharding, into the
memory space its binding names (`.default` or `.host_pinned`), and it is
allocated once per device whatever its parts: in `.default` that is device
memory, while a `.host_pinned` destination takes none.

A composite is described with `TensorStore.Source`: a checkpoint tensor
(`View.source(subkey, tags)`, or `maybeSource`) and `zml.Tensor`'s shape ops
over it, with the same names, tags and arguments: `slice`, `split`,
`chunkExact`, `splitAxis`, `merge`, `reshape`, `transpose`, `pad` (with
zeros), `broad`, `bitCast`, `concatenate` and `stack`. A source records
nothing until `bind(mesh, partitioning, .{ .memory })` makes the model tensor,
partitioned by the source's final tags like `createTensor`.

```zig
const Source = zml.io.TensorStore.Source;
// a, b and c end to end along the rows, padded to 16 rows, row-sharded, in
// host pinned memory.
const abc = Source.concatenate(&.{ view.source("a", .{ .row, .channel }), view.source("b", .{ .row, .channel }), view.source("c", .{ .row, .channel }) }, .row);
const table = abc.pad(0, .{ .row = Source.Pad{ .high = 16 - abc.dim(.row) } })
    .bind(.tp, .{ .row = .model }, .{ .memory = .host_pinned });
// q, k and v cut in one group per device and interleaved, so that each
// device holds its rows of every projection: one contiguous run per member.
var members: [3]Source = undefined;
for (&members, [_][]const u8{ "q.weight", "k.weight", "v.weight" }) |*m, key| {
    m.* = view.source(key, .{ .dout, .d }).splitAxis(.dout, .{ .group = devices, .r = -1 });
}
const qkv = Source.concatenate(&members, .r).merge(.{ .dout = .{ .group, .r } }).bind(.tp, .{ .dout = .model }, .{});
```

A source is a few pieces, each an affine lattice over one checkpoint tensor;
reshapes never touch them, and the other ops rewrite them without
enumerating rows. `bind` lowers them to the binding's `TensorStore.Layout`:
byte `Part`s of the destination's logical bytes (row-major bytes of its
shape, before the sharding splits them), each a run of `width` bytes
repeated over strided dimensions. A source that is one whole checkpoint
tensor binds plainly, like `createTensor`. `TensorStore.sourceOf` turns a
binding back into a source, so code that receives a bound tensor (a
`Linear`'s scales) can rework it with more ops and `Source.rebind` it in
place of the old binding.

`load` checks a composite binding before submitting anything: the layout
fills the tensor's bytes, it has a part, each part reads a source of the
binding, stays inside it and the destination, and parts do not overlap;
otherwise `InvalidLayout`. A zero-byte source is `EmptyTensor`. A strided
part of at most 64 runs, or of runs of at least 1 MiB, becomes one load part
per run. A composite with any other strided part (many small runs: row
interleaves, transposes, broadcasts) goes to the backend as strided parts,
which the direct backend gathers (below). Strided parts must land in run
order without overlapping. Two that interleave are checked exactly when one
of them has at most 1024 runs (each looked for among the other's), or when
they repeat with the same strides, before or after merging the runs that
touch (a search over the differences of their indices); any other pair is
`UnsupportedLayout`. Strided
parts into a host memory space are `UnsupportedLayout` too. A transformed
binding with a layout feeds the assembled tensor to its executable as the
one input (`loadExecute`).

The front end turns a composite binding into one `LoadSpec` whose parts all
share the destination. The direct backend plans each part like a tensor: the
part's source bytes map to the destination's logical offsets, which the
destination's grid places on the devices (`ShardGrid`: every device holds
one box of an even grid over the tensor, so a contiguous logical range lands
as one contiguous range of each box it touches, found by arithmetic rather
than by listing rows). Neighbouring pieces of one
destination merge across parts when block, writers and both offsets are
contiguous. The parts of one destination may come from different files and
plans; each device's target still closes on the submission that completes its
placement bytes. The zero fill goes out with the plan of the destination's
first part, as one last job of that plan. That job reads nothing; it submits
pieces from one pinned zero block that the loader detaches from its block
pool when it is created and keeps until it is destroyed. The buffered backend
zeroes the gaps of its staging buffer, reads each part at its offset, then
places the whole tensor as usual; a strided part reads its source range once
and copies each run where it lands.

A destination fed by strided parts is gathered: the direct backend plans it
whole, with the file of its first part, cutting its logical bytes into
ranges of at most a job. Each range is one job that reads, for every part
with runs in it, the source range those runs come from (from whichever file
holds it), and lands the range in its blocks box by box, so every device
takes one piece per block, whatever the interleave's granularity. A part
whose source runs are in order reads just its runs of the range, scattered
straight into place when the read is exact; a broadcast or a transpose reads
its whole source range in every job through the worker's bounce buffer, and
must fit one read. From the bounce buffer, the runs of a stretch that lands
in one piece (one run of the grid, inside one block) are copied by nested
loops over the part's two innermost dimensions, a run of up to 16 bytes as
one load and one store (`Lattice.copyRuns`); only the runs a stretch's ends
cut, and items with few runs per stretch, go one run at a time. Bytes no part
covers land as zeros.

The block-scale swizzles are sources too: llmd lays MXFP4 and NVFP4 expert
scales, and Blackwell's block-FP8 scales, out in the 128x4 blocks the
block-scaled MMA reads with pads, splits, transposes and broadcasts, so the
loader gathers them from the checkpoint in their final layout: runs of four
bytes, from a source read once per job. A transpose that only moves axes of
one element is a reshape and leaves the pieces alone.

What stays a program: anything that computes rather than moves bytes, such
as a dtype conversion or dequantization. These remain transformed bindings
delivered by `loadExecute`.

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
| `zml/io/ShardGrid.zig` | Pure arithmetic placing logical byte ranges on the devices' boxes |
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

A job's read normally lands in its blocks in file order. A tensor sharded
along any axis but its first then gives every device one piece per row (a
column-sharded `o_proj`), and the per-piece cost dominates: Llama-3.1-8B
sharded that way on four B70 took 1.05M submissions and 5.1 s. Where it
saves at least 64 submissions, the planner stages the job instead: its read
lands grouped by destination and writer set, in destination order, so each
device takes one piece per block (5,468 submissions, 0.71 s). A buffered
read scatters straight into place; a direct read, whose ranges must stay
aligned, lands in a per-worker bounce buffer and is copied from there.
Neither the plan nor the worker lists rows ahead: a staged job records, per
piece, where each box's bytes start in its blocks, and the worker walks the
runs as it reads. Llama's plans take 1-4 ms, against 73-112 ms when every
row and device was a span.

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
