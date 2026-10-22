# Loading checkpoints

`zml.io.Loader` turns checkpoint sources into device buffers. `start` takes a
whole model. Every tensor the loader fills itself, from one source or, for a
composite tensor, from several sources assembled in its destination, is
published at once, and the programs that compute the other tensors start
compiling (`Source.bindProgram`). `wait` then lands the programs' inputs in
groups the devices have room for, runs each group's programs on the calling
task, writes their outputs and frees the inputs, and returns once every
started load is in its buffers. It logs one line: the bytes loaded, the time
since the first `start` and their rate, and what the programs computed.

```zig
// Returns at once; calibration runs meanwhile.
var loader = try zml.io.Loader.init(allocator, io, platform, .{ .checkpoint = path });
defer loader.deinit();

var buffers = try zml.mem.bufferize(allocator, Model, &model);
defer zml.mem.deinitBufferized(allocator, Model, &buffers);

try loader.start(&model, &buffers, &store, .none);
try loader.wait();
```

Placement comes from each tensor's shape, not from the submission: the mesh
and partitioning the tensor was created with (`View.createTensor(subkey, tags,
.mesh_name, partitioning)` resolves the mesh among the `meshes` passed to
`TensorStore.init`, which `View.mesh(name)` returns too;
`createReplicatedTensor` replicates it on every device). A binding's memory
kind picks the destination memory space on each device: every tensor lands
in its default memory unless `View.setMemory(tensor, kind)`, called after the
tensor is bound, names another (`.host_pinned`). A program's inputs are
placed by their own shapes (see Programs over sources), and its output must
be placed like its tensor; it lands in default device memory, and
`setMemory` panics on it.

```zig
const tp = try platform.registerMesh("tp", .mesh(.{ .model = .high_bandwidth }));
const registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
var store: zml.io.TensorStore = .init(allocator, registry, .{ .meshes = &.{tp} });
defer store.deinit(); // the bindings, then the registry
const w = store.view().createTensor("w", .{ .out, .in }, .tp, .{ .out = .model });
const b = store.view().createReplicatedTensor("b", .{.out});
```

The store owns the registry it is given, which must be complete: its map is
locked, so a later `registerTensor` panics in safe builds. `View.get(subkey)`
returns the registry's own entry, or null, in one lookup. An entry's address
is its checkpoint tensor's identity: every binding of the tensor holds the
entry itself. `View.hasPrefix(subkey)` asks whether any key starts with
`subkey`, a scan of every key. A checkpoint tensor's bytes are read with
`safetensors.Tensor.read(io, buffer)`, the whole tensor, or
`readRange(io, file, buffer, offset)`, a range of it from a file the caller
opened.

`TensorStore.Options.quantization` says what the platform's FP8 GEMMs read:
`zml.Quantization.Target.of(platform)` gives the E4M3 encoding and
whether e8m0 block scales are swizzled (CUDA sm_10x). The encoding is FNUZ
on CDNA3 and FN on CUDA, CDNA4 and RDNA4. It is null elsewhere, CPU
included: block-FP8 activations stay BF16 and weights keep the
checkpoint's encoding. Without a target, the store assumes FN,
unswizzled. The store only carries it; code that binds block-FP8 weights
reads it with `View.quantization()` and converts or lays them out while
binding.

A second checkpoint, such as a speculative drafter's, joins the same
registry under a prefix of its own before the store takes it:
`registry.addRepo(io, drafter_repo, "drafter.")` registers each of its
tensors as `drafter.` followed by its name, and
`store.view().withPrefix("drafter")` reaches them: `createTensor` on that
view binds them, as do `source`, `withPartitioning` and `bind`. One store,
and the loads started from it, then cover both checkpoints.

The store is passed to each `start`; initialization needs no store. `start`
also takes a progress node as its final argument (`.none` disables
reporting). The loader copies the node; do not end it until `wait` or
`cancel` returns. `start` adds its own items to the node's estimated total,
and the load completes them all: one per loaded tensor, a composite tensor
once for all its sources, and one per program input; a program's output
adds none. Source progress completes before a program runs; keep the
enclosing node alive until `wait` returns to cover the full loading
lifecycle. The caller owns the store, platform and model buffers; a started
load borrows the buffers and the store until `wait` or `cancel` returns, and
the progress node must not end before then. The model itself is read during
`start` only. The loader owns its backend, the
programs it compiles and its pending submissions. Declare cleanup in the
order above so the loader finishes using buffers before they are freed.
Start and wait serially on the owning task; the backend provides the read and
transfer concurrency, and each program compiles on a task of its own. Several
loads may be started before one `wait`.

`wait` admits the program groups itself. A group holds the tensors of
consecutive programs in model order, up to `Loader.program_group_bytes` of
outputs (8 GiB), so the inputs of one layer's experts, adjacent in the
checkpoint, are read together. Before publishing a group, `wait` waits for its
programs to compile, then retires the oldest pending submissions until the
group fits the room the devices report (their memory limit minus what is in
use, minus the weights the backend has yet to allocate), at a cost of
inputs, outputs and the largest program's temporaries per device (a group's
programs run one after the other); it always admits when no group is
pending. Where a device reports no memory limit (CPU) or the backend cannot
account its allocations, one group is in flight at a time. The tensors the
loader fills itself are never gated: `start` publishes them before any group,
so every group is admitted with them counted.

Every error of `start` or `wait` fails the loader: what was started is
canceled, and every later call returns the first error. `wait` is idempotent.
Its line counts each destination's bytes once, so a composite tensor counts
its padding and not each of its sources, and a program counts its inputs.
`cancel` gives up on everything started, which only an error between `start`
and `wait` leaves behind: it drops the queued groups, aborts the backend,
retires the submissions without running their programs, inputs freed, and
waits for the compiles. The loader is failed afterwards. `deinit` cancels,
then destroys the backend. There is no per-submission handle and no memory
budget: `program_group_bytes` only sizes the program groups.

Whatever fails, the buffers can be freed as after a load. Pass `start` the
buffers `bufferize` made, which are empty; `start` empties them again before
it checks a binding, and a backend or a program writes a buffer whole or not
at all. After an error of
`start`, `wait` or a `cancel`, each buffer is empty or holds its shards, whose
contents are unspecified (unwritten, errored, or partly written): read none,
and free the tree with `zml.mem.deinitBufferized`, which skips the empty
ones.

## Composite tensors

A composite tensor is one destination filled from several checkpoint byte
ranges, with every byte no range covers set to zero. The loader assembles it
as it fills a plain tensor: no program, no buffer per source.
The destination is placed like any tensor, by its shape's sharding, into the
memory space its binding names (`.default` or `.host_pinned`), and it is
allocated once per device whatever its parts: in `.default` that is device
memory, while a `.host_pinned` destination takes none.

A composite is described with `TensorStore.Source`: a checkpoint tensor
(`View.source(subkey, tags)`, or `maybeSource`) and `zml.Tensor`'s shape ops
over it, with the same names, tags and arguments: `slice`, `split`,
`chunkExact`, `splitAxis`, `merge`, `reshape`, `transpose`, `pad` (with
zeros), `broad`, `bitCast`, `concatenate` and `stack`. Its placement comes
last, after every op (an op on a placed source panics):
`withPartitioning(mesh, partitioning)`, keyed by the source's final tags like
`createTensor` (`.replicated` takes only `.{}`), or `as(shape)`, a shape
built elsewhere with the source's element count and dtype, whose dimensions,
tags and sharding it takes; `unplaced()` drops it again, for code handed a
placed source that reworks it by its placement. A source records nothing
until `bind()` makes the model tensor from the placed source.

```zig
const Source = zml.io.TensorStore.Source;
// a, b and c end to end along the rows, padded to 16 rows, row-sharded, in
// host pinned memory.
const abc = Source.concatenate(&.{ view.source("a", .{ .row, .channel }), view.source("b", .{ .row, .channel }), view.source("c", .{ .row, .channel }) }, .row);
const table = abc.pad(0, .{ .row = Source.Pad{ .high = 16 - abc.dim(.row) } })
    .withPartitioning(.tp, .{ .row = .model }).bind();
view.setMemory(table, .host_pinned);
// q, k and v cut in one group per device and interleaved, so that each
// device holds its rows of every projection: one contiguous run per member.
var members: [3]Source = undefined;
for (&members, [_][]const u8{ "q.weight", "k.weight", "v.weight" }) |*m, key| {
    m.* = view.source(key, .{ .dout, .d }).splitAxis(.dout, .{ .group = devices, .r = -1 });
}
const qkv = Source.concatenate(&members, .r).merge(.{ .dout = .{ .group, .r } }).withPartitioning(.tp, .{ .dout = .model }).bind();
```

A source is a few pieces, each an affine lattice over one checkpoint tensor;
reshapes never touch them, and the other ops rewrite them without
enumerating rows. `bind` records the placed source as it is, its shape and
pieces (`TensorStore.Binding.load`). `start` lowers each piece to bytes
(`Source.lowerPiece`, the one rule): a part of the destination's logical
bytes (row-major bytes of its shape, before the sharding splits them), a
run of `width` bytes repeated over strided dimensions. A source that is one
whole checkpoint tensor is one part, its whole bytes.

`start` checks every tensor once, before submitting anything (see Errors).
Only `Source` makes bindings, and its pieces are disjoint by construction:
the loader relies on that and does not check it (Source's tests do). A
strided part of at most 64 runs, or of runs of at least 1 MiB, becomes one
load part per run. A composite with any other strided part (many small
runs: row interleaves, transposes, broadcasts) goes to the backend as
strided parts, which the direct backend gathers (below); strided parts into
a host memory space are `InvalidBinding`.

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
pieces from one pinned zero block, the first block of the pinned set, which no
credit owns. The buffered backend
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

The block-scale swizzles can be sources too: the 128x4 blocks the
block-scaled MMA reads are pads, splits, transposes and broadcasts, so the
loader can gather them from the checkpoint in their final layout, as runs of
four bytes from a source read once per job. A transpose that only moves axes
of one element is a reshape and leaves the pieces alone.

Source ops are Tensor ops, so the same layout code can run in the loader or
as a program over a source that lands in large runs (below). Runs under
4 KiB cost host copies; a program costs a compile and staging memory for its
input. On DeepSeek-V4.1 over two GB300, laying the MXFP4 experts out in the
loader (gate and up rows interleaved, scales swizzled) copied 77 to 189 GiB
through bounce buffers, 17 to 36 s of worker time; as programs it copied
1 to 3 GiB and loaded 0.1 to 1.1 s faster, for 4 to 6 GiB more device memory
at the peak. llmd lays expert stacks out with programs, and Blackwell's
block-FP8 scales (a few MiB) in the loader.

## Programs over sources

What stays a program is what computes rather than moves bytes: a dtype
conversion, a dequantization, a global scale's maximum.
`Source.bindProgram(inputs, output, program)` binds a tensor that
`program.forward(output, inputs)` computes while loading, from placed
sources: each input is placed by its own shape (`withPartitioning` or `as`),
as `bind` places a tensor, so a program can take a fused weight and its
scale grid with every device landing only its slice.

```zig
const Dequantize = struct {
    block_size: i64,

    pub fn forward(self: Dequantize, output: zml.Shape, inputs: []const zml.Tensor) zml.Tensor { ... }
};
const weight = Source.bindProgram(&.{ w.as(placed), s.as(scales) }, placed.withDtype(.bf16), Dequantize{ .block_size = 128 });
```

A program is a value of its own type: plain data, checked at compile time,
whose type name names the compiled module in traces and logs. The binding
keeps it and each placed input (`Binding.compute`, whose `program` is a
`TensorStore.Program`; `Program.as(T)` reads the value back). The loader
compiles one executable for every tensor of an equal program over the same
placed shapes, and starts each compile as soon as `start` sees it. It lands
each input as it lands a bound source, in default device memory, runs the
program over them, then frees them; the program must return the tensor's
shape, placed alike, or `wait` fails with `InvalidBinding`.
`View.maybeCreateBinding(keys, shape, program)` is the shorthand for whole
checkpoint tensors, replicated.

## Errors and diagnostics

`start` and `wait` declare one error set, `Loader.Error`; `Loader.init`
returns `InvalidOptions` or `OutOfMemory`. `Loader.Error` keeps the names a
caller can act on: file I/O (a missing file is `FileNotFound`, a truncated
read `UnexpectedEndOfFile`), allocation, PJRT, compile errors and `Canceled`.
Host-memory allocation and DMA registration log the original PJRT failure
and return `OutOfMemory`. Exceeding the host-memory mapped ceiling also logs
the budget details at `err` and returns `OutOfMemory`.

Every remote scheme of the VFS fails an open, a stat or an `access` alike: a
missing object is `FileNotFound` (404, 410) and a denied one
`PermissionDenied` (401, 403). On S3, a principal without `s3:ListBucket`
gets `PermissionDenied` for a missing key too, so a probe of an optional file
treats both as absent. Any other remote failure is logged at `err` and
surfaces as `Unexpected`.

A model whose bindings are wrong is one error, `InvalidBinding`, whichever
rule it breaks:

- a tensor with no checkpoint binding;
- a source that reads no bytes, or reads an empty checkpoint tensor;
- a source whose bytes differ from its tensor's;
- a source this loader cannot assemble: a run of elements that starts or
  ends inside a byte (sub-byte dtypes), or strided parts into a host memory
  space;
- a tensor whose dimensions its mesh partitions do not divide, or whose mesh
  holds devices the platform lacks, in any memory space;
- a memory kind the platform cannot take;
- a program that does not return its tensor's shape, placed alike.

A failure is logged once, where it is detected, at `err` with the values that
explain it: a rejected binding names its tensor (for a program input, the
input and its tensor), the checkpoint key its source reads and the rule. A
source that reads nothing, such as a padded empty tensor, has no key to name.
Frames that only propagate an error do not log it again, and subsequent
calls return the stored error. Worker cancellation and normal shutdown use
`debug`. The front end checks every tensor and program input once, before
anything is published, and the backends trust what it submits
(`LoadSpec`); the direct backend still refuses, before publication, a
strided part whose source range does not fit one read.

`Loader.init` checks every option, logs a rejected one at `err` and
returns `InvalidOptions`: a read width or a block size out of range, an
empty block list, a checkpoint whose scheme the VFS does not serve, or a
read size that a block calibration may select does not divide. What
`init`'s task fails with comes from the first `start`, as any
`Loader.Error`:

| Error | Meaning |
| --- | --- |
| `UnsupportedPlatform` | More devices than ZML addresses (`Platform.MAX_NUM_DEVICES`: 32, or 64 in builds with TPU) |
| `ConcurrencyUnavailable` | An `io` that cannot run the warm-up's or the workers' tasks concurrently |
| `OutOfMemory`, PJRT errors | Warm-up, calibration or the pinned pre-growth failed |

The backends keep their own names (`UnsupportedLayout`,
`InvalidTensorRange`, `IncompatibleSharding`, `UnsupportedMemoryKind`); the
front end returns them as `InvalidBinding`. The front end panics on `Closed`,
`Internal` and, after `init`, `InvalidOptions`: each means a broken
invariant (a backend returns the error that closed it, never `Closed`).

Error-path tests assert returned errors and cleanup rather than capturing logs.
The loader's log scopes (`zml/io/log.zig`) demote `err` to `debug` in test
builds because the Zig test runner counts logged errors as failures.

## Implementation map

| Module | Responsibility |
| --- | --- |
| `zml/io.zig` | Public IO facade |
| `zml/io/loader.zig` | Loader options and their checks, the set-up task, starts, program compiles and groups, FIFO submissions, admission and its per-device arithmetic, and source preparation |
| `zml/io/backend.zig` | Backend selection, submission dispatch and the shared `LoadSpec` contract |
| `zml/io/TensorStore.zig` | Checkpoint lookup, source bindings and prefixed model views |
| `zml/io/direct_loader.zig` | Planning, FIFO scheduling, source workers and transfer completion |
| `vfs/request.zig` | The governed request loop: retries and the backend-wide hold on rate limiting |
| `zml/io/ShardGrid.zig` | Pure arithmetic placing logical byte ranges on the devices' boxes |
| `zml/io/buffered_loader.zig` | Whole-tensor staging and bounded positional reads |
| `zml/io/dma_calibration.zig` | Representative-device measurement and DMA block selection |
| `zml/io/host_memory.zig` | Internal host arenas, their blocks and placement |
| `zml/io/log.zig` | The loader's log scopes: one log per failure, `err` demoted in test builds |
| `zml/io/testing.zig` | Test support: device counts, safetensors files, per-shard read-back |
| `zml/mem.zig` | Generic buffer conversion and public host-memory placement policy |

The shared front end resolves sources and placements once, from each tensor's
shape, or for a program's inputs from their placed shapes
(`Binding.compute.inputs`). Each `LoadSpec` is one destination:
the target shape with its placement, the destination memory kind, the
caller-owned output, and the parts that fill it, the binding's pieces in
bytes.
Backends consume those specs without knowing about model traversal or
programs.

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

A worker takes a lifecycle credit before it claims a job: a fixed slot of
pinned blocks that only its request reads into and transfers from. One block
may feed several outputs or devices; the request holds its credit until every
piece it queued completes or is abandoned, so the credits bound the pinned
memory in use. A fill job's transfers read the zero block, which no credit
owns. A batch owns its plans
and callback contexts until every job completes. The last-transfer flag goes
on the submission that completes a destination's byte count, allowing reads
and DMA to complete out of source order. Failure retires unclaimed work and
drains existing ownership before batch teardown.

The buffered backend instead stages a whole tensor for `Buffer.from`. It
uses separate read permits and a byte budget for host staging. CPU, CUDA,
ROCm and oneAPI use the direct backend; TPU, neuron and metal use the buffered
backend. CPU's direct arenas are ordinary pages.

## Initialization and host memory

`Loader.init` checks the options, resolves the checkpoint's read hints
through `VFS.fromIo(io)` (a plain `io`, or no checkpoint, takes the local
ones), and returns at once. A task of the loader's own then
warms the device allocators and creates the backend, selected from the
platform target; when `io` cannot run it concurrently, `init` runs it in
place. The first `start` waits for that task: its error fails the loader and
is what that `start` returns. `deinit` cancels a task no `start` waited for,
or destroys what it built. The task holds no pointer into the loader, so the
loader is a value that may be moved before its first `start`. Initialize it
as soon as the platform exists and build or compile the model meanwhile.

A remote checkpoint (`hf://`, `s3://`, `gs://`, `https://`) is read through a
VFS. `VFS.Remotes` registers those four schemes on one HTTP client, each
configured from the environment (tokens, credentials, proxies):

```zig
var vfs: zml.io.VFS = try .init(allocator, init.io);
defer vfs.deinit();
var remotes: zml.io.VFS.Remotes = undefined; // initialized in place
try remotes.init(allocator, &vfs, init.environ_map);
defer remotes.deinit();
const io = vfs.io(); // for the registry, the loader and every other reader
```

Under a VFS, whether a file is read past the page cache
(`Loader.Options.direct_io`) is decided per local file, whatever the
checkpoint's scheme: a local drafter beside an `hf://` model may be read
directly. A remote file is always read exact, and the planner does not open
it to ask. Each remote backend logs its read totals (requests, bytes,
retries, throttles, holds) at debug when it is deinitialized.

The direct backend owns its workspace for its entire lifetime. Its creation
allocates that workspace, calibrates transfer sizing, and cuts the arenas into
blocks and the credits' slots. Calibration arenas become the
load's initial capacity; all arenas are released by `Loader.deinit`. The
workspace type is internal to `io/host_memory.zig`.

Calibration screens the blocks of `Loader.Options.dma_block_sizes`, each at
most 32 MiB and at least the direct I/O alignment (`VFS.direct_io_alignment`,
4 KiB, when `io` is a VFS and `direct_io` is not `off`), at a fixed
eight transfers per device; its windows and its selection tolerance are
fixed too. The direct
backend caps its pinned arenas at a fixed 16 GiB, a safety guard rather than
an allocation target; callers never supply a workspace, a budget, a NUMA
policy or a calibration result. Page-backed arenas automatically interleave across the
host's memory-bearing NUMA nodes when more than one is discoverable.
The direct backend keeps the sizing selected during its creation; buffered
loading has none. `dma_calibration.measure(allocator, io, platform, opts)`
runs the same warm-up and screen without a loader, in a workspace of its
own, with every knob of `dma_calibration.Options` (block sizes, transfers
per device, window length and transfer count, selection tolerance,
confirmation windows); a knob out of range (a block above 32 MiB among
them) is `InvalidOptions`. TPU, neuron, metal and furiosa load through the
buffered backend, so `measure` returns `UnsupportedPlatform` there. The
`examples/io` `dma-bench` command prints its result.

CPU uses default transfer sizing without measurement. Buffered backends skip
calibration. CUDA and oneAPI register host pages with PJRT; ROCm obtains pinned
host buffers from PJRT; CPU uses unregistered pages. The arena allocation
strategy and the loader's transfer path are separate decisions.

The checkpoint's VFS backend supplies a minimum read size (8 MiB for local
files). The effective request size is the larger of that minimum and the
selected DMA block, at most 32 MiB, and a whole number of blocks. The source
width is fixed for the load: `Options.read_parallelism`, or the profile's
default when null (16 reads for local files, 32 for a high-latency source),
clipped to what the pre-grown pinned set holds. The direct backend pre-grows
that many requests plus the DMA reserve before the first read, so nothing
maps during a load. A read gate enforces the width; the lifecycle credits
are the whole pre-grown capacity, so transfers that still hold host blocks
keep the DMA stage fed without ever needing a block the reads did not leave
free. The width never changes during a load: a source that answers 429 (or
503 on the object stores) is handled one layer down, where the VFS holds
every request of that backend until the delay it named has passed, so the
traffic falls without the loader's memory bounds moving.
