# Loading checkpoints

`zml.io.Loader` reads a model's weights from its checkpoint into device
buffers. Model code binds each tensor to checkpoint bytes through a
`TensorStore` view; the loader fills the buffers, assembles the tensors made
of several checkpoint tensors, and runs the programs that compute others.
The doc comments of `zml/io/loader.zig`, `TensorStore.zig`, `Source.zig` and
`vfs/vfs.zig` are the contract; the internals are in the `//!` header of
each file `zml/io.zig` lists.

## The recipe

```zig
// Local paths, and hf://, s3://, gs:// and https:// through `Remotes`,
// configured from the environment (tokens, credentials, proxies).
var vfs: zml.io.VFS = try .init(allocator, init.io);
defer vfs.deinit();
var remotes: zml.io.VFS.Remotes = undefined; // initialized in place
try remotes.init(allocator, &vfs, init.environ_map);
defer remotes.deinit();
const io = vfs.io(); // for the platform, the registry and the loader

const platform: *zml.Platform = try .auto(allocator, io, .{});
defer platform.deinit(allocator, io);

// Returns at once: calibration runs while the model is built and compiled.
var loader: zml.io.Loader = try .init(allocator, io, platform, .{ .checkpoint = path });
defer loader.deinit();

// The checkpoint's headers. The store owns the registry from here on.
const registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
var store: zml.io.TensorStore = .init(allocator, registry, .{});
defer store.deinit();

const model: Model = .init(store.view()); // binds every tensor
// ... compile the model ...

var buffers = try zml.mem.bufferize(allocator, &model);
defer zml.mem.deinitBufferized(allocator, &buffers);
try loader.start(&model, &buffers, &store, .none);
try loader.wait(); // logs "Loaded weights [bytes, time, rate]"
```

`.checkpoint` lets the VFS backend serving the path size the reads; without
it, or without a VFS, reads are sized for local files. `start` adds its own
items to the progress node it is given. Several loads may be started before
one `wait`, from one store or several. A second checkpoint, such as a
drafter's, joins the registry under a prefix before the store takes it:
`registry.addRepo(io, drafter_repo, "drafter")`, then
`store.view().withPrefix("drafter")` reaches its tensors.

## Placement

A tensor is placed by its shape. `View.createTensor(subkey, tags, .tp,
partitioning)` places the checkpoint tensor on the mesh named `.tp` among the
store's `meshes` (`View.mesh(.tp)` returns it), each tag of `partitioning`
naming the mesh axis that splits it; `createReplicatedTensor` puts it whole
on every device. It lands in default device memory, unless `View.setMemory`,
called after binding, names a host kind (`TensorStore.Memory`):
`.host_pinned` gives buffers in pinned host memory, and `.host_mapped` gives
device buffers that are views of pinned host memory, which kernels read in
place. The second suits a table too large for device memory, of which each
step reads a few rows; the memory goes with the last view. XLA counts those
views against device memory: give their size on one device,
`view.hostMappedBytes()`, back through the compile option
`extra_device_memory_bytes`.

```zig
const tp = try platform.registerMesh("tp", .mesh(.{ .model = .high_bandwidth }));
var store: zml.io.TensorStore = .init(allocator, registry, .{ .meshes = &.{tp} });
const view = store.view();
const w = view.createTensor("w", .{ .out, .in }, .tp, .{ .out = .model });
const b = view.createReplicatedTensor("b", .{.out});
const rows = view.createReplicatedTensor("rows", .{ .row, .d });
view.setMemory(rows, .host_mapped);
```

`View.get(subkey)` returns the registry's entry for a key, or null, and
`count()` the number of keys under the view's prefix. Binding cannot fail:
`createTensor` and `source` panic on a missing key, `maybeCreateTensor` and
`maybeSource` return null. `TensorStore.Options.quantization`
(`Quantization.Target.of(platform)`) is for code binding block-FP8 weights.

## Composite tensors

A composite tensor is one destination filled from several checkpoint byte
ranges, every byte no range covers set to zero, with no program and no
buffer per source. It is a `TensorStore.Source`: a checkpoint tensor
(`View.source(subkey, tags)`) and `zml.Tensor`'s shape ops over it, with the
same names, tags and arguments: `slice`, `split`, `chunkExact`, `splitAxis`,
`merge`, `reshape`, `transpose`, `pad` (with zeros), `broad`, `bitCast`,
`concatenate` and `stack`. Placement comes last, after every op:
`withPartitioning(mesh, partitioning)`, keyed by the source's final tags, or
`as(shape)`, a shape built elsewhere with the source's element count and
dtype. Both return a `Source.Placed`, which has no shape ops (its `.shape`
is the placed shape) and drops its placement with `unplaced()`; its
`bind()` makes the model tensor.

```zig
const Source = zml.io.TensorStore.Source;
// a, b and c end to end along the rows, padded to 16 rows, row-sharded.
const abc = Source.concatenate(&.{ view.source("a", .{ .row, .d }), view.source("b", .{ .row, .d }), view.source("c", .{ .row, .d }) }, .row);
const table = abc.pad(0, .{ .row = Source.Pad{ .high = 16 - abc.dim(.row) } })
    .withPartitioning(.tp, .{ .row = .model }).bind();
// q, k and v cut in one group per device and interleaved, so that each
// device holds its rows of every projection.
var members: [3]Source = undefined;
for (&members, [_][]const u8{ "q.weight", "k.weight", "v.weight" }) |*m, key| {
    m.* = view.source(key, .{ .dout, .d }).splitAxis(.dout, .{ .group = devices, .r = -1 });
}
const qkv = Source.concatenate(&members, .r).merge(.{ .dout = .{ .group, .r } }).withPartitioning(.tp, .{ .dout = .model }).bind();
```

Sources live in their store's arena, and sources of two stores do not mix.
The loader only moves bytes: large runs are cheap, runs under 4 KiB cost
host copies. For a layout of many small runs, a program over sources read
in large runs may load faster, at the cost of a compile and staging memory
(measured in the header of `zml/io/direct_loader.zig`).

## Programs

What computes rather than moves bytes is a program: a dtype conversion, a
dequantization, a global scale. A program is a value of its own type, plain
data (checked at compile time), with a `forward` method; its type name
names the compiled module in traces and logs.

```zig
const Dequantize = struct {
    block_size: i64,

    pub fn forward(self: Dequantize, output: zml.Shape, inputs: []const zml.Tensor) zml.Tensor { ... }
};
const weight = Source.bindProgram(&.{ w.as(placed), s.as(scales) }, placed.withDtype(.bf16), Dequantize{ .block_size = 128 });
```

`bindProgram(inputs, output, program)` binds a tensor of `output` that
`program.forward(output, inputs)` computes during the load. Each input is a
`Source.Placed` and lands placed by its own shape, so every device reads
only its slice; a whole checkpoint tensor, replicated, is
`view.source(key, null).withPartitioning(.replicated, .{})`. `start`
starts one compile, on a task of its own, for all the tensors of an equal
program (same type, equal value) over the same placed shapes. `wait` lands
the inputs in groups of at most `Loader.Options.program_group_bytes` of
outputs (8 GiB), each once the devices have room for it, runs the programs
on the calling task and frees the inputs. A program's output lands in
default device memory: `setMemory` panics on it.

## Lifetime and failure

| Handle | Borrows | Until |
| --- | --- | --- |
| `Loader` | the platform; `io`, and the VFS behind it, through which it keeps its source files open | its `deinit` |
| `Loader.start` | the store, the buffers and the progress node | `wait` or `cancel` returns |
| `TensorStore` | the meshes; it owns the registry, the bindings, sources and programs | its `deinit` |
| `View`, `Source`, `Source.Placed` | their store, which must not move | their last use |
| `VFS.Remotes` | the VFS; it must not move | its `deinit`, once every file opened through it is closed |
| Loaded buffers | the platform | `deinitBufferized` |

`init` reads its options and `start` the model during the call only. The
loader copies each file's path, so nothing of the store is kept past `wait`.
The loaded buffers are the caller's and outlive the loader, which holds its
pinned memory and workers until `deinit`: the recipe's deferred `deinit`
keeps them until the function returns, which suits a program that does
little after the load; one that keeps running frees the loader right after
`wait`, or loads in a block of its own. Deferred cleanup in the recipe's
order is right; if code between `start` and `wait` can fail,
`errdefer loader.cancel();` after `start`. The loader is a value. Start and
wait on one task: the backend reads and transfers concurrently.

Every error of `start` or `wait` fails the loader: everything started is
canceled, and every later call returns the first error. `cancel` gives up on
what is started without running its programs; `deinit` cancels first.
Whatever failed, each buffer is empty or holds shards whose contents are
unspecified: read none, and free the tree with `zml.mem.deinitBufferized`.
That holds for the buffers `bufferize` made, which `start` empties again.

## Errors

`Loader.init` returns `InvalidOptions`, logged with the option it refuses,
or `OutOfMemory`. `start` and `wait` return the error where it was
detected, under its own name, and that is where it is logged, once, at
err, with what explains it; a plain allocation failure and `Canceled` are
returned unlogged.

A model whose bindings are wrong is `InvalidBinding`, whichever rule it
breaks; the log names the tensor, the checkpoint key its source reads and
the rule:

- buffers shaped unlike the model: another union arm, optional or slice
  length (the log names the field path instead);
- a tensor the store did not bind;
- a source that reads no bytes, such as an empty checkpoint tensor (the log
  names the tensor, not the key);
- a source whose bytes differ from its tensor's;
- a source the loader cannot assemble: a run that starts or ends inside a
  byte (sub-byte dtypes; the log names the tensor and its source's shape,
  not the key), strided parts into host memory, or a broadcast or transpose
  whose source range does not fit one request (8 to 32 MiB);
- a placement its mesh does not divide, or on devices the platform lacks;
- a memory kind the platform cannot take;
- a program that does not return its tensor's shape, placed alike.

The errors a caller can act on:

| Error | Meaning |
| --- | --- |
| `FileNotFound` | A checkpoint file is missing |
| `AccessDenied`, `PermissionDenied` | A file is denied: `AccessDenied` locally, `PermissionDenied` remotely (on S3, a missing key without `s3:ListBucket` too) |
| `UnexpectedEndOfFile` | A file ends inside a tensor: it changed after its header was read, or the tensor's shape needs more bytes than its `data_offsets` give |
| `OutOfMemory` | Host memory the backend cannot allocate, pin or fit under its mapped ceiling, which the log explains; or a plain allocation that failed, unlogged |
| `ResourceExhausted` | Device memory ran out, for a buffer or a program's run (from PJRT) |
| `ConcurrencyUnavailable` | An `io` that cannot run the loader's tasks concurrently |
| `Canceled` | The load was canceled |

PJRT and compile errors pass through. A platform with more devices than ZML
addresses (`Platform.MAX_NUM_DEVICES`) fails before the loader:
`Platform.init` returns `InvalidDeviceCount`, logged. A remote failure other
than a missing or denied object is logged by the VFS and surfaces as
`Unexpected`. A truncated checkpoint fails before the loader:
`TensorRegistry.fromPath` returns `InvalidSafetensorsHeader`, logged with
the file.
