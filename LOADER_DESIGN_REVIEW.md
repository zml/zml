# Loader design review (2026-10-28)

**Status, 2026-10-31.** Waves 1-3 are done. Wave 1 (M1-M4, M7-M10, with the VFS re-scope that replaces M5 and M6 and keeps the vfs/ diff against master minimal, D4) is committed as `rework` (zml 8af7fd3c, monorepo b791da11). Waves 2 and 3 (M11-M19), with F1 (fixes found by wave 1's GPU runs) and W2F (fixes from the review of wave 2 as a whole), are uncommitted on top. All of it ran on gb300-1 (CUDA) and mi300 (ROCm). M20 (a separate PR) and the Part 3 decisions not taken stay open. CTX.md's fifty-fifth to fifty-seventh passes record what landed and how it differs from this review (wave 1 in the fifty-fifth, its GPU runs and F1 in the fifty-sixth, waves 2 and 3 in the fifty-seventh), which stays pinned to HEAD 93428bd0.

This review looks at the design of the model loading rework in zml (HEAD 93428bd0, against master b37ba315) and in llmd (monorepo HEAD 0bf031d9, against dc7ba734). It judges every surface against one brief: do more with less. Fewer abstractions, not only fewer lines. Logic that reads locally. Abstractions that do not leak. Nothing left to remove.

It follows `LOADER_REVIEW.md` and `LOADER_API_REVIEW.md`, both acted on in full. It does not re-report what they reported unless it is still in the code. Where it re-raises one of their keeps or rejections, Part 4 says why the reason no longer holds.

**Method.**
- 21 reviewers read the code at HEAD:
  - 16 by surface: the loader front end, TensorStore and Source, the backend seam and ShardGrid, the buffered backend, the direct backend's core, its planner and scheduler, its reads, landing and DMA, host memory and calibration, the VFS core, the VFS request loop, the remote backends, the registry and `Bufferized`, llmd's weights layer, llmd's load orchestration, llmd's model binding code, and the callers, examples and docs;
  - 5 by lens: a concept inventory, an end-to-end trace, leaks and non-local rules, the error and failure model, and tests as design feedback.
- They produced about 159 findings. A merge folded them into 105 groups.
- Verifiers told to refute each group checked facts and value. 102 survived: 65 confirmed, 37 partial (most with a corrected claim or a smaller proposal). 3 were refuted (Appendix B).
- Three designers built a target from the survivors, from three angles: delete-first, caller-first and incremental. A critic checked the coverage, the conflicts between findings and the weak claims.
- I scored the designs, took one as the spine and grafted the best of the others and of the critic (below).
- Nothing was built or run.
- I checked the five claims the verdict leans on most, and all hold:
  - a placed source is a runtime flag plus a second type: `Source._placed` with 21 `expectUnplaced` and 2 `expectPlaced` calls, beside `Source.Bound` (zml/io/Source.zig:43, :570, :595-601);
  - both backends copy every spec's parts and dimensions because the front end frees them at publish (zml/io/backend.zig:56-59, zml/io/buffered_loader.zig:85-123, zml/io/direct_loader.zig:538-596, zml/io/loader.zig:406);
  - `backendError` renames seven backend error names into `InvalidBinding` or a panic, over six error-set declarations (zml/io/loader.zig:914-929, zml/io/direct_loader.zig:84-101, zml/io/backend.zig:15-16, zml/io/buffered_loader.zig:33-35);
  - S3's and GCS's 233-line vtable blocks are identical once the backend name is changed: the diff of vfs/s3.zig:367-599 against vfs/gcs.zig:526-758 is empty;
  - llmd's load protocol is two `@hasDecl` hooks, a default that reads fields by name and a flag in 11 structs (../monorepo/llmd/models.zig:240-263: 14 `buffers_loaded` lines, 8 `startLoad`s, 2 `finalizeLoadedBuffers`), and the drafter is decided in five places (main, zoo, `models.zig`'s table, `SpeculativeContext`, five model inits).
- I also checked every critic suggestion I adopt or reject:
  - [22]'s spawn: the buffered backend spawns with `group.async` (zml/io/buffered_loader.zig:294), the direct one with `try worker_group.concurrent` (zml/io/direct_loader.zig:290);
  - `whole()`'s callers: examples/io in production (examples/io/main.zig:491, :496, :530), and about 16 test lines across both repos;
  - std's denial names: `std.Io.Threaded` returns `AccessDenied` for EACCES and `PermissionDenied` for EPERM (zig 0.17 std/Io/Threaded.zig:3364-3365);
  - `Shape.partitioning()` has one production caller, and both `withPartitioning`s take a `Partitioning` value (zml/shape.zig:858-863, zml/tensor.zig:181);
  - the nine root files are referenced by no build file or source;
  - [87]'s compiler half: `collectOutputInfo` takes the returned tensor's shape (zml/Compiler.zig:661), and `attributes` emits its sharding (zml/Compiler.zig:556);
  - [28]'s premise: an XLA copy in the bazel cache shows the transfer manager's destructor erroring every incomplete buffer and `SetBufferError`'s CHECK. I did not check that it is the pinned version.

**How the designs scored.**
- **Delete-first** (12 moves, about −3,700 lines): the most complete after-API and inventory. But its moves bundle 9 to 13 findings each, so they do not land alone. It takes [87], deletes `whole()` although llmd's tests use it, and puts the remote core inside the design although it is the pending half of decision 6's Move 10.
- **Caller-first** (7 moves plus internals, about −2,630): the clearest account of what a caller must know (29 rules). But its internals are an unordered list, and it takes [87].
- **Incremental** (20 moves in three waves, about −2,400): every move lands alone, ordered by risk, each with what to measure. It declines [87] and leaves the remote core to you. Its after-API is thin.
- **The spine is the incremental design.** I grafted delete-first's API sketch and inventory, caller-first's rules table and recipes, and from the critic:
  - the buffered workers spawn with `concurrent`, never `concurrent catch async`;
  - `Shape.partitioning()` and `Logical.set` go;
  - [83]'s `layOut` takes a `Source.Placed`, since [16] puts `bind` and `unplaced` there;
  - its list of conflicts, resolved move by move below.

## Verdict

- **The pipeline earns its place.** Source's bit lattices, one lowering to byte `LoadPart`s, per-file plans, FIFO claims, credits as fixed slots, per-device pumps, ShardGrid, start/wait with one sticky error, the calibration screen, both backends, the VFS as an `std.Io` and the governed request loop. Every measured invariant in CTX sits in one of these. No move in wave 1 touches one.
- **Concepts are written twice at the seams.**
  - A placed source is a flag with 23 guards and a second type, `Bound`.
  - A byte move is a `Lattice` that `LoadPart` and direct's `Part` restate field by field.
  - Read stats, staging ledgers, read hints (`LoadProfile`/`ReadHints`), placed-shape equality (`samePlacedShape`) and struct-literal partitioning (`Partitioning.parse`'s loop) each come in pairs.
- **Translation layers exist only to rename.** `backendError` and six error sets; `Config.alignment` with three meanings; five `consume` functions re-mapping statuses `exchange` let through; the four remote backends mapping one read failure eight times.
- **Lifetimes that a copy or an owner would delete.** The spec copy rule. `dma_block_sizes` borrowed by init's task. `Remotes` declared `undefined`, initialized in place, never moved, deinitialized in order. A loader that "may be moved until its first start" though nothing points into it.
- **Self-checks re-prove what the front end guaranteed.** ShardGrid and TensorTransfer re-derive placement and memory kinds fallibly. `maximumJobLen` refuses what `Loader.init` refused. `awaitBatch` marks buffers that the PJRT manager's destructor already errors. `HostWritten` re-sums the planner's own transfers.
- **Dead or speculative generality.** LimitedGroup's adaptive API and five tests, `FixedBufferPool`, three calibration knobs and its "calibrated" depth that is a constant, `Device.attribute`, `Backend.auto`'s dtype, `ExpertSpec.gate_up`, the inert rate-limit key, and in llmd `ModelParallelism.replicated` and `BakedInputs`.
- **Copies.** Four remote backends repeat one handle table and one 17-function shell. Five `ModelBuffers` are a hand-written `Bufferized`. Three expert key loops, two dense-MLP conventions, eleven hand-built `Linear.init`s.
- **llmd splits decisions across files.** The drafter is decided in five places. A model's load story is split by loader phase (`startLoad`, then `finalizeLoadedBuffers`), with a default that reads `spec.tensors` and `model_buffers` by name.
- **Bugs met along the way**, each fixed by a move: a server-named retry delay with no cap (M6); `dirRealPathFile` keeping the scheme (M5); HTTP's redirect walker overwriting its own URL (M6); buffered helper reads run inline under a busy Threaded io (M17); a staging budget the code does not hold (M17); `Buffer.from` on a sub-mesh (M10); latent: a direct-I/O state overwrite (M5) and a hang on a canceled LimitedGroup admission (M17). Not fixed: S3 and GCS listings stop at 1000 keys (Unverified leads).
- **The design: 19 moves in three waves, about −2,500 lines** (code and tests).
  - Wave 1: 10 moves, about −1,630. No measured path changes.
  - Wave 2: 7 moves, about −775. Measured paths change how they store their data. It ends in one GPU A/B.
  - Wave 3: 2 moves, about −110. They need failure injection on CUDA and ROCm, and an MI300X.
  - The decisions in Part 3 would remove about 1,150 more, most of it the remote core. Three pre-existing llmd cleanups (M20, separate PR) remove about 270.
- **Not taken:** [87] (the loader places a program's result). The compiler would honour it, but it writes a private field from another module and turns a misplaced program into a silent reshard. And [6] (wait releases the backend): the earlier rejection's reason still holds.

**Concept counts, before and after the 19 moves.**

| Concept | Before | After |
|---|---|---|
| Production named types in zml/io (the inventory's count) | 111 | about 93 |
| Tagged unions / arms in zml/io | 9 / 27 | 8 / 23 |
| Named error sets / translation steps in zml/io | 11 / 3 | 4 / 0 |
| Encodings of a placed source | 2 (flag, `Bound`) | 1 (`Source.Placed`) |
| Structs that spell one byte move | 3 | 1 (`Lattice`) |
| Front-end `Loader` fields | 22 | 19 |
| Rules a caller must know (caller-first's list) | 29 | 13 |
| Request-layer names a backend author meets | about 16 | about 12 (about 5 under D4) |
| Remote file shells | 4 | 4 (1 under D4) |
| llmd: places that decide the drafter | 5 | 1 |
| llmd: model load hooks | 2 optional, a default, a flag | 1 required `load` |
| llmd: hand-written buffer trees | 5 `ModelBuffers` + `initBuffers` | 0 |

Sixteen fewer. Eight go: Remotes' in-place and no-move rules; the loader's move rule; the borrowed block list; `addRepo`'s trailing dot; placement-comes-last as a runtime panic (now a compile error); the two empty-entry rules; the restated model type in `bufferize`/`deinitBufferized`; init's `OutOfMemory` and the first start's `UnsupportedPlatform`. llmd's nine (the drafter table, `buffers_loaded`, the `startLoad` default, the optional `finalizeLoadedBuffers`, the hand-paired `ModelBuffers`, the recomputed host-mapped bytes, `splitTensor`'s re-placement, `bindBlockFp8`'s triple, `prefix().?` beside `hasPrefix`) become one: a model's `load` starts its loads and waits. The other 12 that stay are ordinary pinning and borrowing, the early `Loader.init`, `setMemory` after binding (decision 4), "a missing key panics", the start/wait borrow (decision 2) and `errdefer loader.cancel()`.

## Part 1: the target shape

**What a caller writes** (mnist after M9; llmd and examples/llm differ only in the remote VFS and the model):

```zig
var vfs: zml.io.VFS = try .init(allocator, init.io);
defer vfs.deinit();
try vfs.addRemotes(init.environ_map); // one call; the VFS owns them
const io = vfs.io();
const platform: *zml.Platform = try .auto(allocator, io, .{});
defer platform.deinit(allocator, io);
var loader: zml.io.Loader = try .init(allocator, io, platform, .{ .checkpoint = path }); // returns at once; calibrates while we compile
defer loader.deinit();
var store: zml.io.TensorStore = .init(allocator, try .fromPath(allocator, io, path), .{});
defer store.deinit();
const model: Mnist = .init(store.view());
var exe = try platform.compile(allocator, io, model, .forward, .{input}, .{});
defer exe.deinit();
var weights = try zml.mem.bufferize(allocator, &model);
defer zml.mem.deinitBufferized(allocator, &weights);
try loader.start(&model, &weights, &store, .none);
try loader.wait(); // logs "Loaded weights"
```

**The public API after the moves.**

```zig
// zml.io.VFS: an std.Io over local paths and hf://, s3://, gs://, https://
pub fn init(allocator: Allocator, inner: std.Io) !VFS;
pub fn deinit(self: *VFS) void;
pub fn addRemotes(self: *VFS, environ: *std.process.Environ.Map) !void; // setup, once
pub fn registerBackend(self: *VFS, scheme: []const u8, backend: Backend) RegisterError!void; // setup only, no mutex
pub fn io(self: *VFS) std.Io; // the VFS must not move after this
pub fn readHints(self: *VFS, path: []const u8) error{VFSNotRegistered}!ReadHints;
pub const DirectIo = enum { off, on, auto };
pub const ReadHints = base.ReadHints; // { read_chunk_size, high_latency }, with `.local`
pub const MockServer = @import("mock_server.zig").MockServer; // tests: init, start, stop, deinit, path
// unchanged: fromIo, isLocal, useDirectIo

// zml.io.Loader: a value; init returns at once (decision 2: start/wait)
pub const Error = error{ InvalidBinding, Canceled } || Allocator.Error || pjrt.ApiError ||
    std.Io.File.OpenError || safetensors.ExactPositionalError || Compiler.Error || std.Io.ConcurrentError;
pub const Options = struct {
    checkpoint: ?[]const u8 = null,
    read_parallelism: ?usize = null, // 16 local, 32 high-latency, resolved by init
    dma_block_sizes: []const usize = &dma_calibration.default_block_sizes, // copied by init
    direct_io: VFS.DirectIo = .auto,
    program_group_bytes: usize = 8 << 30,
};
pub fn init(allocator: Allocator, io: std.Io, platform: *const Platform, opts: Options) error{InvalidOptions}!Loader; // allocates nothing
pub fn start(self: *Loader, model: anytype, buffers: *Bufferized(@TypeOf(model.*)), store: *const TensorStore, progress: std.Progress.Node) Error!void;
pub fn wait(self: *Loader) Error!void;
pub fn cancel(self: *Loader) void;
pub fn deinit(self: *Loader) void;

// zml.io.TensorStore: owned by main; owns the registry and every binding
pub fn init(allocator: Allocator, registry: TensorRegistry, opts: Options) TensorStore;
pub fn deinit(self: *TensorStore) void;
pub fn view(self: *TensorStore) View;
pub fn bindingOf(self: *const TensorStore, id: Tensor.Id) ?Binding; // the loader's one read
pub const Binding = union(enum) {
    load: struct { source: Source.Placed, memory: Memory = .default },
    compute: Compute,
    pub const Compute = struct {
        program: Program,
        inputs: []const Source.Placed,
        pub fn sameExecutable(self: Compute, output: Shape, other: Compute, other_output: Shape) bool;
        pub fn compile(self: Compute, allocator: Allocator, io: std.Io, platform: *const Platform, meshes: []const *const Mesh, output: Shape) Compiler.Error!Exe;
    };
};
/// T is plain data with `pub fn forward(self: T, output: Shape, inputs: []const Tensor) Tensor`.
pub const Program = struct {
    pub fn init(arena: Allocator, program: anytype) Allocator.Error!Program;
    pub fn as(self: Program, comptime T: type) ?T;
    pub fn name(self: Program) []const u8;
    pub fn eql(a: Program, b: Program) bool;
};
pub const View = struct {
    // unchanged: root, withPrefix, withLayer, get, count, source, maybeSource,
    // createTensor, maybeCreateTensor, createReplicatedTensor, setMemory (decision 4), mesh, quantization
    pub fn prefix(self: *const View) []const u8; // "" at the root
    pub fn hostMappedBytes(self: View) error{IncompatibleSharding}!u64;
};

// zml.io.TensorStore.Source: the unplaced ops are unchanged
pub fn withPartitioning(self: Source, comptime mesh: @EnumLiteral(), partitioning: anytype) Placed;
pub fn as(self: Source, placed: Shape) Placed;
pub fn bindProgram(inputs: []const Placed, output: Shape, program: anytype) Tensor;
/// A source with its placement: the only thing that binds, and what a binding keeps.
pub const Placed = struct {
    store: *TensorStore,
    shape: Shape,
    _pieces: []const Piece,
    pub fn bind(self: Placed) Tensor;
    pub fn unplaced(self: Placed) Source;
    pub fn whole(self: Placed) ?*const safetensors.Tensor; // a test probe of which key bound
    pub fn lower(self: Placed, arena: Allocator) error{ OutOfMemory, SubBytePosition }![]backend.LoadPart;
};

// zml.mem
pub fn bufferize(allocator: Allocator, model: anytype) !Bufferized(@TypeOf(model.*));
pub fn deinitBufferized(allocator: Allocator, bufferized: anytype) void; // a compile error on a Tensor tree
pub fn pairBufferized(allocator: Allocator, model: anytype, bufferized: *Bufferized(@TypeOf(model.*)), report: *std.Io.Writer) error{ Mismatch, OutOfMemory }![]Pair;

// zml.safetensors
pub fn resolveModelRepo(io: std.Io, path: []const u8) ModelPathResolutionError!std.Io.Dir; // directories only
// TensorRegistry: init and registerTensor (tests), fromPath, fromRepo,
// addRepo(io, repo, name) with `name` as View.withPrefix takes it ("drafter")
```

**The seam between the front end and the backends** (internal; io.zig exports none of it):

```zig
// zml/io/backend.zig
pub const Error = Allocator.Error || pjrt.ApiError || std.Io.File.OpenError ||
    safetensors.ExactPositionalError || error{ InvalidBinding, Canceled };
pub const max_dma_block_sizes = 16;
pub const Options = struct {
    read_parallelism: usize,
    read_hints: VFS.ReadHints = .local,
    dma_block_sizes: stdx.BoundedArray(usize, max_dma_block_sizes),
    direct_io: VFS.DirectIo = .auto,
};
/// Checked once by the front end (`checkParts`). Everything a spec names
/// (its parts, their dims, the sources, the output) stays alive until its await.
pub const LoadSpec = struct { shape: Shape, memory: Memory = .default, output: *Buffer, parts: []const LoadPart };
pub const LoadPart = struct { source: *const safetensors.Tensor, lattice: Lattice };
pub const Lattice = struct { src_offset: u64 = 0, dst_offset: u64 = 0, width: u64, dims: []const Dim = &.{} }; // Run, Cursor, predicates, copyRuns unchanged
pub fn checkParts(parts: []const LoadPart, destination_bytes: u64) bool; // one loop
pub fn loadsDirect(platform: *const Platform) bool; // the one platform list
pub fn supportsMemory(platform: *const Platform, kind: Memory) bool;
// Backend and Submission: two-arm unions; submit, await, abort, destroy

// zml/io/dma_calibration.zig (decision 1: kept; the API narrows)
pub const depth: usize = 8;
pub fn calibrate(workspace: *host_memory.Workspace, platform: *const Platform, opts: Options) !usize; // the block size
pub fn measure(allocator: Allocator, io: std.Io, platform: *const Platform, opts: Options) !usize;

// vfs/request.zig
pub const Governor = struct { // one per backend instance
    io: std.Io, client: *std.http.Client, name: []const u8, unavailable: ReadFailure, retry: Retry,
    stats: AtomicReadStats = .{}, mutex: std.Io.Mutex = .init, hold: Hold = .{},
    pub fn init(io: std.Io, client: *std.http.Client, name: []const u8, unavailable: ReadFailure, retry: Retry) Governor;
    pub fn deinit(self: *const Governor) void; // logs the read totals
    pub fn openFailure(self: *const Governor, path: []const u8, err: anyerror) OpenError;
    pub fn readFailure(self: *const Governor, uri: []const u8, offset: u64, err: anyerror) error{Unexpected};
};
pub const RequestSpec = struct { target: []const u8, key: Key = "", retry_once: []const std.http.Status = &.{} }; // `key` goes under D2
// D4 replaces Attempt and the hooks with: perform(T, governor, spec, context, prepare(ctx, previous) Request, consume(ctx, res) T)
```

**llmd after the moves.**

```zig
// llmd/weights.zig
pub fn mismatch(view: View, key: []const u8, comptime reason: []const u8, args: anytype) error{CheckpointMismatch};
pub fn unsupported(view: View, key: []const u8, comptime reason: []const u8, args: anytype) error{UnsupportedQuantization};
pub const ExpertSpec = struct { num_experts: u32, backend: zml.moe.Backend, gate: []const u8 = "gate_proj", up: []const u8 = "up_proj", down: []const u8 = "down_proj" };
pub fn experts(allocator: Allocator, view: View, spec: ExpertSpec) !Experts; // the allocator goes under D3
pub fn moeBackend(platform: *const zml.Platform, view: View, projection: []const u8) !zml.moe.Backend;

// llmd/weights/linear.zig: View, view, maybeView, Definition, detect, hasWeight, create, maybeCreate,
// createBf16, fuse, FusedLinear, withLayout, NormalizeBlockFp8Weight/Scale, and:
pub fn bindQuantized(store: View, scheme: Scheme, weight: Source.Placed, scales: Source.Placed,
    global_scale: ?GlobalScale, input_scale: ?GlobalScale) !struct { Tensor, zml.Quantization };

// llmd/speculative.zig: resolved once, before TensorStore.init takes the registry
pub const Drafter = union(enum) {
    native: ?u32, // DeepSeek's DSpark layers
    checkpoint: drafter_config.Parsed, // its tensors are under "drafter."
    pub fn open(allocator: Allocator, io: std.Io, registry: *TensorRegistry, model_type: ModelType, name: ?[]const u8, draft_count: ?u32) !?Drafter;
    pub fn deinit(self: *Drafter) void;
};

// each model
const Weights = struct { base: Tensors, speculative: ?speculative.Tensors };
buffers: zml.Bufferized(Weights),
pub fn init(allocator, io, platform, repo, store, tokenizer, options, hf_model_type, config, drafter: ?*const speculative.Drafter, progress) !Model;
pub fn load(self: *Model, loader: *zml.io.Loader, store: *const zml.io.TensorStore, progress: std.Progress.Node) !void; // start, wait, bake
```

**What each handle is for.**
- **VFS**: one `std.Io` for every path. It owns its remotes and the direct-I/O state of local descriptors. It is pinned once `io()` is called.
- **Loader**: opened early so calibration and pinned pre-growth overlap the compile. `start` borrows the store, the buffers and the progress node until `wait` or `cancel`. `deinit` frees the pinned memory and the workers.
- **TensorStore**: owned by main. It owns the registry and every binding.
- **View**: what model code holds: navigate, look up, bind, set a memory kind.
- **Source**: a byte expression over checkpoint tensors with Tensor's shape ops. **Source.Placed**: a source with its placement, the only thing that binds.
- **Program**: a plain-data value whose `forward` computes one tensor; the loader compiles and runs it (decision 2).
- **Bufferized(T)**: the loaded tree, built by `bufferize` and freed by `deinitBufferized` whatever happened.
- **LoadSpec / LoadPart / Lattice** (internal): the only contract between the front end and both backends.
- **Drafter** (llmd): which drafter a model gets, decided once. **Model.load** (llmd): a model's whole load story in one function.

## Part 2: the moves

Each move lands alone with tests green. Within a wave the order follows the dependencies noted. Line counts are the verifiers' estimates, netted by hand where findings overlap, code and tests together.

| Move | Findings | Net lines | Risk | Needs |
|---|---|---|---|---|
| **Wave 1** | | **≈ −1,630** | | |
| M1 Registry, mem and Partitioning edges | 60, 57, 58, 59, 61, 62, 56 (half), C3 | −250 | low | CPU |
| M2 The front end keeps lifecycle state | 3, 20, 33.3, 2, 4, 5, 14, 55 | −60 | low | CPU |
| M3 One seam contract | 12, 33.4, 7, 8, 9, 31 (half), 10, 11, 13, 15, 36 | −165 | low | CPU, GPU smoke |
| M4 Placement is a type | 18, 19, 16, 17, 21 | −95 | low | CPU |
| M5 VFS core | 64, 66, 63, 67, 68, 69, 77 | −250 | low | CPU |
| M6 Request layer and small remote fixes | 74, 71, 72, 75, 76, 70 (step 1), 79, 82 | −260 | low | mock server |
| M7 llmd weights | 84, 83, 86, 89, 90, 91, 92, 88 | −215 | low | llmd tests |
| M8 llmd load orchestration | 98, 97, 99, 96 | −150 | low | GPU smoke |
| M9 Examples teach one recipe | 102, 103, 104 | −75 | low | — |
| M10 One test oracle; Buffer.from follows the mesh | 26, 93 | −110 | low | GPU tests |
| **Wave 2** | | **≈ −775** | | one GPU A/B |
| M11 Admission by fixed arrays | 0 | −35 | low | GPU |
| M12 One arena per direct batch | 38, 39, 29, 32, 33.1 | −100 | low | GPU A/B |
| M13 Planner representation | 40, 41, 42, 35, 33.2, 44 | −75 | low | GPU A/B |
| M14 One scatter, one block primitive | 48, 45, 47, 49 | −105 | low | GPU A/B |
| M15 Gates and shutdown | 46, 43, 37, 34 | −100 | medium | GPU A/B |
| M16 Calibration returns its block | 50, 51, 54 | −50 | low | one CUDA or B70 run |
| M17 Buffered backend | 24, 27, 22, 25 (23 as fallback) | −310 | medium | CPU; TPU if available |
| **Wave 3** | | **≈ −110** | | |
| M18 Completion self-checks PJRT already does | 28, 30, 56 (half) | −70 | medium | failure injection on CUDA, ROCm |
| M19 One host-memory arena list | 52 | −40 | low | MI300X |
| **Outside the rework** | | | | |
| M20 Pre-existing llmd cleanups, separate PR | 95, 100, 101 | −270 | low | per-model output checks |

"C3" is the critic's third missed simplification. Dependencies: M3 before M11, M12 and M17 (borrowed specs). M4 before M7 (`Placed`). M1 before M8 (`bufferize(anytype)`, `addRepo("drafter")`). M5 before M6 (`stdx.log.scoped`). M13 before M14 (`landing`). M14 before M15 (one failure check inside `readGated`).

### Wave 1: no measured path changes

**M1. Registry, mem and Partitioning edges** (60, 57, 58, 59, 61, 62, 56's `Device.attribute` half, C3)
- **Goes:**
  - dead code: `FixedBufferPool`, `Buffer.freeDeviceMemoryButKeepHostMetadataMemory`, `TensorRegistry.totalBytes` and `.iterator`, `Device.attribute`;
  - the one-use index pipeline (`SafetensorsIndex` and two parsers), the scratch `allocator` on three functions (scratch comes from the registry), `resolveModelRepo`'s file mode and second catch switch, and the never-returned `Allocator.Error` arm of `ModelPathResolutionError`. Only `fromPath` tests the file suffix;
  - `addRepo`'s rule that the caller adds the dot `View.withPrefix` adds;
  - the pub `readFilePositionalAllV` and `max_positional_iovecs`, which move beside `readGated` (zml/safetensors.zig:12-150), so O_DIRECT's padding rule lives with its only producer;
  - `deinitBufferized`'s type argument and the type restated at about 132 call sites; `pairBufferized`'s context and callback (it returns `[]Pair`); `bufferize`'s second rule for mapping struct fields (a latent tuple compile error);
  - `Partitioning.parse`'s second struct-literal loop (it becomes `Logical.init(p).resolve(...)`);
  - `Shape.partitioning()` and `Logical.set` (zml/shape.zig:1017-1049, zml/Sharding.zig:528-540). Their one production caller, `linear.withLayout`, has just applied `withTags(layout)`, so it can pass `layout._sharding.partition` positionally: `return tagged.withPartitioning(mesh, layout._sharding.partition);`.
- **Stays true:** every registry constructor, `bufferize`'s rollback, `MapRestrict`, the field-path mismatch report, `Logical` itself (linear.create stores, splits and passes it on).
- **Net:** about −250 (60: −69, 57: −90, 58: +3, 59: 0, 61: −15, 62: −26, C3: −45, attribute: −5).
- **Risk:** low, CPU. 61 is mechanical in both repos. 62 and C3 trim two of your own commits (819431cf and c865a2cb), see D9.
- **Measure:** nothing; none of this is on a load path.

**M2. The front end keeps lifecycle state** (3 with 20 and 33.3, 2, 4, 5, 14, 55)
- **Goes:**
  - the loader's reach into Program: `CompiledProgram.compile` and `.name`, five reads of the vtable outside Program, the pub `Program.VTable`, `CompiledProgram.inputs` and its borrowed-until-compiled rule, `BoundExecutable.exe` (undefined until publish), `samePlacedShape` and its test, `Program.init`'s zero-size branch;
  - `retire` (merged into `retireOldest` without its catch), and two of `failure`'s three writers;
  - "may be moved until its first start", in three places;
  - the borrowed `dma_block_sizes`: `backend.Options` holds a `BoundedArray` copy;
  - `Options.readWidth()` and the optional width at the seam: the front end resolves 16 or 32 once;
  - `program_group_bytes` as a pub field written after init: it becomes an option;
  - `dma_calibration.measure`'s copy of the platform list: it calls `backend.loadsDirect`;
  - `UnsupportedPlatform` in `Loader.Error`, and the direct backend's runtime device check: `Platform.validateDeviceCount` refuses more than `MAX_NUM_DEVICES`.
- **After:**
  ```zig
  pub fn cancel(self: *Loader) void { if (self.loadsStarted()) _ = self.fail(error.Canceled); }
  /// Records `err` unless the loader failed already, gives up on everything started, and returns `err`.
  fn fail(self: *Loader, err: Error) Error;
  fn retireOldest(self: *Loader, execute: bool) Error!void; // await, run, count; no failure write
  const BoundExecutable = struct { program: *CompiledProgram, inputs: []Buffer, output: *Buffer }; // exe = &program.exe.?
  ```
- **Stays true:** one compile per distinct program (`io.concurrent catch io.async`), the sticky first error, abort before drain, sizing fixed at init, calibration's option checks.
- **Net:** about −60. **Risk:** low, CPU. 55 narrows Platform's contract: a platform above the maximum with a custom subset mesh is refused at init. None exists in tree, and the default path improves from a panic in `registerMesh` (zml/platform.zig:707) to a logged `InvalidDeviceCount`.

**M3. One seam contract** (12, 33.4, 7, 8 with 9 and 31's error half, 10, 11, 13, 15, 36). Land 12 and 7 before 8.
- **Goes:**
  - `ShardGrid.InitError` and its two returns that cannot fire; five fallible re-derivations in `TensorTransfer` and the comment explaining the front end's invariant; the buffered backend's memory-kind forwarding and its `expectMemory` test (it asserts `.default`);
  - `maximumJobLen`'s error return and its two impossible refusals, `create`'s re-check, `setUp`'s switch;
  - `backendError`, five of six error-set declarations and seven names that cross the seam. Range and gather refusals return `InvalidBinding` where they are logged; the cannot-happen cases assert where they are detected;
  - the copy clause of `LoadSpec`: buffered submit's part and dim copies, direct's `Batch.dims` and its memcpy. The pending submission keeps the specs' arena until retire. `publish` takes the popped group by value, so its `group.executables = &.{}` trick goes;
  - `LoadPart`'s and `Part`'s restated lattice fields and their `lattice()`, `LoadPart.Dim`, Source's second dims bound;
  - `checkStridedParts` and its pre-scan (one loop, one floor);
  - the `boxOf`/`boxOffset` pair (one `locate`);
  - direct submit's zero-plan branch and its Closed-race comment: every planning failure fails the loader, logged at err.
- **After:**
  ```zig
  const PendingSubmission = struct {
      submission: backend.Submission,
      arena: std.heap.ArenaAllocator, // the specs the backend reads until it is awaited
      executables: []BoundExecutable, // empty for a bulk load
      execution: []u64, // DeviceBytes after M11
      logical_bytes: usize,
  };
  pub fn init(shape: Shape) ShardGrid; // Placement.init catch unreachable; two asserts
  pub fn locate(self: *const ShardGrid, offset: u64) struct { usize, u64 };
  fn maximumJobLen(config: Config) usize; // asserts what Loader.init checked
  ```
- **Stays true:** `checkParts` at the front door, including the dims bound; `Loader.Error`; `InvalidBinding` as the one name for model bugs; all-or-nothing publication; `Part`'s `source_slot` and `destination` links.
- **Net:** about −165. 11 adds about 90 lines of mechanical churn (`part.width` becomes `part.lattice.width`).
- **Risk:** low. CPU, plus a GPU smoke test because the direct backend now borrows dims.
- **Measure:** no allocator call per source job (unchanged); planning time unchanged.

**M4. Placement is a type** (18, 19, then 16, 17, 21)
- **Goes:**
  - `Source.emptyOf`, `View.keepEmpty` and prepareSpec's per-piece empty check: one rule, "a source that reads no bytes";
  - `View.maybeCreateBinding`, a second way to bind a program with no llmd caller;
  - `_placed` and its 23 guards, `Bound`, `asBound` and bindProgram's conversion array. A misuse is a compile error at the call site, not a safe-build panic;
  - the loader's five reads of `_pieces`, `testParts` and the pub `lowerPiece` over a private type: `Placed.lower` is the one bits-to-bytes rule;
  - `View.hasPrefix` (`count() != 0` answers it) and `prefix()`'s optional, with its 13 unwraps.
- **Stays true:** `Piece` as a bit lattice, `Rewrite`, the op harness, Binding's two arms, `withPartitioning` and `as`, infallible binding, the `createTensor` shorthands, `whole()` on `Placed` as a test probe.
- The private machinery struct already named `Placed` (zml/io/Source.zig:669) is renamed. Do not add `dim`/`dtype`/`rank` forwarders on `Placed`: `.shape.` covers the ten reads.
- **Cost:** an empty checkpoint tensor bound through a shorthand no longer names its key (the `source()` path already does not); the sub-byte log loses the piece's tensor name.
- **Net:** about −95. **Risk:** low, CPU; about 15 llmd sites change a type to `Source.Placed`.

**M5. VFS core** (64, 66, 63, 67, 68, 69, 77)
- **Goes:**
  - `stripScheme` at 11 sites and the strip-after-lookup rule: `resolve(dir, sub_path)` returns the target and the path its backend takes, which fixes `dirRealPathFile` by construction; `std.Uri.parse`, `schemeRoot`, `localRoot`, the CWD placeholder;
  - the public `VFS.Remotes`, its in-place, no-move and deinit-order rules, and the second allocator: `vfs.addRemotes(environ)` heap-allocates them inside the VFS;
  - the backend-map mutex, which the unlocked read path never took: registration is setup only;
  - `LoadProfile` and its borrowed scheme name: `readHints(path)` returns the backend's `ReadHints`;
  - the plain `ReadStats`, its test-only `sub` and `snapshot`: one atomic record;
  - `decideBuffered`, `enterDirect`, `DirectEntry` and `Handle.read_only`: one `settle()` under the mutex, the access-mode check moved into `direct_io.enable`;
  - two copies of the test-quiet logger: one `stdx.log.scoped`;
  - `startMockServer`, `cleanupMockServer`, `serverPath`, and the caller-held group and joined flag at 16 sites: `MockServer` owns its serving group.
- **Stays true:** the VFS as an `std.Io`, `fromIo`, `isLocal`, one descriptor per handle with demotion, the cache probe, `writableDir`, the error contract, vfs.zig's undemoted logger, `log.failure` in zml/io.
- **Net:** about −250. **Risk:** low, CPU. Bugs fixed: 64's scheme prefix; 68's latent overwrite.

**M6. Request layer and small remote fixes** (74 first, then 71, 72, 75, 76, 70's step 1, 79, 82)
- **Goes:**
  - the second, unbounded policy for a server-named delay: `max_delay` bounds every per-request sleep (a bug: `Retry-After: 3600` on a 503 stalls a load for hours);
  - `request.Context`, four `requestContext()` helpers, four `client` and four `read_stats` fields, `RequestSpec.backend` and `.unavailable` (24 restated spec lines), the S3 and GCS `unavailable` constants and their tests: the Governor holds the client, the name, the stats and what a 503 means;
  - `Accept`, the pub `open_statuses` and five consume-side `openStatus` lines: `ExchangeOptions.open = true` maps them before `consume`; GCS's `accept_any`, `all_client_errors` and optional token;
  - `Attempt.ordinal` with MockServer's `peak_attempt`; `pub` on 11 internal helpers;
  - eight copies of the read-failure mapping and the `InputOutput` name: `Governor.readFailure`, logged through `stdx.log.scoped` so failure-path tests stay quiet;
  - `Attempt.progress`, `.waiting`, `setWaiting`, GCS's `watchedToken` and range_read's pause block: `watched` moves from `perform` into `exchange`, so a nested token refresh is watched by its own request;
  - HTTP's `SizeHop` walker, which overwrote its own URL and followed redirects the GET then forbids: stat is one HEAD with redirects forbidden;
  - GCS's copies of the ListBucketResult parser, the encode predicate and the query writer.
- **Stays true:** the hold as a deadline, the two budgets, `classifyStatus`, the inactivity stall, `retry_once` (decision 6), range_read's validation, `RequestSpec.key` (until D2), HF's redirect walker, one whole Range GET per positional call inside the caller's credit (the fixed width and the credits depend on it).
- **Net:** about −260 (74 adds 16).
- **Risk:** low on the mock server. S3, GCS and HF have not run against a real remote since R15: run one before shipping.
- **Measure:** a mock-server load unchanged; a `Retry-After: 3600` answer returns within `max_delay`.

**M7. llmd weights** (84, 83, then 86, 89, 90, 91, 92, 88)
- **Goes:**
  - three per-expert key loops (one `expertStack`), three never-varying parameters on `stackExpert`, `bindExpert` and `bindExpertRowScales` (they move into `Projection`), the dead `.sc` tag and `multiplier` (`GlobalScale.asMultiplier`);
  - the `Stack` union, the unreachable `UnsupportedFp8NormalizationProgram` refusal (an assert on the layout: zml/kernels/moe/moe.zig:155-166 gives every block-FP8 scheme concatenated and plain), the Source-terms copy of the interleave, `bindExpertRowScales`' dead `.interleaved` arm;
  - `Backend.auto`'s unused dtype and two discarded reads; `ExpertSpec.gate_up`, which no caller sets;
  - `bindBlockFp8`'s three-field result and `swizzled_scales` in model code: `bindQuantized` returns the `Quantization` the GEMM reads;
  - `bf16Linear` in weights.zig (it becomes `linear.createBf16`), six pub helpers, four View accessors and two of three listings of a Linear's components (one `Component`), `View.name`, `swizzleBlockFp8Scale`'s comptime T;
  - 11 hand-built `Linear.init` routers, gates and heads, which also stop bypassing `linear.detect`;
  - five bare `log.err` and return pairs, the copied `is_test` switch, `UnsupportedWeightDtype`: two refusals, `mismatch` and `unsupported`;
  - `splitTensor`, its test, two `BoundedArray(Tensor, 8)` and six post-bind re-placements.
- **After** (with [83]'s sketch fixed for M4):
  ```zig
  const Projection = struct { names: []const []const u8, num_experts: u32, gate_up: zml.moe.ProjectionLayout,
      out: Shape.Tag, contracting: Shape.Tag, layout: ExpertLayout = .device };
  fn expertStack(scratch: Allocator, store: View, num_experts: u32, name: []const u8, subkey: []const u8) !Source;
  /// `landed` laid out for the kernel: by the loader from Sources (`.loader`) or by `LayOutExperts` (`.device`).
  fn layOut(projection: Projection, landed: Source.Placed, stacking: Stacking) Tensor;
  //   .loader => layOutExperts(Source, landed.unplaced(), ...).as(output).bind(),
  //   .device => Source.bindProgram(&.{landed}, output, program),
  ```
- **Stays true:** both ExpertLayout arms (decision 5), now one `layOutExperts` evaluated by either; the scale guard inside `NormalizeBlockFp8Weight/Scale` (decision 3); `linear.View`, `split`, `fuse`, `detect`, `fuseGroups`, `extractOutput`.
- **Net:** about −215. **Risk:** low, llmd's CPU tests.
- **Measure:** the `.device` program counts and bytes in weights.zig's expert tests (:1142, :1219) unchanged. Depends on M4.

**M8. llmd load orchestration** (98, 97, 99, 96)
- **Goes:**
  - `SpeculativeContext`, `addDrafter`, `hasDrafterCheckpoint` and its gate copied in main and zoo, the drafter's second resolve, DeepSeek's own "native" check, `CliArgs` in models.zig and 12 model signatures, zoo's fake `CliArgs`: `speculative.Drafter.open` decides once, before the store;
  - five `ModelBuffers`, `speculative.Tensors.initBuffers` and `speculative.Buffers`: `zml.Bufferized(Weights)`;
  - the `@hasDecl` `startLoad` default and its field-name contract, `finalizeLoadedBuffers` and `buffers_loaded` in 11 structs: one required `load` that starts, waits and bakes;
  - `engramHostPinnedBytes` with its platform switch, `Spec.offload_ngram_size`, the HACK comment copied wrongly into qwen4, and qwen4's `layers[i - 1].ple orelse @panic` walk: `store.hostMappedBytes()`.
- **Stays true:** two starts (base, then drafter) over one store before one wait (decision 6); the drafter under `"drafter."`; the loader scoped to main's init-and-load block; `deinitBufferized` as the only free; `extra_device_memory_bytes`.
- **Fallback for 99:** a required `startLoad` with the wait kept in the dispatch, and `buffers_loaded` dropped (about −8).
- **Net:** about −150. **Risk:** low. **Measure:** a GPU smoke run of llmd with a drafter; DeepSeek and qwen4 compile; `extra_device_memory_bytes` unchanged. Depends on M1.

**M9. Examples teach one recipe** (102, 103, 104)
- **Goes:** three `Mnist.load` methods and the README's text-versus-code contradiction; two hand load timers; the arena allocator handed to the loader; in examples/io, the `check_error` and `load_took` optionals, the errdefer-plus-late-defer pair, the hand byte recount and the dead `gibPerSecond`; optionally (104), the checks' reads of `bindingOf`, the Binding arms and `whole()`, which compare against the entries the harness bound instead.
- **Stays true:** examples/io's instruments (`ZML_LOAD_CHECK`, `StackPack`, the mesh modes, dma-bench), `LoadedModel.loadBuffers`.
- **Net:** about −75. **Risk:** low. **Cost:** CTX's older examples/io "with setup" figures are no longer directly comparable.

**M10. One test oracle; Buffer.from follows the shape's mesh** (26, 93)
- **Goes:** zml/io/testing.zig as a module of its own (it folds into `zml.testing`, with a rank-N `expectShards(..., memory)`); llmd's `pattern`, `expectShards`, `expectBytes` and registry copy loop; `expectContents`, buffered's `expectMemory`, three inline residency loops, six hand-rolled per-device read-back loops; `Devices.initCpu`, `initWith`, `force_cpu` and two comments about buffer.zig, because `Buffer.from` and its read-back siblings iterate the shape's mesh, not the platform's.
- **Stays true:** `Devices` on the accelerator when it has enough devices, `writeSafetensors`, `storeOf`, llmd's `Fixture`, `fuseGroups`, the memory argument (it checks something on accelerators).
- **Net:** about −110. **Risk:** low, but `Buffer.from` is used everywhere: run the GPU tests. Bug fixed: a buffered load onto a sub-mesh.

### Wave 2: measured paths change how they store their data

End the wave with one GB300 and B70 A/B against a HEAD baseline.

**M11. Admission by fixed per-device arrays** (0)
- **Goes:** `Scratch` and init's only allocation, `pending_execution` and `pending_executes` (pending costs are summed from `pending`), the per-group heap `execution` slice and its ownership hand-off across three functions, `Fit` and `measureFit`, init's `OutOfMemory`, the per-tensor PJRT stats calls on platforms that never compare them.
- **After:** `const DeviceBytes = [Platform.MAX_NUM_DEVICES]u64;` filled only while measured; `fn fits(...) bool`.
- **Stays true:** the room arithmetic, FIFO retirement, one program group in flight when unmeasured, `probeMemory` once.
- **Net:** about −35. **Risk:** low. **Measure:** `admission_retires`, `min_room_seen` and peak device use on a program-heavy load (DeepSeek FP8 on GB300). CPU never measures.

**M12. One arena per direct batch** (38, 39, 29, 32, 33.1)
- **Goes:** `Plan.allocator`, `Plan.create`, `Plan.destroy`, the take-on-success rule and its 29-line test, eight production errdefers, `JobCut.dma_submissions` and its three distant increments (the event count is derived from the DMA jobs, by job kind), `Batch.freeing`, `Batch.planning_source` (publishFiles' errdefer logs the file), `Scheduler.publish`'s `sealed_at` assert, LazyOnce's two context structs and trampolines and four reads of an `= undefined` field (`readyValue().?`), `Destination.hostResident`.
- **After:** `fn finish(self: *JobCut, arena: Allocator, source_slot: *SourceSlot) !*Batch.Plan;`
- **Stays true:** requests and events preallocated per plan; no allocator call per source job; submit's two seal paths; LazyOnce's racing first-touch semantics.
- **Net:** about −100. **Risk:** low. **Measure:** no allocator call per job (debug allocator count); plan metadata held until retire (a few MB on DeepSeek).

**M13. Planner representation** (40, 41, 42, 35, 33.2, 44)
- **Goes:** `Kind.dma_staged` and `Staged` (one `dma_land` kind: an extent lands its items), `Plan.direct_reads`, Gather's item fields; Transfer's `(block_index, block_offset)` pair and its second meaning for host jobs (one `landing`); the `class_order` scratch (the delivery class is a sort key); `Config.alignment`'s three meanings, the direct Loader's `direct_io` field and the per-submit `Reads` (built once in create); `readLimit` and `read_limit`; two stale byte-fair claims.
- **After:**
  ```zig
  const Kind = union(enum) { dma_read: Read, dma_land: Land, dma_fill, host_direct: Read, host_bounce: Read, host_fill };
  const Transfer = struct { destination: *Destination, landing: usize, writer_mask: u64, destination_offset: usize, len: usize };
  fn maximumJobLen(request_size: usize, alignment: usize) usize;
  ```
- **Stays true:** `dma_read` as its own kind (no run walk on the hot path), placeDma's estimate and staged landing, cutRuns, per-file publish, fill jobs, `checkGatherReads` before publish with its conservative bound, per-file alignment that never opens a remote file.
- **Net:** about −75. **Risk:** low; one div and one mod per transfer at enqueue.
- **Measure:** `planned_dma_submissions` (about 5,468 on DeepSeek), reads (about 9,524), planning time (1-4 ms), the delay to the first read.

**M14. One scatter, one block primitive** (48 first, then 45, 47, 49)
- **Goes:** `FileReader`, `BytesReader` and the `anytype` reader (CompositeTest replays from two tmp files through the real `readGated`); `HostScatter`, `copyToBlocks`, `copyFromBlocks`, `Scratch.host_iovecs`, one of two identical block-iovec loops; `Scratch.queue_counts` and run's reference rollback (`Dma.enqueue` counts its pieces and takes its references after every reservation); retireBatch's unlink pass and `EventContext.err` (the callback frees the PJRT error it owns).
- **After:** a concrete `Scatter{ loader: *Loader, slot, iovecs, ... }` with `add`, `place`, `flush`; `fn blockPiece(blocks, block_size, at, len) []u8`; `fn enqueue(request: *ReadRequest, blocks: []const Block) !void`.
- **Stays true:** walkItem/walkRuns, copyItem's stretch path, the bounce buffer, the pumps, the retired stack, one event never destroyed in its own callback.
- **Net:** about −105. **Risk:** low. **Measure:** about one DMA submission per tensor, `dma_submit_us_per_piece`, `pump_stops_full`, host-resident load time. Depends on M13.

**M15. Gates and shutdown** (46, 43, 37, 34)
- **Goes:** `RequestGate` (an `std.Io.Semaphore`), `Credits.close` and acquire's optional, four scattered failure checks (one inside `readGated`, after the permit), `unclaimed_total` (it repeats `queue.len != 0`), `Scheduler.stop`, `error.Closed` from the vocabulary (publish returns `Canceled`; log.zig's Closed clause goes), six hand-encoded first-error atomics (one `stdx.FirstError(E)` and the decode-only accessors), two of three sizing narratives in direct's docs.
- **After:**
  ```zig
  read_permits: std.Io.Semaphore, read_width: usize, first_error: stdx.FirstError(StickyError) = .{},
  fn stopWorkers(self: *Loader) void { self.scheduler.fail(self.io); self.worker_group.await(self.io) catch {}; }
  ```
- **Stays true:** width + 1 workers with width permits (exact `read_parallelism`), credits as fixed slots bounding pinned memory, a credit before a claim, fail's claimed/unclaimed split.
- **Risk:** medium. A leaked credit after a failure would now hang destroy; the debug assert `inUse() == 0` stays.
- **Measure:** reads in flight peak at exactly `read_parallelism`; credits high water; credit wait against read time; DeepSeek GiB/s on GB300 within noise; destroy returns after an injected failure.
- **Net:** about −100.

**M16. Calibration returns its block** (50, 51, 54)
- **Goes:** `Result` and `max_in_flight_per_device` (the depth is the constant `dma_calibration.depth`, and "calibrated depth" leaves four docs), three confirmation knobs nobody sets (private constants), isCloseToPeak's margin parameter, Candidate's copies of the session's handles, its dead `buffer_count` and its per-candidate error, `Report` and `measureTransfer`, the stale "allocators still warmed" doc, `Workspace.max_mapped_bytes`.
- **Stays true:** the screen and its confirmation windows (decision 1), the ring kept as the first pinned capacity, the five bench knobs, runWindow's io parameter (the cancellation test needs it), the 16 GiB ceiling.
- **Net:** about −50. **Risk:** low. **Measure:** one CUDA or B70 run picks the same block (the `dma_bench` line).

**M17. Buffered backend** (24, 27, then 22 with 25; 23 as a fallback)
- **Goes:** `StagingAdmission` and `StagingPool` (one `Staging`: live plus retained bytes within one budget, except a request that runs alone), `peak_live_bytes` and release-time eviction, the uncounted per-strided-part allocation; `loadOneStaged` and its two out-parameters; `LimitedGroup` from stdx entirely with its five tests, `submitOne`, `Batch.pending`, `done`, `finish` and the publish sentinel; helper reads that run inline.
- **After:**
  ```zig
  pub const Batch = struct { specs: []const LoadSpec, progress: std.Progress.Node,
      next: std.atomic.Value(usize) = .init(0), workers: std.Io.Group = .init };
  // submit: W = @min(self.width, specs.len) workers over the cursor
  //   for (0..W) |_| batch.workers.concurrent(self.io, work, .{ self, batch }) catch break;
  //   none started (an io without concurrency): work(self, batch) inside submit, as today
  // awaitBatch: workers.await under cancel protection, uncancelable like the direct backend's
  ```
  Never `concurrent catch async`: Threaded runs `async` inline once busy, and an inline first worker would drain the whole cursor inside `submit` (the critic's point). Helpers (25) store the raised claim width before spawning and count only those that started.
- **Stays true:** ChunkedRead and ReadPermits (CTX.md:2095 rejected removing them), staging reuse, the 12× budget with `min(width, 12)` slots, gap zeroing, read once then `copyRuns`, the sticky error with `checkOpen`, one progress item per spec, decision 7.
- **Risk:** medium. CPU-testable only; TPU, neuron and metal are unverified. Dense checkpoints may now retain more staging, up to the budget (unmeasured). Cross-batch concurrency can reach 2W workers, still bounded by permits and the budget.
- **Fallback:** 23 alone (revert LimitedGroup to master, −158, no behaviour change).
- **Measure:** add a strided part to the composite test; on a TPU host, staging high water within the budget and a serial load of the 14.96 GiB checkpoint.
- **Net:** about −310 (LimitedGroup is 215 of it). Depends on M3.

### Wave 3: needs hardware

**M18. Completion self-checks PJRT already does** (28, 30, 56's setBufferError half)
- **Goes:** awaitBatch's `SetBufferError` pass, `Target.closed`, the double-failure CHECK abort (one of the CHECK paths CTX records, including the fifth pass's oneAPI abort); `HostWritten`, `host_written` and `host_total`; pjrt's `setBufferError` and `setBufferErrorUnknown`, then uncalled.
- **Stays true:** `Target.total`, `submitted_bytes`, `nextIsLast`, `fullySubmitted` for DMA targets, the `Internal` failure for a batch that missed a last transfer.
- **First:** write the test that a failed default-memory output's await errors (it passes today through `SetBufferError`).
- **Risk:** medium. Failure injection on CUDA and ROCm. Furiosa, the one non-XLA direct target, is unverified on both paths. **Cost:** no release-build check that a host destination's bytes were all written; coverage rests on the CPU content tests.
- **Net:** about −70.

**M19. One host-memory arena list** (52)
- **Goes:** `HugePageAllocator`, `PinnedHostAllocation`, the `PjrtHost` and `Pages` wrappers, the two per-arm lists and `arenaCount`/`arenaAt`, the `discovered_nodes` copy, the running `mapped_bytes`.
- **After:** `Workspace.arenas: std.ArrayList(Arena)`, `Arena{ bytes, pjrt_buffer: ?*pjrt.Buffer }`, `Backend.map/unmap`; reserve the list's capacity before mapping, because the ROCm balancer counts only retained arenas.
- **Stays true:** Workspace's pub API (R14), each arena kind with its measured reason, NUMA interleave masked by the task policy, THP advice, the arena leaked when DmaUnmap fails, the ROCm byte balancer.
- **Risk:** low, but the `pjrt_host` arm runs only on ROCm and Furiosa. **Measure on MI300X:** `pinned_mapped`, `pregrowth_ms`, bytes per node.
- **Net:** about −40.

### Outside the rework

**M20. Pre-existing llmd cleanups, separate PR** (95, 100, 101). One dense gated `Mlp` beside `llama.RmsNorm` for nine models (not in weights.zig, which holds no layer math); `ModelParallelism` as a plain struct without its never-built `.replicated` arm, `Kind`, `ReplicatedParallelism`, six forwarders and the dead `runtime`/`experts` fields; and the always-empty `BakedInputs` channel. About −270. They predate the rework; 95 needs output checks on each model.

## Part 3: decisions for you

**D1. Should `wait` release the backend? (6)**
- For: callers write `defer loader.deinit()` at function scope; examples/llm's optional loader and manual deinit go, and so do llmd's scope block, the per-wait resets, "since the last wait" and the idempotent-wait contract. About −14.
- Against: the loader gains a terminal state and a "no start after wait" panic. LOADER_API_REVIEW.md:439 rejected this as "a second lifetime rule", and that reason still holds. examples/io's timer would include the teardown.
- My take: decline. The only new evidence is examples/llm's optional.

**D2. Should the inert rate-limit key go? (73)**
- For: nothing reads it (`Governor.holdFor` ignores it). It is host-only where PLAN.md:160 planned host:port. It names the old host after an HF re-resolve. A per-authority scope needs a keyed collection anyway, so the seam would be rebuilt. 13 call lines, about −45.
- Against: CTX.md:2121-2122 records it, in the design agreed with you, as what makes a per-authority scope "a local change later"; CTX.md:2235 leaves that call open.
- My take: delete it. Behaviour is unchanged (one hold per backend instance), and the open call is not foreclosed.

**D3. Should `View.allocator()` provide scratch while binding? (85)**
- For: 18 MoE inits and 9 helpers carry an allocator only to pass freed scratch to `weights.experts`. Source ops already allocate in the store.
- Against: monorepo/AGENTS.md:81 says to pass allocators explicitly, and R11 chose the parameter. After M4 deletes `maybeCreateBinding`, zml no longer takes scratch from the store anywhere. About −10.
- My take: a reading gain, not a line gain. Take it only with an explicit exception in AGENTS.md.

**D4. Should the comptime remote core land now? (78, with 70's full reshape, 80, 81)**
- For: the largest deletion, about −1,030. One handle semantics in vfs/remote.zig over five hooks, in place of four handle tables and about 64 hand-written vtable functions. Three latent defects live in the copies: listing cursors keyed by a movable reader address, a write passed to the inner io with a foreign fd, and HF's table lock held across network work. VFS, VFSBase and its forwarders stay (decision 6). With it, `perform` takes `prepare(previous)` and `consume`, and `Attempt`, `RangeSpec`, `PreparedRequest` and `prepareStatic` go.
- Against: it is the backend half of LOADER_REVIEW Move 10, which the audit left pending as high risk. S3, GCS and HF have not run against a real remote since R15, and HF has no test endpoint. It adds a comptime contract to learn.
- My take: take it, staged: HTTP under http_acceptance_test, then S3 and GCS under the mock tests and zml/safetensors.zig:927, then HF (with 80 and 81) after one real HF run.

**D5. Should ROCm's growth split move into host_memory? (53)**
- For: the loader loses an argument and a comment that exist only for ROCm placement; CUDA, oneAPI and CPU map one arena.
- Against: it reverses R14's keep. Neither shape balances: today's split is 41/59 on 8× MI300X (CTX.md:2183), and the proposed one is about 60/40. CTX.md:1213-1219 measured imbalance as slower. Neither has run on a GPU since R14.
- My take: not without an MI300X run. If you run one, take the variant that sizes one arena per node to its deficit from the mean, which actually balances.

**D6. Should the loader place a program's result? (87)**
- For: program authors lose a rule ("return a tensor of `output`, placed alike"). Four `withLayout(x, output)` calls, two hand restatements of a placement (one with an apology in examples/io) and checkOutput's 22-line per-device walk go. About −25. I checked the compiler half: the result sharding comes from the returned tensor's shape (zml/Compiler.zig:556, :661), so relabelling it is compiled as asked.
- Against: TensorStore would write Tensor's private `_shape._sharding`. A misplaced program becomes a silent reshard inside a load program (a hidden collective) instead of `InvalidBinding`. A rank mismatch must be skipped. Using `withPartitioning` instead drops donations, because it makes a new tensor id (zml/tensor.zig:194-223).
- My take: decline. The rule is local to six of llmd's own programs, and the check it would delete guards a performance trap.

**D7. ExpertLayout (decision 5).** After M7 the two arms differ only in who evaluates `layOutExperts`. Choosing `.device` would delete layOut's Source evaluation, `Projection.layout` and the `.loader` test loops. Choosing `.loader` would delete `LayOutExperts` and its program. Nothing here depends on the choice.

**D8. What ships with the branch.** Nine root files add about 7,600 lines to the zml diff: CTX.md, LOADER_REVIEW.md, LOADER_API_REVIEW.md, PLAN.md, RESEARCH.md and four bench scripts (`bench_s3.sh`, `bench_aws.sh`, `bench_file.sh`, `profile_file.sh`). No build file or source references them. The measurements the keeps rely on are already quoted beside the code. My take: keep docs/learn/loader.md, and keep the logs, reviews (this one included) and scripts out of the merge.

**D9. Two of your own commits.** M1 deletes `Shape.partitioning()` and `Logical.set` (c865a2cb) and `Partitioning.parse`'s second loop (819431cf). Each has one production caller or none beyond `linear.withLayout`. If you want them kept, M1 loses about 70 lines.

**Recorded decisions this design touches, and how:**
- Decision 1: only calibration's API narrows (M16); the screen and its windows stay.
- Decision 2: start/wait stays; the loader still compiles and runs the programs (M2 moves compile onto `Binding.Compute`).
- Decision 3: the scale guard stays inside the conversion program; only the unreachable expert refusal becomes an assert (M7).
- Decision 4: `View.setMemory` is untouched.
- Decision 5: D7.
- Decision 6: two starts over one store (M8), `retry_once` (M6, and D4's `previous`), the forwarders and the handle-table move (D4 is only its backend half), the GCS service-account refusal: all kept.
- Decision 7: both backends stay; the buffered one loses only machinery (M17) and a memory-kind path it never receives (M3).

## Part 4: what earns its place

**Keeps.**
- start/wait/cancel/deinit with one sticky first error, and `cancel` as the exit for caller code that fails between start and wait.
- The set-up future plus `?Backend`; `probeMemory` decided once (the CPU plugin warns on every `memoryStats`).
- `prepareModel` emptying every buffer before any check, so `deinitBufferized` is correct after any error.
- prepareSpec and checkPlacement as the one check of the LoadSpec contract; `checkParts` at the trust boundary.
- Program groups up to `program_group_bytes`, queued before the bulk so compiles overlap reads; compiles through `io.concurrent catch io.async`.
- checkOutput's per-device comparison (see D6).
- Group and PendingSubmission as two phases with two owners (the merge was rejected below).
- Source's pieces as flat affine bit lattices, `Rewrite`, and the op harness with its DoubleWrite replay: the only proof of the disjointness the loader trusts.
- Binding's two arms; `Memory{default, host_pinned, host_mapped}` with `space()`; the createTensor one-liners; `withPartitioning` and `as`; `Source.store`; `Options.quantization`.
- The Backend and Submission unions; `Lattice` and `copyRuns` as the one strided kernel; Lattice's gather predicates; ShardGrid's arithmetic API; `max_read_request_size`; `readAlignment(io, direct_io)` as a function.
- The direct Loader as the single root; `Batch.remaining` with the publish sentinel; Destination versus Part; `LazyOnce`; `Target.total` and `nextIsLast`; the allocation task; `SharedHostAllocation`; credits as fixed slots; the two-arena carve and pre-growth before any read (see D5); Metrics and the summary lines CTX cites.
- Per-file planning and publishing; coalesced runs cut at tensor-safe boundaries; placeDma's estimate and staged landing; `dma_read` as its own kind; the DMA/host split with `host_direct`; fill jobs; FIFO claims; credit before claim; requests and events preallocated per plan; `minimum_staging_gain` as a test seam.
- Per-device pumps with the byte budget, the 64-piece cap and one submitter per device; the retired event stack; walkItem/walkRuns and copyItem's stretch path; the page-aligned bounce buffer.
- `Workspace`'s five calls; per-target arena kinds; NUMA interleave; THP; the 16 GiB ceiling as a constant; the calibration screen; `warmupDeviceAllocators`; `Platform.placedShape`; `Exe.call` returning `pjrt.ApiError`.
- ChunkedRead and ReadPermits; staging reuse; the 12× budget; gap zeroing; the buffered sticky error.
- The VFS as an `std.Io` with `fromIo`; `isLocal`; one descriptor per handle with demotion; the cache probe; `writableDir`; the remote error contract; std streams at indices 0-2; `ReservedScheme`; vfs/BUILD.bazel's shared sources.
- The request policy core: the hold as a deadline, two budgets, `classifyStatus`, the inactivity stall, `exchange`'s closing of failed connections, `Redirects`, range_read's validation, one Range GET per positional call, `OpenError` and `openFailure`; S3's SigV4 on every attempt; GCS's token cache; HF's 32 MiB profile; the three `.auto` constructors.
- `safetensors.Tensor`'s four fields; every registry constructor; `lockPointers`; `Bufferized` and `pairBufferized`'s structural check; `MapRestrict`; `Quantization.Target`; `Partitioning.Logical`.
- llmd: `linear.View`/`split`/`fuse`; `detect`; `View.source`'s per-component granularity; `bindQuantized`'s bind-time platform choice (R9); `weights.experts`/`moeBackend`; `extractOutput`; the NVFP4 global programs; `testing.Fixture`; the store built in a labeled block; `drafterStore` rooting at `View.root()`; the loader scoped to init and load; `deinitBufferized` everywhere; the text-store prefix switch; qwen4's and DeepSeek's `host_mapped` bindings; zoo's `_ = &Model.loadBuffers`.
- Tests: injecting failures through `std.Io` vtable overrides and FailingAllocator, never hooks in production code; admission's pure arithmetic tests (the only CPU coverage); the submitted/allocated counter pair and its absolute per-device checks; MockServer's failure knobs; log.zig's test demotion.

**Keeps re-raised, and why their reason no longer holds.**
- Fixed `[MAX_NUM_DEVICES]` arrays (45th pass, CTX.md:4430-4432, "per-device slices leave the buffered backend's device ids uncapped"): only a measured loader fills them, measuring needs the direct backend, and after M2 the platform itself refuses more devices (M11).
- `backendError` (37th pass, CTX.md:4042-4045): its reason, that a backend returns the error that closed it and init checked the options, is exactly why those names are unreachable (M3).
- `View.hasPrefix` (R7, LOADER_API_REVIEW.md:308, "the one real prefix question"): `View.count` predates R7, and the one production call is `hasPrefix("")` (M4).
- `Source.emptyOf` (43rd and 45th passes): their reason, that the empty case passed by the wrong rule, held only while there were two rules (M4).
- The direct backend's device check (LOADER_REVIEW item 9, "Platform.init only warns and keeps every device"): its premise missed the replicated-mesh registration, which panics on every in-tree physical mesh above the maximum (zml/platform.zig:389, :707) (M2).
- The closable read gate and credits (LOADER_REVIEW Move 2, the sixteenth pass at CTX.md:2092, G-keep-credits): both bounds stay; close bought liveness for a throttle watch that moved into the VFS (M15).
- "The dispatch owns completion" (LOADER_REVIEW Move 6): that keep chose among three awaits, and `loadInto` is gone; each model's `load` holds the only wait (M8, with a fallback).
- The FNUZ refusal for program-laid-out experts (49th-pass audit, CTX.md:4676): no backend lays out block-FP8 experts that way (zml/kernels/moe/moe.zig:155-166); the assert keeps the rule if one ever does (M7).
- `Shape.partitioning` (the registry surface's keep of a runtime tag-keyed placement): the reason holds for `Logical`, which stays; `Shape.partitioning`'s one caller can pass the positional partition (M1).
- `whole()` (TensorStore surface keep, for examples/io): examples/io stops needing it (M9), but llmd's tests still use it to name the key a tensor bound, so it stays on `Placed`.

**Rejected along the way.**
- [87] and [6]: D6 and D1.
- Deleting `whole()` (critic, delete-first): about 16 test lines in both repos use it as a probe; replacing each with `lower(arena)[0].source.name` needs an arena and reads worse.
- One name for a denied file (critic): std itself returns `AccessDenied` for EACCES and `PermissionDenied` for EPERM (std/Io/Threaded.zig:3364-3365), so every match on "denied" keeps both names; renaming the remote half removes no match arm and changes jinja's absent rule for a local EACCES.
- `concurrent catch async` for the buffered workers ([22]'s sketch): replaced in M17.
- The remote core inside the design (delete-first's M10): held as D4, because it is the pending half of Move 10 (decision 6).
- Delete-first's 9-to-13-finding moves: split so each lands alone and each says what to measure.
- `View.createReplicatedTensor` folded into `createTensor` ([91]'s follow-up): it re-raises C-keep-createTensor-front-door with no new evidence.
- The pre-existing llmd cleanups inside the rework: kept as M20, a separate PR.

## Unverified leads

- **Areas nobody read closely:** Source.zig's Rewrite core (zml/io/Source.zig:638-1000) for local cuts; ../monorepo/llmd/models/dflash_drafter.zig and dspark_drafter.zig; the direct backend's Metrics, logBatch, logSummary and Diagnostics, kept without sizing; vfs/http_acceptance_test.zig beyond [67] and [77].
- **S3's "403 may mean absent" rule is enforced in three places:** vfs/s3.zig:762-765, zml/safetensors.zig:883-897 and ../monorepo/llmd/chat-template/jinja.zig:237, :257. Whether one owner could hold it is unchecked; S3 itself cannot tell absent from denied.
- **Listing truncation (bug lead):** S3 and GCS read no `IsTruncated` and send no continuation token, so a prefix with more than 1000 keys lists partly ([82]'s verifier). HF's tree fetch uses `limit=1000` with no cursor follow ([81], not checked without network).
- **`AllocatingLimitedConcurrentGroup`** (stdx/Io.zig:241-295) has no caller at HEAD or at master. It predates the rework; deleting it is about −56.
- **Behaviour no test covers:** the manager destructor on Furiosa (M18); retained staging on dense checkpoints (M17); the `pjrt_host` arena arm (M19); HF's re-resolve (M6, D4); S3, GCS and HF against real remotes since R15.
- **The XLA copy** I read for [28] sits in the bazel cache; I did not check that it is the pinned revision.
- **Line counts** are the verifiers' estimates, netted by hand where findings overlap. [11]'s mechanical churn and [61]'s 132 call sites are counted as zero net.

## Appendix A: every verified finding

Each entry: id, title, verdict, main locations, proposal, and where this design puts it.

- **[0] Admission by fixed per-device arrays, filled only when measured.** Confirmed. zml/io/loader.zig:150-176, :560-627, :648-708. `Scratch`, the running pending sums, the per-group heap slice and `Fit` go; `DeviceBytes = [MAX_NUM_DEVICES]u64` is filled only while measured and pending costs are summed from `pending`. Init allocates nothing. → M11.
- **[2] One writer for `failure`.** Confirmed. zml/io/loader.zig:286-302, :470-475, :672-698. `fail` records the first error and tears down; `cancel` calls `fail(Canceled)` when loads started; `retire` folds into `retireOldest` without its catch. → M2.
- **[3] Program answers name, eql and compile; one Exe handle.** Partial. zml/io/loader.zig:751-848, zml/io/TensorStore.zig:79-160. `Compute.compile` builds its input tensors locally; `Program.name` and `eql`; the vtable becomes private; `BoundExecutable.exe` and `CompiledProgram.inputs` go. `sameExecutable` stays on `Compute`, rewritten over `eqlWithTagsAndPartitioning`. → M2.
- **[4] Drop "may be moved until its first start".** Partial. zml/io/loader.zig:78-80, :119-122, docs/learn/loader.md:161. Nothing points into a Loader; delete the qualifier in three places, with no new promise and no new test. → M2.
- **[5] Copy the block list; one platform list; `program_group_bytes` an option.** Partial. zml/io/loader.zig:189-198, zml/io/backend.zig:31-54, zml/io/dma_calibration.zig:85-94. `backend.Options` holds a `BoundedArray` copy (at most 16); `measure` calls `backend.loadsDirect`; `program_group_bytes` moves into Options. → M2.
- **[6] `wait` releases the backend.** Partial. zml/io/loader.zig:311-325, :401-404, examples/llm/main.zig:75, ../monorepo/llmd/main.zig:408-427. Callers would write `defer loader.deinit()`; the reuse-after-wait machinery goes; a start after wait panics. → D1 (decline).
- **[7] `maximumJobLen` asserts what `Loader.init` checked.** Partial. zml/io/direct_loader.zig:2369-2378, :284-287, zml/io/loader.zig:262-273. `InvalidOptions` leaves the direct backend, `setUp` and `backendError`; three tests of an unreachable path go. The per-platform check and the overflow-guard removal were refuted. → M3.
- **[8] Delete `backendError`.** Partial. zml/io/loader.zig:914-929, zml/io/direct_loader.zig:84-101, zml/io/backend.zig:15-16, zml/io/ShardGrid.zig:29. Backends return one `backend.Error`: binding refusals are `InvalidBinding` where logged, broken invariants panic where detected. Needs [7] and [12]. → M3.
- **[9] SubmitError and AwaitError are one set.** Confirmed. zml/io/direct_loader.zig:100-101, zml/io/buffered_loader.zig:33-35, zml/io/backend.zig:15-16. One `Error` per backend and one union. Subsumed by [8]. → M3.
- **[10] The pending submission owns its specs' arena.** Confirmed. zml/io/backend.zig:56-59, zml/io/loader.zig:405-447, zml/io/buffered_loader.zig:82-131, zml/io/direct_loader.zig:534-597. Both backends borrow parts and dims until await; `publish` takes the popped group by value. → M3.
- **[11] One byte lattice.** Partial. zml/io/backend.zig:76-120, zml/io/direct_loader.zig:1173-1190. `LoadPart{source, lattice}` and `Part{source, source_slot, destination, lattice}`; the field is `lattice`, not `runs`; the dims bound stays in `checkParts`. → M3.
- **[12] Backends trust the front end's placement and memory checks.** Confirmed. zml/io/ShardGrid.zig:29-102, zml/io/direct_loader.zig:1303-1401, zml/io/buffered_loader.zig:217-228. `ShardGrid.init` and `TensorTransfer` become infallible; the buffered backend asserts `.default`. → M3.
- **[13] `checkParts` as one loop.** Partial. zml/io/backend.zig:342-377. One loop with one floor (a contiguous part's end, a strided part's start); the dims bound stays. The bug claim was refuted: gap code never sees a mixed spec. → M3.
- **[14] Resolve `read_parallelism` in the front end.** Confirmed. zml/io/backend.zig:31-54, zml/io/loader.zig:1186-1191. `backend.Options.read_parallelism: usize`; `readWidth()` goes; the 16/32 defaults live beside the option. → M2.
- **[15] `ShardGrid.locate`.** Confirmed. zml/io/ShardGrid.zig:108-120, zml/io/direct_loader.zig:2117, :2995, :3021. Every caller pairs `boxOf` and `boxOffset` on one offset; one walk returns both. → M3.
- **[16] Placement is a type.** Confirmed. zml/io/Source.zig:40-601, zml/io/TensorStore.zig:66-78. `Source.Placed` replaces `_placed`, its 23 guards, `asBound` and `Bound`; only `Placed` binds. → M4.
- **[17] Source lowers itself.** Partial. zml/io/loader.zig:1036-1079, zml/io/Source.zig:603-641. `Placed.lower(arena)`; `lowerPiece` private; `testParts` goes; the loader stops reading `_pieces`. No culprit out-parameter, no pub `key()`. → M4.
- **[18] Drop the empty-entry fake piece.** Confirmed. zml/io/Source.zig:557-566, zml/io/TensorStore.zig:346-366, zml/io/loader.zig:1052-1062. `emptyOf`, `keepEmpty` and the per-piece check go; one rule. No panic in `maybeSource`. → M4.
- **[19] Remove `View.maybeCreateBinding`.** Confirmed. zml/io/TensorStore.zig:411-422, examples/io/main.zig:430-440. One way to bind a program; the callers pass replicated sources to `bindProgram`. → M4.
- **[20] `samePlacedShape` is `Shape.eqlWithTagsAndPartitioning`.** Confirmed. zml/io/TensorStore.zig:93-99, zml/shape.zig:411-417. Delete the copy and its test. → M2, with [3].
- **[21] Drop `hasPrefix`; `prefix()` returns `[]const u8`.** Confirmed. zml/io/TensorStore.zig:324-344, ../monorepo/llmd/models/qwen3_5.zig:1396. `count() != 0` replaces the one call; 13 unwraps go. → M4.
- **[22] Buffered workers over a cursor.** Partial. zml/io/buffered_loader.zig:40-324, stdx/Io.zig:25-125. W workers per batch in an `std.Io.Group`; the sentinel and `LimitedGroup` go; await stays uncancelable. Spawned with `concurrent`, never `catch async`. → M17.
- **[23] Revert `LimitedGroup` to master.** Confirmed. stdx/Io.zig:26-239. Its runtime limit, uncancelable admission and five tests have no caller. Subsumed by [22]. → M17 fallback.
- **[24] One `Staging`.** Confirmed. zml/io/buffered_loader.zig:329-363, :464-553. Live plus retained bytes within one budget, the strided span counted inside it; peak-based eviction goes. → M17.
- **[25] Spawn buffered read helpers with `concurrent`.** Partial (bug). zml/io/buffered_loader.zig:271-281. Store the raised claim width before spawning; count only the helpers that started. → M17.
- **[26] `Buffer.from` iterates the shape's mesh.** Partial (bug). zml/buffer.zig:177, :263, :308, :335, zml/buffer/HostAccessible.zig:252, zml/io/testing.zig:19-41. One rule for shard order; `Devices.initCpu` and `force_cpu` go. → M10.
- **[27] Buffered `loadOne` records each failure where it happens.** Confirmed. zml/io/buffered_loader.zig:158-230. `loadOneStaged` and its out-parameters go; each step calls `fail(err, step, spec, part)`. → M17.
- **[28] Delete awaitBatch's `SetBufferError` pass and `Target.closed`.** Confirmed. zml/io/direct_loader.zig:406-452, :1252-1259, :3478. The manager destructor already errors every incomplete buffer; write the await-errors test first. → M18.
- **[29] Drop `freeing`, the `sealed_at` assert and `planning_source`.** Partial. zml/io/direct_loader.zig:1036-1082, :2523, :1637. The debug flag repeats the counter's assert; publishFiles' errdefer logs the file. The single-seal rewrite and moving progress to Batch were refuted. → M12.
- **[30] Drop `HostWritten`.** Confirmed. zml/io/direct_loader.zig:1285-1288, :3154-3173. The host side of `fullySubmitted` only re-sums the planner's own transfers. → M18.
- **[31] One Error per backend; keep the two counters.** Partial. zml/io/direct_loader.zig:100-101, zml/io/loader.zig:689. The error half folds into [8]; the signed-counter half was refuted. → M3.
- **[32] LazyOnce takes its initializer at the call.** Confirmed. zml/io/direct_loader.zig:3738-3778, :1115-1223. Two context structs and trampolines go; four field reads become `readyValue().?`. → M12.
- **[33] Four forwarders and copies.** Partial. zml/io/direct_loader.zig:1148, :2348, zml/io/loader.zig:834, zml/io/buffered_loader.zig:217. `hostResident` (M12), `readLimit` (M13), `BoundExecutable.exe` (M2) and buffered's host delivery (M3) go. Inlining `bindExpert` was refuted; folding `layOutExperts` is dropped for [83].
- **[34] State the direct sizing once.** Partial. zml/io/direct_loader.zig:6-21, :110-131, :199-253. Two of three sizing narratives and other files' constants leave the docs; the header stays the backend summary (R16). → M15.
- **[35] Build `Planner.Reads` once.** Partial. zml/io/direct_loader.zig:159, :275, :1628-1640, :1711. `Config` keeps one meaning; `Reads` holds `alignmentOf` and `boundOf`; the policy stays a loader option (CTX.md:1627). → M13.
- **[36] Direct submit: one catch branch.** Confirmed. zml/io/direct_loader.zig:307-344. Every planning failure fails the loader, logged at err, and drains the batch; the zero-plan branch goes. → M3.
- **[37] One `stdx.FirstError(E)`.** Partial. zml/io/direct_loader.zig:133, :665-694, zml/io/buffered_loader.zig:54, zml/io/dma_calibration.zig:498, zml/platform.zig:449-477, vfs/mock_server.zig:60. Six encodings become one, and the decode-only accessors go with them. → M15.
- **[38] One arena per direct batch.** Confirmed. zml/io/direct_loader.zig:945-1094, :540-545. `Plan.allocator`, `destroy`, the take-on-success rule and eight errdefers go. → M12.
- **[39] `JobCut.finish` builds the plan.** Partial. zml/io/direct_loader.zig:972-1002, :1747-1765, :1811. `Plan.create` and `dma_submissions` go; the event count is derived by job kind, not by destination memory. → M12.
- **[40] One `dma_land` kind.** Confirmed. zml/io/direct_loader.zig:824-929, :2911-2944. A staged job is a one-extent landing job; `Staged` and `Plan.direct_reads` go. → M13.
- **[41] A Transfer holds one landing offset.** Confirmed. zml/io/direct_loader.zig:931-943, :2261-2297, :3373-3396. The block split happens once, at enqueue; three host-only asserts go. → M13.
- **[42] The delivery class is a sort key.** Confirmed. zml/io/direct_loader.zig:1677-1745, :2380-2403. preparePlan cuts three slices of `order`; `class_order` and two filters go. → M13.
- **[43] Drop `unclaimed_total` and `Scheduler.stop`.** Partial. zml/io/direct_loader.zig:2497-2589, :656-663. `waitForWork` reads the queue; `stopWorkers` calls `scheduler.fail`. Routing it through `abort` was refuted. → M15.
- **[44] Delete the stale byte-fair claims.** Confirmed. zml/io/direct_loader.zig:2483, :1554. Jobs are claimed in file order. → M13.
- **[45] One scatter accumulator, one block primitive.** Partial. zml/io/direct_loader.zig:3046-3203, :2727, :2815. `HostScatter` folds into `Scatter`; `blockPiece`; one iovec scratch. Concrete over `*Loader` once [48] lands. → M14.
- **[46] Read gate as a semaphore; no close.** Confirmed. zml/io/direct_loader.zig:3574-3701, :2724-2855. One failure check inside `readGated`; both bounds stay. → M15.
- **[47] `Dma.enqueue` counts its pieces and takes its references.** Confirmed. zml/io/direct_loader.zig:2697-2752, :3374-3416. The rollback and `queue_counts` go; run and runHost share one ensure loop. → M14.
- **[48] Drop the generic reader.** Confirmed. zml/io/direct_loader.zig:2865-2902, :2911-2946, :5022-5103. `FileReader`, `BytesReader` and their copy of the EOF rule go; CompositeTest reads tmp files. → M14.
- **[49] retireBatch drains every retired stack.** Confirmed. zml/io/direct_loader.zig:3288-3356, :3519. The unlink pass and `EventContext.err` go; the callback frees its error. → M14.
- **[50] Calibration returns the block size.** Confirmed. zml/io/dma_calibration.zig:23-32, :68-70. `Result` goes; `depth` is a constant; three confirmation knobs become private. → M16.
- **[51] Candidate is plain data.** Confirmed. zml/io/dma_calibration.zig:169-227, :490-630. The Session owns the handles and the error; `Report` and `measureTransfer` go; runWindow keeps its io. → M16.
- **[52] One host-memory arena list.** Partial. zml/io/host_memory.zig:30-509. Four wrappers become `Arena{bytes, pjrt_buffer}`; reserve the list before mapping. → M19.
- **[53] `carve(block_size, total_blocks)`.** Partial. zml/io/host_memory.zig:382-429, zml/io/direct_loader.zig:232-237. ROCm's split moves into host_memory; neither shape balances. → D5.
- **[54] Drop `Workspace.max_mapped_bytes`.** Confirmed. zml/io/host_memory.zig:332-446. Only tests vary it; compare against the constant. → M16.
- **[55] Refuse platforms above `MAX_NUM_DEVICES` at init.** Confirmed. zml/platform.zig:35-48, :326-328, zml/io/direct_loader.zig:168-174, zml/io/loader.zig:107-113. `UnsupportedPlatform` leaves `Loader.Error`. → M2.
- **[56] Delete `Device.attribute`; fix `setBufferError`'s signature.** Confirmed. zml/platform.zig:186-190, pjrt/pjrt.zig:1585-1597. The dead accessor goes in M1; the bindings go entirely with [28] in M18.
- **[57] Registry construction.** Confirmed. zml/safetensors.zig:436-893. Inline the one-use index parse; scratch from the registry; only `fromPath` tests the suffix. → M1.
- **[58] `addRepo` takes what `withPrefix` takes.** Confirmed. zml/safetensors.zig:466-475, ../monorepo/llmd/speculative.zig:35. `addRepo(io, repo, "drafter")`. → M1.
- **[59] Move the exact scatter read beside `readGated`.** Confirmed. zml/safetensors.zig:12-150. Two pub declarations leave the format module; `readRange` uses std's loop. → M1.
- **[60] Delete dead pools and walkers.** Confirmed. zml/mem.zig:50-92, zml/buffer.zig:112-122, zml/safetensors.zig:500-514. `FixedBufferPool`, the keep-host-metadata free, `totalBytes`, `iterator`. → M1.
- **[61] mem walkers take `model: anytype`; `pairBufferized` returns pairs.** Partial. zml/mem.zig:19-220, zml/io/loader.zig:963-973. 132 call sites lose a type; one field-mapping rule; a comptime guard against a Tensor tree. → M1.
- **[62] `Partitioning.parse` is `Logical.init(p).resolve(...)`.** Confirmed. zml/Sharding.zig:413-441. The second struct-literal loop goes. → M1 (D9).
- **[63] `readHints` replaces `LoadProfile`.** Confirmed. vfs/vfs.zig:93-115, :266-281, vfs/base.zig:24-32. One hints type; the borrowed name goes; error lines name the checkpoint. → M5.
- **[64] Split a VFS path once.** Confirmed (bug). vfs/vfs.zig:491-530, :735-738. `resolve` returns the target and stripped path; fixes `dirRealPathFile`. → M5.
- **[66] The VFS owns its remotes.** Confirmed. vfs/vfs.zig:143-218. `vfs.addRemotes(environ)`; registration is setup only, without the mutex. → M5.
- **[67] One read-stats record.** Confirmed. vfs/base.zig:34-153. Delete the plain `ReadStats`, `sub` and `snapshot`; tests assert absolute counts. → M5.
- **[68] One locked direct-I/O transition.** Confirmed. vfs/vfs.zig:336-419, vfs/direct_io.zig:74-80. `settle()` replaces three helpers and a union; the access-mode check moves into `enable`. → M5.
- **[69] One test-demoting scoped log in stdx.** Partial. vfs/base.zig:8-22, zml/io/log.zig:14-35. `stdx.log.scoped`; `failure` stays in zml/io. → M5.
- **[70] One request seam.** Partial. vfs/request.zig:101-246, :405-431, vfs/range_read.zig:18-119. Step 1 moves `watched` into `exchange` (M6); the full prepare/consume reshape rides D4.
- **[71] The Governor is the backend's one request object.** Confirmed. vfs/request.zig:70-146, vfs/http.zig:415, vfs/s3.zig:793. It absorbs `Context`, the client, the stats, the name and the 503 meaning; `RequestSpec` keeps target, key, `retry_once`. → M6.
- **[72] `exchange` maps the open statuses.** Confirmed. vfs/request.zig:374-417, :555-567, vfs/gcs.zig:276-364. `ExchangeOptions.open`; `Accept`, `open_statuses` and five consume lines go. → M6.
- **[73] Delete the inert rate-limit key.** Confirmed. vfs/request.zig:39-90, :139. `Key`, `authorityOf`, `holdFor` and 13 call lines. → D2.
- **[74] Cap a server-named delay at `max_delay`.** Confirmed (bug). vfs/request.zig:195-198, :508-512. → M6.
- **[75] Delete the test-only `Attempt.ordinal`; drop pub on internals.** Confirmed. vfs/request.zig:107-109, vfs/range_read.zig:12-224. → M6.
- **[76] One mapping for a failed remote read.** Partial. vfs/http.zig:166-172, :311-322 and the same in hf, s3 and gcs. As `Governor.readFailure`, logged test-quietly. → M6.
- **[77] MockServer owns its serving group.** Confirmed. vfs/mock_server.zig:1-348, 16 call sites. `start`, `stop`, `deinit`, `path`; dead members go. → M5.
- **[78] One comptime remote-file core.** Confirmed. vfs/http.zig:117-347, vfs/s3.zig:323-599, vfs/gcs.zig:482-758, vfs/hf.zig:248-839. `remote.vtable(Store, hooks)` over five hooks. → D4.
- **[79] HTTP's stat becomes one HEAD.** Partial (bug). vfs/http.zig:349-412. Delete the aliasing walker; a shared walker waits for LOADER_REVIEW #13. → M6.
- **[80] An HF handle keeps its URLs until close.** Partial. vfs/hf.zig:851-933. A per-handle arena; the per-read copy goes. → D4 (HF stage).
- **[81] HF's root is a TreeNode.** Confirmed. vfs/hf.zig:131, :459-585. One walker, one encoding of "directory"; the root's size counts subfolders. → D4 (HF stage).
- **[82] S3 and GCS share one ListBucketResult parser.** Partial. vfs/s3.zig:636-744, vfs/gcs.zig:780-889. Parser, encode predicate and query writer live in s3.zig. → M6.
- **[83] One expert layout expression.** Confirmed. ../monorepo/llmd/weights.zig:159-457, :688-761. `layOutExperts` evaluated as Sources or by the program; `Stack` and the unreachable refusal go. → M7.
- **[84] One per-expert key loop.** Confirmed. ../monorepo/llmd/weights.zig:225-413, :564-574, :725-748. `expertStack`; `Projection` carries the layout and axes; `multiplier` goes. → M7.
- **[85] Expert scratch from `View.allocator()`.** Confirmed. ../monorepo/llmd/weights.zig:67, 18 model inits. → D3.
- **[86] `Backend.auto` loses its dtype; `ExpertSpec.gate_up` goes.** Confirmed. ../monorepo/llmd/weights.zig:88-100, zml/kernels/moe/moe.zig:89-91. → M7.
- **[87] The loader places a program's result.** Partial. zml/io/TensorStore.zig:125-172, zml/io/loader.zig:1193-1232. `forward` loses `output`; checkOutput's walk goes. → D6 (decline).
- **[88] Delete `splitTensor`.** Partial. ../monorepo/llmd/weights.zig:537-660, ../monorepo/llmd/models/qwen3_5.zig:999-1024, ../monorepo/llmd/models/qwen4/model.zig:1288-1310. Split a Source, place and bind once; no new prepareSpec check. → M7.
- **[89] `bindQuantized` returns the Quantization.** Confirmed. ../monorepo/llmd/weights/linear.zig:601-641. Four repackings and `swizzled_scales` in model code go. → M7.
- **[90] linear.zig narrows.** Partial. ../monorepo/llmd/weights/linear.zig:20-89, :483-498, ../monorepo/llmd/weights.zig:492-535. `createBf16`; six helpers private; one `Component`; `View.name` and the swizzle generic go. `convertsBlockFp8` goes private once [83]'s assert replaces the guard. → M7.
- **[91] Routers, gates and heads through `linear.create`.** Confirmed. 11 sites in seven models. → M7.
- **[92] Two named refusals.** Partial. ../monorepo/llmd/weights.zig:16-27, :198-206, ../monorepo/llmd/weights/linear.zig:255-265. `mismatch` and `unsupported`; the guard half is superseded by [83]. → M7.
- **[93] One loader test oracle.** Partial. zml/io/testing.zig, ../monorepo/llmd/weights.zig:835-908. Fold into `zml.testing` with a rank-N `expectShards(..., memory)`; `fuseGroups` stays. → M10.
- **[95] One shared dense gated Mlp.** Partial. ../monorepo/llmd/models/llama.zig:1443 and eight more. Beside `llama.RmsNorm`, not in weights.zig. → M20.
- **[96] The store reports its host-mapped bytes.** Confirmed. zml/io/TensorStore.zig:36, ../monorepo/llmd/models/deepseek41/model.zig:1515, ../monorepo/llmd/models/qwen4/spec.zig:62. `View.hostMappedBytes()`, with `try`, not a fallback. → M8.
- **[97] `zml.Bufferized(Weights)` for the five `ModelBuffers`.** Confirmed. ../monorepo/llmd/models/llama.zig:508, ../monorepo/llmd/speculative.zig:246. Both starts stay. → M8.
- **[98] Resolve the drafter once.** Confirmed. ../monorepo/llmd/speculative.zig:31-65, ../monorepo/llmd/models.zig:108-118, ../monorepo/llmd/main.zig:354. `speculative.Drafter.open`, passed to `Model.init` in place of `CliArgs`. → M8.
- **[99] One required `load` per model.** Confirmed. ../monorepo/llmd/models.zig:240-263. Start, wait, bake; `@hasDecl` hooks and `buffers_loaded` go. → M8, with a fallback.
- **[100] `ModelParallelism` as a plain struct.** Confirmed. ../monorepo/llmd/parallelism.zig:5-171. → M20.
- **[101] Drop `BakedInputs`.** Confirmed. ../monorepo/llmd/models.zig:358-434. → M20.
- **[102] Mnist teaches the in-main recipe.** Confirmed. README.md:165-197, examples/mnist/mnist.zig:40-113. → M9.
- **[103] examples/io's load path in straight lines.** Confirmed. examples/io/main.zig:281-336, :621. → M9.
- **[104] examples/io checks against the entries it bound.** Partial. examples/io/main.zig:251-537. No line saving; the oracle no longer reads the code under test. → M9 (optional).

## Appendix B: refuted

**Refuted groups.**
- **[1] A Backend forwarder for unlanded bytes:** `probeMemory` already decides the capability once; a forwarder returning a bool mixes a capability query with a read and removes nothing a reader must know.
- **[65] Generating VFSBase's 106 forwarders at comptime:** lines go, concepts do not; a reflection switch everyone must understand replaces boilerplate nobody reads, and the hand-written literal also guards against a change to `std.Io.VTable` (decision 6 keeps the forwarders).
- **[94] Replacing testLayer's Loader with a hand-filled `readBuffer` walk:** it re-adds the visit-order pairing the fiftieth pass removed (CTX.md:4628-4642).

**Refuted halves of surviving groups.**
- [3]: moving `sameExecutable` into loader.zig relocates a rule and reverses a recorded move (CTX.md:4404).
- [5]: calling `dma_calibration.validate` from backendOptions splits one list's bounds across two files; duping `registerBackend`'s key adds ownership for a rule no caller meets.
- [7]: a per-platform `checkOptions` makes validity depend on the platform (CTX chose the platform-free check); removing the overflow guards would rest safety on the parser and turn a logged error into UB in ReleaseFast.
- [10]: merging Group and PendingSubmission labels the bulk as a program group and adds a field valid only after admission.
- [13]: the "overlap hole" bug: gap code never sees a mixed spec.
- [16]: `dim`/`dtype`/`rank` forwarders on `Placed` add surface for reads `.shape.` covers.
- [17]: a culprit out-parameter and new `NoBytes`/`EmptyTensor` errors move validation into Source; a pub `key()` serves one log line.
- [22]: borrowing specs before [10] (backend.zig:56-58 then required copies); swallowing `Group.await`'s cancel would split `Backend.await`'s contract.
- [26]: rewriting `expectHostTableGathers` adds nothing and loses accelerator coverage.
- [29]: one stored seal result tested three times, `try awaitBatch; unreachable;`, and moving progress onto Batch each make the code less local.
- [31]: a signed ledger counter guts the admission tests' absolute per-device checks, the only CPU coverage (LOADER_REVIEW Move 9).
- [33]: inlining `bindExpert` spreads the `Stack` union to nine call sites.
- [34]: a 12-line header and dropping the MXFP4 numbers would reverse R16 and move evidence for decision 5.
- [35]: calling `alignmentOf` in createDestinations makes a refusal depend on page-cache state; a mutable `VFS.direct_io` field adds a must-not-change rule against CTX.md:1627.
- [43]: routing `stopWorkers` through `abort` sets a spurious `Canceled` on every clean shutdown and orders destroy.
- [53]: "preserves ROCm byte balance": by CTX's own figures it mirrors the imbalance.
- [62]: deleting `Spec.format` and `Logical.format` removes standard `{f}` hooks, not knowledge.
- [67]: generating both stats types with `FieldEnum` keeps the test-only subtraction and adds machinery.
- [69]: moving `failure` into stdx would put the loader's error vocabulary in a shared library.
- [70]: generic `headSize`/`getBody` helpers cannot serve HTTP's and HF's redirect walks; sending `x-goog-user-project` on HEAD and GET is an untested behaviour change inherited from master.
- [79]: a shared redirect walker now generalizes HF's for a caller whose GET forbids redirects.
- [80]: "restores the allocation invariant": every governed attempt already allocates through `io.concurrent`; a path-keyed map is needed only under [78].
- [82]: deleting S3 and GCS listing is a feature removal (examples/io's ls and tree).
- [87]: `withLayout` inside the compile drops donations; a debug-only walk is unsafe in release.
- [88]: a prepareSpec check that a loaded tensor's placement equals its bound source's adds a rule and breaks the checkPlacement test.
- [91]: folding `createReplicatedTensor` into `createTensor` re-raises a keep without new evidence.
- [92]: folding every refusal into `mismatch` gives one concept two names, since zml's MoE backend already returns `UnsupportedQuantization`.
- [93]: folding `fuseGroups` into `fuse` breaks three tests' fixtures and leaves the grouping check untested.
- [96]: a TensorStore method plus a View forwarder is two declarations where one does; a `catch` fallback would over-count.
