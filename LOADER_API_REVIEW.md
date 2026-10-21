# Loader API review (2026-10-21)

This review looks at the loader's APIs. It starts from what llmd and the zml examples write, follows each surface down to the backends, and judges every surface against one brief: code earns its place, fewer abstractions, logic and relations local, simple enough to reason about locally.

It follows `LOADER_REVIEW.md`, the end-to-end review from earlier today, and the thirty-first pass that fixed that review's bugs and did its Move 2 (credits as fixed slots). It does not re-report that review's findings. Where it sharpens one of its moves, it says which.

**Method.**
- Twelve reviewers produced 123 findings:
  - eight by surface: Loader front door, TensorStore and View, Source, llmd's weights layer, the models bottom-up, the backend seam, the direct backend's internal seams, and the VFS and safetensors registry;
  - three cross-cutting: lifetimes and errors, docs and examples, and the end-to-end story.
- A merge pass:
  - folded 36 duplicates;
  - dropped 3 findings that only restated `LOADER_REVIEW.md`;
  - sent 4 reviewers to surfaces nobody had covered. Those reviewers covered `Bufferized` pairing, Source as a stored value, quantization at bind time and registry construction, and added 18 findings.
- Each of the 78 non-"keep" findings was attacked by verifiers told to refute it:
  - high-impact findings and bugs got two lenses, facts and value;
  - the others got one verifier covering both.
  - 77 survived, almost all as "partial" with a corrected claim or a smaller proposal; 1 was refuted (Appendix B).
- Three designers each built a target API from the surviving findings, from three angles: caller-first, seams-first and incremental. A judge scored them and merged them into the one design in Part 1.
- Nothing was built or run.
- I checked these against the code myself, and all hold:
  - the stale-copy bug in examples/llm;
  - the `Closed` leak;
  - the `awaitAll` hole;
  - the unswizzled DSpark projections;
  - the qwen4 free;
  - the unclosed repo directories;
  - that every load caller awaits on the next statement;
  - the four hand-assembled VFS setups;
  - the 8 `recipes.put` pairings;
  - the 2 `normalizeBlockFp8` opt-ins.

**Verdict.** The parts that move bytes earn their place, and so does the vocabulary model code uses to describe them:
- Source's Tensor-mirrored ops;
- View navigation and the `createTensor` one-liners;
- ShardGrid;
- the LoadSpec contract between front end and backends;
- Credits;
- the VFS as an `std.Io`.

What does not earn its place is how the API is cut. It follows the implementation's phases, not the caller's story:
- **Submit and await are two steps, but no caller does anything between them.** Every `load` is followed by `awaitAll` on the next statement. The split costs:
  - a borrow-until-retire contract (outputs, store, executables and progress node);
  - a public `cancel`;
  - an order-sensitive `errdefer loader.cancel()` in llmd;
  - an "outputs are undefined, free the ones that were assigned" rule a caller cannot follow.
- **A computed tensor is described in two repositories:**
  - its inputs live in zml's binding (`Source.bindProgram`);
  - which program computes it lives in llmd's `Packer.recipes`, keyed by the same `Tensor.Id` and paired by hand at 8 sites;
  - when it compiles and loads lives in llmd's `loadPacked`, which re-walks the model as `load` does, and must run before `load` (the `delivered` rule).
- **Binding and placing have too many spellings:**
  - seven ways to make a tensor from a key;
  - six ways to say where a Source lands, plus `Input` and `BindOptions`;
  - five ways to set a memory kind;
  - `.replicated` silently dropping the partitioning that 25 call sites pass.
- **The loader's setup is assembled by hand.** Callers compute a `LoadProfile` from the VFS and pass it to the loader, which already reaches the VFS through `io`. Four callers assemble the same remote VFS by hand. Two callers wrap a by-value `Loader` in a future, and one of them deinitializes the wrong copy.
- **llmd's weights layer is a `Packer` threaded through about 70 init functions.** None of its five fields needs a model-lifetime object. Its two platform post-passes (`normalizeBlockFp8`, `maybeSwizzleBlockFp8Scales`) are opt-ins that 2 of 11 models remember, and the swizzle opt-in already misses DSpark's layers.

The recommended design (Part 1) gives callers three handles, each with one job:
- **View** is what model code holds: navigate, look up, bind.
- **TensorStore** is owned by main and owns the registry.
- **Loader** is opened early and closed after the load. Its one `load` call returns the finished `Bufferized` tree and borrows nothing after it returns.

A program becomes part of its tensor's binding, so llmd states only the recipe. The order of moves in Part 2 lets each land alone: bugs first, then the independent cleanups, then the API change.

## Read this first

**Bugs found through the APIs.** None writes wrong bytes on a shipped path today. The first four are worth fixing now; each is a few lines.

1. **examples/llm deinitializes a stale copy of the Loader.**
   - What happens: `main.zig:86-92` defers `loader_fut.cancel(io)` → `deinit`. After `await`, `Future.cancel` returns the future's stored result, which is the copy taken at init. The live loader awaited at `:145` is never deinitialized.
   - Consequences:
     - its `pending` deque leaks;
     - its admission log is lost;
     - the pinned arenas and worker threads live for the whole chat session.
   - Had anything been pending, the stale copy would have destroyed the backend under in-flight batches.
   - The `errdefer if (fut.cancel(io)) |v| v.deinit()` idiom in `monorepo/AGENTS.md` produces exactly this for any by-value handle that changes after init. llmd avoids it only with its `loader_awaited` flag, a pattern AGENTS.md itself discourages. Fix it at the caller now (R1); R5 removes the future altogether.
2. **Direct `submit` returns `error.Closed` instead of the cause** (LOADER_REVIEW #29, still open).
   - If an older batch fails between `checkOpen` and the first publish, `Scheduler.publish` returns `Closed`. The pre-publication branch (`direct_loader.zig:245-248`) returns it unchanged, and `load` or `loadExecute` hand it to the caller.
   - The fix is `return self.pipeline.errorValue() orelse err`.
3. **`awaitAll` can report success after a terminal failure.**
   - The front end's `failure` is set only in `retireOldest`. A `backend.submit` that fails after a partial publish fails the backend, but not the front end (`loader.zig:443`).
   - With nothing else pending, a later `awaitAll` returns success.
   - It is latent: no caller awaits after a failed `load`. The fix is one sticky record, kept in the backend.
4. **DeepSeek's DSpark layers are not swizzled on sm_10x.**
   - The swizzle is an opt-in list in `Layer.init` (`deepseek41.zig:1001-1003`). `DSparkAttention.init` (`:3779-3786`), `DSparkLayer.init` (`:3901-3923`) and the drafter's `main_proj` (`:4014`) build the same block-FP8 projections and never call it.
   - The verifier checked the checkpoint: the `mtp.*` projections are F8_E4M3 with F8_E8M0 block-32 scales, like the main layers.
   - So on GB300 every draft step's GEMMs take the unswizzled `scaledDot`, which permutes the scales on every call. This is a performance bug, not measured.
   - Stop-gap: add the call in `DSparkLayer.init`, including `main_proj`. R9 removes the list.
5. **qwen4 frees `ngram_host_buffers` with the wrong allocator, after that allocator is gone.**
   - It is allocated with `allocator` (`qwen4/spec.zig:432`) and freed with `self.arena.allocator()` after `self.arena.deinit()` (`:485-487`).
6. **`resolveModelRepo`'s directories are never closed** (`llmd/main.zig:345`, `speculative.zig:31`, `zoo.zig:469`). It is one handle per process.
7. **Latent pairing hazards in `load`.**
   - `prepareModelLoad` pairs tensors with output buffers through two independent visits and checks nothing (`loader.zig:556-643`). A tree whose optional, slice or union structure differs from the model writes out of bounds, or lands a tensor in the wrong slot.
   - `Bufferized(BoundedArray(Tensor, N))` has N buffers where `visit` sees `len` tensors (`meta.zig:345-375, 508-510`). No model stores one today.
   - R1 makes it a compile error; R4 makes pairing hold by construction.
8. **Still open from LOADER_REVIEW, and missing from the examples CTX's thirty-first pass lists as open:**
   - #35: `Loader.init` can leak running tasks (`host_memory.zig:267`);
   - #36: `deinit` can panic after a device error (`platform.zig:468`);
   - #30: binding rejections log only at debug, so llmd exits with a bare `TensorNotFound`.

   CTX's list already names #8 (borrowed URIs) and #6 (remote error mapping); R1 and R6 cover them.
9. **On CDNA3, a model that does not call `normalizeBlockFp8` binds FN weights against FNUZ activations.** Only qwen3_5 and deepseek41 call it. An example is llama serving Qwen3-FP8 on MI300.
   - That FN × FNUZ `scaled_dot` pairing has no test in the tree (`nn.zig:228` skips ROCm), and the ROCm plugin decides what it does.
   - qwen3_5 and deepseek41 also reach it whenever #507's scale guard keeps a projection FN.
   - The result is not confirmed wrong, but the path is untested. R9 makes the choice once, at bind time.
10. **examples/io's `ZML_DMA_BENCH_*` values reach `calibrate`'s debug asserts** (`examples/io/main.zig:705-735`).
    - A value out of range panics in safe builds and is UB in ReleaseFast.
    - CTX:3829-3838 chose asserts and put the range check in the caller, which was never written. R5 turns it into `InvalidOptions` on the bench's own entry point.

**What a caller holds today.** To load llmd's model, main and the models touch about 25 concepts:
- **VFS setup:** an http client and four backends, registered by hand.
- **Registry and store:** a `TensorRegistry` that must outlive the `TensorStore`, and meshes passed to the store, then again to `loadInto`.
- **Loader setup and protocol:**
  - a `LoadProfile` computed from the VFS;
  - a loader future plus a flag;
  - the `load`/`loadExecute` pair and its order rule;
  - `Loader.Binding` and how to group bindings;
  - `awaitAll`, `cancel` and `deinit`;
  - who owns the progress total;
  - `bytesLoaded`, for a log line ten callers rebuild.
- **Binding vocabulary:**
  - seven tensor constructors and six placement calls;
  - `BindOptions` versus `setMemory`;
  - the `Packer`, its recipes and its two post-passes;
  - `FusedLinear` versus `Interleaving` versus `concatenateLinears`.
- **Model protocol and cleanup:** the `loadBuffers` and `finalizeLoadedBuffers` duck-typed protocol, and about 100 hand-written `unloadBuffers` walkers.
- **Drafters:** a second store and a second load phase for each drafter.

After the recommended moves, main holds a VFS with `Remotes`, a store that owns its registry, and a loader. A model holds a View, and loads in one line.

## Part 1: the recommended API

The judge started from the caller-first design and grafted ideas from the other two:
- from the seams-first design: a two-arm `Binding` union, one bits-to-bytes rule, and one `InvalidBinding` error;
- from the incremental design: the move order, the "place last" guard, the option name `checkpoint` and the registry lock.

Calibration stays. Move 2's Credits stay.

```zig
// ===== zml.io.Loader =====
pub const Options = struct {
    /// The checkpoint's path or URI, read by `init` only: the VFS backend serving it sizes the
    /// reads. Null or a plain io: local sizing. Replaces `load_profile` and `vfs.loadProfile`.
    checkpoint: ?[]const u8 = null,
    read_parallelism: ?usize = null,                   // 1..128; null: 16 local, 32 remote
    dma_block_sizes: []const usize = &dma_calibration.default_block_sizes, // replaces Options.dma's 8 knobs
    direct_io: VFS.DirectIo = .auto,
};
/// Every error `load` returns; each fails the loader. Model bugs are one name, logged at err
/// with the tensor, its key and the rule it breaks.
pub const Error = error{ InvalidBinding, Canceled } || std.mem.Allocator.Error || std.Io.File.OpenError
    || safetensors.ExactPositionalError || Compiler.Error || pjrt.ApiError;
/// Validates `opts` and returns at once; warm-up, calibration and pinning run on the loader's
/// own task, and the first `load` waits for what is left.
pub fn init(allocator, io, platform: *const Platform, opts: Options) error{ InvalidOptions, OutOfMemory }!Loader;
pub fn deinit(self: *Loader) void;
/// Loads every tensor of `model` as `store` binds it and returns their buffers. Program-bound
/// tensors compile on their own tasks; their inputs land first, in groups the measured device
/// room admits, and run on this task; the rest loads behind them. One summary line is logged.
/// Borrows nothing after it returns; on error nothing is left running or allocated.
pub fn load(self: *Loader, allocator, model: anytype, store: *const TensorStore, progress: std.Progress.Node)
    Error!mem.Bufferized(@TypeOf(model.*));
// gone: loadExecute, Binding, awaitAll, cancel, bytesLoaded, LoadError/LoadExecuteError/AwaitError

// ===== zml.io.TensorStore: owned by main =====
pub fn init(allocator, registry: safetensors.TensorRegistry, opts: struct { meshes, quantization: Quantization.Target }) TensorStore;
pub fn deinit(self: *TensorStore) void;          // bindings, then the registry it owns
pub fn view(self: *TensorStore) View;
pub const Memory = enum { default, host_pinned, host_mapped };
pub const Binding = union(enum) {                // the loader's only read: bindingOf(tensor)
    load: struct { pieces: []const Source.Piece, memory: Memory = .default },
    compute: struct { program: Source.Program, inputs: []const Source.Placed },
};

// ===== View: the only handle model code holds =====
root, withPrefix, withLayer, prefix            // navigation, unchanged
get(subkey) ?*const safetensors.Tensor          // exact, one hash; replaces getShape, getReader, hasKey-as-exact
hasPrefix(subkey) bool                          // the one prefix scan, named as one
source / maybeSource(subkey, tags) Source
createTensor / maybeCreateTensor / createReplicatedTensor   // one-line shorthands over Source
mesh(name), quantization(), setMemory(tensor, Memory)       // the one memory setter

// ===== Source: placement is a property set last =====
pub fn withPartitioning(self, comptime mesh, partitioning) Source; // .replicated takes only .{}; ops on a placed source panic
pub fn as(self, placed: Shape) Source;
pub fn bind(self) Tensor;
/// `func(args, output, inputs)` computes the tensor during the load. `args` must be plain data
/// (comptime check); equal func, args and placed shapes share one executable.
pub fn bindProgram(inputs: []const Source, output: Shape, comptime func: anytype, args: anytype) Tensor;
// gone: bindAs, rebind, place, placeAs, Input, BindOptions, ofBinding, sourceOf, unbind

// ===== vfs =====
pub const Remotes = struct { init(self, allocator, vfs, environ), deinit }; // https, hf, s3, gs, proxies
// one error contract: FileNotFound (404/410), PermissionDenied (401/403), Canceled, else logged + Unexpected

// ===== llmd weights: free functions over View, no Packer =====
linear.view / maybeView / create / maybeCreate / fuse     // fuse replaces interleaveLinears + concatenateLinears
weights.experts(allocator, view, ExpertSpec) Experts       // exact lookups
weights.moeBackend(platform, view, projection)             // replaces expertsFormat + Backend.auto
weights.bf16Linear(view, out, k, partitioning)
// block-FP8 for the platform decided once, in linear.bindBlockFp8, from view.quantization()
```

What callers then write:

```zig
// llmd/main.zig
var remotes: zml.io.VFS.Remotes = undefined;
try remotes.init(allocator, &vfs, init.environ_map);
defer remotes.deinit();
var store: zml.io.TensorStore = .init(allocator, registry, .{ .meshes = &meshes, .quantization = .of(platform) });
defer store.deinit();                                  // the drafter joins via registry.addRepo(io, drafter, "drafter.")
{
    var loader: zml.io.Loader = try .init(allocator, io, platform, .{ .checkpoint = args.model });
    defer loader.deinit();                             // pinned memory goes right after the load
    try model.init(...);                               // calibration runs meanwhile
    try model.loadBuffers(&loader, &store, &progress);
}

// a model
self.buffers = try loader.load(self.allocator, &self.spec.weights, store, progress); // target + drafter
// ...
if (self.buffers) |*b| zml.mem.deinitBufferized(self.allocator, Weights, b);

// mnist, the README, a new "Loading weights" tutorial step
var buffers = try loader.load(allocator, &model, &store, .none);
defer zml.mem.deinitBufferized(allocator, Mnist, &buffers);
```

## Part 2: the moves

Each move lands as its own commit that compiles and passes tests. "Depends" lists what must land first. R1, R3, R5 and R6 depend on nothing and can go in any order.

| # | Move | What goes | Depends | Risk |
|---|---|---|---|---|
| R1 | Fix the seam bugs | the bugs above, a few lines each | none | very low |
| R2 | Programs ride the binding; `load` compiles and runs them | `loadExecute`, `Loader.Binding`, `delivered`, ~110 lines of executable checks; in llmd `Recipe`, `Packer.recipes`, `Program`, `loadPacked`, `compileRecipe` (~220-400 lines) | none | medium (public API; compile timing) |
| R3 | Free loaded weights with `deinitBufferized` | about 100 hand-written `unloadBuffers` walkers | none | very low |
| R4 | `load` returns the buffers once ready | `awaitAll`, `cancel`, `bytesLoaded`, the borrow-until-retire contract, ~20 public error names, 10 copies of the "Loaded weights" block, 4 dead example loaders | R2 | medium (43 call sites change once) |
| R5 | `Loader.init` returns at once and takes only what a caller sets | the loader futures and flag at both callers, `LoadProfile` at 4 callers, `Options.dma`'s 8 knobs, `pub const limits`, the reach into `loader.backend.direct.calibration` | none | low-medium |
| R6 | VFS: one `Remotes` bundle, one error contract | 4 hand assemblies of 20-24 lines, 4 divergent "absent/denied" mappings (#6), `unregister` | none | low-medium |
| R7 | The store owns a complete registry; an entry is a tensor's identity | entry copies, `getShape`/`getReader` x2, `TensorReader`, ~290 lines of unread metadata parsing, ~30 `defer registry.deinit()` | R1 | low-medium |
| R8 | The drafter joins the model's store and its one load | the second store, the second tree, 3 drafter `loadBuffers`, the second load phase | R4, R7 | low |
| R9 | Block-FP8 for the platform decided once, at bind time | `normalizeBlockFp8`'s tree walk, `maybeSwizzleBlockFp8Scales` and its opt-in lists, `canDoubleScale` and its 92,160 serial opens | R2, R7 | medium (CDNA3 behavior, scale guard) |
| R10 | Source carries its placement; View is the model's one handle | `bindAs`, `place`, `placeAs`, `Input`, `BindOptions`, the host-pinned create pair, `sourceOf`/`unbind`/`rebind`, 26 `store.store` reach-throughs | R2, R7, R9 | medium (~75 call sites) |
| R11 | llmd's weights layer is free functions; every model loads in one line | the `Packer` (266 mentions, ~70 init parameters), `Interleaving`, ~45 lines of MoE setup per 5 models, ministral3's own FP8 Linear | R2, R4, R9, R10 | medium (churn) |
| R12 | A host table the GPU reads in place is a delivery (`host_mapped`) | deepseek41's and qwen4's `createView` swaps and owner lists | R4, R10 | medium (unverified on CUDA) |
| R13 | One binding representation, one validation pass | `Layout`, `Part`, `zero_fill`, the overlap prover, the duplicated backend checks, the front-end admission ledger | R2, R7, R10 | low-medium |
| R14 | Direct backend: one root object, each value stored once | `Pipeline.init` and its 9 forwarded fields, Job's four flags, the Workspace's five-call protocol | R5 | low |
| R15 | The loader meets the VFS only through `io`; remote backends share one shape | read-stats plumbing (4 types for one debug line), 4 copies of the retry knobs | R5, R6 | medium (handle table is high, staged) |
| R16 | Docs: the pub doc comment is the contract | ~150 lines of internals in loader.md; contracts written three times | R4, R5, R10 | none |

These moves sharpen or complete Moves 1, 3, 4, 6, 7, 9 and 10 of LOADER_REVIEW: R12 builds on Move 1, R13 on Moves 3, 7 and 9, R14 on Move 4, R11 completes Move 6, and R6 and R15 cover Move 10.

**R1: fix the seam bugs.** These are items 1-7 of "Read this first", each a few lines with no API change:
- (a) return the cause, not `Closed`;
- (b) the direct backend dupes the file URIs it keeps (#8);
- (c) examples/llm scopes its loader future and deinitializes the awaited loader;
- (d) the qwen4 allocator;
- (e) close the repo directories;
- (f) `MapRestrict` makes a BoundedArray of Tensors a compile error;
- (g) the safetensors header parse rejects offsets that overflow or run backward (#20).
Add the DSpark swizzle stop-gap too, and give (a) and (b) a test each.

**R2: programs ride the binding.** `Source.bindProgram(inputs, output, comptime func, args)` records a type-erased program next to its inputs. `args` is plain data, checked at comptime, so `std.meta.eql` dedupe is exact. That answers Move 6's objection to Shape args; the three recipes that carry a Shape are dead or die with qwen4's n-grams becoming a `Source.concatenate`.

Inside `load`:
1. collect the program tensors;
2. dedupe them;
3. start each compile with `io.async`;
4. publish groups of at most `program_group_bytes` as their executables arrive, admitted as today;
5. load the bulk behind them.
That keeps today's overlap of pack reads with later compiles. One check per executable remains: its output must be placed like its tensor.

In llmd, each `bindProgram` + `recipes.put` pair becomes one call, the Recipe variants become arg structs, and `loadInto` shrinks to `load`. The 8 GiB group size becomes a Loader field that tests can lower, with its rationale (read coalescing, CTX:3943-3959) next to it.

**R3: `deinitBufferized` everywhere.** Mechanical. It also removes a silent device-memory leak: a struct that gains a field its walker forgets.

**R4: `load` returns when done.**
- `load(allocator, model, store, progress)` bufferizes the very value it walks, so the pairing holds by construction.
- It runs R2's sequence, retires everything, logs one line, and returns the tree.
- On error it frees the tree and fails the loader. One sticky record lives in the backend, so the `Closed` leak and the `awaitAll` hole cannot recur.
- Errors collapse to one `Loader.Error`. Model-bug names log the tensor and its key and return `InvalidBinding`, which fixes #30 by construction. Names only a broken invariant can raise become asserts.
- Progress is passed by value, and the loader adds its own items.
- What this gives up is cross-call overlap. Its only user is speculative's resharded lm_head, which R8 puts in the same call.

**R5: init returns at once.**
- `init` validates the four options, resolves the read hints through `VFS.fromIo(io)` and the `checkpoint` path, and starts warm-up, calibration and pre-growth on its own task. It falls back to a synchronous call when the io has no concurrency.
- `load` awaits that task, and its failure becomes the sticky error. `deinit` cancels it, or destroys what it built.
- The Loader stays a value, as CTX:2289-2291 decided: the future holds the backend pointer, and nothing points back into the Loader.
- dma-bench gets `dma_calibration.measure(allocator, io, platform, opts)`. It needs no Loader and keeps the bench's knobs, now validated with `InvalidOptions`.
- `limits.zig` dissolves into Options' bounds and Sizing. execute_admission's two functions move into loader.zig.

**R6: `VFS.Remotes`.** One `init(allocator, vfs, environ)` sets up the http client, https, hf, s3 and gs, with proxies. That replaces four hand assemblies; examples/llm's copy has drifted, with no https and no proxies.

One error contract, mapped by one `request.openStatus`:
- open, stat and access fail with FileNotFound (404/410), PermissionDenied (401/403) or Canceled;
- anything else is logged at err and surfaces as Unexpected.

That fixes #6, the missing optional file that aborts llmd on s3:// and https://.

**R7: the store owns the registry.**
- `TensorStore.init(allocator, registry, opts)` takes the registry and locks its map, so safe builds panic on a later insert.
- `View.source` hands out the registry's own entry, so an entry's address is the tensor's identity. That removes `dupeSource`, Lowering's second copy and llmd's `sameTensor`.
- Lookups:
  - `View.get` replaces `getShape` at both levels, and the 7 exact uses of `hasKey`, which today scans every key;
  - `hasPrefix` keeps the one real prefix question;
  - `safetensors.Tensor.read` and `readRange` replace `TensorReader`.
- Dead code goes: the ~290-line metadata path, which nothing in either repository reads, plus `fetchRegistry`, `resolveFiletype` and `resolveModelEntrypoint`.

**R8: the drafter joins the model's store.**
- main calls `registry.addRepo(io, drafter_repo, "drafter.")` before `TensorStore.init`.
- The drafter's `Spec.init` gets `store.root().withPrefix("drafter")`.
- The 5 speculative models load `Weights{ base, speculative }` in one call, so the drafter's reads plan and coalesce with the target's, instead of waiting for the target to retire.

**R9: block-FP8 at bind time.**
- `zml.Quantization.Target.of(platform)` is the only platform switch, and the store carries it.
- One llmd helper, `linear.bindBlockFp8`, is called by `create`, `fuse` and `experts`:
  - FN weights on an FNUZ target bind the normalize program;
  - e8m0 scales on a swizzling target bind `swizzleBlockFp8Scale` as a Source.
- No opt-in list is left to forget, and DSpark is covered.
- It drops the scale-overflow guard (`canDoubleScale`), which is the user's call (Part 3).

**R10: Source carries its placement.**
- `withPartitioning` and `as` set placement, and `bind`/`bindProgram` are the only terminals. Any op on a placed source panics, so "place last" is checked, not a convention.
- `.replicated` accepts only `.{}`, so the 25 dead partitioning arguments become compile errors.
- `View.setMemory` is the one memory setter. qwen4 decides memory after binding (`qwen4/spec.zig:167`), so the after-bind setter is the one that must exist.
- `sourceOf`, `unbind`, `rebind` and `ofBinding` go: after R9 none has a production caller.
- `bindNew(shape, binding)` replaces `putSourcesNoClobber`.

**R11: no Packer.** After R2 and R9 the Packer holds no state.
- Fusion moves into `weights/linear.zig`. One `fuse` replaces `interleaveLinears(.per_shard)` and `concatenateLinears`: all three concatenate callers are fully replicated, where the two are the same.
- `weights.experts` and `moeBackend` replace the expert setup that 5 models repeat.
- ministral3's o_proj and down_proj use `linear.create`.
- Errors become one `weights.Error`, raised through one `mismatch()` that logs the key, in place of about 48 bare returns under 4 interchangeable names.
- ExpertLayout stays the user's open decision (Move 6).

**R12: `host_mapped`.** It lands like `host_pinned`, through the existing SharedHostAllocation path. Then the loader writes a device-kind view, whose `on_delete` releases the allocation, into the output. That removes deepseek41's `swapEngramHostPinnedBuffers` and qwen4's `createView` swap, with their owner side lists. It rests on an unverified CUDA precondition (Part 3). If that fails, keep the swaps; nothing else depends on this move.

**R13: one binding representation, one validation pass.**
- The binding holds what bind was given, its pieces; no `Layout`/`Part` re-encoding.
- `Source.lowerPiece` is the one bits-to-bytes rule.
- The front end validates every spec once, before admission:
  - empty sources and plain sizes;
  - strided parts into host memory;
  - placement for every memory kind;
  - mesh device ids below the device count.
- Backends trust the contract, so `zero_fill` and the overlap prover go (Move 3). The admission ledger moves into the direct backend (Move 9).

**R14: one root in the direct backend.** This is Move 4, sharpened:
- Pipeline and Sizing fold into Loader, and each sizing value is stored once.
- Job's kind becomes a tagged union.
- The Workspace becomes `{init, deinit, allocate, capacity, carve}`.
No behavior or performance change.

**R15: the VFS only through `io`.**
- Remote backends log their read totals at deinit, so `ReadStatsProvider` and its three relatives go.
- Direct I/O asks `vfs.isLocal(path)` before opening. A local drafter beside an hf:// model then reads directly, and no hf:// file is opened on the planner.
- One `request.Retry` block replaces four copies of the retry knobs.

**R16: docs.**
- The contract lives in the pub doc comments.
- loader.md becomes a ~120-line guide: recipe, placement, composites, programs, lifetime and failure, and the one error.
- Internals move into the `//!` headers of the files that own them.
- `write_first_model` gains a "Loading weights" step, and its broken `fromSlice` calls are fixed.

## Part 3: decisions for you

The verifiers pushed back on five of the judge's choices, and five moves reverse recorded decisions. These are the calls that matter before any of R2-R12 starts.

1. **R2: zml compiles pack programs.**
   - The value verifier's case against it:
     - it is roughly line-neutral in zml;
     - it adds a type-erased program and a comptime args rule to the binding;
     - it blocks starting pack compiles during model init, CTX's recorded next step (0.85-2.4 s measured), because no Loader exists then.
   - The judge's case for it: it removes the cross-repository pairing, the `delivered` rule, the executable-versus-binding validation and the public `loadExecute`. Compile/read overlap is kept inside `load`, and compiling at bind time becomes a one-place change later.
   - **My take: take R2.** The pairing at 8 sites, and the order rule that fails at debug level, are the most non-local logic in the API. To keep the bind-time option open, have `bindProgram` record the program so a later change can start its compile from the store, and measure the wait first.
2. **R4: a blocking `load`.**
   - It loses cross-call overlap, which only the lm_head uses, for tens of ms; R8 then gains the drafter overlap instead.
   - It reverses:
     - the fourteenth pass's load/awaitAll split (CTX:3082);
     - caller-owned progress totals (CTX:62-69);
     - LOADER_REVIEW's "validation that leaves the loader usable". No caller reuses a loader after an error.
3. **R5: init owns its task.**
   - The value verifier would fix only the two concurrent callers. It argues that a hidden await sits in front of every backend access, and that init errors move to the first load.
   - The judge keeps the by-value Loader (CTX:2289) and removes the future, the flag and the cancel-or-deinit dance at every caller.
   - **My take: do R1(c) now, and R5 with R4.** Then every caller gets the overlap for free, and no caller can deinitialize the wrong copy.
4. **R9: drop the block-FP8 scale-overflow guard.**
   - A block scale of 2^127 or more cannot come from a trained model: every nonzero weight in it would decode past bf16's range.
   - Dropping the guard removes:
     - 92,160 serial opens at Qwen3.5-MoE init (1.4 s locally, 1-5 h remote, LOADER_REVIEW Performance #2);
     - the only checkpoint read in model init;
     - the only data-dependent input to the CDNA3 decision.
   - The cost: a pathological block becomes inf or NaN instead of staying FN.
   - The models that skip the pass today then get FNUZ weights matching their activations. That is a fix, but spot-check accuracy on MI300 after.
5. **R12: `host_mapped`.** A device view of loader-owned pinned memory, whose `on_delete` outlives its host owner, is unverified on CUDA. It needs an ownership test on a GPU before the swaps go.
6. **R10 reverses CTX:4973-4974,** "memory is a bind option; createHostPinnedTensor and setMemory stay as shorthands". The after-bind setter is the one that must exist.
7. **R13 reverses CTX:5024-5026** through Move 3: a hand-built overlapping binding is no longer caught. Only Source builds bindings, and its harness gains a double-write check.
8. **R5 reverses CTX:3829-3838's asserts on options and commit 8eb14029's reach into `loader.backend`.** `measure` replaces the reach without a Loader accessor, which keeps 8eb14029's intent.
9. **ExpertLayout (`.loader` vs `.device`) stays open,** as Move 6 left it.

## Part 4: what earns its place

Do not touch:
- **The byte core.** Source's algebra (pieces as affine bit lattices; ops that mirror Tensor's names, tags and panics), ShardGrid, the per-file planner and publication, FIFO claims, per-device pumps, the batch counter and the retired event stack.
- **The memory and failure machinery:**
  - Move 2's Credits, and the read gate as the exact `read_parallelism`;
  - the sticky first error and the drain;
  - direct submit's all-or-nothing publication;
  - SharedHostAllocation and LazyOnce.
- **The front-end policies:**
  - admission with FIFO retirement and no knob;
  - O_DIRECT decided before reads, with demotion on refusal;
  - the 64-run / 1 MiB expansion thresholds.
- **DMA calibration** (only its API narrows), and the two backends while TPU, neuron and metal are unverified on the direct path.
- **The seams and contracts:**
  - the LoadSpec contract;
  - `Lattice.copyRuns` as the one strided-copy kernel;
  - the four-field `safetensors.Tensor` entry;
  - `readFilePositionalAllV`;
  - the VFS as an io, reached through `VFS.fromIo`;
  - `Bufferized` as the type of loaded weights;
  - `Source.store`;
  - infallible binding, where a missing key panics with the key and data errors belong to `load`;
  - the store passed per load.
- **llmd's model decisions:**
  - `Scheme.classify` stays platform-free;
  - `moe.Backend.expertsLayout`;
  - swizzles as Source expressions;
  - `linear.View.split`.

**Rejected along the way:**
- `FusedLinear.forward`: a second way to split outputs.
- `take()`: nothing to rework after R9.
- `mem.slots`: pairing by construction is enough.
- A `View.registry` per view.
- A one-shot `zml.io.load(platform, model, store)`: a second entry point; calibration must start before the model exists.
- Freeing pinned memory at the end of `load`: a second lifetime rule.
- New quantization schemes for FNUZ or swizzled scales.
- Per-file read sizing.
- Moving the Backend union into loader.zig.
- Moving Lattice into TensorStore.zig: an import cycle.

## Appendix A: every surviving finding

Sorted by impact, then id. "Move" names the recommended move that absorbs the finding; "—" means none does (see "Rejected" in Part 3). Every finding survived verification, almost all as "partial": where the verifier note and the title differ, the note is the corrected reading. The full text of each finding and verdict is in the workflow output, not reproduced here.

**A-load-returns-when-done** (high, remove; move R4). Submit and await are split, but every caller awaits right after its last submission; the split costs a borrow-until-retire contract, cancel, and order-sensitive errdefers

- Where: `zml/zml/io/loader.zig:47-55`, `zml/zml/io/loader.zig:164-191` (+10).
- Problem: No caller does any work between submitting and awaiting: - Every `load` is followed by `awaitAll` on the next line: the 8 examples/llm model files, mnist:54, testing.zig:455, examples/io:371, README and the howto. - llmd's loadInto ends in awaitAll (weights.zig:1780).
- Verifier (value, partial): The split buys two overlaps, not one. The first is lm_head (speculative.zig:266/869, tens of ms). The second is the one the finding missed: llmd's loadPacked submits each pack group as soon as that group's programs compile, so earlier groups' reads hide later compiles (weights.zig:1601-1638;

**B-one-way-to-bind** (high, merge; move R10). Seven entry points make a model tensor from a key; source(...).bind(...) plus createTensor/maybeCreateTensor as one-line shorthands covers every caller

- Where: `zml/zml/io/TensorStore.zig:208-309`, `zml/zml/io/Source.zig:498-520` (+9).
- Problem: View has seven ways to make a tensor, and Source.bind/bindAs/bindProgram sit beside them. - The create family is a second copy of placement and binding: maybeCreateTensorInternal plus placeShape, about 25 lines.
- Verifier (value, partial): On top of Move 7 (createTensor and maybeCreateTensor become source().bind(); the host-pinned pair, maybeCreateTensorInternal and placeShape go), only two of this finding's additions earn their place. First, delete View.setMemory: both callers hold the TensorStore.

**C-placed-source** (high, merge; move R10). Six entry points say where a Source lands (bind, bindAs, rebind, place, placeAs, bindProgram), plus the Input and BindOptions types; callers write the target shape twice

- Where: `zml/zml/io/Source.zig:56-59`, `zml/zml/io/Source.zig:199-205` (+20).
- Problem: A Source is never placed: every op resets `_sharding` to replicated (`reshaped` at Source.zig:199-205 and 7 other sites). Placement is given at the terminal call instead. - There are two ways to give it: a mesh name plus a partitioning, or a placed Shape "built elsewhere".
- Verifier (value, partial): `bindAs`/`placeAs` ignore the source's tags yet demand its exact dims. So llmd restates the target shape or relabels for nothing at 8 sites: - reshape(shape) then bindAs/placeAs: 586, 760, 791; - `reshape(.{})` then bindAs: 462; - dead `withTags`: 677, 786, 826 (`withTags(.{.expert})`) and 869.

**D-program-in-binding** (high, reshape; move R2). A computed tensor's program lives in a llmd side table, paired with its zml binding by hand, then replayed through loadExecute before load

- Where: `monorepo/llmd/weights.zig:88-136`, `monorepo/llmd/weights.zig:140` (+33).
- Problem: How one computed tensor is made is split over three places. - zml's binding holds the inputs (TensorStore.Binding.inputs). - llmd's Packer.recipes, keyed by the same Tensor.Id, holds which program. - loadPacked decides when it is compiled and submitted.
- Verifier (value, partial): The problem is real only on the llmd side. - The 8 `bindProgram` + `recipes.put` pairs are kept by hand (already Move 6's `Packer.bindProgram`). - `loadInto` must await because llmd frees its programs on return, so model parts load one after another. - `loadPacked` re-walks the model (about 15 lines).

**E-drafter-one-store** (high, reshape; move R8). A drafter checkpoint gets its own TensorStore, so it also needs its own Packer, its own loadInto, and a second load phase that starts after the target's awaitAll

- Where: `monorepo/llmd/speculative.zig:24-52`, `monorepo/llmd/speculative.zig:256-271` (+18).
- Problem: Loader.load(T, model, buffers, store) resolves every tensor of `model` in one store, and a store wraps one registry (TensorStore.fromRegistry). A drafter is a second checkpoint, so SpeculativeContext builds a second registry and a second store (speculative.zig:39-42).
- Verifier (value, partial): The drafter is loaded through its own TensorStore, so it takes a second Loader.load call. Its load starts after the target's, because weights.loadInto awaits in order to protect the pack programs it owns; the second store does not cause that wait.

**G-one-root-one-copy** (high, merge; move R14). The direct backend has two root objects and several copies of each sizing value: Pipeline forwards 9 of its 12 fields, Sizing forwards all 7 of its fields, and a request carries three handles to the same state

- Where: `zml/zml/io/direct_loader.zig:71-123`, `zml/zml/io/direct_loader.zig:143-201` (+9).
- Problem: Pipeline.init takes 10 parameters. Of Pipeline's 12 fields, only first_error, pumps and dma_budget_bytes are its own state. The other 9 restate the Loader that owns it: allocator, io and platform are copies; read_gate, credits, metrics and scheduler are back-pointers into the Loader;
- Verifier (value, partial): Most of this finding is Move 4 as written: folding Pipeline into Loader, a ReadRequest that holds loader and index with run(self, scratch), one block_size, and claim returning {plan, index}. It adds three things that hold. (a) Loader.width and Loader.maximum_blocks_per_job are duplicates.

**H-profile-from-io** (high, reshape; move R5). Loader.init takes a LoadProfile the caller computes: a second channel to the VFS the loader already reaches through io

- Where: `zml/vfs/vfs.zig:70-92`, `zml/vfs/vfs.zig:186-209` (+16).
- Problem: To size a load, the caller must do four things. It holds a *VFS, calls vfs.loadProfile(path) before Loader.init, passes the result in Options.load_profile, and keeps the VFS and the backend alive, because `name` and `stats` borrow backend state (vfs.zig:190-191).
- Verifier (value, partial): The claim holds: the caller-built LoadProfile is a second channel to the VFS that the loader already reaches through `io`. Replacing `Options.load_profile` with a path the loader resolves through `VFS.fromIo(io)` simplifies the 3 real callers and the HTTP test, and it extends the CTX.md:2588 decision rather than reversing it.

**H-remotes-bundle** (high, missing; move R6). Four callers hand-assemble the same VFS (an http client, four backends, five copies of base_io, four scheme strings), and one copy has already drifted

- Where: `monorepo/llmd/main.zig:319-342`, `monorepo/llmd/zoo.zig:131-154` (+2).
- Problem: Each caller writes 20-24 lines: - a std.http.Client built on base_io, plus initDefaultProxies; - HTTP.init(allocator, base_io, &client, .https), then HF, S3 and GCS each through .auto(allocator, base_io, &client, environ); - four registerBackend("<scheme>", x.backend()) calls;
- Verifier (value, confirmed): I tried to refute the finding and could not. The facts it rests on check out. Callers and counts: - llmd/main.zig:319-342 and llmd/zoo.zig:131-154 are byte-for-byte the same 24-line block: client, initDefaultProxies, HTTP.init(.https), HF/S3/GCS.auto, four registerBackend calls.

**I-store-owns-complete-registry** (high, reshape; move R7). The store copies every registry entry it binds, and Lowering copies it again, only because the registry might still grow; one data set has three lifetimes (registry, store, loader)

- Where: `zml/zml/io/TensorStore.zig:78`, `zml/zml/io/TensorStore.zig:84-93` (+14).
- Problem: TensorStore borrows a `*TensorRegistry` (TensorStore.zig:78). - Every View.source, createTensor* and maybeCreateBinding call copies the registry entry into the store arena through `dupeSource` (:359-366). A copy is about 200 B, since Shape alone is about 170 B.
- Verifier (value, partial): The store's copies of registry entries are dead weight: `dupeSource`, Lowering's second copy, and `Lowering.Entry` with the comment that explains it. - Their only recorded reason (commit fc14935a: models registered derived tensors while binding) no longer holds. No production code registers after the registry is built.

**J-one-sticky-rule** (high, contract; move R4). Whether an error leaves the loader usable depends on which layer caught it, and awaitAll can report success after a terminal failure

- Where: `zml/zml/io/loader.zig:57-59`, `zml/zml/io/loader.zig:72` (+27).
- Problem: Three entry points share 16 error-set declarations (7 in loader.zig, 3 in backend, 4 in direct, 2 in buffered), about 60 members in all: all of File.OpenError, pjrt.ApiError and ReadPositionalError, plus about 20 loader names.
- Verifier (value, partial): The defects are real. - `awaitAll` returns success after a submit that failed with a partially published batch, when nothing else is pending. The cause is two first-error records: the front end's `Loader.failure` is never set from `backend.submit`. - `Closed` leaks (#29, still open).

**L-init-owns-its-task** (high, bug; move R1, R5). Loader.init blocks for calibration, so callers wrap a by-value Loader in a Future: examples/llm deinitializes a stale copy, and llmd needs an ownership flag

- Where: `zml/examples/llm/main.zig:85-92`, `zml/examples/llm/main.zig:145-148` (+6).
- Problem: `Loader.init` warms the device allocators, calibrates DMA and pre-grows the pinned arenas: 0.7-1.25 s on PCIe hosts. So both callers that care run it under `io.concurrent`. `Loader` is returned by value and mutated in place: the pending deque, the delivered map, the failure and the counters all change after init.
- Verifier (value, partial): The bug holds as stated in examples/llm. - The function-scope `defer cancel→deinit` deinitializes the copy taken before any use. - The live loader's pending Deque leaks, and the debug SafeAllocator reports it. - The pinned arenas and workers stay alive for the whole chat session.

**L-programs-ride-the-binding** (high, reshape; move R2). A program-computed tensor's link to its executable lives in three places and comes with a call-order rule; the binding should record it and load should run it

- Where: `zml/zml/io/Source.zig:566-580`, `zml/zml/io/loader.zig:36` (+23).
- Problem: The store records a program's inputs (`Binding.inputs`) but not which program computes it. llmd keeps that link in `Packer.recipes` (tensor id → Recipe), filled at 8 sites that each pair a `bindProgram` with a `recipes.put` by hand.
- Verifier (value, partial): The program can ride the binding, and `load` can run it. Corrections to the finding: - Gain: - In zml: `Loader.Binding`, the public `loadExecute`, `LoadExecuteError`, `delivered`/`DeliveryMap`, `logNotDelivered`, `TransformedTensorNotDelivered`, the plain-binding branch of `checkExecutableBinding`, and the packs-before-bulk …

**M1-load-returns-tree** (high, reshape; move R4). load writes into a buffer tree the caller must build from the same model; every caller bufferizes right before (or at init), and 10 zml functions already wrap load as 'bufferize, load, await, return the tree'

- Where: `zml/zml/io/loader.zig:203-240`, `zml/zml/io/direct_loader.zig:1295` (+22).
- Problem: The backends never read an output slot; they overwrite it whole: `destination.output.* = .fromPjrtBuffers(...)` (direct_loader.zig:1295, 1368) and `spec.output.* = try Buffer.from(...)` (buffered_loader.zig:208).
- Verifier (value, partial): The facts hold: the backends never read output slots, and every caller runs bufferize just before load, or at init so deinit can run on shells. But the proposed `load(allocator, model, store, programs) !Bufferized` costs performance and needs other changes first. - It needs all pack executables compiled before the call.

**M1-pair-by-structure** (high, bug; move R4). load and llmd's loadPacked pair tensors with output buffers by two independent visit indices and check nothing; a tree that is not bufferize of this exact value writes out of bounds or through a wild pointer

- Where: `zml/zml/io/loader.zig:556-563`, `zml/zml/io/loader.zig:580-643` (+7).
- Problem: prepareModelLoad sizes `flattened` by meta.count(Tensor, model), fills it with forEachVisit(buffers, *Buffer), and gives spec i `.output = flattened[i]` (loader.zig:637). Nothing compares the two walks. - Tree has more Buffers (an optional or slice element present at bufferize but absent now): the fill writes past `flattened`.
- Verifier (value, partial): This is a latent hazard, not a live bug, and its impact is medium to low. load and loadPacked pair tensors with outputs through two independent visit-order walks and never compare their lengths.

**M3-bind-time-fp8** (high, reshape; move R9). The platform's block-FP8 encoding and scale layout are opt-in post-passes that 2 of 11 models call; decide them in create/fuse/experts at bind time

- Where: `monorepo/llmd/weights.zig:138-166`, `monorepo/llmd/weights.zig:170-226` (+12).
- Problem: linear.detect classifies block-FP8 for every Linear built through create, fuse or experts (Scheme.classify, linear.zig:263). Whether the bytes then match what the GEMM reads depends on two calls the model must remember after building its tree. Neither is documented anywhere: there are no hits in AGENTS.md or docs. (1) CDNA3.
- Verifier (value, partial): The models' API problem is confirmed: two opt-in post-passes, 2 of 11 models wired, ordering rules, and a hand list whose reasons are gone. But a single bind-time bindBlockFp8 in create/fuse/experts cannot be built today. - The CDNA3 FNUZ conversion is a program, and linear.create has nowhere to put one.

**A-bytesloaded-is-a-log-line** (medium, remove; move R4). bytesLoaded exists so ten callers can rebuild the same 'Loaded weights' log line; four of them sit in functions nothing calls

- Where: `zml/zml/io/loader.zig:73-75`, `zml/zml/io/loader.zig:320-326` (+16).
- Problem: In production, bytesLoaded only feeds one log line. There are ten copies of the same block: a start timestamp, load, awaitAll, `bytesLoaded()`, a bandwidth division, then `log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]")`. One is in llmd models.zig:249-252, with its own `computeLoadBandwidth` helper; 8 are in examples/llm;
- Verifier (both, partial): The production facts hold. `bytesLoaded()` has one production reader, llmd's 'Loaded weights' line. `computeLoadBandwidth` has 1 caller. There are 10 copies of the line, 4 of them in uncalled example functions. Corrections: - The front-end Loader does not timestamp anything.

**A-error-sets** (medium, contract; move R1, R4). Three public error sets union about 20 loader names with file, PJRT and allocation errors; no caller switches on any of them, and most name model bugs rather than conditions a caller can act on

- Where: `zml/zml/io/loader.zig:38-45`, `zml/zml/io/loader.zig:57-59` (+8).
- Problem: A grep of both repos finds no `switch` or `catch error.X` on any loader error. The only mention is speculative.zig:868, which names `LoadError` as a return type. A caller can act on three kinds of error: a missing or forbidden source (File.OpenError and the VFS mappings), OutOfMemory or ResourceExhausted, and Canceled.
- Verifier (both, partial): Correct: no caller in either repo branches on a loader error; only speculative.zig:868 names a set. The three public sets differ only in which binding names they carry. Some names cannot fire after init: - InvalidOptions in PipelineError (maximumJobLen runs on config validated at direct_loader.zig:185).

**A-init-returns-pointer** (medium, bug; move R1, R5). Loader.init returns a stateful struct by value: the house future pattern deinitializes a stale copy (examples/llm), and llmd needs an ownership flag

- Where: `zml/zml/io/loader.zig:116-162`, `zml/zml/io/loader.zig:181-191` (+10).
- Problem: `init` returns `Loader` by value, but the struct changes after init: the pending deque, `delivered`, `failure`, `bytes_loaded` and the counters all live in the caller's copy. `std.Io.Future.cancel` after `await` returns the future's stored result again (zig 0.17 std/Io.zig:1311-1316). That is the init-time copy.
- Verifier (value, confirmed): I tried to refute it and could not. The bug is real, it comes from the API itself, and the fix is cheap. 1. Std semantics. In the zig 0.17 std that the repos pin (rules_zig 0.17.0, Io.zig:1311-1324), `Future.cancel` returns `f.result` once `any_future` is null, and `await` sets it to null.

**A-progress-by-value-loader-total** (medium, reshape; move R4). progress is ?*std.Progress.Node only to be copied, and the caller 'owns the estimated total' that only the loader can compute

- Where: `zml/zml/io/loader.zig:216-218`, `zml/zml/io/loader.zig:226` (+25).
- Problem: The pointer is dereferenced once: `if (progress) |node| node.* else .none` (loader.zig:443). `std.Progress.Node` is already a value handle with `.none`. The loader completes one item per LoadSpec: a composite tensor once, program tensors skipped by `load`, and loadExecute counting inputs rather than outputs.
- Verifier (both, partial): Most of the finding stands; three things change. What stands: - `progress` is `?*std.Progress.Node` only to be dereferenced once, at loader.zig:443. - The loader completes exactly one item per LoadSpec: the direct backend's Destination countdown (direct_loader.zig:1117) and one task per spec in the buffered backend …

**B-one-lookup** (medium, reshape; move R7). Eight lookup functions and an error set answer one question; View.get(subkey) returning the registry entry covers them all, and hasKey is a prefix scan that callers use as an exact test

- Where: `zml/zml/io/TensorStore.zig:13`, `zml/zml/io/TensorStore.zig:100-106` (+13).
- Problem: - hasKey(subkey) means 'some registry key starts with prefix + subkey', and it scans every key to find out. - Every caller uses it as an existence test for a key it could name exactly: q_norm (really q_norm.weight), activation_scale, gate_up_proj, lm_head.weight.
- Verifier (both, partial): - Seven of the eight hasKey callers name an exact key, and get(k) != null replaces them. qwen3_5:1504 does not: hasKey("") asks whether layer 0 has any experts. - In a stacked checkpoint that key is `experts.down_proj`; in a per-expert one it is `experts.0.down_proj.weight` or `.weight_packed`.

**B-registry-entry-is-identity** (medium, reshape; move R7). Each View.source copies the registry entry, so a checkpoint tensor has no identity: Lowering dedupes by name pointer and llmd compares names; hand out the registry's own entry and lock the registry for the store's life

- Where: `zml/zml/io/TensorStore.zig:84-98`, `zml/zml/io/TensorStore.zig:359-366` (+11).
- Problem: - Two copies per tensor, owning nothing. dupeSource arena-copies a whole safetensors.Tensor, Shape included, on every source() and createTensor. bind then copies it a second time in Lowering, 'for the binding to own'. - The copies own nothing: name and file_uri still point into the registry arena. - Nothing writes to them.
- Verifier (both, partial): Hand out the registry's own entry. Lock the registry map's pointers in fromRegistry and unlock them in deinit. dupeSource goes. Binding.tensors, Layout.whole, LoadPart.source and Move 7's Part.source become `*const safetensors.Tensor`, the registry's entry. Until Move 7 lands, Lowering's key is that pointer.

**B-replicated-drops-partitioning** (medium, contract; move R10). `.replicated` silently discards the partitioning argument, and 25 call sites pass one; the placement rule has no public form for Shapes

- Where: `zml/zml/io/TensorStore.zig:214-227`, `zml/zml/io/Source.zig:539-552` (+10).
- Problem: - The argument is dropped. Both placement paths return before reading `partitioning` when the mesh is `.replicated`. Yet 25 calls pass one anyway: `.{ .d = .replicated }`, `.{ .expert = .replicated, .d = .replicated }`, or the literal `.replicated`.
- Verifier (both, partial): `.replicated` drops its partitioning argument, and 25 sites pass one: 21 in llmd and 4 in examples/qwen3_5_moe. All 25 pass only replicated values, so the defect is a contract hazard: `.replicated, .{ .d = .model }` would compile and silently replicate. Five corrections to the finding.

**B-take-replaces-sourceOf-unbind-rebind** (medium, reshape; move R10). Reworking a bound tensor takes sourceOf, unbind and rebind in an order the caller must know; one take() does it

- Where: `zml/zml/io/TensorStore.zig:112-128`, `zml/zml/io/Source.zig:522-527` (+8).
- Problem: There is one intent: replace tensor T's binding with something made from T's source. Three public calls serve it, and they come in two different orders. - rebind needs T still bound when it runs, because it calls unbind itself to read `.memory`.
- Verifier (both, partial): Adopt B's take() as written, which returns `?struct { Source, Memory.Kind }`. It replaces sourceOf, unbind and Source.rebind, and ofBinding stays pub. Corrections: - A forgotten unbind leaves an inert map entry, because nothing iterates id_to_sources.

**C-binding-holds-source** (medium, reshape; move R13). A binding re-encodes its Source as Layout/Part, indexing into a deduplicated tensor list; sourceOf decodes it back (ofBinding) and the front end re-encodes it again as LoadPart

- Where: `zml/zml/io/TensorStore.zig:17-74`, `zml/zml/io/Source.zig:81-115` (+19).
- Problem: The same thing, "elements of checkpoint tensors placed in a tensor", exists in three representations: - Source pieces, in bits; - TensorStore.Part, in bytes, whose `source: u32` indexes `Binding.tensors`, a second deduplicated copy of every registry entry; - backend.LoadPart, in bytes, with a pointer.
- Verifier (both, partial): Binding = { memory, kind: union(enum) { source: Source, program: []const Source.Input } }. `sourceOf` returns the stored source relabeled to `tensor.shape()`'s tags. `bind` keeps its whole-byte check. `load`'s front end lowers pieces to LoadPart{ source: *const safetensors.Tensor, runs: backend.Lattice }.

**C-memory-one-setter** (medium, merge; move R10). The memory kind has five setters and is silently ignored on program bindings

- Where: `zml/zml/io/Source.zig:56-59`, `zml/zml/io/Source.zig:522-527` (+11).
- Problem: Five APIs set the one Binding.memory field: - BindOptions.memory, at bind; - TensorStore.setMemory, after bind, and its View forwarder; - createHostPinnedTensor and maybeCreateHostPinnedTensor (create, then setMemory); - rebind, which inherits the old binding's kind. The after-bind setter is genuinely needed.
- Verifier (both, partial): There are five ways to set Binding.memory. Three of them should simply go: View.setMemory (a forwarder used by examples/io and one test), createHostPinnedTensor (one test) and maybeCreateHostPinnedTensor (used only by createHostPinnedTensor).

**D-error-context** (medium, contract; move R11). The weights layer has about 55 error returns and 9 log lines; four error names mean 'the checkpoint does not match', and invariant breaks surface as checkpoint errors

- Where: `monorepo/llmd/weights.zig:378-392`, `monorepo/llmd/weights.zig:646-666` (+9).
- Problem: Non-test code in weights.zig and linear.zig returns about 13 MissingFusedComponent, 19 InvalidProjectionLayout, 10 IncompatibleProjections and 6 MissingWeight, and logs 9 err/warn lines. No caller in llmd catches any of these names; qwen4/model.zig:618 even reuses MissingFusedComponent for its own lookup.
- Verifier (both, partial): Outside tests, the weights layer returns 13 MissingFusedComponent, 19 InvalidProjectionLayout, 10 IncompatibleProjections and 6 MissingWeight, and logs 9 err lines. No caller in either repo catches these names, and 6 test expectations name them (1030, 1034, 1191, 1195, 1199, 2389).

**D-fused-forward** (medium, missing; move —). Every FusedLinear consumer splits the output by hand: 30 linear.forward sites followed by 75 extractOutput(y, i) calls with integer indexes

- Where: `monorepo/llmd/weights.zig:17-42`, `monorepo/llmd/models/llama.zig:1490-1494` (+7).
- Problem: FusedLinear exposes its Layout and the member-index convention. Each forward calls `.linear.forward(x, dtype)`, then `.layout.extractOutput(y, i)` once per member. The model must remember at forward time the member order it gave at init: for example `fuse(&.{q, k, v})` at init, then `extractOutput(qkv, 2)` for v.
- Verifier (both, partial): The counts hold: 75 extractOutput calls in 11 model files (plus 2 in a weights.zig test) and 30 FusedLinear forward sites. The proposal does not remove the init/forward order coupling; it turns an integer index into a destructuring position.

**D-one-fuse** (medium, merge; move R11). concatenateLinears equals interleaveLinears(.per_shard) for all 3 production callers, and Interleaving.explicit is test-only: one fuse covers both

- Where: `monorepo/llmd/weights.zig:44-56`, `monorepo/llmd/weights.zig:312-357` (+6).
- Problem: - outputShards returns 1 when the partitioning is fully replicated: linear.partitionShape drops the mesh (linear.zig:221-224) and weights.zig:957 tests for that. - All 3 concatenateLinears callers pass fully replicated partitionings (deepseek41.zig:744, qwen4/model.zig:437, :530).
- Verifier (both, partial): The central claim holds. All 3 production concatenateLinears calls are fully replicated, so outputShards gives groups = 1 and the result is identical to interleaveLinears(.per_shard). `.explicit` has no production caller;

**D-packer-state** (medium, remove; move R11). The Packer is not a concept: none of its five fields needs a model-lifetime object that every model threads

- Where: `monorepo/llmd/weights.zig:138-166`, `monorepo/llmd/weights.zig:170-225` (+12).
- Problem: Field by field: - `normalized` (weights.zig:142) is read and written only inside one normalizeBlockFp8 call, to fix up copies of a Linear (:191-194, :223). It still lives in every model's Spec until Model.deinit. - `platform` only gates normalizeBlockFp8 (:171-173) and maybeSwizzleBlockFp8Scales (:232-233).
- Verifier (both, partial): Four of the five fields do not need model lifetime: `normalized` can live for one normalizeBlockFp8 call, `platform` only gates two calls, `expert_layout` is set only by tests, and the allocator could be passed in. `recipes` does need model lifetime.

**E-has-exact** (medium, merge; move R7). Four ways to ask whether a checkpoint tensor exists: hasKey (a prefix scan of every key), getShape != null, linear.hasWeight, and a maybeCreateTensor call that binds a tensor and throws it away

- Where: `zml/zml/io/TensorStore.zig:200-206`, `zml/zml/io/TensorStore.zig:311-315` (+18).
- Problem: View.hasKey is named like an exact lookup but is a prefix test over `registry.tensors.keys()`: a linear scan of every checkpoint key on each call. Model code uses it as an exact existence test: - llama: q_norm and k_norm, every layer; - ministral3: every o_proj and down_proj; - gemma4 and qwen3_5: every MoE layer.
- Verifier (both, partial): - View.hasKey is a prefix scan, linear and stopping early on a hit, named like an exact lookup. Seven of its eight callers use it as an exact existence test. - getShape(k) != null is the O(1) exact idiom in 7 places. - dflash_drafter:830 probes by binding a tensor it throws away.

**E-host-mapped** (medium, missing; move R1, R12). Two models hand-build the 'pinned host table read by the GPU as device memory' path, each differently: marking, a hand-computed compiler HACK, a createView swap after the load, and a side list of owners

- Where: `monorepo/llmd/models/deepseek41.zig:263-274`, `monorepo/llmd/models/deepseek41.zig:1543-1558` (+17).
- Problem: Both production users of `.host_pinned` want the same result: the deepseek41 engram tables and qwen4's offloaded n-grams. Bytes live in pinned host memory and the GPU gathers through a device-kind view, so XLA never copies the table to HBM (CTX twenty-first pass).
- Verifier (both, partial): Both production users of loader `.host_pinned` (deepseek41 engram, qwen4 n-grams) rebuild the same steps by hand, and each places them differently: 1. Turn the loaded host buffers into device-kind views with `createView`. 2. Park the owners in a side list that must outlive the views. 3.

**E-ministral-fp8** (medium, remove; move R11). ministral3 hand-rolls a second FP8 Linear for o_proj and down_proj, while its fused q/k/v and gate/up read the same checkpoint through linear.detect: two FP8 schemes in one layer

- Where: `monorepo/llmd/models/ministral3.zig:997-1086`, `monorepo/llmd/models/ministral3.zig:903` (+7).
- Problem: Mistral FP8 checkpoints store a scalar weight_scale_inv and a scalar activation_scale for each Linear. The model handles the same format two ways: - Fused projections (interleaveLinears over linear.view): linear.detect classifies the scalar weight_scale_inv as fp8_per_tensor (quantization.zig:79) and drops activation_scale, …
- Verifier (both, partial): The finding holds, with three corrections: - No performance loss. Fp8Linear's "static W8A8" only emulates FP8. zml.ops.scaled_dot (ops.zig:3380-3385, its only caller) converts both operands to bf16 and runs a bf16 dot. Moving o_proj and down_proj to linear.create keeps speed and drops simulated activation rounding.

**E-unload-generic** (medium, remove; move R3). Models hand-write 68 unloadBuffers walkers, the inverse of bufferize, while zml.mem.deinitBufferized already does it generically and only tests call it

- Where: `zml/zml/mem.zig:19-29`, `zml/zml/buffer.zig:103-121` (+21).
- Problem: The buffers loader.load fills are a Bufferized(T) that the model allocates with zml.mem.bufferize, which also allocates the layer slices. Freeing them is the exact inverse, and zml provides it: deinitBufferized walks every Buffer, optional, union and slice.
- Verifier (both, partial): Replace the hand-written loaded-weight walkers with zml.mem.deinitBufferized(allocator, T, &buffers) at each owner of a bufferize result. In llmd that is 68 definitions plus 70 calls; the calls span 14 files because models/qwen4.zig only calls.

**F-calibration-read-through-backend-field** (medium, contract; move R5). examples/io reads the DMA sizing through Loader.backend.direct.calibration, so the internal union is de facto public API, and Move 4 deletes that field as dead

- Where: `zml/examples/io/main.zig:172-185`, `zml/zml/io/direct_loader.zig:79` (+11).
- Problem: Outside zml/zml/io, the only code that reaches into the backend seam is examples/io's dma-bench. It runs `switch (loader.backend) { .direct => |backend| backend.calibration, ... }`, and loader.md:353 documents `loader.backend.direct.calibration` as the way to read the selected sizing.
- Verifier (both, partial): `direct_loader.Loader.calibration` is not dead: examples/io's dma-bench reads it at main.zig:183, and loader.md:353 documents that path, so Move 4 must not delete it. The reach-in through `loader.backend` is the user's own choice.

**F-options-validated-thrice** (medium, bug; move R5). Options are validated in three places by three mechanisms; examples/io's environment variables reach debug asserts in calibrate

- Where: `zml/zml/io/loader.zig:785-797`, `zml/zml/io/loader.zig:122` (+16).
- Problem: Option checks live in three places, each with its own mechanism. 1. `validateOptions` returns logged InvalidOptions errors. It checks read_parallelism, and checks read_chunk_size by calling `effectiveSourceRequestSize(chunk, 0)`, a request-size calculator used here as a validator. 2.
- Verifier (value, partial): Two separate claims. 1. Front-end chunk check (the valid part). Loader.init validates `read_chunk_size` through a request-size calculator, then the direct Sizing reruns the same calculator. The rerun can't fail after Loader.init, and its `selected > max` branch can't be reached. This is redundant, not a bug. 2.

**F-zero-fill-is-a-size-check** (medium, reshape; move R13). LoadSpec.zero_fill only exists to make a plain tensor's size mismatch surface inside the backends, where the two backends disagree on its outcome

- Where: `zml/zml/io/backend.zig:81-84`, `zml/zml/io/backend.zig:366` (+15).
- Problem: Composite specs always set zero_fill = true. The front end's own checkParts call passes true too (loader.zig:764), so it can never return SourceSizeMismatch. Plain specs leave it false, and their one part is `.whole(source)`, whose width is `source.byteSize()`.
- Verifier (both, partial): The claim holds as stated: zero_fill only turns a plain tensor's size mismatch into SourceSizeMismatch inside the backends. Direct reports it before publication and stays usable; buffered reports it from a worker and stays failed. loadExecute cannot mismatch, so the check belongs in prepareModelLoad's plain branch.

**G-job-kinds** (medium, reshape; move R14). A Job encodes six kinds of work in four flags, and a Transfer's block fields mean different things for DMA and host jobs

- Where: `zml/zml/io/direct_loader.zig:54-67`, `zml/zml/io/direct_loader.zig:776-812` (+11).
- Problem: Job has 15 fields. Its kind is spread over fill × delivery × gather × (item_len != 0): 24 combinations, of which seven are valid. They are DMA read, DMA staged, DMA gather, DMA fill, host in place, host through blocks, and host fill, which is encoded as `.fill` plus `.host_direct` although it reads nothing.
- Verifier (both, partial): Seven job kinds are valid, not six. They take more than seven flag combinations, because a gather job may have zero items over an uncovered range. The proposed `dma_gather` arm lacks the file offset that the claim-order sort (:1743) and the worker's error log (:654) read.

**H-backend-options** (medium, merge; move R15). Four remote backends, four constructor shapes, and the same five retry knobs declared four times under names that RetryConfig renames

- Where: `zml/vfs/http.zig:14-25`, `zml/vfs/http.zig:67-85` (+11).
- Problem: Each backend is built differently: - HTTP: init(alloc, io, client, protocol) and initWithOptions(..., protocol, opts), with no environment constructor. - HF: init(alloc, io, client, hf_token: ?[]const u8, opts), plus auto. - S3: init(alloc, io, client, config: Config, opts: InitOpts), plus auto.
- Verifier (both, partial): The core holds: the five retry knobs are declared four times, word for word with the same defaults; RetryConfig.fromOptions duck-types them and renames two; HTTP has two entry points; constructor shapes are inconsistent; auth_service is never set; GCS region is never read.

**H-direct-io-per-file** (medium, reshape; move R15). Direct I/O is a per-file VFS decision gated by a per-loader profile field: a local drafter beside an hf:// model is never read directly, and an hf:// drafter beside a local model is opened on the planner

- Where: `zml/zml/io/direct_loader.zig:145`, `zml/zml/io/direct_loader.zig:1575-1591` (+20).
- Problem: Today the loader computes `source_alignment = if (direct_io != .off) profile.direct_io_alignment orelse 0` (direct_loader.zig:145), and sets `Reads.vfs = if (config.alignment != 0) VFS.fromIo(io)` (:1624). The alignment is therefore one value per loader, taken from args.model, while the decision it gates is per file.
- Verifier (both, partial): The proposal holds with these changes. 1. Gate the VFS constant on support: `pub const direct_io_alignment: usize = if (direct_io.supported) direct_io.alignment else 0;`. `Loader.create` then sets `alignment = if (opts.direct_io != .off and VFS.fromIo(io) != null) VFS.direct_io_alignment else 0`. 2.

**H-error-contract** (medium, contract; move R6). The VFS has no error contract: 'absent' and 'denied' surface in four different ways, access() succeeds for any remote path, and the documented error.RateLimited never reaches a caller

- Where: `zml/vfs/http.zig:232-237`, `zml/vfs/http.zig:252` (+11).
- Problem: The same situation gets a different error depending on the scheme: - Missing object: HTTP returns Unexpected. S3 returns FileNotFound for a HEAD 404, but a principal without ListBucket gets a 403 for a missing key, which becomes Unexpected. GCS returns FileNotFound.
- Verifier (both, partial): The VFS has no error contract, and the mappings diverge as the finding says. Corrections: (a) safetensors.resolveModelRepo uses openDir. On http, s3 and gs that sends no request; on hf, dirOpenDir flattens tree failures, Canceled included, to Unexpected, which resolveModelRepo turns into InvalidPath.

**H-request-send** (medium, merge; move R15). Each governed request is written as perform, an attempt hook, exchange and a consume hook, with its RequestSpec spelled out twice; range reads use a second, type-erased hook system

- Where: `zml/vfs/request.zig:143-179`, `zml/vfs/request.zig:181-214` (+6).
- Problem: There are 8 metadata request structs: HTTP SizeHop, HF TreeFetch and ResolveHop, S3 Listing and SizeRequest, GCS TokenRequest, Listing and SizeRequest. Each calls perform(T, ctx, spec, self, attempt). The attempt hook builds its headers and then calls exchange(T, ctx, uri, options, spec, self, consume).
- Verifier (both, partial): The core holds: every governed request states its RequestSpec and Context twice, and each copy is half dead (`unavailable` and `retry_once` are read only through `exchange`; `key` only through `admit`/`reportThrottle`). The copies already disagree in S3 Listing (s3.zig:690 vs :722).

**H-stats-stay-in-vfs** (medium, remove; move R15). VFS read statistics reach the loader through four types to feed one debug line; the backend that owns them should log them

- Where: `zml/vfs/base.zig:115-122`, `zml/vfs/base.zig:147-160` (+7).
- Problem: The statistics take this path: 1. Each backend keeps an AtomicReadStats. 2. provider() type-erases it. 3. Backend.read_stats carries it. 4. loadProfile copies it into LoadProfile.stats. 5. The direct loader snapshots it at submit, subtracts at await, and logs a 'batch source' line at debug.
- Verifier (both, partial): Delete ReadStatsProvider, AtomicReadStats.provider/snapshotOpaque, Backend.read_stats, LoadProfile.stats, Diagnostics.source_stats, the 'batch source' block, and also the direct Loader's stored `load_profile` field (direct_loader.zig:78, :158), which has no other reader after init.

**I-construction-surface** (medium, contract; move R1, R7). The registry has 13 public construction declarations where callers use 2; `parseSafetensors` appends into a shared registry, which is what makes the remote parse serial; llmd never closes the repo Dir

- Where: `zml/zml/safetensors.zig:9-10`, `zml/zml/safetensors.zig:108-157` (+15).
- Problem: The public construction surface is: TensorRegistry.{init, initWithMetadata, fromRepo, fromPath, registerTensor, mergeMetadata}, plus the free functions fetchRegistry, parseSafetensors, resolveFiletype (with FileType), resolveModelRepo and resolveModelEntrypoint (with ModelPathResolutionError).
- Verifier (both, partial): The registry has 13 public construction declarations. Production callers use 3 of them (resolveModelRepo, fromRepo, fromPath) and tests use 3 more (init, registerTensor, parseSafetensors).

**I-dead-metadata** (medium, remove; move R7). Safetensors metadata (`__metadata__` and the index's `metadata`) is parsed, flattened, cloned and merged into every registry, and nothing reads it; the index parse also copies every weight name into lists nobody reads

- Where: `zml/zml/safetensors.zig:7`, `zml/zml/safetensors.zig:10` (+9).
- Problem: Grep across both repositories finds no caller outside safetensors.zig for `registry.metadata`, `Metadatas`, `initWithMetadata` or `mergeMetadata`. `Metadata.wrap` and `Metadata.copySlice` are not called inside the file either.
- Verifier (both, partial): The finding's proposal stands unchanged. Correct its gain and cost as follows: - Three public declarations go (Metadatas, initWithMetadata, mergeMetadata), plus the `metadata` and `mutex` fields of TensorRegistry. `Metadata` itself is file-private (:1077).

**I-reader-is-two-functions** (medium, reshape; move R7). TensorReader is a streaming reader type with two constructors, an ownership flag and two vtables, yet every caller either reads a whole tensor into a buffer or reads a tensor-relative range from a file it already holds

- Where: `zml/zml/safetensors.zig:15-106`, `zml/zml/safetensors.zig:159-390` (+17).
- Problem: The callers fall into two modes. - Streaming through `interface`, always the whole tensor from byte 0 into one buffer: llmd's normalizeLinearBlockFp8 (weights.zig:203-205, through canDoubleScale's 64 KiB loop), zml/testing.zig:473-478, llama_tests:148-152 and lfm2_tests:318-321.
- Verifier (both, partial): Deleting TensorReader holds. This means TensorReader itself, InitOpts, owns_file, both constructors, the stream/discard vtables, Tensor.reader, TensorRegistry.reader (0 callers), TensorStore.getReader, View.getReader and ReaderError.

**J-lifetime-graph** (medium, contract; move R1). Two borrows that last as long as the loader are undocumented: the registry's file URIs and the VFS

- Where: `zml/zml/io/loader.zig:47-51`, `zml/docs/learn/loader.md:55-60` (+8).
- Problem: Each edge of the lifetime graph, against what the docs say: - Platform → loader, executables and buffers. Documented. - io/VFS → loader lifetime. Not documented. Files are opened through `io` and closed through it in `destroy` (direct_loader.zig:364-368).
- Verifier (both, partial): The registry edge is #8 unchanged: `source_slots` keys and `SourceSlot.uri` borrow the registry's `file_uri` for the loader's whole life, while the docs say the borrow ends at retirement. The fix and its test are already specified in #8.

**J-outputs-empty-at-submission** (medium, contract; move R4). After a failure the caller must free "the outputs that were assigned" but cannot tell which, and a forgotten awaitAll is silent

- Where: `zml/zml/io/loader.zig:164-191`, `zml/docs/learn/loader.md:79-86` (+6).
- Problem: `cancel` documents its outputs as "undefined: unwritten, errored, or ... partly written: free the ones that were assigned, read none". Bulk outputs are assigned on worker threads at `TensorTransfer.init` (direct_loader.zig:1295, 1368), so the caller cannot know which ones were.
- Verifier (both, partial): Confirmed: the cancel/deinit contract ("free the ones that were assigned") cannot be acted on for outputs that start `undefined`, because bulk outputs are written on worker tasks (direct_loader.zig:1295, 1368; buffered_loader.zig:208).

**K-dma-options-one-knob** (medium, contract; move R5). Options.dma exposes 8 calibration knobs through a public module; they are checked only by assert and set only by examples/io from environment variables

- Where: `zml/zml/io.zig:7`, `zml/zml/io.zig:12` (+26).
- Problem: - Unused knobs. `confirmation_duration`, `confirmation_minimum_transfers` and `confirmation_margin` are set nowhere in either repo. `minimum_duration`, `minimum_transfers` and `block_selection_tolerance` are set only by dma-bench.
- Verifier (both, partial): Loader.Options.dma shrinks to a single `dma_block_sizes` list, with each value checked in validateOptions and an empty list or an out-of-range value returning InvalidOptions. A one-element list skips measurement on CPU only.

**K-docs-one-place-per-contract** (medium, doc; move R16). Each loader contract is written three times (loader.md, the doc comments, CTX.md) and the copies disagree; two-thirds of the user page describes internals

- Where: `zml/docs/learn/loader.md:64-73`, `zml/docs/learn/loader.md:133-196` (+9).
- Problem: The copies disagree: - Admission. loader.md:68-70 and CTX:150-152 say that without a device memory limit "every pending submission is retired first". loader.zig:331-334 and the loop at :359 retire only until no executable submission is pending. - Binding inputs. loader.zig:107 says one input "per source";
- Verifier (both, partial): Each loader contract has two to four copies: loader.md, CTX "Current design", the pub doc comments, and private doc comments in loader.zig. - Copies that disagree: - admission when room cannot be measured: loader.md:68-70, CTX:150-152 and loader.zig:76-79 (a fourth copy, in the code) contradict admit() at :328-334 and :359;

**K-example-loaders** (medium, remove; move R4). examples/llm carries four dead model loaders and four identical live ones

- Where: `zml/examples/llm/models/llama/model.zig:72-94`, `zml/examples/llm/models/llama/model.zig:156-184` (+11).
- Problem: Each example model has two load functions. The Model-level one (`Model.loadBuffers` or `Model.load`) takes the platform and a `VFS.LoadProfile`, creates and destroys its own Loader (calibration included) and estimates progress. Nothing in either repo calls it.
- Verifier (both, partial): Correct as stated: - The four Model-level loaders (`Model.loadBuffers` and `Model.load`, taking a platform and a LoadProfile) have 0 callers in zml or monorepo, tests and docs included. They come to about 110 lines. - The four `LoadedModel.loadBuffers` are the same 22 lines, differing only by type.

**K-newcomer-recipe** (medium, doc; move R5, R16). No tutorial loads a checkpoint, and the one newcomer sample (mnist/README) sets two knobs nobody needs

- Where: `zml/examples/mnist/mnist.zig:39-64`, `zml/examples/mnist/mnist.zig:99` (+6).
- Problem: The newcomer path (docs/README → getting_started → write_first_model → concepts) never loads a checkpoint: - write_first_model says TensorStore "is optional" and builds buffers from slices.
- Verifier (both, partial): The docs path never loads a checkpoint, the tutorial's `fromSlice` calls are broken, and mnist/README carry leftover knobs. Three of those knobs are simply unneeded: the `replicated_mesh` passed to the store, the README's `.dma = .{ .block_parallelism = 8 }` (the default), and the hand-written `unloadBuffers`.

**L-one-store-handle** (medium, merge; move R8, R10). Model code gets two handles on one store with overlapping method sets, and llmd reaches through view.store 24 times for what View lacks

- Where: `zml/zml/io/TensorStore.zig:100-160`, `zml/zml/io/TensorStore.zig:162-341` (+17).
- Problem: Model code holds a View: a store pointer plus a 256-byte prefix. View duplicates TensorStore's getShape, getReader and setMemory, but lacks mesh, sourceOf and unbind. So llmd reaches through `view.store`: - `.mesh(..)` 11 times; - `.meshes` 5 times, to build the drafter's store (llama, gemma4, laguna, qwen3_5, muse_glimmer);
- Verifier (both, partial): Model code holds a View, and store-wide operations are on TensorStore. llmd reaches through `view.store` 24 times: - `.mesh` 11 times; - `.meshes` 5 times; - a raw store passed to the two whole-model FP8 post-passes 5 times; - `.unbind` 2 times, in bf16Linear; - 1 identity check.

**M2-binding-holds-pieces** (medium, reshape; move R13). If a Binding keeps what bind was given, it should keep the pieces, not a Source. A stored Source points back at its own store and hands the read-only loader a mutable store.

- Where: `zml/zml/io/TensorStore.zig:17-74`, `zml/zml/io/TensorStore.zig:116-120` (+9).
- Problem: Today bind lowers the pieces to Layout/Part bytes, with a deduplicated, copied tensor list indexed by Part.source (Lowering, Source.zig:584-651). sourceOf then decodes those bytes back into pieces (ofBinding, :81-115, 35 lines whose only caller is TensorStore.zig:119). C proposes that the Binding hold the Source instead.
- Verifier (both, partial): The Binding stores Source pieces, not bytes, and the loader front end lowers them once. What the gain really is: - Removed: ofBinding (the only bytes-to-pieces decoder; sole caller TensorStore.zig:119), the public Layout and Part, and the bytes-to-pieces-to-bytes round trip on `sourceOf` then `rebind`.

**M2-view-carries-registry** (medium, reshape; move —). Sources already compose across checkpoints. The drafter has a second TensorStore only because View reaches its registry through the store.

- Where: `zml/zml/io/TensorStore.zig:144-146`, `zml/zml/io/TensorStore.zig:162-166` (+13).
- Problem: A Source piece names a `*const safetensors.Tensor` from any registry. Lowering keys tensors by name pointer and offset (Source.zig:588-592), so two registries never alias. The loader identifies files by URI and never reads `store.registry` (no read in loader.zig, direct_loader.zig, buffered_loader.zig or backend.zig).
- Verifier (both, partial): The proposal holds as a small reshape. It does not unblock a single load phase. Facts confirmed: - View finds keys only through `store.registry`. - The loader and the backends never read the registry. - A Source requires one store but accepts any registry.

**M3-bf16linear-bind-once** (medium, reshape; move R9). bf16Linear binds through create, then unbinds both tensors to rebind a program: the last production caller of TensorStore.unbind

- Where: `monorepo/llmd/weights.zig:260-311`, `monorepo/llmd/models/deepseek41.zig:746` (+3).
- Problem: bf16Linear repeats work and unbinds what it just bound: - It runs linear.detect twice (directly at :267, and inside create at :268). - It re-reads both shapes with getShape (:276-277). - It lets create bind the weight and scales, unbinds both (:298-299), re-sources them with store.source (:295-296), then binds a program.
- Verifier (both, partial): On its own, the finding is a local reshape of bf16Linear (2 callers, both DeepSeek wo_a). It builds the target with linear.view plus linear.partitionShape, the idiom assembleLinears already uses at weights.zig:370.

**M3-drop-scale-guard** (medium, remove; move R9). The scale-overflow guard is why the CDNA3 conversion needs io and the store's binding internals, and its fallback is the untested FN×FNUZ path

- Where: `monorepo/llmd/weights.zig:196-206`, `monorepo/llmd/weights.zig:1396-1418` (+2).
- Problem: Before converting, normalizeLinearBlockFp8 reaches into the binding's internals (getSourcesById(...).tensors and .memory). It then opens and reads every member scale tensor serially at Model.init, through a TensorReader per tensor, to find a scale whose exponent field is 254.
- Verifier (both, partial): Dropping the guard removes the reader loop together with `canDoubleScale` and its test (~55 lines). It also removes the `io` parameter on `normalizeBlockFp8`, the file error set, the only checkpoint read in llmd's model init (92,160 serial opens on Qwen3.5-MoE), and the only data-dependent input to the CDNA3 decision.

**M3-dspark-unswizzled** (medium, bug; move R9). deepseek41's hand-listed swizzle misses the DSpark (MTP) layers, which build the same projections

- Where: `monorepo/llmd/models/deepseek41.zig:1001-1003`, `monorepo/llmd/models/deepseek41.zig:3779-3786` (+3).
- Problem: Layer.init opts in, on sm_10x: attn.wq_a_wkv, wq_b, wo_b, ffn.shared_experts.gate_up and down, indexer.wq_b and engram.wkv. DSparkLayer.init (:3901-3923) builds a DSparkAttention whose wq_a, wq_b, wkv and wo_b come from linear.create (:3781-3785), plus the same MoE type with its shared experts. It calls no swizzle.
- Verifier (value, partial): The bug is confirmed, and the checkpoint answers the finding's open question. mtp.{0,1,2} wq_a, wq_b, wkv, wo_b, shared_experts w1/w2/w3, and mtp.0.main_proj are F8_E4M3 with F8_E8M0 block-32 scales, the same format as the main layers. So on sm_10x every draft step's GEMMs use the unswizzled scaledDot. No dtype guard is needed.

**M3-fp8-target** (medium, missing; move R9). Four places choose the platform's FP8 encoding by three rules, and zml has no API that answers it for weights

- Where: `zml/zml/quantization.zig:124-147`, `zml/zml/quantization.zig:150-181` (+9).
- Problem: Three rules, four places: - Dense activations: chosen from the platform at compile time, whatever the weight's encoding (quantizeInput, quantization.zig:131-134). - MoE activations: chosen from the weight dtype (fused_experts.zig:180 and :201 pass gate_up.dtype() / down.dtype()).
- Verifier (both, partial): Two live copies of the platform-to-E4M3 switch exist, one in each repository: zml's quantizeInput (quantization.zig:128-136) and llmd's normalizeBlockFp8 (weights.zig:171-173). A third copy sits in quantizeMxfp8, which has 0 callers. nn.zig:230 is an unreachable branch in a test that runs only on CUDA.

**B-store-owns-meshes** (low, move; move R2, R8). The model's meshes already live in the store, yet llmd threads the same list through four functions and reads it back through View.store for drafters

- Where: `zml/zml/io/TensorStore.zig:79`, `zml/zml/io/TensorStore.zig:84-93` (+10).
- Problem: - main registers the three meshes once (parallelism.registerMeshes) and passes them to fromRegistry, which keeps an arena copy to serve mesh(name). - The same three meshes come back as `parallelism.meshes()` into loadInto (9 production calls, 13 tests). From there they go through loadPacked, program and compileRecipe.
- Verifier (both, partial): loadInto has 22 callers: 10 in production (including dflash_drafter:877 and dspark_drafter:138) and 12 in tests (11 in weights.zig plus deepseek41:3222). Each passes a `meshes` list identical to its store's `meshes`, in the same order. `meshes` threads through loadInto, loadPacked, program and compileRecipe.

**C-pieces-leak** (low, missing; move R7). Source's lattice representation is public, and llmd derives source identity from it

- Where: `zml/zml/io/Source.zig:24-49`, `monorepo/llmd/weights/linear.zig:99-106` (+3).
- Problem: `pieces`, `Piece` and `Dim` (bit lattices) are public. llmd's `sameTensor` compares `pieces[0].tensor.name`, `.tensor.offset`, the piece `.offset` and the shape to decide whether fused FP8 per-tensor members share one scale. That decision picks fp8_per_tensor or fp8_per_channel.
- Verifier (both, partial): Source's lattice representation is public: the plain `pieces` field and the `pub` types `Piece` and `Dim`. Its only outside reader is llmd. `linear.sameTensor`, with one caller (`haveSharedScale`), decides whether fused fp8_per_tensor members share one scale.

**D-linear-seam** (low, move; move R11). Fusion lives in weights.zig but is built from linear.zig internals, so linear.zig exports helpers only weights.zig uses

- Where: `monorepo/llmd/weights/linear.zig:100-106`, `monorepo/llmd/weights/linear.zig:120-130` (+6).
- Problem: Models use only these parts of linear.zig: View, view, maybeView, split, create, maybeCreate and hasWeight. Its other pub declarations exist for fusion in weights.zig: - partitionShape (weights.zig:370, :956); - sameTensor (:965); - GlobalScaleDefinition.create (:483); - View's component accessors and `definition`.
- Verifier (both, partial): "Fusion is built from linear.zig internals" holds: partitionShape (2), sameTensor (1) and GlobalScaleDefinition.create (1) have callers only in weights.zig's dense fusion. Moving the stateless fusion builders next to View lets those three become file-private.

**E-loadbuffers-protocol** (low, reshape; move R4, R8, R11, R12). Model.loadBuffers is a duck-typed protocol: a default that reaches into four field names, seven overrides for three reasons, a hook nobody implements, and three awaits

- Where: `monorepo/llmd/models.zig:221-253`, `monorepo/llmd/models/llama.zig:643-646` (+9).
- Problem: The default in models.zig (used by gemma3_text, lfm2, ministral3/mistral3) needs `spec.tensors`, `model_buffers`, `spec.packer` and `spec.parallelism` by name (@FieldType at :233-236). That interface is written down nowhere.
- Verifier (both, partial): The problem statement is accurate: - The default branch reaches into four field names. - There are seven overrides for three reasons; four of them are byte-identical. - `finalizeLoadedBuffers` has no implementer. - Completion is awaited three times. - Progress nodes are inconsistent and leaked.

**E-moe-setup** (low, merge; move R11). Every MoE model repeats the same expert setup: expertsFormat then Backend.auto (which ignores dtype), a hand-built layer path, a probe for fused or split gate/up, and two packer.experts calls with the same count and backend

- Where: `monorepo/llmd/weights.zig:62-86`, `monorepo/llmd/weights.zig:410-446` (+15).
- Problem: Five MoE models run the same two lines at Model.init: `const scheme, const dtype = try weights.expertsFormat(<hand-built path to some MoE layer's experts>, "<projection>"); moe_backend = try .auto(platform, scheme, dtype);` - Backend.auto discards dtype (moe.zig:91).
- Verifier (both, partial): The repetition is real, and a paired `weights.experts(packer, store, .{ .num_experts, .backend, .gate_up = "gate_up_proj", .gate = "gate_proj", .up = "up_proj", .down = "down_proj" }) !Experts{ gate_up, down }` is a sound merge.

**E-ngram-concatenate** (low, remove; move R2). qwen4's n-gram table, concatenateRows' only production caller, uses getShape to re-derive what Source.concatenate already computes

- Where: `monorepo/llmd/models/qwen4/model.zig:603-627`, `monorepo/llmd/weights.zig:838-877` (+5).
- Problem: NgramEmbedding.init does a lot of work by hand: - allocates an arena from packer.allocator; - builds the `shard_{d}.weight` keys; - sums rows with getShape, then re-reads the first shape for columns and dtype; - hand-builds the output Shape and its tp_mesh placement; - calls packer.concatenateRows.
- Verifier (both, partial): qwen4's NgramEmbedding.init is the only production caller of Packer.concatenateRows. It always takes the same-dtype branch, because it builds the output dtype from shards[0], and it re-derives with getShape what Source.concatenate checks itself.

**E-replicated-one-spelling** (low, contract; move R10). A replicated checkpoint tensor has two spellings, and the 4-argument one silently ignores its partitioning argument

- Where: `zml/zml/io/TensorStore.zig:209-239`, `zml/zml/io/TensorStore.zig:263-269` (+5).
- Problem: `createTensor(key, tags, .replicated, partitioning)` takes the `.replicated` branch at TensorStore.zig:217-218 and never reads `partitioning`. 21 llmd call sites therefore pass a dead argument (`.{ .d = .replicated }`, `.replicated`). A slip like `.{ .d = .model }` would silently replicate.
- Verifier (both, partial): `maybeCreateTensor` drops `partitioning` when the mesh is `.replicated` (TensorStore.zig:217-218), so 21 llmd sites (25 overall) pass a dead argument, and a wrong spec silently replicates. Of those 21, 13 call createTensor and 8 call maybeCreateTensor, not 6.

**F-backend-union-forwarders** (low, reshape; move R13). Backend and Submission are hand-written forwarders: the two creates take different signatures, every pending submission stores a redundant loader pointer, and the union lives in a file its only user does not own

- Where: `zml/zml/io/backend.zig:12-18`, `zml/zml/io/backend.zig:613-697` (+12).
- Problem: A tagged union earns its place over an interface or two front ends. The variant set is closed at two, and the target is picked at runtime. Dispatch is static, and the front end can switch where the backends differ (probeMemory, readRoom).
- Verifier (both, partial): The forwarders, the different create signatures, the redundant loader pointer per PendingSubmission and loader.zig being the only user are all confirmed. The gain is about 8-10 lines of structure, not 45.

**F-free-functions-misplaced** (low, move; move R12, R13). placedShape, supportsMemory and countsAgainstDevice are platform and memory-kind facts written as free functions in backend.zig, with private copies or second names elsewhere

- Where: `zml/zml/io/backend.zig:577-611`, `zml/zml/io/direct_loader.zig:543` (+11).
- Problem: (a) placedShape. All 5 non-test front-end calls are either `Sharding.Placement.init(backend.placedShape(p, s).packedShape())` or `placedShape(...)._sharding.mesh.?`. The direct loader wraps the same thing as `placementShape`, adding packedShape. buffer.zig keeps its own private copy, `placementOrPanic`.
- Verifier (both, partial): (a) holds only for the loader. Every loader call site of `backend.placedShape` either packs the result or reads only its mesh, and `direct_loader.placementShape` is a one-line wrapper that adds the packing.

**F-leaf-modules-one-consumer-each** (low, move; move R5). limits.zig and execute_admission.zig are leaf modules in which every declaration has exactly one consumer

- Where: `zml/zml/io/limits.zig:1-39`, `zml/zml/io/execute_admission.zig:1-46` (+6).
- Problem: Each declaration has one consumer: - `max_read_parallelism` → validateOptions. - `max_dma_parallelism` → one assert in calibrate. - `local_` and `high_latency_read_parallelism` → defaultReadParallelism. - `defaultReadParallelism` → Options.readWidth.
- Verifier (both, partial): Most limits.zig declarations have one consumer, and execute_admission's two functions have one caller each. The exception is max_read_request_size. It is a bound shared by three files: - the front end's read_chunk_size check, which also protects the buffered backend's division by read_chunk_size;

**G-plan-dedupe-device-check** (low, move; move R13). preparePlan stores a per-plan 'seen' index on the batch-shared Destination, and checks device ids there, after earlier files are already published

- Where: `zml/zml/io/direct_loader.zig:1089-1092`, `zml/zml/io/direct_loader.zig:1671-1696` (+8).
- Problem: plan_slot is written with an index (:1689) that is never read as one; it is only compared against `unplanned`, so it is a 'seen' bit kept as scratch on a shared runtime type. The deduplicated `destinations` list has one reader, appendFillJob, which filters it again to destinations whose parts[0] is in this file (:2337).
- Verifier (both, partial): Prior: this is LOADER_REVIEW.md Part 1 #28, nearly verbatim (validate once before addPlacementBytes; delete Config.device_count, the planner check, plan_slot/unplanned and JobCut.destinations; appendFillJob selects `part == &part.destination.parts[0]` over `order`). It is not "new".

**G-submit-closed** (low, bug; move R1, R4). submit can return error.Closed instead of the loader's sticky error, and the public error sets carry members no caller can act on

- Where: `zml/zml/io/direct_loader.zig:234`, `zml/zml/io/direct_loader.zig:244-249` (+10).
- Problem: Submissions overlap: the front end publishes while older submissions are pending (loader.zig keeps a `pending` deque, and llmd publishes its pack groups through loadExecute this way).
- Verifier (value, partial): The bug is prior #29 (LOADER_REVIEW.md:256-258), not new, and it is still unfixed at direct_loader.zig:248. The real cause is already logged at err level by recordError; only the error returned to the caller is wrong (Closed).

**G-workspace-protocol** (low, contract; move R14). The pinned workspace is driven by five calls in a fixed order, and its one owner does the ceiling arithmetic

- Where: `zml/zml/io/host_memory.zig:298-397`, `zml/zml/io/host_memory.zig:401-422` (+7).
- Problem: Sizing computes capacity from Workspace internals: `usableBlocks + (max_mapped_bytes - mapped_bytes) / block_size` (:459-460). It then grows twice through `pregrow` and calls the free function carveBlocks. host_memory.zig:401-408 states the rule that nothing may map after carving; only the call order enforces it.
- Verifier (both, partial): The central claim holds, but four details are overstated. - Sizing reaches into Workspace's fields for its capacity (`usableBlocks + (max_mapped_bytes - mapped_bytes) / block_size`). - It must call pregrow, which runs growToBlocks twice, before the free function carveBlocks.

**H-vfs-surface** (low, remove; move R6). VFS exports surface no caller uses: unregister (which invalidates open handles), two vtable functions made pub, and re-exported internals

- Where: `zml/vfs/vfs.zig:139-146`, `zml/vfs/vfs.zig:505` (+2).
- Problem: unregister has 0 callers. It uses orderedRemove, which shifts the index stored by every open handle of a later-registered backend (Handle.backend_idx), so it is only safe when no such handle is open. The doc states that constraint; nothing enforces it.
- Verifier (both, partial): Delete `VFS.unregister` (0 callers). Its orderedRemove shifts the backend index stored by open handles. Without it, backend indices only grow. Make the three pub vtable overrides private: `dirStatFile`, `dirAccess` and `dirCreateFile` (vfs.zig:505/514/523).

**J-calibration-entry-point** (low, missing; move R5). dma-bench reads the calibration result through loader.backend.direct, so Move 4's "dead Loader.calibration" is not dead

- Where: `zml/examples/io/main.zig:160-191`, `zml/zml/io/direct_loader.zig:79` (+2).
- Problem: No public API exposes the calibration result. To get it, examples/io's dma-bench builds a whole Loader (warm-up, calibration, a pinned set pregrown to width+1 requests plus the reserve, width+1 workers). It then switches on `loader.backend` to read `backend.calibration` (main.zig:182-185), with a `@panic` for buffered targets.
- Verifier (both, partial): The central claim holds. `direct.Loader.calibration` is never read inside zml, and its only reader is dma-bench (examples/io:183), so Move 4's "dead Loader.calibration" would break dma-bench. Three corrections: - The buffered-target panic does not go away with the proposal as written.

**M1-bufferized-boundedarray** (low, bug; move R1). Bufferized(BoundedArray(Tensor, N)) has N buffers where visit sees len tensors, so a model holding one breaks load's pairing and Exe argument binding

- Where: `zml/zml/meta.zig:508-510`, `zml/zml/meta.zig:345-375` (+3).
- Problem: `visit` special-cases any struct that has `constSlice` and `slice` declarations and walks only `len` elements (meta.zig:508-510). MapRestrict rebuilds BoundedArray(Tensor, N) as a plain struct with only `buffer: [N]Buffer`.
- Verifier (value, confirmed): The central claim holds when traced through the code. 1. `visit` treats any struct that declares both `constSlice` and `slice` as a slice and walks only `v.slice()`, so `len` elements (meta.zig:508-510). `visitFlatStruct` has the same special case (meta.zig:646). 2. `MapRestrict.map` rebuilds such a struct with `@Struct`.

**M1-type-param-restated** (low, reshape; move R4). The model type is restated next to a typed pointer at about 190 call sites: bufferize, deinitBufferized, Buffer.deinitAll, load and llmd's loadInto all take `comptime T` that Zig infers

- Where: `zml/zml/mem.zig:19-29`, `zml/zml/buffer.zig:104-121` (+2).
- Problem: Calls read `bufferize(allocator, Model, &model)`, `load(Model, &model, &buffers, ...)` and `loadInto(Tensors, io, loader, &m.spec.tensors, &m.model_buffers, ...)`. Each `T` repeats the type of the pointer beside it.
- Verifier (both, partial): Zig does not infer comptime parameters. The fix is anytype plus @TypeOf, a pattern the codebase already uses (mem.zig:74, testing.zig:396). For bufferize, Loader.load, weights.loadInto and loadPacked, T restates the pointee type of the model argument and can go;

**M2-source-doc-contract** (low, doc; move R10, R16). Source's module doc claims a purity the type does not have and leaves out the lifetime and one-store rules a caller needs

- Where: `zml/zml/io/Source.zig:1-8`, `zml/zml/io/Source.zig:423` (+2).
- Problem: Source.zig:5-6 says a source 'touches neither the registry nor the store's bindings'. Yet every rewriting op allocates in its store's arena, four pub functions on Source write or remove bindings (bind, bindAs, rebind via unbind, bindProgram), and `place` reads the store's meshes.
- Verifier (both, partial): The module doc does not claim false purity. It says the ops touch neither the registry nor the bindings, and it names `bind` as the step that records a binding. That matches CTX.md:4973-4974, and the arena allocation is already documented (line 8, :278). What is missing is narrower: 1. It names only `bind`.

**M2-store-mints-bound-tensor** (low, contract; move R10). putSourcesNoClobber takes an id that each of its 4 callers mints on the line before. The clobber check, the error union and the errdefer guard a case that cannot happen.

- Where: `zml/zml/io/TensorStore.zig:343-352`, `zml/zml/io/TensorStore.zig:280-283` (+4).
- Problem: Each caller writes `const tensor: Tensor = .fromShape(s); store.putSourcesNoClobber(tensor.id, b) catch oom()/panic/unreachable; return tensor;`. Tensor.fromShape draws ids from a global atomic counter (tensor.zig:41-45), so the 'Id already has associated sources' panic (:346-348) cannot fire.
- Verifier (both, partial): The central claim holds: there are 4 call sites, each mints its id on the line before, and the clobber panic, errdefer and error union are all dead. Two corrections. First, bindNew's doc must keep LOADER_REVIEW Move 3's caveat (LOADER_REVIEW.md:515) that a hand-built Layout must come from Source.

## Appendix B: keeps, refuted and dropped

Keeps (reviewers judged these to earn their place):

- **A-keep-admission-without-knob**: Keep: admission is the loader's job and has no knob
- **B-keep-binding-seam**: Keep getSourcesById as the loader's only read of the store (rename it bindingOf(tensor)); Layout and Part can become loader-internal
- **C-keep-tensor-mirrored-ops**: Keep: the Source ops are pure, Tensor-mirrored values that panic like Tensor; data errors belong at View.maybeSource
- **C-keep-createTensor-front-door**: Keep: View.createTensor, maybeCreateTensor and createReplicatedTensor as the one-line front door, implemented over Source
- **E-keep-view-navigation**: Keep: TensorStore.View navigation plus one-line create calls. The model tree is the load manifest and every placement sits next to its key
- **E-keep-linear-view-split**: Keep: linear.View plus View.split plus per-shard fusion. Fused or separate checkpoints are handled in one expression, and the loader places each shard's rows with no program
- **F-keep-loadspec-contract**: Keep LoadSpec plus submit/await as the only contract between the front end and the backends
- **F-keep-direct-submit-all-or-nothing**: Keep direct submit's all-or-nothing publication: the front end never holds a half-published submission
- **F-keep-lattice-copy-kernel**: Keep Lattice.Cursor and copyRuns as the one strided-copy kernel both backends share
- **G-keep-shardgrid**: Keep ShardGrid's surface: a value type that answers every placement question by arithmetic
- **G-keep-credits**: Keep Credits as Move 2 left it: a closable stack of slot indices, the single bound on pinned memory
- **H-keep-vfs-is-io**: Keep: the VFS as a std.Io, so every reader opens hf://, s3://, gs:// and https:// paths without knowing about the VFS
- **H-keep-fromio**: Keep: VFS.fromIo as the loader's only route to the VFS that owns a handle
- **I-keep-tensor-entry-seam**: Keep `safetensors.Tensor` as the loader's whole view of the checkpoint: four fields, no format knowledge in the backends
- **I-keep-registerTensor-test-seam**: Keep `registerTensor` and `io/testing.writeSafetensors` as the test seam for building checkpoints
- **I-keep-exact-scatter-read**: Keep `readFilePositionalAllV`'s contract, the one exact scatter read on the direct loader's hot path; only move it
- **J-keep-borrow-until-retire**: Keep the borrow-until-retire model, with executables run on the owner's task
- **J-keep-infallible-binding**: Keep model binding infallible (View and Source panic on OOM and on missing names); errors belong to load
- **K-keep-store-per-submission**: Keep: the store is passed per submission, and Loader.init needs none
- **M1-keep-bufferized**: Keep Bufferized(T) as the type of loaded weights, and bufferize's walk by structure
- **M2-keep-source-store**: Keep Source.store. It is the explicit counterpart of Tensor's Compiler.current(): the ops' allocator, mesh resolver and bind target. None of B, C or E needs it removed.
- **M3-keep-classify**: Keep Scheme.classify a pure, platform-free function of the weight and scale shapes
- **M3-keep-experts-layout**: Keep moe.Backend.expertsLayout: the backend states the layout it reads, and experts() writes it at bind time
- **M3-keep-swizzle-source**: Keep swizzleBlockFp8Scale a Source expression the loader places, with no program

Refuted:

- **I-registry-interns-files**: A tensor names its file with a URI string copied into every tensor; the loader, examples/io and llmd's scale check each rebuild file identity from string compares and string maps The facts hold, the value does not: The facts mostly hold. The value does not, so the reshape fails. Facts confirmed: - `registerTensor` dupes `file_uri` into every tensor (safetensors.zig:817). `parseSafetensors` gets it from `realPath` once per shard (:866-868), and shards are parsed one after another (:1011).

Dropped by the merge, because they restate LOADER_REVIEW.md without an API angle:

- **J-init-deinit-error-paths**: Loader.init can leak running tasks, and Loader.deinit can panic after a device error (#35 and #36 still open). Restates #35 and #36 verbatim with the review's own fixes (group.async; log and leak on dmaUnmap failure), and has no API angle. Both are still open in the code (zml/zml/io/host_memory.zig:267, zml/zml/platform.zig:468), contrary to 'listed bugs were fixed'; the orchestrator should know that.
- **C-split-and-replace-after-bind**: Source.split and chunkExact have no production caller, while llmd's splitTensor re-implements split + bind and its callers re-place the result with Tensor.withPartitioning. Restates Move 6 ('splitTensor becomes Source.split + bind at its two call sites'). The only addition is keeping split/chunkExact as Tensor mirrors, which is a non-change.
- **D-testing-dup**: weights.testing re-implements zml's io/testing (pattern, rank-2 expectShards), and llmd tests hand-roll rank-3 shard checks because neither version is rank-N. Restates the review's 'Remaining small: Test oracle' (one rank-N expectShards exported, replacing llmd's copies and the hand-rolled shard loops). The only difference is exporting it as zml.io.testing rather than zml.testing.
