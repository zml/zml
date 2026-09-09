# CUDA Tile IR kernels in Zig

`kernels/cuda_tile` is a Zig DSL that emits NVIDIA's `cuda_tile` MLIR dialect
(CUDA Tile IR). It is the third instance of the `kernels/triton` /
`kernels/mosaic_tpu` pattern: a `Builder` that owns an MLIR module and appends
ops to a block stack, a `Value` with fluent arithmetic, and a
`zml.kernel.cuda_tile.Kernel` wrapper that turns a `run` function into a
`stablehlo.custom_call` XLA compiles with `tileiras`.

The dialect binding lives in `mlir/dialects/cuda_tile`: one function per op
(all 100 of them, in `Ops.td` order), every type through the dialect's C API,
every enum attribute through its C getter.

## A kernel

```zig
const zml = @import("zml");
const cut = zml.kernel.cuda_tile;

const Cfg = struct { n: i64 };

pub const AddOne = cut.Kernel(Cfg, .{
    .name = "add_one",
    .inputs = &.{"x"},
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *cut.Builder, cfg: Cfg) cut.FinishError!void {
    // Every parameter is a rank-0 tile<ptr<T>>; views are built inside.
    const a = try b.declareArgsOpts(.{
        .x = .{ .ptr = .f32 },
        .out = .{ .ptr = .f32 },
    }, .{ .hints = &.{.{ .arch = .sm_120, .num_worker_warps_per_cta = 4 }} });

    // pointer -> tensor_view -> partition_view; the view pads the ragged edge.
    const vx = b.partitionView(b.tensorView(a.x, &.{cfg.n}, &.{1}), &.{128}, .{});
    const vo = b.partitionView(b.tensorView(a.out, &.{cfg.n}, &.{1}), &.{128}, .{});

    const bid = b.tileBlockId();
    _ = b.store(b.load(vx, &.{bid.x}).add(1.0), vo, &.{bid.x});
}

// In a model:
const out = AddOne.call(.{ .x = x }, .{ .out = x.shape() }, .{
    .cfg = .{ .n = 1024 },
    .grid = .{ 8, 1, 1 },
}).out;
```

## What is different from the Triton DSL

* **Rank 0 is the scalar.** Every value is a `tile<...>`; `entry` parameters
  must be rank 0, so buffers arrive as `tile<ptr<T>>`.
* **Memory is a view stack.** `tensorView` → `partitionView` (or
  `stridedView`, `gatherScatterView`) → tile-space `load`/`store`. A partial
  edge tile reads the view's `padding` and its store is masked; there is no
  mask tile. The pointer-tile arm (`loadPtr`/`storePtr` with `offset`) exists
  for gathers but cannot use TMA.
* **Every load and store returns a token.** The dialect has no barrier or
  fence; `loadOpts`/`storeOpts` take and return tokens, `load`/`store` drop
  them.
* **Float ops carry a rounding mode.** `addf` defaults to `nearest_even`;
  `addfOpts` sets it. Integer ops carry none.
* **Loop bodies end with `continue`, `if` bodies with `yield`, and `loop`
  exits with `break`** — `ForScope`, `IfScope`, `LoopScope` do this for you.
* **No `num_warps`/`num_stages`.** Warp and CTA tuning is
  `optimization_hints` on the entry, per architecture (`Opts.hints`), and
  `allow_tma`/`latency` on view loads and stores (`LoadOpts.hints`).
* **Tile dims are powers of two**, at most 2^24 elements; `tileTy` and
  `partitionView` check this in Zig.
* **Some conversions are one-way.** `ftof` to `f8e8m0fnu` (the MX scale
  type) rounds toward zero by default (`CastOpts.rounding = .positive_inf`
  is the other legal choice); an integer reaches it through `f32`, which
  `cast`/`to` do for you. `ftoi` always truncates toward zero. `tanh` takes
  only `approx`/`full`. `Value.cdiv` is integer-only.
* **`i4` is a tile element type, not a pointee.** Declare a 4-bit buffer as
  `.i8` and `unpack` it in the kernel.
* **`atomicRedView` is relaxed-only** on an unpadded view and takes no mask;
  use the pointer atomics (`atomicRmw`, `atomicCas`) for masked updates.
* **Reductions over several tiles at once** (argmax, an online softmax's
  (max, sum)) go through `reduceMulti`/`scanMulti`.
* **A scalar argument is a pointer**: `.{ .n = .{ .ptr = .i32 } }` then
  `b.loadPtr(a.n)` gives the rank-0 tile; XLA passes device addresses only.

## The custom call

`ops.cudaTile` emits `__gpu$xla.gpu.cuda_tile` with a printed dictionary:
`name`, `kernel_type = "cuda_tile"`, `ir` (the textual module), `grid_x/y/z`,
optional `ir_version`. XLA parses the module, renames the entry, checks the
argument count against the operands + result leaves, scrubs locations,
serializes bytecode, runs `tileiras`, and launches over `grid` tile blocks with
a `(1,1,1)` block and no shared memory — cuda-tile's launch ABI. Every operand
and result is one raw device pointer in the default layout, operands first;
XLA rejects an entry whose parameters are not all `tile<ptr<T>>`.

To see what XLA received, run with `--xla_dump_to=<dir>`: it writes
`<instr>.cuda_tile.mlir` (post-rename) and `<instr>.tilebc` (the exact bytes
handed to `tileiras`).

## Finding what exists

There is one function per op and no hidden API, so grep is the index:

```sh
# every op of the dialect, in Ops.td order (115 constructors)
grep -n '^pub fn ' mlir/dialects/cuda_tile/cuda_tile.zig

# the types, enums and attribute helpers they take
grep -n '^pub const ' mlir/dialects/cuda_tile/cuda_tile.zig

# the DSL: Value's fluent methods (add, mul, lt, to, sum, reshape, ...)
awk '/^pub const Value = struct/,/^};/' kernels/cuda_tile/builder.zig | grep 'pub fn '

# the DSL: everything on Builder (views, load/store, math, casts, reductions,
# atomics, control flow) -- the `*Opts` variant of a call takes the options
awk '/^pub const Builder = struct/,0' kernels/cuda_tile/builder.zig | grep '    pub fn '

# what each option struct holds, and its defaults
grep -n '^pub const [A-Za-z]*Opts = struct' -A 8 kernels/cuda_tile/builder.zig

# for / if / loop scopes and their terminators
grep -n 'pub fn ' kernels/cuda_tile/control_flow.zig

# element types
grep -n 'pub const DType = enum' -A 30 kernels/cuda_tile/dtype.zig
```

Every op also has a test: `grep -n '^test "' kernels/cuda_tile/builder.zig`
lists them, and each one prints IR that round-trips through the verifier, so
they double as usage examples. A worked, runnable example (saxpy, softmax,
gemm, in-place relu) lives in `examples/cuda_tile`.
