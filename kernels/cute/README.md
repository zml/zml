# CuTe DSL kernels in Zig

`kernels/cute` is a Zig front end for NVIDIA's CuTe DSL, shaped after the
Python `cutlass.cute` API: a kernel takes tensors, indexes them with
coordinates, and reads `thread_idx()` and friends from `cute.arch`. It is the
fourth instance of the `kernels/triton` / `kernels/mosaic_tpu` /
`kernels/cuda_tile` pattern: a `Builder` that owns an MLIR module and appends
ops to a block stack, a `Value` with fluent scalar arithmetic, and a
`zml.kernel.cute.Kernel` wrapper that turns a `run` function into a
`stablehlo.custom_call` XLA compiles in process with `libcute_compiler.so`.

The dialect binding lives in `mlir/dialects/cute_ir` (`cute` and
`cute_nvgpu`, types through the dialect's C API).

## A kernel

The Python notebook's `naive_elementwise_add_kernel`, in Zig:

```zig
const zml = @import("zml");
const cute = zml.kernel.cute;

const Cfg = struct { m: i64, n: i64 };

pub const NaiveElementwiseAdd = cute.Kernel(Cfg, .{
    .name = "naive_elementwise_add_kernel",
    .inputs = &.{ "gA", "gB" },
    .outputs = &.{"gC"},
    .run = run,
});

fn run(b: *cute.Builder, cfg: Cfg) cute.FinishError!void {
    // A `tensor` argument is XLA's device pointer composed with a static
    // layout; the Python kernel takes `cute.Tensor` the same way.
    const t = try b.declareArgs(.{
        .gA = .{ .tensor = .{ .dtype = .f32, .shape = &.{ cfg.m, cfg.n } } },
        .gB = .{ .tensor = .{ .dtype = .f32, .shape = &.{ cfg.m, cfg.n } } },
        .gC = .{ .tensor = .{ .dtype = .f32, .shape = &.{ cfg.m, cfg.n } } },
    });

    const tidx = b.threadIdx().x;
    const bidx = b.blockIdx().x;
    const bdim = b.blockDim().x;
    const thread_idx = bidx.mul(bdim).add(tidx);

    const n = t.gA.dim(1);
    const ni = thread_idx.rem(n);
    const mi = thread_idx.div(n);

    // gC[mi, ni] = gA[mi, ni] + gB[mi, ni]
    t.gC.set(.{ mi, ni }, t.gA.get(.{ mi, ni }).add(t.gB.get(.{ mi, ni })));
}

// In a model (`kernel.launch(grid=..., block=...)`):
const c = NaiveElementwiseAdd.call(.{ .gA = a, .gB = b }, .{ .gC = a.shape() }, .{
    .cfg = .{ .m = 64, .n = 128 },
    .grid = .{ 32, 1, 1 },
    .block = .{ 256, 1, 1 },
}).gC;
```

## Python to Zig

| `cutlass.cute`                             | `kernels/cute`                                  |
|--------------------------------------------|-------------------------------------------------|
| `gA: cute.Tensor` kernel argument          | `.gA = .{ .tensor = .{ .dtype, .shape } }`      |
| `gA[mi, ni]` / `gC[mi, ni] = v`            | `gA.get(.{ mi, ni })` / `gC.set(.{ mi, ni }, v)`|
| `cute.make_layout(shape, stride=...)`      | `b.makeLayout(shape, .{ .stride = ... })`       |
| `cute.make_tensor(ptr, layout)`            | `b.makeTensor(ptr, layout)`                     |
| `cute.arch.thread_idx()` → `(x, y, z)`     | `b.threadIdx().x`                               |
| `block_idx()`, `block_dim()`, `grid_dim()` | `blockIdx()`, `blockDim()`, `gridDim()`         |
| `cute.arch.alloc_smem(T, n, align)`        | `b.allocSmem(.f32, n, align)`                   |
| `cute.arch.sync_threads()`                 | `b.syncThreads()`                               |
| `cute.arch.shuffle_sync_bfly(v, lane)`     | `b.shuffleXor(v, lane)`                         |
| `cute.arch.griddepcontrol_launch_dependents()` | `b.launchDependents()`                      |
| `x.to(Float32)`                            | `x.to(.f32)`                                    |
| `if cond:` / `for i in range(...)`         | `b.openIf(cond)` / `b.openFor(lb, ub, step, .{})` |
| `cpasync.make_tiled_tma_atom(...)`         | `b.makeTiledTmaLoadAtom(...)` / `...Typed(...)` |
| `cpasync.tma_partition(...)`               | `b.tmaPartition(...)`                           |
| `cute.copy(atom, src, dst)`                | `b.copy(atom, src, dst, pred)`                  |
| `cute.make_tiled_mma(atom)`                | `b.makeTiledMma(atom, result_type)`             |
| `tiled_mma.make_fragment_A/B/C(...)`       | `b.mmaMakeFragment(...)`                        |
| `cute.gemm(...)`                           | `b.gemm(...)`                                   |
| `cute.arch.alloc_tmem(...)`                | `b.allocTmem(...)`                              |
| `cute.arch.retrieve_tmem_ptr(...)`         | `b.retrieveTmemPtr(...)`                        |

* **Layouts are static.** `makeLayout` takes Zig slices and emits one
  `cute.static`; the tensor keeps its shape and stride in Zig, so `dim`,
  `size` and coordinates need no MLIR queries. Coordinates are tuples of
  `Value`s and ints; ints stay static in the `!cute.coord` type.
* **Stride defaults differ by origin.** `makeLayout` defaults to CuTe's
  compact left-most stride like `cute.make_layout`; a `tensor` argument
  defaults to XLA's row-major layout, which is what the buffer holds.
* **Scalars are `i32` and `f32` by default**, like `Int32`/`Float32`;
  `cst(dtype, v)` picks another type and `coerce` lifts literals to the other
  operand's type.
* **`nvvm` is not linked into ZML.** Thread indices and barriers are emitted
  as unregistered ops in generic form; the compiler has the dialect.
* **The module is just the kernel.** `finish` prints `module { func.func
  @name(...) {...} }`; the CuTe compiler makes every public function a kernel
  entry, so there is no `gpu.module` or host launch function to write.
* **Explicit result types:** Zig covers local tiling, TMA partitioning, tiled
  copies, tensor-memory fragments, and SM100 block-scaled MMA. CuTe derives
  their layouts in the compiler, so callers spell out the resulting CuTe type
  with `View` instead of relying on Python's dynamic type objects.
* **Not covered yet:** dynamic tensor shapes and automatic CuTe layout algebra.
* **Typed variants.** Operations whose result type CuTe derives exist in two
  forms: one takes the result type as MLIR text (`localTile`, `slice`,
  `tmaPartition`, ...), the `...Typed` one takes an `*mlir.Type` assembled
  with `layoutSpec`, `memrefType` and `coordTensorType`.
* **Architecture sections.** `Builder` groups TMA (`cute.nvgpu.cpasync`,
  SM90+) and tcgen05 / tensor memory (`cute.nvgpu.tcgen05`, SM100) in their
  own sections; mbarriers, fences and bulk-copy groups sit with `cute.arch`.
  Each doc comment names its minimum architecture when it is above SM80.

## Programs with a host launch

Kernels that need launch-time CuTe objects (TMA descriptors, tiled MMA atoms,
scheduler parameters) use `zml.kernel.cute.Program` instead of `Kernel`. Its
`run` callback emits one `gpu.module` with the `cuda.kernel`s
(`b.beginFunction(name, .cuda_kernel)`), then a public host `func.func`
(`b.beginFunction(name, .host)`) that builds those objects from the XLA
buffers and calls `b.launchEx`. The custom call carries no grid or block:
XLA runs the host function with the real buffers and launches (or records in
a command buffer) whatever it launched. `CallOpts.scalars` are passed to the
host function after the buffers. `zml/moe/cute_kernels/persistent_mxfp4.zig`
is the reference: a warp-specialized SM100 block-scaled GEMM whose host
function builds four TMA tensor maps and a PDL launch.

## The custom call

`ops.cute` emits `__gpu$xla.gpu.cute` with a printed dictionary: `name`,
`kernel_type = "cute"`, `ir` (the textual module), `grid` and `block`
(3-element arrays), optional `zeroed_outputs`. XLA compiles it in process
(`lib/libcute_compiler.so`, which pulls in `lib/cutlass_ir.so`, both from the
sandbox pointed at by `--xla_gpu_cuda_data_dir`), takes the kernel's
dynamic shared memory requirement from the compiler's metadata, and launches
the cubin over `grid` × `block`. Every operand and result is one raw device
pointer in the default layout, operands first; the count is not checked
against the kernel signature, so `inputs`/`outputs` must match `declareArgs`.
A module that carries its own `gpu.module` and host launch still works;
it then owns the launch configuration and `grid`/`block` must be left out.

To see what XLA received, run with `--xla_dump_to=<dir>`: it writes
`<instr>.cute.mlir`, `<instr>.cute.json` and `<instr>.cubin`.

## Finding what exists

```sh
# the DSL: Value's fluent methods (add, mul, lt, to, exp, ...)
awk '/^pub const Value = struct/,/^};/' kernels/cute/builder.zig | grep 'pub fn '

# the DSL: everything on Tensor and Builder
awk '/^pub const Tensor = struct/,/^};/' kernels/cute/builder.zig | grep 'pub fn '
awk '/^pub const Builder = struct/,0' kernels/cute/builder.zig | grep '    pub fn '

# every bound op of the dialect
grep -n '^pub fn ' mlir/dialects/cute_ir/cute_ir.zig mlir/dialects/cute_ir/cute_nvgpu.zig
```

`grep -n '^test "' kernels/cute/builder.zig` lists the tests; each prints a
kernel that round-trips through the verifier.
