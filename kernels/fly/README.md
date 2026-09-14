# ZML FlyDSL DSL — Author's Guide

A Zig builder for AMD's FlyDSL (`fly` / `fly_rocdl` MLIR dialects) that reads
like `flydsl.expr` Python while staying in pure Zig. Kernels are compiled and
launched by ZML's ROCm PJRT plugin through the `__gpu$xla.gpu.fly` custom
call; ZML never links a GPU compiler.

Two layers define the surface:

- **Low layer** — `kernels/fly/builder.zig`: `Builder`, `Value`, `TiledCopy`,
  `TiledMma`, the comptime layout literals in `layout.zig`, and `DType`.
  `Builder.open` takes an `*mlir.Context` from `zml.kernel.fly.newContext()`.
- **High layer** — `zml.kernel.fly.Kernel(Config, spec)`: names the kernel's
  inputs and outputs, hands `run` its arguments by name (`K.args(b)`), and
  `call`s it from a ZML model.

The dialect bindings construct typed MLIR operations. Layout-transforming
operations use FlyDSL's `inferReturnTypes`; static layout literals use typed
attribute constructors and accessors.

## Skeleton of a kernel

```zig
const zml = @import("zml");
const fly = zml.kernel.fly;

pub const Cfg = struct {
    thr: fly.Layout = fly.ordered(.{ 8, 16 }, .{ 1, 0 }), // 128 threads
    val: fly.Layout = fly.ordered(.{ 1, 4 }, .{ 0, 1 }),  // f32x4 along N
};

pub const VectorAdd = fly.Kernel(Cfg, .{
    .name = "vector_add",
    .inputs = &.{ "a", "b" },
    .outputs = &.{"c"},
    .run = run,
});

fn run(b: *fly.Builder, cfg: Cfg) fly.FinishError!void {
    const a = VectorAdd.args(b);           // .a/.b/.c: row-major !fly.memref views
    const shape = a.a.shapeStatic();
    const m = shape.at(0).leaf.s;
    const n = shape.at(1).leaf.s;

    const atom = b.copyAtom(.{ .universal = 128 }, .f32);
    const tc = b.tiledCopyTV(atom, cfg.thr, cfg.val);   // fx.make_tiled_copy_tv
    const tid = b.threadId(.x);
    const bx = b.blockId(.x);
    const by = b.blockId(.y);

    const tile = tc.tileMN();
    const gA = a.a.flatDivide(tile).slice(.{ null, null, bx, by });   // A[None, None, bx, by]
    const gB = a.b.flatDivide(tile).slice(.{ null, null, bx, by });
    const gC = a.c.flatDivide(tile).slice(.{ null, null, bx, by });
    const cC = b.identity(.{ m, n }).flatDivide(tile).slice(.{ null, null, bx, by });

    const thr = tc.getSlice(tid);
    const tgA = thr.partitionS(gA);
    const tgB = thr.partitionS(gB);
    const tgC = thr.partitionD(gC);
    const tcC = thr.partitionS(cC).slice(.{ .{ 0, null }, null, null });

    const rA = tgA.makeFragmentLike(null);
    const rB = tgB.makeFragmentLike(null);
    const rC = tgC.makeFragmentLike(null);
    const pC = tcC.makeFragmentLike(.i1);
    for (0..@intCast(pC.sizeStatic())) |i| pC.set(i, tcC.at(i).elemLess(.{ m, n }));

    b.copy(atom, tgA, rA, .{ .pred = pC });
    b.copy(atom, tgB, rB, .{ .pred = pC });
    rC.store(rA.load().add(rB.load()));
    b.copy(atom, rC, tgC, .{ .pred = pC });
}

// In a model:
const c = VectorAdd.call(.{ .a = a, .b = bt }, .{ .c = a.shape() }, .{
    .cfg = .{},
    .grid = .{ grid_m, grid_n, 1 },
    .threads = 128,
}).c;
```

Complete vector-add and tiled-MMA kernels are the tests in `builder.zig`
and `zml/kernel.zig`.

Model kernels live beside their consumers:

- [MoE routing](../../zml/moe/fly_kernels/moe.zig) and
  [MXFP4 projections](../../zml/moe/fly_kernels/mxfp4.zig).
- [Sparse MLA](../../zml/attention/fly_kernels/sparse_mla.zig).

Fly is a separate backend. The [MoE selector](../../zml/moe/moe.zig) and
[sparse-MLA selector](../../zml/attention/sparse_mla.zig) select by platform,
architecture and dtype. The current native kernels target gfx942; their
callers check shape support before routing or launching kernels and use
Triton for unsupported contracts.

## The ABI

`Kernel.emit(allocator, cfg, shapes)` takes shapes in input-then-output order
and produces

```mlir
module attributes {gpu.container_module} {
  gpu.module @zml_fly_kernels {
    gpu.func @name(%a: !fly.ptr<f32, global>, ..., %c: !fly.ptr<f32, global>) kernel { ... gpu.return }
  }
}
```

with one `!fly.ptr` per input, then per output. The plugin checks only the
argument **count**, so the pointer element type is on you: `DType.storageElem`
maps predicates to `i8` because that is how XLA stores them. Static DSL values
(layouts, tiles, atoms) must be built inside the body, never passed as
arguments. `Kernel.call` bridges each non-scalar argument to a row-major
`!fly.memref` view of its `zml.Shape`. Rank-zero arguments remain pointers;
use `ptrLoad` and `ptrStore` for their scalar values.

Launch geometry lives in `CallOpts`: `grid`, `threads` (threads per block,
converted to wavefronts using the device's wave size — 64 on CDNA, 32 on
RDNA — so it must be a multiple of that), optional `waves_per_eu` (sets the
LLVM `amdgpu-waves-per-eu` attribute to `N, N` when positive),
`shared_mem_bytes` (only for `fly.get_dyn_shared`; static LDS from
`sharedArray` needs none) and `zeroed_args` (indices into inputs-then-outputs;
ignored inside HIP graphs, so zero what you need yourself).

`output_operand_aliases` maps output names to input names for buffers that
the kernel reads and writes. Aliasing preserves the input-then-output ABI;
it does not initialize output memory.

## Layouts, tiles, coordinates

`layout.zig` holds comptime literals. `toAttr` turns one into the matching
fly attribute through the dialect's own constructors, and reading a type back
goes through its accessors — neither direction goes through text:

| Zig | MLIR |
|---|---|
| `fly.it(.{ .{ 2, 4 }, 8 })` | `!fly.int_tuple<((2,4),8)>` |
| `fly.L(.{ 4, 8 }, .{ 1, 4 })` | `!fly.layout<(4,8):(1,4)>` |
| `fly.rowMajor(.{ 4, 8 })` / `fly.colMajor` | `(4,8):(8,1)` / `(4,8):(1,4)` |
| `fly.ordered(.{ 8, 16 }, .{ 1, 0 })` | `fx.make_ordered_layout((8,16),(1,0))`, folded |
| `fly.tile(.{ 128, 64 })`, `fly.tile(.{ null, 8 })` | `!fly.tile<[128\|64]>`, `[*\|8]` |

`b.static(literal)` materializes one as `fly.static`. Wherever the Python
takes a tuple, the Zig takes a tuple literal: `t.flatDivide(.{ 64, 8 })` is a
tile, `t.slice(.{ null, bid })` is `t[None, bid]`, `b.intTuple(.{ m, tid })`
mixes static ints, `null` (`*`) and dynamic `Value`s (`?` leaves, passed as
operands). `v.shapeStatic()` / `v.sizeStatic()` / `v.elemDType()` read the
inferred type back on the host; `v.emitShape()` / `v.emitSize()` emit ops.

## Atoms and tiled ops

- `b.copyAtom(.{ .universal = 128 }, .f32)`, `.{ .buffer_copy = 32 }`,
  `.{ .buffer_copy_lds = 128 }`.
- `b.mmaAtom(mma_op)` takes an MLIR type. Generated MMA type constructors
  return a typed wrapper; handle their error and pass the wrapper's
  `.type_()` result.
- `b.tiledCopyTV(atom, thr, val)`, `b.tiledCopy(atom, layout_tv, tile)`,
  `b.tiledCopyA/B/C(copy_atom, tiled_mma)`; `tc.getSlice(tid).partitionS/D`,
  `.retile`.
- `b.tiledMma(atom, fly.L(.{ 2, 2, 1 }, .{ 1, 2, 0 }), null)`;
  `tm.makeFragmentA/B/C(t)`, `tm.getSlice(tid).partitionA/B/C`.
- `b.copy(atom, src, dst, .{ .pred = p })`, `b.copyAtomCall(atom, src, dst)`,
  `b.gemm(atom, d, a, b, c)`.
- `b.mmaAtomCall(atom, a, b, c)` takes SSA operands and returns the updated
  accumulator. Operand vectors must match the atom's per-lane fragment
  types and layout; this call performs no tiled partitioning.
- `b.bufferTensor(t)` is `fx.rocdl.make_buffer_tensor`: the same tensor over
  a CDNA raw-buffer descriptor, for bounds-checked `buffer_copy` atoms.
- Derived layouts come off the type: `fly.expect(tm.value.type_(), .tiled_mma)`
  then `.getTileSizeMNK()`, `.getTiledThrValLayoutA()`, ...

## Memory and scalars

`t.load()` / `t.store(vec)` move a whole register tensor as a `vector<NxT>`;
`t.at(coord)` / `t.set(coord, v)` are element accesses (an int tuple for a
coordinate tensor); `t.fill(0)` splats. `b.rmemTensor(layout, dtype)` is
`fx.make_rmem_tensor`; `b.sharedArray(.f32, n, 16)` is a static LDS array
(`fx.SharedAllocator().allocate(fx.Array[...])`).

Scalars and vectors: `add/sub/mul/div/rem` (a scalar broadcasts against a
vector), `to(dtype)`, `cmp(.lt, x)`, `cond.select(a, b)`, `v.reduce(.add)`,
`v.shuffleXor(off, 64)`, `b.rsqrt(x)`, `b.constant(.f32, 1.0)`,
`b.threadId(.x)` / `b.blockId(.x)` (already `i32`), `b.barrier()`.

Integer types are signless in MLIR. `div`, `rem`, `cmp` and `to` use signed
integer semantics, except that `to` treats predicates as unsigned.
`divUnsigned`, `remUnsigned`, `cmpUnsigned` and `toUnsigned` make unsigned
operations explicit. `bitAnd`, `bitOr`, `bitXor`, `shl` and `shrU` operate
on integer bits.

`splat` creates a vector, including when its length is one. Numeric casts
preserve vector shape. `bitcast` preserves total bits and adjusts vector
length to the destination element width; scalar bitcasts require equal
widths. Use `extract` to obtain a scalar, `insert` to replace an element and
`shuffleVector` to select elements from two vectors. Lane `shuffle`
supports `xor`, `up`, `down` and `idx` modes with a literal or an `i32`
`Value` offset.

`maximum` propagates NaNs; `maxNum` selects the numeric operand when only
one operand is NaN. `neg`, `Builder.log` and `Builder.fma` provide floating
operations. Fast math is off by default; `setFastMath(true)` tags subsequent
float operations `fastmath<fast>` and permits relaxed numerical semantics.

`Builder.perm` emits AMD byte permutation. `Builder.cvtPkF32Fp8` decodes
two FP8 bytes from the low or high half of an `i32` word using the CDNA3
packed conversion instruction. Architecture selection belongs in the
backend before these operations are emitted.

`Builder.ptrAtomicAdd` adds an `i32` counter and returns its old value.
Use `workgroup` scope for LDS pointers and `agent` scope for global
pointers. The ordering is monotonic; callers provide barriers or kernel
dependencies before consuming completed counters.

Control flow uses `kernels/common`: `b.openFor(lo, hi, step, .{ inits })`
with `.yield(.{ ... })`, `b.openIf(cond)` with `.yieldThen(.{})`,
`b.openIfElse(cond, .{ types })` with `.yieldThen` / `.yieldElse`.
`openWhile` has separate condition and body regions: `before_carried`
feeds `yieldBefore`, `after_carried` feeds `yieldAfter`, and `results`
contains the values returned when the condition becomes false.

## Control flow and register tensors

The published ROCm plugin lowers `scf` to `cf` **before** running the Fly
passes on a native `gpu.func` entry, and Fly's register-alloca promotion
only rewrites a single-block body. Kernels with runtime control flow must
keep their register state in SSA values:

- `copyAtomLoad` returns a vector and `copyAtomStore` stores a vector.
- `mmaAtomCall` returns an accumulator without allocating register memory.
- Pass accumulators through `openFor` initial values, `carried`, `yield`
  and `results`. Branches and while loops also return SSA values through
  their scope objects.

The MoE and sparse-MLA kernels linked above use this path with the published
runtime. Straight-line kernels can still use `rmemTensor`,
`makeFragmentLike`, `fill` and `gemm`.

## Debugging

- A failing `inferReturnTypes` panics with the op name and every operand type
  (`fly.make_tiled_copy: cannot create operation ... operand 1: !fly.layout<...>`).
- `Kernel.emit` returns the module text; `bazel test //kernels/fly:test`
  builds the example kernels on any host, no GPU needed.
- On the device side, `--xla_dump_to=<dir>` dumps the module the plugin saw.
