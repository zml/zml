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

The dialect binding (`mlir/dialects/fly`) is generic: every `fly.*` op is
built by name and FlyDSL's own `inferReturnTypes` computes every layout. Zig
only prints and reads the dialect's type syntax; it never does layout algebra.

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

## The ABI

`Kernel.emit` produces

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
arguments. `Kernel.call` bridges each pointer to a row-major `!fly.memref`
view of the actual `zml.Shape`, so `K.args(b)` gives tensors.

Launch geometry lives in `CallOpts`: `grid`, `threads` (threads per block,
converted to wavefronts using the device's wave size — 64 on CDNA, 32 on
RDNA — so it must be a multiple of that), optional `waves_per_eu` (a hard
occupancy clamp),
`shared_mem_bytes` (only for `fly.get_dyn_shared`; static LDS from
`sharedArray` needs none) and `zeroed_args` (indices into inputs-then-outputs;
ignored inside HIP graphs, so zero what you need yourself).

## Layouts, tiles, coordinates

`layout.zig` holds comptime literals that print exactly like the dialect
(reading a type back goes through the dialect's own accessors in
`mlir/dialects/fly`, never through text):

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
  `.{ .buffer_copy_lds = 128 }`, or `.{ .text = "..." }` for anything else.
- `b.mmaAtom("!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>")`.
- `b.tiledCopyTV(atom, thr, val)`, `b.tiledCopy(atom, layout_tv, tile)`,
  `b.tiledCopyA/B/C(copy_atom, tiled_mma)`; `tc.getSlice(tid).partitionS/D`,
  `.retile`.
- `b.tiledMma(atom, fly.L(.{ 2, 2, 1 }, .{ 1, 2, 0 }), null)`;
  `tm.makeFragmentA/B/C(t)`, `tm.getSlice(tid).partitionA/B/C`.
- `b.copy(atom, src, dst, .{ .pred = p })`, `b.copyAtomCall(atom, src, dst)`,
  `b.gemm(atom, d, a, b, c)`.
- `b.bufferTensor(t)` is `fx.rocdl.make_buffer_tensor`: the same tensor over
  a CDNA raw-buffer descriptor, for bounds-checked `buffer_copy` atoms.
- Derived layouts (`tiled_mma.tile_size_mnk`, `tv_layout_A_tiled`, ...) come
  from `b.derivedStatic(.tiled_mma_tile_size_mnk, tm.value)`.

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
`b.setFastMath(true)` tags subsequent float ops `fastmath<fast>`.

Control flow uses `kernels/common`: `b.openFor(lo, hi, step, .{ inits })`
with `.yield(.{ ... })`, `b.openIf(cond)` with `.yieldThen(.{})`,
`b.openIfElse(cond, .{ types })` with `.yieldThen` / `.yieldElse`.

## Control flow and register tensors

The ROCm plugin lowers `scf` to `cf` **before** running the Fly passes on a
`gpu.func` entry, and Fly's register-alloca promotion only rewrites a
single-block body. So a kernel that has `openIf` / `openFor` must not hold
register tensors (`rmemTensor`, `makeFragmentLike`, `fill`) — use the SSA
atom calls instead: `b.copyAtomLoad(atom, src)` returns the loaded
`vector<NxT>`, `b.copyAtomStore(atom, vec, dst)` stores one. Kernels without
control flow (vectorAdd, tiledMma) can use register tensors freely. The
durable fix is on the XLA side (run `convert-scf-to-cf` after the Fly
passes, as FlyDSL's own pipeline does).

## Debugging

- A failing `inferReturnTypes` panics with the op name and every operand type
  (`fly.make_tiled_copy: cannot create operation ... operand 1: !fly.layout<...>`).
- `Kernel.emit` returns the module text; `bazel test //kernels/fly:test`
  builds the example kernels on any host, no GPU needed.
- On the device side, `--xla_dump_to=<dir>` dumps the module the plugin saw.

## Pins

The `@flydsl` commit in `third_party/flydsl/repo.bzl` must match the one the
ROCm PJRT plugin was built with. The plugin parses the text this dialect
prints, so an atom or type syntax added after the pin fails inside the
plugin, not here.
