# Porting a FlyDSL Python kernel to Zig

The Zig DSL emits the same `fly` ops the Python does, so a port is a
statement-by-statement translation. Work from the pinned FlyDSL tree
(`third_party/flydsl/repo.bzl`, read with `git show <pin>:<path>`), not from
FlyDSL HEAD: the plugin only parses what the pin prints.

## Cheat sheet

| Python (`flydsl.expr`) | Zig |
|---|---|
| `@flyc.kernel def k(A: Tensor, ...)` | `fly.Kernel(Cfg, .{ .inputs = &.{"a"}, .outputs = &.{"c"}, .run = run })`; `const a = K.args(b)` |
| kernel arg `tiled_copy: fx.TiledCopy` | build it inside `run` (static values cannot be arguments) |
| `fx.thread_idx.x`, `fx.block_idx.x` | `b.threadId(.x)`, `b.blockId(.x)` (`i32`) |
| `A.shape.unpack()` | `a.a.shapeStatic().at(i).leaf.s` |
| `fx.make_layout((8,16),(16,1))` | `fly.L(.{ 8, 16 }, .{ 16, 1 })` (comptime) or `b.static(layout)` |
| `fx.make_ordered_layout((8,16),(1,0))` | `fly.ordered(.{ 8, 16 }, .{ 1, 0 })` |
| `fx.make_tile(64, 8)` / tuple divisor `(64, 8)` | `fly.tile(.{ 64, 8 })` / `.{ 64, 8 }` |
| `fx.make_int_tuple((m, tid))`, coords | `b.intTuple(.{ m, tid })` (`Value`s become `?` leaves) |
| `t[None, None, bx, by]` | `t.slice(.{ null, null, bx, by })` |
| `t[(0, None), None, None]` | `t.slice(.{ .{ 0, null }, null, null })` |
| `t[i]` (no `None`) / `t[i] = v` | `t.at(i)` / `t.set(i, v)` |
| `fx.flat_divide(t, tile)` etc. | `t.flatDivide(tile)`, `.logicalDivide`, `.zippedDivide`, `.tiledDivide` |
| `fx.raked_product(a, b)` etc. | `a.rakedProduct(b)`, `.logicalProduct`, ... |
| `fx.right_inverse(l)`, `fx.composition(l, r)` | `l.rightInverse()`, `l.composition(r)` |
| `fx.get_shape(l)`, `fx.size(l)` (emitted) | `l.emitShape()`, `l.emitSize()` |
| `fx.size(t.shape).unpack()` (host int) | `t.sizeStatic()` |
| `fx.select(tup, [0])`, `fx.get(tup, [1])` | `tup.selectModes(&.{0})`, `tup.get(&.{1})` |
| `fx.make_view((0,0), fx.make_identity_layout((M,N)))` | `b.identity(.{ m, n })` |
| `fx.make_view(ptr, layout)` | `ptr.view(layout)` |
| `fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)` | `b.copyAtom(.{ .universal = 128 }, .f32)` |
| `fx.rocdl.BufferCopy32b()` / `BufferCopyLDS128b()` | `.{ .buffer_copy = 32 }` / `.{ .buffer_copy_lds = 128 }` |
| `fx.make_mma_atom(fx.rocdl.MFMA(16,16,4,fx.Float32))` | `b.mmaAtom("!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>")` |
| `fx.make_tiled_mma(atom, layout)` | `b.tiledMma(atom, layout, null)` |
| `fx.make_tiled_copy_tv(atom, thr, val)` | `b.tiledCopyTV(atom, thr, val)` |
| `fx.make_tiled_copy_A(atom, tiled_mma)` | `b.tiledCopyA(atom, tiled_mma)` |
| `tiled_copy.tile_mn` | `tc.tileMN()` |
| `tiled_mma.tile_size_mnk` etc. | `b.derivedStatic(.tiled_mma_tile_size_mnk, tm.value)` |
| `tiled_copy.get_slice(tid).partition_S(t)` | `tc.getSlice(tid).partitionS(t)` |
| `thr_copy.retile(t)` | `thr.retile(t)` |
| `tiled_mma.make_fragment_A(t)` | `tm.makeFragmentA(t)` |
| `fx.make_fragment_like(t[, dtype=fx.Boolean])` | `t.makeFragmentLike(null)` / `t.makeFragmentLike(.i1)` |
| `fx.make_rmem_tensor(8, fx.Float16)` | `b.rmemTensor(fly.L(8, 1), .f16)` |
| `fx.copy(atom, src, dst, pred=p)` | `b.copy(atom, src, dst, .{ .pred = p })` |
| `fx.copy_atom_call(atom, src, dst)` | `b.copyAtomCall(atom, src, dst)` |
| `fx.gemm(atom, d, a, b, c)` | `b.gemm(atom, d, a, bb, c)` |
| `t.load()`, `t.store(v)`, `t.fill(0)` | same names |
| `fx.elem_less(coord, (M, N))` | `coord.elemLess(.{ m, n })` |
| `fx.rocdl.make_buffer_tensor(t)` | `b.bufferTensor(t)` |
| `fx.SharedAllocator().allocate(fx.Array[f32, n, 16])` | `b.sharedArray(.f32, n, 16)` (returns an `n:1` tensor) |
| `gpu.barrier()` | `b.barrier()` |
| `x.shuffle_xor(off, 64)` | `x.shuffleXor(off, 64)` |
| `v.reduce(ReductionOp.ADD)` | `v.reduce(.add)` |
| `v.to(fx.Float32)` | `v.to(.f32)` |
| `x * rrms` (vector × scalar) | `x.mul(rrms)` |
| `fmath.rsqrt(x, fastmath=fast)` | `b.setFastMath(true); b.rsqrt(x)` |
| `lane < RED_SLOTS`, `cond.select(a, b)` | `lane.cmp(.lt, n)`, `cond.select(a, b)` |
| `if tid == 0: ...` | `var s = b.openIf(tid.cmp(.eq, 0)); ...; s.yieldThen(.{});` |
| `for i in range_constexpr(n)` | a plain Zig `for (0..n)` (unrolled at emit time) |
| `for i in range(lo, hi)` with carried values | `b.openFor(lo, hi, 1, .{ init })` + `.yield(.{ ... })` |
| `launch(grid=..., block=(256,1,1))` | `K.call(inputs, outputs, .{ .grid = ..., .threads = 256 })` |

## Method

1. Read the kernel and its helpers at the pin; note every `fx.*` primitive.
2. Write the Zig `run` top to bottom. Keep the same op order so the emitted
   IR lines up with the Python module when diffing.
3. `bazel test //kernels/fly:test` (or a `K.emit` test): the module must
   verify and re-parse. A panic from `fly.<op>: cannot create operation`
   lists the operand types; compare them with the Python trace.
4. Run on the ROCm box against a host reference
   (`bazel test //zml:test --@zml//platforms:rocm=true ...`).

## Things that differ

- Python config (`N`, `dtype_str`, `const_expr(...)` branches) becomes the
  kernel `Config`; branch on it with plain `if` in `run`.
- Python lists of vectors across an unrolled loop are Zig arrays of `Value`.
- Python `Int32(x)` / `Float32(x)` constants are `b.constant(.i32, x)`; most
  arithmetic methods also take a literal directly (`tid.add(256)`).
- Cached scalars from LDS are read with `t.at(i)`; there is no implicit load.
- bf16 outputs use `arith.truncf`; the Python's manual round-to-nearest bit
  trick for pre-gfx950 parts is not reproduced.
