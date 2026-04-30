# ZML Mosaic TPU DSL — Author's Guide

A Zig builder for Mosaic TPU IR (the `tpu` dialect shipped by JAX) that tries
to read like `pallas.tpu` Python while staying in pure Zig. Used by
`examples/mosaic_gdn` and by anything else that wants to emit a TPU kernel
op-for-op against what `pallas_call` produces.

The DSL talks to the `tpu` dialect through `mlir.Operation.make(ctx, "tpu.<op>", …)`.
The dialect must be registered into the MLIR context — see
`mlir/dialects/mosaic_tpu`. Until the JAX repo is wired into `MODULE.bazel`, the
whole stack is `manual`-tagged in Bazel.

Two layers define the surface:

- **Low layer** — `kernels/mosaic_tpu/builder.zig`: the IR builder primitives.
  `Builder.open(allocator, ctx, name)` → `b.declareArgsOpts(spec, results, opts)`
  → body via `b.*` helpers → `b.finishOpts(results, .{ .canonicalize = true })`
  for the final IR string. (`Builder.buildOpts(...)` is a convenience that
  bundles `open + declareArgs` into one call and returns a heap-allocated
  `*Built(Spec)`; reach for it in escape-hatch code that drives the
  lifecycle manually.)
- **High layer** — `zml.kernel.mosaic_tpu.Kernel(Config, spec)`: the
  declarative kernel form, mirroring the Triton DSL. Bundles a config
  type, a typed spec literal (name + named `inputs` / `outputs` + a typed
  `run` function pointer), and produces a kernel type. Exposes
  `.emit(allocator, ctx, cfg)` (IR string) and
  `.call(inputs, outputs, opts)` (IR string + `stablehlo.custom_call`
  targeting `tpu_custom_call` in one shot — the preferred form for
  production model code).

## Table of contents

1. [Skeleton of a kernel](#skeleton-of-a-kernel)
2. [Argument specs](#argument-specs)
3. [Pallas `pallas_call` metadata](#pallas-pallas_call-metadata)
4. [The `Value` type](#the-value-type)
5. [Fluent methods on `Value`](#fluent-methods-on-value)
6. [Polymorphic scalars](#polymorphic-scalars)
7. [Constants and constant hoisting](#constants-and-constant-hoisting)
8. [Loads and stores](#loads-and-stores)
9. [Vector and shape helpers](#vector-and-shape-helpers)
10. [Reductions, scans, sorts](#reductions-scans-sorts)
11. [Compute primitives](#compute-primitives)
12. [Memref ops](#memref-ops)
13. [DMAs, semaphores, and barriers](#dmas-semaphores-and-barriers)
14. [Control flow](#control-flow)
15. [Escape hatches](#escape-hatches)
16. [Pallas Python ↔ Zig DSL cheat-sheet](#pallas-python--zig-dsl-cheat-sheet)
17. [Pitfalls](#pitfalls)

---

## Skeleton of a kernel

```zig
const mlir = @import("mlir");
const mtt = @import("zml").kernel.mosaic_tpu;

pub const MyKernel = struct {
    pub const Cfg = struct { /* shape/dtype/grid knobs */ };
    pub const Kernel = mtt.Kernel(Cfg, .{
        .name = "my_kernel",
        .inputs = &.{ "x", "cu_seqlens" },
        .outputs = &.{"y"},
        .run = run,
    });
    fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
        _ = cfg;
        const a = try b.declareArgsOpts(.{
            // Grid program-ids — i32 scalars. `pl.program_id(0)`, `pl.program_id(1)`.
            .i_b = .{ .scalar = .i32 },

            // Inputs: VMEM memrefs.
            .x = .{ .ref = .{ .shape = &.{ 128, 256 }, .dtype = .bf16 } },

            // SMEM scalar-prefetch.
            .cu_seqlens = .{ .ref = .{ .shape = &.{4}, .dtype = .i32, .memory_space = .smem } },

            // Outputs.
            .y = .{ .ref = .{ .shape = &.{ 128, 256 }, .dtype = .bf16, .role = .output } },
        }, &.{}, .{
            .dimension_semantics = &.{.arbitrary},
            .iteration_bounds = &.{ /* grid */ 4 },
            .pallas_window_params = true,
        });
        _ = a;

        // ... emit body via b.* helpers; reach args through `a.<name>`.
    }
};
```

A kernel is a `struct {…}` wrapper bundling three decls:

1. **`pub const Cfg`** — config type, an inline `struct { ... }` of
   comptime/runtime parameters.
2. **`pub const Kernel = mtt.Kernel(Cfg, .{ .name, .inputs, .outputs, .run })`** —
   the spec literal that turns into a kernel type. `inputs` / `outputs` are
   slices of `[:0]const u8` field names; the factory generates `Inputs`
   (struct of `Tensor`), `Outputs` (struct of `Shape`), and `Results`
   (struct of `Tensor`) types from those names. The `run` field is a typed
   function pointer — wrong signature is a compile error here.
3. **`fn run(b, cfg)`** — the kernel body, mirroring the Triton form. First
   call `try b.declareArgsOpts(.{…}, &.{}, opts)` to create the `func.func`
   op and bind a `NamedArgs(Spec)` struct of `Value`s. The factory has
   already opened the builder; it'll terminate, verify, run
   `canonicalize,cse,canonicalize`, and serialize after `run` returns —
   no manual `b.finish(...)` call.

The names in `spec.inputs` / `spec.outputs` are the **call-site** names
(used to build the `Inputs` / `Outputs` structs) and don't have to match
the names in `b.declareArgsOpts(...)` (which are the **MLIR-level** arg
names — what shows up in the `func.func` signature). By convention they
often do — pick whatever reads best at the boundary.

The arg spec inside `b.declareArgsOpts(...)` is a struct literal — its
variant *tags* (`.scalar`, `.iter`, `.ref`, `.sem`) are comptime-known per
call site, but the *payload values* (shapes, dtypes, memory spaces,
windows) can be runtime values from a config.

> **Why the wrapper struct.** `Cfg` has to be a *named* type so the typed
> `run: *const fn (*Builder, Cfg) FinishError!void` field in the spec can
> reference it. Wrapping the kernel in `pub const MyKernel = struct {...}`
> keeps `Cfg`, `Kernel`, and `run` scoped together and exposes them all to
> callers (`MyKernel.Cfg`, `MyKernel.Kernel`, plus everything the inner
> Kernel re-exports — `MyKernel.Kernel.Config`, `.Inputs`, `.Outputs`,
> `.Results`, `.CallOpts`).
>
> Single-kernel files (e.g. `zml/attention/mosaic_tpu_kernels/ragged_attention.zig`)
> can skip the wrapper and put `pub const Cfg` / `pub const Kernel` at
> module scope directly — the wrapper is just convenient when several
> kernels share a file.

### Two ways to use a kernel

**Offline / tooling** — get the MLIR string and print or diff it:

```zig
const ir = try MyKernel.Kernel.emit(allocator, ctx, .{ /* cfg */ });
defer allocator.free(ir);
try stdout.print("{s}\n", .{ir});
```

**Inside a ZML model** — emit IR and drop a `stablehlo.custom_call`
(targeting `tpu_custom_call`) into the model in one shot:

```zig
const out = MyKernel.Kernel.call(
    .{ .x = x_tensor, .cu_seqlens = cu },        // Inputs (by name)
    .{ .y = x_tensor.shape() },                  // Outputs (Shapes)
    .{
        .cfg = .{ /* cfg */ },
        .extras = .{ .vmem_limit_bytes = 1 << 20 },
    },
).y;
```

`Kernel.call` takes a `mosaic_tpu.CallExtras` via `opts.extras` for
TPU-specific knobs (`vmem_limit_bytes`, `disable_bounds_checks`,
`disable_semaphore_checks`, `has_communication`, plus
`additional_attributes` and `output_operand_aliases` for in-place
updates).

---

## Argument specs

Each field passed to `b.declareArgsOpts(...)` (or to the lower-layer
`Builder.buildOpts(...)`) is an `ArgSpec.Kind` tagged-union literal. The
four kinds mirror what a Mosaic `func.func` can carry:

| Kind literal                                           | MLIR type                            | Use case                                |
|--------------------------------------------------------|--------------------------------------|-----------------------------------------|
| `.{ .scalar = .i32 }`                                  | plain `i32`                          | `program_id`, `num_programs`, sizes     |
| `.{ .iter = .{ .semantics = .parallel } }`             | `index`                              | Loop / grid iter index with optional `tpu.dimension_semantics` |
| `.{ .ref = .{ .shape = &.{...}, .dtype = .bf16 } }`    | `memref<… x T, #tpu.memory_space<...>>` | Pallas-style operand                |
| `.{ .sem_array = .{ .shape = &.{2}, .kind = .dma } }`  | `memref<…x!tpu.(dma_)semaphore, semaphore_mem>` | Scratch sem pool (e.g. DMA double-buffer) |
| `.{ .sem = .regular }` / `.{ .sem = .dma }`            | `!tpu.semaphore` / `!tpu.dma_semaphore` | Single semaphore handle             |

### Roles

`RefSpec.role` mirrors what Pallas `pallas_call` calls each operand:

| Role               | Where it appears                                    | `window_params` / `transform_N` |
|--------------------|-----------------------------------------------------|---------------------------------|
| `.input` (default) | Inputs declared via `in_specs`                      | one entry per input             |
| `.output`          | Outputs declared via `out_specs`                    | one entry per output            |
| `.scalar_prefetch` | Leading SMEM refs (Pallas's `scalar_prefetch`)      | none — but appended to every `transform_N` body's args |
| `.scratch`         | Trailing scratch refs (kv_bufs, l/m/acc, sem_array) | none                            |

The DSL filters `window_params` to only `.input`/`.output` entries (matching
Pallas's `len(grid_mapping.block_mappings)`), and re-emits the
`scalar_prefetch` refs as trailing args on every emitted `transform_N` stub.

The default `memory_space` for a `.ref` is `.vmem`. Override with `.smem`,
`.cmem`, `.hbm`, `.semaphore_mem`, `.vmem_shared`, or `.any` per the
`MemorySpace` enum. Iteration semantics (`.parallel`, `.arbitrary`,
`.sequential`) attach `tpu.dimension_semantics` to the iter arg.

**Runtime payloads:** the variant tags are fixed per kernel, but shape /
dtype / memory_space values can be runtime — typical for kernels
parameterized by a `Config` struct.

---

## Pallas `pallas_call` metadata

The wrapper attributes Pallas attaches to the `func.func` are exposed through
`Builder.Opts`:

| Field                    | Type                      | Meaning                                                      |
|--------------------------|---------------------------|--------------------------------------------------------------|
| `.@"noinline"`           | `bool` (default false)    | `no_inline` attribute on the function.                       |
| `.core_type`             | `?CoreType` (default `.tc`) | `tpu.core_type = #tpu.core_type<…>`.                       |
| `.dimension_semantics`   | `[]const DimensionSemantics` | `dimension_semantics = [...]` — one entry per grid dim.   |
| `.scalar_prefetch`       | `i64` (default 0)         | `scalar_prefetch = N : i64`.                                 |
| `.scratch_operands`      | `i64` (default 0)         | `scratch_operands = N : i64`.                                |
| `.iteration_bounds`      | `?[]const i64`            | `iteration_bounds = array<i64: …>` — the grid shape.         |
| `.pallas_window_params`  | `bool` (default false)    | Auto-emit `window_params = [...]` and trailing `transform_N` stub funcs. |

Per-arg `window_params` behavior follows
`jax/_src/pallas/mosaic/lowering.py:1014–1131` exactly. With
`pallas_window_params = true`:

- **`.hbm` / `.any` / `.semaphore_mem`** memory spaces ⇒ empty `{}` window
  entry, **no `transform_N`**. Anything else gets a populated dict.
- **VMEM trivial-window** (no `RefSpec.window`, or `window.block_shape == ref.shape`)
  ⇒ auto-force `pipeline_mode = synchronous` (matching Pallas's
  `Buffered(1)` injection).
- **Otherwise** ⇒ emit only attributes the user explicitly set on
  `RefSpec.window`.
- **`window_bounds`** = `block_shape` (defaults to operand shape).
- **`transform_N` body** returns `block_rank` i32 zeros, mirroring a trivial
  index map.

Override per ref-arg via `RefSpec.window`:

```zig
.x = .{ .ref = .{
    .shape = &.{ 256, 512 },
    .dtype = .bf16,
    .window = mtt.ArgSpec.WindowSpec{
        .block_shape = &.{ 128, 512 },
        .pipeline_mode = .double_buffered,
        .revisit_mode = .any,
        .element_window = .{ .pad_low = &.{ 0, 0 }, .pad_high = &.{ 0, 0 } },
        // Custom transform_N return — by default the body returns
        // `block_rank` zeros (a trivial index_map). Override per slot.
        .transform_returns = &.{
            .{ .program_id = 1 },  // q_blk_idx
            .{ .program_id = 0 },  // heads_blk_idx
            .zero,
        },
    },
} },
```

The DSL treats the window as **non-trivial** (so it does *not* auto-force
`pipeline_mode = synchronous` for VMEM) when `transform_returns` is non-null
and contains anything other than all `.zero`s. This matches Pallas's
`bm.has_trivial_window()` check.

`PipelineMode` is `{ synchronous, double_buffered }` (matching Pallas's
`Buffered(1)` / `Buffered(2)`). `RevisitMode` is `{ immediate, any }`.

> **Anonymous-struct gotcha.** Anonymous-struct literals (`.{ ... }`) don't
> coerce through the optional-pointer wrapping that Zig 0.16 expects for
> `RefSpec.window`. Always write the type explicitly:
> `.window = mtt.ArgSpec.WindowSpec{ ... }`.

---

## The `Value` type

```zig
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,
    ...
};
```

- `inner` is the underlying MLIR SSA handle; reach for it when calling raw
  `tpu.*` / `arith.*` / `vector.*` constructors.
- `kernel` is the owning `Builder` pointer; all fluent methods route through
  it. The DSL populates `kernel` automatically for every value it creates
  (`emit`, `arg`, loop-region args, etc.). If you construct a `Value` by hand
  (`.{ .inner = raw }`), the fluent methods will panic — use `Builder.*`
  static helpers instead.

Introspection methods always available:

| Method            | Returns                          | Notes                                          |
|-------------------|----------------------------------|------------------------------------------------|
| `type_()`         | `*const mlir.Type`               | Full MLIR type (scalar, vector, or memref).    |
| `elemType()`      | `*const mlir.Type`               | Element type; same as `type_()` for scalars.   |
| `isVector()`      | `bool`                           | `true` for `vector<…>`.                        |
| `isMemRef()`      | `bool`                           | `true` for `memref<…>`.                        |
| `isShaped()`      | `bool`                           | Vector or memref.                              |
| `rank()`          | `usize`                          | `0` for scalars.                               |
| `dim(i)`          | `i64`                            | Traps on scalars.                              |
| `shape()`         | `Shape` (BoundedArray, MAX_RANK) | Use `.constSlice()` to feed `[]const i64`.     |
| `isFloatElem()`   | `bool`                           | True for any float element type.               |
| `isIntElem()`     | `bool`                           | True for any integer element type.             |

---

## Fluent methods on `Value`

Every fluent method returns a new `Value`. They dispatch to
`arith.addi` / `arith.addf` / … based on the element type. The RHS is
`anytype` — see [Polymorphic scalars](#polymorphic-scalars).

### Arithmetic

| Method         | Int op       | Float op         |
|----------------|--------------|------------------|
| `.add(rhs)`    | `arith.addi` | `arith.addf`     |
| `.sub(rhs)`    | `arith.subi` | `arith.subf`     |
| `.mul(rhs)`    | `arith.muli` | `arith.mulf`     |
| `.div(rhs)`    | `arith.divsi`| `arith.divf`     |
| `.rem(rhs)`    | `arith.remsi`| `arith.remf`     |

For Python-style floor-division (`a // b` matches `jax.numpy.floor_divide`'s
sign-correcting expansion — `divsi` plus a `(sign(a) != sign(b)) && (a % b !=
0)` correction), use `b.divFloor(lhs, rhs)`. `divsi` alone truncates toward
zero, which differs for mixed-sign operands.
| `.bitAnd(rhs)` | `arith.andi` | —                |
| `.bitOr(rhs)`  | `arith.ori`  | —                |
| `.minimum(rhs)`| `arith.minsi`| `arith.minnumf`  |
| `.maximum(rhs)`| `arith.maxsi`| `arith.maxnumf`  |

### Comparisons

All return i1 (scalar) or `vector<… x i1>`. Signed integer predicates by
default; ordered float predicates for floats.

| Method     | Int predicate | Float predicate |
|------------|---------------|-----------------|
| `.lt(rhs)` | `.slt`        | `.olt`          |
| `.le(rhs)` | `.sle`        | `.ole`          |
| `.gt(rhs)` | `.sgt`        | `.ogt`          |
| `.ge(rhs)` | `.sge`        | `.oge`          |
| `.eq(rhs)` | `.eq`         | `.oeq`          |
| `.ne(rhs)` | `.ne`         | `.one`          |

### Casts

| Method                | Effect                                              |
|-----------------------|-----------------------------------------------------|
| `.to(dtype)`          | Numeric cast preserving shape; auto-dispatches int↔float.    |
| `.bitcast(dtype)`     | `tpu.bitcast` — same-bitwidth element-type swap.    |
| `.abs()`              | `math.absf` (float) / `math.absi` (int).            |

`.to(...)` chooses the right MLIR op based on the source vs target kind:

| Source kind | Target kind    | MLIR op            |
|-------------|----------------|--------------------|
| int         | wider int      | `arith.extsi`      |
| int         | narrower int   | `arith.trunci`     |
| int         | float          | `arith.sitofp`     |
| float       | int            | `arith.fptosi`     |
| float       | wider float    | `arith.extf`       |
| float       | narrower float | `arith.truncf`     |
| same type   | same type      | no-op              |

For `index`-typed values use `b.indexCast(v, target_ty)` or `b.toIndex(v)`
explicitly — `.to(...)` is for numeric types.

---

## Polymorphic scalars

Most `Builder.*` helpers and every fluent method's RHS take `anytype`. The
lift rule:

- `Value` → pass through.
- `comptime_int` / `comptime_float` → a constant whose element type matches
  the LHS operand. `v_i64.mul(16)` lifts `16` to i64, `v_f32.add(1.0)` lifts
  `1.0` to f32.
- Runtime `i8/i16/i32`, `i64`, `f16/f32/f64`, etc. → a constant that
  preserves the source Zig width.

If you need to force a specific width for a runtime variable, cast at the
call site: `pid.mul(@as(i32, @intCast(block_size)))`.

### When do you need `b.lift` / `b.liftAs` / `b.cIndex`?

| Helper                         | Effect                                                   |
|--------------------------------|----------------------------------------------------------|
| `b.lift(value)`                | Wrap a Zig scalar; dtype inferred from source.           |
| `b.liftAs(value, dtype)`       | Wrap a Zig scalar as a specific `DType`.                 |
| `b.cIndex(value)`              | `arith.constant N : index` — for memref offsets.         |

Reach for these when the DSL surface demands an existing `Value` and there's
no binop sibling to match against — typically loop seeds for
`openWhile(.{inits}, ...)`, `openFor(0, N, 1, .{init})`, or `var x: Value =
b.liftAs(0, .i32);` that a later branch may reassign.

If the constant flows through any `.add` / `.mul` / `splat` etc., skip the
lift and let auto-lift do it.

---

## Constants and constant hoisting

The DSL provides splat-style constructors for the dense-splat fast-path that
Pallas/Mosaic uses:

| Helper                                 | Emits                                       |
|----------------------------------------|---------------------------------------------|
| `b.zeros(shape, dtype)`                | `arith.constant dense<0.0> : vector<…>`     |
| `b.ones(shape, dtype)`                 | Ones vector via dense splat.                |
| `b.full(shape, value, dtype)`          | `arith.constant dense<value> : vector<…>`   |
| `b.splat(value, shape, dtype)`         | Same as `full` for comptime values; `vector.broadcast` for runtime Values. |

These emit a single `arith.constant` (instead of a scalar-const +
`vector.broadcast` pair) so the IR diff vs Pallas stays empty for splat-only
constants.

---

## Loads and stores

Mosaic separates **vector** loads/stores from **scalar** memref loads/stores.
Helpers infer everything from the memref's type — there is no `result_type`
argument.

### Vector

| Helper                                                  | Pallas / Mosaic equivalent                            |
|---------------------------------------------------------|-------------------------------------------------------|
| `b.refLoad(ref)`                                        | `vector.load ref[0, …, 0] : memref<…>, vector<…>`     |
| `b.vectorLoadAt(ref, indices)`                          | `vector.load ref[indices...]` with full ref shape.    |
| `b.vectorLoadShape(ref, indices, shape)`                | Sliced `vector.load` — `ref[i, j, :, k:k+BLOCK]`.     |
| `b.tpuVectorLoad(ref, indices, .{ .strides, .mask })`   | `tpu.vector_load` — sublane stride + mask form.       |
| `b.refStore(ref, value)`                                | Full-shape `tpu.vector_store ref[0, …]`.              |
| `b.vectorStoreAt(ref, value, indices)`                  | `tpu.vector_store ref[indices…]`.                     |
| `b.refStoreOpts(ref, value, indices, .{ .strides, .mask, .add })` | Strided / masked / atomic-add store.    |
| `b.stridedLoad(ref, indices, strides)`                  | `tpu.strided_load` — result shape = `ref.shape / strides` (per-dim ceil). |
| `b.stridedLoadShape(ref, indices, strides, result_shape)` | `tpu.strided_load` with explicit result shape — for cropped loads (e.g. half-rows). |
| `b.stridedStore(ref, value, indices, strides)`          | `tpu.strided_store`.                                  |

> **`refLoad` vs `tpuVectorLoad`.** Pallas's default emit is
> `vector.load`, which is what `refLoad` / `vectorLoadAt` produce. Reach for
> `tpuVectorLoad` only when you need sublane stride or a per-element mask
> (the `tpu.vector_load` form). Same on the store side.

`indices` must be `index`-typed Values (use `b.toIndex(v)` to convert from
i32). If you pass `&.{}`, the helper pads with a cached `arith.constant 0 :
index` to match the ref's rank — the same SSA value is reused per block, so
no `%c0` duplication shows up in the IR.

### Scalar (SMEM)

| Helper                                          | MLIR op                                  |
|-------------------------------------------------|------------------------------------------|
| `b.scalarLoad(ref, indices)`                    | `memref.load`                            |
| `b.scalarStore(ref, value, indices)`            | `memref.store`                           |

Use these for SMEM args (e.g. `cu_seqlens`) where Pallas emits scalar
`memref.load`.

---

## Vector and shape helpers

| Helper                                              | Effect                                                   |
|-----------------------------------------------------|----------------------------------------------------------|
| `b.broadcastTo(src, shape)`                         | `vector.broadcast` — leading-dim broadcast or rank-prepend. |
| `b.shapeCast(src, new_shape)`                       | `vector.shape_cast` — reshape preserving total lanes.    |
| `b.reshape(src, new_shape)`                         | `tpu.reshape` (TPU-flavored reshape; differs from `shapeCast`). |
| `b.repeat(src, dimension, times)`                   | `tpu.repeat`.                                            |
| `b.concatenate(sources, dimension)`                 | `tpu.concatenate`.                                       |
| `b.transpose(src, permutation)`                     | `tpu.transpose`.                                         |
| `b.broadcastInSublanes(src, lane)`                  | `tpu.broadcast_in_sublanes`.                             |
| `b.rotate(src, .{ .amount, .dimension, .stride, .stride_dimension })` | `tpu.rotate`.                          |
| `b.vectorExtract(src, position)`                    | `vector.extract` — scalar / lower-rank slice.            |
| `b.iota(shape, dtype, dimensions)`                  | `tpu.iota` over a chosen dim.                            |
| `b.arange(n, dtype)`                                | Pallas-style 1-D iota: `<1xN>` `tpu.iota` + `vector.shape_cast` to `<N>`. Mirrors `_iota_lowering_rule`. |

> **Use `arange`, not `iota(&.{N})`.** The TPU layout pass requires vectors
> to be 2-D with the lane dim trailing; a bare 1-D `tpu.iota` is rejected.
> `arange` emits the `<1xN>` + `shape_cast` pair Pallas's lowering produces.

> **`shapeCast` vs `reshape`.** `shapeCast` emits the upstream
> `vector.shape_cast` Pallas uses everywhere; `reshape` emits TPU's
> `tpu.reshape`. Stick to `shapeCast` when matching Pallas IR.

### Auto-broadcast

There is **no implicit auto-broadcast** in the Mosaic DSL. Mosaic kernels run
inside the TPU's tile semantics and the layout pass is strict about
`vector<…>` ranks/shapes — surprise-broadcasts would be footguns. Build the
broadcasts you need explicitly with `shapeCast` + `broadcastTo`:

```zig
// b_k[:, None] broadcast to <BK x BV>
const bk_2d = k.shapeCast(b_k, &.{ BK, 1 });
const bk_bc = k.broadcastTo(bk_2d, &.{ BK, BV });
```

For scalar→vector splatting, use `b.broadcastTo(scalar, shape)` (it routes
through `vector.broadcast`) or `b.full / b.splat` for comptime constants.

---

## Reductions, scans, sorts

### Vector-dialect reductions

| Helper                                              | Effect                                                   |
|-----------------------------------------------------|----------------------------------------------------------|
| `b.multiReduction(kind, src, acc, axes)`            | `vector.multi_reduction` — multi-D reduce along `axes`.  |
| `b.reduceToScalar(kind, src, acc)`                  | 1-D vec → scalar via the Pallas idiom: `shape_cast<N>→<1xN>`, `multi_reduction[1]`, `extract[0]`. |
| `b.vectorReductionFlat(kind, src)`                  | `vector.reduction` — bare 1-D reduce. Layout pass may reject; prefer `reduceToScalar`. |

`kind` is a `CombiningKind` (`.add`, `.mul`, `.minsi`, `.maxsi`, `.minnumf`,
`.maxnumf`, `.and_`, `.or_`, `.xor`).

> **Pallas's reduce-to-scalar idiom.** Pallas never emits a bare
> `vector.reduction` for the L2-norm / sum-to-scalar pattern — it always
> goes through `multi_reduction` after a leading-dim `shape_cast`. The
> `reduceToScalar` wrapper produces exactly that shape, so kernels stay
> byte-equivalent to `pallas_call`'s output. See
> `jax/_src/pallas/mosaic/lowering.py:reduce_lowering_rule`.

### TPU-dialect reductions / scan / sort

| Helper                                              | MLIR op                                                  |
|-----------------------------------------------------|----------------------------------------------------------|
| `b.allReduce(input, kind, dim)`                     | `tpu.all_reduce` — keeps `dim` as size 1.                |
| `b.reduceIndex(input, kind, axis)`                  | `tpu.reduce_index` — drop axis, replace with `index`.    |
| `b.scan(input, kind, mask)`                         | `tpu.scan`.                                              |
| `b.sort(keys, values, mask, .{ .descending })`      | `tpu.sort` — returns `[3]Value` (sorted_keys, indices, sorted_values). |

---

## Compute primitives

### Math (scalar / elementwise)

| Helper                                         | MLIR op            |
|------------------------------------------------|--------------------|
| `b.exp(x)`                                     | `math.exp`                       |
| `b.exp2(x)`                                    | `exp(x * ln2)` — matches Pallas's `_exp2_lowering_rule` (forward-compat). |
| `b.mathExp2(x)`                                | Raw `math.exp2` — only legal on cloud TPUs after 2025-07-26. |
| `b.log(x)` / `b.log2(x)`                       | `math.log` / `math.log2`         |
| `b.sqrt(x)` / `b.rsqrt(x)`                     | `math.sqrt` / `math.rsqrt`       |
| `b.sin(x)` / `b.cos(x)` / `b.tanh(x)` / `b.erf(x)` | `math.*`                     |
| `b.absf(x)` / `b.absi(x)` / `b.abs(x)`         | `math.absf` / `math.absi` (auto) |
| `b.floor(x)` / `b.ceil(x)`                     | `math.floor` / `math.ceil`       |
| `b.powf(x, y)`                                 | `math.powf`                      |
| `b.fma(a, b, c)`                               | `math.fma`                       |
| `b.reciprocal(x)` / `b.reciprocalOpts(x, .{ .approximate })` | `tpu.reciprocal`   |

### Matmul

```zig
// y = lhs @ rhs + acc
const y = k.matmul(lhs, rhs, acc);
const y2 = k.matmulOpts(lhs, rhs, acc, .{
    .transpose_lhs = false,
    .transpose_rhs = true,
    .precision = .bf16,
});
```

`matmul` lowers to `tpu.matmul`; result type is taken from `acc`. Use
`matmulOpts` to set transposes, `tpu.contract_precision`, or a custom
`#tpu.dot_dimension_numbers<…>` attribute.

### Selects / where

| Helper                          | MLIR op                  |
|---------------------------------|--------------------------|
| `b.select(cond, t, f)`          | `arith.select`           |
| `b.where(cond, x, y)`           | Alias of `select`.       |

### Masks

| Helper                                                   | Effect                                  |
|----------------------------------------------------------|-----------------------------------------|
| `b.createMask(low_bounds, high_bounds, shape)`           | `vector.create_mask`.                   |
| `b.rangeMask(limits, shape)`                             | `vector.create_mask` from `0..limit`.   |

---

## Memref ops

For carving sub-references out of a larger memref or reinterpreting an
existing one:

| Helper                                                          | MLIR op                  |
|-----------------------------------------------------------------|--------------------------|
| `b.memRefSlice(mem_ref, offsets, sizes, strides, result_shape)` | `memref.subview`.        |
| `b.memRefSqueeze(mem_ref, result_shape)`                        | `memref.collapse_shape` style. |
| `b.memRefReshape(mem_ref, result_shape)`                        | `tpu.memref_reshape`.    |
| `b.memRefBitcast(mem_ref, dtype)`                               | `tpu.memref_bitcast` — same shape, new element dtype.    |
| `b.memRefBitcastShape(mem_ref, result_shape, dtype)`            | `tpu.memref_bitcast` with explicit result shape — for the bf16↔i32 packing trick (`<64x128xbf16>` → `<32x128xi32>`). |
| `b.reinterpretCast(mem_ref, result_shape, dtype)`               | `memref.reinterpret_cast`. |
| `b.assumeLayout(src)`                                           | `tpu.assume_layout`.     |
| `b.eraseMemRefLayout(mem_ref)`                                  | `memref.cast` to no-layout. |

---

## DMAs, semaphores, and barriers

For multi-core / async DMA orchestration:

| Helper                                                                       | MLIR op                |
|------------------------------------------------------------------------------|------------------------|
| `b.semAlloc(.regular)` / `b.semAlloc(.dma)`                                  | `tpu.sem_alloc`.       |
| `b.semBarrier()`                                                             | `tpu.sem_barrier`.     |
| `b.semRead(sem)`                                                             | `tpu.sem_read`.        |
| `b.semWait(sem, amount)`                                                     | `tpu.sem_wait`.        |
| `b.semSignal(sem, amount)` / `b.semSignalOpts(sem, amount, .{ .device_id, .core_id })` | `tpu.sem_signal`. |
| `b.barrier(barrier_id)`                                                      | `tpu.barrier`.         |
| `b.enqueueDma(src, dst, target_sem, .{ .source_semaphore, .device_id, .core_id, .priority, .strict_ordering })` | `tpu.enqueue_dma`. |
| `b.waitDma2(sem, src, dst, .{ .device_id, .core_id, .strict_ordering })`     | `tpu.wait_dma2`.       |
| `b.deviceId()`                                                               | `tpu.device_id`.       |
| `b.delay(cycles)`                                                            | `tpu.delay`.           |
| `b.traceStart(level, message)` / `b.traceStop()`                             | `tpu.trace_start` / `tpu.trace_stop` — Pallas's einsum boundaries. |
| `b.dotDimensionNumbers(lhs_c, rhs_c, lhs_nc, rhs_nc, output_order, lhs_b, rhs_b)` | Build a `#tpu.dot_dimension_numbers<...>` attr for `matmulOpts`. |

---

## Control flow

SCF regions are built with **scope-based** builders — `b.openFor(...)`,
`b.openIf(...)` / `b.openIfElse(...)`, `b.openWhile(...)` return a typed
scope value. You emit body ops into a bare `{}` block, terminate with
`yield` / `yieldThen` / `yieldAfter`, and read results from `scope.results`
afterward. No `ctx` struct, no function literal — lexical scope carries
captures.

### `openFor` — scf.for with iter_args

```zig
var loop = k.openFor(c0_i32, T, c1_i32, .{ b_h_init });
{
    const t = loop.iv;
    const b_h = loop.carried[0];

    // ... body ops ...

    loop.yield(.{ b_h_updated });
}
const final_b_h = loop.results[0];
```

- `lower` / `upper` / `step` are `anytype`: pass Values, comptime ints, or
  runtime Zig ints. Use `i32` constants (`b.lift(@as(i32, 0))`) when you
  want the loop counter to be `i32` (matching Pallas's `scf.for ... : i32`).
  The default for `index`-typed counters is `b.cIndex(N)`.
- `inits` is a tuple literal `.{v1, v2, ...}`. Arity is comptime — `carried`,
  `results`, and the `yield(...)` call are all fixed-size arrays of the
  same length.
- Empty loops: `b.openFor(0, N, 1, .{})` + `loop.yield(.{})`.

### `openIf` — scf.if without else (side-effects only)

```zig
var i = k.openIf(cond);
{
    // then body — store, increment, etc.
    i.yieldThen(.{});
}
```

Use for conditional side-effects (no results). The else block is auto-built
empty.

### `openIfElse` — scf.if with else and optional results

```zig
var i = k.openIfElse(cond, .{ k.scalarTy(.f32) });
{
    i.yieldThen(.{ x });
}
{
    i.yieldElse(.{ y });
}
const r = i.results[0];
```

- `result_types` is a tuple of `*const mlir.Type` — use `k.scalarTy(dtype)`
  / `k.vectorTy(shape, dtype)` / `k.memRefTy(shape, dtype, mem_space)`.
  Pass `.{}` for an if/else with no results.

### `openWhile` — scf.while

```zig
var w = k.openWhile(.{ i0 }, .{ k.scalarTy(.i32) });
{
    const s = w.before_carried[0];
    w.yieldBefore(s.lt(10), .{ s });
}
{
    const s = w.after_carried[0];
    w.yieldAfter(.{ s.add(1) });
}
const r = w.results[0];
```

- `inits` — tuple of `Value`s; before-region arg types = their types.
- `after_types` — tuple of `*const mlir.Type`; after-region arg types + the
  scf.while's result types.
- `yieldBefore(cond, forwarded)` — `forwarded` arity must match `after_types`.
- `yieldAfter(values)` — arity must match `inits`.

### General tips

- Scopes stack naturally: `openFor { openIf { openFor { ... } } }` all just
  push/pop the kernel's current-block stack.
- Empty yields are legal: `loop.yield(.{})`, `i.yieldThen(.{})`, etc.
- After any `yield*`, the scope's `.results` holds the scf op's results as
  a `[N]Value`.

---

## Escape hatches

Everything the DSL doesn't expose is still reachable — `Builder` is a
book-keeping layer, not a walled garden:

- `v.inner` — the raw `*const mlir.Value`; feed it to any `tpu.*` /
  `arith.*` / `vector.*` / `memref.*` / `scf.*` / `math.*` constructor
  directly.
- `b.ctx` — the owning `*mlir.Context`.
- `b.currentBlock()` — the active `mlir.Block` (respects `pushBlock` /
  `popBlock`).
- `b.emit(op)` / `b.emitMulti(op, n)` — if you've built an `mlir.Operation`
  by hand, use these so the result comes back as a `Value` with its
  `kernel` field populated.
- `b.scalarTy(dtype)` / `b.vectorTy(shape, dtype)` /
  `b.memRefTy(shape, dtype, mem_space)` — type constructors used by raw
  ops and by `openIfElse.result_types` / `openWhile.after_types`.

When you drop to raw ops, remember to still **append them to the current
block** — `b.emit(...)` does that for you; `_ = op.appendTo(b.currentBlock())`
is the manual form.

---

## Pallas Python ↔ Zig DSL cheat-sheet

| Python                                                | Zig                                                          |
|-------------------------------------------------------|--------------------------------------------------------------|
| `pl.program_id(0)`                                    | `a.i_v` (declared as `.{ .scalar = .i32 }`)                  |
| `cu_seqlens[i]`                                       | `k.scalarLoad(a.cu_seqlens, &.{ k.toIndex(i) })`             |
| `lax.div(a, b)`                                       | `k.divsi(a, b)`                                              |
| `lax.rem(a, b)`                                       | `k.remsi(a, b)`                                              |
| `jnp.arange(N, dtype=jnp.int32)`                      | `k.arange(N, .i32)`                                          |
| `o_k < KEY_DIM`                                       | `k.cmpi(.slt, o_k, dim_splat)` / `o_k.lt(dim_splat)`         |
| `mask_k[:, None] & mask_v[None, :]`                   | `k.andi(k.broadcastTo(k.shapeCast(mask_k, &.{BK,1}), &.{BK,BV}), k.broadcastTo(mask_v, &.{BK,BV}))` |
| `jnp.where(mask, x, 0.0)`                             | `k.select(mask, x, k.zeros(shape, dtype))`                   |
| `q_ref[0, t, h, :].astype(jnp.float32)`               | `k.vectorLoadShape(a.q, &.{c0, t_idx, h_idx, c0}, &.{1,1,1,BK})` then `.shapeCast(&.{BK})` then `.to(.f32)` |
| `q_ref[0, t, h, :] = b_o.astype(o_ref.dtype)`         | `k.vectorStoreAt(a.q, k.shapeCast(b_o, &.{1,1,1,BK}), &.{c0, t_idx, h_idx, c0})` |
| `jnp.sqrt(jnp.sum(b_q * b_q) + 1e-6)`                 | `k.sqrt(k.addf(k.reduceToScalar(.add, k.mulf(b_q, b_q), cst_zero_1), cst_eps))` |
| `jnp.exp(b_g)`                                        | `k.exp(b_g)`                                                 |
| `jnp.sum(b_h * b_k[:, None], axis=0)`                 | `k.multiReduction(.add, k.mulf(b_h, bk_bc), cst_zero_v, &.{0})` |
| `b_h * jnp.exp(b_g)` (broadcast to <BK,BV>)           | `k.mulf(b_h, k.broadcastTo(exp_g, &.{BK, BV}))`              |
| `b_beta * (b_v - predicted_v)`                        | `k.mulf(k.broadcastTo(b_beta, &.{BV}), k.subf(b_v, predicted_v))` |
| `lax.fori_loop(0, T, body, init)`                     | `var loop = k.openFor(c0_i32, T, c1_i32, .{init}); { …; loop.yield(.{…}); }` |
| `if cond: …` (side-effects only)                      | `var i = k.openIf(cond); { …; i.yieldThen(.{}); }`           |
| `if cond: x = a else: x = b`                          | `var i = k.openIfElse(cond, .{ty}); { …; i.yieldThen(.{a}); }; { …; i.yieldElse(.{b}); }` |
| `while cond:`                                         | `var w = k.openWhile(.{inits}, .{after_types}); { …; w.yieldBefore(cond, .{…}); }; { …; w.yieldAfter(.{…}); }` |
| `pltpu.MemorySpace.SMEM`                              | `.memory_space = .smem` on the `RefSpec`                     |
| `pl.BlockSpec(memory_space=…)` (trivial window)       | Default `RefSpec` — auto-`synchronous` on VMEM               |
| `pl.BlockSpec(block_shape=(BM, BK), …)`               | `.window = .{ .block_shape = &.{BM, BK} }`                   |
| `pl.BlockSpec(..., pipeline_mode=Buffered(2))`        | `.window = .{ .pipeline_mode = .double_buffered }`           |

---

## Pitfalls

1. **TPU vectors are 2-D minimum.** A bare `tpu.iota : vector<Nxi32>` or
   `vector.reduction` on `vector<Nxi32>` is rejected by the layout pass.
   Use `b.arange(N, dtype)` (emits the `<1xN>` iota + `shape_cast`) and
   `b.reduceToScalar(...)` (emits the `shape_cast<N>→<1xN>` +
   `multi_reduction[1]` + `extract[0]` triple). Match Pallas's emit pattern
   verbatim and the layout pass is happy.

2. **No implicit auto-broadcast.** Unlike the Triton DSL, Mosaic won't
   silently `expand_dims` / `broadcast` a rank-mismatched RHS. Build the
   broadcasts explicitly (`shapeCast` + `broadcastTo`). The Pallas Python
   DSL emits the same explicit broadcast pair — mirroring it keeps the IR
   diff empty.

3. **`vector.load` indices must be `index`-typed.** Convert from i32 with
   `k.toIndex(v)` (emits `arith.index_cast`). JAX does no CSE inside
   `pallas_call` lowering, so re-emitting the same `index_cast` per use is
   the byte-equivalent thing to do.

4. **Re-emit derived offsets per use.** Pallas re-runs its index_map
   computation per operand load/store rather than CSE-ing intermediate
   `arith.muli` / `arith.addi`. To match its IR exactly, do the same: emit
   `iv_off = i_v * BV` again at every site that needs it. The MLIR
   canonicalizer collapses duplicates downstream.

5. **`refLoad` vs `tpuVectorLoad`.** Pallas's default `ref[...]` lowers to
   `vector.load` (not `tpu.vector_load`). Use `refLoad` / `vectorLoadAt` /
   `vectorLoadShape` for byte-equivalence; reach for `tpuVectorLoad` only
   when you need `tpu.vector_load`-specific features (sublane stride or
   per-element mask).

6. **`shapeCast` vs `reshape`.** `shapeCast` emits the upstream
   `vector.shape_cast` Pallas uses everywhere; `reshape` emits TPU's
   `tpu.reshape` (different op, different layout semantics). Stick with
   `shapeCast` when matching Pallas IR.

7. **SMEM args take scalar `memref.load`, not `vector.load`.** Use
   `k.scalarLoad(a.cu_seqlens, ...)` for SMEM operands (typically the
   scalar-prefetch path). VMEM args go through the vector-load helpers.

8. **`pallas_window_params = true` only adds metadata.** It does not change
   the kernel body — it just appends `iteration_bounds`, `window_params`,
   and the `transform_N` stub funcs to the `func.func` so downstream passes
   that consume `window_params` (DMA scheduling, double-buffering) work.
   For non-trivial windows or non-VMEM operands, attach a `RefSpec.window`
   to override the auto-`synchronous` default.

9. **`finish` verifies before serializing.** A verifier failure here almost
   always means an operand-type mismatch or a missing block terminator —
   `scf.yield` operand count must match the loop's result types, and
   `func.return` arity must match the declared result types.
