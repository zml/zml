# ZML Triton DSL — Author's Guide

A Zig builder for Triton IR (TTIR) that tries to read like `triton.language`
Python while staying in pure Zig. Used by `zml/moe/triton_kernels.zig` and by
anything else that wants to emit TTIR to hand to `zml.ops.triton(...)`.

Two files define the surface:

- `zml/triton/kernel.zig` — the DSL (`Kernel`, `Value`, `DType`, `ArgSpec`).
- `mlir/dialects/ttir/ttir.zig` — Layer A raw op constructors (escape hatch).

## Table of contents

1. [Skeleton of a kernel](#skeleton-of-a-kernel)
2. [`Kernel.build` and argument specs](#kernelbuild-and-argument-specs)
3. [The `Value` type](#the-value-type)
4. [Fluent methods on `Value`](#fluent-methods-on-value)
5. [Polymorphic scalars](#polymorphic-scalars)
6. [Comptime vs runtime widths](#comptime-vs-runtime-widths)
7. [Shape helpers](#shape-helpers)
8. [Loads and stores](#loads-and-stores)
9. [Reductions and scans](#reductions-and-scans)
10. [Control flow](#control-flow)
11. [Escape hatches](#escape-hatches)
12. [Python Triton ↔ Zig DSL cheat-sheet](#python-triton--zig-dsl-cheat-sheet)
13. [Pitfalls](#pitfalls)

---

## Skeleton of a kernel

```zig
const tri = @import("zml/triton");
const Kernel = tri.Kernel;
const Value = tri.Value;

pub fn myKernelTtir(allocator: std.mem.Allocator) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "my_kernel", .{
        .x_ptr = .{ .ptr = .f32 },
        .y_ptr = .{ .ptr = .f32 },
        .n = .{ .scalar = .i32 },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const pid = k.programId(.x);
    const offs = k.arange(0, 64, .i32).add(pid.mul(@as(i32, 64)));
    const mask = offs.lt(a.n);

    const x = k.loadMasked(a.x_ptr.addPtr(offs), mask);
    const y = x.mul(@as(f32, 2.0));
    k.storeOpts(a.y_ptr.addPtr(offs), y, .{ .mask = mask });

    return k.finish(&.{});
}
```

Four structural pieces, always in this order:

1. Describe the function signature with a named-field struct literal passed to `Kernel.build`.
2. Grab the `Kernel` pointer (`&spec.kernel`) and the named-args struct (`spec.args`).
3. Build the body with fluent `Value` methods and `k.*` helpers; reach block args via `a.<field_name>`.
4. Call `k.finish(results)` to terminate + verify + serialize.

---

## `Kernel.build` and argument specs

`Kernel.build(allocator, ctx, name, spec, result_types)` takes a named-field
struct literal where each field value is an `ArgSpec.Kind` tagged-union literal.
Field names become MLIR arg names, and `spec.args` exposes the corresponding
`Value` for each field.

The four arg kinds mirror what TTIR's `tt.func` can carry:

| Kind literal                                            | MLIR type                | Use case                        |
|---------------------------------------------------------|--------------------------|---------------------------------|
| `.{ .ptr = .f32 }`                                      | `!tt.ptr<f32>`           | Pointers (default divisibility) |
| `.{ .scalar = .i32 }`                                   | plain `i32`              | Strides, sizes, flags           |
| `.{ .tensor = .{ &.{64, 128}, .f32 } }`                 | `tensor<64x128 x f32>`   | Pre-built tensor input (rare)   |
| `.{ .ptr_opts = .{ .dtype = .f32, .divisibility = 16 } }` | `!tt.ptr<f32>` with override | Custom `address_space`/divisibility |

`.ptr` gets a default `tt.divisibility = 32` hint. Use `.ptr_opts` to change
that (or set `divisibility = null` to suppress). Runtime dtypes work by
substituting the enum literal: `.a_ptr = .{ .ptr = dsl(config.a_dtype) }`.

`Kernel.build` returns a heap-allocated `*Built(Spec)`:

```zig
pub fn Built(comptime Spec: type) type {
    return struct {
        kernel: Kernel,
        args: NamedArgs(Spec), // struct mirroring Spec, each field a Value
        allocator: std.mem.Allocator,
        pub fn deinit(self: *@This()) void { ... }
    };
}
```

The heap allocation keeps `&spec.kernel` at a stable address so the back-pointers
carried by `Value` stay valid. The last argument is the list of `tt.func` result
types — pass `&.{}` for kernels that don't return values (almost always).

### Legacy API — `Kernel.init`

`Kernel.init(allocator, ctx, name, args: []const ArgSpec, result_types)` remains
available for dynamic arg counts or when you're constructing the spec at runtime.
It returns a plain `Kernel` and you grab block args with `k.arg(i)`. Prefer
`Kernel.build` for the common named-spec case.

---

## The `Value` type

```zig
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Kernel = null,
    ...
};
```

- `inner` is the underlying MLIR SSA handle; reach for it when you need to call
  into raw `ttir.*` / `arith.*` constructors.
- `kernel` is the owning `Kernel` pointer; all fluent methods route through it.
  The DSL populates `kernel` automatically for every value it creates (`emit`,
  `arg`, loop-region args, reduction-region args, etc.). If you construct a
  `Value` by hand (`.{ .inner = raw }`), the fluent methods will panic — use
  `Kernel.*` static helpers instead.

Introspection methods always available:

| Method            | Returns                           | Notes                                   |
|-------------------|-----------------------------------|-----------------------------------------|
| `type_()`         | `*const mlir.Type`                | Full MLIR type (scalar or tensor).      |
| `elemType()`      | `*const mlir.Type`                | Element type; same as `type_()` for scalars. |
| `isTensor()`      | `bool`                            | `true` for ranked tensors.              |
| `rank()`          | `usize`                           | `0` for scalars.                        |
| `dim(i)`          | `i64`                             | Traps on scalars.                       |
| `shape()`         | `Shape` (BoundedArray, `MAX_RANK`)| Use `.constSlice()` to feed `[]const i64`. |
| `isFloatElem()`   | `bool`                            | True for any float element type.        |
| `isIntElem()`     | `bool`                            | True for any integer element type.      |

---

## Fluent methods on `Value`

Every fluent method returns a new `Value`. They dispatch to
`arith.addi` / `arith.addf` / ... based on the element type. The RHS is
`anytype` — see [Polymorphic scalars](#polymorphic-scalars).

### Arithmetic

| Method         | Int op       | Float op     | Notes                              |
|----------------|--------------|--------------|------------------------------------|
| `.add(rhs)`    | `arith.addi` | `arith.addf` |                                    |
| `.sub(rhs)`    | `arith.subi` | `arith.subf` |                                    |
| `.mul(rhs)`    | `arith.muli` | `arith.mulf` |                                    |
| `.div(rhs)`    | `arith.divsi`| `arith.divf` | Signed integer divide.             |
| `.rem(rhs)`    | `arith.remsi`| `arith.remf` | Signed integer remainder.          |
| `.cdiv(rhs)`   | `arith.ceildivsi` | —       | Int only — for `tl.cdiv`.          |
| `.min(rhs)`    | `arith.minsi`| `arith.minimumf` |                                |
| `.max(rhs)`    | `arith.maxsi`| `arith.maximumf` |                                |
| `.bitAnd(rhs)` | `arith.andi` | —            | Also handles i1 masks.             |
| `.bitOr(rhs)`  | `arith.ori`  | —            |                                    |

### Comparisons

All return i1 (scalar) or `tensor<... x i1>`. Signed integer predicates by
default; `.olt` / `.oge` / ... for floats.

| Method       | Int predicate | Float predicate |
|--------------|---------------|-----------------|
| `.lt(rhs)`   | `.slt`        | `.olt`          |
| `.le(rhs)`   | `.sle`        | `.ole`          |
| `.gt(rhs)`   | `.sgt`        | `.ogt`          |
| `.ge(rhs)`   | `.sge`        | `.oge`          |
| `.eq(rhs)`   | `.eq`         | `.oeq`          |
| `.ne(rhs)`   | `.ne`         | `.one`          |

### Casts — `.to(dtype)`

`v.to(.f32)` chooses the right MLIR op based on the current vs target kind:

| Source kind | Target kind | MLIR op                       |
|-------------|-------------|-------------------------------|
| int         | wider int   | `arith.extsi`                 |
| int         | narrower int| `arith.trunci`                |
| int         | float       | `arith.sitofp`                |
| float       | int         | `arith.fptosi`                |
| float       | float       | `tt.fp_to_fp` (rounding=rtne) |
| same type   | same type   | no-op (returns `self`)        |

### Shape / pointer helpers

| Method                       | Effect                                                      |
|------------------------------|-------------------------------------------------------------|
| `.broadcast2d(axis, m, n)`   | 1-D → `[m, n]`; `axis=1` keeps as column (`v[:, None]`), `axis=0` as row (`v[None, :]`). |
| `.splatTo(shape)`            | `tt.splat` this scalar to the given shape.                  |
| `.addPtr(offset)`            | `tt.addptr`; `offset` can be a Value or a comptime int.     |

---

## Polymorphic scalars

Every fluent method's RHS (and every `k.splat` / `k.lift` input) is `anytype`.
The lift rule:

- `Value` → pass through.
- `comptime_int` / `comptime_float` → a constant whose element type matches
  the LHS operand. `v_i64.mul(16)` lifts `16` to i64, `v_f32.add(1.0)` lifts
  `1.0` to f32, and so on.
- Runtime `i8/i16/i32`, `i64`, `f16/f32/f64`, etc. → a constant that preserves
  the source Zig width. i8/i16/i32 (signed or unsigned) → i32 const; i64/u64 →
  i64 const; f16 → f16; f32 → f32; f64 → f64. Unsigned ints are bit-cast
  (bit pattern preserved), so `@as(u32, 0xFFFF_FFFF)` becomes an i32 with the
  same all-ones bit pattern.

This is why the common Python idiom `offs < 64` works in Zig as
`offs.lt(64)` without any wrapper — as long as the comptime scalar can be
represented in `offs`'s element type.

If you need to force a specific width for a runtime variable, cast at the call
site: `pid.mul(@as(i32, @intCast(block_size)))`.

### Scalar ↔ tensor auto-broadcast

Fluent arithmetic and comparison ops (`.add`, `.sub`, `.mul`, `.div`, `.rem`,
`.cdiv`, `.bitAnd`, `.bitOr`, `.min`, `.max`, `.lt`, `.le`, `.gt`, `.ge`,
`.eq`, `.ne`) are symmetric: if one operand is a tensor and the other a
scalar, the scalar is splatted to match. Either direction works:

```zig
const offs = iv.add(cols);   // scalar iv + tensor cols → tensor
const offs = cols.add(iv);   // tensor cols + scalar iv → tensor
```

This mirrors Triton/NumPy's `i + cols`. Note: size-1 / rank-unification
broadcasting (e.g. `[1,64] + [32,1]`) is **not** auto-inserted — use
`.broadcast2d`, `k.expandDims`, or `k.broadcast` explicitly. `.addPtr` is
also symmetric: a scalar `!tt.ptr<T>` is splatted when the offset is a
tensor, so `x_ptr.addPtr(offs)` works without a manual `splatTo`.

---

## Comptime vs runtime widths

TTIR has no implicit int-width conversion — `arith.muli(i32, i64)` is a verifier
error. The DSL mirrors that: comptime literals adapt, but anything already
typed at the Zig level preserves its width. If both operands are runtime and
their widths disagree, you have to `.to(dtype)` one of them.

Recurring example from `triton_kernels.zig`:

```zig
// pid (i32) vs block (i64 because it came from a usize config)
const block_i32: i32 = @intCast(block);
const token_start_init = pid.mul(block_i32);
// alternative: pid.to(.i64).mul(block)
```

For pointer math it is often cleaner to promote everything to i64 up front:

```zig
const pid = k.programId(.x).to(.i64);
const offs = k.arange(BLOCK, .i64);
```

---

## Shape helpers

| Helper                                 | Effect                                           |
|----------------------------------------|--------------------------------------------------|
| `k.arange(start, end, dtype)`          | `tl.arange(start, end).to(dtype)`. Pass `.i32` for no-op cast. |
| `k.zeros(shape, dtype)`                | `tl.zeros(shape, dtype=dtype)`.                  |
| `k.ones(shape, dtype)`                 | Ones tensor.                                     |
| `k.splat(value, shape)`                | `tl.splat`; accepts Value or comptime scalar.    |
| `k.broadcast2d(vec, axis, m, n)`       | Same as `vec.broadcast2d(axis, m, n)`.           |
| `k.mask2d(cond_m, cond_n, m, n)`       | `cond_m[:, None] & cond_n[None, :]`. Replaces ~5 lines. |
| `k.lift(value)`                        | Wrap a Zig scalar as a DSL Value; dtype inferred from source. |
| `k.liftAs(value, dtype)`               | Wrap a Zig scalar as a DSL Value of a specific `DType`. |

---

## Loads and stores

Load helpers infer everything from the pointer's type — scalar `!tt.ptr<T>`
loads a scalar `T`, `tensor<... x !tt.ptr<T>>` loads a `tensor<... x T>`.
No `result_type` / `shape` / `dtype` arguments required.

| Helper                                          | Python equivalent                         |
|-------------------------------------------------|-------------------------------------------|
| `k.load(ptr)`                                   | `tl.load(ptr)` — scalar or tensor.        |
| `k.loadMasked(ptr, mask)`                       | `tl.load(ptrs, mask=mask, other=0)` — zero-of-dtype auto-built. |
| `k.loadOpts(ptr, .{ .mask, .other, ... })`      | `tl.load(ptrs, mask=..., other=..., cache_modifier=..., eviction_policy=..., volatile=...)`. |
| `k.store(ptr, value)`                           | `tl.store(ptr, value)`.                   |
| `k.storeOpts(ptr, value, .{ .mask, ... })`      | `tl.store(ptr, value, mask=..., cache_modifier=..., eviction_policy=...)`. |

`LoadOpts` / `StoreOpts` fields (`.mask`, `.other`, `.cache_modifier`,
`.eviction_policy`, `.@"volatile"`) take DSL-level `Value`s / enum literals —
no raw MLIR types needed.

### Result-type inference

Most helpers that produce a new Value infer their result type from the inputs,
so user code never constructs `*const mlir.Type` manually:

| Helper                                              | How the result type is derived              |
|-----------------------------------------------------|---------------------------------------------|
| `k.load(ptr)` / `k.loadOpts(ptr, opts)`             | From `ptr.type_()` (scalar or tensor of ptrs). |
| `k.loadMasked(ptr, mask)`                           | From `ptr.type_()`; zero-of-dtype auto-built. |
| `k.dot(a, b, acc)` / `k.dotOpts(a, b, acc, opts)`   | `acc.type_()`.                              |
| `k.dotScaled(...)` / `k.dotScaledOpts(...)`         | `acc.type_()`.                              |
| `k.reshape(src, new_shape)` / `k.reshapeOpts(...)`  | `new_shape` + `src` element type.           |
| `k.gather(src, indices, axis)` / `k.gatherOpts(...)`| `indices` shape + `src` element type.       |
| `k.histogram(src, num_bins)` / `k.histogramOpts(...)` | `tensor<num_bins x i32>`.                 |
| `k.fpToFp(src, out_dtype)` / `k.fpToFpOpts(...)`    | `src` shape + `out_dtype`.                  |
| `k.convertf(src, out_dtype)` / `k.convertfOpts(...)`| `src` shape + `out_dtype`.                  |
| `k.scalingTruncf(src, scale, out_dtype)` / …Opts    | `src` shape + `out_dtype`.                  |
| `k.descriptorLoad(desc, indices, shape, dtype)`     | `shape` + `dtype`.                          |

Rule of thumb: you only supply things the op needs beyond what inputs carry —
a `new_shape`, a `num_bins`, an `out_dtype`, a TMA tile descriptor. The DSL
never asks you to restate information it can read off a `Value`.

`k.scalarTy(dtype)` / `k.tensorTy(shape, dtype)` are still available but only
needed by `openIf.result_types` / `openWhile.after_types` (where block-arg
types can be arbitrary) and by the raw `externElementwise` /
`inlineAsmElementwise` escape hatches.

### `fn X / fn XOpts` pattern

Every Kernel method whose Python Triton counterpart has named keyword-only
arguments comes in two flavours:

- `k.X(non_named_args...)` — defaults applied; matches `tl.X(args...)` in
  Python, which in Python would use the zero-arg defaults.
- `k.XOpts(non_named_args..., struct { named... })` — full named-param set
  as a typed Zig struct literal with field defaults matching Triton.

Pairs in place today: `load` / `loadOpts`, `store` / `storeOpts`,
`dot` / `dotOpts`, `reshape` / `reshapeOpts`,
`atomicRmw` / `atomicRmwOpts`, `atomicCas` / `atomicCasOpts`,
`fpToFp` / `fpToFpOpts`, `clampf` / `clampfOpts`,
`cast` / `castOpts`, `cat` / `catOpts`,
`convertf` / `convertfOpts`, `scalingTruncf` / `scalingTruncfOpts`,
`dotScaled` / `dotScaledOpts`,
`inlineAsmElementwise` / `inlineAsmElementwiseOpts`,
`externElementwise` / `externElementwiseOpts`,
`descriptorLoad` / `descriptorLoadOpts`, `histogram` / `histogramOpts`,
`devicePrint` / `devicePrintOpts`,
`deviceAssert` / `deviceAssertOpts`,
`maximum` / `maximumOpts`, `minimum` / `minimumOpts`,
`sum` / `sumOpts`, `max` / `maxOpts`, `min` / `minOpts`,
`cumsum` / `cumsumOpts`, `cumprod` / `cumprodOpts`.

Ops with no kwargs in Python (pure positional): `gather(src, indices, axis)`,
`trans(src, order)`, `permute(src, order)`, `cat(lhs, rhs)` (dim=0 default),
`split(src)`, `join(lhs, rhs)`, `item(src)`, `broadcast(a, b)` (symmetric),
`broadcastTo(value, shape)`, `where(cond, x, y)`, `zeros(shape, dtype)`,
`full(shape, value, dtype)`.

---

## Reductions and scans

```zig
const absmax = k.max(abs_y);                   // tl.max(abs_y) — axis=None (all dims)
const total  = k.sum(padded_counts);           // tl.sum(padded_counts)
const cumul  = k.cumsum(x);                    // tl.cumsum(x, axis=0)

// Keyword arguments → the *Opts variant with a typed struct literal.
const axis0_max = k.maxOpts(abs_y, .{ .axis = 0 });                // tl.max(x, axis=0)
const absmax_kd = k.maxOpts(abs_y, .{ .axis = 0, .keep_dims = true }); // tl.max(x, 0, keep_dims=True)
const rev_cum   = k.cumsumOpts(x, .{ .reverse = true });            // tl.cumsum(x, 0, reverse=True)
```

Under the hood these build the required `tt.reduce` / `tt.scan` region with a
matching `arith.addf` / `arith.addi` / `maximumf` / `maxsi` combine function —
no more hand-rolled combiner closures for the common cases. When
`.keep_dims = true`, the DSL inserts a trailing `tt.expand_dims` / `splat` so
the result shape matches the input rank with a size-1 reduced axis (matching
Python Triton).

For custom combiners, use the low-level `k.reduce(ctx, .{ .src, .axis, .elem,
.result, .combine })` and `k.scan(ctx, .{ .src, .axis, .elem, .result,
.combine, .reverse })` — they take a `fn(*Kernel, Value, Value, CtxT) Value`
closure for `combine` and give you full control.

---

## Control flow

SCF regions are built with **scope-based** builders — `k.openFor(...)`,
`k.openIf(...)` / `k.openIfElse(...)`, `k.openWhile(...)` return a typed scope
value. You emit body ops into a bare `{}` block, terminate with `yield` /
`yieldThen` / `yieldAfter`, and read results from `scope.results` afterward.
No `ctx` struct, no function literal, no `kk`/`k` split — lexical scope
carries captures because everything runs in the same function.

### `openFor` — scf.for with iter_args

```zig
var loop = k.openFor(0, N, BLOCK, .{acc0});
{
    const iv  = loop.iv;
    const acc = loop.carried[0];
    const offs    = iv.add(cols);
    const tile    = k.loadMasked(x_ptr.addPtr(offs), offs.lt(N));
    const partial = tile.sum();
    loop.yield(.{ acc.add(partial) });
}
const total = loop.results[0];
```

- `lower`/`upper`/`step` are `anytype`: pass Values, comptime ints, or runtime
  Zig ints. Comptime literals adapt to the first operand's type; cast explicitly
  to force a specific width (e.g. `@as(i32, @intCast(config.block))`).
- `inits` is a tuple literal `.{v1, v2, ...}`. Arity is comptime — `carried`,
  `results`, and the `yield(...)` call are all fixed-size arrays of the same
  length.
- Empty loops: `k.openFor(0, N, S, .{})` + `loop.yield(.{})` (no iter_args).

### `openIf` — scf.if without else (side-effects only)

```zig
var i = k.openIf(cond);
{
    // then body
    i.yieldThen(.{});   // closes scope, builds scf.if with empty else
}
```

Use for conditional side-effects (no results). There is no `yieldElse` — the
else block is auto-built empty.

### `openIfElse` — scf.if with else and optional results

```zig
var i = k.openIfElse(cond, .{ k.scalarTy(.f32) });   // result types
{
    // then body
    i.yieldThen(.{ x });
}
{
    // else body
    i.yieldElse(.{ y });
}
const r = i.results[0];
```

- `result_types` is a tuple of `*const mlir.Type` — use `k.scalarTy(dtype)` or
  `k.tensorTy(shape, dtype)`. Pass `.{}` for an if/else with no results but
  distinct then/else bodies.
- `yieldThen` swaps to the else block; `yieldElse` builds the `scf.if` op.
- Both branches must yield arity-matching tuples.

### `openWhile` — scf.while

```zig
var w = k.openWhile(.{ i0 }, .{ k.scalarTy(.i32) });
{
    // before region (condition)
    const s = w.before_carried[0];
    w.yieldBefore(s.lt(10), .{ s });    // scf.condition: cond + values forwarded to after
}
{
    // after region (body)
    const s = w.after_carried[0];
    w.yieldAfter(.{ s.add(1) });        // values forwarded back to before
}
const r = w.results[0];
```

- `inits` — tuple of `Value`s; before-region arg types = their types.
- `after_types` — tuple of `*const mlir.Type`; after-region arg types + the
  scf.while's result types.
- `yieldBefore(cond, forwarded)` — `forwarded` arity must match `after_types`.
- `yieldAfter(values)` — arity must match `inits`.

### `reduce` / `scan` — custom combiners (still callback-based)

```zig
k.reduce(ctx, .{ .src, .axis, .elem, .result, .combine = fn (*Kernel, Value, Value, Ctx) Value });
k.scan  (ctx, .{ .src, .axis, .reverse = false, .elem, .result, .combine });
```

These stay on the callback pattern because the combine fn is a tiny pure
function (`(a, b) -> a ⊕ b`) where a `ctx` struct is a cleaner fit than a
scoped builder. Reach for `k.sum` / `k.max` / `k.cumsum` / etc. first — the
built-in reductions cover the common cases.

### General tips

- Scopes stack naturally: `openFor { openIf { openFor { ... } } }` all just
  push/pop the kernel's current-block stack (same for `openIfElse` /
  `openWhile`).
- Empty yields are legal: `loop.yield(.{})`, `i.yieldThen(.{})`, etc.
- After any `yield*`, the scope's `.results` field holds the scf op's results
  as a `[N]Value`.

---

## Escape hatches

Everything the DSL doesn't expose is still reachable — `Kernel` is a
book-keeping layer, not a walled garden:

- `v.inner` — the raw `*const mlir.Value`; feed it to any `ttir.*` /
  `arith.*` / `math.*` / `scf.*` constructor directly.
- `k.ctx` — the owning `*mlir.Context`.
- `k.currentBlock()` — the active `mlir.Block` (respects `pushBlock` /
  `popBlock`).
- `k.emit(op)` / `k.emitMulti(op, n)` — if you've built an `mlir.Operation`
  by hand, use these so the result comes back as a `Value` with its `kernel`
  field populated (you'll need that for any subsequent fluent calls).

When you drop to raw ops, remember to still **append them to the current
block** — `k.emit(...)` does that for you; `_ = op.appendTo(k.currentBlock())`
is the manual form.

---

## Python Triton ↔ Zig DSL cheat-sheet

| Python                                        | Zig                                                                |
|-----------------------------------------------|--------------------------------------------------------------------|
| `tl.program_id(0)`                            | `k.programId(.x)`                                                  |
| `tl.num_programs(0)`                          | `k.numPrograms(.x)`                                                |
| `tl.arange(0, N)`                             | `k.arange(0, N, .i32)`                                             |
| `x.to(tl.int64)`                              | `x.to(.i64)`                                                       |
| `a + b`, `a * b`, `a // b`                    | `a.add(b)`, `a.mul(b)`, `a.div(b)`                                 |
| `a < b`                                       | `a.lt(b)`                                                          |
| `mask & other`                                | `mask.bitAnd(other)`                                               |
| `tl.where(c, a, b)`                           | `k.where(c, a, b)` (alias of `k.select`)                           |
| `vec[:, None]`                                | `vec.broadcast2d(1, m, n)`                                         |
| `vec[None, :]`                                | `vec.broadcast2d(0, m, n)`                                         |
| `m[:, None] & (c[None, :] < N)`               | `k.mask2d(m, c.lt(N), m_sz, n_sz)`                                 |
| `ptr + offs`                                  | `ptr.addPtr(offs)` (scalar ptr auto-splats to tensor-of-ptrs)      |
| `tl.load(ptr, mask=mask, other=0)`            | `k.loadMasked(ptrs, mask)`                                         |
| `tl.load(ptr)`                                | `k.load(ptrs)`                                                     |
| `tl.load(ptr, mask=m, other=o)`               | `k.loadOpts(ptrs, .{ .mask = m, .other = o })`                     |
| `tl.store(ptr, v)`                            | `k.store(ptr, v)`                                                  |
| `tl.store(ptr, v, mask=mask)`                 | `k.storeOpts(ptr, v, .{ .mask = mask })`                           |
| `tl.zeros((M, N), dtype=tl.float32)`          | `k.zeros(&.{ M, N }, .f32)`                                        |
| `tl.full((M, N), 3, tl.int32)`                | `k.full(&.{ M, N }, 3, .i32)`                                      |
| `tl.sum(x)`                                   | `k.sum(x)` — axis=None default                                     |
| `tl.sum(x, axis=0)`                           | `k.sumOpts(x, .{ .axis = 0 })`                                     |
| `tl.sum(x, axis=0, keep_dims=True)`           | `k.sumOpts(x, .{ .axis = 0, .keep_dims = true })`                  |
| `tl.max(x)` / `tl.min(x)`                     | `k.max(x)` / `k.min(x)`                                            |
| `tl.max(x, axis=0)`                           | `k.maxOpts(x, .{ .axis = 0 })`                                     |
| `tl.maximum(a, b)` / `tl.minimum(a, b)`       | `a.maximum(b)` / `a.minimum(b)` or `k.maximum(a, b)`               |
| `tl.cumsum(x)` / `tl.cumprod(x)`              | `k.cumsum(x)` / `k.cumprod(x)`                                     |
| `tl.cumsum(x, axis=0, reverse=True)`          | `k.cumsumOpts(x, .{ .reverse = true })`                            |
| `tl.dot(a, b, acc=acc)`                       | `k.dot(a, b, acc)`                                                 |
| `tl.dot(a, b, acc, input_precision="ieee")`   | `k.dotOpts(a, b, acc, .{ .input_precision = .ieee })`              |
| `tl.clamp(x, mn, mx)`                         | `k.clampf(x, mn, mx)`                                              |
| `tl.clamp(x, mn, mx, propagate_nan=NAN_ALL)`  | `k.clampfOpts(x, mn, mx, .{ .propagate_nan = .all })`              |
| `tl.atomic_add(p, v)`                         | `k.atomicRmw(.add, p, v)`                                          |
| `tl.atomic_add(p, v, mask=m, sem="relaxed")`  | `k.atomicRmwOpts(.add, p, v, .{ .mask = m, .sem = .relaxed })`     |
| `tl.histogram(x, num_bins)`                   | `k.histogram(x, num_bins)`                                         |
| `tl.histogram(x, num_bins, mask=m)`           | `k.histogramOpts(x, num_bins, .{ .mask = m })`                     |
| `tl.reshape(x, shape)`                        | `k.reshape(x, &.{ ...shape })`                                     |
| `tl.reshape(x, shape, can_reorder=True)`      | `k.reshapeOpts(x, &.{ ...shape }, .{ .can_reorder = true })`       |
| `tl.cast(x, tl.bfloat16)`                     | `k.cast(x, .bf16)` (or fluent `x.to(.bf16)`)                       |
| `tl.cast(x, tl.bf16, fp_downcast_rounding="rtne")` | `k.castOpts(x, .bf16, .{ .fp_downcast_rounding = .rtne })`     |
| `tl.cast(x, tl.int32, bitcast=True)`          | `k.castOpts(x, .i32, .{ .bitcast = true })`                        |
| `src[indices]` (gather)                       | `k.gather(src, indices, axis)`                                     |
| `tl.trans(x, dims)` / `tl.permute(x, dims)`   | `k.trans(x, order)` / `k.permute(x, order)`                        |
| `tl.cat(a, b)`                                | `k.cat(a, b)` (dim=0 default)                                      |
| `tl.cat(a, b, can_reorder=True)`              | `k.catOpts(a, b, .{ .can_reorder = true })`                        |
| `tl.join(a, b)` / `tl.split(x)`               | `k.join(a, b)` / `k.split(x)`                                      |
| `tl.item(x)`                                  | `k.item(x)`                                                        |
| `tl.abs(x)`                                   | `k.abs(x)` (auto-dispatch) or fluent `x.abs()`                     |
| `tl.broadcast(a, b)` (symmetric)              | `k.broadcast(a, b)` → `.{ Value, Value }`                          |
| `tl.broadcast_to(v, shape)`                   | `k.broadcastTo(v, &.{ ...shape })`                                 |
| `tl.umulhi(a, b)`                             | `k.umulhi(a, b)`                                                   |
| `tl.sqrt_rn(x)` / `tl.div_rn(x, y)`           | `k.sqrtRn(x)` / `k.divRn(x, y)`                                    |
| `tl.device_print("dbg", args...)`             | `k.devicePrint("dbg", args, is_signed)`                            |
| `tl.device_assert(cond, "msg")`               | `k.deviceAssert(cond, "msg")`                                      |
| `tl.inline_asm_elementwise(...)`              | `k.inlineAsmElementwise(...)`                                      |
| `tl.extern_elementwise(...)`                  | `k.externElementwise(srcs, shape, dtype, lib, path, sym)`          |
| `for i in range(0, N, S): ...`                | `var loop = k.openFor(0, N, S, .{inits...}); { ...; loop.yield(.{...}); }`                 |
| `while cond: ...`                             | `var w = k.openWhile(.{inits}, .{after_types}); { ...; w.yieldBefore(cond, .{...}); }; { ...; w.yieldAfter(.{...}); }` |
| `if cond: ...` (side-effects, no else)        | `var i = k.openIf(cond); { ...; i.yieldThen(.{}); }`                                         |
| `if cond: ... else: ...`                      | `var i = k.openIfElse(cond, .{result_types}); { ...; i.yieldThen(.{...}); }; { ...; i.yieldElse(.{...}); }` |

---

## Pitfalls

1. **Width mismatches between runtime operands.** `v_i32.mul(v_i64)` will hit
   `arith.muli requires the same type` at verify time. Either cast one with
   `.to(...)` or reshape the kernel to stay within a single int width. Comptime
   literals adapt automatically — runtime values don't.

2. **Tensor-of-pointers via `.addPtr`.** `ptr` is a scalar `!tt.ptr<T>`; when
   `offs` is a tensor, `ptr.addPtr(offs)` auto-splats the pointer to match
   (no manual `.splatTo` needed).

3. **Scope-based control flow — no closures needed.** `openFor` / `openIf` /
   `openIfElse` / `openWhile` push a block onto the kernel's insertion stack;
   body ops emitted in the bare `{}` that follows go into that block. Captures
   are plain lexical scope, so you can reference outer Values, sizes, etc.
   directly — no `ctx` struct, no function literal.

4. **`Value.kernel == null` panics fluent calls.** Only happens if you build a
   `Value` by hand via `.{ .inner = raw }`. Route through `k.emit(...)` to keep
   the back-pointer populated — or just stick with the explicit `Kernel.*`
   helpers for that one op.

5. **Don't worry about duplicate constants.** Every lifted literal (from
   `.add(42)`, `.lt(0)`, `k.splat(1.0, shape)`, an `openFor(0, N, 1, …)`
   bound, etc.) emits a fresh `arith.constant`. This matches Python Triton's
   own frontend — the MLIR canonicalizer + CSE pass collapses duplicates on the
   way to PTX, so the generated code is identical whether you emit one
   constant or a hundred. Optimize for readability at the builder level.

6. **`finish(results, allocator)` verifies before serializing.** A verifier
   failure here almost always means an operand-type mismatch (see #1) or a
   missing block terminator — a common one is forgetting that `scf.yield`
   operand count must match the loop's result types.

7. **Yielding from loop / if / while bodies.** Inside a scope (`openFor`,
   `openIf`, `openIfElse`, `openWhile`), you *don't* return values — you call
   `scope.yield(...)` (or `yieldThen` / `yieldElse` / `yieldBefore` /
   `yieldAfter`) with a tuple of Values matching the scope's declared arity.
   Scopes with results (`openFor`, `openIfElse`, `openWhile`) store the scf
   op's results in `scope.results` as a fixed-size `[N]Value` array. `openIf`
   has no results — `yieldThen(.{})` closes and builds scf.if with an empty
   else block in one step.
