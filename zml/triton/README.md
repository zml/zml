# ZML Triton DSL — Author's Guide

A Zig builder for Triton IR (TTIR) that tries to read like `triton.language`
Python while staying in pure Zig. Used by `zml/moe/triton_kernels.zig` and by
anything else that wants to emit TTIR to hand to `zml.ops.triton(...)`.

Two files define the surface:

- `zml/triton/kernel.zig` — the DSL (`Kernel`, `Value`, `DType`, `ArgSpec`).
- `mlir/dialects/ttir/ttir.zig` — Layer A raw op constructors (escape hatch).

## Table of contents

1. [Skeleton of a kernel](#skeleton-of-a-kernel)
2. [`Kernel.init` and argument specs](#kernelinit-and-argument-specs)
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
const Arg = tri.ArgSpec;

pub fn myKernelTtir(allocator: std.mem.Allocator) ![:0]const u8 {
    const args = [_]Arg{
        Arg.ptr("x_ptr", .f32),
        Arg.ptr("y_ptr", .f32),
        Arg.scalar("n", .i32),
    };
    var kernel = try Kernel.init(allocator, ctx(), "my_kernel", &args, &.{});
    defer kernel.deinit();
    const k = &kernel;

    const x_ptr = k.arg(0);
    const y_ptr = k.arg(1);
    const n = k.arg(2);

    const pid = k.programId(.x);
    const offs = k.arange(0, 64, .i32).add(pid.mul(@as(i32, 64)));
    const mask = offs.lt(n);

    const x = k.loadMasked(x_ptr.splatTo(&.{64}).addPtr(offs), &.{64}, .f32, mask);
    const y = x.mul(@as(f32, 2.0));
    k.store(y_ptr.splatTo(&.{64}).addPtr(offs), y, .{ .mask = mask.inner });

    return kernel.finish(&.{}, allocator);
}
```

Four structural pieces, always in this order:

1. Describe the function signature with an `ArgSpec` array.
2. Build a `Kernel`, grab argument handles with `k.arg(i)`.
3. Build the body with fluent `Value` methods and `k.*` helpers.
4. Call `kernel.finish(results, allocator)` to terminate + verify + serialize.

---

## `Kernel.init` and argument specs

An `ArgSpec` is a name + kind. The three kinds mirror what TTIR's `tt.func` can
carry:

| Constructor                     | MLIR type                       | Use case                           |
|---------------------------------|---------------------------------|------------------------------------|
| `Arg.ptr(name, dtype)`          | `!tt.ptr<dtype>`                | Input/output pointers (default)    |
| `Arg.scalar(name, dtype)`       | plain `dtype`                   | Strides, sizes, flags              |
| `Arg.tensor(name, shape, dtype)`| `tensor<shape x dtype>`         | Pre-built tensor input (rare)      |

Pointer args get a default `tt.divisibility = 32` hint. Override by building the
full `ArgSpec` literal:

```zig
.{ .name = "ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = 16 } } }
// or suppress entirely:
.{ .name = "ptr", .kind = .{ .ptr = .{ .dtype = .f32, .divisibility = null } } }
```

`Kernel.init`'s last argument is the list of `tt.func` result types — pass
`&.{}` for kernels that don't return values (almost always the case).

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
- `comptime_int` / `comptime_float` → a constant whose element type matches the
  LHS operand, via `constMatching`. `v_i64.mul(16)` lifts `16` to i64,
  `v_f32.add(1.0)` lifts `1.0` to f32, and so on.
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
| `k.constMatching(value, elem_type)`    | Same as `liftAs` but takes an MLIR `*const mlir.Type` (e.g. `ref.elemType()`). |

---

## Loads and stores

| Helper                                          | Python equivalent                         |
|-------------------------------------------------|-------------------------------------------|
| `k.loadScalar(ptr, dtype)`                      | `tl.load(ptr)` for a scalar pointer.      |
| `k.loadMasked(ptr, shape, dtype, mask)`         | `tl.load(ptrs, mask=mask, other=0)` — the zero-of-dtype tensor is built for you. |
| `k.load(ptr, result_type, .{ .mask, .other })`  | Raw `tt.load`; use when you need a non-zero `other` or unusual options. |
| `k.store(ptr, value, .{ .mask })`               | `tl.store(ptr, value, mask=mask)`.        |

`tt.load` options (`.mask`, `.other`, `.cache`, `.eviction`, `.is_volatile`)
still take raw `*const mlir.Value` pointers — access them via `v.inner`.

---

## Reductions and scans

```zig
const absmax = k.reduceMax(abs_y, 0);           // tl.max(abs_y, axis=0)
const total  = k.reduceSum(padded_counts, 0);   // tl.sum(padded_counts, axis=0)
const cumul  = k.scanSum(x, 0, false);          // tl.cumsum(x, axis=0)
```

Under the hood these build the required `tt.reduce` / `tt.scan` region with a
matching `arith.addf` / `arith.addi` / `maximumf` / `maxsi` combine function —
no more hand-rolled combiner closures for the common cases.

For custom combiners, use the low-level `k.reduce(.{ .src, .axis, .elem,
.result, .combine, .ctx })` and `k.scan(.{ .src, .axis, .reverse, .elem,
.result, .combine, .ctx })` — they take a `fn(k, lhs, rhs, ctx) Value`
closure for `combine` and give you full control.

---

## Control flow

`Kernel.forLoop`, `Kernel.ifThenElse`, `Kernel.whileLoop`, and
`Kernel.parallel` build their SCF regions via Zig-style closures that receive a
`*Kernel`, the loop IVs / block args as `[]const Value`, and a caller-supplied
`ctx` struct.

**All control-flow helpers take a single struct-literal argument** — no more
remembering positional order. Fields:

| Helper         | Required                                                 | Optional (default)               |
|----------------|----------------------------------------------------------|----------------------------------|
| `forLoop`      | `.lower`, `.upper`, `.body`                              | `.step = 1`, `.inits = &.{}`, `.ctx = {}` |
| `ifThenElse`   | `.cond`, `.then_`, `.else_`                              | `.results = &.{}`, `.ctx = {}`  |
| `whileLoop`    | `.inits`, `.after_types`, `.before`, `.after`            | `.ctx = {}`                      |
| `reduce`       | `.src`, `.axis`, `.elem`, `.result`, `.combine`          | `.ctx = {}`                      |
| `scan`         | `.src`, `.axis`, `.elem`, `.result`, `.combine`          | `.reverse = false`, `.ctx = {}` |

Rules of thumb:

- Capture everything the body needs (Values, sizes, dtypes) in a local struct
  and pass it as `.ctx`. There are no implicit closures — context is the only
  way to smuggle state in.
- Return `&.{}` if the body has no yielded values. Otherwise, allocate via
  `kk.arena.allocator().alloc(Value, N)` and fill it — lifetime only needs to
  cover the call.
- `forLoop`'s `.lower` / `.upper` / `.step` are `anytype`: pass Values, comptime
  ints, or runtime Zig ints directly. Comptime literals lift to match the first
  operand's type (so `.lower=0, .upper=N, .step=1` with `N: Value` of type i32
  yields i32 bounds). To force a specific width on a Zig-side scalar, cast it
  at the call site (e.g. `@as(i32, @intCast(config.block))`).

Example — histogram loop:

```zig
const HistCtx = struct { topk_ids: Value, numel: i32, hist_block: i64, padded: i64 };
const body = struct {
    fn b(kk: *Kernel, iv: Value, iter: []const Value, hc: HistCtx) []const Value {
        const offs  = iv.splatTo(&.{hc.hist_block}).add(kk.arange(0, @intCast(hc.hist_block), .i32));
        const mask  = offs.lt(hc.numel);
        const expert_vals = kk.loadMasked(
            hc.topk_ids.splatTo(&.{hc.hist_block}).addPtr(offs),
            &.{hc.hist_block}, .i32, mask,
        );
        const h = kk.histogram(expert_vals, mask, kk.tensorTy(&.{hc.padded}, .i32));
        const out = kk.arena.allocator().alloc(Value, 1) catch @panic("OOM");
        out[0] = iter[0].add(h);
        return out;
    }
}.b;
_ = k.forLoop(.{
    .lower = 0,
    .upper = numel,
    .step = @as(i32, @intCast(hist_block)),
    .inits = &.{counts_init},
    .body = body,
    .ctx = .{ ... },
});
```

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
| `tl.where(c, a, b)`                           | `k.select(c, a, b)`                                                |
| `vec[:, None]`                                | `vec.broadcast2d(1, m, n)`                                         |
| `vec[None, :]`                                | `vec.broadcast2d(0, m, n)`                                         |
| `m[:, None] & (c[None, :] < N)`               | `k.mask2d(m, c.lt(N), m_sz, n_sz)`                                 |
| `ptr + offs`                                  | `ptr.splatTo(&.{N}).addPtr(offs)` (tensor-of-ptrs)                 |
| `tl.load(ptr, mask=mask, other=0)`            | `k.loadMasked(ptrs, &.{N}, dtype, mask)`                           |
| `tl.store(ptr, v, mask=mask)`                 | `k.store(ptr, v, .{ .mask = mask.inner })`                         |
| `tl.zeros((M, N), dtype=tl.float32)`          | `k.zeros(&.{ M, N }, .f32)`                                        |
| `tl.sum(x, axis=0)`                           | `k.reduceSum(x, 0)`                                                |
| `tl.max(x, axis=0)`                           | `k.reduceMax(x, 0)`                                                |
| `tl.cumsum(x, axis=0)`                        | `k.scanSum(x, 0, false)`                                           |
| `tl.dot(a, b, acc=acc)`                       | `k.dot(a, b, acc, result_ty, .{ .input_precision = .ieee })`       |
| `tl.atomic_add(p, v, mask=m, sem="relaxed")`  | `k.atomicRmw(.add, p, v, .{ .mask = m.inner, .sem = .relaxed, ... })` |
| `for i in range(0, N, S): ...`                | `k.forLoop(.{ .lower=0, .upper=N, .step=S, .inits=..., .body=..., .ctx=... })`              |
| `while cond: ...`                             | `k.whileLoop(.{ .inits=..., .after_types=..., .before=..., .after=..., .ctx=... })`         |
| `if cond: ... else: ...`                      | `k.ifThenElse(.{ .cond=..., .results=..., .then_=..., .else_=..., .ctx=... })`              |

---

## Pitfalls

1. **Width mismatches between runtime operands.** `v_i32.mul(v_i64)` will hit
   `arith.muli requires the same type` at verify time. Either cast one with
   `.to(...)` or reshape the kernel to stay within a single int width. Comptime
   literals adapt automatically — runtime values don't.

2. **Tensor-of-pointers needs a splat.** `ptr` is a scalar `!tt.ptr<T>`. To get
   a tensor of pointers, splat first: `ptr.splatTo(&.{N}).addPtr(offs)`.

3. **Closures can't capture — use `.ctx`.** Zig has no lexical closures.
   Anything a for/while/if body needs must come in through the `.ctx` field
   on the control-flow helper's struct-literal argument.

4. **`Value.kernel == null` panics fluent calls.** Only happens if you build a
   `Value` by hand via `.{ .inner = raw }`. Route through `k.emit(...)` to keep
   the back-pointer populated — or just stick with the explicit `Kernel.*`
   helpers for that one op.

5. **Don't worry about duplicate constants.** Every lifted literal (from
   `.add(42)`, `.lt(0)`, `k.splat(1.0, shape)`, a `forLoop(.{ .lower=0, … })`
   bound, etc.) emits a fresh `arith.constant`. This matches Python Triton's
   own frontend — the MLIR canonicalizer + CSE pass collapses duplicates on the
   way to PTX, so the generated code is identical whether you emit one
   constant or a hundred. Optimize for readability at the builder level.

6. **`finish(results, allocator)` verifies before serializing.** A verifier
   failure here almost always means an operand-type mismatch (see #1) or a
   missing block terminator — a common one is forgetting that `scf.yield`
   operand count must match the loop's result types.

7. **Arena allocations in bodies.** `kk.arena.allocator().alloc(Value, N)` is
   the intended allocator for the tiny result slices you return from loop
   bodies — it's cleared on `kernel.deinit()`. Don't use `std.heap.page`
   allocator or you leak.
