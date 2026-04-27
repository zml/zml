# ZML Triton DSL — Author's Guide

A Zig builder for Triton IR (TTIR) that tries to read like `triton.language`
Python while staying in pure Zig. Used by `zml/moe/triton_kernels.zig`,
`zml/attention/triton_kernels.zig`, and by anything else that wants to call
Triton kernels from a ZML model.

Two layers define the surface:

- **Low layer** — `zml/triton/kernel.zig`: the IR builder primitives.
  `Builder`, `Value`, `DType`, `ArgSpec`. Use these directly when you need
  full control (custom op insertion, dynamic-arity arg lists, escape hatches).
- **High layer** — `zml.Kernel(decl, Impl)`: the declarative kernel form.
  Bundles a name, a config type, an arg spec, and a body into a single type.
  Exposes `.emit(allocator, ctx, cfg)` (TTIR string) and
  `.call(args)` (TTIR string + `stablehlo.custom_call` in one shot — the
  preferred form for production model code).

## Table of contents

1. [Skeleton of a kernel](#skeleton-of-a-kernel)
2. [`zml.Kernel(decl, Impl)` — the declarative form](#zmlkerneldecl-impl--the-declarative-form)
3. [Calling kernels from a ZML model — `K.call(...)`](#calling-kernels-from-a-zml-model--kcall)
4. [Argument specs](#argument-specs)
5. [The `Value` type](#the-value-type)
6. [Fluent methods on `Value`](#fluent-methods-on-value)
7. [Polymorphic scalars](#polymorphic-scalars)
8. [Comptime vs runtime widths](#comptime-vs-runtime-widths)
9. [Shape helpers](#shape-helpers)
10. [Loads and stores](#loads-and-stores)
11. [Reductions and scans](#reductions-and-scans)
12. [Control flow](#control-flow)
13. [Escape hatches](#escape-hatches)
14. [Lower layer — `Builder.build`](#lower-layer--builderbuild)
15. [Python Triton ↔ Zig DSL cheat-sheet](#python-triton--zig-dsl-cheat-sheet)
16. [Pitfalls](#pitfalls)

---

## Skeleton of a kernel

```zig
const tri = @import("zml/triton");
const zml = @import("zml");

pub const MyKernel = zml.Kernel(.{
    .name = "my_kernel",
    .config = struct { BLOCK: i32 = 64 },
}, struct {
    pub fn run(b: *tri.Builder, cfg: anytype) !void {
        const a = try b.declareArgs(.{
            .x_ptr = .{ .ptr = .f32 },
            .y_ptr = .{ .ptr = .f32 },
            .n     = .{ .scalar = .i32 },
        });

        const pid  = b.programId(.x);
        const offs = b.arange(0, cfg.BLOCK, .i32).add(pid.mul(cfg.BLOCK));
        const mask = offs.lt(a.n);

        const x = b.loadOpts(a.x_ptr.addPtr(offs), .{ .mask = mask });
        const y = x.mul(2.0);
        b.storeOpts(a.y_ptr.addPtr(offs), y, .{ .mask = mask });
    }
});
```

A whole kernel is one `pub const` decl. Two structural pieces:

1. **`.name` + `.config`** — kernel name (the `tt.func` symbol) and the
   config type, an inline `struct { ... }` of comptime/runtime parameters.
   Use field defaults for sensible defaults.
2. **`Impl.run(b, cfg)`** — the kernel body. The first thing it should do is
   declare its args via `try b.declareArgs(.{ ... })`, which creates the
   `tt.func` op from the spec literal and returns a struct of `Value`s
   (named after the spec's fields). The rest of `run` builds the body via
   `b.*` helpers. No string return, no allocator, no manual `tt.return` —
   the wrapper terminates, verifies, and serializes for you.

The arg spec inside `b.declareArgs(...)` is a struct literal — its variant
*tags* (`.ptr`, `.scalar`, etc.) are comptime-known per call site, but the
*payload values* (`cfg.input_dtype`, `cfg.output_dtype`, …) can be runtime
values. This is what makes the same form work for both comptime-pinned
configs (offline tools, simple kernels) and runtime configs (production
models where dtypes/sizes come from `Tensor.dim()` / `Options`).

### Two ways to use a kernel

**Offline / tooling** — get the TTIR string:

```zig
const ttir = try MyKernel.emit(allocator, ctx, .{ .BLOCK = 1024 });
defer allocator.free(ttir);
```

**Inside a ZML `forward(...)`** — emit TTIR and drop a `stablehlo.custom_call`
into the model in one shot:

```zig
const out = MyKernel.call(
    .{ x_tensor, y_tensor },        // tuple of input Tensors
    .{ x_tensor.shape() },          // tuple of output Shapes
    .{
        .cfg = .{ .BLOCK = 1024 },
        .grid = .{ ceilDiv(n, 1024), 1, 1 },
        .num_warps = 4,
        .num_stages = 2,
    },
)[0];
```

The returned type exposes `MyKernel.name`, `MyKernel.Config`, and
`MyKernel.CallOpts`.

---

## `zml.Kernel(decl, Impl)` — the declarative form

`zml.Kernel(decl, Impl)` is a comptime function returning a kernel type.

`decl` is a struct literal carrying:

| Field      | Required | Meaning                                                                |
|------------|----------|------------------------------------------------------------------------|
| `.name`    | yes      | Kernel name (used as the `tt.func` symbol).                            |
| `.config`  | no       | A *type* (`struct { ... }`) describing the per-call config. Defaults to `struct {}` if absent. Field defaults on this type apply when callers pass `.{}`. |

`Impl` is a struct with one method:

```zig
pub fn run(b: *tri.Builder, cfg: anytype) !void { ... }
```

`run` is responsible for declaring the kernel's args (via
`b.declareArgs(...)`) and emitting the body. The `cfg:` parameter has type
`KernelType.Config`. Keep it `anytype` in the signature — the concrete type
isn't reachable from inside `Impl` because the kernel type isn't fully
resolved yet.

`run` emits ops via `b.*` helpers and never returns a string;
`KernelType.emit(...)` does the terminate-verify-serialize dance after `run`
returns.

---

## Calling kernels from a ZML model — `K.call(...)`

Inside a model's `forward(...)`, drop in a Triton kernel alongside regular
ZML ops with `K.call(...)`:

```zig
pub fn forward(self: Model, x: zml.Tensor, y: zml.Tensor) zml.Tensor {
    const x_pre = x.mul(2.0);                           // stablehlo

    const out = MyKernel.call(
        .{ x_pre, y, x_pre },                           // input Tensors
        .{ x_pre.shape() },                             // output Shapes
        .{
            .cfg = .{ .BLOCK = 1024 },                  // runtime config
            .grid = .{ ceilDiv(x.dim(0), 1024), 1, 1 },
            .num_warps = 4,
            .num_stages = 2,
        },
    )[0];

    return out.relu();                                  // stablehlo
}
```

`call` is a thin wrapper over `K.emit(...)` + `zml.ops.triton(...)`. It
must be called from inside a `CompilationContext` (i.e., during
`zml.module.compile(...)`). For offline tools without a CompilationContext
(e.g., dump-to-disk diff harnesses), use `K.emit(allocator, ctx, cfg)`
directly to get the TTIR string.

Signature: `K.call(inputs, outputs, opts: K.CallOpts) [outputs.len]Tensor`.

| Positional arg | Type              | Meaning                                  |
|----------------|-------------------|------------------------------------------|
| `inputs`       | tuple of `Tensor` | Operands. Tensors map to ptr args; 0-d tensors map to scalar args. |
| `outputs`      | tuple of `Shape`  | Declared output shapes.                  |

`K.CallOpts` (typed struct, all fields named):

| Field                     | Type                  | Meaning                                  |
|---------------------------|-----------------------|------------------------------------------|
| `.cfg`                    | `K.Config`            | Per-call kernel config.                  |
| `.grid`                   | `[3]i32`              | Triton launch grid `(grid_x, grid_y, grid_z)`. |
| `.num_warps`              | `i32`                 | Triton launch param.                     |
| `.num_stages`             | `i32`                 | Triton launch param.                     |
| `.output_operand_aliases` | `[]const ...` (default `&.{}`) | For in-place updates.       |
| `.debug`                  | `bool` (default false)| Emit `tt.debug = true` on the custom call. |

Returns `[outputs.len]Tensor`. Index with `[0]` for single-output kernels.

The `cfg` value can be a comptime literal *or* a runtime expression built
from `Tensor.dim()` / `Options` fields. The kernel TTIR is regenerated per
call (cached by the MLIR canonicalizer / CSE pass at the XLA layer).

> **Why `inputs` and `outputs` are positional, not in `CallOpts`.** They
> are tuples whose element types vary per call site (different `Tensor`s,
> different `Shape`s), which means they need `anytype`. Zig forbids
> `anytype` inside struct fields, so they stay as positional `anytype`
> parameters. Everything else lives in the typed `CallOpts` struct so that
> `@intCast(...)` inside `.grid` / `.cfg` / `.output_operand_aliases`
> infers a known result type — the previous unified-`anytype` form forced
> `@as(usize, @intCast(x))` boilerplate at every cast site.

---

## Argument specs

Each field passed to `b.declareArgs(...)` is an `ArgSpec.Kind` tagged-union
literal.

The four arg kinds mirror what TTIR's `tt.func` can carry:

| Kind literal                                            | MLIR type                | Use case                        |
|---------------------------------------------------------|--------------------------|---------------------------------|
| `.{ .ptr = .f32 }`                                      | `!tt.ptr<f32>`           | Pointers (default divisibility) |
| `.{ .scalar = .i32 }`                                   | plain `i32`              | Strides, sizes, flags           |
| `.{ .tensor = .{ &.{64, 128}, .f32 } }`                 | `tensor<64x128 x f32>`   | Pre-built tensor input (rare)   |
| `.{ .ptr_opts = .{ .dtype = .f32, .divisibility = 16 } }` | `!tt.ptr<f32>` with override | Custom `address_space`/divisibility |
| `.{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } }` | plain `i32` with `tt.divisibility = 16 : i32` | Scalar with specialization hint |

`.ptr` gets a default `tt.divisibility = 16` hint (matching Python Triton's
default). Use `.ptr_opts` to change that (or set `divisibility = null` to
suppress). `.scalar` carries no extra hint; switch to `.scalar_opts` when
you want to declare the same `tt.divisibility` Python's JIT auto-attaches
to int args whose runtime value is a multiple of 16.

**Runtime dtypes:** the variant *tags* (`.ptr`, `.scalar`, etc.) must be
fixed per kernel — every call to `b.declareArgs(...)` for a given kernel
uses the same struct shape. The variant *payloads* (the dtype enum, the
divisibility int) can be runtime values from `cfg`:
`.a_ptr = .{ .ptr = cfg.a_dtype }` works fine even when `cfg.a_dtype` is a
runtime field.

---

## The `Value` type

```zig
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,
    ...
};
```

- `inner` is the underlying MLIR SSA handle; reach for it when you need to call
  into raw `ttir.*` / `arith.*` constructors.
- `kernel` is the owning `Builder` pointer; all fluent methods route through it.
  The DSL populates `kernel` automatically for every value it creates (`emit`,
  `arg`, loop-region args, reduction-region args, etc.). If you construct a
  `Value` by hand (`.{ .inner = raw }`), the fluent methods will panic — use
  `Builder.*` static helpers instead.

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
| `.minimum(rhs)`| `arith.minsi`| `arith.minnumf`  | Matches `tl.minimum` (non-NaN-prop). |
| `.maximum(rhs)`| `arith.maxsi`| `arith.maxnumf`  | Matches `tl.maximum` (non-NaN-prop). |
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

| Source kind | Target kind      | MLIR op                       |
|-------------|------------------|-------------------------------|
| int         | wider int        | `arith.extsi`                 |
| int         | narrower int     | `arith.trunci`                |
| int         | float            | `arith.sitofp`                |
| **i1**      | **float**        | **`arith.uitofp`** (i1 is unsigned bool; sign-extending `1` would land at `-1.0`) |
| float       | int              | `arith.fptosi`                |
| float       | wider float      | `arith.extf` (no rounding)    |
| float       | narrower float   | fp8 involved or custom rounding → `tt.fp_to_fp`; otherwise `arith.truncf` |
| same type   | same type        | no-op (returns `self`)        |

Triton's verifier rejects `tt.fp_to_fp` with a rounding attribute when the
target is not strictly narrower than the source (e.g. `bf16 → f32`, or
same-width `bf16 ↔ f16`). The DSL routes strict upcasts through
`arith.extf` to match Python Triton's own lowering; only genuine
downcasts carry the `rtne` rounding attribute.

For narrowing FP→FP, the dispatch follows
`triton/python/triton/language/semantic.py:cast()`: standard FP narrowing
between fp64 / fp32 / fp16 / bf16 lowers to `arith.truncf` (rank-independent
— scalar and tensor both), while `tt.fp_to_fp` is reserved for cases where
fp8 is on either side or the caller requested a non-default
`fp_downcast_rounding`.

### Constant hoisting and CSE

Every `arith.constant` emitted by the DSL — whether through `lift`, `splat`
of a comptime literal, `zeros` / `ones` / `full`, or any internal helper —
is deduped against a per-`Builder` cache keyed by `<type>:<value>`. On a hit
the existing SSA value is reused; on a miss the new op is inserted at the
*top* of the entry block (before any non-constant op already there) so the
final IR has all constants gathered above the function body, in roughly
first-use order.

This mirrors what Triton's Python frontend produces and keeps the
`make_ttgir` layout pass from seeing duplicate constants — duplicates can
change the layout-slot assignment for kernels with `tt.dot`, since layout
inference walks the IR and the *first* tensor it sees with a given shape
gets `#blocked`, the second gets `#blocked2`, etc.

`zeros(shape, dtype)`, `ones(shape, dtype)`, and `full(shape, value, dtype)`
emit a single `arith.constant dense<value>:tensor<...>` (instead of the
old scalar-const + `tt.splat` two-op pattern); same for `splat(comptime,
shape)`. `splat(Value, shape)` still emits `tt.splat` since the operand is
a runtime value, not a literal.

### Shape / pointer helpers

| Method                       | Effect                                                      |
|------------------------------|-------------------------------------------------------------|
| `v.expandDims(axis)` / `b.expandDims(v, axis)` | `tt.expand_dims` — insert a size-1 axis at `axis`. Mirrors `v[:, None]` (`axis=1`) / `v[None, :]` (`axis=0`). |
| `b.broadcastTo(v, shape)`    | `tt.broadcast` — explicit broadcast (size-1 dims → target). Rarely needed; auto-broadcast handles it. |
| `.splatTo(shape)`            | `tt.splat` this scalar to the given shape.                  |
| `.addPtr(offset)`            | `tt.addptr`; `offset` can be a Value or a comptime int.     |

> **Auto-broadcast is everywhere.** Mirrors Triton's
> `semantic.broadcast_impl_value` (`triton/python/triton/language/semantic.py:724`):
> rank-unification (prepend size-1 axes) + size-1 → size-N expansion.
> Applied transparently at:
> - every fluent binop (`.add`, `.mul`, `.lt`, `.bitAnd`, …)
> - `.addPtr(offset)` (either side may be scalar or rank-mismatched)
> - `b.where(cond, x, y)` / `b.select(...)`
> - `mask` and `other` of `b.loadOpts` / `b.storeOpts`
>
> So `v[:, None] * stride + w[None, :] * other_stride` is just:
>
> ```zig
> v.expandDims(1).mul(stride).add(w.expandDims(0).mul(other_stride))
> ```
>
> No `broadcastTo`, no full-shape `tt.splat` — the broadcast materializes
> implicitly at the `addi`. This matches Python's emit pattern op-for-op
> (Python's frontend never emits an eager `tt.broadcast` followed by a
> `arith.muli` either; it emits `expand_dims` + `splat<size-1> + muli`).
>
> Reach for `b.broadcastTo` only when the consumer is a low-level op that
> bypasses the DSL's auto-broadcast — direct `arith.*` / raw `ttir.*`
> escape-hatch paths.

---

## Polymorphic scalars

Every fluent method's RHS (and every `b.splat` / `b.lift` input) is `anytype`.
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

### When is `@as` actually needed? (Rule of thumb)

Auto-lift covers most cases. `@as(...)` around a binop rhs is **redundant**
in all of these — drop it:

| Pattern                                  | Auto-lifts to…                        |
|------------------------------------------|---------------------------------------|
| `v_i64.eq(0)` / `.div(16)` / `.add(-1)`  | comptime_int → match lhs elem type    |
| `v_f32.mul(1.0 / config.fp8_max)`        | already-f32 expression → f32 const    |
| `v_i64.lt(NUM_QUERY_HEADS)` (Zig i64)    | runtime i64 → i64 const               |
| `v_i32.add(pad_term)` (Zig i32)          | runtime i32 → i32 const               |
| `b.splat(-1, shape)` / `b.openFor(0, N, 1, ...)` | comptime → matches elem / lower |
| `b.liftAs(0, .i32)`                      | liftAs takes anytype — wrap nothing   |

`@as(...)` **is** required exactly when you're narrowing a runtime Zig int
to a smaller width that `lift` won't pick on its own. The idiom is
`@as(i32, @intCast(X_i64))` — you need the `@as` because `@intCast` has no
target type to infer, and without the narrowing `lift` would emit an i64
constant that mismatches the i32 lhs:

```zig
// BLOCK_Q is a Zig i64 (from config). lhs is i32. Must narrow.
q_start_raw.div(@as(i32, @intCast(BLOCK_Q))).add(seq_idx)
```

The alternative — `q_start_raw.to(.i64).div(BLOCK_Q).to(.i32)` — mirrors
Python's `.to(tl.int64)` style but costs two extra Value-level casts; prefer
the narrowing `@as` or a pre-declared `const FOO_I32: i32 = @intCast(FOO);`
when the same width cast recurs.

### When do you need `b.lift` / `b.liftAs`?

Almost never. The DSL auto-lifts rhs for every fluent binop, `b.splat`,
`b.full`, `b.openFor` bounds, `.addPtr` offsets, etc. Reach for `b.lift` /
`b.liftAs` **only** when the DSL surface demands an existing `Value` and
there's no binop sibling to match against — typically:

- Loop seeds for `openWhile(.{inits}, ...)` where you start with a raw
  comptime literal: `const left_init = b.liftAs(0, .i32);`.
- Mutable `var x: Value = b.liftAs(0, .i32);` that a later branch may
  reassign — the var needs an initial Value even if the comptime path would
  work on its own.

If the constant is about to flow through any `.add` / `.mul` / `.splatTo`
etc., skip the lift and let auto-lift do it. And if a stride is hardcoded
to 1, **delete it**: `x.mul(1)` is a no-op, the DSL isn't Python `constexpr`
where such names are required.

### Scalar ↔ tensor auto-broadcast

Fluent arithmetic and comparison ops (`.add`, `.sub`, `.mul`, `.div`, `.rem`,
`.cdiv`, `.bitAnd`, `.bitOr`, `.min`, `.max`, `.lt`, `.le`, `.gt`, `.ge`,
`.eq`, `.ne`) are symmetric: if one operand is a tensor and the other a
scalar, the scalar is splatted to match. Either direction works:

```zig
const offs = iv.add(cols);   // scalar iv + tensor cols → tensor
const offs = cols.add(iv);   // tensor cols + scalar iv → tensor
```

This mirrors Triton/NumPy's `i + cols`. **Size-1 and rank-unification
broadcasting also work** (mirroring Python's
`semantic.broadcast_impl_value`): a `(M, 1)` and a `(1, N)` operand
broadcast to `(M, N)` automatically, and a `(M,)` lined up against a
`(M, N)` is first `expand_dims`'d to `(1, M)` and then broadcast. So
`offs_token[:, None] + offs_k[None, :]` is just:

```zig
const offs = b.expandDims(offs_token, 1).add(b.expandDims(offs_k, 0));
```

with no `b.broadcastTo` calls needed. `b.where` / `b.select` and the
`mask` / `other` operands of `b.loadOpts` / `b.storeOpts` also auto-broadcast,
so the only time `b.broadcastTo` is needed is when the consumer bypasses
the DSL (raw `arith.*` / `ttir.*` escape-hatch paths).

`.addPtr` auto-splats in **both directions**, matching Triton's
`ptr + offset` semantics: a scalar pointer is splatted to match a tensor
offset (`x_ptr.addPtr(tensor_offs)`), and a scalar offset is splatted to
match a tensor pointer (`a_ptrs.addPtr(BLOCK_K)`). Pre-splatting via
`b.splat` is no longer required for the loop-stride case.

### When is `scalar.splatTo(shape)` redundant?

Because fluent binops (`.add` / `.mul` / …) already auto-splat scalars
against tensor siblings, and `.addPtr` auto-splats either side (scalar
pointer or scalar offset) against a tensor partner, a pre-emptive
`scalar.splatTo(shape)` is usually noise:

```zig
// redundant
pid_n.mul(N).splatTo(shape).add(b.arange(0, N, .i64)).rem(n_blocb.splatTo(shape))
// equivalent — scalars auto-splat against the tensor sibling
pid_n.mul(N).add(b.arange(0, N, .i64)).rem(n_block)
```

Keep the explicit `splatTo` only when the scalar feeds into a raw
`arith.*` / `ttir.*` escape-hatch op that doesn't go through the DSL's
auto-broadcast.

---

## Comptime vs runtime widths

TTIR has no implicit int-width conversion — `arith.muli(i32, i64)` is a verifier
error. The DSL mirrors Python and **auto-promotes** integer binops when the
two runtime operands have different widths: it `extsi`s the narrower side
to the wider, matching `semantic.integer_promote_impl`. So
`v_i32.mul(v_i64)` works without a manual `.to(.i64)` — it emits exactly
the same `extsi` + `muli` pair Python would.

For the rare case where you want explicit control, you can still cast
yourself; the auto-promotion is a no-op when widths already agree.

Recurring example from `triton_kernels.zig`:

```zig
// pid (i32) vs block (i64 because it came from a usize config)
const block_i32: i32 = @intCast(block);
const token_start_init = pid.mul(block_i32);
// alternative: pid.to(.i64).mul(block)
```

For pointer math it is often cleaner to promote everything to i64 up front:

```zig
const pid = b.programId(.x).to(.i64);
const offs = b.arange(BLOCK, .i64);
```

---

## Shape helpers

| Helper                                 | Effect                                           |
|----------------------------------------|--------------------------------------------------|
| `b.arange(start, end, dtype)`          | `tl.arange(start, end).to(dtype)`. Pass `.i32` for no-op cast. |
| `b.zeros(shape, dtype)`                | `tl.zeros(shape, dtype=dtype)`.                  |
| `b.ones(shape, dtype)`                 | Ones tensor.                                     |
| `b.splat(value, shape)`                | `tl.splat`; accepts Value or comptime scalar.    |
| `b.expandDims(v, axis)` / `v.expandDims(axis)` | `tt.expand_dims` — insert a size-1 axis. Mirrors `v[:, None]` (`axis=1`) / `v[None, :]` (`axis=0`). |
| `b.mask2d(cond_m, cond_n, _, _)`       | `cond_m[:, None] & cond_n[None, :]` — emits the `(m,1) & (1,n)` form; auto-broadcast at the next consumer. |
| `b.lift(value)`                        | Wrap a Zig scalar as a DSL Value; dtype inferred from source. |
| `b.liftAs(value, dtype)`               | Wrap a Zig scalar as a DSL Value of a specific `DType`. |

---

## Loads and stores

Load helpers infer everything from the pointer's type — scalar `!tt.ptr<T>`
loads a scalar `T`, `tensor<... x !tt.ptr<T>>` loads a `tensor<... x T>`.
No `result_type` / `shape` / `dtype` arguments required.

| Helper                                          | Python equivalent                         |
|-------------------------------------------------|-------------------------------------------|
| `b.load(ptr)`                                   | `tl.load(ptr)` — scalar or tensor.        |
| `b.loadOpts(ptr, .{ .mask, .other, ... })`      | `tl.load(ptrs, mask=..., other=..., cache_modifier=..., eviction_policy=..., volatile=...)`. |
| `b.store(ptr, value)`                           | `tl.store(ptr, value)`.                   |
| `b.storeOpts(ptr, value, .{ .mask, ... })`      | `tl.store(ptr, value, mask=..., cache_modifier=..., eviction_policy=...)`. |

`LoadOpts` / `StoreOpts` fields (`.mask`, `.other`, `.cache_modifier`,
`.eviction_policy`, `.@"volatile"`) take DSL-level `Value`s / enum literals —
no raw MLIR types needed.

> **Match Python's emit form for masked loads.** `tl.load(ptr, mask=m,
> other=0.0)` lowers to a 3-operand `tt.load` op. To match it byte-for-byte
> in the Zig DSL, pass an explicit `.other`:
>
> ```zig
> const zero = b.zeros(&.{BLOCK}, dtype);
> const x = b.loadOpts(ptr, .{ .mask = mask, .other = zero });
> ```
>
> Calling `loadOpts` with `.mask` only (no `.other`) emits a 2-operand
> `tt.load` — semantically equivalent (out-of-bounds lanes are undefined,
> not zero), but the TTIR/TTGIR diff vs Python will not be empty. Triton's
> `tl.load(ptr, mask=...)` without `other=` lowers to the 2-operand form,
> so you only need the `.other` operand when the Python source explicitly
> passes `other=`.

### Result-type inference

Most helpers that produce a new Value infer their result type from the inputs,
so user code never constructs `*const mlir.Type` manually:

| Helper                                              | How the result type is derived              |
|-----------------------------------------------------|---------------------------------------------|
| `b.load(ptr)` / `b.loadOpts(ptr, opts)`             | From `ptr.type_()` (scalar or tensor of ptrs). |
| `b.dot(a, b, acc)` / `b.dotOpts(a, b, acc, opts)`   | `acc.type_()`.                              |
| `b.dotScaled(...)` / `b.dotScaledOpts(...)`         | `acc.type_()`.                              |
| `b.reshape(src, new_shape)` / `b.reshapeOpts(...)`  | `new_shape` + `src` element type.           |
| `b.gather(src, indices, axis)` / `b.gatherOpts(...)`| `indices` shape + `src` element type.       |
| `b.histogram(src, num_bins)` / `b.histogramOpts(...)` | `tensor<num_bins x i32>`.                 |
| `b.fpToFp(src, out_dtype)` / `b.fpToFpOpts(...)`    | `src` shape + `out_dtype`.                  |
| `b.convertf(src, out_dtype)` / `b.convertfOpts(...)`| `src` shape + `out_dtype`.                  |
| `b.scalingTruncf(src, scale, out_dtype)` / …Opts    | `src` shape + `out_dtype`.                  |
| `b.descriptorLoad(desc, indices, shape, dtype)`     | `shape` + `dtype`.                          |

Rule of thumb: you only supply things the op needs beyond what inputs carry —
a `new_shape`, a `num_bins`, an `out_dtype`, a TMA tile descriptor. The DSL
never asks you to restate information it can read off a `Value`.

`b.scalarTy(dtype)` / `b.tensorTy(shape, dtype)` are still available but only
needed by `openIf.result_types` / `openWhile.after_types` (where block-arg
types can be arbitrary) and by the raw `externElementwise` /
`inlineAsmElementwise` escape hatches.

### `fn X / fn XOpts` pattern

Every Builder method whose Python Triton counterpart has named keyword-only
arguments comes in two flavours:

- `b.X(non_named_args...)` — defaults applied; matches `tl.X(args...)` in
  Python, which in Python would use the zero-arg defaults.
- `b.XOpts(non_named_args..., struct { named... })` — full named-param set
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
const absmax = b.max(abs_y);                   // tl.max(abs_y) — axis=None (all dims)
const total  = b.sum(padded_counts);           // tl.sum(padded_counts)
const cumul  = b.cumsum(x);                    // tl.cumsum(x, axis=0)

// Keyword arguments → the *Opts variant with a typed struct literal.
const axis0_max = b.maxOpts(abs_y, .{ .axis = 0 });                // tl.max(x, axis=0)
const absmax_kd = b.maxOpts(abs_y, .{ .axis = 0, .keep_dims = true }); // tl.max(x, 0, keep_dims=True)
const rev_cum   = b.cumsumOpts(x, .{ .reverse = true });            // tl.cumsum(x, 0, reverse=True)
```

Under the hood these build the required `tt.reduce` / `tt.scan` region with a
matching `arith.addf` / `arith.addi` / `maxnumf` / `maxsi` combine function —
no more hand-rolled combiner closures for the common cases.

> **Gotcha — `maxnumf` vs `maximumf` (and `minnumf` vs `minimumf`):**
> Triton's `tl.max`, `tl.min`, `tl.maximum`, `tl.minimum` all default to
> the **non-NaN-propagating** form (`arith.maxnumf` / `arith.minnumf`).
> The DSL fluent forms match Python:
>
> | Python                              | DSL                                  | MLIR op           |
> |-------------------------------------|--------------------------------------|-------------------|
> | `tl.max(x)`                         | `b.max(x)` / `x.max()`               | `arith.maxnumf`   |
> | `tl.min(x)`                         | `b.min(x)` / `x.min()`               | `arith.minnumf`   |
> | `tl.maximum(a, b)`                  | `a.maximum(b)` / `b.maximum(a, b)`   | `arith.maxnumf`   |
> | `tl.minimum(a, b)`                  | `a.minimum(b)` / `b.minimum(a, b)`   | `arith.minnumf`   |
> | `tl.maximum(a, b, propagate_nan=NAN_ALL)` | `a.maximumOpts(b, .{ .propagate_nan = .all })` | `arith.maximumf` |
>
> The lower-level builders **`b.maximumf(a, b)` / `b.minimumf(a, b)` skip
> Python's defaults** and emit IEEE-754 NaN-propagating ops directly. Reach
> for them only when you genuinely want NaN propagation; otherwise stick
> with `b.maximum` / `b.minimum` (or the fluent `Value.maximum` /
> `Value.minimum`). Mismatched ops here cause subtle TTIR/LLIR diffs vs
> Python and are a frequent porting bug.

When `.keep_dims = true`, the DSL inserts a trailing
`tt.expand_dims` / `splat` so the result shape matches the input rank with a
size-1 reduced axis (matching Python Triton). For `axis = null` (the default),
the reduction goes through a `tt.reshape allow_reorder` first — even on rank-1
input — so the layout-assignment pass picks the same reduction-friendly
layout Python's frontend gets.

For custom combiners, use the low-level `b.reduce(ctx, .{ .src, .axis, .elem,
.result, .combine })` and `b.scan(ctx, .{ .src, .axis, .elem, .result,
.combine, .reverse })` — they take a `fn(*Builder, Value, Value, CtxT) Value`
closure for `combine` and give you full control.

---

## Control flow

SCF regions are built with **scope-based** builders — `b.openFor(...)`,
`b.openIf(...)` / `b.openIfElse(...)`, `b.openWhile(...)` return a typed scope
value. You emit body ops into a bare `{}` block, terminate with `yield` /
`yieldThen` / `yieldAfter`, and read results from `scope.results` afterward.
No `ctx` struct, no function literal, no `kk`/`k` split — lexical scope
carries captures because everything runs in the same function.

### `openFor` — scf.for with iter_args

```zig
var loop = b.openFor(0, N, BLOCK, .{acc0});
{
    const iv  = loop.iv;
    const acc = loop.carried[0];
    const offs    = iv.add(cols);
    const tile    = b.loadOpts(x_ptr.addPtr(offs), .{ .mask = offs.lt(N) });
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
- Empty loops: `b.openFor(0, N, S, .{})` + `loop.yield(.{})` (no iter_args).

### `openIf` — scf.if without else (side-effects only)

```zig
var i = b.openIf(cond);
{
    // then body
    i.yieldThen(.{});   // closes scope, builds scf.if with empty else
}
```

Use for conditional side-effects (no results). There is no `yieldElse` — the
else block is auto-built empty.

### `openIfElse` — scf.if with else and optional results

```zig
var i = b.openIfElse(cond, .{ b.scalarTy(.f32) });   // result types
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

- `result_types` is a tuple of `*const mlir.Type` — use `b.scalarTy(dtype)` or
  `b.tensorTy(shape, dtype)`. Pass `.{}` for an if/else with no results but
  distinct then/else bodies.
- `yieldThen` swaps to the else block; `yieldElse` builds the `scf.if` op.
- Both branches must yield arity-matching tuples.

### `openWhile` — scf.while

```zig
var w = b.openWhile(.{ i0 }, .{ b.scalarTy(.i32) });
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

### `returnIf` — early `tt.return` via cf.cond_br

```zig
// Python: if pid_m * BLOCK >= num_tokens_post_padded: return
const out_of_range = pid_m.mul(block).ge(num_tokens);
b.returnIf(out_of_range, .{});
// ... ops here live in a fresh fall-through block ...
```

Lowers to:

```mlir
cf.cond_br %out_of_range, ^ret, ^cont
^ret:
  tt.return
^cont:
  // subsequent ops
```

- `values` is a tuple of `Value`s matching the `tt.func`'s declared result
  types (pass `.{}` for a void kernel).
- Works **only at the top-level `tt.func` body** — don't call inside an
  `scf.if`/`scf.for`/`scf.while` region (those must stay structured; cf
  branches can't escape their parent scf region).
- After `returnIf`, the kernel's current block is swapped to a fresh
  fall-through block — any subsequent `b.*` calls emit into that block, and
  `b.finish(...)` appends the final `tt.return` there.
- Use this to match Python Triton's `if cond: return` lowering exactly.
  Python emits `cf.cond_br` for early-return idioms while using `scf.for` /
  `scf.if` for structured loops and value-carrying branches; our DSL does
  the same.

### `openReturnIf` — early `tt.return` with side-effects in the taken branch

```zig
// Python: if off_experts == -1: write_zeros_to_output(...); return
var scope = b.openReturnIf(is_dead);
{
    writeZerosToOutput(k, c_ptr, ...);   // ops here emit into ^ret
    scope.yieldReturn(.{});              // closes ^ret with tt.return
}
// ... ops here emit into ^cont (fall-through) ...
```

Lowers to the same `cf.cond_br ^ret, ^cont` shape as `returnIf`, but the
taken branch can carry arbitrary side-effects between the `cf.cond_br` and
the `tt.return`. Use this instead of the
`openIf(cond){body}yieldThen; returnIf(cond, values)` pair — that pattern
duplicates the predicate (emits both `scf.if` and `cf.cond_br` on the same
boolean) and drifts from Triton's reference IR.

Same restrictions as `returnIf`: top-level `tt.func` body only.

### When to use `cf` (returnIf / openReturnIf) vs `scf` (openIf / openIfElse / openFor)

| Python source                          | Emits | DSL builder                     |
|----------------------------------------|-------|---------------------------------|
| `if cond: return`                      | `cf`  | `b.returnIf(cond, .{})`         |
| `if cond: <side-effects>; return`      | `cf`  | `var s = b.openReturnIf(cond); {...; s.yieldReturn(.{})}` |
| `if cond: <side-effects>`              | `scf` | `b.openIf(cond) {...; yieldThen(.{})}` |
| `if cond: x=A else: x=B` (value-carrying) | `scf` | `b.openIfElse(cond, .{ty}) {...; yieldThen(.{a})} {...; yieldElse(.{b})}` |
| `for i in range(...)`                  | `scf` | `b.openFor(lo, hi, step, .{inits})` |
| `while cond:`                          | `scf` | `b.openWhile(.{inits}, .{after_types})` |

Rule of thumb: **`scf` for structured, value-carrying control. `cf` for
early-exits.** Both lower to identical PTX after the
`convert-scf-to-cf` pass — the difference only matters at the TTIR layer
(pattern-matching passes that run before lowering).

### `reduce` / `scan` — custom combiners (still callback-based)

```zig
b.reduce(ctx, .{ .src, .axis, .elem, .result, .combine = fn (*Builder, Value, Value, Ctx) Value });
b.scan  (ctx, .{ .src, .axis, .reverse = false, .elem, .result, .combine });
```

These stay on the callback pattern because the combine fn is a tiny pure
function (`(a, b) -> a ⊕ b`) where a `ctx` struct is a cleaner fit than a
scoped builder. Reach for `b.sum` / `b.max` / `b.cumsum` / etc. first — the
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

Everything the DSL doesn't expose is still reachable — `Builder` is a
book-keeping layer, not a walled garden:

- `v.inner` — the raw `*const mlir.Value`; feed it to any `ttir.*` /
  `arith.*` / `math.*` / `scf.*` constructor directly.
- `b.ctx` — the owning `*mlir.Context`.
- `b.currentBlock()` — the active `mlir.Block` (respects `pushBlock` /
  `popBlock`).
- `b.emit(op)` / `b.emitMulti(op, n)` — if you've built an `mlir.Operation`
  by hand, use these so the result comes back as a `Value` with its `kernel`
  field populated (you'll need that for any subsequent fluent calls).

When you drop to raw ops, remember to still **append them to the current
block** — `b.emit(...)` does that for you; `_ = op.appendTo(b.currentBlock())`
is the manual form.

---

## Lower layer — `Builder.build`

The declarative `zml.Kernel(decl, Impl)` form covers most cases. For
existing code, dynamic-arity arg lists, or escape-hatch scenarios where you
want to drive the IR construction yourself, the lower-layer `Builder.build`
API is still available:

```zig
const Builder = tri.Builder;

pub fn fusedMoeKernel(
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    config: GenerationConfig,
) ![:0]const u8 {
    var spec = try Builder.build(allocator, ctx, "fused_moe_kernel", .{
        .a_ptr = .{ .ptr = config.a_dtype },
        .b_ptr = .{ .ptr = config.b_dtype },
        // ...
    }, &.{});
    defer spec.deinit();
    const b = &spec.kernel;
    const a = spec.args;

    // ... emit body via b.* helpers; same DSL as inside `Kernel(...)` ...

    return b.finish(&.{});
}
```

`Builder.build(allocator, ctx, name, spec, result_types)` returns a
heap-allocated `*Built(Spec)`:

- `spec.kernel` — the runtime `Builder`; pass `&spec.kernel` to helpers.
- `spec.args` — the same `NamedArgs(Spec)` struct that the declarative
  `b.declareArgs(...)` returns.
- `spec.deinit()` — frees the module + bundle.
- `b.finish(results)` — terminates with `tt.return`, verifies, serializes
  to `[:0]const u8`. Pass `&.{}` for kernels with no return values.

For dynamic-arg-count construction, `Builder.init(allocator, ctx, name,
args: []const ArgSpec, result_types)` takes a runtime slice of arg specs
instead of a comptime struct literal; you grab block args with `b.arg(i)`.

You can also use `Builder.open(allocator, ctx, name)` to create an empty
builder and call `b.declareArgs(spec)` mid-stream — that's exactly what
`zml.Kernel(...)` does internally.

Prefer `zml.Kernel(decl, Impl)` for the common case.

---

## Python Triton ↔ Zig DSL cheat-sheet

| Python                                        | Zig                                                                |
|-----------------------------------------------|--------------------------------------------------------------------|
| `tl.program_id(0)`                            | `b.programId(.x)`                                                  |
| `tl.num_programs(0)`                          | `b.numPrograms(.x)`                                                |
| `tl.arange(0, N)`                             | `b.arange(0, N, .i32)`                                             |
| `x.to(tl.int64)`                              | `x.to(.i64)`                                                       |
| `a + b`, `a * b`, `a // b`                    | `a.add(b)`, `a.mul(b)`, `a.div(b)`                                 |
| `a < b`                                       | `a.lt(b)`                                                          |
| `mask & other`                                | `masb.bitAnd(other)`                                               |
| `tl.where(c, a, b)`                           | `b.where(c, a, b)` (alias of `b.select`)                           |
| `vec[:, None]`                                | `vec.expandDims(1)`                                                |
| `vec[None, :]`                                | `vec.expandDims(0)`                                                |
| `m[:, None] & (c[None, :] < N)`               | `b.mask2d(m, c.lt(N), m_sz, n_sz)`                                 |
| `ptr + offs`                                  | `ptr.addPtr(offs)` (auto-splats scalar↔tensor either way)          |
| `tl.load(ptr)`                                | `b.load(ptrs)`                                                     |
| `tl.load(ptr, mask=m)`                        | `b.loadOpts(ptrs, .{ .mask = m })`                                 |
| `tl.load(ptr, mask=m, other=o)`               | `b.loadOpts(ptrs, .{ .mask = m, .other = o })`                     |
| `tl.store(ptr, v)`                            | `b.store(ptr, v)`                                                  |
| `tl.store(ptr, v, mask=mask)`                 | `b.storeOpts(ptr, v, .{ .mask = mask })`                           |
| `tl.zeros((M, N), dtype=tl.float32)`          | `b.zeros(&.{ M, N }, .f32)`                                        |
| `tl.full((M, N), 3, tl.int32)`                | `b.full(&.{ M, N }, 3, .i32)`                                      |
| `tl.sum(x)`                                   | `b.sum(x)` — axis=None default                                     |
| `tl.sum(x, axis=0)`                           | `b.sumOpts(x, .{ .axis = 0 })`                                     |
| `tl.sum(x, axis=0, keep_dims=True)`           | `b.sumOpts(x, .{ .axis = 0, .keep_dims = true })`                  |
| `tl.max(x)` / `tl.min(x)`                     | `b.max(x)` / `b.min(x)`                                            |
| `tl.max(x, axis=0)`                           | `b.maxOpts(x, .{ .axis = 0 })`                                     |
| `tl.maximum(a, b)` / `tl.minimum(a, b)`       | `a.maximum(b)` / `a.minimum(b)` or `b.maximum(a, b)`               |
| `tl.cumsum(x)` / `tl.cumprod(x)`              | `b.cumsum(x)` / `b.cumprod(x)`                                     |
| `tl.cumsum(x, axis=0, reverse=True)`          | `b.cumsumOpts(x, .{ .reverse = true })`                            |
| `tl.dot(a, b, acc=acc)`                       | `b.dot(a, b, acc)` — defaults to `input_precision = tf32` (NVIDIA) |
| `tl.dot(a, b, acc, input_precision="ieee")`   | `b.dotOpts(a, b, acc, .{ .input_precision = .ieee })`              |
| `tl.clamp(x, mn, mx)`                         | `b.clampf(x, mn, mx)`                                              |
| `tl.clamp(x, mn, mx, propagate_nan=NAN_ALL)`  | `b.clampfOpts(x, mn, mx, .{ .propagate_nan = .all })`              |
| `tl.atomic_add(p, v)`                         | `b.atomicRmw(.add, p, v)`                                          |
| `tl.atomic_add(p, v, mask=m, sem="relaxed")`  | `b.atomicRmwOpts(.add, p, v, .{ .mask = m, .sem = .relaxed })`     |
| `tl.histogram(x, num_bins)`                   | `b.histogram(x, num_bins)`                                         |
| `tl.histogram(x, num_bins, mask=m)`           | `b.histogramOpts(x, num_bins, .{ .mask = m })`                     |
| `tl.reshape(x, shape)`                        | `b.reshape(x, &.{ ...shape })`                                     |
| `tl.reshape(x, shape, can_reorder=True)`      | `b.reshapeOpts(x, &.{ ...shape }, .{ .can_reorder = true })`       |
| `tl.cast(x, tl.bfloat16)`                     | `b.cast(x, .bf16)` (or fluent `x.to(.bf16)`)                       |
| `tl.cast(x, tl.bf16, fp_downcast_rounding="rtne")` | `b.castOpts(x, .bf16, .{ .fp_downcast_rounding = .rtne })`     |
| `tl.cast(x, tl.int32, bitcast=True)`          | `b.castOpts(x, .i32, .{ .bitcast = true })`                        |
| `src[indices]` (gather)                       | `b.gather(src, indices, axis)`                                     |
| `tl.trans(x, dims)` / `tl.permute(x, dims)`   | `b.trans(x, order)` / `b.permute(x, order)`                        |
| `tl.cat(a, b)`                                | `b.cat(a, b)` (dim=0 default)                                      |
| `tl.cat(a, b, can_reorder=True)`              | `b.catOpts(a, b, .{ .can_reorder = true })`                        |
| `tl.join(a, b)` / `tl.split(x)`               | `b.join(a, b)` / `b.split(x)`                                      |
| `tl.item(x)`                                  | `b.item(x)`                                                        |
| `tl.abs(x)`                                   | `b.abs(x)` (auto-dispatch) or fluent `x.abs()`                     |
| `tl.broadcast(a, b)` (symmetric)              | `b.broadcast(a, b)` → `.{ Value, Value }`                          |
| `tl.broadcast_to(v, shape)`                   | `b.broadcastTo(v, &.{ ...shape })`                                 |
| `tl.umulhi(a, b)`                             | `b.umulhi(a, b)`                                                   |
| `tl.sqrt_rn(x)` / `tl.div_rn(x, y)`           | `b.sqrtRn(x)` / `b.divRn(x, y)`                                    |
| `tl.device_print("dbg", args...)`             | `b.devicePrint("dbg", args, is_signed)`                            |
| `tl.device_assert(cond, "msg")`               | `b.deviceAssert(cond, "msg")`                                      |
| `tl.inline_asm_elementwise(...)`              | `b.inlineAsmElementwise(...)`                                      |
| `tl.extern_elementwise(...)`                  | `b.externElementwise(srcs, shape, dtype, lib, path, sym)`          |
| `for i in range(0, N, S): ...`                | `var loop = b.openFor(0, N, S, .{inits...}); { ...; loop.yield(.{...}); }`                 |
| `while cond: ...`                             | `var w = b.openWhile(.{inits}, .{after_types}); { ...; w.yieldBefore(cond, .{...}); }; { ...; w.yieldAfter(.{...}); }` |
| `if cond: ...` (side-effects, no else)        | `var i = b.openIf(cond); { ...; i.yieldThen(.{}); }`                                         |
| `if cond: ... else: ...`                      | `var i = b.openIfElse(cond, .{result_types}); { ...; i.yieldThen(.{...}); }; { ...; i.yieldElse(.{...}); }` |
| `if cond: return` (early exit)                | `b.returnIf(cond, .{});` — emits `cf.cond_br` + `tt.return`, continues in fall-through block  |
| `if cond: <stores>; return` (early exit, with side-effects) | `var s = b.openReturnIf(cond); { ...; s.yieldReturn(.{}); }` — single `cf.cond_br`; body emits into `^ret` before the `tt.return` |

---

## Pitfalls

1. **Width mismatches between runtime operands.** `v_i32.mul(v_i64)` will hit
   `arith.muli requires the same type` at verify time. Either cast one with
   `.to(...)` or reshape the kernel to stay within a single int width. Comptime
   literals adapt automatically — runtime values don't.

2. **Tensor-of-pointers via `.addPtr`.** Auto-splats either side: scalar
   `!tt.ptr<T>` + tensor offset (`x_ptr.addPtr(tensor_offs)`) and tensor
   `tensor<...x!tt.ptr<T>>` + scalar offset (`a_ptrs.addPtr(BLOCK_K)`)
   both work without manual `.splatTo`.

3. **Scope-based control flow — no closures needed.** `openFor` / `openIf` /
   `openIfElse` / `openWhile` push a block onto the kernel's insertion stack;
   body ops emitted in the bare `{}` that follows go into that blocb. Captures
   are plain lexical scope, so you can reference outer Values, sizes, etc.
   directly — no `ctx` struct, no function literal.

4. **`Value.kernel == null` panics fluent calls.** Only happens if you build a
   `Value` by hand via `.{ .inner = raw }`. Route through `b.emit(...)` to keep
   the back-pointer populated — or just stick with the explicit `Builder.*`
   helpers for that one op.

5. **Don't worry about duplicate constants.** Every lifted literal (from
   `.add(42)`, `.lt(0)`, `b.splat(1.0, shape)`, an `openFor(0, N, 1, …)`
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
