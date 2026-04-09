# Zig Compiler Segfault in `zml.meta.visit` on `@Struct`-generated Types

## Summary

After updating the Zig compiler from commit `f16eb18ce8c24ed743aae1faa4980052cb9f4f36` to
`fd2718f82ab70d99186f43845e4291515efb66a5`, `zml.meta.visit` causes a **compiler segfault**
when called on a struct type produced by `MapRestrict(...).map(...)` whose substituted type is
non-trivial (e.g. `zml.Buffer`). The compiler should never segfault regardless of input.

---

## Affected Files

| File | Role |
|---|---|
| `zml/meta.zig` | Contains `visit`, `MapRestrict`, `MapType`, `Contains` |
| Any call site that does `zml.meta.visit(cb, ctx, &value)` where `value`'s type was produced by `MapRestrict` with a non-zero-size `To` type |

---

## Minimal Reproduction

### `main.zig`

```zig
const std = @import("std");
const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    _ = init;

    const LocalContext = struct { index: usize = 0 };
    var ctx: LocalContext = .{};
    _ = &ctx;

    // This type is generated via @Struct() inside MapRestrict.map
    const map_buffer: zml.meta.MapRestrict(zml.Tensor, zml.Buffer).map(zml.nn.Linear) = .{
        .weight = undefined,
        .bias = @as(?zml.Buffer, undefined),
    };

    // This call segfaults the compiler
    zml.meta.visit(struct {
        fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
            _ = b;
            ctx_.index += 1;
        }
    }.call, &ctx, &map_buffer);
}
```

Build with:
```bash
bazel build //examples/repro:repro_not_defined_empty
```

Result: **`zig build-lib` segfaults** (exit code 139 / SIGSEGV).

---

## What Works (and Why)

### 1. `EmptyStruct` substitution works

```zig
const EmptyStruct = struct {};
const map_buffer: zml.meta.MapRestrict(zml.Tensor, EmptyStruct).map(zml.nn.Linear) = .{
    .weight = undefined,
    .bias = undefined,
};
zml.meta.visit(struct {
    fn call(ctx_: *LocalContext, b: *const zml.Buffer) void { ... }
}.call, &ctx, &map_buffer);
```

**Why it works:** `EmptyStruct` has `@sizeOf == 0`. The `visit` function has an early-exit:

```zig
if (comptime !Contains(Ptr, K)) return;
```

Since `K = zml.Buffer` and the struct only contains `EmptyStruct` fields (zero-size, no `Buffer`),
`Contains` returns `false` and `visit` never recurses into the `@Struct`-generated type.
The compiler never needs to deeply instantiate the recursive generic on that type's fields.

### 2. Manually-written struct works

```zig
const bufferized_manual: struct { weight: zml.Buffer, bias: ?zml.Buffer } = .{
    .weight = undefined,
    .bias = @as(zml.Buffer, undefined),
};
zml.meta.visit(..., &bufferized_manual);
```

**Why it works:** This is a normal anonymous struct literal, not produced by `@Struct()` (the
builtin used internally by `MapRestrict.map`). The compiler's type-checking and generic
instantiation paths handle hand-written struct types correctly. The bug is specifically in how
the compiler resolves field types of `@Struct()`-generated types during recursive comptime
generic instantiation.

### 3. `zml.Buffer` substitution via `MapRestrict` **segfaults**

```zig
const map_buffer: zml.meta.MapRestrict(zml.Tensor, zml.Buffer).map(zml.nn.Linear) = .{
    .weight = undefined,
    .bias = @as(?zml.Buffer, undefined),
};
zml.meta.visit(..., &map_buffer);
```

**Why it fails:** The type returned by `.map(zml.nn.Linear)` is produced via `@Struct()` with
reified field descriptors. When `visit` recurses into this type's fields:

```zig
// inside visit, .one => .@"struct" branch:
inline for (s.fields) |field| {
    // ...
    visit(cb, ctx, &@field(v, field.name));
}
```

The compiler must instantiate `visit` for `*zml.Buffer` (the field type). On a hand-written
struct this works fine. On a `@Struct()`-generated type, the new Zig version apparently
mishandles the field type resolution during this recursive comptime instantiation, leading to
a null pointer dereference or similar internal error → **segfault**.

---

## Root Cause

This is a **Zig compiler regression** between:

- ✅ `f16eb18ce8c24ed743aae1faa4980052cb9f4f36` (works)
- ❌ `fd2718f82ab70d99186f43845e4291515efb66a5` (segfaults)

The bug is in the compiler's handling of `@Struct()`-reified types when they are used as the
child type of a pointer in a recursive comptime-generic function instantiation. The compiler
should never segfault — this should either compile successfully or produce a compile error.

---

## Short-Term Workaround in `zml/meta.zig`

The fix avoids the problematic deep recursion by **short-circuiting** in `visit` when a struct
field's type directly matches `K` or `?K`, calling `cb` immediately instead of recursing
through another `visit` instantiation.

Apply this patch to the `visit` function's struct-field iteration (inside the
`.one => .@"struct"` branch):

```zig
// filepath: zml/meta.zig
// Inside: visit() -> ptr_info.size == .one -> @typeInfo(Child) == .@"struct"

.@"struct" => |s| inline for (s.fields) |field| {
    if (field.is_comptime or @sizeOf(field.type) == 0) continue;

    // Workaround: avoid recursive visit instantiation on @Struct()-generated
    // types by short-circuiting when the field is directly K or ?K.
    // This prevents a compiler segfault (Zig regression fd2718f8).
    if (field.type == K) {
        if (can_error)
            try cb(ctx, &@field(v, field.name))
        else
            cb(ctx, &@field(v, field.name));
    } else if (field.type == ?K) {
        if (@field(v, field.name)) |*val| {
            if (can_error)
                try cb(ctx, val)
            else
                cb(ctx, val);
        }
    } else {
        if (can_error)
            try visit(cb, ctx, &@field(v, field.name))
        else
            visit(cb, ctx, &@field(v, field.name));
    }
},
```

This is semantically equivalent to what `visit` would do after one level of recursion (it
would hit the `*const K` / `*K` / `*const ?K` / `*?K` match at the top), but avoids
instantiating `visit(*K)` on the `@Struct()`-generated type which triggers the compiler bug.

### Why this workaround is safe

- `visit` already has top-level checks: `*const K, *K => return cb(ctx, v)` and
  `*const ?K, *?K => return if (v.*) |*val| cb(ctx, val)`.
- The workaround simply performs these same checks one level earlier, at the struct-field
  iteration, skipping the intermediate `visit` instantiation entirely.
- All existing tests (`bazel test //zml:test --test_arg=visit`) should continue to pass.

---

## Upstream Bug Report Checklist

When filing with Zig upstream:

1. **Zig version:** `0.16.0-dev.3132` (commit `fd2718f82ab70d99186f43845e4291515efb66a5`)
2. **Last known working:** commit `f16eb18ce8c24ed743aae1faa4980052cb9f4f36`
3. **Symptom:** Compiler segfault (SIGSEGV) during semantic analysis
4. **Trigger:** Recursive comptime-generic function (`visit`) instantiated on a `@Struct()`-reified
   type whose fields include a complex struct type (`zml.Buffer`)
5. **Key detail:** Same logic on a hand-written anonymous struct with identical field types and
   layout does **not** segfault — only `@Struct()`-produced types trigger it
6. **Minimal repro:** A `@Struct()`-generated type with a non-trivial field, passed to a recursive
   inline-for-over-fields generic function

---

## Verification

After applying the workaround:

```bash
# Should no longer segfault
bazel build //examples/repro:repro_not_defined_empty

# Existing tests should still pass
bazel test //zml:test --test_arg=visit
bazel test //zml:test --test_arg=MapRestrict
bazel test //zml:test --test_arg=mapAlloc
```
