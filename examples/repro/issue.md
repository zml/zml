# Compiler segfault: `@Struct()` reification + cyclic comptime type traversal

## Summary

The Zig compiler segfaults (SIGSEGV) when a comptime function enters infinite recursion through a cyclic type graph, **but only when the types being analyzed were produced by the `@Struct()` builtin**. With non-reified types the branch quota is enforced correctly and a clean error is produced. This is a regression.

## Regression range

| Status | Commit |
|--------|--------|
| **OK** | [`f16eb18ce8c24`](https://codeberg.org/ziglang/zig/commit/f16eb18ce8c24) |
| **BAD** | [`fd2718f82ab70`](https://codeberg.org/ziglang/zig/commit/fd2718f82ab70) |

## Reproducer

Single file, zero external dependencies:

```
zig build-exe repro_simple.zig
```

Output:

```
Segmentation fault (core dumped)
```

(exit code 139)

The file `repro_simple.zig` is attached below.

## Environment

- Zig version: `0.16.0-dev.3132+fd2718f82` (x86_64-linux)
- OS: Linux (x86_64)

## Precise crash path

The repro has these types (simplified from real PJRT/XLA C API bindings in [zml](https://github.com/zml/zml)):

```zig
const Device   = struct { platform: *const Platform };
const Platform = struct { devices: []const Device };   // circular ref
const Buffer   = struct { platform: *const Platform };
```

And a comptime `Contains(H, T)` function that recursively checks whether type `H` transitively contains type `T` in its fields.

The crash call chain is:

1. `visit(cb, &ctx, &m)` — `m` is an `@Struct()`-reified mapped struct with a `weight: Buffer` field
2. `Buffer.platform` is `*const Platform` → `.pointer` → recurse `visit` into `Platform`
3. `Platform.devices` is `[]const Device` → `.pointer` → recurse `visit` with a **slice**
4. `visit` enters the **`.slice` branch**, which guards each field with: `comptime !Contains(f.type, K)` where `K = Buffer`
5. For `Device.platform: *const Platform`, it evaluates `Contains(*const Platform, Buffer)`:

```
Contains(*const Platform, Buffer)
  → .pointer → Contains(Platform, Buffer)
    → .struct → field devices: []const Device
      → Contains([]const Device, Buffer)
        → .pointer → Contains(Device, Buffer)
          → .struct → field platform: *const Platform
            → Contains(*const Platform, Buffer)   ← CYCLE
```

This infinite comptime recursion should be caught by `@setEvalBranchQuota(10_000)` and produce a clean "evaluation exceeded backwards branches" error. Instead, when `@Struct()`-reified types are in the type graph, **the compiler segfaults**.

## Why each simplification avoids the crash

| Change | Why it compiles |
|---|---|
| `Device = struct {}` | Breaks the circular type chain — `Contains` terminates immediately |
| `devices: Device` instead of `[]const Device` | `visit` never enters the `.slice` branch (it's a struct, not a slice), so the `Contains` guard is never evaluated — the infinite recursion never starts |

Both confirm the crash is specifically in the branch-quota enforcement for `@Struct()`-reified types during cyclic comptime evaluation.

## What the code does

The repro is a minimized version of a real pattern from [zml](https://github.com/zml/zml), a Zig ML framework that uses comptime type-mapping to convert model structs (containing `Tensor`) into runtime structs (containing `Buffer`).

1. **`MapRestrict(From, To).map(T)`** — walks struct fields at comptime. For each field whose type contains `From`, it recursively maps it, then reifies a new struct via `@Struct()`.

2. **`Contains(H, T)`** — comptime recursive check: does type `H` transitively contain type `T` in its fields/children?

3. **`visit(cb, ctx, v)`** — comptime-generic recursive visitor that walks struct/pointer/slice/array/optional fields and calls `cb` on every `*const K` leaf.

4. **`main()`** — calls `visit` on a `MapRestrict(Tensor, Buffer).map(Linear)` value. The compiler segfaults during comptime analysis of this instantiation.

## Required ingredients

All of the following are necessary. Removing any one produces a clean compile error or successful compilation:

| Ingredient | Why it matters |
|---|---|
| `@Struct()` reification in `map()` | The bug — replacing with a hand-written struct → branch quota is enforced correctly |
| `@setEvalBranchQuota(10_000)` in `Contains` | Without it, the default 1000-branch limit halts analysis before reaching the crash point |
| `.slice` branch in `visit()` | Only this branch has the `Contains(f.type, K)` guard that triggers the infinite recursion |
| Circular struct references (`Platform ↔ Device`) | Creates the cyclic type chain that makes `Contains` loop forever |

Things that are **not** required (already stripped during minimization):
- Self-referential `[*c]` extern structs
- Function pointers
- `std.fmt.comptimePrint` in `@compileError` messages
- Error union / `VisitReturn` handling
- Any external dependency

## Expected behavior

The compiler should enforce `@setEvalBranchQuota` and produce a clean "evaluation exceeded N backwards branches" error (same as it does with non-`@Struct()`-reified types). It should never segfault.
