# Compiler segfault: recursive comptime `@typeInfo` on self-referential struct

## Summary

`@setEvalBranchQuota` above a threshold (~1850) causes the compiler to segfault (stack overflow) when a comptime function recursively walks a self-referential struct via `@typeInfo`. Below the threshold the backwards-branch limit fires correctly with a clean error. This is a regression.

## Reproducer

**12 lines, zero dependencies:**

```zig
const A = struct { a: *const A };

fn walk(comptime H: type) void {
    @setEvalBranchQuota(2000);
    switch (@typeInfo(H)) {
        .@"struct" => |i| inline for (i.fields) |f| walk(f.type),
        .pointer => |i| walk(i.child),
        else => {},
    }
}

comptime { walk(A); }
pub fn main() void {}
```

```
$ zig build-exe repro.zig
Segmentation fault (core dumped)
```

## Regression range

| Status | Commit |
|--------|--------|
| **OK** | [`f16eb18ce8c24`](https://codeberg.org/ziglang/zig/commit/f16eb18ce8c24) |
| **BAD** | [`fd2718f82ab70`](https://codeberg.org/ziglang/zig/commit/fd2718f82ab70) |

## Environment

- Zig version: `0.16.0-dev.3132+fd2718f82` (x86_64-linux)
- OS: Linux (x86_64)

## Threshold behavior

The branch quota determines whether the compiler crashes or emits a clean error:

| Quota | Result |
|-------|--------|
| ≤ 1849 | Clean error: *"evaluation exceeded N backwards branches"* |
| ≥ 1850 | **Segmentation fault** (compiler stack overflow) |

With the default quota (1000), the compiler works correctly. The segfault only happens when the quota is raised enough to allow deeper recursion.

## Root cause

`walk(A)` infinitely recurses: `A` → field `a: *const A` → pointer child `A` → field `a: *const A` → ...

The backwards-branch check should catch this regardless of quota. Instead, when the quota is high enough, the compiler exhausts its own call stack and segfaults before the branch counter trips.

## Expected behavior

The compiler should produce *"evaluation exceeded N backwards branches"* for any value of `@setEvalBranchQuota`. It should never segfault.
