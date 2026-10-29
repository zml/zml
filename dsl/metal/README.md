# ZML Metal DSL

A Zig builder for Metal kernels that reads like MSL. It emits the `air`
dialect (`mlir/dialects/air`) and the parts of `arith`, `cf`, `vector` and
`llvm` that MSL code compiles to: values, locals in memory, blocks and
branches. `zml.kernel.metal.Kernel` wraps a kernel into
a `__gpu$xla.gpu.air` custom call, and XLA does the rest: it lowers the module to
Apple's AIR, runs the Metal frontend's optimization pipeline
(`air-opt -O2 -disable-loop-unrolling`) and builds a metallib. Apple's
backend compiles that metallib the way it compiles an MSL kernel.

## Skeleton of a kernel

```zig
const metal = @import("zml").kernel.metal;

pub const Axpy = struct {
    pub const Cfg = struct {};
    pub const Kernel = metal.Kernel(Cfg, .{
        .name = "axpy",
        .inputs = &.{ "a", "b" },
        .outputs = &.{"out"},
        .constants = &.{ "scale", "n" },
        .run = run,
    });

    fn run(b: *metal.Builder, _: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .a = .{ .buffer = .{ .type = .float, .const_ = true } },
            .b = .{ .buffer = .{ .type = .float, .const_ = true } },
            .out = .{ .buffer = .{ .type = .float } },
            .scale = .{ .constant = .float },
            .n = .{ .constant = .uint },
            .tid = .{ .thread_position_in_grid = .uint },
        });
        b.returnIf(a.tid.ge(a.n));
        a.out.store(a.tid, b.fma(a.a.load(a.tid), a.scale, a.b.load(a.tid)));
    }
};

// In a model:
const out = Axpy.Kernel.call(.{ .a = x, .b = y }, .{ .out = x.shape() }, .{
    .cfg = .{},
    .grid = .{ 4, 1, 1 },
    .threadgroup = .{ 256, 1, 1 },
    .constants = .{ .scale = .of(@as(f32, 0.5)), .n = .of(@as(u32, 1000)) },
}).out;
```

`declareArgs` declares the kernel parameters in order, which is the `[[buffer(n)]]`
order. `inputs` and `outputs` bind buffers to tensors by name. An output named
like an input updates that input's buffer in place. `constants` bind
`constant T &` arguments by value. `Constant.of` takes a scalar, or a host
`extern struct` laid out like the MSL struct. `CallOpts.threadgroup_memory`
sizes `[[threadgroup(0)]]`.

Function constants have no runtime form. A kernel takes them through its `Cfg`,
so each configuration emits its own specialized kernel.

## Arguments

| MSL | `ArgSpec` |
|---|---|
| `device T *p`, `const device T *p` | `.buffer = .{ .type = T, .const_ = true }` |
| `constant T &x` | `.constant = T`, read wherever it is used, as C++ reads a reference |
| `constant S &s` | `.constant_struct = S`, read with `s.field("name")` |
| `threadgroup T *p [[threadgroup(0)]]` | `.threadgroup = T` |
| `uint3 tid [[thread_position_in_grid]]` and the other builtins | `.thread_position_in_grid = .uint3`, ... |

## Values

A `Value` is a scalar or vector (`Type`, spelled as in MSL: `.float4`,
`.ushort`), a pointer, a private array, a `simdgroup_matrix` or a constant
struct. Operations follow C++: integer promotion and the usual arithmetic
conversions pick the result type, signed arithmetic is `nsw`, and float math
carries the kernel's fast-math flags. A Zig number is the MSL literal it
spells, an `int` (a `long` when it does not fit) or a `float`, and a runtime
Zig integer keeps its width and signedness; the usual conversions follow, so
`h.mul(0.5)` on a `half` is a `float` multiply, as `h * 0.5` is in MSL. A
scalar next to a vector takes its element type. Stores, assignments and the
functions of one type (`max`, `min`, `clamp`, `fma`) convert a number to their
type. `0.5h` is `b.constant(.half, 0.5)`.

| MSL | DSL |
|---|---|
| `a + b`, `a * b`, `a < b`, `a & b` | `a.add(b)`, `a.mul(b)`, `a.lt(b)`, `a.bitAnd(b)` |
| `(float) x`, `float4(v)` | `x.to(.float)`, `v.to(.float4)` |
| `as_type<uint>(x)` | `x.asType(.uint)` |
| `v.x`, `v[i]`, `v[i] = e`, `v.y = e` | `v.x()`, `v.at(i)`, `v = v.insert(i, e)`, `v = v.insert(1, e)`: a subscript takes the index's type (a literal is an `int`) |
| `p[i]`, `p[i] = v`, `p + i` | `p.load(i)`, `p.store(i, v)`, `p.add(i)` |
| `(threadgroup float4 *) p` | `p.to(.float4)` |
| `T x;`, `T x = e;`, `x = e` | `const x = b.local(.T, null)`, `b.local(.T, e)`, `x.assign(e)` |
| `device T *p = e;` that the code moves | `const p = b.localPointer(e)`, `p.assign(p.add(i))` |
| `float s[8];`, `float s[8] = {};` | `const s = b.array(.float, 8)`, `b.zeroedArray(.float, 8)`, then `s.load(i)` and `s.store(i, v)` |
| `&x` of a local | `x.address()` |
| `float4(a, b, c, d)`, `1.0f` | `b.vec(.float4, .{ a, b, c, d })`, `b.constant(.float, 1)` |
| `c ? x : y` | `b.ternary(c, x, y)`, a select of evaluated operands; `var t = b.openIfElse(c, .{.T})`, `x`, `t.yieldThen(.{x})`, `y`, `t.yieldElse(.{y})`, `t.results[0]` |
| `exp(x)`, `fast::exp(x)`, `precise::tanh(x)` | `b.exp(x)`, `b.fast().exp(x)`, `b.precise().tanh(x)` |
| `fma`, `dot`, `max`, `clamp` | `b.fma`, `b.dot`, `b.max`, `b.clamp` |
| `simd_sum(x)`, `simd_shuffle_xor(x, m)` | `b.simdSum(x)`, `b.simdShuffleXor(x, m)` |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `b.threadgroupBarrier(.mem_threadgroup)` |
| `simdgroup_load(m, p, ld, origin, transpose)`, `simdgroup_store(m, p, ld)` | `const m = b.simdgroupLoad(p, ld, .{ .origin = ..., .transpose = true })`, `b.simdgroupStore(m, p, ld, .{})` |
| `simdgroup_multiply_accumulate(d, a, b, c)` | `const d = b.simdgroupMultiplyAccumulate(a, b, c)` |
| `m.thread_elements()` read, assigned | `m.threadElements()`; `const m = b.matrix(.T, elements)` |

## Control flow

| MSL | DSL |
|---|---|
| `for (T i = lower; i < upper; i += s)`, `++i` | `var l = b.openFor(lower, upper, s)`, `l.iv`, `l.close()`; a step of 1 is `++i` |
| `#pragma unroll`, `#pragma clang loop unroll(full)` | `b.openForOpts(..., .{ .unroll = .enable })`, `.{ .unroll = .full }` |
| `while (c) { ... }`, `for (init; c; inc)`, `for (;;)` | `var l = b.openLoop(.{})`, `l.body(c)`, the body, `l.next()`, the increment, `l.close()` |
| `break;` | `l.breakLoop()` |
| `if (c) { ... }` | `var s = b.openIf(c)`, `s.close()` |
| `if (c) { ... } else { ... }` | `var s = b.openIfElse(c, .{})`, `s.elseBranch()`, `s.close()` |
| `if (a \|\| (b && !x))` | `var c = b.condition()`, `c.or_(a)`, `c.and_(b)`, `var s = c.openIfNot(x)` |
| `x = a && b && c` | `var l = b.openAnd()`, `l.term(a)`, `l.term(b)`, `l.close(c)` |
| `return;` | `b.return_()`, or `b.returnIf(c)` |

Put each `if` or loop body in a Zig `{ ... }` block right after the opener,
one block per region, each ending with the call that ends the region:
`elseBranch()` or `yieldThen`, then `close()` or `yieldElse` for an `if`;
`body(c)`, `next()`, then `close()` for `openLoop`. The scope variable stays
outside, so its `results` stay readable after the blocks.

A value is evaluated where it is written; a Zig `const` holds it, and an MSL
function is a Zig function that emits its body. A value a loop or branch
changes is a local, which lives in memory until the pipeline promotes it.
The terms of `&&`, `||` and `openIfElse(c, .{T})` run only when the ones
before leave the result open, each in its own block. Loops follow C++: every
loop is `mustprogress`, and its condition and increment follow C++'s
conversions, so a `short` loop variable compares as an `int` and `++i` steps
it as a `short`.

A `simdgroup_matrix` is a value of its 64 elements, of which a lane holds the
first two; the load, store and multiply-accumulate call the
`air.simdgroup_matrix_8x8_*` builtins.

## Checking a port

Fast math lets Apple's backend reassociate and contract float math, so a port
computes what its MSL kernel computes up to rounding, not always to the bit.
Compare its outputs with the MSL kernel's on the same inputs, and time both
with the same grid, threadgroup size and threadgroup memory.
