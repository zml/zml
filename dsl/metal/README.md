# ZML Metal DSL

A Zig builder for Metal kernels that reads like MSL. It emits the `air`
dialect (`mlir/dialects/air`) and the parts of `arith`, `cf`, `vector` and
`llvm` that MSL code compiles to, shaped as clang's codegen shapes it: locals
in memory, blocks and branches. `zml.kernel.metal.Kernel` wraps a kernel into
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
| `const T x = e;`, `constexpr T x = c;` | `const x = b.declare(e)`, `b.declareConstant(.T, c)` |
| `{ ... }` | `const s = b.openScope()`, `s.close()` |
| `device T *p = e;` that the code moves | `const p = b.localPointer(e)`, `p.assign(p.add(i))` |
| `float s[8];`, `float s[8] = {};` | `const s = b.array(.float, 8)`, `b.zeroedArray(.float, 8)`, then `s.load(i)` and `s.store(i, v)` |
| `simdgroup_matrix<T, 8, 8> m;` | `const m = b.matrix(.T)` |
| `&x` of a local | `x.address()` |
| `float4(a, b, c, d)`, `1.0f` | `b.vec(.float4, .{ a, b, c, d })`, `b.constant(.float, 1)` |
| `c ? x : y` | `b.ternary(c, x, y)` of evaluated operands; `var t = b.openIfElse(c, .{.T})`, `x`, `t.yieldThen(.{x})`, `y`, `t.yieldElse(.{y})`, `t.results[0]` |
| `exp(x)`, `fast::exp(x)`, `precise::tanh(x)` | `b.exp(x)`, `b.fast().exp(x)`, `b.precise().tanh(x)` |
| `fma`, `dot`, `max`, `clamp` | `b.fma`, `b.dot`, `b.max`, `b.clamp` |
| `simd_sum(x)`, `simd_shuffle_xor(x, m)` | `b.simdSum(x)`, `b.simdShuffleXor(x, m)` |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `b.threadgroupBarrier(.mem_threadgroup)` |
| `simdgroup_load(m, p, ld, origin, transpose)`, `simdgroup_store(m, p, ld)` | `b.simdgroupLoad(m, p, ld, .{ .origin = ..., .transpose = true })`, `b.simdgroupStore(m, p, ld, .{})` |
| `simdgroup_multiply_accumulate(d, a, b, c)` | `b.simdgroupMultiplyAccumulate(d, a, b, c)` |
| `reinterpret_cast<thread vec<T, 2> &>(m.thread_elements())` | `m.threadElements()`, read or assigned |

## Control flow

| MSL | DSL |
|---|---|
| `for (T i = lower; i < upper; i += s)`, `++i` | `var l = b.openFor(lower, upper, s)`, `l.iv`, `l.close()`; a step of 1 is `++i` |
| `#pragma unroll`, `#pragma clang loop unroll(full)` | `b.openForOpts(..., .{ .unroll = .enable })`, `.{ .unroll = .full }` |
| `while (c) { ... }`, `for (init; c; inc)`, `for (;;)` | `var l = b.openLoop(.{})`, `l.body(c)`, the body, `l.next()`, the increment, `l.close()`; `.init = n` for the `n` locals `init` declares |
| `break;` | `l.breakLoop()` |
| `if (c) { ... }` | `var s = b.openIf(c)`, `s.close()` |
| `if (c) { ... } else { ... }` | `var s = b.openIfElse(c, .{})`, `s.elseBranch()`, `s.close()` |
| `if (a \|\| (b && !x))` | `var c = b.condition()`, `c.or_(a)`, `c.and_(b)`, `var s = c.openIfNot(x)` |
| `x = a && b && c` | `var l = b.openAnd()`, `l.term(a)`, `l.term(b)`, `l.close(c)` |
| `return;`, `return v;` before the end | `b.return_()`, or `b.returnIf(c)`; `b.returnValue(v)` |
| `static void f(...)`, `inline T f(...)` | `var f = try b.openFunction("f", .{ .x = .{ .value = .float } }, result, .{})`, `f.args`, `const fun = f.close(r)`; `b.call(fun, .{...})`; `.assigned` names the parameters the body assigns |

Put each `if` or loop body in a Zig `{ ... }` block right after the opener,
one block per region, each ending with the call that ends the region:
`elseBranch()` or `yieldThen`, then `close()` or `yieldElse` for an `if`;
`body(c)`, `next()`, then `close()` for `openLoop`. The scope variable stays
outside, so its `results` stay readable after the blocks.

Locals live in memory until the pipeline promotes them, as in clang's
codegen: every parameter and every local, `const` and `constexpr` ones too,
gets a slot in declaration order, and each local's lifetime runs from its
declaration to the end of its scope. The pipeline promotes the slots in that
order, which decides the order of the phis; declare every local the MSL
declares. A value a loop or branch changes is a local the code assigns. A
`break` or `return` past live locals goes through clang's cleanup blocks and
its switch on `cleanup.dest.slot`.
The terms of `&&`, `||` and `?:` run only when the ones before leave the
result open, each in its own block. Loops are built as C++ builds them:
every loop is `mustprogress`, its condition and increment follow C++'s
conversions, so a `short` loop variable compares as an `int` and `++i`
steps it as a `short`. Every `return` branches to one return block; a
function that returns a value before its end keeps the value in clang's
`retval` slot.

A `simdgroup_matrix` local is a slot of its 64 elements, and the load, store
and multiply-accumulate call the `air.simdgroup_matrix_8x8_*` builtins on it
directly. Clang goes through the header's `always_inline` functions instead;
after the pipeline, the code is the same, with its phis in another order.

## Porting an MSL kernel bit for bit

Fast math lets Apple's backend reassociate, contract and reorder float math, so
the bits it produces depend on the shape of the IR it gets, not only on its
meaning. A port matches its MSL kernel when it writes the code the way the MSL
does:

- Keep the MSL's control flow and its locals: a `break` is `breakLoop`, `&&`
  and `||` short-circuit, a `?:` whose arms compute something
  (`c ? 0 : fast::exp2(x)`) is `openIfElse(c, .{T})`, and an MSL function is a DSL
  function. `ternary` is for operands that are already values.
- Evaluate in C++'s order: a store's value before its index, a compound
  assignment's right side before its left.
- Keep its loops and pragmas: a rolled loop with `#pragma unroll` stays a rolled
  loop with `.unroll`. The backend unrolls it the way it unrolls the MSL loop.
- Index arrays of vectors as vectors: `float2 S[TK]` read as `S[k][jj]` is
  `s.load(k).at(jj)`. Writes go through `insert` and a store of the whole
  vector, not through a reinterpreted scalar array.
- Keep pointer arithmetic as written: `shmem + a + b` is
  `shmem.add(a).add(b)`, two steps as in C++, not `shmem.add(a.add(b))`.
- Read struct fields where the MSL reads them: `args.scale` inside a loop is
  `a.args.field("scale")` inside the loop, not a value read once at the top.
- Keep the dispatch: the same grid, threadgroup size, threadgroup memory and
  kernel split as the MSL kernel's host code.

XLA lowers the module without folding or erasing anything, as clang's
codegen keeps everything until the pipeline runs: the pipeline's result
depends even on dead code.

To find a mismatch, compare the IR. `metal -S -emit-llvm` on the MSL source
(function constants turned into `constexpr`) gives Apple's IR. The custom
call's AIR, run through `air-opt -O2 -disable-loop-unrolling -S`, gives the
port's. Then diff the order of the float operations.
