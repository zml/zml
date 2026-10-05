# TCL kernels in Zig

`kernels/tcl` is a Zig front end for Furiosa's TCL (Tensor Contraction
Language), shaped after the Python `furiosa.tcl` DSL: a kernel declares named
axes and typed tensors, opens tensor operations that `fetch`, `contract`, run
VE instructions and `commit`, and returns tensors. The `Builder` emits the
native `tcl` MLIR dialect (`mlir/dialects/tcl`), and
`zml.kernel.tcl.Kernel` turns it into a `furiosa.tcl` custom call that the
Furiosa plugin exports to TCL source and compiles with `furiosa-tcc` as one
native unit.

## A kernel

`furiosa.tcl.examples.softmax`, in Zig:

```zig
const zml = @import("zml");
const tcl = zml.kernel.tcl;

const Cfg = struct { b: i64, s: i64, d: i64 };

pub const Softmax = tcl.Kernel(Cfg, .{
    .name = "softmax_kernel",
    .inputs = &.{"input"},
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
    const B = b.axis("B", cfg.b);
    const S = b.axis("S", cfg.s);
    const D = b.axis("D", cfg.d);
    const t = try b.declareArgs(.{ .input = .{ .dtype = .bf16, .axes = &.{ B, S, D } } });

    const op = b.tensorOperation(.{});
    const x = op.fetch(t.input, .{ .typecast_to = .f32 });
    const exped = x.subf(x.reduce(&.{D}, .maxf)).exp();
    const out = try op.commit(exped.divf(exped.reduce(&.{D}, .addf)), .{ .dtype = .bf16 });
    b.ret(&.{out});
}

// In a model:
const y = Softmax.call(.{ .input = x }, .{ .out = x.shape() }, .{
    .cfg = .{ .b = 2, .s = 16, .d = 1024 },
}).out;
```

The plugin compiles it to the same TCL the Python DSL prints:

```
def softmax_kernel(B: Axis = 2, D: Axis = 1024, S: Axis = 16, v0: [B, S, D]/bf16) -> [B, S, D]/bf16:
    v1: [B, S, D]/bf16 = tk.ReduceByVe(
        %0 = read(v0, typecast_to=f32)
        %1 = ve.exec(max_f t[D] %0)
        %2 = ve.exec(%0 - %1)
        %3 = ve.exec(exp %2)
        %4 = ve.exec(+ t[D] %3)
        %5 = ve.exec(%3 / %4)
        write.to([B, S, D]/bf16)
    )
    return v1
```

## Python to Zig

| `furiosa.tcl`                                  | `kernels/tcl`                                         |
|------------------------------------------------|-------------------------------------------------------|
| `x: tcl.Tensor[tcl.bf16, [B, S, D]]`            | `.x = .{ .dtype = .bf16, .axes = &.{ B, S, D } }`     |
| `x: tcl.Dram[tcl.bf16, [tcl.Broadcast], [M, N]]` | `.x = .{ .dtype = .bf16, .axes = &.{ M, N }, .dram = .{} }` |
| composite axis `C: Axis = (A * B)`             | `b.axis("C", b.expr(.mul, A, B))`                     |
| `@tcl.kernel(config={...}, auto_config={...})` | `b.compilerConfig(.{ ... })`, `b.autoConfig(.{ ... })` |
| `@tcl.tensor_operation`                        | `const op = b.tensorOperation(.{})`                   |
| `fetch.typecast(x, tcl.f32)`                   | `op.fetch(x, .{ .typecast_to = .f32 })`               |
| `fetch.broadcast(x, ...)`, `fetch.table_lookup` | `.broadcast_to = &.{...}`, `.table_lookup = table`   |
| `fetch.pad(x, axis=L, left=1, right=1)`        | `.pad = &.{.{ .axis = L, .left = 1, .right = 1 }}`    |
| `fetch.slide(x, axis=L, frame_axis=Lf, window_axis=Lw)` | `.slide = &.{.{ .axis = L, .frame_axis = Lf, .window_axis = Lw }}` |
| `operation.contract(lhs, rhs, to=[B, M, N])`   | `op.contract(lhs, rhs, &.{ B, M, N })`                |
| `operation.subf(a, b)`, `operation.exp(a)`     | `a.subf(b)`, `a.exp()`, or `op.ve(.subf, .{ a, b })`  |
| `operation.reduce(x, axes=[D], op=tcl.Maxf)`   | `x.reduce(&.{D}, .maxf)`                              |
| `operation.where(x > 0, a, b)`                 | `op.where(x, .gt, 0, a, b)`                           |
| `operation.to_fp(x, 15)`, `to_fxp(x, 15)`      | `x.toFp(15)`, `x.toFxp(15)`                           |
| `compiler_hints=tcl.context(layout={Chip: ...})` | `.context = .{ .layout = .{ .chip = .broadcast } }`  |
| `commit.typecast(v, tcl.bf16)` / `commit.permute` | `op.commit(v, .{ .dtype = .bf16, .axes = ... })`   |
| `tcl.reshape`, `transmute`, `concat`, `slice`  | `b.reshape`, `b.transmute`, `b.concat`, `b.slice`     |
| `tcl.gather`, `scatter`, `arange`, `vector`, `all_gather` | `b.gather`, `b.scatter`, `b.arange`, `b.vector`, `b.allGather` |
| `for i in tcl.range(0, n):` (carried `acc`)    | `var loop = b.openFor(n, .{acc})` ... `try loop.yield(.{new_acc})`, then `loop.results` |
| `x[i]`, `out[i] = v`, `tcl.scratchpad()`       | `b.indexRead(x, loop.iv)`, `b.indexWrite(out, loop.iv, v)`, `b.scratchpad` |
| `core.reduce_max_i32(t)`, `core.eval_expr(...)` | `b.reduceMaxI32(t)`, `b.symExpr(.div, n, 64)`        |
| `return out`                                   | `b.ret(&.{out})`                                      |

* **Axes are static.** `b.axis(name, size)` declares `name: Axis = size`,
  with an integer or an `Expr` over other axes; axis names start with an
  uppercase letter. `furiosa-tcc` needs every axis bound. Kernel inputs and
  outputs must match the XLA shapes axis by axis (row-major), which `call`
  checks.
* **DRAM tensors.** A `.dram` argument comes back as its `as_logical` view;
  `b.asDram(t, .{...})` returns a result in a DRAM mapping.
* **Multi-chip kernels.** A kernel that places data or work on chips (a
  chip-split DRAM tensor, a `.context`, `allGather`) is compiled by the plugin
  for all the chips of its SPMD program (2 or 4), and runs as one launch
  across the partitions; other kernels stay per chip. Its arguments and
  results are DRAM tensors: `Dram{ .chip = &.{.{ .axis = C }} }` splits axis
  `C`, sized to the chip count, across chips (each partition holds its share),
  the default `.broadcast` keeps a copy per chip. Call it inside
  `zml.ops.manualComputation` so it sees each partition's shards.
* **Convolutions** pad and slide the input in its read: a slide replaces its
  axis by `window_axis, frame_axis`, then `contract` with the filter. On its
  own (without a slide) the padding is dropped.
* **The mainstream rule.** As in the Python DSL, instructions are pipelined:
  `commit` must take the last instruction, there is at most one contraction
  and it comes before VE instructions. Reads are hoisted in front.
* **VE registers are f32 or i32**, decided by the instruction. Scalar
  operands become `f32`/`i32` constants (`x.mulf(0.5)`).
* **Tactics are guessed** like `@tcl.tensor_operation` (`EinsumByDpe`,
  `ReduceByVe`, `Interleaving`, ...); `b.tensorOperation(.{ .tactic = ... })`
  overrides it.
* **Tensors passed directly** to an instruction are read without modifiers.
* **Constant outputs.** `furiosa-tcc` 2026.3 fails ("No producer") on a
  function returning two input-independent tensors, such as `arange` and
  `vector`.
* **Loops** run on the device. `x[i]` and `acc[i] = v` address the outermost
  axis, and the body must index something with `i`: that bounds the trip
  count. A dynamic trip count must be computed in the kernel (`reduceMaxI32`,
  `symExpr`); one read straight from an argument adds a CPU node the plugin
  rejects.

## Good practices on RNGD

The TCL CPU interpreter (`furiosa.torch.TclModule`) accepts several programs
that compute garbage on the hardware: check kernels on RNGD.

* **One VE stream per tensor operation.** Each instruction takes the previous
  result and one side operand (a read or a constant). A select whose branches
  are both constants must test the operation's first read.
* **Outer broadcasts** (`[B, H, V] * [B, H, K]` to `[B, H, V, K]`, tactic
  `EinsumByVe`) need a shared leading axis, and the read with more axes first.
* **Contraction epilogues** read their extra tensors before the contraction
  and combine the contraction result first: `contract(q, k).addf(bias)`.
* **Conversions.** `typecast_to` converts floats only; `toFp(n)` of an i32
  yields `x / 2^31` whatever `n`.
* **Update caches in place.** `scatter(rows, idx, axis, .{ .init = cache })`
  on a kernel argument shares its buffer with the result (a whole-cache copy
  otherwise), so return it as an aliased output.
* **Few axes, few operations.** tcc's tactic search grows with the number and
  extents of axes, and more than linearly with the number of operations in a
  function: merge axes before contractions (`[N, H, G*Q, C]`, not
  `[N, H, G, Q, P, C]`) and set `.lowering_mode = "Heuristic"` above a few
  tokens.
* **Prefer one-shot graphs to hand-written online softmax.** Furiosa's
  production attention gathers the context, contracts, masks with a select,
  runs the softmax as one `ReduceByVe` and contracts again; gathers take
  `valid_length` to skip padding, and padding indices are spread
  (`idx | (pos % n)`) rather than all pointing at one row.
* **Furiosa's kernels** in the SDK (`furiosa/kernels/`) are the reference for
  layouts, tile sizes and compiler configs: `common/attention.py` (attention
  and decode/prefill configs), `common/topk.py` and
  `common/blockwise_moe_index.py` (routing on the device),
  `qwen3_moe/optimized/` (blockwise MoE for prefill, activation-stationary MoE
  for decode), `exaone_moe/k_exaone_w4fa16kv16/flash_attention.py` (chunked
  softmax over a dynamic trip count).

## The custom call

`ops.tcl` emits `stablehlo.custom_call @furiosa.tcl` with the printed module
as its backend config. The plugin parses it, checks the function signature
against the operand and result shapes, exports TCL source and compiles it
with the rest of the program's units. With `--xla_dump_to=<dir>`,
`*.furiosa-unit-sources.txt` holds each unit's TCL.
