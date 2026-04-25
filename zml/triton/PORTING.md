# Porting Triton kernels from Python to the Zig DSL

This guide covers the day-to-day workflow for **adding a new Triton kernel
to ZML** (port from a Python `@triton.jit` source) and **fixing an existing
Zig kernel that diverges from Python** (using the comparison harness in
`examples/triton_emitter/`).

## Goals (in priority order)

1. **Lowered code matches.** The Zig version produces **the same LLVM IR +
   PTX** as the Python version. TTIR/TTGIR may differ cosmetically (op
   ordering, naming) — what matters is what actually runs on the GPU.

2. **Source code looks alike.** A Python kernel and its Zig port should be
   *visually transparent* side by side: same overall structure, same
   variable names, same comment placement, same line ordering. Someone
   reading both should be able to follow them in lockstep.

   Visual fidelity matters because:
   - It makes future ports trivial — patterns transfer one-to-one.
   - It makes upstream-Triton patches easy to apply (the Python diff
     translates almost mechanically to a Zig diff).
   - It catches porting bugs before they reach the comparison harness:
     when the Zig code reads differently from Python, that's where the
     real-vs-cosmetic divergence usually hides.

   Concretely: if a Python source line is
   ```python
   y_ptr_offset = (row.to(tl.int64) * y_row_stride) + (row_g_id.to(tl.int64) * group_size)
   ```
   the Zig port should be one line, named the same way:
   ```zig
   const y_ptr_offset = row.to(.i64).mul(y_row_stride).add(row_g_id.to(.i64).mul(group_size));
   ```
   Not split across three temporaries with renamed locals — even if the
   resulting LLIR is identical. Goal #1 is a hard requirement; goal #2
   is what makes this codebase maintainable.

For the DSL surface (`zml.Kernel`, `Builder`, `Value`, etc.), see
[`README.md`](./README.md). This file is about the *process*, not the API.

---

## Prerequisites

- CUDA or ROCm GPU (XLA's Triton backend is GPU-only).
- The Python venv at `examples/triton_emitter/.venv` activated, or a
  system-level Python with `triton`, `torch`, `numpy` installed.
- Bazel + the standard ZML build deps.

One-time:

```sh
cd examples/triton_emitter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Phase 1 — Initial port

Start by copying the Python kernel verbatim into the Zig DSL form. The
goal is a kernel that *compiles*, *verifies*, **and reads as similarly
to the Python source as possible**.

> **Author hint:** open the Python file in a left pane and the Zig file
> in a right pane. Translate **line by line**, top to bottom. Keep
> variable names, comment placement, and intermediate-binding granularity
> identical wherever the DSL allows. Don't refactor, don't rename, don't
> "improve" — that's a separate task you do after the LLIR matches.

### Step 1.1 — Drop the Python source next to the Zig file

For comparison purposes, the Python source has to live under
`examples/triton_emitter/kernels_py/`. The dump script auto-discovers
every `@triton.jit` function in that directory.

If your kernel's already in `vllm/` or another upstream tree, copy the
relevant `@triton.jit` definition (and any `@triton.jit` helpers it
calls) into a single `.py` file there. Strip away launch wrappers,
autotune decorators, and any code that isn't actually inside the kernel
function.

### Step 1.2 — Sketch the Zig declaration

Inside `zml/moe/triton_kernels.zig` or `zml/attention/triton_kernels.zig`
(or whichever production file fits), add:

```zig
pub const MyKernel = zml.Kernel(.{
    .name = "my_kernel",                          // EXACTLY matches the Python `def my_kernel`
    .config = struct {
        // One field per `tl.constexpr` parameter. No defaults — every
        // field is required at call time. Use the same names Python uses,
        // converted to snake_case.
        BLOCK_SIZE: usize,
        // ...
    },
}, struct {
    pub fn run(b: *tri.Builder, cfg: anytype) !void {
        const a = try b.declareArgs(.{
            // One field per *runtime* parameter (pointers, dims that
            // arrive as 0-d tensors). Use the same names + order Python
            // uses; the cheat-sheet's "argument specs" table maps the
            // tagged-union variants.
            .x_ptr = .{ .ptr = .f32 },
            // ...
        });

        // Body — translate Python line by line. Cheat-sheet in the
        // README's "Python Triton ↔ Zig DSL cheat-sheet" section is
        // your best friend here.
    }
});
```

### Step 1.3 — Wire the production caller

Inside the same `.zig` directory, find the function that launches your
kernel (e.g. `pagedAttention2d`, `fusedExpertsImpl`). Replace the
TTIR-emit + `ops.triton(...)` two-step with a single
`MyKernel.call(.{ ... })` — see the [README](./README.md#calling-kernels-from-a-zml-model--kcall)
for the exact shape.

If this is a brand-new kernel with no existing caller, write the caller
fresh.

### Step 1.4 — Wire the comparison harness

Add the kernel to `examples/triton_emitter/kernels_zig.zig`:

```zig
withConfig(moe.MyKernel, .{ .BLOCK_SIZE = 64, /* ... */ }),
```

If the kernel takes runtime tensor inputs, also add a `forward` struct +
`KERNELS` entry to `examples/triton_emitter/dump_via_xla.zig` so XLA's
codegen pipeline can lower it. The tensor sizes are dummies — XLA
doesn't launch the kernel, it just runs the lowering passes.

For the Python side, the auto-discovery in `dump_python_ir.py` works for
simple kernels (only `*_ptr` runtime args + `tl.constexpr` params).
**If the signature mixes runtime args and constexpr in the middle of the
parameter list**, write an override function and add it to `_OVERRIDES`
(follow `_per_token_group_quant_fp8` / `_kernel_unified_attention_2d_ptr`
as templates). Use the same constexpr values + tensor shapes you used in
the Zig `withConfig(...)` call.

### Step 1.5 — Build it

```sh
bazel build //examples/triton_emitter:dump_zig_ir   # Zig side compiles + verifies
bazel build //zml:test                              # production side compiles
```

If the Zig DSL `b.finish()` rejects your IR, the most common causes are:
- **Width mismatches** in `arith.muli` / `addi` (e.g. `i32` × `i64`).
  Cast one operand with `.to(.i64)` so both match.
- **Tensor-of-pointers `addPtr` requires same shape** — when the offset
  is a tensor, splat the pointer first (`scalar_ptr.addPtr(tensor_off)`
  auto-splats; `tensor_ptrs.addPtr(scalar)` doesn't — pass
  `tensor_ptrs.addPtr(b.splat(s, &.{shape}))`).
- **`scf.yield` arity** — the tuple you pass to `loop.yield(...)` /
  `i.yieldThen(...)` must match the loop's `inits` / `result_types`
  arity exactly.

### Step 1.6 — First dump

```sh
bash examples/triton_emitter/run.sh
```

This runs the full pipeline (Python TTIR → XLA → per-stage IR; same on
the Zig side; diff each stage). At the end you get a per-kernel
verdict:

```
my_kernel
  ttir      +30/-30
  ttgir     +30/-30
  llir      +50/-50
  ptx       +40/-40
```

Don't worry yet about cosmetic diffs — go through Phase 2 to find
*real* divergences first.

---

## Phase 2 — The comparison pipeline

The harness has six steps (see `run.sh`), output appears under
`examples/triton_emitter/`:

| Step | Output                       | What it is                                    |
|------|------------------------------|-----------------------------------------------|
| 1    | `py_ir/*.ttir`/`.ttgir`/...  | Python's per-stage IR (Triton's own pipeline) |
| 2    | `zig_ir/*.ttir`              | Just the TTIR your DSL emits.                 |
| 3    | `xla_dump_py/`               | XLA's per-pass dumps lowering Python's TTIR.  |
| 4    | `xla_dump_zig/`              | XLA's per-pass dumps lowering the Zig TTIR.   |
| 5    | `xla_py/*.{ttir,ttgir,llir,ptx}` + `xla_zig/*.{...}` | Per-stage IR extracted by `extract_xla_dump.py`. Both sides go through XLA, not Triton-the-Python-frontend, so the comparison is apples-to-apples. |
| 6    | stdout                       | `compare_ir.py` diff summary per kernel.      |

### Useful flags

```sh
# Just one kernel (faster iteration):
# (use ZIG_IR/PY_IR filter — but note dump_zig_ir's CLI expects --kernel=NAME).
bash run.sh                                    # full run
python compare_ir.py --left xla_py --right xla_zig --kernel my_kernel
python compare_ir.py --left xla_py --right xla_zig --kernel my_kernel --stage ttir --show-diffs

# Skip the slow XLA-lowering steps if only the TTIR changed:
python dump_python_ir.py --out-dir py_ir
bazel run //examples/triton_emitter:dump_zig_ir -- --out-dir=$PWD/zig_ir
diff py_ir/my_kernel.ttir zig_ir/my_kernel.ttir
```

### Reading the verdict

```
my_kernel
  ttir      +30/-30      <- TTIR diverges 30 lines
  ttgir     +30/-30      <- so does TTGIR (same shape)
  llir      ✓            <- but LLIR converges! cosmetic-only TTIR diff
  ptx       ✓            <- and PTX matches
```

This kernel is **good** — final code is identical, the TTIR diff is just
op-naming/ordering cosmetics that XLA's lowering canonicalizes away.

```
my_kernel
  ttir      +5/-5
  ttgir     +5/-5
  llir      +60/-58       <- LLIR diverges → real semantic difference
  ptx       +50/-48
```

This kernel needs work: a real divergence appeared somewhere between
TTIR and LLIR. Drill into the LLIR diff to find which op got lowered
differently.

---

## Phase 3 — Fixing divergences

Workflow: pick the *smallest* diverging stage's diff and locate the
op-by-op mismatch. Most fixes fall into one of the categories below.

### Step 3.1 — `--show-diffs` to locate the divergence

```sh
python compare_ir.py --left xla_py --right xla_zig --kernel my_kernel --stage ttir --show-diffs
```

Read the diff carefully. The first diverging op tells you what to look
at; subsequent diffs are usually downstream consequences.

### Step 3.2 — Common porting bugs

These are the gotchas we've hit while porting kernels. If your diff
shows any of these patterns, fix the corresponding Zig DSL call.

#### 3.2.1 — `arith.maxnumf` vs `arith.maximumf` (and `min*`)

**Symptom:**
```
- %v43 = arith.maxnumf %v37, %v14 : f32         (Python)
+ %v43 = arith.maximumf %v37, %v13 : f32        (Zig)
```

**Fix:** **Use `b.maximum` / `Value.maximum` / `b.minimum` /
`Value.minimum`** — they default to `maxnumf`/`minnumf` and match
Python's `tl.max`/`tl.min`/`tl.maximum`/`tl.minimum`. **Don't reach
for `b.maximumf` / `b.minimumf` directly** — those are the IEEE-754
NaN-propagating escape hatches (`arith.maximumf`/`minimumf`), exposed
for the rare case where you genuinely want NaN propagation. The naming
is unfortunate (the `f` suffix reads like "the float version") — it's
actually "the NaN-propagating version." See README's "Reductions and
scans" gotcha table for the full mapping.

#### 3.2.2 — 2-operand vs 3-operand `tt.load`

**Symptom:**
```
- %v33 = tt.load %v32, %v30, %v9 : tensor<128x!tt.ptr<bf16>>   (Python — has `other`)
+ %v33 = tt.load %v32, %v30 : tensor<128x!tt.ptr<bf16>>        (Zig — no `other`)
```

**Fix:** Match Python's exact form. The DSL has only one masked-load
API (`b.loadOpts`); whether it emits 2-operand or 3-operand `tt.load`
depends entirely on whether you pass `.other`:

```zig
// Python: tl.load(ptr, mask=mask, other=0.0)   → 3-operand
const zero = b.zeros(&.{BLOCK}, dtype);
const x = b.loadOpts(ptr, .{ .mask = mask, .other = zero });

// Python: tl.load(ptr, mask=mask)              → 2-operand
const x = b.loadOpts(ptr, .{ .mask = mask });
```

The two forms are *semantically* close but not identical (out-of-bounds
lanes are undefined for 2-operand, zero for 3-operand). Always copy
Python's form, even if you suspect the difference is harmless — XLA's
canonicalization may or may not erase it.

#### 3.2.3 — `i32` vs `i64` width in pointer offsets

**Symptom:**
```
- %v26 = tt.addptr %v6, %v16 : !tt.ptr<bf16>, i32             (Python — uses i32)
+ %v25 = tt.addptr %v6, %v16 : !tt.ptr<bf16>, i64             (Zig — uses i64)
```

**Fix:** Python's pointer arithmetic preserves the original integer
width. `tl.program_id(0)` is i32; if Python writes
`y_s_ptr += tl.program_id(0)` (no cast), the offset stays i32. In Zig,
if you converted `pid` to i64 early to use it elsewhere, pass the raw
`b.programId(.x)` (i32) to that one `addPtr` site:

```zig
// raw i32 program_id for one site, i64 elsewhere
const pid_i32 = b.programId(.x);
const g_id = pid_i32.to(.i64);
// ...
const y_s_ptr_shifted = a.y_s_ptr.addPtr(pid_i32);  // i32, not g_id
```

#### 3.2.4 — Mask comparison width

**Symptom:**
```
- %v30 = arith.cmpi slt, %v28, %v29 : tensor<128xi64>          (Python — i64 vs i64)
+ %v28 = arith.trunci %v10 : i64 to i32
+ %v29 = tt.splat %v28 : i32 -> tensor<128xi32>
+ %v30 = arith.cmpi slt, %v26, %v29 : tensor<128xi32>          (Zig — trunc + splat + i32 cmp)
```

**Fix:** Match Python's promotion direction. Python implicitly promotes
the narrower operand to the wider one. If `cols` is i32 and
`group_size` is i64, Python emits `extsi cols i32→i64` then compares
at i64. The Zig DSL doesn't auto-promote, so do it explicitly:

```zig
const cols_i32 = b.arange(0, BLOCK, .i32);
const cols_i64 = cols_i32.to(.i64);
const mask = cols_i64.lt(group_size);   // both i64, no trunc
```

#### 3.2.5 — Op emit order

**Symptom:** TTIR diff shows the same ops in a different order:
```
- %v18 = arith.divsi %v17, %v15 : i64
- %v19 = arith.remsi %v17, %v15 : i64
- %v20 = arith.muli %v18, %v13 : i64
- %v21 = arith.muli %v19, %v11 : i64
- %v22 = arith.addi %v20, %v21 : i64
+ %v17 = arith.divsi %v16, %v14 : i64
+ %v18 = arith.muli %v17, %v12 : i64
+ %v19 = arith.remsi %v16, %v14 : i64
+ %v20 = arith.muli %v19, %v10 : i64
+ %v21 = arith.addi %v18, %v20 : i64
```

This usually doesn't affect LLIR/PTX (compiler reorders), but if it
does (or if you want a clean TTIR diff), match Python's source order
op-by-op. Bind every intermediate to a named local in the same order
Python evaluates:

```zig
// Python: y_ptr += (g_id // groups_per_row) * y_row_stride + (g_id % groups_per_row) * group_size
//   eval order: divsi, remsi, muli, muli, addi
const row = g_id.div(groups_per_row);             // divsi
const row_g_id = g_id.rem(groups_per_row);        // remsi
const row_off = row.mul(y_row_stride);            // muli
const grp_off = row_g_id.mul(group_size);         // muli
const y_ptr_shifted = a.y_ptr.addPtr(row_off.add(grp_off));  // addi + addptr
```

#### 3.2.6 — `arith.maxnumf` (and `splat`) vs `tt.splat` of a runtime value

**Symptom:**
```
- %splat = tt.splat %scalar : i32 -> tensor<128xi32>    (Python — runtime splat)
+ %const = arith.constant dense<...> : tensor<128xi32>  (Zig — constant pool)
```

**Fix:** `b.splat(comptime_literal, shape)` materializes a constant
dense; `b.splat(runtime_value, shape)` emits `tt.splat`. If Python
uses a runtime value (loaded scalar, function arg) and you accidentally
hardcoded the constant, switch to the runtime path.

#### 3.2.7 — `tl.constexpr` divisibility hints

**Symptom:** TTIR header differs in `tt.divisibility = N : i32` attrs.

**Fix:** Default `.{ .ptr = .dtype }` carries `tt.divisibility = 16`.
For scalars, switch to `.scalar_opts`:

```zig
.n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
```

This matches what Python's JIT runtime auto-attaches when the runtime
value is a multiple of 16.

#### 3.2.8 — Reductions over rank-1 tensors

**Symptom:** Python emits a `tt.reshape allow_reorder` before the
`tt.reduce`; Zig doesn't.

**Fix:** Use the `axis = null` (default) form: `b.max(x)`, not
`b.maxOpts(x, .{ .axis = 0 })`. The `axis = null` path inserts the
reshape; explicit-axis paths skip it.

#### 3.2.9 — `tl.dot` precision and accumulator

**Symptom:** TTIR shows different `tt.dot` `inputPrecision` attribute or
different `acc` type.

**Fix:** Match Python's `tl.dot(a, b, acc, input_precision="...")`
explicitly:

```zig
const new_acc = b.dotOpts(a_val, b_val, acc, .{
    .input_precision = .tf32,        // matches `input_precision="tf32"`
    .max_num_imprecise_acc = 0,
});
```

Default `input_precision` differs across Triton versions — always be
explicit.

#### 3.2.10 — `iter_args` order in `scf.for`

**Symptom:** TTIR diff shows the loop signature reordered:
```
- %v152:3 = scf.for ... iter_args(%v149 = %v131, %v150 = %v143, %v151 = %v37)
-   -> (tensor<...!tt.ptr>, tensor<...!tt.ptr>, tensor<...xf32>)
+ %v152:3 = scf.for ... iter_args(%v149 = %v37, %v150 = %v131, %v151 = %v143)
+   -> (tensor<...xf32>, tensor<...!tt.ptr>, tensor<...!tt.ptr>)
```

**Fix:** Python's `scf.for` lowering captures `iter_args` in the order
they're **declared** above the loop, not the order they're mutated inside.
For `fused_moe_kernel`:

```python
a_ptrs = ...                # declared 1st
b_ptrs = ...                # declared 2nd
accumulator = tl.zeros(...) # declared 3rd
for k in range(...):
    accumulator += tl.dot(a, b)   # mutated 1st
    a_ptrs += BLOCK_SIZE_K        # mutated 2nd
    b_ptrs += BLOCK_SIZE_K        # mutated 3rd
```

Python's `iter_args` order = declaration order = `(a_ptrs, b_ptrs, acc)`,
**not** mutation order. The DSL `openFor` follows the order you pass — so
match Python's declaration order:

```zig
var loop = b.openFor(0, num_k_iters, 1, .{ a_ptrs_init, b_ptrs_init, acc_init });
{
    const a_ptrs = loop.carried[0];
    const b_ptrs = loop.carried[1];
    const acc    = loop.carried[2];
    // ...
    loop.yield(.{ new_a_ptrs, new_b_ptrs, new_acc });
}
const accumulator = loop.results[2];
```

The wrong order survives all the way to PTX as different register-block
layouts.

#### 3.2.11 — Operand order in commutative `arith.muli`

**Symptom:** TTIR diff shows `arith.muli` operands swapped:
```
- %v137 = arith.muli %v136, %v135 : tensor<1x64xi64>     (Python: stride on LHS)
+ %v137 = arith.muli %v135, %v136 : tensor<1x64xi64>     (Zig: offs on LHS)
```

**Fix:** `arith.muli` is commutative semantically, but the operand order
affects downstream layout decisions in TTGIR. Match Python's source order
exactly. For `stride * offs[None, :]`, Python emits `arith.muli %stride_splat,
%offs`:

```zig
// Python: stride_cm * offs_token[:, None]    — stride on LHS
const cm_col = stride_cm.mul(offs_token_col);
// NOT: offs_token_col.mul(stride_cm)         — offs on LHS, wrong order
```

Same applies to `arith.addi`, `arith.subi`, `arith.andi`, etc.

#### 3.2.12 — Two-step `addPtr` for `c_ptr + cm[:, None] + cn[None, :]`

**Symptom:** Python emits two `tt.addptr` ops; the Zig port collapses
them into one big addptr with a pre-merged 2D offset.
```
Python:
  %ptr_col   = tt.addptr %c_ptr_splat_Mx1, %cm_col_Mx1   : Mx1x!tt.ptr, Mx1xi64
  %ptr_2d    = tt.broadcast %ptr_col → MxNx!tt.ptr
  %final_ptr = tt.addptr %ptr_2d, %cn_off_2d             : MxNx!tt.ptr, MxNxi64
Zig:
  %offset    = arith.addi %cm_term, %cn_term             : MxNxi64
  %final_ptr = tt.addptr %c_ptr_splat, %offset           : MxNx!tt.ptr, MxNxi64
```

**Fix:** Python evaluates `c_ptr + cm[:, None] + cn[None, :]` left-to-right,
so it emits two `tt.addptr` ops. Match by splitting:

```zig
// First addptr: scalar c_ptr + (Mx1) cm offset → Mx1 ptr tensor
const cm_col = stride_cm.mul(b.expandDims(offs_token, 1));
const c_ptrs_col = b.splat(c_ptr, &.{ block_size_m, 1 }).addPtr(cm_col);

// Second addptr: broadcast to MxN, add cn offset
const cn_row = b.expandDims(offs_cn, 0);
const c_ptrs_2d = b.broadcastTo(c_ptrs_col, &.{ block_size_m, block_size_n });
const c_ptrs   = c_ptrs_2d.addPtr(b.broadcastTo(cn_row, &.{ block_size_m, block_size_n }));
```

This pattern shows up wherever Python writes `ptr + a[:, None] + b[None, :]`
(both A and B ptrs in fused_moe, the C-output store, etc.).

#### 3.2.13 — `extsi` placement relative to sibling `expand_dims`/`mul`

**Symptom:** Same set of ops, different positions:
```
- %v134 = expand_dims %offs_k_i32 (axis=1)            (Python: offs_k expand)
- %v135 = expand_dims %offs_bn_i64 (axis=0)            (Python: offs_bn expand)
- %v136 = splat %stride_bn → 1x64xi64                  (Python: bn stride)
- %v137 = muli %v135, %v136                            (Python: bn_row mul)
- %v138 = extsi %v134 → 32x1xi64                       (Python: extsi DEFERRED)
+ %v135 = extsi %v134 → 32x1xi64                       (Zig: extsi EAGER)
+ ... (rest in different order)
```

**Fix:** When Python writes `offs_k[:, None]` and uses it later in an
`addi` against an `i64`-typed tensor, the implicit `extsi` happens at the
**use site** (where the auto-promotion fires), not at the `expand_dims`.
Mirror this — keep the chain at i32 until just before it joins the i64
expression:

```zig
// Wrong: extsi happens immediately after expand_dims
const offs_k_col = b.expandDims(offs_k_i32, 1).to(.i64);  // extsi NOW
const offs_bn_row = b.expandDims(offs_bn, 0);
const bn_row = offs_bn_row.mul(stride_bn);

// Right: extsi deferred to after offs_bn's mul
const offs_k_col_i32 = b.expandDims(offs_k_i32, 1);       // 32x1xi32
const offs_bn_row    = b.expandDims(offs_bn, 0);
const bn_row         = offs_bn_row.mul(stride_bn);         // 1x64xi64
const offs_k_col     = offs_k_col_i32.to(.i64);            // extsi HERE
```

Same idea for placing a `muli` before an `extsi` on a sibling load:

```zig
// Python emits muli before the extsi at the cmpi:
const num_tokens_post_padded_i32 = b.load(...);            // load
const pid_m_block               = pid_m.mul(block_size_m); // muli FIRST
const num_tokens_post_padded    = num_tokens_post_padded_i32.to(.i64); // extsi
const out_of_range              = pid_m_block.ge(num_tokens_post_padded);
```

### Step 3.3 — Iterate

After each fix:

```sh
bazel build //examples/triton_emitter:dump_zig_ir
bash examples/triton_emitter/run.sh
```

Re-read the diff. The line counts should drop, and ideally the LLIR/PTX
columns flip to `✓`. If not, drill again — there's another mismatch
hiding behind the first.

### Step 3.4 — Don't be a perfectionist about TTIR (but do read every line)

You only need **LLIR + PTX** to match. If TTIR/TTGIR have +5 cosmetic
lines but LLIR is byte-identical, ship it. The XLA canonicalization +
CSE passes erase a lot of low-level differences before LLIR.

A kernel like `count_and_sort_expert_tokens_kernel` (TTIR/TTGIR +8/-8,
LLIR/PTX ✓) is a clean port. Don't keep grinding it.

**But** — when LLIR is **not** ✓ and the diff is small (tens of lines),
do **not** label the rest "scheduling noise" without a name-normalized
diff. Most lines in a small LLIR diff are downstream SSA renumbering
caused by 1–4 high-leverage TTIR-level fixes (typically the patterns in
3.2.10–3.2.13). Run:

```sh
diff \
  <(sed -E 's/%[a-zA-Z_.][a-zA-Z0-9_.]*[0-9]+/%V/g; s/%[0-9]+/%V/g; s/^[0-9]+:/V:/g' xla_py/foo.llir) \
  <(sed -E 's/%[a-zA-Z_.][a-zA-Z0-9_.]*[0-9]+/%V/g; s/%[0-9]+/%V/g; s/^[0-9]+:/V:/g' xla_zig/foo.llir)
```

If this normalized diff is small (~tens of lines), every remaining line
is pointing at a structural fix. The "+100 LLIR is just XLA scheduling"
intuition has been wrong every time so far — `fused_moe_kernel` went from
LLIR +100 → ✓ via four TTIR-level fixes (iter_args order, muli operand
order, two-step addPtr, extsi placement) that were all visible in the
+100-line diff once read carefully.

### Step 3.5 — Visual-transparency pass

Once the LLIR matches, do one pass over the Zig source with the Python
source open beside it. Check (in order):

- **Variable names.** Python `groups_per_row` → Zig `groups_per_row`,
  not `gp_row` or `gpr`. Same for arg names — `y_ptr` not `output_ptr`.
- **Line order.** Each Python line should correspond to roughly one
  Zig line, in the same order. If a single Python expression became
  three Zig temporaries, ask: does the DSL force the split? If not,
  collapse them onto one chained `.add(...).mul(...)` line.
- **Granularity of locals.** If Python writes
  `_absmax = tl.maximum(tl.max(tl.abs(y)), eps)` as a one-liner, the
  Zig port should also be one line: `const absmax = b.max(b.absf(y)).maximum(eps);`.
- **Comments.** Carry over Python's comments verbatim where they apply.
  If Python writes `# Map the program id to the row of X and Y it
  should compute.` above its block, paste the same comment above the
  Zig block. They serve as anchors for future side-by-side reads.
- **Helper functions.** If Python factors something into a
  `@triton.jit` helper (e.g. `find_seq_idx`), the Zig port should have
  a matching `fn findSeqIdx(b: *Builder, ...) Value` helper, not inline
  the body. Same arguments, same name (snake_case → camelCase).
- **Control flow shape.** Python `for k in range(...)` → `b.openFor`,
  Python `if ... return` → `b.returnIf` / `b.openReturnIf`. Don't
  flatten an `if` ladder into a `select` chain (or vice versa) just
  because the DSL allows it.

If a transparency fix changes the LLIR diff, **the LLIR-match goal
wins** — back out the visual change and leave a comment explaining the
divergence. Goal #1 always beats goal #2.

A useful sanity check: open both files side by side in your editor and
scroll both at the same rate. If the cursors stay roughly aligned, the
port reads transparently. If you have to scroll one twice as fast as
the other, there's structural drift to clean up.

---

## Phase 4 — Production wiring (post-port)

Once the LLIR matches *and* the visual-transparency pass is done:

1. **Side-by-side review.** Pull up Python and Zig in adjacent panes
   one more time. Anything that still reads differently between them
   either has a justification (DSL constraint, language constraint —
   leave a brief comment explaining why) or needs another pass.
2. **Remove any `// TODO match python` comments** from the kernel body.
3. **Verify the production caller passes the right cfg** — the dump
   tool's config and the production caller's config can differ (the
   dump tool uses small dimensions; production uses real ones).
   Production behavior depends on the runtime cfg, so make sure
   `makeFooConfig(...)` in the caller produces sensible values.
4. **Run the model end-to-end** if possible. The comparison harness
   only validates IR equivalence — runtime correctness still needs a
   real test (mismatched grid, missing strides, etc. won't show up in
   the diff but will crash at execution time).

---

## Cheat-sheet: when to look at which file

| Problem                                           | File                                         |
|---------------------------------------------------|----------------------------------------------|
| "How do I express `tl.X(...)` in Zig?"           | `zml/triton/README.md` (Python ↔ Zig table) |
| "What does `b.declareArgs(...)` accept?"          | `zml/triton/README.md` (Argument specs)     |
| "How do I call this kernel from `forward`?"       | `zml/triton/README.md` (`K.call(...)`)      |
| "My kernel diverges from Python at LLIR"          | This file (Phase 3)                         |
| "How do I add a kernel to the comparison harness?"| Phase 1 above + `examples/triton_emitter/`  |
| "I want to inspect raw TTIR ops"                  | `mlir/dialects/ttir/ttir.zig` (escape hatch) |

---

## When you're stuck

If a divergence resists all the standard fixes:

1. **Print both TTIRs side by side.** The visual structure usually
   reveals which control-flow region or loop body is misaligned.
2. **Check if the Python kernel uses any non-`@triton.jit` helpers** —
   if it imports something from `triton.language.extra` or has a custom
   `@triton.jit` helper, that helper might also need a port (or a
   `_SKIP` entry if it's pure-comptime).
3. **Verify the Python override args match the Zig config.** Mismatched
   constexpr values (e.g. `BLOCK_SIZE=64` in Zig vs `1024` auto-default
   in Python) generate completely different IR.
4. **Compare against `zml/moe/triton_kernels.zig` MoE kernels** — they
   went through this loop and are reasonable references for the patterns
   above.

---

## Before "fixing" the DSL

Every time a divergence makes you suspect the DSL itself is buggy and you
reach for `zml/triton/kernel.zig` — **verify against the Triton sources
first** at `/home/rigole/Documents/Git-Repos/triton`. The DSL exists to
mirror what Python's frontend emits; the source of truth is the Python
trace, not what the rule "looks like" or what feels symmetric.

Concretely:

- If you think `Value.cdiv` is wrong, read
  `triton/python/triton/language/standard.py` for the definition of
  `tl.cdiv` and compare. Don't assume the desugaring — check.
- If you think a binop dispatches to the wrong arith op
  (`maxnumf` vs `maximumf`, `divsi` vs `divui`, …), read
  `triton/python/triton/language/semantic.py` and find the cast /
  promotion rule that actually fires.
- If you think the type-promotion is wrong (i32 stays i32, doesn't
  auto-extend to i64, …), read `semantic.py:cast()` and the binop
  promotion helpers.

A second pitfall: **the local kernel may define its own helper that
shadows the stdlib name.** `unified_attention.py` defines a private
`cdiv_fn(x, y) = (x + y - 1) // y` — left-to-right Python parses this
as `((x + y) - 1)`, so it emits `addi` then `subi`. Triton's *stdlib*
`tl.cdiv` is `(x + (div - 1)) // div` and emits `subi` then `addi`.
These desugar identically when the divisor is comptime (Triton folds
`div - 1` at trace time), but differ when the divisor is a runtime
tensor. If your divergence is just an `add↔sub` reorder, check whether
the Python kernel actually uses `tl.cdiv` or a private `cdiv`-shaped
helper, and either keep the DSL aligned with `tl.cdiv` and write a
local Zig helper for the kernel-private form, or the other way around.

In short: when a "bug" in the DSL would also affect kernels that
already match Python, the bug is almost certainly elsewhere.

**And if you do change the DSL**, run `examples/triton_emitter/run.sh`
afterwards and confirm no kernel that previously matched (or was
codegen-equivalent) regressed at LLIR/PTX. A DSL edit that fixes one
kernel by introducing a divergence in another is not a fix — it's a
trade. Always check the full table before committing the change.
