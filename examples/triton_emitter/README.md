# triton_emitter — compare Python Triton vs the Zig DSL

Drives both the Python frontend and our Zig DSL on the same kernel, pushes
both through XLA's real Triton pipeline (the one ZML uses at compile time),
and diffs the post-pipeline IR. Use it for two things:

- **DSL regression check** — when changing `zml/triton`, run the harness,
  expect every kernel to still match.
- **Kernel port verification** — when porting a new kernel to the DSL,
  run the harness on just that one kernel until both sides match.

## Layout

```
examples/triton_emitter/
├── kernels_py/<k>.py        — one @triton.jit fn per file
├── kernels_zig/<k>.zig      — Zig DSL port of the same kernel
├── kernels_zig.zig          — KERNELS list + per-kernel build configs
├── dump_python_ir.py        — Python → py_ir/<k>.{ttir,ttgir,llir,ptx}
├── dump_zig_ir.zig          — Zig DSL → zig_ir/<k>.ttir
├── dump_via_xla.zig         — TTIR file → XLA's per-pass dump
├── extract_xla_dump.py      — XLA dump → per-stage .ttir / .ttgir
├── compare_ir.py            — diff with cosmetic noise stripped
└── run.sh                   — orchestrates all of the above
```

Adding a kernel = drop `kernels_py/<name>.py` + `kernels_zig/<name>.zig`,
register it in `kernels_zig.zig`'s `KERNELS`, and add an entry to
`dump_via_xla.zig`'s `KERNELS` tuple (a per-kernel forward + tensor builder).

## Run

`run.sh` calls `python` from `$PATH`. Either use your system Python (with
`triton`, `torch`, `numpy` already installed) or activate a venv first:

```bash
uv venv --python 3.12 .venv-ir
source .venv-ir/bin/activate
uv pip install -r examples/triton_emitter/requirements.txt
```

Then:

```bash
# Compare every kernel at every stage (TTIR, TTGIR, LLIR, PTX) — default.
./examples/triton_emitter/run.sh

# Just one kernel
./examples/triton_emitter/run.sh --kernel triton_add_kernel

# Just one stage
./examples/triton_emitter/run.sh --stage llir
```

The script does six steps:
1. `dump_python_ir.py` — Python's TTIR/TTGIR/LLIR/PTX into `py_ir/`
2. `dump_zig_ir.zig` — Zig DSL's raw TTIR into `zig_ir/`
3. `dump_via_xla.zig` on `py_ir/` — XLA's per-pass dump into `xla_dump_py/`
4. `dump_via_xla.zig` on `zig_ir/` — XLA's per-pass dump into `xla_dump_zig/`
5. `extract_xla_dump.py` slices both per-pass dumps into per-stage files
6. `compare_ir.py` diffs them at the requested `--stage`

Both sides go through the *same* XLA pipeline
(`xla/backends/gpu/codegen/triton/compilation_pipeline_cuda.cc`), so any
remaining diff is a real semantic difference between the frontends — XLA's
post-passes (module attribute decoration, `scf→cf` lowering, etc.) cancel.

## Currently ported kernels

| File                          | Python kernel                                |
|-------------------------------|----------------------------------------------|
| `vector_add.{py,zig}`         | `triton_add_kernel`                          |
| `vector_exp.{py,zig}`         | `triton_exp_kernel` + `..._backward_kernel`  |
| `low_mem_dropout.{py,zig}`    | `_triton_dropout`                            |
| `sum_scalar.{py,zig}`         | `triton_sum_kernel_scalar_result`            |
| `softmax.{py,zig}`            | `softmax_kernel`                             |
| `moe.{py,zig}`                | `per_token_group_quant_fp8`, `fused_moe_kernel`, `moe_align_block_size_kernel`, `count_and_sort_expert_tokens_kernel` |
