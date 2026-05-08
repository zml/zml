"""All kernel registrations for `tools/dsl-harness`.

To register a new kernel:

  1. Drop a `<name>.zig` file in `kernels/<backend>/zig/` that re-exports
     the production Kernel and declares
     `pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{ ... }`.
  2. Drop the matching Python source in `kernels/<backend>/py/<name>.py`
     (a `@triton.jit` for triton, a `pallas_call`-driving fn for mosaic_tpu).
  3. Add one entry to `TRITON_KERNELS` / `MOSAIC_TPU_KERNELS` below.

`declare_triton_kernels()` is invoked from `kernels/triton/BUILD.bazel`,
`declare_mosaic_tpu_kernels()` from `kernels/mosaic_tpu/BUILD.bazel`. The
macros generate per-kernel targets in the calling package; the helper label
functions below feed this directory's BUILD.bazel.
"""

load(":kernel.bzl", "mosaic_tpu_kernel", "triton_kernel")

TRITON_KERNELS = [
    {
        "name": "vector_add",
        "src": "zig/vector_add.zig",
        "py_src": "py/vector_add.py",
        "py_kernel": "triton_add_kernel",
    },
    {
        "name": "vector_exp_fwd",
        "src": "zig/vector_exp_fwd.zig",
        "py_src": "py/vector_exp_fwd.py",
        "py_kernel": "triton_exp_kernel",
    },
    {
        "name": "vector_exp_bwd",
        "src": "zig/vector_exp_bwd.zig",
        "py_src": "py/vector_exp_bwd.py",
        "py_kernel": "triton_exp_backward_kernel",
    },
    {
        "name": "low_mem_dropout",
        "src": "zig/low_mem_dropout.zig",
        "py_src": "py/low_mem_dropout.py",
        "py_kernel": "_triton_dropout",
        "extra_zig_deps": ["//mlir/dialects"],
    },
    {
        "name": "sum_scalar",
        "src": "zig/sum_scalar.zig",
        "py_src": "py/sum_scalar.py",
        "py_kernel": "triton_sum_kernel_scalar_result",
        "extra_zig_deps": ["//mlir/dialects/ttir"],
    },
    {
        "name": "softmax",
        "src": "zig/softmax.zig",
        "py_src": "py/softmax.py",
        "py_kernel": "softmax_kernel",
    },
    {
        "name": "gdn",
        "src": "zig/gdn.zig",
        "py_src": "py/gdn.py",
        "py_kernel": "fused_recurrent_gated_delta_rule_fwd_kernel_ptr",
    },
    {
        "name": "unified_attention_2d",
        "src": "zig/unified_attention_2d.zig",
        "py_src": "py/unified_attention_2d.py",
        "py_kernel": "kernel_unified_attention_2d_ptr",
        "extra_py_deps": [":unified_attention_kernels"],
    },
    {
        "name": "unified_attention_3d",
        "src": "zig/unified_attention_3d.zig",
        "py_src": "py/unified_attention_3d.py",
        "py_kernel": "kernel_unified_attention_3d_ptr",
        "extra_py_deps": [":unified_attention_kernels"],
    },
    {
        "name": "reduce_segments",
        "src": "zig/reduce_segments.zig",
        "py_src": "py/reduce_segments.py",
        "py_kernel": "reduce_segments_ptr",
        "extra_py_deps": [":unified_attention_kernels"],
    },
]

MOSAIC_TPU_KERNELS = [
    {
        "name": "ragged_paged",
        "src": "zig/ragged_paged.zig",
        "py_src": "py/ragged_paged.py",
        "py_kernel": "ragged_paged_attention",
    },
]

def _kernel_labels(kind, kernels, suffix):
    return ["//tools/dsl-harness/kernels/{}:{}{}".format(kind, k["name"], suffix) for k in kernels]

def triton_kernel_labels(suffix = ""):
    return _kernel_labels("triton", TRITON_KERNELS, suffix)

def mosaic_tpu_kernel_labels(suffix = ""):
    return _kernel_labels("mosaic_tpu", MOSAIC_TPU_KERNELS, suffix)

def declare_triton_kernels():
    for kernel in TRITON_KERNELS:
        triton_kernel(
            name = kernel["name"],
            src = kernel["src"],
            py_src = kernel["py_src"],
            py_kernel = kernel["py_kernel"],
            extra_py_deps = kernel.get("extra_py_deps", []),
            extra_zig_deps = kernel.get("extra_zig_deps", []),
        )

def declare_mosaic_tpu_kernels():
    for kernel in MOSAIC_TPU_KERNELS:
        mosaic_tpu_kernel(
            name = kernel["name"],
            src = kernel["src"],
            py_src = kernel["py_src"],
            py_kernel = kernel["py_kernel"],
            extra_py_deps = kernel.get("extra_py_deps", []),
            extra_zig_deps = kernel.get("extra_zig_deps", []),
        )
