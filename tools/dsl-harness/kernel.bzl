"""Kernel-registration macros for `tools/dsl-harness/`. Each
`triton_kernel(...)` / `mosaic_tpu_kernel(...)` call emits a registration
zig_library, a runner py_binary, and an entry-shim zig_library that
exports `pub const ENTRY: harness.KernelEntry`. `aggregator_gen` walks
the resulting `KernelEntryInfo` providers and emits `aggregator.zig`.
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_zig//zig:defs.bzl", "zig_library")

KernelEntryInfo = provider(
    doc = "Information `aggregator_gen` needs to emit aggregator.zig.",
    fields = {
        "name": "Registered kernel name (matches the `tt.func`/`func.func` symbol).",
        "kind": "'triton' or 'mosaic_tpu'.",
        "import_name": "@import string for the entry shim's ENTRY symbol.",
    },
)

def _kernel_entry_provider_impl(ctx):
    return [
        KernelEntryInfo(
            name = ctx.attr.kernel_name,
            kind = ctx.attr.kind,
            import_name = ctx.attr.import_name,
        ),
    ]

_kernel_entry_provider = rule(
    implementation = _kernel_entry_provider_impl,
    attrs = {
        "kernel_name": attr.string(mandatory = True),
        "kind": attr.string(mandatory = True, values = ["triton", "mosaic_tpu"]),
        "import_name": attr.string(mandatory = True),
    },
    provides = [KernelEntryInfo],
)

def _src_to_module_name(label):
    """Strip directory + ".py" from a Bazel label to a bare module name."""
    s = label
    if ":" in s:
        s = s.split(":", 1)[1]
    if "/" in s:
        s = s.rsplit("/", 1)[1]
    if s.endswith(".py"):
        s = s[:-3]
    return s

def _kernel_macro(
        name,
        src,
        py_src,
        py_kernel,
        kind,
        import_name_prefix,
        py_deps,
        py_runner,
        zig_deps,
        extra_py_deps = [],
        extra_zig_deps = []):
    """Shared body for `triton_kernel` and `mosaic_tpu_kernel`."""

    lib_name = name + "_lib"
    entry_src_name = name + "_entry_src"
    entry_lib_name = name + "_entry"
    py_src_lib_name = name + "_py_src_lib"
    py_bin_name = name + "_py"
    provider_name = name

    lib_import_name = import_name_prefix + "/" + name
    entry_import_name = import_name_prefix + "/" + name + "/entry"

    zig_library(
        name = lib_name,
        main = src,
        import_name = lib_import_name,
        visibility = ["//visibility:public"],
        deps = [
            "//tools/dsl-harness:harness",
        ] + zig_deps + extra_zig_deps,
    )

    module_name = _src_to_module_name(py_src)

    # rlocation path = apparent_repo_name + workspace-relative path.
    py_runfile_path = "zml/tools/dsl-harness/kernels/{kind}/{name}_py".format(
        kind = kind,
        name = name,
    )
    shim_body = """//! Auto-generated entry shim for kernel `{name}`. Do not edit.
const std = @import("std");
const mlir = @import("mlir");
const zml = @import("zml");
const harness = @import("harness");
const kernel_lib = @import("{lib_import}");

const Kernel = kernel_lib.Kernel;
const SWEEPS = kernel_lib.SWEEPS;

fn emitFn(allocator: std.mem.Allocator, cfg_idx: usize) anyerror![:0]const u8 {{
    inline for (SWEEPS, 0..) |sweep, i| {{
        if (i == cfg_idx) return Kernel.emit(allocator, sweep.cfg);
    }}
    return error.InvalidSweepIndex;
}}

fn cfgJsonFn(allocator: std.mem.Allocator, cfg_idx: usize) anyerror![]const u8 {{
    inline for (SWEEPS, 0..) |sweep, i| {{
        if (i == cfg_idx) return std.json.Stringify.valueAlloc(allocator, sweep.cfg, .{{}});
    }}
    return error.InvalidSweepIndex;
}}

/// Wired only when the registration file exposes `forward`, `args`,
/// and `setActiveTtir`; otherwise `compileFn` is null.
const has_xla_driver = @hasDecl(kernel_lib, "forward") and
    @hasDecl(kernel_lib, "args") and
    @hasDecl(kernel_lib, "setActiveTtir");

fn compileFn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    ttir: [:0]const u8,
    out_dir: []const u8,
) anyerror!void {{
    if (!has_xla_driver) return error.XlaDriverNotWired;

    kernel_lib.setActiveTtir(ttir);
    defer kernel_lib.setActiveTtir("");

    const replicated = platform.replicated_sharding;
    var exe = try zml.module.compile(allocator, io, kernel_lib.forward, kernel_lib.args(), platform, .{{
        .program_name = Kernel.name,
        .shardings = &.{{replicated}},
        .xla_dump_to = out_dir,
        .xla_dump_emitter_re = "triton-to-llvm",
    }});
    exe.deinit();
}}

pub const ENTRY: harness.KernelEntry = .{{
    .name = Kernel.name,
    .kind = .{kind},
    .sweeps = harness.projectSweepRefs(Kernel.Config, SWEEPS),
    .emitFn = &emitFn,
    .cfgJsonFn = &cfgJsonFn,
    .py_runfile = "{py_runfile}",
    .py_module = "{py_module}",
    .py_kernel_fn = "{py_kernel_fn}",
    .compileFn = if (has_xla_driver) &compileFn else null,
}};
""".format(
        name = name,
        lib_import = lib_import_name,
        kind = kind,
        py_runfile = py_runfile_path,
        py_module = module_name,
        py_kernel_fn = py_kernel,
    )

    write_file(
        name = entry_src_name,
        out = name + "_entry.zig",
        content = [shim_body],
        newline = "auto",
    )

    zig_library(
        name = entry_lib_name,
        main = ":" + entry_src_name,
        import_name = entry_import_name,
        visibility = ["//visibility:public"],
        deps = [
            ":" + lib_name,
            "//tools/dsl-harness:harness",
            "//mlir",
            "//zml",
        ],
    )

    # `imports = ["py"]` puts py/ on PYTHONPATH so the bare module_name
    # resolves (workspace paths use hyphens, not valid Python identifiers).
    py_library(
        name = py_src_lib_name,
        srcs = [py_src],
        imports = ["py"],
        deps = py_deps + extra_py_deps,
        visibility = ["//visibility:private"],
    )

    py_binary(
        name = py_bin_name,
        main = py_runner,
        srcs = [py_runner],
        python_version = "3.13" if kind == "triton" else "3.12",
        deps = [
            ":" + py_src_lib_name,
        ] + py_deps,
        # Skip wildcard builds; the harness binary pulls them in via `data`.
        tags = ["manual"],
        visibility = ["//visibility:public"],
    )

    _kernel_entry_provider(
        name = provider_name,
        kernel_name = name,
        kind = kind,
        import_name = entry_import_name,
        visibility = ["//visibility:public"],
    )

def triton_kernel(name, src, py_src, py_kernel, extra_py_deps = [], extra_zig_deps = []):
    """Register a Triton kernel. Call from `kernels/triton/kernels.bzl`."""
    _kernel_macro(
        name = name,
        src = src,
        py_src = py_src,
        py_kernel = py_kernel,
        kind = "triton",
        import_name_prefix = "harness/kernels/triton",
        py_deps = [
            "@triton_py_deps//triton:pkg",
            "//tools/dsl-harness/py:runtime",
            "//tools/dsl-harness/py:fake_plugin",
            "//tools/dsl-harness/py:triton_helpers",
        ],
        py_runner = "//tools/dsl-harness/triton:runner.py",
        zig_deps = ["//zml"],
        extra_py_deps = extra_py_deps,
        extra_zig_deps = extra_zig_deps,
    )

def mosaic_tpu_kernel(name, src, py_src, py_kernel, extra_py_deps = [], extra_zig_deps = []):
    """Register a Mosaic-TPU kernel. Call from `kernels/mosaic_tpu/kernels.bzl`."""
    _kernel_macro(
        name = name,
        src = src,
        py_src = py_src,
        py_kernel = py_kernel,
        kind = "mosaic_tpu",
        import_name_prefix = "harness/kernels/mosaic_tpu",
        py_deps = [
            "@jax_py_deps//jax:pkg",
            "@jax_py_deps//jaxlib:pkg",
            "@jax_py_deps//absl_py:pkg",
            "//tools/dsl-harness/py:runtime",
        ],
        py_runner = "//tools/dsl-harness/mosaic_tpu:runner.py",
        zig_deps = ["//mlir", "//zml", "//platforms/tpu:ragged_paged"],
        extra_py_deps = extra_py_deps,
        extra_zig_deps = extra_zig_deps,
    )

def _aggregator_gen_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".zig")

    triton_lines = []
    for dep in ctx.attr.triton_kernels:
        info = dep[KernelEntryInfo]
        if info.kind != "triton":
            fail("aggregator_gen: target {} declared as triton but provider says {}".format(dep.label, info.kind))
        triton_lines.append("    &@import(\"{}\").ENTRY,".format(info.import_name))

    mosaic_tpu_lines = []
    for dep in ctx.attr.mosaic_tpu_kernels:
        info = dep[KernelEntryInfo]
        if info.kind != "mosaic_tpu":
            fail("aggregator_gen: target {} declared as mosaic_tpu but provider says {}".format(dep.label, info.kind))
        mosaic_tpu_lines.append("    &@import(\"{}\").ENTRY,".format(info.import_name))

    content = "// Auto-generated by `aggregator_gen` in tools/dsl-harness/kernel.bzl. Do not edit.\n\n"
    content += "const harness = @import(\"harness\");\n\n"
    content += "pub const TRITON_KERNELS: []const *const harness.KernelEntry = &.{\n"
    content += "\n".join(triton_lines)
    if triton_lines:
        content += "\n"
    content += "};\n\n"
    content += "pub const MOSAIC_TPU_KERNELS: []const *const harness.KernelEntry = &.{\n"
    content += "\n".join(mosaic_tpu_lines)
    if mosaic_tpu_lines:
        content += "\n"
    content += "};\n"

    ctx.actions.write(out, content)
    return [DefaultInfo(files = depset([out]))]

aggregator_gen = rule(
    implementation = _aggregator_gen_impl,
    doc = "Emits aggregator.zig listing every registered kernel.",
    attrs = {
        "triton_kernels": attr.label_list(providers = [KernelEntryInfo]),
        "mosaic_tpu_kernels": attr.label_list(providers = [KernelEntryInfo]),
    },
)
