"""Local repository for the standalone FlashInfer CUTLASS MoE C ABI."""

_ROOT_ENV = "ZML_FLASHINFER_CUTLASS_MOE_ROOT"
_LIBRARIES = [
    "libflashinfer_cutlass_moe_sm90.so",
    "libflashinfer_cutlass_moe_sm100.so",
    "libflashinfer_cutlass_moe_sm120.so",
]

def _repository_impl(repository_ctx):
    source_root_override = repository_ctx.os.environ.get(_ROOT_ENV)
    if source_root_override:
        source_root = repository_ctx.path(source_root_override)
    else:
        zml_root = repository_ctx.path(repository_ctx.attr._workspace_marker).dirname
        source_root = zml_root.dirname.get_child("flashinfer")

    header = source_root.get_child("capi").get_child("flashinfer_cutlass_moe.h")
    repository_ctx.symlink(header, "include/flashinfer_cutlass_moe.h")

    for library_name in _LIBRARIES:
        library = source_root.get_child("bazel-bin").get_child(library_name)
        repository_ctx.symlink(library, "lib/" + library_name)

    repository_ctx.symlink(repository_ctx.attr._build_file, "BUILD.bazel")

_repository = repository_rule(
    implementation = _repository_impl,
    attrs = {
        "_build_file": attr.label(
            allow_single_file = True,
            default = Label(
                "//third_party/flashinfer_cutlass_moe:flashinfer_cutlass_moe.BUILD.bazel",
            ),
        ),
        "_workspace_marker": attr.label(
            allow_single_file = True,
            default = Label("//:MODULE.bazel"),
        ),
    },
    configure = True,
    environ = [_ROOT_ENV],
    local = True,
)

def _extension_impl(_module_ctx):
    _repository(name = "flashinfer_cutlass_moe_linux_amd64")

flashinfer_cutlass_moe = module_extension(
    implementation = _extension_impl,
)
