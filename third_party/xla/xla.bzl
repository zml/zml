load("@xla//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "tf_vendored")
load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
load("@xla//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("@xla//third_party/triton:workspace.bzl", triton = "repo")

_BZL_HELPERS = """\
always_newer_than = lambda wanted_ver, if_true, if_false = []: if_true
always_if_true = lambda if_true, if_false = []: if_true
always_if_false = lambda if_true, if_false = []: if_false
always_false = lambda *args, **kwargs: False
"""

def _simple_files_impl(rctx):
    rctx.file("BUILD.bazel", "")
    for f, content in rctx.attr.files.items():
        rctx.file(f, content)

simple_files = repository_rule(
    implementation = _simple_files_impl,
    attrs = {
        "files": attr.string_dict(),
    },
)

def _dummy_repos(mctx):
    simple_files(name = "local_config_cuda", files = {
        "cuda/BUILD.bazel": "",
        "cuda/build_defs.bzl": _BZL_HELPERS + """\
cuda_library = always_false
if_cuda = always_if_false
if_cuda_is_configured = always_if_false
if_cuda_newer_than = always_newer_than
is_cuda_configured = always_false
""",
    })
    simple_files(name = "local_config_rocm", files = {
        "rocm/BUILD.bazel": "",
        "rocm/build_defs.bzl": _BZL_HELPERS + """\
if_rocm = always_if_false
if_rocm_is_configured = always_if_false
if_rocm_newer_than = always_newer_than
is_rocm_configured = always_false
if_gpu_is_configured = always_if_false
if_cuda_or_rocm = always_if_false
""",
    })
    simple_files(name = "local_config_remote_execution", files = {
        "remote_execution.bzl": """gpu_test_tags = lambda: []""",
    })
    simple_files(name = "local_config_tensorrt", files = {
        "build_defs.bzl": _BZL_HELPERS + """if_tensorrt = always_if_false""",
    })
    simple_files(name = "python_version_repo", files = {
        "py_version.bzl": """USE_PYWRAP_RULES = False""",
    })
    simple_files(name = "rules_ml_toolchain", files = {
        "third_party/gpus/BUILD.bazel": "",
        "third_party/gpus/nvidia_common_rules.bzl": """cuda_rpath_flags = lambda *args, **kwargs: []""",
    })

def _xla_impl(mctx):
    llvm("llvm-raw")
    stablehlo()
    triton()

    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "afbc5d78d6ba6d509cc6e264de0d49dcd7304db435cbf2d630385bacf49e066c",
        strip_prefix = "grpc-1.68.2",
        patch_file = [
            "//third_party/grpc:grpc.patch",
        ],
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/refs/tags/v1.68.2.tar.gz"),
    )
    tf_vendored(name = "tsl", relpath = "third_party/tsl")

    _dummy_repos(mctx)

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

xla = module_extension(
    implementation = _xla_impl,
)
