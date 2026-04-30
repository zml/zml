load("@xla//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "tf_vendored")
load("@xla//third_party/eigen3:workspace.bzl", eigen3 = "repo")
load("@xla//third_party/farmhash:workspace.bzl", farmhash = "repo")
load("@xla//third_party/highwayhash:workspace.bzl", highwayhash = "repo")
load("@xla//third_party/hwloc:workspace.bzl", hwloc = "repo")
load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
load("@xla//third_party/py/ml_dtypes:workspace.bzl", ml_dtypes = "repo")
load("@xla//third_party/shardy:workspace.bzl", shardy = "repo")
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
        "BUILD.bazel": """\
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "is_cuda_enabled",
    define_values = {"local_config_cuda_enabled": "true"},
)

config_setting(
    name = "is_cuda_compiler_nvcc",
    define_values = {"local_config_cuda_compiler_nvcc": "true"},
)
""",
        "cuda/BUILD.bazel": """\
package(default_visibility = ["//visibility:public"])

exports_files(["build_defs.bzl"])

filegroup(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

config_setting(
    name = "using_config_cuda",
    define_values = {"local_config_cuda_using_config": "true"},
)

config_setting(
    name = "FALSE",
    define_values = {"local_config_cuda_false": "true"},
)
""",
        "cuda/build_defs.bzl": _BZL_HELPERS + """\
cuda_library = always_false
if_cuda = always_if_false
if_cuda_is_configured = always_if_false
if_cuda_newer_than = always_newer_than
is_cuda_configured = always_false
""",
    })
    simple_files(name = "local_config_rocm", files = {
        "BUILD.bazel": "",
        "rocm/BUILD.bazel": """\
package(default_visibility = ["//visibility:public"])

exports_files(["build_defs.bzl"])

filegroup(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

config_setting(
    name = "using_hipcc",
    define_values = {"local_config_rocm_using_hipcc": "true"},
)
""",
        "rocm/build_defs.bzl": _BZL_HELPERS + """\
if_rocm = always_if_false
if_rocm_is_configured = always_if_false
if_rocm_newer_than = always_newer_than
is_rocm_configured = always_false
if_gpu_is_configured = always_if_false
if_cuda_or_rocm = always_if_false
get_rbe_amdgpu_pool = always_false
""",
    })
    simple_files(name = "local_config_sycl", files = {
        "BUILD.bazel": "",
        "sycl/BUILD.bazel": """\
package(default_visibility = ["//visibility:public"])

exports_files(["build_defs.bzl"])

filegroup(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
""",
        "crosstool/BUILD.bazel": "",
        "sycl/build_defs.bzl": _BZL_HELPERS + """\
if_sycl = always_if_false
if_sycl_is_configured = always_if_false
sycl_library = native.cc_library
""",
    })
    simple_files(name = "local_config_remote_execution", files = {
        "remote_execution.bzl": """gpu_test_tags = lambda: []""",
    })
    simple_files(name = "local_config_tensorrt", files = {
        "BUILD.bazel": """\
package(default_visibility = ["//visibility:public"])

exports_files(["build_defs.bzl"])

filegroup(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)
""",
        "build_defs.bzl": _BZL_HELPERS + """if_tensorrt = always_if_false""",
    })
    simple_files(name = "python_version_repo", files = {
        "py_version.bzl": """USE_PYWRAP_RULES = False""",
    })
    simple_files(name = "rules_ml_toolchain", files = {
        "py/rules_pywrap/BUILD.bazel": """\
package(default_visibility = ["//visibility:public"])

exports_files([
    "pywrap.default.bzl",
    "pywrap.impl.bzl",
])

filegroup(
    name = "pywrap_bzl",
    srcs = [
        "pywrap.default.bzl",
        "pywrap.impl.bzl",
    ],
)
""",
        "py/rules_pywrap/pywrap.default.bzl": """\
def use_pywrap_rules():
    return False

def _unsupported(*args, **kwargs):
    fail("rules_ml_toolchain pywrap rules are not available in this workspace")

pybind_extension = _unsupported
pywrap_aware_cc_import = _unsupported
pywrap_aware_filegroup = _unsupported
pywrap_aware_genrule = _unsupported
pywrap_binaries = _unsupported
pywrap_common_library = _unsupported
pywrap_library = _unsupported
stripped_cc_info = _unsupported
""",
        "py/rules_pywrap/pywrap.impl.bzl": """\
def _unsupported(*args, **kwargs):
    fail("rules_ml_toolchain pywrap rules are not available in this workspace")

collected_pywrap_infos = _unsupported
generated_common_win_def_file = _unsupported
pybind_extension = _unsupported
python_extension = _unsupported
pywrap_binaries = _unsupported
pywrap_common_library = _unsupported
pywrap_library = _unsupported
stripped_cc_info = _unsupported
""",
        "third_party/gpus/BUILD.bazel": "",
        "third_party/gpus/nvidia_common_rules.bzl": """cuda_rpath_flags = lambda *args, **kwargs: []""",
        "third_party/extensions/sycl_configure.bzl": "",
    })
    simple_files(name = "sycl_configure_ext", files = {})
    simple_files(name = "sycl_configure", files = {})

def _xla_impl(mctx):
    llvm("llvm-raw")
    stablehlo()
    shardy()
    triton()
    farmhash()
    highwayhash()
    hwloc()
    eigen3()
    ml_dtypes()

    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "e2ace790a5f2d0f83259d1390a816a33b013ea34df2e86084d927e58daa4c5d9",
        strip_prefix = "grpc-1.78.0",
        patch_file = ["//third_party/grpc:grpc.patch"],
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/refs/tags/v1.78.0.tar.gz"),
    )
    tf_vendored(name = "tsl", path = "third_party/tsl")

    tf_http_archive(
        name = "snappy",
        build_file = "//third_party:snappy.BUILD",
        sha256 = "736aeb64d86566d2236ddffa2865ee5d7a82d26c9016b36218fcc27ea4f09f86",
        strip_prefix = "snappy-1.2.1",
        urls = tf_mirror_urls("https://github.com/google/snappy/archive/refs/tags/1.2.1.tar.gz"),
    )

    _dummy_repos(mctx)

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

xla = module_extension(
    implementation = _xla_impl,
)
