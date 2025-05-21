load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")
load("//third_party/gpus:rocm_configure.bzl", "rocm_configure")
load("//third_party/gpus:sycl_configure.bzl", "sycl_configure")
load("//third_party/llvm:workspace.bzl", llvm = "repo")
load("//third_party/pybind11_bazel:workspace.bzl", pybind11_bazel = "repo")
load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("//third_party/shardy:workspace.bzl", shardy = "repo")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
load("//third_party/triton:workspace.bzl", triton = "repo")
load("//tools/toolchains/remote:configure.bzl", "remote_execution_configure")

def _xla_workspace_impl(mctx):
    cuda_configure(name = "local_config_cuda")
    remote_execution_configure(name = "local_config_remote_execution")
    rocm_configure(name = "local_config_rocm")
    sycl_configure(name = "local_config_sycl")
    tensorrt_configure(name = "local_config_tensorrt")
    pybind11_bazel()
    triton()
    llvm("llvm-raw")
    stablehlo()
    shardy()
    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = "1a432ccbd597ea7b9faa1557b1752328d6adc1a3db8969f6fe793ff704be3bf0",
        strip_prefix = "eigen-4c38131a16803130b66266a912029504f2cf23cd",
        urls = tf_mirror_urls("https://gitlab.com/libeigen/eigen/-/archive/4c38131a16803130b66266a912029504f2cf23cd/eigen-4c38131a16803130b66266a912029504f2cf23cd.tar.gz"),
    )
    tf_http_archive(
        name = "farmhash_archive",
        build_file = "//third_party/farmhash:farmhash.BUILD",
        sha256 = "18392cf0736e1d62ecbb8d695c31496b6507859e8c75541d7ad0ba092dc52115",
        strip_prefix = "farmhash-0d859a811870d10f53a594927d0d0b97573ad06d",
        urls = tf_mirror_urls("https://github.com/google/farmhash/archive/0d859a811870d10f53a594927d0d0b97573ad06d.tar.gz"),
    )
    tf_http_archive(
        name = "ml_dtypes_py",
        build_file = "//third_party/py/ml_dtypes:ml_dtypes_py.BUILD",
        link_files = {
            "//third_party/py/ml_dtypes:ml_dtypes.BUILD": "ml_dtypes/BUILD.bazel",
        },
        sha256 = "f6e5880666661351e6cd084ac4178ddc4dabcde7e9a73722981c0d1500cf5937",
        strip_prefix = "ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9",
        urls = tf_mirror_urls("https://github.com/jax-ml/ml_dtypes/archive/00d98cd92ade342fef589c0470379abb27baebe9/ml_dtypes-00d98cd92ade342fef589c0470379abb27baebe9.tar.gz"),
    )
    tf_http_archive(
        name = "snappy",
        build_file = "//third_party:snappy.BUILD",
        sha256 = "2e458b7017cd58dcf1469ab315389e85e7f445bd035188f2983f81fb19ecfb29",
        strip_prefix = "snappy-984b191f0fefdeb17050b42a90b7625999c13b8d",
        system_build_file = "//third_party/systemlibs:snappy.BUILD",
        urls = tf_mirror_urls("https://github.com/google/snappy/archive/984b191f0fefdeb17050b42a90b7625999c13b8d.tar.gz"),
    )
    tf_http_archive(
        name = "com_github_grpc_grpc",
        sha256 = "b956598d8cbe168b5ee717b5dafa56563eb5201a947856a6688bbeac9cac4e1f",
        strip_prefix = "grpc-b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd",
        system_build_file = "//third_party/systemlibs:grpc.BUILD",
        patch_file = [
            "//third_party/grpc:generate_cc_env_fix.patch",
            "//third_party/grpc:register_go_toolchain.patch",
        ],
        system_link_files = {
            "//third_party/systemlibs:BUILD.bazel": "bazel/BUILD.bazel",
            "//third_party/systemlibs:grpc.BUILD": "src/compiler/BUILD",
            "//third_party/systemlibs:grpc.bazel.grpc_deps.bzl": "bazel/grpc_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.grpc_extra_deps.bzl": "bazel/grpc_extra_deps.bzl",
            "//third_party/systemlibs:grpc.bazel.cc_grpc_library.bzl": "bazel/cc_grpc_library.bzl",
            "//third_party/systemlibs:grpc.bazel.generate_cc.bzl": "bazel/generate_cc.bzl",
            "//third_party/systemlibs:grpc.bazel.protobuf.bzl": "bazel/protobuf.bzl",
        },
        urls = tf_mirror_urls("https://github.com/grpc/grpc/archive/b54a5b338637f92bfcf4b0bc05e0f57a5fd8fadd.tar.gz"),
    )
    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["//third_party/protobuf:protobuf.patch"],
        sha256 = "f66073dee0bc159157b0bd7f502d7d1ee0bc76b3c1eac9836927511bdc4b3fc1",
        strip_prefix = "protobuf-3.21.9",
        system_build_file = "//third_party/systemlibs:protobuf.BUILD",
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
            "//third_party/systemlibs:protobuf_deps.bzl": "protobuf_deps.bzl",
        },
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/v3.21.9.zip"),
    )
    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

xla_workspace = module_extension(
    implementation = _xla_workspace_impl,
)
