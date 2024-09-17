load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

ARCH = "linux-x86_64"

CUDA_VERSION = "12.6.1"
CUDNN_VERSION = "9.3.0"

_CC_IMPORT_TPL = """\
cc_import(
    name = "{name}",
    shared_library = "lib/{shared_library}",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)
"""

CUDA_PACKAGES = {
    "cuda_cudart": _CC_IMPORT_TPL.format(name = "cudart", shared_library = "libcudart.so.12"),
    "cuda_cupti": _CC_IMPORT_TPL.format(name = "cupti", shared_library = "libcupti.so.12"),
    "libcufft": _CC_IMPORT_TPL.format(name = "cufft", shared_library = "libcufft.so.11"),
    "libcusolver": _CC_IMPORT_TPL.format(name = "cusolver", shared_library = "libcusolver.so.11"),
    "libcusparse": _CC_IMPORT_TPL.format(name = "cusparse", shared_library = "libcusparse.so.12"),
    "libnvjitlink": _CC_IMPORT_TPL.format(name = "nvjitlink", shared_library = "libnvJitLink.so.12"),
    "cuda_nvcc": """\
filegroup(
    name = "ptxas",
    srcs = ["bin/ptxas"],
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)

filegroup(
    name = "libdevice",
    srcs = ["nvvm/libdevice/libdevice.10.bc"],
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)

cc_import(
    name = "nvvm",
    shared_library = "nvvm/lib64/libnvvm.so.4",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)
""",
    "cuda_nvrtc": """\
cc_import(
    name = "nvrtc",
    shared_library = "lib/libnvrtc.so.12",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
    deps = [":nvrtc_builtins"],
)

cc_import(
    name = "nvrtc_builtins",
    shared_library = "lib/libnvrtc-builtins.so.12.6",
)
""",
    "libcublas": """\
cc_import(
    name = "cublasLt",
    shared_library = "lib/libcublasLt.so.12",
)

cc_import(
    name = "cublas",
    shared_library = "lib/libcublas.so.12",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
    deps = [":cublasLt"],
)
""",
}

CUDNN_PACKAGES = {
    "cudnn": """\
cc_import(
    name = "cudnn",
    shared_library = "lib/libcudnn.so.9",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
    deps = [
        ":cudnn_adv",
        ":cudnn_ops",
        ":cudnn_cnn",
        ":cudnn_graph",
        ":cudnn_engines_precompiled",
        ":cudnn_engines_runtime_compiled",
        ":cudnn_heuristic",
    ],
)

cc_import(
    name = "cudnn_adv",
    shared_library = "lib/libcudnn_adv.so.9",
)

cc_import(
    name = "cudnn_ops",
    shared_library = "lib/libcudnn_ops.so.9",
)

cc_import(
    name = "cudnn_cnn",
    shared_library = "lib/libcudnn_cnn.so.9",
    deps = [":cudnn_ops"],
)

cc_import(
    name = "cudnn_graph",
    shared_library = "lib/libcudnn_graph.so.9",
)

cc_import(
    name = "cudnn_engines_precompiled",
    shared_library = "lib/libcudnn_engines_precompiled.so.9",
)

cc_import(
    name = "cudnn_engines_runtime_compiled",
    shared_library = "lib/libcudnn_engines_runtime_compiled.so.9",
)

cc_import(
    name = "cudnn_heuristic",
    shared_library = "lib/libcudnn_heuristic.so.9",
)
""",
}

def _cuda_impl(mctx):
    CUDA_REDIST = json.decode(mctx.read(Label("@zml//runtimes/cuda:cuda.redistrib_{}.json".format(CUDA_VERSION))))
    CUDNN_REDIST = json.decode(mctx.read(Label("@zml//runtimes/cuda:cudnn.redistrib_{}.json".format(CUDNN_VERSION))))

    for pkg, build_file_content in CUDA_PACKAGES.items():
        pkg_data = CUDA_REDIST[pkg]
        arch_data = pkg_data.get(ARCH)
        if not arch_data:
            continue
        http_archive(
            name = pkg,
            build_file_content = build_file_content,
            url = "https://developer.download.nvidia.com/compute/cuda/redist/" + arch_data["relative_path"],
            sha256 = arch_data["sha256"],
            strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
        )

    for pkg, build_file_content in CUDNN_PACKAGES.items():
        pkg_data = CUDNN_REDIST[pkg]
        arch_data = pkg_data.get(ARCH)
        if not arch_data:
            continue
        arch_data = arch_data.get("cuda12", arch_data)
        http_archive(
            name = pkg,
            build_file_content = build_file_content,
            url = "https://developer.download.nvidia.com/compute/cudnn/redist/" + arch_data["relative_path"],
            sha256 = arch_data["sha256"],
            strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
        )

    http_deb_archive(
        name = "libnccl",
        urls = ["https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnccl2_2.22.3-1+cuda12.6_amd64.deb"],
        sha256 = "2f64685bcd503150ab45d00503236a56da58a15eac5fd36508045a74f4e10678",
        build_file_content = """\
cc_import(
    name = "nccl",
    shared_library = "usr/lib/x86_64-linux-gnu/libnccl.so.2",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)
""",
    )
    http_deb_archive(
        name = "zlib",
        urls = ["http://archive.ubuntu.com/ubuntu/pool/main/z/zlib/zlib1g_1.3.dfsg-3.1ubuntu2.1_amd64.deb"],
        sha256 = "7074b6a2f6367a10d280c00a1cb02e74277709180bab4f2491a2f355ab2d6c20",
        build_file_content = """\
cc_import(
    name = "zlib",
    shared_library = "usr/lib/x86_64-linux-gnu/libz.so.1",
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)
""",
    )
    http_archive(
        name = "libpjrt_cuda",
        build_file = "libpjrt_cuda.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v0.1.13/pjrt-cuda_linux-amd64.tar.gz",
        sha256 = "b705f761e24d85ecd750df992a88715d9c461b7561c31722b9f878eeab32f39e",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_cuda"],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(
    implementation = _cuda_impl,
)
