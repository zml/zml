load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//runtimes/common:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

ARCH = "linux-x86_64"

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VERSION = "12.8.1"
CUDA_REDIST_JSON_SHA256 = "249e28a83008d711d5f72880541c8be6253f6d61608461de4fcb715554a6cf17"

CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
CUDNN_VERSION = "9.8.0"
CUDNN_REDIST_JSON_SHA256 = "a1599fa1f8dcb81235157be5de5ab7d3936e75dfc4e1e442d07970afad3c4843"

CUDA_PACKAGES = {
    "cuda_cudart": "\n".join([
        # Driver API only
        packages.cc_library(
            name = "cuda",
            hdrs = ["include/cuda.h"],
            includes = ["include"],
        ),
        #TODO: Remove me as soon we use the Driver API in tracer.zig
        packages.filegroup(
            name = "so_files",
            srcs = ["lib/libcudart.so.12"],
        ),
    ]),
    "cuda_cupti": packages.filegroup(
        name = "so_files",
        srcs = ["lib/libcupti.so.12"],
    ),
    "cuda_nvtx": "\n".join([
        # packages.cc_library(
        #     name = "nvtx",
        #     hdrs = glob(["include/nvtx3/**/*.h"]),
        #     visibility = ["//visibility:public"],
        # ),
        packages.filegroup(
            name = "so_files",
            srcs = ["lib/libnvToolsExt.so.1"],
        ),
    ]),
    "libcufft": packages.filegroup(
        name = "so_files",
        srcs = ["lib/libcufft.so.11"],
    ),
    "libcusolver": packages.filegroup(
        name = "so_files",
        srcs = ["lib/libcusolver.so.11"],
    ),
    "libcusparse": packages.filegroup(
        name = "so_files",
        srcs = ["lib/libcusparse.so.12"],
    ),
    "libnvjitlink": packages.filegroup(
        name = "so_files",
        srcs = ["lib/libnvJitLink.so.12"],
    ),
    "cuda_nvcc": "\n".join([
        packages.filegroup(
            name = "ptxas",
            srcs = ["bin/ptxas"],
        ),
        packages.filegroup(
            name = "nvlink",
            srcs = ["bin/nvlink"],
        ),
        packages.filegroup(
            name = "libdevice",
            srcs = ["nvvm/libdevice/libdevice.10.bc"],
        ),
        packages.filegroup(
            name = "so_files",
            srcs = ["nvvm/lib64/libnvvm.so.4"],
        ),
        packages.cc_import(
            name = "nvptxcompiler",
            static_library = "lib/libnvptxcompiler_static.a",
        ),
    ]),
    "cuda_nvrtc": "\n".join([
        packages.filegroup(
            name = "so_files",
            srcs = [
                "lib/libnvrtc.so.12",
                "lib/libnvrtc-builtins.so.12.8",
            ],
        ),
    ]),
    "libcublas": "\n".join([
        packages.filegroup(
            name = "so_files",
            srcs = [
                "lib/libcublasLt.so.12",
                "lib/libcublas.so.12",
            ],
        ),
    ]),
}

CUDNN_PACKAGES = {
    "cudnn": "\n".join([
        packages.filegroup(
            name = "so_files",
            srcs = [
                "lib/libcudnn.so.9",
                "lib/libcudnn_adv.so.9",
                "lib/libcudnn_ops.so.9",
                "lib/libcudnn_cnn.so.9",
                "lib/libcudnn_graph.so.9",
                "lib/libcudnn_engines_precompiled.so.9",
                "lib/libcudnn_engines_runtime_compiled.so.9",
                "lib/libcudnn_heuristic.so.9",
            ],
        ),
    ]),
}

def _read_redist_json(mctx, url, sha256):
    fname = ".{}.json".format(sha256)
    mctx.download(
        url = url,
        output = fname,
        sha256 = sha256,
    )
    return json.decode(mctx.read(fname))

def _cuda_impl(mctx):
    CUDA_REDIST = _read_redist_json(
        mctx,
        url = CUDA_REDIST_PREFIX + "redistrib_{}.json".format(CUDA_VERSION),
        sha256 = CUDA_REDIST_JSON_SHA256,
    )

    CUDNN_REDIST = _read_redist_json(
        mctx,
        url = CUDNN_REDIST_PREFIX + "redistrib_{}.json".format(CUDNN_VERSION),
        sha256 = CUDNN_REDIST_JSON_SHA256,
    )

    for pkg, build_file_content in CUDA_PACKAGES.items():
        pkg_data = CUDA_REDIST[pkg]
        arch_data = pkg_data.get(ARCH)
        if not arch_data:
            continue
        http_archive(
            name = pkg,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
            url = CUDA_REDIST_PREFIX + arch_data["relative_path"],
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
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
            url = CUDNN_REDIST_PREFIX + arch_data["relative_path"],
            sha256 = arch_data["sha256"],
            strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
        )

    http_archive(
        name = "nccl",
        urls = ["https://files.pythonhosted.org/packages/11/0c/8c78b7603f4e685624a3ea944940f1e75f36d71bd6504330511f4a0e1557/nvidia_nccl_cu12-2.25.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"],
        type = "zip",
        sha256 = "362aed5963fb9ea2ed2f264409baae30143498fd0e5c503aeaa1badd88cdc54a",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + packages.filegroup(
            name = "so_files",
            srcs = ["nvidia/nccl/lib/libnccl.so.2"],
        ),
    )

    http_archive(
        name = "libpjrt_cuda",
        build_file = "libpjrt_cuda.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v10.0.0/pjrt-cuda_linux-amd64.tar.gz",
        sha256 = "eddf4db325aaeb1692e9eff1b5021dbeda27c08e527cae87295a61d94e654395",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_cuda"],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(
    implementation = _cuda_impl,
)
