load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

ARCH = "linux-x86_64"

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VERSION = "13.0.2"
CUDA_REDIST_JSON_SHA256 = ""

CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
CUDNN_VERSION = "9.14.0"
CUDNN_REDIST_JSON_SHA256 = "fe58e8e9559ef5c61ab7a9954472d16acdcbad3b099004296ae410d25982830d"

NVSHMEM_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"
NVSHMEM_VERSION = "3.4.5"
NVSHMEM_REDIST_JSON_SHA256 = "a656614a6ec638d85922bc816e5e26063308c3905273a72a863cf0f24e188f38"

_UBUNTU_PACKAGES = {
    "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}

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
            name = "cuda_cudart",
            srcs = ["lib/libcudart.so.13"],
        ),
    ]),
    "cuda_cupti": packages.filegroup(
        name = "cuda_cupti",
        srcs = ["lib/libcupti.so.13"],
    ),
    "cuda_nvtx": "\n".join([
        packages.filegroup(
            name = "cuda_nvtx",
            srcs = ["lib/libnvtx3interop.so"],
        ),
    ]),
    "libcufft": packages.filegroup(
        name = "libcufft",
        srcs = ["lib/libcufft.so.12"],
    ),
    "libcusolver": packages.filegroup(
        name = "libcusolver",
        srcs = ["lib/libcusolver.so.12"],
    ),
    "libcusparse": packages.filegroup(
        name = "libcusparse",
        srcs = ["lib/libcusparse.so.12"],
    ),
    "libnvjitlink": packages.filegroup(
        name = "libnvjitlink",
        srcs = ["lib/libnvJitLink.so.13"],
    ),
    "cuda_nvcc": "\n".join([
        packages.filegroup(
            name = "cuda_nvcc",
            srcs = [
                "bin/ptxas",
                "bin/nvlink",
            ],
        ),
        packages.cc_import(
            name = "nvptxcompiler",
            static_library = "lib/libnvptxcompiler_static.a",
        ),
    ]),
    "libnvvm": "\n".join([
        packages.filegroup(
            name = "libnvvm",
            srcs = [
                "nvvm/bin/cicc",
                "nvvm/libdevice/libdevice.10.bc",
            ],
        ),
        packages.cc_import(
            name = "nvptxcompiler",
            static_library = "lib/libnvptxcompiler_static.a",
        ),
    ]),
    "cuda_nvrtc": "\n".join([
        packages.filegroup(
            name = "cuda_nvrtc",
            srcs = [
                "lib/libnvrtc.so.13",
                "lib/libnvrtc-builtins.so.13.0",
            ],
        ),
    ]),
    "libcublas": "\n".join([
        packages.filegroup(
            name = "libcublas",
            srcs = [
                "lib/libcublasLt.so.13",
                "lib/libcublas.so.13",
            ],
        ),
    ]),
}

CUDNN_PACKAGES = {
    "cudnn": "\n".join([
        packages.filegroup(
            name = "cudnn",
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

NVSHMEM_PACKAGES = {
    "libnvshmem": packages.filegroup(
        name = "libnvshmem",
        srcs = [
            "lib/libnvshmem_host.so.3",
            "lib/nvshmem_bootstrap_uid.so.3",
            "lib/nvshmem_transport_ibrc.so.3",
        ],
    ),
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
    loaded_packages = packages.read(mctx, [
        "@zml//platforms/cuda:packages.lock.json",
    ])
    CUDA_REDIST = _read_redist_json(
        mctx,
        url = CUDA_REDIST_PREFIX + "redistrib_{}.json".format(CUDA_VERSION),
        sha256 = CUDA_REDIST_JSON_SHA256,
    )

    NVSHMEM_REDIST = _read_redist_json(
        mctx,
        url = NVSHMEM_REDIST_PREFIX + "redistrib_{}.json".format(NVSHMEM_VERSION),
        sha256 = NVSHMEM_REDIST_JSON_SHA256,
    )

    CUDNN_REDIST = _read_redist_json(
        mctx,
        url = CUDNN_REDIST_PREFIX + "redistrib_{}.json".format(CUDNN_VERSION),
        sha256 = CUDNN_REDIST_JSON_SHA256,
    )

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
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

    for pkg, build_file_content in NVSHMEM_PACKAGES.items():
        pkg_data = NVSHMEM_REDIST[pkg]
        arch_data = pkg_data.get(ARCH)
        if not arch_data:
            continue
        arch_data = arch_data.get("cuda12", arch_data)
        http_archive(
            name = pkg,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
            url = NVSHMEM_REDIST_PREFIX + arch_data["relative_path"],
            sha256 = arch_data["sha256"],
            strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
        )

    http_archive(
        name = "nccl",
        urls = ["https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.27.7-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"],
        type = "zip",
        sha256 = "b28a524abd8389b76a4a3f133c76a7aaa7005e47fcaa9d9603b90103927a3f93",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + packages.filegroup(
            name = "nccl",
            srcs = ["nvidia/nccl/lib/libnccl.so.2"],
        ),
    )

    http_archive(
        name = "libpjrt_cuda",
        build_file = "libpjrt_cuda.BUILD.bazel",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v15.0.2/pjrt-cuda_linux-amd64.tar.gz",
        sha256 = "ebc5f0fa54d38ec85346f0f02b29a6497ee394b9e5fa4832da5db16d3296ae84",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_cuda"],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(
    implementation = _cuda_impl,
)
