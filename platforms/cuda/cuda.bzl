load("@bazel_skylib//lib:paths.bzl", "paths")
load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

ARCHS = ["linux-x86_64", "linux-sbsa"]

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VERSION = "13.1.1"
CUDA_VARIANT = "cuda13.1"
CUDA_REDIST_JSON_SHA256 = "97cf605ccc4751825b1865f4af571c9b50dd29ffd13e9a38b296a9ecb1f0d422"

CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
CUDNN_VERSION = "9.19.1"
CUDNN_REDIST_JSON_SHA256 = "ee7bd6872b8611017bfc9ac99a4a71932652d1851b5917aa2c66bf29a12f8fd4"

NVSHMEM_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"
NVSHMEM_VERSION = "3.5.19"
NVSHMEM_REDIST_JSON_SHA256 = "6dced4193eb728542504b346cfb768da6e3de2abca0cded95fda3a69729994d2"

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@zml//bazel:patchelf.bzl", "patchelf")
"""

CUDA_COMPAT_FILES = [
    "libcuda.so.1",
    "libcudadebugger.so.1",
    "libnvidia-nvvm.so.4",
    "libnvidia-nvvm70.so.4",
    "libnvidia-ptxjitcompiler.so.1",
]

CUDA_PACKAGES = {
    "cuda_nvml_dev": "\n".join([
        packages.cc_library(
            name = "nvml",
            hdrs = ["include/nvml.h"],
            includes = ["include"],
            visibility = ["//visibility:public"],
        ),
    ]),
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
    "cuda_compat": "\n".join([
        packages.patchelf(
            name = "{}.patchelf".format(file),
            set_rpath = "$ORIGIN",
            src = "compat/{}".format(file),
            soname = file,
        )
        for file in CUDA_COMPAT_FILES
    ] + [
        packages.filegroup(
            name = "cuda_compat",
            srcs = [
                "compat/libnvidia-gpucomp.so.590.48.01",
                "compat/libnvidia-tileiras.so.590.48.01",
                # "compat/libnvidia-pkcs11-openssl3.so.590.48.01",
            ] + [
                ":{}.patchelf".format(file)
                for file in CUDA_COMPAT_FILES
            ],
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
    ]),
    "cuda_nvrtc": "\n".join([
        packages.filegroup(
            name = "cuda_nvrtc",
            srcs = [
                "lib/libnvrtc.so.13",
                "lib/libnvrtc-builtins.so.13.1",
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
            "lib/nvshmem_transport_ibrc.so.4",
        ],
    ),
}

_UBUNTU_PACKAGES = {
    "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}

PJRT_CUDA_RELEASE = "manual-2026-04-21T10-00-00Z"

_PJRT_CUDA_ASSETS = {
    "amd64": {
        "sha256": "8e34f4ead657b697e1c670cb35acb562bee9f5ff31948411d1b8ad11416df417",
        "url": "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-cuda_linux-amd64.tar.gz",
        "nccl_url": "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.29.3-py3-none-manylinux_2_18_x86_64.whl",
        "nccl_sha256": "2a321629f49490e4e0122ecb578a4b4a6f89e72740dd988e04dfa4758fab7fc3",
    },
    "arm64": {
        "sha256": "54da63a8342f848e42fb26ea1f854db5211b55f9abc753e93ad06ae2964bcb44",
        "url": "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-cuda_linux-arm64.tar.gz",
        "nccl_url": "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.29.3-py3-none-manylinux_2_18_aarch64.whl",
        "nccl_sha256": "eab9f5c565ab3326906f1d1b5be5773a174c2a1b47002faed76f9e957392f713",
    },
}

def _repo_suffix(arch):
    return "linux_{}".format(arch)

def _repo_name(name, arch):
    return "{}_{}".format(name, _repo_suffix(arch))

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

    for pkg, build_file_content in CUDA_PACKAGES.items():
        pkg_data = CUDA_REDIST[pkg]
        for arch in ARCHS:
            pkg_repo_name = pkg + "_" + arch.replace("-", "_")
            arch_data = pkg_data.get(arch)
            if not arch_data:
                fail("CUDA redist package {} does not have data for architecture {}".format(pkg, arch))
            arch_data = arch_data.get(CUDA_VARIANT, None) or arch_data
            http_archive(
                name = pkg_repo_name,
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
                url = CUDA_REDIST_PREFIX + arch_data["relative_path"],
                sha256 = arch_data["sha256"],
                strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
            )

    for pkg, build_file_content in CUDNN_PACKAGES.items():
        for arch in ARCHS:
            pkg_repo_name = pkg + "_" + arch.replace("-", "_")
            pkg_data = CUDNN_REDIST[pkg]
            arch_data = pkg_data.get(arch)
            if not arch_data:
                fail("CUDA redist package {} does not have data for architecture {}".format(pkg, arch))
            arch_data = arch_data.get("cuda13", arch_data)
            http_archive(
                name = pkg_repo_name,
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
                url = CUDNN_REDIST_PREFIX + arch_data["relative_path"],
                sha256 = arch_data["sha256"],
                strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
            )

    for pkg, build_file_content in NVSHMEM_PACKAGES.items():
        for arch in ARCHS:
            pkg_repo_name = pkg + "_" + arch.replace("-", "_")
            pkg_data = NVSHMEM_REDIST[pkg]
            arch_data = pkg_data.get(arch)
            if not arch_data:
                fail("CUDA redist package {} does not have data for architecture {}".format(pkg, arch))
            arch_data = arch_data.get("cuda13", arch_data)
            http_archive(
                name = pkg_repo_name,
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
                url = NVSHMEM_REDIST_PREFIX + arch_data["relative_path"],
                sha256 = arch_data["sha256"],
                strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
            )

    #TODO(cerisier): for each architecture
    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[pkg_name]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    for arch, arch_config in _PJRT_CUDA_ASSETS.items():
        http_archive(
            name = _repo_name("nccl", arch),
            urls = [arch_config["nccl_url"]],
            type = "zip",
            sha256 = arch_config["nccl_sha256"],
            build_file_content = "\n".join([
                _BUILD_FILE_DEFAULT_VISIBILITY,
                packages.filegroup(
                    name = "nccl",
                    srcs = ["nvidia/nccl/lib/libnccl.so.2"],
                ),
            ]),
        )

        http_archive(
            name = _repo_name("libpjrt_cuda", arch),
            build_file = "libpjrt_cuda.BUILD.bazel",
            url = arch_config["url"].format(release = PJRT_CUDA_RELEASE),
            sha256 = arch_config["sha256"],
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libpjrt_cuda_linux_amd64",
            "libpjrt_cuda_linux_arm64",
        ],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(
    implementation = _cuda_impl,
)
