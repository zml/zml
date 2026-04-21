load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

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

def _cuda_impl(mctx):
    loaded_packages = packages.read(mctx, [
        "@zml//platforms/cuda:packages.lock.json",
    ])

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
