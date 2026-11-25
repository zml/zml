load("@bazel_skylib//lib:paths.bzl", "paths")
load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

ARCHS = ["linux-x86_64", "linux-sbsa"]

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VERSION = "13.5.0"
CUDA_MAJOR_MINOR = ".".join(CUDA_VERSION.split(".")[:2])
CUDA_VARIANT = "cuda" + CUDA_MAJOR_MINOR
CUDA_REDIST_JSON_SHA256 = "401529e951b845788177f999c0756c520c4d0db1cd4cd05c0a295e331884b596"

CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
CUDNN_VERSION = "9.28.1"
CUDNN_REDIST_JSON_SHA256 = "0c579b4897afeeba4e7349beb39f7f91eefd50fb566db79201b5729974d48cc1"

NVSHMEM_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"
NVSHMEM_VERSION = "3.8.0"
NVSHMEM_REDIST_JSON_SHA256 = "5920f88ccae8261083570f740463eddfdf8ad2708d88ebede380ce55d84eaf6c"

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
    "cuda_nvml_dev": [
        packages.cc_library(
            name = "nvml",
            hdrs = ["include/nvml.h"],
            includes = ["include"],
            visibility = ["//visibility:public"],
        ),
    ],
    "cuda_cudart": [
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
    ],
    "cuda_cupti": [
        packages.filegroup(
            name = "cuda_cupti",
            srcs = ["lib/libcupti.so.13"],
        ),
    ],
    "cuda_nvtx": [
        packages.cc_library_hdrs_glob(
            name = "headers",
            hdrs_glob = ["include/nvtx3/**"],
            includes = ["include"],
        ),
        packages.filegroup(
            name = "cuda_nvtx",
            srcs = ["lib/libnvtx3interop.so"],
        ),
    ],
    "libcufft": [
        packages.filegroup(
            name = "libcufft",
            srcs = ["lib/libcufft.so.12"],
        ),
    ],
    "libcusolver": [
        packages.filegroup(
            name = "libcusolver",
            srcs = ["lib/libcusolver.so.12"],
        ),
    ],
    "libcusparse": [
        packages.filegroup(
            name = "libcusparse",
            srcs = ["lib/libcusparse.so.12"],
        ),
    ],
    "libnvjitlink": [
        packages.filegroup(
            name = "libnvjitlink",
            srcs = ["lib/libnvJitLink.so.13"],
        ),
    ],
    "cuda_nvcc": [
        packages.filegroup(
            name = "cuda_nvcc",
            srcs = [
                "bin/ptxas",
                "bin/nvlink",
            ],
        ),
    ],
    "cuda_tileiras": [
        packages.filegroup(
            name = "cuda_tileiras",
            srcs = ["bin/tileiras"],
        ),
    ],
    "libnvvm": [
        packages.filegroup(
            name = "libnvvm",
            srcs = [
                "nvvm/bin/cicc",
                "nvvm/libdevice/libdevice.10.bc",
                "nvvm/lib64/libnvvm.so",
                "nvvm/lib64/libnvvm.so.4",
                "nvvm/lib64/libnvvm.so.4.0.0",
            ],
        ),
    ],
    "cuda_nvrtc": [
        packages.filegroup(
            name = "cuda_nvrtc",
            srcs = [
                "lib/libnvrtc.so.13",
                "lib/libnvrtc-builtins.so.13.5",
            ],
        ),
    ],
    "libcublas": [
        packages.filegroup(
            name = "libcublas",
            srcs = [
                "lib/libcublasLt.so.13",
                "lib/libcublas.so.13",
            ],
        ),
    ],
}

_CUDA_COMPAT_BUILD = [
    packages.patchelf(
        name = "{}.patchelf".format(file),
        set_rpath = "$ORIGIN",
        src = "compat/{}".format(file),
        soname = file,
    )
    for file in CUDA_COMPAT_FILES
] + [
    """filegroup(
        name = "cuda_compat",
        srcs = [
            "compat/libnvidia-gpucomp.so.620.70.06",
            "compat/libnvidia-tileiras.so.620.70.06",
        ] + {} + select({{
            "@llvm//platforms/config:linux_x86_64": [
                "compat/libnvidia-pkcs11-openssl3.so.620.70.06",
            ],
            "@llvm//platforms/config:linux_aarch64": [
                "compat/libnvcuextend.so.620.70.06",
            ],
        }}),
    )""".format(repr([
        ":{}.patchelf".format(file)
        for file in CUDA_COMPAT_FILES
    ])),
]

CUDNN_PACKAGES = {
    "cudnn": [
        packages.filegroup(
            name = "cudnn",
            srcs = [
                "lib/libcudnn.so.9",
                "lib/libcudnn_ext.so.9",
                "lib/libcudnn_adv.so.9",
                "lib/libcudnn_ops.so.9",
                "lib/libcudnn_cnn.so.9",
                "lib/libcudnn_graph.so.9",
                "lib/libcudnn_engines_precompiled.so.9",
                "lib/libcudnn_engines_runtime_compiled.so.9",
                "lib/libcudnn_engines_tensor_ir.so.9",
                "lib/libcudnn_heuristic.so.9",
            ],
        ),
    ],
}

NVSHMEM_PACKAGES = {
    "libnvshmem": [
        packages.filegroup(
            name = "libnvshmem",
            srcs = [
                "lib/libnvshmem_host.so.3",
                "lib/nvshmem_bootstrap_uid.so.3",
                "lib/nvshmem_transport_ibrc.so.7",
            ],
        ),
    ],
}

_UBUNTU_PACKAGES = {
    "llvm-libunwind1": [
        """filegroup(
            name = "libunwind",
            srcs = select({
                "@llvm//platforms/config:linux_x86_64": ["usr/lib/x86_64-linux-gnu/libunwind.so.1"],
                "@llvm//platforms/config:linux_aarch64": ["usr/lib/aarch64-linux-gnu/libunwind.so.1"],
            }),
        )""",
    ],
    "zlib1g": [
        """filegroup(
            name = "zlib1g",
            srcs = select({
                "@llvm//platforms/config:linux_x86_64": ["lib/x86_64-linux-gnu/libz.so.1"],
                "@llvm//platforms/config:linux_aarch64": ["lib/aarch64-linux-gnu/libz.so.1"],
            }),
        )""",
    ],
}

_PLUGINS = {
    "amd64": {
        "sha256": "58853fa18f3f2e1a331656d06f647caf0cd5ef47375ad9e2a8b1c75731bdb18c",
        "url": "https://mirror.zml.ai/plugins/202611251235.407.1.475061b3c599/zml-cuda-linux-amd64.tar.zst",
    },
    "arm64": {
        "sha256": "36db83c099b095f14e7141e0d4c9a02d04245fe75d66dae5f36ca3e4c1aa3082",
        "url": "https://mirror.zml.ai/plugins/202611251235.407.1.475061b3c599/zml-cuda-linux-arm64.tar.zst",
    },
}

_NCCL_ASSETS = {
    "amd64": {
        "url": "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.32.6-py3-none-manylinux_2_27_x86_64.whl",
        "sha256": "76e7aad00482092baf01e13e66f441d54c96b5cf1e3f143dee27db7f731d41ba",
    },
    "arm64": {
        "url": "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.32.6-py3-none-manylinux_2_27_aarch64.whl",
        "sha256": "ee63091d0c00f5a0419e04d7f59f6dd6ebd54b14931cb0f9e3a626b91243233d",
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

    #TODO(cerisier): for each architecture
    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkgs = loaded_packages[pkg_name]
        for arch, pkg in pkgs.items():
            http_deb_archive(
                name = _repo_name(pkg_name, arch),
                urls = pkg["urls"],
                sha256 = pkg["sha256"],
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + "\n".join(build_file_content),
            )

    for arch, pkg in loaded_packages["cuda-compat-" + CUDA_MAJOR_MINOR.replace(".", "-")].items():
        http_deb_archive(
            name = "cuda_compat_linux_" + {"amd64": "x86_64", "arm64": "sbsa"}[arch],
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = "./usr/local/cuda-" + CUDA_MAJOR_MINOR,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + "\n".join(_CUDA_COMPAT_BUILD),
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
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + "\n".join(build_file_content),
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
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + "\n".join(build_file_content),
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
                build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + "\n".join(build_file_content),
                url = NVSHMEM_REDIST_PREFIX + arch_data["relative_path"],
                sha256 = arch_data["sha256"],
                strip_prefix = paths.basename(arch_data["relative_path"]).replace(".tar.xz", ""),
            )

    for arch, arch_config in _NCCL_ASSETS.items():
        http_archive(
            name = _repo_name("nccl", arch),
            urls = [arch_config["url"]],
            type = "zip",
            sha256 = arch_config["sha256"],
            build_file_content = "\n".join([
                _BUILD_FILE_DEFAULT_VISIBILITY,
                packages.filegroup(
                    name = "nccl",
                    srcs = ["nvidia/nccl/lib/libnccl.so.2"],
                ),
            ]),
        )

    for arch, arch_config in _PLUGINS.items():
        http_archive(
            name = _repo_name("libzml_cuda", arch),
            build_file = "libzml_cuda.BUILD.bazel",
            url = arch_config["url"],
            sha256 = arch_config["sha256"],
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libzml_cuda_linux_amd64",
            "libzml_cuda_linux_arm64",
            "cuda_nvml_dev_linux_x86_64",
            "cuda_nvml_dev_linux_sbsa",
            "zlib1g_linux_arm64",
            "zlib1g_linux_amd64",
            "cuda_nvtx_linux_x86_64",
            "cuda_nvtx_linux_sbsa",
        ],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(
    implementation = _cuda_impl,
)
