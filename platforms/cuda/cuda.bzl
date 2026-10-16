load("@bazel_skylib//lib:paths.bzl", "paths")
load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

ARCHS = ["linux-x86_64", "linux-sbsa"]

CUDA_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VERSION = "13.4.3"
CUDA_MAJOR_MINOR = ".".join(CUDA_VERSION.split(".")[:2])
CUDA_VARIANT = "cuda{}".format(CUDA_MAJOR_MINOR)
CUDA_REDIST_JSON_SHA256 = "8e99511eee1bd78664bdb6cf43dc17a7ba6afeab78c8162bd5f23571ed6423c6"

CUDNN_REDIST_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
CUDNN_VERSION = "9.27.0"
CUDNN_REDIST_JSON_SHA256 = "d3f870db59da3fd12581aeca5fda381afa50ce1d4d9cccc8dcd79d7a628c22a2"

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

def _cuda_compat_packages(driver_version):
    return [
        packages.patchelf(
            name = "{}.patchelf".format(file),
            set_rpath = "$ORIGIN",
            src = "compat/{}".format(file),
            soname = file,
        )
        for file in CUDA_COMPAT_FILES
    ] + [
        packages.patchelf(
            name = "nvcuextend",
            set_rpath = "$ORIGIN",
            src = "compat/libnvcuextend.so.1",
            soname = "libnvcuextend.so.1",
        ),
        """filegroup(
                name = "cuda_compat",
                srcs = [
                    "compat/libnvidia-gpucomp.so.{driver_version}",
                    "compat/libnvidia-tileiras.so.{driver_version}",
                ] + {} + select({{
                    "@llvm//platforms/config:linux_x86_64": [
                        "compat/libnvidia-pkcs11-openssl3.so.{driver_version}",
                    ],
                    "@llvm//platforms/config:linux_aarch64": [
                        ":nvcuextend",
                    ],
                }}),
            )""".format(repr([
            ":{}.patchelf".format(file)
            for file in CUDA_COMPAT_FILES
        ]), driver_version = driver_version),
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
                "lib/libnvrtc-builtins.so.13.4",
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

CUDNN_PACKAGES = {
    "cudnn": [
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
                "lib/libcudnn_engines_tensor_ir.so.9",
                "lib/libcudnn_heuristic.so.9",
                "lib/libcudnn_ext.so.9",
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
        "sha256": "b1b33616999bfe479dfff050e6934f7b2d935c08621a269014aff585925614c2",
        "url": "https://mirror.zml.ai/plugins/202610160837.234.1.e1adc5dca29d/zml-cuda-linux-amd64.tar.zst",
    },
    "arm64": {
        "sha256": "c7eb68169e0976157606c96f10be8b15066a9be661ae91369996bbe303a3cc3b",
        "url": "file:///tmp/claude-46326/-home-benjamin-github-openxla-xla/fd2bd4cb-e7f4-43f1-8087-c8f42f8bc175/scratchpad/zml-cuda-bound-args-fixed-arm64.tar.zst",
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

    compat_package = "cuda-compat-{}".format(CUDA_MAJOR_MINOR.replace(".", "-"))
    for arch, pkg in loaded_packages[compat_package].items():
        http_deb_archive(
            name = "cuda_compat_linux_{}".format({"amd64": "x86_64", "arm64": "sbsa"}[arch]),
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = "./usr/local/cuda-{}".format(CUDA_MAJOR_MINOR),
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + "\n".join(_cuda_compat_packages(pkg["version"].split("-")[0])),
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

    for arch, pkg in loaded_packages["libnccl2"].items():
        http_deb_archive(
            name = _repo_name("nccl", arch),
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = "\n".join([
                _BUILD_FILE_DEFAULT_VISIBILITY,
                packages.filegroup(
                    name = "nccl",
                    srcs = ["usr/lib/{}-linux-gnu/libnccl.so.2".format({"amd64": "x86_64", "arm64": "aarch64"}[arch])],
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
