load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

ARCH = "linux-x86_64"

CUDA_VERSION = "12.6.3"
CUDNN_VERSION = "9.5.1"

def _filegroup(name, srcs):
    return """\
filegroup(
    name = {name},
    srcs = {srcs},
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)
""".format(name = repr(name), srcs = repr(srcs))

def _cc_import(name, shared_library, deps = []):
    return """\
cc_import(
    name = {name},
    shared_library = {shared_library},
    deps = {deps},
    visibility = ["@libpjrt_cuda//:__subpackages__"],
)
""".format(name = repr(name), shared_library = repr(shared_library), deps = repr(deps))

CUDA_PACKAGES = {
    "cuda_cudart": _cc_import(
        name = "cudart",
        shared_library = "lib/libcudart.so.12",
    ),
    "cuda_cupti": _cc_import(
        name = "cupti",
        shared_library = "lib/libcupti.so.12",
    ),
    "libcufft": _cc_import(
        name = "cufft",
        shared_library = "lib/libcufft.so.11",
    ),
    "libcusolver": _cc_import(
        name = "cusolver",
        shared_library = "lib/libcusolver.so.11",
    ),
    "libcusparse": _cc_import(
        name = "cusparse",
        shared_library = "lib/libcusparse.so.12",
    ),
    "libnvjitlink": _cc_import(
        name = "nvjitlink",
        shared_library = "lib/libnvJitLink.so.12",
    ),
    "cuda_nvcc": "\n".join([
        _filegroup(
            name = "ptxas",
            srcs = ["bin/ptxas"],
        ),
        _filegroup(
            name = "libdevice",
            srcs = ["nvvm/libdevice/libdevice.10.bc"],
        ),
        _cc_import(
            name = "nvvm",
            shared_library = "nvvm/lib64/libnvvm.so.4",
        ),
    ]),
    "cuda_nvrtc": "\n".join([
        _cc_import(
            name = "nvrtc",
            shared_library = "lib/libnvrtc.so.12",
            deps = [":nvrtc_builtins"],
        ),
        _cc_import(
            name = "nvrtc_builtins",
            shared_library = "lib/libnvrtc-builtins.so.12.6",
        ),
    ]),
    "libcublas": "\n".join([
        _cc_import(
            name = "cublasLt",
            shared_library = "lib/libcublasLt.so.12",
        ),
        _cc_import(
            name = "cublas",
            shared_library = "lib/libcublas.so.12",
            deps = [":cublasLt"],
        ),
    ]),
}

CUDNN_PACKAGES = {
    "cudnn": "\n".join([
        _cc_import(
            name = "cudnn",
            shared_library = "lib/libcudnn.so.9",
            deps = [
                ":cudnn_adv",
                ":cudnn_ops",
                ":cudnn_cnn",
                ":cudnn_graph",
                ":cudnn_engines_precompiled",
                ":cudnn_engines_runtime_compiled",
                ":cudnn_heuristic",
            ],
        ),
        _cc_import(
            name = "cudnn_adv",
            shared_library = "lib/libcudnn_adv.so.9",
        ),
        _cc_import(
            name = "cudnn_ops",
            shared_library = "lib/libcudnn_ops.so.9",
        ),
        _cc_import(
            name = "cudnn_cnn",
            shared_library = "lib/libcudnn_cnn.so.9",
            deps = [":cudnn_ops"],
        ),
        _cc_import(
            name = "cudnn_graph",
            shared_library = "lib/libcudnn_graph.so.9",
        ),
        _cc_import(
            name = "cudnn_engines_precompiled",
            shared_library = "lib/libcudnn_engines_precompiled.so.9",
        ),
        _cc_import(
            name = "cudnn_engines_runtime_compiled",
            shared_library = "lib/libcudnn_engines_runtime_compiled.so.9",
        ),
        _cc_import(
            name = "cudnn_heuristic",
            shared_library = "lib/libcudnn_heuristic.so.9",
        ),
    ]),
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

    http_archive(
        name = "nccl",
        urls = ["https://files.pythonhosted.org/packages/ed/1f/6482380ec8dcec4894e7503490fc536d846b0d59694acad9cf99f27d0e7d/nvidia_nccl_cu12-2.23.4-py3-none-manylinux2014_x86_64.whl"],
        type = "zip",
        sha256 = "b097258d9aab2fa9f686e33c6fe40ae57b27df60cedbd15d139701bb5509e0c1",
        build_file_content = _cc_import(
            name = "nccl",
            shared_library = "nvidia/nccl/lib/libnccl.so.2",
        ),
    )
    http_deb_archive(
        name = "zlib",
        urls = ["https://snapshot-cloudflare.debian.org/archive/debian/20241127T143620Z/pool/main/z/zlib/zlib1g_1.3.dfsg%2Breally1.3.1-1%2Bb1_amd64.deb"],
        sha256 = "015be740d6236ad114582dea500c1d907f29e16d6db00566ca32fb68d71ac90d",
        build_file_content = _cc_import(
            name = "zlib",
            shared_library = "usr/lib/x86_64-linux-gnu/libz.so.1",
        ),
    )

    http_archive(
        name = "libpjrt_cuda",
        build_file = "libpjrt_cuda.BUILD.bazel",
        url = "https://files.pythonhosted.org/packages/90/43/ac2c369e202e3e3e7e5aa7929b197801ba02eaf11868437adaa5341704e4/jax_cuda12_pjrt-0.4.38-py3-none-manylinux2014_x86_64.whl",
        type = "zip",
        sha256 = "83be4c59fbcf30077a60085d98e7d59dc738b1c91e0d628e4ac1779fde15ac2b",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_cuda"],
        root_module_direct_dev_deps = [],
    )

cuda_packages = module_extension(
    implementation = _cuda_impl,
)
