load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

PJRT_ONEAPI_RELEASE = "manual-2026-07-03T15-19-00Z"
PJRT_ONEAPI_ARTIFACT_SHA256 = "dece3736b9521485885770c82f335a5b13dbcf61f53cce7ac9f6c63226eb5293"
PJRT_ONEAPI_ARTIFACT_URL = "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-oneapi_linux-amd64.tar.gz".format(
    release = PJRT_ONEAPI_RELEASE,
)

ONEAPI_VERSION = "2026.0"
ONEAPI_TCM_VERSION = "1.5"
ONEAPI_UMF_VERSION = "1.1"

ONEAPI_CCL_LIB = "ccl/2022.0/lib"
ONEAPI_COMPILER_LIB = "compiler/{}/lib".format(ONEAPI_VERSION)
ONEAPI_MPI_LIB = "mpi/2021.18/lib"
ONEAPI_MPI_LIBFABRIC_LIB = "mpi/2021.18/opt/mpi/libfabric/lib"
ONEAPI_MKL_LIB = "mkl/{}/lib".format(ONEAPI_VERSION)
ONEAPI_TCM_LIB = "tcm/{}/lib".format(ONEAPI_TCM_VERSION)
ONEAPI_UMF_LIB = "umf/{}/lib".format(ONEAPI_UMF_VERSION)

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_UBUNTU_STRIP_PREFIX = "./usr"
_ONEAPI_STRIP_PREFIX = "./opt/intel/oneapi"

def _read_packages(mctx, labels):
    ret = {}
    for label in labels:
        data = json.decode(mctx.read(Label(label)))
        for pkg in data["packages"]:
            ret.setdefault(pkg["name"], {})[pkg["arch"]] = pkg
    return ret

_ROOT_PACKAGES = {
    "libnl-3-200": """
filegroup(
    name = "libnl_3",
    srcs = glob(["lib/x86_64-linux-gnu/libnl-3.so.200*"]),
)""",
    "libnl-genl-3-200": """
filegroup(
    name = "libnl_genl_3",
    srcs = glob(["lib/x86_64-linux-gnu/libnl-genl-3.so.200*"]),
)""",
}

_UBUNTU_PACKAGES = {
    "libnl-route-3-200": """
filegroup(
    name = "libnl_route_3",
    srcs = glob(["lib/x86_64-linux-gnu/libnl-route-3.so.200*"]),
)""",
    "libnuma1": """
filegroup(
    name = "libnuma1",
    srcs = glob(["lib/x86_64-linux-gnu/libnuma.so.1*"]),
)""",
    "libigdgmm12": """
filegroup(
    name = "libigdgmm12",
    srcs = glob(["lib/x86_64-linux-gnu/libigdgmm.so.12*"]),
)""",
    "libigc2": """
genrule(
    name = "libigc_so_2",
    srcs = glob(["lib/x86_64-linux-gnu/libigc.so.2.*"]),
    outs = ["lib/x86_64-linux-gnu/libigc.so.2"],
    cmd = "cp $(SRCS) $@",
)
genrule(
    name = "libiga64_so_2",
    srcs = glob(["lib/x86_64-linux-gnu/libiga64.so.2.*"]),
    outs = ["lib/x86_64-linux-gnu/libiga64.so.2"],
    cmd = "cp $(SRCS) $@",
)
filegroup(
    name = "libigc2",
    srcs = [
        ":libiga64_so_2",
        ":libigc_so_2",
    ] + glob([
        "lib/x86_64-linux-gnu/libiga64.so.2.*",
        "lib/x86_64-linux-gnu/libigc.so.2.*",
    ]),
)""",
    "libigdfcl2": """
genrule(
    name = "libigdfcl_so_2",
    srcs = glob(["lib/x86_64-linux-gnu/libigdfcl.so.2.*"]),
    outs = ["lib/x86_64-linux-gnu/libigdfcl.so.2"],
    cmd = "cp $(SRCS) $@",
)
filegroup(
    name = "libigdfcl2",
    srcs = glob([
        "lib/x86_64-linux-gnu/libigdfcl.so.2.*",
        "lib/x86_64-linux-gnu/libopencl-clang.so.*",
    ]),
)""",
    "libze-intel-gpu1": """
filegroup(
    name = "libze_intel_gpu_so",
    srcs = glob(["lib/x86_64-linux-gnu/libze_intel_gpu.so.1.*"]),
)
filegroup(
    name = "libze_intel_gpu",
    srcs = glob(["lib/x86_64-linux-gnu/libze_intel_gpu.so.1*"]),
)""",
    "libze1": """
filegroup(
    name = "libze1",
    srcs = glob(["lib/x86_64-linux-gnu/libze*.so.1*"]),
)""",
    "zlib1g": """
filegroup(
    name = "zlib1g",
    srcs = ["lib/x86_64-linux-gnu/libz.so.1"],
)""",
}

_ONEAPI_PACKAGES = {
    "intel-oneapi-ccl-2022.0": """
filegroup(
    name = "ccl_runtime",
    srcs = glob(
        include = ["{ONEAPI_CCL_LIB}/**"],
        exclude = ["{ONEAPI_CCL_LIB}/libccl_legacy.so"],
    ),
)""".format(ONEAPI_CCL_LIB = ONEAPI_CCL_LIB),
    "intel-oneapi-mpi-2021.18": """
filegroup(
    name = "mpi_runtime",
    srcs = glob([
        "{ONEAPI_MPI_LIB}/libmpi*.so*",
        "{ONEAPI_MPI_LIB}/libmpicxx*.so*",
        "{ONEAPI_MPI_LIB}/libmpifort*.so*",
        "{ONEAPI_MPI_LIB}/mpi/libmpi_ze_hooks.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/libfabric.so*",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/*.so",
    ]),
)""".format(
        ONEAPI_MPI_LIB = ONEAPI_MPI_LIB,
        ONEAPI_MPI_LIBFABRIC_LIB = ONEAPI_MPI_LIBFABRIC_LIB,
    ),
    "intel-oneapi-tcm-1.5": """
filegroup(
    name = "hwloc",
    srcs = ["{ONEAPI_TCM_LIB}/libhwloc.so.15"],
)""".format(ONEAPI_TCM_LIB = ONEAPI_TCM_LIB),
    "intel-oneapi-umf-1.1": """
filegroup(
    name = "umf",
    srcs = ["{ONEAPI_UMF_LIB}/libumf.so.1"],
)""".format(ONEAPI_UMF_LIB = ONEAPI_UMF_LIB),
    "intel-oneapi-compiler-dpcpp-cpp-runtime-2026.0": """
filegroup(
    name = "libsycl_so",
    srcs = ["{ONEAPI_COMPILER_LIB}/libsycl.so.9"],
)
filegroup(
    name = "sycl_runtime",
    srcs = glob(["{ONEAPI_COMPILER_LIB}/*.spv"]) + [
        "{ONEAPI_COMPILER_LIB}/libur_adapter_level_zero.so.0",
        "{ONEAPI_COMPILER_LIB}/libur_adapter_level_zero_v2.so.0",
        "{ONEAPI_COMPILER_LIB}/libur_adapter_opencl.so.0",
        "{ONEAPI_COMPILER_LIB}/libur_loader.so.0",
    ],
)""".format(ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB),
    "intel-oneapi-compiler-shared-runtime-2026.0": """
filegroup(
    name = "compiler_runtime",
    srcs = glob(["{ONEAPI_COMPILER_LIB}/libOpenCL.so*"]) + [
        "{ONEAPI_COMPILER_LIB}/libimf.so",
        "{ONEAPI_COMPILER_LIB}/libintlc.so.5",
        "{ONEAPI_COMPILER_LIB}/libirc.so",
        "{ONEAPI_COMPILER_LIB}/libirng.so",
        "{ONEAPI_COMPILER_LIB}/libsvml.so",
    ],
)""".format(ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB),
    "intel-oneapi-mkl-core-2026.0": """
filegroup(
    name = "mkl_core_runtime",
    srcs = [
        "{ONEAPI_MKL_LIB}/libmkl_core.so.3",
        "{ONEAPI_MKL_LIB}/libmkl_intel_ilp64.so.3",
        "{ONEAPI_MKL_LIB}/libmkl_sequential.so.3",
    ],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-blas-2026.0": """
filegroup(
    name = "mkl_sycl_blas",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_blas.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-dft-2026.0": """
filegroup(
    name = "mkl_sycl_dft",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_dft.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-lapack-2026.0": """
filegroup(
    name = "mkl_sycl_lapack",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_lapack.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-rng-2026.0": """
filegroup(
    name = "mkl_sycl_rng",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_rng.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-sparse-2026.0": """
filegroup(
    name = "mkl_sycl_sparse",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_sparse.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
}

_ROOT_MODULE_DIRECT_DEPS = [
    "intel-oneapi-ccl-2022.0",
    "intel-oneapi-compiler-dpcpp-cpp-runtime-2026.0",
    "intel-oneapi-compiler-shared-runtime-2026.0",
    "intel-oneapi-mkl-core-2026.0",
    "intel-oneapi-mkl-sycl-blas-2026.0",
    "intel-oneapi-mkl-sycl-dft-2026.0",
    "intel-oneapi-mkl-sycl-lapack-2026.0",
    "intel-oneapi-mkl-sycl-rng-2026.0",
    "intel-oneapi-mkl-sycl-sparse-2026.0",
    "intel-oneapi-mpi-2021.18",
    "intel-oneapi-tcm-1.5",
    "intel-oneapi-umf-1.1",
    "libnl-3-200",
    "libnl-genl-3-200",
    "libnl-route-3-200",
    "libnuma1",
    "libigdgmm12",
    "libigc2",
    "libigdfcl2",
    "libze-intel-gpu1",
    "libze1",
    "libpjrt_oneapi",
    "zlib1g",
]

def _oneapi_impl(mctx):
    loaded_packages = _read_packages(mctx, [
        "@zml//platforms/oneapi:packages.lock.json",
    ])

    http_archive(
        name = "libpjrt_oneapi",
        build_file = "libpjrt_oneapi.BUILD.bazel",
        sha256 = PJRT_ONEAPI_ARTIFACT_SHA256,
        url = PJRT_ONEAPI_ARTIFACT_URL,
    )

    for pkg_name, build_file_content in _ROOT_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    for pkg_name, build_file_content in _UBUNTU_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = _UBUNTU_STRIP_PREFIX,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    for pkg_name, build_file_content in _ONEAPI_PACKAGES.items():
        pkg = loaded_packages[pkg_name]["amd64"]
        http_deb_archive(
            name = pkg_name,
            urls = pkg["urls"],
            sha256 = pkg["sha256"],
            strip_prefix = _ONEAPI_STRIP_PREFIX,
            build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + build_file_content,
        )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = _ROOT_MODULE_DIRECT_DEPS,
        root_module_direct_dev_deps = [],
    )

oneapi_packages = module_extension(
    implementation = _oneapi_impl,
)
