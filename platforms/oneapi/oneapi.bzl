load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")

PJRT_ONEAPI_RELEASE = "manual-2026-07-17T10-00-00Z"
PJRT_ONEAPI_ARTIFACT_SHA256 = "89ce473e700f56270b1c1309f4804d8ddc310356aa691701a2cace5695bf84a2"
PJRT_ONEAPI_ARTIFACT_URL = "file:///home/brabier/github/openxla/pjrt-oneapi_linux-amd64-2026-07-19_22-43.tar"

ONEAPI_VERSION = "2026.1"
ONEAPI_TCM_VERSION = "1.5"
ONEAPI_UMF_VERSION = "1.1"

ONEAPI_CCL_LIB = "ccl/2022.1/lib"
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

_UBUNTU_PACKAGES = {
    "libnl-3-200": """
filegroup(
    name = "libnl_3",
    srcs = ["lib/x86_64-linux-gnu/libnl-3.so.200"],
)""",
    "libnl-genl-3-200": """
filegroup(
    name = "libnl_genl_3",
    srcs = ["lib/x86_64-linux-gnu/libnl-genl-3.so.200"],
)""",
    "libnl-route-3-200": """
filegroup(
    name = "libnl_route_3",
    srcs = ["lib/x86_64-linux-gnu/libnl-route-3.so.200"],
)""",
    "libnuma1": """
filegroup(
    name = "libnuma1",
    srcs = ["lib/x86_64-linux-gnu/libnuma.so.1"],
)""",
    "libigdgmm12": """
filegroup(
    name = "libigdgmm12",
    srcs = ["lib/x86_64-linux-gnu/libigdgmm.so.12"],
)""",
    "libigc2": """
filegroup(
    name = "libigc_so_2",
    srcs = ["lib/x86_64-linux-gnu/libigc.so.2.36.3+0"],
)
filegroup(
    name = "libiga64_so_2",
    srcs = ["lib/x86_64-linux-gnu/libiga64.so.2.36.3+0"],
)
filegroup(
    name = "libigc2",
    srcs = [
        ":libiga64_so_2",
        ":libigc_so_2",
    ],
)""",
    "libigdfcl2": """
filegroup(
    name = "libigdfcl_so_2",
    srcs = ["lib/x86_64-linux-gnu/libigdfcl.so.2.36.3+0"],
)
filegroup(
    name = "libopencl_clang",
    srcs = ["lib/x86_64-linux-gnu/libopencl-clang.so.16"],
)
filegroup(
    name = "libigdfcl2",
    srcs = [
        ":libigdfcl_so_2",
        ":libopencl_clang",
    ],
)""",
    "libze-intel-gpu1": """
filegroup(
    name = "libze_intel_gpu_so",
    srcs = ["lib/x86_64-linux-gnu/libze_intel_gpu.so.1"],
)
filegroup(
    name = "libze_intel_gpu",
    srcs = ["lib/x86_64-linux-gnu/libze_intel_gpu.so.1"],
)""",
    "libze1": """
filegroup(
    name = "libze1",
    srcs = [
        "lib/x86_64-linux-gnu/libze_loader.so.1",
        "lib/x86_64-linux-gnu/libze_tracing_layer.so.1",
        "lib/x86_64-linux-gnu/libze_validation_layer.so.1",
    ],
)""",
    "zlib1g": """
filegroup(
    name = "zlib1g",
    srcs = ["lib/x86_64-linux-gnu/libz.so.1"],
)""",
}

_ONEAPI_PACKAGES = {
    "intel-oneapi-ccl-2022.1": """
filegroup(
    name = "libccl_legacy_so",
    srcs = ["{ONEAPI_CCL_LIB}/libccl_legacy.so"],
)
filegroup(
    name = "ccl_runtime",
    srcs = [
        "{ONEAPI_CCL_LIB}/ccl/cpu/lib/libccl.so",
        "{ONEAPI_CCL_LIB}/ccl/cpu/lib/libccl.so.1",
        "{ONEAPI_CCL_LIB}/ccl/cpu/lib/libccl.so.1.0",
        "{ONEAPI_CCL_LIB}/ccl/cpu/lib/libccl_openmp.so",
        "{ONEAPI_CCL_LIB}/ccl/cpu/lib/libccl_openmp.so.0",
        "{ONEAPI_CCL_LIB}/ccl/cpu/lib/libccl_openmp.so.0.1",
        "{ONEAPI_CCL_LIB}/ccl/kernels/kernels.spv",
        "{ONEAPI_CCL_LIB}/libccl.so",
        "{ONEAPI_CCL_LIB}/libccl.so.1",
        "{ONEAPI_CCL_LIB}/libccl.so.1.0",
        "{ONEAPI_CCL_LIB}/libccl.so.2",
        "{ONEAPI_CCL_LIB}/libccl.so.2.0",
        "{ONEAPI_CCL_LIB}/libccl_legacy_cpu.so",
        "{ONEAPI_CCL_LIB}/libccl_openmp.so",
        "{ONEAPI_CCL_LIB}/libccl_openmp.so.0",
        "{ONEAPI_CCL_LIB}/libccl_openmp.so.0.1",
    ],
)""".format(ONEAPI_CCL_LIB = ONEAPI_CCL_LIB),
    "intel-oneapi-mpi-2021.18": """
filegroup(
    name = "mpi_runtime",
    srcs = [
        "{ONEAPI_MPI_LIB}/libmpi.so",
        "{ONEAPI_MPI_LIB}/libmpi.so.12",
        "{ONEAPI_MPI_LIB}/libmpi.so.12.0",
        "{ONEAPI_MPI_LIB}/libmpi.so.12.0.0",
        "{ONEAPI_MPI_LIB}/libmpi_abi.so",
        "{ONEAPI_MPI_LIB}/libmpi_abi.so.1",
        "{ONEAPI_MPI_LIB}/libmpi_abi.so.1.0",
        "{ONEAPI_MPI_LIB}/libmpi_ilp64.so",
        "{ONEAPI_MPI_LIB}/libmpi_ilp64.so.4",
        "{ONEAPI_MPI_LIB}/libmpi_ilp64.so.4.1",
        "{ONEAPI_MPI_LIB}/libmpi_shm_heap_proxy.so",
        "{ONEAPI_MPI_LIB}/libmpicxx.so",
        "{ONEAPI_MPI_LIB}/libmpicxx.so.12",
        "{ONEAPI_MPI_LIB}/libmpicxx.so.12.0",
        "{ONEAPI_MPI_LIB}/libmpicxx.so.12.0.0",
        "{ONEAPI_MPI_LIB}/libmpifort.so",
        "{ONEAPI_MPI_LIB}/libmpifort.so.12",
        "{ONEAPI_MPI_LIB}/libmpifort.so.12.0",
        "{ONEAPI_MPI_LIB}/libmpifort.so.12.0.0",
        "{ONEAPI_MPI_LIB}/libmpifort_abi.so",
        "{ONEAPI_MPI_LIB}/libmpifort_abi.so.1",
        "{ONEAPI_MPI_LIB}/libmpifort_abi.so.1.0",
        "{ONEAPI_MPI_LIB}/libmpijava.so",
        "{ONEAPI_MPI_LIB}/libmpijava.so.1",
        "{ONEAPI_MPI_LIB}/libmpijava.so.1.0",
        "{ONEAPI_MPI_LIB}/mpi/libmpi_ze_hooks.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/libfabric.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/libfabric.so.1",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libefa-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libmlx-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libpsm3-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libpsmx2-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/librxm-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libshm-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libtcp-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libverbs-1.1-fi.so",
        "{ONEAPI_MPI_LIBFABRIC_LIB}/prov/libverbs-1.12-fi.so",
    ],
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
    "intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1": """
filegroup(
    name = "libsycl_so",
    srcs = ["{ONEAPI_COMPILER_LIB}/libsycl.so.9"],
)
filegroup(
    name = "sycl_runtime",
    srcs = [
        "{ONEAPI_COMPILER_LIB}/libur_adapter_level_zero.so.0",
        "{ONEAPI_COMPILER_LIB}/libur_adapter_level_zero_v2.so.0",
        "{ONEAPI_COMPILER_LIB}/libur_adapter_opencl.so.0",
        "{ONEAPI_COMPILER_LIB}/libur_loader.so.0",
    ],
)""".format(ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB),
    "intel-oneapi-compiler-shared-runtime-2026.1": """
filegroup(
    name = "compiler_runtime",
    srcs = [
        "{ONEAPI_COMPILER_LIB}/libOpenCL.so",
        "{ONEAPI_COMPILER_LIB}/libOpenCL.so.1",
        "{ONEAPI_COMPILER_LIB}/libOpenCL.so.1.0.0",
        "{ONEAPI_COMPILER_LIB}/libimf.so",
        "{ONEAPI_COMPILER_LIB}/libintlc.so.5",
        "{ONEAPI_COMPILER_LIB}/libirc.so",
        "{ONEAPI_COMPILER_LIB}/libirng.so",
        "{ONEAPI_COMPILER_LIB}/libsvml.so",
    ],
)""".format(ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB),
    "intel-oneapi-mkl-core-2026.1": """
filegroup(
    name = "mkl_core_runtime",
    srcs = [
        "{ONEAPI_MKL_LIB}/libmkl_core.so.3",
        "{ONEAPI_MKL_LIB}/libmkl_intel_ilp64.so.3",
        "{ONEAPI_MKL_LIB}/libmkl_sequential.so.3",
    ],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-blas-2026.1": """
filegroup(
    name = "mkl_sycl_blas",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_blas.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-dft-2026.1": """
filegroup(
    name = "mkl_sycl_dft",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_dft.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-lapack-2026.1": """
filegroup(
    name = "mkl_sycl_lapack",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_lapack.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-rng-2026.1": """
filegroup(
    name = "mkl_sycl_rng",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_rng.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
    "intel-oneapi-mkl-sycl-sparse-2026.1": """
filegroup(
    name = "mkl_sycl_sparse",
    srcs = ["{ONEAPI_MKL_LIB}/libmkl_sycl_sparse.so.6"],
)""".format(ONEAPI_MKL_LIB = ONEAPI_MKL_LIB),
}

_ROOT_MODULE_DIRECT_DEPS = ["libpjrt_oneapi"]

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
