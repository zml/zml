load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//bazel:http_deb_archive.bzl", "http_deb_archive")
load("//platforms:packages.bzl", "packages")

PJRT_ONEAPI_RELEASE = "manual-2026-05-21T17-13-00Z"
PJRT_ONEAPI_ARTIFACT_SHA256 = "24c518768b5f4994cde22603c7451ce6c53e9d236c3e574e65ca73cd32c6368a"
PJRT_ONEAPI_ARTIFACT_URL = "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-oneapi_linux-amd64.tar.gz".format(
    release = PJRT_ONEAPI_RELEASE,
)

ONEAPI_VERSION = "2026.0"
ONEAPI_TCM_VERSION = "1.5"
ONEAPI_UMF_VERSION = "1.1"

ONEAPI_ZERO_LOADER_URL = "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/ze_loader_libs.tar.gz"
ONEAPI_ZERO_LOADER_SHA256 = "71cbfd8ac59e1231f013e827ea8efe6cf5da36fad771da2e75e202423bd6b82e"

ONEAPI_COMPILER_LIB = "compiler/{}/lib".format(ONEAPI_VERSION)
ONEAPI_MKL_LIB = "mkl/{}/lib".format(ONEAPI_VERSION)
ONEAPI_TCM_LIB = "tcm/{}/lib".format(ONEAPI_TCM_VERSION)
ONEAPI_UMF_LIB = "umf/{}/lib".format(ONEAPI_UMF_VERSION)

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_ONEAPI_STRIP_PREFIX = "./opt/intel/oneapi"

_UBUNTU_PACKAGES = {
    "zlib1g": packages.filegroup(name = "zlib1g", srcs = ["lib/x86_64-linux-gnu/libz.so.1"]),
}

_ONEAPI_PACKAGES = {
    "intel-oneapi-tcm-1.5": packages.filegroup(
        name = "hwloc",
        srcs = ["{}/libhwloc.so.15".format(ONEAPI_TCM_LIB)],
    ),
    "intel-oneapi-umf-1.1": packages.filegroup(
        name = "umf",
        srcs = ["{}/libumf.so.1".format(ONEAPI_UMF_LIB)],
    ),
    "intel-oneapi-compiler-dpcpp-cpp-runtime-2026.0": "\n".join([
        packages.filegroup(
            name = "libsycl_so",
            srcs = ["{}/libsycl.so.9".format(ONEAPI_COMPILER_LIB)],
        ),
        """filegroup(
            name = "sycl_runtime",
            srcs = glob(["{ONEAPI_COMPILER_LIB}/*.spv"]) + [
                "{ONEAPI_COMPILER_LIB}/libur_adapter_level_zero.so.0",
                "{ONEAPI_COMPILER_LIB}/libur_adapter_opencl.so.0",
                "{ONEAPI_COMPILER_LIB}/libur_loader.so.0",
            ],
        )""".format(ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB),
    ]),
    "intel-oneapi-compiler-shared-runtime-2026.0": """filegroup(
        name = "compiler_runtime",
        srcs = [
            "{ONEAPI_COMPILER_LIB}/libOpenCL.so.1",
            "{ONEAPI_COMPILER_LIB}/libimf.so",
            "{ONEAPI_COMPILER_LIB}/libintlc.so.5",
            "{ONEAPI_COMPILER_LIB}/libirc.so",
            "{ONEAPI_COMPILER_LIB}/libirng.so",
            "{ONEAPI_COMPILER_LIB}/libsvml.so",
        ],
    )""".format(ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB),
    "intel-oneapi-mkl-core-2026.0": packages.filegroup(
        name = "mkl_core_runtime",
        srcs = [
            "{}/libmkl_core.so.3".format(ONEAPI_MKL_LIB),
            "{}/libmkl_intel_ilp64.so.3".format(ONEAPI_MKL_LIB),
            "{}/libmkl_sequential.so.3".format(ONEAPI_MKL_LIB),
        ],
    ),
    "intel-oneapi-mkl-sycl-blas-2026.0": packages.filegroup(
        name = "mkl_sycl_blas",
        srcs = ["{}/libmkl_sycl_blas.so.6".format(ONEAPI_MKL_LIB)],
    ),
}

_ZERO_LOADER_BUILD_FILE_CONTENT = """\
filegroup(
    name = "all",
    srcs = glob(["lib/**"]),
    visibility = ["//visibility:public"],
)
"""

def _oneapi_impl(mctx):
    loaded_packages = packages.read(mctx, [
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

    http_archive(
        name = "zero_loader",
        build_file_content = _ZERO_LOADER_BUILD_FILE_CONTENT,
        sha256 = ONEAPI_ZERO_LOADER_SHA256,
        url = ONEAPI_ZERO_LOADER_URL,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libpjrt_oneapi",
        ],
        root_module_direct_dev_deps = [],
    )

oneapi_packages = module_extension(
    implementation = _oneapi_impl,
)
