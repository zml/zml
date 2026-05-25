load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

PJRT_ONEAPI_RELEASE = "manual-2026-05-21T17-13-00Z"
PJRT_ONEAPI_ARTIFACT_SHA256 = "24c518768b5f4994cde22603c7451ce6c53e9d236c3e574e65ca73cd32c6368a"
PJRT_ONEAPI_ARTIFACT_URL = "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-oneapi_linux-amd64.tar.gz".format(
    release = PJRT_ONEAPI_RELEASE,
)

ONEAPI_VERSION = "2025.1"
ONEAPI_BASE_TOOLKIT_VERSION = "2025.1.3.7"

ONEAPI_BASE_LIB = "{}/lib".format(ONEAPI_VERSION)
ONEAPI_COMPILER_LIB = "compiler/{}/lib".format(ONEAPI_VERSION)
ONEAPI_MKL_LIB = "mkl/{}/lib".format(ONEAPI_VERSION)

ONEAPI_RUNTIME_URL = "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/intel-oneapi-base-toolkit-{version}.tar.gz".format(
    version = ONEAPI_BASE_TOOLKIT_VERSION,
)
ONEAPI_RUNTIME_SHA256 = "2213104bd122336551aa144512e7ab99e4a84220e77980b5f346edc14ebd458a"
ONEAPI_RUNTIME_STRIP_PREFIX = "oneapi"

ONEAPI_ZERO_LOADER_URL = "https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/ze_loader_libs.tar.gz"
ONEAPI_ZERO_LOADER_SHA256 = "71cbfd8ac59e1231f013e827ea8efe6cf5da36fad771da2e75e202423bd6b82e"

_ONEAPI_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "base_runtime",
    srcs = [
        "{ONEAPI_BASE_LIB}/libhwloc.so.15",
        "{ONEAPI_BASE_LIB}/libumf.so.0",
        "{ONEAPI_BASE_LIB}/libur_adapter_level_zero.so.0",
        "{ONEAPI_BASE_LIB}/libur_adapter_opencl.so.0",
        "{ONEAPI_BASE_LIB}/libur_loader.so.0",
    ],
)

filegroup(
    name = "compiler_runtime",
    srcs = glob(["{ONEAPI_COMPILER_LIB}/*.spv"]) + [
        "{ONEAPI_COMPILER_LIB}/libOpenCL.so.1",
        "{ONEAPI_COMPILER_LIB}/libimf.so",
        "{ONEAPI_COMPILER_LIB}/libintlc.so.5",
        "{ONEAPI_COMPILER_LIB}/libirc.so",
        "{ONEAPI_COMPILER_LIB}/libirng.so",
        "{ONEAPI_COMPILER_LIB}/libsvml.so",
    ],
)

filegroup(
    name = "libsycl_so",
    srcs = ["{ONEAPI_COMPILER_LIB}/libsycl.so.8"]
)

filegroup(
    name = "mkl_runtime",
    srcs = [
        "{ONEAPI_MKL_LIB}/libmkl_core.so.2",
        "{ONEAPI_MKL_LIB}/libmkl_intel_ilp64.so.2",
        "{ONEAPI_MKL_LIB}/libmkl_sequential.so.2",
        "{ONEAPI_MKL_LIB}/libmkl_sycl_blas.so.5",
    ],
)
""".format(
    ONEAPI_BASE_LIB = ONEAPI_BASE_LIB,
    ONEAPI_COMPILER_LIB = ONEAPI_COMPILER_LIB,
    ONEAPI_MKL_LIB = ONEAPI_MKL_LIB,
)

_ZERO_LOADER_BUILD_FILE_CONTENT = """\
filegroup(
    name = "all",
    srcs = glob(["lib/**"]),
    visibility = ["//visibility:public"],
)
"""

def _oneapi_impl(mctx):
    http_archive(
        name = "libpjrt_oneapi",
        build_file = "libpjrt_oneapi.BUILD.bazel",
        sha256 = PJRT_ONEAPI_ARTIFACT_SHA256,
        url = PJRT_ONEAPI_ARTIFACT_URL,
    )
    http_archive(
        name = "oneapi",
        build_file_content = _ONEAPI_BUILD_FILE_CONTENT,
        sha256 = ONEAPI_RUNTIME_SHA256,
        strip_prefix = ONEAPI_RUNTIME_STRIP_PREFIX,
        url = ONEAPI_RUNTIME_URL,
    )
    http_archive(
        name = "zero_loader",
        build_file_content = _ZERO_LOADER_BUILD_FILE_CONTENT,
        sha256 = ONEAPI_ZERO_LOADER_SHA256,
        url = ONEAPI_ZERO_LOADER_URL,
    )

    return mctx.extension_metadata(reproducible = True)

oneapi_packages = module_extension(
    implementation = _oneapi_impl,
)
