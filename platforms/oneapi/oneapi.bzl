load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

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

def _base_lib(file):
    return "{}/{}".format(ONEAPI_BASE_LIB, file)

def _compiler_lib(file):
    return "{}/{}".format(ONEAPI_COMPILER_LIB, file)

def _mkl_lib(file):
    return "{}/{}".format(ONEAPI_MKL_LIB, file)

_ONEAPI_FILEGROUPS = [
    packages.filegroup(
        name = "base_runtime",
        srcs = [
            _base_lib("libhwloc.so.15"),
            _base_lib("libumf.so.0"),
            _base_lib("libur_adapter_level_zero.so.0"),
            _base_lib("libur_adapter_opencl.so.0"),
            _base_lib("libur_loader.so.0"),
        ],
    ),
    packages.filegroup_glob(
        name = "compiler_runtime",
        srcs_glob = [
            _compiler_lib("*.spv"),
        ],
        srcs = [
            _compiler_lib("libOpenCL.so.1"),
            _compiler_lib("libimf.so"),
            _compiler_lib("libintlc.so.5"),
            _compiler_lib("libirc.so"),
            _compiler_lib("libirng.so"),
            _compiler_lib("libsvml.so"),
        ],
    ),
    packages.filegroup(
        name = "libsycl_so",
        srcs = [
            _compiler_lib("libsycl.so.8"),
        ],
    ),
    packages.filegroup(
        name = "mkl_runtime",
        srcs = [
            _mkl_lib("libmkl_core.so.2"),
            _mkl_lib("libmkl_intel_ilp64.so.2"),
            _mkl_lib("libmkl_sequential.so.2"),
            _mkl_lib("libmkl_sycl_blas.so.5"),
        ],
    ),
]

_ONEAPI_BUILD_FILE_CONTENT = "\n".join([
    _BUILD_FILE_DEFAULT_VISIBILITY,
] + _ONEAPI_FILEGROUPS)

_ZERO_LOADER_FILEGROUPS = [
    packages.filegroup_glob(
        name = "all",
        srcs_glob = ["lib/**"],
    ),
]

_ZERO_LOADER_BUILD_FILE_CONTENT = "\n".join([
    _BUILD_FILE_DEFAULT_VISIBILITY,
] + _ZERO_LOADER_FILEGROUPS)

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

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libpjrt_oneapi",
            "oneapi",
            "zero_loader",
        ],
        root_module_direct_dev_deps = [],
    )

oneapi_packages = module_extension(
    implementation = _oneapi_impl,
)
