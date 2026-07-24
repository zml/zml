load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")


# bazel run //examples/benchmark \
#   --@zml//platforms:cpu=false \
#   --@zml//platforms:musa=true \
#   --override_repository=+musa_packages+libpjrt_musa=/home/kevin/xla-override/libpjrt_musa/
PJRT_MUSA_RELEASE = "musa-rc3.1.1-xla-12b1929d"
# Replace with the sha256sum of pjrt-musa_linux-amd64.tar.gz after publishing this release.
PJRT_MUSA_ARTIFACT_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
PJRT_MUSA_ARTIFACT_URL = "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-musa_linux-amd64.tar.gz".format(
    release = PJRT_MUSA_RELEASE,
)

MUSA_SDK_VERSION = "4.0.1"
MUSA_SDK_RELEASE = "musa-v{MUSA_SDK_VERSION}-musa_sdk_4_0_1_intel_ubuntu-ubuntu-x86_64".format(
    MUSA_SDK_VERSION = MUSA_SDK_VERSION
)
MUSA_TOOLKIT_RELEASE = "musa-toolkit-{MUSA_SDK_VERSION}-musa_sdk_4_0_1_intel_ubuntu-ubuntu-x86_64.tar.zst".format(
    MUSA_SDK_VERSION = MUSA_SDK_VERSION
)
MUSA_SDK_URL = "https://github.com/neudinger/rules-ml-toolchain-redists/releases/download/{MUSA_SDK_RELEASE}/{MUSA_TOOLKIT_RELEASE}".format(
    MUSA_SDK_RELEASE = MUSA_SDK_RELEASE,
    MUSA_TOOLKIT_RELEASE = MUSA_TOOLKIT_RELEASE
)
MUSA_SDK_SHA256 = "f775d7ba3866aaffbbf3f51cb7941fc795f47df5e27dbf1ab629ef14e987c7d6"
MUSA_SDK_STRIP_PREFIX = "musa"

_MUSA_SDK_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "runtime_libs",
    srcs = glob([
        "lib/libmusa.so*",
        "lib/libmusart.so*",
        "lib/libmublas.so*",
    ]),
)
"""

def _musa_impl(mctx):
    http_archive(
        name = "libpjrt_musa",
        build_file = "libpjrt_musa.BUILD.bazel",
        sha256 = PJRT_MUSA_ARTIFACT_SHA256,
        url = PJRT_MUSA_ARTIFACT_URL,
    )
    http_archive(
        name = "musa_sdk",
        build_file_content = _MUSA_SDK_BUILD_FILE_CONTENT,
        sha256 = MUSA_SDK_SHA256,
        strip_prefix = MUSA_SDK_STRIP_PREFIX,
        url = MUSA_SDK_URL,
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = [
            "libpjrt_musa",
        ],
        root_module_direct_dev_deps = [],
    )

musa_packages = module_extension(
    implementation = _musa_impl,
)
