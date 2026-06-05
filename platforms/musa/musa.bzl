load("@llvm//:http_bsdtar_archive.bzl", http_archive = "http_bsdtar_archive")

PJRT_MUSA_RELEASE = "musa-rc3.1.1-xla-12b1929d"
# Replace with the sha256sum of pjrt-musa_linux-amd64.tar.gz after publishing this release.
PJRT_MUSA_ARTIFACT_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
PJRT_MUSA_ARTIFACT_URL = "https://github.com/zml/pjrt-artifacts/releases/download/{release}/pjrt-musa_linux-amd64.tar.gz".format(
    release = PJRT_MUSA_RELEASE,
)

MUSA_SDK_RELEASE = "musa-vrc3.1.1-musa_sdk_rc3_1_1-ubuntu-x86_64"
MUSA_SDK_URL = "https://github.com/neudinger/rules-ml-toolchain-redists/releases/download/{release}/musa-toolkit-rc3.1.1-musa_sdk_rc3_1_1-ubuntu-x86_64.tar.zst".format(
    release = MUSA_SDK_RELEASE,
)
MUSA_SDK_SHA256 = "4f76277bd7e32614d2beab7e48c2efa57d9b25543c01be313b4ed88beaabe51f"
MUSA_SDK_STRIP_PREFIX = "musa"

_MUSA_SDK_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "runtime_libs",
    srcs = glob(["lib/*.so*"]),
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
