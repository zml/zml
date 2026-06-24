load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

_BUILD_FILE_CONTENT = """\
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libpjrt_metal",
    srcs = ["libpjrt_c_api_gpu_plugin.dylib"],
    visibility = ["@zml//platforms/metal:__subpackages__"],
)
"""

def _metal_impl(mctx):
    http_archive(
        name = "libpjrt_metal",
        build_file_content = _BUILD_FILE_CONTENT,
        sha256 = "0765d9ee9a40da2f27fc9d8241b0d6d436e7e38ad6d1024840d9c1e0b492c964",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/manual-2026-06-23T00-20-00Z/pjrt-metal_macos-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = ["libpjrt_metal"],
        root_module_direct_dev_deps = [],
    )

metal_packages = module_extension(
    implementation = _metal_impl,
)
