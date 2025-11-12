load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//runtimes/common:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_BUILD_LINUX = "\n".join([
    packages.filegroup(
        name = "libpjrt_cpu",
        srcs = ["libpjrt_cpu.so"],
        visibility = ["@zml//runtimes/cpu:__subpackages__"],
    ),
])

_BUILD_DARWIN = packages.filegroup(
    name = "libpjrt_cpu",
    srcs = ["libpjrt_cpu.dylib"],
    visibility = ["@zml//runtimes/cpu:__subpackages__"],
)

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_LINUX,
        sha256 = "ecc26dc792d2577474348eb48f3989aba8c3bb8d3cbd6df77ccf43357092a700",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v14.0.1/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "4a21db4ecd015fb772614ce4b491551d483ce11321c8784e3d0e07a9a425d5eb",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v14.0.1/pjrt-cpu_darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "e0ab4492468999ae7861a27837427846a708f4346fdae9ad1e84b80e1313566a",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v14.0.1/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
