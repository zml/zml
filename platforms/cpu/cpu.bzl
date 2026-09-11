load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_BUILD_LINUX = "\n".join([
    packages.filegroup(
        name = "libzml_cpu",
        srcs = ["libzml_cpu.so"],
        visibility = ["@zml//platforms/cpu:__subpackages__"],
    ),
])

_BUILD_DARWIN = packages.filegroup(
    name = "libzml_cpu",
    srcs = ["libzml_cpu.dylib"],
    visibility = ["@zml//platforms/cpu:__subpackages__"],
)

def _cpu_plugin_impl(mctx):
    http_archive(
        name = "libzml_cpu_linux_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_LINUX,
        sha256 = "f1670d21b92102b8c7c7f9d0f00d6a7a1a5d90a83d7e28ba5fbeb1f9ce4c170c",
        url = "https://mirror.zml.ai/plugins/202609101243.20.1.7ca6884ea2cb/zml-cpu-linux-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "d8c7dea830773eb37bb6161f1ed62f69f30bd20a2d669083c5d6a7d60c19844b",
        url = "https://mirror.zml.ai/plugins/202609101243.20.1.7ca6884ea2cb/zml-cpu-darwin-amd64.tar.zst",
    )

    http_archive(
        name = "libzml_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "b8c5a1d15fe3b6ce8520852901ccec3cb12d092b052fdd527449d822833becba",
        url = "https://mirror.zml.ai/plugins/202609101243.20.1.7ca6884ea2cb/zml-cpu-darwin-arm64.tar.zst",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_plugin = module_extension(
    implementation = _cpu_plugin_impl,
)
