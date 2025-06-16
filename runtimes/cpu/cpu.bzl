load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//runtimes/common:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_BUILD_LINUX = "\n".join([
    packages.load_("@zml//bazel:cc_import.bzl", "cc_import"),
    packages.cc_import(
        name = "libpjrt_cpu",
        shared_library = "libpjrt_cpu.so",
        soname = "libpjrt_cpu.so",
        visibility = ["@zml//runtimes/cpu:__subpackages__"],
    ),
])

_BUILD_DARWIN = packages.cc_import(
    name = "libpjrt_cpu",
    shared_library = "libpjrt_cpu.dylib",
    visibility = ["@zml//runtimes/cpu:__subpackages__"],
)

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_LINUX,
        sha256 = "4106ca11ab41bc9ec000d536ae084442139b5639ca329bfb62c7e0742acdc47a",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v10.0.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "7be4d98f0737601fba7b29563917054aac3d09365139e6d3f5f96023a8c71c87",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v10.0.0/pjrt-cpu_darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "442cccd98d7adf4afe0f818ebba265baca6b68dea95b10ef2b4d4229b81d5412",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v10.0.0/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
