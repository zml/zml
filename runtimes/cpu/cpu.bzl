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
        sha256 = "1cda1325095c12bd0019838d28ee92d811ac478d22ed3c08020d5a0cd2d9f34a",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v7.0.0/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "35af82d9e5c70d16ac15f4c18024a2dd5ed2faefc89940eafe3d5350d2cbd9e7",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v7.0.0/pjrt-cpu_darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "da4deaf850d715997614768b2fc0283595ee8181133ab3243d65635e3439de69",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v7.0.0/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
