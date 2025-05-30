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
        sha256 = "ca92bccefa168881f98d01354971d6f598381cc4c5f07b161a0908d327610b66",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v9.0.1/pjrt-cpu_linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "b6d05b5cd0382a7bd8943b8df98dc229853e402488127895e47786395afb73a7",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v9.0.1/pjrt-cpu_darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "e1ac13cf80b0975eec1dc0643a6ec08001d6e07a6a0d500a38e1c4477f49a78c",
        url = "https://github.com/zml/pjrt-artifacts/releases/download/v9.0.1/pjrt-cpu_darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
