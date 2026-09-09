load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//platforms:packages.bzl", "packages")

_BUILD_FILE_DEFAULT_VISIBILITY = """\
package(default_visibility = ["//visibility:public"])
"""

_BUILD_LINUX = "\n".join([
    packages.filegroup(
        name = "libpjrt_cpu",
        srcs = ["libpjrt_cpu.so"],
        visibility = ["@zml//platforms/cpu:__subpackages__"],
    ),
])

_BUILD_DARWIN = packages.filegroup(
    name = "libpjrt_cpu",
    srcs = ["libpjrt_cpu.dylib"],
    visibility = ["@zml//platforms/cpu:__subpackages__"],
)

def _cpu_pjrt_plugin_impl(mctx):
    http_archive(
        name = "libpjrt_cpu_linux_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_LINUX,
        sha256 = "551cd6e2e4999504c1e9b7639680756e79ae6d2bb15c597a955dbae67c2ef65d",
        url = "https://mirror.zml.ai/pjrt-plugins/20260909.69bab05e48b6.14.1/pjrt-cpu-linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "1650d56a9a734ce7bf9269d29ee426da8b45550a253cf9f565daf1c11b51cfe5",
        url = "https://mirror.zml.ai/pjrt-plugins/20260909.69bab05e48b6.14.1/pjrt-cpu-darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "8b7bb1280559bae111a8b1a91b88f4edee5870267b20c745df94ee62a0ebf587",
        url = "https://mirror.zml.ai/pjrt-plugins/20260909.69bab05e48b6.14.1/pjrt-cpu-darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
