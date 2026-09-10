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
        sha256 = "9861bb5314ecc0f28846c49c61c5c2a64a61d6ae2637c0fb7b1e307412a70e8f",
        url = "https://mirror.zml.ai/pjrt-plugins/202609100934.19.1.c22f39a3b260/pjrt-cpu-linux-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_amd64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "d6c3dd715aca5d1453d8282896163612e377b2cc29688686e51733153e712b8b",
        url = "https://mirror.zml.ai/pjrt-plugins/202609100934.19.1.c22f39a3b260/pjrt-cpu-darwin-amd64.tar.gz",
    )

    http_archive(
        name = "libpjrt_cpu_darwin_arm64",
        build_file_content = _BUILD_FILE_DEFAULT_VISIBILITY + _BUILD_DARWIN,
        sha256 = "d88cf830ea5020e86ab38c6ed8e2582559b4187acdfb8dc9587d9f9742f34793",
        url = "https://mirror.zml.ai/pjrt-plugins/202609100934.19.1.c22f39a3b260/pjrt-cpu-darwin-arm64.tar.gz",
    )

    return mctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

cpu_pjrt_plugin = module_extension(
    implementation = _cpu_pjrt_plugin_impl,
)
